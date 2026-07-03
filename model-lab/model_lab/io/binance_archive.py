"""Download + store Binance spot ``aggTrades`` daily archives (data.binance.vision).

This is the historical-data acquisition primitive behind ``model_lab.hist`` and
``model_lab.hist_integrity``. It is the only part of the lab that touches the
network; everything else reads local files.

On-disk shape (one zstd-parquet file per symbol-day, the atomic unit for
incremental download + per-day integrity)::

    <hist_dir>/aggtrades-{SYMBOL}-{YYYY-MM-DD}.parquet
    <hist_dir>/manifest.json

Each parquet holds the seven columns of :data:`AGGTRADES_COLS` (ids int64, price
and quantity float64 — research not accounting, per the lab's boundary rule —
``transact_time`` normalized to **microseconds**, ``is_buyer_maker`` bool).

External contract (verified against binance/binance-public-data):

- URL: ``{BASE_URL}/{SYMBOL}/{SYMBOL}-aggTrades-{YYYY-MM-DD}.zip``
- Each ``.zip`` has a sibling ``.zip.CHECKSUM`` in ``sha256sum`` format
  (``"<sha256hex>  <filename>"``) — verified when present.
- The CSV inside is headerless on older archives, header-carrying on some 2025+
  ones, and has 7 or 8 columns (a trailing ``is_best_match`` we drop) — parsed
  defensively by sniffing the first line.
- ``transact_time`` is microseconds for spot from 2025-01-01, milliseconds
  before — normalized to microseconds here.
- A day's archive publishes ~next day, so the most recent day(s) 404 and are
  reported ``missing`` (self-healing: filled on a later run).

All downloaders take a ``fetch`` callable (default :func:`http_fetch`) so tests
inject a fake returning synthetic bytes — no network in the test suite.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

BASE_URL = "https://data.binance.vision/data/spot/daily/aggTrades"
SYMBOLS: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")
SCHEMA_VERSION = 1
TIME_UNIT = "microseconds"
MANIFEST_NAME = "manifest.json"

# The seven columns we keep, in Binance's on-disk order. A trailing
# ``is_best_match`` column (present on some archives) is intentionally dropped.
AGGTRADES_COLS: list[str] = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time",
    "is_buyer_maker",
]

_DTYPES: dict[str, str] = {
    "agg_trade_id": "int64",
    "price": "float64",
    "quantity": "float64",
    "first_trade_id": "int64",
    "last_trade_id": "int64",
    "transact_time": "int64",
    "is_buyer_maker": "string",
}

_DAY_RE = re.compile(r"^aggtrades-(?P<sym>.+)-(?P<day>\d{4}-\d{2}-\d{2})\.parquet$")


class ArchiveNotFound(Exception):
    """The requested daily archive does not exist on the server (HTTP 404)."""


# --------------------------------------------------------------------------- #
# Fetch seam
# --------------------------------------------------------------------------- #

FetchFn = Callable[[str], bytes]


def http_fetch(url: str, *, timeout: float = 60.0, retries: int = 3, backoff: float = 1.5) -> bytes:
    """Fetch ``url`` and return the body bytes.

    Raises :class:`ArchiveNotFound` on HTTP 404 (missing day — never retried);
    retries transient transport errors with linear backoff, then re-raises.
    """
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "model-lab/0.1 (aggTrades archive fetch)"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:  # subclass of URLError — catch first
            if e.code == 404:
                raise ArchiveNotFound(url) from e
            last_err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last_err = e
        if attempt < retries - 1:
            time.sleep(backoff * (attempt + 1))
    raise last_err if last_err is not None else RuntimeError(f"fetch failed: {url}")


# --------------------------------------------------------------------------- #
# URL / path helpers
# --------------------------------------------------------------------------- #


def _day_str(day: date) -> str:
    return day.isoformat()  # YYYY-MM-DD


def day_url(symbol: str, day: date) -> str:
    """URL of the daily aggTrades zip for ``symbol`` on ``day``."""
    return f"{BASE_URL}/{symbol}/{symbol}-aggTrades-{_day_str(day)}.zip"


def checksum_url(symbol: str, day: date) -> str:
    """URL of the ``.CHECKSUM`` sidecar for the daily aggTrades zip."""
    return day_url(symbol, day) + ".CHECKSUM"


def day_parquet_path(hist_dir: Path, symbol: str, day: date) -> Path:
    """On-disk parquet path for ``symbol``'s ``day``."""
    return hist_dir / f"aggtrades-{symbol}-{_day_str(day)}.parquet"


# --------------------------------------------------------------------------- #
# Checksum
# --------------------------------------------------------------------------- #


def parse_checksum(checksum_text: str) -> str | None:
    """Extract the SHA-256 hex from a ``sha256sum``-format ``.CHECKSUM`` body."""
    text = (checksum_text or "").strip()
    if not text:
        return None
    first = text.split()[0].lower()
    if len(first) == 64 and all(c in "0123456789abcdef" for c in first):
        return first
    return None


def verify_sha256(zip_bytes: bytes, checksum_text: str) -> bool:
    """Whether ``zip_bytes`` matches the SHA-256 in ``checksum_text``."""
    expected = parse_checksum(checksum_text)
    if expected is None:
        return False
    return hashlib.sha256(zip_bytes).hexdigest() == expected


# --------------------------------------------------------------------------- #
# ZIP → CSV → typed DataFrame
# --------------------------------------------------------------------------- #


def _extract_csv(zip_bytes: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        if not names:
            raise ValueError("empty zip archive")
        csv_name = next((n for n in names if n.lower().endswith(".csv")), names[0])
        return zf.read(csv_name)


def _sniff(csv_bytes: bytes) -> tuple[bool, int]:
    """Return ``(has_header, n_columns)`` from the first line of the CSV."""
    nl = csv_bytes.find(b"\n")
    first = csv_bytes[:nl] if nl != -1 else csv_bytes
    ncols = first.count(b",") + 1
    field0 = first.split(b",", 1)[0].strip()
    try:
        int(field0)
        has_header = False
    except ValueError:
        has_header = True
    return has_header, ncols


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "agg_trade_id": pd.Series([], dtype="int64"),
            "price": pd.Series([], dtype="float64"),
            "quantity": pd.Series([], dtype="float64"),
            "first_trade_id": pd.Series([], dtype="int64"),
            "last_trade_id": pd.Series([], dtype="int64"),
            "transact_time": pd.Series([], dtype="int64"),
            "is_buyer_maker": pd.Series([], dtype="bool"),
        }
    )


def _normalize_time_us(s: pd.Series) -> pd.Series:
    """Normalize the raw ``transact_time`` to microseconds.

    Binance spot aggTrades are microseconds from 2025-01-01, milliseconds before.
    Detect by magnitude of the first value (all rows in a day share the unit) and
    upconvert ms → µs so the store is a single unit.
    """
    s = s.astype("int64")
    if len(s) == 0:
        return s
    first = int(s.iloc[0])
    if first >= 1_000_000_000_000_000:  # ~1e15 → already microseconds
        return s
    if first >= 100_000_000_000:  # ~1e11 → milliseconds
        return s * 1000
    return s  # degenerate/unknown — leave as-is (integrity will flag if bad)


def parse_aggtrades_csv(csv_bytes: bytes) -> pd.DataFrame:
    """Parse a raw aggTrades CSV into the typed :data:`AGGTRADES_COLS` frame.

    Handles headered/headerless and 7/8-column variants, maps
    ``is_buyer_maker`` to bool, and normalizes ``transact_time`` to microseconds.
    """
    if not csv_bytes.strip():
        return _empty_frame()

    has_header, ncols = _sniff(csv_bytes)
    if ncols < len(AGGTRADES_COLS):
        raise ValueError(f"aggTrades CSV has only {ncols} columns (expected >= {len(AGGTRADES_COLS)})")
    # Name every column so `names` length matches the file; keep the first seven.
    names = list(AGGTRADES_COLS)
    names += [f"extra_{i}" for i in range(ncols - len(AGGTRADES_COLS))]

    df = pd.read_csv(
        io.BytesIO(csv_bytes),
        header=None,
        names=names,
        usecols=AGGTRADES_COLS,
        skiprows=1 if has_header else 0,
        dtype=_DTYPES,
    )
    if df.empty:
        return _empty_frame()

    df = df[AGGTRADES_COLS]  # enforce column order
    df["transact_time"] = _normalize_time_us(df["transact_time"])
    maker = df["is_buyer_maker"].astype("string").str.strip().str.lower()
    df["is_buyer_maker"] = maker.map({"true": True, "false": False}).fillna(False).astype(bool)
    return df


# --------------------------------------------------------------------------- #
# Download one day (skip-if-present, verify, atomic write)
# --------------------------------------------------------------------------- #


@dataclass
class DayResult:
    """Outcome of attempting one symbol-day."""

    symbol: str
    day: date
    status: str  # downloaded | skipped | missing | checksum_failed | error
    rows: int = 0
    bytes: int = 0
    sha256: str | None = None
    checksum: str = "absent"  # verified | mismatch | absent | skipped
    error: str | None = None


def _parquet_rows(path: Path) -> int:
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:  # noqa: BLE001 — a corrupt file simply counts as 0 here
        return 0


def download_day(
    symbol: str,
    day: date,
    hist_dir: Path,
    *,
    fetch: FetchFn = http_fetch,
    verify_checksum: bool = True,
) -> DayResult:
    """Download, verify, and store one symbol-day. Idempotent: an existing final
    parquet is left untouched and reported ``skipped`` (incremental/resumable)."""
    path = day_parquet_path(hist_dir, symbol, day)
    if path.exists():
        return DayResult(symbol, day, "skipped", rows=_parquet_rows(path), bytes=path.stat().st_size)

    try:
        zip_bytes = fetch(day_url(symbol, day))
    except ArchiveNotFound:
        return DayResult(symbol, day, "missing")
    except Exception as e:  # noqa: BLE001 — transport failure for this day only
        return DayResult(symbol, day, "error", error=f"fetch: {e}")

    sha = hashlib.sha256(zip_bytes).hexdigest()
    checksum_status = "skipped"
    if verify_checksum:
        try:
            checksum_text: str | None = fetch(checksum_url(symbol, day)).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 — 404 or transport: treat checksum as absent
            checksum_text = None
        if checksum_text is None:
            checksum_status = "absent"
        elif verify_sha256(zip_bytes, checksum_text):
            checksum_status = "verified"
        else:
            return DayResult(symbol, day, "checksum_failed", sha256=sha, checksum="mismatch")

    try:
        df = parse_aggtrades_csv(_extract_csv(zip_bytes))
    except Exception as e:  # noqa: BLE001 — malformed archive for this day only
        return DayResult(symbol, day, "error", sha256=sha, error=f"parse: {e}")

    hist_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    try:
        df.to_parquet(tmp, engine="pyarrow", compression="zstd", index=False)
        os.replace(tmp, path)  # atomic — no partial finals on interruption
    except Exception as e:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return DayResult(symbol, day, "error", sha256=sha, error=f"write: {e}")

    return DayResult(
        symbol,
        day,
        "downloaded",
        rows=len(df),
        bytes=path.stat().st_size,
        sha256=sha,
        checksum=checksum_status,
    )


# --------------------------------------------------------------------------- #
# Readers (integrity + future research)
# --------------------------------------------------------------------------- #


def aggtrades_day_paths(hist_dir: Path, symbol: str) -> list[Path]:
    """All day-parquets for ``symbol``, chronological order (name == date sort)."""
    if not hist_dir.is_dir():
        return []
    return sorted(hist_dir.glob(f"aggtrades-{symbol}-*.parquet"))


def day_of_path(path: Path) -> date | None:
    """The trade date encoded in a day-parquet filename, or ``None``."""
    m = _DAY_RE.match(path.name)
    if not m:
        return None
    try:
        return date.fromisoformat(m.group("day"))
    except ValueError:
        return None


def read_aggtrades_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    """Read only ``columns`` from a day-parquet (cheap for integrity checks)."""
    return pq.read_table(path, columns=columns).to_pandas()


def symbols_present(hist_dir: Path) -> list[str]:
    """Symbols that have at least one day-parquet in ``hist_dir``."""
    if not hist_dir.is_dir():
        return []
    syms: set[str] = set()
    for p in hist_dir.glob("aggtrades-*.parquet"):
        m = _DAY_RE.match(p.name)
        if m:
            syms.add(m.group("sym"))
    return sorted(syms)


def has_aggtrades(hist_dir: Path) -> bool:
    """Whether any aggTrades day-parquet is present."""
    return hist_dir.is_dir() and any(hist_dir.glob("aggtrades-*.parquet"))


# --------------------------------------------------------------------------- #
# Disk usage
# --------------------------------------------------------------------------- #


def dir_disk_usage(hist_dir: Path) -> int:
    """Total bytes of the aggTrades day-parquet files under ``hist_dir``.

    Only the data files are counted (not the tiny, self-referential
    ``manifest.json``), so the figure is stable and reproducible across runs.
    """
    if not hist_dir.is_dir():
        return 0
    return sum(p.stat().st_size for p in hist_dir.glob("aggtrades-*.parquet"))


def human_bytes(n: int) -> str:
    """A compact human-readable byte size (e.g. ``"858.3 MiB"``)."""
    val = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if val < 1024.0 or unit == "TiB":
            return f"{int(val)} B" if unit == "B" else f"{val:.1f} {unit}"
        val /= 1024.0
    return f"{val:.1f} TiB"  # unreachable, keeps type-checkers happy


# --------------------------------------------------------------------------- #
# Manifest (rebuilt from disk = ground truth, so it never drifts)
# --------------------------------------------------------------------------- #


def manifest_path(hist_dir: Path) -> Path:
    """Path to the coverage manifest."""
    return hist_dir / MANIFEST_NAME


def build_manifest(
    hist_dir: Path,
    symbols: list[str],
    missing: dict[str, list[date]] | None = None,
) -> dict[str, Any]:
    """Scan ``hist_dir`` and assemble the coverage manifest for ``symbols``."""
    missing = missing or {}
    symbols_meta: dict[str, Any] = {}
    for sym in symbols:
        per_day: dict[str, Any] = {}
        rows_total = 0
        bytes_total = 0
        days_present: list[str] = []
        for p in aggtrades_day_paths(hist_dir, sym):
            d = day_of_path(p)
            if d is None:
                continue
            ds = d.isoformat()
            rows = _parquet_rows(p)
            b = p.stat().st_size
            per_day[ds] = {"rows": rows, "bytes": b}
            rows_total += rows
            bytes_total += b
            days_present.append(ds)
        days_present.sort()
        symbols_meta[sym] = {
            "day_count": len(days_present),
            "first_day": days_present[0] if days_present else None,
            "last_day": days_present[-1] if days_present else None,
            "days_present": days_present,
            "missing_in_range": sorted(d.isoformat() for d in missing.get(sym, [])),
            "rows_total": rows_total,
            "bytes_total": bytes_total,
            "per_day": per_day,
        }
    disk = dir_disk_usage(hist_dir)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_utc": now,
        "time_unit": TIME_UNIT,
        "base_url": BASE_URL,
        "symbols": symbols_meta,
        "disk_usage_bytes": disk,
        "disk_usage_human": human_bytes(disk),
    }


def write_manifest(hist_dir: Path, manifest: dict[str, Any]) -> None:
    """Write ``manifest.json`` (pretty-printed, the lab's metrics.json idiom)."""
    hist_dir.mkdir(parents=True, exist_ok=True)
    manifest_path(hist_dir).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_manifest(hist_dir: Path) -> dict[str, Any] | None:
    """Read the coverage manifest, or ``None`` if absent/unreadable."""
    p = manifest_path(hist_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# Date helpers
# --------------------------------------------------------------------------- #


def daterange(start: date, end: date) -> list[date]:
    """Inclusive list of dates from ``start`` to ``end`` (empty if end < start)."""
    if end < start:
        return []
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def default_end() -> date:
    """Yesterday (UTC) — today's daily archive is not published yet."""
    return datetime.now(timezone.utc).date() - timedelta(days=1)


def default_range(days: int = 90, end: date | None = None) -> tuple[date, date]:
    """The default ``days``-long window ending at ``end`` (yesterday UTC)."""
    e = end or default_end()
    return e - timedelta(days=days - 1), e
