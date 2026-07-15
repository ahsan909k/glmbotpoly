"""Telonex vendor-data client + on-disk store (Polymarket + Binance parquet).

This is the network primitive behind ``model_lab.telonex_ingest`` /
``model_lab.telonex_validate`` — the second (and only other) part of the lab that
touches the network. Everything is driven through a ``fetch`` seam so the whole
path tests offline with synthetic responses (mirrors ``io.binance_archive``).

External contract (read from telonex.io/docs, verified with live probes — see
``telonex_notes.md``):

- Base ``https://api.telonex.io/v1``; auth ``Authorization: Bearer <key>`` on the
  **download** endpoints only. **availability** + **datasets** are public.
- ``GET /downloads/{exchange}/{channel}/{date}?<ident>`` → **HTTP 302** to a
  pre-signed S3 URL (15-min expiry). We fetch it in **two steps** — GET without
  following (to read ``Location`` + ``X-Downloads-Remaining``), then stream the S3
  URL with **no** Authorization header (S3 rejects a doubled auth mechanism).
  Errors: 400 / 403 (download limit) / 404 (no data) / 422.
- ``GET /availability/{exchange}?<ident>`` (public) → per-channel ``from_date`` /
  ``to_date`` (``to_date`` exclusive).
- ``GET /datasets/polymarket/markets`` (public) → the whole markets catalog parquet.
- Crypto up/down windows are slugged ``{asset}-updown-{dur}-{open_epoch_secs}`` with
  outcome ``Up``/``Down``; Binance ``slug = asset_id = <lowercase symbol>``.
- Files are Apache Parquet; timestamps are ``timestamp_us`` / ``local_timestamp_us``
  (int64 **microseconds**), same across both exchanges.

Everything streams to disk (16 GB box, prior OOMs) with atomic ``tmp → os.replace``.
The API key is read from the env var ``telonexdata`` (falling back to ``.env``) and
is **never** logged, returned, or written to any manifest.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT
from .binance_archive import human_bytes  # shared byte formatter

API_BASE = "https://api.telonex.io/v1"
POLYMARKET = "polymarket"
BINANCE = "binance"
ENV_KEY = "telonexdata"
MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1
_UA = "model-lab/0.1 (telonex trial pull)"
_TIMEOUT = 120.0
_REDIRECT_CODES = {301, 302, 303, 307, 308}
_DUR_SECS: dict[str, int] = {"5m": 300, "15m": 900, "1h": 3600}

POLYMARKET_CHANNELS = (
    "trades", "quotes", "book_snapshot_5", "book_snapshot_25",
    "book_snapshot_full", "onchain_fills", "crypto_prices",
)
BINANCE_CHANNELS = ("trades", "quotes", "book_snapshot_5", "book_snapshot_25")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class TelonexError(Exception):
    """Any Telonex client failure."""


class TelonexNotFound(TelonexError):
    """No data for the requested asset/day/channel (HTTP 404)."""


class TelonexAuthError(TelonexError):
    """Authentication failed (HTTP 401) — bad/absent key."""


class TelonexDownloadLimit(TelonexError):
    """Download limit exceeded (HTTP 403). ``remaining`` from ``X-Downloads-Remaining``."""

    def __init__(self, url: str, remaining: int | None = None):
        super().__init__(
            f"download limit exceeded (HTTP 403) for {url}; X-Downloads-Remaining="
            f"{remaining}. A free-tier key allows only 5 downloads total — a paid "
            f"(Single-Exchange/Pro) plan is required for the trial."
        )
        self.remaining = remaining


class TelonexPlanRestricted(TelonexError):
    """The requested exchange is not on the account's plan (HTTP 403 ``exchange_restricted``).

    Telonex's Single-Exchange plan covers one exchange (e.g. Polymarket); Binance needs
    Pro. This is NOT a per-download quota — it is permanent for that exchange on this key,
    so the ingest skips the exchange (rather than aborting the whole trial)."""

    def __init__(self, url: str, detail: str = ""):
        super().__init__(f"exchange not on plan (HTTP 403 exchange_restricted) for {url}: {detail[:200]}")
        self.detail = detail


class TelonexHTTPError(TelonexError):
    """A raw non-2xx HTTP status from the API (before it is classified)."""

    def __init__(self, status: int, headers: dict[str, str], url: str, body: bytes = b""):
        super().__init__(f"HTTP {status} for {url}")
        self.status = status
        self.headers = headers
        self.url = url
        self.body = body


# --------------------------------------------------------------------------- #
# API key (never logged / returned to a manifest)
# --------------------------------------------------------------------------- #


def _parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        out[key] = val
    return out


def load_api_key(*, env_key: str = ENV_KEY, env_file: str | Path | None = None) -> str:
    """Return the Telonex API key from ``$telonexdata`` or the repo ``.env``.

    Never printed, logged, or written anywhere. Raises :class:`TelonexError` with an
    actionable message (and no key material) if it can't be found.
    """
    val = os.environ.get(env_key)
    if val and val.strip():
        return val.strip()
    path = Path(env_file) if env_file else (REPO_ROOT / ".env")
    parsed = _parse_env_file(path)
    val = parsed.get(env_key)
    if val:
        return val
    raise TelonexError(
        f"Telonex API key not found — set {env_key} in the environment or in {path}. "
        "(The key is never printed or committed.)"
    )


# --------------------------------------------------------------------------- #
# Fetch seam
# --------------------------------------------------------------------------- #


class TelonexResponse:
    """A minimal HTTP response: ``status``, ``headers``, and a streamable body."""

    def __init__(self, status: int, headers: dict[str, str], reader: Any):
        self.status = status
        self.headers = headers
        self._reader = reader

    def read(self) -> bytes:
        return self._reader.read()

    def stream_to(self, fh: Any, chunk: int = 1 << 20) -> int:
        """Copy the body to an open file object in ``chunk``-byte pieces; return bytes."""
        total = 0
        while True:
            block = self._reader.read(chunk)
            if not block:
                break
            fh.write(block)
            total += len(block)
        return total

    def close(self) -> None:
        try:
            self._reader.close()
        except Exception:  # noqa: BLE001 — best-effort close
            pass


# (url, headers, follow_redirects) -> TelonexResponse; raises TelonexHTTPError on >=400
Fetch = Callable[[str, dict | None, bool], TelonexResponse]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, D102
        return None  # don't follow — urlopen then raises HTTPError(code)


_no_redirect_opener: urllib.request.OpenerDirector | None = None


def _opener_no_redirect() -> urllib.request.OpenerDirector:
    global _no_redirect_opener
    if _no_redirect_opener is None:
        _no_redirect_opener = urllib.request.build_opener(_NoRedirect)
    return _no_redirect_opener


def _headers_dict(headers: Any) -> dict[str, str]:
    if headers is None:
        return {}
    try:
        return {str(k): str(v) for k, v in headers.items()}
    except AttributeError:
        return {}


def urllib_fetch(url: str, headers: dict | None = None, follow_redirects: bool = True) -> TelonexResponse:
    """Real fetch over stdlib ``urllib``. Raises :class:`TelonexHTTPError` on >=400.

    With ``follow_redirects=False`` a 3xx is returned as a response (so the caller can
    read the ``Location`` + ``X-Downloads-Remaining`` headers) instead of being followed.
    """
    req = urllib.request.Request(url, headers={"User-Agent": _UA, **(headers or {})})
    try:
        if follow_redirects:
            resp = urllib.request.urlopen(req, timeout=_TIMEOUT)
        else:
            resp = _opener_no_redirect().open(req, timeout=_TIMEOUT)
        return TelonexResponse(int(getattr(resp, "status", 200) or 200), _headers_dict(resp.headers), resp)
    except urllib.error.HTTPError as e:
        hdrs = _headers_dict(e.headers)
        if not follow_redirects and e.code in _REDIRECT_CODES:
            try:
                e.read()
            except Exception:  # noqa: BLE001
                pass
            return TelonexResponse(int(e.code), hdrs, io.BytesIO(b""))
        body = b""
        try:
            body = e.read()[:4096]
        except Exception:  # noqa: BLE001
            pass
        raise TelonexHTTPError(int(e.code), hdrs, url, body) from e


def _hget(headers: dict[str, str], name: str) -> str | None:
    """Case-insensitive header lookup (works whether keys are original- or lower-cased)."""
    name = name.lower()
    for k, v in headers.items():
        if k.lower() == name:
            return v
    return None


def _remaining(headers: dict[str, str]) -> int | None:
    val = _hget(headers, "X-Downloads-Remaining")
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _classify(e: TelonexHTTPError) -> TelonexError:
    if e.status == 403:
        return TelonexDownloadLimit(e.url, _remaining(e.headers))
    if e.status == 401:
        return TelonexAuthError(f"unauthorized (HTTP 401) for {e.url} — check {ENV_KEY}")
    return TelonexError(f"HTTP {e.status} for {e.url}: {e.body[:200]!r}")


# --------------------------------------------------------------------------- #
# JSON GET (availability) + streaming download (files, catalog)
# --------------------------------------------------------------------------- #


def get_json(
    url: str, *, headers: dict | None = None, fetch: Fetch = urllib_fetch,
    retries: int = 3, backoff: float = 1.5,
) -> Any:
    """GET ``url`` and parse the JSON body. 404 → :class:`TelonexNotFound`."""
    last: Exception | None = None
    for attempt in range(retries):
        resp = None
        try:
            resp = fetch(url, headers, True)
            return json.loads(resp.read().decode("utf-8"))
        except TelonexHTTPError as e:
            if e.status == 404:
                raise TelonexNotFound(url) from e
            if e.status in (400, 401, 403, 422):
                raise _classify(e) from e
            last = e  # 5xx / 429 → retry
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError) as e:
            last = e
        finally:
            if resp is not None:
                resp.close()
        if attempt < retries - 1:
            time.sleep(backoff * (attempt + 1))
    raise last if last is not None else RuntimeError(f"telonex GET failed: {url}")


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def download_url(
    url: str, dest: Path, *, headers: dict | None = None, fetch: Fetch = urllib_fetch,
    retries: int = 3, backoff: float = 1.5,
) -> dict[str, Any]:
    """Stream ``url`` to ``dest`` (atomic ``tmp → os.replace``), two-step through the 302.

    Returns ``{"status": "downloaded"|"no_data", "bytes": int, "remaining": int|None}``.
    404 → ``no_data`` (a legitimately empty asset/day). 403 → :class:`TelonexDownloadLimit`.
    """
    dest = Path(dest)
    tmp = dest.with_name(dest.name + ".tmp")
    last: Exception | None = None
    for attempt in range(retries):
        r1 = r2 = None
        try:
            r1 = fetch(url, headers, False)  # do NOT follow — read Location + remaining
            remaining = _remaining(r1.headers)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if r1.status in _REDIRECT_CODES:
                loc = _hget(r1.headers, "Location")
                if not loc:
                    raise TelonexError(f"redirect without Location: {url}")
                r2 = fetch(loc, None, True)  # S3 pre-signed — NO Authorization header
                with open(tmp, "wb") as fh:
                    r2.stream_to(fh)
            elif r1.status == 200:
                with open(tmp, "wb") as fh:
                    r1.stream_to(fh)
            else:
                raise TelonexError(f"unexpected status {r1.status} for {url}")
            os.replace(tmp, dest)
            return {"status": "downloaded", "bytes": dest.stat().st_size, "remaining": remaining}
        except TelonexHTTPError as e:
            _safe_unlink(tmp)
            if e.status == 404:
                return {"status": "no_data", "bytes": 0, "remaining": _remaining(e.headers)}
            if e.status == 403:
                detail = e.body.decode("utf-8", "replace") if e.body else ""
                if "exchange_restricted" in detail or "Upgrade to Pro" in detail:
                    raise TelonexPlanRestricted(url, detail) from e
                # An unexplained 403 may be a transient throttle — retry with backoff, and
                # only treat it as a hard download-limit if it persists across all attempts.
                last = e
            elif e.status in (400, 401, 422):
                raise _classify(e) from e
            else:
                last = e  # 5xx / 429 → retry
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            _safe_unlink(tmp)
            last = e
        finally:
            if r1 is not None:
                r1.close()
            if r2 is not None:
                r2.close()
        if attempt < retries - 1:
            time.sleep(backoff * (attempt + 1))
    _safe_unlink(tmp)
    if isinstance(last, TelonexHTTPError) and last.status == 403:
        raise TelonexDownloadLimit(last.url, _remaining(last.headers))
    raise last if last is not None else RuntimeError(f"telonex download failed: {url}")


# --------------------------------------------------------------------------- #
# High-level endpoints
# --------------------------------------------------------------------------- #


def _ident_query(ident: dict[str, Any]) -> str:
    return urllib.parse.urlencode({k: v for k, v in ident.items() if v is not None})


def availability(exchange: str, *, fetch: Fetch = urllib_fetch, **ident: Any) -> dict[str, Any]:
    """Public availability for one asset → ``{channels: {name: {from_date, to_date}}, …}``.

    Raises :class:`TelonexNotFound` when the asset has no data (unopened window)."""
    url = f"{API_BASE}/availability/{exchange}?{_ident_query(ident)}"
    return get_json(url, fetch=fetch)


def build_download_url(exchange: str, channel: str, date_str: str, ident: dict[str, Any]) -> str:
    q = _ident_query(ident)
    base = f"{API_BASE}/downloads/{exchange}/{channel}/{date_str}"
    return f"{base}?{q}" if q else base


def download_asset_day(
    exchange: str, channel: str, date_str: str, dest: Path, *, api_key: str,
    fetch: Fetch = urllib_fetch, **ident: Any,
) -> dict[str, Any]:
    """Download one (asset, channel, day) parquet to ``dest``. Skip-if-exists (resumable)."""
    dest = Path(dest)
    if dest.exists():
        return {"status": "skipped", "bytes": dest.stat().st_size, "remaining": None}
    url = build_download_url(exchange, channel, date_str, ident)
    headers = {"Authorization": f"Bearer {api_key}"}
    return download_url(url, dest, headers=headers, fetch=fetch)


def download_markets_dataset(exchange: str, dest: Path, *, fetch: Fetch = urllib_fetch) -> dict[str, Any]:
    """Stream the public markets catalog parquet for ``exchange`` to ``dest``."""
    return download_url(f"{API_BASE}/datasets/{exchange}/markets", Path(dest), fetch=fetch)


def catalog_path(telonex_dir: Path) -> Path:
    """On-disk path of the Polymarket markets catalog parquet."""
    return raw_root(telonex_dir) / POLYMARKET / "markets.parquet"


def ensure_markets_catalog(telonex_dir: Path, *, fetch: Fetch = urllib_fetch) -> Path:
    """Download the Polymarket markets catalog if not already on disk; return its path."""
    p = catalog_path(telonex_dir)
    if not p.exists():
        download_markets_dataset(POLYMARKET, p, fetch=fetch)
    return p


def windows_from_catalog(cat_path: Path, asset: str, dur: str, day: date,
                         *, require_channel: str = "book_snapshot_full") -> list[dict[str, Any]]:
    """Discover a day's ``{asset}-updown-{dur}`` windows-with-data from the markets catalog.

    Row-group-streams the (large) catalog reading only the needed columns; keeps windows
    whose open epoch falls on ``day`` and whose ``{require_channel}_from`` (else
    ``trades_from``) is non-empty. Returns the join keys (``slug``, ``open_epoch``,
    ``open_ms``, ``up_asset_id`` = the ``Up`` outcome's token, ``market_id``). One request
    instead of 288 availability probes — no throttling, and it yields the asset ids."""
    import pyarrow.parquet as pq  # local import: keep module import light

    rx = re.compile(rf"^{re.escape(asset)}-updown-{re.escape(dur)}-(\d+)$")
    start = day_start_epoch(day)
    end = start + 86_400
    pf = pq.ParquetFile(Path(cat_path))
    have = set(pf.schema_arrow.names)
    fromcol = f"{require_channel}_from" if f"{require_channel}_from" in have else (
        "trades_from" if "trades_from" in have else None)
    wanted = ["slug", "outcome_0", "outcome_1", "asset_id_0", "asset_id_1", "market_id"]
    cols = [c for c in wanted if c in have] + ([fromcol] if fromcol else [])
    out: list[dict[str, Any]] = []
    for batch in pf.iter_batches(batch_size=300_000, columns=cols):
        d = batch.to_pandas()
        m = d["slug"].astype(str).str.extract(rx)[0]
        mask = m.notna()
        if not mask.any():
            continue
        sub = d[mask].copy()
        sub["epoch"] = m[mask].astype("int64")
        sub = sub[(sub["epoch"] >= start) & (sub["epoch"] < end)]
        if fromcol:
            sub = sub[sub[fromcol].astype(str).str.len() > 0]
        for _, r in sub.iterrows():
            up = r.get("asset_id_0") if str(r.get("outcome_0", "")).lower() == "up" else r.get("asset_id_1")
            out.append({"slug": str(r["slug"]), "open_epoch": int(r["epoch"]), "open_ms": int(r["epoch"]) * 1000,
                        "up_asset_id": str(up), "market_id": str(r.get("market_id", "")), "covered_channels": []})
    out.sort(key=lambda w: w["open_epoch"])
    return out


# --------------------------------------------------------------------------- #
# Slugs / series / coverage helpers
# --------------------------------------------------------------------------- #

_SERIES_RE = re.compile(r"^([a-z]+)-updown-(5m|15m|1h)$")


def parse_series(series: str) -> tuple[str, str]:
    """``"btc-updown-5m"`` → ``("btc", "5m")``. Raises ``ValueError`` on a bad form."""
    m = _SERIES_RE.match(series.strip().lower())
    if not m:
        raise ValueError(f"bad series {series!r}; expected e.g. btc-updown-5m")
    return m.group(1), m.group(2)


def duration_seconds(dur: str) -> int:
    return _DUR_SECS[dur]


def day_start_epoch(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def updown_slugs(asset: str, dur: str, day: date) -> list[str]:
    """All ``{asset}-updown-{dur}-{epoch}`` window slugs on a UTC ``day`` (5m ⇒ 288)."""
    step = _DUR_SECS[dur]
    start = day_start_epoch(day)
    return [f"{asset}-updown-{dur}-{e}" for e in range(start, start + 86_400, step)]


def channel_covers_day(rng: dict[str, Any] | None, day: date) -> bool:
    """Whether an availability ``{from_date, to_date}`` (``to_date`` exclusive) covers ``day``."""
    if not isinstance(rng, dict):
        return False
    f, t = rng.get("from_date"), rng.get("to_date")
    if not f or not t:
        return False
    return str(f) <= day.isoformat() < str(t)


# --------------------------------------------------------------------------- #
# On-disk store layout
# --------------------------------------------------------------------------- #


def raw_root(telonex_dir: Path) -> Path:
    return Path(telonex_dir) / "raw"


def polymarket_raw_path(telonex_dir: Path, channel: str, slug: str, outcome: str, day_str: str) -> Path:
    return raw_root(telonex_dir) / POLYMARKET / channel / f"{slug}-{outcome}-{day_str}.parquet"


def binance_raw_path(telonex_dir: Path, channel: str, symbol: str, day_str: str) -> Path:
    return raw_root(telonex_dir) / BINANCE / channel / f"{symbol}-{day_str}.parquet"


def iter_raw_files(telonex_dir: Path) -> list[Path]:
    """All downloaded parquet files under ``<telonex_dir>/raw`` (sorted, no ``.tmp``)."""
    root = raw_root(telonex_dir)
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.parquet") if p.is_file())


def dir_disk_usage(telonex_dir: Path) -> int:
    return sum(p.stat().st_size for p in iter_raw_files(telonex_dir))


def free_disk_bytes(path: Path) -> int:
    """Free bytes on the filesystem holding ``path`` (walks up to an existing parent)."""
    p = Path(path)
    while not p.exists():
        if p.parent == p:
            break
        p = p.parent
    return shutil.disk_usage(p).free


def manifest_path(telonex_dir: Path) -> Path:
    return Path(telonex_dir) / MANIFEST_NAME


def build_manifest(telonex_dir: Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Scan ``<telonex_dir>/raw`` (ground truth, never drifts) into a coverage manifest.

    Contains NO key material. ``extra`` (e.g. trial parameters) is merged verbatim.
    """
    files = iter_raw_files(telonex_dir)
    by_group: dict[str, dict[str, int]] = {}
    total = 0
    for p in files:
        try:
            rel = p.relative_to(raw_root(telonex_dir))
            group = "/".join(rel.parts[:2]) if len(rel.parts) >= 2 else rel.parts[0]
        except ValueError:
            group = "other"
        b = p.stat().st_size
        g = by_group.setdefault(group, {"files": 0, "bytes": 0})
        g["files"] += 1
        g["bytes"] += b
        total += b
    now = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "updated_utc": now,
        "base_url": API_BASE,
        "file_count": len(files),
        "disk_usage_bytes": total,
        "disk_usage_human": human_bytes(total),
        "by_group": by_group,
    }
    if extra:
        manifest["trial"] = extra
    return manifest


def write_manifest(telonex_dir: Path, manifest: dict[str, Any]) -> None:
    Path(telonex_dir).mkdir(parents=True, exist_ok=True)
    manifest_path(telonex_dir).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_manifest(telonex_dir: Path) -> dict[str, Any] | None:
    p = manifest_path(telonex_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- #
# zstd whole-file compression (lossless, bit-identical roundtrip)
# --------------------------------------------------------------------------- #


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def compress_file_zstd(src: Path, dst: Path | None = None, *, level: int = 19) -> Path:
    """Whole-file zstd-compress ``src`` → ``dst`` (default ``src + ".zst"``), streamed."""
    import zstandard  # optional [telonex] extra — imported lazily

    src = Path(src)
    dst = Path(dst) if dst is not None else src.with_name(src.name + ".zst")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    cctx = zstandard.ZstdCompressor(level=level)
    with open(src, "rb") as fin, open(tmp, "wb") as fout:
        cctx.copy_stream(fin, fout, read_size=1 << 20, write_size=1 << 20)
    os.replace(tmp, dst)
    return dst


def decompress_zstd(src_zst: Path, dst: Path) -> Path:
    """Whole-file zstd-decompress ``src_zst`` → ``dst`` (streamed)."""
    import zstandard  # optional [telonex] extra — imported lazily

    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dctx = zstandard.ZstdDecompressor()
    with open(src_zst, "rb") as fin, open(dst, "wb") as fout:
        dctx.copy_stream(fin, fout, read_size=1 << 20, write_size=1 << 20)
    return dst


def verify_roundtrip(src: Path, src_zst: Path, *, scratch_dir: Path | None = None) -> bool:
    """Decompress ``src_zst`` and confirm it is **bit-identical** to ``src`` (sha256)."""
    src = Path(src)
    scratch_dir = Path(scratch_dir) if scratch_dir is not None else src.parent
    scratch_dir.mkdir(parents=True, exist_ok=True)
    tmp = scratch_dir / (src.name + ".roundtrip.tmp")
    try:
        decompress_zstd(src_zst, tmp)
        return sha256_file(tmp) == sha256_file(src)
    except Exception:  # noqa: BLE001 — a corrupt/undecompressable archive is a failed roundtrip
        return False
    finally:
        _safe_unlink(tmp)


# --------------------------------------------------------------------------- #
# Production full-pull helpers (model_lab.telonex_pull / telonex_coverage)
# --------------------------------------------------------------------------- #
#
# Additive, backward-compatible extensions used by the full-history download.
# ``telonex_ingest`` / ``telonex_validate`` do not use these — they live here so the
# network, on-disk, and catalog primitives stay in one place.


def zst_sibling(raw_path: Path) -> Path:
    """The ``.zst`` companion of a raw ``.parquet`` (``…parquet`` → ``…parquet.zst``)."""
    raw_path = Path(raw_path)
    return raw_path.with_name(raw_path.name + ".zst")


def raw_or_zst_exists(raw_path: Path) -> bool:
    """True if either the raw ``.parquet`` or its ``.zst`` is on disk.

    The download-skip predicate for the full pull: once a day is finalized the raw is
    deleted and only the ``.zst`` remains, so a resume must skip on the ``.zst`` too —
    otherwise it re-downloads (and re-bills) every finalized file. ``download_asset_day``
    alone only checks the raw path, which is why this exists."""
    raw_path = Path(raw_path)
    return raw_path.exists() or zst_sibling(raw_path).exists()


def zst_finalized(raw_path: Path) -> bool:
    """True when a file is in its final form: ``.zst`` present and raw ``.parquet`` gone."""
    raw_path = Path(raw_path)
    return zst_sibling(raw_path).exists() and not raw_path.exists()


def enumerate_pull_windows(
    cat_path: Path, series: Sequence[str], *, channels: Sequence[str] = ("quotes", "trades"),
) -> list[dict[str, Any]]:
    """One row-group-streamed pass over the markets catalog → every in-scope window.

    Generalizes :func:`windows_from_catalog` (one day, one series) and
    ``telonex_validate._scan_catalog`` (all series, presence only): scans the whole
    catalog once for all requested ``series`` (e.g. ``btc-updown-5m``), keeping a window
    iff ≥1 requested channel has a non-empty ``{channel}_from``. The catalog caution says
    unopened windows have empty availability — those are correctly excluded (never
    "missing"). 1h and non-requested series are excluded by construction.

    Each returned window carries its join keys plus, per requested channel that has data,
    the availability ``{from, to, covers_day}`` (``to`` exclusive; ``covers_day`` tests
    the window's UTC open day). Deduplicated by slug, sorted by ``(series, open_epoch)``.
    """
    import pyarrow.parquet as pq  # local import: keep module import light

    series_set = {s.strip().lower() for s in series}
    rx = re.compile(r"^([a-z]+)-updown-(5m|15m|1h)-(\d+)$")
    pf = pq.ParquetFile(Path(cat_path))
    have = set(pf.schema_arrow.names)
    chan_cols = [f"{ch}_{suf}" for ch in channels for suf in ("from", "to")
                 if f"{ch}_{suf}" in have]
    wanted = ["slug", "outcome_0", "outcome_1", "asset_id_0", "asset_id_1", "market_id"]
    cols = [c for c in wanted if c in have] + chan_cols
    out: dict[str, dict[str, Any]] = {}
    for batch in pf.iter_batches(batch_size=300_000, columns=cols):
        d = batch.to_pandas()
        m = d["slug"].astype(str).str.extract(rx)
        mask = m[0].notna()
        if not mask.any():
            continue
        sub = d[mask].copy()
        sub["asset"] = m.loc[mask, 0].values
        sub["dur"] = m.loc[mask, 1].values
        sub["epoch"] = m.loc[mask, 2].astype("int64").values
        sub["series"] = sub["asset"] + "-updown-" + sub["dur"]
        sub = sub[sub["series"].isin(series_set)]
        if sub.empty:
            continue
        for rd in sub.to_dict("records"):
            epoch = int(rd["epoch"])
            open_day = datetime.fromtimestamp(epoch, tz=timezone.utc).date()
            chans: dict[str, Any] = {}
            for ch in channels:
                frm = str(rd.get(f"{ch}_from") or "").strip()
                if not frm or frm.lower() == "nan":
                    continue
                to = str(rd.get(f"{ch}_to") or "").strip()
                chans[ch] = {"from": frm, "to": to,
                             "covers_day": channel_covers_day({"from_date": frm, "to_date": to}, open_day)}
            if not chans:
                continue
            outcome0 = str(rd.get("outcome_0", "")).lower()
            aid0, aid1 = str(rd.get("asset_id_0", "")), str(rd.get("asset_id_1", ""))
            up_id, down_id = (aid0, aid1) if outcome0 == "up" else (aid1, aid0)
            slug = str(rd["slug"])
            out[slug] = {
                "series": str(rd["series"]), "asset": str(rd["asset"]), "dur": str(rd["dur"]),
                "slug": slug, "open_epoch": epoch, "open_ms": epoch * 1000,
                "open_day": open_day.isoformat(), "up_asset_id": up_id, "down_asset_id": down_id,
                "market_id": str(rd.get("market_id", "")), "channels": chans,
            }
    return sorted(out.values(), key=lambda w: (w["series"], w["open_epoch"]))


def sample_roundtrip(
    pairs: Sequence[tuple[Path, Path]], *, k: int, rng: Any, scratch: Path,
) -> dict[str, Any]:
    """Bit-verify a random ``k`` of the ``(raw, zst)`` pairs via :func:`verify_roundtrip`.

    Per the full-pull spec we sample per day rather than roundtrip every file before
    deleting the raw; unsampled files trust the atomic streamed compression. Returns
    ``{"checked", "ok", "failed": [zst paths…]}``."""
    pairs = list(pairs)
    scratch = Path(scratch)
    if k <= 0 or not pairs:
        return {"checked": 0, "ok": 0, "failed": []}
    chosen = pairs if len(pairs) <= k else rng.sample(pairs, k)
    ok = 0
    failed: list[str] = []
    for raw, zst in chosen:
        if verify_roundtrip(Path(raw), Path(zst), scratch_dir=scratch):
            ok += 1
        else:
            failed.append(str(zst))
    return {"checked": len(chosen), "ok": ok, "failed": failed}


def probe_remaining(api_key: str, sample_url: str, *, fetch: Fetch = urllib_fetch) -> int | None:
    """Read ``X-Downloads-Remaining`` for a download URL **without streaming any data**.

    Issues the 302 (no-follow) GET so the header can be read, but never fetches the
    pre-signed S3 body — the closest thing to checking the quota "before downloading
    anything". Returns the remaining count, or ``None`` if the header is absent (the
    expected shape on an unlimited plan). A 403 raises :class:`TelonexDownloadLimit`."""
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = fetch(sample_url, headers, False)
    except TelonexHTTPError as e:
        if e.status == 403:
            raise _classify(e) from e
        return _remaining(e.headers)
    try:
        return _remaining(resp.headers)
    finally:
        resp.close()
