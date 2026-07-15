"""Read-only client for Polymarket's public HTTP APIs, with a fetch seam.

Two hosts are used, both public / unauthenticated:

- **Data API** ``https://data-api.polymarket.com`` — user activity, trades and
  positions, keyed by wallet address (``/activity``, ``/trades``, ``/positions``).
- **Gamma API** ``https://gamma-api.polymarket.com`` — ``/public-search`` (resolve
  a username/handle -> profile with ``proxyWallet``) and ``/public-profile``
  (profile by address).

Design mirrors ``io/polymarket.py``: a ``fetch`` seam (default :func:`http_get_json`)
keeps callers testable offline, transient errors are retried with backoff, and a
polite inter-request delay keeps us well under any rate limit — this is meant to
run alongside other jobs.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable

DATA_BASE = "https://data-api.polymarket.com"
GAMMA_BASE = "https://gamma-api.polymarket.com"
# Undocumented, site-internal: the profile P/L chart endpoint. Different host.
# GET /user-pnl?user_address=&interval=<1d|1w|1m|all>&fidelity=<1h|1d>
#   -> [{"t": unix_secs, "p": pnl}, ...] where `p` is CUMULATIVE all-time PnL
#   (realized + unrealized mark-to-market, in $), NOT rebased per interval.
USER_PNL_BASE = "https://user-pnl-api.polymarket.com"

# Polite defaults — this is background research, not a hot path.
DEFAULT_DELAY_S = 0.4          # sleep between successful requests
DEFAULT_TIMEOUT_S = 25.0
DEFAULT_RETRIES = 4
DEFAULT_BACKOFF_S = 2.0        # linear backoff base; 429 waits longer

# The Data API caps page size; offset is capped too, so large histories are
# walked by shifting the `end` time cursor (see fetch.py).
PAGE_LIMIT = 500
OFFSET_CAP = 10_000

FetchFn = Callable[[str], object]


class PolymarketHTTPError(Exception):
    """A non-404 HTTP / transport error after retries."""


class PolymarketNotFound(Exception):
    """HTTP 404 — never retried."""


class PolymarketBadRequest(Exception):
    """HTTP 400 — typically the offset/pagination cap on this endpoint. Never
    retried; the paginator treats it as end-of-window and advances the time cursor."""


def http_get_json(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF_S,
):
    """GET ``url`` -> parsed JSON. 404 -> :class:`PolymarketNotFound` (no retry);
    429 and transient 5xx / transport errors are retried with linear backoff
    (429 waits extra), then :class:`PolymarketHTTPError`."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "model-lab/0.1 (competitor research; read-only)"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:  # subclass of URLError — catch first
            if e.code == 404:
                raise PolymarketNotFound(url) from e
            if e.code == 400:
                raise PolymarketBadRequest(url) from e
            last = e
            wait = backoff * (attempt + 1)
            if e.code == 429:
                wait = max(wait, 5.0 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, json.JSONDecodeError) as e:
            last = e
            wait = backoff * (attempt + 1)
        if attempt < retries - 1:
            time.sleep(wait)
    raise PolymarketHTTPError(f"{url}: {last}")


def _build(base: str, path: str, params: dict) -> str:
    clean = {k: v for k, v in params.items() if v is not None}
    return f"{base}{path}?{urllib.parse.urlencode(clean, doseq=True)}"


class PolymarketClient:
    """Thin, rate-limited wrapper over the public endpoints used here."""

    def __init__(
        self,
        *,
        fetch: FetchFn = http_get_json,
        delay_s: float = DEFAULT_DELAY_S,
        data_base: str = DATA_BASE,
        gamma_base: str = GAMMA_BASE,
        user_pnl_base: str = USER_PNL_BASE,
    ) -> None:
        self._fetch = fetch
        self._delay_s = delay_s
        self._data_base = data_base
        self._gamma_base = gamma_base
        self._user_pnl_base = user_pnl_base
        self.request_count = 0

    def _get(self, url: str):
        out = self._fetch(url)
        self.request_count += 1
        if self._delay_s > 0:
            time.sleep(self._delay_s)
        return out

    # --- Gamma: resolution ------------------------------------------------
    def search_profiles(self, query: str, *, limit_per_type: int = 20) -> list[dict]:
        """``/public-search`` profile hits for ``query`` (a handle). Returns the
        ``profiles`` array (each ``{name, pseudonym, proxyWallet, ...}``), or []."""
        url = _build(
            self._gamma_base,
            "/public-search",
            {"q": query, "search_profiles": "true", "limit_per_type": limit_per_type,
             "events_status": "all"},
        )
        try:
            out = self._get(url)
        except PolymarketNotFound:
            return []
        if isinstance(out, dict):
            profs = out.get("profiles")
            if isinstance(profs, list):
                return profs
        return []

    def public_profile(self, address: str) -> dict | None:
        """``/public-profile?address=`` -> profile dict, or None if not found."""
        url = _build(self._gamma_base, "/public-profile", {"address": address})
        try:
            out = self._get(url)
        except PolymarketNotFound:
            return None
        return out if isinstance(out, dict) else None

    # --- Data API: user history -------------------------------------------
    def _get_page(self, url: str) -> list[dict]:
        """A paginated GET where HTTP 400 (offset past the endpoint's cap) is a
        soft end-of-window signal, returned as an empty page."""
        try:
            return _as_list(self._get(url))
        except (PolymarketBadRequest, PolymarketNotFound):
            return []

    def activity_page(
        self, user: str, *, limit: int = PAGE_LIMIT, offset: int = 0,
        types: str | None = None, end: int | None = None, start: int | None = None,
    ) -> list[dict]:
        """One page of ``/activity`` (newest-first). ``types`` is a comma list of
        TRADE,SPLIT,MERGE,REDEEM,REWARD,CONVERSION; ``end``/``start`` are epoch
        seconds. Returns the JSON array (empty at/after the offset cap)."""
        url = _build(self._data_base, "/activity", {
            "user": user, "limit": limit, "offset": offset, "type": types,
            "end": end, "start": start, "sortBy": "TIMESTAMP", "sortDirection": "DESC",
        })
        return self._get_page(url)

    def trades_page(
        self, user: str, *, limit: int = PAGE_LIMIT, offset: int = 0,
        taker_only: bool = True, end: int | None = None, start: int | None = None,
    ) -> list[dict]:
        """One page of ``/trades`` (newest-first). ``taker_only=True`` returns only
        trades where the user was the taker (aggressor); ``False`` returns all
        trades involving the user. Returns the JSON array (possibly empty)."""
        url = _build(self._data_base, "/trades", {
            "user": user, "limit": limit, "offset": offset,
            "takerOnly": "true" if taker_only else "false",
            "end": end, "start": start,
        })
        return self._get_page(url)

    def positions(self, user: str, *, limit: int = 500, offset: int = 0,
                  size_threshold: float = 0.0) -> list[dict]:
        """One page of ``/positions`` (current open positions)."""
        url = _build(self._data_base, "/positions", {
            "user": user, "limit": limit, "offset": offset,
            "sizeThreshold": size_threshold,
        })
        return self._get_page(url)

    # --- user-pnl: the official profile P/L chart (authoritative) ----------
    def user_pnl(self, address: str, *, interval: str = "all",
                 fidelity: str = "1d") -> list[dict]:
        """The official Polymarket profile P/L series (different host).

        Returns ``[{"t": unix_secs, "p": pnl}, ...]`` where ``p`` is CUMULATIVE
        all-time PnL (realized + unrealized MTM, in $), not rebased per interval.
        The last point is the account's authoritative all-time PnL. Empty list on
        404 / bad request. Note: this is NOT paginated (one shot per interval)."""
        url = _build(self._user_pnl_base, "/user-pnl", {
            "user_address": address, "interval": interval, "fidelity": fidelity,
        })
        try:
            return _as_list(self._get(url))
        except (PolymarketNotFound, PolymarketBadRequest):
            return []


def _as_list(out) -> list[dict]:
    if isinstance(out, list):
        return [x for x in out if isinstance(x, dict)]
    # Some endpoints wrap results; tolerate {"data": [...]}.
    if isinstance(out, dict):
        for key in ("data", "history", "positions"):
            v = out.get(key)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []
