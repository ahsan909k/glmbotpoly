"""Minimal read-only client for the Polymarket **CLOB** public API — official market
resolutions.

``GET https://clob.polymarket.com/markets/{condition_id}`` returns the market object with
``tokens: [{token_id, outcome, price, winner}, …]``; the resolved winner is the token with
``winner == true`` (``price == 1``). Used to **validate** the Telonex catalog's ``result_id``
resolutions against the official source and to fill the handful the catalog lacks.

A ``fetch`` seam (default :func:`http_fetch_json`) keeps the callers testable offline.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable

CLOB_BASE = "https://clob.polymarket.com"
FetchFn = Callable[[str], object]


class PolymarketNotFound(Exception):
    """The market does not exist / is not served (HTTP 404)."""


class PolymarketHTTPError(Exception):
    """A non-404 HTTP or transport error after retries."""


def http_fetch_json(url: str, *, timeout: float = 20.0, retries: int = 3, backoff: float = 1.5):
    """GET ``url`` and parse JSON. ``PolymarketNotFound`` on 404 (never retried); transient
    transport / 5xx errors retried with linear backoff, then ``PolymarketHTTPError``."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "model-lab/0.1 (resolution validation)"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:  # subclass of URLError — catch first
            if e.code == 404:
                raise PolymarketNotFound(url) from e
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, json.JSONDecodeError) as e:
            last = e
        if attempt < retries - 1:
            time.sleep(backoff * (attempt + 1))
    raise PolymarketHTTPError(f"{url}: {last}")


def clob_market(condition_id: str, *, fetch: FetchFn = http_fetch_json, base: str = CLOB_BASE):
    """The CLOB market object for ``condition_id``, or ``None`` if not served (404)."""
    try:
        return fetch(f"{base}/markets/{condition_id}")
    except PolymarketNotFound:
        return None


def resolution_winner(market) -> str | None:
    """The winning outcome (``"Up"``/``"Down"``) of a resolved CLOB market, or ``None`` if
    the market is missing / not yet resolved / has no flagged winner token."""
    if not isinstance(market, dict):
        return None
    tokens = market.get("tokens") or []
    for t in tokens:
        if isinstance(t, dict) and t.get("winner"):
            out = t.get("outcome")
            if out in ("Up", "Down"):
                return out
    return None
