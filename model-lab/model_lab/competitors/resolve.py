"""Resolve Polymarket handles to proxy wallet addresses.

Input handles are of three kinds:

- **Full address** ``0x`` + 40 hex — used directly.
- **Username** (``wolf9478``, ``bonereaper``, and even ``0x50f7`` / ``0x33d33``
  which turn out to be *chosen usernames*, not truncated addresses) — resolved
  via Gamma ``/public-search`` by exact (case-insensitive) ``name`` match. The
  public API has **no** username->address endpoint other than this search.
- **Truncated address** (``0x`` + <40 hex) — best-effort reconstruction by
  appending the missing trailing hex nibble and keeping the candidate that has
  on-chain activity. Flagged low-confidence; never guessed silently.

Writes ``data/competitors/handles.json``. Run: ``python -m model_lab.competitors.resolve``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from . import api
from .paths import competitors_dir

# The operator's account list, order preserved.
DEFAULT_HANDLES: list[str] = [
    "wolf9478",
    "takerner",
    "0x50f7",
    "0x33d33",
    "nndrekop",
    "bonereaper",
    "polkadot-frog",
    "0xb27bc932bf8110d8f78e55da7d5f0497a18b5b82",
    "0xa6896d11f76dfa2820662c1f441496f51553559",
    "nagi777",  # merge-heavy account — added for the operating-manual roster (fetch fresh).
    "gabigol",  # @dontoverfit — reference-class maker on our BTC/ETH 5m+15m series (primary live-eval report card).
    "0xf3531b23b504cf0aed4ff21325232b2a2d496685",  # xrang2731 — taker-only Diamond.
]

_FULL_ADDR = re.compile(r"^0x[0-9a-fA-F]{40}$")
_ADDR_LIKE = re.compile(r"^0x[0-9a-fA-F]{4,39}$")  # 0x + too-few hex = truncated
_HEX = "0123456789abcdef"


def is_full_address(s: str) -> bool:
    return bool(_FULL_ADDR.match(s.strip()))


def _looks_truncated_address(s: str) -> bool:
    return bool(_ADDR_LIKE.match(s.strip()))


def _exact_profile(profiles: list[dict], handle: str) -> tuple[dict | None, list[dict]]:
    """Return (exact case-insensitive name match, other candidates)."""
    want = handle.strip().lower()
    exact = [p for p in profiles if str(p.get("name", "")).strip().lower() == want]
    others = [p for p in profiles if p not in exact]
    return (exact[0] if exact else None), (exact[1:] + others)


def _profile_meta(client: api.PolymarketClient, address: str) -> dict:
    """Public-profile enrichment (pseudonym / taker tier / volume), best-effort."""
    prof = client.public_profile(address) or {}
    return {
        "name": prof.get("name"),
        "pseudonym": prof.get("pseudonym"),
        "createdAt": prof.get("createdAt"),
        "takerTier": prof.get("takerTier"),
        "takerTierName": prof.get("takerTierName"),
        "weightedVolume": prof.get("weightedVolume"),
        "verifiedBadge": prof.get("verifiedBadge"),
    }


def _activity_sample_volume(client: api.PolymarketClient, address: str) -> tuple[int, float]:
    """(#records, Σ usdcSize) over one small activity page — a cheap 'does this
    address actually trade?' probe used to disambiguate reconstructed addresses."""
    try:
        page = client.activity_page(address, limit=20)
    except api.PolymarketHTTPError:
        return 0, 0.0
    vol = sum(float(r.get("usdcSize") or 0.0) for r in page)
    return len(page), vol


def _reconstruct_truncated(client: api.PolymarketClient, handle: str) -> dict:
    """Best-effort resolution of a ``0x`` + <40 hex handle by appending the missing
    trailing nibble(s). Only single-missing-nibble (16 candidates) is attempted —
    the overwhelmingly common copy-paste truncation. A candidate is accepted only
    if it has on-chain activity; ambiguity/absence is reported, never guessed."""
    core = handle.strip().lower()
    hexpart = core[2:]
    missing = 40 - len(hexpart)
    entry: dict = {
        "handle": handle, "kind": "truncated_address", "address": None,
        "confidence": "unresolved", "method": "end-append reconstruction",
        "notes": f"{len(hexpart)}/40 hex chars — {missing} missing", "alternatives": [],
    }
    if missing != 1:
        entry["notes"] += "; >1 missing nibble — not attempted (too many candidates)"
        return entry
    hits: list[dict] = []
    for c in _HEX:
        cand = "0x" + hexpart + c
        n, vol = _activity_sample_volume(client, cand)
        if n > 0:
            hits.append({"address": cand, "sample_records": n, "sample_volume_usdc": round(vol, 2)})
    if len(hits) == 1:
        h = hits[0]
        entry.update(address=h["address"], confidence="reconstructed",
                     notes=entry["notes"] + f"; unique active candidate ({h['sample_records']} "
                     f"sample records) — VERIFY before trusting")
        entry["profile"] = _profile_meta(client, h["address"])
    elif len(hits) > 1:
        entry.update(confidence="ambiguous", alternatives=hits,
                     notes=entry["notes"] + f"; {len(hits)} active candidates — ambiguous, flagged")
    else:
        entry["notes"] += "; no active single-nibble candidate found"
    return entry


def resolve_handle(client: api.PolymarketClient, handle: str) -> dict:
    h = handle.strip()
    if is_full_address(h):
        addr = h.lower()
        return {
            "handle": handle, "kind": "address", "address": addr,
            "confidence": "high", "method": "given as full 40-hex address",
            "notes": "", "alternatives": [], "profile": _profile_meta(client, addr),
        }
    # Try username resolution first (covers 0x50f7 / 0x33d33 chosen usernames).
    profiles = client.search_profiles(h)
    exact, others = _exact_profile(profiles, h)
    if exact and exact.get("proxyWallet"):
        addr = str(exact["proxyWallet"]).lower()
        alts = [{"name": p.get("name"), "proxyWallet": p.get("proxyWallet")}
                for p in others if p.get("proxyWallet")]
        return {
            "handle": handle, "kind": "username", "address": addr,
            "confidence": "high",
            "method": "gamma /public-search exact name match",
            "notes": (f"pseudonym={exact.get('pseudonym')}"
                      + (f"; {len(alts)} similar-name accounts also exist" if alts else "")),
            "alternatives": alts, "profile": _profile_meta(client, addr),
        }
    # No exact username; if it looks like a truncated address, try reconstruction.
    if _looks_truncated_address(h):
        return _reconstruct_truncated(client, h)
    # Otherwise: unresolved username (renamed / private / not found).
    near = [{"name": p.get("name"), "proxyWallet": p.get("proxyWallet")} for p in profiles[:5]]
    return {
        "handle": handle, "kind": "username", "address": None,
        "confidence": "unresolved",
        "method": "gamma /public-search (no exact name match)",
        "notes": "no public profile with this exact name — may be renamed, private "
                 "(displayUsernamePublic=false), or a different account",
        "alternatives": near,
    }


def resolve_all(client: api.PolymarketClient, handles: list[str]) -> list[dict]:
    out = []
    for h in handles:
        entry = resolve_handle(client, h)
        addr = entry.get("address") or "UNRESOLVED"
        print(f"[resolve] {h:<44} -> {addr}  ({entry['confidence']})")
        out.append(entry)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Resolve Polymarket handles -> wallet addresses.")
    ap.add_argument("--handles", nargs="*", default=None, help="override the default handle list")
    ap.add_argument("--out", default=None, help="output dir (default data/competitors)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out) if args.out else competitors_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    handles = args.handles or DEFAULT_HANDLES

    client = api.PolymarketClient()
    resolved = resolve_all(client, handles)

    path = out_dir / "handles.json"
    # Merge with any existing handles.json so `resolve --handles nagi777` ADDS an account
    # rather than overwriting the roster (each entry keyed by handle; order preserved).
    merged: dict[str, dict] = {}
    if path.exists():
        try:
            merged = {h["handle"]: h for h in json.loads(path.read_text(encoding="utf-8"))["handles"]}
        except (OSError, json.JSONDecodeError, KeyError):
            merged = {}
    for entry in resolved:
        merged[entry["handle"]] = entry
    out_handles = list(merged.values())
    path.write_text(json.dumps({"handles": out_handles, "requests": client.request_count},
                               indent=2), encoding="utf-8")
    n_ok = sum(1 for r in out_handles if r.get("address"))
    print(f"[resolve] wrote {path} — {n_ok}/{len(out_handles)} resolved "
          f"({len(resolved)} this run), {client.request_count} requests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
