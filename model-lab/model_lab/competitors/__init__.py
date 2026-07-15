"""Read-only competitor analysis over Polymarket public data.

A self-contained research tool (NOT part of the trading bot, NOT part of the
model-lab training pipeline): resolve a set of Polymarket handles to wallet
addresses, pull each account's public trade / activity / position history from
``data-api.polymarket.com`` (cached + resumable under ``data/competitors/``),
restrict to our four series (btc/eth up-down 5m & 15m), classify each account by
measurable behavioural signatures, and render ``out/competitors/report.html``.

Everything here is read-only against public endpoints. No bot state is touched.

Stages (run via ``python -m model_lab.competitors.<stage>`` or ``... all``):
  resolve  — handle -> proxy wallet address (data/competitors/handles.json)
  fetch    — pull activity/trades/positions per account (cached, resumable)
  analyze  — compute per-account signatures + PnL (out/competitors/analysis.json)
  report   — render out/competitors/report.html
"""
