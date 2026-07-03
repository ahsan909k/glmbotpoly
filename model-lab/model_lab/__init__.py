"""model-lab — offline research over the Polymarket Up/Down bot's journal.

A standalone Python research project, fully separate from the Rust trading
engine. It reads the bot's gzip JSONL journal (and the new Binance
``depth20@100ms`` capture), reproduces and validates the engine's fair-value
math offline, and researches whether L2 order-book microstructure signals add
predictive value.

Run one stage at a time, e.g. ``python -m model_lab.ingest`` — see the README.
"""

__version__ = "0.1.0"
