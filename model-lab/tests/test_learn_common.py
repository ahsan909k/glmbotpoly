"""Contract test for the shared walk-forward scaffolding (``learn_common``).

Both challengers (``learn`` and ``learn_gbt``) import the fold schedule, purge,
matrix loader, and harness-scoring helpers from ``learn_common``. This test imports
those names so a future rename/removal fails loudly here (and it is lightgbm-free,
so it runs in the core suite). It also pins that ``learn`` still re-exports the
fold helpers ``test_learn.py`` depends on.
"""

from __future__ import annotations

import numpy as np


def test_learn_common_exports_are_stable():
    from model_lab import learn_common as lc

    assert lc.MS_PER_DAY == 86_400_000
    assert set(lc.TARGETS) == {"fwd30", "outcome"}
    assert lc.TARGETS["fwd30"]["label"] == "fwd_up_30s"
    assert lc.TARGETS["outcome"]["horizon_secs"] is None
    for name in ("_parse_series", "_load_matrix", "_fold_schedule", "_fold_masks",
                 "_fold_metrics", "_label_info_ms", "_harness_grid",
                 "_harness_summary", "score_through_harness"):
        assert callable(getattr(lc, name)), name


def test_learn_reexports_fold_helpers():
    # test_learn.py imports these from model_lab.learn — the extraction must keep them.
    from model_lab.learn import MS_PER_DAY, TARGETS, _fold_masks, _fold_schedule  # noqa: F401

    folds, mode = _fold_schedule(np.arange(0, 21), train_days=7, test_days=2)
    assert mode == "walk_forward" and folds


def test_score_through_harness_signature_takes_subdir():
    import inspect

    from model_lab.learn_common import score_through_harness

    params = inspect.signature(score_through_harness).parameters
    assert "subdir" in params and params["subdir"].kind == inspect.Parameter.KEYWORD_ONLY
