//! Performance measurement over the journal: signed markouts on passive
//! fills (the adverse-selection detector — persistently negative markouts
//! mean our resting quotes are being picked off), PnL attribution split into
//! locked-pair PnL vs inventory PnL, fees paid and rebates earned, and
//! per-series aggregates (windows traded, hit rate, PnL per window, fill
//! counts, minimum-sample warnings) that power the dashboard's series
//! comparison table — the view the operator uses to narrow six series down
//! to the best two or three (CLAUDE.md §10).
