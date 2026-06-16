"use strict";
/* Polymarket bot dashboard — single-page UI, no framework, no build step.
   Talks to the backend's REST snapshots + WebSocket push (CLAUDE.md §10). */

// ---------------------------------------------------------------- token / auth
const TOKEN_KEY = "pmbot_token";
function getToken() { return localStorage.getItem(TOKEN_KEY) || ""; }
function setToken(t) {
  if (t) localStorage.setItem(TOKEN_KEY, t); else localStorage.removeItem(TOKEN_KEY);
}
// Convenience: open .../#token=SECRET → store it and clean the address bar.
(function seedTokenFromHash() {
  const m = /(?:^|[#&])token=([^&]+)/.exec(location.hash || "");
  if (m) {
    setToken(decodeURIComponent(m[1]));
    history.replaceState(null, "", location.pathname + location.search);
  }
})();

function authHeaders(extra) {
  const h = Object.assign({}, extra || {});
  const t = getToken();
  if (t) h["Authorization"] = "Bearer " + t;
  return h;
}

async function api(path, opts) {
  opts = opts || {};
  const init = { method: opts.method || "GET", headers: authHeaders(opts.headers) };
  if (opts.body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(opts.body);
  }
  const resp = await fetch(path, init);
  if (resp.status === 401) { showAuthBanner(true); throw new Error("unauthorized"); }
  if (!resp.ok) {
    let msg = resp.status + "";
    try { const b = await resp.json(); if (b && b.error) msg = b.error; } catch (e) {}
    throw new Error(msg);
  }
  const txt = await resp.text();
  return txt ? JSON.parse(txt) : null;
}

// ----------------------------------------------------------------------- state
const state = {
  mode: "paper",
  view: "overview",
  compareWindow: "all",
  sort: { col: null, dir: null },     // null → server default (NetPnl desc)
  sortColumns: {},                    // key → {label, higher_is_better}
  equity: { paper: [], live: [] },    // [{t, v}]
  killed: false,
  overview: null,
  live: { windows: [], selected: null, detail: null },
  fills: { series: "", rows: [] },
  risk: null,
  params: { noAtm: 25, noPassive: 5, lateTau: 30, atmBand: 0.10 },
  paramsLoaded: false,
};
const EQUITY_CAP = 5000;

// --------------------------------------------------------------------- helpers
const $ = (id) => document.getElementById(id);
const num = (s) => { const n = parseFloat(s); return Number.isFinite(n) ? n : 0; };

function fmtUsd(v, signed) {
  const n = typeof v === "string" ? num(v) : (v || 0);
  const abs = Math.abs(n).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const sign = signed ? (n > 0 ? "+" : n < 0 ? "−" : "") : (n < 0 ? "−" : "");
  return sign + "$" + abs;
}
function moneyCell(v, signed) {
  const n = num(v);
  const cls = n > 0 ? "pos" : n < 0 ? "neg" : "zero";
  return `<span class="${signed ? cls : ""}">${fmtUsd(v, signed)}</span>`;
}
function pctCell(f) { return ((f || 0) * 100).toFixed(0) + "%"; }
function markoutCell(f) {
  if (f === null || f === undefined) return `<span class="zero">—</span>`;
  const cls = f > 0 ? "pos" : f < 0 ? "neg" : "zero";
  const sign = f > 0 ? "+" : f < 0 ? "−" : "";
  return `<span class="${cls}">${sign}${Math.abs(f).toFixed(4)}</span>`;
}
function fracCell(f) { return (f === null || f === undefined) ? "—" : (f * 100).toFixed(0) + "%"; }
function healthCell(h) {
  if (h === "Ok") return `<span class="pill ok">OK</span>`;
  if (h === "Alarm") return `<span class="pill alarm">ALARM</span>`;
  return `<span class="pill lown">low n</span>`;
}

function toast(msg, isErr) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.toggle("err", !!isErr);
  t.classList.remove("hidden");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.add("hidden"), 2600);
}

function confirmAction(text, onOk) {
  const m = $("modal");
  $("modal-text").textContent = text;
  m.classList.remove("hidden");
  const ok = $("modal-ok"), cancel = $("modal-cancel");
  const close = () => { m.classList.add("hidden"); ok.onclick = null; cancel.onclick = null; };
  ok.onclick = () => { close(); onOk(); };
  cancel.onclick = close;
}

function showAuthBanner(show) { $("auth-banner").classList.toggle("hidden", !show); }

// ------------------------------------------------------------------- overview
const COLUMNS = [
  { key: "WindowsTraded", field: "windows_traded", kind: "count", label: "Windows",
    tip: "Windows traded in the selected period. Low counts are muted — too little sample to trust." },
  { key: "FractionProfitable", field: "fraction_profitable", kind: "pct", label: "% profit",
    tip: "Share of traded windows that ended with positive PnL." },
  { key: "NetPnl", field: "net_pnl", kind: "money", label: "Net PnL",
    tip: "Net ledger PnL over the period (locked-pair + inventory − fees + rebates)." },
  { key: "PnlPerWindow", field: "pnl_per_window", kind: "money", label: "PnL / win",
    tip: "Net PnL divided by windows traded — the per-window normalization for comparing series." },
  { key: "LockedPairPnl", field: "locked_pair_pnl", kind: "money", label: "Locked",
    tip: "PnL from matched Up+Down pairs — the guaranteed-edge half of the split." },
  { key: "InventoryPnl", field: "inventory_pnl", kind: "money", label: "Inventory",
    tip: "PnL from unmatched inventory + settlement remainder — the risk-taking half." },
  { key: "FeesPaid", field: "fees_paid", kind: "cost", label: "Fees",
    tip: "Taker fees paid (a cost). Makers pay zero." },
  { key: "RebatesEarned", field: "rebates_earned", kind: "money", label: "Rebates",
    tip: "Estimated maker rebates earned." },
  { key: "AvgMarkout5s", field: "avg_markout_5s", kind: "markout", label: "5s markout",
    tip: "Avg 5-second fair-value move after a passive fill. Persistently negative = adverse selection (being picked off)." },
  { key: "MakerFills", field: "maker_fills", kind: "count", label: "Maker",
    tip: "Number of maker (passive) fills." },
  { key: "TakerFills", field: "taker_fills", kind: "count", label: "Taker",
    tip: "Number of taker (aggressive) fills." },
  { key: "TakerBudgetUsed", field: "taker_budget_used_fraction", kind: "frac", label: "Taker bgt",
    tip: "Mean per-window taker spend as a fraction of the taker budget. 100% = budget fully used." },
  { key: "Health", field: "health", kind: "health", label: "Health",
    tip: "Adverse-selection health: OK, ALARM (negative markout), or low-n (insufficient sample)." },
];

function renderBadges() {
  const o = state.overview;
  const set = (which) => {
    const m = o ? o.modes[which] : null;
    const badge = $("badge-" + which);
    const stateEl = $("badge-" + which + "-state");
    badge.classList.remove("on", "armed");
    if (!m) { stateEl.textContent = "—"; return; }
    if (which === "live") {
      badge.classList.toggle("on", m.running);
      if (m.armed) badge.classList.add("armed");
      stateEl.textContent = m.running ? "running" : (m.armed ? "armed" : "off");
    } else {
      badge.classList.toggle("on", m.running);
      stateEl.textContent = m.running ? "running" : "stopped";
    }
  };
  set("paper"); set("live");
}

function renderKill() {
  $("kill-banner").classList.toggle("hidden", !state.killed);
  $("risk-kill-banner").classList.toggle("hidden", !state.killed);
}

function renderCapital() {
  const isPaper = state.mode === "paper";
  $("capital-card").classList.toggle("hidden", !isPaper);
  if (!isPaper || !state.overview) return;
  const m = state.overview.modes.paper;
  $("cap-start").textContent = m.paper_capital ? fmtUsd(m.paper_capital, false) : "—";
  $("cap-equity").textContent = m.wallet ? fmtUsd(m.wallet.collateral_total, false) : "—";
  const l = m.ledger;
  $("inc-fees").textContent = l ? fmtUsd(l.fees_paid, false) : "—";
  $("inc-reb-acc").textContent = l ? fmtUsd(l.rebate_accrued, false) : "—";
  $("inc-reb-cr").textContent = l ? fmtUsd(l.rebate_credited, false) : "—";
}

// hand-rolled canvas equity chart (no chart library) -------------------------
function drawEquityChart() {
  const canvas = $("equity-chart");
  const pts = state.equity[state.mode] || [];
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || canvas.parentElement.clientWidth || 320;
  const cssH = 180;
  canvas.width = Math.floor(cssW * dpr);
  canvas.height = Math.floor(cssH * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  $("equity-mode").textContent = state.mode;
  if (pts.length === 0) {
    $("equity-now").textContent = "—"; $("equity-min").textContent = "—"; $("equity-max").textContent = "—";
    ctx.fillStyle = "#8b97a7"; ctx.font = "13px sans-serif"; ctx.textAlign = "center";
    ctx.fillText("no equity samples yet", cssW / 2, cssH / 2);
    return;
  }
  const xs = pts.map((p) => p.t), ys = pts.map((p) => p.v);
  let lo = Math.min.apply(null, ys), hi = Math.max.apply(null, ys);
  if (lo === hi) { lo -= 1; hi += 1; }
  const padY = (hi - lo) * 0.08; lo -= padY; hi += padY;
  const x0 = xs[0], x1 = xs[xs.length - 1] || x0 + 1;
  const padL = 6, padR = 6, padT = 8, padB = 8;
  const W = cssW - padL - padR, H = cssH - padT - padB;
  const px = (t) => padL + (x1 === x0 ? W : ((t - x0) / (x1 - x0)) * W);
  const py = (v) => padT + (1 - (v - lo) / (hi - lo)) * H;

  // area fill + line
  ctx.beginPath();
  ctx.moveTo(px(xs[0]), py(ys[0]));
  for (let i = 1; i < pts.length; i++) ctx.lineTo(px(xs[i]), py(ys[i]));
  const lastUp = ys[ys.length - 1] >= ys[0];
  const line = lastUp ? "#2ecc71" : "#ff5c5c";
  ctx.lineTo(px(xs[xs.length - 1]), padT + H);
  ctx.lineTo(px(xs[0]), padT + H);
  ctx.closePath();
  ctx.fillStyle = lastUp ? "rgba(46,204,113,.10)" : "rgba(255,92,92,.10)";
  ctx.fill();

  ctx.beginPath();
  ctx.moveTo(px(xs[0]), py(ys[0]));
  for (let i = 1; i < pts.length; i++) ctx.lineTo(px(xs[i]), py(ys[i]));
  ctx.strokeStyle = line; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.stroke();

  // last point marker
  const lx = px(xs[xs.length - 1]), ly = py(ys[ys.length - 1]);
  ctx.fillStyle = line; ctx.beginPath(); ctx.arc(lx, ly, 3, 0, Math.PI * 2); ctx.fill();

  $("equity-now").textContent = fmtUsd(ys[ys.length - 1], false);
  $("equity-min").textContent = "min " + fmtUsd(Math.min.apply(null, ys), false);
  $("equity-max").textContent = "max " + fmtUsd(Math.max.apply(null, ys), false);
}

// --------------------------------------------------------------- series table
function currentSortCol(defaultSort) { return state.sort.col || defaultSort; }
function currentSortDir(col) {
  if (state.sort.col === col && state.sort.dir) return state.sort.dir;
  const meta = state.sortColumns[col];
  return (meta && meta.higher_is_better === false) ? "asc" : "desc";
}

function renderSeriesHead(defaultSort) {
  const tr = $("series-head");
  const sortedCol = currentSortCol(defaultSort);
  const dir = currentSortDir(sortedCol);
  let html = `<th class="col-series">Series</th>`;
  for (const c of COLUMNS) {
    const label = (state.sortColumns[c.key] && state.sortColumns[c.key].label) || c.label;
    const isSorted = c.key === sortedCol;
    const arrow = isSorted ? `<span class="arrow">${dir === "asc" ? "▲" : "▼"}</span>` : "";
    html += `<th data-key="${c.key}" class="${isSorted ? "sorted" : ""}" title="${c.tip}">` +
            `${label}${arrow}<span class="info" data-tip="${encodeURIComponent(c.tip)}">&#9432;</span></th>`;
  }
  tr.innerHTML = html;
  tr.querySelectorAll("th[data-key]").forEach((th) => {
    th.addEventListener("click", (e) => {
      if (e.target.classList.contains("info")) {
        e.stopPropagation();
        showTip(e.target, decodeURIComponent(e.target.dataset.tip));
        return;
      }
      onSortClick(th.dataset.key);
    });
  });
}

function renderSeriesBody(rows) {
  const tb = $("series-body");
  $("series-empty").classList.toggle("hidden", rows.length > 0);
  let html = "";
  for (const r of rows) {
    const muted = r.min_sample_warning ? " class=\"muted\"" : "";
    let cells = `<td class="col-series">${r.series}</td>`;
    for (const c of COLUMNS) {
      const v = r[c.field];
      let cell;
      switch (c.kind) {
        case "money": cell = moneyCell(v, true); break;
        case "cost": cell = moneyCell(v, false); break;
        case "pct": cell = pctCell(v); break;
        case "markout": cell = markoutCell(v); break;
        case "frac": cell = fracCell(v); break;
        case "health": cell = healthCell(v); break;
        default: cell = (v === null || v === undefined) ? "—" : v;
      }
      cells += `<td>${cell}</td>`;
    }
    html += `<tr${muted}>${cells}</tr>`;
  }
  tb.innerHTML = html;
}

function onSortClick(key) {
  if (state.sort.col === key) {
    state.sort.dir = state.sort.dir === "asc" ? "desc" : "asc";
  } else {
    state.sort.col = key;
    state.sort.dir = (state.sortColumns[key] && state.sortColumns[key].higher_is_better === false) ? "asc" : "desc";
  }
  loadSeries();
}

function showTip(anchor, text) {
  const tip = $("tip");
  tip.textContent = text;
  tip.classList.remove("hidden");
  const r = anchor.getBoundingClientRect();
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  let left = Math.min(r.left, window.innerWidth - tw - 8);
  let top = r.bottom + 6;
  if (top + th > window.innerHeight) top = r.top - th - 6;
  tip.style.left = Math.max(8, left) + "px";
  tip.style.top = Math.max(8, top) + "px";
  clearTimeout(showTip._t);
  showTip._t = setTimeout(() => tip.classList.add("hidden"), 4000);
}
document.addEventListener("click", (e) => {
  if (!e.target.classList || !e.target.classList.contains("info")) $("tip").classList.add("hidden");
});

// ----------------------------------------------------------------- data loads
async function loadOverview() {
  try {
    const o = await api("/api/overview");
    state.overview = o;
    // Replace local equity from the authoritative snapshot (WS appends after).
    for (const m of ["paper", "live"]) {
      const arr = (o.modes[m].equity || []).map((p) => ({ t: p.ts_ms, v: num(p.equity) }));
      state.equity[m] = arr.slice(-EQUITY_CAP);
    }
    renderBadges(); renderCapital(); drawEquityChart();
    showAuthBanner(false);
  } catch (e) { if (e.message !== "unauthorized") toast("overview: " + e.message, true); }
}

async function loadSeries() {
  try {
    let url = `/api/series-comparison?mode=${state.mode}&window=${state.compareWindow}`;
    if (state.sort.col) url += `&sort=${state.sort.col}&dir=${state.sort.dir}`;
    const resp = await api(url);
    state.sortColumns = {};
    for (const c of resp.sort_columns) state.sortColumns[c.key] = { label: c.label, higher_is_better: c.higher_is_better };
    renderSeriesHead(resp.comparison.default_sort);
    renderSeriesBody(resp.comparison.rows);
    showAuthBanner(false);
  } catch (e) { if (e.message !== "unauthorized") toast("series: " + e.message, true); }
}

function loadAll() {
  if (!state.paramsLoaded) loadParams();
  loadOverview();
  refreshActiveView();
}

let refreshTimer = null;
function scheduleRefresh() {
  if (refreshTimer) return;
  refreshTimer = setTimeout(() => {
    refreshTimer = null;
    loadOverview();
    if (state.view === "series") loadSeries();
  }, 1500);
}

// ----------------------------------------------- live / fills / risk helpers
function fmtClock(ms) { return new Date(ms).toLocaleTimeString([], { hour12: false }); }
function fmtCountdown(secs) {
  secs = Math.max(0, Math.floor(secs));
  return `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`;
}
function seriesOf(windowKey) { return (windowKey || "").split("@")[0]; }
function windowSecs(seriesKey) {
  const dur = (seriesKey || "").split("-")[1];
  if (dur === "5m") return 300;
  if (dur === "15m") return 900;
  if (dur === "1h") return 3600;
  return 300;
}
function isActiveLifecycle(lc) { return lc === "Open" || lc === "Closing"; }
function px3(p) { return num(p).toFixed(3); }

async function loadParams() {
  try {
    const p = await api("/api/params");
    const m = {};
    for (const e of (p.entries || [])) m[e.key] = e.value;
    const pick = (k, def) => { const v = parseFloat(m[k]); return Number.isFinite(v) ? v : def; };
    state.params.noAtm = pick("engine.no_atm_final_secs", 25);
    state.params.noPassive = pick("engine.no_passive_final_secs", 5);
    state.params.lateTau = pick("engine.late_window_tau_secs", 30);
    state.params.atmBand = pick("engine.atm_band", 0.10);
    state.paramsLoaded = true;
  } catch (e) { /* params are best-effort; defaults stand */ }
}

// ---- live window view ----
async function loadWindows() {
  try {
    const list = await api(`/api/windows?mode=${state.mode}`);
    const active = (list || []).filter((w) => isActiveLifecycle(w.lifecycle));
    active.sort((a, b) => seriesOf(a.window).localeCompare(seriesOf(b.window)));
    state.live.windows = active;
    if (!active.find((w) => w.window === state.live.selected)) {
      state.live.selected = active.length ? active[0].window : null;
    }
    renderWindowChips();
    if (state.live.selected) loadWindowDetail();
    else renderLiveDetail(null);
    showAuthBanner(false);
  } catch (e) { if (e.message !== "unauthorized") toast("windows: " + e.message, true); }
}

async function loadWindowDetail() {
  if (!state.live.selected) { renderLiveDetail(null); return; }
  try {
    const d = await api(`/api/windows/${encodeURIComponent(state.live.selected)}?mode=${state.mode}`);
    renderLiveDetail(d);
  } catch (e) {
    if (e.message === "not found" || e.message === "404") loadWindows();
    // else transient: keep the last detail rendered
  }
}

function renderWindowChips() {
  const el = $("window-chips");
  el.innerHTML = state.live.windows.map((w) => {
    const sel = w.window === state.live.selected ? " active" : "";
    const cls = w.lifecycle === "Closing" ? " closing" : " live";
    return `<button class="chip${cls}${sel}" data-window="${w.window}"><span class="chip-dot"></span>${seriesOf(w.window)}</button>`;
  }).join("");
  el.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => {
    state.live.selected = c.dataset.window;
    renderWindowChips();
    loadWindowDetail();
  }));
}

function renderLiveDetail(d) {
  state.live.detail = d || null;
  $("live-empty").classList.toggle("hidden", !!d);
  $("live-card").classList.toggle("hidden", !d);
  if (!d) return;
  $("live-title").textContent = seriesOf(d.window);
  const link = $("live-link");
  if (d.event_slug) { link.href = `https://polymarket.com/event/${d.event_slug}`; link.classList.remove("hidden"); }
  else link.classList.add("hidden");
  renderCountdown();
  renderFairMid(d);
  renderLadders(d);
  renderInventory(d);
  renderPrints(d);
}

function renderCountdown() {
  const d = state.live.detail;
  if (!d) return;
  const total = windowSecs(seriesOf(d.window));
  const tau = Math.max(0, (d.close_time_ms - Date.now()) / 1000);
  const elapsed = total > 0 ? Math.min(1, (total - tau) / total) : 1;
  const fill = $("count-fill");
  fill.style.width = (elapsed * 100).toFixed(2) + "%";
  fill.classList.remove("atm", "passive");
  if (tau <= state.params.noPassive) fill.classList.add("passive");
  else if (tau <= state.params.noAtm) fill.classList.add("atm");
  const gx = (g) => total > 0 ? Math.max(0, Math.min(100, ((total - g) / total) * 100)) : 100;
  setGate("gate-late", gx(state.params.lateTau), tau <= state.params.lateTau);
  setGate("gate-atm", gx(state.params.noAtm), tau <= state.params.noAtm);
  setGate("gate-passive", gx(state.params.noPassive), tau <= state.params.noPassive);
  $("count-label").textContent = "closes in " + fmtCountdown(tau);
  const g = (cls, lbl, on) => `<span class="g ${cls}${on ? " on" : ""}">${lbl}</span>`;
  $("count-gates").innerHTML =
    g("late", `late ${state.params.lateTau}s`, tau <= state.params.lateTau) +
    g("atm", `no-ATM ${state.params.noAtm}s`, tau <= state.params.noAtm) +
    g("passive", `cancel ${state.params.noPassive}s`, tau <= state.params.noPassive);
}
function setGate(id, leftPct, active) {
  const el = $(id);
  el.style.left = leftPct.toFixed(2) + "%";
  el.classList.toggle("active", active);
}

function healthPill(h) { return h === "Ready" ? "ok" : h === "Degraded" ? "taker" : "alarm"; }

function renderFairMid(d) {
  const m = d.model;
  const pUp = m && Number.isFinite(m.p_up) ? m.p_up : null;
  const upMid = d.up_mid != null ? num(d.up_mid) : null;
  const delta = (pUp != null && upMid != null) ? pUp - upMid : null;
  const fm = (label, val, extra) => `<div class="fm${extra ? " " + extra : ""}"><label>${label}</label><span>${val}</span></div>`;
  let html = fm("fair (p_up)", pUp != null ? pUp.toFixed(3) : "—");
  html += fm("Up mid", upMid != null ? upMid.toFixed(3) : "—");
  if (delta != null) {
    const cls = delta > 0 ? "pos" : delta < 0 ? "neg" : "zero";
    html += `<div class="fm"><label>Δ fair−mid</label><span class="${cls}">${delta > 0 ? "+" : ""}${delta.toFixed(3)}</span></div>`;
  } else html += fm("Δ fair−mid", "—");
  if (m) {
    html += fm("health", `<span class="pill ${healthPill(m.health)}">${m.health || "—"}</span>`, "small");
    html += fm("anchor", m.anchor || "—", "small");
  } else {
    html += `<div class="fm small"><label>model</label><span class="mk-pending">off — run with --with-model</span></div>`;
  }
  $("fair-mid").innerHTML = html;
}

function ladderRows(book, ours, outcome) {
  if (!book) return `<div class="lad-empty">no book</div>`;
  const oursAt = {};
  for (const o of (ours || [])) {
    if (o.outcome === outcome && o.side === "Buy") oursAt[px3(o.price)] = o.remaining;
  }
  const asks = (book.asks || []).slice(0, 6).reverse(); // high→low above the touch
  const bids = (book.bids || []).slice(0, 6);           // high→low below the touch
  let html = "";
  for (const l of asks) {
    html += `<div class="lad-row ask"><span class="lad-price">${px3(l.price)}</span><span class="lad-size">${num(l.size)}</span></div>`;
  }
  html += `<div class="lad-row spread">— spread —</div>`;
  for (const l of bids) {
    const key = px3(l.price);
    const mine = oursAt[key];
    const tag = mine !== undefined ? ` <span class="lad-ours">●${num(mine)}</span>` : "";
    html += `<div class="lad-row bid${mine !== undefined ? " ours" : ""}"><span class="lad-price">${key}${tag}</span><span class="lad-size">${num(l.size)}</span></div>`;
  }
  return html;
}
function renderLadders(d) {
  $("ladder-up").innerHTML = ladderRows(d.up_book, d.our_orders, "Up");
  $("ladder-down").innerHTML = ladderRows(d.down_book, d.our_orders, "Down");
  $("up-mid").textContent = d.up_mid != null ? num(d.up_mid).toFixed(3) : "";
  $("down-mid").textContent = d.down_mid != null ? num(d.down_mid).toFixed(3) : "";
}

function renderInventory(d) {
  const inv = d.inventory;
  const el = $("inv-row");
  if (!inv) { el.innerHTML = `<span class="inv-chip">no position</span>`; return; }
  const chip = (label, val) => `<span class="inv-chip">${label} <b>${val}</b></span>`;
  const up = inv.up || { shares: "0" };
  const dn = inv.down || { shares: "0" };
  const exc = num(inv.excess);
  el.innerHTML =
    chip("Up", `${num(up.shares)} sh`) +
    chip("Down", `${num(dn.shares)} sh`) +
    chip("pairs", num(inv.matched_pairs)) +
    chip("pair-cost", inv.pair_cost != null ? num(inv.pair_cost).toFixed(3) : "—") +
    chip("excess", `${exc > 0 ? "+" : ""}${exc}`) +
    chip("worst-case", fmtUsd(inv.worst_case_if_excess_loses, false));
}

function renderPrints(d) {
  const prints = (d.recent_prints || []).slice().reverse().slice(0, 24); // newest first
  $("prints").innerHTML = prints.length
    ? prints.map((p) => `<span class="print ${p.side === "Buy" ? "buy" : "sell"}">${px3(p.price)}×${num(p.size)}</span>`).join("")
    : `<span class="health-none">no prints yet</span>`;
}

// ---- fills blotter ----
async function loadFills() {
  try {
    const resp = await api(`/api/fills?mode=${state.mode}&limit=300`);
    state.fills.rows = resp.fills || [];
    renderFills();
    showAuthBanner(false);
  } catch (e) { if (e.message !== "unauthorized") toast("fills: " + e.message, true); }
}
function attrPill(a) {
  if (a === "maker") return `<span class="pill maker">maker</span>`;
  if (a === "late") return `<span class="pill late">late</span>`;
  return `<span class="pill taker">taker</span>`;
}
function fillMarkout(r) {
  if (r.markout_5s === null || r.markout_5s === undefined) {
    return r.markout_pending ? `<span class="mk-pending">…</span>` : `<span class="zero">—</span>`;
  }
  return markoutCell(r.markout_5s);
}
function renderFills() {
  const tb = $("fills-body");
  const sel = state.fills.series;
  const rows = state.fills.rows.filter((r) => !sel || seriesOf(r.window) === sel);
  $("fills-empty").classList.toggle("hidden", rows.length > 0);
  tb.innerHTML = rows.map((r) => {
    const t = r.ts_venue || r.ts_local;
    return "<tr>" +
      `<td class="col-time">${fmtClock(t)}</td>` +
      `<td>${seriesOf(r.window)}</td>` +
      `<td>${r.side}</td>` +
      `<td>${r.outcome}</td>` +
      `<td>${px3(r.price)}</td>` +
      `<td>${num(r.size)}</td>` +
      `<td>${attrPill(r.attribution)}</td>` +
      `<td>${fmtUsd(r.fee, false)}</td>` +
      `<td>${fillMarkout(r)}</td>` +
      "</tr>";
  }).join("");
}

// ---- risk panel ----
const BREAKERS = ["FeedStale", "WsDisconnect", "ClockSkew", "DailyStop", "WindowLoss", "ErrorRate", "FairVsMid", "Manual", "EngineRestart"];
async function loadRisk() {
  try {
    state.risk = await api(`/api/risk?mode=${state.mode}`);
    renderRisk();
    showAuthBanner(false);
  } catch (e) { if (e.message !== "unauthorized") toast("risk: " + e.message, true); }
}
function renderRisk() {
  const r = state.risk;
  if (!r) return;
  const tripped = new Set(r.tripped || []);
  $("breaker-chips").innerHTML = BREAKERS.map((b) => {
    const on = tripped.has(b);
    return `<span class="chip" style="cursor:default"><span class="chip-dot" style="background:${on ? "var(--neg)" : "var(--pos)"}"></span>${b}</span>`;
  }).join("");
  const halt = $("risk-halt");
  const halted = !!(r.snapshot && r.snapshot.globally_halted);
  halt.textContent = halted ? "HALTED" : "running";
  halt.className = "pill " + (halted ? "bad" : "ok");
  $("last-cancel").textContent = r.last_cancel_all ? `last cancel-all: ${r.last_cancel_all}` : "no cancel-all this session";
  const s = r.snapshot;
  $("risk-stats").innerHTML = s ? [
    ["daily PnL", `<span class="${num(s.daily_pnl) >= 0 ? "pos" : "neg"}">${fmtUsd(s.daily_pnl, true)}</span>`],
    ["open notional", `${fmtUsd(s.open_notional, false)} / ${fmtUsd(s.open_notional_cap, false)}`],
    ["errors", s.error_count],
    ["halted windows", s.halted_windows],
  ].map(([k, v]) => `<div class="kv"><label>${k}</label><span>${v}</span></div>`).join("")
    : `<span class="health-none">no risk snapshot</span>`;
  $("ws-state").textContent = r.ws_connected ? "connected" : "—";
  $("ws-badge").classList.toggle("on", !!r.ws_connected);
  $("feed-health").innerHTML = (r.feeds && r.feeds.length)
    ? r.feeds.map((f) => `<div class="health-item bad"><span>${f.source} ${f.asset} ${f.kind}</span><span>stale ${(f.age_ms / 1000).toFixed(1)}s</span></div>`).join("")
    : `<div class="health-item ok"><span>all feeds live</span><span>✓</span></div>`;
  $("model-health").innerHTML = (r.model_health && r.model_health.length)
    ? r.model_health.map((m) => {
        const cls = m.health === "Ready" ? "ok" : m.health === "Degraded" ? "warn" : "bad";
        return `<div class="health-item ${cls}"><span>${m.asset}</span><span>${m.health} (${m.reason})</span></div>`;
      }).join("")
    : `<div class="health-none">no model health (run with --with-model)</div>`;
  $("book-health").innerHTML = (r.books && r.books.length)
    ? r.books.map((b) => `<div class="health-item bad"><span>${seriesOf(b.window)}</span><span>${b.reason}</span></div>`).join("")
    : `<div class="health-item ok"><span>all books reliable</span><span>✓</span></div>`;
}

// debounced live/fills refreshers (high-frequency WS frames)
let liveRefreshTimer = null;
function scheduleLiveRefresh() {
  if (liveRefreshTimer) return;
  liveRefreshTimer = setTimeout(() => {
    liveRefreshTimer = null;
    if (state.view === "live") loadWindowDetail();
  }, 400);
}
let fillsRefreshTimer = null;
function scheduleFillsRefresh() {
  if (fillsRefreshTimer) return;
  fillsRefreshTimer = setTimeout(() => {
    fillsRefreshTimer = null;
    if (state.view === "fills") loadFills();
  }, 800);
}

function refreshActiveView() {
  if (state.view === "series") loadSeries();
  else if (state.view === "live") loadWindows();
  else if (state.view === "fills") loadFills();
  else if (state.view === "risk") loadRisk();
}

// ----------------------------------------------------------------- websocket
let ws = null, wsBackoff = 1000, wsTimer = null;
function setConn(on) {
  const dot = $("conn-dot");
  dot.classList.toggle("on", on);
  dot.title = on ? "WebSocket connected" : "WebSocket disconnected";
}
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const t = getToken();
  const url = `${proto}://${location.host}/api/ws` + (t ? "?token=" + encodeURIComponent(t) : "");
  try { ws = new WebSocket(url); } catch (e) { scheduleReconnect(); return; }
  ws.onopen = () => { setConn(true); wsBackoff = 1000; loadAll(); };
  ws.onmessage = (ev) => { try { onWsMessage(JSON.parse(ev.data)); } catch (e) {} };
  ws.onclose = () => { setConn(false); scheduleReconnect(); };
  ws.onerror = () => { try { ws.close(); } catch (e) {} };
}
function scheduleReconnect() {
  if (wsTimer) return;
  wsTimer = setTimeout(() => { wsTimer = null; connectWS(); }, wsBackoff);
  wsBackoff = Math.min(wsBackoff * 2, 15000);
}
function onWsMessage(msg) {
  switch (msg.type) {
    case "hello": setConn(true); break;
    case "equity": {
      const arr = state.equity[msg.mode] || (state.equity[msg.mode] = []);
      arr.push({ t: msg.ts_ms, v: num(msg.equity) });
      if (arr.length > EQUITY_CAP) arr.shift();
      if (msg.mode === state.mode && state.view === "overview") { drawEquityChart(); renderCapital(); }
      break;
    }
    case "control":
      if (state.overview && state.overview.modes[msg.mode]) {
        state.overview.modes[msg.mode].running = msg.running;
        state.overview.modes[msg.mode].armed = msg.armed;
        renderBadges();
      }
      break;
    case "breaker":
      if (msg.breaker === "Manual") {
        if (msg.event === "tripped") state.killed = true;
        else if (msg.event === "cleared") state.killed = false;
        renderKill();
      }
      if (state.view === "risk") loadRisk();
      break;
    case "fill":
      scheduleRefresh();
      if (state.view === "fills") scheduleFillsRefresh();
      if (state.view === "live") scheduleLiveRefresh();
      break;
    case "lifecycle":
      scheduleRefresh();
      if (state.view === "live") loadWindows(); // the active-window set may have changed
      break;
    case "top":
    case "quote":
    case "model":
      if (state.view === "live") scheduleLiveRefresh();
      break;
    case "resync":
      loadAll();
      break;
    default: break;
  }
}

// ------------------------------------------------------------------ controls
async function postControl(path, body, okMsg) {
  try {
    await api(path, { method: "POST", body });
    toast(okMsg);
    loadControlStatus();
  } catch (e) { toast(e.message, true); }
}

// Polls /api/control/status and renders the arming gates, series toggles, and
// the parameter overrides (the §11 prominent state display).
async function loadControlStatus() {
  let s;
  try { s = await api("/api/control/status"); }
  catch { return; } // 404 until the orchestrator pushes a snapshot.
  // Arming state + gates.
  const a = s.arming || {};
  $("arming-state").textContent =
    a.session_armed ? "ARMED" : a.pending ? "pending confirmation" : "disarmed";
  $("arming-state").className = a.session_armed ? "pos" : a.pending ? "warn-text" : "";
  const gate = (ok, label) =>
    `<span class="gate-pill ${ok ? "on" : "off"}">${ok ? "✓" : "✗"} ${label}</span>`;
  $("arming-gates").innerHTML =
    gate(a.config_enabled, "config") + gate(a.env_confirmed, "env phrase") +
    gate(a.session_armed, "session");
  $("arm-btn").disabled = !a.can_arm || a.session_armed;
  // Series toggles (enabled set from the snapshot).
  const enabled = new Set(s.enabled_series || []);
  const all = ["BTC-5m", "BTC-15m", "BTC-1h", "ETH-5m", "ETH-15m", "ETH-1h"];
  $("series-toggles").innerHTML = all.map((k) =>
    `<button class="seg-btn ${enabled.has(k) ? "active" : ""}" data-series="${k}">${k}</button>`
  ).join("");
  document.querySelectorAll("#series-toggles .seg-btn").forEach((b) =>
    b.addEventListener("click", () => {
      const k = b.dataset.series, on = b.classList.contains("active");
      postControl(on ? "/api/control/disable-series" : "/api/control/enable-series",
        { series: k }, `${k} ${on ? "disabled" : "enabled"}`);
    }));
  // Active parameter overrides.
  const ovr = s.param_overrides || [];
  $("param-overrides").innerHTML = ovr.length
    ? ovr.map((o) => `<span class="ovr">${o.series || "all"}: ${o.key} = ${o.value}</span>`).join("")
    : `<span class="muted">no overrides</span>`;
}

const ARM_PHRASE = "arm-live-i-accept-real-money-losses";

function wireControls() {
  $("kill-btn").addEventListener("click", () =>
    confirmAction("Kill ALL trading? This cancels every open order and halts the bot (latched).",
      () => postControl("/api/control/kill", undefined, "Kill issued")));
  $("reset-btn").addEventListener("click", () =>
    confirmAction("Reset breakers and resume trading?",
      () => postControl("/api/control/reset", undefined, "Reset issued")));
  $("cap-set").addEventListener("click", () => {
    const v = $("cap-input").value.trim();
    if (v === "") { toast("enter an amount", true); return; }
    confirmAction(`Set paper capital to $${v}?`,
      () => postControl("/api/control/paper-capital", { amount: v }, "Capital set"));
  });
  $("cap-plus").addEventListener("click", () =>
    postControl("/api/control/paper-capital", { delta: "1000" }, "+$1k"));
  $("cap-minus").addEventListener("click", () =>
    postControl("/api/control/paper-capital", { delta: "-1000" }, "−$1k"));
  // Risk-panel kill/reset (same endpoints as the overview controls).
  $("risk-kill-btn").addEventListener("click", () =>
    confirmAction("Kill ALL trading? This cancels every open order and halts the bot (latched).",
      () => postControl("/api/control/kill", undefined, "Kill issued")));
  $("risk-reset-btn").addEventListener("click", () =>
    confirmAction("Reset every breaker (including the daily stop-loss) and resume trading?",
      () => postControl("/api/control/reset", undefined, "Reset issued")));
  // Control card: arming flow, daily-stop reset, parameter editor.
  $("reset-daily-btn").addEventListener("click", () =>
    confirmAction("Clear the daily stop-loss latch (leaving a manual kill in place)?",
      () => postControl("/api/control/reset-daily-stop", undefined, "Daily stop cleared")));
  $("arm-btn").addEventListener("click", () =>
    confirmAction("Begin live arming? You will be asked to re-type the confirmation phrase.",
      async () => {
        try { await api("/api/control/arm-live/begin", { method: "POST" }); }
        catch (e) { toast(e.message, true); loadControlStatus(); return; }
        const phrase = window.prompt(
          "Type the live confirmation phrase to ARM live trading:\n(" + ARM_PHRASE + ")");
        if (phrase === null) { loadControlStatus(); return; }
        postControl("/api/control/arm-live/confirm", { phrase }, "Live armed");
      }));
  $("disarm-btn").addEventListener("click", () =>
    postControl("/api/control/disarm", undefined, "Disarmed"));
  $("param-set").addEventListener("click", () => {
    const series = $("param-series").value.trim();
    const key = $("param-key").value.trim();
    const value = $("param-value").value.trim();
    if (key === "" || value === "") { toast("enter a param and value", true); return; }
    postControl("/api/control/set-param",
      { series: series === "" ? null : series, key, value },
      `set ${key} = ${value}`);
  });
  loadControlStatus();
  setInterval(loadControlStatus, 5000);
}

// --------------------------------------------------------------------- wiring
const VIEWS = ["overview", "series", "live", "fills", "risk"];
function switchView(view) {
  state.view = view;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.view === view));
  VIEWS.forEach((v) => $("view-" + v).classList.toggle("hidden", v !== view));
  if (view === "overview") drawEquityChart();
  else refreshActiveView();
}
function switchMode(mode) {
  state.mode = mode;
  document.querySelectorAll("#mode-toggle .seg-btn").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  renderBadges(); renderCapital(); drawEquityChart();
  refreshActiveView();
}

function init() {
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => switchView(t.dataset.view)));
  document.querySelectorAll("#mode-toggle .seg-btn").forEach((b) => b.addEventListener("click", () => switchMode(b.dataset.mode)));
  document.querySelectorAll("#window-toggle .seg-btn").forEach((b) => b.addEventListener("click", () => {
    state.compareWindow = b.dataset.window;
    document.querySelectorAll("#window-toggle .seg-btn").forEach((x) => x.classList.toggle("active", x === b));
    loadSeries();
  }));
  $("token-btn").addEventListener("click", () => {
    const cur = getToken();
    showAuthBanner(true);
    $("token-input").value = cur;
    $("token-input").focus();
  });
  $("token-save").addEventListener("click", () => {
    setToken($("token-input").value.trim());
    showAuthBanner(false);
    if (ws) try { ws.close(); } catch (e) {}
    connectWS(); loadAll();
    toast("token saved");
  });
  $("fills-series").addEventListener("change", (e) => {
    state.fills.series = e.target.value;
    renderFills();
  });
  wireControls();
  window.addEventListener("resize", () => { if (state.view === "overview") drawEquityChart(); });

  // Client-side countdown ticker; fills re-poll to surface matured 5s markouts.
  setInterval(() => { if (state.view === "live") renderCountdown(); }, 1000);
  setInterval(() => { if (state.view === "fills") loadFills(); }, 5000);

  loadParams();
  loadOverview();
  connectWS();
}
document.addEventListener("DOMContentLoaded", init);
