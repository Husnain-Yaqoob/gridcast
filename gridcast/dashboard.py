"""The live dashboard page.

Kept in its own module rather than embedded in `api.py` because a few hundred
lines of HTML wedged into a routing file makes both harder to read.

It is deliberately one self-contained page with no build step, no framework and
no external requests. Three reasons, in order of how much they matter:

- It has to work on a laptop with no internet, in a room, during an interview.
  A CDN that is slow or blocked turns the demo into an apology.
- Every byte it draws comes from this project's own API. Nothing is baked in,
  so if the scheduled ingest stops, the page visibly goes stale rather than
  continuing to show a reassuring picture of yesterday.
- The charts are hand-drawn SVG. A charting library would be faster to write
  and would also mean the interesting decisions — what the error bars mean,
  why persistence is on the same axes — happen inside somebody else's
  defaults.

The page never invents a number. Where the API has no answer, the tile says so.
"""

from __future__ import annotations

_DASHBOARD_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>gridcast — Irish grid, live</title>
<style>
  :root {
    color-scheme: light;
    --surface: #fcfcfb;
    --plane: #f9f9f7;
    --ink: #0b0b0b;
    --ink-2: #52514e;
    --muted: #898781;
    --grid: #e1e0d9;
    --axis: #c3c2b7;
    --border: rgba(11,11,11,0.10);
    --actual: #2a78d6;
    --model: #eb6834;
    --naive: #1baf7a;
    --bad: #e34948;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      color-scheme: dark;
      --surface: #1a1a19;
      --plane: #0d0d0d;
      --ink: #ffffff;
      --ink-2: #c3c2b7;
      --muted: #898781;
      --grid: #2c2c2a;
      --axis: #383835;
      --border: rgba(255,255,255,0.10);
      --actual: #3987e5;
      --model: #d95926;
      --naive: #199e70;
      --bad: #e66767;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px;
    background: var(--plane); color: var(--ink);
    font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  .wrap { max-width: 1060px; margin: 0 auto; }
  header { margin-bottom: 20px; }
  h1 { font-size: 22px; margin: 0 0 4px; letter-spacing: -0.01em; }
  .sub { color: var(--ink-2); font-size: 14px; margin: 0; }
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 18px 20px; margin-bottom: 16px;
  }
  .tiles { display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); }
  .tile { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
  .tile .label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
  .tile .lag { text-transform: none; letter-spacing: 0; font-size: 11px;
               margin-left: 6px; padding: 1px 5px; border-radius: 3px;
               background: var(--grid); color: var(--muted);
               white-space: nowrap; display: inline-block; }
  .tile .value { font-size: 26px; font-weight: 700; margin-top: 4px; letter-spacing: -0.02em;
                 white-space: nowrap; }
  .tile .unit { font-size: 14px; font-weight: 500; color: var(--ink-2); }
  h2 { font-size: 16px; margin: 0 0 2px; }
  .note { color: var(--ink-2); font-size: 13px; margin: 0 0 14px; }
  svg { width: 100%; height: auto; display: block; overflow: visible; }
  .legend { display: flex; gap: 18px; flex-wrap: wrap; margin-top: 10px; font-size: 13px; color: var(--ink-2); }
  .legend span { display: inline-flex; align-items: center; gap: 7px; }
  .swatch { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
  table { border-collapse: collapse; width: 100%; font-size: 14px; font-variant-numeric: tabular-nums; }
  th, td { text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--grid); }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--muted); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; }
  .win { color: var(--actual); font-weight: 600; }
  .lose { color: var(--bad); font-weight: 600; }
  footer { color: var(--muted); font-size: 12px; margin-top: 22px; }
  .stale { color: var(--bad); font-weight: 600; }
  .err { background: var(--surface); border: 1px solid var(--bad); border-radius: 10px; padding: 16px 18px; color: var(--ink); }
  a { color: var(--actual); }
  .scroll { overflow-x: auto; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>gridcast</h1>
    <p class="sub">Live wind generation on the all-island Irish grid, and what this project predicts happens next.</p>
  </header>

  <div id="alert"></div>
  <div class="tiles" id="tiles"></div>

  <div class="card">
    <h2>Next twelve hours</h2>
    <p class="note" id="forecast-note">Recent output, and the forecast at each trained horizon. Shaded bands are the model's validated mean absolute error &mdash; a prediction drawn without one invites more confidence than it has earned.</p>
    <div id="forecast-chart"></div>
    <div class="legend">
      <span><i class="swatch" style="background:var(--actual)"></i>observed</span>
      <span><i class="swatch" style="background:var(--model)"></i>model forecast (&plusmn; validated error)</span>
      <span><i class="swatch" style="background:var(--naive)"></i>persistence baseline</span>
    </div>
  </div>

  <div class="card">
    <h2>Does the model beat doing nothing clever?</h2>
    <p class="note">Skill against persistence &mdash; the naive forecast that says output in <em>h</em> hours will be whatever it is now. Above the line the model earns its complexity; below it, it does not.</p>
    <div id="skill-chart"></div>
    <div class="scroll"><table id="skill-table"></table></div>
  </div>

  <footer>
    <span id="freshness"></span> &middot; Supported by EirGrid Group Data &middot;
    <a href="/docs">API documentation</a>
  </footer>
</div>

<script>
const F = (n, d = 0) => n === null || n === undefined || Number.isNaN(n)
  ? "—" : Number(n).toLocaleString(undefined, {minimumFractionDigits: d, maximumFractionDigits: d});

async function get(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(path + " returned " + response.status);
  return response.json();
}

function el(tag, attrs = {}, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

function tiles(latest) {
  // Carbon intensity and SNSP settle later than wind and demand, so their
  // newest value can be an hour or two behind the headline reading. Showing
  // the number is right; showing it as though it were from the same instant
  // is not. The lag is labelled on the tile that has one.
  const lag = latest.lagging_by_minutes || {};
  const behind = (key) => {
    const mins = lag[key];
    if (!mins) return "";
    const text = mins < 90 ? `${mins}m` : `${Math.round(mins / 60)}h`;
    return `<span class="lag" title="This series settles later than wind and demand">${text} behind</span>`;
  };

  const items = [
    ["Wind now", F(latest.wind_mw), "MW", ""],
    ["Demand", F(latest.demand_mw), "MW", ""],
    ["Wind share", F(latest.wind_share_pct, 1), "%", ""],
    ["Carbon intensity", F(latest.co2_intensity), "gCO₂/kWh", behind("co2_intensity")],
    ["SNSP", F(latest.snsp, 1), "%", behind("snsp")],
  ];
  document.getElementById("tiles").innerHTML = items.map(([label, value, unit, note]) =>
    `<div class="tile"><div class="label">${label}${note}</div>
     <div class="value">${value} <span class="unit">${unit}</span></div></div>`
  ).join("");
}

// ---------------------------------------------------------------- forecast
function drawForecast(history, forecasts, latest) {
  const W = 980, H = 320, L = 62, R = 40, T = 34, B = 34;
  const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Recent wind output with forecasts at each trained horizon"});

  const now = new Date(latest.observed_at_utc).getTime();
  const past = history.map(r => ({t: new Date(r.observed_at_utc).getTime(), v: r.wind_mw}));
  const maxHorizon = forecasts.length ? Math.max(...forecasts.map(f => f.horizon_hours)) : 1;

  const t0 = past.length ? past[0].t : now;
  const t1 = now + maxHorizon * 3600e3;

  // One y-scale across observed, forecasts and error bands. Rescaling the
  // forecast half would make a 790 MW error look the same size as a 105 MW one.
  const values = past.map(p => p.v)
    .concat(forecasts.flatMap(f => [f.predicted_wind_mw + (f.expected_error_mw || 0),
                                    f.predicted_wind_mw - (f.expected_error_mw || 0)]))
    .concat([latest.wind_mw]);
  const lo = Math.min(0, Math.min(...values));
  const hi = Math.max(...values) * 1.08 || 1;

  const x = t => L + (t - t0) / (t1 - t0) * (W - L - R);
  const y = v => H - B - (v - lo) / (hi - lo) * (H - T - B);

  const ticks = 4;
  for (let i = 0; i <= ticks; i++) {
    const v = lo + (hi - lo) * i / ticks;
    svg.appendChild(el("line", {x1: L, x2: W - R, y1: y(v), y2: y(v),
      stroke: "var(--grid)", "stroke-width": 1}));
    svg.appendChild(el("text", {x: L - 10, y: y(v) + 4, "text-anchor": "end",
      fill: "var(--muted)", "font-size": 11}, F(v)));
  }
  // Above the first gridline, not level with it — at T + 2 this label sat on
  // top of the topmost tick value.
  svg.appendChild(el("text", {x: L - 10, y: T - 12, "text-anchor": "end",
    fill: "var(--muted)", "font-size": 11}, "MW"));

  // "Now" divider — everything right of it is prediction, not measurement.
  svg.appendChild(el("line", {x1: x(now), x2: x(now), y1: T, y2: H - B,
    stroke: "var(--axis)", "stroke-width": 1, "stroke-dasharray": "3 3"}));
  svg.appendChild(el("text", {x: x(now) + 5, y: T - 12, fill: "var(--muted)",
    "font-size": 11}, "forecast \u2192"));

  // Labelled relative to now rather than by clock time. On a chart whose
  // whole point is "this half is measured, that half is predicted", hours from
  // now is the axis a reader actually wants — and absolute times repeat
  // confusingly across a 36-hour window.
  for (const hours of [-24, -12, 0, 6, 12]) {
    const t = now + hours * 3600e3;
    if (t < t0 || t > t1) continue;
    const label = hours === 0 ? "now" : (hours > 0 ? "+" : "\u2212") + Math.abs(hours) + "h";
    svg.appendChild(el("text", {x: x(t), y: H - B + 18, "text-anchor": "middle",
      fill: "var(--muted)", "font-size": 11}, label));
  }

  // Error bands first, so the lines sit on top of them.
  for (const f of forecasts) {
    if (!f.expected_error_mw) continue;
    const ft = now + f.horizon_hours * 3600e3;
    svg.appendChild(el("path", {
      d: `M ${x(now)} ${y(latest.wind_mw)} L ${x(ft)} ${y(f.predicted_wind_mw + f.expected_error_mw)}
          L ${x(ft)} ${y(f.predicted_wind_mw - f.expected_error_mw)} Z`,
      fill: "var(--model)", opacity: 0.13}));
  }

  const line = (points, colour, width, dash) => {
    if (points.length < 2) return;
    svg.appendChild(el("path", {
      d: points.map((p, i) => `${i ? "L" : "M"} ${x(p.t)} ${y(p.v)}`).join(" "),
      fill: "none", stroke: colour, "stroke-width": width,
      "stroke-linejoin": "round", "stroke-linecap": "round",
      ...(dash ? {"stroke-dasharray": dash} : {})}));
  };

  line(past, "var(--actual)", 2.2);

  const points = [{t: now, v: latest.wind_mw}]
    .concat(forecasts.map(f => ({t: now + f.horizon_hours * 3600e3, v: f.predicted_wind_mw})));
  line(points, "var(--model)", 2);

  // Persistence last, so it stays visible. Drawn before the model line it
  // vanished underneath wherever the two agreed — which at short horizons is
  // almost everywhere, and is exactly the comparison the chart exists to make.
  line([{t: now, v: latest.wind_mw}, {t: t1, v: latest.wind_mw}], "var(--naive)", 1.8, "5 4");

  for (const f of forecasts) {
    const ft = now + f.horizon_hours * 3600e3;
    svg.appendChild(el("circle", {cx: x(ft), cy: y(f.predicted_wind_mw), r: 4.5,
      fill: "var(--model)", stroke: "var(--surface)", "stroke-width": 2}));
    svg.appendChild(el("title", {}, `${f.horizon_hours}h ahead: ${F(f.predicted_wind_mw)} MW `
      + `± ${F(f.expected_error_mw)} MW`));
  }

  const host = document.getElementById("forecast-chart");
  host.innerHTML = "";
  host.appendChild(svg);
}

// ------------------------------------------------------------------- skill
function drawSkill(horizons) {
  const W = 980, H = 260, L = 56, R = 20, T = 26, B = 42;
  const svg = el("svg", {viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": "Skill against the persistence baseline, by forecast horizon"});

  const skills = horizons.map(h => h.skill_vs_persistence_pct || 0);
  // Limits follow the data rather than being forced symmetric about zero.
  // Symmetric limits looked tidier and wasted half the chart: a +8% bar beside
  // a -48% one left the entire upper half blank and the winning bar too small
  // to read. Zero stays on the axis, which is what actually matters here.
  const hi = Math.max(0, ...skills) * 1.35 + 2;
  const lo = Math.min(0, ...skills) * 1.35 - 2;
  const y = v => T + (hi - v) / (hi - lo) * (H - T - B);
  const bandWidth = (W - L - R) / Math.max(horizons.length, 1);

  svg.appendChild(el("line", {x1: L, x2: W - R, y1: y(0), y2: y(0),
    stroke: "var(--ink-2)", "stroke-width": 1.2}));

  horizons.forEach((h, i) => {
    const skill = h.skill_vs_persistence_pct || 0;
    const cx = L + bandWidth * (i + 0.5);
    const w = Math.min(74, bandWidth * 0.42);
    const top = Math.min(y(skill), y(0));
    const height = Math.max(Math.abs(y(skill) - y(0)), 2);
    // Rounded at the data end, square against the baseline. A rect with `rx`
    // rounds all four corners, which turns a short bar into a pill and detaches
    // it visually from the zero line it is measured against.
    const r = Math.min(4, height, w / 2);
    const up = skill > 0;
    const d = up
      ? `M ${cx - w/2} ${top + height} L ${cx - w/2} ${top + r} Q ${cx - w/2} ${top} ${cx - w/2 + r} ${top}`
        + ` L ${cx + w/2 - r} ${top} Q ${cx + w/2} ${top} ${cx + w/2} ${top + r} L ${cx + w/2} ${top + height} Z`
      : `M ${cx - w/2} ${top} L ${cx - w/2} ${top + height - r} Q ${cx - w/2} ${top + height} ${cx - w/2 + r} ${top + height}`
        + ` L ${cx + w/2 - r} ${top + height} Q ${cx + w/2} ${top + height} ${cx + w/2} ${top + height - r} L ${cx + w/2} ${top} Z`;
    svg.appendChild(el("path", {d, fill: up ? "var(--actual)" : "var(--bad)"}));
    svg.appendChild(el("text", {x: cx, y: skill > 0 ? top - 8 : top + height + 17,
      "text-anchor": "middle", fill: "var(--ink)", "font-size": 13,
      "font-weight": 700}, (skill > 0 ? "+" : "") + skill.toFixed(1) + "%"));
    svg.appendChild(el("text", {x: cx, y: H - 12, "text-anchor": "middle",
      fill: "var(--muted)", "font-size": 12}, h.hours + "h ahead"));
  });

  const host = document.getElementById("skill-chart");
  host.innerHTML = "";
  host.appendChild(svg);

  document.getElementById("skill-table").innerHTML =
    "<thead><tr><th>Horizon</th><th>Model MAE</th><th>Persistence MAE</th>"
    + "<th>Skill</th><th>Verdict</th></tr></thead><tbody>"
    + horizons.map(h => {
        const skill = h.skill_vs_persistence_pct;
        const win = skill > 0;
        const caveat = win && h.beat_baseline_in_every_fold === false
          ? " (not in every fold)" : "";
        return `<tr><td>${h.hours}h</td><td>${F(h.expected_error_mw, 1)} MW</td>`
          + `<td>${F(h.persistence_error_mw, 1)} MW</td>`
          + `<td class="${win ? "win" : "lose"}">${win ? "+" : ""}${F(skill, 1)}%</td>`
          + `<td class="${win ? "win" : "lose"}">${win ? "beats persistence" : "loses to persistence"}${caveat}</td></tr>`;
      }).join("")
    + "</tbody>";
}

// -------------------------------------------------------------------- load
async function load() {
  try {
    const [latest, horizonList, history] = await Promise.all([
      get("/latest"), get("/horizons"), get("/history?hours=36"),
    ]);
    document.getElementById("alert").innerHTML = "";
    tiles(latest);

    const horizons = horizonList.horizons || [];
    const forecasts = [];
    for (const h of horizons) {
      try { forecasts.push(await get("/forecast?horizon=" + h.hours)); }
      catch (e) { /* a horizon can be untrained; the others still draw */ }
    }

    drawForecast(history.readings || [], forecasts, latest);
    drawSkill(horizons);

    const observed = new Date(latest.observed_at_utc);
    const ageMinutes = Math.round((Date.now() - observed.getTime()) / 60000);
    // Matches STALE_AFTER_HOURS in config.py — see the note where the constant
    // is defined. The two are substituted in below rather than written twice,
    // because a dashboard and a CLI disagreeing about whether the same data is
    // stale is worse than either threshold being slightly wrong.
    const stale = ageMinutes > STALE_AFTER_MINUTES;
    const age = ageMinutes < 120 ? `${ageMinutes} min ago`
      : ageMinutes < 2880 ? `${Math.round(ageMinutes / 60)} hours ago`
      : `${Math.round(ageMinutes / 1440)} days ago`;
    document.getElementById("freshness").innerHTML =
      `Last reading ${observed.toISOString().slice(0, 16).replace("T", " ")} UTC `
      + `(<span class="${stale ? "stale" : ""}">${age}${stale ? " — the hourly ingest may have stopped" : ""}</span>)`;
  } catch (error) {
    document.getElementById("alert").innerHTML =
      `<div class="err"><strong>Could not load data.</strong> ${error.message}<br>`
      + `The service is running, but it has nothing to show. Check <code>python -m gridcast status</code>, `
      + `and that models are trained with <code>python -m gridcast train --save</code>.</div>`;
  }
}

load();
// The source publishes every fifteen minutes; refreshing faster than that only
// re-fetches numbers that have not changed.
setInterval(load, 5 * 60 * 1000);
</script>
</body>
</html>
"""


# The freshness threshold is defined once, in config, and substituted here.
#
# It was hardcoded at 120 minutes while `gridcast status` used six hours, so
# the same database could be "stale" on the dashboard and healthy on the
# command line at the same moment. Two hours is also too tight for this
# pipeline: the scheduled job runs hourly and EirGrid settles a period about
# half an hour after it ends, so a single missed run was enough to put a red
# warning on the page during normal operation.
from .config import STALE_AFTER_HOURS   # noqa: E402

DASHBOARD_HTML = _DASHBOARD_TEMPLATE.replace(
    "STALE_AFTER_MINUTES", str(STALE_AFTER_HOURS * 60)
)
