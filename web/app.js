// ModelRouter UI. Everything shown is computed from files in data/ (real recorded runs) or from the live gateway.
const $ = (s) => document.querySelector(s);
const fmt = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x) ? "n/a" : Number(x).toFixed(d));
const pct = (x, d = 0) => (x === null || x === undefined || Number.isNaN(x) ? "n/a" : (100 * x).toFixed(d) + "%");
const usd = (x) => "$" + (x < 1 ? x.toFixed(3) : x.toFixed(2));
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const ease = (u) => u * u * (3 - 2 * u);
let LIVE = false;

async function getJSON(path) {
  const r = await fetch(path, { cache: "no-cache" });
  if (!r.ok) throw new Error(path + " " + r.status);
  return r.json();
}
async function detectBackend() {
  if (document.querySelector('meta[name="mr-mode"]')?.content === "static") return false; // docs/ build: no backend
  try {
    const ctl = new AbortController();
    const t = setTimeout(() => ctl.abort(), 1500);
    const r = await fetch("health", { signal: ctl.signal });
    clearTimeout(t);
    if (!r.ok) return false;
    const j = await r.json();
    return j.status === "ok";
  } catch { return false; }
}
function colors() {
  const cs = getComputedStyle(document.documentElement);
  const g = (n) => cs.getPropertyValue(n).trim();
  return { bone: g("--bone"), lime: g("--lime"), ember: g("--ember"), dim: g("--dim"), line: g("--line-2"), bg: g("--canvas-bg"), mute: g("--mute") };
}
function withAlpha(hex, a) {
  const m = /^#([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return hex;
  const n = parseInt(m[1], 16);
  return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
}

/* ---------- eval set with prefix sums: slider -> exact answer in O(log n) ---------- */
class EvalSet {
  constructor(d) {
    this.d = d; this.n = d.n;
    this.p = Float32Array.from(d.p); this.sq = Float32Array.from(d.sq); this.bq = Float32Array.from(d.bq);
    this.sc = Float32Array.from(d.sc); this.bc = Float32Array.from(d.bc);
    const idx = Array.from({ length: this.n }, (_, i) => i).sort((a, b) => this.p[b] - this.p[a]);
    this.order = Int32Array.from(idx);
    this.ps = new Float32Array(this.n);
    this.cumQ = new Float64Array(this.n + 1); this.cumC = new Float64Array(this.n + 1);
    for (let k = 0; k < this.n; k++) {
      const i = idx[k]; this.ps[k] = this.p[i];
      this.cumQ[k + 1] = this.cumQ[k] + (this.bq[i] - this.sq[i]);
      this.cumC[k + 1] = this.cumC[k] + (this.bc[i] - this.sc[i]);
    }
    this.S = this.sq.reduce((a, b) => a + b, 0); this.SC = this.sc.reduce((a, b) => a + b, 0);
    this.qs = this.S / this.n; this.qb = (this.S + this.cumQ[this.n]) / this.n;
    this.cs = this.SC / this.n; this.cb = (this.SC + this.cumC[this.n]) / this.n;
  }
  countGE(t) { let lo = 0, hi = this.n; while (lo < hi) { const m = (lo + hi) >> 1; if (this.ps[m] >= t) lo = m + 1; else hi = m; } return lo; }
  thresholdForK(k) { return k <= 0 ? 1.0001 : this.ps[Math.min(k, this.n) - 1]; }
  at(t) {
    const k = this.countGE(t);
    return { k, frac: k / this.n, quality: (this.S + this.cumQ[k]) / this.n, cost: (this.SC + this.cumC[k]) / this.n };
  }
  quantileSlider(v) { // slider 0..1000: left = send everything to big, right = send nothing
    const k = Math.round((1 - v / 1000) * this.n);
    const t = this.thresholdForK(k);
    return t;
  }
  sliderForT(t) { return Math.round((1 - this.countGE(t) / this.n) * 1000); }
}

/* ---------- state ---------- */
const S = { threshold: 0.5, ev: null, off: null, live: null, theme: "dark" };

/* ---------- boot ---------- */
async function boot() {
  try { S.theme = localStorage.getItem("mr-theme") || "dark"; } catch { /* storage blocked */ }
  document.documentElement.dataset.theme = S.theme;
  $("#theme-btn").addEventListener("click", () => {
    S.theme = S.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = S.theme;
    try { localStorage.setItem("mr-theme", S.theme); } catch { /* ignore */ }
    stream.recolor(); drawAllCharts();
  });
  const [backend, ev, off] = await Promise.all([detectBackend(), getJSON("data/eval.json"), getJSON("data/offline.json")]);
  LIVE = backend; S.off = off; S.ev = new EvalSet(ev);
  try { S.live = await getJSON("data/live.json"); } catch { S.live = null; }
  const badge = $("#mode-badge");
  badge.textContent = LIVE ? "live gateway" : "recorded run";
  badge.classList.toggle("live", LIVE);
  badge.title = LIVE ? "Connected to the running FastAPI gateway" : "Static demo: replaying results saved from a real run of this code";
  $("#pairline").innerHTML = `<span>small <b>${off.small}</b></span><span>big <b>${off.big}</b></span><span>held-out prompts <b>${S.ev.n.toLocaleString()}</b></span><span>classifier <b>${off.selected.model}</b> (target ${off.selected.target})</span>`;
  $("#foot-data").textContent = `Data: RouterBench (Hu et al. 2024), recorded costs at late-2023 prices.`;
  const t0 = off.test.cpt95 !== undefined ? S.ev.thresholdForK(Math.round(off.test.cpt95 * S.ev.n)) : 0.5;
  S.threshold = t0;
  $("#thr").value = S.ev.sliderForT(t0);
  initThreshold(); stream.init(); initTables(); initPlayground(); drawAllCharts();
  const ro = new ResizeObserver(() => { stream.resize(); drawAllCharts(); });
  ro.observe($("#stream")); ro.observe($("#curve")); ro.observe($("#live-curve"));
  setupFades();
  update();
}

/* tables scroll sideways inside their own container; an edge fade shows there is more */
function setupFades() {
  document.querySelectorAll(".tablewrap").forEach((w) => {
    const f = () => { w.classList.toggle("fade-l", w.scrollLeft > 4); w.classList.toggle("fade-r", w.scrollLeft + w.clientWidth < w.scrollWidth - 4); };
    w.addEventListener("scroll", f, { passive: true }); new ResizeObserver(f).observe(w); f();
  });
}

/* ---------- threshold control ---------- */
function initThreshold() {
  const r = $("#thr");
  r.addEventListener("input", () => setThreshold(S.ev.quantileSlider(+r.value), false));
  document.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => {
    const v = b.dataset.preset;
    let k;
    if (v === "cpt95") k = Math.round(S.off.test.cpt95 * S.ev.n);
    else k = Math.round(parseFloat(v) * S.ev.n);
    setThreshold(S.ev.thresholdForK(k), true);
  }));
}
function setThreshold(t, syncSlider) {
  S.threshold = t;
  if (syncSlider) $("#thr").value = S.ev.sliderForT(t);
  update();
}
function update() {
  const r = $("#thr");
  r.style.setProperty("--fill", ((+r.value) / 10) + "%");
  const a = S.ev.at(S.threshold);
  $("#thr-out").textContent = a.k === 0 ? "none to big" : "t = " + S.threshold.toFixed(3);
  $("#c-frac").innerHTML = pct(a.frac, 1);
  $("#c-qual").innerHTML = fmt(a.quality, 3) + `<small>${pct(a.quality / S.ev.qb, 1)} of big</small>`;
  const rnd = S.ev.qs + a.frac * (S.ev.qb - S.ev.qs);
  const adv = a.quality - rnd;
  $("#c-adv").textContent = (adv >= 0 ? "+" : "") + fmt(adv, 3);
  $("#c-cost").innerHTML = pct(a.cost / S.ev.cb, 1) + `<small>${usd(a.cost * 1000)} /1k</small>`;
  $("#curve-note").textContent = `At this threshold the router recovers ${pct((a.quality - S.ev.qs) / (S.ev.qb - S.ev.qs), 0)} of the quality gap between always-small and always-big, using ${pct(a.frac, 0)} of big-model calls. Random routing with the same share would score ${fmt(rnd, 3)}.`;
  stream.setThreshold(S.threshold);
  drawAllCharts();
}

/* ---------- stream canvas ----------
   A seeded, continuous flow of real held-out prompts. Each dot enters from the left, slides to its predicted
   difficulty on the router column, pauses there, then travels to the lane the threshold picks. On arrival it is
   coloured from the recorded outcomes of both models:
     lime   routed right      sent small and the small model was right
     bone   big needed        sent big and the big model did better (or both failed)
     ember  misroute          sent small where only the big model was right
     ring   wasted spend      sent big where the small model was already right */
const stream = (() => {
  const SPEEDS = [1, 4, 16], RATE = 13, TRAVEL = 0.42, MAX_PARTS = 420;
  let cv, ctx, W = 0, H = 0, dpr = 1, C = colors(), playing = true, speed = 1, visible = true, raf = 0, last = 0;
  let parts = [], acc = 0, order = [], cursor = 0, feed = [], boxFlash = [0, 0], thr = 0.5, seed = 12345, pmin = 0, pmax = 1;
  const tot = { n: 0, small: 0, big: 0, cost: 0, bigCost: 0, q: 0, bigQ: 0, miss: 0, waste: 0 };
  const rand = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 4294967296);
  const geo = () => ({ RX: W * 0.4, top: H * 0.12, bot: H * 0.88, SM: [W * 0.85, H * 0.2], LG: [W * 0.85, H * 0.8] });
  const yOf = (p, g) => g.top + clamp((p - pmin) / (pmax - pmin || 1), 0, 1) * (g.bot - g.top);
  function bez(a, u) { const v = 1 - u; return v * v * v * a[0] + 3 * v * v * u * a[1] + 3 * v * u * u * a[2] + u * u * u * a[3]; }
  function laneXY(g, y0, lane, u, k) { // u: 0..1 along the lane path from the router column
    const B = lane === 1 ? g.LG : g.SM;
    return [bez([g.RX, g.RX + 60 * k, B[0] - 90 * k, B[0] - 44 * k], u), bez([y0, y0, B[1], B[1]], u)];
  }
  function outcome(i, big) {
    const e = S.ev, s = e.sq[i], b = e.bq[i];
    if (!big) return s >= 0.5 && s >= b ? "ok" : s < b ? "miss" : "fail";
    return s >= 0.5 && s >= b ? "waste" : "need";
  }
  function init() {
    cv = $("#stream"); ctx = cv.getContext("2d");
    const n = S.ev.n; order = Array.from({ length: n }, (_, i) => i);
    let lo = 1, hi = 0; for (let i = 0; i < n; i++) { lo = Math.min(lo, S.ev.p[i]); hi = Math.max(hi, S.ev.p[i]); }
    pmin = lo; pmax = hi;
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
    $("#play-btn").addEventListener("click", () => { playing = !playing; sync(); });
    document.querySelectorAll("[data-speed]").forEach((b) => b.addEventListener("click", () => {
      speed = +b.dataset.speed;
      document.querySelectorAll("[data-speed]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    }));
    $("#reset-btn").addEventListener("click", reset);
    new IntersectionObserver((es) => { visible = es[0].isIntersecting; loop(); }, { threshold: 0.05 }).observe(cv);
    document.addEventListener("visibilitychange", () => { if (!document.hidden) loop(); });
    if (reduceMotion) {
      playing = false; $("#play-btn").disabled = true;
      document.querySelectorAll("[data-speed]").forEach((b) => (b.disabled = true));
      staticFill();
    }
    thr = S.threshold;
    sync(); resize();
    if (!reduceMotion) for (let k = 0; k < 70; k++) advance(0.05); // start mid-flow, not empty
    loop();
  }
  function sync() { const b = $("#play-btn"); b.textContent = playing ? "Pause" : "Play"; b.setAttribute("aria-pressed", String(playing)); loop(); }
  function resize() {
    if (!cv) return;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    const r = cv.getBoundingClientRect(); W = r.width; H = r.height;
    cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }
  function recolor() { C = colors(); draw(); }
  function clearTotals() { Object.keys(tot).forEach((k) => (tot[k] = 0)); }
  function reset() { clearTotals(); parts = []; feed = []; meter(); if (reduceMotion) staticFill(); draw(); }
  function staticFill() { // reduced motion: the settled state, the whole held-out set routed at the current threshold
    clearTotals(); feed = [];
    for (let i = 0; i < S.ev.n; i++) land(i, S.ev.p[i] >= thr, true);
    meter();
  }
  function spawn() {
    const i = order[cursor++ % order.length];
    parts.push({ i, p: 0, y0: H * (0.1 + 0.8 * rand()), big: false, lane: -1, yR: 0, blend: 1, from: null, oc: "" });
  }
  function land(i, big, silent) {
    const e = S.ev, oc = outcome(i, big);
    tot.n++; big ? tot.big++ : tot.small++;
    tot.cost += big ? e.bc[i] : e.sc[i]; tot.q += big ? e.bq[i] : e.sq[i];
    tot.bigCost += e.bc[i]; tot.bigQ += e.bq[i];
    if (oc === "miss") tot.miss++; if (oc === "waste") tot.waste++;
    if (!silent) {
      boxFlash[big ? 1 : 0] = 1;
      const fam = e.d.families[e.d.fam[i]];
      const tag = oc === "miss" ? "  misroute" : oc === "waste" ? "  wasted" : "";
      feed.unshift({ t: `${fam}  d=${e.p[i].toFixed(2)} > ${big ? "big" : "small"}${tag}`, bad: oc === "miss" || oc === "waste" });
      feed.length = Math.min(feed.length, 3);
    }
    return oc;
  }
  function meter() {
    const per = tot.n || 1;
    const big = tot.bigCost / per * 1000, routed = tot.cost / per * 1000;
    $("#bar-big").style.width = tot.n ? "100%" : "0%";
    $("#bar-routed").style.width = tot.n ? clamp(routed / big, 0, 1) * 100 + "%" : "0%";
    $("#val-big").textContent = tot.n ? usd(big) : "$0"; $("#val-routed").textContent = tot.n ? usd(routed) : "$0";
    $("#st-save").textContent = tot.n ? pct(1 - routed / big, 0) : "0%";
    $("#st-qual").textContent = tot.n ? pct(tot.q / tot.bigQ, 1) : "0%";
    $("#st-big").textContent = tot.n ? pct(tot.big / tot.n, 0) : "0%";
    $("#st-miss").textContent = tot.miss;
    $("#stream-count").textContent = tot.n.toLocaleString() + " requests";
  }
  let meterDirty = false;
  function setThreshold(t) {
    thr = t;
    if (!S.ev || !cv) return;
    if (reduceMotion) { staticFill(); draw(); return; }
    clearTotals(); meterDirty = true;     // the meter restarts so it always reflects the current threshold
    const g = geo(), k = clamp(W / 700, 0.62, 1.15);
    for (const q of parts) { // dots already on their way to a lane are re-routed from where they are
      if (q.lane < 0) continue;
      const lane = S.ev.p[q.i] >= thr ? 1 : 0;
      if (lane !== q.lane && q.p < 0.9) {
        const u = ease(clamp((q.p - 0.55) / 0.45, 0, 1));
        q.from = laneXY(g, q.yR, q.lane, u, k); q.blend = 0; q.lane = lane; q.big = lane === 1;
      }
    }
  }
  function loop() {
    cancelAnimationFrame(raf);
    if (!playing || !visible || document.hidden || reduceMotion) { draw(); return; }
    last = performance.now();
    const step = (now) => {
      const dt = Math.min(0.05, (now - last) / 1000); last = now;
      advance(dt); draw();
      if (playing && visible && !document.hidden) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
  }
  function advance(dt) {
    const sdt = dt * speed;
    acc += sdt * RATE;
    while (acc >= 1) { acc -= 1; if (parts.length < MAX_PARTS) spawn(); }
    const g = geo();
    for (const q of parts) {
      const was = q.p; q.p += sdt * TRAVEL;
      if (was < 0.55 && q.p >= 0.55) { q.big = S.ev.p[q.i] >= thr; q.lane = q.big ? 1 : 0; q.yR = yOf(S.ev.p[q.i], g); q.oc = outcome(q.i, q.big); }
      if (q.blend < 1) q.blend = Math.min(1, q.blend + dt * 4);
    }
    let landed = false;
    parts = parts.filter((q) => { if (q.p >= 1) { land(q.i, q.big); landed = true; return false; } return true; });
    if (landed || meterDirty) { meter(); meterDirty = false; }
    boxFlash[0] = Math.max(0, boxFlash[0] - dt * 3); boxFlash[1] = Math.max(0, boxFlash[1] - dt * 3);
  }
  function fit(text, maxW) { // truncate with an ellipsis so log lines never clip
    if (ctx.measureText(text).width <= maxW) return text;
    while (text.length > 4 && ctx.measureText(text + "...").width > maxW) text = text.slice(0, -1);
    return text + "...";
  }
  const OC = (oc) => (oc === "ok" ? C.lime : oc === "miss" ? C.ember : oc === "waste" ? C.ember : C.bone);
  function dot(x, y, r, oc, a) {
    ctx.globalAlpha = a;
    if (oc === "waste") { ctx.strokeStyle = C.ember; ctx.lineWidth = 1.6; ctx.beginPath(); ctx.arc(x, y, r + 0.8, 0, 6.2832); ctx.stroke(); }
    else { ctx.fillStyle = OC(oc); ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.fill(); }
    ctx.globalAlpha = 1;
  }
  function draw() {
    if (!ctx || !W) return;
    const g = geo(), k = clamp(W / 700, 0.62, 1.15), { RX, top, bot, SM, LG } = g;
    ctx.clearRect(0, 0, W, H);
    // lane guides
    ctx.setLineDash([3 * k, 5 * k]); ctx.lineWidth = 1; ctx.strokeStyle = withAlpha(C.bone, 0.12);
    for (const [B, y0] of [[SM, top], [LG, bot]]) { ctx.beginPath(); ctx.moveTo(RX, y0); ctx.bezierCurveTo(RX + 60 * k, y0, B[0] - 90 * k, B[1], B[0] - 44 * k, B[1]); ctx.stroke(); }
    ctx.setLineDash([]);
    // router column with the difficulty axis and the threshold line
    ctx.strokeStyle = withAlpha(C.bone, 0.22); ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(RX, top - 8); ctx.lineTo(RX, bot + 8); ctx.stroke();
    const ty = yOf(thr, g);
    ctx.strokeStyle = C.lime; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(RX - 22 * k, ty); ctx.lineTo(RX + 22 * k, ty); ctx.stroke();
    ctx.fillStyle = C.lime; ctx.beginPath(); ctx.moveTo(RX + 22 * k, ty); ctx.lineTo(RX + 30 * k, ty - 4); ctx.lineTo(RX + 30 * k, ty + 4); ctx.fill();
    const mono = (px) => `${Math.round(px * k + 1)}px "JetBrains Mono", monospace`;
    ctx.font = mono(9); ctx.textAlign = "center"; ctx.fillStyle = C.dim;
    ctx.fillText("ROUTER", RX, top - 14); ctx.fillText("easy", RX - 28 * k, top + 4); ctx.fillText("hard", RX - 28 * k, bot + 4);
    ctx.textAlign = "left"; ctx.fillStyle = C.lime; ctx.fillText("t=" + thr.toFixed(2), RX + 34 * k, ty + 4);
    // dots
    for (const q of parts) {
      const d = S.ev.p[q.i], r = (2.1 + d * 2.6) * k;
      let x, y, oc = "", a = 0.85;
      if (q.p < 0.3) { const u = ease(q.p / 0.3); x = -6 + (RX + 6) * u; y = q.y0 + (yOf(d, g) - q.y0) * u; a = 0.45 + 0.4 * u; }
      else if (q.p < 0.55) { x = RX; y = yOf(d, g); const w = (q.p - 0.3) / 0.25; ctx.fillStyle = withAlpha(C.lime, 0.2 * Math.sin(w * Math.PI)); ctx.beginPath(); ctx.arc(x, y, r + 5 * k, 0, 6.2832); ctx.fill(); }
      else {
        const u = ease((q.p - 0.55) / 0.45);
        [x, y] = laneXY(g, q.yR, q.lane, u, k);
        if (q.blend < 1 && q.from) { const b = ease(q.blend); x = q.from[0] + (x - q.from[0]) * b; y = q.from[1] + (y - q.from[1]) * b; }
        if (u > 0.7) oc = q.oc; // the outcome shows on arrival
      }
      dot(x, y, r, oc || "need", oc ? 1 : a);
      if (!oc) { /* neutral until it arrives: bone */ }
    }
    // lane boxes
    const boxes = [[SM, "SMALL", S.off.small, tot.small, C.lime, 0], [LG, "BIG", S.off.big, tot.big, C.bone, 1]];
    for (const [B, lab, model, n, col, bi] of boxes) {
      const bw = 88 * k, bh = 46 * k;
      ctx.fillStyle = withAlpha(col, 0.08 + boxFlash[bi] * 0.3); ctx.beginPath(); ctx.roundRect(B[0] - bw / 2, B[1] - bh / 2, bw, bh, 9 * k); ctx.fill();
      ctx.strokeStyle = col; ctx.lineWidth = 1.2; ctx.stroke();
      ctx.textAlign = "center"; ctx.fillStyle = C.bone; ctx.font = `600 ${Math.round(15 * k)}px "Inter Tight", sans-serif`; ctx.fillText(n.toLocaleString(), B[0], B[1] + 5 * k);
      ctx.font = mono(9); ctx.fillStyle = C.dim;
      ctx.fillText(lab, B[0], B[1] + bh / 2 + 13 * k); ctx.fillText(model.slice(0, 22), B[0], B[1] + bh / 2 + 25 * k);
    }
    // log, bottom left, never wider than the canvas
    ctx.textAlign = "left"; ctx.font = mono(9.5);
    feed.forEach((f, i) => { ctx.fillStyle = f.bad ? C.ember : C.dim; ctx.globalAlpha = clamp(1 - i * 0.3, 0.3, 1); ctx.fillText(fit(f.t, W - 24), 12, H - 12 - i * 15 * k); });
    ctx.globalAlpha = 1;
  }
  return { init, resize, recolor, setThreshold, draw, tot };
})();

/* ---------- charts ---------- */
function drawCurve(canvas, cfg) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2), r = canvas.getBoundingClientRect();
  if (!r.width) return;
  canvas.width = Math.round(r.width * dpr); canvas.height = Math.round(r.height * dpr);
  const g = canvas.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const C = colors(), W = r.width, H = r.height, L = 46, R = 12, T = 14, B = 34;
  const { curve, band, oracle, ymin, ymax, marker } = cfg;
  const X = (f) => L + f * (W - L - R), Y = (q) => T + (1 - (q - ymin) / (ymax - ymin)) * (H - T - B);
  g.clearRect(0, 0, W, H);
  g.font = '10.5px "JetBrains Mono", monospace'; g.fillStyle = C.dim; g.strokeStyle = withAlpha(C.bone, 0.1); g.lineWidth = 1;
  const ticks = 4;
  for (let i = 0; i <= ticks; i++) {
    const q = ymin + ((ymax - ymin) * i) / ticks, y = Y(q);
    g.beginPath(); g.moveTo(L, y); g.lineTo(W - R, y); g.stroke(); g.textAlign = "right"; g.fillText(q.toFixed(2), L - 6, y + 3.5);
  }
  g.textAlign = "center";
  for (let i = 0; i <= 4; i++) { g.fillText(i * 25 + "%", X(i / 4), H - 12); }
  g.fillText("share of traffic sent to the big model", L + (W - L - R) / 2, H - 0.5);
  if (band) {
    g.fillStyle = withAlpha(C.bone, 0.14); g.beginPath();
    band.frac.forEach((f, i) => (i ? g.lineTo(X(f), Y(band.hi[i])) : g.moveTo(X(f), Y(band.hi[i]))));
    for (let i = band.frac.length - 1; i >= 0; i--) g.lineTo(X(band.frac[i]), Y(band.lo[i]));
    g.closePath(); g.fill();
  }
  if (oracle) {
    g.setLineDash([4, 4]); g.strokeStyle = withAlpha(C.bone, 0.45); g.lineWidth = 1.2; g.beginPath();
    oracle.frac_big.forEach((f, i) => (i ? g.lineTo(X(f), Y(oracle.quality[i])) : g.moveTo(X(f), Y(oracle.quality[i])))); g.stroke(); g.setLineDash([]);
  }
  g.strokeStyle = C.lime; g.lineWidth = 2.2; g.lineJoin = "round"; g.beginPath();
  curve.frac_big.forEach((f, i) => (i ? g.lineTo(X(f), Y(curve.quality[i])) : g.moveTo(X(f), Y(curve.quality[i])))); g.stroke();
  g.fillStyle = C.bone; for (const [f, q] of [[0, curve.quality[0]], [1, curve.quality[curve.quality.length - 1]]]) { g.beginPath(); g.arc(X(f), Y(q), 4, 0, 6.2832); g.fill(); }
  g.textAlign = "left"; g.fillStyle = C.mute; g.fillText("all small", X(0) + 8, Y(curve.quality[0]) + 14);
  g.textAlign = "right"; g.fillText("all big", X(1) - 8, Y(curve.quality[curve.quality.length - 1]) + 14);
  if (marker) {
    g.strokeStyle = withAlpha(C.lime, 0.5); g.lineWidth = 1; g.beginPath(); g.moveTo(X(marker.frac), T); g.lineTo(X(marker.frac), H - B); g.stroke();
    g.fillStyle = C.lime; g.beginPath(); g.arc(X(marker.frac), Y(marker.quality), 7, 0, 6.2832); g.fill();
    g.strokeStyle = C.bone; g.lineWidth = 2; g.stroke();
  }
  canvas._map = { L, R, W };
}
function drawAllCharts() {
  if (!S.ev || !S.off) return;
  const o = S.off, e = S.ev, a = e.at(S.threshold);
  const qs = [...o.test_oracle_curve.quality, ...o.test_random_band.lo];
  drawCurve($("#curve"), { curve: o.test_curve, band: o.test_random_band, oracle: o.test_oracle_curve,
    ymin: Math.floor((Math.min(...qs) - 0.01) * 20) / 20, ymax: Math.ceil((Math.max(...qs) + 0.01) * 20) / 20, marker: a });
  if (S.live) {
    const L = S.live, rc = L.router_cv.curve;
    const lo = Math.min(...rc.quality, ...L.random_band.lo), hi = Math.max(...rc.quality, ...L.random_band.hi);
    drawCurve($("#live-curve"), { curve: rc, band: L.random_band, ymin: Math.floor((lo - 0.02) * 20) / 20, ymax: Math.ceil((hi + 0.02) * 20) / 20 });
  }
}
function chartPointer(canvas) {
  const setFrom = (ev) => {
    const m = canvas._map; if (!m) return;
    const r = canvas.getBoundingClientRect();
    const f = clamp((ev.clientX - r.left - m.L) / (m.W - m.L - m.R), 0, 1);
    setThreshold(S.ev.thresholdForK(Math.round(f * S.ev.n)), true);
  };
  let down = false;
  canvas.addEventListener("pointerdown", (e) => { down = true; canvas.setPointerCapture(e.pointerId); setFrom(e); });
  canvas.addEventListener("pointermove", (e) => { if (down) setFrom(e); });
  canvas.addEventListener("pointerup", () => (down = false));
  canvas.addEventListener("pointercancel", () => (down = false));
}

/* ---------- tables ---------- */
function row(cells, cls = "") { return `<tr class="${cls}">` + cells.map((c) => `<td${c && c.cls ? ` class="${c.cls}"` : ""}>${c && c.v !== undefined ? c.v : c}</td>`).join("") + "</tr>"; }
function head(cols) { return "<thead><tr>" + cols.map((c) => `<th>${c}</th>`).join("") + "</tr></thead>"; }
function initTables() {
  chartPointer($("#curve"));
  const o = S.off, t = o.test;
  $("#metrics-table").innerHTML = head(["Held-out test", "value"]) + "<tbody>" +
    row(["AUC (classifier)", fmt(t.auc)]) + row(["APGR", fmt(t.apgr)]) + row(["CPT 90% / 95% / 99%", `${pct(t.cpt90)} / ${pct(t.cpt95)} / ${pct(t.cpt99)}`]) +
    row(["AIQ (random = " + fmt(t.aiq_random, 3) + ")", fmt(t.aiq, 3)]) + row(["Oracle CPT 95%", pct(o.test_oracle.cpt95)]) + "</tbody>";
  const adv = o.test_advantage_vs_random;
  // CV table
  const sel = o.selected, rows = [];
  for (const [k, m] of Object.entries(o.cv)) {
    const [tg, kind] = k.split("/"); if (tg !== sel.target) continue;
    rows.push(row([kind.replace("_", " "), fmt(m.auc), fmt(m.brier), fmt(m.ece), fmt(m.apgr), pct(m.cpt95)], kind === sel.model ? "sel" : ""));
  }
  $("#cv-table").innerHTML = head(["model", "AUC", "Brier", "ECE", "APGR", "CPT95"]) + "<tbody>" + rows.join("") + "</tbody>";
  // leave one family out
  const lf = o.leave_family_out, byf = o.test_by_family, lrows = [];
  for (const [f, m] of Object.entries(lf)) {
    const a30 = m.advantage_vs_random_30pct, sig = a30.lo > 0 ? "good" : a30.hi < 0 ? "bad" : "";
    lrows.push(row([f, m.n, m.auc === undefined ? "n/a" : fmt(m.auc), fmt(m.apgr),
      { v: `${a30.mean >= 0 ? "+" : ""}${fmt(a30.mean, 3)} [${fmt(a30.lo, 3)}, ${fmt(a30.hi, 3)}]`, cls: sig },
      byf[f] && byf[f].within_family_auc !== undefined ? fmt(byf[f].within_family_auc) : "n/a"]));
  }
  $("#lofo-table").innerHTML = head(["family", "n", "AUC", "APGR", "vs random @30%", "in-dist AUC"]) + "<tbody>" + lrows.join("") + "</tbody>";
  // honest finding, written from the numbers
  const fams = Object.values(lf), nSig = fams.filter((m) => m.advantage_vs_random_30pct.lo > 0).length;
  const meanLofo = fams.reduce((a, m) => a + m.apgr, 0) / fams.length;
  const fp = o.test_family_prior;
  $("#finding").innerHTML = `<b>Reading this honestly.</b> On the held-out test set the router reaches APGR ${fmt(t.apgr, 2)}, while a router that only knows which benchmark a prompt came from reaches ${fmt(fp.apgr, 2)}. ` +
    `When a whole benchmark family is held out, mean APGR falls to ${fmt(meanLofo, 2)} and the router beats random routing with a 95% interval above zero in ${nSig} of ${fams.length} families. ` +
    `Part of the in-distribution skill is therefore recognising the kind of task, not only how hard an item is; the table shows where it transfers and where it does not.`;
  // live experiment table
  if (S.live) {
    const L = S.live, rows2 = [];
    const pick = (n, name) => L[n] && rows2.push(row([name, fmt(L[n].auc), fmt(L[n].apgr), pct(L[n].cpt95),
      `${fmt(L[n].advantage_vs_random["30%"].mean, 3)} [${fmt(L[n].advantage_vs_random["30%"].lo, 3)}, ${fmt(L[n].advantage_vs_random["30%"].hi, 3)}]`]));
    pick("router_cv", "router (5-fold CV)"); pick("source_prior", "benchmark identity only"); pick("routerbench_transfer", "RouterBench router, transferred");
    $("#live-table").innerHTML = head(["router", "AUC", "APGR", "CPT95", "vs random @30%"]) + "<tbody>" + rows2.join("") + "</tbody>";
    const run = L.run || {};
    $("#live-note").textContent = `${L.n} multiple-choice questions on ${run.gpu || run.device || "one device"} (${run.dtype || "?"}), one request at a time. ${L.small.split("/")[1]} accuracy ${pct(L.accuracy.small, 1)}, ${L.big.split("/")[1]} ${pct(L.accuracy.big, 1)}, oracle ${pct(L.accuracy.oracle, 1)}. Mean latency ${fmt(L.mean_latency_s.small, 2)} s vs ${fmt(L.mean_latency_s.big, 2)} s per question; these are ${run.device === "cuda" ? "GPU" : "CPU"} figures, measured while the machine was shared with other jobs. The cost proxy is compute (parameters x tokens), not dollars.`;
  } else $("#live-sec").hidden = true;
}

/* ---------- playground ---------- */
const EXAMPLES = ["What is the capital of France?", "Write a Python function that merges overlapping intervals and explain its complexity.",
  "A train leaves at 3:15 pm travelling 84 km/h; another leaves the same station at 4:00 pm at 112 km/h. When does the second catch up?",
  "Summarise the causes of the 2008 financial crisis in two sentences.", "Which of the following is a prime number? A. 21 B. 27 C. 31 D. 33"];
function showRoute(d, extra = "") {
  const t = d.threshold ?? S.threshold;
  $("#gauge-thr").style.left = clamp(t, 0, 1) * 100 + "%";
  $("#gauge-dot").style.left = clamp(d.difficulty, 0, 1) * 100 + "%";
  const tier = d.tier;
  $("#route-out").innerHTML = `<div class="route-line"><span>predicted difficulty</span><b>${fmt(d.difficulty, 3)}</b></div>
    <div class="route-line"><span>route</span><span class="tier-tag ${tier}">${tier} model</span></div>` + extra;
}
function initPlayground() {
  const chips = $("#chips"), ta = $("#prompt");
  $("#play-note").textContent = LIVE ? "Connected to the running gateway. Scoring uses the real classifier; the gateway can also run the prompt." : "Static demo: free text needs the running gateway. Browse recorded prompts instead, each with the real outcome of both models.";
  if (LIVE) {
    EXAMPLES.forEach((x) => { const b = document.createElement("button"); b.className = "chip"; b.textContent = x.length > 54 ? x.slice(0, 52) + "..." : x; b.onclick = () => { ta.value = x; $("#score-btn").click(); }; chips.appendChild(b); });
    $("#play-help").textContent = "Threshold for this gateway is set in config/gateway.yaml (or PUT /v1/policy). The slider above re-routes the recorded evaluation set only.";
    $("#score-btn").onclick = async () => {
      if (!ta.value.trim()) return;
      try {
        const r = await fetch("v1/route", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt: ta.value }) });
        if (!r.ok) throw new Error(r.status); showRoute(await r.json());
      } catch (e) { $("#route-out").innerHTML = `<p class="note ember">Could not score the prompt (${e.message}).</p>`; }
    };
    $("#run-btn").onclick = async () => {
      if (!ta.value.trim()) return;
      $("#route-out").innerHTML = `<p class="note">Running on the local models, this can take a few seconds...</p>`;
      try {
        const r = await fetch("v1/chat/completions", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ messages: [{ role: "user", content: ta.value }], max_tokens: 160 }) });
        if (!r.ok) throw new Error(r.status);
        const j = await r.json(), m = j.router;
        showRoute({ difficulty: m.difficulty, threshold: m.threshold, tier: m.tier },
          `<div class="route-line"><span>model</span><b>${m.model}</b></div>
           <div class="route-line"><span>escalated</span><b>${m.escalated ? "yes (" + m.escalation_reason + ")" : "no"}</b></div>
           <div class="route-line"><span>cached</span><b>${m.cached ? "yes" : "no"}</b></div>
           <div class="route-line"><span>est. cost if priced as ${m.price_basis}</span><b>$${m.cost_usd.toFixed(6)}</b></div>
           <div class="route-line"><span>vs big tier alone</span><b>$${m.big_only_cost_usd.toFixed(6)} (saves $${m.saving_usd.toFixed(6)})</b></div>
           <div class="route-line"><span>latency</span><b>${m.latency_s.toFixed(2)} s</b></div>
           <div class="answer"></div>`);
        $("#route-out .answer").textContent = j.choices[0].message.content;
      } catch (e) { $("#route-out").innerHTML = `<p class="note ember">The gateway returned an error (${e.message}).</p>`; }
    };
  } else {
    $("#score-btn").textContent = "Score difficulty"; $("#run-btn").hidden = true;
    ta.readOnly = true; ta.placeholder = "Free text needs the running gateway. Pick a recorded prompt below.";
    $("#play-help").textContent = "To score your own text: clone the repository and run `modelrouter serve` (see the README).";
    $("#score-btn").hidden = true;
  }
  buildItemList();
}
function buildItemList() {
  const d = S.ev.d, ids = Object.keys(d.text).map(Number).slice(0, 80);
  if (!ids.length) return;
  $("#items-panel").hidden = false;
  $("#items-panel .sec-note").textContent += " · prompt text, shortened: " + Object.values(d.attribution || {}).join(" · ");
  const list = $("#item-list");
  ids.forEach((i) => {
    const b = document.createElement("button"); b.className = "item"; b.setAttribute("aria-pressed", "false");
    b.innerHTML = `<span></span><small>${d.families[d.fam[i]]} · d=${d.p[i].toFixed(2)}</small>`;
    b.firstChild.textContent = d.text[i];
    b.onclick = () => {
      list.querySelectorAll(".item").forEach((x) => x.setAttribute("aria-pressed", "false")); b.setAttribute("aria-pressed", "true");
      $("#prompt").value = d.text[i];
      const big = d.p[i] >= S.threshold;
      showRoute({ difficulty: d.p[i], threshold: S.threshold, tier: big ? "big" : "small" },
        `<div class="route-line"><span>recorded: ${S.off.small}</span><b>${d.sq[i] >= 0.5 ? "correct" : "wrong"} · $${(d.sc[i]).toFixed(5)}</b></div>
         <div class="route-line"><span>recorded: ${S.off.big}</span><b>${d.bq[i] >= 0.5 ? "correct" : "wrong"} · $${(d.bc[i]).toFixed(5)}</b></div>
         <p class="note">${big ? (d.sq[i] >= d.bq[i] ? "Sent to the big model although the small one was just as good: money spent for nothing." : "Sent to the big model, which did better here.")
          : (d.sq[i] < d.bq[i] ? "Sent to the small model, but the big one would have been right: a misroute." : "Sent to the small model, which was good enough.")}</p>`);
    };
    list.appendChild(b);
  });
  list.firstChild.click();
}

boot().catch((e) => { console.error(e); $("#mode-badge").textContent = "failed to load data"; });
