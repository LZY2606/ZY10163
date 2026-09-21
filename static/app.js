"use strict";

const state = {
  meta: null,
  antennas: [],
  vis: [],
  flags: [],
  runs: [],
  selectedRun: null,
  runDetail: null,
};

const $ = (id) => document.getElementById(id);
const API = async (path, opts) => {
  const resp = await fetch(path, opts);
  if (!resp.ok) {
    let detail = resp.statusText;
    try { detail = (await resp.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return resp.json();
};
const deg = (rad) => (rad * 180) / Math.PI;
const wrap = (x) => Math.atan2(Math.sin(x), Math.cos(x));
const jsonOpts = (body) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

async function loadAll() {
  const st = await API("/api/state");
  state.meta = st.meta;
  state.antennas = st.antennas.map((a) => a.name);
  state.flags = st.flags;
  state.runs = st.runs;
  const data = await API("/api/data");
  state.vis = data.visibilities;
  renderChrome(st);
  renderFlags();
  renderRuns();
  renderCharts();
  renderGains();
}

function renderChrome(st) {
  $("meta").textContent =
    `版本 ${st.version} · 数据哈希 ${st.data_hash.slice(0, 16)}… · ` +
    `${state.meta.ntime} 时刻 × ${state.meta.nchan} 频道 · ` +
    `源相位斜率 ${state.meta.source_phase_slope} rad/频道`;

  const baselines = [];
  for (let i = 0; i < state.antennas.length; i++)
    for (let j = i + 1; j < state.antennas.length; j++)
      baselines.push(`${state.antennas[i]}-${state.antennas[j]}`);
  fillSelect($("baseline"), baselines, baselines[0]);
  fillSelect($("refAntenna"), state.antennas, "A0");
  fillSelect($("flagA"), state.antennas, "");
  fillSelect($("flagB"), state.antennas, "");
  $("tEnd").value = state.meta.ntime - 1;
}

function fillSelect(el, values, selected) {
  el.innerHTML = "";
  for (const value of values) {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = value;
    if (value === selected) opt.selected = true;
    el.appendChild(opt);
  }
}

function cellKey(a, b, t, ch) {
  return `${a}|${b}|${t}|${ch}`;
}

function visIndex() {
  const map = new Map();
  for (const row of state.vis)
    map.set(cellKey(row.antenna_a, row.antenna_b, row.t_idx, row.channel), row);
  return map;
}

function flagCovers(flag, a, b, t, ch) {
  const names = new Set([a, b]);
  if (flag.antenna_a && !names.has(flag.antenna_a)) return false;
  if (flag.antenna_b && !names.has(flag.antenna_b)) return false;
  if (flag.t_start !== null && t < flag.t_start) return false;
  if (flag.t_end !== null && t > flag.t_end) return false;
  if (flag.ch_start !== null && ch < flag.ch_start) return false;
  if (flag.ch_end !== null && ch > flag.ch_end) return false;
  return true;
}

function coveringFlags(a, b, t, ch, statuses) {
  return state.flags.filter(
    (f) => statuses.includes(f.status) && flagCovers(f, a, b, t, ch)
  );
}

const SVG_NS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
  return node;
}

function makePlot(host, opts) {
  host.innerHTML = "";
  const W = host.clientWidth || 520;
  const H = 230;
  const M = { t: 14, r: 14, b: 28, l: 42 };
  const pw = W - M.l - M.r;
  const ph = H - M.t - M.b;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}` });
  host.appendChild(svg);

  const xs = opts.points.map((p) => p.x);
  const ys = opts.points.map((p) => p.y).filter((y) => y !== null && isFinite(y));
  const ymin = opts.ymin !== undefined ? opts.ymin : Math.min(...ys);
  const ymax = opts.ymax !== undefined ? opts.ymax : Math.max(...ys);
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  const sx = (x) => M.l + ((x - xmin) / Math.max(1e-9, xmax - xmin)) * pw;
  const sy = (y) => M.t + ph - ((y - ymin) / Math.max(1e-9, ymax - ymin)) * ph;

  // axis
  svg.appendChild(svgEl("line", { class: "axis", x1: M.l, y1: M.t + ph, x2: M.l + pw, y2: M.t + ph }));
  svg.appendChild(svgEl("line", { class: "axis", x1: M.l, y1: M.t, x2: M.l, y2: M.t + ph }));
  const yTicks = opts.yTicks || 4;
  for (let i = 0; i <= yTicks; i++) {
    const yv = ymin + ((ymax - ymin) * i) / yTicks;
    const yy = sy(yv);
    svg.appendChild(svgEl("line", { class: "axis", x1: M.l - 3, y1: yy, x2: M.l, y2: yy }));
    const t = svgEl("text", { class: "axis-label", x: M.l - 5, y: yy + 3, "text-anchor": "end" });
    t.textContent = opts.yfmt ? opts.yfmt(yv) : yv.toFixed(2);
    svg.appendChild(t);
  }
  (opts.xLabels || []).forEach(({ x, label }) => {
    const t = svgEl("text", { class: "axis-label", x: sx(x), y: H - 8, "text-anchor": "middle" });
    t.textContent = label;
    svg.appendChild(t);
  });

  // flagged bands (rectangles in data coords)
  (opts.bands || []).forEach((band) => {
    svg.appendChild(
      svgEl("rect", {
        class: "band",
        x: sx(band.x0),
        y: M.t,
        width: Math.max(1, sx(band.x1) - sx(band.x0)),
        height: ph,
      })
    );
  });

  const tooltip = $("tooltip");
  const seriesGroups = new Map();
  opts.series.forEach((s) => seriesGroups.set(s.name, { color: s.color, pts: [] }));

  opts.points.forEach((p) => {
    const grp = seriesGroups.get(p.series);
    if (!grp) return;
    if (p.y === null || !isFinite(p.y)) return;
    const dot = svgEl("circle", {
      class: `point ${p.flagged ? "flagged" : ""} ${p.quality ? "quality" : ""}`,
      cx: sx(p.x),
      cy: sy(p.y),
      r: p.r || 2.6,
      fill: p.color || grp.color,
    });
    dot.addEventListener("mouseenter", (ev) => {
      tooltip.style.display = "block";
      tooltip.innerHTML = p.tip || "";
    });
    dot.addEventListener("mousemove", (ev) => {
      tooltip.style.left = `${ev.clientX + 10}px`;
      tooltip.style.top = `${ev.clientY + 10}px`;
    });
    dot.addEventListener("mouseleave", () => (tooltip.style.display = "none"));
    grp.pts.push([sx(p.x), sy(p.y)]);
    svg.appendChild(dot);
  });

  if (opts.lines !== false) {
    for (const grp of seriesGroups.values()) {
      if (grp.pts.length < 2) continue;
      const d = grp.pts.map((p, i) => `${i ? "L" : "M"}${p[0]},${p[1]}`).join(" ");
      svg.appendChild(svgEl("path", { class: "series", d, stroke: grp.color }));
    }
  }

  let lx = M.l + 6;
  for (const grp of seriesGroups.values()) {
    svg.appendChild(svgEl("rect", { x: lx, y: 4, width: 8, height: 8, fill: grp.color }));
    const t = svgEl("text", { class: "legend", x: lx + 12, y: 12 });
    const first = [...seriesGroups].find(([, g]) => g === grp);
    t.textContent = first[0];
    svg.appendChild(t);
    lx += 20 + t.textContent.length * 7;
  }
}

const COLORS = ["#5db0ff", "#57d18a", "#f2c14e", "#c792ea", "#7fdbff", "#ff9e64"];

function baselineCells(idx, a, b) {
  const out = [];
  for (let t = 0; t < state.meta.ntime; t++)
    for (let ch = 0; ch < state.meta.nchan; ch++)
      out.push(idx.get(cellKey(a, b, t, ch)));
  return out;
}

function activeStatuses() {
  const view = $("view").value;
  if (view === "raw") return [];
  if (view === "manual") return ["applied"];
  if (view === "auto") return ["applied", "suggested"];
  return ["applied"];
}

function renderCharts() {
  const [a, b] = $("baseline").value.split("-");
  const idx = visIndex();
  const cells = baselineCells(idx, a, b);
  const statuses = activeStatuses();
  const view = $("view").value;
  const showFlags = view !== "raw";
  const showSuggest = view === "auto";

  const ampPts = [];
  const phasePts = [];
  const ampSeries = [];
  const freqs = state.meta.frequencies_ghz;
  const nchan = state.meta.nchan;

  for (let t = 0; t < state.meta.ntime; t++) {
    const color = COLORS[t % COLORS.length];
    ampSeries.push({ name: `t${t}`, color });
    for (let ch = 0; ch < nchan; ch++) {
      const row = idx.get(cellKey(a, b, t, ch));
      const z = { re: row.re, im: row.im };
      const amp = Math.hypot(z.re, z.im);
      const phase = Math.atan2(z.im, z.re);
      const flagsHere = coveringFlags(a, b, t, ch, statuses);
      const flagged = showFlags && flagsHere.some((f) => f.status === "applied");
      const suggested = showSuggest && flagsHere.some((f) => f.status === "suggested");
      const x = ch;
      const tip =
        `${a}-${b} t${t} ch${ch} (${freqs[ch]}GHz)<br>` +
        `|V|=${amp.toFixed(3)} φ=${deg(phase).toFixed(1)}°` +
        (row.quality ? '<br><span style="color:#f2c14e">quality=1</span>' : "") +
        (flagsHere.length
          ? "<br>" + flagsHere.map((f) => `[${f.source}/${f.status}] ${f.reason}`).join("<br>")
          : "");
      ampPts.push({
        x, y: amp, series: `t${t}`, color: suggested ? "#5db0ff" : color,
        flagged, quality: row.quality === 1, tip,
      });
      phasePts.push({
        x, y: phase, series: `t${t}`, color: suggested ? "#5db0ff" : color,
        flagged, quality: row.quality === 1, tip,
      });
    }
  }

  const bands = [];
  if (showFlags) {
    const applied = state.flags.filter((f) => f.status === "applied");
    for (let ch = 0; ch < nchan; ch++) {
      let coveredAtSomeT = false;
      for (let t = 0; t < state.meta.ntime; t++) {
        if (coveringFlags(a, b, t, ch, ["applied"]).length) coveredAtSomeT = true;
      }
      if (coveredAtSomeT) bands.push({ x0: ch - 0.4, x1: ch + 0.4 });
    }
  }

  makePlot($("chartAmp"), {
    points: ampPts,
    series: ampSeries,
    bands,
    ymin: 0,
    xLabels: freqs.map((f, ch) => ({ x: ch, label: f.toFixed(1) })),
    yfmt: (y) => y.toFixed(1),
  });
  makePlot($("chartPhase"), {
    points: phasePts,
    series: ampSeries,
    bands,
    ymin: -Math.PI,
    ymax: Math.PI,
    xLabels: freqs.map((f, ch) => ({ x: ch, label: f.toFixed(1) })),
    yfmt: (y) => `${deg(y).toFixed(0)}°`,
  });

  renderClosure();
  renderResidual();
}

function renderClosure() {
  // raw closure phases grouped by the requested group size, all 4 triplets
  const groupSize = Math.max(1, parseInt($("groupSize").value || "2", 10));
  const idx = visIndex();
  const tStart = parseInt($("tStart").value, 10);
  const tEnd = parseInt($("tEnd").value, 10);
  const ants = state.antennas;
  const triplets = [];
  for (let i = 0; i < ants.length; i++)
    for (let j = i + 1; j < ants.length; j++)
      for (let k = j + 1; k < ants.length; k++)
        triplets.push([ants[i], ants[j], ants[k]]);

  const points = [];
  const series = triplets.map((tri, i) => ({
    name: tri.join(""),
    color: COLORS[i % COLORS.length],
  }));
  let x = 0;
  const xLabels = [];
  for (let g = 0; g < state.meta.nchan; g += groupSize) {
    for (let t = tStart; t <= tEnd; t++) {
      for (const [aa, bb, cc] of triplets) {
        let v = [0, 0];
        for (let ch = g; ch < Math.min(state.meta.nchan, g + groupSize); ch++) {
          const z1 = rowComplex(idx.get(cellKey(...sortedPair(aa, bb), t, ch)));
          const z2 = rowComplex(idx.get(cellKey(...sortedPair(bb, cc), t, ch)));
          const z3 = rowComplex(idx.get(cellKey(...sortedPair(aa, cc), t, ch)));
          const c = mul(z1, mul(z2, conj(z3)));
          v[0] += c.re;
          v[1] += c.im;
        }
        points.push({
          x,
          y: Math.atan2(v[1], v[0]),
          series: [aa, bb, cc].join(""),
          color: COLORS[triplets.findIndex((q) => q[0] === aa && q[1] === bb && q[2] === cc) % COLORS.length],
          tip: `${aa}-${bb}-${cc} t${t} g${g}<br>闭合相位=${deg(Math.atan2(v[1], v[0])).toFixed(2)}°`,
        });
      }
      x++;
    }
    xLabels.push({ x: x - 1, label: `g${g}` });
  }
  makePlot($("chartClosure"), {
    points,
    series,
    ymin: -Math.PI,
    ymax: Math.PI,
    xLabels,
    yfmt: (y) => `${deg(y).toFixed(0)}°`,
  });
}

function conj(z) {
  return { re: z.re, im: -z.im };
}
function mul(x, y) {
  return { re: x.re * y.re - x.im * y.im, im: x.re * y.im + x.im * y.re };
}
function rowComplex(row) {
  return { re: row.re, im: row.im };
}
function sortedPair(a, b) {
  return a < b ? [a, b] : [b, a];
}

function renderResidual() {
  const host = $("chartResid");
  if (!state.runDetail) {
    host.innerHTML = '<p class="muted">尚未选择运行；求解后可查看残差。</p>';
    return;
  }
  const [a, b] = $("baseline").value.split("-");
  const rows = state.runDetail.residuals.filter(
    (r) => r.antenna_a === a && r.antenna_b === b
  );
  const points = [];
  const tSet = [...new Set(rows.map((r) => r.t_idx))];
  const series = tSet.map((t, i) => ({ name: `t${t}`, color: COLORS[i % COLORS.length] }));
  rows.forEach((r) => {
    points.push({
      x: r.channel,
      y: r.phase_residual,
      series: `t${r.t_idx}`,
      color: COLORS[tSet.indexOf(r.t_idx) % COLORS.length],
      flagged: r.flagged === 1,
      tip: `${a}-${b} t${r.t_idx} ch${r.channel}<br>` +
        (r.flagged ? "已旗标（不进方程）" :
          `相位残差=${r.phase_residual === null ? "—" : deg(r.phase_residual).toFixed(2) + "°"}`),
    });
  });
  makePlot(host, {
    points,
    series,
    ymin: -Math.PI,
    ymax: Math.PI,
    xLabels: state.meta.frequencies_ghz.map((f, ch) => ({ x: ch, label: f.toFixed(1) })),
    yfmt: (y) => `${deg(y).toFixed(0)}°`,
  });
}

function intervalText(f) {
  const ant = f.antenna_b
    ? `${f.antenna_a}-${f.antenna_b}`
    : f.antenna_a || "(全天线)";
  const rng = (x, y) => `${x ?? "*"}..${y ?? "*"}`;
  return `${ant} t[${rng(f.t_start, f.t_end)}] ch[${rng(f.ch_start, f.ch_end)}]`;
}

function renderFlags() {
  const tbody = $("flagTable").querySelector("tbody");
  tbody.innerHTML = "";
  for (const f of state.flags) {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td>${f.id}</td>` +
      `<td><span class="tag ${f.source}">${f.source}</span></td>` +
      `<td><span class="tag ${f.status}">${f.status}</span></td>` +
      `<td>${intervalText(f)}</td>` +
      `<td>${escapeHtml(f.reason)}</td>` +
      `<td class="actions"></td>`;
    const actions = tr.querySelector(".actions");
    if (f.status === "suggested") {
      const apply = document.createElement("button");
      apply.className = "mini";
      apply.textContent = "采纳";
      apply.onclick = async () => {
        await API(`/api/flags/${f.id}/apply`, { method: "POST" });
        await loadAll();
      };
      const reject = document.createElement("button");
      reject.className = "mini";
      reject.textContent = "拒绝";
      reject.onclick = async () => {
        await API(`/api/flags/${f.id}/reject`, { method: "POST" });
        await loadAll();
      };
      actions.append(apply, reject);
    } else {
      const del = document.createElement("button");
      del.className = "mini";
      del.textContent = "删除";
      del.onclick = async () => {
        await API(`/api/flags/${f.id}`, { method: "DELETE" });
        await loadAll();
      };
      actions.appendChild(del);
    }
    tbody.appendChild(tr);
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])
  );
}

function renderGains() {
  const tbody = $("gainTable").querySelector("tbody");
  tbody.innerHTML = "";
  const diagBox = $("diagnostics");
  if (!state.runDetail) {
    diagBox.innerHTML = '<span class="muted">选择一个运行查看增益与诊断。</span>';
    return;
  }
  const run = state.runDetail.run;
  const diag = JSON.parse(run.diagnostics_json);
  const tags =
    `<span class="kv">状态 <span class="tag ${run.status}">${run.status}</span></span>` +
    `<span class="kv">参考 ${run.ref_antenna}（有数据：${diag.ref_has_data ? "是" : "否"}）</span>` +
    `<span class="kv">未定标天线：${diag.uncalibrated_antennas.join(", ") || "无"}</span>`;
  diagBox.innerHTML = tags;

  for (const group of state.runDetail.gains.reduce((acc, row) => {
    const g = acc.find((x) => x.group_index === row.group_index) ||
      (() => { const ng = { group_index: row.group_index, rows: [] }; acc.push(ng); return ng; })();
    g.rows.push(row);
    return acc;
  }, [])) {
    for (const row of group.rows) {
      const tr = document.createElement("tr");
      if (row.solvable) {
        const z = { re: row.gain_re, im: row.gain_im };
        const amp = Math.hypot(z.re, z.im);
        const phase = deg(Math.atan2(z.im, z.re));
        tr.innerHTML =
          `<td>${group.group_index}</td><td>${row.antenna}</td>` +
          `<td>${amp.toFixed(4)}</td><td>${phase.toFixed(2)}</td>` +
          `<td>${z.re.toFixed(4)}${z.im >= 0 ? "+" : ""}${z.im.toFixed(4)}j</td>` +
          `<td><span class="tag applied">已标定</span></td>`;
      } else {
        tr.innerHTML =
          `<td>${group.group_index}</td><td>${row.antenna}</td>` +
          `<td class="null">N/A</td><td class="null">N/A</td>` +
          `<td class="null">N/A</td><td><span class="tag failed_ref_no_data">未定标</span></td>`;
      }
      tbody.appendChild(tr);
    }
  }

  const cond = diag.groups
    .map((g) =>
      `频组${g.group_index}: 分量${g.components.map((c) => "[" + c.join(",") + "]").join(" ")} ` +
      `条件数=${g.condition_number === null ? "N/A" : g.condition_number.toFixed(2)} ` +
      `秩亏损=${g.rank_deficiency} 方程=${g.n_equations}`
    )
    .join("<br>");
  diagBox.innerHTML = tags + `<div style="margin-top:6px">${cond}</div>`;
}

function renderRuns() {
  const tbody = $("runTable").querySelector("tbody");
  tbody.innerHTML = "";
  for (const run of state.runs) {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td>${run.id}</td><td>${run.ref_antenna}</td>` +
      `<td>t[${run.t_start}..${run.t_end}] g${run.group_size}</td>` +
      `<td><span class="tag ${run.status}">${run.status}</span></td><td class="acts"></td>`;
    const acts = tr.querySelector(".acts");
    const view = document.createElement("button");
    view.className = "mini";
    view.textContent = "查看";
    view.onclick = () => selectRun(run.id);
    const replay = document.createElement("button");
    replay.className = "mini";
    replay.textContent = "重放";
    replay.onclick = () => replayRun(run.id);
    const exp = document.createElement("button");
    exp.className = "mini";
    exp.textContent = "导出";
    exp.onclick = () => exportRun(run.id);
    acts.append(view, replay, exp);
    tbody.appendChild(tr);
  }
}

async function selectRun(runId) {
  state.selectedRun = runId;
  state.runDetail = await API(`/api/runs/${runId}`);
  renderGains();
  renderResidual();
}

async function replayRun(runId) {
  const out = await API(`/api/runs/${runId}/replay`, { method: "POST" });
  const box = $("replayBox");
  const c = out.checks;
  box.innerHTML =
    `运行 #${runId} 重放<br>` +
    `重放一致：<b class="${c.replay_matches ? "tag applied" : "tag failed_ref_no_data"}">` +
    `${c.replay_matches ? "是" : "否"}</b><br>` +
    `数据哈希匹配：${c.data_hash_match} · 参数哈希：${c.params_hash_match} · ` +
    `旗标集哈希：${c.flagset_hash_match}<br>` +
    `最大增益差：${c.max_gain_difference.toExponential(2)}（容差 ${c.tolerance}）` +
    (c.mismatches.length ? `<br>不一致：${JSON.stringify(c.mismatches)}` : "");
}

async function exportRun(runId) {
  const data = await API(`/api/runs/${runId}/export`);
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `run-${runId}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

function numOrNull(el) {
  const v = el.value.trim();
  return v === "" || v === "*" ? null : parseInt(v, 10);
}

function bindControls() {
  ["view", "baseline"].forEach((id) =>
    $(id).addEventListener("change", () => {
      renderCharts();
      renderFlags();
    })
  );
  ["tStart", "tEnd", "groupSize"].forEach((id) =>
    $(id).addEventListener("change", renderCharts)
  );

  $("btnSuggest").onclick = async () => {
    try {
      const out = await API("/api/suggest", { method: "POST" });
      await loadAll();
      $("view").value = "auto";
      renderCharts();
      alert(`生成 ${out.suggestions.length} 条自动建议，可在表格中采纳或拒绝。`);
    } catch (e) {
      alert(e.message);
    }
  };

  $("btnSolve").onclick = async () => {
    try {
      const body = {
        ref_antenna: $("refAntenna").value,
        t_start: parseInt($("tStart").value, 10),
        t_end: parseInt($("tEnd").value, 10),
        group_size: parseInt($("groupSize").value, 10),
      };
      const out = await API("/api/solve", jsonOpts(body));
      await loadAll();
      $("view").value = "run";
      await selectRun(out.run_id);
      renderCharts();
    } catch (e) {
      alert(e.message);
    }
  };

  $("btnAddFlag").onclick = async () => {
    try {
      const body = {
        source: "manual",
        reason: $("flagReason").value || "手工旗标",
        antenna_a: $("flagA").value || null,
        antenna_b: $("flagB").value || null,
        t_start: numOrNull($("flagT0")),
        t_end: numOrNull($("flagT1")),
        ch_start: numOrNull($("flagC0")),
        ch_end: numOrNull($("flagC1")),
        status: "applied",
      };
      await API("/api/flags", jsonOpts(body));
      $("flagReason").value = "";
      await loadAll();
    } catch (e) {
      alert(e.message);
    }
  };

  $("btnReimport").onclick = async () => {
    if (!confirm("将清空所有旗标与运行并重新导入固定 fixture，确定？")) return;
    await API("/api/reimport", { method: "POST" });
    state.selectedRun = null;
    state.runDetail = null;
    await loadAll();
  };
}

document.addEventListener("DOMContentLoaded", async () => {
  bindControls();
  try {
    await loadAll();
  } catch (e) {
    document.body.insertAdjacentHTML(
      "afterbegin",
      `<div style="padding:10px;color:#f06b6b">加载失败：${e.message}</div>`
    );
  }
});
