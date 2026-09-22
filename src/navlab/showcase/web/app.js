/* Robot Navigation Lab showcase.  The generated document embeds its catalog in
 * #navlab-data, so this file intentionally has no network or framework dependency. */
(function () {
  "use strict";

  const GROUP_LABELS = {
    end_to_end: "端到端導航",
    controllers: "控制器比較",
    ablation: "速度與前瞻煞車消融",
    heldout: "固定測試路線",
  };
  const STATUS_LABELS = {
    success: "成功到達",
    collision: "碰撞終止",
    timeout: "時間上限",
    no_path: "無可行路徑",
    budget_exhausted: "規劃預算耗盡",
    trajectory_failed: "軌跡執行失敗",
    data_issue: "資料無法回放",
    unknown: "記錄終止",
  };
  const statusClass = (status) => `status-${status || "unknown"}`;
  const finite = (value) => typeof value === "number" && Number.isFinite(value);
  const asNumber = (value, fallback = null) => finite(value) ? value : fallback;
  const rowCache = new WeakMap();
  const text = (value, fallback = "—") => value === null || value === undefined || value === "" ? fallback : String(value);
  const decimal = (value, digits = 2) => finite(value) ? value.toFixed(digits) : "—";
  const percent = (value) => finite(value) ? `${(value * 100).toFixed(0)}%` : "—";
  const chineseGroup = (group) => GROUP_LABELS[group] || text(group);

  function indexedRows(payload) {
    if (!payload || !Array.isArray(payload.columns) || !Array.isArray(payload.rows)) return [];
    const cached = rowCache.get(payload);
    if (cached) return cached;
    const rows = payload.rows.map((values) => Object.fromEntries(payload.columns.map((key, index) => [key, values[index]])));
    rowCache.set(payload, rows);
    return rows;
  }

  function statusForCase(caseData) {
    const planning = caseData && caseData.manifest && caseData.manifest.planning;
    const planningStatus = planning && planning.status;
    if (planningStatus === "no_path" || planningStatus === "budget_exhausted") return planningStatus;
    const metrics = caseData && caseData.metrics || {};
    if (metrics.success === true) return "success";
    if (metrics.collision === true || String(metrics.termination_reason || "").toLowerCase() === "collision") return "collision";
    if (String(metrics.termination_reason || "").toLowerCase() === "timeout") return "timeout";
    // A planning success only says that a nominal path was made.  It cannot
    // turn an unsuccessful/no-trace rollout into a successful outcome.
    if (planningStatus === "success") return "trajectory_failed";
    if (planningStatus) return "unknown";
    return metrics.termination_reason ? "unknown" : "data_issue";
  }

  function traceRows(caseData) { return indexedRows(caseData && caseData.trace); }
  function trajectoryRows(caseData) { return indexedRows(caseData && caseData.trajectory); }
  function duration(caseData) {
    const trace = traceRows(caseData);
    return trace.length ? Math.max(0, asNumber(trace[trace.length - 1].t, 0)) : 0;
  }

  function frameAt(caseData, requestedTime) {
    const start = caseData && caseData.start || {};
    const trace = traceRows(caseData);
    if (!trace.length || requestedTime < asNumber(trace[0].t, Infinity)) {
      return { t: 0, x: asNumber(start.x, 0), y: asNumber(start.y, 0), yaw: asNumber(start.yaw, 0), v: asNumber(start.v, 0), delta: asNumber(start.delta, 0), v_ref: null, cte: null, s: 0, sampleIndex: -1 };
    }
    let low = 0;
    let high = trace.length - 1;
    let answer = 0;
    while (low <= high) {
      const middle = Math.floor((low + high) / 2);
      if (asNumber(trace[middle].t, Infinity) <= requestedTime + 1e-9) { answer = middle; low = middle + 1; }
      else high = middle - 1;
    }
    return { ...trace[answer], sampleIndex: answer };
  }

  function shownTrace(caseData, requestedTime) {
    return traceRows(caseData).filter((row) => asNumber(row.t, Infinity) <= requestedTime + 1e-9);
  }

  function sameReference(a, b) {
    if (!a || !b || a.map_id !== b.map_id) return false;
    const ar = trajectoryRows(a); const br = trajectoryRows(b);
    if (!ar.length || ar.length !== br.length) return false;
    return ar.every((row, index) => ["x", "y", "yaw", "s", "v_ref", "kappa"].every((key) => row[key] === br[index][key]));
  }

  function sourceFilename(caseData, suffix) {
    return `${text(caseData.id, "case")}-${suffix}`.replace(/[^a-zA-Z0-9_.-]/g, "-");
  }

  function downloadText(filename, value, mimeType) {
    const blob = new Blob([String(value || "")], { type: `${mimeType};charset=utf-8` });
    const href = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = href; anchor.download = filename; anchor.hidden = true;
    document.body.append(anchor); anchor.click(); anchor.remove();
    // Keep the blob URL alive long enough for file:// browsers to begin their
    // download before releasing this short-lived object URL.
    window.setTimeout(() => URL.revokeObjectURL(href), 1000);
  }

  const logic = { indexedRows, statusForCase, frameAt, shownTrace, sameReference, duration, sourceFilename };
  if (typeof globalThis !== "undefined") globalThis.NavLabShowcaseLogic = logic;
  if (typeof document === "undefined") return;

  function required(id) {
    const node = document.getElementById(id);
    if (!node) throw new Error(`showcase template is missing #${id}`);
    return node;
  }

  function make(tag, options = {}) {
    const element = document.createElement(tag);
    if (options.className) element.className = options.className;
    if (options.id) element.id = options.id;
    if (options.text !== undefined) element.textContent = options.text;
    for (const [name, value] of Object.entries(options.attributes || {})) element.setAttribute(name, String(value));
    return element;
  }

  function setOptions(select, rows, selected, labelFor, valueFor) {
    select.replaceChildren(...rows.map((row) => {
      const option = make("option", { text: labelFor(row) });
      option.value = valueFor(row);
      option.selected = option.value === String(selected);
      return option;
    }));
  }

  function addMetric(target, label, value) {
    const box = make("div");
    const term = make("dt", { text: label });
    const definition = make("dd", { text: value });
    box.append(term, definition); target.append(box);
  }

  function addStatus(target, status, suffix = "", prefix = "") {
    target.replaceChildren(make("span", { className: `status-badge ${statusClass(status)}`, text: `${prefix}${STATUS_LABELS[status] || STATUS_LABELS.unknown}${suffix}` }));
  }

  function formatReason(caseData) {
    const status = statusForCase(caseData);
    if (status === "unknown") return text(caseData.metrics && caseData.metrics.termination_reason, STATUS_LABELS.unknown);
    return STATUS_LABELS[status] || STATUS_LABELS.unknown;
  }

  function metricAt(frame) {
    return [
      ["時間", `${decimal(frame.t)} s`],
      ["實際速度", `${decimal(frame.v)} m/s`],
      ["目標速度", finite(frame.v_ref) ? `${decimal(frame.v_ref)} m/s` : "—"],
      ["橫向誤差 CTE", finite(frame.cte) ? `${decimal(frame.cte)} m` : "—"],
      ["進度 s", finite(frame.s) ? `${decimal(frame.s)} m` : "—"],
      ["轉向角", `${decimal(frame.delta, 3)} rad`],
    ];
  }

  function svgElement(tag, attrs = {}) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
    Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, String(value)));
    return element;
  }

  function chart(svg, rows, series, currentTime, emptyText) {
    svg.replaceChildren();
    const width = 320, height = 112, inset = { l: 30, r: 7, t: 8, b: 19 };
    const valid = rows.filter((row) => finite(row.t) && series.some((item) => finite(row[item.key])));
    if (!valid.length) {
      const label = svgElement("text", { x: width / 2, y: height / 2, "text-anchor": "middle", fill: "#6b7d91", "font-size": "11" });
      label.textContent = emptyText; svg.append(label); return;
    }
    const xs = valid.map((row) => row.t);
    const ys = valid.flatMap((row) => series.map((item) => row[item.key]).filter(finite));
    const xMin = 0, xMax = Math.max(...xs, 0.01);
    let yMin = Math.min(...ys), yMax = Math.max(...ys);
    if (series.some((item) => item.centerZero)) { yMin = Math.min(yMin, 0); yMax = Math.max(yMax, 0); }
    const padding = Math.max(.06 * (yMax - yMin), .04); yMin -= padding; yMax += padding;
    const x = (value) => inset.l + (value - xMin) / (xMax - xMin) * (width - inset.l - inset.r);
    const y = (value) => height - inset.b - (value - yMin) / (yMax - yMin || 1) * (height - inset.t - inset.b);
    [0, .5, 1].forEach((portion) => {
      const yy = inset.t + portion * (height - inset.t - inset.b);
      svg.append(svgElement("line", { x1: inset.l, x2: width - inset.r, y1: yy, y2: yy, stroke: "#e2eaf2", "stroke-width": 1 }));
    });
    const low = svgElement("text", { x: 1, y: height - inset.b, fill: "#708196", "font-size": 9 }); low.textContent = decimal(yMin, 1);
    const high = svgElement("text", { x: 1, y: inset.t + 7, fill: "#708196", "font-size": 9 }); high.textContent = decimal(yMax, 1); svg.append(low, high);
    series.forEach((item) => {
      const points = valid.filter((row) => finite(row[item.key]));
      if (!points.length) return;
      const d = points.map((row, index) => `${index ? "L" : "M"}${x(row.t).toFixed(2)},${y(row[item.key]).toFixed(2)}`).join(" ");
      svg.append(svgElement("path", { d, fill: "none", stroke: item.color, "stroke-width": 2, "stroke-dasharray": item.dash || "", "stroke-linejoin": "round", "stroke-linecap": "round" }));
    });
    const markerX = x(Math.min(Math.max(currentTime, xMin), xMax));
    svg.append(svgElement("line", { x1: markerX, x2: markerX, y1: inset.t, y2: height - inset.b, stroke: "#66798d", "stroke-width": 1, "stroke-dasharray": "3 3" }));
    const start = svgElement("text", { x: inset.l, y: height - 4, fill: "#708196", "font-size": 9 }); start.textContent = "0 s";
    const end = svgElement("text", { x: width - inset.r, y: height - 4, "text-anchor": "end", fill: "#708196", "font-size": 9 }); end.textContent = `${decimal(xMax, 1)} s`;
    svg.append(start, end);
  }

  function canvasSize(canvas, ratio) {
    const displayWidth = Math.max(260, Math.round(canvas.clientWidth || canvas.parentElement.clientWidth || 600));
    const displayHeight = Math.max(220, Math.round(displayWidth / ratio));
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== Math.round(displayWidth * dpr) || canvas.height !== Math.round(displayHeight * dpr)) {
      canvas.width = Math.round(displayWidth * dpr); canvas.height = Math.round(displayHeight * dpr);
      canvas.style.height = `${displayHeight}px`;
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width: displayWidth, height: displayHeight };
  }

  function drawCanvas(canvas, catalog, caseData, atTime, accent) {
    const { ctx, width, height } = canvasSize(canvas, canvas.id === "replay-canvas" ? 1.5 : 1.44);
    ctx.clearRect(0, 0, width, height); ctx.fillStyle = "#f7f9fc"; ctx.fillRect(0, 0, width, height);
    if (!caseData || !catalog.maps || !catalog.maps[caseData.map_id]) {
      ctx.fillStyle = "#60758d"; ctx.font = "14px system-ui"; ctx.textAlign = "center"; ctx.fillText("此組合沒有可回放的資料", width / 2, height / 2); return;
    }
    const map = catalog.maps[caseData.map_id];
    const shape = map.shape || [1, 1]; const resolution = asNumber(map.resolution, 1);
    const origin = map.origin || [0, 0]; const mapWidth = shape[1] * resolution; const mapHeight = shape[0] * resolution;
    const pad = 20, scale = Math.min((width - pad * 2) / mapWidth, (height - pad * 2) / mapHeight);
    const left = (width - mapWidth * scale) / 2, top = (height - mapHeight * scale) / 2;
    const point = (x, y) => [left + (x - origin[0]) * scale, top + (mapHeight - (y - origin[1])) * scale];
    ctx.fillStyle = "#fbfcfe"; ctx.fillRect(left, top, mapWidth * scale, mapHeight * scale);
    ctx.fillStyle = "#d1dce8";
    (map.occupied || []).forEach(([row, column]) => {
      const [x, y] = point(origin[0] + column * resolution, origin[1] + (row + 1) * resolution);
      ctx.fillRect(x, y, resolution * scale + .4, resolution * scale + .4);
    });
    ctx.strokeStyle = "#aebfd0"; ctx.lineWidth = 1; ctx.strokeRect(left, top, mapWidth * scale, mapHeight * scale);
    const ref = trajectoryRows(caseData);
    if (ref.length) {
      ctx.save(); ctx.strokeStyle = "#2374c6"; ctx.lineWidth = 2; ctx.setLineDash([5, 5]); ctx.beginPath();
      ref.forEach((row, index) => { const [x, y] = point(row.x, row.y); index ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); ctx.restore();
    }
    const actual = shownTrace(caseData, atTime);
    if (actual.length) {
      ctx.save(); ctx.strokeStyle = accent; ctx.lineWidth = 2.5; ctx.lineJoin = "round"; ctx.beginPath();
      actual.forEach((row, index) => { const [x, y] = point(row.x, row.y); index ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); ctx.restore();
    }
    const [goalX, goalY] = point(caseData.goal[0], caseData.goal[1]);
    ctx.save(); ctx.strokeStyle = "#167a4b"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(goalX, goalY, 5, 0, Math.PI * 2); ctx.stroke(); ctx.beginPath(); ctx.moveTo(goalX - 8, goalY); ctx.lineTo(goalX + 8, goalY); ctx.moveTo(goalX, goalY - 8); ctx.lineTo(goalX, goalY + 8); ctx.stroke(); ctx.restore();
    const state = frameAt(caseData, atTime); const vehicle = caseData.vehicle || {};
    const front = asNumber(vehicle.wheelbase, 2.5) + asNumber(vehicle.front_overhang, .8);
    const rear = asNumber(vehicle.rear_overhang, .8); const half = asNumber(vehicle.width, 1.6) / 2;
    const forward = [Math.cos(state.yaw), Math.sin(state.yaw)], leftVector = [-Math.sin(state.yaw), Math.cos(state.yaw)];
    const corner = (longitudinal, lateral) => point(state.x + forward[0] * longitudinal + leftVector[0] * lateral, state.y + forward[1] * longitudinal + leftVector[1] * lateral);
    const corners = [corner(front, half), corner(front, -half), corner(-rear, -half), corner(-rear, half)];
    ctx.save(); ctx.fillStyle = `${accent}bb`; ctx.strokeStyle = accent; ctx.lineWidth = 1.7; ctx.beginPath(); corners.forEach(([x, y], index) => index ? ctx.lineTo(x, y) : ctx.moveTo(x, y)); ctx.closePath(); ctx.fill(); ctx.stroke();
    const [rearX, rearY] = point(state.x, state.y); const [noseX, noseY] = point(state.x + forward[0] * Math.min(front, 1.15), state.y + forward[1] * Math.min(front, 1.15));
    ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(rearX, rearY, 2.8, 0, Math.PI * 2); ctx.fill(); ctx.beginPath(); ctx.moveTo(rearX, rearY); ctx.lineTo(noseX, noseY); ctx.stroke(); ctx.restore();
    ctx.fillStyle = "#3d5168"; ctx.font = "11px system-ui"; ctx.textAlign = "left"; ctx.fillText("x→   y↑", left + 4, top + 14);
  }

  function createApp(catalog) {
    const cases = Array.isArray(catalog.cases) ? catalog.cases : [];
    const byId = new Map(cases.map((caseData) => [caseData.id, caseData]));
    const heldout = catalog.comparisons && catalog.comparisons.heldout || { route_seeds: [], policies: [] };
    const app = {
      catalog, cases, byId, heldout,
      replay: { id: "", time: 0, playing: false, rate: 1, raf: null, previousTimestamp: null },
      compare: { route: 0, a: "pure_pursuit", b: "ppo_seed0", time: 0, playing: false, rate: 1, raf: null, previousTimestamp: null },
      mode: "replay",
    };
    app.caseFor = (id) => byId.get(id) || null;
    app.heldoutCase = (route, policy) => cases.find((item) => item.group === "heldout" && Number(item.route_seed) === Number(route) && item.policy_id === policy) || null;
    return app;
  }

  function boot() {
    const dataNode = document.getElementById("navlab-data");
    if (!dataNode) return;
    let catalog;
    try { catalog = JSON.parse(dataNode.textContent); }
    catch (error) { document.body.replaceChildren(make("p", { className: "empty-state", text: `無法讀取展示資料：${error.message}` })); return; }
    if (!catalog || catalog.schema_version !== 1 || !Array.isArray(catalog.cases)) {
      document.body.replaceChildren(make("p", { className: "empty-state", text: "展示資料格式不受支援。" })); return;
    }
    const app = createApp(catalog);
    window.NavLabShowcase = app;
    const controls = {
      group: required("case-group"), caseSelect: required("case-select"), caseStatus: required("case-status"), caseTitle: required("case-title"), caseSubtitle: required("case-subtitle"), readout: required("case-readout"), replayCanvas: required("replay-canvas"), play: required("play-button"), restart: required("restart-button"), timeline: required("timeline"), timelineOutput: required("timeline-output"), liveMetrics: required("live-metrics"), speed: required("speed-chart"), cte: required("cte-chart"), detail: required("case-detail-content"), downloads: required("case-downloads"),
      route: required("route-select"), a: required("compare-a"), b: required("compare-b"), compareStatus: required("compare-status"), comparePlay: required("compare-play"), compareRestart: required("compare-restart"), compareTimeline: required("compare-timeline"), compareOutput: required("compare-timeline-output"), aCanvas: required("compare-a-canvas"), bCanvas: required("compare-b-canvas"), aTitle: required("compare-a-title"), bTitle: required("compare-b-title"), aReason: required("compare-a-reason"), bReason: required("compare-b-reason"), aMetrics: required("compare-a-metrics"), bMetrics: required("compare-b-metrics"), summaryBody: required("policy-summary").querySelector("tbody"),
    };
    let detailCaseId = null;

    function stop(which) {
      const state = app[which]; state.playing = false; state.previousTimestamp = null;
      if (state.raf !== null) { cancelAnimationFrame(state.raf); state.raf = null; }
    }
    function stopAll() { stop("replay"); stop("compare"); }
    function setMode(mode) {
      stopAll(); app.mode = mode;
      document.querySelectorAll(".mode-tab").forEach((button) => { const active = button.dataset.mode === mode; button.classList.toggle("is-active", active); button.setAttribute("aria-selected", String(active)); });
      document.querySelectorAll("[data-panel]").forEach((panel) => { const active = panel.dataset.panel === mode; panel.hidden = !active; panel.classList.toggle("is-hidden", !active); });
      const panel = document.getElementById(`${mode}-panel`); if (panel) panel.scrollIntoView({ behavior: "smooth", block: "start" });
      render();
    }
    function currentCase() { return app.caseFor(app.replay.id); }
    function selectedCompare() { return [app.heldoutCase(app.compare.route, app.compare.a), app.heldoutCase(app.compare.route, app.compare.b)]; }
    function replayDuration() { return duration(currentCase()); }
    function compareDuration() { const [a, b] = selectedCompare(); return Math.max(duration(a), duration(b)); }
    function setReplayTime(value) { app.replay.time = Math.min(Math.max(0, Number(value) || 0), replayDuration()); renderReplay(); }
    function setCompareTime(value) { app.compare.time = Math.min(Math.max(0, Number(value) || 0), compareDuration()); renderCompare(); }

    function populateGroups() {
      const present = Object.keys(GROUP_LABELS).filter((group) => app.cases.some((caseData) => caseData.group === group));
      setOptions(controls.group, present, "end_to_end", chineseGroup, (group) => group);
      const defaultCase = app.cases.find((caseData) => caseData.id === "end-to-end-detour-astar") || app.cases.find((caseData) => caseData.group === "end_to_end" && traceRows(caseData).length) || app.cases[0];
      app.replay.id = defaultCase ? defaultCase.id : "";
      controls.group.value = defaultCase ? defaultCase.group : present[0] || "";
      populateCases();
    }
    function populateCases() {
      const group = controls.group.value;
      const choices = app.cases.filter((caseData) => caseData.group === group);
      if (!choices.some((caseData) => caseData.id === app.replay.id)) app.replay.id = choices[0] ? choices[0].id : "";
      setOptions(controls.caseSelect, choices, app.replay.id, (caseData) => `${caseData.title} · ${formatReason(caseData)}`, (caseData) => caseData.id);
      controls.caseSelect.value = app.replay.id;
    }
    function populateCompare() {
      const routes = app.heldout.route_seeds || [];
      if (!routes.includes(app.compare.route)) app.compare.route = routes[0] || 0;
      setOptions(controls.route, routes, app.compare.route, (route) => `路線 ${route}`, (route) => route);
      const policies = app.heldout.policies || [];
      const ids = policies.map((item) => item.id);
      if (!ids.includes(app.compare.a)) app.compare.a = ids[0] || "";
      if (!ids.includes(app.compare.b)) app.compare.b = ids.includes("ppo_seed0") ? "ppo_seed0" : (ids[1] || ids[0] || "");
      const label = (policy) => `${policy.label || policy.id}${app.heldoutCase(app.compare.route, policy.id) ? "" : "（無資料）"}`;
      setOptions(controls.a, policies, app.compare.a, label, (item) => item.id); setOptions(controls.b, policies, app.compare.b, label, (item) => item.id);
      controls.route.value = String(app.compare.route); controls.a.value = app.compare.a; controls.b.value = app.compare.b;
    }

    function renderDownloads(caseData) {
      controls.downloads.replaceChildren();
      if (!caseData || !caseData.raw) return;
      [["run_json", "run.json", "application/json"], ["trace_csv", "trace.csv", "text/csv"], ["trajectory_csv", "trajectory.csv", "text/csv"]].forEach(([key, suffix, type]) => {
        const button = make("button", { id: `download-${key.replace("_", "-")}`, className: "download-button", text: `下載 ${suffix}`, attributes: { type: "button", "data-download": key } });
        button.addEventListener("click", () => downloadText(sourceFilename(caseData, suffix), caseData.raw[key], type)); controls.downloads.append(button);
      });
    }
    function renderDetail(caseData) {
      controls.detail.replaceChildren();
      if (!caseData) return;
      const list = make("div", { className: "detail-list" });
      const source = caseData.source || {}; const pairs = [["來源 archive", source.archive], ["archive 內容", source.member], ["地圖", caseData.map_id], ["終止原因", formatReason(caseData)], ["trace 樣本", traceRows(caseData).length], ["參考樣本", trajectoryRows(caseData).length]];
      pairs.forEach(([name, value]) => { list.append(make("b", { text: name }), make("span", { text: text(value) })); });
      if (Array.isArray(caseData.notes)) caseData.notes.forEach((note) => list.append(make("b", { text: "資料註記" }), make("span", { text: note })));
      controls.detail.append(list); renderDownloads(caseData);
    }
    function renderReplay() {
      const caseData = currentCase(); const total = replayDuration();
      if (!caseData) { controls.caseTitle.textContent = "找不到案例"; return; }
      app.replay.time = Math.min(app.replay.time, total); const frame = frameAt(caseData, app.replay.time); const trace = traceRows(caseData); const status = statusForCase(caseData);
      controls.caseTitle.textContent = caseData.title; controls.caseSubtitle.textContent = `${chineseGroup(caseData.group)} · ${caseData.policy_label || caseData.policy_id}`;
      addStatus(controls.caseStatus, status, trace.length ? "" : " · 無逐步 trace", "本回合結果：");
      controls.readout.textContent = `t = ${decimal(frame.t)} s${trace.length ? ` · 樣本 ${Math.max(frame.sampleIndex + 1, 0)}/${trace.length}` : " · 初始狀態"}`;
      controls.timeline.max = String(total); controls.timeline.value = String(app.replay.time); controls.timeline.disabled = !trace.length; controls.timelineOutput.value = `${decimal(app.replay.time)} s`;
      controls.play.disabled = !trace.length; controls.restart.disabled = !trace.length; controls.play.textContent = app.replay.playing ? "❚❚" : "▶"; controls.play.setAttribute("aria-label", app.replay.playing ? "暫停回放" : "播放回放");
      controls.liveMetrics.replaceChildren(); metricAt(frame).forEach(([name, value]) => addMetric(controls.liveMetrics, name, value));
      drawCanvas(controls.replayCanvas, app.catalog, caseData, app.replay.time, "#e76f51");
      chart(controls.speed, trace, [{ key: "v", color: "#e76f51" }, { key: "v_ref", color: "#2374c6", dash: "5 4" }], app.replay.time, "沒有逐步速度資料");
      chart(controls.cte, trace, [{ key: "cte", color: "#7654c9", centerZero: true }], app.replay.time, "沒有逐步 CTE 資料");
      if (detailCaseId !== caseData.id) { renderDetail(caseData); detailCaseId = caseData.id; }
    }
    function renderCompact(target, caseData, time) {
      target.replaceChildren(); if (!caseData) { addMetric(target, "狀態", "無可回放資料"); return; }
      const frame = frameAt(caseData, time); [["時間", `${decimal(Math.min(time, duration(caseData)))} s`], ["速度", `${decimal(frame.v)} m/s`], ["CTE", finite(frame.cte) ? `${decimal(frame.cte)} m` : "—"]].forEach(([name, value]) => addMetric(target, name, value));
    }
    function renderSummary() {
      controls.summaryBody.replaceChildren();
      (app.heldout.policies || []).forEach((policy) => {
        const summary = policy.summary || {}; const quality = summary.quality_all_rollouts || {}; const row = make("tr", { className: policy.id === app.compare.a || policy.id === app.compare.b ? "is-selected" : "" });
        const cells = [policy.label || policy.id, `${summary.successes ?? "—"}/${summary.episodes ?? "—"}（${percent(summary.success_rate)}）`, finite(quality.mean_abs_cte_m) ? `${decimal(quality.mean_abs_cte_m, 3)} m` : "—", finite(summary.mean_time_to_goal_s_success_only) ? `${decimal(summary.mean_time_to_goal_s_success_only)} s` : "—"];
        cells.forEach((value) => row.append(make("td", { text: value }))); controls.summaryBody.append(row);
      });
    }
    function renderCompare() {
      const [a, b] = selectedCompare(); const valid = sameReference(a, b); const total = valid ? compareDuration() : 0;
      app.compare.time = Math.min(app.compare.time, total); controls.compareTimeline.max = String(total); controls.compareTimeline.value = String(app.compare.time); controls.compareTimeline.disabled = !valid || total <= 0;
      controls.compareOutput.value = `${decimal(app.compare.time)} s`; controls.comparePlay.disabled = !valid || total <= 0; controls.compareRestart.disabled = !valid || total <= 0; controls.comparePlay.textContent = app.compare.playing ? "❚❚" : "▶";
      const replayStatus = (caseData) => {
        if (!caseData) return "無資料";
        if (app.compare.time <= 1e-9) return `初始 · 最終：${formatReason(caseData)}`;
        return app.compare.time + 1e-9 < duration(caseData) ? `回放中 · 最終：${formatReason(caseData)}` : `已結束：${formatReason(caseData)}`;
      };
      controls.aTitle.textContent = a ? a.policy_label : "策略 A：無資料"; controls.bTitle.textContent = b ? b.policy_label : "策略 B：無資料";
      controls.aReason.textContent = replayStatus(a); controls.bReason.textContent = replayStatus(b);
      [controls.aReason, controls.bReason].forEach((node, index) => { const caseData = index ? b : a; node.className = `status-badge ${statusClass(caseData ? statusForCase(caseData) : "data_issue")}`; });
      if (!valid) {
        controls.compareStatus.textContent = "此策略組合缺少資料，或不是相同地圖與參考軌跡；不會以其他路線替代。";
      } else controls.compareStatus.textContent = `路線 ${app.compare.route}：兩側共用 ${trajectoryRows(a).length} 個參考樣本。`;
      drawCanvas(controls.aCanvas, app.catalog, valid ? a : null, app.compare.time, "#e76f51"); drawCanvas(controls.bCanvas, app.catalog, valid ? b : null, app.compare.time, "#7654c9");
      renderCompact(controls.aMetrics, a, app.compare.time); renderCompact(controls.bMetrics, b, app.compare.time); renderSummary();
    }
    function render() { renderReplay(); renderCompare(); }
    function tick(which, timestamp) {
      const state = app[which]; state.raf = null; if (!state.playing) return;
      if (state.previousTimestamp === null) state.previousTimestamp = timestamp;
      const elapsed = Math.min((timestamp - state.previousTimestamp) / 1000, .2); state.previousTimestamp = timestamp;
      const total = which === "replay" ? replayDuration() : compareDuration(); state.time = Math.min(total, state.time + elapsed * state.rate);
      if (which === "replay") renderReplay(); else renderCompare();
      if (state.time >= total || total <= 0) { state.playing = false; state.previousTimestamp = null; if (which === "replay") renderReplay(); else renderCompare(); return; }
      state.raf = requestAnimationFrame((time) => tick(which, time));
    }
    function toggle(which) {
      const state = app[which]; const total = which === "replay" ? replayDuration() : compareDuration(); if (total <= 0) return;
      if (state.playing) { stop(which); if (which === "replay") renderReplay(); else renderCompare(); return; }
      if (state.time >= total) state.time = 0; state.playing = true; state.previousTimestamp = null; state.raf = requestAnimationFrame((time) => tick(which, time)); if (which === "replay") renderReplay(); else renderCompare();
    }
    function setRate(which, rate) {
      app[which].rate = rate; document.querySelectorAll(which === "replay" ? "[data-rate]" : "[data-compare-rate]").forEach((button) => button.classList.toggle("is-active", Number(button.dataset[which === "replay" ? "rate" : "compareRate"]) === rate));
    }

    document.querySelectorAll(".mode-tab").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
    document.querySelectorAll("[data-jump]").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.jump)));
    controls.group.addEventListener("change", () => { stop("replay"); app.replay.time = 0; populateCases(); renderReplay(); });
    controls.caseSelect.addEventListener("change", () => { stop("replay"); app.replay.id = controls.caseSelect.value; app.replay.time = 0; renderReplay(); });
    controls.play.addEventListener("click", () => toggle("replay")); controls.restart.addEventListener("click", () => { stop("replay"); setReplayTime(0); }); controls.timeline.addEventListener("input", () => { stop("replay"); setReplayTime(controls.timeline.value); });
    document.querySelectorAll("[data-rate]").forEach((button) => button.addEventListener("click", () => setRate("replay", Number(button.dataset.rate))));
    [controls.route, controls.a, controls.b].forEach((node) => node.addEventListener("change", () => { stop("compare"); app.compare.route = Number(controls.route.value); app.compare.a = controls.a.value; app.compare.b = controls.b.value; app.compare.time = 0; renderCompare(); }));
    controls.comparePlay.addEventListener("click", () => toggle("compare")); controls.compareRestart.addEventListener("click", () => { stop("compare"); setCompareTime(0); }); controls.compareTimeline.addEventListener("input", () => { stop("compare"); setCompareTime(controls.compareTimeline.value); });
    document.querySelectorAll("[data-compare-rate]").forEach((button) => button.addEventListener("click", () => setRate("compare", Number(button.dataset.compareRate))));
    controls.replayCanvas.addEventListener("keydown", (event) => { if (event.key === " ") { event.preventDefault(); toggle("replay"); } if (event.key === "ArrowRight" || event.key === "ArrowLeft") { event.preventDefault(); stop("replay"); const trace = traceRows(currentCase()); const current = frameAt(currentCase(), app.replay.time).sampleIndex; const next = Math.max(-1, Math.min(trace.length - 1, current + (event.key === "ArrowRight" ? 1 : -1))); setReplayTime(next < 0 ? 0 : trace[next].t); } });
    window.addEventListener("resize", () => { window.requestAnimationFrame(render); });
    window.addEventListener("beforeunload", stopAll);

    function renderProvenance() {
      const archives = catalog.provenance && catalog.provenance.source_archives || []; const target = required("archive-downloads"); target.replaceChildren();
      archives.forEach((archive) => { const item = make("a", { className: "download-button", text: `下載 ${archive.name}`, attributes: { href: `raw/${encodeURIComponent(archive.name)}`, download: archive.name, "data-archive-download": archive.name } }); target.append(item); });
      const notes = required("provenance-notes"); notes.replaceChildren(...(catalog.provenance && catalog.provenance.notes || []).map((note) => make("p", { text: note })));
      required("catalog-footnote").textContent = `${app.cases.length} 個封存案例 · schema v${catalog.schema_version}`;
      const seedTarget = required("seed-results"); seedTarget.replaceChildren();
      ["ppo_seed0", "ppo_seed1", "ppo_seed2"].forEach((id) => { const policy = (app.heldout.policies || []).find((item) => item.id === id); if (!policy) return; const summary = policy.summary || {}; const row = make("div", { className: "seed-result" }); const label = make("div"); label.append(make("strong", { text: policy.label }), make("div", { text: `${summary.successes ?? "—"}/${summary.episodes ?? "—"} 條固定測試路線成功` })); const rate = make("span", { className: `seed-rate ${summary.success_rate && summary.success_rate < 1 ? "is-moderate" : ""}`, text: percent(summary.success_rate) }); row.append(label, rate); seedTarget.append(row); });
    }

    populateGroups(); populateCompare(); renderProvenance(); render();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot, { once: true }); else boot();
}());
