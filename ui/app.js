const state = {
  bootstrap: null,
  selectedMode: 'serial',
  running: false,
  result: null,
  staticData: (window.TRAFFIC_DASHBOARD_DATA || null),
  modeConfig: {},
};

const modeMeta = {
  serial: {
    title: 'Serial',
    description: 'Single-process baseline',
    accent: '#0f766e',
  },
  openmp: {
    title: 'OpenMP',
    description: 'Shared-memory parallel run',
    accent: '#1d4ed8',
  },
  mpi: {
    title: 'MPI',
    description: 'Distributed-memory scaling',
    accent: '#7c3aed',
  },
  hybrid: {
    title: 'Hybrid',
    description: 'MPI + OpenMP combined',
    accent: '#d97706',
  },
};

const palette = ['#0f766e', '#1d4ed8', '#7c3aed', '#d97706', '#be123c', '#2563eb', '#059669', '#9333ea'];

const fallbackBootstrap = {
  host: window.location.protocol === 'file:' ? 'file' : 'local',
  notes: 'Open via the local server at http://127.0.0.1:8000 to enable compiling and running benchmarks from the dashboard.',
  modes: [
    { id: 'serial', label: 'Serial', description: 'Single-process baseline' },
    { id: 'openmp', label: 'OpenMP', description: 'Shared-memory parallel run' },
    { id: 'mpi', label: 'MPI', description: 'Distributed-memory scaling' },
    { id: 'hybrid', label: 'Hybrid', description: 'MPI + OpenMP combined' },
  ],
  modeConfig: {
    serial: {},
    openmp: {
      defaults: { threads: 4, schedule: 'static' },
      threads: { min: 1, max: 64, presets: [1, 2, 4, 8] },
      schedules: ['static', 'dynamic', 'collapse'],
    },
    mpi: {
      defaults: { processes: 4 },
      processes: { min: 1, max: 64, presets: [1, 2, 4, 8] },
    },
    hybrid: {
      defaults: { processes: 2, threads: 4 },
      processes: { min: 1, max: 64, presets: [1, 2, 4] },
      threads: { min: 1, max: 64, presets: [1, 2, 4, 8] },
    },
  },
  defaultMode: 'serial',
};

function $(id) {
  return document.getElementById(id);
}

function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return 'N/A';
  }
  return Number(value).toFixed(digits);
}

function formatMetricValue(label, value) {
  if (value === null || value === undefined) {
    return 'N/A';
  }
  if (typeof value === 'number' && Number.isFinite(value)) {
    if (label.toLowerCase().includes('time')) {
      return `${value.toFixed(4)} s`;
    }
    return value.toFixed(4);
  }
  return String(value);
}

function setRunning(running) {
  state.running = running;
  $('runButton').disabled = running;
  $('runButton').textContent = running ? 'Running benchmark...' : 'Run selected benchmark';
  $('statusChip').textContent = running ? 'Running' : 'Idle';
  $('statusChip').style.background = running ? 'rgba(15, 118, 110, 0.12)' : 'rgba(15, 23, 42, 0.06)';
}

function buildModeCards() {
  const container = $('modeList');
  container.innerHTML = '';
  for (const mode of state.bootstrap.modes) {
    const meta = modeMeta[mode.id] || {};
    const button = document.createElement('button');
    button.className = 'mode-card';
    button.dataset.mode = mode.id;
    button.innerHTML = `
      <div class="mode-card-title">${mode.label}</div>
      <div class="mode-card-desc">${mode.description}</div>
    `;
    button.addEventListener('click', () => selectMode(mode.id));
    container.appendChild(button);
  }
  updateModeSelection();
}

function updateModeSelection() {
  document.querySelectorAll('.mode-card').forEach((button) => {
    button.classList.toggle('active', button.dataset.mode === state.selectedMode);
    button.style.borderColor = button.dataset.mode === state.selectedMode
      ? `${(modeMeta[state.selectedMode] || {}).accent || '#0f766e'}66`
      : 'rgba(15, 23, 42, 0.10)';
  });
  const meta = modeMeta[state.selectedMode] || modeMeta.serial;
  $('chartsTitle').textContent = `${meta.title} performance curves`;
  $('heatmapTitle').textContent = `${meta.title} traffic heatmap`;
}

function ensureModeConfigDefaults() {
  const allModes = (state.bootstrap && state.bootstrap.modeConfig) || {};
  for (const [mode, config] of Object.entries(allModes)) {
    const defaults = (config && config.defaults) || {};
    if (!state.modeConfig[mode]) {
      state.modeConfig[mode] = { ...defaults };
    } else {
      state.modeConfig[mode] = { ...defaults, ...state.modeConfig[mode] };
    }
  }
}

function currentModeConfig() {
  return state.modeConfig[state.selectedMode] || {};
}

function setModeConfig(key, value) {
  const current = currentModeConfig();
  state.modeConfig[state.selectedMode] = { ...current, [key]: value };
  updateCommandPreview();
  if (window.location.protocol === 'file:' && state.staticData && state.staticData[state.selectedMode]) {
    renderPreviewForCurrentSelection();
    return;
  }
  if ($('autoRunToggle').checked && !state.running) {
    runSelectedBenchmark();
  }
}

function runtimeInputId(key) {
  return `${state.selectedMode}-runtime-${key}`;
}

function clampValue(value, minimum, maximum, fallback) {
  const parsed = Number.parseInt(String(value), 10);
  if (Number.isNaN(parsed)) {
    return fallback;
  }
  return Math.max(minimum, Math.min(maximum, parsed));
}

function bindLimitedNumberInput(inputElement, minimum, maximum, fallback, commit) {
  let lastCommittedValue = clampValue(inputElement.value, minimum, maximum, fallback);

  const normalize = () => {
    const nextValue = clampValue(inputElement.value, minimum, maximum, fallback);
    inputElement.value = String(nextValue);
    return nextValue;
  };

  inputElement.addEventListener('input', () => {
    const nextValue = normalize();
    if (nextValue !== lastCommittedValue) {
      lastCommittedValue = nextValue;
      commit(nextValue);
    }
  });

  inputElement.addEventListener('change', () => {
    const nextValue = normalize();
    if (nextValue !== lastCommittedValue) {
      lastCommittedValue = nextValue;
      commit(nextValue);
    }
  });

  inputElement.addEventListener('blur', () => {
    const nextValue = normalize();
    if (nextValue !== lastCommittedValue) {
      lastCommittedValue = nextValue;
      commit(nextValue);
    }
  });
}

function renderRuntimeControls() {
  const container = $('runtimeControls');
  container.innerHTML = '';
  const modeConfigMeta = (state.bootstrap && state.bootstrap.modeConfig && state.bootstrap.modeConfig[state.selectedMode]) || {};
  const conf = currentModeConfig();

  if (state.selectedMode === 'serial') {
    const note = document.createElement('p');
    note.className = 'runtime-note';
    note.textContent = 'Serial mode runs as a single process and single thread. Select OpenMP, MPI, or Hybrid above to configure runtime options.';
    container.appendChild(note);
    return;
  }

  if (state.selectedMode === 'openmp') {
    const threadMin = modeConfigMeta.threads?.min || 1;
    const threadMax = modeConfigMeta.threads?.max || 64;
    const schedulePresets = modeConfigMeta.schedules || ['static', 'dynamic', 'collapse'];
    const threadField = document.createElement('div');
    threadField.className = 'runtime-field';
    threadField.innerHTML = `
      <label for="${runtimeInputId('threads')}">Threads</label>
      <input id="${runtimeInputId('threads')}" type="number" min="${threadMin}" max="${threadMax}" step="1" inputmode="numeric" value="${conf.threads || 4}" title="Allowed range: ${threadMin}-${threadMax}">
    `;
    container.appendChild(threadField);

    const threadHelp = document.createElement('p');
    threadHelp.className = 'runtime-note';
    threadHelp.textContent = `Allowed range: ${threadMin}-${threadMax} threads.`;
    container.appendChild(threadHelp);

    const scheduleField = document.createElement('div');
    scheduleField.className = 'runtime-field';
    scheduleField.innerHTML = `
      <label for="${runtimeInputId('schedule')}">Schedule</label>
      <select id="${runtimeInputId('schedule')}">
        ${schedulePresets.map((s) => `<option value="${s}" ${conf.schedule === s ? 'selected' : ''}>${s}</option>`).join('')}
      </select>
    `;
    container.appendChild(scheduleField);

    bindLimitedNumberInput($(runtimeInputId('threads')), threadMin, threadMax, conf.threads || 4, (value) => {
      setModeConfig('threads', value);
    });
    $(runtimeInputId('schedule')).addEventListener('change', (event) => {
      setModeConfig('schedule', event.target.value);
    });
    return;
  }

  const addProcessField = () => {
    const processMin = modeConfigMeta.processes?.min || 1;
    const processMax = modeConfigMeta.processes?.max || 64;
    const processField = document.createElement('div');
    processField.className = 'runtime-field';
    processField.innerHTML = `
      <label for="${runtimeInputId('processes')}">MPI processes</label>
      <input id="${runtimeInputId('processes')}" type="number" min="${processMin}" max="${processMax}" step="1" inputmode="numeric" value="${conf.processes || 2}" title="Allowed range: ${processMin}-${processMax}">
    `;
    container.appendChild(processField);
    const processHelp = document.createElement('p');
    processHelp.className = 'runtime-note';
    processHelp.textContent = `Allowed range: ${processMin}-${processMax} MPI processes.`;
    container.appendChild(processHelp);

    bindLimitedNumberInput($(runtimeInputId('processes')), processMin, processMax, conf.processes || 2, (value) => {
      setModeConfig('processes', value);
    });
  };

  addProcessField();

  if (state.selectedMode === 'hybrid') {
    const threadMin = modeConfigMeta.threads?.min || 1;
    const threadMax = modeConfigMeta.threads?.max || 64;
    const threadField = document.createElement('div');
    threadField.className = 'runtime-field';
    threadField.innerHTML = `
      <label for="${runtimeInputId('threads')}">OMP threads per process</label>
      <input id="${runtimeInputId('threads')}" type="number" min="${threadMin}" max="${threadMax}" step="1" inputmode="numeric" value="${conf.threads || 4}" title="Allowed range: ${threadMin}-${threadMax}">
    `;
    container.appendChild(threadField);
    const threadHelp = document.createElement('p');
    threadHelp.className = 'runtime-note';
    threadHelp.textContent = `Allowed range: ${threadMin}-${threadMax} threads per process.`;
    container.appendChild(threadHelp);

    bindLimitedNumberInput($(runtimeInputId('threads')), threadMin, threadMax, conf.threads || 4, (value) => {
      setModeConfig('threads', value);
    });
  }
}

function selectMode(mode) {
  state.selectedMode = mode;
  updateModeSelection();
  renderRuntimeControls();
  updateCommandPreview();
  if (window.location.protocol === 'file:' && state.staticData && state.staticData[mode]) {
    renderPreviewForCurrentSelection();
    return;
  }
  if ($('autoRunToggle').checked) {
    runSelectedBenchmark();
  }
}

function updatePlatformInfo() {
  $('platformBadge').textContent = `${state.bootstrap.host.toUpperCase()} profile`;
  $('platformHint').textContent = state.bootstrap.notes;
}

function updateCommandPreview() {
  const mode = state.selectedMode;
  const host = state.bootstrap ? state.bootstrap.host : 'unknown';
  const conf = currentModeConfig();
  const commands = [];
  if (mode === 'serial') {
    const compiler = host === 'macos' ? 'clang' : 'gcc';
    commands.push(`cd serial`);
    commands.push(`${compiler} -O2 -o traffic_serial traffic_serial.c -lm`);
    commands.push(`./traffic_serial`);
  } else if (mode === 'openmp') {
    commands.push(`cd openmp`);
    if (host === 'macos') {
      commands.push('clang -O2 -Xpreprocessor -fopenmp -o traffic_openmp traffic_openmp.c -lm -lomp');
    } else {
      commands.push('gcc -O2 -fopenmp -o traffic_openmp traffic_openmp.c -lm');
    }
    commands.push(`./traffic_openmp ${conf.threads || 4} ${conf.schedule || 'static'}`);
  } else if (mode === 'mpi') {
    commands.push(`cd mpi`);
    commands.push('mpicc -O2 -o traffic_mpi traffic_mpi.c -lm');
    commands.push(`mpirun --oversubscribe -np ${conf.processes || 4} ./traffic_mpi`);
  } else {
    commands.push(`cd hybrid`);
    if (host === 'macos') {
      commands.push('mpicc -O2 -Xpreprocessor -fopenmp -o traffic_hybrid traffic_hybrid.c -lm -lomp');
    } else {
      commands.push('mpicc -O2 -fopenmp -o traffic_hybrid traffic_hybrid.c -lm');
    }
    commands.push(`mpirun --oversubscribe -np ${conf.processes || 2} ./traffic_hybrid ${conf.threads || 4}`);
  }
  $('commandPreview').textContent = commands.join('\n');
}

function cloneSnapshot(snapshot) {
  return JSON.parse(JSON.stringify(snapshot));
}

function pickPoint(points, xTarget) {
  if (!points || !points.length) {
    return null;
  }
  const exact = points.find((point) => Number(point.x) === Number(xTarget));
  if (exact) {
    return exact;
  }
  let nearest = points[0];
  let bestDist = Math.abs(Number(points[0].x) - Number(xTarget));
  for (const point of points.slice(1)) {
    const dist = Math.abs(Number(point.x) - Number(xTarget));
    if (dist < bestDist) {
      nearest = point;
      bestDist = dist;
    }
  }
  return nearest;
}

function configuredPreviewSnapshot(mode, conf) {
  if (!state.staticData || !state.staticData[mode]) {
    return null;
  }
  const snapshot = cloneSnapshot(state.staticData[mode]);
  const summary = snapshot.summary || {};
  const host = state.bootstrap ? state.bootstrap.host : 'local';

  if (mode === 'openmp') {
    const schedule = conf.schedule || 'static';
    const threads = Number(conf.threads || 4);
    const timeSeries = (snapshot.charts?.[0]?.series || []).find((s) => s.label === schedule);
    const speedSeries = (snapshot.charts?.[1]?.series || []).find((s) => s.label === schedule);
    const timePoint = pickPoint(timeSeries?.points || [], threads);
    const speedPoint = pickPoint(speedSeries?.points || [], threads);
    if (timePoint) {
      summary.executionTime = Number(timePoint.y);
      summary.bestConfig = `${Number(timePoint.x)} threads / ${schedule}`;
    }
    if (speedPoint && timePoint) {
      summary.bestSpeedup = Number(speedPoint.y);
      summary.bestEfficiency = Number(speedPoint.y) / Number(timePoint.x);
    }
    const compileCmd = host === 'macos'
      ? 'clang -O2 -Xpreprocessor -fopenmp -o traffic_openmp traffic_openmp.c -lm -lomp'
      : 'gcc -O2 -fopenmp -o traffic_openmp traffic_openmp.c -lm';
    let log = `[Preview — static snapshot data]\n\n$ ${compileCmd}\n$ ./traffic_openmp ${threads} ${schedule}`;
    if (timePoint) {
      log += `\n\nSelected config: ${threads} threads / ${schedule} schedule`;
      log += `\n  Execution time : ${Number(timePoint.y).toFixed(4)} s`;
      if (speedPoint) {
        log += `\n  Speedup        : ${Number(speedPoint.y).toFixed(4)}x`;
        log += `\n  Efficiency     : ${(Number(speedPoint.y) / threads * 100).toFixed(2)}%`;
      }
    } else {
      log += `\n\nNo recorded data for ${threads} threads / ${schedule}.`;
    }
    snapshot.log = log;
  } else if (mode === 'mpi') {
    const np = Number(conf.processes || 4);
    const timeSeries = snapshot.charts?.[0]?.series?.[0];
    const speedSeries = snapshot.charts?.[1]?.series?.[0];
    const timePoint = pickPoint(timeSeries?.points || [], np);
    const speedPoint = pickPoint(speedSeries?.points || [], np);
    if (timePoint) {
      summary.executionTime = Number(timePoint.y);
      summary.bestConfig = `${Number(timePoint.x)} processes`;
    }
    if (speedPoint && timePoint) {
      summary.bestSpeedup = Number(speedPoint.y);
      summary.bestEfficiency = Number(speedPoint.y) / Number(timePoint.x);
    }
    let log = `[Preview — static snapshot data]\n\n$ mpicc -O2 -o traffic_mpi traffic_mpi.c -lm\n$ mpirun --oversubscribe -np ${np} ./traffic_mpi`;
    if (timePoint) {
      log += `\n\nSelected config: ${np} MPI processes`;
      log += `\n  Execution time : ${Number(timePoint.y).toFixed(4)} s`;
      if (speedPoint) {
        log += `\n  Speedup        : ${Number(speedPoint.y).toFixed(4)}x`;
        log += `\n  Efficiency     : ${(Number(speedPoint.y) / np * 100).toFixed(2)}%`;
      }
    } else {
      log += `\n\nNo recorded data for ${np} processes.`;
    }
    snapshot.log = log;
  } else if (mode === 'hybrid') {
    const np = Number(conf.processes || 2);
    const nt = Number(conf.threads || 4);
    const targetTotal = np * nt;
    const label = `${np} MPI procs`;
    const timeSeries = (snapshot.charts?.[0]?.series || []).find((s) => s.label === label);
    const speedSeries = (snapshot.charts?.[1]?.series || []).find((s) => s.label === label);
    const timePoint = pickPoint(timeSeries?.points || [], targetTotal);
    const speedPoint = pickPoint(speedSeries?.points || [], targetTotal);
    if (timePoint) {
      summary.executionTime = Number(timePoint.y);
      summary.bestConfig = `${np}P x ${nt}T`;
    }
    if (speedPoint && timePoint) {
      summary.bestSpeedup = Number(speedPoint.y);
      summary.bestEfficiency = Number(speedPoint.y) / Number(timePoint.x);
    }
    const compileCmd = host === 'macos'
      ? 'mpicc -O2 -Xpreprocessor -fopenmp -o traffic_hybrid traffic_hybrid.c -lm -lomp'
      : 'mpicc -O2 -fopenmp -o traffic_hybrid traffic_hybrid.c -lm';
    let log = `[Preview — static snapshot data]\n\n$ ${compileCmd}\n$ mpirun --oversubscribe -np ${np} ./traffic_hybrid ${nt}`;
    if (timePoint) {
      log += `\n\nSelected config: ${np} MPI processes x ${nt} threads (${targetTotal} total workers)`;
      log += `\n  Execution time : ${Number(timePoint.y).toFixed(4)} s`;
      if (speedPoint) {
        log += `\n  Speedup        : ${Number(speedPoint.y).toFixed(4)}x`;
        log += `\n  Efficiency     : ${(Number(speedPoint.y) / targetTotal * 100).toFixed(2)}%`;
      }
    } else {
      log += `\n\nNo recorded data for ${np}P x ${nt}T.`;
    }
    snapshot.log = log;
  }

  snapshot.summary = summary;
  snapshot.requestedConfig = { ...conf };
  return snapshot;
}

function renderPreviewForCurrentSelection() {
  const snapshot = configuredPreviewSnapshot(state.selectedMode, currentModeConfig());
  if (!snapshot) {
    return;
  }
  renderResult(snapshot);
  $('statusChip').textContent = 'Preview data';
}

function formatCommandSteps(result) {
  const sections = [];
  if (result.log) {
    sections.push(result.log.trimEnd());
  }
  if (result.report) {
    sections.push(result.report.trimEnd());
  }
  return sections.filter(Boolean).join('\n\n');
}

function renderMetrics(summary) {
  const grid = $('metricsGrid');
  const metrics = [
    { label: 'Execution time', value: summary.executionTime, note: 'Best observed runtime' },
    { label: 'Best config', value: summary.bestConfig, note: 'From the latest outputs' },
    { label: 'Grid average', value: summary.gridAvg, note: 'Lane-averaged density' },
    { label: 'Speedup / efficiency', value: summary.bestSpeedup && summary.bestEfficiency ? `${formatNumber(summary.bestSpeedup)}x / ${formatNumber(summary.bestEfficiency * 100, 2)}%` : 'N/A', note: 'Derived from the run' },
  ];
  grid.innerHTML = '';
  for (const metric of metrics) {
    const card = document.createElement('article');
    card.className = 'metric-card';
    card.innerHTML = `
      <div class="metric-label">${metric.label}</div>
      <div class="metric-value">${formatMetricValue(metric.label, metric.value)}</div>
      <div class="metric-subtext">${metric.note}</div>
    `;
    grid.appendChild(card);
  }
}

function fitCanvas(canvas, height) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.floor(rect.width));
  const actualHeight = height || Math.max(260, Math.floor(rect.height || 320));
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(actualHeight * dpr);
  canvas.style.height = `${actualHeight}px`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width, height: actualHeight };
}

function heatColor(value, min, max) {
  const range = Math.max(1e-9, max - min);
  const ratio = Math.max(0, Math.min(1, (value - min) / range));
  const hue = 230 - ratio * 230;
  const lightness = 34 + ratio * 30;
  return `hsl(${hue}, 82%, ${lightness}%)`;
}

function renderHeatmap(grid, title) {
  const canvas = $('heatmapCanvas');
  if (!grid || !grid.values || !grid.values.length) {
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    $('heatmapMeta').textContent = 'No grid data available';
    return;
  }
  const { ctx, width, height } = fitCanvas(canvas, Math.max(520, Math.min(760, Math.floor(grid.rows * 3.2))));
  const values = grid.values;
  const rows = values.length;
  const cols = values[0].length;
  const padding = 18;
  const legendWidth = 18;
  const gridWidth = width - padding * 2 - legendWidth - 10;
  const gridHeight = height - padding * 2 - 18;
  const cellW = gridWidth / cols;
  const cellH = gridHeight / rows;
  const min = grid.min;
  const max = grid.max;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, width, height);

  for (let i = 0; i < rows; i += 1) {
    for (let j = 0; j < cols; j += 1) {
      ctx.fillStyle = heatColor(values[i][j], min, max);
      ctx.fillRect(padding + j * cellW, padding + i * cellH, cellW + 0.5, cellH + 0.5);
    }
  }

  const rainX = padding + (cols / 3) * cellW;
  const rainY = padding + (rows / 3) * cellH;
  const rainW = (cols / 3) * cellW;
  const rainH = (rows / 3) * cellH;
  const accX = padding + (3 * cols / 4) * cellW;
  const accY = padding + (3 * rows / 4) * cellH;
  const accW = (cols / 4) * cellW;
  const accH = (rows / 4) * cellH;

  ctx.strokeStyle = 'rgba(255,255,255,0.85)';
  ctx.setLineDash([6, 4]);
  ctx.lineWidth = 2;
  ctx.strokeRect(rainX, rainY, rainW, rainH);
  ctx.strokeStyle = 'rgba(15, 118, 110, 0.9)';
  ctx.strokeRect(accX, accY, accW, accH);
  ctx.setLineDash([]);

  ctx.fillStyle = 'rgba(255,255,255,0.92)';
  ctx.fillRect(padding + 12, padding + 12, 150, 56);
  ctx.fillStyle = '#0f172a';
  ctx.font = '700 14px Manrope, sans-serif';
  ctx.fillText(title, padding + 22, padding + 32);
  ctx.font = '600 11px IBM Plex Mono, monospace';
  ctx.fillText(`min ${formatNumber(min, 2)}  max ${formatNumber(max, 2)}`, padding + 22, padding + 50);

  const legendX = width - padding - legendWidth;
  const legendTop = padding;
  const legendHeight = gridHeight;
  for (let i = 0; i < legendHeight; i += 1) {
    const ratio = 1 - i / Math.max(1, legendHeight - 1);
    ctx.fillStyle = heatColor(min + ratio * (max - min), min, max);
    ctx.fillRect(legendX, legendTop + i, legendWidth, 1);
  }
  ctx.strokeStyle = 'rgba(15, 23, 42, 0.18)';
  ctx.strokeRect(legendX, legendTop, legendWidth, legendHeight);

  $('heatmapMeta').textContent = `${rows} x ${cols} cells | avg ${formatNumber(grid.avg, 2)}`;
}

function chartBounds(series) {
  const points = series.flatMap((item) => item.points || []);
  if (!points.length) {
    return { xMin: 0, xMax: 1, yMin: 0, yMax: 1 };
  }
  const xValues = points.map((point) => point.x);
  const yValues = points.map((point) => point.y);
  let xMin = Math.min(...xValues);
  let xMax = Math.max(...xValues);
  let yMin = Math.min(...yValues);
  let yMax = Math.max(...yValues);
  if (xMin === xMax) {
    xMin -= 1;
    xMax += 1;
  }
  if (yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }
  const xPad = (xMax - xMin) * 0.08;
  const yPad = (yMax - yMin) * 0.12;
  return { xMin: xMin - xPad, xMax: xMax + xPad, yMin: yMin - yPad, yMax: yMax + yPad };
}

function niceTicks(min, max, count = 5) {
  const ticks = [];
  const step = (max - min) / count;
  for (let i = 0; i <= count; i += 1) {
    ticks.push(min + step * i);
  }
  return ticks;
}

function renderLineChart(canvas, config) {
  const height = 280;
  const { ctx, width, height: actualHeight } = fitCanvas(canvas, height);
  if (!config.series || !config.series.length || !config.series.some((series) => (series.points || []).length)) {
    ctx.clearRect(0, 0, width, actualHeight);
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, width, actualHeight);
    ctx.fillStyle = '#5b6778';
    ctx.font = '600 14px Manrope, sans-serif';
    ctx.fillText('No chart data available', 18, actualHeight / 2);
    return;
  }
  const padding = { left: 58, right: 18, top: 18, bottom: 48 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = actualHeight - padding.top - padding.bottom;
  const bounds = chartBounds(config.series || []);
  const xTicks = niceTicks(bounds.xMin, bounds.xMax, 5);
  const yTicks = niceTicks(bounds.yMin, bounds.yMax, 5);

  ctx.clearRect(0, 0, width, actualHeight);
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, width, actualHeight);

  ctx.strokeStyle = 'rgba(15, 23, 42, 0.08)';
  ctx.lineWidth = 1;
  ctx.font = '11px IBM Plex Mono, monospace';
  ctx.fillStyle = '#0f172a';

  yTicks.forEach((tick) => {
    const y = padding.top + chartHeight - ((tick - bounds.yMin) / (bounds.yMax - bounds.yMin)) * chartHeight;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(padding.left + chartWidth, y);
    ctx.stroke();
    ctx.fillText(tick.toFixed(2), 10, y + 4);
  });

  xTicks.forEach((tick) => {
    const x = padding.left + ((tick - bounds.xMin) / (bounds.xMax - bounds.xMin)) * chartWidth;
    ctx.beginPath();
    ctx.moveTo(x, padding.top);
    ctx.lineTo(x, padding.top + chartHeight);
    ctx.stroke();
    ctx.fillText(tick.toFixed(0), x - 8, padding.top + chartHeight + 18);
  });

  ctx.strokeStyle = 'rgba(15, 23, 42, 0.82)';
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(padding.left, padding.top);
  ctx.lineTo(padding.left, padding.top + chartHeight);
  ctx.lineTo(padding.left + chartWidth, padding.top + chartHeight);
  ctx.stroke();

  config.series.forEach((series, index) => {
    const color = series.color || palette[index % palette.length];
    const points = (series.points || []).slice().sort((a, b) => a.x - b.x);
    if (!points.length) {
      return;
    }
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    points.forEach((point, pointIndex) => {
      const x = padding.left + ((point.x - bounds.xMin) / (bounds.xMax - bounds.xMin)) * chartWidth;
      const y = padding.top + chartHeight - ((point.y - bounds.yMin) / (bounds.yMax - bounds.yMin)) * chartHeight;
      if (pointIndex === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
    points.forEach((point) => {
      const x = padding.left + ((point.x - bounds.xMin) / (bounds.xMax - bounds.xMin)) * chartWidth;
      const y = padding.top + chartHeight - ((point.y - bounds.yMin) / (bounds.yMax - bounds.yMin)) * chartHeight;
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, Math.PI * 2);
      ctx.fill();
    });
  });

  ctx.font = '700 14px Manrope, sans-serif';
  ctx.fillStyle = '#0f172a';
  ctx.fillText(config.title, padding.left, 16);
  ctx.font = '600 12px Manrope, sans-serif';
  ctx.fillStyle = '#51606f';
  ctx.fillText(config.xLabel, padding.left + chartWidth / 2 - 30, actualHeight - 10);
  ctx.save();
  ctx.translate(16, padding.top + chartHeight / 2 + 40);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(config.yLabel, 0, 0);
  ctx.restore();

  const legendX = padding.left + 10;
  const legendY = 26;
  config.series.forEach((series, index) => {
    const color = series.color || palette[index % palette.length];
    const x = legendX + index * 130;
    ctx.fillStyle = color;
    ctx.fillRect(x, legendY - 10, 14, 3);
    ctx.fillStyle = '#0f172a';
    ctx.fillText(series.label, x + 20, legendY - 5);
  });
}

function renderCharts(charts) {
  const container = $('chartsContainer');
  container.innerHTML = '';
  if (!charts || !charts.length) {
    const empty = document.createElement('div');
    empty.className = 'chart-card';
    empty.textContent = 'No chart data available.';
    container.appendChild(empty);
    return;
  }
  for (const chart of charts) {
    const card = document.createElement('article');
    card.className = 'chart-card';
    card.innerHTML = `
      <div class="chart-title">${chart.title}</div>
      <canvas class="chart-canvas"></canvas>
    `;
    container.appendChild(card);
    const canvas = card.querySelector('canvas');
    renderLineChart(canvas, chart);
  }
}

function renderResult(result) {
  state.result = result;
  renderMetrics(result.summary || {});
  renderHeatmap(result.grid, `${modeMeta[state.selectedMode].title} traffic density`);
  renderCharts(result.charts || []);
  $('outputLog').textContent = formatCommandSteps(result);
  if ((result.steps || []).length) {
    $('commandPreview').textContent = (result.steps || [])
      .map((step) => `$ ${Array.isArray(step.command) ? step.command.join(' ') : step.command}`)
      .join('\n');
  }
  $('statusChip').textContent = 'Done';
}

async function runSelectedBenchmark() {
  if (state.running) {
    return;
  }
  if (window.location.protocol === 'file:') {
    if (state.staticData && state.staticData[state.selectedMode]) {
      renderPreviewForCurrentSelection();
    } else {
      $('statusChip').textContent = 'File mode';
    }
    return;
  }
  setRunning(true);
  const conf = currentModeConfig();
  let runLabel = state.selectedMode;
  if (state.selectedMode === 'openmp') {
    runLabel = `OpenMP with ${conf.threads || 4} threads`;
  } else if (state.selectedMode === 'mpi') {
    runLabel = `MPI with ${conf.processes || 4} processes`;
  } else if (state.selectedMode === 'hybrid') {
    runLabel = `Hybrid with ${conf.processes || 2} processes and ${conf.threads || 4} threads`;
  }
  $('outputLog').textContent = `Running ${runLabel}...`;
  try {
    const response = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: state.selectedMode, config: conf }),
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || `Run failed with status ${response.status}`);
    }
    state.bootstrap = payload.bootstrap || state.bootstrap;
    updatePlatformInfo();
    renderResult(payload.result);
  } catch (error) {
    $('statusChip').textContent = 'Error';
    $('outputLog').textContent = String(error.message || error);
  } finally {
    setRunning(false);
  }
}

async function init() {
  try {
    const response = await fetch('/api/bootstrap');
    state.bootstrap = await response.json();
  } catch (error) {
    state.bootstrap = fallbackBootstrap;
  }
  state.selectedMode = state.bootstrap.defaultMode || 'serial';
  ensureModeConfigDefaults();
  buildModeCards();
  updatePlatformInfo();
  renderRuntimeControls();
  updateCommandPreview();
  if (window.location.protocol === 'file:') {
    $('statusChip').textContent = 'Preview only';
  }
  $('runButton').addEventListener('click', runSelectedBenchmark);
  $('autoRunToggle').addEventListener('change', () => {
    if ($('autoRunToggle').checked && state.result) {
      selectMode(state.selectedMode);
    }
  });
  window.addEventListener('resize', () => {
    if (state.result) {
      renderHeatmap(state.result.grid, `${modeMeta[state.selectedMode].title} traffic density`);
      renderCharts(state.result.charts || []);
    }
  });
  await runSelectedBenchmark();
}

init().catch((error) => {
  $('statusChip').textContent = 'Init error';
  $('outputLog').textContent = String(error.message || error);
});