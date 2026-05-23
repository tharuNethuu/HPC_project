"""
==============================================================
 analysis_hybrid.py
 Serial / OpenMP / MPI / Hybrid — Complete 4-Way Comparison
 Group 10 | EE7218/EC7207 High Performance Computing

 Reads:
   ../serial_output.txt              -> serial execution time
   ../traffic_values.txt             -> serial final traffic grid
   ../openmp/openmp_performance.txt  -> OpenMP timing (all schedules)
   ../openmp/openmp_traffic_values.txt -> OpenMP final grid
   ../mpi/mpi_performance.txt        -> MPI timing (1,2,4,8 procs)
   ../mpi/mpi_traffic_values.txt     -> MPI final grid
   hybrid_performance.txt            -> Hybrid timing (np x nt combos)
   hybrid_traffic_values.txt         -> Hybrid final grid

 Generates:
   hybrid_heatmap_4way.png         -> 4-panel heatmap (S/OMP/MPI/Hybrid)
   hybrid_speedup_comparison.png   -> Speedup curves all 4 implementations
   hybrid_efficiency_comparison.png-> Efficiency curves all 4 implementations
   hybrid_time_bars.png            -> Execution time bar chart all configs
   hybrid_scaling_surface.png      -> 2D surface: Hybrid np x nt performance
   hybrid_full_report.png          -> 6-panel master report for submission

 Run:
   python3 analysis_hybrid.py
==============================================================
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

# ── File paths ────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SERIAL_OUTPUT = os.path.join(SCRIPT_DIR, '..', 'serial', 'serial_output.txt')
SERIAL_VALUES = os.path.join(SCRIPT_DIR, '..', 'serial', 'traffic_values.txt')
OMP_PERF      = os.path.join(SCRIPT_DIR, '..', 'openmp', 'openmp_performance.txt')
OMP_VALUES    = os.path.join(SCRIPT_DIR, '..', 'openmp', 'openmp_traffic_values.txt')
MPI_PERF      = os.path.join(SCRIPT_DIR, '..', 'mpi', 'mpi_performance.txt')
MPI_VALUES    = os.path.join(SCRIPT_DIR, '..', 'mpi', 'mpi_traffic_values.txt')
HYB_PERF      = os.path.join(SCRIPT_DIR, 'hybrid_performance.txt')
HYB_VALUES    = os.path.join(SCRIPT_DIR, 'hybrid_traffic_values.txt')

ROWS_G, COLS_G = 200, 200

# ── Custom colormap ───────────────────────────────────────────
def traffic_cmap():
    colors = ['#0000cc', '#0088ff', '#00ccaa', '#aaff00', '#ff8800', '#cc0000']
    return LinearSegmentedColormap.from_list('traffic', colors, N=256)

CMAP = traffic_cmap()

# ── Colour / marker scheme ────────────────────────────────────
C = {
    'serial':   '#D32F2F',
    'static':   '#1976D2',
    'dynamic':  '#F57C00',
    'collapse': '#388E3C',
    'mpi':      '#7B1FA2',
    'hybrid':   '#00695C',
    'ideal':    '#757575',
}
M = {
    'static': 'o', 'dynamic': 's', 'collapse': '^',
    'mpi': 'D', 'hybrid': '*',
}


# =============================================================================
# I/O Helpers
# =============================================================================

def load_traffic_values(filepath):
    if not os.path.exists(filepath):
        print(f"  [WARN] Not found: {filepath}")
        return None, 0, 0, 0
    with open(filepath, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    rows, cols, lanes = map(int, lines[0].split())
    grid = []
    for line in lines[1:]:
        vals = list(map(float, line.split()))
        if vals:
            grid.append(vals)
    arr = np.array(grid, dtype=np.float64)
    return arr, rows, cols, lanes


def parse_serial_time(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        content = f.read()
    m = re.search(r'Execution Time:\s*([\d.]+)\s*seconds', content)
    return float(m.group(1)) if m else None


def parse_mpi_performance(filepath):
    if not os.path.exists(filepath):
        print(f"  [WARN] Not found: {filepath}")
        return []
    records = []
    with open(filepath, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if any(line.startswith(x) for x in
                   ['=', '-', 'Grid', '[', 'Procs', 'Speed', 'Effic',
                    'MPI', 'Group']):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                np_val = int(parts[0])
                time_s = float(parts[1])
                min_d  = float(parts[2]) if len(parts) > 2 else 0.0
                max_d  = float(parts[3]) if len(parts) > 3 else 0.0
                avg_d  = float(parts[4]) if len(parts) > 4 else 0.0
                records.append({'np': np_val, 'time': time_s,
                                 'min': min_d, 'max': max_d, 'avg': avg_d})
            except (ValueError, IndexError):
                continue
    base = next((r['time'] for r in records if r['np'] == 1), None)
    for r in records:
        r['speedup']    = base / r['time'] if base else 1.0
        r['efficiency'] = r['speedup'] / r['np']
    return records


def parse_omp_performance(filepath):
    if not os.path.exists(filepath):
        print(f"  [WARN] Not found: {filepath}")
        return {}
    results = {}
    with open(filepath, 'r') as f:
        for raw in f:
            line = raw.strip()
            if (not line or any(line.startswith(x) for x in
                    ['=', '-', 'Grid', '[', 'Speed', 'Effic',
                     'MaxDiff', 'Threads', 'OpenMP'])):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                threads    = int(parts[0])
                sched      = parts[1]
                time_s     = float(parts[2])
                speedup    = float(parts[3])
                efficiency = float(parts[4])
                results.setdefault(sched, []).append(
                    (threads, time_s, speedup, efficiency))
            except (ValueError, IndexError):
                continue
    return results


def parse_hybrid_performance(filepath):
    """
    Parse hybrid_performance.txt.
    Data rows: '  np  nt  total  time  min  max  avg'
    Returns list of dicts.
    """
    if not os.path.exists(filepath):
        print(f"  [WARN] Not found: {filepath}")
        return []
    records = []
    with open(filepath, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if any(line.startswith(x) for x in
                   ['=', '-', 'Grid', '[', 'Procs', 'Hybrid', 'Group']):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                np_val  = int(parts[0])
                nt_val  = int(parts[1])
                tot_val = int(parts[2])
                time_s  = float(parts[3])
                min_d   = float(parts[4]) if len(parts) > 4 else 0.0
                max_d   = float(parts[5]) if len(parts) > 5 else 0.0
                avg_d   = float(parts[6]) if len(parts) > 6 else 0.0
                records.append({
                    'np': np_val, 'nt': nt_val, 'total': tot_val,
                    'time': time_s,
                    'min': min_d, 'max': max_d, 'avg': avg_d,
                    'label': f'{np_val}P×{nt_val}T'
                })
            except (ValueError, IndexError):
                continue
    return records


def compute_accuracy(ref_grid, cmp_grid, label=''):
    if ref_grid is None or cmp_grid is None:
        return None, None, None
    r = min(ref_grid.shape[0], cmp_grid.shape[0])
    c = min(ref_grid.shape[1], cmp_grid.shape[1])
    diff     = ref_grid[:r, :c] - cmp_grid[:r, :c]
    rmse     = float(np.sqrt(np.mean(diff ** 2)))
    mae      = float(np.mean(np.abs(diff)))
    max_diff = float(np.max(np.abs(diff)))
    if label:
        print(f"  {label}: RMSE={rmse:.6f}  MAE={mae:.6f}  Max|d|={max_diff:.6f}")
    return rmse, mae, max_diff


# =============================================================================
# Figure 1: 4-Way Heatmap (Serial | OpenMP | MPI | Hybrid)
# =============================================================================

def plot_4way_heatmaps(serial_grid, omp_grid, mpi_grid, hyb_grid,
                       rmse_omp, rmse_mpi, rmse_hyb,
                       outfile='hybrid_heatmap_4way.png'):
    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    fig.suptitle(
        'Traffic Density Heatmap — Serial  |  OpenMP  |  MPI  |  Hybrid MPI+OpenMP\n'
        'Grid: 200×200  |  Lanes: 3  |  Time Steps: 200  |  Group 10',
        fontsize=13, fontweight='bold')

    x0, x1 = COLS_G // 3, 2 * COLS_G // 3
    y0, y1 = ROWS_G // 3, 2 * ROWS_G // 3

    configs = [
        (axes[0], serial_grid, '(a) Serial\n[Baseline — 1 Process, 1 Thread]', None, C['serial']),
        (axes[1], omp_grid,    '(b) OpenMP\n[Best: 8 Threads, Static]',         rmse_omp, C['static']),
        (axes[2], mpi_grid,    '(c) MPI\n[Best: 4 Processes]',                  rmse_mpi, C['mpi']),
        (axes[3], hyb_grid,    '(d) Hybrid MPI+OpenMP\n[Best: 2P × 4T = 8]',   rmse_hyb, C['hybrid']),
    ]

    for ax, grid, title, rmse, color in configs:
        ax.set_title(title, fontsize=10, fontweight='bold', color=color)
        if grid is not None:
            im = ax.imshow(grid, cmap=CMAP, vmin=0, vmax=100,
                           aspect='equal', origin='upper')
            plt.colorbar(im, ax=ax, label='Traffic Density (0–100)',
                         fraction=0.046, pad=0.04)
            rect = mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       lw=2, edgecolor='white',
                                       facecolor='none', linestyle='--')
            ax.add_patch(rect)
            ax.text(x0 + 3, y0 + 9, 'Rain\n(0.7×)',
                    color='white', fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.55, pad=2))
            if rmse is not None:
                verdict = 'PASS' if rmse < 0.5 else 'CHECK'
                ax.annotate(f'vs Serial RMSE: {rmse:.6f}  [{verdict}]',
                            xy=(0.02, 0.03), xycoords='axes fraction',
                            fontsize=8, color='white', fontweight='bold',
                            bbox=dict(facecolor='black', alpha=0.65, pad=3))
        else:
            ax.text(0.5, 0.5, 'Output not found.\nRun the simulation first.',
                    ha='center', va='center', fontsize=10,
                    transform=ax.transAxes, color='gray')
            ax.set_facecolor('#f5f5f5')
        ax.set_xlabel('Column (Road Segment)', fontsize=8)
        ax.set_ylabel('Row (Road Segment)',    fontsize=8)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")


# =============================================================================
# Figure 2: Speedup Comparison — all 4 implementations
# =============================================================================

def plot_speedup_comparison(mpi_records, omp_perf, hyb_records, serial_time,
                             outfile='hybrid_speedup_comparison.png'):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        'Speedup vs Serial Baseline  (T_serial / T_N)\n'
        'Group 10 | Serial  ·  OpenMP (all schedules)  ·  MPI  ·  Hybrid MPI+OpenMP',
        fontsize=13, fontweight='bold')

    ax_sp, ax_eff = axes

    x_ticks = [1, 2, 4, 8]

    def omp_curve(sched, data, ax, metric='speedup'):
        ts  = [d[0] for d in data]
        if metric == 'speedup':
            ys = [serial_time / d[1] for d in data] if serial_time else [d[2] for d in data]
        else:
            ys = [serial_time / (d[1] * d[0]) for d in data] if serial_time else [d[3] for d in data]
        ax.plot(ts, ys, color=C[sched], marker=M[sched],
                lw=2, ms=8, ls=':', label=f'OMP {sched}')
        for x, y in zip(ts, ys):
            ax.annotate(f'{y:.2f}', (x, y),
                        textcoords='offset points', xytext=(4, 3),
                        fontsize=7, color=C[sched])

    # ── (a) Speedup ──────────────────────────────────────────────
    ax_sp.set_title('(a) Speedup vs Serial Baseline', fontsize=11, fontweight='bold')
    ax_sp.plot(x_ticks, x_ticks, color=C['ideal'], lw=1.5,
               ls='--', alpha=0.5, label='Ideal linear scaling')
    ax_sp.axhline(1.0, color=C['serial'], lw=1.5, ls=':',
                  alpha=0.6, label=f'Serial baseline  ({serial_time:.4f} s)' if serial_time else 'Serial')

    for sched, data in sorted(omp_perf.items()):
        omp_curve(sched, data, ax_sp, 'speedup')

    if mpi_records:
        np_v = [r['np']  for r in mpi_records]
        sp_v = [serial_time / r['time'] for r in mpi_records] if serial_time else [r['speedup'] for r in mpi_records]
        ax_sp.plot(np_v, sp_v, color=C['mpi'], marker=M['mpi'],
                   lw=2.5, ms=10, label='MPI', zorder=5)
        for x, y in zip(np_v, sp_v):
            ax_sp.annotate(f'{y:.2f}', (x, y),
                           textcoords='offset points', xytext=(4, -13),
                           fontsize=8, color=C['mpi'], fontweight='bold')

    if hyb_records:
        tot_v = [r['total'] for r in hyb_records]
        sp_h  = [serial_time / r['time'] for r in hyb_records] if serial_time else [1.0]*len(hyb_records)
        lbl_v = [r['label'] for r in hyb_records]
        ax_sp.plot(tot_v, sp_h, color=C['hybrid'], marker=M['hybrid'],
                   lw=2.5, ms=12, label='Hybrid MPI+OMP', zorder=6)
        for x, y, lbl in zip(tot_v, sp_h, lbl_v):
            ax_sp.annotate(f'{y:.2f}\n({lbl})', (x, y),
                           textcoords='offset points', xytext=(5, 5),
                           fontsize=7, color=C['hybrid'], fontweight='bold')

    ax_sp.set_xlabel('Total Workers  (Threads or Processes)', fontsize=10)
    ax_sp.set_ylabel('Speedup  =  T_serial / T_N', fontsize=10)
    ax_sp.set_xticks(x_ticks)
    ax_sp.set_xlim(0.5, 9)
    ax_sp.legend(fontsize=8, loc='upper left')
    ax_sp.grid(True, alpha=0.3)

    # ── (b) Efficiency ───────────────────────────────────────────
    ax_eff.set_title('(b) Parallel Efficiency  (T_serial / (N × T_N))',
                     fontsize=11, fontweight='bold')
    ax_eff.axhline(1.0, color=C['ideal'], lw=1.5, ls='--',
                   alpha=0.5, label='Ideal (100%)')

    for sched, data in sorted(omp_perf.items()):
        omp_curve(sched, data, ax_eff, 'efficiency')

    if mpi_records:
        np_v  = [r['np']  for r in mpi_records]
        eff_v = [serial_time / (r['time'] * r['np']) for r in mpi_records] if serial_time else [r['efficiency'] for r in mpi_records]
        ax_eff.plot(np_v, eff_v, color=C['mpi'], marker=M['mpi'],
                    lw=2.5, ms=10, label='MPI', zorder=5)
        for x, y in zip(np_v, eff_v):
            ax_eff.annotate(f'{y:.2f}\n({y*100:.0f}%)', (x, y),
                            textcoords='offset points', xytext=(0, -20),
                            ha='center', fontsize=7, color=C['mpi'])

    if hyb_records:
        tot_v = [r['total'] for r in hyb_records]
        eff_h = [serial_time / (r['time'] * r['total']) for r in hyb_records] if serial_time else [1.0]*len(hyb_records)
        lbl_v = [r['label'] for r in hyb_records]
        ax_eff.plot(tot_v, eff_h, color=C['hybrid'], marker=M['hybrid'],
                    lw=2.5, ms=12, label='Hybrid MPI+OMP', zorder=6)
        for x, y, lbl in zip(tot_v, eff_h, lbl_v):
            ax_eff.annotate(f'{y:.2f}\n({lbl})', (x, y),
                            textcoords='offset points', xytext=(5, 5),
                            fontsize=7, color=C['hybrid'], fontweight='bold')

    ax_eff.set_xlabel('Total Workers  (Threads or Processes)', fontsize=10)
    ax_eff.set_ylabel('Efficiency  =  T_serial / (N × T_N)', fontsize=10)
    ax_eff.set_xticks(x_ticks)
    ax_eff.set_ylim(0, 1.8)
    ax_eff.set_xlim(0.5, 9)
    ax_eff.legend(fontsize=8, loc='upper right')
    ax_eff.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")


# =============================================================================
# Figure 3: Execution Time Bar Chart — all configurations
# =============================================================================

def plot_time_bars(mpi_records, omp_perf, hyb_records, serial_time,
                   outfile='hybrid_time_bars.png'):
    fig, ax = plt.subplots(figsize=(20, 7))
    fig.suptitle(
        'Execution Time — All Configurations\n'
        'Group 10 | Serial  ·  OpenMP  ·  MPI  ·  Hybrid MPI+OpenMP',
        fontsize=13, fontweight='bold')

    labels, times, colors, groups = [], [], [], []

    if serial_time:
        labels.append('Serial\n1P·1T'); times.append(serial_time)
        colors.append(C['serial']); groups.append(0)

    for sched in sorted(omp_perf.keys()):
        for nt, tm, _, _ in omp_perf[sched]:
            labels.append(f'OMP\n{sched}\n{nt}T')
            times.append(tm); colors.append(C[sched]); groups.append(1)

    for r in mpi_records:
        labels.append(f'MPI\n{r["np"]}P')
        times.append(r['time']); colors.append(C['mpi']); groups.append(2)

    for r in hyb_records:
        labels.append(f'Hybrid\n{r["np"]}P·{r["nt"]}T\n={r["total"]}')
        times.append(r['time']); colors.append(C['hybrid']); groups.append(3)

    x_pos = range(len(labels))
    bars  = ax.bar(x_pos, times, color=colors, edgecolor='white', lw=0.8, width=0.7)

    # Annotate speedup vs serial on each bar
    max_t = max(times) if times else 1
    for bar, t in zip(bars, times):
        sp_str = f'{t:.4f}s'
        if serial_time:
            sp_str += f'\n({serial_time/t:.2f}×)'
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + max_t * 0.004,
                sp_str, ha='center', va='bottom', fontsize=6.5, rotation=0)

    # Mark best of each group
    group_best = {}
    for i, (t, g) in enumerate(zip(times, groups)):
        if g not in group_best or t < times[group_best[g]]:
            group_best[g] = i
    for g, idx in group_best.items():
        ax.annotate('★ Best',
                    xy=(idx, times[idx]),
                    xytext=(idx, times[idx] + max_t * 0.07),
                    ha='center', fontsize=8, fontweight='bold',
                    color=colors[idx],
                    arrowprops=dict(arrowstyle='->', color=colors[idx], lw=1.5))

    # Dividers between groups
    group_boundaries = []
    current = groups[0] if groups else 0
    for i, g in enumerate(groups):
        if g != current:
            group_boundaries.append(i - 0.5)
            current = g
    for xb in group_boundaries:
        ax.axvline(xb, color='gray', lw=1, ls='--', alpha=0.5)

    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel('Execution Time (seconds)', fontsize=11)
    ax.set_xlabel('Configuration', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max_t * 1.22)

    legend_patches = [
        mpatches.Patch(color=C['serial'],   label='Serial'),
        mpatches.Patch(color=C['static'],   label='OMP static'),
        mpatches.Patch(color=C['dynamic'],  label='OMP dynamic'),
        mpatches.Patch(color=C['collapse'], label='OMP collapse'),
        mpatches.Patch(color=C['mpi'],      label='MPI'),
        mpatches.Patch(color=C['hybrid'],   label='Hybrid MPI+OMP'),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc='upper right', ncol=2)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")


# =============================================================================
# Figure 4: Hybrid 2D Scaling Surface (np x nt heat-map)
# =============================================================================

def plot_hybrid_scaling_surface(hyb_records, serial_time,
                                 outfile='hybrid_scaling_surface.png'):
    if not hyb_records:
        print("  [WARN] No hybrid data — skipping scaling surface.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(
        'Hybrid MPI+OpenMP — 2D Scaling Analysis\n'
        'Group 10 | X-axis: OMP Threads/Process  |  Lines: MPI Processes',
        fontsize=13, fontweight='bold')

    ax_t, ax_sp, ax_eff = axes

    # Organise by MPI process count
    by_np = {}
    for r in hyb_records:
        by_np.setdefault(r['np'], []).append(r)
    for np_val in by_np:
        by_np[np_val].sort(key=lambda r: r['nt'])

    cmap_lines = plt.cm.viridis
    np_vals_sorted = sorted(by_np.keys())
    line_colors = {np_val: cmap_lines(i / max(len(np_vals_sorted) - 1, 1))
                   for i, np_val in enumerate(np_vals_sorted)}

    all_nt = sorted(set(r['nt'] for r in hyb_records))

    # ── (a) Time ─────────────────────────────────────────────────
    ax_t.set_title('(a) Execution Time', fontsize=11, fontweight='bold')
    if serial_time:
        ax_t.axhline(serial_time, color=C['serial'], lw=2, ls='--',
                     label=f'Serial ({serial_time:.4f}s)', alpha=0.8)
    for np_val in np_vals_sorted:
        data = by_np[np_val]
        nt_v = [r['nt']   for r in data]
        tm_v = [r['time'] for r in data]
        ax_t.plot(nt_v, tm_v, color=line_colors[np_val], marker='o',
                  lw=2, ms=9, label=f'{np_val} MPI proc(s)')
        for x, y in zip(nt_v, tm_v):
            ax_t.annotate(f'{y:.4f}', (x, y),
                          textcoords='offset points', xytext=(4, 4),
                          fontsize=7.5)
    ax_t.set_xlabel('OpenMP Threads per Process', fontsize=10)
    ax_t.set_ylabel('Execution Time (s)', fontsize=10)
    ax_t.set_xticks(all_nt)
    ax_t.legend(fontsize=8)
    ax_t.grid(True, alpha=0.3)

    # ── (b) Speedup ──────────────────────────────────────────────
    ax_sp.set_title('(b) Speedup vs Serial Baseline', fontsize=11, fontweight='bold')
    ax_sp.axhline(1.0, color=C['serial'], lw=1.5, ls=':',
                  alpha=0.5, label='Serial (1×)')
    for np_val in np_vals_sorted:
        data = by_np[np_val]
        nt_v = [r['nt']  for r in data]
        sp_v = [serial_time / r['time'] for r in data] if serial_time else [1.0]*len(data)
        ax_sp.plot(nt_v, sp_v, color=line_colors[np_val], marker='o',
                   lw=2, ms=9, label=f'{np_val} MPI proc(s)')
        for x, y in zip(nt_v, sp_v):
            ax_sp.annotate(f'{y:.2f}×', (x, y),
                           textcoords='offset points', xytext=(4, 4),
                           fontsize=7.5)
    ax_sp.set_xlabel('OpenMP Threads per Process', fontsize=10)
    ax_sp.set_ylabel('Speedup  =  T_serial / T_N', fontsize=10)
    ax_sp.set_xticks(all_nt)
    ax_sp.legend(fontsize=8)
    ax_sp.grid(True, alpha=0.3)

    # ── (c) Efficiency ───────────────────────────────────────────
    ax_eff.set_title('(c) Parallel Efficiency', fontsize=11, fontweight='bold')
    ax_eff.axhline(1.0, color=C['ideal'], lw=1.5, ls='--',
                   alpha=0.5, label='Ideal (100%)')
    for np_val in np_vals_sorted:
        data = by_np[np_val]
        nt_v  = [r['nt']    for r in data]
        tot_v = [r['total'] for r in data]
        eff_v = ([serial_time / (r['time'] * r['total']) for r in data]
                 if serial_time else [1.0]*len(data))
        ax_eff.plot(nt_v, eff_v, color=line_colors[np_val], marker='o',
                    lw=2, ms=9, label=f'{np_val} MPI proc(s)')
        for x, y in zip(nt_v, eff_v):
            ax_eff.annotate(f'{y*100:.0f}%', (x, y),
                            textcoords='offset points', xytext=(4, 4),
                            fontsize=7.5)
    ax_eff.set_xlabel('OpenMP Threads per Process', fontsize=10)
    ax_eff.set_ylabel('Efficiency  =  T_serial / (N_total × T_N)', fontsize=10)
    ax_eff.set_xticks(all_nt)
    ax_eff.set_ylim(0, 1.5)
    ax_eff.legend(fontsize=8)
    ax_eff.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")


# =============================================================================
# Figure 5: 6-Panel Master Report
# =============================================================================

def plot_full_report(serial_grid, omp_grid, mpi_grid, hyb_grid,
                     mpi_records, omp_perf, hyb_records, serial_time,
                     rmse_omp, rmse_mpi, rmse_hyb,
                     outfile='hybrid_full_report.png'):

    fig = plt.figure(figsize=(20, 28))
    fig.suptitle(
        'Parallel Traffic Density Simulation — Complete Analysis Report\n'
        'Serial  |  OpenMP  |  MPI  |  Hybrid MPI+OpenMP\n'
        'Group 10  |  EE7218/EC7207 High Performance Computing',
        fontsize=15, fontweight='bold', y=0.995)

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.46, wspace=0.3,
                           top=0.960, bottom=0.03,
                           left=0.07, right=0.97)

    ax_sh  = fig.add_subplot(gs[0, 0])   # (a) Serial heatmap
    ax_hh  = fig.add_subplot(gs[0, 1])   # (b) Hybrid heatmap
    ax_sp  = fig.add_subplot(gs[1, 0])   # (c) Speedup
    ax_eff = fig.add_subplot(gs[1, 1])   # (d) Efficiency
    ax_bar = fig.add_subplot(gs[2, 0])   # (e) Time bar chart
    ax_hyb = fig.add_subplot(gs[2, 1])   # (f) Hybrid 2D scaling
    ax_tbl = fig.add_subplot(gs[3, :])   # (g) Summary table (full width)

    x0, x1 = COLS_G // 3, 2 * COLS_G // 3
    y0, y1 = ROWS_G // 3, 2 * ROWS_G // 3
    x_ticks = [1, 2, 4, 8]

    # ── (a) Serial Heatmap ───────────────────────────────────────
    ax_sh.set_title('(a) Serial — Final Traffic Density  [Baseline]',
                    fontsize=10, fontweight='bold', color=C['serial'])
    if serial_grid is not None:
        im = ax_sh.imshow(serial_grid, cmap=CMAP, vmin=0, vmax=100,
                          aspect='equal', origin='upper')
        plt.colorbar(im, ax=ax_sh, label='Density', fraction=0.046, pad=0.04)
        ax_sh.add_patch(mpatches.Rectangle((x0, y0), x1-x0, y1-y0,
                        lw=2, edgecolor='white', facecolor='none', ls='--'))
        ax_sh.text(x0+3, y0+9, 'Rain\n(0.7×)', color='white', fontsize=8,
                   bbox=dict(facecolor='#000000aa', pad=2))
        if serial_time:
            ax_sh.annotate(f'Time: {serial_time:.6f} s',
                           xy=(0.02, 0.03), xycoords='axes fraction',
                           fontsize=8.5, color='white', fontweight='bold',
                           bbox=dict(facecolor='black', alpha=0.65, pad=3))
    else:
        ax_sh.text(0.5, 0.5, 'Not found', ha='center', va='center',
                   transform=ax_sh.transAxes, color='gray')
    ax_sh.set_xlabel('Column'); ax_sh.set_ylabel('Row')

    # ── (b) Hybrid Heatmap ──────────────────────────────────────
    best_hyb = min(hyb_records, key=lambda r: r['time']) if hyb_records else None
    hyb_title = f'(b) Hybrid MPI+OpenMP — Final Traffic Density'
    if best_hyb:
        hyb_title += f'\n[Best: {best_hyb["np"]}P × {best_hyb["nt"]}T = {best_hyb["total"]} workers]'
    ax_hh.set_title(hyb_title, fontsize=10, fontweight='bold', color=C['hybrid'])
    if hyb_grid is not None:
        im = ax_hh.imshow(hyb_grid, cmap=CMAP, vmin=0, vmax=100,
                          aspect='equal', origin='upper')
        plt.colorbar(im, ax=ax_hh, label='Density', fraction=0.046, pad=0.04)
        ax_hh.add_patch(mpatches.Rectangle((x0, y0), x1-x0, y1-y0,
                        lw=2, edgecolor='white', facecolor='none', ls='--'))
        ax_hh.text(x0+3, y0+9, 'Rain\n(0.7×)', color='white', fontsize=8,
                   bbox=dict(facecolor='#000000aa', pad=2))
        if rmse_hyb is not None:
            verdict = 'PASS' if rmse_hyb < 0.5 else 'CHECK'
            ax_hh.annotate(f'vs Serial RMSE: {rmse_hyb:.6f}  [{verdict}]',
                           xy=(0.02, 0.03), xycoords='axes fraction',
                           fontsize=8.5, color='white', fontweight='bold',
                           bbox=dict(facecolor='black', alpha=0.65, pad=3))
    else:
        ax_hh.text(0.5, 0.5, 'Not found\nRun: bash run_scaling_hybrid.sh',
                   ha='center', va='center', transform=ax_hh.transAxes, color='gray')
    ax_hh.set_xlabel('Column'); ax_hh.set_ylabel('Row')

    # ── (c) Speedup vs serial ────────────────────────────────────
    ax_sp.set_title('(c) Speedup vs Serial Baseline  (T_serial / T_N)',
                    fontsize=10, fontweight='bold')
    ax_sp.plot(x_ticks, x_ticks, color=C['ideal'], lw=1.5,
               ls='--', alpha=0.5, label='Ideal linear')
    ax_sp.axhline(1.0, color=C['serial'], lw=1.5, ls=':', alpha=0.5, label='Serial (1×)')

    for sched, data in sorted(omp_perf.items()):
        ts = [d[0] for d in data]
        sp = [serial_time / d[1] for d in data] if serial_time else [d[2] for d in data]
        ax_sp.plot(ts, sp, color=C[sched], marker=M[sched],
                   lw=1.8, ms=7, ls=':', label=f'OMP {sched}')

    if mpi_records:
        np_v = [r['np'] for r in mpi_records]
        sp_v = [serial_time / r['time'] for r in mpi_records] if serial_time else [r['speedup'] for r in mpi_records]
        ax_sp.plot(np_v, sp_v, color=C['mpi'], marker=M['mpi'],
                   lw=2.5, ms=9, label='MPI', zorder=5)

    if hyb_records:
        tot_v = [r['total'] for r in hyb_records]
        sp_h  = [serial_time / r['time'] for r in hyb_records] if serial_time else [1.0]*len(hyb_records)
        ax_sp.plot(tot_v, sp_h, color=C['hybrid'], marker=M['hybrid'],
                   lw=2.5, ms=11, label='Hybrid MPI+OMP', zorder=6)
        for x, y, r in zip(tot_v, sp_h, hyb_records):
            ax_sp.annotate(f'{y:.2f}×\n{r["label"]}', (x, y),
                           textcoords='offset points', xytext=(5, 4),
                           fontsize=6.5, color=C['hybrid'])

    ax_sp.set_xlabel('Total Workers', fontsize=9)
    ax_sp.set_ylabel('Speedup', fontsize=9)
    ax_sp.set_xticks(x_ticks)
    ax_sp.set_xlim(0.5, 9)
    ax_sp.legend(fontsize=7.5, ncol=2)
    ax_sp.grid(True, alpha=0.3)

    # ── (d) Efficiency ───────────────────────────────────────────
    ax_eff.set_title('(d) Parallel Efficiency  (T_serial / (N × T_N))',
                     fontsize=10, fontweight='bold')
    ax_eff.axhline(1.0, color=C['ideal'], lw=1.5, ls='--', alpha=0.5, label='Ideal (100%)')

    for sched, data in sorted(omp_perf.items()):
        ts  = [d[0] for d in data]
        eff = [serial_time / (d[1] * d[0]) for d in data] if serial_time else [d[3] for d in data]
        ax_eff.plot(ts, eff, color=C[sched], marker=M[sched],
                    lw=1.8, ms=7, ls=':', label=f'OMP {sched}')

    if mpi_records:
        np_v  = [r['np'] for r in mpi_records]
        eff_v = [serial_time / (r['time'] * r['np']) for r in mpi_records] if serial_time else [r['efficiency'] for r in mpi_records]
        ax_eff.plot(np_v, eff_v, color=C['mpi'], marker=M['mpi'],
                    lw=2.5, ms=9, label='MPI', zorder=5)

    if hyb_records:
        tot_v = [r['total'] for r in hyb_records]
        eff_h = [serial_time / (r['time'] * r['total']) for r in hyb_records] if serial_time else [1.0]*len(hyb_records)
        ax_eff.plot(tot_v, eff_h, color=C['hybrid'], marker=M['hybrid'],
                    lw=2.5, ms=11, label='Hybrid MPI+OMP', zorder=6)

    ax_eff.set_xlabel('Total Workers', fontsize=9)
    ax_eff.set_ylabel('Efficiency', fontsize=9)
    ax_eff.set_xticks(x_ticks)
    ax_eff.set_ylim(0, 1.8)
    ax_eff.set_xlim(0.5, 9)
    ax_eff.legend(fontsize=7.5, ncol=2)
    ax_eff.grid(True, alpha=0.3)

    # ── (e) Execution Time Bar Chart ─────────────────────────────
    ax_bar.set_title('(e) Execution Time — All Configurations',
                     fontsize=10, fontweight='bold')
    bar_labels, bar_times, bar_colors = [], [], []

    if serial_time:
        bar_labels.append('Serial'); bar_times.append(serial_time)
        bar_colors.append(C['serial'])

    for sched in sorted(omp_perf.keys()):
        best = min(omp_perf[sched], key=lambda d: d[1])
        bar_labels.append(f'OMP {sched}\n{best[0]}T')
        bar_times.append(best[1]); bar_colors.append(C[sched])

    if mpi_records:
        best = min(mpi_records, key=lambda r: r['time'])
        bar_labels.append(f'MPI\n{best["np"]}P')
        bar_times.append(best['time']); bar_colors.append(C['mpi'])

    if hyb_records:
        best = min(hyb_records, key=lambda r: r['time'])
        bar_labels.append(f'Hybrid\n{best["np"]}P·{best["nt"]}T')
        bar_times.append(best['time']); bar_colors.append(C['hybrid'])

    x_pos = range(len(bar_labels))
    bars  = ax_bar.bar(x_pos, bar_times, color=bar_colors,
                       edgecolor='white', lw=0.8, width=0.6)
    max_t = max(bar_times) if bar_times else 1
    for bar, t in zip(bars, bar_times):
        sp_s = f'{serial_time/t:.2f}×' if serial_time else ''
        ax_bar.text(bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + max_t * 0.015,
                    f'{t:.4f}s\n{sp_s}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax_bar.set_xticks(list(x_pos))
    ax_bar.set_xticklabels(bar_labels, fontsize=8.5)
    ax_bar.set_ylabel('Execution Time (seconds)', fontsize=9)
    ax_bar.set_ylim(0, max_t * 1.30)
    ax_bar.grid(axis='y', alpha=0.3)

    legend_p = [
        mpatches.Patch(color=C['serial'],   label='Serial'),
        mpatches.Patch(color=C['static'],   label='OMP static'),
        mpatches.Patch(color=C['dynamic'],  label='OMP dynamic'),
        mpatches.Patch(color=C['collapse'], label='OMP collapse'),
        mpatches.Patch(color=C['mpi'],      label='MPI'),
        mpatches.Patch(color=C['hybrid'],   label='Hybrid MPI+OMP'),
    ]
    ax_bar.legend(handles=legend_p, fontsize=7.5, loc='upper right', ncol=2)

    # ── (f) Hybrid 2D Scaling ────────────────────────────────────
    ax_hyb.set_title('(f) Hybrid 2D Scaling  (MPI Procs × OMP Threads)',
                     fontsize=10, fontweight='bold')
    if hyb_records:
        by_np = {}
        for r in hyb_records:
            by_np.setdefault(r['np'], []).append(r)
        cmap_lines = plt.cm.cool
        np_vals_s = sorted(by_np.keys())
        lc = {np_v: cmap_lines(i / max(len(np_vals_s)-1, 1))
              for i, np_v in enumerate(np_vals_s)}
        all_nt = sorted(set(r['nt'] for r in hyb_records))
        for np_v in np_vals_s:
            data = sorted(by_np[np_v], key=lambda r: r['nt'])
            nt_v = [r['nt']  for r in data]
            sp_v = [serial_time / r['time'] for r in data] if serial_time else [1.0]*len(data)
            ax_hyb.plot(nt_v, sp_v, color=lc[np_v], marker='o', lw=2, ms=9,
                        label=f'{np_v} MPI proc(s)')
            for x, y in zip(nt_v, sp_v):
                ax_hyb.annotate(f'{y:.2f}×', (x, y),
                                textcoords='offset points', xytext=(4, 4),
                                fontsize=7.5)
        ax_hyb.set_xticks(all_nt)
        ax_hyb.legend(fontsize=8)
        ax_hyb.grid(True, alpha=0.3)
    else:
        ax_hyb.text(0.5, 0.5, 'No hybrid data.\nRun bash run_scaling_hybrid.sh',
                    ha='center', va='center', transform=ax_hyb.transAxes, color='gray')
    ax_hyb.set_xlabel('OpenMP Threads per Process', fontsize=9)
    ax_hyb.set_ylabel('Speedup vs Serial', fontsize=9)

    # ── (g) Summary Table ────────────────────────────────────────
    ax_tbl.set_title('(g) Complete 4-Way Performance Summary Table',
                     fontsize=11, fontweight='bold')
    ax_tbl.axis('off')

    col_labels = ['Implementation', 'MPI\nProcs', 'OMP\nThreads',
                  'Total\nWorkers', 'Time (s)', 'Speedup\nvs Serial',
                  'Parallel\nEfficiency', 'Accuracy\nvs Serial']
    rows_data = []
    row_colors = []

    def _sp(t):
        return f'{serial_time/t:.4f}×' if serial_time and t else 'N/A'
    def _eff(t, n):
        return f'{serial_time/(t*n):.4f}\n({serial_time/(t*n)*100:.1f}%)' if serial_time and t and n else 'N/A'

    if serial_time:
        rows_data.append(['Serial', '1', '1', '1',
                          f'{serial_time:.6f}', '1.0000×', '1.0000\n(100.0%)', 'Reference'])
        row_colors.append([C['serial'] + '33'] * 8)

    for sched in sorted(omp_perf.keys()):
        best = min(omp_perf[sched], key=lambda d: d[1])
        rows_data.append([f'OpenMP ({sched})', '1', str(best[0]), str(best[0]),
                          f'{best[1]:.6f}', _sp(best[1]), _eff(best[1], best[0]),
                          f'RMSE={rmse_omp:.4f}' if rmse_omp is not None else 'N/A'])
        row_colors.append([C[sched] + '33'] * 8)

    if mpi_records:
        for r in mpi_records:
            rows_data.append(['MPI', str(r['np']), '1', str(r['np']),
                              f'{r["time"]:.6f}', _sp(r['time']), _eff(r['time'], r['np']),
                              f'RMSE={rmse_mpi:.4f}' if rmse_mpi is not None else 'N/A'])
            row_colors.append([C['mpi'] + '33'] * 8)

    if hyb_records:
        for r in hyb_records:
            rows_data.append(['Hybrid MPI+OMP', str(r['np']), str(r['nt']),
                              str(r['total']), f'{r["time"]:.6f}',
                              _sp(r['time']), _eff(r['time'], r['total']),
                              f'RMSE={rmse_hyb:.4f}' if rmse_hyb is not None else 'N/A'])
            row_colors.append([C['hybrid'] + '33'] * 8)

    if rows_data:
        tbl = ax_tbl.table(
            cellText=rows_data,
            colLabels=col_labels,
            cellColours=row_colors,
            cellLoc='center',
            loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.scale(1.0, 1.6)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_facecolor('#eeeeee')
                cell.set_text_props(fontweight='bold')

    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")


# =============================================================================
# Main
# =============================================================================

def main():
    print('\n' + '=' * 65)
    print('  Hybrid MPI+OpenMP Traffic Simulation — 4-Way Analysis')
    print('  Group 10  |  EE7218/EC7207 High Performance Computing')
    print('=' * 65)

    # ── Load all data ─────────────────────────────────────────────
    print('\n[Loading Data]')
    serial_grid, *_ = load_traffic_values(SERIAL_VALUES)
    omp_grid,    *_ = load_traffic_values(OMP_VALUES)
    mpi_grid,    *_ = load_traffic_values(MPI_VALUES)
    hyb_grid,    *_ = load_traffic_values(HYB_VALUES)
    serial_time     = parse_serial_time(SERIAL_OUTPUT)
    omp_perf        = parse_omp_performance(OMP_PERF)
    mpi_records     = parse_mpi_performance(MPI_PERF)
    hyb_records     = parse_hybrid_performance(HYB_PERF)

    def _shape(g): return str(g.shape) if g is not None else 'NOT FOUND'
    print(f'  Serial grid   : {_shape(serial_grid)}')
    print(f'  OpenMP grid   : {_shape(omp_grid)}')
    print(f'  MPI grid      : {_shape(mpi_grid)}')
    print(f'  Hybrid grid   : {_shape(hyb_grid)}')
    print(f'  Serial time   : ' +
          (f'{serial_time:.6f} s' if serial_time else 'NOT FOUND'))
    print(f'  OMP schedules : {list(omp_perf.keys())}')
    print(f'  MPI configs   : {[r["np"] for r in mpi_records]}')
    print(f'  Hybrid configs: {[r["label"] for r in hyb_records]}')

    # ── Accuracy ──────────────────────────────────────────────────
    print('\n[Accuracy Analysis  (Serial = reference)]')
    rmse_omp, _, _ = compute_accuracy(serial_grid, omp_grid, 'Serial vs OpenMP')
    rmse_mpi, _, _ = compute_accuracy(serial_grid, mpi_grid, 'Serial vs MPI   ')
    rmse_hyb, _, _ = compute_accuracy(serial_grid, hyb_grid, 'Serial vs Hybrid')

    for lbl, rmse in [('OpenMP', rmse_omp), ('MPI', rmse_mpi), ('Hybrid', rmse_hyb)]:
        if rmse is not None:
            v = 'PASS (match)' if rmse < 0.5 else 'CHECK seeds'
            print(f'  Serial vs {lbl:<6}: RMSE={rmse:.6f}  [{v}]')

    # ── Console performance summary ───────────────────────────────
    if serial_time:
        print(f'\n[Performance Summary  |  Serial baseline: {serial_time:.6f} s]')

        print('\n  OpenMP:')
        print(f'  {"Sched":<10} {"Threads":<8} {"Time(s)":<14} {"Speedup":<10} {"Eff":<8}')
        print(f'  {"-"*50}')
        for sched, data in sorted(omp_perf.items()):
            best = min(data, key=lambda d: d[1])
            sp  = serial_time / best[1]
            eff = sp / best[0]
            print(f'  {sched:<10} {best[0]:<8} {best[1]:<14.6f} {sp:<10.4f} {eff:<8.4f}')

        if mpi_records:
            print('\n  MPI:')
            print(f'  {"Procs":<8} {"Time(s)":<14} {"Speedup":<10} {"Eff":<8}')
            print(f'  {"-"*40}')
            for r in mpi_records:
                sp  = serial_time / r['time']
                eff = sp / r['np']
                print(f'  {r["np"]:<8} {r["time"]:<14.6f} {sp:<10.4f} {eff:<8.4f}')

        if hyb_records:
            print('\n  Hybrid MPI+OpenMP:')
            print(f'  {"Config":<12} {"Total":<8} {"Time(s)":<14} {"Speedup":<10} {"Eff":<8}')
            print(f'  {"-"*52}')
            for r in hyb_records:
                sp  = serial_time / r['time']
                eff = sp / r['total']
                print(f'  {r["label"]:<12} {r["total"]:<8} {r["time"]:<14.6f} {sp:<10.4f} {eff:<8.4f}')

        # Best of each
        print('\n  === Best Configuration Per Implementation ===')
        if serial_time:
            print(f'  Serial  : {serial_time:.6f} s  (1.0000×  baseline)')
        if omp_perf:
            all_omp = [(d[1], d[0], s) for s, data in omp_perf.items() for d in data]
            bt, bn, bs = min(all_omp, key=lambda x: x[0])
            print(f'  OpenMP  : {bt:.6f} s  ({serial_time/bt:.4f}×)  [{bn}T {bs}]')
        if mpi_records:
            br = min(mpi_records, key=lambda r: r['time'])
            print(f'  MPI     : {br["time"]:.6f} s  ({serial_time/br["time"]:.4f}×)  [{br["np"]}P]')
        if hyb_records:
            bh = min(hyb_records, key=lambda r: r['time'])
            print(f'  Hybrid  : {bh["time"]:.6f} s  ({serial_time/bh["time"]:.4f}×)  [{bh["label"]}]')

    # ── Generate all figures ──────────────────────────────────────
    print('\n[Generating Figures]')
    out = SCRIPT_DIR

    plot_4way_heatmaps(
        serial_grid, omp_grid, mpi_grid, hyb_grid,
        rmse_omp, rmse_mpi, rmse_hyb,
        outfile=os.path.join(out, 'hybrid_heatmap_4way.png'))

    plot_speedup_comparison(
        mpi_records, omp_perf, hyb_records, serial_time,
        outfile=os.path.join(out, 'hybrid_speedup_comparison.png'))

    plot_time_bars(
        mpi_records, omp_perf, hyb_records, serial_time,
        outfile=os.path.join(out, 'hybrid_time_bars.png'))

    plot_hybrid_scaling_surface(
        hyb_records, serial_time,
        outfile=os.path.join(out, 'hybrid_scaling_surface.png'))

    plot_full_report(
        serial_grid, omp_grid, mpi_grid, hyb_grid,
        mpi_records, omp_perf, hyb_records, serial_time,
        rmse_omp, rmse_mpi, rmse_hyb,
        outfile=os.path.join(out, 'hybrid_full_report.png'))

    # ── Output file status ────────────────────────────────────────
    print('\n[Output Files Status]')
    figs = [
        'hybrid_heatmap_4way.png',
        'hybrid_speedup_comparison.png',
        'hybrid_time_bars.png',
        'hybrid_scaling_surface.png',
        'hybrid_full_report.png',
    ]
    for fname in figs:
        path = os.path.join(out, fname)
        status = 'OK' if os.path.exists(path) else 'MISSING'
        print(f'  [{status}] {fname}')

    print('\n' + '=' * 65 + '\n')


if __name__ == '__main__':
    main()
