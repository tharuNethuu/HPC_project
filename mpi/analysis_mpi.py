"""
==============================================================
 analysis_mpi.py
 Serial / OpenMP / MPI Traffic Simulation — Full Comparison
 Group 10 | EE7218/EC7207 High Performance Computing

 Reads:
   ../serial_output.txt             -> serial execution time
   ../traffic_values.txt            -> serial final traffic grid
   ../openmp/openmp_performance.txt -> OpenMP timing (all schedules)
   ../openmp/openmp_traffic_values.txt -> OpenMP final grid
   mpi_performance.txt              -> MPI timing (all np values)
   mpi_traffic_values.txt           -> MPI final traffic grid

 Generates:
   mpi_heatmap_comparison.png   -> Serial / OpenMP / MPI side-by-side
   mpi_3way_comparison.png      -> Dedicated 3-way performance comparison
                                   (speedup vs serial baseline, efficiency,
                                    timing, summary table for all 3 impls)
   mpi_performance_analysis.png -> MPI detail + OMP overlay (4 panels)
   mpi_full_report.png          -> Combined 6-panel report for submission

 Run:
   python3 analysis_mpi.py
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
from matplotlib.colors import LinearSegmentedColormap

# ── File paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

SERIAL_OUTPUT  = os.path.join(SCRIPT_DIR, '..', 'serial_output.txt')
SERIAL_VALUES  = os.path.join(SCRIPT_DIR, '..', 'traffic_values.txt')
OMP_PERF       = os.path.join(SCRIPT_DIR, '..', 'openmp', 'openmp_performance.txt')
OMP_VALUES     = os.path.join(SCRIPT_DIR, '..', 'openmp', 'openmp_traffic_values.txt')
MPI_PERF       = os.path.join(SCRIPT_DIR, 'mpi_performance.txt')
MPI_VALUES     = os.path.join(SCRIPT_DIR, 'mpi_traffic_values.txt')


# ── Custom traffic heatmap colormap ──────────────────────────────────────────
def traffic_cmap():
    colors = ['#0000cc', '#0088ff', '#00ccaa', '#aaff00', '#ff8800', '#cc0000']
    return LinearSegmentedColormap.from_list('traffic', colors, N=256)

CMAP = traffic_cmap()

ROWS_G, COLS_G = 200, 200   # global grid dimensions


# =============================================================================
# I/O Helpers
# =============================================================================

def load_traffic_values(filepath):
    """
    Load a *_traffic_values.txt into a 2-D numpy array (rows x cols).
    Returns (array, rows, cols, lanes) or (None, 0, 0, 0) if not found.
    """
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
    if arr.shape != (rows, cols):
        print(f"  [WARN] Shape mismatch in {filepath}: "
              f"expected ({rows},{cols}) got {arr.shape}")
    return arr, rows, cols, lanes


def parse_serial_time(filepath):
    """Extract execution time (float) from serial_output.txt."""
    if not os.path.exists(filepath):
        print(f"  [WARN] Not found: {filepath}")
        return None
    with open(filepath, 'r') as f:
        content = f.read()
    m = re.search(r'Execution Time:\s*([\d.]+)\s*seconds', content)
    return float(m.group(1)) if m else None


def parse_mpi_performance(filepath):
    """
    Parse mpi_performance.txt.
    Data rows look like:
      '  1        0.050000  0.0000  99.0000  43.4092'
    Returns list of dicts:
      [{'np': int, 'time': float, 'min': float, 'max': float, 'avg': float}, ...]
    Speedup and efficiency are derived here relative to np=1.
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
            # Skip header / separator / footer lines
            if (line.startswith('=') or line.startswith('-')
                    or line.startswith('Grid') or line.startswith('[')
                    or line.startswith('Procs') or line.startswith('Speed')
                    or line.startswith('Effic') or line.startswith('MPI')
                    or line.startswith('Group')):
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

    # Compute speedup / efficiency relative to np=1
    base = next((r['time'] for r in records if r['np'] == 1), None)
    for r in records:
        r['speedup']    = base / r['time'] if base else 1.0
        r['efficiency'] = r['speedup'] / r['np']

    return records


def parse_omp_performance(filepath):
    """
    Parse openmp_performance.txt.
    Returns { schedule: [(threads, time, speedup, efficiency), ...] }
    """
    if not os.path.exists(filepath):
        print(f"  [WARN] Not found: {filepath}")
        return {}
    results = {}
    with open(filepath, 'r') as f:
        for raw in f:
            line = raw.strip()
            if (not line or line.startswith('=') or line.startswith('-')
                    or line.startswith('Grid') or line.startswith('[')
                    or line.startswith('Speedup') or line.startswith('Efficiency')
                    or line.startswith('MaxDiff') or line.startswith('Threads')
                    or line.startswith('OpenMP')):
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


# =============================================================================
# Accuracy
# =============================================================================

def compute_accuracy(ref_grid, cmp_grid, label=''):
    """Return (RMSE, MAE, max_abs_diff) between two 2-D grids."""
    if ref_grid is None or cmp_grid is None:
        return None, None, None
    r = min(ref_grid.shape[0], cmp_grid.shape[0])
    c = min(ref_grid.shape[1], cmp_grid.shape[1])
    diff     = ref_grid[:r, :c] - cmp_grid[:r, :c]
    rmse     = float(np.sqrt(np.mean(diff ** 2)))
    mae      = float(np.mean(np.abs(diff)))
    max_diff = float(np.max(np.abs(diff)))
    if label:
        print(f"  {label}: RMSE={rmse:.6f}  MAE={mae:.6f}  Max|Δ|={max_diff:.6f}")
    return rmse, mae, max_diff


# =============================================================================
# Figure 1: Side-by-side heatmap (Serial / OpenMP / MPI)
# =============================================================================

def plot_heatmaps(serial_grid, omp_grid, mpi_grid,
                  rmse_omp, rmse_mpi,
                  outfile='mpi_heatmap_comparison.png'):
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    fig.suptitle(
        'Traffic Density Heatmap Comparison — Serial | OpenMP | MPI\n'
        'Grid: 200×200  |  Lanes: 3  |  Time Steps: 200  |  Group 10',
        fontsize=13, fontweight='bold')

    x0, x1 = COLS_G // 3, 2 * COLS_G // 3
    y0, y1 = ROWS_G // 3, 2 * ROWS_G // 3

    configs = [
        (axes[0], serial_grid, '(a) Serial\n[Baseline — 1 Process]', None),
        (axes[1], omp_grid,    '(b) OpenMP\n[Best thread config]',   rmse_omp),
        (axes[2], mpi_grid,    '(c) MPI\n[Best process config]',     rmse_mpi),
    ]

    for ax, grid, title, rmse in configs:
        ax.set_title(title, fontsize=11, fontweight='bold')
        if grid is not None:
            im = ax.imshow(grid, cmap=CMAP, vmin=0, vmax=100,
                           aspect='equal', origin='upper')
            plt.colorbar(im, ax=ax, label='Traffic Density (0–100)',
                         fraction=0.046, pad=0.04)
            rect = mpatches.Rectangle((x0, y0), x1-x0, y1-y0,
                                       lw=2, edgecolor='white',
                                       facecolor='none', linestyle='--')
            ax.add_patch(rect)
            ax.text(x0 + 3, y0 + 9, 'Rain Region\nweather=0.7',
                    color='white', fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.55, pad=2))
            if rmse is not None:
                ax.annotate(f'vs Serial RMSE: {rmse:.6f}',
                            xy=(0.02, 0.03), xycoords='axes fraction',
                            fontsize=8.5, color='white', fontweight='bold',
                            bbox=dict(facecolor='black', alpha=0.65, pad=3))
        else:
            ax.text(0.5, 0.5, 'Output not found.\nRun the simulation first.',
                    ha='center', va='center', fontsize=11,
                    transform=ax.transAxes, color='gray')
            ax.set_facecolor('#f5f5f5')
        ax.set_xlabel('Column (Road Segment)', fontsize=9)
        ax.set_ylabel('Row (Road Segment)',    fontsize=9)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")


# =============================================================================
# Figure 2: Dedicated 3-way comparison (Serial | OpenMP all schedules | MPI)
# =============================================================================

# Colour / marker scheme shared by Figure 2 and Figure 3
_C = {
    'serial':   '#D32F2F',
    'static':   '#1976D2',
    'dynamic':  '#F57C00',
    'collapse': '#388E3C',
    'mpi':      '#7B1FA2',
    'ideal':    '#555555',
}
_M = {'static': 'o', 'dynamic': 's', 'collapse': '^', 'mpi': 'D'}


def _omp_speedup_vs_serial(omp_perf, serial_time):
    """Return {sched: (threads_list, speedup_vs_serial_list, efficiency_list)}."""
    out = {}
    if not serial_time:
        return out
    for sched, data in omp_perf.items():
        ts  = [d[0] for d in data]
        sp  = [serial_time / d[1] for d in data]
        eff = [serial_time / (d[1] * d[0]) for d in data]
        out[sched] = (ts, sp, eff)
    return out


def _mpi_speedup_vs_serial(mpi_records, serial_time):
    """Return (np_list, speedup_vs_serial_list, efficiency_list)."""
    if not serial_time or not mpi_records:
        return [], [], []
    np_vals = [r['np']   for r in mpi_records]
    sp      = [serial_time / r['time'] for r in mpi_records]
    eff     = [serial_time / (r['time'] * r['np']) for r in mpi_records]
    return np_vals, sp, eff


def plot_3way_comparison(mpi_records, omp_perf, serial_time,
                         outfile='mpi_3way_comparison.png'):
    """
    Dedicated 4-panel figure comparing all three implementations.
      (a) Execution time  — serial hline + all OMP schedules + MPI
      (b) Speedup         — T_serial / T_N  for OMP and MPI
      (c) Parallel efficiency — (T_serial / T_N) / N
      (d) Summary table   — all configs in one view
    """
    thread_configs = [1, 2, 4, 8]

    omp_sp_data  = _omp_speedup_vs_serial(omp_perf, serial_time)
    mpi_np, mpi_sp, mpi_eff = _mpi_speedup_vs_serial(mpi_records, serial_time)
    mpi_times = [r['time'] for r in mpi_records]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        'Serial  |  OpenMP  |  MPI — 3-Way Performance Comparison\n'
        'Group 10 | Grid: 200×200 | Lanes: 3 | Time Steps: 200',
        fontsize=13, fontweight='bold')

    ax_t, ax_sp = axes[0]
    ax_ef, ax_tb = axes[1]

    # ── (a) Execution Time ───────────────────────────────────────────────────
    ax_t.set_title('(a) Execution Time vs Parallelism Level',
                   fontsize=10, fontweight='bold')
    if serial_time:
        ax_t.axhline(serial_time, color=_C['serial'], lw=2.5, ls='--',
                     label=f'Serial  ({serial_time:.4f} s)', zorder=5)
    for sched, data in sorted(omp_perf.items()):
        ts = [d[0] for d in data]
        tm = [d[1] for d in data]
        ax_t.plot(ts, tm, color=_C[sched], marker=_M[sched],
                  lw=2, ms=8, label=f'OMP {sched}')
        for x, y in zip(ts, tm):
            ax_t.annotate(f'{y:.4f}', (x, y),
                          textcoords='offset points', xytext=(4, 3),
                          fontsize=7, color=_C[sched])
    if mpi_np:
        ax_t.plot(mpi_np, mpi_times, color=_C['mpi'], marker=_M['mpi'],
                  lw=2.5, ms=10, label='MPI', zorder=4)
        for x, y in zip(mpi_np, mpi_times):
            ax_t.annotate(f'{y:.4f}', (x, y),
                          textcoords='offset points', xytext=(4, -12),
                          fontsize=7, color=_C['mpi'])
    ax_t.set_xlabel('Threads (OMP) / Processes (MPI)', fontsize=9)
    ax_t.set_ylabel('Execution Time (s)', fontsize=9)
    ax_t.set_xticks(thread_configs)
    ax_t.set_xlim(0.5, 9)
    ax_t.legend(fontsize=8)
    ax_t.grid(True, alpha=0.3)

    # ── (b) Speedup vs Serial Baseline ───────────────────────────────────────
    ax_sp.set_title('(b) Speedup vs Serial Baseline  (T_serial / T_N)',
                    fontsize=10, fontweight='bold')
    ax_sp.plot(thread_configs, thread_configs, color=_C['ideal'], lw=1.5,
               ls='--', alpha=0.55, label='Ideal (linear)')
    ax_sp.axhline(1.0, color=_C['serial'], lw=1.5, ls=':',
                  alpha=0.6, label='Serial baseline  (1×)')
    for sched, (ts, sp, _) in sorted(omp_sp_data.items()):
        ax_sp.plot(ts, sp, color=_C[sched], marker=_M[sched],
                   lw=2, ms=8, label=f'OMP {sched}')
        for x, y in zip(ts, sp):
            ax_sp.annotate(f'{y:.2f}×', (x, y),
                           textcoords='offset points', xytext=(4, 3), fontsize=7)
    if mpi_np:
        ax_sp.plot(mpi_np, mpi_sp, color=_C['mpi'], marker=_M['mpi'],
                   lw=2.5, ms=10, label='MPI', zorder=4)
        for x, y in zip(mpi_np, mpi_sp):
            ax_sp.annotate(f'{y:.2f}×', (x, y),
                           textcoords='offset points', xytext=(4, -12), fontsize=7)
    ax_sp.set_xlabel('Threads (OMP) / Processes (MPI)', fontsize=9)
    ax_sp.set_ylabel('Speedup  =  T_serial / T_N', fontsize=9)
    ax_sp.set_xticks(thread_configs)
    ax_sp.set_xlim(0.5, 9)
    ax_sp.legend(fontsize=8)
    ax_sp.grid(True, alpha=0.3)

    # ── (c) Efficiency vs Serial Baseline ────────────────────────────────────
    ax_ef.set_title('(c) Parallel Efficiency  (T_serial / (N × T_N))',
                    fontsize=10, fontweight='bold')
    ax_ef.axhline(1.0, color=_C['ideal'], lw=1.5, ls='--',
                  alpha=0.55, label='Ideal (100%)')
    for sched, (ts, _, eff) in sorted(omp_sp_data.items()):
        ax_ef.plot(ts, eff, color=_C[sched], marker=_M[sched],
                   lw=2, ms=8, label=f'OMP {sched}')
        for x, y in zip(ts, eff):
            ax_ef.annotate(f'{y:.2f}\n({y*100:.0f}%)', (x, y),
                           textcoords='offset points', xytext=(0, 8),
                           ha='center', fontsize=7)
    if mpi_np:
        ax_ef.plot(mpi_np, mpi_eff, color=_C['mpi'], marker=_M['mpi'],
                   lw=2.5, ms=10, label='MPI', zorder=4)
        for x, y in zip(mpi_np, mpi_eff):
            ax_ef.annotate(f'{y:.2f}\n({y*100:.0f}%)', (x, y),
                           textcoords='offset points', xytext=(0, -18),
                           ha='center', fontsize=7)
    ax_ef.set_xlabel('Threads (OMP) / Processes (MPI)', fontsize=9)
    ax_ef.set_ylabel('Efficiency  =  T_serial / (N × T_N)', fontsize=9)
    ax_ef.set_xticks(thread_configs)
    ax_ef.set_ylim(0, 1.5)
    ax_ef.set_xlim(0.5, 9)
    ax_ef.legend(fontsize=8)
    ax_ef.grid(True, alpha=0.3)

    # ── (d) Summary comparison table ─────────────────────────────────────────
    ax_tb.set_title('(d) 3-Way Summary Table', fontsize=10, fontweight='bold')
    ax_tb.axis('off')

    rows = []
    if serial_time:
        rows.append(('Serial', 1, f'{serial_time:.6f}', '1.0000×', '1.0000',
                     _C['serial']))
    for sched in sorted(omp_perf.keys()):
        for nt, tm, _, _ in omp_perf[sched]:
            sp  = serial_time / tm if serial_time else float('nan')
            eff = sp / nt
            rows.append((f'OMP {sched}', nt, f'{tm:.6f}',
                         f'{sp:.4f}×', f'{eff:.4f}', _C[sched]))
    for r in mpi_records:
        sp  = serial_time / r['time'] if serial_time else float('nan')
        eff = sp / r['np']
        rows.append(('MPI', r['np'], f'{r["time"]:.6f}',
                     f'{sp:.4f}×', f'{eff:.4f}', _C['mpi']))

    col_labels = ['Impl', 'N', 'Time (s)', 'Speedup\nvs Serial', 'Efficiency']
    cell_text  = [[r[0], r[1], r[2], r[3], r[4]] for r in rows]
    cell_colors = [[r[5] + '22'] * 5 for r in rows]   # light tint

    tbl = ax_tb.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc='center',
        loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.15, 1.5)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#eeeeee')
            cell.set_text_props(fontweight='bold')

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")


# =============================================================================
# Figure 3: MPI-only detail analysis (4 panels)
# =============================================================================

def plot_performance(mpi_records, serial_time, omp_perf,
                     outfile='mpi_performance_analysis.png'):
    """4-panel MPI detail view with all OMP schedules overlaid."""
    if not mpi_records:
        print("  [WARN] No MPI performance data — skipping performance plot.")
        return

    np_vals  = [r['np']   for r in mpi_records]
    times    = [r['time'] for r in mpi_records]
    # Internal MPI speedup (T_mpi_1 / T_mpi_n)
    int_sp   = [r['speedup']    for r in mpi_records]
    int_eff  = [r['efficiency'] for r in mpi_records]
    # Cross-impl speedup (T_serial / T_mpi_n)
    ser_sp   = ([serial_time / t for t in times] if serial_time else int_sp)
    ser_eff  = ([serial_time / (t * n) for t, n in zip(times, np_vals)]
                if serial_time else int_eff)

    thread_configs = [1, 2, 4, 8]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        'MPI Traffic Simulation — Detailed Performance Analysis\n'
        'Group 10 | Grid: 200×200 | Lanes: 3 | Time Steps: 200',
        fontsize=13, fontweight='bold')

    ax_t, ax_sp = axes[0]
    ax_ef, ax_sm = axes[1]

    # ── (a) Execution Time — all 3 implementations ───────────────────────────
    ax_t.set_title('(a) Execution Time  [Serial | OpenMP | MPI]',
                   fontsize=10, fontweight='bold')
    if serial_time:
        ax_t.axhline(serial_time, color=_C['serial'], lw=2.5, ls='--',
                     label=f'Serial  ({serial_time:.4f} s)', zorder=5)
    for sched, data in sorted(omp_perf.items()):
        ts = [d[0] for d in data]
        tm = [d[1] for d in data]
        ax_t.plot(ts, tm, color=_C[sched], marker=_M[sched],
                  lw=1.8, ms=7, ls=':', label=f'OMP {sched}')
    ax_t.plot(np_vals, times, color=_C['mpi'], marker=_M['mpi'],
              lw=2.5, ms=10, label='MPI', zorder=4)
    for x, y in zip(np_vals, times):
        ax_t.annotate(f'{y:.4f}s', (x, y),
                      textcoords='offset points', xytext=(5, 4),
                      fontsize=8, color=_C['mpi'])
    ax_t.set_xlabel('Threads (OMP) / Processes (MPI)', fontsize=9)
    ax_t.set_ylabel('Execution Time (s)', fontsize=9)
    ax_t.set_xticks(thread_configs)
    ax_t.set_xlim(0.5, 9)
    ax_t.legend(fontsize=7.5)
    ax_t.grid(True, alpha=0.3)

    # ── (b) Speedup — both internal and vs serial ────────────────────────────
    ax_sp.set_title('(b) Speedup  [all 3 implementations vs serial baseline]',
                    fontsize=10, fontweight='bold')
    ax_sp.plot(thread_configs, thread_configs, color=_C['ideal'], lw=1.5,
               ls='--', alpha=0.55, label='Ideal (linear)')
    ax_sp.axhline(1.0, color=_C['serial'], lw=1.5, ls=':',
                  alpha=0.6, label='Serial (1×)')
    # All OMP schedules vs serial
    for sched, data in sorted(omp_perf.items()):
        ts = [d[0] for d in data]
        sp = [serial_time / d[1] for d in data] if serial_time else [d[2] for d in data]
        ax_sp.plot(ts, sp, color=_C[sched], marker=_M[sched],
                   lw=1.8, ms=7, ls=':', label=f'OMP {sched}')
    # MPI vs serial (solid) and MPI internal (dashed)
    ax_sp.plot(np_vals, ser_sp, color=_C['mpi'], marker=_M['mpi'],
               lw=2.5, ms=10, label='MPI vs serial', zorder=4)
    ax_sp.plot(np_vals, int_sp, color=_C['mpi'], marker=_M['mpi'],
               lw=1.5, ms=7, ls='--', alpha=0.6, label='MPI internal (T₁/Tₙ)')
    for x, y in zip(np_vals, ser_sp):
        ax_sp.annotate(f'{y:.2f}×', (x, y),
                       textcoords='offset points', xytext=(5, 4), fontsize=8)
    ax_sp.set_xlabel('Threads (OMP) / Processes (MPI)', fontsize=9)
    ax_sp.set_ylabel('Speedup  =  T_serial / T_N', fontsize=9)
    ax_sp.set_xticks(thread_configs)
    ax_sp.set_xlim(0.5, 9)
    ax_sp.legend(fontsize=7.5, ncol=2)
    ax_sp.grid(True, alpha=0.3)

    # ── (c) Efficiency — all 3 ───────────────────────────────────────────────
    ax_ef.set_title('(c) Parallel Efficiency  [all 3 implementations]',
                    fontsize=10, fontweight='bold')
    ax_ef.axhline(1.0, color=_C['ideal'], lw=1.5, ls='--',
                  alpha=0.55, label='Ideal (100%)')
    for sched, data in sorted(omp_perf.items()):
        ts  = [d[0] for d in data]
        eff = ([serial_time / (d[1] * d[0]) for d in data]
               if serial_time else [d[3] for d in data])
        ax_ef.plot(ts, eff, color=_C[sched], marker=_M[sched],
                   lw=1.8, ms=7, ls=':', label=f'OMP {sched}')
    ax_ef.plot(np_vals, ser_eff, color=_C['mpi'], marker=_M['mpi'],
               lw=2.5, ms=10, label='MPI', zorder=4)
    for x, y in zip(np_vals, ser_eff):
        ax_ef.annotate(f'{y:.2f}\n({y*100:.0f}%)', (x, y),
                       textcoords='offset points', xytext=(0, 9),
                       ha='center', fontsize=8)
    ax_ef.set_xlabel('Threads (OMP) / Processes (MPI)', fontsize=9)
    ax_ef.set_ylabel('Efficiency  =  T_serial / (N × T_N)', fontsize=9)
    ax_ef.set_xticks(thread_configs)
    ax_ef.set_ylim(0, 1.5)
    ax_ef.set_xlim(0.5, 9)
    ax_ef.legend(fontsize=8)
    ax_ef.grid(True, alpha=0.3)

    # ── (d) MPI summary text ──────────────────────────────────────────────────
    ax_sm.set_title('(d) MPI Performance Summary', fontsize=10, fontweight='bold')
    ax_sm.axis('off')

    best_r = min(mpi_records, key=lambda r: r['time']) if mpi_records else None
    lines = [
        '┌────────────────────────────────────────────────────┐',
        '│   MPI PERFORMANCE SUMMARY   (Group 10)            │',
        '├──────────┬────────────┬────────────┬──────────────┤',
        '│  Procs   │  Time (s)  │  Sp (int.) │  Sp (serial) │',
        '├──────────┼────────────┼────────────┼──────────────┤',
    ]
    for r, ss in zip(mpi_records, ser_sp):
        lines.append(
            f'│  {r["np"]:<7d}│  {r["time"]:<10.6f}│  '
            f'{r["speedup"]:<10.4f}│  {ss:<12.4f}│')
    lines.append('└──────────┴────────────┴────────────┴──────────────┘')
    if serial_time and best_r:
        best_ser_sp = serial_time / best_r['time']
        lines += [
            '',
            f'  Serial baseline   : {serial_time:.6f} s',
            f'  MPI best          : {best_r["time"]:.6f} s  (np={best_r["np"]})',
            f'  Speedup vs serial : {best_ser_sp:.4f} ×',
            '',
        ]
    lines += [
        '  MPI design notes:',
        '  • 1-D row decomposition across processes',
        '  • Non-blocking halo exchange (MPI_Isend/Irecv)',
        '  • MPI_Scatterv / MPI_Gatherv for data I/O',
        '  • MPI_Reduce for global statistics',
        '  • MPI_Barrier + MPI_Wtime for accurate timing',
    ]
    ax_sm.text(0.02, 0.97, '\n'.join(lines),
               transform=ax_sm.transAxes,
               fontsize=8, verticalalignment='top',
               fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.4',
                         facecolor='#f3e5f5', edgecolor='#7B1FA2', lw=1.5))

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")


# =============================================================================
# Figure 4: Full 6-panel Report
# =============================================================================

def plot_full_report(serial_grid, omp_grid, mpi_grid,
                     mpi_records, omp_perf, serial_time,
                     rmse_omp, rmse_mpi,
                     outfile='mpi_full_report.png'):

    np_vals  = [r['np']   for r in mpi_records] if mpi_records else []
    times    = [r['time'] for r in mpi_records] if mpi_records else []
    speedups = [r['speedup']    for r in mpi_records] if mpi_records else []
    effics   = [r['efficiency'] for r in mpi_records] if mpi_records else []

    thread_configs = [1, 2, 4, 8]

    x0, x1 = COLS_G // 3, 2 * COLS_G // 3
    y0, y1 = ROWS_G // 3, 2 * ROWS_G // 3

    fig = plt.figure(figsize=(18, 24))
    fig.suptitle(
        'Parallel Traffic Density Simulation — Full Analysis Report\n'
        'Serial  |  OpenMP  |  MPI  |  Group 10  |  EE7218/EC7207',
        fontsize=15, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.3,
                           top=0.95, bottom=0.04, left=0.07, right=0.97)

    ax_sh  = fig.add_subplot(gs[0, 0])   # (a) Serial heatmap
    ax_mh  = fig.add_subplot(gs[0, 1])   # (b) MPI heatmap
    ax_sp  = fig.add_subplot(gs[1, 0])   # (c) Speedup
    ax_eff = fig.add_subplot(gs[1, 1])   # (d) Efficiency
    ax_bar = fig.add_subplot(gs[2, 0])   # (e) Timing bar chart
    ax_sum = fig.add_subplot(gs[2, 1])   # (f) Summary text

    # ── (a) Serial Heatmap ───────────────────────────────────────────────────
    ax_sh.set_title('(a) Serial — Final Traffic Density', fontsize=11, fontweight='bold')
    if serial_grid is not None:
        im = ax_sh.imshow(serial_grid, cmap=CMAP, vmin=0, vmax=100,
                          aspect='equal', origin='upper')
        plt.colorbar(im, ax=ax_sh, label='Density', fraction=0.046, pad=0.04)
        ax_sh.add_patch(mpatches.Rectangle((x0, y0), x1-x0, y1-y0,
                        lw=2, edgecolor='white', facecolor='none', ls='--'))
        ax_sh.text(x0+3, y0+9, 'Rain\n(0.7×)', color='white', fontsize=8,
                   bbox=dict(facecolor='#000000aa', pad=2))
    else:
        ax_sh.text(0.5, 0.5, 'Not found', ha='center', va='center',
                   transform=ax_sh.transAxes, color='gray')
    ax_sh.set_xlabel('Column'); ax_sh.set_ylabel('Row')

    # ── (b) MPI Heatmap ──────────────────────────────────────────────────────
    ax_mh.set_title('(b) MPI — Final Traffic Density', fontsize=11, fontweight='bold')
    if mpi_grid is not None:
        im = ax_mh.imshow(mpi_grid, cmap=CMAP, vmin=0, vmax=100,
                          aspect='equal', origin='upper')
        plt.colorbar(im, ax=ax_mh, label='Density', fraction=0.046, pad=0.04)
        ax_mh.add_patch(mpatches.Rectangle((x0, y0), x1-x0, y1-y0,
                        lw=2, edgecolor='white', facecolor='none', ls='--'))
        ax_mh.text(x0+3, y0+9, 'Rain\n(0.7×)', color='white', fontsize=8,
                   bbox=dict(facecolor='#000000aa', pad=2))
        if rmse_mpi is not None:
            ax_mh.annotate(f'vs Serial RMSE: {rmse_mpi:.6f}',
                           xy=(0.02, 0.03), xycoords='axes fraction',
                           fontsize=9, color='white', fontweight='bold',
                           bbox=dict(facecolor='black', alpha=0.65, pad=3))
    else:
        ax_mh.text(0.5, 0.5, 'Not found\nRun: mpirun -np 4 ./traffic_mpi',
                   ha='center', va='center', transform=ax_mh.transAxes, color='gray')
    ax_mh.set_xlabel('Column'); ax_mh.set_ylabel('Row')

    # ── (c) Speedup vs serial — all 3 implementations ───────────────────────
    ax_sp.set_title('(c) Speedup vs Serial Baseline  (T_serial / T_N)',
                    fontsize=11, fontweight='bold')
    ax_sp.plot(thread_configs, thread_configs, color=_C['ideal'], lw=1.5,
               ls='--', alpha=0.55, label='Ideal (linear)')
    ax_sp.axhline(1.0, color=_C['serial'], lw=1.5, ls=':',
                  alpha=0.6, label='Serial (1×)')

    for sched, data in sorted(omp_perf.items()):
        ts = [d[0] for d in data]
        sp = ([serial_time / d[1] for d in data]
              if serial_time else [d[2] for d in data])
        ax_sp.plot(ts, sp, color=_C[sched], marker=_M[sched],
                   lw=1.8, ms=7, ls=':', label=f'OMP {sched}')

    if np_vals:
        mpi_ser_sp = ([serial_time / t for t in times]
                      if serial_time else speedups)
        ax_sp.plot(np_vals, mpi_ser_sp, color=_C['mpi'], marker=_M['mpi'],
                   lw=2.5, ms=9, label='MPI', zorder=4)
        for x, y in zip(np_vals, mpi_ser_sp):
            ax_sp.annotate(f'{y:.2f}×', (x, y),
                           textcoords='offset points', xytext=(5, 4), fontsize=8.5)

    ax_sp.set_xlabel('Processes (MPI) / Threads (OMP)', fontsize=10)
    ax_sp.set_ylabel('Speedup  =  T_serial / T_N', fontsize=10)
    ax_sp.set_xticks(thread_configs)
    ax_sp.legend(fontsize=8, ncol=2)
    ax_sp.grid(True, alpha=0.3)
    ax_sp.set_xlim(0.5, 9)

    # ── (d) Efficiency — all 3 ────────────────────────────────────────────────
    ax_eff.set_title('(d) Parallel Efficiency  (T_serial / (N × T_N))',
                     fontsize=11, fontweight='bold')
    ax_eff.axhline(1.0, color=_C['ideal'], lw=1.5, ls='--', alpha=0.55,
                   label='Ideal (100%)')

    for sched, data in sorted(omp_perf.items()):
        ts  = [d[0] for d in data]
        eff = ([serial_time / (d[1] * d[0]) for d in data]
               if serial_time else [d[3] for d in data])
        ax_eff.plot(ts, eff, color=_C[sched], marker=_M[sched],
                    lw=1.8, ms=7, ls=':', label=f'OMP {sched}')

    if np_vals:
        mpi_ser_eff = ([serial_time / (t * n) for t, n in zip(times, np_vals)]
                       if serial_time else effics)
        ax_eff.plot(np_vals, mpi_ser_eff, color=_C['mpi'], marker=_M['mpi'],
                    lw=2.5, ms=9, label='MPI', zorder=4)
        for x, y in zip(np_vals, mpi_ser_eff):
            ax_eff.annotate(f'{y:.2f}\n({y*100:.0f}%)', (x, y),
                            textcoords='offset points', xytext=(0, 9),
                            ha='center', fontsize=8)

    ax_eff.set_xlabel('Processes (MPI) / Threads (OMP)', fontsize=10)
    ax_eff.set_ylabel('Efficiency  =  T_serial / (N × T_N)', fontsize=10)
    ax_eff.set_xticks(thread_configs)
    ax_eff.set_ylim(0, 1.5)
    ax_eff.legend(fontsize=8, ncol=2)
    ax_eff.grid(True, alpha=0.3)
    ax_eff.set_xlim(0.5, 9)

    # ── (e) Timing Bar Chart — serial + ALL OMP schedules + MPI ──────────────
    ax_bar.set_title('(e) Execution Time — All Configurations  [Serial | OMP | MPI]',
                     fontsize=11, fontweight='bold')
    labels, bar_times, bar_colors = [], [], []

    if serial_time:
        labels.append('Serial\n1T'); bar_times.append(serial_time)
        bar_colors.append(_C['serial'])

    for sched in sorted(omp_perf.keys()):
        for nt, tm, _, _ in omp_perf[sched]:
            labels.append(f'OMP\n{sched}\n{nt}T')
            bar_times.append(tm); bar_colors.append(_C[sched])

    for r in mpi_records:
        labels.append(f'MPI\n{r["np"]}P')
        bar_times.append(r['time']); bar_colors.append(_C['mpi'])

    x_pos = range(len(labels))
    bars  = ax_bar.bar(x_pos, bar_times, color=bar_colors,
                       edgecolor='white', lw=0.8, width=0.65)
    ax_bar.set_xticks(list(x_pos))
    ax_bar.set_xticklabels(labels, fontsize=7.5)
    ax_bar.set_ylabel('Execution Time (seconds)', fontsize=10)
    ax_bar.set_xlabel('Configuration', fontsize=10)
    ax_bar.grid(axis='y', alpha=0.3)

    max_t = max(bar_times) if bar_times else 1
    for bar, t in zip(bars, bar_times):
        ax_bar.text(bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + max_t * 0.005,
                    f'{t:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)

    leg_patches = [
        mpatches.Patch(color=_C['serial'],   label='Serial'),
        mpatches.Patch(color=_C['static'],   label='OMP static'),
        mpatches.Patch(color=_C['dynamic'],  label='OMP dynamic'),
        mpatches.Patch(color=_C['collapse'], label='OMP collapse'),
        mpatches.Patch(color=_C['mpi'],      label='MPI'),
    ]
    ax_bar.legend(handles=leg_patches, fontsize=7.5, loc='upper right', ncol=2)

    # ── (f) Summary Text ─────────────────────────────────────────────────────
    ax_sum.set_title('(f) Analysis Summary', fontsize=11, fontweight='bold')
    ax_sum.axis('off')

    best_mpi = min(mpi_records, key=lambda r: r['time']) if mpi_records else None

    summary = [
        '  EE7218/EC7207 — High Performance Computing',
        '  Group 10 | Traffic Density Simulation',
        '',
        '  Accuracy (Serial = reference)',
        '  ──────────────────────────────────────────',
    ]
    if rmse_omp is not None:
        v = 'PASS' if rmse_omp < 0.5 else 'CHECK'
        summary.append(f'  Serial vs OMP : RMSE={rmse_omp:.6f}  [{v}]')
    if rmse_mpi is not None:
        v = 'PASS' if rmse_mpi < 0.5 else 'CHECK'
        summary.append(f'  Serial vs MPI : RMSE={rmse_mpi:.6f}  [{v}]')
    summary += [
        '',
        '  Best times (vs serial baseline)',
        '  ──────────────────────────────────────────',
    ]
    if serial_time:
        summary.append(f'  Serial        : {serial_time:.6f} s  (1.0000×)')
    for sched in sorted(omp_perf.keys()):
        best_o = min(omp_perf[sched], key=lambda d: d[1])
        sp = serial_time / best_o[1] if serial_time else best_o[2]
        summary.append(
            f'  OMP {sched:<9}: {best_o[1]:.6f} s  '
            f'({sp:.4f}×)  [{best_o[0]}T]')
    if serial_time and best_mpi:
        sp = serial_time / best_mpi['time']
        summary.append(
            f'  MPI           : {best_mpi["time"]:.6f} s  '
            f'({sp:.4f}×)  [{best_mpi["np"]}P]')
    summary += [
        '',
        '  MPI design notes:',
        '  ──────────────────────────────────────────',
        '  • 1-D row decomposition across processes',
        '  • Non-blocking halo exchange (MPI_Isend/Irecv)',
        '  • MPI_Scatterv / MPI_Gatherv for data I/O',
        '  • MPI_Reduce (global min / max / avg)',
        '  • MPI_Barrier + MPI_Wtime for timing',
    ]

    ax_sum.text(0.02, 0.98, '\n'.join(summary),
                transform=ax_sum.transAxes,
                fontsize=8, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.6',
                          facecolor='#f3e5f5', edgecolor='#7B1FA2', lw=1.5))

    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outfile}")


# =============================================================================
# Main
# =============================================================================

def main():
    print('\n' + '='*60)
    print('  MPI Traffic Simulation — Performance Analysis  |  Group 10')
    print('='*60)

    # ── Load data ─────────────────────────────────────────────────────────────
    print('\n[Loading Data]')
    serial_grid, *_ = load_traffic_values(SERIAL_VALUES)
    omp_grid,    *_ = load_traffic_values(OMP_VALUES)
    mpi_grid,    *_ = load_traffic_values(MPI_VALUES)
    serial_time     = parse_serial_time(SERIAL_OUTPUT)
    mpi_records     = parse_mpi_performance(MPI_PERF)
    omp_perf        = parse_omp_performance(OMP_PERF)

    def _shape(g): return str(g.shape) if g is not None else 'NOT FOUND'
    print(f'  Serial grid     : {_shape(serial_grid)}')
    print(f'  OpenMP grid     : {_shape(omp_grid)}')
    print(f'  MPI grid        : {_shape(mpi_grid)}')
    print(f'  Serial time     : '
          + (f'{serial_time:.6f} s' if serial_time else 'NOT FOUND'))
    print(f'  MPI configs     : {[r["np"] for r in mpi_records]}')
    print(f'  OMP schedules   : {list(omp_perf.keys())}')

    # ── Accuracy ──────────────────────────────────────────────────────────────
    print('\n[Accuracy Analysis]')
    rmse_omp, mae_omp, max_omp = compute_accuracy(serial_grid, omp_grid, 'Serial vs OMP')
    rmse_mpi, mae_mpi, max_mpi = compute_accuracy(serial_grid, mpi_grid, 'Serial vs MPI')

    for label, rmse in [('Serial vs OMP', rmse_omp), ('Serial vs MPI', rmse_mpi)]:
        if rmse is not None:
            verdict = '✓ match' if rmse < 0.5 else '⚠  check seeds'
            print(f'  {label}: {verdict}')

    # ── Performance console summary ───────────────────────────────────────────
    if mpi_records:
        print('\n[MPI Scalability]')
        print(f'  {"Procs":<8} {"Time(s)":<14} {"Speedup":<10} {"Efficiency":<12}')
        print(f'  {"-"*44}')
        for r in mpi_records:
            print(f'  {r["np"]:<8d} {r["time"]:<14.6f} '
                  f'{r["speedup"]:<10.4f} {r["efficiency"]:<12.4f}')

    # ── Generate plots ────────────────────────────────────────────────────────
    print('\n[Generating Figures]')
    out = SCRIPT_DIR

    plot_heatmaps(
        serial_grid, omp_grid, mpi_grid,
        rmse_omp, rmse_mpi,
        outfile=os.path.join(out, 'mpi_heatmap_comparison.png'))

    plot_3way_comparison(
        mpi_records, omp_perf, serial_time,
        outfile=os.path.join(out, 'mpi_3way_comparison.png'))

    plot_performance(
        mpi_records, serial_time, omp_perf,
        outfile=os.path.join(out, 'mpi_performance_analysis.png'))

    plot_full_report(
        serial_grid, omp_grid, mpi_grid,
        mpi_records, omp_perf, serial_time,
        rmse_omp, rmse_mpi,
        outfile=os.path.join(out, 'mpi_full_report.png'))

    # ── Output file status ────────────────────────────────────────────────────
    print('\n[Output Files]')
    for fname in ['mpi_heatmap_comparison.png',
                  'mpi_3way_comparison.png',
                  'mpi_performance_analysis.png',
                  'mpi_full_report.png']:
        path = os.path.join(out, fname)
        status = '✓' if os.path.exists(path) else '✗'
        print(f'  {status} {fname}')

    print('\n' + '='*60 + '\n')


if __name__ == '__main__':
    main()
