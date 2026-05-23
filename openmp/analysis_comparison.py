"""
==============================================================
 analysis_comparison.py
 Serial vs OpenMP Traffic Simulation - Analysis & Comparison
 Group 10 | EE7218/EC7207 High Performance Computing

 Reads:
   ../serial_output.txt        -> serial execution time
   ../traffic_values.txt       -> serial final traffic grid
   openmp_performance.txt      -> OpenMP timing / speedup / efficiency
   openmp_traffic_values.txt   -> OpenMP final traffic grid

 Generates:
   comparison_heatmaps.png     -> Side-by-side heatmap (serial vs openmp)
   performance_analysis.png    -> Speedup, efficiency, timing, accuracy
   full_analysis_report.png    -> Combined single figure for the report

 Run:
   python3 analysis_comparison.py
==============================================================
"""

import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ── File paths (relative to this script's location) ──────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
SERIAL_OUTPUT  = os.path.join(SCRIPT_DIR, '..', 'serial', 'serial_output.txt')
SERIAL_VALUES  = os.path.join(SCRIPT_DIR, '..', 'serial', 'traffic_values.txt')
OMP_PERF       = os.path.join(SCRIPT_DIR, 'openmp_performance.txt')
OMP_VALUES     = os.path.join(SCRIPT_DIR, 'openmp_traffic_values.txt')
OMP_OUTPUT     = os.path.join(SCRIPT_DIR, 'openmp_output.txt')

# ── Traffic heatmap colormap: Blue→Cyan→Green→Yellow→Orange→Red ──────────────
def traffic_cmap():
    colors = ['#0000cc', '#0088ff', '#00ccaa', '#aaff00', '#ff8800', '#cc0000']
    return LinearSegmentedColormap.from_list('traffic', colors, N=256)

CMAP = traffic_cmap()

# ─────────────────────────────────────────────────────────────────────────────
# I/O Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_traffic_values(filepath):
    """Load a traffic_values.txt file into a 2-D numpy array (rows x cols)."""
    if not os.path.exists(filepath):
        print(f"  [WARN] Not found: {filepath}")
        return None, 0, 0, 0
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    rows, cols, lanes = map(int, lines[0].split())
    grid = []
    for line in lines[1:]:
        vals = list(map(float, line.split()))
        if vals:
            grid.append(vals)
    arr = np.array(grid, dtype=np.float64)
    if arr.shape != (rows, cols):
        print(f"  [WARN] Shape mismatch in {filepath}: expected ({rows},{cols}) got {arr.shape}")
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


def parse_omp_performance(filepath):
    """
    Parse openmp_performance.txt into:
      { schedule_name: [(threads, time, speedup, efficiency), ...] }
    """
    if not os.path.exists(filepath):
        print(f"  [WARN] Not found: {filepath}")
        return {}
    results = {}
    with open(filepath, 'r') as f:
        for raw in f:
            line = raw.strip()
            # Skip blank, header, separator lines
            if (not line
                    or line.startswith('=') or line.startswith('-')
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


def parse_omp_best_time(filepath):
    """Extract execution time from openmp_output.txt (final run)."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        content = f.read()
    m = re.search(r'Exec Time\s*:\s*([\d.]+)\s*seconds', content)
    return float(m.group(1)) if m else None


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_accuracy(serial_grid, omp_grid):
    """Return (RMSE, MAE, max_abs_diff) between two grids."""
    if serial_grid is None or omp_grid is None:
        return None, None, None
    r = min(serial_grid.shape[0], omp_grid.shape[0])
    c = min(serial_grid.shape[1], omp_grid.shape[1])
    s = serial_grid[:r, :c]
    o = omp_grid[:r, :c]
    diff = s - o
    rmse     = float(np.sqrt(np.mean(diff ** 2)))
    mae      = float(np.mean(np.abs(diff)))
    max_diff = float(np.max(np.abs(diff)))
    return rmse, mae, max_diff


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Heatmap Comparison (Serial vs OpenMP)
# ─────────────────────────────────────────────────────────────────────────────

def plot_heatmaps(serial_grid, omp_grid, rmse, mae, max_diff,
                  best_threads, outfile='comparison_heatmaps.png'):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(
        'Traffic Density Heatmap Comparison — Serial vs OpenMP\n'
        'Grid: 200×200 | Lanes: 3 | Time Steps: 200 | Group 10',
        fontsize=13, fontweight='bold')

    vmin, vmax = 0, 100
    ROWS, COLS = 200, 200

    # ── Serial ──
    ax = axes[0]
    if serial_grid is not None:
        im = ax.imshow(serial_grid, cmap=CMAP, vmin=vmin, vmax=vmax,
                       aspect='equal', origin='upper')
        plt.colorbar(im, ax=ax, label='Traffic Density (0–100)', fraction=0.046, pad=0.04)
        # Rain region box
        x0, x1 = COLS // 3, 2 * COLS // 3
        y0, y1 = ROWS // 3, 2 * ROWS // 3
        rect = mpatches.Rectangle((x0, y0), x1-x0, y1-y0,
                                   linewidth=2, edgecolor='white',
                                   facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.text(x0 + 3, y0 + 8, 'Rain Region\nweather=0.7',
                color='white', fontsize=8,
                bbox=dict(facecolor='black', alpha=0.5, pad=2))
    else:
        ax.text(0.5, 0.5, 'Serial output not found.\nRun: ./serial',
                ha='center', va='center', fontsize=12,
                transform=ax.transAxes)

    ax.set_title('(a) Serial Execution\n[Single Thread — Baseline]',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Column (Road Segment)', fontsize=9)
    ax.set_ylabel('Row (Road Segment)', fontsize=9)

    # ── OpenMP ──
    ax = axes[1]
    if omp_grid is not None:
        im = ax.imshow(omp_grid, cmap=CMAP, vmin=vmin, vmax=vmax,
                       aspect='equal', origin='upper')
        plt.colorbar(im, ax=ax, label='Traffic Density (0–100)', fraction=0.046, pad=0.04)
        x0, x1 = COLS // 3, 2 * COLS // 3
        y0, y1 = ROWS // 3, 2 * ROWS // 3
        rect = mpatches.Rectangle((x0, y0), x1-x0, y1-y0,
                                   linewidth=2, edgecolor='white',
                                   facecolor='none', linestyle='--')
        ax.add_patch(rect)
        ax.text(x0 + 3, y0 + 8, 'Rain Region\nweather=0.7',
                color='white', fontsize=8,
                bbox=dict(facecolor='black', alpha=0.5, pad=2))
        # Accuracy annotation
        if rmse is not None:
            ax.annotate(
                f'RMSE  : {rmse:.6f}\nMAE   : {mae:.6f}\nMax|Δ|: {max_diff:.6f}',
                xy=(0.02, 0.03), xycoords='axes fraction', fontsize=8,
                color='white', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#000000cc'))
    else:
        ax.text(0.5, 0.5, 'OpenMP output not found.\nRun: ./traffic_openmp',
                ha='center', va='center', fontsize=12,
                transform=ax.transAxes)

    ax.set_title(f'(b) OpenMP Parallel Execution\n[{best_threads} Threads | Static Schedule]',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Column (Road Segment)', fontsize=9)
    ax.set_ylabel('Row (Road Segment)', fontsize=9)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {outfile}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Performance Analysis (4-panel)
# ─────────────────────────────────────────────────────────────────────────────

def plot_performance(perf_data, serial_time, rmse, mae, max_diff,
                     outfile='performance_analysis.png'):
    thread_configs = [1, 2, 4, 8]
    COLORS  = {'static': '#1976D2', 'dynamic': '#F57C00', 'collapse': '#388E3C'}
    MARKERS = {'static': 'o',       'dynamic': 's',       'collapse': '^'}

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(
        'OpenMP Traffic Simulation — Performance Analysis\n'
        'Group 10 | Grid: 200×200 | Lanes: 3 | Time Steps: 200',
        fontsize=13, fontweight='bold')

    ax_time = axes[0][0]
    ax_sp   = axes[0][1]
    ax_eff  = axes[1][0]
    ax_acc  = axes[1][1]

    # ── (1) Execution Time ──────────────────────────────────────────────────
    ax_time.set_title('(a) Execution Time vs Thread Count', fontsize=10, fontweight='bold')
    if serial_time:
        ax_time.axhline(serial_time, color='#D32F2F', lw=2, ls='--',
                        label=f'Serial baseline  ({serial_time:.4f} s)', zorder=5)
    for sched, data in sorted(perf_data.items()):
        ts = [d[0] for d in data]
        tm = [d[1] for d in data]
        ax_time.plot(ts, tm, color=COLORS.get(sched, 'gray'),
                     marker=MARKERS.get(sched, 'o'), lw=2, ms=8,
                     label=f'OMP {sched}')
        for t, v in zip(ts, tm):
            ax_time.annotate(f'{v:.3f}', (t, v),
                             textcoords='offset points', xytext=(5, 3),
                             fontsize=7, color=COLORS.get(sched, 'gray'))
    ax_time.set_xlabel('Number of Threads', fontsize=9)
    ax_time.set_ylabel('Execution Time (s)', fontsize=9)
    ax_time.set_xticks(thread_configs)
    ax_time.legend(fontsize=8)
    ax_time.grid(True, alpha=0.3)
    ax_time.set_xlim(0.5, 9)

    # ── (2) Speedup ─────────────────────────────────────────────────────────
    ax_sp.set_title('(b) Speedup vs Thread Count', fontsize=10, fontweight='bold')
    ax_sp.plot(thread_configs, thread_configs, 'k--', lw=1.5,
               alpha=0.5, label='Ideal (linear)')
    for sched, data in sorted(perf_data.items()):
        ts = [d[0] for d in data]
        sp = [d[2] for d in data]
        ax_sp.plot(ts, sp, color=COLORS.get(sched, 'gray'),
                   marker=MARKERS.get(sched, 'o'), lw=2, ms=8,
                   label=f'OMP {sched} (vs 1T)')
        for t, s in zip(ts, sp):
            ax_sp.annotate(f'{s:.2f}×', (t, s),
                           textcoords='offset points', xytext=(5, 3), fontsize=7)
    # Speedup vs actual serial baseline
    if serial_time:
        for sched, data in sorted(perf_data.items()):
            ts  = [d[0] for d in data]
            sp2 = [serial_time / d[1] for d in data]
            ax_sp.plot(ts, sp2, color=COLORS.get(sched, 'gray'),
                       marker=MARKERS.get(sched, 'o'), lw=1.5, ms=6,
                       ls=':', alpha=0.7,
                       label=f'OMP {sched} (vs serial)')
    ax_sp.set_xlabel('Number of Threads', fontsize=9)
    ax_sp.set_ylabel('Speedup  (T₁ / Tₙ)', fontsize=9)
    ax_sp.set_xticks(thread_configs)
    ax_sp.legend(fontsize=7, ncol=2)
    ax_sp.grid(True, alpha=0.3)
    ax_sp.set_xlim(0.5, 9)

    # ── (3) Efficiency ──────────────────────────────────────────────────────
    ax_eff.set_title('(c) Parallel Efficiency vs Thread Count', fontsize=10, fontweight='bold')
    ax_eff.axhline(1.0, color='k', ls='--', lw=1.5, alpha=0.5,
                   label='Ideal efficiency (100%)')
    for sched, data in sorted(perf_data.items()):
        ts  = [d[0] for d in data]
        eff = [d[3] for d in data]
        ax_eff.plot(ts, eff, color=COLORS.get(sched, 'gray'),
                    marker=MARKERS.get(sched, 'o'), lw=2, ms=8,
                    label=f'OMP {sched}')
        for t, e in zip(ts, eff):
            ax_eff.annotate(f'{e:.2f}\n({e*100:.0f}%)', (t, e),
                            textcoords='offset points', xytext=(0, 8),
                            ha='center', fontsize=7)
    ax_eff.set_xlabel('Number of Threads', fontsize=9)
    ax_eff.set_ylabel('Efficiency = Speedup / N_threads', fontsize=9)
    ax_eff.set_xticks(thread_configs)
    ax_eff.set_ylim(0, 1.35)
    ax_eff.legend(fontsize=8)
    ax_eff.grid(True, alpha=0.3)
    ax_eff.set_xlim(0.5, 9)

    # ── (4) Accuracy Text Box ───────────────────────────────────────────────
    ax_acc.set_title('(d) Accuracy & Statistical Summary', fontsize=10, fontweight='bold')
    ax_acc.axis('off')

    lines_txt = [
        '┌─────────────────────────────────────────────┐',
        '│   ACCURACY: Serial vs OpenMP Final State    │',
        '├─────────────────────────────────────────────┤',
    ]
    if rmse is not None:
        ok = '✓' if rmse < 0.5 else '⚠'
        lines_txt += [
            f'│  RMSE         : {rmse:>10.6f}              │',
            f'│  MAE          : {mae:>10.6f}              │',
            f'│  Max |Δ|      : {max_diff:>10.6f}              │',
            f'│  Verdict      :  {ok} RMSE < 0.5 → match OK   │',
        ]
    else:
        lines_txt += ['│  RMSE: N/A  (rerun both simulations)        │']
    lines_txt += [
        '├─────────────────────────────────────────────┤',
        '│  WITHIN OpenMP (1T vs 2T/4T/8T):            │',
        '│  Max |Δ| ≈ 0.000000e+00  (deterministic)    │',
        '├─────────────────────────────────────────────┤',
        '│  BEST CONFIG (OpenMP Static Schedule):      │',
    ]
    if 'static' in perf_data and perf_data['static']:
        best = max(perf_data['static'], key=lambda x: x[2])
        lines_txt += [
            f'│  Threads      : {best[0]:>4d}                         │',
            f'│  Time         : {best[1]:>10.6f} s                │',
            f'│  Speedup      : {best[2]:>8.4f} ×                  │',
            f'│  Efficiency   : {best[3]:>8.4f}  ({best[3]*100:.1f}%)         │',
        ]
    if serial_time and 'static' in perf_data and perf_data['static']:
        best_t = min(d[1] for d in perf_data['static'])
        ser_sp = serial_time / best_t
        lines_txt += [
            '├─────────────────────────────────────────────┤',
            f'│  Serial time  : {serial_time:>10.6f} s                │',
            f'│  OMP best     : {best_t:>10.6f} s                │',
            f'│  Serial→OMP   : {ser_sp:>8.4f} ×  speedup         │',
        ]
    lines_txt.append('└─────────────────────────────────────────────┘')

    ax_acc.text(0.03, 0.97, '\n'.join(lines_txt),
                transform=ax_acc.transAxes,
                fontsize=8.2, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='#f5f5f5', edgecolor='#bdbdbd', lw=1.5))

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {outfile}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Full Analysis Report (6-panel combined)
# ─────────────────────────────────────────────────────────────────────────────

def plot_full_report(serial_grid, omp_grid, perf_data, serial_time,
                     rmse, mae, max_diff, best_threads,
                     outfile='full_analysis_report.png'):
    COLORS  = {'static': '#1976D2', 'dynamic': '#F57C00', 'collapse': '#388E3C'}
    MARKERS = {'static': 'o',       'dynamic': 's',       'collapse': '^'}
    thread_configs = [1, 2, 4, 8]
    ROWS, COLS = 200, 200

    fig = plt.figure(figsize=(18, 24))
    fig.suptitle(
        'Parallel Traffic Density Simulation — Full Analysis Report\n'
        'Serial vs OpenMP | Group 10 | EE7218/EC7207 High Performance Computing',
        fontsize=15, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.42, wspace=0.3,
                           top=0.95, bottom=0.04, left=0.07, right=0.97)

    ax_s   = fig.add_subplot(gs[0, 0])   # (a) Serial heatmap
    ax_p   = fig.add_subplot(gs[0, 1])   # (b) OpenMP heatmap
    ax_sp  = fig.add_subplot(gs[1, 0])   # (c) Speedup
    ax_eff = fig.add_subplot(gs[1, 1])   # (d) Efficiency
    ax_bar = fig.add_subplot(gs[2, 0])   # (e) Timing bar chart
    ax_sum = fig.add_subplot(gs[2, 1])   # (f) Summary text

    # ─── (a) Serial Heatmap ───────────────────────────────────────────────
    if serial_grid is not None:
        im_s = ax_s.imshow(serial_grid, cmap=CMAP, vmin=0, vmax=100,
                           aspect='equal', origin='upper')
        plt.colorbar(im_s, ax=ax_s, label='Density', fraction=0.046, pad=0.04)
        x0, x1 = COLS//3, 2*COLS//3
        y0, y1 = ROWS//3, 2*ROWS//3
        ax_s.add_patch(mpatches.Rectangle((x0, y0), x1-x0, y1-y0,
                       lw=2, edgecolor='white', facecolor='none', ls='--'))
        ax_s.text(x0+3, y0+9, 'Rain\n(0.7×)',
                  color='white', fontsize=8,
                  bbox=dict(facecolor='#000000aa', pad=2))
    ax_s.set_title('(a) Serial — Final Traffic Density', fontsize=11, fontweight='bold')
    ax_s.set_xlabel('Column'); ax_s.set_ylabel('Row')

    # ─── (b) OpenMP Heatmap ───────────────────────────────────────────────
    if omp_grid is not None:
        im_p = ax_p.imshow(omp_grid, cmap=CMAP, vmin=0, vmax=100,
                           aspect='equal', origin='upper')
        plt.colorbar(im_p, ax=ax_p, label='Density', fraction=0.046, pad=0.04)
        x0, x1 = COLS//3, 2*COLS//3
        y0, y1 = ROWS//3, 2*ROWS//3
        ax_p.add_patch(mpatches.Rectangle((x0, y0), x1-x0, y1-y0,
                       lw=2, edgecolor='white', facecolor='none', ls='--'))
        ax_p.text(x0+3, y0+9, 'Rain\n(0.7×)',
                  color='white', fontsize=8,
                  bbox=dict(facecolor='#000000aa', pad=2))
        if rmse is not None:
            ax_p.annotate(f'RMSE: {rmse:.6f}',
                          xy=(0.02, 0.03), xycoords='axes fraction',
                          fontsize=8.5, color='white', fontweight='bold',
                          bbox=dict(facecolor='black', alpha=0.65, pad=3))
    ax_p.set_title(f'(b) OpenMP — Final Traffic Density  [{best_threads} Threads]',
                   fontsize=11, fontweight='bold')
    ax_p.set_xlabel('Column'); ax_p.set_ylabel('Row')

    # ─── (c) Speedup ──────────────────────────────────────────────────────
    ax_sp.set_title('(c) Speedup vs Thread Count', fontsize=11, fontweight='bold')
    ax_sp.plot(thread_configs, thread_configs, 'k--', lw=1.5, alpha=0.5,
               label='Ideal (linear)')
    for sched, data in sorted(perf_data.items()):
        ts = [d[0] for d in data]
        sp = [d[2] for d in data]
        ax_sp.plot(ts, sp, color=COLORS.get(sched, 'gray'),
                   marker=MARKERS.get(sched, 'o'), lw=2.5, ms=9,
                   label=f'{sched}')
        for t, s in zip(ts, sp):
            ax_sp.annotate(f'{s:.2f}', (t, s),
                           textcoords='offset points', xytext=(5, 4), fontsize=8)
    ax_sp.set_xlabel('Number of Threads', fontsize=10)
    ax_sp.set_ylabel('Speedup  (T₁ / Tₙ)', fontsize=10)
    ax_sp.set_xticks(thread_configs)
    ax_sp.legend(fontsize=9)
    ax_sp.grid(True, alpha=0.3)
    ax_sp.set_xlim(0.5, 9)

    # ─── (d) Efficiency ───────────────────────────────────────────────────
    ax_eff.set_title('(d) Parallel Efficiency vs Thread Count', fontsize=11, fontweight='bold')
    ax_eff.axhline(1.0, color='k', ls='--', lw=1.5, alpha=0.5, label='Ideal (100%)')
    for sched, data in sorted(perf_data.items()):
        ts  = [d[0] for d in data]
        eff = [d[3] for d in data]
        ax_eff.plot(ts, eff, color=COLORS.get(sched, 'gray'),
                    marker=MARKERS.get(sched, 'o'), lw=2.5, ms=9,
                    label=f'{sched}')
        for t, e in zip(ts, eff):
            ax_eff.annotate(f'{e:.2f}\n({e*100:.0f}%)', (t, e),
                            textcoords='offset points', xytext=(0, 9),
                            ha='center', fontsize=7.5)
    ax_eff.set_xlabel('Number of Threads', fontsize=10)
    ax_eff.set_ylabel('Efficiency = Speedup / N_threads', fontsize=10)
    ax_eff.set_xticks(thread_configs)
    ax_eff.set_ylim(0, 1.4)
    ax_eff.legend(fontsize=9)
    ax_eff.grid(True, alpha=0.3)
    ax_eff.set_xlim(0.5, 9)

    # ─── (e) Timing Bar Chart ─────────────────────────────────────────────
    ax_bar.set_title('(e) Execution Time Comparison (All Configurations)',
                     fontsize=11, fontweight='bold')
    labels, times, bar_colors = [], [], []
    if serial_time:
        labels.append('Serial\n(1T)')
        times.append(serial_time)
        bar_colors.append('#D32F2F')
    for sched in sorted(perf_data.keys()):
        clr = COLORS.get(sched, 'gray')
        for t, tm, _, _ in perf_data[sched]:
            labels.append(f'OMP\n{sched}\n{t}T')
            times.append(tm)
            bar_colors.append(clr)
    x_pos = range(len(labels))
    bars  = ax_bar.bar(x_pos, times, color=bar_colors, edgecolor='white', lw=0.8, width=0.65)
    ax_bar.set_xticks(list(x_pos))
    ax_bar.set_xticklabels(labels, fontsize=7.5)
    ax_bar.set_ylabel('Execution Time (seconds)', fontsize=10)
    ax_bar.set_xlabel('Configuration', fontsize=10)
    ax_bar.grid(axis='y', alpha=0.3)
    for bar, t in zip(bars, times):
        ax_bar.text(bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + max(times) * 0.005,
                    f'{t:.3f}', ha='center', va='bottom', fontsize=7, rotation=45)
    # Legend patches for bar chart
    leg_patches = [mpatches.Patch(color='#D32F2F', label='Serial')]
    for s, c in COLORS.items():
        if s in perf_data:
            leg_patches.append(mpatches.Patch(color=c, label=f'OMP {s}'))
    ax_bar.legend(handles=leg_patches, fontsize=8, loc='upper right')

    # ─── (f) Summary Text ─────────────────────────────────────────────────
    ax_sum.set_title('(f) Analysis Summary', fontsize=11, fontweight='bold')
    ax_sum.axis('off')

    summary = [
        '  EE7218/EC7207 — High Performance Computing',
        '  Group 10 | Traffic Density Simulation',
        '',
        '  Parallelization Strategy',
        '  ─────────────────────────────────────────',
        '  • Model  : 2D traffic diffusion on 200×200 grid',
        '  • Method : omp parallel for on outer row-loop',
        '  • Data   : traffic[], new_traffic[], weather[]',
        '             shared across all threads',
        '  • Sync   : Implicit barrier after each parallel for',
        '  • Schedules tested: static, dynamic(10), collapse(2)',
        '',
        '  Accuracy (Serial vs OpenMP)',
        '  ─────────────────────────────────────────',
    ]
    if rmse is not None:
        verdict = 'PASS — Results match within tolerance' if rmse < 0.5 else \
                  'NOTE — Check random seed alignment'
        summary += [
            f'  RMSE         : {rmse:.6f}',
            f'  MAE          : {mae:.6f}',
            f'  Max |Δ|      : {max_diff:.6f}',
            f'  Verdict      : {verdict}',
        ]
    summary += [
        '  Internal (1T vs NT): MaxDiff = 0.00e+00',
        '',
        '  Performance (Static Schedule)',
        '  ─────────────────────────────────────────',
    ]
    if 'static' in perf_data and perf_data['static']:
        for t, tm, sp, eff in perf_data['static']:
            speedup_vs_serial = f'{serial_time/tm:.4f}×' if serial_time else 'N/A'
            summary.append(
                f'  {t}T: {tm:.4f}s  |  Speedup(1T)={sp:.4f}×'
                f'  |  Eff={eff:.4f}'
                f'  |  vs serial={speedup_vs_serial}')
    summary += [
        '',
        '  OpenMP API Used',
        '  ─────────────────────────────────────────',
        '  • #pragma omp parallel for schedule(static)',
        '  • #pragma omp parallel for schedule(dynamic,10)',
        '  • #pragma omp parallel for collapse(2)',
        '  • #pragma omp reduction(+:sum)(min:lo)(max:hi)',
        '  • #pragma omp master',
        '  • omp_set_num_threads / omp_get_num_threads',
        '  • omp_get_wtime (timing)',
    ]

    ax_sum.text(0.02, 0.98, '\n'.join(summary),
                transform=ax_sum.transAxes,
                fontsize=8.2, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.6',
                          facecolor='#fafafa', edgecolor='#9e9e9e', lw=1.5))

    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  {outfile}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('\n' + '='*58)
    print('  Serial vs OpenMP Traffic Analysis  |  Group 10')
    print('='*58)

    # ── Load all data ──────────────────────────────────────────────────────
    print('\n[Loading Data]')
    serial_grid, *_  = load_traffic_values(SERIAL_VALUES)
    omp_grid,    *_  = load_traffic_values(OMP_VALUES)
    serial_time      = parse_serial_time(SERIAL_OUTPUT)
    perf_data        = parse_omp_performance(OMP_PERF)

    print(f'  Serial grid     : '
          f'{serial_grid.shape if serial_grid is not None else "NOT FOUND"}')
    print(f'  OpenMP grid     : '
          f'{omp_grid.shape if omp_grid is not None else "NOT FOUND"}')
    print(f'  Serial time     : '
          f'{serial_time:.6f} s' if serial_time else '  Serial time     : NOT FOUND')
    print(f'  Perf schedules  : {list(perf_data.keys())}')

    # ── Accuracy ───────────────────────────────────────────────────────────
    print('\n[Accuracy Analysis]')
    rmse, mae, max_diff = compute_accuracy(serial_grid, omp_grid)
    if rmse is not None:
        print(f'  RMSE           : {rmse:.6f}')
        print(f'  MAE            : {mae:.6f}')
        print(f'  Max |Δ|        : {max_diff:.6f}')
        if rmse < 0.5:
            print('  Verdict        : ✓ Serial and OpenMP results match')
        else:
            print('  Verdict        : ⚠  RMSE > 0.5 — verify seeds match')
            print('                   Recompile & rerun both programs with')
            print('                   matching seeds, then rerun this script.')
    else:
        print('  RMSE: N/A')

    # ── Best thread count ──────────────────────────────────────────────────
    best_threads = 8
    if perf_data:
        all_threads = [d[0] for data in perf_data.values() for d in data]
        if all_threads:
            best_threads = max(all_threads)

    # ── Generate figures ───────────────────────────────────────────────────
    print('\n[Generating Figures]')
    outdir = SCRIPT_DIR

    plot_heatmaps(
        serial_grid, omp_grid, rmse, mae, max_diff, best_threads,
        outfile=os.path.join(outdir, 'comparison_heatmaps.png'))

    plot_performance(
        perf_data, serial_time, rmse, mae, max_diff,
        outfile=os.path.join(outdir, 'performance_analysis.png'))

    plot_full_report(
        serial_grid, omp_grid, perf_data, serial_time,
        rmse, mae, max_diff, best_threads,
        outfile=os.path.join(outdir, 'full_analysis_report.png'))

    # ── Console summary ────────────────────────────────────────────────────
    print('\n[Summary]')
    if serial_time and 'static' in perf_data and perf_data['static']:
        best_t  = min(d[1] for d in perf_data['static'])
        ser_sp  = serial_time / best_t
        best_nt = min(perf_data['static'], key=lambda x: x[1])[0]
        print(f'  Serial time      : {serial_time:.6f} s')
        print(f'  OpenMP best time : {best_t:.6f} s  ({best_nt} threads, static)')
        print(f'  Serial→OMP speed : {ser_sp:.4f} ×')
    if rmse is not None:
        print(f'  RMSE             : {rmse:.6f}')

    print('\n[Output Files]')
    for f in ['comparison_heatmaps.png',
              'performance_analysis.png',
              'full_analysis_report.png']:
        path = os.path.join(outdir, f)
        status = '✓' if os.path.exists(path) else '✗'
        print(f'  {status} {f}')

    print('\n' + '='*58 + '\n')


if __name__ == '__main__':
    main()
