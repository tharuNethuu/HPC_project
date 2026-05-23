#!/usr/bin/env python3
"""Local dashboard for the traffic benchmark suite."""

from __future__ import annotations

import json
import os
import platform
import shlex
import re
import shutil
import subprocess
import sys
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parent.parent
UI_ROOT = Path(__file__).resolve().parent
DEFAULT_PORT = 8000
ALLOWED_MODES = {"serial", "openmp", "mpi", "hybrid"}


def detect_host_profile() -> str:
    """Return macos, wsl, linux, or other."""
    release = platform.release().lower()
    version = platform.version().lower()
    if (
        "microsoft" in release
        or "microsoft" in version
        or os.environ.get("WSL_DISTRO_NAME")
        or os.environ.get("WSL_INTEROP")
    ):
        return "wsl"
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    return "other"


def which(program: str) -> Optional[str]:
    return shutil.which(program)


def shell_quote(command: Iterable[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def clamp_int(value: object, default: int, minimum: int = 1, maximum: int = 256) -> int:
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return default
    if ivalue < minimum:
        return minimum
    if ivalue > maximum:
        return maximum
    return ivalue


def normalize_mode_config(mode: str, raw: Optional[Dict[str, object]]) -> Dict[str, object]:
    config = raw or {}
    if mode == "serial":
        return {}
    if mode == "openmp":
        schedule = str(config.get("schedule", "static")).strip().lower()
        if schedule not in {"static", "dynamic", "collapse"}:
            schedule = "static"
        return {
            "threads": clamp_int(config.get("threads"), default=4, minimum=1, maximum=64),
            "schedule": schedule,
        }
    if mode == "mpi":
        return {
            "processes": clamp_int(config.get("processes"), default=4, minimum=1, maximum=64),
        }
    return {
        "processes": clamp_int(config.get("processes"), default=2, minimum=1, maximum=64),
        "threads": clamp_int(config.get("threads"), default=4, minimum=1, maximum=64),
    }


def run_command(command: List[str], cwd: Path, timeout: int = 1800) -> Dict[str, object]:
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "cwd": str(cwd),
            "returnCode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except FileNotFoundError as exc:
        return {
            "command": command,
            "cwd": str(cwd),
            "returnCode": 127,
            "stdout": "",
            "stderr": f"Command not found: {command[0]}\n{exc}",
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": command,
            "cwd": str(cwd),
            "returnCode": 124,
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + "\nCommand timed out.",
        }


def read_text(path: Path, limit_lines: Optional[int] = None) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace")
    if limit_lines is None:
        return content
    return "\n".join(content.splitlines()[:limit_lines])


def parse_grid_file(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if not lines:
        return None
    rows, cols, lanes = map(int, lines[0].split())
    values = [[float(value) for value in line.split()] for line in lines[1:]]
    flat = [value for row in values for value in row]
    return {
        "rows": rows,
        "cols": cols,
        "lanes": lanes,
        "values": values,
        "min": min(flat) if flat else 0.0,
        "max": max(flat) if flat else 0.0,
        "avg": (sum(flat) / len(flat)) if flat else 0.0,
    }


def parse_openmp_performance(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    records: List[Dict[str, object]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("=", "-", "Grid", "[", "Threads", "OpenMP", "Speedup", "Efficiency", "MaxDiff")):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            records.append(
                {
                    "threads": int(parts[0]),
                    "schedule": parts[1],
                    "time": float(parts[2]),
                    "speedup": float(parts[3]),
                    "efficiency": float(parts[4]),
                }
            )
        except ValueError:
            continue
    return records


def parse_mpi_performance(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    records: List[Dict[str, object]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("=", "-", "Grid", "[", "Procs", "MPI", "Speedup", "Efficiency", "Group")):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            records.append(
                {
                    "np": int(parts[0]),
                    "time": float(parts[1]),
                    "min": float(parts[2]) if len(parts) > 2 else 0.0,
                    "max": float(parts[3]) if len(parts) > 3 else 0.0,
                    "avg": float(parts[4]) if len(parts) > 4 else 0.0,
                }
            )
        except ValueError:
            continue
    base = next((record["time"] for record in records if record["np"] == 1), None)
    for record in records:
        record["speedup"] = (base / record["time"]) if base else 1.0
        record["efficiency"] = record["speedup"] / record["np"]
    return records


def parse_hybrid_performance(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    records: List[Dict[str, object]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(("=", "-", "Grid", "[", "Procs", "Hybrid", "Speed", "Efficiency", "Group")):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            np_val = int(parts[0])
            nt_val = int(parts[1])
            total = int(parts[2])
            time_s = float(parts[3])
            records.append(
                {
                    "np": np_val,
                    "nt": nt_val,
                    "total": total,
                    "time": time_s,
                    "min": float(parts[4]) if len(parts) > 4 else 0.0,
                    "max": float(parts[5]) if len(parts) > 5 else 0.0,
                    "avg": float(parts[6]) if len(parts) > 6 else 0.0,
                    "label": f"{np_val}P x {nt_val}T",
                }
            )
        except ValueError:
            continue
    for record in records:
        record["speedup"] = 1.0
        record["efficiency"] = 1.0 / record["total"]
    serial_time = parse_serial_time(PROJECT_ROOT / "serial" / "serial_output.txt")
    if serial_time:
        for record in records:
            record["speedup"] = serial_time / record["time"]
            record["efficiency"] = record["speedup"] / record["total"]
    return records


def parse_serial_time(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Execution Time:\s*([\d.]+)\s*seconds", content)
    return float(match.group(1)) if match else None


def build_mode_snapshot(mode: str) -> Dict[str, object]:
    serial_dir = PROJECT_ROOT / "serial"
    openmp_dir = PROJECT_ROOT / "openmp"
    mpi_dir = PROJECT_ROOT / "mpi"
    hybrid_dir = PROJECT_ROOT / "hybrid"

    if mode == "serial":
        grid = parse_grid_file(serial_dir / "traffic_values.txt")
        time_value = parse_serial_time(serial_dir / "serial_output.txt")
        profile = []
        if grid:
            for row in grid["values"]:
                profile.append(sum(row) / len(row))
        report = read_text(serial_dir / "serial_output.txt", limit_lines=12)
        summary = {
            "executionTime": time_value,
            "bestConfig": "1 process / 1 thread",
            "gridAvg": grid["avg"] if grid else None,
            "gridMin": grid["min"] if grid else None,
            "gridMax": grid["max"] if grid else None,
        }
        charts = [
            {
                "title": "Lane-averaged density profile",
                "xLabel": "Row",
                "yLabel": "Average density",
                "series": [
                    {
                        "label": "Row average",
                        "points": [{"x": idx + 1, "y": value} for idx, value in enumerate(profile)],
                        "color": "#0f766e",
                    }
                ],
            }
        ]
    elif mode == "openmp":
        grid = parse_grid_file(openmp_dir / "openmp_traffic_values.txt")
        perf = parse_openmp_performance(openmp_dir / "openmp_performance.txt")
        report = read_text(openmp_dir / "openmp_performance.txt")
        best = min(perf, key=lambda record: record["time"]) if perf else None
        summary = {
            "executionTime": best["time"] if best else None,
            "bestConfig": f"{best['threads']} threads / {best['schedule']}" if best else "Unknown",
            "gridAvg": grid["avg"] if grid else None,
            "gridMin": grid["min"] if grid else None,
            "gridMax": grid["max"] if grid else None,
            "bestSpeedup": best["speedup"] if best else None,
            "bestEfficiency": best["efficiency"] if best else None,
        }
        schedules = sorted({record["schedule"] for record in perf})
        charts = [
            {
                "title": "Execution time by schedule",
                "xLabel": "Threads",
                "yLabel": "Seconds",
                "series": [
                    {
                        "label": schedule,
                        "points": [
                            {"x": record["threads"], "y": record["time"]}
                            for record in perf
                            if record["schedule"] == schedule
                        ],
                        "color": {"static": "#0f766e", "dynamic": "#d97706", "collapse": "#1d4ed8"}.get(schedule, "#334155"),
                    }
                    for schedule in schedules
                ],
            },
            {
                "title": "Speedup by schedule",
                "xLabel": "Threads",
                "yLabel": "Speedup",
                "series": [
                    {
                        "label": schedule,
                        "points": [
                            {"x": record["threads"], "y": record["speedup"]}
                            for record in perf
                            if record["schedule"] == schedule
                        ],
                        "color": {"static": "#0f766e", "dynamic": "#d97706", "collapse": "#1d4ed8"}.get(schedule, "#334155"),
                    }
                    for schedule in schedules
                ],
            },
        ]
    elif mode == "mpi":
        grid = parse_grid_file(mpi_dir / "mpi_traffic_values.txt")
        perf = parse_mpi_performance(mpi_dir / "mpi_performance.txt")
        report = read_text(mpi_dir / "mpi_performance.txt")
        best = min(perf, key=lambda record: record["time"]) if perf else None
        summary = {
            "executionTime": best["time"] if best else None,
            "bestConfig": f"{best['np']} processes" if best else "Unknown",
            "gridAvg": grid["avg"] if grid else None,
            "gridMin": grid["min"] if grid else None,
            "gridMax": grid["max"] if grid else None,
            "bestSpeedup": best["speedup"] if best else None,
            "bestEfficiency": best["efficiency"] if best else None,
        }
        charts = [
            {
                "title": "Execution time vs processes",
                "xLabel": "Processes",
                "yLabel": "Seconds",
                "series": [
                    {
                        "label": "MPI runtime",
                        "points": [{"x": record["np"], "y": record["time"]} for record in perf],
                        "color": "#7c3aed",
                    }
                ],
            },
            {
                "title": "Speedup vs processes",
                "xLabel": "Processes",
                "yLabel": "Speedup",
                "series": [
                    {
                        "label": "Speedup",
                        "points": [{"x": record["np"], "y": record["speedup"]} for record in perf],
                        "color": "#0f766e",
                    }
                ],
            },
        ]
    else:
        grid = parse_grid_file(hybrid_dir / "hybrid_traffic_values.txt")
        perf = parse_hybrid_performance(hybrid_dir / "hybrid_performance.txt")
        report = read_text(hybrid_dir / "hybrid_performance.txt")
        best = min(perf, key=lambda record: record["time"]) if perf else None
        summary = {
            "executionTime": best["time"] if best else None,
            "bestConfig": f"{best['np']}P x {best['nt']}T" if best else "Unknown",
            "gridAvg": grid["avg"] if grid else None,
            "gridMin": grid["min"] if grid else None,
            "gridMax": grid["max"] if grid else None,
            "bestSpeedup": best["speedup"] if best else None,
            "bestEfficiency": best["efficiency"] if best else None,
        }
        by_np: Dict[int, List[Dict[str, object]]] = {}
        for record in perf:
            by_np.setdefault(record["np"], []).append(record)
        for records in by_np.values():
            records.sort(key=lambda record: record["total"])
        charts = [
            {
                "title": "Execution time vs total workers",
                "xLabel": "Total workers",
                "yLabel": "Seconds",
                "series": [
                    {
                        "label": f"{np_val} MPI procs",
                        "points": [
                            {"x": record["total"], "y": record["time"]}
                            for record in records
                        ],
                        "color": {1: "#7c3aed", 2: "#0f766e", 4: "#d97706"}.get(np_val, "#334155"),
                    }
                    for np_val, records in sorted(by_np.items())
                ],
            },
            {
                "title": "Speedup vs total workers",
                "xLabel": "Total workers",
                "yLabel": "Speedup",
                "series": [
                    {
                        "label": f"{np_val} MPI procs",
                        "points": [
                            {"x": record["total"], "y": record["speedup"]}
                            for record in records
                        ],
                        "color": {1: "#7c3aed", 2: "#0f766e", 4: "#d97706"}.get(np_val, "#334155"),
                    }
                    for np_val, records in sorted(by_np.items())
                ],
            },
        ]

    return {
        "mode": mode,
        "grid": grid,
        "summary": summary,
        "charts": charts,
        "report": report,
    }


def build_compile_candidates(mode: str, host: str) -> List[List[str]]:
    serial_dir = PROJECT_ROOT / "serial"
    openmp_dir = PROJECT_ROOT / "openmp"
    mpi_dir = PROJECT_ROOT / "mpi"
    hybrid_dir = PROJECT_ROOT / "hybrid"

    if mode == "serial":
        compiler = which("clang") if host == "macos" else which("gcc")
        compiler = compiler or which("gcc") or which("clang") or which("cc")
        if not compiler:
            raise RuntimeError("No C compiler found on this machine.")
        return [[compiler, "-O2", "-o", "traffic_serial", "traffic_serial.c", "-lm"]]

    if mode == "openmp":
        candidates: List[List[str]] = []
        brew = which("brew")
        libomp_prefix: Optional[str] = None
        if brew:
            proc = subprocess.run([brew, "--prefix", "libomp"], capture_output=True, text=True, check=False)
            if proc.returncode == 0:
                libomp_prefix = proc.stdout.strip()
        if host == "macos":
            clang = which("clang")
            if clang:
                base = [clang, "-O2", "-Xpreprocessor", "-fopenmp"]
                if libomp_prefix:
                    base.extend([f"-I{libomp_prefix}/include", f"-L{libomp_prefix}/lib"])
                base.extend(["-o", "traffic_openmp", "traffic_openmp.c", "-lm", "-lomp"])
                candidates.append(base)
            for gcc_name in ("gcc-14", "gcc-13", "gcc-12", "gcc-11"):
                gcc = which(gcc_name)
                if gcc:
                    candidates.append([gcc, "-O2", "-fopenmp", "-o", "traffic_openmp", "traffic_openmp.c", "-lm"])
        else:
            gcc = which("gcc")
            if gcc:
                candidates.append([gcc, "-O2", "-fopenmp", "-o", "traffic_openmp", "traffic_openmp.c", "-lm"])
            clang = which("clang")
            if clang:
                candidates.append([clang, "-O2", "-fopenmp", "-o", "traffic_openmp", "traffic_openmp.c", "-lm"])
        return candidates

    if mode == "mpi":
        mpicc = which("mpicc")
        if not mpicc:
            raise RuntimeError("mpicc was not found. Install an MPI toolchain and try again.")
        return [[mpicc, "-O2", "-o", "traffic_mpi", "traffic_mpi.c", "-lm"]]

    mpicc = which("mpicc")
    if not mpicc:
        raise RuntimeError("mpicc was not found. Install an MPI toolchain and try again.")
    candidates = []
    brew = which("brew")
    libomp_prefix = None
    if brew:
        proc = subprocess.run([brew, "--prefix", "libomp"], capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            libomp_prefix = proc.stdout.strip()
    base = [mpicc, "-O2"]
    if host == "macos":
        if libomp_prefix:
            candidates.append(
                base
                + ["-Xpreprocessor", "-fopenmp", f"-I{libomp_prefix}/include", f"-L{libomp_prefix}/lib", "-lomp", "-o", "traffic_hybrid", "traffic_hybrid.c", "-lm"]
            )
        candidates.append(base + ["-Xpreprocessor", "-fopenmp", "-o", "traffic_hybrid", "traffic_hybrid.c", "-lm", "-lomp"])
        candidates.append(base + ["-fopenmp", "-o", "traffic_hybrid", "traffic_hybrid.c", "-lm"])
    else:
        candidates.append([mpicc, "-O2", "-fopenmp", "-o", "traffic_hybrid", "traffic_hybrid.c", "-lm"])
        candidates.append([mpicc, "-O2", "-o", "traffic_hybrid", "traffic_hybrid.c", "-lm"])
    return candidates


def build_run_steps(mode: str, host: str, mode_config: Dict[str, object]) -> List[Dict[str, object]]:
    serial_dir = PROJECT_ROOT / "serial"
    openmp_dir = PROJECT_ROOT / "openmp"
    mpi_dir = PROJECT_ROOT / "mpi"
    hybrid_dir = PROJECT_ROOT / "hybrid"

    if mode == "serial":
        return [
            {"kind": "compile", "cwd": serial_dir, "commands": build_compile_candidates(mode, host), "timeout": 300},
            {"kind": "run", "cwd": serial_dir, "command": ["./traffic_serial"], "timeout": 600},
        ]

    if mode == "openmp":
        threads = int(mode_config.get("threads", 4))
        schedule = str(mode_config.get("schedule", "static"))
        return [
            {"kind": "compile", "cwd": openmp_dir, "commands": build_compile_candidates(mode, host), "timeout": 300},
            {"kind": "run", "cwd": openmp_dir, "command": ["./traffic_openmp", str(threads), schedule], "timeout": 900},
        ]

    if mode == "mpi":
        processes = int(mode_config.get("processes", 4))
        return [
            {"kind": "compile", "cwd": mpi_dir, "commands": build_compile_candidates(mode, host), "timeout": 300},
            {"kind": "run", "cwd": mpi_dir, "command": ["mpirun", "--oversubscribe", "-np", str(processes), "./traffic_mpi"], "timeout": 1800},
        ]

    processes = int(mode_config.get("processes", 2))
    threads = int(mode_config.get("threads", 4))
    return [
        {"kind": "compile", "cwd": hybrid_dir, "commands": build_compile_candidates(mode, host), "timeout": 300},
        {"kind": "run", "cwd": hybrid_dir, "command": ["mpirun", "--oversubscribe", "-np", str(processes), "./traffic_hybrid", str(threads)], "timeout": 1800},
    ]


def run_mode(mode: str, host: str, mode_config: Dict[str, object]) -> Dict[str, object]:
    if mode not in ALLOWED_MODES:
        raise ValueError(f"Unsupported mode: {mode}")

    steps = build_run_steps(mode, host, mode_config)
    transcript: List[str] = []
    executed_steps: List[Dict[str, object]] = []

    for step in steps:
        if step["kind"] == "compile":
            last_error = None
            for candidate in step["commands"]:
                transcript.append(f"$ {shell_quote(candidate)}\n")
                result = run_command(candidate, step["cwd"], timeout=step["timeout"])
                executed_steps.append({**result, "kind": "compile"})
                if result["stdout"]:
                    transcript.append(result["stdout"])
                    if not result["stdout"].endswith("\n"):
                        transcript.append("\n")
                if result["stderr"]:
                    transcript.append(result["stderr"])
                    if not result["stderr"].endswith("\n"):
                        transcript.append("\n")
                if result["returnCode"] == 0:
                    break
                last_error = result
            else:
                raise RuntimeError(
                    f"Compilation failed for {mode}.\n\n"
                    f"Last command:\n{shell_quote(last_error['command']) if last_error else 'unknown'}\n\n"
                    f"stderr:\n{last_error['stderr'] if last_error else 'No compiler output.'}\n\n"
                    "On macOS, OpenMP often needs Homebrew libomp or a GNU compiler."
                )
        else:
            command = step["command"]
            transcript.append(f"$ {shell_quote(command)}\n")
            result = run_command(command, step["cwd"], timeout=step["timeout"])
            executed_steps.append({**result, "kind": step["kind"]})
            if result["stdout"]:
                transcript.append(result["stdout"])
                if not result["stdout"].endswith("\n"):
                    transcript.append("\n")
            if result["stderr"]:
                transcript.append(result["stderr"])
                if not result["stderr"].endswith("\n"):
                    transcript.append("\n")
            if result["returnCode"] != 0:
                raise RuntimeError(
                    f"{mode} run failed.\n\nCommand:\n{shell_quote(command)}\n\n"
                    f"stderr:\n{result['stderr']}"
                )

    snapshot = build_mode_snapshot(mode)
    snapshot["steps"] = executed_steps
    snapshot["log"] = "".join(transcript).strip()
    snapshot["host"] = host
    snapshot["requestedConfig"] = mode_config
    return snapshot


def make_bootstrap() -> Dict[str, object]:
    host = detect_host_profile()
    notes = {
        "macos": "Detected macOS. Serial will build with clang, while OpenMP and hybrid may require Homebrew libomp or a GNU compiler.",
        "wsl": "Detected WSL/Linux. The dashboard will use native Linux toolchains and mpirun commands.",
        "linux": "Detected Linux. The dashboard will use native compiler and MPI commands.",
    }.get(host, "Detected platform is not fully recognised. The dashboard will still try the local compiler and MPI toolchain.")
    return {
        "host": host,
        "notes": notes,
        "modes": [
            {"id": "serial", "label": "Serial", "description": "Single-process baseline"},
            {"id": "openmp", "label": "OpenMP", "description": "Shared-memory parallel run"},
            {"id": "mpi", "label": "MPI", "description": "Distributed-memory scaling"},
            {"id": "hybrid", "label": "Hybrid", "description": "MPI + OpenMP combined"},
        ],
        "modeConfig": {
            "serial": {},
            "openmp": {
                "defaults": {"threads": 4, "schedule": "static"},
                "threads": {"min": 1, "max": 64, "presets": [1, 2, 4, 8]},
                "schedules": ["static", "dynamic", "collapse"],
            },
            "mpi": {
                "defaults": {"processes": 4},
                "processes": {"min": 1, "max": 64, "presets": [1, 2, 4, 8]},
            },
            "hybrid": {
                "defaults": {"processes": 2, "threads": 4},
                "processes": {"min": 1, "max": 64, "presets": [1, 2, 4]},
                "threads": {"min": 1, "max": 64, "presets": [1, 2, 4, 8]},
            },
        },
        "defaultMode": "serial",
        "projectRoot": str(PROJECT_ROOT),
    }


def send_json(handler: BaseHTTPRequestHandler, payload: Dict[str, object], status: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "TrafficDashboard/1.0"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    def _serve_static(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        data = path.read_bytes()
        if path.suffix == ".html":
            content_type = "text/html; charset=utf-8"
        elif path.suffix == ".css":
            content_type = "text/css; charset=utf-8"
        elif path.suffix == ".js":
            content_type = "application/javascript; charset=utf-8"
        elif path.suffix == ".svg":
            content_type = "image/svg+xml"
        else:
            content_type = "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/bootstrap":
            send_json(self, make_bootstrap())
            return
        if parsed.path in {"/", "/index.html"}:
            self._serve_static(UI_ROOT / "index.html")
            return
        if parsed.path == "/styles.css":
            self._serve_static(UI_ROOT / "styles.css")
            return
        if parsed.path == "/app.js":
            self._serve_static(UI_ROOT / "app.js")
            return
        if parsed.path == "/data.js":
            self._serve_static(UI_ROOT / "data.js")
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            mode = str(payload.get("mode", "serial"))
            mode_config = normalize_mode_config(mode, payload.get("config"))
            host = detect_host_profile()
            result = run_mode(mode, host, mode_config)
            send_json(self, {"ok": True, "result": result, "bootstrap": make_bootstrap()})
        except Exception as exc:  # noqa: BLE001
            send_json(
                self,
                {
                    "ok": False,
                    "error": str(exc),
                    "trace": traceback.format_exc(),
                    "bootstrap": make_bootstrap(),
                },
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def main() -> None:
    port = int(os.environ.get("TRAFFIC_DASHBOARD_PORT", DEFAULT_PORT))
    server = ThreadingHTTPServer(("127.0.0.1", port), DashboardHandler)
    print(f"Traffic dashboard running at http://127.0.0.1:{port}")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Host profile: {detect_host_profile()}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()