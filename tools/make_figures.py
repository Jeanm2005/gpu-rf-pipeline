#!/usr/bin/env python3
"""Collect profiling data into docs/results/ and render the charts in docs/figures/.

    python tools/make_figures.py collect-ncu      # needs ncu (reads saved reports; no GPU)
    python tools/make_figures.py collect-times    # needs a GPU, build/rfgpu, out/bench inputs
    python tools/make_figures.py plot             # needs only matplotlib and docs/results/

The data files are committed, so `plot` reproduces every figure without profiling again:

  docs/results/ncu_metrics.csv     per-kernel Nsight Compute metrics from docs/profiles/*.ncu-rep
  docs/results/rfgpu_times.csv     CUDA-event times for every rfgpu variant, one row per round
  docs/results/sigflow_*.json      compiler/tools/measure.py summaries (--json-out)

Every figure is written twice, NAME.png for light backgrounds and NAME-dark.png for dark ones,
so the README can pick one with <picture> and prefers-color-scheme.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"
FIGURES = ROOT / "docs" / "figures"
PROFILES = ROOT / "docs" / "profiles"

NCU_METRICS = {
    "duration_ms": "gpu__time_duration.sum",
    "memory_pct": "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "compute_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
}

TO_MS = {"nsecond": 1e-6, "ns": 1e-6, "usecond": 1e-3, "us": 1e-3, "msecond": 1.0, "ms": 1.0,
         "second": 1e3, "s": 1e3}

# rfgpu configurations timed by collect-times: (label, op, impl, taps file or None).
RFGPU_CONFIGS = [
    ("fir naive 63", "fir", "naive", "lowpass.f32"),
    ("fir tiled 63", "fir", "tiled", "lowpass.f32"),
    ("fir tiled_const 63", "fir", "tiled_const", "lowpass.f32"),
    ("fir naive 1025", "fir", "naive", "lowpass_long.f32"),
    ("fir tiled 1025", "fir", "tiled", "lowpass_long.f32"),
    ("fir tiled_const 1025", "fir", "tiled_const", "lowpass_long.f32"),
    ("xcorr naive", "xcorr", "naive", None),
    ("xcorr tiled", "xcorr", "tiled", None),
    ("xcorr fft", "xcorr", "fft", None),
]


# --- collect -------------------------------------------------------------------------------

def run(args: list[str]) -> str:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"failed: {' '.join(args)}\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def collect_ncu(ncu: str) -> None:
    rows = []
    for report in sorted(PROFILES.glob("*.ncu-rep")):
        out = run([ncu, "--import", str(report), "--csv", "--page", "raw",
                   "--metrics", ",".join(NCU_METRICS.values())])
        lines = out.splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith('"ID"'))
        table = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
        units = table[0]  # the first row after the header holds each column's unit
        for entry in table[1:]:
            row = {"report": report.stem, "kernel": entry["Kernel Name"].split("(")[0].strip()}
            for name, metric in NCU_METRICS.items():
                value = float(entry[metric].replace(",", ""))
                # ncu picks the duration unit per report (ms for long kernels, us for short).
                row[name] = value * TO_MS.get(units[metric], 1.0) if name == "duration_ms" else value
            rows.append(row)
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "ncu_metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["report", "kernel", *NCU_METRICS])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} kernels to {RESULTS / 'ncu_metrics.csv'}")


def collect_times(rfgpu: Path, bench_dir: Path, rounds: int, runs: int, warmup: int) -> None:
    fixtures = ROOT / "tests" / "fixtures"
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    rows = []
    for round_index in range(rounds):  # interleaved, so clock drift hits every variant alike
        for label, op, impl, taps in RFGPU_CONFIGS:
            if op == "fir":
                args = ["fir", "--in", str(bench_dir / "bench_tone.cf32"),
                        "--taps", str(fixtures / taps)]
            else:
                args = ["xcorr", "--a", str(bench_dir / "bench_ch0.cf32"),
                        "--b", str(bench_dir / "bench_ch1.cf32")]
            report = json.loads(run([str(rfgpu), *args, "--out", str(out / "figures_tmp.cf32"),
                                     "--impl", impl, "--bench", str(runs),
                                     "--warmup", str(warmup)]))
            rows.append({"config": label, "round": round_index, "gpu": report["gpu"],
                         "kernel_ms": report["kernel_ms"]["mean"],
                         "h2d_ms": report["h2d_ms"]["mean"], "d2h_ms": report["d2h_ms"]["mean"]})
            print(f"round {round_index}: {label}: {report['kernel_ms']['mean']:.4f} ms")
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "rfgpu_times.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# --- plot ----------------------------------------------------------------------------------

# Colors from a categorical palette validated for color-vision deficiency, stepped separately
# for each surface. Text is always in text colors, never in a series color.
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "series": ["#2a78d6", "#eb6834", "#1baf7a"]},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "muted": "#c3c2b7", "grid": "#2c2c2a",
             "axis": "#383835", "series": ["#3987e5", "#d95926", "#199e70"]},
}


def medians(rows: list[dict], key: str) -> dict[str, tuple[float, float, float]]:
    """config -> (median, min, max) of the per-round values of `key`."""
    by_config: dict[str, list[float]] = {}
    for row in rows:
        by_config.setdefault(row["config"], []).append(float(row[key]))
    return {c: (statistics.median(v), min(v), max(v)) for c, v in by_config.items()}


def style_axes(ax, theme: dict) -> None:
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["axis"])
    ax.tick_params(colors=theme["muted"], length=0, labelsize=9)
    ax.xaxis.grid(True, color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name: str, theme_name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    suffix = "" if theme_name == "light" else "-dark"
    fig.savefig(FIGURES / f"{name}{suffix}.png", dpi=160, facecolor=fig.get_facecolor(),
                bbox_inches="tight")


def bar_panel(ax, theme, labels, values, ranges, color, unit, log=False, fmt="{:.3g}"):
    """Horizontal bars, first label on top, each with its value printed at the end and a thin
    line for the range of the round means."""
    ys = list(range(len(labels)))[::-1]
    if log:
        # A log axis has no zero for a bar to start from, so values are dots.
        ax.scatter(values, ys, s=64, color=color, zorder=3)
    else:
        ax.barh(ys, values, height=0.56, color=color, zorder=2)
    for y, value, (low, high) in zip(ys, values, ranges):
        ax.plot([low, high], [y, y], color=theme["text"], linewidth=1, zorder=4)
        ax.text(high * (1.4 if log else 1.0) + (0 if log else max(values) * 0.02), y,
                f"{fmt.format(value)} {unit}", va="center", ha="left", color=theme["text"],
                fontsize=9)
    ax.set_yticks(ys, labels, color=theme["text"], fontsize=9)
    if log:
        from matplotlib.ticker import FuncFormatter

        ax.set_xscale("log")
        ax.set_xlim(min(v[0] for v in ranges) / 3, max(v[1] for v in ranges) * 12)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.set_ylim(-0.6, len(labels) - 0.4)
    else:
        ax.set_xlim(0, max(v[1] for v in ranges) * 1.3)
    style_axes(ax, theme)


def plot_fir(plt, theme_name: str, times: dict) -> None:
    theme = THEMES[theme_name]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 2.6), facecolor=theme["surface"])
    variants = ["naive", "tiled", "tiled_const"]
    labels = ["naive", "shared-memory tile", "tile + constant-memory taps"]
    for ax, taps in zip(axes, ("63", "1025")):
        stats = [times[f"fir {v} {taps}"] for v in variants]
        bar_panel(ax, theme, labels, [s[0] for s in stats], [(s[1], s[2]) for s in stats],
                  theme["series"][0], "ms")
        ax.set_title(f"{taps} taps", color=theme["text"], fontsize=10, loc="left")
        ax.set_xlabel("kernel time (ms)", color=theme["muted"], fontsize=9)
    axes[1].set_yticklabels([])
    fig.suptitle("FIR filter, 2^24 complex samples: kernel time by variant "
                 "(median of 5 rounds; line: range)",
                 color=theme["text"], fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, "fir_kernel_time", theme_name)
    plt.close(fig)


def plot_xcorr(plt, theme_name: str, times: dict) -> None:
    theme = THEMES[theme_name]
    fig, ax = plt.subplots(figsize=(9.6, 2.0), facecolor=theme["surface"])
    variants = ["naive", "tiled", "fft"]
    labels = ["naive (time domain)", "shared-memory tile", "cuFFT-based"]
    stats = [times[f"xcorr {v}"] for v in variants]
    bar_panel(ax, theme, labels, [s[0] for s in stats], [(s[1], s[2]) for s in stats],
              theme["series"][0], "ms", log=True)
    ax.set_xlabel("kernel time (ms, log scale)", color=theme["muted"], fontsize=9)
    ax.set_title("Cross-correlation, 2^18 samples per channel: kernel time by variant "
                 "(median of 5 rounds; line: range)", color=theme["text"], fontsize=11, loc="left")
    fig.tight_layout()
    save(fig, "xcorr_kernel_time", theme_name)
    plt.close(fig)


def plot_ncu(plt, theme_name: str, metrics: list[dict]) -> None:
    """Memory throughput and achieved occupancy per kernel: two measures on one percent axis,
    so two series with a legend and every bar labelled."""
    theme = THEMES[theme_name]
    order = [("fir_naive_1025", "fir_naive_kernel", "FIR naive"),
             ("fir_tiled_1025", "fir_tiled_kernel", "FIR tiled"),
             ("fir_tiled_const_1025", "fir_tiled_const_kernel", "FIR tiled + const taps"),
             ("xcorr_naive", "xcorr_naive_kernel", "xcorr naive"),
             ("xcorr_tiled", "xcorr_tiled_kernel", "xcorr tiled"),
             ("xcorr_fft", "xcorr_conj_mul_kernel", "xcorr fft (conj. multiply)")]
    found = {(m["report"], m["kernel"]): m for m in metrics}
    rows = [(label, found[(r, k)]) for r, k, label in order if (r, k) in found]
    fig, ax = plt.subplots(figsize=(9.6, 3.4), facecolor=theme["surface"])
    ys = list(range(len(rows)))[::-1]
    height = 0.36
    for offset, key, name, color in ((height / 2, "memory_pct", "memory throughput",
                                      theme["series"][0]),
                                     (-height / 2, "occupancy_pct", "achieved occupancy",
                                      theme["series"][1])):
        values = [float(m[key]) for _, m in rows]
        # A 2 px surface gap between the two bars of a pair keeps them separate marks.
        ax.barh([y + offset for y in ys], values, height=height - 0.03, color=color, label=name,
                zorder=2)
        for y, value in zip(ys, values):
            ax.text(value + 1, y + offset, f"{value:.0f} %", va="center", ha="left",
                    color=theme["text"], fontsize=8)
    ax.set_yticks(ys, [label for label, _ in rows], color=theme["text"], fontsize=9)
    ax.set_xlim(0, 112)
    ax.set_xlabel("% of peak, from Nsight Compute (FIR kernels with 1025 taps)",
                  color=theme["muted"], fontsize=9)
    style_axes(ax, theme)
    legend = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
                       fontsize=9)
    for text in legend.get_texts():
        text.set_color(theme["text"])
    ax.set_title("Memory throughput and achieved occupancy per kernel", color=theme["text"],
                 fontsize=11, loc="left", pad=24)
    fig.tight_layout()
    save(fig, "ncu_utilization", theme_name)
    plt.close(fig)


def plot_sigflow(plt, theme_name: str, summary: dict) -> None:
    """End-to-end time for each sigflowc variant, split into upload, kernels, and download."""
    theme = THEMES[theme_name]
    variants = summary["variants"]
    fig, ax = plt.subplots(figsize=(9.6, 0.6 + 0.55 * len(variants)), facecolor=theme["surface"])
    ys = list(range(len(variants)))[::-1]
    parts = (("h2d_ms", "upload"), ("kernel_ms", "kernels"), ("d2h_ms", "download"))
    left = [0.0] * len(variants)
    for (key, name), color in zip(parts, theme["series"]):
        values = [v[key] for v in variants]
        # Segments separated by a thin surface-colored edge (the 2 px gap between fills).
        ax.barh(ys, values, left=left, height=0.56, color=color, label=name,
                edgecolor=theme["surface"], linewidth=1.5, zorder=2)
        left = [a + b for a, b in zip(left, values)]
    for y, total, v in zip(ys, left, variants):
        ax.text(total + max(left) * 0.01, y, f"{total:.3g} ms  ·  {v['launches']} launches",
                va="center", ha="left", color=theme["text"], fontsize=9)
    ax.set_yticks(ys, [v["label"] for v in variants], color=theme["text"], fontsize=9)
    ax.set_xlim(0, max(left) * 1.35)
    ax.set_xlabel("end-to-end time per pipeline run (ms, median of 5 rounds)",
                  color=theme["muted"], fontsize=9)
    style_axes(ax, theme)
    legend = ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
                       fontsize=9)
    for text in legend.get_texts():
        text.set_color(theme["text"])
    ax.set_title(summary["title"], color=theme["text"], fontsize=11, loc="left", pad=24)
    fig.tight_layout()
    save(fig, summary["figure"], theme_name)
    plt.close(fig)


def plot() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "DejaVu Sans"
    with open(RESULTS / "rfgpu_times.csv") as f:
        times = medians(list(csv.DictReader(f)), "kernel_ms")
    with open(RESULTS / "ncu_metrics.csv") as f:
        metrics = list(csv.DictReader(f))
    summaries = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("sigflow_*.json"))]
    for theme_name in THEMES:
        plot_fir(plt, theme_name, times)
        plot_xcorr(plt, theme_name, times)
        plot_ncu(plt, theme_name, metrics)
        for summary in summaries:
            plot_sigflow(plt, theme_name, summary)
    print(f"wrote figures to {FIGURES}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    ncu = sub.add_parser("collect-ncu")
    ncu.add_argument("--ncu", default="ncu")
    times = sub.add_parser("collect-times")
    times.add_argument("--rfgpu", type=Path,
                       default=Path(os.environ.get("RFGPU_BIN", ROOT / "build" / "rfgpu")))
    times.add_argument("--bench-dir", type=Path, default=ROOT / "out" / "bench")
    times.add_argument("--rounds", type=int, default=5)
    times.add_argument("--runs", type=int, default=20)
    times.add_argument("--warmup", type=int, default=3)
    sub.add_parser("plot")
    args = parser.parse_args()

    if args.command == "collect-ncu":
        collect_ncu(args.ncu)
    elif args.command == "collect-times":
        collect_times(args.rfgpu, args.bench_dir, args.rounds, args.runs, args.warmup)
    else:
        plot()


if __name__ == "__main__":
    main()
