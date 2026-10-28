#!/usr/bin/env python3
"""Measure SigFlow programs compiled with the CUDA backend, one variant per set of sigflowc flags.

For each variant this compiles the program with `sigflowc --emit=cuda FLAGS`, builds it with
tools/build_cuda.sh, and runs it `--rounds` times, interleaving the variants so clock drift hits
all of them alike. Every run is `PROGRAM --bench N --warmup W`, and its JSON report gives
launches, GPU memory, and CUDA-event phase times. The summary is the median of the round means,
with their range, following the measurement rules in AGENTS.md.

With --nsys, each variant also runs once under Nsight Systems, and the number of GPU kernels it
recorded for one pipeline run is reported: the launch count as the GPU saw it, cuFFT's internal
kernels included, rather than as the program counted its own calls.

Inputs: --inputs fixtures runs on tests/fixtures. --inputs bench runs on the large files from
tools/gen_bench.py (out/bench by default): the program's fixtures/ch0.cf32, ch1.cf32, and
tone.cf32 become bench_ch0, bench_ch1, and bench_tone; filter taps still come from tests/fixtures.

    python compiler/tools/measure.py compiler/examples/two_channel.sf --variant= \\
        --variant=--no-reuse --inputs fixtures --rounds 5 --nsys

Write --variant=FLAGS with the "=", since the flags start with a dash; --variant= alone is the
default compilation with no extra flags.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import statistics
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH_NAMES = {"ch0.cf32": "bench_ch0.cf32", "ch1.cf32": "bench_ch1.cf32",
               "tone.cf32": "bench_tone.cf32"}


def run(args: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    proc = subprocess.run(args, capture_output=True, text=True, cwd=cwd, env=env)
    if proc.returncode != 0:
        raise SystemExit(f"failed: {' '.join(args)}\n{proc.stdout}\n{proc.stderr}")
    return proc.stdout


def make_fixtures(workdir: Path, inputs: str, bench_dir: Path) -> None:
    """workdir/fixtures: tests/fixtures itself, or a directory of links to the bench files."""
    fixtures = workdir / "fixtures"
    if inputs == "fixtures":
        fixtures.symlink_to(ROOT / "tests" / "fixtures", target_is_directory=True)
        return
    fixtures.mkdir()
    for entry in (ROOT / "tests" / "fixtures").iterdir():
        target = bench_dir / BENCH_NAMES[entry.name] if entry.name in BENCH_NAMES else entry
        if not target.exists():
            raise SystemExit(f"{target} missing; run tools/gen_bench.py --out {bench_dir}")
        (fixtures / entry.name).symlink_to(target.resolve())


def nsys_kernels(binary: Path, workdir: Path, nsys: str) -> int:
    """GPU kernels Nsight Systems recorded for one pipeline run (--bench 1, no warm-up)."""
    report = workdir / "nsys"
    run([nsys, "profile", "--force-overwrite=true", "--trace=cuda", "-o", str(report),
         str(binary), "--bench", "1", "--warmup", "0"], cwd=workdir)
    out = run([nsys, "stats", "--force-export=true", "--report", "cuda_gpu_kern_sum",
               "--format", "csv", "--output", "-", str(report) + ".nsys-rep"], cwd=workdir)
    # The CSV follows any notice lines; its header names an "Instances" column.
    lines = out.splitlines()
    start = next(i for i, line in enumerate(lines) if "Instances" in line)
    rows = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    return sum(int(row["Instances"]) for row in rows)


def median_range(values: list[float]) -> str:
    return f"{statistics.median(values):.4f} ({min(values):.4f} to {max(values):.4f})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("program", type=Path)
    parser.add_argument("--variant", action="append", default=[],
                        help="sigflowc flags for one variant, as --variant=--no-reuse; repeatable")
    parser.add_argument("--inputs", choices=["fixtures", "bench"], default="fixtures")
    parser.add_argument("--bench-dir", type=Path, default=ROOT / "out" / "bench")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--bench", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--sigflowc", default=os.environ.get(
        "SIGFLOWC_BIN", str(ROOT / "compiler" / "build" / "sigflowc")))
    parser.add_argument("--rfgpu-build", default=os.environ.get(
        "RFGPU_BUILD", str(ROOT / "build")))
    parser.add_argument("--nsys", nargs="?", const="nsys", default=None,
                        help="also count GPU kernels with Nsight Systems (optional: its path)")
    args = parser.parse_args()
    variants = args.variant or [""]

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        make_fixtures(workdir, args.inputs, args.bench_dir)
        binaries = []
        for index, flags in enumerate(variants):
            source = workdir / f"v{index}.cu"
            run([args.sigflowc, str(args.program.resolve()), *flags.split(), "--emit=cuda",
                 "-o", str(source)])
            binary = workdir / f"v{index}"
            run([str(ROOT / "compiler" / "tools" / "build_cuda.sh"), str(source), str(binary)],
                env={**os.environ, "RFGPU_BUILD": args.rfgpu_build})
            binaries.append(binary)

        reports: list[list[dict]] = [[] for _ in variants]
        for _ in range(args.rounds):
            for index, binary in enumerate(binaries):
                out = run([str(binary), "--bench", str(args.bench), "--warmup",
                           str(args.warmup)], cwd=workdir)
                reports[index].append(json.loads(out.strip().splitlines()[-1]))
        kernels = [nsys_kernels(b, workdir, args.nsys) if args.nsys else None for b in binaries]

    print(f"program: {args.program}, inputs: {args.inputs}, {args.rounds} rounds of "
          f"--bench {args.bench} --warmup {args.warmup}; times are median (range) of round means")
    for flags, rounds, gpu_kernels in zip(variants, reports, kernels):
        first = rounds[0]
        print(json.dumps({
            "flags": flags,
            "gpu": first["gpu"],
            "launches": first["launches"],
            "launches_by_kind": first["launches_by_kind"],
            "nsys_gpu_kernels": gpu_kernels,
            "slots": first["slots"],
            "slot_bytes": first["slot_bytes"],
            "taps_bytes": first["taps_bytes"],
            "fft_work_bytes": first["fft_work_bytes"],
            "allocated_bytes": first["allocated_bytes"],
            "device_used_delta_bytes": sorted({r["device_used_delta_bytes"] for r in rounds}),
            "h2d_ms": median_range([r["h2d_ms"] for r in rounds]),
            "kernel_ms": median_range([r["kernel_ms"] for r in rounds]),
            "kernel_std_ms_median": round(statistics.median(r["kernel_std_ms"] for r in rounds), 4),
            "d2h_ms": median_range([r["d2h_ms"] for r in rounds]),
            "total_ms_median": round(statistics.median(
                r["h2d_ms"] + r["kernel_ms"] + r["d2h_ms"] for r in rounds), 4),
        }))


if __name__ == "__main__":
    main()
