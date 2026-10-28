"""End-to-end: compile every program in compiler/examples/, run it on the GPU, check the output.

For each example, sigflowc writes the IR as JSON (--emit=json) and one of the two backends: the
phase A script of rfgpu commands, or the phase C CUDA program (--emit=cuda), which this test
builds with compiler/tools/build_cuda.sh against the librfgpu_ops.a next to rfgpu. Either runs in
a temporary directory next to a `fixtures` link to tests/fixtures. The JSON is then evaluated
node by node with tools/reference.py (float64), and every emitted file is compared to the
reference value of the node it emits, with the tolerance stated below for that node's operation.

The module is skipped when sigflowc or rfgpu is not built, or rfgpu cannot reach a GPU. Set
SIGFLOWC_BIN or RFGPU_BIN to test binaries somewhere else; the CUDA backend needs nvcc too.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
COMPILER = ROOT / "compiler"
FIXTURES = ROOT / "tests" / "fixtures"
SIGFLOWC = Path(os.environ.get("SIGFLOWC_BIN", COMPILER / "build" / "sigflowc"))
RFGPU = Path(os.environ.get("RFGPU_BIN", ROOT / "build" / "rfgpu"))
EXAMPLES = sorted((COMPILER / "examples").glob("*.sf"))
BUILD_CUDA = COMPILER / "tools" / "build_cuda.sh"
sys.path.insert(0, str(ROOT / "tools"))
import reference  # noqa: E402


def require_binaries() -> None:
    """Module-level skip unless both binaries exist and rfgpu can use a GPU."""
    for binary in (SIGFLOWC, RFGPU):
        if not binary.exists():
            pytest.skip(f"{binary} not built", allow_module_level=True)
    proc = subprocess.run([str(RFGPU), "info"], capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        pytest.skip(f"rfgpu cannot use a GPU: {proc.stderr.strip()}", allow_module_level=True)


require_binaries()

# Tolerances for an emitted value, keyed by the operation that produced it. Errors compound
# along a chain, so each entry covers every chain in examples/ that ends in that operation.
# Measured 2026-10-27 on the RTX 5080 Laptop GPU, CUDA 13.4, rfgpu's default --impl for each
# op, against the float64 reference; "left for atol" is max(|error| - rtol * |expected|):
#
#   capture    copied byte for byte.
#   fir        order-1 outputs; worst error 2.3e-6 (lowpass_long on tone_odd), all within rtol.
#   magnitude  up to fir -> fft(1024) -> magnitude. Values reach 1.03e3 on the tone (an
#              unnormalized FFT); worst error 2.4e-4, at most 7.8e-5 left for atol. atol leaves
#              a factor of 12.
#   xcorr      up to fir -> xcorr on ch0 / ch1. Worst error 1.1e-2 on the raw pair's 1.65e4 peak
#              and 1.1e-2 at |r| = 2.4e3 on the filtered pair; at most 3.7e-4 left for atol.
#              atol leaves a factor of 13.
TOLERANCES: dict[str, tuple[float, float]] = {
    "capture": (0.0, 0.0),
    "fir": (1e-5, 1e-5),
    "magnitude": (1e-5, 1e-3),
    "xcorr": (1e-5, 5e-3),
}


def run(*args: object, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    proc = subprocess.run([str(a) for a in args], capture_output=True, text=True, timeout=300,
                          cwd=cwd, env=env)
    assert proc.returncode == 0, f"{' '.join(map(str, args))}\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout


def evaluate(graph: dict, workdir: Path) -> dict[int, np.ndarray]:
    """Reference value of every non-emit node. Inputs always have smaller ids than their users
    (see compiler/include/sigflow/ir.hpp), so id order is a valid evaluation order."""
    inputs: dict[int, list[int]] = {}
    for edge in sorted(graph["edges"], key=lambda e: (e["to"], e["input"])):
        inputs.setdefault(edge["to"], []).append(edge["from"])

    values: dict[int, np.ndarray] = {}
    for node in sorted(graph["nodes"], key=lambda n: n["id"]):
        if not node["live"]:
            continue
        args = [values[i] for i in inputs.get(node["id"], [])]
        op = node["op"]
        if op == "capture":
            values[node["id"]] = reference.read_cf32(workdir / node["path"])
        elif op == "fir":
            taps = reference.read_f32(workdir / node["taps"])
            values[node["id"]] = reference.fir(args[0], taps)
        elif op == "fft":
            values[node["id"]] = reference.spectrum(args[0], node["n"]).ravel()
        elif op == "magnitude":
            values[node["id"]] = reference.magnitude(args[0])
        elif op == "xcorr":
            values[node["id"]] = reference.xcorr(args[0], args[1])
        else:
            assert op == "emit", f"unknown op {op}"
    return values


def compile_and_run(example: Path, workdir: Path, *flags: str, backend: str = "script") -> dict:
    """Compiles `example` with `flags` for `backend` ("script": the phase A rfgpu script;
    "cuda": the phase C program, built with tools/build_cuda.sh against RFGPU_BIN's build
    directory), runs it in `workdir`, and returns its IR as JSON."""
    (workdir / "fixtures").symlink_to(FIXTURES, target_is_directory=True)
    graph = json.loads(run(SIGFLOWC, example, "--emit=json"))
    if backend == "script":
        script = workdir / "run.sh"
        run(SIGFLOWC, example, *flags, "-o", script)
        run("bash", script, cwd=workdir, env={**os.environ, "RFGPU": str(RFGPU)})
    else:
        source = workdir / "run.cu"
        run(SIGFLOWC, example, *flags, "--emit=cuda", "-o", source)
        run(BUILD_CUDA, source, workdir / "run",
            env={**os.environ, "RFGPU_BUILD": str(RFGPU.parent)})
        run(workdir / "run", cwd=workdir)
    return graph


# Both backends, with and without buffer reuse. With reuse, a later value overwrites a dead
# value's buffer (a work file in the script, a GPU buffer in the CUDA program), so a liveness bug
# that frees a buffer too early shows up here as a wrong output; --no-reuse is the baseline.
@pytest.mark.parametrize("flags", [(), ("--no-reuse",)], ids=["reuse", "no_reuse"])
@pytest.mark.parametrize("backend", ["script", "cuda"])
@pytest.mark.parametrize("example", EXAMPLES, ids=[e.stem for e in EXAMPLES])
def test_example_matches_reference(tmp_path: Path, example: Path, backend: str,
                                   flags: tuple[str, ...]) -> None:
    graph = compile_and_run(example, tmp_path, *flags, backend=backend)
    nodes = {node["id"]: node for node in graph["nodes"]}
    sources = {edge["to"]: edge["from"] for edge in graph["edges"]
               if nodes[edge["to"]]["op"] == "emit"}
    values = evaluate(graph, tmp_path)

    emits = [node for node in graph["nodes"] if node["op"] == "emit"]
    assert emits, "every example emits something"
    for emit in emits:
        source = nodes[sources[emit["id"]]]
        path = tmp_path / emit["path"]
        is_real = source["type"].startswith("frames<f32")
        actual = reference.read_f32(path) if is_real else reference.read_cf32(path)
        expected = values[source["id"]]
        rtol, atol = TOLERANCES[source["op"]]
        assert actual.shape == expected.shape, f"{emit['path']}: shape"
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol,
                                   err_msg=f"{example.name}: {emit['path']}")


@pytest.mark.parametrize("example", EXAMPLES, ids=[e.stem for e in EXAMPLES])
def test_example_rates_match_the_fixtures(example: Path) -> None:
    """A capture's declared rate must be the one in the fixture's .json sidecar."""
    graph = json.loads(run(SIGFLOWC, example, "--emit=json"))
    for node in graph["nodes"]:
        if node["op"] == "capture":
            sidecar = FIXTURES / Path(node["path"]).with_suffix(".json").name
            meta = json.loads(sidecar.read_text())
            assert float(node["rate"]) == meta["sample_rate"], f"{example.name}: {node['path']}"


def test_delay_example_recovers_the_known_delay(tmp_path: Path) -> None:
    example = COMPILER / "examples" / "delay.sf"
    compile_and_run(example, tmp_path)
    r = reference.read_cf32(tmp_path / "out" / "delay_xcorr.cf32")
    num_a = len(reference.read_cf32(FIXTURES / "ch0.cf32"))
    num_b = len(reference.read_cf32(FIXTURES / "ch1.cf32"))
    lags = reference.xcorr_lags(num_a, num_b)
    expected = json.loads((FIXTURES / "ch1.json").read_text())["true_delay_samples"]
    assert lags[int(np.argmax(np.abs(r)))] == expected
