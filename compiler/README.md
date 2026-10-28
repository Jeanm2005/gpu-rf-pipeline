# sigflowc — a compiler for GPU signal-processing pipelines

`sigflowc` compiles **SigFlow**, a small DSL (domain-specific language) for describing radio
signal-processing pipelines, into an execution plan for the hand-written CUDA kernels in
[`gpu-rf-pipeline`](../README.md).

```
# examples/two_channel.sf
a    = capture("fixtures/ch0.cf32", rate=2.4e6)
b    = capture("fixtures/ch1.cf32", rate=2.4e6)
la   = a |> fir(taps="fixtures/lowpass.f32")
lb   = b |> fir(taps="fixtures/lowpass.f32")
spec = la |> fft(n=1024) |> magnitude
corr = xcorr(la, lb)

emit spec -> "out/spectrum.f32"
emit corr -> "out/xcorr.cf32"
```

## Why a compiler

A pipeline is a graph: each stage depends on earlier ones. Running it well means deciding the
launch order, which GPU buffers can be reused, and which stages can be merged into one kernel so
data doesn't make extra round-trips through GPU memory. Those are compiler problems: parsing,
semantic analysis, graph IR (intermediate representation), scheduling, dataflow analysis, and
code generation.

## Compiler stages

| Stage | What it does | Phase | Status |
|---|---|---|---|
| Lexer | Source text → tokens, with line/column positions | A | ✅ |
| Parser | Recursive descent → AST (abstract syntax tree) | A | ✅ |
| Semantic analysis | Name resolution; type, argument, and sample-rate checks | A | ✅ |
| IR builder | AST → DAG (directed acyclic graph) of operations | A | ✅ |
| Scheduler | Topological sort (Kahn's algorithm), cycle detection, source-order tie-breaking | A | ✅ |
| Dead-node elimination | Drops stages that no `emit` depends on | A | ✅ |
| Script backend | Emits an `rfgpu` command script | A | ✅ |
| Liveness analysis | Finds when each buffer is last used, so GPU memory can be reused | B | ✅ |
| DAG viewer | TypeScript page drawing the IR before and after each pass | B | ✅ |
| Kernel fusion | Merges compatible adjacent stages into one kernel | C | ⬜ |
| CUDA backend | Emits CUDA C++ calling the existing and fused kernels | C | ⬜ |

## The language

### Grammar (EBNF, extended Backus–Naur form)

```
program    = { statement } EOF ;
statement  = ( binding | emit ) NEWLINE ;
binding    = IDENT "=" expr ;
emit       = "emit" IDENT "->" STRING ;
expr       = term { "|>" call } ;               (* a |> f(x) means f(a, x) *)
term       = call | IDENT | literal | "(" expr ")" ;
call       = IDENT "(" [ arg_list ] ")" | IDENT ;
arg_list   = arg { "," arg } ;
arg        = IDENT "=" literal | expr ;         (* keyword args come after positional *)
literal    = NUMBER | STRING ;

IDENT      = letter { letter | digit | "_" } ;
NUMBER     = digit { digit } [ "." digit { digit } ] [ ( "e" | "E" ) [ "-" ] digit { digit } ] ;
STRING     = '"' { any character except '"' or newline } '"' ;
comment    = "#" { any character } NEWLINE ;
```

The grammar is a starting draft. Ambiguities and changes found while building the parser are
recorded here:

- `term` accepts a `literal` (2026-10-27). The first draft had none, so `capture`'s positional
  path string could not be parsed.
- `call = IDENT` and `term = IDENT` overlap. The parser reads a bare `IDENT` as a call only on
  the right of `|>` (`x |> magnitude`); everywhere else it is a reference to a binding.
- Bindings are declarative (2026-10-27): a name may be used on a line before the one that
  defines it. That is what makes the program a graph to schedule rather than a list, and what
  makes a cycle (`x = xcorr(a, y)`, `y = x |> fir(...)`) possible and worth reporting.
- `arg` needs two tokens of lookahead: `IDENT "="` starts a keyword argument, any other `IDENT`
  starts an expression.
- Lexing details: a `+` is accepted after the exponent marker (`1e+6`), and `\r\n` line endings
  are accepted.

### Types

| Type | Meaning |
|---|---|
| `stream<cf32>` | Continuous complex float32 samples, with a sample rate |
| `frames<cf32, N>` | Blocks of N complex samples (FFT output) |
| `frames<f32, N>` | Blocks of N real samples (e.g., magnitude spectrum) |

### Built-in operations

| Operation | Signature | GPU implementation |
|---|---|---|
| `capture(path, rate=)` | → `stream<cf32>` | file read |
| `fir(s, taps=)` | `stream<cf32>` → `stream<cf32>` | custom kernel |
| `fft(s, n=)` | `stream<cf32>` → `frames<cf32, n>` | cuFFT |
| `magnitude(f)` | `frames<cf32, N>` → `frames<f32, N>` | element-wise |
| `xcorr(a, b)` | `stream<cf32>`, `stream<cf32>` → `stream<cf32>` (same rate required) | custom kernel |

### Example errors

```
examples/bad_rate.sf:4:8: error: xcorr inputs have different sample rates (2.4e6 vs 1.2e6)
  c = xcorr(a, b)
      ^~~~~
```

## Quick start

```bash
# Build the GPU pipeline first (see ../README.md), then:
cd compiler
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j

./build/sigflowc examples/two_channel.sf --emit=schedule
./build/sigflowc examples/two_channel.sf -o out/run.sh
ln -sfn ../tests/fixtures fixtures       # the examples read fixtures/*.cf32 (git-ignored link)
RFGPU=../build/rfgpu bash out/run.sh     # writes out/spectrum.f32 and out/xcorr.cf32

# Phase C backend: a CUDA program that keeps every value on the GPU
./build/sigflowc examples/two_channel.sf --emit=cuda -o out/run.cu
tools/build_cuda.sh out/run.cu out/run   # links ../build/librfgpu_ops.a
out/run --bench 20 --warmup 3            # same outputs, plus a JSON report of launches,
                                         # GPU memory, and phase times

ctest --test-dir build --output-on-failure
pytest -q tests/e2e
```

## How it is tested

- **Unit tests** (GoogleTest): lexer, scheduler (ordering, tie-breaking, cycles), and later
  liveness and fusion on hand-built graphs.
- **Golden tests:** each `tests/golden/*.sf` has an expected `--emit` output or expected
  diagnostics, compared exactly.
- **End-to-end tests** (pytest, `tests/e2e/`): every program in `examples/` is compiled, its
  script is run on the fixtures, and every emitted file is compared to the NumPy/SciPy reference
  from `../tools/reference.py`, evaluated on the same IR (`--emit=json`). Needs `../build/rfgpu`
  and a GPU; skips otherwise. `SIGFLOWC_BIN` and `RFGPU_BIN` point it at other builds.

## Results

> Filled in from real measurements only (phase C).

| Program | Kernel launches (unfused → fused) | GPU memory peak (no reuse → reuse) | End-to-end time (unfused → fused) |
|---|---|---|---|
| `two_channel.sf` | — | — | — |

### Planned GPU memory (phase B)

Computed by `sigflowc --emit=liveness --input-dir=../tests` (and `--no-reuse`) from the fixture
sizes: the bytes every value needs if each gets its own buffer, against one buffer per slot after
liveness analysis. These are compile-time plans, not measurements. The phase A backend runs each
`rfgpu` command as its own process, so GPU buffers are not shared between operations and reuse
cannot show up in a real GPU-memory measurement until the phase C CUDA backend keeps buffers on
the device; that measurement goes in the table above.

| Program | Values | Buffers (no reuse → reuse) | Bytes (no reuse → reuse) | Saved | Busiest step |
|---|---|---|---|---|---|
| `examples/two_channel.sf` | 7 | 7 → 4 | 983,032 → 589,816 | 40.0% | 589,816 |
| `examples/spectrum.sf` | 3 | 3 → 2 | 1,310,720 → 1,048,576 | 20.0% | 1,048,576 |
| `examples/delay.sf` | 5 | 5 → 4 | 684,392 → 604,336 | 11.7% | 604,336 |

In all three, the plan with reuse equals the bytes live at the busiest step, which no plan can go
below, so first-fit assignment is already optimal for these programs.

## Roadmap

1. **Phase A:** front end, graph IR, topological scheduling, dead-node elimination, script backend.
2. **Phase B:** liveness analysis and buffer reuse; TypeScript DAG viewer.
3. **Phase C:** kernel fusion and CUDA code generation, measured with Nsight Systems and Nsight
   Compute.
4. **Later:** streaming execution (CUDA streams), direction-finding operations (GCC-PHAT,
   generalized cross-correlation with phase transform), a parser fuzzer (libFuzzer).

## Related work

Halide, TVM, and XLA compile array and tensor programs with graph IRs and kernel fusion. GNU
Radio represents radio pipelines as flowgraphs but schedules them at runtime rather than
compiling and fusing them. SigFlow is a small, domain-specific take on the same ideas, built on
hand-written kernels so every fusion decision can be measured.

## Working with AI agents

Agent rules for this folder live in [`AGENTS.md`](AGENTS.md). The core compiler passes are
written by hand by the author; agents help with tooling, tests, and reviews.
