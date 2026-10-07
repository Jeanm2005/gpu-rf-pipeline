# AGENTS.md — compiler/

Instructions for AI coding agents (Codex, GitHub Copilot, Claude Code, and others) working on
files under `compiler/`. This file adds to the repository root `AGENTS.md`; where they
conflict, this file wins for anything under `compiler/`.

## What this is

`sigflowc` compiles SigFlow, a small domain-specific language (DSL) for radio signal-processing
pipelines, into an execution plan for the GPU kernels in `../src/`. It parses a `.sf` program,
checks it, builds a graph IR (intermediate representation) shaped as a DAG (directed acyclic
graph), schedules it with a topological sort, and generates code: a script of `rfgpu` commands
in phase A, and generated CUDA C++ with fused kernels in phase C. See `README.md` for the
language and the phase plan.

## The most important rule: the owner writes the core compiler

This is a learning and portfolio project. The repository owner (Jean) writes these himself and
must be able to explain every line in interviews:

| Owner-written (agents: explain, sketch, review — do NOT implement) | Files |
|---|---|
| Lexer | `src/lexer.cpp` |
| Recursive-descent parser | `src/parser.cpp` |
| Semantic analysis (name resolution, type and rate checking) | `src/sema.cpp` |
| Topological-sort scheduler and cycle detection | `src/schedule.cpp` |
| Liveness analysis and buffer reuse (phase B) | `src/liveness.cpp` |
| Kernel fusion pass (phase C) | `src/fuse.cpp` |
| Custom CUDA kernels (unchanged rule from root `AGENTS.md`) | `../src/*.cu` |

For these files:
- **Do NOT write or complete function bodies** unless the owner explicitly asks in the current
  request.
- You MAY: explain the algorithm and point to the matching Dragon Book chapter, write signatures
  and `// TODO` skeletons, write pseudocode, review his code, find bugs, and propose test cases
  that would break it.
- Copilot inline completions in these files should stay to a single line.

Agents MAY write freely: `src/main.cpp` (CLI and flag parsing), the diagnostics printer, data
structure boilerplate in headers once the owner has designed them, `src/backend_script.cpp`,
CMake, GoogleTest and golden-test infrastructure, the pytest end-to-end harness, docs, and
scaffolding for `viewer/` (the owner writes the graph layout and rendering logic).

## Build, run, test

```bash
# From compiler/
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug     # Release for timing in phase C
cmake --build build -j

# Compile a program
./build/sigflowc examples/spectrum.sf --emit=tokens     # lexer output
./build/sigflowc examples/spectrum.sf --emit=ast        # parsed AST
./build/sigflowc examples/spectrum.sf --emit=ir         # graph IR, text form
./build/sigflowc examples/spectrum.sf --emit=schedule   # topologically sorted launch order
./build/sigflowc examples/spectrum.sf --emit=json       # IR as JSON for the viewer
./build/sigflowc examples/spectrum.sf -o out/run.sh     # phase A backend

# Tests
ctest --test-dir build --output-on-failure     # GoogleTest unit tests + golden tests
pytest -q tests/e2e                            # end-to-end (needs ../build/rfgpu and a GPU)

# Update golden files ONLY when the owner asks, then show him the diff
./build/sigflowc --update-golden tests/golden
```

## Design rules

- **Pipeline order is fixed:** lexer → parser → AST → sema → IR builder → passes → backend.
  Each stage only consumes the previous stage's output. No backend code reads the AST.
- **No exceptions for user errors.** Every user mistake becomes a diagnostic with file, line,
  column, a message, and the source line with a caret under the problem. Collect several
  diagnostics per run when it's safe to continue. Internal bugs use `assert`.
- **Deterministic output.** The same input always produces byte-identical `--emit` output. When
  the topological sort has a choice, break ties by source order. Never iterate an unordered
  container when producing output.
- **IR is the contract.** Every pass takes IR and returns IR, and must keep the `--emit=ir`
  dump valid. Each pass can be turned off with a flag (`--no-dce`, `--no-fuse`) so its effect
  can be measured.
- C++17, `include/sigflow/` for headers, `snake_case` functions, `PascalCase` types, no global
  mutable state, `std::unique_ptr` ownership for AST nodes.

## Testing rules

- Every grammar rule has at least one positive and one negative golden test.
- Required negative tests: unknown name, wrong argument count, unknown keyword argument, type
  mismatch (e.g., `magnitude` applied to a stream rather than frames), sample-rate mismatch in
  `xcorr`, a cycle, an unterminated string, a stray character, and an empty program.
- Golden files compare exact text. A golden change must be explained in the commit message.
- Unit tests (GoogleTest) cover the lexer, the scheduler (including tie-breaking and cycles),
  and later liveness and fusion on hand-built IR graphs.
- End-to-end tests compile every program in `examples/`, run it on the fixtures in
  `../tests/fixtures`, and compare outputs to `../tools/reference.py` with explicit tolerances.

## Measurement rules (phase C)

Fusion and buffer-reuse claims follow the root `AGENTS.md` measurement rules: Nsight or CUDA
events, warm-up, averaged runs, transfers reported separately, and an entry in
`../docs/PROFILING_LOG.md` comparing `--no-fuse` against fused output. Never invent numbers.

## Scope

Work on the current phase only (see `README.md`). Ideas for later phases go in
`docs/ROADMAP.md`, not into code.

## Git

Small commits, one working step each, imperative messages ("Add Kahn scheduler with
source-order tie-breaking"). Don't push or force-push unless asked.
