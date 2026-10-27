# SigFlow IR viewer

A static page that draws the graph IR `sigflowc` produces, before and after each pass, with the
schedule and the buffer slots from liveness analysis:

- **Graph:** the DAG laid out left to right in layers. Pick a pass to see the graph after it;
  nodes a pass removed stay in place as faded outlines (or hide them). Each value shows its
  buffer slot (`b2`, with the slot's color along its edge), schedule step, and size.
- **Buffer slots over the schedule:** one lane per slot, one bar per value from the step that
  writes it to the last step that reads it, so a lane with several bars is a reused buffer.
- **Schedule:** the same data as a table. Selecting a node anywhere highlights it everywhere.

It reads the output of `sigflowc PROGRAM.sf --emit=json` (format:
`../include/sigflow/json.hpp`).

TypeScript compiled by `tsc` alone, no bundler or framework, and no runtime dependencies.

```bash
cd compiler/viewer
npm ci
npm test          # type-check, compile to dist/, run the node:test suites in test/
npm run serve     # http://localhost:8000/viewer/ (serves compiler/, so ?src= can reach outputs)
```

Then open a JSON file with the picker, drop it on the page, or link to one:

```bash
cd compiler
./build/sigflowc examples/two_channel.sf --emit=json --input-dir=../tests > out/two_channel.json
# http://localhost:8000/viewer/?src=/out/two_channel.json
```

`--input-dir` lets `sigflowc` read the capture files so each buffer shows its size in bytes.

## Layout

| Path | What |
|---|---|
| `src/model.ts` | JSON types and a validating parser; which nodes are alive after each pass |
| `src/layout.ts` | layered layout: longest-path layers, barycenter sweeps to cut crossings |
| `src/render.ts` | SVG for the graph and the buffer timeline |
| `src/main.ts` | page wiring: file picker, drag and drop, `?src=`, pass selection |
| `test/` | `node:test` suites, run against the compiler's golden JSON in `../tests/golden/` |
| `scripts/serve.mjs` | a tiny static server, since browsers won't load ES modules from `file://` |
