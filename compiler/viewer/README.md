# SigFlow IR viewer

A static page that draws the graph IR `sigflowc` produces, before and after each pass, with the
schedule and the buffer slots from liveness analysis. It reads the output of
`sigflowc PROGRAM.sf --emit=json` (format: `../include/sigflow/json.hpp`).

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
| `src/main.ts` | page wiring: file picker, drag and drop, `?src=` |
| `test/` | `node:test` suites, run against the compiler's golden JSON in `../tests/golden/` |
| `scripts/serve.mjs` | a tiny static server, since browsers won't load ES modules from `file://` |
