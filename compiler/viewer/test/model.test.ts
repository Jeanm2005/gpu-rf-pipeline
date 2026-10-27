import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { liveAfter, parseGraph } from "../src/model.js";

// Real compiler output: the golden file sigflowc's own tests pin (compiler/tests/golden), so the
// viewer is tested against exactly what the compiler writes. Compiled tests run from dist/test/.
const here = dirname(fileURLToPath(import.meta.url));
const golden = (name: string) =>
  readFileSync(join(here, "..", "..", "..", "tests", "golden", name), "utf8");

test("parses the compiler's JSON", () => {
  const result = parseGraph(golden("json_viewer.expected"));
  assert.ok(result.ok, result.ok ? "" : result.error);
  const { graph } = result;
  assert.deepEqual(graph.passes, ["build", "dce"]);
  assert.equal(graph.nodes.length, 7);
  assert.equal(graph.slots, 3);
  assert.deepEqual(graph.schedule, [0, 1, 2, 5, 6]);

  const la = graph.nodes[2];
  assert.equal(la.name, "la");
  assert.equal(la.taps, "fixtures/lowpass.f32");
  assert.deepEqual(la.liveRange, [2, 3]);
  assert.equal(la.bytes, 131072);

  const fft = graph.nodes[3];
  assert.equal(fft.live, false);
  assert.equal(fft.removedBy, "dce");
  assert.equal(fft.n, 1024);
});

test("each stage shows the nodes alive after that pass", () => {
  const result = parseGraph(golden("json_viewer.expected"));
  assert.ok(result.ok);
  assert.equal(liveAfter(result.graph, 0).size, 7); // build: everything
  assert.deepEqual([...liveAfter(result.graph, 1)].sort(), [0, 1, 2, 5, 6]); // after dce
});

test("rejects input that is not sigflowc JSON, with a reason", () => {
  const cases: [string, RegExp][] = [
    ["{", /not JSON/],
    ["[]", /"nodes" and "edges"/],
    ['{"nodes": [], "edges": []}', /rebuild sigflowc/],
    [
      '{"passes": [], "nodes": [{"id": 1}], "edges": [], "schedule": [], "slots": 0}',
      /ids must equal their index/,
    ],
    [
      '{"passes": [], "nodes": [{"id": 0, "op": "fuse"}], "edges": [], "schedule": [], "slots": 0}',
      /unknown op/,
    ],
    [
      '{"passes": [], "nodes": [], "edges": [{"from": 0, "to": 1, "input": 0}], "schedule": [], "slots": 0}',
      /does not exist/,
    ],
  ];
  for (const [text, message] of cases) {
    const result = parseGraph(text);
    assert.equal(result.ok, false, text);
    if (!result.ok) {
      assert.match(result.error, message);
    }
  }
});
