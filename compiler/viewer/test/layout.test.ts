import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { assignLayers, countCrossings, layoutGraph, NODE_HEIGHT, NODE_WIDTH } from "../src/layout.js";
import { IrEdge, IrGraph, IrNode, parseGraph } from "../src/model.js";

const here = dirname(fileURLToPath(import.meta.url));
const golden = (name: string): IrGraph => {
  const result = parseGraph(
    readFileSync(join(here, "..", "..", "..", "tests", "golden", name), "utf8"));
  assert.ok(result.ok);
  return result.graph;
};

// A hand-built graph: node i is written on line `lines[i]`; edges are [from, to] pairs.
function graphOf(lines: number[], edges: [number, number][]): IrGraph {
  const nodes: IrNode[] = lines.map((line, id) => ({ id, op: "fir", line, column: 1, live: true }));
  const inputs = new Map<number, number>();
  const irEdges: IrEdge[] = edges.map(([from, to]) => {
    const input = inputs.get(to) ?? 0;
    inputs.set(to, input + 1);
    return { from, to, input };
  });
  return { passes: ["build"], nodes, edges: irEdges, schedule: [], slots: 0 };
}

test("layers are longest paths from a source", () => {
  // 0 -> 1 -> 2, and 0 -> 2 directly: 2 must still sit after 1.
  const graph = graphOf([1, 2, 3], [[0, 1], [1, 2], [0, 2]]);
  assert.deepEqual(assignLayers(graph), [0, 1, 2]);
});

test("every edge points to a later column", () => {
  for (const name of ["json_two_channel.expected", "json_viewer.expected"]) {
    const graph = golden(name);
    const layout = layoutGraph(graph);
    for (const edge of graph.edges) {
      assert.ok(layout.boxes[edge.from].x < layout.boxes[edge.to].x, `${name}: ${edge.from}->${edge.to}`);
    }
  }
});

test("boxes in a column never overlap", () => {
  const layout = layoutGraph(golden("json_two_channel.expected"));
  for (const column of layout.layers) {
    for (let i = 1; i < column.length; i++) {
      const above = layout.boxes[column[i - 1]];
      const below = layout.boxes[column[i]];
      assert.ok(above.y + NODE_HEIGHT <= below.y);
      assert.equal(above.x, below.x);
    }
  }
  for (const box of layout.boxes) {
    assert.ok(box.x + NODE_WIDTH <= layout.width && box.y + NODE_HEIGHT <= layout.height);
  }
});

test("barycenter sweeps untangle crossed edges", () => {
  // Sources in order 0, 1 feed targets written in the opposite order: 0 -> 3, 1 -> 2 with 2
  // on an earlier line than 3. Source order alone draws an X; the sweep must uncross it.
  const graph = graphOf([1, 2, 3, 4], [[0, 3], [1, 2]]);
  assert.equal(countCrossings(graph, [[0, 1], [2, 3]]), 1);
  const layout = layoutGraph(graph);
  assert.equal(countCrossings(graph, layout.layers), 0);
});

test("the layout is deterministic and keeps removed nodes in place", () => {
  const graph = golden("json_viewer.expected");
  const first = layoutGraph(graph);
  const second = layoutGraph(graph);
  assert.deepEqual(first, second);
  assert.ok(first.boxes[3] !== undefined, "the node dce removed still has a box");
});
