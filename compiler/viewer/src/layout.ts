// Layered graph layout for the IR DAG, left to right (a simplified Sugiyama layout):
//
//   1. Layers: a node's layer is the length of the longest path from a source to it, so every
//      edge points to a later layer.
//   2. Order within a layer: start in source order (line, column), then sweep left-to-right and
//      right-to-left a few times, moving each node to the average position (barycenter) of its
//      neighbours in the previous layer, which removes most edge crossings.
//   3. Coordinates: layers become columns, positions become rows, each column centred.
//
// The layout is computed once on every node the builder made, including ones a pass removed,
// so nodes keep their place when the viewer switches between passes. Everything is
// deterministic: ties always fall back to source order.

import { IrGraph, IrNode } from "./model.js";

export interface NodeBox {
  id: number;
  layer: number;
  row: number; // position within the layer, 0 at the top
  x: number; // top-left corner
  y: number;
}

export interface Layout {
  boxes: NodeBox[]; // indexed by node id
  layers: number[][]; // node ids per layer, top to bottom
  width: number;
  height: number;
}

export const NODE_WIDTH = 168;
export const NODE_HEIGHT = 58;
export const GAP_X = 64;
export const GAP_Y = 22;
export const MARGIN = 16;
const SWEEPS = 4;

function sourceOrder(a: IrNode, b: IrNode): number {
  return a.line - b.line || a.column - b.column || a.id - b.id;
}

export function assignLayers(graph: IrGraph): number[] {
  // Inputs always have smaller ids than their users (sigflow/ir.hpp), so one pass in id order
  // sees every input's layer before it is needed.
  const inputs: number[][] = graph.nodes.map(() => []);
  for (const edge of graph.edges) {
    inputs[edge.to].push(edge.from);
  }
  const layer: number[] = [];
  for (const node of graph.nodes) {
    layer[node.id] = inputs[node.id].reduce((max, from) => Math.max(max, layer[from] + 1), 0);
  }
  return layer;
}

// Average row of `id`'s neighbours (in `neighbours`) in the layer whose rows are `rowOf`, or
// undefined when it has none there.
function barycenter(id: number, neighbours: number[][], rowOf: Map<number, number>) {
  const rows = neighbours[id].filter((n) => rowOf.has(n)).map((n) => rowOf.get(n) as number);
  return rows.length === 0 ? undefined : rows.reduce((a, b) => a + b, 0) / rows.length;
}

function reorder(layer: number[], neighbours: number[][], fixed: number[]): number[] {
  const rowOf = new Map(fixed.map((id, row) => [id, row]));
  const current = new Map(layer.map((id, row) => [id, row]));
  // A node with no neighbours in the fixed layer keeps its current row as its key.
  const row = (id: number) => current.get(id) as number;
  const key = (id: number) => barycenter(id, neighbours, rowOf) ?? row(id);
  return [...layer].sort((a, b) => key(a) - key(b) || row(a) - row(b));
}

// Number of edge crossings between consecutive layers: the measure the sweeps reduce.
export function countCrossings(graph: IrGraph, layers: number[][]): number {
  const rowOf = new Map<number, number>();
  const layerOf = new Map<number, number>();
  layers.forEach((ids, l) => ids.forEach((id, row) => {
    rowOf.set(id, row);
    layerOf.set(id, l);
  }));
  // Edges spanning several layers are compared by their end rows, as straight segments.
  const edges = graph.edges.map((e) => [rowOf.get(e.from) as number, rowOf.get(e.to) as number,
    layerOf.get(e.from) as number, layerOf.get(e.to) as number]);
  let crossings = 0;
  for (let i = 0; i < edges.length; i++) {
    for (let j = i + 1; j < edges.length; j++) {
      const [a1, a2, la1, la2] = edges[i];
      const [b1, b2, lb1, lb2] = edges[j];
      if (la1 !== lb1 || la2 !== lb2) {
        continue;
      }
      if ((a1 - b1) * (a2 - b2) < 0) {
        crossings++;
      }
    }
  }
  return crossings;
}

export function layoutGraph(graph: IrGraph): Layout {
  const layerOf = assignLayers(graph);
  const layerCount = layerOf.reduce((max, l) => Math.max(max, l + 1), 0);
  const layers: number[][] = Array.from({ length: layerCount }, () => []);
  for (const node of [...graph.nodes].sort(sourceOrder)) {
    layers[layerOf[node.id]].push(node.id);
  }

  const inputs: number[][] = graph.nodes.map(() => []);
  const users: number[][] = graph.nodes.map(() => []);
  for (const edge of graph.edges) {
    inputs[edge.to].push(edge.from);
    users[edge.from].push(edge.to);
  }

  // Keep the best ordering seen: a sweep can occasionally make things worse.
  let best = layers.map((ids) => [...ids]);
  let bestCrossings = countCrossings(graph, best);
  for (let sweep = 0; sweep < SWEEPS && bestCrossings > 0; sweep++) {
    for (let l = 1; l < layerCount; l++) {
      layers[l] = reorder(layers[l], inputs, layers[l - 1]);
    }
    for (let l = layerCount - 2; l >= 0; l--) {
      layers[l] = reorder(layers[l], users, layers[l + 1]);
    }
    const crossings = countCrossings(graph, layers);
    if (crossings < bestCrossings) {
      best = layers.map((ids) => [...ids]);
      bestCrossings = crossings;
    }
  }

  const tallest = best.reduce((max, ids) => Math.max(max, ids.length), 0);
  const height = 2 * MARGIN + tallest * NODE_HEIGHT + Math.max(0, tallest - 1) * GAP_Y;
  const width = 2 * MARGIN + layerCount * NODE_WIDTH + Math.max(0, layerCount - 1) * GAP_X;
  const boxes: NodeBox[] = [];
  best.forEach((ids, layer) => {
    const columnHeight = ids.length * NODE_HEIGHT + (ids.length - 1) * GAP_Y;
    const top = (height - columnHeight) / 2;
    ids.forEach((id, row) => {
      boxes[id] = {
        id,
        layer,
        row,
        x: MARGIN + layer * (NODE_WIDTH + GAP_X),
        y: top + row * (NODE_HEIGHT + GAP_Y),
      };
    });
  });
  return { boxes, layers: best, width, height };
}
