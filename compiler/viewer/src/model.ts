// The graph IR as sigflowc --emit=json writes it (compiler/include/sigflow/json.hpp), and a
// parser that checks the shape before anything is drawn, so a stale or hand-edited file fails
// with a message instead of a broken picture.

export type Op = "capture" | "fir" | "fft" | "magnitude" | "xcorr" | "emit";

export interface IrNode {
  id: number;
  op: Op;
  line: number;
  column: number;
  live: boolean;
  type?: string; // every op but emit
  rate?: string;
  name?: string;
  path?: string; // capture: file read; emit: file written
  taps?: string; // fir
  n?: number; // fft
  removedBy?: string; // the pass that removed the node, when !live
  step?: number; // schedule step, live nodes only
  liveRange?: [number, number]; // [def, last use], live value nodes only
  slot?: number;
  group?: number; // kernel-fusion group: nodes that run as one launch
  bytes?: number; // only when sigflowc could size the captures (--input-dir)
}

export interface IrEdge {
  from: number;
  to: number;
  input: number;
}

export interface IrGraph {
  passes: string[];
  nodes: IrNode[]; // indexed by id
  edges: IrEdge[];
  schedule: number[];
  slots: number;
}

export type ParseResult = { ok: true; graph: IrGraph } | { ok: false; error: string };

const OPS: readonly string[] = ["capture", "fir", "fft", "magnitude", "xcorr", "emit"];

type Json = Record<string, unknown>;

function isObject(value: unknown): value is Json {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isInt(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value);
}

function optionalString(node: Json, key: string): string | undefined {
  const value = node[key];
  if (value === undefined) {
    return undefined;
  }
  if (typeof value !== "string") {
    throw new Error(`node ${String(node.id)}: "${key}" must be a string`);
  }
  return value;
}

function optionalInt(node: Json, key: string): number | undefined {
  const value = node[key];
  if (value === undefined) {
    return undefined;
  }
  if (!isInt(value)) {
    throw new Error(`node ${String(node.id)}: "${key}" must be an integer`);
  }
  return value;
}

function parseNode(raw: unknown, index: number): IrNode {
  if (!isObject(raw)) {
    throw new Error(`nodes[${index}] is not an object`);
  }
  if (raw.id !== index) {
    throw new Error(`nodes[${index}] has id ${String(raw.id)}; ids must equal their index`);
  }
  if (typeof raw.op !== "string" || !OPS.includes(raw.op)) {
    throw new Error(`node ${index}: unknown op ${JSON.stringify(raw.op)}`);
  }
  if (!isInt(raw.line) || !isInt(raw.column) || typeof raw.live !== "boolean") {
    throw new Error(`node ${index}: "line", "column", and "live" are required`);
  }
  const node: IrNode = {
    id: index,
    op: raw.op as Op,
    line: raw.line,
    column: raw.column,
    live: raw.live,
    type: optionalString(raw, "type"),
    rate: optionalString(raw, "rate"),
    name: optionalString(raw, "name"),
    path: optionalString(raw, "path"),
    taps: optionalString(raw, "taps"),
    n: optionalInt(raw, "n"),
    removedBy: optionalString(raw, "removed_by"),
    step: optionalInt(raw, "step"),
    slot: optionalInt(raw, "slot"),
    group: optionalInt(raw, "group"),
    bytes: optionalInt(raw, "bytes"),
  };
  const range = raw.live_range;
  if (range !== undefined) {
    if (!Array.isArray(range) || range.length !== 2 || !isInt(range[0]) || !isInt(range[1])) {
      throw new Error(`node ${index}: "live_range" must be [def, last use]`);
    }
    node.liveRange = [range[0], range[1]];
  }
  return node;
}

function parseEdge(raw: unknown, index: number, nodeCount: number): IrEdge {
  if (!isObject(raw) || !isInt(raw.from) || !isInt(raw.to) || !isInt(raw.input)) {
    throw new Error(`edges[${index}] must have integer "from", "to", and "input"`);
  }
  for (const end of [raw.from, raw.to]) {
    if (end < 0 || end >= nodeCount) {
      throw new Error(`edges[${index}] refers to node ${end}, which does not exist`);
    }
  }
  return { from: raw.from, to: raw.to, input: raw.input };
}

// Parses the text of a sigflowc --emit=json file.
export function parseGraph(text: string): ParseResult {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch (e) {
    return { ok: false, error: `not JSON: ${(e as Error).message}` };
  }
  try {
    if (!isObject(raw) || !Array.isArray(raw.nodes) || !Array.isArray(raw.edges)) {
      throw new Error('expected an object with "nodes" and "edges" (sigflowc --emit=json)');
    }
    if (!Array.isArray(raw.passes) || !Array.isArray(raw.schedule) || !isInt(raw.slots)) {
      throw new Error('"passes", "schedule", and "slots" are missing; rebuild sigflowc');
    }
    const nodes = raw.nodes.map(parseNode);
    const edges = raw.edges.map((edge, i) => parseEdge(edge, i, nodes.length));
    const schedule = raw.schedule.map((id) => {
      if (!isInt(id) || id < 0 || id >= nodes.length) {
        throw new Error(`schedule refers to node ${String(id)}, which does not exist`);
      }
      return id;
    });
    const passes = raw.passes.map((pass) => String(pass));
    return { ok: true, graph: { passes, nodes, edges, schedule, slots: raw.slots } };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
}

// What the viewer shows for one stage of compilation: the nodes alive after `passes[stage]`.
// Stage 0 ("build") is the graph as built, before any pass removed anything.
export function liveAfter(graph: IrGraph, stage: number): Set<number> {
  const removedBy = new Set(graph.passes.slice(1, stage + 1));
  const live = new Set<number>();
  for (const node of graph.nodes) {
    if (node.live || node.removedBy === undefined || !removedBy.has(node.removedBy)) {
      live.add(node.id);
    }
  }
  return live;
}
