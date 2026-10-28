// SVG drawing: the DAG for one stage of compilation, and the buffer-slot timeline.
//
// Buffer slots are an identity, so each gets a fixed categorical color (slot 0 is always the
// first color, never reassigned by rank) from the CSS variables --slot-0 ... --slot-7 in
// index.html. Slots past 8 share a neutral --slot-other. Color is never the only cue: every node
// and timeline bar also prints its slot as text ("b2").

import { NODE_HEIGHT, NODE_WIDTH, Layout } from "./layout.js";
import { IrGraph, IrNode } from "./model.js";

const SVG = "http://www.w3.org/2000/svg";
const PALETTE_SIZE = 8;

function svg<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attributes: Record<string, string | number> = {},
  text?: string,
): SVGElementTagNameMap[K] {
  const element = document.createElementNS(SVG, tag);
  for (const [name, value] of Object.entries(attributes)) {
    element.setAttribute(name, String(value));
  }
  if (text !== undefined) {
    element.textContent = text;
  }
  return element;
}

export function slotColor(slot: number): string {
  return slot < PALETTE_SIZE ? `var(--slot-${slot})` : "var(--slot-other)";
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  const kib = bytes / 1024;
  return kib < 1024 ? `${kib.toFixed(1)} KiB` : `${(kib / 1024).toFixed(2)} MiB`;
}

function clip(text: string, max: number): string {
  return text.length <= max ? text : `…${text.slice(text.length - max + 1)}`;
}

// The second line of a node: what it computes on, in a few words.
function detail(node: IrNode): string {
  switch (node.op) {
    case "capture":
      return clip(node.path ?? "", 24);
    case "fir":
      return `taps ${clip(node.taps ?? "", 19)}`;
    case "fft":
      return `n = ${node.n}`;
    case "emit":
      return `→ ${clip(node.path ?? "", 22)}`;
    default:
      return "";
  }
}

export function describe(node: IrNode): string {
  const lines = [`%${node.id} ${node.op}${node.name ? ` (${node.name})` : ""}`];
  lines.push(`line ${node.line}, column ${node.column}`);
  if (node.type) {
    lines.push(`${node.type} at ${node.rate} samples/s`);
  }
  if (node.path) {
    lines.push(node.op === "emit" ? `writes ${node.path}` : `reads ${node.path}`);
  }
  if (node.taps) {
    lines.push(`taps ${node.taps}`);
  }
  if (!node.live) {
    lines.push(`removed by ${node.removedBy}`);
  }
  if (node.step !== undefined) {
    lines.push(`schedule step ${node.step}`);
  }
  if (node.group !== undefined) {
    lines.push(`fused: runs in one launch with the rest of group ${node.group}`);
  }
  if (node.liveRange && node.slot !== undefined) {
    lines.push(`live steps ${node.liveRange[0]}..${node.liveRange[1]}, buffer b${node.slot}`);
  }
  if (node.bytes !== undefined) {
    lines.push(formatBytes(node.bytes));
  }
  return lines.join("\n");
}

export interface GraphView {
  visible: Set<number>; // nodes alive in the stage shown
  showRemoved: boolean; // draw removed nodes as faded outlines instead of hiding them
  selected?: number;
  onSelect: (id: number) => void;
}

export function renderGraph(target: SVGSVGElement, graph: IrGraph, layout: Layout, view: GraphView) {
  target.replaceChildren();
  target.setAttribute("viewBox", `0 0 ${layout.width} ${layout.height}`);
  target.setAttribute("width", String(layout.width));
  target.setAttribute("height", String(layout.height));

  const shown = (id: number) => view.visible.has(id) || view.showRemoved;
  const inputCount = graph.nodes.map(() => 0);
  for (const edge of graph.edges) {
    inputCount[edge.to]++;
  }

  const edges = svg("g", { class: "edges" });
  for (const edge of graph.edges) {
    if (!shown(edge.from) || !shown(edge.to)) {
      continue;
    }
    const from = layout.boxes[edge.from];
    const to = layout.boxes[edge.to];
    const x1 = from.x + NODE_WIDTH;
    const y1 = from.y + NODE_HEIGHT / 2;
    const x2 = to.x;
    // Inputs enter top to bottom in argument order: xcorr's a above its b.
    const y2 = to.y + (NODE_HEIGHT * (edge.input + 1)) / (inputCount[edge.to] + 1);
    const bend = Math.max(24, (x2 - x1) / 2);
    const live = view.visible.has(edge.from) && view.visible.has(edge.to);
    edges.append(svg("path", {
      d: `M ${x1} ${y1} C ${x1 + bend} ${y1}, ${x2 - bend} ${y2}, ${x2} ${y2}`,
      class: live ? "edge" : "edge removed",
      "marker-end": "url(#arrow)",
    }));
  }

  const defs = svg("defs");
  const marker = svg("marker", {
    id: "arrow", viewBox: "0 0 8 8", refX: 8, refY: 4, markerWidth: 8, markerHeight: 8,
    orient: "auto-start-reverse",
  });
  marker.append(svg("path", { d: "M 0 0 L 8 4 L 0 8 z", class: "arrow" }));
  defs.append(marker);
  target.append(defs, edges);

  const nodes = svg("g", { class: "nodes" });
  for (const node of graph.nodes) {
    if (!shown(node.id)) {
      continue;
    }
    const box = layout.boxes[node.id];
    const classes = ["node", `op-${node.op}`];
    if (!view.visible.has(node.id)) {
      classes.push("removed");
    }
    if (view.selected === node.id) {
      classes.push("selected");
    }
    if (node.group !== undefined) {
      classes.push("fused");
    }
    const group = svg("g", {
      class: classes.join(" "),
      transform: `translate(${box.x} ${box.y})`,
      tabindex: 0,
      role: "button",
      "aria-label": describe(node).replace(/\n/g, "; "),
    });
    group.append(svg("title", {}, describe(node)));
    group.append(svg("rect", { class: "box", width: NODE_WIDTH, height: NODE_HEIGHT, rx: 6 }));
    if (node.slot !== undefined && view.visible.has(node.id)) {
      // Slot color as a bar along the left edge, inset so it keeps the box's rounded corner.
      group.append(svg("rect", {
        class: "slot-bar", x: 3, y: 6, width: 4, height: NODE_HEIGHT - 12, rx: 2,
        fill: slotColor(node.slot),
      }));
    }
    group.append(svg("text", { x: 14, y: 19, class: "title" },
      `%${node.id} ${node.op}${node.name ? `  ${node.name}` : ""}` +
        (node.group !== undefined ? `  [g${node.group}]` : "")));
    group.append(svg("text", { x: 14, y: 35, class: "detail" }, detail(node)));
    // Schedule and buffers describe the final graph, so a node a later pass removes has none
    // even in the stages where it is still shown as alive.
    const footer = !view.visible.has(node.id)
      ? `removed by ${node.removedBy}`
      : !node.live
        ? `${node.removedBy} removes it later`
        : node.slot !== undefined
        ? `b${node.slot} · step ${node.step}${node.bytes !== undefined ? ` · ${formatBytes(node.bytes)}` : ""}`
        : `step ${node.step}`;
    group.append(svg("text", { x: 14, y: 50, class: "detail" }, footer));
    group.addEventListener("click", () => view.onSelect(node.id));
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        view.onSelect(node.id);
      }
    });
    nodes.append(group);
  }
  target.append(nodes);
}

// One lane per buffer slot, one bar per value from the step that writes it to the last step
// that reads it, so reuse reads directly: a lane with several bars is one buffer used several
// times.
export function renderTimeline(target: SVGSVGElement, graph: IrGraph, selected?: number) {
  const LANE = 26;
  const STEP = 56;
  const LEFT = 44;
  const TOP = 22;
  const steps = graph.schedule.length;
  const width = LEFT + steps * STEP + 8;
  const height = TOP + graph.slots * LANE + 8;
  target.replaceChildren();
  target.setAttribute("viewBox", `0 0 ${width} ${height}`);
  target.setAttribute("width", String(width));
  target.setAttribute("height", String(height));

  for (let step = 0; step < steps; step++) {
    const x = LEFT + step * STEP;
    target.append(svg("line", { x1: x, x2: x, y1: TOP - 4, y2: height - 8, class: "grid" }));
    target.append(svg("text", { x: x + STEP / 2, y: 14, class: "axis", "text-anchor": "middle" },
      String(step)));
  }
  for (let slot = 0; slot < graph.slots; slot++) {
    target.append(svg("text", { x: LEFT - 8, y: TOP + slot * LANE + 23, class: "axis",
      "text-anchor": "end" }, `b${slot}`));
  }

  for (const node of graph.nodes) {
    if (!node.live || node.liveRange === undefined || node.slot === undefined) {
      continue;
    }
    const [def, last] = node.liveRange;
    // 2px gaps between bars so neighbours in a lane stay separate marks.
    // A thin bar along the bottom of the lane, its label above it in text ink (never in the
    // slot color, which carries identity only).
    const x = LEFT + def * STEP + 1;
    const barWidth = (last - def + 1) * STEP - 2;
    const lane = TOP + node.slot * LANE;
    const group = svg("g", { class: selected === node.id ? "bar selected" : "bar" });
    group.append(svg("title", {}, describe(node)));
    // Invisible hit target the size of the lane, so the thin bar is easy to hover.
    group.append(svg("rect", { x, y: lane, width: barWidth, height: LANE, class: "hit" }));
    group.append(svg("rect", {
      x, y: lane + 16, width: barWidth, height: 7, rx: 3.5, fill: slotColor(node.slot),
    }));
    group.append(svg("text", { x: x + 2, y: lane + 12, class: "bar-label" },
      `%${node.id}${node.name ? ` ${node.name}` : ""}`));
    target.append(group);
  }
}
