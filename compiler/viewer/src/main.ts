// Page wiring: load a sigflowc --emit=json file (file picker, drag and drop, or ?src=URL), then
// draw it: the DAG for the selected pass, the buffer-slot timeline, and the schedule table.

import { layoutGraph, Layout } from "./layout.js";
import { IrGraph, liveAfter, parseGraph } from "./model.js";
import { describe, formatBytes, renderGraph, renderTimeline, slotColor } from "./render.js";

function element<T extends Element>(id: string): T {
  const found = document.getElementById(id);
  if (found === null) {
    throw new Error(`index.html is missing #${id}`);
  }
  return found as unknown as T;
}

const status = element<HTMLParagraphElement>("status");
const viewer = element<HTMLElement>("viewer");
const stages = element<HTMLElement>("stages");
const showRemoved = element<HTMLInputElement>("show-removed");
const graphSvg = element<SVGSVGElement>("graph");
const timelineSvg = element<SVGSVGElement>("timeline");
const details = element<HTMLPreElement>("details");
const scheduleBody = element<HTMLTableSectionElement>("schedule-body");

interface State {
  graph: IrGraph;
  layout: Layout;
  stage: number;
  selected?: number;
}
let state: State | undefined;

function draw(): void {
  if (!state) {
    return;
  }
  const { graph, layout, stage, selected } = state;
  const select = (id: number) => {
    if (state) {
      state.selected = id;
      draw();
    }
  };
  renderGraph(graphSvg, graph, layout, {
    visible: liveAfter(graph, stage),
    showRemoved: showRemoved.checked,
    selected,
    onSelect: select,
  });
  renderTimeline(timelineSvg, graph, selected);
  details.textContent = selected === undefined
    ? "Select a node to see its details."
    : describe(graph.nodes[selected]);

  for (const row of scheduleBody.rows) {
    row.classList.toggle("selected", Number(row.dataset.id) === selected);
  }
}

function buildStageControls(graph: IrGraph): void {
  stages.replaceChildren();
  graph.passes.forEach((pass, index) => {
    const label = document.createElement("label");
    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "stage";
    radio.checked = index === graph.passes.length - 1;
    radio.addEventListener("change", () => {
      if (state) {
        state.stage = index;
        draw();
      }
    });
    label.append(radio, index === 0 ? " as built" : ` after ${pass}`);
    stages.append(label);
  });
}

function buildScheduleTable(graph: IrGraph): void {
  scheduleBody.replaceChildren();
  graph.schedule.forEach((id, step) => {
    const node = graph.nodes[id];
    const row = scheduleBody.insertRow();
    row.dataset.id = String(id);
    const cells = [
      String(step),
      `%${id}`,
      node.op,
      node.name ?? "",
      node.liveRange ? `${node.liveRange[0]}..${node.liveRange[1]}` : "",
      node.slot !== undefined ? `b${node.slot}` : "",
      node.bytes !== undefined ? formatBytes(node.bytes) : "",
    ];
    for (const text of cells) {
      row.insertCell().textContent = text;
    }
    if (node.slot !== undefined) {
      // A swatch beside the slot name; the name itself carries the identity.
      const swatch = document.createElement("span");
      swatch.className = "swatch";
      swatch.style.background = slotColor(node.slot);
      row.cells[5].prepend(swatch);
    }
    row.addEventListener("click", () => {
      if (state) {
        state.selected = id;
        draw();
      }
    });
  });
}

function show(graph: IrGraph, source: string): void {
  const live = graph.nodes.filter((node) => node.live).length;
  const sized = graph.nodes.filter((node) => node.live && node.slot !== undefined);
  const known = sized.every((node) => node.bytes !== undefined);
  const total = known ? sized.reduce((sum, node) => sum + (node.bytes ?? 0), 0) : undefined;
  status.textContent = `${source}: ${graph.nodes.length} nodes (${live} live), ` +
    `${sized.length} values in ${graph.slots} buffer slots` +
    (total !== undefined ? `, ${formatBytes(total)} of values` : "") +
    `; passes: ${graph.passes.join(" → ")}`;
  status.classList.remove("error");

  state = { graph, layout: layoutGraph(graph), stage: graph.passes.length - 1 };
  buildStageControls(graph);
  buildScheduleTable(graph);
  viewer.hidden = false;
  draw();
}

function fail(source: string, message: string): void {
  status.textContent = `${source}: ${message}`;
  status.classList.add("error");
  viewer.hidden = true;
  state = undefined;
}

function load(text: string, source: string): void {
  const result = parseGraph(text);
  if (result.ok) {
    show(result.graph, source);
  } else {
    fail(source, result.error);
  }
}

async function loadFile(file: File): Promise<void> {
  load(await file.text(), file.name);
}

showRemoved.addEventListener("change", draw);

element<HTMLInputElement>("file").addEventListener("change", (event) => {
  const file = (event.target as HTMLInputElement).files?.[0];
  if (file) {
    void loadFile(file);
  }
});

document.addEventListener("dragover", (event) => event.preventDefault());
document.addEventListener("drop", (event) => {
  event.preventDefault();
  const file = event.dataTransfer?.files[0];
  if (file) {
    void loadFile(file);
  }
});

const src = new URLSearchParams(window.location.search).get("src");
if (src) {
  fetch(src)
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return response.text();
    })
    .then((text) => load(text, src))
    .catch((error: Error) => fail(src, error.message));
}
