// Page wiring: load a sigflowc --emit=json file (file picker, drag and drop, or ?src=URL) and
// hand the parsed graph to the view.

import { IrGraph, parseGraph } from "./model.js";

function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (found === null) {
    throw new Error(`index.html is missing #${id}`);
  }
  return found as T;
}

const status = element<HTMLParagraphElement>("status");
const summary = element<HTMLElement>("summary");

function show(graph: IrGraph, source: string): void {
  const live = graph.nodes.filter((node) => node.live).length;
  status.textContent = `${source}: ${graph.nodes.length} nodes (${live} live), ` +
    `${graph.edges.length} edges, ${graph.slots} buffer slots; passes: ${graph.passes.join(" → ")}`;
  status.classList.remove("error");

  summary.replaceChildren();
  for (const id of graph.schedule) {
    const node = graph.nodes[id];
    const item = document.createElement("li");
    item.textContent = `%${node.id} ${node.op}${node.name ? ` (${node.name})` : ""}`;
    summary.append(item);
  }
}

function load(text: string, source: string): void {
  const result = parseGraph(text);
  if (!result.ok) {
    status.textContent = `${source}: ${result.error}`;
    status.classList.add("error");
    summary.replaceChildren();
    return;
  }
  show(result.graph, source);
}

async function loadFile(file: File): Promise<void> {
  load(await file.text(), file.name);
}

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
    .catch((error: Error) => {
      status.textContent = `${src}: ${error.message}`;
      status.classList.add("error");
    });
}
