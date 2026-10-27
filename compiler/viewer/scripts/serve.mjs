// Minimal static file server for the viewer: browsers will not load ES modules from file://.
// Serves compiler/ (one level up), so ?src= can point at compiler outputs, e.g.
//   npm run serve, then open http://localhost:8000/viewer/?src=/tests/golden/json_viewer.expected

import { createReadStream, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(fileURLToPath(new URL("../..", import.meta.url)));
const port = Number(process.env.PORT ?? 8000);
const types = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".map": "application/json",
  ".json": "application/json",
  ".expected": "application/json",
};

createServer((request, response) => {
  const url = new URL(request.url ?? "/", "http://localhost");
  let path = normalize(join(root, decodeURIComponent(url.pathname)));
  if (!path.startsWith(root)) {
    response.writeHead(403).end();
    return;
  }
  try {
    if (statSync(path).isDirectory()) {
      path = join(path, "index.html");
    }
    statSync(path);
  } catch {
    response.writeHead(404).end("not found");
    return;
  }
  response.writeHead(200, { "content-type": types[extname(path)] ?? "application/octet-stream" });
  createReadStream(path).pipe(response);
}).listen(port, "127.0.0.1", () => {
  console.log(`serving ${root} at http://localhost:${port}/viewer/`);
});
