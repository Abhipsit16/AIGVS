import argparse
import hashlib
import json
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import numpy as np


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_DIMENSION = 64
DEFAULT_STORAGE_PATH = "./aigvs_ui_data"


def embed_text(text: str, embedding_dim: int) -> np.ndarray:
    """Create a deterministic local embedding for interface demos."""
    vector = np.zeros(embedding_dim, dtype=np.float32)
    tokens = [token.strip().lower() for token in text.split() if token.strip()]
    if not tokens:
        tokens = [text.strip().lower() or "empty"]

    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
        index = int.from_bytes(digest[:4], "little") % embedding_dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        weight = 1.0 + (len(token) % 5) * 0.15
        vector[index] += sign * weight

    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


class InterfaceRuntime:
    def __init__(
        self,
        embedding_dim: int = DEFAULT_DIMENSION,
        storage_path: str = DEFAULT_STORAGE_PATH,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.storage_path = storage_path
        self.pipeline = None
        self.backend_error: Optional[str] = None
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        try:
            from src.aigvs.pipeline import AIGVSPipeline

            self.pipeline = AIGVSPipeline(
                embedding_dim=self.embedding_dim,
                storage_path=self.storage_path,
            )
            self.pipeline.load_index()
            self.backend_error = None
        except Exception as exc:  # pragma: no cover - depends on local extension build
            self.pipeline = None
            self.backend_error = str(exc)

    @property
    def ready(self) -> bool:
        return self.pipeline is not None

    def health(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "embedding_dim": self.embedding_dim,
            "storage_path": self.storage_path,
            "backend_error": self.backend_error,
        }

    def stats(self) -> Dict[str, Any]:
        if not self.pipeline:
            return self.health()
        return {
            **self.pipeline.stats(),
            "storage_path": self.storage_path,
            "topology": self._topology(),
        }

    def ingest(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.pipeline:
            raise RuntimeError(self.backend_error or "AIGVS backend is unavailable.")

        from src.aigvs.core.models import TextChunk

        text = str(payload.get("text", "")).strip()
        if not text:
            raise ValueError("Text is required.")

        chunk_id = str(payload.get("chunk_id", "")).strip()
        if not chunk_id:
            chunk_id = f"chunk_{int(time.time() * 1000)}"

        metadata = payload.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a JSON object.")

        chunk = TextChunk(
            chunk_id=chunk_id,
            text=text,
            embedding=embed_text(text, self.embedding_dim),
            metadata=metadata,
        )
        self.pipeline.insert_chunk(chunk)
        self.pipeline.save_index()
        return {
            "inserted": True,
            "chunk": {
                "id": chunk_id,
                "text": text,
                "metadata": metadata,
            },
            "stats": self.stats(),
        }

    def query(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.pipeline:
            raise RuntimeError(self.backend_error or "AIGVS backend is unavailable.")

        query_text = str(payload.get("query", "")).strip()
        if not query_text:
            raise ValueError("Query text is required.")

        top_k = int(payload.get("top_k", 5))
        top_k = max(1, min(top_k, 25))

        results = self.pipeline.query(embed_text(query_text, self.embedding_dim), top_k=top_k)
        return {
            "query": query_text,
            "top_k": top_k,
            "results": results,
            "stats": self.stats(),
        }

    def delete(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.pipeline:
            raise RuntimeError(self.backend_error or "AIGVS backend is unavailable.")

        chunk_id = str(payload.get("chunk_id", "")).strip()
        if not chunk_id:
            raise ValueError("Chunk ID is required.")
        deleted = self.pipeline.delete_chunk(chunk_id)
        if deleted:
            self.pipeline.save_index()
        return {"deleted": deleted, "stats": self.stats()}

    def _topology(self) -> List[Dict[str, Any]]:
        nodes: List[Dict[str, Any]] = []
        root = getattr(self.pipeline, "root", None)
        if root is None:
            return nodes

        def visit(node: Any, depth: int) -> None:
            children = getattr(node, "children", {})
            is_branch = hasattr(node, "children")
            chunk_ids = getattr(node, "chunk_ids", [])
            active_count = getattr(node, "active_count", len(chunk_ids))
            nodes.append(
                {
                    "id": getattr(node, "node_id", "unknown"),
                    "parent": getattr(node, "parent_id", None),
                    "depth": depth,
                    "type": "branch" if is_branch else "leaf",
                    "children": len(children),
                    "chunks": int(active_count),
                }
            )
            for child in children.values():
                visit(child, depth + 1)

        visit(root, 0)
        return nodes


def create_handler(runtime: InterfaceRuntime):
    class AIGVSInterfaceHandler(BaseHTTPRequestHandler):
        server_version = "AIGVSInterface/0.1"

        def do_GET(self) -> None:
            route = urlparse(self.path).path
            if route == "/":
                self._send_html(INDEX_HTML)
            elif route == "/api/health":
                self._send_json(runtime.health())
            elif route == "/api/stats":
                self._send_json(runtime.stats())
            else:
                self._send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            route = urlparse(self.path).path
            try:
                payload = self._read_json()
                if route == "/api/ingest":
                    self._send_json(runtime.ingest(payload))
                elif route == "/api/query":
                    self._send_json(runtime.query(payload))
                elif route == "/api/delete":
                    self._send_json(runtime.delete(payload))
                else:
                    self._send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length == 0:
                return {}
            raw = self.rfile.read(length).decode("utf-8")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError("Request body must be valid JSON.") from exc
            if not isinstance(data, dict):
                raise ValueError("Request body must be a JSON object.")
            return data

        def _send_html(self, html: str, status: int = HTTPStatus.OK) -> None:
            encoded = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_json(self, data: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
            encoded = json.dumps(data, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    return AIGVSInterfaceHandler


def serve(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    embedding_dim: int = DEFAULT_DIMENSION,
    storage_path: str = DEFAULT_STORAGE_PATH,
) -> None:
    runtime = InterfaceRuntime(embedding_dim=embedding_dim, storage_path=storage_path)
    handler = create_handler(runtime)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"AIGVS interface running at http://{host}:{port}")
    if not runtime.ready:
        print(f"Backend unavailable: {runtime.backend_error}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AIGVS browser interface.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--embedding-dim", type=int, default=DEFAULT_DIMENSION)
    parser.add_argument("--storage-path", default=DEFAULT_STORAGE_PATH)
    args = parser.parse_args()
    serve(
        host=args.host,
        port=args.port,
        embedding_dim=args.embedding_dim,
        storage_path=args.storage_path,
    )


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AIGVS Interface</title>
  <style>
    :root {
      --bg: #f6f7f4;
      --panel: #ffffff;
      --ink: #17201a;
      --muted: #637067;
      --line: #d8ddd6;
      --green: #2f6f4f;
      --green-dark: #214f39;
      --blue: #245f86;
      --gold: #a06422;
      --danger: #9b3131;
      --shadow: 0 16px 40px rgba(23, 32, 26, 0.09);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background:
        linear-gradient(120deg, rgba(47, 111, 79, 0.08), transparent 38%),
        linear-gradient(260deg, rgba(36, 95, 134, 0.08), transparent 42%),
        var(--bg);
      color: var(--ink);
      font: 15px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }

    button, input, textarea {
      font: inherit;
      letter-spacing: 0;
    }

    .shell {
      width: min(1420px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 36px;
    }

    .topbar {
      display: grid;
      grid-template-columns: 1fr auto;
      align-items: end;
      gap: 16px;
      padding-bottom: 18px;
      border-bottom: 1px solid var(--line);
    }

    h1 {
      margin: 0;
      font-size: clamp(28px, 5vw, 54px);
      line-height: 0.98;
      font-weight: 760;
      max-width: 780px;
    }

    .subline {
      margin-top: 10px;
      color: var(--muted);
      max-width: 720px;
    }

    .status {
      min-width: 220px;
      padding: 12px 14px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.74);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .status strong {
      display: block;
      font-size: 13px;
      text-transform: uppercase;
      color: var(--muted);
    }

    .status span {
      display: block;
      margin-top: 2px;
      font-weight: 720;
    }

    .workspace {
      display: grid;
      grid-template-columns: minmax(320px, 430px) minmax(0, 1fr);
      gap: 18px;
      margin-top: 22px;
      align-items: start;
    }

    .panel {
      background: rgba(255, 255, 255, 0.88);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }

    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 15px 16px;
      border-bottom: 1px solid var(--line);
    }

    h2 {
      margin: 0;
      font-size: 15px;
      text-transform: uppercase;
      color: var(--muted);
    }

    .panel-body { padding: 16px; }

    .stack {
      display: grid;
      gap: 14px;
    }

    label {
      display: grid;
      gap: 7px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 680;
    }

    input, textarea {
      width: 100%;
      border: 1px solid var(--line);
      background: #fbfcfa;
      color: var(--ink);
      border-radius: 6px;
      padding: 11px 12px;
      outline: none;
    }

    textarea {
      min-height: 130px;
      resize: vertical;
    }

    input:focus, textarea:focus {
      border-color: rgba(47, 111, 79, 0.7);
      box-shadow: 0 0 0 3px rgba(47, 111, 79, 0.13);
    }

    .row {
      display: grid;
      grid-template-columns: 1fr 112px;
      gap: 10px;
      align-items: end;
    }

    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }

    button {
      border: 1px solid transparent;
      border-radius: 6px;
      background: var(--green);
      color: white;
      padding: 10px 13px;
      min-height: 42px;
      font-weight: 760;
      cursor: pointer;
    }

    button.secondary {
      background: #eef3ef;
      color: var(--green-dark);
      border-color: #cdd9d0;
    }

    button.danger {
      background: #f5e9e9;
      color: var(--danger);
      border-color: #e2c9c9;
    }

    button:disabled {
      cursor: wait;
      opacity: 0.62;
    }

    .metrics {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }

    .metric {
      min-height: 92px;
      padding: 14px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.86);
      border-radius: 8px;
    }

    .metric span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      font-weight: 760;
    }

    .metric strong {
      display: block;
      margin-top: 8px;
      font-size: 28px;
      line-height: 1;
    }

    .results {
      display: grid;
      gap: 10px;
    }

    .result {
      border: 1px solid var(--line);
      background: #fbfcfa;
      border-radius: 8px;
      padding: 14px;
    }

    .result-head {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 760;
      text-transform: uppercase;
    }

    .result p {
      margin: 10px 0 0;
      color: #29342d;
      overflow-wrap: anywhere;
    }

    .topology {
      display: grid;
      gap: 8px;
      max-height: 340px;
      overflow: auto;
      padding-right: 4px;
    }

    .node {
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 10px;
      align-items: center;
      padding: 9px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfa;
    }

    .node-id {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      font-weight: 720;
    }

    .badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 58px;
      min-height: 26px;
      border-radius: 999px;
      color: white;
      font-size: 12px;
      font-weight: 760;
      background: var(--blue);
    }

    .badge.leaf { background: var(--green); }
    .badge.branch { background: var(--gold); }

    .notice {
      margin-top: 12px;
      padding: 11px 12px;
      border-radius: 6px;
      border: 1px solid var(--line);
      background: #fbfcfa;
      color: var(--muted);
      overflow-wrap: anywhere;
    }

    .notice.error {
      color: var(--danger);
      border-color: #e2c9c9;
      background: #fff7f7;
    }

    @media (max-width: 980px) {
      .topbar, .workspace {
        grid-template-columns: 1fr;
      }

      .metrics {
        grid-template-columns: repeat(2, minmax(120px, 1fr));
      }
    }

    @media (max-width: 560px) {
      .shell {
        width: min(100vw - 20px, 1420px);
        padding-top: 16px;
      }

      .row, .metrics {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="topbar">
      <div>
        <h1>AIGVS Control Surface</h1>
        <div class="subline">Attractor-indexed ingestion, search, and topology inspection.</div>
      </div>
      <div class="status">
        <strong>Backend</strong>
        <span id="backendStatus">Checking</span>
      </div>
    </header>

    <section class="workspace">
      <div class="stack">
        <section class="panel">
          <div class="panel-header">
            <h2>Ingest</h2>
          </div>
          <div class="panel-body stack">
            <label>Chunk ID
              <input id="chunkId" type="text" autocomplete="off" placeholder="auto-generated">
            </label>
            <label>Text
              <textarea id="ingestText" placeholder="Paste a document chunk"></textarea>
            </label>
            <label>Metadata JSON
              <textarea id="metadataText" spellcheck="false">{}</textarea>
            </label>
            <div class="actions">
              <button id="ingestButton">Ingest</button>
              <button class="secondary" id="clearIngestButton">Clear</button>
            </div>
          </div>
        </section>

        <section class="panel">
          <div class="panel-header">
            <h2>Delete</h2>
          </div>
          <div class="panel-body stack">
            <label>Chunk ID
              <input id="deleteId" type="text" autocomplete="off" placeholder="chunk_...">
            </label>
            <div class="actions">
              <button class="danger" id="deleteButton">Delete</button>
            </div>
          </div>
        </section>
      </div>

      <div class="stack">
        <section class="metrics" aria-label="Index metrics">
          <div class="metric"><span>Chunks</span><strong id="metricChunks">0</strong></div>
          <div class="metric"><span>Leaves</span><strong id="metricLeaves">0</strong></div>
          <div class="metric"><span>Branches</span><strong id="metricBranches">0</strong></div>
          <div class="metric"><span>Cache</span><strong id="metricCache">0</strong></div>
        </section>

        <section class="panel">
          <div class="panel-header">
            <h2>Query</h2>
          </div>
          <div class="panel-body stack">
            <div class="row">
              <label>Search Text
                <input id="queryText" type="text" autocomplete="off" placeholder="Ask for related context">
              </label>
              <label>Top K
                <input id="topK" type="number" min="1" max="25" value="5">
              </label>
            </div>
            <div class="actions">
              <button id="queryButton">Search</button>
              <button class="secondary" id="refreshButton">Refresh</button>
            </div>
            <div id="results" class="results"></div>
          </div>
        </section>

        <section class="panel">
          <div class="panel-header">
            <h2>Topology</h2>
          </div>
          <div class="panel-body">
            <div id="topology" class="topology"></div>
          </div>
        </section>
      </div>
    </section>

    <div id="notice" class="notice"></div>
  </main>

  <script>
    const state = { busy: false };
    const $ = (id) => document.getElementById(id);

    async function api(path, options = {}) {
      const response = await fetch(path, {
        headers: { "Content-Type": "application/json" },
        ...options
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Request failed");
      }
      return data;
    }

    function setBusy(value) {
      state.busy = value;
      for (const button of document.querySelectorAll("button")) {
        button.disabled = value;
      }
    }

    function notice(message, isError = false) {
      $("notice").textContent = message || "";
      $("notice").classList.toggle("error", Boolean(isError));
    }

    function renderStats(stats = {}) {
      $("metricChunks").textContent = stats.indexed_chunks ?? 0;
      $("metricLeaves").textContent = stats.leaves ?? 0;
      $("metricBranches").textContent = stats.branches ?? 0;
      $("metricCache").textContent = stats.payload_cache_size ?? 0;
      renderTopology(stats.topology || []);
    }

    function renderTopology(nodes) {
      const target = $("topology");
      if (!nodes.length) {
        target.innerHTML = '<div class="notice">No topology available.</div>';
        return;
      }
      target.innerHTML = nodes.map((node) => {
        const indent = Math.min(node.depth || 0, 8) * 18;
        const count = node.type === "branch" ? `${node.children} child` : `${node.chunks} chunk`;
        return `
          <div class="node" style="margin-left:${indent}px">
            <div class="node-id" title="${escapeHtml(node.id)}">${escapeHtml(node.id)}</div>
            <span>${count}</span>
            <span class="badge ${node.type}">${node.type}</span>
          </div>
        `;
      }).join("");
    }

    function renderResults(results) {
      const target = $("results");
      if (!results.length) {
        target.innerHTML = '<div class="notice">No matches returned.</div>';
        return;
      }
      target.innerHTML = results.map((result, index) => `
        <article class="result">
          <div class="result-head">
            <span>${index + 1}. ${escapeHtml(result.id || "unknown")}</span>
            <span>${escapeHtml(JSON.stringify(result.metadata || {}))}</span>
          </div>
          <p>${escapeHtml(result.text || "")}</p>
        </article>
      `).join("");
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    async function refresh() {
      try {
        const health = await api("/api/health");
        $("backendStatus").textContent = health.ready ? "Ready" : "Unavailable";
        if (!health.ready) {
          notice(health.backend_error || "Backend unavailable.", true);
        }
        const stats = await api("/api/stats");
        renderStats(stats);
      } catch (error) {
        $("backendStatus").textContent = "Unavailable";
        notice(error.message, true);
      }
    }

    $("ingestButton").addEventListener("click", async () => {
      setBusy(true);
      try {
        const metadataText = $("metadataText").value.trim() || "{}";
        const payload = {
          chunk_id: $("chunkId").value.trim(),
          text: $("ingestText").value,
          metadata: JSON.parse(metadataText)
        };
        const data = await api("/api/ingest", {
          method: "POST",
          body: JSON.stringify(payload)
        });
        renderStats(data.stats);
        notice(`Inserted ${data.chunk.id}`);
      } catch (error) {
        notice(error.message, true);
      } finally {
        setBusy(false);
      }
    });

    $("queryButton").addEventListener("click", async () => {
      setBusy(true);
      try {
        const data = await api("/api/query", {
          method: "POST",
          body: JSON.stringify({
            query: $("queryText").value,
            top_k: Number($("topK").value || 5)
          })
        });
        renderResults(data.results);
        renderStats(data.stats);
        notice(`${data.results.length} result${data.results.length === 1 ? "" : "s"}`);
      } catch (error) {
        notice(error.message, true);
      } finally {
        setBusy(false);
      }
    });

    $("deleteButton").addEventListener("click", async () => {
      setBusy(true);
      try {
        const data = await api("/api/delete", {
          method: "POST",
          body: JSON.stringify({ chunk_id: $("deleteId").value })
        });
        renderStats(data.stats);
        notice(data.deleted ? "Deleted." : "Chunk not found.", !data.deleted);
      } catch (error) {
        notice(error.message, true);
      } finally {
        setBusy(false);
      }
    });

    $("clearIngestButton").addEventListener("click", () => {
      $("chunkId").value = "";
      $("ingestText").value = "";
      $("metadataText").value = "{}";
      notice("");
    });

    $("refreshButton").addEventListener("click", refresh);
    refresh();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
