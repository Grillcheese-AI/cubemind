"""Interactive graph explorer for the CubeMind pipeline.

Renders the full CubeMind architecture as an interactive web-based
node graph. Click nodes to expand into subcomponents, see live metrics,
and trace data flow through the pipeline.

Uses a lightweight HTML/JS approach — no heavy framework dependencies.
Generates a self-contained HTML file with D3.js for the graph layout.

Usage:
    from cubemind.telemetry.graph_explorer import GraphExplorer

    explorer = GraphExplorer(cubemind_model)
    explorer.export_html("pipeline.html")  # static snapshot
    explorer.serve(port=8765)              # live interactive server
"""

from __future__ import annotations

import json
import webbrowser
from pathlib import Path

from cubemind.telemetry.collector import MetricsCollector


class PipelineNode:
    """A node in the CubeMind pipeline graph."""

    __slots__ = ("id", "label", "stage", "metrics", "children", "metadata")

    def __init__(self, id: str, label: str, stage: str,
                 metrics: list[str] | None = None,
                 children: list["PipelineNode"] | None = None,
                 metadata: dict | None = None):
        self.id = id
        self.label = label
        self.stage = stage
        self.metrics = metrics or []
        self.children = children or []
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "stage": self.stage,
            "metrics": self.metrics,
            "children": [c.to_dict() for c in self.children],
            "metadata": self.metadata,
        }


class PipelineEdge:
    """A directed edge between pipeline nodes."""

    __slots__ = ("source", "target", "label", "data_type")

    def __init__(self, source: str, target: str, label: str = "", data_type: str = "block_code"):
        self.source = source
        self.target = target
        self.label = label
        self.data_type = data_type

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "label": self.label,
            "data_type": self.data_type,
        }


def build_default_graph() -> tuple[list[PipelineNode], list[PipelineEdge]]:
    """Build the default CubeMind pipeline graph.

    Returns the full architecture as nodes and edges.
    """
    nodes = [
        # Top-level pipeline stages
        PipelineNode("input", "Input", "input", metadata={"desc": "Text or vector input"}),

        PipelineNode("perception", "Perception", "perception",
                     metrics=["perception.latency_ms", "perception.batch_size"],
                     children=[
                         PipelineNode("encoder", "BatchVSAEncoder", "perception",
                                      metrics=["perception.encode_ms"]),
                         PipelineNode("discretize", "Discretize", "perception",
                                      metadata={"desc": "Continuous → one-hot block code"}),
                     ]),

        PipelineNode("routing", "Routing", "routing",
                     metrics=["routing.top_score", "routing.latency_ms"],
                     children=[
                         PipelineNode("prototype_sim", "Prototype Similarity", "routing",
                                      metrics=["routing.similarity_batch_ms"]),
                         PipelineNode("moe_gate", "DSelectK Gate", "routing",
                                      metrics=["routing.gate_entropy"],
                                      metadata={"desc": "Sparse k-hot expert selection"}),
                     ]),

        PipelineNode("memory", "Memory", "memory",
                     metrics=["memory.surprise", "memory.stress", "memory.cache_size"],
                     children=[
                         PipelineNode("cache", "VSACache", "memory",
                                      metrics=["memory.hit_rate", "memory.eviction_count"]),
                         PipelineNode("hippocampal", "Hippocampal DG/CA3", "memory",
                                      metrics=["memory.episode_count"]),
                     ]),

        PipelineNode("detection", "Detection", "detection",
                     metrics=["detection.log_likelihood", "detection.latency_ms"],
                     children=[
                         PipelineNode("hmm_forward", "HMM Forward", "detection",
                                      metrics=["detection.forward_ms"]),
                         PipelineNode("hmm_ensemble", "HMM Ensemble", "detection",
                                      metrics=["detection.ensemble_diversity"],
                                      metadata={"desc": "M independent HMM rules weighted by likelihood"}),
                     ]),

        PipelineNode("execution", "Execution", "execution",
                     metrics=["execution.latency_ms"],
                     children=[
                         PipelineNode("hyla", "HYLA Hypernetwork", "execution",
                                      metrics=["execution.weight_sparsity", "execution.weight_norm"],
                                      metadata={"desc": "Block-code conditioned weight generation"}),
                         PipelineNode("cvl", "Contrastive Value", "execution",
                                      metrics=["execution.q_value"],
                                      metadata={"desc": "InfoNCE Q-value estimation"}),
                     ]),

        PipelineNode("answer", "Answer", "output",
                     metrics=["answer.confidence", "answer.latency_ms"],
                     children=[
                         PipelineNode("decoder", "Codebook Decoder", "output"),
                     ]),

        # Training subsystem
        PipelineNode("training", "Training", "training",
                     metrics=["training.loss", "training.effective_lr"],
                     children=[
                         PipelineNode("surprise_optim", "Surprise Momentum", "training",
                                      metrics=["training.surprise", "training.momentum"]),
                         PipelineNode("disarm", "DisARM Gradients", "training",
                                      metrics=["training.grad_variance"]),
                         PipelineNode("losses", "CIW + DROPS", "training"),
                     ]),
    ]

    edges = [
        PipelineEdge("input", "perception", "text", "text"),
        PipelineEdge("perception", "routing", "block_code (k×l)", "block_code"),
        PipelineEdge("perception", "memory", "block_code", "block_code"),
        PipelineEdge("routing", "detection", "expert indices", "indices"),
        PipelineEdge("memory", "detection", "surprise + context", "signal"),
        PipelineEdge("detection", "execution", "rule weights", "weights"),
        PipelineEdge("execution", "answer", "output block_code", "block_code"),
        PipelineEdge("answer", "memory", "store result", "block_code"),
        PipelineEdge("execution", "training", "gradients", "gradient"),
        PipelineEdge("memory", "training", "surprise signal", "signal"),
        PipelineEdge("training", "execution", "updated weights", "weights"),
    ]

    return nodes, edges


class GraphExplorer:
    """Interactive web-based CubeMind pipeline explorer.

    Generates a self-contained HTML file with D3.js force-directed graph.
    Nodes are clickable to expand subcomponents. Live metrics are shown
    on hover when connected to a MetricsCollector.

    Args:
        collector: Optional MetricsCollector for live metric values.
    """

    def __init__(self, collector: MetricsCollector | None = None) -> None:
        self._collector = collector
        self._nodes, self._edges = build_default_graph()

    def _get_metrics_snapshot(self) -> dict:
        """Get current metric values for all nodes."""
        if not self._collector:
            return {}
        snapshot = {}
        for node in self._nodes:
            for metric in node.metrics:
                latest = self._collector.get_latest(metric)
                if latest:
                    snapshot[metric] = latest.value
            for child in node.children:
                for metric in child.metrics:
                    latest = self._collector.get_latest(metric)
                    if latest:
                        snapshot[metric] = latest.value
        return snapshot

    def export_html(self, path: str = "cubemind_pipeline.html") -> str:
        """Export interactive pipeline graph as self-contained HTML.

        Args:
            path: Output file path.

        Returns:
            Absolute path to the generated file.
        """
        graph_data = {
            "nodes": [n.to_dict() for n in self._nodes],
            "edges": [e.to_dict() for e in self._edges],
            "metrics": self._get_metrics_snapshot(),
        }

        html = _GRAPH_HTML_TEMPLATE.replace("__GRAPH_DATA__", json.dumps(graph_data, indent=2))
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        return str(out.resolve())

    def open_browser(self, path: str = "cubemind_pipeline.html") -> None:
        """Export and open in default browser."""
        filepath = self.export_html(path)
        webbrowser.open(f"file:///{filepath}")


# ── Self-contained HTML template with D3.js ──────────────────────────────

_GRAPH_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CubeMind Pipeline Explorer</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  body { margin: 0; font-family: 'Segoe UI', system-ui, sans-serif; background: #0d1117; color: #c9d1d9; }
  #header { padding: 12px 20px; background: #161b22; border-bottom: 1px solid #30363d; display: flex; align-items: center; gap: 16px; }
  #header h1 { margin: 0; font-size: 18px; color: #58a6ff; }
  #header .subtitle { color: #8b949e; font-size: 13px; }
  #graph { width: 100vw; height: calc(100vh - 50px); }
  .node { cursor: pointer; }
  .node circle { stroke-width: 2px; }
  .node text { font-size: 11px; fill: #c9d1d9; pointer-events: none; }
  .node .metric { font-size: 9px; fill: #8b949e; }
  .link { stroke-opacity: 0.5; }
  .link text { font-size: 9px; fill: #8b949e; }
  #tooltip { position: absolute; background: #1c2128; border: 1px solid #30363d; border-radius: 6px; padding: 10px; font-size: 12px; pointer-events: none; display: none; max-width: 300px; }
  #tooltip .title { color: #58a6ff; font-weight: bold; margin-bottom: 4px; }
  #tooltip .metric-row { color: #8b949e; }
  #tooltip .metric-val { color: #3fb950; font-weight: bold; }
</style>
</head>
<body>
<div id="header">
  <h1>CubeMind Pipeline Explorer</h1>
  <span class="subtitle">Click nodes to expand • Hover for metrics • Drag to rearrange</span>
</div>
<div id="graph"></div>
<div id="tooltip"></div>
<script>
const data = __GRAPH_DATA__;
const stageColors = {
  input: "#8b949e", perception: "#0072B2", routing: "#E69F00",
  memory: "#009E73", detection: "#D55E00", execution: "#CC79A7",
  output: "#56B4E9", training: "#F0E442"
};

const width = window.innerWidth;
const height = window.innerHeight - 50;

const svg = d3.select("#graph").append("svg")
  .attr("width", width).attr("height", height);

const g = svg.append("g");
svg.call(d3.zoom().on("zoom", (e) => g.attr("transform", e.transform)));

// Flatten nodes (top-level only initially)
let nodes = data.nodes.map((n, i) => ({
  ...n, x: width/2 + Math.cos(i*2*Math.PI/data.nodes.length) * 200,
  y: height/2 + Math.sin(i*2*Math.PI/data.nodes.length) * 200,
  expanded: false, radius: 28
}));

let links = data.edges.map(e => ({
  source: e.source, target: e.target, label: e.label, data_type: e.data_type
}));

const simulation = d3.forceSimulation(nodes)
  .force("link", d3.forceLink(links).id(d => d.id).distance(150))
  .force("charge", d3.forceManyBody().strength(-400))
  .force("center", d3.forceCenter(width/2, height/2))
  .force("collision", d3.forceCollide(50));

function render() {
  g.selectAll("*").remove();

  // Edges
  const link = g.selectAll(".link").data(links).join("g").attr("class", "link");
  link.append("line")
    .attr("stroke", d => stageColors[nodes.find(n=>n.id===d.source?.id||d.source)?.stage] || "#30363d")
    .attr("stroke-width", 2);
  link.append("text").text(d => d.label).attr("text-anchor", "middle");

  // Nodes
  const node = g.selectAll(".node").data(nodes).join("g").attr("class", "node")
    .call(d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended));

  node.append("circle")
    .attr("r", d => d.radius)
    .attr("fill", d => stageColors[d.stage] || "#30363d")
    .attr("stroke", d => d.expanded ? "#f0e442" : "#30363d")
    .on("click", (e, d) => toggleExpand(d))
    .on("mouseover", (e, d) => showTooltip(e, d))
    .on("mouseout", () => hideTooltip());

  node.append("text").text(d => d.label).attr("dy", 4).attr("text-anchor", "middle");

  // Children indicator
  node.filter(d => d.children && d.children.length > 0 && !d.expanded)
    .append("text").text(d => "+" + d.children.length).attr("class", "metric")
    .attr("dy", 16).attr("text-anchor", "middle");

  simulation.nodes(nodes);
  simulation.force("link").links(links);
  simulation.alpha(0.3).restart();

  simulation.on("tick", () => {
    link.select("line")
      .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    link.select("text")
      .attr("x", d => (d.source.x+d.target.x)/2).attr("y", d => (d.source.y+d.target.y)/2 - 8);
    node.attr("transform", d => `translate(${d.x},${d.y})`);
  });
}

function toggleExpand(d) {
  if (!d.children || d.children.length === 0) return;
  d.expanded = !d.expanded;
  if (d.expanded) {
    d.children.forEach((c, i) => {
      const child = { ...c, x: d.x + Math.cos(i*Math.PI/3)*80, y: d.y + Math.sin(i*Math.PI/3)*80, radius: 20 };
      nodes.push(child);
      links.push({ source: d.id, target: child.id, label: "", data_type: "internal" });
    });
  } else {
    const childIds = new Set(d.children.map(c => c.id));
    nodes = nodes.filter(n => !childIds.has(n.id));
    links = links.filter(l => !childIds.has(l.target?.id || l.target) && !childIds.has(l.source?.id || l.source));
  }
  render();
}

function showTooltip(event, d) {
  const tip = document.getElementById("tooltip");
  let html = `<div class="title">${d.label}</div>`;
  if (d.metadata?.desc) html += `<div>${d.metadata.desc}</div>`;
  if (d.metrics?.length) {
    d.metrics.forEach(m => {
      const val = data.metrics[m];
      html += `<div class="metric-row">${m}: <span class="metric-val">${val !== undefined ? val.toFixed(3) : '---'}</span></div>`;
    });
  }
  if (d.children?.length && !d.expanded) html += `<div style="margin-top:4px;color:#f0e442">Click to expand ${d.children.length} subcomponents</div>`;
  tip.innerHTML = html;
  tip.style.display = "block";
  tip.style.left = (event.pageX + 12) + "px";
  tip.style.top = (event.pageY - 10) + "px";
}

function hideTooltip() { document.getElementById("tooltip").style.display = "none"; }

function dragstarted(e, d) { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }
function dragged(e, d) { d.fx = e.x; d.fy = e.y; }
function dragended(e, d) { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }

render();
</script>
</body>
</html>"""
