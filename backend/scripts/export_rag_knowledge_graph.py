"""Export the actual persisted project LightRAG graph without modifying the index.

From the repository root:
    backend/.venv/Scripts/python.exe backend/scripts/export_rag_knowledge_graph.py

Uses installed NetworkXStorage's supported loader, not capped query APIs.
No ingestion, LLM calls, embedding service, or fabricated graph data.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import sys
import textwrap
from collections import Counter
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from app.core.config import Settings

PALETTE = ['#2176ae', '#d49a24', '#6554b0', '#d36659', '#299282', '#7a8e30',
           '#bb5c9b', '#537ba2', '#9d693e', '#4b969f', '#76639b', '#8a7770']


def relationship(attrs):
    return str(attrs.get('relationship') or attrs.get('keywords') or attrs.get('description') or '')


def svg_subset(graph, colors):
    # Deterministic connected ego graph of the earliest image anchor in storage.
    # Every included node and edge comes from the original graph.
    anchor = next((n for n, d in graph.nodes(data=True) if d.get('entity_type') == 'image'), None)
    if anchor is None:
        anchor = max(graph, key=lambda n: (graph.degree[n], str(n))) if graph else None
    # BFS keeps the chosen subset connected even when the neighborhood is large.
    selected = list(nx.bfs_tree(graph.to_undirected(), anchor))[:min(18, 1 + len(set(graph.to_undirected().neighbors(anchor))))] if anchor else []
    sub = graph.subgraph(selected)
    neighbors = sorted((n for n in sub if n != anchor), key=str)
    width, height = 1920, max(1450, 470 + math.ceil(len(neighbors)/2)*160)
    center = (width/2, height/2 + 40)
    positions = {anchor: center} if anchor else {}
    # Wide two-column layout keeps long source and entity labels readable.
    left, right = neighbors[::2], neighbors[1::2]
    for column, x in [(left, 270), (right, width-270)]:
        for index, name in enumerate(column):
            positions[name] = (x, 290 + index*160)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<title>Actual LightRAG knowledge graph — connected image neighborhood subset</title>',
             f'<desc>Connected induced subset: {len(sub)} of {len(graph)} nodes, {sub.number_of_edges()} of {graph.number_of_edges()} edges. Original IDs and attributes in node and edge titles. No synthetic graph entities.</desc>',
             '<rect width="100%" height="100%" fill="#f8fbff"/>',
             '<style>text{font-family:Segoe UI,Arial,sans-serif}.entity{font-size:16px;font-weight:600;fill:#182e48}.category{font-size:12px;fill:#4c6179}.edge-label{font-size:14px;fill:#617389}</style>',
             '<text x="54" y="57" font-size="30" font-weight="700" fill="#142a43">Lahore Traffic Intelligence AI</text>',
             '<text x="54" y="94" font-size="21" fill="#435b77">Actual RAG-Anything / LightRAG knowledge graph</text>',
             f'<text x="54" y="128" font-size="16" fill="#435b77">CONNECTED SUBSET · {len(sub)} / {len(graph)} nodes · {sub.number_of_edges()} / {graph.number_of_edges()} edges · complete graph in GraphML</text>',
             '<text x="54" y="156" font-size="13" fill="#617389">Selected: first stored image anchor and its neighbors, with all relationships between those nodes. Edge orientation is undirected in storage.</text>']
    # Draw secondary edges first, with quadratic curves outside the node columns.
    for index, (u, v, attrs) in enumerate(sub.edges(data=True)):
        x1, y1 = positions[u]; x2, y2 = positions[v]
        grounding = 'belongs_to' in str(attrs.get('keywords', '')) and anchor in (u, v)
        color = '#9564c5' if grounding else '#b2c1d2'
        dash = ' stroke-dasharray="7 5"' if grounding else ''
        if anchor in (u, v):
            path = f'M {x1} {y1} L {x2} {y2}'
            tx, ty = x1*0.45+x2*0.55, y1*0.45+y2*0.55
        else:
            cx = 45 if x1 == x2 == 270 else width-45 if x1 == x2 else width/2
            cy = (y1+y2)/2 if x1 == x2 else 200+(index%5)*25
            path = f'M {x1} {y1} Q {cx} {cy} {x2} {y2}'
            tx, ty = (x1+2*cx+x2)/4, (y1+2*cy+y2)/4
        title = escape(json.dumps({'source': u, 'target': v, **attrs}, ensure_ascii=False))
        parts.append(f'<g class="graph-edge" data-source="{escape(u, quote=True)}" data-target="{escape(v, quote=True)}"><title>{title}</title><path d="{path}" fill="none" stroke="{color}" stroke-width="1.6"{dash}/>')
        label = 'belongs_to' if grounding else textwrap.shorten(relationship(attrs), width=38, placeholder='…')
        # Label anchor links. Secondary relationships remain available as SVG tooltips.
        if anchor in (u, v):
            label_width = max(90, len(label)*8)
            parts.append(f'<rect x="{tx-label_width/2}" y="{ty-20}" width="{label_width}" height="20" rx="4" fill="#f8fbff"/>')
            parts.append(f'<text font-size="14" fill="#617389" class="edge-label" x="{tx}" y="{ty-5}" text-anchor="middle">{escape(label)}</text>')
        parts.append('</g>')
    for name in sub:
        x, y = positions[name]
        attrs = dict(sub.nodes[name]); category = str(attrs.get('entity_type', 'unrecorded'))
        lines = textwrap.wrap(str(attrs.get('entity_name') or name), width=42, break_long_words=True) or ['']
        box_height = 30 + len(lines)*21
        parts.append(f'<g class="graph-node" data-node-id="{escape(str(name), quote=True)}"><title>{escape(json.dumps({"id": name, **attrs}, ensure_ascii=False))}</title>')
        parts.append(f'<rect x="{x-220}" y="{y-box_height/2}" width="440" height="{box_height}" rx="12" fill="white" stroke="{colors[category]}" stroke-width="{4 if name == anchor else 2}"/>')
        for i, line in enumerate(lines):
            parts.append(f'<text font-size="16" font-weight="600" fill="#182e48" class="entity" x="{x}" y="{y-box_height/2+23+i*21}" text-anchor="middle">{escape(line)}</text>')
        parts.append(f'<text font-size="12" fill="#4c6179" class="category" x="{x}" y="{y+box_height/2-9}" text-anchor="middle">{escape(category)}{ " · stored image anchor" if name == anchor else ""}</text></g>')
    legend_y = height-130
    for index, category in enumerate(sorted({d.get('entity_type', 'unrecorded') for _, d in sub.nodes(data=True)})):
        x = 54+(index%7)*245; y = legend_y+(index//7)*28
        parts.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{colors[category]}"/><text x="{x+14}" y="{y+5}" font-size="13" fill="#435b77">{escape(category)}</text>')
    parts.extend([f'<text x="54" y="{height-52}" font-size="13" fill="#617389">Purple dashed: stored belongs_to image grounding · gray: other stored relationships · full relationship metadata in SVG tooltips </text>',
                  f'<text x="54" y="{height-27}" font-size="12" fill="#617389">Image anchor may contain source-caption fallback. Dates and extracted assertions retain their original meaning; this view does not independently verify them.</text>', '</svg>'])
    return '\n'.join(parts).replace('<text ', '<text font-family="Segoe UI,Arial,sans-serif" '), sub


def missing_metadata(graph):
    """Count absent/blank fields; preserve originals rather than filling gaps."""
    nodes = list(graph.nodes(data=True))
    edges = list(graph.edges(data=True))
    def missing(records, fields):
        return {field: sum(attrs.get(field) in (None, "") for attrs in records)
                for field in fields}
    return {
        "nodes_missing_or_blank": missing([d for _, d in nodes],
            ["entity_id", "entity_name", "entity_type", "description", "source_id", "file_path", "modality"]),
        "edges_missing_or_blank": missing([d for _, _, d in edges],
            ["description", "keywords", "weight", "source_id", "file_path", "modality"]),
        "name_display": "Stored entity_name when present, otherwise original XML node ID; no invented names.",
        "modality_display": "Stored image category and belongs_to keywords only; no new modality assertions.",
    }


def ingestion_diagnostics(settings):
    result = {"working_directory": str(settings.rag_storage)}
    for name, path in {
        "ingestion_manifest": settings.data_dir / "processed/metadata/ingestion_manifest.json",
        "document_status": settings.rag_storage / "kv_store_doc_status.json",
    }.items():
        if not path.is_file():
            result[name] = "unavailable"
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if name == "document_status":
            result[name] = [{"doc_id": k, "status": v.get("status"),
                             "error": v.get("error_msg"),
                             "multimodal_processed": v.get("multimodal_processed")}
                            for k, v in data.items()]
        else:
            result[name] = {k: data.get(k) for k in ("status", "failures", "totals", "completed_at")}
    return result


def validate_svg(svg, subset):
    xml = ET.fromstring(svg)
    node_elements = [e for e in xml.iter() if "data-node-id" in e.attrib]
    if len(node_elements) != len(subset) or {e.attrib["data-node-id"] for e in node_elements} != set(subset):
        raise RuntimeError("SVG node IDs differ from extracted subset")
    for e in node_elements:
        node_id = e.attrib["data-node-id"]
        attrs = json.loads(e.find("{http://www.w3.org/2000/svg}title").text)
        if attrs != {"id": node_id, **dict(subset.nodes[node_id])}:
            raise RuntimeError("SVG node metadata differs from source")
    edge_elements = [e for e in xml.iter() if e.attrib.get("class") == "graph-edge"]
    if len(edge_elements) != subset.number_of_edges():
        raise RuntimeError("SVG edge count differs from extracted subset")
    seen = set()
    for e in edge_elements:
        u, v = e.attrib["data-source"], e.attrib["data-target"]
        if not subset.has_edge(u, v):
            raise RuntimeError("SVG relationship is absent from source")
        identity = (u, v) if subset.is_directed() else frozenset((u, v))
        if identity in seen:
            raise RuntimeError("Duplicate SVG relationship")
        seen.add(identity)
        attrs = json.loads(e.find("{http://www.w3.org/2000/svg}title").text)
        if attrs != {"source": u, "target": v, **dict(subset.edges[u, v])}:
            raise RuntimeError("SVG relationship metadata differs from source")


def main():
    # Absolute dotenv paths make configuration independent of the terminal's cwd.
    # Pydantic still gives OS environment variables precedence, as the app does.
    settings = Settings(_env_file=(ROOT / ".env", ROOT / "backend/.env"))
    source = settings.rag_storage / "graph_chunk_entity_relation.graphml"
    if not source.is_file():
        raise RuntimeError("No persisted LightRAG graph at " + str(source) + "\n" +
                           json.dumps(ingestion_diagnostics(settings), indent=2))
    source_bytes = source.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    from lightrag.kg.networkx_impl import NetworkXStorage
    # This public installed storage loader uses nx.read_graphml. No storage
    # initialization or write callbacks are needed for a persisted snapshot.
    source_graph = NetworkXStorage.load_nx_graph(str(source))
    graph = nx.read_graphml(io.BytesIO(source_bytes))
    if not nx.utils.graphs_equal(source_graph, graph):
        raise RuntimeError("Storage changed during loading; rerun after ingestion finishes")
    if not graph or not graph.number_of_edges():
        raise RuntimeError("Graph has no entities or relationships; no sample graph generated.\n" +
                           json.dumps(ingestion_diagnostics(settings), indent=2))
    if graph.is_multigraph() or graph.is_directed():
        raise RuntimeError("SVG renderer expects this project's undirected simple NetworkXStorage graph")
    # Validate XML IDs and endpoints before NetworkX can implicitly add nodes.
    xml = ET.fromstring(source_bytes)
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    ids = [n.attrib["id"] for n in xml.findall(".//g:node", ns)]
    id_set = set(ids)
    xml_edges = xml.findall(".//g:edge", ns)
    if len(ids) != len(set(ids)) or set(ids) != set(graph):
        raise RuntimeError("Duplicate or inconsistent original XML node IDs")
    if len(xml_edges) != graph.number_of_edges():
        raise RuntimeError("XML relationship count differs from storage reader")
    if any(e.attrib["source"] not in id_set or e.attrib["target"] not in id_set for e in xml_edges):
        raise RuntimeError("Original GraphML contains an invalid relationship endpoint")
    categories = Counter(str(d.get("entity_type", "unrecorded")) for _, d in graph.nodes(data=True))
    colors = {category: PALETTE[i % len(PALETTE)] for i, category in enumerate(sorted(categories))}
    svg, subset = svg_subset(graph, colors)
    validate_svg(svg, subset)
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Source changed during export; rerun after ingestion finishes")
    out = ROOT / "graphs"
    out.mkdir(parents=True, exist_ok=True)
    graphml_path = out / "lahore_traffic_rag_anything.graphml"
    svg_path = out / "lahore_traffic_rag_anything.svg"
    if graphml_path.resolve() == source.resolve():
        raise RuntimeError("Export must not overwrite the original graph")
    # Copy exact XML: preserve every key, ID, reference, attribute and direction.
    graphml_path.write_bytes(source_bytes)
    svg_path.write_text(svg, encoding="utf-8")
    ET.parse(graphml_path)
    restored = nx.read_graphml(graphml_path)
    if not nx.utils.graphs_equal(restored, graph):
        raise RuntimeError("Exported GraphML differs from source graph")
    validate_svg(svg_path.read_text(encoding="utf-8"), subset)
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise RuntimeError("Source changed during export; rerun after ingestion finishes")
    print(json.dumps({
        "source": str(source), "backend": "NetworkXStorage", "source_sha256": digest,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {p: importlib.metadata.version(p) for p in ("raganything", "lightrag-hku", "networkx")},
        "source_nodes": len(source_graph), "source_edges": source_graph.number_of_edges(),
        "exported_nodes": len(restored), "exported_edges": restored.number_of_edges(),
        "directed": graph.is_directed(), "categories": dict(categories),
        "svg_nodes": len(subset), "svg_edges": subset.number_of_edges(),
        "svg_selection": "Connected image anchor neighborhood, capped at 18 nodes; all stored edges between selected nodes",
        "graphml_byte_identical": graphml_path.read_bytes() == source_bytes,
        "metadata_report": missing_metadata(graph),
        "outputs": [str(graphml_path), str(svg_path)],
        "validation": "XML, unique IDs, endpoints, source/export topology and attributes, SVG node/edge metadata, source hash: passed",
    }, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
