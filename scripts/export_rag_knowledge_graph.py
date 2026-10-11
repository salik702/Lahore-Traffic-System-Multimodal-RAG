"""Read-only export of the project's actual persisted LightRAG graph.

Run with backend/.venv/Scripts/python.exe scripts/export_rag_knowledge_graph.py.
No LLM calls, ingestion, source mutations, or illustrative graph data.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import textwrap
from collections import Counter
from datetime import datetime, timezone
from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'backend/rag_storage/graph_chunk_entity_relation.graphml'
OUT = ROOT / 'graphs'
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
    selected = list(nx.ego_graph(graph, anchor, radius=1))[:18] if anchor else []
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
             '<style>text{font-family:Segoe UI,Arial,sans-serif}.entity{font-size:16px;font-weight:600;fill:#182e48}.category{font-size:12px;fill:#4c6179}.edge-label{font-size:11px;fill:#617389;paint-order:stroke;stroke:#f8fbff;stroke-width:5px;stroke-linejoin:round}</style>',
             '<text x="54" y="57" font-size="30" font-weight="700" fill="#142a43">Lahore Traffic Intelligence AI</text>',
             '<text x="54" y="94" font-size="21" fill="#435b77">Actual RAG-Anything / LightRAG knowledge graph</text>',
             f'<text x="54" y="128" font-size="16" fill="#435b77">CONNECTED SUBSET · {len(sub)} / {len(graph)} nodes · {sub.number_of_edges()} / {graph.number_of_edges()} edges · complete graph in GraphML and HTML</text>',
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
            parts.append(f'<text class="edge-label" x="{tx}" y="{ty-5}" text-anchor="middle">{escape(label)}</text>')
        parts.append('</g>')
    for name in sub:
        x, y = positions[name]
        attrs = dict(sub.nodes[name]); category = str(attrs.get('entity_type', 'unrecorded'))
        lines = textwrap.wrap(str(name), width=42, break_long_words=True) or ['']
        box_height = 30 + len(lines)*21
        parts.append(f'<g class="graph-node" data-node-id="{escape(str(name), quote=True)}"><title>{escape(json.dumps({"id": name, **attrs}, ensure_ascii=False))}</title>')
        parts.append(f'<rect x="{x-220}" y="{y-box_height/2}" width="440" height="{box_height}" rx="12" fill="white" stroke="{colors[category]}" stroke-width="{4 if name == anchor else 2}"/>')
        for i, line in enumerate(lines):
            parts.append(f'<text class="entity" x="{x}" y="{y-box_height/2+23+i*21}" text-anchor="middle">{escape(line)}</text>')
        parts.append(f'<text class="category" x="{x}" y="{y+box_height/2-9}" text-anchor="middle">{escape(category)}{ " · stored image anchor" if name == anchor else ""}</text></g>')
    legend_y = height-130
    for index, category in enumerate(sorted({d.get('entity_type', 'unrecorded') for _, d in sub.nodes(data=True)})):
        x = 54+(index%7)*245; y = legend_y+(index//7)*28
        parts.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{colors[category]}"/><text x="{x+14}" y="{y+5}" font-size="13" fill="#435b77">{escape(category)}</text>')
    parts.extend([f'<text x="54" y="{height-52}" font-size="13" fill="#617389">Purple dashed: stored belongs_to image grounding · gray: other stored relationships · full relationship metadata in SVG tooltips and HTML</text>',
                  f'<text x="54" y="{height-27}" font-size="12" fill="#617389">Image anchor may contain source-caption fallback. Dates and extracted assertions retain their original meaning; this view does not independently verify them.</text>', '</svg>'])
    return '\n'.join(parts), sub


def main():
    source_bytes = SOURCE.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    # Same public NetworkX reader used by installed NetworkXStorage.load_nx_graph.
    # Reading bytes once gives every output the same consistent snapshot.
    graph = nx.read_graphml(io.BytesIO(source_bytes))
    if not graph:
        raise RuntimeError('Stored graph is empty; inspect ingestion before exporting. No sample data will be generated.')
    OUT.mkdir(parents=True, exist_ok=True)
    exported = OUT / 'rag_anything_knowledge_graph.graphml'
    exported.write_bytes(source_bytes)  # preserve every XML key, ID, and original attribute
    colors = {category: PALETTE[i % len(PALETTE)] for i, category in enumerate(sorted({str(d.get('entity_type', 'unrecorded')) for _, d in graph.nodes(data=True)}))}
    layout = nx.spring_layout(graph, seed=42, iterations=250, weight=None, scale=1100)
    nodes = []
    for name, attrs in graph.nodes(data=True):
        category = str(attrs.get('entity_type', 'unrecorded'))
        explicit_modality = attrs.get('modality')
        modality = explicit_modality or ('image' if category.lower() == 'image' else None)
        nodes.append({'id': name, 'name': attrs.get('entity_name', name), 'category': category,
                      'modality': modality, 'modality_basis': 'stored modality' if explicit_modality else 'stored entity_type=image' if modality else 'not explicitly recorded',
                      'degree': graph.degree[name], 'x': float(layout[name][0]), 'y': float(layout[name][1]), 'attributes': dict(attrs)})
    edges = [{'id': f'edge-{i}', 'source': u, 'target': v, 'relationship': relationship(attrs),
              'image_grounding': 'belongs_to' in str(attrs.get('keywords', '')) and any(str(graph.nodes[n].get('entity_type', '')).lower() == 'image' for n in (u, v)),
              'attributes': dict(attrs)} for i, (u, v, attrs) in enumerate(graph.edges(data=True))]
    provenance = {'source': str(SOURCE.relative_to(ROOT)).replace('\\', '/'), 'sha256': digest,
                  'exported_at_utc': datetime.now(timezone.utc).isoformat(), 'directed': graph.is_directed(),
                  'nodes': len(graph), 'edges': graph.number_of_edges(),
                  'versions': {p: importlib.metadata.version(p) for p in ['raganything', 'lightrag-hku', 'networkx']}}
    payload = {'nodes': nodes, 'edges': edges, 'colors': colors, 'provenance': provenance}
    data = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    library = (OUT / 'vendor/vis-network-9.1.9.min.js').read_text(encoding='utf-8')
    template = (ROOT / 'scripts/rag_graph_viewer.html').read_text(encoding='utf-8')
    html = template.replace('__GRAPH_DATA__', data).replace('__VIS_LIBRARY__', library.replace('</script', '<\\/script'))
    (OUT / 'rag_anything_knowledge_graph.html').write_text(html, encoding='utf-8')
    svg, subgraph = svg_subset(graph, colors)
    (OUT / 'rag_anything_knowledge_graph.svg').write_text(svg, encoding='utf-8')
    # Export integrity checks, not an automated test suite.
    ET.parse(exported)
    restored = nx.read_graphml(exported)
    if dict(restored.nodes(data=True)) != dict(graph.nodes(data=True)) or list(restored.edges(data=True)) != list(graph.edges(data=True)):
        raise RuntimeError('GraphML attributes or topology do not match source')
    if any(e['source'] not in graph or e['target'] not in graph for e in edges):
        raise RuntimeError('Invalid edge endpoint')
    embedded = html.split('<script id="graph-data" type="application/json">', 1)[1].split('</script>', 1)[0]
    restored_payload = json.loads(embedded)
    if restored_payload != payload:
        raise RuntimeError('Embedded HTML graph data differs from export payload')
    svg_xml = ET.fromstring(svg)
    svg_ids = {element.attrib['data-node-id'] for element in svg_xml.iter() if 'data-node-id' in element.attrib}
    if svg_ids != set(subgraph):
        raise RuntimeError('SVG node IDs do not match selected real subgraph')
    svg_edges = [element for element in svg_xml.iter() if element.attrib.get('class') == 'graph-edge']
    if len(svg_edges) != subgraph.number_of_edges():
        raise RuntimeError('SVG edge count differs from selected real subgraph')
    for element in svg_edges:
        u, v = element.attrib['data-source'], element.attrib['data-target']
        if not subgraph.has_edge(u, v):
            raise RuntimeError('SVG edge does not exist in original graph')
        title = element.find('{http://www.w3.org/2000/svg}title')
        if json.loads(title.text) != {'source': u, 'target': v, **dict(subgraph.edges[u, v])}:
            raise RuntimeError('SVG relationship metadata differs from original graph')
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Source changed during export; rerun for a consistent snapshot')
    print(json.dumps({**provenance, 'categories': dict(Counter(n['category'] for n in nodes)),
                      'image_grounding_edges': sum(e['image_grounding'] for e in edges),
                      'svg_nodes': len(subgraph), 'svg_edges': subgraph.number_of_edges(),
                      'graphml_byte_identical': exported.read_bytes() == source_bytes}, indent=2))


if __name__ == '__main__':
    main()
