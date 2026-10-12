"""Download, cache, and parse a real Lahore OSM extract into a simulation graph."""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import httpx

from app.core.config import Settings

HIGHWAYS = {
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified",
    "residential", "living_street", "motorway_link", "trunk_link", "primary_link",
    "secondary_link", "tertiary_link",
}
DEFAULT_SPEED = {
    "motorway": 22.0, "trunk": 16.7, "primary": 13.9, "secondary": 13.9,
    "tertiary": 11.1, "unclassified": 8.3, "residential": 8.3, "living_street": 5.6,
    "service": 5.6,
}
KALMA_BBOX = {"south": 31.4990, "west": 74.3260, "north": 31.5080, "east": 74.3370}


def simulation_dir(settings: Settings) -> Path:
    path = settings.project_root / "data" / "simulation" / "kalma_chowk"
    path.mkdir(parents=True, exist_ok=True)
    return path


def osm_path(settings: Settings) -> Path:
    return simulation_dir(settings) / "kalma_chowk.osm.xml"


def network_path(settings: Settings) -> Path:
    return simulation_dir(settings) / "network.json"


def download_osm(settings: Settings, bbox: dict | None = None, force: bool = False) -> Path:
    dest = osm_path(settings)
    if dest.is_file() and dest.stat().st_size > 10_000 and not force:
        return dest
    box = bbox or KALMA_BBOX
    url = (
        "https://api.openstreetmap.org/api/0.6/map"
        f"?bbox={box['west']},{box['south']},{box['east']},{box['north']}"
    )
    headers = {"User-Agent": "LahoreTrafficDigitalTwin/1.0 (research; local cache)"}
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=90.0, follow_redirects=True, headers=headers) as client:
        response = client.get(url)
        response.raise_for_status()
        dest.write_bytes(response.content)
    if dest.stat().st_size < 1000:
        raise RuntimeError("OSM download was empty")
    return dest


def _tags(element: ET.Element) -> dict[str, str]:
    return {child.attrib["k"]: child.attrib["v"] for child in element if child.tag == "tag" and "k" in child.attrib}


def _parse_speed(raw: str | None, highway: str) -> float:
    if not raw:
        return DEFAULT_SPEED.get(highway, 11.1)
    text = raw.lower().replace("km/h", "").replace("kph", "").strip()
    try:
        value = float(text.split()[0])
        if "mph" in raw.lower():
            return value * 0.44704
        return value / 3.6
    except ValueError:
        return DEFAULT_SPEED.get(highway, 11.1)


def _parse_lanes(raw: str | None, highway: str, oneway: bool, directional_raw: str | None = None) -> int:
    lane_value = directional_raw or raw
    if lane_value:
        try:
            lanes = max(1, int(float(lane_value.split(";")[0])))
            return lanes if directional_raw or oneway else max(1, math.ceil(lanes / 2))
        except ValueError:
            pass
    return 2 if highway in {"motorway", "trunk", "primary"} else 1


def _is_oneway(tags: dict[str, str], highway: str) -> tuple[bool, bool]:
    value = tags.get("oneway", "").lower()
    if value in {"yes", "true", "1"}:
        return True, False
    if value == "-1":
        return True, True
    if highway in {"motorway", "motorway_link", "trunk_link"}:
        return True, False
    return False, False


def _project(lat: float, lon: float, origin: tuple[float, float]) -> tuple[float, float]:
    lat0, lon0 = origin
    x = (lon - lon0) * math.cos(math.radians(lat0)) * 111_320
    y = (lat - lat0) * 110_540
    return x, y


def _heading(ax: float, ay: float, bx: float, by: float) -> float:
    return math.degrees(math.atan2(bx - ax, by - ay)) % 360


def _distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(bx - ax, by - ay)


def parse_osm(path: Path) -> dict:
    tree = ET.parse(path)
    root = tree.getroot()
    nodes: dict[str, dict] = {}
    for element in root.findall("node"):
        nid = element.attrib["id"]
        lat = float(element.attrib["lat"])
        lon = float(element.attrib["lon"])
        tags = _tags(element)
        nodes[nid] = {"id": nid, "lat": lat, "lon": lon, "tags": tags}

    ways = []
    for element in root.findall("way"):
        tags = _tags(element)
        highway = tags.get("highway", "")
        if highway not in HIGHWAYS:
            continue
        refs = [nd.attrib["ref"] for nd in element.findall("nd") if nd.attrib.get("ref") in nodes]
        if len(refs) < 2:
            continue
        ways.append({"id": element.attrib["id"], "refs": refs, "tags": tags, "highway": highway})

    usage: dict[str, int] = defaultdict(int)
    for way in ways:
        for ref in way["refs"]:
            usage[ref] += 1
        usage[way["refs"][0]] += 1
        usage[way["refs"][-1]] += 1

    split_ways = []
    for way in ways:
        refs = way["refs"]
        current = [refs[0]]
        for ref in refs[1:]:
            current.append(ref)
            tags = nodes[ref]["tags"]
            is_junction = usage[ref] >= 3 or tags.get("highway") == "traffic_signals" or "crossing" in tags
            if is_junction or ref == refs[-1]:
                if len(current) >= 2:
                    split_ways.append({**way, "refs": current[:]})
                current = [ref]

    if not nodes or not split_ways:
        raise RuntimeError("OSM extract contained no usable roads")

    lats = [n["lat"] for n in nodes.values()]
    lons = [n["lon"] for n in nodes.values()]
    origin = (sum(lats) / len(lats), sum(lons) / len(lons))
    for node in nodes.values():
        node["x"], node["y"] = _project(node["lat"], node["lon"], origin)

    edges: list[dict] = []
    incoming: dict[str, list[str]] = defaultdict(list)
    outgoing: dict[str, list[str]] = defaultdict(list)

    def add_edge(way: dict, refs: list[str], reverse: bool) -> None:
        chain = list(reversed(refs)) if reverse else refs
        pts = [nodes[r] for r in chain]
        poly = []
        length = 0.0
        for index, pt in enumerate(pts):
            if index:
                length += _distance(pts[index - 1]["x"], pts[index - 1]["y"], pt["x"], pt["y"])
            poly.append({"lat": pt["lat"], "lon": pt["lon"], "x": pt["x"], "y": pt["y"], "s": length})
        if length < 8:
            return
        tags = way["tags"]
        highway = way["highway"]
        oneway, _ = _is_oneway(tags, highway)
        lane_direction = "backward" if reverse else "forward"
        directional_lanes = tags.get(f"lanes:{lane_direction}")
        lane_count = _parse_lanes(tags.get("lanes"), highway, oneway, directional_lanes)
        lane_source = (
            f"OSM lanes:{lane_direction}"
            if directional_lanes
            else "OSM lanes tag split by direction"
            if tags.get("lanes") and not oneway
            else "OSM lanes tag"
            if tags.get("lanes")
            else "estimated from road class"
        )
        speed = _parse_speed(tags.get("maxspeed"), highway)
        speed_source = "OSM maxspeed" if tags.get("maxspeed") else "estimated from road class"
        edge_id = f"{way['id']}:{'rev' if reverse else 'fwd'}:{chain[0]}-{chain[-1]}"
        name = tags.get("name") or tags.get("name:en") or highway
        heading = _heading(pts[0]["x"], pts[0]["y"], pts[-1]["x"], pts[-1]["y"])
        edge = {
            "id": edge_id,
            "osm_way": way["id"],
            "from": chain[0],
            "to": chain[-1],
            "name": name,
            "highway": highway,
            "length": round(length, 2),
            "lanes": lane_count,
            "lane_source": lane_source,
            "speed": round(speed, 3),
            "speed_source": speed_source,
            "heading": round(heading, 1),
            "polyline": poly,
        }
        edges.append(edge)
        outgoing[chain[0]].append(edge_id)
        incoming[chain[-1]].append(edge_id)

    for way in split_ways:
        oneway, reversed_dir = _is_oneway(way["tags"], way["highway"])
        if reversed_dir:
            add_edge(way, way["refs"], True)
        else:
            add_edge(way, way["refs"], False)
            if not oneway:
                add_edge(way, way["refs"], True)

    used_nodes = {edge["from"] for edge in edges} | {edge["to"] for edge in edges}
    signal_nodes = {
        nid for nid, node in nodes.items()
        if node["tags"].get("highway") == "traffic_signals" or node["tags"].get("crossing") == "traffic_signals"
    }

    junctions = []
    for nid in used_nodes:
        inc = incoming.get(nid, [])
        out = outgoing.get(nid, [])
        named = nodes[nid]["tags"].get("name") or nodes[nid]["tags"].get("name:en")
        degree = len(inc)
        controllable = degree >= 3 or nid in signal_nodes
        if degree < 2 and nid not in signal_nodes:
            continue
        phases = _phases_for(inc, {e["id"]: e for e in edges}) if controllable else []
        junctions.append({
            "id": nid,
            "lat": nodes[nid]["lat"],
            "lon": nodes[nid]["lon"],
            "x": nodes[nid]["x"],
            "y": nodes[nid]["y"],
            "name": named or ("Kalma Chowk" if _near_kalma(nodes[nid]) else f"Junction {nid}"),
            "incoming": inc,
            "outgoing": out,
            "controllable": controllable and bool(phases),
            "has_osm_signals": nid in signal_nodes,
            "phases": phases,
        })

    if not any(j["controllable"] for j in junctions):
        # Guarantee at least one signalized intersection on the densest node.
        densest = max(junctions, key=lambda item: len(item["incoming"]), default=None)
        if densest and densest["incoming"]:
            densest["controllable"] = True
            densest["phases"] = _phases_for(densest["incoming"], {e["id"]: e for e in edges})

    node_out = {
        nid: {"id": nid, "lat": nodes[nid]["lat"], "lon": nodes[nid]["lon"], "x": nodes[nid]["x"], "y": nodes[nid]["y"]}
        for nid in used_nodes
    }
    bounds = {
        "min_lat": min(n["lat"] for n in node_out.values()),
        "max_lat": max(n["lat"] for n in node_out.values()),
        "min_lon": min(n["lon"] for n in node_out.values()),
        "max_lon": max(n["lon"] for n in node_out.values()),
    }
    return {
        "source": "OpenStreetMap",
        "license": "ODbL-1.0",
        "area": "Kalma Chowk, Lahore, Pakistan",
        "synthetic": False,
        "origin": {"lat": origin[0], "lon": origin[1]},
        "bounds": bounds,
        "nodes": node_out,
        "edges": {edge["id"]: edge for edge in edges},
        "junctions": {item["id"]: item for item in junctions},
        "stats": {
            "nodes": len(node_out),
            "edges": len(edges),
            "junctions": len(junctions),
            "signals": sum(1 for item in junctions if item["controllable"]),
        },
    }


def _near_kalma(node: dict) -> bool:
    return abs(node["lat"] - 31.5035) < 0.0015 and abs(node["lon"] - 74.3315) < 0.0015


def _phases_for(incoming_ids: list[str], edges: dict) -> list[dict]:
    if not incoming_ids:
        return []
    groups: list[list[str]] = []
    remaining = list(incoming_ids)
    while remaining:
        seed = remaining.pop(0)
        heading = edges[seed]["heading"]
        allied = [seed]
        rest = []
        for edge_id in remaining:
            delta = abs((edges[edge_id]["heading"] - heading + 180) % 360 - 180)
            # Opposite approaches (≈180°) can share green; perpendicular cannot.
            if delta < 35 or delta > 145:
                allied.append(edge_id)
            else:
                rest.append(edge_id)
        groups.append(allied)
        remaining = rest
    if len(groups) == 1 and len(incoming_ids) >= 2:
        groups = [[incoming_ids[i]] for i in range(len(incoming_ids))]
    phases = []
    for index, group in enumerate(groups[:4]):
        phases.append({"id": index, "green": group, "min_green": 8.0, "max_green": 45.0, "yellow": 3.0})
    return phases or [{"id": 0, "green": incoming_ids[:1], "min_green": 8.0, "max_green": 45.0, "yellow": 3.0}]


def synthetic_network() -> dict:
    """Labeled fallback if OSM cannot be parsed. Four-arm intersection, not a real Lahore layout."""
    nodes = {
        "c": {"id": "c", "lat": 31.5035, "lon": 74.3310, "x": 0, "y": 0},
        "n": {"id": "n", "lat": 31.5053, "lon": 74.3310, "x": 0, "y": 200},
        "s": {"id": "s", "lat": 31.5017, "lon": 74.3310, "x": 0, "y": -200},
        "e": {"id": "e", "lat": 31.5035, "lon": 74.3331, "x": 200, "y": 0},
        "w": {"id": "w", "lat": 31.5035, "lon": 74.3289, "x": -200, "y": 0},
    }

    def edge(eid, frm, to, heading, name):
        a, b = nodes[frm], nodes[to]
        length = _distance(a["x"], a["y"], b["x"], b["y"])
        return {
            "id": eid, "osm_way": "synthetic", "from": frm, "to": to, "name": name,
            "highway": "primary", "length": length, "lanes": 2, "speed": 13.9, "heading": heading,
            "polyline": [
                {"lat": a["lat"], "lon": a["lon"], "x": a["x"], "y": a["y"], "s": 0},
                {"lat": b["lat"], "lon": b["lon"], "x": b["x"], "y": b["y"], "s": length},
            ],
        }

    edges = [
        edge("n-c", "n", "c", 180, "Synthetic N approach"),
        edge("s-c", "s", "c", 0, "Synthetic S approach"),
        edge("e-c", "e", "c", 270, "Synthetic E approach"),
        edge("w-c", "w", "c", 90, "Synthetic W approach"),
        edge("c-n", "c", "n", 0, "Synthetic N exit"),
        edge("c-s", "c", "s", 180, "Synthetic S exit"),
        edge("c-e", "c", "e", 90, "Synthetic E exit"),
        edge("c-w", "c", "w", 270, "Synthetic W exit"),
    ]
    incoming = ["n-c", "s-c", "e-c", "w-c"]
    return {
        "source": "synthetic-demonstration",
        "license": "not-osm",
        "area": "Synthetic demonstration network (OSM unavailable)",
        "synthetic": True,
        "origin": {"lat": 31.5035, "lon": 74.3310},
        "bounds": {"min_lat": 31.5017, "max_lat": 31.5053, "min_lon": 74.3289, "max_lon": 74.3331},
        "nodes": nodes,
        "edges": {item["id"]: item for item in edges},
        "junctions": {
            "c": {
                "id": "c", "lat": 31.5035, "lon": 74.3310, "x": 0, "y": 0,
                "name": "Synthetic four-arm junction",
                "incoming": incoming, "outgoing": ["c-n", "c-s", "c-e", "c-w"],
                "controllable": True, "has_osm_signals": False,
                "phases": [
                    {"id": 0, "green": ["n-c", "s-c"], "min_green": 8.0, "max_green": 45.0, "yellow": 3.0},
                    {"id": 1, "green": ["e-c", "w-c"], "min_green": 8.0, "max_green": 45.0, "yellow": 3.0},
                ],
            }
        },
        "stats": {"nodes": 5, "edges": 8, "junctions": 1, "signals": 1},
    }


def point_along(edge: dict, pos: float) -> tuple[float, float, float]:
    poly = edge["polyline"]
    pos = max(0.0, min(pos, edge["length"]))
    for index in range(1, len(poly)):
        if poly[index]["s"] >= pos:
            a, b = poly[index - 1], poly[index]
            span = max(1e-6, b["s"] - a["s"])
            t = (pos - a["s"]) / span
            lat = a["lat"] + (b["lat"] - a["lat"]) * t
            lon = a["lon"] + (b["lon"] - a["lon"]) * t
            heading = _heading(a["x"], a["y"], b["x"], b["y"])
            return lat, lon, heading
    last = poly[-1]
    return last["lat"], last["lon"], edge["heading"]


def save_network(network: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(network), encoding="utf-8")


def load_or_build(settings: Settings) -> dict:
    cached = network_path(settings)
    if cached.is_file():
        data = json.loads(cached.read_text(encoding="utf-8"))
        if data.get("edges"):
            return data
    try:
        osm = download_osm(settings)
        network = parse_osm(osm)
    except Exception:
        osm_existing = osm_path(settings)
        if osm_existing.is_file():
            try:
                network = parse_osm(osm_existing)
            except Exception:
                network = synthetic_network()
        else:
            network = synthetic_network()
    save_network(network, cached)
    return network
