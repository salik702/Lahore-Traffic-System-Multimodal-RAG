"""Optional Eclipse SUMO / TraCI engine. Never reports SUMO as active unless TraCI actually starts."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

from app.simulation.engine import Engine
from app.simulation.sumo_detect import detect_sumo


def write_plain_network(network: dict, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    nodes_path = folder / "nodes.xml"
    edges_path = folder / "edges.xml"
    node_xml = ['<?xml version="1.0" encoding="UTF-8"?>', "<nodes>"]
    for node in network["nodes"].values():
        junc = network["junctions"].get(node["id"], {})
        ntype = "traffic_light" if junc.get("controllable") else "priority"
        node_xml.append(f'<node id="{escape(node["id"])}" x="{node["x"]:.2f}" y="{node["y"]:.2f}" type="{ntype}"/>')
    node_xml.append("</nodes>")
    nodes_path.write_text("\n".join(node_xml), encoding="utf-8")
    edge_xml = ['<?xml version="1.0" encoding="UTF-8"?>', "<edges>"]
    for edge in network["edges"].values():
        edge_xml.append(
            f'<edge id="{escape(edge["id"])}" from="{escape(edge["from"])}" to="{escape(edge["to"])}" numLanes="{max(1, edge["lanes"])}" speed="{edge["speed"]:.2f}"/>'
        )
    edge_xml.append("</edges>")
    edges_path.write_text("\n".join(edge_xml), encoding="utf-8")
    return nodes_path


def convert_with_netconvert(settings, network: dict) -> Path | None:
    info = detect_sumo(settings)
    if not info.get("netconvert"):
        return None
    folder = settings.project_root / "data" / "simulation" / "kalma_chowk" / "sumo"
    write_plain_network(network, folder)
    net_path = folder / "kalma.net.xml"
    cmd = [
        info["netconvert"],
        "--node-files", str(folder / "nodes.xml"),
        "--edge-files", str(folder / "edges.xml"),
        "--output-file", str(net_path),
        "--no-turnarounds",
        "--tls.guess",
        "--geometry.remove", "false",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    except Exception:
        return None
    if proc.returncode != 0 or not net_path.is_file():
        return None
    return net_path


class SumoLiveEngine:
    """TraCI wrapper. Falls back by raising; the manager then uses the native engine."""

    def __init__(self, settings, network: dict, **kwargs):
        self.settings = settings
        self.native = Engine(network=network, **kwargs)
        self.network = network
        self.traci = None
        self.connected = False
        self.info = detect_sumo(settings)
        self.net_path = convert_with_netconvert(settings, network)

    def start_traci(self) -> bool:
        if not self.info.get("installed") or not self.net_path:
            return False
        try:
            import traci
        except Exception:
            if self.info.get("tools"):
                sys.path.append(self.info["tools"])
                try:
                    import traci  # type: ignore
                except Exception:
                    return False
            else:
                return False
        os.environ["SUMO_HOME"] = self.info["sumo_home"]
        folder = self.net_path.parent
        cfg = folder / "live.sumocfg"
        cfg.write_text(
            f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
  <input><net-file value="{self.net_path.name}"/></input>
  <time><begin value="0"/><step-length value="0.5"/></time>
  <processing><collision.action value="warn"/></processing>
</configuration>
""",
            encoding="utf-8",
        )
        import socket
        try:
            with socket.socket() as sock:
                sock.bind(("", 0))
                port = sock.getsockname()[1]
            traci.start([self.info["sumo_binary"], "-c", str(cfg), "--no-step-log", "true", "--quit-on-end", "true"], port=port, label=f"lti-{os.getpid()}-{port}")
            self.traci = traci
            self.connected = True
            return True
        except Exception:
            self.connected = False
            return False

    def close(self) -> None:
        if self.connected and self.traci is not None:
            try:
                self.traci.close(False)
            except Exception:
                pass
        self.connected = False
