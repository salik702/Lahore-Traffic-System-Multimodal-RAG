"""Download/cache the Kalma Chowk OSM extract, export SUMO files, and validate them."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from app.core.config import get_settings
from app.simulation.osm_network import download_osm, network_path, parse_osm, save_network
from app.simulation.sumo_detect import detect_sumo
from app.simulation.sumo_engine import convert_with_netconvert


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-download", action="store_true", help="Refresh the cached OpenStreetMap extract")
    args = parser.parse_args()
    settings = get_settings()
    osm_file = download_osm(settings, force=args.force_download)
    network = parse_osm(osm_file)
    save_network(network, network_path(settings))
    print(f"OSM: {osm_file} ({osm_file.stat().st_size:,} bytes)")
    print(f"Network: {network['stats']} · source={network['source']} · license={network['license']}")

    sumo = detect_sumo(settings)
    if not sumo["installed"]:
        print(sumo["message"], file=sys.stderr)
        return 2
    net_file = convert_with_netconvert(settings, network)
    if not net_file:
        print("netconvert failed to produce a SUMO network", file=sys.stderr)
        return 3
    env = os.environ.copy()
    env["SUMO_HOME"] = sumo["sumo_home"]
    check = subprocess.run(
        [sumo["sumo_binary"], "-n", str(net_file), "--duration-log.disable", "--no-step-log", "true", "--end", "1"],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        check=False,
    )
    if check.returncode:
        print(check.stdout + check.stderr, file=sys.stderr)
        return check.returncode
    print(f"SUMO {sumo['version']} validated: {net_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
