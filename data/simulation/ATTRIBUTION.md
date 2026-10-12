# Road-network data attribution

The Kalma Chowk extract is obtained from [OpenStreetMap](https://www.openstreetmap.org/) via the official OSM API (`/api/0.6/map`).

© OpenStreetMap contributors. The data is available under the [Open Database License (ODbL) 1.0](https://opendatacommons.org/licenses/odbl/).

Bounding box used for the default extract (WGS84):

- south: 31.4990
- west: 74.3260
- north: 31.5080
- east: 74.3370

Area: Kalma Chowk / Liberty Chowk, Lahore, Punjab, Pakistan.

Re-download and rebuild:

```powershell
python scripts/import_lahore_network.py
```

If Eclipse SUMO `netconvert` is available, the same script also writes a SUMO `.net.xml`. SUMO itself is licensed under the Eclipse Public License 2.0.
