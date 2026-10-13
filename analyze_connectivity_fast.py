import json
from pathlib import Path
import geopandas as gpd
import networkx as nx

base = Path("data/processed/features")
roads = gpd.read_file(base / "roads_suspect_exposure.geojson")
flagged = gpd.read_file(base / "roads_potentially_exposed.geojson")
out = Path("reports")
out.mkdir(exist_ok=True)

flagged_ids = set(flagged.index)
G = nx.Graph()
H = nx.Graph()

for i, geom in enumerate(roads.geometry):
    if geom is None or geom.is_empty:
        continue
    lines = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
    for line in lines:
        if line.geom_type != "LineString" or len(line.coords) < 2:
            continue
        a = tuple(round(c, 4) for c in line.coords[0][:2])
        b = tuple(round(c, 4) for c in line.coords[-1][:2])
        if a == b:
            continue
        G.add_edge(a, b)
        if i not in flagged_ids:
            H.add_edge(a, b)

def stats(graph):
    components = list(nx.connected_components(graph))
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "components": len(components),
        "largest_component_nodes": max(map(len, components), default=0),
    }

result = {
    "baseline": stats(G),
    "simulated_after_flagged_links_removed": stats(H),
    "warning": "Scenario only: flagged roads are not confirmed flooded or closed."
}
(out / "connectivity_summary.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
print("Saved reports/connectivity_summary.json")
