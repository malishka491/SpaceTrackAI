import json
import geopandas as gpd
from shapely.geometry import LineString
from pathlib import Path

source = Path("data/raw/osm/roads_test.json")
destination = Path("data/processed/roads.geojson")

with source.open() as f:
    data = json.load(f)

records = []

for element in data.get("elements", []):
    geometry = element.get("geometry", [])
    if len(geometry) < 2:
        continue

    coordinates = [
        (point["lon"], point["lat"])
        for point in geometry
        if "lon" in point and "lat" in point
    ]

    if len(coordinates) < 2:
        continue

    tags = element.get("tags", {})
    records.append({
        "osm_id": element["id"],
        "highway": tags.get("highway", "unknown"),
        "name": tags.get("name", "Unnamed road"),
        "geometry": LineString(coordinates),
    })

if not records:
    raise RuntimeError("No usable road geometries found.")

gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
destination.parent.mkdir(parents=True, exist_ok=True)
gdf.to_file(destination, driver="GeoJSON")

print(f"Converted {len(gdf)} roads.")
print(f"Saved: {destination}")
print(gdf["highway"].value_counts().head(10))
