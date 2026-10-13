import requests
import geopandas as gpd
from pathlib import Path

BASE = "https://api.ohsome.org/v1/elements/geometry"
OUT = Path("data/raw/osm")
OUT.mkdir(parents=True, exist_ok=True)

# Small initial study area around the Trishuli region.
# Refine this bounding box against the actual event corridor.
bbox = "84.8,27.7,85.3,28.1"
snapshot = "2026-07-27"

queries = {
    "roads": "highway=* and type:way",
    "buildings": "building=* and type:way",
}

for name, filter_value in queries.items():
    print(f"Downloading {name}...")

    response = requests.post(
        BASE,
        data={
            "bboxes": bbox,
            "time": snapshot,
            "filter": filter_value,
            "properties": "tags",
        },
        timeout=180,
    )
    response.raise_for_status()

    path = OUT / f"{name}.geojson"
    path.write_bytes(response.content)

    # Validate that the downloaded response is readable.
    data = gpd.read_file(path)
    print(f"{name}: {len(data)} features saved to {path}")

print("Pre-event OSM download and validation complete.")
