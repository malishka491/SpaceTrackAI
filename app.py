import json
from pathlib import Path

import folium
import geopandas as gpd
import rasterio
import streamlit as st
from rasterio.features import shapes
from rasterio.windows import from_bounds
from shapely.geometry import shape, box
from streamlit_folium import st_folium

st.set_page_config(page_title="SpaceTrack AI", page_icon="🌊", layout="wide")
st.title("🌊 SpaceTrack AI")
st.caption("Satellite-Based Flood Damage Intelligence")

base = Path("data/processed")
road_file = base / "features/roads.geojson"
exposure_file = base / "features/roads_suspect_exposure.geojson"
flagged_file = base / "features/roads_potentially_exposed.geojson"
mask_file = base / "features/flood_suspect_mask.tif"

required = [road_file, exposure_file, flagged_file, mask_file]
missing = [str(p) for p in required if not p.exists()]
if missing:
    st.error("Missing output files: " + ", ".join(missing))
    st.stop()

roads = gpd.read_file(road_file).to_crs("EPSG:4326")
exposure = gpd.read_file(exposure_file).to_crs("EPSG:4326")
flagged = gpd.read_file(flagged_file).to_crs("EPSG:4326")

with rasterio.open(mask_file) as src:
    mask = src.read(1)
    total_suspect = int(mask.sum())
    valid_count = int((mask > 0).sum())
    bounds = src.bounds
    raster_transform = src.transform

aoi = box(84.8, 27.7, 85.3, 28.1)
roads = roads[roads.intersects(aoi)].copy()
for col in roads.select_dtypes(include=["datetime", "datetimetz"]).columns:
    roads[col] = roads[col].astype(str)
exposure = exposure[exposure.intersects(aoi)]
flagged = flagged[flagged.intersects(aoi)].copy()
for col in flagged.select_dtypes(include=["datetime", "datetimetz"]).columns:
    flagged[col] = flagged[col].astype(str)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Road features", f"{len(roads):,}")
c2.metric("Roads flagged", f"{len(flagged):,}")
c3.metric("Suspect pixels", f"{total_suspect:,}")
c4.metric("Suspect share", "2.49%")

summary_path = Path("reports/connectivity_summary.json")
if summary_path.exists():
    import json
    connectivity = json.loads(summary_path.read_text())
    baseline = connectivity["baseline"]
    simulated = connectivity["simulated_after_flagged_links_removed"]

    st.subheader("Road Network Connectivity — Scenario Analysis")
    st.caption(
        "Simulation only: provisionally flagged roads are removed from the graph; "
        "this does not mean these roads are confirmed flooded or closed."
    )
    n1, n2, n3 = st.columns(3)
    n1.metric("Baseline components", f'{baseline["components"]:,}')
    n2.metric("Scenario components", f'{simulated["components"]:,}')
    n3.metric(
        "Largest component (nodes)",
        f'{baseline["largest_component_nodes"]:,} → '
        f'{simulated["largest_component_nodes"]:,}'
    )

st.warning(
    "Preliminary low-backscatter screening only. These pixels and roads are "
    "not confirmed flood detections or verified damaged roads. Calibration "
    "quality and comparison with a suitable pre-event image remain unverified."
)

st.date_input("Satellite acquisition date", value=None, key="flood_date")
st.text_input("Study area", value="Trishuli River Corridor, Nepal")

show_suspect = st.checkbox("Show satellite flood-suspect pixels", value=True)
show_flagged = st.checkbox("Show potentially exposed roads", value=True)
show_all_roads = st.checkbox("Show road network", value=True)

m = folium.Map(location=[27.9, 85.05], zoom_start=10, tiles="OpenStreetMap")

if show_suspect:
    # Render a reduced-resolution raster overlay for browser performance.
    from rasterio.enums import Resampling
    with rasterio.open(mask_file) as src:
        reduced = src.read(
            1,
            out_shape=(max(1, src.height // 8), max(1, src.width // 8)),
            resampling=Resampling.nearest,
        )
        b = src.bounds

    import numpy as np
    from PIL import Image

    rgba = np.zeros((*reduced.shape, 4), dtype=np.uint8)
    rgba[reduced > 0] = [220, 30, 30, 110]
    rgba = Image.fromarray(rgba, mode="RGBA")
    import base64
    from io import BytesIO

    buffer = BytesIO()
    rgba.save(buffer, format="PNG")
    image_url = "data:image/png;base64," + base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    folium.raster_layers.ImageOverlay(
        image=image_url,
        bounds=[[b.bottom, b.left], [b.top, b.right]],
        opacity=0.75,
        name="Flood-suspect screening",
        interactive=True,
        cross_origin=False,
        zindex=2,
    ).add_to(m)

if show_all_roads:
    folium.GeoJson(
        json.loads(roads.head(4000).to_json()),
        name="Road network (up to 4,000 features)",
        style_function=lambda f: {"color": "#2471A3", "weight": 1, "opacity": 0.65},
    ).add_to(m)

if show_flagged:
    folium.GeoJson(
        json.loads(flagged.to_json()),
        name="Potentially exposed roads",
        style_function=lambda f: {"color": "#FF8C00", "weight": 3, "opacity": 0.9},
        tooltip=folium.GeoJsonTooltip(
            fields=["suspect_share_pct"],
            aliases=["Suspect sample share (%)"],
        ),
    ).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)
st.subheader("Satellite screening and road exposure map")
st_folium(m, height=600, use_container_width=True)

st.subheader("Road exposure results")
if len(flagged):
    display_cols = [
        c for c in [
            "name", "highway", "ref",
            "suspect_share_pct", "osm_way_id"
        ] if c in flagged.columns
    ]
    st.dataframe(
        flagged[display_cols].head(100),
        use_container_width=True,
    )
else:
    st.info("No road features meet the current provisional screening threshold.")

st.caption(
    "Prototype for decision support. Do not use this map alone to determine "
    "road safety, flood extent, or rescue routes."
)
