import json

from datetime import date

import base64, html, json
from datetime import datetime
from io import BytesIO
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import rasterio
import streamlit as st
from PIL import Image
from rasterio.enums import Resampling
from shapely.geometry import box
from streamlit_folium import st_folium

st.set_page_config(page_title="SpaceTrack AI", page_icon="🌊", layout="wide")

# Load study configuration
CONFIG_PATH = Path("config/study_config.json")

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    study_config = json.load(f)

aoi = study_config["aoi"]
map_center = [
    (aoi["south"] + aoi["north"]) / 2,
    (aoi["west"] + aoi["east"]) / 2,
]

st.title("🌊 SpaceTrack AI")
st.caption("Satellite-based flood-suspect screening and road-exposure scenario analysis")

base = Path("data/processed/features")
road_file = base / "roads.geojson"
exposure_file = base / "roads_suspect_exposure.geojson"
flagged_file = base / "roads_potentially_exposed.geojson"
mask_file = base / "flood_suspect_mask.tif"
change_file = base / "vv_change_20260828_20260909_db.tif"
filtered_roads_file = base / "roads_vv_change_screening_filtered.geojson"
summary_path = Path("reports/connectivity_summary.json")

missing = [str(p) for p in [road_file, exposure_file, flagged_file, mask_file] if not p.exists()]
if missing:
    st.error("Missing required processed files:")
    st.code("\\n".join(missing))
    st.info("Copy your existing processed files into these paths; see README.md.")
    st.stop()

roads = gpd.read_file(road_file).to_crs("EPSG:4326")
exposure = gpd.read_file(exposure_file).to_crs("EPSG:4326")
flagged = gpd.read_file(flagged_file).to_crs("EPSG:4326")
with rasterio.open(mask_file) as src:
    mask = src.read(1)
    bounds = src.bounds
    suspect_pixels = int(np.count_nonzero(mask > 0))
    suspect_share = 100 * suspect_pixels / mask.size if mask.size else 0


study_area = st.sidebar.text_input(
    "Study area",
    value=study_config["study_area"],
)

acq_date = st.sidebar.date_input(
    "Satellite acquisition date",
    value=date.fromisoformat(study_config["event_date"]),
)

# Current project's known AOI
aoi = box(84.8, 27.7, 85.3, 28.1)
roads = roads[roads.intersects(aoi)].copy()
flagged = flagged[flagged.intersects(aoi)].copy()
exposure = exposure[exposure.intersects(aoi)].copy()

c1,c2,c3,c4 = st.columns(4)
c1.metric("Road features", f"{len(roads):,}")
c2.metric("Potentially exposed roads", f"{len(flagged):,}")
c3.metric("Suspect pixels", f"{suspect_pixels:,}")
c4.metric("Suspect share of raster cells", f"{suspect_share:.2f}%")

st.warning("Preliminary low-backscatter screening only. Candidates are not confirmed flood detections or verified road damage. The Sentinel calibration warning and lack of a validated pre-event comparison remain unresolved. Do not use this dashboard alone for road safety or rescue routing.")

show_suspect = st.checkbox("Show existing suspect-pixel overlay", True)
show_change = st.checkbox("Show Sentinel-1 VV change (28 Aug to 9 Sep)", True)
show_unet = st.checkbox("Show U-Net predicted water (9 Sep)", True)
show_overlap = st.checkbox("Show unvalidated overlap candidates", True)
show_filtered = st.checkbox("Show filtered backscatter-change road candidates", True)
show_flagged = st.checkbox("Show existing potentially exposed roads", True)
show_roads = st.checkbox("Show road network (up to 4,000 features)", True)

change_candidates = None
if filtered_roads_file.exists():
    change_candidates = gpd.read_file(filtered_roads_file).to_crs("EPSG:4326")
    change_candidates = change_candidates[change_candidates.intersects(aoi)].copy()

if change_file.exists():
    with rasterio.open(change_file) as src:
        change_pixels = src.read(1, out_shape=(
            max(1, src.height // 8), max(1, src.width // 8)
        ), resampling=Resampling.nearest)
        change_bounds = src.bounds
else:
    change_pixels = None
m = folium.Map(
    location=map_center,
    zoom_start=10,
    tiles="OpenStreetMap",
    control_scale=True,
)

if show_suspect:
    with rasterio.open(mask_file) as src:
        reduced = src.read(1, out_shape=(max(1,src.height//8),max(1,src.width//8)), resampling=Resampling.nearest)
        b = src.bounds
    rgba = np.zeros((*reduced.shape,4),dtype=np.uint8)
    rgba[reduced > 0] = [220,30,30,120]
    buf=BytesIO()
    Image.fromarray(rgba,mode="RGBA").save(buf,format="PNG")

if show_suspect:
    with rasterio.open(mask_file) as src:
        reduced = src.read(
            1,
            out_shape=(max(1, src.height // 8), max(1, src.width // 8)),
            resampling=Resampling.nearest
        )
        b = src.bounds
        nodata = src.nodata

    rgba = np.zeros((*reduced.shape, 4), dtype=np.uint8)
    valid_mask = np.isfinite(reduced)

    if nodata is not None:
        valid_mask &= reduced != nodata

    rgba[valid_mask & (reduced > 0)] = [220, 30, 30, 120]

    buf = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    folium.raster_layers.ImageOverlay(
        image=url,
        bounds=[[b.bottom, b.left], [b.top, b.right]],
        opacity=0.75,
        name="Low-backscatter suspect mask",
        interactive=True,
        cross_origin=False,
        zindex=2
    ).add_to(m)

if show_change and change_pixels is not None:
    from matplotlib import colormaps
    from matplotlib.colors import Normalize

    valid_change = np.isfinite(change_pixels) & (change_pixels != -9999)

    rgba = np.zeros((*change_pixels.shape, 4), dtype=np.uint8)
    norm = Normalize(vmin=-5, vmax=5, clip=True)
    cmap = colormaps["RdBu_r"]

    rgba[valid_change] = (
        cmap(norm(change_pixels[valid_change])) * 255
    ).astype(np.uint8)

    rgba[..., 3] = np.where(valid_change, 180, 0).astype(np.uint8)

    buf = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    folium.raster_layers.ImageOverlay(
        image=url,
        bounds=[
            [change_bounds.bottom, change_bounds.left],
            [change_bounds.top, change_bounds.right]
        ],
        opacity=0.75,
        name="Sentinel-1 VV change (dB)",
        interactive=True,
        cross_origin=False,
        zindex=3
    ).add_to(m)


unet_file = Path("data/processed/features/trishuli_unet_water_probability_20260909.tif")

if show_unet and unet_file.exists():
    with rasterio.open(unet_file) as src:
        probability = src.read(
            1,
            out_shape=(max(1, src.height // 8), max(1, src.width // 8)),
            resampling=Resampling.average
        )
        bounds = src.bounds
        nodata = src.nodata

    valid = np.isfinite(probability)
    if nodata is not None:
        valid &= probability != nodata

    rgba = np.zeros((*probability.shape, 4), dtype=np.uint8)
    predicted = valid & (probability >= 0.5)
    rgba[predicted] = [255, 220, 0, 210]

    buf = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    folium.raster_layers.ImageOverlay(
        image=url,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=0.85,
        name="U-Net predicted water (not confirmed flood)",
        interactive=True,
        cross_origin=False,
        zindex=4
    ).add_to(m)
elif show_unet:
    st.warning("U-Net probability raster not found. Run the inference script first.")


overlap_file = Path("data/processed/features/trishuli_overlap_screening_20260909.tif")

if show_overlap and overlap_file.exists():
    with rasterio.open(overlap_file) as src:
        overlap = src.read(
            1,
            out_shape=(max(1, src.height // 8), max(1, src.width // 8)),
            resampling=Resampling.nearest
        )
        bounds = src.bounds
        nodata = src.nodata

    rgba = np.zeros((*overlap.shape, 4), dtype=np.uint8)
    rgba[overlap == 1] = [220, 0, 220, 220]
    rgba[overlap == nodata] = [0, 0, 0, 0]

    buf = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    folium.raster_layers.ImageOverlay(
        image=url,
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=0.9,
        name="Unvalidated flood-screening overlap",
        interactive=True,
        cross_origin=False,
        zindex=5
    ).add_to(m)
elif show_overlap:
    st.warning("Overlap raster not found. Create it before enabling this layer.")

if show_filtered and change_candidates is not None and not change_candidates.empty:
    fields = [c for c in [
        "osm_id", "highway", "valid_samples",
        "median_vv_change_db", "decrease_le_minus3db_pct"
    ] if c in change_candidates.columns]
    folium.GeoJson(
        json.loads(change_candidates.to_json()),
        name="Potential backscatter-change road candidates",
        style_function=lambda f: {
            "color": "#8E44AD", "weight": 3, "opacity": 0.9
        },
        tooltip=folium.GeoJsonTooltip(fields=fields) if fields else None
    ).add_to(m)

if show_roads and not roads.empty:
    fields = [c for c in ["name", "highway", "ref", "osm_way_id"] if c in roads.columns]
    road_display = roads.head(4000).copy()
    for col in road_display.columns:
        if col != road_display.geometry.name:
            road_display[col] = road_display[col].map(
                lambda v: v.isoformat() if hasattr(v, "isoformat")
                else (None if v is None else str(v) if not isinstance(v, (str, int, float, bool)) else v)
            )
    folium.GeoJson(
        json.loads(road_display.to_json()),
        name="Road network",
        style_function=lambda f: {"color": "#2471A3", "weight": 1, "opacity": 0.65},
        tooltip=folium.GeoJsonTooltip(fields=fields) if fields else None,
    ).add_to(m)
if show_flagged and not flagged.empty:
    fields=[c for c in ["name","highway","ref","suspect_share_pct"] if c in flagged.columns]
    folium.GeoJson(json.loads(flagged.to_json()),name="Potentially exposed roads",style_function=lambda f:{"color":"#FF8C00","weight":3,"opacity":.95},tooltip=folium.GeoJsonTooltip(fields=fields) if fields else None).add_to(m)

# --- Terrain layers: SRTM elevation and slope ---
import numpy as np
import rasterio
from rasterio.enums import Resampling
from matplotlib import colormaps
from matplotlib.colors import Normalize
from branca.colormap import LinearColormap

terrain_layers = [
    (
        "Terrain elevation (SRTM, metres)",
        "data/processed/terrain/trishuli_dem.tif",
        "elevation",
    ),
    (
        "Terrain slope (degrees)",
        "data/processed/terrain/trishuli_slope_degrees.tif",
        "slope",
    ),
    (
        "Low-slope screening (<= 5 degrees)",
        "data/processed/terrain/trishuli_low_slope_mask.tif",
        "low_slope",
    ),
]

for layer_name, raster_path, layer_type in terrain_layers:
    try:
        with rasterio.open(raster_path) as src:
            arr = src.read(
                1,
                out_shape=(
                    min(src.height, 700),
                    min(src.width, 700),
                ),
                resampling=Resampling.nearest,
            ).astype(np.float32)

            bounds = src.bounds
            nodata = src.nodata

        valid = np.isfinite(arr)
        if nodata is not None:
            valid &= arr != nodata
        valid &= arr != -9999

        rgba = np.zeros((*arr.shape, 4), dtype=np.uint8)

        if layer_type == "elevation":
            low, high = 300, 4200
            cmap = colormaps["terrain"]
            norm = Normalize(vmin=low, vmax=high, clip=True)
            rgba[valid] = (
                cmap(norm(arr[valid])) * 255
            ).astype(np.uint8)
            rgba[..., 3] = np.where(valid, 165, 0).astype(np.uint8)

        elif layer_type == "slope":
            cmap = colormaps["YlOrRd"]
            norm = Normalize(vmin=0, vmax=45, clip=True)
            rgba[valid] = (
                cmap(norm(arr[valid])) * 255
            ).astype(np.uint8)
            rgba[..., 3] = np.where(valid, 175, 0).astype(np.uint8)

        else:
            low_slope_pixels = valid & (arr == 1)
            rgba[low_slope_pixels] = [40, 160, 80, 180]

        folium.raster_layers.ImageOverlay(
            image=rgba,
            bounds=[
                [bounds.bottom, bounds.left],
                [bounds.top, bounds.right],
            ],
            name=layer_name,
            opacity=1,
            interactive=True,
            cross_origin=False,
            zindex=2,
        ).add_to(m)

    except Exception as exc:
        st.warning(f"Could not load {layer_name}: {exc}")

st.caption(
    "Terrain source: SRTM elevation data. "
    "Low slope is a terrain-screening indicator, not proof of flooding. "
    "Terrain layers are optional and can be toggled in the map controls."
)

folium.LayerControl(collapsed=False).add_to(m)
st.subheader("Satellite screening and road exposure map")
st_folium(m,height=600,use_container_width=True)

st.subheader("Road exposure results")
cols=[c for c in ["name","highway","ref","suspect_share_pct","osm_way_id"] if c in flagged.columns]
if len(flagged):
    st.dataframe(flagged[cols].head(500),use_container_width=True,hide_index=True)
    st.download_button("Download flagged roads (CSV)",flagged[cols].to_csv(index=False).encode(), "SpaceTrackAI_flagged_roads.csv","text/csv")
    st.download_button("Download flagged roads (GeoJSON)",flagged.to_json().encode(),"SpaceTrackAI_flagged_roads.geojson","application/geo+json")
else:
    st.info("No road features meet the current screening threshold.")

conn = {}
if summary_path.exists():
    try: conn=json.loads(summary_path.read_text())
    except Exception: st.warning("Connectivity summary could not be read.")
baseline=conn.get("baseline",{})
simulated=conn.get("after_flagged_edges_removed",conn.get("simulated_after_flagged_links_removed",{}))
if baseline and simulated:
    st.subheader("Road-network connectivity — scenario analysis")
    st.caption(conn.get("warning","Scenario only; flagged roads are not confirmed flooded or closed."))
    a,b,c=st.columns(3)
    a.metric("Baseline components",f'{baseline.get("components","N/A"):,}' if isinstance(baseline.get("components"),int) else "N/A")
    b.metric("Scenario components",f'{simulated.get("components","N/A"):,}' if isinstance(simulated.get("components"),int) else "N/A")
    n0,n1=baseline.get("largest_component_nodes"),simulated.get("largest_component_nodes")
    c.metric("Largest component nodes",f"{n0:,} → {n1:,}" if isinstance(n0,int) and isinstance(n1,int) else "N/A")
else:
    st.info("Connectivity summary unavailable; other dashboard functions remain available.")

st.subheader("Export situation report")
date_text=acq_date.isoformat() if acq_date else "Not specified"
table=flagged[cols].head(100).to_html(index=False,escape=True,border=0) if len(flagged) else "<p>No roads met the screening threshold.</p>"
conn_html="<p>No connectivity scenario available.</p>"
if baseline and simulated:
    conn_html=f"<ul><li>Baseline components: {baseline.get('components','N/A')}</li><li>Scenario components: {simulated.get('components','N/A')}</li><li>Largest component nodes: {baseline.get('largest_component_nodes','N/A')} → {simulated.get('largest_component_nodes','N/A')}</li></ul>"
report=f"""<!doctype html><html><head><meta charset="utf-8"><title>SpaceTrack AI Situation Report</title><style>body{{font-family:Arial;max-width:1000px;margin:30px auto;padding:0 20px;line-height:1.5;color:#1f2937}}h1,h2{{color:#164e63}}.warning{{background:#fff4d6;padding:14px;border-left:5px solid #d99b00}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{border:1px solid #ddd;padding:7px;text-align:left}}</style></head><body><h1>SpaceTrack AI — Flood-Suspect Screening Report</h1><p><b>Study area:</b> {html.escape(study_area)}<br><b>Acquisition date:</b> {html.escape(date_text)}<br><b>Generated:</b> {datetime.now().astimezone().isoformat(timespec="seconds")}</p><div class="warning"><b>Important:</b> Preliminary low-backscatter screening only. These candidates are not confirmed flood detections or verified road damage. Calibration quality and pre-event comparison remain unverified. Do not use alone for safety or rescue routing.</div><h2>Summary</h2><ul><li>Road features: {len(roads):,}</li><li>Potentially exposed roads: {len(flagged):,}</li><li>Suspect pixels: {suspect_pixels:,}</li><li>Suspect share of raster cells: {suspect_share:.2f}%</li></ul><h2>Connectivity scenario</h2>{conn_html}<h2>Flagged roads (up to 100)</h2>{table}<h2>Limitations</h2><ul><li>Road exposure is based on existing provisional screening output.</li><li>Connectivity graph does not model bridges, grade separation, turn restrictions or travel time.</li><li>Reliable flood change detection requires comparable pre-event imagery and independent validation.</li></ul></body></html>"""
st.download_button("Download Situation Report (HTML)",report.encode(),"SpaceTrackAI_Situation_Report.html","text/html",use_container_width=True)
st.caption(
    "Research prototype. Contains modified Copernicus Sentinel data 2026. "
    "© OpenStreetMap contributors. VV changes and road candidates are not "
    "confirmed flooding or verified damage; independent validation is required."
)
