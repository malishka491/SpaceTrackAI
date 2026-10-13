import numpy as np
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from pathlib import Path
from shapely.geometry import box

base = Path("data/processed")
rdir = base / "snap/trishuli_20260909_calibrated.data"
out = base / "features"
out.mkdir(parents=True, exist_ok=True)

aoi = box(84.8, 27.7, 85.3, 28.1)
roads = gpd.read_file(out / "roads.geojson").to_crs("EPSG:4326")
roads = roads[roads.intersects(aoi)].copy()
roads["geometry"] = roads.geometry.intersection(aoi)
roads = roads[~roads.geometry.is_empty].copy()

with rasterio.open(rdir / "Sigma0_VV.img") as vv, \
     rasterio.open(rdir / "Sigma0_VH.img") as vh:

    win = from_bounds(84.8, 27.7, 85.3, 28.1, vv.transform)
    win = win.round_offsets().round_lengths()
    v = vv.read(1, window=win)
    h = vh.read(1, window=win)

    valid = np.isfinite(v) & np.isfinite(h) & (v > 0) & (h > 0)
    suspect = valid & (v < 0.02) & (h < 0.01)

    profile = vv.profile.copy()
    profile.update(
        driver="GTiff", height=v.shape[0], width=v.shape[1],
        transform=vv.window_transform(win), count=1,
        dtype="uint8", nodata=0, compress="deflate"
    )
    with rasterio.open(out / "flood_suspect_mask.tif", "w", **profile) as dst:
        dst.write(suspect.astype("uint8"), 1)

    print("Valid pixels:", int(valid.sum()))
    print("Suspect pixels:", int(suspect.sum()))
    print("Suspect share (%):", round(100 * suspect.sum() / max(valid.sum(), 1), 2))

    projected = roads.to_crs("EPSG:32645")
    shares = []

    for geom in projected.geometry:
        lines = list(geom.geoms) if geom.geom_type == "MultiLineString" else [geom]
        points = []
        for line in lines:
            n = max(2, int(line.length / 30) + 1)
            points.extend(line.interpolate(i / (n - 1), normalized=True) for i in range(n))

        if not points:
            shares.append(0.0)
            continue

        from rasterio.warp import transform
        lon, lat = transform(
            "EPSG:32645", "EPSG:4326",
            [p.x for p in points], [p.y for p in points]
        )
        pairs = zip(vv.sample(zip(lon, lat)), vh.sample(zip(lon, lat)))
        vals = [(float(a[0]), float(b[0])) for a, b in pairs
                if np.isfinite(a[0]) and np.isfinite(b[0]) and a[0] > 0 and b[0] > 0]
        hits = sum(a < 0.02 and b < 0.01 for a, b in vals)
        shares.append(100 * hits / max(len(vals), 1))

roads["suspect_share_pct"] = np.round(shares, 2)
roads.to_file(out / "roads_suspect_exposure.geojson", driver="GeoJSON")
roads[roads["suspect_share_pct"] >= 20].to_file(
    out / "roads_potentially_exposed.geojson", driver="GeoJSON"
)

print("Roads checked:", len(roads))
print("Roads flagged:", int((roads["suspect_share_pct"] >= 20).sum()))
print("Output files saved in:", out)
