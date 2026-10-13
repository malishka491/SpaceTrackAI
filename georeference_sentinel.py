import os
import glob
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.warp import calculate_default_transform, reproject, Resampling

base = "data/processed/sentinel1"
out_dir = "data/processed/georeferenced"
os.makedirs(out_dir, exist_ok=True)

tiffs = glob.glob(base + "/**/measurement/*vv*COG.tiff", recursive=True)
if not tiffs:
    tiffs = glob.glob(base + "/**/measurement/*vv*.tiff", recursive=True)

xmls = glob.glob(base + "/**/annotation/s1d-iw-grd-vv-*-cog.xml", recursive=True)

if not tiffs or not xmls:
    raise SystemExit(
        "Could not find VV TIFF or matching annotation XML. "
        "Check data/processed/sentinel1."
    )

src_path = tiffs[0]
xml_path = xmls[0]
print("Source TIFF:", src_path)
print("Geolocation XML:", xml_path)

root = ET.parse(xml_path).getroot()
gcps = []

for elem in root.iter():
    if elem.tag.split("}")[-1] != "geolocationGridPoint":
        continue

    values = {}
    for child in elem:
        tag = child.tag.split("}")[-1]
        if tag in {"line", "pixel", "latitude", "longitude"}:
            values[tag] = (child.text or "").strip()

    try:
        gcps.append(
            GroundControlPoint(
                row=float(values["line"]),
                col=float(values["pixel"]),
                x=float(values["longitude"]),
                y=float(values["latitude"]),
                z=0.0,
            )
        )
    except (KeyError, ValueError):
        continue

if len(gcps) < 6:
    raise SystemExit(f"Not enough valid GCPs: {len(gcps)}")

print("Control points:", len(gcps))

with rasterio.open(src_path) as src:
    # Estimate an output grid from the GCPs.
    dst_transform, dst_width, dst_height = calculate_default_transform(
        None,
        "EPSG:4326",
        src.width,
        src.height,
        gcps=gcps,
    )

    output = os.path.join(out_dir, "sentinel1_vv_georeferenced.tif")

    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        crs="EPSG:4326",
        transform=dst_transform,
        width=dst_width,
        height=dst_height,
        dtype="float32",
        nodata=0,
        compress="deflate",
        tiled=True,
    )

    with rasterio.open(output, "w", **profile) as dst:
        reproject(
            source=rasterio.band(src, 1),
            destination=rasterio.band(dst, 1),
            gcps=gcps,
            src_crs=None,
            dst_transform=dst_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.bilinear,
            dst_nodata=0,
        )

print("\nOutput:", output)

with rasterio.open(output) as check:
    print("CRS:", check.crs)
    print("Dimensions:", check.width, "x", check.height)
    print("Bounds:", check.bounds)
    print("Transform:", check.transform)

    b = check.bounds
    if not (70 < b.left < 100 and 70 < b.right < 100
            and 15 < b.bottom < 40 and 15 < b.top < 40):
        print(
            "\nWARNING: Bounds do not look like the expected Nepal region. "
            "Do not use this output on the dashboard until validated."
        )
