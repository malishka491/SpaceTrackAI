import os
import rasterio
from rasterio.mask import mask
from shapely.geometry import box, mapping

src_path = "data/processed/georeferenced/sentinel1_vv_georeferenced.tif"
out_path = "data/processed/georeferenced/trishuli_vv_clip.tif"

# Approximate Trishuli study-area bounds
min_lon, min_lat, max_lon, max_lat = 84.8, 27.7, 85.3, 28.1
study_area = box(min_lon, min_lat, max_lon, max_lat)

os.makedirs(os.path.dirname(out_path), exist_ok=True)

with rasterio.open(src_path) as src:
    print("Source CRS:", src.crs)
    print("Source bounds:", src.bounds)

    if src.crs is None or src.crs.to_string() != "EPSG:4326":
        raise SystemExit("Unexpected CRS. Correct georeferencing first.")

    bounds = src.bounds

    if (
        bounds.right <= min_lon
        or bounds.left >= max_lon
        or bounds.top <= min_lat
        or bounds.bottom >= max_lat
    ):
        raise SystemExit("The source raster does not intersect the study area.")

    image, transform = mask(
        src,
        [mapping(study_area)],
        crop=True,
        filled=False
    )

    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        height=image.shape[1],
        width=image.shape[2],
        transform=transform,
        compress="deflate",
        nodata=0
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(image.filled(0))

print("Clipped raster saved:", out_path)
