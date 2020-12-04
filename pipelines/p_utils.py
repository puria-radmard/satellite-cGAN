import json
import pygeohash
import pyproj
import rasterio
import logging
import os
from glob import glob
from itertools import groupby
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
import ee
from tqdm import tqdm


def group_bands(root: str, bands: List[str]) -> List[Dict[str, str]]:
    """
    Given a root folder (e.g. "/../../../LONDON_DATASET") and a list of bands, such as ["B2", "B3", "B4"], this
    returns a list of dicts like
    [
        {
            "B2": "/../../../LONDON_DATASET/img1.B2.tif",
            "B3": "/../../../LONDON_DATASET/img1.B3.tif",
            "B4": "/../../../LONDON_DATASET/img1.B4.tif"
        },
        {
            "B2": "/../../../LONDON_DATASET/img2.B2.tif",
            "B3": "/../../../LONDON_DATASET/img2.B3.tif",
            "B4": "/../../../LONDON_DATASET/img2.B4.tif"
        },
        ...
    ]
    """

    imlist = glob(os.path.join(root, "*.tif"))
    logging.warning(f"Found {len(imlist)} images in total")

    projection = lambda x: x.split(".")[-2]

    im_sorted = sorted(imlist, key=projection)
    im_grouped = [list(it) for k, it in groupby(im_sorted, projection)]
    im_grouped = [sorted(i) for i in im_grouped if projection(i[0]) in bands]

    band_sorter = lambda x: bands.index(x.split(".")[-2])
    groups = [sorted(list(g), key=band_sorter) for g in zip(*im_grouped)]

    # Missing/duplicate logic
    missing_dict = {b: 0 for b in bands}
    for g in groups:
        group_bands_list = [a.split(".")[-2] for a in g]
        for band in bands:
            if group_bands_list.count(band) == 0:
                missing_dict[band] += 1

    for band, missing_count in missing_dict.items():
        if missing_count:
            logging.warning(f"Band {band} is missing from {missing_count} image groups")

    logging.warning(f"Found {len(groups)} groups in root directory")

    groups = [{bands[j]: group[j] for j in range(len(group))} for group in groups]

    return groups


def slice_middle(image, size=256, remove_nan=True):
    mix, miy = [int(m / 2) for m in image.shape[:2]]
    s = int(size / 2)
    sliced_image = image[mix - s : mix + s, miy - s : miy + s]
    if remove_nan:
        sliced_image[sliced_image != sliced_image] = 0.0
    if sliced_image.shape != (size, size, 1):
        return None
    return sliced_image


def purge_groups(groups, target_band="B3"):
    """
    Removes any groups where the raw data has NaN edges due to satellite projections.
    We use the raw band as a target as we can remove NaNs during processing.
    
    (This started as a quick fix on 11/11/2020)
    """

    outgroups = []

    for group in groups:

        target_path = get_property_path(group, prop_name=target_band)
        image = slice_middle(
            read_raster(target_path, remove_zero=True)[0][:, :, np.newaxis],
            remove_nan=False,
        )
        if type(image) == type(None):
            continue
        elif any(image[0] != image[0]):  # Any NaNs
            continue
        else:
            outgroups.append(group)

    print(f"Removed {len(groups)-len(outgroups)} groups for having NaN padding")

    return outgroups


def save_calculated_raster(raster_meta: Dict, path: str, image: np.ndarray):
    """Save image as a tif with metadata"""

    image = image[0]
    try:
        raster_meta["dtype"] = "float32"
        with rasterio.open(path, "w", **raster_meta) as dst:
            dst.write_band(1, image)
    except ValueError:
        raster_meta["dtype"] = "float64"
        with rasterio.open(path, "w", **raster_meta) as dst:
            dst.write_band(1, image)


def get_property_path(group: Dict[str, str], prop_name: str):
    """
    Given a group, as in an element of a list produced by group_bands, and a band name, this returns the new
    filename for that band image.
    """
    group_name = list(group.values())[0].split(".")[-3].split("/")[-1]
    prop_path = os.path.join(
        "/".join(list(group.values())[0].split("/")[:-1]),
        f"{group_name}.{prop_name}.tif",
    )
    return prop_path


def read_raster(path: str, remove_zero=False):
    """
    Read raster by path. remove_zero = True replaces 0 values with np.nan
    """

    with rasterio.open(path) as src1:
        raster_meta = src1.meta
        raster = src1.read(1)
        if remove_zero:
            raster = np.where(raster == 0, np.nan, raster)
    return raster, raster_meta


def visualise_bands(root, image_id, bands, show=False, viz_factors=[1, 1, 1]):
    "Unused"

    if len(bands) != 1 and len(bands) != 3:
        raise ValueError(
            f"visualise_bands requires 1 or 3 bands to visualise, not {len(bands)}"
        )

    band_images = []
    band_originals = []
    image_max = 1

    for i, band in enumerate(bands):
        path = f"{root}/{image_id}.{band}.tif"
        opened_raster, _ = read_raster(path)
        band_image = opened_raster * viz_factors[i]
        band_image -= np.amin(np.where(np.isnan(band_image), np.inf, band_image))
        band_images.append(band_image)
        band_originals.append(opened_raster)

    final_image = np.dstack(band_images)
    max_val = np.amax(np.where(np.isnan(final_image), -np.inf, final_image))
    final_image = final_image / max_val

    if len(bands) == 1:
        final_image = final_image[:, :, 0]

    if show:
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))

        axs[0].hist(
            [band_originals[j].flatten() for j in range(len(bands))],
            label=bands,
            alpha=0.5,
            bins=30,
        )

        axs[1].imshow(final_image)
        axs[0].legend()

    else:
        pass

    return final_image


def rasterise_bands(root, bands):
    "Unused"

    groups = group_bands(root, bands)

    for group in tqdm(groups):
        band_group_name = list(group.values())[0].split("/")[-1].split(".")[0]

    with rasterio.open(list(group.values())[0]) as src0:
        meta = src0.metam
        meta.update(count=len(bands))

    stack_path = get_property_path(group, "MAIN")

    with rasterio.open(stack_path, "w", **meta) as dst:
        for id, layer in enumerate(group.values(), start=1):

            if layer.split(".")[-2] not in bands:
                continue

            with rasterio.open(layer) as src1:
                fa = src1.read(1)
                dst.write_band(id, fa)


def get_metadata(group: Dict[str, str]):
    """
    Given a group produced by group_bands, this gets the metadata from the same folder
    """

    # TODO: Standardise this in a function
    group_name = list(group.values())[0].split(".")[-3].split("/")[-1]
    metadata_json_path = os.path.join(
        "/".join(list(group.values())[0].split("/")[:-1]), f"{group_name}.METADATA.json"
    )

    with open(metadata_json_path, "r") as jfile:
        metadata = json.load(jfile)
        return metadata


def get_area_bbox(geojson):
    "Unused for now"

    coords = [np.array(a["geometry"]["coordinates"][0]) for a in geojson["features"]]
    Xs = np.concatenate([a[:, 0] for a in coords])
    Ys = np.concatenate([a[:, 1] for a in coords])
    bounds = {"top": max(Ys), "bottom": min(Ys), "right": max(Xs), "left": min(Xs)}

    coords = [
        [bounds["left"], bounds["top"]],
        [bounds["right"], bounds["top"]],
        [bounds["right"], bounds["bottom"]],
        [bounds["left"], bounds["bottom"]],
        [bounds["left"], bounds["top"]],
    ]
    return coords


def get_geohash(geometry):
    """
    Given a gee geometry, this returns the centroid as a geohash
    """
    cs = np.array(geometry.coordinates().getInfo()).reshape([-1, 2])
    cent = np.mean(cs, axis=0)
    return pygeohash.encode(cent[1], cent[0])


def split_into_grid(geometry, end_code: str):
    """
    Needs parameterisation for image size, and fixing for exact sizes
    Given a geometry and a EPSG code, this splits geometry into sqaures of roughly 256 pixels for Landsat 8
    (i.e. 256*8 m square sides)
    """

    # Set up projections
    p_ll = pyproj.Proj(init="epsg:4326")
    p_mt = pyproj.Proj(init=end_code)  # metric; same as EPSG:900913

    # Create corners of rectangle to be transformed to a grid
    geometry = np.array(geometry).reshape(-1, 2)
    cs = geometry
    ne = max(cs, key=lambda x: (x[0], x[1]))
    sw = min(cs, key=lambda x: (x[0], x[1]))

    stepsize = 256 * 30

    # Project corners to target projection
    transformed_sw = pyproj.transform(
        p_ll, p_mt, sw[0], sw[1]
    )  # Transform NW point to epsg:3857
    transformed_ne = pyproj.transform(p_ll, p_mt, ne[0], ne[1])  # .. same for SE

    grid = np.zeros(
        [
            int((transformed_ne[0] - transformed_sw[0]) // stepsize + 2),
            int((transformed_ne[1] - transformed_sw[1]) // stepsize + 2),
            2,
        ]
    )

    # Iterate over 2D area
    squares = []
    x = transformed_sw[0]
    i, j = 0, 0
    while x < transformed_ne[0] + stepsize:

        y = transformed_sw[1]
        while y < transformed_ne[1] + stepsize:
            llcords = pyproj.transform(p_mt, p_ll, x, y)
            p = list(llcords)
            # print(i, j)
            grid[j][i] = p

            if i > 0 and j > 0:
                square = np.stack(
                    [
                        grid[j][i],
                        grid[j - 1][i],
                        grid[j - 1][i - 1],
                        grid[j][i - 1],
                        grid[j][i],
                    ]
                )
                squares.append(square)

            y += stepsize
            i += 1

        x += stepsize
        j += 1
        i = 0

    return [ee.Geometry.Polygon(s.tolist()) for s in squares]
