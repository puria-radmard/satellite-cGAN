from glob import glob
from itertools import groupby
from typing import Dict
import os

def group_cities_by_time(root: str, band: str) -> Dict[str, Dict[str, str]]:
    """
    This is similar to group_bands, except returns a mapping of images taken at the same time in the root
    """

    imlist = glob(os.path.join(root, f"*.{band}.tif"))

    city_projection = lambda x: x.split("/")[-1].split("--")[1]
    time_projection1 = lambda x: "--".join(
        x.split("/")[-1].split(".")[0].split("--")[-2:]
    )
    # Remove second
    time_projection = lambda x: "-".join(time_projection1(x).split("-")[:6])

    im_sorted = sorted(imlist, key=city_projection)
    city_groups = [list(it) for k, it in groupby(im_sorted, city_projection)]

    out_dict = {}

    for city_group in city_groups:
        city_name = city_projection(city_group[0])
        out_dict[city_name] = {}
        city_group.sort(key=time_projection)
        time_groups = [list(it) for k, it in groupby(city_group, time_projection)]

        for time_group in time_groups:
            time_string = time_projection(time_group[0])
            out_dict[city_name][time_string] = time_group

    return out_dict