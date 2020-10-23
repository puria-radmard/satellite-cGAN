from .opt import *
from ..pipelines.utils import read_raster

def optimised_NDVI_for_LSTN(model_path, image_root, config):

    map_optimiser = MapOptimiser(
        model_path=model_path,
        flat = ["NDBI", "NDWI"],
        sub = ["NDVI"],
        classes = ["LSTN"]
    )
    loss_agent = NDVIToLSTNEvaluation()

    optimizer = torch.optim.Adam(map_optimiser.parameters(), lr=config.lr)

    NDBI_image = read_raster(f"{image_root}.NDBI.tif")[0]
    NDWI_image = read_raster(f"{image_root}.NDBI.tif")[0]
    LSTN_gt = read_raster(f"{image_root}.LSTN.tif")[0]

    map_optimiser.init_image(
        {
            "NDBI": NDBI_image,
            "NWDI": NDWI_image
        }
    )

    for _ in tqdm(range(5)):

        LSTN_map = map_optimiser()
        loss = loss_agent(
            LSTN_map = LSTN_map,
            NDVI_map = map_optimiser.sub_image,
            original_LSTN_map = LSTN_gt
        )
        loss.backwards()
        optimizer.step()
