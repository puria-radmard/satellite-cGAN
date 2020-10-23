import sys
from train_cGAN import Config
from opt import *
from cGAN import slice_middle
from pipelines.utils import read_raster

default_config = Config({'lr': 1e-4})

def optimised_NDVI_for_LSTN(model_path, image_root, config = default_config):

    map_optimiser = MapOptimiser(
        model_path=model_path,
        flat = ["NDBI", "NDWI"],
        sub = ["NDVI"],
        classes = ["LSTN"]
    )
    map_optimiser.cuda()
    loss_agent = NDVIToLSTNEvaluation()

    optimizer = torch.optim.Adam(map_optimiser.parameters(), lr=config.lr)

    NDBI_image = slice_middle(read_raster(f"{image_root}.NDBI.tif")[0][:,:,np.newaxis])
    NDWI_image = slice_middle(read_raster(f"{image_root}.NDBI.tif")[0][:,:,np.newaxis])
    LSTN_gt = read_raster(f"{image_root}.LSTN.tif")[0]

    map_optimiser.init_image(
        {
            "NDBI": NDBI_image,
            "NDWI": NDWI_image
        }
    )

    for _ in tqdm(range(5)):

        LSTN_map = map_optimiser()
        loss = loss_agent(
            LSTN_map = LSTN_map,
            NDVI_map = map_optimiser.sub_image,
            original_LSTN_map = LSTN_gt
        )
        loss.backward()
        optimizer.step()

if __name__ == '__main__':

    optimised_NDVI_for_LSTN(*sys.argv[1:])
