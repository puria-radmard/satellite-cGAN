import sys
import wandb
from opt import *
from train_cGAN import Config
from cGAN import slice_middle
from pipelines.utils import read_raster

default_config = Config({'lr': 1e-4})

def optimised_NDVI_for_LSTN(config):

    map_optimiser = MapOptimiser(
        model_path=config.model_path,
        flat = ["NDBI", "NDWI"],
        sub = ["NDVI"],
        classes = ["LSTN"]
    )
    map_optimiser.cuda()
    loss_agent = NDVIToLSTNEvaluation()

    optimizer = torch.optim.Adam(map_optimiser.parameters(), lr=config.lr)

    NDBI_image = slice_middle(read_raster(f"{config.image_root}.NDBI.tif")[0][:,:,np.newaxis])
    NDWI_image = slice_middle(read_raster(f"{config.image_root}.NDBI.tif")[0][:,:,np.newaxis])
    LSTN_gt    = slice_middle(read_raster(f"{config.image_root}.LSTN.tif")[0][:,:,np.newaxis])

    wandb.log(
        {
            "NDBI image": [wandb.Image(NDBI_image, caption=f"NDBI image")],
            "NDWI image": [wandb.Image(NDWI_image, caption=f"NDWI image")],
            "Ground Truth temp image": [wandb.Image(LSTN_gt, caption=f"Ground Truth temp image")],
        }
    )

    map_optimiser.init_image(
        {
            "NDBI": NDBI_image,
            "NDWI": NDWI_image
        },
        LSTN_gt
    )

    for round_num in tqdm(range(config.epochs)):

        LSTN_map = map_optimiser()
        loss = loss_agent(
            LSTN_map = LSTN_map,
            NDVI_map = map_optimiser.sub_image,
            original_LSTN_map = LSTN_gt
        )
        loss.backward()
        optimizer.step()
        wandb.log({"Round image": [wandb.Image(map_optimiser.sub_image.detach().cpu().numpy(), caption=f"Round {round_num}")]})



if __name__ == '__main__':

    config = {
        "lr": 1e-4,
        "model_path": sys.argv[1],
        "image_root": sys.argv[2],
        "epochs": 500
    }
    wandb.init(project="satellite-cGAN", config=config)
    optimised_NDVI_for_LSTN(wandb.config)
