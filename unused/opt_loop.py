import sys
import wandb
from opt import *
from train_cGAN import Config
from cGAN import *
from pipelines.p_utils import read_raster


def optimised_NDVI_for_LSTN(config):

    map_optimiser = MapOptimiser(
        model_path=config.model_path,
        flat=["NDBI", "NDWI"],
        sub="NDVI",
        classes=["LSTN"],
    )
    map_optimiser.cuda()

    NDBI_image = slice_middle(
        read_raster(f"{config.image_root}.NDBI.tif")[0][:, :, np.newaxis]
    )
    NDWI_image = slice_middle(
        read_raster(f"{config.image_root}.NDBI.tif")[0][:, :, np.newaxis]
    )
    NDVI_image = slice_middle(
        read_raster(f"{config.image_root}.NDVI.tif")[0][:, :, np.newaxis]
    )

    LSTN_gt = slice_middle(
        read_raster(f"{config.image_root}.LSTN.tif")[0][:, :, np.newaxis]
    )

    wandb.log(
        {
            "NDBI image": [wandb.Image(NDBI_image, caption=f"NDBI image")],
            "NDWI image": [wandb.Image(NDWI_image, caption=f"NDWI image")],
            "Ground Truth NDVI image": [
                wandb.Image(NDVI_image, caption=f"Ground Truth temp image")
            ],
            "Ground Truth LSTN image": [
                wandb.Image(LSTN_gt, caption=f"Ground Truth temp image")
            ],
        }
    )

    map_optimiser.init_image(
        {"NDBI": NDBI_image, "NDWI": NDWI_image, "NDVI": NDVI_image}
    )

    optimizer = torch.optim.Adam([map_optimiser.sub_image], lr=config.lr)

    for round_num in tqdm(range(config.epochs)):

        LSTN_map = map_optimiser.prediction()

        ploss = map_optimiser.evaluation_loss(
            LSTN_map=LSTN_map,
            NDVI_map=map_optimiser.sub_image,
            original_LSTN_map=LSTN_gt,
        )
        ploss.backward()

        sloss = map_optimiser.structural_loss()
        sloss.backward()

        optimizer.step()

        wandb.log(
            {
                "Round NDVI image": [
                    wandb.Image(
                        map_optimiser.sub_image.detach().cpu().numpy(),
                        caption=f"Round {round_num}",
                    )
                ]
            }
        )
        wandb.log(
            {
                "Round LSTN image": [
                    wandb.Image(
                        LSTN_map.detach().cpu().numpy(), caption=f"Round {round_num}"
                    )
                ]
            }
        )


if __name__ == "__main__":

    config = {
        "lr": 1e-4,
        "model_path": sys.argv[1],
        "image_root": sys.argv[2],
        "epochs": 500,
    }

    wandb.init(project="satellite-opt", config=config)
    config = wandb.config
    # config = Config(config)
    optimised_NDVI_for_LSTN(config)
