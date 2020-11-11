
from cGAN import *
from matplotlib import cm
import matplotlib as mpl
from train_cGAN import Config
from pipelines.utils import *
import torch
from glob import glob

model_weights = torch.load("saves/reg_nodis_model.epoch79.t7")["state"]
cGAN = ConditionalGAN(["LSTN"], ["NDVI", "NDBI", "NDWI"], 0, 0, True, True)
cGAN.load_state_dict(model_weights)

image_ids = glob("../data_source/LONDON_DATASET/*NDVI.tif")
image_ids = [".".join(a.split(".")[:-2]) for a in image_ids]

for image_id in tqdm(image_ids):

    imV = slice_middle(read_raster(f"{image_id}.NDVI.tif")[0][:, :, np.newaxis])
    imB = slice_middle(read_raster(f"{image_id}.NDBI.tif")[0][:, :, np.newaxis])
    imW = slice_middle(read_raster(f"{image_id}.NDWI.tif")[0][:, :, np.newaxis])

    if isinstance(imW, type(None)):
        continue

    imV[imV != imV] = 0
    imB[imB != imB] = 0
    imW[imW != imW] = 0
    image = np.dstack([imV, imB, imW])

    LSTN_real = read_raster(f"{image_id}.LSTN.tif")[0][:, :, np.newaxis]
    LSTN_real = slice_middle(LSTN_real).reshape(256, 256)

    LST_real = read_raster(f"{image_id}.LST.tif")[0][:, :, np.newaxis]
    LST_real = slice_middle(LST_real).reshape(256, 256)

    image_name = image_id.split("/")[-1]

    LSTN_pred = cGAN.generator(torch.tensor(image)).reshape(256, 256).detach().numpy()

    fig, axs = plt.subplots(2, 3, figsize=(30, 20))
    fig.suptitle(image_name, fontsize=50)

    axs[0, 0].imshow(imV, cmap="Greens")
    axs[0, 0].set_title("NDVI", fontsize=30)
    m = cm.ScalarMappable(cmap="Greens")
    m.set_clim(-1.0, 1.0)
    fig.colorbar(m, ax=axs[0, 0])

    axs[0, 1].imshow(imB, cmap="gray", vmin=-1.0, vmax=1.0)
    axs[0, 1].set_title("NDBI", fontsize=30)
    m = cm.ScalarMappable(cmap="gray")
    m.set_clim(-1.0, 1.0)
    fig.colorbar(m, ax=axs[0, 1])

    axs[0, 2].imshow(imW, cmap="Blues", vmin=-1.0, vmax=1.0)
    axs[0, 2].set_title("NDWI", fontsize=30)
    m = cm.ScalarMappable(cmap="Blues")
    m.set_clim(-1.0, 1.0)
    fig.colorbar(m, ax=axs[0, 2])

    axs[1, 0].imshow(LST_real, cmap="inferno")
    axs[1, 0].set_title("Real temperature (C)", fontsize=30)
    m = cm.ScalarMappable(cmap="inferno")
    m.set_clim(np.amin(LST_real), np.amax(LST_real))
    fig.colorbar(m, ax=axs[1, 0])

    axs[1, 1].imshow(LSTN_pred, cmap="magma")
    axs[1, 1].set_title("Predicted normalised temperature", fontsize=30)
    m = cm.ScalarMappable(cmap="magma")
    m.set_clim(np.amin(LSTN_pred), np.amax(LSTN_pred))
    fig.colorbar(m, ax=axs[1, 1])

    diff = LSTN_pred - LSTN_real
    axs[1, 2].imshow(diff, cmap="plasma")
    axs[1, 2].set_title("Predicted - real normalised LST", fontsize=30)
    m = cm.ScalarMappable(cmap="plasma")
    m.set_clim(np.amin(diff), np.amax(diff))
    fig.colorbar(m, ax=axs[1, 2])

    fig.savefig(f"results/NO_DIS_NO_SKIPS_ALL_ENGLAND_CITIES/{image_name}.RESULTS.png")

