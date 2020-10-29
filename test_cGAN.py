from cGAN import *
from train_cGAN import Config
from pipelines.utils import *
import torch
from glob import glob

model_weights = torch.load("saves/reg_no_sig_model.epoch79.t7")["state"]
cGAN = ConditionalGAN(["LSTN"], ["NDVI", "NDBI", "NDWI"], 0, 0)
cGAN.load_state_dict(model_weights)

image_ids = glob('../data_source/LONDON_DATASET/*NDVI.tif')
image_ids = [".".join(a.split(".")[:-2]) for a in image_ids]

for image_id in tqdm(image_ids):

    imV = read_raster(f'{image_id}.NDVI.tif')[0]
    imB = read_raster(f'{image_id}.NDBI.tif')[0]
    imW = read_raster(f'{image_id}.NDWI.tif')[0]

    image = np.dstack([imV, imB, imW])
    image[image!=image]=0

    mix, miy = [int(m / 2) for m in image.shape[:2]]
    s = int(256 / 2)
    image = image[mix - s : mix + s, miy - s : miy + s]

    save_image = image - np.nanmin(image)
    save_image = save_image / np.nanmax(save_image)

    LSTN_real = read_raster(f'{image_id}.LSTN.tif')[0][:,:,np.newaxis]
    LSTN_real = slice_middle(LSTN_real)
    # LSTN_real -= np.nanmin(LSTN_real)
    # LSTN_real /= np.nanmax(LSTN_real)    

    if isinstance(LSTN_real, type(None)):
        continue

    image_name = image_id.split("/")[-1]
    
    LSTN_pred = cGAN.generator(torch.tensor(image)).reshape(256, 256).detach()

    plt.imsave(f"results/LONDON_ONLY/{image_name}.LSTN_PRED.png", LSTN_pred)

    plt.imsave(f"results/LONDON_ONLY/{image_name}.LSTN_L1.png", LSTN_real.reshape(256, 256) - LSTN_pred.numpy())
    plt.imsave(f"results/LONDON_ONLY/{image_name}.INDEX_MAP.png", save_image)
