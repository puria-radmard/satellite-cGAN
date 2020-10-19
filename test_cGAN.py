from cGAN import *
from train_cGAN import Config
from pipelines.utils import *
import torch

model_weights = torch.load("saves/reg_model.epoch79.t7")["state"]
cGAN = ConditionalGAN(["LSTN"], ["NDVI", "NDBI", "NDWI"], 0, 0)
cGAN.load_state_dict(model_weights)

image_id = '../data_source/LONDON_DATASET/gcpgf4du7rsv--LONDON--2018-06-29--10-51-26'

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
plt.imsave("example_index_map.png", save_image)

LSTN_real = read_raster(f'{image_id}.LSTN.tif')[0][:,:,np.newaxis]
LSTN_real = slice_middle(LSTN_real)
LSTN_real -= np.nanmin(LSTN_real)
LSTN_real /= np.nanmax(LSTN_real)
plt.imsave("example_LSTN_real.png", (LSTN_real[:,:,0] + 1) / 2)

LSTN_pred = cGAN.generator(torch.tensor(image)).reshape(256, 256)
plt.imsave("example_LSTN_with_sigmoid.png", LSTN_pred.detach())
