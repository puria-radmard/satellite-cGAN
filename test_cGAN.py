from cGAN import *
from train_cGAN import Config
from pipelines.utils import *
import torch

model_weights = torch.load("saves/reg_model.epoch79.t7")["state"]
cGAN = ConditionalGAN(["LSTN"], ["NDVI", "NDBI", "NDWI"], 0, 0)
cGAN.load_state_dict(model_weights)

image_id = '../data_source/LONDON_DATASET/gcpev16xwfth--LONDON--2020-06-25--10-58-18'

imV = read_raster(f'{image_id}.NDVI.tif')[0]
imB = read_raster(f'{image_id}.NDBI.tif')[0]
imW = read_raster(f'{image_id}.NDWI.tif')[0]

image = np.dstack([imV, imB, imW])
mix, miy = [int(m / 2) for m in image.shape[:2]]
s = int(256 / 2)
image = image[mix - s : mix + s, miy - s : miy + s]
plt.imsave("example_index_map.png", image)

LSTN_real = read_raster(f'{image_id}.LSTN.tif')[0][:,:,np.newaxis]
plt.imsave("example_LSTN_real.png", (LSTN_real[:,:,0] + 1) / 2)

LSTN_pred = cGAN.generator(torch.tensor(image)).reshape(256, 256)
import pdb; pdb.set_trace()
plt.imsave("example_LSTN_with_sigmoid.png", LSTN_pred.detach())
