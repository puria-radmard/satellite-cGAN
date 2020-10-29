from cGAN import *
from matplotlib import cm
import matplotlib as mpl
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

    imV = slice_middle(read_raster(f'{image_id}.NDVI.tif')[0][:,:,np.newaxis])
    imB = slice_middle(read_raster(f'{image_id}.NDBI.tif')[0][:,:,np.newaxis])
    imW = slice_middle(read_raster(f'{image_id}.NDWI.tif')[0][:,:,np.newaxis])

    if isinstance(imW, type(None)):
        continue

    imV[imV!=imV]=0
    imB[imB!=imB]=0
    imW[imW!=imW]=0
    image = np.dstack([imV, imB, imW])

    LSTN_real = read_raster(f'{image_id}.LSTN.tif')[0][:,:,np.newaxis]
    LSTN_real = slice_middle(LSTN_real).reshape(256, 256)

    LST_real = read_raster(f'{image_id}.LST.tif')[0][:,:,np.newaxis]
    LST_real = slice_middle(LST_real).reshape(256, 256)

    image_name = image_id.split("/")[-1]
    
    LSTN_pred = cGAN.generator(torch.tensor(image)).reshape(256, 256).detach().numpy()

    fig, axs = plt.subplots(2, 3, figsize = (30, 20))
    fig.suptitle(image_name, fontsize=50)

    axs[0,0].imshow(imV, cmap='Greens')
    axs[0,0].set_title("NDVI", fontsize = 30)
    fig.colorbar(cm.ScalarMappable(cmap='Greens'), ax=axs[0,0], norm=mpl.colors.Normalize(vmin=-1.0, vmax=1.0))

    axs[0,1].imshow(imB, cmap='gray', vmin=-1.0, vmax=1.0)
    axs[0,1].set_title("NDBI", fontsize = 30)
    fig.colorbar(cm.ScalarMappable(cmap='gray'), ax=axs[0,1], norm=mpl.colors.Normalize(vmin=-1.0, vmax=1.0))

    axs[0,2].imshow(imW, cmap='Blues', vmin=-1.0, vmax=1.0)
    axs[0,2].set_title("NDWI", fontsize = 30)
    fig.colorbar(cm.ScalarMappable(cmap='Blues'), ax=axs[0,2], norm=mpl.colors.Normalize(vmin=-1.0, vmax=1.0))

    axs[1,0].imshow(LST_real, cmap='inferno')
    axs[1,0].set_title("Real temperature (C)", fontsize = 30)
    fig.colorbar(cm.ScalarMappable(cmap='inferno'), ax=axs[1,0], norm=mpl.colors.Normalize(vmin=np.amin(LST_real), vmax=np.amax(LST_real)))

    axs[1,1].imshow(LSTN_pred, cmap='magma')  
    axs[1,1].set_title("Predicted normalised temperature", fontsize = 30)
    fig.colorbar(cm.ScalarMappable(cmap='magma'), ax=axs[1,1], norm=mpl.colors.Normalize(vmin=np.amin(LSTN_pred), vmax=np.amax(LSTN_pred)))

    diff = LSTN_pred-LSTN_real
    axs[1,2].imshow(diff, cmap='plasma')
    axs[1,2].set_title("Predicted - real normalised LST", fontsize = 30)
    fig.colorbar(cm.ScalarMappable(cmap='plasma'), ax=axs[1,2], norm=mpl.colors.Normalize(vmin=np.amin(diff), vmax=np.amax(diff)))

    fig.savefig(f"results/LONDON_ONLY/{image_name}.RESULTS.png")
