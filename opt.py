import wandb
import torch
from torch import nn
from pipelines.utils import *
from typing import List, Dict
from cGAN import ConditionalGAN

class MapOptimiser(nn.Module):

    def __init__(self, model_path, flat: List[str], sub: str, classes = ["LSTN"]):

        super(MapOptimiser, self).__init__()

        self.cGAN = ConditionalGAN(
            classes=classes,
            channels= flat + [sub],
            dis_dropout=0, # Change this
            gen_dropout=0, # Change this
        )
        model_weights = torch.load(model_path)["state"]
        self.cGAN.load_state_dict(model_weights)
        for name, p in self.cGAN.named_parameters():
            p.requires_grad = False

        self.sub = sub
        self.flat = flat
        self.model_path = model_path

    def init_image(self, flat_images, init_image):

        images = []
        for band in self.flat:
            images.append(flat_images[band])
        self.flat_image = torch.Tensor(np.dstack(images)).cuda()
        self.sub_image = torch.autograd.Variable(
            torch.randn(256, 256, 1).cuda() + torch.tensor(init_image).cuda(),
            requires_grad=True
        ).double()

    def forward(self):

        input_image = torch.cat([self.sub_image, self.flat_image], -1)    # CHANGE THIS ASAP TO BE CONFIGURABLE
        LSTN_map = self.cGAN.generator(input_image)

        return LSTN_map

    def show_image(self):

        return torch.cat([self.sub_image, self.flat_image], -1)    # CHANGE THIS ASAP TO BE CONFIGURABLE


class NDVIToLSTNEvaluation(nn.Module):

    def __init__(self, NDVI_factor = 0.5):
        super(NDVIToLSTNEvaluation, self).__init__()
        self.NDVI_factor = NDVI_factor

    @staticmethod
    def extract_thresh(image, thres):
        return torch.clamp(image, thres, 1) - thres

    def forward(self, LSTN_map, NDVI_map, original_LSTN_map):

        vegetation = self.extract_thresh(NDVI_map, 0.2)
        vegetation_squared_sum = torch.sum(torch.mul(vegetation, vegetation))

        UHI = self.extract_thresh(LSTN_map, 1.0)
        UHI_squared_sum = torch.sum(torch.mul(UHI, UHI))
        
        UHI_baseline = self.extract_thresh(torch.tensor(original_LSTN_map), 1.0)
        UHI_squared_sum_baseline = torch.sum(torch.mul(UHI_baseline, UHI_baseline))
        import pdb; pdb.set_trace()

        wandb.log(
            {
                "UHI_diff": UHI_squared_sum - UHI_squared_sum_baseline,
                "Vegetation squared sum": vegetation_squared_sum
            }
        )

        squared_loss = UHI_squared_sum - UHI_squared_sum_baseline\
              + self.NDVI_factor * vegetation_squared_sum

        return squared_loss

# Penalise vegetation cross with water & buildings
# Penalise area of UHI
# Apply discriminator to it
