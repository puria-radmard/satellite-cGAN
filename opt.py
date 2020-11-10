import wandb
import torch
from torch import nn
from pipelines.utils import *
from typing import List, Dict
from cGAN import ConditionalGAN

class MapOptimiser(nn.Module):

    def __init__(self, model_path, flat: List[str], sub: str, classes = ["LSTN"], NDVI_factor = 0.5):

        super(MapOptimiser, self).__init__()

        self.Predictor = ConditionalGAN(
            classes=classes,
            channels= flat + [sub],
            dis_dropout=0, # Change this
            gen_dropout=0, # Change this
        )
        self.Predictor.eval()
        model_weights = torch.load(model_path)["state"]
        self.Predictor.load_state_dict(model_weights)
        for name, p in self.Predictor.named_parameters():
            p.requires_grad = False

        self.Predictor = self.Predictor.generator
        self.StructureLayers = list(self.Predictor._modules.items())[:5] # This may change

        self.sub = sub
        self.flat = flat
        self.model_path = model_path

        self.NDVI_factor = NDVI_factor

    def init_image(self, image_dict):

        images = []
        for band in self.flat:
            images.append(image_dict[band])
        self.flat_image = torch.Tensor(np.dstack(images)).cuda().double()
        self.P_channel = torch.Tensor(image_dict[self.sub]).cuda().double()
        self.sub_image = torch.autograd.Variable(
            0.5*torch.randn(256, 256, 1).cuda() + torch.tensor(image_dict[self.sub]).cuda(),
            requires_grad=True
        ).double()

    def prediction(self):

        # Sub in the variable image channel
        input_image = torch.cat([self.sub_image, self.flat_image], -1)    # CHANGE THIS ASAP TO BE CONFIGURABLE ORDER
        LSTN_map = self.Predictor(input_image)

        return LSTN_map

    @staticmethod
    def layer_loss(x_image, p_image):

        return nn.functional.mse_loss(x_image, p_image)

    def structural_loss(self):

        x_image = torch.cat([self.sub_image, self.flat_image], -1).reshape(1, 3, 256, 256)
        p_image = torch.cat([self.P_channel, self.flat_image], -1).reshape(1, 3, 256, 256).detach() # Make this a constant!

        layer_losses = {}

        for layer_name, layer_module in self.StructureLayers:

            x_image = layer_module(x_image)
            p_image = layer_module(p_image)

            layer_losses[layer_name] = self.layer_loss(x_image, p_image)

        #Weight them here:
        structural_loss = torch.stack(list(layer_losses.values())).sum()

        wandb.log({"Full structure loss": structural_loss})
        wandb.log(layer_losses)

        return structural_loss

    def show_image(self):

        return torch.cat([self.sub_image, self.flat_image], -1)    # CHANGE THIS ASAP TO BE CONFIGURABLE ORDER

    @staticmethod
    def extract_thresh(image, thres):
        return torch.clamp(image, thres, 1) - thres

    def evaluation_loss(self, LSTN_map, NDVI_map, original_LSTN_map):

        vegetation = self.extract_thresh(NDVI_map, 0.2)
        vegetation_squared_sum = torch.sum(torch.mul(vegetation, vegetation))

        UHI = self.extract_thresh(LSTN_map, 0.0) # Changed from 1.0!
        UHI_squared_sum = torch.sum(torch.mul(UHI, UHI))

        wandb.log({
            "LSTN > 0 squared sum": UHI_squared_sum,
            "Vegetation squared sum": vegetation_squared_sum
        })

        performance_loss = UHI_squared_sum + self.NDVI_factor * vegetation_squared_sum

        return performance_loss

# Penalise vegetation cross with water & buildings
# Penalise area of UHI
# Apply discriminator to it
