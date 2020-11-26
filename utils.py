from typing import List
import os
import time
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from pipelines.utils import read_raster, get_property_path, slice_middle
from datetime import datetime


def reshape_for_discriminator(a, num_classes):
    # Change shape from [N, C, H, W] to [NxC, 1, H, W]
    return a.view(a.shape[0] * num_classes, 1, a.shape[2], a.shape[3])


def reshape_for_discriminator2(a, num_classes):
    # Change shape from [N, C, H, W] to [NxC, 1, H, W]
    return a.view(a.shape[0] * num_classes, 1, a.shape[1], a.shape[2])


def skip_tris(batch):
    batch = list(filter(lambda x: x["image"][0] is not None, batch))
    return default_collate(batch)


def record_groups(train_groups, test_groups, root_dir):

    with open(os.path.join(root_dir, "groups.txt"), "w") as fout:

        fout.write("TRAIN_GROUPS")
        fout.write("\n")
        for group in train_groups:
            fout.write(get_property_path(group, "XXX"))
            fout.write("\n")

        fout.write("\n")

        fout.write("TEST_GROUPS")
        fout.write("\n")
        for group in train_groups:
            fout.write(get_property_path(group, "XXX"))
            fout.write("\n")


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, X):
        print(X.shape)
        return X


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, X):
        return self.lambd(X)


class BaseCNNDatabase(Dataset):
    def __init__(
        self,
        groups,
        channels: List[str],
        classes: List[str],
        normalise_input,
        transform,
    ):
        """
        TODO: Ask about transformation viability
        """

        # Group comes in as a train/test set - it is split up before it gets here
        self.groups = groups
        self.transform = transform
        # Channel names must be in the right order
        self.channels = channels
        self.classes = classes
        self.normalise_input = normalise_input

    def __len__(self):
        return len(self.groups)

    def base_getitem(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        group = self.groups[idx]
        input_images = []
        label_images = []

        for input_channel in self.channels:
            image = read_raster(group[input_channel])[0]
            # image -= np.nanmin(image)
            # image = 2 * (image / np.nanmax(image)) - 1
            image = np.expand_dims(image, -1)
            image = slice_middle(image)
            if isinstance(image, type(None)):
                return {"image": [None], "label": [None]}
            if self.normalise_input:
                mean = np.nanmean(image)
                std = np.nanstd(image)
                image = (image - mean) / std
            input_images.append(image)

        for label_channel in self.classes:
            image = read_raster(group[label_channel])[0]
            # image -= np.nanmin(image)
            # image = 2 * (image / np.nanmax(image)) - 1
            image = np.expand_dims(image, -1)
            image = slice_middle(image)
            label_images.append(image)

        sample = {"image": np.dstack(input_images), "label": np.dstack(label_images)}

        return sample


class LandsatDataset(BaseCNNDatabase):
    def __init__(
        self,
        groups,
        channels: List[str],
        classes: List[str],
        normalise_input=False,
        transform=None,
    ):

        super(LandsatDataset, self).__init__(
            groups=groups,
            channels=channels,
            classes=classes,
            normalise_input=normalise_input,
            transform=transform,
        )

    def __getitem__(self, idx):

        return self.base_getitem(idx)


class AutoEncoderDataset(BaseCNNDatabase):
    def __init__(self, groups, channels: List[str], classes: List[str], transform=None):

        super(AutoEncoderDataset, self).__init__(
            groups=groups, channels=channels, classes=classes, transform=transform
        )

    def __getitem__(self, idx):

        sample = self.base_getitem(idx)
        split_image = []
        split_label = []

        for j in range(int(256 / 32)):
            split_image.append(
                sample["image"][j * 32 : (j + 1) * 32, j * 32 : (j + 1) * 32, :]
            )
            split_label.append(
                sample["label"][j * 32 : (j + 1) * 32, j * 32 : (j + 1) * 32, :]
            )

        sample = {
            "image": np.stack(split_image, axis=0),
            "label": np.stack(split_label, axis=0),
        }


class DummyDataset(Dataset):
    def __init__(self, channels, classes):
        self.channels = channels
        self.classes = classes
        self.groups = [1 for a in range(60)]

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        dummy_instance = self.groups[idx]

        sample = {
            "image": np.random.rand(256, 256, len(self.channels)),
            "label": np.random.rand(len(self.classes), 256, 256),
        }

        return sample


def write_loading_bar_string(
    metrics, step, epoch_metric_tot, num_steps, start_time, epoch, training=True
):

    if training:
        metric_name = "Loss"
        title = "E"
    else:
        metric_name = "Score"
        title = "Evaluating e"

    metric = sum(metrics)
    epoch_metric_tot += metric
    epoch_metric = epoch_metric_tot / ((step + 1))
    steps_left = num_steps - step
    time_passed = time.time() - start_time
    ETA = (time_passed / (step + 1)) * (steps_left)
    ETA = "{} m  {} s".format(np.floor(ETA / 60), int(ETA % 60))

    string = "{}poch: {}   Step: {}   Batch {}: {:.4f}   Epoch {}: {:.4f}   Epoch ETA: {}".format(
        title, epoch, step, metric_name, metric, metric_name, epoch_metric, ETA
    )

    return string, epoch_metric_tot


def construct_debug_model(layers, debug=False):
    modules = []
    for layer in layers:
        modules.append(layer)
        if debug:
            modules.append(PrintLayer())
    model = nn.Sequential(*modules)
    return model
