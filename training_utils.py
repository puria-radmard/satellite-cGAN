from imports import *
from metrics import *
from models import *
from pipelines.utils import group_bands
from utils import *
import os
from datetime import datetime

models_dict = {"unet": UNet, "fpn": FPN}

models_args_dict = {"unet": ["no_skips"], "fpn": ["resnet_encoder"]}

metric_dict = {
    "bce_loss": nn.BCELoss,
    "mse_loss": nn.MSELoss,
    "ternaus_loss": TernausLossFunc,
    "targetted_ternaus_and_MSE": TargettedTernausAndMSE,
    "dice_coefficient": DiceCoefficient,
}

req_args_dict = {
    "bce_loss": [],
    "mse_loss": ["size_average"],
    "ternaus_loss": ["beta", "l"],
    "targetted_ternaus_and_MSE": [
        "cls_layer",
        "reg_layer",
        "cls_lambda",
        "reg_lambda",
        "beta",
        "l",
    ],
    "dice_coefficient": [],
}

TASK_CHANNELS = {
    "reg": {"channels": ["NDVI", "NDBI", "NDWI"], "classes": ["LSTN2"]},
    "cls": {"channels": ["NDVI", "NDBI", "NDWI"], "classes": ["UHI"]},
    "mix": {"channels": ["NDVI", "NDBI", "NDWI"], "classes": ["LSTN", "UHI"]},
}


def make_root_dir(config):

    ts = "-".join((str(datetime.now()).split()))
    root_dir = f"run-{ts}"
    os.mkdir(root_dir)

    with open(os.path.join(root_dir, "config.txt"), "w") as f:
        for k, v in config.__dict__.items():
            f.write(f"{k}\t{v}\n")

    return root_dir


def normalise_loss_factor(model, comparison_loss_factor):

    if model.has_discriminator:
        loss_mag = (1 + comparison_loss_factor ** 2) ** 0.5
    else:
        loss_mag, comparison_loss_factor = 1, 1
    comparison_loss_factor /= loss_mag

    return comparison_loss_factor, loss_mag


def landsat_train_test_dataset(
    data_dir,
    channels: List[str],
    classes: List[str],
    test_size,
    train_size,
    random_state,
    purge_data,
    normalise_indices,
):

    if train_size == None:
        train_size = 1.0 - test_size
    try:
        assert test_size + train_size <= 1.0
    except AssertionError:
        raise AssertionError("test_size + train_size > 1, which is not allowed")

    groups = group_bands(data_dir, channels + classes)
    if purge_data:
        groups = purge_groups(groups)

    train_groups, test_groups = train_test_split(
        groups, test_size=test_size, train_size=train_size, random_state=random_state
    )

    print(
        f"{len(train_groups)} training instances, {len(test_groups)} testing instances"
    )

    train_dataset = LandsatDataset(
        groups=train_groups,
        channels=channels,
        classes=classes,
        normalise_input=normalise_indices,
    )
    test_dataset = LandsatDataset(
        groups=test_groups,
        channels=channels,
        classes=classes,
        normalise_input=normalise_indices,
    )

    return train_dataset, test_dataset


def prepare_training(config):

    if config.task == "reg":
        sigmoid_channels = [False]
    elif config.task == "cls":
        sigmoid_channels = [True]
    elif config.task == "mix":
        sigmoid_channels = [None, None]
        sigmoid_channels[config.reg_layer] = False
        sigmoid_channels[config.cls_layer] = True
    else:
        raise ValueError(f"{config.task} is not a recognised task (reg, cls, mix)")

    cGAN = ConditionalGAN(
        classes=config.classes,
        channels=config.channels,
        dis_dropout=config.dis_dropout,
        gen_dropout=config.gen_dropout,
        no_discriminator=config.no_discriminator,
        sigmoid_channels=sigmoid_channels,
        generator_class=models_dict[config.model],
        generator_params=config.model_parameters,
    )

    comparison_loss_fn = metric_dict[config.comparison_loss_fn](
        **config.loss_parameters
    )
    test_metric = metric_dict[config.test_metric](**config.test_parameters)
    adversarial_loss_fn = nn.BCELoss()

    if config.wandb:
        wandb.watch(cGAN)

    optimizer_G = torch.optim.Adam(cGAN.generator.parameters(), lr=config.lr)
    if config.no_discriminator:
        optimizer_D = None
    else:
        optimizer_D = torch.optim.Adam(cGAN.discriminator.parameters(), lr=config.lr)

    train_dataset, test_dataset = landsat_train_test_dataset(
        data_dir=config.data_dir,
        channels=config.channels,
        classes=config.classes,
        test_size=config.test_size,
        train_size=config.train_size,
        random_state=config.random_state,
        purge_data=config.purge_data,
        normalise_indices=config.normalise_indices,
    )

    root_dir = make_root_dir(config)
    record_groups(
        train_groups=train_dataset.groups,
        test_groups=test_dataset.groups,
        root_dir=root_dir,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.batch_size  # collate_fn=skip_tris
    )  # Change to own batch size?
    train_num_steps = len(DataLoader(train_dataset, batch_size=config.batch_size))
    test_num_steps = len(test_dataloader)
    print(
        "Starting training for {} epochs of {} training steps and {} evaluation steps".format(
            config.num_epochs, train_num_steps, test_num_steps
        )
    )

    return (
        cGAN,
        comparison_loss_fn,
        test_metric,
        adversarial_loss_fn,
        optimizer_D,
        optimizer_G,
        train_dataset,
        test_dataset,
        train_num_steps,
        test_num_steps,
        root_dir,
    )


def generate_adversarial_loss(cGAN, preds, labels, loss_mag, adversarial_loss_fn):

    reshaped_preds = reshape_for_discriminator(preds, len(cGAN.classes))
    dis_probs_gene = cGAN.discriminator.forward(reshaped_preds, reorder=False)
    gene_targets = torch.zeros(dis_probs_gene.shape)
    generator_adversarial_loss_gene = adversarial_loss_fn(dis_probs_gene, gene_targets)
    generator_adversarial_loss_gene /= loss_mag

    reshaped_detached_preds = reshape_for_discriminator(
        preds.detach(), len(cGAN.classes)
    )
    dis_probs_gene = cGAN.discriminator.forward(reshaped_detached_preds, reorder=False)
    adversarial_loss_gene = adversarial_loss_fn(dis_probs_gene, gene_targets)

    dis_targets_real = torch.cat([torch.eye(len(cGAN.classes)) for _ in preds])
    reshaped_labels = reshape_for_discriminator2(labels, len(cGAN.classes))
    dis_probs_real = cGAN.discriminator.forward(reshaped_labels, reorder=False)
    adversarial_loss_real = adversarial_loss_fn(dis_probs_real, dis_targets_real)

    discriminator_adversarial_loss = (adversarial_loss_real + adversarial_loss_gene) / (
        2 * loss_mag
    )

    return generator_adversarial_loss_gene, discriminator_adversarial_loss
