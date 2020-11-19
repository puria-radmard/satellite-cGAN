import wandb
import argparse
import yaml

from training_utils import req_args_dict, TASK_CHANNELS, models_args_dict
from cGAN import train_cGAN

from imports import *
from cGAN import *

dir_path = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(dir_path, "..", "data_source", "LONDON_DATASET")


class Config:
    def __init__(self, conf):
        self.__dict__.update(conf)

    def __repr__(self):
        return str(self.__dict__)


def generate_config(args):

    if args.arg_source == None:  # i.e. sweep or manual

        if args.train_size == None:
            train_size = 1 - args.test_size

        # Some leftover params here from the loss_parameters but no worries
        config_dict = vars(args)
        config_dict.update({"train_size": train_size})

    else:  # i.e. constant run, no sweep, with config file

        with open(args.arg_source, "r") as stream:

            config_dict = yaml.safe_load(stream)
            config_dict["task"] = args.task
            config_dict["wandb"] = args.wandb

    loss_parameters = {
        param: config_dict[param] for param in req_args_dict[args.comparison_loss_fn]
    }
    test_parameters = {
        param: config_dict[param] for param in req_args_dict[args.test_metric]
    }
    model_parameters = {
        param: config_dict[param] for param in models_args_dict[args.model]
    }

    config_dict["test_parameters"] = test_parameters
    config_dict["loss_parameters"] = loss_parameters
    config_dict["model_parameters"] = model_parameters
    config_dict["channels"] = TASK_CHANNELS[args.task]["channels"]
    config_dict["classes"] = TASK_CHANNELS[args.task]["classes"]

    if args.wandb:
        wandb.init(project="satellite-cGAN", config=config_dict)
        return wandb.config

    else:
        return Config(config_dict)


def parse_args():

    parser = argparse.ArgumentParser(
        description="""
        Give loss function, evaluation metric, and hyperparameters
        There are three ways to launch the training script:
            1: running `python train_cGAN.py --model unet ...`, i.e. manual argument entry
            2: running `python train_cGAN.py --task reg --arg_source training_config/reg_constant_unet.yaml --wandb 0`
                With this one the --arg_source file takes care of most of the parameters, but task and wandb are left
                Other params are ignored in this case!!
            3: running with wandb sweep, which has its own interface
        """,
        add_help=False,
    )

    task_config = parser.add_argument_group("Task configuration")
    task_config.add_argument("--task", type=str)  # reg, cls, mix
    task_config.add_argument(
        "--arg_source", type=str, default=None
    )  # args yaml path, None for sweeps
    task_config.add_argument("--wandb", type=int, default=1)  # 1 to include wandb

    # Set up parameters
    set_up_params = parser.add_argument_group("General setup parameters")
    set_up_params.add_argument(
        "--model",
        type=str,
        help="Name of the model, used by config.model_dict to intialise the mode",
    )
    set_up_params.add_argument(
        "--save_rate",
        type=int,
        help="How many epochs between saving the model. Model saves are not overwritten, and their file names are set by --task",
    )
    set_up_params.add_argument(
        "--random_state", type=int, default=1, help="Set a random state for numpy"
    )
    set_up_params.add_argument(
        "--num_epochs", type=int, default=20, help="Number of epochs to train for"
    )
    set_up_params.add_argument(
        "--test_size", type=float, help="Proportion of dataset to use for testing"
    )
    set_up_params.add_argument(
        "--train_size",
        type=float,
        default=None,
        help="Proportion of dataset to use for testing. <= 1-test_size",
    )
    set_up_params.add_argument(
        "--data_dir", type=str, help="Directory where the data images are stored"
    )
    set_up_params.add_argument(
        "--comparison_loss_fn",
        type=str,
        help="The function used for loss compared to training targets (only loss if there is no discriminator)",
    )
    set_up_params.add_argument(
        "--test_metric", type=str, help="Metric used for testing"
    )
    set_up_params.add_argument(
        "--no_discriminator",
        type=bool,
        help="Set true for no adversarial loss, i.e. 'vanilla' FCN case",
    )
    set_up_params.add_argument("--purge_data", type=bool, help="Largely unused now")

    universal_hyperparameters = parser.add_argument_group("Universal hyperparameters")
    universal_hyperparameters.add_argument(
        "--lr", type=float, help="Learning rate for generator and discriminator"
    )
    universal_hyperparameters.add_argument(
        "--batch_size", type=int, help="Batch size for training and testing"
    )
    universal_hyperparameters.add_argument(
        "--dis_dropout", type=float, help="Training dropout rate for discriminator"
    )
    universal_hyperparameters.add_argument(
        "--gen_dropout", type=float, help="Training dropout rate for generator"
    )
    universal_hyperparameters.add_argument(
        "--comparison_loss_factor",
        type=float,
        help="Final loss = adversarial loss + comparison_loss_factor * comparison loss. Automatically normalised in script",
    )

    loss_arguments = parser.add_argument_group(
        "Specialised (loss/score) parameters, picked up using config.req_args_dict"
    )
    loss_arguments.add_argument(
        "--cls_layer", type=int
    )  # For mixed loss, so not really needed right now
    loss_arguments.add_argument(
        "--reg_layer", type=int
    )  # For mixed loss, so not really needed right now
    loss_arguments.add_argument(
        "--cls_lambda", type=float
    )  # For mixed loss, so not really needed right now
    loss_arguments.add_argument(
        "--reg_lambda", type=float
    )  # For mixed loss, so not really needed right now
    loss_arguments.add_argument("--beta", type=float)  # For ternaus (BCE part)
    loss_arguments.add_argument(
        "--l", type=float
    )  # For ternaus (weight of Jaccard part)

    model_arguments = parser.add_argument_group(
        "Specialised (model) parameters, picked up by config.model_args_dict"
    )
    model_arguments.add_argument(
        "--no_skips", type=bool
    )  # Used by unet. True for no skips

    args = parser.parse_args()
    config = generate_config(args)
    print(config)

    return config


if __name__ == "__main__":

    config = parse_args()

    with torch.autograd.set_detect_anomaly(True):
        train_cGAN(config)
