import wandb
import argparse
import yaml

from config import req_args_dict, TASK_CHANNELS
from cGAN import train_cGAN

from imports import *
from cGAN import *

dir_path = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(dir_path, "..", "data_source", "LONDON_DATASET")


class Config:
    def __init__(self, config):
        self.__dict__.update(config)

    def __repr__(self):
        return str(self.__dict__)


def generate_config(args):

    if args.arg_source == None:  # i.e. sweep

        if args.train_size == None:
            train_size = 1 - args.test_size

        config_dict = {
            "wandb": args.wandb,
            "task": args.task,
            "save_rate": args.save_rate,
            "random_state": args.random_state,
            "num_epochs": args.num_epochs,
            "test_size": args.test_size,
            "train_size": train_size,
            "data_dir": args.data_dir,
            "comparison_loss_fn": args.comparison_loss_fn,
            "test_metric": args.test_metric,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "dis_dropout": args.dis_dropout,
            "gen_dropout": args.gen_dropout,
            "comparison_loss_factor": args.comparison_loss_factor,
            "no_discriminator": args.no_discriminator,
            "no_skips": args.no_skips,
        }

        loss_parameters = {
            param: vars(args)[param] for param in req_args_dict[args.comparison_loss_fn]
        }
        test_parameters = {
            param: vars(args)[param] for param in req_args_dict[args.test_metric]
        }

        config_dict["test_parameters"] = test_parameters
        config_dict["loss_parameters"] = loss_parameters
        config_dict["channels"] = TASK_CHANNELS[args.task]["channels"]
        config_dict["classes"] = TASK_CHANNELS[args.task]["classes"]

        wandb.init(project="satellite-cGAN", config=config_dict)
        return wandb.config

    else:  # i.e. constant run, no sweep
        with open(args.arg_source, "r") as stream:
            config_dict = yaml.safe_load(stream)
            config_dict["task"] = args.task
            config_dict["wandb"] = args.wandb

        if args.wandb:
            wandb.init(project="satellite-cGAN", config=config_dict)
            return wandb.config

        else:
            return Config(config_dict)


if __name__ == "__main__":
    print("jbsgubhgio")
    parser = argparse.ArgumentParser(
        description="Give loss function, evaluation metric, and hyperparameters"
    )

    parser.add_argument("--task", type=str)  # reg, cls, mix
    parser.add_argument(
        "--arg_source", type=str, default=None
    )  # args yaml path, None for sweeps
    parser.add_argument("--wandb", type=int, default=1)  # 1 to include wandb

    # If you provide an arg_source, none of this is needed. However, if you are calling via a
    # wandb sweep, this is how your args are set.

    # Set up parameters
    parser.add_argument("--save_rate", type=int)
    parser.add_argument("--random_state", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--test_size", type=float)
    parser.add_argument("--train_size", type=float, default=None)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--comparison_loss_fn", type=str)
    parser.add_argument("--test_metric", type=str)
    parser.add_argument("--no_discriminator", type=bool)
    parser.add_argument("--no_skips", type=bool)
    parser.add_argument("--purge_data", type=bool)

    # Universal parameters
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--dis_dropout", type=float)
    parser.add_argument("--gen_dropout", type=float)
    parser.add_argument("--comparison_loss_factor", type=float)

    # Specialised (loss) parameters, picked up using req_args_dict
    parser.add_argument("--cls_layer", type=int)
    parser.add_argument("--reg_layer", type=int)
    parser.add_argument("--cls_lambda", type=float)
    parser.add_argument("--reg_lambda", type=float)
    parser.add_argument("--beta", type=float)  # For ternaus (BCE part)
    parser.add_argument("--l", type=float)  # For ternaus (weight of Jaccard part)

    args = parser.parse_args()
    config = generate_config(args)
    print(config)

    with torch.autograd.set_detect_anomaly(True):
        train_cGAN(config)
