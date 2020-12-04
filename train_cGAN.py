#  Copyright (c) 2020. Puria and Hanchen, Email: {pr450, hw501}@cam.ac.uk
import os, sys, pdb, yaml, time, torch, wandb, random, argparse
sys.path.append('models/')
from torch.utils.data import DataLoader
from test_cGAN import save_results_images
from training_utils import req_args_dict, TASK_CHANNELS, models_args_dict, \
    prepare_training, normalise_loss_factor, generate_adversarial_loss
from utils import write_loading_bar_string

dir_path = os.path.dirname(os.path.realpath(__file__))


def train_cGAN_epoch(
        cGAN,
        epoch,
        optimizer_G,
        optimizer_D,
        dataset,
        batch_size,
        comparison_loss_fn,
        adversarial_loss_fn,
        num_steps,
        comparison_loss_factor,
        wandb_flag,
        log_file,
):
    # Might need to fix this
    cGAN.train()

    epoch_loss_tot = 0
    start_time = time.time()

    comparison_loss_factor, loss_mag = normalise_loss_factor(
        cGAN, comparison_loss_factor
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # pdb.set_trace()
    for step, batch in enumerate(dataloader):

        images = batch["image"].float()
        labels = batch["label"].float()

        optimizer_G.zero_grad()
        if optimizer_D:
            optimizer_D.zero_grad()

        preds = cGAN.generator.forward(images)

        # Train generator
        comparison_loss = comparison_loss_factor * comparison_loss_fn(
            preds.float(), labels.float().reshape(preds.shape)
        )
        comparison_loss.backward(retain_graph=True)

        losses = [comparison_loss.item()]

        if cGAN.has_discriminator:
            (
                generator_adversarial_loss_gene,
                discriminator_adversarial_loss,
            ) = generate_adversarial_loss(
                cGAN, preds, labels, loss_mag, adversarial_loss_fn
            )
            generator_adversarial_loss_gene.backward()
            discriminator_adversarial_loss.backward()
            optimizer_D.step()

            losses.extend(
                [
                    generator_adversarial_loss_gene.item(),
                    discriminator_adversarial_loss.item(),
                ]
            )

        optimizer_G.step()

        loading_bar_string, epoch_loss_tot = write_loading_bar_string(
            losses, step, epoch_loss_tot, num_steps, start_time, epoch, training=True
        )
        log_file.write(loading_bar_string)
        log_file.write("\n")

        sys.stdout.write("\r" + loading_bar_string)
        time.sleep(0.1)

        # after executing the function, this variable will be erased
        # del images
        # del labels
        # del comparison_loss
        #
        # if cGAN.has_discriminator:
        #     del generator_adversarial_loss_gene
        #     del discriminator_adversarial_loss

        if wandb_flag:
            wandb.log({"iteration_loss": sum(losses)})

        # it will automatically ends after a looping
        # if step == num_steps:
        #     break

    return epoch_loss_tot / num_steps


def test_cGAN_epoch(cGAN, epoch, dataset, num_steps, test_metric, log_file):
    # Again might need to fix this
    cGAN.eval()

    epoch_score_tot = 0
    start_time = time.time()

    dataloader = DataLoader(dataset, batch_size=config.batch_size)

    for step, batch in enumerate(dataloader):

        images = batch["image"].float()
        labels = batch["label"].float()

        preds = cGAN.generator.forward(images)
        labels = labels.type_as(preds)
        score = test_metric(preds.float(), labels.reshape(preds.shape).float())
        score = score.item()
        if not isinstance(score, list):
            score = [score]

        loading_bar_string, epoch_score_tot = write_loading_bar_string(
            score, step, epoch_score_tot, num_steps, start_time, epoch, training=False
        )

        sys.stdout.write("\r" + loading_bar_string)
        time.sleep(0.1)

        log_file.write(loading_bar_string)
        log_file.write("\n")

        # del images
        # del labels
        # del score

        # if step == num_steps:
        #     break

    print(f"Epoch: {epoch}, test metric: {epoch_score_tot}")
    return epoch_score_tot / num_steps


def train_cGAN(config):
    (
        cGAN,
        comparison_loss_fn,
        test_metric,
        adversarial_loss_fn,
        optimizer_D,
        optimizer_G,
        scheduler_G,
        train_dataset,
        test_dataset,
        train_num_steps,
        test_num_steps,
        root_dir,
    ) = prepare_training(config=config)
    # pdb.set_trace()

    cGAN.float()
    # cGAN.load_state_dict(torch.load("saves/reg_LSTN2_model.epoch79.t7")["state"])

    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        cGAN.cuda()

    for epoch in range(config.num_epochs):

        scheduler_G.step()

        epoch_root_dir = os.path.join(root_dir, f"epoch-{epoch}")
        os.mkdir(os.path.join(root_dir, f"epoch-{epoch}"))

        log_file = open(os.path.join(epoch_root_dir, f"training_log.txt"), "w")
        epoch_loss = train_cGAN_epoch(
            cGAN=cGAN,
            epoch=epoch,
            optimizer_D=optimizer_D,
            optimizer_G=optimizer_G,
            dataset=train_dataset,
            batch_size=config.batch_size,
            comparison_loss_fn=comparison_loss_fn,
            adversarial_loss_fn=adversarial_loss_fn,
            num_steps=train_num_steps,
            comparison_loss_factor=config.comparison_loss_factor,
            wandb_flag=config.wandb,
            log_file=log_file,
        )
        log_file.close()

        print(f"\nTraining epoch {epoch} done")

        test_log_file = open(os.path.join(epoch_root_dir, f"testing_log.txt"), "w")
        epoch_score = test_cGAN_epoch(
            cGAN=cGAN,
            epoch=epoch,
            dataset=test_dataset,
            num_steps=test_num_steps,
            test_metric=test_metric,
            log_file=test_log_file,
        )
        test_log_file.close()

        epoch_metrics = {f"epoch_loss": epoch_loss, f"epoch_score": epoch_score}

        if config.wandb:
            wandb.log(epoch_metrics)

        if (epoch + 1) % config.save_rate == 0:
            state = {"config": config, "epoch": epoch, "state": cGAN.state_dict()}
            torch.save(
                state,
                os.path.join(
                    dir_path, root_dir, f"{config.task}_{config.model}.epoch{epoch}.t7"
                ),
            )

            example_train_groups = random.sample(train_dataset.groups, 20)
            save_results_images(
                groups=example_train_groups,
                cGAN=cGAN,
                destination_dir=epoch_root_dir,
                normalise_indices=config.normalise_indices,
            )
            example_test_groups = random.sample(test_dataset.groups, 20)
            save_results_images(
                groups=example_test_groups,
                cGAN=cGAN,
                destination_dir=epoch_root_dir,
                normalise_indices=config.normalise_indices,
            )


class Config:
    def __init__(self, conf):
        self.__dict__.update(conf)

    def __repr__(self):
        return str(self.__dict__)


def generate_config(args):
    if not args.arg_source:  # i.e. sweep or manual

        # TODO: settings are too complicated
        if not args.train_size:
            train_size = 1 - args.test_size

        # Some leftover params here from the loss_parameters but no worries
        config_dict = vars(args)
        config_dict.update({"train_size": train_size})
        # Since lists are not allowed in sweeps (or manuals?)
        config_dict["channels"] = TASK_CHANNELS[args.task]["channels"]
        config_dict["classes"] = TASK_CHANNELS[args.task]["classes"]

    else:  # i.e. constant run, no sweep, with config file

        with open(args.arg_source, "r") as stream:

            config_dict = yaml.safe_load(stream)
            config_dict["task"] = args.task
            config_dict["wandb"] = args.wandb

    loss_parameters = {
        param: config_dict[param]
        for param in req_args_dict[config_dict["comparison_loss_fn"]]
    }
    test_parameters = {
        param: config_dict[param] for param in req_args_dict[config_dict["test_metric"]]
    }
    model_parameters = {
        param: config_dict[param] for param in models_args_dict[config_dict["model"]]
    }

    config_dict["test_parameters"] = test_parameters
    config_dict["loss_parameters"] = loss_parameters
    config_dict["model_parameters"] = model_parameters

    if args.wandb:
        wandb.init(project="satellite-cGAN", config=config_dict)
        return wandb.config

    else:
        return Config(config_dict)


def parse_args():
    parser = argparse.ArgumentParser(
        description="""
        Give loss function, evaluation metric, and hyper-parameters
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
    task_config.add_argument("--discretize", type=str) # none, lin, log
    task_config.add_argument("--arg_source", type=str, default=None)  # args yaml path, None for sweeps
    task_config.add_argument("--wandb", type=int, default=1)  # 1 to include wandb

    # Set up parameters
    set_up_params = parser.add_argument_group("General setup parameters")
    set_up_params.add_argument("--model", type=str, help="model, see config.model_dict")
    set_up_params.add_argument("--save_rate", type=int, help="number of epochs to save the model.")
    set_up_params.add_argument("--random_state", type=int, default=1, help="Set a random state for numpy")
    set_up_params.add_argument("--num_epochs", type=int, default=20, help="Number of epochs to train for")
    set_up_params.add_argument("--test_size", type=float, help="Proportion of dataset to use for testing")
    set_up_params.add_argument("--train_size", type=float, default=None,
                               help="Proportion of dataset to use for training. <= 1-test_size", )
    set_up_params.add_argument("--data_dir", type=str, help="Data directory")
    set_up_params.add_argument(
        "--comparison_loss_fn",
        type=str,
        help="The function used for loss compared to training targets (only loss if there is no discriminator)",
    )
    set_up_params.add_argument("--test_metric", type=str, help="Metric used for testing")
    set_up_params.add_argument("--no_discriminator", type=bool,
                               help="True for no adversarial loss, i.e. 'vanilla' UNet")
    set_up_params.add_argument("--normalise_indices", type=bool, help="Normalise inputs (NDVI etc.) to 0 mean, 1 std")
    set_up_params.add_argument("--purge_data", type=bool, help="Largely unused now")

    universal_hyperparameters = parser.add_argument_group("Universal hyperparameters")
    universal_hyperparameters.add_argument("--lr", type=float, help="Learning rate for generator and discriminator")
    universal_hyperparameters.add_argument("--batch_size", type=int, help="Batch size for training and testing")
    universal_hyperparameters.add_argument("--dis_dropout", type=float, help="Training dropout rate for discriminator")
    universal_hyperparameters.add_argument("--gen_dropout", type=float, help="Training dropout rate for generator")
    universal_hyperparameters.add_argument("--scheduler_epoch", type=int, help="scheduler epochs")
    universal_hyperparameters.add_argument("--scheduler_gamma", type=float, help="scheduler gamma")
    universal_hyperparameters.add_argument(
        "--comparison_loss_factor",
        type=float,
        help="Final loss = adversarial loss + comparison_loss_factor * comparison loss. Automatically normalised",
    )

    # TODO: do we need this?
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

    model_arguments = parser.add_argument_group("Specialised (model) parameters, picked up by config.model_args_dict")
    model_arguments.add_argument("--no_skips", type=bool)  # Used by unet. True for no skip connections

    args = parser.parse_args()
    config = generate_config(args)
    print(config)

    return config


if __name__ == "__main__":
    config = parse_args()

    with torch.autograd.set_detect_anomaly(True):
        train_cGAN(config)
