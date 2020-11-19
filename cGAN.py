from pprint import pprint
from pipelines.utils import *
from imports import *
from utils import *
from training_utils import metric_dict, prepare_training, normalise_loss_factor
from models import ConditionalGAN, UNet, FPN

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
):

    # Might need to fix this
    cGAN.train()

    epoch_loss_tot = 0
    start_time = time.time()

    comparison_loss_factor, loss_mag = normalise_loss_factor(
        cGAN, comparison_loss_factor
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    for step, batch in enumerate(dataloader):

        images = batch["image"]

        optimizer_G.zero_grad()
        if optimizer_D:
            optimizer_D.zero_grad()

        preds = cGAN.generator.forward(images)
        labels = batch["label"].type_as(preds)

        # Train generator
        # cGAN.float()

        comparison_loss = comparison_loss_factor * comparison_loss_fn(
            preds.float(), labels.float().reshape(preds.shape)
        )
        comparison_loss.backward(retain_graph=True)

        losses = [comparison_loss.item()]

        if cGAN.has_discriminator:

            generator_adversarial_loss_gene, discriminator_adversarial_loss = generate_adversarial_loss(
                cGAN, preds, adversarial_loss_fn
            )
            generator_adversarial_loss_gene.backward()
            discriminator_adversarial_loss.backward()
            optimizer_D.step()

            losses.extend(
                [
                    generator_adversarial_loss.item(),
                    discriminator_adversarial_loss.item(),
                ]
            )

        optimizer_G.step()

        loading_bar_string, epoch_loss_tot = write_loading_bar_string(
            losses, step, epoch_loss_tot, num_steps, start_time, epoch, training=True
        )

        sys.stdout.write("\r" + loading_bar_string)
        time.sleep(0.1)

        del images
        del labels
        del comparison_loss

        if cGAN.discriminator:
            del adversarial_loss
            del adversarial_loss_real
            del adversarial_loss_gene

        if wandb_flag:
            wandb.log({"iteration_loss": sum(losses)})

        if step == num_steps:
            break

    return epoch_loss_tot / num_steps


def test_cGAN_epoch(cGAN, epoch, dataloader, num_steps, test_metric):

    # Again might need to fix this
    cGAN.eval()

    epoch_score_tot = 0
    start_time = time.time()

    for step, batch in enumerate(dataloader):

        images = batch["image"]
        labels = batch["label"]

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

        del images
        del labels
        del score

        if step == num_steps:
            break

    print(f"Epoch: {epoch}, test metric: {epoch_score_tot}")
    return epoch_score_tot / num_steps


def train_cGAN(config):

    cGAN, comparison_loss_fn, test_metric, adversarial_loss_fn, optimizer_D, optimizer_G, train_dataset, test_dataloader, train_num_steps, test_num_steps = prepare_training(
        config=config
    )

    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        cGAN.cuda()

    for epoch in range(config.num_epochs):

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
        )

        print(f"\nTraining epoch {epoch} done")

        epoch_score = test_cGAN_epoch(
            cGAN=cGAN,
            epoch=epoch,
            dataloader=test_dataloader,
            num_steps=test_num_steps,
            test_metric=test_metric,
        )

        epoch_metrics = {f"epoch_loss": epoch_loss, f"epoch_score": epoch_score}

        if config.wandb:
            wandb.log(epoch_metrics)

        if (epoch + 1) % config.save_rate == 0:

            state = {"config": config, "epoch": epoch, "state": cGAN.state_dict()}
            torch.save(
                state,
                os.path.join(
                    dir_path, f"saves/{config.task}_LSTN2_model.epoch{epoch}.t7"
                ),
            )
