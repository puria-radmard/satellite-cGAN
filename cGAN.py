from pprint import pprint
from pipelines.utils import *
from imports import *
from utils import *
from unet import *
from config import metric_dict

dir_path = os.path.dirname(os.path.realpath(__file__))




class Discriminator(nn.Module):
    def __init__(self, num_classes, dropout=0.4):
        """
        TODO: parameterise architecture?
        """
        super(Discriminator, self).__init__()

        self.num_classes = num_classes
        self.dropout = dropout

        convchan1 = 64
        convchan2 = 128
        convchan3 = 256
        convchan4 = 512
        convchan5 = 1024

        layers = [
            # For now we assume all images have 1 channel
            UNetDownBlock(1, convchan1, dropout, max_before=False),
            UNetDownBlock(convchan1, convchan2, dropout),
            UNetDownBlock(convchan2, convchan3, dropout),
            UNetDownBlock(convchan3, convchan4, dropout),
            # TODO: Should I change ReLU for this last one? Don't think so right?
            UNetDownBlock(convchan4, convchan5, dropout),
            # The final size here is [N x 1024 x 16 x 16] as per UNet specifications
            # I have added a mean over the last two dimensions here, but it's not
            # that pretty, there must be a better way
            LambdaLayer(lambd=lambda X: X.mean(-1).mean(-1)),
            # Fully connected -> real probability for each type
            nn.Linear(convchan5, self.num_classes),
            nn.Sigmoid(),
        ]

        self.model = construct_debug_model(layers, False)

    def forward(self, X, reorder=True):
        """
        X comes in as [N, H, W, C]
        """
        if len(np.shape(X)) == 3:
            X = X[np.newaxis, :]
        if reorder:
            X = X.permute(0, 3, 1, 2)
        if np.shape(X)[1] == 4:
            X = X[:, :3, :, :]

        logits = self.model(X)
        return logits


class ConditionalGAN(nn.Module):
    def __init__(self, classes, channels, dis_dropout, gen_dropout):

        # Is this needed?
        super(ConditionalGAN, self).__init__()
        self.classes = classes
        self.channels = channels

        self.discriminator = Discriminator(
            num_classes=len(classes), dropout=dis_dropout
        )

        self.generator = UNet(
            dropout=gen_dropout, n_channels=len(channels), n_classes=len(classes)
        )



def landsat_train_test_dataset(
    data_dir,
    channels: List[str],
    classes: List[str],
    test_size=0.3,
    train_size=None,
    random_state=None,
):

    if train_size == None:
        train_size = 1.0 - test_size
    try:
        assert test_size + train_size <= 1.0
    except AssertionError:
        raise AssertionError("test_size + train_size > 1, which is not allowed")

    groups = group_bands(data_dir, channels + classes)

    train_groups, test_groups = train_test_split(
        groups, test_size=test_size, train_size=train_size, random_state=random_state
    )

    print(
        f"{len(train_groups)} training instances, {len(test_groups)} testing instances"
    )

    train_dataset = LandsatDataset(
        groups=train_groups, channels=channels, classes=classes
    )
    test_dataset = LandsatDataset(
        groups=test_groups, channels=channels, classes=classes
    )

    return train_dataset, test_dataset


def train_cGAN_epoch(
    cGAN,
    epoch,
    optimizer_G,
    optimizer_D,
    dataloader,
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

    for step, batch in enumerate(dataloader):

        images = batch["image"]
        labels = batch["label"]

        optimizer_G.zero_grad()
        optimizer_D.zero_grad()

        # print("generating images")
        preds = cGAN.generator.forward(images)

        # Train generator
        cGAN.float()

        # print("doing comparison loss between")
        # print("preds:", preds.shape)
        # print("labels:", labels.shape)
        comparison_loss = comparison_loss_factor * comparison_loss_fn(
            preds.float(), labels.float().reshape(preds.shape)
        )
        comparison_loss.backward(retain_graph=True)

        # print(
        #    "discriminating preds for generator with input size",
        #    reshape_for_discriminator(preds, len(cGAN.classes)).shape
        # )
        dis_probs_gene = cGAN.discriminator.forward(
            reshape_for_discriminator(preds, len(cGAN.classes)), reorder=False
        )

        # print("dis_probs_gene", dis_probs_gene.shape)
        adversarial_loss_gene = adversarial_loss_fn(
            dis_probs_gene, torch.zeros(dis_probs_gene.shape)
        )
        adversarial_loss_gene.backward()

        optimizer_G.step()

        # Train discriminator
        # Very dodgy way to do this
        dis_targets_real = torch.cat([torch.eye(len(cGAN.classes)) for _ in preds])
        # print("dis_targets_real", dis_targets_real)
        labels = labels.type_as(preds)

        # print(
        #     "discriminating real labels for discriminator with input size",
        #     reshape_for_discriminator2(labels, len(cGAN.classes)).shape
        # )
        dis_probs_real = cGAN.discriminator.forward(
            reshape_for_discriminator2(labels, len(cGAN.classes)), reorder=False
        )
        # print("dis_probs_real", dis_probs_real)
        # print("discriminating preds for discriminator")
        dis_probs_gene = cGAN.discriminator.forward(
            reshape_for_discriminator(preds.detach(), len(cGAN.classes)), reorder=False
        )
        adversarial_loss_gene = adversarial_loss_fn(
            dis_probs_gene, torch.zeros(dis_probs_gene.shape)
        )
        adversarial_loss_real = adversarial_loss_fn(dis_probs_real, dis_targets_real)
        adversarial_loss = (adversarial_loss_real + adversarial_loss_gene) / 2
        adversarial_loss.backward()

        optimizer_D.step()

        losses = [comparison_loss.item(), adversarial_loss.item()]

        loading_bar_string, epoch_loss_tot = write_loading_bar_string(
            losses, step, epoch_loss_tot, num_steps, start_time, epoch, training=True
        )

        sys.stdout.write("\r" + loading_bar_string)
        time.sleep(0.1)

        del images
        del labels
        del comparison_loss
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
        score = test_metric(preds, labels.reshape(preds.shape))
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

    cGAN = ConditionalGAN(
        classes=config.classes,
        channels=config.channels,
        dis_dropout=config.dis_dropout,
        gen_dropout=config.gen_dropout,
    )

    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        cGAN.cuda()

    if config.loss_parameters:
        comparison_loss_fn = metric_dict[config.comparison_loss_fn](
            **config.loss_parameters
        )
    else:
        comparison_loss_fn = metric_dict[config.comparison_loss_fn]()
    if config.test_parameters:
        test_metric = metric_dict[config.test_metric](**config.test_parameters)
    else:
        test_metric = metric_dict[config.test_metric]()
    adversarial_loss_fn = nn.BCELoss()

    if config.wandb:
        wandb.watch(cGAN)

    optimizer_G = torch.optim.Adam(cGAN.generator.parameters(), lr=config.lr)
    optimizer_D = torch.optim.Adam(cGAN.discriminator.parameters(), lr=config.lr)

    if config.data_dir:
        train_dataset, test_dataset = landsat_train_test_dataset(
            data_dir=config.data_dir,
            channels=config.channels,
            classes=config.classes,
            test_size=config.test_size,
            train_size=config.train_size,
            random_state=config.random_state,
        )
    else:
        # Debug case
        train_dataset = DummyDataset(channels=config.channels, classes=config.classes)
        test_dataset = DummyDataset(channels=config.channels, classes=config.classes)

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, # collate_fn=skip_tris
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.batch_size, # collate_fn=skip_tris
    )  # Change to own batch size?

    train_num_steps = len(train_dataloader)
    test_num_steps = len(test_dataloader)
    print(
        "Starting training for {} epochs of {} training steps and {} evaluation steps".format(
            config.num_epochs, train_num_steps, test_num_steps
        )
    )

    for epoch in range(config.num_epochs):

        epoch_loss = train_cGAN_epoch(
            cGAN=cGAN,
            epoch=epoch,
            optimizer_D=optimizer_D,
            optimizer_G=optimizer_G,
            dataloader=train_dataloader,
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
            # print("Would be saving now")
            state = {"config": config, "epoch": epoch, "state": cGAN.state_dict()}
            torch.save(
                state,
                os.path.join(dir_path, f"saves/{config.task}_model.epoch{epoch}.t7"),
            )
