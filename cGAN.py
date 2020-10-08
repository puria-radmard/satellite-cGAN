from imports import *
from unet import *

class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, X):
        return self.lambd(X)

class Discriminator(nn.Module):

  def __init__(self, num_classes, dropout = 0.4):
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

    self.model = torch.nn.Sequential(
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
      LambdaLayer(lambd = lambda X: X.mean(-1).mean(-1)),
      # Fully connected -> real probability for each type
      nn.Linear(convchan5, self.num_classes),
      nn.Sigmoid()
    )

  def forward(self, X, reorder = True):
    """
    X comes in as [N, H, W, C] 
    """
    try:
      assert torch.min(X) >= -1 and torch.max(X) <= 1
    except:
      import pdb; pdb.set_trace()
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
          num_classes = len(classes),
          dropout = dis_dropout
      )

      self.generator = UNet(
          dropout=gen_dropout,
          n_channels=len(channels),
          n_classes=len(classes)
      )


class LandsatDataset(Dataset):
    def __init__(
        self,
        groups,
        channels: List[str],
        classes: List[str],
        transform=None,
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

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        group = self.groups[idx]
        input_images = []
        label_images = []

        for input_channel in self.channels:
          image = read_raster(
              group[input_channel]
          )[0]
          image -= np.nanmin(image)
          image = 2*(image/np.nanmax(image)) - 1
          image = np.expand_dims(image, -1)
          image = slice_middle(image)
          input_images.append(image)
          
        for label_channel in self.classes:
          image = read_raster(
              group[label_channel]
          )[0]
          image -= np.nanmin(image)
          image = 2*(image/np.nanmax(image)) - 1
          image = np.expand_dims(image, -1)
          image = slice_middle(image)
          label_images.append(image)

        sample = {
            "image": np.dstack(input_images),
            "label": np.dstack(label_images),
        }

        return sample


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


def slice_middle(image, size=256):
  mix, miy = [int(m/2) for m in image.shape[:2]]
  s = int(size/2)
  return image[mix-s:mix+s,miy-s:miy+s]


def write_loading_bar_string(metrics, step, epoch_metric_tot, num_steps, start_time, epoch, training = True):

    if training: 
      metric_name = "Loss"; title = "E"
    else:
      metric_name = "Score"; title = "Evaluating e"

    metric = sum([metric.mean() for metric in metrics])
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

    groups = group_bands(
        data_dir,
        channels + classes
    )

    train_groups, test_groups = train_test_split(
        groups,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
    )

    print(f"{len(train_groups)} training instances, {len(test_groups)} testing instances")

    train_dataset = LandsatDataset(
        groups=train_groups,
        channels=channels,
        classes=classes,
    )
    test_dataset = LandsatDataset(
        groups=test_groups,
        channels=channels,
        classes=classes,
    )

    return train_dataset, test_dataset

def reshape_for_discriminator(a):
  # Changes shape from [N, C, H, W] to [NxC, 1, H, W]
  return a.view(a.shape[0]*a.shape[1], 1, a.shape[2], a.shape[3])


def skip_tris(batch):
    batch = list(filter(lambda x:x["image"].size is not (256, 256), batch))
    return default_collate(batch)


def train_cGAN_epoch(
    cGAN,
    epoch,
    optimizer_G,
    optimizer_D,
    dataloader,
    comparison_loss_fn,
    adversarial_loss_fn,
    num_steps,
    comparison_loss_factor
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

        preds = cGAN.generator.forward(images)



        # Train generator
        comparison_loss = comparison_loss_factor * comparison_loss_fn(
            preds,
            labels,
        )
        comparison_loss.backward()

        dis_probs_gene = cGAN.discriminator.forward(
            reshape_for_discriminator(preds),
            reorder=False
        )
        adversarial_loss_gene = adversarial_loss_fn(
            dis_probs_gene,
            torch.zeros(dis_probs_gene.shape)
        )
        adversarial_loss_gene.backward(retain_graph = True)

        cGAN.float()
        optimizer_G.step()

        # Train discriminator
        # Very dodgy way to do this
        dis_targets_real = torch.cat(
            [torch.eye(len(cGAN.classes)) for _ in preds],
        )
        labels = torch.tensor(labels).type_as(preds)
        dis_probs_real = cGAN.discriminator.forward(
            reshape_for_discriminator(labels),
            reorder=False
        )
        dis_probs_gene = cGAN.discriminator.forward(
            reshape_for_discriminator(preds).detach(),
            reorder=False
        )
        adversarial_loss_gene = adversarial_loss_fn(
            dis_probs_gene,
            torch.zeros(dis_probs_gene.shape)
        )
        adversarial_loss_real = adversarial_loss_fn(
            dis_probs_real,
            dis_targets_real
        )
        adversarial_loss = (adversarial_loss_real + adversarial_loss_gene)/2
        adversarial_loss.backward()

        optimizer_D.step()

        losses = [comparison_loss, adversarial_loss_real, adversarial_loss_gene]
        loading_bar_string, epoch_loss_tot = write_loading_bar_string(
            losses, step, epoch_loss_tot, num_steps, start_time, epoch, training = True
        )

        sys.stdout.write("\r" + loading_bar_string)
        time.sleep(0.1)

        # Needs converting
        try:
          wandb.log({"iteration_loss": loss.mean()})
        except NameError:
          pass

        if step == num_steps:
            break

    return epoch_loss_tot/num_steps


def test_cGAN_epoch(cGAN, epoch, dataloader, num_steps, test_metric):

    # Again might need to fix this
    cGAN.eval()

    epoch_score_tot = 0
    start_time = time.time()

    for step, batch in enumerate(dataloader):

        images = batch["image"]
        labels = batch["label"]

        preds = cGAN.generator.forward(images)
        labels = torch.tensor(labels).type_as(preds)
        score = test_metric(preds, labels)
        if not isinstance(score, list):
            score = [score]

        loading_bar_string, epoch_score_tot = write_loading_bar_string(
            score, step, epoch_score_tot, num_steps, start_time, epoch, training = False
        )

        sys.stdout.write("\r" + loading_bar_string)
        time.sleep(0.1)

        if step == num_steps:
            break

    print(f"Epoch: {epoch}, test metric: {epoch_score_tot}")
    return epoch_score_tot/num_steps


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

    comparison_loss_fn = metric_dict[config.comparison_loss_fn](**config.loss_parameters)
    test_metric = metric_dict[config.test_metric](**config.test_parameters)
    adversarial_loss_fn = nn.BCELoss()

    # try:
    #   wandb.watch(cGAN)
    # except NameError:
    #   pass

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
      train_dataset = DummyDataset(channels=channels, classes=classes)
      test_dataset = DummyDataset(channels=channels, classes=classes)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  collate_fn=skip_tris
                      )
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config.batch_size,
                                 collate_fn=skip_tris
                      )  # Change to own batch size?

    train_num_steps = len(train_dataloader)
    test_num_steps = len(test_dataloader)
    print("Starting training for {} epochs of {} training steps and {} evaluation steps".format(
            config.num_epochs, train_num_steps, test_num_steps
    ))

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
            comparison_loss_factor = config.comparison_loss_factor
        )
        print(f"Training epoch {epoch} done")
        epoch_score = test_cGAN_epoch(
            cGAN=cGAN,
            epoch=epoch,
            dataloader=test_dataloader,
            num_steps=test_num_steps,
            test_metric=test_metric
        )

        epoch_metrics = {f"epoch_loss": epoch_loss, f"epoch_score": epoch_score}

        # try:
        #   wandb.log(epoch_metrics)
        # except NameError:
        #   pass

        if (epoch + 1) % config.save_rate == 0:
            print("Would be saving now")