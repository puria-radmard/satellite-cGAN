from imports import *


class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, max_before=True):

        super().__init__()

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=dropout),
        ]

        if max_before:
            layers.insert(0, nn.MaxPool2d(2))

        self.downblock = nn.Sequential(*layers)

    def forward(self, X):
        X = X.float()
        return self.downblock(X)


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, no_skips):
        super().__init__()

        # Channels are doubled by upconv
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )

        self.no_skips = no_skips
        if no_skips:
            in_channels = int(in_channels / 2)

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=dropout),
        ]

        self.dbconv = nn.Sequential(*layers)

    def forward(self, X1, X2):
        X1 = X1.float()
        X2 = X2.float()
        X1 = self.upconv(X1)

        # Taking input NCHW
        if self.no_skips:
            X = X1
        else:
            X = torch.cat([X2, X1], dim=1)

        return self.dbconv(X)


class UNetOutBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetOutBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, X):
        X = X.float()
        return self.conv(X)


class UNet(nn.Module):
    def __init__(self, dropout, no_skips, sigmoid_channels, n_channels=3, n_classes=1):

        super(UNet, self).__init__()

        # To call later
        self.no_skips = no_skips
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.sigmoid_channels = sigmoid_channels

        if len(sigmoid_channels) != n_classes:
            raise ValueError("len(sigmoid_channels) != n_classes")

        convchan1 = 64
        convchan2 = 128
        convchan3 = 256
        convchan4 = 512
        convchan5 = 1024

        self.down1 = UNetDownBlock(n_channels, convchan1, dropout, max_before=False)
        self.down2 = UNetDownBlock(convchan1, convchan2, dropout)
        self.down3 = UNetDownBlock(convchan2, convchan3, dropout)
        self.down4 = UNetDownBlock(convchan3, convchan4, dropout)
        self.down5 = UNetDownBlock(convchan4, convchan5, dropout)

        convchan6 = 512
        convchan7 = 256
        convchan8 = 128
        convchan9 = 64

        self.up1 = UNetUpBlock(convchan5, convchan6, dropout, no_skips=no_skips)
        self.up2 = UNetUpBlock(convchan6, convchan7, dropout, no_skips=no_skips)
        self.up3 = UNetUpBlock(convchan7, convchan8, dropout, no_skips=no_skips)
        self.up4 = UNetUpBlock(convchan8, convchan9, dropout, no_skips=no_skips)

        self.out = UNetOutBlock(convchan9, n_classes)

    def forward(self, X, reorder=True):
        """
        Layers take inputs of size [N, C, H, W]
        Forward takes inputs of size [N, H, W, C] or [H, W, C]
        """

        if len(np.shape(X)) == 3:
            X = X[np.newaxis, :]
        if reorder:
            X = X.permute(0, 3, 1, 2)
        if np.shape(X)[1] == 4:
            X = X[:, :3, :, :]

        X1 = self.down1(X)
        X2 = self.down2(X1)
        X3 = self.down3(X2)
        X4 = self.down4(X3)
        X5 = self.down5(X4)
        up = self.up1(X5, X4)
        up = self.up2(up, X3)
        up = self.up3(up, X2)
        up = self.up4(up, X1)
        logits = self.out(up)

        if len(self.sigmoid_channels) != 1:
            raise NotImplementedError("Haven't figured out mixed result yet")

        if self.sigmoid_channels[0]:
            return nn.Sigmoid()(logits)
        else:
            return logits


def trainEpoch(model, epoch, optimizer, dataloader, num_steps, loss_fn):

    model.train()

    epoch_loss_tot = 0
    start_time = time.time()

    for step, batch in enumerate(dataloader):

        images = batch["image"]
        labels = batch["label"]
        preds = model.forward(images)
        labels = torch.tensor(labels).type_as(preds)
        loss = loss_fn(preds, labels)
        loss.backward()
        model.float()
        optimizer.step()

        epoch_loss_tot += loss.mean()
        epoch_loss = epoch_loss_tot / ((step + 1))
        steps_left = num_steps - step
        time_passed = time.time() - start_time

        ETA = (time_passed / (step + 1)) * (steps_left)
        ETA = "{} m  {} s".format(np.floor(ETA / 60), int(ETA % 60))

        string = "Epoch: {}   Step: {}   Batch Loss: {:.4f}   Epoch Loss: {:.4f}   Epoch ETA: {}".format(
            epoch, step, loss, epoch_loss.mean(), ETA
        )

        sys.stdout.write("\r" + string)
        time.sleep(0.5)

        try:
            wandb.log({"iteration_loss": loss.mean()})
        except NameError:
            pass

        if step == num_steps:
            break

    return epoch_loss


def testModel(model, epoch, dataloader, num_steps, test_metric):

    model.eval()

    epoch_score_tot = 0
    start_time = time.time()

    for step, batch in enumerate(dataloader):

        images = batch["image"]
        labels = batch["label"]
        preds = model.forward(images)
        labels = torch.tensor(labels).type_as(preds)
        score = test_metric(preds, labels)

        epoch_score_tot += score.mean()
        epoch_score = epoch_score_tot / ((step + 1))
        steps_left = num_steps - step
        time_passed = time.time() - start_time

        ETA = (time_passed / (step + 1)) * (steps_left)
        ETA = "{} m  {} s".format(np.floor(ETA / 60), int(ETA % 60))

        string = "Evaluating epoch: {}   Step: {}   Batch score: {:.4f}   Epoch score: {:.4f}   Epoch ETA: {}".format(
            epoch, step, score, epoch_score.mean(), ETA
        )

        sys.stdout.write("\r" + string)
        time.sleep(0.5)

        del preds
        del labels
        del images

        if step == num_steps:
            break

    print(f"Epoch: {epoch}, test metric: {epoch_score}")
    return epoch_score
