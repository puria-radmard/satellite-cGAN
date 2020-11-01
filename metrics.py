from imports import *


def dice_coef(preds, labels):
    smooth = 1.0
    preds = preds.cpu().detach()
    labels = labels.cpu().detach()

    preds_flat = preds.view(-1)
    labels_flat = labels.view(-1)
    intersection = (preds_flat * labels_flat).sum()

    return (2.0 * intersection + smooth) / (
        preds_flat.sum() + labels_flat.sum() + smooth
    )


class DiceCoefficient(nn.Module):
    def __init__(self, **kwargs):
        """
        Beta is on positive side, so a higher beta stops false negatives more
        """
        super(DiceCoefficient, self).__init__()

    def forward(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return dice_coef(preds, labels)


def cross_entropy(output, target, beta):

    # TODO: Fix bug below due to inplace operation, implemeting weighting
    weights = [beta, 1]
    weights = torch.tensor(weights).float()
    weights = weights / torch.norm(weights, dim=0)

    loss = weights[1] * (target * torch.log(output)) + weights[0] * (
        (1 - target) * torch.log(1 - output)
    )
    print("cross_entropy", loss.shape)

    return loss


def perPixelCrossEntropy(preds, labels, beta):
    size = torch.prod(torch.tensor(labels.shape)).float()
    assert preds.shape == labels.shape
    loss = -(1 / size) * torch.sum(cross_entropy(preds, labels, beta))
    print("ppce loss", loss)
    return loss


def jaccardIndex(preds, labels, class_weights=None):
    size = torch.prod(torch.tensor(labels.shape)).float()
    assert preds.shape == labels.shape

    preds -= preds.min().item()
    preds /= preds.max().item()
    labels -= labels.min().item()
    labels /= labels.max().item()
    indices = (1 / size) * torch.sum(
        preds * labels / (preds + labels - labels * preds + 1e-10)
    )

    return indices


def ternausLossfunc(preds, labels, l=1, beta=1):
    # Derived from https://arxiv.org/abs/1801.05746
    H = perPixelCrossEntropy(preds, labels, beta)
    J = jaccardIndex(preds, labels)
    T = H - l * torch.log(J + 1e-10)
    return T


class TernausLossFunc(nn.Module):
    def __init__(self, **kwargs):
        """
        Beta is on positive side, so a higher beta stops false negatives more
        """
        super(TernausLossFunc, self).__init__()
        self.l = kwargs.get("l", 1)
        self.beta = kwargs.get("beta", 1)

    def forward(
        self, preds: torch.Tensor, labels: torch.Tensor, reorder=False
    ) -> torch.Tensor:
        if reorder:
            labels = labels.permute(0, 3, 1, 2)
        return ternausLossfunc(preds, labels, self.l, self.beta)


class TargettedRegressionClassification(nn.Module):
    # This provides a classification loss to the segmentation layer,
    # and a regression loss to the continuous layer weighted by the segmentation

    def __init__(self, **kwargs):
        super(TargettedRegressionClassification, self).__init__()

        # Which channel of the inputted images is cls and reg
        self.cls_layer = kwargs["cls_layer"]
        self.reg_layer = kwargs["reg_layer"]

        self.reg_loss_func = kwargs["reg_loss_func"]
        self.cls_loss_func = kwargs["cls_loss_func"]

        self.cls_lambda = kwargs["cls_lambda"]
        self.reg_lambda = kwargs["reg_lambda"]

    def forward(self, preds, labels):
        reg_preds = preds[:, self.reg_layer]
        cls_preds = preds[:, self.cls_layer]
        mul_preds = cls_preds * reg_preds

        reg_labels = labels[:, :, :, self.reg_layer]
        cls_labels = labels[:, :, :, self.cls_layer]
        import pdb

        pdb.set_trace()  # check that cls_layer is binary

        reg_loss = self.reg_loss_func(mul_preds, reg_labels)
        cls_loss = self.cls_loss_func(cls_preds, cls_labels)

        comparison_loss = self.reg_lambda * reg_loss + self.cls_lambda * cls_loss

        return comparison_loss


class TargettedTernausAndMSE(TargettedRegressionClassification):
    # A targetted loss that uses Ternaus for cls and MSE for reg

    def __init__(self, **kwargs):

        kwargs["reg_loss_func"] = TernausLossFunc(**kwargs)
        kwargs["cls_loss_func"] = nn.MSELoss()
        super(TargettedTernausAndMSE, self).__init__(**kwargs)
