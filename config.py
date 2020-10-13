from imports import *
from metrics import *

metric_dict = {
    "bce_loss": nn.BCELoss,
    "mse_loss": nn.MSELoss,
    "ternaus_loss": TernausLossFunc,
    "targetted_ternaus_and_MSE": TargettedTernausAndMSE,
    "dice_coefficient": DiceCoefficient
}

req_args_dict = {
    "bce_loss": [],
    "mse_loss": [],
    "ternaus_loss": [
        "beta",
        "l"
    ],
    "targetted_ternaus_and_MSE": [
        "cls_layer",
        "reg_layer",
        "cls_lambda",
        "reg_lambda",
        "beta",
        "l"
    ],
    "dice_coefficient": []
}

TASK_CHANNELS = {
    "reg": {
        "channels": ["NDVI", "NDBI", "NDWI"],
        "classes": ["LSTN"]
    },
    "cls": {
        "channels": ["NDVI", "NDBI", "NDWI"],
        "classes": ["UHI"]
    },
    "mix":{
        "channels": ["NDVI", "NDBI", "NDWI"],
        "classes": ["LSTN", "UHI"]
    }
}
