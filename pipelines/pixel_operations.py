from utils import *
from root_classes import *
from imports import *


"""
Example usage:

landsat_operation_pipeline = OperationPipeline(
    sequence = [
      NDVIOperation(),
      NDBIOperation(),
      NDWIOperation(),
      LSTOperation(),
    ]
)

landsat_operation_pipeline(root = "/content/drive/My Drive/LONDON_DATASET/")
"""


class NDBIOperation(Operation):
    band_name = "NDBI"
    bands_required = ["B6", "B5"]

    def operation(self, rasters):
        return [
            (rasters["B6"][0] - rasters["B5"][0])
            / (rasters["B6"][0] + rasters["B5"][0])
        ], None


class NDWIOperation(Operation):
    band_name = "NDWI"
    bands_required = ["B3", "B5"]

    def operation(self, rasters):
        return [
            (rasters["B3"][0] - rasters["B5"][0])
            / (rasters["B3"][0] + rasters["B5"][0])
        ], None


class NDVIOperation(Operation):
    band_name = "NDVI"
    bands_required = ["B4", "B5"]

    def operation(self, rasters):
        return [
            (rasters["B5"][0] - rasters["B4"][0])
            / (rasters["B5"][0] + rasters["B4"][0])
        ], None


def emissivity_func(P_v, NDVI, args):
    C_l = args["C_lambda"]
    e_v = args["EPS_v_lambda"]
    e_s = args["EPS_s_lambda"]
    NDVI_s = args["NDVIs"]
    NDVI_v = args["NDVIv"]

    if NDVI < 0:
        return 0.991
    elif NDVI < NDVI_s:
        return e_s
    elif NDVI > NDVI_v:
        return e_s + C_l
    else:
        return e_v * P_v + e_s * (1 - P_v) + C_l


class LSTOperation(Operation):
    band_name = "LST"
    bands_required = ["B10", "NDVI"]
    vidx = {
        "M_L": ["properties", "RADIANCE_MULT_BAND_10"],
        "A_L": ["properties", "RADIANCE_ADD_BAND_10"],
        "K1": ["properties", "K1_CONSTANT_BAND_10"],
        "K2": ["properties", "K2_CONSTANT_BAND_10"],
    }
    variables = dict(
        NDVIs=0.2,
        NDVIv=0.5,
        EPS_v_lambda=0.968,
        EPS_s_lambda=0.996,
        C_lambda=0.005,
        CONST_A=10.895e-6,  # 0.00115,725
        CONST_B=0.01438,
    )

    def operation(self, rasters):
        # Notice format return [image], None
        TOA_raster = (
            self.variables["M_L"] * rasters["B10"][0] + self.variables["A_L"] - 0.29
        )
        BT_raster = (
            self.variables["K2"] / np.log(1 + self.variables["K1"] / TOA_raster)
        ) - 273.15
        VEP_raster = (
            (rasters["NDVI"][0] - self.variables["NDVIs"])
            / (self.variables["NDVIv"] - self.variables["NDVIs"])
        ) ** 2
        # EPS_raster = self.variables["C_lambda"] + self.variables["EPS_v_lambda"]*VEP_raster + self.variables["EPS_s_lambda"]*(1-VEP_raster)
        EPS_raster = np.vectorize(emissivity_func)(
            P_v=VEP_raster, NDVI=rasters["NDVI"][0], args=self.variables
        )
        LST_raster = BT_raster / (
            1
            + np.log(EPS_raster)
            * self.variables["CONST_A"]
            * BT_raster
            / self.variables["CONST_B"]
        )
        return [LST_raster], None
