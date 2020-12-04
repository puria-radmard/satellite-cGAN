from utils import read_raster
from tqdm import tqdm
import rasterio.features
import rasterio.mask
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np


class StatisticalAnalysisOperation:
    def __init__(self):
        self.core_col_names = ["city", "hash", "datetime", "sample_name"]
        self.stats_col_names = None
        self.output_band = None
        self.input_bands = None

    @staticmethod
    def replace_band_name(path, target_band):
        dot_list = path.split(".")
        pre = ".".join(dot_list[:-2])
        return f"{pre}.{target_band}.tif"

    def construct_result_row(self, path, sample_name, result):
        row = {
            "city": [
                path.split("/")[-1].split("--")[1]
                for _ in range(len(list(result.values())[0]))
            ],
            "hash": [
                path.split("/")[-1].split("--")[0]
                for _ in range(len(list(result.values())[0]))
            ],
            "datetime": [
                "--".join(
                    path.split("/")[-1].split(".")[0].split("--")[-2:]
                )
                for _ in range(len(list(result.values())[0]))
            ],
            "sample_name": [
                sample_name
                for _ in range(len(list(result.values())[0]))
            ],
        }
        return row

    def operation(self, path_list, sampler):

        dependant_variable_dict = {v: read_raster(v)[0] for v in path_list}
        independent_variable_dict = {
            original_path: {
                band: read_raster(self.replace_band_name(original_path, band))[0]
                for band in self.input_bands
            }
            for original_path in dependant_variable_dict.keys()
        }

        path_dict = sampler.sample_from_dicts(dependant_variable_dict, independent_variable_dict)
        # {path: {sample_name: {"dep": arr, "ind": {band: arr}}}}

        for path, sample_dict in path_dict.items():

            for sample_name, sample in path_dict.items():
                try:
                    dep, ind = sample["dep"], sample["ind"]
                    result = self.stats_op(ind, dep)
                    row = self.construct_result_row(path, sample_name, result)
                    row.update(result)
                    self.results = self.results.append(pd.DataFrame(row), ignore_index=True)
                except ValueError as e:
                    print(e)




class RegressionOperation(StatisticalAnalysisOperation):

    stats_col_names = [
        "m_NDVI_only",
        "c_NDVI_only",
        "m_NDBI_only",
        "c_NDBI_only",
        "m_NDWI_only",
        "c_NDWI_only",
        "m_NDVI_join",
        "m_NDBI_join",
        "m_NDWI_join",
        "c_join",
    ]
    output_band = "LST"
    input_bands = ["NDVI", "NDBI", "NDWI"]

    def stats_op(self, ind, dep):

        out_dict = {}

        # Stack all input and output
        for inpb in self.input_bands:
            regression_array = np.dstack([ind[inpb].copy(), dep.copy()]).reshape(-1, 2)
            regression_array = regression_array[~np.isnan(regression_array[:, 0])]
            regression_array = regression_array[~np.isnan(regression_array[:, 1])]
            reg = self.RegressionClass().fit(
                regression_array[:, :-1].reshape(-1, 1),
                regression_array[:, -1].reshape(-1, 1),
            )
            out_dict[f"m_{inpb}_only"] = [reg.coef_[0][0]]
            out_dict[f"c_{inpb}_only"] = [reg.intercept_[0]]


        # Stack all input and output
        regression_array = np.dstack([ind[inpb].copy() for inpb in self.input_bands] + [dep.copy()]).reshape(-1, 3)
        # Remove any pixels which have nan on ANY of the images (input or output)
        regression_array = regression_array[~np.isnan(regression_array[:, 0])]
        regression_array = regression_array[~np.isnan(regression_array[:, 1])]
        regression_array = regression_array[~np.isnan(regression_array[:, 2])]
        reg = self.RegressionClass().fit(regression_array[:, :-1].reshape(-1, 2),regression_array[:, -1].reshape(-1, 1))

        for j, inpb in enumerate(self.input_bands):
            out_dict[f"m_{inpb}_join"] = [reg.coef_[j][0]]
        out_dict["c_join"] = [reg.intercept_[0]]

        return out_dict



class LinearLSTRegression(RegressionOperation):

    RegressionClass = LinearRegression()


class LogisticUHIRegression(RegressionOperation):

    RegressionClass = LogisticRegression()



class ScatterIndex(StatisticalAnalysisOperation):
    def __init__(self, x_val, y_val):
        print("in")
        self.output_band = y_val
        self.input_bands = [x_val]
        self.stats_col_names = [x_val, y_val]
        super().__init__()

    def stats_op(self, ind, dep):

        out_dict = {}

        y_val = self.output_band
        x_val = self.input_bands[0]

        # Stack all input and output
        regression_array = np.dstack([ind[x_val].copy(), dep.copy()]).reshape(-1, 2)

        # Remove any pixels which have nan on ANY of the images (input or output)
        regression_array = regression_array[~np.isnan(regression_array[:, 0])]
        regression_array = regression_array[~np.isnan(regression_array[:, 1])]

        out_dict[x_val] = list(regression_array[:, 0])
        out_dict[y_val] = list(regression_array[:, 1])

        return out_dict
