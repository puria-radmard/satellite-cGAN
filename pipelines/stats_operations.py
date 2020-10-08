from utils import *
from root_classes import *
from imports import *


class LinearLSTRegression(StatisticalAnalysisOperation):

  stats_col_names = ["m_V_only", "c_V_only", "m_B_only", "c_B_only", "m_V_join", "m_B_join", "c_join"]
  output_band = "LST"
  input_bands = ["NDVI", "NDBI"]


  def stats_op(self, ind, dep):

    out_dict = {}
    
    # TODO: functionise these please

    # Stack all input and output
    regression_array = np.dstack(
        [ind["NDVI"].copy(), dep.copy()]
    ).reshape(-1, 2)


    # Remove any pixels which have nan on ANY of the images (input or output)
    regression_array = regression_array[~np.isnan(regression_array[:,0])]
    regression_array = regression_array[~np.isnan(regression_array[:,1])]
    reg = LinearRegression().fit(
        regression_array[:,:-1].reshape(-1, 1),
        regression_array[:,-1].reshape(-1, 1),
    )
    out_dict["m_V_only"] = [reg.coef_[0][0]]
    out_dict["c_V_only"] = [reg.intercept_[0]]

    regression_array = np.dstack(
        [ind["NDBI"].copy(), dep.copy()]
    ).reshape(-1, 2)
    regression_array = regression_array[~np.isnan(regression_array[:,0])]
    regression_array = regression_array[~np.isnan(regression_array[:,1])]
    reg = LinearRegression().fit(
        regression_array[:,:-1].reshape(-1, 1),
        regression_array[:,-1].reshape(-1, 1),
    )
    out_dict["m_B_only"] = [reg.coef_[0][0]]
    out_dict["c_B_only"] = [reg.intercept_[0]]

    # Stack all input and output
    regression_array = np.dstack(
        [ind["NDVI"].copy(), ind["NDBI"].copy(), dep.copy()]
    ).reshape(-1, 3)
    # Remove any pixels which have nan on ANY of the images (input or output)
    regression_array = regression_array[~np.isnan(regression_array[:,0])]
    regression_array = regression_array[~np.isnan(regression_array[:,1])]
    regression_array = regression_array[~np.isnan(regression_array[:,2])]
    reg = LinearRegression().fit(
        regression_array[:,:-1].reshape(-1, 2),
        regression_array[:,-1].reshape(-1, 1),
    )
    out_dict["m_V_join"] = [reg.coef_[0][0]]     ## NDVI FIRST
    out_dict["m_B_join"] = [reg.coef_[0][1]]
    out_dict["c_join"] = [reg.intercept_[0]]

    return out_dict


class LogisticUHIRegression(StatisticalAnalysisOperation):

  stats_col_names = ["m_V_only", "c_V_only", "m_B_only", "c_B_only", "m_V_join", "m_B_join", "c_join"]
  output_band = "UHI"
  input_bands = ["NDVI", "NDBI"]


  def stats_op(self, ind, dep):

    out_dict = {}
    
    # TODO: functionise these please
    dep = np.vectorize(
            lambda x: np.nan if np.isnan(x) else int(round(x))
          )(dep)

    # Stack all input and output
    regression_array = np.dstack(
        [ind["NDVI"].copy(), dep.copy()]
    ).reshape(-1, 2)

    # Remove any pixels which have nan on ANY of the images (input or output)
    regression_array = regression_array[~np.isnan(regression_array[:,0])]
    regression_array = regression_array[~np.isnan(regression_array[:,1])]
    reg = LogisticRegression().fit(
        regression_array[:,:-1].reshape(-1, 1),
        regression_array[:,-1].reshape(-1, 1),
    )
    out_dict["m_V_only"] = [reg.coef_[0][0]]
    out_dict["c_V_only"] = [reg.intercept_[0]]

    regression_array = np.dstack(
        [ind["NDBI"].copy(), dep.copy()]
    ).reshape(-1, 2)
    regression_array = regression_array[~np.isnan(regression_array[:,0])]
    regression_array = regression_array[~np.isnan(regression_array[:,1])]
    reg = LogisticRegression().fit(
        regression_array[:,:-1].reshape(-1, 1),
        regression_array[:,-1].reshape(-1, 1),
    )
    out_dict["m_B_only"] = [reg.coef_[0][0]]
    out_dict["c_B_only"] = [reg.intercept_[0]]

    # Stack all input and output
    regression_array = np.dstack(
        [ind["NDVI"].copy(), ind["NDBI"].copy(), dep.copy()]
    ).reshape(-1, 3)
    # Remove any pixels which have nan on ANY of the images (input or output)
    regression_array = regression_array[~np.isnan(regression_array[:,0])]
    regression_array = regression_array[~np.isnan(regression_array[:,1])]
    regression_array = regression_array[~np.isnan(regression_array[:,2])]
    reg = LogisticRegression().fit(
        regression_array[:,:-1].reshape(-1, 2),
        regression_array[:,-1].reshape(-1, 1),
    )
    out_dict["m_V_join"] = [reg.coef_[0][0]]     ## NDVI FIRST
    out_dict["m_B_join"] = [reg.coef_[0][1]]
    out_dict["c_join"] = [reg.intercept_[0]]

    return out_dict


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
    regression_array = np.dstack(
        [ind[x_val].copy(), dep.copy()]
    ).reshape(-1, 2)

    # Remove any pixels which have nan on ANY of the images (input or output)
    regression_array = regression_array[~np.isnan(regression_array[:,0])]
    regression_array = regression_array[~np.isnan(regression_array[:,1])]
    
    out_dict[x_val] = list(regression_array[:,0])
    out_dict[y_val] = list(regression_array[:,1])

    return out_dict