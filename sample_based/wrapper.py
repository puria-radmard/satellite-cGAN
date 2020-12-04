from tqdm import tqdm
import sys
from stats_operations import LinearLSTRegression, LogisticUHIRegression
from sampler import WholeImageAggregationSampler
from s_utils import group_cities_by_time
import pandas as pd
import rasterio

SAMPLING_DICT = {"aggregation": WholeImageAggregationSampler}
OPERATION_DICT = {"linreg": LinearLSTRegression, "logreg": LogisticUHIRegression}


class SampleOperationWrapper:

    def __init__(self, operation_type, sampler_type):

        self.operator = OPERATION_DICT[operation_type]()
        self.sampler = SAMPLING_DICT[sampler_type]()

    def mapping_operation(self, time_mapping):

        # print(time_mapping)
        for time, path_list in tqdm(time_mapping.items()):
            try:
                self.operator.operation(path_list, self.sampler)
            except rasterio.RasterioIOError as e:
                print(e)

    def __call__(self, root):

        city_time_mapping = group_cities_by_time(root, self.operator.output_band)
        self.operator.results = pd.DataFrame(columns=self.operator.core_col_names + self.operator.stats_col_names)
        print(f"Found {len(city_time_mapping)} cities in {root}")

        for city_name, city_variable_dict in city_time_mapping.items():
            self.mapping_operation(city_variable_dict)

        return self.operator.results


if __name__ == '__main__':

    sample_operation = SampleOperationWrapper(
        operation_type="linreg",
        sampler_type="aggregation"
    )
    results = sample_operation(sys.argv[1])
    results.to_csv("aasdfasdf")
