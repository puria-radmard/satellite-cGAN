import copy
import cv2

class Sampler:

    def __init__(self):
        pass

    def sampling_operation(self, dep_array, ind_dict):
        pass

    def sample_from_dicts(self, dependant_variable_dict, independent_variable_dict):
        path_dict = {}
        for path, dep_array in dependant_variable_dict.items():
            ind_dict = independent_variable_dict[path]
            sample_dict = self.sampling_operation(dep_array, ind_dict)
            path_dict[path] = sample_dict
        return path_dict


class WholeImageAggregationSampler(Sampler):

    agg_list = [1, 2, 3]

    def sampling_operation(self, dep_array, ind_dict):
        sample_dict = {}
        for agg_level in self.agg_list:
            dep = cv2.resize(copy.deepcopy(dep_array), (0, 0), fx=1 / agg_level, fy=1 / agg_level)
            ind = {
                band: cv2.resize(copy.deepcopy(arr), (0, 0), fx=1 / agg_level, fy=1 / agg_level)
                for band, arr in ind_dict.items()
            }
            sample_dict[f"agg{agg_level}"] = {"dep": dep, "ind": ind}
        return sample_dict
