
from utils import get_metadata


class Operation:
    def __init__(self):
        try:
            self.variables
        except AttributeError:
            self.variables = {}

        try:
            self.vidx
        except AttributeError:
            self.vidx = {}

    def group_operation(self, group):

        try:
            metadata = get_metadata(group)
        except FileNotFoundError:
            logging.warning("No MD json for", list(group.values())[0])
            return None
        # TODO: Add proper error messages here
        for var_name, var_route in self.vidx.items():
            m = metadata.copy()
            for var_idx in var_route:
                m = m[var_idx]
            self.variables[var_name] = m

        rasters = {k: read_raster(v) for k, v in group.items()}
        raster_meta = read_raster(list(group.values())[0])[1]

        try:
            output_image, output_meta = self.operation(rasters)
            if not isinstance(output_image, list):
                raise TypeError(
                    "Please return output image(s) as a list in operation method"
                )
            raster_meta = output_meta if output_meta != None else raster_meta
            output_path = get_property_path(group, self.band_name)
            save_calculated_raster(raster_meta, output_path, output_image)

        except (ValueError, KeyError) as e:
            print(list(group.values())[0])
            print(e)

    def __call__(self, root):

        groups = group_bands(root, self.bands_required)
        pool = Pool()
        pool.map(self.group_operation, groups)
        pool.close()
        pool.join()


class OperationPipeline:
    def __init__(self, sequence):
        self.operations = collections.OrderedDict()
        self.add_operations(sequence)

    # TODO: make saving optional?
    def __call__(self, root):
        for op_name, oper in tqdm(self.operations.items()):
            oper(root)

    def add_operations(self, operations: List[Tuple[str, Operation]]):
        for op in operations:
            self.operations[op.band_name] = op

    def show_examples(self, root, image_id, band_sequence=None):
        if not band_sequence:
            band_sequence = self.operations.keys()
        for band_name in band_sequence:
            if isinstance(band_name, str):
                band_name = [band_name]
            visualise_bands(root, image_id=image_id, bands=band_name, show=True)

    def show_overlay(
        self, root, image_id, overlay_band, underlay_bands=["B2", "B3", "B4"]
    ):
        overlay_image = visualise_bands(
            root, image_id=image_id, bands=overlay_band, show=False
        )
        underlay_image = visualise_bands(
            root, image_id=image_id, bands=underlay_bands, show=False
        )

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(underlay_image, alpha=1)
        ax.imshow(overlay_image, alpha=0.5)


class AggregationOperation:
    @staticmethod
    def replace_band_name(path, target_band):
        dot_list = path.split(".")
        pre = ".".join(dot_list[:-2])
        return f"{pre}.{target_band}.tif"

    def mapping_operation(self, time_mapping):

        # print(time_mapping)
        for time, path_list in tqdm(time_mapping.items()):
            array_dict = {v: read_raster(v) for v in path_list}

            self.operation(array_dict)

    def __call__(self, root):

        city_time_mapping = group_cities_by_time(root, self.band_required)

        print(f"Found {len(city_time_mapping)} cities in {root})")

        pool = Pool()
        pool.map(self.mapping_operation, city_time_mapping.values())
        pool.close()
        pool.join()


class StatisticalAnalysisOperation:
    def __init__(self):
        self.core_col_names = ["city", "hash", "datetime", "agg_value"]

    @staticmethod
    def replace_band_name(path, target_band):
        dot_list = path.split(".")
        pre = ".".join(dot_list[:-2])
        return f"{pre}.{target_band}.tif"

    def mapping_operation(self, time_mapping, aggregation_list):

        # print(time_mapping)
        for time, path_list in tqdm(time_mapping.items()):
            array_dict = {v: read_raster(v)[0] for v in path_list}
            try:
                self.operation(array_dict, aggregation_list)
            except rasterio.RasterioIOError as e:
                print(e)
                pass

    def operation(self, dependant_variable_dict, aggregation_list):

        independant_variable_dict = {
            original_path: {
                band: read_raster(self.replace_band_name(original_path, band))[0]
                for band in self.input_bands
            }
            for original_path in dependant_variable_dict.keys()
        }
        # i.e. input (array) = dependant_variable_dict[path]; output (dict) = independant_variable_dict[path]

        for agg_level in aggregation_list:

            dep = copy.deepcopy(dependant_variable_dict)
            for path, arr in dep.items():
                dep[path] = cv2.resize(
                    dep[path], (0, 0), fx=1 / agg_level, fy=1 / agg_level
                )

            for path, arr in dep.items():
                ind = copy.deepcopy(independant_variable_dict)

                for band, arr1 in ind[path].items():
                    ind[path][band] = cv2.resize(
                        arr1, (0, 0), fx=1 / agg_level, fy=1 / agg_level
                    )

                try:
                    aggregated_result = self.stats_op(ind[path], dep[path])
                    row = {
                        "city": [
                            path.split("/")[-1].split("--")[1]
                            for _ in range(len(list(aggregated_result.values())[0]))
                        ],
                        "hash": [
                            path.split("/")[-1].split("--")[0]
                            for _ in range(len(list(aggregated_result.values())[0]))
                        ],
                        "datetime": [
                            "--".join(
                                path.split("/")[-1].split(".")[0].split("--")[-2:]
                            )
                            for _ in range(len(list(aggregated_result.values())[0]))
                        ],
                        "agg_value": [
                            agg_level
                            for _ in range(len(list(aggregated_result.values())[0]))
                        ],
                    }
                    row.update(aggregated_result)
                    self.results = self.results.append(
                        pd.DataFrame(row), ignore_index=True
                    )
                except ValueError as e:
                    print(e)

    def __call__(self, root, aggregation_list=[1]):

        city_time_mapping = group_cities_by_time(root, self.output_band)
        self.results = pd.DataFrame(columns=self.core_col_names + self.stats_col_names)

        print(f"Found {len(city_time_mapping)} cities in {root}")

        for city_name, city_variable_dict in city_time_mapping.items():
            self.mapping_operation(city_variable_dict, aggregation_list)

        return self.results
