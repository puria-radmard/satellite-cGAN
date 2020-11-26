from utils import *
from root_classes import *
from imports import *
import shutil


class EarthEngineDownloader:
    """
    Downloads data straight to Google Drive. Currently only works for Landsat 8 :/
    Example usage:

      eed = EarthEngineDownloader()
      eed("example_dataset.json", multiple_mode = True)

    for now, just keep multiple_mode as True
    """

    def __init__(self):

        pass
        # try:
        #    ee.Initialize()
        # except ee.EEException:
        #    ee.Authenticate()
        #    ee.Initialize()

    @staticmethod
    def read_dataset_geojson(geojson_addr: str) -> dict:
        with open(geojson_addr, "r") as gjfile:
            gjinfo = json.load(gjfile)
        return gjinfo

    @staticmethod
    def generate_metadata_json(dataset_dir, image_description, image_information):
        # Await folder to be created - clean this code
        for i in range(20):
            try:
                with open(
                    os.path.join(dataset_dir, f"{image_description}.METADATA.json"), "w"
                ) as jfile:
                    json.dump(image_information, jfile)
                    return None
            except FileNotFoundError:
                print(
                    "Awaiting folder creation for",
                    dataset_dir,
                    image_description,
                    "." * i,
                )
                time.sleep(3)
        raise FileNotFoundError("Unknown timeout from GEE")

    @staticmethod
    def prepare_download_info(geometry, dataset_properties, raw_image_collection):

        dates = dataset_properties["dates"]
        image_collection = raw_image_collection.filterBounds(geometry).filterDate(
            dates[0], dates[1]
        )

        all_bands = image_collection.first().bandNames().getInfo()
        bands = list(set(dataset_properties["bands"]).intersection(all_bands))
        bands.sort(key=lambda b: dataset_properties["bands"].index(b))

        collection_list = image_collection.toList(image_collection.size())
        collection_size = collection_list.size().getInfo()

        region = geometry.getInfo()["coordinates"]
        dataset_dir = dataset_properties["area_id"]

        return {
            "collection_list": collection_list,
            "collection_size": collection_size,
            "bands": bands,
            "dataset_dir": dataset_dir,
            "region": region,
        }

    def download_area(
        self,
        geometry,
        geometry_id,
        dataset_properties,
        raw_image_collection,
        multiple_mode=True,
    ):

        exampledownload_info = self.prepare_download_info(
            ee.Geometry.Polygon(geometry), dataset_properties, raw_image_collection
        )
        exampleimage = ee.Image(exampledownload_info["collection_list"].get(0))
        exampleimage_information = exampleimage.getInfo()
        end_code = exampleimage_information["bands"][0]["crs"]

        if not multiple_mode:
            # i.e. if it's a single place, we don't split into grids
            grid_geoms = {get_geohash(geometry): geometry}
        else:
            grid_geoms = {
                get_geohash(geom): geom for geom in split_into_grid(geometry, end_code)
            }

        # Remove index!!!!
        for square_geohash, square_geom in list(grid_geoms.items())[9:]:

            download_info = self.prepare_download_info(
                square_geom, dataset_properties, raw_image_collection
            )

            bands = download_info["bands"]
            region = download_info["region"]
            dataset_dir = download_info["dataset_dir"]
            collection_list = download_info["collection_list"]
            collection_size = download_info["collection_size"]

            area_id = dataset_properties["area_id"]
            cloud_cover_thres = dataset_properties["cloud_cover_thres"]
            cloud_access_path = dataset_properties["cloud_access_path"]

            for j in tqdm(range(collection_size)):

                image = ee.Image(collection_list.get(j))
                image_information = image.getInfo()
                image_millis = int(image_information["properties"]["system:time_start"])
                image_datetime = datetime.fromtimestamp(
                    int(image_millis) / 1000.0
                ).strftime("%Y-%m-%d--%H-%M-%S")
                image_description = f"{square_geohash}--{geometry_id}--{image_datetime}"

                m = image_information.copy()
                for k in cloud_access_path:
                    m = m[k]
                if float(m) > cloud_cover_thres:
                    continue
                print(f"Downloading {image_description} with cloud cover {m}")

                for band in bands:

                    description = image_description + f".{band}"

                    task = ee.batch.Export.image.toDrive(
                        image=image.select(band),
                        description=description,
                        folder=dataset_dir,
                        region=region,
                    )
                    task.start()

                self.generate_metadata_json(
                    dataset_dir=dataset_dir,
                    image_description=image_description,
                    image_information=image_information,
                )

    def __call__(self, geojson_addr: str, multiple_mode):
        # multiple_mode = False if we want to download full extent of geometries
        # multiple_mode = True if we only want grids in the area of interest

        gjinfo = self.read_dataset_geojson(geojson_addr)

        if not multiple_mode:
            bounds = get_area_bbox(gjinfo)
            geometry = {gjinfo["properties"]["area_id"]: ee.Geometry.Polygon(bounds)}

        else:
            geometry = {
                b["id"]: b["geometry"]["coordinates"] for b in gjinfo["features"]
            }

        dataset_properties = gjinfo["properties"]
        raw_image_collection = ee.ImageCollection(dataset_properties["earth_engine_id"])

        download_operation = lambda gid, g: self.download_area(
            geometry=g,
            geometry_id=gid,
            dataset_properties=dataset_properties,
            raw_image_collection=raw_image_collection,
            multiple_mode=multiple_mode,
        )

        for geom_id, geom in tqdm(geometry.items()):
            print(f"Downloading from {geom_id}")
            download_operation(geom_id, geom)

    def find_top_left_coord(self, image, window_size=256):
        for i in range(len(image)):
            row = image[i]
            non_nan_elements = row[row == row]
            if len(non_nan_elements) >= window_size:
                y_max = i
                break
        x_min = np.where(~np.isnan(image[y_max]))[0].min()
        x_max = np.where(~np.isnan(image[y_max]))[0].max()
        return x_min, x_max, y_max

    def find_bottom_right_coord(self, image, window_size=256):
        Y, X = tuple(a - 1 for a in image.shape)
        for i in range(len(image)):
            row = image[Y - i]
            non_nan_elements = row[row == row]
            if len(non_nan_elements) >= window_size:
                y_min = i
                break
        x_min = np.where(~np.isnan(image[y_min]))[0].min()
        x_max = np.where(~np.isnan(image[y_min]))[0].max()
        return x_min, x_max, y_min

    def roll_window_horizontally(self, image, y_max, stride=128, window_size=256):
        if image.shape[0] < 256 or image.shape[0] < 256:
            return []
        x_min_bottom, x_max_bottom, _ = self.find_top_left_coord(
            image[y_max : y_max + window_size, :], window_size=window_size
        )
        x_min_top, x_max_top, _ = self.find_bottom_right_coord(
            image[y_max : y_max + window_size, :], window_size=256
        )
        x_max = min([x_max_bottom, x_max_top])  # ; print([x_max_bottom, x_max_top])
        x_min = max([x_min_bottom, x_min_top])  # ; print([x_min_bottom, x_min_top])
        x_mins = [x_min + x * stride for x in range((x_max - x_min) // stride - 1)]
        return x_mins

    def roll_window_vertically(self, image, stride=128, window_size=256):
        aug_image = image.copy()
        if image.shape[0] < 256 or image.shape[0] < 256:
            return []
        _, _, y_top = self.find_top_left_coord(image, window_size=window_size)
        windows = []
        while y_top <= image.shape[0] - window_size:
            xmins = self.roll_window_horizontally(
                aug_image, y_top, stride=stride, window_size=window_size
            )
            windows.extend([y_top, x_min, window_size] for x_min in xmins)
            y_top += stride
        return windows

    def purge_windows(self, image, windows):
        purged_windows = []
        for j, (y_max, x_min, window_size) in enumerate(windows):
            window = image[y_max : y_max + window_size, x_min : x_min + window_size]
            aug_image = image.copy()
            aug_image[y_max : y_max + window_size, x_min : x_min + window_size] = 0
            if np.isnan(window.sum()):
                continue
            else:
                purged_windows.append((y_max, x_min, window_size))
        return purged_windows

    def split_image(self, image, stride, window_size):
        windows = self.roll_window_vertically(image, stride, window_size)
        return self.purge_windows(image, windows)

    def split_directory_images(self, earch_dir, destination_dir, window_size, stride):

        if "*" not in search_dir:
            raise ValueError("Need '*' in search_dir!")

        tif_list = glob(f"{search_dir}*.tif")

        for tif_path in tqdm(tif_list):
            raster, metadata = read_raster(tif_path, remove_zero=True)
            # destination_path = os.path.join(destination_dir, tif_path.split("/")[-1])
            base_image_name = tif_path.split("/")[-1]
            windows = self.split_image(raster, stride, window_size)
            for y_max, x_min, window_size in windows:

                try:

                    destination_path = os.path.join(
                        destination_dir,
                        f"{y_max}-{x_min}-{window_size}--{base_image_name}",
                    )
                    # new_image = raster[np.newaxis,y_max:y_max+window_size,x_min:x_min+window_size]
                    # metadata["height"] = window_size
                    metadata["width"] = window_size
                    # save_calculated_raster(metadata, destination_path, new_image)

                    original_json_name = (
                        ".".join(tif_path.split(".")[:-2]) + ".METADATA.json"
                    )
                    destination_json_name = (
                        ".".join(destination_path.split(".")[:-2]) + ".METADATA.json"
                    )
                    shutil.copy(original_json_name, destination_json_name)

                except:

                    print("skipping", tif_path)
                    pass


if __name__ == "__main__":

    city = "LONDON"  # ["MANCHESTER", "BIRMINGHAM", "LEEDS", "PARIS", "LYON"]
    months = ["-03-", "-04-", "-05-", "-06-", "-07-", "-08-"]
    destination_dir = "../../data_source/SUMMER_LONDON_DATASET"
    dl = EarthEngineDownloader()

    for month in months:
        print("CONVERTING FROM", month)
        search_dir = (
            f"../../data_source/WHOLE_LONDON_DATASET/*{city}*--201*{month}*--*-*-*.*"
        )
        dl.split_directory_images(search_dir, destination_dir, 256, 128)
