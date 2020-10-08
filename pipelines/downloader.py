from utils import *
from root_classes import *
from imports import *


class EarthEngineDownloader:
  """
  Downloads data straight to Google Drive. Currently only works for Landsat 8 :/
  Example usage:

    eed = EarthEngineDownloader()
    eed("example_dataset.json", multiple_mode = True)

  for now, just keep multiple_mode as True
  """

  def __init__(self):

    try:
      ee.Initialize()
    except ee.EEException:
      ee.Authenticate()
      ee.Initialize()

  @staticmethod
  def read_dataset_geojson(geojson_addr: str) -> dict:
    with open(geojson_addr, "r") as gjfile:
        gjinfo = json.load(gjfile)
    gjinfo = cambs_geojson.copy()
    return gjinfo

  @staticmethod
  def generate_metadata_json(dataset_dir, image_description, image_information):
      # Await folder to be created - clean this code
      for i in range(20):
        try:
          with open(os.path.join(dataset_dir, f"{image_description}.METADATA.json"), "w") as jfile:
            json.dump(image_information, jfile)
            return None
        except FileNotFoundError:
          print("Awaiting folder creation for", dataset_dir, image_description, "."*i)
          time.sleep(3)
      raise FileNotFoundError("Unknown timeout from GEE")

  @staticmethod
  def prepare_download_info(geometry, dataset_properties, raw_image_collection):

    dates = dataset_properties["dates"]
    image_collection = raw_image_collection.filterBounds(geometry).filterDate(dates[0], dates[1])

    all_bands = image_collection.first().bandNames().getInfo()
    bands = list(set(dataset_properties["bands"]).intersection(all_bands))
    bands.sort(key = lambda b: dataset_properties["bands"].index(b))

    collection_list = image_collection.toList(image_collection.size())
    collection_size = collection_list.size().getInfo()

    region = geometry.getInfo()['coordinates']
    dataset_dir = dataset_properties["area_id"]

    return ({
        "collection_list": collection_list,
        "collection_size": collection_size,
        "bands": bands,
        "dataset_dir": dataset_dir,
        "region": region
    })


  def download_area(self, geometry, geometry_id, dataset_properties, raw_image_collection, multiple_mode = True):

    exampledownload_info = self.prepare_download_info(ee.Geometry.Polygon(geometry), dataset_properties, raw_image_collection)
    exampleimage = ee.Image(exampledownload_info["collection_list"].get(0))
    exampleimage_information = exampleimage.getInfo()
    end_code = exampleimage_information['bands'][0]['crs']

    if not multiple_mode:
      # i.e. if it's a single place, we don't split into grids
      grid_geoms = {get_geohash(geometry): geometry}
    else:
      grid_geoms = {get_geohash(geom): geom for geom in split_into_grid(geometry, end_code)}

    # Remove index!!!!
    for square_geohash, square_geom in list(grid_geoms.items())[9:]:

      download_info = self.prepare_download_info(square_geom, dataset_properties, raw_image_collection)

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
          image_millis = int(image_information['properties']['system:time_start'])
          image_datetime = datetime.fromtimestamp(int(image_millis)/1000.0).strftime("%Y-%m-%d--%H-%M-%S")
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
            dataset_dir = dataset_dir,
            image_description = image_description,
            image_information=image_information
          )


  def __call__(self, geojson_addr: str, multiple_mode):
    # multiple_mode = False if we want to download full extent of geometries
    # multiple_mode = True if we only want grids in the area of interest
    
    gjinfo = self.read_dataset_geojson(geojson_addr)
    
    if not multiple_mode:
      bounds = get_area_bbox(gjinfo)
      geometry = {gjinfo["properties"]["area_id"]: ee.Geometry.Polygon(bounds)}
      
    else:
      geometry = {b["id"]: b["geometry"]["coordinates"] for b in gjinfo["features"]}

    dataset_properties = gjinfo["properties"]
    raw_image_collection = ee.ImageCollection(dataset_properties["earth_engine_id"])

    download_operation = lambda gid, g: self.download_area(geometry = g, geometry_id = gid, dataset_properties = dataset_properties, raw_image_collection=raw_image_collection, multiple_mode=multiple_mode)

    for geom_id, geom in tqdm(geometry.items()):
      print(f"Downloading from {geom_id}")
      download_operation(geom_id, geom)
