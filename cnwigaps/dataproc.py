import os
import geopandas as gpd
import pandas as pd


"""
get data -> needs to get all shapefile paths under the data directory
    - needs to be sorted by into train, validation and region
load the data -> needs to load the shapefiles into a geopandas dataframe
 - on load we need to add a column for the Eco region id and the data type (train, validation and region)

convert the crs to epsg:4326 -> needs to convert the crs to epsg:4326

combine the data -> needs to combine the data into a single geopandas dataframe

{"gdf": loaded file, "region_id": int, "data_type": int}


"""


class DataManifest:
    def __init__(self, root: str):
        self.root = root
        self.manifest = None

    def get_files(self):
        pass


class DataProcessor:
    def __init__(self, data_dir: str):
        self.manifest = None
        self.train = []
        self.test = []
        self.region = []
        self.gdf = None
        self.lookup = None

    def get_data_paths(self) -> list[str]:
        """gets all the files in the data directory"""
        pass

    def load_data_paths(self):
        # load the training file
        for file in self.get_files():
            pass
        pass

    def convert_crs(self):
        pass

    def combine_data(self):
        pass

    def get_lookup(self, column: str):
        pass

    def remap_labels(self, label_col: str, lookup: dict[str, str] = None):
        pass

    def process(self):
        pass


def mk_data_manifest(data_dir: str) -> pd.DataFrame:
    """creates the data manifest"""
    pass


def process_data_manifest(data_manifest: pd.DataFrame) -> gpd.GeoDataFrame:
    """processes the data manifest"""
    pass
