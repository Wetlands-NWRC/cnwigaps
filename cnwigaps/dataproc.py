import os

from pathlib import Path

import geopandas as gpd
import pandas as pd

from cnwigaps.data.data import load_data

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
        self.manifest = Path(root)
        self.groupby_col = "ECOZONE_ID"

    def __iter__(self):
        for idx, group in self.manifest.groupby(self.groupby_col):
            yield idx, group

    @property
    def manifest(self):
        return self._manifest

    @manifest.setter
    def manifest(self, root: str):
        self._manifest = self._create_manifest(root)

    @staticmethod
    def _create_manifest(root: Path) -> pd.DataFrame:
        """creates the data manifest"""
        files = list(root.glob("**/*.shp"))
        M = {"trainingPoints": 1, "validationPoints": 2, "region": 3}
        manifest = pd.DataFrame(files, columns=["file_path"])
        manifest["ECOREGION_ID"] = (
            manifest["file_path"].astype(str).str.extract(r"(\b\d{1,3}\b)").astype(int)
        )
        manifest["type"] = (
            manifest["file_path"]
            .astype(str)
            .str.extract(r"(\btrainingPoints\b|\bvalidationPoints\b|\bregion\b)")
        )
        manifest["type"] = manifest["type"].map(M)

        manifest = manifest[manifest["file_path"].astype(str).str.contains(r"(\.shp)")]

        manifest = manifest.sort_values(by=["ECOREGION_ID", "type"])
        manifest = manifest.reset_index(drop=True)

        eco = load_data()

        manifest = pd.merge(manifest, eco, on="ECOREGION_ID", how="inner")

        return manifest[["file_path", "ECOREGION_ID", "type", "ECOZONE_ID"]]


class DataProcessor:
    def __init__(self, manifest: DataManifest):
        self.manifest = manifest
        self.training = {}
        self.regions = {}

    def load_files(self):
        # load the training file
        for idx, group in self.manifest:
            # training_files
            tmp = []
            for _, row in group.iterrows():
                if row["type"] == 1 or row["type"] == 2:
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
