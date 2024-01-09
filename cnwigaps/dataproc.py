import os

from pathlib import Path

import geopandas as gpd
import pandas as pd

from cnwigaps.data.data import load_data


class DataManifest:
    """
    Represents a data manifest that contains information about shapefiles in a directory.
    """

    def __init__(self, root: str):
        """
        Initializes a DataManifest object.

        Args:
            root (str): The root directory path.
        """
        self.manifest = Path(root)
        self.groupby_col = "ECOZONE_ID"

    def __iter__(self):
        """
        Iterates over the data manifest.

        Yields:
            tuple: A tuple containing the Eco region id and the group of shapefiles.
        """
        for idx, group in self.manifest.groupby(self.groupby_col):
            yield idx, group

    @property
    def manifest(self):
        """
        Gets the data manifest.

        Returns:
            pd.DataFrame: The data manifest.
        """
        return self._manifest

    @manifest.setter
    def manifest(self, root: str):
        """
        Sets the data manifest.

        Args:
            root (str): The root directory path.
        """
        self._manifest = self._create_manifest(root)

    @staticmethod
    def _create_manifest(root: Path) -> pd.DataFrame:
        """
        Creates the data manifest.

        Args:
            root (Path): The root directory path.

        Returns:
            pd.DataFrame: The data manifest.
        """
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
    """
    Represents a data processor that performs various operations on shapefile data.
    """

    def __init__(self, manifest: DataManifest):
        """
        Initializes a DataProcessor object.

        Args:
            manifest (DataManifest): The data manifest.
        """
        self.manifest = manifest
        self.training = {}
        self.regions = {}

    def load_files(self):
        """
        Loads the shapefile data.
        """
        for idx, group in self.manifest:
            # training_files
            tmp = []
            for _, row in group.iterrows():
                if row["type"] == 1 or row["type"] == 2:
                    pass

    def convert_crs(self):
        """
        Converts the coordinate reference system (CRS) of the shapefile data.
        """
        pass

    def combine_data(self):
        """
        Combines the shapefile data into a single geopandas dataframe.
        """
        pass

    def get_lookup(self, column: str):
        """
        Gets a lookup dictionary for a specific column in the shapefile data.

        Args:
            column (str): The column name.

        Returns:
            dict: The lookup dictionary.
        """
        pass

    def remap_labels(self, label_col: str, lookup: dict[str, str] = None):
        """
        Remaps labels in a specific column of the shapefile data using a lookup dictionary.

        Args:
            label_col (str): The column name containing the labels.
            lookup (dict[str, str], optional): The lookup dictionary. Defaults to None.
        """
        pass

    def process(self):
        """
        Processes the shapefile data.
        """
        pass


def mk_data_manifest(data_dir: str) -> pd.DataFrame:
    """
    Creates a data manifest.

    Args:
        data_dir (str): The directory path containing the shapefiles.

    Returns:
        pd.DataFrame: The data manifest.
    """
    pass


def process_data_manifest(data_manifest: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Processes the data manifest.

    Args:
        data_manifest (pd.DataFrame): The data manifest.

    Returns:
        gpd.GeoDataFrame: The processed data.
    """
    pass
