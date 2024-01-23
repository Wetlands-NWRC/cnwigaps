import os

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
from zipfile import ZipFile

from cnwigaps.data.data import load_data


@dataclass
class TrainingData:
    """
    Represents training data.
    """

    data: gpd.GeoDataFrame


@dataclass
class RegionData:
    """
    Represents region data.
    """

    data: gpd.GeoDataFrame


class TrainingDataProcessor:
    def __init__(self) -> None:
        self.gdfs = []

    def add(self, gdf: gpd.GeoDataFrame) -> None:
        self.gdfs.append(gdf)
        return None

    def process(self, n_rows=10_000, balance: bool = False) -> TrainingData:
        gdf = gpd.GeoDataFrame(pd.concat(self.gdfs))
        gdf.to_crs(epsg=4326, inplace=True)

        labels = gdf["class_name"].unique().tolist()
        int_labels = list(range(1, len(labels) + 1))

        label_map = dict(zip(labels, int_labels))
        gdf["class_name"] = gdf["class_name"].map(label_map)

        if balance:
            # target number of observations per class
            n = n_rows // len(gdf["class_name"].unique().tolist())

            # if any class has n observations less then the target use that value for all labels
            grouped_counts = gdf.groupby("class_name").size().reset_index(name="Count")
            min_count = grouped_counts["Count"].min()

            if n > min_count:
                n = min_count

            selected = gdf.groupby("class_name").sample(n=n)
        else:
            selected = gdf.sample(n=n_rows)
        selected = selected.reset_index(drop=True)
        return TrainingData(selected)


class RegionDataProcessor:
    def __init__(self) -> None:
        self.gdfs = []

    def add(self, gdf: gpd.GeoDataFrame) -> None:
        self.gdfs.append(gdf)
        return None

    def process(self) -> RegionData:
        gdf = gpd.GeoDataFrame(pd.concat(self.gdfs))
        gdf.to_crs(epsg=4326, inplace=True)
        gdf = gdf.dissolve(by="ECOZONE_ID")
        return RegionData(gdf)


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


def process_data_manifest(
    manifest: DataManifest, n_rows: int = 10_000, balance: bool = False
) -> tuple[TrainingData, RegionData]:
    for _, group in manifest:
        # sub routine that loads the data before moving to processing
        ## Load data
        tdp = TrainingDataProcessor()
        rdp = RegionDataProcessor()
        for idx, row in group.iterrows():
            if row["type"] == 1 or row["type"] == 2:
                gdf = gpd.read_file(row["file_path"])
                gdf["type"] = row["type"]
                gdf["ECOZONE_ID"] = row["ECOZONE_ID"]
                tdp.add(gdf)
            if row["type"] == 3:
                gdf = gpd.read_file(row["file_path"])
                gdf["type"] = row["type"]
                gdf["ECOZONE_ID"] = row["ECOZONE_ID"]
                rdp.add(gdf)

    td = tdp.process(n_rows=n_rows, balance=balance)
    rd = rdp.process()

    return td, rd
