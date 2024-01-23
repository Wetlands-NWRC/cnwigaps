import os

from pathlib import Path

import geopandas as gpd
import pandas as pd
from zipfile import ZipFile

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


def process_data_manifest(manifest: DataManifest):
    n_rows = 10_000
    # load data
    output = {}
    for _, group in manifest:
        # sub routine that loads the data before moving to processing
        ## Load data
        tmp, tmp_r = [], []
        for idx, row in group.iterrows():
            if row["type"] == 1 or row["type"] == 2:
                gdf = gpd.read_file(row["file_path"])
                gdf["type"] = row["type"]
                gdf["ECOZONE_ID"] = row["ECOZONE_ID"]
                tmp.append(gdf)
            if row["type"] == 3:
                gdf = gpd.read_file(row["file_path"])
                gdf["type"] = row["type"]
                gdf["ECOZONE_ID"] = row["ECOZONE_ID"]
                tmp_r.append(gdf)
        ## Process data

        ### Processing for training and validation data
        gdf = gpd.GeoDataFrame(pd.concat(tmp))
        gdf.to_crs(epsg=4326, inplace=True)

        labels = gdf["class_name"].unique().tolist()
        int_labels = list(range(1, len(labels) + 1))

        label_map = dict(zip(labels, int_labels))
        gdf["class_name"] = gdf["class_name"].map(label_map)

        # target number of observations per class
        n = n_rows // len(gdf["class_name"].unique().tolist())

        # if any class has n observations less then the target use that value for all labels
        grouped_counts = gdf.groupby("class_name").size().reset_index(name="Count")
        min_count = grouped_counts["Count"].min()

        if n > min_count:
            n = min_count

        selected = gdf.groupby("class_name").sample(n=n)
        selected = selected.reset_index(drop=True)

        ### Processing for region data
        gdf_r = gpd.GeoDataFrame(pd.concat(tmp_r))
        gdf_r.to_crs(epsg=4326, inplace=True)
        gdf_r = gdf_r.dissolve(by="ECOZONE_ID")

        output[idx] = {"t": selected, "r": gdf_r}
    return output


def split_and_zip(
    gdf: gpd.GeoDataFrame, groupby_col: str, where, file_prefix: str = None
) -> None:
    """
    Compresses and archives GeoDataFrame groups based on a specified column.

    Args:
        gdf (gpd.GeoDataFrame): The GeoDataFrame containing the data to be grouped and compressed.
        groupby_col (str): The column name to group the GeoDataFrame by.
        where: The directory path where the compressed files will be saved.
        file_prefix (str, optional): The prefix to be used for the compressed file names. Defaults to None.

    Returns:
        None
    """

    scratch = where / "scratch"
    scratch.mkdir(exist_ok=True)

    for _, group in gdf.groupby(groupby_col):
        # save each group
        file_prefix = file_prefix or "features"
        group.to_file(scratch / f"{file_prefix}_{_}.shp", driver="ESRI Shapefile")
        archive_name = f"{file_prefix}_{_}"
        # compress each dataset
        try:
            with ZipFile(scratch / f"{archive_name}.zip", "w") as zip:
                zip.write(scratch / f"{archive_name}.shp")
                zip.write(scratch / f"{archive_name}.dbf")
                zip.write(scratch / f"{archive_name}.prj")
                zip.write(scratch / f"{archive_name}.shx")
                zip.write(scratch / f"{archive_name}.cpg")
        except FileExistsError:
            continue

        # move the zip file to the processed directory
        (where / "zipped").mkdir(exist_ok=True)
        (scratch / f"{archive_name}.zip").rename(
            where / "zipped" / f"{archive_name}.zip"
        )

        # clean up the scratch directory
        (scratch / f"{archive_name}.shp").unlink()
        (scratch / f"{archive_name}.dbf").unlink()
        (scratch / f"{archive_name}.prj").unlink()
        (scratch / f"{archive_name}.shx").unlink()
        (scratch / f"{archive_name}.cpg").unlink()
        # output processed/shapefile AND processed/zipped

    # remove the scratch directory
    scratch.rmdir()
