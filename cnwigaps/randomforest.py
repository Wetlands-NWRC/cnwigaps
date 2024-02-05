from __future__ import annotations
import sys

from typing import Any
from dataclasses import dataclass, InitVar

import ee


# Remote Sensing Datasets and Processing
S1 = "COPERNICUS/S1_GRD"
S2 = "COPERNICUS/S2"
ALOS = "JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH"
DEM = ""


@dataclass(frozen=True)
class DateRanges:
    s1 = [("2019-03-01", "2019-10-01")]
    s2 = [
        ("2018-03-01", "2018-10-01"),
        ("2019-03-01", "2019-10-01"),
        ("2020-03-01", "2020-10-01"),
    ]
    alos = [("2018", "2021")]


@dataclass
class S1:
    dataset: ee.ImageCollection = ee.ImageCollection(S1)
    bands: list = ["VV", "VH"]


@dataclass
class S2:
    dataset: ee.ImageCollection = ee.ImageCollection(S2)
    bands: list = ["B2", "B3", "B4", "B8"]


@dataclass
class ALOS:
    dataset: ee.ImageCollection = ee.ImageCollection(ALOS)
    bands: list = ["HH", "HV"]


@dataclass
class DEM:
    dataset: ee.Image = ee.Image(DEM)
    bands: list = ["elevation"]


# Dataset Preprocessing and processing Pipelines
def s1_preprocessing_pipeline(dataset: S1, aoi, start, end) -> ee.ImageCollection:
    """Preprocess Sentinel-1 image."""

    # apply preprocessing pipeline
    dataset.dataset = (
        dataset.dataset.filterDate(start, end)
        .filterBounds(aoi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        .filter(ee.Filter.eq("resolution_meters", 10))
        .filter(ee.Filter.eq("transmitterReceiverPolarization", ["VV", "VH"]))
        .select(dataset.bands)
    )

    return dataset


def s2_preprocessing_pipeline(dataset: S2, aoi, start, end) -> ee.ImageCollection:
    """Preprocess Sentinel-2 image."""

    # apply preprocessing pipeline
    dataset.dataset = (
        dataset.dataset.filterDate(start, end)
        .filterBounds(aoi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .select(dataset.bands)
    )

    return dataset


def alos_preprocessing_pipeline(dataset: ALOS, aoi, start, end) -> ee.ImageCollection:
    """Preprocess ALOS image."""

    dataset.dataset = (
        dataset.dataset.filterDate(start, end).filterBounds(aoi).select(dataset.bands)
    )
    return dataset


def sar_processing_pipeline(dataset: S1 | ALOS) -> ee.ImageCollection:
    """Apply processing pipeline to SAR dataset."""
    # apply processing pipeline
    compute_ratio = lambda x: x.select(dataset.bands[0]).divide(
        x.select(dataset.bands[1])
    )
    dataset.dataset = dataset.dataset.map(
        lambda x: x.convolve(ee.Kernel.square(1))
    ).map(compute_ratio)
    return dataset


def optical_processing_pipeline(dataset: S2) -> ee.ImageCollection:
    """Apply processing pipeline to Optical dataset."""

    def compute_savi(img: ee.Image):
        """computes soil adjusted vegetation index."""
        ...

    def compute_tasseled_cap(img: ee.Image):
        """computes tasseled cap transformation."""
        ...

    # apply processing pipeline
    dataset.dataset = (
        dataset.dataset.map(
            lambda x: x.normalizedDifference(["B8", "B4"]).rename("NDVI")
        )
        .map(compute_savi)
        .map(compute_tasseled_cap)
    )
    return dataset


def dem_processing_pipeline(dataset: DEM) -> ee.Image:
    """Apply processing pipeline to DEM dataset."""
    # apply processing pipeline
    elevation = dataset.dataset.select(dataset.bands)
    slope = ee.Terrain.slope(elevation)
    dataset.dataset = dataset.dataset.addBands(slope)
    return dataset


def processing_pipeline(dataset, aoi, dates=DateRanges()):
    """Apply processing pipeline to dataset."""
    # apply processing pipeline
    # construct year ranges
    if isinstance(dataset, S1):
        # target 2019 spring to fall
        # inacte the preprocessing pipeline
        dataset = s1_preprocessing_pipeline(dataset, aoi, *dates.s1[0])
        dataset = sar_processing_pipeline(dataset)
        return dataset

    elif isinstance(dataset, S2):
        # target 2018 - 2020 spring to fall
        for start, end in dates.s2:
            dataset = s2_preprocessing_pipeline(dataset, aoi, start, end)
            dataset = optical_processing_pipeline(dataset)
        return dataset

    elif isinstance(dataset, ALOS):
        # target 2018 - 2021 yearly composites
        dataset = alos_preprocessing_pipeline(dataset, aoi, *dates.alos[0])
        dataset = sar_processing_pipeline(dataset)
        return dataset
    else:
        # dem
        return dem_processing_pipeline(dataset)


def composite_images(datasets) -> ee.Image:
    """Composite images in the dataset."""
    for dataset in datasets:
        if (
            isinstance(dataset, S1)
            or isinstance(dataset, S2)
            or isinstance(dataset, ALOS)
        ):
            dataset.dataset = dataset.dataset.median()
        else:
            dataset.dataset = dataset.dataset
    return None


def stack_images(datasets) -> ee.Image:
    """Stack images in the dataset."""
    return ee.Image.cat(*datasets)


# Remote Sensing Datasets and Processing ends here
#########################################################################################


# Data Engineering and Feature Extraction
@dataclass
class Features:
    dataset: str  # TODO make init var
    label_col: str

    def __post_init__(self):
        self.dataset = ee.FeatureCollection(self.dataset)


def add_label_into_features(features: Features) -> Features:
    """adds a column to the features dataset."""
    class_labels = features.dataset.aggregate_array(features.label_col).distinct()
    class_ints = ee.List.sequence(1, class_labels.size)

    features.dataset = features.dataset.remap(
        class_labels, class_ints, features.label_col
    )
    return features


def extract(image, features: Features, **kwargs) -> Features:
    """Extract features from the image."""
    # extract features

    features.dataset = image.sampleRegions(
        collection=features.dataset,
        propertie=kwargs.get("properties", []),
        scale=kwargs.get("scale", 10),
        tileScale=kwargs.get("tileScale", 16),
        geometries=kwargs.get("geometries", True),
    )
    return features


# Data Engineering and Feature Extraction ends here
#########################################################################################


# Model Training and Evaluation
@dataclass
class Hyperparameters:
    numberOfTrees: int
    variablesPerSplit: int = None
    minLeafPopulation: int = 1
    bagFraction: float = 0.5
    maxNodes: int = None
    seed: int = 0


@dataclass
class Metrics:
    matrix: InitVar[Any]
    accuracy: InitVar[Any]
    producers_accuracy: InitVar[Any]
    consumers_accuracy: InitVar[Any]

    def __post_init__(self, matrix, accuracy, producers_accuracy, consumers_accuracy):
        self.components = ee.FeatureCollection(
            [
                ee.Feature(None, {"matrix": matrix}),
                ee.Feature(None, {"accuracy": accuracy}),
                ee.Feature(None, {"producers_accuracy": producers_accuracy}),
                ee.Feature(None, {"consumers_accuracy": consumers_accuracy}),
            ]
        )

    def save(self, file_name: str, folder: str = None) -> ee.Batch.Task:
        """exports to google drive."""
        task = ee.batch.Export.table.toDrive(
            collection=self.components,
            description="",
            fileFormat="GeoJSON",
            fileNamePrefix=file_name,
            folder=folder,
        )
        task.start()
        return task


class SmileRandomForest:
    def __init__(self, pramas: Hyperparameters) -> None:
        self.params = pramas
        self._model = None

    def fit(
        self, features: Features, predictors: list[str] | ee.List
    ) -> SmileRandomForest:
        """Train the model."""
        # train the model
        self._model = ee.Classifier.smileRandomForest(
            numberOfTrees=self.params.numberOfTrees,
            variablesPerSplit=self.params.variablesPerSplit,
            minLeafPopulation=self.params.minLeafPopulation,
            bagFraction=self.params.bagFraction,
            maxNodes=self.params.maxNodes,
            seed=self.params.seed,
        ).train(
            features=features.dataset,
            classProperty=features.lable_col,
            inputProperties=predictors,
        )
        return self

    def predict(self, data):
        return self._model.classify(data)

    def assess(self, data) -> Metrics:
        prdicted = self.predict(data)
        if isinstance(data, Features):
            order = data.dataset.aggregate_array(data.label_col).distinct()
            cfm = prdicted.errorMatrix(data.label_col, "classification", order)

        accuracy = cfm.accuracy()
        producers_accuracy = cfm.producersAccuracy()
        consumers_accuracy = cfm.consumersAccuracy()

        return Metrics(cfm, accuracy, producers_accuracy, consumers_accuracy)

    def save(self, asset_id: str) -> ee.Batch.Task:
        """Save the model."""
        task = ee.batch.Export.classifier.toAsset(
            model=self._model,
            description="RandomForestModel",
            assetId=asset_id,
        )
        task.start()
        return task


# Model Training and Evaluation ends here
#########################################################################################


# Utility functions
def monitor_task(task: ee.Task) -> None:
    import time

    """Monitor task progress."""
    while task.status()["state"] in ["READY", "RUNNING"]:
        print(task.status())
        time.sleep(5)
    print(task.status())


def main(args: list[str]) -> int:
    return 0


if __name__ == "__main__":
    ee.Initialize()
    sys.exit(main(sys.argv[1:]))
