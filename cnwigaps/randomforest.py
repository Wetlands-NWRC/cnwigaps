from __future__ import annotations
import sys

from typing import Any
from dataclasses import dataclass, InitVar, field

import ee


# Remote Sensing Datasets and Processing
@dataclass
class Sentinel1GRD:
    dataset_id: str = field(default="COPERNICUS/S1_GRD")
    bands: list = field(default_factory=lambda: ["VV", "VH"])
    date: list = field(default_factory=lambda: [("2019-03-01", "2019-10-01")])

    def __post_init__(self):
        self.dataset = ee.ImageCollection(self.dataset_id)


@dataclass
class Sentinel2TOA:
    dataset_id: str = field(default="COPERNICUS/S2")
    bands: list = field(default_factory=lambda: ["B2", "B3", "B4", "B8"])
    date: list = field(
        default_factory=lambda: [
            ("2019-03-01", "2019-10-01"),
            ("2019-03-01", "2019-10-01"),
            ("2019-03-01", "2019-10-01"),
        ]
    )

    def __post_init__(self):
        self.dataset = ee.ImageCollection(self.dataset_id)


@dataclass
class ALOS:
    dataset_id: str = field(default="JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH")
    bands: list = field(default_factory=lambda: ["HH", "HV"])
    date: list = field(default_factory=lambda: [("2018", "2021")])

    def __post_init__(self):
        self.dataset = ee.ImageCollection(self.dataset_id)


@dataclass
class DEM:
    dataset_id: str = field(default="USGS/SRTMGL1_003")
    bands: list = field(default_factory=lambda: ["elevation"])

    def __post_init__(self):
        self.dataset = ee.Image(self.dataset_id)


def sar_processing(dataset) -> ee.ImageCollection:
    """Apply processing pipeline to SAR dataset."""
    # apply processing pipeline
    compute_ratio = lambda x: x.select(dataset.bands[0]).divide(
        x.select(dataset.bands[1])
    )
    dataset.dataset = dataset.dataset.map(
        lambda x: x.convolve(ee.Kernel.square(1))
    ).map(compute_ratio)
    return dataset


def sentinel_1_processor(rsd, aoi, start, end, bands):
    compute_ratio = lambda x: x.select("VV").divide(x.select("VH"))

    return (
        rsd.filterDate(start, end)
        .filterBounds(aoi)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
        .filter(ee.Filter.eq("resolution_meters", 10))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(bands)
        .map(lambda x: x.convolve(ee.Kernel.square(1)))
        .map(compute_ratio)
    )


def sentinel_2_processing(rsd, aoi, start, end, bands) -> ee.ImageCollection:
    """Preprocess Sentinel-2 image."""

    def cloud_mask(img: ee.Image):
        """Apply cloud mask."""
        ...

    def compute_savi(img: ee.Image):
        """computes soil adjusted vegetation index."""
        ...

    def compute_tasseled_cap(img: ee.Image):
        """computes tasseled cap transformation."""
        ...

        # apply preprocessing pipeline

        return (
            rsd.filterDate(start, end)
            .filterBounds(aoi)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .select(bands)
            .map(cloud_mask)
            .map(compute_savi)
            .map(compute_tasseled_cap)
            .map(lambda x: x.normalizedDifference(["B8", "B4"]).rename("NDVI"))
        )


def alos_processing(rsd, aoi, start, end, bands) -> ee.ImageCollection:
    """Preprocess ALOS image."""
    compute_ratio = lambda x: x.select("HH").divide(x.select("HV"))
    return (
        rsd.filterDate(start, end)
        .filterBounds(aoi)
        .select(bands)
        .map(lambda x: x.convolve(ee.Kernel.square(1)))
        .map(compute_ratio)
    )


def dem_processing_pipeline(dataset: DEM) -> ee.Image:
    """Apply processing pipeline to DEM dataset."""
    # apply processing pipeline
    elevation = dataset.dataset.select(dataset.bands)
    slope = ee.Terrain.slope(elevation)
    dataset.dataset = dataset.dataset.addBands(slope)
    return dataset


def processing_pipeline(aoi, s1, s2, alos, dem) -> ee.Image:
    # s1 create a single year composite
    s1_composite = sentinel_1_processor(
        aoi=aoi,
        rsd=s1.dataset,
        start=s1.date[0][0],
        end=s1.date[0][1],
        bands=s1.bands,
    ).median()

    # s2 create a single year composite
    s2_2018 = sentinel_2_processing(
        aoi=aoi,
        rsd=s2.dataset,
        start=s2.date[0][0],
        end=s2.date[0][1],
        bands=s2.bands,
    )

    s2_2019 = sentinel_2_processing(
        aoi=aoi,
        rsd=s2.dataset,
        start=s2.date[1][0],
        end=s2.date[1][1],
        bands=s2.bands,
    )

    s2_2020 = sentinel_2_processing(
        aoi=aoi,
        rsd=s2.dataset,
        start=s2.date[2][0],
        end=s2.date[2][1],
        bands=s2.bands,
    )

    s2_composite = s2_2018.merge(s2_2019).merge(s2_2020).median()

    # create multi year alos composite
    # alos_composite = alos_processing(rsd=alos.dataset, aoi=aoi, start=alos.d).median()

    # create a dem composite
    dem_composite = dem_processing_pipeline()
    return ee.Image.cat(s1_composite, s2_composite, dem_composite)


# Remote Sensing Datasets and Processing ends here
#########################################################################################


# Data Engineering and Feature Extraction
@dataclass
class Features:
    dataset: str  # TODO make init var
    label_col: str

    def __post_init__(self):
        self.dataset = ee.FeatureCollection(self.dataset)


def remap_class_labels(features: Features) -> Features:
    """remaps class labels to integers does in place modification of the dataset"""
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


# TODO add a click command line interface to run the script
def main(args: list[str]) -> int:
    feature_id = args[0]
    aoi = args[1]

    features = Features(feature_id, "class_name")
    features = remap_class_labels(features)

    # set up the remote sensing datasets
    s1 = Sentinel1GRD()
    s2 = Sentinel2TOA()
    alos = ALOS()
    dem = DEM()

    # process the datasets
    processed = processing_pipeline(features, s1, s2, alos, dem)

    samples = extract(processed, features)

    ## export to asset

    # ee.Reset()

    # Model assess and save
    hyperparams = Hyperparameters(numberOfTrees=1000)
    model = SmileRandomForest(hyperparams)
    model.fit(
        samples, predictors=["VV", "VH", "B2", "B3", "B4", "B8", "elevation", "slope"]
    )
    metrics = model.assess(samples)
    # model.save("users/username/forest_model")

    # metrics.save()

    # monitor

    # classify the image
    stack = processing_pipeline(aoi, s1, s2, alos, dem)
    classified = model.predict(stack)

    # export to drive or cloud storage

    return 0
