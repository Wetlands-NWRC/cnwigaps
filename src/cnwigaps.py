from __future__ import annotations
import sys

from typing import Any
from dataclasses import dataclass, InitVar, field
from pprint import pprint

import click
import ee


@dataclass
class RemoteSensingDataset:
    dataset_id: str
    bands: list[str]
    date_range: tuple[str]
    aoi: ee.FeatureCollection | ee.Geometry


class RemoteSensingDatasetProcessor:
    def __init__(self, dataset=None) -> None:
        self.dataset = dataset

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, arg):
        if arg is None:
            self._dataset = arg
        elif isinstance(arg, ee.ImageCollection):
            self._dataset = arg
        else:
            self._dataset = ee.ImageCollection(arg)

    def filter_bounds(self, aoi) -> RemoteSensingDatasetProcessor:
        self._dataset = self._dataset.filterBounds(aoi)
        return self

    def filter_date(self, dates: tuple[str]) -> RemoteSensingDatasetProcessor:
        self._dataset = self._dataset.filterDate(*dates)
        return self

    def select(self, var_args: Any) -> RemoteSensingDatasetProcessor:
        self._dataset = self._dataset.select(var_args)
        return self

    def filter_dv(self, look_direction: str = None) -> RemoteSensingDatasetProcessor:
        look_direction = look_direction or "DESCENDING"
        self._dataset = (
            self._dataset.filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.eq("orbitProperties_pass", look_direction))
            .filter(ee.Filter.eq("resolution_meters", 10))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        )
        return self

    def s2_cloud_mask(self):
        def cloud_mask(img):
            """Apply cloud mask."""
            qa = img.select("QA60")

            # Bits 10 and 11 are clouds and cirrus, respectively.
            cloud_bit_mask = 1 << 10
            cirrus_bit_mask = 1 << 11

            # Both flags should be set to zero, indicating clear conditions.
            mask = (
                qa.bitwiseAnd(cloud_bit_mask)
                .eq(0)
                .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
            )

            return img.updateMask(mask)

        self._dataset = self._dataset.map(cloud_mask)

    def add_boxcar(self, radius: int = 1) -> RemoteSensingDatasetProcessor:
        self._dataset = self._dataset.map(
            lambda x: x.convolve(ee.Kernel.square(radius)).set("boxcar", "True")
        )
        return self

    def compute_ratio(self, b1, b2) -> RemoteSensingDatasetProcessor:
        self._dataset = self._dataset.map(
            lambda x: x.addBands(x.select(b1).divide(x.select(b2)).rename(f"{b1}_{b2}"))
        )
        return self

    def compute_savi(self):
        """computes soil adjusted vegetation index."""
        self._dataset = self._dataset.map(
            lambda x: x.addBands(
                x.expression(
                    "1 + L * (NIR - RED) / (NIR + RED + L)",
                    {
                        "NIR": x.select("B8"),
                        "RED": x.select("B4"),
                        "L": 0.5,
                    },
                ).rename("SAVI")
            )
        )
        return self

    def compute_ndvi(self):
        self._dataset = self._dataset.map(
            lambda x: x.addBands(x.normalizedDifference(["B8", "B4"]))
        )
        return self

    def compute_tasseled_cap(self):
        def compute_tasseled_cap(img: ee.Image):
            """computes tasseled cap transformation."""
            coefficients = ee.Array(
                [
                    [0.3029, 0.2786, 0.4733, 0.5599, 0.508, 0.1872],
                    [-0.2941, -0.243, -0.5424, 0.7276, 0.0713, -0.1608],
                    [0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559],
                    [-0.8239, 0.0849, 0.4396, -0.058, 0.2013, -0.2773],
                    [-0.3294, 0.0557, 0.1056, 0.1855, -0.4349, 0.8085],
                    [0.1079, -0.9023, 0.4119, 0.0575, -0.0259, 0.0252],
                ]
            )

            image_inpt = img.select(["B2", "B3", "B4", "B8", "B11", "B12"])
            array_image = image_inpt.toArray()
            array_image_2d = array_image.toArray(1)

            components = (
                ee.Image(coefficients)
                .matrixMultiply(array_image_2d)
                .arrayProject([0])
                .arrayFlatten(
                    [["brightness", "greenness", "wetness", "fourth", "fifth", "sixth"]]
                )
            )
            components = components.select(["brightness", "greenness", "wetness"])
            return img.addBands(components)

        self._dataset = self._dataset.map(compute_tasseled_cap)
        return self

    def add_slope(self):
        self._dataset = self._dataset.map(
            lambda x: x.addBands(ee.Terrain.slope(x.select("elevation")))
        )
        return self

    def build(self):
        return self._dataset


# class Sentinel2Processing:
#     def __init__(self):
#         self.processor = RemoteSensingDatasetProcessor()

#     def build_product(self, rsd: RemoteSensingDataset) -> :
#         """exicutes the processing chain"""
#         self.processor.dataset = rsd.dataset_id
#         return (
#             self.processor.
#         )

#     def build_multi_year_product(self, rsds: list[RemoteSensingDataset]):
#         pass


class Sentinel1Processing:
    def __init__(self) -> None:
        self.processor = RemoteSensingDatasetProcessor()

    def build_products():
        pass


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
    bands: list = field(
        default_factory=lambda: [
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7" "B8",
            "B8A",
            "B11",
            "B12",
        ]
    )
    date: list = field(
        default_factory=lambda: [
            ("2019-06-20", "2019-09-21"),
            ("2019-06-20", "2019-09-21"),
            ("2019-06-20", "2019-09-21"),
        ]
    )

    def __post_init__(self):
        self.dataset = ee.ImageCollection(self.dataset_id)


def sentinel_2_processing(rsd, aoi, start, end, bands) -> ee.ImageCollection:
    """Preprocess Sentinel-2 image."""

    def cloud_mask(img: ee.Image):
        """Apply cloud mask."""
        qa = img.select("QA60")

        # Bits 10 and 11 are clouds and cirrus, respectively.
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11

        # Both flags should be set to zero, indicating clear conditions.
        mask = (
            qa.bitwiseAnd(cloud_bit_mask)
            .eq(0)
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        )

        return img.updateMask(mask)

    def compute_savi(img: ee.Image):
        """computes soil adjusted vegetation index."""
        return img.addBands(
            img.expression(
                "1 + L * (NIR - RED) / (NIR + RED + L)",
                {
                    "NIR": img.select("B8"),
                    "RED": img.select("B4"),
                    "L": 0.5,
                },
            ).rename("SAVI")
        )

    def compute_tasseled_cap(img: ee.Image):
        """computes tasseled cap transformation."""
        coefficients = ee.Array(
            [
                [0.3029, 0.2786, 0.4733, 0.5599, 0.508, 0.1872],
                [-0.2941, -0.243, -0.5424, 0.7276, 0.0713, -0.1608],
                [0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559],
                [-0.8239, 0.0849, 0.4396, -0.058, 0.2013, -0.2773],
                [-0.3294, 0.0557, 0.1056, 0.1855, -0.4349, 0.8085],
                [0.1079, -0.9023, 0.4119, 0.0575, -0.0259, 0.0252],
            ]
        )

        image_inpt = img.select(["B2", "B3", "B4", "B8", "B11", "B12"])
        array_image = image_inpt.toArray()
        array_image_2d = array_image.toArray(1)

        components = (
            ee.Image(coefficients)
            .matrixMultiply(array_image_2d)
            .arrayProject([0])
            .arrayFlatten(
                [["brightness", "greenness", "wetness", "fourth", "fifth", "sixth"]]
            )
        )
        components = components.select(["brightness", "greenness", "wetness"])
        return img.addBands(components)

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


@dataclass
class ALOS:
    dataset_id: str = field(default="JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH")
    bands: list = field(default_factory=lambda: ["HH", "HV"])
    date: list = field(default_factory=lambda: [("2018", "2021")])

    def __post_init__(self):
        self.dataset = ee.ImageCollection(self.dataset_id)


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


@dataclass
class DEM:
    dataset_id: str = field(default="USGS/SRTMGL1_003")
    bands: list = field(default_factory=lambda: ["elevation"])

    def __post_init__(self):
        self.dataset = ee.Image(self.dataset_id)


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
    alos_composite = alos_processing(
        rsd=alos.dataset, aoi=aoi, start=alos.date[0][0], end=alos.date[0][1]
    )

    # create a dem composite
    dem_composite = dem_processing_pipeline()
    return ee.Image.cat(s1_composite, s2_composite, alos_composite, dem_composite)


# Remote Sensing Datasets and Processing ends here
#########################################################################################


# Data Engineering and Feature Extraction
@dataclass
class Features:
    dataset: InitVar[str]  # TODO make init var
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

    features = ee.FeatureCollection(feature_id)

    s1_dataset = RemoteSensingDataset(
        dataset_id="COPERNICUS/S1_GRD",
        bands=["VV", "VH"],
        date_range=("2019-06-20", "2019-09-21"),
        aoi=features,
    )

    procssor = RemoteSensingDatasetProcessor()

    # S1 processing pipeline
    # s1 = (
    #     RemoteSensingDatasetProcessor(dataset=s1_dataset.dataset_id)
    #     .filter_bounds(s1_dataset.aoi)
    #     .filter_date(s1_dataset.date_range)
    #     .filter_dv()
    #     .select("V.*")
    #     .add_boxcar()
    #     .compute_ratio("VV", "VH")
    #     .build()
    # )

    pprint(s1.first().getInfo())

    return 0


if __name__ == "__main__":
    ee.Initialize()
    sys.exit(main(sys.argv[1:]))
