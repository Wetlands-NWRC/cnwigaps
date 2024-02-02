from __future__ import annotations
import ee


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
# Remote Sensing Datasets and Processing
##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##


# preprocessing helper
def preprocess_collection(
    collection: ee.ImageCollection, region: ee.Geometry, date: tuple[str, str]
) -> ee.ImageCollection:
    return collection.filterBounds(region).filterDate(date[0], date[1])


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##


# SAR datasets
## Helpers
def bxcar_filter(image: ee.Image) -> ee.Image:
    """applies 3x3 boxcar filter"""
    return image.convolve(ee.Kernel.square(1))


def compute_ratio(band1: str, band2: str) -> ee.Image:
    """computes the ratio of two bands"""
    band_name = f"{band1}_{band2}"
    return lambda image: image.addBands(
        image.select(band1).divide(image.select(band2)).rename(band_name)
    )


def s1_dataset(
    aoi, date: tuple[str, str], look_dir: str = "ASCENDING"
) -> ee.ImageCollection:
    dataset = (
        preprocess_collection("COPERNICUS/S1_GRD", aoi, date)
        .filter("instrumentMode", "IW")
        .filter("orbitProperties_pass", look_dir)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .select(["VV", "VH"])
        .map(bxcar_filter)
        .map(compute_ratio("VV", "VH"))
    )
    return dataset


def alos_dataset(aoi) -> ee.ImageCollection:
    years = "2018", "2021"
    dataset = (
        preprocess_collection("JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH", aoi, years)
        .select(["HH", "HV"])
        .map(bxcar_filter)
        .map(compute_ratio("HH", "HV"))
    )
    return dataset


def compute_alos_composite(aoi: ee.Geometry) -> ee.Image:
    return alos_dataset(aoi).median()


def compute_s1_composite(aoi: ee.Geometry) -> ee.Image:
    date_range = "2019-06-20", "2019-09-21"
    return s1_dataset(aoi, date_range).median()


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##


# Optical datasets
## Helpers
def mask_s2_clouds(image: ee.Image) -> ee.Image:
    """Masks clouds in a Sentinel-2 image using the QA band."""
    qa = image.select("QA60")

    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11

    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))

    return image.updateMask(mask)


def compute_ndvi(image: ee.Image) -> ee.Image:
    """computes ndvi"""
    return image.addBands(image.normalizedDifference(["B8", "B4"]).rename("NDVI"))


def compute_savi(image: ee.Image) -> ee.Image:
    """computes savi"""
    return image.addBands(
        image.expression(
            "((NIR - RED) / (NIR + RED + 0.5)) * (1.5)",
            {"NIR": image.select("B8"), "RED": image.select("B4")},
        ).rename("SAVI")
    )


def compute_tasseled_cap(image: ee.Image) -> ee.Image:
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

    input_img = image.select(["B2", "B3", "B4", "B8", "B11", "B12"])
    array_image = input_img.toArray()
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
    return image.addBands(components)


def s2_dataset(aoi: ee.Geometry, date: tuple[str, str]) -> ee.ImageCollection:
    dataset = (
        preprocess_collection("COPERNICUS/S2_HARMONIZED", aoi, date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .map(mask_s2_clouds)
        .select(["B2", "B3", "B4", "B5", "B6", "B8", "B8A", "B11", "B12"])
        .map(compute_ndvi)
        .map(compute_savi)
        .map(compute_tasseled_cap)
    )
    return dataset


def compute_s2_composite(aoi: ee.Geometry) -> ee.Image:
    s2_2018 = s2_dataset(aoi, ("2018-06-20", "2018-09-21"))
    s2_2019 = s2_dataset(aoi, ("2019-06-20", "2019-09-21"))
    s2_2020 = s2_dataset(aoi, ("2020-06-20", "2020-09-21"))

    s2 = s2_2018.merge(s2_2019).merge(s2_2020)

    return s2.median()


##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##


# Elevation
def compute_elevation() -> ee.Image:
    dem = ee.Image("NASA/NASADEM_HGT/001").select("elevation")
    slope = ee.Terrain.slope(dem)
    return dem.addBands(slope)
