from __future__ import annotations
import ee


class RSDProcessor:
    """base class for remote sensing dataset processors"""

    def __init__(self, dataset, start_date, end_date, region) -> None:
        self.rsd = ee.ImageCollection(dataset)
        self.start_date = start_date
        self.end_date = end_date
        self.region = region

    def __add__(self, other) -> RSDProcessor:
        # TODO add type checking
        self.rsd = self.rsd.merge(other.rsd)
        return self

    def process(self) -> RSDProcessor:
        """does bare bones processing (date and region)"""
        self.rsd = self.rsd.filterBounds(self.region).filterDate(
            self.start_date, self.end_date
        )
        return self


class S1Proc(RSDProcessor):
    def __init__(self, start_date, end_date, region) -> None:
        super().__init__("COPERNICUS/S1_GRD", start_date, end_date, region)

    @staticmethod
    def ratio(image: ee.Image):
        return image.addBands(
            image.select("VV").divide(image.select("VH")).rename("VV_VH")
        )

    @staticmethod
    def boxcar(image: ee.Image):
        return image.convolve(ee.Kernel.square(1, "pixels"))

    def process(self) -> S1Proc:
        """
        Process the SAR data by applying filters and selecting specific bands.

        Returns:
            S1Proc: The processed SAR data.
        """
        self.rsd = (
            super()
            .process()
            .rsd.filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select("V.*")
            .map(self.boxcar)
            .map(self.ratio)
        )
        return self


class S2Proc(RSDProcessor):
    BANDS = ["B2", "B3", "B4", "B5", "B6", "B8", "B8A", "B11", "B12"]

    def __init__(self, start_date, end_date, region) -> None:
        super().__init__("COPERNICUS/S2_HARMONIZED", start_date, end_date, region)

    @staticmethod
    def mask_s2_clouds(image: ee.Image) -> ee.Image:
        """Masks clouds in a Sentinel-2 image using the QA band.

        Args:
            image (ee.Image): A Sentinel-2 image.

        Returns:
            ee.Image: A cloud-masked Sentinel-2 image.
        """
        qa = image.select("QA60")

        # Bits 10 and 11 are clouds and cirrus, respectively.
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11

        # Both flags should be set to zero, indicating clear conditions.
        mask = (
            qa.bitwiseAnd(cloud_bit_mask)
            .eq(0)
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        )

        return image.updateMask(mask)

    @staticmethod
    def ndvi(image: ee.Image) -> ee.Image:
        return image.addBands(image.normalizedDifference(["B8", "B4"]).rename("NDVI"))

    @staticmethod
    def savi(image: ee.Image) -> ee.Image:
        return image.addBands(
            image.expression(
                "((NIR - RED) / (NIR + RED + 0.5)) * (1.5)",
                {"NIR": image.select("B8"), "RED": image.select("B4")},
            ).rename("SAVI")
        )

    @staticmethod
    def tasseled_cap(image: ee.Image) -> ee.Image:
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

    def process(self) -> S2Proc:
        self.rsd = (
            super()
            .process()
            .rsd.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
            .map(self.mask_s2_clouds)
            .select(self.BANDS)
            .map(self.ndvi)
            .map(self.savi)
            .map(self.tasseled_cap)
        )
        return self


class ALOSProc(RSDProcessor):
    def __init__(self, start_date, end_date, region) -> None:
        super().__init__(
            "JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH", start_date, end_date, region
        )

    @staticmethod
    def ratio(image: ee.Image):
        return image.addBands(
            image.select("HH").divide(image.select("HV")).rename("HH_HV")
        )

    @staticmethod
    def boxcar(image: ee.Image):
        return image.convolve(ee.Kernel.square(1, "pixels"))

    def process(self) -> ALOSProc:
        self.rsd = super().process().rsd.select("H.*").map(self.boxcar).map(self.ratio)
        return self


class DEMProc(RSDProcessor):
    pass


def process_rsd(rsd_type: int, start_date: str, end_date: str, region: ee.Geometry):
    """processes a remote sensing dataset

    RSD_MAP = {
        0: S1Proc,
        1: S2Proc,
        2: ALOSProc,
        3: DEMProc
    }

    Args:
        rsd_type (int): the type of remote sensing dataset to process
        start_date (str): the start date of the dataset
        end_date (str): the end date of the dataset
        region (ee.Geometry): the region to process the dataset over

    Returns:
        ee.ImageCollection: the processed remote sensing dataset
    """

    MAP = {0: S1Proc, 1: S2Proc, 2: ALOSProc, 3: DEMProc}
    processor = MAP.get(rsd_type, None)
    if processor is None:
        raise ValueError("Invalid RSD type")
    if rsd_type == 3:
        return processor().process().rsd
    else:
        return processor(start_date, end_date, region).process().rsd


def batch_process_rsd(
    rsd_type, years: list[int], start_mm_dd: str, end_mm_dd: str, region
) -> RSDProcessor:
    """
    Batch process remote sensing data.

    Args:
        rsd_type (int): The type of remote sensing data to process.
        years (list[int]): List of years to process.
        start_mm_dd (str): Start date in MM-DD format.
        end_mm_dd (str): End date in MM-DD format.
        region: The region to process the data for.

    Returns:
        RSDProcessor: The processed data.

    Raises:
        ValueError: If an invalid RSD type is provided.
    """

    MAP = {0: S1Proc, 1: S2Proc, 2: ALOSProc, 3: DEMProc}
    processor = MAP.get(rsd_type, None)
    if processor is None:
        raise ValueError("Invalid RSD type")

    processed = [
        processor(f"{year}-{start_mm_dd}", f"{year}-{end_mm_dd}", region).process()
        for year in years
    ]

    # combine
    combined = processed[0]
    for ds in processed[1:]:
        combined += ds

    return combined
