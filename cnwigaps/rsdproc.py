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

    def merge(self, other: RSDProcessor) -> RSDProcessor:
        return self.__add__(other)

    def band_selector(self, bands: list[str]) -> RSDProcessor:
        self.rsd = self.rsd.select(bands)
        return self

    def add_boxcar(self, size: int) -> RSDProcessor:
        """creates a boxcar filter"""
        self.rsd = self.rsd.map(lambda image: image.convolve(ee.Kernel.square(size)))
        return self

    def compute_ratio(self, band1: str, band2: str) -> RSDProcessor:
        """computes the ratio of two bands"""
        band_name = f"{band1}_{band2}"
        self.rsd = self.rsd.map(
            lambda image: image.addBands(
                image.select(band1).divide(image.select(band2)).rename(band_name)
            )
        )
        return self

    def compute_ndvi(self, nir: str, red: str) -> RSDProcessor:
        """computes ndvi"""
        self.rsd = self.rsd.map(
            lambda image: image.addBands(
                image.normalizedDifference([nir, red]).rename("NDVI")
            )
        )
        return self

    def compute_savi(self, nir: str, red: str) -> RSDProcessor:
        self.rsd = self.rsd.map(
            lambda image: image.addBands(
                image.expression(
                    "((NIR - RED) / (NIR + RED + 0.5)) * (1.5)",
                    {"NIR": image.select(nir), "RED": image.select(red)},
                ).rename("SAVI")
            )
        )
        return self

    def compute_tasseled_cap(self, blue, green, red, nir, swir1, swir2) -> ee.Image:
        def _compute_tasseled_cap(image: ee.Image) -> ee.Image:
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

            input_img = image.select([blue, green, red, nir, swir1, swir2])
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

        self.rsd = self.rsd.map(_compute_tasseled_cap)
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

    def filter_iw_mode(self) -> S1Proc:
        self.rsd = self.rsd.filter(ee.Filter.eq("instrumentMode", "IW"))
        return self

    def filter_asc(self) -> S1Proc:
        self.rsd = self.rsd.filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
        return self

    def filter_vv_vh(self) -> S1Proc:
        self.rsd = self.rsd.filter(
            ee.Filter.listContains("transmitterReceiverPolarisation", "VV")
        )
        self.rsd = self.rsd.filter(
            ee.Filter.listContains("transmitterReceiverPolarisation", "VH")
        )
        return self

    def process(self) -> S1Proc:
        """
        Process the SAR data by applying filters and selecting specific bands.

        Returns:
            S1Proc: The processed SAR data.
        """
        (
            super()
            .process()
            .filter_iw_mode()
            .filter_asc()
            .filter_vv_vh()
            .band_selector(["VV", "VH"])
            .add_boxcar(1)
            .compute_ratio("VV", "VH")
        )
        return self


class S2Proc(RSDProcessor):
    BANDS = ["B2", "B3", "B4", "B5", "B6", "B8", "B8A", "B11", "B12"]

    def __init__(self, start_date, end_date, region) -> None:
        super().__init__("COPERNICUS/S2_HARMONIZED", start_date, end_date, region)

    def add_cloud_mask(self) -> S1Proc:
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

        self.rsd = self.rsd.map(mask_s2_clouds)
        return self

    def filter_clouds(self, value: int = 20) -> S2Proc:
        """Filters out cloudy images.

        Args:
            value (int, optional): The maximum cloud percentage. Defaults to 20.

        Returns:
            S2Proc: The processed Sentinel-2 data.
        """
        self.rsd = self.rsd.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", value))
        return self

    def process(self) -> S2Proc:
        (
            super()
            .process()
            .filter_clouds(value=20)
            .add_cloud_mask()
            .band_selector(self.BANDS)
            .compute_ndvi("B8", "B4")
            .compute_savi("B8", "B4")
            .compute_tasseled_cap("B2", "B3", "B4", "B8", "B11", "B12")
        )
        return self


class ALOSProc(RSDProcessor):
    def __init__(self, start_date, end_date, region) -> None:
        super().__init__(
            "JAXA/ALOS/PALSAR/YEARLY/SAR_EPOCH", start_date, end_date, region
        )

    def process(self) -> ALOSProc:
        (
            super()
            .process()
            .band_selector(["HH", "HV"])
            .add_boxcar(1)
            .compute_ratio("HH", "HV")
        )
        return self


class DEMProc:
    def process(self) -> ee.Image:
        img = ee.Image("NASA/NASADEM_HGT/001")
        return img.select("elevation").addBands(ee.Terrain.slope(img))


class ProcessingFactory:
    DATE_MAP = {
        "2018": ("2018-06-01", "2018-08-31"),
        "2019": ("2019-06-01", "2019-08-31"),
        "2020": ("2020-06-01", "2020-08-31"),
    }

    @staticmethod
    def _process_s1(region: ee.Geometry, date: tuple[str, str]) -> RSDProcessor:
        return S1Proc(start_date=date[0], end_date=date[1], region=region).process()

    @staticmethod
    def _process_s2(region: ee.Geometry, date: tuple[str, str]) -> RSDProcessor:
        return S2Proc(start_date=date[0], end_date=date[1], region=region).process()

    @staticmethod
    def _process_alos(region: ee.Geometry, date: tuple[str, str]) -> RSDProcessor:
        cur_year = date[0].split("-")[0]
        next_year = str(int(cur_year) + 1)

        return ALOSProc(
            start_date=cur_year, end_date=next_year, region=region
        ).process()

    def process_datasets(self, level: int, region: ee.Geometry) -> ee.Image | None:
        """processes the remote sensing datasets. returns the median of the processed images
        level 1: S1
        level 2: S2
        level 3: ALOS
        """
        if level == 1:
            return (
                self._process_s1(region, self.DATE_MAP["2018"])
                .merge(self._process_s1(region, self.DATE_MAP["2019"]))
                .merge(self._process_s1(region, self.DATE_MAP["2020"]))
                .rsd.median()
            )
        elif level == 2:
            return (
                self._process_s2(region, self.DATE_MAP["2018"])
                .merge(self._process_s2(region, self.DATE_MAP["2019"]))
                .merge(self._process_s2(region, self.DATE_MAP["2020"]))
                .rsd.median()
            )
        elif level == 3:
            return (
                self._process_alos(region, self.DATE_MAP["2018"])
                .merge(self._process_alos(region, self.DATE_MAP["2019"]))
                .merge(self._process_alos(region, self.DATE_MAP["2020"]))
                .rsd.median()
            )
        else:
            return None


def remote_sensing_dataset_processing(
    region: ee.Geometry | ee.FeatureCollection,
) -> ee.Image:
    """
    Process remote sensing datasets for a given region.

    Args:
        region: The region of interest.

    Returns:
        An ee.Image object containing the processed datasets.
    """
    factory = ProcessingFactory()
    s1 = factory.process_datasets(1, region)
    s2 = factory.process_datasets(2, region)
    alos = factory.process_datasets(3, region)
    dem = DEMProc().process()

    return ee.Image.cat(s1, s2, alos, dem)
