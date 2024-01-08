import ee


class ImageProc:
    pass


class S2Processor:
    def __init__(self, start_date, end_date, region)
        self.start_date = start_date
        self.end_date = end_date
        self.region = region

    """
    s2 processing

    processing steps
    filter by region
    filter by date range
    apply a cloud mask
    select only Bands 1 - 12
    add ndvi
    add savi
    add tasseled cap
    reduce 

    S1 processing
    filter by region
    filter by date
    filter by acs
    filter by polerization
    filter by IW mode
    select bands
    add 7 x 7 refined lee filter
    add ratio
    reduce

    Alos Processing
    - filter by date
    - filter by region
    - add 7 x 7 lee
    - add ratio
    - select HH and HV channels
    """

class RSDProcessor:
    """ Remote Sensing Dataset Processor"""
    def __init__(self, col_id, start_date, end_date, region):
        self.rsd = ee.ImageCollection(col_id)
        self.start_date = start_date
        self.end_date = end_date
        self.region = region

    def process(self) -> ImgColProc:
        """does bare bones processing (date and region)"""
        self.rsd = self.rsd.filterBounds(self.region).filterDate(self.start_date, self.end_date)
        return self
"""
S1 processing
    filter by region
    filter by date
    filter by acs
    filter by polerization
    filter by IW mode
    select bands
    add 7 x 7 refined lee filter
    add ratio
    reduce
"""
class S1Proc(RSDProcessor):
    def __init__(self, start_date, end_date, region)
        super.__init__("", start_date, end_date, region)
    
    @staticmethod
    def ratio(image: ee.Image):
        calc = None
        return image.addBands(calc).rename('VV_VH')

    def process(self):
        self.rsd = super().process().rsd.filter().filter().filter().filter().select("V.*").map(ratio)
        :w

