import ee


class AssessmentTable:
    def __init__(
        self,
        predictions: ee.FeatureCollection,
        actual: str,
        predicted: str = None,
        class_order: list[str] | ee.List = None,
    ) -> None:
        self.actual = actual
        self.predicted = predicted or "classification"
        self.class_order = class_order
        self.matrix = predictions

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, predictions: ee.FeatureCollection) -> None:
        self._matrix = predictions.errorMatrix(
            self.actual, self.predicted, self.class_order
        )

    @property
    def producers(self) -> ee.List:
        return self.matrix.producersAccuracy().toList().flatten()

    @property
    def consumers(self) -> ee.List:
        return self.matrix.consumersAccuracy().toList().flatten()

    @property
    def overall(self) -> ee.Number:
        return self.matrix.accuracy()

    @property
    def kappa(self) -> ee.Number:
        return self.matrix.kappa()

    def create_metrics(self) -> ee.FeatureCollection:
        return ee.FeatureCollection(
            [
                ee.Feature(None, {"matrix": self.matrix.array()}),
                ee.Feature(None, {"overall": self.overall}),
                ee.Feature(None, {"producers": self.producers}),
                ee.Feature(None, {"consumers": self.consumers}),
                ee.Feature(None, {"order": self.class_order}),
            ]
        )
