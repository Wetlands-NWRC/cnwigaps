from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import ee


@dataclass
class Assessment:
    overall: ee.Number
    kappa: ee.Number
    producers: ee.List
    consumers: ee.List
    order: ee.List
    confusion_matrix: Any

    def prep_for_export(self) -> ee.FeatureCollection:
        return ee.FeatureCollection(
            [
                ee.Feature(None, {"matrix": self.matrix.array()}),
                ee.Feature(None, {"overall": self.overall}),
                ee.Feature(None, {"producers": self.producers}),
                ee.Feature(None, {"consumers": self.consumers}),
                ee.Feature(None, {"order": self.class_order}),
            ]
        )


class SmileRandomForest:
    @classmethod
    def from_asset(cls, asset_id: str) -> SmileRandomForest:
        cls.model = ee.Classifier.load(asset_id)
        return cls

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.model = None

    def train(self, features, labels, predictors) -> SmileRandomForest:
        self.model = ee.SmileRandomForest(**self.kwargs).train(
            features, labels, predictors
        )
        return self

    def predict(self, x: ee.Image | ee.FeatureCollection):
        return x.classify(self.model)

    def assess(self, dataset, class_order: list[int] = None):
        if self.model is None:
            raise ValueError("Model not trained")

        predictions = self.predict(dataset)

        if isinstance(dataset, ee.FeatureCollection):
            confusion_matrix = predictions.errorMatrix(
                "class", "classification", class_order
            )

        overall_accuracy = confusion_matrix.accuracy()
        kappa = confusion_matrix.kappa()
        producers = confusion_matrix.producersAccuracy()
        consumers = confusion_matrix.consumersAccuracy()

        return Assessment(
            overall_accuracy, kappa, producers, consumers, class_order, confusion_matrix
        )
