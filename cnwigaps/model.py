from __future__ import annotations

import ee


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
