"""
helpers.py

This file contains helper functions for the cnwigaps package.
- stacking function
- monitor task function
"""

from typing import Any
import ee

from . import rsdp, model


def stack_images(aoi: ee.Geometry):
    s1 = rsdp.compute_s1_composite(aoi)
    s2 = rsdp.compute_s2_composite(aoi)
    also = rsdp.compute_alos_composite(aoi)
    elev = rsdp.compute_elevation(aoi)
    return ee.Image.cat(s1, s2, also, elev)


def monitor_task():
    pass


def compute_and_asses_model(
    features: ee.FeatureCollection, **kwargs
) -> tuple[Any, Any]:
    pass


def extract_features(
    image: ee.Image, features: ee.FeatureCollection
) -> ee.FeatureCollection:
    return image.sampleRegions(
        collection=features, scale=10, geometries=True, tileScale=16
    )
