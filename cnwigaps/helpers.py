"""
helpers.py

This file contains helper functions for the cnwigaps package.
- stacking function
- monitor task function
"""

import ee

from . import rsdp


def stack_images(aoi: ee.Geometry):
    s1 = rsdp.compute_s1_composite(aoi)
    s2 = rsdp.compute_s2_composite(aoi)
    also = rsdp.compute_alos_composite(aoi)
    elev = rsdp.compute_elevation(aoi)
    return ee.Image.cat(s1, s2, also, elev)


def monitor_task():
    pass
