import os
import sys

import ee

from . import helpers


def main(args: list[str]) -> int:
    feature_id, aoi_id = args

    # step 1: load in features for modelling
    features = ee.FeatureCollection(feature_id)
    aoi = ee.FeatureCollection(aoi_id)
    # step 2: create features to extract from the image
    image = helpers.stack_images(aoi)
    extracted_features = helpers.extract_features(image, features)
    extracted_features_task = ee.batch.Export.table.toAsset(
        assetId=None,
        collection=extracted_features,
        description="",
    )
    extracted_features_task.start()
    # monitor task
    helpers.monitor_task(extracted_features_task)

    # step 3: model and assess
    model, assessment = helpers.compute_and_asses_model(extracted_features)
    model.save()
    assessment.prepare().save()

    # step 5: classify the image and export
    image = helpers.stack_images(aoi)
    classified_image = model.classify(image)

    return 0


if __name__ == "__main__":
    ee.Initialize()
    # sys.exit(main(sys.argv[1:]))
