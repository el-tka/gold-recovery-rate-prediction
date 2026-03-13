import pandas as pd


def split_features_target(data):

    target = data[
        ["rougher.output.recovery", "final.output.recovery"]
    ]

    features = data.drop(
        ["rougher.output.recovery", "final.output.recovery"],
        axis=1
    )

    return features, target


def align_features(train_features, test_features):
    """Приведение train/test к одинаковому набору признаков"""

    train_features, test_features = train_features.align(
        test_features,
        join="left",
        axis=1,
        fill_value=0
    )

    return train_features, test_features