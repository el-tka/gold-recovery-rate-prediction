import numpy as np


def smape(target, prediction):

    numerator = np.abs(target - prediction)
    denominator = (np.abs(target) + np.abs(prediction)) / 2

    return np.mean(numerator / denominator) * 100


def final_smape(rougher_true, rougher_pred,
                final_true, final_pred):

    rougher_smape = smape(rougher_true, rougher_pred)
    final_smape = smape(final_true, final_pred)

    return 0.25 * rougher_smape + 0.75 * final_smape