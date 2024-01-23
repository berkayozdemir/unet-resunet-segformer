import datasets
from datasets import Sequence, Value
from torchmetrics.classification import Dice
import numpy as np


def dice_coefficient(y_true, y_pred,img_size):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))/img_size

class DiceCoef(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description="dice_coef",citation="dice_coef",
            features=datasets.Features({
                'predictions': Sequence(feature=Sequence(feature=Value(dtype='uint16', id=None), length=-1, id=None), length=-1, id=None),
                'references':Sequence(feature=Sequence(feature=Value(dtype='uint16', id=None), length=-1, id=None), length=-1, id=None),
            }),


        )

    def _compute(self, predictions, references, img_size):
        return {"accuracy": dice_coefficient(predictions, references, img_size)}