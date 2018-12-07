from lasagne.objectives import categorical_crossentropy
import numpy as np

def norm_cross_entropy(y_pred, y_true):
    return categorical_crossentropy(y_pred, y_true) / np.log(2)

def cross_entropy(y_pred, y_true):
    return categorical_crossentropy(y_pred, y_true)