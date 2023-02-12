
import numpy as np
from sklearn import metrics

"""回归类模型评价指标

:param y_true: array-like of shape (n_samples,)
            真实值
:param y_pred: array-like of shape (n_samples,)
            预测值
"""


def get_mae(y_true, y_pred):
    """平均绝对误差

    """
    mae = metrics.mean_absolute_error(y_true, y_pred)
    return mae


def get_mape(y_true, y_pred):
    """
    平均绝对百分比误差
    """
    mape = metrics.mean_absolute_percentage_error(y_true, y_pred)
    return mape


def get_mse(y_true, y_pred):
    """
    均方误差
    """
    mse = metrics.mean_squared_error(y_true, y_pred, squared=True)
    return mse


def get_rmse(y_true, y_pred):
    """
    均方根误差
    """
    rmse = metrics.mean_squared_error(y_true, y_pred, squared=False)
    return rmse


def get_r2(y_true, y_pred):
    """
    判定系数
    """
    r2 = metrics.r2_score(y_true, y_pred)
    return r2


def get_max_error(y_true, y_pred):
    """
    最大误差
    """
    max_error = metrics.max_error(y_true, y_pred)
    return max_error


def get_absolute_error(y_true, y_pred):
    """
    绝对误差
    """
    absolute_error = abs(y_true-y_pred)
    return absolute_error
