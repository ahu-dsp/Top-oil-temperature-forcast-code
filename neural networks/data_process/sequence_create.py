
import numpy as np
import pandas as pd


def create_inout_train(sequence, history_size, predict_size):
    """
    制作供给模型使用的训练集序列和标签
    本函数只适合滑窗的滑动步长为1

    :param sequence:
    :param history_size:
    :param predict_size:
    :return:
    """
    series = pd.DataFrame(sequence)
    # print(series.head())
    series_copy = series.copy()
    for i in range(history_size + predict_size - 1):
        # 训练集输出时间步可以重叠
        series = pd.concat([series, series_copy.shift(-(i + 1))], axis=1)
    series.dropna(axis=0, inplace=True)
    # print(series.head())
    # print(series.shape)
    return series.values


def create_inout_test(sequence, history_size, predict_size):
    """
    制造测试集，测试集即预测集

    :param sequence:
    :param history_size:
    :param predict_size:
    :return:
    """
    # 时间窗步长+预测步长的总长度
    seq_label_len = history_size + predict_size
    # 实际样本的个数，保证测试集输出时间步不会重叠，能够依次相邻
    seq_num = (len(sequence) - history_size) // predict_size
    dataset = []
    for i in range(0, seq_num):
        a = sequence[i * predict_size:i * predict_size + seq_label_len, 0]
        dataset.append(a)

    return np.array(dataset)
