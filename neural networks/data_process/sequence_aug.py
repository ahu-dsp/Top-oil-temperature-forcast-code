


from sklearn.preprocessing import MinMaxScaler, StandardScaler


def normalize(train_dataset, method="-1-1"):
    """
    此函数是对训练集归一化，测试集归一化应当使用训练集的标准，从而保证两类结果在同一“标准”下！

    :param train_dataset: 一维的时间序列
    :param method: zscore or -1-1 or 0-1 or none, 分别对应：标准化、归一化（-1-1）、归一化（0-1）、不归化
    :return: normalized time series
    """
    assert method in ["zscore", "-1-1", "0-1", None]

    if method == "zscore":
        scaler = StandardScaler()
        data_normalized = scaler.fit_transform(train_dataset)
        return data_normalized, scaler

    elif method == "-1-1":
        scaler = MinMaxScaler(feature_range=(-1, 1))
        data_normalized = scaler.fit_transform(train_dataset)
        return data_normalized, scaler

    elif method == "0-1":
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_normalized = scaler.fit_transform(train_dataset)
        return data_normalized, scaler

    elif method is None:
        print("未归一化！无 scaler!")
        return train_dataset
