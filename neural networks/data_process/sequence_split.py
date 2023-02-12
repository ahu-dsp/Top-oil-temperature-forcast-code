


def train_test_split(my_data, val=False):
    """
    划分训练集、验证集、测试集

    :param val: 是否划分验证集，默认值 False，不划分
    :param my_data:
    :return:
    """
    if not val:
        # train_num = int(len(my_data) * 0.89)
        # 训练集
        # train_num = 5723
        test_num = 24
        train_data = my_data[0:-test_num]
        # 测试集
        test_data = my_data[-test_num:]

        return train_data, test_data
    else:
        train_num = int(len(my_data) * 0.8)
        test_num = int(len(my_data) * 0.1)
        # 训练集
        train_data = my_data[0:train_num]
        # 测试集
        test_data = my_data[-test_num:]
        # 验证集
        val_data = my_data[train_num:-test_num]

        return train_data, val_data, test_data
