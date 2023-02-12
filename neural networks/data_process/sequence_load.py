
import pandas as pd


def load_data(file_name):
    """
    加载数据集

    :param file_name:
    :return:
    """
    dataframe = pd.read_csv('dataset/GuQuan_data/' + file_name)  # 用绝对路径，相对路径报错
    # dataframe = pd.read_csv('dataset/CSDN_data/' + file_name)
    print(dataframe.head())
    # data_value = dataframe['value'].values[8444:14823]  # 获取温度数据
    data_value = dataframe['value'].values
    data_value = data_value.astype('float32')  # float64-->float32
    print(data_value[:5])

    # dataset_index = dataframe['time'].values.tolist()  # 获取时间索引
    # 填充缺失值
    # dataframe.fillna(dataframe.mean(), inplace=True)
    return data_value