
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models import BPNN
from data_process import sequence_create, sequence_split, sequence_load, sequence_aug
from metrics.regression_metrics import get_mae, get_mse, get_mape, get_rmse, get_r2
"""
方法：BPNN
策略：单模型多输出
"""

if __name__ == '__main__':
    # 设置随机种子
    np.random.seed(0)

    # == == == == == == == == == 制造模型需要的数据 == == == == == == == == ==

    # 加载数据
    data_origin = sequence_load.load_data(' ')
    train_data, test_data = sequence_split.train_test_split(data_origin)
    # print(train_data.shape)

    # 归一化
    train_data_normalized, scaler = sequence_aug.normalize(train_data.reshape(-1, 1), method='-1-1')
    test_data_normalized = scaler.transform(test_data.reshape(-1, 1))
    # print(train_data_normalized[:5])

    # 制造训练集序列及对应标签
    """
    【注】

    1. 输入时间窗步长一般大于预测步长
    2. 若利用训练集最后的时间窗预测测试集初始部分，则必须满足 输入步长>预测步长，以防止训练集渗入测试集作弊
    3. 故本代码训练集最后一个时间窗步长数据不参与训练 
    """
    history_size = 24 # 输入时间窗步长
    predict_size = 24 # 预测步长

    # 训练序列最后一个时间窗步长数据不参与训练
    train_real_normalized = train_data_normalized[:-history_size]

    # 测试序列前面增加训练序列的最后一个时间窗步长数据
    test_real_normalized = np.concatenate(
        (train_data_normalized[-history_size:], test_data_normalized))

    train_inout_seq = sequence_create.create_inout_train(train_real_normalized, history_size, predict_size)
    # train_inout_seq = shuffle(train_inout_seq, random_state=1)  # 每次运行 shuffle 函数得到打乱后的结果都是相同的
    train_x, train_y = train_inout_seq[:, :-predict_size], train_inout_seq[:, -predict_size:]
    test_inout_seq = sequence_create.create_inout_test(test_real_normalized, history_size, predict_size)
    test_x, test_y = test_inout_seq[:, :-predict_size], test_inout_seq[:, -predict_size:]

    # 样本数量
    train_data_num = len(train_x)
    test_data_num = len(test_x)  # 数量应该等于len(train_data_normalized)

    # == == == == == == == == == 搭建模型 == == == == == == == == ==

    # 实例化网络模型
    bp_mlp = BPNN.bp_model()

    # == == == == == == == == == 模型训练 == == == == == == == == ==
    bp_fit = bp_mlp.fit(train_x, train_y)

    # == == == == == == == == == 模型预测 == == == == == == == == ==
    # 训练集
    train_predict = bp_fit.predict(train_x)
    # print(train_predict)
    train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))  # 逆归一化
    # 测试集
    test_predict = bp_fit.predict(test_x)
    test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))  # 逆归一化

    # == == == == == == == == == == 数据索引 == == == == == == == == == ==

    data_number_index = np.arange(0, len(data_origin), 1)  # 数字索引，非时间索引
    test_number_index = data_number_index[len(data_origin) - test_data_num*predict_size:]
    output = pd.DataFrame(test_predict)
    output.to_csv(' ')

    # == == == == == == == == ==  预测误差评价指标 == == == == == == == == ==

    # 平均绝对误差(MAE)
    print('---------评估指标--------')
    mae = get_mae(data_origin[test_number_index], test_predict)
    print('MAE:{}'.format(mae))
    # 平均绝对百分比误差(MAPE)
    mape = get_mape(data_origin[test_number_index], test_predict) * 100
    print('MAPE:{}%'.format(mape))
    # 均方误差(MSE)
    mse = get_mse(data_origin[test_number_index], test_predict)
    print('MSE:{}'.format(mse))
    # 均方根误差(RMSE)
    rmse = get_rmse(data_origin[test_number_index], test_predict)
    print('RMSE:{}'.format(rmse))
    # 判别系数R^2
    r2 = get_r2(data_origin[test_number_index], test_predict)
    print('R^2:{}'.format(r2))

    # == == == == == == == == == == 绘图 == == == == == == == == == ==

    plt.plot(test_number_index, data_origin[test_number_index], 'b', label='real')
    plt.plot(test_number_index, test_predict, 'y-', label='prediction')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
