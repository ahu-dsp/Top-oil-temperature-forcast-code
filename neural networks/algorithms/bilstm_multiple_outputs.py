

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from data_process import sequence_create, sequence_split, sequence_load, sequence_aug, SequenceDatasets
from models.BiLSTM import BiLstmNet
from metrics.regression_metrics import get_mae, get_mse, get_mape, get_rmse, get_r2
from utils.set_seed import set_seed

def model_train(lstm_linear, train_loader, loss_fn, optimizer, epoch=10):
    """

    :param lstm_linear: 实例化模型
    :param train_loader: 属于Dataloader类的训练集
    :param loss_fn: 损失函数，默认为均方损失
    :param optimizer: 优化器
    :param epoch: 训练轮数，默认为10
    :return:
    """
    # 设置训练网络的一些参数
    # 记录训练的次数    （注：按每一轮的训练次数）
    total_train_step = 0
    # 记录测试的次数   （注：按轮数）
    total_test_step = 0

    for i in range(epoch):
        print("-------第 {} 轮训练开始-------".format(i + 1))
        train_loss = []
        # train_predict = []

        # 训练步骤开始
        lstm_linear.train()
        for inout in train_loader:
            seq, label = inout
            seq = seq.to(device)
            label = label.to(device)

            outputs = lstm_linear(seq)
            loss = loss_fn(outputs, label)
            train_loss.append(loss.item())
            # train_predict.append(outputs.detach().numpy())

            # 优化器优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1
            if total_train_step % 10 == 0:
                print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
                pass

            pass


def model_predict_MIMO(lstm_linear, data_loader):
    """
    模型预测——多步预测（单模型多输出法）

    :param lstm_linear: 模型实例化对象
    :param data_loader: 滑窗划分好的训练数据集
    :return: 多输入多输出法多步预测结果（虽然只有一个输入，原始论文是这样表述的）
    """
    lstm_linear.eval()  # 不启用 BatchNormalization、Dropout等
    test_num = len(data_loader)
    predict = []
    with torch.no_grad():
        for data in data_loader:
            seq, label = data
            seq = seq.to(device)
            label = label.to(device).unsqueeze(1)
            outputs = lstm_linear(seq)
            predict.append(outputs.data.cpu().numpy().tolist())
    out = np.array(sum(predict, []))  # 列表扁平化处理：将多个列表合并成一个列表
    return out


if __name__ == '__main__':
    # 定义训练的设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 设置随机种子
    set_seed(1)

    # == == == == == == == == == 制造模型需要的数据 == == == == == == == == ==

    # 加载数据
    data_origin = sequence_load.load_data(' ')
    train_data, test_data = sequence_split.train_test_split(data_origin)
    # print(train_data.shape)

    # 归一化
    train_data_normalized, scaler = sequence_aug.normalize(train_data.reshape(-1, 1), method='-1-1')
    test_data_normalized = scaler.transform(test_data.reshape(-1, 1))
    # print(train_data_normalized[:5])

    # 制造序列及对应标签
    """
    【注】

    1. 输入时间窗步长一般大于预测步长
    2. 若利用训练集最后的时间窗预测测试集初始部分，则必须满足 输入步长>预测步长，以防止训练集渗入测试集作弊
    3. 故本代码训练集最后一个时间窗步长数据不参与训练 
    """
    history_size =12 # 序列长度
    predict_size = 6  # 预测步长

    # 训练序列最后一个时间窗步长数据不参与训练
    train_real_normalized = train_data_normalized[:-history_size]
    # 测试序列前面增加训练序列的最后一个时间窗步长数据
    test_real_normalized = np.concatenate(
        (train_data_normalized[-history_size:], test_data_normalized))

    train_inout_seq = sequence_create.create_inout_train(train_real_normalized, history_size, predict_size)
    train_x, train_y = train_inout_seq[:, :-predict_size], train_inout_seq[:, -predict_size:]
    test_inout_seq = sequence_create.create_inout_test(test_real_normalized, history_size, predict_size)
    test_x, test_y = test_inout_seq[:, :-predict_size], test_inout_seq[:, -predict_size:]

    # 数据集制造最后一步
    train_process_data = SequenceDatasets.MyData(train_x, train_y)
    test_process_data = SequenceDatasets.MyData(test_x, test_y)
    # 长度
    train_data_num = len(train_process_data)
    test_data_num = len(test_process_data)

    """
    【注】

    1. 经实验，发现batch_size设置越小，训练时损失loss震荡越大
    2. 
    """
    batch_size = 15
    train_loader = DataLoader(train_process_data, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_process_data, batch_size=1, shuffle=False, drop_last=False)

    # == == == == == == == == == 搭建模型 == == == == == == == == ==

    # 实例化网络模型
    lstm_linear_model = BiLstmNet(input_size=1,
                                   hidden_size=100,
                                   num_layers=2,
                                   output_size=predict_size,
                                   history_size=history_size,
                                   dropout=0
                                   )
    print('网络骨架：{}'.format(lstm_linear_model))
    lstm_linear_model = lstm_linear_model.to(device)

    # 损失函数
    loss_fn = nn.MSELoss()
    loss_fn = loss_fn.to(device)
    # 优化器
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(lstm_linear_model.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)   # 学习率固定步长衰减
    # 设置训练轮数
    epoch =80

    # == == == == == == == == == 模型训练 == == == == == == == == ==

    model_train(lstm_linear_model,
                train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epoch=epoch
                )

    # == == == == == == == == == 模型预测 == == == == == == == == ==

    # 测试集
    test_predict = model_predict_MIMO(lstm_linear_model, test_loader)
    # print(train_predict)
    test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))  # 逆归一化

    # == == == == == == == == == == 数据索引 == == == == == == == == == ==

    data_number_index = np.arange(0, len(data_origin), 1)  # 数字索引，非时间索引
    # train_number_index = data_number_index[history_size:train_data_num + history_size]
    test_number_index = np.arange(len(data_origin) - len(test_data_normalized), len(data_origin) - len(
        test_data_normalized) + test_data_num * predict_size, 1)

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

    # plt.plot(data_number_index, data_origin, 'b', label='real')
    plt.plot(test_number_index, data_origin[test_number_index], 'b', label='real')
    # plt.plot(train_number_index, train_predict, 'r--', label='prediction')
    plt.plot(test_number_index, test_predict, 'y--', label='prediction')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
