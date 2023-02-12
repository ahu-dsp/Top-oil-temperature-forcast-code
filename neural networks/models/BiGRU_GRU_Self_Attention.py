
from math import sqrt

import torch
from torch import nn

# from torchinfo import summary

# 定义训练的设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 搭建神经网络
class BiGRU_GRU(nn.Module):
    def __init__(self, l_hidden_size,
                 l_num_layers,
                 g_hidden_size,
                 g_num_layers,
                 dim_k,
                 dim_v,
                 output_size,
                 history_size=1,
                 ):
        """
        网络架构：单向LSTM层 + 全连接层

        :param l_input_size: 输入的特征数量
        :param hidden_size: 隐含层的神经元个数
        :param l_num_layers: 隐含层个数
        :param output_size: 输出大小，即预测步长
        :param history_size: 时间窗长度
        :param batch_first: 设置默认值 True, 模型的输入输出类型：(batch_size, seq_len, num_directions * hidden_size)
        :param dropout: 默认值 0
        """
        super(BiGRU_GRU, self).__init__()
        self.l_hidden_size = l_hidden_size
        self.l_num_layers = l_num_layers

        self.g_hidden_size = g_hidden_size
        self.g_num_layers = g_num_layers

        self.history_size = history_size


        self.lstm = nn.GRU(
            input_size=1
            , hidden_size=self.l_hidden_size
            , num_layers=self.l_num_layers
            , batch_first=True
            , dropout=0
            , bidirectional=True
        )
        self.gru = nn.GRU(
            input_size=self.l_hidden_size*2
            , hidden_size=self.g_hidden_size
            , num_layers=self.g_num_layers
            , batch_first=True
            , dropout=0
            , bidirectional=False
        )

        self.q = nn.Linear(self.g_hidden_size, dim_k)
        self.k = nn.Linear(self.g_hidden_size, dim_k)
        self.v = nn.Linear(self.g_hidden_size, dim_v)
        self._norm_fact = 1 / sqrt(dim_k)

        self.fc1 = nn.Linear(dim_v, 32)

        self.fc2 = nn.Linear(32, output_size)

    def forward(self, input_seq):
        # batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]

        h_batch_size = len(input_seq)
        # 对单特征数据做预测时，每个序列长度即可当作是输入特征数input_size, 也可看做是seq_len, input_size=1

        lstm_h_0 = torch.randn(self.l_num_layers * 2, h_batch_size, self.l_hidden_size).to(device)

        gru_h_0 = torch.randn(self.g_num_layers * 1, h_batch_size, self.g_hidden_size).to(device)
        # gru_c_0 = torch.randn(self.g_num_layers * 1, h_batch_size, self.g_hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        # h_t, c_t)
        lstm_output, lstm_h_t = self.lstm(input_seq, lstm_h_0)
        output, gru_h_t = self.gru(lstm_output, gru_h_0)
        # output, hidden_cell = self.lstm(input_seq, hidden_cell)
        # print("the shape of output:{}".format(output.shape))

        # 引入注意力机制
        Q = self.q(output)  # Q: batch_size * seq_len * dim_k
        K = self.k(output)  # K: batch_size * seq_len * dim_k
        V = self.v(output)  # V: batch_size * seq_len * dim_v
        # Q * K.T() # batch_size * seq_len * seq_len
        attention_weights = nn.Softmax(dim=-1)(torch.bmm(Q, K.permute(0, 2, 1)) * self._norm_fact)
        atten_output = torch.bmm(attention_weights, V)  # Q * K.T() * V # batch_size * seq_len * dim_v

        # pred = self.linear(atten_output)    # batch_size * seq_len * output_size
        # return pred[:, -1, :]
        # atten_output = torch.flatten(atten_output, start_dim=1)
        # pred = self.fc1(atten_output.sum(1))
        pred = self.fc1(atten_output)
        pred = self.fc2(pred)
        return pred[:, -1, :]

        # pred = self.linear1(output)
        # # pred = torch.tanh(pred)
        # pred = self.linear2(pred)
        # # print("the shape of pred:{}".format(pred.shape))
        # pred = pred[:, -1, :]  # 获得最后一个时间步的输出
        # return pred


# 验证模型的正确性
if __name__ == '__main__':
    batch_size = 32
    seq_len = 4
    gru_linear = BiGRU_GRU(l_input_size=1, l_hidden_size=1, l_num_layers=2, g_hidden_size=1, output_size=1, dropout=0)
    gru_linear = gru_linear.to(device)
    # 打印网络结构
    print(gru_linear)
    # input & output(batch_size, seq_len, num_directions * hidden_size)
    input = torch.ones((batch_size, seq_len)).to(device)
    output = gru_linear(input)
    print(output.shape)
    print(output)
