

import torch
from torch import nn

# from torchinfo import summary

# 定义训练的设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 搭建神经网络
class BiLstmNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, history_size=1, dropout=0):
        """
         网络架构：双向LSTM层 + 全连接层

         :param input_size: 输入的特征数量
         :param hidden_size: 隐含层的神经元个数
         :param num_layers: 隐含层个数
         :param output_size: 输出大小，即预测步长
         :param history_size: 时间窗长度
         :param dropout: 默认值 0
         """
        super(BiLstmNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.history_size = history_size

        self.lstm = nn.LSTM(
            input_size=input_size
            , hidden_size=self.hidden_size
            , num_layers=self.num_layers
            , batch_first=True
            , dropout=dropout
            , bidirectional=True    # 双向
        )
        self.linear1 = nn.Linear(2 * self.hidden_size, 1)
        self.linear2 = nn.Linear(1, output_size)

    def forward(self, input_seq):
        # batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]

        h_batch_size = len(input_seq)
        input_seq = input_seq.unsqueeze(2)  # (batch_size, seq_len)-->(batch_size, seq_len, input_size)
        h_0 = torch.zeros(2 * self.num_layers, h_batch_size, self.hidden_size).to(device)
        c_0 = torch.zeros(2 * self.num_layers, h_batch_size, self.hidden_size).to(device)
        hidden_cell = (h_0, c_0)  # 对每个新句子，都需要重新初始化隐藏状态，因为两个句子属于不同句子

        # output: (batch_size, seq_len, num_directions * hidden_size)
        # h_t & c_t: (num_directions * num_layers, batch_size, hidden_size)
        output, (h_t, c_t) = self.lstm(input_seq, hidden_cell)
        # print("the shape of output:{}".format(output.shape))

        pred = self.linear1(output)
        # pred = torch.tanh(pred)
        pred = self.linear2(pred)
        # print("the shape of pred:{}".format(pred.shape))

        #
        pred = pred[:, 0, -self.hidden_size:]+pred[:, -1, :self.hidden_size]
        return pred


# 验证模型的正确性
if __name__ == '__main__':
    batch_size = 32
    seq_len = 1
    lstm_linear = BiLstmNet(input_size=10, hidden_size=20, num_layers=2, output_size=1, dropout=0)

    # 打印网络结构
    print(lstm_linear)
    # input & output(batch_size, seq_len, num_directions * hidden_size)
    input = torch.ones((batch_size, 10))
    output = lstm_linear(input)
    print(output.shape)
    print(output)
