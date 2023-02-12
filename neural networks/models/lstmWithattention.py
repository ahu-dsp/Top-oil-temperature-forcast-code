

import torch

"""
【注】

这只是个测试LSTM with attention的文件
参考文献：https://zhuanlan.zhihu.com/p/410031664
"""
input_size = 4
hidden_size = 5
batch_size = 2
seq_len = 3
lstm_linear = torch.nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True
                            )

# 打印网络结构
print(lstm_linear)
# input & output(batch_size, seq_len, num_directions * hidden_size)

input = torch.randn((batch_size, seq_len, input_size))
print('shape of input: {}'.format(input.shape))

output, (h,c) = lstm_linear(input)
print('shape of output: {}'.format(output.shape))
print(output)

w_omiga = torch.randn(batch_size, hidden_size, 1)
print('shape of w_omiga: {}'.format(w_omiga.shape))

H = torch.nn.Tanh()(output)
print('shape of H: {}'.format(H.shape))

weights = torch.nn.Softmax(dim=-1)(torch.bmm(H, w_omiga).squeeze()).unsqueeze(dim=-1).repeat(1, 1, hidden_size)
print('shape of weights: {}'.format(weights.shape))

atten_output = torch.mul(output, weights).sum(dim=1)  # atten_output : [batch_size, hidden_size]
print('shape of atten_output: {}'.format(atten_output.shape))

print('-------------------')
print('input: {}'.format(input))
print('-------------------')
print('output: {}'.format(output))
print('-------------------')
print('w_omiga: {}'.format(w_omiga))
print('-------------------')
print('H: {}'.format(H))
print('-------------------')
print('weights: {}'.format(weights))
print('--------output[:,-1,:] vs atten_output-----------')
print('output: {}'.format(output[:, -1, :]))
print('atten_output: {}'.format(atten_output))
