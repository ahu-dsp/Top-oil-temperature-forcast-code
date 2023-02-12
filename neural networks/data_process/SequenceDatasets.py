import torch
from torch.utils.data import Dataset


class MyData(Dataset):

    def __init__(self, data_x, data_y):
        self.Data = data_x
        self.Label = data_y

    # 重写__getitem__，通过给定索引获取数据和标签，默认一次只能获取一个数据及对应标签
    def __getitem__(self, index):
        data_tensor = torch.from_numpy(self.Data[index])  # numpy-->tensor
        label_tensor = torch.from_numpy(self.Label[index]).view(-1)
        return data_tensor, label_tensor

    # 重写__len__，获取数据的大小（size）
    def __len__(self):
        return len(self.Data)