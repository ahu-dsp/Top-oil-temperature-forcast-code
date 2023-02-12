
import os
import random
import numpy as np
import torch


def set_seed(seed):
    """
    【- - - - - 注 - - - - -】

    1. 有时修改了网络、做了数据预处理、调参等，希望测试下实验结果是否有变化，
    2. 若不固定随机数种子，那由于神经网络里的一些随机过程和cudnn的优化等问题，会使每次实验结果不一样。
    3. 通过固定随机数种子，能够使实验结果可复现，方便调参、调试网络。
    4. 随机数种子也不要故意去测试得到一个使你实验结果最好的，因为这样也并不能代表你模型的鲁棒性，所以随机数随便整一个就好，
    5. 如果不是为了实验可复现时，可以不用固定，多测几组来取平均，这样能更好的评估模型的性能。

    【- - - - - 坑 - - - - - 】

    1. 在jupyter notebook里要注意，比如模型预测要和随机数种子的设置要在同个code cell里才有用

    :param seed: any random number
    :return:
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    # os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    # torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    # torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现
