
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor


def bp_model():
    bp = MLPRegressor(
        hidden_layer_sizes=(100,100),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=1e-3,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=None

    )
    return bp

# 下面代码，属于拟合验证，
if __name__ == '__main__':
    # 生成模拟的自变量数据集
    X = np.linspace(-3.14, 3.14, 400)
    # 转换数据类型
    X1 = X.reshape(-1, 1)
    # 生成模拟的目标变量数据
    y = np.sin(X) + 0.3 * np.random.rand(len(X))
    # 创建模型对象
    clf = bp_model()
    print(clf)
    # 训练模型
    clf_fit = clf.fit(X1, y)
    # 预测
    y2 = clf.predict(X1)
    # 画图
    plt.plot(X, y, 'b', label='real')
    plt.plot(X, y2, 'r-', label='real')
    plt.show()


