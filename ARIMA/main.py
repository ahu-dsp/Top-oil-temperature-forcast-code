# -*- coding: utf-8 -*-
import urllib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import datetime as dt
tw =24
datas = dt.date.today()
datas = str(datas)
print(datas)

# ————————差分并绘图，判断是否平稳的时候使用————————————————
# sentiment_short['diff_1'] = sentiment_short['UMCSENT'].diff(1)
#
# sentiment_short['diff_2'] = sentiment_short['diff_1'].diff(1)
#
# sentiment_short.plot(subplots=True, figsize=(18, 12))

plt.rcParams["font.sans-serif"] = ["SimHei"]

df = pd.read_csv(' ', usecols=[0], engine='python')


# —————————————————————改索引为时间—————————————————————————
data = df.values
data = data.astype('float32')
# data=data.set_index('数据时间')
data_len = len(data)
t = np.linspace(0, data_len, data_len)
plt.plot(t, data)
plt.show()  # 打印原始数据波形

# ————————————————————划分训练集与测试集——————————————————————
'''
目前从现有代码中看到过两种常用划分方法。一种是用比例划分。使用data[0:size]的形式。另外一种使用data.loc指令进行划分，注意，data.loc后面跟的
是标准的时间指标。
使用函数：data.loc[datatime：datatime，numb：numb]，前者是行索引，后者是列索引。
在接下来的程序中会把两种方式都写下来
'''

# ————————比例式————————
test_len  =24
# size = int(len(data) * 0.98)
size = len(data) - test_len
train, test = data[:size], data[size:]
# history = [data for data in train]
history = train
history = history.astype('float32')
# history = train
predictions = []
# model = sm.tsa.arima.ARIMA(history, order=(2,1,7))
# model_fit = model.fit()
# residuals = pd.DataFrame(model_fit.resid)
# # fig, ax = plt.subplots(1,2)
# residuals.plot(title="Residuals")
# residuals.to_csv("b64trainr.csv")
# plt.show()
st = len(train)
i = 0
len_test = len(test)
len_6 = int(len_test / tw)
for t in range(len_6):
    model = sm.tsa.arima.ARIMA(history, order=(2,1,7))
    model_fit = model.fit()
    output = model_fit.predict(start=st + i, end=st+i+tw-1)
    yhat = list(output)
    predictions.append(yhat)
    obs = test[t *tw : t * tw + tw]
    history = history
    history = np.append(history,obs)
    # history = np.array(history)
    i = i + tw
    print('第{}次'.format(t))
    print('predicted=%f, expected=%f' % (yhat[0], obs[0]))
#_______6步输出的时候predictions会成为(-1,6)的列表，以下三步是为了将其转化为一列多行的标准
predictions1 = np.array(predictions)
predictions1 = predictions1.reshape(-1,)
predictions = predictions1.tolist()
#___________记录预测值
predictions2 = pd.DataFrame(predictions1)
predictions2.to_csv(' ')

t_for_test_data1 = np.arange(size, len(data), 1)
t_for_test_data2 = np.arange(size, len(data), 1)
plt.plot(t_for_test_data1, test)
plt.plot(t_for_test_data2, predictions)
plt.legend(['y_true', 'y_pred'])
# plt.savefig('来自于s4-6.csv,拟合图')
plt.show()
print(len(predictions))


# ———————————— 获得残差 ————————————————
# res=test-predict
predictions = np.array(predictions)
res = test.reshape(-1) - predictions.reshape(-1)
residual = list(res)
plt.plot(residual)
plt.show()
# residual1 = pd.DataFrame(residual)
# residual1.to_csv("ARIMA6步残差预测结果_1680_1.csv")
# 打印指标
mae = metrics.mean_absolute_error(test, predictions)
mape = metrics.mean_absolute_percentage_error(test, predictions)*100
mse = metrics.mean_squared_error(test, predictions, squared=True)
rmse = metrics.mean_squared_error(test, predictions, squared=False)
R2 = metrics.r2_score(test, predictions)
print('mae={}'.format(mae))
print('mape={}'.format(mape))
print('MSE={}'.format(mse))
print('RMSE={}'.format(rmse))
print('R^2={}'.format(R2))

# ——————检查残差的正态性————————————
import seaborn as sns
from scipy import stats

# plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.title("Histogram plus estimated density")
sns.distplot(residual, fit=stats.norm)
plt.subplot(1, 2, 2)
plt.rcParams["font.sans-serif"] = ["SimHei"]
res = stats.probplot(residual, plot=plt)
plt.title("normal_qq")
plt.show()
