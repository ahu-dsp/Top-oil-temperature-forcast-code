# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from matplotlib.ticker import MultipleLocator
from sklearn import metrics
from regression_metrics import get_mae, get_mape, get_mse, get_rmse, get_r2

arima_pre = pd.read_csv(' ',usecols=[1]).astype('float32')
res_pre   = pd.read_csv(' ',usecols=[1]).astype('float32')
org = pd.read_csv(' ',usecols=[0]).astype('float32')


arima_pre1  = np.array(arima_pre)
res_pre = np.array(res_pre)
org1 = np.array(org)


arima_res_pre = arima_pre1[-len(res_pre):] + res_pre
print(arima_res_pre)
result8 = pd.DataFrame(arima_res_pre)
result8.to_csv(' ')
org1 = org1[-len(res_pre):]
t = np.linspace(1,len(arima_res_pre)+1,len(arima_res_pre))
plt.subplot(411)
plt.plot(t,arima_res_pre,'b', label='pre')
plt.plot(t,org1,'r',label= 'org')
plt.title(' ')
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.legend()
plt.subplot(412)
plt.plot(t,org1)
plt.title('org')
plt.subplot(413)
plt.plot(t,arima_pre1[-len(res_pre):])
plt.plot(t,org1,'r',label= 'org')
plt.legend()
plt.title('ar_pre')
plt.subplot(414)
plt.plot(t,res_pre)
plt.title('res_pre')
plt.show()



plt.figure(figsize=(4.5, 3))
plt.xlim(1,24)
plt.plot(t,org1,'r',label= '原始数据')
plt.plot(t,arima_res_pre,label= '预测数据')
plt.title('  ')
x = MultipleLocator(4)
ax=plt.gca()
ax.xaxis.set_major_locator(x)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.legend()
plt.show()
print('---------评估指标--------')
mae = get_mae(org1, arima_res_pre)
print('MAE:{}'.format(mae))
# 平均绝对百分比误差(MAPE)
mape = get_mape(org1, arima_res_pre)*100
print('MAPE:{}'.format(mape))
# 均方误差(MSE)
mse = get_mse(org1, arima_res_pre)
print('MSE:{}'.format(mse))
# 均方根误差(RMSE)
rmse = get_rmse(org1, arima_res_pre)
print('RMSE:{}'.format(rmse))
# 判别系数R^2
r2 = get_r2(org1, arima_res_pre)
print('R^2:{}'.format(r2))




