# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.sans-serif"] = ["SimHei"]
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
#————————差分并绘图，判断是否平稳的时候使用————————————————
# sentiment_short['diff_1'] = sentiment_short['UMCSENT'].diff(1)
#
# sentiment_short['diff_2'] = sentiment_short['diff_1'].diff(1)
#
# sentiment_short.plot(subplots=True, figsize=(18, 12))


#——————————————————————数据载入——————————————————————————
#df=pd.read_csv('本体变中性点套管(AO1)接线柱B相-hour.csv',parse_dates=['数据时间'])
df=pd.read_csv(' ', usecols=[0], engine='python')
df.info()            #数据表的基本信息（维度，列名称，数据格式，所占空间等）,不用打印就会直接显示，类似的有df.shape

#—————————————————————改索引为时间—————————————————————————
data=df.copy()
# data=data.set_index('数据时间')
t = np.arange(0,len(data),1)
plt.plot(t,data.values)
plt.show()           #打印原始数据波形
#——————————————————观察差分后的曲线情况——————————————————————
diff1 = data.diff(1)
diff2 = data.diff(2)
fig = plt.figure(figsize=(12, 8))
plt.title('diiff1')
ax1 = fig.add_subplot(111)
diff1.plot(ax=ax1)
plt.show()
fig = plt.figure(figsize=(12, 8))
plt.title('diff2')
ax2 = fig.add_subplot(111)
diff2.plot(ax=ax2)


plt.show()
#发现差分以后平稳性更好，使用差分以后的数据来进行判断
data1 = data.diff(1)
data = data1
data.dropna(inplace=True)
#————————————————————划分训练集与测试集——————————————————————
'''
目前从现有代码中看到过两种常用划分方法。一种是用比例划分。使用data[0:size]的形式。另外一种使用data.loc指令进行划分，注意，data.loc后面跟的
是标准的时间指标。
使用函数：data.loc[datatime：datatime，numb：numb]，前者是行索引，后者是列索引。
在接下来的程序中会把两种方式都写下来
'''
#————————data.loc式写法，这种方式比较繁琐，个人觉得不太好————————
# train=data.loc[:'2018/1/13 23:45:00',:]
# test=data.loc['2018/1/14 0:00:00':,:]
#————————比例式————————
size = int(len(data) * 0.90)
train, test = data[0:size], data[size:len(data)]
#————————————平稳性检验（adf）————————————————
print('___平稳性检验___')
print(sm.tsa.stattools.adfuller(train))   #确定数据是否是平稳的，只有平稳序列才可以进行arima，如果不是就要先进行差分
#————————————纯随机白噪声检验——————————————————(显著水平为0。05)
print('___白噪声检验___')
print(acorr_ljungbox(train, lags = [6, 12],boxpierce=True))
#——————————计算ACF，PACF，这是确定arima模型的超参数的最关键的方法————————————
'''
ACF:自相关函数，代表P参数（第一个参数）
PACF：偏自相关函数，代表q参数（第三个参数）
具体的确定方法在笔记里有写过
自相关系数会很快衰减向0，所以可认为是平稳序列。
'''
plt.rcParams["font.sans-serif"] = ["SimHei"]
acf=plot_acf(train,lags=70)
plt.title("自相关图")
plt.show()

pacf=plot_pacf(train,lags=70)
plt.title("偏自相关图")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.show()       #用绘图和判断截尾或者进入平稳性的方法具有很大的主观性，所以采用aic和bic方法来辅助判断————信息准则定阶（另外还有HQIC可以使用）
trend_evaluate = sm.tsa.arma_order_select_ic(train, ic=['aic', 'bic','hqic'], trend='n', max_ar=10, max_ma=10)#为了控制计算量，在设置AR=6，MA设置为4（源程序为20,6）。 但是这样带来的坏处是可能为局部最优。
print('train AIC', trend_evaluate.aic_min_order)
print('train BIC', trend_evaluate.bic_min_order)
print('train HQIC', trend_evaluate.hqic_min_order)

