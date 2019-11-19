# /home/qinbo/anaconda3/envs/neural-py27/bin/python
# -*-  coding: utf-8 -*-
"""
注:
更多关于python 2 与 python 3 代码兼容的详情, 可参考:
[Cheat Sheet: Writing Python 2-3 compatible code](http://python-future.org/compatible_idioms.html)
"""

#filename:deep_neural_modeling.py
from __future__ import print_function
from __future__ import division
__author__='Qinbo Liu'
Comment = False
Code = True

import numpy as np
np.set_printoptions(threshold = np.infty)
import pandas as pd
import matplotlib.pyplot as plt
import random


'''
x=np.linspace(-5,5,1000)
y=[1/(1+np.exp(-i)) for i in x]
plt.plot(x,y)
plt.show()
'''
'''
consider diffference equation:
y_t=alpha + beta * y_{t-1} + epsilon_t, 
y_0=1, alpha=-0.25, beta=0.95
'''

# 1. 熟悉pandas.Series 的时序数据操作
seed=2018
np.random.seed(seed)
y_0=1.0
alpha=-0.25
beta=0.95
y=pd.Series(y_0)
print(y)
for i in y.index:
    print(y[i])
#plt.plot(y)
#plt.show()
num=10
for i in range(num):
    y_t=alpha+(beta*y_0)+np.random.uniform(-1,1)
    y[i]=y_t
    y_0=y_t
plt.plot(y)
plt.show()
print("y.index is: ", y.index)
for i in y.index:
    print(i, y[i])
print(y.sort_values())
z = y.sort_values()

fig=plt.figure(figsize=(10,5))

axes1 = fig.add_subplot(2,1,1)
axes1.set_xticklabels(z.index)
axes1.plot(z,'b')

axes2 = fig.add_subplot(2,1,2)
axes2.plot(y.sort_values(),'r')
plt.show()

plt.plot(z.index, z.values, 'y')
plt.show()

# 2.1
# 一直一来，我们都是用 TensorFlow 框架搭建深度神经网络，但其实 python
# 也提供了相应的统计和学习模块，比如我们要拟合函数 y = x**2
# 2.1.1
# 首先生成数据集 x 和标准数据 y = x **2
random.seed(2016)
sample_size =50
sample = pd.Series(random.sample(\
    range(-10000, 10000), sample_size)\
    )
x= sample /10000
y=x**2
plt.scatter(x,y)
plt.show()
if Comment:
    print(x.head(10))
    print(type(x))
    print(x.describe())
# 2.1.2
# 然后将数据加载进 dataSet ,格式是 [ ([x_input],[y_input]), ([x_input],[y_input]) .... ]
count = 0
dataSet = [([x.ix[count]], [y.ix[count]])]
count = 1
while count < sample_size:
    print("更新数据集:\nWorking on data item:", count)
    # 注意 loc 闭区间， iloc 和 ix 右端为开.
    dataSet = dataSet + [([x.ix[count,0]], [y.ix[count]])]
    count += 1
# 2.1.3
# 导入 neuralpy 包搭建神经网络并训练
import neuralpy
nn = neuralpy.Network(1, 3, 7, 1)
''' 包含两个隐藏层，分别含有 3 个节点和 7 个节点，其中输入一个节点，输出一个节点 '''
# 设置学习周期
epochs = 300
# 学习率设置
learning_rate = 0.5
print(" 现在训练模型:\ntraining model right now")
print("模型训练调用形式为\nnn.train(dataSet,epochs=300,learning_rate=0.5)")
nn.train(dataSet,epochs,learning_rate)

# 2.1.4
# 评估模型表现
count = 0
forecast = []
while count < sample_size:
    out = nn.forward(x[count]) # forward 表示前向传播
    print("Obs: ", count + 1, "y = ", round(y[count], 4),
          "forecast = ", round(pd.Series(out), 4))
    forecast.append(out[0])  # out是一个有一个数值元素的列表
    count += 1
mse = np.sqrt(np.sum(np.square(y.values - forecast)))
print("The MSE for the forecast is: ", round(mse, 4))
mse = np.linalg.norm(y.values-np.array(forecast))
print("The MSE for the forecast is: ", round(mse, 4))
# 2.1.5
# 模型表现做图
# 下面三条命令应该可以让图形可以显示中文， 这里我先注释掉，下次重启后可以开放为代码
if Code:
    from matplotlib.pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

#from matplotlib import font_manager
#from matplotlib.pylab import mpl
#fname = '/home/qinbo/anaconda3/envs/neural-py27/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf'
#myfont = font_manager.FontProperties(fname=fname)
#mpl.rcParams['font.serif'] = ['SimHei'] # 指定默认字体
#mpl.rcParams['axes.unicode_minus'] = False

myfont = ''  # 做图时预置的字体为空, 作为后面画图时的判断条件
# 可以使用下面这种方式（置于代码前面）
if Code: # Code -> 标题字体 SimSun; Comment -> 标题字体为 SimHei
    import matplotlib
    # matplotlib.use('qt4agg')
    from matplotlib import font_manager
    fname = '/home/qinbo/anaconda3/envs/neural-py27/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimSun.ttf'
    myfont = font_manager.FontProperties(fname=fname)
    from matplotlib.pylab import mpl
    mpl.rcParams['axes.unicode_minus'] = False
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1)
ax.scatter(x.values, y.values, color = "black")
ax.scatter(x.values, np.array(forecast), color = 'red')
ax.set_xlabel(u'X：50个随机值')
ax.set_ylabel(u'Y =X^2')
if myfont == '':
    ax.set_title(u'黑色为实际值, 红色为预测值： MSE = %f (字体是SimHei)' % mse)
else:
    ax.set_title(u'黑色为实际值, 红色为预测值： MSE = %f (字体是SimSun)' % mse, fontproperties = myfont)
plt.show()

# 3.
# 第3章 时序深度神经网络
# 3.1 时间数据描述
# Certificate of Entitlement (COE), 上牌照, 价格是由一个 open bidding system决定的.
# 我们获取这个价格的时间序列数据
print("\n\n\n******* 第3章 *******\n")
COE_has_been_downloaded = True
import urllib
url = "https://goo.gl/WymYzd"
loc = "./data/coe.xls"
if not COE_has_been_downloaded:
    urllib.urlretrieve(url, loc)
else:
    print("The COE data is already downloaded, we don't need to redownload it.")
# 3.2 清洗xls数据
# 3.2.1 用pandas读取xls数据文件
Excel_file = pd.ExcelFile(loc)
# 3.2.2 获取xls文件的sheet_names, 列表
print(Excel_file.sheet_names)
# [u'COE data']
# 3.2.3 将coe数据转为pandas DataFrame类型 并获取列名列表
spreadsheet = Excel_file.parse('COE data')
list_of_columns_of_spreadsheet = spreadsheet.columns.to_list()
print(spreadsheet.info())
# 3.2.4 获取目标变量数据(DataFrame)
# target 目标变量是历史价格 'COE$'
data = spreadsheet['COE$']
print(data.head())

# below is something else
if Comment:
    np.random.seed(2016)
    test_random_matrice=np.random.rand(3,4)
    print(test_random_matrice)
    print(test_random_matrice[:,0:4])
    #the following is to try pandas dataframe defn
    d={'col1':[1,2],'col2':[3,4]}
    df1=pd.DataFrame(data=d, index=('row1','row2'))
    print('df1 is:\n ', df1)
    print(df1['col1'])
    print(df1['col2'])
    np.random.seed(2019)
    ar1=np.random.randint(low=1,high=100,size=(4,8))
    np.random.seed(2019)
    ar2=np.random.randint(low=1,high=100,size=(4,8))
    print('ar1 is:\n',ar1)
    print('ar2 is:\n',ar2)
    print('ar1 = ar2 ', ar1==ar2)
    df2=pd.DataFrame(ar1, columns=[1,2,3,4,5,6,9,20],\
                     index=['r1','r2','r3','r4'])
    print('df2 is:\n',df2)
    df2_info=df2.describe()
    df3=pd.DataFrame(np.ndarray.tolist(df2_info._get_values), index=list(df2_info.index)\
                     ,columns=df2_info.columns)
    print('+++++,df3 is:\n',df3)
    print(df2.describe())
    print(type(df2_info))
    # The following is to compare the usages of loc, iloc, ix for DataFrame
    entry1=df1.loc['row1','col2']
    print('the row1, col2 entry of df1 is:\n',entry1)
    submatrix1=df1.iloc[0:2,0:2]
    print('**********\n',submatrix1)
    entry1_ix=df1.ix['row1','col2']
    submatrix1_ix=df1.ix[0:2,0:2]
    print(entry1_ix)
    print(submatrix1_ix)
    ######################
    
    for j in df2_info.index:
        print('type of j is', type(j))
        print(j)
    print(list(df2_info.columns))
    print(type(df2_info._get_values))
    df3.to_csv('df3DataFrame.csv',index=True,header=True)
    df4=pd.read_csv('df3DataFrame.csv')
    print(df4.index)
    print(df4)





