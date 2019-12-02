# /home/qinbo/anaconda3/envs/neural-py27/bin/python
# -*-  coding: utf-8 -*-
"""
注:
更多关于python 2 与 python 3 代码兼容的详情, 可参考:
[Cheat Sheet: Writing Python 2-3 compatible code](http://python-future.org/compatible_idioms.html)
"""

# filename:deep_neural_modeling.py
from __future__ import print_function
from __future__ import division

__author__ = 'Qinbo Liu'
Comment = False
Code = True

import time
import numpy as np

np.set_printoptions(threshold=np.infty)
import pandas as pd
import matplotlib.pyplot as plt
import random
import warnings

warnings.filterwarnings('ignore')

'''
formula
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

start = time.time()  # 我们计算程序一共花了多久
# Content 是代码目录, 可以作为输出文档
Content = ''
# 1. 熟悉pandas.Series 的时序数据操作
Content += "1. 熟悉pandas.Series 的时序数据操作\n"
seed = 2018
np.random.seed(seed)
y_0 = 1.0
alpha = -0.25
beta = 0.95
y = pd.Series(y_0)
print(y)
for i in y.index:
    print(y[i])  # plt.plot(y)
# plt.show()
num = 10
for i in range(num):
    y_t = alpha + (beta * y_0) + np.random.uniform(-1, 1)
    y[i] = y_t
    y_0 = y_t
plt.plot(y)
plt.show()
print("y.index is: ", y.index)
for i in y.index:
    print(i, y[i])
print(y.sort_values())
z = y.sort_values()

fig = plt.figure(figsize=(10, 5))

axes1 = fig.add_subplot(2, 1, 1)
axes1.set_xticklabels(z.index)
axes1.plot(z, 'b')

axes2 = fig.add_subplot(2, 1, 2)
axes2.plot(y.sort_values(), 'r')
plt.show()

plt.plot(z.index, z.values, 'y')
plt.show()

UseMacbook = True
# /Users/qinbo/miniconda3/envs/neural-py27/lib/python2.7/site-packages/matplotlib/mpl-data/matplotlibrc
Content += "下面三条命令应该可以让图形可以显示中文， 这里我先注释掉，下次重启后可以开放为代码\n"
if UseMacbook:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] =False
else:
    from matplotlib.pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体

    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 2.1
# 一直一来，我们都是用 TensorFlow 框架搭建深度神经网络，但其实 python
# 也提供了相应的统计和学习模块，比如我们要拟合函数 y = x**2
# 2.1.1
# 首先生成数据集 x 和标准数据 y = x **2
Content += "2.1\n"
Content += "一直一来，我们都是用 TensorFlow 框架搭建深度神经网络，但其实 python\n" \
           "也提供了相应的统计和学习模块，比如我们要拟合函数 y = x**2\n"
Content += "2.1.1\n"
Content += "首先生成数据集 x 和标准数据 y = x **2\n"
Chapter2NeedToRun = [False, True][0]
if Chapter2NeedToRun:
    random.seed(2016)
    sample_size = 50
    sample = pd.Series(random.sample( \
        range(-10000, 10000), sample_size) \
        )
    x = sample / 10000
    y = x ** 2
    plt.scatter(x, y)
    plt.show()
    if Comment:
        print(x.head(10))
        print(type(x))
        print(x.describe())
    # 2.1.2
    # 然后将数据加载进 dataSet ,格式是 [ ([x_input],[y_input]), ([x_input],[y_input]) .... ]
    Content += "2.1.2\n"
    Content += "然后将数据加载进 dataSet ,格式是 [ ([x_input],[y_input]), ([x_input],[y_input]) .... ]\n"
    count = 0
    dataSet = [([x.ix[count]], [y.ix[count]])]
    count = 1
    while count < sample_size:
        print("更新数据集:\nWorking on data item:", count)
        # 注意 loc 闭区间， iloc 和 ix 右端为开.
        dataSet = dataSet + [([x.ix[count, 0]], [y.ix[count]])]
        count += 1
    # 2.1.3
    # 导入 neuralpy 包搭建神经网络并训练
    Content += "2.1.3\n"
    Content += "导入 neuralpy 包搭建神经网络并训练\n"
    import neuralpy
    
    nn = neuralpy.Network(1, 3, 7, 1)
    ''' 包含两个隐藏层，分别含有 3 个节点和 7 个节点，其中输入一个节点，输出一个节点 '''
    # 设置学习周期
    epochs = 300
    # 学习率设置
    learning_rate = 0.5
    print(" 现在训练模型:\ntraining model right now")
    print("模型训练调用形式为\nnn.train(dataSet,epochs=300,learning_rate=0.5)")
    nn.train(dataSet, epochs, learning_rate)
    
    # 2.1.4
    # 评估模型表现
    Content += "2.1.4\n评估模型表现\n"
    count = 0
    forecast = []
    while count < sample_size:
        out = nn.forward(x[count])  # forward 表示前向传播
        print("Obs: ", count + 1, "y = ", round(y[count], 4),
              "forecast = ", round(pd.Series(out), 4))
        forecast.append(out[0])  # out是一个有一个数值元素的列表
        count += 1
    mse = np.sqrt(np.sum(np.square(y.values - forecast)))
    print("The MSE for the forecast is: ", round(mse, 4))
    mse = np.linalg.norm(y.values - np.array(forecast))
    print("The MSE for the forecast is: ", round(mse, 4))
    # 2.1.5
    # 模型表现做图
    # 下面三条命令应该可以让图形可以显示中文， 这里我先注释掉，下次重启后可以开放为代码
    Content += "2.1.5\n模型表现做图\n"

    # from matplotlib import font_manager
    # from matplotlib.pylab import mpl
    # fname = '/home/qinbo/anaconda3/envs/neural-py27/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf'
    # myfont = font_manager.FontProperties(fname=fname)
    # mpl.rcParams['font.serif'] = ['SimHei'] # 指定默认字体
    # mpl.rcParams['axes.unicode_minus'] = False
    
    myfont = ''  # 做图时预置的字体为空, 作为后面画图时的判断条件
    # 可以使用下面这种方式（置于代码前面）
    if Comment:  # Code -> 标题字体 SimSun; Comment -> 标题字体为 SimHei
        import matplotlib
        # matplotlib.use('qt4agg')
        from matplotlib import font_manager
        
        fname = '/home/qinbo/anaconda3/envs/neural-py27/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimSun.ttf'
        myfont = font_manager.FontProperties(fname=fname)
        from matplotlib.pylab import mpl
        
        mpl.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x.values, y.values, color="black")
    ax.scatter(x.values, np.array(forecast), color='red')
    ax.set_xlabel(u'X：50个随机值')
    ax.set_ylabel(u'Y =X^2')
    if myfont == '':
        ax.set_title(u'黑色为实际值, 红色为预测值： MSE = %f (字体是SimHei)' % mse)
    else:
        ax.set_title(u'黑色为实际值, 红色为预测值： MSE = %f (字体是SimSun)' % mse, fontproperties=myfont)
    fig = plt.gcf()
    plt.show()
    fig.savefig('./pictures/nn2_nonlinear_fitting.png')

# 3.
# 第3章 时序深度神经网络
# 3.1 时间数据描述
# Certificate of Entitlement (COE), 上牌照, 价格是由一个 open bidding system决定的.
# 我们获取这个价格的时间序列数据
Chapter3NeedToRun = [False, True][0]
if Chapter3NeedToRun:
    Content += "3. 第3章 时序深度神经网络\n3.1 " \
               "时间数据描述\nCertificate of " \
               "Entitlement (COE),\n我们获取这个价格的时间序列数据\n"
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
    Content += "3.2 清洗xls数据\n3.2.1 用pandas读取xls数据文件\n3.2.2 " \
               "获取xls文件的sheet_names, 列表\n3.2.3 " \
               "将coe数据转为pandas DataFrame类型 并获取列名列表\n3.2.4" \
               " 获取目标变量数据(DataFrame)\n(a)" \
               " target 目标变量是历史价格 'COE$'\n(b) 调整错误数据\n"
    Excel_file = pd.ExcelFile(loc)
    # 3.2.2 获取xls文件的sheet_names, 列表
    print(Excel_file.sheet_names)
    # [u'COE data']
    # 3.2.3 将coe数据转为pandas DataFrame类型 并获取列名列表
    spreadsheet = Excel_file.parse('COE data')
    list_of_columns_of_spreadsheet = spreadsheet.columns.to_list()
    print("list_of_columns is: \n", list_of_columns_of_spreadsheet)
    print(spreadsheet.info())
    # 3.2.4 获取目标变量数据(DataFrame)
    # (a) target 目标变量是历史价格 'COE$'
    data = spreadsheet['COE$']
    print(data.head())
    # (b) 调整错误数据
    #     打印错误日期段
    print(spreadsheet['DATE'][193:204])
    #     设定为正确日期
    spreadsheet.set_value(index=194, col='DATE', value='2004-02-15')
    spreadsheet.set_value(index=198, col='DATE', value='2004-04-15')
    spreadsheet.set_value(index=202, col='DATE', value='2004-06-15')
    # print(spreadsheet['DATE'][193:204])
    loc = './data/coe.csv'
    COE_CSV_HAS_GENERATED = True
    if not COE_CSV_HAS_GENERATED:
        spreadsheet.to_csv(loc, index=None)
    
    # 3.3此处为激活函数、梯度下降对学习
    Content += "3.3 此处为激活函数、梯度下降对学习\n"
    # 3.4 利用 sklearn preprocessing 对 目标变量 data 进行归一化
    Content += "3.4 利用 sklearn preprocessing 对 目标变量 data 进行归一化\n"
    from sklearn import preprocessing
    
    x = data
    x = np.array(x).reshape(len(x), )
    # np.log 取自然对数
    x = np.log(x)  # x.shape = (265, ), x 是一个向量
    x = x.reshape(-1, 1)  # scaler.fit_transform 接受二维np.array参数
    # scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
    scaler = preprocessing.MinMaxScaler(
        feature_range=(0, 1))
    x = scaler.fit_transform(x)
    x = x.reshape(-1)  # 调整x shape 到之前 x.shape = (265,)
    print(round(x.min(), 2), round(x.max(), 2))
    
    # 3.5 考察偏自相关系数(Autocorrelation)
    Content += "3.5 考察偏自相关系数\n"
    from statsmodels.tsa.stattools import pacf
    
    x_pacf = pacf(x, nlags=5, method='ols')
    print(x_pacf)
    # 3.6 利用 nnet-ts 来进行下一个时间的价格预测, 预测的舒适区间为 ＋／－$1,500
    Content += "3.6 利用 nnet-ts(依赖tensorflow, keras等) 来进行下一个时间的价格预测, 预测的舒适区间为 ＋／－$1,500\n"
    # import tensorflow as tf
    from nnet_ts import *
    import os
    
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # 去除 tensorflow 的log信息
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    # 去除打印区域的日志信息: (DEBUG, INFO, WARNING, ERROR, EMERGENCY 等 Level)
    import logging
    
    Logger = logging.getLogger()
    Logger.setLevel(level=logging.WARNING)  # 只显示 LEVEL >= WARNING 的日志信息
    # 解决 libiomp5.dylib already initialized 的错误
    # 3.6.1 利用TimeSeriesNet(hidden_layers = [7, 3] 预测下次价格(连续预测12次))
    Content += "3.6.1 利用TimeSeriesNet(hidden_layers = [7, 3] 预测下次价格(连续预测12次))\n"
    count = 0  # 计数器: 对ahead次数对循环
    ahead = 12  # 每次预测12个值
    forecast = []  # 存储预测值, 12 个数字组成的数组
    while count < ahead:
        end = len(x) - ahead + count
        np.random.seed(2016)
        tsnn = TimeSeriesNnet(hidden_layers=[7, 3],
                              activation_functions=["tanh", "tanh"])
        tsnn.fit(x[0: end], lag=1, epochs=100)
        # out 是含有一个数值变量的列表
        out = tsnn.predict_ahead(n_ahead=1)
        print("Obs: ", count + 1, "x = ",
              round(x[count], 4), " prediction = ", round(pd.Series(out), 4))
        
        forecast.append(out[0])
        count += 1
    
    forecast1 = scaler.inverse_transform(np.array(forecast).reshape((-1, 1)))
    forecast1 = np.exp(forecast1)
    print(np.round(forecast1, 1))
    
    # 3.6.2 做预测效果比较图
    Content += "3.6.2 做预测效果比较图\n"
    # (a) 所有时间数据都显示
    Content += "(a) 所有时间数据都显示\n"
    plt.plot(range(0, ahead), forecast1.reshape(-1), '-r', label=u"预测", linewidth=1)
    real = np.array(data).reshape(-1)
    plt.plot(range(0, ahead), real[len(real) - ahead: len(real)], color='black', label=u"实际", linewidth=1)
    comfort_lower = real - 1500.0
    comfort_super = real + 1500.0
    plt.plot(range(-1 * (len(real) - ahead), 0), real[0: len(real) - ahead], '-b', label=u"历史价格", linewidth=1)
    plt.xlabel(u"预测时间为正, 历史时间为负")
    plt.ylabel(u"价格, 新加坡币")
    plt.legend()
    plt.show()
    
    # (b) 显示部分历史时间数据
    Content += "(b) 显示部分历史时间数据\n"
    history_time_length = 50
    plt.plot(range(0, ahead), forecast1.reshape(-1), '-r', label=u"预测", linewidth=1)
    plt.plot(range(0, ahead), real[len(real) - ahead: len(real)], color='black', label=u"实际", linewidth=1)
    plt.plot(range(0, ahead), comfort_lower[len(comfort_lower) - ahead: len(comfort_lower)], '--k',
             label=u"实际价格 - $1500", linewidth=1)
    plt.plot(range(0, ahead), comfort_super[len(comfort_super) - ahead: len(comfort_super)], '--k',
             label=u"实际价格 + $1500", linewidth=1)
    plt.plot(range(-history_time_length, 0),
             real[len(real) - ahead - history_time_length: len(real) - ahead], '-b', label=u"历史价格", linewidth=1)
    plt.xlabel(u"预测时间为正, 历史时间为负")
    plt.ylabel(u"价格, 新加坡币")
    plt.legend()
    fig = plt.gcf()
    plt.show()
    fig.savefig('./pictures/nn3_forecast.png')
    # 3.7 参考文献
    # [1] Kasstra, Iebeling, Milton Boyd,
    #     Designing a neural network for forecasting financial and economic time series. Neuro-computing,
    #     10.3 (1996): 215-236
    # [2] Frank, Ray J., Neil Davey, and Stephen P. Hunt,
    #     Time series prediction and neural networks. Journal of intelligent and robotic systems
    #     31.1-3 (2001): 91-103
    Content += "3.7 参考文献\n"
    Content += "[1] Kasstra, Iebeling, Milton Boyd,\n    " \
               "Designing a neural network for forecasting financial " \
               "and economic time series. Neuro-computing,\n    " \
               "10.3 (1996): 215-236\n"
    Content += "[2] Frank, Ray J., Neil Davey, and Stephen P. Hunt,\n    " \
               "Time series prediction and neural networks. Journal of intelligent and robotic systems\n" \
               "    31.1-3 (2001): 91-103\n"
# time.sleep(2)

# 4. 第4章 模型中加入其它特征(Additional Attributes)
print("4. 第4章 模型中加入其它特征(Additional Attributes)")
Content += "4. 第4章 模型中加入其它特征(Additional Attributes)\n"
Chapter4NeedToRun = [False, True][0]  # 控制这一章代码是否执行, 提高程序运行效率
if Chapter4NeedToRun:
    loc = './data/coe.csv'  # 事实上, 程序运行至此, loc 的值就是 coe.csv 路径
    temp = pd.read_csv(loc)
    print("查看coe.csv前几行数据")
    print(temp.head())
    # 4.1 删除日期列
    Content += "4.1 删除日期列\n"
    print("我们删除日期列数据(DATE)")
    # temp.columns = Index([u'DATE', u'COE$', u'COE$_1', u'#Bids', u'Quota', u'Open?'], dtype='object')
    # temp.drop(temp.columns[[0]], axis=1)
    data = temp.drop(temp.columns[0], axis=1)
    print("得到:")
    print(data.head())
    # 4.2 其它特征含义描述
    Content += "4.2 其它特征含义描述\n"
    # (a) COE$_1, 上一时间价格
    # (b) #Bids 居民竞价次数
    # (c) Quota 证书可获得量
    # (d) Open? Bid是open还是closed
    Content += "(a) COE$_1, 上一时间价格\n(b) #Bids 居民竞价次数\n(c) " \
               "Quota 证书可获得量\n(d) Open? Bid是open还是closed\n"
    
    # 4.3 target数据, 特征数据处理
    Content += "4.3 target数据, 特征数据处理\n"
    # 4.3.1 target 数据: price
    Content += "4.3.1 target 数据: price\n"
    y = data['COE$']
    # 4.3.2 对 "price_lag_1="COE$_1", "#Bids", "Quota" 进行log变换
    Content += '4.3.2 对 "price_lag_1="COE$_1", "#Bids", "Quota" 进行log变换\n'
    x = data.drop(data.columns[[0, 4]], axis=1)
    x = x.apply(np.log)
    # 4.3.3 加入 binary 特征 "Open?"
    Content += '4.3.3 加入 binary 特征 "Open?"\n'
    x = pd.concat([x, data['Open?']], axis=1)
    print("x.head():")
    print(x.head())
    print("x.tail():")
    print(x.tail())
    # 4.3.4  用sklearn.preprocessing 进行归一化
    Content += "4.3.4 用sklearn.preprocessing 进行归一化\n"
    from sklearn import preprocessing
    
    scaler_x = preprocessing.MinMaxScaler(
        feature_range=(0, 1))
    x = np.array(x).reshape(len(x), 4)
    x = scaler_x.fit_transform(x)
    print("输入特征归一化后x二维np矩阵的前4行数据: ")
    print(x[0:4, :])
    scaler_y = preprocessing.MinMaxScaler(
        feature_range=(0, 1))
    y = np.array(y).reshape(len(y), 1)
    y = np.log(y)
    y = scaler_y.fit_transform(y)
    print("输入特征归一化log(y)后二维np矩阵的前4行数据: ")
    print(y[0:4, :])
    
    # 4.4 使用 pyneurgen工具(pip install) 进行深度学习建模
    Content += "4.4 使用 pyneurgen工具(pip install) 进行深度学习建模\n"
    # 4.4.1 需要将 input:x, output: y 转换为list
    y = y.tolist()
    x = x.tolist()
    # 4.4.2 导入 NeuralNet 工具
    from pyneurgen.neuralnet import NeuralNet
    
    # 4.4.3 指定网络结构
    Content += "4.4.1 需要将 input:x, output: y 转换为list\n" \
               "4.4.2 导入 NeuralNet 工具\n" \
               "4.4.3 指定网络结构\n"
    import random
    
    random.seed(2019)
    nn4 = NeuralNet()
    nn4.init_layers(4, [7, 3], 1)
    # 4.4.4 随机初始化权重weight 与 偏差bias
    Content += "4.4.4 随机初始化权重 weight 与偏差 bias(脑补随机梯度下降)\n"
    nn4.randomize_network()
    nn4.set_halt_on_extremes(True)
    # 4.4.5 选择学习速率
    Content += "4.4.5 选择学习速率\n"
    nn4.set_random_constraint(0.5)
    nn4.set_learnrate(0.05)
    # 4.4.6 指定训练测试结构
    Content += "4.4.6 指定训练测试结构\n"
    nn4.set_all_inputs(x)
    nn4.set_all_targets(y)
    length = len(x)
    learn_end_point = int(length * 0.95)
    # 4.4.6.1 set_learn_range 和 set_test_range 为左右闭区间
    Content += "4.4.6.1 set_learn_range 和 set_test_range 为左右闭区间\n"
    print("lenth = %d" % length)
    print("learn_end_point = %d" % learn_end_point)
    nn4.set_learn_range(0, learn_end_point)
    # BUG REPORT 相信这里有一个bug, 无法测试最后一个时间点的数据数据
    Content += "BUG REPORT 相信这里self.set_learn_range()有一个bug, 无法测试最后一个时间点的数据数据\n"
    nn4.set_test_range(learn_end_point + 1, length - 1)
    # 4.4.6.2 选择每层网络的激活函数
    Content += "4.4.6.2 选择每层网络的激活函数\n"
    nn4.layers[1].set_activation_type('tanh')
    nn4.layers[2].set_activation_type('tanh')
    # 4.4.7 模型运行
    Content += "4.4.7 模型运行\n"
    nn4.learn(epochs=100, show_epoch_results=True, random_testing=False)
    Content += "模型测试集mse可以由mse = nn4.test()得到\n"
    mse = nn4.test()
    print("test set MSE = ", np.round(mse, 6))
    
    # 4.4.8 获取模型预测结果的更多细节信息
    Content += "4.4.8 获取模型预测结果的更多细节信息\n"
    Content += "nn4.test_targets_activations给出测试部分实际值和预测值的pairs\n"
    # 4.4.8.1 参考 (http://pyneurgen.sourceforge.net/tutorial_nn.html)
    Content += "4.4.8.1 参考 (http://pyneurgen.sourceforge.net/tutorial_nn.html)\n"
    
    # 测试实际值
    test_reals = data['COE$'][learn_end_point + 1:length].tolist()
    print("test_reals are (时间长度为%d):" % len(test_reals))
    print(test_reals)
    
    # 模型反归一化变换得到真实际值
    retrieved_reals = [np.exp(
        scaler_y.inverse_transform(
            np.array(item).reshape(-1, 1)
        ))[0][0] for item in nn4.test_targets_activations]
    print("retrieved_reals are(时间长度为%d):" % len(retrieved_reals))
    print(retrieved_reals)
    
    # 模型对价格的预测值
    forecast = [np.exp(
        scaler_y.inverse_transform(
            np.array(item).reshape(-1, 1)
        ))[1][0] for item in nn4.test_targets_activations]
    print("forecasts are(时间长度为%d):" % len(forecast))
    print(forecast)
    
    Content += "作图:显示部分历史时间数据\n"
    real = np.array(data['COE$']).reshape(-1)
    history_time_length = 50
    ahead = 12
    plt.plot(range(0, ahead), forecast, '-r', label=u"预测", linewidth=1)
    plt.plot(range(0, ahead), test_reals[0:ahead], color='black', label=u"实际", linewidth=1)
    plt.plot(range(0, ahead), np.array(test_reals[0:ahead]) - 1500, '--k',
             label=u"实际价格 - $1500", linewidth=1)
    plt.plot(range(0, ahead), np.array(test_reals[0:ahead]) + 1500, '--k',
             label=u"实际价格 + $1500", linewidth=1)
    plt.plot(range(-history_time_length, 0),
             real[len(real) - ahead - history_time_length - 1: len(real) - ahead - 1],
             '-b', label=u"历史价格", linewidth=1)
    plt.xlabel(u"预测时间为正, 历史时间为负")
    plt.ylabel(u"价格, 新加坡币")
    plt.legend()
    fig = plt.gcf()
    plt.show()
    fig.savefig('./pictures/nn4_forecast.png')
    
    plt.plot(range(1, len(nn4.accum_mse) + 1, 1), nn4.accum_mse)
    plt.xlabel('epochs')
    plt.ylabel('mean squared error')
    plt.grid(True)
    plt.title("Mean Squared Error by Epoch")
    fig = plt.gcf()
    plt.show()
    fig.savefig('./pictures/nn4_mse_by_epoch.png')
    # print(np.round(nn4.test_targets_activations, 4))
    # 参考: http://pydoc.net/pyneurgen/0.3.1/pyneurgen.neuralnet/
    Content += "参考: http://pydoc.net/pyneurgen/0.3.1/pyneurgen.neuralnet/\n"

Chapter5NeedToRun = [False, True][0]
if Chapter5NeedToRun:
    # 5.
    # 第5章 循环神经网络入门(基于Keras)
    Content += "5.\n第5章 循环神经网络入门\n"
    # 5.1 循环神经网络(RNN)的示意图
    Content += "5.1 循环神经网络(RNN)的示意图(演示200秒)\n"
    import matplotlib.image as mpimg  # 用于加载图片
    rnn_image = mpimg.imread("./pictures/rnn_image.png")
    plt.imshow(rnn_image)
    plt.axis('off')
    plt.show()
    import signal
    class InputTimeoutError(Exception):
        pass
    def interrupted(signum, frame):
        raise InputTimeoutError
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(200)  # 设置自动停留时间
    try:
        your_input = raw_input('请看rnn示意图, 200秒后自动跳过, 按任意键手动跳过:\n'
                               '可以访问\n'
                               'http://www.wildml.com/2015/09/'
                               'recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/\n'
                               '以获得更多信息.\n\n')
    except InputTimeoutError:
        print("\n观看结束.")
        your_input = ''
    signal.alarm(0)  # 读到键盘输入的话重置信号
    print("\n您输入了: %s, 现在跳过rnn示意图学习" % your_input)
    time.sleep(3)
    
    # 5.2 Keras, TensorFlow, Theano 的关系
    console_print = "5.2 Keras, TensorFlow, Theano 的关系\n"
    print(console_print)
    Content += "5.2 Keras, TensorFlow, Theano 的关系\n"
    # 5.3 运行一个 RNN 模型
    console_print = "5.3 运行一个 RNN 模型\n"
    print(console_print)
    Content += console_print
    # 5.3.1 如前面演示学习中所述, RNN 网络中的第 l+1 层(显式地)记忆了第 l 层 的隐藏层信息, 同时第 l 层的输出又是根据第 l 层的隐藏层通过
    #           权重学习, 然后被非线性激活函数处理后得到的结果, 因此我们可以理解为: 后一层的计算使用了前一层的输出, 如此循环下去, 最终构成了
    #           循环神经网络.
    console_print = "5.3.1 如前面演示学习中所述, RNN 网络中的第 l+1 层(显式地)记忆了第 l 层 的隐藏层信息, " \
                    "同时第 l 层的输出又是根据第 l 层的隐藏层通过\n权重学习, 然后被非线性激活函数处理后得到的结果, " \
                    "因此我们可以理解为: 后一层的计算使用了前一层的输出, 如此循环下去, 最终构成了循环神经网络\n"
    print(console_print)
    Content += console_print
    
    # 5.3.2 获取 coe 数据 含有列: DATE,COE$,COE$_1,#Bids,Quota,Open?
    console_print = "5.3.2 获取 coe 数据 含有列: DATE,COE$,COE$_1,#Bids,Quota,Open?\n"
    print(console_print)
    Content += console_print
    loc = './data/coe.csv'
    temp = pd.read_csv(loc)
    # 丢掉日期列
    data = temp.drop(temp.columns[[0]], axis=1)
    # y 是价格时间序列
    y = data['COE$']
    # x是输入特征(一共4个), 先(从data中)去掉目标价格时间序列和binary列(后面再加上), 对前一次的价格 COE$_1 和Bids, Quota 进行log变换.
    x = data.drop(data.columns[[0, 4]], axis=1)
    # log 变换
    x = x.apply(np.log)
    # 添加 Open? 列
    x = pd.concat([x, data["Open?"]], axis=1)
    # 对输入和输出进行归一化((0,1)), 使用 scaler_x.inverse_transform() 和 scaler_y.inverse_transform() 可以对数据还原.
    from sklearn import preprocessing
    scaler_x = preprocessing.MinMaxScaler(
        feature_range=(0, 1)
    )
    x = np.array(x).reshape(len(x), 4)
    x = scaler_x.fit_transform(x)
    scaler_y = preprocessing.MinMaxScaler(
        feature_range=(0, 1)
    )
    y = np.array(y).reshape(len(y), 1)
    y = np.log(y)
    y = scaler_y.fit_transform(y)
    # 5.3.3 生成训练数据与测试数据
    console_print = "5.3.3 生成训练数据(95%)与测试数据(5%)\n"
    print(console_print)
    Content += console_print
    end = len(x) - 1
    learn_end = int(end * 0.954)  # =int(251.856) -> 251, len(x) = 265
    """
    x_train = x[0:learn_end - 1,]
    x_test = x[learn_end:end - 1,]
    y_train = y[1:learn_end]
    y_test = y[learn_end + 1:end]
    """
    x_train = x[0:learn_end, ]
    x_test = x[learn_end:end - 1, ]
    y_train = y[0:learn_end]
    y_test = y[learn_end + 1:end]
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    print("x_train 的形状是: ", x_train.shape)
    print("x_test 的形状是: ", x_test.shape)
    
    # 5.3.4 引入 Keras 模块
    console_print = "5.3.4 引入 Keras 模块\n"
    print(console_print)
    Content += console_print
    console_print = """
          from keras.models import Sequential  # 目的: 可以做linear stacking of layers
          from keras.optimizers import SGD  # 目的: 随机梯度下降
          from keras.layers.core import Dense, Activation  # 目的: 引入激活函数以及一个全连接层(output 层)
          from keras.layers.recurrent import SimpleRNN  # 目的: 引入一个将输出喂给输入的全连接循环神经网络\n"""
    print(console_print)
    Content += console_print
    console_print = "     全连接层的含义: 全连接层中的每个神经元与其前一层的所有神经元进行全连接\n"
    print(console_print)
    Content += console_print
    from keras.models import Sequential  # 目的: 可以做linear stacking of layers
    from keras.optimizers import SGD  # 目的: 随机梯度下降
    from keras.layers.core import Dense, Activation  # 目的: 引入激活函数以及一个全连接层(output 层)
    from keras.layers.recurrent import SimpleRNN  # 目的: 引入一个将输出喂给输入的全连接循环神经网络
    
    # 5.3.5 确定模型参数
    console_print = "5.3.5 确定模型参数\n"
    print(console_print)
    Content += console_print
    seed = 2019
    np.random.seed(seed)
    console_print = """
          rnn5 = Sequential() # 目的: 确定后面可以通过 add 方法来增加网络中的层
          rnn5.add(
              SimpleRNN(output_dim=8, activation="tanh", input_shape=(4, 1))
          )
          # 目的: 构造8层循环, 激活函数为tanh函数(取值介于(-1,1)), 输入含4个特征维度, 时间步长为1.
          rnn5.add(
              Dense(output_dim=1, activation="linear")
          )
          # 目的: 增加1个全输出连接层, 激活函数使用线性函数
    """
    print(console_print)
    Content += console_print
    rnn5 = Sequential()
    # 目的: 确定后面可以通过 add 方法来增加网络中的层
    rnn5.add(
        SimpleRNN(output_dim=8, activation="tanh", input_shape=(4, 1))
    )
    # 目的: 构造8层循环, 激活函数为tanh函数(取值介于(-1,1)), 输入含4个特征维度, 时间步长为1.
    rnn5.add(
        Dense(output_dim=1, activation="linear")
    )
    # 目的: 增加1个全输出连接层, 激活函数使用线性函数
    
    # 5.3.6 使用 momentum
    console_print = "5.3.6 使用 momentum\n"
    print(console_print)
    Content += console_print
    console_print = "学习速率与momentum配合使用是一种技巧, 谁用谁知道.\n"
    print(console_print)
    Content += console_print
    console_print = " momentum 示意图(演示40秒)\n"
    print(console_print)
    Content += console_print
    #import matplotlib.image as mpimg  # 用于加载图片
    benifit_of_momentum = mpimg.imread("./pictures/benifit_of_momentum.png")
    plt.imshow(benifit_of_momentum)
    plt.axis('off')
    plt.show()
    #import signal
    """
    class InputTimeoutError(Exception):
        pass
    def interrupted(signum, frame):
        raise InputTimeoutError
    """
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(200)  # 设置自动停留时间
    try:
        your_input = raw_input('momentum 示意图, 200秒后自动跳过, 按任意键手动跳过:\n'
                               '可以访问\n'
                               'https://blog.csdn.net/BVL10101111/article/details/72615621\n'
                               '以获得更多信息.\n\n')
    except InputTimeoutError:
        print("\n观看结束.")
        your_input = ''
    signal.alarm(0)  # 读到键盘输入的话重置信号
    print("\n您输入了: %s, 现在跳过 momentum 示意图学习" % your_input)
    time.sleep(3)
    
    # 5.3.7 选择 momentum(动量), 并在 Netsterov 梯度加速下降方法使用
    console_print = "5.3.7 选择 momentum(动量), 并在 Netsterov 梯度加速下降方法使用\n"
    print(console_print)
    Content += console_print
    # 5.3.7.1 Netsterov 梯度加速下降 示意图
    console_print = "5.3.7.1 Netsterov 梯度加速下降(1阶优化方法, 提高稳定性与速度(Newton 法是几阶?)) 示意图(演示200秒)\n"
    print(console_print)
    Content += console_print
    #import matplotlib.image as mpimg  # 用于加载图片
    netsterov_accelerated_gd = mpimg.imread("./pictures/netsterov_accelerated_gd.png")
    plt.imshow(netsterov_accelerated_gd)
    plt.axis('off')
    plt.show()
    #import signal
    """
    class InputTimeoutError(Exception):
        pass
    def interrupted(signum, frame):
        raise InputTimeoutError
    """
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(200)  # 设置自动停留时间
    try:
        your_input = raw_input('netsterov_accelerated_gd 示意图, 200秒后自动跳过, 按任意键手动跳过:\n'
                               '可以访问\n'
                               'https://blog.csdn.net/tsyccnh/article/details/76673073\n'
                               '以获得更多信息.\n\n')
    except InputTimeoutError:
        print("\n观看结束.")
        your_input = ''
    signal.alarm(0)  # 读到键盘输入的话重置信号
    print("\n您输入了: %s, 现在跳过 netsterov_accelerated_gd 示意图学习" % your_input)
    time.sleep(3)
    # 5.3.7.2 Netsterov 与 momentum 调用方式
    console_print = "5.3.7.2 Netsterov 与 momentum 调用方式\n"
    print(console_print)
    Content += console_print
    console_print = """
          sgd = SGD(lr=0.0001, momentum=0.95,nesterov=True)
          rnn5.compile(loss="mean_squared_error", optimizer=sgd)\n"""
    print(console_print)
    Content += console_print
    sgd = SGD(lr=0.0001, momentum=0.95,nesterov=True)
    rnn5.compile(loss="mean_squared_error", optimizer=sgd)
    # 5.3.8 现在我们还要使用 Mini Batch
    console_print = "5.3.8 现在我们还要使用 Mini Batch\n"
    print(console_print)
    Content += console_print
    console_print = "一个epoch遍历一次数据, 为了完成这个epoch, 我们可以选择全部训练样本数据去做计算更新一步参数(很多)" \
                    ",\n 也可以只用一个(随机梯度默认就会这样子做了), 我们还可以把一个epoch的数据分成很多个mini batch,\n 一个mini batch" \
                    "(如 batch_size=10)可以用来更新一次参数, 然后迭代多次来完成这个epoch.\n"
    print(console_print)
    Content += console_print
    # 5.3.9 模型训练 fit
    # 5.3.9.1 调用rnn5.fit()
    console_print = "5.3.9 模型训练 fit\n5.3.9.1 调用rnn5.fit()\n"
    print(console_print)
    Content += console_print
    #rnn5.fit(x_train, y_train, batch_size=10, nb_epoch=700)
    rnn5.fit(x_train, y_train, batch_size=10, nb_epoch=100)
    # 5.3.9.2 查看训练测试(预测)的误差
    console_print = "5.3.9.2 查看训练测试(预测)的误差\n"
    print(console_print)
    Content += console_print
    score_train = rnn5.evaluate(x_train, y_train, batch_size=10)
    score_test = rnn5.evaluate(x_test, y_test, batch_size=10)
    print("in train MSE = ", round(score_train, 6))
    print("in test MSE = ", round(score_test, 6))
    # 5.3.10 获取预测数据并还原成原尺度
    console_print = "5.3.10 获取预测数据并还原成原尺度\n"
    print(console_print)
    Content += console_print
    forecast5 = rnn5.predict(x_test)
    forecast5 = scaler_y.inverse_transform(np.array(forecast5).reshape((len(forecast5), 1)))
    forecast5 = np.exp(forecast5).reshape(-1)
    print(forecast5)
    
    Content += "作图:显示部分历史时间数据\n"
    real = np.array(data['COE$'])[0:263].reshape(-1)
    # 测试实际值
    test_reals = data['COE$'][251:263].tolist()
    history_time_length = 50
    ahead = 12
    plt.plot(range(0, ahead), forecast5, '-r', label=u"预测", linewidth=1)
    plt.plot(range(0, ahead), test_reals[0:ahead], color='black', label=u"实际", linewidth=1)
    plt.plot(range(0, ahead), np.array(test_reals[0:ahead]) - 1500, '--k',
             label=u"实际价格 - $1500", linewidth=1)
    plt.plot(range(0, ahead), np.array(test_reals[0:ahead]) + 1500, '--k',
             label=u"实际价格 + $1500", linewidth=1)
    plt.plot(range(-history_time_length, 0),
             real[len(real) - ahead - history_time_length - 1: len(real) - ahead - 1],
             '-b', label=u"历史价格", linewidth=1)
    plt.xlabel(u"预测时间为正, 历史时间为负")
    plt.ylabel(u"价格, 新加坡币")
    plt.legend()
    fig = plt.gcf()
    plt.show()
    rnn5_forecast_image_saved = True
    if not rnn5_forecast_image_saved:
        fig.savefig('./pictures/rnn5_forecast.png')

Chapter6NeedToRun = [False, True][0]
if Chapter6NeedToRun:
    # 6.
    # 第6章 循环神经网络进阶: Elman Neural Networks(含延滞层)
    # 6.1 Elman NN广泛应用于控制, 优化, 模式分类
    Content += "6.\n第6章 循环神经网络进阶: Elman Neural Networks(含延滞层)\n" \
               "6.1 Elman NN广泛应用于控制, 优化, 模式分类\n"
    # 为节省时间, 我们仅简单介绍其原理, 主要还是看代码.
    # 6.2 Elman RNN 结构图(隐藏层与Delay层全连接)
    console_print = "6.2 Elman RNN 结构图(隐藏层与Delay层全连接)\n"
    print(console_print)
    Content += console_print
    import matplotlib.image as mpimg  # 用于加载图片
    ernn6_structure = mpimg.imread("./pictures/ernn6_structure.png")
    plt.imshow(ernn6_structure)
    plt.axis('off')
    plt.show()
    time.sleep(2)
    
    # 数据准备
    loc = './data/coe.csv'
    temp = pd.read_csv(loc)
    # 丢掉日期列
    data = temp.drop(temp.columns[[0]], axis=1)
    # y 是价格时间序列
    y = data['COE$']
    # x是输入特征(一共4个), 先(从data中)去掉目标价格时间序列和binary列(后面再加上), 对前一次的价格 COE$_1 和Bids, Quota 进行log变换.
    x = data.drop(data.columns[[0, 4]], axis=1)
    # log 变换
    x = x.apply(np.log)
    # 添加 Open? 列
    x = pd.concat([x, data["Open?"]], axis=1)
    # 对输入和输出进行归一化((0,1)), 使用 scaler_x.inverse_transform() 和 scaler_y.inverse_transform() 可以对数据还原.
    from sklearn import preprocessing
    scaler_x = preprocessing.MinMaxScaler(
        feature_range=(0, 1)
    )
    x = np.array(x).reshape(len(x), 4)
    x = scaler_x.fit_transform(x)
    scaler_y = preprocessing.MinMaxScaler(
        feature_range=(0, 1)
    )
    y = np.array(y).reshape(len(y), 1)
    y = np.log(y)
    y = scaler_y.fit_transform(y)
    y = y.tolist()
    x = x.tolist()
    
    # 调用erman 神经网络
    from pyneurgen.neuralnet import NeuralNet
    from pyneurgen.recurrent import ElmanSimpleRecurrent
    random.seed(2019)
    ernn6 = NeuralNet()
    input_nodes = 4
    hidden_nodes = 7
    output_nodes =1
    ernn6.init_layers(input_nodes, [hidden_nodes], output_nodes, ElmanSimpleRecurrent())
    ernn6.randomize_network()
    ernn6.layers[1].set_activation_type('sigmoid')
    ernn6.set_learnrate(0.05)
    ernn6.set_all_inputs(x)
    ernn6.set_all_targets(y)
    #import matplotlib.image as mpimg  # 用于加载图片
    
    # 6.3 我们优化的损失函数可能有很多的局部最小值, 如下图所示
    console_print = "6.3 我们优化的损失函数可能有很多的局部最小值, 如下图所示\n"
    print(console_print)
    Content += console_print
    typical_error_surface6 = mpimg.imread("./pictures/typical_error_surface6.png")
    plt.imshow(typical_error_surface6)
    plt.axis('off')
    plt.show()
    time.sleep(2)
    
    # 6.4 Fit 模型 与 mse-epoch 作图
    console_print = "Fit 模型 与 mse-epoch 作图\n"
    print(console_print)
    Content += console_print
    length = len(x)
    learn_end_point = int(length * 0.95)
    ernn6.set_learn_range(0, learn_end_point)
    ernn6.set_test_range(learn_end_point + 1, length - 1)
    ernn6.learn(epochs=100, show_epoch_results=True, random_testing=False)
    plt.plot(range(1, len(ernn6.accum_mse) + 1, 1), ernn6.accum_mse)
    plt.xlabel('epochs')
    plt.ylabel('mean squared error')
    plt.grid(True)
    plt.title("Mean Squared Error by Epoch")
    fig = plt.gcf()
    plt.show()
    fig.savefig('./pictures/ernn6_mse_by_epoch.png')
    
    # 6.5 预测与真实值作图
    console_print = "6.5 预测与真实值作图\n"
    print(console_print)
    Content += console_print
    mse = ernn6.test()
    print("测试集的mse是: ", np.round(mse, 6))
    
    # 测试实际值
    test_reals = data['COE$'][learn_end_point + 1:length].tolist()
    print("test_reals are (时间长度为%d):" % len(test_reals))
    print(test_reals)
    
    # 模型反归一化变换得到真实际值
    retrieved_reals = [np.exp(
        scaler_y.inverse_transform(
            np.array(item).reshape(-1, 1)
        ))[0][0] for item in ernn6.test_targets_activations]
    print("retrieved_reals are(时间长度为%d):" % len(retrieved_reals))
    print(retrieved_reals)
    
    # 模型对价格的预测值
    forecast = [np.exp(
        scaler_y.inverse_transform(
            np.array(item).reshape(-1, 1)
        ))[1][0] for item in ernn6.test_targets_activations]
    print("forecasts are(时间长度为%d):" % len(forecast))
    print(forecast)
    
    Content += "作图:显示部分历史时间数据\n"
    real = np.array(data['COE$']).reshape(-1)
    history_time_length = 50
    ahead = 12
    plt.plot(range(0, ahead), forecast, '-r', label=u"预测", linewidth=1)
    plt.plot(range(0, ahead), test_reals[0:ahead], color='black', label=u"实际", linewidth=1)
    plt.plot(range(0, ahead), np.array(test_reals[0:ahead]) - 1500, '--k',
             label=u"实际价格 - $1500", linewidth=1)
    plt.plot(range(0, ahead), np.array(test_reals[0:ahead]) + 1500, '--k',
             label=u"实际价格 + $1500", linewidth=1)
    plt.plot(range(-history_time_length, 0),
             real[len(real) - ahead - history_time_length - 1: len(real) - ahead - 1],
             '-b', label=u"历史价格", linewidth=1)
    plt.xlabel(u"预测时间为正, 历史时间为负")
    plt.ylabel(u"价格, 新加坡币")
    plt.legend()
    fig = plt.gcf()
    plt.show()
    fig.savefig('./pictures/ernn6_forecast.png')
    # 第6章完成.

Chapter9NeedToRun = [False, True][1]
if Chapter9NeedToRun:
    # 第9章  LSTM
    # 广泛应用于语音,手写识别, 时序预测(周期性趋势的长短记忆信息学习)等
    # 我们应用lstm模型学习太阳黑子活动规律, 舒适区间为 (+/-)50
    import urllib
    #url = "https://goo.gl/uWbihf"
    #data = pd.read_csv(url, sep=";")
    loc = "./data/monthly_sunspots.csv"
    #data.to_csv(loc, index=False)
    data_csv = pd.read_csv(loc, header=None)
    yt = data_csv.iloc[0:3210, 3]
    print(yt.head())
    print(yt.tail())
    # 9.2 考察偏自相关系数(Autocorrelation)
    #Content += "9.2 考察偏自相关系数\n"
    from statsmodels.tsa.stattools import pacf
    yt_pacf = pacf(yt, nlags=30, method='ols')
    print(yt_pacf)
    plt.plot(range(0, 31), yt_pacf)
    plt.xlabel(u'月')
    plt.ylabel(u'偏自相关系数')
    plt.grid(True)
    plt.title(u"277年以来的太阳黑子活动相关性分析")
    fig = plt.gcf()
    plt.show()
    sunspot_pacf_image_saved = True
    if not sunspot_pacf_image_saved:
        fig.savefig('./pictures/sunspot_pacf.png')
    yt_1 = yt.shift(1)
    yt_2 = yt.shift(2)
    yt_3 = yt.shift(3)
    yt_4 = yt.shift(4)
    yt_5 = yt.shift(5)
    data = pd.concat([yt, yt_1, yt_2, yt_3, yt_4, yt_5], axis=1)
    data.columns = ["yt", "yt_1", "yt_2", "yt_3", "yt_4","yt_5"]
    # yt_n 取前n个月的数据, 于是我们获取到的数据含有nan值(5行)
    console_print = "yt_n 取前n个月的数据, 于是我们获取到的数据含有nan值(5行)\n"
    print(console_print)
    print(data.head(6))
    data = data.dropna()
    y = data['yt']
    cols = ["yt_1", "yt_2", "yt_3", "yt_4","yt_5"]
    x = data[cols]
    
    # 下图演示了一个简单的 lstm 记忆块的结构: 含有一个输入门, 一个输出门, 一个遗忘门.
    import matplotlib.image as mpimg  # 用于加载图片
    simple_memory_block_lstm9 = mpimg.imread("./pictures/simple_memory_block_lstm9.png")
    plt.imshow(simple_memory_block_lstm9)
    plt.axis('off')
    plt.show()
    if not Chapter5NeedToRun:
        import signal
        class InputTimeoutError(Exception):
            pass
        def interrupted(signum, frame):
            raise InputTimeoutError
    presentation = False
    if presentation:
        signal.signal(signal.SIGALRM, interrupted)
        signal.alarm(1)  # 设置自动停留时间
        try:
            your_input = raw_input('请看lstm 记忆块 示意图, 1秒后自动跳过, 按任意键手动跳过:\n'
                                   '可以访问\n'
                                   'https://blog.csdn.net/shijing_0214/article/details/52081301\n'
                                   '以获得更多信息.\n\n')
        except InputTimeoutError:
            print("\n观看结束.")
            your_input = ''
        signal.alarm(0)  # 读到键盘输入的话重置信号
        print("\n您输入了: %s, 现在跳过lstm 记忆块示意图学习" % your_input)
        time.sleep(2)
        
        # 1 个记忆细胞, 3 个乘积门
        # inner hard sigmoid function 可分段线性
        
        cec_1 = mpimg.imread("./pictures/cec_1.png")
        plt.imshow(cec_1); plt.axis('off')
        plt.show()
        signal.signal(signal.SIGALRM, interrupted)
        signal.alarm(1)  # 设置自动停留时间
        try:
            your_input = raw_input('请看CEC 示意图1, 1秒后自动跳过, 按任意键手动跳过:\n'
                                   '可以访问\n'
                                   'https://deepai.org/machine-learning-glossary-and-terms/constant%20error%20carousel\n'
                                   '以获得更多信息.\n\n')
        except InputTimeoutError:
            print("\n观看结束.")
            your_input = ''
        signal.alarm(0)  # 读到键盘输入的话重置信号
        print("\n您输入了: %s, 现在跳过CEC 示意图1 学习" % your_input)
        time.sleep(2)
        
        
        cec_2 = mpimg.imread("./pictures/cec_2.png")
        plt.imshow(cec_2)
        plt.axis('off')
        plt.show()
        signal.signal(signal.SIGALRM, interrupted)
        signal.alarm(1)  # 设置自动停留时间
        try:
            your_input = raw_input('请看CEC 示意图2, 1秒后自动跳过, 按任意键手动跳过:\n'
                                   '可以访问\n'
                                   'https://www.quantinfo.com/Article/View/695.html\n'
                                   '以获得更多信息.\n\n')
        except InputTimeoutError:
            print("\n观看结束.")
            your_input = ''
        signal.alarm(0)  # 读到键盘输入的话重置信号
        print("\n您输入了: %s, 现在跳过CEC 示意图2 学习" % your_input)
        time.sleep(2)
        
        cec_3 = mpimg.imread("./pictures/cec_3.png")
        plt.imshow(cec_3)
        plt.axis('off')
        plt.show()
        signal.signal(signal.SIGALRM, interrupted)
        signal.alarm(1)  # 设置自动停留时间
        try:
            your_input = raw_input('请看CEC 示意图3, 1秒后自动跳过, 按任意键手动跳过:\n'
                                   '可以访问\n'
                                   'https://www.quantinfo.com/Article/View/695.html\n'
                                   '以获得更多信息.\n\n')
        except InputTimeoutError:
            print("\n观看结束.")
            your_input = ''
        signal.alarm(0)  # 读到键盘输入的话重置信号
        print("\n您输入了: %s, 现在跳过CEC 示意图3 学习" % your_input)
        time.sleep(2)
        
        hard_sigmoid_image = mpimg.imread("./pictures/hard_sigmoid.png")
        plt.imshow(hard_sigmoid_image)
        plt.axis('off')
        plt.show()
        signal.signal(signal.SIGALRM, interrupted)
        signal.alarm(1)  # 设置自动停留时间
        try:
            your_input = raw_input('请看hard_sigmoid 示意图, 1秒后自动跳过, 按任意键手动跳过:\n'
                                   '可以访问\n'
                                   'https://www.quantinfo.com/Article/View/695.html\n'
                                   '以获得更多信息.\n\n')
        except InputTimeoutError:
            print("\n观看结束.")
            your_input = ''
        signal.alarm(0)  # 读到键盘输入的话重置信号
        print("\n您输入了: %s, 现在跳过hard_sigmoid 示意图 学习" % your_input)
        time.sleep(2)
    
    from sklearn import preprocessing
    scaler_x = preprocessing.MinMaxScaler(
        feature_range=(-1, 1))
    x = np.array(x).reshape(len(x), 5)
    x = scaler_x.fit_transform(x)
    
    scaler_y = preprocessing.MinMaxScaler(
        feature_range=(-1, 1))
    y = np.array(y).reshape(len(y), 1)
    #y = np.log(y) 为什么不取对数呢
    y = scaler_y.fit_transform(y)
    train_end = 3042
    x_train = x[0:train_end, ]
    x_test = x[train_end + 1: 3205, ]
    y_train = y[0:train_end]
    y_test = y[train_end + 1:3205]
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    print("shape of x_train is: ")
    print(x_train.shape)
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation
    from keras.layers.recurrent import LSTM
    seed = 2019
    np.random.seed(seed)
    # 基础版 lstm(可以选择shuffle 为 True和False)
    basic_lstm = False
    if basic_lstm:
        lstm9 = Sequential()
        lstm9.add(LSTM(output_dim=4,
                       activation='tanh',
                       inner_activation='hard_sigmoid',
                       input_shape=(5,1)
                       )
                  )
        lstm9.add(Dense(output_dim=1, activation='linear'))
        # 使用 rmsprop 自动调整学习速率
        lstm9.compile(loss="mean_squared_error", optimizer="rmsprop")
        # 时序不shuffle
        lstm9.fit(x_train, y_train, batch_size=1, nb_epoch=5, shuffle=False)
        
        print(lstm9.summary())
        score_train = lstm9.evaluate(x_train, y_train, batch_size=1)
        score_test=lstm9.evaluate(x_test, y_test, batch_size=1)
        print("训练误差是:", round(score_train, 4))
        print("测试误差是:", round(score_test, 4))
        forecast9 = lstm9.predict(x_test)
        forecast9 = scaler_y.inverse_transform(np.array(forecast9).reshape(len(forecast9), 1))
        forecast9 = np.array(forecast9).reshape(-1)
        print("预测值是: ")
        print(forecast9)
        print("作图:显示部分历史时间数据\n")
        real = np.array(data['yt'][0:3042]).reshape(-1)
        test_reals = data['yt'][3043:3205].tolist()
        history_time_length = 400
        ahead = 162
        plt.plot(range(0, ahead), forecast9, '-r', label=u"预测", linewidth=1)
        plt.plot(range(0, ahead), test_reals[0:ahead], color='black', label=u"实际", linewidth=1)
        plt.plot(range(0, ahead), np.array(test_reals[0:ahead]) - 50, '--k',
                 label=u"实际活动 - 50", linewidth=1)
        plt.plot(range(0, ahead), np.array(test_reals[0:ahead]) + 50, '--k',
                 label=u"实际活动 + 50", linewidth=1)
        plt.plot(range(-history_time_length, 0),
                 real[len(real) - ahead - history_time_length - 1: len(real) - ahead - 1],
                 '-b', label=u"历史活动", linewidth=1)
        plt.xlabel(u"预测时间为正, 历史时间为负")
        plt.ylabel(u"活动值")
        plt.legend()
        fig = plt.gcf()
        plt.show()
        basic_lstm9_forecast_image_saved = True
        if not basic_lstm9_forecast_image_saved:
            fig.savefig('./pictures/lstm9_forecast_basic.png')
    # 高级一点的lstm (引入 statefulness, 也可以配置shuffle 值为True 和False)
    advanced_lstm = True
    if advanced_lstm:
        lstm9 = Sequential()
        # 设置stateful 为 True
        lstm9.add(LSTM(output_dim=4,
                       stateful=True,
                       activation='tanh',
                       inner_activation='hard_sigmoid',
                       batch_input_shape=(2, 5, 1)
                       )
                  )
        lstm9.add(Dense(output_dim=1, activation='linear'))
        # 使用 rmsprop 自动调整学习速率
        lstm9.compile(loss="mean_squared_error", optimizer="rmsprop")
        # shuffle
        end_point = len(x_train)
        start_point = end_point - 500
        for i in range(0, 10):
            print("Fitting example %s: " % (i+1))
            # verbose = 0，在控制台没有任何输出
            # verbose = 1 ：显示进度条
            # verbose =2：为每个epoch输出一行记录
            lstm9.fit(x_train[start_point: end_point],
                      y_train[start_point: end_point],
                      batch_size=2, nb_epoch=1, verbose=1, shuffle=True)
            lstm9.reset_states()
        # print(lstm9.summary())
        score_train = lstm9.evaluate(x_train[start_point: end_point],
                                     y_train[start_point: end_point],
                                     batch_size=2)
        score_test=lstm9.evaluate(x_test, y_test, batch_size=2)
        print("训练误差是:", round(score_train, 4))
        print("测试误差是:", round(score_test, 4))
        forecast9 = lstm9.predict(x_test, batch_size=2)
        forecast9 = scaler_y.inverse_transform(np.array(forecast9).reshape(len(forecast9), 1))
        forecast9 = np.array(forecast9).reshape(-1)
        print("预测值是: ")
        print(forecast9)
        print("作图:显示部分历史时间数据\n")
        real = np.array(data['yt'][0:3042]).reshape(-1)
        test_reals = data['yt'][3043:3205].tolist()
        history_time_length = 400
        ahead = 162
        plt.plot(range(0, ahead), forecast9, '-r', label=u"预测", linewidth=1)
        plt.plot(range(0, ahead), test_reals[0:ahead], color='black', label=u"实际", linewidth=1)
        plt.plot(range(0, ahead), np.array(test_reals[0:ahead]) - 50, '--k',
                 label=u"实际活动 - 50", linewidth=1)
        plt.plot(range(0, ahead), np.array(test_reals[0:ahead]) + 50, '--k',
                 label=u"实际活动 + 50", linewidth=1)
        plt.plot(range(-history_time_length, 0),
                 real[len(real) - ahead - history_time_length - 1: len(real) - ahead - 1],
                 '-b', label=u"历史活动", linewidth=1)
        plt.xlabel(u"预测时间为正, 历史时间为负")
        plt.ylabel(u"活动值")
        plt.legend()
        fig = plt.gcf()
        plt.show()
        advanced_lstm9_forecast_image_saved = True
        if not advanced_lstm9_forecast_image_saved:
            fig.savefig('./pictures/lstm9_forecast_advanced.png')
    # 第 9 章 lstm 完.

ContentFileNeedToUpdate = False
if ContentFileNeedToUpdate:
    Content_txt_file = open("./Context.txt", "w")
    Content_txt_file.write("目录:\n%s" % Content)
    Content_txt_file.close()
print(Content)
elapsed = time.time() - start
print("程序一共执行了 ", elapsed, "秒.")

