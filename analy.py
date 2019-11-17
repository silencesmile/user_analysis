# -*- coding: utf-8 -*-
# @Time    : 2019/10/31 11:23 AM
# @Author  : Python小学僧
# @File    : analy.py
# @Software: PyCharm

import pandas as pd

from math import ceil
from pandas import DataFrame
from datetime import datetime
from sklearn.cluster import KMeans
from pandas import Series
from sklearn.externals import joblib

filePath = "./RFM聚类分析【样本数据】.xlsx"

def readUserInfo(filePath):
    ''' 数据读取及预处理'''
    # 获取数据
    data = pd.read_excel(filePath, index_col="用户编码")
    # print(data)
    # 数据描述信息
    data_des = data.describe(include="all")
    print(data_des)

    return data

def dataChange(data):
    deadline_time = datetime(2016,7,20)
    print(deadline_time)

    # 时间相减 得到天数查 timedelta64类型
    diff_R = deadline_time - data["最近一次投资时间"]

    # 渠道具体天数
    # days = diff_R[0].days
    R = []
    for i in diff_R:
        days = i.days
        R.append(days)

    print(R)
    '''
    用户在投时长(月
    Python没有直接获取月数差的函数
    1、获取用户在投天数
    2、月=在投天数/30，向上取整
    '''
    diff = deadline_time - data["首次投资时间"]
    print(diff)

    # 利用向上取整函数
    months = []
    for i in diff:
        month = ceil(i.days/30)
        months.append(month)

    print(months)

    # 月均投资次数
    month_ave = data["总计投标总次数"]/months
    F = month_ave.values
    print(F)

    # 月均投资金额
    M = (data["总计投资总金额"]/months).values
    print(M)

    return R, M, F

# 计算新用户的 R, M, F
def user_info_change(user_info):
    # 获取当前时间
    # user_info = ["lily", "2016-06-12", "2016-06-25", 20000, 6]

    # 1 最后一次投资距提数日的时间

    # str_p = '2019-01-30 15:29:08'
    deadline_touzi = datetime.strptime(user_info[2], '%Y-%m-%d').date()
    # print(dateTime_p) # 2019-01-30 15:29:08

    # 今天的时间
    # today = datetime.date.today()

    # 截止时间
    deadline_date = "2016-07-03"
    deadline_time = datetime.strptime(deadline_date, '%Y-%m-%d').date()

    # 时间相减 得到天数查 timedelta64类型
    Rdays = deadline_time - deadline_touzi
    R = Rdays.days
    print(R)

    # 计算在投时长/月数
    dateTime_p = datetime.strptime(user_info[1], '%Y-%m-%d').date()
    diff = deadline_time - dateTime_p

    # 利用向上取整函数
    month = ceil(diff.days / 30)

    # 2 月均投资次数
    F = user_info[4] / month
    print(F)

    # 3 月均投资金额
    M = user_info[3] / month
    print(M)

    return R, int(M), int(F)


def analy_data(data, R, M, F):
    cdata = DataFrame([R, list(F), list(M)]).T
    # 指定cdata的index和colums
    cdata.index = data.index
    cdata.columns = ["R-最近一次投资时间距提数日的天数", "F-月均投资次数", "月均投资金额"]
    print("cdata_info:\n", cdata)

    print("cdata:\n", cdata.describe())

    # K-Means聚类分析

    # 01 数据标准化  均值：cdata.mean()   标准差：cdata.std()
    # 对应位置分别先相减 再相除
    zcdata = (cdata-cdata.mean())/cdata.std()
    print("zcdata:\n", zcdata)

    # n_clusters：分类种数  n_jobs：计算的分配资源  max_iter：最大迭代次数  random_state：随机数种子，种子相同，参数固定
    kModel = KMeans(n_clusters=4, n_jobs=4, max_iter=100, random_state=0)
    kModel.fit(zcdata)
    print(kModel.labels_)

    save_model(kModel)

    # # 模拟一条用户数据
    # user_data_info = DataFrame([[12.5], [18.0], [20000.0]]).T
    # user_data_info.index = ["lily"]
    # user_data_info.columns = cdata.columns
    # print("cdata_info:\n", user_data_info)
    #
    # new_zcdata = (user_data_info-cdata.mean())/cdata.std()
    # print("new_zcdata", new_zcdata)
    # ret = kModel.predict(new_zcdata)
    # print("new_zcdata_ret:", ret)

    # 统计每个类别的频率
    value_counts = Series(kModel.labels_).value_counts()
    print(value_counts)

    # 将类别标签赋回原来的数据
    cdata_rst = pd.concat([cdata, Series(kModel.labels_, index=cdata.index)], axis=1)
    print(cdata_rst)

    # 命名最后一列为类别
    cdata_rst.columns = list(cdata.columns) + ["类别"]
    print(cdata_rst)

    # 按照类别分组统计R, F, M的指标均值
    user_ret = cdata_rst.groupby(cdata_rst["类别"]).mean()
    print(user_ret)

    '''
        R-最近一次投资时间距提数日的天数   F-月均投资次数         月均投资金额
  类别                                             
    0            27.859375               2.820312          21906.754297
    1            20.684211               4.552632          115842.105263
    2            10.568182               5.579545          26984.313636
    3            12.111111               17.277778         107986.000000
    
    结论：
    类别3：R、F、M都比较高，属于重要价值客户 或 超级用户
    类别0：R、F、M都比较低，属于低价值客户
    类别1：R一般、F一般、M很高，也属于重要价值客户
    
    '''

    return kModel, cdata
# 保存模型
def save_model(model):
    joblib.dump(model, "user_classes.pkl")

# 加载模型
def load_model(modelPath):
    model = joblib.load(modelPath)
    return model

def user_classes(cdata, user_info):
    '''
    # 模拟一条用户数据
    1、获取当前时间表示为截止时间
    2.计算出: R F M

    '''
    R, M, F = user_info_change(user_info)
    user_data_info = DataFrame([[R], [F], [M]]).T
    print(user_data_info)

    # user_data_info = DataFrame([[12.5], [18.0], [20000.0]]).T
    user_data_info.index = ["lily"]
    user_data_info.columns = cdata.columns
    print("cdata_info:\n", user_data_info)

    new_zcdata = (user_data_info-cdata.mean())/cdata.std()
    print("new_zcdata", new_zcdata)

    kModel = load_model("user_classes.pkl")
    ret = kModel.predict(new_zcdata)
    print("new_zcdata_ret:", ret)
    # new_zcdata_ret: [3]

    '''
    有一个问题：如果是新注册的用户，即使投资的金额大，也会由于投资次数不足被划分到低端用户列
    '''


if __name__ == '__main__':
    data = readUserInfo(filePath)
    R, M, F = dataChange(data)
    kModel, cdata = analy_data(data, R, M, F)

    # 模型训练时用的是标准化后的数据  现在要预测新数据 怎么处理
    user_info = ["lily", "2016-06-12", "2016-06-25", 60000, 12]
    user_classes(cdata, user_info)