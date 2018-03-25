# 数据处理
from urllib.request import urlretrieve

import pandas as pd


def load_data(download=True):
    '''
    下载数据
    :param download:
    :return:
    '''
    if download:
        data_path,_=urlretrieve('http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data','car.csv')
        print('Downloaded to car.csv')

    col_names = ['buying','maint','doors','persons','lug_boot','safety','class']
    data = pd.read_csv('car.csv',names=col_names)
    return data

def convert2onehot(data):
    '''
    处理数据
    :param data:
    :return:
    '''
    return pd.get_dummies(data,prefix=data.columns)


# if __name__ == '__main__':
#     data = load_data(download=True)
#     new_data = convert2onehot(data)
#
#     print(data.head())
#
#
#     for name in data.keys():
#         print(name,pd.unique(data[name]))
#
#     print(new_data.head())
#
#     for name in new_data.keys():
#         print(name,pd.unique(new_data[name]))