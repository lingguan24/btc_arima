# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats
import statsmodels.api as sm
import warnings
from itertools import product
from datetime import datetime

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import arma_order_select_ic
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')
plt.style.use('seaborn-poster')
# raw_data = pd.read_csv('D:/btc_arima/data/BTCUSDT-1d-2024-01/BTCUSDT-1d-2024-01.csv')


import glob

# 获取所有匹配的CSV文件路径
csv_files = glob.glob(r'.\data\spot\BTCUSDT\klines\1d\*.csv')

# # 读取每个CSV文件并将它们存储在一个列表中，并强制转换为相同的数据类型
dfs = [pd.read_csv(file, header=None, dtype=str) for file in csv_files]

# 如果需要，可以尝试将所有列转换为浮点数类型（根据具体需求）
dfs = [df.astype(float) for df in dfs]

# 将所有DataFrame合并为一个
combined_df = pd.concat(dfs, ignore_index=True)

raw_data = combined_df
raw_data.columns = [
'open_time',
'open',
'high',	
'low',
'close',
'volume',	
'close_time',
'quote_volume',	
'count',
'taker_buy_volume',
'taker_buy_quote_volume',
'ignore'
]
# print(data.head())

# indexing the data
raw_data['open_time'] = pd.to_datetime(raw_data['open_time'], unit='ms')
raw_data.index = raw_data['open_time']
raw_data.drop('open_time', axis=1, inplace=True)
raw_data['average_price'] = raw_data['quote_volume'] / raw_data['volume'] #计算均价

data = raw_data[['average_price']]
# #计算每日较前日回报率
# data = (raw_data[['average_price']].diff(periods=1) / raw_data[['average_price']].shift(1)).dropna()

print(data)

rolling_width = 50
def rolling_predict(data):
    print('rolling_predict() begins...')
    data_diff = data
    d = 0
    if adfuller(data)[1] < 0.05:
        pass
    else:
        for i in range(1, rolling_width):
            data_diff = data.diff(periods=i)[i:]
            # print(f'{data.diff=}\n')
            if adfuller(data_diff)[1] < 0.05:
            # print(f'{adfuller(data_diff)=}')
                d = i
                break
    # print(f'{d=}')

    data_to_proccess = data_diff[['average_price']]
    # print(f'data_to_proccess=\n{data_to_proccess}')
    # print(f'{d=}')
    # PLOTS
    # fig = plt.figure(figsize=[15, 7])
    # plt.plot(data.open[0:1000], '-')
    # plt.legend()

    # print(f'{adfuller(data_train_diff1.close)=}')
    # print(f'''{arma_order_select_ic(data_train_diff1.close, max_ar=5, max_ma=5, ic='aic')=}''')
    
    order = arma_order_select_ic(data_to_proccess, max_ar=10, max_ma=10, ic='bic')['bic_min_order']
    print('arma_order_select_ic() ends.')
    # print(f'{order=}')
    # from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    # plot_acf(data_train_diff1.close)
    # plot_pacf(data_train_diff1.close)
    try:

        print('ARIMA fitting begins...')
        model = ARIMA(data, order=(order[0], d, order[1])).fit()
    except Exception as e:
        print(f'ARIMA fitting failed: {e}')
        return pd.Series(dtype='float64')
    print(f'Model fits successfully,\n{model.summary()}')
    
    result = model.forecast()
    # print(result)
    print('rolling_predict() ends...\n\
----------------------------------------------------------------')
    return result
    # data[1:200].open.plot(color='g', label='true', figsize=(12,4))
    # preds.plot(color='r', label='predicts')
    # plt.legend()

    # plt.show()


# print('data=');print(data)

predictions = pd.DataFrame(index=raw_data.index, columns=['average_price'])
for i in range(rolling_width, len(data)):
    # print(f'{data[i - 10 : i]=}')
    print(f'Predicting {data.index[i]}...')
    rel = rolling_predict(data[i - rolling_width : i])
    # 检查 rel 是否为空的同时，将第一个预测值分配给 predictions DataFrame
    if not rel.empty:
        predictions.iat[i, 0] = rel.iloc[0]
    else:
        predictions.iat[i, 0] = None
    print(f'{data.index[i]} predicted.\n\
================================================================')
    
print(f'{predictions=}')


plt.rcParams['font.sans-serif'] = [u'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
 
plt.plot(data,color="r",label="real") #颜色表示
 
plt.plot(predictions,color=(0,0,0),label="predictions") 
 
plt.xlabel("时间") #x轴命名表示
 
plt.ylabel("价格") #y轴命名表示
 
plt.title("实际值与预测值折线图") 
 
plt.legend()#增加图例
 
plt.show() #显示图片