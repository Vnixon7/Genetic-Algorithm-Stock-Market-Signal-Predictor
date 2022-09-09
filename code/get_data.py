import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import yfinance as yf
import sys
import pandas_ta as ta
import talib
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import neat
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)




class Get_data(object):
    def __init__(self, companies, start_date, end_date, eval_year) -> None:
        self.companies = companies
        self.start_date = start_date
        self.end_date = end_date
        self.eval_year = eval_year
        self.preprocess = []
        self.train = []
        self.test = []
        self.download()

    def __dl_data(self):
        for company in self.companies:
            df = yf.download(company, start=self.start_date, 
                        end=self.end_date, group_by="ticker")

            self.preprocess.append(df)

            df = yf.download(company, start=f'{str(self.eval_year)}-01-01', 
                    end=f'{str(int(self.eval_year)+1)}-01-01', group_by="ticker")

            self.test.append(df)

    def __metrics(self):
        for df in self.preprocess:
            # df = yf.download(company, start=self.start_date, 
            #         end=self.end_date, group_by="ticker")
            #df.reset_index(inplace=True)
            #self.initial_open.append(df['Open'][0])
            df.ta.sma(close='Open', length=20, append=True,)
            df.ta.sma(close='Open', length=50, append=True)
            df.ta.sma(close='Open', length=100, append=True)
            df.ta.macd(close='Open', fast=12, slow=26, signal=9, append=True)
            df.ta.ema(close='Low', length=20, append=True)
            df.ta.ema(close='Low', length=50, append=True)
            df.ta.ema(close='Low', length=100, append=True)
            df.ta.ppo(close='Adj Close', fast=12, slow=26, signal=9, scalar=None, 
                mamode=None, talib=None, offset=None, append=True)
            df['Psar'] = talib.SAR(df['High'], df['Low'], acceleration=0.02, maximum=0.2)
            df['Adx'] = talib.ADX(df['High'], df['Low'], df['Adj Close'], timeperiod=14)
            df['Cci'] = talib.CCI(df['High'], df['Low'], df['Adj Close'], timeperiod=14)
            df['Atr'] = talib.ATR(df['High'], df['Low'], df['Adj Close'], timeperiod=14)
            df['bb_upperband'], df['bb_middleband'], df['bb_lowerband'] = talib.BBANDS(df['Adj Close'], 
                                                                            timeperiod=5, nbdevup=2, 
                                                                            nbdevdn=2, matype=0)
            df['Roc'] = talib.ROC(df['Adj Close'], timeperiod=10)
            df['Rsi'] = talib.RSI(df['Adj Close'], timeperiod=14)
            df['stochk'], df['stochd'] = talib.STOCH(df['High'], df['Low'], df['Adj Close'], fastk_period=14, 
                slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)  
            df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Adj Close'], timeperiod=14)  
            df['Mfi'] = talib.MFI(df['High'],df['Low'],df['Adj Close'],df['Volume'],timeperiod=14) 
            df['Obv'] = talib.OBV(df['Adj Close'], df['Volume'])
            df['Adosc'] = talib.ADOSC(df['High'], df['Low'], df['Adj Close'], df['Volume'], fastperiod=3, slowperiod=10)
            
            #df['Date'] = [str(i) for i in df['Date']]
            scaler = MinMaxScaler()
            pca = PCA()
            df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
            df = df.replace(np.nan, 0)
            df = pd.DataFrame(pca.fit_transform(df), columns=df.columns)
            self.train.append(df)

    def download(self):
        self.__dl_data()
        self.__metrics()
