import math
from statistics import mean
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



class Trader(object):
    def __init__(self, starting_capital) -> None:
        
        self.fitness = 0.0
        self.invest_weights = []
        self.starting_capital = starting_capital
        self.current_capital = 0.0
        self.position = ""
        self.past_pos = ['Long']
        self.dailyProfit = 0
        self.posDailyProfit = 0
        self.meanPosDailyProfit = 0
        self.transactions = []
        self.transaction_cost = 0
        self.rollup = {}
        self.capital_history = []

    
    def begining_trans(self, tickers, open_prices:list):
        allocate_cap = float((self.starting_capital)/len(tickers))
        self.bought = True
        self.sold = False
        op = []
        dt = []
        
        for i, dataset in enumerate(open_prices):
            #dataset.reset_index(inplace=True)
            op.append(dataset['Open'].iloc[[0]])
            dt.append(dataset['Date'].iloc[[0]])

        for j, tick in enumerate(tickers):
            date = str(dt[j]).strip('nName: Date, dtype: datetime64[ns]')
            date = date.strip('\n')
            shares = float(self.starting_capital/op[j])
            self.rollup[tick] = [tick, allocate_cap, shares, self.bought]
            self.capital_history = [allocate_cap]
            self.transactions.append([f"{str(date)}| bought {str(float(shares))} of {tick} | cost: {str(float(allocate_cap))}"])
            
    def buy(self, tickers:list, current_price, date:str, shares):
        for tick in tickers:
            if float(shares) == 0.0:
                shares = float(self.rollup[tick][1]/current_price)
                bought = True
                allocate_cap = shares*float(current_price)
                self.capital_history.append(allocate_cap)
                self.rollup[tick] = [tick, allocate_cap, shares, bought]
                self.transactions.append([f"{str(date)}| bought {str(round(shares,2))} of {str(tick)} | cost: {str(allocate_cap)}"])
            else:
                bought = True
                allocate_cap = shares*float(current_price)
                self.capital_history.append(allocate_cap)
                self.rollup[tick] = [tick, allocate_cap, shares, bought]
                self.transactions.append([f"{str(date)}| holding {str(round(float(self.rollup[tick][2]),2))} \
                                            of {str(tick)} | capital: {str(float(allocate_cap))}"])


    def sell(self, tickers:list, current_price, date:str, shares):
        for tick in tickers:
            shares = float(self.rollup[tick][2])
            if float(shares) != 0.0:
                bought = False
                allocate_cap = shares*float(current_price)
                self.capital_history.append(allocate_cap)
                shares = 0
                self.rollup[tick] = [tick, allocate_cap, shares, bought]
                self.transactions.append([f"{date}| sold {self.rollup[tick][2]} of {tick} | capital {self.rollup[tick][2]}"])
            if float(shares) == 0.0:
                bought = False
                self.rollup[tick][3] = bought
                self.capital_history.append(self.rollup[tick][1])
                self.transactions.append([f"{date}| currently own {self.rollup[tick][3]} \
                                            of {tick} | capital {self.rollup[tick][1]}"])




    def calc_fitness(self, tickers, current_price, original_price, index):
        self.fitness = 0
        for tick in tickers:
            fitness = 0
            self.dailyProfit = ((float(current_price) - float(original_price) - float(self.transaction_cost))/float(original_price))
        
            if self.dailyProfit > 0 and self.past_pos[index-1] == 'Long':
                self.posDailyProfit = self.dailyProfit * 2
                #self.posDailyProfit +=1
            

            if self.dailyProfit <= 0 and self.past_pos[index-1] == 'Long':
                self.posDailyProfit = self.dailyProfit
                #self.posDailyProfit -=1 

            if self.past_pos[index-1] == 'Neutral':
                self.posDailyProfit = 0

            # average = np.mean([original_price, current_price])
            # self.meanPosDailyProfit = average*self.posDailyProfit

            try:              

                #ror = (final_capital - trader.starting_capital)/trader.starting_capital
                if float(self.rollup[tick][2]) != 0.0:
                    ror = (((self.rollup[tick][1]) - (self.rollup[tick][2]*original_price))/(self.rollup[tick][2]*original_price))-1
                else:
                    ror = 0
                
            except IndexError:
                #print(self.rollup[tick][1])
                print(index)
                print(self.capital_history)
                print(self.past_pos)
                print(self.rollup)
                sys.exit()
            fitness = (self.meanPosDailyProfit+0.1)+(self.meanPosDailyProfit+0.1)*math.sqrt(abs(ror))
            #self.fitness += fitness
            self.fitness += self.posDailyProfit

        return self.fitness
