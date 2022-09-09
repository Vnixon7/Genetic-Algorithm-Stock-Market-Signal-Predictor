from __future__ import print_function
import math
from matplotlib import ticker
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
import yfinance as yf
import sys
import pandas_ta as ta
import talib
import pickle
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import neat
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
from get_data import Get_data

import os
import neat
import visualize
from trader import Trader


def eval_genomes(genomes, config):
    stock = 'GME'
    tickers = [stock]
    data = Get_data(tickers,
                    start_date="2017-01-01", 
                    end_date="2020-01-01",
                    eval_year='2021')
    
    # data.download()
    train = data.train
    test = data.test
    nets = []
    traders = []
    gnomes = []
    load_in = open(r'BEST!.pickle', 'rb')
    bestNet = pickle.load(load_in)
    for df in test:
        df.reset_index(inplace=True)
        #df = df.iloc[-2:]
        
    #test[0] = test[0].iloc[-3:]
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        #nets.append(bestNet)
        traders.append(Trader(50000))
        gnomes.append(genome)

    for x, trader in enumerate(traders):
        trader.begining_trans(tickers, test)
        output = nets[x].activate(train[0].T)
        for j in range(1, len(test[0])): #num of lines in dataset
            # print(test[0]['Date'].iloc[[1]])
            # sys.exit()
            #print(output[0])       
            #print(output[j])
            #print(trader.rollup)
            current_price = test[0]['Open'].iloc[[j]]
            original_price = test[0]['Open'].iloc[[j-1]]
            if output[j-1] > 0.5:
                trader.buy(tickers, current_price, test[0]['Date'].iloc[[j]], 
                                    trader.rollup[tickers[0]][2])
                trader.past_pos.append('Long')
                #print('BUY')
                #print(trader.rollup)

            if output[j-1] < 0.5:
                trader.sell(tickers, current_price, test[0]['Date'].iloc[[j]], 
                                    trader.rollup[tickers[0]][2])
                trader.past_pos.append('Neutral')
                #print('SELL')
                #print(trader.rollup)
            
            #print(trader.rollup)
            fitness = trader.calc_fitness(tickers, current_price, original_price, j)
            gnomes[x].fitness += fitness
            #print(gnomes[x].fitness)
                
        print(f"{trader.rollup}  Fitness: {gnomes[x].fitness}")

    best = 0.0
    index = 0
    for x, ge in enumerate(gnomes):
        if ge.fitness > best:
            best = ge.fitness
            index = x
    
    for i in traders[index].transactions:
        with open(r'transactions.txt', 'w') as f:
            f.write(str(i))


    print(best, index)
    
    pickle.dump(nets[index], open(r"BEST!.pickle", "wb"))

    # capital = []
    # for x, trader in enumerate(traders):
    #     capital.append(trader.rollup[stock][1])
    # best = capital.index(max(capital))
    # print(traders[best].rollup)




def run(config_file):
    # Load configuration.
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(eval_genomes, 500)
    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, r'C:\Users\Vnixo\OneDrive\Desktop\ML_projects\gen_algo_stock_market\Dev\config.txt')
    run(config_path)


    
