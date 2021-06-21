import strawberryfields as sf
from strawberryfields.apps import  sample, plot, subgraph, clique, similarity
import networkx as nx
import numpy as np
import pandas as pd
import time

import pandas_datareader as web   #used to grab the stock prices, with yahoo
import matplotlib.pyplot as plt
import seaborn, statistics, xlrd
import itertools
from BasicCompute import BasicCompute
from SamplingSetting import SamplingSetting
from PearsonMutation import PearsonMutation
from CommonFunc import CommonFunc
import concurrent.futures as CF
import multiprocessing as MP

class Quantum(object):
    def __init__(self, save_path, file_name, start, end, rolling=1, window=5, past_days=126):
        self.save_path = save_path
        self.file_name = file_name
        self.start = start
        self.end = end
        self.rolling = rolling
        self.window = window
        self.past_days = past_days

    def draw_cliques(self, cliq_size_p, save_cliques_file):
        dates = []
        for d in range(self.start, self.end, self.rolling):
            df = pd.read_csv(self.file_name)[d:d + self.window]
            df.dropna(axis=1, how='any', thresh=None, subset=None, inplace=True)
            dates.append(df['x'].values[0])
        new_list = list(zip(dates, cliq_size_p))
        #print(new_list)
        new_df = pd.DataFrame(new_list, columns=['Date', 'CliqueSize'])
        plt.figure()
        plt.xlabel('Date(s)')
        plt.ylabel('Clique_size')
        new_df.plot(figsize=(25, 8), style=['--o'], color='black')
        plt.axhline(y=np.mean(cliq_size_p), color='r', linestyle='-')
        plt.savefig(save_cliques_file+'.jpg')

    def thread_part(self, d):
        print("sample: ", d, " - ", d + self.rolling)
        df = pd.read_csv(self.file_name)[d:d + self.window]  # shape(day, 45+x)
        df.dropna(axis=1, how='any', thresh=None, subset=None, inplace=True)
        pearson_mutation = PearsonMutation(self.file_name, df, d + self.window, self.past_days)
        BB = pearson_mutation.pearson_correlation()  # this is the correlation matrix from Pearson

        samples = []
        step = 2  # In this version, change step to be 2.
        for i in range(0, int(BB.shape[0] / step) + 2, step):  # I consider 100 symbols for the moment, 9
            for j in range(0, int(BB.shape[0] / step) + 2, step):
                # print('(i:4+i)', i, ': ', 4+i, '(j: 4+j)', j, ':', 4+j)
                B = BB[i:(4 + i), j:(4 + j)]
                samping_settring = SamplingSetting(device)
                res = samping_settring.run_device(B)
                samples.append(list(res.samples[0]))
        common_func.saveList(samples, self.save_path + str(self.past_days) + '_' + str(d) + '.npy')

    def do_sampling(self, common_func, device):
        """Do sampling:
        start: the index of started date.
        end: the index of end date.
        rolling: the days of dataframe moves each step.
        window: the size of the window we select to do sampling.
        past_days: how many days we selected to compute the mean."""
        #for d in range(self.start, self.end, self.rolling):  # day: How many rows we cut each time
        d = np.arange(self.start, self.end)
        #print(d)
       # with CF.ThreadPoolExecutor() as executor:
        with CF.ProcessPoolExecutor() as executor:
            executor.map(self.thread_part, d)

    def do_cliques(self, common_func):
        cliq_size_p = []
        clique_symbols = []
        print("Calculating the cliques and storing the clique size ....")

        for d in range(self.start, self.end, self.rolling):
            samples = common_func.loadList(self.save_path + str(self.past_days) + '_' + str(d) + '.npy')
            df = pd.read_csv(self.file_name)[d:d+self.window]
            df.dropna(axis=1, how='any', thresh=None, subset=None, inplace=True)
            #df.fillna(df.mean(axis=1), inplace=True)
            pearson_mutation = PearsonMutation(self.file_name, df, d + self.window, self.past_days)
            A = pearson_mutation.pearson_correlation()
            np.fill_diagonal(A, 0)
            A = (A > 0.8) * A
            g = nx.Graph(A, with_labels=True)
            samples = sample.to_subgraphs(samples[0:10], graph=g)

            cliques = [clique.shrink(i, g) for i in samples]
            cliques = [clique.search(c, g, 10) for c in cliques]
            cliques.sort()
            cliques = list(k for k, _ in itertools.groupby(cliques))
            cliques = sorted(cliques, key=len, reverse=True)
            print(len(cliques[0]))
            cliq_size_p.append(len(cliques[0]))
            clique_symbols.append(cliques)
        return cliq_size_p, clique_symbols

if __name__ == '__main__':
    file_name = '31_Stocks_Index_symbols_data.csv'
    # file_name = '45_symbols_data.csv'
    #save_path = '45sym/simulon_gaussian_window_ten-'
    save_path = 'Stock_Sym/'
    window = 5
    rolling = 1
    start_date = '2007-01-01' #'2016-01-01'
    end_date = '2015-12-31' #'2012-12-30'
    device = 'gaussian'  # 'gaussian' 'simulon_gaussian' For sampling
    save_path += str(window) + device + '_'
    path = 'save_img/'
    start_time = time.perf_counter()
    common_func = CommonFunc()
    df = pd.read_csv(file_name)

    # Given start and end dates, find the indexes of the dates.
    basic_compute = BasicCompute(df)
    start = basic_compute.get_meaningful_index(start_date)
    end = basic_compute.get_meaningful_index(end_date)
    print(start, end)
    past_days = 63
    #start =  23196
    my_quantum = Quantum(save_path, file_name, start, end, rolling, window, past_days=past_days)
    samples = my_quantum.do_sampling(common_func, device)
    finish_time = time.perf_counter()
    #start = 22490
    #cliq_size_p,clique_symbols = my_quantum.do_cliques(common_func)

    #save_cliques_file = common_func.save_to_file(path, cliq_size_p, clique_symbols)      # save cliques to the file

    #my_quantum.draw_cliques(cliq_size_p, save_cliques_file)
    print('Fini! The total time: ', finish_time-start_time)