import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.stats import norm
from sklearn.decomposition import NMF
import math


# CSS4_COLORSから色のリストを取得
color_list = ['royalblue', 'red', 'green', 'orange']
# ラベルと筋肉名の対応付け
num2muscle = {  1 : "Right Biceps",
                2 : "Right Triceps",
                3 : "Left Biceps",
                4 : "Left Triceps"}


def load_csv(folder_path):
    datas = {}
    # フォルダ内の全てのCSVファイルを処理
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            print(file_path)
            # CSVファイルをPandasのDataFrameとして読み込む
            df = pd.read_csv(file_path, index_col=0)
            # DataFrameからNumPy配列に変換
            datas[filename] = df
            
            # 読み込んだデータを使って何かを行う（ここでは例として表示）
            print("File:", filename)
            print("Data shape:", df.shape)
            print("Data:")
    
    return datas

def plot_bar(H, type, n_conponents, name, save_path):
    if type == 'r':
        labels = list(num2muscle.values())[:2]
        n_muscle = 2
    elif type == 'l':
        labels = list(num2muscle.values())[2:]
        n_muscle = 2
    elif type == None:
        labels = list(num2muscle.values())
        n_muscle = 4

    #サブプロットの設定
    if n_conponents > 1:
        fig , axes = plt.subplots(1, n_conponents, figsize = (5*n_conponents, 5))
        plt.suptitle("synergy={}, {}".format(n_conponents, name))  
        plt.tight_layout() 
        for i in range(n_conponents):
            data = H[i, :]
            print(data)
            axes[i].bar(labels, data) 
            axes[i].set_title("W{}".format(i+1))
            axes[i].set_ylim(0, 13)
        plt.savefig(save_path + "{}_muscle{}_synergy{}.png".format(name, n_muscle , n_conponents))

    else:
        data = H.flatten()
        plt.bar(labels, data, linewidth = 1)
        plt.title("synergy={}, {}".format(n_conponents, name))
        plt.ylim(0, 13)
        plt.savefig(save_path + "{}_muscle{}_synergy{}.png".format(name, n_muscle,  n_conponents))
    plt.close()

class Emg_processor:
    def __init__(self, sampling = 2000):
        self.sampling = sampling
        self.lowcut = 40
        self.highcut = 35
    
    def __call__(self, emg_datas):
        self.law_emg_datas = emg_datas
        emg_data =  self.adjust_emg(self.law_emg_datas)
        #emg_data = self.law_emg_datas
        emg_data = self.high_pass_filter(emg_data)
        emg_data = self.demeaned(emg_data)
        emg_data = np.abs(emg_data)
        emg_data = self.low_pass_filter(emg_data)
        self.emg_data = np.clip(emg_data, 0, None) #マイナス対策
        return self.emg_data

    def high_pass_filter(self, data):
        num_cols = data.shape[1]
        filtered_data = np.zeros_like(data)
        for i in range(num_cols):
            b, a = signal.butter(3, self.highcut, btype="high", analog=False, fs = self.sampling)
            filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
        return filtered_data
    
    def low_pass_filter(self, data):
        num_cols = data.shape[1]
        filtered_data = np.zeros_like(data)
        for i in range(num_cols):
            b, a = signal.butter(1, self.lowcut, btype="low", analog=False, fs = self.sampling)
            filtered_data[:, i] = signal.filtfilt(b, a, data[:, i])
        return filtered_data
    
    def demeaned(self, data):
        mean = np.mean(data, axis=0)
        return data - mean

    def adjust_emg(self, data):
        adjusted_data = np.zeros_like(data)
        for i in range(data.shape[1]):
            emg_data = data[:, i]
            mean = np.mean(emg_data)
            std = np.std(emg_data)
            # # 正規分布の信頼区間を計算します
            # lower_bound, upper_bound = norm.interval(0.95, loc=mean, scale=std)
            # # 信頼区間から外れる値を修正します
            lower_bound = mean - 3*std
            upper_bound = mean + 3*std
            adjusted_emg_data = np.clip(emg_data, lower_bound , upper_bound)
            adjusted_data[:, i] = adjusted_emg_data
        return adjusted_data
    
    def get_normalized_emg(self):
        data = self.emg_data
        max = np.max(data, axis=0)
        norm_data = data/max
        self.normalized_emg_data = norm_data
        return norm_data
    
    def plot_prosessed_data(self, save_path, save_name, file_name):
        self.plotter(self.emg_data, save_path, save_name, file_name)

    def plot_normalized_data(self, save_path, save_name, file_name):
        self.sub_plotter(self.normalized_emg_data , save_path, save_name, file_name)

    def plotter(self, data, save_path, save_name, file_name):
        t = np.arange(0, data.shape[0]/self.sampling, 1/self.sampling)
        for i in range(data.shape[1]):
            plt.plot(t, data[:, i], label = num2muscle[i+1])
        plt.xlabel("time(s)")
        plt.ylabel("EMG")
        plt.title(file_name)
        plt.legend()
        plt.savefig(save_path + save_name + file_name + ".png")
        plt.close()
        #np.save(save_path + save_name + file_name + ".npy", data)

    def sub_plotter(self, data, save_path, save_name, file_name):
        t = np.arange(0, data.shape[0]/self.sampling, 1/self.sampling)
        # サブプロットを作成
        fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10,15))
        for i in range(4):
            axes[i].set_ylim(0, 1)
            axes[i].set_title(num2muscle[i+1])
            axes[i].grid()
            axes[i].set_ylabel("EMG")
            axes[i].set_xlabel("time(s)")
            axes[i].plot(t ,data[:, i] , color = color_list[i])
        plt.suptitle(file_name)
        plt.tight_layout(pad = 2.0)
        fig.savefig(save_path + save_name + file_name + ".png")
        plt.close()

    def get_2muscle_NMF(self, hand = None, conponent = 1):
        data = self.normalized_emg_data
        if hand == 'r':
            data = data[:, :2]
        elif hand == 'l':
            data = data[:, 2:]
        else:
            print("error: chose hand")
            raise
        W, H = self.fit_NMF_towmuscle(data, conponent)
        return W, H
    
    def get_4muscle_NMF(self, conponent = 1):
        data = self.normalized_emg_data
        W, H = self.fit_NMF_towmuscle(data, conponent)
        return W, H

    def fit_NMF_towmuscle(self, data, conponent = 1):
        nmf = NMF(n_components=conponent, init = 'nndsvd', max_iter=1000)
        nmf.fit(data.T) #(2, N)
        W = nmf.fit_transform(data)
        H = nmf.components_
        return W, H
    
    def get_VAF(self, W, H):
        restoration = W@H #復元
        squared_err = (self.normalized_emg_data - restoration)**2
        squared_data = self.normalized_emg_data ** 2
        var = np.mean(np.var(squared_err, axis=0)/np.mean(squared_data, axis=0))
        VAF = 1 - np.mean((np.sum(squared_err, axis=0)/ np.sum(squared_data, axis=0)))
        return VAF, var



