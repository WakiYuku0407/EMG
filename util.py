import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.stats import norm
from sklearn.decomposition import NMF


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


class Emg_processor:
    def __init__(self, sampling = 2000):
        self.sampling = sampling
        self.lowcut = 40
        self.highcut = 35
    
    def __call__(self, emg_datas):
        self.law_emg_datas = emg_datas
        emg_data =  self.adjust_emg(self.law_emg_datas)
        emg_data = self.high_pass_filter(emg_data)
        emg_data = self.demeaned(emg_data)
        emg_data = np.abs(emg_data)
        emg_data = self.low_pass_filter(emg_data)
        self.emg_data = emg_data
        return emg_data

    def high_pass_filter(self, data):
        num_cols = data.shape[1]
        filtered_data = np.zeros_like(data)
        for i in range(num_cols):
            b, a = signal.butter(3, self.highcut, btype="high", analog=False, fs = self.sampling)
            filtered_data[:, i] = signal.lfilter(b, a, data[:, i])
        return filtered_data
    
    def low_pass_filter(self, data):
        num_cols = data.shape[1]
        filtered_data = np.zeros_like(data)
        for i in range(num_cols):
            b, a = signal.butter(1, self.lowcut, btype="low", analog=False, fs = self.sampling)
            filtered_data[:, i] = signal.lfilter(b, a, data[:, i])
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
            # 正規分布の信頼区間を計算します
            lower_bound, upper_bound = norm.interval(0.95, loc=mean, scale=std)
            # 信頼区間から外れる値を修正します
            adjusted_emg_data = np.clip(emg_data, lower_bound, upper_bound)
            adjusted_data[:, i] = adjusted_emg_data
        return adjusted_data
    
    def get_NMF(self):
        data = self.emg_data
        nmf = NMF(n_components=3)
        nmf.fit(data.T) #(4, N)
        W = nmf.fit_transform(data)
        H = nmf.components_
        return W, H


