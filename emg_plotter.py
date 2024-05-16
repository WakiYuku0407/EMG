import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from util import *

# CSVファイルがあるフォルダのパス
folder_path = "data/"
datas = load_csv(folder_path)

#生データのプロット
for name, emg_datas in datas.items():
    t = np.arange(0, emg_datas.shape[0]/2000, 1/2000)
    # サブプロットを作成
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10,15))
    for i in range(4):
        axes[i].plot(t, emg_datas.iloc[:, i].to_numpy(), color = color_list[i], linewidth = 0.5)
        axes[i].set_ylim(-150, 150)
        axes[i].set_title(num2muscle[i+1])
        axes[i].set_ylabel("EMG")
        axes[i].set_xlabel("time(s)")
        axes[i].grid()
    plt.suptitle(name)
    plt.tight_layout(pad = 2.0)
    plt.savefig("./figure/{}.png".format(os.path.splitext(name)[0]))

