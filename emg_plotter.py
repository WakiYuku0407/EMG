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
    # サブプロットを作成
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10,15))
    for i in range(4):
        axes[i].set_ylim(-120, 120)
        axes[i].set_title(num2muscle[i+1])
        axes[i].grid()
        axes[i].set_ylabel("EMG")
        axes[i].set_xlabel("time(s)")
        emg_datas.iloc[:, i].plot(ax = axes[i], color = color_list[i])
    plt.suptitle(name)
    plt.tight_layout(pad = 2.0)
    plt.savefig("./figure/{}.png".format(os.path.splitext(name)[0]))
