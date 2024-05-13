import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from util import *

# CSVファイルがあるフォルダのパス
folder_path = "data/"
all_muscle_emg_datas = load_csv(folder_path)

sampling = 2000
processor = Emg_processor(sampling)
for name, emg_datas in all_muscle_emg_datas.items():
    #EMGデータをフィルタリングして描画
    prosessed_data = processor(emg_datas.values)
    plt.plot(prosessed_data[:2000, :])
    plt.xlabel("time(s)")
    plt.ylabel("EMG")
    plt.savefig("./figure/processed_{}.png".format(os.path.splitext(name)[0]))
    plt.close()
    np.save("./results/{}.npy".format(os.path.splitext(name)[0]), prosessed_data)
    
    #NMF
    W, H = processor.get_NMF()
    print(np.round(W).shape, np.round(H))

