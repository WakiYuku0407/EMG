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
    file_name = os.path.splitext(name)[0]
    
    #EMGデータをフィルタリングして描画
    prosessed_data = processor(emg_datas.values)
    processor.plot_prosessed_data(save_path="./figure/", save_name = "processed_", file_name = file_name)
    normalized_data = processor.get_normalized_emg()
    processor.plot_normalized_data(save_path="./figure/", save_name = "normalized_", file_name = file_name)
    
    #右手か左手か
    if file_name.endswith('r'):
        hand = 'r'
    elif file_name.endswith('l'):
        hand = 'l'

    #W, H = processor.get_2muscle_NMF(hand = hand, conponent=1)
    #print(np.round(W).shape, np.round(H))
    W, H = processor.get_4muscle_NMF(conponent=3)
    print(np.round(W).shape)
    print(np.round(H))
    print(np.round(W).shape, np.round(H))
    #plt.plot(W)
    plt.plot(W@H)
    plt.show()
    plt.close()

