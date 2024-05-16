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

    #2筋の1シナジーを求める
    for i in range(2):
        n_synergy = i+1
        W, H = processor.get_2muscle_NMF(hand = hand, conponent=n_synergy)
        plot_bar(H, hand, n_synergy, name, "./figure/synergy/")
        #print(processor.get_VAF(W, H))

    #4筋の1~3を調べる
    VAFs = np.zeros(4)
    vars = np.zeros(4)
    for i in range(4):
        n_synergy = i+1
        W, H = processor.get_4muscle_NMF(conponent=n_synergy)
        plot_bar(H, None, n_synergy, name, "./figure/synergy/")
        VAFs[i], vars[i] = processor.get_VAF(W, H)
    print("vars", vars)

    #VAFのプロット
    plt.errorbar([1, 2, 3, 4], VAFs, yerr=vars, fmt='o', capsize=5)
    plt.ylim(0.5, 1.05)
    plt.xticks(range(1,5))
    plt.title("VAF:{}".format(name))
    plt.xlabel("number of synergy")
    plt.ylabel("VAF(%)")
    plt.savefig("./figure/VAF/VAF_{}.png".format(name))
    plt.close()
    

