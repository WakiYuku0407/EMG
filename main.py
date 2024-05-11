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
    prosessed_data = processor(emg_datas.values)
    plt.plot(prosessed_data[2000:10000, :])
    plt.show()
    print(prosessed_data.shape)

