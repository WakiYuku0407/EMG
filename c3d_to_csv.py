import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import c3d

class Data_changer:
    def c3d_to_np(self, c3d_path):
        data = np.zeros((1, 16))
        with open(c3d_path, 'rb') as handle:
            reader = c3d.Reader(handle)
            for _, _, analog in reader.read_frames():
                data = np.concatenate((data, analog.T), axis=0)
        return data[1:, :]
    
    def save_csv(self, np_data, save_path):
        df = pd.DataFrame(np_data)
        df.to_csv(save_path)


def get_file_names(name):
    file_names = []
    for bps in [50, 75, 100]:
        for hand in ["l", "r"]:
            file_names.append("{}{}{}".format(name, bps, hand))
    return file_names

file_names = get_file_names("waki")
data_dir = "data/"
print(file_names)
data_changer = Data_changer()

for file_name in file_names:
    data_path = data_dir + file_name + ".c3d"
    save_name = data_dir + file_name + ".csv"
    emg_np = data_changer.c3d_to_np(data_path)
    data_changer.save_csv(emg_np[:, :4], save_name)
