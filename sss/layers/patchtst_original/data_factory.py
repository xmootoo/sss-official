import os
from sss.layers.patchtst_original_data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from sss.utils.dataloading import load_patchtst_data
from sss.utils.datasets import ForecastingDataset
from torch.utils.data import DataLoader


data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    "electricity": Dataset_Custom,
    "weather": Dataset_Custom,
    "traffic": Dataset_Custom,
    "exchange_rate": Dataset_Custom,
    "illness": Dataset_Custom,
}

def get_og_dataset(data='ETTm1',
                  flag='train',
                  seq_len=512,
                  pred_len=96,
                  root_path="./data/forecasting"): # Only relevant for univariate forecasting

    data_path = data + ".csv"

    Data = data_dict[data]

    dataset = Data(
        root_path=root_path,
        data_path=data_path,
        flag=flag,
        size=[seq_len, 48, pred_len], # label_len=48 (unused)
        features='M', # For multivariate forecasting
        target='OT', # The target channel for univariate ETT
        timeenc=0, # Used for separate time tracking tensors seq_x_mark and seq_y_mark (unused)
        freq='h' # Used for separate time tracking tensors seq_x_mark and seq_y_mark (unused)
    )
    return dataset


if __name__ == "__main__":
    import numpy as np
    import random
    import torch
    root_path = "../../../data/forecasting"
    dataset_name = "electricity"

    datas = []

    for flag in ["train", "val", "test"]:
        dataset = get_og_dataset(data=dataset_name, flag=flag, root_path=root_path)
        data = dataset
        datas.append(data.data_x)

    train, val, test = load_patchtst_data(dataset_name=dataset_name, seq_len=512, pred_len=96, scale=True, train_split=0.7, val_split=0.1)

    my_datas = [train, val, test]

    for data, my_data in zip(datas, my_datas):
        print(np.array_equal(data, my_data.T))

    og_train_data = get_og_dataset(data=dataset_name, flag="train", root_path=root_path)
    my_train_data = ForecastingDataset(data=train, seq_len=512, pred_len=96, dtype="float64")

    n = random.randint(0, len(og_train_data))

    print(type(og_train_data[n]))
    og_x, og_y = og_train_data[n]
    my_x, my_y = my_train_data[n]
    my_x = my_x.numpy()
    my_y = my_y.numpy()

    print(np.array_equal(og_x, my_x))
    print(np.array_equal(og_y, my_y))
