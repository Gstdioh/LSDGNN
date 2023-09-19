import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import copy


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class MinMaxScaler:
    """
    Standard the input
    """

    def __init__(self, min_data, max_data):
        self.min_data = min_data
        self.max_data = max_data

    def transform(self, data):
        mid = self.min_data
        base = self.max_data - self.min_data
        normalized_data = (data - mid) / base
        normalized_data = 2. * normalized_data - 1.

        return normalized_data

    def inverse_transform(self, data):
        mid = self.min_data
        base = self.max_data - self.min_data

        recovered_data = (data + 1) / 2.
        recovered_data = recovered_data * base + mid

        return recovered_data


class LoadData(Dataset):
    def __init__(self, data, scaler, time_interval=5, history_length=12, prediction_length=12):
        """
        加载数据的类
        :param data: np.array, 交通数据（分为训练集、验证集、测试集）
        :param time_interval: int, time interval between two traffic data records (min).
        :param history_length: int, length of history data to be used.
        :param prediction_length: int, length of data to be predicted.
        """

        self.data = data
        self.time_interval = time_interval  # 5 min
        self.history_length = history_length  # 12
        self.prediction_length = prediction_length  # 12

        # 归一化的参数
        # self.mean = self.data[..., 0].mean()
        # self.std = self.data[..., 0].std()

        # 对交通数据进行预处理
        # # (times, num_nodes, node_features) -> (times, num_nodes, 1) 速度，弃，不止一个特征
        # self.data = self.data[:, :, 0][:, :, np.newaxis]
        # 只对第一个特征（速度）进行归一化，其他特征是一天中所处的时间/所处的星期（可选）
        self.target_src = copy.deepcopy(self.data[..., 0: 1])  # 深拷贝，目标值不进行归一化
        self.data[..., 0] = scaler.transform(self.data[..., 0])

    def __len__(self):
        """
        获取数据集的长度
        :return: length of dataset (number of samples).
        """
        # data (times, num_nodes, 1)
        # len = times - history_length - prediction_length + 1
        # times = self.data.shape[0] // 3  # 这里除以 60 减少数据，为了加快训练，用于 debug
        times = self.data.shape[0]
        return times - self.history_length - self.prediction_length + 1

    def __getitem__(self, index):
        """
        取样本
        :param index: int
        :return:
            graph: torch.tensor, (num_nodes, num_nodes)
            data_x: torch.tensor, (history_length, num_nodes, 1)
            data_y: torch.tensor, (prediction_length, num_nodes, 1)
        """
        x_start = index
        x_end = index + self.history_length
        y_start = x_end
        y_end = y_start + self.prediction_length

        # data_x 进行了归一化
        data_x = LoadData.to_tensor(self.data[x_start: x_end, :, :])  # (history_length, num_nodes, num_features)
        # data_y 是原始的值，没有进行归一化
        data_y = LoadData.to_tensor(self.target_src[y_start: y_end, :, 0: 1])  # (prediction_length, num_nodes, 1)

        return {"x": data_x, "y": data_y}

    @staticmethod
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float)


def load_dataset(data_file, forcast_type, batch_size=32, val_batch_size=1, test_batch_size=1,
                 time_interval=5, history_length=12, prediction_length=12, **kwargs):
    all_data = {}

    # 读取数据文件
    np_data = np.load(data_file)

    scaler = None
    if forcast_type == 'flow':
        # min_max 用于交通流量的归一化
        scaler = MinMaxScaler(min_data=np_data['train'][..., 0].min(), max_data=np_data['train'][..., 0].max())
    else:
        # standard 用于交通速度的归一化
        scaler = StandardScaler(mean=np_data['train'][..., 0].mean(), std=np_data['train'][..., 0].std())

    # 创建 Dataset
    for category in np_data.files:
        all_data[category + '_data'] = LoadData(data=np_data[category],
                                                scaler=scaler,
                                                time_interval=time_interval,
                                                history_length=history_length,
                                                prediction_length=prediction_length)

    # 创建 DataLoader
    all_data['train_loader'] = DataLoader(all_data['train_data'], batch_size, shuffle=True)
    all_data['val_loader'] = DataLoader(all_data['val_data'], val_batch_size, shuffle=False)
    all_data['test_loader'] = DataLoader(all_data['test_data'], test_batch_size, shuffle=False)

    # 归一化对所有的数据都要统一！！！
    # 之前我是分别对train、val、test进行不同的归一化，见get_traffic_data_src.py代码
    all_data['scaler'] = scaler

    return all_data


if __name__ == '__main__':
    all_data = load_dataset(r"D:\1_program\0_Traffic_Predition\DCRNN_My\dataset\PEMS04\pems04_6_2_2.npz",
                            32)

    print(all_data['train_data'].data.shape)
    print(all_data['val_data'].data.shape)
    print(all_data['test_data'].data.shape)

    print(len(all_data['train_loader']))

    for item in all_data['train_loader']:
        x = item['x']
        y = item['y']
        print(x)
        print(y)

    for data in all_data['train_loader']:
        x = data['x']
        y = data['y']

        print(x.shape)
        print(y.shape)

        break
