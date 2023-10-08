import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import random
import os

from utils.data import load_graph_data
from utils.supervisor import Supervisor

from models.model import LSDGNN


def seed_torch(seed=101):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False


def transition_matrix(adj):
    r"""
    Description:
    -----------
    Calculate the transition matrix `P` proposed in DCRNN and Graph WaveNet.
    P = D^{-1}A = A/rowsum(A)

    Parameters:
    -----------
    adj: np.ndarray
        Adjacent matrix A

    Returns:
    -----------
    P:np.matrix
        Renormalized message passing adj in `GCN`.
    """
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    # P = d_mat.dot(adj)
    P = d_mat.dot(adj).astype(np.float32).todense()
    return P


def visualize(data, node_id):
    # figure_size = 12
    line_width = 1
    font_size = 10

    node_ids = [0]
    begin_time = 0
    end_time = 500

    labels = ["truth", "prediction"]

    # time_span = data.shape[0]
    # plt.rcParams['figure.figsize'] = figure_size
    for i, _ in enumerate(data):
        plot_data = data[i][begin_time:end_time, node_id, 0]
        plot_index = np.arange(plot_data.shape[0])
        plt.plot(plot_index, plot_data, linewidth=line_width, label=labels[i])
    plt.grid()
    plt.legend(fontsize=font_size)
    plt.show()
    plt.clf()


def main(args):
    with open(args.config_filename, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)

        # ============================== 下面构建模型 ============================== #
        model_args = config['model_args']

        _, _, adj_mx = load_graph_data(config['data']['graph_pkl_filename'])

        adj_mx = [transition_matrix(adj_mx), transition_matrix(adj_mx.T)]

        model_args['pre_graph'] = [torch.tensor(i) for i in adj_mx]

        model = LSDGNN(**model_args)
        # ============================== 上面构建模型 ============================== #

        supervisor = Supervisor(model, **config)

        mode = config.get('mode', 'train')

        if mode == 'train':
            # 训练模式
            supervisor.train()
        elif mode == 'test':
            # 测试模式
            # 第一次可视化，需要进行一次测试，然后保存结果
            # test_loss, test_results = supervisor.evaluate(dataset='test')
            # prediction = test_results['prediction'][0].numpy()
            # truth = test_results['truth'][0][12:].numpy()

            # np.savez_compressed("test_results.npz", prediction=prediction, truth=truth)

            # 后续可视化，可以使用已保存的结果
            test_results = np.load("test_results.npz")
            prediction = test_results['prediction']
            truth = test_results['truth']

            data = [truth, prediction]

            visualize(data, 0)

            # base_message = ''

            # 输出评价指标：MAE、RMSE、MAPE
            # supervisor.show_metrics(test_results['prediction'], test_results['truth'], base_message, 0.0)


if __name__ == '__main__':
    seed_torch(101)

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_filename', default='configs/PEMS04/test.yaml', type=str,
                        help='Configuration filename for restoring the model.')

    # ============================== 下面设置其他参数 ============================== #

    # ============================== 上面设置其他参数 ============================== #

    args = parser.parse_args()

    main(args)
