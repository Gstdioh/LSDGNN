import argparse
import yaml
import torch
import numpy as np
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


def main(args):
    with open(args.config_filename, 'r', encoding="utf-8") as f:
        config = yaml.safe_load(f)

        # ============================== 下面构建模型 ============================== #
        model_args = config['model_args']

        if config['data']['graph_pkl_filename'] is not None:
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
            test_loss, test_results = supervisor.evaluate(dataset='test')

            base_message = ''

            # 输出评价指标：MAE、RMSE、MAPE
            supervisor.show_metrics(test_results['prediction'], test_results['truth'], base_message, 0.0)


if __name__ == '__main__':
    seed_torch(101)

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_filename', default='configs/PEMS03/train.yaml', type=str,
                        help='Configuration filename for restoring the model.')

    # ============================== 下面设置其他参数 ============================== #

    # ============================== 上面设置其他参数 ============================== #

    args = parser.parse_args()

    main(args)
