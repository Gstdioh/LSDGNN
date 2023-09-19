import torch
import torch.nn as nn
import numpy as np
from torch.nn import init


class GCNReset(nn.Module):
    def __init__(self, input_dim, output_dim, **model_args):
        """

        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            ks: 最大阶数
            reset_beta: 重启概率
            **model_args:
        """
        super(GCNReset, self).__init__()

        self.ks = model_args['ks']  # 2
        self.reset_beta = model_args['reset_beta']  # 0.05
        # 选择构造哪些矩阵
        self.use_pre = model_args['use_pre']
        self.use_ada = model_args['use_ada']
        self.use_dy = model_args['use_dy']
        self.ada_adj_i = -10  # 初始记为没有自适应矩阵
        if self.use_ada:
            # 有自适应矩阵，则去找其下标，因为只有自适应矩阵需要进行：加自环、归一化
            self.ada_adj_i = int(self.use_pre) * 2  # 没有预定义矩阵，则下标为 0；有，则下标为 2

        # 无order下，矩阵的个数
        self.num_adjs = (int(self.use_pre) + int(self.use_ada) + int(self.use_dy)) * 2

        # 矩阵的个数 = num_adjs * ks + 1，（num_adjs表示邻接矩阵的个数，ks表示最大阶数，1表示输入自身）
        self.mlp = nn.Linear((self.num_adjs * self.ks + 1) * input_dim, output_dim)

    def forward(self, x, adjs):
        """带重启概率的GCN

        Args:
            x: (batch_size, seq_len, num_nodes, input_dim)
            adjs: [(num_nodes, num_nodes), ...], 所有的邻接矩阵

        Returns:
            (batch_size, seq_len, num_nodes, output_dim)
        """
        # x (batch_size, seq_len, input_dim, num_nodes) 转换一下，方便右乘矩阵
        x = x.transpose(-2, -1).contiguous()

        output = [x]
        for adj_i, adj in enumerate(adjs):
            # 只有自适应邻接矩阵需要进行：加自环、归一化
            if adj_i == self.ada_adj_i or adj_i == self.ada_adj_i + 1:
                adj = adj + torch.eye(adj.size(0)).to(x.device)
                d = adj.sum(1).unsqueeze(-1)
                adj = adj / d

            h = x
            for i in range(self.ks):
                # 右乘邻接矩阵
                # h (batch_size, seq_len, input_dim, num_nodes)
                # adj ((batch_size), num_nodes, num_nodes) -> ((batch_size), 1, num_nodes, num_nodes) 可能有batch
                h = torch.matmul(h, adj.unsqueeze(-3))  # 之前的 torch.einsum('blnd,nm->blmd', (h, adj))
                h = self.reset_beta * x + (1 - self.reset_beta) * h
                output.append(h)

        # ho (batch_size, seq_len, (num_adjs * ks + 1) * input_dim, num_nodes)
        ho = torch.cat(output, dim=-2)

        # ho (batch_size, seq_len, num_nodes, (num_adjs * ks + 1) * input_dim)
        ho = ho.transpose(-2, -1)

        # ho (batch_size, seq_len, num_nodes, output_dim)
        ho = self.mlp(ho)

        return ho
