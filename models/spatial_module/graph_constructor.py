import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GraphConstructor(nn.Module):
    def __init__(self, **model_args):
        super(GraphConstructor, self).__init__()
        # attributes
        # 选择构造哪些矩阵
        self.use_pre = model_args['use_pre']
        self.pre_graph = model_args.get('pre_graph', [])
        self.use_ada = model_args['use_ada']
        self.use_dy = model_args['use_dy']
        # 自适应矩阵
        self.tanh_alpha = model_args['tanh_alpha']  # 3
        self.top_k = model_args['top_k']  # 用于稀疏化
        # 动态矩阵
        self.hidden_dim = model_args['hidden_dim']  # 历史数据信息转换后的维度
        self.node_dim = model_args['node_dim']  # 40
        self.time_day_dim = model_args['time_day_dim']  # 16
        self.time_week_dim = model_args['time_week_dim']  # 7
        self.input_seq_len = model_args['input_seq_len']  # 输入序列的长度

        # 自适应矩阵的构建
        self.lin1 = nn.Linear(self.node_dim, self.node_dim)
        self.lin2 = nn.Linear(self.node_dim, self.node_dim)

        # 动态矩阵的构建
        # Time Series Feature Extraction
        self.dropout = nn.Dropout(model_args['dy_graph_dropout'])
        self.fc_ts_emb1 = nn.Linear(self.input_seq_len, self.hidden_dim * 2)
        self.fc_ts_emb2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.ts_feat_dim = self.hidden_dim
        # Distance Score
        self.all_feat_dim = self.ts_feat_dim + self.node_dim + self.time_day_dim + self.time_week_dim
        self.WQ = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.WK = nn.Linear(self.all_feat_dim, self.hidden_dim, bias=False)
        self.bn = nn.BatchNorm1d(self.hidden_dim * 2)

    def forward(self, history_data, node_emb_s, node_emb_t, time_in_day_feat, day_in_week_feat):
        """

        Args:
            history_data: (batch_size, seq_len, num_nodes, 1)
            node_emb_s: (num_nodes, node_dim)
            node_emb_t: (num_nodes, node_dim)
            time_in_day_feat: (batch_size, seq_len, num_nodes, time_emb_dim)
            day_in_week_feat: (batch_size, seq_len, num_nodes, time_emb_dim)

        Returns:
            adjs: [(num_nodes, num_nodes), ...]
        """
        device = history_data.device

        adjs = []

        # 预定义矩阵，2个
        if self.use_pre:
            adjs.append(self.pre_graph[0].to(device))
            adjs.append(self.pre_graph[1].to(device))

        # 自适应矩阵，2个
        if self.use_ada:
            node_emb_s = torch.tanh(self.tanh_alpha * self.lin1(node_emb_s))
            node_emb_t = torch.tanh(self.tanh_alpha * self.lin2(node_emb_t))

            a = torch.mm(node_emb_s, node_emb_t.transpose(1, 0)) - torch.mm(node_emb_t, node_emb_s.transpose(1, 0))
            adj = F.relu(torch.tanh(self.tanh_alpha * a))

            # 稀疏化
            mask = torch.zeros(node_emb_s.shape[0], node_emb_s.shape[0]).to(device)  # 记得放入cuda
            mask.fill_(float('0'))
            s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.top_k, 1)
            mask.scatter_(1, t1, s1.fill_(1))
            adj = adj * mask

            adjs += [adj, adj.T]

        # 动态矩阵，2个
        if self.use_dy:
            # 时间戳信息，取最后的一个时间 (batch_size, num_nodes, time_emb_dim)
            time_in_day_feat = time_in_day_feat[:, -1, :, :]
            day_in_week_feat = day_in_week_feat[:, -1, :, :]

            # 历史数据信息
            # X -> (batch_size, seq_len, num_nodes) -> (batch_size, num_nodes, seq_len)
            X = history_data[:, :, :, 0].transpose(1, 2).contiguous()
            [batch_size, num_nodes, seq_len] = X.shape
            X = X.view(batch_size * num_nodes, seq_len)
            # (batch_size * num_nodes, hidden_dim)
            dy_feat = self.fc_ts_emb2(self.dropout(self.bn(F.relu(self.fc_ts_emb1(X)))))
            # (batch_size, num_nodes, hidden_dim)
            dy_feat = dy_feat.view(batch_size, num_nodes, -1)

            # 节点嵌入信息 (batch_size, num_nodes, node_dim)
            node_emb_s = node_emb_s.unsqueeze(0).expand(batch_size, -1, -1)
            node_emb_t = node_emb_t.unsqueeze(0).expand(batch_size, -1, -1)

            # 计算动态矩阵
            X1 = torch.cat([dy_feat, time_in_day_feat, day_in_week_feat, node_emb_s], dim=-1)  # 前向
            X2 = torch.cat([dy_feat, time_in_day_feat, day_in_week_feat, node_emb_t], dim=-1)  # 后向
            X = [X1, X2]
            for _ in X:
                Q = self.WQ(_)
                K = self.WK(_)
                QKT = torch.bmm(Q, K.transpose(-1, -2)) / math.sqrt(self.hidden_dim)
                W = torch.softmax(QKT, dim=-1)
                adjs.append(W)

        return adjs
