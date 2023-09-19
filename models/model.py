import torch
import torch.nn as nn
import torch.nn.functional as F
from .spatial_module import *
from .temporal_module import *


class LSDGNLayer(nn.Module):
    def __init__(self, c_out_set, seq_len, short_padding, long_padding,
                 temporal_emb=True, spatial_emb=True, **model_args):
        super().__init__()
        # 属性
        self.hidden_dim = model_args['hidden_dim']  # 32
        self.skip_channels = model_args['skip_channels']  # 64
        self.input_seq_len = model_args['input_seq_len']  # 12
        self.dropout = model_args['dropout']  # 0.3
        self.num_nodes = model_args['num_nodes']  # 207
        self.c_out_set = c_out_set  # [16, 8, 8]
        self.conv_channels = sum(self.c_out_set)  # 32, 单个tcn的输出维度
        self.gcn_input_channels = self.conv_channels
        self.seq_len = seq_len  # 当前输入的序列长度，开始为12

        self.short_kernel_set = model_args['short_kernel_set']  # [1, 2, 3]
        self.long_kernel_set = model_args['long_kernel_set']  # [1, 6, 7]

        self.short_padding = short_padding
        self.long_padding = long_padding

        self.temporal_emb = temporal_emb
        self.spatial_emb = spatial_emb

        # 时间和空间嵌入
        if self.temporal_emb:
            # input (B, D, N, L), emb (1, D, 1, L)
            self.temporal_embedding = nn.Parameter(torch.FloatTensor(1, self.hidden_dim, 1, self.seq_len))
        if self.spatial_emb:
            # input (B, D, N, L), emb (1, D, N, 1)
            self.spatial_embedding = nn.Parameter(torch.FloatTensor(1, self.hidden_dim, self.num_nodes, 1))

        # sl_tcn_filter
        self.sl_tcn_filter = SLTCN(self.hidden_dim, self.c_out_set, self.short_padding, self.long_padding, self.seq_len, model_args)

        # sl_tcn_gate
        self.sl_tcn_gate = SLTCN(self.hidden_dim, self.c_out_set, self.short_padding, self.long_padding, self.seq_len, model_args)

        # sl_tcn_dropout
        self.sl_tcn_dropout = nn.Dropout(self.dropout)

        self.residual_len = self.seq_len - self.short_kernel_set[-1] + 1 + self.short_padding  # tcn卷积后剩下序列的长度

        # skip_conv
        self.tcn_skip_conv = nn.Conv2d(self.gcn_input_channels, self.skip_channels,
                                       kernel_size=(1, self.residual_len))

        # gcn_reset
        self.gcn_reset = GCNReset(self.gcn_input_channels, self.hidden_dim, **model_args)

        # normalization, (B, D, N, L) 计算(D, N, L)的归一化参数，然后进行归一化
        self.norm = nn.LayerNorm([self.hidden_dim, self.num_nodes * self.residual_len])

        self.reset_parameter()

    def reset_parameter(self):
        if self.temporal_emb:
            nn.init.xavier_normal_(self.temporal_embedding, gain=0.0003)

        if self.spatial_emb:
            nn.init.xavier_normal_(self.spatial_embedding, gain=0.0003)

    def forward(self, x, adjs):
        # x (batch_size, hidden_dim, num_nodes, input_seq_len) (B, D, N, L)
        batch_size, residual_channels, num_nodes, input_seq_len = x.shape

        residual = x

        # 时间和空间嵌入
        if self.temporal_emb:
            x = x + self.temporal_embedding
        if self.spatial_emb:
            x = x + self.spatial_embedding

        # sl_tcn_filter (batch_size, conv_channels, num_nodes, input_seq_len)
        sl_tcn_filter = self.sl_tcn_filter(x)
        sl_tcn_filter = torch.tanh(sl_tcn_filter)

        # sl_filter (batch_size, conv_channels, num_nodes, input_seq_len)
        sl_tcn_gate = self.sl_tcn_gate(x)
        sl_tcn_gate = torch.sigmoid(sl_tcn_gate)

        x = sl_tcn_filter * sl_tcn_gate

        x = self.sl_tcn_dropout(x)

        # tcn的跳跃连接 skip_tcn (batch_size, skip_channels, num_nodes, input_seq_len)
        skip_tcn = self.tcn_skip_conv(x)

        # gcn_reset
        # x (batch_size, input_seq_len, num_nodes, gcn_input_channels)
        x = x.transpose(1, 3)
        # x (batch_size, input_seq_len, num_nodes, hidden_dim) (B, L, N, D)
        x = self.gcn_reset(x, adjs)
        # x (batch_size, hidden_dim, num_nodes, input_seq_len)
        x = x.transpose(1, 3)

        # 残差连接 + 归一化, post norm
        x = x + residual[:, :, :, -x.size(3):]
        # x (batch_size, hidden_dim, num_nodes * input_seq_len)
        x = x.reshape(batch_size, residual_channels, -1)
        x = self.norm(x)
        # x (batch_size, hidden_dim, num_nodes, input_seq_len)
        x = x.reshape(batch_size, residual_channels, num_nodes, self.residual_len)

        # x (batch_size, hidden_dim, num_nodes, input_seq_len)
        # skip (batch_size, skip_channels, num_nodes, input_seq_len)
        return x, skip_tcn


class LSDGNN(nn.Module):
    def __init__(self, **model_args):
        super().__init__()
        # 属性
        self.num_feat = model_args['num_feat']  # 1
        self.hidden_dim = model_args['hidden_dim']  # 32
        self.skip_channels = model_args['skip_channels']  # 64
        self.end_channels = model_args['end_channels']  # 128

        self.input_seq_len = model_args['input_seq_len']  # 12
        self.output_seq_len = model_args['output_seq_len']  # 12

        self.node_dim = model_args['node_dim']  # 40
        self.time_day_dim = model_args['time_day_dim']  # 16
        self.time_week_dim = model_args['time_week_dim']  # 7

        self.num_nodes = model_args['num_nodes']
        self.num_layers = model_args['num_layers']

        # node embeddings, (source, target)
        self.node_emb_s = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
        self.node_emb_t = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))

        # time embedding
        self.T_i_D_emb = nn.Parameter(torch.empty(288, self.time_day_dim))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, self.time_week_dim))

        # 邻接矩阵的构建 (预定义矩阵、自适应矩阵、动态矩阵)
        self.graph_constructor = GraphConstructor(**model_args)

        # start embedding layer
        self.embedding = nn.Linear(self.num_feat + 1, self.hidden_dim)

        residual_len = self.input_seq_len  # 12

        # start skip
        self.start_skip_conv = nn.Conv2d(self.hidden_dim, self.skip_channels,
                                         kernel_size=(1, residual_len), bias=True)

        # layer
        self.layers = nn.ModuleList()
        self.c_out_sets = [[16, 8, 8], [8, 16, 8], [8, 8, 16]]
        self.short_kernel_set = model_args['short_kernel_set']
        # 基础padding
        self.short_base_padding = 0
        self.long_base_padding = 4
        # 在short_paddings上加padding
        padding = 1
        for i in range(self.num_layers):
            if i < 1:
                c_out_set = self.c_out_sets[0]
            elif i < 2:
                c_out_set = self.c_out_sets[1]
            else:
                c_out_set = self.c_out_sets[2]

            self.layers.append(LSDGNLayer(c_out_set, residual_len,
                                          self.short_base_padding + padding, self.long_base_padding + padding,
                                          **model_args))

            # tcn卷积后剩下序列的长度
            residual_len = residual_len - self.short_kernel_set[-1] + 1 + self.short_base_padding + padding

        # end skip
        self.end_skip_conv = nn.Conv2d(self.hidden_dim, self.skip_channels,
                                       kernel_size=(1, residual_len), bias=True)

        # end conv, 64 -> 128 -> 12
        self.end_conv_1 = nn.Conv2d(self.skip_channels, self.end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(self.end_channels, self.output_seq_len, kernel_size=(1, 1), bias=True)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.node_emb_s)
        nn.init.xavier_uniform_(self.node_emb_t)
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)

    def _prepare_inputs(self, history_data):
        num_feat = self.num_feat
        # node embeddings
        node_emb_s = self.node_emb_s  # [N, d]
        node_emb_t = self.node_emb_t  # [N, d]
        # time slot embedding
        time_in_day_feat = self.T_i_D_emb[
            (history_data[:, :, :, num_feat] * 288).type(torch.LongTensor)]  # [B, L, N, d]
        day_in_week_feat = self.D_i_W_emb[(history_data[:, :, :, num_feat + 1]).type(torch.LongTensor)]  # [B, L, N, d]
        # traffic signals
        history_data = history_data[:, :, :, :num_feat + 1]

        return history_data, node_emb_s, node_emb_t, time_in_day_feat, day_in_week_feat

    def forward(self, history_data):
        """Feed forward

        Args:
            history_data (Tensor): history data with shape: [B, L, N, C]

        Returns:
            torch.Tensor: prediction data with shape: [B, L, N, 1]
        """

        # ========================== 预处理输入数据 ========================== #
        # 历史数据信息、节点嵌入、时间戳嵌入
        # (B, L, N, num_feat(1)), (N, node_dim), (B, L, N, time_day(week)_dim)
        history_data, node_emb_s, node_emb_t, time_in_day_feat, day_in_week_feat = self._prepare_inputs(
            history_data)

        # ========================= 构造邻接矩阵 ========================== #
        # [预定义矩阵、自适应矩阵、动态矩阵]
        # [(N, N), ...]
        adjs = self.graph_constructor(history_data=history_data,
                                      node_emb_s=node_emb_s,
                                      node_emb_t=node_emb_t,
                                      time_in_day_feat=time_in_day_feat,
                                      day_in_week_feat=day_in_week_feat)

        # Start embedding layer，将历史信息特征转换为 num_hidden(32)
        # history_data (B, L, N, num_feat) -> x (B, L, N, num_hidden)
        x = self.embedding(history_data)
        # x (B, num_hidden, N, L)
        x = x.transpose(1, 3)

        # x = nn.functional.pad(x, (self.receptive_field - x.size(-1), 0, 0, 0))

        # ========================= 进入ModelLayer ========================== #
        # skips (B, skip_channels, N, 1)
        skip = self.start_skip_conv(x)
        # ModelLayer
        for _, layer in enumerate(self.layers):
            # x = torch.relu(x)
            x, skip_tcn = layer(x, adjs)
            skip += skip_tcn
        skip += self.end_skip_conv(x)

        # x (B, skip_channels, N, 1)
        x = F.relu(skip)

        # x (B, end_channels, N, 1)
        x = F.relu(self.end_conv_1(x))

        # x (B, output_seq_len, N, 1)
        x = self.end_conv_2(x)

        return x
