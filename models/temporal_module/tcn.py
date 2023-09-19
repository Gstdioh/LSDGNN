import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SLTCN(nn.Module):
    def __init__(self, c_in, c_out_set, short_padding, long_padding, seq_len, model_args):
        super(SLTCN, self).__init__()
        # short_tcn
        self.short_kernel_set = model_args['short_kernel_set']  # [1, 2, 3]
        self.short_padding = short_padding
        self.short_conv = TCNInception(c_in, c_out_set, self.short_kernel_set)

        # long_tcn
        self.long_kernel_set = model_args['long_kernel_set']  # [1, 6, 7]
        self.long_padding = long_padding
        self.long_conv = TCNInception(c_in, c_out_set, self.long_kernel_set)

        self.conv_channels = sum(c_out_set)  # 32, 单个tcn的输出维度

        self.seq_len = seq_len  # 当前序列的长度
        self.residual_len = self.seq_len - self.short_kernel_set[-1] + 1 + self.short_padding  # tcn卷积后剩下序列的长度

        # rate add
        self.short_rate = nn.Parameter(torch.ones([1, self.conv_channels, 1, self.residual_len]) / 2)
        self.long_rate = nn.Parameter(torch.ones([1, self.conv_channels, 1, self.residual_len]) / 2)

        # mx
        self.short_mx = nn.Parameter(torch.empty([1, self.residual_len, self.conv_channels, self.conv_channels]))
        self.long_mx = nn.Parameter(torch.empty([1, self.residual_len, self.conv_channels, self.conv_channels]))
        self.bias_mx = nn.Parameter(torch.empty([1, self.residual_len, 1, 1]))

        self.bias_end = nn.Parameter(torch.empty([1, self.conv_channels, 1, self.residual_len]))

        torch.nn.init.kaiming_normal_(self.short_mx)
        torch.nn.init.kaiming_normal_(self.long_mx)
        torch.nn.init.constant_(self.bias_mx, val=0.0)
        torch.nn.init.constant_(self.bias_end, val=0.0)

    def forward(self, x):
        # x (batch_size, hidden_dim, num_nodes, input_seq_len)

        # short_x (batch_size, conv_channels, num_nodes, input_seq_len)
        short_x = self.short_conv(x, self.short_padding)

        # long_x (batch_size, conv_channels, num_nodes, input_seq_len)
        long_x = self.long_conv(x, self.long_padding)

        # mx
        # short_x (B, D, N, L) -> (B, L, N, D); short_mx (1, L, D, D); -> (B, L, N, D)
        short_tmp = torch.matmul(short_x.permute(0, 3, 2, 1), self.short_mx)
        long_tmp = torch.matmul(long_x.permute(0, 3, 2, 1), self.long_mx)

        # gated fusion
        # z (B, L, N, D)
        z = torch.sigmoid(short_tmp + long_tmp + self.bias_mx)
        # z (B, D, N, L)
        z = z.permute(0, 3, 2, 1)

        # (B, D, N, L)
        x = (z + self.short_rate) * short_x + (1 - z + self.long_rate) * long_x

        return x


class TCNInception(nn.Module):
    def __init__(self, c_in, c_out_set, kernel_set):
        """

        Args:
            c_in: 输入的通道
            c_out_set: 如[16, 8, 8], 每个卷积核的输出的通道, 最后进行cat
            kernel_set: inception中的卷积核大小集合，从小到大，如[1, 6, 7]
        """
        super(TCNInception, self).__init__()
        self.kernel_set = kernel_set
        self.padding = kernel_set[-1] - 1

        self.pre_tconv = nn.ModuleList()  # 对[6, 7] 例，在1×6卷积前面进行一个1×1卷积
        self.tconv = nn.ModuleList()  # 对[1, 6, 7]

        for i, kernel_size in enumerate(self.kernel_set):
            # if i != 0:
            #     self.pre_tconv.append(nn.Conv2d(c_in, c_in, (1, 1)))
            self.tconv.append(nn.Conv2d(c_in, c_out_set[i], (1, kernel_size)))

    def forward(self, x, padding=None):
        # x (batch_size, input_dim, num_nodes, seq_len)
        #                                  上下       左右
        # 在序列左边添加 padding
        # nn.functional.pad的添加顺序是(左, 右, 上, 下)
        if padding is None:
            padding = self.padding
        x = nn.functional.pad(x, (padding, 0, 0, 0))

        output = []

        # 进行卷积操作
        for i in range(len(self.kernel_set)):
            tmp = x
            # if i != 0:
            #     tmp = self.pre_tconv[i - 1](tmp)  # 1×1
            output.append(self.tconv[i](tmp))  # 1×6

        # 最后一个维度进行对齐
        for i in range(len(self.kernel_set)):
            output[i] = output[i][..., -output[-1].size(3):]

        # cat
        output = torch.cat(output, dim=1)

        return output
