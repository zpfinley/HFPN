import numpy as np
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    source: sample_size_1 * feature_size
    target: sample_size_2 * feature_size
    kernel_mul: parameter to control the bandwidth
    kernel_num: number of kernel
    fix_sigma: fixed bandwidth
    return: combination of multiple kernels
    """

    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    total0 = total.unsqueeze(0).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), \
                                       int(total.size(0)), \
                                       int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)


    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]


    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    '''
    :param source: [T,dim]
    :param target: [T,dim]
    :return:
    '''

    kernels = guassian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)


    n = int(source.size()[0])
    m = int(target.size()[0])

    XX = kernels[:n, :n]
    YY = kernels[n:, n:]
    XY = kernels[:n, n:]
    YX = kernels[n:, :n]

    XX = torch.div(XX, n * n).sum(dim=1).view(1, -1)  # K_ss矩阵，Source<->Source
    XY = torch.div(XY, -n * m).sum(dim=1).view(1, -1)  # K_st矩阵，Source<->Target

    YX = torch.div(YX, -m * n).sum(dim=1).view(1, -1)  # K_ts矩阵,Target<->Source
    YY = torch.div(YY, m * m).sum(dim=1).view(1, -1)  # K_tt矩阵,Target<->Target

    loss = (XX + XY).sum() + (YX + YY).sum()

    return loss


def mmd_loss(source, target, segment_flag_pos, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    '''
    :param source: [bz,T,dim], audio
    :param target: [bz,T,dim], visual
    :return:
    '''

    loss_b = 0
    bz = source.size(0)
    count = 0
    for i in range(0, bz):
        segment_flag_pos_i = segment_flag_pos[i, :] # [10]
        even_num_i = int(segment_flag_pos_i.sum().item())
        if even_num_i != 0:
            count += 1
            source_even_i = []
            target_even_i = []
            for j in range(0, 10):
                if segment_flag_pos_i[j] == 1:
                    source_even_i.append(source[i, j, :])
                    target_even_i.append(target[i, j, :])
            source_even_i = torch.stack(source_even_i, dim=0)
            target_even_i = torch.stack(target_even_i, dim=0)
            # print(source_even_i[1,:].requires_grad)
            mmd_loss_i = mmd(source_even_i, target_even_i)
        else:
            mmd_loss_i = 0.0

        loss_b = loss_b + mmd_loss_i
    loss = loss_b / count

    return loss
        