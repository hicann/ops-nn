#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
import numpy as np


def sigmoid(x):
    shape = x.shape
    x = x.flatten()
    l = len(x)
    y = []
    for i in range(l):
        if x[i] >= 0:
            y.append(1.0 / (1 + np.exp(-x[i])))
        else:
            y.append(np.exp(x[i]) / (1 + np.exp(-x[i])))

    y = np.array(y).astype(np.float32)
    y = y.reshape(shape)
    return y


def expand_4_tensorlist(tensorlist):
    ex_tensorlist = []
    for tensor in tensorlist:
        ex_tensorlist.append(np.expand_dims(tensor, 0))
    return ex_tensorlist


def expand_4_tensorlist_and_concat(tensorlist, ex_tensorlist):
    index = 0
    for tensor, xtensor in zip(tensorlist, ex_tensorlist):
        xtensor = np.concatenate((xtensor, np.expand_dims(tensor, 0)), axis=0)
        ex_tensorlist[index] = xtensor
        index += 1
    return ex_tensorlist


def single_lstm_cell(x1, h0, c0, weight, bias, forget_bias=0.0):
    xh = np.concatenate((x1, h0), axis=1)
    mat_res = np.matmul(xh, weight) + bias
    res_i, res_j, res_f, res_o, = np.split(mat_res, 4, axis=1)
    res_f = res_f + forget_bias

    i1 = sigmoid(res_i)
    j1 = np.tanh(res_j)
    f1 = sigmoid(res_f)
    o1 = sigmoid(res_o)
    c1 = c0 * f1 * j1 * i1
    tanhc1 = np.tanh(c1)
    h1 = tanhc1 * o1
    return  i1, j1, f1, o1, c1, tanhc1, h1


def dynamic_lstm(shape_a, shape_b, t_size):

    input_dim = shape_a[1] - shape_b[1] // 4
    output_dim = shape_b[1] // 4

    data_range = [-1, 1]

    xh0 = np.random.uniform(*data_range, shape_a).astype(np.float32)
    xh0.tofile("x_ori.bin")

    weight = np.random.uniform(*data_range, shape_b).astype(np.float32)
    weight0, weight1 = np.split(weight, [input_dim], axis=0)
    weight.tofile("w.bin")
    weight0.tofile("w_half.bin")

    bias = np.random.uniform(*data_range, shape_b[1:2]).astype(np.float32)
    bias.tofile("b.bin")

    x1, h0 = np.split(xh0, [input_dim], axis=1)
    h0 = np.random.uniform(*data_range, h0.shape).astype(np.float32)
    h0.tofile("inth.bin")
    x1.tofile("x_t1.bin")

    c0 = np.random.uniform(*data_range, [shape_a[0], output_dim]).astype(np.float32)
    c0.tofile("intc.bin")

    x_t = np.random.uniform(*data_range, [t_size - 1, shape_a[0], input_dim]).astype(np.float32)
    x_t.tofile("x_tn.bin")

    seq = np.random.randint(0, 2, [t_size, shape_a[0], output_dim]).astype(np.float32)
    seq.tofile("seq.bin")

    if t_size > 1:
        x1 = x1.reshape(1, shape_a[0], input_dim)
        x_all = np.concatenate((x1, x_t), axis=0)
        x1 = x1.reshape(shape_a[0], input_dim)
        x_all.tofile("x.bin")
    else:
        x1.tofile("x.bin")

    x_1 = x1
    h_0 = h0
    c_0 = c0

    for i in range(t_size):
        i_1, j_1, f_1, o_1, c_1, tanhc_1, h_1 = single_lstm_cell(x_1, h_0, c_0, weight, bias, forget_bias=0.0)

        if i == 0:
            input_x, output_h, output_c, output_i, output_j, output_f, output_o, output_tnahc = \
                expand_4_tensorlist([x_1, h_1, c_1, i_1, j_1, f_1, o_1, tanhc_1])
        else:
            input_x, output_h, output_c, output_i, output_j, output_f, output_o, output_tnahc = \
                expand_4_tensorlist_and_concat([x_1, h_1, c_1, i_1, j_1, f_1, o_1, tanhc_1],
                                               [input_x,
                                                output_h, output_c, output_i, output_j, output_f, output_o,
                                                output_tnahc])
        if i < t_size - 1:
            x_1 = x_t[i:i+1, :]
            x_1 = x_1.reshape(shape_a[0], input_dim)
            h_0 = h_1
            c_0 = c_1
    return input_x, output_h, output_c, output_i, output_j, output_f, output_o, output_tnahc, weight, bias, h0, c0


if __name__ == '__main__':

    rnn_time, rnn_batch, rnn_input_size, rnn_hidden_size = [int(p) for p in sys.argv[1:]]

    rnn_size = rnn_input_size + rnn_hidden_size
    input_x, output_h, output_c, output_i, output_j, output_f, output_o, output_tnahc, weight, bias, h0, c0 = \
        dynamic_lstm([rnn_batch, rnn_size], [rnn_size, 4 * rnn_hidden_size], rnn_time)
    output_h.tofile("out.bin")
    output_c.tofile("out_c.bin")
