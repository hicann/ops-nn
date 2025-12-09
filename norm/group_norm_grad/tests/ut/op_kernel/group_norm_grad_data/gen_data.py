#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#/
import sys
import numpy as np
import torch

def write_file(inputs):
    for input_name in inputs:
        inputs[input_name].tofile("./input_"+input_name+".bin")
        print("input "+input_name+" has been gen in ./input_"+input_name+".bin")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    N = int(sys.argv[1])
    C = int(sys.argv[2])
    HXW = int(sys.argv[3])
    G = int(sys.argv[4])
    np_dtype = sys.argv[5]
    NXG = N * G
    C_G = C // G

    setup_seed(20)
    x = (np.random.randn(N, C, HXW)).astype(np_dtype)
    x_t = torch.from_numpy(x)
    x_t.requires_grad_()

    dy = (np.random.randn(N, C, HXW)).astype(np_dtype)
    Mean = np.zeros(shape=(NXG), dtype=np_dtype)
    Rstd = np.zeros(shape=(NXG), dtype=np_dtype)
    gamma = np.ones(shape=(C), dtype=np_dtype)
    for n_i in range(N):
        for g_i in range(G):
            Var = np.var(x[n_i, g_i*(C//G):(g_i+1)*(C//G)])
            Rstd[n_i * G + g_i] = 1/((Var+1e-5)**0.5)
            Mean[n_i * G + g_i] = np.mean(x[n_i, g_i*(C//G):(g_i+1)*(C//G)])
    inputs_npu = {"dy":dy.astype(np.float32) ,"mean": Mean.astype(np.float32), "rstd":Rstd.astype(np.float32), "x": x.astype(np.float32), "gamma": gamma.astype(np.float32)}
    print("dy_shape:",dy.shape)
    print("Mean_shape:",Mean.shape)
    print("Rstd_shape:",Rstd.shape)
    print("x_shape:",x.shape)
    print("gamma_shape:",gamma.shape)
    write_file(inputs_npu)