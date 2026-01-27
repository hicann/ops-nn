#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import torch
import torch_npu
import numpy as np
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat

# Tanh constants
COEFFICIENT_A1 = -0.07135481627260025
COEFFICIENT_A2 = -1.5957691216057308
COEFFICIENT_A3 = -0.21406444881780075
COEFFICIENT_A4 = -1.5957691216057307

# Erf constants
TH_MAX = 3.92
TH_MIN = -3.92
COEFFICIENT_1 = 0.70710678118  # 1 / sqrt(2)
COEFFICIENT_2 = 0.3989422804   # 1 / sqrt(2π)
COEFFICIENT_3 = 0.053443748819
COEFFICIENT_4 = 7.5517016694
COEFFICIENT_5 = 101.62808918
COEFFICIENT_6 = 1393.8061484
COEFFICIENT_7 = 5063.7915060
COEFFICIENT_8 = 29639.384698
COEFFICIENT_9 = 31.212858877
COEFFICIENT_10 = 398.56963806
COEFFICIENT_11 = 3023.1248150
COEFFICIENT_12 = 13243.365831
COEFFICIENT_13 = 26267.224157

@register("aclnn_geglu_v3_backward")
class AscendGeGluV3BackwardApi(BaseApi):

    @staticmethod
    def self_gelu(x):
        alpha = np.sqrt(2.0 / np.pi)
        beta = 0.044715
        temp_x = alpha * (x + beta * x**3)
        gelu_y = 0.5 * x * (1.0 + np.tanh(temp_x))
        return gelu_y

    @staticmethod
    def gelu_grad_tanh(dy, x):
        g1 = 1.0 / (np.exp(x * (x * x * COEFFICIENT_A1 + COEFFICIENT_A2)) + 1)
        g2 = x * x * COEFFICIENT_A3 + COEFFICIENT_A4
        return (x * (g1 - 1) * g2 + 1) * g1 * dy

    @staticmethod
    def erf_compute_with_simplified_formula(input_x):
        input_x = np.clip(input_x, TH_MIN, TH_MAX)
        x2 = input_x * input_x

        numer = x2 * COEFFICIENT_3 + COEFFICIENT_4
        numer = numer * x2 + COEFFICIENT_5
        numer = numer * x2 + COEFFICIENT_6
        numer = numer * x2 + COEFFICIENT_7
        numer = numer * x2 + COEFFICIENT_8
        numer = numer * input_x

        denom = (x2 + COEFFICIENT_9) * x2
        denom = (denom + COEFFICIENT_10) * x2
        denom = (denom + COEFFICIENT_11) * x2
        denom = (denom + COEFFICIENT_12) * x2
        denom = denom + COEFFICIENT_13

        return numer / denom

    @staticmethod
    def cdf(x):
        erf_input = x * COEFFICIENT_1
        erf_res = AscendGeGluV3BackwardApi.erf_compute_with_simplified_formula(erf_input)
        return erf_res * 0.5 + 0.5

    @staticmethod
    def pdf(x):
        return np.exp(-0.5 * x * x) * COEFFICIENT_2

    @staticmethod
    def gelu_grad_erf(dy, x):
        cdf_res = AscendGeGluV3BackwardApi.cdf(x)
        pdf_res = AscendGeGluV3BackwardApi.pdf(x)
        return (cdf_res + x * pdf_res) * dy

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        dy = input_data.kwargs["gradOutput"]
        x = input_data.kwargs["self"]
        gelu = input_data.kwargs["gelu"]
        dim = input_data.kwargs["dim"]
        approximate = input_data.kwargs["approximate"]

        # Convert to NumPy
        dy = dy.cpu().numpy()
        x = x.cpu().numpy()
        gelu = gelu.cpu().numpy()

        x1, x2 = np.split(x, 2, axis=dim)
        x1 = dy * x1

        if approximate == 1:
            dx2 = self.gelu_grad_tanh(x1, x2)
        else:
            dx2 = self.gelu_grad_erf(x1, x2)

        dx1 = dy * gelu
        x_grad = np.concatenate((dx1, dx2), axis=dim)

        return torch.from_numpy(x_grad)
