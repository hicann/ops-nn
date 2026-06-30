#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import torch
import torch_npu
import ascend_ops_nn
import logging

logging.basicConfig(level=logging.INFO)
epsilon = 1e-5
batch_size = 2
seq_len = 10
hidden_dim = 64
supported_dtypes = {torch.bfloat16, torch.float16}

for dtype in supported_dtypes:
    logging.info(f"start testing dtype : {'BF16' if dtype == torch.bfloat16 else 'HALF'}")
    input = torch.randn(batch_size, seq_len, hidden_dim).to(dtype)
    gamma = torch.randn(hidden_dim).to(dtype)
    beta = torch.randn(hidden_dim).to(dtype)
    output = torch.empty_like(input)

    input_squared = torch.square(input)
    mean_squared = input_squared.mean(dim = -1, keepdim = True)
    normalized = input / torch.sqrt(mean_squared + epsilon)
    output_cpu = normalized * gamma + beta

    input_npu = input.npu()
    gamma_npu = gamma.npu()
    beta_npu = beta.npu()
    output_npu = output.npu()
    torch.ops.ascend_ops_nn.layernorm_stride(40, hidden_dim, epsilon, input_npu, gamma_npu, beta_npu, output_npu)

    abs_error = torch.abs(output_npu.cpu() - output_cpu)
    rel_error = abs_error / (torch.abs(output_npu.cpu()) + epsilon)

    logging.info(f"input tensor: \n{input}")
    logging.info(f"cpu result tensor: \n{output_cpu}")
    logging.info(f"npu result tensor: \n{output_npu}")
    logging.info(f"max absolute error: {abs_error.max().item():.8f}")
    logging.info(f"average absolute error: {abs_error.mean().item():.8f}")
    logging.info(f"max relative error: {rel_error.max().item():.8f}")
    logging.info(f"average relative error: {rel_error.mean().item():.8f}")