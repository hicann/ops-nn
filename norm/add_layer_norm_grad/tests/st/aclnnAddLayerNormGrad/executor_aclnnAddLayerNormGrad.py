#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensorStruct, AclFormat
import copy

@register("add_layer_norm_grad")
class AclnnAddLayerNormBackward(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):

        origin_dtype = input_data.kwargs["dy"].dtype
        dy = input_data.kwargs["dy"]
        x1 = input_data.kwargs["x1"]
        x2 = input_data.kwargs["x2"]
        mean = input_data.kwargs["mean"]
        rstd = input_data.kwargs["rstd"]
        gamma = input_data.kwargs["gamma"]
        dsumOptional = input_data.kwargs["dsumOptional"]
        normalizedShape = [gamma.shape[0]]

        if self.device == 'cpu' :
            output = mycall(dy, x1, x2, mean, rstd, gamma, dsumOptional)
            
            return (output[0].to(origin_dtype),output[1],output[2])

        elif self.device == 'gpu' :
            bias = torch.zeros_like(gamma).type(origin_dtype).cuda()
            x = x1 + x2
            a,b,c = torch.ops.aten.native_layer_norm_backward(dy.cuda(), x.cuda(), normalizedShape, mean.cuda(), rstd.cuda(), gamma.cuda(), bias.cuda(), [True,True,True])
            a = a + dsumOptional
            return (a.cuda(),b.cuda(),c.cuda())
        elif self.device == 'npu':
            import torch_npu
            torch.use_deterministic_algorithms(True)
            torch.npu.set_compile_mode(jit_compile=False)
            a,b,c,d = torch_npu.npu_add_layer_norm_backward(dy.npu(), x1.npu(),x2.npu(), rstd.npu(), mean.npu(), gamma.npu(),dsumOptional.npu())
            return (a,c,d)


def mycall(dy, x1, x2, mean, rstd, gamma, dsumOptional):
    dy = dy.to(torch.float32)
    x1 = x1.to(torch.float32)
    x2 = x2.to(torch.float32)
    mean = mean.to(torch.float32).expand_as(dy)
    rstd = rstd.to(torch.float32).expand_as(dy)
    gamma = gamma.to(torch.float32).expand_as(dy)
    dsumOptional = dsumOptional.to(torch.float32)
    x1 = x1 + x2
    x2 = dy * gamma
    outdx = x1 - mean
    a3 = rstd * rstd *rstd
    x1 = outdx * a3
    x1 = x2 * x1
    meanx = torch.sum(copy.deepcopy(x1),dim = -1, keepdim = True).expand_as(dy)
    meanx = meanx / x1.shape[-1]
    meanx = -1 * meanx
    x1 = outdx * rstd
    x1 = x1 * dy
    outdgamma = torch.sum(copy.deepcopy(x1).view(-1,x1.shape[-1]),dim = 0)
    outdbeta = torch.sum(copy.deepcopy(dy).view(-1,x1.shape[-1]),dim = 0)
    x1 = outdx * meanx
    outdx = x2 * rstd
    dy = outdx * -1
    x2 = x1 + outdx
    meany = torch.sum(copy.deepcopy(dy),dim = -1, keepdim = True).expand_as(dy)
    meany = meany / dy.shape[-1]
    outdx = x2 + meany
    outdx = outdx + dsumOptional
    return (outdx, outdgamma, outdbeta)



