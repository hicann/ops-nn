#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
import math
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi


@register("executor_apply_fused_ema_adam")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        
        def correction_compute(bias_correction, beta1, beta2, step):
            bias_correction1 = 1.0
            bias_correction2 = 1.0
            if bias_correction == True:
                bias_correction1 = 1 - math.pow(beta1, step)
                bias_correction2 = 1 - math.pow(beta2, step)
            return bias_correction1, bias_correction2

        def fusedEmaAdam(grad, var, m, v, s, step, lr, emaDecay, beta1,
                         beta2, eps, mode, biasCorrection, weightDecay):
            beta1_correction, beta2_correction = correction_compute(biasCorrection, beta1, beta2, step.reshape(-1)[0])
            if mode == 0:
                grad_ = grad + weightDecay *var
            elif mode == 1:
                grad_ = grad
            m_ = beta1 * m + (1 - beta1) * grad_
            v_ = beta2 * v + (1 - beta2) * grad_ * grad_
            next_m = m_ / beta1_correction
            next_v = v_ / beta2_correction
            denom = torch.sqrt(next_v) + eps
            if mode == 0:
                update = next_m / denom
            elif mode == 1:
                update = next_m / denom + weightDecay * var
            var_ = var - lr* update
            s_ = emaDecay * s + (1 - emaDecay) * var_
            return var_, m_, v_, s_
        
        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"
            var_, m_, v_, s_ = fusedEmaAdam(input_data.kwargs["grad"],
                                            input_data.kwargs["var"],
                                            input_data.kwargs["m"],
                                            input_data.kwargs["v"],
                                            input_data.kwargs["s"],
                                            input_data.kwargs["step"],
                                            input_data.kwargs["lr"],
                                            input_data.kwargs["emaDecay"],
                                            input_data.kwargs["beta1"],
                                            input_data.kwargs["beta2"],
                                            input_data.kwargs["eps"],
                                            input_data.kwargs["mode"],
                                            input_data.kwargs["biasCorrection"],
                                            input_data.kwargs["weightDecay"]
                                            )
            return var_, m_, v_, s_

@register("aclnn_apply_fused_ema_adam")
class FusedAdamWV2AclnnApi(AclnnBaseApi):
    def after_call(self, output_packages):
        output = []
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        return output

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        input_args.pop()
        input_args.pop()
        input_args.pop()
        input_args.pop()
        output_packages[:] = [input_args[1], input_args[2], input_args[3], input_args[4]]
        return input_args, output_packages