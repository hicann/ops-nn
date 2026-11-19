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
import math
import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
import numpy

torch.serialization.add_safe_globals(
    [numpy.core.multiarray.scalar, numpy.dtype, numpy.ndarray]
)


def fused_adamw(
    params_list,
    grads_list,
    exp_avgs_list,
    exp_avg_sqs_list,
    max_exp_avg_sqs_list,
    steps_list,
    *,
    lr,
    beta1,
    beta2,
    weight_decay,
    eps,
    amsgrad,
    maximize,
    grad_scale=None,
    found_inf=None,
):
    """手写 fused AdamW，与 NPU kernel ComputeFP32 逻辑一致。

    NPU kernel 计算流程:
      1. found_inf 检查
      2. 梯度缩放: grad = grad / grad_scale
      3. 最大化: grad = -grad
      4. weight decay: param = param * (1 - lr * weight_decay)
      5. 更新一阶动量: exp_avg = beta1 * exp_avg + (1-beta1) * grad
      6. 更新二阶动量: exp_avg_sq = beta2 * exp_avg_sq + (1-beta2) * grad^2
      7. bias correction: step_size = lr / (1 - beta1^step)
                          bias_correction2_sqrt = sqrt(1 - beta2^step)
      8. denom = sqrt(exp_avg_sq) / bias_correction2_sqrt + eps
         (amsgrad: max_exp_avg_sq = max(max_exp_avg_sq, exp_avg_sq), 用 max_exp_avg_sq 代替)
      9. param = param - step_size * exp_avg / denom
    """
    # found_inf 检查: 如果 found_inf == 1.0，跳过所有更新
    if found_inf is not None and found_inf.numel() > 0:
        if found_inf.item() == 1.0:
            return

    # 检查数据类型，fp16/bf16 需要 cast 到 fp32 计算
    compute_dtype = params_list[0].dtype
    need_cast = compute_dtype in (torch.float16, torch.bfloat16)
    if need_cast:
        orig_params_list = params_list
        orig_exp_avgs_list = exp_avgs_list
        orig_exp_avg_sqs_list = exp_avg_sqs_list
        if amsgrad:
            orig_max_exp_avg_sqs_list = max_exp_avg_sqs_list

        params_list = [p.float() for p in params_list]
        grads_list = [g.float() for g in grads_list]
        exp_avgs_list = [e.float() for e in exp_avgs_list]
        exp_avg_sqs_list = [e.float() for e in exp_avg_sqs_list]
        if amsgrad:
            max_exp_avg_sqs_list = [m.float() for m in max_exp_avg_sqs_list]

    for i in range(len(params_list)):
        param = params_list[i]
        grad = grads_list[i].clone()
        exp_avg = exp_avgs_list[i]
        exp_avg_sq = exp_avg_sqs_list[i]
        step_val = steps_list[i].item()

        # Step 1: 梯度缩放
        if grad_scale is not None and grad_scale.numel() > 0:
            inv_grad_scale = 1.0 / grad_scale.item()
            grad.mul_(inv_grad_scale)

        # Step 2: 最大化处理
        if maximize:
            grad.neg_()

        # Step 3: weight decay
        if weight_decay != 0.0:
            decay_factor = 1.0 - lr * weight_decay
            param.mul_(decay_factor)

        # Step 4: 更新一阶动量 exp_avg = beta1 * exp_avg + (1-beta1) * grad
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

        # Step 5: 更新二阶动量 exp_avg_sq = beta2 * exp_avg_sq + (1-beta2) * grad^2
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        step_val += 1
        # 防止 step <= 0 导致除零 NaN
        if step_val <= 0:
            step_val = 1
        # Step 6: bias correction (step 已由 executor +1)
        bias_correction1 = 1.0 - beta1**step_val
        bias_correction2 = 1.0 - beta2**step_val
        step_size = lr / bias_correction1

        bias_correction2_sqrt = math.sqrt(bias_correction2)

        # Step 7: 分母 denom
        if amsgrad:
            max_exp_avg_sq = max_exp_avg_sqs_list[i]
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            denom = max_exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(eps)
        else:
            denom = exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(eps)
        param.addcdiv_(exp_avg, denom, value=-step_size)

    # 计算结果 cast 回原始 dtype
    if need_cast:
        for i in range(len(params_list)):
            orig_params_list[i].copy_(params_list[i].to(compute_dtype))
            orig_exp_avgs_list[i].copy_(exp_avgs_list[i].to(compute_dtype))
            orig_exp_avg_sqs_list[i].copy_(exp_avg_sqs_list[i].to(compute_dtype))
            if amsgrad:
                orig_max_exp_avg_sqs_list[i].copy_(
                    max_exp_avg_sqs_list[i].to(compute_dtype)
                )


@register("function_aclnnFusedAdamw")
class aclnnFusedAdamwExecutor(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        torch.serialization.add_safe_globals(
            [numpy.core.multiarray.scalar, numpy.dtype, numpy.ndarray]
        )
        params_list = input_data.kwargs["params_ref"]
        params_list_1 = params_list[0]
        print(params_list_1.device)
        grads_list = input_data.kwargs["grads"]
        exp_avgs_list = input_data.kwargs["exp_avgs_ref"]
        exp_avg_sqs_list = input_data.kwargs["exp_avg_sqs_ref"]
        max_exp_avg_sqs_list = input_data.kwargs["max_exp_avg_sqs_ref"]
        steps_list = input_data.kwargs["state_steps"]
        gradScaleOptional = input_data.kwargs["gradScaleOptional"]
        foundInfOptional = input_data.kwargs["foundInfOptional"]

        lr = input_data.kwargs["lr"]
        beta1 = input_data.kwargs["beta1"]
        beta2 = input_data.kwargs["beta2"]
        weight_decay = input_data.kwargs["weight_decay"]
        eps = input_data.kwargs["eps"]
        amsgrad_in = input_data.kwargs["amsgrad"]
        maximize_in = input_data.kwargs["maximize"]
        if not amsgrad_in:
            max_exp_avg_sqs_list = []
        fused_adamw(
            params_list,
            grads_list,
            exp_avgs_list,
            exp_avg_sqs_list,
            max_exp_avg_sqs_list,
            steps_list,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
            amsgrad=amsgrad_in,
            maximize=maximize_in,
            grad_scale=gradScaleOptional,
            found_inf=foundInfOptional,
        )
        if amsgrad_in:
            return [params_list, exp_avgs_list, exp_avg_sqs_list, max_exp_avg_sqs_list]
        else:
            return [params_list, exp_avgs_list, exp_avg_sqs_list]


@register("aclnnfunction_aclnnFusedAdamw")
class aclnnfunctionExecutor(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        amsgrad_in = input_data.kwargs["amsgrad"]
        num = len(self.task_result.output_info_list)
        ams_num = 3
        if amsgrad_in:
            ams_num = 4
        tmp2 = []
        for i in range(0, num, int(num // ams_num)):
            tmp = []
            for j in range(int(num // ams_num)):
                tmp.append(self.task_result.output_info_list[i + j])
            tmp2.append(tmp)
        orgin_output_info_list = self.task_result.output_info_list.copy()
        self.task_result.output_info_list = tmp2
        input_args, output_packages = super().init_by_input_data(input_data)
        input_args.pop()
        input_args.pop()
        input_args.pop()
        if amsgrad_in:
            input_args.pop()
            output_packages[:] = [
                input_args[0],
                input_args[2],
                input_args[3],
                input_args[4],
            ]
        else:
            output_packages[:] = [input_args[0], input_args[2], input_args[3]]
        self.task_result.output_info_list = orgin_output_info_list

        return input_args, output_packages

    def after_call(self, output_packages):
        return super().after_call(output_packages)
