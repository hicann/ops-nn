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
from dataclasses import dataclass
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi


@dataclass
class SgdParams:
    param: any
    grad: any
    momentum_buffer: any
    lr: float
    momentum: float
    dampening: float
    weight_decay: float
    nesterov: bool
    maximize: bool
    is_first_step: bool


def sgd_step(
    param,
    grad,
    momentum_buf,
    grad_scale=None,
    lr=1.0,
    momentum=0.0,
    dampening=0.0,
    weight_decay=0.0,
    nesterov=False,
    maximize=False,
    is_first_step=False,
):
    outPram = None
    outGrad = None
    outMomentum = None
    if grad_scale is not None and grad_scale.numel() > 0:
        grad = grad / (grad_scale[0])
        outGrad = grad

    if maximize:
        grad = grad * -1.0

    if weight_decay != 0:
        grad = grad + weight_decay * param

    buf = None
    if momentum_buf is not None:
        if is_first_step:
            buf = grad
        else:
            buf = momentum_buf * momentum + (1 - dampening) * grad
        outMomentum = buf

        if nesterov:
            grad = grad + momentum * buf
        else:
            grad = buf

    param = param - lr * grad
    outPram = param

    return outPram, outGrad, outMomentum


@register("function_torch._fused_sgd_")
class aclnnFusedSgdExecutor(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # print("==================================================================")
        params_list = input_data.kwargs["paramsRef"]
        grads_list = input_data.kwargs["gradsRef"]
        momentum_list = input_data.kwargs["momentumBufferListOptionalRef"]

        weight_decay = input_data.kwargs["weightDecay"]
        momentum = input_data.kwargs["momentum"]
        lr = input_data.kwargs["lr"]
        dampening = input_data.kwargs["dampening"]
        nesterov = input_data.kwargs["nesterov"]
        maximize = input_data.kwargs["maximize"]
        is_first_step = input_data.kwargs["isFirstStep"]

        grad_scale = input_data.kwargs.get("gradScaleOptional", None)

        if self.device == "cpu":
            out_params_list = []
            out_grads_list = []
            out_momentum_list = []
            # print("momentum_list = ", momentum_list)
            # print("momentum_list[0].numel() = ", momentum_list[0].numel())

            for i in range(len(params_list)):
                param = params_list[i]
                grad = grads_list[i]
                momentum_buf = None
                if momentum_list is not None and momentum_list[0].numel() != 0:
                    momentum_buf = momentum_list[i]
                oridtype = param.dtype
                if oridtype == torch.float16 or oridtype == torch.bfloat16:
                    param = param.to(torch.float32)
                    grad = grad.to(torch.float32)
                    if momentum_list is not None and momentum_list[0].numel() != 0:
                        momentum_buf = momentum_buf.to(torch.float32)

                out_p, out_g, out_m = sgd_step(
                    param,
                    grad,
                    momentum_buf,
                    grad_scale,
                    lr,
                    momentum,
                    dampening,
                    weight_decay,
                    nesterov,
                    maximize,
                    is_first_step,
                )
                if oridtype == torch.float16 or oridtype == torch.bfloat16:
                    out_p = out_p.to(oridtype)
                    out_g = out_g.to(oridtype)
                    if momentum_list is not None and momentum_list[0].numel() != 0:
                        out_m = out_m.to(oridtype)

                out_params_list.append(out_p)
                out_grads_list.append(out_g)
                if momentum_list is not None and momentum_list[0].numel() != 0:
                    out_momentum_list.append(out_m)

            if momentum_list is not None and momentum_list[0].numel() != 0:
                return [out_params_list, out_grads_list, out_momentum_list]
            else:
                return [out_params_list, out_grads_list]
        elif self.device == "npu":
            torch._fused_sgd_(
                params_list,
                grads_list,
                momentum_list,
                weight_decay=weight_decay,
                momentum=momentum,
                lr=lr,
                dampening=dampening,
                nesterov=nesterov,
                maximize=maximize,
                is_first_step=is_first_step,
                grad_scale=grad_scale,
            )
            if momentum_list is not None and momentum_list[0].numel() != 0:
                return [params_list, grads_list, momentum_list]
            else:
                return [params_list, grads_list]


@register("aclnnfunction_aclnnFusedSgd")
class aclnnfunctionExecutor(AclnnBaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        num = len(self.task_result.output_info_list)
        tmp2 = []
        for i in range(0, num, int(num // 3)):
            tmp = []
            for j in range(int(num // 3)):
                tmp.append(self.task_result.output_info_list[i + j])
            tmp2.append(tmp)
        self.task_result.output_info_list = tmp2
        input_args, output_packages = super().init_by_input_data(input_data)
        input_args.pop()
        input_args.pop()
        input_args.pop()
        output_packages[:] = [input_args[0], input_args[1], input_args[2]]

        return input_args, output_packages

    def after_call(self, output_packages):
        return super().after_call(output_packages)
