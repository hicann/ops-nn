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
#
# ATK dual-executor for the experimental ApplyFtrl operator.
#
#   * golden_apply_ftrl  (BaseApi)      -> json "api_type"        : high-precision (fp64) FTRL reference
#   * aclnn_apply_ftrl    (AclnnBaseApi) -> json "aclnn_api_type"  : calls the deployed aclnnApplyFtrl
#
# ApplyFtrl has NO single torch / numpy callable (it is a TF/GE fused optimizer op), so the golden is a
# hand-written fp32-spec FTRL formula evaluated in fp64 (high-precision truth), per
# experimental/optim/apply_ftrl/docs/spec.yaml math_semantics.formula.
#
# var / accum / linear are in-place (Ref) tensors: they are BOTH inputs (passed to GetWorkspaceSize) and
# outputs (written back in place). The aclnn executor keeps them in input_args AND places them in
# output_packages so the three updated tensors are read back and compared individually.
#
# aclnn L2 signature (must match the deployed header for -cp):
#   aclnnApplyFtrlGetWorkspaceSize(aclTensor* varRef, aclTensor* accumRef, aclTensor* linearRef,
#       const aclTensor* grad, const aclTensor* lr, const aclTensor* l1, const aclTensor* l2,
#       const aclTensor* lrPower, bool useLocking, uint64_t* workspaceSize, aclOpExecutor** executor)
# ----------------------------------------------------------------------------

import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi


_APPLY_FTRL_SIGNATURE = (
    "aclnnApplyFtrlGetWorkspaceSize("
    "aclTensor* varRef, aclTensor* accumRef, aclTensor* linearRef, "
    "const aclTensor* grad, const aclTensor* lr, const aclTensor* l1, "
    "const aclTensor* l2, const aclTensor* lrPower, bool useLocking, "
    "uint64_t* workspaceSize, aclOpExecutor** executor)"
)


def _ftrl_golden_fp64(var, accum, linear, grad, lr, l1, l2, lr_power):
    """Compute the FTRL-proximal update in fp64 (high-precision truth).

    Mirrors docs/spec.yaml math_semantics.formula (RHS values are pre-update). Scalars
    (lr/l1/l2/lr_power, shape [] or [1]) broadcast against the elementwise tensors.
    Returns (var_out, accum_out, linear_out) as fp64 tensors.
    """
    var = var.cpu().double()
    accum = accum.cpu().double()
    linear = linear.cpu().double()
    grad = grad.cpu().double()
    lr = lr.cpu().double()
    l1 = l1.cpu().double()
    l2 = l2.cpu().double()
    lr_power = lr_power.cpu().double()

    accum_new = accum + grad * grad
    accum_new_pow = accum_new ** (-lr_power)
    accum_pow = accum ** (-lr_power)
    linear_t = linear + grad - ((accum_new_pow - accum_pow) / lr) * var
    quadratic = accum_new_pow / lr + 2.0 * l2
    x_res = torch.sign(linear_t) * l1 - linear_t
    var_out = torch.where(torch.abs(linear_t) > l1, x_res / quadratic, torch.zeros_like(linear_t))
    accum_out = accum_new
    linear_out = linear_t
    return var_out, accum_out, linear_out


@register("golden_apply_ftrl")
class ApplyFtrlGolden(BaseApi):
    """High-precision (fp64) FTRL reference. Returns the three in-place outputs in spec order."""

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        k = input_data.kwargs
        var_out, accum_out, linear_out = _ftrl_golden_fp64(
            k["varRef"], k["accumRef"], k["linearRef"], k["grad"],
            k["lr"], k["l1"], k["l2"], k["lrPower"],
        )
        # Keep high-precision (fp64) truth; ATK picks the per-dtype threshold from the NPU output dtype
        # and casts both sides to a common precision for the relative-error comparison.
        return var_out.contiguous(), accum_out.contiguous(), linear_out.contiguous()


@register("aclnn_apply_ftrl")
class ApplyFtrlAclnn(AclnnBaseApi):
    """Calls the deployed aclnnApplyFtrl. var/accum/linear are in-place (inputs[0,1,2] == outputs)."""

    def __call__(self):
        super().__call__()

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        # The base auto-allocates output tensors for every non-const aclTensor* (var/accum/linear) and
        # APPENDS them to the tail of input_args (and to output_packages). For a normal op the final
        # call is input_args + [workspaceSize, executor]. ApplyFtrl is in-place: var/accum/linear are
        # the FIRST three args (input_args[0,1,2], already carrying the generated input data) AND the
        # three outputs -- there are NO separate output parameters in the 11-arg signature
        # (var,accum,linear,grad,lr,l1,l2,lrPower,useLocking,workspaceSize,executor).
        #
        # So drop the freshly-allocated appended outputs (they hold no input data and would make the
        # call pass 14 args instead of 11), and read the three in-place tensors back after the kernel.
        n_out = len(output_packages)
        self._inplace_refs = [input_args[0], input_args[1], input_args[2]]
        if n_out:
            del input_args[-n_out:]
        output_packages[:] = []
        return input_args, output_packages

    def after_call(self, output_packages):
        # var/accum/linear were updated in place; convert the stashed in/out refs (post-update) to torch.
        return [self.acl_tensor_to_torch(p) for p in self._inplace_refs]

    def get_cpp_func_signature_type(self):
        return _APPLY_FTRL_SIGNATURE
