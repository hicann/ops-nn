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
# ATK dual-executor for the experimental ApplyGradientDescent operator.
#
#   * golden_apply_gradient_descent (BaseApi)     -> json "api_type":       high-precision (fp64) golden
#   * aclnn_apply_gradient_descent  (AclnnBaseApi) -> json "aclnn_api_type": calls deployed aclnnApplyGradientDescent
#
# Semantics: var = var - alpha * delta, computed elementwise. The op is IN-PLACE: `var` is BOTH an input
# (passed to GetWorkspaceSize) AND the output (written back into the caller's var buffer via ViewCopy).
# alpha is a 1-element scalar tensor; delta has the same shape & dtype as var.
#
# aclnn L2 signature (must match the deployed header for -cp AND drives the base's output allocation):
#   aclnnApplyGradientDescentGetWorkspaceSize(aclTensor* var, const aclTensor* alpha,
#       const aclTensor* delta, uint64_t* workspaceSize, aclOpExecutor** executor)
#
# Only `var` is a non-const aclTensor* -> the base auto-allocates ONE output for it and appends it to the
# tail of input_args. Being in-place, var is ALREADY input_args[0] (carrying the generated data) AND the
# output, so we drop the freshly-appended output tensor and read input_args[0] back after the kernel.
# ----------------------------------------------------------------------------


from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi


_AGD_SIGNATURE = (
    "aclnnApplyGradientDescentGetWorkspaceSize("
    "aclTensor* var, const aclTensor* alpha, const aclTensor* delta, "
    "uint64_t* workspaceSize, aclOpExecutor** executor)"
)


def _agd_golden_fp64(var, alpha, delta):
    """var - alpha * delta in fp64 (high-precision truth).

    The kernel casts bf16/fp16 inputs up to fp32, computes, then casts the result back. We evaluate the
    same formula in fp64 from the dtype-rounded inputs ATK generated, keeping the high-precision result;
    ATK picks the per-dtype threshold from the NPU output dtype and casts both sides to a common
    precision for the relative-error comparison. alpha (shape [1] or []) is a scalar.
    """
    var = var.cpu().double()
    delta = delta.cpu().double()
    alpha = alpha.cpu().double().flatten()[0]
    return var - alpha * delta


@register("golden_apply_gradient_descent")
class ApplyGradientDescentGolden(BaseApi):
    """High-precision (fp64) reference for the single in-place output `var`."""

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        k = input_data.kwargs
        var_out = _agd_golden_fp64(k["var"], k["alpha"], k["delta"])
        return (var_out.contiguous(),)


@register("aclnn_apply_gradient_descent")
class ApplyGradientDescentAclnn(AclnnBaseApi):
    """Calls the deployed aclnnApplyGradientDescent. `var` is in-place (input_args[0] == the output)."""

    def __call__(self):
        super().__call__()

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        # The base appended a fresh output tensor for the sole non-const aclTensor* `var`. The 5-arg
        # in-place signature has NO separate output param (var,alpha,delta,workspaceSize,executor), so
        # drop the appended output and read the in-place `var` (input_args[0]) back after the kernel.
        n_out = len(output_packages)
        self._inplace_refs = [input_args[0]]
        if n_out:
            del input_args[-n_out:]
        output_packages[:] = []
        return input_args, output_packages

    def after_call(self, output_packages):
        # `var` was updated in place; convert the stashed in/out ref (post-update) to torch.
        return [self.acl_tensor_to_torch(p) for p in self._inplace_refs]

    def get_cpp_func_signature_type(self):
        return _AGD_SIGNATURE
