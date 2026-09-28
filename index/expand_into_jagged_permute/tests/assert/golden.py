#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""FBGEMM reference for ExpandIntoJaggedPermute.

Requires NumPy, PyTorch and a compatible FBGEMM package (fbgemm_gpu).
Importing fbgemm_gpu registers the native torch.ops.fbgemm operators.
The array golden uses CPU dispatch; the third-party callable uses the
provided tensors' device. Inputs must satisfy FBGEMM's operator contract.
"""

import fbgemm_gpu  # noqa: F401 -- registers the native FBGEMM operators
import numpy as np
import torch

__golden__ = {
    "kernel": {"expand_into_jagged_permute": "expand_into_jagged_permute_golden"},
}
__spec__ = {"expand_into_jagged_permute": "ExpandIntoJaggedPermuteSpec"}


def expand_into_jagged_permute_golden(
    permute, input_offsets, output_offsets, *, output_size, **kwargs
):
    """Accept NumPy inputs in OpDef order and return the native FBGEMM result."""
    permute_tensor = torch.from_numpy(np.ascontiguousarray(permute))
    input_offsets_tensor = torch.from_numpy(np.ascontiguousarray(input_offsets))
    output_offsets_tensor = torch.from_numpy(np.ascontiguousarray(output_offsets))
    result = torch.ops.fbgemm.expand_into_jagged_permute(
        permute_tensor, input_offsets_tensor, output_offsets_tensor, int(output_size)
    )
    return result.numpy()


class ExpandIntoJaggedPermuteThirdParty:
    """Call FBGEMM with prepared contiguous tensors of matching dtype/device."""

    def __init__(self, *, output_size, **kwargs):
        self.output_size = int(output_size)

    def __call__(self, permute, input_offsets, output_offsets, **kwargs):
        return torch.ops.fbgemm.expand_into_jagged_permute(
            permute, input_offsets, output_offsets, self.output_size
        )


class ExpandIntoJaggedPermuteSpec:
    golden = staticmethod(expand_into_jagged_permute_golden)
    third_party = {"torch": ExpandIntoJaggedPermuteThirdParty}
    tolerance = {"int32": {"standard": "binary_equal"}}
