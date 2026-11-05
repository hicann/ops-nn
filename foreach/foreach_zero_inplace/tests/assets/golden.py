#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""ForeachZeroInplace unified TTK TestSpec for CPU, GPU and NPU cross-check."""

import numpy as np
import torch


__spec__ = {
    "foreach_zero_inplace": "ForeachZeroInplaceKernelSpec",
    "aclnnForeachZeroInplace": "ForeachZeroInplaceAclnnSpec",
    "torch._foreach_zero_": "TorchForeachZeroInplaceSpec",
}


_TORCH_L2 = {
    "float16": {"standard": "cross_check", "level": "L2"},
    "bfloat16": {"standard": "cross_check", "level": "L2"},
    "float32": {"standard": "cross_check", "level": "L2"},
    "int16": {"standard": "cross_check", "level": "L2"},
    "int32": {"standard": "cross_check", "level": "L2"},
}


def _as_list(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _clone_torch_tensor(tensor):
    return tensor.detach().clone(memory_format=torch.preserve_format)


def _torch_foreach_zero_post_state(x):
    """Run the native foreach API on clones and return the post-state list."""
    tensors = [_clone_torch_tensor(tensor) for tensor in _as_list(x)]
    if tensors:
        torch._foreach_zero_(tensors)
    return tensors


def _numpy_to_torch_tensor(tensor):
    """Bridge NumPy input to torch while retaining the bfloat16 bit pattern."""
    if isinstance(tensor, torch.Tensor):
        return tensor
    array = np.asarray(tensor)
    if "bfloat16" in str(array.dtype):
        return torch.from_numpy(np.ascontiguousarray(array).view(np.int16)).view(
            torch.bfloat16
        )
    return torch.from_numpy(np.ascontiguousarray(array))


def _torch_to_numpy_tensor(tensor):
    """Bridge torch output to NumPy while retaining the bfloat16 bit pattern."""
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        import ml_dtypes

        return tensor.view(torch.int16).numpy().view(ml_dtypes.bfloat16)
    return tensor.numpy()


def foreach_zero_inplace_golden(x, **kwargs):
    """Kernel/GEIR CPU golden; NumPy is only the input/output bridge."""
    del kwargs
    tensors = [_numpy_to_torch_tensor(tensor) for tensor in _as_list(x)]
    return [
        _torch_to_numpy_tensor(tensor)
        for tensor in _torch_foreach_zero_post_state(tensors)
    ]


def aclnn_foreach_zero_inplace_golden(x, **kwargs):
    """ACLNN CPU golden; inputs are CPU torch tensors."""
    del kwargs
    return _torch_foreach_zero_post_state(x)


def torch_foreach_zero_inplace(x):
    """PyTorch golden and competitor shared by TTK and the offline GPU runner."""
    return _torch_foreach_zero_post_state(x)


class TorchForeachZeroInplace:
    def __call__(self, *args, **kwargs):
        del kwargs
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            inputs = args[0]
        else:
            inputs = list(args)
        return torch_foreach_zero_inplace(inputs)


class _TpAclnnForeachZero:
    """ACLNN third-party adapter with the public parameter name and empty-list semantics."""

    def __call__(self, x, **kwargs):
        del kwargs
        return _torch_foreach_zero_post_state(x)


# Compatibility with callers that imported the previous provider name.
TorchForeachZero = TorchForeachZeroInplace


class TorchForeachZeroGraph(torch.nn.Module):
    """Graph-mode wrapper for the native Torch foreach operation."""

    def forward(self, x):
        return _torch_foreach_zero_post_state(x)


class ForeachZeroInplaceKernelSpec:
    golden = staticmethod(foreach_zero_inplace_golden)
    third_party = {"torch": "torch._foreach_zero_"}
    tolerance = _TORCH_L2


class ForeachZeroInplaceAclnnSpec:
    golden = staticmethod(aclnn_foreach_zero_inplace_golden)
    third_party = {"torch": _TpAclnnForeachZero}
    tolerance = _TORCH_L2


class TorchForeachZeroInplaceSpec:
    golden = staticmethod(torch_foreach_zero_inplace)
    third_party = {"torch": "torch._foreach_zero_"}
    torch_graph = TorchForeachZeroGraph
    tolerance = _TORCH_L2
