#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2020-2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Golden TestSpec for the npu_alloc_float_status operator.

The operator has no inputs and no attributes. It produces a fixed output
tensor of shape (8,) with dtype float32 where every element is zero
(8 elements = NPU float-overflow status flags, 8 x 4B = 32B). It is a
non-computational constant-fill operator used for float-status allocation
in loss scaling / overflow detection scenarios.

The expected output is bit-exact, so the float32 output is compared with
binary_equal (bitwise equality, rtol=0 / atol=0).

Registration keys:
  - "npu_alloc_float_status"  : kernel/GEIR op name (snake_case)
  - "NPUAllocFloatStatus"     : GE operator type name (CamelCase)
  - "aclnnNPUAllocFloatStatus": aclnn interface name designed in the SE
    analysis but never delivered (the build excludes aclnn); kept
    registered for name continuity with historical test assets
  - "npu_device.gen_npu_ops.npu_alloc_float_status": official 2.x TF
    plugin-op api_name key, consumed by the tf e2e guard lane; its
    golden is computed with the TensorFlow API (tf.zeros) so the tf
    lane compares against a tf-native reference
"""

__spec__ = {
    "npu_alloc_float_status": "NpuAllocFloatStatusTestSpec",
    "NPUAllocFloatStatus": "NpuAllocFloatStatusTestSpec",
    "aclnnNPUAllocFloatStatus": "AclnnNPUAllocFloatStatusTestSpec",
    "npu_device.gen_npu_ops.npu_alloc_float_status": "TfNpuAllocFloatStatusTestSpec",
}

# Bitwise comparison contract for the constant-zero output:
# standard=binary_equal with rtol=0 / atol=0 (equivalent to the
# bitwise_equal metric named on the spec side).
_TOLERANCE = {
    "float32": {"standard": "binary_equal"},
}


def _torch_compute():
    """Shared computation core: produce 8 float32 zeros on caller device.

    This is a bit-exact non-computational operator (constant zero fill).
    The output is always identical regardless of any input or Promote
    mode. torch is imported lazily inside the function so that loading
    this module never imports a competitor framework at module scope.
    """
    import torch

    return torch.zeros(8, dtype=torch.float32)


class NpuAllocFloatStatusTestSpec:
    """Kernel/GEIR TestSpec for npu_alloc_float_status.

    The operator has no tensor inputs. Output is fixed: shape=(8,),
    dtype=float32, all zeros. Binary-equal comparison is required because
    the output is bit-exact.
    """

    def golden(**kwargs):
        """Numpy golden: return [zeros(8, float32)] as a numpy array.

        Accepts **kwargs for framework metadata (dtype, shape, format,
        etc.) but ignores all of them since the output is fixed.
        """
        result = _torch_compute()
        return [result.detach().cpu().numpy()]

    tolerance = _TOLERANCE


class AclnnNPUAllocFloatStatusTestSpec:
    """ACLNN TestSpec for the designed-but-undelivered aclnn interface.

    The aclnn interface has no input tensors and one output tensor
    ``data`` (shape=(8,), dtype=float32). Golden receives torch.Tensor
    inputs per ACLNN convention, but this operator needs none.
    """

    def golden(**kwargs):
        """Return [torch.zeros(8, float32)] as the expected output."""
        result = _torch_compute()
        return [result]

    tolerance = _TOLERANCE


class TfNpuAllocFloatStatusTestSpec:
    """TF TestSpec for the official plugin-op api_name key.

    The tf e2e lane dispatches on the npu_device plugin-op api name, so
    its golden is computed with the TensorFlow API. The operator is a
    constant zero fill: output shape=(8,), dtype=float32, all zeros,
    compared with binary_equal (bit-exact). tensorflow is imported
    lazily inside the function so that loading this module never
    imports a framework at module scope.
    """

    def golden(**kwargs):
        """Return [tf.zeros([8], float32)] converted to a numpy array.

        Accepts **kwargs for framework metadata (dtype, shape, format,
        etc.) but ignores all of them since the output is fixed. The
        numpy conversion is lossless (float32 to float32 memory copy),
        so the returned array is bit-identical to the constant-zero
        output.
        """
        import tensorflow as tf

        result = tf.zeros([8], dtype=tf.float32)
        return [result.numpy()]

    tolerance = _TOLERANCE
