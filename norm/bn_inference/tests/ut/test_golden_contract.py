# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU-only regression tests for the TTK golden contract (no device access)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

_PATH = Path(__file__).parents[1] / "assets" / "golden.py"
_SPEC = importlib.util.spec_from_file_location(
    "golden_contract_" + _PATH.parents[2].name, _PATH
)
_G = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_G)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_promote_does_not_round_to_declared_output(dtype):
    x = np.array([2048], dtype=dtype).reshape(1, 1, 1, 1)
    a = np.array([1], dtype=dtype)
    b = np.array([2], dtype=dtype)
    out = _G.BNInferenceKernelSpec.golden(
        x,
        a,
        b,
        a,
        mode=0,
        input_formats=("NCHW",),
        golden_mode="Promote",
        output_dtypes=("float16",),
    )[0]
    assert out.dtype == dtype
    assert out.item() == 4098


@pytest.mark.parametrize(
    "dtype,x,expected",
    [(torch.float16, 2048, 3074), (torch.bfloat16, 256, 386)],
)
def test_pre_folded_add_retains_fp32_until_final_cast(dtype, x, expected):
    inputs = [torch.tensor([value], dtype=dtype) for value in (x, 1, 1.5, 1)]
    inputs[0] = inputs[0].reshape(1, 1, 1, 1)
    out = _G._BNInferenceNative(mode=0, input_formats=("NCHW",))(*inputs)[0]
    assert out.dtype == dtype
    assert out.item() == expected


@pytest.mark.parametrize(
    "dtype,step,scale,offset,expected",
    [
        (torch.float16, 2**-10, 512, -513, 2**-11),
        (torch.bfloat16, 2**-7, 64, -65, 2**-8),
    ],
)
def test_pre_folded_affine_retains_base_product(dtype, step, scale, offset, expected):
    values = (1 + step, 0, 1 + step, 1, scale, offset)
    inputs = [torch.tensor([value], dtype=dtype) for value in values]
    inputs[0] = inputs[0].reshape(1, 1, 1, 1)
    out = _G._BNInferenceNative(mode=0, input_formats=("NCHW",))(*inputs)[0]
    assert out.dtype == dtype
    assert out.item() == expected


@pytest.mark.parametrize(
    "x_dtype,values,expected",
    [
        pytest.param(np.float32, (1, 0, 0.5, 2**-17), 2**-8, id="factor-overflow"),
        pytest.param(
            np.float32, (1, -128, 64, 2**-10), 512 + 2**-8, id="alpha-overflow"
        ),
        pytest.param(
            np.float32, (0, -1024, 64, 1, 1, 10000), 10128, id="affine-alpha-overflow"
        ),
        pytest.param(np.float32, (0, 0, 2**-24, 1, 32768, 1), 1, id="beta-overflow"),
        pytest.param(np.float16, (32768, -32, 64, 2**-10), 256, id="add-overflow"),
    ],
)
def test_folded_reference_avoids_artificial_fp16_overflow(x_dtype, values, expected):
    inputs = [np.array([value], dtype=np.float16) for value in values]
    inputs[0] = np.array([values[0]], dtype=x_dtype).reshape(1, 1, 1, 1)
    out = _G.BNInferenceKernelSpec.golden(
        *inputs, epsilon=0, mode=1, input_formats=("NCHW",), golden_mode="Enable"
    )[0]
    assert out.dtype == x_dtype
    assert np.isfinite(out).all()
    assert out.item() == expected


def test_promote_mixed_dtypes_follow_native_tensor_arithmetic():
    x = np.ones((1, 1, 1, 1), np.float32)
    a = np.array([2**-40], np.float64)
    b = np.array([1], np.float64)
    out = _G.BNInferenceKernelSpec.golden(
        x, a, b, b, mode=0, input_formats=("NCHW",), golden_mode="Promote"
    )[0]
    assert out.dtype == np.float64
    assert out.item() == 1 + 2**-40
