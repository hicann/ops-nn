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

_PATH = Path(__file__).parents[1] / "assets" / "golden.py"
_SPEC = importlib.util.spec_from_file_location(
    "golden_contract_" + _PATH.parents[2].name, _PATH
)
_G = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_G)


@pytest.mark.parametrize(
    "dtype", [np.float16, np.float32, np.float64, np.complex64, np.complex128]
)
def test_precision_is_supplied_by_ttk(dtype):
    x = np.array([[1.0], [3.0]], dtype=dtype)
    v = np.array([[2.0**-24]], dtype=dtype)
    before = x.copy()
    out = _G.InplaceSubKernelSpec.golden(
        x, np.array([0], np.int32), v, golden_mode="Promote"
    )[0]
    expected = x.copy()
    expected[0] = x[0] - v[0]
    assert out.dtype == dtype
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(x, before)


@pytest.mark.parametrize(
    "dtype",
    [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64],
)
def test_integer_wraparound(dtype):
    info = np.iinfo(dtype)
    x = np.array([[info.max], [info.min]], dtype=dtype)
    v = np.array([[1], [1]], dtype=dtype)
    out = _G.InplaceSubKernelSpec.golden(x, np.array([0, 1], np.int32), v)[0]
    with np.errstate(over="ignore"):
        expected = x - v
    assert out.dtype == dtype
    np.testing.assert_array_equal(out, expected)


def test_bfloat16_container_and_wrapped_indices():
    from ml_dtypes import bfloat16

    x = np.array([[1], [3]], dtype=bfloat16)
    v = np.array([[2]], dtype=bfloat16)
    out = _G.InplaceSubKernelSpec.golden(x, np.array([-1], np.int32), v)[0]
    assert out.dtype == np.dtype(bfloat16)
    assert float(out[1, 0]) == 3 - 2


@pytest.mark.parametrize(
    "shapes",
    [
        ((2, 3), (), (1, 3)),
        ((), (1,), (1,)),
        ((2, 3), (3,), (3, 3)),
        ((2, 3), (1,), (1, 4)),
    ],
)
def test_invalid_shapes_are_not_repaired_by_input_hook(shapes):
    x, idx, v = (
        np.zeros(shape, dtype=dtype)
        for shape, dtype in zip(shapes, [np.float32, np.int32, np.float32])
    )
    out = _G.InplaceSubKernelSpec.customize_inputs(
        x, idx, v, testcase_name="exception_row"
    )
    for actual, original in zip(out, [x, idx, v]):
        assert actual.shape == original.shape
        np.testing.assert_array_equal(actual, original)


def test_empty_input_and_empty_indices():
    x = np.empty((0, 3), dtype=np.float32)
    out = _G.InplaceSubKernelSpec.golden(x, np.empty(0, np.int32), x.copy())[0]
    assert out.shape == x.shape


def test_tolerance_covers_registered_types():
    required = {
        "float16",
        "float32",
        "bfloat16",
        "complex32",
        "complex64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
    assert required <= _G.InplaceSubKernelSpec.tolerance.keys()


def test_e2e_uses_native_ttk_promote_path():
    assert _G.InplaceSubTfSpec.golden is None
    assert "e2e" not in _G.__golden__
    assert set(_G.InplaceSubTfSpec.third_party) == {"torch"}
    assert set(_G.InplaceSubKernelSpec.third_party) == {"torch"}


@pytest.mark.parametrize("dtype_name", ["complex32", "complex64"])
@pytest.mark.parametrize(
    "update_values",
    [
        [complex(np.inf, 2), complex(3, -np.inf)],
        [complex(np.inf, np.inf), complex(-np.inf, -np.inf)],
        [complex(np.nan, 2), complex(3, np.nan)],
    ],
)
def test_third_party_complex_special_values_keep_components_independent(
    dtype_name, update_values
):
    import torch

    dtype = getattr(torch, dtype_name)
    x = torch.tensor([[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=dtype)
    v = torch.tensor([update_values], dtype=dtype)
    before = x.clone()
    output = _G._InplaceSubTorchCompose()(x, torch.tensor([-1], dtype=torch.int32), v)[
        0
    ]
    expected = torch.view_as_real(before).clone()
    expected[1] -= torch.view_as_real(v[0])
    assert output.dtype == dtype
    torch.testing.assert_close(
        torch.view_as_real(output), expected, rtol=0, atol=0, equal_nan=True
    )
    torch.testing.assert_close(x, before, rtol=0, atol=0)
