# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import importlib

import pytest
import torch


def _npu_available():
    try:
        importlib.import_module("torch_npu")

        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


def _load_cann_ops_nn():
    importlib.import_module("cann_ops_nn")


# Locking each schema fragment separately is robust to namespace-prefix /
# whitespace formatting while still failing on any drift in op name, parameter
# name, parameter type, default value, or return-optionality.
_REQUIRED_SCHEMA_FRAGMENTS = (
    "swiglu_group_backward",
    "Tensor grad_output",
    "Tensor x",
    "Tensor? weight=None",
    "Tensor? y_origin=None",
    "Tensor? group_index=None",
    "float clamp_limit=0.0",
    "(Tensor, Tensor?)",
)


def test_op_registered_under_new_name():
    _load_cann_ops_nn()

    assert hasattr(torch.ops.cann_ops_nn, "swiglu_group_backward"), (
        "dispatcher must expose swiglu_group_backward after rename"
    )


def test_dispatcher_schema_matches_contract():
    _load_cann_ops_nn()

    schema = str(torch.ops.cann_ops_nn.swiglu_group_backward.default._schema)
    missing = [f for f in _REQUIRED_SCHEMA_FRAGMENTS if f not in schema]
    assert not missing, (
        f"schema missing required fragments {missing!r}; full schema: {schema}"
    )


def test_public_quant_export_preserves_backward_name():
    from cann_ops_nn.ops import quant

    assert "swiglu_group_backward" in quant.__all__
    assert not hasattr(torch.ops.cann_ops_nn, "swiglu_group_grad"), (
        "the operator-local package alias must not create a second dispatcher schema"
    )


def test_meta_without_weight_returns_none_grad_weight():
    _load_cann_ops_nn()

    grad_output = torch.empty((8, 128), dtype=torch.float16, device="meta")
    x = torch.empty((8, 256), dtype=torch.float16, device="meta")

    grad_x, grad_weight = torch.ops.cann_ops_nn.swiglu_group_backward(
        grad_output, x, clamp_limit=0.0
    )

    assert grad_x.shape == x.shape
    assert grad_x.dtype == grad_output.dtype
    assert grad_weight is None


def test_meta_with_weight_and_y_origin_returns_float32_grad_weight():
    _load_cann_ops_nn()

    grad_output = torch.empty((8, 128), dtype=torch.float16, device="meta")
    x = torch.empty((8, 256), dtype=torch.float16, device="meta")
    weight = torch.empty((8, 1), dtype=torch.float32, device="meta")
    y_origin = torch.empty((8, 128), dtype=torch.float16, device="meta")
    group_index = torch.empty((2,), dtype=torch.int64, device="meta")

    grad_x, grad_weight = torch.ops.cann_ops_nn.swiglu_group_backward(
        grad_output,
        x,
        weight=weight,
        y_origin=y_origin,
        group_index=group_index,
        clamp_limit=5.0,
    )

    assert grad_x.shape == x.shape
    assert grad_x.dtype == grad_output.dtype
    assert grad_weight.shape == weight.shape
    assert grad_weight.dtype == torch.float32


def test_meta_accepts_3d_input_without_optional_pair():
    _load_cann_ops_nn()

    grad_output = torch.empty((2, 4, 16), dtype=torch.bfloat16, device="meta")
    x = torch.empty((2, 4, 32), dtype=torch.bfloat16, device="meta")

    grad_x, grad_weight = torch.ops.cann_ops_nn.swiglu_group_backward(grad_output, x)

    assert grad_x.shape == x.shape
    assert grad_x.dtype == grad_output.dtype
    assert grad_weight is None


def test_meta_accepts_3d_input_with_all_optional_inputs():
    _load_cann_ops_nn()

    grad_output = torch.empty((2, 4, 16), dtype=torch.float32, device="meta")
    x = torch.empty((2, 4, 32), dtype=torch.float32, device="meta")
    weight = torch.empty((8,), dtype=torch.float32, device="meta")
    y_origin = torch.empty((2, 4, 16), dtype=torch.float32, device="meta")
    group_index = torch.empty((2,), dtype=torch.int64, device="meta")

    grad_x, grad_weight = torch.ops.cann_ops_nn.swiglu_group_backward(
        grad_output,
        x,
        weight=weight,
        y_origin=y_origin,
        group_index=group_index,
        clamp_limit=3.0,
    )

    assert grad_x.shape == x.shape
    assert grad_x.dtype == grad_output.dtype
    assert grad_weight.shape == weight.shape
    assert grad_weight.dtype == torch.float32


def test_meta_rejects_unsupported_dtype():
    _load_cann_ops_nn()

    grad_output = torch.empty((8, 16), dtype=torch.int32, device="meta")
    x = torch.empty((8, 32), dtype=torch.int32, device="meta")

    with pytest.raises(RuntimeError, match="grad_output must be"):
        torch.ops.cann_ops_nn.swiglu_group_backward(grad_output, x)


def test_meta_rejects_nan_clamp_limit():
    _load_cann_ops_nn()

    grad_output = torch.empty((8, 16), dtype=torch.float16, device="meta")
    x = torch.empty((8, 32), dtype=torch.float16, device="meta")

    with pytest.raises(RuntimeError, match="clamp_limit"):
        torch.ops.cann_ops_nn.swiglu_group_backward(
            grad_output, x, clamp_limit=float("nan")
        )


def test_meta_rejects_zero_hidden_size():
    _load_cann_ops_nn()

    grad_output = torch.empty((8, 0), dtype=torch.float16, device="meta")
    x = torch.empty((8, 0), dtype=torch.float16, device="meta")

    with pytest.raises(RuntimeError, match="greater than 0"):
        torch.ops.cann_ops_nn.swiglu_group_backward(grad_output, x)


def test_meta_rejects_empty_group_index():
    _load_cann_ops_nn()

    grad_output = torch.empty((8, 16), dtype=torch.float16, device="meta")
    x = torch.empty((8, 32), dtype=torch.float16, device="meta")
    group_index = torch.empty((0,), dtype=torch.int64, device="meta")

    with pytest.raises(RuntimeError, match="must not be empty"):
        torch.ops.cann_ops_nn.swiglu_group_backward(
            grad_output, x, group_index=group_index
        )


@pytest.mark.parametrize("missing", ["weight", "y_origin"])
def test_meta_rejects_unpaired_optional_inputs(missing):
    _load_cann_ops_nn()

    grad_output = torch.empty((8, 128), dtype=torch.float16, device="meta")
    x = torch.empty((8, 256), dtype=torch.float16, device="meta")
    weight = torch.empty((8, 1), dtype=torch.float32, device="meta")
    y_origin = torch.empty((8, 128), dtype=torch.float16, device="meta")

    kwargs = {"weight": weight, "y_origin": y_origin}
    kwargs.pop(missing)

    with pytest.raises(RuntimeError, match="weight and y_origin"):
        torch.ops.cann_ops_nn.swiglu_group_backward(grad_output, x, **kwargs)


@pytest.mark.skipif(not _npu_available(), reason="torch_npu/NPU is not available")
def test_eager_without_optional_pair_returns_none_grad_weight():
    _load_cann_ops_nn()
    importlib.import_module("torch_npu")

    device = "npu:0"
    grad_output = torch.randn((8, 16), dtype=torch.float16).to(device)
    x = torch.randn((8, 32), dtype=torch.float16).to(device)

    grad_x, grad_weight = torch.ops.cann_ops_nn.swiglu_group_backward(
        grad_output, x, clamp_limit=0.0
    )

    assert grad_x.shape == x.shape
    assert grad_x.dtype == grad_output.dtype
    assert grad_weight is None


@pytest.mark.skipif(not _npu_available(), reason="torch_npu/NPU is not available")
def test_eager_accepts_3d_input_with_all_optional_inputs():
    _load_cann_ops_nn()
    importlib.import_module("torch_npu")

    device = "npu:0"
    grad_output = torch.randn((2, 4, 16), dtype=torch.float16).to(device)
    x = torch.randn((2, 4, 32), dtype=torch.float16).to(device)
    weight = torch.randn((2, 4, 1), dtype=torch.float32).to(device)
    y_origin = torch.randn((2, 4, 16), dtype=torch.float16).to(device)
    group_index = torch.tensor([8], dtype=torch.int64, device=device)

    grad_x, grad_weight = torch.ops.cann_ops_nn.swiglu_group_backward(
        grad_output,
        x,
        weight=weight,
        y_origin=y_origin,
        group_index=group_index,
        clamp_limit=3.0,
    )

    assert grad_x.shape == x.shape
    assert grad_x.dtype == grad_output.dtype
    assert grad_weight.shape == weight.shape
    assert grad_weight.dtype == torch.float32
