#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Parameter validation tests for the add_rms_norm_dynamic_quant torch API.

Device-independent (no NPU hardware required):

  1. Type interception at the raw Python entry
     (``cann_ops_nn.ops.add_rms_norm_dynamic_quant``, the @impl function):
     tensors must be torch.Tensor (beta/x3 may be None); scalars must be
     Python-native. Rejected: bool for int slots, torch.dtype, and numpy
     scalars — including np.float64/np.str_, which subclass the Python
     builtins and would slip through a plain isinstance() gate. Accepted:
     enum.IntEnum for int slots (a Python-native int subclass). The gate runs
     before JIT load, so these tests neither compile anything nor touch a device.

  2. Value validation in the Meta kernel (torch.ops + meta tensors):
      x1/x2/gamma/beta dtype rules, round_mode values per dst_type, scale_alg
      range, dst_type domain — exercised through the dispatcher with meta
      tensors. These mirror the csrc TORCH_CHECKs, which sit behind the NPU
      device checks and therefore need Ascend 950 hardware to fire (covered by
      test_torch_extension.py negative cases).

  3. Framework behavior lock: the top-level ``cann_ops_nn.<op>`` attribute
      resolves (package __getattr__) to the dispatcher OpOverloadPacket, where
      torch converts torch.dtype to its ScalarType ordinal (float8_e5m2 -> 23)
      before any op code runs. Type interception cannot fire on that path; the
      dst_type domain check is the only guard. These tests pin that behavior.

Usage:
    cd <ops-nn repo root>
    pip install torch_extension/dist/cann_ops_nn-*.whl
    pytest norm/add_rms_norm_dynamic_mx_quant/tests/ut/torch_extension/ \
        test_torch_extension_param_validation.py -v
"""

import enum
import types

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401
import cann_ops_nn  # noqa: F401

# Raw Python entry (bypasses the dispatcher). NOTE: the top-level
# cann_ops_nn.add_rms_norm_dynamic_quant resolves to torch.ops (dispatcher),
# NOT this function — see test_toplevel_entry_is_dispatcher_packet.
_RAW_OP = cann_ops_nn.ops.add_rms_norm_dynamic_quant
_OP = torch.ops.cann_ops_nn.add_rms_norm_dynamic_quant


def _cpu_inputs(shape=(4, 64), dtype=torch.float16):
    x1 = torch.randn(shape, dtype=dtype)
    x2 = torch.randn(shape, dtype=dtype)
    gamma = torch.ones(shape[-1], dtype=dtype)
    return x1, x2, gamma


def _meta_inputs(shape=(4, 64), dtype=torch.float16, gamma_dtype=None):
    x1 = torch.randn(shape, dtype=dtype, device="meta")
    x2 = torch.randn(shape, dtype=dtype, device="meta")
    gamma = torch.ones(shape[-1], dtype=gamma_dtype or dtype, device="meta")
    return x1, x2, gamma


# ---------------------------------------------------------------------------
# 1. Scalar type interception (raw Python entry)
# ---------------------------------------------------------------------------

INVALID_SCALARS = [
    # (param, bad value, expected message fragment)
    ("dst_type", torch.float8_e5m2, "dst_type must be a Python int"),
    ("dst_type", torch.int4, "dst_type must be a Python int"),
    ("dst_type", np.int64(36), "dst_type must be a Python int"),
    ("dst_type", np.int32(36), "dst_type must be a Python int"),
    ("dst_type", True, "dst_type must be a Python int"),
    ("dst_type", 36.0, "dst_type must be a Python int"),
    ("dst_type", "36", "dst_type must be a Python int"),
    ("scale_alg", np.int64(0), "scale_alg must be a Python int"),
    ("scale_alg", False, "scale_alg must be a Python int"),
    ("scale_alg", 0.0, "scale_alg must be a Python int"),
    ("round_mode", 1, "round_mode must be a Python str"),
    ("round_mode", b"rint", "round_mode must be a Python str"),
    # np.str_ subclasses Python str — a plain isinstance() gate would let it
    # through, so the gate must use an exact type check here.
    ("round_mode", np.str_("rint"), "round_mode must be a Python str"),
    ("output_rstd", 1, "output_rstd must be a Python bool"),
    ("output_rstd", "true", "output_rstd must be a Python bool"),
    # np.bool_ is NOT a Python bool subclass, and in numpy 2.x its type name
    # renders as "bool" — the message must be module-qualified to stay readable.
    ("output_rstd", np.bool_(True), "output_rstd must be a Python bool"),
    ("epsilon", "1e-6", "epsilon must be a Python int or float"),
    ("epsilon", True, "epsilon must be a Python int or float"),
    ("epsilon", np.float32(1e-6), "epsilon must be a Python int or float"),
    # np.float64 DOES subclass Python float — same exact-type-check requirement.
    ("epsilon", np.float64(1e-6), "epsilon must be a Python int or float"),
]


class _DstType(enum.IntEnum):
    """IntEnum is a Python-native int subclass and must stay accepted."""

    FP8_E4M3FN = 36


INVALID_TENSORS = [
    # (param, bad value, expected message fragment)
    ("x1", "not a tensor", "x1 must be a torch.Tensor"),
    ("x1", [1, 2, 3], "x1 must be a torch.Tensor"),
    ("x1", np.zeros((4, 64), dtype=np.float16), "x1 must be a torch.Tensor"),
    ("x2", 1.0, "x2 must be a torch.Tensor"),
    ("gamma", None, "gamma must be a torch.Tensor"),
    ("beta", 1.0, "beta must be a torch.Tensor or None"),
    ("x3", "oops", "x3 must be a torch.Tensor or None"),
]


@pytest.mark.parametrize(
    "param,bad_value,err_frag",
    INVALID_TENSORS,
    ids=[
        f"{param}={type(bad_value).__name__}" for param, bad_value, _ in INVALID_TENSORS
    ],
)
def test_tensor_type_interception(param, bad_value, err_frag):
    """Non-tensor inputs must be rejected by the gate with a named parameter,
    instead of falling through to pybind11's unreadable argument dump."""
    x1, x2, gamma = _cpu_inputs()
    overrides = {param: bad_value}
    args = (
        overrides.pop("x1", x1),
        overrides.pop("x2", x2),
        overrides.pop("gamma", gamma),
    )
    with pytest.raises(TypeError, match=err_frag):
        _RAW_OP(*args, **overrides)


@pytest.mark.parametrize(
    "param,bad_value,err_frag",
    INVALID_SCALARS,
    ids=[f"{param}={bad_value!r}" for param, bad_value, _ in INVALID_SCALARS],
)
def test_scalar_type_interception(param, bad_value, err_frag):
    """Non-Python-native scalars must be rejected with a TypeError naming the
    parameter, before JIT load / device access."""
    x1, x2, gamma = _cpu_inputs()
    with pytest.raises(TypeError, match=err_frag):
        _RAW_OP(x1, x2, gamma, **{param: bad_value})


def test_valid_scalars_pass_type_gate(monkeypatch):
    """Valid Python-native scalars must get past the gate and reach JIT load
    (mocked with a sentinel so the test stays device- and build-free)."""
    import importlib

    op_submod = importlib.import_module(
        "cann_ops_nn.ops.norm.add_rms_norm_dynamic_quant.add_rms_norm_dynamic_quant"
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("PASSED_TYPE_GATE")

    monkeypatch.setattr(op_submod.add_rms_norm_dynamic_quant_builder, "load", _boom)
    x1, x2, gamma = _cpu_inputs()
    with pytest.raises(RuntimeError, match="PASSED_TYPE_GATE"):
        _RAW_OP(
            x1,
            x2,
            gamma,
            beta=None,
            x3=None,
            epsilon=1e-6,
            scale_alg=0,
            round_mode="rint",
            dst_type=36,
            output_rstd=True,
        )


def test_intenum_passes_type_gate(monkeypatch):
    """enum.IntEnum is a Python-native int subclass and must stay accepted —
    the int slots deliberately use isinstance() rather than an exact type check
    so this ergonomic pattern keeps working."""
    import importlib

    op_submod = importlib.import_module(
        "cann_ops_nn.ops.norm.add_rms_norm_dynamic_quant.add_rms_norm_dynamic_quant"
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("PASSED_TYPE_GATE")

    monkeypatch.setattr(op_submod.add_rms_norm_dynamic_quant_builder, "load", _boom)
    x1, x2, gamma = _cpu_inputs()
    with pytest.raises(RuntimeError, match="PASSED_TYPE_GATE"):
        _RAW_OP(x1, x2, gamma, dst_type=_DstType.FP8_E4M3FN)


def test_raw_entry_is_python_function():
    """cann_ops_nn.ops.<op> must stay the raw @impl function (not the
    dispatcher OpOverloadPacket) — the type gate depends on it."""
    assert isinstance(_RAW_OP, types.FunctionType)


# ---------------------------------------------------------------------------
# 2. Value validation (Meta kernel via dispatcher, meta tensors)
# ---------------------------------------------------------------------------


def test_meta_rejects_x2_dtype_mismatch():
    """Regression: Meta used to validate x2.shape but never x2.dtype, so
    torch.compile accepted x1=fp16 / x2=fp32 while eager (csrc) rejected it."""
    x1, _, gamma = _meta_inputs()
    x2 = torch.randn(4, 64, dtype=torch.float32, device="meta")
    with pytest.raises(RuntimeError, match="x2 dtype must match x1 dtype"):
        _OP(x1, x2, gamma)


def test_meta_rejects_invalid_gamma_dtype():
    x1, x2, _ = _meta_inputs()
    gamma = torch.ones(64, dtype=torch.float64, device="meta")
    with pytest.raises(RuntimeError, match="gamma dtype must match"):
        _OP(x1, x2, gamma)


def test_meta_accepts_fp32_gamma():
    x1, x2, _ = _meta_inputs()
    gamma = torch.ones(64, dtype=torch.float32, device="meta")
    y, x_out, mxscale, rstd = _OP(x1, x2, gamma)
    assert y.dtype == torch.uint8  # default dst_type=40 (FP4), packed as uint8
    assert x_out.dtype == torch.float16
    assert list(y.shape) == [4, 32]


def test_meta_rejects_beta_dtype_mismatch():
    x1, x2, gamma = _meta_inputs()  # gamma fp16
    beta = torch.zeros(64, dtype=torch.float32, device="meta")
    with pytest.raises(RuntimeError, match="beta dtype must match gamma dtype"):
        _OP(x1, x2, gamma, beta=beta)


def test_meta_rejects_invalid_round_mode_for_fp8():
    x1, x2, gamma = _meta_inputs()
    with pytest.raises(RuntimeError, match="round_mode must be 'rint' for FP8"):
        _OP(x1, x2, gamma, dst_type=36, round_mode="floor")


def test_meta_accepts_all_round_modes_for_fp4():
    x1, x2, gamma = _meta_inputs()
    for round_mode in ("rint", "floor", "round"):
        y, _, _, _ = _OP(x1, x2, gamma, dst_type=40, round_mode=round_mode)
        assert y.dtype == torch.uint8


def test_meta_rejects_scale_alg_out_of_range():
    x1, x2, gamma = _meta_inputs()
    with pytest.raises(RuntimeError, match=r"scale_alg must be 0 \(OCP\) or 1"):
        _OP(x1, x2, gamma, scale_alg=2)


def test_meta_rejects_scale_alg1_for_fp4():
    x1, x2, gamma = _meta_inputs()
    with pytest.raises(RuntimeError, match=r"scale_alg must be 0 \(OCP\) for FP4"):
        _OP(x1, x2, gamma, dst_type=40, scale_alg=1)


def test_meta_rejects_invalid_dst_type_value():
    x1, x2, gamma = _meta_inputs()
    with pytest.raises(RuntimeError, match="invalid dst_type 23"):
        _OP(x1, x2, gamma, dst_type=23)


def test_meta_valid_full_call():
    x1, x2, gamma = _meta_inputs(shape=(4, 128))
    y, x_out, mxscale, rstd = _OP(
        x1,
        x2,
        gamma,
        dst_type=36,
        epsilon=1e-6,
        scale_alg=1,
        round_mode="rint",
        output_rstd=True,
    )
    assert list(y.shape) == [4, 128]
    assert y.dtype == torch.float8_e4m3fn
    assert list(x_out.shape) == [4, 128]
    assert x_out.dtype == torch.float16
    assert list(mxscale.shape) == [4, 2, 2]  # ceil(ceil(128/32)/2)=2, tail 2
    assert mxscale.dtype == torch.float8_e8m0fnu
    assert list(rstd.shape) == [4, 1]
    assert rstd.dtype == torch.float32


# ---------------------------------------------------------------------------
# 3. Framework behavior lock: top-level entry + dispatcher dtype coercion
# ---------------------------------------------------------------------------


def test_toplevel_entry_is_dispatcher_packet():
    """cann_ops_nn.<op> resolves via the package __getattr__ to the dispatcher
    OpOverloadPacket — NOT the raw function. Scalar args are schema-converted
    before any op code runs, so the Python type gate does not fire there."""
    assert not isinstance(cann_ops_nn.add_rms_norm_dynamic_quant, types.FunctionType)


def test_dispatcher_coerces_torch_dtype_to_ordinal():
    """torch.dtype passed to an `int` schema slot is silently converted by the
    torch dispatcher to its ScalarType ordinal (float8_e5m2 -> 23); the op can
    only guard this via the dst_type domain check. Pins the framework behavior
    so accidental "fixes" that silently change semantics get noticed."""
    x1, x2, gamma = _meta_inputs()
    with pytest.raises(RuntimeError, match="invalid dst_type 23"):
        cann_ops_nn.add_rms_norm_dynamic_quant(
            x1, x2, gamma, dst_type=torch.float8_e5m2
        )
