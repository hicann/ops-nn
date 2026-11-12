#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Behavior golden for the AssertAsync operator (mode B, self-built golden).

AssertAsync has **no output tensor** and performs no floating-point computation, so
``ops-precision-standard`` (atol/rtol on output tensors) does NOT apply.  The authority
for the pass/fail criterion is the test plan 2.1 §2.3 "judgement criterion table", which
declares **behavior bit-exact** alignment with ``torch._assert_async``:

  * the single element of ``input_condition`` decides pass/fail:
      - BOOL      : ``False``       -> FAIL,  ``True``            -> PASS
      - INT*/UINT8: ``0``           -> FAIL,  any non-zero        -> PASS
      - FP16/32/BF16/DOUBLE: ``+0.0``/``-0.0`` -> FAIL, ``NaN``/``+inf``/``-inf``/non-zero -> PASS
      - COMPLEX64/COMPLEX128: both real and imag parts exactly 0 (incl. ``-0.0`` parts) -> FAIL;
        any NaN/``±inf``/non-zero part -> PASS
  * on FAIL the operator must (A) print the raw ``assert_msg`` bytes to the device log and
    (B) surface the failure (signal B, see the test plan §2.3 B1->B3 degradation).

This module is a **pure CPU/Python** implementation; it never re-uses the kernel logic.
Only the two observable signals are produced by :func:`expected_behavior`.

The dtype -> (dtype_id, elem_byte_size, kind) table mirrors the locked kernel/tiling
metadata (``op_kernel/arch35/assert_async_tiling_data.h``); it is *data setup* used to
inject the correct byte width, not a re-statement of the assertion criterion.
"""

# ---------------------------------------------------------------------------
# dtype metadata (must match op_kernel/arch35/assert_async_tiling_data.h)
# ---------------------------------------------------------------------------
KIND_INT = 0
KIND_FLOAT = 1
KIND_COMPLEX = 2

DTYPE_META = {
    "BOOL": {"dtype_id": 0, "elem_byte_size": 1, "kind": KIND_INT, "family": "bool"},
    "INT8": {"dtype_id": 1, "elem_byte_size": 1, "kind": KIND_INT, "family": "int"},
    "INT16": {"dtype_id": 2, "elem_byte_size": 2, "kind": KIND_INT, "family": "int"},
    "INT32": {"dtype_id": 3, "elem_byte_size": 4, "kind": KIND_INT, "family": "int"},
    "INT64": {"dtype_id": 4, "elem_byte_size": 8, "kind": KIND_INT, "family": "int"},
    "UINT8": {"dtype_id": 5, "elem_byte_size": 1, "kind": KIND_INT, "family": "int"},
    "FP16": {"dtype_id": 6, "elem_byte_size": 2, "kind": KIND_FLOAT, "family": "float"},
    "FP32": {
        "dtype_id": 7,
        "elem_byte_size": 4,
        "kind": KIND_FLOAT,
        "family": "float",
    },
    "BF16": {
        "dtype_id": 8,
        "elem_byte_size": 2,
        "kind": KIND_FLOAT,
        "family": "float",
    },
    "DOUBLE": {
        "dtype_id": 9,
        "elem_byte_size": 8,
        "kind": KIND_FLOAT,
        "family": "float",
    },
    "COMPLEX64": {
        "dtype_id": 10,
        "elem_byte_size": 8,
        "kind": KIND_COMPLEX,
        "family": "complex",
    },
    "COMPLEX128": {
        "dtype_id": 11,
        "elem_byte_size": 16,
        "kind": KIND_COMPLEX,
        "family": "complex",
    },
}

# dtypes NOT accepted by the operator (test plan §4.3 S-E-06..08 / §5.3); the unsigned
# wider-than-8-bit types were removed from the supported set (CUDA _assert_async parity)
UNSUPPORTED_DTYPES = ("STRING", "INT4", "UINT16", "UINT32", "UINT64")

# Host/tiling cap for assert_msg: op_kernel/arch35/assert_async_tiling_data.h declares
# ASSERT_ASYNC_MAX_MSG_LEN = 256 (including the terminating '\0'), so at most 254 bytes of
# the message survive into the kernel (accepted max, CUDA parity). Over-length messages
# (strlen >= 255) are now rejected at both the aclnn layer (161002) and the host tiling
# (dev plan R4); the [:TILING_MAX_MSG_LEN] slicing below is only a defensive fallback and
# is no longer reachable by the shipped ST cases.
ASSERT_ASYNC_MAX_MSG_LEN = 256
TILING_MAX_MSG_LEN = ASSERT_ASYNC_MAX_MSG_LEN - 1  # == 255


def dtype_meta(dtype):
    """Return the (dtype_id, elem_byte_size, kind, family) metadata for a supported dtype."""
    if dtype not in DTYPE_META:
        raise KeyError("unsupported dtype for golden metadata: %r" % (dtype,))
    return DTYPE_META[dtype]


def expected_behavior(dtype, value, msg=""):
    """
    Pure behavior golden for one AssertAsync case.

    Parameters
    ----------
    dtype : str
        One of the supported dtype names in :data:`DTYPE_META`.
    value : object
        The element value.  BOOL -> bool; INT*/UINT* -> int; FP* -> float
        (``float('nan')`` / ``float('inf')`` / ``float('-inf')`` allowed);
        COMPLEX* -> scalar following the existing scalar convention (fail iff the
        scalar is 0, i.e. both the real and the imag parts are zero), or an
        explicit ``(real, imag)`` pair / list.
    msg : str
        The raw ``assert_msg`` bytes (already decoded as a Python str).

    Returns
    -------
    dict with keys:
        expect_fail      : bool   -- operator execution must fail
        expect_msg_text  : str    -- the byte-exact text expected in the device log on failure
        msg_expectation  : str    -- "命中" (must appear) / "不命中" (must not appear) / "平凡" (empty msg)
    """
    meta = dtype_meta(dtype)
    if meta["family"] == "bool":
        expect_fail = bool(value) is False
    elif meta["kind"] == KIND_INT:
        expect_fail = int(value) == 0
    elif meta["kind"] == KIND_FLOAT:
        v = float(value)
        # Bit-exact per element width (elem_byte_size 2/4/8 -> FP16/BF16 & 0x7FFF,
        # FP32 & 0x7FFFFFFF, DOUBLE & 0x7FFFFFFFFFFFFFFF): == 0.0 is True for both
        # +0.0 and -0.0; False for NaN / +/-inf / non-zero.
        expect_fail = v == 0.0
    else:  # complex family (COMPLEX64 / COMPLEX128)
        # torch semantics: complex(l, r) != complex(0, 0) -> the assertion fails iff BOTH
        # the real and the imag part are zero (each +0.0/-0.0 counts as zero; any NaN /
        # +/-inf / non-zero part passes).  Scalar values follow the existing scalar
        # convention: a scalar v denotes both parts, so fail iff v == 0.
        if isinstance(value, (tuple, list)):
            real_part = float(value[0])
            imag_part = float(value[1])
        else:
            real_part = imag_part = float(value)
        expect_fail = (real_part == 0.0) and (imag_part == 0.0)

    msg_text = "" if msg is None else msg
    # R4: over-length assert_msg is now rejected at both the aclnn layer (161002) and the
    # host tiling, so the kernel never sees a message longer than 254 bytes; the slicing
    # below is a defensive fallback only, no longer reachable by shipped ST cases.
    expect_msg_text = msg_text[:TILING_MAX_MSG_LEN]
    if not expect_fail:
        msg_expectation = "不命中"
    elif msg_text == "":
        # Empty message: an empty string is a substring of any string, so "hit" is trivial;
        # such a case can only be judged by signal B (test plan §2.1 "空消息专项").
        msg_expectation = "平凡"
    else:
        msg_expectation = "命中"

    return {
        "expect_fail": expect_fail,
        "expect_msg_text": expect_msg_text,
        "msg_truncated": len(msg_text) > TILING_MAX_MSG_LEN,
        "raw_msg_len": len(msg_text),
        "msg_expectation": msg_expectation,
    }


def expected_reject(shape, dtype):
    """
    Host-side parameter-validation golden (L2).  Returns True when the host must reject
    the input (return non-ACLNN_SUCCESS and not launch any kernel).

    Reject conditions (test plan §4.3 / §5.3):
      * numel != 1   (includes empty tensors, whose numel == 0)
      * dtype not in the supported set
    """
    numel = 1
    for d in shape:
        numel *= int(d)
    if numel != 1:
        return True
    if dtype not in DTYPE_META:
        return True
    return False


def format_value(value):
    """Stable display string for a case value (used in case tables and result records)."""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "+inf"
        if value == float("-inf"):
            return "-inf"
        return repr(value)
    return str(value)


if __name__ == "__main__":
    # Tiny self-check: behaviour golden sanity (not part of the ST run).
    assert expected_behavior("BOOL", False)["expect_fail"] is True
    assert expected_behavior("BOOL", True)["expect_fail"] is False
    assert expected_behavior("INT32", 0)["expect_fail"] is True
    assert expected_behavior("INT32", -1)["expect_fail"] is False
    assert expected_behavior("FP32", 0.0)["expect_fail"] is True
    assert expected_behavior("FP32", -0.0)["expect_fail"] is True
    assert expected_behavior("FP32", float("nan"))["expect_fail"] is False
    assert expected_behavior("FP32", float("inf"))["expect_fail"] is False
    assert expected_behavior("FP32", 0.0, "")["msg_expectation"] == "平凡"
    assert expected_behavior("FP32", 0.0, "m")["msg_expectation"] == "命中"
    assert expected_behavior("FP32", 1.0, "m")["msg_expectation"] == "不命中"
    assert expected_behavior("DOUBLE", 0.0)["expect_fail"] is True
    assert expected_behavior("DOUBLE", -0.0)["expect_fail"] is True
    assert expected_behavior("DOUBLE", float("nan"))["expect_fail"] is False
    assert expected_behavior("DOUBLE", 1.0)["expect_fail"] is False
    assert expected_behavior("COMPLEX64", 0.0)["expect_fail"] is True
    assert expected_behavior("COMPLEX64", (0.0, -0.0))["expect_fail"] is True
    assert expected_behavior("COMPLEX64", (0.0, 1.0))["expect_fail"] is False
    assert expected_behavior("COMPLEX64", (float("nan"), 0.0))["expect_fail"] is False
    assert expected_behavior("COMPLEX64", 1.0)["expect_fail"] is False
    assert expected_behavior("COMPLEX128", 0.0)["expect_fail"] is True
    assert expected_behavior("COMPLEX128", (-0.0, -0.0))["expect_fail"] is True
    assert expected_behavior("COMPLEX128", (1.0, 0.0))["expect_fail"] is False
    assert (
        expected_behavior("COMPLEX128", (float("inf"), float("nan")))["expect_fail"]
        is False
    )
    assert expected_reject([2], "FP32") is True
    assert expected_reject([0], "FP32") is True
    assert expected_reject([1], "STRING") is True
    assert expected_reject([1], "INT4") is True
    assert expected_reject([1], "UINT16") is True
    assert expected_reject([1], "UINT32") is True
    assert expected_reject([1], "UINT64") is True
    assert expected_reject([1], "DOUBLE") is False
    assert expected_reject([1], "COMPLEX64") is False
    assert expected_reject([1], "COMPLEX128") is False
    assert expected_reject([1], "FP32") is False
    print("[golden] self-check OK")
