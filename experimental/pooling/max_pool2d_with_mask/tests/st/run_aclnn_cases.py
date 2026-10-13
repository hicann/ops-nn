#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Replay task cases through ACLNN and compare values and valid INT32 indices.

The local default ignores unused index-container bytes. Use --indices-check full
to reproduce the original Golden's zero-tail comparison. The Golden is unchanged.

With --no-verify the comparison is skipped so that a variant which does not
produce the Golden bytes can still be timed; the report then records only that
the cases ran, never that they passed.

This is a correctness workload for CANN msopprof, not a timing implementation.
The official cases and Golden module are supplied as external input files.
"""

import argparse
import ctypes as ct
import importlib.util
import json
import re
from pathlib import Path

import numpy as np

P = ct.c_void_p
I64 = ct.c_int64
U64 = ct.c_uint64
DTYPES = {"float": (np.float32, 0), "float16": (np.float16, 1), "int8": (np.int8, 2)}


def compare_output(actual, expected, output_index, output_elements, indices_check):
    """Compare exact output bytes, with an explicit index-container policy."""
    if indices_check not in ("valid", "full"):
        raise ValueError("unknown indices comparison policy")
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"output {output_index}: shape or dtype mismatch")
    a = np.frombuffer(actual.tobytes(), dtype=np.uint8)
    b = np.frombuffer(expected.tobytes(), dtype=np.uint8)
    if output_index == 1:
        valid_bytes = int(output_elements) * np.dtype("<i4").itemsize
        if expected.dtype != np.int8 or valid_bytes > b.size:
            raise AssertionError("indices container cannot hold the valid INT32 prefix")
        if indices_check == "valid":
            a, b = a[:valid_bytes], b[:valid_bytes]
    differences = np.flatnonzero(a != b)
    if differences.size:
        first = int(differences[0])
        raise AssertionError(
            f"output {output_index}: first differing byte {first}, "
            f"actual={a[first]}, expected={b[first]}"
        )


def bind(lib, name, args, result=ct.c_int):
    fn = getattr(lib, name)
    fn.argtypes, fn.restype = args, result
    return fn


def check(status, operation):
    if status != 0:
        raise RuntimeError(f"{operation}: ACL error {status}")


class Runtime:
    def __init__(self, library):
        self.acl = ct.CDLL("libascendcl.so", mode=ct.RTLD_GLOBAL)
        self.base = ct.CDLL("libnnopbase.so", mode=ct.RTLD_GLOBAL)
        self.op = ct.CDLL(str(library), mode=ct.RTLD_GLOBAL)
        self.init = bind(self.acl, "aclInit", [ct.c_char_p])
        self.finalize = bind(self.acl, "aclFinalize", [])
        self.set_device = bind(self.acl, "aclrtSetDevice", [ct.c_int])
        self.reset_device = bind(self.acl, "aclrtResetDevice", [ct.c_int])
        self.create_stream = bind(self.acl, "aclrtCreateStream", [ct.POINTER(P)])
        self.destroy_stream = bind(self.acl, "aclrtDestroyStream", [P])
        self.sync = bind(self.acl, "aclrtSynchronizeStream", [P])
        self.malloc = bind(
            self.acl, "aclrtMalloc", [ct.POINTER(P), ct.c_size_t, ct.c_int]
        )
        self.free = bind(self.acl, "aclrtFree", [P])
        self.copy = bind(
            self.acl, "aclrtMemcpy", [P, ct.c_size_t, P, ct.c_size_t, ct.c_int]
        )
        self.create_tensor = bind(
            self.base,
            "aclCreateTensor",
            [
                ct.POINTER(I64),
                U64,
                ct.c_int,
                ct.POINTER(I64),
                I64,
                ct.c_int,
                ct.POINTER(I64),
                U64,
                P,
            ],
            P,
        )
        self.destroy_tensor = bind(self.base, "aclDestroyTensor", [P])
        self.create_array = bind(
            self.base, "aclCreateIntArray", [ct.POINTER(I64), U64], P
        )
        self.destroy_array = bind(self.base, "aclDestroyIntArray", [P])
        self.prepare = bind(
            self.op,
            "aclnnMaxPool2dWithMaskGetWorkspaceSize",
            [P, P, P, P, P, ct.c_bool, P, P, ct.POINTER(U64), ct.POINTER(P)],
        )
        self.run = bind(self.op, "aclnnMaxPool2dWithMask", [P, U64, P, P])

    def allocate(self, size):
        ptr = P()
        check(self.malloc(ct.byref(ptr), size, 0), "aclrtMalloc")
        return ptr

    def tensor(self, array, dtype, fmt, strided):
        shape = tuple(array.shape)
        storage_shape = list(shape)
        if strided:
            storage_shape[-1] *= 2
            storage = np.full(storage_shape, 13, dtype=array.dtype)
            storage[..., ::2] = array
            strides = tuple(s // array.itemsize for s in storage[..., ::2].strides)
        else:
            storage = np.ascontiguousarray(array)
            strides = tuple(s // array.itemsize for s in storage.strides)
        dev = self.allocate(storage.nbytes)
        tensor = None
        try:
            check(
                self.copy(
                    dev, storage.nbytes, P(storage.ctypes.data), storage.nbytes, 1
                ),
                "copy H2D",
            )
            dims = (I64 * len(shape))(*shape)
            tensor = self.create_tensor(
                dims,
                len(shape),
                dtype,
                (I64 * len(shape))(*strides),
                0,
                fmt,
                (I64 * len(shape))(*storage_shape),
                len(shape),
                dev,
            )
            if not tensor:
                raise RuntimeError("aclCreateTensor returned null")
            return tensor, dev, storage
        except Exception:
            if tensor:
                self.destroy_tensor(tensor)
            self.free(dev)
            raise


def run_case(rt, stream, case, golden, args, ordinal):
    desc = case["input_desc"][0]
    np_type, acl_type = DTYPES[desc["data_type"]]
    shape = desc["shape"]
    rng = np.random.default_rng(args.seed + ordinal)
    low, high = desc.get("value_range", [-5, 5])
    source = rng.uniform(low, high, size=shape).astype(np_type)
    if args.pattern == "ascending":
        source = (
            np.arange(source.size).reshape(shape) % (shape[-2] * shape[-1])
        ).astype(np_type)
    elif args.pattern == "ties":
        source = (np.arange(source.size).reshape(shape) % 7).astype(np_type)
    elif args.pattern == "zeros":
        source = np.zeros(shape, dtype=np_type)
        source.reshape(-1)[::3] = -0.0
    elif args.pattern == "adjacent":
        if np_type == np.float32:
            source = (
                (np.uint32(0x3F800000) + np.arange(source.size, dtype=np.uint32) % 17)
                .view(np.float32)
                .reshape(shape)
            )
        elif np_type == np.float16:
            source = (
                (np.uint16(0x3C00) + np.arange(source.size, dtype=np.uint16) % 17)
                .view(np.float16)
                .reshape(shape)
            )
        else:
            source = (1 + (np.arange(source.size).reshape(shape) % 17) / 128).astype(
                np_type
            )
    elif args.pattern == "subnormal":
        bits_type = np.uint32 if np_type == np.float32 else np.uint16
        source = (
            (1 + np.arange(source.size, dtype=bits_type) % 31)
            .view(np_type)
            .reshape(shape)
        )
        source.reshape(-1)[::3] *= -1
    elif args.pattern == "positive_inf":
        source.reshape(-1)[::7] = np.inf
    elif args.pattern in ("half_bits", "bf16_bits"):
        expected_type = (
            np.float16 if args.pattern == "half_bits" else DTYPES["bfloat16"][0]
        )
        if np_type != expected_type:
            raise ValueError(f"{args.pattern} requires its matching 16-bit dtype")
        bits = (np.arange(source.size, dtype=np.uint32) % 65536).astype(np.uint16)
        source = bits.view(np_type).reshape(shape)
        source[np.isnan(source) | np.isneginf(source)] = 0
    attrs = {a["name"]: a["value"] for a in case["attr_desc"]}
    expected = golden(source, **attrs)
    for desc_out, value in zip(case["output_desc"], expected):
        if list(value.shape) != desc_out["shape"]:
            raise ValueError(f"Golden shape differs from case: {case['case_name']}")
    tensors, arrays, workspace = [], [], None
    try:
        tensors.append(
            rt.tensor(
                source,
                acl_type,
                2 if desc["format"] == "ND" else 0,
                args.layout in ("input", "both"),
            )
        )
        for i, value in enumerate(expected):
            data = np.full(value.shape, 0x7F if i == 1 else 0, dtype=value.dtype)
            fmt = 2 if case["output_desc"][i]["format"] == "ND" else 0
            tensors.append(
                rt.tensor(
                    data,
                    acl_type if i == 0 else 2,
                    fmt,
                    i == 0 and args.layout in ("output", "both"),
                )
            )
        for key in ("kernelSize", "stride", "padding", "dilation"):
            values = attrs[key]
            arr = rt.create_array((I64 * len(values))(*values), len(values))
            if not arr:
                raise RuntimeError("aclCreateIntArray returned null")
            arrays.append(arr)
        workspace_size, executor = U64(), P()
        check(
            rt.prepare(
                tensors[0][0],
                *arrays,
                attrs["ceilMode"],
                tensors[1][0],
                tensors[2][0],
                ct.byref(workspace_size),
                ct.byref(executor),
            ),
            "GetWorkspaceSize",
        )
        if workspace_size.value:
            workspace = rt.allocate(workspace_size.value)
        check(
            rt.run(workspace, workspace_size, executor, stream),
            "aclnnMaxPool2dWithMask",
        )
        check(rt.sync(stream), "synchronize")
        for i, (_, dev, storage) in enumerate(tensors[1:]):
            check(
                rt.copy(P(storage.ctypes.data), storage.nbytes, dev, storage.nbytes, 2),
                "copy D2H",
            )
            actual = storage
            if i == 0 and args.layout in ("output", "both"):
                if not np.all(storage[..., 1::2] == 13):
                    raise AssertionError("noncontiguous output overwrote storage gaps")
                actual = storage[..., ::2]
            if args.no_verify:
                continue
            compare_output(actual, expected[i], i, expected[0].size, args.indices_check)
    finally:
        for tensor, dev, _ in tensors:
            rt.destroy_tensor(tensor)
            rt.free(dev)
        for arr in arrays:
            rt.destroy_array(arr)
        if workspace:
            rt.free(workspace)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--golden", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument(
        "--match", default=".*", help="Regular expression matching case names"
    )
    parser.add_argument("--types", default="float,float16,bfloat16")
    parser.add_argument(
        "--layout",
        choices=("contiguous", "input", "output", "both"),
        default="contiguous",
    )
    parser.add_argument(
        "--pattern",
        choices=(
            "normal",
            "ascending",
            "ties",
            "zeros",
            "adjacent",
            "subnormal",
            "positive_inf",
            "half_bits",
            "bf16_bits",
        ),
        default="normal",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat each case for official profiling with the selected comparison policy",
    )
    parser.add_argument(
        "--indices-check",
        choices=("valid", "full"),
        default="valid",
        help="Compare the valid INT32 prefix (local default), or the full original Golden container",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Run the cases without comparing the outputs. Only for timing a variant that is "
        "not expected to reproduce the Golden bytes, such as the built-in implementation or "
        "a deliberately incomplete build; never use it to report accuracy",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if "bfloat16" in args.types.split(","):
        from ml_dtypes import bfloat16

        DTYPES["bfloat16"] = (bfloat16, 27)
    spec = importlib.util.spec_from_file_location("task_golden", args.golden)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    selected = [
        (i, c)
        for i, c in enumerate(json.loads(args.cases.read_text()))
        if re.search(args.match, c["case_name"])
        and c["input_desc"][0]["data_type"] in args.types.split(",")
    ]
    if not selected:
        raise ValueError("no cases selected")
    rt = Runtime(args.library)
    check(rt.init(None), "aclInit")
    stream = P()
    records = []
    try:
        check(rt.set_device(args.device), "aclrtSetDevice")
        check(rt.create_stream(ct.byref(stream)), "aclrtCreateStream")
        for ordinal, case in [
            (ordinal, case) for ordinal, case in selected for _ in range(args.repeat)
        ]:
            try:
                run_case(
                    rt, stream, case, module.maxpool2d_with_argmax_expect, args, ordinal
                )
                # Nothing was compared under --no-verify, so the record must not read PASS:
                # a timing report has to stay distinguishable from an accuracy result.
                record = {
                    "case": case["case_name"],
                    "status": "RAN" if args.no_verify else "PASS",
                    "indices_check": "none" if args.no_verify else args.indices_check,
                }
            except Exception as exc:
                record = {
                    "case": case["case_name"],
                    "status": "FAIL",
                    "error": str(exc),
                }
            records.append(record)
            print(json.dumps(record), flush=True)
            if record["status"] == "FAIL":
                break
    finally:
        if stream:
            rt.destroy_stream(stream)
        rt.reset_device(args.device)
        rt.finalize()
        if args.report:
            args.report.write_text(
                json.dumps(
                    {
                        "selected": len(selected) * args.repeat,
                        "pattern": args.pattern,
                        "layout": args.layout,
                        "library": str(args.library),
                        "verified": not args.no_verify,
                        "indices_check": "none"
                        if args.no_verify
                        else args.indices_check,
                        "out_check": "none" if args.no_verify else "exact_bytes",
                        "records": records,
                    },
                    indent=2,
                )
                + "\n"
            )
    expected_status = "RAN" if args.no_verify else "PASS"
    return (
        0
        if len(records) == len(selected) * args.repeat
        and all(r["status"] == expected_status for r in records)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
