#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Public scalar scatter regression on Ascend 950.

Run in a fresh process with ASCEND_CUSTOM_OPP_PATH pointing at the package under
test and LD_PRELOAD pointing at its libcust_opapi.so. Pass that library to
--op-api to exercise ACLNN directly, avoiding Torch's CPU fallback for some
scalar overloads. CPU scatter is the oracle for unique-index updates. Repeated
indices in the non-deterministic template are execution checks, not precision gates.
"""

import argparse
import ctypes as C
import json
import warnings
from pathlib import Path

import torch
import torch_npu  # noqa: F401


class ScatterApi:
    """Call the public ACLNN APIs with Torch-owned device buffers."""

    DTYPES = {
        torch.float32: 0,
        torch.float16: 1,
        torch.int8: 2,
        torch.int32: 3,
        torch.uint8: 4,
        torch.int16: 6,
        torch.int64: 9,
        torch.float64: 11,
        torch.bool: 12,
        torch.bfloat16: 27,
    }

    def __init__(self, path):
        self.lib = C.CDLL(str(path))
        self.meta = C.CDLL("libnnopbase.so")
        self.runtime = C.CDLL("libascendcl.so")
        self.runtime.aclrtCtxSetSysParamOpt.argtypes = [C.c_int, C.c_int64]
        pointer, integer, size, dims = (
            C.c_void_p,
            C.c_int64,
            C.c_uint64,
            C.POINTER(C.c_int64),
        )
        self.meta.aclCreateTensor.argtypes = [
            dims,
            size,
            C.c_int,
            dims,
            integer,
            C.c_int,
            dims,
            size,
            pointer,
        ]
        self.meta.aclCreateTensor.restype = pointer
        self.meta.aclCreateScalar.argtypes = [pointer, C.c_int]
        self.meta.aclCreateScalar.restype = pointer
        self.meta.aclDestroyTensor.argtypes = [pointer]
        self.meta.aclDestroyScalar.argtypes = [pointer]
        for name in (
            "aclnnScatterValue",
            "aclnnInplaceScatterValue",
            "aclnnScatter",
            "aclnnInplaceScatter",
        ):
            get_workspace = getattr(self.lib, name + "GetWorkspaceSize")
            get_workspace.argtypes = (
                [pointer, integer, pointer, pointer, integer]
                + ([] if "Inplace" in name else [pointer])
                + [C.POINTER(size), C.POINTER(pointer)]
            )
            get_workspace.restype = C.c_int
            getattr(self.lib, name).argtypes = [pointer, size, pointer, pointer]
            getattr(self.lib, name).restype = C.c_int

    def set_deterministic(self, enabled):
        status = self.runtime.aclrtCtxSetSysParamOpt(0, int(enabled))
        if status:
            raise RuntimeError(f"aclrtCtxSetSysParamOpt: {status}")

    def tensor(self, tensor):
        def array(values):
            return (C.c_int64 * len(values))(*values)

        storage = [tensor.untyped_storage().nbytes() // tensor.element_size()]
        handle = self.meta.aclCreateTensor(
            array(tensor.shape),
            tensor.dim(),
            self.DTYPES[tensor.dtype],
            array(tensor.stride()),
            tensor.storage_offset(),
            2,
            array(storage),
            1,
            tensor.untyped_storage().data_ptr(),
        )
        if not handle:
            raise RuntimeError("aclCreateTensor failed")
        return handle

    def __call__(self, source, axis, index, value, reduction="none", inplace=False):
        self.set_deterministic(torch.are_deterministic_algorithms_enabled())
        tensor_value = isinstance(value, torch.Tensor)
        name = (
            ("aclnnInplaceScatter" if inplace else "aclnnScatter")
            if tensor_value
            else ("aclnnInplaceScatterValue" if inplace else "aclnnScatterValue")
        )
        result = source if inplace else torch.empty_like(source)
        handles = [self.tensor(tensor) for tensor in (source, index, result)]
        if tensor_value:
            update = self.tensor(value)
        else:
            scalar = torch.tensor(value, dtype=source.dtype)
            update = self.meta.aclCreateScalar(
                scalar.data_ptr(), self.DTYPES[source.dtype]
            )
        size, executor = C.c_uint64(), C.c_void_p()
        try:
            args = [
                handles[0],
                axis,
                handles[1],
                update,
                {"none": 0, "add": 1, "multiply": 2}[reduction],
            ]
            if "Inplace" not in name:
                args.append(handles[2])
            status = getattr(self.lib, name + "GetWorkspaceSize")(
                *args, C.byref(size), C.byref(executor)
            )
            if status:
                raise RuntimeError(f"{name}GetWorkspaceSize: {status}")
            workspace = torch.empty(size.value, dtype=torch.uint8, device=source.device)
            status = getattr(self.lib, name)(
                workspace.data_ptr(),
                size,
                executor,
                torch.npu.current_stream().npu_stream,
            )
            if status:
                raise RuntimeError(f"{name}: {status}")
            torch.npu.synchronize()
            return result
        finally:
            for handle in handles:
                self.meta.aclDestroyTensor(handle)
            (
                self.meta.aclDestroyTensor
                if tensor_value
                else self.meta.aclDestroyScalar
            )(update)


def torch_scatter(source, axis, index, value, reduction="none", inplace=False):
    options = {} if reduction == "none" else {"reduce": reduction}
    method = source.scatter_ if inplace else source.scatter
    return method(axis, index, value, **options)


def check_tensor_input_contract(scatter, index_dtypes):
    count = 0
    cases = (
        (torch.zeros(8), [0, 2, 4, 6], torch.tensor(3.0), False),
        (torch.zeros(8), [0, 2, 4, 6], torch.tensor([3.0]), False),
        (torch.arange(8.0), [2], torch.tensor(3.0), True),
        (torch.arange(8.0), [2], torch.tensor([3.0]), True),
        (torch.tensor(2.0), 0, torch.tensor(3.0), True),
        (torch.arange(8.0), [], torch.tensor(3.0), True),
        (torch.arange(8.0), [0, 2, 4, 6], 3.0, True),
    )
    torch.use_deterministic_algorithms(False)
    for index_dtype in index_dtypes:
        for reduction in ("none", "add", "multiply"):
            for inplace in (False, True):
                for source, positions, update, valid in cases:
                    index = torch.tensor(positions, dtype=torch.int64)
                    device_update = (
                        update.npu() if isinstance(update, torch.Tensor) else update
                    )
                    device_source = source.npu()
                    try:
                        actual = scatter(
                            device_source,
                            0,
                            index.to(index_dtype).npu(),
                            device_update,
                            reduction,
                            inplace,
                        )
                    except RuntimeError as error:
                        if valid:
                            raise
                        if isinstance(scatter, ScatterApi):
                            assert str(error).endswith("GetWorkspaceSize: 161002"), str(
                                error
                            )
                    else:
                        assert valid, (
                            "tensor scatter unexpectedly accepted scalar broadcasting"
                        )
                        expected = torch_scatter(
                            source.clone(), 0, index, update, reduction
                        )
                        torch.testing.assert_close(
                            actual.cpu(), expected, rtol=0, atol=0
                        )
                        if not inplace:
                            torch.testing.assert_close(
                                device_source.cpu(), source, rtol=0, atol=0
                            )
                    count += 1
    return count


def check_scalar_input_contract(scatter, index_dtypes):
    cases = (
        ((2, 3), (2,), 0),  # self/index rank mismatch
        ((2, 3), (1, 4), 0),  # index exceeds self outside the scatter axis
        ((2, 3), (1, 1), 2),  # positive axis out of range
        ((2, 3), (1, 1), -3),  # negative axis out of range
        ((1,) * 9, (1,) * 9, 0),  # rank exceeds the API limit
    )
    count = 0
    for index_dtype in index_dtypes:
        for reduction in ("none", "add", "multiply"):
            for inplace in (False, True):
                for shape, index_shape, axis in cases:
                    source = torch.zeros(shape, device="npu")
                    index = torch.zeros(index_shape, dtype=index_dtype, device="npu")
                    try:
                        scatter(source, axis, index, 3.0, reduction, inplace)
                    except RuntimeError as error:
                        if isinstance(scatter, ScatterApi):
                            assert str(error).endswith("GetWorkspaceSize: 161002"), str(
                                error
                            )
                    else:
                        raise AssertionError("scalar scatter accepted invalid metadata")
                    count += 1
    return count


def check_case(
    scatter,
    shape,
    index_shape,
    axis,
    dtype,
    reduction,
    index_dtype,
    deterministic,
    inplace,
):
    generator = torch.Generator().manual_seed(2026)
    source = torch.randint(0, 3, shape, generator=generator).to(dtype)
    base_shape = list(index_shape)
    base_shape[axis] = 1
    axis_shape = [1] * len(shape)
    axis_shape[axis] = index_shape[axis]
    index = (
        torch.randint(shape[axis], base_shape, generator=generator)
        + torch.arange(index_shape[axis]).reshape(axis_shape)
    ) % shape[axis]
    value = True if dtype == torch.bool else (3 if dtype == torch.uint8 else -1)
    options = {} if reduction == "none" else {"reduce": reduction}
    expected = source.clone().scatter_(axis, index, value, **options)
    actual = source.npu()
    device_index = index.to(index_dtype).npu()
    torch.use_deterministic_algorithms(deterministic)
    result = scatter(actual, axis, device_index, value, reduction, inplace)
    if not inplace:
        torch.testing.assert_close(actual.cpu(), source, rtol=0, atol=0)
    torch.testing.assert_close(
        result.cpu(),
        expected,
        rtol=0,
        atol=0,
        msg=lambda message: f"{dtype=}, {index_dtype=}, {reduction=}, {shape=}, {axis=}, {deterministic=}, {inplace=}: {message}",
    )


def check_multiply_neighbours(scatter, dtype, index_dtype):
    count = 0
    generator = torch.Generator().manual_seed(20261012)
    value = 3 if dtype == torch.uint8 else -3
    for size in (513, 65537):
        for offset in (1, 3):
            for deterministic in (False, True):
                source = torch.randint(0, 17, (size + 8,), generator=generator).to(
                    dtype
                )
                index = torch.randperm(size, generator=generator)
                expected = source.clone()
                expected[offset : offset + size].scatter_(
                    0, index, value, reduce="multiply"
                )
                parent = source.npu()
                torch.use_deterministic_algorithms(deterministic)
                scatter(
                    parent[offset : offset + size],
                    0,
                    index.to(index_dtype).npu(),
                    value,
                    "multiply",
                    True,
                )
                torch.testing.assert_close(parent.cpu(), expected, rtol=0, atol=0)
                count += 1
    return count


def run(dtypes, index_dtypes, scatter):
    count = 0
    duplicate_execution_checks = 0
    saved_determinism = torch.are_deterministic_algorithms_enabled()
    try:
        for index_dtype in index_dtypes:
            for rank in range(1, 9):
                for axis in range(rank):
                    shape = [3] * rank
                    shape[axis] = 7
                    index_shape = [2] * rank
                    index_shape[axis] = 5
                    for reduction in ("none", "add", "multiply"):
                        for deterministic in (False, True):
                            for inplace in (False, True):
                                check_case(
                                    scatter,
                                    shape,
                                    index_shape,
                                    axis,
                                    torch.float32,
                                    reduction,
                                    index_dtype,
                                    deterministic,
                                    inplace,
                                )
                                count += 1
            # Unique-index tails across SIMT launch boundaries must update each element once.
            for index_size in (129, 513, 1025):
                for reduction in ("none", "add", "multiply"):
                    for inplace in (False, True):
                        check_case(
                            scatter,
                            [8193],
                            [index_size],
                            0,
                            torch.float32,
                            reduction,
                            index_dtype,
                            False,
                            inplace,
                        )
                        count += 1
            # Equal trailing dimensions exercise tiling coalescing.
            check_case(
                scatter,
                [2, 5, 5, 7],
                [2, 4, 5, 7],
                1,
                torch.float32,
                "add",
                index_dtype,
                True,
                False,
            )
            count += 1
            for dtype in dtypes:
                reductions = (
                    ("none",)
                    if dtype == torch.float64
                    else (
                        ("none", "add")
                        if dtype == torch.bool
                        else ("none", "add", "multiply")
                    )
                )
                for reduction in reductions:
                    for deterministic in (False, True):
                        for inplace in (False, True):
                            check_case(
                                scatter,
                                [8193],
                                [4097],
                                0,
                                dtype,
                                reduction,
                                index_dtype,
                                deterministic,
                                inplace,
                            )
                            count += 1
                    # Repeated indices exercise contention without a precision assertion.
                    torch.use_deterministic_algorithms(False)
                    repeated = torch.arange(4097, dtype=index_dtype).remainder(17).npu()
                    value = (
                        True
                        if dtype == torch.bool
                        else (3 if dtype == torch.uint8 else -1)
                    )
                    result = scatter(
                        torch.ones(17, dtype=dtype, device="npu"),
                        0,
                        repeated,
                        value,
                        reduction,
                        True,
                    )
                    torch.npu.synchronize()
                    assert result.shape == (17,)
                    count += 1
                    duplicate_execution_checks += 1
                if dtype in (
                    torch.int8,
                    torch.uint8,
                    torch.int16,
                    torch.float16,
                    torch.bfloat16,
                ):
                    # Distinct subword destinations must preserve neighbours and guards.
                    count += check_multiply_neighbours(scatter, dtype, index_dtype)
                    torch.use_deterministic_algorithms(False)
                    parent = torch.arange(11).to(dtype).npu()
                    expected = parent.cpu()
                    value = 3 if dtype == torch.uint8 else -1
                    idx = torch.tensor([0, 1, 6], dtype=index_dtype)
                    expected[1:8].scatter_(0, idx.long(), value, reduce="multiply")
                    scatter(parent[1:8], 0, idx.npu(), value, "multiply", True)
                    torch.testing.assert_close(parent.cpu(), expected, rtol=0, atol=0)
                    count += 1
                    repeated = torch.tensor(
                        [0, 0, 1, 6, 6, 6], dtype=index_dtype, device="npu"
                    )
                    scatter(parent[1:8], 0, repeated, value, "multiply", True)
                    torch.testing.assert_close(
                        parent[0].cpu(), expected[0], rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        parent[8:].cpu(), expected[8:], rtol=0, atol=0
                    )
                    count += 1
                    duplicate_execution_checks += 1

        # A one-element tensor src retains the tensor API's shape contract.
        data = torch.zeros((2, 3), device="npu")
        index = torch.zeros((2, 2), dtype=torch.int64, device="npu")
        try:
            scatter(data, 1, index, torch.ones(1, device="npu"))
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "tensor scatter unexpectedly accepted scalar broadcasting"
            )
        count += 1
        count += check_tensor_input_contract(scatter, index_dtypes)
        count += check_scalar_input_contract(scatter, index_dtypes)
        for reduction in ("none", "add", "multiply"):
            scalar = torch.tensor(2.0, device="npu")
            scatter(
                scalar,
                0,
                torch.tensor(0, dtype=torch.int64, device="npu"),
                3.0,
                reduction,
                True,
            )
            expected = {"none": 3.0, "add": 5.0, "multiply": 6.0}[reduction]
            torch.testing.assert_close(
                scalar.cpu(), torch.tensor(expected), rtol=0, atol=0
            )
            empty_index = torch.empty((0,), dtype=torch.int64, device="npu")
            original = torch.arange(7.0, device="npu")
            torch.testing.assert_close(
                scatter(original, 0, empty_index, 3.0, reduction).cpu(),
                original.cpu(),
                rtol=0,
                atol=0,
            )
            count += 2
    finally:
        torch.use_deterministic_algorithms(saved_determinism)
        if isinstance(scatter, ScatterApi):
            scatter.set_deterministic(saved_determinism)
    return {"passed": count, "duplicate_execution_checks": duplicate_execution_checks}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dtypes",
        nargs="+",
        default=[
            "float32",
            "float16",
            "bfloat16",
            "int8",
            "uint8",
            "int16",
            "int32",
            "int64",
            "bool",
            "float64",
        ],
    )
    parser.add_argument("--index-dtypes", nargs="+", default=["int32", "int64"])
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--op-api",
        type=Path,
        help="Path to the tested libcust_opapi.so for direct ACLNN calls",
    )
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The reduce argument of torch.scatter"
        )
        scatter = ScatterApi(args.op_api) if args.op_api else torch_scatter
        result = {
            "interface": str(args.op_api) if args.op_api else "torch",
            **run(
                [getattr(torch, dtype) for dtype in args.dtypes],
                [getattr(torch, dtype) for dtype in args.index_dtypes],
                scatter,
            ),
        }
    print(json.dumps(result))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2))
