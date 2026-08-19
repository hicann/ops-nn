#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Generate deterministic GroupNorm kernel generalization cases for TTK."""

import csv
from pathlib import Path


COLUMNS = [
    "testcase_name",
    "network_name",
    "op_name",
    "input_shapes",
    "input_dtypes",
    "input_formats",
    "output_shapes",
    "output_dtypes",
    "output_formats",
    "input_ori_shapes",
    "input_ori_formats",
    "output_ori_shapes",
    "output_ori_formats",
    "attributes",
    "input_data_ranges",
    "precision_tolerances",
    "absolute_precision",
    "output_inplace_indexes",
    "output_shape_unknown_indexes",
    "is_enabled",
    "remark",
    "soc_series",
    "priority",
    "dump_file_prefix",
    "manual_input_binaries",
    "manual_golden_binaries",
]


def tuple_repr(values):
    return repr(tuple(values))


def shape_tuple_repr(shapes):
    return repr(tuple(tuple(shape) for shape in shapes))


def make_case(index, shape, num_groups, dtype, family, eps, is_training):
    channel = shape[1]
    stats_shape = (shape[0], num_groups)
    input_shapes = [shape, (channel,), (channel,)]
    output_shapes = [shape, stats_shape, stats_shape]
    dtypes = [dtype] * 3
    formats = ["ND"] * 3
    tolerance = 0.001 if dtype == "float16" else 0.0001
    return {
        "testcase_name": f"case{index:05d}_{family}",
        "network_name": "UNKNOWN",
        "op_name": "group_norm",
        "input_shapes": shape_tuple_repr(input_shapes),
        "input_dtypes": tuple_repr(dtypes),
        "input_formats": tuple_repr(formats),
        "output_shapes": shape_tuple_repr(output_shapes),
        "output_dtypes": tuple_repr(dtypes),
        "output_formats": tuple_repr(formats),
        "input_ori_shapes": shape_tuple_repr(input_shapes),
        "input_ori_formats": tuple_repr(formats),
        "output_ori_shapes": shape_tuple_repr(output_shapes),
        "output_ori_formats": tuple_repr(formats),
        "attributes": repr(
            {
                "num_groups": num_groups,
                "data_format": "NCHW",
                "eps": eps,
                "is_training": is_training,
            }
        ),
        "input_data_ranges": "((-2, 2), (-1, 1), (-1, 1))",
        "precision_tolerances": f"(({tolerance}, {tolerance}),)",
        "absolute_precision": str(tolerance),
        "output_inplace_indexes": "()",
        "output_shape_unknown_indexes": "()",
        "is_enabled": "TRUE",
        "remark": family,
        "soc_series": "",
        "priority": "0",
        "dump_file_prefix": "",
        "manual_input_binaries": "()",
        "manual_golden_binaries": "()",
    }


def twopass_performance_cases():
    channels = [4, 8, 16, 32, 48, 64, 96, 128]
    spatial = [(1,), (3,), (7,), (2, 5), (4, 4), (2, 3, 5)]
    cases = []
    for i in range(120):
        channel = channels[i % len(channels)]
        divisors = [value for value in (1, 2, 4, 6, 8, 16, 32) if channel % value == 0]
        groups = divisors[(i // len(channels)) % len(divisors)]
        shape = (i % 4 + 1, channel, *spatial[(i // 3) % len(spatial)])
        cases.append((shape, groups))
    return cases


def welford_performance_cases():
    cases = []
    fp32_reduce = [24576, 32768, 49152, 65536, 98304]
    fp16_reduce = [49152, 65536, 98304, 131072]
    channels = [1, 2, 4, 8, 16, 32, 64]
    for i in range(120):
        dtype = "float16" if i % 2 == 0 else "float32"
        channel = channels[(i // 2) % len(channels)]
        reduce_sizes = fp16_reduce if dtype == "float16" else fp32_reduce
        reduce_size = reduce_sizes[(i // 5) % len(reduce_sizes)]
        spatial_size = reduce_size // channel
        cases.append(((i % 2 + 1, channel, spatial_size), 1, dtype))
    return cases


def welford_generalized_cases():
    cases = []
    channels = [5120, 6144, 8192, 12288, 16384]
    for i in range(120):
        dtype = "float16" if i % 2 == 0 else "float32"
        channel = channels[(i // 2) % len(channels)]
        # Keep large-channel FP16 cases beyond the full gamma/beta TwoPass UB threshold.
        spatial_size = 7 + (i // 10) % 4
        cases.append(((i % 2 + 1, channel, spatial_size), 1, dtype))
    return cases


def twopass_generalized_cases():
    cases = []
    fp16_channels = [65536, 98304, 131072]
    fp32_channels = [32768, 49152, 65536]
    for i in range(120):
        dtype = "float16" if i % 2 == 0 else "float32"
        channels = fp16_channels if dtype == "float16" else fp32_channels
        channel = channels[(i // 2) % len(channels)]
        groups = channel if i % 4 < 2 else channel // 2
        shape = (1, channel, i % 4 + 1)
        cases.append((shape, groups, dtype))
    return cases


def empty_batch_cases():
    cases = []
    channels = [4, 8, 16, 32, 64]
    for i in range(20):
        channel = channels[i % len(channels)]
        groups = [1, 2, 4][i % 3]
        if channel % groups != 0:
            groups = 1
        shape = (0, channel, i % 3 + 1, 8)
        cases.append((shape, groups, "float16" if i % 2 == 0 else "float32"))
    return cases


def generate(output_path):
    rows = []
    index = 1
    eps_values = [1e-5, 1e-4, 1e-3]

    for i, (shape, groups) in enumerate(twopass_performance_cases()):
        dtype = "float16" if i % 2 == 0 else "float32"
        rows.append(
            make_case(
                index, shape, groups, dtype, "key1110", eps_values[i % 3], i % 2 == 0
            )
        )
        index += 1
    for i, (shape, groups, dtype) in enumerate(welford_performance_cases()):
        rows.append(
            make_case(
                index, shape, groups, dtype, "key1100", eps_values[i % 3], i % 2 == 0
            )
        )
        index += 1
    for i, (shape, groups, dtype) in enumerate(welford_generalized_cases()):
        rows.append(
            make_case(
                index, shape, groups, dtype, "key1120", eps_values[i % 3], i % 2 == 0
            )
        )
        index += 1
    for i, (shape, groups, dtype) in enumerate(twopass_generalized_cases()):
        rows.append(
            make_case(
                index, shape, groups, dtype, "key1130", eps_values[i % 3], i % 2 == 0
            )
        )
        index += 1
    for i, (shape, groups, dtype) in enumerate(empty_batch_cases()):
        rows.append(
            make_case(
                index, shape, groups, dtype, "empty_batch_key1100", 1e-4, i % 2 == 0
            )
        )
        index += 1

    if len(rows) != 500:
        raise RuntimeError(f"Expected 500 cases, got {len(rows)}")
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    generate(Path(__file__).with_name("ttk_kernel_group_norm_generalization.csv"))
