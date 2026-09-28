#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Prepare small TTK manual-data regressions without changing Golden semantics.

Run with an unused output directory. The generated cases.csv is used with
kernel --manual-data-dirs <output>/data --golden-mode Enable --compare mixed,
the sibling golden.py plugin, -d=false -b=release and --pc=1.
"""

import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np


def prepare(output):
    assets = Path(__file__).resolve().parent
    module_spec = importlib.util.spec_from_file_location(
        "intu2_golden", assets / "golden.py"
    )
    golden = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(golden)
    template_path = assets.parent / "st/arch35/ttk_kernel_in_training_update_v2.csv"
    with template_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        template = next(reader)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for dtype in ("float16", "float32"):
        for layout in ("NCHW", "NHWC"):
            for mode in (
                "cancellation",
                "inf_gamma",
                "control",
                "finite_scale_overflow",
                "zero_std_round",
                "square_inf",
                "square_inf_noaffine",
                "infinite_mean_std",
                "exact_mean",
            ):
                name = f"intu2_fix_{dtype}_{layout}_{mode}"
                shape = (1, 1, 1, 3) if layout == "NCHW" else (1, 1, 3, 1)
                if mode == "exact_mean":
                    shape = (1, 1, 1, 6290) if layout == "NCHW" else (1, 1, 6290, 1)
                stat = (1, 1, 1, 1)
                cancellation = mode == "cancellation"
                x = (
                    np.full(shape, -1.50390625, dtype=dtype)
                    if mode == "exact_mean"
                    else np.full(shape, 65504.0, dtype=dtype)
                    if cancellation
                    else np.array([1.0, 2.0, 3.0], dtype=dtype).reshape(shape)
                )
                # Deliberately negative raw variance exercises the documented clamp.
                inputs = [
                    x,
                    np.full(
                        stat, 196512.015625 if cancellation else 6.0, dtype=np.float32
                    ),
                    np.full(stat, 0.0 if cancellation else 14.0, dtype=np.float32),
                    np.full(
                        stat, np.inf if mode == "inf_gamma" else 1.0, dtype=np.float32
                    ),
                    np.zeros(stat, dtype=np.float32),
                    None,
                    None,
                ]
                epsilon = 1e-5
                if mode == "infinite_mean_std":
                    inputs[1].fill(np.inf)
                    inputs[3] = inputs[4] = None
                    epsilon = np.inf
                elif mode in ("square_inf", "square_inf_noaffine"):
                    if dtype == "float32":
                        inputs[0] = np.array(
                            [-np.finfo(np.float32).max, 0, np.finfo(np.float32).max],
                            dtype=dtype,
                        ).reshape(shape)
                    # Exact sum/R isolates intermediate overflow from the
                    # unrelated absolute-error floor of a rounded huge mean.
                    inputs[1].fill(-3.0 * 2.0**124)
                    inputs[2].fill(np.inf)
                    inputs[4].fill(0.25)
                    if mode == "square_inf_noaffine":
                        inputs[3] = inputs[4] = None
                elif mode == "exact_mean":
                    inputs[1].fill(-1.50390625 * 6290)
                    inputs[2].fill((1.50390625**2 + 0.25) * 6290)
                    inputs[3].fill(-np.finfo(np.float32).max)
                elif mode == "finite_scale_overflow":
                    inputs[0] = np.array([-0.25, 0.0, 0.25], dtype=dtype).reshape(shape)
                    inputs[1].fill(0.0)
                    inputs[2].fill(0.1875)
                    inputs[3].fill(np.finfo(np.float32).max)
                    epsilon = 0.0
                elif mode == "zero_std_round":
                    inputs[0].fill(-65504.0)
                    inputs[1].fill(-196512.0)
                    inputs[2].fill(0.0)
                    inputs[3] = inputs[4] = None
                    epsilon = 0.0
                shapes = (shape, stat, stat, stat, stat, None, None)
                formats = (layout,) * 5 + ("ND", "ND")
                if mode in (
                    "zero_std_round",
                    "square_inf_noaffine",
                    "infinite_mean_std",
                ):
                    shapes = (shape, stat, stat, None, None, None, None)
                    formats = (layout,) * 3 + ("ND",) * 4
                row = dict(
                    template,
                    testcase_name=name,
                    attributes=str({"epsilon": epsilon, "momentum": 0.1}),
                    input_shapes=str(shapes),
                    input_ori_shapes=str(shapes),
                    output_shapes=str((shape, stat, stat)),
                    output_ori_shapes=str((shape, stat, stat)),
                    input_formats=str(formats),
                    input_ori_formats=str(formats),
                    output_formats=str((layout,) * 3),
                    output_ori_formats=str((layout,) * 3),
                    input_dtypes=str((dtype,) + ("float32",) * 6),
                    output_dtypes=str((dtype, "float32", "float32")),
                    input_data_ranges=str(((0, 1),) * 7),
                    remark="fixed-input precision regression",
                    dump_file_prefix="",
                )
                promoted = [
                    value.astype(np.float64) if value is not None else None
                    for value in inputs
                ]
                if mode == "infinite_mean_std":
                    row["attributes"] = "{'epsilon': float('inf'), 'momentum': 0.1}"
                outputs = golden._kernel_golden(
                    *promoted, epsilon=epsilon, momentum=0.1, input_formats=formats
                )
                directory = output / "data" / name
                directory.mkdir(parents=True)
                for role, values in (("input", inputs), ("golden", outputs)):
                    for index, value in enumerate(values):
                        if value is None:
                            (directory / f"{role}_{index}_none.npy").touch()
                        else:
                            np.save(
                                directory / f"{role}_{index}_{value.dtype.name}.npy",
                                value,
                            )
                rows.append(row)
    with (output / "cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    prepare(parser.parse_args().output)
