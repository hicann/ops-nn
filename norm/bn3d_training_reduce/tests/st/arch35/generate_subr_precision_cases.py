#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Reproduce the frozen FP16 sub-R failures; references use tests/assets/golden.py."""

import argparse
import csv
from pathlib import Path

import numpy as np


def generate(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for channels in (1, 4):
        shape = (1, channels, 1048575)
        name = f"bn3d_subr_fp16_c{channels}_square_sum"
        array = np.random.default_rng(731).uniform(1, 8, size=shape).astype(np.float16)
        path = output_dir / f"{name}.npy"
        np.save(path, array)
        rows.append(
            {
                "testcase_name": name,
                "op_name": "bn3d_training_reduce",
                "input_shapes": repr((shape,)),
                "input_dtypes": repr(("float16",)),
                "input_formats": repr(("NCDHW",)),
                "input_ori_shapes": repr((shape,)),
                "input_ori_formats": repr(("NCDHW",)),
                "output_shapes": repr(((channels,), (channels,))),
                "output_dtypes": repr(("float32", "float32")),
                "output_formats": repr(("NCDHW", "NCDHW")),
                "dyn_input_shapes": repr(((-1, -1, -1),)),
                "attributes": "{}",
                "manual_input_binaries": repr((str(path.resolve()),)),
                "remark": "FP16 sub-R block and slot rounding; fixed seed 731; original cross_check L1.",
            }
        )
    with (output_dir / "cases.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(f"Generated {generate(args.output_dir.resolve())} cases")
