#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Generate portable TTK regression fixtures for reduction precision boundaries.

Use --output-dir in the task's MyTestCase checkout. No reference output is
embedded: the repository Torch FP64 golden and the GPU composition remain the
independent oracles. --large additionally exercises grouped sub-R carries.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def generate(output_dir, large=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = output_dir / "precision-inputs"
    inputs.mkdir(exist_ok=True)
    cases = (
        Path(__file__).parents[1] / "st/arch35/ttk_kernel_in_training_reduce_v2_st.csv"
    )
    with cases.open() as source:
        template = next(csv.DictReader(source))
    rows, manifest = [], []

    def add(name, shape, layout, dtype, values, purpose):
        np_dtype = np.dtype(dtype)
        count = int(np.prod(shape, dtype=object))
        pattern = np.asarray(values, dtype=np_dtype)
        path = inputs / (name + ".bin")
        # Stream repeated input patterns so grouped tests need no large generator allocation.
        chunk = np.tile(pattern, max(1, (1 << 18) // pattern.size))
        digest = hashlib.sha256()
        with path.open("wb") as target:
            written = 0
            while written < count:
                data = chunk[: min(chunk.size, count - written)].tobytes()
                target.write(data)
                digest.update(data)
                written += len(data) // np_dtype.itemsize
        channel = len(shape) - 1 if layout in ("NHWC", "NDHWC") else 1
        output_shape = tuple(
            value if axis in (0, channel) else 1 for axis, value in enumerate(shape)
        )
        row = dict(template)
        row.update(
            testcase_name=name,
            input_shapes=repr((shape,)),
            input_ori_shapes=repr((shape,)),
            input_formats=repr((layout,)),
            input_ori_formats=repr((layout,)),
            input_dtypes=repr((dtype,)),
            output_shapes=repr((output_shape, output_shape)),
            output_ori_shapes=repr((output_shape, output_shape)),
            manual_input_binaries=repr((str(path.resolve()),)),
            remark=purpose,
            input_data_ranges="((None,None),)",
        )
        rows.append(row)
        manifest.append(
            {
                "case": name,
                "shape": shape,
                "layout": layout,
                "dtype": dtype,
                "purpose": purpose,
                "input_sha256": digest.hexdigest(),
            }
        )

    for dtype in ("float16", "float32"):
        info = np.finfo(dtype)
        tiny = np.nextafter(np.array(0, dtype=dtype), np.array(1, dtype=dtype)).item()
        below_normal = np.nextafter(
            np.array(info.tiny, dtype=dtype), np.array(0, dtype=dtype)
        ).item()
        denormals = [
            tiny,
            -tiny,
            below_normal,
            -below_normal,
            info.tiny,
            -info.tiny,
            tiny,
            tiny,
        ]
        patterns = {
            "denorm": denormals,
            "posinf": [1, np.inf, -2, 3],
            "neginf": [1, -np.inf, -2, 3],
            "bothinf": [1, np.inf, -np.inf, 3],
            "nan": [1, np.nan, -2, 3],
            "cancel": [info.max, info.max, -info.max, -info.max],
        }
        for layout, shape in (
            ("ND", (1, 3, 129)),
            ("NHWC", (1, 1, 129, 7)),
            ("NDHWC", (1, 1, 1, 129, 65)),
        ):
            for kind, values in patterns.items():
                # Channel-last inputs use whole repeated rows. Each channel sees
                # the complete finite/special pattern instead of a constant column.
                if layout != "ND":
                    values = np.repeat(values, shape[-1])
                add(
                    f"precision_{dtype}_{layout}_{kind}",
                    shape,
                    layout,
                    dtype,
                    values,
                    "Explicit " + kind + " and finite mixture; AR or actual C>1 ARA",
                )
        add(
            f"precision_{dtype}_single",
            (1, 1),
            "ND",
            dtype,
            [tiny],
            "numel=1 minimum legal input",
        )
        add(
            f"precision_{dtype}_subr_cancel",
            (1, 1, 32769),
            "ND",
            dtype,
            patterns["cancel"],
            "Finite extreme cancellation across sub-R chunks",
        )
        add(
            f"precision_{dtype}_subr_bothinf",
            (1, 1, 32769),
            "ND",
            dtype,
            patterns["bothinf"],
            "Opposite infinities and finite values across sub-R chunks",
        )
        add(
            f"precision_{dtype}_ara_chunks_cancel",
            (1, 1, 1, 4097, 65),
            "NDHWC",
            dtype,
            np.repeat(patterns["cancel"], 65),
            "Finite extreme cancellation across ARA R chunks and C tails",
        )
        for reduce_size in (
            63,
            64,
            65,
            127,
            128,
            129,
            817,
            4236,
            16383,
            16384,
            16385,
            32769,
        ):
            add(
                f"precision_{dtype}_r{reduce_size}",
                (2, 3, reduce_size),
                "ND",
                dtype,
                [1, 0.125, -0.5, -0.25, 2, -1],
                "VL, folding and full/sub-R capacity boundaries",
            )
        for channels in (8, 16, 17, 63, 64, 65, 127, 128, 129):
            add(
                f"precision_{dtype}_c{channels}",
                (2, 1, 4097, channels),
                "NHWC",
                dtype,
                [1, 0.125, -0.5, -0.25, 2, -1],
                "ARA gather, channel/vector tails and multi-chunk R",
            )
        if large:
            add(
                f"precision_{dtype}_grouped",
                (1, 1, (1 << 28) + 1),
                "ND",
                dtype,
                [1, -0.25, 0.5, -0.125],
                "Grouped sub-R: multiple groups, carry, slot tail and chunk tail",
            )
    csv_path = output_dir / "precision-regression.csv"
    with csv_path.open("w") as target:
        writer = csv.DictWriter(target, fieldnames=list(template), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "precision-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Generated {len(rows)} cases in {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--large", action="store_true")
    options = parser.parse_args()
    generate(options.output_dir, options.large)
