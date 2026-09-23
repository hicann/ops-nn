# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Generate GEIR example fixtures exclusively from the user-provided Golden."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--golden", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
expected_sha = "1de4ac85e8c2c6376d307647ae585ec5904f7d859d4c199686c2a9a18c0c2ce9"
assert hashlib.sha256(args.golden.read_bytes()).hexdigest() == expected_sha
spec = importlib.util.spec_from_file_location("user_gru_golden", args.golden)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
args.output.mkdir(parents=True, exist_ok=False)
manifest = {
    "golden": str(args.golden.resolve()),
    "golden_sha256": expected_sha,
    "golden_mode": "Promote",
    "rtol": 1e-5,
    "atol": 1e-6,
    "threshold_source": "operator-artifacts/GRUBlockCellGrad/spec.yaml numerical_tolerance.float32",
    "seed": 42,
    "cases": [],
}
rng = np.random.default_rng(42)
for case_id, (b, i, c) in enumerate(((2, 3, 4), (1, 7, 5), (3, 5, 7))):
    shapes = [
        (b, i),
        (b, c),
        (i + c, 2 * c),
        (i + c, c),
        (2 * c,),
        (c,),
        (b, c),
        (b, c),
        (b, c),
        (b, c),
    ]
    values = [
        rng.uniform(0.1, 0.9, size=shape).astype(np.float32)
        if j in (6, 7)
        else rng.uniform(-0.5, 0.5, size=shape).astype(np.float32)
        for j, shape in enumerate(shapes)
    ]
    outputs = module.GRUBlockCellGradKernelSpec.golden(
        *[v.astype(np.float64) for v in values], golden_mode="Promote"
    )
    assert len(outputs) == 4 and all(v.dtype == np.float64 for v in outputs)
    files = {}
    for kind, arrays in [("input", values), ("golden", outputs)]:
        for j, value in enumerate(arrays):
            path = args.output / f"case{case_id}_{kind}{j}.bin"
            np.ascontiguousarray(value).tofile(path)
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest["cases"].append(
        {
            "case_id": case_id,
            "BIC": [b, i, c],
            "input_shapes": shapes,
            "output_shapes": [list(v.shape) for v in outputs],
            "files": files,
        }
    )
(args.output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"cases": 3, "golden_fp64": True, "output": str(args.output)}))
