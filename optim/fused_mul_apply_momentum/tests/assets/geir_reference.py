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
"""Build a standalone GEIR reference plugin around the unchanged native golden.

The selected policy must come from the actual graph's Cast/compute descriptors.
It is not a policy for the operator's native Kernel interface.
"""

import argparse
from pathlib import Path

_WRAPPER = """

# GEIR adapter: explicit graph casts, followed by the native public-TF reference.
_GEIR_COMPUTE_DTYPE = {compute_dtype!r}
_GEIR_OUTPUT_DTYPE = {output_dtype!r}
_GEIR_NATIVE_SPEC = globals()[next(iter(__spec__.values()))]
_GEIR_NATIVE_GOLDEN = _GEIR_NATIVE_SPEC.golden
_GEIR_NATIVE_THIRD = _GEIR_NATIVE_SPEC.third_party["tf"]


def _geir_inputs(values, promote):
    converted = []
    with tf.device("/CPU:0"):
        for value in values:
            # Quantize at the graph's input Cast boundary, before CPU promotion.
            value = tf.cast(tf.convert_to_tensor(_numpy(value)),
                            tf.as_dtype(_GEIR_COMPUTE_DTYPE)).numpy()
            if promote:
                promoted_dtype = "float64" if _GEIR_COMPUTE_DTYPE == "float32" else "float32"
                value = value.astype(promoted_dtype)
            converted.append(value)
    return converted


def _geir_outputs(values):
    if _GEIR_OUTPUT_DTYPE is None:
        return values
    # Only an explicitly observed graph output Cast may quantize the output.
    with tf.device("/CPU:0"):
        return [tf.cast(tf.convert_to_tensor(value),
                        tf.as_dtype(_GEIR_OUTPUT_DTYPE)).numpy() for value in values]


def _geir_golden({signature}, **kwargs):
    values = ({arguments},)
    values = _geir_inputs(values, kwargs.get("golden_mode") == "Promote")
    return _geir_outputs(_GEIR_NATIVE_GOLDEN(*values, **kwargs))


class _GEIRTensorFlow(_GEIR_NATIVE_THIRD):
    def __call__(self, {signature}, **kwargs):
        values = ({arguments},)
        return _geir_outputs(super().__call__(*_geir_inputs(values, False), **kwargs))


_GEIR_NATIVE_SPEC.golden = staticmethod(_geir_golden)
_GEIR_NATIVE_SPEC.third_party = {{"tf": _GEIRTensorFlow}}
"""


def write_plugin(native_golden, output, compute_dtype, output_dtype=None):
    supported = ("float16", "bfloat16", "float32")
    if compute_dtype not in supported or output_dtype not in (*supported, None):
        raise ValueError("Unsupported graph dtype policy")
    source = Path(native_golden).read_text(encoding="utf-8")
    signature = (
        "y, dy" if '"acosh_grad"' in source else "var, accum, lr, x1, momentum, x2"
    )
    text = (
        source.rstrip()
        + "\n"
        + _WRAPPER.format(
            compute_dtype=compute_dtype,
            output_dtype=output_dtype,
            signature=signature,
            arguments=signature,
        )
    )
    compile(text, str(output), "exec")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native-golden", type=Path, default=Path(__file__).with_name("golden.py")
    )
    parser.add_argument("--compute-dtype", required=True)
    parser.add_argument("--output-dtype")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_plugin(args.native_golden, args.output, args.compute_dtype, args.output_dtype)


if __name__ == "__main__":
    main()
