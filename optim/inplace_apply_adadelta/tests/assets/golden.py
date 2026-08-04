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

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


__golden__ = {
    "kernel": {
        "inplace_apply_adadelta": "inplace_apply_adadelta_golden",
    }
}


_GOLDEN_PYTHON_ENV = "INPLACE_APPLY_ADADELTA_GOLDEN_PYTHON"


def inplace_apply_adadelta_golden(
    var,
    accum,
    accum_update,
    lr,
    rho,
    epsilon,
    grad,
    *,
    use_locking=False,
    **kwargs,
):
    """
    Kernel golden for inplace_apply_adadelta.
    All the parameters follow @inplace_apply_adadelta_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    kwargs may contain: short_soc_version, input_ori_shapes, output_ori_shapes,
                        input_formats, output_formats, input_ori_formats,
                        output_ori_formats, input_dtypes, output_dtypes.
    """
    del kwargs
    input_dtype = var.dtype

    with tempfile.TemporaryDirectory(
        prefix="inplace_apply_adadelta_golden_"
    ) as temp_dir:
        input_path = Path(temp_dir) / "input.npz"
        output_path = Path(temp_dir) / "output.npz"
        np.savez(
            input_path,
            var=var,
            accum=accum,
            accum_update=accum_update,
            lr=lr,
            rho=rho,
            epsilon=epsilon,
            grad=grad,
            use_locking=np.asarray(use_locking, dtype=np.bool_),
        )

        worker_env = os.environ.copy()
        worker_env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        worker_env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        python_executable = worker_env.get(_GOLDEN_PYTHON_ENV, sys.executable)
        subprocess.run(
            [
                python_executable,
                str(Path(__file__).resolve()),
                "--tensorflow-worker",
                str(input_path),
                str(output_path),
            ],
            check=True,
            env=worker_env,
        )

        with np.load(output_path) as output_data:
            return [
                output_data["var"].astype(input_dtype, copy=True),
                output_data["accum"].astype(input_dtype, copy=True),
                output_data["accum_update"].astype(input_dtype, copy=True),
            ]


def _run_tensorflow_worker(input_path, output_path):
    with np.load(input_path) as input_data:
        var = np.ascontiguousarray(input_data["var"])
        accum = np.ascontiguousarray(input_data["accum"])
        accum_update = np.ascontiguousarray(input_data["accum_update"])
        grad = np.ascontiguousarray(input_data["grad"])
        dtype = var.dtype
        lr = np.asarray(input_data["lr"], dtype=dtype).reshape(())
        rho = np.asarray(input_data["rho"], dtype=dtype).reshape(())
        epsilon = np.asarray(input_data["epsilon"], dtype=dtype).reshape(())
        use_locking = bool(input_data["use_locking"].item())

    import tensorflow as tf

    tf_dtype = tf.as_dtype(dtype)
    var_tensor = tf.Variable(var, dtype=tf_dtype)
    accum_tensor = tf.Variable(accum, dtype=tf_dtype)
    accum_update_tensor = tf.Variable(accum_update, dtype=tf_dtype)

    tf.raw_ops.ResourceApplyAdadelta(
        var=var_tensor.handle,
        accum=accum_tensor.handle,
        accum_update=accum_update_tensor.handle,
        lr=tf.convert_to_tensor(lr, dtype=tf_dtype),
        rho=tf.convert_to_tensor(rho, dtype=tf_dtype),
        epsilon=tf.convert_to_tensor(epsilon, dtype=tf_dtype),
        grad=tf.convert_to_tensor(grad, dtype=tf_dtype),
        use_locking=use_locking,
    )

    np.savez(
        output_path,
        var=var_tensor.numpy(),
        accum=accum_tensor.numpy(),
        accum_update=accum_update_tensor.numpy(),
    )


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "--tensorflow-worker":
        raise SystemExit(
            "golden.py is a TTK plugin; direct execution requires --tensorflow-worker"
        )
    _run_tensorflow_worker(sys.argv[2], sys.argv[3])
