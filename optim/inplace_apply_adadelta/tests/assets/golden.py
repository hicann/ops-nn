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

import argparse
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
_BFLOAT16_DTYPE_NAME = "bfloat16"


def _serialize_tensor(tensor, dtype):
    array = np.asarray(tensor, dtype=dtype)
    if dtype.name == _BFLOAT16_DTYPE_NAME:
        return array.view(np.uint16)
    return array


def _restore_tensor(tensor, dtype):
    if dtype.name == _BFLOAT16_DTYPE_NAME:
        return tensor.view(dtype).copy()
    return tensor.astype(dtype, copy=True)


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

    All parameters follow inplace_apply_adadelta_def.cpp without outputs.
    Input tensors are numpy.ndarray. The three returned arrays correspond to
    var, accum and accum_update.
    """
    del kwargs
    var = np.asarray(var)
    input_dtype = var.dtype
    dtype_name = input_dtype.name
    if any(
        np.asarray(tensor).dtype != input_dtype
        for tensor in (accum, accum_update, grad)
    ):
        raise TypeError("var/accum/accum_update/grad must have the same dtype")

    with tempfile.TemporaryDirectory(
        prefix="inplace_apply_adadelta_golden_"
    ) as temp_dir:
        input_path = Path(temp_dir) / "input.npz"
        output_path = Path(temp_dir) / "output.npz"
        np.savez(
            input_path,
            var=_serialize_tensor(var, input_dtype),
            accum=_serialize_tensor(accum, input_dtype),
            accum_update=_serialize_tensor(accum_update, input_dtype),
            lr=_serialize_tensor(lr, input_dtype),
            rho=_serialize_tensor(rho, input_dtype),
            epsilon=_serialize_tensor(epsilon, input_dtype),
            grad=_serialize_tensor(grad, input_dtype),
            use_locking=np.asarray(use_locking, dtype=np.bool_),
            dtype_name=np.asarray(dtype_name),
        )

        worker_env = os.environ.copy()
        worker_env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        worker_env.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        python_executable = worker_env.get(_GOLDEN_PYTHON_ENV, sys.executable)
        subprocess.run(
            [
                python_executable,
                str(Path(__file__).resolve()),
                "--worker",
                str(input_path),
                str(output_path),
            ],
            check=True,
            env=worker_env,
        )

        with np.load(output_path) as output_data:
            return [
                _restore_tensor(output_data["var"], input_dtype),
                _restore_tensor(output_data["accum"], input_dtype),
                _restore_tensor(output_data["accum_update"], input_dtype),
            ]


def _run_worker(input_path, output_path):
    with np.load(input_path) as input_data:
        var = np.ascontiguousarray(input_data["var"])
        accum = np.ascontiguousarray(input_data["accum"])
        accum_update = np.ascontiguousarray(input_data["accum_update"])
        grad = np.ascontiguousarray(input_data["grad"])
        lr = input_data["lr"]
        rho = input_data["rho"]
        epsilon = input_data["epsilon"]
        use_locking = bool(input_data["use_locking"].item())
        dtype_name = str(input_data["dtype_name"].item())

    dtype = var.dtype
    if any(tensor.dtype != dtype for tensor in (accum, accum_update, grad)):
        raise TypeError("var/accum/accum_update/grad must have the same dtype")
    if dtype_name == _BFLOAT16_DTYPE_NAME:
        if dtype != np.dtype(np.uint16):
            raise TypeError("bfloat16 tensors must use uint16 storage")
        lr = np.asarray(lr, dtype=np.uint16).reshape(())
        rho = np.asarray(rho, dtype=np.uint16).reshape(())
        epsilon = np.asarray(epsilon, dtype=np.uint16).reshape(())
    else:
        if dtype.name != dtype_name:
            raise TypeError("serialized tensor dtype does not match dtype_name")
        lr = np.asarray(lr, dtype=dtype).reshape(())
        rho = np.asarray(rho, dtype=dtype).reshape(())
        epsilon = np.asarray(epsilon, dtype=dtype).reshape(())

    import tensorflow as tf

    if dtype_name == _BFLOAT16_DTYPE_NAME:

        def to_tf_tensor(tensor):
            return tf.bitcast(
                tf.convert_to_tensor(tensor, dtype=tf.uint16), tf.bfloat16
            )

        tf_dtype = tf.bfloat16
        output_dtype = np.uint16
    else:

        def to_tf_tensor(tensor):
            return tf.convert_to_tensor(tensor, dtype=tf_dtype)

        tf_dtype = tf.as_dtype(dtype)
        output_dtype = dtype

    var_t = tf.Variable(to_tf_tensor(var), dtype=tf_dtype)
    accum_t = tf.Variable(to_tf_tensor(accum), dtype=tf_dtype)
    accum_update_t = tf.Variable(to_tf_tensor(accum_update), dtype=tf_dtype)

    tf.raw_ops.ResourceApplyAdadelta(
        var=var_t.handle,
        accum=accum_t.handle,
        accum_update=accum_update_t.handle,
        lr=to_tf_tensor(lr),
        rho=to_tf_tensor(rho),
        epsilon=to_tf_tensor(epsilon),
        grad=to_tf_tensor(grad),
        use_locking=use_locking,
    )

    if dtype_name == _BFLOAT16_DTYPE_NAME:
        var_output = tf.bitcast(var_t.read_value(), tf.uint16).numpy()
        accum_output = tf.bitcast(accum_t.read_value(), tf.uint16).numpy()
        accum_update_output = tf.bitcast(accum_update_t.read_value(), tf.uint16).numpy()
    else:
        var_output = var_t.numpy()
        accum_output = accum_t.numpy()
        accum_update_output = accum_update_t.numpy()

    np.savez(
        output_path,
        var=np.asarray(var_output, dtype=output_dtype),
        accum=np.asarray(accum_output, dtype=output_dtype),
        accum_update=np.asarray(accum_update_output, dtype=output_dtype),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    args = parser.parse_args()
    if not args.worker:
        parser.error("--worker is required")
    _run_worker(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
