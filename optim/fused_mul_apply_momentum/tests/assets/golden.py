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
"""TensorFlow multiply and ApplyMomentum reference; no final dtype downcast."""

from typing import ClassVar
import numpy
import tensorflow as tf

__spec__ = {"fused_mul_apply_momentum": "FusedMulApplyMomentumTestSpec"}
_L1 = {
    name: {"standard": "cross_check", "level": "L1"} for name in ("float16", "float32")
}


def _numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return numpy.asarray(value)


def _compute(var, accum, lr, x1, momentum, x2, use_nesterov, device):
    var, accum, lr, x1, momentum, x2 = map(_numpy, (var, accum, lr, x1, momentum, x2))
    if var.size == 0:
        return [var.copy(), accum.copy()]
    graph, feeds, variable, slot, initialize, seeds, train = _build_graph(
        [value.shape for value in (var, accum, lr, x1, momentum, x2)],
        [value.dtype for value in (var, accum, lr, x1, momentum, x2)],
        use_nesterov,
        device,
    )
    config = tf.compat.v1.ConfigProto(
        allow_soft_placement=False,
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
    )
    config.gpu_options.allow_growth = True
    if device == "/CPU:0":
        config.device_count["GPU"] = 0
    feed = dict(zip(feeds, (var, accum, lr, x1, momentum, x2)))
    with tf.compat.v1.Session(graph=graph, config=config) as session:
        session.run(initialize)
        session.run(seeds, feed_dict=feed)
        session.run(train, feed_dict=feed)
        return session.run([variable, slot])


def _build_graph(shapes, dtypes, use_nesterov, device):
    # Runtime feeds avoid embedding multi-GiB tensors in a GraphDef/TensorProto.
    graph = tf.Graph()
    with graph.as_default(), tf.device(device):
        feeds = [
            tf.compat.v1.placeholder(tf.as_dtype(dtype), shape=shape)
            for shape, dtype in zip(shapes, dtypes)
        ]
        var, accum, lr, x1, momentum, x2 = feeds
        variable = tf.compat.v1.get_variable(
            "var",
            shape=shapes[0],
            dtype=tf.as_dtype(dtypes[0]),
            initializer=tf.zeros_initializer(),
            use_resource=False,
        )
        grad = tf.multiply(x1, tf.reshape(x2, ()))
        slot = tf.compat.v1.get_variable(
            "momentum",
            shape=shapes[0],
            dtype=tf.as_dtype(dtypes[0]),
            initializer=tf.zeros_initializer(),
            trainable=False,
            use_resource=False,
        )
        train = tf.raw_ops.ApplyMomentum(
            var=variable,
            accum=slot,
            lr=tf.cast(tf.reshape(lr, ()), variable.dtype.base_dtype),
            grad=grad,
            momentum=tf.cast(tf.reshape(momentum, ()), variable.dtype.base_dtype),
            use_nesterov=bool(use_nesterov),
            use_locking=False,
        )
        initialize = tf.compat.v1.global_variables_initializer()
        seeds = [variable.assign(var), slot.assign(accum)]
    return graph, feeds, variable, slot, initialize, seeds, train


class _TensorFlowMomentum:
    def __init__(self, *, use_nesterov=False, _ttk_geir_output_dtype=None, **kwargs):
        self.use_nesterov = use_nesterov
        self.output_dtype = _ttk_geir_output_dtype

    def __call__(self, var, accum, lr, x1, momentum, x2, **kwargs):
        outputs = _compute(
            var,
            accum,
            lr,
            x1,
            momentum,
            x2,
            kwargs.get("use_nesterov", self.use_nesterov),
            "/GPU:0",
        )
        output_dtype = kwargs.get("_ttk_geir_output_dtype", self.output_dtype)
        if output_dtype is not None:
            with tf.device("/GPU:0"):
                return [
                    tf.cast(value, tf.as_dtype(output_dtype)).numpy()
                    for value in outputs
                ]
        return outputs


def _geir_reference_inputs(*, inputs, input_dtypes, testcase_name):
    """Reproduce verified GE input casts; native Kernel inputs are unaffected."""
    import json
    from pathlib import Path

    if testcase_name not in {
        "FusedMulApplyMomentum_L2_exc_dtype_001",
        "FusedMulApplyMomentum_L2_exc_dtype_004",
        "FusedMulApplyMomentum_L2_exc_dtype_007",
        "FusedMulApplyMomentum_L2_exc_dtype_010",
        "FusedMulApplyMomentum_L2_exc_dtype_013",
        "FusedMulApplyMomentum_L2_exc_dtype_016",
    }:
        return None
    path = Path(__file__).with_name("geir_case_policies.json")
    if not path.exists():
        raise FileNotFoundError("Missing GEIR reference input policy: " + str(path))
    policy = json.loads(path.read_text()).get(testcase_name, {})
    if policy.get("reference_adapter") != "input_cast_before_promote":
        return None
    if tuple(input_dtypes) != tuple(policy["input_dtypes"]):
        raise ValueError("GEIR inputs no longer match the verified dtype policy")
    dtype = policy["compute_dtype"]
    prepared = {
        "inputs": tuple(numpy.asarray(value).astype(dtype) for value in inputs),
        "input_dtypes": (dtype,) * len(inputs),
        "output_dtypes": (policy.get("output_dtype", dtype),) * 2,
    }
    if policy.get("output_dtype"):
        prepared["third_party_attributes"] = {
            "_ttk_geir_output_dtype": policy["output_dtype"]
        }
    return prepared


class FusedMulApplyMomentumTestSpec:
    @staticmethod
    def golden(var, accum, lr, x1, momentum, x2, use_nesterov=False, **kwargs):
        return _compute(var, accum, lr, x1, momentum, x2, use_nesterov, "/CPU:0")

    third_party: ClassVar = {"tf": _TensorFlowMomentum}
    tolerance: ClassVar = _L1
    geir_reference_inputs = staticmethod(_geir_reference_inputs)
