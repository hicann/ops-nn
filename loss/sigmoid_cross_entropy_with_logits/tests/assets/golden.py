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

import numpy as np
import tensorflow as tf

__golden__ = {
    "kernel": {
        "sigmoid_cross_entropy_with_logits": "sigmoid_cross_entropy_with_logits_golden"
    }
}


def sigmoid_cross_entropy_with_logits_golden(predict, target, **kwargs):
    """
    Golden function for sigmoid_cross_entropy_with_logits.
    All the parameters (names and order) follow
    @sigmoid_cross_entropy_with_logits_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Reference: the canonical framework op
        tf.nn.sigmoid_cross_entropy_with_logits(logits=predict, labels=target)
    which returns the element-wise loss (no reduction), equivalent to the stable
        loss = max(logits, 0) - logits * labels + log1p(exp(-|logits|))
    We call the framework op directly (not a self-written NumPy formula) so the
    golden matches the standard SigmoidCrossEntropyWithLogits contract. The op
    runs on CPU and is independent of the tested NPU operator.

    Contract (ordinary op, NOT V2):
        - inputs are predict (logits) and target (labels), same shape/dtype;
        - output is the element-wise loss, shape == predict shape;
        - NO mean/sum reduction, NO weight / pos_weight / reduction attribute,
          and V2 (SigmoidCrossEntropyWithLogitsV2) semantics are not mixed in;
        - registration name stays sigmoid_cross_entropy_with_logits.

    NumPy is used only for dtype/shape adaptation around the framework op: the
    custom golden path is not auto-promoted by TTK, so float16/bfloat16 inputs
    are upcast to float32 before the op for a stable high-precision reference
    (fp16-direct vs fp64 ref ~7e-3, fp32-upcast ~9e-7), then the loss is written
    back to the predict dtype to match the IR output dtype.

    Args:
        predict: logits, dtype float16 / float / bfloat16.
        target:  labels, same shape and dtype as predict.
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        [loss] following the IR output order; loss shape == predict shape,
        dtype == predict dtype.
    """
    predict_dtype = predict.dtype

    # Upcast to float32 for a stable, dtype-agnostic reference (see docstring),
    # then call the framework op; NumPy only adapts dtype in/out.
    p = predict.astype(np.float32)
    t = target.astype(np.float32)

    loss = np.asarray(tf.nn.sigmoid_cross_entropy_with_logits(logits=p, labels=t))

    loss = loss.astype(predict_dtype, copy=False)
    return [loss]
