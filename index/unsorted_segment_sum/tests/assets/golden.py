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

import torch

__spec__ = {
    "unsorted_segment_sum": "UnsortedSegmentSumKernelSpec",
    "aclnnUnsortedSegmentSum": "AclnnUnsortedSegmentSumSpec",
}

__golden__ = {
    "kernel": {"unsorted_segment_sum": "unsorted_segment_sum_golden"},
    "aclnn": {"aclnnUnsortedSegmentSum": "aclnn_unsorted_segment_sum_golden"},
}


def unsorted_segment_sum_golden(
    x, segment_ids, num_segments, *, is_preprocessed=False, check_ids=False, **kwargs
):
    """
    Golden function for unsorted_segment_sum.
    All the parameters (names and order) follow @unsorted_segment_sum_def.cpp without outputs.
    All the input Tensors are numpy.ndarray.

    Args:
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name

    Returns:
        Output tensor
    """
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
    out = tf.unsorted_segment_sum(x, segment_ids, int(num_segments))
    with tf.Session() as sess:
        res = sess.run(out)
    return [res]


def aclnn_unsorted_segment_sum_golden(x, segmentIds, numSegments, out=None, **kwargs):
    """
    Aclnn golden for aclnnUnsortedSegmentSum.
    Parameters follow @aclnnUnsortedSegmentSumGetWorkspaceSize without workspaceSize & executor.
    All the input Tensors are torch.Tensor.
    """
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()

    x_dtype = x.dtype
    if "bfloat16" in str(x_dtype):
        x_np = x.to(torch.float32).numpy()
    else:
        x_np = x.numpy()
    ids_np = segmentIds.numpy()

    out_tf = tf.unsorted_segment_sum(x_np, ids_np, int(numSegments))
    with tf.Session() as sess:
        res = sess.run(out_tf)

    return [torch.from_numpy(res).to(x_dtype)]


def aclnn_unsorted_segment_sum_customize_inputs(
    x, segmentIds, numSegments, out=None, **kwargs
):
    """
    Customize inputs for aclnnUnsortedSegmentSum.
    All the input Tensors are torch.Tensor.
    """
    num_segments = int(numSegments)
    if num_segments > 0:
        new_ids = torch.randint(0, num_segments, segmentIds.shape).to(segmentIds.dtype)
        segmentIds.copy_(new_ids)


class UnsortedSegmentSumKernelSpec:
    golden = unsorted_segment_sum_golden


class AclnnUnsortedSegmentSumSpec:
    golden = aclnn_unsorted_segment_sum_golden
    customize_inputs = aclnn_unsorted_segment_sum_customize_inputs
    tolerance = {"float32": {"standard": "mix_tolerance", "rtol": 0.0001}}
