#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

import torch


class AddRowRangesComposeSpec:
    """Class form — small-op composition"""

    # -- golden — function form --
    def golden(x, src, indices, **kwargs):
        """
        Golden function for add_row_ranges.
        All the parameters (names and order) follow SE doc prototype definition without outputs.
        All the input Tensors are numpy.ndarray.

        Algorithm:
            For each row r and column c of the output matrix x_out:
                x_out(r, c) = x(r, c) + sum(src(j, c)), j in [indices[r, 0], indices[r, 1])

        Indices clipping:
            1. start >= end || start == -1 || end == -1 -> skip
            2. negative index: start<0 -> max(K+start, 0); end<0 -> max(K+end, 0)
            3. clip to [0, K]: start>K -> K; end>K -> K
            4. post-clip empty range: start >= end -> skip

        Args:
            x: numpy.ndarray, shape=(M, N), dtype=float16/float32
            src: numpy.ndarray, shape=(K, N), dtype=float16/float32 (same as x)
            indices: numpy.ndarray, shape=(M, 2), dtype=int32
            **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                      full_soc_version, short_soc_version, testcase_name

        Returns:
            Output tensor x_out, shape=(M, N), dtype=same as x
        """
        orig_dtype = x.dtype
        x_out = x.copy()

        m = x.shape[0]
        n = x.shape[1]
        k = src.shape[0]

        for r in range(m):
            start = int(indices[r, 0])
            end = int(indices[r, 1])

            # 1. empty range check
            if start >= end or start == -1 or end == -1:
                continue

            # 2. negative index resolution
            if start < 0:
                start = max(start + k, 0)
            if end < 0:
                end = max(end + k, 0)

            # 3. clip to [0, K]
            if start > k:
                start = k
            if end > k:
                end = k

            # 4. post-clip empty range check
            if start >= end:
                continue

            # 5. sequential accumulation (original dtype, no precision promotion)
            for c in range(n):
                acc = orig_dtype.type(0)
                for j in range(start, end):
                    acc += src[j, c]
                x_out[r, c] = x[r, c] + acc

        return x_out

    # -- third_party — dict multi-vendor --
    class ThirdPartyImpl:
        def __init__(self, **kwargs):
            pass

        def __call__(self, x, src, indices, **kwargs):
            # add_row_ranges has no direct torch competitor, use torch functions
            # x: (M, N), src: (K, N), indices: (M, 2)
            output = x.clone()
            m = x.shape[0]
            k = src.shape[0]

            for r in range(m):
                start = int(indices[r, 0].item())
                end = int(indices[r, 1].item())

                # empty range check
                if start >= end or start == -1 or end == -1:
                    continue

                # negative index resolution
                if start < 0:
                    start = max(start + k, 0)
                if end < 0:
                    end = max(end + k, 0)

                # clip to [0, K]
                if start > k:
                    start = k
                if end > k:
                    end = k

                # post-clip empty range check
                if start >= end:
                    continue

                # torch.sum for row range, accumulate to output
                row_sum = torch.sum(src[start:end, :], dim=0)
                output[r, :] = output[r, :] + row_sum

            return output

    third_party = {
        "torch": ThirdPartyImpl,
    }

    tolerance = {
        "float32": {"standard": "cross_check", "level": "L1"},
        "float16": {"standard": "cross_check", "level": "L1"},
    }


# Explicit registration: class names use *Spec suffix (not *TestSpec),
# so __spec__ dict is needed for discovery.
__spec__ = {
    "add_row_ranges": AddRowRangesComposeSpec,
}
