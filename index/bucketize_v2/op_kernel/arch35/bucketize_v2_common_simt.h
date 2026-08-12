/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bucketize_v2_common_simt.h
 * \brief
 */
#ifndef BUCKETIZE_V2_COMMON_SIMT_H
#define BUCKETIZE_V2_COMMON_SIMT_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "bucketize_v2_struct.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

namespace BucketizeV2 {
using namespace AscendC;

constexpr uint32_t THREAD_DIM_2048 = 2048;

template <typename X_T, typename B_T, typename Y_T, typename INDICES_T = int64_t, bool RIGHT = false>
__simt_callee__ __aicore__ inline Y_T InnerBinaryQuery(X_T value, Y_T start, Y_T end, __gm__ B_T* bound,
                                                       int64_t innerMaxIter)
{
    INDICES_T left = start;
    INDICES_T right = end;
    for (int64_t i = 0; i < innerMaxIter; i++) {
        if (left >= right) {
            break;
        }
        INDICES_T mid = left + ((right - left) >> 1);
        B_T midValue = bound[mid];
        bool cond = false;
        if constexpr (RIGHT) {
            cond = !(midValue > value);
        } else {
            cond = !(midValue >= value);
        }
        left = cond ? mid + 1 : left;
        right = cond ? right : mid;
    }
    return static_cast<Y_T>(left);
}

} // namespace BucketizeV2
#endif // BUCKETIZE_V2_COMMON_SIMT_H
