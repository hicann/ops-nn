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
 * \file bucketize_v2_simt.h
 * \brief
 */
#ifndef BUCKETIZE_V2_SIMT_H
#define BUCKETIZE_V2_SIMT_H

#include "bucketize_v2_common_simt.h"
#include "bucketize_v2_struct.h"

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"

namespace BucketizeV2 {
using namespace AscendC;

template <typename X_T, typename B_T, typename Y_T, typename INDICES_T, bool RIGHT = false,
          uint32_t THREAD_NUM_LAUNCH_BOUND = 1024>
__simt_vf__ __aicore__ __launch_bounds__(THREAD_NUM_LAUNCH_BOUND) inline void BucketizeSimtImpl(
    __gm__ X_T* value, __gm__ B_T* bound, __gm__ Y_T* out, int64_t dataLen, int64_t maxIter, int64_t boundSize)
{
    for (int64_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < dataLen; idx += blockDim.x * gridDim.x) {
        out[idx] = InnerBinaryQuery<X_T, B_T, Y_T, INDICES_T, RIGHT>(value[idx], 0, boundSize, bound, maxIter);
    }
}

template <typename X_T, typename B_T, typename Y_T, typename INDICES_T, bool RIGHT = false>
__aicore__ inline void BucketizeSimt(__gm__ X_T* value, __gm__ B_T* bound, __gm__ Y_T* out, int64_t dataLen,
                                     int64_t maxIter, int64_t boundSize)
{
    asc_vf_call<BucketizeSimtImpl<X_T, B_T, Y_T, INDICES_T, RIGHT, THREAD_DIM_2048>>(
        dim3{THREAD_DIM_2048}, (__gm__ X_T*)(value), (__gm__ B_T*)(bound), (__gm__ Y_T*)(out), dataLen, maxIter,
        boundSize);
}

} // namespace BucketizeV2
#endif // BUCKETIZE_V2_SIMT_H
