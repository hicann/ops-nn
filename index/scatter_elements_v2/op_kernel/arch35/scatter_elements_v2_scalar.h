/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCENDC_SCATTER_ELEMENTS_V2_SCALAR_H_
#define ASCENDC_SCATTER_ELEMENTS_V2_SCALAR_H_

#include <type_traits>
#include "scatter_elements_v2_common.h"

namespace ScatterElements {

constexpr uint32_t SCALAR_THREAD_NUM = 512;

template <typename DATA_T>
__simt_callee__ __aicore__ inline DATA_T MultiplyScalarValue(DATA_T lhs, DATA_T rhs)
{
    // Integer multiplication retains the low bits without signed overflow.
    if constexpr (IsSameType<DATA_T, int32_t>::value) {
        return static_cast<DATA_T>(static_cast<uint32_t>(lhs) * static_cast<uint32_t>(rhs));
    } else if constexpr (IsSameType<DATA_T, int64_t>::value) {
        return static_cast<DATA_T>(static_cast<uint64_t>(lhs) * static_cast<uint64_t>(rhs));
    } else {
        return static_cast<DATA_T>(lhs * rhs);
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, uint32_t REDU, uint16_t RANK, uint16_t DIM>
__simt_vf__ __aicore__ LAUNCH_BOUND(SCALAR_THREAD_NUM) inline void SimtComputeScalar(
    __gm__ IDX_T* indices, __gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* dataWorkspace,
    __ubuf__ uint64_t* strides, COMP_T count, __ubuf__ COMP_T* divisionParams)
{
    ScatterOffsetParams<COMP_T, RANK, false> params;
    LoadScatterStrides(params, strides);
#pragma unroll
    for (uint16_t d = 0; d < RANK - 1; ++d) {
        params.magic[d] = divisionParams[d];
        params.shift[d] = divisionParams[RANK - 1 + d];
    }
    using VALUE_T = std::conditional_t<SCATTER_USE_WORKSPACE<DATA_T, REDU>, CAST_T, DATA_T>;
    const VALUE_T value = static_cast<VALUE_T>(updates[0]);
    __gm__ VALUE_T* output;
    if constexpr (SCATTER_USE_WORKSPACE<DATA_T, REDU>) {
        output = dataWorkspace;
    } else {
        output = y;
    }
    for (COMP_T i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        auto offsets = CalcScatterOffsets<RANK, DIM, false>(i, static_cast<COMP_T>(indices[i]), params);
        if constexpr (REDU == REDU_MUL) {
            // Multiply follows the tensor path's non-deterministic update semantics.
            output[offsets.output] = MultiplyScalarValue(output[offsets.output], value);
        } else {
            UpdateScatterValue<REDU>(output + offsets.output, value);
        }
    }
}

} // namespace ScatterElements
#endif // ASCENDC_SCATTER_ELEMENTS_V2_SCALAR_H_
