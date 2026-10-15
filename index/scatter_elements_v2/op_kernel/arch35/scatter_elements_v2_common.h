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
 * \file scatter_elements_v2_common.h
 * \brief scatter_elements
 */
#ifndef ASCENDC_SCATTER_ELEMENTS_V2_COMMON_H_
#define ASCENDC_SCATTER_ELEMENTS_V2_COMMON_H_

#include "kernel_operator.h"

#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"

namespace ScatterElements {
using namespace AscendC;

constexpr int16_t DIM_1 = 1;
constexpr int16_t DIM_2 = 2;
constexpr int16_t DIM_3 = 3;
constexpr int16_t DIM_4 = 4;
constexpr int16_t DIM_5 = 5;
constexpr int16_t DIM_6 = 6;
constexpr int16_t DIM_7 = 7;
constexpr int16_t DIM_8 = 8;
constexpr uint32_t REDU_NONE = 0;
constexpr uint32_t REDU_ADD = 1;
constexpr uint32_t REDU_MUL = 2;
constexpr uint32_t USED_THREAD = 512;
constexpr uint32_t USED_THREAD1024 = 1024;
constexpr uint32_t VECTOR_LENGTH = platform::GetVRegSize();
constexpr uint32_t VL_B32 = VECTOR_LENGTH / sizeof(uint32_t);
constexpr uint32_t PARAM_DIM5_NUM = 8;
constexpr uint32_t PARAM_DIM6_NUM = 10;
constexpr uint32_t PARAM_DIM7_NUM = 12;
constexpr uint32_t PARAM_DIM8_NUM = 14;
constexpr uint32_t PARAM_UB_NUM = 16;
constexpr uint32_t TILING_DATA_UINT64_NUM = 21;
constexpr uint32_t TILING_DATA_UB_NUM = 24;
constexpr int64_t DB_BUFFER = 1;
constexpr int64_t GM_ALIGN = 512;
constexpr int16_t TILING_ARRAY_LEN = 7;
constexpr int16_t TWO_TILING_ARRAY_LEN = 14;
constexpr int16_t THREE_TILING_ARRAY_LEN = 21;
constexpr int64_t THREAD_SUM = 32768;

static constexpr Reg::CastTrait castTraitB8B162B32 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN,
                                                      Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

template <typename COMP_T>
struct ScatterElementsQuickDivParamDim8 {
    COMP_T m0{1};
    COMP_T m1{1};
    COMP_T m2{1};
    COMP_T m3{1};
    COMP_T m4{1};
    COMP_T m5{1};
    COMP_T m6{1};
    COMP_T shift0{1};
    COMP_T shift1{1};
    COMP_T shift2{1};
    COMP_T shift3{1};
    COMP_T shift4{1};
    COMP_T shift5{1};
    COMP_T shift6{1};
};

template <typename COMP_T>
struct ScatterElementsQuickDivParamDim7 {
    COMP_T m0{1};
    COMP_T m1{1};
    COMP_T m2{1};
    COMP_T m3{1};
    COMP_T m4{1};
    COMP_T m5{1};
    COMP_T shift0{1};
    COMP_T shift1{1};
    COMP_T shift2{1};
    COMP_T shift3{1};
    COMP_T shift4{1};
    COMP_T shift5{1};
};

template <typename COMP_T>
struct ScatterElementsQuickDivParamDim6 {
    COMP_T m0{1};
    COMP_T m1{1};
    COMP_T m2{1};
    COMP_T m3{1};
    COMP_T m4{1};
    COMP_T shift0{1};
    COMP_T shift1{1};
    COMP_T shift2{1};
    COMP_T shift3{1};
    COMP_T shift4{1};
};

template <typename COMP_T>
struct ScatterElementsQuickDivParamDim5 {
    COMP_T m0{1};
    COMP_T m1{1};
    COMP_T m2{1};
    COMP_T m3{1};
    COMP_T shift0{1};
    COMP_T shift1{1};
    COMP_T shift2{1};
    COMP_T shift3{1};
};

template <typename DATA_T, typename CAST_T>
__simd_vf__ inline void CastToInt32Vf(__ubuf__ DATA_T* srcAddr, __ubuf__ CAST_T* dstAddr, uint32_t dataLen,
                                      uint16_t loopTimes)
{
    Reg::RegTensor<DATA_T> srcValue;
    Reg::RegTensor<CAST_T> dstValue;
    Reg::MaskReg preg;
    uint32_t sregMask = dataLen;
    for (uint16_t j = 0; j < loopTimes; j++) {
        preg = Reg::UpdateMask<uint32_t>(sregMask);
        if constexpr (IsSameType<DATA_T, int16_t>::value) {
            Reg::LoadAlign<DATA_T, Reg::LoadDist::DIST_UNPACK_B16>(srcValue, srcAddr + VL_B32 * j);
        } else {
            Reg::LoadAlign<DATA_T, Reg::LoadDist::DIST_UNPACK4_B8>(srcValue, srcAddr + VL_B32 * j);
        }
        Reg::Cast<CAST_T, DATA_T, castTraitB8B162B32>(dstValue, srcValue, preg);
        Reg::StoreAlign<CAST_T, Reg::StoreDist::DIST_NORM>(dstAddr + VL_B32 * j, dstValue, preg);
    }
}

template <typename DATA_T, typename CAST_T>
__simd_vf__ inline void CastToOriginVf(__ubuf__ CAST_T* srcAddr, __ubuf__ DATA_T* dstAddr, uint32_t dataLen,
                                       uint16_t loopTimes)
{
    Reg::RegTensor<CAST_T> srcValue;
    Reg::MaskReg preg;
    uint32_t sregMask = dataLen;
    for (uint16_t j = 0; j < loopTimes; j++) {
        preg = Reg::UpdateMask<uint32_t>(sregMask);
        Reg::LoadAlign<CAST_T, Reg::LoadDist::DIST_NORM>(srcValue, srcAddr + VL_B32 * j);
        if constexpr (IsSameType<DATA_T, int16_t>::value) {
            Reg::StoreAlign<DATA_T, Reg::StoreDist::DIST_PACK_B32>(dstAddr + VL_B32 * j,
                                                                   (Reg::RegTensor<DATA_T>&)srcValue, preg);
        } else {
            Reg::StoreAlign<DATA_T, Reg::StoreDist::DIST_PACK4_B32>(dstAddr + VL_B32 * j,
                                                                    (Reg::RegTensor<DATA_T>&)srcValue, preg);
        }
    }
}

template <typename DATA_T, uint32_t REDU>
constexpr bool SCATTER_USE_WORKSPACE = REDU == REDU_ADD &&
                                       (IsSameType<DATA_T, int8_t>::value || IsSameType<DATA_T, uint8_t>::value ||
                                        IsSameType<DATA_T, int16_t>::value);

template <typename COMP_T, uint16_t RANK, bool NEED_UPDATES>
struct ScatterOffsetParams {
    COMP_T dataStride[RANK];
    COMP_T indicesStride[RANK];
    COMP_T updatesStride[NEED_UPDATES ? RANK : 1];
    COMP_T magic[RANK];
    COMP_T shift[RANK];
};

template <typename COMP_T, uint16_t RANK, bool NEED_UPDATES>
__simt_callee__ __aicore__ inline void LoadScatterStrides(ScatterOffsetParams<COMP_T, RANK, NEED_UPDATES>& params,
                                                          __ubuf__ uint64_t* strides)
{
#pragma unroll
    for (uint16_t d = 0; d < RANK - 1; ++d) {
        params.dataStride[d] = static_cast<COMP_T>(strides[d]);
        params.indicesStride[d] = static_cast<COMP_T>(strides[TILING_ARRAY_LEN + d]);
        if constexpr (NEED_UPDATES) {
            params.updatesStride[d] = static_cast<COMP_T>(strides[TWO_TILING_ARRAY_LEN + d]);
        }
    }
    params.dataStride[RANK - 1] = 1;
    if constexpr (NEED_UPDATES) {
        params.updatesStride[RANK - 1] = 1;
    }
}

template <typename COMP_T>
struct ScatterOffsets {
    COMP_T output;
    COMP_T updates;
};

// Decompose the original indices coordinate once. Only the output coordinate
// substitutes indexValue on DIM; tensor updates retain the original coordinate.
template <uint16_t RANK, uint16_t DIM, bool NEED_UPDATES, typename COMP_T>
__simt_callee__ __aicore__ inline ScatterOffsets<COMP_T> CalcScatterOffsets(
    COMP_T linearIndex, COMP_T indexValue, const ScatterOffsetParams<COMP_T, RANK, NEED_UPDATES>& params)
{
    ScatterOffsets<COMP_T> offsets{indexValue * params.dataStride[DIM], 0};
    COMP_T remaining = linearIndex;
#pragma unroll
    for (uint16_t d = 0; d < RANK - 1; ++d) {
        COMP_T coordinate = Simt::UintDiv(remaining, params.magic[d], params.shift[d]);
        remaining -= coordinate * params.indicesStride[d];
        if (d != DIM) {
            offsets.output += coordinate * params.dataStride[d];
        }
        if constexpr (NEED_UPDATES) {
            offsets.updates += coordinate * params.updatesStride[d];
        }
    }
    if constexpr (DIM != RANK - 1) {
        offsets.output += remaining;
    }
    if constexpr (NEED_UPDATES) {
        offsets.updates += remaining;
    }
    return offsets;
}

template <uint32_t REDU, typename VALUE_T>
__simt_callee__ __aicore__ inline void UpdateScatterValue(__gm__ VALUE_T* output, VALUE_T value)
{
    if constexpr (REDU == REDU_ADD) {
        asc_atomic_add(output, value);
    } else if constexpr (REDU == REDU_MUL) {
        *output = *output * value;
    } else {
        *output = value;
    }
}

} // namespace ScatterElements
#endif // ASCENDC_SCATTER_ELEMENTS_V2_COMMON_H_
