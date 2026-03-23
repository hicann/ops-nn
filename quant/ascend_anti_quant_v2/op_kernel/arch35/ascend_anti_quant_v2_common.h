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
 * \file ascend_anti_quant_v2_common.h
 * \brief ascendantiquantv2 kernel base
 */

#ifndef QUANTIZE_H
#define QUANTIZE_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"

namespace AscendAntiQuantV2 {
using namespace AscendC;
/**
 * \brief Type mapping helper
 */

__aicore__ inline constexpr uint32_t GetUbBlockSize()
{
    return 32U;
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
class AscendAntiQuantV2Base {
public:
    __aicore__ inline AscendAntiQuantV2Base(){};

protected:
    __aicore__ inline void GetXInCopyParams(
        int64_t dim1, int64_t baseLen, int64_t xN, int64_t xLen, DataCopyExtParams& copyParams);
    __aicore__ inline void GetOutCopyParams(
        int64_t dim1, int64_t baseLen, int64_t yN, int64_t yLen, DataCopyExtParams& copyParams);
    __aicore__ inline int64_t CeilAlign(int64_t i, int64_t align);

protected:
    constexpr static int32_t BLOCK_SIZE = GetUbBlockSize();
    constexpr static int64_t INT4_NUMS_IN_INT8_SPACE = 2;
    constexpr static uint8_t MULTI_COPY_DIM = 2;
    using xCopyDtype = std::conditional_t<IsSameType<T, int4b_t>::value, uint8_t, T>;

protected:
    constexpr static AscendC::Reg::CastTrait CAST_TRAIT_INT8_TO_HALF = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
        AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    constexpr static AscendC::Reg::CastTrait CAST_TRAIT_HALF_TO_FP32 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
        AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    constexpr static AscendC::Reg::CastTrait CAST_TRAIT_BF16_TO_FP32 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
        AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    constexpr static AscendC::Reg::CastTrait CAST_TRAIT_HIFP8_TO_FP32 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
        AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    constexpr static AscendC::Reg::CastTrait CAST_TRAIT_FP8E5M2_TO_FP32 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
        AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    constexpr static AscendC::Reg::CastTrait CAST_TRAIT_FP8E4M3_TO_FP32 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
        AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    constexpr static AscendC::Reg::CastTrait CAST_TRAIT_FP32_TO_HALF = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
        AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

    constexpr static AscendC::Reg::CastTrait CAST_TRAIT_FP32_TO_BF16 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT,
        AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
};

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline int64_t AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::CeilAlign(int64_t i, int64_t align)
{
    if (align == 0) {
        return i;
    }
    return (i + align - 1) / align * align;
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::GetOutCopyParams(
    int64_t dim1, int64_t baseLen, int64_t yN, int64_t yLen, DataCopyExtParams& copyParams)
{
    copyParams.blockCount = yN;
    copyParams.blockLen = yLen * sizeof(U);
    if (dim1 > yLen) {
        copyParams.dstStride = (dim1 - yLen) * sizeof(U);
    } else {
        copyParams.dstStride = 0;
    }
    if (baseLen > yLen) {
        copyParams.srcStride = (baseLen - yLen) * sizeof(U) / BLOCK_SIZE;
    } else {
        copyParams.srcStride = 0;
    }
    copyParams.rsv = 0;
}

template <typename T, typename T1, typename T2, typename U, uint64_t SqrtMode>
__aicore__ inline void AscendAntiQuantV2Base<T, T1, T2, U, SqrtMode>::GetXInCopyParams(
    int64_t dim1, int64_t baseLen, int64_t xN, int64_t xLen, DataCopyExtParams& copyParams)
{
    int64_t xLenReal = xLen;
    if constexpr (IsSameType<T, int4b_t>::value) {
        xLenReal = xLenReal / INT4_NUMS_IN_INT8_SPACE;
        copyParams.blockLen = xLenReal * sizeof(xCopyDtype);
    } else {
        copyParams.blockLen = xLenReal * sizeof(T);
    }
    copyParams.blockCount = xN;
    if (dim1 > xLen) {
        if constexpr (IsSameType<T, int4b_t>::value) {
            copyParams.srcStride = (dim1 - xLen) * sizeof(xCopyDtype) / INT4_NUMS_IN_INT8_SPACE;
        } else {
            copyParams.srcStride = (dim1 - xLen) * sizeof(T);
        }
    } else {
        copyParams.srcStride = 0;
    }
    if (baseLen > xLenReal) {
        copyParams.dstStride = (baseLen - xLenReal) * sizeof(xCopyDtype) / BLOCK_SIZE;
    } else {
        copyParams.dstStride = 0;
    }
}

} // namespace AscendAntiQuantV2

#endif