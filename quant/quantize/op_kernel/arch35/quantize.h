/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file quantize.h
 * \brief quantize kernel base
 */

#ifndef QUANTIZE_H
#define QUANTIZE_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "quantize_tpl_def.h"
#include "quantize_tilingdata.h"

namespace QuantizeOp {
using namespace AscendC;

/**
 * \brief Type mapping helper
 */
template <uint64_t typeEnum>
struct TypeFromEnum;

template <>
struct TypeFromEnum<TPL_INT8> {
    using type = int8_t;
};
template <>
struct TypeFromEnum<TPL_UINT8> {
    using type = uint8_t;
};
template <>
struct TypeFromEnum<TPL_INT32> {
    using type = int32_t;
};
template <>
struct TypeFromEnum<TPL_HIFLOAT8> {
    using type = hifloat8_t;
};
template <>
struct TypeFromEnum<TPL_FP8_E5M2> {
    using type = fp8_e5m2_t;
};
template <>
struct TypeFromEnum<TPL_FP8_E4M3FN> {
    using type = fp8_e4m3fn_t;
};
template <>
struct TypeFromEnum<TPL_BF16> {
    using type = bfloat16_t;
};
template <>
struct TypeFromEnum<TPL_FLOAT> {
    using type = float;
};
template <>
struct TypeFromEnum<TPL_NONE> {
    using type = void;
};

__aicore__ inline constexpr uint32_t GetUbBlockSize()
{
    return 32U;
}

__aicore__ inline constexpr uint32_t GetVRegSize()
{
#if __CCE_AICORE__ == 310
    return AscendC::VECTOR_REG_WIDTH;
#else
    return 256U;
#endif
}

template <typename T, typename T1, typename T2, typename U, uint64_t DivMode, uint64_t RoundMode, uint64_t SqrtMode>
class QuantizeBase {
public:
    __aicore__ inline QuantizeBase(){};

protected:
    __aicore__ inline void ParseTilingData(const QuantizeTilingData* tilingData, QuantizeTilingData& runTilingData);
    __aicore__ inline void ParseCoreBlocks(
        const QuantizeTilingData& runTilingData, int32_t blockIdx, int64_t& blockN, int64_t& blockLen);
    __aicore__ inline void GetXInCopyParams(
        const QuantizeTilingData& runTilingData, int64_t xN, int64_t xLen, DataCopyExtParams& copyParams);
    __aicore__ inline void GetOutCopyParams(
        const QuantizeTilingData& runTilingData, int64_t yN, int64_t yLen, DataCopyExtParams& copyParams);
    __aicore__ inline int64_t CeilAlign(int64_t i, int64_t align);

protected:
    constexpr static int32_t BLOCK_SIZE = GetUbBlockSize();
    constexpr static int32_t BLOCK_NUM_X = BLOCK_SIZE / sizeof(T);
    constexpr static int32_t VEC_INTRI_FP16_NUM = GetVRegSize() / sizeof(half);
    constexpr static int64_t INT4_NUMS_IN_INT8_SPACE = 2;
    constexpr static uint8_t MULTI_COPY_DIM = 2;
    using yCopyDtype = std::conditional_t<IsSameType<U, int4b_t>::value, uint8_t, U>;

protected:
    constexpr static AscendC::MicroAPI::CastTrait CAST_TRAIT_INT8_TO_HALF = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    constexpr static AscendC::MicroAPI::CastTrait CAST_TRAIT_UINT8_TO_HALF = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    constexpr static AscendC::MicroAPI::CastTrait CAST_TRAIT_HALF_TO_FP32 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    constexpr static AscendC::MicroAPI::CastTrait CAST_TRAIT_BF16_TO_FP32 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

    constexpr static AscendC::MicroAPI::CastTrait CAST_TRAIT_INT32_TO_FP32 = {
        AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

    static constexpr AscendC::MicroAPI::CastTrait CAST_TRAIT_FP32_TO_INT16 = []() {
        if constexpr (RoundMode == TPL_ROUND_MODE_ROUND) {
            return AscendC::MicroAPI::CastTrait{
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        } else if constexpr (RoundMode == TPL_ROUND_MODE_FLOOR) {
            return AscendC::MicroAPI::CastTrait{
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_FLOOR};
        } else if constexpr (RoundMode == TPL_ROUND_MODE_CEIL) {
            return AscendC::MicroAPI::CastTrait{
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_CEIL};
        } else if constexpr (RoundMode == TPL_ROUND_MODE_TRUNC) {
            return AscendC::MicroAPI::CastTrait{
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
        } else {
            return AscendC::MicroAPI::CastTrait{
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        }
    }();

    static constexpr AscendC::MicroAPI::CastTrait CAST_TRAIT_INT16_TO_HALF = []() {
        if constexpr (RoundMode == TPL_ROUND_MODE_ROUND) {
            return AscendC::MicroAPI::CastTrait{
                AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::NO_SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        } else if constexpr (RoundMode == TPL_ROUND_MODE_FLOOR) {
            return AscendC::MicroAPI::CastTrait{
                AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::NO_SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_FLOOR};
        } else if constexpr (RoundMode == TPL_ROUND_MODE_CEIL) {
            return AscendC::MicroAPI::CastTrait{
                AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::NO_SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_CEIL};
        } else if constexpr (RoundMode == TPL_ROUND_MODE_TRUNC) {
            return AscendC::MicroAPI::CastTrait{
                AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::NO_SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
        } else {
            return AscendC::MicroAPI::CastTrait{
                AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::NO_SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        }
    }();

    static constexpr AscendC::MicroAPI::CastTrait CAST_TRAIT_HALF_TO_INT8 = []() {
        if constexpr (RoundMode == TPL_ROUND_MODE_ROUND) {
            return AscendC::MicroAPI::CastTrait{
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        } else if constexpr (RoundMode == TPL_ROUND_MODE_FLOOR) {
            return AscendC::MicroAPI::CastTrait{
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_FLOOR};
        } else if constexpr (RoundMode == TPL_ROUND_MODE_CEIL) {
            return AscendC::MicroAPI::CastTrait{
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_CEIL};
        } else if constexpr (RoundMode == TPL_ROUND_MODE_TRUNC) {
            return AscendC::MicroAPI::CastTrait{
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};
        } else {
            return AscendC::MicroAPI::CastTrait{
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        }
    }();

    constexpr static AscendC::MicroAPI::CastTrait CAST_TRAIT_HALF_TO_UINT8 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

    constexpr static AscendC::MicroAPI::CastTrait CAST_TRAIT_FP32_TO_INT32 = {
        AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

    static constexpr AscendC::MicroAPI::CastTrait CAST_TRAIT_FP32_TO_HIFP8 = []() {
        if constexpr (RoundMode == TPL_ROUND_MODE_HYBRID) {
            return AscendC::MicroAPI::CastTrait{
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_HYBRID};
        } else {
            return AscendC::MicroAPI::CastTrait{
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
        }
    }();

    static constexpr AscendC::MicroAPI::CastTrait CAST_TRAIT_FP32_TO_FP8E5M2 = []() {
        return AscendC::MicroAPI::CastTrait{
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    }();

    static constexpr AscendC::MicroAPI::CastTrait CAST_TRAIT_FP32_TO_FP8E4M3 = []() {
        return AscendC::MicroAPI::CastTrait{
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    }();

    static constexpr AscendC::MicroAPI::CastTrait CAST_TRAIT_F16_TO_I8 = []() {
        return AscendC::MicroAPI::CastTrait{
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    }();
};

template <typename T, typename T1, typename T2, typename U, uint64_t DivMode, uint64_t RoundMode, uint64_t SqrtMode>
__aicore__ inline int64_t QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::CeilAlign(int64_t i, int64_t align)
{
    if (align == 0) {
        return i;
    }
    return (i + align - 1) / align * align;
}

template <typename T, typename T1, typename T2, typename U, uint64_t DivMode, uint64_t RoundMode, uint64_t SqrtMode>
__aicore__ inline void QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::ParseTilingData(
    const QuantizeTilingData* tilingData, QuantizeTilingData& runTilingData)
{
    runTilingData.numCore = tilingData->numCore;
    runTilingData.blockAxis = tilingData->blockAxis;
    runTilingData.ubAxis = tilingData->ubAxis;
    runTilingData.dim0 = tilingData->dim0;
    runTilingData.dim1 = tilingData->dim1;
    runTilingData.dim2 = tilingData->dim2;
    runTilingData.blockUnion = tilingData->blockUnion;
    runTilingData.blockFactor = tilingData->blockFactor;
    runTilingData.blockTailFactor = tilingData->blockTailFactor;
    runTilingData.baseN = tilingData->baseN;
    runTilingData.baseLen = tilingData->baseLen;
    runTilingData.hasZeroPoint = tilingData->hasZeroPoint;
    runTilingData.axis = tilingData->axis;
    runTilingData.roundMode = tilingData->roundMode;
    runTilingData.sqrtMode = tilingData->sqrtMode;
}

template <typename T, typename T1, typename T2, typename U, uint64_t DivMode, uint64_t RoundMode, uint64_t SqrtMode>
__aicore__ inline void QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::ParseCoreBlocks(
    const QuantizeTilingData& runTilingData, int32_t blockIdx, int64_t& blockN, int64_t& blockLen)
{
    if (runTilingData.blockAxis == 0) {
        if (blockIdx == runTilingData.numCore - 1) {
            blockN = runTilingData.blockTailFactor;
        } else {
            blockN = runTilingData.blockFactor;
        }
        blockLen = runTilingData.dim1;
    } else if (runTilingData.blockAxis == 1) {
        blockN = runTilingData.dim0;
        if (blockIdx == runTilingData.numCore - 1) {
            blockLen = runTilingData.blockTailFactor;
        } else {
            blockLen = runTilingData.blockFactor;
        }
    }
}

template <typename T, typename T1, typename T2, typename U, uint64_t DivMode, uint64_t RoundMode, uint64_t SqrtMode>
__aicore__ inline void QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::GetXInCopyParams(
    const QuantizeTilingData& runTilingData, int64_t xN, int64_t xLen, DataCopyExtParams& copyParams)
{
    copyParams.blockCount = xN;
    copyParams.blockLen = xLen * sizeof(T);
    if (runTilingData.baseLen > xLen) {
        copyParams.dstStride = (runTilingData.baseLen - xLen) * sizeof(T) / BLOCK_SIZE;
    } else {
        copyParams.dstStride = 0;
    }
    if (runTilingData.dim1 > xLen) {
        copyParams.srcStride = (runTilingData.dim1 - xLen) * sizeof(T);
    } else {
        copyParams.srcStride = 0;
    }
    copyParams.rsv = 0;
}

template <typename T, typename T1, typename T2, typename U, uint64_t DivMode, uint64_t RoundMode, uint64_t SqrtMode>
__aicore__ inline void QuantizeBase<T, T1, T2, U, DivMode, RoundMode, SqrtMode>::GetOutCopyParams(
    const QuantizeTilingData& runTilingData, int64_t yN, int64_t yLen, DataCopyExtParams& copyParams)
{
    int64_t yLenReal = yLen;
    if constexpr (IsSameType<U, int4b_t>::value) {
        yLenReal = yLenReal / INT4_NUMS_IN_INT8_SPACE;
        copyParams.blockLen = yLenReal * sizeof(int8_t);
    } else {
        copyParams.blockLen = yLenReal * sizeof(yCopyDtype);
    }
    copyParams.blockCount = yN;
    if (runTilingData.dim1 > yLen) {
        if constexpr (IsSameType<U, int4b_t>::value) {
            copyParams.dstStride = (runTilingData.dim1 - yLen) * sizeof(yCopyDtype) / INT4_NUMS_IN_INT8_SPACE;
        } else {
            copyParams.dstStride = (runTilingData.dim1 - yLen) * sizeof(yCopyDtype);
        }
    } else {
        copyParams.dstStride = 0;
    }
    if (runTilingData.baseLen > yLenReal) {
        copyParams.srcStride = (runTilingData.baseLen - yLenReal) * sizeof(yCopyDtype) / BLOCK_SIZE;
    } else {
        copyParams.srcStride = 0;
    }
}

} // namespace QuantizeOp

#endif