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
 * \file ascend quant.h
 * \brief ascend quant kernel base
 */

#ifndef ASCEND_QUANT_H
#define ASCEND_QUANT_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "../inc/platform.h"
#include "ascend_quant_struct.h"
#include "ascend_quant_tilingdata.h"
#include "op_kernel/platform_util.h"

namespace AscendQuantOp {
using namespace AscendC;
using namespace Ops::Base;

template <typename T, typename U, uint64_t RoundMode>
class AscendQuantBase {
public:
    __aicore__ inline AscendQuantBase(){};

protected:
    __aicore__ inline void ParseTilingData(
        const AscendQuantTilingData* tilingData, AscendQuantTilingData& runTilingData);
    __aicore__ inline void ParseCoreBlocks(
        const AscendQuantTilingData& runTilingData, int32_t blockIdx, int64_t& blockLen);
    __aicore__ inline void GetXInCopyParams(
        const AscendQuantTilingData& runTilingData, int64_t xLen, AscendC::DataCopyExtParams& copyParams);
    __aicore__ inline void GetOutCopyParams(
        const AscendQuantTilingData& runTilingData, int64_t yLen, AscendC::DataCopyExtParams& copyParams);

protected:
    constexpr static int32_t BLOCK_SIZE = GetUbBlockSize();
    constexpr static int64_t INT4_NUMS_IN_INT8_SPACE = 2;
    using yCopyDtype = std::conditional_t<IsSameType<U, int4b_t>::value, uint8_t, U>;

protected:
    constexpr static AscendC::MicroAPI::CastTrait CAST_TRAIT_HALF_TO_FP32 = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

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

    static constexpr AscendC::MicroAPI::CastTrait CAST_TRAIT_F16_TO_I8 = []() {
        return AscendC::MicroAPI::CastTrait{
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    }();
};

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void AscendQuantBase<T, U, RoundMode>::ParseTilingData(
    const AscendQuantTilingData* tilingData, AscendQuantTilingData& runTilingData)
{
    runTilingData.numCore = tilingData->numCore;
    runTilingData.dim0 = tilingData->dim0;
    runTilingData.blockFactor = tilingData->blockFactor;
    runTilingData.blockTailFactor = tilingData->blockTailFactor;
    runTilingData.baseLen = tilingData->baseLen;
    runTilingData.roundMode = tilingData->roundMode;
    runTilingData.scale = tilingData->scale;
    runTilingData.offset = tilingData->offset;
}

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void AscendQuantBase<T, U, RoundMode>::ParseCoreBlocks(
    const AscendQuantTilingData& runTilingData, int32_t blockIdx, int64_t& blockLen)
{
    if (blockIdx == runTilingData.numCore - 1) {
        blockLen = runTilingData.blockTailFactor;
    } else {
        blockLen = runTilingData.blockFactor;
    }
}

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void AscendQuantBase<T, U, RoundMode>::GetXInCopyParams(
    const AscendQuantTilingData& runTilingData, int64_t xLen, DataCopyExtParams& copyParams)
{
    copyParams.blockCount = 1;
    copyParams.blockLen = xLen * sizeof(T);
    if (runTilingData.baseLen > xLen) {
        copyParams.dstStride = (runTilingData.baseLen - xLen) * sizeof(T) / BLOCK_SIZE;
    } else {
        copyParams.dstStride = 0;
    }
    if (runTilingData.dim0 > xLen) {
        copyParams.srcStride = (runTilingData.dim0 - xLen) * sizeof(T);
    } else {
        copyParams.srcStride = 0;
    }
    copyParams.rsv = 0;
}

template <typename T, typename U, uint64_t RoundMode>
__aicore__ inline void AscendQuantBase<T, U, RoundMode>::GetOutCopyParams(
    const AscendQuantTilingData& runTilingData, int64_t yLen, DataCopyExtParams& copyParams)
{
    int64_t yLenReal = yLen;
    if constexpr (IsSameType<U, int4b_t>::value) {
        yLenReal = yLenReal / INT4_NUMS_IN_INT8_SPACE;
        copyParams.blockLen = yLenReal * sizeof(int8_t);
    } else {
        copyParams.blockLen = yLenReal * sizeof(yCopyDtype);
    }
    copyParams.blockCount = 1;
    if (runTilingData.dim0 > yLen) {
        if constexpr (IsSameType<U, int4b_t>::value) {
            copyParams.dstStride = (runTilingData.dim0 - yLen) * sizeof(yCopyDtype) / INT4_NUMS_IN_INT8_SPACE;
        } else {
            copyParams.dstStride = (runTilingData.dim0 - yLen) * sizeof(yCopyDtype);
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

} // namespace AscendQuantOp

#endif