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
 * \file quant_update_scatter_base.h
 * \brief quant_update_scatter kernel base file
 */
#ifndef QUANT_UPDATE_SCATTER_BASE_H_
#define QUANT_UPDATE_SCATTER_BASE_H_

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "quant_update_scatter_tpl_def.h"

namespace QuantUpdateScatter
{
using namespace AscendC;

/**
 * \brief Type mapping helper
 */
template <uint64_t typeEnum>
struct TypeFromEnum;

template <>
struct TypeFromEnum<TPL_INT32> {
    using type = int32_t;
};
template <>
struct TypeFromEnum<TPL_BF16> {
    using type = bfloat16_t;
};
template <>
struct TypeFromEnum<TPL_NONE> {
    // void 会报错：cannot form a reference to 'void'，因此用bool替代void
    using type = bool;
};

template <typename VarType, typename IndicesType, typename UpdatesType, typename ScalesType, typename OffsetsType,
          uint64_t DivMode, uint64_t CastRoundMode>
class QuantUpdateScatterBase
{
public:
    __aicore__ inline QuantUpdateScatterBase(){};

protected:
    constexpr static int64_t BUFFER_NUM = 2;
    constexpr static int64_t BLOCK_SIZE = 32;
    constexpr static int64_t INDICES_SHAPE_RANK_2 = 2;
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
        return AscendC::MicroAPI::CastTrait{AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
                                            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    }();

    static constexpr AscendC::MicroAPI::CastTrait CAST_TRAIT_INT16_TO_HALF = []() {
        return AscendC::MicroAPI::CastTrait{AscendC::MicroAPI::RegLayout::UNKNOWN, AscendC::MicroAPI::SatMode::NO_SAT,
                                            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    }();

    static constexpr AscendC::MicroAPI::CastTrait CAST_TRAIT_HALF_TO_INT8 = []() {
        return AscendC::MicroAPI::CastTrait{AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
                                            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    }();

    static constexpr AscendC::MicroAPI::CastTrait CAST_TRAIT_FP32_TO_HIFP8 = []() {
        if constexpr (CastRoundMode == TPL_ROUND_MODE_HYBRID) {
            return AscendC::MicroAPI::CastTrait{AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
                                                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_HYBRID};
        } else {
            return AscendC::MicroAPI::CastTrait{AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
                                                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
        }
    }();

    static constexpr AscendC::MicroAPI::CastTrait CAST_TRAIT_FP32_TO_FP8E5M2 = []() {
        return AscendC::MicroAPI::CastTrait{AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
                                            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    }();

    static constexpr AscendC::MicroAPI::CastTrait CAST_TRAIT_FP32_TO_FP8E4M3 = []() {
        return AscendC::MicroAPI::CastTrait{AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
                                            AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    }();

    __aicore__ inline void ParseTilingData(const QuantUpdateScatterTilingData* tilingData,
                                           QuantUpdateScatterTilingData& runTilingData)
    {
        runTilingData.coreNum = tilingData->coreNum;
        runTilingData.eachCoreBsNum = tilingData->eachCoreBsNum;
        runTilingData.lastCoreBsNum = tilingData->lastCoreBsNum;
        runTilingData.srcBsStride = tilingData->srcBsStride;
        runTilingData.dstBsStride = tilingData->dstBsStride;
        runTilingData.indexElements = tilingData->indexElements;
        runTilingData.varDim1 = tilingData->varDim1;
        runTilingData.varDim2 = tilingData->varDim2;
        runTilingData.varDim3 = tilingData->varDim3;
        runTilingData.innerLoopEle = tilingData->innerLoopEle;
        runTilingData.innerLoopTimes = tilingData->innerLoopTimes;
        runTilingData.innerLoopTail = tilingData->innerLoopTail;
        runTilingData.indicesShapeRank = tilingData->indicesShapeRank;
        runTilingData.quantScalesElements = tilingData->quantScalesElements;
        runTilingData.quantZeroPointsElements = tilingData->quantZeroPointsElements;
        runTilingData.innerLoopTimesLastCore = tilingData->innerLoopTimesLastCore;
        runTilingData.innerLoopTailLastCore = tilingData->innerLoopTailLastCore;
        runTilingData.innerLoopFullRpt = tilingData->innerLoopFullRpt;
        runTilingData.innerLoopFullRptLastCore = tilingData->innerLoopFullRptLastCore;
        runTilingData.innerLoopTailRpt = tilingData->innerLoopTailRpt;
        runTilingData.innerLoopTailRptLastCore = tilingData->innerLoopTailRptLastCore;
        runTilingData.srcFirBsStride = tilingData->srcFirBsStride;
        runTilingData.dstFirSecBsStride = tilingData->dstFirSecBsStride;
        runTilingData.updateDim0 = tilingData->updateDim0;
        runTilingData.updateDim1 = tilingData->updateDim1;
        runTilingData.updateDim2 = tilingData->updateDim2;
        runTilingData.updateDim3 = tilingData->updateDim3;
        runTilingData.updateOriLastDim = tilingData->updateOriLastDim;
        runTilingData.updateOriLastDimAlign = tilingData->updateOriLastDimAlign;
    }

    __aicore__ int64_t CeilAlign(int64_t i, int64_t align)
    {
        if (align == 0) {
            return i;
        }
        return (i + align - 1) / align * align;
    }
};

}  // namespace QuantUpdateScatter
#endif  // QUANT_UPDATE_SCATTER_BASE_H_