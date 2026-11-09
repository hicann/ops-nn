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
 * \file layer_norm_v4_split_r_tiling.cpp
 * \brief
 */

#include <climits>
#include "layer_norm_v4_tiling.h"

namespace optiling {
namespace {
constexpr int64_t SPLIT_R_TEMPLATE = 8;
// Conservative small-A/large-R dispatch policy, not hardware limits.
// Keep enough work per split to amortize workspace traffic and SyncAll.
constexpr int64_t SPLIT_R_MIN_ELEMENTS = 16384;
constexpr int64_t SPLIT_R_MIN_ROW = 65536;
constexpr int64_t SPLIT_R_MAX_CORES = 16;
constexpr int64_t SPLIT_R_CORES_PER_ROW = 4;
constexpr int64_t SPLIT_R_MIN_CORES = 2;
constexpr uint64_t STATISTIC_PLANES = 2;
} // namespace

class LayerNormV4SplitRTiling : public LayerNormV4WelfordTiling {
public:
    explicit LayerNormV4SplitRTiling(gert::TilingContext* context) : LayerNormV4WelfordTiling(context) {}

protected:
    bool IsCapable() override
    {
        // A <= floor(AIV cores / 4), 65536 <= R <= INT32_MAX.
        // With 64 AIV cores this admits A <= 16. Each split has >= 16384 elements.
        if (!commonParams.isRegBase || commonParams.isV3) {
            return false;
        }
        if (commonParams.colSize == 0 || commonParams.colSize > commonParams.coreNum / SPLIT_R_CORES_PER_ROW) {
            return false;
        }
        splitR_ = std::min<int64_t>(commonParams.coreNum / commonParams.colSize, SPLIT_R_MAX_CORES);
        if (commonParams.rowSize < SPLIT_R_MIN_ROW || commonParams.rowSize > INT32_MAX) {
            return false;
        }
        splitR_ = std::min<int64_t>(splitR_, commonParams.rowSize / SPLIT_R_MIN_ELEMENTS);
        return splitR_ >= SPLIT_R_MIN_CORES;
    }

    // The UB deduction must wrap the base DoLibApiTiling call: the tileLength
    // selection loop (IsValidTileLength vs commonParams.ubSizePlatForm) runs
    // inside it, so the reserved partial-queue UB is excluded before
    // tileLength is chosen. The base DoOpTiling only derives the block
    // partitioning from M/coreNum and reads no UB budget, so it needs no
    // deduction and is inherited unchanged.
    ge::graphStatus DoLibApiTiling() override
    {
        // Separate mean/variance planes coexist with the Welford buffers.
        // Match the vector-aligned partial planes allocated by the kernel.
        uint64_t vectorBytes = Ops::Base::GetVRegSize(context_);
        uint64_t planeBytes = ((splitR_ * sizeof(float) + vectorBytes - 1) / vectorBytes) * vectorBytes;
        uint64_t reserved = STATISTIC_PLANES * planeBytes;
        if (commonParams.ubSizePlatForm <= reserved) {
            return ge::GRAPH_FAILED;
        }
        commonParams.ubSizePlatForm -= reserved;
        auto status = LayerNormV4WelfordTiling::DoLibApiTiling();
        commonParams.ubSizePlatForm += reserved;
        return status;
    }

    uint64_t GetTilingKey() const override { return SPLIT_R_TEMPLATE * LN_TEMPLATE_KEY_WEIGHT + commonParams.dtypeKey; }

    ge::graphStatus PostTiling() override
    {
        splitTd_.set_M(td_.get_M());
        splitTd_.set_N(td_.get_N());
        splitTd_.set_rAlign(td_.get_rAlign());
        splitTd_.set_numBlocks(td_.get_numBlocks());
        splitTd_.set_mainBlockCount(td_.get_mainBlockCount());
        splitTd_.set_mainBlockFactor(td_.get_mainBlockFactor());
        splitTd_.set_tailBlockFactor(td_.get_tailBlockFactor());
        splitTd_.set_tileLength(td_.get_tileLength());
        splitTd_.set_welfordTempSize(td_.get_welfordTempSize());
        splitTd_.set_welfordUpdateTimes(td_.get_welfordUpdateTimes());
        splitTd_.set_welfordUpdateTail(td_.get_welfordUpdateTail());
        splitTd_.set_nullptrGamma(td_.get_nullptrGamma());
        splitTd_.set_nullptrBeta(td_.get_nullptrBeta());
        splitTd_.set_apiTempBufferSize(td_.get_apiTempBufferSize());
        splitTd_.set_epsilon(td_.get_epsilon());
        splitTd_.set_splitR(splitR_);
        context_->SetBlockDim(commonParams.colSize * splitR_);
        auto* raw = context_->GetRawTilingData();
        if (splitTd_.GetDataSize() > raw->GetCapacity()) {
            return ge::GRAPH_FAILED;
        }
        splitTd_.SaveToBuffer(raw->GetData(), raw->GetCapacity());
        raw->SetDataSize(splitTd_.GetDataSize());
        auto* sizes = context_->GetWorkspaceSizes(1);
        auto* info = context_->GetPlatformInfo();
        OP_CHECK_NULL_WITH_CONTEXT(context_, sizes);
        OP_CHECK_NULL_WITH_CONTEXT(context_, info);
        platform_ascendc::PlatformAscendC platform(info);
        // Two statistic planes (mean and variance), independent of double buffering.
        sizes[0] = platform.GetLibApiWorkSpaceSize() +
                   STATISTIC_PLANES * commonParams.colSize * splitR_ * Ops::Base::GetUbBlockSize(context_);
        return context_->SetScheduleMode(1U);
    }

private:
    int64_t splitR_ = 1;
    LayerNormV4TilingDataSplitR splitTd_;
};

REGISTER_OPS_TILING_TEMPLATE(LayerNormV4, LayerNormV4SplitRTiling, 140);
} // namespace optiling
