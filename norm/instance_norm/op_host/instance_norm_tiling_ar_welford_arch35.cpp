/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file instance_norm_tiling_ar_welford_arch35.cpp
 * \brief
 */

#include <vector>
#include <algorithm>
#include "instance_norm_tiling.h"

using namespace ge;

namespace optiling {
constexpr int64_t TILINGKEY_AR_WELFORD = 300000;

constexpr static int64_t WELFORD_CONSTANT_TWO = 2;
constexpr static int64_t WELFORD_TILELENGTH_STEP_SIZE = 64;
constexpr static int64_t WELFORD_DOUBLE_BUFFER = 2;
constexpr static int64_t WELFORD_B32_SIZE = 4;
constexpr static int64_t WELFORD_B16_SIZE = 2;

void InstanceNormARWelfordTiling::Reset(gert::TilingContext* context)
{
    InstanceNormRegbaseTilingBase::Reset(context);
    blockNum_ = 0;
}

bool InstanceNormARWelfordTiling::IsCapable()
{
    if (format != FORMAT_NCHW && format != FORMAT_NCDHW && format != FORMAT_ND) {
        // NHWC和NHDWC场景中，如果a0是1，也放在AR模版处理
        if (a0 != 1) {
            return false;
        }
    }

    OP_LOGI(context_->GetNodeName(), "InstanceNormARWelfordTiling IsCapable: IsCapable is true !");

    return true;
}

ge::graphStatus InstanceNormARWelfordTiling::DoOpTiling()
{
    td_.r = r;
    td_.a1 = a1;
    td_.a0 = a0;
    td_.epsilon = epsilon;

    GammaBetaTypeSize = WELFORD_B32_SIZE;
    if (gammaDataType == ge::DT_FLOAT16 || gammaDataType == ge::DT_BF16) {
        GammaBetaTypeSize = WELFORD_B16_SIZE;
    }
    int64_t a0Inner = WELFORD_CONSTANT_TWO * vectorLength / GammaBetaTypeSize;
    OP_LOGI(context_->GetNodeName(), "InstanceNormARWelfordTiling DoOpTiling: class member vectorLength is %lu !", vectorLength);
    int64_t a0Outer = Ops::Base::CeilDiv(a0, a0Inner);
    int64_t a0Tail = a0 - a0Inner * (a0Outer - 1);

    // 分核
    int64_t totalTiles = a1 * a0Outer;
    int64_t tilesPerCore = Ops::Base::CeilDiv(totalTiles, static_cast<int64_t>(aicoreParams_.blockDim));
    blockNum_ = Ops::Base::CeilDiv(totalTiles, tilesPerCore);

    td_.totalTiles = totalTiles;
    td_.tilesPerCore = tilesPerCore;
    td_.blockNum = blockNum_;
    td_.a0Outer = a0Outer;
    td_.a0Inner = a0Inner;
    td_.a0Tail = a0Tail;

    OP_LOGI(context_->GetNodeName(), "InstanceNormARWelfordTiling DoOpTiling: r is %lu,"
            " a1 is %lu, a0 is %lu, totalTiles is %lu, tilesPerCore is %lu, blockNum_ is %lu,"
            " a0Outer is %lu, a0Inner is %lu, a0Tail is %lu !",
            r, a1, a0, totalTiles, tilesPerCore, blockNum_, a0Outer, a0Inner, a0Tail);

    return ge::GRAPH_SUCCESS;
}

bool InstanceNormARWelfordTiling::IsValidwelfordTileLength(int64_t welfordTileLength)
{
    int64_t xSize = 0;
    int64_t ySize = 0;
    int64_t meanSize = 0;
    int64_t varianceSize = 0;
    int64_t welfordTempSize = 0;
    int64_t gammaSize = 0;
    int64_t betaSize = 0;
    int64_t apiTempSize = 0;
    int64_t cntBufSize = 0;

    // tensor size
    int64_t xDataTypeSize = WELFORD_B32_SIZE;
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        xSize = WELFORD_DOUBLE_BUFFER * welfordTileLength * WELFORD_B16_SIZE;
        ySize = WELFORD_DOUBLE_BUFFER * welfordTileLength * WELFORD_B16_SIZE;
        xDataTypeSize = WELFORD_B16_SIZE;
    } else {
        xSize = WELFORD_DOUBLE_BUFFER * welfordTileLength * WELFORD_B32_SIZE;
        ySize = WELFORD_DOUBLE_BUFFER * welfordTileLength * WELFORD_B32_SIZE;
    }

    // mean rstd welfordTemp size
    meanSize = WELFORD_DOUBLE_BUFFER * td_.a0Inner * WELFORD_B32_SIZE;
    varianceSize = WELFORD_DOUBLE_BUFFER * td_.a0Inner * WELFORD_B32_SIZE;
    welfordTempSize += WELFORD_DOUBLE_BUFFER * welfordTileLength * WELFORD_B32_SIZE;
    welfordTempSize += td_.a0Inner * WELFORD_B32_SIZE;
    td_.welfordTempSize = welfordTempSize;
    cntBufSize = welfordTileLength * WELFORD_B32_SIZE;

    // gamma beta size
    gammaSize = WELFORD_DOUBLE_BUFFER * td_.a0Inner * GammaBetaTypeSize;
    betaSize = WELFORD_DOUBLE_BUFFER * td_.a0Inner * GammaBetaTypeSize;

    // apiTemp size
    int64_t welfordUpdateApiTempSize = 0;
    int64_t welfordFinalizeApiTempSize = 0;
    uint32_t minValue{0};
    uint32_t maxValue{0};
    ge::Shape tensorShape({1, welfordTileLength});
    AscendC::GetWelfordUpdateMaxMinTmpSize(
        tensorShape, xDataTypeSize, WELFORD_B32_SIZE, false, true, maxValue, minValue);
    welfordUpdateApiTempSize = minValue;
    AscendC::GetWelfordFinalizeMaxMinTmpSize(tensorShape, WELFORD_B32_SIZE, false, maxValue, minValue);
    welfordFinalizeApiTempSize = minValue;
    apiTempSize = welfordUpdateApiTempSize + welfordFinalizeApiTempSize;
    td_.apiTempBufferSize = apiTempSize;

    // total size
    int64_t totalSize =
        (xSize + ySize) + (meanSize + varianceSize) + (gammaSize + betaSize) + welfordTempSize + apiTempSize + cntBufSize;
    return (totalSize <= static_cast<int64_t>(aicoreParams_.ubSize));
}

ge::graphStatus InstanceNormARWelfordTiling::DoLibApiTiling()
{
    auto meanDesc = context_->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, meanDesc);
    meanDataType = meanDesc->GetDataType();

    int64_t welfordTileLength = 0;
    OP_LOGI(context_->GetNodeName(), "InstanceNormARWelfordTiling DoLibApiTiling:  static_cast<int64_t>(aicoreParams_.ubSize) is %lu !",  static_cast<int64_t>(aicoreParams_.ubSize));
    while (IsValidwelfordTileLength(welfordTileLength + WELFORD_CONSTANT_TWO * WELFORD_TILELENGTH_STEP_SIZE)) {
        welfordTileLength += WELFORD_TILELENGTH_STEP_SIZE;
    }
    OP_LOGI(context_->GetNodeName(), "InstanceNormARWelfordTiling DoLibApiTiling: welfordTileLength is %lu !", welfordTileLength);

    int64_t welfordUpdateTimes = r / welfordTileLength;
    OP_LOGI(context_->GetNodeName(), "InstanceNormARWelfordTiling DoLibApiTiling: welfordUpdateTimes is %lu !", welfordUpdateTimes);

    int64_t welfordUpdateTail = r - welfordUpdateTimes * welfordTileLength;
    OP_LOGI(context_->GetNodeName(), "InstanceNormARWelfordTiling DoLibApiTiling: welfordUpdateTail is %lu !", welfordUpdateTail);
    
    td_.welfordTileLength = welfordTileLength;
    td_.welfordUpdateTimes = welfordUpdateTimes;
    td_.welfordUpdateTail = welfordUpdateTail;
    return ge::GRAPH_SUCCESS;
}

uint64_t InstanceNormARWelfordTiling::GetTilingKey() const
{
    return TILINGKEY_AR_WELFORD;
}

ge::graphStatus InstanceNormARWelfordTiling::PostTiling()
{
    context_->SetBlockDim(blockNum_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = workspaceSize_;
    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_IF(
        sizeof(td_) > rawTilingData->GetCapacity(),
        OP_LOGE(
            context_->GetNodeName(), "actual tiling data size %zu > context tiling data size %zu", sizeof(td_),
            rawTilingData->GetCapacity()),
        return ge::GRAPH_FAILED);
    auto capSize = rawTilingData->GetCapacity();
    void* ptrData = rawTilingData->GetData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrData);
    void* ptrStruct = static_cast<void*>(&td_);
    OP_CHECK_NULL_WITH_CONTEXT(context_, ptrStruct);
    OP_CHECK_IF(
        memcpy_s(ptrData, capSize, ptrStruct, sizeof(td_)) != 0,
        OP_LOGE(context_->GetNodeName(), "Set tiling data is failed!"), return ge::GRAPH_FAILED);
    rawTilingData->SetDataSize(sizeof(td_));
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(InstanceNorm, InstanceNormARWelfordTiling, IN_AR_WELFORD_PRIORITY);
} // namespace optiling