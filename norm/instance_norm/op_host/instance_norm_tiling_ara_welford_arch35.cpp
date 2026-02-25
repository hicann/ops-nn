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
 * \file instance_norm_tiling_ara_welford_arch35.cpp
 * \brief
 */

#include <vector>
#include <algorithm>
#include "instance_norm_tiling.h"

using namespace ge;

namespace {
constexpr int64_t TILINGKEY_ARA_WELFORD = 500000;

constexpr int64_t FP32_BYTE = 4;
constexpr int64_t FP16_BYTE = 2;
constexpr int64_t LOG_2 = 2;
constexpr int64_t BINARY_ADD_COEF = 2;
constexpr int64_t BINARY_ADD_COEF_FOUR = 4;
constexpr int64_t ARA_BINARY_ADD_THRESHOLD = 4;
constexpr int64_t MEAN_VAR_BUFFER_NUM = 2;
constexpr int64_t BETA_GAMMA_BUFFER_NUM = 2;
constexpr int64_t DOUBLE_BUFFER_NUM = 2;
constexpr int64_t A0_UB_FACTOR_COEF = 2;
} // namespace

namespace optiling {
bool InstanceNormARAWelfordTiling::IsCapable()
{
    if (format != FORMAT_NHWC && format != FORMAT_NDHWC) {
        return false;
    }
    
    return true;
}

// tileA0Len must be aligned
ge::graphStatus InstanceNormARAWelfordTiling::BinaryAddTiling(int64_t elemSize, int64_t gammaElemSize, int64_t tileA0Len)
{
    // rFactor
    int64_t saveMeanVarianceSize = tileA0Len * static_cast<int64_t>(sizeof(float)) * MEAN_VAR_BUFFER_NUM;
    int64_t tmpRstdSize = tileA0Len * static_cast<int64_t>(sizeof(float));
    int64_t betaGammaSize = tileA0Len * gammaElemSize * BETA_GAMMA_BUFFER_NUM;
    int64_t xSizePerR = tileA0Len * elemSize * DOUBLE_BUFFER_NUM;
    int64_t ySizePerR = tileA0Len * elemSize * DOUBLE_BUFFER_NUM;
    int64_t tmpMeanM2PerR = tileA0Len * static_cast<int64_t>(sizeof(float)) * MEAN_VAR_BUFFER_NUM;
    int64_t tmpCountPerR = sizeof(float);

    int64_t ubSizeCanUse = aicoreParams_.ubSize - tmpRstdSize - saveMeanVarianceSize - betaGammaSize;
    OP_CHECK_IF(
        ubSizeCanUse <= 0, OP_LOGI(context_->GetNodeName(), "ubSizeCanUse is not a positive number."),
        return ge::GRAPH_PARAM_INVALID);
    int64_t rFactor = ubSizeCanUse / (xSizePerR + ySizePerR + tmpMeanM2PerR + tmpCountPerR);
    rFactor = Ops::Base::FloorAlign(rFactor, ARA_BINARY_ADD_THRESHOLD);
    OP_CHECK_IF(rFactor == 0, OP_LOGI(context_->GetNodeName(), "rfactor is 0."), return ge::GRAPH_PARAM_INVALID);
    int64_t rFactorAlign =
        Ops::Base::CeilAlign(static_cast<int64_t>(rFactor * sizeof(float)), ubBlockSize) / sizeof(float);
    if ((rFactor != rFactorAlign) &&
        ((rFactor * (xSizePerR + ySizePerR + tmpMeanM2PerR) + rFactorAlign * tmpCountPerR) > ubSizeCanUse)) {
        rFactor -= ARA_BINARY_ADD_THRESHOLD;
    }
    if (rFactor > r) {
        rFactor = Ops::Base::FloorAlign(r, ARA_BINARY_ADD_THRESHOLD);
    }
    td_.welfordrFactor = rFactor;

    int64_t binaryQuotient = ARA_BINARY_ADD_THRESHOLD;
    while (binaryQuotient < rFactor) {
        binaryQuotient *= BINARY_ADD_COEF;
    }
    binaryQuotient /= BINARY_ADD_COEF;
    td_.binaryAddQuotient = binaryQuotient;
    int64_t binaryAddNum = binaryQuotient / ARA_BINARY_ADD_THRESHOLD;
    int64_t binaryAddK = 0;
    int64_t curBinaryAddNum = 1;
    while (curBinaryAddNum < binaryAddNum) {
        binaryAddK++;
        curBinaryAddNum *= BINARY_ADD_COEF_FOUR;
    }
    if (curBinaryAddNum == binaryAddNum) {
        td_.binaryAddK = binaryAddK;
        td_.binaryAddLast = 0;
    } else if (curBinaryAddNum == binaryAddNum * BINARY_ADD_COEF) {
        td_.binaryAddK = binaryAddK - 1;
        td_.binaryAddLast = 1;
    } else {
        OP_LOGE(context_->GetNodeName(), "Binary add calculate error.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InstanceNormARAWelfordTiling::DoOpTiling()
{
    // dim
    td_.r = r;
    td_.a1 = a1;
    td_.a0 = a0;
    // attr
    td_.epsilon = epsilon;

    int64_t elemSize = FP32_BYTE;
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        elemSize = FP16_BYTE;
    }
    int64_t gammaElemSize = FP32_BYTE;
    if (gammaDataType == ge::DT_FLOAT16 || gammaDataType == ge::DT_BF16) {
        gammaElemSize = FP16_BYTE;
    }

    int64_t vlLength = vectorLength / sizeof(float);

    int64_t tileA0Len = vlLength * A0_UB_FACTOR_COEF;
    int64_t a0Outer = Ops::Base::CeilDiv(a0, tileA0Len);
    int64_t tileA0Tail = a0 - tileA0Len * (a0Outer - 1);
    int64_t totalTiles = a1 * a0Outer;
    int64_t tilesPerCore = Ops::Base::CeilDiv(totalTiles, static_cast<int64_t>(aicoreParams_.blockDim));
    blockNum_ = Ops::Base::CeilDiv(totalTiles, tilesPerCore);
    
    td_.a0Outer = a0Outer;
    td_.tileA0Tail = tileA0Tail;
    td_.totalTiles = totalTiles;
    td_.tileA0Len = tileA0Len;
    td_.tilesPerCore = tilesPerCore;
    td_.usedCoreNum = blockNum_;

    uint64_t powerOfTwoForR = std::floor(std::log(r - 1) / std::log(2));
    powerOfTwoForR = std::pow(LOG_2, powerOfTwoForR);
    td_.powerOfTwoForR = powerOfTwoForR;

    return BinaryAddTiling(elemSize, gammaElemSize, tileA0Len);
}

uint64_t InstanceNormARAWelfordTiling::GetTilingKey() const
{
    return TILINGKEY_ARA_WELFORD;
}

ge::graphStatus InstanceNormARAWelfordTiling::PostTiling()
{
    context_->SetBlockDim(blockNum_);
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
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

REGISTER_OPS_TILING_TEMPLATE(InstanceNorm, InstanceNormARAWelfordTiling, IN_ARA_WELFORD_PRIORITY);
} // namespace optiling
