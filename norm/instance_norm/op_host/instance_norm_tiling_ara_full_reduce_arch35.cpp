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
 * \file instance_norm_tiling_ara_full_reduce_arch35.cpp
 * \brief
 */

#include <vector>
#include <algorithm>
#include "instance_norm_tiling.h"

using namespace ge;

namespace {
constexpr int64_t TILINGKEY_ARA_FULL_REDUCE = 400000;

constexpr int64_t FP32_BYTE = 4;
constexpr int64_t FP16_BYTE = 2;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t BINARY_ADD_COEF = 2;
constexpr int64_t BINARY_ADD_COEF_FOUR = 4;
constexpr int64_t ARA_BINARY_ADD_THRESHOLD = 8;
constexpr int64_t VECTOR_LENGTH = 256;
constexpr int64_t LOG_2 = 2;
} // namespace

namespace optiling {
bool InstanceNormARAFullReduceTiling::IsCapable()
{
    if (format != FORMAT_NHWC && format != FORMAT_NDHWC) {
        return false;
    }
    int64_t a0TileBase = VECTOR_LENGTH / sizeof(float);
    int64_t elemSize = FP32_BYTE;
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        elemSize = FP16_BYTE;
    }

    // 切a0, 尽量占多核 factorMax -> aInner
    int64_t factorMax = aicoreParams_.ubSize /
                        ((elemSize * DOUBLE_BUFFER * 2 + // x + y
                          FP32_BYTE)                     // castbuf
                             * (r + 1) +
                         FP32_BYTE * DOUBLE_BUFFER * 2 + // mean + variance
                         FP32_BYTE * 2 +                 // gamma + beta
                         FP32_BYTE) /                    // rstdbuf
                        a0TileBase;
    if (factorMax >= 1) {
        return true;
    }
    return false;
}

ge::graphStatus InstanceNormARAFullReduceTiling::BinaryAddTiling()
{
    int64_t binaryQuotient = ARA_BINARY_ADD_THRESHOLD;
    while (binaryQuotient < r) {
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

ge::graphStatus InstanceNormARAFullReduceTiling::DoOpTiling()
{
    int64_t elemSize = FP32_BYTE;
    if (dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16) {
        elemSize = FP16_BYTE;
    }

    int64_t a0TileBase = vectorLength / sizeof(float);

    // 切a0, 尽量占多核
    int64_t factorMax = aicoreParams_.ubSize /
                        ((elemSize * DOUBLE_BUFFER * 2 + // x + y
                          FP32_BYTE)                     // castbuf
                             * (r + 1) +
                         FP32_BYTE * DOUBLE_BUFFER * 2 + // mean + variance
                         FP32_BYTE * 2 +                 // gamma + beta
                         FP32_BYTE) /                    // rstdbuf
                        a0TileBase;

    int64_t a0FactorMax = Ops::Base::CeilDiv(a0, a0TileBase);
    int64_t totalTilesMax = a1 * a0FactorMax;
    int64_t a0InnerMax = Ops::Base::CeilDiv(totalTilesMax, static_cast<int64_t>(aicoreParams_.blockDim));
    int64_t a0Inner = a0InnerMax < factorMax ? a0InnerMax : factorMax;
    a0Inner = a0Inner < a0FactorMax ? a0Inner : a0FactorMax;
    int64_t tileA0Len = a0Inner * a0TileBase;
    int64_t a0Outer = Ops::Base::CeilDiv(a0, tileA0Len);
    int64_t tileA0Tail = a0 - tileA0Len * (a0Outer - 1);

    // 分核
    int64_t totalTiles = a1 * a0Outer;
    int64_t tilesPerCore = Ops::Base::CeilDiv(totalTiles, static_cast<int64_t>(aicoreParams_.blockDim));
    blockNum_ = Ops::Base::CeilDiv(totalTiles, tilesPerCore);

    td_.totalTiles = totalTiles;
    td_.tilesPerCore = tilesPerCore;
    td_.usedCoreNum = blockNum_;
    td_.totalA1Len = a1;
    td_.totalRLen = r;
    td_.totalA0Len = a0;
    td_.a0Outer = a0Outer;
    td_.tileA0Len = tileA0Len;
    td_.tileA0Tail = tileA0Tail;
    // attr
    td_.epsilon = epsilon;

    uint64_t powerOfTwoForR = std::floor(std::log(r - 1) / std::log(2));
    powerOfTwoForR = std::pow(LOG_2, powerOfTwoForR);
    td_.powerOfTwoForR = powerOfTwoForR;

    if (r <= ARA_BINARY_ADD_THRESHOLD) {
        return ge::GRAPH_SUCCESS;
    }

    return BinaryAddTiling();
}

uint64_t InstanceNormARAFullReduceTiling::GetTilingKey() const
{
    return TILINGKEY_ARA_FULL_REDUCE;
}

ge::graphStatus InstanceNormARAFullReduceTiling::PostTiling()
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

REGISTER_OPS_TILING_TEMPLATE(InstanceNorm, InstanceNormARAFullReduceTiling, IN_ARA_FULL_REDUCE_PRIORITY);
} // namespace optiling
