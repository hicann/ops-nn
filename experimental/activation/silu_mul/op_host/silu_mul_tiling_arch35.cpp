/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the 'License').
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file silu_mul_tiling_arch35.cpp
 * \brief Tiling implementation for SiluMul Arch35 (Ascend 950)
 *
 * Differences vs 910b tiling:
 *  - UB size queried from platform via GetCoreMemSize instead of hardcoded 184KB.
 *  - Core-num estimation uses full lastDimSize (aligned with kernel's
 *    `if (d > PPMaxCalNum)` path selection), fixing the spurious /2 in 910b.
 *  - Uses a plain POD tiling struct with REGISTER_TILING_DEFAULT, decoupled from
 *    the 910b BEGIN_TILING_DATA_DEF-based SiluMulTilingData.
 */

#include <vector>
#include <iostream>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "register/op_def_registry.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "../op_kernel/arch35/silu_mul_tiling_struct.h"

namespace optiling {

static constexpr int32_t ARCH35_ONE_BLOCK_SIZE = 32;
static constexpr int32_t ARCH35_CALC_BUF_NUM = 8;
static constexpr int32_t ARCH35_HALF_SIZE = 2;
static constexpr int32_t ARCH35_BF16_SIZE = 2;
static constexpr int32_t ARCH35_SIZE_16 = 16;
static constexpr int32_t ARCH35_LENGTH_1024 = 1024;
static constexpr int32_t ARCH35_LENGTH_LIMIT = 200000;
static constexpr int32_t ARCH35_UB_RESERVE = 8 * 1024; // headroom for framework UB stack

class SiluMulArch35Tiling {
public:
    explicit SiluMulArch35Tiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunTiling();

private:
    ge::graphStatus FillTilingKey();
    ge::graphStatus ShapeCheck();

    static inline int32_t CeilA2B(const int32_t a, const int32_t b)
    {
        if (b != 0) {
            return (a + b - 1) / b;
        } else {
            return a;
        }
    }

    int32_t GetNeedCoreNum(const int32_t coreNumPlatform)
    {
        int32_t needCoreNum = 1;
        // Use full lastDimSize to align with kernel path selection (d > PPMaxCalNum).
        if (lastDimSize > PPMaxCalNum) {
            needCoreNum = batchSize;
        } else {
            auto dAlign = (lastDimSize + oneBlockNum - 1) / oneBlockNum * oneBlockNum;
            const int32_t n = PPMaxCalNum / dAlign;
            needCoreNum = CeilA2B(batchSize, n);
        }
        if (needCoreNum == 0) {
            needCoreNum = 1;
        }
        if (needCoreNum >= coreNumPlatform) {
            return coreNumPlatform;
        } else {
            return needCoreNum;
        }
    }

    ge::DataType dataType = ge::DT_UNDEFINED;
    gert::TilingContext* tilingContext = nullptr;
    gert::Shape inputShape;
    int32_t batchSize = 0;
    int32_t inputShapeSize = 0;
    int32_t lastDimSize = 0;
    int32_t oneBlockNum = 0;
    int32_t PPMaxCalNum = 0;
    uint64_t ubSize_ = 0;
    int32_t workspaceSize_ = ARCH35_SIZE_16 * ARCH35_LENGTH_1024 * ARCH35_LENGTH_1024;
};

ge::graphStatus SiluMulArch35Tiling::ShapeCheck()
{
    OP_CHECK_IF((lastDimSize > ARCH35_LENGTH_1024),
                OP_LOGE(tilingContext->GetNodeName(), "Last dim size should be no more than 1024."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((lastDimSize % 2 == 1), OP_LOGE(tilingContext->GetNodeName(), "Last dim size should be even."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((batchSize > ARCH35_LENGTH_LIMIT),
                OP_LOGE(tilingContext->GetNodeName(), "Batch dim size should be no more than 200000."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SiluMulArch35Tiling::FillTilingKey()
{
    auto temp = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, temp);
    dataType = tilingContext->GetInputDesc(0)->GetDataType();
    if (dataType == ge::DT_FLOAT16) {
        oneBlockNum = ARCH35_ONE_BLOCK_SIZE / ARCH35_HALF_SIZE;
    } else if (dataType == ge::DT_FLOAT) {
        oneBlockNum = ARCH35_ONE_BLOCK_SIZE / static_cast<int32_t>(sizeof(float));
    } else if (dataType == ge::DT_BF16) {
        oneBlockNum = ARCH35_ONE_BLOCK_SIZE / ARCH35_BF16_SIZE;
    } else {
        return ge::GRAPH_FAILED;
    }
    tilingContext->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SiluMulArch35Tiling::RunTiling()
{
    auto srcTensor = tilingContext->GetInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, srcTensor);

    auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    // Query UB size from platform instead of hardcoding 184KB.
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    OP_CHECK_IF((ubSize_ == 0), OP_LOGE(tilingContext->GetNodeName(), "Get UB size failed, ub size is 0."),
                return ge::GRAPH_FAILED);
    // Reserve a small headroom for the framework UB stack; both PPMaxCalNum and the
    // kernel's maxUbSize are derived from this same usable value to stay consistent.
    uint64_t usableUb = (ubSize_ > static_cast<uint64_t>(ARCH35_UB_RESERVE)) ?
                            (ubSize_ - static_cast<uint64_t>(ARCH35_UB_RESERVE)) :
                            ubSize_;
    PPMaxCalNum = static_cast<int32_t>(usableUb) / ARCH35_CALC_BUF_NUM / static_cast<int32_t>(sizeof(float));

    FillTilingKey();

    auto srcShape = tilingContext->GetInputShape(0);
    inputShape = srcShape->GetOriginShape();
    size_t inputShapeDim = inputShape.GetDimNum();
    OP_CHECK_IF((inputShapeDim < static_cast<size_t>(2)),
                OP_LOGE(tilingContext->GetNodeName(), "Input shape dim should be no less than 2."),
                return ge::GRAPH_FAILED);
    lastDimSize = inputShape.GetDim(inputShapeDim - 1);
    inputShapeSize = inputShape.GetShapeSize();

    if (lastDimSize == 0) {
        OP_LOGE(tilingContext->GetNodeName(), "Last dim elements can not be zero.");
        return ge::GRAPH_FAILED;
    }

    batchSize = inputShapeSize / lastDimSize;

    int32_t needCoreNum = GetNeedCoreNum(platformInfo.GetCoreNumAiv());

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(workspaceSize_);
    OP_CHECK_IF((ShapeCheck() == ge::GRAPH_FAILED), OP_LOGE(tilingContext->GetNodeName(), "ShapeCheck failed!"),
                return ge::GRAPH_FAILED);

    auto tilingData = tilingContext->GetTilingData<SiluMulArch35TilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, tilingData);
    tilingData->lastDimSize = lastDimSize;
    tilingData->batchSize = batchSize;
    tilingData->PPMaxCalNum = PPMaxCalNum;
    tilingData->needCoreNum = needCoreNum;
    tilingData->maxUbSize = static_cast<uint32_t>(usableUb);

    tilingContext->SetBlockDim(needCoreNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Tiling4SiluMulArch35(gert::TilingContext* context)
{
    SiluMulArch35Tiling tilingObject(context);
    return tilingObject.RunTiling();
}

} // namespace optiling
