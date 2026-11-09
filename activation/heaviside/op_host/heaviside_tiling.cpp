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
 * \file heaviside_tiling.cpp
 * \brief heaviside_tiling source file
 */
#include "heaviside_tiling.h"
#include <limits>
#include <vector>
#include "op_common/log/log.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"

namespace optiling {

constexpr int32_t MAX_ELEMENT_NUM_EACH_CORE = 8 * 1024;
constexpr int32_t MAX_INPUT_DIM_NUM = 8;

constexpr uint64_t TILING_KEY_DEFAULT = 1;

class HeavisideTiling {
public:
    explicit HeavisideTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunBigKernelTiling();

private:
    ge::DataType dataType = ge::DT_UNDEFINED;
    gert::TilingContext* tilingContext = nullptr;
    gert::Shape inputShape;
    gert::Shape valuesShape;
    HeavisideTilingData tilingData;

    int64_t inputShapeSize = 0;
    int64_t valuesShapeSize = 0;
    bool valuesType = false;

    inline int64_t CeilA2B(const int64_t a, const int64_t b) const
    {
        if (b == 0) {
            return a;
        }
        if (a <= 0) {
            return 0;
        }
        // (a + b - 1) / b can overflow signed int64 when a is near INT64_MAX;
        // (a - 1) / b + 1 is equivalent for positive a and never overflows.
        return (a - 1) / b + 1;
    }

    int32_t GetNeedCoreNum(const int32_t coreNumPlatform) const
    {
        int64_t needCoreNum = CeilA2B(inputShapeSize, MAX_ELEMENT_NUM_EACH_CORE);
        if (needCoreNum == 0) {
            needCoreNum = 1;
        }
        if (needCoreNum >= coreNumPlatform) {
            return coreNumPlatform;
        } else {
            return needCoreNum;
        }
    }

    bool GetValuesType()
    {
        if (valuesShapeSize == 1) {
            valuesType = true;
        } else if (valuesShapeSize > 0 && valuesShape == inputShape) {
            valuesType = false;
        } else if (valuesShapeSize == 0) {
            OP_LOGE(tilingContext->GetNodeName(),
                    "Heaviside does not support empty tensor; values shape size is 0 (input shape size: %d).",
                    static_cast<int32_t>(inputShapeSize));
            return false;
        } else {
            OP_LOGE(tilingContext->GetNodeName(),
                    "Heaviside does not support broadcast; values must be a scalar (shape size 1) or have the same "
                    "shape as input (current values shape size: %d, input shape size: %d).",
                    static_cast<int32_t>(valuesShapeSize), static_cast<int32_t>(inputShapeSize));
            return false;
        }
        return true;
    }
};

ge::graphStatus HeavisideTiling::RunBigKernelTiling()
{
    auto srcTensor = tilingContext->GetInputTensor(0);
    if (srcTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto valuesTensor = tilingContext->GetInputTensor(1);
    if (valuesTensor == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto inputDesc = tilingContext->GetInputDesc(0);
    if (inputDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    dataType = inputDesc->GetDataType();

    auto valuesDesc = tilingContext->GetInputDesc(1);
    if (valuesDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (valuesDesc->GetDataType() != dataType) {
        OP_LOGE(tilingContext->GetNodeName(),
                "values dtype %d is inconsistent with input dtype %d, they must be the same.",
                static_cast<int32_t>(valuesDesc->GetDataType()), static_cast<int32_t>(dataType));
        return ge::GRAPH_FAILED;
    }

    tilingContext->SetTilingKey(TILING_KEY_DEFAULT);

    // 获取input和values的shape
    auto srcShape0 = tilingContext->GetInputShape(0);
    auto srcShape1 = tilingContext->GetInputShape(1);
    if (srcShape0 == nullptr || srcShape1 == nullptr) {
        OP_LOGE(tilingContext->GetNodeName(),
                "Tiling framework object input shape is unavailable; check the operator registration and "
                "context construction.");
        return ge::GRAPH_FAILED;
    }
    inputShape = srcShape0->GetOriginShape();
    inputShapeSize = inputShape.GetShapeSize();
    valuesShape = srcShape1->GetOriginShape();
    valuesShapeSize = valuesShape.GetShapeSize();
    if (inputShapeSize == std::numeric_limits<int64_t>::min() ||
        valuesShapeSize == std::numeric_limits<int64_t>::min()) {
        OP_LOGE(tilingContext->GetNodeName(),
                "input/values shape size overflow (GetShapeSize returned the invalid sentinel).");
        return ge::GRAPH_FAILED;
    }
    if (inputShape.GetDimNum() > static_cast<size_t>(MAX_INPUT_DIM_NUM)) {
        OP_LOGE(tilingContext->GetNodeName(), "input dim num %d exceeds the supported maximum %d.",
                static_cast<int32_t>(inputShape.GetDimNum()), MAX_INPUT_DIM_NUM);
        return ge::GRAPH_FAILED;
    }
    if (GetValuesType() == false) {
        return ge::GRAPH_FAILED;
    }

    auto platformInfoPtr = tilingContext->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(tilingContext->GetNodeName(),
                "Tiling framework object platformInfo is unavailable; check the operator registration and "
                "context construction.");
        return ge::GRAPH_FAILED;
    }
    auto platformInfo = platform_ascendc::PlatformAscendC(platformInfoPtr);
    uint32_t coreNumPlatform = platformInfo.GetCoreNumAiv();
    if (coreNumPlatform == 0) {
        OP_LOGE(tilingContext->GetNodeName(), "platform core num is 0, GetCoreNumAiv failed.");
        return ge::GRAPH_FAILED;
    }
    uint32_t needCoreNum = GetNeedCoreNum(static_cast<int32_t>(coreNumPlatform));

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    if (currentWorkspace == nullptr) {
        OP_LOGE(tilingContext->GetNodeName(),
                "Tiling framework object workspace is unavailable; check the operator registration and "
                "context construction.");
        return ge::GRAPH_FAILED;
    }
    currentWorkspace[0] = 0;

    tilingData.set_elementNum(inputShapeSize);
    tilingData.set_valuesType(valuesType);
    tilingData.set_needCoreNum(needCoreNum);

    auto rawTilingData = tilingContext->GetRawTilingData();
    if (rawTilingData == nullptr) {
        OP_LOGE(tilingContext->GetNodeName(),
                "Tiling framework object rawTilingData is unavailable; check the operator registration and "
                "context construction.");
        return ge::GRAPH_FAILED;
    }
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());

    tilingContext->SetBlockDim(needCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4HeavisideTiling(gert::TilingParseContext* /*context*/)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingHeavisideTiling(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin the tiling process for Arch35 architecture");
    HeavisideTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

IMPL_OP_OPTILING(Heaviside)
    .Tiling(TilingHeavisideTiling)
    .TilingParse<HeavisideCompileInfo>(TilingPrepare4HeavisideTiling);
} // namespace optiling
