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
 * \file fatrelu_mul_tiling.cpp
 * \brief fatrelu_mul_tiling source file
 */
#include "fatrelu_mul_tiling.h"
#include <vector>
#include "log/log.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <string>
#include "base/err_msg.h"

namespace optiling {

constexpr int32_t MAX_ELEMENT_NUM_EACH_CORE = 10 * 1024;
constexpr int32_t ONE_BLOCK_SIZE = 32;

constexpr uint64_t TILING_KEY_HALF = 1;
constexpr uint64_t TILING_KEY_FLOAT = 2;
constexpr uint64_t TILING_KEY_BFLOAT16 = 3;

const static int32_t SIZE_2 = 2;
const static int32_t SIZE_16 = 16;
const static int32_t LENGTH_1024 = 1024;
const static int32_t LENGTH_LIMIT = 200000;

class FatreluMulTiling {
public:
    explicit FatreluMulTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunBigKernelTiling();

private:
    void GetTilingKey();
    ge::DataType dataType = ge::DT_UNDEFINED;
    gert::TilingContext* tilingContext = nullptr;
    gert::Shape inputShape;
    FatreluMulTilingData tilingData;
    int64_t batchSize = 0;
    int64_t inputShapeSize = 0;
    int64_t lastDimSize = 0;
    int64_t oneBlockNum = 0;
    uint64_t tilingKey = 0;
    const int32_t workspaceSize_ = SIZE_16 * LENGTH_1024 * LENGTH_1024;

    inline int32_t CeilA2B(const int32_t a, const int32_t b) const
    {
        if (b != 0) {
            return (a + b - 1) / b;
        } else {
            return a;
        }
    }

    uint32_t GetNeedCoreNum(const uint32_t coreNumPlatform) const
    {
        uint32_t needCoreNum = 1;
        if (lastDimSize / SIZE_2 > MAX_ELEMENT_NUM_EACH_CORE) {
            needCoreNum = batchSize;
        } else {
            int32_t d = lastDimSize / SIZE_2;
            auto dAlign = (d + oneBlockNum - 1) / oneBlockNum * oneBlockNum;
            int32_t n = MAX_ELEMENT_NUM_EACH_CORE / dAlign;
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
};

void FatreluMulTiling::GetTilingKey()
{
    if (dataType == ge::DT_FLOAT16) {
        oneBlockNum = ONE_BLOCK_SIZE / SIZE_2;
        tilingKey = TILING_KEY_HALF;
    } else if (dataType == ge::DT_FLOAT) {
        oneBlockNum = ONE_BLOCK_SIZE / sizeof(float);
        tilingKey = TILING_KEY_FLOAT;
    } else if (dataType == ge::DT_BF16) {
        oneBlockNum = ONE_BLOCK_SIZE / SIZE_2;
        tilingKey = TILING_KEY_BFLOAT16;
    }
}

ge::graphStatus FatreluMulTiling::RunBigKernelTiling()
{
    // 获取输入矩阵
    auto srcTensor = tilingContext->GetInputTensor(0);
    OP_CHECK_IF(srcTensor == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(((tilingContext) == nullptr || (tilingContext)->GetNodeName() == nullptr ?
                                                "fatrelu_mul" :
                                                (tilingContext)->GetNodeName()),
                                           "x"),
                return ge::GRAPH_FAILED);
    // 获取数据类型
    auto temp = tilingContext->GetInputDesc(0);
    OP_CHECK_IF(temp == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(((tilingContext) == nullptr || (tilingContext)->GetNodeName() == nullptr ?
                                                "fatrelu_mul" :
                                                (tilingContext)->GetNodeName()),
                                           "x"),
                return ge::GRAPH_FAILED);
    dataType = tilingContext->GetInputDesc(0)->GetDataType();
    GetTilingKey();
    tilingContext->SetTilingKey(tilingKey);
    // 获取输入的shape
    auto srcShape = tilingContext->GetInputShape(0);
    inputShape = srcShape->GetOriginShape();
    size_t inputShapeDim = inputShape.GetDimNum();
    OP_CHECK_IF(
        (inputShapeDim < static_cast<size_t>(SIZE_2)),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(tilingContext->GetNodeName(), "x", std::to_string(inputShapeDim),
                                                 "Input shape dim should be no less than 2."),
        return ge::GRAPH_FAILED);
    lastDimSize = inputShape.GetDim(inputShapeDim - 1);
    inputShapeSize = inputShape.GetShapeSize();
    OP_CHECK_IF(
        (lastDimSize == 0),
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(tilingContext->GetNodeName(), "x", std::to_string(lastDimSize),
                                                  "Last dim elements can not be zero."),
        return ge::GRAPH_FAILED);
    batchSize = inputShapeSize / lastDimSize;
    auto thresholdShape = tilingContext->GetInputShape(1)->GetStorageShape();
    OP_CHECK_IF((thresholdShape.GetShapeSize() != 1),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(tilingContext->GetNodeName(), "threshold",
                                                          std::to_string(thresholdShape.GetShapeSize()),
                                                          "Threshold elements can only be one."),
                return ge::GRAPH_FAILED);
    auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    uint32_t needCoreNum = GetNeedCoreNum(platformInfo.GetCoreNumAiv());
    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = workspaceSize_;
    OP_CHECK_IF(
        (lastDimSize > LENGTH_1024),
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(tilingContext->GetNodeName(), "x", std::to_string(lastDimSize),
                                                  "Last dim size should be no more than 1024."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((lastDimSize % SIZE_2 == 1),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(tilingContext->GetNodeName(), "x",
                                                          std::to_string(lastDimSize), "Last dim size should be even."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((batchSize > LENGTH_LIMIT),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(tilingContext->GetNodeName(), "x", std::to_string(batchSize),
                                                          "Batch dim size should be no more than 200000."),
                return ge::GRAPH_FAILED);

    auto yShapePtr = tilingContext->GetOutputShape(0);
    OP_CHECK_IF(yShapePtr == nullptr, OP_LOGE_WITH_INVALID_INPUT(tilingContext->GetNodeName(), "out"),
                return ge::GRAPH_FAILED);
    auto yShape = yShapePtr->GetOriginShape();
    size_t yShapeDim = yShape.GetDimNum();
    OP_CHECK_IF((yShapeDim != inputShapeDim),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(tilingContext->GetNodeName(), "out", std::to_string(yShapeDim),
                                                         "Output shape dim should be the same as input x."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((yShape.GetDim(inputShapeDim - 1) != lastDimSize / 2),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(tilingContext->GetNodeName(), "out",
                                                          std::to_string(yShape.GetDim(inputShapeDim - 1)),
                                                          "Output last dim should be half of input x last dim."),
                return ge::GRAPH_FAILED);
    for (size_t i = 0; i + 1 < inputShapeDim; ++i) {
        OP_CHECK_IF((yShape.GetDim(i) != inputShape.GetDim(i)),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        tilingContext->GetNodeName(), "out, x",
                        std::to_string(yShape.GetDim(i)) + ", " + std::to_string(inputShape.GetDim(i)),
                        "Output shape dim should be the same as input x dim."),
                    return ge::GRAPH_FAILED);
    }

    tilingData.set_lastDimSize(lastDimSize);
    tilingData.set_batchSize(batchSize);
    tilingData.set_needCoreNum(needCoreNum);
    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                            tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    tilingContext->SetBlockDim(needCoreNum);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4FatreluMulTiling(gert::TilingParseContext* context)
{
    if ((context == nullptr)) {
        const std::string msgOpName = std::string(Ops::Base::GetOpInfo(context));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s", "TilingPrepare4FatreluMulTiling context is nullptr.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "TilingPrepare4FatreluMulTiling context is nullptr."}));
        return ge::GRAPH_FAILED;
    };
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFatreluMulTiling(gert::TilingContext* context)
{
    FatreluMulTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

IMPL_OP_OPTILING(FatreluMul)
    .Tiling(TilingFatreluMulTiling)
    .TilingParse<FatreluMulCompileInfo>(TilingPrepare4FatreluMulTiling);
} // namespace optiling
