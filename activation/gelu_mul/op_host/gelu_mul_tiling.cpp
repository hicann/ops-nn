/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gelu_mul_tiling.cpp
 * \brief gelu_mul_tiling source file
 */
#include "gelu_mul_tiling.h"
#include <vector>
#include <iostream>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"

namespace optiling {


constexpr int32_t UB_SIZE = 192 * 1024;
constexpr int32_t ONE_BLOCK_SIZE = 32;
constexpr int32_t ERF_BUF_NUM = 8;
constexpr int32_t TANH_BUF_NUM = 6;
constexpr int32_t HALF_SIZE = 2;
constexpr int32_t BF16_SIZE = 2;
constexpr int32_t SIZE_2 = 2;
constexpr int32_t TILING_KEY_HALF = 1;
constexpr int32_t TILING_KEY_FLOAT = 2;
constexpr int32_t TILING_KEY_BFLOAT16 = 3;

const static int32_t SIZE_16 = 16;
const static int32_t LENGTH_1024 = 1024;
const static int32_t LENGTH_LIMIT = 200000;

class GeluMulTiling {
public:
    explicit GeluMulTiling(gert::TilingContext* context) : tilingContext(context){};
    ge::graphStatus RunBigKernelTiling();
    ge::graphStatus FillTilingKey();

private:
    ge::graphStatus ShapeCheck();
    ge::DataType dataType = ge::DT_UNDEFINED;
    gert::TilingContext* tilingContext = nullptr;
    gert::Shape inputShape;
    GeluMulTilingData tilingData;
    int32_t batchSize = 0;
    int32_t inputShapeSize = 0;
    int32_t lastDimSize = 0;
    int32_t oneBlockNum = 0;
    int32_t PPMaxCalNum = 0;
    const int32_t workspaceSize_ = SIZE_16 * LENGTH_1024 * LENGTH_1024;

    inline int32_t CeilA2B(const int32_t a, const int32_t b) {
        if (b != 0) {
            return (a + b - 1) / b;
        } else {
            return a;
        }
    }

    int32_t GetNeedCoreNum(const int32_t coreNumPlatform) {
        int32_t needCoreNum = 1;
        if (lastDimSize / SIZE_2 > PPMaxCalNum) {
            needCoreNum = batchSize;
        } else {
            const int32_t d = lastDimSize / SIZE_2;
            auto dAlign = (d + oneBlockNum - 1) / oneBlockNum * oneBlockNum;
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
};

ge::graphStatus GeluMulTiling::ShapeCheck() {
    OP_CHECK_IF((lastDimSize > LENGTH_1024),
        OP_LOGE(tilingContext->GetNodeName(), "Last dim size should be no more than 1024."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((lastDimSize % SIZE_2 == 1),
        OP_LOGE(tilingContext->GetNodeName(), "Last dim size should be even."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((batchSize > LENGTH_LIMIT),
        OP_LOGE(tilingContext->GetNodeName(), "Batch dim size should be no more than 200000."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluMulTiling::RunBigKernelTiling() {
    // 获取输入矩阵
    auto srcTensor = tilingContext->GetInputTensor(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, srcTensor);
    // 获取输入的参数
    const gert::RuntimeAttrs* attrs = tilingContext->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, attrs);
    const std::string approximate = static_cast<std::string>(attrs->GetStr(0));

    int32_t approximateMode = 0;
    if (approximate == "none") {
        approximateMode = 0;
        PPMaxCalNum = UB_SIZE / ERF_BUF_NUM / static_cast<int32_t>(sizeof(float));
    } else if (approximate == "tanh") {
        approximateMode = 1;
        PPMaxCalNum = UB_SIZE / TANH_BUF_NUM / static_cast<int32_t>(sizeof(float));
    }
    OP_CHECK_IF((approximate != "none" && approximate != "tanh"),
        OP_LOGE(tilingContext->GetNodeName(), "Approximate can only be 'tanh' or 'none'."),
        return ge::GRAPH_FAILED);
    FillTilingKey();
    // 获取输入的shape
    auto srcShape = tilingContext->GetInputShape(0);
    inputShape = srcShape->GetOriginShape();
    size_t inputShapeDim = inputShape.GetDimNum();
    OP_CHECK_IF((inputShapeDim < static_cast<size_t>(SIZE_2)),
        OP_LOGE(tilingContext->GetNodeName(), "Input shape dim should be no less than 2."),
        return ge::GRAPH_FAILED);
    lastDimSize = inputShape.GetDim(inputShapeDim - 1);
    inputShapeSize = inputShape.GetShapeSize();
    OP_CHECK_IF((lastDimSize == 0),
        OP_LOGE(tilingContext->GetNodeName(), "Last dim elements can not be zero."),
        return ge::GRAPH_FAILED);

    if(lastDimSize == 0)
    {
       OP_LOGE(tilingContext->GetNodeName(), "Last dim elements can not be zero.");
       return ge::GRAPH_FAILED;
    }

    batchSize = inputShapeSize / lastDimSize;

    auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    int32_t needCoreNum = GetNeedCoreNum(platformInfo.GetCoreNumAiv());

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(workspaceSize_);
    OP_CHECK_IF((ShapeCheck() == ge::GRAPH_FAILED),
        OP_LOGE(tilingContext->GetNodeName(), "ShapeCheck failed!"),
        return ge::GRAPH_FAILED);
    tilingData.set_lastDimSize(lastDimSize);
    tilingData.set_batchSize(batchSize);
    tilingData.set_approximateMode(approximateMode);
    tilingData.set_PPMaxCalNum(PPMaxCalNum);
    tilingData.set_needCoreNum(needCoreNum);

    tilingData.SaveToBuffer(tilingContext->GetRawTilingData()->GetData(),
                            tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    tilingContext->SetBlockDim(needCoreNum);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluMulTiling::FillTilingKey() {
    // 获取数据类型
    auto temp = tilingContext->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, temp);
    dataType = tilingContext->GetInputDesc(0)->GetDataType();
    int32_t tilingKey = 0;
    if (dataType == ge::DT_FLOAT16) {
        oneBlockNum = ONE_BLOCK_SIZE / HALF_SIZE;
        tilingKey = TILING_KEY_HALF;
    } else if (dataType == ge::DT_FLOAT) {
        oneBlockNum = ONE_BLOCK_SIZE / static_cast<int32_t>(sizeof(float));
        tilingKey = TILING_KEY_FLOAT;
    } else if (dataType == ge::DT_BF16) {
        oneBlockNum = ONE_BLOCK_SIZE / BF16_SIZE;
        tilingKey = TILING_KEY_BFLOAT16;
    } else {
        return ge::GRAPH_FAILED;
    }
    tilingContext->SetTilingKey(tilingKey);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4GeluMulTiling([[maybe_unused]] gert::TilingParseContext* context) {
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingGeluMulTiling(gert::TilingContext* context) {
    GeluMulTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

IMPL_OP_OPTILING(GeluMul)
    .Tiling(TilingGeluMulTiling)
    .TilingParse<GeluMulCompileInfo>(TilingPrepare4GeluMulTiling);
}