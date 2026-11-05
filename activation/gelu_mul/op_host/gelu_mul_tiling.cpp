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
 * \file gelu_mul_tiling.cpp
 * \brief gelu_mul_tiling source file
 */
#include "gelu_mul_tiling.h"
#include <vector>
#include <iostream>
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include <string>
#include "base/err_msg.h"

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
    explicit GeluMulTiling(gert::TilingContext* context) : tilingContext(context) {};
    ge::graphStatus RunBigKernelTiling();
    ge::graphStatus FillTilingKey();

private:
    ge::graphStatus ShapeCheck();
    ge::graphStatus OutShapeCheck();
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

    inline int32_t CeilA2B(const int32_t a, const int32_t b)
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

ge::graphStatus GeluMulTiling::ShapeCheck()
{
    OP_CHECK_IF(
        (lastDimSize > LENGTH_1024),
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(tilingContext->GetNodeName(), "input", std::to_string(lastDimSize),
                                                  "Last dim size should be no more than 1024."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF((lastDimSize % SIZE_2 == 1),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(tilingContext->GetNodeName(), "input",
                                                          std::to_string(lastDimSize), "Last dim size should be even."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        (batchSize > LENGTH_LIMIT),
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(tilingContext->GetNodeName(), "input", std::to_string(batchSize),
                                                  "Batch dim size should be no more than 200000."),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluMulTiling::OutShapeCheck()
{
    // 直调aclnn两段式接口时，tiling上下文的输出shape为用户声明的out；图模式执行时为推导结果，天然一致。
    // 声明值与推导值不一致时拒绝，避免kernel按输入推导的元素数写入导致越界。
    auto outShapePtr = tilingContext->GetOutputShape(0);
    if (outShapePtr == nullptr) {
        const std::string msgOpName = std::string(
            ((tilingContext) == nullptr || (tilingContext)->GetNodeName() == nullptr ? "gelu_mul" :
                                                                                       (tilingContext)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(msgOpName.c_str(), "%s",
                               "Tiling framework object outShape is unavailable; check the operator registration and "
                               "context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "Tiling framework object outShape is unavailable; check the "
                                                         "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
    const gert::Shape& outShape = outShapePtr->GetStorageShape();
    size_t outDimNum = outShape.GetDimNum();
    // 含未知维（动态shape）时不校验
    for (size_t idx = 0; idx < outDimNum; idx++) {
        if (outShape.GetDim(idx) < 0) {
            return ge::GRAPH_SUCCESS;
        }
    }
    size_t inputShapeDim = inputShape.GetDimNum();
    OP_CHECK_IF(
        (outDimNum != inputShapeDim),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            tilingContext->GetNodeName(), "input,output",
            std::to_string(inputShapeDim) + "," + std::to_string(outDimNum),
            ("The dims of output should be equal to the dims of input, but got " + std::to_string(outDimNum) + ".")),
        return ge::GRAPH_FAILED);
    for (size_t idx = 0; idx + 1 < inputShapeDim; idx++) {
        OP_CHECK_IF((outShape.GetDim(idx) != inputShape.GetDim(idx)),
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        tilingContext->GetNodeName(), "output", std::to_string(outShape.GetDim(idx)),
                        ("Except the last dim, the output shape should be the same with input shape, but got dim[" +
                         std::to_string(idx) + "] = " + std::to_string(outShape.GetDim(idx)) + ".")),
                    return ge::GRAPH_FAILED);
    }
    OP_CHECK_IF((outShape.GetDim(inputShapeDim - 1) != inputShape.GetDim(inputShapeDim - 1) / SIZE_2),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    tilingContext->GetNodeName(), "output", std::to_string(outShape.GetDim(inputShapeDim - 1)),
                    ("The last dim of output should be half of the input's last dim, but got " +
                     std::to_string(outShape.GetDim(inputShapeDim - 1)) + ".")),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GeluMulTiling::RunBigKernelTiling()
{
    // 获取输入矩阵
    auto srcTensor = tilingContext->GetInputTensor(0);
    OP_CHECK_IF(srcTensor == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(((tilingContext) == nullptr || (tilingContext)->GetNodeName() == nullptr ?
                                                "gelu_mul" :
                                                (tilingContext)->GetNodeName()),
                                           "input"),
                return ge::GRAPH_FAILED);
    // 获取输入的参数
    const gert::RuntimeAttrs* attrs = tilingContext->GetAttrs();
    if (attrs == nullptr) {
        const std::string msgOpName = std::string(
            ((tilingContext) == nullptr || (tilingContext)->GetNodeName() == nullptr ? "gelu_mul" :
                                                                                       (tilingContext)->GetNodeName()));
        OP_LOGE_WITHOUT_REPORT(
            msgOpName.c_str(), "%s",
            "Tiling framework object attrs is unavailable; check the operator registration and context construction.");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{msgOpName.c_str(), "Tiling framework object attrs is unavailable; check the "
                                                         "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    };
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
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(tilingContext->GetNodeName(), "approximate", approximate,
                                                      "Approximate can only be 'tanh' or 'none'."),
                return ge::GRAPH_FAILED);
    FillTilingKey();
    // 获取输入的shape
    auto srcShape = tilingContext->GetInputShape(0);
    inputShape = srcShape->GetOriginShape();
    size_t inputShapeDim = inputShape.GetDimNum();
    OP_CHECK_IF(
        (inputShapeDim < static_cast<size_t>(SIZE_2)),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(tilingContext->GetNodeName(), "input", std::to_string(inputShapeDim),
                                                 "Input shape dim should be no less than 2."),
        return ge::GRAPH_FAILED);
    lastDimSize = inputShape.GetDim(inputShapeDim - 1);
    inputShapeSize = inputShape.GetShapeSize();
    OP_CHECK_IF(
        (lastDimSize == 0),
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(tilingContext->GetNodeName(), "input", std::to_string(lastDimSize),
                                                  "Last dim elements can not be zero."),
        return ge::GRAPH_FAILED);

    batchSize = inputShapeSize / lastDimSize;

    // 形状约束校验需先于分核计算：末维为1时GetNeedCoreNum中dAlign为0，先校验末维合法性避免整除零
    OP_CHECK_IF((ShapeCheck() == ge::GRAPH_FAILED),
                OP_LOGE_WITHOUT_REPORT(tilingContext->GetNodeName(), "ShapeCheck failed!"), return ge::GRAPH_FAILED);
    OP_CHECK_IF((OutShapeCheck() == ge::GRAPH_FAILED),
                OP_LOGE_WITHOUT_REPORT(tilingContext->GetNodeName(), "OutShapeCheck failed!"), return ge::GRAPH_FAILED);

    auto platformInfo = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    int32_t needCoreNum = GetNeedCoreNum(platformInfo.GetCoreNumAiv());

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(workspaceSize_);
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

ge::graphStatus GeluMulTiling::FillTilingKey()
{
    // 获取数据类型
    auto temp = tilingContext->GetInputDesc(0);
    OP_CHECK_IF(temp == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(((tilingContext) == nullptr || (tilingContext)->GetNodeName() == nullptr ?
                                                "gelu_mul" :
                                                (tilingContext)->GetNodeName()),
                                           "input"),
                return ge::GRAPH_FAILED);
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

static ge::graphStatus TilingPrepare4GeluMulTiling([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingGeluMulTiling(gert::TilingContext* context)
{
    GeluMulTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

IMPL_OP_OPTILING(GeluMul).Tiling(TilingGeluMulTiling).TilingParse<GeluMulCompileInfo>(TilingPrepare4GeluMulTiling);
} // namespace optiling
