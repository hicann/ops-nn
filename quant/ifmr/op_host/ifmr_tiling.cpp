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
 * \file ifmr_tiling.cpp
 * \brief
 */
#include "ifmr_tiling.h"
#include "log/log.h"
#include "error_util.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_templates_registry.h"
#include "ifmr/op_kernel/ifmr_tiling_data.h"

namespace optiling {

struct IfmrCompileInfo {
    int32_t dataNum;
};

ge::graphStatus IfmrTiling::CheckIfmrTilingAttrs(void)
{
    if (attrs_.minPercentile <= K_PERCENTILE_LOW_BOUND || attrs_.minPercentile > K_PERCENTILE_UPPER_BOUND) {
        OP_LOGE(nodeName_, "The minPercentile must be greater than 0.5 "\
            "and less than or equal to 1.0");
        return ge::GRAPH_FAILED;
    }
    if (attrs_.maxPercentile <= K_PERCENTILE_LOW_BOUND || attrs_.maxPercentile > K_PERCENTILE_UPPER_BOUND) {
        OP_LOGE(nodeName_, "The maxPercentile must be greater than 0.5 "\
            "and less than or equal to 1.0");
        return ge::GRAPH_FAILED;
    }
    if (attrs_.searchRange[0] <= 0) {
        OP_LOGE(nodeName_, "search_start must be greater than zero.");
        return ge::GRAPH_FAILED;
    }
    if (attrs_.searchRange[0] >= attrs_.searchRange[1]) {
        OP_LOGE(nodeName_, "search_start must be less than search_end.");
        return ge::GRAPH_FAILED;
    }
    if (attrs_.searchStep <= 0) {
        OP_LOGE(nodeName_, "The searchStep must be greater than zero.");
        return ge::GRAPH_FAILED;
    }
    if (std::round((attrs_.searchRange[1] - attrs_.searchRange[0]) / attrs_.searchStep) + 1 > MAX_STEP_NUMS) {
        OP_LOGE(nodeName_, "step size should be equal or less than 4096");
        return ge::GRAPH_FAILED;
    }
    if (std::find(SUPPORTED_QUANT_BITS, (SUPPORTED_QUANT_BITS + SUPPORT_QUANT_BITS_NUM), attrs_.quantBits) ==
        (SUPPORTED_QUANT_BITS + SUPPORT_QUANT_BITS_NUM)) {
        OP_LOGE(nodeName_, "quant bits only support 8 or 16");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus IfmrTiling::GetIfmrTilingAttrInfo(void)
{
    OP_LOGD(nodeName_, "[IFMR] GetIfmrTilingAttrInfo start running");
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetAttrs());
    auto minPercentilPtr = context_->GetAttrs()->GetFloat(ATTR_MIN_PERCENTILE_INDEX);
    auto maxPercentilPtr = context_->GetAttrs()->GetFloat(ATTR_MAX_PERCENTILE_INDEX);
    auto searchRangePtr = context_->GetAttrs()->GetListFloat(ATTR_SEARCH_RANGE_INDEX);
    auto searchStepPtr = context_->GetAttrs()->GetFloat(ATTR_SEARCH_STEP_INDEX);
    auto withOffsetPtr = context_->GetAttrs()->GetBool(ATTR_WITH_OFFSET_INDEX);
    auto quantBitsPtr = context_->GetAttrs()->GetInt(ATTR_QUANT_BITS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, minPercentilPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, maxPercentilPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, searchRangePtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, searchStepPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, withOffsetPtr);
    OP_CHECK_NULL_WITH_CONTEXT(context_, quantBitsPtr);
    attrs_.minPercentile = *minPercentilPtr;
    attrs_.maxPercentile = *maxPercentilPtr;
    if (searchRangePtr->GetSize() != ATTR_SEARCH_RANGE_SIZE) {
        OP_LOGE(nodeName_, "[IFMR] step_range dim should be 2");
        return ge::GRAPH_FAILED;
    }
    attrs_.searchRange[0] = searchRangePtr->GetData()[0];
    attrs_.searchRange[1] = searchRangePtr->GetData()[1];
    attrs_.searchStep = *searchStepPtr;
    attrs_.withOffset = *withOffsetPtr;
    attrs_.quantBits = *quantBitsPtr;
    return CheckIfmrTilingAttrs();
}
ge::graphStatus IfmrTiling::CheckIfmrTilingInputDataShape(uint32_t inputIndex, std::string inputName)
{
    auto dataShapePtr = context_->GetInputShape(inputIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dataShapePtr);
    auto dataShape = dataShapePtr->GetStorageShape();
    auto dataDataDim = dataShape.GetDimNum();
    if (dataDataDim != 1 || dataShapePtr->GetStorageShape().GetDim(0) != 1) {
        OP_LOGE(nodeName_, "The shape of %s must be [1]!", inputName.c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus IfmrTiling::CheckIfmrTilingInputDtype(void)
{
    // check input dtype
    auto dataDesc = context_->GetInputDesc(DATA_INPUT_INDEX);
    auto dataDtype = dataDesc->GetDataType();
    if (dataDtype != ge::DataType::DT_FLOAT && \
        dataDtype != ge::DataType::DT_FLOAT16) {
        OP_LOGE(nodeName_, "Input data support only DT_FLOAT and DT_FLOAT16!");
        return ge::GRAPH_FAILED;
    }
    auto dataMinDesc = context_->GetInputDesc(DATA_MIN_INPUT_INDEX);
    auto dataMinDtype = dataMinDesc->GetDataType();
    auto dataMaxDesc = context_->GetInputDesc(DATA_MAX_INPUT_INDEX);
    auto dataMaxDtype = dataMaxDesc->GetDataType();
    if (dataDtype != dataMaxDtype || dataDtype != dataMinDtype) {
        OP_LOGE(nodeName_, "Input data, data_max, data_min must have same data type!");
        return ge::GRAPH_FAILED;
    }
    auto cumsumDesc = context_->GetInputDesc(CUMSUM_INPUT_INDEX);
    if (cumsumDesc->GetDataType() != ge::DataType::DT_INT32) {
        OP_LOGE(nodeName_, "Input cumsum support only DT_INT32!");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus IfmrTiling::GetDataLength(void)
{
    auto dataShapePtr = context_->GetInputShape(DATA_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, dataShapePtr);
    auto dataShape = dataShapePtr->GetStorageShape();
    auto dataDataDim = dataShape.GetDimNum();
    uint64_t dataLength = 1;
    for (uint32_t i = 0; i < dataDataDim; i++) {
        int64_t shape = dataShapePtr->GetStorageShape().GetDim(i);
        if (shape <= 0) {
            OP_LOGE(nodeName_, "The input shape should be greater than 0!");
            return ge::GRAPH_FAILED;
        }
        uint64_t preDataLength = dataLength;
        dataLength *= shape;
        // DataLength overflow validity check
        if ((dataLength > SHAPE_SIZE_LIMIT) || ((shape != 0) && (dataLength / shape != preDataLength))) {
            OP_LOGE(nodeName_, "Excessive amount of input_data(more than 2^31)!");
            return ge::GRAPH_FAILED;
        }
    }
    attrs_.dataLength = dataLength;
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus IfmrTiling::GetIfmrTilingInputInfo(void)
{
    // check input shape
    if (GetDataLength() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto cumsumShapePtr = context_->GetInputShape(CUMSUM_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cumsumShapePtr);
    auto cumSumShape = cumsumShapePtr->GetStorageShape();
    auto cumsumDataDim = cumSumShape.GetDimNum();
    if (cumsumDataDim != 1) {
        OP_LOGE(nodeName_, "The shape of input_cumsum must be (x,)!");
        return ge::GRAPH_FAILED;
    }
    uint32_t cumsumLength = cumsumShapePtr->GetStorageShape().GetDim(0);
    if (cumsumLength > MAX_CUMSUM_LENGTH || cumsumLength == 0) {
        OP_LOGE(nodeName_, "Excessive amount of input_cumsum(more than 8192) or cumsum_len is 0!");
        return ge::GRAPH_FAILED;
    }
    attrs_.cumsumLength = cumsumLength;
    if (CheckIfmrTilingInputDataShape(DATA_MIN_INPUT_INDEX, "input_min") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckIfmrTilingInputDataShape(DATA_MAX_INPUT_INDEX, "input_max") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckIfmrTilingInputDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSizePlatform;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatform);
    if (ubSizePlatform <= UB_SIZE_RESERVE) {
        OP_LOGE(nodeName_, "UB size is not enough!");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus IfmrTiling::CheckIfmrTilingOutputInfo(void)
{
    auto scaleDesc = context_->GetOutputDesc(0);
    auto offsetDesc = context_->GetOutputDesc(1);
    OP_TILING_CHECK(scaleDesc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "scaleDesc cannot be nullptr!"),
        return ge::GRAPH_FAILED);
    OP_TILING_CHECK(offsetDesc == nullptr, VECTOR_INNER_ERR_REPORT_TILIING(nodeName_, "offsetDesc cannot be nullptr!"),
        return ge::GRAPH_FAILED);
    if (scaleDesc->GetDataType() != ge::DataType::DT_FLOAT) {
        OP_LOGE(nodeName_, "Output scale support only DT_FLOAT!");
        return ge::GRAPH_FAILED;
    }
    if (offsetDesc->GetDataType() != ge::DataType::DT_FLOAT) {
        OP_LOGE(nodeName_, "Output offset support only DT_FLOAT!");
        return ge::GRAPH_FAILED;
    }
    auto scaleShapePtr = context_->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, scaleShapePtr);
    auto scaleShape = scaleShapePtr->GetStorageShape();
    auto scaleDataDim = scaleShape.GetDimNum();
    if (scaleDataDim != 1 || scaleShapePtr->GetStorageShape().GetDim(0) != 1) {
        OP_LOGE(nodeName_, "The shape of scale must be [1]!");
        return ge::GRAPH_FAILED;
    }
    auto offsetShapePtr = context_->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, offsetShapePtr);
    auto offsetShape = offsetShapePtr->GetStorageShape();
    auto offsetDataDim = offsetShape.GetDimNum();
    if (offsetDataDim != 1 || offsetShapePtr->GetStorageShape().GetDim(0) != 1) {
        OP_LOGE(nodeName_, "The shape of offset must be [1]!");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
void IfmrTiling::SetIfmrTiling(void)
{
    OP_LOGD(nodeName_, "[IFMR] SetIfmrTiling start running");
    IfmrTilingData *tilingData = context_->GetTilingData<IfmrTilingData>();
    (void)memset_s(tilingData, sizeof(IfmrTilingData), 0, sizeof(IfmrTilingData));
    tilingData->minPercentile = attrs_.minPercentile;
    tilingData->maxPercentile = attrs_.maxPercentile;
    tilingData->searchRange[0] = attrs_.searchRange[0];
    tilingData->searchRange[1] = attrs_.searchRange[1];
    tilingData->searchStep = attrs_.searchStep;
    tilingData->withOffset = attrs_.withOffset;
    tilingData->quantBits = attrs_.quantBits;
    tilingData->dataLength = attrs_.dataLength;
    tilingData->cumsumLength = attrs_.cumsumLength;
    OP_LOGI("IFMR", "nodeName: %s, minPercentile: %lf, maxPercentile: %lf, searchRange: [%lf, %lf],\
        searchStep: %lf, withOffset: %d, quantBits: %d, dataLength: %d, cumsumLength: %d",
        nodeName_, tilingData->minPercentile, tilingData->maxPercentile, 
        tilingData->searchRange[0], tilingData->searchRange[1], tilingData->searchStep,
        tilingData->withOffset, tilingData->quantBits,
        tilingData->dataLength, tilingData->cumsumLength);
    return;
}
void IfmrTiling::PostTiling(void)
{
    context_->SetTilingKey(0);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    context_->SetBlockDim(ascendcPlatform.GetCoreNumAiv());
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    size_t workspaceSize = static_cast<size_t>(20) * 1024 * 1024; // 16M for AscendC framework, 4M reserved for ifmr op
    workspaces[0] = workspaceSize;
    OP_LOGD(nodeName_, "[IFMR] PostTiling run completed");
}

// tiling 分发入口
ge::graphStatus IfmrTiling::IfmrTilingFunc(void)
{
    OP_LOGD(nodeName_, "[IFMR] RunIfmrTiling start running");

    auto ret = GetIfmrTilingAttrInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = GetIfmrTilingInputInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckIfmrTilingOutputInfo();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    SetIfmrTiling();
    PostTiling();
    return ret;
}

static ge::graphStatus TilingForIfmr(gert::TilingContext* context)
{
    OP_TILING_CHECK(context == nullptr, VECTOR_INNER_ERR_REPORT_TILIING("IFMR", "context should not be nullptr."),
                return ge::GRAPH_FAILED);
    IfmrTiling tiling(context);
    return tiling.IfmrTilingFunc();
}

static ge::graphStatus TilingParseForIfmr(gert::TilingParseContext* context)
{
    // The operator does not need compile_info, return success
    (void)context;
    return ge::GRAPH_SUCCESS;
}

// tiling注册入口.
IMPL_OP_OPTILING(IFMR).Tiling(TilingForIfmr).TilingParse<IfmrCompileInfo>(TilingParseForIfmr);
} // namespace optiling
