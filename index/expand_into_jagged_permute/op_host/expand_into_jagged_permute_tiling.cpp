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
 * \file expand_into_jagged_permute_tiling.cpp
 * \brief expand_into_jagged_permute_tiling.cpp
 */
#include "expand_into_jagged_permute_tiling.h"
#include "register/op_impl_registry.h"
#include "util/math_util.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "tiling/platform/platform_ascendc.h"


namespace optiling {

template <typename T>
static T GetCeilInt(const T& value1, const T& value2) {
    if (value2 == 0) {
        return value2;
    }
    return (value1 + value2 - 1) / value2;
    }

static inline ge::graphStatus InputParamCheck(const gert::TilingContext* context) {
    const gert::StorageShape* permuteShape = context->GetInputShape(0);
    const gert::StorageShape* inputoffsetShape = context->GetInputShape(1);
    const gert::StorageShape* outputoffsetShape = context->GetInputShape(2);

    auto dataTensor0 = context->GetInputTensor(0);
    auto dataTensor1 = context->GetInputTensor(1);
    auto dataTensor2 = context->GetInputTensor(2);
    auto nodeName = context->GetNodeName();

    auto permuteTypePtr = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, permuteTypePtr);
    auto permuteType = permuteTypePtr->GetDataType();

    auto inputoffsetTypePtr = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputoffsetTypePtr);
    auto inputoffsetType = inputoffsetTypePtr->GetDataType();

    auto outputoffsetTypePtr = context->GetInputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputoffsetTypePtr);
    auto outputoffsetType = outputoffsetTypePtr->GetDataType();

    OP_CHECK_IF(
        permuteType != inputoffsetType && permuteType != outputoffsetType && inputoffsetType != outputoffsetType,
        OP_LOGE(nodeName, "[ExpandIntoJaggedPermuteTilingData] the datatype of permute、inputoffset and outputoffset is not same."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        permuteShape == nullptr || inputoffsetShape == nullptr || outputoffsetShape == nullptr || dataTensor0 == nullptr || dataTensor1 == nullptr||
        dataTensor2 == nullptr, OP_LOGE(nodeName, "[ExpandIntoJaggedPermuteTilingData] permute or inputOffsets is nullptr or outputoff is nullptr."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(permuteShape->GetStorageShape().GetDimNum() != 1,
                    OP_LOGE(nodeName, "[ExpandIntoJaggedPermuteTilingData] permuteShape's shape is not 1D."),
                    return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(inputoffsetShape->GetStorageShape().GetDimNum() != 1,
                    OP_LOGE(nodeName, "[ExpandIntoJaggedPermuteTilingData] inputoffsetShape's shape is not 1D."),
                    return ge::GRAPH_FAILED);

    OP_CHECK_IF(outputoffsetShape->GetStorageShape().GetDimNum() != 1,
                    OP_LOGE(nodeName, "[ExpandIntoJaggedPermuteTilingData] outputoffsetShape's shape is not 1D."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;

    OP_CHECK_IF(permuteShape->GetStorageShape().GetDim(0) + 1 != outputoffsetShape->GetStorageShape().GetDim(0) ||
                    permuteShape->GetStorageShape().GetDim(0) + 1 != inputoffsetShape->GetStorageShape().GetDim(0),
                    OP_LOGE(nodeName, "[ExpandIntoJaggedPermuteTilingData] outputoffsetShape and inputoffsetShape should be one more than permuteShape."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static inline void Init(gert::TilingContext *context, ExpandIntoJaggedPermuteParam& param)
{
    size_t sysWorkspaceSize = 16 * 1024 * 1024;
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;
    OP_LOGI(context, "get platform from compileInfo");

    auto ascendPlaform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    param.coreNum = static_cast<int64_t>(ascendPlaform.GetCoreNumAiv());
    uint64_t maxCoreMemery;
    ascendPlaform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, maxCoreMemery);
    param.maxCoreMemery = static_cast<int64_t>(maxCoreMemery);

    const gert::StorageShape* permuteShape = context->GetInputShape(0);  //inputLen
    const gert::StorageShape* inputoffsetShape = context->GetInputShape(1);
    const gert::StorageShape* outputoffsetShape = context->GetInputShape(2);

    auto attrs = context->GetAttrs();
    const int32_t* outputSizePtr = attrs->GetAttrPointer<int32_t>(0);
    param.outputSize = *outputSizePtr;

    param.permute = permuteShape->GetStorageShape().GetDim(0); 
    param.inputoffset = inputoffsetShape->GetStorageShape().GetDim(0);
    param.outputoffset = outputoffsetShape->GetStorageShape().GetDim(0);
    param.dtypeSize = ge::GetSizeByDataType(context->GetInputDesc(0)->GetDataType());

    param.oneTaskInputOffsetLen = SIZE;
    param.cacheLine = SIZE;
    param.numBlocks = param.coreNum;
    param.oneTaskLen = param.cacheLine / param.dtypeSize; 
    param.taskNum = GetCeilInt(param.permute, param.oneTaskInputOffsetLen); 
    param.numTail = param.permute - SIZE * (param.taskNum - 1);
}

static ge::graphStatus SetTilingKey(ExpandIntoJaggedPermuteParam& param) {
    uint64_t tilingkey = 0;
    int64_t limitedSize = 512;
    if(param.inputoffset <= limitedSize){
        param.tilingKey = tilingkey;
    }else{
        param.tilingKey = tilingkey + 1;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetTilingData(gert::TilingContext *context, const ExpandIntoJaggedPermuteParam& param)
{
    ExpandIntoJaggedPermuteTilingData tilingData;
    tilingData.set_outputSize(param.outputSize);
    context->SetBlockDim(param.numBlocks);
    tilingData.set_blockFactor(param.taskNum);
    tilingData.set_inputLen(param.permute);
    tilingData.set_offsetLen(param.permute + 1);
    tilingData.set_lastTaskLen(param.numTail);
    tilingData.set_oneTaskOffsetLen(param.oneTaskInputOffsetLen); 
    tilingData.set_oneTaskLen(param.oneTaskLen);
    auto rawTilingData = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());
    context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static inline void PrintTilingData(const gert::TilingContext* context, const ExpandIntoJaggedPermuteParam& param) {
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print ExpandIntoJaggedPermute tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD("offsetLen", " param.offsetLen %ld", param.offsetLen);
    OP_LOGD("inputLen", " param.permute %ld", param.permute);
    OP_LOGD("oneTaskLen", " param.oneTaskLen %ld", param.oneTaskLen);
    OP_LOGD("oneTaskOffsetLen", " param.oneTaskInputOffsetLen %ld", param.oneTaskInputOffsetLen);
    OP_LOGD("numBlocks", " param.numBlocks %ld", param.numBlocks);
    OP_LOGD("taskNum", " param.taskNum %ld" , param.taskNum);
    OP_LOGD("outputSize", " param.outputSize %ld", param.outputSize);
    OP_LOGD("numTail", " param.numTail %ld", param.numTail);
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print ExpandIntoJaggedPermute tiling data end <<<<<<<<<<<<<<<<");
}

static ge::graphStatus TilingForExpandIntoJaggedPermute(gert::TilingContext* context)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> ExpandIntoJaggedPermute Tiling start <<<<<<<<<<<<<<<<.");

    if (InputParamCheck(context) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    ExpandIntoJaggedPermuteParam param;
    Init(context, param);
    SetTilingKey(param);
    SetTilingData(context, param);
    PrintTilingData(context, param);
    return context->SetTilingKey(param.tilingKey);
}


ge::graphStatus TilingPrepareExpandIntoJaggedPermute(gert::TilingParseContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}


IMPL_OP_OPTILING(ExpandIntoJaggedPermute)
    .Tiling(TilingForExpandIntoJaggedPermute)
    .TilingParse<ExpandIntoJaggedPermuteCompileInfo>(TilingPrepareExpandIntoJaggedPermute);
} // namespace optiling