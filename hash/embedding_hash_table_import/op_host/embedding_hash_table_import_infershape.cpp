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
 * \file embedding_hash_table_import_infershape.cpp
 * \brief embedding_hash_table_import infer
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_templates_registry.h"

using namespace ge;
namespace ops {
// ------------------- HashTableImport ops START---------------------
static constexpr int64_t TABLE_HANDLES_IDX = 0;
static constexpr int64_t EMBEDDING_DIMS_IDX = 1;
static constexpr int64_t BUCKET_SIZES_IDX = 2;
static constexpr int64_t KEYS_IDX = 3;
static constexpr int64_t COUNTERS_IDX = 4;
static constexpr int64_t FILTER_FLAGS_IDX = 5;
static constexpr int64_t VALUES_IDX = 6;

static constexpr int64_t INPUT_NODE_NUM = 7;

ge::graphStatus Infershape4EmbeddingHashTableImport(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do EmbeddingHashTableImportInfershape.");

    // 获取输入值shape
    const gert::Shape *tableHandleShape = context->GetRequiredInputShape(TABLE_HANDLES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, tableHandleShape);

    OP_CHECK_IF(
        tableHandleShape->GetDimNum() > 1,
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            context->GetNodeName(), "table_handles", std::to_string(tableHandleShape->GetDimNum()).c_str(), "1D"),
        return ge::GRAPH_FAILED);

    const gert::Shape *embeddingDimShape = context->GetRequiredInputShape(EMBEDDING_DIMS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, embeddingDimShape);
    auto embeddingDimCount = embeddingDimShape->GetShapeSize();

    const gert::Shape *bucketSizeShape = context->GetRequiredInputShape(BUCKET_SIZES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, bucketSizeShape);
    auto bucketSizeCount = bucketSizeShape->GetShapeSize();

    OP_CHECK_IF(
        embeddingDimCount != bucketSizeCount,
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            context->GetNodeName(), "embedding_dims and bucket_sizes",
            (std::to_string(embeddingDimCount) + " and " + std::to_string(bucketSizeCount)).c_str(),
            "The shape sizes of embedding_dims and bucket_sizes must be the same"),
        return ge::GRAPH_FAILED);

    // 获取动态输入值shape
    const auto keysInfo = context->GetIrInputInstanceInfo(KEYS_IDX);
    for (uint32_t i = 0; i < keysInfo->GetInstanceNum(); i++) {
        auto keysShape = context->GetDynamicInputShape(KEYS_IDX, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, keysShape);
    }

    const auto countersInfo = context->GetIrInputInstanceInfo(COUNTERS_IDX);
    for (uint32_t i = 0; i < countersInfo->GetInstanceNum(); i++) {
        auto countersShape = context->GetDynamicInputShape(COUNTERS_IDX, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, countersShape);
    }

    const auto filterFlagsInfo = context->GetIrInputInstanceInfo(FILTER_FLAGS_IDX);
    for (uint32_t i = 0; i < filterFlagsInfo->GetInstanceNum(); i++) {
        auto filterFlagsShape = context->GetDynamicInputShape(FILTER_FLAGS_IDX, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, filterFlagsShape);
    }

    const auto valuesInfo = context->GetIrInputInstanceInfo(VALUES_IDX);
    for (uint32_t i = 0; i < valuesInfo->GetInstanceNum(); i++) {
        auto valuesShape = context->GetDynamicInputShape(VALUES_IDX, i);
        OP_CHECK_NULL_WITH_CONTEXT(context, valuesShape);
    }

    OP_LOGD(context->GetNodeName(), "End to do EmbeddingHashTableImportInfershape.");

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4EmbeddingHashTableImport(gert::InferDataTypeContext *context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do EmbeddingHashTableImportInferDataType.");
    auto computeNodeInfo = context->GetComputeNodeInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, computeNodeInfo);
    int64_t totalInNum = computeNodeInfo->GetIrInputsNum();    
    if (!(totalInNum == INPUT_NODE_NUM)) {
        OP_LOGE(context->GetNodeName(), "Check input nums failed, actual InputNum is %ld", totalInNum);
        return GRAPH_FAILED;
    }
    auto tableHandleDtype = context->GetInputDataType(TABLE_HANDLES_IDX);    // table_handle
    OP_CHECK_IF(
        tableHandleDtype != DT_INT64,
        OP_LOGE_FOR_INVALID_DTYPE(
            context->GetNodeName(), "table_handle", ge::TypeUtils::DataTypeToSerialString(tableHandleDtype).c_str(),
            "int64"),
        return ge::GRAPH_FAILED);

    auto embeddingDimsDtype = context->GetInputDataType(EMBEDDING_DIMS_IDX);  // embedding_dims
    OP_CHECK_IF(
        embeddingDimsDtype != DT_INT64,
        OP_LOGE_FOR_INVALID_DTYPE(
            context->GetNodeName(), "embedding_dims", ge::TypeUtils::DataTypeToSerialString(embeddingDimsDtype).c_str(),
            "int64"),
        return ge::GRAPH_FAILED);

    auto bucketSizesDtype = context->GetInputDataType(BUCKET_SIZES_IDX);     // bucket_sizes
    OP_CHECK_IF(
        bucketSizesDtype != DT_INT64,
        OP_LOGE_FOR_INVALID_DTYPE(
            context->GetNodeName(), "bucket_sizes", ge::TypeUtils::DataTypeToSerialString(bucketSizesDtype).c_str(),
            "int64"),
        return ge::GRAPH_FAILED);

    const auto keysInfo = context->GetIrInputInstanceInfo(KEYS_IDX);         // keys
    for (uint32_t i = 0; i < keysInfo->GetInstanceNum(); i++) {
        auto keysDtype = context->GetDynamicInputDataType(KEYS_IDX, i);
        OP_CHECK_IF(
            keysDtype != DT_INT64,
            OP_LOGE_FOR_INVALID_DTYPE(
                context->GetNodeName(), "keys", ge::TypeUtils::DataTypeToSerialString(bucketSizesDtype).c_str(),
                "int64"),
            return ge::GRAPH_FAILED);
    }

    const auto countersInfo = context->GetIrInputInstanceInfo(COUNTERS_IDX); // counters
    for (uint32_t i = 0; i < countersInfo->GetInstanceNum(); i++) {
        auto countersDtype = context->GetDynamicInputDataType(COUNTERS_IDX, i);
        if(countersDtype != DT_UINT64){
            std::string errMsg = "The datatype of " + std::to_string(i) +
                "th counters must be same as uint64";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context->GetNodeName(), "counters", ge::TypeUtils::DataTypeToSerialString(countersDtype).c_str(),
                errMsg.c_str());
        }
    }

    const auto filterFlagsInfo = context->GetIrInputInstanceInfo(FILTER_FLAGS_IDX); // filter_flags
    for (uint32_t i = 0; i < filterFlagsInfo->GetInstanceNum(); i++) {
        auto filterFlagsDtype = context->GetDynamicInputDataType(FILTER_FLAGS_IDX, i);
        if (filterFlagsDtype != DT_UINT8) {
            std::string errMsg = "The datatype of " + std::to_string(i) +
                "th filter_flags must be same as uint8";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context->GetNodeName(), "filter_flags", ge::TypeUtils::DataTypeToSerialString(filterFlagsDtype).c_str(),
                errMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    const auto valuesInfo = context->GetIrInputInstanceInfo(VALUES_IDX); // values
    for (uint32_t i = 0; i < valuesInfo->GetInstanceNum(); i++) {
        auto valuesDtype = context->GetDynamicInputDataType(VALUES_IDX, i);
        if (valuesDtype != DT_FLOAT) {
            std::string errMsg = "The datatype of " + std::to_string(i) +
                "th values must be same as float";
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                context->GetNodeName(), "values", ge::TypeUtils::DataTypeToSerialString(valuesDtype).c_str(),
                errMsg.c_str());
            return ge::GRAPH_FAILED;
        }
    }

    OP_LOGD(context->GetNodeName(), "End to do EmbeddingHashTableImportInferDataType.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(EmbeddingHashTableImport)
    .InferShape(Infershape4EmbeddingHashTableImport)
    .InferDataType(InferDataType4EmbeddingHashTableImport);

// -------------------EmbeddingHashTableImport Ops END---------------------
} // namespace ops