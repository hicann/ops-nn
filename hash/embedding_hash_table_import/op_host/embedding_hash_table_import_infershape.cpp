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

    OP_CHECK_IF(tableHandleShape->GetDimNum() > 1,
        OP_LOGE(context->GetNodeName(), "the rank of table shape must be 1, but shape is %s", Ops::Base::ToString(*tableHandleShape).c_str()),
        return ge::GRAPH_FAILED);

    const gert::Shape *embeddingDimShape = context->GetRequiredInputShape(EMBEDDING_DIMS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, embeddingDimShape);
    auto embeddingDimCount = embeddingDimShape->GetShapeSize();

    const gert::Shape *bucketSizeShape = context->GetRequiredInputShape(BUCKET_SIZES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, bucketSizeShape);
    auto bucketSizeCount = bucketSizeShape->GetShapeSize();

    OP_CHECK_IF(embeddingDimCount != bucketSizeCount,
        OP_LOGE(context->GetNodeName(), "the size of embedding not equal with bucket, please check."),
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
    OP_CHECK_IF(tableHandleDtype != DT_INT64, OP_LOGE(context->GetNodeName(), "tableHandleDtype [%d] is not match int64.",
             tableHandleDtype), return ge::GRAPH_FAILED);

    auto embeddingDimsDtype = context->GetInputDataType(EMBEDDING_DIMS_IDX);  // embedding_dims
    OP_CHECK_IF(embeddingDimsDtype != DT_INT64, OP_LOGE(context->GetNodeName(),
            "embeddingDimsDtype [%d] is not match int64.", embeddingDimsDtype),
             return ge::GRAPH_FAILED);

    auto bucketSizesDtype = context->GetInputDataType(BUCKET_SIZES_IDX);     // bucket_sizes
    OP_CHECK_IF(bucketSizesDtype != DT_INT64, OP_LOGE(context->GetNodeName(), "bucketSizesDtype [%d] is not match int64.",
             bucketSizesDtype), return ge::GRAPH_FAILED);

    const auto keysInfo = context->GetIrInputInstanceInfo(KEYS_IDX);         // keys
    for (uint32_t i = 0; i < keysInfo->GetInstanceNum(); i++) {
        auto keysDtype = context->GetDynamicInputDataType(KEYS_IDX, i);
        OP_CHECK_IF(keysDtype != DT_INT64, OP_LOGE(context->GetNodeName(), "%uth keysDtype [%d] is not match int64.", i, 
                  keysDtype), return ge::GRAPH_FAILED);
    }

    const auto countersInfo = context->GetIrInputInstanceInfo(COUNTERS_IDX); // counters
    for (uint32_t i = 0; i < countersInfo->GetInstanceNum(); i++) {
        auto countersDtype = context->GetDynamicInputDataType(COUNTERS_IDX, i);
        OP_CHECK_IF(countersDtype != DT_UINT64, OP_LOGE(context->GetNodeName(),
                 "%uth countersDtype [%d] is not match uint64.", i, countersDtype),
                 return ge::GRAPH_FAILED);
    }

    const auto filterFlagsInfo = context->GetIrInputInstanceInfo(FILTER_FLAGS_IDX); // filter_flags
    for (uint32_t i = 0; i < filterFlagsInfo->GetInstanceNum(); i++) {
        auto filterFlagsDtype = context->GetDynamicInputDataType(FILTER_FLAGS_IDX, i);
        OP_CHECK_IF(filterFlagsDtype != DT_UINT8, OP_LOGE(context->GetNodeName(),
                 "%uth filterFlagsDtype [%d] is not match uint8.", i, filterFlagsDtype),
                 return ge::GRAPH_FAILED);
    }

    const auto valuesInfo = context->GetIrInputInstanceInfo(VALUES_IDX); // values
    for (uint32_t i = 0; i < valuesInfo->GetInstanceNum(); i++) {
        auto valuesDtype = context->GetDynamicInputDataType(VALUES_IDX, i);
        OP_CHECK_IF(valuesDtype != DT_FLOAT, OP_LOGE(context->GetNodeName(),
                 "valuesDtype [%d] is not match float.", valuesDtype),
                 return ge::GRAPH_FAILED);
    }

    OP_LOGD(context->GetNodeName(), "End to do EmbeddingHashTableImportInferDataType.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(EmbeddingHashTableImport)
    .InferShape(Infershape4EmbeddingHashTableImport)
    .InferDataType(InferDataType4EmbeddingHashTableImport);

// -------------------EmbeddingHashTableImport Ops END---------------------
} // namespace ops