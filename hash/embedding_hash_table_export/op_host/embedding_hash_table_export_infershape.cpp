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
 * \file embedding_hash_table_export_infershape.cpp
 * \brief embedding hash table export
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {

static constexpr size_t INPUT_TABLE_HANDLE_IDX = 0;
static constexpr size_t INPUT_TABLE_SIZE_IDX = 1;
static constexpr size_t INPUT_EMBEDDING_DIMS_IDX = 2;
static constexpr size_t INPUT_BUCKET_SIZE_IDX = 3;

static constexpr size_t OUPUT_KEYS_IDX = 0;
static constexpr size_t OUPUT_COUNTERS_IDX = 1;
static constexpr size_t OUPUT_FILTER_FLAGS_IDX = 2;
static constexpr size_t OUPUT_VALUES_IDX = 3;
static constexpr size_t OUPUT_VALUES_DIM_NUM = 2;
static constexpr size_t DYNAMIC_OUTPUT_NUM = 4;

// ----------------EmbeddingHashTableExport InferShape Begin-------------------
graphStatus CheckEmbeddingHashTableExportParams(
    gert::InferShapeContext* context, int64_t numTable, int64_t numEmbeddingDim)
{
    OP_CHECK_IF(numTable < 0, OP_LOGE(context->GetNodeName(), "numTable must be equal or greater than 0, but it's %ld", numTable),
        return GRAPH_FAILED);

    OP_CHECK_IF(numEmbeddingDim != numTable,
        OP_LOGE(context->GetNodeName(), "numEmbeddingDim must be equal to numTable, ",
            "but numEmbeddingDim is %ld numTable is %ld", numEmbeddingDim, numTable),
        return GRAPH_FAILED);
    auto tableHandleShape = context->GetInputShape(INPUT_TABLE_HANDLE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, tableHandleShape);
    int64_t tableHandleNum = tableHandleShape->GetShapeSize();
    OP_CHECK_IF(tableHandleNum != numTable,
        OP_LOGE(context->GetNodeName(), "tableHandleNum must be equal to numTable, ",
            "but tableHandleNum is %ld numTable is %ld", tableHandleNum, numTable),
        return GRAPH_FAILED);
    auto bucketSizeShape = context->GetInputShape(INPUT_BUCKET_SIZE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, bucketSizeShape);
    int64_t bucketSizeNum = bucketSizeShape->GetShapeSize();
    OP_CHECK_IF(bucketSizeNum != numTable,
        OP_LOGE(context->GetNodeName(), "bucketSizeNum must be equal to numTable, ", "but bucketSizeNum is numTable is ", bucketSizeNum, numTable),
        return GRAPH_FAILED);
    int64_t totalOutNum = context->GetComputeNodeOutputNum();
    int64_t exceptOutNum = static_cast<int64_t>(numTable * DYNAMIC_OUTPUT_NUM);
    OP_CHECK_IF(totalOutNum != exceptOutNum,
        OP_LOGE(context->GetNodeName(), "Expected the number of total outputs is %ld but actual is %ld", exceptOutNum, totalOutNum),
        return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

graphStatus InferShape4EmbeddingHashTableExportNull(gert::InferShapeContext *context)
{
    OP_LOGD(context->GetNodeName(), "InferShape4EmbeddingHashTableExportNull start");
    auto embeddingDimTensor = context->GetInputTensor(INPUT_EMBEDDING_DIMS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, embeddingDimTensor);
    int64_t numTable = embeddingDimTensor->GetShapeSize();
    for (int64_t i = 0; i < numTable; i++) {
        gert::Shape *outKeyShape = context->GetOutputShape(OUPUT_KEYS_IDX * numTable + i);
        OP_CHECK_NULL_WITH_CONTEXT(context, outKeyShape);
        outKeyShape->SetDimNum(0);
        outKeyShape->AppendDim(-1);
        gert::Shape *outCounterShape = context->GetOutputShape(OUPUT_COUNTERS_IDX * numTable + i);
        OP_CHECK_NULL_WITH_CONTEXT(context, outCounterShape);
        outCounterShape->SetDimNum(0);
        outCounterShape->AppendDim(-1);
        gert::Shape *outFilterFlagShape = context->GetOutputShape(OUPUT_FILTER_FLAGS_IDX * numTable + i);
        OP_CHECK_NULL_WITH_CONTEXT(context, outFilterFlagShape);
        outFilterFlagShape->SetDimNum(0);
        outFilterFlagShape->AppendDim(-1);
        gert::Shape *outValueShape = context->GetOutputShape(OUPUT_VALUES_IDX * numTable + i);
        OP_CHECK_NULL_WITH_CONTEXT(context, outValueShape);
        outValueShape->SetDimNum(0);
        outValueShape->AppendDim(-1);
        outValueShape->AppendDim(-1);
    }

    OP_LOGD(context->GetNodeName(), "InferShape4EmbeddingHashTableExportNull end");
    return GRAPH_SUCCESS;
}

graphStatus InferShape4EmbeddingHashTableExport(gert::InferShapeContext* context) {
  auto tableSizeTensor = context->GetInputTensor(INPUT_TABLE_SIZE_IDX);
  OP_CHECK_NULL_WITH_CONTEXT(context, tableSizeTensor);
  const int64_t*  tableSizeValue = tableSizeTensor->GetData<int64_t>();
  if (tableSizeValue == nullptr){
        return InferShape4EmbeddingHashTableExportNull(context);
  }

  OP_CHECK_NULL_WITH_CONTEXT(context, tableSizeValue);
  int64_t numTable = tableSizeTensor->GetShapeSize();

  auto embeddingDimTensor = context->GetInputTensor(INPUT_EMBEDDING_DIMS_IDX);
  OP_CHECK_NULL_WITH_CONTEXT(context, embeddingDimTensor);
  const int64_t*  embeddingDimValue = embeddingDimTensor->GetData<int64_t>();
  OP_CHECK_NULL_WITH_CONTEXT(context, embeddingDimValue);
  int64_t numEmbeddingDim = embeddingDimTensor->GetShapeSize();
  OP_CHECK_IF(CheckEmbeddingHashTableExportParams(context, numTable, numEmbeddingDim) == GRAPH_FAILED,
           OP_LOGE(context->GetNodeName(), "check split params failed."),
           return GRAPH_FAILED);
  for (int64_t i = 0; i < numTable; i++) {
      gert::Shape*  outKeyShape = context->GetOutputShape(OUPUT_KEYS_IDX * numTable + i);
      OP_CHECK_NULL_WITH_CONTEXT(context, outKeyShape);
      outKeyShape->SetDimNum(0);
      outKeyShape->AppendDim(tableSizeValue[i]);
      gert::Shape*  outCounterShape = context->GetOutputShape(OUPUT_COUNTERS_IDX * numTable + i);
      OP_CHECK_NULL_WITH_CONTEXT(context, outCounterShape);
      outCounterShape->SetDimNum(0);
      outCounterShape->AppendDim(tableSizeValue[i]);
      gert::Shape*  outFilterFlagShape = context->GetOutputShape(OUPUT_FILTER_FLAGS_IDX * numTable + i);
      OP_CHECK_NULL_WITH_CONTEXT(context, outFilterFlagShape);
      outFilterFlagShape->SetDimNum(0);
      outFilterFlagShape->AppendDim(tableSizeValue[i]);
      gert::Shape*  outValueShape = context->GetOutputShape(OUPUT_VALUES_IDX * numTable + i);
      OP_CHECK_NULL_WITH_CONTEXT(context, outValueShape);
      outValueShape->SetDimNum(0);
      outValueShape->AppendDim(tableSizeValue[i]);
      outValueShape->AppendDim(embeddingDimValue[i]);
  }

  return GRAPH_SUCCESS;
}

graphStatus InferDataType4EmbeddingHashTableExport(gert::InferDataTypeContext *context) {
  OP_LOGD(context->GetNodeName(), "InferDataType4EmbeddingHashTableExport start");
  int64_t totalOutNum = context->GetComputeNodeOutputNum();;
  int64_t numTable = static_cast<int64_t>(totalOutNum / DYNAMIC_OUTPUT_NUM);
  for (int64_t i = 0; i < numTable; i++) {
      context->SetOutputDataType(OUPUT_KEYS_IDX * numTable + i, DT_INT64);     // keys
      context->SetOutputDataType(OUPUT_COUNTERS_IDX * numTable + i, DT_UINT64);    // counters
      context->SetOutputDataType(OUPUT_FILTER_FLAGS_IDX * numTable + i, DT_UINT8);     // filter_flags
      context->SetOutputDataType(OUPUT_VALUES_IDX * numTable + i, DT_FLOAT);     // values
  }
  OP_LOGD(context->GetNodeName(), "InferDataType4EmbeddingHashTableExport end");
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(EmbeddingHashTableExport).InferShape(InferShape4EmbeddingHashTableExport)
              .InputsDataDependency({INPUT_TABLE_SIZE_IDX, INPUT_EMBEDDING_DIMS_IDX})
              .InferDataType(InferDataType4EmbeddingHashTableExport);
// ----------------EmbeddingHashTableExport InferShape End----------------------

}  // namespace ops
