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
 * \file embedding_hash_table_lookup_or_insert.cc
 * \brief embedding_hash_table_lookup_or_insert
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace {
constexpr size_t EMBEDDING_HASHTABLE_INPUT_KEY_IDX = 1;
constexpr size_t EMBEDDING_HASHTABLE_ATTR_EMBEDDING_DIM_IDX = 1;
static constexpr int64_t INPUT_NODE_NUM = 2;
static constexpr int64_t TABLE_HANDLES_IDX = 0;
static constexpr int64_t KEYS_IDX = 1;
static constexpr int64_t VALUES_IDX = 0;
}  // namespace

namespace ops {
ge::graphStatus InferShapeForLookupOrInsert(gert::InferShapeContext* context) {
  auto key_shape = context->GetInputShape(EMBEDDING_HASHTABLE_INPUT_KEY_IDX);
  OP_CHECK_NULL_WITH_CONTEXT(context, key_shape);
  auto out_shape = context->GetOutputShape(0);
  OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);

  int64_t output_shape_len = 2;
  int64_t input_shape_len = key_shape->GetDimNum();
  int64_t key_size = 1;
  for (int64_t i = 0; i < input_shape_len; ++i) {
    key_size *= key_shape->GetDim(i);
  }
  out_shape->SetDimNum(output_shape_len);
  out_shape->SetDim(0, key_size);

  auto attrs = context->GetAttrs();
  OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
  auto embedding_dims = attrs->GetAttrPointer<int64_t>(EMBEDDING_HASHTABLE_ATTR_EMBEDDING_DIM_IDX);
  OP_CHECK_NULL_WITH_CONTEXT(context, embedding_dims);
  out_shape->SetDim(1, *embedding_dims);
  return GRAPH_SUCCESS;
}

graphStatus InferDtypeForLookupOrInsert (gert::InferDataTypeContext* context) {
  OP_LOGD(context->GetNodeName(), "Begin to do InferDtypeForLookupOrInsert rt2.0");
  int64_t totalInNum = context->GetComputeNodeInputNum();
    if (!(totalInNum == INPUT_NODE_NUM)) {
        return GRAPH_FAILED;
    }
    auto tableHandleDtype = context->GetInputDataType(TABLE_HANDLES_IDX);    // table_handle
    OP_CHECK_IF(tableHandleDtype != DT_INT64, OP_LOGE(context->GetNodeName(),
            "tableHandleDtype [%d] is not match int64.", tableHandleDtype),
             return ge::GRAPH_FAILED);

    auto keysDtype = context->GetInputDataType(KEYS_IDX);                    // keys
    OP_CHECK_IF(keysDtype != DT_INT64, OP_LOGE(context->GetNodeName(),
            "keysDtype [%d] is not match int64.", keysDtype),
             return ge::GRAPH_FAILED);

    auto valuesDtype = context->GetOutputDataType(VALUES_IDX);                // values
    OP_CHECK_IF(valuesDtype != DT_FLOAT, OP_LOGE(context->GetNodeName(),
            "valuesDtype [%d] is not match float.", valuesDtype),
             return ge::GRAPH_FAILED);
  OP_LOGD(context->GetNodeName(), "End to do InferDtypeForLookupOrInsert");
  return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(EmbeddingHashTableLookupOrInsert).InferShape(InferShapeForLookupOrInsert)
                              .InferDataType(InferDtypeForLookupOrInsert);
}  // namespace ops