/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "es_c_graph_builder.h"
#include "compliant_node_builder.h"
#include "utils/extern_math_util.h"
#include "es_log.h"

#ifdef __cplusplus
extern "C" {
#endif

static inline EsCTensorHolder *EsIndexPutV3(
    EsCTensorHolder *selfRef,
    EsCTensorHolder *values,
    EsCTensorHolder *masks,
    EsCTensorHolder **indices,
    int64_t indices_num,
    bool accumulate,
    bool deterministic) {
  ES_ASSERT_NOTNULL(selfRef);
  ES_ASSERT_NOTNULL(values);
  ES_ASSERT_NOTNULL(masks);
  ES_ASSERT_NOTNULL(indices);

  auto &builder = selfRef->GetOwnerBuilder();
  auto ge_graph = builder.GetGraph();

  auto node = ge::es::CompliantNodeBuilder(ge_graph)
      .OpType("IndexPutV3")
      .Name(builder.GenerateNodeName("IndexPutV3").GetString())
      .IrDefInputs({
          {"selfRef", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
          {"values", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
          {"masks", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
          {"indices", ge::es::CompliantNodeBuilder::kEsIrInputDynamic, ""},
      })
      .InstanceDynamicInputNum("indices", static_cast<int32_t>(indices_num))
      .IrDefOutputs({
          {"selfRef", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
      })
      .IrDefAttrs({
          {"accumulate", ge::es::CompliantNodeBuilder::kEsAttrOptional, "Bool", ge::es::CreateFrom(accumulate)},
          {"deterministic", ge::es::CompliantNodeBuilder::kEsAttrOptional, "Bool", ge::es::CreateFrom(deterministic)},
      })
      .Build();

  ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, selfRef->GetProducer(), selfRef->GetOutIndex(), node, 0));
  ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, values->GetProducer(), values->GetOutIndex(), node, 1));
  ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, masks->GetProducer(), masks->GetOutIndex(), node, 2));

  for (int64_t i = 0; i < indices_num; ++i) {
    ES_ASSERT_GRAPH_SUCCESS(
        ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, indices[i]->GetProducer(), indices[i]->GetOutIndex(), node, static_cast<int32_t>(3 + i)));
  }

  return builder.GetTensorHolderFromNode(std::move(node), 0);
}

static inline EsCTensorHolder *EsIndexPutWithSortV2(
    EsCTensorHolder *self,
    EsCTensorHolder *linear_index,
    EsCTensorHolder *pos_idx,
    EsCTensorHolder *values,
    const int64_t *indexed_sizes,
    int64_t indexed_sizes_num,
    bool accumulate) {
  // Assert inputs are not null
  ES_ASSERT_NOTNULL(self);
  ES_ASSERT_NOTNULL(linear_index);
  ES_ASSERT_NOTNULL(pos_idx);
  ES_ASSERT_NOTNULL(values);

  // 1. Get GraphBuilder from input tensor
  auto &builder = self->GetOwnerBuilder();
  auto ge_graph = builder.GetGraph();

  // 2. Build compliant node with IR definition
  auto node = ge::es::CompliantNodeBuilder(ge_graph)
      .OpType("IndexPutWithSortV2")
      .Name(builder.GenerateNodeName("IndexPutWithSortV2").GetString())
      .IrDefInputs({
          {"self", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
          {"linear_index", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
          {"pos_idx", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
          {"values", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
      })
      .IrDefOutputs({
          {"self", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
      })
      .IrDefAttrs({
          {
              "indexed_sizes",
              ge::es::CompliantNodeBuilder::kEsAttrRequired,
              "ListInt",
              ge::es::CreateFrom(std::vector<int64_t>(indexed_sizes, indexed_sizes + indexed_sizes_num))
          },
          {
              "accumulate",
              ge::es::CompliantNodeBuilder::kEsAttrOptional,
              "Bool",
              ge::es::CreateFrom(accumulate)
          },
      })
      .Build();

  // 3. Connect inputs to node
  ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, self->GetProducer(), self->GetOutIndex(), node, 0));
  ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, linear_index->GetProducer(), linear_index->GetOutIndex(), node, 1));
  ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, pos_idx->GetProducer(), pos_idx->GetOutIndex(), node, 2));
  ES_ASSERT_GRAPH_SUCCESS(ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, values->GetProducer(), values->GetOutIndex(), node, 3));

  // 4. Return TensorHolder from node output
  return builder.GetTensorHolderFromNode(std::move(node), 0);
}

static inline EsCTensorHolder *EsLinearIndexV2(
    EsCTensorHolder **indices_list,
    int64_t indices_list_num,
    EsCTensorHolder *stride,
    EsCTensorHolder *value_size) {
  // Assert inputs are valid
  ES_ASSERT_TRUE(ge::IntegerChecker<int32_t>::Compat(indices_list_num));
  ES_ASSERT_NOTNULL(stride);
  ES_ASSERT_NOTNULL(value_size);

  // Find a non-null input to get builder
  EsCTensorHolder *non_null_in = nullptr;
  ES_ASSERT_NOTNULL(indices_list);
  for (int64_t i = 0; i < indices_list_num; ++i) {
    if (indices_list[i] != nullptr) {
      non_null_in = indices_list[i];
      break;
    }
  }
  
  // If indices_list is empty, use stride to get builder
  if (non_null_in == nullptr) {
    non_null_in = stride;
  }
  ES_ASSERT_NOTNULL(non_null_in);

  // 1. Get GraphBuilder from input tensor
  auto &builder = non_null_in->GetOwnerBuilder();
  auto ge_graph = builder.GetGraph();

  // 2. Build compliant node with IR definition
  auto node = ge::es::CompliantNodeBuilder(ge_graph)
      .OpType("LinearIndexV2")
      .Name(builder.GenerateNodeName("LinearIndexV2").GetString())
      .IrDefInputs({
          {"indices_list", ge::es::CompliantNodeBuilder::kEsIrInputDynamic, ""},
          {"stride", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
          {"value_size", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
      })
      .IrDefOutputs({
          {"index", ge::es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
      })
      .InstanceDynamicInputNum("indices_list", static_cast<int32_t>(indices_list_num))
      .Build();

  // 3. Connect dynamic inputs (indices_list)
  for (int64_t i = 0; i < indices_list_num; ++i) {
    auto one_indices = indices_list[i];
    if (one_indices != nullptr) {
      ES_ASSERT_GRAPH_SUCCESS(
          ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, one_indices->GetProducer(), one_indices->GetOutIndex(), node, static_cast<int32_t>(i)));
    }
  }

  // Connect stride and value_size after indices_list
  ES_ASSERT_GRAPH_SUCCESS(
      ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, stride->GetProducer(), stride->GetOutIndex(), node, static_cast<int32_t>(indices_list_num)));
  ES_ASSERT_GRAPH_SUCCESS(
      ge::es::AddEdgeAndUpdatePeerDesc(*ge_graph, value_size->GetProducer(), value_size->GetOutIndex(), node, static_cast<int32_t>(indices_list_num + 1)));

  // 4. Return TensorHolder from node output
  return builder.GetTensorHolderFromNode(std::move(node), 0);
}

#ifdef __cplusplus
}
#endif