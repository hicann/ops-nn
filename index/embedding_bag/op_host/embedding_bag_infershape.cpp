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
 * \file embedding_bag.cc
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

namespace {
const char* ATTR_MODE_MAX = "max";
constexpr int64_t WEIGHT_IDX = 0;
constexpr int64_t INDICES_IDX = 1;
constexpr int64_t OFFSETS_IDX = 2;
constexpr int64_t Y_IDX = 0;
constexpr int64_t OFFSET_2_BAG_IDX = 1;
constexpr int64_t BAG_SIZE_IDX = 2;
constexpr int64_t BAG_SIZE_DIM = 1;
constexpr int64_t MAX_INDICES_IDX = 3;
constexpr int64_t MAX_INDICES_DIM = 1;
constexpr int64_t MAX_INDICES_ZERO_DIM = 0;
constexpr int64_t MAX_INDICES_DIM_TWO = 2;
constexpr int64_t MAX_INDICES_ONE_DIM = 1;
constexpr int64_t INDICES_ONE_DIM = 1;
constexpr int64_t EMBEDDING_DIM_IDX = 1;
constexpr int64_t OFFSETS_LEN_IDX = 0;
constexpr int64_t INCLUDE_LAST_WEIGHT_IDX = 3;
constexpr int64_t INDICES_TWO_DIM = 2;
constexpr int64_t OUTPUT_DIMS = 2;
constexpr int64_t INDICES_ZERO_DIM = 0;
constexpr int64_t BATCH_DIM = 0;
constexpr int64_t MODE_IDX = 0;
constexpr int64_t BAG_SIZE_ZERO_DIM = 0;
constexpr int64_t MINUS_ONE = -1;

} // namespace
using namespace ge;
namespace ops {
static int64_t get_batch(const bool& include_last_offset, int64_t offsets_lens)
{
    if (offsets_lens == MINUS_ONE) {
        return MINUS_ONE;
    }
    int64_t output_dim = 0;
    if (include_last_offset) {
        output_dim = offsets_lens - 1;
    } else {
        output_dim = offsets_lens;
    }
    return output_dim;
}

inline ge::graphStatus InferShape4Output(
    gert::InferShapeContext* context, int64_t batch, int64_t embedding_dim, bool is_unknown_rank)
{
    auto output_shape = context->GetOutputShape(Y_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape);
    if (is_unknown_rank) {
        Ops::Base::SetUnknownRank(*output_shape);
        return ge::GRAPH_SUCCESS;
    }
    output_shape->SetDimNum(OUTPUT_DIMS);
    output_shape->SetDim(BATCH_DIM, batch);
    output_shape->SetDim(EMBEDDING_DIM_IDX, embedding_dim);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus InferShape4Offset2Bag(
    gert::InferShapeContext* context, int64_t indices_num, bool is_unknown_rank)
{
    auto offset_2_bag_shape = context->GetOutputShape(OFFSET_2_BAG_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, offset_2_bag_shape);
    if (is_unknown_rank) {
        Ops::Base::SetUnknownRank(*offset_2_bag_shape);
        return ge::GRAPH_SUCCESS;
    }
    offset_2_bag_shape->SetDimNum(INDICES_ONE_DIM);
    offset_2_bag_shape->SetDim(INDICES_ZERO_DIM, indices_num);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus InferShape4BagSize(gert::InferShapeContext* context, int64_t offsets_lens, bool is_unknown_rank)
{
    auto bag_size_shape = context->GetOutputShape(BAG_SIZE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, bag_size_shape);
    if (is_unknown_rank) {
        Ops::Base::SetUnknownRank(*bag_size_shape);
        return ge::GRAPH_SUCCESS;
    }
    bag_size_shape->SetDimNum(BAG_SIZE_DIM);
    bag_size_shape->SetDim(BAG_SIZE_ZERO_DIM, offsets_lens);
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus InferShape4MaxIndices(
    gert::InferShapeContext* context, int64_t batch, int64_t embedding_dim, bool is_unknown_rank)
{
    auto max_indices_shape = context->GetOutputShape(MAX_INDICES_IDX);
    if (max_indices_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (is_unknown_rank) {
        Ops::Base::SetUnknownRank(*max_indices_shape);
        return ge::GRAPH_SUCCESS;
    }

    auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const char* mode = attrs->GetAttrPointer<char>(MODE_IDX);
    if (mode == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (strcmp(mode, ATTR_MODE_MAX) == 0) {
        max_indices_shape->SetDimNum(MAX_INDICES_DIM_TWO);
        max_indices_shape->SetDim(MAX_INDICES_ZERO_DIM, batch);
        max_indices_shape->SetDim(MAX_INDICES_ONE_DIM, embedding_dim);
    } else {
        max_indices_shape->SetDimNum(MAX_INDICES_DIM);
        max_indices_shape->SetDim(MAX_INDICES_ZERO_DIM, batch);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForEmbeddingBag(gert::InferShapeContext* context)
{
    auto const weight_shape = context->GetInputShape(WEIGHT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, weight_shape);
    auto const indices_shape = context->GetInputShape(INDICES_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, indices_shape);
    auto const offsets_shape = context->GetInputShape(OFFSETS_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, offsets_shape);

    auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* include_last_offset = attrs->GetAttrPointer<bool>(INCLUDE_LAST_WEIGHT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, include_last_offset);
    const char* mode = attrs->GetAttrPointer<char>(MODE_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, mode);

    int64_t embedding_dim = weight_shape->GetDim(EMBEDDING_DIM_IDX);
    int64_t indices_num = indices_shape->GetDim(INDICES_ZERO_DIM);
    int64_t offsets_lens = offsets_shape->GetDim(OFFSETS_LEN_IDX);
    int64_t batch = get_batch(*include_last_offset, offsets_lens);
    bool is_unknown_rank = Ops::Base::IsUnknownRank(*weight_shape);

    OP_CHECK_IF(
        InferShape4Output(context, batch, embedding_dim, is_unknown_rank) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "failed to infer shape for output."), return GRAPH_FAILED);
    OP_CHECK_IF(
        InferShape4Offset2Bag(context, indices_num, is_unknown_rank) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "failed to infer shape for offset2bag."), return GRAPH_FAILED);
    OP_CHECK_IF(
        InferShape4BagSize(context, batch, is_unknown_rank) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "failed to infer shape for bag_size."), return GRAPH_FAILED);
    OP_CHECK_IF(
        InferShape4MaxIndices(context, batch, embedding_dim, is_unknown_rank) != ge::GRAPH_SUCCESS,
        OP_LOGE(context, "failed to infer shape for max_indices."), return GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

graphStatus InferDtypeForEmbeddingBag(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDtypeForEmbeddingBag");

    auto y_dtype = context->GetInputDataType(WEIGHT_IDX);
    auto indices_dtype = context->GetInputDataType(INDICES_IDX);
    context->SetOutputDataType(Y_IDX, y_dtype);
    context->SetOutputDataType(OFFSET_2_BAG_IDX, indices_dtype);
    context->SetOutputDataType(BAG_SIZE_IDX, indices_dtype);
    context->SetOutputDataType(MAX_INDICES_IDX, indices_dtype);

    OP_LOGD(context, "End to do InferDtypeForEmbeddingBag");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(EmbeddingBag).InferShape(InferShapeForEmbeddingBag).InferDataType(InferDtypeForEmbeddingBag);
} // namespace ops