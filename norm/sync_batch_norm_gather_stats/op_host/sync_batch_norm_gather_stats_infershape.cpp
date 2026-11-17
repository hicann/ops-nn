/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sync_batch_norm_gather_stats_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "op_api/runtime2_util_nn.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {
static constexpr size_t SYNC_IDX_IN_SUM = 0;
static constexpr size_t SYNC_IDX_IN_SQUARE_SUM = 1;
static constexpr size_t SYNC_IDX_IN_COUNT = 2;
static constexpr size_t SYNC_IDX_IN_RN_MEAN = 3;
static constexpr size_t SYNC_IDX_IN_RN_VAR = 4;
static constexpr size_t SYNC_IDX_OUT_BT_MEAN = 0;
static constexpr size_t SYNC_IDX_OUT_BT_STD = 1;
static constexpr size_t SYNC_IDX_OUT_UP_MEAN = 2;
static constexpr size_t SYNC_IDX_OUT_UP_VAR = 3;
static constexpr size_t DIM_LEN = 1;
static constexpr size_t SYNC_DIM_LEN = 2;
static constexpr size_t N_IDX = 0;
static constexpr size_t C_IDX = 1;

static bool IsUnknownDim(int64_t dim) { return dim == ge::UNKNOWN_DIM; }

static bool DimsEqualOrUnknown(int64_t lhs, int64_t rhs)
{
    return lhs == rhs || IsUnknownDim(lhs) || IsUnknownDim(rhs);
}

static bool DimsMismatch(const gert::Shape* lhs, size_t lhs_idx, const gert::Shape* rhs, size_t rhs_idx)
{
    if (Ops::Base::IsUnknownRank(*lhs) || Ops::Base::IsUnknownRank(*rhs)) {
        return false;
    }
    if (lhs_idx >= lhs->GetDimNum() || rhs_idx >= rhs->GetDimNum()) {
        return false;
    }
    return !DimsEqualOrUnknown(lhs->GetDim(lhs_idx), rhs->GetDim(rhs_idx));
}

static int64_t KnownDimOf(const gert::Shape* shape, size_t idx)
{
    if (Ops::Base::IsUnknownRank(*shape) || idx >= shape->GetDimNum()) {
        return ge::UNKNOWN_DIM;
    }
    return shape->GetDim(idx);
}

static ge::graphStatus SyncBatchNormGatherStatsInferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do SyncBatchNormGatherStatsInferShape");

    // get input shapes
    const gert::Shape* sum_shape = context->GetInputShape(SYNC_IDX_IN_SUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, sum_shape);
    const gert::Shape* square_sum_shape = context->GetInputShape(SYNC_IDX_IN_SQUARE_SUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, square_sum_shape);
    const gert::Shape* count_shape = context->GetInputShape(SYNC_IDX_IN_COUNT);
    OP_CHECK_NULL_WITH_CONTEXT(context, count_shape);
    const gert::Shape* rn_mean_shape = context->GetInputShape(SYNC_IDX_IN_RN_MEAN);
    OP_CHECK_NULL_WITH_CONTEXT(context, rn_mean_shape);
    const gert::Shape* rn_var_shape = context->GetInputShape(SYNC_IDX_IN_RN_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context, rn_var_shape);

    // get output shapes
    gert::Shape* bt_mean_shape = context->GetOutputShape(SYNC_IDX_OUT_BT_MEAN);
    OP_CHECK_NULL_WITH_CONTEXT(context, bt_mean_shape);
    gert::Shape* bt_std_shape = context->GetOutputShape(SYNC_IDX_OUT_BT_STD);
    OP_CHECK_NULL_WITH_CONTEXT(context, bt_std_shape);
    gert::Shape* up_mean_shape = context->GetOutputShape(SYNC_IDX_OUT_UP_MEAN);
    OP_CHECK_NULL_WITH_CONTEXT(context, up_mean_shape);
    gert::Shape* up_var_shape = context->GetOutputShape(SYNC_IDX_OUT_UP_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context, up_var_shape);

    // check shape
    const size_t sum_size = sum_shape->GetDimNum();
    const size_t square_sum_size = square_sum_shape->GetDimNum();
    const size_t count_size = count_shape->GetDimNum();
    const size_t rn_mean_size = rn_mean_shape->GetDimNum();
    const size_t rn_var_size = rn_var_shape->GetDimNum();

    OP_CHECK_IF((!Ops::Base::IsUnknownRank(*sum_shape) && sum_size != SYNC_DIM_LEN) ||
                    (!Ops::Base::IsUnknownRank(*square_sum_shape) && square_sum_size != SYNC_DIM_LEN),
                OP_LOGE(context->GetNodeName(), "sum_size and square_sum_size should be equal to 2."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF((!Ops::Base::IsUnknownRank(*count_shape) && count_size != DIM_LEN) ||
                    (!Ops::Base::IsUnknownRank(*rn_mean_shape) && rn_mean_size != DIM_LEN) ||
                    (!Ops::Base::IsUnknownRank(*rn_var_shape) && rn_var_size != DIM_LEN),
                OP_LOGE(context->GetNodeName(), "count_size, rn_mean_size and rn_var_size should be equal to 1."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(DimsMismatch(sum_shape, N_IDX, square_sum_shape, N_IDX) ||
                    DimsMismatch(count_shape, N_IDX, sum_shape, N_IDX) ||
                    DimsMismatch(count_shape, N_IDX, square_sum_shape, N_IDX),
                OP_LOGE(context->GetNodeName(), "N must be same."), return GRAPH_FAILED);

    OP_CHECK_IF(DimsMismatch(sum_shape, C_IDX, square_sum_shape, C_IDX) ||
                    DimsMismatch(rn_mean_shape, N_IDX, sum_shape, C_IDX) ||
                    DimsMismatch(rn_mean_shape, N_IDX, square_sum_shape, C_IDX) ||
                    DimsMismatch(rn_var_shape, N_IDX, sum_shape, C_IDX) ||
                    DimsMismatch(rn_var_shape, N_IDX, square_sum_shape, C_IDX) ||
                    DimsMismatch(rn_mean_shape, N_IDX, rn_var_shape, N_IDX),
                OP_LOGE(context->GetNodeName(), "C must be same."), return GRAPH_FAILED);

    int64_t channel = KnownDimOf(sum_shape, C_IDX);
    if (IsUnknownDim(channel)) {
        channel = KnownDimOf(square_sum_shape, C_IDX);
    }
    if (IsUnknownDim(channel)) {
        channel = KnownDimOf(rn_mean_shape, N_IDX);
    }
    if (IsUnknownDim(channel)) {
        channel = KnownDimOf(rn_var_shape, N_IDX);
    }

    // outputs are 1-D with length equal to the channel axis of the statistics
    bt_mean_shape->SetDimNum(DIM_LEN);
    bt_std_shape->SetDimNum(DIM_LEN);
    up_mean_shape->SetDimNum(DIM_LEN);
    up_var_shape->SetDimNum(DIM_LEN);
    bt_mean_shape->SetDim(0, channel);
    bt_std_shape->SetDim(0, channel);
    up_mean_shape->SetDim(0, channel);
    up_var_shape->SetDim(0, channel);

    OP_LOGD(context->GetNodeName(), "End to do SyncBatchNormGatherStatsInferShape");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SyncBatchNormGatherStats).InferShape(SyncBatchNormGatherStatsInferShape);
} // namespace ops
