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
 * \file gn_training_reduce_infershape.cpp
 * \brief GNTrainingReduce shape inference implementation.
 */

// op_impl_registry.h: provides IMPL_OP_INFERSHAPE macro, gert::InferShapeContext.
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "op_common/log/log.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "graph/types.h"

using namespace ge;
using namespace Ops::Base;

namespace ops {

namespace {

constexpr size_t kRank = 4;
constexpr size_t kOutputRank = 5;
constexpr size_t kOutputCount = 2;
constexpr size_t kChannelAxisNchw = 1;
constexpr size_t kChannelAxisNhwc = 3;
constexpr int64_t kDefaultNumGroups = 2;

bool IsUnknownRank(const gert::Shape* shape)
{
    return shape != nullptr && shape->GetDimNum() == 1 && shape->GetDim(0) == ge::UNKNOWN_DIM_NUM;
}

} // namespace

/**
 * InferShapeGNTrainingReduce: GE shape inference callback.
 *
 * Parameters:
 *   context — [in/out] provides the input shape and accepts the output shapes.
 *
 * Returns:
 *   GRAPH_SUCCESS for a rank-4 NCHW/NHWC input with a valid num_groups; GRAPH_FAILED
 *   for invalid contexts, wrong rank, unsupported format, or a num_groups that does
 *   not divide the (known) channel dim.
 *
 * Design:
 *   Input 0 = x, outputs 0/1 = sum / square_sum.  Unknown rank propagates.  For a
 *   known rank, the channel axis follows the origin format and the output is the
 *   rank-5 keepdims reduction of the grouped view.  C % num_groups == 0 is enforced
 *   only when C is known (dynamic shapes leave C = UNKNOWN_DIM).
 */
static ge::graphStatus InferShapeGNTrainingReduce(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return GRAPH_FAILED;
    }
    const gert::Shape* xShape = context->GetInputShape(0);
    gert::Shape* sumShape = context->GetOutputShape(0);
    gert::Shape* squareSumShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, sumShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, squareSumShape);

    // Unknown rank propagates to both outputs unchanged.
    if (IsUnknownRank(xShape)) {
        *sumShape = *xShape;
        *squareSumShape = *xShape;
        return GRAPH_SUCCESS;
    }

    if (xShape->GetDimNum() != kRank) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x",
                                                 std::to_string(xShape->GetDimNum()).c_str(),
                                                 "The shape dim of input x must be 4");
        return GRAPH_FAILED;
    }

    // Channel axis depends on the input layout (NCHW -> 1, NHWC -> 3).
    const gert::CompileTimeTensorDesc* xDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const ge::Format format = xDesc->GetOriginFormat();
    size_t channelAxis = 0;
    if (format == ge::FORMAT_NCHW) {
        channelAxis = kChannelAxisNchw;
    } else if (format == ge::FORMAT_NHWC) {
        channelAxis = kChannelAxisNhwc;
    } else {
        OP_LOGE_FOR_INVALID_FORMAT(context->GetNodeName(), "x", ToString(format).c_str(), "NCHW or NHWC");
        return GRAPH_FAILED;
    }

    // Both outputs are always ND: sum / square_sum 仅 ND，其它 format 不支持.  The host tiling enforces the same
    // contract, but graph-mode compilation routes through InferShape, so reject a
    // caller-declared non-ND output format here as well.
    for (size_t i = 0; i < kOutputCount; ++i) {
        const gert::CompileTimeTensorDesc* outDesc = context->GetOutputDesc(i);
        if (outDesc == nullptr || outDesc->GetOriginFormat() != ge::FORMAT_ND) {
            const std::string outName = "output[" + std::to_string(i) + "]";
            const std::string actualFormat = (outDesc == nullptr) ? "nullptr" : ToString(outDesc->GetOriginFormat());
            OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(context->GetNodeName(), outName.c_str(), actualFormat.c_str(),
                                                   "The origin format of output must be ND");
            return GRAPH_FAILED;
        }
    }

    // num_groups is optional with default 2; GE fills the default into the attrs.
    int64_t numGroups = kDefaultNumGroups;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    if (attrs != nullptr && attrs->GetAttrPointer<int64_t>(0) != nullptr) {
        numGroups = *attrs->GetAttrPointer<int64_t>(0);
    }
    if (numGroups < 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "num_groups", std::to_string(numGroups).c_str(),
                                              "The attribute num_groups must be greater than or equal to 1");
        return GRAPH_FAILED;
    }

    const int64_t channel = xShape->GetDim(channelAxis);
    if (channel != ge::UNKNOWN_DIM && (channel % numGroups) != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "num_groups", std::to_string(numGroups).c_str(),
            ("The channel dim C=" + std::to_string(channel) + " of input x must be divisible by num_groups").c_str());
        return GRAPH_FAILED;
    }

    const int64_t n = xShape->GetDim(0);
    const int64_t one = 1;
    sumShape->SetDimNum(kOutputRank);
    if (format == ge::FORMAT_NCHW) {
        sumShape->SetDim(0, n);         // N
        sumShape->SetDim(1, numGroups); // G
        sumShape->SetDim(2, one);
        sumShape->SetDim(3, one);
        sumShape->SetDim(4, one);
    } else {
        sumShape->SetDim(0, n); // N
        sumShape->SetDim(1, one);
        sumShape->SetDim(2, one);
        sumShape->SetDim(3, numGroups); // G
        sumShape->SetDim(4, one);
    }
    squareSumShape->SetDimNum(kOutputRank);
    *squareSumShape = *sumShape;
    return GRAPH_SUCCESS;
}

/**
 * IMPL_OP_INFERSHAPE(GNTrainingReduce).InferShape(InferShapeGNTrainingReduce):
 *   Registers the shape inference function at static init time.  When the
 *   framework needs to determine the output shapes of a GNTrainingReduce node, it
 *   calls InferShapeGNTrainingReduce.
 */
IMPL_OP_INFERSHAPE(GNTrainingReduce).InferShape(InferShapeGNTrainingReduce);

} // namespace ops
