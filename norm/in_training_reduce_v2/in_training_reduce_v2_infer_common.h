/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IN_TRAINING_REDUCE_V2_INFER_COMMON_H
#define IN_TRAINING_REDUCE_V2_INFER_COMMON_H

#include <cstddef>
#include <cstdint>
#include <string>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "util/shape_util.h"

namespace ops {
inline bool IsINTrainingReduceV2RankValid(ge::Format format, size_t rank)
{
    if (format == ge::FORMAT_NCHW || format == ge::FORMAT_NHWC) {
        return rank == 4U;
    }
    if (format == ge::FORMAT_NCDHW || format == ge::FORMAT_NDHWC) {
        return rank == 5U;
    }
    return format == ge::FORMAT_ND && rank >= 2U && rank <= 8U;
}

inline ge::graphStatus InferShape4INTrainingReduceV2(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape4INTrainingReduceV2");

    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    gert::Shape* sumShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, sumShape);
    gert::Shape* squareSumShape = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, squareSumShape);

    if (Ops::Base::IsUnknownRank(*xShape)) {
        Ops::Base::SetUnknownRank(*sumShape);
        Ops::Base::SetUnknownRank(*squareSumShape);
        return ge::GRAPH_SUCCESS;
    }

    const auto* xDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const ge::Format originFormat = xDesc->GetOriginFormat();
    const bool isChannelLast = (originFormat == ge::FORMAT_NHWC || originFormat == ge::FORMAT_NDHWC);

    const size_t rank = xShape->GetDimNum();
    OP_CHECK_IF(!IsINTrainingReduceV2RankValid(originFormat, rank),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x", std::to_string(rank).c_str(),
                                                         "NCHW/NHWC require rank 4, NCDHW/NDHWC require rank 5, and "
                                                         "ND requires rank in [2, 8]"),
                return ge::GRAPH_FAILED);
    const size_t channelIndex = (isChannelLast && rank > 0U) ? (rank - 1U) : 1U;
    sumShape->SetDimNum(rank);
    squareSumShape->SetDimNum(rank);
    for (size_t i = 0; i < rank; ++i) {
        const int64_t dim = (i == 0U || i == channelIndex) ? xShape->GetDim(i) : 1;
        sumShape->SetDim(i, dim);
        squareSumShape->SetDim(i, dim);
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShape4INTrainingReduceV2");
    return ge::GRAPH_SUCCESS;
}
} // namespace ops

#endif // IN_TRAINING_REDUCE_V2_INFER_COMMON_H
