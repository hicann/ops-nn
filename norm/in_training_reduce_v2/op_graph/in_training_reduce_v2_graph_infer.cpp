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
 * \file in_training_reduce_v2_graph_infer.cpp
 * \brief INTrainingReduceV2 graph infer registration.
 */

#include <cstddef>
#include <cstdint>
#include <vector>

#include "../in_training_reduce_v2_infer_common.h"
#include "graph/operator_reg.h"
#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ge {
IMPLEMT_COMMON_INFERFUNC(INTrainingReduceV2LegacyV1InferShape)
{
    const TensorDesc xDesc = op.GetInputDescByName("x");
    const std::vector<int64_t> xDims = xDesc.GetShape().GetDims();
    const Format originFormat = xDesc.GetOriginFormat();
    const bool isChannelLast = (originFormat == FORMAT_NHWC || originFormat == FORMAT_NDHWC);
    if (xDims != UNKNOWN_RANK && !ops::IsINTrainingReduceV2RankValid(originFormat, xDims.size())) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            "INTrainingReduceV2", "x", std::to_string(xDims.size()).c_str(),
            "NCHW/NHWC require rank 4, NCDHW/NDHWC require rank 5, and ND requires rank in [2, 8]");
        return GRAPH_FAILED;
    }

    std::vector<int64_t> outputDims(xDims.size(), 1);
    const size_t channelIndex = (isChannelLast && !xDims.empty()) ? (xDims.size() - 1U) : 1U;
    for (size_t i = 0; i < xDims.size(); ++i) {
        if (i == 0U || i == channelIndex) {
            outputDims[i] = xDims[i];
        }
    }

    TensorDesc sumDesc = op.GetOutputDescByName("sum");
    TensorDesc squareSumDesc = op.GetOutputDescByName("square_sum");
    sumDesc.SetShape(Shape(outputDims));
    squareSumDesc.SetShape(Shape(outputDims));
    sumDesc.SetOriginShape(Shape(outputDims));
    squareSumDesc.SetOriginShape(Shape(outputDims));
    sumDesc.SetDataType(DT_FLOAT);
    squareSumDesc.SetDataType(DT_FLOAT);
    if (op.UpdateOutputDesc("sum", sumDesc) != GRAPH_SUCCESS ||
        op.UpdateOutputDesc("square_sum", squareSumDesc) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

// Static GEIR graphs still invoke the legacy V1 registry for this built-in op.
// The custom package is loaded before libopsproto.so, so register the corrected
// ND channel-second contract here as well as in the runtime2.0 registry below.
COMMON_INFER_FUNC_REG(INTrainingReduceV2, INTrainingReduceV2LegacyV1InferShape);
} // namespace ge

namespace ops {
ge::graphStatus InferShapeForINTrainingReduceV2(gert::InferShapeContext* context)
{
    return InferShape4INTrainingReduceV2(context);
}

ge::graphStatus InferDataType4INTrainingReduceV2(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataType4INTrainingReduceV2");
    if (context->SetOutputDataType(0, ge::DT_FLOAT) != ge::GRAPH_SUCCESS ||
        context->SetOutputDataType(1, ge::DT_FLOAT) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "End to do InferDataType4INTrainingReduceV2");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(INTrainingReduceV2).InferShape(InferShapeForINTrainingReduceV2).InferDataType(InferDataType4INTrainingReduceV2);
} // namespace ops
