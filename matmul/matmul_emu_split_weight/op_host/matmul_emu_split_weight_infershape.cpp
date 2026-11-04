/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "error_util.h"
#include "matmul/common/op_host/matmul_error_report.h"
#include <string>

namespace {
const size_t kXIdx = 0;
const size_t kWHighIdx = 1;
const size_t kWLowIdx = 2;
const size_t kYIdx = 0;
const size_t kMinDimNum = 2;
const size_t kIdxM = 0;
const size_t kIdxN = 1;
const size_t kAttrWLowScaleIdx = 0;
const size_t kAttrTransXIdx = 1;
const size_t kAttrTransWIdx = 2;
} // namespace

namespace Ops::NN::MatMul {
ge::graphStatus InferShapeForMatmulEmuSplitWeight(gert::InferShapeContext* context)
{
    OP_CHECK_IF(
        context == nullptr,
        {
            Ops::NN::ReportMatmulInferShapeError("MatmulEmuSplitWeight",
                                                 "InferShapeContext is null; check the infer-shape invocation.");
        },
        return ge::GRAPH_FAILED);
    auto op_name = context->GetNodeName();
    auto xShape = context->GetInputShape(kXIdx);
    auto wHighShape = context->GetInputShape(kWHighIdx);
    auto wLowShape = context->GetInputShape(kWLowIdx);
    auto yShape = context->GetOutputShape(kYIdx);
    auto attrs = context->GetAttrs();

    OP_CHECK_IF(
        xShape == nullptr || wHighShape == nullptr || wLowShape == nullptr || yShape == nullptr || attrs == nullptr,
        {
            if (xShape == nullptr || wHighShape == nullptr || wLowShape == nullptr) {
                OP_LOGE_WITH_INVALID_INPUT(op_name,
                                           xShape == nullptr ? "x" : (wHighShape == nullptr ? "w_high" : "w_low"));
            } else {
                const char* reason = yShape == nullptr ? "Output y shape storage is null." :
                                                         "Infer-shape attributes are null.";
                Ops::NN::ReportMatmulInferShapeError(op_name, reason);
            }
        },
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(xShape->GetDimNum() != kMinDimNum || wHighShape->GetDimNum() != kMinDimNum ||
                    wLowShape->GetDimNum() != kMinDimNum,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    op_name, "x, w_high, w_low",
                    (std::to_string(xShape->GetDimNum()) + ", " + std::to_string(wHighShape->GetDimNum()) + ", " +
                     std::to_string(wLowShape->GetDimNum()))
                        .c_str(),
                    "All inputs must have rank 2."),
                return ge::GRAPH_FAILED);

    auto transXPtr = attrs->GetAttrPointer<bool>(kAttrTransXIdx);
    auto transWPtr = attrs->GetAttrPointer<bool>(kAttrTransWIdx);
    OP_CHECK_IF(transXPtr == nullptr || transWPtr == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(op_name, transXPtr == nullptr ? "transpose_x" : "transpose_w"),
                return ge::GRAPH_FAILED);
    bool transX = *transXPtr;
    bool transW = *transWPtr;

    int64_t M = transX ? xShape->GetDim(1) : xShape->GetDim(0);
    int64_t K = transX ? xShape->GetDim(0) : xShape->GetDim(1);
    int64_t wHighK = transW ? wHighShape->GetDim(1) : wHighShape->GetDim(0);
    int64_t N = transW ? wHighShape->GetDim(0) : wHighShape->GetDim(1);
    int64_t wLowK = transW ? wLowShape->GetDim(1) : wLowShape->GetDim(0);
    int64_t wLowN = transW ? wLowShape->GetDim(0) : wLowShape->GetDim(1);

    OP_CHECK_IF(
        K != wHighK,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            op_name, "x, w_high", (Ops::Base::ToString(*xShape) + ", " + Ops::Base::ToString(*wHighShape)).c_str(),
            "The contraction dimensions K of x and w_high must match after applying transpose_x and transpose_w."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        K != wLowK,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            op_name, "x, w_low", (Ops::Base::ToString(*xShape) + ", " + Ops::Base::ToString(*wLowShape)).c_str(),
            "The contraction dimensions K of x and w_low must match after applying transpose_x and transpose_w."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(N != wLowN,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    op_name, "w_high, w_low",
                    (Ops::Base::ToString(*wHighShape) + ", " + Ops::Base::ToString(*wLowShape)).c_str(),
                    "The output dimensions N of w_high and w_low must match after applying transpose_w."),
                return ge::GRAPH_FAILED);

    yShape->SetDimNum(kMinDimNum);
    yShape->SetDim(kIdxM, M);
    yShape->SetDim(kIdxN, N);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MatmulEmuSplitWeight).InferShape(InferShapeForMatmulEmuSplitWeight);
} // namespace Ops::NN::MatMul
