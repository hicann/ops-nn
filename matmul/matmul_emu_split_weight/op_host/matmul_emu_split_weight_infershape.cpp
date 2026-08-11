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
    OP_CHECK_IF(context == nullptr, CUBE_INNER_ERR_REPORT("MatmulEmuSplitWeight", "context is null"),
                return ge::GRAPH_FAILED);
    auto op_name = context->GetNodeName();
    auto xShape = context->GetInputShape(kXIdx);
    auto wHighShape = context->GetInputShape(kWHighIdx);
    auto wLowShape = context->GetInputShape(kWLowIdx);
    auto yShape = context->GetOutputShape(kYIdx);
    auto attrs = context->GetAttrs();

    OP_CHECK_IF(
        xShape == nullptr || wHighShape == nullptr || wLowShape == nullptr || yShape == nullptr || attrs == nullptr,
        CUBE_INNER_ERR_REPORT(op_name, "input or output shape or attrs is null"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(xShape->GetDimNum() != kMinDimNum || wHighShape->GetDimNum() != kMinDimNum ||
                    wLowShape->GetDimNum() != kMinDimNum,
                CUBE_INNER_ERR_REPORT(op_name, "all inputs must be 2D"), return ge::GRAPH_FAILED);

    bool transX = *(attrs->GetAttrPointer<bool>(kAttrTransXIdx));
    bool transW = *(attrs->GetAttrPointer<bool>(kAttrTransWIdx));

    int64_t M = transX ? xShape->GetDim(1) : xShape->GetDim(0);
    int64_t K = transX ? xShape->GetDim(0) : xShape->GetDim(1);
    int64_t wHighK = transW ? wHighShape->GetDim(1) : wHighShape->GetDim(0);
    int64_t N = transW ? wHighShape->GetDim(0) : wHighShape->GetDim(1);
    int64_t wLowK = transW ? wLowShape->GetDim(1) : wLowShape->GetDim(0);
    int64_t wLowN = transW ? wLowShape->GetDim(0) : wLowShape->GetDim(1);

    OP_CHECK_IF(K != wHighK, CUBE_INNER_ERR_REPORT(op_name, "x K(%ld) must match w_high K(%ld)", K, wHighK),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(K != wLowK, CUBE_INNER_ERR_REPORT(op_name, "x K(%ld) must match w_low K(%ld)", K, wLowK),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(N != wLowN, CUBE_INNER_ERR_REPORT(op_name, "w_high N(%ld) must match w_low N(%ld)", N, wLowN),
                return ge::GRAPH_FAILED);

    yShape->SetDimNum(kMinDimNum);
    yShape->SetDim(kIdxM, M);
    yShape->SetDim(kIdxN, N);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MatmulEmuSplitWeight).InferShape(InferShapeForMatmulEmuSplitWeight);
} // namespace Ops::NN::MatMul
