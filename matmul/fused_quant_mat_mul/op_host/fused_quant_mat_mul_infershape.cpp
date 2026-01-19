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
 * \file fused_quant_mat_mul_infershape.cpp
 * \brief
 */
#include <string>

#include "common/op_host/matmul_common_infershape.h"
#include "error_util.h"
#include "log/log.h"
#include "runtime/infer_datatype_context.h"

using namespace gert;
namespace {
const size_t kX1Idx = 0;
const size_t kX2Idx = 1;
const size_t kBiasIdx = 2;
const size_t kX1ScaleIdx = 3;
const size_t kX2ScaleIdx = 4;
const size_t kYScaleIdx = 5;
const size_t kX1OffsetIdx = 6;
const size_t kX2OffsetIdx = 7;
const size_t kYOffsetIdx = 8;
const size_t kX2TableIdx = 9;
const size_t kX3Idx = 10;
const size_t kOutputIdx = 0;

const size_t kAttrDtypeIdx = 0;
const size_t kAttrComputeTypeIdx = 1;
const size_t kAttrTransposeX1Idx = 2;
const size_t kAttrTransposeX2Idx = 3;
const size_t kAttrGroupSizeIdx = 4;
const size_t kAttrFusedOpTypeIdx = 5;

const size_t kMatmulV2MinShapeSize = 2;

ge::graphStatus InferShapeForFusedQuantMatMul(InferShapeContext *context) {
    OP_CHECK_IF(context == nullptr, CUBE_INNER_ERR_REPORT("FusedQuantMatMul", "context is null"),
                return ge::GRAPH_FAILED);
    auto op_name = context->GetNodeName();
    auto shape_a = context->GetInputShape(kX1Idx);
    auto shape_b = context->GetInputShape(kX2Idx);
    auto shape_out = context->GetOutputShape(kOutputIdx);
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(shape_a == nullptr || shape_b == nullptr || shape_out == nullptr || attrs == nullptr,
                CUBE_INNER_ERR_REPORT(op_name, "shape or attrs is null"), return ge::GRAPH_FAILED);

    const bool *trans_a = attrs->GetAttrPointer<bool>(kAttrTransposeX1Idx);
    const bool *trans_b = attrs->GetAttrPointer<bool>(kAttrTransposeX2Idx);
    const char *fused_op_type = attrs->GetAttrPointer<char>(kAttrFusedOpTypeIdx);
    OP_CHECK_IF(trans_a == nullptr || trans_b == nullptr || fused_op_type == nullptr,
                CUBE_INNER_ERR_REPORT(op_name, "attribute is null"), return ge::GRAPH_FAILED);

    OP_CHECK_IF(strcmp(fused_op_type, "") == 0, CUBE_INNER_ERR_REPORT(op_name, "the fused_op_type must have input!"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(strcmp(fused_op_type, "relu") != 0 && strcmp(fused_op_type, "swiglu") != 0 && 
                strcmp(fused_op_type, "gelu_tanh") != 0 && strcmp(fused_op_type, "gelu_erf") != 0,
                CUBE_INNER_ERR_REPORT(op_name, "the input fused_op_type:%s is illegal, only relu/swiglu/gelu_tanh/gelu_erf support", fused_op_type),
                return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "a_shape: %s, b_shape: %s, transpose_a: %d, transpose_b: %d",
            Shape2String(*shape_a).c_str(), Shape2String(*shape_b).c_str(), *trans_a, *trans_b);

    // first transpose attr is transpose_x1, its index is 2 and bias input tensor index is 2, is_x2_packed is true
    return Ops::NN::InferShapeForBatchMatMul(context, kAttrTransposeX1Idx, kBiasIdx, true);
}

} // namespace

namespace Ops::NN::MatMul
{
IMPL_OP_INFERSHAPE(FusedQuantMatMul).InferShape(InferShapeForFusedQuantMatMul);
} // namespace name

