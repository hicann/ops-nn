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
 * \file fused_mat_mul_infershape.cpp
 * \brief
 */
#include <string>

#include "error_util.h"
#include "log/log.h"
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
using namespace gert;
namespace {
const size_t kMatMulX1Idx = 0;
const size_t kMatMulX2Idx = 1;
const size_t kMatMulX3Idx = 2;
const size_t kMatmulV2BiasShapeSize = 1;
const size_t kMatmulV2MinShapeSize = 2;
const size_t kFusedMatMulX3Idx = 3;
const size_t kOutputIdx = 0;

ge::graphStatus InferShapeForFusedMatMul(InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, CUBE_INNER_ERR_REPORT("FusedMatmul", "context is null"), return ge::GRAPH_FAILED);
    auto op_name = context->GetNodeName();
    auto shape_a = context->GetInputShape(kMatMulX1Idx);
    auto tensor_a = context->GetInputDesc(kMatMulX1Idx);
    auto shape_b = context->GetInputShape(kMatMulX2Idx);
    auto shape_bias = context->GetOptionalInputShape(kMatMulX3Idx);
    auto shape_c = context->GetOptionalInputShape(kFusedMatMulX3Idx);
    auto shape_out = context->GetOutputShape(kOutputIdx);
    auto attrs = context->GetAttrs();

    OP_CHECK_IF(
        shape_a == nullptr || shape_b == nullptr || shape_out == nullptr || attrs == nullptr || tensor_a == nullptr,
        CUBE_INNER_ERR_REPORT(op_name, "shape or attrs is null"), return ge::GRAPH_FAILED);

    const bool* trans_a = attrs->GetAttrPointer<bool>(kMatMulX1Idx);
    const bool* trans_b = attrs->GetAttrPointer<bool>(kMatMulX2Idx);
    const bool* enable_hf32 = attrs->GetAttrPointer<bool>(kMatMulX3Idx);
    const char* fused_op_type = attrs->GetAttrPointer<char>(kFusedMatMulX3Idx);

    OP_CHECK_IF(
        trans_a == nullptr || trans_b == nullptr || enable_hf32 == nullptr  || fused_op_type == nullptr,
        CUBE_INNER_ERR_REPORT(op_name, "attribute is null"), return ge::GRAPH_FAILED);

    ge::DataType dtype = tensor_a->GetDataType();
    OP_CHECK_IF(
        dtype == ge::DT_FLOAT && !(*enable_hf32),
        CUBE_INNER_ERR_REPORT(op_name, "fusedmatmul is only supported bf16/fp16/hf32, do not surrport fp32."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (strcmp(fused_op_type, "") != 0 && strcmp(fused_op_type, "add") != 0 && strcmp(fused_op_type, "mul") != 0 &&
         strcmp(fused_op_type, "gelu_erf") != 0 && strcmp(fused_op_type, "gelu_tanh") != 0 &&
         strcmp(fused_op_type, "relu") != 0),
        CUBE_INNER_ERR_REPORT(op_name, "fusedOpType must be in the type of /add/mul/gelu_erf/gelu_tanh/relu"),
        return ge::GRAPH_FAILED);
    // bias拦截
    if (strcmp(fused_op_type, "") != 0 && strcmp(fused_op_type, "relu") != 0 && strcmp(fused_op_type, "add") != 0 &&
        strcmp(fused_op_type, "mul") != 0) {
        OP_CHECK_IF(
            shape_bias != nullptr && shape_bias->GetDimNum() != 0,
            CUBE_INNER_ERR_REPORT(op_name, "not support bias in fused_op_type gelu_erf/gelu_tanh"),
            return ge::GRAPH_FAILED);
    }
    // x3输入拦截
    if (strcmp(fused_op_type, "add") == 0 || strcmp(fused_op_type, "mul") == 0) {
        OP_CHECK_IF(
            shape_c == nullptr, CUBE_INNER_ERR_REPORT(op_name, "shape or attrs is null"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            shape_c->GetDimNum() == 0,
            CUBE_INNER_ERR_REPORT(op_name, "shape c must have some data when fused_op_type is add or mul "),
            return ge::GRAPH_FAILED);
    }
    if (strcmp(fused_op_type, "") == 0 || strcmp(fused_op_type, "gelu_tanh") == 0 ||
        strcmp(fused_op_type, "gelu_erf") == 0 || strcmp(fused_op_type, "relu") == 0) {
        OP_CHECK_IF(
            shape_c != nullptr && shape_c->GetDimNum() != 0,
            CUBE_INNER_ERR_REPORT(
                op_name, "shape c must have no data when fused_op_type is '', gelu_tanh, gelu_erf or relu"),
            return ge::GRAPH_FAILED);
    }

    OP_LOGD(
        context->GetNodeName(), "a_shape: %s, b_shape: %s, transpose_a: %d, transpose_b: %d",
        Shape2String(*shape_a).c_str(), Shape2String(*shape_b).c_str(), *trans_a, *trans_b);

    OP_LOGD(op_name, "check the input shape length.");
    OP_CHECK_IF(
        (shape_a->GetDimNum() != kMatmulV2MinShapeSize || shape_b->GetDimNum() != kMatmulV2MinShapeSize),
        CUBE_INNER_ERR_REPORT(
            op_name, "input dim num[%zu] [%zu] is not 2!", shape_a->GetDimNum(), shape_b->GetDimNum()),
        return ge::GRAPH_FAILED);

    int idx_m = *trans_a ? 1 : 0;
    int idx_k_a = *trans_a ? 0 : 1;
    int idx_k_b = *trans_b ? 1 : 0;
    int idx_n = *trans_b ? 0 : 1;

    OP_CHECK_IF(
        shape_a->GetDim(idx_k_a) != shape_b->GetDim(idx_k_b),
        CUBE_INNER_ERR_REPORT(
            op_name, "The k-axis of a(%ld) and b(%ld) tensors must be the same", shape_a->GetDim(idx_k_a),
            shape_b->GetDim(idx_k_b)),
        return ge::GRAPH_FAILED);

    if (shape_bias != nullptr && shape_bias->GetDimNum() != 0) {
        if (shape_bias->GetDimNum() == 1) {
            OP_CHECK_IF(
                shape_b->GetDim(idx_n) != shape_bias->GetDim(0),
                CUBE_INNER_ERR_REPORT(
                    op_name, "The n(%ld) tensors must be the same bias(%ld,)", shape_b->GetDim(idx_n),
                    shape_bias->GetDim(0)),
                return ge::GRAPH_FAILED);
        } else if (shape_bias->GetDimNum() == 2) {
            OP_CHECK_IF(
                shape_bias->GetDim(0) != 1,
                CUBE_INNER_ERR_REPORT(op_name, "The m(%ld) of bias must be 1", shape_bias->GetDim(0)),
                return ge::GRAPH_FAILED);
            OP_CHECK_IF(
                shape_bias->GetDim(1) != shape_b->GetDim(idx_n),
                CUBE_INNER_ERR_REPORT(
                    op_name, "The n(%ld) tensors must be the same bias(%ld,)", shape_b->GetDim(idx_n),
                    shape_bias->GetDim(1)),
                return ge::GRAPH_FAILED);
        } else {
            OP_LOGD(op_name, "input dim num[%zu] of bias is illegal", shape_bias->GetDimNum());
            return ge::GRAPH_FAILED;
        }
    }

    if (shape_c != nullptr && shape_c->GetDimNum() != 0) {
        OP_LOGD(context->GetNodeName(), "c_shape: %s", Shape2String(*shape_c).c_str());
        OP_CHECK_IF(
            (shape_c->GetDimNum() != kMatmulV2MinShapeSize),
            CUBE_INNER_ERR_REPORT(op_name, "input dim num[%zu] is not 2!", shape_c->GetDimNum()),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            shape_a->GetDim(idx_m) != shape_c->GetDim(0) || shape_b->GetDim(idx_n) != shape_c->GetDim(1),
            CUBE_INNER_ERR_REPORT(
                op_name, "The m(%ld), n(%ld) tensors must be the same c(%ld, %ld)", shape_a->GetDim(idx_m),
                shape_b->GetDim(idx_n), shape_c->GetDim(0), shape_c->GetDim(1)),
            return ge::GRAPH_FAILED);
    }
    shape_out->SetDimNum(kMatmulV2MinShapeSize);
    shape_out->SetDim(0, shape_a->GetDim(idx_m));
    shape_out->SetDim(1, shape_b->GetDim(idx_n));

    OP_LOGI(op_name, "end infershape.");
    return ge::GRAPH_SUCCESS;
}

} // namespace

namespace Ops::NN::MatMul {
IMPL_OP_INFERSHAPE(FusedMatMul).InferShape(InferShapeForFusedMatMul);
}