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
 * \file gemm_v3_infershape.cpp
 * \brief
 */
#include <string>

#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"

#include "error_util.h"
#include "log/log.h"
using namespace gert;
namespace {
const size_t kMatMulX1Idx = 0;
const size_t kMatMulX2Idx = 1;
const size_t kMatmulV2MinShapeSize = 2;
const size_t kOutputIdx = 0;
constexpr float epsilon = 1e-6f;

ge::graphStatus InferShapeForGemmV3(InferShapeContext* context)
{
    auto op_name = context->GetNodeName();
    auto shape_a = context->GetInputShape(kMatMulX1Idx);
    auto shape_b = context->GetInputShape(kMatMulX2Idx);
    auto shape_c = context->GetOptionalInputShape(kMatmulV2MinShapeSize);
    auto shape_out = context->GetOutputShape(kOutputIdx);
    auto attrs = context->GetAttrs();
    // 当前仅支持累加场景，shape_c不支持为空
    OP_CHECK_IF(
        shape_a == nullptr || shape_b == nullptr || shape_c == nullptr || shape_out == nullptr || attrs == nullptr,
        CUBE_INNER_ERR_REPORT(op_name, "gemmv3 input or output or attrs is null"), return ge::GRAPH_FAILED);

    const float* alpha = attrs->GetAttrPointer<float>(0);
    const float* beta = attrs->GetAttrPointer<float>(1);
    OP_CHECK_IF(
        alpha == nullptr || beta == nullptr, CUBE_INNER_ERR_REPORT(op_name, "gemmv3 attribute is null"),
        return ge::GRAPH_FAILED);
    // 校验alpha和beta是否等于1.0f
    OP_CHECK_IF(
        std::abs(*alpha - 1.0f) > epsilon || std::abs(*beta - 1.0f) > epsilon,
        CUBE_INNER_ERR_REPORT(op_name, "alpha and beta must be 1.0"), return ge::GRAPH_FAILED);
    const bool* trans_a = attrs->GetAttrPointer<bool>(2);
    const bool* trans_b = attrs->GetAttrPointer<bool>(3);
    const bool* enable_hf32 = attrs->GetAttrPointer<bool>(4);
    OP_CHECK_IF(
        trans_a == nullptr || trans_b == nullptr || enable_hf32 == nullptr,
        CUBE_INNER_ERR_REPORT(op_name, "gemmv3 attribute is null"), return ge::GRAPH_FAILED);
    OP_LOGD(
        context->GetNodeName(),
        "gemmv3 a_shape: %s, b_shape: %s, c_shape:%s, transpose_a: %d, transpose_b: %d, enable_hf32: %d",
        Shape2String(*shape_a).c_str(), Shape2String(*shape_b).c_str(), Shape2String(*shape_c).c_str(), *trans_a,
        *trans_b, *enable_hf32);
    OP_CHECK_IF(
        (shape_a->GetDimNum() != kMatmulV2MinShapeSize || shape_b->GetDimNum() != kMatmulV2MinShapeSize ||
         shape_c->GetDimNum() != kMatmulV2MinShapeSize),
        CUBE_INNER_ERR_REPORT(
            op_name, "gemmv3 input dim num[%zu] [%zu] [%zu]is not 2!", shape_a->GetDimNum(), shape_b->GetDimNum(),
            shape_c->GetDimNum()),
        return ge::GRAPH_FAILED);
    size_t idx_m = static_cast<size_t>(*trans_a);
    size_t idx_k_a = idx_m == 0UL ? 1UL : 0UL;
    size_t idx_k_b = static_cast<size_t>(*trans_b);
    size_t idx_n = idx_k_b == 0UL ? 1UL : 0UL;
    // 校验a和b的K是否相同
    OP_CHECK_IF(
        shape_a->GetDim(idx_k_a) != shape_b->GetDim(idx_k_b),
        CUBE_INNER_ERR_REPORT(
            op_name, "The k-axis of a(%ld) and b(%ld) tensors must be the same", shape_a->GetDim(idx_k_a),
            shape_b->GetDim(idx_k_b)),
        return ge::GRAPH_FAILED);
    // 校验y和a b 的M N是否相同
    OP_CHECK_IF(
        shape_a->GetDim(idx_m) != shape_c->GetDim(0) || shape_b->GetDim(idx_n) != shape_c->GetDim(1),
        CUBE_INNER_ERR_REPORT(
            op_name, "The m(%ld), n(%ld) tensors must be the same c(%ld, %ld)", shape_a->GetDim(idx_m),
            shape_b->GetDim(idx_n), shape_c->GetDim(0), shape_c->GetDim(1)),
        return ge::GRAPH_FAILED);
    shape_out->SetDimNum(kMatmulV2MinShapeSize);
    shape_out->SetDim(0, shape_a->GetDim(idx_m));
    shape_out->SetDim(1, shape_b->GetDim(idx_n));
    // 校验y和c是否相同shape
    OP_CHECK_IF(
        shape_c->GetDim(0) != shape_out->GetDim(0) || shape_c->GetDim(1) != shape_out->GetDim(1),
        CUBE_INNER_ERR_REPORT(
            op_name, "The y(%ld, %ld) tensors must be same with c(%ld, %ld)", shape_out->GetDim(0),
            shape_out->GetDim(1), shape_c->GetDim(0), shape_c->GetDim(1)),
        return ge::GRAPH_FAILED);
    OP_LOGI(op_name, "end infershape for gemmv3.");
    return ge::GRAPH_SUCCESS;
}
} // namespace

namespace Ops::NN::MatMul {
    IMPL_OP_INFERSHAPE(GemmV3).InferShape(InferShapeForGemmV3);
}
