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
#include "error_util.h"
#include "log/log.h"
#include "common/op_host/matmul_common_infershape.h"
#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include <algorithm>
using namespace gert;
namespace {
const int kMatMulX1Idx = 0;
const int kMatMulX2Idx = 1;
const int kMatMulX3Idx = 2;
const int kMatmulV2BiasShapeSize = 1;
const int kMatmulV2MinShapeSize = 2;
const int kMatmulV2MaxShapeSize = 3;
const int kMatmulV2ReluMaxShapeSize = 6;
const int kFusedMatMulX3Idx = 3;
const int kInnerPreciseIdx = 4;
const int kOutputIdx = 0;
const int DIM_SIZE_TWO = 2;
const int64_t INNER_PRECISE_HIGH_PRECISION = 0;
const int64_t INNER_PRECISE_HIGH_PERFORMANCE = 1;

const std::vector<const char*> kAllSupportedOpTypes = {
    "", "16cast32", "add", "mul", "gelu_erf", "gelu_tanh", "relu", "quant", "relu_quant", "scale_add", "sigmoid"};
const std::vector<const char*> kSupportedBiasOpTypes = {"",      "16cast32",   "relu",     "add",       "mul",
                                                        "quant", "relu_quant", "gelu_erf", "gelu_tanh", "sigmoid"};
const std::vector<const char*> kSupportedX3OpTypes = {"add", "mul", "quant", "relu_quant", "scale_add"};
const std::vector<const char*> kQuantOpTypes = {"quant", "relu_quant"};
const std::vector<const char*> kBroadcastBatchOpTypes = {"relu", "quant", "relu_quant"};

bool IsInSupportedOpTypes(const char* fusedOpType, const std::vector<const char*>& types)
{
    for (const auto& type : types) {
        if (type && fusedOpType && strcmp(fusedOpType, type) == 0) {
            return true;
        }
    }
    return false;
}

ge::graphStatus InferShapeForFusedMatMul(InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, CUBE_INNER_ERR_REPORT("FusedMatmul", "context is null"), return ge::GRAPH_FAILED);
    auto op_name = context->GetNodeName();
    auto shape_x1 = context->GetInputShape(kMatMulX1Idx);
    auto tensor_x1 = context->GetInputDesc(kMatMulX1Idx);
    auto shape_x2 = context->GetInputShape(kMatMulX2Idx);
    auto shape_bias = context->GetOptionalInputShape(kMatMulX3Idx);
    auto shape_x3 = context->GetOptionalInputShape(kFusedMatMulX3Idx);
    auto shape_out = context->GetOutputShape(kOutputIdx);
    auto attrs = context->GetAttrs();

    OP_CHECK_IF(
        shape_x1 == nullptr || shape_x2 == nullptr || shape_out == nullptr || attrs == nullptr || tensor_x1 == nullptr,
        CUBE_INNER_ERR_REPORT(op_name, "shape or attrs is null"), return ge::GRAPH_FAILED);
    const bool* trans_a = attrs->GetAttrPointer<bool>(kMatMulX1Idx);
    const bool* trans_b = attrs->GetAttrPointer<bool>(kMatMulX2Idx);
    const bool* enable_hf32 = attrs->GetAttrPointer<bool>(kMatMulX3Idx);
    const char* fused_op_type = attrs->GetAttrPointer<char>(kFusedMatMulX3Idx);
    const int64_t* inner_precise = attrs->GetAttrPointer<int64_t>(kInnerPreciseIdx);

    OP_CHECK_IF(trans_a == nullptr || trans_b == nullptr || enable_hf32 == nullptr || fused_op_type == nullptr ||
                    inner_precise == nullptr,
                CUBE_INNER_ERR_REPORT(op_name, "attribute is null"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(*inner_precise != INNER_PRECISE_HIGH_PRECISION && *inner_precise != INNER_PRECISE_HIGH_PERFORMANCE,
                CUBE_INNER_ERR_REPORT(op_name, "inner_precise only supports 0 or 1"), return ge::GRAPH_FAILED);

    // FLOAT storage is legal. A5 aclnn still rejects KEEP_DTYPE FP32; A2/A3 allow it via MatMulV3.
    // OpType合法性校验
    OP_CHECK_IF(!IsInSupportedOpTypes(fused_op_type, kAllSupportedOpTypes),
                CUBE_INNER_ERR_REPORT(op_name,
                                      "fusedOpType must be in the type of "
                                      "''/16cast32/add/mul/gelu_erf/gelu_tanh/relu/quant/relu_quant/scale_add/sigmoid"),
                return ge::GRAPH_FAILED);
    // 不支持bias的OpType拦截bias
    if (!IsInSupportedOpTypes(fused_op_type, kSupportedBiasOpTypes)) {
        OP_CHECK_IF(shape_bias != nullptr && shape_bias->GetDimNum() != 0,
                    CUBE_INNER_ERR_REPORT(op_name, "not support bias in fused_op_type %s", fused_op_type),
                    return ge::GRAPH_FAILED);
    }
    const bool is_quant_op = IsInSupportedOpTypes(fused_op_type, kQuantOpTypes);
    // 支持x3输入的OpType拦截x3为空
    if (IsInSupportedOpTypes(fused_op_type, kSupportedX3OpTypes)) {
        OP_CHECK_IF(shape_x3 == nullptr || shape_x3->GetDimNum() == 0,
                    CUBE_INNER_ERR_REPORT(
                        op_name, "shape x3 must be valid when fused_op_type is add/mul/quant/relu_quant/scale_add"),
                    return ge::GRAPH_FAILED);
    } else {
        // 不支持x3输入的OpType拦截x3为非空
        OP_CHECK_IF(shape_x3 != nullptr && shape_x3->GetDimNum() != 0,
                    CUBE_INNER_ERR_REPORT(
                        op_name,
                        "shape x3 must have no data when fused_op_type is ''/16cast32/gelu_tanh/gelu_erf/relu/sigmoid"),
                    return ge::GRAPH_FAILED);
    }
    OP_LOGD(context->GetNodeName(), "a_shape: %s, b_shape: %s, transpose_a: %d, transpose_b: %d",
            Shape2String(*shape_x1).c_str(), Shape2String(*shape_x2).c_str(), *trans_a, *trans_b);

    OP_LOGD(op_name, "check the input shape length.");
    int dim_a = shape_x1->GetDimNum();
    int dim_b = shape_x2->GetDimNum();
    const bool is_empty = strcmp(fused_op_type, "") == 0;
    const bool is_sigmoid = strcmp(fused_op_type, "sigmoid") == 0;
    const bool support_broadcast_batch = IsInSupportedOpTypes(fused_op_type, kBroadcastBatchOpTypes);
    const int max_dim = is_sigmoid                            ? kMatmulV2MinShapeSize :
                        (is_empty || support_broadcast_batch) ? kMatmulV2ReluMaxShapeSize :
                                                                kMatmulV2MaxShapeSize;
    const int dim_out = support_broadcast_batch ? std::max(dim_a, dim_b) : dim_a;
    OP_CHECK_IF(dim_a < kMatmulV2MinShapeSize || dim_b < kMatmulV2MinShapeSize || dim_a > max_dim || dim_b > max_dim ||
                    (!support_broadcast_batch && dim_a != dim_b),
                CUBE_INNER_ERR_REPORT(op_name, "input dim num[%d] [%d] is illegal!", dim_a, dim_b),
                return ge::GRAPH_FAILED);
    if (!support_broadcast_batch && is_empty) {
        for (int i = 0; i < dim_a - kMatmulV2MinShapeSize; ++i) {
            OP_CHECK_IF(shape_x1->GetDim(i) != shape_x2->GetDim(i),
                        CUBE_INNER_ERR_REPORT(
                            op_name,
                            "current fused op type only supports no-broadcast batch shape, but x1 batch dim[%d] "
                            "is %ld, x2 batch dim[%d] is %ld",
                            i, shape_x1->GetDim(i), i, shape_x2->GetDim(i)),
                        return ge::GRAPH_FAILED);
        }
    }

    int idx_m = *trans_a ? 0 : 1;
    int idx_k_a = *trans_a ? 1 : 0;
    int idx_k_b = *trans_b ? 0 : 1;
    int idx_n = *trans_b ? 1 : 0;

    int64_t a_m = shape_x1->GetDim(dim_a - idx_m - 1);
    int64_t b_n = shape_x2->GetDim(dim_b - idx_n - 1);
    int64_t a_k = shape_x1->GetDim(dim_a - idx_k_a - 1);
    int64_t b_k = shape_x2->GetDim(dim_b - idx_k_b - 1);

    OP_CHECK_IF(a_k != b_k,
                CUBE_INNER_ERR_REPORT(op_name, "The k-axis of x1(%d) and x2(%d) tensors must be the same", a_k, b_k),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(is_sigmoid && a_k == 0,
                CUBE_INNER_ERR_REPORT(op_name, "The K-axis of x1 and x2 must be greater than 0 for sigmoid op type"),
                return ge::GRAPH_FAILED);

    if (shape_bias != nullptr && shape_bias->GetDimNum() != 0) {
        const size_t bias_dim = shape_bias->GetDimNum();
        if (bias_dim == 1) {
            OP_CHECK_IF(b_n != shape_bias->GetDim(0),
                        CUBE_INNER_ERR_REPORT(op_name, "The n(%d) tensors must be the same bias(%ld,)", b_n,
                                              shape_bias->GetDim(0)),
                        return ge::GRAPH_FAILED);
        } else if (bias_dim == DIM_SIZE_TWO) {
            OP_CHECK_IF(is_quant_op, CUBE_INNER_ERR_REPORT(op_name, "Only 1D bias is supported for quant/relu_quant"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(shape_bias->GetDim(0) != 1,
                        CUBE_INNER_ERR_REPORT(op_name, "The m(%ld) of bias must be 1", shape_bias->GetDim(0)),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(shape_bias->GetDim(1) != b_n,
                        CUBE_INNER_ERR_REPORT(op_name, "The n(%d) tensors must be the same bias(%ld,)", b_n,
                                              shape_bias->GetDim(1)),
                        return ge::GRAPH_FAILED);
        } else {
            OP_LOGD(op_name, "input dim num[%zu] of bias is illegal", bias_dim);
            return ge::GRAPH_FAILED;
        }
    }

    if (shape_x3 != nullptr && shape_x3->GetDimNum() != 0) {
        if (is_quant_op) {
            OP_CHECK_IF(shape_x3->GetDimNum() != 1 || shape_x3->GetShapeSize() != 1,
                        CUBE_INNER_ERR_REPORT(op_name, "shape x3 must be [1] when fused_op_type is quant/relu_quant"),
                        return ge::GRAPH_FAILED);
        } else {
            int dim_c = shape_x3->GetDimNum();
            OP_LOGD(context->GetNodeName(), "c_shape: %s", Shape2String(*shape_x3).c_str());
            OP_CHECK_IF(dim_c < kMatmulV2MinShapeSize || dim_c > kMatmulV2MaxShapeSize,
                        CUBE_INNER_ERR_REPORT(op_name, "input dim num[%d] is illegal!", dim_c),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(dim_c == kMatmulV2MinShapeSize && (a_m != shape_x3->GetDim(0) || b_n != shape_x3->GetDim(1)),
                        CUBE_INNER_ERR_REPORT(op_name, "The m(%d), n(%d) tensors must be the same c(%ld, %ld)", a_m,
                                              b_n, shape_x3->GetDim(0), shape_x3->GetDim(1)),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF(dim_c == kMatmulV2MaxShapeSize &&
                            (dim_a != kMatmulV2MaxShapeSize ||
                             (shape_x3->GetDim(0) != 1 && shape_x3->GetDim(0) != shape_x1->GetDim(0)) ||
                             a_m != shape_x3->GetDim(1) || b_n != shape_x3->GetDim(kMatMulX3Idx)),
                        CUBE_INNER_ERR_REPORT(op_name, "The shape x3(%ld, %ld, %ld) is illegal!", shape_x3->GetDim(0),
                                              shape_x3->GetDim(1), shape_x3->GetDim(kMatMulX3Idx)),
                        return ge::GRAPH_FAILED);
        }
    }
    shape_out->SetDimNum(dim_out);
    if (support_broadcast_batch) {
        const int valid_offset = dim_out - std::min(dim_a, dim_b);
        const Shape& shape_long = dim_a < dim_b ? *shape_x2 : *shape_x1;
        const Shape& shape_short = dim_a < dim_b ? *shape_x1 : *shape_x2;
        for (int i = 0; i < valid_offset; ++i) {
            shape_out->SetDim(i, shape_long.GetDim(i));
        }
        for (int i = valid_offset; i < dim_out - kMatmulV2MinShapeSize; ++i) {
            int64_t batch_dim_out = 1;
            OP_CHECK_IF(!Ops::NN::BroadcastBatchDim(op_name, shape_short.GetDim(i - valid_offset), shape_long.GetDim(i),
                                                    batch_dim_out),
                        OP_LOGE_WITHOUT_REPORT(op_name, "failed to infer broadcast batch shape"),
                        return ge::GRAPH_FAILED);
            shape_out->SetDim(i, batch_dim_out);
        }
    } else {
        for (int i = 0; i < dim_out - kMatmulV2MinShapeSize; ++i) {
            shape_out->SetDim(i, shape_x1->GetDim(i));
        }
    }
    shape_out->SetDim(dim_out - DIM_SIZE_TWO, a_m);
    shape_out->SetDim(dim_out - 1, b_n);

    OP_LOGI(op_name, "end infershape.");
    return ge::GRAPH_SUCCESS;
}

} // namespace

namespace Ops::NN::MatMul {
IMPL_OP_INFERSHAPE(FusedMatMul)
    .InferShape(InferShapeForFusedMatMul)
    .PrivateAttr("enable_uncache", static_cast<int64_t>(0))
    .PrivateAttr("fixed_shift_value", static_cast<int64_t>(0));
}
