/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_matmul_activation_quant_infershape.cpp
 * \brief InferShape and InferDataType for QuantMatmulActivationQuant
 */
#include <string>

#include "common/op_host/matmul_common_infershape.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "matmul/common/op_host/log_format_util.h"
#include "runtime/infer_datatype_context.h"

using Ops::NN::FormatString;

namespace {
constexpr uint32_t X1_INDEX = 0;
constexpr uint32_t X2_INDEX = 1;
constexpr uint32_t BIAS_INDEX = 2;
constexpr uint32_t X1_SCALE_INDEX = 3;
constexpr uint32_t X2_SCALE_INDEX = 4;

constexpr uint32_t Y_INDEX = 0;
constexpr uint32_t Y_SCALE_INDEX = 1;

constexpr uint32_t Y_DTYPE_INDEX = 4;
constexpr uint32_t ACTIVATION_TYPE_INDEX = 3;
constexpr uint32_t GROUP_SIZE_INDEX = 2;
constexpr uint32_t QUANT_MODE_INDEX = 5;
constexpr uint32_t ROUND_MODE_INDEX = 6;
constexpr uint32_t SCALE_ALG_INDEX = 7;
constexpr int64_t MX_GROUP_SIZE = 4295032864;

constexpr int64_t ALIGN_NUM = 2;
constexpr int64_t MX_BLOCK_SIZE = 32;
constexpr int64_t SWIGLU_N_ALIGN = 64;
constexpr int64_t UNKNOWN_DIM = -1;
constexpr int64_t UNKNOWN_DIM_NUM = -2;
constexpr size_t QUANT_MATMUL_MIN_SHAPE_SIZE = 2;
constexpr size_t QUANT_MATMUL_MAX_SHAPE_SIZE = 6;
constexpr const char* OP_NAME = "QuantMatmulActivationQuant";

const char* GetValidOpNameForInferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return OP_NAME;
    }
    const char* nodeName = context->GetNodeName();
    if (nodeName != nullptr && nodeName[0] != '\0') {
        return nodeName;
    }
    return OP_NAME;
}

const char* GetValidOpNameForInferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return OP_NAME;
    }
    const char* nodeName = context->GetNodeName();
    if (nodeName != nullptr && nodeName[0] != '\0') {
        return nodeName;
    }
    return OP_NAME;
}

template <typename Context>
static bool CheckSwiGluAttrs(Context* context, const char* opName)
{
    const auto* attrs = context->GetAttrs();
    const auto* activation = attrs == nullptr ? nullptr : attrs->template GetAttrPointer<char>(ACTIVATION_TYPE_INDEX);
    if (activation == nullptr || std::string(activation) != "swiglu") {
        return true;
    }
    const auto* groupSize = attrs->template GetAttrPointer<int64_t>(GROUP_SIZE_INDEX);
    const auto* quantMode = attrs->template GetAttrPointer<char>(QUANT_MODE_INDEX);
    const auto* roundMode = attrs->template GetAttrPointer<char>(ROUND_MODE_INDEX);
    const auto* scaleAlg = attrs->template GetAttrPointer<int64_t>(SCALE_ALG_INDEX);
    const auto* yDtype = attrs->template GetAttrPointer<int64_t>(Y_DTYPE_INDEX);
    if ((groupSize != nullptr && *groupSize != 0 && *groupSize != MX_GROUP_SIZE) ||
        (quantMode != nullptr && std::string(quantMode) != "mx") ||
        (roundMode != nullptr && std::string(roundMode) != "rint") ||
        (scaleAlg != nullptr && *scaleAlg != 0 && *scaleAlg != 1) ||
        (yDtype != nullptr && *yDtype != ge::DT_FLOAT8_E4M3FN && *yDtype != ge::DT_FLOAT8_E5M2)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            opName, "SwiGLU MX attributes", "invalid",
            "the attributes require FP8, group_size=[1,1,32], quant_mode=mx, round_mode=rint and scale_alg=0/1");
        return false;
    }
    return true;
}

static bool CheckSwiGluInputTypes(gert::InferShapeContext* context, const char* opName)
{
    const auto isFp8 = [](const auto* desc) {
        return desc != nullptr &&
               (desc->GetDataType() == ge::DT_FLOAT8_E4M3FN || desc->GetDataType() == ge::DT_FLOAT8_E5M2);
    };
    const auto* x1Desc = context->GetInputDesc(X1_INDEX);
    const auto* x2Desc = context->GetInputDesc(X2_INDEX);
    const auto x2Format = x2Desc == nullptr ? ge::FORMAT_RESERVED : ge::GetPrimaryFormat(x2Desc->GetStorageFormat());
    const bool isSupportedX2Format = x2Format == ge::FORMAT_ND || (x2Format == ge::FORMAT_FRACTAL_NZ &&
                                                                   x2Desc->GetDataType() == ge::DT_FLOAT8_E4M3FN);
    if (!isFp8(x1Desc) || ge::GetPrimaryFormat(x1Desc->GetStorageFormat()) != ge::FORMAT_ND || !isFp8(x2Desc) ||
        !isSupportedX2Format) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName, "x1, x2", "unsupported dtype/format",
                                               "SwiGLU requires FP8 x1/x2, ND x1, and ND x2 or E4M3FN FRACTAL_NZ x2");
        return false;
    }
    const auto* attrs = context->GetAttrs();
    const auto* transposeX1 = attrs == nullptr ? nullptr : attrs->GetAttrPointer<bool>(0);
    if (x2Format == ge::FORMAT_FRACTAL_NZ && transposeX1 != nullptr && *transposeX1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "transposeX1", "true",
                                              "SwiGLU WeightNZ requires transposeX1 to be false");
        return false;
    }
    const auto* yDtype = attrs == nullptr ? nullptr : attrs->GetAttrPointer<int64_t>(Y_DTYPE_INDEX);
    if (yDtype == nullptr || *yDtype != static_cast<int64_t>(x1Desc->GetDataType())) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName, "y_dtype, x1 dtype",
            FormatString("%lld, %s", static_cast<long long>(yDtype == nullptr ? -1LL : *yDtype),
                         ge::TypeUtils::DataTypeToSerialString(x1Desc->GetDataType()).c_str())
                .c_str(),
            "SwiGLU requires y_dtype to match x1");
        return false;
    }
    for (uint32_t index : {X1_SCALE_INDEX, X2_SCALE_INDEX}) {
        const auto* desc = context->GetOptionalInputDesc(index);
        if (desc == nullptr || desc->GetDataType() != ge::DT_FLOAT8_E8M0) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                opName, index == X1_SCALE_INDEX ? "x1Scale" : "x2Scale",
                desc == nullptr ? "null" : ge::TypeUtils::DataTypeToSerialString(desc->GetDataType()).c_str(),
                "SwiGLU input scales must have FLOAT8_E8M0 dtype");
            return false;
        }
    }
    const auto* bias = context->GetOptionalInputShape(BIAS_INDEX);
    const auto* biasDesc = context->GetOptionalInputDesc(BIAS_INDEX);
    const bool hasBias = bias != nullptr && bias->GetDimNum() > 0;
    if (hasBias && (biasDesc == nullptr || biasDesc->GetDataType() != ge::DT_FLOAT)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            opName, "bias",
            biasDesc == nullptr ? "null" : ge::TypeUtils::DataTypeToSerialString(biasDesc->GetDataType()).c_str(),
            "SwiGLU bias must have FLOAT dtype");
        return false;
    }
    return true;
}

static ge::graphStatus InferShape4QuantMatmulActivationQuant(gert::InferShapeContext* context)
{
    OP_LOGI(context, "Begin InferShape4QuantMatmulActivationQuant.");

    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const char* opName = GetValidOpNameForInferShape(context);

    auto shape_x1 = context->GetInputShape(X1_INDEX);
    auto shape_x2 = context->GetInputShape(X2_INDEX);
    auto shape_x1_scale = context->GetOptionalInputShape(X1_SCALE_INDEX);
    auto shape_x2_scale = context->GetOptionalInputShape(X2_SCALE_INDEX);

    OP_CHECK_NULL_WITH_CONTEXT(context, shape_x1);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_x2);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_x1_scale);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape_x2_scale);

    auto dim_a = shape_x1->GetDimNum();
    auto dim_b = shape_x2->GetDimNum();
    bool any_unknown_rank = Ops::NN::CheckIsUnknownDimNum(*shape_x1) || Ops::NN::CheckIsUnknownDimNum(*shape_x2);

    auto out_shape_y = context->GetOutputShape(Y_INDEX);
    auto out_shape_y_scale = context->GetOutputShape(Y_SCALE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape_y);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape_y_scale);
    if (!CheckSwiGluAttrs(context, opName)) {
        return ge::GRAPH_FAILED;
    }
    const auto* shapeAttrs = context->GetAttrs();
    const auto* activation = shapeAttrs == nullptr ? nullptr : shapeAttrs->GetAttrPointer<char>(ACTIVATION_TYPE_INDEX);
    const bool is_swiglu = activation != nullptr && std::string(activation) == "swiglu";
    if (is_swiglu && !CheckSwiGluInputTypes(context, opName)) {
        return ge::GRAPH_FAILED;
    }

    if (!any_unknown_rank && (dim_a < QUANT_MATMUL_MIN_SHAPE_SIZE || dim_a > QUANT_MATMUL_MAX_SHAPE_SIZE ||
                              dim_b < QUANT_MATMUL_MIN_SHAPE_SIZE || dim_b > QUANT_MATMUL_MAX_SHAPE_SIZE)) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(opName, "x1, x2",
                                                  Ops::NN::FormatString("%zuD, %zuD", dim_a, dim_b).c_str(),
                                                  "the ranks of x1 and x2 must be in the range [2, 6]");
        return ge::GRAPH_FAILED;
    }

    if (any_unknown_rank) {
        out_shape_y->SetDimNum(1);
        out_shape_y->SetDim(0, UNKNOWN_DIM_NUM);
        out_shape_y_scale->SetDimNum(1);
        out_shape_y_scale->SetDim(0, UNKNOWN_DIM_NUM);
        OP_LOGD(opName, "Input rank is unknown; setting y and yScale to unknown rank [-2].");
        return ge::GRAPH_SUCCESS;
    }

    OP_LOGD(opName, "Input shapes: x1=%s, x2=%s", Ops::Base::ToString(*shape_x1).c_str(),
            Ops::Base::ToString(*shape_x2).c_str());

    auto attrs = context->GetAttrs();
    bool trans_x1 = false;
    bool trans_x2 = false;
    if (attrs != nullptr) {
        const bool* trans_x1_ptr = attrs->GetAttrPointer<bool>(0);
        const bool* trans_x2_ptr = attrs->GetAttrPointer<bool>(1);
        if (trans_x1_ptr != nullptr) {
            trans_x1 = *trans_x1_ptr;
        }
        if (trans_x2_ptr != nullptr) {
            trans_x2 = *trans_x2_ptr;
        }
    }

    int64_t M = trans_x1 ? shape_x1->GetDim(dim_a - 1) : shape_x1->GetDim(dim_a - 2);
    int64_t K = trans_x1 ? shape_x1->GetDim(dim_a - 2) : shape_x1->GetDim(dim_a - 1);
    int64_t K_x2 = trans_x2 ? shape_x2->GetDim(dim_b - 1) : shape_x2->GetDim(dim_b - 2);
    int64_t N = trans_x2 ? shape_x2->GetDim(dim_b - 2) : shape_x2->GetDim(dim_b - 1);

    if (K != UNKNOWN_DIM && K_x2 != UNKNOWN_DIM && K != K_x2) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName, "x1 K, x2 K",
            FormatString("%lld, %lld", static_cast<long long>(K), static_cast<long long>(K_x2)).c_str(),
            "the K dimension of x1 and x2 must be equal");
        return ge::GRAPH_FAILED;
    }
    if (is_swiglu && N != UNKNOWN_DIM && (N <= 0 || N % SWIGLU_N_ALIGN != 0)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "x2 N", std::to_string(N).c_str(),
                                              "SwiGLU requires positive pre-activation N divisible by 64");
        return ge::GRAPH_FAILED;
    }
    const int64_t outputN = is_swiglu && N >= 0 ? N / 2 : N;

    size_t num_dim = std::max(dim_a, dim_b);
    out_shape_y->SetDimNum(num_dim);

    // batch 维从右对齐，x1/x2 的 batch 维度数各自不同，不足的补 1
    size_t x1BatchCount = (dim_a >= 2) ? (dim_a - 2) : 0;
    size_t x2BatchCount = (dim_b >= 2) ? (dim_b - 2) : 0;
    size_t batchDimNum = std::max(x1BatchCount, x2BatchCount);
    for (size_t i = 0; i < num_dim - 2; ++i) {
        // 从右侧对齐的 batch 维计算：逻辑输出位置 i 对应 x1/x2 中的右对齐索引
        int64_t x1Idx = static_cast<int64_t>(i) - static_cast<int64_t>(batchDimNum - x1BatchCount);
        int64_t x2Idx = static_cast<int64_t>(i) - static_cast<int64_t>(batchDimNum - x2BatchCount);
        int64_t dim_a_val = (x1Idx >= 0) ? shape_x1->GetDim(static_cast<size_t>(x1Idx)) : 1;
        int64_t dim_b_val = (x2Idx >= 0) ? shape_x2->GetDim(static_cast<size_t>(x2Idx)) : 1;

        if (dim_a_val != UNKNOWN_DIM && dim_b_val != UNKNOWN_DIM && dim_a_val != dim_b_val && dim_a_val != 1 &&
            dim_b_val != 1) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                opName, "x1/x2 batch dim",
                FormatString("dim %zu: %lld, %lld", i, static_cast<long long>(dim_a_val),
                             static_cast<long long>(dim_b_val))
                    .c_str(),
                "the batch dimensions of x1 and x2 must be broadcastable");
            return ge::GRAPH_FAILED;
        }

        // 动态维广播：一侧未知时输出仍应保持未知，除非另一侧为已知且大于 1
        int64_t broadcast_dim;
        if (dim_a_val == UNKNOWN_DIM || dim_b_val == UNKNOWN_DIM) {
            int64_t known = (dim_a_val != UNKNOWN_DIM) ? dim_a_val :
                            (dim_b_val != UNKNOWN_DIM) ? dim_b_val :
                                                         UNKNOWN_DIM;
            broadcast_dim = (known > 1) ? known : UNKNOWN_DIM;
        } else {
            broadcast_dim = std::max(dim_a_val, dim_b_val);
        }
        out_shape_y->SetDim(i, broadcast_dim);
    }

    out_shape_y->SetDim(num_dim - 2, M);
    out_shape_y->SetDim(num_dim - 1, outputN);

    if (is_swiglu) {
        const auto* bias = context->GetOptionalInputShape(BIAS_INDEX);
        if (bias != nullptr && bias->GetDimNum() > 0 && !Ops::NN::CheckIsUnknownDimNum(*bias)) {
            const size_t biasRank = bias->GetDimNum();
            const bool validRank = biasRank == 1 || (num_dim == 3 && biasRank == 3);
            if (!validRank) {
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(opName, "bias", FormatString("%zuD", biasRank).c_str(),
                                                         "SwiGLU bias must be [N], or [B,1,N] for a rank-3 output");
                return ge::GRAPH_FAILED;
            }
            const auto matches = [](int64_t a, int64_t b) { return a == UNKNOWN_DIM || b == UNKNOWN_DIM || a == b; };
            if (!matches(bias->GetDim(biasRank - 1), N) ||
                (biasRank == 3 &&
                 (!matches(bias->GetDim(0), out_shape_y->GetDim(0)) || !matches(bias->GetDim(1), 1)))) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "bias", Ops::Base::ToString(*bias).c_str(),
                                                      "SwiGLU bias dimensions must match the pre-split matmul output");
                return ge::GRAPH_FAILED;
            }
        }
    }

    OP_LOGD(opName, "Output y shape: %s", Ops::Base::ToString(*out_shape_y).c_str());

    int64_t scale_n = UNKNOWN_DIM;
    if (outputN != UNKNOWN_DIM && outputN != UNKNOWN_DIM_NUM) {
        constexpr int64_t SCALE_STORAGE_GROUP = MX_BLOCK_SIZE * ALIGN_NUM;
        scale_n = outputN / SCALE_STORAGE_GROUP + (outputN % SCALE_STORAGE_GROUP != 0);
    }

    size_t y_scale_dim = num_dim + 1;
    out_shape_y_scale->SetDimNum(y_scale_dim);

    for (size_t i = 0; i < num_dim - 2; ++i) {
        out_shape_y_scale->SetDim(i, out_shape_y->GetDim(i));
    }

    out_shape_y_scale->SetDim(num_dim - 2, M);
    out_shape_y_scale->SetDim(num_dim - 1, scale_n);
    out_shape_y_scale->SetDim(num_dim, ALIGN_NUM);

    OP_LOGD(opName, "Output yScale shape: %s", Ops::Base::ToString(*out_shape_y_scale).c_str());

    OP_LOGI(context, "End InferShape4QuantMatmulActivationQuant.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4QuantMatmulActivationQuant(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin InferDataType4QuantMatmulActivationQuant.");

    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const char* opName = GetValidOpNameForInferDataType(context);
    if (!CheckSwiGluAttrs(context, opName)) {
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    int64_t y_dtype = -1;
    if (attrs != nullptr) {
        const int64_t* y_dtype_ptr = attrs->GetAttrPointer<int64_t>(Y_DTYPE_INDEX);
        if (y_dtype_ptr != nullptr) {
            y_dtype = *y_dtype_ptr;
        }
    }

    if (y_dtype < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, "y_dtype", "missing", "y_dtype must be specified");
        return ge::GRAPH_FAILED;
    }
    const auto* activation = attrs->GetAttrPointer<char>(ACTIVATION_TYPE_INDEX);
    if (activation != nullptr && std::string(activation) == "swiglu" &&
        y_dtype != static_cast<int64_t>(context->GetInputDataType(X1_INDEX))) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName, "y_dtype, x1 dtype",
            FormatString("%lld, %s", static_cast<long long>(y_dtype),
                         ge::TypeUtils::DataTypeToSerialString(context->GetInputDataType(X1_INDEX)).c_str())
                .c_str(),
            "SwiGLU requires y_dtype to match x1");
        return ge::GRAPH_FAILED;
    }

    context->SetOutputDataType(Y_INDEX, static_cast<ge::DataType>(y_dtype));
    context->SetOutputDataType(Y_SCALE_INDEX, ge::DT_FLOAT8_E8M0);

    OP_LOGD(context, "End InferDataType4QuantMatmulActivationQuant.");
    return ge::GRAPH_SUCCESS;
}
} // namespace

namespace Ops::NN::MatMul {
IMPL_OP_INFERSHAPE(QuantMatmulActivationQuant)
    .InferShape(InferShape4QuantMatmulActivationQuant)
    .InferDataType(InferDataType4QuantMatmulActivationQuant);
}
