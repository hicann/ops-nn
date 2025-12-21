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
 * \file rms_norm_quant_infershape.cpp
 * \brief
 */

#include <cstddef>
#include <cstdint>
#include <exception>
#include "op_common/log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {

namespace {
class NullPtrException : public std::exception {
public:
    explicit NullPtrException(const char* message = "Null pointer exception occurred") : msg_(message)
    {}
    const char* what() const noexcept override
    {
        return msg_;
    }

private:
    const char* msg_;
};

size_t GetInputDimNum(size_t inputIdx, const gert::InferShapeContext* context)
{
    const auto* shape = context->GetInputShape(inputIdx);
    if (shape == nullptr) {
        throw NullPtrException("shape is nullptr");
    }
    return shape->GetDimNum();
}

size_t GetInputDim(size_t inputIdx, size_t dimIdx, const gert::InferShapeContext* context)
{
    const auto* shape = context->GetInputShape(inputIdx);
    if (shape == nullptr) {
        throw NullPtrException("shape is nullptr");
    }
    return shape->GetDim(dimIdx);
}

void SetOutputDimNum(size_t outputIdx, size_t dimNum, gert::InferShapeContext* context)
{
    auto* shape = context->GetOutputShape(outputIdx);
    if (shape == nullptr) {
        throw NullPtrException("shape is nullptr");
    }
    shape->SetDimNum(dimNum);
}

void SetOutputDim(size_t outputIdx, size_t dimIdx, int64_t value, gert::InferShapeContext* context)
{
    auto* shape = context->GetOutputShape(outputIdx);
    if (shape == nullptr) {
        throw NullPtrException("shape is nullptr");
    }
    shape->SetDim(dimIdx, value);
}

void CopyInputDimsToOutput(size_t inputIdx, size_t outputIdx, gert::InferShapeContext* context)
{
    size_t dimNum = GetInputDimNum(inputIdx, context);
    SetOutputDimNum(outputIdx, dimNum, context);
    for (size_t dimIdx = 0; dimIdx < dimNum; dimIdx++) {
        SetOutputDim(outputIdx, dimIdx, GetInputDim(inputIdx, dimIdx, context), context);
    }
}

} // namespace

constexpr size_t INPUT_X = 0;
constexpr size_t INPUT_GAMMA = 1;
constexpr size_t INPUT_BETA = 2;

constexpr size_t OUTPUT_Y = 0;
constexpr size_t OUTPUT_RES_OUT = 1;
constexpr size_t ATTR_DST_TYPE = 3;

static ge::graphStatus InferShape4RmsNormQuant(gert::InferShapeContext* context)
{
    bool EN_QUANT = true;
    bool EN_PRE_POST = false;
    if (context == nullptr) {
        OP_LOGE("InferShape4RmsNormQuant", "Context is nullptr, check failed.");
        return ge::GRAPH_FAILED;
    }

    std::string OP_NAME = "RmsNorm";
    if (EN_QUANT) {
        OP_NAME = OP_NAME + "Quant";
    }
    if (EN_PRE_POST) {
        OP_NAME = "PrePost" + OP_NAME;
    }

    OP_LOGD(OP_NAME, "Begin to do NormInferShape");

    if (GetInputDimNum(INPUT_BETA, context) != 0) {
        if (GetInputDimNum(INPUT_GAMMA, context) != GetInputDimNum(INPUT_BETA, context)) {
            OP_LOGE(OP_NAME, "gamma_shape are not equal to beta_shape");
            return ge::GRAPH_FAILED;
        }

        for (size_t i = 0; i < GetInputDimNum(INPUT_GAMMA, context); i++) {
            if (GetInputDim(INPUT_GAMMA, i, context) != GetInputDim(INPUT_BETA, i, context)) {
                OP_LOGE(OP_NAME, "gamma_shape dim are not equal to beta_shape dim.");
                return ge::GRAPH_FAILED;
            }
        }
    }

    if (GetInputDim(INPUT_GAMMA, GetInputDimNum(INPUT_GAMMA, context) - 1, context) !=
        GetInputDim(INPUT_X, GetInputDimNum(INPUT_X, context) - 1, context)) {
        OP_LOGE(OP_NAME, "gamma_shape last dim are not equal to x_shape last dim.");
        return ge::GRAPH_FAILED;
    }

    if (EN_PRE_POST) {
        constexpr uint32_t INPUT_RES_IN = 3;
        if (GetInputDimNum(INPUT_X, context) != GetInputDimNum(INPUT_RES_IN, context)) {
            OP_LOGE(OP_NAME, "x shape are not equal to resIn shape");
            return ge::GRAPH_FAILED;
        }
        for (size_t i = 0; i < GetInputDimNum(INPUT_X, context); i++) {
            if (GetInputDim(INPUT_X, i, context) != GetInputDim(INPUT_RES_IN, i, context)) {
                OP_LOGE(OP_NAME, "x dimension are not equal to resIn dimension.");
                return ge::GRAPH_FAILED;
            }
        }
        CopyInputDimsToOutput(INPUT_RES_IN, OUTPUT_RES_OUT, context);
    }

    CopyInputDimsToOutput(INPUT_X, OUTPUT_Y, context);
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType4RmsNormQuant(gert::InferDataTypeContext* context)
{
    bool EN_QUANT = true;
    bool EN_PRE_POST = false;
    ge::DataType y_dtype = DT_INT8;
    auto dstTypePtr = context->GetAttrs()->GetInt(ATTR_DST_TYPE);
    if (dstTypePtr != nullptr) {
        y_dtype = static_cast<ge::DataType>(*dstTypePtr);
    }
    context->SetOutputDataType(OUTPUT_Y, y_dtype);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RmsNormQuant).InferShape(InferShape4RmsNormQuant).InferDataType(InferDataType4RmsNormQuant);
} // namespace ops