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
 * \file rotate_quant_infershape.cpp
 * \brief RotateQuant算子形状推导实现
 */

#include "register/op_impl_registry.h"
#include "error_util.h"
#include "log/log.h"
#include "util/math_util.h"
#include "platform/platform_info.h"

namespace ops {

constexpr int64_t X_INDEX = 0;
constexpr int64_t ROT_INDEX = 1;
constexpr int64_t ALPHA_INDEX = 2;
constexpr int64_t Y_INDEX = 0;
constexpr int64_t SCALE_INDEX = 1;

constexpr int32_t DIMENSION_2 = 2;
constexpr int32_t MIN_X_DIM_NUM = 1;
constexpr int32_t MAX_X_DIM_NUM = 7;
constexpr int32_t MIN_ROT_DIM_NUM = 2;
constexpr int32_t MAX_ROT_DIM_NUM = 3;

constexpr size_t ATTR_INDEX_DST_TYPE = 0;
constexpr uint32_t OUTPUT_NUM_ROTATE_QUANT = 2;
constexpr uint32_t INPUT_NUM_MIN_ROTATE_QUANT = 2;
constexpr uint32_t INPUT_NUM_MAX_ROTATE_QUANT = 3;

constexpr int64_t MX_BLOCK_SIZE = 64;
constexpr int64_t MX_SCALE_LAST_DIM = 2;
constexpr int64_t ALPHA_DIM_NUM = 1;
constexpr int64_t ALPHA_DIM_SIZE = 1;

const std::initializer_list<ge::DataType> INT_OUT_TYPE_LIST = {ge::DT_INT8, ge::DT_INT4};

static std::set<std::string> RotateQuant950SupportSoc = {"Ascend950"};

static bool IsRotateQuant950Support()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    auto ret = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
    return (ret == ge::GRAPH_SUCCESS && RotateQuant950SupportSoc.count(platformInfo.str_info.short_soc_version) > 0);
}

static ge::graphStatus CheckComputeNodeNum(gert::InferShapeContext* context)
{
    size_t inputNum = context->GetComputeNodeInputNum();
    if (inputNum < INPUT_NUM_MIN_ROTATE_QUANT || inputNum > INPUT_NUM_MAX_ROTATE_QUANT) {
        OP_LOGE(
            context, "Input num must be in [%u, %u], but got %zu", INPUT_NUM_MIN_ROTATE_QUANT,
            INPUT_NUM_MAX_ROTATE_QUANT, inputNum);
        return ge::GRAPH_FAILED;
    }

    if (context->GetComputeNodeOutputNum() != OUTPUT_NUM_ROTATE_QUANT) {
        OP_LOGE(
            context, "Output num must be %u, but got %zu", OUTPUT_NUM_ROTATE_QUANT, context->GetComputeNodeOutputNum());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeCheck(gert::InferShapeContext* context)
{
    if (context == nullptr || CheckComputeNodeNum(context) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    for (uint32_t i = 0; i < OUTPUT_NUM_ROTATE_QUANT; ++i) {
        if (context->GetOutputShape(i) == nullptr) {
            OP_LOGE(context, "Output shape %u is nullptr", i);
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAlphaShape(gert::InferShapeContext* context)
{
    size_t inputNum = context->GetComputeNodeInputNum();
    if (inputNum > ALPHA_INDEX) {
        const gert::Shape* alphaShape = context->GetInputShape(ALPHA_INDEX);
        if (alphaShape != nullptr) {
            if (alphaShape->GetDimNum() != ALPHA_DIM_NUM || alphaShape->GetDim(0) != ALPHA_DIM_SIZE) {
                OP_LOGE(context, "alpha shape must be (1), but got %zu dims", alphaShape->GetDimNum());
                return ge::GRAPH_FAILED;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputShape4Int(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(X_INDEX);
    const gert::Shape* rotShape = context->GetInputShape(ROT_INDEX);

    if (xShape == nullptr || rotShape == nullptr) {
        OP_LOGE(context, "Input shape is nullptr");
        return ge::GRAPH_FAILED;
    }

    size_t xDimNum = xShape->GetDimNum();
    if (xDimNum != DIMENSION_2) {
        OP_LOGE(context, "x shape dimension must be 2, but got %zu", xDimNum);
        return ge::GRAPH_FAILED;
    }

    size_t rotDimNum = rotShape->GetDimNum();
    if (rotDimNum != DIMENSION_2) {
        OP_LOGE(context, "rot shape dimension must be 2, but got %zu", rotDimNum);
        return ge::GRAPH_FAILED;
    }

    int64_t xN = xShape->GetDim(1);
    int64_t rotK = rotShape->GetDim(0);
    int64_t rotN = rotShape->GetDim(1);
    if (rotK != rotN) {
        OP_LOGE(context, "rot must be a square matrix, but got shape [%ld, %ld]", rotK, rotN);
        return ge::GRAPH_FAILED;
    }

    if (rotK == 0) {
        OP_LOGE(context, "rot K dimension cannot be zero");
        return ge::GRAPH_FAILED;
    }
    if (xN % rotK != 0) {
        OP_LOGE(context, "x N dimension (%ld) must be divisible by rot K dimension (%ld)", xN, rotK);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputShape4Mx(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(X_INDEX);
    const gert::Shape* rotShape = context->GetInputShape(ROT_INDEX);

    if (xShape == nullptr || rotShape == nullptr) {
        OP_LOGE(context, "Input shape is nullptr");
        return ge::GRAPH_FAILED;
    }

    size_t xDimNum = xShape->GetDimNum();
    if (xDimNum < MIN_X_DIM_NUM || xDimNum > MAX_X_DIM_NUM) {
        OP_LOGE(
            context, "For mxfp8, x shape dimension must be in [%d, %d], but got %zu", MIN_X_DIM_NUM, MAX_X_DIM_NUM,
            xDimNum);
        return ge::GRAPH_FAILED;
    }

    size_t rotDimNum = rotShape->GetDimNum();
    if (rotDimNum < MIN_ROT_DIM_NUM || rotDimNum > MAX_ROT_DIM_NUM) {
        OP_LOGE(
            context, "For mxfp8, rot shape dimension must be in [%d, %d], but got %zu", MIN_ROT_DIM_NUM,
            MAX_ROT_DIM_NUM, rotDimNum);
        return ge::GRAPH_FAILED;
    }

    int64_t xN = xShape->GetDim(xDimNum - 1);
    int64_t rotLastDim = rotShape->GetDim(rotDimNum - 1);
    int64_t rotSecondLastDim = rotShape->GetDim(rotDimNum - 2);

    if (rotSecondLastDim != rotLastDim) {
        OP_LOGE(context, "rot last two dimensions must be equal, but got [%ld, %ld]", rotSecondLastDim, rotLastDim);
        return ge::GRAPH_FAILED;
    }

    if (rotLastDim == 0) {
        OP_LOGE(context, "rot K dimension cannot be zero");
        return ge::GRAPH_FAILED;
    }
    if (xN % rotLastDim != 0) {
        OP_LOGE(context, "x last dimension (%ld) must be divisible by rot K dimension (%ld)", xN, rotLastDim);
        return ge::GRAPH_FAILED;
    }

    if (rotDimNum == MAX_ROT_DIM_NUM) {
        int64_t nKRatio = xN / rotLastDim;
        int64_t rotFirstDim = rotShape->GetDim(0);
        if (rotFirstDim != nKRatio) {
            OP_LOGE(context, "For 3D rot, first dimension (%ld) must equal N/K (%ld)", rotFirstDim, nKRatio);
            return ge::GRAPH_FAILED;
        }
    }

    if (CheckAlphaShape(context) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4Int(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(X_INDEX);
    gert::Shape* yShape = context->GetOutputShape(Y_INDEX);
    gert::Shape* scaleShape = context->GetOutputShape(SCALE_INDEX);

    yShape->SetDimNum(DIMENSION_2);
    yShape->SetDim(0, xShape->GetDim(0));
    yShape->SetDim(1, xShape->GetDim(1));
    scaleShape->SetDimNum(1);
    scaleShape->SetDim(0, xShape->GetDim(0));
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4Mx(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(X_INDEX);
    gert::Shape* yShape = context->GetOutputShape(Y_INDEX);
    gert::Shape* scaleShape = context->GetOutputShape(SCALE_INDEX);

    size_t xDimNum = xShape->GetDimNum();
    int64_t xN = xShape->GetDim(xDimNum - 1);

    yShape->SetDimNum(xDimNum);
    for (size_t i = 0; i < xDimNum; ++i) {
        yShape->SetDim(i, xShape->GetDim(i));
    }

    scaleShape->SetDimNum(xDimNum + 1);
    for (size_t i = 0; i < xDimNum - 1; ++i) {
        scaleShape->SetDim(i, xShape->GetDim(i));
    }
    scaleShape->SetDim(xDimNum - 1, Ops::Base::CeilDiv(xN, MX_BLOCK_SIZE));
    scaleShape->SetDim(xDimNum, MX_SCALE_LAST_DIM);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RotateQuantInferShape(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape RotateQuant");
    if (InferShapeCheck(context) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    ge::DataType yDtype = ge::DT_INT8;
    auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int32_t* pDstDtype = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_DST_TYPE);
        if (pDstDtype != nullptr) {
            yDtype = static_cast<ge::DataType>(*pDstDtype);
        }
    }

    if (IsRotateQuant950Support()) {
        if (yDtype == ge::DT_FLOAT4_E2M1 || yDtype == ge::DT_FLOAT8_E4M3FN || yDtype == ge::DT_FLOAT8_E5M2) {
            OP_CHECK_IF(
                CheckInputShape4Mx(context) == ge::GRAPH_FAILED, OP_LOGE(context, "Invalid input shape for mxfp8"),
                return ge::GRAPH_FAILED);
            return InferShape4Mx(context);
        } else {
            OP_LOGE(
                context, "For ascend950, y_dtype must be float4_e2m1, float8_e4m3fn, or float8_e5m2, but got %d",
                static_cast<int32_t>(yDtype));
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_CHECK_IF(
            std::find(INT_OUT_TYPE_LIST.begin(), INT_OUT_TYPE_LIST.end(), yDtype) == INT_OUT_TYPE_LIST.end(),
            OP_LOGE(context, "For ascend910b, y_dtype only support int4, int8"), return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            CheckInputShape4Int(context) == ge::GRAPH_FAILED, OP_LOGE(context, "Invalid input shape for int"),
            return ge::GRAPH_FAILED);
        return InferShape4Int(context);
    }
}

static ge::graphStatus RotateQuantInferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "RotateQuantInferDataType begin");

    ge::DataType yDtype = ge::DT_INT8;
    auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int32_t* pDstDtype = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_DST_TYPE);
        if (pDstDtype != nullptr) {
            yDtype = static_cast<ge::DataType>(*pDstDtype);
        }
    }

    if (IsRotateQuant950Support()) {
        if (yDtype == ge::DT_FLOAT4_E2M1) {
            context->SetOutputDataType(Y_INDEX, ge::DT_FLOAT4_E2M1);
            context->SetOutputDataType(SCALE_INDEX, ge::DT_FLOAT8_E8M0);
        } else if (yDtype == ge::DT_FLOAT8_E4M3FN || yDtype == ge::DT_FLOAT8_E5M2) {
            context->SetOutputDataType(Y_INDEX, yDtype);
            context->SetOutputDataType(SCALE_INDEX, ge::DT_FLOAT8_E8M0);
        } else {
            OP_LOGE(context, "For ascend950, y_dtype must be float4_e2m1, float8_e4m3fn, or float8_e5m2");
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_CHECK_IF(
            std::find(INT_OUT_TYPE_LIST.begin(), INT_OUT_TYPE_LIST.end(), yDtype) == INT_OUT_TYPE_LIST.end(),
            OP_LOGE(context, "For ascend910b, y_dtype only support int4, int8"), return ge::GRAPH_FAILED);
        context->SetOutputDataType(Y_INDEX, yDtype);
        context->SetOutputDataType(SCALE_INDEX, ge::DT_FLOAT);
    }

    OP_LOGD(context, "RotateQuantInferDataType end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RotateQuant).InferShape(RotateQuantInferShape).InferDataType(RotateQuantInferDataType);

} // namespace ops