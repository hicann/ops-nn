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

#include "../../../common/inc/op_host/op_const_def_nn.h"
#include "matmul/common/op_host/log_format_util.h"
#include "matmul/common/op_host/matmul_error_report.h"
#include "register/op_impl_registry.h"
#include "error_util.h"
#include "log/log.h"
#include "util/math_util.h"
#include "platform/platform_info.h"
#include "op_common/op_host/util/op_const_def.h"
#include "acl/acl_platform.h"

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
constexpr uint32_t INPUT_NUM_MAX_ROTATE_QUANT = 4;

constexpr int64_t MX_BLOCK_SIZE = 64;
constexpr int64_t MX_SCALE_LAST_DIM = 2;
constexpr int64_t HIF4_SCALE_LAST_DIM = 1;
constexpr int64_t ALPHA_DIM_NUM = 1;
constexpr int64_t ALPHA_DIM_SIZE = 1;

const std::initializer_list<ge::DataType> INT_OUT_TYPE_LIST = {ge::DT_INT8, ge::DT_INT4};

static bool IsRotateQuantDav3510Support()
{
    char buf[64] = {0};
    aclError ret = aclplatformGetDeviceInfo(ACL_PLATFORM_NPU_ARCH, buf, sizeof(buf));
    if (ret != ACL_SUCCESS) {
        OP_LOGE("IsRotateQuantDav3510Support", "get npu arch failed.");
        return false;
    }
    return (std::atoi(buf) == static_cast<int32_t>(Ops::Base::DAV_3510) ||
            std::atoi(buf) == static_cast<int32_t>(Ops::Base::DAV_9201) ||
            std::atoi(buf) == static_cast<int32_t>(Ops::Base::DAV_9202));
}

static bool IsGroupedRotateQuantSupport()
{
    fe::PlatformInfo platformInfo;
    fe::OptionalInfo optionalInfo;
    auto ret = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo);
    return ret == ge::GRAPH_SUCCESS && platformInfo.str_info.short_soc_version == "Ascend950";
}

static ge::graphStatus CheckComputeNodeNum(gert::InferShapeContext* context)
{
    size_t inputNum = context->GetComputeNodeInputNum();
    if (inputNum < INPUT_NUM_MIN_ROTATE_QUANT || inputNum > INPUT_NUM_MAX_ROTATE_QUANT) {
        OP_LOGE_FOR_INVALID_TENSORNUMS_WITH_REASON(
            context->GetNodeName(), "x, rotation, alpha, group_list", Ops::NN::FormatString("%zu", inputNum).c_str(),
            "Provide x and rotation, with optional alpha and group_list (2 to 4 inputs).");
        return ge::GRAPH_FAILED;
    }

    if (context->GetComputeNodeOutputNum() != OUTPUT_NUM_ROTATE_QUANT) {
        OP_LOGE_FOR_INVALID_TENSORNUM(context->GetNodeName(), "y, scale", context->GetComputeNodeOutputNum(), "2");
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
            Ops::NN::ReportMatmulInferShapeError(
                context->GetNodeName(),
                Ops::NN::FormatString(
                    "Output shape storage at index %u is null; check InferShape output initialization.", i)
                    .c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAlphaShape(gert::InferShapeContext* context)
{
    const gert::Shape* alphaShape = context->GetOptionalInputShape(ALPHA_INDEX);
    if (alphaShape != nullptr) {
        if (alphaShape->GetDimNum() != ALPHA_DIM_NUM || alphaShape->GetDim(0) != ALPHA_DIM_SIZE) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "alpha",
                                                  Ops::Base::ToString(*alphaShape).c_str(),
                                                  "alpha must have shape [1].");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputShape4Int(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(X_INDEX);
    const gert::Shape* rotShape = context->GetInputShape(ROT_INDEX);

    if (xShape == nullptr || rotShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), xShape == nullptr ? "x" : "rotation");
        return ge::GRAPH_FAILED;
    }

    size_t xDimNum = xShape->GetDimNum();
    if (xDimNum != DIMENSION_2) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x",
                                                 Ops::NN::FormatString("%zu", xDimNum).c_str(), "x must have rank 2.");
        return ge::GRAPH_FAILED;
    }

    size_t rotDimNum = rotShape->GetDimNum();
    if (rotDimNum != DIMENSION_2) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "rotation",
                                                 Ops::NN::FormatString("%zu", rotDimNum).c_str(),
                                                 "rotation must have rank 2.");
        return ge::GRAPH_FAILED;
    }

    int64_t xN = xShape->GetDim(1);
    int64_t rotK = rotShape->GetDim(0);
    int64_t rotN = rotShape->GetDim(1);
    if (rotK != rotN) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "rotation",
                                              Ops::Base::ToString(*rotShape).c_str(),
                                              "rotation must be a square matrix.");
        return ge::GRAPH_FAILED;
    }

    if (rotK == 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "rotation",
                                              Ops::Base::ToString(*rotShape).c_str(), "rotation K must be nonzero.");
        return ge::GRAPH_FAILED;
    }
    if (xN % rotK != 0) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, rotation",
                                               Ops::NN::FormatString("%s, %s", Ops::Base::ToString(*xShape).c_str(),
                                                                     Ops::Base::ToString(*rotShape).c_str())
                                                   .c_str(),
                                               "x N must be divisible by rotation K.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputShape4Mx(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(X_INDEX);
    const gert::Shape* rotShape = context->GetInputShape(ROT_INDEX);

    if (xShape == nullptr || rotShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), xShape == nullptr ? "x" : "rotation");
        return ge::GRAPH_FAILED;
    }

    size_t xDimNum = xShape->GetDimNum();
    if (xDimNum < MIN_X_DIM_NUM || xDimNum > MAX_X_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x",
                                                 Ops::NN::FormatString("%zu", xDimNum).c_str(),
                                                 "For MXFP output, x must have rank between 1 and 7.");
        return ge::GRAPH_FAILED;
    }

    size_t rotDimNum = rotShape->GetDimNum();
    const gert::Shape* groupShape = context->GetOptionalInputShape(3);
    const size_t minRotRank = groupShape != nullptr ? 3 : MIN_ROT_DIM_NUM;
    const size_t maxRotRank = groupShape != nullptr ? 4 : MAX_ROT_DIM_NUM;
    if (rotDimNum < minRotRank || rotDimNum > maxRotRank) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(), "rotation", Ops::NN::FormatString("%zu", rotDimNum).c_str(),
            Ops::NN::FormatString(
                "For MXFP output, rotation rank must be between %zu and %zu for the selected group mode.", minRotRank,
                maxRotRank)
                .c_str());
        return ge::GRAPH_FAILED;
    }

    int64_t xN = xShape->GetDim(xDimNum - 1);
    int64_t rotLastDim = rotShape->GetDim(rotDimNum - 1);
    int64_t rotSecondLastDim = rotShape->GetDim(rotDimNum - 2);

    if (rotSecondLastDim != rotLastDim) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "rotation",
                                              Ops::Base::ToString(*rotShape).c_str(),
                                              "The last two dimensions of rotation must be equal.");
        return ge::GRAPH_FAILED;
    }

    if (rotLastDim == 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "rotation",
                                              Ops::Base::ToString(*rotShape).c_str(), "rotation K must be nonzero.");
        return ge::GRAPH_FAILED;
    }
    if (xN % rotLastDim != 0) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, rotation",
                                               Ops::NN::FormatString("%s, %s", Ops::Base::ToString(*xShape).c_str(),
                                                                     Ops::Base::ToString(*rotShape).c_str())
                                                   .c_str(),
                                               "The last dimension of x must be divisible by rotation K.");
        return ge::GRAPH_FAILED;
    }

    if (groupShape != nullptr) {
        if (xDimNum != DIMENSION_2 || groupShape->GetDimNum() != 1 || groupShape->GetDim(0) < 1 ||
            groupShape->GetDim(0) > 1024 || rotShape->GetDim(0) != groupShape->GetDim(0) ||
            (rotDimNum == 4 && rotShape->GetDim(1) != xN / rotLastDim)) {
            OP_LOGE(context,
                    "Grouped mode requires x[M,N], rotation[E,K,K] or [E,N/K,K,K], group_list[E], 1 <= E <= 1024.");
            return ge::GRAPH_FAILED;
        }
        auto attrs = context->GetAttrs();
        if (attrs != nullptr && attrs->GetAttrNum() > 6) {
            auto type = attrs->GetAttrPointer<int64_t>(6);
            if (type == nullptr || (*type != 0 && *type != 1)) {
                OP_LOGE(context, "group_list_type must be 0 or 1.");
                return ge::GRAPH_FAILED;
            }
        }
    } else if (rotDimNum == MAX_ROT_DIM_NUM) {
        int64_t nKRatio = xN / rotLastDim;
        int64_t rotFirstDim = rotShape->GetDim(0);
        if (rotFirstDim != nKRatio) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context->GetNodeName(), "x, rotation",
                Ops::NN::FormatString("%s, %s", Ops::Base::ToString(*xShape).c_str(),
                                      Ops::Base::ToString(*rotShape).c_str())
                    .c_str(),
                "For a 3D rotation, its first dimension must equal x N divided by rotation K.");
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

    ge::DataType yDtype = ge::DT_INT8;
    auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* pDstDtype = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DST_TYPE);
        if (pDstDtype != nullptr) {
            yDtype = static_cast<ge::DataType>(*pDstDtype);
        }
    }
    int64_t scaleLastDim = (yDtype == Ops::NN::DT_HIFLOAT4) ? HIF4_SCALE_LAST_DIM : MX_SCALE_LAST_DIM;

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
    scaleShape->SetDim(xDimNum, scaleLastDim);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus RotateQuantInferShape(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        Ops::NN::ReportMatmulInferShapeError("RotateQuant",
                                             "InferShape context is null; check the shape inference invocation.");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferShape RotateQuant");
    if (InferShapeCheck(context) == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }
    if (context->GetOptionalInputShape(3) != nullptr && !IsGroupedRotateQuantSupport()) {
        OP_LOGE(context, "Grouped RotateQuant is only supported on Ascend950.");
        return ge::GRAPH_FAILED;
    }

    ge::DataType yDtype = ge::DT_INT8;
    auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* pDstDtype = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DST_TYPE);
        if (pDstDtype != nullptr) {
            yDtype = static_cast<ge::DataType>(*pDstDtype);
        }
    }

    if (IsRotateQuantDav3510Support()) {
        if (yDtype == ge::DT_FLOAT4_E2M1 || yDtype == ge::DT_FLOAT8_E4M3FN || yDtype == ge::DT_FLOAT8_E5M2 ||
            yDtype == Ops::NN::DT_HIFLOAT4) {
            OP_CHECK_IF(CheckInputShape4Mx(context) == ge::GRAPH_FAILED,
                        OP_LOGE_WITHOUT_REPORT(context, "Invalid input shape for mxfp8"), return ge::GRAPH_FAILED);
            return InferShape4Mx(context);
        } else {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context->GetNodeName(), "y_dtype", Ops::NN::FormatString("%d", static_cast<int32_t>(yDtype)).c_str(),
                "For this platform, supported output dtypes are FLOAT4_E2M1, FLOAT8_E4M3FN, FLOAT8_E5M2 and HIFLOAT4.");
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_CHECK_IF(
            std::find(INT_OUT_TYPE_LIST.begin(), INT_OUT_TYPE_LIST.end(), yDtype) == INT_OUT_TYPE_LIST.end(),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "y_dtype",
                                                  Ops::NN::FormatString("%d", static_cast<int32_t>(yDtype)).c_str(),
                                                  "For Ascend910B, supported output dtypes are INT4 and INT8."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(CheckInputShape4Int(context) == ge::GRAPH_FAILED,
                    OP_LOGE_WITHOUT_REPORT(context, "Invalid input shape for int"), return ge::GRAPH_FAILED);
        return InferShape4Int(context);
    }
}

static ge::graphStatus RotateQuantInferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        Ops::NN::ReportMatmulInferShapeError("RotateQuant",
                                             "InferDataType context is null; check the dtype inference invocation.");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "RotateQuantInferDataType begin");

    ge::DataType yDtype = ge::DT_INT8;
    auto* attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const int64_t* pDstDtype = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DST_TYPE);
        if (pDstDtype != nullptr) {
            yDtype = static_cast<ge::DataType>(*pDstDtype);
        }
    }

    if (IsRotateQuantDav3510Support()) {
        if (yDtype == ge::DT_FLOAT4_E2M1) {
            context->SetOutputDataType(Y_INDEX, ge::DT_FLOAT4_E2M1);
            context->SetOutputDataType(SCALE_INDEX, ge::DT_FLOAT8_E8M0);
        } else if (yDtype == ge::DT_FLOAT8_E4M3FN || yDtype == ge::DT_FLOAT8_E5M2) {
            context->SetOutputDataType(Y_INDEX, yDtype);
            context->SetOutputDataType(SCALE_INDEX, ge::DT_FLOAT8_E8M0);
        } else if (yDtype == Ops::NN::DT_HIFLOAT4) {
            context->SetOutputDataType(Y_INDEX, Ops::NN::DT_HIFLOAT4);
            context->SetOutputDataType(SCALE_INDEX, Ops::NN::DT_HIFLOAT4_SCALE);
        } else {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context->GetNodeName(), "y_dtype", Ops::NN::FormatString("%d", static_cast<int32_t>(yDtype)).c_str(),
                "For this platform, supported output dtypes are FLOAT4_E2M1, FLOAT8_E4M3FN, FLOAT8_E5M2 and HIFLOAT4.");
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_CHECK_IF(
            std::find(INT_OUT_TYPE_LIST.begin(), INT_OUT_TYPE_LIST.end(), yDtype) == INT_OUT_TYPE_LIST.end(),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "y_dtype",
                                                  Ops::NN::FormatString("%d", static_cast<int32_t>(yDtype)).c_str(),
                                                  "For Ascend910B, supported output dtypes are INT4 and INT8."),
            return ge::GRAPH_FAILED);
        context->SetOutputDataType(Y_INDEX, yDtype);
        context->SetOutputDataType(SCALE_INDEX, ge::DT_FLOAT);
    }

    OP_LOGD(context, "RotateQuantInferDataType end");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(RotateQuant).InferShape(RotateQuantInferShape).InferDataType(RotateQuantInferDataType);

} // namespace ops
