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
 * \file foreach_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "error_util.h"
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"
#include "runtime/storage_shape.h"
#include "util/shape_util.h"
#include "common_dtype.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShape4ForeachCommon(gert::InferShapeContext* context)
{
    uint32_t outputNumInferShape = context->GetComputeNodeOutputNum();
    const auto inputInfoInferShape = context->GetIrInputInstanceInfo(0);
    if (inputInfoInferShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    std::string errMsg = optiling::ConcatString("num of dynamic input0 ", inputInfoInferShape->GetInstanceNum(),
                                                " is not equal to num of dynamic output0 ", outputNumInferShape);
    OP_CHECK_IF(inputInfoInferShape->GetInstanceNum() != outputNumInferShape, ([&]() {
                    const std::string e9Op = (context == nullptr || context->GetNodeName() == nullptr ?
                                                  "Foreach" :
                                                  context->GetNodeName());
                    const std::string e9Parameter = "x";
                    const std::string e9Size = std::to_string(inputInfoInferShape->GetInstanceNum());
                    const std::string e9Reason = (std::string(errMsg.c_str()));
                    OP_LOGE_WITHOUT_REPORT(e9Op.c_str(), "Parameter %s has invalid list size %s: %s.",
                                           e9Parameter.c_str(), e9Size.c_str(), e9Reason.c_str());
                    const std::vector<const char*> e9Keys = {"param_name", "op_name", "incorrect_size", "reason"};
                    const std::vector<const char*> e9Values = {e9Parameter.c_str(), e9Op.c_str(), e9Size.c_str(),
                                                               e9Reason.c_str()};
                    REPORT_PREDEFINED_ERR_MSG("EZ0038", e9Keys, e9Values);
                }()),
                return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < inputInfoInferShape->GetInstanceNum(); i++) {
        auto xShape = context->GetDynamicInputShape(0, i);
        auto yShape = context->GetOutputShape(i);
        if ((xShape == nullptr) || (yShape == nullptr)) {
            return ge::GRAPH_FAILED;
        }
        *yShape = *xShape;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4ForeachCommon(gert::InferDataTypeContext* context)
{
    uint32_t outputNumDataType = context->GetComputeNodeOutputNum();
    const auto inputInfoDataType = context->GetIrInputInstanceInfo(0);
    if (inputInfoDataType == nullptr) {
        return ge::GRAPH_FAILED;
    }

    std::string errMsg = optiling::ConcatString("num of dynamic input0 ", inputInfoDataType->GetInstanceNum(),
                                                " is not equal to num of dynamic output0 ", outputNumDataType);
    OP_CHECK_IF(inputInfoDataType->GetInstanceNum() != outputNumDataType, ([&]() {
                    const std::string e9Op = (context == nullptr || context->GetNodeName() == nullptr ?
                                                  "Foreach" :
                                                  context->GetNodeName());
                    const std::string e9Parameter = "x";
                    const std::string e9Size = std::to_string(inputInfoDataType->GetInstanceNum());
                    const std::string e9Reason = (std::string(errMsg.c_str()));
                    OP_LOGE_WITHOUT_REPORT(e9Op.c_str(), "Parameter %s has invalid list size %s: %s.",
                                           e9Parameter.c_str(), e9Size.c_str(), e9Reason.c_str());
                    const std::vector<const char*> e9Keys = {"param_name", "op_name", "incorrect_size", "reason"};
                    const std::vector<const char*> e9Values = {e9Parameter.c_str(), e9Op.c_str(), e9Size.c_str(),
                                                               e9Reason.c_str()};
                    REPORT_PREDEFINED_ERR_MSG("EZ0038", e9Keys, e9Values);
                }()),
                return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < inputInfoDataType->GetInstanceNum(); i++) {
        auto xDtype = context->GetDynamicInputDataType(0, i);

        context->SetOutputDataType(i, xDtype);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4ForeachInplace(gert::InferShapeContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

// 整进浮出: 整数输入(int16/int8/uint8)推导输出 DT_FLOAT, 浮点输入输出同 dtype
static ge::graphStatus InferDataType4ForeachIntToFloat(gert::InferDataTypeContext* context)
{
    uint32_t outputNumDataType = context->GetComputeNodeOutputNum();
    const auto inputInfoDataType = context->GetIrInputInstanceInfo(0);
    if (inputInfoDataType == nullptr) {
        return ge::GRAPH_FAILED;
    }

    std::string errMsg = optiling::ConcatString("num of dynamic input0 ", inputInfoDataType->GetInstanceNum(),
                                                "not equal num of dynamic output0 ", outputNumDataType);
    OP_CHECK_IF(inputInfoDataType->GetInstanceNum() != outputNumDataType,
                OP_LOGE(context->GetNodeName(), "%s", errMsg.c_str()), return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < inputInfoDataType->GetInstanceNum(); i++) {
        auto xDtype = context->GetDynamicInputDataType(0, i);
        if (xDtype == ge::DT_INT16 || xDtype == ge::DT_INT8 || xDtype == ge::DT_UINT8) {
            context->SetOutputDataType(i, ge::DT_FLOAT);
        } else {
            context->SetOutputDataType(i, xDtype);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4ForeachInplace(gert::InferDataTypeContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

constexpr size_t FOREACH_ZERO_INPLACE_MAX_RANK = 8;

bool IsForeachZeroInplaceRankSupported(const gert::Shape& shape)
{
    return Ops::Base::IsUnknownRank(shape) || shape.GetDimNum() <= FOREACH_ZERO_INPLACE_MAX_RANK;
}

bool IsForeachZeroInplaceDtypeMatched(ge::DataType expected, ge::DataType actual) { return expected == actual; }

static bool IsForeachZeroInplaceDtypeSupported(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT16 || dtype == ge::DT_FLOAT || dtype == ge::DT_BF16 || dtype == ge::DT_INT16 ||
           dtype == ge::DT_INT32;
}

static ge::graphStatus InferShape4ForeachZeroInplace(gert::InferShapeContext* context)
{
    const auto inputInfo = context->GetIrInputInstanceInfo(0);
    OP_CHECK_IF(inputInfo == nullptr || inputInfo->GetInstanceNum() == 0,
                OP_LOGE(context->GetNodeName(), "ForeachZeroInplace requires at least one input tensor"),
                return ge::GRAPH_FAILED);

    for (uint32_t i = 0; i < inputInfo->GetInstanceNum(); ++i) {
        const auto xShape = context->GetDynamicInputShape(0, i);
        OP_CHECK_IF(xShape == nullptr, OP_LOGE(context->GetNodeName(), "x[%u] shape is null", i),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(!IsForeachZeroInplaceRankSupported(*xShape),
                    OP_LOGE(context->GetNodeName(), "x[%u] rank must not exceed %zu", i, FOREACH_ZERO_INPLACE_MAX_RANK),
                    return ge::GRAPH_FAILED);
    }
    return InferShape4ForeachCommon(context);
}

static ge::graphStatus InferDataType4ForeachZeroInplace(gert::InferDataTypeContext* context)
{
    const auto inputInfo = context->GetIrInputInstanceInfo(0);
    OP_CHECK_IF(inputInfo == nullptr || inputInfo->GetInstanceNum() == 0,
                OP_LOGE(context->GetNodeName(), "ForeachZeroInplace requires at least one input tensor"),
                return ge::GRAPH_FAILED);

    const ge::DataType dtype = context->GetDynamicInputDataType(0, 0);
    OP_CHECK_IF(
        !IsForeachZeroInplaceDtypeSupported(dtype),
        OP_LOGE(context->GetNodeName(), "ForeachZeroInplace does not support dtype %d", static_cast<int32_t>(dtype)),
        return ge::GRAPH_FAILED);
    for (uint32_t i = 1; i < inputInfo->GetInstanceNum(); ++i) {
        OP_CHECK_IF(!IsForeachZeroInplaceDtypeMatched(dtype, context->GetDynamicInputDataType(0, i)),
                    OP_LOGE(context->GetNodeName(), "x tensor dtypes must match"), return ge::GRAPH_FAILED);
    }
    return InferDataType4ForeachCommon(context);
}

// In-place foreach ops whose in-place parameter is mirrored by an IR output of the same name (ref
// pairing, which is what lets the generated aclnn write results back to a non-contiguous parameter)
// do have an output to infer, and it is always identical to the input list.
static ge::graphStatus InferShape4ForeachInplaceRef(gert::InferShapeContext* context)
{
    return InferShape4ForeachCommon(context);
}

static ge::graphStatus InferDataType4ForeachInplaceRef(gert::InferDataTypeContext* context)
{
    return InferDataType4ForeachCommon(context);
}

IMPL_OP_INFERSHAPE(ForeachAbs)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachAcos)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachAddList)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachBinaryOp)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachAddScalarList)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachAddScalar)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachAddcdivList)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachAddcdivScalar)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachAddcdivScalarList)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachAddcmulList)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachAddcmulScalar)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachAddcmulScalarList)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachAsin)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachAtan)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachCopy)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachCos)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachCosh)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachDivList)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachDivScalar)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachDivScalarList)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachErf)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachMaximumScalarList)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachMaximumScalar)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachMaximumList)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachMinimumScalarList)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachMinimumScalar)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachMinimumList)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachMulScalarList)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachMulScalar)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachMulList)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachNeg)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachPowList)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachPowScalar)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachPowScalarList)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachSign)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachSin)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachSqrt)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachTan)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachTanh)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachZeroInplace)
    .InferShape(ops::InferShape4ForeachZeroInplace)
    .InferDataType(ops::InferDataType4ForeachZeroInplace);

IMPL_OP_INFERSHAPE(ForeachMulScalarInplace)
    .InferShape(ops::InferShape4ForeachInplaceRef)
    .InferDataType(ops::InferDataType4ForeachInplaceRef);

IMPL_OP_INFERSHAPE(ForeachSubScalarInplace)
    .InferShape(ops::InferShape4ForeachInplaceRef)
    .InferDataType(ops::InferDataType4ForeachInplaceRef);

IMPL_OP_INFERSHAPE(ForeachMulListInplace)
    .InferShape(ops::InferShape4ForeachInplaceRef)
    .InferDataType(ops::InferDataType4ForeachInplaceRef);

IMPL_OP_INFERSHAPE(ForeachDivListInplace)
    .InferShape(ops::InferShape4ForeachInplaceRef)
    .InferDataType(ops::InferDataType4ForeachInplaceRef);

IMPL_OP_INFERSHAPE(ForeachAddListInplace)
    .InferShape(ops::InferShape4ForeachInplaceRef)
    .InferDataType(ops::InferDataType4ForeachInplaceRef);

IMPL_OP_INFERSHAPE(ForeachSubListInplace)
    .InferShape(ops::InferShape4ForeachInplaceRef)
    .InferDataType(ops::InferDataType4ForeachInplaceRef);

IMPL_OP_INFERSHAPE(ForeachACosInplace)
    .InferShape(ops::InferShape4ForeachInplaceRef)
    .InferDataType(ops::InferDataType4ForeachInplaceRef);

IMPL_OP_INFERSHAPE(ForeachLogInplace)
    .InferShape(ops::InferShape4ForeachInplaceRef)
    .InferDataType(ops::InferDataType4ForeachInplaceRef);

IMPL_OP_INFERSHAPE(ForeachNonFiniteCheckAndUnscale)
    .InferShape(ops::InferShape4ForeachInplace)
    .InferDataType(ops::InferDataType4ForeachInplace);

IMPL_OP_INFERSHAPE(ForeachSigmoid).InferShape(InferShape4ForeachCommon).InferDataType(InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachSinh).InferShape(InferShape4ForeachCommon).InferDataType(InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachSubList).InferShape(InferShape4ForeachCommon).InferDataType(InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachSubScalarList)
    .InferShape(InferShape4ForeachCommon)
    .InferDataType(InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachSubScalar).InferShape(InferShape4ForeachCommon).InferDataType(InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachReciprocal)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachRoundOffNumber)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachErfc)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachExp)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachIntToFloat);

IMPL_OP_INFERSHAPE(ForeachExpm1)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachIntToFloat);

IMPL_OP_INFERSHAPE(ForeachLerpList)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachLog1p)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachLog)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachLog2)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachLog10)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachLerpScalar)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

} // namespace ops
