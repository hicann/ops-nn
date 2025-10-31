/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
#include "common_dtype.h"

using namespace ge;
namespace ops {
static ge::graphStatus InferShape4ForeachCommon(gert::InferShapeContext* context)
{
    uint32_t outputNumInferShape = context->GetComputeNodeOutputNum();
    uint32_t inputNumInferShape = context->GetComputeNodeInputNum();

    std::string errMsg = optiling::ConcatString(
        "num of dynamic input0 ", inputNumInferShape, "not equal num of dynamic output0 ",
        outputNumInferShape);

    for (uint32_t i = 0; i < outputNumInferShape; i++) {
        auto xShape = context->GetInputShape(i);
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
    uint32_t inputNumInferShape = context->GetComputeNodeInputNum();

    std::string errMsg = optiling::ConcatString(
        "num of dynamic input0 ", inputNumInferShape, "not equal num of dynamic output0 ",
        outputNumDataType);

    for (uint32_t i = 0; i < outputNumDataType; i++) {
        auto xDtype = context->GetInputDataType(i);

        context->SetOutputDataType(i, xDtype);
    }
    return ge::GRAPH_SUCCESS;
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
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachSigmoid)
    .InferShape(InferShape4ForeachCommon)
    .InferDataType(InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachSinh)
    .InferShape(InferShape4ForeachCommon)
    .InferDataType(InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachSubList)
    .InferShape(InferShape4ForeachCommon)
    .InferDataType(InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachSubScalarList)
    .InferShape(InferShape4ForeachCommon)
    .InferDataType(InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachSubScalar)
    .InferShape(InferShape4ForeachCommon)
    .InferDataType(InferDataType4ForeachCommon);

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
    .InferDataType(ops::InferDataType4ForeachCommon);

IMPL_OP_INFERSHAPE(ForeachExpm1)
    .InferShape(ops::InferShape4ForeachCommon)
    .InferDataType(ops::InferDataType4ForeachCommon);

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