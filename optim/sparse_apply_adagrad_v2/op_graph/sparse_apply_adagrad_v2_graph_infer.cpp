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
 * \file sparse_apply_adagrad_v2_graph_infer.cpp
 * \brief
 */

#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "error_util.h"
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
namespace {
constexpr int64_t IDX_VAR = 0;
constexpr int64_t IDX_ACCUM = 1;
constexpr int64_t IDX_LR = 2;
constexpr int64_t IDX_EPSILON = 3;
constexpr int64_t IDX_GRAD = 4;
constexpr int64_t IDX_INDICES = 5;
} // namespace

static ge::graphStatus InferDataTypeSparseApplyAdagradV2Graph(gert::InferDataTypeContext* context)
{
    OP_LOGI(context->GetNodeName(), "***runtime2.0 inferDtype***");
    // dtype 一致性校验（与 README"约束说明"公开声明一致）:
    // var.dtype == accum.dtype == lr.dtype == epsilon.dtype == grad.dtype
    auto varDtype = context->GetInputDataType(IDX_VAR);
    auto accumDtype = context->GetInputDataType(IDX_ACCUM);
    auto lrDtype = context->GetInputDataType(IDX_LR);
    auto epsilonDtype = context->GetInputDataType(IDX_EPSILON);
    auto gradDtype = context->GetInputDataType(IDX_GRAD);

    OP_CHECK_IF(varDtype != accumDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "var and accum",
                                                       (ge::TypeUtils::DataTypeToSerialString(varDtype) + ", " +
                                                        ge::TypeUtils::DataTypeToSerialString(accumDtype))
                                                           .c_str(),
                                                       "var dtype must equal accum dtype"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(varDtype != lrDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "var and lr",
                                                       (ge::TypeUtils::DataTypeToSerialString(varDtype) + ", " +
                                                        ge::TypeUtils::DataTypeToSerialString(lrDtype))
                                                           .c_str(),
                                                       "var dtype must equal lr dtype"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(varDtype != epsilonDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "var and epsilon",
                                                       (ge::TypeUtils::DataTypeToSerialString(varDtype) + ", " +
                                                        ge::TypeUtils::DataTypeToSerialString(epsilonDtype))
                                                           .c_str(),
                                                       "var dtype must equal epsilon dtype"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(varDtype != gradDtype,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "var and grad",
                                                       (ge::TypeUtils::DataTypeToSerialString(varDtype) + ", " +
                                                        ge::TypeUtils::DataTypeToSerialString(gradDtype))
                                                           .c_str(),
                                                       "var dtype must equal grad dtype"),
                return ge::GRAPH_FAILED);

    // indices.dtype == int32 或 int64（与基线算子信息库 int32/int64 组合声明一致）
    auto indicesDtype = context->GetInputDataType(IDX_INDICES);
    OP_CHECK_IF(indicesDtype != ge::DT_INT32 && indicesDtype != ge::DT_INT64,
                OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "indices",
                                          ge::TypeUtils::DataTypeToSerialString(indicesDtype).c_str(), "INT32/INT64"),
                return ge::GRAPH_FAILED);

    // 输出 var dtype = 输入 var dtype（原地更新）
    context->SetOutputDataType(IDX_VAR, varDtype);

    // 输出 accum dtype = 输入 accum dtype（原地更新）
    context->SetOutputDataType(IDX_ACCUM, accumDtype);

    return GRAPH_SUCCESS;
}

IMPL_OP(SparseApplyAdagradV2).InferDataType(InferDataTypeSparseApplyAdagradV2Graph);

}; // namespace ops
