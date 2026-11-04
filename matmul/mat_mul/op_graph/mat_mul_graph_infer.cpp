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
 * \file mat_mul_graph_infer.cpp
 * \brief
 */
#include "error_util.h"
#include "log/log.h"
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t MATMUL_X1_IDX = 0;
constexpr size_t MATMUL_X2_IDX = 1;
constexpr size_t MATMUL_BIAS_IDX = 2;
constexpr size_t MATMUL_Y_IDX = 0;

} // namespace

ge::graphStatus InferDataTypeMatMul(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeMatMul");
    const ge::DataType x1Dtype = context->GetInputDataType(MATMUL_X1_IDX);
    const ge::DataType x2Dtype = context->GetInputDataType(MATMUL_X2_IDX);
    const ge::DataType biasDtype = context->GetOptionalInputDataType(MATMUL_BIAS_IDX);
    // 对齐canndev RT1.0的SetMatMulOutputDtype完整规则:
    // int8/int4/int16走整数累加出int32; hifp8进尊重声明的y(量化融合图); 其余跟随x1
    const ge::DataType declaredYDtype = context->GetOutputDataType(MATMUL_Y_IDX);
    ge::DataType yDtype;
    if (x1Dtype == ge::DT_INT8 || x1Dtype == ge::DT_INT4 || x1Dtype == ge::DT_INT16) {
        yDtype = ge::DT_INT32;
    } else if (x1Dtype == ge::DT_HIFLOAT8) {
        yDtype = declaredYDtype;
    } else {
        yDtype = x1Dtype;
    }
    context->SetOutputDataType(MATMUL_Y_IDX, yDtype);
    OP_LOGD(
        context->GetNodeName(), "End to do InferDataTypeMatMul, x1 dtype:%s, x2 dtype:%s, bias dtype:%s, y dtype:%s",
        ge::TypeUtils::DataTypeToSerialString(x1Dtype).c_str(), ge::TypeUtils::DataTypeToSerialString(x2Dtype).c_str(),
        ge::TypeUtils::DataTypeToSerialString(biasDtype).c_str(),
        ge::TypeUtils::DataTypeToSerialString(yDtype).c_str());
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(MatMul).InferDataType(InferDataTypeMatMul);
} // namespace ops
