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
 * \file weight_quant_batch_matmul_v2_graph_infer.cpp
 * \brief
 */
#include "graph/utils/type_utils.h"
#include "error_util.h"
#include "register/op_impl_registry.h"

namespace ops {
constexpr size_t QUANT_SCALE_IDX = 4UL;
constexpr size_t ATTR_DTYPE_IDX = 3UL;

static ge::graphStatus InferDataTypeForWeightQuantBatchMatmulV2(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeForWeightQuantBatchMatmulV2");
    OP_CHECK_IF(context == nullptr, CUBE_INNER_ERR_REPORT("", "Get %s failed", "context"), return ge::GRAPH_FAILED);
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* outputDtype = attrs->GetAttrPointer<int64_t>(ATTR_DTYPE_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, outputDtype);
    auto xDtype = context->GetInputDataType(0);
    auto quantScaleDtype = context->GetOptionalInputDataType(QUANT_SCALE_IDX);
    // 非onnx场景outputDtype为-1, C8时quantScaleDtype为ge::DT_UNDEFINED
    // onnx场景outputDtype不为-1，C8时outputDtype为ge::DT_INT8
    bool isOutputDtypeINT8 = (*outputDtype == -1 && quantScaleDtype != ge::DT_UNDEFINED) ||
                             (*outputDtype != -1 && static_cast<ge::DataType>(*outputDtype) == ge::DT_INT8);
    context->SetOutputDataType(0, isOutputDtypeINT8 ? ge::DT_INT8 : xDtype);
    OP_LOGD(context->GetNodeName(), "get x dtype %s, quant scale dtype %s and output dtype %ld, set y dtype %s",
            ge::TypeUtils::DataTypeToAscendString(xDtype).GetString(),
            ge::TypeUtils::DataTypeToAscendString(quantScaleDtype).GetString(), *outputDtype,
            ge::TypeUtils::DataTypeToAscendString(context->GetOutputDataType(0)).GetString());
    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeForWeightQuantBatchMatmulV2");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(WeightQuantBatchMatmulV2).InferDataType(InferDataTypeForWeightQuantBatchMatmulV2);
} // namespace ops
