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
 * \file quant_batch_matmul_v3_graph_infer.cpp
 * \brief Graph-level data type inference for QuantBatchMatmulV3.
 */

#include <string>

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "graph/utils/type_utils.h"
#include "op_common/log/log.h"
#include "error_util.h"

namespace ops {
namespace {
constexpr size_t kDtypeAttrIdx = 0U;
constexpr size_t kOutputIdx = 0U;
constexpr const char* kOpName = "QuantBatchMatmulV3";
} // namespace

bool IsSupportedYDtypeValue(const int64_t dtypeValue)
{
    return dtypeValue == static_cast<int64_t>(ge::DT_FLOAT) || dtypeValue == static_cast<int64_t>(ge::DT_FLOAT16) ||
           dtypeValue == static_cast<int64_t>(ge::DT_INT8) || dtypeValue == static_cast<int64_t>(ge::DT_INT32) ||
           dtypeValue == static_cast<int64_t>(ge::DT_BF16);
}

ge::graphStatus SetQuantBatchMatmulV3OutputDtype(gert::InferDataTypeContext* context, const int64_t dtypeValue)
{
    // Validate the raw 64-bit value before casting: casting first would truncate out-of-range
    // values (e.g. 4294967297) into legal enum values and silently accept them.
    if (!IsSupportedYDtypeValue(dtypeValue)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "dtype", std::to_string(dtypeValue).c_str(),
                                              "y dtype must be one of float32/float16/int8/int32/bfloat16");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType outDtype = static_cast<ge::DataType>(dtypeValue);
    if (context->SetOutputDataType(kOutputIdx, outDtype) != ge::GRAPH_SUCCESS) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "output dtype", "setter returned failure",
                                              "output dtype must be writable");
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "QuantBatchMatmulV3 set y dtype %s",
            ge::TypeUtils::DataTypeToSerialString(outDtype).c_str());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeForQuantBatchMatmulV3(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        OP_LOGE(kOpName, "[InferDataType] context is null.");
        return ge::GRAPH_FAILED;
    }
    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr || attrs->GetAttrNum() <= kDtypeAttrIdx) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "dtype", "null", "dtype attr is required");
        return ge::GRAPH_FAILED;
    }
    const int64_t* dtypeNum = attrs->GetAttrPointer<int64_t>(kDtypeAttrIdx);
    if (dtypeNum == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "dtype", "null", "dtype attr is required");
        return ge::GRAPH_FAILED;
    }
    return SetQuantBatchMatmulV3OutputDtype(context, *dtypeNum);
}

// op_host infershape.cpp already occupies IMPL_OP_INFERSHAPE for this operator, so the graph-level
// callback is registered through the IMPL_OP chain to keep a single registration unit.
IMPL_OP(QuantBatchMatmulV3).InferDataType(InferDataTypeForQuantBatchMatmulV3);
} // namespace ops
