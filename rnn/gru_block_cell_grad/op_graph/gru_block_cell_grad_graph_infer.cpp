/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "exe_graph/runtime/compute_node_info.h"
#include "../op_host/gru_block_cell_grad_infer_common.h"
namespace ops {
using namespace gru_block_cell_grad;
static ge::graphStatus InferDataTypeForGRUBlockCellGrad(gert::InferDataTypeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const char* node = (context->GetNodeName() != nullptr) ? context->GetNodeName() : "GRUBlockCellGrad";

    // Validate input descriptors before deriving output datatypes.
    if (!InputTensorDescsAreLegal(context, node)) {
        return ge::GRAPH_FAILED;
    }

    // All ten required inputs must be float32.
    for (size_t i = 0; i < kNumInputs; ++i) {
        const ge::DataType dtype = context->GetInputDataType(i);
        if (dtype != ge::DT_FLOAT) {
            OP_LOGE(node, "input %s dtype must be float32", kInputNames[i]);
            return ge::GRAPH_FAILED;
        }
    }

    // Derive output datatypes without reading previous output holders.
    // Declared output constraints are checked by the GE verifier.
    if (context->SetOutputDataType(kOutDX, context->GetInputDataType(kInX)) != ge::GRAPH_SUCCESS ||
        context->SetOutputDataType(kOutDHPrev, context->GetInputDataType(kInHPrev)) != ge::GRAPH_SUCCESS ||
        context->SetOutputDataType(kOutDCBar, context->GetInputDataType(kInHPrev)) != ge::GRAPH_SUCCESS ||
        context->SetOutputDataType(kOutDRub, context->GetInputDataType(kInWRu)) != ge::GRAPH_SUCCESS) {
        OP_LOGE(node, "failed to set output datatypes");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(GRUBlockCellGrad).InferDataType(InferDataTypeForGRUBlockCellGrad);
} // namespace ops
