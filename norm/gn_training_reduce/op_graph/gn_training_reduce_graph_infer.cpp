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
 * \file gn_training_reduce_graph_infer.cpp
 * \brief GNTrainingReduce graph-level data type inference.
 */

// op_impl_registry.h: provides IMPL_OP macro, gert::InferDataTypeContext.
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_datatype_context.h"
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

/**
 * InferDataTypeForGNTrainingReduce: GE data-type inference callback for GNTrainingReduce.
 *
 * Parameters:
 *   context — [in/out] inference context; provides the input dtype and accepts
 *             the computed output dtypes.
 *
 * Returns:
 *   ge::GRAPH_SUCCESS when input 0 (x) is float16 or float32; ge::GRAPH_FAILED
 *   otherwise.
 *
 * Logic:
 *   Both outputs are fixed to float32 regardless of the input dtype (fp16 input
 *   is accumulated in fp32).  An unsupported input dtype is rejected without
 *   touching the output descriptors.
 */
static ge::graphStatus InferDataTypeForGNTrainingReduce(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const ge::DataType xDataType = context->GetInputDataType(0);
    if (xDataType != ge::DT_FLOAT16 && xDataType != ge::DT_FLOAT) {
        OP_LOGE(context, "GNTrainingReduce only supports float16/float32 input, but got %d.",
                static_cast<int32_t>(xDataType));
        return ge::GRAPH_FAILED;
    }

    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

/**
 * IMPL_OP(GNTrainingReduce).InferDataType(...): registers the dtype inference function
 *   for the operator named "GNTrainingReduce" at static init time.  When GE encounters
 *   a GNTrainingReduce node during graph compilation, it calls InferDataTypeForGNTrainingReduce
 *   to determine the output types.
 */
IMPL_OP(GNTrainingReduce).InferDataType(InferDataTypeForGNTrainingReduce);

} // namespace ops
