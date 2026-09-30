/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// bn3d_training_update_grad_package/op_graph/bn3d_training_update_grad_graph_infer.cpp
// =============================================================================
//
// ROLE: Graph-level data type inference for the BN3DTrainingUpdateGrad operator.
//   GE needs the output data types before execution. This host-side callback
//   tells GE what dtype each output carries. It is loaded via op_build and run
//   during graph compilation (never on the device).
//
//   Both outputs are ALWAYS float32. This is independent
//   of the grads/x dtype (fp16/bf16 are cast to fp32 inside the kernel), so the
//   callback sets DT_FLOAT unconditionally rather than following any input.
//
// CONTENTS:
//   - InferDataTypeForBN3DTrainingUpdateGrad() — the type inference function
//   - IMPL_OP(BN3DTrainingUpdateGrad).InferDataType(...) — registration macro
//
// =============================================================================

#include "register/op_impl_registry.h" // IMPL_OP macro for operator registration
#include "op_common/log/log.h"         // OP_LOGE / OP_CHECK_IF for input-legality rejection

using namespace ge;

namespace ops {

// Input index constants: grads=0, x=1.
constexpr size_t INPUT_GRADS_INDEX = 0;
constexpr size_t INPUT_X_INDEX = 1;

// Output index constants.
constexpr size_t OUTPUT_DIFF_SCALE_INDEX = 0;
constexpr size_t OUTPUT_DIFF_OFFSET_INDEX = 1;

// ---------------------------------------------------------------------------
// InferDataTypeForBN3DTrainingUpdateGrad(context) — data type inference callback
//
// diff_scale.dtype = diff_offset.dtype = float32, unconditionally. The op has no
// dtype promotion (dtype_policy.promotion = fixed): regardless of whether
// grads/x are fp16/fp32/bf16, both parameter-gradient outputs are fp32.
// ---------------------------------------------------------------------------
static ge::graphStatus InferDataTypeForBN3DTrainingUpdateGrad(gert::InferDataTypeContext* context)
{
    // Input-legality check: grads and x must carry the SAME dtype.
    //   The proto declares "grads: same dtype as x" and the supported combinations are
    //   only {fp16/fp16, fp32/fp32, bf16/bf16}. A mismatched pair
    //   (e.g. fp16 grads + fp32 x) is illegal; the kernel casts both through one shared
    //   path and would silently mis-read x otherwise. Reject here so GE fails graph
    //   compilation instead of executing an unsupported combination.
    const ge::DataType grads_dtype = context->GetInputDataType(INPUT_GRADS_INDEX);
    const ge::DataType x_dtype = context->GetInputDataType(INPUT_X_INDEX);
    const char* node = (context->GetNodeName() == nullptr) ? "BN3DTrainingUpdateGrad" : context->GetNodeName();
    OP_CHECK_IF(grads_dtype != x_dtype,
                OP_LOGE(node, "grads dtype (%d) must equal x dtype (%d).", static_cast<int32_t>(grads_dtype),
                        static_cast<int32_t>(x_dtype)),
                return ge::GRAPH_FAILED);

    context->SetOutputDataType(OUTPUT_DIFF_SCALE_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(OUTPUT_DIFF_OFFSET_INDEX, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

// IMPL_OP(BN3DTrainingUpdateGrad).InferDataType(func):
//   Registers InferDataTypeForBN3DTrainingUpdateGrad as the type inference
//   function for the BN3DTrainingUpdateGrad operator type.
IMPL_OP(BN3DTrainingUpdateGrad).InferDataType(InferDataTypeForBN3DTrainingUpdateGrad);
} // namespace ops
