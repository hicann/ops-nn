/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Graph-level data-type inference for the LSTMBlockCellGrad operator,
 * registered via IMPL_OP(LSTMBlockCellGrad).InferDataType.  This tells the
 * GE (Graph Engine) compiler how to deduce the output data types from the
 * input data types during graph compilation.
 *
 * Derivation rule (TF type_attr T semantics — single dtype across all 21
 * tensors): all 5 outputs (cs_prev_grad / dicfo / wci_grad / wcf_grad /
 * wco_grad) take x.dtype (first input), promoted to NO other dtype.
 *
 * Validation (violations → GRAPH_FAILED + ERROR log, keyword
 * dtype_not_supported):
 *   - x.dtype not in {DT_FLOAT, DT_FLOAT16}
 *   - any of the 16 inputs' dtype != x.dtype (mixed dtype violates type_attr T)
 */

#include "register/op_impl_registry.h"

#include <cstdio>

#include "graph/types.h"

using namespace ge;

namespace ops {

namespace {

// OpDef input order — index-aligned names for error messages.
constexpr int NUM_INPUTS = 16;
constexpr int NUM_OUTPUTS = 5;
const char* const INPUT_NAMES[NUM_INPUTS] = {"x", "cs_prev", "h_prev", "w", "wci", "wcf", "wco",     "b",
                                             "i", "cs",      "f",      "o", "ci",  "co",  "cs_grad", "h_grad"};

} // namespace

/**
 * InferDataTypeForLSTMBlockCellGrad: GE data-type inference callback.
 *
 * Parameters:
 *   context — [in/out] inference context; provides input dtypes and accepts
 *             the computed output dtype.
 *
 * Returns:
 *   ge::GRAPH_SUCCESS on the valid path (x.dtype → all 5 outputs);
 *   ge::GRAPH_FAILED + ERROR log (dtype_not_supported) for an unsupported
 *   x.dtype or a mixed-dtype input set.
 */
static ge::graphStatus InferDataTypeForLSTMBlockCellGrad(gert::InferDataTypeContext* context)
{
    const ge::DataType xDataType = context->GetInputDataType(0);

    // ---- x.dtype must be in the OpDef-registered set {float32, float16} --
    if (xDataType != ge::DT_FLOAT && xDataType != ge::DT_FLOAT16) {
        std::fprintf(stderr,
                     "[ERROR][LSTMBlockCellGrad][InferDataType] dtype_not_supported: x.dtype = %d "
                     "(input 'x', index 0); supported dtypes are {float32, float16}\n",
                     static_cast<int>(xDataType));
        return ge::GRAPH_FAILED;
    }

    // ---- single dtype across all 16 inputs (TF type_attr T semantics) ----
    for (int idx = 1; idx < NUM_INPUTS; ++idx) {
        const ge::DataType other = context->GetInputDataType(idx);
        if (other != xDataType) {
            std::fprintf(stderr,
                         "[ERROR][LSTMBlockCellGrad][InferDataType] dtype_not_supported: mixed input dtype — "
                         "x.dtype = %d but '%s' (index %d).dtype = %d; all 16 inputs must share x.dtype "
                         "(type_attr T)\n",
                         static_cast<int>(xDataType), INPUT_NAMES[idx], idx, static_cast<int>(other));
            return ge::GRAPH_FAILED;
        }
    }

    // ---- valid path: propagate x.dtype to all 5 outputs ------------------
    for (int outIdx = 0; outIdx < NUM_OUTPUTS; ++outIdx) {
        context->SetOutputDataType(outIdx, xDataType);
    }
    return ge::GRAPH_SUCCESS;
}

/**
 * IMPL_OP(LSTMBlockCellGrad).InferDataType(...): registers the dtype inference
 *   function for the operator named "LSTMBlockCellGrad" at static init time.
 *   When GE encounters an LSTMBlockCellGrad node during graph compilation, it
 *   calls InferDataTypeForLSTMBlockCellGrad to determine the output types.
 */
IMPL_OP(LSTMBlockCellGrad).InferDataType(InferDataTypeForLSTMBlockCellGrad);

} // namespace ops
