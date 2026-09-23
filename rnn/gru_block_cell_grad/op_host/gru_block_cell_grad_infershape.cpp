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
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/compute_node_info.h"
#include "gru_block_cell_grad_infer_common.h"
namespace ops {
namespace {
using namespace gru_block_cell_grad;
// A dim is "known" only when non-negative; dynamic-shape graphs carry -1 (or
// other negative sentinels) for unknown dims, which cannot be value-compared.
inline bool IsKnownDim(const int64_t dim) { return dim >= 0; }

// Compare a declared dim against a derived dim; unknown on either side means
// "cannot judge" and is accepted (rank checks still apply elsewhere).
inline bool DimCompatible(const int64_t declared, const int64_t derived)
{
    return !IsKnownDim(declared) || !IsKnownDim(derived) || declared == derived;
}

// ---------------------------------------------------------------------------
// Cross-tensor shape consistency from (batch, input_size, cell_size).
// Value comparisons only run for dims that are known on both sides.
// ---------------------------------------------------------------------------
bool InputShapesAreConsistent(const gert::Shape* const in[kNumInputs], const char* node)
{
    const int64_t batch = in[kInX]->GetDim(0);
    const int64_t inputSize = in[kInX]->GetDim(1);
    const int64_t cellSize = in[kInHPrev]->GetDim(1);
    if (cellSize == 0) {
        OP_LOGE(node, "cell_size must be greater than zero (C=0 is not supported)");
        return false;
    }

    const auto match2 = [&](size_t idx, int64_t d0, int64_t d1) {
        return DimCompatible(in[idx]->GetDim(0), d0) && DimCompatible(in[idx]->GetDim(1), d1);
    };
    const auto match1 = [&](size_t idx, int64_t d0) { return DimCompatible(in[idx]->GetDim(0), d0); };

    // h_prev / r / u / c / d_h == (batch, cell_size)
    for (const size_t idx : {kInHPrev, kInR, kInU, kInC, kInDH}) {
        if (!match2(idx, batch, cellSize)) {
            OP_LOGE(node, "input %s shape must be (batch=%lld, cell_size=%lld)", kInputNames[idx],
                    static_cast<long long>(batch), static_cast<long long>(cellSize));
            return false;
        }
    }
    if (IsKnownDim(inputSize) && IsKnownDim(cellSize)) {
        const int64_t kDim = inputSize + cellSize;
        if (!match2(kInWRu, kDim, 2 * cellSize)) {
            OP_LOGE(node, "input w_ru shape must be (input_size+cell_size=%lld, 2*cell_size=%lld)",
                    static_cast<long long>(kDim), static_cast<long long>(2 * cellSize));
            return false;
        }
        if (!match2(kInWC, kDim, cellSize)) {
            OP_LOGE(node, "input w_c shape must be (input_size+cell_size=%lld, cell_size=%lld)",
                    static_cast<long long>(kDim), static_cast<long long>(cellSize));
            return false;
        }
        if (!match1(kInBRu, 2 * cellSize)) {
            OP_LOGE(node, "input b_ru shape must be (2*cell_size=%lld,)", static_cast<long long>(2 * cellSize));
            return false;
        }
        if (!match1(kInBC, cellSize)) {
            OP_LOGE(node, "input b_c shape must be (cell_size=%lld,)", static_cast<long long>(cellSize));
            return false;
        }
    }
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// InferShape4GRUBlockCellGrad(context) — validation + shape derivation
// ---------------------------------------------------------------------------
static ge::graphStatus InferShape4GRUBlockCellGrad(gert::InferShapeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const char* node = (context->GetNodeName() != nullptr) ? context->GetNodeName() : "GRUBlockCellGrad";

    // 1) compile-time desc checks: dtype fp32 / format ND (inputs + outputs)
    if (!TensorDescsAreLegal(context, node)) {
        return ge::GRAPH_FAILED;
    }

    // 2) input shapes present
    const gert::Shape* in[kNumInputs] = {nullptr};
    gert::Shape normalized[kNumInputs];
    for (size_t i = 0; i < kNumInputs; ++i) {
        in[i] = context->GetInputShape(i);
        if (in[i] == nullptr) {
            OP_LOGE(node, "input %s shape is null", kInputNames[i]);
            return ge::GRAPH_FAILED;
        }
    }

    // 3) input rank checks
    for (size_t i = 0; i < kNumInputs; ++i) {
        // [-2] is a compile-time unknown rank, not an additional legal runtime rank.
        if (in[i]->GetDimNum() == 1 && in[i]->GetDim(0) == -2) {
            normalized[i].SetDimNum(kInputRank[i]);
            for (size_t dim = 0; dim < kInputRank[i]; ++dim) {
                normalized[i].SetDim(dim, -1);
            }
            in[i] = &normalized[i];
        }
        if (in[i]->GetDimNum() != kInputRank[i]) {
            OP_LOGE(node, "input %s rank must be %zu, got rank %zu", kInputNames[i], kInputRank[i], in[i]->GetDimNum());
            return ge::GRAPH_FAILED;
        }
    }

    // 4) cross-tensor shape consistency (known dims only)
    if (!InputShapesAreConsistent(in, node)) {
        return ge::GRAPH_FAILED;
    }

    // 5) derived output shapes (static per-dim copies, unknown dims preserved)
    gert::Shape derived[kNumOutputs];
    derived[kOutDX] = *in[kInX];         // d_x.shape = x.shape
    derived[kOutDHPrev] = *in[kInHPrev]; // d_h_prev.shape = h_prev.shape
    derived[kOutDCBar] = *in[kInHPrev];  // d_c_bar.shape = h_prev.shape
    derived[kOutDRub].SetDimNum(2);      // d_r_bar_u_bar = (h_prev.dim(0), w_ru.dim(1))
    derived[kOutDRub].SetDim(0, in[kInHPrev]->GetDim(0));
    derived[kOutDRub].SetDim(1, in[kInWRu]->GetDim(1));

    // 6) Output holders may retain the previous dynamic execution shape.  The
    // descriptor contract was checked above; Tiling repeats it for kernel-mode
    // callers that do not execute InferShape.
    gert::Shape* out[kNumOutputs] = {nullptr};
    for (size_t i = 0; i < kNumOutputs; ++i) {
        out[i] = context->GetOutputShape(i);
        if (out[i] == nullptr) {
            OP_LOGE(node, "output %s shape holder is null", kOutputNames[i]);
            return ge::GRAPH_FAILED;
        }
    }

    // 7) commit derived shapes
    *out[kOutDX] = derived[kOutDX];
    *out[kOutDHPrev] = derived[kOutDHPrev];
    *out[kOutDCBar] = derived[kOutDCBar];
    *out[kOutDRub] = derived[kOutDRub];
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GRUBlockCellGrad).InferShape(InferShape4GRUBlockCellGrad);
} // namespace ops
