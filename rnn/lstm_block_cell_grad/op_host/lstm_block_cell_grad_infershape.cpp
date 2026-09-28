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
 * Shape inference for the LSTMBlockCellGrad operator, registered via
 * IMPL_OP_INFERSHAPE(LSTMBlockCellGrad).
 *
 * Input order (OpDef):
 *   0=x 1=cs_prev 2=h_prev 3=w 4=wci 5=wcf 6=wco 7=b
 *   8=i 9=cs 10=f 11=o 12=ci 13=co 14=cs_grad 15=h_grad
 * Output order:
 *   0=cs_prev_grad 1=dicfo 2=wci_grad 3=wcf_grad 4=wco_grad
 *
 * Derivation (pure mirror, no broadcast):
 *   cs_prev_grad.shape = cs_prev.shape                      → (B, C)
 *   dicfo.shape        = cs_prev.shape[:-1] + (w.shape[-1],) → (B, 4C)
 *   wci_grad.shape     = wci.shape                          → (C,)
 *   wcf_grad.shape     = wcf.shape                          → (C,)
 *   wco_grad.shape     = wco.shape                          → (C,)
 *
 * Validation (violations → GRAPH_FAILED + ERROR log, keywords null_input /
 * shape_mismatch):
 *   - null input shape
 *   - rank (12 rank-2 + 4 rank-1 fixed)
 *   - dim value < 0 and != -1
 *   - batch dim consistency (B)
 *   - cell dim consistency (C, incl. wci/wcf/wco length)
 *   - w.shape == (N + C, 4C)
 *   - b.shape[0] == w.shape[1] == 4C
 * -1 (unknown) dims are skipped in consistency checks and passed through in
 * the outputs (symbolic-dim transparency); 0 (empty tensor) dims are legal
 * and passed through unchanged.
 */

#include "register/op_impl_registry.h"

#include <cstdint>
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

// Rank contract: 12 inputs rank=2 (x / cs_prev / h_prev / w / i / cs / f / o /
// ci / co / cs_grad / h_grad), 4 inputs rank=1 (wci / wcf / wco / b).
constexpr int RANK2_INPUTS[] = {0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15};
constexpr int RANK1_INPUTS[] = {4, 5, 6, 7};

// Batch-bearing (B, ·) inputs besides x: cs_prev / h_prev / i / cs / f / o /
// ci / co / cs_grad / h_grad.
constexpr int BATCH_INPUTS[] = {1, 2, 8, 9, 10, 11, 12, 13, 14, 15};

// Cell-bearing (·, C) inputs besides cs_prev.
constexpr int CELL_INPUTS[] = {2, 8, 9, 10, 11, 12, 13, 14, 15};

// (·, C) peephole weight vectors: wci / wcf / wco (shape[0] == C).
constexpr int PEEPHOLE_INPUTS[] = {4, 5, 6};

// Unknown dim (symbolic -1, skipped in consistency checks, passed through).
constexpr int64_t UNKNOWN_DIM = -1;

// icfo 列块数 (i/c/f/o 四列块; 与 kernel 侧 NUM_DICFO_BLOCKS 同源).
constexpr int64_t DICFO_COL_BLOCKS = 4;

// True when the dim participates in a consistency check (known, non-symbolic).
inline bool IsKnownDim(int64_t dim) { return dim != UNKNOWN_DIM; }

} // namespace

/**
 * InferShapeForLSTMBlockCellGrad: GE shape inference callback.
 *
 * Valid input path → mirror-derivation stub results;
 * invalid input path → GRAPH_FAILED rejection + ERROR log (shape_mismatch /
 * null_input).
 */
static ge::graphStatus InferShapeForLSTMBlockCellGrad(gert::InferShapeContext* context)
{
    // ---- null input shapes → null_input --------------------------------
    const gert::Shape* in[NUM_INPUTS] = {nullptr};
    for (int idx = 0; idx < NUM_INPUTS; ++idx) {
        in[idx] = context->GetInputShape(idx);
        if (in[idx] == nullptr) {
            std::fprintf(stderr,
                         "[ERROR][LSTMBlockCellGrad][InferShape] null_input: input '%s' (index %d) "
                         "has no shape on the InferShape context\n",
                         INPUT_NAMES[idx], idx);
            return GRAPH_FAILED;
        }
    }

    // ---- dim value sanity: >= 0 or symbolic -1 --------------------------
    for (int idx = 0; idx < NUM_INPUTS; ++idx) {
        const size_t dimNum = in[idx]->GetDimNum();
        for (size_t d = 0; d < dimNum; ++d) {
            const int64_t dim = in[idx]->GetDim(d);
            if (dim < 0 && dim != UNKNOWN_DIM) {
                std::fprintf(stderr,
                             "[ERROR][LSTMBlockCellGrad][InferShape] shape_mismatch: input '%s' (index %d) "
                             "dim[%zu] = %lld is neither >= 0 nor the unknown dim -1\n",
                             INPUT_NAMES[idx], idx, d, static_cast<long long>(dim));
                return GRAPH_FAILED;
            }
        }
    }

    // ---- rank contract: 12 rank-2 + 4 rank-1 ----------------------------
    for (int idx : RANK2_INPUTS) {
        if (in[idx]->GetDimNum() != 2) {
            std::fprintf(stderr,
                         "[ERROR][LSTMBlockCellGrad][InferShape] shape_mismatch: input '%s' (index %d) "
                         "must be rank 2, got rank %zu\n",
                         INPUT_NAMES[idx], idx, in[idx]->GetDimNum());
            return GRAPH_FAILED;
        }
    }
    for (int idx : RANK1_INPUTS) {
        if (in[idx]->GetDimNum() != 1) {
            std::fprintf(stderr,
                         "[ERROR][LSTMBlockCellGrad][InferShape] shape_mismatch: input '%s' (index %d) "
                         "must be rank 1, got rank %zu\n",
                         INPUT_NAMES[idx], idx, in[idx]->GetDimNum());
            return GRAPH_FAILED;
        }
    }

    // Anchor dims: B = x.shape[0], N = x.shape[1], C = cs_prev.shape[1].
    const int64_t batchDim = in[0]->GetDim(0);
    const int64_t numInputsDim = in[0]->GetDim(1);
    const int64_t cellDim = in[1]->GetDim(1);

    // ---- batch dim consistency (B) --------------------------------------
    for (int idx : BATCH_INPUTS) {
        const int64_t other = in[idx]->GetDim(0);
        if (IsKnownDim(batchDim) && IsKnownDim(other) && batchDim != other) {
            std::fprintf(stderr,
                         "[ERROR][LSTMBlockCellGrad][InferShape] shape_mismatch: batch dim mismatch — "
                         "x.shape[0] = %lld but '%s' (index %d).shape[0] = %lld\n",
                         static_cast<long long>(batchDim), INPUT_NAMES[idx], idx, static_cast<long long>(other));
            return GRAPH_FAILED;
        }
    }

    // ---- cell dim consistency (C, incl. wci/wcf/wco lengths) ------------
    for (int idx : CELL_INPUTS) {
        const int64_t other = in[idx]->GetDim(1);
        if (IsKnownDim(cellDim) && IsKnownDim(other) && cellDim != other) {
            std::fprintf(stderr,
                         "[ERROR][LSTMBlockCellGrad][InferShape] shape_mismatch: cell dim mismatch — "
                         "cs_prev.shape[1] = %lld but '%s' (index %d).shape[1] = %lld\n",
                         static_cast<long long>(cellDim), INPUT_NAMES[idx], idx, static_cast<long long>(other));
            return GRAPH_FAILED;
        }
    }
    for (int idx : PEEPHOLE_INPUTS) {
        const int64_t other = in[idx]->GetDim(0);
        if (IsKnownDim(cellDim) && IsKnownDim(other) && cellDim != other) {
            std::fprintf(stderr,
                         "[ERROR][LSTMBlockCellGrad][InferShape] shape_mismatch: peephole weight length != cell — "
                         "cs_prev.shape[1] = %lld but '%s' (index %d).shape[0] = %lld\n",
                         static_cast<long long>(cellDim), INPUT_NAMES[idx], idx, static_cast<long long>(other));
            return GRAPH_FAILED;
        }
    }

    // ---- w.shape == (N + C, 4C) (icfo column block) ---------------------
    const int64_t wRows = in[3]->GetDim(0);
    const int64_t wCols = in[3]->GetDim(1);
    if (IsKnownDim(numInputsDim) && IsKnownDim(cellDim) && IsKnownDim(wRows)) {
        if (numInputsDim > INT64_MAX - cellDim) {
            std::fprintf(stderr,
                         "[ERROR][LSTMBlockCellGrad][InferShape] shape_mismatch: w row count overflow — "
                         "x.shape[1] (%lld) + cs_prev.shape[1] (%lld) exceeds int64 range\n",
                         static_cast<long long>(numInputsDim), static_cast<long long>(cellDim));
            return GRAPH_FAILED;
        }
        const int64_t expectedRows = numInputsDim + cellDim;
        if (wRows != expectedRows) {
            std::fprintf(stderr,
                         "[ERROR][LSTMBlockCellGrad][InferShape] shape_mismatch: w.shape[0] = %lld, "
                         "expected x.shape[1] + cs_prev.shape[1] = %lld + %lld = %lld\n",
                         static_cast<long long>(wRows), static_cast<long long>(numInputsDim),
                         static_cast<long long>(cellDim), static_cast<long long>(expectedRows));
            return GRAPH_FAILED;
        }
    }
    if (IsKnownDim(cellDim) && IsKnownDim(wCols)) {
        if (cellDim > INT64_MAX / DICFO_COL_BLOCKS) {
            std::fprintf(stderr,
                         "[ERROR][LSTMBlockCellGrad][InferShape] shape_mismatch: 4 * cs_prev.shape[1] "
                         "(%lld) exceeds int64 range\n",
                         static_cast<long long>(cellDim));
            return GRAPH_FAILED;
        }
        const int64_t expectedCols = DICFO_COL_BLOCKS * cellDim;
        if (wCols != expectedCols) {
            std::fprintf(stderr,
                         "[ERROR][LSTMBlockCellGrad][InferShape] shape_mismatch: w.shape[1] = %lld, "
                         "expected 4 * cs_prev.shape[1] = %lld\n",
                         static_cast<long long>(wCols), static_cast<long long>(expectedCols));
            return GRAPH_FAILED;
        }
    }

    // ---- b.shape[0] == w.shape[1] == 4C ----------------------------------
    const int64_t bLen = in[7]->GetDim(0);
    if (IsKnownDim(wCols) && IsKnownDim(bLen) && bLen != wCols) {
        std::fprintf(stderr,
                     "[ERROR][LSTMBlockCellGrad][InferShape] shape_mismatch: b.shape[0] = %lld, "
                     "expected w.shape[1] = %lld\n",
                     static_cast<long long>(bLen), static_cast<long long>(wCols));
        return GRAPH_FAILED;
    } else if (!IsKnownDim(wCols) && IsKnownDim(cellDim) && IsKnownDim(bLen)) {
        if (cellDim > INT64_MAX / DICFO_COL_BLOCKS) {
            std::fprintf(stderr,
                         "[ERROR][LSTMBlockCellGrad][InferShape] shape_mismatch: 4 * cs_prev.shape[1] "
                         "(%lld) exceeds int64 range\n",
                         static_cast<long long>(cellDim));
            return GRAPH_FAILED;
        }
        const int64_t expectedLen = DICFO_COL_BLOCKS * cellDim;
        if (bLen != expectedLen) {
            std::fprintf(stderr,
                         "[ERROR][LSTMBlockCellGrad][InferShape] shape_mismatch: b.shape[0] = %lld, "
                         "expected 4 * cs_prev.shape[1] = %lld\n",
                         static_cast<long long>(bLen), static_cast<long long>(expectedLen));
            return GRAPH_FAILED;
        }
    }

    // ---- mirror-derivation results (valid input path) --------------------
    // 0-dim (empty tensors B==0 / C==0) and -1 (unknown) dims are passed
    // through unchanged — no round-up / alignment rewrite.
    // Output shape pointers are null-checked symmetrically with the input
    // side — a null would mean GE failed to pre-allocate the output shape,
    // i.e. an infrastructure fault.
    gert::Shape* csPrevGrad = context->GetOutputShape(0);
    if (csPrevGrad == nullptr) {
        std::fprintf(stderr, "[ERROR][LSTMBlockCellGrad][InferShape] null_output_shape: output 0 (cs_prev_grad)\n");
        return GRAPH_FAILED;
    }
    csPrevGrad->SetDimNum(0);
    csPrevGrad->AppendDim(in[1]->GetDim(0)); // batch (0 / -1 passed through)
    csPrevGrad->AppendDim(in[1]->GetDim(1)); // cell

    gert::Shape* dicfo = context->GetOutputShape(1);
    if (dicfo == nullptr) {
        std::fprintf(stderr, "[ERROR][LSTMBlockCellGrad][InferShape] null_output_shape: output 1 (dicfo)\n");
        return GRAPH_FAILED;
    }
    dicfo->SetDimNum(0);
    dicfo->AppendDim(in[1]->GetDim(0)); // batch
    dicfo->AppendDim(in[3]->GetDim(1)); // 4*cell (w last dim, icfo)

    const int peepholeIdx[] = {4, 5, 6}; // wci / wcf / wco → outputs 2/3/4
    for (int outIdx = 2; outIdx < NUM_OUTPUTS; ++outIdx) {
        gert::Shape* peepGrad = context->GetOutputShape(outIdx);
        if (peepGrad == nullptr) {
            std::fprintf(stderr,
                         "[ERROR][LSTMBlockCellGrad][InferShape] null_output_shape: output %d (peephole grad)\n",
                         outIdx);
            return GRAPH_FAILED;
        }
        peepGrad->SetDimNum(0);
        peepGrad->AppendDim(in[peepholeIdx[outIdx - 2]]->GetDim(0)); // cell
    }
    return GRAPH_SUCCESS;
}

/**
 * IMPL_OP_INFERSHAPE(LSTMBlockCellGrad).InferShape(InferShapeForLSTMBlockCellGrad):
 *   Registers the shape inference function at static init time.  When the
 *   framework needs to determine the output shapes of an LSTMBlockCellGrad
 *   node, it calls InferShapeForLSTMBlockCellGrad.
 */
IMPL_OP_INFERSHAPE(LSTMBlockCellGrad).InferShape(InferShapeForLSTMBlockCellGrad);

} // namespace ops
