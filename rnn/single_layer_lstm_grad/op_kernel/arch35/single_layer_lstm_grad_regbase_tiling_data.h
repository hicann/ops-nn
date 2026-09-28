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
 * \file single_layer_lstm_grad_regbase_tiling_data.h
 * \brief tiling data struct + UB layout shared by host tiling and the arch35 small-shape kernel.
 *
 * Path S ("regbase small") kernel model:
 *   - AIV-only, no cube, no matmul lib, no cross-core sync.
 *   - Narrow IO uses private FP32 forward replay and dw/db workspace.
 *   - Every core redundantly computes the full backward recurrence chain in its own UB
 *     (all chain tensors resident; dgate kept in UB in [t][b][4H] layout, fp32).
 *   - Output columns of the (inputSize + hiddenSize) dimension are partitioned disjointly
 *     across cores: each core produces dx[:, cols] and dw[:, cols] for its own column
 *     chunks; the last core additionally produces the hidden-column part of dw plus
 *     db / dh_prev / dc_prev. Disjoint columns => no atomics, no reduction across cores.
 *
 * Eligibility is decided on the host with the exact same UB layout formula the kernel
 * uses for addressing (LstmGradRegbaseSmallUbLayout), so the two sides can never diverge.
 */

#ifndef SINGLE_LAYER_LSTM_GRAD_REGBASE_TILING_DATA_H
#define SINGLE_LAYER_LSTM_GRAD_REGBASE_TILING_DATA_H

#include <cstdint>

#if defined(__CCE_AICORE__) || defined(__CCE_KT_TEST__)
#define LSTM_REGBASE_HOST_DEVICE __aicore__ inline
#else
#define LSTM_REGBASE_HOST_DEVICE inline
#endif

constexpr uint64_t LSTM_GRAD_TILING_KEY_REGBASE_SMALL = 20000;

struct LstmGradRegbaseSmallTilingData {
    int64_t timeStep;
    int64_t batch;
    int64_t inputSize;
    int64_t hiddenSize;
    int64_t isBias;     // 1: db output present
    int64_t direction;  // 0: UNIDIRECTIONAL(forward), 1: REDIRECTIONAL(backward)
    int64_t gateOrder;  // 0: ijfo, 1: ifjo (physical slot order of w rows / dgate)
    int64_t usedCores;  // == blockDim, AIV count
    int64_t tBlock;     // timesteps staged in UB at once; 0 means "all of them" (legacy callers)
    int64_t chunkCols;  // column chunk width for dx/dw streaming (<= 64)
    int64_t mBlock;     // rows per staging block in the column phase (<= 64)
    int64_t numIChunks; // CeilDiv(inputSize, chunkCols)
    /* Batch rows staged in UB at once; 0 means "all of them" (legacy callers).
     *
     * The recurrence is independent per batch row -- h_t[b] depends only on h_{t-1}[b] -- so the
     * batch is the one axis a block can be cut on without carrying anything across the cut, which
     * is the same reason the forward makes it its only partitioning axis. Everything that scales
     * with the batch scales with THIS instead: the saved planes, dgate, the ping-pong state and the
     * staged initial state. dw and db accumulate across batch blocks the same way they accumulate
     * across time blocks, through the fp32 accumulator. */
    int64_t bBlock;
    /* Gate rows staged at once, within one gate slot; 0 means "all H of the slot" (legacy callers).
     *
     * The three buffers that scale with 4H -- the weight chunk, the dw accumulator and the output
     * staging -- are 192 KB together at hidden_size 512 with the narrowest column chunk, which is
     * the whole budget. They are the last thing that refuses a large hidden_size, and the gate axis
     * is a pure reduction axis for dx and dh_next and a pure output axis for dw, so cutting it
     * carries nothing across the cut. */
    int64_t gBlock;
    int64_t biasComponents; // 0: absent, 1: fused [4H], 2: original biases [8H]
};

namespace LstmGradRegbase {

LSTM_REGBASE_HOST_DEVICE int64_t AlignUpI64(int64_t x, int64_t a) { return (x + a - 1) / a * a; }

LSTM_REGBASE_HOST_DEVICE int64_t CeilDivI64(int64_t x, int64_t a) { return (x + a - 1) / a; }

// Byte offsets of every UB region used by the Path S kernel. All offsets 64B aligned.
// Every logical row is stored with a 32B-aligned pitch, because the regbase aligned
// vector load/store (vlds/vsts) requires 32B-aligned addresses; masks cover the H tail.
struct LstmGradRegbaseSmallUbLayout {
    // semantic constants of the layout (class-scoped to stay independent from kernel-side names)
    static constexpr int64_t GATE_NUM = 4;         // LSTM i/j/f/o gates; dgate holds one pitched H-row per gate
    static constexpr int64_t FP32_BYTES = 4;       // dgate / recurrent state / accumulators are always fp32
    static constexpr int64_t UB_ACCESS_ALIGN = 32; // bytes; aligned vlds/vsts require 32B-aligned addresses
    static constexpr int64_t REGION_ALIGN = 64;    // bytes; region bases use a stricter 64B alignment
    static constexpr int64_t PING_PONG_NUM = 2;    // dh/dc recurrent state alternates between two buffers
    static constexpr int64_t PREV_STATE_NUM = 2;   // staged final outputs: dh_prev and dc_prev
    static constexpr int64_t TAIL_PAD_BYTES = 256; // one vector register width (VL): full-VL loads may read
                                                   // up to VL-1 elements past the last region

    int64_t hAlignT; // row pitch (elements) of dtype-T [.., H] rows: AlignUp(H*dsz,32)/dsz
    int64_t hAlignF; // row pitch (elements) of fp32 [.., H] rows: AlignUp(H*4,32)/4
    // FP32 UB planes. IO remain dtype T; narrow saved states are recomputed privately.
    int64_t dyOff;
    int64_t igOff;
    int64_t jgOff;
    int64_t fgOff;
    int64_t ogOff;
    int64_t tanhOff;
    int64_t cOff;
    int64_t hOff;
    // the step just outside the block: saved state, fp32, B rows of pitch hAlignF
    int64_t initHOff;
    int64_t initCOff;
    // incoming gradients, dtype T, B rows of pitch hAlignT
    int64_t dh0Off;
    int64_t dc0Off;
    // fp32 blocks
    int64_t dgateOff; // T*B rows x 4 slots, each slot one hAlignF-pitched H-row
    int64_t dhCurOff; // 2 x B rows of pitch hAlignF (ping-pong recurrent state)
    int64_t dcCurOff; // 2 x B rows of pitch hAlignF
    // column-phase streaming buffers (pitches are 32B-aligned by construction)
    /* 4H rows x chunkCols, dtype T. Holds w[:, col0:col0+w] for whichever chunk is being walked --
     * the input columns in the dx / dw phase, and the recurrent columns the dh_next product
     * streams in one step at a time. W_hh is NOT resident: [4H, H] is O(hidden_size^2) and was
     * what capped this path. */
    int64_t wChunkOff;
    int64_t xChunkOff;   // block rows, fp32, pitch AlignUp(chunkCols * 4, 32) -- x is fp32, see dyOff
    int64_t dwAccOff;    // one gate chunk x chunkCols, fp32
    int64_t dxAccOff;    // block rows x chunkCols, fp32
    int64_t outStageOff; // max(block rows, gate chunk) rows, dtype T, pitch AlignUp(chunkCols*sizeof(T),32)
    // Persistent bias sum and rounding residual, each 4 rows of pitch hAlignF.
    int64_t dbStageOff;
    int64_t dbCompOff;
    int64_t smallStageOff; // 2*B rows of pitch hAlignT, dtype T (dh_prev | dc_prev)
    int64_t totalBytes;
    int64_t replayScratchOff; // aliases dgate before backward; three padded FP32 temporaries
    int64_t replayMaskOff;
    int64_t replayPitch;

    /* `tBlock` IS THE TIMESTEPS HELD IN UB AT ONCE, NOT THE SEQUENCE LENGTH.
     *
     * The eight resident planes and dgate are the only regions that scale with time, and they are
     * what used to cap this path: at T*B = 221 with H = 64 fp32 the planes alone want 452 KB
     * against a 248 KB budget, so every long sequence and every large batch was refused. Sizing
     * them by a BLOCK of timesteps instead lets the kernel walk the sequence in pieces -- the
     * recurrence carries dh/dc across the boundary in the ping-pong buffers it already has, and
     * dw/db accumulate into GM with atomic add.
     *
     * Passing the full timeStep here still works and reproduces the old layout exactly, which is
     * what a caller that has not been taught about blocking gets. */
    /* `batch` here is the BATCH BLOCK, not the operator's batch -- see LstmGradRegbaseSmallTilingData
     * ::bBlock. A caller that stages the whole batch passes it and gets the same layout as before. */
    LSTM_REGBASE_HOST_DEVICE void Fill(int64_t tBlock, int64_t batch, int64_t hidden, int64_t chunkCols, int64_t mBlock,
                                       int64_t dtypeSize, int64_t gBlock = 0)
    {
        const int64_t rows = tBlock * batch;
        const int64_t gates = GATE_NUM * hidden;
        hAlignT = AlignUpI64(hidden * dtypeSize, UB_ACCESS_ALIGN) / dtypeSize;
        hAlignF = AlignUpI64(hidden * FP32_BYTES, UB_ACCESS_ALIGN) / FP32_BYTES;
        const int64_t resT = AlignUpI64(rows * hAlignT * dtypeSize, REGION_ALIGN);
        const int64_t resB = AlignUpI64(batch * hAlignT * dtypeSize, REGION_ALIGN);
        /* Saved inputs are widened in UB, so these compute regions use hAlignF. */
        const int64_t resS = AlignUpI64(rows * hAlignF * FP32_BYTES, REGION_ALIGN);
        const int64_t resBS = AlignUpI64(batch * hAlignF * FP32_BYTES, REGION_ALIGN);

        int64_t off = 0;
        dyOff = off;
        off += resS;
        igOff = off;
        off += resS;
        jgOff = off;
        off += resS;
        fgOff = off;
        off += resS;
        ogOff = off;
        off += resS;
        tanhOff = off;
        off += resS;
        cOff = off;
        off += resS;
        hOff = off;
        off += resS;
        initHOff = off;
        off += resBS;
        initCOff = off;
        off += resBS;
        dh0Off = off;
        off += resB;
        dc0Off = off;
        off += resB;
        dgateOff = off;
        replayPitch = AlignUpI64(hidden * FP32_BYTES, TAIL_PAD_BYTES);
        replayScratchOff = dgateOff;
        replayMaskOff = replayScratchOff + 3 * replayPitch;
        const int64_t replayBytes = 3 * replayPitch + AlignUpI64(replayPitch / FP32_BYTES, REGION_ALIGN);
        const int64_t dgateBytes = AlignUpI64(rows * GATE_NUM * hAlignF * FP32_BYTES, REGION_ALIGN);
        // Replay finishes before ProcessChain produces dgate. Sharing storage
        // avoids adding three H-wide planes to the large-hidden UB budget.
        off += (dtypeSize != FP32_BYTES && replayBytes > dgateBytes) ? replayBytes : dgateBytes;
        dhCurOff = off;
        off += AlignUpI64(PING_PONG_NUM * batch * hAlignF * FP32_BYTES, REGION_ALIGN);
        dcCurOff = off;
        off += AlignUpI64(PING_PONG_NUM * batch * hAlignF * FP32_BYTES, REGION_ALIGN);
        /* THE ROW PITCH OF A dtype-T CHUNK IS THE 32-BYTE ROUNDING OF ITS WIDTH, NOT ITS WIDTH.
         * Every one of these buffers is filled or drained by a DataCopyPad with stride 0 on the UB
         * side, which advances by AlignUp(w * dtypeSize, 32) per row, and the vector code addresses
         * them at that same pitch. At two bytes with chunkCols 8 the width is 16 bytes and the
         * pitch is 32, so sizing by the width alone is half of what gets written. */
        const int64_t chunkPitchT = AlignUpI64(chunkCols * dtypeSize, UB_ACCESS_ALIGN);
        const int64_t chunkPitchF = AlignUpI64(chunkCols * FP32_BYTES, UB_ACCESS_ALIGN);
        /* The gate-scaled buffers hold ONE gate chunk, not all 4H rows. gBlock is bounded by the
         * slot it sits in, so a chunk never straddles two gates and the (slot, h) loops keep their
         * shape. */
        const int64_t gRows = (gBlock > 0 && gBlock < hidden) ? gBlock : hidden;
        wChunkOff = off;
        off += AlignUpI64(gRows * chunkPitchT, REGION_ALIGN);
        /* x is staged for every row of the block at once, so the gate chunks can be walked without
         * re-reading it. */
        xChunkOff = off;
        off += AlignUpI64(rows * chunkPitchF, REGION_ALIGN);
        dwAccOff = off;
        off += AlignUpI64(gRows * chunkCols * FP32_BYTES, REGION_ALIGN);
        /* dx's accumulator: the gate axis is its reduction axis, so the partial sums have to
         * survive across gate chunks, and they stay fp32 for the same reason dw's do. */
        dxAccOff = off;
        off += AlignUpI64(rows * chunkCols * FP32_BYTES, REGION_ALIGN);
        /* outStage carries two different things and has to hold the larger: a column chunk of dw
         * narrowed back down (one gate chunk's rows at chunkPitchT) and db narrowed back down
         * (GATE_NUM rows at hAlignT). The second is not bounded by chunkCols, so it needs its own
         * floor -- at chunkCols 8 and hidden 64 a gate chunk of one row would otherwise be 32 bytes
         * against the 512 db writes. */
        const int64_t outRows = (rows > gRows) ? rows : gRows;
        const int64_t outByChunk = outRows * chunkPitchT;
        const int64_t outByDb = GATE_NUM * hAlignT * dtypeSize;
        outStageOff = off;
        off += AlignUpI64((outByChunk > outByDb) ? outByChunk : outByDb, REGION_ALIGN);
        (void)mBlock;
        dbStageOff = off;
        off += AlignUpI64(GATE_NUM * hAlignF * FP32_BYTES, REGION_ALIGN);
        dbCompOff = off;
        off += AlignUpI64(GATE_NUM * hAlignF * FP32_BYTES, REGION_ALIGN);
        smallStageOff = off;
        off += AlignUpI64(PREV_STATE_NUM * batch * hAlignT * dtypeSize, REGION_ALIGN);
        off += TAIL_PAD_BYTES; // full-VL vector loads may read up to VL-1 elements past a region
        totalBytes = off;
    }
};

} // namespace LstmGradRegbase

#endif // SINGLE_LAYER_LSTM_GRAD_REGBASE_TILING_DATA_H
