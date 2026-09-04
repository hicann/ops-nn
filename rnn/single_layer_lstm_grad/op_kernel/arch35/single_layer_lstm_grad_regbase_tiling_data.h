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
 *   - AIV-only, no cube, no matmul lib, no cross-core sync, workspace = 0.
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
    int64_t chunkCols;  // column chunk width for dx/dw streaming (<= 64)
    int64_t mBlock;     // rows per staging block in the column phase (<= 64)
    int64_t numIChunks; // CeilDiv(inputSize, chunkCols)
    int64_t reserved0;
};

namespace LstmGradRegbase {

LSTM_REGBASE_HOST_DEVICE int64_t AlignUpI64(int64_t x, int64_t a) { return (x + a - 1) / a * a; }

LSTM_REGBASE_HOST_DEVICE int64_t CeilDivI64(int64_t x, int64_t a) { return (x + a - 1) / a; }

// Byte offsets of every UB region used by the Path S kernel. All offsets 64B aligned.
// Every logical row is stored with a 32B-aligned pitch, because the regbase aligned
// vector load/store (vlds/vsts) requires 32B-aligned addresses; masks cover the H tail.
struct LstmGradRegbaseSmallUbLayout {
    int64_t hAlignT; // row pitch (elements) of dtype-T [.., H] rows: AlignUp(H*dsz,32)/dsz
    int64_t hAlignF; // row pitch (elements) of fp32 [.., H] rows: AlignUp(H*4,32)/4
    // resident inputs, dtype T, T*B rows of pitch hAlignT
    int64_t dyOff;
    int64_t igOff;
    int64_t jgOff;
    int64_t fgOff;
    int64_t ogOff;
    int64_t tanhOff;
    int64_t cOff;
    int64_t hOff;
    // resident inputs, dtype T, B rows of pitch hAlignT
    int64_t initHOff;
    int64_t initCOff;
    int64_t dh0Off;
    int64_t dc0Off;
    // w[:, I:I+H] resident, dtype T, 4H rows of pitch hAlignT
    int64_t whOff;
    // fp32 blocks
    int64_t dgateOff; // T*B rows x 4 slots, each slot one hAlignF-pitched H-row
    int64_t dhCurOff; // 2 x B rows of pitch hAlignF (ping-pong recurrent state)
    int64_t dcCurOff; // 2 x B rows of pitch hAlignF
    // column-phase streaming buffers (pitches are 32B-aligned by construction)
    int64_t wChunkOff;     // 4H rows x chunkCols, dtype T
    int64_t xChunkOff;     // mBlock rows x chunkCols, dtype T
    int64_t dwAccOff;      // 4H rows x chunkCols, fp32
    int64_t outStageOff;   // max(mBlock, 4H) rows x chunkCols, dtype T
    int64_t smallStageOff; // (4 + 2*B) rows of pitch hAlignT, dtype T (db slots | dh_prev | dc_prev)
    int64_t totalBytes;

    LSTM_REGBASE_HOST_DEVICE void Fill(int64_t timeStep, int64_t batch, int64_t hidden, int64_t chunkCols,
                                       int64_t mBlock, int64_t dtypeSize)
    {
        const int64_t kAlign = 64;
        const int64_t rows = timeStep * batch;
        const int64_t gates = 4 * hidden;
        hAlignT = AlignUpI64(hidden * dtypeSize, 32) / dtypeSize;
        hAlignF = AlignUpI64(hidden * 4, 32) / 4;
        const int64_t resT = AlignUpI64(rows * hAlignT * dtypeSize, kAlign);
        const int64_t resB = AlignUpI64(batch * hAlignT * dtypeSize, kAlign);

        int64_t off = 0;
        dyOff = off;
        off += resT;
        igOff = off;
        off += resT;
        jgOff = off;
        off += resT;
        fgOff = off;
        off += resT;
        ogOff = off;
        off += resT;
        tanhOff = off;
        off += resT;
        cOff = off;
        off += resT;
        hOff = off;
        off += resT;
        initHOff = off;
        off += resB;
        initCOff = off;
        off += resB;
        dh0Off = off;
        off += resB;
        dc0Off = off;
        off += resB;
        whOff = off;
        off += AlignUpI64(gates * hAlignT * dtypeSize, kAlign);
        dgateOff = off;
        off += AlignUpI64(rows * 4 * hAlignF * 4, kAlign);
        dhCurOff = off;
        off += AlignUpI64(2 * batch * hAlignF * 4, kAlign);
        dcCurOff = off;
        off += AlignUpI64(2 * batch * hAlignF * 4, kAlign);
        wChunkOff = off;
        off += AlignUpI64(gates * chunkCols * dtypeSize, kAlign);
        xChunkOff = off;
        off += AlignUpI64(mBlock * chunkCols * dtypeSize, kAlign);
        dwAccOff = off;
        off += AlignUpI64(gates * chunkCols * 4, kAlign);
        const int64_t outRows = (mBlock > gates) ? mBlock : gates;
        outStageOff = off;
        off += AlignUpI64(outRows * chunkCols * dtypeSize, kAlign);
        smallStageOff = off;
        off += AlignUpI64((4 + 2 * batch) * hAlignT * dtypeSize, kAlign);
        off += 256; // tail pad: full-VL vector loads may read up to VL-1 elements past a region
        totalBytes = off;
    }
};

} // namespace LstmGradRegbase

#endif // SINGLE_LAYER_LSTM_GRAD_REGBASE_TILING_DATA_H
