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
 * \file nonzero.h
 * \brief NonZero AscendC kernel — vector compare + UB mask word-scan, all dtypes
 *
 * Algorithm (dtype is compile-time D_T_X, fp32/fp16/bf16/int32):
 *   - Each core scans its contiguous [rowsPerCore x cols] slice in TILE_ELEMS-wide
 *     tiles: DataCopyPad (GM->UB) -> (Cast to float for bf16/int32) -> Compares
 *     against 0 (vector, CMPMODE::NE, bit-packed uint8 mask) -> word-scan of the
 *     UB mask (uint32 words, least-set-bit extraction), emitting (row, col) index
 *     pairs. The scan cost tracks nnz, not the tile width — the old per-bit
 *     GetValue loop was the bottleneck on large tensors. No scalar per-element GM
 *     reads (the older xGm.GetValue loop was bandwidth-bound).
 *   - bf16 and int32 have no selectable setcc on the backend, so both are Cast to
 *     float first (int32 NE 0 compares as float NE 0.0f); fp32/fp16 compare
 *     directly (SDK Compares count-form: dst uint8_t, src float / half).
 *   - Pairs are staged in UB in PAIR_BATCH-granularity batches and flushed to GM
 *     with a single DataCopyPad per batch (4 int32 hi/lo words per pair), instead
 *     of one 32B DataCopyPad per 2 pairs — the per-write overhead dominated dense
 *     scans. (row, col) are advanced incrementally per emitted pair (col += gap,
 *     single div/mod wrap) instead of a 64-bit div/mod per nnz over the absolute
 *     linear index.
 *   - Where the pairs land depends on the build:
 *       framework build (NONZERO_USE_TILING_KEY): directly to the output tensor
 *         y at packed int32-word offset (single core; aclnn contract requires a
 *         contiguous [num_nonzero, 2] buffer and there is no host gather).
 *       custom direct-launch build: to the per-core WORKSPACE pair region; a
 *         second kernel (nonzero_compact) gathers the per-core regions into the
 *         contiguous output, so the host only D2Hs the actual data.
 *   - Custom-build workspace layout (int32):
 *       counts: usedCoreNum * 8, count for core i at [i * 8] (contiguous).
 *       pairs:  usedCoreNum * wsStride, core i pairs at [i * wsStride + k * 4]
 *               where wsStride = 8 * ceil(rowsPerCore*cols / 2) (worst case).
 *   - Host reads the counts, computes per-core prefixes, gathers the pairs.
 */
#ifndef __NONZERO_H__
#define __NONZERO_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "nonzero_tiling_data.h"
#ifdef NONZERO_USE_TILING_KEY
// Tiling key (dtype-composed template dispatch) is only needed by the framework
// build (ops-nn repo kernel entry). Custom direct-launch builds do not have the
// host-side tiling headers in the kernel include path, so the include is guarded.
#include "nonzero_tiling_key.h"
#endif

#include <type_traits>

namespace NsNonzero {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t ALIGN_BYTES = 32;
// Vector tile: 8192 elems = 16KB (bf16/half) / 32KB (fp32/int32), 256B-aligned.
// Large tiles amortize the per-tile DataCopyPad + barrier cost; the mask is
// scanned as uint32 words, so the scan cost scales with nnz, not tile width.
constexpr int32_t TILE_ELEMS = 8192;
// Compares outputs a bit-packed mask (1 bit per element); round bytes to 32B.
constexpr int32_t MASK_BYTES = ((TILE_ELEMS / 8) + 31) / 32 * 32;
// Pairs staged in UB before a single DataCopyPad to GM: PAIR_BATCH*16 bytes.
constexpr int32_t PAIR_BATCH = 512;

template <typename T>
class Nonzero {
public:
    __aicore__ inline Nonzero(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR workspace, GM_ADDR output, const NonzeroTilingData* t);
    __aicore__ inline void Process(int64_t outBaseOffset = 0);

private:
    __aicore__ inline void FlushStaged(int32_t numWords, int64_t& wordPos);

    TPipe pipe;
    TQue<TPosition::VECOUT, BUFFER_NUM> wsQueue; // int32 view of the per-core header
    TBuf<TPosition::VECIN> dataBuf;
    TBuf<TPosition::VECCALC> castBuf;
    TBuf<TPosition::VECCALC> maskBuf;
    TBuf<TPosition::VECIN> batchBuf; // staged (row,col) int32 words
    GlobalTensor<T> xGm;
    GlobalTensor<int32_t> wsGm;    // counts (custom) / workspace header (framework)
    GlobalTensor<int32_t> pairGm;  // custom: per-core pair regions
    GlobalTensor<int32_t> outGm32; // int32 view of the output tensor (framework)
    uint32_t blockIdx;
    int64_t totalRows, cols, rowsPerCore, rowStride, wsStride;
    int64_t wsBase;   // count base slot index
    int64_t pairBase; // custom: this core's pair-region base slot index
};

// --- Implementation (inlined for single-source kernel build) ---

template <typename T>
__aicore__ inline void Nonzero<T>::Init(GM_ADDR x, GM_ADDR workspace, GM_ADDR output, const NonzeroTilingData* t)
{
    this->blockIdx = GetBlockIdx();
    this->totalRows = t->totalRows;
    this->cols = t->cols;
    this->rowsPerCore = t->rowsPerCore;
    this->rowStride = t->rowStride;
    this->wsStride = t->wsStride;
#ifdef NONZERO_USE_TILING_KEY
    // Framework build: workspace header (count) per core, pairs go straight to y.
    this->wsBase = (int64_t)blockIdx * wsStride;
    xGm.SetGlobalBuffer((__gm__ T*)x);
    wsGm.SetGlobalBuffer((__gm__ int32_t*)workspace);
    outGm32.SetGlobalBuffer((__gm__ int32_t*)output);
#else
    // Custom build: workspace = contiguous per-core counts; output = per-core
    // pair regions (output tensor is never written, see file header).
    this->wsBase = (int64_t)blockIdx * 8;
    this->pairBase = (int64_t)blockIdx * wsStride;
    xGm.SetGlobalBuffer((__gm__ T*)x);
    wsGm.SetGlobalBuffer((__gm__ int32_t*)workspace);
    pairGm.SetGlobalBuffer((__gm__ int32_t*)output);
#endif
    pipe.InitBuffer(wsQueue, BUFFER_NUM, ALIGN_BYTES); // 8 int32s = 32 bytes
    pipe.InitBuffer(dataBuf, TILE_ELEMS * sizeof(T));
    pipe.InitBuffer(maskBuf, MASK_BYTES);
    pipe.InitBuffer(batchBuf, PAIR_BATCH * 4 * sizeof(int32_t));
    if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, int32_t>::value) {
        pipe.InitBuffer(castBuf, TILE_ELEMS * sizeof(float));
    }
}

// Write numWords staged int32 words (numWords/4 pairs) to this core's output
// region with one DataCopyPad. The byte-based DataCopyExtParams is the SDK
// canonical scalar-to-GM pattern (masked_select_v3); plain DataCopy(gm, ub,
// count) drops writes nondeterministically here.
template <typename T>
__aicore__ inline void Nonzero<T>::FlushStaged(int32_t numWords, int64_t& wordPos)
{
    LocalTensor<int32_t> ub = batchBuf.Get<int32_t>();
    PipeBarrier<PIPE_ALL>();
    DataCopyExtParams cp{1, static_cast<uint32_t>(numWords * 4), 0, 0, 0};
#ifdef NONZERO_USE_TILING_KEY
    DataCopyPad<int32_t>(outGm32[wordPos], ub[0], cp);
#else
    DataCopyPad<int32_t>(pairGm[pairBase + wordPos], ub[0], cp);
#endif
    PipeBarrier<PIPE_ALL>();
    wordPos += numWords;
}

template <typename T>
__aicore__ inline void Nonzero<T>::Process(int64_t outBaseOffset)
{
    (void)outBaseOffset; // unused: wordPos is always core-local (0 origin)
    int64_t startRow = (int64_t)blockIdx * rowsPerCore;
    int64_t endRow = startRow + rowsPerCore;
    if (endRow > totalRows)
        endRow = totalRows;

    int64_t rowsThisCore = endRow - startRow;
    int64_t totalNonzeros = 0;
    int64_t wordPos = 0; // int32 words staged to this core's region
    int32_t staged = 0;  // words currently in batchBuf

    // Idle cores (rows beyond the last) still must zero their workspace count so
    // the host's per-core read sees a real value (workspace is not pre-zeroed).
    if (rowsThisCore <= 0) {
        LocalTensor<int32_t> cntLocal = wsQueue.AllocTensor<int32_t>();
        cntLocal.SetValue(0, 0);
        PipeBarrier<PIPE_ALL>();
        wsQueue.EnQue<int32_t>(cntLocal);
        LocalTensor<int32_t> cntDeq = wsQueue.DeQue<int32_t>();
        DataCopy(wsGm[wsBase], cntDeq, 8);
        wsQueue.FreeTensor(cntDeq);
        return;
    }

    // Vector path for every dtype: DataCopyPad tile -> (Cast) -> Compares mask
    // -> word-scan. DataCopyPad zero-extends the tail, and bits beyond cur are
    // filtered, so padded mask bits are never emitted. The contiguous core slice
    // is processed in TILE_ELEMS-wide tiles; (row, col) advance incrementally.
    int64_t baseOffset = startRow * rowStride;
    int64_t totalElems = rowsThisCore * cols;
    int64_t prevG = baseOffset; // last emitted pair's absolute linear index
    int64_t row = startRow;
    int64_t col = 0;
    LocalTensor<int32_t> batchLocal = batchBuf.Get<int32_t>();
    for (int64_t off = 0; off < totalElems; off += TILE_ELEMS) {
        int32_t cur = static_cast<int32_t>(totalElems - off > TILE_ELEMS ? TILE_ELEMS : totalElems - off);
        LocalTensor<T> dataLocal = dataBuf.Get<T>();
        DataCopyParams cp;
        cp.blockCount = 1;
        cp.blockLen = static_cast<uint16_t>(cur * sizeof(T));
        cp.srcStride = 0;
        cp.dstStride = 0;
        DataCopyPad(dataLocal, xGm[baseOffset + off], cp, {false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
        LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();
        if constexpr (std::is_same<T, bfloat16_t>::value || std::is_same<T, int32_t>::value) {
            // bf16/int32: no selectable setcc on the backend -> cast to float first.
            LocalTensor<float> castLocal = castBuf.Get<float>();
            Cast(castLocal, dataLocal, RoundMode::CAST_NONE, TILE_ELEMS);
            Compares(maskLocal, castLocal, 0.0f, CMPMODE::NE, TILE_ELEMS);
        } else {
            // float / half compare directly against 0.
            Compares(maskLocal, dataLocal, (T)0.0f, CMPMODE::NE, TILE_ELEMS);
        }
        PipeBarrier<PIPE_ALL>();
        // Mask is bit-packed 1 bit/element. Scan as uint32 words and walk only
        // the set bits (least-set-bit extraction), so the cost tracks the number
        // of non-zeros rather than the tile width. Bits beyond cur are filtered.
        LocalTensor<uint32_t> maskWords = maskBuf.Get<uint32_t>();
        int32_t nWords = (cur + 31) / 32;
        for (int32_t wIdx = 0; wIdx < nWords; wIdx++) {
            uint32_t w = maskWords.GetValue(wIdx);
            while (w != 0) {
                int32_t bit = 0;
                uint32_t tmp = w;
                while ((tmp & 1) == 0) {
                    tmp >>= 1;
                    bit++;
                }
                int32_t i = wIdx * 32 + bit;
                if (i < cur) {
                    // Advance (row, col) by the column gap from the last emitted
                    // pair; single div/mod wrap keeps this O(1) regardless of the
                    // gap size (the while-loop equivalent blows up for sparse
                    // inputs with small cols).
                    int64_t g = baseOffset + off + i;
                    col += g - prevG;
                    prevG = g;
                    row += col / cols;
                    col %= cols;
                    totalNonzeros++;
                    int32_t b = staged;
                    batchLocal.SetValue(b + 0, (int32_t)(row & 0xFFFFFFFF));
                    batchLocal.SetValue(b + 1, (int32_t)(row >> 32));
                    batchLocal.SetValue(b + 2, (int32_t)(col & 0xFFFFFFFF));
                    batchLocal.SetValue(b + 3, (int32_t)(col >> 32));
                    staged += 4;
                    if (staged == PAIR_BATCH * 4) {
                        FlushStaged(staged, wordPos);
                        staged = 0;
                        batchLocal = batchBuf.Get<int32_t>();
                    }
                }
                w &= w - 1;
            }
        }
    }

    PipeBarrier<PIPE_ALL>();
    if (staged > 0) {
        FlushStaged(staged, wordPos);
    }

    // Per-core count (slot 0 of the 8-int32 header).
    LocalTensor<int32_t> cntLocal = wsQueue.AllocTensor<int32_t>();
    cntLocal.SetValue(0, (int32_t)totalNonzeros);
    PipeBarrier<PIPE_ALL>();
    wsQueue.EnQue<int32_t>(cntLocal);
    LocalTensor<int32_t> cntDeq = wsQueue.DeQue<int32_t>();
    DataCopy(wsGm[wsBase], cntDeq, 8);
    wsQueue.FreeTensor(cntDeq);
    // NOTE: no SyncAll() here — it is a chip-wide cross-core barrier and
    // deadlocks even at full blockDim on this platform (probe-verified).
}

} // namespace NsNonzero

#endif // __NONZERO_H__
