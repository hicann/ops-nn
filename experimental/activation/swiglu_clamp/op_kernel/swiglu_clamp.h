/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_clamp.h
 * \brief SwigluClamp fused kernel class (single x[...,2N] input -> out[...,N]):
 *        out = silu(gate).clamp(max=limit) * up.clamp(-limit, limit)
 *        where gate = x[..., :N], up = x[..., N:] (contiguous halves of each row).
 *
 *        Compute logic is ported from the verified vllm-ascend feat/swiglustep-ascendc-fused
 *        implementation (7/7 precision cases pass on 910B). Per-row Cast splits gate/up (each
 *        row's first N / last N are contiguous, no stride API needed); the whole tile is then
 *        processed with Sigmoid/Mul/Mins/Maxs in fp32. No inter-op PipeBarrier is emitted:
 *        the vector engine serialises RAW hazards on the same tensor, and this was verified.
 */
#ifndef SWIGLU_CLAMP_H
#define SWIGLU_CLAMP_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "swiglu_clamp_tiling_data.h"
#include "swiglu_clamp_tiling_key.h"

namespace MySwigluClamp {

using namespace AscendC;

constexpr int64_t BUFFER_NUM = 2; // double buffer, hide GM<->UB latency

template <typename DATA_T> // DATA_T = half / bfloat16_t / float (DTYPE_X at build time)
class KernelSwigluClamp {
public:
    __aicore__ inline KernelSwigluClamp() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const SwigluClampTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t rows);
    __aicore__ inline void Compute(int64_t progress, int64_t rows);
    __aicore__ inline void CopyOut(int64_t progress, int64_t rows);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;   // x [tileM, 2N]
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY; // y [tileM, N]
    TBuf<QuePosition::VECCALC> gateFp32Buf;
    TBuf<QuePosition::VECCALC> upFp32Buf;
    TBuf<QuePosition::VECCALC> outFp32Buf;
    TBuf<QuePosition::VECCALC> tmpABuf; // sigmoid then silu, then silu-clamped
    TBuf<QuePosition::VECCALC> tmpBBuf; // up-clamped
    GlobalTensor<DATA_T> xGm;
    GlobalTensor<DATA_T> yGm;

    int64_t blockIdx;
    int64_t coreRowOffset;
    int64_t coreRows;
    int64_t tileM;
    int64_t N;
    int64_t formerNum;
    int64_t formerLength;
    int64_t tailNum;
    int64_t tailLength;
    float limit;
};

template <typename DATA_T>
__aicore__ inline void KernelSwigluClamp<DATA_T>::Init(GM_ADDR x, GM_ADDR y, const SwigluClampTilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");

    // 1) read tiling (row semantics: tileLength=tileM, formerLength/tailLength in rows)
    N = tilingData->N;
    tileM = tilingData->tileLength;
    formerNum = tilingData->formerNum;
    formerLength = tilingData->formerLength;
    tailNum = tilingData->tailNum;
    tailLength = tilingData->tailLength;
    limit = tilingData->limit;

    // 2) this core's row range (former cores take formerLength rows, last core takes the tail)
    blockIdx = GetBlockIdx();
    int64_t usedCoreNum = formerNum + tailNum;
    coreRows = (blockIdx == usedCoreNum - 1) ? tailLength : formerLength;
    coreRowOffset = (blockIdx < formerNum) ? (blockIdx * formerLength) : (formerNum * formerLength);

    // 3) GM tensors: x [M,2N] row-major, y [M,N] row-major (this core's row range)
    xGm.SetGlobalBuffer((__gm__ DATA_T*)x + coreRowOffset * 2 * N, coreRows * 2 * N);
    yGm.SetGlobalBuffer((__gm__ DATA_T*)y + coreRowOffset * N, coreRows * N);

    // 4) UB buffers. Per output element: inQueueX(2 buf * 2N/N * dtype) + outQueueY(2 buf * dtype)
    //    + 5 fp32 TBufs (4B each) = 32 B/out-elem for bf16/fp16 (bufferCoefficient in tiling).
    int64_t tileEle = tileM * N; // output elements per tile
    pipe.InitBuffer(inQueueX, BUFFER_NUM, tileM * 2 * N * sizeof(DATA_T));
    pipe.InitBuffer(outQueueY, BUFFER_NUM, tileEle * sizeof(DATA_T));
    pipe.InitBuffer(gateFp32Buf, tileEle * sizeof(float));
    pipe.InitBuffer(upFp32Buf, tileEle * sizeof(float));
    pipe.InitBuffer(outFp32Buf, tileEle * sizeof(float));
    pipe.InitBuffer(tmpABuf, tileEle * sizeof(float));
    pipe.InitBuffer(tmpBBuf, tileEle * sizeof(float));
}

template <typename DATA_T>
__aicore__ inline void KernelSwigluClamp<DATA_T>::Process()
{
    // empty / edge-case guard: coreRows<=0 would make tileNum=0 and underflow tailTileRows
    if (coreRows <= 0) {
        return;
    }
    int64_t tileNum = (coreRows + tileM - 1) / tileM;
    int64_t tailTileRows = coreRows - (tileNum - 1) * tileM;
    for (int64_t i = 0; i < tileNum - 1; ++i) {
        CopyIn(i, tileM);
        Compute(i, tileM);
        CopyOut(i, tileM);
    }
    // tail tile (rows < tileM)
    CopyIn(tileNum - 1, tailTileRows);
    Compute(tileNum - 1, tailTileRows);
    CopyOut(tileNum - 1, tailTileRows);
}

// ---- CopyIn: read x[rows, 2N] contiguous (one DataCopy over the full tile) ----
template <typename DATA_T>
__aicore__ inline void KernelSwigluClamp<DATA_T>::CopyIn(int64_t progress, int64_t rows)
{
    LocalTensor<DATA_T> xLocal = inQueueX.AllocTensor<DATA_T>();
    DataCopy(xLocal, xGm[progress * tileM * 2 * N], rows * 2 * N);
    inQueueX.EnQue(xLocal);
}

// ---- Compute: split gate/up per row (contiguous N-segs), then silu+clamp+mul ----
template <typename DATA_T>
__aicore__ inline void KernelSwigluClamp<DATA_T>::Compute(int64_t progress, int64_t rows)
{
    (void)progress;
    LocalTensor<DATA_T> xLocal = inQueueX.DeQue<DATA_T>();
    LocalTensor<DATA_T> yLocal = outQueueY.AllocTensor<DATA_T>();
    LocalTensor<float> tmpA = tmpABuf.Get<float>();
    LocalTensor<float> tmpB = tmpBBuf.Get<float>();
    const int64_t calEle = rows * N;

    if constexpr (std::is_same_v<DATA_T, float>) {
        // fp32: no upcast (Cast(float->float) is invalid). Each row's gate = xLocal[r*2N:r*2N+N]
        // and up = xLocal[r*2N+N:r*2N+2N] are contiguous N-segs; operate on them directly as
        // float sub-views and write yLocal. (gateFp32/upFp32/outFp32 buffers stay unused here.)
        for (int64_t r = 0; r < rows; ++r) {
            int64_t goff = r * 2 * N;
            Sigmoid(tmpA, xLocal[goff], N);         // sigmoid(gate)
            Mul(tmpA, xLocal[goff], tmpA, N);       // silu = gate * sigmoid(gate)
            Mins(tmpA, tmpA, limit, N);             // clamp silu upper bound
            Mins(tmpB, xLocal[goff + N], limit, N); // clamp up upper bound
            Maxs(tmpB, tmpB, -limit, N);            // clamp up lower bound
            Mul(yLocal[r * N], tmpA, tmpB, N);      // out = silu_c * up_c
        }
    } else {
        // bf16/fp16: upcast to fp32. Per-row Cast gate/up (contiguous N-segs) into fp32 buffers,
        // compute the whole tile at once, then Cast back.
        LocalTensor<float> gateFp32 = gateFp32Buf.Get<float>();
        LocalTensor<float> upFp32 = upFp32Buf.Get<float>();
        LocalTensor<float> outFp32 = outFp32Buf.Get<float>();
        for (int64_t r = 0; r < rows; ++r) {
            Cast(gateFp32[r * N], xLocal[r * 2 * N], RoundMode::CAST_NONE, N);
            Cast(upFp32[r * N], xLocal[r * 2 * N + N], RoundMode::CAST_NONE, N);
        }
        Sigmoid(tmpA, gateFp32, calEle);                     // sigmoid(gate)
        Mul(tmpA, gateFp32, tmpA, calEle);                   // silu = gate * sigmoid(gate)
        Mins(tmpA, tmpA, limit, calEle);                     // clamp silu upper bound
        Mins(tmpB, upFp32, limit, calEle);                   // clamp up upper bound
        Maxs(tmpB, tmpB, -limit, calEle);                    // clamp up lower bound
        Mul(outFp32, tmpA, tmpB, calEle);                    // out = silu_c * up_c
        Cast(yLocal, outFp32, RoundMode::CAST_RINT, calEle); // downcast fp32 -> bf16/fp16
    }

    inQueueX.FreeTensor(xLocal);
    outQueueY.EnQue<DATA_T>(yLocal);
}

// ---- CopyOut: write y[rows, N] contiguous ----
template <typename DATA_T>
__aicore__ inline void KernelSwigluClamp<DATA_T>::CopyOut(int64_t progress, int64_t rows)
{
    LocalTensor<DATA_T> yLocal = outQueueY.DeQue<DATA_T>();
    DataCopy(yGm[progress * tileM * N], yLocal, rows * N);
    outQueueY.FreeTensor(yLocal);
}

} // namespace MySwigluClamp
#endif // SWIGLU_CLAMP_H
