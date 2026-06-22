/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef APPLY_ADAGRAD_D_H
#define APPLY_ADAGRAD_D_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "apply_adagrad_d_tiling_data.h"
#include "apply_adagrad_d_tiling_key.h"

namespace NsApplyAdagradD {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

// T: external dtype (half / bfloat16_t / float)
// updateSlots is read at runtime from tilingData->updateSlots
template <typename T>
class ApplyAdagradD {
public:
    __aicore__ inline ApplyAdagradD() {}

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR accum, GM_ADDR lr, GM_ADDR grad,
                                GM_ADDR var_out, GM_ADDR accum_out,
                                const ApplyAdagradDTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);

private:
    TPipe pipe;

    // VECIN queues — double-buffered: var, accum, grad
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueVar;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueAccum;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueGrad;

    // VECOUT queues — double-buffered: var_out, accum_out
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueVar;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueAccum;

    // Scratch VECCALC buffers
    // For float16/bfloat16: 4 float32 scratch tensors (varF, accumF, gradF, tmpF)
    // For float32: 1 scratch tensor (tmpF, same dtype T) — reused for both grad_square and sqrtAccum
    TBuf<QuePosition::VECCALC> tmpBuf0;   // varF       (float16/bf16) OR tmpF (float32)
    TBuf<QuePosition::VECCALC> tmpBuf1;   // accumF     (float16/bf16 only)
    TBuf<QuePosition::VECCALC> tmpBuf2;   // gradF      (float16/bf16 only)
    TBuf<QuePosition::VECCALC> tmpBuf3;   // tmpF       (float16/bf16 only)

    // Global memory tensors
    GlobalTensor<T> gmVar;
    GlobalTensor<T> gmAccum;
    GlobalTensor<T> gmGrad;
    GlobalTensor<T> gmLrRaw;   // lr scalar on GM (dtype=T, shape=[1])
    GlobalTensor<T> gmVarOut;
    GlobalTensor<T> gmAccumOut;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    float    lrScalar;
    int32_t  updateSlots;  // runtime: 1=true, 0=false
};

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void ApplyAdagradD<T>::Init(
    GM_ADDR var, GM_ADDR accum, GM_ADDR lr, GM_ADDR grad,
    GM_ADDR var_out, GM_ADDR accum_out,
    const ApplyAdagradDTilingData* tilingData)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t blockIdx = AscendC::GetBlockIdx();

    this->tileDataNum  = static_cast<uint32_t>(tilingData->tileDataNum);
    this->updateSlots  = tilingData->updateSlots;
    if (this->tileDataNum == 0) {
        this->tileDataNum = 1;
    }

    uint64_t globalOffset = static_cast<uint64_t>(tilingData->bigCoreDataNum) * blockIdx;
    if (blockIdx < static_cast<uint32_t>(tilingData->tailBlockNum)) {
        this->coreDataNum = static_cast<uint32_t>(tilingData->bigCoreDataNum);
        this->tileNum     = static_cast<uint32_t>(tilingData->finalBigTileNum);
        this->tailDataNum = static_cast<uint32_t>(tilingData->bigTailDataNum);
    } else {
        this->coreDataNum = static_cast<uint32_t>(tilingData->smallCoreDataNum);
        this->tileNum     = static_cast<uint32_t>(tilingData->finalSmallTileNum);
        this->tailDataNum = static_cast<uint32_t>(tilingData->smallTailDataNum);
        globalOffset -= static_cast<uint64_t>(
            (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) *
            (blockIdx - tilingData->tailBlockNum));
    }

    uint64_t totalDataNum = static_cast<uint64_t>(tilingData->totalDataNum);
    if (globalOffset >= totalDataNum) {
        this->coreDataNum = 0;
        this->tileNum = 0;
        this->tailDataNum = 0;
    } else {
        uint64_t remaining = totalDataNum - globalOffset;
        if (static_cast<uint64_t>(this->coreDataNum) > remaining) {
            this->coreDataNum = static_cast<uint32_t>(remaining);
            this->tileNum = (this->coreDataNum + this->tileDataNum - 1) / this->tileDataNum;
            uint32_t fullTileNum = (this->tileNum == 0) ? 0 : (this->tileNum - 1);
            this->tailDataNum = this->coreDataNum - fullTileNum * this->tileDataNum;
        }
    }

    gmLrRaw.SetGlobalBuffer((__gm__ T*)lr, 1);
    gmVar.SetGlobalBuffer((__gm__ T*)var + globalOffset, this->coreDataNum);
    gmAccum.SetGlobalBuffer((__gm__ T*)accum + globalOffset, this->coreDataNum);
    gmGrad.SetGlobalBuffer((__gm__ T*)grad + globalOffset, this->coreDataNum);
    gmVarOut.SetGlobalBuffer((__gm__ T*)var_out + globalOffset, this->coreDataNum);
    gmAccumOut.SetGlobalBuffer((__gm__ T*)accum_out + globalOffset, this->coreDataNum);

    // Init VECIN/VECOUT double-buffers
    pipe.InitBuffer(inQueueVar,    BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(inQueueAccum,  BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(inQueueGrad,   BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(outQueueVar,   BUFFER_NUM, this->tileDataNum * sizeof(T));
    pipe.InitBuffer(outQueueAccum, BUFFER_NUM, this->tileDataNum * sizeof(T));

    // Init scratch buffers based on dtype (mirrors hard_swish_v2)
    if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value ||
                  AscendC::Std::is_same<T, half>::value) {
        // Need 4 float32 scratch tensors
        pipe.InitBuffer(tmpBuf0, this->tileDataNum * sizeof(float));  // varF
        pipe.InitBuffer(tmpBuf1, this->tileDataNum * sizeof(float));  // accumF
        pipe.InitBuffer(tmpBuf2, this->tileDataNum * sizeof(float));  // gradF
        pipe.InitBuffer(tmpBuf3, this->tileDataNum * sizeof(float));  // tmpF (sqrtAccum/grad_square)
    } else {
        // float32: only 1 scratch tensor needed (tmpF)
        pipe.InitBuffer(tmpBuf0, this->tileDataNum * sizeof(T));
    }

    // Read lr scalar directly from GM (no DataCopy needed)
    // CANN 8.5.0 backend does not support scalar cast for bf16/half,
    // so we manually reconstruct the float32 bit-pattern.
    if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
        // bfloat16 → float32: shift 16-bit payload to high 16 bits of fp32
        GlobalTensor<uint16_t> gmLrU16;
        gmLrU16.SetGlobalBuffer((__gm__ uint16_t*)lr, 1);
        uint16_t raw = gmLrU16.GetValue(0);
        uint32_t u32 = static_cast<uint32_t>(raw) << 16;
        float f;
        __builtin_memcpy(&f, &u32, sizeof(f));
        this->lrScalar = f;
    } else if constexpr (AscendC::Std::is_same<T, half>::value) {
        // half → float32: manual IEEE-754 fp16→fp32 bit reconstruction
        GlobalTensor<uint16_t> gmLrU16;
        gmLrU16.SetGlobalBuffer((__gm__ uint16_t*)lr, 1);
        uint16_t raw = gmLrU16.GetValue(0);
        uint32_t sign = (raw >> 15) & 0x1U;
        uint32_t exp  = (raw >> 10) & 0x1FU;
        uint32_t mant = raw & 0x3FFU;
        uint32_t u32;
        if (exp == 0) {
            if (mant == 0) {
                u32 = sign << 31;                     // zero
            } else {
                int k = 9;
                for (; k >= 0; --k) {
                    if ((mant >> k) & 1U) break;       // highest set bit
                }
                u32 = (sign << 31) |
                      (static_cast<uint32_t>(k + 103) << 23) |
                      ((mant << (23 - k)) & 0x7FFFFFU);
            }
        } else if (exp == 31) {
            u32 = (sign << 31) | (0xFFU << 23) | (mant << 13);  // inf / NaN
        } else {
            u32 = (sign << 31) |
                  (static_cast<uint32_t>(exp + 112) << 23) |
                  (mant << 13);
        }
        float f;
        __builtin_memcpy(&f, &u32, sizeof(f));
        this->lrScalar = f;
    } else {
        this->lrScalar = gmLrRaw.GetValue(0);
    }
}

// ---------------------------------------------------------------------------
// CopyIn
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void ApplyAdagradD<T>::CopyIn(int32_t progress)
{
    LocalTensor<T> varLocal   = inQueueVar.AllocTensor<T>();
    LocalTensor<T> accumLocal = inQueueAccum.AllocTensor<T>();
    LocalTensor<T> gradLocal  = inQueueGrad.AllocTensor<T>();

    DataCopy(varLocal,   gmVar  [progress * this->tileDataNum], this->processDataNum);
    DataCopy(accumLocal, gmAccum[progress * this->tileDataNum], this->processDataNum);
    DataCopy(gradLocal,  gmGrad [progress * this->tileDataNum], this->processDataNum);

    inQueueVar.EnQue(varLocal);
    inQueueAccum.EnQue(accumLocal);
    inQueueGrad.EnQue(gradLocal);
}

// ---------------------------------------------------------------------------
// CopyOut
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void ApplyAdagradD<T>::CopyOut(int32_t progress)
{
    LocalTensor<T> varOutLocal   = outQueueVar.DeQue<T>();
    LocalTensor<T> accumOutLocal = outQueueAccum.DeQue<T>();

    DataCopy(gmVarOut  [progress * this->tileDataNum], varOutLocal,   this->processDataNum);
    DataCopy(gmAccumOut[progress * this->tileDataNum], accumOutLocal, this->processDataNum);

    outQueueVar.FreeTensor(varOutLocal);
    outQueueAccum.FreeTensor(accumOutLocal);
}

// ---------------------------------------------------------------------------
// Compute
//
// float16 / bfloat16:
//   Cast inputs to float32 → compute → cast results back
//   bfloat16 back-cast uses CAST_RINT (banker's rounding, matches hard_swish_v2)
//   float16  back-cast uses CAST_ROUND
//
// float32:
//   Compute in-place using T buffers directly, no cast needed
//   tmpBuf0 reused as sqrtAccum and (when updateSlots) grad_square
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void ApplyAdagradD<T>::Compute(int32_t progress)
{
    LocalTensor<T> varLocal   = inQueueVar.DeQue<T>();
    LocalTensor<T> accumLocal = inQueueAccum.DeQue<T>();
    LocalTensor<T> gradLocal  = inQueueGrad.DeQue<T>();

    LocalTensor<T> varOutLocal   = outQueueVar.AllocTensor<T>();
    LocalTensor<T> accumOutLocal = outQueueAccum.AllocTensor<T>();

    uint32_t cnt = this->processDataNum;

    if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value ||
                  AscendC::Std::is_same<T, half>::value) {
        // --- float16 / bfloat16 path ---
        LocalTensor<float> varF   = tmpBuf0.Get<float>();
        LocalTensor<float> accumF = tmpBuf1.Get<float>();
        LocalTensor<float> gradF  = tmpBuf2.Get<float>();
        LocalTensor<float> tmpF   = tmpBuf3.Get<float>();

        // Cast inputs to float32
        Cast(varF,   varLocal,   RoundMode::CAST_NONE, cnt);
        PipeBarrier<PIPE_V>();
        Cast(accumF, accumLocal, RoundMode::CAST_NONE, cnt);
        PipeBarrier<PIPE_V>();
        Cast(gradF,  gradLocal,  RoundMode::CAST_NONE, cnt);
        PipeBarrier<PIPE_V>();

        if (this->updateSlots) {
            // grad_square = gradF * gradF → tmpF
            Mul(tmpF, gradF, gradF, cnt);
            PipeBarrier<PIPE_V>();
            // accumF += grad_square
            Add(accumF, accumF, tmpF, cnt);
            PipeBarrier<PIPE_V>();
        }

        // sqrtAccum → tmpF
        Sqrt(tmpF, accumF, cnt);
        PipeBarrier<PIPE_V>();

        // lr_grad = lrScalar * gradF → gradF
        Muls(gradF, gradF, this->lrScalar, cnt);
        PipeBarrier<PIPE_V>();
        // update = lr_grad / sqrtAccum → gradF
        Div(gradF, gradF, tmpF, cnt);
        PipeBarrier<PIPE_V>();
        // varF -= update
        Sub(varF, varF, gradF, cnt);
        PipeBarrier<PIPE_V>();

        // TBE adds vadds(var, 0) and vadds(accum, 0) before cast back
        Adds(varF,   varF,   static_cast<float>(0), cnt);
        PipeBarrier<PIPE_V>();
        Adds(accumF, accumF, static_cast<float>(0), cnt);
        PipeBarrier<PIPE_V>();

        // Cast results back to T
        if constexpr (AscendC::Std::is_same<T, bfloat16_t>::value) {
            Cast(varOutLocal,   varF,   RoundMode::CAST_RINT, cnt);
            PipeBarrier<PIPE_V>();
            Cast(accumOutLocal, accumF, RoundMode::CAST_RINT, cnt);
        } else {
            // float16
            Cast(varOutLocal,   varF,   RoundMode::CAST_ROUND, cnt);
            PipeBarrier<PIPE_V>();
            Cast(accumOutLocal, accumF, RoundMode::CAST_ROUND, cnt);
        }
        PipeBarrier<PIPE_V>();

    } else {
        // --- float32 path ---
        // Compute in-place with T (=float) buffers; tmpBuf0 used as scratch
        LocalTensor<T> tmpF = tmpBuf0.Get<T>();

        if (this->updateSlots) {
            // grad_square = gradLocal * gradLocal → tmpF
            Mul(tmpF, gradLocal, gradLocal, cnt);
            PipeBarrier<PIPE_V>();
            // accumLocal += grad_square
            Add(accumLocal, accumLocal, tmpF, cnt);
            PipeBarrier<PIPE_V>();
        }

        // sqrtAccum → tmpF
        Sqrt(tmpF, accumLocal, cnt);
        PipeBarrier<PIPE_V>();
        // lr_grad = lrScalar * gradLocal → gradLocal
        Muls(gradLocal, gradLocal, static_cast<T>(this->lrScalar), cnt);
        PipeBarrier<PIPE_V>();
        // update = lr_grad / sqrtAccum → gradLocal
        Div(gradLocal, gradLocal, tmpF, cnt);
        PipeBarrier<PIPE_V>();
        // varLocal -= update
        Sub(varLocal, varLocal, gradLocal, cnt);
        PipeBarrier<PIPE_V>();

        // Move results to output tensors (add 0 covers full cnt regardless of size)
        Adds(varOutLocal,   varLocal,   static_cast<T>(0), cnt);
        PipeBarrier<PIPE_V>();
        Adds(accumOutLocal, accumLocal, static_cast<T>(0), cnt);
    }

    outQueueVar.EnQue<T>(varOutLocal);
    outQueueAccum.EnQue<T>(accumOutLocal);

    inQueueVar.FreeTensor(varLocal);
    inQueueAccum.FreeTensor(accumLocal);
    inQueueGrad.FreeTensor(gradLocal);
}

// ---------------------------------------------------------------------------
// Process
// ---------------------------------------------------------------------------
template <typename T>
__aicore__ inline void ApplyAdagradD<T>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount; i++) {
        if (i == loopCount - 1) {
            this->processDataNum = this->tailDataNum;
        }
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}

} // namespace NsApplyAdagradD
#endif // APPLY_ADAGRAD_D_H
