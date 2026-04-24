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
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */
/*!
 * \file apply_adadelta.h
 * \brief ApplyAdadelta kernel class definition (arch35)
 *
 * Template parameters:
 *   - T:           Data type (float / half)
 *   - BUFFER_MODE: Buffer mode (0=single buffer, 1=double buffer)
 *
 * Adadelta update formula (4 steps):
 *   Step1: accum_new      = rho * accum + (1-rho) * grad^2
 *   Step2: update         = sqrt(accum_update + eps) / sqrt(accum_new + eps) * grad
 *   Step3: var_new        = var - lr * update
 *   Step4: accum_update_new = rho * accum_update + (1-rho) * update^2
 */
#ifndef APPLY_ADADELTA_H
#define APPLY_ADADELTA_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "apply_adadelta_tiling_data.h"
#include "apply_adadelta_tiling_key.h"

namespace NsApplyAdadelta {

using namespace AscendC;

template <typename T, int BUFFER_MODE>
class ApplyAdadelta {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;
    static constexpr bool IS_FP16 = std::is_same_v<T, half>;

public:
    __aicore__ inline ApplyAdadelta() {}

    __aicore__ inline void Init(
        GM_ADDR var, GM_ADDR accum, GM_ADDR accumUpdate, GM_ADDR grad,
        GM_ADDR varOut, GM_ADDR accumOut, GM_ADDR accumUpdateOut,
        const ApplyAdadeltaTilingData* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe;
    // 4 input queues
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueVar;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueAccum;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueAccumUpdate;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueGrad;
    // 3 output queues
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueVar;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueAccum;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueAccumUpdate;
    // FP32 path: 2 temporary buffers for intermediate computation
    TBuf<QuePosition::VECCALC> tmpBuf1;  // tmp1: grad^2 / den / updateSq etc.
    TBuf<QuePosition::VECCALC> tmpBuf2;  // tmp2: num / update / lr*update etc.
    // FP16 path: 6 fp32 TBuf for Cast workspace + computation
    // (4 for Cast-up workspace: var/accum/accumUpdate/grad fp32 copies,
    //  2 for intermediate computation: tmp1/tmp2)
    TBuf<QuePosition::VECCALC> fp32VarBuf;
    TBuf<QuePosition::VECCALC> fp32AccumBuf;
    TBuf<QuePosition::VECCALC> fp32AccumUpdateBuf;
    TBuf<QuePosition::VECCALC> fp32GradBuf;

    GlobalTensor<T> varGm, accumGm, accumUpdateGm, gradGm;
    GlobalTensor<T> varOutGm, accumOutGm, accumUpdateOutGm;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    float lr_ = 0.0f;
    float rho_ = 0.0f;
    float oneMinusRho_ = 1.0f;
    float eps_ = 1e-6f;
};

template <typename T, int BUFFER_MODE>
__aicore__ inline void ApplyAdadelta<T, BUFFER_MODE>::Init(
    GM_ADDR var, GM_ADDR accum, GM_ADDR accumUpdate, GM_ADDR grad,
    GM_ADDR varOut, GM_ADDR accumOut, GM_ADDR accumUpdateOut,
    const ApplyAdadeltaTilingData* tilingData)
{
    // Compute per-core element range
    int64_t remain = tilingData->totalNum - tilingData->blockFactor * GetBlockIdx();
    blockLength_ = (remain > tilingData->blockFactor) ? tilingData->blockFactor : remain;
    ubLength_ = tilingData->ubFactor;

    // Read scalar parameters from TilingData
    lr_ = tilingData->lr;
    rho_ = tilingData->rho;
    eps_ = tilingData->epsilon;
    oneMinusRho_ = tilingData->oneMinusRho;

    // Set up GM tensors with per-core offset
    int64_t off = tilingData->blockFactor * GetBlockIdx();
    varGm.SetGlobalBuffer((__gm__ T*)var + off, blockLength_);
    accumGm.SetGlobalBuffer((__gm__ T*)accum + off, blockLength_);
    accumUpdateGm.SetGlobalBuffer((__gm__ T*)accumUpdate + off, blockLength_);
    gradGm.SetGlobalBuffer((__gm__ T*)grad + off, blockLength_);
    varOutGm.SetGlobalBuffer((__gm__ T*)varOut + off, blockLength_);
    accumOutGm.SetGlobalBuffer((__gm__ T*)accumOut + off, blockLength_);
    accumUpdateOutGm.SetGlobalBuffer((__gm__ T*)accumUpdateOut + off, blockLength_);

    // Initialize queues
    pipe.InitBuffer(inQueVar, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(inQueAccum, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(inQueAccumUpdate, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(inQueGrad, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outQueVar, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outQueAccum, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outQueAccumUpdate, BUFFER_NUM, ubLength_ * sizeof(T));

    // Initialize temporary buffers based on dtype path
    if constexpr (IS_FP16) {
        // FP16 path: 4 fp32 Cast workspace + 2 fp32 tmp = 6 fp32 TBuf
        pipe.InitBuffer(fp32VarBuf,         ubLength_ * sizeof(float));
        pipe.InitBuffer(fp32AccumBuf,       ubLength_ * sizeof(float));
        pipe.InitBuffer(fp32AccumUpdateBuf, ubLength_ * sizeof(float));
        pipe.InitBuffer(fp32GradBuf,        ubLength_ * sizeof(float));
        pipe.InitBuffer(tmpBuf1,            ubLength_ * sizeof(float));
        pipe.InitBuffer(tmpBuf2,            ubLength_ * sizeof(float));
    } else {
        // FP32 path: 2 fp32 TBuf for intermediate computation
        pipe.InitBuffer(tmpBuf1, ubLength_ * sizeof(float));
        pipe.InitBuffer(tmpBuf2, ubLength_ * sizeof(float));
    }
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void ApplyAdadelta<T, BUFFER_MODE>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> v = inQueVar.template AllocTensor<T>();
    LocalTensor<T> a = inQueAccum.template AllocTensor<T>();
    LocalTensor<T> au = inQueAccumUpdate.template AllocTensor<T>();
    LocalTensor<T> g = inQueGrad.template AllocTensor<T>();

    DataCopyExtParams cp;
    cp.blockCount = 1;
    cp.blockLen = currentNum * sizeof(T);
    cp.srcStride = 0;
    cp.dstStride = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    int64_t gmOff = progress * ubLength_;
    DataCopyPad(v, varGm[gmOff], cp, padParams);
    DataCopyPad(a, accumGm[gmOff], cp, padParams);
    DataCopyPad(au, accumUpdateGm[gmOff], cp, padParams);
    DataCopyPad(g, gradGm[gmOff], cp, padParams);

    inQueVar.EnQue(v);
    inQueAccum.EnQue(a);
    inQueAccumUpdate.EnQue(au);
    inQueGrad.EnQue(g);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void ApplyAdadelta<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    LocalTensor<T> v = inQueVar.template DeQue<T>();
    LocalTensor<T> a = inQueAccum.template DeQue<T>();
    LocalTensor<T> au = inQueAccumUpdate.template DeQue<T>();
    LocalTensor<T> g = inQueGrad.template DeQue<T>();

    LocalTensor<T> vOut = outQueVar.template AllocTensor<T>();
    LocalTensor<T> aOut = outQueAccum.template AllocTensor<T>();
    LocalTensor<T> auOut = outQueAccumUpdate.template AllocTensor<T>();

    // Get tmp buffers
    LocalTensor<float> tmp1 = tmpBuf1.Get<float>();
    LocalTensor<float> tmp2 = tmpBuf2.Get<float>();

    if constexpr (IS_FP16) {
        // FP16 path: Cast up to fp32 -> compute -> Cast down to fp16
        // Get fp32 workspace buffers
        LocalTensor<float> vF  = fp32VarBuf.Get<float>();
        LocalTensor<float> aF  = fp32AccumBuf.Get<float>();
        LocalTensor<float> auF = fp32AccumUpdateBuf.Get<float>();
        LocalTensor<float> gF  = fp32GradBuf.Get<float>();

        // Cast up: half -> fp32
        Cast(vF,  v,  RoundMode::CAST_NONE, currentNum);
        Cast(aF,  a,  RoundMode::CAST_NONE, currentNum);
        Cast(auF, au, RoundMode::CAST_NONE, currentNum);
        Cast(gF,  g,  RoundMode::CAST_NONE, currentNum);

        // Step1: accum_new = accum * rho + grad^2 * (1 - rho)
        Mul(tmp1, gF, gF, currentNum);               // tmp1 = grad^2
        Muls(tmp1, tmp1, oneMinusRho_, currentNum);   // tmp1 = grad^2 * (1-rho)
        Muls(aF, aF, rho_, currentNum);               // aF = accum * rho
        Add(aF, aF, tmp1, currentNum);                // aF = accum_new

        // Step2: update = sqrt(au_old + eps) / sqrt(accum_new + eps) * grad
        Adds(tmp1, auF, eps_, currentNum);            // tmp1 = au_old + eps
        Sqrt(tmp1, tmp1, currentNum);                 // tmp1 = sqrt(au_old + eps) = numerator
        Adds(tmp2, aF, eps_, currentNum);             // tmp2 = accum_new + eps
        Sqrt(tmp2, tmp2, currentNum);                 // tmp2 = sqrt(accum_new + eps) = denominator
        Div(tmp1, tmp1, tmp2, currentNum);            // tmp1 = ratio
        Mul(tmp1, tmp1, gF, currentNum);              // tmp1 = update

        // Step3: var_new = var - lr * update
        Muls(tmp2, tmp1, lr_, currentNum);            // tmp2 = lr * update
        Sub(vF, vF, tmp2, currentNum);                // vF = var_new

        // Step4: accum_update_new = au_old * rho + update^2 * (1 - rho)
        Mul(tmp2, tmp1, tmp1, currentNum);            // tmp2 = update^2
        Muls(tmp2, tmp2, oneMinusRho_, currentNum);   // tmp2 = update^2 * (1-rho)
        Muls(auF, auF, rho_, currentNum);             // auF = au_old * rho
        Add(auF, auF, tmp2, currentNum);              // auF = accum_update_new

        // Cast down: fp32 -> half
        Cast(vOut,  vF,  RoundMode::CAST_RINT, currentNum);
        Cast(aOut,  aF,  RoundMode::CAST_RINT, currentNum);
        Cast(auOut, auF, RoundMode::CAST_RINT, currentNum);
    } else {
        // FP32 path: T == float, so v/a/au/g/vOut/aOut/auOut are already
        // LocalTensor<float>. No Cast needed; operate directly on them.

        // Step1: accum_new = rho * accum + (1-rho) * grad^2
        Mul(tmp1, g, g, currentNum);                  // tmp1 = grad^2
        Muls(tmp1, tmp1, oneMinusRho_, currentNum);   // tmp1 = grad^2 * (1-rho)
        Muls(aOut, a, rho_, currentNum);              // aOut = accum * rho
        Add(aOut, aOut, tmp1, currentNum);            // aOut = accum*rho + grad^2*(1-rho)

        // Step2: update = sqrt(accum_update + eps) / sqrt(accum_new + eps) * grad
        Adds(tmp1, au, eps_, currentNum);             // tmp1 = accum_update + eps
        Sqrt(tmp1, tmp1, currentNum);                 // tmp1 = sqrt(accum_update + eps) = numerator
        Adds(tmp2, aOut, eps_, currentNum);           // tmp2 = accum_new + eps
        Sqrt(tmp2, tmp2, currentNum);                 // tmp2 = sqrt(accum_new + eps) = denominator
        Div(tmp1, tmp1, tmp2, currentNum);            // tmp1 = ratio = num / den
        Mul(tmp1, tmp1, g, currentNum);               // tmp1 = update = ratio * grad

        // Step3: var_new = var - lr * update
        Muls(tmp2, tmp1, lr_, currentNum);            // tmp2 = lr * update
        Sub(vOut, v, tmp2, currentNum);               // vOut = var - lr * update

        // Step4: accum_update_new = rho * accum_update + (1-rho) * update^2
        Mul(tmp2, tmp1, tmp1, currentNum);            // tmp2 = update^2
        Muls(tmp2, tmp2, oneMinusRho_, currentNum);   // tmp2 = update^2 * (1-rho)
        Muls(auOut, au, rho_, currentNum);            // auOut = accum_update * rho
        Add(auOut, auOut, tmp2, currentNum);          // auOut = au*rho + update^2*(1-rho)
    }

    // Free input tensors
    inQueVar.FreeTensor(v);
    inQueAccum.FreeTensor(a);
    inQueAccumUpdate.FreeTensor(au);
    inQueGrad.FreeTensor(g);

    // Enqueue output tensors
    outQueVar.EnQue(vOut);
    outQueAccum.EnQue(aOut);
    outQueAccumUpdate.EnQue(auOut);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void ApplyAdadelta<T, BUFFER_MODE>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> vOut = outQueVar.template DeQue<T>();
    LocalTensor<T> aOut = outQueAccum.template DeQue<T>();
    LocalTensor<T> auOut = outQueAccumUpdate.template DeQue<T>();

    DataCopyExtParams cp;
    cp.blockCount = 1;
    cp.blockLen = currentNum * sizeof(T);
    cp.srcStride = 0;
    cp.dstStride = 0;
    int64_t gmOff = progress * ubLength_;
    DataCopyPad(varOutGm[gmOff], vOut, cp);
    DataCopyPad(accumOutGm[gmOff], aOut, cp);
    DataCopyPad(accumUpdateOutGm[gmOff], auOut, cp);

    outQueVar.FreeTensor(vOut);
    outQueAccum.FreeTensor(aOut);
    outQueAccumUpdate.FreeTensor(auOut);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void ApplyAdadelta<T, BUFFER_MODE>::Process()
{
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsApplyAdadelta
#endif // APPLY_ADADELTA_H
