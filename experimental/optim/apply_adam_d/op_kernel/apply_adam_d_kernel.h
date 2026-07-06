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
 * \file apply_adam_d_kernel.h
 * \brief ApplyAdamD kernel class definition (ascend910b / DAV_2201)
 *
 * Formula (内部统一 FP32 计算，两分支分母均用更新后的 v_new)：
 *   lr_t      = lr * sqrt(1 - beta2_power) / (1 - beta1_power)
 *   m_new     = beta1 * m + (1 - beta1) * grad
 *   v_new     = beta2 * v + (1 - beta2) * grad * grad
 *   numerator = (m_new * beta1 + (1 - beta1) * grad)  if use_nesterov  else  m_new
 *   var_new   = var - lr_t * numerator / (epsilon + sqrt(v_new))
 *
 * Template parameters:
 *   - T:            data type (float / half / bfloat16_t)
 *   - USE_NESTEROV: 0/1 compile-time branch for numerator.
 *
 * 6 个标量 (beta1_power/beta2_power/lr/beta1/beta2/epsilon) 为 GM [1] 张量，
 * Init 中经 DataCopyPad->UB->GetValue 读取(fp16/bf16 经 Cast 升精度)，仅一次性，不入热路径。
 */
#ifndef APPLY_ADAM_D_KERNEL_H
#define APPLY_ADAM_D_KERNEL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "apply_adam_d_tiling_data.h"

namespace NsApplyAdamD {

using namespace AscendC;

template <typename T, int USE_NESTEROV>
class ApplyAdamD {
    static constexpr int32_t BUFFER_NUM = 2; // double buffer
    static constexpr bool IS_FP16 = !AscendC::IsSameType<T, float>::value;
    static constexpr int32_t SCALAR_NUM = 6;   // beta1_power..epsilon
    static constexpr int32_t BLOCK_BYTES = 32; // 32B-aligned scalar slot
    static constexpr int32_t ELEMS_PER_SLOT = BLOCK_BYTES / static_cast<int32_t>(sizeof(T));

public:
    __aicore__ inline ApplyAdamD() = default;

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR m, GM_ADDR v, GM_ADDR beta1Power, GM_ADDR beta2Power, GM_ADDR lr,
                                GM_ADDR beta1, GM_ADDR beta2, GM_ADDR epsilon, GM_ADDR grad, GM_ADDR varOut,
                                GM_ADDR mOut, GM_ADDR vOut, const ApplyAdamDTilingData* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void LoadScalars(GM_ADDR beta1Power, GM_ADDR beta2Power, GM_ADDR lr, GM_ADDR beta1, GM_ADDR beta2,
                                       GM_ADDR epsilon);
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void ComputeFp32(LocalTensor<T>& varIn, LocalTensor<T>& mIn, LocalTensor<T>& vIn,
                                       LocalTensor<T>& gradIn, LocalTensor<T>& varOutLocal, LocalTensor<T>& mOutLocal,
                                       LocalTensor<T>& vOutLocal, LocalTensor<float>& tmpA, LocalTensor<float>& tmpB,
                                       int32_t n);
    __aicore__ inline void ComputeCastFp32(LocalTensor<T>& varIn, LocalTensor<T>& mIn, LocalTensor<T>& vIn,
                                           LocalTensor<T>& gradIn, LocalTensor<T>& varOutLocal,
                                           LocalTensor<T>& mOutLocal, LocalTensor<T>& vOutLocal,
                                           LocalTensor<float>& tmpA, LocalTensor<float>& tmpB, int32_t n);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe;

    // Input queues (var, m, v, grad)
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueVar;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueM;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueV;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueGrad;

    // Output queues (varOut, mOut, vOut)
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueVar;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueM;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueV;

    // Intermediate fp32 compute buffers
    TBuf<TPosition::VECCALC> tmpABuf;
    TBuf<TPosition::VECCALC> tmpBBuf;
    // FP16/BF16 path Cast-up buffers (fp32) - only allocated when IS_FP16
    TBuf<TPosition::VECCALC> castVarBuf;
    TBuf<TPosition::VECCALC> castMBuf;
    TBuf<TPosition::VECCALC> castVBuf;
    TBuf<TPosition::VECCALC> castGradBuf;
    // Scalar staging buffers
    TBuf<TPosition::VECCALC> scalarBufT; // T view (6 x 32B slots)
    TBuf<TPosition::VECCALC> scalarBufF; // fp32 view (only IS_FP16)

    // Global Tensors
    GlobalTensor<T> varGm;
    GlobalTensor<T> mGm;
    GlobalTensor<T> vGm;
    GlobalTensor<T> gradGm;
    GlobalTensor<T> varOutGm;
    GlobalTensor<T> mOutGm;
    GlobalTensor<T> vOutGm;

    // Scalars (FP32, read once in Init)
    float lrT_ = 0.0f;
    float beta1_ = 0.0f;
    float beta2_ = 0.0f;
    float eps_ = 0.0f;
    float oneMinusBeta1_ = 0.0f;
    float oneMinusBeta2_ = 0.0f;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T, int USE_NESTEROV>
__aicore__ inline void ApplyAdamD<T, USE_NESTEROV>::Init(GM_ADDR var, GM_ADDR m, GM_ADDR v, GM_ADDR beta1Power,
                                                         GM_ADDR beta2Power, GM_ADDR lr, GM_ADDR beta1, GM_ADDR beta2,
                                                         GM_ADDR epsilon, GM_ADDR grad, GM_ADDR varOut, GM_ADDR mOut,
                                                         GM_ADDR vOut, const ApplyAdamDTilingData* tilingData)
{
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * blockIdx;
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    ubLength_ = tilingData->ubFactor;
    if (ubLength_ <= 0) {
        ubLength_ = 1;
    }

    int64_t offset = tilingData->blockFactor * blockIdx;

    varGm.SetGlobalBuffer((__gm__ T*)var + offset, blockLength_);
    mGm.SetGlobalBuffer((__gm__ T*)m + offset, blockLength_);
    vGm.SetGlobalBuffer((__gm__ T*)v + offset, blockLength_);
    gradGm.SetGlobalBuffer((__gm__ T*)grad + offset, blockLength_);
    varOutGm.SetGlobalBuffer((__gm__ T*)varOut + offset, blockLength_);
    mOutGm.SetGlobalBuffer((__gm__ T*)mOut + offset, blockLength_);
    vOutGm.SetGlobalBuffer((__gm__ T*)vOut + offset, blockLength_);

    // Tensor UB queues
    pipe.InitBuffer(inQueVar, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(inQueM, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(inQueV, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(inQueGrad, BUFFER_NUM, ubLength_ * sizeof(T));

    pipe.InitBuffer(outQueVar, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outQueM, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outQueV, BUFFER_NUM, ubLength_ * sizeof(T));

    pipe.InitBuffer(tmpABuf, ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpBBuf, ubLength_ * sizeof(float));

    if constexpr (IS_FP16) {
        pipe.InitBuffer(castVarBuf, ubLength_ * sizeof(float));
        pipe.InitBuffer(castMBuf, ubLength_ * sizeof(float));
        pipe.InitBuffer(castVBuf, ubLength_ * sizeof(float));
        pipe.InitBuffer(castGradBuf, ubLength_ * sizeof(float));
        pipe.InitBuffer(scalarBufF, SCALAR_NUM * ELEMS_PER_SLOT * sizeof(float));
    }
    pipe.InitBuffer(scalarBufT, SCALAR_NUM * BLOCK_BYTES);

    LoadScalars(beta1Power, beta2Power, lr, beta1, beta2, epsilon);
}

template <typename T, int USE_NESTEROV>
__aicore__ inline void ApplyAdamD<T, USE_NESTEROV>::LoadScalars(GM_ADDR beta1Power, GM_ADDR beta2Power, GM_ADDR lr,
                                                                GM_ADDR beta1, GM_ADDR beta2, GM_ADDR epsilon)
{
    GlobalTensor<T> gB1P, gB2P, gLr, gB1, gB2, gEps;
    gB1P.SetGlobalBuffer((__gm__ T*)beta1Power, 1);
    gB2P.SetGlobalBuffer((__gm__ T*)beta2Power, 1);
    gLr.SetGlobalBuffer((__gm__ T*)lr, 1);
    gB1.SetGlobalBuffer((__gm__ T*)beta1, 1);
    gB2.SetGlobalBuffer((__gm__ T*)beta2, 1);
    gEps.SetGlobalBuffer((__gm__ T*)epsilon, 1);

    LocalTensor<T> sT = scalarBufT.template Get<T>();
    DataCopyExtParams p{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{false, 0, 0, 0};
    DataCopyPad(sT[0 * ELEMS_PER_SLOT], gB1P, p, pad);
    DataCopyPad(sT[1 * ELEMS_PER_SLOT], gB2P, p, pad);
    DataCopyPad(sT[2 * ELEMS_PER_SLOT], gLr, p, pad);
    DataCopyPad(sT[3 * ELEMS_PER_SLOT], gB1, p, pad);
    DataCopyPad(sT[4 * ELEMS_PER_SLOT], gB2, p, pad);
    DataCopyPad(sT[5 * ELEMS_PER_SLOT], gEps, p, pad);

    float b1p = 0.0f, b2p = 0.0f, lrv = 0.0f, b1 = 0.0f, b2 = 0.0f, epsv = 0.0f;
    if constexpr (!IS_FP16) {
        // T == float: read directly.
        event_t evt = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(evt);
        WaitFlag<HardEvent::MTE2_S>(evt);
        b1p = sT.GetValue(0 * ELEMS_PER_SLOT);
        b2p = sT.GetValue(1 * ELEMS_PER_SLOT);
        lrv = sT.GetValue(2 * ELEMS_PER_SLOT);
        b1 = sT.GetValue(3 * ELEMS_PER_SLOT);
        b2 = sT.GetValue(4 * ELEMS_PER_SLOT);
        epsv = sT.GetValue(5 * ELEMS_PER_SLOT);
    } else {
        // fp16/bf16: Cast-up to fp32 via vector Cast (arch-portable), then read.
        event_t evtM2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(evtM2V);
        WaitFlag<HardEvent::MTE2_V>(evtM2V);
        LocalTensor<float> sF = scalarBufF.template Get<float>();
        Cast(sF, sT, RoundMode::CAST_NONE, SCALAR_NUM * ELEMS_PER_SLOT);
        event_t evtV2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(evtV2S);
        WaitFlag<HardEvent::V_S>(evtV2S);
        b1p = sF.GetValue(0 * ELEMS_PER_SLOT);
        b2p = sF.GetValue(1 * ELEMS_PER_SLOT);
        lrv = sF.GetValue(2 * ELEMS_PER_SLOT);
        b1 = sF.GetValue(3 * ELEMS_PER_SLOT);
        b2 = sF.GetValue(4 * ELEMS_PER_SLOT);
        epsv = sF.GetValue(5 * ELEMS_PER_SLOT);
    }

    beta1_ = b1;
    beta2_ = b2;
    eps_ = epsv;
    oneMinusBeta1_ = 1.0f - b1;
    oneMinusBeta2_ = 1.0f - b2;
    // lr_t = lr * sqrt(1 - beta2_power) / (1 - beta1_power)  (scalar sqrt, matches numpy golden)
    lrT_ = lrv * sqrt(1.0f - b2p) / (1.0f - b1p);
}

template <typename T, int USE_NESTEROV>
__aicore__ inline void ApplyAdamD<T, USE_NESTEROV>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> varLocal = inQueVar.template AllocTensor<T>();
    LocalTensor<T> mLocal = inQueM.template AllocTensor<T>();
    LocalTensor<T> vLocal = inQueV.template AllocTensor<T>();
    LocalTensor<T> gradLocal = inQueGrad.template AllocTensor<T>();

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    int64_t gmOffset = progress * ubLength_;
    DataCopyPad(varLocal, varGm[gmOffset], copyParams, padParams);
    DataCopyPad(mLocal, mGm[gmOffset], copyParams, padParams);
    DataCopyPad(vLocal, vGm[gmOffset], copyParams, padParams);
    DataCopyPad(gradLocal, gradGm[gmOffset], copyParams, padParams);

    inQueVar.EnQue(varLocal);
    inQueM.EnQue(mLocal);
    inQueV.EnQue(vLocal);
    inQueGrad.EnQue(gradLocal);
}

template <typename T, int USE_NESTEROV>
__aicore__ inline void ApplyAdamD<T, USE_NESTEROV>::ComputeFp32(LocalTensor<T>& varIn, LocalTensor<T>& mIn,
                                                                LocalTensor<T>& vIn, LocalTensor<T>& gradIn,
                                                                LocalTensor<T>& varOutLocal, LocalTensor<T>& mOutLocal,
                                                                LocalTensor<T>& vOutLocal, LocalTensor<float>& tmpA,
                                                                LocalTensor<float>& tmpB, int32_t n)
{
    Muls(tmpA, mIn, beta1_, n);
    Muls(tmpB, gradIn, oneMinusBeta1_, n);
    Add(mOutLocal, tmpA, tmpB, n);

    Mul(tmpA, gradIn, gradIn, n);
    Muls(tmpA, tmpA, oneMinusBeta2_, n);
    Muls(tmpB, vIn, beta2_, n);
    Add(vOutLocal, tmpA, tmpB, n);

    if constexpr (USE_NESTEROV == 1) {
        Muls(tmpA, mOutLocal, beta1_, n);
        Muls(tmpB, gradIn, oneMinusBeta1_, n);
        Add(tmpA, tmpA, tmpB, n);
        Muls(tmpA, tmpA, lrT_, n);
    } else {
        Muls(tmpA, mOutLocal, lrT_, n);
    }

    Sqrt(tmpB, vOutLocal, n);
    Adds(tmpB, tmpB, eps_, n);
    Div(tmpA, tmpA, tmpB, n);
    Sub(varOutLocal, varIn, tmpA, n);
}

template <typename T, int USE_NESTEROV>
__aicore__ inline void ApplyAdamD<T, USE_NESTEROV>::ComputeCastFp32(LocalTensor<T>& varIn, LocalTensor<T>& mIn,
                                                                    LocalTensor<T>& vIn, LocalTensor<T>& gradIn,
                                                                    LocalTensor<T>& varOutLocal,
                                                                    LocalTensor<T>& mOutLocal,
                                                                    LocalTensor<T>& vOutLocal, LocalTensor<float>& tmpA,
                                                                    LocalTensor<float>& tmpB, int32_t n)
{
    LocalTensor<float> castVar = castVarBuf.Get<float>();
    LocalTensor<float> castM = castMBuf.Get<float>();
    LocalTensor<float> castV = castVBuf.Get<float>();
    LocalTensor<float> castGrad = castGradBuf.Get<float>();

    Cast(castVar, varIn, RoundMode::CAST_NONE, n);
    Cast(castM, mIn, RoundMode::CAST_NONE, n);
    Cast(castV, vIn, RoundMode::CAST_NONE, n);
    Cast(castGrad, gradIn, RoundMode::CAST_NONE, n);

    Muls(tmpA, castM, beta1_, n);
    Muls(tmpB, castGrad, oneMinusBeta1_, n);
    Add(castM, tmpA, tmpB, n);

    Mul(tmpA, castGrad, castGrad, n);
    Muls(tmpA, tmpA, oneMinusBeta2_, n);
    Muls(tmpB, castV, beta2_, n);
    Add(castV, tmpA, tmpB, n);

    if constexpr (USE_NESTEROV == 1) {
        Muls(tmpA, castM, beta1_, n);
        Muls(tmpB, castGrad, oneMinusBeta1_, n);
        Add(tmpA, tmpA, tmpB, n);
        Muls(tmpA, tmpA, lrT_, n);
    } else {
        Muls(tmpA, castM, lrT_, n);
    }

    Sqrt(tmpB, castV, n);
    Adds(tmpB, tmpB, eps_, n);
    Div(tmpA, tmpA, tmpB, n);
    Sub(castVar, castVar, tmpA, n);

    Cast(mOutLocal, castM, RoundMode::CAST_RINT, n);
    Cast(vOutLocal, castV, RoundMode::CAST_RINT, n);
    Cast(varOutLocal, castVar, RoundMode::CAST_RINT, n);
}

template <typename T, int USE_NESTEROV>
__aicore__ inline void ApplyAdamD<T, USE_NESTEROV>::Compute(int64_t currentNum)
{
    LocalTensor<T> varIn = inQueVar.template DeQue<T>();
    LocalTensor<T> mIn = inQueM.template DeQue<T>();
    LocalTensor<T> vIn = inQueV.template DeQue<T>();
    LocalTensor<T> gradIn = inQueGrad.template DeQue<T>();
    LocalTensor<T> varOutLocal = outQueVar.template AllocTensor<T>();
    LocalTensor<T> mOutLocal = outQueM.template AllocTensor<T>();
    LocalTensor<T> vOutLocal = outQueV.template AllocTensor<T>();
    LocalTensor<float> tmpA = tmpABuf.Get<float>();
    LocalTensor<float> tmpB = tmpBBuf.Get<float>();
    int32_t n = static_cast<int32_t>(currentNum);

    if constexpr (!IS_FP16) {
        ComputeFp32(varIn, mIn, vIn, gradIn, varOutLocal, mOutLocal, vOutLocal, tmpA, tmpB, n);
    } else {
        ComputeCastFp32(varIn, mIn, vIn, gradIn, varOutLocal, mOutLocal, vOutLocal, tmpA, tmpB, n);
    }

    outQueVar.template EnQue<T>(varOutLocal);
    outQueM.template EnQue<T>(mOutLocal);
    outQueV.template EnQue<T>(vOutLocal);

    inQueVar.FreeTensor(varIn);
    inQueM.FreeTensor(mIn);
    inQueV.FreeTensor(vIn);
    inQueGrad.FreeTensor(gradIn);
}

template <typename T, int USE_NESTEROV>
__aicore__ inline void ApplyAdamD<T, USE_NESTEROV>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> varOutLocal = outQueVar.template DeQue<T>();
    LocalTensor<T> mOutLocal = outQueM.template DeQue<T>();
    LocalTensor<T> vOutLocal = outQueV.template DeQue<T>();

    DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
    int64_t gmOffset = progress * ubLength_;
    DataCopyPad(varOutGm[gmOffset], varOutLocal, copyParams);
    DataCopyPad(mOutGm[gmOffset], mOutLocal, copyParams);
    DataCopyPad(vOutGm[gmOffset], vOutLocal, copyParams);

    outQueVar.FreeTensor(varOutLocal);
    outQueM.FreeTensor(mOutLocal);
    outQueV.FreeTensor(vOutLocal);
}

template <typename T, int USE_NESTEROV>
__aicore__ inline void ApplyAdamD<T, USE_NESTEROV>::Process()
{
    if (blockLength_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; ++i) {
        int64_t currentNum = (i == loopCount - 1) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsApplyAdamD

#endif // APPLY_ADAM_D_KERNEL_H
