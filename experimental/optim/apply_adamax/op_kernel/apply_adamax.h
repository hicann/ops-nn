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
 * \file apply_adamax.h
 * \brief ApplyAdamax kernel class definition (arch35 / Ascend950)
 *
 * Formula:
 *   m_new   = beta1 * m + (1 - beta1) * grad
 *   v_new   = max(beta2 * v, |grad|)
 *   var_new = var - lrAdj * m_new / (v_new + epsilon)
 *
 * where lrAdj = lr / (1 - beta1Power), all five scalars (beta1Power, lr,
 * beta1, beta2, epsilon) come from TilingData (set from Attr by Tiling).
 *
 * Template parameters:
 *   - T:        Data type (float / half)
 *   - K_ALIGN:  32B alignment flag (0=unaligned, 1=aligned). Currently used
 *               for TilingKey dispatch only; DataCopyPad covers both paths.
 */

#ifndef APPLY_ADAMAX_H
#define APPLY_ADAMAX_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "apply_adamax_tiling_data.h"
#include "apply_adamax_tiling_key.h"

namespace NsApplyAdamax {

using namespace AscendC;

template <typename T, int K_ALIGN>
class ApplyAdamax {
    static constexpr int32_t BUFFER_NUM = 2;  // double buffer
    static constexpr bool IS_FP16 = !AscendC::IsSameType<T, float>::value;

public:
    __aicore__ inline ApplyAdamax() = default;

    __aicore__ inline void Init(
        GM_ADDR var, GM_ADDR m, GM_ADDR v, GM_ADDR grad,
        GM_ADDR varOut, GM_ADDR mOut, GM_ADDR vOut,
        const ApplyAdamaxTilingData* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void LoadScalars(const ApplyAdamaxTilingData* tilingData);
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
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

    // Intermediate compute buffers (fp32)
    TBuf<TPosition::VECCALC> tmpABuf;
    TBuf<TPosition::VECCALC> tmpBBuf;
    // FP16 path Cast buffers (fp32) - only allocated when IS_FP16
    TBuf<TPosition::VECCALC> castVarBuf;
    TBuf<TPosition::VECCALC> castMBuf;
    TBuf<TPosition::VECCALC> castVBuf;
    TBuf<TPosition::VECCALC> castGradBuf;

    // Global Tensors
    GlobalTensor<T> varGm;
    GlobalTensor<T> mGm;
    GlobalTensor<T> vGm;
    GlobalTensor<T> gradGm;
    GlobalTensor<T> varOutGm;
    GlobalTensor<T> mOutGm;
    GlobalTensor<T> vOutGm;

    // Scalar parameters (from TilingData)
    float beta1_ = 0.0f;
    float beta2_ = 0.0f;
    float epsilon_ = 0.0f;
    float oneMinusBeta1_ = 0.0f;
    float lrAdj_ = 0.0f;  // lr / (1 - beta1Power)

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T, int K_ALIGN>
__aicore__ inline void ApplyAdamax<T, K_ALIGN>::Init(
    GM_ADDR var, GM_ADDR m, GM_ADDR v, GM_ADDR grad,
    GM_ADDR varOut, GM_ADDR mOut, GM_ADDR vOut,
    const ApplyAdamaxTilingData* tilingData)
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

    // Init UB queues
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
    }

    // Load scalars from TilingData
    LoadScalars(tilingData);
}

template <typename T, int K_ALIGN>
__aicore__ inline void ApplyAdamax<T, K_ALIGN>::LoadScalars(const ApplyAdamaxTilingData* tilingData)
{
    beta1_ = tilingData->beta1;
    beta2_ = tilingData->beta2;
    epsilon_ = tilingData->epsilon;
    oneMinusBeta1_ = tilingData->oneMinusBeta1;
    lrAdj_ = tilingData->lrAdj;

    // Epsilon floor guard (Tiling layer cannot guarantee > 0 from arbitrary user
    // input; Kernel-level last line of defense to avoid 0/0 -> NaN/Inf).
    constexpr float EPS_FLOOR = 1.1754944e-38f;  // FLT_MIN
    if (epsilon_ >= 0.0f && epsilon_ < EPS_FLOOR) {
        epsilon_ = EPS_FLOOR;
    }
}

template <typename T, int K_ALIGN>
__aicore__ inline void ApplyAdamax<T, K_ALIGN>::CopyIn(int64_t progress, int64_t currentNum)
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

template <typename T, int K_ALIGN>
__aicore__ inline void ApplyAdamax<T, K_ALIGN>::Compute(int64_t currentNum)
{
    LocalTensor<T> varIn = inQueVar.template DeQue<T>();
    LocalTensor<T> mIn = inQueM.template DeQue<T>();
    LocalTensor<T> vIn = inQueV.template DeQue<T>();
    LocalTensor<T> gradIn = inQueGrad.template DeQue<T>();

    LocalTensor<T> varOutLocal = outQueVar.template AllocTensor<T>();
    LocalTensor<T> mOutLocal = outQueM.template AllocTensor<T>();
    LocalTensor<T> vOutLocal = outQueV.template AllocTensor<T>();

    int32_t len = static_cast<int32_t>(currentNum);

    LocalTensor<float> tmpA = tmpABuf.Get<float>();
    LocalTensor<float> tmpB = tmpBBuf.Get<float>();

    if constexpr (!IS_FP16) {
        // FP32 path
        LocalTensor<float> varF = varIn.template ReinterpretCast<float>();
        LocalTensor<float> mF = mIn.template ReinterpretCast<float>();
        LocalTensor<float> vF = vIn.template ReinterpretCast<float>();
        LocalTensor<float> gradF = gradIn.template ReinterpretCast<float>();
        LocalTensor<float> varOutF = varOutLocal.template ReinterpretCast<float>();
        LocalTensor<float> mOutF = mOutLocal.template ReinterpretCast<float>();
        LocalTensor<float> vOutF = vOutLocal.template ReinterpretCast<float>();

        // Step 1: m_new = beta1 * m + (1 - beta1) * grad
        Muls(tmpA, mF, beta1_, len);
        Muls(tmpB, gradF, oneMinusBeta1_, len);
        Add(mOutF, tmpA, tmpB, len);

        // Step 2: v_new = max(beta2 * v, |grad|)
        Muls(tmpA, vF, beta2_, len);
        Abs(tmpB, gradF, len);
        Max(vOutF, tmpA, tmpB, len);

        // Step 3: var_new = var - lrAdj * m_new / (v_new + eps)
        Adds(tmpA, vOutF, epsilon_, len);
        Muls(tmpB, mOutF, lrAdj_, len);
        Div(tmpB, tmpB, tmpA, len);
        Sub(varOutF, varF, tmpB, len);
    } else {
        // FP16 path: Cast up to fp32, compute, Cast back
        LocalTensor<float> castVar = castVarBuf.Get<float>();
        LocalTensor<float> castM = castMBuf.Get<float>();
        LocalTensor<float> castV = castVBuf.Get<float>();
        LocalTensor<float> castGrad = castGradBuf.Get<float>();

        Cast(castVar, varIn, RoundMode::CAST_NONE, len);
        Cast(castM, mIn, RoundMode::CAST_NONE, len);
        Cast(castV, vIn, RoundMode::CAST_NONE, len);
        Cast(castGrad, gradIn, RoundMode::CAST_NONE, len);

        // Step 1: m_new = beta1 * m + (1 - beta1) * grad
        Muls(tmpA, castM, beta1_, len);
        Muls(tmpB, castGrad, oneMinusBeta1_, len);
        Add(castM, tmpA, tmpB, len);  // castM = m_new (fp32)

        // Step 2: v_new = max(beta2 * v, |grad|)
        Muls(tmpA, castV, beta2_, len);
        Abs(tmpB, castGrad, len);
        Max(castV, tmpA, tmpB, len);  // castV = v_new (fp32)

        // Step 3: var_new = var - lrAdj * m_new / (v_new + eps)
        Adds(tmpA, castV, epsilon_, len);
        Muls(tmpB, castM, lrAdj_, len);
        Div(tmpB, tmpB, tmpA, len);
        Sub(castVar, castVar, tmpB, len);  // castVar = var_new (fp32)

        // Cast back to fp16
        Cast(mOutLocal, castM, RoundMode::CAST_RINT, len);
        Cast(vOutLocal, castV, RoundMode::CAST_RINT, len);
        Cast(varOutLocal, castVar, RoundMode::CAST_RINT, len);
    }

    outQueVar.template EnQue<T>(varOutLocal);
    outQueM.template EnQue<T>(mOutLocal);
    outQueV.template EnQue<T>(vOutLocal);

    inQueVar.FreeTensor(varIn);
    inQueM.FreeTensor(mIn);
    inQueV.FreeTensor(vIn);
    inQueGrad.FreeTensor(gradIn);
}

template <typename T, int K_ALIGN>
__aicore__ inline void ApplyAdamax<T, K_ALIGN>::CopyOut(int64_t progress, int64_t currentNum)
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

template <typename T, int K_ALIGN>
__aicore__ inline void ApplyAdamax<T, K_ALIGN>::Process()
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

} // namespace NsApplyAdamax

#endif // APPLY_ADAMAX_H
