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

/**
 * \file apply_centered_rms_prop.h
 * \brief ApplyCenteredRMSProp kernel class (arch35 / Ascend950).
 *
 * Implements the Centered RMSProp optimizer step per element:
 *   mg_new  = rho * mg + (1 - rho) * grad
 *   ms_new  = rho * ms + (1 - rho) * grad * grad
 *   denom   = sqrt(ms_new - mg_new * mg_new + epsilon)
 *   mom_new = momentum * mom + lr * grad / denom
 *   var_new = var - mom_new
 *
 * Routes:
 *   - D_T_VAR = half  (TilingKey 2): fp16 path. Cast(fp16->fp32) on CopyIn,
 *     compute in fp32, Cast(fp32->fp16) on CopyOut.
 *   - D_T_VAR = float (TilingKey 1): fp32 path. Direct in-place fp32 compute
 *     (no Cast); reuses denom/tmp1/tmp2 fp32 scratch buffers. Maxs(...,0.0f)
 *     clamp shared with fp16 path for sqrt-of-negative numerical safety.
 */

#ifndef APPLY_CENTERED_RMS_PROP_H
#define APPLY_CENTERED_RMS_PROP_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "apply_centered_rms_prop_tiling_data.h"
#include "apply_centered_rms_prop_tiling_key.h"

namespace NsApplyCenteredRMSProp {

using AscendC::TPipe;
using AscendC::TQue;
using AscendC::TBuf;
using AscendC::QuePosition;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPad;
using AscendC::DataCopyPadExtParams;
using AscendC::GetBlockIdx;
using AscendC::Add;
using AscendC::Sub;
using AscendC::Mul;
using AscendC::Muls;
using AscendC::Adds;
using AscendC::Maxs;
using AscendC::Div;
using AscendC::Sqrt;
using AscendC::Cast;
using AscendC::RoundMode;

template <typename T>
class ApplyCenteredRMSProp {
public:
    __aicore__ inline ApplyCenteredRMSProp() {}

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR mg, GM_ADDR ms, GM_ADDR mom,
                                GM_ADDR lr, GM_ADDR rho, GM_ADDR momentum,
                                GM_ADDR epsilon, GM_ADDR grad,
                                GM_ADDR varOut, GM_ADDR mgOut,
                                GM_ADDR msOut, GM_ADDR momOut,
                                const ApplyCenteredRMSPropTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInTile(int64_t gmOffset, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOutTile(int64_t gmOffset, int64_t currentNum);

    __aicore__ inline float LoadScalar(const GlobalTensor<T>& src) const;

    // Suggestion-3: shared DataCopyExtParams builder for tile DMA blocks.
    static __aicore__ inline DataCopyExtParams MakeTileCopyParams(int64_t currentNum)
    {
        DataCopyExtParams p;
        p.blockCount = 1;
        p.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
        p.srcStride = 0;
        p.dstStride = 0;
        return p;
    }

private:
    TPipe pipe_;
    // IN queues: var / mg / ms / mom / grad, double-buffered.
    TQue<QuePosition::VECIN, 2> varInQue_;
    TQue<QuePosition::VECIN, 2> mgInQue_;
    TQue<QuePosition::VECIN, 2> msInQue_;
    TQue<QuePosition::VECIN, 2> momInQue_;
    TQue<QuePosition::VECIN, 2> gradInQue_;
    // OUT queues: var / mg / ms / mom, double-buffered.
    TQue<QuePosition::VECOUT, 2> varOutQue_;
    TQue<QuePosition::VECOUT, 2> mgOutQue_;
    TQue<QuePosition::VECOUT, 2> msOutQue_;
    TQue<QuePosition::VECOUT, 2> momOutQue_;
    // fp32 scratch buffers (only used on the fp16 path for Cast intermediates,
    // and on both paths for compute scratch).
    TBuf<QuePosition::VECCALC> varF32Buf_;
    TBuf<QuePosition::VECCALC> mgF32Buf_;
    TBuf<QuePosition::VECCALC> msF32Buf_;
    TBuf<QuePosition::VECCALC> momF32Buf_;
    TBuf<QuePosition::VECCALC> gradF32Buf_;
    TBuf<QuePosition::VECCALC> denomBuf_;
    TBuf<QuePosition::VECCALC> tmp1Buf_;
    TBuf<QuePosition::VECCALC> tmp2Buf_;

    GlobalTensor<T> varGm_;
    GlobalTensor<T> mgGm_;
    GlobalTensor<T> msGm_;
    GlobalTensor<T> momGm_;
    GlobalTensor<T> gradGm_;
    GlobalTensor<T> lrGm_;
    GlobalTensor<T> rhoGm_;
    GlobalTensor<T> momentumGm_;
    GlobalTensor<T> epsilonGm_;
    GlobalTensor<T> varOutGm_;
    GlobalTensor<T> mgOutGm_;
    GlobalTensor<T> msOutGm_;
    GlobalTensor<T> momOutGm_;

    // Scalars loaded from GM at Init time (always promoted to float for
    // intra-kernel use, regardless of dtype path).
    float lrScalar_ = 0.0f;
    float rhoScalar_ = 0.0f;
    float momentumScalar_ = 0.0f;
    float epsilonScalar_ = 0.0f;
    // Derived: oneMinusRho_ = 1.0f - rho.
    float oneMinusRho_ = 0.0f;

    int64_t blockOffset_ = 0;
    int64_t blockLen_ = 0;
    int64_t ubFactor_ = 0;
};

// =============================================================================
// LoadScalar: read element[0] of a 1-element GM tensor as float.
// =============================================================================
template <typename T>
__aicore__ inline float ApplyCenteredRMSProp<T>::LoadScalar(
    const GlobalTensor<T>& src) const
{
    return static_cast<float>(src.GetValue(0));
}

// =============================================================================
// Init
// =============================================================================
template <typename T>
__aicore__ inline void ApplyCenteredRMSProp<T>::Init(
    GM_ADDR var, GM_ADDR mg, GM_ADDR ms, GM_ADDR mom,
    GM_ADDR lr, GM_ADDR rho, GM_ADDR momentum, GM_ADDR epsilon,
    GM_ADDR grad,
    GM_ADDR varOut, GM_ADDR mgOut, GM_ADDR msOut, GM_ADDR momOut,
    const ApplyCenteredRMSPropTilingData* tilingData)
{
    ubFactor_ = tilingData->ubFactor;

    // Empty-tensor / degenerate tiling -> just record zero length; Process()
    // will short-circuit before any compute.
    if (tilingData->totalElements == 0 || tilingData->blockFactor == 0) {
        blockOffset_ = 0;
        blockLen_ = 0;
        return;
    }

    blockOffset_ = tilingData->blockFactor * static_cast<int64_t>(GetBlockIdx());
    int64_t remaining = tilingData->totalElements - blockOffset_;
    if (remaining <= 0) {
        blockLen_ = 0;
        return;
    }
    blockLen_ = (remaining > tilingData->blockFactor)
                    ? tilingData->blockFactor
                    : remaining;

    // Main vectorised tensors -- slice each core's view.
    varGm_.SetGlobalBuffer((__gm__ T*)var + blockOffset_, blockLen_);
    mgGm_.SetGlobalBuffer((__gm__ T*)mg + blockOffset_, blockLen_);
    msGm_.SetGlobalBuffer((__gm__ T*)ms + blockOffset_, blockLen_);
    momGm_.SetGlobalBuffer((__gm__ T*)mom + blockOffset_, blockLen_);
    gradGm_.SetGlobalBuffer((__gm__ T*)grad + blockOffset_, blockLen_);
    varOutGm_.SetGlobalBuffer((__gm__ T*)varOut + blockOffset_, blockLen_);
    mgOutGm_.SetGlobalBuffer((__gm__ T*)mgOut + blockOffset_, blockLen_);
    msOutGm_.SetGlobalBuffer((__gm__ T*)msOut + blockOffset_, blockLen_);
    momOutGm_.SetGlobalBuffer((__gm__ T*)momOut + blockOffset_, blockLen_);

    // Scalar tensors: single-element broadcast-read.
    lrGm_.SetGlobalBuffer((__gm__ T*)lr, 1);
    rhoGm_.SetGlobalBuffer((__gm__ T*)rho, 1);
    momentumGm_.SetGlobalBuffer((__gm__ T*)momentum, 1);
    epsilonGm_.SetGlobalBuffer((__gm__ T*)epsilon, 1);
    lrScalar_       = LoadScalar(lrGm_);
    rhoScalar_      = LoadScalar(rhoGm_);
    momentumScalar_ = LoadScalar(momentumGm_);
    epsilonScalar_  = LoadScalar(epsilonGm_);
    oneMinusRho_    = 1.0f - rhoScalar_;

    // UB buffer allocation -- mixed dtype layout for fp16 path.
    pipe_.InitBuffer(varInQue_,  2, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(mgInQue_,   2, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(msInQue_,   2, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(momInQue_,  2, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(gradInQue_, 2, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(varOutQue_, 2, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(mgOutQue_,  2, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(msOutQue_,  2, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(momOutQue_, 2, ubFactor_ * sizeof(T));

    // fp32 scratch (also reused on fp32 path -- single buffer each, no DB).
    // Suggestion-1 (UB layout optimisation): the 5 fp32 cast-buffers
    // (var/mg/ms/mom/grad F32) are only needed on the fp16 path; on the
    // fp32 path Compute() reuses denom/tmp1/tmp2 as scratch. Conditionally
    // allocate to free ~9 fp32-units / elem for the fp32 path (Tiling sets
    // UB_BUFFER_COUNT_FP32 = 17 to match).
    if constexpr (std::is_same_v<T, half>) {
        pipe_.InitBuffer(varF32Buf_,  ubFactor_ * sizeof(float));
        pipe_.InitBuffer(mgF32Buf_,   ubFactor_ * sizeof(float));
        pipe_.InitBuffer(msF32Buf_,   ubFactor_ * sizeof(float));
        pipe_.InitBuffer(momF32Buf_,  ubFactor_ * sizeof(float));
        pipe_.InitBuffer(gradF32Buf_, ubFactor_ * sizeof(float));
    }
    pipe_.InitBuffer(denomBuf_,   ubFactor_ * sizeof(float));
    pipe_.InitBuffer(tmp1Buf_,    ubFactor_ * sizeof(float));
    pipe_.InitBuffer(tmp2Buf_,    ubFactor_ * sizeof(float));
}

// =============================================================================
// CopyInTile: pad-aware DataCopyPad of var / mg / ms / mom / grad.
// =============================================================================
template <typename T>
__aicore__ inline void ApplyCenteredRMSProp<T>::CopyInTile(
    int64_t gmOffset, int64_t currentNum)
{
    LocalTensor<T> varLocal  = varInQue_.template AllocTensor<T>();
    LocalTensor<T> mgLocal   = mgInQue_.template AllocTensor<T>();
    LocalTensor<T> msLocal   = msInQue_.template AllocTensor<T>();
    LocalTensor<T> momLocal  = momInQue_.template AllocTensor<T>();
    LocalTensor<T> gradLocal = gradInQue_.template AllocTensor<T>();

    DataCopyExtParams copyParams = MakeTileCopyParams(currentNum);

    // Right-pad tail lanes ([currentNum, alignedNum)) with safe values:
    //   - var/mg/mom/grad: pad with 0 (Add/Mul with 0 preserves accumulators).
    //   - ms:              pad with 1 (sqrt(1 - 0 + eps) > 0, avoids div-by-0).
    constexpr int64_t kAlignBlock = 32 / sizeof(T);
    int64_t alignedNum =
        ((currentNum + kAlignBlock - 1) / kAlignBlock) * kAlignBlock;
    uint8_t rightPadCount = static_cast<uint8_t>(alignedNum - currentNum);
    DataCopyPadExtParams<T> padZero{true, 0, rightPadCount, static_cast<T>(0)};
    DataCopyPadExtParams<T> padOne{true, 0, rightPadCount, static_cast<T>(1.0f)};

    DataCopyPad(varLocal,  varGm_[gmOffset],  copyParams, padZero);
    DataCopyPad(mgLocal,   mgGm_[gmOffset],   copyParams, padZero);
    DataCopyPad(msLocal,   msGm_[gmOffset],   copyParams, padOne);
    DataCopyPad(momLocal,  momGm_[gmOffset],  copyParams, padZero);
    DataCopyPad(gradLocal, gradGm_[gmOffset], copyParams, padZero);

    varInQue_.EnQue(varLocal);
    mgInQue_.EnQue(mgLocal);
    msInQue_.EnQue(msLocal);
    momInQue_.EnQue(momLocal);
    gradInQue_.EnQue(gradLocal);
}

// =============================================================================
// CopyOutTile: write back var / mg / ms / mom to their inplace GM slots.
// =============================================================================
template <typename T>
__aicore__ inline void ApplyCenteredRMSProp<T>::CopyOutTile(
    int64_t gmOffset, int64_t currentNum)
{
    LocalTensor<T> varOutLocal = varOutQue_.template DeQue<T>();
    LocalTensor<T> mgOutLocal  = mgOutQue_.template DeQue<T>();
    LocalTensor<T> msOutLocal  = msOutQue_.template DeQue<T>();
    LocalTensor<T> momOutLocal = momOutQue_.template DeQue<T>();

    DataCopyExtParams copyParams = MakeTileCopyParams(currentNum);

    DataCopyPad(varOutGm_[gmOffset], varOutLocal, copyParams);
    DataCopyPad(mgOutGm_[gmOffset],  mgOutLocal,  copyParams);
    DataCopyPad(msOutGm_[gmOffset],  msOutLocal,  copyParams);
    DataCopyPad(momOutGm_[gmOffset], momOutLocal, copyParams);

    varOutQue_.FreeTensor(varOutLocal);
    mgOutQue_.FreeTensor(mgOutLocal);
    msOutQue_.FreeTensor(msOutLocal);
    momOutQue_.FreeTensor(momOutLocal);
}

// =============================================================================
// Compute: per-tile compute body.
//
// fp16 path (TilingKey 2):
//   1. Cast(varF32, varLocal_fp16)  ... same for mg/ms/mom/grad
//   2. mg_new  = rho * mg + (1-rho) * grad
//   3. ms_new  = rho * ms + (1-rho) * grad * grad
//   4. tmp     = ms_new - mg_new * mg_new + epsilon
//   5. denom   = sqrt(tmp)
//   6. mom_new = momentum * mom + lr * grad / denom
//   7. var_new = var - mom_new
//   8. Cast(varOutLocal_fp16, varF32) ... same for mg/ms/mom
//
// fp32 path (TilingKey 1): direct in-place fp32 compute (no Cast, reuses
// denom/tmp1/tmp2 scratch buffers).
// =============================================================================
template <typename T>
__aicore__ inline void ApplyCenteredRMSProp<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> varLocal  = varInQue_.template DeQue<T>();
    LocalTensor<T> mgLocal   = mgInQue_.template DeQue<T>();
    LocalTensor<T> msLocal   = msInQue_.template DeQue<T>();
    LocalTensor<T> momLocal  = momInQue_.template DeQue<T>();
    LocalTensor<T> gradLocal = gradInQue_.template DeQue<T>();

    LocalTensor<T> varOutLocal = varOutQue_.template AllocTensor<T>();
    LocalTensor<T> mgOutLocal  = mgOutQue_.template AllocTensor<T>();
    LocalTensor<T> msOutLocal  = msOutQue_.template AllocTensor<T>();
    LocalTensor<T> momOutLocal = momOutQue_.template AllocTensor<T>();

    // 32-byte align the work count (matches ubBlockSize for any dtype).
    constexpr int64_t kAlignBlock = 32 / sizeof(T);
    int64_t alignedNum =
        ((currentNum + kAlignBlock - 1) / kAlignBlock) * kAlignBlock;
    int32_t n = static_cast<int32_t>(alignedNum);

    if constexpr (std::is_same_v<T, half>) {
        // ---------- fp16 path (TilingKey 2) -----------------------------------
        LocalTensor<float> varF32  = varF32Buf_.template Get<float>();
        LocalTensor<float> mgF32   = mgF32Buf_.template Get<float>();
        LocalTensor<float> msF32   = msF32Buf_.template Get<float>();
        LocalTensor<float> momF32  = momF32Buf_.template Get<float>();
        LocalTensor<float> gradF32 = gradF32Buf_.template Get<float>();
        LocalTensor<float> denom   = denomBuf_.template Get<float>();
        LocalTensor<float> tmp1    = tmp1Buf_.template Get<float>();
        LocalTensor<float> tmp2    = tmp2Buf_.template Get<float>();

        // 1. Cast inputs fp16 -> fp32 (RoundMode::CAST_NONE for half->float).
        Cast(varF32,  varLocal,  RoundMode::CAST_NONE, n);
        Cast(mgF32,   mgLocal,   RoundMode::CAST_NONE, n);
        Cast(msF32,   msLocal,   RoundMode::CAST_NONE, n);
        Cast(momF32,  momLocal,  RoundMode::CAST_NONE, n);
        Cast(gradF32, gradLocal, RoundMode::CAST_NONE, n);

        // 2. mg_new = rho * mg + (1-rho) * grad
        Muls(tmp1, mgF32,   rhoScalar_,   n);     // tmp1 = rho * mg
        Muls(tmp2, gradF32, oneMinusRho_, n);     // tmp2 = (1-rho) * grad
        Add (mgF32, tmp1, tmp2, n);               // mgF32 = mg_new

        // 3. ms_new = rho * ms + (1-rho) * grad*grad
        Mul (tmp1, gradF32, gradF32, n);          // tmp1 = grad*grad
        Muls(tmp1, tmp1,    oneMinusRho_, n);     // tmp1 = (1-rho)*grad*grad
        Muls(tmp2, msF32,   rhoScalar_,   n);     // tmp2 = rho * ms
        Add (msF32, tmp1, tmp2, n);               // msF32 = ms_new

        // 4. tmp = ms_new - mg_new*mg_new + epsilon
        //    Clamp inner to >= 0 to mirror CPU golden's numerical-safety guard
        //    (ms_new < mg_new^2 can happen due to fp16 truncation, leading to
        //    Sqrt(NaN) and downstream NaN propagation). See
        //    issues/issue_20260423_sqrt_negative_clamp_1.md.
        Mul (tmp1, mgF32, mgF32, n);              // tmp1 = mg_new^2
        Sub (tmp2, msF32, tmp1,  n);              // tmp2 = ms_new - mg_new^2
        Maxs(tmp2, tmp2, 0.0f, n);                // tmp2 = max(tmp2, 0) -- clamp
        Adds(tmp2, tmp2, epsilonScalar_, n);      // tmp2 = ... + epsilon

        // 5. denom = sqrt(tmp)
        Sqrt(denom, tmp2, n);

        // 6. mom_new = momentum * mom + lr * grad / denom
        Div (tmp1, gradF32, denom, n);            // tmp1 = grad / denom
        Muls(tmp1, tmp1, lrScalar_, n);           // tmp1 = lr * grad / denom
        Muls(tmp2, momF32, momentumScalar_, n);   // tmp2 = momentum * mom
        Add (momF32, tmp1, tmp2, n);              // momF32 = mom_new

        // 7. var_new = var - mom_new
        Sub (varF32, varF32, momF32, n);

        // 8. Cast back fp32 -> fp16 (RoundMode::CAST_RINT for float->half).
        Cast(varOutLocal, varF32, RoundMode::CAST_RINT, n);
        Cast(mgOutLocal,  mgF32,  RoundMode::CAST_RINT, n);
        Cast(msOutLocal,  msF32,  RoundMode::CAST_RINT, n);
        Cast(momOutLocal, momF32, RoundMode::CAST_RINT, n);
    } else {
        // ---------- fp32 path (TilingKey 1) ----------------------------------
        // Direct fp32 compute: no Cast, operate on the fp16 queue tensors as
        // LocalTensor<float> aliases. Reuse denomBuf_ / tmp1Buf_ / tmp2Buf_
        // as fp32 scratch (mgF32/msF32/momF32/gradF32 buffers are left unused
        // on this path; UB budget is computed accordingly in Tiling).
        //
        // Computation matches the fp16 path (DESIGN §6.1 fp32 route):
        //   mg_new  = rho * mg + (1 - rho) * grad
        //   ms_new  = rho * ms + (1 - rho) * grad * grad
        //   denom   = sqrt(max(ms_new - mg_new^2, 0) + epsilon)
        //   mom_new = momentum * mom + lr * grad / denom
        //   var_new = var - mom_new
        // Maxs(..., 0.0f) clamp is shared with fp16 path for numerical safety.
        LocalTensor<float> denom = denomBuf_.template Get<float>();
        LocalTensor<float> tmp1  = tmp1Buf_.template Get<float>();
        LocalTensor<float> tmp2  = tmp2Buf_.template Get<float>();

        // 2. mg_new = rho * mg + (1-rho) * grad
        Muls(tmp1,       mgLocal,   static_cast<T>(rhoScalar_),   n);
        Muls(tmp2,       gradLocal, static_cast<T>(oneMinusRho_), n);
        Add (mgOutLocal, tmp1, tmp2, n);            // mgOut = mg_new

        // 3. ms_new = rho * ms + (1-rho) * grad*grad
        Mul (tmp1,       gradLocal, gradLocal, n);
        Muls(tmp1,       tmp1,      static_cast<T>(oneMinusRho_), n);
        Muls(tmp2,       msLocal,   static_cast<T>(rhoScalar_),   n);
        Add (msOutLocal, tmp1, tmp2, n);            // msOut = ms_new

        // 4. inner = max(ms_new - mg_new^2, 0) + epsilon
        Mul (tmp1, mgOutLocal, mgOutLocal, n);      // tmp1 = mg_new^2
        Sub (tmp2, msOutLocal, tmp1, n);            // tmp2 = ms_new - mg_new^2
        Maxs(tmp2, tmp2, 0.0f, n);                  // clamp >= 0
        Adds(tmp2, tmp2, static_cast<T>(epsilonScalar_), n);

        // 5. denom = sqrt(inner)
        Sqrt(denom, tmp2, n);

        // 6. mom_new = momentum * mom + lr * grad / denom
        Div (tmp1,        gradLocal, denom, n);
        Muls(tmp1,        tmp1,     static_cast<T>(lrScalar_),       n);
        Muls(tmp2,        momLocal, static_cast<T>(momentumScalar_), n);
        Add (momOutLocal, tmp1, tmp2, n);           // momOut = mom_new

        // 7. var_new = var - mom_new
        Sub (varOutLocal, varLocal, momOutLocal, n);
    }

    varOutQue_.template EnQue<T>(varOutLocal);
    mgOutQue_.template EnQue<T>(mgOutLocal);
    msOutQue_.template EnQue<T>(msOutLocal);
    momOutQue_.template EnQue<T>(momOutLocal);

    varInQue_.FreeTensor(varLocal);
    mgInQue_.FreeTensor(mgLocal);
    msInQue_.FreeTensor(msLocal);
    momInQue_.FreeTensor(momLocal);
    gradInQue_.FreeTensor(gradLocal);
}

// =============================================================================
// Process: main loop over UB-sized chunks.
// =============================================================================
template <typename T>
__aicore__ inline void ApplyCenteredRMSProp<T>::Process()
{
    if (blockLen_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLen_ + ubFactor_ - 1) / ubFactor_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t gmOffset = i * ubFactor_;
        int64_t currentNum = (i == (loopCount - 1))
                                 ? (blockLen_ - gmOffset)
                                 : ubFactor_;
        CopyInTile(gmOffset, currentNum);
        Compute(currentNum);
        CopyOutTile(gmOffset, currentNum);
    }
}

} // namespace NsApplyCenteredRMSProp

#endif // APPLY_CENTERED_RMS_PROP_H
