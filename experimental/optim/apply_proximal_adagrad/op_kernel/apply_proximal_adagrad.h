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
 * \file apply_proximal_adagrad.h
 * \brief ApplyProximalAdagrad kernel class (arch35 / Ascend950).
 *
 * Implements the Proximal Adagrad update per element:
 *
 *   accum = accum + grad * grad
 *   eta   = lr * rsqrt(accum)
 *   prox  = var - eta * grad
 *   if (l1 > 0):
 *     var = sign(prox) * max(|prox| - eta*l1, 0) / (1 + eta*l2)
 *   else:
 *     var = prox / (1 + eta*l2)
 *
 * Iteration-2 routes:
 *   - PAD_TAIL: 0 (aligned tail) / 1 (non-aligned tail).  Both branches use
 *     DataCopyPad which is alignment-tolerant; the flag is preserved so future
 *     iterations can swap in a faster pure-DataCopy fast path on PAD_TAIL=0.
 *   - HAS_L1: 1 (general path with sign + soft-threshold).  The kernel also
 *     applies a runtime fast-path when l1Scalar_ == 0 inside this branch (this
 *     is the path taken when host cannot read l1 from GM at tiling time).
 *   - HAS_L1: 0 (TilingKey 10003 dedicated fast path - sign / soft-threshold
 *     statically dropped).  Reachable via host-side hint or UT.
 */

#ifndef APPLY_PROXIMAL_ADAGRAD_H
#define APPLY_PROXIMAL_ADAGRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "apply_proximal_adagrad_tiling_data.h"
#include "apply_proximal_adagrad_tiling_key.h"

namespace NsApplyProximalAdagrad {

// SUG-002 fix: encode the UB resident-tensor count as a kernel-side constant so
// that any change to the buffer layout below MUST update this value. The host
// tiling (apply_proximal_adagrad_tiling.cpp::UB_BUFFER_COUNT) sizes the per-tile
// UB capacity using the same constant; the two values MUST be kept in sync, or
// the tile size computed by the host may exceed actual UB usage and risk
// overflow. The breakdown is:
//   - 3 IN queues  (var / accum / grad) x double-buffer (2)            =  6
//   - 2 OUT queues (var / accum)        x double-buffer (2)            =  4
//   - 4 scratch buffers (eta / prox / thresh / scale), single-buffered =  4
//                                                                  total = 14
constexpr int64_t kUbResidentFp32TensorCount = 14;
static_assert(kUbResidentFp32TensorCount == 14,
              "UB resident tensor count must stay in sync with host tiling "
              "constant UB_BUFFER_COUNT in apply_proximal_adagrad_tiling.cpp; "
              "update both together when adding/removing UB buffers.");

using AscendC::TPipe;
using AscendC::TQue;
using AscendC::TBuf;
using AscendC::QuePosition;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::DataCopyParams;
using AscendC::DataCopyPad;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPadExtParams;
using AscendC::GetBlockIdx;
using AscendC::Add;
using AscendC::Sub;
using AscendC::Mul;
using AscendC::Muls;
using AscendC::Adds;
using AscendC::Abs;
using AscendC::Div;
using AscendC::Rsqrt;
using AscendC::Maxs;
using AscendC::Compare;
using AscendC::Compares;
using AscendC::CompareScalar;
using AscendC::Select;
using AscendC::Duplicate;
using AscendC::CMPMODE;
using AscendC::SELMODE;

template <typename T, bool PAD_TAIL = true, bool HAS_L1 = true>
class ApplyProximalAdagrad {
public:
    __aicore__ inline ApplyProximalAdagrad() {}

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR accum,
                                GM_ADDR lr, GM_ADDR l1, GM_ADDR l2,
                                GM_ADDR grad,
                                GM_ADDR varOut, GM_ADDR accumOut,
                                const ApplyProximalAdagradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInTile(int64_t gmOffset, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOutTile(int64_t gmOffset, int64_t currentNum);

    __aicore__ inline float LoadScalar(const GlobalTensor<T>& src) const;

private:
    TPipe pipe_;
    // IN queues: var / accum / grad, double-buffered.
    TQue<QuePosition::VECIN, 2> varInQue_;
    TQue<QuePosition::VECIN, 2> accumInQue_;
    TQue<QuePosition::VECIN, 2> gradInQue_;
    // OUT queues: var / accum, double-buffered.
    TQue<QuePosition::VECOUT, 2> varOutQue_;
    TQue<QuePosition::VECOUT, 2> accumOutQue_;
    // Scratch compute buffers (VECCALC, not queue-synchronised).
    TBuf<QuePosition::VECCALC> etaBuf_;       // lr * rsqrt(accum)
    TBuf<QuePosition::VECCALC> proxBuf_;      // var - eta * grad
    TBuf<QuePosition::VECCALC> threshBuf_;    // max(|prox| - eta*l1, 0) and helpers
    TBuf<QuePosition::VECCALC> scaleBuf_;     // 1 + eta * l2

    GlobalTensor<T> varGm_;
    GlobalTensor<T> accumGm_;
    GlobalTensor<T> gradGm_;
    GlobalTensor<T> lrGm_;
    GlobalTensor<T> l1Gm_;
    GlobalTensor<T> l2Gm_;
    GlobalTensor<T> varOutGm_;
    GlobalTensor<T> accumOutGm_;

    // Scalars loaded from GM at Init time.
    float lrScalar_ = 0.0f;
    float l1Scalar_ = 0.0f;
    float l2Scalar_ = 0.0f;

    int64_t blockOffset_ = 0;
    int64_t blockLen_ = 0;
    int64_t ubFactor_ = 0;
};

// =============================================================================
// LoadScalar: pick the first element of a 1-element GM tensor into a register.
// =============================================================================
template <typename T, bool PAD_TAIL, bool HAS_L1>
__aicore__ inline float ApplyProximalAdagrad<T, PAD_TAIL, HAS_L1>::LoadScalar(
    const GlobalTensor<T>& src) const
{
    return static_cast<float>(src.GetValue(0));
}

// =============================================================================
// Init
// =============================================================================
template <typename T, bool PAD_TAIL, bool HAS_L1>
__aicore__ inline void ApplyProximalAdagrad<T, PAD_TAIL, HAS_L1>::Init(
    GM_ADDR var, GM_ADDR accum,
    GM_ADDR lr, GM_ADDR l1, GM_ADDR l2,
    GM_ADDR grad,
    GM_ADDR varOut, GM_ADDR accumOut,
    const ApplyProximalAdagradTilingData* tilingData)
{
    ubFactor_ = tilingData->ubFactor;

    // Empty-tensor / degenerate tiling -> just record zero length; Process()
    // will short-circuit.
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
    accumGm_.SetGlobalBuffer((__gm__ T*)accum + blockOffset_, blockLen_);
    gradGm_.SetGlobalBuffer((__gm__ T*)grad + blockOffset_, blockLen_);
    varOutGm_.SetGlobalBuffer((__gm__ T*)varOut + blockOffset_, blockLen_);
    accumOutGm_.SetGlobalBuffer((__gm__ T*)accumOut + blockOffset_, blockLen_);

    // Scalar tensors: single element broadcast-read.
    lrGm_.SetGlobalBuffer((__gm__ T*)lr, 1);
    l1Gm_.SetGlobalBuffer((__gm__ T*)l1, 1);
    l2Gm_.SetGlobalBuffer((__gm__ T*)l2, 1);
    lrScalar_ = LoadScalar(lrGm_);
    l1Scalar_ = LoadScalar(l1Gm_);
    l2Scalar_ = LoadScalar(l2Gm_);

    // UB buffer allocation.
    pipe_.InitBuffer(varInQue_,   2, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(accumInQue_, 2, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(gradInQue_,  2, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(varOutQue_,   2, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(accumOutQue_, 2, ubFactor_ * sizeof(T));

    pipe_.InitBuffer(etaBuf_,    ubFactor_ * sizeof(float));
    pipe_.InitBuffer(proxBuf_,   ubFactor_ * sizeof(float));
    pipe_.InitBuffer(threshBuf_, ubFactor_ * sizeof(float));
    pipe_.InitBuffer(scaleBuf_,  ubFactor_ * sizeof(float));
}

// =============================================================================
// CopyInTile: pad-aware DataCopyPad of var / accum / grad.
//   DataCopyPad transparently handles aligned and non-aligned blockLen, so the
//   same primitive serves both PAD_TAIL=0 (aligned tile) and PAD_TAIL=1
//   (non-aligned tail) routes.
// =============================================================================
template <typename T, bool PAD_TAIL, bool HAS_L1>
__aicore__ inline void ApplyProximalAdagrad<T, PAD_TAIL, HAS_L1>::CopyInTile(
    int64_t gmOffset, int64_t currentNum)
{
    LocalTensor<T> varLocal   = varInQue_.template AllocTensor<T>();
    LocalTensor<T> accumLocal = accumInQue_.template AllocTensor<T>();
    LocalTensor<T> gradLocal  = gradInQue_.template AllocTensor<T>();

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    // ISSUE-001 fix: defensively right-pad the tail lanes ([currentNum, alignedNum))
    // with safe values so Compute's Rsqrt/Div on padding lanes does not produce
    // Inf/NaN. We compute padding count up to the next 8-element (32B) block.
    // The padding lanes are never written back to GM (DataCopyPad honours the
    // exact byte length), so the value choice only matters for in-UB compute
    // stability:
    //   - var / grad: pad with 0.0f -> grad=0 keeps accum unchanged, prox=var.
    //   - accum:      pad with 1.0f -> Rsqrt(accum + grad^2) = Rsqrt(1) = 1,
    //                 avoiding the Rsqrt(0)=+Inf -> NaN cascade.
    constexpr int64_t kAlignBlock = 32 / sizeof(T);
    int64_t alignedNum =
        ((currentNum + kAlignBlock - 1) / kAlignBlock) * kAlignBlock;
    uint8_t rightPadCount = static_cast<uint8_t>(alignedNum - currentNum);
    DataCopyPadExtParams<T> padZeroParams{true, 0, rightPadCount, static_cast<T>(0)};
    DataCopyPadExtParams<T> padOneParams{true, 0, rightPadCount, static_cast<T>(1.0f)};

    DataCopyPad(varLocal,   varGm_[gmOffset],   copyParams, padZeroParams);
    DataCopyPad(accumLocal, accumGm_[gmOffset], copyParams, padOneParams);
    DataCopyPad(gradLocal,  gradGm_[gmOffset],  copyParams, padZeroParams);

    varInQue_.EnQue(varLocal);
    accumInQue_.EnQue(accumLocal);
    gradInQue_.EnQue(gradLocal);
}

// =============================================================================
// CopyOutTile: write back var / accum to their inplace GM slots.  DataCopyPad
//   honours the exact byte length, so non-aligned tails do not over-write
//   neighbouring cores' data.
// =============================================================================
template <typename T, bool PAD_TAIL, bool HAS_L1>
__aicore__ inline void ApplyProximalAdagrad<T, PAD_TAIL, HAS_L1>::CopyOutTile(
    int64_t gmOffset, int64_t currentNum)
{
    LocalTensor<T> varOutLocal = varOutQue_.template DeQue<T>();
    LocalTensor<T> accumOutLocal = accumOutQue_.template DeQue<T>();

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPad(varOutGm_[gmOffset],   varOutLocal,   copyParams);
    DataCopyPad(accumOutGm_[gmOffset], accumOutLocal, copyParams);

    varOutQue_.FreeTensor(varOutLocal);
    accumOutQue_.FreeTensor(accumOutLocal);
}

// =============================================================================
// Compute: main per-tile computation (float32 only in iteration 1/2).
//
// accum' = accum + grad*grad
// eta    = lr * rsqrt(accum')
// prox   = var - eta * grad
//
// HAS_L1 = true  (TilingKey 10001 / 10002 - general path):
//   thresh = max(|prox| - eta*l1, 0)
//   signed = prox >= 0 ? thresh : -thresh
//   var'   = signed / (1 + eta*l2)
//   Includes a runtime fast-path: if l1Scalar_ == 0 the kernel skips the
//   sign / soft-threshold steps and behaves like the HAS_L1=false branch.
//
// HAS_L1 = false (TilingKey 10003 - dedicated fast path):
//   var'   = prox / (1 + eta*l2)
// =============================================================================
template <typename T, bool PAD_TAIL, bool HAS_L1>
__aicore__ inline void ApplyProximalAdagrad<T, PAD_TAIL, HAS_L1>::Compute(
    int64_t currentNum)
{
    LocalTensor<T> varLocal   = varInQue_.template DeQue<T>();
    LocalTensor<T> accumLocal = accumInQue_.template DeQue<T>();
    LocalTensor<T> gradLocal  = gradInQue_.template DeQue<T>();

    LocalTensor<T> varOutLocal   = varOutQue_.template AllocTensor<T>();
    LocalTensor<T> accumOutLocal = accumOutQue_.template AllocTensor<T>();

    LocalTensor<float> etaTmp    = etaBuf_.template Get<float>();
    LocalTensor<float> proxTmp   = proxBuf_.template Get<float>();
    LocalTensor<float> threshTmp = threshBuf_.template Get<float>();
    LocalTensor<float> scaleTmp  = scaleBuf_.template Get<float>();

    // Align work count to 32B / sizeof(float) = 8 elements.  This guarantees
    // Compare / Select 256-byte alignment (ubFactor is a 2048-elem multiple,
    // and we round non-aligned tails up to the next 8-elem block; the extra
    // padding lanes are computed but never written back to GM thanks to
    // DataCopyPad honouring the exact byte length).
    constexpr int64_t kAlignBlock = 32 / sizeof(float);
    int64_t alignedNum =
        ((currentNum + kAlignBlock - 1) / kAlignBlock) * kAlignBlock;
    int32_t n = static_cast<int32_t>(alignedNum);

    // Compile-time specialisation: iteration 1/2 only supports float32.  If
    // new dtypes are added later they should Cast at the beginning and Cast
    // back at the end, leaving the inner block below unchanged.
    if constexpr (std::is_same_v<T, float>) {
        // ----- Common: S1 / S2 / S3 -----
        // S1: accum' = accum + grad*grad  -> write into accumOutLocal.
        Mul(accumOutLocal, gradLocal, gradLocal, n);
        Add(accumOutLocal, accumLocal, accumOutLocal, n);

        // S2: eta = lr * rsqrt(accum')
        // NOTE: Caller contract guarantees accum >= 0 (typically > 0). If
        // accum + grad^2 == 0, Rsqrt returns +Inf and downstream eta*grad
        // becomes Inf*0 = NaN. This matches the semantics of PyTorch's
        // ApplyProximalAdagrad and TensorFlow's ApplyProximalAdagrad, where
        // the safe-input contract is the caller's responsibility (see
        // README.md / DESIGN.md). Defensive padding in CopyInTile ensures
        // the [currentNum, alignedNum) tail lanes use accum=1, not 0, so
        // padding lanes never trigger this path.
        Rsqrt(etaTmp, accumOutLocal, n);
        Muls(etaTmp, etaTmp, lrScalar_, n);

        // S3: prox = var - eta * grad
        Mul(proxTmp, etaTmp, gradLocal, n);
        Sub(proxTmp, varLocal, proxTmp, n);

        // ----- S4 / S5 branch -----
        if constexpr (HAS_L1) {
            // Runtime fast-path: if l1 == 0 the sign + soft-threshold steps
            // collapse to identity, so we behave like the HAS_L1=false branch.
            if (l1Scalar_ == 0.0f) {
                // scale = 1 + eta*l2;  var = prox / scale
                Muls(scaleTmp, etaTmp, l2Scalar_, n);
                Adds(scaleTmp, scaleTmp, 1.0f, n);
                Div(varOutLocal, proxTmp, scaleTmp, n);
            } else {
                // S4a: thresh = max(|prox| - eta*l1, 0)
                Abs(threshTmp, proxTmp, n);
                Muls(scaleTmp, etaTmp, l1Scalar_, n);   // scaleTmp reused as eta*l1
                Sub(threshTmp, threshTmp, scaleTmp, n);
                Maxs(threshTmp, threshTmp, 0.0f, n);

                // S4b: signed_thresh = prox >= 0 ? thresh : -thresh
                Muls(scaleTmp, threshTmp, -1.0f, n);   // scaleTmp = -thresh
                // SUG-001 NOTE: We temporarily reinterpret varOutLocal's storage
                // as a uint8_t mask buffer. This relies on an implicit ordering
                // invariant: varOutLocal must NOT be read between this point and
                // the final Div() below that overwrites it. The mask is consumed
                // by the immediately following Select() call and never read again
                // in this Compute(). If future maintenance inserts any read of
                // varOutLocal before the final Div, this aliasing must be
                // replaced by a dedicated mask buffer (e.g. add a maskBuf_ in
                // Init or carve out the tail of threshBuf_).
                LocalTensor<uint8_t> maskTensor =
                    varOutLocal.template ReinterpretCast<uint8_t>();
                CompareScalar(maskTensor, proxTmp, 0.0f, CMPMODE::GE, n);
                Select(proxTmp, maskTensor, threshTmp, scaleTmp,
                       SELMODE::VSEL_TENSOR_TENSOR_MODE, n);

                // S5: scale = 1 + eta*l2
                Muls(scaleTmp, etaTmp, l2Scalar_, n);
                Adds(scaleTmp, scaleTmp, 1.0f, n);

                // var' = signed_thresh / scale
                Div(varOutLocal, proxTmp, scaleTmp, n);
            }
        } else {
            // HAS_L1 == false (TilingKey 10003): dedicated simplified path.
            // var = prox / (1 + eta*l2)
            Muls(scaleTmp, etaTmp, l2Scalar_, n);
            Adds(scaleTmp, scaleTmp, 1.0f, n);
            Div(varOutLocal, proxTmp, scaleTmp, n);
        }
    }

    varOutQue_.template EnQue<T>(varOutLocal);
    accumOutQue_.template EnQue<T>(accumOutLocal);

    varInQue_.FreeTensor(varLocal);
    accumInQue_.FreeTensor(accumLocal);
    gradInQue_.FreeTensor(gradLocal);
}

// =============================================================================
// Process: main loop over UB-sized chunks.
// =============================================================================
template <typename T, bool PAD_TAIL, bool HAS_L1>
__aicore__ inline void ApplyProximalAdagrad<T, PAD_TAIL, HAS_L1>::Process()
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

} // namespace NsApplyProximalAdagrad

#endif // APPLY_PROXIMAL_ADAGRAD_H
