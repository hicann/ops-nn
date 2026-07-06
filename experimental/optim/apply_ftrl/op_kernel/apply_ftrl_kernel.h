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
 * \file apply_ftrl_kernel.h
 * \brief ApplyFtrl kernel class (ascend910b / DAV_2201, classic vector intrinsic).
 *
 * Per-element FTRL-proximal update; the right-hand accum/linear/var are the
 * pre-update values, var uses the updated linear (linear_t).  Internal compute is
 * always fp32 (bf16/fp16 Cast up on entry, Cast back on exit; fp32 ReinterpretCast).
 *
 *   accum_new = accum + grad*grad
 *   pow_new   = accum_new^(-lr_power) = Exp(Muls(Ln(accum_new), -lr_power))
 *   pow_old   = accum^(-lr_power)
 *   linear_t  = linear + grad - ((pow_new - pow_old)/lr) * var
 *   quadratic = pow_new/lr + 2*l2
 *   x_res     = sign(linear_t)*l1 - linear_t
 *   var       = (|linear_t| > l1) ? x_res/quadratic : 0
 *   accum_out = accum_new ; linear_out = linear_t ; var_out = var
 *
 * Three outputs are written in place: var -> var_out (declared output port),
 * accum/linear -> their own input GM (Ref Tensor), matching the arch35 DAG ABI
 * sch.Init(..., var_out, accum, linear) (DESIGN s3.4).
 *
 * Template params (driven by ASCENDC_TPL_SEL_PARAM on the host):
 *   - T        : input/output dtype (half / bfloat16_t / float).
 *   - PAD_TAIL : non-32B-aligned tail flag (kernel always DataCopyPad; reserved).
 *   - HAS_L1   : 1 = general sign + soft-threshold + gate path (with a runtime
 *                l1==0 fast path); 0 = statically dropped fast path.
 */

#ifndef APPLY_FTRL_KERNEL_H
#define APPLY_FTRL_KERNEL_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "apply_ftrl_tiling_data.h"
#include "apply_ftrl_tiling_key.h"

namespace NsApplyFtrl {

// Internal compute dtype is always fp32 -> CompareScalar / Select count must be
// 256B aligned.  256B / sizeof(float) = 64 elements; tile tails round up to this.
constexpr int64_t kCmpAlignElemFp32 = 256 / static_cast<int64_t>(sizeof(float));

// SUG-002 style sync constant: per-element fp32-equivalent UB slots reserved by
// the host tiling.  MUST stay in sync with `UB_FP32_SLOTS` in
// op_host/apply_ftrl_tiling.cpp; the fp32 path (tightest) uses ~20.25 slots/elem
// (IN 8*4 + OUT 6*4 + scratch 24 + mask 1 = 81B / 4B), 24 keeps headroom.  Any
// change to the buffer layout below MUST update both constants together.
constexpr int64_t kUbFp32Slots = 24;
static_assert(kUbFp32Slots == 24, "UB fp32 slot count must stay in sync with host tiling constant "
                                  "UB_FP32_SLOTS in apply_ftrl_tiling.cpp; update both together when "
                                  "adding/removing UB buffers.");

// Scalar staging buffer (32B) for the bf16 scalar borrow path.
constexpr int32_t kScalarBufBytes = 32;

using AscendC::Abs;
using AscendC::Add;
using AscendC::Adds;
using AscendC::Cast;
using AscendC::CMPMODE;
using AscendC::CompareScalar;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPad;
using AscendC::DataCopyPadExtParams;
using AscendC::Div;
using AscendC::Duplicate;
using AscendC::Exp;
using AscendC::GetBlockIdx;
using AscendC::GlobalTensor;
using AscendC::Ln;
using AscendC::LocalTensor;
using AscendC::Mul;
using AscendC::Muls;
using AscendC::PipeBarrier;
using AscendC::QuePosition;
using AscendC::RoundMode;
using AscendC::Select;
using AscendC::SELMODE;
using AscendC::Sub;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TQue;

template <typename T, bool PAD_TAIL = true, bool HAS_L1 = true>
class ApplyFtrl {
    // fp32: ReinterpretCast IN tensors as the fp32 source (no separate buffers).
    static constexpr bool IS_FLOAT = std::is_same<T, float>::value;
    // half/fp32 scalars cast directly from a register; bf16 must borrow via a
    // LocalTensor + Vector Cast (bisheng forbids bf16 scalar static_cast).
    static constexpr bool DIRECT_SCALAR = std::is_same<T, float>::value || std::is_same<T, half>::value;
    // DMA alignment block (32B): fp32 -> 8, half/bf16 -> 16 elements.  DataCopyPad
    // rightPadding cannot span more than one 32B block, so the tail is padded only
    // to this granularity; the remaining [nDma, nComp) gap is filled in the fp32
    // domain via Duplicate (NPU-validated two-level alignment; 507035 root cause).
    static constexpr int64_t kDmaBlockElem = 32 / static_cast<int64_t>(sizeof(T));

public:
    __aicore__ inline ApplyFtrl() {}

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR accum, GM_ADDR linear, GM_ADDR grad, GM_ADDR lr, GM_ADDR l1,
                                GM_ADDR l2, GM_ADDR lrPower, GM_ADDR varOut, const ApplyFtrlTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInTile(int64_t gmOffset, int64_t currentNum, int64_t nDma);
    __aicore__ inline void Compute(int64_t nDma, int64_t nComp);
    __aicore__ inline void CopyOutTile(int64_t gmOffset, int64_t currentNum);

    __aicore__ inline float LoadScalar(GM_ADDR gmAddr);

private:
    TPipe pipe_;
    // IN queues (dtype T, double-buffered): var / accum / linear / grad.
    TQue<QuePosition::VECIN, 2> qVar_;
    TQue<QuePosition::VECIN, 2> qAccum_;
    TQue<QuePosition::VECIN, 2> qLinear_;
    TQue<QuePosition::VECIN, 2> qGrad_;
    // OUT queues (dtype T, double-buffered): var / accum / linear.
    TQue<QuePosition::VECOUT, 2> qVarOut_;
    TQue<QuePosition::VECOUT, 2> qAccumOut_;
    TQue<QuePosition::VECOUT, 2> qLinearOut_;
    // fp32 source buffers (only allocated on the bf16/fp16 path).
    TBuf<QuePosition::VECCALC> bVarF_;
    TBuf<QuePosition::VECCALC> bAccumF_;
    TBuf<QuePosition::VECCALC> bLinearF_;
    TBuf<QuePosition::VECCALC> bGradF_;
    // fp32 scratch buffers (always allocated).
    TBuf<QuePosition::VECCALC> bAccumNew_;
    TBuf<QuePosition::VECCALC> bPowNew_;
    TBuf<QuePosition::VECCALC> bLinearNew_;
    TBuf<QuePosition::VECCALC> bQuad_;
    TBuf<QuePosition::VECCALC> bTmpA_;
    TBuf<QuePosition::VECCALC> bTmpB_;
    // uint8 mask buffer (for CompareScalar + Select).
    TBuf<QuePosition::VECCALC> bMask_;
    // Scalar staging buffers (only allocated on the bf16 borrow path).
    TBuf<QuePosition::VECCALC> bScalarRaw_;
    TBuf<QuePosition::VECCALC> bScalarF32_;

    GlobalTensor<T> varGm_;
    GlobalTensor<T> accumGm_;  // read + in-place write back (Ref Tensor).
    GlobalTensor<T> linearGm_; // read + in-place write back (Ref Tensor).
    GlobalTensor<T> gradGm_;
    GlobalTensor<T> varOutGm_;

    // host-like fp32 scalars (loaded once at Init).
    float lrScalar_ = 0.0f;
    float l1Scalar_ = 0.0f;
    float l2Scalar_ = 0.0f;
    float lrPowerScalar_ = 0.0f;
    float invLr_ = 0.0f;
    float negLrPower_ = 0.0f;
    float twoL2_ = 0.0f;

    int64_t blockOffset_ = 0;
    int64_t blockLen_ = 0;
    int64_t ubFactor_ = 0;
};

// =============================================================================
// LoadScalar: read a 1-element scalar GM tensor into an fp32 register.
//   half/fp32: scalar static_cast is legal.
//   bf16:      borrow via DataCopyPad to UB + Vector Cast (bisheng forbids the
//              bf16 scalar static_cast).
// =============================================================================
template <typename T, bool PAD_TAIL, bool HAS_L1>
__aicore__ inline float ApplyFtrl<T, PAD_TAIL, HAS_L1>::LoadScalar(GM_ADDR gmAddr)
{
    GlobalTensor<T> gm;
    gm.SetGlobalBuffer((__gm__ T*)gmAddr, 1);
    if constexpr (DIRECT_SCALAR) {
        return static_cast<float>(gm.GetValue(0));
    } else {
        LocalTensor<T> rawUb = bScalarRaw_.template Get<T>();
        LocalTensor<float> f32Ub = bScalarF32_.template Get<float>();
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint32_t>(sizeof(T));
        params.srcStride = 0;
        params.dstStride = 0;
        DataCopyPad(rawUb, gm, params, DataCopyPadExtParams<T>{false, 0, 0, 0});
        PipeBarrier<PIPE_ALL>();
        // 8 == 32B / sizeof(float); only element 0 is meaningful.
        Cast(f32Ub, rawUb, RoundMode::CAST_NONE, 8);
        PipeBarrier<PIPE_ALL>();
        return f32Ub.GetValue(0);
    }
}

// =============================================================================
// Init
// =============================================================================
template <typename T, bool PAD_TAIL, bool HAS_L1>
__aicore__ inline void ApplyFtrl<T, PAD_TAIL, HAS_L1>::Init(GM_ADDR var, GM_ADDR accum, GM_ADDR linear, GM_ADDR grad,
                                                            GM_ADDR lr, GM_ADDR l1, GM_ADDR l2, GM_ADDR lrPower,
                                                            GM_ADDR varOut, const ApplyFtrlTilingData* tilingData)
{
    ubFactor_ = tilingData->ubFactor;

    // Empty-tensor / degenerate tiling -> Process() short-circuits.
    if (tilingData->totalElements == 0 || tilingData->blockFactor == 0) {
        blockLen_ = 0;
        return;
    }

    blockOffset_ = tilingData->blockFactor * static_cast<int64_t>(GetBlockIdx());
    int64_t remaining = tilingData->totalElements - blockOffset_;
    if (remaining <= 0) {
        blockLen_ = 0;
        return;
    }
    blockLen_ = (remaining > tilingData->blockFactor) ? tilingData->blockFactor : remaining;

    // Per-core view; accum/linear use the same GM for read and in-place write-back.
    varGm_.SetGlobalBuffer((__gm__ T*)var + blockOffset_, blockLen_);
    accumGm_.SetGlobalBuffer((__gm__ T*)accum + blockOffset_, blockLen_);
    linearGm_.SetGlobalBuffer((__gm__ T*)linear + blockOffset_, blockLen_);
    gradGm_.SetGlobalBuffer((__gm__ T*)grad + blockOffset_, blockLen_);
    varOutGm_.SetGlobalBuffer((__gm__ T*)varOut + blockOffset_, blockLen_);

    // UB buffers (sized to ubFactor_; ubFactor_ is a 256B/64-element multiple).
    pipe_.InitBuffer(qVar_, 2, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(qAccum_, 2, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(qLinear_, 2, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(qGrad_, 2, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(qVarOut_, 2, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(qAccumOut_, 2, ubFactor_ * sizeof(T));
    pipe_.InitBuffer(qLinearOut_, 2, ubFactor_ * sizeof(T));

    if constexpr (!IS_FLOAT) {
        pipe_.InitBuffer(bVarF_, ubFactor_ * sizeof(float));
        pipe_.InitBuffer(bAccumF_, ubFactor_ * sizeof(float));
        pipe_.InitBuffer(bLinearF_, ubFactor_ * sizeof(float));
        pipe_.InitBuffer(bGradF_, ubFactor_ * sizeof(float));
    }
    pipe_.InitBuffer(bAccumNew_, ubFactor_ * sizeof(float));
    pipe_.InitBuffer(bPowNew_, ubFactor_ * sizeof(float));
    pipe_.InitBuffer(bLinearNew_, ubFactor_ * sizeof(float));
    pipe_.InitBuffer(bQuad_, ubFactor_ * sizeof(float));
    pipe_.InitBuffer(bTmpA_, ubFactor_ * sizeof(float));
    pipe_.InitBuffer(bTmpB_, ubFactor_ * sizeof(float));
    pipe_.InitBuffer(bMask_, ubFactor_ * sizeof(uint8_t));
    if constexpr (!DIRECT_SCALAR) {
        pipe_.InitBuffer(bScalarRaw_, kScalarBufBytes);
        pipe_.InitBuffer(bScalarF32_, kScalarBufBytes);
    }

    // Load scalars once and precompute host-like fp32 derived values.
    lrScalar_ = LoadScalar(lr);
    l1Scalar_ = LoadScalar(l1);
    l2Scalar_ = LoadScalar(l2);
    lrPowerScalar_ = LoadScalar(lrPower);
    invLr_ = 1.0f / lrScalar_;
    negLrPower_ = -lrPowerScalar_;
    twoL2_ = 2.0f * l2Scalar_;
}

// =============================================================================
// CopyInTile: pad-aware DataCopyPad of var / accum / linear / grad.
//   Two-level tail alignment (NPU-validated, 507035 root cause): DataCopyPad's
//   rightPadding cannot span more than one 32B block, so here the tail lanes
//   [currentNum, nDma) are padded ONLY to the 32B DMA block (nDma = ceil(currentNum,
//   kDmaBlockElem); rightPad <= 15 for fp16/bf16, <= 7 for fp32 -> hardware-legal).
//   The remaining [nDma, nComp) gap up to the 256B compute width is filled later in
//   the fp32 domain via Duplicate (see Compute).  Pad values:
//     - accum -> 1.0  (Ln(1)=0, avoids the Ln(0)=-Inf cascade in the _pow chain)
//     - var/linear/grad -> 0.0
//   Padding lanes are never written back to GM (CopyOut honours the exact byte
//   length), so the pad value only matters for in-UB compute stability.
//   PAD_TAIL=0 / 32B-aligned tile -> rightPad == 0 (fast path unchanged).
// =============================================================================
template <typename T, bool PAD_TAIL, bool HAS_L1>
__aicore__ inline void ApplyFtrl<T, PAD_TAIL, HAS_L1>::CopyInTile(int64_t gmOffset, int64_t currentNum, int64_t nDma)
{
    LocalTensor<T> varLocal = qVar_.template AllocTensor<T>();
    LocalTensor<T> accumLocal = qAccum_.template AllocTensor<T>();
    LocalTensor<T> linearLocal = qLinear_.template AllocTensor<T>();
    LocalTensor<T> gradLocal = qGrad_.template AllocTensor<T>();

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    // rightPad <= kDmaBlockElem-1 (one 32B block) -> hardware-feasible DataCopyPad.
    uint8_t rightPad = static_cast<uint8_t>(nDma - currentNum);
    DataCopyPadExtParams<T> padZero{true, 0, rightPad, static_cast<T>(0)};
    DataCopyPadExtParams<T> padOne{true, 0, rightPad, static_cast<T>(1.0f)};

    DataCopyPad(varLocal, varGm_[gmOffset], copyParams, padZero);
    DataCopyPad(accumLocal, accumGm_[gmOffset], copyParams, padOne);
    DataCopyPad(linearLocal, linearGm_[gmOffset], copyParams, padZero);
    DataCopyPad(gradLocal, gradGm_[gmOffset], copyParams, padZero);

    qVar_.EnQue(varLocal);
    qAccum_.EnQue(accumLocal);
    qLinear_.EnQue(linearLocal);
    qGrad_.EnQue(gradLocal);
}

// =============================================================================
// Compute: full FTRL chain in fp32, then convert back to T.
//   nDma  = real + 32B-pad valid lanes (from DataCopyPad);
//   nComp = 256B/64-elem compute width (CompareScalar/Select count rule), n.
//   The fp32 sources are made valid over [0, nComp): real+32B-pad in [0, nDma)
//   plus Ln-safe defaults (accum=1.0, others=0.0) in the [nDma, nComp) gap
//   (NPU-validated two-level alignment; 507035 root cause).
// =============================================================================
template <typename T, bool PAD_TAIL, bool HAS_L1>
__aicore__ inline void ApplyFtrl<T, PAD_TAIL, HAS_L1>::Compute(int64_t nDma, int64_t nComp)
{
    int32_t nd = static_cast<int32_t>(nDma);
    int32_t n = static_cast<int32_t>(nComp);

    LocalTensor<T> varLocal = qVar_.template DeQue<T>();
    LocalTensor<T> accumLocal = qAccum_.template DeQue<T>();
    LocalTensor<T> linearLocal = qLinear_.template DeQue<T>();
    LocalTensor<T> gradLocal = qGrad_.template DeQue<T>();

    LocalTensor<T> varOutLocal = qVarOut_.template AllocTensor<T>();
    LocalTensor<T> accumOutLocal = qAccumOut_.template AllocTensor<T>();
    LocalTensor<T> linearOutLocal = qLinearOut_.template AllocTensor<T>();

    LocalTensor<float> bAccumNew = bAccumNew_.template Get<float>();
    LocalTensor<float> bPowNew = bPowNew_.template Get<float>();
    LocalTensor<float> bLinearNew = bLinearNew_.template Get<float>();
    LocalTensor<float> bQuad = bQuad_.template Get<float>();
    LocalTensor<float> bTmpA = bTmpA_.template Get<float>();
    LocalTensor<float> bTmpB = bTmpB_.template Get<float>();
    LocalTensor<uint8_t> mask = bMask_.template Get<uint8_t>();

    // Establish fp32 sources valid over [0, nComp) with the two-level fill.
    LocalTensor<float> srcVar;
    LocalTensor<float> srcAccum;
    LocalTensor<float> srcLinear;
    LocalTensor<float> srcGrad;
    if constexpr (IS_FLOAT) {
        // fp32: ReinterpretCast IN tensors; real+32B-pad already in [0, nDma).
        srcVar = varLocal.template ReinterpretCast<float>();
        srcAccum = accumLocal.template ReinterpretCast<float>();
        srcLinear = linearLocal.template ReinterpretCast<float>();
        srcGrad = gradLocal.template ReinterpretCast<float>();
        // Fill the [nDma, nComp) gap with Ln-safe defaults (no Cast on fp32 path).
        if (nComp > nDma) {
            int32_t gap = static_cast<int32_t>(nComp - nDma);
            Duplicate(srcAccum[nDma], 1.0f, gap);
            Duplicate(srcVar[nDma], 0.0f, gap);
            Duplicate(srcLinear[nDma], 0.0f, gap);
            Duplicate(srcGrad[nDma], 0.0f, gap);
        }
    } else {
        // bf16/fp16: pre-fill [0, nComp) with Ln-safe defaults, then Cast overwrites
        // the real+32B-pad region [0, nDma); the [nDma, nComp) gap keeps defaults.
        srcVar = bVarF_.template Get<float>();
        srcAccum = bAccumF_.template Get<float>();
        srcLinear = bLinearF_.template Get<float>();
        srcGrad = bGradF_.template Get<float>();
        if (nComp > nDma) {
            Duplicate(srcAccum, 1.0f, n);
            Duplicate(srcVar, 0.0f, n);
            Duplicate(srcLinear, 0.0f, n);
            Duplicate(srcGrad, 0.0f, n);
        }
        Cast(srcVar, varLocal, RoundMode::CAST_NONE, nd);
        Cast(srcAccum, accumLocal, RoundMode::CAST_NONE, nd);
        Cast(srcLinear, linearLocal, RoundMode::CAST_NONE, nd);
        Cast(srcGrad, gradLocal, RoundMode::CAST_NONE, nd);
    }

    // S1: accum_new = accum + grad*grad
    Mul(bAccumNew, srcGrad, srcGrad, n);
    Add(bAccumNew, srcAccum, bAccumNew, n);

    // S2: pow_new = accum_new^(-lr_power) = exp(-lr_power * ln(accum_new))
    Ln(bTmpA, bAccumNew, n);
    Muls(bTmpA, bTmpA, negLrPower_, n);
    Exp(bPowNew, bTmpA, n);

    // S3: pow_old = accum^(-lr_power)
    Ln(bTmpB, srcAccum, n);
    Muls(bTmpB, bTmpB, negLrPower_, n);
    Exp(bTmpB, bTmpB, n);

    // S4: (pow_new - pow_old)/lr
    Sub(bTmpA, bPowNew, bTmpB, n);
    Muls(bTmpA, bTmpA, invLr_, n);

    // S5: linear_t = linear + grad - (...)*var
    Mul(bTmpA, bTmpA, srcVar, n);
    Sub(bLinearNew, srcGrad, bTmpA, n);
    Add(bLinearNew, srcLinear, bLinearNew, n);

    // S6: quadratic = pow_new/lr + 2*l2
    Muls(bQuad, bPowNew, invLr_, n);
    Adds(bQuad, bQuad, twoL2_, n);

    // S7-S9: var = (|linear_t| > l1) ? (sign(linear_t)*l1 - linear_t)/quadratic : 0
    //   bTmpB carries the final var result for both branches.
    if constexpr (HAS_L1) {
        if (l1Scalar_ == 0.0f) {
            // Runtime fast path: l1 == 0 -> var = -linear_t / quadratic (DESIGN s3.1).
            Muls(bTmpA, bLinearNew, -1.0f, n);
            Div(bTmpB, bTmpA, bQuad, n);
        } else {
            // x_res = sign(linear_t)*l1 - linear_t.
            //   linear_t >= 0 -> ( l1 - linear_t)   [posVal in bTmpA]
            //   linear_t <  0 -> (-l1 - linear_t)   [negVal in bTmpB]
            // (linear_t == 0 yields posVal = l1, but |0| > l1 is false so the gate
            //  zeroes it -> consistent with sign(0)*l1 - 0 = 0.)
            Muls(bTmpA, bLinearNew, -1.0f, n);        // -linear_t
            Adds(bTmpA, bTmpA, l1Scalar_, n);         // posVal = l1 - linear_t
            Adds(bTmpB, bTmpA, -2.0f * l1Scalar_, n); // negVal = -l1 - linear_t
            CompareScalar(mask, bLinearNew, 0.0f, CMPMODE::GE, n);
            Select(bPowNew, mask, bTmpA, bTmpB, SELMODE::VSEL_TENSOR_TENSOR_MODE, n); // x_res

            // var_true = x_res / quadratic
            Div(bTmpA, bPowNew, bQuad, n);

            // gate: |linear_t| > l1 ? var_true : 0
            Abs(bTmpB, bLinearNew, n);
            CompareScalar(mask, bTmpB, l1Scalar_, CMPMODE::GT, n);
            Duplicate(bPowNew, 0.0f, n);                                              // zeros (bPowNew free after S6)
            Select(bTmpB, mask, bTmpA, bPowNew, SELMODE::VSEL_TENSOR_TENSOR_MODE, n); // var
        }
    } else {
        // HAS_L1=0 compile-time fast path: var = -linear_t / quadratic.
        Muls(bTmpA, bLinearNew, -1.0f, n);
        Div(bTmpB, bTmpA, bQuad, n);
    }

    // Convert the three fp32 results back to T (in-place write-back ABI):
    //   accum_new -> accum_out ; linear_t -> linear_out ; var -> var_out.
    if constexpr (IS_FLOAT) {
        Adds(accumOutLocal.template ReinterpretCast<float>(), bAccumNew, 0.0f, n);
        Adds(linearOutLocal.template ReinterpretCast<float>(), bLinearNew, 0.0f, n);
        Adds(varOutLocal.template ReinterpretCast<float>(), bTmpB, 0.0f, n);
    } else {
        Cast(accumOutLocal, bAccumNew, RoundMode::CAST_RINT, n);
        Cast(linearOutLocal, bLinearNew, RoundMode::CAST_RINT, n);
        Cast(varOutLocal, bTmpB, RoundMode::CAST_RINT, n);
    }

    qAccumOut_.template EnQue<T>(accumOutLocal);
    qLinearOut_.template EnQue<T>(linearOutLocal);
    qVarOut_.template EnQue<T>(varOutLocal);

    qVar_.FreeTensor(varLocal);
    qAccum_.FreeTensor(accumLocal);
    qLinear_.FreeTensor(linearLocal);
    qGrad_.FreeTensor(gradLocal);
}

// =============================================================================
// CopyOutTile: write var/accum/linear back, exact byte length (non-aligned tail
// never over-writes the neighbouring core's region).  accum/linear go to their
// own input GM (Ref Tensor in-place); var goes to var_out.
// =============================================================================
template <typename T, bool PAD_TAIL, bool HAS_L1>
__aicore__ inline void ApplyFtrl<T, PAD_TAIL, HAS_L1>::CopyOutTile(int64_t gmOffset, int64_t currentNum)
{
    LocalTensor<T> accumOutLocal = qAccumOut_.template DeQue<T>();
    LocalTensor<T> linearOutLocal = qLinearOut_.template DeQue<T>();
    LocalTensor<T> varOutLocal = qVarOut_.template DeQue<T>();

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(currentNum * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    DataCopyPad(accumGm_[gmOffset], accumOutLocal, copyParams);
    DataCopyPad(linearGm_[gmOffset], linearOutLocal, copyParams);
    DataCopyPad(varOutGm_[gmOffset], varOutLocal, copyParams);

    qAccumOut_.FreeTensor(accumOutLocal);
    qLinearOut_.FreeTensor(linearOutLocal);
    qVarOut_.FreeTensor(varOutLocal);
}

// =============================================================================
// Process: loop over UB-sized chunks.  Each tile derives two alignment widths:
//   nDma  = ceil(currentNum, kDmaBlockElem)  -- 32B DMA pad granularity;
//   nComp = ceil(currentNum, kCmpAlignElemFp32) -- 256B fp32 compute width.
// =============================================================================
template <typename T, bool PAD_TAIL, bool HAS_L1>
__aicore__ inline void ApplyFtrl<T, PAD_TAIL, HAS_L1>::Process()
{
    if (blockLen_ <= 0) {
        return;
    }
    int64_t loopCount = (blockLen_ + ubFactor_ - 1) / ubFactor_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t gmOffset = i * ubFactor_;
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLen_ - gmOffset) : ubFactor_;
        int64_t nDma = ((currentNum + kDmaBlockElem - 1) / kDmaBlockElem) * kDmaBlockElem;
        int64_t nComp = ((currentNum + kCmpAlignElemFp32 - 1) / kCmpAlignElemFp32) * kCmpAlignElemFp32;
        CopyInTile(gmOffset, currentNum, nDma);
        Compute(nDma, nComp);
        CopyOutTile(gmOffset, currentNum);
    }
}

} // namespace NsApplyFtrl

#endif // APPLY_FTRL_KERNEL_H
