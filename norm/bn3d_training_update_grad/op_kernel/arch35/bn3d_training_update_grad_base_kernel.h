/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// bn3d_training_update_grad_package/op_kernel/arch35/bn3d_training_update_grad_base_kernel.h
// =============================================================================
//
// Ascend C kernel implementation for BN3DTrainingUpdateGrad on arch35 (Ascend950).
// Reduction (binary-cache-tree) + Broadcast (mean NDDMA / inv_std UB broadcast) paradigm.
//
//   This header carries the shared kernel utilities (namespace-level constants,
//   Bn3d* arithmetic helpers, __simd_vf__ VF kernels) plus the base kernel
//   BN3DTrainingUpdateGradBaseKernel (tilingKey 0). The group / empty variants
//   live in bn3d_training_update_grad_group_kernel.h /
//   bn3d_training_update_grad_empty_kernel.h and reuse these utilities.
//
//   Math (per golden.py):
//     rstd_c      = 1 / sqrt(batch_variance_c + epsilon)          (per channel C=A axis)
//     x_norm      = (x - batch_mean) * rstd                       (broadcast mean/rstd over R)
//     diff_scale  = sum_over_R( grads * x_norm )                  (keep C)
//     diff_offset = sum_over_R( grads )                           (keep C)
//
//   tuple-reduce: two serial passes over the same UB-split / binary-cache-tree:
//     processIdx 0 (p_offset): preOut = cast(grads)          -> Sum_R -> diff_offset
//     processIdx 1 (p_scale) : preOut = cast(grads) * x_norm -> Sum_R -> diff_scale
//
//   Structurally adapted from the reduction binary-base reference kernel
//   (euclidean_norm_base.h): TBuf + Mutex Lock/Unlock segment sync, ReduceSum(AR/RA),
//   DataCopyPad+Loop transpose CopyIn, NDDMA broadcast, __simd_vf__ + asc_vf_call VF.
//
//   NOTE (epsilon): 「eps 由 host 透传」, the attr epsilon is
//   read on host (GetFloat(0), nullptr→default 0.0001) into TilingData.epsilon and the
//   kernel consumes td_->epsilon per case for rstd=1/sqrt(batch_variance+epsilon).
// =============================================================================

#ifndef BN3D_TRAINING_UPDATE_GRAD_BASE_KERNEL_H_
#define BN3D_TRAINING_UPDATE_GRAD_BASE_KERNEL_H_

#include "kernel_operator.h"
#include "adv_api/reduce/reduce.h"
#include "bn3d_training_update_grad_tiling_struct.h"
#include "bn3d_training_update_grad_struct.h"

using namespace AscendC;

// ---------------------------------------------------------------------------
// Namespace-level constants / helpers (Ascend950 fixed platform params).
// ---------------------------------------------------------------------------
constexpr uint32_t kVlBytes = 256;                     // vector register width (bytes)
constexpr uint32_t kRepF32 = kVlBytes / sizeof(float); // 64 fp32 lanes / repeat
constexpr uint16_t kRepF32U16 = static_cast<uint16_t>(kRepF32);
constexpr uint32_t kUbBlockBytes = 32;                          // UB block granularity (bytes)
constexpr uint32_t kUbBlockF32 = kUbBlockBytes / sizeof(float); // 8
constexpr int32_t kAxisInterval = 2;                            // even idx=A, odd idx=R
constexpr size_t kBytesPerB16 = 2;
constexpr size_t kReduceShapeDim = 2;
constexpr float kPadClearValue = 0.0f; // sum reducer pad value

__aicore__ inline int64_t Bn3dCeilDiv(int64_t a, int64_t b) { return (b == 0) ? 0 : (a + b - 1) / b; }
__aicore__ inline int64_t Bn3dCeilAlign(int64_t v, int64_t f) { return (f == 0) ? v : Bn3dCeilDiv(v, f) * f; }

// Binary-cache-tree helpers (loop form).
__aicore__ inline int64_t Bn3dFindNearestPower2(int64_t v)
{
    if (v == 0)
        return 0;
    if (v <= 2)
        return 1;
    int64_t p = 1;
    while ((p << 1) < v)
        p <<= 1; // largest power-of-2 strictly less than v
    return p;
}
__aicore__ inline uint16_t Bn3dGetCacheID(int64_t idx)
{
    int64_t x = idx ^ (idx + 1); // trailing run of 1s -> 0..011..1
    int32_t c = 0;
    while (x) {
        c += static_cast<int32_t>(x & 1);
        x >>= 1;
    }
    return static_cast<uint16_t>(c - 1);
}
__aicore__ inline int64_t Bn3dCalLog2(int64_t v)
{
    int64_t l = 0;
    while ((static_cast<int64_t>(1) << (l + 1)) <= v)
        ++l;
    return l;
}

// b16 -> fp32 widen / fp32 -> b16 narrow cast traits.
constexpr AscendC::Reg::CastTrait kCastTraitToFp32{AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                   AscendC::Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_NONE};

template <typename T>
__aicore__ inline constexpr bool Bn3dNeedCast()
{
    return !std::is_same_v<T, float>;
}

// ===========================================================================
// __simd_vf__ VF impls (C-style free functions; no class members, no runtime if).
// ===========================================================================

// p_offset PreElewise: g_f32 = cast(grads) -> preOut (flat).
template <typename DType>
__simd_vf__ inline void Bn3dCastGradsVfImpl(__ubuf__ DType* src, __ubuf__ float* dst, uint32_t totalElems,
                                            uint16_t repeatTime)
{
    AscendC::Reg::RegTensor<float> f32Reg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        if constexpr (Bn3dNeedCast<DType>()) {
            AscendC::Reg::RegTensor<DType> bReg;
            AscendC::Reg::LoadAlign<DType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bReg, src + off);
            AscendC::Reg::Cast<float, DType, kCastTraitToFp32>(f32Reg, bReg, mask);
        } else {
            AscendC::Reg::LoadAlign(f32Reg, reinterpret_cast<__ubuf__ float*>(src) + off);
        }
        AscendC::Reg::StoreAlign(dst + off, f32Reg, mask);
    }
}

// p_scale invStd (per-A): var_eps = var + eps -> std = sqrt(var_eps) -> invStd = 1/std.
__simd_vf__ inline void Bn3dInvStdPerAVfImpl(__ubuf__ float* varUb, __ubuf__ float* invStdOut, float eps,
                                             uint32_t laneAValid, uint16_t repeatTime)
{
    AscendC::Reg::RegTensor<float> vReg, oneReg;
    AscendC::Reg::MaskReg mask;
    AscendC::Reg::Duplicate(oneReg, 1.0f);
    uint32_t remaining = laneAValid;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(vReg, varUb + off);
        AscendC::Reg::Adds(vReg, vReg, eps, mask);   // var + eps
        AscendC::Reg::Sqrt(vReg, vReg, mask);        // std = sqrt(var+eps)
        AscendC::Reg::Div(vReg, oneReg, vReg, mask); // invStd = 1 / std
        AscendC::Reg::StoreAlign(invStdOut + off, vReg, mask);
    }
}

// p_scale VFCHAIN_001 (tail-R / AB): x_f32=cast(x); x_centered=x_f32-mean_bc; x_norm=x_centered*inv_std_bc.
template <typename DType>
__simd_vf__ inline void Bn3dXNormTailRVfImpl(__ubuf__ DType* xUb, __ubuf__ float* meanBc, __ubuf__ float* invStdPerA,
                                             __ubuf__ float* xNormOut, uint32_t rBundle, uint16_t aEntriesU16,
                                             uint16_t repPerRow)
{
    AscendC::Reg::RegTensor<float> xReg, mReg, invReg;
    AscendC::Reg::MaskReg mask;
    for (uint16_t a = 0; a < aEntriesU16; ++a) {
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(invReg,
                                                                             invStdPerA + a); // read1->all lanes
        int32_t rowBase = static_cast<int32_t>(a) * static_cast<int32_t>(rBundle);
        uint32_t remaining = rBundle;
        for (uint16_t r = 0; r < repPerRow; ++r) {
            int32_t off = rowBase + static_cast<int32_t>(r) * static_cast<int32_t>(kRepF32);
            mask = AscendC::Reg::UpdateMask<float>(remaining);
            if constexpr (Bn3dNeedCast<DType>()) {
                AscendC::Reg::RegTensor<DType> bReg;
                AscendC::Reg::LoadAlign<DType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bReg, xUb + off);
                AscendC::Reg::Cast<float, DType, kCastTraitToFp32>(xReg, bReg, mask);
            } else {
                AscendC::Reg::LoadAlign(xReg, reinterpret_cast<__ubuf__ float*>(xUb) + off);
            }
            AscendC::Reg::LoadAlign(mReg, meanBc + off);
            AscendC::Reg::Sub(xReg, xReg, mReg, mask);
            AscendC::Reg::Mul(xReg, xReg, invReg, mask);
            AscendC::Reg::StoreAlign(xNormOut + off, xReg, mask);
        }
    }
}

// p_scale VFCHAIN_001 (tail-A / BA): inv_std varies along inner A (tail axis), repeats along outer R rows.
template <typename DType>
__simd_vf__ inline void Bn3dXNormTailAVfImpl(__ubuf__ DType* xUb, __ubuf__ float* meanBc, __ubuf__ float* invStdPerA,
                                             __ubuf__ float* xNormOut, uint32_t aBundle, uint16_t rRowsU16,
                                             uint16_t repPerRow)
{
    AscendC::Reg::RegTensor<float> xReg, mReg, invReg;
    AscendC::Reg::MaskReg mask;
    for (uint16_t r = 0; r < rRowsU16; ++r) {
        int32_t rowBase = static_cast<int32_t>(r) * static_cast<int32_t>(aBundle);
        uint32_t remaining = aBundle;
        for (uint16_t j = 0; j < repPerRow; ++j) {
            int32_t segOff = static_cast<int32_t>(j) * static_cast<int32_t>(kRepF32);
            int32_t off = rowBase + segOff;
            mask = AscendC::Reg::UpdateMask<float>(remaining);
            if constexpr (Bn3dNeedCast<DType>()) {
                AscendC::Reg::RegTensor<DType> bReg;
                AscendC::Reg::LoadAlign<DType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bReg, xUb + off);
                AscendC::Reg::Cast<float, DType, kCastTraitToFp32>(xReg, bReg, mask);
            } else {
                AscendC::Reg::LoadAlign(xReg, reinterpret_cast<__ubuf__ float*>(xUb) + off);
            }
            AscendC::Reg::LoadAlign(mReg, meanBc + off);
            AscendC::Reg::LoadAlign(invReg, invStdPerA + segOff);
            AscendC::Reg::Sub(xReg, xReg, mReg, mask);
            AscendC::Reg::Mul(xReg, xReg, invReg, mask);
            AscendC::Reg::StoreAlign(xNormOut + off, xReg, mask);
        }
    }
}

// p_scale VFCHAIN_000: g_f32=cast(grads); scale_mul = g_f32 * x_norm -> preOut (flat).
template <typename DType>
__simd_vf__ inline void Bn3dScaleMulVfImpl(__ubuf__ DType* gradsUb, __ubuf__ float* xNorm, __ubuf__ float* preOut,
                                           uint32_t totalElems, uint16_t repeatTime)
{
    AscendC::Reg::RegTensor<float> gReg, xnReg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        if constexpr (Bn3dNeedCast<DType>()) {
            AscendC::Reg::RegTensor<DType> bReg;
            AscendC::Reg::LoadAlign<DType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(bReg, gradsUb + off);
            AscendC::Reg::Cast<float, DType, kCastTraitToFp32>(gReg, bReg, mask);
        } else {
            AscendC::Reg::LoadAlign(gReg, reinterpret_cast<__ubuf__ float*>(gradsUb) + off);
        }
        AscendC::Reg::LoadAlign(xnReg, xNorm + off);
        AscendC::Reg::Mul(gReg, gReg, xnReg, mask);
        AscendC::Reg::StoreAlign(preOut + off, gReg, mask);
    }
}

// tail-R ExtensionPad clear (per A-bundle row, [extStart, paddedWidth)).
__simd_vf__ inline void Bn3dClearChunkExtTailRVfImpl(__ubuf__ float* base, uint32_t extStart, uint32_t aStride,
                                                     uint32_t extLanes, uint16_t aU16, uint16_t repPerA)
{
    AscendC::Reg::RegTensor<float> idReg;
    AscendC::Reg::Duplicate(idReg, kPadClearValue);
    for (uint16_t a = 0; a < aU16; ++a) {
        int32_t aOff = static_cast<int32_t>(a) * static_cast<int32_t>(aStride);
        uint32_t remaining = extLanes;
        for (uint16_t r = 0; r < repPerA; ++r) {
            int32_t off = aOff + static_cast<int32_t>(extStart) +
                          static_cast<int32_t>(r) * static_cast<int32_t>(kRepF32);
            auto mask = AscendC::Reg::UpdateMask<float>(remaining);
            AscendC::Reg::StoreAlign(base + off, idReg, mask);
        }
    }
}

// tail-A ExtensionPad clear (single contiguous stale region).
__simd_vf__ inline void Bn3dClearChunkExtTailAVfImpl(__ubuf__ float* base, uint32_t startElem, uint32_t totalClear,
                                                     uint16_t repCount)
{
    AscendC::Reg::RegTensor<float> idReg;
    AscendC::Reg::Duplicate(idReg, kPadClearValue);
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalClear;
    for (uint16_t i = 0; i < repCount; ++i) {
        int32_t off = static_cast<int32_t>(startElem) + static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::StoreAlign(base + off, idReg, mask);
    }
}

// BurstPad clear (tail-R burst-tail block pad window per row).
__simd_vf__ inline void Bn3dClearInnerBurstTailPadVfImpl(__ubuf__ float* base, uint16_t rowCntU16, int32_t rowStrideI,
                                                         int32_t windowOff, uint32_t padEnd,
                                                         uint32_t partialStartInBlock)
{
    AscendC::Reg::RegTensor<float> idReg;
    AscendC::Reg::Duplicate(idReg, kPadClearValue);
    uint32_t cntEnd = padEnd;
    uint32_t cntStart = partialStartInBlock;
    auto maskEnd = AscendC::Reg::UpdateMask<float>(cntEnd);
    auto maskStart = AscendC::Reg::UpdateMask<float>(cntStart);
    auto allMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg notStart, padMask;
    AscendC::Reg::Not(notStart, maskStart, allMask);
    AscendC::Reg::And(padMask, maskEnd, notStart, allMask);
    for (uint16_t row = 0; row < rowCntU16; ++row) {
        int32_t rowOff = static_cast<int32_t>(row) * rowStrideI;
        AscendC::Reg::StoreAlign(base + rowOff + windowOff, idReg, padMask);
    }
}

// Phase A tail merge: mainBuf += tailBuf (element-wise).
__simd_vf__ inline void Bn3dMergeTmpBufVfImpl(__ubuf__ float* mainBuf, __ubuf__ float* tailBuf, uint32_t totalElems,
                                              uint16_t repeatTime)
{
    AscendC::Reg::RegTensor<float> aReg, bReg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(aReg, mainBuf + off);
        AscendC::Reg::LoadAlign(bReg, tailBuf + off);
        AscendC::Reg::Add(aReg, aReg, bReg, mask);
        AscendC::Reg::StoreAlign(mainBuf + off, aReg, mask);
    }
}

// Binary-cache-tree absorb: cacheBuf[levelOff] += sum(cacheBuf[0..cacheLevelCnt-1]) then overwrite.
__simd_vf__ inline void Bn3dDoCachingVfImpl(__ubuf__ float* cacheBuf, uint32_t laneN, uint32_t levelStride,
                                            int32_t levelOff, uint16_t repeatTime, uint16_t cacheLevelCnt)
{
    AscendC::Reg::RegTensor<float> aReg, bReg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = laneN;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(kRepF32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(aReg, cacheBuf + levelOff + off);
        for (uint16_t j = 0; j < cacheLevelCnt; ++j) {
            int32_t lowerOff = static_cast<int32_t>(j) * static_cast<int32_t>(levelStride) + off;
            AscendC::Reg::LoadAlign(bReg, cacheBuf + lowerOff);
            AscendC::Reg::Add(aReg, aReg, bReg, mask);
        }
        AscendC::Reg::StoreAlign(cacheBuf + levelOff + off, aReg, mask);
    }
}

// empty branch: duplicate fixed value (0.0f) into output buffer.
template <typename DType>
__simd_vf__ inline void Bn3dDuplicateVfImpl(__ubuf__ DType* outPtr, DType value, uint32_t totalElems,
                                            uint16_t repeatTime)
{
    constexpr uint32_t repPerVf = kVlBytes / sizeof(DType);
    AscendC::Reg::RegTensor<DType> dReg;
    AscendC::Reg::Duplicate(dReg, value);
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(repPerVf);
        mask = AscendC::Reg::UpdateMask<DType>(remaining);
        AscendC::Reg::StoreAlign(outPtr + off, dReg, mask);
    }
}

// GM->UB axis mapping descriptor.
struct Bn3dUBAxisDesc {
    int32_t gmIdx;
    int64_t actualNum;
    int64_t paddedNum;
    int64_t gmStride;
};

// ===========================================================================
// BN3DTrainingUpdateGradBaseKernel<DType> — base template (tilingKey 0).
// ===========================================================================
template <typename DType>
class BN3DTrainingUpdateGradBaseKernel {
public:
    using DT = DType;
    __aicore__ inline BN3DTrainingUpdateGradBaseKernel() {}

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR x, GM_ADDR batchMean, GM_ADDR batchVariance, GM_ADDR diffScale,
                                GM_ADDR diffOffset, const BN3DTrainingUpdateGradTilingData* td, TPipe* pipe)
    {
        td_ = td;
        isTailR_ = (td_->axisNum % kAxisInterval == 0);
        rSplitChunkCnt_ = Bn3dCeilDiv(td_->axisShape[td_->rSplitIdx], td_->rUbFactor);
        bisectionPos_ = Bn3dFindNearestPower2(td_->rLoopCntTotal);
        bisectionTail_ = td_->rLoopCntTotal - bisectionPos_;
        cacheCount_ = Bn3dCalLog2(bisectionPos_) + 1;

        int64_t acc = 1;
        for (int32_t i = td_->axisNum - 1; i >= 0; --i) {
            if (i % kAxisInterval == 0) {
                outStride_[i] = acc;
                acc *= td_->axisShape[i];
            }
        }

        gradsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(grads));
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(x));
        batchMeanGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(batchMean));
        batchVarianceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(batchVariance));
        diffScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(diffScale));
        diffOffsetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(diffOffset));

        pipe_ = pipe;
        pipe_->InitBuffer(meanBcBuf_, td_->preBufSize);           // SLOT_000
        pipe_->InitBuffer(varBuf_, td_->preBufSize);              // SLOT_004
        pipe_->InitBuffer(preInBuf_, td_->preBufSize);            // SLOT_005
        pipe_->InitBuffer(invStdBuf_, td_->preBufSize);           // SLOT_006
        pipe_->InitBuffer(xNormBuf_, td_->preBufSize);            // SLOT_007
        pipe_->InitBuffer(preReduceResult_, td_->preBufSize);     // SLOT_002
        pipe_->InitBuffer(preReduceResultTail_, td_->preBufSize); // SLOT_003
        pipe_->InitBuffer(cacheBuf_, td_->cacheBufUbSize);        // SLOT_001

        AscendC::NdDmaDci(); // NDDMA cache prefetch (no write-after-read dep at Init)
    }

    __aicore__ inline void Process()
    {
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        if (blockIdx >= static_cast<int64_t>(td_->usedCoreNum)) {
            return;
        }

        int64_t aLoopStart = 0, aLoopEnd = 0;
        UnravelBlockLoop(aLoopStart, aLoopEnd);

        const int64_t aSplitAxisSize = td_->axisShape[td_->aSplitIdx];
        const int64_t aSplitStride = td_->axisStride[td_->aSplitIdx];
        const int64_t aSplitOutStr = outStride_[td_->aSplitIdx];

        mutexId_ = AscendC::AllocMutexID();

        for (int32_t processIdx = 0; processIdx < 2; ++processIdx) { // p_offset(0) -> p_scale(1)
            for (int64_t aLoopIdx = aLoopStart; aLoopIdx < aLoopEnd; ++aLoopIdx) {
                int64_t aIdx[MAX_PATTERN_RANK] = {0};
                int64_t aSplitChunkIdx = 0;
                UnravelALoop(aLoopIdx, aIdx, aSplitChunkIdx);

                int64_t chunkGmOff = 0, chunkOutOff = 0;
                for (int32_t k = td_->aSplitIdx - kAxisInterval; k >= 0; k -= kAxisInterval) {
                    chunkGmOff += aIdx[k] * td_->axisStride[k];
                    chunkOutOff += aIdx[k] * outStride_[k];
                }
                const int64_t aChunkStart = aSplitChunkIdx * td_->aUbFactor;
                const int64_t aEnd = aChunkStart + td_->aUbFactor;
                const int64_t aLen = (aEnd > aSplitAxisSize) ? (aSplitAxisSize - aChunkStart) : td_->aUbFactor;
                chunkGmOff += aChunkStart * aSplitStride;
                chunkOutOff += aChunkStart * aSplitOutStr;

                DoOneAChunk(processIdx, chunkGmOff, aLen, aIdx, aSplitChunkIdx);

                AscendC::Mutex::Lock<PIPE_MTE3>(mutexId_);
                CopyOut(processIdx, chunkOutOff, aLen);
                AscendC::Mutex::Unlock<PIPE_MTE3>(mutexId_);
            }
        }

        AscendC::ReleaseMutexID(mutexId_);
    }

private:
    __aicore__ inline int32_t LastAAxis() const
    {
        for (int32_t i = td_->axisNum - 1; i >= 0; --i) {
            if (i % kAxisInterval == 0)
                return i;
        }
        return 0;
    }
    __aicore__ inline int32_t LastRAxis() const
    {
        for (int32_t i = td_->axisNum - 1; i >= 0; --i) {
            if (i % kAxisInterval == 1)
                return i;
        }
        return 1;
    }
    // innerAProd = product of A axes strictly inside aSplitIdx (real, not padded).
    __aicore__ inline int64_t InnerAProd() const
    {
        int64_t p = 1;
        for (int32_t k = td_->aSplitIdx + kAxisInterval; k <= LastAAxis(); k += kAxisInterval)
            p *= td_->axisShape[k];
        return p;
    }

    __aicore__ inline void UnravelBlockLoop(int64_t& aLoopStart, int64_t& aLoopEnd)
    {
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        if (blockIdx < static_cast<int64_t>(td_->aBigCoreCnt)) {
            aLoopStart = blockIdx * td_->aBigCoreLoopCnt;
            aLoopEnd = aLoopStart + td_->aBigCoreLoopCnt;
        } else {
            aLoopStart = static_cast<int64_t>(td_->aBigCoreCnt) * td_->aBigCoreLoopCnt +
                         (blockIdx - static_cast<int64_t>(td_->aBigCoreCnt)) * td_->aSmallCoreLoopCnt;
            aLoopEnd = aLoopStart + td_->aSmallCoreLoopCnt;
        }
    }
    __aicore__ inline void UnravelALoop(int64_t aLoopIdx, int64_t aIdx[], int64_t& aSplitChunkIdx)
    {
        int64_t rem = aLoopIdx;
        aSplitChunkIdx = rem % td_->aSplitChunkCnt;
        rem /= td_->aSplitChunkCnt;
        for (int32_t k = td_->aSplitIdx - kAxisInterval; k >= 0; k -= kAxisInterval) {
            aIdx[k] = rem % td_->axisShape[k];
            rem /= td_->axisShape[k];
        }
    }
    __aicore__ inline int64_t UnravelRLoop(int64_t rIdx, int64_t rOuterIdx[], int64_t& rChunkIdx, int64_t& rLen)
    {
        rChunkIdx = rIdx % rSplitChunkCnt_;
        int64_t rem = rIdx / rSplitChunkCnt_;
        int64_t gmOff = 0;
        for (int32_t k = td_->rSplitIdx - kAxisInterval; k >= 1; k -= kAxisInterval) {
            rOuterIdx[k] = rem % td_->axisShape[k];
            rem /= td_->axisShape[k];
            gmOff += rOuterIdx[k] * td_->axisStride[k];
        }
        const int64_t rAxisSize = td_->axisShape[td_->rSplitIdx];
        const int64_t start = rChunkIdx * td_->rUbFactor;
        rLen = (start + td_->rUbFactor > rAxisSize) ? (rAxisSize - start) : td_->rUbFactor;
        return gmOff + start * td_->axisStride[td_->rSplitIdx];
    }

    // Pure-A dense offset (channel index) for mean/variance (only A axes, no R).
    __aicore__ inline int64_t CalcStatOffset(const int64_t aIdx[], int64_t aSplitChunkIdx)
    {
        int64_t off = 0, ostride = 1;
        for (int32_t k = LastAAxis(); k >= 0; k -= kAxisInterval) {
            if (k == td_->aSplitIdx)
                off += aSplitChunkIdx * td_->aUbFactor * ostride;
            else
                off += aIdx[k] * ostride;
            ostride *= td_->axisShape[k];
        }
        return off;
    }

    __aicore__ inline void DoOneAChunk(int32_t processIdx, int64_t chunkGmOff, int64_t aLen, const int64_t aIdx[],
                                       int64_t aSplitChunkIdx)
    {
        // p_scale resident statistics (once per A-chunk, reused across R chunks).
        if (processIdx == 1) {
            AscendC::Mutex::Lock<PIPE_MTE2>(mutexId_);
            NddmaBroadcastMean(aIdx, aSplitChunkIdx, aLen);
            CopyInVarianceTile(aIdx, aSplitChunkIdx, aLen);
            AscendC::Mutex::Unlock<PIPE_MTE2>(mutexId_);

            AscendC::Mutex::Lock<PIPE_V>(mutexId_);
            ComputeInvStdPerA(aLen);
            AscendC::Mutex::Unlock<PIPE_V>(mutexId_);
        }

        __ubuf__ float* preRes = reinterpret_cast<__ubuf__ float*>(preReduceResult_.Get<float>().GetPhyAddr());
        __ubuf__ float* preResTail = reinterpret_cast<__ubuf__ float*>(preReduceResultTail_.Get<float>().GetPhyAddr());

        const uint32_t laneA = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
        const uint32_t levelStride = static_cast<uint32_t>(Bn3dCeilAlign(laneA, kUbBlockF32));

        for (int64_t rIdx = 0; rIdx < bisectionPos_; ++rIdx) {
            int64_t rOuterIdx[MAX_PATTERN_RANK] = {0};
            int64_t rChunkIdxMain = 0, rLenMain = 0;
            const int64_t rOffMain = UnravelRLoop(rIdx, rOuterIdx, rChunkIdxMain, rLenMain);
            ProcessOneRChunk(processIdx, chunkGmOff + rOffMain, aLen, rLenMain, preRes);

            if (rIdx < bisectionTail_) {
                int64_t rOuterIdxTail[MAX_PATTERN_RANK] = {0};
                int64_t rChunkIdxTail = 0, rLenTail = 0;
                const int64_t rOffTail = UnravelRLoop(rIdx + bisectionPos_, rOuterIdxTail, rChunkIdxTail, rLenTail);
                ProcessOneRChunk(processIdx, chunkGmOff + rOffTail, aLen, rLenTail, preResTail);

                AscendC::Mutex::Lock<PIPE_V>(mutexId_);
                MergeTmpBufVf(preRes, preResTail);
                AscendC::Mutex::Unlock<PIPE_V>(mutexId_);
            }

            const uint16_t cacheID = Bn3dGetCacheID(rIdx);
            const int32_t levelOff = static_cast<int32_t>(cacheID) * static_cast<int32_t>(levelStride);

            AscendC::Mutex::Lock<PIPE_V>(mutexId_);
            if (isTailR_) {
                uint32_t srcShape[kReduceShapeDim] = {
                    laneA, static_cast<uint32_t>(td_->rUbFactorAlign * td_->innerRProdAlign)};
                AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, /*isReuseSource=*/true>(
                    cacheBuf_.Get<float>()[levelOff], preReduceResult_.Get<float>(),
                    preReduceResultTail_.Get<uint8_t>(), srcShape, /*srcInnerPad=*/true);
            } else {
                uint32_t srcShape[kReduceShapeDim] = {static_cast<uint32_t>(td_->rUbFactorAlign * td_->innerRProdAlign),
                                                      laneA};
                AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, /*isReuseSource=*/true>(
                    cacheBuf_.Get<float>()[levelOff], preReduceResult_.Get<float>(),
                    preReduceResultTail_.Get<uint8_t>(), srcShape, /*srcInnerPad=*/true);
            }
            DoCachingVf(cacheID);
            AscendC::Mutex::Unlock<PIPE_V>(mutexId_);
        }
    }

    // One R chunk: CopyIn + PreElewise VF chain + pad clear -> preOut (in one MTE2/V segment pair).
    __aicore__ inline void ProcessOneRChunk(int32_t processIdx, int64_t baseGmOff, int64_t aLen, int64_t rLen,
                                            __ubuf__ float* preOut)
    {
        __ubuf__ DT* preIn = reinterpret_cast<__ubuf__ DT*>(preInBuf_.Get<DT>().GetPhyAddr());

        if (processIdx == 1) { // p_scale: x -> x_norm; grads -> scale_mul
            __ubuf__ float* meanBc = reinterpret_cast<__ubuf__ float*>(meanBcBuf_.Get<float>().GetPhyAddr());
            __ubuf__ float* invStd = reinterpret_cast<__ubuf__ float*>(invStdBuf_.Get<float>().GetPhyAddr());
            __ubuf__ float* xNorm = reinterpret_cast<__ubuf__ float*>(xNormBuf_.Get<float>().GetPhyAddr());

            AscendC::Mutex::Lock<PIPE_MTE2>(mutexId_);
            DoCopyInTile(xGm_, baseGmOff, aLen, rLen);
            AscendC::Mutex::Unlock<PIPE_MTE2>(mutexId_);

            AscendC::Mutex::Lock<PIPE_V>(mutexId_);
            XNormChainVf(preIn, meanBc, invStd, xNorm, aLen);
            AscendC::Mutex::Unlock<PIPE_V>(mutexId_);

            AscendC::Mutex::Lock<PIPE_MTE2>(mutexId_);
            DoCopyInTile(gradsGm_, baseGmOff, aLen, rLen);
            AscendC::Mutex::Unlock<PIPE_MTE2>(mutexId_);

            AscendC::Mutex::Lock<PIPE_V>(mutexId_);
            ScaleMulChainVf(preIn, xNorm, preOut);
            ClearChunkExtensionVf(preOut, rLen);
            if (isTailR_) {
                ClearInnerBurstTailPadVf(preOut, rLen);
            }
            AscendC::Mutex::Unlock<PIPE_V>(mutexId_);
        } else { // p_offset: grads -> cast
            AscendC::Mutex::Lock<PIPE_MTE2>(mutexId_);
            DoCopyInTile(gradsGm_, baseGmOff, aLen, rLen);
            AscendC::Mutex::Unlock<PIPE_MTE2>(mutexId_);

            AscendC::Mutex::Lock<PIPE_V>(mutexId_);
            CastGradsChainVf(preIn, preOut);
            ClearChunkExtensionVf(preOut, rLen);
            if (isTailR_) {
                ClearInnerBurstTailPadVf(preOut, rLen);
            }
            AscendC::Mutex::Unlock<PIPE_V>(mutexId_);
        }
    }

    // ── CopyIn: batch_mean NDDMA broadcast along R -> meanBcBuf_ (SLOT_000) ──
    __aicore__ inline void NddmaBroadcastMean(const int64_t aIdx[], int64_t aSplitChunkIdx, int64_t aLen)
    {
        const int64_t meanOff = CalcStatOffset(aIdx, aSplitChunkIdx);
        auto meanBc = meanBcBuf_.Get<float>();
        const uint32_t aBundle = static_cast<uint32_t>(aLen) * static_cast<uint32_t>(InnerAProd());
        const uint32_t rBundle = static_cast<uint32_t>(td_->rUbFactorAlign) *
                                 static_cast<uint32_t>(td_->innerRProdAlign);
        if (isTailR_) {
            AscendC::NdDmaLoopInfo<2> lp{/*loopSrcStride*/ {0, 1}, /*loopDstStride*/ {1, rBundle},
                                         /*loopSize*/ {rBundle, aBundle}, /*loopLpSize*/ {0, 0}, /*loopRpSize*/ {0, 0}};
            AscendC::NdDmaParams<float, 2> params{lp, 0.0f};
            AscendC::DataCopy<float, 2>(meanBc, batchMeanGm_[meanOff], params);
        } else {
            // tail-A：UB = [R 外, A 内]。dst 外层（R）行 stride 必须是 **padded** laneA
            // (=aUbFactor*innerAProdAlign)，与 x-tile 行 stride 一致；inner 只读 aBundle 个
            // 真实通道（避免 batch_mean GM 越界读），padding lane 留脏且被 A 独立归约隔离。
            const uint32_t laneAPadded = static_cast<uint32_t>(td_->aUbFactor) *
                                         static_cast<uint32_t>(td_->innerAProdAlign);
            AscendC::NdDmaLoopInfo<2> lp{/*loopSrcStride*/ {1, 0}, /*loopDstStride*/ {1, laneAPadded},
                                         /*loopSize*/ {aBundle, rBundle}, /*loopLpSize*/ {0, 0}, /*loopRpSize*/ {0, 0}};
            AscendC::NdDmaParams<float, 2> params{lp, 0.0f};
            AscendC::DataCopy<float, 2>(meanBc, batchMeanGm_[meanOff], params);
        }
    }

    // ── CopyIn: batch_variance per-A dense -> varBuf_ (SLOT_004) ──
    __aicore__ inline void CopyInVarianceTile(const int64_t aIdx[], int64_t aSplitChunkIdx, int64_t aLen)
    {
        const int64_t varOff = CalcStatOffset(aIdx, aSplitChunkIdx);
        auto varUb = varBuf_.Get<float>();
        const int64_t aValid = aLen * InnerAProd();
        DataCopyExtParams ext;
        ext.blockCount = 1;
        ext.blockLen = static_cast<uint32_t>(aValid * static_cast<int64_t>(sizeof(float)));
        ext.srcStride = 0;
        ext.dstStride = 0;
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(varUb, batchVarianceGm_[varOff], ext, padParams);
    }

    // ── PreElewise VF wrappers ──
    __aicore__ inline void ComputeInvStdPerA(int64_t aLen)
    {
        __ubuf__ float* varUb = reinterpret_cast<__ubuf__ float*>(varBuf_.Get<float>().GetPhyAddr());
        __ubuf__ float* invStd = reinterpret_cast<__ubuf__ float*>(invStdBuf_.Get<float>().GetPhyAddr());
        const uint32_t laneAValid = static_cast<uint32_t>(aLen * InnerAProd());
        const uint16_t repeatTime = static_cast<uint16_t>(Bn3dCeilDiv(laneAValid, kRepF32U16));
        asc_vf_call<Bn3dInvStdPerAVfImpl>(varUb, invStd, td_->epsilon, laneAValid, repeatTime);
    }
    __aicore__ inline void CastGradsChainVf(__ubuf__ DT* gradsUb, __ubuf__ float* preOut)
    {
        const uint32_t totalElems = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->rUbFactorAlign *
                                                          td_->innerRProdAlign);
        const uint16_t repeatTime = static_cast<uint16_t>(Bn3dCeilDiv(totalElems, kRepF32U16));
        asc_vf_call<Bn3dCastGradsVfImpl<DT>>(gradsUb, preOut, totalElems, repeatTime);
    }
    __aicore__ inline void XNormChainVf(__ubuf__ DT* xUb, __ubuf__ float* meanBc, __ubuf__ float* invStd,
                                        __ubuf__ float* xNorm, int64_t aLen)
    {
        const uint32_t laneA = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
        const uint32_t rBundle = static_cast<uint32_t>(td_->rUbFactorAlign * td_->innerRProdAlign);
        if (isTailR_) {
            const uint16_t repPerRow = static_cast<uint16_t>(Bn3dCeilDiv(rBundle, kRepF32U16));
            asc_vf_call<Bn3dXNormTailRVfImpl<DT>>(xUb, meanBc, invStd, xNorm, rBundle, static_cast<uint16_t>(laneA),
                                                  repPerRow);
        } else {
            const uint16_t repPerRow = static_cast<uint16_t>(Bn3dCeilDiv(laneA, kRepF32U16));
            asc_vf_call<Bn3dXNormTailAVfImpl<DT>>(xUb, meanBc, invStd, xNorm, laneA, static_cast<uint16_t>(rBundle),
                                                  repPerRow);
        }
    }
    __aicore__ inline void ScaleMulChainVf(__ubuf__ DT* gradsUb, __ubuf__ float* xNorm, __ubuf__ float* preOut)
    {
        const uint32_t totalElems = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->rUbFactorAlign *
                                                          td_->innerRProdAlign);
        const uint16_t repeatTime = static_cast<uint16_t>(Bn3dCeilDiv(totalElems, kRepF32U16));
        asc_vf_call<Bn3dScaleMulVfImpl<DT>>(gradsUb, xNorm, preOut, totalElems, repeatTime);
    }
    __aicore__ inline void MergeTmpBufVf(__ubuf__ float* mainBuf, __ubuf__ float* tailBuf)
    {
        const uint32_t totalElems = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->rUbFactorAlign *
                                                          td_->innerRProdAlign);
        const uint16_t repeatTime = static_cast<uint16_t>(Bn3dCeilDiv(totalElems, kRepF32U16));
        asc_vf_call<Bn3dMergeTmpBufVfImpl>(mainBuf, tailBuf, totalElems, repeatTime);
    }
    __aicore__ inline void DoCachingVf(uint16_t cacheID)
    {
        const uint32_t laneN = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
        const uint32_t levelStride = static_cast<uint32_t>(Bn3dCeilAlign(laneN, kUbBlockF32));
        const int32_t levelOff = static_cast<int32_t>(cacheID) * static_cast<int32_t>(levelStride);
        const uint16_t repeatTime = static_cast<uint16_t>(Bn3dCeilDiv(laneN, kRepF32U16));
        __ubuf__ float* cachePtr = reinterpret_cast<__ubuf__ float*>(cacheBuf_.Get<float>().GetPhyAddr());
        asc_vf_call<Bn3dDoCachingVfImpl>(cachePtr, laneN, levelStride, levelOff, repeatTime, cacheID);
    }

    // ── ExtensionPad / BurstPad clear (sum reducer, fp32) ──
    __aicore__ inline void ClearChunkExtensionVf(__ubuf__ float* base, int64_t rLen)
    {
        if (rLen >= td_->rUbFactor) {
            return;
        }
        if (isTailR_) {
            const uint32_t aBundleEntries = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
            const uint32_t innerRPA = static_cast<uint32_t>(td_->innerRProdAlign);
            const uint32_t rLenInner = static_cast<uint32_t>(rLen) * innerRPA;
            const uint32_t extStart = static_cast<uint32_t>(Bn3dCeilAlign(rLenInner, kUbBlockF32));
            const uint32_t aStride = static_cast<uint32_t>(td_->rUbFactorAlign) * innerRPA;
            if (extStart >= aStride) {
                return;
            }
            const uint32_t extLanes = aStride - extStart;
            const uint32_t repPerA = static_cast<uint32_t>(Bn3dCeilDiv(extLanes, kRepF32));
            asc_vf_call<Bn3dClearChunkExtTailRVfImpl>(base, extStart, aStride, extLanes,
                                                      static_cast<uint16_t>(aBundleEntries),
                                                      static_cast<uint16_t>(repPerA));
        } else {
            const uint32_t cellElems = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign *
                                                             td_->innerRProdAlign);
            const uint32_t startElem = static_cast<uint32_t>(rLen) * cellElems;
            const uint32_t totalClear = (static_cast<uint32_t>(td_->rUbFactor) - static_cast<uint32_t>(rLen)) *
                                        cellElems;
            const uint32_t repCount = static_cast<uint32_t>(Bn3dCeilDiv(totalClear, kRepF32));
            asc_vf_call<Bn3dClearChunkExtTailAVfImpl>(base, startElem, totalClear, static_cast<uint16_t>(repCount));
        }
    }
    __aicore__ inline void ClearInnerBurstTailPadVf(__ubuf__ float* base, int64_t rLen)
    {
        const uint32_t bsInput = kUbBlockBytes / static_cast<uint32_t>(sizeof(DT));
        const int32_t lastR = LastRAxis();
        const uint32_t validR = (td_->rSplitIdx == lastR) ? static_cast<uint32_t>(rLen) :
                                                            static_cast<uint32_t>(td_->axisShape[lastR]);
        if (validR % bsInput == 0) {
            return;
        }
        const uint32_t rowStride = (td_->rSplitIdx == lastR) ?
                                       static_cast<uint32_t>(td_->rUbFactorAlign * td_->innerRProdAlign) :
                                       static_cast<uint32_t>(
                                           Bn3dCeilAlign(td_->axisShape[lastR], static_cast<int64_t>(bsInput)));
        const uint32_t rowCnt = (td_->rSplitIdx == lastR) ?
                                    static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign) :
                                    static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->rUbFactorAlign *
                                                          td_->innerRProdAlign) /
                                        rowStride;
        const uint32_t padEndInRow = static_cast<uint32_t>(Bn3dCeilAlign(validR, static_cast<int64_t>(bsInput)));
        const uint32_t partialBlockIdx = validR / kUbBlockF32;
        const uint32_t partialStartInBlock = validR % kUbBlockF32;
        const uint32_t padEnd = padEndInRow - partialBlockIdx * kUbBlockF32;
        asc_vf_call<Bn3dClearInnerBurstTailPadVfImpl>(
            base, static_cast<uint16_t>(rowCnt), static_cast<int32_t>(rowStride),
            static_cast<int32_t>(partialBlockIdx * kUbBlockF32), padEnd, partialStartInBlock);
    }

    // ── CopyIn: grads/x DataCopyPad + Loop transpose装入 -> preInBuf_ (SLOT_005) ──
    __aicore__ inline int32_t BuildUBAxes(int64_t aLen, int64_t rLen, Bn3dUBAxisDesc out[])
    {
        int32_t k = 0;
        const int32_t lastA = LastAAxis();
        const int32_t lastR = LastRAxis();
        const int64_t bsElem = static_cast<int64_t>(kUbBlockBytes) / static_cast<int64_t>(sizeof(DT));
        if (isTailR_) {
            for (int32_t i = td_->axisNum - 1; i >= td_->rSplitIdx; --i) {
                if (i % kAxisInterval != 1) {
                    continue;
                }
                int64_t actual, padded;
                if (i == td_->rSplitIdx) {
                    actual = rLen;
                    padded = td_->rUbFactorAlign;
                } else if (i == lastR) {
                    actual = td_->axisShape[i];
                    padded = Bn3dCeilAlign(actual, bsElem);
                } else {
                    actual = td_->axisShape[i];
                    padded = actual;
                }
                out[k++] = Bn3dUBAxisDesc{i, actual, padded, td_->axisStride[i]};
            }
            for (int32_t i = td_->axisNum - 1; i >= td_->aSplitIdx; --i) {
                if (i % kAxisInterval != 0) {
                    continue;
                }
                int64_t actual, padded;
                if (i == td_->aSplitIdx) {
                    actual = aLen;
                    padded = td_->aUbFactor;
                } else {
                    actual = td_->axisShape[i];
                    padded = actual;
                }
                out[k++] = Bn3dUBAxisDesc{i, actual, padded, td_->axisStride[i]};
            }
        } else {
            for (int32_t i = td_->axisNum - 1; i >= td_->aSplitIdx; --i) {
                if (i % kAxisInterval != 0) {
                    continue;
                }
                int64_t actual, padded;
                if (i == td_->aSplitIdx) {
                    actual = aLen;
                    padded = td_->aUbFactor;
                } else if (i == lastA) {
                    actual = td_->axisShape[i];
                    padded = Bn3dCeilAlign(actual, bsElem);
                } else {
                    actual = td_->axisShape[i];
                    padded = actual;
                }
                out[k++] = Bn3dUBAxisDesc{i, actual, padded, td_->axisStride[i]};
            }
            for (int32_t i = td_->axisNum - 1; i >= td_->rSplitIdx; --i) {
                if (i % kAxisInterval != 1) {
                    continue;
                }
                int64_t actual, padded;
                if (i == td_->rSplitIdx) {
                    actual = rLen;
                    padded = td_->rUbFactorAlign;
                } else {
                    actual = td_->axisShape[i];
                    padded = actual;
                }
                out[k++] = Bn3dUBAxisDesc{i, actual, padded, td_->axisStride[i]};
            }
        }
        return k;
    }
    __aicore__ inline void DoCopyInTile(const AscendC::GlobalTensor<DT>& srcGm, int64_t baseGmOff, int64_t aLen,
                                        int64_t rLen)
    {
        Bn3dUBAxisDesc ubAxes[MAX_PATTERN_RANK];
        const int32_t axisCnt = BuildUBAxes(aLen, rLen, ubAxes);
        const int64_t dtBytes = static_cast<int64_t>(sizeof(DT));

        DataCopyExtParams ext;
        LoopModeParams lp;
        ext.blockLen = static_cast<uint32_t>(ubAxes[0].actualNum * dtBytes);
        DataCopyPadExtParams<DT> padParams{false, 0, 0, 0};
        const int64_t copyPadBytes = Bn3dCeilAlign(static_cast<int64_t>(ext.blockLen),
                                                   static_cast<int64_t>(kUbBlockBytes));
        const int64_t target0Bytes = ubAxes[0].paddedNum * dtBytes;
        ext.dstStride = (target0Bytes - copyPadBytes) / static_cast<int64_t>(kUbBlockBytes);

        if (axisCnt > 1) {
            ext.blockCount = static_cast<uint16_t>(ubAxes[1].actualNum);
            ext.srcStride = ubAxes[1].gmStride * dtBytes - static_cast<int64_t>(ext.blockLen);
        } else {
            ext.blockCount = 1;
            ext.srcStride = 0;
        }

        int64_t ubStride[MAX_PATTERN_RANK] = {0};
        ubStride[0] = dtBytes;
        for (int32_t i = 1; i < axisCnt; ++i) {
            ubStride[i] = ubStride[i - 1] * ubAxes[i - 1].paddedNum;
        }

        if (axisCnt > 2) {
            lp.loop1Size = static_cast<uint32_t>(ubAxes[2].actualNum);
            lp.loop1SrcStride = static_cast<uint64_t>(ubAxes[2].gmStride) * static_cast<uint64_t>(dtBytes);
            lp.loop1DstStride = static_cast<uint64_t>(ubStride[2]);
            lp.loop2Size = 1;
        }
        if (axisCnt > 3) {
            lp.loop2Size = static_cast<uint32_t>(ubAxes[3].actualNum);
            lp.loop2SrcStride = static_cast<uint64_t>(ubAxes[3].gmStride) * static_cast<uint64_t>(dtBytes);
            lp.loop2DstStride = static_cast<uint64_t>(ubStride[3]);
        }

        const bool useLoopMode = (axisCnt > 2);
        if (useLoopMode) {
            SetLoopModePara(lp, DataCopyMVType::OUT_TO_UB);
        }

        int64_t outerProd = 1;
        for (int32_t k = 4; k < axisCnt; ++k) {
            outerProd *= ubAxes[k].actualNum;
        }
        auto preInLocal = preInBuf_.Get<DT>();
        for (int64_t flat = 0; flat < outerProd; ++flat) {
            int64_t addGm = 0, addUbBytes = 0, cur = flat;
            for (int32_t k = 4; k < axisCnt; ++k) {
                const int64_t sz = ubAxes[k].actualNum;
                const int64_t ix = cur % sz;
                cur /= sz;
                addGm += ix * ubAxes[k].gmStride;
                addUbBytes += ix * ubStride[k];
            }
            DataCopyPad(preInLocal[addUbBytes / dtBytes], srcGm[baseGmOff + addGm], ext, padParams);
        }
        if (useLoopMode) {
            ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        }
    }

    // ── CopyOut: cacheBuf tree root (fp32 identity) -> diff_offset / diff_scale GM ──
    __aicore__ inline void CopyOut(int32_t processIdx, int64_t outerOutOff, int64_t aLen)
    {
        const uint32_t laneA = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
        const uint32_t levelStride = static_cast<uint32_t>(Bn3dCeilAlign(laneA, kUbBlockF32));
        const int32_t rootOff = static_cast<int32_t>(cacheCount_ - 1) * static_cast<int32_t>(levelStride);
        auto rootLocal = cacheBuf_.Get<float>()[rootOff];

        const int64_t innerAProd = InnerAProd();
        DataCopyExtParams outParams;
        if (isTailR_) {
            outParams.blockLen = static_cast<uint32_t>(aLen * innerAProd * static_cast<int64_t>(sizeof(float)));
            outParams.blockCount = 1;
        } else {
            const int32_t lastA = LastAAxis();
            const int64_t lastASize = td_->axisShape[lastA];
            if (td_->aSplitIdx == lastA) {
                outParams.blockLen = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(float)));
                outParams.blockCount = 1;
            } else {
                outParams.blockLen = static_cast<uint32_t>(lastASize * static_cast<int64_t>(sizeof(float)));
                outParams.blockCount = static_cast<uint16_t>(aLen * innerAProd / lastASize);
            }
        }
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        if (processIdx == 0) {
            DataCopyPad(diffOffsetGm_[outerOutOff], rootLocal, outParams);
        } else {
            DataCopyPad(diffScaleGm_[outerOutOff], rootLocal, outParams);
        }
    }

    const BN3DTrainingUpdateGradTilingData* td_ = nullptr;
    bool isTailR_ = false;
    int64_t rSplitChunkCnt_ = 0;
    int64_t bisectionPos_ = 0;
    int64_t bisectionTail_ = 0;
    int64_t cacheCount_ = 0;
    int64_t outStride_[MAX_PATTERN_RANK] = {0};

    GlobalTensor<DT> gradsGm_, xGm_;
    GlobalTensor<float> batchMeanGm_, batchVarianceGm_, diffScaleGm_, diffOffsetGm_;
    TPipe* pipe_ = nullptr;
    TBuf<QuePosition::VECCALC> meanBcBuf_;           // SLOT_000
    TBuf<QuePosition::VECCALC> varBuf_;              // SLOT_004
    TBuf<QuePosition::VECCALC> preInBuf_;            // SLOT_005
    TBuf<QuePosition::VECCALC> invStdBuf_;           // SLOT_006
    TBuf<QuePosition::VECCALC> xNormBuf_;            // SLOT_007
    TBuf<QuePosition::VECCALC> preReduceResult_;     // SLOT_002
    TBuf<QuePosition::VECCALC> preReduceResultTail_; // SLOT_003 (+ ReduceSum sharedTmp)
    TBuf<QuePosition::VECCALC> cacheBuf_;            // SLOT_001
    uint8_t mutexId_ = 0;
};

#endif // BN3D_TRAINING_UPDATE_GRAD_BASE_KERNEL_H_
