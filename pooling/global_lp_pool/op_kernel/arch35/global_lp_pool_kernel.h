/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "kernel_operator.h"
#include "global_lp_pool_tiling_data.h"
#include "global_lp_pool_tiling_key.h"

using namespace AscendC;

// ─── arch35 constants ──────────────────────────────────────────────────────
static constexpr int32_t VF_VL_F32 = 64;   // 256 / sizeof(float)
static constexpr int32_t VF_BLK_SIZE = 32; // bytes per dataBlock

// FTZ_FALSE 档：dav_3510 上 Ln/Exp/Sqrt 的默认 INTRINSIC 档会把次正规输入/输出
// 冲刷成 0（|x|^p 或 Σ 落次正规区时结果整段失真），故统一改用 FTZ_FALSE 档。
// mode 仅作编译期模板实参（Reg::Ln/Exp/Sqrt 的 NTTP），不产生运行时访问。
static constexpr AscendC::Reg::LnSpecificMode kLnFtzFalse = {AscendC::Reg::MaskMergeMode::ZEROING,
                                                             AscendC::LnAlgo::PRECISION_1ULP_FTZ_FALSE};
static constexpr AscendC::Reg::ExpSpecificMode kExpFtzFalse = {AscendC::Reg::MaskMergeMode::ZEROING,
                                                               AscendC::ExpAlgo::PRECISION_1ULP_FTZ_FALSE};
static constexpr AscendC::Reg::SqrtSpecificMode kSqrtFtzFalse = {AscendC::Reg::MaskMergeMode::ZEROING, false,
                                                                 AscendC::SqrtAlgo::PRECISION_0ULP_FTZ_FALSE};

// ─── FindNearestPower2 (using ScalarCountLeadingZero) ──────────────────────
__aicore__ static inline uint64_t FindNearestPower2(uint64_t value)
{
    if (value == 0)
        return 0; // guard: rLoopCntTotal==0 edge case
    if (value <= 2)
        return 1;
    if (value <= 4)
        return 2;
    const uint64_t num = value - 1;
    const uint64_t pow = 63 - AscendC::ScalarCountLeadingZero(num);
    return static_cast<uint64_t>(1) << pow;
}

// ─── CalLog2 ───────────────────────────────────────────────────────────────
__aicore__ static inline uint64_t CalLog2(uint64_t value)
{
    uint64_t res = 0;
    while (value > 1) {
        value >>= 1;
        res++;
    }
    return res;
}

// ─── GetCacheID (using ScalarGetCountOfValue) ──────────────────────────────
__aicore__ static inline int64_t GetCacheID(int64_t idx)
{
    return AscendC::ScalarGetCountOfValue<1>(idx ^ (idx + 1)) - 1;
}

// ═══════════════════════════════════════════════════════════════════════════
// GlobalLpPoolKernel
// ═══════════════════════════════════════════════════════════════════════════
template <typename D_T, bool isEmptyTensor, bool isTailR>
class GlobalLpPoolKernel {
public:
    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR outputY, GM_ADDR workspace, GM_ADDR tiling,
                                const GlobalLpPoolTilingData* tilingData)
    {
        tilingData_ = const_cast<GlobalLpPoolTilingData*>(tilingData);

        xGm_.SetGlobalBuffer((__gm__ D_T*)inputX);
        yGm_.SetGlobalBuffer((__gm__ D_T*)outputY);

        aUbFactor_ = tilingData->aUbFactor;
        aUbFactorAlign_ = tilingData->aUbFactorAlign;
        rUbFactor_ = tilingData->rUbFactor;
        rUbFactorAlign_ = tilingData->rUbFactorAlign;
        rLoopCntTotal_ = tilingData->rLoopCntTotal;
        aLoopCntTotal_ = tilingData->aLoopCntTotal;
        aSplitChunkCnt_ = tilingData->aSplitChunkCnt;
        usedCoreNum_ = tilingData->usedCoreNum;
        aBigCoreCnt_ = tilingData->aBigCoreCnt;
        aBigCoreLoopCnt_ = tilingData->aBigCoreLoopCnt;
        aSmallCoreLoopCnt_ = tilingData->aSmallCoreLoopCnt;
        axisShape0_ = tilingData->axisShape[0];
        axisShape1_ = tilingData->axisShape[1];
        axisShape2_ = tilingData->axisShape[2];
        aSplitAxisIdx_ = tilingData->aSplitAxisIdx;
        aSplitSize_ = tilingData->axisShape[aSplitAxisIdx_];
        // tail 类型运行期判定：按 A/R 规整的固定间距（偶位 A / 奇位 R），
        // 偶数轴 → tail-R (AR)；奇数轴 → tail-A (ARA)。
        isTailR_ = ((tilingData->axisNum % GLLP_AXIS_INTERVAL) == 0);
        p_ = tilingData->p;
        invP_ = tilingData->invP;

        int64_t rTotal = rLoopCntTotal_;
        bisectionPos_ = static_cast<int64_t>(FindNearestPower2(static_cast<uint64_t>(rTotal)));
        bisectionTail_ = rTotal - bisectionPos_;
        // treeDepth_ == cacheCount (L = log2(P) + 1).
        // The tree occupies exactly `treeDepth_` layers, indices [0, treeDepth_-1];
        // the root lives at (treeDepth_-1) × cacheLevelStride. There is NO extra
        // scratch layer — ReduceSum writes straight into its own tree layer.
        treeDepth_ = static_cast<int64_t>(CalLog2(static_cast<uint64_t>(bisectionPos_)) + 1);

        bufSizePre_ = tilingData->preReduceUbSize;
        bufSizePost_ = tilingData->postReduceUbSize;
        bufSizeTmp_ = tilingData->tmpBufUbSize;
        bufSizeCache_ = tilingData->cacheBufUbSize;
        cacheLevelStride_ = tilingData->cacheLevelStride; // 32B-aligned
        // is_fast_path reserved in TilingData for future use

        // UB buffer allocation — mirrors the reduction paradigm's 5 physical
        // buffers (P_pre=2 + P_pre_ext=1 pre-phase slots, cacheBuf, outBuf):
        //   preInQue_  DB=1        preInBuf        (pre-phase slot 0)
        //   tmpBuf_    copies=2    preReduceResult (slot 1) + preReduceResultTail (slot 2)
        //                          preReduceResultTail also doubles as the Reduce API
        //                          sharedTmpBuffer (added when ReduceSum is converted)
        //   cacheBuf_  single      16KB binary accumulation tree
        //   outQue_    single      output buffer
        // Double buffer is NOT enabled: enabling it adds one full pre-phase buffer
        // and pushes the UB budget over the platform limit.
        // All three pre-phase slots share preBufSize (sized with maxDtypeSize),
        // so no per-buffer VL padding is needed or present.
        pipe_.InitBuffer(preInQue_, 1, static_cast<uint32_t>(bufSizePre_));
        pipe_.InitBuffer(tmpBuf_, static_cast<uint32_t>(bufSizeTmp_ * 2));
        pipe_.InitBuffer(cacheBuf_, static_cast<uint32_t>(bufSizeCache_));
        pipe_.InitBuffer(outQue_, 1, static_cast<uint32_t>(bufSizePost_));
    }

    __aicore__ inline void Process()
    {
        if constexpr (isEmptyTensor) {
            // EMPTY_A: usedCoreNum==0 → no kernel launched, this path unreachable
            // EMPTY_R: R==0, A>0 → write identity (0.0f) to each output position
            if (usedCoreNum_ == 0)
                return; // EMPTY_A guard

            uint32_t blockIdx = GetBlockIdx();
            if (blockIdx >= static_cast<uint32_t>(usedCoreNum_))
                return;

            int64_t aLoopStart, aLoopEnd;
            UnravelBlockRange(blockIdx, &aLoopStart, &aLoopEnd);

            for (int64_t aIdx = aLoopStart; aIdx < aLoopEnd; aIdx++) {
                int64_t aOff = aIdx * aUbFactor_;
                int64_t aLen = (aOff + aUbFactor_ <= axisShape0_) ? aUbFactor_ : (axisShape0_ - aOff);
                // Fill output with empty_r_output_value (0.0f for ReduceSum)
                auto outBuf = outQue_.AllocTensor<D_T>();
                int32_t n = static_cast<int32_t>(aLen);
                Duplicate<D_T>(outBuf, static_cast<D_T>(0.0f), n);
                outQue_.EnQue(outBuf);
                outBuf = outQue_.DeQue<D_T>();
                CopyOut(outBuf, aOff, aLen);
                outQue_.FreeTensor(outBuf);
            }
            return;
        }

        uint32_t blockIdx = GetBlockIdx();
        if (blockIdx >= static_cast<uint32_t>(usedCoreNum_))
            return;

        int64_t aLoopStart, aLoopEnd;
        UnravelBlockRange(blockIdx, &aLoopStart, &aLoopEnd);

        for (int64_t aIdx = aLoopStart; aIdx < aLoopEnd; aIdx++) {
            int64_t aOff = aIdx * aUbFactor_;
            int64_t aLen = (aOff + aUbFactor_ <= aSplitSize_) ? aUbFactor_ : (aSplitSize_ - aOff);
            ProcessAChunk(aOff, aLen);
        }
    }

private:
    // ─── CopyIn (tail-R, K=2) ─────────────────────────────────────────────
    __aicore__ inline void DoCopyInTile(LocalTensor<D_T>& dst, int64_t aOff, int64_t aLen, int64_t rChunkIdx,
                                        int64_t rLen)
    {
        DataCopyPadExtParams<D_T> padParams(false, 0, 0, static_cast<D_T>(0));
        if (isTailR_) {
            // tail-R: UB = [A_bundle, R_bundle]，burst 尾轴是 R（最内层）
            uint32_t blockLen = static_cast<uint32_t>(rLen * sizeof(D_T));
            uint16_t blockCount = static_cast<uint16_t>(aLen);

            int64_t srcStride = axisShape1_ * static_cast<int64_t>(sizeof(D_T)) - static_cast<int64_t>(blockLen);

            int64_t paddedRowBytes = rUbFactorAlign_ * static_cast<int64_t>(sizeof(D_T));

            // dstStride: skip the 32B-aligned gap between rows on UB. The UB row width
            // is always CeilAlign(blockLen, 32B) regardless of rightPadding, so
            // rightPadding stays 0 and the pad lanes are left dirty.
            uint32_t validBytes = blockLen;
            uint32_t alignedBytes = (validBytes + VF_BLK_SIZE - 1) & ~(VF_BLK_SIZE - 1); // CeilAlign(blockLen, 32)
            int64_t dstStride = (paddedRowBytes - static_cast<int64_t>(alignedBytes)) / VF_BLK_SIZE;

            DataCopyExtParams copyParams(blockCount, blockLen, srcStride, dstStride, 0);
            // isPad=false, rightPadding=0, paddingValue=0:
            //    the pad region is dirty and carries no semantics. The kernel clears the
            //    R-direction pad explicitly via ClearRPadVf before the reduce.
            int64_t gmOff = aOff * axisShape1_ + rChunkIdx * rUbFactor_;
            DataCopyPad(dst, xGm_[gmOff], copyParams, padParams);
        } else {
            // tail-A (ARA): UB = [R_bundle, A_bundle]，burst 尾轴是 A（最内层）。
            // A 方向的 BurstPad/stale 不参与 R 方向 Reduce，无需清零；R 方向无
            // ExtensionPad（R 全驻，rLen == rUbFactor == 1）。
            uint32_t blockLen = static_cast<uint32_t>(aLen * sizeof(D_T));
            uint16_t blockCount = static_cast<uint16_t>(rLen);

            int64_t srcStride = (blockCount > 1) ?
                                    (axisShape2_ * static_cast<int64_t>(sizeof(D_T)) - static_cast<int64_t>(blockLen)) :
                                    0;

            int64_t paddedRowBytes = aUbFactorAlign_ * static_cast<int64_t>(sizeof(D_T));
            uint32_t alignedBytes = (blockLen + VF_BLK_SIZE - 1) & ~(VF_BLK_SIZE - 1);
            int64_t dstStride = (paddedRowBytes - static_cast<int64_t>(alignedBytes)) / VF_BLK_SIZE;

            DataCopyExtParams copyParams(blockCount, blockLen, srcStride, dstStride, 0);
            // R0 行 stride = A（axisShape2_），A1 stride = 1
            int64_t gmOff = aOff + rChunkIdx * rUbFactor_ * axisShape2_;
            DataCopyPad(dst, xGm_[gmOff], copyParams, padParams);
        }
    }

    // ─── CopyOut (tail-R, single burst) ───────────────────────────────────
    __aicore__ inline void CopyOut(LocalTensor<D_T>& src, int64_t aOff, int64_t aLen)
    {
        uint32_t blockLen = static_cast<uint32_t>(aLen * sizeof(D_T));
        uint16_t blockCount = 1;
        DataCopyExtParams copyParams(blockCount, blockLen, 0, 0, 0);
        DataCopyPad(yGm_[aOff], src, copyParams);
    }

    // ─── ProcessOneChunkVf ────────────────────────────────────────────────
    __aicore__ inline void ProcessOneChunkVf(int64_t aOff, int64_t aLen, int64_t rIdx, int64_t rLen, int32_t tmpBufIdx,
                                             bool needZeroTail)
    {
        auto preIn = preInQue_.AllocTensor<D_T>();
        DoCopyInTile(preIn, aOff, aLen, rIdx, rLen);
        preInQue_.EnQue(preIn);
        preIn = preInQue_.DeQue<D_T>();
        PreElewiseVf(preIn, tmpBufIdx, rLen);
        preInQue_.FreeTensor(preIn);
        if (isTailR_) {
            // tail-R only: R_bundle pad clearing (BurstPad + ExtensionPad). The VFs
            // self-gate (BurstPad only when validR is unaligned, ExtensionPad only for a
            // partial chunk). CopyIn uses isPad=false so this is the only thing that
            // makes the pad lanes zero before they are folded into the reduce.
            // tail-A 下 BurstPad/stale 都在 A 方向，不参与 R 方向 Reduce，无需清零。
            ClearRPadVf(tmpBufIdx, rLen);
        }
    }

    // ─── Process One A Chunk ──────────────────────────────────────────────
    __aicore__ inline void ProcessAChunk(int64_t aOff, int64_t aLen)
    {
        ClearCacheTreeVf();
        for (int64_t rIdx = 0; rIdx < bisectionPos_; rIdx++) {
            int64_t rLenMain = (rUbFactor_ < (axisShape1_ - rIdx * rUbFactor_)) ? rUbFactor_ :
                                                                                  (axisShape1_ - rIdx * rUbFactor_);
            if (rIdx < bisectionTail_) {
                int64_t tailIdx = rIdx + bisectionPos_;
                int64_t rLenTail = (rUbFactor_ < (axisShape1_ - tailIdx * rUbFactor_)) ?
                                       rUbFactor_ :
                                       (axisShape1_ - tailIdx * rUbFactor_);
                ProcessOneChunkVf(aOff, aLen, rIdx, rLenMain, 0, true);
                ProcessOneChunkVf(aOff, aLen, tailIdx, rLenTail, 1, true);
                MergeTmpBufVf();
                ReduceSumToCache(rIdx);
            } else {
                ProcessOneChunkVf(aOff, aLen, rIdx, rLenMain, 0, false);
                ReduceSumToCache(rIdx);
            }
            DoCaching(rIdx);
        }
        int64_t rootIdx = treeDepth_ - 1;
        auto outBuf = outQue_.AllocTensor<D_T>();
        PostElewiseVf(rootIdx, aLen, outBuf);
        outQue_.EnQue(outBuf);
        outBuf = outQue_.DeQue<D_T>();
        CopyOut(outBuf, aOff, aLen);
        outQue_.FreeTensor(outBuf);
    }

    // ══════════════════════════════════════════════════════════════════════
    // PreElewise (Cast + Power) — extracted sub-functions to reduce size
    // ══════════════════════════════════════════════════════════════════════

    // ─── AbsLoopVf: shared by p==1.0 branch and general-path Abs sub-step ──
    __aicore__ inline void AbsLoopVf(__ubuf__ float* pwrAddr, uint16_t pwrLoopNum, uint32_t pwrRemaining)
    {
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> vx;
            Reg::MaskReg mask;
            uint32_t remain = pwrRemaining;
            for (uint16_t i = 0; i < pwrLoopNum; i++) {
                mask = Reg::UpdateMask<float>(remain);
                Reg::LoadAlign<float>(vx, pwrAddr + static_cast<int32_t>(i) * VF_VL_F32);
                Reg::Abs(vx, vx, mask);
                Reg::StoreAlign<float>(pwrAddr + static_cast<int32_t>(i) * VF_VL_F32, vx, mask);
            }
        }
    }

    // ─── CastToFloatVf: D_T→float cast (LoadAlign+Cast+StoreAlign) ──
    template <bool kNeedCast>
    __aicore__ inline void CastToFloatVf(LocalTensor<float>& workBuf, LocalTensor<D_T>& preIn, int64_t totalF32,
                                         int32_t totalCount)
    {
        if constexpr (kNeedCast) {
            uintptr_t srcBase = reinterpret_cast<uintptr_t>(preIn.GetPhyAddr());
            __ubuf__ float* dstAddr = (__ubuf__ float*)workBuf.GetPhyAddr();

            uint16_t vfLoopNum = static_cast<uint16_t>(
                (static_cast<uint32_t>(totalF32) + static_cast<uint32_t>(VF_VL_F32) - 1) /
                static_cast<uint32_t>(VF_VL_F32));
            uint32_t remaining = static_cast<uint32_t>(totalF32);

            static constexpr AscendC::Reg::CastTrait kCtWiden = {
                AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
                AscendC::RoundMode::CAST_NONE};

            __VEC_SCOPE__
            {
                Reg::RegTensor<float> vx;
                Reg::RegTensor<D_T> vh;
                Reg::MaskReg mask;
                uint32_t remain = remaining;

                for (uint16_t i = 0; i < vfLoopNum; i++) {
                    mask = Reg::UpdateMask<float>(remain);
                    Reg::LoadAlign<D_T, Reg::LoadDist::DIST_UNPACK_B16>(
                        vh, (__ubuf__ D_T*)(srcBase +
                                            static_cast<int32_t>(i) * VF_VL_F32 * static_cast<int32_t>(sizeof(D_T))));
                    Reg::Cast<float, D_T, kCtWiden>(vx, vh, mask);
                    Reg::StoreAlign<float>(dstAddr + static_cast<int32_t>(i) * VF_VL_F32, vx, mask);
                }
            }
        } else {
            DataCopy(workBuf, preIn, totalCount);
        }
    }

    // ─── GeneralPowerVf: |x|^p for p∉{1,2} — single fused register chain ──
    __aicore__ inline void GeneralPowerVf(__ubuf__ float* pwrAddr, uint16_t pwrLoopNum, uint32_t pwrRemaining,
                                          float pVal)
    {
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> vx;
            Reg::RegTensor<float> vp;
            Reg::MaskReg mask;
            uint32_t remain = pwrRemaining;
            for (uint16_t i = 0; i < pwrLoopNum; i++) {
                mask = Reg::UpdateMask<float>(remain);
                Reg::LoadAlign<float>(vx, pwrAddr + static_cast<int32_t>(i) * VF_VL_F32);
                Reg::Abs(vx, vx, mask);
                Reg::Ln<float, &kLnFtzFalse>(vx, vx, mask);
                Reg::Muls(vx, vx, static_cast<float>(pVal), mask);
                // Exp 必须非原地（dst != src）：FTZ_FALSE 档内部先 vexp(dst,src) 再按 src 重算
                // 次正规 lane，原地调用会读到已被冲刷的 src，得 exp(0)^2 = 1.0。
                Reg::Exp<float, &kExpFtzFalse>(vp, vx, mask);
                Reg::StoreAlign<float>(pwrAddr + static_cast<int32_t>(i) * VF_VL_F32, vp, mask);
            }
        }
    }

    // ─── PowerVf: |x|^p dispatcher (p=1 → Abs, p=2 → Mul, else → GeneralPowerVf) ──
    __aicore__ inline void PowerVf(__ubuf__ float* pwrAddr, uint16_t pwrLoopNum, uint32_t pwrRemaining, float pVal)
    {
        if (pVal == 1.0f) {
            AbsLoopVf(pwrAddr, pwrLoopNum, pwrRemaining);
        } else if (pVal == 2.0f) {
            __VEC_SCOPE__
            {
                Reg::RegTensor<float> vx;
                Reg::MaskReg mask;
                uint32_t remain = pwrRemaining;
                for (uint16_t i = 0; i < pwrLoopNum; i++) {
                    mask = Reg::UpdateMask<float>(remain);
                    Reg::LoadAlign<float>(vx, pwrAddr + static_cast<int32_t>(i) * VF_VL_F32);
                    Reg::Mul(vx, vx, vx, mask);
                    Reg::StoreAlign<float>(pwrAddr + static_cast<int32_t>(i) * VF_VL_F32, vx, mask);
                }
            }
        } else {
            GeneralPowerVf(pwrAddr, pwrLoopNum, pwrRemaining, pVal);
        }
    }

    __aicore__ inline void PreElewiseVf(LocalTensor<D_T>& preIn, int32_t tmpBufIdx, int64_t rLen)
    {
        constexpr bool kNeedCast = !(std::is_same_v<D_T, float>);
        int64_t totalF32 = aUbFactorAlign_ * rUbFactorAlign_;
        float pVal = p_;
        int32_t totalCount = static_cast<int32_t>(totalF32);

        int32_t slotOff = tmpBufIdx * static_cast<int32_t>(bufSizeTmp_ / sizeof(float));
        LocalTensor<float> floatBuf = tmpBuf_.Get<float>();
        LocalTensor<float> workBuf = floatBuf[slotOff];

        // No PipeBarrier between the cast VF and the power VF: on Ascend950
        // consecutive VF calls are hardware-ordered.
        CastToFloatVf<kNeedCast>(workBuf, preIn, totalF32, totalCount);

        __ubuf__ float* pwrAddr = (__ubuf__ float*)workBuf.GetPhyAddr();
        uint16_t pwrLoopNum = static_cast<uint16_t>(
            (static_cast<uint32_t>(totalCount) + static_cast<uint32_t>(VF_VL_F32) - 1) /
            static_cast<uint32_t>(VF_VL_F32));
        PowerVf(pwrAddr, pwrLoopNum, static_cast<uint32_t>(totalCount), pVal);
    }
    // ─── PostCastAndStoreVf (eliminates 3× duplicate cast+store in PostElewiseVf) ──
    template <typename DT, bool kNeedCast>
    __aicore__ inline void PostCastAndStoreVf(AscendC::Reg::RegTensor<float>& f32Reg, AscendC::Reg::MaskReg mask,
                                              __ubuf__ DT* outPtr, int32_t off)
    {
        if constexpr (kNeedCast) {
            static constexpr AscendC::Reg::CastTrait kCtNarrow = {
                AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
                AscendC::RoundMode::CAST_RINT};
            AscendC::Reg::RegTensor<DT> b16Reg;
            AscendC::Reg::Cast<DT, float, kCtNarrow>(b16Reg, f32Reg, mask);
            AscendC::Reg::StoreAlign<DT, AscendC::Reg::StoreDist::DIST_PACK_B32>(outPtr + off, b16Reg, mask);
        } else {
            AscendC::Reg::StoreAlign<float>(outPtr + off, f32Reg, mask);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // PostElewise (Inverse Power + Cast back → outQue_ tensor)
    // Runtime if-else (invPVal == 0.5f etc.) must be OUTSIDE __VEC_SCOPE__
    // (VF constraint: no runtime if/else inside VF scope → bisheng segfault).
    // ══════════════════════════════════════════════════════════════════════
    template <bool kNeedCast>
    __aicore__ inline void PostElewiseVfBody(__ubuf__ float* rootPtr, __ubuf__ D_T* outPtr, uint32_t laneN,
                                             float invPVal, uint16_t repeatTime)
    {
        if (invPVal == 1.0f) {
            __VEC_SCOPE__
            {
                AscendC::Reg::RegTensor<float> f32Reg;
                AscendC::Reg::MaskReg mask;
                uint32_t remaining = laneN;
                for (uint16_t i = 0; i < repeatTime; ++i) {
                    int32_t off = static_cast<int32_t>(i) * VF_VL_F32;
                    mask = AscendC::Reg::UpdateMask<float>(remaining);
                    AscendC::Reg::LoadAlign(f32Reg, rootPtr + off);
                    PostCastAndStoreVf<D_T, kNeedCast>(f32Reg, mask, outPtr, off);
                }
            }
        } else if (invPVal == 0.5f) {
            __VEC_SCOPE__
            {
                AscendC::Reg::RegTensor<float> f32Reg;
                AscendC::Reg::MaskReg mask;
                uint32_t remaining = laneN;
                for (uint16_t i = 0; i < repeatTime; ++i) {
                    int32_t off = static_cast<int32_t>(i) * VF_VL_F32;
                    mask = AscendC::Reg::UpdateMask<float>(remaining);
                    AscendC::Reg::LoadAlign(f32Reg, rootPtr + off);
                    AscendC::Reg::Sqrt<float, &kSqrtFtzFalse>(f32Reg, f32Reg, mask);
                    PostCastAndStoreVf<D_T, kNeedCast>(f32Reg, mask, outPtr, off);
                }
            }
        } else {
            __VEC_SCOPE__
            {
                AscendC::Reg::RegTensor<float> f32Reg;
                AscendC::Reg::RegTensor<float> f32Exp;
                AscendC::Reg::MaskReg mask;
                uint32_t remaining = laneN;
                for (uint16_t i = 0; i < repeatTime; ++i) {
                    int32_t off = static_cast<int32_t>(i) * VF_VL_F32;
                    mask = AscendC::Reg::UpdateMask<float>(remaining);
                    AscendC::Reg::LoadAlign(f32Reg, rootPtr + off);
                    // (Σ)^(1/p) = exp(ln(Σ)/p)，同为 Reg:: 寄存器级链。
                    AscendC::Reg::Ln<float, &kLnFtzFalse>(f32Reg, f32Reg, mask);
                    AscendC::Reg::Muls(f32Reg, f32Reg, static_cast<float>(invPVal), mask);
                    // 同 Pre：FTZ_FALSE 档 Exp 必须非原地（dst != src）。
                    AscendC::Reg::Exp<float, &kExpFtzFalse>(f32Exp, f32Reg, mask);
                    PostCastAndStoreVf<D_T, kNeedCast>(f32Exp, mask, outPtr, off);
                }
            }
        }
    }

    __aicore__ inline void PostElewiseVf(int64_t rootIdx, int64_t aLen, LocalTensor<D_T>& outBuf)
    {
        constexpr bool kNeedCast = !(std::is_same_v<D_T, float>);
        __ubuf__ float* rootPtr = reinterpret_cast<__ubuf__ float*>(cacheBuf_.Get<float>().GetPhyAddr()) +
                                  static_cast<int32_t>(rootIdx * cacheLevelStride_);
        __ubuf__ D_T* outPtr = reinterpret_cast<__ubuf__ D_T*>(outBuf.GetPhyAddr());
        uint32_t laneN = static_cast<uint32_t>(aUbFactorAlign_);
        uint16_t repeatTime = static_cast<uint16_t>((laneN + static_cast<uint32_t>(VF_VL_F32) - 1) /
                                                    static_cast<uint32_t>(VF_VL_F32));
        PostElewiseVfBody<kNeedCast>(rootPtr, outPtr, laneN, invP_, repeatTime);
    }
    // ══════════════════════════════════════════════════════════════════════
    // MergeTmpBuf VF  (element-wise vadd of tmpBuf[0] += tmpBuf[1])
    // ══════════════════════════════════════════════════════════════════════
    __aicore__ inline void MergeTmpBufVf()
    {
        int64_t totalF32 = aUbFactorAlign_ * rUbFactorAlign_;

        __ubuf__ float* buf0 = (__ubuf__ float*)tmpBuf_.Get<float>().GetPhyAddr();
        __ubuf__ float* buf1 = buf0 + (bufSizeTmp_ / sizeof(float));

        uint16_t vfLoopNum = static_cast<uint16_t>(
            (static_cast<uint32_t>(totalF32) + static_cast<uint32_t>(VF_VL_F32) - 1) /
            static_cast<uint32_t>(VF_VL_F32));
        uint32_t remaining = static_cast<uint32_t>(totalF32);

        __VEC_SCOPE__
        {
            Reg::RegTensor<float> va, vb, vr;
            Reg::MaskReg mask;
            uint32_t remain = remaining;

            for (uint16_t i = 0; i < vfLoopNum; i++) {
                mask = Reg::UpdateMask<float>(remain);
                Reg::LoadAlign<float>(va, buf0 + static_cast<int32_t>(i) * VF_VL_F32);
                Reg::LoadAlign<float>(vb, buf1 + static_cast<int32_t>(i) * VF_VL_F32);
                Reg::Add<float, Reg::MaskMergeMode::ZEROING>(vr, va, vb, mask);
                Reg::StoreAlign<float>(buf0 + static_cast<int32_t>(i) * VF_VL_F32, vr, mask);
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // ReduceSum → cacheBuf[levelOff]   (Level 2 API, outside VF scope)
    //
    //   dst            = cacheBuf[levelOff], levelOff = cacheID × levelStride
    //                    (the tree slot IS the result slot — no scratch layer,
    //                     no bounce buffer)
    //   src            = preReduceResult      (tmpBuf slot 0)
    //   sharedTmpBuffer= preReduceResultTail  (tmpBuf slot 1; idle after Phase A
    //                    pairing, so reusing it costs no extra UB)
    //   srcInnerPad    = true                 (always, tail-R and tail-A alike)
    //   isReuseSource  = true                 (src may be clobbered; we re-read it
    //                    only via DoCaching, which reads the cacheBuf slot, not src)
    // Hard constraints:
    //    src/dst must not overlap; src base must be 32B aligned;
    //    sharedTmpBuffer must not overlap src or dst.
    // Using the 3-arg overload instead would make the API grab "whatever UB is
    // left after all InitBuffers" via PopStackBuffer — an unbudgeted region.
    // ══════════════════════════════════════════════════════════════════════
    __aicore__ inline void ReduceSumToCache(int64_t rIdx)
    {
        const int64_t cacheID = GetCacheID(rIdx);
        const int64_t levelOff = cacheID * cacheLevelStride_;

        LocalTensor<float> cacheFloat = cacheBuf_.Get<float>();
        LocalTensor<float> floatBuf = tmpBuf_.Get<float>();
        LocalTensor<float> preReduceResult = floatBuf;
        LocalTensor<uint8_t> sharedTmp = floatBuf[(bufSizeTmp_ / static_cast<int64_t>(sizeof(float)))]
                                             .template ReinterpretCast<uint8_t>();

        if (isTailR_) {
            // UB layout is [A_bundle, R_bundle] from DoCopyInTile → shape {A, R}
            // Pattern::Reduce::AR: first dim (A) kept, second dim (R) reduced
            uint32_t shape[] = {static_cast<uint32_t>(aUbFactorAlign_), static_cast<uint32_t>(rUbFactorAlign_)};
            ReduceSum<float, Pattern::Reduce::AR, true>(cacheFloat[levelOff], preReduceResult, sharedTmp, shape, true);
        } else {
            // tail-A: UB layout is [R_bundle, A_bundle] from DoCopyInTile → shape {R, A}
            // Pattern::Reduce::RA: first dim (R) reduced, second dim (A) kept
            uint32_t shape[] = {static_cast<uint32_t>(rUbFactorAlign_), static_cast<uint32_t>(aUbFactorAlign_)};
            ReduceSum<float, Pattern::Reduce::RA, true>(cacheFloat[levelOff], preReduceResult, sharedTmp, shape, true);
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // R_bundle pad clearing.
    //
    //   CopyIn uses isPad=false, so BurstPad / ExtensionPad carry dirty data.
    //   The R direction of the pad DOES get folded into the reduce, so it must be
    //   explicitly cleared (the A direction does not — A lanes are independent —
    //   and extra A-direction clearing is forbidden).
    //
    //   Two disjoint regions, two VFs:
    //     BurstPad     [validR, CeilAlign(validR, bsInput))
    //                  → ClearInnerBurstTailPadVf  (only when validR unaligned)
    //     ExtensionPad [CeilAlign(validR, bsInput), rUbFactorAlign)
    //                  → ClearChunkExtensionVf     (only for a partial r chunk)
    //
    //   Both share one hard rule: StoreAlign/LoadAlign 起点必须 block 对齐，
    //   非对齐起点会运行时错 (errcode 340).
    //   ClearInnerBurstTailPadVf achieves this by starting at FloorAlign(validR, 8)
    //   and masking the write to [validR%8, padEnd) — it never touches a valid lane
    //   and never writes into the next row.
    // ══════════════════════════════════════════════════════════════════════
    static constexpr int32_t VF_BLK_F32 = VF_BLK_SIZE / static_cast<int32_t>(sizeof(float)); // = 8

    // ── ClearInnerBurstTailPadVf: clear [validR, CeilAlign(validR, bsInput)) ──
    // Three-segment mask: maskEnd = [0,padEnd), maskStart = [0,partialStartInBlock),
    // padMask = maskEnd AND NOT maskStart = [partialStartInBlock, padEnd).
    // windowOff = FloorAlign(validR, 8) keeps the StoreAlign base 32B aligned.
    __aicore__ static inline void ClearInnerBurstTailPadVfImpl(__ubuf__ float* base, uint16_t rowCnt, int32_t rowStride,
                                                               int32_t windowOff, uint32_t padEnd,
                                                               uint32_t partialStartInBlock)
    {
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> vz;
            Reg::Duplicate<float>(vz, 0.0f);

            uint32_t cntEnd = padEnd;
            uint32_t cntStart = partialStartInBlock;
            auto maskEnd = Reg::UpdateMask<float>(cntEnd);
            auto maskStart = Reg::UpdateMask<float>(cntStart);
            auto allMask = Reg::CreateMask<float, Reg::MaskPattern::ALL>();
            Reg::MaskReg notStart, padMask;
            Reg::Not(notStart, maskStart, allMask);
            Reg::And(padMask, maskEnd, notStart, allMask);

            for (uint16_t row = 0; row < rowCnt; row++) {
                int32_t rowOff = static_cast<int32_t>(row) * rowStride;
                Reg::StoreAlign<float>(base + rowOff + windowOff, vz, padMask);
            }
        }
    }

    __aicore__ inline void ClearInnerBurstTailPadVf(__ubuf__ float* base, int64_t rLen)
    {
        const uint32_t bsInput = VF_BLK_SIZE / static_cast<uint32_t>(sizeof(D_T));
        const uint32_t validR = static_cast<uint32_t>(rLen);
        if (validR % bsInput == 0) {
            return; // burst tail already block aligned → no BurstPad region
        }
        const uint32_t rowStride = static_cast<uint32_t>(rUbFactorAlign_);
        if (validR >= rowStride) {
            return; // no padding in this row
        }
        const uint32_t padEndInRow = ((validR + bsInput - 1) / bsInput) * bsInput;
        const uint32_t partialBlockIdx = validR / static_cast<uint32_t>(VF_BLK_F32);
        const uint32_t partialStartInBlock = validR % static_cast<uint32_t>(VF_BLK_F32);
        const uint32_t padEnd = padEndInRow - partialBlockIdx * static_cast<uint32_t>(VF_BLK_F32);
        const int32_t windowOff = static_cast<int32_t>(partialBlockIdx * static_cast<uint32_t>(VF_BLK_F32));

        ClearInnerBurstTailPadVfImpl(base, static_cast<uint16_t>(aUbFactorAlign_), static_cast<int32_t>(rowStride),
                                     windowOff, padEnd, partialStartInBlock);
    }

    // ── ClearChunkExtensionVf: clear [CeilAlign(rLen,8), rUbFactorAlign) ──
    // extStart is CeilAlign'd to 32B, so the StoreAlign base is always aligned.
    __aicore__ static inline void ClearChunkExtTailRVfImpl(__ubuf__ float* base, uint32_t extStart, uint32_t aStride,
                                                           uint32_t extLanes, uint16_t aU16, uint16_t repPerA)
    {
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> vz;
            Reg::Duplicate<float>(vz, 0.0f);
            for (uint16_t a = 0; a < aU16; a++) {
                int32_t aOff = static_cast<int32_t>(a) * static_cast<int32_t>(aStride);
                uint32_t remaining = extLanes;
                for (uint16_t r = 0; r < repPerA; r++) {
                    int32_t off = aOff + static_cast<int32_t>(extStart) + static_cast<int32_t>(r) * VF_VL_F32;
                    auto mask = Reg::UpdateMask<float>(remaining);
                    Reg::StoreAlign<float>(base + off, vz, mask);
                }
            }
        }
    }

    __aicore__ inline void ClearChunkExtensionVf(__ubuf__ float* base, int64_t rLen)
    {
        if (rLen >= rUbFactor_) {
            return; // full chunk → no ExtensionPad
        }
        const uint32_t extStart = (static_cast<uint32_t>(rLen) + static_cast<uint32_t>(VF_BLK_F32) - 1) /
                                  static_cast<uint32_t>(VF_BLK_F32) * static_cast<uint32_t>(VF_BLK_F32);
        const uint32_t aStride = static_cast<uint32_t>(rUbFactorAlign_);
        if (extStart >= aStride) {
            return;
        }
        const uint32_t extLanes = aStride - extStart;
        const uint32_t repPerA = (extLanes + static_cast<uint32_t>(VF_VL_F32) - 1) / static_cast<uint32_t>(VF_VL_F32);
        ClearChunkExtTailRVfImpl(base, extStart, aStride, extLanes, static_cast<uint16_t>(aUbFactorAlign_),
                                 static_cast<uint16_t>(repPerA));
    }

    // ── R_bundle pad clear entry: BurstPad + ExtensionPad ──
    __aicore__ inline void ClearRPadVf(int32_t tmpBufIdx, int64_t rLen)
    {
        const int64_t slotOff = static_cast<int64_t>(tmpBufIdx) * (bufSizeTmp_ / static_cast<int64_t>(sizeof(float)));
        __ubuf__ float* base = (__ubuf__ float*)tmpBuf_.Get<float>().GetPhyAddr() + static_cast<int32_t>(slotOff);
        ClearChunkExtensionVf(base, rLen);    // [CeilAlign(rLen,8), rUbFactorAlign)
        ClearInnerBurstTailPadVf(base, rLen); // [validR, CeilAlign(validR,bsInput))
    }

    // ══════════════════════════════════════════════════════════════════════
    // ClearCacheTree VF  (fill cacheBuf with 0.0f)
    // ══════════════════════════════════════════════════════════════════════
    __aicore__ inline void ClearCacheTreeVf()
    {
        int64_t totalF32 = bufSizeCache_ / sizeof(float);
        __ubuf__ float* addr = (__ubuf__ float*)cacheBuf_.Get<float>().GetPhyAddr();

        uint16_t vfLoopNum = static_cast<uint16_t>(
            (static_cast<uint32_t>(totalF32) + static_cast<uint32_t>(VF_VL_F32) - 1) /
            static_cast<uint32_t>(VF_VL_F32));
        uint32_t remaining = static_cast<uint32_t>(totalF32);

        __VEC_SCOPE__
        {
            Reg::RegTensor<float> vz;
            Reg::Duplicate<float>(vz, 0.0f);
            Reg::MaskReg mask;
            uint32_t remain = remaining;

            for (uint16_t i = 0; i < vfLoopNum; i++) {
                mask = Reg::UpdateMask<float>(remain);
                Reg::StoreAlign<float>(addr + static_cast<int32_t>(i) * VF_VL_F32, vz, mask);
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // DoCaching VF  (binary-tree accumulation)
    // ══════════════════════════════════════════════════════════════════════
    __aicore__ inline void DoCaching(int64_t rIdx)
    {
        //   ReduceSum already wrote this chunk's partial sum straight into
        //   cacheBuf[cid × levelStride]. DoCaching then absorbs ALL lower layers
        //   [0..cid-1] IN ASCENDING ORDER into that same slot and overwrites it.
        //   - ascending order matters: cache[j] covers 2^j chunks, so each add
        //     pairs operands of matching magnitude (best precision). Descending
        //     order would drop low-order bits.
        //   - overwrite (not accumulate): the previous value in this slot was
        //     already absorbed by a higher layer, accumulating it would double-count.
        // So the accumulator source is cacheBase + cid*levelStride — NOT tmpBuf.
        int64_t cid = GetCacheID(rIdx);
        int64_t lvlStride = cacheLevelStride_; // 32B-aligned per-layer stride
        int64_t laneN = aUbFactorAlign_;       // valid lane count for mask

        __ubuf__ float* cacheBase = (__ubuf__ float*)cacheBuf_.Get<float>().GetPhyAddr();
        int32_t levelOff = static_cast<int32_t>(cid) * static_cast<int32_t>(lvlStride);

        uint32_t remaining = static_cast<uint32_t>(laneN);
        uint16_t repeatTime = static_cast<uint16_t>((remaining + static_cast<uint32_t>(VF_VL_F32) - 1) /
                                                    static_cast<uint32_t>(VF_VL_F32));

        uint16_t cidU16 = static_cast<uint16_t>(cid);

        __VEC_SCOPE__
        {
            Reg::RegTensor<float> va, vc;
            Reg::MaskReg mask;
            uint32_t remain = remaining;

            for (uint16_t k = 0; k < repeatTime; k++) {
                int32_t off = static_cast<int32_t>(k) * VF_VL_F32;
                mask = Reg::UpdateMask<float>(remain);

                Reg::LoadAlign<float>(va, cacheBase + levelOff + off);

                // Absorb all lower cache layers [0..cid-1] in ascending order
                // (cache[j] covers 2^j chunks, so operands pair by magnitude)
                for (uint16_t j = 0; j < cidU16; j++) {
                    int32_t jOff = static_cast<int32_t>(j) * static_cast<int32_t>(lvlStride) + off;
                    Reg::LoadAlign<float>(vc, cacheBase + jOff);
                    Reg::Add<float, Reg::MaskMergeMode::ZEROING>(va, va, vc, mask);
                }

                Reg::StoreAlign<float>(cacheBase + levelOff + off, va, mask);
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // UnravelBlockRange  (big/small core distribution)
    // ══════════════════════════════════════════════════════════════════════
    __aicore__ inline void UnravelBlockRange(uint32_t blockIdx, int64_t* start, int64_t* end)
    {
        if (static_cast<int32_t>(blockIdx) < aBigCoreCnt_) {
            *start = static_cast<int64_t>(blockIdx) * aBigCoreLoopCnt_;
            *end = *start + aBigCoreLoopCnt_;
        } else {
            int64_t bigChunks = static_cast<int64_t>(aBigCoreCnt_) * aBigCoreLoopCnt_;
            int64_t smallIdx = static_cast<int64_t>(blockIdx) - aBigCoreCnt_;
            *start = bigChunks + smallIdx * aSmallCoreLoopCnt_;
            *end = *start + aSmallCoreLoopCnt_;
        }
    }

    // ─── Member variables ─────────────────────────────────────────────────
    GlobalLpPoolTilingData* tilingData_ = nullptr;
    GlobalTensor<D_T> xGm_;
    GlobalTensor<D_T> yGm_;
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> preInQue_; // depth 1: double buffer disabled
    TBuf<QuePosition::VECCALC> tmpBuf_;
    TBuf<QuePosition::VECCALC> cacheBuf_;
    TQue<QuePosition::VECOUT, 1> outQue_;

    int64_t aUbFactor_ = 0;
    int64_t aUbFactorAlign_ = 0;
    int64_t rUbFactor_ = 0;
    int64_t rUbFactorAlign_ = 0;
    int64_t rLoopCntTotal_ = 0;
    int64_t aLoopCntTotal_ = 0;
    int64_t aSplitChunkCnt_ = 0;
    int32_t usedCoreNum_ = 0;
    int32_t aBigCoreCnt_ = 0;
    int64_t aBigCoreLoopCnt_ = 0;
    int64_t aSmallCoreLoopCnt_ = 0;
    int64_t axisShape0_ = 0;
    int64_t axisShape1_ = 0;
    int64_t axisShape2_ = 1;
    int32_t aSplitAxisIdx_ = 0;
    int64_t aSplitSize_ = 0;
    bool isTailR_ = true;
    float p_ = 2.0f;
    float invP_ = 0.5f;

    int64_t bisectionPos_ = 0;
    int64_t bisectionTail_ = 0;
    int64_t treeDepth_ = 0;

    int64_t bufSizePre_ = 0;
    int64_t bufSizePost_ = 0;
    int64_t bufSizeTmp_ = 0;
    int64_t bufSizeCache_ = 0;
    int64_t cacheLevelStride_ = 0; // = CeilAlign(aUbFactorAlign_, 8)
};

// Entry point is in the separate APT file (op_kernel/global_lp_pool_apt.cpp)
// This header only contains the kernel class definition.
