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
        p_ = tilingData->p;
        invP_ = tilingData->invP;

        int64_t rTotal = rLoopCntTotal_;
        bisectionPos_ = static_cast<int64_t>(FindNearestPower2(static_cast<uint64_t>(rTotal)));
        bisectionTail_ = rTotal - bisectionPos_;
        treeDepth_ = static_cast<int64_t>(CalLog2(static_cast<uint64_t>(bisectionPos_)) + 1);

        bufSizePre_ = tilingData->preReduceUbSize;
        bufSizePost_ = tilingData->postReduceUbSize;
        bufSizeTmp_ = tilingData->tmpBufUbSize;
        bufSizeCache_ = tilingData->cacheBufUbSize;
        cacheLevelStride_ = tilingData->cacheLevelStride; // 32B-aligned
        // is_fast_path reserved in TilingData for future use

        // UB buffer allocation
        //   preInQue_  DB=2  double-buffer for MTE/Vector pipeline
        //   tmpBuf_    copies=2  Phase A main+tail pairing
        //   cacheBuf_  single  16KB binary accumulation tree
        //   outQue_    single  output buffer
        pipe_.InitBuffer(preInQue_, 2, static_cast<uint32_t>(bufSizePre_));
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
                CopyOutFn(outBuf, aOff, aLen);
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
            int64_t aLen = (aOff + aUbFactor_ <= axisShape0_) ? aUbFactor_ : (axisShape0_ - aOff);
            ProcessAChunk(aOff, aLen);
        }
    }

private:
    // ─── CopyIn (tail-R, K=2) ─────────────────────────────────────────────
    __aicore__ inline void CopyInChunk(LocalTensor<D_T>& dst, int64_t aOff, int64_t aLen, int64_t rChunkIdx,
                                       int64_t rLen)
    {
        uint32_t blockLen = static_cast<uint32_t>(rLen * sizeof(D_T));
        uint16_t blockCount = static_cast<uint16_t>(aLen);

        int64_t srcStride = axisShape1_ * static_cast<int64_t>(sizeof(D_T)) - static_cast<int64_t>(blockLen);

        int64_t paddedRowBytes = rUbFactorAlign_ * static_cast<int64_t>(sizeof(D_T));

        // dstStride: skip 32B blocks on UB between rows (partial chunk > 0)
        //   ref: DESIGN.md §8.4, reduce-design-pitfalls #17
        uint32_t validBytes = blockLen;
        uint32_t alignedBytes = (validBytes + VF_BLK_SIZE - 1) & ~(VF_BLK_SIZE - 1); // CeilAlign(blockLen, 32)
        int64_t dstStride = (paddedRowBytes - static_cast<int64_t>(alignedBytes)) / VF_BLK_SIZE;
        uint8_t rightPad = static_cast<uint8_t>((alignedBytes - validBytes) / sizeof(D_T));

        DataCopyExtParams copyParams(blockCount, blockLen, srcStride, dstStride, 0);
        DataCopyPadExtParams<D_T> padParams(true, 0, rightPad, static_cast<D_T>(0));

        int64_t gmOff = aOff * axisShape1_ + rChunkIdx * rUbFactor_;
        DataCopyPad(dst, xGm_[gmOff], copyParams, padParams);
    }

    // ─── CopyOut (tail-R, single burst) ───────────────────────────────────
    __aicore__ inline void CopyOutFn(LocalTensor<D_T>& src, int64_t aOff, int64_t aLen)
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
        CopyInChunk(preIn, aOff, aLen, rIdx, rLen);
        preInQue_.EnQue(preIn);
        preIn = preInQue_.DeQue<D_T>();
        PreElewiseVf(preIn, tmpBufIdx, rLen);
        preInQue_.FreeTensor(preIn);
        if (needZeroTail && rLen < rUbFactorAlign_)
            ZeroRowTailVf(tmpBufIdx, rLen);
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
                ReduceSumOnTmp0();
            } else {
                ProcessOneChunkVf(aOff, aLen, rIdx, rLenMain, 0, false);
                ReduceSumOnTmp0();
            }
            DoCachingVf(rIdx);
        }
        int64_t rootIdx = treeDepth_ - 1;
        auto outBuf = outQue_.AllocTensor<D_T>();
        PostElewiseVf(rootIdx, aLen, outBuf);
        outQue_.EnQue(outBuf);
        outBuf = outQue_.DeQue<D_T>();
        CopyOutFn(outBuf, aOff, aLen);
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

    // ─── GeneralPowerVf: |x|^p for p∉{1,2} — Abs→Ln→Muls→Exp chain ──
    __aicore__ inline void GeneralPowerVf(__ubuf__ float* pwrAddr, uint16_t pwrLoopNum, uint32_t pwrRemaining,
                                          float pVal)
    {
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> vx;
            Reg::MaskReg mask;
            uint32_t remain = pwrRemaining;
            // Abs
            for (uint16_t i = 0; i < pwrLoopNum; i++) {
                mask = Reg::UpdateMask<float>(remain);
                Reg::LoadAlign<float>(vx, pwrAddr + static_cast<int32_t>(i) * VF_VL_F32);
                Reg::Abs(vx, vx, mask);
                Reg::StoreAlign<float>(pwrAddr + static_cast<int32_t>(i) * VF_VL_F32, vx, mask);
            }
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            // Ln
            remain = pwrRemaining;
            for (uint16_t i = 0; i < pwrLoopNum; i++) {
                mask = Reg::UpdateMask<float>(remain);
                Reg::LoadAlign<float>(vx, pwrAddr + static_cast<int32_t>(i) * VF_VL_F32);
                Reg::Ln(vx, vx, mask);
                Reg::StoreAlign<float>(pwrAddr + static_cast<int32_t>(i) * VF_VL_F32, vx, mask);
            }
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            // Muls
            remain = pwrRemaining;
            for (uint16_t i = 0; i < pwrLoopNum; i++) {
                mask = Reg::UpdateMask<float>(remain);
                Reg::LoadAlign<float>(vx, pwrAddr + static_cast<int32_t>(i) * VF_VL_F32);
                Reg::Muls(vx, vx, static_cast<float>(pVal), mask);
                Reg::StoreAlign<float>(pwrAddr + static_cast<int32_t>(i) * VF_VL_F32, vx, mask);
            }
            Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
            // Exp
            remain = pwrRemaining;
            for (uint16_t i = 0; i < pwrLoopNum; i++) {
                mask = Reg::UpdateMask<float>(remain);
                Reg::LoadAlign<float>(vx, pwrAddr + static_cast<int32_t>(i) * VF_VL_F32);
                Reg::Exp(vx, vx, mask);
                Reg::StoreAlign<float>(pwrAddr + static_cast<int32_t>(i) * VF_VL_F32, vx, mask);
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

        CastToFloatVf<kNeedCast>(workBuf, preIn, totalF32, totalCount);
        PipeBarrier<PIPE_V>();

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
                    AscendC::Reg::Sqrt(f32Reg, f32Reg, mask);
                    PostCastAndStoreVf<D_T, kNeedCast>(f32Reg, mask, outPtr, off);
                }
            }
        } else {
            __VEC_SCOPE__
            {
                AscendC::Reg::RegTensor<float> f32Reg;
                AscendC::Reg::MaskReg mask;
                uint32_t remaining = laneN;
                for (uint16_t i = 0; i < repeatTime; ++i) {
                    int32_t off = static_cast<int32_t>(i) * VF_VL_F32;
                    mask = AscendC::Reg::UpdateMask<float>(remaining);
                    AscendC::Reg::LoadAlign(f32Reg, rootPtr + off);
                    AscendC::Reg::Ln(f32Reg, f32Reg, mask);
                    AscendC::Reg::Muls(f32Reg, f32Reg, static_cast<float>(invPVal), mask);
                    AscendC::Reg::Exp(f32Reg, f32Reg, mask);
                    PostCastAndStoreVf<D_T, kNeedCast>(f32Reg, mask, outPtr, off);
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
    // ReduceSum (Level 2 API, outside VF scope)
    // Uses separate dst (cacheBuf slice) to avoid src==dst overlap violation.
    // ══════════════════════════════════════════════════════════════════════
    __aicore__ inline void ReduceSumOnTmp0()
    {
        auto reduceSrc = tmpBuf_.Get<float>();
        // ⚠️ MUST NOT use cacheBuf_[0] as dst — it overlaps with cache layer 0,
        //    which DoCachingVf reads for cid≥1 (reduce-bisection-algorithm §3.2).
        //    Use an offset at treeDepth_ * cacheLevelStride_ (past all valid layers).
        LocalTensor<float> cacheFloat = cacheBuf_.Get<float>();
        int64_t reduceDstOff = treeDepth_ * cacheLevelStride_;
        auto reduceDst = cacheFloat[reduceDstOff];

        // UB layout is [A_bundle, R_bundle] from CopyInChunk → shape {A, R}
        // Pattern::Reduce::AR: first dim (A) kept, second dim (R) reduced
        uint32_t shape[] = {static_cast<uint32_t>(aUbFactorAlign_), static_cast<uint32_t>(rUbFactorAlign_)};
        ReduceSum<float, Pattern::Reduce::AR, true>(reduceDst, reduceSrc, shape, true);

        // Copy result from cacheBuf offset back to tmpBuf[0]
        int64_t copyElems = aUbFactorAlign_;
        uint16_t vfLoopNum2 = static_cast<uint16_t>(
            (static_cast<uint32_t>(copyElems) + static_cast<uint32_t>(VF_VL_F32) - 1) /
            static_cast<uint32_t>(VF_VL_F32));
        uint32_t remaining2 = static_cast<uint32_t>(copyElems);

        __ubuf__ float* dstPtr = (__ubuf__ float*)tmpBuf_.Get<float>().GetPhyAddr();
        __ubuf__ float* srcPtr = (__ubuf__ float*)cacheBuf_.Get<float>().GetPhyAddr() + reduceDstOff;

        __VEC_SCOPE__
        {
            Reg::RegTensor<float> vtmp;
            Reg::MaskReg mask2;
            uint32_t rem = remaining2;

            for (uint16_t i = 0; i < vfLoopNum2; i++) {
                mask2 = Reg::UpdateMask<float>(rem);
                Reg::LoadAlign<float>(vtmp, srcPtr + static_cast<int32_t>(i) * VF_VL_F32);
                Reg::StoreAlign<float>(dstPtr + static_cast<int32_t>(i) * VF_VL_F32, vtmp, mask2);
            }
        }
    }

    // ══════════════════════════════════════════════════════════════════════
    // ZeroRowTailVf — zero tail of each tmpBuf row for partial R chunks.
    //
    // When rLen < rUbFactorAlign, CopyInChunk only writes rLen valid R-elements
    // per row + DataCopyPad rightPad. The tail [rLen+rightPad, rUbFactorAlign)
    // holds stale data from a previous chunk's PreElewiseVf output.
    //
    // For large shapes (rLen typically 8-aligned), StoreAlign at offset
    // rLen is 32B-aligned → safe. For small shapes (rLen not 8-aligned),
    // VL padding ensures buffer is large enough for the LoadAlign access.
    // ══════════════════════════════════════════════════════════════════════
    __aicore__ inline void ZeroRowTailVf(int32_t tmpBufIdx, int64_t rLen)
    {
        if (rLen >= rUbFactorAlign_)
            return;

        int64_t tailStart = rLen;
        int64_t tailCount = rUbFactorAlign_ - rLen;
        int64_t rowStride = rUbFactorAlign_; // fp32 elements per row
        int64_t rowCnt = aUbFactorAlign_;    // number of rows (padded)

        int32_t slotOff = tmpBufIdx * static_cast<int32_t>(bufSizeTmp_ / sizeof(float));
        __ubuf__ float* base = (__ubuf__ float*)tmpBuf_.Get<float>().GetPhyAddr() + slotOff;

        uint16_t rowLoop = static_cast<uint16_t>(rowCnt);
        uint16_t colLoop = static_cast<uint16_t>(
            (static_cast<uint32_t>(tailCount) + static_cast<uint32_t>(VF_VL_F32) - 1) /
            static_cast<uint32_t>(VF_VL_F32));
        uint32_t colRemaining = static_cast<uint32_t>(tailCount);

        __VEC_SCOPE__
        {
            Reg::RegTensor<float> vz;
            Reg::Duplicate<float>(vz, 0.0f);
            Reg::MaskReg mask;
            for (uint16_t r = 0; r < rowLoop; r++) {
                uint32_t remain = colRemaining;
                for (uint16_t c = 0; c < colLoop; c++) {
                    mask = Reg::UpdateMask<float>(remain);
                    int32_t off = static_cast<int32_t>(r) * static_cast<int32_t>(rowStride) +
                                  static_cast<int32_t>(tailStart) + static_cast<int32_t>(c) * VF_VL_F32;
                    Reg::StoreAlign<float>(base + off, vz, mask);
                }
            }
        }
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
    __aicore__ inline void DoCachingVf(int64_t rIdx)
    {
        int64_t cid = GetCacheID(rIdx);
        int64_t lvlStride = cacheLevelStride_; // 32B-aligned per-layer stride
        int64_t laneN = aUbFactorAlign_;       // valid lane count for mask

        __ubuf__ float* reduceOut = (__ubuf__ float*)tmpBuf_.Get<float>().GetPhyAddr();
        __ubuf__ float* cacheBase = (__ubuf__ float*)cacheBuf_.Get<float>().GetPhyAddr();

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

                // 1) Load reduce result (in tmpBuf[0])
                Reg::LoadAlign<float>(va, reduceOut + off);

                // 2) Absorb all lower cache layers [0..cid-1]
                for (uint16_t j = 0; j < cidU16; j++) {
                    int32_t jOff = static_cast<int32_t>(j) * static_cast<int32_t>(lvlStride) + off;
                    Reg::LoadAlign<float>(vc, cacheBase + jOff);
                    Reg::Add<float, Reg::MaskMergeMode::ZEROING>(va, va, vc, mask);
                }

                // 3) Overwrite cache[cid] layer
                int32_t dstOff = static_cast<int32_t>(cid) * static_cast<int32_t>(lvlStride) + off;
                Reg::StoreAlign<float>(cacheBase + dstOff, va, mask);
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
    TQue<QuePosition::VECIN, 2> preInQue_;
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
