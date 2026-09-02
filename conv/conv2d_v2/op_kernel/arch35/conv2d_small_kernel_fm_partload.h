/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV2D_SMALL_KERNEL_FM_PARTLOAD_H
#define CONV2D_SMALL_KERNEL_FM_PARTLOAD_H

#include "conv2d_small_kernel_parallelism.h"

using namespace AscendC;

constexpr uint32_t FMP_BLOCK_BYTES = 32; // DataCopy block size in bytes
// cinL1_ is derived from tiling-side decided kAL1: cinL1_ = kAL1 / kernelHxkernelW

// Entry constraints for this template:
//   1. FM not fullload L1 (FM is chunked over Cin), Weight fullload L1 (one-shot)
//   2. N axis fullload L0 (singleCoreCo loaded to L0B at once, no N-axis split)
//   3. pad <= kernel (no pad larger than kernel supported)
//   4. dtype: FP16*FP16 and INT8*INT8 only (FP16*INT8 not supported)
// Requirements:
//   - tiling supports M mode and HW mode
//   - multi-batch reuses the two FM ping-pong buffers after the current batch releases A1
//   - supports both conv2dv2 and extendconv2d (via ExtendParams)
//   - format supports NHWC and NCHW for both input and output
//   - FM chunked loading (shared with Conv2dSmallKernelParallelism)
//   - Weight one-shot loading into L1 (this subclass's only difference)

namespace {
static constexpr event_t FMP_EVT_WBS_DONE = static_cast<event_t>(0);
} // namespace

// FmPartload inherits from Parallelism to reuse the FM-chunked helpers
// (CalcChunkFmap / CalcChunkFmapW / LoadFmapL1Chunk / SetupLoad3DForChunk /
// PrepareCinBlock / RunKL0Loop / ProcessCinBlocks / CopyOutResult and shared members).
// It only overrides Init and Process, and adds its own LoadWeightL1Full:
//   - Weight is loaded ONCE into L1 (one-shot), not per Cin chunk.
//   - kL0 is taken directly from tiling (no per-chunk kL0 recomputation).
template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type = half,
          bool isNHWCin = false, bool isNHWCout = false, bool IsHwMode = false>
class Conv2dSmallKernelFmPartload : public Conv2dSmallKernelParallelism<FmapType, weightType, biasType, out0Type,
                                                                        out1Type, isNHWCin, isNHWCout, IsHwMode> {
public:
    using BaseT = Conv2dSmallKernelParallelism<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                               IsHwMode>;
    using L0cT = typename BaseT::L0cT;
    __aicore__ inline void Init(const Conv2DTilingData& tiling);
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                                   const ExtendParams* extendParams);

private:
    __aicore__ inline void LoadWeightL1Full(GM_ADDR filter);
    __aicore__ inline void ProcessOneBatch(uint32_t kL0, uint32_t kL0Iters, uint32_t kernelHxW, uint32_t convN,
                                           uint64_t hwOut, GM_ADDR y, const ExtendParams* extendParams,
                                           LocalTensor<weightType>& bl1Full, uint32_t batchBufBase,
                                           bool firstCinPrefetched, event_t batchEv);
    __aicore__ inline void ProcessHwMode(uint32_t kL0, uint32_t kL0Iters, uint32_t kernelHxW, uint32_t mmadN,
                                         uint64_t hwOut, GM_ADDR y, const ExtendParams* extendParams,
                                         LocalTensor<weightType>& bl1Full, uint32_t batchBufBase,
                                         bool firstCinPrefetched, event_t batchEv);
    __aicore__ inline void ProcessMMode(uint32_t kL0, uint32_t kL0Iters, uint32_t kernelHxW, uint32_t mmadN,
                                        uint64_t hwOut, GM_ADDR y, const ExtendParams* extendParams,
                                        LocalTensor<weightType>& bl1Full, uint32_t batchBufBase,
                                        bool firstCinPrefetched, event_t batchEv);
    __aicore__ inline void ProcessCinBlocks(LocalTensor<L0cT>& cl0, MmadParams& mp, LocalTensor<weightType>& bl1Full,
                                            uint32_t kL0, uint32_t kL0Iters, uint32_t kernelHxW, uint32_t curHi,
                                            uint32_t padTop, uint32_t padBottom, uint32_t hiLoadOff, uint32_t curWi,
                                            uint32_t wiLoadOff, uint32_t curM, uint32_t setupMOff, uint32_t setupWoOff,
                                            int32_t padLeft, int32_t padRight, bool loadWeight, uint32_t batchBufBase,
                                            bool firstCinPrefetched, event_t batchEv);
};

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernelFmPartload<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin,
                                                   isNHWCout, IsHwMode>::Init(const Conv2DTilingData& tiling)
{
    this->InitCommon(tiling);
    if (!this->coreActive_) {
        return;
    }

    if constexpr (IsHwMode) {
        // HW-mode: fmap W range from this core's Wo range (base InitCommon sets orgWin_ = win).
        uint32_t woStart = this->woIdxStart_;
        uint32_t woEnd = this->woIdxStart_ + this->actualWo_ - 1;
        uint32_t wiStart = woStart * this->tiling_->strideW;
        uint32_t wiEnd = woEnd * this->tiling_->strideW + this->tiling_->dilationW * (this->tiling_->kw - 1);
        uint32_t wiTotal = wiEnd - wiStart + 1;

        this->padLeftL1_ = 0;
        this->padRightL1_ = 0;
        this->curWiLoadL1_ = wiTotal;
        this->wiLoadStart_ = wiStart;
        if (wiStart < this->tiling_->padLeft) {
            this->padLeftL1_ = this->tiling_->padLeft - wiStart;
            this->curWiLoadL1_ -= this->padLeftL1_;
            this->wiLoadStart_ = 0;
        } else {
            this->wiLoadStart_ = wiStart - this->tiling_->padLeft;
        }
        if (wiEnd >= static_cast<uint32_t>(this->tiling_->win) + this->tiling_->padLeft) {
            this->padRightL1_ = wiEnd - (static_cast<uint32_t>(this->tiling_->win) + this->tiling_->padLeft) + 1;
            this->curWiLoadL1_ -= this->padRightL1_;
        }
        this->orgWin_ = this->curWiLoadL1_;
    }

    // Per-group cout: each core handles groupsPerCore groups.
    uint32_t groupCoutPar = this->tiling_->cout / this->tiling_->groups;
    uint32_t groupsPerCorePar = this->tiling_->groups / this->tiling_->groupDim;
    this->coutAligned_ = AlignB(groupsPerCorePar * groupCoutPar, GN0);

    this->n1Total_ = this->coutAligned_ / GN0;

    // Use tiling-side decided kAL1 for Cin L1 partitioning.
    // cinL1_ derived from kAL1: kAL1 = cinL1 * kernelHxkernelW.
    // hoL0_/woL0_ are already set by InitCommon from tiling_->hoL0/woL0.
    this->cinL1_ = AlignB(this->tiling_->kAL1 / this->tiling_->kernelHxkernelW, this->GK0);
    this->cinL1Blocks_ = CeilDiv(AlignB(this->tiling_->singleCoreCi, this->GK0Fmap), this->cinL1_);

    this->al1BufBytes_ = this->tiling_->aL1SpaceSize;
    this->al1ElemPerBuf_ = this->tiling_->aL1SpaceSize / sizeof(FmapType);

    // L1 layout: [FM pingpong 2 bufs][Weight fullload][Bias][Scale0][ReluWeight0][Scale1][ReluWeight1]
    // Adjacent batches reuse the same two buffers after the current batch finishes consuming A1.
    this->SetupL1SplitLayout();
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernelFmPartload<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin,
                                                   isNHWCout, IsHwMode>::LoadWeightL1Full(GM_ADDR filter)
{
    // Weight one-shot fullload into L1 (after FM pingpong buffers).
    // N axis fullload L0: singleCoreCo (per-core N partition) is fully loaded; no further N split.
    GlobalTensor<weightType> filterGm;
    filterGm.SetGlobalBuffer(reinterpret_cast<__gm__ weightType*>(filter),
                             this->k1Total_ * this->n1Total_ * GN0 * this->GK0);

    LocalTensor<weightType> bl1(TPosition::B1, this->bl1OffBytes_, this->bl1ElemCount_);

    if (this->tiling_->nDim == 1) {
        // Whole weight fits in this core's N partition: direct copy.
        DataCopy(bl1, filterGm[0], this->bl1ElemCount_);
    } else {
        // N-axis partitioning across cores, but per-core N is fullload L0 (one-shot).
        uint32_t n1Start = this->nIdx_ * this->tiling_->singleCoreCo / GN0;
        uint32_t tileBytes = GN0 * this->GK0 * sizeof(weightType);
        uint32_t srcGmOff = n1Start * GN0 * this->GK0;
        uint16_t blkLen = static_cast<uint16_t>((this->n1PerCore_ * tileBytes) / FMP_BLOCK_BYTES);
        uint16_t srcGap = static_cast<uint16_t>(((this->n1Total_ - this->n1PerCore_) * tileBytes) / FMP_BLOCK_BYTES);
        DataCopyParams cp(static_cast<uint16_t>(this->k1Total_), blkLen, srcGap, 0);
        DataCopy(bl1, filterGm[srcGmOff], cp);
    }
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernelFmPartload<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin,
                                                   isNHWCout, IsHwMode>::Process(GM_ADDR x, GM_ADDR filter,
                                                                                 GM_ADDR bias, GM_ADDR y,
                                                                                 const ExtendParams* extendParams)
{
    if (!this->coreActive_ || this->actualCo_ == 0 || this->singleCoreBatch_ == 0) {
        return;
    }

    // kL0 from tiling directly (N axis fullload L0: nl0 = nbl1, no per-chunk recomputation).
    uint32_t kL0 = this->tiling_->kL0;
    uint32_t kL0Iters = CeilDiv(this->kTotalFmap_, kL0);
    uint32_t kernelHxW = this->tiling_->kh * this->tiling_->kw;

    LocalTensor<weightType> bl1Full(TPosition::B1, this->bl1OffBytes_, this->bl1ElemCount_);
    uint32_t mmadN = AlignB(this->actualCo_, GN0);
    uint64_t hwOut = static_cast<uint64_t>(this->tiling_->hout) * this->tiling_->wout;

    // Stage 1: Load Bias/Scale/ReluWeight into L1 (small, channel-wise).
    this->LoadBiasScaleL1(bias, extendParams);
    SetFlag<HardEvent::MTE2_MTE1>(FMP_EVT_WBS_DONE);
    WaitFlag<HardEvent::MTE2_MTE1>(FMP_EVT_WBS_DONE);

    SetFlag<HardEvent::MTE2_FIX>(static_cast<event_t>(0));
    WaitFlag<HardEvent::MTE2_FIX>(static_cast<event_t>(0));

    // Stage 2: Weight fullload L1 (one-shot, no per-K splitting).
    // N axis fullload L0: per-core singleCoreCo loaded at once.
    LoadWeightL1Full(filter);
    SetFlag<HardEvent::MTE2_MTE1>(FMP_EVT_WBS_DONE);
    WaitFlag<HardEvent::MTE2_MTE1>(FMP_EVT_WBS_DONE);

    if (this->tiling_->hasBias) {
        this->LoadBiasToBT();
    }

    // Stage 3: Execute batches. Adjacent batches reuse the same two Cin ping-pong buffers.
    this->innerBatchIter_ = 0;
    this->curBatchIdx_ = this->batchStart_;
    this->SetFmapGmBatch(x, this->curBatchIdx_, 0);

    if (this->singleCoreBatch_ <= 1) {
        ProcessOneBatch(kL0, kL0Iters, kernelHxW, mmadN, hwOut, y, extendParams, bl1Full, 0, false,
                        EVT_BATCH_PREFETCH0);
    } else {
        uint32_t firstCurHi;
        uint32_t firstHiLoadOff;
        uint32_t firstCurWi = this->orgWin_;
        uint32_t firstWiLoadOff = 0;
        if constexpr (IsHwMode) {
            uint32_t firstHo = this->hoL0_ < this->actualHo_ ? this->hoL0_ : this->actualHo_;
            uint32_t firstWo = this->woL0_ < this->actualWo_ ? this->woL0_ : this->actualWo_;
            uint32_t firstPadTop;
            uint32_t firstPadBottom;
            int32_t firstPadLeft;
            int32_t firstPadRight;
            this->CalcChunkFmap(0, firstHo, firstCurHi, firstPadTop, firstPadBottom, firstHiLoadOff);
            this->CalcChunkFmapW(0, firstWo, firstCurWi, firstPadLeft, firstPadRight, firstWiLoadOff);
        } else {
            uint32_t firstCurM = this->hoL0_ < this->actualM_ ? this->hoL0_ : this->actualM_;
            uint32_t firstPadTop;
            uint32_t firstPadBottom;
            this->CalcChunkFmap(0, firstCurM, firstCurHi, firstPadTop, firstPadBottom, firstHiLoadOff);
        }

        this->PrefetchFirstCinBlock(kernelHxW, firstCurHi, firstHiLoadOff, firstCurWi, firstWiLoadOff, 0,
                                    EVT_BATCH_PREFETCH0);
        for (this->innerBatchIter_ = 0; this->innerBatchIter_ < this->singleCoreBatch_; this->innerBatchIter_++) {
            event_t curBatchEv = (this->innerBatchIter_ % 2 == 0) ? EVT_BATCH_PREFETCH0 : EVT_BATCH_PREFETCH1;
            event_t nextBatchEv = (this->innerBatchIter_ % 2 == 0) ? EVT_BATCH_PREFETCH1 : EVT_BATCH_PREFETCH0;

            this->curBatchIdx_ = this->batchStart_ + this->innerBatchIter_;
            this->SetFmapGmBatch(x, this->curBatchIdx_, 0);

            ProcessOneBatch(kL0, kL0Iters, kernelHxW, mmadN, hwOut, y, extendParams, bl1Full, 0, true, curBatchEv);
            if (this->innerBatchIter_ + 1 < this->singleCoreBatch_) {
                uint32_t savedBatchIdx = this->curBatchIdx_;
                this->curBatchIdx_ = this->batchStart_ + this->innerBatchIter_ + 1;
                this->SetFmapGmBatch(x, this->curBatchIdx_, 0);
                this->PrefetchFirstCinBlock(kernelHxW, firstCurHi, firstHiLoadOff, firstCurWi, firstWiLoadOff, 0,
                                            nextBatchEv);
                this->curBatchIdx_ = savedBatchIdx;
                this->SetFmapGmBatch(x, this->curBatchIdx_, 0);
            }
        }
    }
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, bool IsHwMode>
__aicore__ inline void
Conv2dSmallKernelFmPartload<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                            IsHwMode>::ProcessOneBatch(uint32_t kL0, uint32_t kL0Iters, uint32_t kernelHxW,
                                                       uint32_t convN, uint64_t hwOut, GM_ADDR y,
                                                       const ExtendParams* extendParams,
                                                       LocalTensor<weightType>& bl1Full, uint32_t batchBufBase,
                                                       bool firstCinPrefetched, event_t batchEv)
{
    if constexpr (IsHwMode) {
        ProcessHwMode(kL0, kL0Iters, kernelHxW, convN, hwOut, y, extendParams, bl1Full, batchBufBase,
                      firstCinPrefetched, batchEv);
    } else {
        ProcessMMode(kL0, kL0Iters, kernelHxW, convN, hwOut, y, extendParams, bl1Full, batchBufBase, firstCinPrefetched,
                     batchEv);
    }
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernelFmPartload<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin,
                                                   isNHWCout, IsHwMode>::ProcessHwMode(uint32_t kL0, uint32_t kL0Iters,
                                                                                       uint32_t kernelHxW,
                                                                                       uint32_t mmadN, uint64_t hwOut,
                                                                                       GM_ADDR y,
                                                                                       const ExtendParams* extendParams,
                                                                                       LocalTensor<weightType>& bl1Full,
                                                                                       uint32_t batchBufBase,
                                                                                       bool firstCinPrefetched,
                                                                                       event_t batchEv)
{
    // HW-mode: nested Ho/Wo-chunk loop; each chunk accumulates over all cin blocks.
    // Weight already fully loaded into L1 (one-shot), so loadWeight = false.
    bool needRowSplit = (this->actualWo_ < static_cast<uint32_t>(this->tiling_->wout));
    bool firstChunk = true;
    for (uint32_t hoOff = 0; hoOff < this->actualHo_; hoOff += this->hoL0_) {
        uint32_t curHo = this->hoL0_;
        if (hoOff + curHo > this->actualHo_) {
            curHo = this->actualHo_ - hoOff;
        }
        for (uint32_t woOff = 0; woOff < this->actualWo_; woOff += this->woL0_) {
            uint32_t curWo = this->woL0_;
            if (woOff + curWo > this->actualWo_) {
                curWo = this->actualWo_ - woOff;
            }
            uint32_t curM = curHo * curWo;
            uint32_t curMAlign = AlignB(curM, GM0);

            uint32_t curHi, padTop, padBottom, hiLoadOff;
            this->CalcChunkFmap(hoOff, curHo, curHi, padTop, padBottom, hiLoadOff);

            uint32_t curWi;
            int32_t padLeft, padRight;
            uint32_t wiLoadOff;
            this->CalcChunkFmapW(woOff, curWo, curWi, padLeft, padRight, wiLoadOff);

            LocalTensor<L0cT> cl0(TPosition::CO1, 0, this->L0C_ELEMS);
            MmadParams mp;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
            if constexpr (AscendC::IsSameType<FmapType, half>::value) {
                mp.fixShiftVal = this->tiling_->fixedShiftValue;
            }
#endif
            mp.m = curMAlign;
            mp.n = mmadN;
            mp.cmatrixInitVal = !(this->tiling_->hasBias);
            mp.cmatrixSource = (this->tiling_->hasBias != 0);

            bool chunkPrefetched = firstCinPrefetched && firstChunk;
            this->ProcessCinBlocks(cl0, mp, bl1Full, kL0, kL0Iters, kernelHxW, curHi, padTop, padBottom, hiLoadOff,
                                   curWi, wiLoadOff, curM, hoOff, woOff, padLeft, padRight, false, batchBufBase,
                                   chunkPrefetched, batchEv);
            firstChunk = false;

            uint32_t outOff = (this->hoIdxStart_ + hoOff) * static_cast<uint32_t>(this->tiling_->wout) +
                              this->woIdxStart_ + woOff;
            uint32_t fpMSize = needRowSplit ? curWo : curM;
            uint32_t fpDnNum = needRowSplit ? curHo : 1;
            uint32_t fpDstDnStride = needRowSplit ? static_cast<uint32_t>(this->tiling_->wout) :
                                                    static_cast<uint32_t>(hwOut);
            this->CopyOutResult(cl0, y, extendParams, outOff, fpMSize, curMAlign, fpDnNum, fpDstDnStride);
        }
    }
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernelFmPartload<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin,
                                                   isNHWCout, IsHwMode>::ProcessMMode(uint32_t kL0, uint32_t kL0Iters,
                                                                                      uint32_t kernelHxW,
                                                                                      uint32_t mmadN, uint64_t hwOut,
                                                                                      GM_ADDR y,
                                                                                      const ExtendParams* extendParams,
                                                                                      LocalTensor<weightType>& bl1Full,
                                                                                      uint32_t batchBufBase,
                                                                                      bool firstCinPrefetched,
                                                                                      event_t batchEv)
{
    // M-mode: M-loop -> CinL1 chunk loop -> KL0 inner loop -> Fixpipe.
    // Weight already fully loaded into L1 (one-shot), so loadWeight = false.
    this->curGroupCoutOff_ = 0;
    // N axis fullload L0: actualCo_ used in full (no per-group reload).
    for (uint32_t mOff = 0; mOff < this->actualM_; mOff += this->hoL0_) {
        uint32_t curM = this->hoL0_;
        if (mOff + curM > this->actualM_) {
            curM = this->actualM_ - mOff;
        }

        uint32_t curHi, padTop, padBottom, hiLoadOff;
        this->CalcChunkFmap(mOff, curM, curHi, padTop, padBottom, hiLoadOff);

        uint32_t curMAlign = AlignB(curM, GM0);
        LocalTensor<L0cT> cl0(TPosition::CO1, 0, this->L0C_ELEMS);

        MmadParams mp;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
        if constexpr (AscendC::IsSameType<FmapType, half>::value) {
            mp.fixShiftVal = this->tiling_->fixedShiftValue;
        }
#endif
        mp.m = curMAlign;
        mp.n = mmadN; // N axis fullload L0: actualCo_ used in full.
        mp.cmatrixInitVal = !(this->tiling_->hasBias);
        mp.cmatrixSource = (this->tiling_->hasBias != 0);

        // ProcessCinBlocks handles the CinL1 chunk loop + KL0 inner loop internally.
        // loadWeight = false: weight already fully loaded, skip per-chunk LoadWeightL1Block.
        bool chunkPrefetched = firstCinPrefetched && (mOff == 0);
        this->ProcessCinBlocks(cl0, mp, bl1Full, kL0, kL0Iters, kernelHxW, curHi, padTop, padBottom, hiLoadOff,
                               this->orgWin_, 0, curM, mOff, 0, 0, 0, false, batchBufBase, chunkPrefetched, batchEv);

        // Fixpipe out (supports NHWC and NCHW output formats).
        this->CopyOutResult(cl0, y, extendParams, this->mIdxStart_ + mOff, curM, curMAlign, 1,
                            static_cast<uint32_t>(hwOut));
    }
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, bool IsHwMode>
__aicore__ inline void
Conv2dSmallKernelFmPartload<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                            IsHwMode>::ProcessCinBlocks(LocalTensor<L0cT>& cl0, MmadParams& mp,
                                                        LocalTensor<weightType>& bl1Full, uint32_t kL0,
                                                        uint32_t kL0Iters, uint32_t kernelHxW, uint32_t curHi,
                                                        uint32_t padTop, uint32_t padBottom, uint32_t hiLoadOff,
                                                        uint32_t curWi, uint32_t wiLoadOff, uint32_t curM,
                                                        uint32_t setupMOff, uint32_t setupWoOff, int32_t padLeft,
                                                        int32_t padRight, bool loadWeight, uint32_t batchBufBase,
                                                        bool firstCinPrefetched, event_t batchEv)
{
    BaseT::ProcessCinBlocks(cl0, mp, bl1Full, kL0, kL0Iters, kernelHxW, curHi, padTop, padBottom, hiLoadOff, curWi,
                            wiLoadOff, curM, setupMOff, setupWoOff, padLeft, padRight, loadWeight, batchBufBase,
                            firstCinPrefetched, batchEv);
}
#endif // CONV2D_SMALL_KERNEL_FM_PARTLOAD_H
