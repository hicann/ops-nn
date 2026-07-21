/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV2D_SMALL_KERNEL_PARALLELISM_H
#define CONV2D_SMALL_KERNEL_PARALLELISM_H

#include "conv2d_small_kernel.h"

using namespace AscendC;

constexpr uint32_t CIN_L1_SPLIT_QUARTER = 4; // L1 split count for quarter division
constexpr uint32_t CIN_L1_SPLIT_HALF = 2;    // L1 split count for half division
// Minimum GK0 multiplier per split for 1x1 kernel
constexpr uint32_t MIN_GK0_MULTIPLIER_PER_SPLIT_1X1 = 2;

namespace {
static constexpr event_t EVT_WBS_DONE = static_cast<event_t>(0);
static constexpr event_t EVT_FMAP_BUF0 = static_cast<event_t>(1);
static constexpr event_t EVT_FMAP_BUF1 = static_cast<event_t>(2);
} // namespace

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type = half,
          bool isNHWCin = false, bool isNHWCout = false, bool IsHwMode = false>
class Conv2dSmallKernelParallelism : public Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type,
                                                              isNHWCin, isNHWCout, ConvFormat::FRACTAL_Z, IsHwMode> {
public:
    __aicore__ inline void Init(const Conv2DTilingData& tiling);
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                                   const ExtendParams* extendParams);

private:
    using Base = Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                   ConvFormat::FRACTAL_Z, IsHwMode>;
    using Output0T = typename Base::Output0T;
    using Output1T = typename Base::Output1T;
    using L0cT = typename Base::L0cT;

    __aicore__ inline void CalcChunkFmap(uint32_t mOff, uint32_t curM, uint32_t& curHi, uint32_t& padTop,
                                         uint32_t& padBottom, uint32_t& hiLoadOff);
    __aicore__ inline void CalcChunkFmapW(uint32_t woOff, uint32_t curWo, uint32_t& curWi, int32_t& padLeft,
                                          int32_t& padRight, uint32_t& wiLoadOff);
    __aicore__ inline void LoadFmapL1ChunkHw(LocalTensor<FmapType>& al1, uint32_t curHi, uint32_t hiLoadOff,
                                             uint32_t curWi, uint32_t wiLoadOff, uint32_t cinOff, uint32_t curCin);
    __aicore__ inline void LoadFmapL1ChunkM(LocalTensor<FmapType>& al1, uint32_t curHi, uint32_t hiLoadOff,
                                            uint32_t curWi, uint32_t cinOff, uint32_t curCin);
    __aicore__ inline void LoadFmapL1Chunk(uint32_t bufIdx, uint32_t curHi, uint32_t hiLoadOff, uint32_t padTop,
                                           uint32_t padBottom, uint32_t curWi, uint32_t wiLoadOff, uint32_t cinOff,
                                           uint32_t curCin);
    __aicore__ inline void LoadWeightL1Block(uint32_t kOff, uint32_t curK);
    __aicore__ inline void SetupLoad3DForChunk(uint32_t curHi, uint32_t mOff, uint32_t curM, uint32_t padTop,
                                               uint32_t padBottom, uint32_t woOff, int32_t padLeft, int32_t padRight,
                                               uint32_t curWi);
    __aicore__ inline uint32_t CalcKL0();
    __aicore__ inline void PrepareCinBlock(uint32_t kl1, uint32_t kernelHxW, uint32_t& cinOff, uint32_t& curCin,
                                           uint32_t& curCinOri, uint32_t& kOff, uint32_t& curKL1, uint32_t& kl1Buf,
                                           event_t& kl1Ev);
    __aicore__ inline void RunKL0Loop(LocalTensor<FmapType>& al1, LocalTensor<weightType>& bl1Full,
                                      LocalTensor<L0cT>& cl0, MmadParams& mp, uint32_t kOff, uint32_t curKL1,
                                      uint32_t kl1, uint32_t kL0, uint32_t kL0Iters);
    __aicore__ inline void ProcessCinBlocks(LocalTensor<L0cT>& cl0, MmadParams& mp, LocalTensor<weightType>& bl1Full,
                                            uint32_t kL0, uint32_t kL0Iters, uint32_t kernelHxW, uint32_t curHi,
                                            uint32_t padTop, uint32_t padBottom, uint32_t hiLoadOff, uint32_t curWi,
                                            uint32_t wiLoadOff, uint32_t curM, uint32_t setupMOff, uint32_t setupWoOff,
                                            int32_t padLeft, int32_t padRight, bool loadWeight);
    __aicore__ inline void CopyOutResult(LocalTensor<L0cT>& cl0, GM_ADDR y, const ExtendParams* extendParams,
                                         uint32_t outOff, uint32_t fpMSize, uint32_t curMAlign, uint32_t fpDnNum,
                                         uint32_t fpDstDnStride);
    __aicore__ inline void ProcessHwMode(uint32_t kL0, uint32_t kL0Iters, uint32_t kernelHxW, uint32_t mmadN,
                                         uint64_t hwOut, GM_ADDR y, const ExtendParams* extendParams,
                                         LocalTensor<weightType>& bl1Full);
    __aicore__ inline void ProcessMMode(uint32_t kL0, uint32_t kL0Iters, uint32_t kernelHxW, uint32_t mmadN,
                                        uint64_t hwOut, GM_ADDR y, const ExtendParams* extendParams,
                                        LocalTensor<weightType>& bl1Full);

    uint32_t n1Total_;
    uint32_t coutAligned_;
    uint32_t cinL1_;
    uint32_t cinL1Blocks_;
    uint32_t al1BufBytes_;
    uint32_t al1ElemPerBuf_;

    GlobalTensor<FmapType> fmapGm_;
    GlobalTensor<weightType> filterGm_;
};

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernelParallelism<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin,
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

    coutAligned_ = AlignB(this->tiling_->cout, GN0);

    n1Total_ = coutAligned_ / GN0;

    if (this->tiling_->kernelHxkernelW == 1) {
        if (this->cinAligned_ >= CIN_L1_SPLIT_QUARTER * MIN_GK0_MULTIPLIER_PER_SPLIT_1X1 * this->GK0) {
            cinL1_ = AlignB(this->cinAligned_ / CIN_L1_SPLIT_QUARTER, this->GK0);
        } else if (this->cinAligned_ >= CIN_L1_SPLIT_HALF * MIN_GK0_MULTIPLIER_PER_SPLIT_1X1 * this->GK0) {
            cinL1_ = AlignB(this->cinAligned_ / CIN_L1_SPLIT_HALF, this->GK0);
        }
    } else {
        if (this->cinAligned_ >= CIN_L1_SPLIT_QUARTER * this->GK0) {
            cinL1_ = AlignB(this->cinAligned_ / CIN_L1_SPLIT_QUARTER, this->GK0);
        } else if (this->cinAligned_ >= CIN_L1_SPLIT_HALF * this->GK0) {
            cinL1_ = AlignB(this->cinAligned_ / CIN_L1_SPLIT_HALF, this->GK0);
        }
    }

    cinL1Blocks_ = CeilDiv(this->cinAligned_, cinL1_);

    uint32_t maxHoRelEnd;
    if constexpr (IsHwMode) {
        maxHoRelEnd = this->hoL0_ - 1; // HW-mode: hoL0_ is a count of Ho rows.
    } else {
        maxHoRelEnd = (this->hoL0_ + this->tiling_->wout - 2) / this->tiling_->wout;
    }
    uint32_t maxHiL1 = maxHoRelEnd * this->tiling_->strideH + this->tiling_->dilationH * (this->tiling_->kh - 1) +
                       this->tiling_->padTop + 1;
    uint32_t hinFull = static_cast<uint32_t>(this->tiling_->hin) + this->tiling_->padTop + this->tiling_->padBottom;
    if (maxHiL1 > hinFull) {
        maxHiL1 = hinFull;
    }

    al1ElemPerBuf_ = maxHiL1 * this->orgWin_ * cinL1_;
    al1BufBytes_ = al1ElemPerBuf_ * sizeof(FmapType);

    this->bl1OffBytes_ = 2 * al1BufBytes_; // 2 pingpong includes two blocks.

    this->bl1ElemCount_ = this->k1Total_ * this->n1PerCore_ * GN0 * this->GK0;
    uint32_t afterBl1 = this->bl1OffBytes_ + this->bl1ElemCount_ * sizeof(weightType);
    this->biasL1OffBytes_ = AlignB(afterBl1, ADDR_ALIGN_SIZE);
    uint32_t afterBias = this->tiling_->hasBias ?
                             this->biasL1OffBytes_ + this->tiling_->singleCoreCo * sizeof(int32_t) :
                             this->biasL1OffBytes_;
    this->scale0L1OffBytes_ = AlignB(afterBias, ADDR_ALIGN_SIZE);
    uint32_t afterScale0 = this->tiling_->quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT) ?
                               afterBias + this->tiling_->singleCoreCo * sizeof(uint64_t) :
                               afterBias;
    this->reluWeight0L1OffBytes_ = AlignB(afterScale0, ADDR_ALIGN_SIZE);
    uint32_t afterReluWeight0 = this->tiling_->reluMode0 == static_cast<uint8_t>(ReluMode::VECTOR_RELU) ?
                                    afterScale0 + this->tiling_->singleCoreCo * sizeof(float) :
                                    afterScale0;
    this->scale1L1OffBytes_ = AlignB(afterReluWeight0, ADDR_ALIGN_SIZE);
    uint32_t afterScale1 = this->tiling_->quantMode1 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT) ?
                               afterReluWeight0 + this->tiling_->singleCoreCo * sizeof(uint64_t) :
                               afterReluWeight0;
    this->reluWeight1L1OffBytes_ = AlignB(afterScale1, ADDR_ALIGN_SIZE);
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernelParallelism<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin,
                                                    isNHWCout, IsHwMode>::CalcChunkFmap(uint32_t mOff, uint32_t curM,
                                                                                        uint32_t& curHi,
                                                                                        uint32_t& padTop,
                                                                                        uint32_t& padBottom,
                                                                                        uint32_t& hiLoadOff)
{
    uint32_t hoStart;
    uint32_t hoEnd;
    if constexpr (IsHwMode) {
        // HW-mode: mOff/curM are Ho-row offset/count for this core.
        hoStart = this->hoIdxStart_ + mOff;
        hoEnd = hoStart + curM - 1;
    } else {
        hoStart = (this->mIdxStart_ + mOff) / this->tiling_->wout;
        hoEnd = (this->mIdxStart_ + mOff + curM - 1) / this->tiling_->wout;
    }
    uint32_t hoRelEnd = hoEnd - hoStart;

    uint32_t hiStart = hoStart * this->tiling_->strideH;
    uint32_t hiEnd = hoEnd * this->tiling_->strideH + this->tiling_->dilationH * (this->tiling_->kh - 1);
    if (hiStart < this->tiling_->padTop) {
        padTop = this->tiling_->padTop - hiStart;
        hiLoadOff = 0;
    } else {
        padTop = 0;
        hiLoadOff = hiStart - this->tiling_->padTop;
    }

    uint32_t needHi = hoRelEnd * this->tiling_->strideH + this->tiling_->dilationH * (this->tiling_->kh - 1) + padTop +
                      1;
    uint32_t maxGmRows = static_cast<uint32_t>(this->tiling_->hin) - hiLoadOff;
    curHi = (needHi < maxGmRows) ? needHi : maxGmRows;
    padBottom = 0;
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernelParallelism<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin,
                                                    isNHWCout, IsHwMode>::CalcChunkFmapW(uint32_t woOff, uint32_t curWo,
                                                                                         uint32_t& curWi,
                                                                                         int32_t& padLeft,
                                                                                         int32_t& padRight,
                                                                                         uint32_t& wiLoadOff)
{
    if constexpr (!IsHwMode) {
        // M-mode: 全宽，直接用核全局 W 范围
        curWi = this->orgWin_;
        padLeft = 0;
        padRight = 0;
        wiLoadOff = this->wiLoadStart_; // M 模式 wiLoadStart_ 未初始化，实际未用此路径
        return;
    }

    // HW-mode: 按本 chunk 的 Wo 范围重算 Wi
    uint32_t woStart = this->woIdxStart_ + woOff;
    uint32_t woEnd = woStart + curWo - 1;
    uint32_t wiStart = woStart * this->tiling_->strideW;
    uint32_t wiEnd = woEnd * this->tiling_->strideW + this->tiling_->dilationW * (this->tiling_->kw - 1);
    uint32_t wiTotal = wiEnd - wiStart + 1;

    padLeft = 0;
    padRight = 0;
    curWi = wiTotal;
    wiLoadOff = wiStart;

    if (wiStart < this->tiling_->padLeft) {
        padLeft = this->tiling_->padLeft - wiStart;
        curWi -= padLeft;
        wiLoadOff = 0;
    } else {
        wiLoadOff = wiStart - this->tiling_->padLeft;
    }

    if (wiEnd >= static_cast<uint32_t>(this->tiling_->win) + this->tiling_->padLeft) {
        padRight = wiEnd - (static_cast<uint32_t>(this->tiling_->win) + this->tiling_->padLeft) + 1;
        curWi -= padRight;
    }
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, bool IsHwMode>
__aicore__ inline uint32_t Conv2dSmallKernelParallelism<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin,
                                                        isNHWCout, IsHwMode>::CalcKL0()
{
    // kL0 is the largest factor of kL1OverK0 other than itself.
    uint32_t kL1OverK0 = cinL1_ * this->tiling_->kernelHxkernelW / this->GK0;
    uint32_t maxFactor = kL1OverK0 / 2;
    uint32_t maxAllowedFactor = this->tiling_->kL0 / this->GK0;
    uint32_t upperBound = (maxFactor < maxAllowedFactor) ? maxFactor : maxAllowedFactor;

    for (uint32_t i = upperBound; i > 0; --i) {
        if (kL1OverK0 % i == 0) {
            return i * this->GK0;
        }
    }
    return this->GK0;
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, bool IsHwMode>
__aicore__ inline void
Conv2dSmallKernelParallelism<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                             IsHwMode>::PrepareCinBlock(uint32_t kl1, uint32_t kernelHxW, uint32_t& cinOff,
                                                        uint32_t& curCin, uint32_t& curCinOri, uint32_t& kOff,
                                                        uint32_t& curKL1, uint32_t& kl1Buf, event_t& kl1Ev)
{
    cinOff = kl1 * cinL1_;
    curCinOri = cinL1_;
    curCin = cinL1_;
    if (cinOff + curCin > this->tiling_->singleCoreCi) {
        curCin = this->cinAligned_ - cinOff;
        curCinOri = this->tiling_->singleCoreCi - cinOff;
    }

    kOff = cinOff * kernelHxW;
    curKL1 = curCin * kernelHxW;

    kl1Buf = kl1 % 2; // 2 pingpong
    kl1Ev = (kl1Buf == 0) ? EVT_FMAP_BUF0 : EVT_FMAP_BUF1;
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernelParallelism<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin,
                                                    isNHWCout, IsHwMode>::RunKL0Loop(LocalTensor<FmapType>& al1,
                                                                                     LocalTensor<weightType>& bl1Full,
                                                                                     LocalTensor<L0cT>& cl0,
                                                                                     MmadParams& mp, uint32_t kOff,
                                                                                     uint32_t curKL1, uint32_t kl1,
                                                                                     uint32_t kL0, uint32_t kL0Iters)
{
    uint32_t kl0Start = kOff / kL0;
    uint32_t kl0End = CeilDiv(kOff + curKL1, kL0);
    if (kl0End > kL0Iters) {
        kl0End = kL0Iters;
    }

    SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(0));
    SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(1));

    for (uint32_t kl0 = kl0Start; kl0 < kl0End; kl0++) {
        uint32_t kOffInner = kl0 * kL0;
        uint32_t curKInner = kL0;
        if (kOffInner + curKInner > kOff + curKL1) {
            curKInner = kOff + curKL1 - kOffInner;
        }

        uint32_t lbuf = kl0 % 2; // 2 pingpong
        event_t lev = static_cast<event_t>(lbuf);

        LocalTensor<FmapType> al0(TPosition::A2, lbuf * AL0_BUF_BYTES, AL0_BUF_ELEMS);
        LocalTensor<weightType> bl0(TPosition::B2, lbuf * BL0_BUF_BYTES, BL0_BUF_ELEMS);

        WaitFlag<HardEvent::M_MTE1>(lev);

        this->DoLoadAL0(al1, al0, kOffInner - kOff, curKInner);
        this->DoLoadBL0(bl0, bl1Full, kOffInner, curKInner);
        SetFlag<HardEvent::MTE1_M>(lev);
        WaitFlag<HardEvent::MTE1_M>(lev);

        bool isLast = (kl1 == cinL1Blocks_ - 1) && (kl0 == kl0End - 1);
        mp.unitFlag = isLast ? UNIT_FLAG_ENABLE_WITH_FLIP : UNIT_FLAG_ENABLE_ONLY;
        mp.k = curKInner;

        Mmad(cl0, al0, bl0, mp);

        mp.cmatrixInitVal = false;
        mp.cmatrixSource = false;

        SetFlag<HardEvent::M_MTE1>(lev);
    }

    WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(0));
    WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(1));
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernelParallelism<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin,
                                                    isNHWCout, IsHwMode>::LoadFmapL1ChunkHw(LocalTensor<FmapType>& al1,
                                                                                            uint32_t curHi,
                                                                                            uint32_t hiLoadOff,
                                                                                            uint32_t curWi,
                                                                                            uint32_t wiLoadOff,
                                                                                            uint32_t cinOff,
                                                                                            uint32_t curCin)
{
    const Conv2DTilingData& t = *this->tiling_;
    if constexpr (isNHWCin) {
        // NHWC input: [N,H,W,C]
        Nd2NzParams p;
        if (curWi == static_cast<uint32_t>(t.win)) {
            // Full-width: single Nd
            p.ndNum = 1;
            p.nValue = curHi * curWi;
            p.srcNdMatrixStride = 0;
            p.dstNzMatrixStride = 0;
        } else {
            // W-window: one Nd per loaded H row
            p.ndNum = curHi;
            p.nValue = curWi;
            p.srcNdMatrixStride = static_cast<uint32_t>(t.win * t.cin);
            p.dstNzMatrixStride = curWi * this->GK0;
        }
        p.dValue = curCin;
        p.srcDValue = static_cast<uint32_t>(t.cin);
        p.dstNzC0Stride = curHi * curWi;
        p.dstNzNStride = 1;

        uint64_t gmOff = static_cast<uint64_t>(hiLoadOff) * t.win * t.cin + static_cast<uint64_t>(wiLoadOff) * t.cin +
                         cinOff;
        DataCopy(al1, fmapGm_[gmOff], p);
    } else {
        // NCHW input: [N,C,H,W]
        Dn2NzParams p;
        if (curWi == static_cast<uint32_t>(t.win)) {
            // Full-width rows: single contiguous Dn.
            p.dnNum = 1;
            p.nValue = curHi * curWi;
            p.srcDnMatrixStride = 0;
            p.dstNzMatrixStride = 0;
        } else {
            // W-window: one Dn per loaded H row, stride win in GM.
            p.dnNum = curHi;
            p.nValue = curWi;
            p.srcDnMatrixStride = static_cast<uint32_t>(t.win);
            p.dstNzMatrixStride = curWi * this->GK0;
        }
        p.dValue = curCin;
        p.srcDValue = static_cast<uint32_t>(t.hin * t.win);
        p.dstNzC0Stride = curHi * curWi;
        p.dstNzNStride = 1;

        uint64_t gmOff = static_cast<uint64_t>(cinOff) * static_cast<uint64_t>(t.hin) * static_cast<uint64_t>(t.win) +
                         static_cast<uint64_t>(hiLoadOff) * t.win + wiLoadOff;
        DataCopy(al1, fmapGm_[gmOff], p);
    }
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernelParallelism<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin,
                                                    isNHWCout, IsHwMode>::LoadFmapL1ChunkM(LocalTensor<FmapType>& al1,
                                                                                           uint32_t curHi,
                                                                                           uint32_t hiLoadOff,
                                                                                           uint32_t curWi,
                                                                                           uint32_t cinOff,
                                                                                           uint32_t curCin)
{
    const Conv2DTilingData& t = *this->tiling_;
    if constexpr (isNHWCin) {
        // M-mode NHWC: full-width, single Nd per cin-block
        Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = curHi * curWi;
        p.dValue = curCin;
        p.srcDValue = static_cast<uint32_t>(t.cin);
        p.srcNdMatrixStride = 0;
        p.dstNzC0Stride = curHi * curWi;
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 0;

        uint64_t gmOff = static_cast<uint64_t>(cinOff) + static_cast<uint64_t>(hiLoadOff) * t.win * t.cin;
        DataCopy(al1, fmapGm_[gmOff], p);
    } else {
        // M-mode NCHW: full-width, single Dn per cin-block
        Dn2NzParams p;
        p.dnNum = 1;
        p.nValue = curHi * curWi;
        p.dValue = curCin;
        p.srcDnMatrixStride = 0;
        p.srcDValue = static_cast<uint32_t>(t.hin * t.win);
        p.dstNzC0Stride = curHi * curWi;
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 0;

        uint64_t gmOff = static_cast<uint64_t>(cinOff) * static_cast<uint64_t>(t.hin) * static_cast<uint64_t>(t.win) +
                         static_cast<uint64_t>(hiLoadOff) * curWi;
        DataCopy(al1, fmapGm_[gmOff], p);
    }
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, bool IsHwMode>
__aicore__ inline void
Conv2dSmallKernelParallelism<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                             IsHwMode>::LoadFmapL1Chunk(uint32_t bufIdx, uint32_t curHi, uint32_t hiLoadOff,
                                                        uint32_t padTop, uint32_t padBottom, uint32_t curWi,
                                                        uint32_t wiLoadOff, uint32_t cinOff, uint32_t curCin)
{
    (void)padTop;
    (void)padBottom;
    uint32_t elemCount = curHi * curWi * curCin;
    uint32_t bufByteOff = bufIdx * al1BufBytes_;
    LocalTensor<FmapType> al1(TPosition::A1, bufByteOff, elemCount);

    if constexpr (IsHwMode) {
        LoadFmapL1ChunkHw(al1, curHi, hiLoadOff, curWi, wiLoadOff, cinOff, curCin);
    } else {
        LoadFmapL1ChunkM(al1, curHi, hiLoadOff, curWi, cinOff, curCin);
    }
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernelParallelism<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin,
                                                    isNHWCout, IsHwMode>::LoadWeightL1Block(uint32_t kOff,
                                                                                            uint32_t curK)
{
    uint32_t k1Block = CeilDiv(curK, this->GK0);
    uint32_t k1Start = kOff / this->GK0;
    uint32_t elemCount = k1Block * this->n1PerCore_ * GN0 * this->GK0;

    uint32_t l1ByteOff = this->bl1OffBytes_ + k1Start * this->n1PerCore_ * GN0 * this->GK0 * sizeof(weightType);
    LocalTensor<weightType> bl1(TPosition::B1, l1ByteOff, elemCount);

    if (this->tiling_->nDim == 1) {
        uint32_t gmOff = k1Start * n1Total_ * GN0 * this->GK0;
        DataCopy(bl1, filterGm_[gmOff], elemCount);
    } else {
        uint32_t n1Start = this->nIdx_ * this->tiling_->singleCoreCo / GN0;
        uint32_t tileBytes = GN0 * this->GK0 * sizeof(weightType);
        uint32_t srcGmOff = k1Start * n1Total_ * GN0 * this->GK0 + n1Start * GN0 * this->GK0;
        uint16_t blkLen = static_cast<uint16_t>((this->n1PerCore_ * tileBytes) / 32);
        uint16_t srcGap = static_cast<uint16_t>(((n1Total_ - this->n1PerCore_) * tileBytes) / 32);
        DataCopyParams cp(static_cast<uint16_t>(k1Block), blkLen, srcGap, 0);
        DataCopy(bl1, filterGm_[srcGmOff], cp);
    }
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, bool IsHwMode>
__aicore__ inline void
Conv2dSmallKernelParallelism<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                             IsHwMode>::SetupLoad3DForChunk(uint32_t curHi, uint32_t mOff, uint32_t curM,
                                                            uint32_t padTop, uint32_t padBottom, uint32_t woOff,
                                                            int32_t padLeft, int32_t padRight, uint32_t curWi)
{
    uint8_t padList[PAD_LIST_NUM] = {
        static_cast<uint8_t>(IsHwMode ? padLeft : static_cast<int32_t>(this->tiling_->padLeft)),
        static_cast<uint8_t>(IsHwMode ? padRight : static_cast<int32_t>(this->tiling_->padRight)),
        static_cast<uint8_t>(padTop), static_cast<uint8_t>(PAD_BOTTOM_VALUE)};
    Load3DSetFMatrixCal(curHi, curWi, padList);
    Load3DSetPaddingCal(this->tiling_->offsetx);

    uint32_t curMAlign = AlignB(curM, GM0);
    // HW-mode: per-chunk fmap covers only this chunk's columns; Load3D starts at FMatrix column 0.
    // M-mode:  fmap covers the full core H×W range; Load3D starts at the correct output column.
    uint32_t firstOutCol = IsHwMode ? 0 : ((this->mIdxStart_ + mOff) % this->tiling_->wout);

    this->load3dXmTmp_ = ((static_cast<uint64_t>(curMAlign) & MASK_16) << MSTEP_OFFSET) |
                         ((static_cast<uint64_t>(firstOutCol) & MASK_16) << POSM_OFFSET);

    this->load3dXtBase_ = ((static_cast<uint64_t>(this->tiling_->strideW) & MASK_6) << 0) |
                          ((static_cast<uint64_t>(this->tiling_->kw) & MASK_8) << KERNELW_OFFSET) |
                          ((static_cast<uint64_t>(this->tiling_->kh) & MASK_8) << KERNELH_OFFSET) |
                          ((static_cast<uint64_t>(this->tiling_->strideH) & MASK_6) << STRIDEH_OFFSET) |
                          ((static_cast<uint64_t>(this->tiling_->kw) & NINTH_BIT_MASK) << KERNELW_HIGHEST_BIT_OFFSET) |
                          ((static_cast<uint64_t>(this->tiling_->kh) & NINTH_BIT_MASK) << KERNELH_HIGHEST_BIT_OFFSET) |
                          ((static_cast<uint64_t>(this->tiling_->dilationW) & MASK_8) << DILATIONW_OFFSET) |
                          ((static_cast<uint64_t>(this->tiling_->dilationH) & MASK_8) << DILATIONH_OFFSET) |
                          ((static_cast<uint64_t>(cinL1_) & MASK_16) << CIN_OFFSET);

#if defined(ASC_DEVKIT_VERSION_NUM) && (ASC_DEVKIT_VERSION_NUM >= 90000000)
    LoadDataRepeatParamWithStride repeatParams(0, 1, 0, static_cast<uint16_t>(curMAlign / GM0));
    SetLoadDataRepeatWithStride(repeatParams);
#else
    LoadDataRepeatParam repeatParams(0, 1, 0, static_cast<uint16_t>(curMAlign / GM0));
    SetLoadDataRepeat(repeatParams);
#endif
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, bool IsHwMode>
__aicore__ inline void
Conv2dSmallKernelParallelism<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                             IsHwMode>::ProcessCinBlocks(LocalTensor<L0cT>& cl0, MmadParams& mp,
                                                         LocalTensor<weightType>& bl1Full, uint32_t kL0,
                                                         uint32_t kL0Iters, uint32_t kernelHxW, uint32_t curHi,
                                                         uint32_t padTop, uint32_t padBottom, uint32_t hiLoadOff,
                                                         uint32_t curWi, uint32_t wiLoadOff, uint32_t curM,
                                                         uint32_t setupMOff, uint32_t setupWoOff, int32_t padLeft,
                                                         int32_t padRight, bool loadWeight)
{
    SetFlag<HardEvent::MTE1_MTE2>(EVT_FMAP_BUF0);
    SetFlag<HardEvent::MTE1_MTE2>(EVT_FMAP_BUF1);

    for (uint32_t kl1 = 0; kl1 < cinL1Blocks_; kl1++) {
        uint32_t cinOff;
        uint32_t curCin;
        uint32_t curCinOri;
        uint32_t kOff;
        uint32_t curKL1;
        uint32_t kl1Buf;
        event_t kl1Ev;
        PrepareCinBlock(kl1, kernelHxW, cinOff, curCin, curCinOri, kOff, curKL1, kl1Buf, kl1Ev);

        WaitFlag<HardEvent::MTE1_MTE2>(kl1Ev);

        LoadFmapL1Chunk(kl1Buf, curHi, hiLoadOff, padTop, padBottom, curWi, wiLoadOff, cinOff, curCinOri);
        if (loadWeight) {
            LoadWeightL1Block(kOff, curKL1);
        }

        SetFlag<HardEvent::MTE2_MTE1>(kl1Ev);
        WaitFlag<HardEvent::MTE2_MTE1>(kl1Ev);

        SetupLoad3DForChunk(curHi, setupMOff, curM, padTop, padBottom, setupWoOff, padLeft, padRight, curWi);

        uint32_t al1ElemCount = curHi * curWi * curCin;
        uint32_t al1BufOff = kl1Buf * al1BufBytes_;
        LocalTensor<FmapType> al1(TPosition::A1, al1BufOff, al1ElemCount);

        RunKL0Loop(al1, bl1Full, cl0, mp, kOff, curKL1, kl1, kL0, kL0Iters);

        SetFlag<HardEvent::MTE1_MTE2>(kl1Ev);
    }

    WaitFlag<HardEvent::MTE1_MTE2>(EVT_FMAP_BUF0);
    WaitFlag<HardEvent::MTE1_MTE2>(EVT_FMAP_BUF1);
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, bool IsHwMode>
__aicore__ inline void
Conv2dSmallKernelParallelism<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                             IsHwMode>::CopyOutResult(LocalTensor<L0cT>& cl0, GM_ADDR y,
                                                      const ExtendParams* extendParams, uint32_t outOff,
                                                      uint32_t fpMSize, uint32_t curMAlign, uint32_t fpDnNum,
                                                      uint32_t fpDstDnStride)
{
    if constexpr (isNHWCout) {
        this->template DoCopyOut<Output0T, 0, CFG_ROW_MAJOR, CFG_ROW_MAJOR_FIXED_POINT>(
            y, cl0, outOff, fpMSize, curMAlign, this->actualCo_, fpDnNum, fpDstDnStride);
        if (this->tiling_->dualOutput) {
            this->template DoCopyOut<Output1T, 1, CFG_ROW_MAJOR, CFG_ROW_MAJOR_FIXED_POINT>(
                extendParams->y1, cl0, outOff, fpMSize, curMAlign, this->actualCo_, fpDnNum, fpDstDnStride);
        }
    } else {
        this->template DoCopyOut<Output0T, 0, CFG_COLUMN_MAJOR, CFG_COLUMN_MAJOR_FIXED_POINT>(
            y, cl0, outOff, fpMSize, curMAlign, this->actualCo_, fpDnNum, fpDstDnStride);
        if (this->tiling_->dualOutput) {
            this->template DoCopyOut<Output1T, 1, CFG_COLUMN_MAJOR, CFG_COLUMN_MAJOR_FIXED_POINT>(
                extendParams->y1, cl0, outOff, fpMSize, curMAlign, this->actualCo_, fpDnNum, fpDstDnStride);
        }
    }
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, bool IsHwMode>
__aicore__ inline void
Conv2dSmallKernelParallelism<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                             IsHwMode>::ProcessHwMode(uint32_t kL0, uint32_t kL0Iters, uint32_t kernelHxW,
                                                      uint32_t mmadN, uint64_t hwOut, GM_ADDR y,
                                                      const ExtendParams* extendParams,
                                                      LocalTensor<weightType>& bl1Full)
{
    // HW-mode: nested Ho/Wo-chunk loop; each chunk accumulates over all cin blocks.
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
            CalcChunkFmap(hoOff, curHo, curHi, padTop, padBottom, hiLoadOff);

            uint32_t curWi;
            int32_t padLeft, padRight;
            uint32_t wiLoadOff;
            CalcChunkFmapW(woOff, curWo, curWi, padLeft, padRight, wiLoadOff);

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

            ProcessCinBlocks(cl0, mp, bl1Full, kL0, kL0Iters, kernelHxW, curHi, padTop, padBottom, hiLoadOff, curWi,
                             wiLoadOff, curM, hoOff, woOff, padLeft, padRight, firstChunk);
            firstChunk = false;

            uint32_t outOff = (this->hoIdxStart_ + hoOff) * static_cast<uint32_t>(this->tiling_->wout) +
                              this->woIdxStart_ + woOff;
            uint32_t fpMSize = needRowSplit ? curWo : curM;
            uint32_t fpDnNum = needRowSplit ? curHo : 1;
            uint32_t fpDstDnStride = needRowSplit ? static_cast<uint32_t>(this->tiling_->wout) :
                                                    static_cast<uint32_t>(hwOut);
            CopyOutResult(cl0, y, extendParams, outOff, fpMSize, curMAlign, fpDnNum, fpDstDnStride);
        }
    }
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernelParallelism<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin,
                                                    isNHWCout, IsHwMode>::ProcessMMode(uint32_t kL0, uint32_t kL0Iters,
                                                                                       uint32_t kernelHxW,
                                                                                       uint32_t mmadN, uint64_t hwOut,
                                                                                       GM_ADDR y,
                                                                                       const ExtendParams* extendParams,
                                                                                       LocalTensor<weightType>& bl1Full)
{
    for (uint32_t mOff = 0; mOff < this->actualM_; mOff += this->hoL0_) {
        uint32_t curM = this->hoL0_;
        if (mOff + curM > this->actualM_) {
            curM = this->actualM_ - mOff;
        }

        uint32_t curHi, padTop, padBottom, hiLoadOff;
        CalcChunkFmap(mOff, curM, curHi, padTop, padBottom, hiLoadOff);

        uint32_t curMAlign = AlignB(curM, GM0);
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

        ProcessCinBlocks(cl0, mp, bl1Full, kL0, kL0Iters, kernelHxW, curHi, padTop, padBottom, hiLoadOff, this->orgWin_,
                         0, curM, mOff, 0, 0, 0, (mOff == 0));

        CopyOutResult(cl0, y, extendParams, this->mIdxStart_ + mOff, curM, curMAlign, 1, static_cast<uint32_t>(hwOut));
    }
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernelParallelism<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin,
                                                    isNHWCout, IsHwMode>::Process(GM_ADDR x, GM_ADDR filter,
                                                                                  GM_ADDR bias, GM_ADDR y,
                                                                                  const ExtendParams* extendParams)
{
    if (!this->coreActive_ || this->actualCo_ == 0) {
        return;
    }

    this->LoadBiasScaleL1(bias, extendParams);
    SetFlag<HardEvent::MTE2_MTE1>(EVT_WBS_DONE);
    WaitFlag<HardEvent::MTE2_MTE1>(EVT_WBS_DONE);

    SetFlag<HardEvent::MTE2_FIX>(static_cast<event_t>(0));
    WaitFlag<HardEvent::MTE2_FIX>(static_cast<event_t>(0));

    uint32_t kL0 = CalcKL0();

    uint32_t kL0Iters = CeilDiv(this->kTotal_, kL0);
    uint32_t kernelHxW = this->tiling_->kh * this->tiling_->kw;

    LocalTensor<weightType> bl1Full(TPosition::B1, this->bl1OffBytes_, this->bl1ElemCount_);

    if constexpr (isNHWCin) {
        // NHWC input: [N,H,W,C]
        uint64_t batchFmapOff = static_cast<uint64_t>(this->batchIdx_) * this->tiling_->hin * this->tiling_->win *
                                this->tiling_->cin;
        fmapGm_.SetGlobalBuffer(reinterpret_cast<__gm__ FmapType*>(x) + batchFmapOff);
    } else {
        // NCHW input: [N,C,H,W]
        uint64_t batchFmapOff = static_cast<uint64_t>(this->batchIdx_) * this->tiling_->cin * this->tiling_->hin *
                                this->tiling_->win;
        fmapGm_.SetGlobalBuffer(reinterpret_cast<__gm__ FmapType*>(x) + batchFmapOff);
    }
    filterGm_.SetGlobalBuffer(reinterpret_cast<__gm__ weightType*>(filter));

    if (this->tiling_->hasBias) {
        this->LoadBiasToBT();
    }

    uint32_t mmadN = AlignB(this->actualCo_, GN0);
    uint64_t hwOut = static_cast<uint64_t>(this->tiling_->hout) * this->tiling_->wout;

    if constexpr (IsHwMode) {
        ProcessHwMode(kL0, kL0Iters, kernelHxW, mmadN, hwOut, y, extendParams, bl1Full);
    } else {
        ProcessMMode(kL0, kL0Iters, kernelHxW, mmadN, hwOut, y, extendParams, bl1Full);
    }
}

#endif
