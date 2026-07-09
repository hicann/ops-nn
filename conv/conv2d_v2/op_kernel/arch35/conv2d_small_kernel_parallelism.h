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
          bool isNHWCin = false, bool isNHWCout = false>
class Conv2dSmallKernelParallelism
    : public Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout> {
public:
    __aicore__ inline void Init(const Conv2DTilingData& tiling);
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                                   const ExtendParams* extendParams);

private:
    __aicore__ inline void CalcChunkFmap(uint32_t mOff, uint32_t curM, uint32_t& curHi, uint32_t& padTop,
                                         uint32_t& padBottom, uint32_t& hiLoadOff);
    __aicore__ inline void LoadFmapL1Chunk(uint32_t bufIdx, uint32_t curHi, uint32_t hiLoadOff, uint32_t padTop,
                                           uint32_t padBottom, uint32_t cinOff, uint32_t curCin);
    __aicore__ inline void LoadWeightL1Block(uint32_t kOff, uint32_t curK);
    __aicore__ inline void SetupLoad3DForChunk(uint32_t curHi, uint32_t mOff, uint32_t curM, uint32_t padTop,
                                               uint32_t padBottom);
    __aicore__ inline uint32_t CalcKL0();

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
          bool isNHWCin, bool isNHWCout>
__aicore__ inline void Conv2dSmallKernelParallelism<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin,
                                                    isNHWCout>::Init(const Conv2DTilingData& tiling)
{
    this->InitCommon(tiling);
    if (!this->coreActive_)
        return;

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

    uint32_t maxHoRelEnd = (this->hoL0_ + this->tiling_->wout - 2) / this->tiling_->wout;
    uint32_t maxHiL1 = maxHoRelEnd * this->tiling_->strideH + this->tiling_->dilationH * (this->tiling_->kh - 1) +
                       this->tiling_->padTop + 1;
    uint32_t hinFull = static_cast<uint32_t>(this->tiling_->hin) + this->tiling_->padTop + this->tiling_->padBottom;
    if (maxHiL1 > hinFull)
        maxHiL1 = hinFull;

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
          bool isNHWCin, bool isNHWCout>
__aicore__ inline void Conv2dSmallKernelParallelism<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin,
                                                    isNHWCout>::CalcChunkFmap(uint32_t mOff, uint32_t curM,
                                                                              uint32_t& curHi, uint32_t& padTop,
                                                                              uint32_t& padBottom, uint32_t& hiLoadOff)
{
    uint32_t hoStart = (this->mIdxStart_ + mOff) / this->tiling_->wout;
    uint32_t hoEnd = (this->mIdxStart_ + mOff + curM - 1) / this->tiling_->wout;
    uint32_t hoRelEnd = hoEnd - hoStart;

    uint32_t hiStart = hoStart * this->tiling_->strideH;
    uint32_t hiEnd = hoEnd * this->tiling_->strideH + this->tiling_->dilationH * (this->tiling_->kh - 1);
    uint32_t hinVal = static_cast<uint32_t>(this->tiling_->hin);

    if (hiStart < this->tiling_->padTop) {
        padTop = this->tiling_->padTop - hiStart;
        hiLoadOff = 0;
    } else {
        padTop = 0;
        hiLoadOff = hiStart - this->tiling_->padTop;
    }

    uint32_t needHi = hoRelEnd * this->tiling_->strideH + this->tiling_->dilationH * (this->tiling_->kh - 1) + padTop +
                      1;
    uint32_t maxGmRows = hinVal - hiLoadOff;
    curHi = (needHi < maxGmRows) ? needHi : maxGmRows;
    padBottom = 0;
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout>
__aicore__ inline void Conv2dSmallKernelParallelism<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin,
                                                    isNHWCout>::LoadFmapL1Chunk(uint32_t bufIdx, uint32_t curHi,
                                                                                uint32_t hiLoadOff, uint32_t padTop,
                                                                                uint32_t padBottom, uint32_t cinOff,
                                                                                uint32_t curCin)
{
    uint32_t loadRows = curHi - padBottom;
    uint32_t elemCount = curHi * this->orgWin_ * curCin;
    uint32_t hinVal = static_cast<uint32_t>(this->tiling_->hin);
    if (hiLoadOff + loadRows > hinVal)
        loadRows = hinVal - hiLoadOff;

    uint32_t bufByteOff = bufIdx * al1BufBytes_;
    LocalTensor<FmapType> al1(TPosition::A1, bufByteOff, elemCount);

    if constexpr (isNHWCin) {
        Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = curHi * this->orgWin_;
        p.dValue = curCin;
        p.srcDValue = static_cast<uint32_t>(this->tiling_->cin);
        p.dstNzNStride = 1;
        p.dstNzC0Stride = curHi * this->orgWin_;

        uint64_t gmOff = static_cast<uint64_t>(hiLoadOff) * this->orgWin_ * this->tiling_->cin + cinOff;
        DataCopy(al1, fmapGm_[gmOff], p);
    } else {
        Dn2NzParams p;
        p.dnNum = 1;
        p.nValue = curHi * this->orgWin_;
        p.dValue = curCin;
        p.srcDnMatrixStride = 0;
        p.srcDValue = static_cast<uint32_t>(this->tiling_->hin * this->tiling_->win);
        p.dstNzC0Stride = curHi * this->orgWin_;
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 0;

        uint64_t gmOff = static_cast<uint64_t>(cinOff) * static_cast<uint64_t>(this->tiling_->hin) *
                             static_cast<uint64_t>(this->tiling_->win) +
                         static_cast<uint64_t>(hiLoadOff) * this->orgWin_;
        DataCopy(al1, fmapGm_[gmOff], p);
    }
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout>
__aicore__ inline void Conv2dSmallKernelParallelism<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin,
                                                    isNHWCout>::LoadWeightL1Block(uint32_t kOff, uint32_t curK)
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
          bool isNHWCin, bool isNHWCout>
__aicore__ inline void Conv2dSmallKernelParallelism<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin,
                                                    isNHWCout>::SetupLoad3DForChunk(uint32_t curHi, uint32_t mOff,
                                                                                    uint32_t curM, uint32_t padTop,
                                                                                    uint32_t padBottom)
{
    uint8_t padList[PAD_LIST_NUM] = {static_cast<uint8_t>(this->tiling_->padLeft),
                                     static_cast<uint8_t>(this->tiling_->padRight), static_cast<uint8_t>(padTop),
                                     static_cast<uint8_t>(PAD_BOTTOM_VALUE)};
    Load3DSetFMatrixCal(curHi, this->orgWin_, padList);
    Load3DSetPaddingCal(this->tiling_->offsetx);

    uint32_t curMAlign = AlignB(curM, GM0);
    uint32_t firstOutCol = (this->mIdxStart_ + mOff) % this->tiling_->wout;

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
          bool isNHWCin, bool isNHWCout>
__aicore__ inline uint32_t
Conv2dSmallKernelParallelism<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout>::CalcKL0()
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
          bool isNHWCin, bool isNHWCout>
__aicore__ inline void Conv2dSmallKernelParallelism<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin,
                                                    isNHWCout>::Process(GM_ADDR x, GM_ADDR filter, GM_ADDR bias,
                                                                        GM_ADDR y, const ExtendParams* extendParams)
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

    uint64_t batchFmapOff = static_cast<uint64_t>(this->batchIdx_) * this->tiling_->hin * this->tiling_->win *
                            this->tiling_->cin;

    fmapGm_.SetGlobalBuffer(reinterpret_cast<__gm__ FmapType*>(x) + batchFmapOff,
                            this->tiling_->cin * this->tiling_->hin * this->tiling_->win);
    filterGm_.SetGlobalBuffer(reinterpret_cast<__gm__ weightType*>(filter),
                              this->k1Total_ * n1Total_ * GN0 * this->GK0);

    if (this->tiling_->hasBias) {
        this->LoadBiasToBT();
    }

    for (uint32_t mOff = 0; mOff < this->actualM_; mOff += this->hoL0_) {
        uint32_t curM = this->hoL0_;
        if (mOff + curM > this->actualM_)
            curM = this->actualM_ - mOff;

        uint32_t curHi, padTop, padBottom, hiLoadOff;
        CalcChunkFmap(mOff, curM, curHi, padTop, padBottom, hiLoadOff);

        uint32_t curMAlign = AlignB(curM, GM0);
        LocalTensor<int32_t> cl0(TPosition::CO1, 0, this->L0C_ELEMS);

        MmadParams mp;
        mp.m = curMAlign;
        mp.n = AlignB(this->actualCo_, GN0);
        bool firstMmad = true;

        SetFlag<HardEvent::MTE1_MTE2>(EVT_FMAP_BUF0);
        SetFlag<HardEvent::MTE1_MTE2>(EVT_FMAP_BUF1);

        for (uint32_t kl1 = 0; kl1 < cinL1Blocks_; kl1++) {
            uint32_t cinOff = kl1 * cinL1_;
            uint32_t curCinOri = cinL1_;
            uint32_t curCin = cinL1_;

            if (cinOff + curCin > this->tiling_->singleCoreCi) {
                curCin = this->cinAligned_ - cinOff;
                curCinOri = this->tiling_->singleCoreCi - cinOff;
            }

            uint32_t kOff = cinOff * kernelHxW;
            uint32_t curKL1 = curCin * kernelHxW;

            uint32_t kl1Buf = kl1 % 2; // 2 pingpong
            event_t kl1Ev = (kl1Buf == 0) ? EVT_FMAP_BUF0 : EVT_FMAP_BUF1;

            WaitFlag<HardEvent::MTE1_MTE2>(kl1Ev);

            LoadFmapL1Chunk(kl1Buf, curHi, hiLoadOff, padTop, padBottom, cinOff, curCinOri);
            if (mOff == 0) {
                LoadWeightL1Block(kOff, curKL1);
            }

            SetFlag<HardEvent::MTE2_MTE1>(kl1Ev);
            WaitFlag<HardEvent::MTE2_MTE1>(kl1Ev);

            SetupLoad3DForChunk(curHi, mOff, curM, padTop, padBottom);

            uint32_t al1ElemCount = curHi * this->orgWin_ * curCin;
            uint32_t al1BufOff = kl1Buf * al1BufBytes_;
            LocalTensor<FmapType> al1(TPosition::A1, al1BufOff, al1ElemCount);

            uint32_t kl0Start = kOff / kL0;
            uint32_t kl0End = CeilDiv(kOff + curKL1, kL0);
            if (kl0End > kL0Iters)
                kl0End = kL0Iters;

            SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(0));
            SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(1));

            for (uint32_t kl0 = kl0Start; kl0 < kl0End; kl0++) {
                uint32_t kOffInner = kl0 * kL0;
                uint32_t curKInner = kL0;
                if (kOffInner + curKInner > kOff + curKL1)
                    curKInner = kOff + curKL1 - kOffInner;

                uint32_t lbuf = kl0 % 2; // 2 pingpong
                event_t lev = static_cast<event_t>(lbuf);

                LocalTensor<FmapType> al0(TPosition::A2, lbuf * AL0_BUF_BYTES, AL0_BUF_ELEMS);
                LocalTensor<weightType> bl0(TPosition::B2, lbuf * BL0_BUF_BYTES, BL0_BUF_ELEMS);

                WaitFlag<HardEvent::M_MTE1>(lev);

                this->DoLoadAL0(al1, al0, kOffInner - kOff, curKInner);
                this->DoLoadBL0(bl0, bl1Full, kOffInner, curKInner);
                SetFlag<HardEvent::MTE1_M>(lev);
                WaitFlag<HardEvent::MTE1_M>(lev);

                if (firstMmad) {
                    mp.cmatrixInitVal = !(this->tiling_->hasBias);
                    mp.cmatrixSource = (this->tiling_->hasBias != 0);
                    firstMmad = false;
                } else {
                    mp.cmatrixInitVal = false;
                    mp.cmatrixSource = false;
                }

                bool isLast = (kl1 == cinL1Blocks_ - 1) && (kl0 == kl0End - 1);
                mp.unitFlag = isLast ? UNIT_FLAG_ENABLE_WITH_FLIP : UNIT_FLAG_ENABLE_ONLY;
                mp.k = curKInner;

                Mmad(cl0, al0, bl0, mp);
                SetFlag<HardEvent::M_MTE1>(lev);
            }

            WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(0));
            WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(1));

            SetFlag<HardEvent::MTE1_MTE2>(kl1Ev);
        }

        WaitFlag<HardEvent::MTE1_MTE2>(EVT_FMAP_BUF0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVT_FMAP_BUF1);

        if constexpr (isNHWCout) {
            this->template DoCopyOut<out0Type, 0, CFG_ROW_MAJOR, CFG_ROW_MAJOR_FIXED_POINT>(y, cl0, mOff, curM,
                                                                                            curMAlign, this->actualCo_);
            if (this->tiling_->dualOutput) {
                this->template DoCopyOut<out1Type, 1, CFG_ROW_MAJOR, CFG_ROW_MAJOR_FIXED_POINT>(
                    extendParams->y1, cl0, mOff, curM, curMAlign, this->actualCo_);
            }
        } else {
            this->template DoCopyOut<out0Type, 0, CFG_COLUMN_MAJOR, CFG_COLUMN_MAJOR_FIXED_POINT>(
                y, cl0, mOff, curM, curMAlign, this->actualCo_);
            if (this->tiling_->dualOutput) {
                this->template DoCopyOut<out1Type, 1, CFG_COLUMN_MAJOR, CFG_COLUMN_MAJOR_FIXED_POINT>(
                    extendParams->y1, cl0, mOff, curM, curMAlign, this->actualCo_);
            }
        }
    }
}

#endif
