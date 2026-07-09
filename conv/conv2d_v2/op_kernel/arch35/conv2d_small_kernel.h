/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV2D_SMALL_KERNEL_H
#define CONV2D_SMALL_KERNEL_H

#include "kernel_operator.h"
#include "conv2d_v2_util.h"

using namespace AscendC;

constexpr uint32_t GN0 = 16;
constexpr uint32_t GM0 = 16;
constexpr uint32_t AL0_BUF_BYTES = L0A_SIZE / 2;
constexpr uint32_t BL0_BUF_BYTES = L0B_SIZE / 2;
constexpr uint32_t AL0_BUF_ELEMS = AL0_BUF_BYTES / sizeof(int8_t);
constexpr uint32_t BL0_BUF_ELEMS = BL0_BUF_BYTES / sizeof(int8_t);
constexpr uint32_t PAD_LIST_NUM = 4;
constexpr uint32_t PAD_BOTTOM_VALUE = 255;
constexpr uint32_t FLOAT_ONE_FIXED_POINT = 1065353216UL; // IEEE 754 representation of float 1.0

static constexpr event_t EVT_MTE2_DONE = static_cast<event_t>(2);

template <typename ChannelWiseT>
__aicore__ inline void LoadChannelWiseL1FullLoad(const LocalTensor<ChannelWiseT>& tensorL1,
                                                 const GlobalTensor<ChannelWiseT>& tensorGm, uint64_t loadNum)
{
    uint64_t byteNum = sizeof(ChannelWiseT);
    InitConstValueParams<ChannelWiseT> initParams(1, static_cast<uint16_t>(AlignB(loadNum, BLOCK_L0_N) * 4 / C0_SIZE),
                                                  0, 0); // 4 for 64 bit align.
    InitConstValue<ChannelWiseT>(tensorL1, initParams);
    PipeBarrier<PIPE_MTE2>();
    DataCopyParams dataCopyParams(1, loadNum * byteNum, 0, 0);
    uint8_t rightPadding = (uint8_t)(AlignB(loadNum * byteNum, PADDING_ALIGN_SIZE) / byteNum - loadNum);
    DataCopyPadParams padParams(true, 0, rightPadding, 0);
    DataCopyPad<ChannelWiseT>(tensorL1, tensorGm, dataCopyParams, padParams);
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type = half,
          bool isNHWCin = false, bool isNHWCout = false, ConvFormat WeightFmt = ConvFormat::FRACTAL_Z>
class Conv2dSmallKernel {
public:
    using FmapT = FmapType;
    using WeightT = weightType;
    using BiasT = biasType;
    using Output0T = out0Type;
    using Output1T = out1Type;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
    using L0cT = int32_t;
#else
    using L0cT = float;
#endif
    const static uint32_t GK0 = C0_SIZE / sizeof(WeightT);

    __aicore__ inline void Init(const Conv2DTilingData& tiling);
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                                   const ExtendParams* extendParams);

protected:
    static constexpr uint32_t L0C_ELEMS = L0C_SIZE / sizeof(L0cT);
    __aicore__ inline void InitCommon(const Conv2DTilingData& tiling);
    __aicore__ inline void LoadFmapL1(GM_ADDR x);
    __aicore__ inline void LoadWeightL1(GM_ADDR filter);
    __aicore__ inline void LoadBiasScaleL1(GM_ADDR bias, const ExtendParams* extendParams);
    __aicore__ inline void LoadBiasToBT();
    __aicore__ inline void SetupLoad3DBase();
    __aicore__ inline void DoLoadAL0(LocalTensor<FmapT>& al1, LocalTensor<FmapT>& al0, uint32_t kOff, uint32_t curK);
    __aicore__ inline void DoLoadBL0(LocalTensor<WeightT>& bl0, LocalTensor<WeightT>& bl1, uint32_t kOff,
                                     uint32_t curK);
    template <typename OutputT, uint64_t FixpipeIdx, const FixpipeConfig& config, const FixpipeConfig& configFp>
    __aicore__ inline void DoCopyOut(GM_ADDR yAddr, LocalTensor<L0cT>& cl0, uint32_t mOff, uint32_t curM,
                                     uint32_t curMAlign, uint32_t actualCo);
    template <typename OutputT, uint64_t FixpipeIdx>
    __aicore__ inline QuantMode_t GetQuantPreInt32();
    template <typename OutputT, uint64_t FixpipeIdx>
    __aicore__ inline QuantMode_t GetQuantPreFp32();

    const Conv2DTilingData* tiling_;
    bool coreActive_;

    uint32_t batchIdx_;
    uint32_t nIdx_;
    uint32_t mIdx_;

    uint32_t mIdxStart_;
    uint32_t actualM_;
    uint32_t ml0_;
    uint32_t totalM_;
    uint32_t hoL0_;

    uint32_t actualCo_; // valid N for this core (Fixpipe nSize)
    uint32_t n1PerCore_;
    uint32_t k1Total_;
    uint32_t kTotal_;

    uint32_t cinAligned_;
    uint32_t orgWin_;

    uint32_t al1ElemCount_;
    uint32_t bl1ElemCount_;

    uint32_t bl1OffBytes_;
    uint32_t biasL1OffBytes_;
    uint32_t scale0L1OffBytes_;
    uint32_t scale1L1OffBytes_;
    uint32_t reluWeight0L1OffBytes_;
    uint32_t reluWeight1L1OffBytes_;

    uint32_t curHiLoadL1_;
    int32_t padTopL1_;
    int32_t padBottomL1_;
    uint32_t hiLoadStart_;

    uint64_t load3dXtBase_;
    uint64_t load3dXmTmp_;

    GlobalTensor<uint64_t> scale0Gm_;
    GlobalTensor<uint64_t> scale1Gm_;
    GlobalTensor<float> reluWeight0Gm_;
    GlobalTensor<float> reluWeight1Gm_;
};

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt>
__aicore__ inline void Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                         WeightFmt>::InitCommon(const Conv2DTilingData& tiling)
{
    tiling_ = &tiling;

    uint32_t blockIdx = GetBlockIdx();
    uint32_t totalBlocks = tiling_->batchDim * tiling_->nDim * tiling_->hoDim;
    if (blockIdx >= totalBlocks) {
        coreActive_ = false;
        return;
    }
    coreActive_ = true;

    // Decompose: batch-first, then N, then M
    uint32_t nxm = tiling_->nDim * tiling_->hoDim;
    batchIdx_ = blockIdx / nxm;
    uint32_t rem = blockIdx % nxm;
    nIdx_ = rem / tiling_->hoDim;
    mIdx_ = rem % tiling_->hoDim;

    cinAligned_ = AlignB(tiling_->singleCoreCi, GK0);
    orgWin_ = static_cast<uint32_t>(tiling_->win);

    // N-tail: last N-partition may have fewer real channels than singleCoreCo.
    {
        uint32_t coutStart = nIdx_ * tiling_->singleCoreCo;
        if (coutStart >= tiling_->cout) {
            actualCo_ = 0;
        } else if (coutStart + tiling_->singleCoreCo > tiling_->cout) {
            actualCo_ = tiling_->cout - coutStart;
        } else {
            actualCo_ = tiling_->singleCoreCo;
        }
    }

    totalM_ = static_cast<uint32_t>(tiling_->hout * tiling_->wout);
    uint32_t singleCoreM = static_cast<uint32_t>(tiling_->singleCoreHo);
    mIdxStart_ = mIdx_ * singleCoreM;
    if (mIdxStart_ >= totalM_) {
        coreActive_ = false;
        return;
    }
    actualM_ = singleCoreM;
    if (mIdxStart_ + actualM_ > totalM_)
        actualM_ = totalM_ - mIdxStart_;
    ml0_ = AlignB(actualM_, GM0);
    hoL0_ = tiling_->hoL0;
    if (hoL0_ == 0)
        hoL0_ = ml0_;

    // K/N dimensions
    kTotal_ = cinAligned_ * tiling_->kh * tiling_->kw;
    n1PerCore_ = CeilDiv(actualCo_, GN0);
    k1Total_ = CeilDiv(kTotal_, GK0);
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt>
__aicore__ inline void Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                         WeightFmt>::Init(const Conv2DTilingData& tiling)
{
    InitCommon(tiling);
    if (!coreActive_)
        return;

    uint32_t orgWo = static_cast<uint32_t>(tiling_->wout);
    uint32_t hoStart = mIdxStart_ / orgWo;
    uint32_t hoEnd = (mIdxStart_ + actualM_ - 1) / orgWo;
    uint32_t hiStart = hoStart * tiling_->strideH;
    uint32_t hiEnd = hoEnd * tiling_->strideH + tiling_->dilationH * (tiling_->kh - 1);
    uint32_t hiTotal = hiEnd - hiStart + 1;

    padTopL1_ = 0;
    padBottomL1_ = 0;
    curHiLoadL1_ = hiTotal;
    hiLoadStart_ = hiStart;

    if (hiStart < tiling_->padTop) {
        padTopL1_ = tiling_->padTop - hiStart;
        curHiLoadL1_ -= padTopL1_;
        hiLoadStart_ = 0;
    } else {
        hiLoadStart_ = hiStart - tiling_->padTop;
    }

    uint32_t hinVal = static_cast<uint32_t>(tiling_->hin);
    if (hiEnd >= hinVal + tiling_->padTop) {
        padBottomL1_ = hiEnd - (hinVal + tiling_->padTop) + 1;
        curHiLoadL1_ -= padBottomL1_;
    }

    al1ElemCount_ = curHiLoadL1_ * orgWin_ * cinAligned_;
    bl1OffBytes_ = al1ElemCount_ * sizeof(FmapT);
    bl1ElemCount_ = k1Total_ * n1PerCore_ * GN0 * GK0;
    uint32_t afterBl1 = bl1OffBytes_ + bl1ElemCount_ * sizeof(WeightT);
    biasL1OffBytes_ = AlignB(afterBl1, ADDR_ALIGN_SIZE);
    uint32_t afterBias = tiling_->hasBias ? biasL1OffBytes_ + tiling_->singleCoreCo * sizeof(L0cT) : biasL1OffBytes_;
    scale0L1OffBytes_ = AlignB(afterBias, ADDR_ALIGN_SIZE);
    uint32_t afterScale0 = tiling_->quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT) ?
                               afterBias + tiling_->singleCoreCo * sizeof(uint64_t) :
                               afterBias;
    reluWeight0L1OffBytes_ = AlignB(afterScale0, ADDR_ALIGN_SIZE);
    uint32_t afterReluWeight0 = tiling_->reluMode0 == static_cast<uint8_t>(ReluMode::VECTOR_RELU) ?
                                    afterScale0 + tiling_->singleCoreCo * sizeof(float) :
                                    afterScale0;
    scale1L1OffBytes_ = AlignB(afterReluWeight0, ADDR_ALIGN_SIZE);
    uint32_t afterScale1 = tiling_->quantMode1 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT) ?
                               afterReluWeight0 + tiling_->singleCoreCo * sizeof(uint64_t) :
                               afterReluWeight0;
    reluWeight1L1OffBytes_ = AlignB(afterScale1, ADDR_ALIGN_SIZE);
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt>
__aicore__ inline void Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                         WeightFmt>::LoadFmapL1(GM_ADDR x)
{
    LocalTensor<FmapT> al1(TPosition::A1, 0, al1ElemCount_);
    GlobalTensor<FmapT> fmapGm;
    if constexpr (isNHWCin) {
        uint64_t batchFmapOff = static_cast<uint64_t>(batchIdx_) * tiling_->hin * tiling_->win * tiling_->cin;
        fmapGm.SetGlobalBuffer(reinterpret_cast<__gm__ FmapT*>(x) + batchFmapOff +
                                   static_cast<uint64_t>(hiLoadStart_) * orgWin_ * tiling_->cin,
                               tiling_->hin * tiling_->win * tiling_->cin);

        Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = curHiLoadL1_ * orgWin_;
        p.dValue = static_cast<uint32_t>(tiling_->cin);
        p.srcDValue = static_cast<uint32_t>(tiling_->cin);
        p.dstNzNStride = 1;
        p.dstNzC0Stride = curHiLoadL1_ * orgWin_;

        DataCopy(al1, fmapGm, p);
    } else {
        uint64_t batchFmapOff = static_cast<uint64_t>(batchIdx_) * tiling_->cin * tiling_->hin * tiling_->win;
        fmapGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ FmapT*>(x) + batchFmapOff + static_cast<uint64_t>(hiLoadStart_) * orgWin_,
            tiling_->cin * tiling_->hin * tiling_->win);

        Dn2NzParams p;
        p.dnNum = 1;
        p.nValue = curHiLoadL1_ * orgWin_;
        p.dValue = static_cast<uint32_t>(tiling_->cin);
        p.srcDnMatrixStride = 0;
        p.srcDValue = static_cast<uint32_t>(tiling_->hin * tiling_->win);
        p.dstNzC0Stride = curHiLoadL1_ * orgWin_;
        p.dstNzNStride = 1;
        p.dstNzMatrixStride = 0;

        DataCopy(al1, fmapGm, p);
    }
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt>
__aicore__ inline void Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                         WeightFmt>::LoadWeightL1(GM_ADDR filter)
{
    uint32_t n1Total = AlignB(tiling_->cout, GN0) / GN0;
    GlobalTensor<WeightT> filterGm;
    filterGm.SetGlobalBuffer(reinterpret_cast<__gm__ WeightT*>(filter), k1Total_ * n1Total * GN0 * GK0);
    LocalTensor<WeightT> bl1(TPosition::B1, bl1OffBytes_, bl1ElemCount_);

    if constexpr (WeightFmt == ConvFormat::NCHW) {
        uint32_t khkw = tiling_->kh * tiling_->kw;
        uint64_t gmOff = static_cast<uint64_t>(nIdx_) * tiling_->singleCoreCo * tiling_->singleCoreCi * khkw;
        if (khkw == 1) {
            // 1x1 kernel: Nd2NzParams
            Nd2NzParams p;
            p.ndNum = 1;
            p.nValue = actualCo_;
            p.dValue = tiling_->singleCoreCi;
            p.srcNdMatrixStride = 0;
            p.srcDValue = tiling_->singleCoreCi;
            p.dstNzC0Stride = n1PerCore_ * GN0;
            p.dstNzNStride = 1;
            p.dstNzMatrixStride = 0;
            DataCopy(bl1, filterGm[gmOff], p);
        } else {
            // General kernel: Dn2NzParams (matching LoadBL1Tools)
            Dn2NzParams p;
            p.dnNum = actualCo_;
            p.nValue = khkw;
            p.dValue = tiling_->singleCoreCi;
            p.srcDnMatrixStride = tiling_->coutOffsetBlock;
            p.srcDValue = khkw;
            p.dstNzC0Stride = khkw * n1PerCore_ * GN0;
            p.dstNzNStride = n1PerCore_ * GN0;
            p.dstNzMatrixStride = GK0;
            DataCopy(bl1, filterGm[gmOff], p);
        }
    } else {
        // FRACTAL_Z weight: direct copy
        if (tiling_->nDim == 1) {
            DataCopy(bl1, filterGm[0], bl1ElemCount_);
        } else {
            uint32_t n1Start = nIdx_ * tiling_->singleCoreCo / GN0;
            uint32_t tileBytes = GN0 * GK0 * sizeof(WeightT);
            uint32_t srcGmOff = n1Start * GN0 * GK0;
            uint16_t blkLen = static_cast<uint16_t>((n1PerCore_ * tileBytes) / 32);
            uint32_t srcGap = static_cast<uint16_t>(((n1Total - n1PerCore_) * tileBytes) / 32);
            DataCopyParams cp(static_cast<uint16_t>(k1Total_), blkLen, srcGap, 0);
            DataCopy(bl1, filterGm[srcGmOff], cp);
        }
    }
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt>
__aicore__ inline void Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                         WeightFmt>::LoadBiasScaleL1(GM_ADDR bias, const ExtendParams* extendParams)
{
    uint32_t bsOff = nIdx_ * tiling_->singleCoreCo;
    GlobalTensor<BiasT> biasGm;
    biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT*>(bias) + bsOff, actualCo_);
    if (tiling_->hasBias && actualCo_ > 0) {
        LocalTensor<BiasT> biasL1(TPosition::A1, biasL1OffBytes_, tiling_->singleCoreCo);
        LoadChannelWiseL1FullLoad<BiasT>(biasL1, biasGm[0], actualCo_);
    }
    {
        if (tiling_->quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
            scale0Gm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ uint64_t*>(extendParams->scale0) + nIdx_ * tiling_->singleCoreCo,
                tiling_->singleCoreCo);
            LocalTensor<uint64_t> scale0L1(TPosition::A1, scale0L1OffBytes_, tiling_->singleCoreCo);
            LoadChannelWiseL1FullLoad<uint64_t>(scale0L1, scale0Gm_[0], actualCo_);
        } else if (tiling_->quantMode0 == static_cast<uint8_t>(QuantModeType::SCALAR_QUANT)) {
            scale0Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(extendParams->scale0));
        }
        if (tiling_->reluMode0 == static_cast<uint8_t>(ReluMode::VECTOR_RELU)) {
            reluWeight0Gm_.SetGlobalBuffer(
                reinterpret_cast<__gm__ float*>(extendParams->reluWeight0) + nIdx_ * tiling_->singleCoreCo,
                tiling_->singleCoreCo);
            LocalTensor<float> reluWeight0L1(TPosition::A1, reluWeight0L1OffBytes_, tiling_->singleCoreCo);
            LoadChannelWiseL1FullLoad<float>(reluWeight0L1, reluWeight0Gm_[0], actualCo_);
        } else if (tiling_->reluMode0 == static_cast<uint8_t>(ReluMode::SCALAR_RELU)) {
            reluWeight0Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(extendParams->reluWeight0));
        }

        if (tiling_->dualOutput) {
            if (tiling_->quantMode1 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                scale1Gm_.SetGlobalBuffer(
                    reinterpret_cast<__gm__ uint64_t*>(extendParams->scale1) + nIdx_ * tiling_->singleCoreCo,
                    tiling_->singleCoreCo);
                LocalTensor<uint64_t> scale1L1(TPosition::A1, scale1L1OffBytes_, tiling_->singleCoreCo);
                LoadChannelWiseL1FullLoad<uint64_t>(scale1L1, scale1Gm_[0], actualCo_);
            } else if (tiling_->quantMode1 == static_cast<uint8_t>(QuantModeType::SCALAR_QUANT)) {
                scale1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(extendParams->scale1));
            }
            if (tiling_->reluMode1 == static_cast<uint8_t>(ReluMode::VECTOR_RELU)) {
                reluWeight1Gm_.SetGlobalBuffer(
                    reinterpret_cast<__gm__ float*>(extendParams->reluWeight1) + nIdx_ * tiling_->singleCoreCo,
                    tiling_->singleCoreCo);
                LocalTensor<float> reluWeight1L1(TPosition::A1, reluWeight1L1OffBytes_, tiling_->singleCoreCo);
                LoadChannelWiseL1FullLoad<float>(reluWeight1L1, reluWeight1Gm_[0], actualCo_);
            } else if (tiling_->reluMode1 == static_cast<uint8_t>(ReluMode::SCALAR_RELU)) {
                reluWeight1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(extendParams->reluWeight1));
            }
        }
    }
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt>
__aicore__ inline void
Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout, WeightFmt>::LoadBiasToBT()
{
    constexpr uint32_t BT_ALIGN = 64;
    uint32_t btElemNum = AlignB(tiling_->singleCoreCo * sizeof(L0cT), BT_ALIGN) / sizeof(L0cT);
    LocalTensor<L0cT> biasBT(TPosition::C2, 0, btElemNum);
    LocalTensor<BiasT> biasL1src(TPosition::A1, biasL1OffBytes_, tiling_->singleCoreCo);
    uint32_t blkCnt = AlignB(actualCo_ * sizeof(BiasT), BT_ALIGN) / 32;
    DataCopyParams cp(1, static_cast<uint16_t>(blkCnt), 0, 0);
    DataCopy(biasBT, biasL1src[0], cp);
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt>
__aicore__ inline void
Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout, WeightFmt>::SetupLoad3DBase()
{
    uint8_t padList[PAD_LIST_NUM] = {static_cast<uint8_t>(tiling_->padLeft), static_cast<uint8_t>(tiling_->padRight),
                                     static_cast<uint8_t>(padTopL1_), static_cast<uint8_t>(PAD_BOTTOM_VALUE)};
    Load3DSetFMatrixCal(curHiLoadL1_, orgWin_, padList);
    Load3DSetPaddingCal(tiling_->offsetx);

    load3dXtBase_ = ((static_cast<uint64_t>(tiling_->strideW) & MASK_6) << 0) |
                    ((static_cast<uint64_t>(tiling_->kw) & MASK_8) << KERNELW_OFFSET) |
                    ((static_cast<uint64_t>(tiling_->kh) & MASK_8) << KERNELH_OFFSET) |
                    ((static_cast<uint64_t>(tiling_->strideH) & MASK_6) << STRIDEH_OFFSET) |
                    ((static_cast<uint64_t>(tiling_->kw) & NINTH_BIT_MASK) << KERNELW_HIGHEST_BIT_OFFSET) |
                    ((static_cast<uint64_t>(tiling_->kh) & NINTH_BIT_MASK) << KERNELH_HIGHEST_BIT_OFFSET) |
                    ((static_cast<uint64_t>(tiling_->dilationW) & MASK_8) << DILATIONW_OFFSET) |
                    ((static_cast<uint64_t>(tiling_->dilationH) & MASK_8) << DILATIONH_OFFSET) |
                    ((static_cast<uint64_t>(cinAligned_) & MASK_16) << CIN_OFFSET);

#if defined(ASC_DEVKIT_VERSION_NUM) && (ASC_DEVKIT_VERSION_NUM >= 90000000)
    LoadDataRepeatParamWithStride repeatParams(0, 1, 0, static_cast<uint16_t>(ml0_ / GM0));
    SetLoadDataRepeatWithStride(repeatParams);
#else
    LoadDataRepeatParam repeatParams(0, 1, 0, static_cast<uint16_t>(ml0_ / GM0));
    SetLoadDataRepeat(repeatParams);
#endif

    uint32_t posM = mIdxStart_ % static_cast<uint32_t>(tiling_->wout);
    load3dXmTmp_ = ((static_cast<uint64_t>(ml0_) & MASK_16) << MSTEP_OFFSET) |
                   ((static_cast<uint64_t>(posM) & MASK_16) << POSM_OFFSET);
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt>
__aicore__ inline void Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                         WeightFmt>::DoLoadAL0(LocalTensor<FmapT>& al1, LocalTensor<FmapT>& al0,
                                                               uint32_t kOff, uint32_t curK)
{
    uint64_t posK = kOff;
    uint64_t xm = static_cast<uint64_t>(curK) | (posK << POSK_OFFSET) | load3dXmTmp_;
    Load3DBitModeParam param;
    param.SetConfig0(xm);
    param.SetConfig1(load3dXtBase_);
    LoadData<TPosition::A2, TPosition::A1, FmapT>(al0, al1, param);
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt>
__aicore__ inline void Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                         WeightFmt>::DoLoadBL0(LocalTensor<WeightT>& bl0, LocalTensor<WeightT>& bl1,
                                                               uint32_t kOff, uint32_t curK)
{
    uint32_t kStep = CeilDiv(curK, GK0);
    uint32_t mStep = n1PerCore_;
    Load2DBitModeParam param;
    param.SetMStartPosition(0);
    param.SetKStartPosition(kOff / GK0);
    param.SetMStep(static_cast<uint16_t>(mStep));
    param.SetKStep(static_cast<uint16_t>(kStep));
    param.SetSrcStride(static_cast<int32_t>(n1PerCore_));
    param.SetDstStride(static_cast<uint16_t>(mStep));
    param.SetIfTranspose(false);
    LoadData<TPosition::B2, TPosition::B1, WeightT>(bl0, bl1, param);
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt>
template <typename OutputT, uint64_t FixpipeIdx>
__aicore__ inline QuantMode_t Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                                WeightFmt>::GetQuantPreInt32()
{
    // l0c (int32) -> ddr(fp16/int8) — for NPU_ARCH 5102
    if constexpr (AscendC::IsSameType<OutputT, half>::value) {
        if constexpr (AscendC::IsSameType<WeightT, int8_t>::value) {
            uint8_t quantMode = (FixpipeIdx == 0) ? tiling_->quantMode0 : tiling_->quantMode1;
            bool isVector = (quantMode == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT));
            return isVector ? QuantMode_t::VDEQF16 : QuantMode_t::DEQF16;
        } else if constexpr (AscendC::IsSameType<WeightT, half>::value) {
            return QuantMode_t::DEQF16;
        }
    } else if constexpr (AscendC::IsSameType<OutputT, int8_t>::value) {
        if constexpr (AscendC::IsSameType<WeightT, int8_t>::value) {
            uint8_t quantMode = (FixpipeIdx == 0) ? tiling_->quantMode0 : tiling_->quantMode1;
            bool isVector = (quantMode == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT));
            return isVector ? QuantMode_t::VREQ8 : QuantMode_t::REQ8;
        } else if constexpr (AscendC::IsSameType<WeightT, half>::value) {
            return QuantMode_t::REQ8;
        }
    }
    return QuantMode_t::DEQF16;
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt>
template <typename OutputT, uint64_t FixpipeIdx>
__aicore__ inline QuantMode_t
Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout, WeightFmt>::GetQuantPreFp32()
{
    // l0c (float) -> ddr(fp16/bf16/fp32) — for IsWeightND (NCHW weight)
    if constexpr (AscendC::IsSameType<OutputT, float>::value) {
        return QuantMode_t::NoQuant;
    } else if constexpr (AscendC::IsSameType<OutputT, bfloat16_t>::value) {
        return QuantMode_t::F322BF16;
    }
    return QuantMode_t::F322F16;
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt>
template <typename OutputT, uint64_t FixpipeIdx, const FixpipeConfig& config, const FixpipeConfig& configFp>
__aicore__ inline void Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                         WeightFmt>::DoCopyOut(GM_ADDR yAddr, LocalTensor<L0cT>& cl0, uint32_t mOff,
                                                               uint32_t curM, uint32_t curMAlign, uint32_t actualCo)
{
    constexpr bool IsOutput0 = (FixpipeIdx == 0);
    uint8_t reluMode;
    uint8_t quantMode;
    uint32_t scaleL1OffBytes;
    uint32_t reluWeightL1OffBytes;
    uint32_t unitFlag;
    if constexpr (IsOutput0) {
        reluMode = tiling_->reluMode0;
        quantMode = tiling_->quantMode0;
        scaleL1OffBytes = scale0L1OffBytes_;
        reluWeightL1OffBytes = reluWeight0L1OffBytes_;
        unitFlag = tiling_->dualOutput == 1 ? UNIT_FLAG_ENABLE_ONLY : UNIT_FLAG_ENABLE_WITH_FLIP;
    } else {
        reluMode = tiling_->reluMode1;
        quantMode = tiling_->quantMode1;
        scaleL1OffBytes = scale1L1OffBytes_;
        reluWeightL1OffBytes = reluWeight1L1OffBytes_;
        unitFlag = UNIT_FLAG_ENABLE_WITH_FLIP;
    }

    constexpr CO2Layout Layout = isNHWCout ? CO2Layout::ROW_MAJOR : CO2Layout::COLUMN_MAJOR;
    uint64_t hwOut = static_cast<uint64_t>(tiling_->hout) * tiling_->wout;
    uint64_t batchOutOff = static_cast<uint64_t>(batchIdx_) * hwOut * tiling_->cout;
    uint64_t nOutOff;
    uint32_t dstStride;
    uint64_t outOff;
    if constexpr (isNHWCout) {
        nOutOff = static_cast<uint64_t>(nIdx_) * tiling_->singleCoreCo;
        dstStride = static_cast<uint32_t>(tiling_->cout);
        outOff = static_cast<uint64_t>(mIdxStart_ + mOff) * tiling_->cout;
    } else {
        nOutOff = static_cast<uint64_t>(nIdx_) * tiling_->singleCoreCo * hwOut;
        dstStride = static_cast<uint32_t>(hwOut);
        outOff = static_cast<uint64_t>(mIdxStart_ + mOff);
    }

    GlobalTensor<OutputT> outputGm;
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutputT*>(yAddr) + batchOutOff + nOutOff);

    FixpipeParamsC310<Layout> fp;
    fp.nSize = actualCo;
    fp.mSize = curM;
    fp.srcStride = curMAlign;
    fp.dstStride = dstStride;
    if constexpr (AscendC::IsSameType<L0cT, float>::value) {
        fp.quantPre = GetQuantPreFp32<OutputT, FixpipeIdx>();
    } else {
        fp.quantPre = GetQuantPreInt32<OutputT, FixpipeIdx>();
    }
    fp.unitFlag = unitFlag;
    fp.reluEn = reluMode != 0;

    // fp.params: different field names per layout, must be if-constexpr
    if constexpr (isNHWCout) {
        fp.params.ndNum = 1;
        fp.params.dstNdStride = 0;
        fp.params.srcNdStride = 0;
    } else {
        fp.params.dnNum = 1;
        fp.params.srcNzMatrixStride = curMAlign * CeilDiv(actualCo, GM0);
        fp.params.dstDnMatrixStride = hwOut;
        fp.params.srcNzC0Stride = 1;
    }

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
    fp.preReluMode = static_cast<ReluMode>(reluMode);
    if (reluMode == static_cast<uint8_t>(ReluMode::SCALAR_RELU)) {
        float m2 = IsOutput0 ? reluWeight0Gm_.GetValue(0) : reluWeight1Gm_.GetValue(0);
        fp.reluScalar = reinterpret_cast<uint64_t&>(m2);
    } else if (reluMode == static_cast<uint8_t>(ReluMode::VECTOR_RELU)) {
        LocalTensor<float> reluWeightL1(TPosition::A1, reluWeightL1OffBytes, tiling_->singleCoreCo);
        fp.vectorRelu = reluWeightL1.GetPhyAddr();
    }

    if (quantMode == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
        LocalTensor<uint64_t> scaleL1(TPosition::A1, scaleL1OffBytes, tiling_->singleCoreCo);
        Fixpipe<OutputT, L0cT, config>(outputGm[outOff], cl0, scaleL1, fp);
    } else if (quantMode == static_cast<uint8_t>(QuantModeType::SCALAR_QUANT)) {
        fp.deqScalar = IsOutput0 ? scale0Gm_.GetValue(0) : scale1Gm_.GetValue(0);
        if constexpr (AscendC::IsSameType<WeightT, half>::value) {
            Fixpipe<OutputT, L0cT, configFp>(outputGm[outOff], cl0, fp);
        } else {
            Fixpipe<OutputT, L0cT, config>(outputGm[outOff], cl0, fp);
        }
    } else {
        fp.deqScalar = FLOAT_ONE_FIXED_POINT;
        Fixpipe<OutputT, L0cT, configFp>(outputGm[outOff], cl0, fp);
    }
#else
    fp.deqScalar = FLOAT_ONE_FIXED_POINT;
    Fixpipe<OutputT, L0cT, config>(outputGm[outOff], cl0, fp);
#endif
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt>
__aicore__ inline void Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                         WeightFmt>::Process(GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR y,
                                                             const ExtendParams* extendParams)
{
    if (!coreActive_ || actualCo_ == 0)
        return;

    // Stage 1: Load everything to L1 (fullload).
    LoadFmapL1(x);
    LoadWeightL1(filter);
    LoadBiasScaleL1(bias, extendParams);

    SetFlag<HardEvent::MTE2_FIX>(static_cast<event_t>(0));
    WaitFlag<HardEvent::MTE2_FIX>(static_cast<event_t>(0));

    SetFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_DONE);
    WaitFlag<HardEvent::MTE2_MTE1>(EVT_MTE2_DONE);

    // Stage 2: Setup Load3D invariant + bias->BT once per core.
    if (tiling_->hasBias) {
        LoadBiasToBT();
    }
    SetupLoad3DBase();

    uint32_t kL0 = tiling_->kL0;
    uint32_t kL0MaxIter = CeilDiv(kTotal_, kL0);
    LocalTensor<FmapT> al1(TPosition::A1, 0, al1ElemCount_);
    LocalTensor<WeightT> bl1(TPosition::B1, bl1OffBytes_, bl1ElemCount_);

    // Stage 3: M-loop -> K-loop -> Fixpipe.
    for (uint32_t mOff = 0; mOff < actualM_; mOff += hoL0_) {
        uint32_t curM = hoL0_;
        if (mOff + curM > actualM_)
            curM = actualM_ - mOff;
        uint32_t curMAlign = AlignB(curM, GM0);
        uint32_t posM = mOff + (mIdxStart_ % static_cast<uint32_t>(tiling_->wout));

        load3dXmTmp_ = ((static_cast<uint64_t>(curMAlign) & MASK_16) << MSTEP_OFFSET) |
                       ((static_cast<uint64_t>(posM) & MASK_16) << POSM_OFFSET);
#if defined(ASC_DEVKIT_VERSION_NUM) && (ASC_DEVKIT_VERSION_NUM >= 90000000)
        SetLoadDataRepeatWithStride(LoadDataRepeatParamWithStride(0, 1, 0, static_cast<uint16_t>(curMAlign / GM0)));
#else
        SetLoadDataRepeat(LoadDataRepeatParam(0, 1, 0, static_cast<uint16_t>(curMAlign / GM0)));
#endif

        LocalTensor<L0cT> cl0(TPosition::CO1, 0, L0C_ELEMS);

        MmadParams mp;
        mp.m = curMAlign;
        mp.n = AlignB(actualCo_, GN0);
        bool firstMmad = true;

        SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(0));
        SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(1));

        for (uint32_t kl0Iter = 0; kl0Iter < kL0MaxIter; kl0Iter++) {
            uint32_t kOff = kl0Iter * kL0;
            uint32_t curK = kL0;
            if (kOff + curK > kTotal_)
                curK = kTotal_ - kOff;
            uint32_t buf = kl0Iter & 1;
            event_t ev = static_cast<event_t>(buf);

            LocalTensor<FmapT> al0(TPosition::A2, buf * AL0_BUF_BYTES, AL0_BUF_ELEMS);
            LocalTensor<WeightT> bl0(TPosition::B2, buf * BL0_BUF_BYTES, BL0_BUF_ELEMS);

            WaitFlag<HardEvent::M_MTE1>(ev);

            DoLoadAL0(al1, al0, kOff, curK);
            DoLoadBL0(bl0, bl1, kOff, curK);
            SetFlag<HardEvent::MTE1_M>(ev);
            WaitFlag<HardEvent::MTE1_M>(ev);

            mp.k = curK;
            mp.unitFlag = (kl0Iter == kL0MaxIter - 1) ? UNIT_FLAG_ENABLE_WITH_FLIP : UNIT_FLAG_ENABLE_ONLY;
            if (firstMmad) {
                mp.cmatrixInitVal = !(tiling_->hasBias);
                mp.cmatrixSource = (tiling_->hasBias != 0);
                firstMmad = false;
            } else {
                mp.cmatrixInitVal = false;
                mp.cmatrixSource = false;
            }

            Mmad(cl0, al0, bl0, mp);
            SetFlag<HardEvent::M_MTE1>(ev);
        }

        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(0));
        WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(1));

        if constexpr (isNHWCout) {
            DoCopyOut<Output0T, 0, CFG_ROW_MAJOR, CFG_ROW_MAJOR_FIXED_POINT>(y, cl0, mOff, curM, curMAlign, actualCo_);
            if (tiling_->dualOutput) {
                DoCopyOut<Output1T, 1, CFG_ROW_MAJOR, CFG_ROW_MAJOR_FIXED_POINT>(extendParams->y1, cl0, mOff, curM,
                                                                                 curMAlign, actualCo_);
            }
        } else {
            DoCopyOut<Output0T, 0, CFG_COLUMN_MAJOR, CFG_COLUMN_MAJOR_FIXED_POINT>(y, cl0, mOff, curM, curMAlign,
                                                                                   actualCo_);
            if (tiling_->dualOutput) {
                DoCopyOut<Output1T, 1, CFG_COLUMN_MAJOR, CFG_COLUMN_MAJOR_FIXED_POINT>(extendParams->y1, cl0, mOff,
                                                                                       curM, curMAlign, actualCo_);
            }
        }
    }
}
#endif
