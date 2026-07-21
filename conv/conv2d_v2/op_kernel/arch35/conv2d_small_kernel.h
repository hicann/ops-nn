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
          bool isNHWCin = false, bool isNHWCout = false, ConvFormat WeightFmt = ConvFormat::FRACTAL_Z,
          bool IsHwMode = false>
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
    __aicore__ inline void LoadFmapL1HwMode(LocalTensor<FmapT>& al1, GM_ADDR x);
    __aicore__ inline void LoadFmapL1MMode(LocalTensor<FmapT>& al1, GM_ADDR x);
    __aicore__ inline void LoadWeightL1(GM_ADDR filter);
    __aicore__ inline void LoadBiasScaleL1(GM_ADDR bias, const ExtendParams* extendParams);
    __aicore__ inline void LoadBiasToBT();
    __aicore__ inline void SetupLoad3DBase();
    __aicore__ inline void ComputeHiPadRange(uint32_t hiStart, uint32_t hiEnd);
    __aicore__ inline void MmadAccumulateTile(LocalTensor<FmapT>& al1, LocalTensor<WeightT>& bl1,
                                              LocalTensor<L0cT>& cl0, uint32_t curMAlign, uint32_t mmadN,
                                              uint32_t kL0MaxIter);
    __aicore__ inline void CopyOutResult(LocalTensor<L0cT>& cl0, GM_ADDR y, const ExtendParams* extendParams,
                                         uint32_t outOff, uint32_t fpMSize, uint32_t curMAlign, uint32_t fpDnNum,
                                         uint32_t fpDstDnStride);
    __aicore__ inline void ProcessHwMode(LocalTensor<FmapT>& al1, LocalTensor<WeightT>& bl1, uint32_t mmadN,
                                         uint32_t kL0MaxIter, uint64_t hwOut, GM_ADDR y,
                                         const ExtendParams* extendParams);
    __aicore__ inline void ProcessMMode(LocalTensor<FmapT>& al1, LocalTensor<WeightT>& bl1, uint32_t mmadN,
                                        uint32_t kL0MaxIter, uint64_t hwOut, GM_ADDR y,
                                        const ExtendParams* extendParams);
    __aicore__ inline void DoLoadAL0(LocalTensor<FmapT>& al1, LocalTensor<FmapT>& al0, uint32_t kOff, uint32_t curK);
    __aicore__ inline void DoLoadBL0(LocalTensor<WeightT>& bl0, LocalTensor<WeightT>& bl1, uint32_t kOff,
                                     uint32_t curK);
    template <typename OutputT, uint64_t FixpipeIdx, const FixpipeConfig& config, const FixpipeConfig& configFp>
    __aicore__ inline void DoCopyOut(GM_ADDR yAddr, LocalTensor<L0cT>& cl0, uint32_t outOffset, uint32_t curM,
                                     uint32_t curMAlign, uint32_t actualCo, uint32_t fpDnNum, uint32_t fpDstDnStride);
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

    // HW-mode (Ho/Wo split) specific
    uint32_t hoIdx_;
    uint32_t woIdx_;
    uint32_t hoIdxStart_;
    uint32_t woIdxStart_;
    uint32_t actualHo_;
    uint32_t actualWo_;
    uint32_t singleCoreWo_;
    uint32_t woL0_;
    int32_t padLeftL1_;
    int32_t padRightL1_;
    uint32_t curWiLoadL1_;
    uint32_t wiLoadStart_;

    uint64_t load3dXtBase_;
    uint64_t load3dXmTmp_;

    GlobalTensor<uint64_t> scale0Gm_;
    GlobalTensor<uint64_t> scale1Gm_;
    GlobalTensor<float> reluWeight0Gm_;
    GlobalTensor<float> reluWeight1Gm_;
};

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                         WeightFmt, IsHwMode>::InitCommon(const Conv2DTilingData& tiling)
{
    tiling_ = &tiling;

    uint32_t blockIdx = GetBlockIdx();
    if constexpr (IsHwMode) {
        uint32_t totalBlocks = tiling_->batchDim * tiling_->nDim * tiling_->hoDim * tiling_->woDim;
        if (blockIdx >= totalBlocks) {
            coreActive_ = false;
            return;
        }
        coreActive_ = true;

        // Decompose: batch-first, then N, then Ho, then Wo
        uint32_t nxhxw = tiling_->nDim * tiling_->hoDim * tiling_->woDim;
        batchIdx_ = blockIdx / nxhxw;
        uint32_t rem = blockIdx % nxhxw;
        nIdx_ = rem / (tiling_->hoDim * tiling_->woDim);
        rem = rem % (tiling_->hoDim * tiling_->woDim);
        hoIdx_ = rem / tiling_->woDim;
        woIdx_ = rem % tiling_->woDim;
    } else {
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
    }

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

    if constexpr (IsHwMode) {
        hoIdxStart_ = hoIdx_ * static_cast<uint32_t>(tiling_->singleCoreHo);
        if (hoIdxStart_ >= tiling_->hout) {
            coreActive_ = false;
            return;
        }
        actualHo_ = static_cast<uint32_t>(tiling_->singleCoreHo);
        if (hoIdxStart_ + actualHo_ > tiling_->hout) {
            actualHo_ = tiling_->hout - hoIdxStart_;
        }

        singleCoreWo_ = static_cast<uint32_t>(tiling_->singleCoreWo);
        woIdxStart_ = woIdx_ * singleCoreWo_;
        if (woIdxStart_ >= tiling_->wout) {
            coreActive_ = false;
            return;
        }
        actualWo_ = singleCoreWo_;
        if (woIdxStart_ + actualWo_ > tiling_->wout) {
            actualWo_ = tiling_->wout - woIdxStart_;
        }

        totalM_ = actualHo_ * actualWo_;
        actualM_ = totalM_;
        ml0_ = AlignB(actualM_, GM0);
        hoL0_ = tiling_->hoL0;
        woL0_ = tiling_->woL0;
    } else {
        totalM_ = static_cast<uint32_t>(tiling_->hout * tiling_->wout);
        mIdxStart_ = mIdx_ * static_cast<uint32_t>(tiling_->singleCoreHo);
        if (mIdxStart_ >= totalM_) {
            coreActive_ = false;
            return;
        }
        actualM_ = static_cast<uint32_t>(tiling_->singleCoreHo);
        if (mIdxStart_ + actualM_ > totalM_) {
            actualM_ = totalM_ - mIdxStart_;
        }
        ml0_ = AlignB(actualM_, GM0);
        hoL0_ = tiling_->hoL0;
    }

    // K/N dimensions
    kTotal_ = cinAligned_ * tiling_->kh * tiling_->kw;
    n1PerCore_ = CeilDiv(actualCo_, GN0);
    k1Total_ = CeilDiv(kTotal_, GK0);
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                         WeightFmt, IsHwMode>::ComputeHiPadRange(uint32_t hiStart, uint32_t hiEnd)
{
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

    if (hiEnd >= static_cast<uint32_t>(tiling_->hin) + tiling_->padTop) {
        padBottomL1_ = hiEnd - (static_cast<uint32_t>(tiling_->hin) + tiling_->padTop) + 1;
        curHiLoadL1_ -= padBottomL1_;
    }
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                         WeightFmt, IsHwMode>::MmadAccumulateTile(LocalTensor<FmapT>& al1,
                                                                                  LocalTensor<WeightT>& bl1,
                                                                                  LocalTensor<L0cT>& cl0,
                                                                                  uint32_t curMAlign, uint32_t mmadN,
                                                                                  uint32_t kL0MaxIter)
{
    MmadParams mp;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
    if constexpr (AscendC::IsSameType<FmapType, half>::value) {
        mp.fixShiftVal = tiling_->fixedShiftValue;
    }
#endif
    mp.m = curMAlign;
    mp.n = mmadN;
    mp.cmatrixInitVal = !(tiling_->hasBias);
    mp.cmatrixSource = (tiling_->hasBias != 0);

    SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(0));
    SetFlag<HardEvent::M_MTE1>(static_cast<event_t>(1));

    for (uint32_t kl0Iter = 0; kl0Iter < kL0MaxIter; kl0Iter++) {
        uint32_t kOff = kl0Iter * tiling_->kL0;
        uint32_t curK = tiling_->kL0;
        if (kOff + curK > kTotal_) {
            curK = kTotal_ - kOff;
        }
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

        Mmad(cl0, al0, bl0, mp);

        mp.cmatrixInitVal = false;
        mp.cmatrixSource = false;
        SetFlag<HardEvent::M_MTE1>(ev);
    }

    WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(0));
    WaitFlag<HardEvent::M_MTE1>(static_cast<event_t>(1));
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                         WeightFmt, IsHwMode>::Init(const Conv2DTilingData& tiling)
{
    InitCommon(tiling);
    if (!coreActive_) {
        return;
    }

    if constexpr (IsHwMode) {
        // HW-mode: fmap H range from this core's Ho range.
        uint32_t hoStart = hoIdxStart_;
        uint32_t hoEnd = hoIdxStart_ + actualHo_ - 1;
        uint32_t hiStart = hoStart * tiling_->strideH;
        uint32_t hiEnd = hoEnd * tiling_->strideH + tiling_->dilationH * (tiling_->kh - 1);
        ComputeHiPadRange(hiStart, hiEnd);

        // HW-mode: fmap W range from this core's Wo range.
        uint32_t woStart = woIdxStart_;
        uint32_t woEnd = woIdxStart_ + actualWo_ - 1;
        uint32_t wiStart = woStart * tiling_->strideW;
        uint32_t wiEnd = woEnd * tiling_->strideW + tiling_->dilationW * (tiling_->kw - 1);
        uint32_t wiTotal = wiEnd - wiStart + 1;

        padLeftL1_ = 0;
        padRightL1_ = 0;
        curWiLoadL1_ = wiTotal;
        wiLoadStart_ = wiStart;
        if (wiStart < tiling_->padLeft) {
            padLeftL1_ = tiling_->padLeft - wiStart;
            curWiLoadL1_ -= padLeftL1_;
            wiLoadStart_ = 0;
        } else {
            wiLoadStart_ = wiStart - tiling_->padLeft;
        }
        if (wiEnd >= static_cast<uint32_t>(tiling_->win) + tiling_->padLeft) {
            padRightL1_ = wiEnd - (static_cast<uint32_t>(tiling_->win) + tiling_->padLeft) + 1;
            curWiLoadL1_ -= padRightL1_;
        }
        orgWin_ = curWiLoadL1_;
    } else {
        // Compute fmap H range for this core's M range
        uint32_t hoStart = mIdxStart_ / static_cast<uint32_t>(tiling_->wout);
        uint32_t hoEnd = (mIdxStart_ + actualM_ - 1) / static_cast<uint32_t>(tiling_->wout);
        uint32_t hiStart = hoStart * tiling_->strideH;
        uint32_t hiEnd = hoEnd * tiling_->strideH + tiling_->dilationH * (tiling_->kh - 1);
        ComputeHiPadRange(hiStart, hiEnd);
    }

    // L1 layout (512KB unified): [fmap | weight | bias | (scale)]
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
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                         WeightFmt, IsHwMode>::LoadFmapL1HwMode(LocalTensor<FmapT>& al1, GM_ADDR x)
{
    // HW-mode: W-window loading, support NCHW/NHWC input
    GlobalTensor<FmapT> fmapGm;
    if constexpr (isNHWCin) {
        // NHWC input: [N,H,W,C]
        uint64_t batchFmapOff = static_cast<uint64_t>(batchIdx_) * tiling_->hin * tiling_->win * tiling_->cin;
        fmapGm.SetGlobalBuffer(reinterpret_cast<__gm__ FmapT*>(x) + batchFmapOff);

        Nd2NzParams p;
        if (curWiLoadL1_ == static_cast<uint32_t>(tiling_->win)) {
            // Full-width: single Nd
            p.ndNum = 1;
            p.nValue = curHiLoadL1_ * orgWin_;
            p.srcNdMatrixStride = 0;
            p.dstNzMatrixStride = 0;
        } else {
            // W-window: one Nd per loaded H row
            p.ndNum = curHiLoadL1_;
            p.nValue = curWiLoadL1_;
            p.srcNdMatrixStride = static_cast<uint32_t>(tiling_->win * tiling_->cin);
            p.dstNzMatrixStride = curWiLoadL1_ * GK0;
        }
        p.dValue = static_cast<uint32_t>(tiling_->cin);
        p.srcDValue = static_cast<uint32_t>(tiling_->cin);
        p.dstNzNStride = 1;
        p.dstNzC0Stride = curHiLoadL1_ * orgWin_;

        uint64_t gmOff = static_cast<uint64_t>(hiLoadStart_) * tiling_->win * tiling_->cin +
                         wiLoadStart_ * tiling_->cin;
        DataCopy(al1, fmapGm[gmOff], p);
    } else {
        // NCHW input: [N,C,H,W]
        uint64_t batchFmapOff = static_cast<uint64_t>(batchIdx_) * tiling_->cin * tiling_->hin * tiling_->win;
        fmapGm.SetGlobalBuffer(reinterpret_cast<__gm__ FmapT*>(x) + batchFmapOff);

        Dn2NzParams p;
        if (curWiLoadL1_ == static_cast<uint32_t>(tiling_->win)) {
            // Full-width rows: single contiguous Dn.
            p.dnNum = 1;
            p.nValue = curHiLoadL1_ * orgWin_;
            p.srcDnMatrixStride = 0;
            p.dstNzMatrixStride = 0;
        } else {
            // W-window: one Dn per loaded H row, stride win in GM.
            p.dnNum = curHiLoadL1_;
            p.nValue = curWiLoadL1_;
            p.srcDnMatrixStride = static_cast<uint32_t>(tiling_->win);
            p.dstNzMatrixStride = curWiLoadL1_ * GK0;
        }
        p.dValue = static_cast<uint32_t>(tiling_->cin);
        p.srcDValue = static_cast<uint32_t>(tiling_->hin * tiling_->win);
        p.dstNzC0Stride = curHiLoadL1_ * orgWin_;
        p.dstNzNStride = 1;

        uint64_t gmOff = static_cast<uint64_t>(hiLoadStart_) * tiling_->win + wiLoadStart_;
        DataCopy(al1, fmapGm[gmOff], p);
    }
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                         WeightFmt, IsHwMode>::LoadFmapL1MMode(LocalTensor<FmapT>& al1, GM_ADDR x)
{
    // M-mode: full-width loading, support NCHW/NHWC input
    GlobalTensor<FmapT> fmapGm;
    if constexpr (isNHWCin) {
        // NHWC input: [N,H,W,C], use Nd2NzParams
        uint64_t batchFmapOff = static_cast<uint64_t>(batchIdx_) * tiling_->hin * tiling_->win * tiling_->cin;
        fmapGm.SetGlobalBuffer(reinterpret_cast<__gm__ FmapT*>(x) + batchFmapOff +
                               static_cast<uint64_t>(hiLoadStart_) * orgWin_ * tiling_->cin);

        Nd2NzParams p;
        p.ndNum = 1;
        p.nValue = curHiLoadL1_ * orgWin_;
        p.dValue = static_cast<uint32_t>(tiling_->cin);
        p.srcDValue = static_cast<uint32_t>(tiling_->cin);
        p.dstNzNStride = 1;
        p.dstNzC0Stride = curHiLoadL1_ * orgWin_;

        DataCopy(al1, fmapGm, p);
    } else {
        // NCHW input: [N,C,H,W], use Dn2NzParams
        uint64_t batchFmapOff = static_cast<uint64_t>(batchIdx_) * tiling_->cin * tiling_->hin * tiling_->win;
        fmapGm.SetGlobalBuffer(reinterpret_cast<__gm__ FmapT*>(x) + batchFmapOff +
                               static_cast<uint64_t>(hiLoadStart_) * orgWin_);

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
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                         WeightFmt, IsHwMode>::LoadFmapL1(GM_ADDR x)
{
    LocalTensor<FmapT> al1(TPosition::A1, 0, al1ElemCount_);

    if constexpr (IsHwMode) {
        LoadFmapL1HwMode(al1, x);
    } else {
        LoadFmapL1MMode(al1, x);
    }
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                         WeightFmt, IsHwMode>::LoadWeightL1(GM_ADDR filter)
{
    uint32_t n1Total = AlignB(tiling_->cout, GN0) / GN0;
    GlobalTensor<WeightT> filterGm;
    filterGm.SetGlobalBuffer(reinterpret_cast<__gm__ WeightT*>(filter));
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
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                         WeightFmt, IsHwMode>::LoadBiasScaleL1(GM_ADDR bias,
                                                                               const ExtendParams* extendParams)
{
    uint32_t bsOff = nIdx_ * tiling_->singleCoreCo;
    GlobalTensor<BiasT> biasGm;
    biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT*>(bias) + bsOff);
    if (tiling_->hasBias && actualCo_ > 0) {
        LocalTensor<BiasT> biasL1(TPosition::A1, biasL1OffBytes_, tiling_->singleCoreCo);
        LoadChannelWiseL1FullLoad<BiasT>(biasL1, biasGm[0], actualCo_);
    }
    {
        if (tiling_->quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
            scale0Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(extendParams->scale0) +
                                      nIdx_ * tiling_->singleCoreCo);
            LocalTensor<uint64_t> scale0L1(TPosition::A1, scale0L1OffBytes_, tiling_->singleCoreCo);
            LoadChannelWiseL1FullLoad<uint64_t>(scale0L1, scale0Gm_[0], actualCo_);
        } else if (tiling_->quantMode0 == static_cast<uint8_t>(QuantModeType::SCALAR_QUANT)) {
            scale0Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(extendParams->scale0));
        }
        if (tiling_->reluMode0 == static_cast<uint8_t>(ReluMode::VECTOR_RELU)) {
            reluWeight0Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(extendParams->reluWeight0) +
                                           nIdx_ * tiling_->singleCoreCo);
            LocalTensor<float> reluWeight0L1(TPosition::A1, reluWeight0L1OffBytes_, tiling_->singleCoreCo);
            LoadChannelWiseL1FullLoad<float>(reluWeight0L1, reluWeight0Gm_[0], actualCo_);
        } else if (tiling_->reluMode0 == static_cast<uint8_t>(ReluMode::SCALAR_RELU)) {
            reluWeight0Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(extendParams->reluWeight0));
        }

        if (tiling_->dualOutput) {
            if (tiling_->quantMode1 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                scale1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(extendParams->scale1) +
                                          nIdx_ * tiling_->singleCoreCo);
                LocalTensor<uint64_t> scale1L1(TPosition::A1, scale1L1OffBytes_, tiling_->singleCoreCo);
                LoadChannelWiseL1FullLoad<uint64_t>(scale1L1, scale1Gm_[0], actualCo_);
            } else if (tiling_->quantMode1 == static_cast<uint8_t>(QuantModeType::SCALAR_QUANT)) {
                scale1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(extendParams->scale1));
            }
            if (tiling_->reluMode1 == static_cast<uint8_t>(ReluMode::VECTOR_RELU)) {
                reluWeight1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(extendParams->reluWeight1) +
                                               nIdx_ * tiling_->singleCoreCo);
                LocalTensor<float> reluWeight1L1(TPosition::A1, reluWeight1L1OffBytes_, tiling_->singleCoreCo);
                LoadChannelWiseL1FullLoad<float>(reluWeight1L1, reluWeight1Gm_[0], actualCo_);
            } else if (tiling_->reluMode1 == static_cast<uint8_t>(ReluMode::SCALAR_RELU)) {
                reluWeight1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(extendParams->reluWeight1));
            }
        }
    }
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                         WeightFmt, IsHwMode>::LoadBiasToBT()
{
    constexpr uint32_t BT_ALIGN = 64;
    uint32_t btElemNum = AlignB(tiling_->singleCoreCo * sizeof(L0cT), BT_ALIGN) / sizeof(L0cT);
    LocalTensor<L0cT> biasBT(TPosition::C2, 0, btElemNum);
    LocalTensor<BiasT> biasL1src(TPosition::A1, biasL1OffBytes_, tiling_->singleCoreCo);
    uint32_t blkCnt = AlignB(actualCo_ * sizeof(BiasT), BT_ALIGN) / 32;
    DataCopyParams cp(1, static_cast<uint16_t>(blkCnt), 0, 0);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
    if constexpr (AscendC::IsSameType<weightType, half>::value) {
        cp.fixShiftVal = FIX_SHIFT_VAL_LEN_A16W16 - tiling_->fixedShiftValue;
    }
#endif
    DataCopy(biasBT, biasL1src[0], cp);
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                         WeightFmt, IsHwMode>::SetupLoad3DBase()
{
    uint8_t padList[PAD_LIST_NUM] = {
        static_cast<uint8_t>(IsHwMode ? padLeftL1_ : static_cast<int32_t>(tiling_->padLeft)),
        static_cast<uint8_t>(IsHwMode ? padRightL1_ : static_cast<int32_t>(tiling_->padRight)),
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

    // Process recomputes load3dXmTmp_ per chunk; this seeds the M-mode base only.
    uint32_t posM = IsHwMode ? woIdxStart_ : (mIdxStart_ % static_cast<uint32_t>(tiling_->wout));
    load3dXmTmp_ = ((static_cast<uint64_t>(ml0_) & MASK_16) << MSTEP_OFFSET) |
                   ((static_cast<uint64_t>(posM) & MASK_16) << POSM_OFFSET);
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                         WeightFmt, IsHwMode>::DoLoadAL0(LocalTensor<FmapT>& al1,
                                                                         LocalTensor<FmapT>& al0, uint32_t kOff,
                                                                         uint32_t curK)
{
    uint64_t posK = kOff;
    uint64_t xm = static_cast<uint64_t>(curK) | (posK << POSK_OFFSET) | load3dXmTmp_;
    Load3DBitModeParam param;
    param.SetConfig0(xm);
    param.SetConfig1(load3dXtBase_);
    LoadData<TPosition::A2, TPosition::A1, FmapT>(al0, al1, param);
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                         WeightFmt, IsHwMode>::DoLoadBL0(LocalTensor<WeightT>& bl0,
                                                                         LocalTensor<WeightT>& bl1, uint32_t kOff,
                                                                         uint32_t curK)
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
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt, bool IsHwMode>
template <typename OutputT, uint64_t FixpipeIdx>
__aicore__ inline QuantMode_t Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                                WeightFmt, IsHwMode>::GetQuantPreInt32()
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
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt, bool IsHwMode>
template <typename OutputT, uint64_t FixpipeIdx>
__aicore__ inline QuantMode_t Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                                WeightFmt, IsHwMode>::GetQuantPreFp32()
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
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt, bool IsHwMode>
template <typename OutputT, uint64_t FixpipeIdx, const FixpipeConfig& config, const FixpipeConfig& configFp>
__aicore__ inline void Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                         WeightFmt, IsHwMode>::DoCopyOut(GM_ADDR yAddr, LocalTensor<L0cT>& cl0,
                                                                         uint32_t outOffset, uint32_t curM,
                                                                         uint32_t curMAlign, uint32_t actualCo,
                                                                         uint32_t fpDnNum, uint32_t fpDstDnStride)
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
        outOff = static_cast<uint64_t>(outOffset) * tiling_->cout;
    } else {
        nOutOff = static_cast<uint64_t>(nIdx_) * tiling_->singleCoreCo * hwOut;
        dstStride = static_cast<uint32_t>(hwOut);
        outOff = static_cast<uint64_t>(outOffset);
    }

    GlobalTensor<OutputT> outputGm;
    outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ OutputT*>(yAddr) + batchOutOff + nOutOff);

    FixpipeParamsC310<Layout> fp;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
    if constexpr (AscendC::IsSameType<weightType, half>::value) {
        fp.fixShiftVal = FIX_SHIFT_VAL_LEN_A16W16 - tiling_->fixedShiftValue;
    }
#endif
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

    // fp.params: different field names per layout, must be if-constexpr.
    // HW-mode Wo-split uses multi-Dn (fpDnNum>1); M-mode uses single-Dn (fpDnNum==1).
    if constexpr (isNHWCout) {
        fp.params.ndNum = fpDnNum;
        fp.params.srcNdStride = (fpDnNum == 1) ? 0 : AlignB(curM, GM0);
        fp.params.dstNdStride = fpDstDnStride * tiling_->cout;
    } else {
        fp.params.dnNum = fpDnNum;
        fp.params.srcNzMatrixStride = (fpDnNum == 1) ? (curMAlign * CeilDiv(actualCo, GM0)) : AlignB(curM, GM0);
        fp.params.dstDnMatrixStride = fpDstDnStride;
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
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                         WeightFmt, IsHwMode>::CopyOutResult(LocalTensor<L0cT>& cl0, GM_ADDR y,
                                                                             const ExtendParams* extendParams,
                                                                             uint32_t outOff, uint32_t fpMSize,
                                                                             uint32_t curMAlign, uint32_t fpDnNum,
                                                                             uint32_t fpDstDnStride)
{
    if constexpr (isNHWCout) {
        DoCopyOut<Output0T, 0, CFG_ROW_MAJOR, CFG_ROW_MAJOR_FIXED_POINT>(y, cl0, outOff, fpMSize, curMAlign, actualCo_,
                                                                         fpDnNum, fpDstDnStride);
        if (tiling_->dualOutput) {
            DoCopyOut<Output1T, 1, CFG_ROW_MAJOR, CFG_ROW_MAJOR_FIXED_POINT>(
                extendParams->y1, cl0, outOff, fpMSize, curMAlign, actualCo_, fpDnNum, fpDstDnStride);
        }
    } else {
        DoCopyOut<Output0T, 0, CFG_COLUMN_MAJOR, CFG_COLUMN_MAJOR_FIXED_POINT>(y, cl0, outOff, fpMSize, curMAlign,
                                                                               actualCo_, fpDnNum, fpDstDnStride);
        if (tiling_->dualOutput) {
            DoCopyOut<Output1T, 1, CFG_COLUMN_MAJOR, CFG_COLUMN_MAJOR_FIXED_POINT>(
                extendParams->y1, cl0, outOff, fpMSize, curMAlign, actualCo_, fpDnNum, fpDstDnStride);
        }
    }
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                         WeightFmt, IsHwMode>::ProcessHwMode(LocalTensor<FmapT>& al1,
                                                                             LocalTensor<WeightT>& bl1, uint32_t mmadN,
                                                                             uint32_t kL0MaxIter, uint64_t hwOut,
                                                                             GM_ADDR y,
                                                                             const ExtendParams* extendParams)
{
    // Stage 3 (HW-mode): Ho/Wo-chunk loop -> K-loop -> Fixpipe.
    bool needRowSplit = (actualWo_ < static_cast<uint32_t>(tiling_->wout));
    for (uint32_t hoOff = 0; hoOff < actualHo_; hoOff += hoL0_) {
        uint32_t curHo = hoL0_;
        if (hoOff + curHo > actualHo_) {
            curHo = actualHo_ - hoOff;
        }
        for (uint32_t woOff = 0; woOff < actualWo_; woOff += woL0_) {
            uint32_t curWo = woL0_;
            if (woOff + curWo > actualWo_) {
                curWo = actualWo_ - woOff;
            }
            uint32_t curM = curHo * curWo;
            uint32_t curMAlign = AlignB(curM, GM0);
            uint32_t posM = hoOff * static_cast<uint32_t>(tiling_->wout) + woOff;

            load3dXmTmp_ = ((static_cast<uint64_t>(curMAlign) & MASK_16) << MSTEP_OFFSET) |
                           ((static_cast<uint64_t>(posM) & MASK_16) << POSM_OFFSET);
#if defined(ASC_DEVKIT_VERSION_NUM) && (ASC_DEVKIT_VERSION_NUM >= 90000000)
            SetLoadDataRepeatWithStride(LoadDataRepeatParamWithStride(0, 1, 0, static_cast<uint16_t>(curMAlign / GM0)));
#else
            SetLoadDataRepeat(LoadDataRepeatParam(0, 1, 0, static_cast<uint16_t>(curMAlign / GM0)));
#endif

            LocalTensor<L0cT> cl0(TPosition::CO1, 0, L0C_ELEMS);
            MmadAccumulateTile(al1, bl1, cl0, curMAlign, mmadN, kL0MaxIter);

            uint32_t outOff = (hoIdxStart_ + hoOff) * static_cast<uint32_t>(tiling_->wout) + woIdxStart_ + woOff;
            uint32_t fpMSize = needRowSplit ? curWo : curM;
            uint32_t fpDnNum = needRowSplit ? curHo : 1;
            uint32_t fpDstDnStride = needRowSplit ? static_cast<uint32_t>(tiling_->wout) : static_cast<uint32_t>(hwOut);
            CopyOutResult(cl0, y, extendParams, outOff, fpMSize, curMAlign, fpDnNum, fpDstDnStride);
        }
    }
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                         WeightFmt, IsHwMode>::ProcessMMode(LocalTensor<FmapT>& al1,
                                                                            LocalTensor<WeightT>& bl1, uint32_t mmadN,
                                                                            uint32_t kL0MaxIter, uint64_t hwOut,
                                                                            GM_ADDR y, const ExtendParams* extendParams)
{
    // Stage 3 (M-mode): M-loop -> K-loop -> Fixpipe.
    for (uint32_t mOff = 0; mOff < actualM_; mOff += hoL0_) {
        uint32_t curM = hoL0_;
        if (mOff + curM > actualM_) {
            curM = actualM_ - mOff;
        }
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
        MmadAccumulateTile(al1, bl1, cl0, curMAlign, mmadN, kL0MaxIter);

        CopyOutResult(cl0, y, extendParams, mIdxStart_ + mOff, curM, curMAlign, 1, static_cast<uint32_t>(hwOut));
    }
}

template <typename FmapType, typename weightType, typename biasType, typename out0Type, typename out1Type,
          bool isNHWCin, bool isNHWCout, ConvFormat WeightFmt, bool IsHwMode>
__aicore__ inline void Conv2dSmallKernel<FmapType, weightType, biasType, out0Type, out1Type, isNHWCin, isNHWCout,
                                         WeightFmt, IsHwMode>::Process(GM_ADDR x, GM_ADDR filter, GM_ADDR bias,
                                                                       GM_ADDR y, const ExtendParams* extendParams)
{
    if (!coreActive_ || actualCo_ == 0) {
        return;
    }

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

    uint32_t kL0MaxIter = CeilDiv(kTotal_, tiling_->kL0);
    uint64_t hwOut = static_cast<uint64_t>(tiling_->hout) * tiling_->wout;
    LocalTensor<FmapT> al1(TPosition::A1, 0, al1ElemCount_);
    LocalTensor<WeightT> bl1(TPosition::B1, bl1OffBytes_, bl1ElemCount_);
    uint32_t mmadN = AlignB(actualCo_, GN0);

    if constexpr (IsHwMode) {
        ProcessHwMode(al1, bl1, mmadN, kL0MaxIter, hwOut, y, extendParams);
    } else {
        ProcessMMode(al1, bl1, mmadN, kL0MaxIter, hwOut, y, extendParams);
    }
}
#endif
