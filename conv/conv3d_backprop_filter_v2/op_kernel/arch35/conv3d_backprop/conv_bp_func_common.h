/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv_bp_func_common.h
 * \brief
 */

#ifndef CONV_BP_FUNC_COMMON_H
#define CONV_BP_FUNC_COMMON_H
#include "impl/conv_bp_filter_sub_func_load_gm_to_l1.h"

namespace ConvolutionBackpropFunc {

template <class Intf>
__aicore__ inline void FreeA1Tensor(Intf* self, bool a1PingPongFlag)
{
    if (a1PingPongFlag) {
        self->ctx.a1Ping_.FreeTensor(self->ctx.cacheA1BufPing_);
#ifdef ASCENDC_CPU_DEBUG
        // ASCENDC_CPU_DEBUG就是__CCE_KT_TEST__
        self->ctx.cacheA1BufPing_.SetSize(0);
#endif
    } else {
        self->ctx.a1Pong_.FreeTensor(self->ctx.cacheA1BufPong_);
#ifdef ASCENDC_CPU_DEBUG
        self->ctx.cacheA1BufPong_.SetSize(0);
#endif
    }
}

template <class Intf>
__aicore__ inline void FreeB1Tensor(Intf* self, bool b1PingPongFlag)
{
    if (b1PingPongFlag) {
        self->ctx.b1Ping_.FreeTensor(self->ctx.cacheB1BufPing_);
#ifdef ASCENDC_CPU_DEBUG
        self->ctx.cacheB1BufPing_.SetSize(0);
#endif
    } else {
        self->ctx.b1Pong_.FreeTensor(self->ctx.cacheB1BufPong_);
#ifdef ASCENDC_CPU_DEBUG
        self->ctx.cacheB1BufPong_.SetSize(0);
#endif
    }
}

template <class Intf>
__aicore__ inline void updateParasForSplitW(Intf* self, Out2L1ScalarParams& out2L1Params, int32_t startWo,
                                            uint64_t out2A1SrcAddrStart, uint64_t out2B1SrcAddrStart)
{
    uint64_t singleCoreHoWo = static_cast<uint64_t>(self->ctx.singleShapeHo_) * self->ctx.singleShapeWo_;
    uint64_t kIter = Ceil(singleCoreHoWo, self->ctx.tiling_->baseK);
    self->ctx.kIter_ = kIter;
    self->ctx.tailK_ = singleCoreHoWo - self->ctx.tiling_->baseK * (kIter - 1);
    self->ctx.stepKbRound = Ceil(kIter, self->ctx.tiling_->stepKb);
    self->ctx.stepKaRound = Ceil(kIter, self->ctx.tiling_->stepKa);

    self->ctx.load3d_.padList[0] = 0;
    int64_t b1SrcWiLeftOffGm = static_cast<int64_t>(startWo) * self->ctx.tiling_->strideW - self->ctx.tiling_->padLeft;
    if (b1SrcWiLeftOffGm < 0) {
        self->ctx.load3d_.padList[0] = -b1SrcWiLeftOffGm;
    }
    self->ctx.load3d_.padList[1] = 0;
    int64_t b1SrcWiRightOffGm = static_cast<int64_t>(startWo + self->ctx.singleShapeWo_) * self->ctx.tiling_->strideW +
                                self->ctx.strideKernelDilationW - (self->ctx.tiling_->padLeft + self->ctx.tiling_->wi);
    if (b1SrcWiRightOffGm > 0) {
        self->ctx.load3d_.padList[1] = b1SrcWiRightOffGm;
    }

    // A矩阵不用做LOAD3D操作，不存在交叠；
    if constexpr (Intf::Config::cType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
        out2L1Params.out2A1SrcAddr = out2A1SrcAddrStart + startWo;
    } else if constexpr (Intf::Config::cType::format == ConvolutionBackprop::CubeFormat::NDHWC) {
        out2L1Params.out2A1SrcAddr = out2A1SrcAddrStart + startWo * self->ctx.tiling_->cout;
    }

    // B矩阵考虑卷积操作，导致前后两个split交叠问题；
    if constexpr (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
        if (self->ctx.load3d_.padList[0]) {
            out2L1Params.out2B1SrcAddr = out2B1SrcAddrStart;
        } else {
            out2L1Params.out2B1SrcAddr = out2B1SrcAddrStart + b1SrcWiLeftOffGm;
        }
    } else if constexpr (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NDHWC) {
        if (self->ctx.load3d_.padList[0]) {
            out2L1Params.out2B1SrcAddr = out2B1SrcAddrStart;
        } else {
            out2L1Params.out2B1SrcAddr = out2B1SrcAddrStart + b1SrcWiLeftOffGm * self->ctx.tiling_->cin;
        }
    }

    if (out2L1Params.singleShapeWi > (self->ctx.load3d_.padList[0] + self->ctx.load3d_.padList[1])) {
        self->ctx.load3d_.l1W = out2L1Params.singleShapeWi - self->ctx.load3d_.padList[0] -
                                self->ctx.load3d_.padList[1];
    } else {
        self->ctx.load3d_.l1W = 0;
    }
}

template <class Intf>
__aicore__ inline void calculateWoIterTimes(Intf* self, int32_t& woIterateTimes, const int32_t splitWo)
{
    if (splitWo == 0) {
        woIterateTimes = 1;
        return;
    }
    woIterateTimes = Ceil(self->ctx.tiling_->wo, splitWo);
}

template <class Intf>
__aicore__ inline void updateSingleShapeWoI(Intf* self, Out2L1ScalarParams& out2L1Params, const int32_t woIterateTimes,
                                            const int32_t splitWoIdx, const int32_t splitWo)
{
    if (woIterateTimes > 1) {
        if ((splitWoIdx + 1) == woIterateTimes) {
            self->ctx.singleShapeWo_ = self->ctx.tiling_->wo - splitWoIdx * splitWo;
        } else {
            self->ctx.singleShapeWo_ = splitWo;
        }
        // 包含pad等在内，所以singleShapeWi可能大于wi。关注特殊case，当singleShapeWi > wi时是否能够正常运行；
        out2L1Params.singleShapeWi = self->ctx.singleShapeWo_ * self->ctx.tiling_->strideW +
                                     self->ctx.strideKernelDilationW;
    } else {
        self->ctx.singleShapeWo_ = self->ctx.tiling_->wo;
        uint64_t singleShapeWi = self->ctx.singleShapeWo_ * self->ctx.tiling_->strideW +
                                 self->ctx.strideKernelDilationW;
        out2L1Params.singleShapeWi = singleShapeWi;
    }
}

template <class Intf>
static __aicore__ inline void CalcParamsMmad(Intf* self)
{
    self->ctx.mmad_.m = self->ctx.baseUseM_;
    self->ctx.mmad_.n = self->ctx.baseUseN_;
}

template <class Intf>
__aicore__ inline void ClearL0CLoad3dParams(Intf* self, LocalTensor<typename Intf::SrcT>& l0b)
{
    constexpr uint32_t DEFAULT_MEXTENSION = 16;
    constexpr uint32_t DEFAULT_PAD_DOWN = 255;

    using LoadData3DParamsV2SrcT = LoadData3DParamsV2<typename Intf::SrcT>;
    LoadData3DParamsV2SrcT load3d;
    load3d.padList[0] = 0;
    load3d.padList[1] = 0;
    load3d.padList[2] = 0;
    load3d.padList[3] = DEFAULT_PAD_DOWN;
    load3d.l1W = 1;
    load3d.l1H = 1;
    load3d.channelSize = Ceil(self->ctx.tiling_->baseN, self->ctx.tiling_->n0) * self->ctx.tiling_->n0;
    load3d.kStartPt = 0;
    load3d.mStartPt = 0;
    load3d.kExtension = self->ctx.tiling_->baseN;
    load3d.mExtension = DEFAULT_MEXTENSION;
    load3d.strideW = 1;
    load3d.strideH = 1;
    load3d.filterH = 1;
    load3d.filterW = 1;
    load3d.dilationFilterW = 1;
    load3d.dilationFilterH = 1;

#if defined(ASC_DEVKIT_VERSION_NUM) && (ASC_DEVKIT_VERSION_NUM >= 90000000)
    LoadDataRepeatParamWithStride repeatParam = {
        0, 1, 0, static_cast<uint16_t>(ShiftCeilM0(self->ctx.tiling_->baseN, self->ctx.tiling_->n0))};
    SetLoadDataRepeatWithStride(repeatParam);
    LoadDataWithStride(l0b[0], self->ctx.cacheB1BufPing_, load3d);
#else
    LoadDataRepeatParam repeatParam = {
        0, 1, 0, static_cast<uint16_t>(ShiftCeilM0(self->ctx.tiling_->baseN, self->ctx.tiling_->n0))};
    SetLoadDataRepeat(repeatParam);
    LoadData(l0b[0], self->ctx.cacheB1BufPing_, load3d);
#endif
}

template <class Intf>
__aicore__ inline void ClearL0CLoad2dParams(Intf* self, LocalTensor<typename Intf::SrcT>& l0a)
{
    LoadData2DParamsV2 load2d;
    load2d.mStartPosition = 0;
    load2d.kStartPosition = 0;
    load2d.mStep = Ceil(self->ctx.tiling_->baseM, self->ctx.tiling_->m0);
    if (IsSameType<typename Intf::SrcT, float>::value) {
        load2d.kStep = 2; // fp32类型，kstep一定是2的倍数
    } else {
        load2d.kStep = 1;
    }
    load2d.srcStride = load2d.mStep;
    load2d.dstStride = load2d.mStep;
    load2d.ifTranspose = 0;
    LoadData(l0a[0], self->ctx.cacheA1BufPing_, load2d);
}

template <class Intf>
__aicore__ inline void ClearBaseMNL0C(Intf* self, LocalTensor<typename Intf::L0cT>& l0c)
{
    LocalTensor<typename Intf::SrcT> l0a = self->ctx.l0aBuf_.template Get<typename Intf::SrcT>();
    LocalTensor<typename Intf::SrcT> l0b = self->ctx.l0bBuf_.template Get<typename Intf::SrcT>();

    constexpr uint32_t l0aPingPongAddr = TOTAL_L0A_SIZE / 2 / sizeof(typename Intf::SrcT);
    constexpr uint32_t l0bPingPongAddr = TOTAL_L0B_SIZE / 2 / sizeof(typename Intf::SrcT);

    if (self->ctx.l0aPingPongFlag_) {
        l0a = l0a[l0aPingPongAddr];
        l0b = l0b[l0bPingPongAddr];
    }

    LocalTensor<typename Intf::SrcT> useB1Buf = self->ctx.b1Ping_.template AllocTensor<typename Intf::SrcT>();
    InitZeroValue(self, useB1Buf);
    self->ctx.b1Ping_.EnQue(useB1Buf);

    LocalTensor<typename Intf::SrcT> useA1Buf = self->ctx.a1Ping_.template AllocTensor<typename Intf::SrcT>();
    InitZeroValue(self, useA1Buf);
    self->ctx.a1Ping_.EnQue(useA1Buf);

    self->ctx.cacheB1BufPing_ = self->ctx.b1Ping_.template DeQue<typename Intf::SrcT>();
    self->ctx.cacheA1BufPing_ = self->ctx.a1Ping_.template DeQue<typename Intf::SrcT>();

    WaitFlag<HardEvent::M_MTE1>(self->ctx.l0aPingPongFlag_);

    ClearL0CLoad3dParams<Intf>(self, l0b);
    ClearL0CLoad2dParams<Intf>(self, l0a);

    FreeB1Tensor(self, 1);
    FreeA1Tensor(self, 1);
    MmadParams mmad_;
    mmad_.m = self->ctx.tiling_->baseM;
    mmad_.n = self->ctx.tiling_->baseN;
    mmad_.k = 16;
    mmad_.cmatrixInitVal = true;

    SetFlag<HardEvent::MTE1_M>(self->ctx.l0aPingPongFlag_);
    WaitFlag<HardEvent::MTE1_M>(self->ctx.l0aPingPongFlag_);

    Mmad(l0c[0], l0a[0], l0b[0], mmad_);
    if (mmad_.m * mmad_.n < 2560) {
        PipeBarrier<PIPE_M>();
    }

    SetFlag<HardEvent::M_MTE1>(self->ctx.l0aPingPongFlag_);
    self->ctx.l0aPingPongFlag_ ^= self->ctx.useL0PingPong_;
}

__aicore__ inline void UpdateIdx(bool isLastStepKa, bool isLastStepKb, uint32_t& kaIdx, uint32_t& kbIdx,
                                 uint64_t& kaStepIdx, uint64_t& kbStepIdx)
{
    if (isLastStepKa) {
        ++kaStepIdx;
        kaIdx = 0;
    } else {
        ++kaIdx;
    }
    if (isLastStepKb) {
        ++kbStepIdx;
        kbIdx = 0;
    } else {
        ++kbIdx;
    }
}

template <class Intf>
__aicore__ inline void ComputeInit(Intf* self, Out2L1ScalarParams& out2L1Params, LocalTensor<typename Intf::L0cT>& l0c,
                                   uint32_t& baseUseMBak)
{
    if (self->ctx.l0cPingPongFlag_) {
        l0c = self->ctx.l0cPing_.template AllocTensor<typename Intf::L0cT>();
    } else {
        l0c = self->ctx.l0cPong_.template AllocTensor<typename Intf::L0cT>();
    }

    if constexpr (Intf::conv3ddwConfig.isSplitKernelHW) {
        ClearBaseMNL0C<Intf>(self, l0c);
    }

    baseUseMBak = self->ctx.baseUseM_;
    if (self->ctx.baseUseM_ == 1) {
        self->ctx.baseUseM_ = ShiftCeilM0(self->ctx.baseUseM_, self->ctx.tiling_->m0) * self->ctx.tiling_->m0;
    }

    CalcParamsL12L0a<Intf>(self);
    CalcParamsL12L0b<Intf>(self);
    CalcParamsMmad<Intf>(self);
    CalOut2L1ScalarParams(self, out2L1Params);
}

template <class Intf>
__aicore__ inline void ExecuteMTE1L0b(Intf* self, Out2L1ScalarParams& out2L1Params,
                                      LocalTensor<typename Intf::SrcT>& l0b, bool b1PingPongFlag, bool isLoadB1,
                                      bool isLastStepKb, bool isLastKIter, uint64_t k, uint32_t kbIdx)
{
    constexpr uint32_t l0bPingPongAddr = TOTAL_L0B_SIZE / 2 / sizeof(typename Intf::SrcT);

    l0b = self->ctx.l0bBuf_.template Get<typename Intf::SrcT>();
    if (self->ctx.l0aPingPongFlag_) {
        l0b = l0b[l0bPingPongAddr];
    }
    self->ctx.load3d_.mStartPt = (k - kbIdx) * self->ctx.tiling_->baseK % self->ctx.singleShapeWo_ +
                                 kbIdx * self->ctx.tiling_->baseK;

    if (unlikely(out2L1Params.isLoad2L1B && isLoadB1)) {
        if (b1PingPongFlag) {
            self->ctx.cacheB1BufPing_ = self->ctx.b1Ping_.template DeQue<typename Intf::SrcT>();
        } else {
            self->ctx.cacheB1BufPong_ = self->ctx.b1Pong_.template DeQue<typename Intf::SrcT>();
        }
    }
    if (b1PingPongFlag) {
        self->ctx.load3d_.l1H = self->ctx.bL1HiCopyLenPing;
        self->ctx.load3d_.padList[2] = self->ctx.bL1PadUpPing;
        LoadL12L0b<Intf>(self, self->ctx.cacheB1BufPing_, l0b);
    } else {
        self->ctx.load3d_.l1H = self->ctx.bL1HiCopyLenPong;
        self->ctx.load3d_.padList[2] = self->ctx.bL1PadUpPong;
        LoadL12L0b<Intf>(self, self->ctx.cacheB1BufPong_, l0b);
    }
    if (out2L1Params.isFreeBL1 && (isLastStepKb || isLastKIter)) {
        FreeB1Tensor(self, b1PingPongFlag);
    }
}

template <class Intf>
__aicore__ inline void ExecuteMTE1L0a(Intf* self, Out2L1ScalarParams& out2L1Params,
                                      LocalTensor<typename Intf::SrcT>& l0a, bool a1PingPongFlag, bool isLoadA1,
                                      bool isLastStepKa, bool isLastKIter, uint64_t k)
{
    constexpr uint32_t l0aPingPongAddr = TOTAL_L0A_SIZE / 2 / sizeof(typename Intf::SrcT);

    l0a = self->ctx.l0aBuf_.template Get<typename Intf::SrcT>();
    if (self->ctx.l0aPingPongFlag_) {
        l0a = l0a[l0aPingPongAddr];
    }
    if (unlikely(out2L1Params.isLoad2L1A && isLoadA1)) {
        if (a1PingPongFlag) {
            self->ctx.cacheA1BufPing_ = self->ctx.a1Ping_.template DeQue<typename Intf::SrcT>();
        } else {
            self->ctx.cacheA1BufPong_ = self->ctx.a1Pong_.template DeQue<typename Intf::SrcT>();
        }
    }
    if (a1PingPongFlag) {
        LoadL12L0a<Intf>(self, self->ctx.cacheA1BufPing_, k, l0a, self->ctx.alignedL1UseKaPing_,
                         self->ctx.alignedL1UseMPing_);
    } else {
        LoadL12L0a<Intf>(self, self->ctx.cacheA1BufPong_, k, l0a, self->ctx.alignedL1UseKaPong_,
                         self->ctx.alignedL1UseMPong_);
    }
    if (out2L1Params.isFreeAL1 && (isLastStepKa || isLastKIter)) {
        FreeA1Tensor(self, a1PingPongFlag);
    }
}

template <class Intf>
__aicore__ inline void ExecuteMmad(Intf* self, const LocalTensor<typename Intf::SrcT>& l0a,
                                   const LocalTensor<typename Intf::SrcT>& l0b, LocalTensor<typename Intf::L0cT>& l0c,
                                   bool& isFirstMmad)
{
    SetFlag<HardEvent::MTE1_M>(self->ctx.l0aPingPongFlag_);
    WaitFlag<HardEvent::MTE1_M>(self->ctx.l0aPingPongFlag_);
    self->ctx.mmad_.cmatrixInitVal = isFirstMmad;
    self->ctx.mmad_.k = self->ctx.baseUseK_;
    MmadLocal<Intf>(self, l0a, l0b, l0c);
    isFirstMmad = false;
}

template <class Intf>
__aicore__ inline void ComputeLoop(Intf* self, Out2L1ScalarParams& out2L1Params, LocalTensor<typename Intf::SrcT>& l0a,
                                   LocalTensor<typename Intf::SrcT>& l0b, LocalTensor<typename Intf::L0cT>& l0c,
                                   bool& isFirstMmad, uint64_t curMKL1Idx, uint64_t curNKL1Idx, uint64_t hkIdx)
{
    bool isAL1PingPong = self->ctx.tiling_->al1Pbuffer > 1;
    bool isBL1PingPong = self->ctx.tiling_->bl1Pbuffer > 1;
    uint32_t kaIdx = 0, kbIdx = 0;
    uint64_t kaStepIdx = 0, kbStepIdx = 0;
    bool skipCurrentHiCompute = false;
    bool skipCurrentHiComputePreLoad = false;
    bool isB1NormalLoad = true;
    bool isA1NormalLoad = true;

    for (uint64_t k = 0; k < self->ctx.kIter_; k++) {
        bool isLastKIter = k + 1 == self->ctx.kIter_;
        bool isLastStepKa = kaIdx + 1 == self->ctx.tiling_->stepKa;
        bool isLastStepKb = kbIdx + 1 == self->ctx.tiling_->stepKb;
        self->ctx.baseUseK_ = isLastKIter ? self->ctx.tailK_ : self->ctx.tiling_->baseK;
        bool b1PingPongFlag = true;
        bool a1PingPongFlag = true;
        bool isLoadA1 = (kaIdx == 0);
        bool isLoadB1 = (kbIdx == 0);
        /*
            通过M*K的奇偶判断load到L1A ping还是L1A pong, BL1同理
                        kL1Idx=0  kL1Idx=1 kL1Idx=2
                        ----------------------------
            mL1Idx=0    |  ping  |  pong  |  ping  |
                        ----------------------------
            mL1Idx=1    |  pong  |  ping  |  pong  |
                        ----------------------------
            mL1Idx=2    |  ping  |  pong  |  ping  |
                        ----------------------------
        */
        if (isBL1PingPong) {
            b1PingPongFlag = (curNKL1Idx + kbStepIdx + 1) & 1;
        }
        if constexpr (!Intf::conv3ddwConfig.isSplitKernelHW) {
            ComputeLoadToB1<Intf>(self, b1PingPongFlag, out2L1Params, kbStepIdx, isB1NormalLoad, isBL1PingPong,
                                  isLoadB1, skipCurrentHiCompute, skipCurrentHiComputePreLoad);
        } else {
            LoadToB1SplitKernelHW<Intf>(self, b1PingPongFlag, out2L1Params, kbStepIdx, hkIdx, isLoadB1,
                                        skipCurrentHiCompute);
        }
        if (skipCurrentHiCompute) {
            isB1NormalLoad = true;
            isA1NormalLoad = true;
            UpdateIdx(isLastStepKa, isLastStepKb, kaIdx, kbIdx, kaStepIdx, kbStepIdx);
            continue;
        }

        if (isAL1PingPong) {
            a1PingPongFlag = (curMKL1Idx + kaStepIdx + 1) & 1;
        }
        ComputeLoadToA1<Intf>(self, a1PingPongFlag, k, out2L1Params, kaStepIdx, isA1NormalLoad, isAL1PingPong,
                              isLoadA1);

        // MTE2流水中对L1B、L1A进行预取处理
        if constexpr (!Intf::conv3ddwConfig.isSplitKernelHW) {
            ComputeLoadToB1PreLoad<Intf>(self, b1PingPongFlag, out2L1Params, kbStepIdx, k, isB1NormalLoad,
                                         isBL1PingPong, isLoadB1, skipCurrentHiComputePreLoad);
            ComputeLoadToA1PreLoad<Intf>(self, a1PingPongFlag, k, out2L1Params, kaStepIdx, isA1NormalLoad,
                                         isAL1PingPong, isLoadA1, skipCurrentHiComputePreLoad);
        }

        WaitFlag<HardEvent::M_MTE1>(self->ctx.l0aPingPongFlag_ & 1);
        ExecuteMTE1L0b<Intf>(self, out2L1Params, l0b, b1PingPongFlag, isLoadB1, isLastStepKb, isLastKIter, k, kbIdx);
        ExecuteMTE1L0a<Intf>(self, out2L1Params, l0a, a1PingPongFlag, isLoadA1, isLastStepKa, isLastKIter, k);
        ExecuteMmad<Intf>(self, l0a, l0b, l0c, isFirstMmad);
        SetFlag<HardEvent::M_MTE1>(self->ctx.l0aPingPongFlag_);
        self->ctx.l0aPingPongFlag_ ^= self->ctx.useL0PingPong_;
        UpdateIdx(isLastStepKa, isLastStepKb, kaIdx, kbIdx, kaStepIdx, kbStepIdx);
    }
}

} // namespace ConvolutionBackpropFunc
#endif
