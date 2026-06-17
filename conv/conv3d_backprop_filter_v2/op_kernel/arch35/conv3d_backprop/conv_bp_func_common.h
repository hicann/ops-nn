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

namespace ConvolutionBackpropFunc {

template <class Intf>
__aicore__ inline void FreeA1Tensor(Intf *self, bool a1PingPongFlag)
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
__aicore__ inline void FreeB1Tensor(Intf *self, bool b1PingPongFlag)
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
__aicore__ inline void updateParasForSplitW(Intf *self, Out2L1ScalarParams& out2L1Params, int32_t startWo, uint64_t out2A1SrcAddrStart, uint64_t out2B1SrcAddrStart) {
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

    //A矩阵不用做LOAD3D操作，不存在交叠；
    if constexpr (Intf::Config::cType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
        out2L1Params.out2A1SrcAddr = out2A1SrcAddrStart + startWo;
    } else if constexpr (Intf::Config::cType::format == ConvolutionBackprop::CubeFormat::NDHWC) {
        out2L1Params.out2A1SrcAddr = out2A1SrcAddrStart + startWo * self->ctx.tiling_->cout;
    }

    //B矩阵考虑卷积操作，导致前后两个split交叠问题；
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
        self->ctx.load3d_.l1W = out2L1Params.singleShapeWi - self->ctx.load3d_.padList[0] - self->ctx.load3d_.padList[1];
    } else {
        self->ctx.load3d_.l1W = 0;
    }
}

template <class Intf>
__aicore__ inline void calculateWoIterTimes(Intf *self, int32_t &woIterateTimes, const int32_t splitWo) {
    if (splitWo == 0) {
        woIterateTimes = 1;
        return;
    }
    woIterateTimes = Ceil(self->ctx.tiling_->wo, splitWo);
}

template <class Intf>
__aicore__ inline void updateSingleShapeWoI(Intf *self, Out2L1ScalarParams& out2L1Params, const int32_t woIterateTimes,
                                            const int32_t splitWoIdx, const int32_t splitWo) {
    if (woIterateTimes > 1) {
        if ((splitWoIdx + 1) == woIterateTimes) {
            self->ctx.singleShapeWo_ = self->ctx.tiling_->wo - splitWoIdx * splitWo;
        } else {
            self->ctx.singleShapeWo_ = splitWo;
        }
        // 包含pad等在内，所以singleShapeWi可能大于wi。关注特殊case，当singleShapeWi > wi时是否能够正常运行；
        out2L1Params.singleShapeWi = self->ctx.singleShapeWo_ * self->ctx.tiling_->strideW + self->ctx.strideKernelDilationW;
    } else {
        self->ctx.singleShapeWo_ = self->ctx.tiling_->wo;
        uint64_t singleShapeWi = self->ctx.singleShapeWo_ * self->ctx.tiling_->strideW + self->ctx.strideKernelDilationW;
        out2L1Params.singleShapeWi = singleShapeWi;
    }
}

template <class Intf>
static __aicore__ inline void CalcParamsMmad(Intf* self) {
    self->ctx.mmad_.m = self->ctx.baseUseM_;
    self->ctx.mmad_.n = self->ctx.baseUseN_;
}

template <class Intf>
__aicore__ inline void ClearL0CLoad3dParams(Intf *self, LocalTensor<typename Intf::SrcT> &l0b) {
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
    LoadDataRepeatParamWithStride repeatParam = {0, 1, 0,
        static_cast<uint16_t>(ShiftCeilM0(self->ctx.tiling_->baseN, self->ctx.tiling_->n0))};
    SetLoadDataRepeatWithStride(repeatParam);
    LoadDataWithStride(l0b[0], self->ctx.cacheB1BufPing_, load3d);
#else
    LoadDataRepeatParam repeatParam = {0, 1, 0,
        static_cast<uint16_t>(ShiftCeilM0(self->ctx.tiling_->baseN, self->ctx.tiling_->n0))};
    SetLoadDataRepeat(repeatParam);
    LoadData(l0b[0], self->ctx.cacheB1BufPing_, load3d);
#endif
}

template <class Intf>
__aicore__ inline void ClearL0CLoad2dParams(Intf *self, LocalTensor<typename Intf::SrcT> &l0a) {
    LoadData2DParamsV2 load2d;
    load2d.mStartPosition = 0;
    load2d.kStartPosition = 0;
    load2d.mStep = Ceil(self->ctx.tiling_->baseM, self->ctx.tiling_->m0);
    if (IsSameType<typename Intf::SrcT, float>::value) {
        load2d.kStep = 2;    //fp32类型，kstep一定是2的倍数
    } else {
        load2d.kStep = 1;
    }
    load2d.srcStride = load2d.mStep;
    load2d.dstStride = load2d.mStep;
    load2d.ifTranspose = 0;
    LoadData(l0a[0], self->ctx.cacheA1BufPing_, load2d);
}

template <class Intf>
__aicore__ inline void ClearBaseMNL0C(Intf *self, LocalTensor<typename Intf::L0cT> &l0c) {
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
    if (mmad_.m * mmad_.n <2560) {
        PipeBarrier<PIPE_M>();
    }

    SetFlag<HardEvent::M_MTE1>(self->ctx.l0aPingPongFlag_);
    self->ctx.l0aPingPongFlag_ ^= self->ctx.useL0PingPong_;
}

__aicore__ inline void UpdateIdx(
    bool isLastStepKa, bool isLastStepKb, uint32_t& kaIdx, uint32_t& kbIdx,
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

}  // namespace ConvolutionBackpropFunc
#endif