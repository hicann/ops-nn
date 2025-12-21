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
 * \file conv_bp_large_kernel_func.h
 * \brief
 */

#ifndef CONV_BP_LARGE_KERNEL_FUNC_H
#define CONV_BP_LARGE_KERNEL_FUNC_H

#include "conv_bp_config_base.h"
#include "conv_bp_util.h"
#include "kernel_operator.h"
#include "../conv3d_backprop_filter_v2/conv3d_backprop_filter_v2_tiling_data.h"
#include "conv_bp_func_common.h"

namespace ConvolutionBackpropFunc {

template <class Intf>
__aicore__ inline void updateParasForSplitKernelHW(Intf *self, Out2L1ScalarParams& out2L1Params, uint32_t startWo,
                                    uint64_t out2B1SrcAddrStart, uint32_t wkIdx) {
    int32_t padLeft = 0;
    int32_t padRight = 0;
    int64_t b1SrcWiLeftOffGm = static_cast<int64_t>(startWo) * self->ctx.tiling_->strideW - self->ctx.tiling_->padLeft;
    if (b1SrcWiLeftOffGm < 0) {
        padLeft = -b1SrcWiLeftOffGm;
    }
    padLeft = padLeft - wkIdx * self->ctx.tiling_->dilationW;
    self->ctx.load3d_.padList[0] = padLeft < 0 ? (0) : (padLeft);

    int64_t b1SrcWiRightOffGm = static_cast<int64_t>(startWo + self->ctx.singleShapeWo_) * self->ctx.tiling_->strideW +
        self->ctx.strideKernelDilationW - (self->ctx.tiling_->padLeft + self->ctx.tiling_->wi);
    if (b1SrcWiRightOffGm > 0) {
        padRight = b1SrcWiRightOffGm;
    }
    padRight = padRight - (self->ctx.tiling_->wk - wkIdx - 1) * self->ctx.tiling_->dilationW;
    self->ctx.load3d_.padList[1] = padRight < 0 ? (0) : (padRight);

    if (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
        if (self->ctx.load3d_.padList[0]) {
            out2L1Params.out2B1SrcAddr = out2B1SrcAddrStart;
        } else {
            out2L1Params.out2B1SrcAddr = out2B1SrcAddrStart + b1SrcWiLeftOffGm + static_cast<uint64_t>(wkIdx) *
                                self->ctx.tiling_->dilationW;
        }
    } else if (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NDHWC) {
        if (self->ctx.load3d_.padList[0]) {
            out2L1Params.out2B1SrcAddr = out2B1SrcAddrStart;
        } else {
            out2L1Params.out2B1SrcAddr = out2B1SrcAddrStart + (b1SrcWiLeftOffGm + static_cast<uint64_t>(wkIdx) *
                                self->ctx.tiling_->dilationW) * self->ctx.tiling_->cin;
        }
    }

    out2L1Params.singleShapeWi = (self->ctx.singleShapeWo_ - 1) * self->ctx.tiling_->strideW + 1;
    if (out2L1Params.singleShapeWi > (self->ctx.load3d_.padList[0] + self->ctx.load3d_.padList[1])) {
        self->ctx.load3d_.l1W = out2L1Params.singleShapeWi - self->ctx.load3d_.padList[0] - self->ctx.load3d_.padList[1];
    } else {
        self->ctx.load3d_.l1W = 0;
    }
}

template <class Intf>
__aicore__ inline void initParasSplitKernelHW(Intf *self) {

    //矩阵计算的值，默认为baseUseN_，baseUseN_可能等于baseN或TailN，但都一定是n0对齐的,splitkernel场景，
    // 每次只循环一个hkwk=1,因此一定是n0
    self->ctx.mmad_.n = self->ctx.tiling_->n0;

    //kExtension是N轴，由于切了Kernel，不管是fp16和32(两个C0的Wk连续)场景均为16,正好一个n0单元;
    self->ctx.load3d_.kExtension = self->ctx.tiling_->n0;
    self->ctx.load3d_.kStartPt = 0; //stepM=stepN=1，每次N都是从0开始读取

    self->ctx.load3d_.channelSize = 16; //cin等于16，避免循环cin1
    self->ctx.load3d_.filterW = 1;
    self->ctx.load3d_.filterH = 1;

    self->ctx.load3d_.dilationFilterW = 1;
    self->ctx.load3d_.dilationFilterH = 1;

    self->ctx.load3d_.filterSizeW = false;
    self->ctx.load3d_.filterSizeH = false;
}

template <class Intf>
__aicore__ inline void ClearBaseMNL0C(Intf *self, LocalTensor<typename Intf::L0cT> &l0c) {
    LocalTensor<typename Intf::SrcT> l0a = self->ctx.l0aBuf_.template Get<typename Intf::SrcT>();
    LocalTensor<typename Intf::SrcT> l0b = self->ctx.l0bBuf_.template Get<typename Intf::SrcT>();

    LocalTensor<typename Intf::SrcT> useB1Buf = self->ctx.b1Ping_.template AllocTensor<typename Intf::SrcT>();
    InitZeroValue(self, useB1Buf);
    self->ctx.b1Ping_.EnQue(useB1Buf);
    LocalTensor<typename Intf::SrcT> useA1Buf = self->ctx.a1Ping_.template AllocTensor<typename Intf::SrcT>();
    InitZeroValue(self, useA1Buf);
    self->ctx.a1Ping_.EnQue(useA1Buf);

    self->ctx.cacheB1BufPing_ = self->ctx.b1Ping_.template DeQue<typename Intf::SrcT>();
    self->ctx.cacheA1BufPing_ = self->ctx.a1Ping_.template DeQue<typename Intf::SrcT>();

    using LoadData3DParamsV2SrcT = LoadData3DParamsV2<typename Intf::SrcT>;
    LoadData3DParamsV2SrcT load3d_;
    load3d_.padList[0] = 0;
    load3d_.padList[1] = 0;
    load3d_.padList[2] = 0;
    load3d_.padList[3] = 255;
    load3d_.l1W = 1;
    load3d_.l1H = 1;
    load3d_.channelSize = Ceil(self->ctx.tiling_->baseN, self->ctx.tiling_->n0) * self->ctx.tiling_->n0;
    load3d_.kStartPt = 0;
    load3d_.mStartPt = 0;
    load3d_.kExtension = self->ctx.tiling_->baseN;
    load3d_.mExtension = 16;
    load3d_.strideW = 1;
    load3d_.strideH = 1;
    load3d_.filterH = 1;
    load3d_.filterW = 1;
    load3d_.dilationFilterW = 1;
    load3d_.dilationFilterH = 1;

    LoadDataRepeatParam repeatParam = {0, 1, 0,
        static_cast<uint16_t>(ShiftCeilM0(self->ctx.tiling_->baseN, self->ctx.tiling_->n0))};
    SetLoadDataRepeat(repeatParam);
    LoadData(l0b[0], self->ctx.cacheB1BufPing_, load3d_);

    LoadData2DParamsV2 load2dv2_;
    load2dv2_.mStartPosition = 0;
    load2dv2_.kStartPosition = 0;
    load2dv2_.mStep = Ceil(self->ctx.tiling_->baseM, self->ctx.tiling_->m0);
    if (IsSameType<typename Intf::SrcT, float>::value) {
        load2dv2_.kStep = 2;    //fp32类型，kstep一定是2的倍数
    } else {
        load2dv2_.kStep = 1;
    }
    load2dv2_.srcStride = load2dv2_.mStep;
    load2dv2_.dstStride = load2dv2_.kStep;
    load2dv2_.ifTranspose = 1;
    LoadData(l0a[0], self->ctx.cacheA1BufPing_, load2dv2_);
    FreeB1Tensor(self, 1);
    FreeA1Tensor(self, 1);
    MmadParams mmad_;
    mmad_.m = self->ctx.tiling_->baseM;
    mmad_.n = self->ctx.tiling_->baseN;
    mmad_.k = 16;
    mmad_.cmatrixInitVal = true;

    TEventID eventId = GetTPipePtr()->FetchEventID<HardEvent::MTE1_M>();
    SetFlag<HardEvent::MTE1_M>(eventId);
    WaitFlag<HardEvent::MTE1_M>(eventId);
    Mmad(l0c[0], l0a[0], l0b[0], mmad_);
    if (mmad_.m * mmad_.n <2560) {
        PipeBarrier<PIPE_M>();
    }
}

template <class Intf>
__aicore__ inline void getHWkIdx(Intf *self, uint64_t hwkLoopIdx, uint64_t &hkIdx, uint64_t &wkIdx) {
    hkIdx = hwkLoopIdx / self->ctx.tiling_->wk;
    wkIdx = hwkLoopIdx % self->ctx.tiling_->wk;
}

__aicore__ inline void UpdateIdx(bool isLastStepKa, bool isLastStepKb, uint32_t &kaIdx, uint32_t &kbIdx,
                        uint64_t &kaStepIdx, uint64_t &kbStepIdx) {
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
static __aicore__ inline void LoadToB1Dn2NzSplitKernelHW(Intf *self, const uint32_t hiCopyLen, const uint32_t wiCopyLen,
    const uint64_t out2B1SrcAddrOffset, const Out2L1ScalarParams &params, LocalTensor<typename Intf::SrcT> &useB1Buf)
{
    Dn2NzParams dn2NzParams;
    dn2NzParams.dnNum = hiCopyLen;
    dn2NzParams.nValue = params.singleShapeWi - self->ctx.load3d_.padList[0] - self->ctx.load3d_.padList[1];
    dn2NzParams.dValue = self->ctx.bL1cin1CopyLen;
    dn2NzParams.srcDValue = self->ctx.dhwI_;
    dn2NzParams.srcDnMatrixStride = self->ctx.tiling_->wi;

    dn2NzParams.dstNzC0Stride = dn2NzParams.nValue * dn2NzParams.dnNum;
    dn2NzParams.dstNzNStride = 1;
    dn2NzParams.dstNzMatrixStride = dn2NzParams.nValue * self->ctx.tiling_->k0;
    self->ctx.load3d_.l1W = dn2NzParams.nValue;
    InitZeroValue(self, useB1Buf);
    DataCopy(useB1Buf, self->ctx.fmapGlobal_[out2B1SrcAddrOffset], dn2NzParams);
}

template <class Intf>
static __aicore__ inline void LoadToB1Nd2NzSplitKernelHW(Intf *self, const uint32_t hiCopyLen, const uint32_t wiCopyLen,
    const uint64_t out2B1SrcAddrOffset, const Out2L1ScalarParams &params, LocalTensor<typename Intf::SrcT> &useB1Buf)
{
    Nd2NzParams nd2NzParams;
    nd2NzParams.ndNum = hiCopyLen;
    nd2NzParams.srcNdMatrixStride = static_cast<uint64_t>(self->ctx.tiling_->wi) * self->ctx.tiling_->cin;

    nd2NzParams.nValue = params.singleShapeWi - self->ctx.load3d_.padList[0] - self->ctx.load3d_.padList[1];
    nd2NzParams.dValue = self->ctx.bL1cin1CopyLen;

    nd2NzParams.srcDValue = self->ctx.tiling_->cin;
    nd2NzParams.dstNzC0Stride = static_cast<uint64_t>(nd2NzParams.nValue) * nd2NzParams.ndNum;
    nd2NzParams.dstNzNStride = 1;
    nd2NzParams.dstNzMatrixStride = static_cast<uint64_t>(nd2NzParams.nValue) * self->ctx.tiling_->k0;
    self->ctx.load3d_.l1W = nd2NzParams.nValue;
    InitZeroValue(self, useB1Buf);
    DataCopy(useB1Buf, self->ctx.fmapGlobal_[out2B1SrcAddrOffset], nd2NzParams);
}

template <class Intf, class src1_T>
__aicore__ inline void LoadToB1SplitKernelHW(Intf *self, bool cachePosB1, const Out2L1ScalarParams& params,
                    bool isLoadB1, uint64_t kbStepIdx, uint64_t hkIdx, uint32_t startWo, bool &skipCurrentHiCompute)
{
    if (!isLoadB1) {
        return;
    }
    skipCurrentHiCompute = false;
    // 需要载入BL1的条件为，被计算的BL0块是BL1上的第一块数据，一次载入完整BL1大小
    // 此时满足以下条件之一需要载入BL1：
    // 1.BL1上无db，并且K方向需要多于一个buffer，每次都需要载入；BL1开db，并且K方向buffer数量小于等于2
    // 2.singleShapeK / stepKb > 2, 优先循环k方向，BL1上数据无法复用
    // 3.order_M时，L1上驻留AL1, BL1数据不复用
    // 4.order_N时，BL1驻留在L1上，且K <=
    // 2，即L1上可以栽下全部Kb，此时遍历M方向，BL1数据上数据不会被覆盖，只在M方向循环第一次时载入BL1
    if (params.isLoad2L1B) {
        // L0shape到orgShape的对应关系，L0和L1是16对齐的，orgShape是Wi对齐的,先算Wo对齐再算Wi对齐
        // 先算L0B所在BL1块的起始地址，16对齐的
        uint64_t b1SrcKAlign = static_cast<uint64_t>(kbStepIdx) * self->ctx.kbl1_;
        // load3d必须有完整Wo，做Wo对齐，计算起始地址所在的Ho
        uint32_t b1SrcHo = b1SrcKAlign / self->ctx.singleShapeWo_;
        uint32_t b1SrcHoGm = b1SrcHo + self->ctx.hoStartIdx_;
        // 计算Ho对应的Hi，根据卷积原理
        int64_t b1SrcHiGm = static_cast<uint64_t>(b1SrcHoGm) * self->ctx.tiling_->strideH +
                    static_cast<uint64_t>(hkIdx) * self->ctx.tiling_->dilationH - self->ctx.tiling_->padUp;
        uint32_t b1SrcHi = 0;
        if (b1SrcHiGm > 0 && self->ctx.hiStartIdx_ > 0) {
            b1SrcHi = b1SrcHiGm - self->ctx.hiStartIdx_;
        } else if (b1SrcHiGm > 0) {
            b1SrcHi = b1SrcHiGm;
        }

        uint32_t kbl1 = self->ctx.kbl1_;
        if (self->ctx.stepKbRound == (kbStepIdx + 1)) {
            kbl1 = self->ctx.singleShapeHo_ * self->ctx.singleShapeWo_ - b1SrcKAlign;
        }

        uint32_t ho = CalRows2Copy(kbl1, self->ctx.singleShapeWo_);
        uint32_t hiCopyLen = (ho - 1) * self->ctx.tiling_->strideH + 1; //hk采用循环，每次只循环一个hk,filterH=1

        //当拷贝的行完全处于padUp部分或是padDown部分时跳出搬运, 注意此处的hicopyLen保含pad行
        if ((b1SrcHiGm + hiCopyLen < 0) || (b1SrcHiGm >= self->ctx.tiling_->hi)) {
            skipCurrentHiCompute = true;
            return ;
        }
        uint32_t padUp = 0;
        if (b1SrcHiGm < 0) {
            hiCopyLen = hiCopyLen + b1SrcHiGm;
            if (hiCopyLen > self->ctx.tiling_->hi) {
                hiCopyLen = self->ctx.tiling_->hi;
            }
            //起始地址计算
            padUp = -b1SrcHiGm;
        }
        if (b1SrcHiGm + hiCopyLen > self->ctx.tiling_->hi) {    //上面一个if保证了当b1SrcHiGm < 0时, b1SrcHiGm + hiCopyLen一定小于hi
            hiCopyLen = self->ctx.tiling_->hi - b1SrcHiGm;
        }

        uint32_t hiRemainLen = params.singleShapeHi - b1SrcHi;
        hiCopyLen = hiRemainLen > hiCopyLen ? hiCopyLen: hiRemainLen;
        if (hiCopyLen == 0) {
            skipCurrentHiCompute = true;
            return ;
        }
        LocalTensor<typename Intf::SrcT> useB1Buf;
        if (cachePosB1) {
            useB1Buf = self->ctx.b1Ping_.template AllocTensor<typename Intf::SrcT>();
        } else {
            useB1Buf = self->ctx.b1Pong_.template AllocTensor<typename Intf::SrcT>();
        }
        // 得到gm的偏移量
        uint64_t out2B1SrcAddrOffset = 0;
        if constexpr (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
            out2B1SrcAddrOffset = params.out2B1SrcAddr + static_cast<uint64_t>(b1SrcHi) * self->ctx.tiling_->wi;
        } else if constexpr (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NDHWC) {
            out2B1SrcAddrOffset = params.out2B1SrcAddr + static_cast<uint64_t>(b1SrcHi) *
                            self->ctx.tiling_->wi * self->ctx.tiling_->cin;
        }

        if constexpr (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
            LoadToB1Dn2NzSplitKernelHW(self, hiCopyLen, 0, out2B1SrcAddrOffset, params, useB1Buf);
        } else if constexpr (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NDHWC) {
            LoadToB1Nd2NzSplitKernelHW(self, hiCopyLen, 0, out2B1SrcAddrOffset, params, useB1Buf);
        }

        if (cachePosB1) {
            self->ctx.bL1HiCopyLenPing = hiCopyLen;
            self->ctx.bL1PadUpPing = padUp;
            self->ctx.b1Ping_.EnQue(useB1Buf);
        } else {
            self->ctx.bL1HiCopyLenPong = hiCopyLen;
            self->ctx.bL1PadUpPong = padUp;
            self->ctx.b1Pong_.EnQue(useB1Buf);
        }
    }
}

template <class Intf>
__aicore__ inline void ComputeSplitKernelHW(Intf *self, Out2L1ScalarParams& out2L1Params)
{
    if ASCEND_IS_AIV {
        return;
    }
    LocalTensor<typename Intf::L0cT> l0c;
    if (self->ctx.l0cPingPongFlag_) {
        l0c = self->ctx.l0cPing_.template AllocTensor<typename Intf::L0cT>();
    } else {
        l0c = self->ctx.l0cPong_.template AllocTensor<typename Intf::L0cT>();
    }
    ClearBaseMNL0C<Intf>(self, l0c);

    CalcParamsL12L0a<Intf>(self);
    //todo: baseUseM_=1会走到特殊的处理分支，使用Gemv实现mmad,当前按照m0进行对其，走通用分支使用GEMM，后续需要评估特殊分支是否有性能收益决定特殊分支是否有必要存在；
    uint32_t baseUseMBak = self->ctx.baseUseM_;
    if (self->ctx.baseUseM_ == 1) {
        self->ctx.baseUseM_ = ShiftCeilM0(self->ctx.baseUseM_, self->ctx.tiling_->m0) * self->ctx.tiling_->m0;
        self->ctx.mmad_.m = self->ctx.baseUseM_;
        CalcParamsL12L0a<Intf>(self);
    }
    LocalTensor<typename Intf::SrcT> l0a;
    LocalTensor<typename Intf::SrcT> l0b;

    constexpr uint32_t l0aPingPongAddr = TOTAL_L0A_SIZE / 2 / sizeof(typename Intf::SrcT);
    constexpr uint32_t l0bPingPongAddr = TOTAL_L0B_SIZE / 2 / sizeof(typename Intf::SrcT);

    CalcParamsL12L0b<Intf>(self);
    CalcParamsMmad<Intf>(self);
    CalOut2L1ScalarParams(self, out2L1Params);
    bool isFirstMmad = true;
    uint64_t dstL0cOffsetBase = self->ctx.dstL0cOffset_;
    uint64_t usedN = self->ctx.curNL1Idx_ * self->ctx.tiling_->baseN;   //基本块模板中stepN一定等于1，此处使用curNL1Idx和curNL0Idx均可
    uint64_t hwkLoopStart = usedN / self->ctx.tiling_->n0;
    uint64_t hwkLoopEnd = (usedN + self->ctx.baseUseN_) / self->ctx.tiling_->n0;
    uint64_t hkIdx = 0;
    uint64_t wkIdx = 0;
    for (uint64_t hwkLoopIdx = hwkLoopStart; hwkLoopIdx < hwkLoopEnd; hwkLoopIdx++) {
        getHWkIdx(self, hwkLoopIdx, hkIdx, wkIdx);
        self->ctx.dstL0cOffset_ = dstL0cOffsetBase +
            (hwkLoopIdx - hwkLoopStart) * self->ctx.tiling_->baseM * self->ctx.tiling_->n0;
        initParasSplitKernelHW(self);
        isFirstMmad = true;

        uint64_t out2A1BatchDoutSrcAddrStart = out2L1Params.out2A1SrcAddr;
        uint64_t out2B1BatchDoutSrcAddrStart = out2L1Params.out2B1SrcAddr;
        uint64_t batchDoutEndIdx = self->ctx.batchDoutStartIdx_ + self->ctx.singleShapeBatch_;
        for (uint64_t batchDoutIdx = self->ctx.batchDoutStartIdx_; batchDoutIdx < batchDoutEndIdx; batchDoutIdx++) {
            bool skipCurrentDinCompute = false; // dinIdx小于padFront或大于din+padFront则跳过本轮计算
            UpdateSrcAddrBaseOnBatchDoutIdx<Intf>(self, batchDoutIdx, out2L1Params, skipCurrentDinCompute);
            if (skipCurrentDinCompute) {
                continue;
            }

            const int32_t splitWo = self->ctx.tiling_->splitWo;
            uint64_t out2A1SrcAddrStart = out2L1Params.out2A1SrcAddr;
            uint64_t out2B1SrcAddrStart = out2L1Params.out2B1SrcAddr;

            int32_t woIterateTimes = 1;
            calculateWoIterTimes(self, woIterateTimes, splitWo);
            for (int32_t splitWoIdx = 0; splitWoIdx < woIterateTimes; splitWoIdx++) {
                updateSingleShapeWoI(self, out2L1Params, woIterateTimes, splitWoIdx, splitWo);
                if (self->ctx.isSplitWo_) {
                    updateParasForSplitW(self, out2L1Params, splitWoIdx * splitWo, out2A1SrcAddrStart, out2B1SrcAddrStart);
                }
                updateParasForSplitKernelHW(self, out2L1Params, splitWoIdx * splitWo, out2B1SrcAddrStart, wkIdx);
                if (!self->ctx.load3d_.l1W) {
                    PipeBarrier<PIPE_ALL>();
                    continue ;
                }
                bool a1PingPongFlag = true;
                bool b1PingPongFlag = true;
                bool isAL1PingPong = self->ctx.tiling_->al1Pbuffer > 1;
                bool isBL1PingPong = self->ctx.tiling_->bl1Pbuffer > 1;
                uint32_t kaIdx = 0;
                uint32_t kbIdx = 0;
                uint64_t kaStepIdx = 0;
                uint64_t kbStepIdx = 0;
                uint64_t curMKL1Idx = self->ctx.stepKaRound * DivStepM(self->ctx.curML1Idx_, self->ctx.tiling_->stepM);
                uint64_t curNKL1Idx = self->ctx.stepKbRound * DivStepN(self->ctx.curNL1Idx_, self->ctx.tiling_->stepN);

                bool skipCurrentHiCompute = false;
                for (uint64_t k = 0; k < self->ctx.kIter_; k++) {
                    bool isLastKIter = k + 1 == self->ctx.kIter_;
                    bool isLastStepKa = kaIdx + 1 == self->ctx.tiling_->stepKa;
                    bool isLastStepKb = kbIdx + 1 == self->ctx.tiling_->stepKb;
                    bool isLoadA1 = kaIdx == 0;
                    bool isLoadB1 = kbIdx == 0;
                    self->ctx.baseUseK_ = isLastKIter ? self->ctx.tailK_ : self->ctx.tiling_->baseK;

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
                    LoadToB1SplitKernelHW<Intf, typename Intf::SrcT>(self, b1PingPongFlag,
                                    out2L1Params, isLoadB1, kbStepIdx, hkIdx, splitWoIdx * splitWo, skipCurrentHiCompute);
                    if (skipCurrentHiCompute) {
                        UpdateIdx(isLastStepKa, isLastStepKb, kaIdx, kbIdx, kaStepIdx, kbStepIdx);
                        continue;
                    }
                    if (isAL1PingPong) {
                        a1PingPongFlag = (curMKL1Idx + kaStepIdx + 1) & 1;
                    }
                    ConvolutionBackpropFunc::LoadToA1<Intf, typename Intf::SrcT>(self, a1PingPongFlag, k,
                                                                                out2L1Params, isLoadA1, kaStepIdx);

                    WaitFlag<HardEvent::M_MTE1>(self->ctx.l0aPingPongFlag_ & 1);

                    l0b = self->ctx.l0bBuf_.template Get<typename Intf::SrcT>();
                    if (self->ctx.l0aPingPongFlag_) {
                        l0b = l0b[l0bPingPongAddr];
                    }
                    // posM
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
                        LoadL12L0a<Intf>(self, self->ctx.cacheA1BufPing_, k, l0a);
                    } else {
                        LoadL12L0a<Intf>(self, self->ctx.cacheA1BufPong_, k, l0a);
                    }

                    if (out2L1Params.isFreeAL1 && (isLastStepKa || isLastKIter)) {
                        FreeA1Tensor(self, a1PingPongFlag);
                    }

                    SetFlag<HardEvent::MTE1_M>(self->ctx.l0aPingPongFlag_);
                    WaitFlag<HardEvent::MTE1_M>(self->ctx.l0aPingPongFlag_);
                    // isFirstMmadd为true时，compute还没有被完整执行过，此时dkIdx第一次滑窗到din的有效位置上进行mmad计算，因此重置L0C
                    self->ctx.mmad_.cmatrixInitVal = isFirstMmad;
                    self->ctx.mmad_.k = self->ctx.baseUseK_;
                    MmadLocal<Intf>(self, l0a, l0b, l0c);
                    isFirstMmad = false;
                    SetFlag<HardEvent::M_MTE1>(self->ctx.l0aPingPongFlag_);

                    self->ctx.l0aPingPongFlag_ ^= self->ctx.useL0PingPong_;
                    UpdateIdx(isLastStepKa, isLastStepKb, kaIdx, kbIdx, kaStepIdx, kbStepIdx);
                }
            }
            out2L1Params.out2A1SrcAddr = out2A1SrcAddrStart;
            out2L1Params.out2B1SrcAddr = out2B1SrcAddrStart;
        }
        //batchout 偏移后的地址要还原回来
        out2L1Params.out2A1SrcAddr = out2A1BatchDoutSrcAddrStart;
        out2L1Params.out2B1SrcAddr = out2B1BatchDoutSrcAddrStart;
    }
    self->ctx.dstL0cOffset_ = dstL0cOffsetBase;
    self->ctx.baseUseM_ = baseUseMBak;
    if (self->ctx.l0cPingPongFlag_) {
        self->ctx.l0cPing_.EnQue(l0c);
    } else {
        self->ctx.l0cPong_.EnQue(l0c);
    }
}

}  // namespace ConvolutionBackpropFunc
#endif