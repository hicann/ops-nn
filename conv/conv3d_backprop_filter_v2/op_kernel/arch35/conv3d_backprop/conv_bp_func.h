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
 * \file conv_bp_func.h
 * \brief
 */

#ifndef CONV_BP_FUNC_H
#define CONV_BP_FUNC_H

#include "conv_bp_config_base.h"
#include "conv_bp_util.h"
#include "basic_api/kernel_basic_intf.h"
#include "../conv3d_backprop_filter_v2/conv3d_backprop_filter_v2_tiling_data.h"
#include "impl/conv_bp_filter_sub_func.h"
#include "impl/conv_bp_sub_func_deterministic_calculate.h"
#include "conv_bp_large_kernel_func.h"
#include "conv_bp_func_common.h"

DECLARE_CHECK_IMPL(Init);
DECLARE_CHECK_IMPL(SetFmap);
DECLARE_CHECK_IMPL(SetOutBackprop);
DECLARE_CHECK_IMPL(SetSingleShape);
DECLARE_CHECK_IMPL(SetSingleShapeK);
DECLARE_CHECK_IMPL(SetStartIdx);
DECLARE_CHECK_IMPL(SetDeterministicCoreInfo);
DECLARE_CHECK_SYNC_IMPL(DeterministicReduceKInUb);
DECLARE_CHECK_SYNC_IMPL(UpdateMNIdx);
DECLARE_CHECK_SYNC_IMPL(Compute);
DECLARE_CHECK_SYNC_IMPL(Iterate);
DECLARE_CHECK_SYNC_IMPL(IterateAll);
DECLARE_CHECK_SYNC_IMPL(GetTensorC);
DECLARE_CHECK_SYNC_IMPL(VecPostProcess);
DECLARE_CHECK_IMPL(End);

namespace ConvolutionBackpropFunc {
constexpr uint8_t FLAG_FIXP_ID = 5;
constexpr uint8_t FLAG_MTE3_ID = 6;

using TypeFalse = struct {
    __uint128_t _[1024];
};

enum class IterateOrder {
    ORDER_M = 0,
    ORDER_N,
    UNDEF,
};

template <class Intf>
__aicore__ inline void ComputeNormal(Intf* self, Out2L1ScalarParams& out2L1Params)
{
    if ASCEND_IS_AIV {
        return;
    }

    CalcParamsL12L0a<Intf>(self);
    CalcParamsL12L0b<Intf>(self);
    CalcParamsMmad<Intf>(self);
    LocalTensor<typename Intf::SrcT> l0a;
    LocalTensor<typename Intf::SrcT> l0b;
    LocalTensor<typename Intf::L0cT> l0c;
    constexpr uint32_t l0aPingPongAddr = TOTAL_L0A_SIZE / 2 / sizeof(typename Intf::SrcT);
    constexpr uint32_t l0bPingPongAddr = TOTAL_L0B_SIZE / 2 / sizeof(typename Intf::SrcT);
    CalOut2L1ScalarParams(self, out2L1Params);

    if (self->ctx.l0cPingPongFlag_) {
        l0c = self->ctx.l0cPing_.template AllocTensor<typename Intf::L0cT>();
    } else {
        l0c = self->ctx.l0cPong_.template AllocTensor<typename Intf::L0cT>();
    }
    uint64_t batchDoutEndIdx = self->ctx.batchDoutStartIdx_ + self->ctx.singleShapeBatch_;
    bool isFirstMmad = true;

    // todo:
    // baseUseM_=1会走到特殊的处理分支，使用Gemv实现mmad,当前按照m0进行对其，走通用分支使用GEMM，后续需要评估特殊分支是否有性能收益决定特殊分支是否有必要存在；
    uint32_t baseUseMBak = self->ctx.baseUseM_;
    if (self->ctx.baseUseM_ == 1) {
        self->ctx.baseUseM_ = ShiftCeilM0(self->ctx.baseUseM_, self->ctx.tiling_->m0) * self->ctx.tiling_->m0;
        self->ctx.mmad_.m = self->ctx.baseUseM_;
        CalcParamsL12L0a<Intf>(self);
    }
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
            if (unlikely(self->ctx.isSplitWo_)) {
                updateParasForSplitW(self, out2L1Params, splitWoIdx * splitWo, out2A1SrcAddrStart, out2B1SrcAddrStart);
            }
            if (!self->ctx.load3d_.l1W) {
                continue;
            }
            bool a1PingPongFlag = true;
            bool b1PingPongFlag = true;
            bool isAL1PingPong = self->ctx.tiling_->al1Pbuffer > 1;
            bool isBL1PingPong = self->ctx.tiling_->bl1Pbuffer > 1;
            uint32_t kaIdx = 0;
            uint32_t kbIdx = 0;
            uint64_t kaStepIdx = 0;
            uint64_t kbStepIdx = 0;
            uint64_t curMKL1Idx = self->ctx.stepKaRound * self->ctx.curMIdx_;
            uint64_t curNKL1Idx = self->ctx.stepKbRound * self->ctx.curNIdx_;

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
                if (isLoadB1) {
                    ConvolutionBackpropFunc::LoadToB1<Intf, typename Intf::SrcT>(self, b1PingPongFlag, out2L1Params,
                                                                                 kbStepIdx, skipCurrentHiCompute);
                }
                if (skipCurrentHiCompute) {
                    UpdateIdx(isLastStepKa, isLastStepKb, kaIdx, kbIdx, kaStepIdx, kbStepIdx);
                    continue;
                }
                if (isAL1PingPong) {
                    a1PingPongFlag = (curMKL1Idx + kaStepIdx + 1) & 1;
                }
                ConvolutionBackpropFunc::LoadToA1<Intf, typename Intf::SrcT>(self, a1PingPongFlag, k, out2L1Params,
                                                                             isLoadA1, kaStepIdx);

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
    if (isFirstMmad == true) {
        ClearBaseMNL0C<Intf>(self, l0c);
    }
    self->ctx.baseUseM_ = baseUseMBak;
    if (self->ctx.l0cPingPongFlag_) {
        self->ctx.l0cPing_.EnQue(l0c);
    } else {
        self->ctx.l0cPong_.EnQue(l0c);
    }
}

template <class Intf, bool sync>
struct Compute {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf* self, Out2L1ScalarParams& out2L1Params)
    {
        self->ctx.baseUseM_ = (self->ctx.curMIdx_ + 1 == self->ctx.mIter_) ? self->ctx.tailM_ :
                                                                             self->ctx.tiling_->baseM;
        // 由于N循环可能包含cinhkwk循环和dk循环，因此，self->ctx.baseUseN_的判断条件需改变
        self->ctx.baseUseN_ = ((self->ctx.curNIdx_ + 1) % self->ctx.cinHkWkLoop_ == 0) ? self->ctx.tailN_ :
                                                                                         self->ctx.tiling_->baseN;
        if constexpr (!Intf::conv3ddwConfig.isSplitKernelHW) {
            ComputeNormal<Intf>(self, out2L1Params);
        } else {
            ComputeSplitKernelHW<Intf>(self, out2L1Params);
        }
    }
};

template <class Intf>
struct Init {
    // 定义call函数的默认重载函数，支持任意类型任意数量的参数
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf* self, const AscendC::conv_bp_v2_kernel::TConv3DDwTiling* __restrict tiling)
    {
        self->ctx.tiling_ = tiling;
        CheckTiling<Intf>(self);
        InitParams<Intf>(self);
        InitTque<Intf>(self);
        if (self->ctx.tiling_->hf32Flag) {
            SetHF32Mode(true);
        }
    }
};

template <class Intf>
struct SetFmap {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf* self, const GlobalTensor<typename Intf::SrcT>& fmap)
    {
        self->ctx.fmapGlobal_ = fmap;
    }
};

template <class Intf>
struct SetOutBackprop {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf* self, const GlobalTensor<typename Intf::SrcT>& outBackprop)
    {
        self->ctx.outBackPropGlobal_ = outBackprop;
    }
};

template <class Intf>
struct SetSingleShapeK {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf* self, uint64_t singleShapeK)
    {
        self->ctx.singleShapeHo_ = singleShapeK / self->ctx.tiling_->wo;
        InitStepKParams<Intf>(self);
    }
};

template <class Intf>
struct SetSingleShape {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf* self, uint64_t singleShapeM, uint64_t singleShapeN, uint64_t singleShapeK,
                                       uint32_t singleShapeCin, uint32_t singleShapeBatch)
    {
        self->ctx.singleShapeCout_ = singleShapeM;
        self->ctx.singleShapeBatch_ = singleShapeBatch;
        self->ctx.singleShapeCin_ = DivHkWk(singleShapeN, self->ctx.hwK_);
        if (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NDC1HWC0) {
            self->ctx.singleShapeCin_ = ShiftCeilChannelSize<Intf>(self->ctx.singleShapeCin_,
                                                                   self->ctx.tiling_->channelSize) *
                                        self->ctx.tiling_->channelSize;
        } else {
            // self->ctx.singleShapeCin_包含singleShapeCin和singleShapeDk
            self->ctx.singleShapeDk_ = Ceil(self->ctx.singleShapeCin_, singleShapeCin);
            self->ctx.singleShapeCin_ = singleShapeCin;
        }

        InitStepMParams<Intf>(self);
        self->SetSingleShapeK(singleShapeK);
        InitStepNParams<Intf>(self);
        // 每次iterall之前都需将如下变量初始化赋值一次，否则会出现尾块精度问题
        self->ctx.head_ = 0;
        self->ctx.tail_ = 0;
        self->ctx.nLoopHead_ = 0;
        self->ctx.lastNIdx_ = -1;
        self->ctx.cinRemainLen_ = self->ctx.singleShapeCin_;
        self->ctx.nLoopCinRemainLen_ = self->ctx.singleShapeCin_;
    }
};

template <class Intf>
struct SetStartIdx {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf* self, uint64_t batchDoutIdx, uint32_t hoStartIdx, int32_t dkIdx)
    {
        self->ctx.batchDoutStartIdx_ = batchDoutIdx;
        self->ctx.hoStartIdx_ = hoStartIdx;
        self->ctx.hiStartIdx_ = static_cast<int32_t>(hoStartIdx) * self->ctx.tiling_->strideH -
                                self->ctx.tiling_->padUp;
        self->ctx.dkStartIdx_ = dkIdx;
    }
};

template <class Intf, bool sync>
struct UpdateMNIdx {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline bool call(Intf* self, Out2L1ScalarParams& out2L1Params)
    {
        /*
        |   <---------singleShapeM------->        |
        |  <---L1A_ping--->  |  <---L1A_pong--->  |
        |_L0A1_|_L0A2_|_L0A3_|_L0A4_|_L0A5_|_L0A6_|
        ↑  <--curStepM_-->    |                    ↑
        curML0Idx_            ↑                  mIter_
        curML1Idx_        next_curML1Idx

        |   <---------singleShapeN------->        |
        |  <---L1B_ping--->  |  <---L1B_pong--->  |
        |_L0B1_|_L0B2_|_L0B3_|_L0B4_|_L0B5_|_L0B6_|
        ↑  <--curStepN_-->    |                    ↑
        curNL0Idx_            ↑                   nIter_
        curNL1Idx_       next_curNL1Idx

        order_N表示L1上驻留B循环A，顺序为L1A_ping * L1B_ping, L1A_pong * L1B_ping，L1A_ping * L1B_pong，L1A_pong *
        L1B_pong L0上也是驻留B，循环A order_N: L0A1*L0B1, L0A2*L0B1, L0A3*L0B1, L0A1*L0B2 …………
        L0A3*L0B3，L0A4*L0B1，L0A5*L0B1 …… L0A6*L0B6 order_M: L0A1*L0B1, L0A1*L0B2, L0A1*L0B3, L0A2*L0B1 …………
        L0A3*L0B3，L0A1*L0B4，L0A1*L0B5 …… L0A6*L0B6
        */
        // 更新idx，用L1、L1step、L0三个指针控制走位和计算offset，表示计算第几个mL0 * baseN

        // 以开DB为例，循环顺序NMK时，如果K方向需要两块buffer，M轴每循环到stepM最后一次时需要释放AL1上的ping pong
        // buffer，第一次循环时载入 如果循环顺序为MNK时，如果K方向需要两块buffer，M轴最后一次循环时需要释放AL1上的ping
        // pong buffer，第一次循环时载入 如果K方向需要的buffer数量大于bl1Pbuffer，当K循环到stepKa时就需要置换AL1
        // B矩阵计算思路同A矩阵，区别是MN反过来

        if (unlikely(self->ctx.isFirstIter_)) {
            UpdateMNIdxFirstIterate<Intf>(self, out2L1Params);
            return true;
        }

        // 当singleShapeBatch大于1时，batchDout循环内移动，L1不驻留数据;当启动splitWo时,L1不能驻留数据
        if constexpr (Intf::conv3ddwConfig.isSplitKernelHW) {
            out2L1Params.isLoad2L1A = true;
            out2L1Params.isFreeAL1 = true;
            out2L1Params.isLoad2L1B = true;
            out2L1Params.isFreeBL1 = true;
        } else if (unlikely(self->ctx.isSplitWo_)) {
            out2L1Params.isLoad2L1A = true;
            out2L1Params.isFreeAL1 = true;
            out2L1Params.isLoad2L1B = true;
            out2L1Params.isFreeBL1 = true;
        } else {
            bool kIterCeilStepKaGreaterAl1Pbuffer = self->ctx.kIter_ >
                                                    self->ctx.tiling_->stepKa * self->ctx.tiling_->al1Pbuffer;
            bool kIterCeilStepKbGreaterBl1Pbuffer = self->ctx.kIter_ >
                                                    self->ctx.tiling_->stepKb * self->ctx.tiling_->bl1Pbuffer;
            out2L1Params.isLoad2L1A = kIterCeilStepKaGreaterAl1Pbuffer || self->ctx.singleShapeBatch_ > 1;
            out2L1Params.isFreeAL1 = out2L1Params.isLoad2L1A;
            out2L1Params.isLoad2L1B = kIterCeilStepKbGreaterBl1Pbuffer || self->ctx.singleShapeBatch_ > 1;
            out2L1Params.isFreeBL1 = out2L1Params.isLoad2L1B;
        }

        if (likely(self->ctx.tiling_->iterateOrder == static_cast<int>(IterateOrder::ORDER_N))) {
            return UpdateMNIdxOrderN<Intf>(self, out2L1Params);
        } else { // order_M
            return UpdateMNIdxOrderM<Intf>(self, out2L1Params);
        }
        return true;
    }
};

template <class Intf, bool sync>
struct Iterate {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline bool call(Intf* self, bool enPartialSum)
    {
        Out2L1ScalarParams out2L1Params;
        if (!self->template UpdateMNIdx<sync>(out2L1Params)) {
            return false;
        }
        self->template Compute(out2L1Params);
        return true;
    }
};

template <class Intf, bool sync>
struct IterateAll {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf* self, const GlobalTensor<typename Intf::DstT>& output, uint8_t enAtomic)
    {
        if constexpr (!Intf::conv3ddwConfig.groupEnlarge) {
            while (self->template Iterate<sync>()) {
                self->template GetTensorC<sync>(output, enAtomic);
            }
        } else {
            while (self->template Iterate<sync>()) {
                if ASCEND_IS_AIC {
                    CrossCoreWaitFlag<SYNC_MODE4, PIPE_FIX>(FLAG_FIXP_ID);
                    self->template GetTensorC<sync>(output, enAtomic);
                    CrossCoreSetFlag<SYNC_MODE4, PIPE_FIX>(FLAG_MTE3_ID);
                }

                if ASCEND_IS_AIV {
                    CrossCoreSetFlag<SYNC_MODE4, PIPE_MTE3>(FLAG_FIXP_ID);
                    CrossCoreWaitFlag<SYNC_MODE4, PIPE_V>(FLAG_MTE3_ID);
                    self->template VecPostProcess<sync>(output, enAtomic);
                }
            }
        }
        self->ctx.isFirstIter_ = true;
    }
};

template <class Intf, bool sync>
struct GetTensorC {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf* self, const GlobalTensor<typename Intf::DstT>& output,
                                       uint8_t enAtomic = 0, bool enSequentialWrite = false)
    {
        LoadL0c2Gm<Intf>(self, output, enAtomic, enSequentialWrite);
    }
};

template <class Intf, bool sync>
struct VecPostProcess {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf* self, const GlobalTensor<typename Intf::DstT>& output,
                                       uint8_t enAtomic = 0)
    {
        Rearrange2Gm<Intf>(self, output, enAtomic);
    }
};

template <class Intf>
struct End {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf* self)
    {
        self->ctx.a1Ping_.FreeAllEvent();
        if (self->ctx.tiling_->al1Pbuffer > 1) {
            self->ctx.a1Pong_.FreeAllEvent();
        }
        self->ctx.b1Ping_.FreeAllEvent();
        if (self->ctx.tiling_->bl1Pbuffer > 1) {
            self->ctx.b1Pong_.FreeAllEvent();
        }
        self->ctx.l0cPing_.FreeAllEvent();
        if (self->ctx.tiling_->cl0Pbuffer > 1) {
            self->ctx.l0cPong_.FreeAllEvent();
        }

        if (self->ctx.tiling_->hf32Flag) {
            SetHF32Mode(false);
        }
    }
};

template <class Intf>
struct SetDeterministicCoreInfo {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf* self, uint32_t addCoreNum, uint32_t addCoreIndex,
                                       uint32_t addCoreIndexTotal, bool isNoDeter)
    {
        self->ctx.deterAddCoreNum_ = addCoreNum;
        self->ctx.deterAddCoreIndex_ = addCoreIndex;
        self->ctx.coreStartIndexTotal_ = addCoreIndexTotal;
        self->ctx.isNoDeterministic_ = isNoDeter;
    }
};

template <class Intf, bool sync>
struct DeterministicReduceKInUb {
    DECLARE_DEFAULT_OVERLOADING_FUN(Intf, ConvolutionBackpropFunc);
    static __aicore__ inline void call(Intf* self, const GlobalTensor<typename Intf::DstT>& output,
                                       const GlobalTensor<typename Intf::DstT>& userGm)
    {
        // 确定性计算累加
        DeterministicAddFunc<Intf>(self, output, userGm);
    }
};

} // namespace ConvolutionBackpropFunc
#endif