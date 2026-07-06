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
 * \file conv_bp_filter_sub_func_load_gm_to_l1.h
 * \brief
 */

#ifndef CONV3D_BP_FILTER_SUB_FUNC_LOAD_GM_TO_L1_H
#define CONV3D_BP_FILTER_SUB_FUNC_LOAD_GM_TO_L1_H
#include "conv_bp_filter_sub_func_load_a1.h"
#include "conv_bp_filter_sub_func_load_b1.h"
#include "conv_bp_filter_sub_func_load_common.h"

namespace ConvolutionBackpropFunc {

template <class Intf>
static __aicore__ inline void CalOut2L1ScalarParams(Intf* self, Out2L1ScalarParams& params)
{
    // to L1A
    if (params.isLoad2L1A) {
        if constexpr (Intf::Config::cType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
            params.out2A1SrcAddr = static_cast<uint64_t>(self->ctx.curMIdx_) * self->ctx.tiling_->baseM *
                                   self->ctx.tiling_->dout * self->ctx.hwO_;
        } else {
            params.out2A1SrcAddr = static_cast<uint64_t>(self->ctx.curMIdx_) * self->ctx.tiling_->baseM;
        }
        params.isLastMAL1 = (self->ctx.mIter_ - 1) == self->ctx.curMIdx_;
    }

    // to L1B
    if (params.isLoad2L1B) {
        if ((self->ctx.tiling_->baseN >> 4) % self->ctx.hwK_ != 0) { // 4 用来替换除法运算
            CalOut2B1ParamsBaseNUndivided(self, params);
        } else {
            CalOut2B1Params(self, params);
        }
        uint64_t singleShapeHi = self->ctx.singleShapeHo_ * self->ctx.tiling_->strideH +
                                 self->ctx.strideKernelDilationH;
        params.singleShapeHi = self->ctx.tiling_->hi > singleShapeHi ? singleShapeHi : self->ctx.tiling_->hi;
    }
}

template <class Intf>
__aicore__ inline void UpdateSrcAddrBaseOnBatchDoutIdx(Intf* self, uint64_t curLoopBatchDoutIdx,
                                                       Out2L1ScalarParams& params, bool& skipCurrentDinCompute)
{
    int32_t curBatchIdx = curLoopBatchDoutIdx / self->ctx.tiling_->dout;
    int32_t curDoutIdx = curLoopBatchDoutIdx - curBatchIdx * self->ctx.tiling_->dout;
    uint64_t curDinIdx = static_cast<uint64_t>(curDoutIdx) * self->ctx.tiling_->strideD +
                         self->ctx.dkStartIdx_ * self->ctx.tiling_->dilationD;
    // 在拆分dk的情况下，如果涉及到的dinIdx在padding部分，则跳过Compute计算
    if (curDinIdx < self->ctx.tiling_->padFront || curDinIdx >= (self->ctx.tiling_->di + self->ctx.tiling_->padFront)) {
        // dinIdx当前在padding部分，因此跳过本轮mmad的计算。
        // 由于每次A, B矩阵的偏移计算依赖于前一次BatchDoutIdx迭代中的偏移，因此不能直接return跳过接下来偏移的计算
        skipCurrentDinCompute = true;
    }
    // 当batchLoopBatchDoutIdx为起始Idx时，无需更新地址
    if (curLoopBatchDoutIdx == self->ctx.batchDoutStartIdx_) {
        return;
    }
    // 更新batchIdx和doutIdx，重新计算offsetA和offsetB的值，优先循环dout方向；offsetC不需要更新
    int32_t preBatchIdx = curBatchIdx;
    int32_t preDoutIdx = curDoutIdx - 1;
    if (curDoutIdx == 0) {
        preBatchIdx--;
        preDoutIdx = (curLoopBatchDoutIdx - 1) - preBatchIdx * self->ctx.tiling_->dout;
    }

    int64_t batchOffsetAIncre = static_cast<int64_t>(curBatchIdx - preBatchIdx) * self->ctx.tiling_->cout *
                                self->ctx.tiling_->dout * self->ctx.tiling_->ho * self->ctx.tiling_->wo;
    int64_t doutOffsetAIncre = static_cast<int64_t>(curDoutIdx - preDoutIdx) * self->ctx.tiling_->ho *
                               self->ctx.tiling_->wo;
    if constexpr (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NDHWC) {
        doutOffsetAIncre = doutOffsetAIncre * self->ctx.tiling_->cout;
    }
    params.out2A1SrcAddr = params.out2A1SrcAddr + (batchOffsetAIncre + doutOffsetAIncre);

    int64_t batchOffsetBIncre = static_cast<int64_t>(curBatchIdx - preBatchIdx) * self->ctx.tiling_->cin *
                                self->ctx.tiling_->di * self->ctx.tiling_->hi * self->ctx.tiling_->wi;
    int64_t doutOffsetBIncre = static_cast<int64_t>(curDoutIdx - preDoutIdx) * self->ctx.tiling_->strideD *
                               self->ctx.tiling_->hi * self->ctx.tiling_->wi;
    if constexpr (Intf::Config::cType::format == ConvolutionBackprop::CubeFormat::NDHWC) {
        doutOffsetBIncre = doutOffsetBIncre * self->ctx.tiling_->cin;
    }
    params.out2B1SrcAddr = params.out2B1SrcAddr + (batchOffsetBIncre + doutOffsetBIncre);
}

template <class Intf, class src0_T>
__aicore__ inline void LoadToA1(Intf* self, bool cachePosA1, uint64_t kaIdx, const Out2L1ScalarParams& params,
                                bool isLoadA1, uint64_t kaStepIdx)
{
    if (!isLoadA1) {
        return;
    }

    auto l1UseKa_ = (kaStepIdx + 1 == DivCeil(self->ctx.kIter_, self->ctx.tiling_->stepKa)) ?
                        self->ctx.singleShapeHo_ * self->ctx.singleShapeWo_ -
                            kaStepIdx * self->ctx.tiling_->stepKa * self->ctx.tiling_->baseK :
                        self->ctx.tiling_->stepKa * self->ctx.tiling_->baseK;
    self->ctx.alignedL1UseKa_ = ShiftCeilChannelSize<Intf>(l1UseKa_, self->ctx.tiling_->k0) * self->ctx.tiling_->k0;

    if constexpr (IsSameType<typename Intf::SrcT, float>::value) {
        if (self->ctx.baseUseM_ != 1) {
            auto l1UseM = params.isLastMAL1 ? self->ctx.tailM_ : self->ctx.tiling_->baseM;
            self->ctx.alignedL1UseM_ = ShiftCeilM0(l1UseM, self->ctx.tiling_->m0) * self->ctx.tiling_->m0;
        }
    }

    if (params.isLoad2L1A) {
        LocalTensor<typename Intf::SrcT> useA1Buf;
        if (cachePosA1) {
            useA1Buf = self->ctx.a1Ping_.template AllocTensor<typename Intf::SrcT>();
        } else {
            useA1Buf = self->ctx.a1Pong_.template AllocTensor<typename Intf::SrcT>();
        }

        LoadToA1ForTransFormat(self, kaIdx, params, kaStepIdx, useA1Buf);

        if (cachePosA1) {
            self->ctx.a1Ping_.EnQue(useA1Buf);
        } else {
            self->ctx.a1Pong_.EnQue(useA1Buf);
        }
    }
}

template <class Intf, class src1_T>
__aicore__ inline void LoadToB1(Intf* self, bool cachePosB1, const Out2L1ScalarParams& params, uint64_t kbStepIdx,
                                bool& skipCurrentHiCompute)
{
    skipCurrentHiCompute = false;
    if (params.isLoad2L1B) {
        B1HiCopyParams hiParams;
        if (CalB1HiCopyParams(self, kbStepIdx, params, hiParams)) {
            skipCurrentHiCompute = true;
            return;
        }
        LocalTensor<typename Intf::SrcT> useB1Buf = cachePosB1 ?
                                                        self->ctx.b1Ping_.template AllocTensor<typename Intf::SrcT>() :
                                                        self->ctx.b1Pong_.template AllocTensor<typename Intf::SrcT>();
        uint64_t out2B1SrcAddrOffset = CalB1GmOffset(self, hiParams.b1SrcHi, params);
        if constexpr (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
            LoadToB1Dn2Nz(self, hiParams.hiCopyLen, out2B1SrcAddrOffset, params, useB1Buf);
        } else {
            LoadToB1Nd2Nz(self, hiParams.hiCopyLen, out2B1SrcAddrOffset, params, useB1Buf);
        }
        if (cachePosB1) {
            self->ctx.bL1HiCopyLenPing = hiParams.hiCopyLen;
            self->ctx.bL1PadUpPing = hiParams.padUp;
            self->ctx.b1Ping_.EnQue(useB1Buf);
        } else {
            self->ctx.bL1HiCopyLenPong = hiParams.hiCopyLen;
            self->ctx.bL1PadUpPong = hiParams.padUp;
            self->ctx.b1Pong_.EnQue(useB1Buf);
        }
    }
}

template <class Intf, class src1_T>
__aicore__ inline void LoadToB1SplitKernelHW(Intf* self, bool cachePosB1, const Out2L1ScalarParams& params,
                                             uint64_t kbStepIdx, uint64_t hkIdx, bool& skipCurrentHiCompute)
{
    skipCurrentHiCompute = false;
    // 需要载入BL1的条件为，被计算的BL0块是BL1上的第一块数据，一次载入完整BL1大小
    // 此时满足以下条件之一需要载入BL1：
    // 1.BL1上无db，并且K方向需要多于一个buffer，每次都需要载入；BL1开db，并且K方向buffer数量小于等于2
    // 2.singleShapeK / stepKb > 2, 优先循环k方向，BL1上数据无法复用
    // 3.order_M时，L1上驻留AL1, BL1数据不复用
    // 4.order_N时，BL1驻留在L1上，且K <=
    // 2，即L1上可以载下全部Kb，此时遍历M方向，BL1数据上数据不会被覆盖，只在M方向循环第一次时载入BL1
    if (params.isLoad2L1B) {
        B1HiCopyParams hiParams;
        if (CalB1HiCopyParamsSplitKernelHW(self, kbStepIdx, hkIdx, params, hiParams)) {
            skipCurrentHiCompute = true;
            return;
        }
        LocalTensor<typename Intf::SrcT> useB1Buf = cachePosB1 ?
                                                        self->ctx.b1Ping_.template AllocTensor<typename Intf::SrcT>() :
                                                        self->ctx.b1Pong_.template AllocTensor<typename Intf::SrcT>();

        // 得到gm的偏移量
        uint64_t out2B1SrcAddrOffset = 0;
        if constexpr (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
            out2B1SrcAddrOffset = params.out2B1SrcAddr +
                                  static_cast<uint64_t>(hiParams.b1SrcHi + hiParams.hiUpValidOffset) *
                                      self->ctx.tiling_->wi;
        } else if constexpr (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NDHWC) {
            out2B1SrcAddrOffset = params.out2B1SrcAddr +
                                  static_cast<uint64_t>(hiParams.b1SrcHi + hiParams.hiUpValidOffset) *
                                      self->ctx.tiling_->wi * self->ctx.tiling_->cin;
        }

        if constexpr (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
            LoadToB1Dn2NzSplitKernelHW(self, hiParams.hiCopyLen, 0, out2B1SrcAddrOffset, params, useB1Buf);
        } else if constexpr (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NDHWC) {
            LoadToB1Nd2NzSplitKernelHW(self, hiParams.hiCopyLen, 0, out2B1SrcAddrOffset, params, useB1Buf);
        }

        if (cachePosB1) {
            self->ctx.bL1HiCopyLenPing = hiParams.hiCopyLen;
            self->ctx.bL1PadUpPing = Ceil(hiParams.padUp, self->ctx.tiling_->strideH);
            self->ctx.b1Ping_.EnQue(useB1Buf);
        } else {
            self->ctx.bL1HiCopyLenPong = hiParams.hiCopyLen;
            self->ctx.bL1PadUpPong = Ceil(hiParams.padUp, self->ctx.tiling_->strideH);
            self->ctx.b1Pong_.EnQue(useB1Buf);
        }
    }
}

} // namespace ConvolutionBackpropFunc

#endif