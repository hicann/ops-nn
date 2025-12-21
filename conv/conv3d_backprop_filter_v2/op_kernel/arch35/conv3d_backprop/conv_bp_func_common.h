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

    return ;
}

template <class Intf>
__aicore__ inline void calculateWoIterTimes(Intf *self, int32_t &woIterateTimes, const int32_t splitWo) {
    if (splitWo == 0) {
        woIterateTimes = 1;
        return ;
    }

    woIterateTimes = Ceil(self->ctx.tiling_->wo, splitWo);

    return ;
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
        out2L1Params.singleShapeWi = self->ctx.singleShapeWo_ * self->ctx.tiling_->strideW +
                            self->ctx.strideKernelDilationW;

    } else {
        self->ctx.singleShapeWo_ = self->ctx.tiling_->wo;
        uint64_t singleShapeWi = self->ctx.singleShapeWo_ * self->ctx.tiling_->strideW + self->ctx.strideKernelDilationW;
        out2L1Params.singleShapeWi = singleShapeWi;
    }

    return ;
}

}  // namespace ConvolutionBackpropFunc
#endif