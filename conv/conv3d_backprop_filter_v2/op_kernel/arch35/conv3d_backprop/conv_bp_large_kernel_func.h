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
#include "basic_api/kernel_basic_intf.h"
#include "../conv3d_backprop_filter_v2/conv3d_backprop_filter_v2_tiling_data.h"
#include "conv_bp_func_common.h"

namespace ConvolutionBackpropFunc {
template <class Intf>
__aicore__ inline void updateParasForSplitKernelHW(Intf* self, Out2L1ScalarParams& out2L1Params, uint32_t startWo,
                                                   uint64_t out2B1SrcAddrStart, uint32_t wkIdx)
{
    int64_t padLeft = 0;
    int64_t padRight = 0;
    int64_t leftValidAddrOffset = 0;
    int64_t b1SrcWiLeftOffGm = static_cast<int64_t>(startWo) * self->ctx.tiling_->strideW - self->ctx.tiling_->padLeft +
                               wkIdx * self->ctx.tiling_->dilationW;
    if (b1SrcWiLeftOffGm < 0) {
        padLeft = -b1SrcWiLeftOffGm;
    }
    leftValidAddrOffset = Ceil(padLeft, self->ctx.tiling_->strideW) * self->ctx.tiling_->strideW + b1SrcWiLeftOffGm;

    int64_t b1SrcWiRightOffGm = static_cast<int64_t>(startWo + self->ctx.singleShapeWo_) * self->ctx.tiling_->strideW +
                                self->ctx.strideKernelDilationW - (self->ctx.tiling_->padLeft + self->ctx.tiling_->wi);
    if (b1SrcWiRightOffGm > 0) {
        padRight = b1SrcWiRightOffGm;
    }
    padRight = padRight - (self->ctx.tiling_->wk - wkIdx - 1) * self->ctx.tiling_->dilationW;

    padLeft = padLeft < 0 ? (0) : Ceil(padLeft, self->ctx.tiling_->strideW);
    padRight = padRight < 0 ? (0) : Ceil(padRight, self->ctx.tiling_->strideW);

    if (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NCDHW) {
        if (padLeft) {
            out2L1Params.out2B1SrcAddr = out2B1SrcAddrStart + leftValidAddrOffset;
        } else {
            out2L1Params.out2B1SrcAddr = out2B1SrcAddrStart + b1SrcWiLeftOffGm;
        }
    } else if (Intf::Config::xType::format == ConvolutionBackprop::CubeFormat::NDHWC) {
        if (padLeft) {
            out2L1Params.out2B1SrcAddr = out2B1SrcAddrStart + leftValidAddrOffset * self->ctx.tiling_->cin;
        } else {
            out2L1Params.out2B1SrcAddr = out2B1SrcAddrStart + b1SrcWiLeftOffGm * self->ctx.tiling_->cin;
        }
    }

    out2L1Params.singleShapeWi = self->ctx.singleShapeWo_;
    if (out2L1Params.singleShapeWi > (padLeft + padRight)) {
        self->ctx.load3d_.l1W = out2L1Params.singleShapeWi - padLeft - padRight;
    } else {
        self->ctx.load3d_.l1W = 0;
    }
    // singleShapeWi已经被限制到256，因此padLeft和padRight最大可能为256，
    // 当padLeft或者padRight大于uint8的最大值255，此时l1w等于0，因此padList值溢出也不影响结果
    self->ctx.load3d_.padList[0] = padLeft;
    self->ctx.load3d_.padList[1] = padRight;
}

template <class Intf>
__aicore__ inline void initParasSplitKernelHW(Intf* self)
{
    // 矩阵计算的值，默认为baseUseN_，baseUseN_可能等于baseN或TailN，但都一定是n0对齐的,splitkernel场景，
    //  每次只循环一个hkwk=1,因此一定是n0
    self->ctx.mmad_.n = self->ctx.tiling_->n0;

    // kExtension是N轴，由于切了Kernel，不管是fp16和32(两个C0的Wk连续)场景均为16,正好一个n0单元;
    self->ctx.load3d_.kExtension = self->ctx.tiling_->n0;
    self->ctx.load3d_.kStartPt = 0; // stepM=stepN=1，每次N都是从0开始读取

    self->ctx.load3d_.channelSize = 16; // cin等于16，避免循环cin1
    self->ctx.load3d_.filterW = 1;
    self->ctx.load3d_.filterH = 1;

    self->ctx.load3d_.dilationFilterW = 1;
    self->ctx.load3d_.dilationFilterH = 1;

    self->ctx.load3d_.filterSizeW = false;
    self->ctx.load3d_.filterSizeH = false;

    // 跳着搬运数据，stride等价于1
    self->ctx.load3d_.strideW = 1;
    self->ctx.load3d_.strideH = 1;
}

template <class Intf>
__aicore__ inline void getHWkIdx(Intf* self, uint64_t hwkLoopIdx, uint64_t& hkIdx, uint64_t& wkIdx)
{
    hkIdx = hwkLoopIdx / self->ctx.tiling_->wk;
    wkIdx = hwkLoopIdx % self->ctx.tiling_->wk;
}

template <class Intf>
__aicore__ inline void ComputeSplitKernelHW(Intf* self, Out2L1ScalarParams& out2L1Params)
{
    if ASCEND_IS_AIV {
        return;
    }

    LocalTensor<typename Intf::SrcT> l0a;
    LocalTensor<typename Intf::SrcT> l0b;
    LocalTensor<typename Intf::L0cT> l0c;
    uint32_t baseUseMBak;
    ComputeInit<Intf>(self, out2L1Params, l0c, baseUseMBak);

    uint64_t dstL0cOffsetBase = self->ctx.dstL0cOffset_;
    // 基本块模板中stepN一定等于1，此处使用curNL1Idx和curNL0Idx均可
    uint64_t usedN = self->ctx.curNIdx_ * self->ctx.tiling_->baseN;
    uint64_t hwkLoopStart = usedN / self->ctx.tiling_->n0;
    uint64_t hwkLoopEnd = (usedN + self->ctx.baseUseN_) / self->ctx.tiling_->n0;
    uint64_t hkIdx = 0, wkIdx = 0;
    for (uint64_t hwkLoopIdx = hwkLoopStart; hwkLoopIdx < hwkLoopEnd; hwkLoopIdx++) {
        getHWkIdx(self, hwkLoopIdx, hkIdx, wkIdx);
        self->ctx.dstL0cOffset_ = dstL0cOffsetBase +
                                  (hwkLoopIdx - hwkLoopStart) * self->ctx.tiling_->baseM * self->ctx.tiling_->n0;
        initParasSplitKernelHW(self);
        bool isFirstMmad = true;

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
                if (unlikely(self->ctx.isSplitWo_)) {
                    updateParasForSplitW(self, out2L1Params, splitWoIdx * splitWo, out2A1SrcAddrStart,
                                         out2B1SrcAddrStart);
                }
                updateParasForSplitKernelHW(self, out2L1Params, splitWoIdx * splitWo, out2B1SrcAddrStart, wkIdx);
                if (!self->ctx.load3d_.l1W) {
                    PipeBarrier<PIPE_ALL>();
                    continue;
                }

                uint64_t curMKL1Idx = self->ctx.stepKaRound * self->ctx.curMIdx_;
                uint64_t curNKL1Idx = self->ctx.stepKbRound * self->ctx.curNIdx_;
                ComputeLoop<Intf>(self, out2L1Params, l0a, l0b, l0c, isFirstMmad, curMKL1Idx, curNKL1Idx, hkIdx);
            }
            out2L1Params.out2A1SrcAddr = out2A1SrcAddrStart;
            out2L1Params.out2B1SrcAddr = out2B1SrcAddrStart;
        }
        // batchout 偏移后的地址要还原回来
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
} // namespace ConvolutionBackpropFunc
#endif
