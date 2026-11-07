/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv_bp_sub_func_load_gm_to_l1a_arch51.h
 * \brief
 */

#ifndef CONV3D_BP_INPUT_SUB_FUNC_LOAD_GM_TO_L1A_ARCH51_H
#define CONV3D_BP_INPUT_SUB_FUNC_LOAD_GM_TO_L1A_ARCH51_H

#include "conv_bp_input_sub_func_utils_arch51.h"
#include "conv_bp_input_sub_func_index_calc_arch51.h"
#include "conv_bp_input_sub_func_load_gm_to_l1_arch51.h"

using AscendC::DivCeil;
using AscendC::Dn2NzParams;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;

namespace Convolution3DBackpropFunc {
// 设置fmatrix参数
template <class Intf>
static __aicore__ inline void CalcSetFmatrixParams(Intf* self, uint32_t fmapH, uint32_t fmapW)
{
    // W
    self->ctx.load3d_.l1W = fmapW;
    // H
    self->ctx.load3d_.l1H = fmapH;
    // 设置pad, pad列表[left, right, top, down], 默认是0, 范围[0, 255]
    if (self->ctx.tiling_->backpropPadLeft < 0) {
        self->ctx.load3d_.padList[0] = DivCeil(abs(self->ctx.tiling_->backpropPadLeft), self->ctx.tiling_->strideW) *
                                           self->ctx.tiling_->strideW +
                                       self->ctx.tiling_->backpropPadLeft;
    } else {
        self->ctx.load3d_.padList[0] = self->ctx.tiling_->backpropPadLeft;
    }

    if (self->ctx.tiling_->backpropPadRight < 0) {
        uint32_t woExpand = ((self->ctx.tiling_->wo - 1) * self->ctx.tiling_->strideW) + 1;
        uint32_t woEnd = DivCeil((woExpand + self->ctx.tiling_->backpropPadRight), self->ctx.tiling_->strideW);
        self->ctx.load3d_.padList[1] = woExpand + self->ctx.tiling_->backpropPadRight -
                                       ((woEnd - 1) * self->ctx.tiling_->strideW + 1);
    } else {
        self->ctx.load3d_.padList[1] = self->ctx.tiling_->backpropPadRight;
    }

    if (self->ctx.curHoIdx_ < 0) {
        self->ctx.load3d_.padList[2] = abs(self->ctx.curHoIdx_);
    } else {
        self->ctx.load3d_.padList[2] = 0;
    }
    self->ctx.load3d_.padList[3] = 255;
}

template <class Intf>
static __aicore__ inline void CalcCurCoutSizeA1(Intf* self, const uint64_t kIdx, const uint32_t curCoutIdx,
                                                uint32_t& curCoutSize)
{
    // 考虑到Preload的场景，L1的载入量要根据传入的KIdx确定，不能使用全局变量
    uint32_t kaL1Size = 0;
    if (unlikely(kIdx >= self->ctx.kIterStepKaTail)) {
        kaL1Size = (self->ctx.stepKaTail - 1) * self->ctx.tiling_->baseK + self->ctx.tailK_;
    } else {
        kaL1Size = self->ctx.curStepKa_ * self->ctx.tiling_->baseK;
    }
    curCoutSize = DivHkWk<Intf>(self, kaL1Size);

    curCoutSize = (curCoutSize < self->ctx.singleShapeCout_ - curCoutIdx) ? curCoutSize :
                                                                            (self->ctx.singleShapeCout_ - curCoutIdx);
}

template <class Intf>
static __aicore__ inline void CalcOutToA1DstAddr(Intf* self, const uint32_t strideH, const uint32_t curHoSize,
                                                 uint32_t& loadToA1HLoop, uint64_t& out2A1DstAddrOffset,
                                                 uint32_t& woExpand)
{
    uint32_t hDstDataSkipLine = 0;
    uint32_t actualDstDataStartIdx = (self->ctx.curHoIdx_ < 0) ? 0 : self->ctx.curHoIdx_;
    if (strideH > 1 && actualDstDataStartIdx % strideH) {
        hDstDataSkipLine = AlignUp(actualDstDataStartIdx, strideH) - actualDstDataStartIdx;
    }

    loadToA1HLoop = DivCeil(curHoSize - hDstDataSkipLine, strideH);
    out2A1DstAddrOffset = static_cast<uint64_t>(hDstDataSkipLine) * woExpand << self->ctx.tiling_->c0BitsA;
}

template <class Intf, class src0_T>
static __aicore__ inline void CalcLoadToA1Dn2NzParams(Intf* self, Dn2NzParams& dn2NzParams,
                                                      uint64_t& out2A1DstAddrOffset, uint32_t& curCoutIdx,
                                                      uint32_t kIdx)
{
    uint32_t strideH = self->ctx.tiling_->strideH;

    uint32_t curHoSize = self->ctx.curHoSize_;
    uint32_t curCoutSize = 0;
    CalcCurCoutSizeA1<Intf>(self, kIdx, curCoutIdx, curCoutSize);
    uint32_t loadToA1HLoop = 0;
    uint32_t woExpand = ((self->ctx.tiling_->wo - 1) * self->ctx.tiling_->strideW) + 1;
    bool resetWoExpand = self->ctx.tiling_->backpropPadLeft < 0 || self->ctx.tiling_->backpropPadRight < 0;
    uint32_t woStart = self->ctx.tiling_->backpropPadLeft < 0 ?
                           DivCeil(abs(self->ctx.tiling_->backpropPadLeft), self->ctx.tiling_->strideW) :
                           0;
    uint32_t woEnd = self->ctx.tiling_->backpropPadRight < 0 ?
                         DivCeil((woExpand + self->ctx.tiling_->backpropPadRight), self->ctx.tiling_->strideW) :
                         DivCeil(woExpand, self->ctx.tiling_->strideW);

    if (resetWoExpand) {
        woExpand = woEnd == woStart ? 1 : (woEnd - woStart - 1) * self->ctx.tiling_->strideW + 1;
    }
    CalcOutToA1DstAddr(self, strideH, curHoSize, loadToA1HLoop, out2A1DstAddrOffset, woExpand);

    if (unlikely(self->ctx.tiling_->strideW * strideH > 1 || self->ctx.realWoSize_ != self->ctx.tiling_->wo ||
                 resetWoExpand)) {
        dn2NzParams.dnNum = loadToA1HLoop;
        if (resetWoExpand) {
            self->ctx.realWoSize_ = woEnd - woStart;
        }
        dn2NzParams.nValue = self->ctx.realWoSize_;
        dn2NzParams.dValue = curCoutSize;
        dn2NzParams.srcDnMatrixStride = self->ctx.tiling_->wo;
        dn2NzParams.srcDValue = self->ctx.doHoWo_;
        dn2NzParams.dstNzC0Stride = curHoSize * woExpand;
        dn2NzParams.dstNzNStride = self->ctx.tiling_->strideW;
        dn2NzParams.dstNzMatrixStride = strideH * woExpand << self->ctx.tiling_->c0BitsA;
    } else {
        dn2NzParams.dnNum = 1;
        dn2NzParams.nValue = loadToA1HLoop * self->ctx.tiling_->wo;
        dn2NzParams.dValue = curCoutSize;
        dn2NzParams.srcDnMatrixStride = 0;
        dn2NzParams.srcDValue = self->ctx.doHoWo_;
        dn2NzParams.dstNzC0Stride = curHoSize * woExpand;
        dn2NzParams.dstNzNStride = 1;
        dn2NzParams.dstNzMatrixStride = 0;
    }

    CalcSetFmatrixParams(self, curHoSize, woExpand);
}

template <class Intf, class src0_T>
__aicore__ inline void LoadToA1ForDn2Nz(Intf* self, LocalTensor<typename Intf::SrcAT>& useA1Buf, uint32_t kIdx,
                                        uint32_t curDoutIdx)
{
    Dn2NzParams dn2NzParams;
    uint64_t out2A1DstAddrOffset = 0;
    uint64_t coOffset = 0;
    uint64_t hoOffset = 0;
    uint64_t woOffset = 0;
    // GM的绝对地址已经在API外部计算，这里采用绝对坐标时，需要去除起始坐标
    int64_t curHoStartOffset = self->ctx.curHoStartIdx_ < 0 ? 0 : self->ctx.curHoStartIdx_;
    int64_t curHoIdx = self->ctx.curHoIdx_ < 0 ? 0 : (self->ctx.curHoIdx_ - curHoStartOffset);
    // 换算回放大前的相对Ho坐标（以单核HoStartIdx为原点）
    int64_t curOriHoIdx = curHoIdx;
    uint32_t curCoutIdx = 0;
    curCoutIdx = DivHkWk<Intf>(self, kIdx * self->ctx.tiling_->baseK);
    coOffset = curCoutIdx * self->ctx.doHoWo_;
    uint32_t strideH = self->ctx.tiling_->strideH;
    if (unlikely(self->ctx.tiling_->strideW * strideH > 1)) {
        InitZeroValue<Intf, typename Intf::SrcAT>(self, useA1Buf, true);
    }
    CalcLoadToA1Dn2NzParams<Intf, typename Intf::SrcAT>(self, dn2NzParams, out2A1DstAddrOffset, curCoutIdx, kIdx);
    if (strideH > 1) {
        curHoIdx = self->ctx.curHoIdx_ < 0 ? 0 : self->ctx.curHoIdx_;
        // 换算回放大前的相对Ho坐标（以单核HoStartIdx为原点, 用当前Ho的坐标（均是放大前的坐标）,
        // 减去当前分核起始点的坐标,获取当前点的src ho偏移（相对偏移）
        curOriHoIdx = DivCeil(curHoIdx, strideH) - DivCeil(curHoStartOffset, strideH);
    }
    hoOffset = curOriHoIdx * self->ctx.tiling_->wo;
    if (self->ctx.tiling_->backpropPadLeft < 0) {
        woOffset += DivCeil(abs(self->ctx.tiling_->backpropPadLeft), self->ctx.tiling_->strideW);
    }
    uint64_t doOffset = static_cast<uint64_t>(curDoutIdx) * self->ctx.hoWo_;
    uint64_t out2A1SrcAddrOffset = coOffset + doOffset + hoOffset + woOffset;
    if (dn2NzParams.dnNum > 0) { // dn2nz参数有效时，才执行加载，否则先跳过；后续考虑提前计算dn2nz参数减少无效计算
        DataCopy(useA1Buf[out2A1DstAddrOffset], self->ctx.outBackPropGlobal_[out2A1SrcAddrOffset], dn2NzParams);
    }
}

template <class Intf, class src0_T>
__aicore__ inline void LoadToA1(Intf* self, uint32_t kIdx, uint32_t curDoutIdx, bool loadFlag)
{
    if (!loadFlag || unlikely(kIdx >= self->ctx.kIter_ || (self->ctx.isA1FullLoadFlag_ && !self->ctx.isLoadA1_))) {
        return;
    }
    LocalTensor<typename Intf::SrcAT> useA1Buf = self->ctx.inQueL1A_.template AllocTensor<typename Intf::SrcAT>();

    LoadToA1ForDn2Nz<Intf, typename Intf::SrcAT>(self, useA1Buf, kIdx, curDoutIdx);
    self->ctx.inQueL1A_.EnQue(useA1Buf);
}
} // namespace Convolution3DBackpropFunc

#endif
