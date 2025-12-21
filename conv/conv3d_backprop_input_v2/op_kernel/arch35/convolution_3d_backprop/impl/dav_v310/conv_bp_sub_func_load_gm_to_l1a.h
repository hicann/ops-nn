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
 * \file conv_bp_sub_func_load_gm_to_l1a.h
 * \brief
 */

#ifndef CONV3D_BP_INPUT_SUB_FUNC_LOAD_GM_TO_L1A_ADVANCE_H
#define CONV3D_BP_INPUT_SUB_FUNC_LOAD_GM_TO_L1A_ADVANCE_H

#include "conv_bp_sub_func_mix.h"
#include "conv_bp_sub_func_load_gm_to_l1.h"

using AscendC::DivCeil;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::Dn2NzParams;
using AscendC::Nd2NzParams;

namespace Convolution3DBackpropFunc {
// 设置fmatrix参数
template <class Intf>
static __aicore__ inline void CalcSetFmatrixParams(Intf *self, uint32_t fmapH, uint32_t fmapW)
{
    // W
    self->ctx.load3d_.l1W = fmapW;
    // H
    self->ctx.load3d_.l1H = fmapH;
    // 设置pad, pad列表[left, right, top, down], 默认是0, 范围[0, 255]
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
        uint32_t splitIndex = self->ctx.splitHIndex_ * self->ctx.tiling_->strideW + self->ctx.splitWIndex_;
        self->ctx.load3d_.padList[0] =
            self->ctx.subPadLeftList_[splitIndex] < 0 ? 0 : self->ctx.subPadLeftList_[splitIndex];
        self->ctx.load3d_.padList[1] =
            self->ctx.subPadRightList_[splitIndex] < 0 ? 0 : self->ctx.subPadRightList_[splitIndex];
        if (self->ctx.curHoIdx_ < 0) {
            self->ctx.load3d_.padList[2] =
                self->ctx.subPadUpList_[splitIndex] < 0 ? 0 : self->ctx.subPadUpList_[splitIndex];
        } else {
            self->ctx.load3d_.padList[2] = 0;
        }
    } else {
        self->ctx.load3d_.padList[0] = self->ctx.tiling_->backpropPadLeft;
        self->ctx.load3d_.padList[1] = self->ctx.tiling_->backpropPadRight;
        if (self->ctx.curHoIdx_ < 0) {
            self->ctx.load3d_.padList[2] = abs(self->ctx.curHoIdx_);
        } else {
            self->ctx.load3d_.padList[2] = 0;
        }

        if constexpr (Intf::conv3dConfig.loadB1Condition == TPL_GM_TO_L1_NO_HK_WK) {
            self->ctx.load3d_.padList[0] = self->ctx.curWoLeftIdx_ < 0 ? abs(self->ctx.curWoLeftIdx_) : 0;
            self->ctx.load3d_.padList[1] = self->ctx.curWoRightIdx_ > 0 ? abs(self->ctx.curWoRightIdx_) : 0;
        }
    }
    self->ctx.load3d_.padList[3] = 255;
}

template <class Intf>
static __aicore__ inline void CalcCurCoutSizeA1(Intf *self, const uint64_t kIdx, const uint32_t curCoutIdx, uint32_t &curCoutSize)
{
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
        if (self->ctx.kSCoutFullLoad_) {    // cout全载场景
            curCoutSize = self->ctx.tiling_->singleCoreCout;
            return;
        }
    }

    // 考虑到Preload的场景，L1的载入量要根据传入的KIdx确定，不能使用全局变量
    uint32_t kaL1Size = 0;
    if (unlikely(kIdx >= self->ctx.kIterStepKaTail)) {
        kaL1Size = (self->ctx.stepKaTail - 1) * self->ctx.tiling_->baseK + self->ctx.tailK_;
    } else {
        kaL1Size = self->ctx.curStepKa_ * self->ctx.tiling_->baseK;
    }
    if constexpr (Intf::conv3dConfig.enableC04Flag) {
        curCoutSize = C04_COUT_SIZE;
    } else {
        curCoutSize = DivHkWk<Intf>(self, kaL1Size);
    }
    curCoutSize = (curCoutSize < self->ctx.singleShapeCout_ - curCoutIdx) ?
        curCoutSize : (self->ctx.singleShapeCout_ - curCoutIdx);
}

template <class Intf>
static __aicore__ inline void CalcRealWo4KernelSplit(Intf *self, uint32_t &realWo)
{
    uint32_t splitIndex = self->ctx.splitHIndex_ * self->ctx.tiling_->strideW + self->ctx.splitWIndex_;
    if (self->ctx.tiling_->hk == 4 || self->ctx.tiling_->hk == 2) { // 4: kernel size 4*4 2: kernel size 2*2
        if ((self->ctx.tiling_->wi & 0x1) == 1) {
            if (self->ctx.subPadLeftList_[splitIndex] < 0) {
                --realWo;
            }
        } else {
            --realWo;
        }
    } else if (self->ctx.tiling_->hk == 3) { // 3: kernel size 3*3
        if (self->ctx.subPadLeftList_[splitIndex] < 0) {
            --realWo;
        }
    }
}

template <class Intf>
static __aicore__ inline void CalcLoadToA1Dn2NzParams4KernelSplit(Intf *self, uint32_t kIdx, Dn2NzParams &dn2NzParams,
    int64_t &curOriHoIdx, uint64_t &woOffset)
{
    uint32_t curHoSize = self->ctx.curHoSize_;
    uint32_t realWo = self->ctx.tiling_->wo;
    // 反向pad为0时，子kernel的A矩阵需要跳过脏数据，目前只支持各方向pad相等
    uint32_t splitIndex = self->ctx.splitHIndex_ * self->ctx.tiling_->strideW + self->ctx.splitWIndex_;
    if (self->ctx.tiling_->backpropPadUp == 0) {
        if (self->ctx.subPadUpList_[splitIndex] < 0) {
            curOriHoIdx -= self->ctx.subPadUpList_[splitIndex];
        }
    }
    if (self->ctx.tiling_->backpropPadLeft == 0) {
        if (self->ctx.subPadLeftList_[splitIndex] < 0) {
            woOffset -= self->ctx.subPadLeftList_[splitIndex];
        }
        CalcRealWo4KernelSplit(self, realWo);
    }

    uint32_t curCoutIdx = DivHkWk<Intf>(self, kIdx * self->ctx.tiling_->baseK);
    uint32_t curCoutSize = 0;
    CalcCurCoutSizeA1(self, kIdx, curCoutIdx, curCoutSize);
    // 目前只支持各方向pad相等，只判断左pad，简化计算
    if (self->ctx.tiling_->backpropPadLeft == 0) {
        dn2NzParams.dnNum = curHoSize;
        dn2NzParams.nValue = realWo;
        dn2NzParams.dValue = curCoutSize;
        dn2NzParams.srcDnMatrixStride = self->ctx.tiling_->wo;
        dn2NzParams.srcDValue = self->ctx.doHoWo_;
        dn2NzParams.dstNzC0Stride = curHoSize * realWo;
        dn2NzParams.dstNzNStride = 1;
        dn2NzParams.dstNzMatrixStride = realWo << self->ctx.tiling_->c0Bits;
    } else {
        dn2NzParams.dnNum = 1;
        dn2NzParams.nValue = curHoSize * realWo;   // A1全载，howo连续
        dn2NzParams.dValue = curCoutSize;
        dn2NzParams.srcDnMatrixStride = 0;
        dn2NzParams.srcDValue = self->ctx.doHoWo_;
        dn2NzParams.dstNzC0Stride = curHoSize * realWo;
        dn2NzParams.dstNzNStride = 1;
        dn2NzParams.dstNzMatrixStride = 0;
    }
    CalcSetFmatrixParams(self, curHoSize, realWo);
}

template <class Intf>
static __aicore__ inline void CalcLoadToA1Nd2NzParams4KernelSplit(Intf *self, Nd2NzParams &nd2NzParams)
{
    uint32_t curHoSize = self->ctx.curHoSize_;
    uint32_t hoExpand = self->ctx.tiling_->ho;

    nd2NzParams.ndNum = 1;
    nd2NzParams.nValue = curHoSize * self->ctx.tiling_->wo;   // A1全载，howo连续
    nd2NzParams.dValue = self->ctx.channelSize_;
    nd2NzParams.srcNdMatrixStride = 0;
    nd2NzParams.srcDValue = self->ctx.tiling_->cout;
    nd2NzParams.dstNzC0Stride = curHoSize * self->ctx.tiling_->wo;
    nd2NzParams.dstNzNStride = 1;
    nd2NzParams.dstNzMatrixStride = 0;

    CalcSetFmatrixParams(self, curHoSize, self->ctx.tiling_->wo);
}

template <class Intf>
static __aicore__ inline void CalcOutToA1DstAddr(Intf *self, const uint32_t strideH,
    const uint32_t curHoSize, uint32_t &loadToA1HLoop, uint64_t &out2A1DstAddrOffset, uint32_t &woExpand)
{
    uint32_t hDstDataSkipLine = 0;
    uint32_t actualDstDataStartIdx = (self->ctx.curHoIdx_ < 0) ? 0 : self->ctx.curHoIdx_;
    if (strideH > 1 && actualDstDataStartIdx % strideH) {
        hDstDataSkipLine = AlignUp(actualDstDataStartIdx, strideH) - actualDstDataStartIdx;
    }

    loadToA1HLoop = DivCeil(curHoSize - hDstDataSkipLine, strideH);
    if constexpr (Intf::conv3dConfig.enableC04Flag) {
        out2A1DstAddrOffset = static_cast<uint64_t>(hDstDataSkipLine) * woExpand * C04_COUT_SIZE;
    } else {
        out2A1DstAddrOffset = static_cast<uint64_t>(hDstDataSkipLine) * woExpand << self->ctx.tiling_->c0Bits;
    }

    if constexpr (Intf::conv3dConfig.loadB1Condition == TPL_GM_TO_L1_NO_HK_WK) {
        loadToA1HLoop = 1;  // 此时只需加载一行
        uint32_t allWoExpand = woExpand;
        woExpand = ((self->ctx.realWoSize_ - 1) * self->ctx.tiling_->strideW) + 1;
        // 子wk可能不需要加载完整地wo参与计算，计算右侧wo跳过部分
        uint32_t actualDstWoEndIdx = self->ctx.curWoRightIdx_ < 0 ? woExpand - 1 + self->ctx.curWoRightIdx_ : 0;
        if (self->ctx.tiling_->strideW > 1 && actualDstWoEndIdx % self->ctx.tiling_->strideW) {
            // stride大于1时，可能存在结束位置刚好在前放大补0位置处，需要加载这一部分wo
            woExpand += (actualDstWoEndIdx - AlignDown(actualDstWoEndIdx, self->ctx.tiling_->strideW));
        }

        // 子wk可能不需要加载完整地wo参与计算，计算左侧wo跳过部分
        uint32_t wDstDataSkipPoint = 0;
        uint32_t actualDstWoStartIdx = self->ctx.curWoLeftIdx_ < 0 ? 0 : self->ctx.curWoLeftIdx_;
        if (self->ctx.tiling_->strideW > 1 && actualDstWoStartIdx % self->ctx.tiling_->strideW) {
            wDstDataSkipPoint = AlignUp(actualDstWoStartIdx, self->ctx.tiling_->strideW) - actualDstWoStartIdx;
            // stride大于1时，可能存在结束位置刚好在前放大补0位置处，需要加载这一部分wo
            woExpand += wDstDataSkipPoint;
        }

        if (hDstDataSkipLine > 0 && woExpand < allWoExpand) {   // h方向最开始补0时地址需要去除跳过Wo的部分
            out2A1DstAddrOffset -= (hDstDataSkipLine * (allWoExpand - woExpand)) << self->ctx.tiling_->c0Bits;
        }
        out2A1DstAddrOffset += wDstDataSkipPoint << self->ctx.tiling_->c0Bits;
    }
}

template <class Intf, class src0_T>
static __aicore__ inline void CalcLoadToA1Dn2NzParams(Intf *self, Dn2NzParams &dn2NzParams,
    uint64_t &out2A1DstAddrOffset, uint32_t &curCoutIdx, uint32_t kIdx)
{
    uint32_t strideH = self->ctx.tiling_->strideH;
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_H) {
        strideH = 1;
    }

    uint32_t curHoSize = self->ctx.curHoSize_;
    uint32_t curCoutSize = 0;
    CalcCurCoutSizeA1(self, kIdx, curCoutIdx, curCoutSize);
    uint32_t loadToA1HLoop = 0;
    uint32_t woExpand = ((self->ctx.tiling_->wo - 1) * self->ctx.tiling_->strideW) + 1;
    CalcOutToA1DstAddr(self, strideH, curHoSize, loadToA1HLoop, out2A1DstAddrOffset, woExpand);

    if (unlikely(self->ctx.tiling_->strideW * strideH > 1) || self->ctx.realWoSize_ != self->ctx.tiling_->wo) {
        dn2NzParams.dnNum = loadToA1HLoop;
        dn2NzParams.nValue = self->ctx.realWoSize_;
        dn2NzParams.dValue = curCoutSize;
        dn2NzParams.srcDnMatrixStride = self->ctx.tiling_->wo;
        dn2NzParams.srcDValue = self->ctx.doHoWo_;
        dn2NzParams.dstNzC0Stride = curHoSize * woExpand;
        dn2NzParams.dstNzNStride = self->ctx.tiling_->strideW;
        dn2NzParams.dstNzMatrixStride = strideH * woExpand << self->ctx.tiling_->c0Bits;
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
static __aicore__ inline void CalcLoadToA1Nd2NzParams(Intf *self, Nd2NzParams &nd2NzParams,
    uint64_t &out2A1DstAddrOffset, uint32_t &curCoutIdx, uint64_t kIdx)
{
    uint32_t curHoSize = self->ctx.curHoSize_;
    uint32_t curCoutSize = 0;
    uint32_t strideH = self->ctx.tiling_->strideH;
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_H) {
        strideH = 1;
    }
    CalcCurCoutSizeA1(self, kIdx, curCoutIdx, curCoutSize);
    uint32_t loadToA1HLoop = 0;
    uint32_t woExpand = ((self->ctx.tiling_->wo - 1) * self->ctx.tiling_->strideW) + 1;
    CalcOutToA1DstAddr(self, strideH, curHoSize, loadToA1HLoop, out2A1DstAddrOffset, woExpand);

    if (unlikely(self->ctx.tiling_->strideW * strideH > 1)) {
        nd2NzParams.ndNum = loadToA1HLoop;
        nd2NzParams.nValue = self->ctx.tiling_->wo;
        nd2NzParams.dValue = curCoutSize;
        nd2NzParams.srcNdMatrixStride = self->ctx.tiling_->wo * self->ctx.tiling_->cout;
        nd2NzParams.srcDValue = self->ctx.tiling_->cout;
        nd2NzParams.dstNzC0Stride = curHoSize * woExpand;
        nd2NzParams.dstNzNStride = self->ctx.tiling_->strideW;
        nd2NzParams.dstNzMatrixStride = strideH * woExpand << self->ctx.tiling_->c0Bits;
    } else {
        nd2NzParams.ndNum = 1;
        nd2NzParams.nValue = loadToA1HLoop * self->ctx.tiling_->wo;
        nd2NzParams.dValue = curCoutSize;
        nd2NzParams.srcNdMatrixStride = 0;
        nd2NzParams.srcDValue = self->ctx.tiling_->cout;
        nd2NzParams.dstNzC0Stride = curHoSize * woExpand;
        nd2NzParams.dstNzNStride = 1;
        nd2NzParams.dstNzMatrixStride = 0;
    }

    CalcSetFmatrixParams(self, curHoSize, woExpand);
}

template <class Intf, class src0_T>
__aicore__ inline void LoadToA1ForDn2Nz(Intf *self, LocalTensor<typename Intf::SrcT> &useA1Buf,
    uint32_t kIdx, uint32_t curDoutIdx)
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
    if constexpr (!Intf::conv3dConfig.enableC04Flag) {
        curCoutIdx = DivHkWk<Intf>(self, kIdx * self->ctx.tiling_->baseK);
    }
    coOffset = curCoutIdx * self->ctx.doHoWo_;

    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
        CalcLoadToA1Dn2NzParams4KernelSplit(self, kIdx, dn2NzParams, curOriHoIdx, woOffset);
    } else {
        uint32_t strideH = self->ctx.tiling_->strideH;
        if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_H) {
            strideH = 1;
        }
        if (unlikely(self->ctx.tiling_->strideW * strideH > 1)) {
            InitZeroValue(self, useA1Buf);
        }
        CalcLoadToA1Dn2NzParams<Intf, typename Intf::SrcT>(self, dn2NzParams, out2A1DstAddrOffset,
            curCoutIdx, kIdx);
        if (strideH > 1) {
            curHoIdx = self->ctx.curHoIdx_ < 0 ? 0 : self->ctx.curHoIdx_;
            // 换算回放大前的相对Ho坐标（以单核HoStartIdx为原点, 用当前Ho的坐标（均是放大前的坐标）
            // 减去当前分核起始点的坐标,获取当前点的src ho偏移（相对偏移）
            curOriHoIdx = DivCeil(curHoIdx, strideH) - DivCeil(curHoStartOffset, strideH);
        }
    }

    hoOffset = curOriHoIdx * self->ctx.tiling_->wo;
    if constexpr (Intf::conv3dConfig.loadB1Condition == TPL_GM_TO_L1_NO_HK_WK) {
        woOffset += (self->ctx.curWoLeftIdx_ <= 0 ? 0 : DivCeil(self->ctx.curWoLeftIdx_, self->ctx.tiling_->strideW));
    }
    uint64_t doOffset = static_cast<uint64_t>(curDoutIdx) * self->ctx.hoWo_;
    uint64_t out2A1SrcAddrOffset = coOffset + doOffset + hoOffset + woOffset;
    if (dn2NzParams.dnNum > 0) { // dn2nz参数有效时，才执行加载，否则先跳过；后续考虑提前计算dn2nz参数减少无效计算
        if constexpr (Intf::conv3dConfig.enableC04Flag) {
            DataCopy<typename Intf::SrcT, true>(useA1Buf[out2A1DstAddrOffset], self->ctx.outBackPropGlobal_[out2A1SrcAddrOffset], dn2NzParams);
        } else {
            DataCopy(useA1Buf[out2A1DstAddrOffset], self->ctx.outBackPropGlobal_[out2A1SrcAddrOffset], dn2NzParams);
        }
    }
}

template <class Intf, class src0_T>
__aicore__ inline void LoadToA1ForNd2Nz(Intf *self, LocalTensor<typename Intf::SrcT> &useA1Buf,
    uint32_t kIdx, uint32_t curDoutIdx)
{
    Nd2NzParams nd2NzParams;
    uint64_t out2A1DstAddrOffset = 0;
    uint64_t coOffset = 0;
    uint64_t hoOffset = 0;
    // GM的绝对地址已经在API外部计算，这里采用绝对坐标时，需要去除起始坐标
    int64_t curHoStartOffset = self->ctx.curHoStartIdx_ < 0 ? 0 : self->ctx.curHoStartIdx_;
    int64_t curHoIdx = self->ctx.curHoIdx_ < 0 ? 0 : (self->ctx.curHoIdx_ - curHoStartOffset);
    // 换算回放大前的相对Ho坐标（以单核HoStartIdx为原点）
    int64_t curOriHoIdx = curHoIdx;

    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
        CalcLoadToA1Nd2NzParams4KernelSplit(self, nd2NzParams);
    } else {
        uint32_t strideH = self->ctx.tiling_->strideH;
        if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_H) {
            strideH = 1;
        }
        if (unlikely(self->ctx.tiling_->strideW * strideH > 1)) {
            InitZeroValue(self, useA1Buf);
        }

        uint32_t curCoutIdx = 0;
        if constexpr (!Intf::conv3dConfig.enableC04Flag) {
            curCoutIdx = DivHkWk<Intf>(self, kIdx * self->ctx.tiling_->baseK);
        }
        CalcLoadToA1Nd2NzParams<Intf, typename Intf::SrcT>(self, nd2NzParams, out2A1DstAddrOffset,
            curCoutIdx, kIdx);
        coOffset = curCoutIdx;
        if (strideH > 1) {
            curHoIdx = self->ctx.curHoIdx_ < 0 ? 0 : self->ctx.curHoIdx_;
            // 换算回放大前的相对Ho坐标（以单核HoStartIdx为原点, 用当前Ho的坐标（均是放大前的坐标）
            // 减去当前分核起始点的坐标,获取当前点的src ho偏移（相对偏移）
            curOriHoIdx = DivCeil(curHoIdx, strideH) - DivCeil(curHoStartOffset, strideH);
        }
    }

    hoOffset = curOriHoIdx * self->ctx.tiling_->wo * self->ctx.tiling_->cout;
    uint64_t woOffset = 0;
    if constexpr (Intf::conv3dConfig.loadB1Condition == TPL_GM_TO_L1_NO_HK_WK) {
        woOffset = (self->ctx.curWoLeftIdx_ <= 0 ? 0 :
            DivCeil(self->ctx.curWoLeftIdx_, self->ctx.tiling_->strideW) * self->ctx.tiling_->cout);
    }
    uint64_t doOffset = static_cast<uint64_t>(curDoutIdx) * self->ctx.hoWo_ * self->ctx.tiling_->cout;
    uint64_t out2A1SrcAddrOffset = coOffset + doOffset + hoOffset + woOffset;
    if (nd2NzParams.ndNum > 0) { // nd2nz参数有效时，才执行加载，否则先跳过；后续考虑提前计算dn2nz参数减少无效计算
        if constexpr (Intf::conv3dConfig.enableC04Flag) {
            DataCopy<typename Intf::SrcT, true>(useA1Buf[out2A1DstAddrOffset], self->ctx.outBackPropGlobal_[out2A1SrcAddrOffset], nd2NzParams);
        } else {
            DataCopy(useA1Buf[out2A1DstAddrOffset], self->ctx.outBackPropGlobal_[out2A1SrcAddrOffset], nd2NzParams);
        }
    }
}

template <class Intf, class src0_T>
__aicore__ inline void LoadToA1(Intf *self, uint32_t kIdx, uint32_t curDoutIdx, bool loadFlag)
{
    if (!loadFlag || unlikely(kIdx >= self->ctx.kIter_ || (self->ctx.isA1FullLoadFlag_ && !self->ctx.isLoadA1_))) {
        return;
    }
    LocalTensor<typename Intf::SrcT> useA1Buf = self->ctx.inQueL1A_.template AllocTensor<typename Intf::SrcT>();
    if constexpr (Intf::Config::cType::format == Convolution3DBackprop::CubeFormat::NCDHW) {
        LoadToA1ForDn2Nz<Intf, typename Intf::SrcT>(self, useA1Buf, kIdx, curDoutIdx);
    } else {
        LoadToA1ForNd2Nz<Intf, typename Intf::SrcT>(self, useA1Buf, kIdx, curDoutIdx);
    }
    self->ctx.inQueL1A_.EnQue(useA1Buf);
}
} // namespace Convolution3DBackpropFunc

#endif
