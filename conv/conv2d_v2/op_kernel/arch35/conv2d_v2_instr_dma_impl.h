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
 * \file conv2d_v2_instr_dma_impl.h
 * \brief
 */

#ifndef CONV2D_V2_INSTR_DMA_IMPL_H
#define CONV2D_V2_INSTR_DMA_IMPL_H

#include "conv2d_v2_config.h"
#include "conv2d_v2_util.h"

namespace Conv2dFunc {
using namespace AscendC;
using namespace conv;

template <class Intf>
class DmaLoadGM2UBTools {
public:
    __aicore__ inline DmaLoadGM2UBTools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
        if constexpr (Intf::formatFmap == ConvFormat::NCHW) {
            srcCiStride = self_->ctx.convTiling->orgHixWi * Intf::k0;
        } else if constexpr (Intf::formatFmap == ConvFormat::NHWC) {
            srcCiStride = Intf::k0;
        }
        srcKhStride = self_->ctx.convTiling->khUb * self_->ctx.convTiling->dilationH;
        srcKwStride = self_->ctx.convTiling->kwUb * self_->ctx.convTiling->dilationW;
    }

    __aicore__ inline void LoadGM2UB()
    {
        self_->ctx.kwAL1Iter = self_->ctx.kAL1Iter % self_->ctx.ddr2L1LoopKw;
        self_->ctx.khAL1Iter = self_->ctx.kAL1Iter / self_->ctx.ddr2L1LoopKw % self_->ctx.ddr2L1LoopKh;
        self_->ctx.cinAL1Iter = (self_->ctx.kAL1Iter / (self_->ctx.ddr2L1LoopKw * self_->ctx.ddr2L1LoopKh)) %
                                self_->ctx.cinAL1LoopTimes;
        srckhAL1IterOffset = self_->ctx.khAL1Iter * self_->ctx.convTiling->dilationH * self_->ctx.convTiling->khL1 +
                             self_->ctx.vecKhIter * srcKhStride;
        srckwAL1IterOffset = self_->ctx.kwAL1Iter * self_->ctx.convTiling->dilationW * self_->ctx.convTiling->kwL1 +
                             self_->ctx.vecKwIter * srcKwStride;
        if (unlikely(self_->ctx.isFirstIterate)) {
            if constexpr (Intf::formatFmap == ConvFormat::NCHW) {
                copyParams.loopInfo.loopSrcStride[NDDMA_LOOP0_INDEX] = self_->ctx.convTiling->orgHixWi;
                copyParams.loopInfo.loopSrcStride[NDDMA_LOOP1_INDEX] = self_->ctx.convTiling->dilationW;
                copyParams.loopInfo.loopSrcStride[NDDMA_LOOP2_INDEX] = self_->ctx.convTiling->dilationH *
                    self_->ctx.orgWi;
            } else {
                copyParams.loopInfo.loopSrcStride[NDDMA_LOOP0_INDEX] = 1;
                copyParams.loopInfo.loopSrcStride[NDDMA_LOOP1_INDEX] = self_->ctx.convTiling->dilationW *
                    self_->ctx.convTiling->orgCi;
                copyParams.loopInfo.loopSrcStride[NDDMA_LOOP2_INDEX] = self_->ctx.convTiling->dilationH *
                    self_->ctx.orgWi * self_->ctx.convTiling->orgCi;
            }
            copyParams.loopInfo.loopDstStride[NDDMA_LOOP0_INDEX] = 1;
        }

        if constexpr (Intf::isQuantScene) {
            LocalTensor<int8_t> zeroTensor = self_->ctx.ubBuf.template Get<int8_t>();
            Duplicate<int8_t>(zeroTensor, 0, self_->ctx.ubBufSize);
        } else {
            Duplicate<typename Intf::FmapT>(self_->ctx.img2ColTensor, 0, self_->ctx.ubBufSize);
        }

        event_t eventId = static_cast<event_t>(self_->ctx.pipe.FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
        copyParams.loopInfo.loopSize[NDDMA_LOOP0_INDEX] = IsCiTail() ? self_->ctx.ciTail : Intf::k0;

        uint32_t baseOffset = self_->ctx.batchIter * self_->ctx.fmapOneBatchSize +
                              self_->ctx.vecCi1Iter * srcCiStride;
        if constexpr (Intf::formatFmap == ConvFormat::NCHW) {
            baseOffset += self_->ctx.cinAL1Iter * self_->ctx.convTiling->cinOffsetBlockInGM;
            if (self_->ctx.vecId == 1) {
                baseOffset += self_->ctx.currentVec0Ci * self_->ctx.convTiling->orgHixWi;
            }
        } else {
            baseOffset += self_->ctx.cinAL1Iter * self_->ctx.cinAL1;
            if (self_->ctx.vecId == 1) {
                baseOffset += self_->ctx.currentVec0Ci;
            }
        }
 
        hoL1Idx = self_->ctx.hoAL1Iter * self_->ctx.convTiling->hoL1 * self_->ctx.convTiling->strideH;
        woL1Idx = self_->ctx.woAL1Iter * self_->ctx.convTiling->woL1 * self_->ctx.convTiling->strideW;
        uint32_t dstHoStride = AlignB(self_->ctx.currentWoL1, BLOCK_L0_M) * Intf::k0;
        uint32_t dstKwStride = self_->ctx.currentHoL1xWoL1Align * Intf::k0;
        uint32_t dstKhStride = dstKwStride * self_->ctx.convTiling->kwUb;
        copyParams.loopInfo.loopDstStride[NDDMA_LOOP1_INDEX] = dstKwStride;
        copyParams.loopInfo.loopDstStride[NDDMA_LOOP2_INDEX] = dstKhStride;
        uint32_t srcOffset = 0;

        for (uint16_t hoIdx = 0; hoIdx < self_->ctx.currentHoL1; hoIdx++) {
            for (uint16_t woIdx = 0; woIdx < self_->ctx.currentWoL1; woIdx++) {
                if (IsOnPadAndUpdateKernelSize(hoIdx, woIdx)) {
                    continue;
                }
                copyParams.loopInfo.loopSize[NDDMA_LOOP1_INDEX] =
                    self_->ctx.convTiling->kwUb - kwUbOnPadLeft - kwUbOnPadRight;
                copyParams.loopInfo.loopSize[NDDMA_LOOP2_INDEX] =
                    self_->ctx.convTiling->khUb - khUbOnPadTop - khUbOnPadBottom;
                if constexpr (Intf::formatFmap == ConvFormat::NCHW) {
                    srcOffset = baseOffset + hIdxWithPadStart * self_->ctx.orgWi + wIdxWithPadStart;
                } else {
                    srcOffset = baseOffset + hIdxWithPadStart * self_->ctx.orgWi * self_->ctx.convTiling->orgCi +
                        wIdxWithPadStart * self_->ctx.convTiling->orgCi;
                }
                uint32_t dstOffset = hoIdx * dstHoStride + woIdx * Intf::k0;
                if (unlikely(khUbOnPadTop != 0)) {
                    dstOffset += khUbOnPadTop * dstKhStride;
                }
                if (unlikely(kwUbOnPadLeft != 0)) {
                    dstOffset += kwUbOnPadLeft * dstKwStride;
                }
                DataCopy<typename Intf::FmapT, NDDMA_DIMS_LOAD_FMAP, kDefaultMultiCopyConfig>(
                    self_->ctx.img2ColTensor[dstOffset], self_->ctx.agm[srcOffset], copyParams);
            }
        }
    }

private:
    __aicore__ inline bool IsCiTail()
    {
        return (self_->ctx.kAL1Iter == self_->ctx.maxKAL1Iter && self_->ctx.vecCi1Iter == self_->ctx.maxVecCi1Iter);
    }

    __aicore__ inline bool IsOnPadAndUpdateKernelSize(uint16_t hoIdx, uint16_t woIdx)
    {
        hIdxWithPadStart = hoL1Idx + hoIdx * self_->ctx.convTiling->strideH + srckhAL1IterOffset;
        wIdxWithPadStart = woL1Idx + woIdx * self_->ctx.convTiling->strideW + srckwAL1IterOffset;

        hIdxWithPadStartOrg = hIdxWithPadStart + self_->ctx.hiStartPos;
        wIdxWithPadStartOrg = wIdxWithPadStart + self_->ctx.wiStartPos;
        hIdxWithPadStart = self_->ctx.hiStartPos < 0 ? hIdxWithPadStart + self_->ctx.hiStartPos : hIdxWithPadStart;
        wIdxWithPadStart = self_->ctx.wiStartPos < 0 ? wIdxWithPadStart + self_->ctx.wiStartPos : wIdxWithPadStart;
        hIdxWithPadEndOrg = hIdxWithPadStartOrg + (self_->ctx.convTiling->khUb - 1) * self_->ctx.convTiling->dilationH;
        wIdxWithPadEndOrg = wIdxWithPadStartOrg + (self_->ctx.convTiling->kwUb - 1) * self_->ctx.convTiling->dilationW;

        kwUbOnPadLeft = 0;
        kwUbOnPadRight = 0;
        khUbOnPadTop = 0;
        khUbOnPadBottom = 0;
        if (unlikely(wIdxWithPadEndOrg < 0 || wIdxWithPadStartOrg >= static_cast<int64_t>(self_->ctx.orgWi) ||
                     hIdxWithPadEndOrg < 0 || hIdxWithPadStartOrg >=
                     static_cast<int64_t>(static_cast<int64_t>(self_->ctx.orgHi)))) {
            return true;
        }
        if (likely(hIdxWithPadStartOrg >= 0 && hIdxWithPadEndOrg < static_cast<int64_t>(self_->ctx.orgHi) &&
                   wIdxWithPadStartOrg >= 0 && wIdxWithPadEndOrg < static_cast<int64_t>(self_->ctx.orgWi))) {
            return false;
        }
        UpdateRealKernelSize();

        return false;
    }

    __aicore__ inline void UpdateRealKernelSize()
    {
        // update real khUbSize in fmap.
        if (hIdxWithPadStartOrg < 0 && hIdxWithPadEndOrg < static_cast<int64_t>(self_->ctx.orgHi)) {
            khUbOnPadTop = (-hIdxWithPadStartOrg - 1) / self_->ctx.convTiling->dilationH + 1;
            hIdxWithPadStart += khUbOnPadTop * self_->ctx.convTiling->dilationH;
        } else if (hIdxWithPadStartOrg >= 0 && hIdxWithPadEndOrg >= static_cast<int64_t>(self_->ctx.orgHi)) {
            uint32_t khUbOnFmapCountTemp = (static_cast<int64_t>(self_->ctx.orgHi) - 1 - hIdxWithPadStartOrg) /
            self_->ctx.convTiling->dilationH + 1;
            khUbOnPadBottom = self_->ctx.convTiling->khUb - khUbOnFmapCountTemp;
        } else if (hIdxWithPadStartOrg < 0 && hIdxWithPadEndOrg >= static_cast<int64_t>(self_->ctx.orgHi)) {
            khUbOnPadTop = (-hIdxWithPadStartOrg - 1) / self_->ctx.convTiling->dilationH + 1;
            uint32_t khUbOnPadTopAndFmapCountTemp = (static_cast<int64_t>(self_->ctx.orgHi) - 1 - hIdxWithPadStartOrg) /
                self_->ctx.convTiling->dilationH + 1;
            khUbOnPadBottom = self_->ctx.convTiling->khUb - khUbOnPadTopAndFmapCountTemp;
        }

        // update real kwUbSize in fmap.
        if (wIdxWithPadStartOrg < 0 && wIdxWithPadEndOrg < static_cast<int64_t>(self_->ctx.orgWi)) {
            kwUbOnPadLeft = (-wIdxWithPadStartOrg - 1) / self_->ctx.convTiling->dilationW + 1;
            wIdxWithPadStart += kwUbOnPadLeft * self_->ctx.convTiling->dilationW;
        } else if (wIdxWithPadStartOrg >= 0 && wIdxWithPadEndOrg >= static_cast<int64_t>(self_->ctx.orgWi)) {
            uint32_t kwUbOnFmapCountTemp = (static_cast<int64_t>(self_->ctx.orgWi) - 1 - wIdxWithPadStartOrg) /
                self_->ctx.convTiling->dilationW + 1;
            kwUbOnPadRight = self_->ctx.convTiling->kwUb - kwUbOnFmapCountTemp;
        } else if (wIdxWithPadStartOrg < 0 && wIdxWithPadEndOrg >= static_cast<int64_t>(self_->ctx.orgWi)) {
            kwUbOnPadLeft = (-wIdxWithPadStartOrg - 1) / self_->ctx.convTiling->dilationW + 1;
            uint32_t kwUbOnPadLeftAndFmapCountTemp =
                (static_cast<int64_t>(self_->ctx.orgWi) - 1 - wIdxWithPadStartOrg) /
                    self_->ctx.convTiling->dilationW + 1;
            kwUbOnPadRight = self_->ctx.convTiling->kwUb - kwUbOnPadLeftAndFmapCountTemp;
        }        
    }

private:
    Intf *self_ = nullptr;
    // MultiCopyParams<typename Intf::FmapT, NDDMA_DIMS_BASE> copyParams;
    MultiCopyParams<typename Intf::FmapT, NDDMA_DIMS_LOAD_FMAP> copyParams;
    uint32_t srcCiStride = 0;
    uint32_t srcKhStride = 0;
    uint32_t srcKwStride = 0;
    uint32_t hIdxWithPad = 0;
    uint32_t wIdxWithPad = 0;
    int64_t wIdxWithPadStart = 0;
    int64_t wIdxWithPadEnd = 0;
    int64_t hIdxWithPadStart = 0;
    int64_t hIdxWithPadEnd = 0;
    int64_t hIdxWithPadStartOrg = 0;
    int64_t wIdxWithPadStartOrg = 0;
    int64_t hIdxWithPadEndOrg = 0;
    int64_t wIdxWithPadEndOrg = 0;
 
    uint32_t hoL1Idx = 0;
    uint32_t woL1Idx = 0;
    uint32_t srckhAL1IterOffset = 0;
    uint32_t srckwAL1IterOffset = 0;
    uint32_t kwUbOnFmap = 0;
    uint32_t kwUbOnPadLeft = 0;
    uint32_t kwUbOnPadRight = 0;
    uint32_t khUbOnPadTop = 0;
    uint32_t khUbOnPadBottom = 0;
};

template <class Intf>
class DmaUB2L1Tools {
public:
    __aicore__ inline DmaUB2L1Tools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;

        dstStrideBase = (self_->ctx.convTiling->kwL1 - self_->ctx.convTiling->kwUb) * Intf::k0;
        ci1StrideBase = self_->ctx.convTiling->khL1 * self_->ctx.convTiling->kwL1 * Intf::k0;
        khStrideBase = self_->ctx.convTiling->khUb * self_->ctx.convTiling->kernelW * Intf::k0;
        kwStrideBase = self_->ctx.convTiling->kwUb * Intf::k0;
    }

    __aicore__ inline void LoadUB2L1()
    {
        if (unlikely(self_->ctx.isFirstIterate)) {
            copyParams.blockCount = self_->ctx.convTiling->khUb;
            copyParams.srcStride = 0;
        }
        copyParams.blockLen = self_->ctx.currentHoL1xWoL1Align * self_->ctx.convTiling->kwUb;
        copyParams.dstStride = self_->ctx.currentHoL1xWoL1Align * dstStrideBase;

        uint64_t dstOffset = self_->ctx.vecCi1Iter * self_->ctx.currentHoL1xWoL1Align * ci1StrideBase +
            self_->ctx.vecKhIter * self_->ctx.currentHoL1xWoL1Align * khStrideBase +
            self_->ctx.vecKwIter * self_->ctx.currentHoL1xWoL1Align * kwStrideBase;

        if (self_->ctx.vecId == 1) {
            dstOffset += self_->ctx.vec0TotalSize;
        }

        DataCopy<typename Intf::FmapT>(self_->ctx.al1[dstOffset], self_->ctx.img2ColTensor, copyParams);
    }

private:
    Intf *self_ = nullptr;

    DataCopyParams copyParams;

    uint32_t dstStrideBase = 0;
    uint32_t ci1StrideBase = 0;
    uint32_t khStrideBase = 0;
    uint32_t kwStrideBase = 0;
};

}; // namespace Conv2dFunc

#endif // CONV2D_V2_INSTR_DMA_IMPL_H