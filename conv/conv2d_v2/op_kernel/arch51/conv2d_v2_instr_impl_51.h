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
 * \file conv2d_v2_instr_impl_51.h
 * \brief
 */

#ifndef CONV2D_V2_INSTR_IMPL_5162_H_
#define CONV2D_V2_INSTR_IMPL_5162_H_

#include "../arch35/conv2d_v2_instr_impl.h"

namespace Conv2dFunc {

template <class Intf>
__aicore__ inline void LoadAL1ToolsHWMode<Intf>::UpdatePaddingHiWiAL1(IterateParam& iterParam)
{
    paddingHiAL1 = (iterParam.currentHoL1 - 1) * self_->ctx.convTilingData->strideH + iterParam.dilatedKernelH;
    paddingWiAL1 = (iterParam.currentWoL1 - 1) * self_->ctx.convTilingData->strideW + self_->ctx.dilatedKernelW;
}

template <class Intf>
__aicore__ inline void LoadAL1ToolsHWMode<Intf>::UpdatePadIdxAL1(IterateParam& iterParam, uint64_t woAL1Iter,
                                                                 uint64_t hoAL1Iter)
{
    int64_t hiRealStartPos = iterParam.hiStartPos > 0 ? 0 : iterParam.hiStartPos;
    // hiTopPadIdx is inaccurate when hiStartPos > 0
    hiTopPadIdx = hoAL1Iter * self_->ctx.convTilingData->hoL1 * self_->ctx.convTilingData->strideH + hiRealStartPos;
    int64_t wiRealStartPos = iterParam.wiStartPos > 0 ? 0 : iterParam.wiStartPos;
    // wiLeftPadIdx is inaccurate when wiStartPos > 0
    if (unlikely(iterParam.woL1SmallTail != 0 && woAL1Iter == iterParam.maxWoL1Iter)) {
        wiLeftPadIdx = ((woAL1Iter - 1) * self_->ctx.convTilingData->woL1 + iterParam.woAL1Tail) *
                           self_->ctx.convTilingData->strideW +
                       wiRealStartPos;
    } else {
        wiLeftPadIdx = woAL1Iter * self_->ctx.convTilingData->woL1 * self_->ctx.convTilingData->strideW +
                       wiRealStartPos;
    }
    hiBottomPadIdx = hiTopPadIdx + paddingHiAL1;
    wiRightPadIdx = wiLeftPadIdx + paddingWiAL1;
}

template <class Intf>
__aicore__ inline void LoadAL1ToolsHWMode<Intf>::UpdateHiWiAL1(IterateParam& iterParam)
{
    // Calculate AL1 pads
    this->padTopL1 = hiTopPadIdx < 0 ? (0 - hiTopPadIdx) : 0;
    hiBottomPadIdx = iterParam.hiStartPos > 0 ? hiBottomPadIdx + iterParam.hiStartPos : hiBottomPadIdx;
    padBottomL1 = hiBottomPadIdx > self_->ctx.convTilingData->orgHi ?
                      hiBottomPadIdx - self_->ctx.convTilingData->orgHi :
                      0;
    this->padLeftL1 = wiLeftPadIdx < 0 ? (0 - wiLeftPadIdx) : 0;
    wiRightPadIdx = iterParam.wiStartPos > 0 ? wiRightPadIdx + iterParam.wiStartPos : wiRightPadIdx;
    this->padRightL1 = wiRightPadIdx > self_->ctx.convTilingData->orgWi ?
                           (wiRightPadIdx - self_->ctx.convTilingData->orgWi) :
                           0;
    if (this->padTopL1 >= paddingHiAL1 || padBottomL1 >= paddingHiAL1 || this->padLeftL1 >= paddingWiAL1 ||
        this->padRightL1 >= paddingWiAL1) {
        this->allPadFlag = true;
        return;
    }
    this->allPadFlag = false;
    // Calculate hiAL1, wiAL1
    this->hiLoadL1 = paddingHiAL1 - this->padTopL1 - padBottomL1;
    this->hiLoadL1 = this->hiLoadL1 > self_->ctx.convTilingData->orgHi ? self_->ctx.convTilingData->orgHi :
                                                                         this->hiLoadL1;
    this->wiLoadL1 = paddingWiAL1 - this->padLeftL1 - this->padRightL1;
}

template <class Intf>
__aicore__ inline void LoadAL1ToolsHWMode<Intf>::SetDn2NzIntriParams(IterateParam& iterParam, Dn2NzParams& intriParams,
                                                                     uint64_t kAL1Iter)
{
    if (likely(this->wiLoadL1 == self_->ctx.convTilingData->orgWi)) {
        intriParams.dnNum = 1;
        intriParams.nValue = this->hiLoadL1 * this->wiLoadL1;
        intriParams.srcDnMatrixStride = 0;
        intriParams.dstNzMatrixStride = 0;
    } else {
        intriParams.dnNum = this->hiLoadL1;
        intriParams.nValue = this->wiLoadL1;
        intriParams.srcDnMatrixStride = self_->ctx.convTilingData->orgWi;
        intriParams.dstNzMatrixStride = this->wiLoadL1 * Intf::k0FmapTail;
    }

    if constexpr (Intf::groupOptFlag) {
        intriParams.dValue = kAL1Iter == iterParam.maxKAL1Iter ?
                                 (iterParam.kAL1Tail / self_->ctx.convTilingData->kernelHxkernelW) :
                                 self_->ctx.convTilingData->cinAInCore;
    } else {
        intriParams.dValue = kAL1Iter == iterParam.maxKAL1Iter ? self_->ctx.convTilingData->cinATailInCore :
                                                                 self_->ctx.convTilingData->cinAInCore;
    }
    intriParams.srcDValue = self_->ctx.convTilingData->orgHixWi;
    intriParams.dstNzC0Stride = this->hiLoadL1 * this->wiLoadL1;
    intriParams.dstNzNStride = 1;
}

template <class Intf>
__aicore__ inline void LoadAL1ToolsHWMode<Intf>::LoadAl1Data(IterateParam& iterParam, uint64_t kAL1Iter,
                                                             uint64_t batchIter, uint64_t groupOptIter)
{
    // Calculate aL1GmOffset
    int64_t realHiTopGmIdx = hiTopPadIdx < 0 ? 0 : hiTopPadIdx;
    int64_t realWiTopGmIdx = wiLeftPadIdx < 0 ? 0 : wiLeftPadIdx;
    int64_t aL1GmOffset = groupOptIter * self_->ctx.convTilingData->singleCoreCi * self_->ctx.convTilingData->orgWi *
                              self_->ctx.convTilingData->orgHi +
                          realHiTopGmIdx * self_->ctx.convTilingData->orgWi + realWiTopGmIdx;
    if constexpr (!Intf::isOneBatch) {
        aL1GmOffset += batchIter * self_->ctx.fmapOneBatchSize;
    }

    Dn2NzParams intriParams;
    SetDn2NzIntriParams(iterParam, intriParams, kAL1Iter);
    DataCopy<typename Intf::FmapT>(self_->ctx.al1, self_->ctx.agm[aL1GmOffset], intriParams);
}

template <class Intf>
__aicore__ inline void LoadAL1ToolsHWMode<Intf>::LoadAL1(IterateParam& iterParam)
{
    uint64_t kAL1Iter = iterParam.kAL1Iter;
    uint64_t woAL1Iter = iterParam.woAL1Iter;
    uint64_t hoAL1Iter = iterParam.hoAL1Iter;
    uint64_t batchIter = iterParam.batchIter;
    UpdatePaddingHiWiAL1(iterParam);
    UpdatePadIdxAL1(iterParam, woAL1Iter, hoAL1Iter);
    UpdateHiWiAL1(iterParam);

    ConvFMatrixBitModeParams fMatrixParam;
    fMatrixParam.SetHFmap(this->hiLoadL1);
    fMatrixParam.SetWFmap(this->wiLoadL1);
    iterParam.currentL1Hin = this->hiLoadL1 * this->wiLoadL1;
    fMatrixParam.SetLeftPaddingSize(this->padLeftL1);
    fMatrixParam.SetRightPaddingSize(this->padRightL1);
    fMatrixParam.SetTopPaddingSize(this->padTopL1);
    fMatrixParam.SetBottomPaddingSize(padBottomL1);
    SetConvFMatrix(fMatrixParam);

    typename Intf::FmapT paddingValue = Intf::isQuantScene ?
                                            static_cast<typename Intf::FmapT>(self_->ctx.convTilingData->offsetx) :
                                            0;
    ConvPaddingBitModeParams<typename Intf::FmapT> paddingParams;
    paddingParams.SetPaddingValue(paddingValue);
    SetConvPadding<typename Intf::FmapT>(paddingParams);
    LoadAl1Data(iterParam, kAL1Iter, batchIter, 0);
}

} // namespace Conv2dFunc

#endif // CONV2D_V2_INSTR_IMPL_5162_H_
