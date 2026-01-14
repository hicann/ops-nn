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
 * \file conv2d_v2_instr_impl.h
 * \brief
 */

#ifndef CONV2D_V2_INSTR_IMPL_H
#define CONV2D_V2_INSTR_IMPL_H

#include "conv2d_v2_instr_base_impl.h"

namespace Conv2dFunc {
using namespace AscendC;
using namespace conv;

template <class Intf>
class LoadAL1ToolsHWMode : public LoadAL1ToolsBase<Intf> {
public:
    __aicore__ inline LoadAL1ToolsHWMode() {}
    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
        this->SetIntf(self);
    }

    __aicore__ inline void UpdatePaddingHiWiAL1()
    {
        paddingHiAL1 = (self_->ctx.currentHoL1 - 1) * self_->ctx.convTiling->strideH + self_->ctx.dilatedKernelH;
        if constexpr (Intf::c04Flag) {
            if (self_->ctx.convTiling->orgWo <= self_->ctx.convTiling->woL1) {
                paddingWiAL1 = self_->ctx.convTiling->orgWi + self_->ctx.convTiling->padLeft +
                               self_->ctx.convTiling->padRight;
            } else {
                paddingWiAL1 = (self_->ctx.currentWoL1 - 1) * self_->ctx.convTiling->strideW +
                               self_->ctx.dilatedKernelW;
            }
        } else {
            paddingWiAL1 = (self_->ctx.currentWoL1 - 1) * self_->ctx.convTiling->strideW +
                            self_->ctx.dilatedKernelW;
        }
    }

    __aicore__ inline void UpdatePadIdxAL1(uint64_t woAL1Iter, uint64_t hoAL1Iter)
    {
        int64_t hiRealStartPos = self_->ctx.hiStartPos > 0 ? 0 : self_->ctx.hiStartPos;
        // hiTopPadIdx is inaccurate when hiStartPos > 0
        hiTopPadIdx = hoAL1Iter * self_->ctx.convTiling->hoL1 * self_->ctx.convTiling->strideH + hiRealStartPos;
        int64_t wiRealStartPos = self_->ctx.wiStartPos > 0 ? 0 : self_->ctx.wiStartPos;
        // wiLeftPadIdx is inaccurate when wiStartPos > 0
        if (unlikely(self_->ctx.woL1SmallTail != 0 && woAL1Iter == self_->ctx.maxWoL1Iter)) {
            wiLeftPadIdx = ((woAL1Iter - 1) * self_->ctx.convTiling->woL1 + self_->ctx.woAL1Tail) *
                            self_->ctx.convTiling->strideW + wiRealStartPos;
        } else {
            wiLeftPadIdx = woAL1Iter * self_->ctx.convTiling->woL1 * self_->ctx.convTiling->strideW + wiRealStartPos;
        }
        hiBottomPadIdx = hiTopPadIdx + paddingHiAL1;
        wiRightPadIdx = wiLeftPadIdx + paddingWiAL1;
    }

    __aicore__ inline void UpdateHiWiAL1()
    {
        // Calculate AL1 pads
        padTopL1 = hiTopPadIdx < 0 ? (0 - hiTopPadIdx) : 0;
        hiBottomPadIdx = self_->ctx.hiStartPos > 0 ? hiBottomPadIdx + self_->ctx.hiStartPos : hiBottomPadIdx;
        padBottomL1 = hiBottomPadIdx > self_->ctx.orgHi ? hiBottomPadIdx - self_->ctx.orgHi : 0;
        padLeftL1 = wiLeftPadIdx < 0 ? (0 - wiLeftPadIdx) : 0;
        wiRightPadIdx = self_->ctx.wiStartPos > 0 ? wiRightPadIdx + self_->ctx.wiStartPos : wiRightPadIdx;
        padRightL1 = wiRightPadIdx > self_->ctx.orgWi ? (wiRightPadIdx - self_->ctx.orgWi) : 0;
        if (padTopL1 >= paddingHiAL1 || padBottomL1 >= paddingHiAL1 ||
            padLeftL1 >= paddingWiAL1 || padRightL1 >= paddingWiAL1) {
            this->allPadFlag = true;
            return;
        }
        // Calculate hiAL1, wiAL1
        hiLoadL1 = paddingHiAL1 - padTopL1 - padBottomL1;
        hiLoadL1 = hiLoadL1 > self_->ctx.orgHi ? self_->ctx.orgHi : hiLoadL1;
        wiLoadL1 = paddingWiAL1 - padLeftL1 - padRightL1;
    }

    __aicore__ inline void SetDn2NzIntriParams(Dn2NzParams &intriParams, uint64_t kAL1Iter)
    {
        if (likely(wiLoadL1 == self_->ctx.orgWi)) {
            intriParams.dnNum = 1;
            intriParams.nValue = hiLoadL1 * wiLoadL1;
            intriParams.srcDnMatrixStride = 0;
            intriParams.dstNzMatrixStride = 0;
        } else {
            intriParams.dnNum = hiLoadL1;
            intriParams.nValue = wiLoadL1;
            intriParams.srcDnMatrixStride = self_->ctx.orgWi;
            intriParams.dstNzMatrixStride = wiLoadL1 * Intf::k0;
        }

        if constexpr (Intf::groupOptFlag) {
            intriParams.dValue = kAL1Iter == self_->ctx.maxKAL1Iter ?
                (self_->ctx.kAL1Tail / self_->ctx.convTiling->kernelHxkernelW) : self_->ctx.convTiling->cinAInCore;
        } else {
            intriParams.dValue = kAL1Iter == self_->ctx.maxKAL1Iter ?
                self_->ctx.convTiling->cinATailInCore : self_->ctx.convTiling->cinAInCore;
        }
        intriParams.srcDValue = self_->ctx.convTiling->orgHixWi;
        intriParams.dstNzC0Stride = hiLoadL1 * wiLoadL1;
        intriParams.dstNzNStride = 1;
    }

    __aicore__ inline void SetDn2NzIntriParamsC04(Dn2NzParams &intriParams)
    {
        intriParams.dnNum = 1;
        intriParams.nValue = hiLoadL1 * wiLoadL1;
        intriParams.dValue = self_->ctx.convTiling->singleCoreCi;
        intriParams.srcDnMatrixStride = 0;
        intriParams.srcDValue = self_->ctx.convTiling->orgHixWi;
        intriParams.dstNzC0Stride = 0;
        intriParams.dstNzNStride = 1;
        intriParams.dstNzMatrixStride = 0;
    }

    __aicore__ inline void SetNd2NzIntriParamsC04(Nd2NzParams &intriParams)
    {
        intriParams.ndNum = 1;
        intriParams.nValue = hiLoadL1 * wiLoadL1;
        intriParams.dValue = self_->ctx.convTiling->singleCoreCi;
        intriParams.srcDValue = self_->ctx.convTiling->singleCoreCi;
    }

    __aicore__ inline void SetNd2NzIntriParams(Nd2NzParams &intriParams, uint64_t kAL1Iter)
    {
        intriParams.ndNum = hiLoadL1;
        intriParams.nValue = wiLoadL1;
        intriParams.dValue = kAL1Iter == self_->ctx.maxKAL1Iter ?
            self_->ctx.convTiling->cinATailInCore : self_->ctx.convTiling->cinAInCore;
        intriParams.srcNdMatrixStride = self_->ctx.orgWi * self_->ctx.orgCi;
        intriParams.srcDValue = self_->ctx.orgCi;
        intriParams.dstNzC0Stride = hiLoadL1 * wiLoadL1;
        intriParams.dstNzNStride = 1;
        intriParams.dstNzMatrixStride = wiLoadL1 * Intf::k0;
    }

    __aicore__ inline void LoadAl1Data(uint64_t kAL1Iter, uint64_t batchIter)
    {
        // Calculate aL1GmOffset
        int64_t realHiTopGmIdx = hiTopPadIdx < 0 ? 0 : hiTopPadIdx;
        int64_t realWiTopGmIdx = wiLeftPadIdx < 0 ? 0 : wiLeftPadIdx;
        int64_t aL1GmOffset = kAL1Iter * self_->ctx.convTiling->cinOffsetBlockInGM +
                              realHiTopGmIdx * self_->ctx.orgWi + realWiTopGmIdx +
                              batchIter * self_->ctx.fmapOneBatchSize;

        Dn2NzParams intriParams;
        if constexpr(Intf::c04Flag) {
            SetDn2NzIntriParamsC04(intriParams);
            DataCopy<typename Intf::FmapT, true>(self_->ctx.al1, self_->ctx.agm[aL1GmOffset], intriParams);
        } else {
            SetDn2NzIntriParams(intriParams, kAL1Iter);
            DataCopy<typename Intf::FmapT>(self_->ctx.al1, self_->ctx.agm[aL1GmOffset], intriParams);
        }
    }

    __aicore__ inline void LoadAl1DataHWC(uint64_t kAL1Iter, uint64_t batchIter)
    {
        // Calculate aL1GmOffset
        int64_t realHiTopGmIdx = hiTopPadIdx < 0 ? 0 : hiTopPadIdx;
        int64_t realWiTopGmIdx = wiLeftPadIdx < 0 ? 0 : wiLeftPadIdx;
        int64_t aL1GmOffset = batchIter * self_->ctx.fmapOneBatchSize +
                              realHiTopGmIdx * self_->ctx.orgWi * self_->ctx.orgCi +
                              realWiTopGmIdx * self_->ctx.orgCi +
                              kAL1Iter * self_->ctx.convTiling->cinAInCore;

        Nd2NzParams intriParams;
        if constexpr(Intf::c04Flag) {
            SetNd2NzIntriParamsC04(intriParams);
            DataCopy<typename Intf::FmapT, true>(self_->ctx.al1, self_->ctx.agm[aL1GmOffset], intriParams);
        } else {
            SetNd2NzIntriParams(intriParams, kAL1Iter);
            DataCopy<typename Intf::FmapT>(self_->ctx.al1, self_->ctx.agm[aL1GmOffset], intriParams);
        }
    }

    __aicore__ inline void LoadAL1()
    {
        LoadAL1(self_->ctx.kAL1Iter, self_->ctx.woAL1Iter, self_->ctx.hoAL1Iter, self_->ctx.batchIter);
    }

    __aicore__ inline void LoadAL1(uint64_t kAL1Iter, uint64_t woAL1Iter, uint64_t hoAL1Iter, uint64_t batchIter)
    {
        UpdatePaddingHiWiAL1();
        UpdatePadIdxAL1(woAL1Iter, hoAL1Iter);
        UpdateHiWiAL1();
        this->SetLoad3dFMatrix(padLeftL1, padRightL1, padTopL1, hiLoadL1, wiLoadL1);

        if (this->allPadFlag) {
            this->SetPadData();
            return;
        }

        if constexpr (Intf::formatFmap == ConvFormat::NCHW) {
            LoadAl1Data(kAL1Iter, batchIter);
        } else {
            LoadAl1DataHWC(kAL1Iter, batchIter);
        }
    }

private:
    Intf *self_ = nullptr;

    int64_t padTopL1 = 0;
    int64_t padLeftL1 = 0;
    int64_t padRightL1 = 0;
    int64_t hiTopPadIdx = 0;
    int64_t wiLeftPadIdx = 0;
    int64_t padBottomL1 = 0;
    int64_t hiBottomPadIdx = 0;
    int64_t wiRightPadIdx = 0;
    int64_t paddingHiAL1 = 0;
    int64_t paddingWiAL1 = 0;
    int64_t hiLoadL1 = 0;
    int64_t wiLoadL1 = 0;
};

template <class Intf>
class LoadAL1ToolsMMode : public LoadAL1ToolsBase<Intf> {
public:
    __aicore__ inline LoadAL1ToolsMMode()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
        this->SetIntf(self);
    }

    __aicore__ inline void LoadAL1()
    {
        LoadAL1(self_->ctx.kAL1Iter, self_->ctx.mAL1Iter, self_->ctx.batchIter);
    }

    __aicore__ inline void LoadAL1(uint64_t kAL1Iter, uint64_t mAL1Iter, uint64_t batchIter)
    {
        if (unlikely(self_->ctx.mAL1UpdateFlag)) {
            CalcHiL1Pad(mAL1Iter);
            this->SetLoad3dFMatrix(self_->ctx.convTiling->padLeft, self_->ctx.convTiling->padRight, padTopL1,
                                   hiLoadL1, self_->ctx.orgWi);
            self_->ctx.mAL1UpdateFlag = false;
        }

        if (this->allPadFlag) {
            this->SetPadData();
            return;
        }

        if constexpr (Intf::formatFmap == ConvFormat::NCHW) {
            uint64_t aL1GmOffset = batchIter * self_->ctx.fmapOneBatchSize +
                                   kAL1Iter * self_->ctx.convTiling->cinOffsetBlockInGM +
                                   hiIdx * self_->ctx.orgWi;

            Dn2NzParams intriParams;

            if constexpr (Intf::c04Flag) {
                LoadAL1DataC04(intriParams, aL1GmOffset);
            } else {
                SetDn2NzIntriParams(intriParams, kAL1Iter);
                DataCopy<typename Intf::FmapT>(self_->ctx.al1, self_->ctx.agm[aL1GmOffset], intriParams);
            }
        } else {
            uint64_t aL1GmOffset = batchIter * self_->ctx.fmapOneBatchSize +
                                   hiIdx * self_->ctx.orgWi * self_->ctx.orgCi + 
                                   kAL1Iter * self_->ctx.convTiling->cinAInCore;

            Nd2NzParams intriParams;
            if constexpr (Intf::c04Flag) {
                SetNd2NzIntriParamsC04(intriParams);
                DataCopy<typename Intf::FmapT, true>(self_->ctx.al1, self_->ctx.agm[aL1GmOffset], intriParams);
            } else {
                SetNd2NzIntriParams(intriParams, kAL1Iter);
                DataCopy<typename Intf::FmapT>(self_->ctx.al1, self_->ctx.agm[aL1GmOffset], intriParams);
            }
        }
    }

private:
    __aicore__ inline void CalcHiL1Pad(uint64_t mAL1Iter)
    {
        padTopL1 = 0;
        padBottomL1 = 0;

        uint64_t currentM = self_->ctx.mStartPos + mAL1Iter * self_->ctx.mAL1;
        uint64_t currentML1 = IsMAL1Tail(mAL1Iter) ? self_->ctx.mAL1Tail : self_->ctx.mAL1;
        uint64_t hoStartIdx = currentM / self_->ctx.orgWo;
        uint64_t hoEndIdx = CeilDiv(currentM + currentML1, self_->ctx.orgWo);
        hiLoadL1 = ((hoEndIdx - hoStartIdx) - 1) * self_->ctx.convTiling->strideH + self_->ctx.dilatedKernelH;

        uint64_t hiStartIdxWithPad = hoStartIdx * self_->ctx.convTiling->strideH;
        uint64_t hiEndIdxWithPad = hiStartIdxWithPad + hiLoadL1;
        hiIdx = hiStartIdxWithPad - self_->ctx.convTiling->padTop;

        if (hiEndIdxWithPad <= self_->ctx.convTiling->padTop ||
            hiStartIdxWithPad >= self_->ctx.orgHi + self_->ctx.convTiling->padTop) {
            this->allPadFlag = true;
            return;
        }

        if (hiStartIdxWithPad < self_->ctx.convTiling->padTop) {
            hiIdx = 0;
            padTopL1 = self_->ctx.convTiling->padTop - hiStartIdxWithPad;
            hiLoadL1 -= padTopL1;
        }

        if (hiEndIdxWithPad > self_->ctx.orgHi + self_->ctx.convTiling->padTop) {
            padBottomL1 = hiEndIdxWithPad - (self_->ctx.orgHi + self_->ctx.convTiling->padTop);
            hiLoadL1 -= padBottomL1;
        }
    }

    __aicore__ inline void LoadAL1DataC04(Dn2NzParams &intriParams, uint64_t aL1GmOffset)
    {
        uint64_t aL1Mi = hiLoadL1 * self_->ctx.orgWi;
        if (aL1Mi > N_VALUE_MAX) {
            uint64_t hiLoadPerStep = N_VALUE_MAX / self_->ctx.orgWi;
            uint64_t step = CeilDiv(hiLoadL1, hiLoadPerStep);
            uint64_t hiLoadTail = hiLoadL1 % hiLoadPerStep;
            if (hiLoadTail == 0) {
                SetDn2NzIntriParamsC04(intriParams, step, hiLoadPerStep * self_->ctx.orgWi);
                DataCopy<typename Intf::FmapT, true>(self_->ctx.al1, self_->ctx.agm[aL1GmOffset], intriParams);
            } else {
                SetDn2NzIntriParamsC04(intriParams, step - 1, hiLoadPerStep * self_->ctx.orgWi);
                DataCopy<typename Intf::FmapT, true>(self_->ctx.al1, self_->ctx.agm[aL1GmOffset], intriParams);

                // hiLoadTail
                uint64_t offset = (hiLoadL1 - hiLoadTail) * self_->ctx.orgWi;
                uint64_t aL1Offset = offset * C04_CIN_SIZE;
                aL1GmOffset += offset;
                SetDn2NzIntriParamsC04(intriParams, 1, hiLoadTail * self_->ctx.orgWi);
                DataCopy<typename Intf::FmapT, true>(
                    self_->ctx.al1[aL1Offset], self_->ctx.agm[aL1GmOffset], intriParams);
            }
        } else {
            SetDn2NzIntriParamsC04(intriParams, 1, aL1Mi);
            DataCopy<typename Intf::FmapT, true>(self_->ctx.al1, self_->ctx.agm[aL1GmOffset], intriParams);
        }
    }

    __aicore__ inline void SetDn2NzIntriParams(Dn2NzParams &intriParams, uint64_t kAL1Iter)
    {
        uint64_t aL1Mi = hiLoadL1 * self_->ctx.orgWi;
        intriParams.dnNum = 1;
        intriParams.nValue = aL1Mi;
        if constexpr (Intf::groupOptFlag) {
            intriParams.dValue = IsKAL1Tail(kAL1Iter) ?
                (self_->ctx.kAL1Tail / self_->ctx.convTiling->kernelHxkernelW) : self_->ctx.convTiling->cinAInCore;
        } else {
            intriParams.dValue = IsKAL1Tail(kAL1Iter) ?
                self_->ctx.convTiling->cinATailInCore : self_->ctx.convTiling->cinAInCore;
        }
        intriParams.srcDValue = self_->ctx.convTiling->orgHixWi;
        intriParams.dstNzC0Stride = aL1Mi;
        intriParams.dstNzNStride = 1;
    }

    __aicore__ inline void SetDn2NzIntriParamsC04(Dn2NzParams &intriParams, uint64_t dnNum, uint64_t nValue)
    {
        intriParams.dnNum = dnNum;
        intriParams.nValue = nValue;
        intriParams.dValue = self_->ctx.convTiling->singleCoreCi;
        intriParams.srcDValue = self_->ctx.convTiling->orgHixWi;
        intriParams.dstNzNStride = 1;
    }

    __aicore__ inline void SetNd2NzIntriParamsC04(Nd2NzParams &intriParams)
    {
        intriParams.ndNum = 1;
        intriParams.nValue = hiLoadL1 * self_->ctx.orgWi;
        intriParams.dValue = self_->ctx.convTiling->singleCoreCi;
        intriParams.srcDValue = self_->ctx.convTiling->singleCoreCi;
    }

    __aicore__ inline void SetNd2NzIntriParams(Nd2NzParams &intriParams, uint64_t kAL1Iter)
    {
        uint64_t aL1Mi = hiLoadL1 * self_->ctx.orgWi;
        intriParams.ndNum = 1;
        intriParams.nValue = aL1Mi;
        intriParams.dValue = IsKAL1Tail(kAL1Iter) ?
            self_->ctx.convTiling->cinATailInCore : self_->ctx.convTiling->cinAInCore;
        intriParams.srcDValue = self_->ctx.convTiling->orgCi;
        intriParams.dstNzNStride = 1;
        intriParams.dstNzC0Stride = aL1Mi;
    }

    __aicore__ inline bool IsMAL1Tail(uint64_t mAL1Iter)
    {
        return mAL1Iter == self_->ctx.maxMAL1Iter;
    }

    __aicore__ inline bool IsKAL1Tail(uint64_t kAL1Iter)
    {
        return kAL1Iter == self_->ctx.maxKAL1Iter;
    }

private:
    Intf *self_ = nullptr;
    uint64_t hiLoadL1 = 0;
    uint64_t padTopL1 = 0;
    uint64_t padBottomL1 = 0;
    uint64_t hiIdx = 0;
};

};

#endif // CONV2D_V2_INSTR_IMPL_H