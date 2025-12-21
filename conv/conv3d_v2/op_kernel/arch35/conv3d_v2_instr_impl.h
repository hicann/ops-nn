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
 * \file conv3d_v2_instr_impl.h
 * \brief
 */

#ifndef CONV3D_V2_INSTR_IMPL_H
#define CONV3D_V2_INSTR_IMPL_H

#include "conv3d_v2_instr_base_impl.h"

namespace Conv3dFunc {
using namespace AscendC;
using namespace conv;

template <class Intf>
class LoadAL1ToolsMMode : public LoadAL1ToolsBase<Intf> {
public:
    __aicore__ inline LoadAL1ToolsMMode() {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
        this->SetIntf(self);
    }

    __aicore__ inline void LoadAL1Data()
    {
        uint64_t aL1GmPos = self_->ctx.batchIter * self_->ctx.fmapOneBatchSize+
                            this->cinIdx * self_->ctx.orgDi * self_->ctx.convTiling->orgHixWi +
                            this->diIdx * self_->ctx.convTiling->orgHixWi +
                            hiIdx * self_->ctx.orgWi;
        if (likely(this->dkLoadL1 > 0)) {
            Dn2NzParams intriParams;
            SetDn2NzIntriParams(intriParams);
            DataCopy<typename Intf::FmapT>(self_->ctx.al1[this->aL1Pos], self_->ctx.agm[aL1GmPos], intriParams);
        }
    }

    __aicore__ inline void LoadAL1DataHWC()
    {
        uint64_t aL1GmPos = self_->ctx.batchIter * self_->ctx.fmapOneBatchSize +
                            this->diIdx * self_->ctx.convTiling->orgHixWi * self_->ctx.orgCi +
                            hiIdx * self_->ctx.orgWi * self_->ctx.orgCi +
                            this->cinIdx;
        if (likely(this->dkLoadL1 > 0)) {
            Nd2NzParams intriParams;
            SetNd2NzIntriParams(intriParams);
            DataCopy<typename Intf::FmapT>(self_->ctx.al1[this->aL1Pos], self_->ctx.agm[aL1GmPos], intriParams);
        }
    }

    __aicore__ inline void LoadAL1()
    {
        this->aL1Pos = 0;
        this->CalcCiL1Pad();
        if (unlikely(self_->ctx.mAL1UpdateFlag)) {
            CalcHiL1Pad();
            this->SetLoad3dFMatrix(self_->ctx.convTiling->padLeft, self_->ctx.convTiling->padRight, padTopL1,
                                   hiLoadL1, self_->ctx.orgWi);
            self_->ctx.mAL1UpdateFlag = false;
        }

        if (this->allPadFlag) {
            this->SetPadData();
            return;
        }

        if (this->cin1LoadL1PadHead > 0) {
            this->Al1PadHeadSet2d(hiLoadL1, self_->ctx.orgWi);
        }

        if constexpr (Intf::formatOutput == ConvFormat::NCDHW) {
            LoadAL1Data();
        } else {
            LoadAL1DataHWC();
        }

        if (this->cin1LoadL1PadTail > 0) {
            this->aL1Pos += this->dkLoadL1 * AlignB(this->cinLoadL1, Intf::k0) * hiLoadL1 * self_->ctx.orgWi;
            this->Al1PadTailSet2d(hiLoadL1, self_->ctx.orgWi);
        }
    }

private:
    __aicore__ inline void CalcHiL1Pad()
    {
        padTopL1 = 0;
        padBottomL1 = 0;

        uint64_t currentM = self_->ctx.mStartPos + self_->ctx.mAL1Iter * self_->ctx.mAL1;
        uint64_t currentML1 = IsMAL1Tail() ? self_->ctx.mAL1Tail : self_->ctx.mAL1;
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

    __aicore__ inline void SetDn2NzIntriParams(Dn2NzParams &intriParams)
    {
        uint64_t aL1Mi = hiLoadL1 * self_->ctx.orgWi;
        intriParams.dnNum = this->dkLoadL1;
        intriParams.nValue = aL1Mi;
        intriParams.dValue = this->cinLoadL1;
        intriParams.dstNzC0Stride = aL1Mi;
        intriParams.dstNzNStride = 1;
        intriParams.dstNzMatrixStride = AlignB(this->cinLoadL1, Intf::k0) * aL1Mi;
        intriParams.srcDValue = self_->ctx.orgDi * self_->ctx.convTiling->orgHixWi;
        intriParams.srcDnMatrixStride = self_->ctx.convTiling->orgHixWi * self_->ctx.convTiling->dilationD;
    }

    __aicore__ inline void SetNd2NzIntriParams(Nd2NzParams& intriParams)
    {
        uint64_t aL1Mi = hiLoadL1 * self_->ctx.orgWi;
        intriParams.ndNum = this->dkLoadL1;
        intriParams.nValue = aL1Mi;
        intriParams.dValue = this->cinLoadL1;
        intriParams.dstNzC0Stride = aL1Mi;
        intriParams.dstNzNStride = 1;
        intriParams.dstNzMatrixStride = AlignB(this->cinLoadL1, Intf::k0) * aL1Mi;
        intriParams.srcDValue = self_->ctx.orgCi;
        intriParams.srcNdMatrixStride = self_->ctx.convTiling->orgHixWi * self_->ctx.convTiling->dilationD *
            self_->ctx.orgCi;
    }

    __aicore__ inline bool IsMAL1Tail()
    {
        return self_->ctx.mAL1Iter == self_->ctx.maxMAL1Iter;
    }

private:
    Intf *self_ = nullptr;
    uint64_t padTopL1 = 0;
    uint64_t padBottomL1 = 0;
    uint64_t hiLoadL1 = 0;
    uint64_t hiIdx = 0;
};

template <class Intf>
class LoadAL1ToolsHWmode : public LoadAL1ToolsBase<Intf> {
public:
    __aicore__ inline LoadAL1ToolsHWmode() {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
        this->SetIntf(self);
    }

    __aicore__ inline void CalcWiAL1Pad(uint64_t woAL1Iter)
    {
        paddingWiAL1 = (self_->ctx.currentWoL1 - 1) * self_->ctx.convTiling->strideW + self_->ctx.dilatedKernelW;
        if (unlikely(self_->ctx.woL1SmallTail != 0 && woAL1Iter == self_->ctx.maxWoL1Iter)) {
            wiLeftPadIdx = ((woAL1Iter - 1) * self_->ctx.convTiling->woL1 + self_->ctx.woAL1Tail) *
                self_->ctx.convTiling->strideW - self_->ctx.convTiling->padLeft;
        } else {
            wiLeftPadIdx = woAL1Iter * self_->ctx.convTiling->woL1 * self_->ctx.convTiling->strideW -
                self_->ctx.convTiling->padLeft;
        }
        wiRightPadIdx = wiLeftPadIdx + paddingWiAL1;
        padLeftL1 = wiLeftPadIdx < 0 ? (0 - wiLeftPadIdx) : 0;
        padRightL1 = wiRightPadIdx > self_->ctx.orgWi ? (wiRightPadIdx - self_->ctx.orgWi) : 0;
        if (padLeftL1 >= paddingWiAL1 || padRightL1 >= paddingWiAL1) {
            this->allPadFlag = true;
            return;
        }
        wiLoadL1 = paddingWiAL1 - padLeftL1 - padRightL1;
    }

    __aicore__ inline void CalcHiAL1Pad(uint64_t hoAL1Iter)
    {
        paddingHiAL1 = (self_->ctx.currentHoL1 - 1) * self_->ctx.convTiling->strideH + self_->ctx.dilatedKernelH;
        int64_t hiRealStartPos = self_->ctx.hiStartPos > 0 ? 0 : self_->ctx.hiStartPos;
        // hiTopPadIdx is inaccurate when hiStartPos > 0
        hiTopPadIdx = hoAL1Iter * self_->ctx.convTiling->hoL1 * self_->ctx.convTiling->strideH + hiRealStartPos;
        hiBottomPadIdx = hiTopPadIdx + paddingHiAL1;
        padTopL1 = hiTopPadIdx < 0 ? (0 - hiTopPadIdx) : 0;
        hiBottomPadIdx = self_->ctx.hiStartPos > 0 ? hiBottomPadIdx + self_->ctx.hiStartPos : hiBottomPadIdx;
        padBottomL1 = hiBottomPadIdx > self_->ctx.orgHi ? hiBottomPadIdx - self_->ctx.orgHi : 0;
        if (padTopL1 >= paddingHiAL1 || padBottomL1 >= paddingHiAL1) {
            this->allPadFlag = true;
            return;
        }
        hiLoadL1 = paddingHiAL1 - padTopL1 - padBottomL1;
        hiLoadL1 = hiLoadL1 > self_->ctx.orgHi ? self_->ctx.orgHi : hiLoadL1;
    }

    __aicore__ inline void LoadAL1()
    {
        CalcHiAL1Pad(self_->ctx.hoAL1Iter);
        CalcWiAL1Pad(self_->ctx.woAL1Iter);
        this->CalcCiL1Pad();
        this->SetLoad3dFMatrix(padLeftL1, padRightL1, padTopL1, hiLoadL1, wiLoadL1);
        if (this->allPadFlag) {
            this->SetPadData();
            return;
        }
        LoadAl1Data();
    }
private:
    __aicore__ inline void LoadAl1Data()
    {
        int64_t realHiTopGmIdx = hiTopPadIdx < 0 ? 0 : hiTopPadIdx;
        int64_t realWiTopGmIdx = wiLeftPadIdx < 0 ? 0 : wiLeftPadIdx;

        int64_t al1Offset = AlignB(this->cinLoadL1, Intf::k0) * hiLoadL1 * wiLoadL1;
        int64_t aL1GmOffset = 0;

        if (this->cin1LoadL1PadHead > 0) {
            this->Al1PadHeadSet2d(hiLoadL1, wiLoadL1);
        }

        if constexpr (Intf::formatOutput == ConvFormat::NCDHW) {
            aL1GmOffset = self_->ctx.convTiling->orgHixWi * self_->ctx.convTiling->dilationD;
            LoadAl1Data(realHiTopGmIdx, realWiTopGmIdx, al1Offset, aL1GmOffset);
        } else {
            aL1GmOffset = self_->ctx.convTiling->orgHixWi * self_->ctx.convTiling->dilationD * self_->ctx.orgCi;
            LoadAl1DataHWC(realHiTopGmIdx, realWiTopGmIdx, al1Offset, aL1GmOffset);
        }

        if (this->cin1LoadL1PadTail > 0) {
            this->Al1PadTailSet2d(hiLoadL1, wiLoadL1);
        }
    }

    __aicore__ inline void LoadAl1Data(int64_t realHiTopGmIdx, int64_t realWiTopGmIdx,
                                       int64_t al1Offset, int64_t aL1GmOffset)
    {
        this->aL1Pos = 0;
        uint64_t aL1GmPos = self_->ctx.batchIter * self_->ctx.fmapOneBatchSize +
                            this->cinIdx * self_->ctx.orgDi * self_->ctx.convTiling->orgHixWi +
                            this->diIdx * self_->ctx.convTiling->orgHixWi +
                            realHiTopGmIdx * self_->ctx.orgWi +
                            realWiTopGmIdx;
        Dn2NzParams intriParams;
        SetDn2NzIntriParams(intriParams);
        for (uint64_t dkIter = 0; dkIter < this->dkLoadL1; ++dkIter) {
            DataCopy<typename Intf::FmapT>(self_->ctx.al1[this->aL1Pos], self_->ctx.agm[aL1GmPos], intriParams);
            this->aL1Pos += al1Offset;
            aL1GmPos += aL1GmOffset;
        }
    }

    __aicore__ inline void LoadAl1DataHWC(int64_t realHiTopGmIdx, int64_t realWiTopGmIdx,
                                          int64_t al1Offset, int64_t aL1GmOffset)
    {
        this->aL1Pos = 0;
        uint64_t aL1GmPos = self_->ctx.batchIter * self_->ctx.fmapOneBatchSize +
                            this->diIdx * self_->ctx.convTiling->orgHixWi * self_->ctx.orgCi +
                            realHiTopGmIdx * self_->ctx.orgWi * self_->ctx.orgCi +
                            realWiTopGmIdx * self_->ctx.orgCi +
                            this->cinIdx;
        Nd2NzParams intriParams;
        SetNd2NzIntriParams(intriParams);
        for (uint64_t dkIter = 0; dkIter < this->dkLoadL1; ++dkIter) {
            DataCopy<typename Intf::FmapT>(self_->ctx.al1[this->aL1Pos], self_->ctx.agm[aL1GmPos], intriParams);
            this->aL1Pos += al1Offset;
            aL1GmPos += aL1GmOffset;
        }
    }

    __aicore__ inline void SetDn2NzIntriParams(Dn2NzParams &intriParams)
    {
        if (likely(wiLoadL1 == self_->ctx.orgWi)) {
            intriParams.dnNum = 1;
            intriParams.nValue = hiLoadL1 * wiLoadL1;
            intriParams.srcDnMatrixStride = 0;
            intriParams.dstNzMatrixStride = 0;
        } else {
            intriParams.dnNum = hiLoadL1;
            intriParams.nValue = wiLoadL1;
            if constexpr (Intf::formatOutput == ConvFormat::NCDHW) {
                intriParams.srcDnMatrixStride = self_->ctx.orgWi;
            } else {
                intriParams.srcDnMatrixStride = self_->ctx.orgWi * self_->ctx.orgCi;
            }
            intriParams.dstNzMatrixStride = wiLoadL1 * Intf::k0;
        }

        if constexpr (Intf::formatOutput == ConvFormat::NCDHW) {
            intriParams.srcDValue = self_->ctx.convTiling->orgHixWi * self_->ctx.convTiling->orgDi;
        } else {
            intriParams.srcDValue = self_->ctx.orgCi;
        }
        intriParams.dValue = this->cinLoadL1;
        intriParams.dstNzC0Stride = hiLoadL1 * wiLoadL1;
        intriParams.dstNzNStride = 1;
    }

    __aicore__ inline void SetNd2NzIntriParams(Nd2NzParams& intriParams)
    {
        if (likely(wiLoadL1 == self_->ctx.orgWi)) {
            intriParams.ndNum = 1;
            intriParams.nValue = hiLoadL1 * wiLoadL1;
            intriParams.srcNdMatrixStride = 0;
            intriParams.dstNzMatrixStride = 0;
        } else {
            intriParams.ndNum = hiLoadL1;
            intriParams.nValue = wiLoadL1;
            intriParams.srcNdMatrixStride = self_->ctx.orgWi * self_->ctx.orgCi;
            intriParams.dstNzMatrixStride = wiLoadL1 * Intf::k0;
        }
        intriParams.srcDValue = self_->ctx.orgCi;
        intriParams.dValue = this->cinLoadL1;
        intriParams.dstNzC0Stride = hiLoadL1 * wiLoadL1;
        intriParams.dstNzNStride = 1;
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

};

#endif  // CONV3D_V2_INSTR_IMPL_H