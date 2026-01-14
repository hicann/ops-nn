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
 * \file conv3d_v2_instr_base_impl.h
 * \brief
 */

#ifndef CONV3D_V2_INSTR_BASE_IMPL_H
#define CONV3D_V2_INSTR_BASE_IMPL_H

#include "conv3d_v2_config.h"
#include "conv3d_v2_util.h"

namespace Conv3dFunc {
using namespace AscendC;
using namespace conv;

template <class Intf>
class LoadAL1ToolsBase {
public:
    __aicore__ inline LoadAL1ToolsBase() {}

    __aicore__ inline void SetIntf(Intf *self)
    {
        self_ = self;
        buffAddr.logicPos = static_cast<uint8_t>(QuePosition::A1);
    }

    __aicore__ inline void Al1PadHeadSet2d(uint64_t hiLoadL1, uint64_t wiLoadL1) {
        uint32_t padHeadSize = cin1LoadL1PadHead * hiLoadL1 * wiLoadL1;
        if constexpr (Intf::isQuantScene || Intf::isDeQuantFlag) {
            uint16_t paddingValue = (static_cast<uint16_t>(self_->ctx.convTiling->offsetx)) << BIT_OFFSET_8 |
                                    (static_cast<uint16_t>(self_->ctx.convTiling->offsetx));
            uint16_t b16PadHeadblockNum = padHeadSize / Intf::k0;
            InitConstValueParams<uint16_t> initConstValueParams(1, b16PadHeadblockNum, 0, paddingValue);
            buffAddr.dataLen = b16PadHeadblockNum * C0_SIZE;
            buffAddr.bufferAddr = self_->ctx.al1.GetPhyAddr();
            al1tmp.SetAddr(buffAddr);
            InitConstValue<uint16_t>(al1tmp, initConstValueParams);
            aL1Pos += padHeadSize;
        } else {
            InitConstValueParams<typename Intf::FmapT> initConstValueParams;
            initConstValueParams.repeatTimes = 1;
            initConstValueParams.blockNum = padHeadSize / Intf::k0;
            InitConstValue<typename Intf::FmapT>(self_->ctx.al1, initConstValueParams);
            aL1Pos += padHeadSize;
        }
    }

    __aicore__ inline void Al1PadTailSet2d(uint64_t hiLoadL1, uint64_t wiLoadL1) {
        uint32_t padTailSize = cin1LoadL1PadTail * hiLoadL1 * wiLoadL1;
        if constexpr (Intf::isQuantScene || Intf::isDeQuantFlag) {
            uint16_t paddingValue = (static_cast<uint16_t>(self_->ctx.convTiling->offsetx)) << BIT_OFFSET_8 |
                                    (static_cast<uint16_t>(self_->ctx.convTiling->offsetx));
            uint16_t b16PadTailblockNum = padTailSize / Intf::k0;
            InitConstValueParams<uint16_t> initConstValueParams(1, b16PadTailblockNum, 0, paddingValue);
            buffAddr.dataLen = b16PadTailblockNum * C0_SIZE;
            buffAddr.bufferAddr = self_->ctx.al1[aL1Pos].GetPhyAddr();
            al1tmp.SetAddr(buffAddr);
            InitConstValue<uint16_t>(al1tmp, initConstValueParams);
        } else {
            InitConstValueParams<typename Intf::FmapT> initConstValueParams;
            initConstValueParams.repeatTimes = 1;
            initConstValueParams.blockNum = padTailSize / Intf::k0;
            InitConstValue<typename Intf::FmapT>(self_->ctx.al1[aL1Pos], initConstValueParams);
        }
    }

    __aicore__ inline void SetPadData()
    {
        if constexpr (Intf::isQuantScene || Intf::isDeQuantFlag) {
            uint16_t padValue = (static_cast<uint16_t>(self_->ctx.convTiling->offsetx)) << BIT_OFFSET_8 |
                                (static_cast<uint16_t>(self_->ctx.convTiling->offsetx));
            InitConstValueParams<uint16_t> params(1, static_cast<uint16_t>(
                self_->ctx.convTiling->aL1SpaceSize / C0_SIZE), 0, padValue);
            buffAddr.dataLen = self_->ctx.convTiling->aL1SpaceSize;
            buffAddr.bufferAddr = self_->ctx.al1.GetPhyAddr();
            al1tmp.SetAddr(buffAddr);
            InitConstValue<uint16_t>(al1tmp, params);
        } else {
            InitConstValueParams<typename Intf::FmapT> params(
                1, static_cast<uint16_t>(self_->ctx.convTiling->aL1SpaceSize / C0_SIZE), 0, 0);
            InitConstValue<typename Intf::FmapT>(self_->ctx.al1, params);
        }
        allPadFlag = false;
    }

    __aicore__ inline void CalcCiL1Pad()
    {
        dkLoadL1 = IsKAL1Tail() ? self_->ctx.aL1DkTail : self_->ctx.aL1Dk;
        cinLoadL1 = IsKAL1Tail() ? self_->ctx.aL1CinTail : self_->ctx.aL1Cin;

        uint64_t kdL1Idx = 0;
        if (self_->ctx.aL1CinLoadNum == 1) {
            cinLoadL1 = self_->ctx.singleCoreCi;
            cinIdx = 0;
            kdL1Idx = self_->ctx.kAL1Iter * self_->ctx.aL1Dk;
        } else {
            cinIdx = (self_->ctx.kAL1Iter % self_->ctx.aL1CinLoadNum) * self_->ctx.aL1Cin;
            kdL1Idx = (self_->ctx.kAL1Iter / self_->ctx.aL1CinLoadNum) * self_->ctx.aL1Dk;
        }

        uint64_t diStartWithPad = self_->ctx.diStartPos + self_->ctx.dOutIter * self_->ctx.convTiling->strideD +
                                  kdL1Idx * self_->ctx.convTiling->dilationD;
        diIdx = self_->ctx.diStartPos <= self_->ctx.convTiling->padHead ?
            diStartWithPad - self_->ctx.convTiling->padHead : diStartWithPad - self_->ctx.diStartPos;

        uint64_t diEndWithPad = diStartWithPad + (dkLoadL1 - 1) * self_->ctx.convTiling->dilationD + 1;

        cin1LoadL1PadHead = 0;
        cin1LoadL1PadTail = 0;

        if (diStartWithPad < self_->ctx.convTiling->padHead) {
            uint64_t kdTmp = CeilDiv((self_->ctx.convTiling->padHead - diStartWithPad),
                self_->ctx.convTiling->dilationD);
            kdTmp = kdTmp > dkLoadL1 ? dkLoadL1 : kdTmp;
            diIdx = self_->ctx.convTiling->dilationD == 1 ? 0 :
                kdTmp * self_->ctx.convTiling->dilationD - self_->ctx.convTiling->padHead + diStartWithPad;
            cin1LoadL1PadHead = kdTmp * AlignB(cinLoadL1, Intf::k0);
            dkLoadL1 -= kdTmp;
        }

        if (diEndWithPad > self_->ctx.orgDi + self_->ctx.convTiling->padHead) {
            uint64_t kdTmp = CeilDiv(diEndWithPad - (self_->ctx.orgDi + self_->ctx.convTiling->padHead),
                                     self_->ctx.convTiling->dilationD);
            kdTmp = kdTmp > dkLoadL1 ? dkLoadL1 : kdTmp;
            cin1LoadL1PadTail = kdTmp * AlignB(cinLoadL1, Intf::k0);
            dkLoadL1 -= kdTmp;
        }
    }

    __aicore__ inline bool IsKAL1Tail()
    {
        if (self_->ctx.aL1CinLoadNum == 1) {
            return self_->ctx.kAL1Iter == self_->ctx.maxKAL1Iter;
        } else {
            return (self_->ctx.kAL1Iter + 1) % self_->ctx.aL1CinLoadNum == 0;
        }
    }

    __aicore__ inline void SetLoad3dFMatrix(uint64_t padLeftL1, uint64_t padRightL1, uint64_t padTopL1,
                                            uint64_t hiLoadL1, uint64_t wiLoadL1)
    {
        uint8_t padList[PAD_SIZE] = {MAX_PAD_R, MAX_PAD_R, MAX_PAD_R, MAX_PAD_R};
        if (unlikely(allPadFlag)) {
            Load3DSetFMatrixCal(MIN_HI_WI, MIN_HI_WI, padList);
        } else {
            padList[PAD_IDX_L] = padLeftL1;
            padList[PAD_IDX_R] = padRightL1;
            padList[PAD_IDX_T] = padTopL1;
            Load3DSetFMatrixCal(hiLoadL1, wiLoadL1, padList);
        }
        Load3DSetPaddingCal(self_->ctx.convTiling->offsetx);
    }

public:
    Intf *self_ = nullptr;
    uint64_t dkLoadL1 = 0;
    uint64_t cinLoadL1 = 0;
    uint64_t diIdx = 0;
    uint64_t cinIdx = 0;
    uint64_t cin1LoadL1PadHead = 0;
    uint64_t cin1LoadL1PadTail = 0;
    uint64_t aL1Pos = 0;
    bool allPadFlag = false;
    LocalTensor<uint16_t> al1tmp;
    TBuffAddr buffAddr;
};

template <class Intf>
class LoadBL1Tools {
public:
    __aicore__ inline LoadBL1Tools()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
    }

    __aicore__ inline void LoadBL1()
    {
        uint64_t currentBL1Dk = IsKBL1Tail() ? self_->ctx.bL1DkTail : self_->ctx.bL1Dk;

        uint64_t dkIdx = 0;
        uint64_t cinIdx = 0;
        uint64_t coutIdx = self_->ctx.nBL1Iter * self_->ctx.convTiling->nBL1;

        currentBL1Cin1 = IsKBL1Tail() ? self_->ctx.bL1CinTail : self_->ctx.bL1Cin;
        if (self_->ctx.bL1CinLoadNum == 1) {
            currentBL1Cin1 = self_->ctx.singleCoreCi;
            dkIdx = self_->ctx.kBL1Iter * self_->ctx.bL1Dk;
        } else {
            cinIdx = (self_->ctx.kBL1Iter % self_->ctx.bL1CinLoadNum) * self_->ctx.bL1Cin;
            dkIdx = (self_->ctx.kBL1Iter / self_->ctx.bL1CinLoadNum) * self_->ctx.bL1Dk;
        }

        uint64_t currentNBL1 = IsNBL1Tail() ? self_->ctx.nBL1Tail : self_->ctx.convTiling->nBL1;

        Dn2NzParams intriParams;
        SetDn2NzIntriParams(intriParams, currentNBL1);

        uint64_t bL1GmPos = 0;
        uint64_t bL1GmOffset = 0;
        if constexpr (Intf::formatWeight == ConvFormat::NCDHW) {
            bL1GmPos = dkIdx * self_->ctx.convTiling->kernelHxkernelW + cinIdx *
                       self_->ctx.convTiling->kernelHxkernelWxkernelD + coutIdx * self_->ctx.singleCoreCi *
                       self_->ctx.convTiling->kernelHxkernelWxkernelD;
            bL1GmOffset = self_->ctx.convTiling->kernelHxkernelW;
        } else {
            bL1GmPos = dkIdx * self_->ctx.convTiling->kernelHxkernelW * self_->ctx.singleCoreCi *
                       self_->ctx.orgCo + cinIdx * self_->ctx.orgCo + coutIdx;
            bL1GmOffset = self_->ctx.convTiling->kernelHxkernelW * self_->ctx.singleCoreCi *
                       self_->ctx.orgCo;
        }
        uint64_t bL1Pos = 0;
        uint64_t bL1Offset = AlignB(currentBL1Cin1, Intf::k0) * self_->ctx.convTiling->kernelHxkernelW *
                             self_->ctx.convTiling->nBL1;

        for (uint64_t dkIter = 0; dkIter < currentBL1Dk; ++dkIter) {
            DataCopy<typename Intf::WeightT>(self_->ctx.bl1[bL1Pos], self_->ctx.bgm[bL1GmPos], intriParams);
            bL1Pos += bL1Offset;
            bL1GmPos += bL1GmOffset;
        }
    }

private:
    __aicore__ inline void SetDn2NzIntriParams(Dn2NzParams &intriParams, const uint64_t currentNBL1)
    {
        intriParams.dValue = currentBL1Cin1;
        intriParams.dstNzC0Stride = self_->ctx.convTiling->nBL1 * self_->ctx.convTiling->kernelHxkernelW;
        if constexpr (Intf::formatWeight == ConvFormat::NCDHW) {
            intriParams.srcDValue = self_->ctx.convTiling->kernelHxkernelWxkernelD;
            intriParams.dnNum = currentNBL1;
            intriParams.nValue = self_->ctx.convTiling->kernelHxkernelW;
            intriParams.srcDnMatrixStride = self_->ctx.singleCoreCi * self_->ctx.convTiling->kernelHxkernelWxkernelD;
            intriParams.dstNzNStride = self_->ctx.convTiling->nBL1;
            intriParams.dstNzMatrixStride = C0_SIZE / Intf::sizeOfWeight;
        } else {
            intriParams.dnNum = self_->ctx.convTiling->kernelHxkernelW;
            intriParams.nValue = currentNBL1;
            intriParams.srcDnMatrixStride = self_->ctx.singleCoreCi * self_->ctx.orgCo;
            intriParams.srcDValue = self_->ctx.orgCo;
            intriParams.dstNzNStride = 1;
            intriParams.dstNzMatrixStride = self_->ctx.convTiling->nBL1 * C0_SIZE / Intf::sizeOfWeight;
        }
    }

    __aicore__ inline bool IsNBL1Tail()
    {
        return self_->ctx.nBL1Iter == self_->ctx.maxNBL1Iter;
    }

    __aicore__ inline bool IsKBL1Tail()
    {
        if (self_->ctx.bL1CinLoadNum == 1) {
            return self_->ctx.kBL1Iter == self_->ctx.maxKBL1Iter;
        } else {
            return (self_->ctx.kBL1Iter + 1) % self_->ctx.bL1CinLoadNum == 0;
        }
    }

private:
    Intf *self_ = nullptr;
    uint64_t currentBL1Cin1 = 0;
};

};

#endif  // CONV3D_V2_INSTR_BASE_IMPL_H