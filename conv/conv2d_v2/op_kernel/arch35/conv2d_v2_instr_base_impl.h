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
 * \file conv2d_v2_instr_base_impl.h
 * \brief
 */

#ifndef CONV2D_V2_INSTR_BASE_IMPL_H
#define CONV2D_V2_INSTR_BASE_IMPL_H

#include "conv2d_v2_config.h"
#include "conv2d_v2_util.h"

namespace Conv2dFunc {
using namespace AscendC;
using namespace conv;

template <class Intf>
class LoadAL1ToolsBase {
public:
    __aicore__ inline LoadAL1ToolsBase() {}

    __aicore__ inline void SetIntf(Intf *self)
    {
        self_ = self;
        buffAddr.dataLen = self_->ctx.convTiling->aL1SpaceSize;
        buffAddr.logicPos = static_cast<uint8_t>(QuePosition::A1);
    }

    __aicore__ inline void SetPadData()
    {
        if constexpr (Intf::isQuantScene) {
            uint16_t padValue = (static_cast<uint16_t>(self_->ctx.convTiling->offsetx)) << BIT_OFFSET_8 |
                                (static_cast<uint16_t>(self_->ctx.convTiling->offsetx));
            InitConstValueParams<uint16_t> params(
                1, static_cast<uint16_t>(self_->ctx.convTiling->aL1SpaceSize / C0_SIZE), 0, padValue);
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
    bool allPadFlag = false;
    LocalTensor<uint16_t> al1tmp;
    TBuffAddr buffAddr;
};

template <class Intf>
class LoadAL1ToolsInnerBatch {
public:
    __aicore__ inline LoadAL1ToolsInnerBatch()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;

        hiLoadL1 = (self_->ctx.convTiling->orgHo - 1) * self_->ctx.convTiling->strideH + self_->ctx.dilatedKernelH;
        hiLoadL1 = hiLoadL1 > self_->ctx.convTiling->orgHi ? self_->ctx.convTiling->orgHi : hiLoadL1;
        realHixWi = hiLoadL1 * self_->ctx.convTiling->orgWi;

        padList[PAD_IDX_L] = static_cast<uint8_t>(self_->ctx.convTiling->padLeft);
        padList[PAD_IDX_R] = static_cast<uint8_t>(self_->ctx.convTiling->padRight);
        padList[PAD_IDX_T] = static_cast<uint8_t>(self_->ctx.convTiling->padTop);
        Load3DSetPaddingCal(self_->ctx.convTiling->offsetx);
        if constexpr (Intf::ConvParam::innerBatch == static_cast<int8_t>(ConvInnerBatch::MULTI_BATCH)) {
            Load3DSetFMatrixCal(hiLoadL1, self_->ctx.convTiling->orgWi, padList);
        }
    }

    __aicore__ inline void LoadAL1()
    {
        LoadAL1(self_->ctx.kAL1Iter, self_->ctx.mAL1Iter, self_->ctx.batchIter);
    }

    __aicore__ inline void LoadAL1(uint64_t kAL1Iter, uint64_t mAL1Iter, uint64_t batchIter)
    {
        if constexpr (Intf::ConvParam::innerBatch == static_cast<int8_t>(ConvInnerBatch::KERNEL_1X1_MULTI_BATCH)) {
            if constexpr (Intf::formatOutput == ConvFormat::NCHW && Intf::c04Flag) {
                Load3DSetFMatrixCal(self_->ctx.innerBatch, AlignB(C04_CIN_SIZE * self_->ctx.convTiling->orgHixWi, Intf::k0) / C04_CIN_SIZE, padList);
            } else {
                Load3DSetFMatrixCal(self_->ctx.innerBatch * hiLoadL1, self_->ctx.convTiling->orgWi, padList);
            }
        }

        if constexpr (Intf::disContinuousFlag) {
            LoadAL1InputHWNC(kAL1Iter, mAL1Iter, batchIter);
        } else if constexpr (Intf::formatFmap == ConvFormat::NCHW) {
            uint64_t aL1GmOffset = batchIter * self_->ctx.convTiling->innerBatch * self_->ctx.fmapOneBatchSize +
                                   kAL1Iter * self_->ctx.convTiling->cinOffsetBlockInGM;

            Dn2NzParams intriParams;
            if constexpr (Intf::c04Flag) {
                SetDn2NzIntriParamsC04(intriParams);
                DataCopy<typename Intf::FmapT, true>(self_->ctx.al1, self_->ctx.agm[aL1GmOffset], intriParams);
            } else {
                SetDn2NzIntriParams(intriParams, kAL1Iter);
                DataCopy<typename Intf::FmapT>(self_->ctx.al1, self_->ctx.agm[aL1GmOffset], intriParams);
            }
        } else if constexpr (Intf::formatOutput == ConvFormat::NHWC) {
            uint64_t aL1GmOffset = batchIter * self_->ctx.convTiling->innerBatch * self_->ctx.fmapOneBatchSize +
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

    __aicore__ inline void LoadAL1InputHWNC(uint64_t kAL1Iter, uint64_t mAL1Iter, uint64_t batchIter)
    {
        uint64_t aL1GmOffset = batchIter * self_->ctx.convTiling->innerBatch * self_->ctx.orgCi +
                               kAL1Iter * self_->ctx.convTiling->cinAInCore;

        Nd2NzParams intriParams;
        SetNd2NzIntriParamsInputHWNC(intriParams, kAL1Iter);
        DataCopy<typename Intf::FmapT>(self_->ctx.al1, self_->ctx.agm[aL1GmOffset], intriParams);
    }

private:
    __aicore__ inline void SetDn2NzIntriParamsC04(Dn2NzParams &intriParams)
    {
        intriParams.dnNum = self_->ctx.innerBatch;
        intriParams.nValue = self_->ctx.convTiling->orgHixWi;
        intriParams.dValue = self_->ctx.convTiling->orgCi;
        intriParams.srcDValue = self_->ctx.convTiling->orgHixWi;
        intriParams.srcDnMatrixStride = self_->ctx.convTiling->orgCi * self_->ctx.convTiling->orgHixWi;
        intriParams.dstNzNStride = 1;
        intriParams.dstNzMatrixStride = AlignB(C04_CIN_SIZE * self_->ctx.convTiling->orgHixWi, Intf::k0);
    }
 
    __aicore__ inline void SetDn2NzIntriParams(Dn2NzParams &intriParams, uint64_t kAL1Iter)
    {
        uint32_t al1Ci = IsKAL1Tail(kAL1Iter) ?
            self_->ctx.convTiling->cinATailInCore : self_->ctx.convTiling->cinAInCore;
        intriParams.dnNum = self_->ctx.innerBatch;
        intriParams.nValue = realHixWi;
        intriParams.dValue = al1Ci;
        intriParams.srcDValue = self_->ctx.convTiling->orgHixWi;
        intriParams.srcDnMatrixStride = self_->ctx.convTiling->orgCi * self_->ctx.convTiling->orgHixWi;
        intriParams.dstNzNStride = 1;

        if constexpr (Intf::ConvParam::innerBatch == static_cast<int8_t>(ConvInnerBatch::KERNEL_1X1_MULTI_BATCH)) {
            intriParams.dstNzC0Stride = self_->ctx.innerBatch * realHixWi;
            intriParams.dstNzMatrixStride = realHixWi * Intf::k0;
        } else {
            intriParams.dstNzC0Stride = realHixWi;
            intriParams.dstNzMatrixStride = AlignB(al1Ci, Intf::k0) * realHixWi;
        }
    }

    __aicore__ inline void SetNd2NzIntriParamsC04(Nd2NzParams &intriParams)
    {
        if constexpr (Intf::ConvParam::innerBatch == static_cast<int8_t>(ConvInnerBatch::KERNEL_1X1_MULTI_BATCH)) {
            intriParams.ndNum = 1;
            intriParams.nValue = self_->ctx.innerBatch * self_->ctx.convTiling->orgHixWi;
        } else {
            intriParams.ndNum = self_->ctx.innerBatch;
            intriParams.nValue = self_->ctx.convTiling->orgHixWi;
            intriParams.srcNdMatrixStride = self_->ctx.convTiling->orgCi * self_->ctx.convTiling->orgHixWi;
            intriParams.dstNzNStride = 1;
            intriParams.dstNzMatrixStride = AlignB(C04_CIN_SIZE * self_->ctx.convTiling->orgHixWi, Intf::k0);
        }  
        intriParams.dValue = self_->ctx.convTiling->orgCi;
        intriParams.srcDValue = self_->ctx.convTiling->orgCi;
    }

    __aicore__ inline void SetNd2NzIntriParams(Nd2NzParams &intriParams, uint64_t kAL1Iter)
    {
        uint32_t al1Ci = IsKAL1Tail(kAL1Iter) ?
            self_->ctx.convTiling->cinATailInCore : self_->ctx.convTiling->cinAInCore;
        intriParams.ndNum = self_->ctx.innerBatch;
        intriParams.nValue = self_->ctx.convTiling->orgHixWi;
        intriParams.dValue = al1Ci;
        intriParams.srcDValue = self_->ctx.orgCi;
        intriParams.srcNdMatrixStride = self_->ctx.convTiling->orgCi * self_->ctx.convTiling->orgHixWi;
        intriParams.dstNzNStride = 1;

        if constexpr (Intf::ConvParam::innerBatch == static_cast<int8_t>(ConvInnerBatch::KERNEL_1X1_MULTI_BATCH)) {
            intriParams.dstNzC0Stride = self_->ctx.innerBatch * realHixWi;
            intriParams.dstNzMatrixStride = realHixWi * Intf::k0;
        } else {
            intriParams.dstNzC0Stride = realHixWi;
            intriParams.dstNzMatrixStride = AlignB(al1Ci, Intf::k0) * realHixWi;
        }
    }

    __aicore__ inline void SetNd2NzIntriParamsInputHWNC(Nd2NzParams &intriParams, uint64_t kAL1Iter)
    {
        uint32_t al1Ci = IsKAL1Tail(kAL1Iter) ?
            self_->ctx.convTiling->cinATailInCore : self_->ctx.convTiling->cinAInCore;
        intriParams.ndNum = self_->ctx.innerBatch;
        intriParams.nValue = self_->ctx.convTiling->orgHixWi;
        intriParams.dValue = al1Ci;
        intriParams.srcNdMatrixStride = self_->ctx.convTiling->orgCi;
        intriParams.srcDValue = self_->ctx.orgCi * self_->ctx.orgBatch;
        intriParams.dstNzNStride = 1;

        if constexpr (Intf::ConvParam::innerBatch == static_cast<int8_t>(ConvInnerBatch::KERNEL_1X1_MULTI_BATCH)) {
            intriParams.dstNzC0Stride = self_->ctx.innerBatch * realHixWi;
            intriParams.dstNzMatrixStride = realHixWi * Intf::k0;
        } else {
            intriParams.dstNzC0Stride = realHixWi;
            intriParams.dstNzMatrixStride = AlignB(al1Ci, Intf::k0) * realHixWi;
        }
    }

    __aicore__ inline bool IsKAL1Tail(uint64_t kAL1Iter)
    {
        return kAL1Iter == self_->ctx.maxKAL1Iter;
    }

private:
    Intf *self_ = nullptr;
    uint8_t padList[PAD_SIZE] = {MAX_PAD_R, MAX_PAD_R, MAX_PAD_R, MAX_PAD_R};
    uint64_t hiLoadL1 = 0;
    uint64_t realHixWi = 0;
};

template <class Intf>
class LoadBL1Tools {
public:
    __aicore__ inline LoadBL1Tools() {}
    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;

        isOneByOneKernelScenario = self_->ctx.convTiling->kernelHxkernelW  == 1 &&
                                   !(AscendC::IsSameType<typename Intf::WeightT, hifloat8_t>::value) &&
                                   !(AscendC::IsSameType<typename Intf::WeightT, fp8_e4m3fn_t>::value);
    }

    __aicore__ inline void LoadBL1()
    {
        LoadBL1(self_->ctx.kBL1Iter, self_->ctx.nBL1Iter);
    }

    __aicore__ inline void LoadBL1(uint64_t kBL1Iter, uint64_t nBL1Iter)
    {
        if constexpr (Intf::formatWeight == ConvFormat::NCHW) {
            if constexpr (Intf::isDmaFlag) {
                LoadBL1DataKernelSplit(kBL1Iter, nBL1Iter);
            } else {
                LoadBL1Data(kBL1Iter, nBL1Iter);
            }
        } else {
            if constexpr (Intf::isDmaFlag) {
                LoadBL1DataKernelSplitHWC(kBL1Iter, nBL1Iter);
            } else {
                LoadBL1DataHWC(kBL1Iter, nBL1Iter);
            }
        }
    }

private:
    __aicore__ inline void LoadBL1Data(uint64_t kBL1Iter, uint64_t nBL1Iter)
    {
        uint64_t bL1GmOffset;
        if constexpr (!Intf::hasNL1IterFlag) {
            bL1GmOffset = kBL1Iter * self_->ctx.convTiling->kBL1;
        } else {
            bL1GmOffset = nBL1Iter * self_->ctx.convTiling->nBL1 * self_->ctx.convTiling->coutOffsetBlock +
                kBL1Iter * self_->ctx.convTiling->kBL1;
            self_->ctx.currentNBL1 = nBL1Iter == self_->ctx.maxNBL1Iter ?
                self_->ctx.nBL1Tail : self_->ctx.convTiling->nBL1;
        }

        if (isOneByOneKernelScenario) {
            Nd2NzParams intriParams;
            intriParams.ndNum = 1;
            intriParams.nValue = self_->ctx.currentNBL1;
            intriParams.dValue = kBL1Iter == self_->ctx.maxKBL1Iter ? self_->ctx.kBL1Tail : self_->ctx.convTiling->kBL1;
            intriParams.srcNdMatrixStride = 0;
            intriParams.srcDValue = self_->ctx.orgCi / self_->ctx.convTiling->groups;
            intriParams.dstNzC0Stride = self_->ctx.convTiling->nBL1;
            intriParams.dstNzNStride = 1;
            intriParams.dstNzMatrixStride = 0;
            DataCopy<typename Intf::WeightT>(self_->ctx.bl1, self_->ctx.bgm[bL1GmOffset], intriParams);
            return;
        }
        Dn2NzParams intriParams;
        intriParams.dnNum = self_->ctx.currentNBL1;
        intriParams.nValue = self_->ctx.convTiling->kernelHxkernelW;
        intriParams.dValue = kBL1Iter == self_->ctx.maxKBL1Iter ?
            self_->ctx.convTiling->cinBTailInCore : self_->ctx.convTiling->cinBInCore;
        intriParams.srcDnMatrixStride =  self_->ctx.convTiling->coutOffsetBlock;
        intriParams.srcDValue = self_->ctx.convTiling->kernelHxkernelW;
        intriParams.dstNzC0Stride = self_->ctx.convTiling->kernelHxkernelW * self_->ctx.convTiling->nBL1;
        intriParams.dstNzNStride = self_->ctx.convTiling->nBL1;
        intriParams.dstNzMatrixStride = Intf::k0;
        DataCopy<typename Intf::WeightT>(self_->ctx.bl1, self_->ctx.bgm[bL1GmOffset], intriParams);
    }

    __aicore__ inline void LoadBL1DataHWC(uint64_t kBL1Iter, uint64_t nBL1Iter)
    {
        uint64_t bL1GmOffset = kBL1Iter * self_->ctx.convTiling->cinBInCore * self_->ctx.orgCo;
        if constexpr (Intf::hasNL1IterFlag) {
            bL1GmOffset += nBL1Iter * self_->ctx.convTiling->nBL1;
            self_->ctx.currentNBL1 = nBL1Iter == self_->ctx.maxNBL1Iter ?
                self_->ctx.nBL1Tail : self_->ctx.convTiling->nBL1;
        }

        Dn2NzParams intriParams;
        intriParams.dnNum = self_->ctx.convTiling->kernelHxkernelW;
        intriParams.nValue = self_->ctx.currentNBL1;
        intriParams.dValue = kBL1Iter == self_->ctx.maxKBL1Iter ?
            self_->ctx.convTiling->cinBTailInCore : self_->ctx.convTiling->cinBInCore;
        intriParams.srcDnMatrixStride =  self_->ctx.orgCi / self_->ctx.convTiling->groups * self_->ctx.orgCo;
        intriParams.srcDValue = self_->ctx.orgCo;
        intriParams.dstNzC0Stride = self_->ctx.convTiling->kernelHxkernelW * self_->ctx.convTiling->nBL1;
        intriParams.dstNzNStride = 1;
        intriParams.dstNzMatrixStride = self_->ctx.convTiling->nBL1 * Intf::k0;
        DataCopy<typename Intf::WeightT>(self_->ctx.bl1, self_->ctx.bgm[bL1GmOffset], intriParams);
    }

    __aicore__ inline void LoadBL1DataKernelSplit(uint64_t kBL1Iter, uint64_t nBL1Iter)
    {
        uint64_t bL1GmOffset;
        if constexpr (!Intf::hasNL1IterFlag) {
            bL1GmOffset = self_->ctx.cinBL1Iter * self_->ctx.cinBL1 * self_->ctx.convTiling->kernelHxkernelW +
                          self_->ctx.khBL1Iter * self_->ctx.convTiling->khL1 * self_->ctx.convTiling->kernelW +
                          self_->ctx.kwBL1Iter * self_->ctx.convTiling->kwL1;
        } else {
            bL1GmOffset = nBL1Iter * self_->ctx.convTiling->nBL1 * self_->ctx.convTiling->coutOffsetBlock +
                          self_->ctx.cinBL1Iter * self_->ctx.cinBL1 * self_->ctx.convTiling->kernelHxkernelW +
                          self_->ctx.khBL1Iter * self_->ctx.convTiling->khL1 * self_->ctx.convTiling->kernelW +
                          self_->ctx.kwBL1Iter * self_->ctx.convTiling->kwL1;
            self_->ctx.currentNBL1 = nBL1Iter == self_->ctx.maxNBL1Iter ?
                self_->ctx.nBL1Tail : self_->ctx.convTiling->nBL1;
        }
        Dn2NzParams intriParams;
        intriParams.dnNum = self_->ctx.currentNBL1;
        intriParams.nValue = self_->ctx.convTiling->kwL1;
        intriParams.dValue = self_->ctx.cinBL1Iter == self_->ctx.maxCinBL1Iter ?
            self_->ctx.convTiling->cinBTailInCore : self_->ctx.convTiling->cinBInCore;
        intriParams.srcDnMatrixStride =  self_->ctx.convTiling->coutOffsetBlock;
        intriParams.srcDValue = self_->ctx.convTiling->kernelHxkernelW;
        intriParams.dstNzC0Stride = self_->ctx.convTiling->kwL1 * self_->ctx.convTiling->khL1 *
                                    self_->ctx.convTiling->nBL1;
        intriParams.dstNzNStride = self_->ctx.convTiling->nBL1;
        intriParams.dstNzMatrixStride = Intf::k0;
        uint64_t bl1DstOffset = 0;
        for (uint16_t khIterIdx = 0; khIterIdx < self_->ctx.convTiling->khL1; khIterIdx++) {
            bl1DstOffset = khIterIdx * self_->ctx.convTiling->kwL1 * self_->ctx.convTiling->nBL1 * Intf::k0;
            DataCopy<typename Intf::WeightT>(self_->ctx.bl1[bl1DstOffset], self_->ctx.bgm[bL1GmOffset], intriParams);
            bL1GmOffset += self_->ctx.convTiling->kernelW;
        }
    }

    __aicore__ inline void LoadBL1DataKernelSplitHWC(uint64_t kBL1Iter, uint64_t nBL1Iter)
    {
        uint64_t bL1GmOffset = self_->ctx.cinBL1Iter * self_->ctx.cinBL1 * self_->ctx.orgCo +
                               self_->ctx.khBL1Iter * self_->ctx.convTiling->khL1 * self_->ctx.convTiling->kernelW *
                               self_->ctx.orgCi / self_->ctx.convTiling->groups * self_->ctx.orgCo +
                               self_->ctx.kwBL1Iter * self_->ctx.convTiling->kwL1 * self_->ctx.orgCi /
                               self_->ctx.convTiling->groups * self_->ctx.orgCo;
        if constexpr (Intf::hasNL1IterFlag) {
            bL1GmOffset += nBL1Iter * self_->ctx.convTiling->nBL1;
            self_->ctx.currentNBL1 = nBL1Iter == self_->ctx.maxNBL1Iter ?
                self_->ctx.nBL1Tail : self_->ctx.convTiling->nBL1;
        }

        Dn2NzParams intriParams;
        intriParams.dnNum = self_->ctx.convTiling->kwL1;
        intriParams.nValue = self_->ctx.currentNBL1;
        intriParams.dValue = self_->ctx.cinBL1Iter == self_->ctx.maxCinBL1Iter ?
            self_->ctx.convTiling->cinBTailInCore : self_->ctx.convTiling->cinBInCore;
        intriParams.srcDnMatrixStride =  self_->ctx.orgCi / self_->ctx.convTiling->groups * self_->ctx.orgCo;
        intriParams.srcDValue = self_->ctx.orgCo;
        intriParams.dstNzC0Stride = self_->ctx.convTiling->khL1 * self_->ctx.convTiling->kwL1 *
            self_->ctx.convTiling->nBL1;
        intriParams.dstNzNStride = 1;
        intriParams.dstNzMatrixStride = self_->ctx.convTiling->nBL1 * Intf::k0;
        uint64_t bl1DstOffset = 0;
        for (uint16_t khIterIdx = 0; khIterIdx < self_->ctx.convTiling->khL1; khIterIdx++) {
            bl1DstOffset = khIterIdx * self_->ctx.convTiling->kwL1 * self_->ctx.convTiling->nBL1 * Intf::k0;
            DataCopy<typename Intf::WeightT>(self_->ctx.bl1[bl1DstOffset], self_->ctx.bgm[bL1GmOffset], intriParams);
            bL1GmOffset += self_->ctx.convTiling->kernelW * self_->ctx.orgCi / self_->ctx.convTiling->groups *
                           self_->ctx.orgCo;
        }
    }

private:
    Intf *self_ = nullptr;
    bool isOneByOneKernelScenario = false;
};

};

#endif // CONV2D_V2_INSTR_BASE_IMPL_H