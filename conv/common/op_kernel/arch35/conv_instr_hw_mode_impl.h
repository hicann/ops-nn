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
 * \file conv_instr_hw_mode_impl.h
 * \brief
 */

#ifndef CONV_INSTR_HW_MODE_IMPL_H
#define CONV_INSTR_HW_MODE_IMPL_H

#include "conv_config.h"
#include "conv_util.h"

namespace ConvFunc {
using namespace AscendC;
using namespace conv;

template <class Intf>
class LoadAL0ToolsHWMode {
public:
    __aicore__ inline LoadAL0ToolsHWMode() {}
    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
        if constexpr (Intf::isConv3D) {
            if constexpr (Intf::kPreLoadFlag) {
                channelSize_ = self_->ctx.kAL1Iter - 1 != self_->ctx.maxKAL1Iter ? self_->ctx.convTilingData->convApiTiling.cinAInCore :
                    self_->ctx.convTilingData->convApiTiling.cinATailInCore;
            } else {
                channelSize_ = self_->ctx.kAL1Iter != self_->ctx.maxKAL1Iter ? self_->ctx.convTilingData->convApiTiling.cinAInCore :
                    self_->ctx.convTilingData->convApiTiling.cinATailInCore;
            }
        } else {
            if constexpr (Intf::c04Flag) {
                channelSize_ = conv::C04_CIN_SIZE;
                c04KStepTail = (channelSize_ * self_->ctx.convTilingData->convApiTiling.kernelHxkernelW) % self_->ctx.convTilingData->convApiTiling.kL0;
                c04KStepTail = c04KStepTail == 0 ? self_->ctx.convTilingData->convApiTiling.kL0 : c04KStepTail;
            } else {
                if constexpr (Intf::kPreLoadFlag) {
                    if constexpr (Intf::isKL1NL0FullLoad) {
                        channelSize_ = AlignB(self_->ctx.convTilingData->convApiTiling.cinATailInCore, Intf::k0);
                    } else {
                        channelSize_ = self_->ctx.kAL1Iter - 1 != self_->ctx.maxKAL1Iter ?
                            self_->ctx.convTilingData->convApiTiling.cinAInCore : AlignB(self_->ctx.convTilingData->convApiTiling.cinATailInCore, Intf::k0);
                    }
                } else {
                    if constexpr (Intf::isKL1NL0FullLoad) {
                        channelSize_ = AlignB(self_->ctx.convTilingData->convApiTiling.cinATailInCore, Intf::k0);
                    } else {
                        channelSize_ = self_->ctx.kAL1Iter != self_->ctx.maxKAL1Iter ?
                            self_->ctx.convTilingData->convApiTiling.cinAInCore : AlignB(self_->ctx.convTilingData->convApiTiling.cinATailInCore, Intf::k0);
                    }
                }
            }
        }
    }

    __aicore__ inline void SetM(uint64_t m, uint64_t mNotAlign)
    {
        currentML0_ = m;
        if constexpr (!Intf::isDmaFlag) {
            currentML0Align_ = mNotAlign;
            mStartPt_ = self_->ctx.hoL0Iter * self_->ctx.currentWoL1 * self_->ctx.convTilingData->convApiTiling.hoL0 +
                        self_->ctx.woL0Iter * self_->ctx.convTilingData->convApiTiling.woL0;
            SetLoad3dRepeatParams();
        }
    }

    __aicore__ inline void SetLoad3dRepeatParams()
    {
        uint16_t dstStride = 0;
        uint16_t repeatStride = 0;
        uint8_t repeatTime = 1;
        uint8_t repeatMode = 0;
        mExtension_ = currentML0Align_;

        if (self_->ctx.currentWoL0 == self_->ctx.convTilingData->convApiTiling.woL0 &&
            self_->ctx.currentHoL0 == self_->ctx.convTilingData->convApiTiling.hoL0) {
            dstStride = self_->ctx.convTilingData->convApiTiling.fmapKStride;
        } else {
            dstStride = currentML0_ / BLOCK_L0_M;
        }

        if (unlikely(CeilDiv(self_->ctx.currentWoL1, self_->ctx.convTilingData->convApiTiling.woL0) > 1)) {
            mExtension_ = self_->ctx.currentWoL0;
            repeatStride = self_->ctx.currentWoL1 / BLOCK_L0_M;
            repeatTime = self_->ctx.currentHoL0;
        }
#if defined(ASC_DEVKIT_VERSION_NUM) && (ASC_DEVKIT_VERSION_NUM >= 90000000)
        LoadDataRepeatParamWithStride repeatParams = {repeatStride, repeatTime, repeatMode, dstStride};
        SetLoadDataRepeatWithStride(repeatParams);
#else
        LoadDataRepeatParam repeatParams = {repeatStride, repeatTime, repeatMode, dstStride};
        SetLoadDataRepeat(repeatParams);
#endif
    }

    __aicore__ inline void SetFirst()
    {
        xm_.bf.mExtension_ = mExtension_ & MASK_16;
        xm_.bf.mStartPt_ = mStartPt_ & MASK_16;

        xt_.n = static_cast<uint64_t>(self_->ctx.convTilingData->convApiTiling.unionDataXt);
        xt_.bf.channelSize = channelSize_;
        param_.SetConfig1(xt_.n);
    }

    __aicore__ inline uint64_t GetC04KStepTail()
    {
        return c04KStepTail;
    }

    __aicore__ inline void LoadAL0(const uint64_t &currentKL0, const uint64_t &posK, const uint64_t &kIter, const LocalTensor<typename Intf::FmapT> &al0)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (Intf::isDmaFlag) {
            DmaLoad2DImpl(kIter, al0);
            return;
        } else {
            xm_.bf.currentKL0 = currentKL0 & MASK_16;
            xm_.bf.posK = posK & MASK_16;
            param_.SetConfig0(xm_.n);

            LoadData<TPosition::A2, TPosition::A1, typename Intf::FmapT>(al0, self_->ctx.al1, param_);
        }
    };

private:
    __aicore__ inline bool IsKL0Tail(const uint64_t &kIter)
    {
        return kIter == self_->ctx.maxKL0Iter;
    }

 __aicore__ inline void DmaLoad2DImpl(const uint64_t &kIter, const LocalTensor<typename Intf::FmapT> &al0)
    {
        uint64_t currentWoL1Align = AlignB(self_->ctx.currentWoL1, BLOCK_L0_M);
        uint64_t currentKL0 = IsKL0Tail(kIter) ? self_->ctx.kL0Tail : self_->ctx.convTilingData->convApiTiling.kL0;
 
        LoadData2DParamsV2 loadParams;
        loadParams.kStartPosition = (kIter % self_->ctx.multiKAL1) * self_->ctx.convTilingData->convApiTiling.kL0 / Intf::k0;
        loadParams.mStep = CeilDiv(self_->ctx.currentWoL0, BLOCK_L0_M);
        loadParams.kStep = currentKL0 / Intf::k0;
        loadParams.srcStride = self_->ctx.currentHoL1 * currentWoL1Align / BLOCK_L0_M;
        loadParams.dstStride = currentML0_ / BLOCK_L0_M;
 
        uint32_t mStartPosition = self_->ctx.hoL0Iter * self_->ctx.convTilingData->convApiTiling.hoL0 * currentWoL1Align +
                                  self_->ctx.woL0Iter * self_->ctx.convTilingData->convApiTiling.woL0;
        uint32_t dstOffset = 0;
        uint32_t dstOffsetStride = AlignB(self_->ctx.currentWoL0, BLOCK_L0_M) * Intf::k0;
        for (uint16_t hoL0Idx = 0; hoL0Idx < self_->ctx.currentHoL0; ++ hoL0Idx) {
            loadParams.mStartPosition = mStartPosition / BLOCK_L0_M;
            LoadData<typename Intf::FmapT>(al0[dstOffset], self_->ctx.al1, loadParams);
            mStartPosition += currentWoL1Align;
            dstOffset += dstOffsetStride;
        }
    }

private:
    Intf *self_ = nullptr;
    uint64_t currentML0_ = 0;
    uint64_t currentML0Align_ = 0;
    uint16_t mExtension_ = 0;
    uint16_t mStartPt_ = 0;
    uint16_t channelSize_ = 0;
    uint64_t c04KStepTail = 0;
    Load3DBitModeParam param_;
    UnionDataXt xt_;
    UnionDataXm xm_;
};

template <class Intf, typename OutputT, uint64_t FixpipeIdx = 0>
class CopyOutToolsHWMode {
public:
    __aicore__ inline CopyOutToolsHWMode() {}
    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
        valueHoWo_ = self_->ctx.convTilingData->convApiTiling.orgHo * self_->ctx.convTilingData->convApiTiling.orgWo;
    }

    __aicore__ inline void SetMN(uint64_t m, uint64_t n)
    {
        currentML0_ = m;
        currentNL0_ = n;
    }

    __aicore__ inline void SetFixpipeIntriParams(FixpipeParamsC310<CO2Layout::COLUMN_MAJOR> &intriParams)
    {
        if constexpr (Intf::isDmaFlag) {
            intriParams.mSize = self_->ctx.currentWoL0;
            intriParams.params.srcNzMatrixStride = AlignB(self_->ctx.currentWoL0, BLOCK_L0_M);
            intriParams.params.dnNum = self_->ctx.currentHoL0;
            intriParams.params.dstDnMatrixStride = self_->ctx.singleCoreWo;
        } else {
            if (likely(self_->ctx.convTilingData->convApiTiling.woL0 >= self_->ctx.singleCoreWo)) {
                intriParams.mSize = currentML0_;
                intriParams.params.srcNzMatrixStride = 0;
                intriParams.params.dnNum = 1;
                intriParams.params.dstDnMatrixStride = 0;
            } else {
                intriParams.mSize = self_->ctx.currentWoL0;
                intriParams.params.srcNzMatrixStride = self_->ctx.currentWoL0;
                intriParams.params.dnNum = self_->ctx.currentHoL0;
                intriParams.params.dstDnMatrixStride = self_->ctx.singleCoreWo;
            }
        }

        intriParams.nSize = currentNL0_;
        intriParams.srcStride = AlignB(currentML0_, BLOCK_L0_M);
        if constexpr (Intf::formatOutput == ConvFormat::NCDHW) {
            intriParams.dstStride = self_->ctx.convTilingData->convApiTiling.orgDo * valueHoWo_;
        } else {
            intriParams.dstStride = valueHoWo_;
        }
        intriParams.params.srcNzC0Stride = 1;
        SetBaseParams<CO2Layout::COLUMN_MAJOR>(intriParams);
    }

    __aicore__ inline void SetFixpipeIntriParamsHWC(FixpipeParamsC310<CO2Layout::ROW_MAJOR> &intriParams)
    {
        if constexpr (Intf::isDmaFlag) {
            intriParams.mSize = self_->ctx.currentWoL0;
            intriParams.params.srcNdStride = AlignB(self_->ctx.currentWoL0, BLOCK_L0_M);
            intriParams.params.ndNum = self_->ctx.currentHoL0;
            intriParams.params.dstNdStride = self_->ctx.singleCoreWo * self_->ctx.convTilingData->convApiTiling.orgCo;
        } else if (likely(self_->ctx.convTilingData->convApiTiling.woL0 >= self_->ctx.singleCoreWo)) {
            intriParams.mSize = currentML0_;
            intriParams.params.srcNdStride = 0;
            intriParams.params.ndNum = 1;
            intriParams.params.dstNdStride = 0;
        } else {
            intriParams.mSize = self_->ctx.currentWoL0;
            intriParams.params.srcNdStride = self_->ctx.currentWoL0;
            intriParams.params.ndNum = self_->ctx.currentHoL0;
            intriParams.params.dstNdStride = self_->ctx.singleCoreWo * self_->ctx.convTilingData->convApiTiling.orgCo;
        }

        intriParams.dstStride = self_->ctx.convTilingData->convApiTiling.orgCo;
        intriParams.srcStride = AlignB(currentML0_, BLOCK_L0_M);
        intriParams.nSize = currentNL0_;
        SetBaseParams<CO2Layout::ROW_MAJOR>(intriParams);
    }

    template <CO2Layout format>
    __aicore__ inline void SetFixpipeIntriParamsUb(FixpipeParamsC310<format> &intriParams, CopyUbInfo* ubInfo = nullptr)
    {
        if (ubInfo == nullptr) {
            return;
        }
        uint64_t unUsedNL0 = currentNL0_ >= ubInfo->nLoopIdx * ubInfo->nUb ? currentNL0_ - ubInfo->nLoopIdx * ubInfo->nUb : 0;
        uint64_t unUsedML0 = currentML0_ >= ubInfo->mLoopIdx * ubInfo->mUb ? currentML0_ - ubInfo->mLoopIdx * ubInfo->mUb : 0;
        ubInfo->realNUb = unUsedNL0 < ubInfo->nUb ? unUsedNL0 : ubInfo->nUb;
        ubInfo->realHUb = 0;
        ubInfo->realWUb = unUsedML0 < ubInfo->mUb ? unUsedML0 : ubInfo->mUb;
        ubInfo->realBatchUb = 1;
        ubInfo->outBatchIdx = self_->ctx.batchIter + ubInfo->batchLoopIdx * ubInfo->batchUb;
        if constexpr (Intf::isKL1NL0FullLoad) {
            ubInfo->outCIdx = ubInfo->nLoopIdx * ubInfo->nUb;
        } else {
            ubInfo->outCIdx = self_->ctx.nBL1Iter * self_->ctx.convTilingData->convApiTiling.nBL1 +
                              self_->ctx.nL0Iter * self_->ctx.convTilingData->convApiTiling.nL0 + ubInfo->nLoopIdx * ubInfo->nUb;
        }
        ubInfo->outHIdx = self_->ctx.hoAL1Iter * self_->ctx.convTilingData->convApiTiling.hoL1 +
                         self_->ctx.hoL0Iter * self_->ctx.convTilingData->convApiTiling.hoL0 + ubInfo->mLoopIdx * ubInfo->mUb / self_->ctx.convTilingData->convApiTiling.woL0;
        if (self_->ctx.woL1SmallTail == 0) {
            ubInfo->outWIdx = self_->ctx.woAL1Iter * self_->ctx.convTilingData->convApiTiling.woL1 +
                             self_->ctx.woL0Iter * self_->ctx.convTilingData->convApiTiling.woL0 + ubInfo->mLoopIdx * ubInfo->mUb % self_->ctx.convTilingData->convApiTiling.woL0;
        } else {
            if (self_->ctx.woAL1Iter == self_->ctx.maxWoL1Iter) {
                ubInfo->outWIdx = ((self_->ctx.woAL1Iter - 1) * self_->ctx.convTilingData->convApiTiling.woL1 + self_->ctx.woAL1Tail) +
                                 self_->ctx.woL0Iter * self_->ctx.convTilingData->convApiTiling.woL0 + ubInfo->mLoopIdx * ubInfo->mUb % self_->ctx.convTilingData->convApiTiling.woL0;
            } else {
                ubInfo->outWIdx = self_->ctx.woAL1Iter * self_->ctx.convTilingData->convApiTiling.woL1 +
                                 self_->ctx.woL0Iter * self_->ctx.convTilingData->convApiTiling.woL0 + ubInfo->mLoopIdx * ubInfo->mUb % self_->ctx.convTilingData->convApiTiling.woL0;
            }
        }

        intriParams.nSize = AlignB(ubInfo->realNUb, BLOCK_L0_N);
        intriParams.mSize = AlignB(ubInfo->realWUb, BLOCK_L0_M);
        intriParams.srcStride = self_->ctx.currentML0Align;
        if constexpr (format == CO2Layout::ROW_MAJOR) {
            intriParams.dstStride = AlignB(ubInfo->realNUb, BLOCK_L0_N);
            intriParams.params.dstNdStride = 0;
            intriParams.params.srcNdStride = 0;
            intriParams.params.ndNum = 1;
        } else if constexpr (format == CO2Layout::COLUMN_MAJOR) {
            intriParams.dstStride = AlignB(ubInfo->realWUb, BLOCK_L0_M);;
            intriParams.params.dnNum = 1;
            intriParams.params.dstDnMatrixStride = 0;
            intriParams.params.srcNzMatrixStride = 0;
            intriParams.params.srcNzC0Stride = 1;
        }
        SetBaseParams<format>(intriParams);
    }

    template <CO2Layout format>
    __aicore__ inline void SetBaseParams(FixpipeParamsC310<format> &intriParams)
    {
        intriParams.quantPre = GetQuantPre<Intf, OutputT, FixpipeIdx>(self_);
        if (self_->ctx.convTilingData->convApiTiling.hasScale == 0) {
            intriParams.deqScalar = DEQ_SCALAR_ONE;
        }
        if constexpr (Intf::isExtendConv2d) {
            if constexpr (FixpipeIdx == 0) {
                intriParams.reluEn = self_->ctx.convTilingData->convApiTiling.reluMode0 != 0;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
                intriParams.preReluMode = static_cast<ReluMode>(self_->ctx.convTilingData->convApiTiling.reluMode0);
                if (self_->ctx.convTilingData->convApiTiling.reluMode0 == static_cast<uint8_t>(ReluMode::SCALAR_RELU)) {
                    intriParams.reluScalar = self_->ctx.preReluScalar0;
                } else if (self_->ctx.convTilingData->convApiTiling.reluMode0 == static_cast<uint8_t>(ReluMode::VECTOR_RELU)) {
                    intriParams.vectorRelu =
                        self_->ctx.reluWeightL1[GetExtendConv2dScaleL1Addr()].GetPhyAddr();
                }
#endif
                intriParams.deqScalar = self_->ctx.deqScalar0;
            } else {
                intriParams.reluEn = self_->ctx.convTilingData->convApiTiling.reluMode1 != 0;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
                intriParams.preReluMode = static_cast<ReluMode>(self_->ctx.convTilingData->convApiTiling.reluMode1);
                if (self_->ctx.convTilingData->convApiTiling.reluMode1 == static_cast<uint8_t>(ReluMode::SCALAR_RELU)) {
                    intriParams.reluScalar = self_->ctx.preReluScalar1;
                } else if(self_->ctx.convTilingData->convApiTiling.reluMode1 == static_cast<uint8_t>(ReluMode::VECTOR_RELU)) {
                    intriParams.vectorRelu =
                        self_->ctx.reluWeightL1[GetExtendConv2dScaleL1Addr() + self_->ctx.reluWeight1L1offset].GetPhyAddr();
                }
#endif
                intriParams.deqScalar = self_->ctx.deqScalar1;
            }
        }
        if constexpr (FixpipeIdx == 1) {
            intriParams.unitFlag = UNIT_FLAG_ENABLE_WITH_FLIP;
        } else {
            if constexpr (Intf::isExtendConv2d) {
                if (self_->ctx.convTilingData->convApiTiling.dualOutput) {
                    intriParams.unitFlag = UNIT_FLAG_ENABLE_ONLY;
                } else {
                    intriParams.unitFlag = UNIT_FLAG_ENABLE_WITH_FLIP;
                }
            } else {
                intriParams.unitFlag = UNIT_FLAG_ENABLE_WITH_FLIP;
            }
        }
    }

    __aicore__ inline uint64_t CalcFixpipeOffset()
    {
        uint64_t offset = self_->ctx.batchIter * self_->ctx.outputOneBatchSize;
        uint64_t offsetH = self_->ctx.hoAL1Iter * self_->ctx.convTilingData->convApiTiling.hoL1 +
                           self_->ctx.hoL0Iter * self_->ctx.convTilingData->convApiTiling.hoL0;
        uint64_t offsetW;
        if (self_->ctx.woL1SmallTail == 0) {
            offsetW = self_->ctx.woAL1Iter * self_->ctx.convTilingData->convApiTiling.woL1 +
                      self_->ctx.woL0Iter * self_->ctx.convTilingData->convApiTiling.woL0;
        } else {
            if (self_->ctx.woAL1Iter == self_->ctx.maxWoL1Iter) {
                offsetW = ((self_->ctx.woAL1Iter - 1) * self_->ctx.convTilingData->convApiTiling.woL1 + self_->ctx.woAL1Tail) +
                          self_->ctx.woL0Iter * self_->ctx.convTilingData->convApiTiling.woL0;
            } else {
                offsetW = self_->ctx.woAL1Iter * self_->ctx.convTilingData->convApiTiling.woL1 +
                          self_->ctx.woL0Iter * self_->ctx.convTilingData->convApiTiling.woL0;
            }
        }

        uint64_t offsetCout = 0U;
        if constexpr (!Intf::isKL1NL0FullLoad) {
            offsetCout = self_->ctx.nBL1Iter * self_->ctx.convTilingData->convApiTiling.nBL1 +
                         self_->ctx.nL0Iter * self_->ctx.convTilingData->convApiTiling.nL0;
        }
        if constexpr (Intf::groupOptPreloadFlag) {
            offsetCout += self_->ctx.groupOptIter * self_->ctx.convTilingData->convApiTiling.orgCo /
                          self_->ctx.convTilingData->convApiTiling.groups *
                          self_->ctx.convTilingData->convApiTiling.enlarge; 
        }
        if constexpr (Intf::formatOutput == ConvFormat::NCDHW) {
            offset += offsetCout * self_->ctx.convTilingData->convApiTiling.orgDo * valueHoWo_ + self_->ctx.dOutIter * valueHoWo_ +
                offsetH * self_->ctx.convTilingData->convApiTiling.orgWo + offsetW;
        } else if constexpr (Intf::formatOutput == ConvFormat::NDHWC) {
            offset += self_->ctx.dOutIter * valueHoWo_ * self_->ctx.convTilingData->convApiTiling.orgCo + offsetH * self_->ctx.convTilingData->convApiTiling.orgWo *
                self_->ctx.convTilingData->convApiTiling.orgCo + offsetW * self_->ctx.convTilingData->convApiTiling.orgCo + offsetCout;
        } else if constexpr (Intf::formatOutput == ConvFormat::NCHW) {
            offset += offsetCout * valueHoWo_ + offsetH * self_->ctx.convTilingData->convApiTiling.orgWo + offsetW;
        } else {
            offset += offsetH * self_->ctx.convTilingData->convApiTiling.orgWo * self_->ctx.convTilingData->convApiTiling.orgCo + offsetW * self_->ctx.convTilingData->convApiTiling.orgCo + offsetCout;
        }

        return offset;
    }

    template <template <typename> class TensorTypeT, const FixpipeConfig &config>
    __aicore__ inline void ExtendConv2DFixpipe(const TensorTypeT<OutputT> &output, 
        FixpipeParamsC310<config.format> &intriParams, uint64_t offset)
    {
        if (self_->ctx.enableVectorQuant) {
            if constexpr (FixpipeIdx == 0) {
                Fixpipe<OutputT, typename Intf::L0cT, config>(
                    output[offset], self_->ctx.cl0, self_->ctx.scaleL1[GetExtendConv2dScaleL1Addr()], intriParams);
            } else {
                Fixpipe<OutputT, typename Intf::L0cT, config>(
                    output[offset], self_->ctx.cl0,
                    self_->ctx.scaleL1[GetExtendConv2dScaleL1Addr() + self_->ctx.scale1L1offset], intriParams);
            }
        } else {
            Fixpipe<OutputT, typename Intf::L0cT, config>(output[offset], self_->ctx.cl0, intriParams);
        }
    }

    __aicore__ inline uint64_t GetExtendConv2dScaleL1Addr()
    {
        if constexpr (Intf::isKL1NL0FullLoad) {
            return 0;
        }
        if (self_->ctx.convTilingData->convApiTiling.fixpParamsFullLoadFlag) {
            return self_->ctx.nBL1Iter * self_->ctx.convTilingData->convApiTiling.nBL1 +
                self_->ctx.nL0Iter * self_->ctx.convTilingData->convApiTiling.nL0;
        }
        return 0;
    }

    template <template <typename> class TensorTypeT, const FixpipeConfig &config>
    __aicore__ inline void CopyOut(const TensorTypeT<OutputT> &output, CopyUbInfo* ubInfo = nullptr)
    {
        uint64_t offset = 0;
        if constexpr (Intf::posOutput == TPosition::GM) {
            offset = CalcFixpipeOffset();
        }

        FixpipeParamsC310<config.format> intriParams;
        if constexpr (Intf::posOutput == TPosition::VECCALC) {
            SetFixpipeIntriParamsUb<config.format>(intriParams, ubInfo);
            if (ubInfo->realNUb == 0 || ubInfo->realWUb == 0) {
                return;
            }
        } else if constexpr (Intf::formatOutput == ConvFormat::NDHWC || Intf::formatOutput == ConvFormat::NHWC) {
            SetFixpipeIntriParamsHWC(intriParams);
        } else {
            SetFixpipeIntriParams(intriParams);
        }

        if constexpr (Intf::isExtendConv2d) {
            ExtendConv2DFixpipe<TensorTypeT, config>(output, intriParams, offset);
        } else if constexpr (Intf::isQuantScene) {
            if (self_->ctx.convTilingData->convApiTiling.hasScale != 0) {
                Fixpipe<OutputT, typename Intf::L0cT, config>(
                    output[offset], self_->ctx.cl0, self_->ctx.scaleL1[GetScaleL1Addr()], intriParams);
            } else {
                Fixpipe<OutputT, typename Intf::L0cT, config>(output[offset], self_->ctx.cl0, intriParams);
            }
        } else {
            Fixpipe<OutputT, typename Intf::L0cT, config>(output[offset], self_->ctx.cl0, intriParams);
        }
    }

private:
    Intf *self_ = nullptr;
    uint64_t valueHoWo_ = 0;
    uint64_t currentML0_ = 0;
    uint64_t currentNL0_ = 0;

    __aicore__ inline uint64_t GetScaleL1Addr()
    {
        if constexpr (Intf::isKL1NL0FullLoad) {
            return 0;
        }
        if constexpr (Intf::isQuantScene) {
            if (self_->ctx.convTilingData->convApiTiling.hasScale != 0 && self_->ctx.convTilingData->convApiTiling.fixpParamsFullLoadFlag) {
                return self_->ctx.nBL1Iter * self_->ctx.convTilingData->convApiTiling.nBL1 +
                       self_->ctx.nL0Iter * self_->ctx.convTilingData->convApiTiling.nL0;
            }
        }
        return 0;
    }
};

};

#endif // CONV_INSTR_HW_MODE_IMPL_H