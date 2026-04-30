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
 * \file conv_instr_m_mode_impl.h
 * \brief
 */

#ifndef CONV_INSTR_M_MODE_IMPL_H
#define CONV_INSTR_M_MODE_IMPL_H

#include "conv_config.h"
#include "conv_util.h"

namespace ConvFunc {
using namespace AscendC;
using namespace conv;

template <class Intf>
class LoadAL0ToolsMMode {
public:
    __aicore__ inline LoadAL0ToolsMMode()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
        alignCinATailInCore_ = AlignB(self_->ctx.convTilingData->convApiTiling.cinATailInCore, Intf::k0FmapTail);

        if constexpr (Intf::c04Flag) {
            channelSize_ = conv::C04_CIN_SIZE;
            c04KStepTail = (conv::C04_CIN_SIZE * self_->ctx.convTilingData->convApiTiling.kernelHxkernelW) % self_->ctx.convTilingData->convApiTiling.kL0;
            c04KStepTail = c04KStepTail == 0 ? self_->ctx.convTilingData->convApiTiling.kL0 : c04KStepTail;
        } else {
            if constexpr (Intf::kPreLoadFlag) {
                if constexpr (Intf::isKL1NL0FullLoad) {
                    channelSize_ = alignCinATailInCore_;
                } else {
                    channelSize_ = (self_->ctx.kAL1Iter - 1 != self_->ctx.maxKAL1Iter ?
                                    self_->ctx.convTilingData->convApiTiling.cinAInCore : alignCinATailInCore_);
                }
            } else {
                if constexpr (Intf::isKL1NL0FullLoad) {
                    channelSize_ = alignCinATailInCore_;
                } else {
                    channelSize_ = (self_->ctx.kAL1Iter != self_->ctx.maxKAL1Iter ?
                                    self_->ctx.convTilingData->convApiTiling.cinAInCore : alignCinATailInCore_);
                }
            }
        }

        if constexpr (Intf::ConvParam::innerBatch == static_cast<int8_t>(ConvInnerBatch::MULTI_BATCH)) {
            uint64_t hiLoadL1 = (self_->ctx.convTilingData->convApiTiling.orgHo - 1) * self_->ctx.convTilingData->convApiTiling.strideH +
                                self_->ctx.dilatedKernelH;
            hiLoadL1 = hiLoadL1 > self_->ctx.convTilingData->convApiTiling.orgHi ? self_->ctx.convTilingData->convApiTiling.orgHi : hiLoadL1;
            realHixWi = hiLoadL1 * self_->ctx.convTilingData->convApiTiling.orgWi;
        }
    }

    __aicore__ inline void SetM(uint64_t m, uint64_t mNotAlign)
    {
        if constexpr (Intf::ConvParam::innerBatch == static_cast<int8_t>(ConvInnerBatch::KERNEL_1X1_MULTI_BATCH)) {
            if constexpr (Intf::formatOutput == ConvFormat::NCHW && Intf::c04Flag) {
                currentML0_ = self_->ctx.innerBatch * AlignB(C04_CIN_SIZE * mNotAlign, Intf::k0) / C04_CIN_SIZE;
                currentML0Align_ = AlignB(currentML0_, BLOCK_L0_M);
            } else {
                currentML0Align_ = AlignB(self_->ctx.innerBatch * mNotAlign, BLOCK_L0_M);
                currentML0_ = self_->ctx.innerBatch * mNotAlign;
            }
        } else {
            currentML0Align_ = m;
            currentML0_ = mNotAlign;
        }
#if defined(ASC_DEVKIT_VERSION_NUM) && (ASC_DEVKIT_VERSION_NUM >= 90000000)
        LoadDataRepeatParamWithStride repeatParams = {0, 1, 0, static_cast<uint16_t>(currentML0Align_ / BLOCK_L0_M)};
        SetLoadDataRepeatWithStride(repeatParams);
#else
        LoadDataRepeatParam repeatParams = {0, 1, 0, static_cast<uint16_t>(currentML0Align_ / BLOCK_L0_M)};
        SetLoadDataRepeat(repeatParams);
#endif
    }

    __aicore__ inline void SetFirst()
    {
        uint64_t posM = self_->ctx.mL0Iter * self_->ctx.mL0 +
                            (self_->ctx.mStartPos + self_->ctx.mAL1Iter * self_->ctx.mAL1) % self_->ctx.convTilingData->convApiTiling.orgWo;
        xm_.bf.mExtension_ = currentML0_ & MASK_16;
        xm_.bf.mStartPt_ = posM & MASK_16;

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
        xm_.bf.currentKL0 = currentKL0 & MASK_16;
        xm_.bf.posK = posK & MASK_16;
        param_.SetConfig0(xm_.n);

        if constexpr (Intf::preFusionFlag) {
            uint64_t aL1Offset = 0;
            if ((self_->ctx.convTilingData->convApiTiling.pBufferFlag & AL1_DB_IDX) >> AL1_DB_OFFSET) {
                aL1Offset = (self_->ctx.al1PingPongFlag^1) * self_->ctx.convTilingData->convApiTiling.aL1SpaceSize / Intf::sizeOfFmap;
            }
            LoadData<TPosition::A2, TPosition::A1, typename Intf::FmapT>(al0,
                self_->ctx.al1[aL1Offset], param_);
        } else if constexpr (Intf::ConvParam::innerBatch == static_cast<int8_t>(ConvInnerBatch::MULTI_BATCH)) {
            uint32_t srcOffset = 0;
            uint32_t dstOffset = 0;
            uint32_t srcBatchStride = 0;
            if constexpr (Intf::c04Flag) {
                srcBatchStride = AlignB(C04_CIN_SIZE * self_->ctx.convTilingData->convApiTiling.orgHixWi, Intf::k0);
            } else {
                srcBatchStride = ((kIter / self_->ctx.multiKAL1) != self_->ctx.maxKAL1Iter ?
                    self_->ctx.convTilingData->convApiTiling.cinAInCore : alignCinATailInCore_) * realHixWi;
            }
            uint32_t dstBatchStride = currentML0Align_ * self_->ctx.convTilingData->convApiTiling.kL0;
            for (uint16_t batchIter = 0; batchIter < self_->ctx.innerBatch; batchIter++) {
                LoadData<TPosition::A2, TPosition::A1, typename Intf::FmapT>(al0[dstOffset],
                                                                            self_->ctx.al1[srcOffset], param_);
                srcOffset += srcBatchStride;
                dstOffset += dstBatchStride;
            }
        } else {
            LoadData<TPosition::A2, TPosition::A1, typename Intf::FmapT>(al0, self_->ctx.al1, param_);
        }
    };


private:
    Intf *self_ = nullptr;
    uint64_t currentML0_ = 0;
    uint64_t currentML0Align_ = 0;
    uint64_t alignCinATailInCore_ = 0;
    uint16_t channelSize_ = 0;
    uint64_t c04KStepTail = 0;
    uint64_t realHixWi = 0;
    Load3DBitModeParam param_;
    UnionDataXt xt_;
    UnionDataXm xm_;
};

template <class Intf, typename OutputT, uint64_t FixpipeIdx = 0>
class CopyOutToolsMMode {
public:
    __aicore__ inline CopyOutToolsMMode()
    {}

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

    template <template <typename> class TensorTypeT, const FixpipeConfig &config>
    __aicore__ inline void CopyOut(const TensorTypeT<OutputT> &output, CopyUbInfo* ubInfo = nullptr)
    {
        uint64_t offset = 0;
        if constexpr (Intf::posOutput == TPosition::GM) {
            offset = CalcFixpipeOffset();
        }

        if constexpr (Intf::isInnerBatchFlag) {
            CopyOutInnerBatch<TensorTypeT, config.format, config>(output, offset, ubInfo);
        } else {
            FixpipeParamsC310<config.format> intriParams;
            if constexpr (Intf::posOutput == TPosition::VECCALC) {
                SetFixpipeIntriParamsUb<config.format>(intriParams, ubInfo);
                if (ubInfo->realWUb == 0 || ubInfo->realNUb == 0) {
                    return;
                }
            } else if constexpr (Intf::formatOutput == ConvFormat::NHWC || Intf::formatOutput == ConvFormat::NDHWC) {
                SetFixpipeIntriParamsHWC(intriParams);
            } else {
                SetFixpipeIntriParams(intriParams);
            }

            if constexpr (Intf::isExtendConv2d) {
                ExtendConv2DFixpipe<TensorTypeT, config>(output, intriParams, offset);
            } else if constexpr (Intf::isQuantScene) {
                if (self_->ctx.convTilingData->convApiTiling.hasScale == 0) {
                    Fixpipe<OutputT, typename Intf::L0cT, config>(output[offset], self_->ctx.cl0, intriParams);
                } else {
                    Fixpipe<OutputT, typename Intf::L0cT, config>(
                        output[offset], self_->ctx.cl0, self_->ctx.scaleL1[GetScaleL1Addr()], intriParams);
                }
            } else {
                Fixpipe<OutputT, typename Intf::L0cT, config>(output[offset], self_->ctx.cl0, intriParams);
            }
        }
    }

private:
    template <template <typename> class TensorTypeT, CO2Layout format, const FixpipeConfig &config>
    __aicore__ inline void CopyOutInnerBatch(const TensorTypeT<OutputT> &output, uint64_t offset, CopyUbInfo* ubInfo = nullptr)
    {
        FixpipeParamsC310<format> intriParams;
        if constexpr (Intf::posOutput == TPosition::VECCALC) {
            SetFixpipeIntriParamsUb<format>(intriParams, ubInfo);
            if (ubInfo->realBatchUb == 0 || ubInfo->realNUb == 0 || ubInfo->realWUb == 0) {
                return;
            }
        } else if constexpr (Intf::formatOutput == ConvFormat::NHWC) {
            InnerBatchParamsHWC(intriParams);
        } else {
            InnerBatchParamsCHW(intriParams);
        }
        SetBaseParams<format>(intriParams);

        if constexpr (Intf::isExtendConv2d) {
            ExtendConv2DFixpipe<TensorTypeT, config>(output, intriParams, offset);
        } else if constexpr (Intf::isQuantScene) {
            if (self_->ctx.convTilingData->convApiTiling.hasScale != 0) {
                Fixpipe<OutputT, typename Intf::L0cT, config>(
                    output[offset], self_->ctx.cl0, self_->ctx.scaleL1[GetScaleL1Addr()], intriParams);
            } else {
                Fixpipe<OutputT, typename Intf::L0cT, config>(
                    output[offset], self_->ctx.cl0, intriParams);
            }
        } else {
            Fixpipe<OutputT, typename Intf::L0cT, config>(output[offset], self_->ctx.cl0, intriParams);
        }
    }

    __aicore__ inline void InnerBatchParamsCHW(FixpipeParamsC310<CO2Layout::COLUMN_MAJOR> &intriParams)
    {
        intriParams.mSize = currentML0_;
        intriParams.nSize = currentNL0_;
        intriParams.params.dnNum = self_->ctx.innerBatch;
        if constexpr (Intf::ConvParam::innerBatch == static_cast<int8_t>(ConvInnerBatch::KERNEL_1X1_MULTI_BATCH)) {
            if constexpr (Intf::formatOutput == ConvFormat::NCHW && Intf::c04Flag) {
                intriParams.srcStride = AlignB(self_->ctx.innerBatch * AlignB(C04_CIN_SIZE * currentML0_, Intf::k0) / C04_CIN_SIZE, BLOCK_L0_M);
                intriParams.params.srcNzMatrixStride = AlignB(C04_CIN_SIZE * currentML0_, Intf::k0) / C04_CIN_SIZE;
            } else {
                intriParams.srcStride = AlignB(self_->ctx.innerBatch * currentML0_, BLOCK_L0_M);
                intriParams.params.srcNzMatrixStride = currentML0_;
            }       
        } else {
            intriParams.srcStride = self_->ctx.currentML0Align;
            intriParams.params.srcNzMatrixStride = self_->ctx.currentML0Align * CeilDiv(currentNL0_ , BLOCK_L0_N);
        }
        intriParams.params.srcNzC0Stride = 1;
        intriParams.params.dstDnMatrixStride = self_->ctx.convTilingData->convApiTiling.orgCo * valueHoWo_;
        intriParams.dstStride = valueHoWo_;
    }

    __aicore__ inline void InnerBatchParamsHWC(FixpipeParamsC310<CO2Layout::ROW_MAJOR> &intriParams)
    {
        intriParams.mSize = currentML0_;
        intriParams.nSize = currentNL0_;
        intriParams.params.ndNum = self_->ctx.innerBatch;
        if constexpr (Intf::ConvParam::innerBatch == static_cast<int8_t>(ConvInnerBatch::KERNEL_1X1_MULTI_BATCH)) {
            intriParams.srcStride = AlignB(self_->ctx.innerBatch * currentML0_, BLOCK_L0_M);
            intriParams.params.srcNdStride = currentML0_;
        } else {
            intriParams.srcStride = self_->ctx.currentML0Align;
            intriParams.params.srcNdStride = self_->ctx.currentML0Align * CeilDiv(currentNL0_ , BLOCK_L0_N);
        }
        intriParams.params.dstNdStride = self_->ctx.convTilingData->convApiTiling.orgCo * valueHoWo_;
        intriParams.dstStride = self_->ctx.convTilingData->convApiTiling.orgCo;
    }

    __aicore__ inline void SetFixpipeIntriParamsHWC(FixpipeParamsC310<CO2Layout::ROW_MAJOR> &intriParams)
    {
        intriParams.nSize = currentNL0_;
        intriParams.mSize = currentML0_;
        intriParams.srcStride = AlignB(currentML0_, BLOCK_L0_M);
        intriParams.dstStride = self_->ctx.convTilingData->convApiTiling.orgCo;
        intriParams.params.ndNum = 1;
        intriParams.params.dstNdStride = 0;
        intriParams.params.srcNdStride = 0;
        SetBaseParams<CO2Layout::ROW_MAJOR>(intriParams);
    }

    __aicore__ inline void SetFixpipeIntriParams(FixpipeParamsC310<CO2Layout::COLUMN_MAJOR> &intriParams)
    {
        intriParams.nSize = currentNL0_;
        intriParams.mSize = currentML0_;
        intriParams.srcStride = AlignB(currentML0_, BLOCK_L0_M);
        if constexpr (Intf::formatOutput == ConvFormat::NCDHW) {
            intriParams.dstStride = self_->ctx.convTilingData->convApiTiling.orgDo * valueHoWo_;
        } else {
            intriParams.dstStride = valueHoWo_;
        }
        intriParams.params.dnNum = 1;
        intriParams.params.srcNzMatrixStride = currentNL0_ * currentML0_ / BLOCK_L0_N;
        intriParams.params.dstDnMatrixStride = valueHoWo_;
        intriParams.params.srcNzC0Stride = 1;
        SetBaseParams<CO2Layout::COLUMN_MAJOR>(intriParams);
    }

    __aicore__ inline void SetUbInfo(CopyUbInfo* ubInfo)
    {
        uint64_t unUsedNL0 = currentNL0_ >= ubInfo->nLoopIdx * ubInfo->nUb ? currentNL0_ - ubInfo->nLoopIdx * ubInfo->nUb : 0;
        uint64_t unUsedML0 = currentML0_ >= ubInfo->mLoopIdx * ubInfo->mUb ? currentML0_ - ubInfo->mLoopIdx * ubInfo->mUb : 0;
        ubInfo->realNUb = unUsedNL0 < ubInfo->nUb ? unUsedNL0 : ubInfo->nUb;
        ubInfo->realHUb = 0;
        ubInfo->realWUb = unUsedML0 < ubInfo->mUb ? unUsedML0 : ubInfo->mUb;
        if constexpr (Intf::isInnerBatchFlag) {
            uint64_t unUsedInnerbatch = self_->innerBatch >= ubInfo->batchLoopIdx * ubInfo->batchUb ? self_->innerBatch - ubInfo->batchLoopIdx * ubInfo->batchUb : 0;
            ubInfo->realBatchUb = unUsedInnerbatch < ubInfo->batchUb ? unUsedInnerbatch : ubInfo->batchUb;
        } else {
            ubInfo->realBatchUb = 1;
        }
        ubInfo->outBatchIdx = self_->ctx.batchIter + ubInfo->batchLoopIdx * ubInfo->batchUb;
        if constexpr (Intf::isKL1NL0FullLoad) {
            ubInfo->outCIdx = ubInfo->nLoopIdx * ubInfo->nUb;
        } else {
            ubInfo->outCIdx = self_->ctx.nBL1Iter * self_->ctx.convTilingData->convApiTiling.nBL1 +
                            self_->ctx.nL0Iter * self_->ctx.convTilingData->convApiTiling.nL0 + ubInfo->nLoopIdx * ubInfo->nUb;
        }
        ubInfo->outHIdx = 0;
        ubInfo->outWIdx = self_->ctx.mAL1Iter * self_->ctx.mAL1 +
                         self_->ctx.mL0Iter * self_->ctx.mL0 + ubInfo->mLoopIdx * ubInfo->mUb;
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

        if constexpr (!Intf::isInnerBatchFlag) {
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
    }

    template <CO2Layout format>
    __aicore__ inline void SetFixpipeIntriParamsUb(FixpipeParamsC310<format> &intriParams, CopyUbInfo* ubInfo = nullptr)
    {
        if (ubInfo == nullptr) {
            return;
        }

        SetUbInfo(ubInfo);

        intriParams.nSize = AlignB(ubInfo->realNUb, BLOCK_L0_N);
        intriParams.mSize = AlignB(ubInfo->realWUb, BLOCK_L0_M);
        if constexpr (format == CO2Layout::ROW_MAJOR) {
            intriParams.dstStride = AlignB(ubInfo->realNUb, BLOCK_L0_N);
            intriParams.params.dstNdStride = 0;
            if constexpr (Intf::isInnerBatchFlag) {
                intriParams.params.ndNum = ubInfo->realBatchUb;
                if constexpr (Intf::ConvParam::innerBatch == static_cast<int8_t>(ConvInnerBatch::KERNEL_1X1_MULTI_BATCH)) {
                    intriParams.srcStride = AlignB(self_->ctx.innerBatch * currentML0_, BLOCK_L0_M);
                    intriParams.params.srcNdStride = currentML0_;
                } else {
                    intriParams.srcStride = self_->ctx.currentML0Align;
                    intriParams.params.srcNdStride = self_->ctx.currentML0Align * CeilDiv(currentNL0_, BLOCK_L0_N);
                }
            } else {
                intriParams.params.ndNum = 1;
                intriParams.srcStride = self_->ctx.currentML0Align;
                intriParams.params.srcNdStride = 0;
            }
        } else if constexpr (format == CO2Layout::COLUMN_MAJOR) {
            intriParams.dstStride = AlignB(ubInfo->realWUb, BLOCK_L0_M);
            intriParams.params.srcNzC0Stride = 1;
            if constexpr (Intf::isInnerBatchFlag) {
                intriParams.params.dnNum = ubInfo->realBatchUb;
                intriParams.params.dstDnMatrixStride = AlignB(ubInfo->realWUb, BLOCK_L0_M) * AlignB(ubInfo->realNUb, BLOCK_L0_N);
                if constexpr (Intf::ConvParam::innerBatch == static_cast<int8_t>(ConvInnerBatch::KERNEL_1X1_MULTI_BATCH)) {
                    intriParams.srcStride = AlignB(self_->ctx.innerBatch * currentML0_, BLOCK_L0_N);
                    intriParams.params.srcNzMatrixStride = currentML0_;
                } else {
                    intriParams.srcStride = self_->ctx.currentML0Align;
                    intriParams.params.srcNzMatrixStride = self_->ctx.currentML0Align * CeilDiv(currentNL0_, BLOCK_L0_N);
                }
            } else {
                intriParams.srcStride = self_->ctx.currentML0Align;
                intriParams.params.dnNum = 1;
                intriParams.params.dstDnMatrixStride = 0;
                intriParams.params.srcNzMatrixStride = 0;
            }     
        }
        SetBaseParams<format>(intriParams);
    }

    __aicore__ inline uint64_t CalcFixpipeOffset()
    {
        uint64_t offset = 0;
        if constexpr (!Intf::isOneBatch) {
            offset = self_->ctx.batchIter * self_->ctx.outputOneBatchSize;
        }
        if constexpr (Intf::isInnerBatchFlag) {
            offset *= self_->ctx.convTilingData->convApiTiling.innerBatch;
        }

        uint64_t offsetCout = 0;
        if constexpr (Intf::groupOptPreloadFlag) {
            offsetCout += self_->ctx.groupOptIter * self_->ctx.convTilingData->convApiTiling.orgCo / self_->ctx.convTilingData->convApiTiling.groups *
                          self_->ctx.convTilingData->convApiTiling.enlarge;
        }
        if constexpr (!Intf::isKL1NL0FullLoad) {
            offsetCout += self_->ctx.nBL1Iter * self_->ctx.convTilingData->convApiTiling.nBL1 +
                          self_->ctx.nL0Iter * self_->ctx.convTilingData->convApiTiling.nL0;
        }
        uint64_t offsetMAL1 = self_->ctx.mAL1Iter * self_->ctx.mAL1 +
                              self_->ctx.mL0Iter * self_->ctx.mL0;
                          
        if constexpr (Intf::formatOutput == ConvFormat::NCDHW) {
            offset += offsetCout * self_->ctx.convTilingData->convApiTiling.orgDo * valueHoWo_ + self_->ctx.dOutIter * valueHoWo_ + offsetMAL1;
        } else if constexpr (Intf::formatOutput == ConvFormat::NDHWC) {
            offset += self_->ctx.dOutIter * valueHoWo_ * self_->ctx.convTilingData->convApiTiling.orgCo + offsetMAL1 * self_->ctx.convTilingData->convApiTiling.orgCo + offsetCout;
        } else if constexpr (Intf::formatOutput == ConvFormat::NCHW) {
            offset += offsetCout * valueHoWo_ + offsetMAL1;
        } else {
            offset += offsetMAL1 * self_->ctx.convTilingData->convApiTiling.orgCo + offsetCout;
        }

        return offset;
    }

    template <template <typename> class TensorTypeT, const FixpipeConfig &config>
    __aicore__ inline void ExtendConv2DFixpipe(const TensorTypeT<OutputT> &output, 
        FixpipeParamsC310<config.format> &intriParams, uint64_t offset)
    {
        if (!self_->ctx.enableVectorQuant) {
            Fixpipe<OutputT, typename Intf::L0cT, config>(output[offset], self_->ctx.cl0, intriParams);
        } else {
            if constexpr (FixpipeIdx == 0) {
                Fixpipe<OutputT, typename Intf::L0cT, config>(
                    output[offset], self_->ctx.cl0, self_->ctx.scaleL1[GetExtendConv2dScaleL1Addr()], intriParams);
            } else {
                Fixpipe<OutputT, typename Intf::L0cT, config>(
                    output[offset], self_->ctx.cl0,
                    self_->ctx.scaleL1[GetExtendConv2dScaleL1Addr() + self_->ctx.scale1L1offset], intriParams);
            }
        }
    }

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

private:
    Intf *self_ = nullptr;
    uint64_t valueHoWo_ = 0;
    uint64_t currentML0_ = 0;
    uint64_t currentNL0_ = 0;
};

};

#endif // CONV_INSTR_M_MODE_IMPL_H