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
        alignCinATailInCore_ = AlignB(self_->ctx.convTiling->cinATailInCore, Intf::k0FmapTail);

        if constexpr (Intf::c04Flag) {
            channelSize_ = conv::C04_CIN_SIZE;
            c04KStepTail = (channelSize_ * self_->ctx.convTiling->kernelHxkernelW) % self_->ctx.convTiling->kL0;
            c04KStepTail = c04KStepTail == 0 ? self_->ctx.convTiling->kL0 : c04KStepTail;
        } else {
            if constexpr (Intf::kPreLoadFlag) {
                channelSize_ = (self_->ctx.kAL1Iter - 1 != self_->ctx.maxKAL1Iter ?
                                self_->ctx.convTiling->cinAInCore : alignCinATailInCore_);
            } else {
                channelSize_ = (self_->ctx.kAL1Iter != self_->ctx.maxKAL1Iter ?
                                self_->ctx.convTiling->cinAInCore : alignCinATailInCore_);
            }
        }

        if constexpr (Intf::ConvParam::innerBatch == static_cast<int8_t>(ConvInnerBatch::MULTI_BATCH)) {
            uint64_t hiLoadL1 = (self_->ctx.convTiling->orgHo - 1) * self_->ctx.convTiling->strideH +
                                self_->ctx.dilatedKernelH;
            hiLoadL1 = hiLoadL1 > self_->ctx.convTiling->orgHi ? self_->ctx.convTiling->orgHi : hiLoadL1;
            realHixWi = hiLoadL1 * self_->ctx.convTiling->orgWi;
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
        LoadDataRepeatParam repeatParams = {0, 1, 0, static_cast<uint16_t>(currentML0Align_ / BLOCK_L0_M)};
        SetLoadDataRepeat(repeatParams);
    }

    __aicore__ inline void LoadAL0(bool isFirst = true)
    {
        if ASCEND_IS_AIV {
            return;
        }
        if constexpr (Intf::c04Flag) {
            currentKL0 = IsKL0Tail() ? c04KStepTail : self_->ctx.convTiling->kL0;
        } else if constexpr (Intf::k0 == Intf::k0FmapTail) {
            currentKL0 = IsKL0Tail() ? self_->ctx.kL0Tail : self_->ctx.convTiling->kL0;
        } else {
            currentKL0 = IsKL0Tail() ? self_->ctx.kAL0Tail : self_->ctx.convTiling->kL0;
        }

        uint64_t posK = self_->ctx.kAL0Iter * self_->ctx.convTiling->kL0;
        if (isFirst) {
            uint64_t posM = self_->ctx.mL0Iter * self_->ctx.mL0 +
                            (self_->ctx.mStartPos + self_->ctx.mAL1Iter * self_->ctx.mAL1) % self_->ctx.orgWo;
            xmtmp_ = ((currentML0_ & MASK_16) << MSTEP_OFFSET) | ((posM & MASK_16) << POSM_OFFSET);
            xt_ = ((static_cast<uint64_t>(self_->ctx.convTiling->strideW) & MASK_6) << 0) |
                  ((static_cast<uint64_t>(self_->ctx.convTiling->strideH) & MASK_6) << STRIDEH_OFFSET) |
                  ((static_cast<uint64_t>(self_->ctx.kernelW) & MASK_8) << KERNELW_OFFSET) |
                  ((static_cast<uint64_t>(self_->ctx.kernelW) & NINTH_BIT_MASK) << KERNELW_HIGHEST_BIT_OFFSET) |
                  ((static_cast<uint64_t>(self_->ctx.kernelH) & MASK_8) << KERNELH_OFFSET) |
                  ((static_cast<uint64_t>(self_->ctx.kernelH) & NINTH_BIT_MASK) << KERNELH_HIGHEST_BIT_OFFSET) |
                  ((static_cast<uint64_t>(self_->ctx.convTiling->dilationW) & MASK_8) << DILATIONW_OFFSET) |
                  ((static_cast<uint64_t>(self_->ctx.convTiling->dilationH) & MASK_8) << DILATIONH_OFFSET) |
                  ((static_cast<uint64_t>(channelSize_) & MASK_16) << CIN_OFFSET);
            param_.SetConfig1(xt_);
        }
        xm_ = currentKL0 | (posK << POSK_OFFSET) | xmtmp_;
        param_.SetConfig0(xm_);

        if constexpr (Intf::ConvParam::innerBatch == static_cast<int8_t>(ConvInnerBatch::MULTI_BATCH)) {
            uint32_t srcOffset = 0;
            uint32_t dstOffset = 0;
            uint32_t srcBatchStride = 0;
            if constexpr (Intf::c04Flag) {
                srcBatchStride = AlignB(C04_CIN_SIZE * self_->ctx.convTiling->orgHixWi, Intf::k0);
            } else {
                srcBatchStride = ((self_->ctx.kIter / self_->ctx.multiKAL1) != self_->ctx.maxKAL1Iter ?
                                    self_->ctx.convTiling->cinAInCore : alignCinATailInCore_) * realHixWi;
            }           
            uint32_t dstBatchStride = currentML0Align_ * self_->ctx.convTiling->kL0;
            for (uint16_t batchIter = 0; batchIter < self_->ctx.innerBatch; batchIter++) {
                LoadData<TPosition::A2, TPosition::A1, typename Intf::FmapT>(self_->ctx.al0[dstOffset],
                                                                            self_->ctx.al1[srcOffset], param_);
                srcOffset += srcBatchStride;
                dstOffset += dstBatchStride;
            }
        } else {
            LoadData<TPosition::A2, TPosition::A1, typename Intf::FmapT>(self_->ctx.al0, self_->ctx.al1, param_);
        }
    };

private:
    __aicore__ inline bool IsKL0Tail()
    {
        return self_->ctx.kIter == self_->ctx.maxKL0Iter;
    }

private:
    Intf *self_ = nullptr;
    uint64_t currentML0_ = 0;
    uint64_t currentML0Align_ = 0;
    uint64_t xm_ = 0;
    uint64_t xt_ = 0;
    uint64_t xmtmp_ = 0;
    uint64_t alignCinATailInCore_ = 0;
    uint16_t channelSize_ = 0;
    uint64_t c04KStepTail = 0;
    uint64_t realHixWi = 0;
    uint64_t currentKL0 = 0;
    Load3DBitModeParam param_;
};

template <class Intf, typename OutputT, uint64_t FixpipeIdx = 0>
class CopyOutToolsMMode {
public:
    __aicore__ inline CopyOutToolsMMode()
    {}

    __aicore__ inline void SetParams(Intf *self)
    {
        self_ = self;
        valueHoWo_ = self_->ctx.orgHo * self_->ctx.orgWo;
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
                if (self_->ctx.convTiling->hasScale != 0) {
                    Fixpipe<OutputT, typename Intf::L0cT, config>(
                        output[offset], self_->ctx.cl0, self_->ctx.scaleL1[GetScaleL1Addr()], intriParams);
                } else {
                    Fixpipe<OutputT, typename Intf::L0cT, config>(output[offset], self_->ctx.cl0, intriParams);
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
            if (self_->ctx.convTiling->hasScale != 0) {
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
        intriParams.params.dstDnMatrixStride = self_->ctx.orgCo * valueHoWo_;
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
        intriParams.params.dstNdStride = self_->ctx.orgCo * valueHoWo_;
        intriParams.dstStride = self_->ctx.orgCo;
    }

    __aicore__ inline void SetFixpipeIntriParamsHWC(FixpipeParamsC310<CO2Layout::ROW_MAJOR> &intriParams)
    {
        intriParams.nSize = currentNL0_;
        intriParams.mSize = currentML0_;
        intriParams.srcStride = AlignB(currentML0_, BLOCK_L0_M);
        intriParams.dstStride = self_->ctx.orgCo;
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
            intriParams.dstStride = self_->ctx.orgDo * valueHoWo_;
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
        ubInfo->outCIdx = self_->ctx.nBL1Iter * self_->ctx.convTiling->nBL1 +
                         self_->ctx.nL0Iter * self_->ctx.convTiling->nL0 + ubInfo->nLoopIdx * ubInfo->nUb;
        ubInfo->outHIdx = 0;
        ubInfo->outWIdx = self_->ctx.mAL1Iter * self_->ctx.mAL1 +
                         self_->ctx.mL0Iter * self_->ctx.mL0 + ubInfo->mLoopIdx * ubInfo->mUb;
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

    template <CO2Layout format>
    __aicore__ inline void SetBaseParams(FixpipeParamsC310<format> &intriParams)
    {
        intriParams.quantPre = GetQuantPre<Intf, OutputT, FixpipeIdx>(self_);
        if (self_->ctx.convTiling->hasScale == 0) {
            intriParams.deqScalar = DEQ_SCALAR_ONE;
        }
        if constexpr (Intf::isExtendConv2d) {
            if constexpr (FixpipeIdx == 0) {
                intriParams.reluEn = self_->ctx.convTiling->reluMode0 != 0;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
                intriParams.preReluMode = static_cast<ReluMode>(self_->ctx.convTiling->reluMode0);
                if (self_->ctx.convTiling->reluMode0 == static_cast<uint8_t>(ReluMode::SCALAR_RELU)) {
                    intriParams.reluScalar = self_->ctx.preReluScalar0;
                }
#endif
                intriParams.deqScalar = self_->ctx.deqScalar0;
            } else {
                intriParams.reluEn = self_->ctx.convTiling->reluMode1 != 0;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
                intriParams.preReluMode = static_cast<ReluMode>(self_->ctx.convTiling->reluMode1);
                if (self_->ctx.convTiling->reluMode1 == static_cast<uint8_t>(ReluMode::SCALAR_RELU)) {
                    intriParams.reluScalar = self_->ctx.preReluScalar1;
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
                    if (self_->ctx.convTiling->dualOutput) {
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

    __aicore__ inline uint64_t CalcFixpipeOffset()
    {
        uint64_t offset = self_->ctx.batchIter * self_->ctx.outputOneBatchSize;
        if constexpr (Intf::isInnerBatchFlag) {
            offset *= self_->ctx.convTiling->innerBatch;
        }
        uint64_t offsetCout = self_->ctx.nBL1Iter * self_->ctx.convTiling->nBL1 +
                              self_->ctx.nL0Iter * self_->ctx.convTiling->nL0;
        uint64_t offsetMAL1 = self_->ctx.mAL1Iter * self_->ctx.mAL1 +
                              self_->ctx.mL0Iter * self_->ctx.mL0;
                          
        if constexpr (Intf::formatOutput == ConvFormat::NCDHW) {
            offset += offsetCout * self_->ctx.orgDo * valueHoWo_ + self_->ctx.dOutIter * valueHoWo_ + offsetMAL1;
        } else if constexpr (Intf::formatOutput == ConvFormat::NDHWC) {
            offset += self_->ctx.dOutIter * valueHoWo_ * self_->ctx.orgCo + offsetMAL1 * self_->ctx.orgCo + offsetCout;
        } else if constexpr (Intf::formatOutput == ConvFormat::NCHW) {
            offset += offsetCout * valueHoWo_ + offsetMAL1;
        } else {
            offset += offsetMAL1 * self_->ctx.orgCo + offsetCout;
        }

        return offset;
    }

    __aicore__ inline uint64_t GetScaleL1Addr()
    {
        if constexpr (Intf::isQuantScene) {
            if (self_->ctx.convTiling->hasScale != 0 && self_->ctx.convTiling->fixpParamsFullLoadFlag) {
                return self_->ctx.nBL1Iter * self_->ctx.convTiling->nBL1 +
                       self_->ctx.nL0Iter * self_->ctx.convTiling->nL0;
            }
        }

        return 0;
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
        if (self_->ctx.convTiling->fixpParamsFullLoadFlag) {
            return self_->ctx.nBL1Iter * self_->ctx.convTiling->nBL1 +
                self_->ctx.nL0Iter * self_->ctx.convTiling->nL0;
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