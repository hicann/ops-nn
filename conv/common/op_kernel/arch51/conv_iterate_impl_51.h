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
 * \file conv_iterate_impl_51.h
 * \brief
 */

#ifndef CONV_ITERATE_IMPL_5162_H
#define CONV_ITERATE_IMPL_5162_H

#include "conv_iterate_base_impl_51.h"
#include "../arch35/conv_iterate_impl_declare.h"
#include "../arch35/conv_iterate_hw_mode_impl.h"
#include "../arch35/conv_instr_impl_declare.h"
#include "../arch35/conv_instr_impl.h"

constexpr uint8_t SHIFT_VAL_DEFAULT = 42;

namespace ConvFunc {
using namespace AscendC;
using namespace conv;

template <class Intf, uint32_t ImplType>
template <bool sync>
__aicore__ inline bool Iterate<Intf, ImplType>::call(Intf* self, IterateParam& iterParam, bool enPartialSum,
                                                     bool* updateL1Flag)
{
    if ASCEND_IS_AIC_CONV {
        return IterateImplHWMode(self, iterParam, enPartialSum);
    }
}

template <class Intf, uint32_t ImplType>
__aicore__ inline void Iterate<Intf, ImplType>::InitIterParam(Intf* self, IterateParam& iterParam)
{
    // init base value
    uint64_t singleCoreCi = self->ctx.convTilingData->singleCoreCi;
    uint64_t singleCoreCo = self->ctx.singleCoreCo;
    uint64_t singleCoreHo = self->ctx.convTilingData->singleCoreHo;
    uint64_t singleCoreWo = self->ctx.convTilingData->singleCoreWo;
    uint64_t totalKAlignK0 = AlignB(singleCoreCi, Intf::k0) * self->ctx.convTilingData->kernelHxkernelW;
    iterParam.singleCoreCi = singleCoreCi;
    iterParam.singleCoreWo = singleCoreWo;
    iterParam.maxKAL1Iter = CeilDiv(totalKAlignK0, self->ctx.convTilingData->kAL1) - 1;
    iterParam.maxKBL1Iter = CeilDiv(totalKAlignK0, self->ctx.convTilingData->kBL1) - 1;
    iterParam.ddr2l0LoopK = CeilDiv(totalKAlignK0, self->ctx.convTilingData->kL0);
    iterParam.maxKL0Iter = self->ctx.ddr2l0LoopK - 1;
    iterParam.kAL1Tail = (singleCoreCi * self->ctx.convTilingData->kernelHxkernelW) % self->ctx.convTilingData->kAL1;
    iterParam.kAL1Tail = iterParam.kAL1Tail == 0 ? self->ctx.convTilingData->kAL1 : iterParam.kAL1Tail;
    iterParam.kBL1Tail = (singleCoreCi * self->ctx.convTilingData->kernelHxkernelW) % self->ctx.convTilingData->kBL1;
    iterParam.kBL1Tail = iterParam.kBL1Tail == 0 ? self->ctx.convTilingData->kBL1 : iterParam.kBL1Tail;
    iterParam.kL0Tail = totalKAlignK0 % self->ctx.convTilingData->kL0;
    iterParam.kL0Tail = iterParam.kL0Tail == 0 ? self->ctx.convTilingData->kL0 : iterParam.kL0Tail;
    iterParam.multiKAL1 = CeilDiv(self->ctx.convTilingData->kAL1, self->ctx.convTilingData->kL0);
    iterParam.multiKBL1 = CeilDiv(self->ctx.convTilingData->kBL1, self->ctx.convTilingData->kL0);
    if constexpr (Intf::k0 != Intf::k0FmapTail) {
        iterParam.kAL0Tail = AlignB(singleCoreCi, Intf::k0FmapTail) * self->ctx.convTilingData->kernelHxkernelW %
                             self->ctx.convTilingData->kL0;
        iterParam.kAL0Tail = iterParam.kAL0Tail == 0 ? self->ctx.convTilingData->kL0 : iterParam.kAL0Tail;
        // update A16W8 loopk iter num
        iterParam.ddr2l0LoopK = CeilDiv(
            AlignB(singleCoreCi, Intf::k0FmapTail) * self->ctx.convTilingData->kernelHxkernelW,
            self->ctx.convTilingData->kL0);
        iterParam.maxKL0Iter = iterParam.ddr2l0LoopK - 1;
    }
    if constexpr (Intf::kPreLoadFlag || Intf::WEIGHT_NZ_FLAG) {
        iterParam.kBL1AlignK0Tail = totalKAlignK0 % self->ctx.convTilingData->kBL1;
        iterParam.kBL1AlignK0Tail = iterParam.kBL1AlignK0Tail == 0 ? self->ctx.convTilingData->kBL1 :
                                                                     iterParam.kBL1AlignK0Tail;
    }

    iterParam.maxKhIter = self->ctx.convTilingData->khConv == 0 ?
                              0 :
                              CeilDiv(self->ctx.convTilingData->kernelH, self->ctx.convTilingData->khConv) - 1;
    iterParam.khConvTail = self->ctx.convTilingData->kernelH - self->ctx.convTilingData->khConv * iterParam.maxKhIter;

    // init ho params
    iterParam.hoAL1Tail = singleCoreHo % self->ctx.convTilingData->hoL1;
    iterParam.hoAL1Tail = iterParam.hoAL1Tail == 0 ? self->ctx.convTilingData->hoL1 : iterParam.hoAL1Tail;
    iterParam.currentHoL1 = self->ctx.hoAL1Tail;
    iterParam.hoL0Tail = iterParam.currentHoL1 % self->ctx.convTilingData->hoL0;
    iterParam.hoL0Tail = iterParam.hoL0Tail == 0 ? self->ctx.convTilingData->hoL0 : iterParam.hoL0Tail;
    iterParam.currentHoL0 = iterParam.hoL0Tail;
    if constexpr (!Intf::hasHL1IterFlag) {
        iterParam.ddr2l1LoopH = 1;
        iterParam.maxHoL1Iter = 0;
    } else {
        iterParam.ddr2l1LoopH = CeilDiv(singleCoreHo, self->ctx.convTilingData->hoL1);
        iterParam.maxHoL1Iter = iterParam.ddr2l1LoopH - 1;
    }

    // init wo params
    iterParam.woAL1Tail = singleCoreWo % self->ctx.convTilingData->woL1;
    iterParam.woAL1Tail = iterParam.woAL1Tail == 0 ? self->ctx.convTilingData->woL1 : iterParam.woAL1Tail;
    iterParam.currentWoL1 = iterParam.woAL1Tail;
    iterParam.maxWoL1Iter = 0;
    iterParam.woL0Tail = iterParam.currentWoL1 % self->ctx.convTilingData->woL0;
    iterParam.woL0Tail = iterParam.woL0Tail == 0 ? self->ctx.convTilingData->woL0 : iterParam.woL0Tail;
    iterParam.currentWoL0 = iterParam.woL0Tail;
    iterParam.maxWoL0Iter = 0;
    iterParam.l12l0LoopW = 1;
    if constexpr (Intf::hasWL1IterFlag) {
        iterParam.maxWoL1Iter = CeilDiv(singleCoreWo, self->ctx.convTilingData->woL1) - 1;
    }

    if constexpr (!Intf::isDmaFlag) {
        if (Intf::hasWL0IterFlag && self->ctx.convTilingData->hoL0 > 1 && self->ctx.woAL1Tail % BLOCK_L0_N > 0 &&
            CeilDiv(iterParam.woAL1Tail, self->ctx.convTilingData->woL0) > 1) {
            iterParam.woL1SmallTail = iterParam.woAL1Tail % self->ctx.convTilingData->woL0;
            iterParam.woAL1Tail = (iterParam.woAL1Tail / self->ctx.convTilingData->woL0) *
                                  self->ctx.convTilingData->woL0;
        }
        if (iterParam.woL1SmallTail > 0) {
            iterParam.maxWoL1Iter += 1;
        }
    }
    iterParam.ddr2l1LoopW = iterParam.maxWoL1Iter + 1;

    // init co params
    iterParam.nBL1Tail = singleCoreCo % self->ctx.convTilingData->nBL1;
    iterParam.nBL1Tail = iterParam.nBL1Tail == 0 ? self->ctx.convTilingData->nBL1 : iterParam.nBL1Tail;
    if constexpr (Intf::WEIGHT_NZ_FLAG) {
        iterParam.nBL1Tail = AlignB(iterParam.nBL1Tail, BLOCK_L0_N);
    }
    iterParam.nL0Tail = iterParam.nBL1Tail % self->ctx.convTilingData->nL0;
    iterParam.nL0Tail = iterParam.nL0Tail == 0 ? self->ctx.convTilingData->nL0 : iterParam.nL0Tail;
    if constexpr (!Intf::hasNL1IterFlag) {
        iterParam.maxNBL1Iter = 0;
        iterParam.ddr2l1LoopN = 1;
    } else {
        iterParam.ddr2l1LoopN = CeilDiv(singleCoreCo, self->ctx.convTilingData->nBL1);
        iterParam.maxNBL1Iter = iterParam.ddr2l1LoopN - 1;
    }
    if constexpr (!Intf::hasNL0IterFlag) {
        iterParam.l12l0LoopN = 1;
        iterParam.maxNL0Iter = 0;
    }
    iterParam.currentNBL1 = iterParam.nBL1Tail;

    // init other params
    iterParam.ddr2l1LoopBatch = self->ctx.singleCoreBatch;
    iterParam.hiStartPos = self->ctx.hiStartPos;
    iterParam.wiStartPos = self->ctx.wiStartPos;
    iterParam.dilatedKernelH = 1 + (self->ctx.convTilingData->kernelH - 1) * self->ctx.convTilingData->dilationH;
    iterParam.dilatedKernelW = 1 + (self->ctx.convTilingData->kernelW - 1) * self->ctx.convTilingData->dilationW;
    iterParam.currentL1Hin = 0;
    iterParam.maxAL1Iter = iterParam.ddr2l1LoopH * iterParam.ddr2l1LoopW * (iterParam.maxKAL1Iter + 1);
    iterParam.cubeWaitMte1Flag = self->ctx.enableBias || self->ctx.enableVectorQuant ||
                                 self->ctx.convTilingData->reluMode0 == static_cast<uint8_t>(ReluMode::VECTOR_RELU);
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::ReduceOneK(Intf* self, IterateParam& iterParam)
{
    if (unlikely(self->ctx.loadAL1Flag)) {
        LoadAL1Module<Intf>(self, iterParam);
        if constexpr (!Intf::preFusionFlag) {
            self->ctx.al1 = self->ctx.queueAL1.template DeQue<typename Intf::FmapT>();
        }
    }
    if (Intf::convToL1Mode != static_cast<int8_t>(ConvToL1::OPEN_WEIGHT_BY_PASS)) {
        if (unlikely(self->ctx.loadBL1Flag)) {
            LoadBL1Module<Intf>(self, iterParam);
            self->ctx.bl1 = self->ctx.queueBL1.template DeQue<typename Intf::WeightT>();
            if constexpr (Intf::isL0BFullLoadable) {
                self->ctx.loadBL0Ins.FullLoadBL0(self->ctx.wholeBl0Tensor[0]);
            }
        }
    }

    uint64_t currentKL0;
    if constexpr (Intf::k0 == Intf::k0FmapTail) {
        currentKL0 = iterParam.kL0Tail;
    } else {
        currentKL0 = iterParam.kAL0Tail;
    }
    uint64_t kStep = iterParam.kL0Tail / Intf::k0;

    ConvKernelBitModeParams kernelParams;
    ConvFMatrixStepPosBitModeParams convFMatrixParams;
    ConvStrideBitModeParams strideParams;
    SetParamsFor5162(self, kernelParams, convFMatrixParams, strideParams, iterParam);
    uint16_t hStartPosIter = uint16_t(iterParam.hoL0Iter * self->ctx.convTilingData->hoL0 *
                                      self->ctx.convTilingData->strideH);
    uint16_t processCin = self->ctx.convTilingData->kL0 / self->ctx.convTilingData->kernelHxkernelW;
    if constexpr (Intf::groupOptNZFlag) {
        uint16_t modValueOpt = (self->ctx.convTilingData->orgCi / self->ctx.convTilingData->groups *
                                self->ctx.convTilingData->enlarge) %
                               processCin;
        if (modValueOpt > 0) {
            processCin = modValueOpt;
        }
    } else {
        uint16_t modValue = self->ctx.convTilingData->orgCi % processCin;
        if (modValue > 0) {
            processCin = modValue;
        }
    }

    ConvFixBitModeParams convFixParams;
    if (self->ctx.convTilingData->hasBias) {
        convFixParams.SetBiasMatrixInitVal(false);
        convFixParams.SetBiasMatrixBroadcast(true);
    } else {
        convFixParams.SetBiasMatrixInitVal(true);
        convFixParams.SetBiasMatrixBroadcast(false);
    }
    convFixParams.SetCin(processCin);
    ProcessL0ModuleFor5162(self, iterParam, 0, kStep, hStartPosIter, kernelParams, convFMatrixParams, strideParams,
                           convFixParams);
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::ReduceK(Intf* self, IterateParam& iterParam)
{
    iterParam.kIter = 0;
    const bool isConvToL1WeightByPass = Intf::convToL1Mode == static_cast<int8_t>(ConvToL1::OPEN_WEIGHT_BY_PASS);

    if (self->ctx.loadAL1Flag || (!self->ctx.kAL1fullload)) {
        if (iterParam.al1PingPongFlag > 1) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(iterParam.al1PingPongFlag & 0x1);
        }
        LoadAL1Module<Intf>(self, iterParam);
        self->ctx.al1 = self->ctx.queueAL1.template DeQue<typename Intf::FmapT>();
    }
    if constexpr (!isConvToL1WeightByPass) {
        if (self->ctx.loadBL1Flag || (!self->ctx.kBL1fullload)) {
            LoadBL1Module<Intf>(self, iterParam);
            self->ctx.bl1 = self->ctx.queueBL1.template DeQue<typename Intf::WeightT>();
            self->ctx.loadBL0Flag = true;
            if constexpr (Intf::isL0BFullLoadable) {
                self->ctx.loadBL0Ins.FullLoadBL0(self->ctx.wholeBl0Tensor[0]);
            }
        }
    }

    uint64_t hoL0xStrideH = self->ctx.convTilingData->hoL0 * self->ctx.convTilingData->strideH;
    uint16_t hStartPosIter = static_cast<uint16_t>(iterParam.hoL0Iter * hoL0xStrideH);
    uint64_t kStartPosition = 0;
    uint64_t kStep = self->ctx.convTilingData->kStep;
    uint64_t& multiKAL1 = iterParam.multiKAL1;
    uint64_t& multiKBL1 = iterParam.multiKBL1;
    uint64_t loadAkIter = self->ctx.kAL1fullload ? iterParam.ddr2l0LoopK : multiKAL1;
    uint64_t loadBkIter = self->ctx.kBL1fullload ? iterParam.ddr2l0LoopK : multiKBL1;
    const uint64_t maxKInterLoop = iterParam.ddr2l0LoopK - 1;

    ConvKernelBitModeParams kernelParams;
    ConvFMatrixStepPosBitModeParams convFMatrixParams;
    ConvStrideBitModeParams strideParams;
    SetParamsFor5162(self, kernelParams, convFMatrixParams, strideParams, iterParam);
    ConvFixBitModeParams convFixParams;
    if (self->ctx.convTilingData->hasBias) {
        convFixParams.SetBiasMatrixInitVal(false);
        convFixParams.SetBiasMatrixBroadcast(true);
    } else {
        convFixParams.SetBiasMatrixInitVal(true);
        convFixParams.SetBiasMatrixBroadcast(false);
    }
    convFixParams.SetDependEnable(false);
    if (iterParam.kIter < maxKInterLoop) {
        uint16_t processCin = self->ctx.convTilingData->kL0 / self->ctx.convTilingData->kernelHxkernelW;
        convFixParams.SetCin(processCin);
        ProcessL0ModuleFor5162(self, iterParam, kStartPosition, kStep, hStartPosIter, kernelParams, convFMatrixParams,
                               strideParams, convFixParams);
        iterParam.kIter++;
        kStartPosition += kStep;
        if (iterParam.kIter == loadAkIter) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(iterParam.al1PingPongFlag & 0x1);
            iterParam.al1PingPongFlag++;
        }
    }

    while (iterParam.kIter < maxKInterLoop) {
        if (iterParam.kIter == loadAkIter) {
            self->ctx.queueAL1.FreeTensor(self->ctx.al1);
            if (iterParam.al1PingPongFlag > 1) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(iterParam.al1PingPongFlag & 0x1);
            }
            LoadAL1Module<Intf>(self, iterParam);
            self->ctx.al1 = self->ctx.queueAL1.template DeQue<typename Intf::FmapT>();
            loadAkIter += multiKAL1;
        }
        if (iterParam.kIter == loadBkIter) {
            if constexpr (!isConvToL1WeightByPass) {
                self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
                LoadBL1Module<Intf>(self, iterParam);
                self->ctx.bl1 = self->ctx.queueBL1.template DeQue<typename Intf::WeightT>();
                self->ctx.loadBL0Flag = true;
            }
            loadBkIter += multiKBL1;
            if constexpr (Intf::isL0BFullLoadable) {
                self->ctx.loadBL0Ins.FullLoadBL0(self->ctx.wholeBl0Tensor[0]);
            }
            kStartPosition = 0;
        }
        ProcessL0ModuleFor5162(self, iterParam, kStartPosition, kStep, hStartPosIter, kernelParams, convFMatrixParams,
                               strideParams, convFixParams);
        iterParam.kIter++;
        kStartPosition += kStep;
        if (iterParam.kIter == loadAkIter) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(iterParam.al1PingPongFlag & 0x1);
            iterParam.al1PingPongFlag++;
        }
    }

    uint64_t currentKL0 = self->ctx.convTilingData->kL0;
    if constexpr (Intf::k0 == Intf::k0FmapTail) {
        currentKL0 = iterParam.kL0Tail;
    } else {
        currentKL0 = iterParam.kAL0Tail;
    }
    kStartPosition = (iterParam.kIter % multiKBL1) * kStep;
    kStep = iterParam.kL0Tail / Intf::k0;

    if (iterParam.kIter == loadAkIter) {
        self->ctx.queueAL1.FreeTensor(self->ctx.al1);
        if (iterParam.al1PingPongFlag > 1) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(iterParam.al1PingPongFlag & 0x1);
        }
        LoadAL1Module<Intf>(self, iterParam);
        self->ctx.al1 = self->ctx.queueAL1.template DeQue<typename Intf::FmapT>();
    }
    if (iterParam.kIter == loadBkIter) {
        if constexpr (!isConvToL1WeightByPass) {
            self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
            LoadBL1Module<Intf>(self, iterParam);
            self->ctx.bl1 = self->ctx.queueBL1.template DeQue<typename Intf::WeightT>();
        }
        self->ctx.loadBL0Flag = true;
        if constexpr (Intf::isL0BFullLoadable) {
            self->ctx.loadBL0Ins.FullLoadBL0(self->ctx.wholeBl0Tensor[0]);
        }
        kStartPosition = 0;
    }

    uint16_t processCinTail = 0;
    if constexpr (Intf::k0 == Intf::k0FmapTail) {
        processCinTail = static_cast<uint16_t>(iterParam.kL0Tail / self->ctx.convTilingData->kernelHxkernelW);
    } else {
        processCinTail = static_cast<uint16_t>(iterParam.kAL0Tail / self->ctx.convTilingData->kernelHxkernelW);
    }
    if constexpr (Intf::groupOptNZFlag) {
        uint16_t modValueOpt = (self->ctx.convTilingData->orgCi / self->ctx.convTilingData->groups *
                                self->ctx.convTilingData->enlarge) %
                               processCinTail;
        if (modValueOpt > 0) {
            processCinTail = modValueOpt;
        }
    } else {
        uint16_t modValue = self->ctx.convTilingData->orgCi % processCinTail;
        if (modValue > 0) {
            processCinTail = modValue;
        }
    }
    convFixParams.SetCin(processCinTail);
    ProcessL0ModuleFor5162(self, iterParam, kStartPosition, kStep, hStartPosIter, kernelParams, convFMatrixParams,
                           strideParams, convFixParams);
    if (iterParam.kIter == loadAkIter) {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(iterParam.al1PingPongFlag & 0x1);
        iterParam.al1PingPongFlag++;
    }
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::IterateK(Intf* self, IterateParam& iterParam)
{
    SetMNBeforeIterateK<Intf>(self, iterParam);
    SetFirstBeforeIterateK<Intf>(self, iterParam);
    if constexpr (!Intf::isDeQuantFlag) {
        IterateBiasScale(self, iterParam);
    }

    if constexpr (Intf::kl0FullLoadFlag) {
        ReduceOneK(self, iterParam);
    } else {
        ReduceK(self, iterParam);
    }

    FreeL1Tensor<Intf>(self);
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::SetParamsFor5162(Intf* self, ConvKernelBitModeParams& kernelParams,
                                                          ConvFMatrixStepPosBitModeParams& convFMatrixParams,
                                                          ConvStrideBitModeParams& strideParams,
                                                          IterateParam& iterParam)
{
    kernelParams.SetWStride(self->ctx.convTilingData->strideW);
    kernelParams.SetHStride(self->ctx.convTilingData->strideH);
    kernelParams.SetWKernel(self->ctx.convTilingData->kernelW);
    kernelParams.SetHKernel(self->ctx.convTilingData->kernelH);
    kernelParams.SetWDilation(self->ctx.convTilingData->dilationW);
    kernelParams.SetHDilation(self->ctx.convTilingData->dilationH);
    uint32_t kernelNum = (iterParam.nL0Iter == iterParam.maxNL0Iter && iterParam.nBL1Iter == iterParam.maxNBL1Iter) ?
                             iterParam.nL0Tail :
                             self->ctx.convTilingData->nL0;
    kernelParams.SetKernelNum(kernelNum);
    SetConvKernel(kernelParams);

    uint16_t hStep = static_cast<uint16_t>((iterParam.currentHoL0 - 1) * self->ctx.convTilingData->strideH + 1);
    uint16_t wStep = static_cast<uint16_t>((iterParam.currentWoL0 - 1) * self->ctx.convTilingData->strideW + 1);
    uint16_t hStartPos = static_cast<uint16_t>(iterParam.hoL0Iter * self->ctx.convTilingData->hoL0 *
                                               self->ctx.convTilingData->strideH);
    uint16_t wStartPos = static_cast<uint16_t>(iterParam.woL0Iter * self->ctx.convTilingData->woL0 *
                                               self->ctx.convTilingData->strideW);
    convFMatrixParams.SetWStep(wStep);
    convFMatrixParams.SetHStep(hStep);
    convFMatrixParams.SetWStartPos(wStartPos);
    convFMatrixParams.SetHStartPos(hStartPos);
    SetConvFMatrixStepPos(convFMatrixParams);

    constexpr uint16_t strideFactor = C0_SIZE / BLOCK_L0_N;
    strideParams.SetLoop3DstStride(iterParam.currentWoL0 * strideFactor);
    strideParams.SetLoop4DstStride(iterParam.currentHoL0 * iterParam.currentWoL0 * strideFactor);
    SetConvStridePara(strideParams);
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::SetPreQuantAndReluMode(Intf* self, ConvFixBitModeParams& convFixParams)
{
    auto quantPre = ConvFunc::GetQuantPre<Intf, typename Intf::OutputT, 0>(self);
    pre_quant_t preQuantMode = MapQuantModeToPreQuant(quantPre);
    auto reluMode = static_cast<ReluMode_t>(self->ctx.convTilingData->reluMode0);
    convFixParams.SetPreQuantMode(preQuantMode);
    convFixParams.SetPreReluMode(reluMode);
    if (self->ctx.convTilingData->clipMode0 == static_cast<uint8_t>(ClipMode::SCALARCLIPRELU)) {
        convFixParams.SetPreReluMode(ReluMode_t::NormalRelu);
        convFixParams.SetPreClipReluMode(true);
    }
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::ProcessL0ModuleFor5162(
    Intf* self, IterateParam& iterParam, const uint64_t kStartPosition, const uint64_t kStep, uint16_t hStartPosIter,
    ConvKernelBitModeParams& kernelParams, ConvFMatrixStepPosBitModeParams& convFMatrixParams,
    ConvStrideBitModeParams& strideParams, ConvFixBitModeParams& convFixParams)

{
    uint64_t khStartPos = kStartPosition;
    uint64_t khStep = kStep;
    uint64_t currentKh = 1;
    for (uint64_t khIdx = 0; khIdx <= iterParam.maxKhIter; khIdx++) {
        iterParam.khIter = khIdx;
        if constexpr (Intf::convToL1Mode == static_cast<int8_t>(ConvToL1::OPEN_WEIGHT_L1_WITH_CUT_KERNEL)) {
            currentKh = khIdx == iterParam.maxKhIter ? iterParam.khConvTail : self->ctx.convTilingData->khConv;
            kernelParams.SetHKernel(currentKh);
            SetConvKernel(kernelParams);
            khStep = currentKh * self->ctx.convTilingData->kernelW;
            convFMatrixParams.SetHStartPos(hStartPosIter);
            SetConvFMatrixStepPos(convFMatrixParams);
            hStartPosIter += self->ctx.convTilingData->khConv * self->ctx.convTilingData->dilationH;
        }
        if (iterParam.kIter == iterParam.maxKL0Iter && iterParam.khIter == iterParam.maxKhIter) {
            strideParams.SetLoop3DstStride(iterParam.currentWoL0);
            strideParams.SetLoop4DstStride(iterParam.currentHoL0 * iterParam.currentWoL0);
            SetConvStridePara(strideParams);
            SetPreQuantAndReluMode(self, convFixParams);
        }
        LoadL0ModuleFor5162<Intf>(self, khStartPos, khStep, convFixParams, iterParam);
        khStartPos += khStep;
        if (iterParam.kIter == 0 && khIdx == 0) {
            convFixParams.SetBiasMatrixInitVal(false);
            convFixParams.SetBiasMatrixBroadcast(false);
            convFixParams.SetDependEnable(true);
        }
    }
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::IterateBiasScale(Intf* self, IterateParam& iterParam)
{
    if (self->ctx.enableBias) {
        if (!self->ctx.convTilingData->biasFullLoadFlag) {
            self->ctx.biasL1 = self->ctx.queueBiasL1.template AllocTensor<typename Intf::BiasT>();
            self->ctx.loadBiasL1Ins.LoadChannelWiseL1(self->ctx.biasL1, self->ctx.biasgm, iterParam);
            self->ctx.queueBiasL1.EnQue(self->ctx.biasL1);
            self->ctx.biasL1 = self->ctx.queueBiasL1.template DeQue<typename Intf::BiasT>();
        }
        self->ctx.biasBT = self->ctx.queueBiasBT.template AllocTensor<typename Intf::L0cT>();
        self->ctx.loadBiasBTIns.LoadBiasBt(self->ctx.biasBT, self->ctx.biasL1, iterParam);
        self->ctx.queueBiasBT.EnQue(self->ctx.biasBT);
        self->ctx.biasBT = self->ctx.queueBiasBT.template DeQue<typename Intf::L0cT>();
    }

    if (!self->ctx.convTilingData->fixpParamsFullLoadFlag) {
        if (self->ctx.enableVectorQuant || self->ctx.enableVectorRelu) {
            event_t eventId = static_cast<event_t>(self->ctx.pipe.FetchEventID(HardEvent::MTE1_MTE2));
            SetFlag<HardEvent::MTE1_MTE2>(eventId);
            WaitFlag<HardEvent::MTE1_MTE2>(eventId);
        }
        if (self->ctx.enableVectorQuant) {
            self->ctx.scaleL1 = self->ctx.queueScaleL1.template AllocTensor<typename Intf::ScaleT>();
            if constexpr (Intf::isExtendConv2d) {
                if (self->ctx.convTilingData->quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                    self->ctx.loadScaleL1Ins.LoadChannelWiseL1(self->ctx.scaleL1, self->ctx.scalegm, iterParam);
                }
                if (self->ctx.convTilingData->dualOutput &&
                    self->ctx.convTilingData->quantMode1 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                    self->ctx.loadScaleL1Ins.LoadChannelWiseL1(self->ctx.scaleL1[self->ctx.scale1L1offset],
                                                               self->ctx.scale1gm, iterParam);
                }
            } else {
                self->ctx.loadScaleL1Ins.LoadChannelWiseL1(self->ctx.scaleL1, self->ctx.scalegm, iterParam);
            }
            self->ctx.queueScaleL1.EnQue(self->ctx.scaleL1);
            self->ctx.scaleL1 = self->ctx.queueScaleL1.template DeQue<typename Intf::ScaleT>();
        }
        if constexpr (Intf::isExtendConv2d) {
            if (self->ctx.enableVectorRelu) {
                self->ctx.reluWeightL1 = self->ctx.queueReluWeightL1.template AllocTensor<typename Intf::ReluWeightT>();
                if (self->ctx.convTilingData->reluMode0 == static_cast<uint8_t>(ReluMode::VECTOR_RELU)) {
                    self->ctx.loadReluWeightL1Ins.LoadChannelWiseL1(self->ctx.reluWeightL1, self->ctx.reluWeightGM,
                                                                    iterParam);
                }
                if (self->ctx.convTilingData->dualOutput &&
                    self->ctx.convTilingData->reluMode1 == static_cast<uint8_t>(ReluMode::VECTOR_RELU)) {
                    self->ctx.loadReluWeightL1Ins.LoadChannelWiseL1(
                        self->ctx.reluWeightL1[self->ctx.reluWeight1L1offset], self->ctx.reluWeight1GM, iterParam);
                }
                self->ctx.queueReluWeightL1.EnQue(self->ctx.reluWeightL1);
                self->ctx.reluWeightL1 = self->ctx.queueReluWeightL1.template DeQue<typename Intf::ReluWeightT>();
            }
        }
        if (self->ctx.enableVectorQuant || self->ctx.enableVectorRelu) {
            event_t eventId = static_cast<event_t>(self->ctx.pipe.FetchEventID(HardEvent::MTE2_MTE1));
            SetFlag<HardEvent::MTE2_MTE1>(eventId);
            WaitFlag<HardEvent::MTE2_MTE1>(eventId);
        }
    }

    FixPipeConfigBitModeParams param;
    param.SetShiftValue(SHIFT_VAL_DEFAULT);
    // move scale from l1 to ppb
    if (self->ctx.enableVectorQuant) {
        self->ctx.scaleFB = self->ctx.queueScaleFB.template AllocTensor<typename Intf::ScaleT>();
        self->ctx.loadScaleFBIns.LoadChannelWiseFB(self->ctx.scaleFB, self->ctx.scaleL1, iterParam,
                                                   PostProcBufBlockMode::PRE_QUANT);
        self->ctx.queueScaleFB.EnQue(self->ctx.scaleFB);
        self->ctx.scaleFB = self->ctx.queueScaleFB.template DeQue<typename Intf::ScaleT>();
        param.SetPreQuantAddr(CeilDiv(self->ctx.scaleFB.GetPhyAddr(), DST_MEM_BLOCK_PRE_QUANT_UINT));
    }
    if (self->ctx.convTilingData->reluMode0 == static_cast<uint8_t>(ReluMode::VECTOR_RELU)) {
        self->ctx.reluWeightFB = self->ctx.queueReluWeightFB.template AllocTensor<typename Intf::ReluWeightT>();
        self->ctx.loadReluWeightFBIns.LoadChannelWiseFB(self->ctx.reluWeightFB, self->ctx.reluWeightL1, iterParam,
                                                        PostProcBufBlockMode::PRE_RELU);
        self->ctx.queueReluWeightFB.EnQue(self->ctx.reluWeightFB);
        self->ctx.reluWeightFB = self->ctx.queueReluWeightFB.template DeQue<typename Intf::ReluWeightT>();
        param.SetPreReluAddr(CeilDiv(self->ctx.reluWeightFB.GetPhyAddr(), DST_MEM_BLOCK_PRE_RELU_UINT));
    }
    if constexpr (Intf::isFixedPoint) {
        uint8_t restorationCtrl = 1;
        param.SetRestorationCtrl(restorationCtrl);
    }
    SetFixPipeConfig(param);
    if (iterParam.cubeWaitMte1Flag) {
        event_t eventId2 = static_cast<event_t>(self->ctx.l1ResPingPongFlag & 0x1);
        SetFlag<HardEvent::MTE1_M>(eventId2);
        WaitFlag<HardEvent::MTE1_M>(eventId2);
    }
}

template <class Intf, uint32_t ImplType>
__aicore__ bool Iterate<Intf, ImplType>::IterateImplHWMode(Intf* self, IterateParam& iterParam, bool enPartialSum)
{
    if (self->ctx.isFirstIterate) {
        FirstIterateImplHWMode<Intf>(self, iterParam);
    } else if (self->iterateMFirstRT) {
        if (IterateMFirstHWMode<Intf>(self, iterParam) == false) {
            return false;
        }
    } else {
        if (IterateNFirstHWMode<Intf>(self, iterParam) == false) {
            return false;
        }
    }
    UpdateHoL0WoL0<Intf>(self, iterParam);
    IterateK(self, iterParam);
    return true;
}
} // namespace ConvFunc
#endif // CONV_ITERATE_IMPL_5162_H
