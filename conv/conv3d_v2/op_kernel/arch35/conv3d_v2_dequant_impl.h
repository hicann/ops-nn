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
 * \file conv3d_v2_dequant_impl.h
 * \brief
 */

#ifndef CONV3D_V2_DEQUANT_IMPL_H
#define CONV3D_V2_DEQUANT_IMPL_H

#include "conv3d_v2_config.h"
#include "conv3d_v2_util.h"
#include "conv3d_v2_instr.h"
#include "../../common/arch35/conv_iterate_base_impl.h"
#include "../../common/arch35/conv_iterate_hw_mode_impl.h"
#include "../../common/arch35/conv_iterate_m_mode_impl.h"

namespace Conv3dFunc {
using namespace ConvFunc;
using namespace conv;

template <class Intf>
__aicore__ inline void DeQuantCalcFixpTimes(Intf *self)
{
    self->ctx.fixpTimesTotal = self->ctx.ddr2l1LoopBatch * self->ctx.ddr2l1LoopN * self->ctx.l12l0LoopN *
                               self->ctx.ddr2l1LoopD;
    if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
        uint64_t ddr2UbLoopM = CeilDiv(self->ctx.singleCoreM, self->ctx.convTiling->mUB);
        self->ctx.fixpTimesTotal *= ddr2UbLoopM;
    } else {
        uint64_t ddr2UbLoopW = 0;
        if constexpr (Intf::hasWL0IterFlag) {
            if (self->ctx.woL1SmallTail > 0) {
                ddr2UbLoopW = (self->ctx.ddr2l1LoopW - W_TAIL_NUM) *
                    CeilDiv(self->ctx.convTiling->woL1, self->ctx.convTiling->mUB) +
                    CeilDiv(self->ctx.woAL1Tail, self->ctx.convTiling->mUB) +
                    CeilDiv(self->ctx.woL1SmallTail, self->ctx.convTiling->mUB);
            } else {
                ddr2UbLoopW = CeilDiv(self->ctx.singleCoreWo, self->ctx.convTiling->mUB);
            }
        } else {
            ddr2UbLoopW = CeilDiv(self->ctx.singleCoreWo, self->ctx.convTiling->mUB);
        }
        self->ctx.fixpTimesTotal *= self->ctx.singleCoreHo * ddr2UbLoopW;
    }
}

template <class Intf>
__aicore__ inline void DeQuantInitBuf(Intf *self)
{
    self->ctx.pipe.InitBuffer(
        self->ctx.mmadResUbBuf, self->ctx.convTiling->mUB * self->ctx.convTiling->nUB * Intf::sizeOfL0c);
    self->ctx.mmadResUbTensor = self->ctx.mmadResUbBuf.template Get<typename Intf::L0cT>();
    self->ctx.outputResUbTensor = self->ctx.mmadResUbBuf.template Get<typename Intf::OutputT>();

    self->ctx.pipe.InitBuffer(self->ctx.scaleUbBuf, self->ctx.convTiling->nUB * Intf::sizeOfScale);
    self->ctx.scaleTensor = self->ctx.scaleUbBuf.template Get<typename Intf::ScaleT>();

    if constexpr (AscendC::IsSameType<typename Intf::BiasT, float>::value) {
        self->ctx.pipe.InitBuffer(self->ctx.biasB32UbBuf, self->ctx.convTiling->nUB * Intf::sizeOfBias);
    } else {
        self->ctx.pipe.InitBuffer(self->ctx.biasB16UbBuf, self->ctx.convTiling->nUB * Intf::sizeOfBias);
        self->ctx.biasB16Tensor = self->ctx.biasB16UbBuf.template Get<typename Intf::BiasT>();

        self->ctx.pipe.InitBuffer(self->ctx.biasB32UbBuf, self->ctx.convTiling->nUB * DTYPE_SIZE_B32);   
    }
    self->ctx.biasB32Tensor = self->ctx.biasB32UbBuf.template Get<float>();
}

template <class Intf>
__aicore__ inline void DeQuantVecInit(Intf *self)
{
    self->ctx.vecId = GetSubBlockIdx();

    self->ctx.orgCo = self->ctx.convTiling->orgCo;
    self->ctx.orgDo = self->ctx.convTiling->orgDo;
    self->ctx.orgHo = self->ctx.convTiling->orgHo;
    self->ctx.orgWo = self->ctx.convTiling->orgWo;

    self->ctx.singleCoreCo = self->ctx.convTiling->singleCoreCo;
    self->ctx.singleCoreDo = self->ctx.convTiling->singleCoreDo;
    self->ctx.ddr2l1LoopD = self->ctx.singleCoreDo;
    if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
        self->ctx.singleCoreM = self->ctx.convTiling->singleCoreHo;
        self->ctx.mAL1 = self->ctx.convTiling->hoL1;
        self->ctx.mL0 = self->ctx.convTiling->hoL0;
        InitMDirectionValue<Intf>(self);
    } else {
        self->ctx.singleCoreHo = self->ctx.convTiling->singleCoreHo;
        self->ctx.singleCoreWo = self->ctx.convTiling->singleCoreWo;
        InitHoDirectionValue<Intf>(self);
        InitWoDirectionValue<Intf>(self);
    }
    InitCoDirectionValue<Intf>(self);

    self->ctx.outputOneBatchSize = self->ctx.orgCo * self->ctx.orgHo * self->ctx.orgWo * self->ctx.orgDo;

    DeQuantInitBuf<Intf>(self);

    self->ctx.dequantL0C2UBTools.SetParams(self);
    self->ctx.dequantLoadParamsTools.SetParams(self);
    self->ctx.dequantCalcTools.SetParams(self);
    self->ctx.dequantUB2GmTools.SetParams(self);
}

template <class Intf>
__aicore__ inline void DeQuantUBParamsUpdate(Intf *self)
{
    uint32_t mUbTail = 0;
    if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
        uint64_t currentML0 =
            self->ctx.mAL1Iter == self->ctx.maxMAL1Iter && self->ctx.mL0Iter == self->ctx.maxML0Iter ?
            self->ctx.mAL0Tail : self->ctx.mL0;
        self->ctx.l0C2UbLoopM = CeilDiv(currentML0, self->ctx.convTiling->mUB);
        self->ctx.maxMUbIter = self->ctx.l0C2UbLoopM - 1;
        mUbTail = currentML0 % self->ctx.convTiling->mUB;
        mUbTail = mUbTail == 0 ? self->ctx.convTiling->mUB : mUbTail;
        self->ctx.currentMUb = self->ctx.mUbIter == self->ctx.maxMUbIter ? mUbTail : self->ctx.convTiling->mUB;
    } else {
        self->ctx.currentHoL0 = self->ctx.hoL0Iter == self->ctx.maxHoL0Iter ?
            self->ctx.hoL0Tail : self->ctx.convTiling->hoL0;
        self->ctx.currentWoL0 = self->ctx.woL0Iter == self->ctx.maxWoL0Iter ?
            self->ctx.woL0Tail : self->ctx.convTiling->woL0;
        self->ctx.l0C2UbLoopWo = CeilDiv(self->ctx.currentWoL0, self->ctx.convTiling->mUB);
        self->ctx.l0C2UbLoopM = self->ctx.l0C2UbLoopWo * self->ctx.currentHoL0;
        self->ctx.maxMUbIter = self->ctx.l0C2UbLoopM - 1;
        mUbTail = self->ctx.currentWoL0 % self->ctx.convTiling->mUB;
        mUbTail = mUbTail == 0 ? self->ctx.convTiling->mUB : mUbTail;
        self->ctx.currentMUb = (self->ctx.mUbIter % self->ctx.l0C2UbLoopWo) == (self->ctx.l0C2UbLoopWo - 1) ?
            mUbTail : self->ctx.convTiling->mUB;
    }

    self->ctx.currentNUb = CalcCurrentNL0<Intf>(self);
}

template <class Intf>
__aicore__ inline void DeQuantIterInit(Intf *self)
{
    self->ctx.batchIter = 0;
    self->ctx.dOutIter = 0;
    self->ctx.nBL1Iter = 0;
    self->ctx.nL0Iter = 0;
    CalcCoDirectionVar<Intf>(self);

    if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
        self->ctx.mAL1Iter = 0;
        self->ctx.mL0Iter = 0;
        CalcMDirectionVar<Intf>(self);
    } else {
        self->ctx.hoAL1Iter = 0;
        self->ctx.woAL1Iter = 0;
        self->ctx.hoL0Iter = 0;
        self->ctx.woL0Iter = 0;
        CalcHoDirectionVar<Intf>(self);
        CalcWoDirectionVar<Intf>(self);
    }
    DeQuantUBParamsUpdate<Intf>(self);
    DeQuantCalcFixpTimes<Intf>(self);

    self->ctx.mUbIter = 0;
    self->ctx.fixpTimes = self->ctx.vecId == 0 ? 0 : 1;
    self->ctx.pingPongFlag = 0;
    self->ctx.isFirstIterate = true;
}

template <class Intf>
__aicore__ inline bool DeQuantIterUpdate(Intf *self)
{
    self->ctx.fixpTimes++;

    if (unlikely(self->ctx.isFirstIterate && self->ctx.vecId == 0)) {
        return true;
    }

    self->ctx.pingPongFlag ^= 1;

    self->ctx.mUbIter++;
    if (self->ctx.mUbIter == self->ctx.l0C2UbLoopM) {
        self->ctx.mUbIter = 0;
    } else {
        return true;
    }

    if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
        if constexpr (Intf::iterateMFirstFlag) {
            return IterateMFirstMMode<Intf>(self);
        } else {
            return IterateNFirstMMode<Intf>(self);
        }
    } else {
        if constexpr (Intf::iterateMFirstFlag) {
            return IterateMFirstHWMode<Intf>(self);
        } else {
            return IterateNFirstHWMode<Intf>(self);
        }
    }

    return false;
}


template <class Intf>
__aicore__ inline bool DeQuantVecImpl(Intf *self)
{
    bool loadParamsFlag = false;
    DeQuantIterInit<Intf>(self);
    while (DeQuantIterUpdate<Intf>(self)) {
        DeQuantUBParamsUpdate<Intf>(self);

        if constexpr (Intf::iterateMFirstFlag) {
            if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
                loadParamsFlag = loadParamsFlag ||
                    (self->ctx.isFirstIterate || (self->ctx.mUbIter == 0 && self->ctx.mL0Iter == 0));
            } else {
                loadParamsFlag = loadParamsFlag || (self->ctx.isFirstIterate ||
                    (self->ctx.mUbIter == 0 && self->ctx.hoL0Iter == 0 && self->ctx.woL0Iter == 0));
            }
        } else {
                loadParamsFlag = loadParamsFlag || (self->ctx.isFirstIterate || self->ctx.mUbIter == 0);
        }

        if (self->ctx.pingPongFlag != self->ctx.vecId) {
            continue;
        }

        if (unlikely(self->ctx.isFirstIterate)) {
            CrossCoreSetFlag<CV_ENHANCE_MODE, PIPE_MTE3>(CV_SYNC_ID_MTE3_FIXP);
        }

        if (loadParamsFlag) {
            self->ctx.dequantLoadParamsTools.LoadParams();
            loadParamsFlag = false;
        }

        CrossCoreWaitFlag<CV_ENHANCE_MODE, PIPE_V>(CV_SYNC_ID_FIXP_V);
        self->ctx.dequantCalcTools.DeQuantCalc();

        self->ctx.dequantUB2GmTools.CopyUB2GM();
        if (self->ctx.fixpTimes < self->ctx.fixpTimesTotal) {
            CrossCoreSetFlag<CV_ENHANCE_MODE, PIPE_MTE3>(CV_SYNC_ID_MTE3_FIXP);
        }

        self->ctx.isFirstIterate = false;
    }

    return false;
}

} // namespace Conv3dFunc

#endif // CONV3D_V2_DEQUANT_IMPL_H