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
 * \file conv_iterate_m_mode_impl.h
 * \brief
 */

#ifndef CONV_ITERATE_M_MODE_IMPL_H
#define CONV_ITERATE_M_MODE_IMPL_H

#include "conv_config.h"
#include "conv_iterate_base_impl.h"
#include "conv_util.h"

namespace ConvFunc {
using namespace AscendC;
using namespace conv;

template <class Intf>
__aicore__ inline void InitMDirectionValue(Intf *self)
{
    // M方向变量计算
    self->ctx.mAL1Tail = self->ctx.singleCoreM % self->ctx.mAL1;
    self->ctx.mAL1Tail = self->ctx.mAL1Tail == 0 ? self->ctx.mAL1 : self->ctx.mAL1Tail;
    self->ctx.ddr2l1LoopM = CeilDiv(self->ctx.singleCoreM, self->ctx.mAL1);
    self->ctx.maxMAL1Iter = self->ctx.ddr2l1LoopM - 1;
    self->ctx.mAL0Tail = self->ctx.mAL1Tail % self->ctx.mL0;
    self->ctx.mAL0Tail = self->ctx.mAL0Tail == 0 ? self->ctx.mL0 : self->ctx.mAL0Tail;
    self->ctx.l12l0LoopM = CeilDiv(self->ctx.mAL1, self->ctx.mL0);
    self->ctx.maxML0Iter = self->ctx.l12l0LoopM - 1;
}

template <class Intf>
__aicore__ inline void CalcMDirectionVar(Intf *self)
{
    self->ctx.l12l0LoopM = self->ctx.mAL1Iter == self->ctx.maxMAL1Iter ?
        CeilDiv(self->ctx.mAL1Tail, self->ctx.mL0) : CeilDiv(self->ctx.mAL1, self->ctx.mL0);
    self->ctx.maxML0Iter = self->ctx.l12l0LoopM - 1;
    self->ctx.mAL1UpdateFlag = true;
}

template <class Intf>
__aicore__ inline void FirstIterateImplMMode(Intf *self)
{
    self->ctx.mL0Iter = 0;
    self->ctx.nL0Iter = 0;
    self->ctx.mAL1Iter = 0;
    self->ctx.nBL1Iter = 0;
    if constexpr (Intf::isConv3D) {
        self->ctx.dOutIter = 0;
    }
    self->ctx.batchIter = 0;
    self->ctx.loadAL1Flag = true;
    self->ctx.loadBL1Flag = true;
    self->ctx.loadAL0Flag = true;
    self->ctx.loadBL0Flag = true;
    self->ctx.isFirstIterate = false;

    FirstIterateImplVec<Intf>(self);

    CalcMDirectionVar<Intf>(self);
    CalcCoDirectionVar<Intf>(self);
}

template <class Intf>
__aicore__ inline bool IterateL0MFirstMMode(Intf *self)
{
    self->ctx.mL0Iter++;
    if (self->ctx.mL0Iter == self->ctx.l12l0LoopM) {
        self->ctx.mL0Iter = 0;
    } else {
        self->ctx.loadBL0Flag = false;
        if constexpr (Intf::kl0FullLoadFlag) {
            self->ctx.kL0FullLoadAl0PingPongFlag =
                CheckReduceOneKNotSupportDBCase<Intf>(self) ? 0 : self->ctx.kL0FullLoadAl0PingPongFlag;
        }
        return true;
    }

    if constexpr (Intf::kl0FullLoadFlag) {
        self->ctx.kL0FullLoadAl0PingPongFlag = 0;
    }

    if constexpr (Intf::hasNL0IterFlag) {
        self->ctx.nL0Iter++;
        self->ctx.loadBL0Flag = true;
        if (self->ctx.nL0Iter == self->ctx.l12l0LoopN) {
            self->ctx.nL0Iter = 0;
        } else {
            return true;
        }
    }

    return false;
}

template <class Intf>
__aicore__ inline bool IterateMFirstMMode(Intf *self)
{
    if (IterateL0MFirstMMode<Intf>(self)) {
        return true;
    }

    if (self->ctx.kAL1fullload) {
        self->ctx.queueAL1.FreeTensor(self->ctx.al1);
    }

    self->ctx.mAL1Iter++;
    self->ctx.loadAL1Flag = true;
    if (self->ctx.mAL1Iter == self->ctx.ddr2l1LoopM) {
        self->ctx.mAL1Iter = 0;
        CalcMDirectionVar<Intf>(self);
    } else {
        CalcMDirectionVar<Intf>(self);
        return true;
    }

    if constexpr (Intf::isConv3D) {
        self->ctx.dOutIter++;
        if (self->ctx.dOutIter == self->ctx.ddr2l1LoopD) {
            self->ctx.dOutIter = 0;
        } else {
            return true;
        }
    }

    self->ctx.batchIter++;
    if (self->ctx.batchIter == self->ctx.ddr2l1LoopBatch) {
        self->ctx.batchIter = 0;
    } else {
        self->ctx.loadAL1Flag = true;
        self->ctx.loadBL1Flag = false;
        return true;
    }

    if (self->ctx.kBL1fullload) {
        self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
    }

    if constexpr (Intf::hasNL1IterFlag) {
        self->ctx.nBL1Iter++;
        CalcCoDirectionVar<Intf>(self);
        self->ctx.loadBL1Flag = true;
        if (self->ctx.nBL1Iter != self->ctx.ddr2l1LoopN) {
            return true;
        }
    }

    return false;
}

template <class Intf>
__aicore__ inline bool IterateL0NFirstMMode(Intf *self)
{
    if constexpr (Intf::hasNL0IterFlag) {
        self->ctx.nL0Iter++;
        if (self->ctx.nL0Iter == self->ctx.l12l0LoopN) {
            self->ctx.nL0Iter = 0;
        } else {
            self->ctx.loadAL0Flag = false;
            if constexpr (Intf::kl0FullLoadFlag) {
                self->ctx.kL0FullLoadBl0PingPongFlag =
                    CheckReduceOneKNotSupportDBCase<Intf>(self) ? 0 : self->ctx.kL0FullLoadBl0PingPongFlag;
            }
            return true;
        }
    }

    if constexpr (Intf::kl0FullLoadFlag) {
        self->ctx.kL0FullLoadBl0PingPongFlag = 0;
    }

    self->ctx.mL0Iter++;
    self->ctx.loadAL0Flag = true;
    if (self->ctx.mL0Iter == self->ctx.l12l0LoopM) {
        self->ctx.mL0Iter = 0;
    } else {
        return true;
    }

    return false;
}

template <class Intf>
__aicore__ inline bool IterateNFirstMMode(Intf *self)
{
    if (IterateL0NFirstMMode<Intf>(self)) {
        return true;
    }

    if (self->ctx.kBL1fullload) {
        self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
    }

    if constexpr (Intf::hasNL1IterFlag) {
        self->ctx.nBL1Iter++;
        self->ctx.loadBL1Flag = true;
        if (self->ctx.nBL1Iter == self->ctx.ddr2l1LoopN) {
            self->ctx.nBL1Iter = 0;
            CalcCoDirectionVar<Intf>(self);
            
        } else {
            CalcCoDirectionVar<Intf>(self);
            return true;
        }
    }

    if (self->ctx.kAL1fullload) {
        self->ctx.queueAL1.FreeTensor(self->ctx.al1);
    }

    self->ctx.mAL1Iter++;
    self->ctx.loadAL1Flag = true;
    if (self->ctx.mAL1Iter == self->ctx.ddr2l1LoopM) {
        self->ctx.mAL1Iter = 0;
        CalcMDirectionVar<Intf>(self);
    } else {
        CalcMDirectionVar<Intf>(self);
        return true;
    }

    if constexpr (Intf::isConv3D) {
        self->ctx.dOutIter++;
        if (self->ctx.dOutIter == self->ctx.ddr2l1LoopD) {
            self->ctx.dOutIter = 0;
        } else {
            return true;
        }
    }

    self->ctx.batchIter++;
    if (likely(self->ctx.batchIter != self->ctx.ddr2l1LoopBatch)) {
        self->ctx.loadAL1Flag = true;
        self->ctx.loadBL1Flag = true;
        return true;
    }

    return false;
}

template <class Intf>
__aicore__ inline bool UpdateCommonItersMModeMFirst(Intf *self, TempIters& tempIters)
{
    tempIters.mAL1Iter = self->ctx.mAL1Iter;
    tempIters.batchIter = self->ctx.batchIter;
    tempIters.nBL1Iter = self->ctx.nBL1Iter;

    tempIters.mAL1Iter += 1;
    if (tempIters.mAL1Iter <= self->ctx.maxMAL1Iter) {
        return false;
    }
    tempIters.mAL1Iter = 0;
    tempIters.batchIter += 1;
    if (tempIters.batchIter < self->ctx.ddr2l1LoopBatch) {
        return false;
    }
    tempIters.batchIter = 0;
    tempIters.nBL1Iter += 1;
    if (tempIters.nBL1Iter <= self->ctx.maxNBL1Iter) {
        return false;
    }
    tempIters.endTag = true;
    return true;
}

template <class Intf>
__aicore__ inline bool UpdateCommonItersMModeNFirst(Intf *self, TempIters& tempIters)
{
    tempIters.nBL1Iter = self->ctx.nBL1Iter;
    tempIters.mAL1Iter = self->ctx.mAL1Iter;
    tempIters.batchIter = self->ctx.batchIter;

    tempIters.nBL1Iter += 1;
    if (tempIters.nBL1Iter <= self->ctx.maxNBL1Iter) {
        return false;
    }
    tempIters.nBL1Iter = 0;
    tempIters.mAL1Iter += 1;
    if (tempIters.mAL1Iter <= self->ctx.maxMAL1Iter) {
        return false;
    }
    tempIters.mAL1Iter = 0;
    tempIters.batchIter += 1;
    if (tempIters.batchIter < self->ctx.ddr2l1LoopBatch) {
        return false;
    }
    tempIters.endTag = true;
    return true;
}

};

#endif // CONV_ITERATE_M_MODE_IMPL_H