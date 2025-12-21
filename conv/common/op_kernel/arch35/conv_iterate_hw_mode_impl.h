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
 * \file conv_iterate_hw_mode_impl.h
 * \brief
 */

#include "conv_config.h"
#include "conv_iterate_base_impl.h"
#include "conv_util.h"

#ifndef CONV_ITERATE_HW_MODE_IMPL_H
#define CONV_ITERATE_HW_MODE_IMPL_H

namespace ConvFunc {
using namespace AscendC;
using namespace conv;

template <class Intf>
static __aicore__ inline void InitHoDirectionValue(Intf *self)
{
    // Ho方向L1及L0均全载时相关变量赋值
    self->ctx.hoAL1Tail = self->ctx.singleCoreHo % self->ctx.convTiling->hoL1;
    self->ctx.hoAL1Tail = self->ctx.hoAL1Tail == 0 ? self->ctx.convTiling->hoL1 : self->ctx.hoAL1Tail;
    self->ctx.currentHoL1 = self->ctx.hoAL1Tail;
    self->ctx.hoL0Tail = self->ctx.currentHoL1 % self->ctx.convTiling->hoL0;
    self->ctx.hoL0Tail = self->ctx.hoL0Tail == 0 ? self->ctx.convTiling->hoL0 : self->ctx.hoL0Tail;
    self->ctx.currentHoL0 = self->ctx.hoL0Tail;

    // Ho方向变量计算
    if constexpr (!Intf::hasHL1IterFlag) {
        self->ctx.ddr2l1LoopH = 1;
        self->ctx.maxHoL1Iter = 0;
    } else {
        self->ctx.ddr2l1LoopH = CeilDiv(self->ctx.singleCoreHo, self->ctx.convTiling->hoL1);
        self->ctx.maxHoL1Iter = self->ctx.ddr2l1LoopH - 1;
    }
}

template <class Intf>
static __aicore__ inline void InitWoDirectionValue(Intf *self)
{
    // Wo方向L1及L0均全载时相关变量赋值
    self->ctx.woAL1Tail = self->ctx.singleCoreWo % self->ctx.convTiling->woL1;
    self->ctx.woAL1Tail = self->ctx.woAL1Tail == 0 ?  self->ctx.convTiling->woL1 : self->ctx.woAL1Tail;
    self->ctx.currentWoL1 = self->ctx.woAL1Tail;
    self->ctx.maxWoL1Iter = 0;
    self->ctx.woL0Tail = self->ctx.currentWoL1 % self->ctx.convTiling->woL0;
    self->ctx.woL0Tail = self->ctx.woL0Tail == 0 ? self->ctx.convTiling->woL0 : self->ctx.woL0Tail;
    self->ctx.currentWoL0 = self->ctx.woL0Tail;
    self->ctx.maxWoL0Iter = 0;
    self->ctx.l12l0LoopW = 1;

    // Wo方向变量计算
    if constexpr (Intf::hasWL1IterFlag) {
        self->ctx.maxWoL1Iter = CeilDiv(self->ctx.singleCoreWo, self->ctx.convTiling->woL1) - 1;
    }

    if constexpr (!Intf::isDmaFlag) {
        if (Intf::hasWL0IterFlag && self->ctx.convTiling->hoL0 > 1 && self->ctx.woAL1Tail % BLOCK_L0_N > 0 &&
            CeilDiv(self->ctx.woAL1Tail, self->ctx.convTiling->woL0) > 1) {
            self->ctx.woL1SmallTail = self->ctx.woAL1Tail % self->ctx.convTiling->woL0;
            self->ctx.woAL1Tail = (self->ctx.woAL1Tail / self->ctx.convTiling->woL0) * self->ctx.convTiling->woL0;
        }
        
        if (self->ctx.woL1SmallTail > 0) {
            self->ctx.maxWoL1Iter += 1;
        }
    }

    self->ctx.ddr2l1LoopW = self->ctx.maxWoL1Iter + 1;
}

template <class Intf>
__aicore__ inline void CalcWoDirectionVar(Intf *self)
{
    if constexpr (Intf::hasWL1IterFlag) {
        if (self->ctx.woL1SmallTail > 0) {
            if (self->ctx.woAL1Iter == self->ctx.maxWoL1Iter) {
                self->ctx.currentWoL1 = self->ctx.woL1SmallTail;
            } else if (self->ctx.woAL1Iter == self->ctx.maxWoL1Iter - 1) {
                self->ctx.currentWoL1 = self->ctx.woAL1Tail;
            } else {
                self->ctx.currentWoL1 = self->ctx.convTiling->woL1;
            }
        } else {
            self->ctx.currentWoL1 = self->ctx.woAL1Iter == self->ctx.maxWoL1Iter ?
                self->ctx.woAL1Tail : self->ctx.convTiling->woL1;
        }
    }
    if constexpr (Intf::hasWL0IterFlag) {
        self->ctx.woL0Tail = self->ctx.currentWoL1 % self->ctx.convTiling->woL0;
        self->ctx.woL0Tail = self->ctx.woL0Tail == 0 ? self->ctx.convTiling->woL0 : self->ctx.woL0Tail;
        self->ctx.maxWoL0Iter = CeilDiv(self->ctx.currentWoL1, self->ctx.convTiling->woL0) - 1;
        self->ctx.l12l0LoopW = self->ctx.maxWoL0Iter + 1;
    }
}

template <class Intf>
__aicore__ inline void CalcHoDirectionVar(Intf *self)
{
    if constexpr (Intf::hasHL1IterFlag) {
        self->ctx.currentHoL1 = self->ctx.hoAL1Iter == self->ctx.maxHoL1Iter ?
            self->ctx.hoAL1Tail : self->ctx.convTiling->hoL1;
    }
    
    if constexpr (Intf::hasHL0IterFlag) {
        self->ctx.hoL0Tail = self->ctx.currentHoL1 % self->ctx.convTiling->hoL0;
        self->ctx.hoL0Tail = self->ctx.hoL0Tail == 0 ? self->ctx.convTiling->hoL0 : self->ctx.hoL0Tail;
        self->ctx.maxHoL0Iter = CeilDiv(self->ctx.currentHoL1, self->ctx.convTiling->hoL0) - 1;
        self->ctx.l12l0LoopH = self->ctx.maxHoL0Iter + 1;
    }
}

template <class Intf>
__aicore__ inline void UpdateHoL0WoL0(Intf *self)
{
    self->ctx.currentHoL0 = self->ctx.hoL0Iter == self->ctx.maxHoL0Iter ?
        self->ctx.hoL0Tail : self->ctx.convTiling->hoL0;
    self->ctx.currentWoL0 = self->ctx.woL0Iter == self->ctx.maxWoL0Iter ?
        self->ctx.woL0Tail : self->ctx.convTiling->woL0;
}

template <class Intf>
__aicore__ inline void FirstIterateImplHWMode(Intf *self)
{
    self->ctx.nL0Iter = 0;
    self->ctx.hoAL1Iter = 0;
    self->ctx.woAL1Iter = 0;
    self->ctx.nBL1Iter = 0;
    self->ctx.woL0Iter = 0;
    self->ctx.hoL0Iter = 0;
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

    CalcWoDirectionVar<Intf>(self);
    CalcHoDirectionVar<Intf>(self);
    CalcCoDirectionVar<Intf>(self);
}

template <class Intf>
__aicore__ inline bool IterateHWL1(Intf *self)
{
    if constexpr (Intf::hasWL1IterFlag) {
        self->ctx.woAL1Iter++;
        self->ctx.loadAL1Flag = true;
        if (self->ctx.woAL1Iter == self->ctx.ddr2l1LoopW) {
            self->ctx.woAL1Iter = 0;
            CalcWoDirectionVar<Intf>(self);
        } else {
            CalcWoDirectionVar<Intf>(self);
            if constexpr (Intf::iterateMFirstFlag && !Intf::hasNL0IterFlag) {
                self->ctx.loadBL0Flag = false;
            }
            return true;
        }
    }

    if constexpr (Intf::hasHL1IterFlag) {
        self->ctx.hoAL1Iter++;
        if (self->ctx.hoAL1Iter == self->ctx.ddr2l1LoopH) {
            self->ctx.hoAL1Iter = 0;
            CalcHoDirectionVar<Intf>(self);
        } else {
            CalcHoDirectionVar<Intf>(self);
            if constexpr (Intf::iterateMFirstFlag && !Intf::hasNL0IterFlag) {
                self->ctx.loadBL0Flag = false;
            }
            return true;
        }
    }

    return false;
}

template <class Intf>
__aicore__ inline bool IterateL0MFirstHWMode(Intf *self)
{
    if constexpr (Intf::hasWL0IterFlag) {
        self->ctx.woL0Iter++;
        if (self->ctx.woL0Iter == self->ctx.l12l0LoopW) {
            self->ctx.woL0Iter = 0;
        } else {
            self->ctx.loadBL0Flag = false;
            if constexpr (Intf::kl0FullLoadFlag) {
                self->ctx.kL0FullLoadAl0PingPongFlag =
                    CheckReduceOneKNotSupportDBCase<Intf>(self) ? 0 : self->ctx.kL0FullLoadAl0PingPongFlag;
            }
            return true;
        }
    }

    if constexpr (Intf::hasHL0IterFlag) {
        self->ctx.hoL0Iter++;
        if (self->ctx.hoL0Iter == self->ctx.l12l0LoopH) {
            self->ctx.hoL0Iter = 0;
        } else {
            self->ctx.loadBL0Flag = false;
            if constexpr (Intf::kl0FullLoadFlag) {
                self->ctx.kL0FullLoadAl0PingPongFlag =
                    CheckReduceOneKNotSupportDBCase<Intf>(self) ? 0 : self->ctx.kL0FullLoadAl0PingPongFlag;
            }
            return true;
        }
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
__aicore__ inline bool IterateMFirstHWMode(Intf *self)
{
    if (IterateL0MFirstHWMode<Intf>(self)) {
        return true;
    }

    if (self->ctx.kAL1fullload) {
        self->ctx.queueAL1.FreeTensor(self->ctx.al1);
    }

    if (IterateHWL1<Intf>(self)) {
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
        if (likely(self->ctx.nBL1Iter != self->ctx.ddr2l1LoopN)) {
            return true;
        }
    }

    return false;
}

template <class Intf>
__aicore__ inline bool IterateL0NFirstHWMode(Intf *self)
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
    if constexpr (Intf::hasWL0IterFlag) {
        self->ctx.woL0Iter++;
        self->ctx.loadAL0Flag = true;
        if (self->ctx.woL0Iter == self->ctx.l12l0LoopW) {
            self->ctx.woL0Iter = 0;
        } else {
            return true;
        }
    }
    if constexpr (Intf::hasHL0IterFlag) {
        self->ctx.hoL0Iter++;
        self->ctx.loadAL0Flag = true;
        if (self->ctx.hoL0Iter == self->ctx.l12l0LoopH) {
            self->ctx.hoL0Iter = 0;
        } else {
            return true;
        }
    }

    return false;
}

template <class Intf>
__aicore__ inline bool IterateNFirstHWMode(Intf *self)
{
    if (IterateL0NFirstHWMode<Intf>(self)) {
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

    if (IterateHWL1<Intf>(self)) {
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
__aicore__ inline bool UpdateCommonItersHWModeMFirst(Intf *self, TempIters& tempIters)
{
    tempIters.woAL1Iter = self->ctx.woAL1Iter;
    tempIters.hoAL1Iter = self->ctx.hoAL1Iter;
    tempIters.batchIter = self->ctx.batchIter;
    tempIters.nBL1Iter = self->ctx.nBL1Iter;

    tempIters.woAL1Iter += 1;
    if (tempIters.woAL1Iter <= self->ctx.maxWoL1Iter) {
        return false;
    }
    tempIters.woAL1Iter = 0;
    tempIters.hoAL1Iter += 1;
    if (tempIters.hoAL1Iter <= self->ctx.maxHoL1Iter) {
        return false;
    }
    tempIters.hoAL1Iter = 0;
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
__aicore__ inline bool UpdateCommonItersHWModeNFirst(Intf *self, TempIters& tempIters)
{
    tempIters.nBL1Iter = self->ctx.nBL1Iter;
    tempIters.woAL1Iter = self->ctx.woAL1Iter;
    tempIters.hoAL1Iter = self->ctx.hoAL1Iter;
    tempIters.batchIter = self->ctx.batchIter;

    tempIters.nBL1Iter += 1;
    if (tempIters.nBL1Iter <= self->ctx.maxNBL1Iter) {
        return false;
    }
    tempIters.nBL1Iter = 0;
    tempIters.woAL1Iter = self->ctx.woAL1Iter + 1;
    if (tempIters.woAL1Iter <= self->ctx.maxWoL1Iter) {
        return false;
    }
    tempIters.woAL1Iter = 0;
    tempIters.hoAL1Iter += 1;
    if (tempIters.hoAL1Iter <= self->ctx.maxHoL1Iter) {
        return false;
    }
    tempIters.hoAL1Iter = 0;
    tempIters.batchIter += 1;
    if (tempIters.batchIter < self->ctx.ddr2l1LoopBatch) {
        return false;
    }
    tempIters.endTag = true;
    return true;
}

};


#endif // CONV_ITERATE_HW_MODE_IMPL_H