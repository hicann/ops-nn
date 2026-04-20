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
__aicore__ inline void CalcGroupOptParamForMMode(Intf *self)
{
    if (((self->ctx.groupOptIter + 1 == self->ctx.singleGroupOpt - 1 && self->ctx.groupOptIter != 0) ||
        self->ctx.singleGroupOpt == 1) && self->ctx.updateEnlarge != self->ctx.convTilingData->convApiTiling.enlarge) {
        self->ctx.singleGroups = self->ctx.updateEnlarge;
        self->ctx.singleGroups = self->ctx.singleGroups == 0 ? self->ctx.convTilingData->convApiTiling.enlarge : self->ctx.singleGroups;
    }

    uint64_t enlargeTail = self->ctx.singleGroups % self->ctx.convTilingData->convApiTiling.enlarge;
    enlargeTail = enlargeTail == 0 ? self->ctx.convTilingData->convApiTiling.enlarge : enlargeTail;
    if (enlargeTail != self->ctx.convTilingData->convApiTiling.enlarge) {
        self->ctx.singleCoreCi = enlargeTail * (self->ctx.convTilingData->convApiTiling.orgCi / self->ctx.convTilingData->convApiTiling.groups);
        if (self->ctx.groupOptIter == self->ctx.singleGroupOpt - 1) {
            self->ctx.singleCoreCo = self->ctx.updateSingleCoOpt;

            uint64_t totalKAlignK0 = AlignB(self->ctx.singleCoreCi, Intf::k0) * self->ctx.convTilingData->convApiTiling.kernelHxkernelW;
            self->ctx.ddr2l0LoopK = CeilDiv(totalKAlignK0, self->ctx.convTilingData->convApiTiling.kL0);
            self->ctx.maxKL0Iter = self->ctx.ddr2l0LoopK - 1;
            self->ctx.kL0Tail = totalKAlignK0 % self->ctx.convTilingData->convApiTiling.kL0;
            if constexpr (Intf::k0 != Intf::k0FmapTail) {
                self->ctx.kAL0Tail = AlignB(self->ctx.singleCoreCi, Intf::k0FmapTail) *
                    self->ctx.convTilingData->convApiTiling.kernelHxkernelW % self->ctx.convTilingData->convApiTiling.kL0;
                self->ctx.kAL0Tail = self->ctx.kAL0Tail == 0 ? self->ctx.convTilingData->convApiTiling.kL0 : self->ctx.kAL0Tail;
            }
            self->ctx.kL0Tail = self->ctx.kL0Tail == 0 ? self->ctx.convTilingData->convApiTiling.kL0 : self->ctx.kL0Tail;
            
            InitCoDirectionValue<Intf>(self);
        }
    }
    if ASCEND_IS_AIC_CONV {
        CalcMDirectionVar<Intf>(self);
        CalcCoDirectionVar<Intf>(self);
        if constexpr (Intf::groupOptPreloadFlag) {
            OptGroupCalcBL1LoadTimes<Intf>(self);
        }
    }
    if ASCEND_IS_AIV_CONV {
        if constexpr (Intf::groupOptNDFlag) {
            OptGroupInitKValue<Intf>(self);
        }
    }
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
    self->ctx.loadAL0Flag = true;
    self->ctx.loadBL0Flag = true;
    self->ctx.loadAL1Flag = true;
    self->ctx.loadBL1Flag = true;
    self->ctx.isFirstIterate = false;

    FirstIterateImplVec<Intf>(self);

    CalcMDirectionVar<Intf>(self);
    CalcCoDirectionVar<Intf>(self);

    if constexpr (Intf::groupOptPreloadFlag) {
        if (self->ctx.singleGroupOpt == 1 && self->ctx.updateEnlarge != self->ctx.convTilingData->convApiTiling.enlarge) {
            self->ctx.singleGroups = self->ctx.updateEnlarge;
            self->ctx.singleGroups = self->ctx.singleGroups == 0 ?
                self->ctx.convTilingData->convApiTiling.enlarge : self->ctx.singleGroups;
            CalcGroupOptParamForMMode<Intf>(self);
        }
        LoadAL1BaseModule<Intf>(self);
        self->ctx.loadAL1Flag = true;
        if (self->ctx.singleGroupOpt == 2 && self->ctx.updateEnlarge != self->ctx.convTilingData->convApiTiling.enlarge) {
            self->ctx.singleGroups = self->ctx.updateEnlarge;
            self->ctx.singleGroups = self->ctx.singleGroups == 0 ?
                self->ctx.convTilingData->convApiTiling.enlarge : self->ctx.singleGroups;
            CalcGroupOptParamForMMode<Intf>(self);
        }
    } else if constexpr (Intf::isMPreLoad) {
        self->ctx.al1 = self->ctx.queueAL1.template AllocTensor<typename Intf::FmapT>();
        self->ctx.loadAl1Ins.LoadAL1();
        self->ctx.queueAL1.EnQue(self->ctx.al1);
        self->ctx.mAL1UpdateFlag = true;
        LoadBL1BaseModule<Intf>(self);
        self->ctx.bl1 = self->ctx.queueBL1.template DeQue<typename Intf::WeightT>();
        self->ctx.loadBL0Flag = true;
        if constexpr (Intf::isL0BFullLoadable) {
            self->ctx.loadBL0Ins.FullLoadBL0(self->ctx.wholeBl0Tensor[0]);
        }
    }
}

template <class Intf>
__aicore__ inline bool IterateL0MFirstMMode(Intf *self)
{
    self->ctx.mL0Iter++;
    if (self->ctx.mL0Iter != self->ctx.l12l0LoopM) {
        self->ctx.loadBL0Flag = false;
        if ASCEND_IS_AIC_CONV {
            if constexpr (Intf::kl0FullLoadFlag) {
                self->ctx.kL0FullLoadAl0PingPongFlag =
                    CheckReduceOneKNotSupportDBCase<Intf>(self) ? 0 : self->ctx.kL0FullLoadAl0PingPongFlag;
            }
        }
        return true;
    } else {
        self->ctx.mL0Iter = 0;
    }

    if ASCEND_IS_AIC_CONV {
        if constexpr (Intf::kl0FullLoadFlag) {
            self->ctx.kL0FullLoadAl0PingPongFlag = 0;
        }
    }

    if constexpr (Intf::hasNL0IterFlag) {
        self->ctx.loadBL0Flag = true;
        self->ctx.nL0Iter++;
        if (self->ctx.nL0Iter != self->ctx.l12l0LoopN) {
            return true;
        } else {
            self->ctx.nL0Iter = 0;
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

    if ASCEND_IS_AIC_CONV {
        if constexpr (!Intf::isMPreLoad && !Intf::groupOptPreloadFlag) {
            if (self->ctx.kAL1fullload) {
                self->ctx.queueAL1.FreeTensor(self->ctx.al1);
            }
        }
    }

    self->ctx.mAL1Iter++;
    self->ctx.loadAL1Flag = true;
    if (self->ctx.mAL1Iter != self->ctx.ddr2l1LoopM) {
        CalcMDirectionVar<Intf>(self);
        return true;
    } else {
        self->ctx.mAL1Iter = 0;
        CalcMDirectionVar<Intf>(self);
    }

    if constexpr (Intf::isConv3D) {
        self->ctx.dOutIter++;
        if (self->ctx.dOutIter != self->ctx.ddr2l1LoopD) {
            return true;
        } else {
            self->ctx.dOutIter = 0;
        }
    }

    self->ctx.batchIter++;
    if (self->ctx.batchIter != self->ctx.ddr2l1LoopBatch) {
        self->ctx.loadAL1Flag = true;
        self->ctx.loadBL1Flag = false;
        return true;
    } else {
        self->ctx.batchIter = 0;
    }

    if ASCEND_IS_AIC_CONV {
        if constexpr (!Intf::groupOptPreloadFlag && !Intf::isMPreLoad) {
            if (self->ctx.kBL1fullload) {
                self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
            }
        }
    }

    if constexpr (Intf::hasNL1IterFlag) {
        self->ctx.nBL1Iter++;
        CalcCoDirectionVar<Intf>(self);
        self->ctx.loadBL1Flag = true;
        if (self->ctx.nBL1Iter != self->ctx.ddr2l1LoopN) {
            return true;
        }
    }

    if constexpr (Intf::groupOptPreloadFlag) {
        self->ctx.groupOptIter++;
        self->ctx.vecId = (self->ctx.groupOptIter % VEC_NUM) * VEC_ID_MAX;
        self->ctx.nBL1Iter = 0;
        CalcGroupOptParamForMMode<Intf>(self);
        if (self->ctx.groupOptIter < self->ctx.singleGroupOpt - 1) {
            return true;
        } else if (self->ctx.groupOptIter == self->ctx.singleGroupOpt - 1) {
            if (self->ctx.updateSingleCoOpt == 0 && self->ctx.updateEnlarge != self->ctx.convTilingData->convApiTiling.enlarge) {
                return false;
            }
            return true;
        } else {
            self->ctx.loadAL1Flag = false;
        }
    }

    return false;
}

template <class Intf>
__aicore__ inline bool IterateL0NFirstMMode(Intf *self)
{
    if constexpr (Intf::hasNL0IterFlag) {
        self->ctx.nL0Iter++;
        if (self->ctx.nL0Iter != self->ctx.l12l0LoopN) {
            self->ctx.loadAL0Flag = false;
            if ASCEND_IS_AIC_CONV {
                if constexpr (Intf::kl0FullLoadFlag) {
                    self->ctx.kL0FullLoadBl0PingPongFlag =
                        CheckReduceOneKNotSupportDBCase<Intf>(self) ? 0 : self->ctx.kL0FullLoadBl0PingPongFlag;
                }
            }
            return true;
        } else {
            self->ctx.nL0Iter = 0;
        }
    }

    if ASCEND_IS_AIC_CONV {
        if constexpr (Intf::kl0FullLoadFlag) {
            self->ctx.kL0FullLoadBl0PingPongFlag = 0;
        }
    }

    self->ctx.loadAL0Flag = true;
    self->ctx.mL0Iter++;
    if (self->ctx.mL0Iter != self->ctx.l12l0LoopM) {
        return true;
    } else {
        self->ctx.mL0Iter = 0;
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

template <class Intf>
__aicore__ inline bool IterateNFirstMMode(Intf *self)
{
    if (IterateL0NFirstMMode<Intf>(self)) {
        return true;
    }

    if ASCEND_IS_AIC_CONV {
        if constexpr (!(Intf::preFusionFlag && ConvParam::weightTiling == 0)) {
            if (self->ctx.kBL1fullload) {
                self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
            }
        }
    }

    if constexpr (Intf::hasNL1IterFlag) {
        self->ctx.loadBL1Flag = true;
        self->ctx.nBL1Iter++;
        if (self->ctx.nBL1Iter != self->ctx.ddr2l1LoopN) {
            CalcCoDirectionVar<Intf>(self);
            return true;
        } else {
            self->ctx.nBL1Iter = 0;
            CalcCoDirectionVar<Intf>(self);
        }
    }

    if ASCEND_IS_AIC_CONV {
        if constexpr (!Intf::preFusionFlag) {
            if (self->ctx.kAL1fullload) {
                self->ctx.queueAL1.FreeTensor(self->ctx.al1);
            }
        }
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
        if constexpr (!(Intf::preFusionFlag && ConvParam::weightTiling == 0)) {
            self->ctx.loadBL1Flag = true;
        }
        return true;
    }

    return false;
}

};

#endif // CONV_ITERATE_M_MODE_IMPL_H