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
 * \file conv_iterate_impl.h
 * \brief
 */

#ifndef CONV_ITERATE_IMPL_H
#define CONV_ITERATE_IMPL_H

#include "conv_config.h"
#include "conv_framework_util.h"
#include "conv_iterate_base_impl.h"
#include "conv_iterate_hw_mode_impl.h"
#include "conv_iterate_m_mode_impl.h"
#include "conv_util.h"
#if defined(FORMAT_X) && (FORMAT_X == FORMAT_NCDHW || FORMAT_X == FORMAT_NDHWC)
#include "../../conv3d_v2/arch35/conv3d_v2_dequant_impl.h"
#endif

namespace ConvFunc {
using namespace AscendC;
using namespace conv;

CONV_DECLARE_REG_IMPL(Iterate);
CONV_DECLARE_REG_IMPL(PreFusionProcess);

template <class Intf, uint32_t ImplType>
struct Iterate {
    template <bool sync = true>
    static __aicore__ inline bool call(Intf *self, bool enPartialSum = false, bool *updateL1Flag = nullptr)
    {
        if ASCEND_IS_AIC_CONV {
            if constexpr (Intf::preFusionFlag) {
 	            self->ctx.updateL1Flag = false;
 	        }
            if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
                bool ret = IterateImplMMode(self, enPartialSum);
                if constexpr (Intf::preFusionFlag) {
                    if (updateL1Flag != nullptr) {
                        *updateL1Flag = self->ctx.updateL1Flag;
                    }
                }
                return ret;
            } else {
                return IterateImplHWMode(self, enPartialSum);
            }
        }
        if ASCEND_IS_AIV_CONV {
            if constexpr (Intf::groupOptPreloadFlag) {
                return OptGroupPreloadVecImpl<Intf>(self);
            }else if constexpr (Intf::groupOptNDFlag) {
                return OptGroupVecImpl<Intf>(self);
            } else if constexpr (Intf::c04NDFlag) {
                return self->ctx.c04ProcessTools.C04VecImpl(self);
            } else if constexpr (Intf::weightUbTrans) {
                return self->ctx.weightUbProcessTools.WeightUbTransVecImpl(self);
            } else if constexpr (Intf::isDmaFlag) {
                return self->ctx.dmaProcessTools.DmaVecImpl(self);
            } else if constexpr (Intf::isDeQuantFlag) {
#if defined(FORMAT_X) && (FORMAT_X == FORMAT_NCDHW || FORMAT_X == FORMAT_NDHWC)
                return Conv3dFunc::DeQuantVecImpl<Intf>(self);
#endif
            }
        }
        return false;
    }

    static __aicore__ inline bool IterateImplHWMode(Intf* self, bool enPartialSum);
    static __aicore__ inline bool IterateImplMMode(Intf* self, bool enPartialSum);
    static __aicore__ inline void IterateK(Intf* self);
    static __aicore__ inline void IterateBiasScale(Intf* self);
    static __aicore__ inline void SetInitialMmadParams(Intf* self, MmadParams &mmadParams, const uint64_t &currentKL0);
    static __aicore__ inline void SetFinalMmadParams(Intf* self, MmadParams &mmadParams, const uint64_t &currentKL0);
    static __aicore__ inline void ReduceK(Intf* self, MmadParams &mmadParams);
    static __aicore__ inline void ReduceOneK(Intf* self, MmadParams &mmadParams);
    static __aicore__ inline void ReduceKPreloadFmapIter(
        Intf* self, TempIters& tempIters, bool updateIterByFmapTag, const uint64_t& kIter, const uint64_t& multiKAL1);
    static __aicore__ inline void ReduceKPreloadWeightIter(
        Intf* self, TempIters& tempIters, bool updateIterByFmapTag, const uint64_t& kIter, const uint64_t& multiKBL1);
    static __aicore__ inline void ReduceKPreloadFmapIter(
        Intf* self, TempIters& tempIters, bool updateIterByFmapTag, bool ddr2l0LoopKExactDivideMultiKAL1,
        const uint64_t& kIter, const uint64_t& multiKAL1);
    static __aicore__ inline void ReduceKPreloadWeightIter(
        Intf* self, TempIters& tempIters, bool updateIterByFmapTag, bool ddr2l0LoopKExactDivideMultiKBL1,
        const uint64_t& kIter, const uint64_t& multiKBL1);
    static __aicore__ inline void ReduceKPreload(Intf* self, MmadParams &mmadParams);
    static __aicore__ inline void ReduceKFmapPreload(Intf* self, MmadParams &mmadParams);
    static __aicore__ inline void ReduceGroupOptFmapPreload(Intf *self, MmadParams &mmadParams);
    static __aicore__ inline void UpdateNextGroupOptIters(Intf *self, TempIters& tempIters);
    static __aicore__ inline void ReduceKPreloadWithWeightFullloadL0(Intf* self, MmadParams &mmadParams);
    static __aicore__ inline bool UpdateItersByFmap(
        Intf* self, TempIters& tempIters, bool updateIterByFmapTag, const uint64_t& kIter, const uint64_t& multiKAL1);
    static __aicore__ inline bool UpdateItersByWeight(
        Intf* self, TempIters& tempIters, bool updateIterByFmapTag, const uint64_t& kIter, const uint64_t& multiKBL1);
    static __aicore__ inline bool UpdateCommonIters(Intf* self, TempIters& tempIters);
    static __aicore__ inline void ConfigUpdateL1Flag(Intf* self);
};

template <class Intf, uint32_t ImplType>
struct PreFusionProcess {
    template <bool sync = true>
    static __aicore__ inline void call(Intf *self, uint64_t& aL1Offset, AscendC::TEventID& eventIdMte3ToMte1,
                                    AscendC::TEventID& eventIdMte1ToMte3)
    {
        if constexpr (!Intf::preFusionFlag) {
            return;
        }
        if ((self->ctx.convTilingData->convApiTiling.pBufferFlag & AL1_DB_IDX) >> AL1_DB_OFFSET) {
            if (!self->ctx.al1PingPongFlag) {
                eventIdMte3ToMte1 = self->ctx.eventIdMte3ToMte1Ping;
                eventIdMte1ToMte3 = self->ctx.eventIdMte1ToMte3Ping;
            } else {
                eventIdMte3ToMte1 = self->ctx.eventIdMte3ToMte1Pong;
                eventIdMte1ToMte3 = self->ctx.eventIdMte1ToMte3Pong;
            }
            aL1Offset = self->ctx.al1PingPongFlag * self->ctx.convTilingData->convApiTiling.aL1SpaceSize;
        } else {
            eventIdMte3ToMte1 = self->ctx.eventIdMte3ToMte1Ping;
            eventIdMte1ToMte3 = self->ctx.eventIdMte1ToMte3Ping;
            aL1Offset = 0;
        }
        self->ctx.eventIdMte3ToMte1 = eventIdMte3ToMte1;
        self->ctx.eventIdMte1ToMte3 = eventIdMte1ToMte3;
    }
};

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::ReduceOneK(Intf *self, MmadParams &mmadParams)
{
    if (unlikely(self->ctx.loadAL1Flag)) {
        LoadAL1Module<Intf>(self, 0);
        if constexpr (!Intf::preFusionFlag) {
            self->ctx.al1 = self->ctx.queueAL1.template DeQue<typename Intf::FmapT>();
        }
    }
    if (unlikely(self->ctx.loadBL1Flag)) {
        LoadBL1Module<Intf>(self, 0);
        self->ctx.bl1 = self->ctx.queueBL1.template DeQue<typename Intf::WeightT>();
        if constexpr (Intf::isL0BFullLoadable) {
            self->ctx.loadBL0Ins.FullLoadBL0(self->ctx.wholeBl0Tensor[0]);
        }
    }

    uint64_t currentKL0;
    if constexpr (Intf::c04Flag) {
        currentKL0 = self->ctx.loadAL0Ins.GetC04KStepTail();
    } else if constexpr (Intf::k0 == Intf::k0FmapTail) {
        currentKL0 = self->ctx.kL0Tail;
    } else {
        currentKL0 = self->ctx.kAL0Tail;
    }

    SetInitialMmadParams(self, mmadParams, currentKL0);
    if constexpr (!Intf::isInnerBatchFlag && !Intf::isDeQuantFlag) {
        mmadParams.unitFlag = UNIT_FLAG_ENABLE_WITH_FLIP;
    }

    uint64_t kStep = self->ctx.kL0Tail / Intf::k0;
    LoadL0Module<Intf>(self, 0, currentKL0, 0, 0, kStep, mmadParams);

    if constexpr (Intf::preFusionFlag) {
        if (self->ctx.updateL1Flag) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE3>(self->ctx.eventIdMte1ToMte3);
        }
    }
}

template <class Intf, uint32_t ImplType>
__aicore__ bool Iterate<Intf, ImplType>::UpdateCommonIters(Intf *self, TempIters& tempIters)
{
    if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE) &&
        Intf::iterateMFirstFlag) {
        return UpdateCommonItersMModeMFirst<Intf>(self, tempIters);
    } else if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE) &&
        Intf::iterateNFirstFlag) {
        return UpdateCommonItersMModeNFirst<Intf>(self, tempIters);
    } else if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::HW_MODE) &&
        Intf::iterateMFirstFlag) {
        return UpdateCommonItersHWModeMFirst<Intf>(self, tempIters);
    } else if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::HW_MODE) &&
        Intf::iterateNFirstFlag) {
        return UpdateCommonItersHWModeNFirst<Intf>(self, tempIters);
    }
}

template <class Intf, uint32_t ImplType>
__aicore__ bool Iterate<Intf, ImplType>::UpdateItersByFmap(
    Intf* self, TempIters& tempIters, bool updateIterByFmapTag, const uint64_t& kIter, const uint64_t& multiKAL1)
{
    if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::HW_MODE)) {
        tempIters.woAL1Iter = self->ctx.woAL1Iter;
        tempIters.hoAL1Iter = self->ctx.hoAL1Iter;
    } else {
        tempIters.mAL1Iter = self->ctx.mAL1Iter;
    }
    tempIters.batchIter = self->ctx.batchIter;
    self->ctx.kAL1Iter = kIter / multiKAL1 + 1;
    tempIters.kAL1Iter = self->ctx.kAL1Iter;
    if (tempIters.kAL1Iter <= self->ctx.maxKAL1Iter) {
        return false;
    }
    tempIters.kAL1Iter = 0;
    if (updateIterByFmapTag) {
        return UpdateCommonIters(self, tempIters);
    }
    if (tempIters.endTag) {
        return true;
    }
    return false;
}

template <class Intf, uint32_t ImplType>
__aicore__ bool Iterate<Intf, ImplType>::UpdateItersByWeight(
    Intf* self, TempIters& tempIters, bool updateIterByFmapTag, const uint64_t& kIter, const uint64_t& multiKBL1)
{
    tempIters.nBL1Iter = self->ctx.nBL1Iter;
    self->ctx.kBL1Iter = kIter / multiKBL1 + 1;
    tempIters.kBL1Iter = self->ctx.kBL1Iter;
    if (tempIters.kBL1Iter <= self->ctx.maxKBL1Iter) {
        return false;
    }
    tempIters.kBL1Iter = 0;
    if (!updateIterByFmapTag) {
        return UpdateCommonIters(self, tempIters);
    }
    if (tempIters.endTag) {
        return true;
    }
    return false;
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::ReduceKPreloadFmapIter(
    Intf* self, TempIters& tempIters, bool updateIterByFmapTag, const uint64_t& kIter, const uint64_t& multiKAL1)
{
    if (self->ctx.loadAL1Flag || !self->ctx.kAL1fullload) {
        if (kIter % multiKAL1 == 0 && kIter < self->ctx.lastLoopKAL1StartPos) {
            if (kIter != 0) {
                self->ctx.queueAL1.FreeTensor(self->ctx.al1);
            }
            if (!UpdateItersByFmap(self, tempIters, updateIterByFmapTag, kIter, multiKAL1)) {
                LoadAL1BaseModule<Intf>(self, tempIters);
            }
        }
        if ((self->ctx.ddr2l0LoopK % multiKAL1 == 0 &&
            kIter == self->ctx.lastLoopKAL1StartPos) ||
            (self->ctx.ddr2l0LoopK % multiKAL1 != 0 &&
            kIter == self->ctx.lastLoopKAL1StartPosTail)) {
            self->ctx.queueAL1.FreeTensor(self->ctx.al1); // last iter of multiKAL1 only Free AL1 Tensor
        }
        if (kIter % multiKAL1 == 0) {
            self->ctx.al1 = self->ctx.queueAL1.template DeQue<typename Intf::FmapT>();
            self->ctx.loadAL0Flag = true;
        }
    }
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::ReduceKPreloadFmapIter(
    Intf* self, TempIters& tempIters, bool updateIterByFmapTag, bool ddr2l0LoopKExactDivideMultiKAL1,
    const uint64_t& kIter, const uint64_t& multiKAL1)
{
    if (self->ctx.loadAL1Flag || !self->ctx.kAL1fullload) {
        if (kIter % multiKAL1 == 0 && kIter < self->ctx.lastLoopKAL1StartPos) {
            if (kIter != 0) {
                self->ctx.queueAL1.FreeTensor(self->ctx.al1);
            }
            if (!UpdateItersByFmap(self, tempIters, updateIterByFmapTag, kIter, multiKAL1)) {
                LoadAL1BaseModule<Intf>(self, tempIters);
            }
        }
        
        // 使用预先计算的条件
        const bool condition1 = ddr2l0LoopKExactDivideMultiKAL1 &&
                               (kIter == self->ctx.lastLoopKAL1StartPos);
        const bool condition2 = !ddr2l0LoopKExactDivideMultiKAL1 &&
                               (kIter == self->ctx.lastLoopKAL1StartPosTail);
        if (condition1 || condition2) {
            self->ctx.queueAL1.FreeTensor(self->ctx.al1); // last iter of multiKAL1 only Free AL1 Tensor
        }
        
        if (kIter % multiKAL1 == 0) {
            self->ctx.al1 = self->ctx.queueAL1.template DeQue<typename Intf::FmapT>();
            self->ctx.loadAL0Flag = true;
        }
    }
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::ReduceKPreloadWeightIter(
    Intf* self, TempIters& tempIters, bool updateIterByFmapTag, const uint64_t& kIter, const uint64_t& multiKBL1)
{
    if (self->ctx.loadBL1Flag || !self->ctx.kBL1fullload) {
        if (kIter % multiKBL1 == 0 && kIter < self->ctx.lastLoopKBL1StartPos) {
            if (kIter != 0) {
                self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
            }
            if (!UpdateItersByWeight(self, tempIters, updateIterByFmapTag, kIter, multiKBL1)) {
                LoadBL1BaseModule<Intf>(self, tempIters);
            }
        }
        if ((self->ctx.ddr2l0LoopK % multiKBL1 == 0 &&
            kIter == self->ctx.lastLoopKBL1StartPos) ||
            (self->ctx.ddr2l0LoopK % multiKBL1 != 0 &&
            kIter == self->ctx.lastLoopKBL1StartPosTail)) {
            self->ctx.queueBL1.FreeTensor(self->ctx.bl1); // last iter of multiKBL1 only Free BL1 Tensor
        }
        if (kIter % multiKBL1 == 0) {
            self->ctx.bl1 = self->ctx.queueBL1.template DeQue<typename Intf::WeightT>();
            self->ctx.loadBL0Flag = true;
        }
    }
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::ReduceKPreloadWeightIter(
    Intf* self, TempIters& tempIters, bool updateIterByFmapTag, bool ddr2l0LoopKExactDivideMultiKBL1,
    const uint64_t& kIter, const uint64_t& multiKBL1)
{
    if (self->ctx.loadBL1Flag || !self->ctx.kBL1fullload) {
        if (kIter % multiKBL1 == 0 && kIter < self->ctx.lastLoopKBL1StartPos) {
            if (kIter != 0) {
                self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
            }
            if (!UpdateItersByWeight(self, tempIters, updateIterByFmapTag, kIter, multiKBL1)) {
                LoadBL1BaseModule<Intf>(self, tempIters);
            }
        }
        
        // 使用预先计算的条件
        const bool condition1 = ddr2l0LoopKExactDivideMultiKBL1 &&
                               (kIter == self->ctx.lastLoopKBL1StartPos);
        const bool condition2 = !ddr2l0LoopKExactDivideMultiKBL1 &&
                               (kIter == self->ctx.lastLoopKBL1StartPosTail);
        if (condition1 || condition2) {
            self->ctx.queueBL1.FreeTensor(self->ctx.bl1); // last iter of multiKBL1 only Free BL1 Tensor
        }
        
        if (kIter % multiKBL1 == 0) {
            self->ctx.bl1 = self->ctx.queueBL1.template DeQue<typename Intf::WeightT>();
            self->ctx.loadBL0Flag = true;
        }
    }
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::SetInitialMmadParams(Intf *self, MmadParams &mmadParams, const uint64_t &currentKL0)
{
    mmadParams.k = currentKL0;
    const bool eb = self->ctx.enableBias;
    if constexpr (Intf::isDeQuantFlag) {
        mmadParams.cmatrixInitVal = true;
        mmadParams.cmatrixSource  = false;
    } else {
        mmadParams.cmatrixInitVal = !eb;
        mmadParams.cmatrixSource  = eb;
    }
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::SetFinalMmadParams(Intf *self, MmadParams &mmadParams, const uint64_t &currentKL0)
{
    mmadParams.k = currentKL0;
    if constexpr (!Intf::isInnerBatchFlag && !Intf::isDeQuantFlag) {
        mmadParams.unitFlag = UNIT_FLAG_ENABLE_WITH_FLIP;
    }
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::ReduceKPreload(Intf *self, MmadParams &mmadParams)
{
    // updateIterByFmapTag is true means fm update; false means weight update
    bool updateIterByFmapTag = self->ctx.convTilingData->convApiTiling.kAL1 > self->ctx.convTilingData->convApiTiling.kBL1;
    if (self->ctx.kAL1fullload && !self->ctx.kBL1fullload) {
        updateIterByFmapTag = false;
    } else if (!self->ctx.kAL1fullload && self->ctx.kBL1fullload) {
        updateIterByFmapTag = true;
    }

    if (self->ctx.loadAL1Flag || !self->ctx.kAL1fullload) {
        self->ctx.kAL1Iter = 0;
        LoadAL1BaseModule<Intf>(self);
        if (self->ctx.kAL1fullload) {
            self->ctx.al1 = self->ctx.queueAL1.template DeQue<typename Intf::FmapT>();
            self->ctx.loadAL0Flag = true;
        }
    }
    if (self->ctx.loadBL1Flag || !self->ctx.kBL1fullload) {
        self->ctx.kBL1Iter = 0;
        LoadBL1BaseModule<Intf>(self);
        if (self->ctx.kBL1fullload) {
            self->ctx.bl1 = self->ctx.queueBL1.template DeQue<typename Intf::WeightT>();
            self->ctx.loadBL0Flag = true;
            if constexpr (Intf::isL0BFullLoadable) {
                self->ctx.loadBL0Ins.FullLoadBL0(self->ctx.wholeBl0Tensor[0]);
            }
        }
    }
    // state
    TempIters tempIters;

    uint64_t currentKL0 = self->ctx.convTilingData->convApiTiling.kL0;
    uint64_t posK = 0;
    uint64_t KStartPosition = 0;
    uint64_t kStep = self->ctx.convTilingData->convApiTiling.kStep;
    uint64_t multiKAL1 = self->ctx.multiKAL1;
    uint64_t multiKBL1 = self->ctx.multiKBL1;

    uint64_t kIter = 0;
    uint64_t ddr2l0LoopK = self->ctx.ddr2l0LoopK;
    uint64_t maxKInterLoop = ddr2l0LoopK - 1;

    SetInitialMmadParams(self, mmadParams, currentKL0);
    if constexpr (!Intf::isInnerBatchFlag && !Intf::isDeQuantFlag) {
        mmadParams.unitFlag = UNIT_FLAG_ENABLE_ONLY;
    }

    // 预先计算循环不变的条件
    const bool ddr2l0LoopKExactDivideMultiKAL1 = (ddr2l0LoopK % multiKAL1 == 0);
    const bool ddr2l0LoopKExactDivideMultiKBL1 = (ddr2l0LoopK % multiKBL1 == 0);

    if (kIter < maxKInterLoop) {
        ReduceKPreloadFmapIter(self, tempIters, updateIterByFmapTag, ddr2l0LoopKExactDivideMultiKAL1, kIter, multiKAL1);
        ReduceKPreloadWeightIter(self, tempIters, updateIterByFmapTag, ddr2l0LoopKExactDivideMultiKBL1, kIter, multiKBL1);
        LoadL0Module<Intf>(self, kIter, currentKL0, posK, KStartPosition, kStep, mmadParams);
        kIter++;

        mmadParams.cmatrixInitVal = false;
        mmadParams.cmatrixSource  = false;
    }

    while (kIter < maxKInterLoop) {
        ReduceKPreloadFmapIter(self, tempIters, updateIterByFmapTag, ddr2l0LoopKExactDivideMultiKAL1, kIter, multiKAL1);
        ReduceKPreloadWeightIter(self, tempIters, updateIterByFmapTag, ddr2l0LoopKExactDivideMultiKBL1, kIter, multiKBL1);
        posK = (kIter % multiKAL1) * currentKL0;
        KStartPosition = (kIter % multiKBL1) * kStep;
        LoadL0Module<Intf>(self, kIter, currentKL0, posK, KStartPosition, kStep, mmadParams);
        kIter++;
    }
    
    posK = (kIter % multiKAL1) * currentKL0;
    if constexpr (Intf::c04Flag) {
        currentKL0 = self->ctx.loadAL0Ins.GetC04KStepTail();
    } else if constexpr (Intf::k0 == Intf::k0FmapTail) {
        currentKL0 = self->ctx.kL0Tail;
    } else {
        currentKL0 = self->ctx.kAL0Tail;
    }

    SetFinalMmadParams(self, mmadParams, currentKL0);
    
    KStartPosition = (kIter % multiKBL1) * kStep;
    kStep = self->ctx.kL0Tail / Intf::k0;

    ReduceKPreloadFmapIter(self, tempIters, updateIterByFmapTag, ddr2l0LoopKExactDivideMultiKAL1, kIter, multiKAL1);
    ReduceKPreloadWeightIter(self, tempIters, updateIterByFmapTag, ddr2l0LoopKExactDivideMultiKBL1, kIter, multiKBL1);
    LoadL0Module<Intf>(self, kIter, currentKL0, posK, KStartPosition, kStep, mmadParams);
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::ReduceKFmapPreload(Intf *self, MmadParams &mmadParams)
{
    // updateIterByFmapTag is true means fm update; false means weight update
    bool updateIterByFmapTag = self->ctx.convTilingData->convApiTiling.kAL1 > self->ctx.convTilingData->convApiTiling.kBL1;
    if (self->ctx.kAL1fullload && !self->ctx.kBL1fullload) {
        updateIterByFmapTag = false;
    } else if (!self->ctx.kAL1fullload && self->ctx.kBL1fullload) {
        updateIterByFmapTag = true;
    }
    if (self->ctx.loadAL1Flag || !self->ctx.kAL1fullload) {
        self->ctx.kAL1Iter = 0;
        LoadAL1BaseModule<Intf>(self);
        if (self->ctx.kAL1fullload) {
            self->ctx.al1 = self->ctx.queueAL1.template DeQue<typename Intf::FmapT>();
            self->ctx.loadAL0Flag = true;
        }
    }

    // state
    TempIters tempIters;

    uint64_t currentKL0 = self->ctx.convTilingData->convApiTiling.kL0;
    uint64_t posK = 0;
    uint64_t KStartPosition = 0;
    uint64_t kStep = self->ctx.convTilingData->convApiTiling.kStep;
    uint64_t multiKAL1 = self->ctx.multiKAL1;
    uint64_t multiKBL1 = self->ctx.multiKBL1;

    uint64_t kIter = 0;
    uint64_t maxKInterLoop = self->ctx.ddr2l0LoopK - 1;

    SetInitialMmadParams(self, mmadParams, currentKL0);

    if (kIter < maxKInterLoop) {
        if constexpr (!Intf::isInnerBatchFlag && !Intf::isDeQuantFlag) {
            mmadParams.unitFlag = UNIT_FLAG_ENABLE_ONLY;
        }
        ReduceKPreloadFmapIter(self, tempIters, updateIterByFmapTag, kIter, multiKAL1);
        LoadBL1Module<Intf>(self, kIter);
        self->ctx.loadBL0Flag = true;
        LoadL0Module<Intf>(self, kIter, currentKL0, posK, KStartPosition, kStep, mmadParams);
        kIter++;

        mmadParams.cmatrixInitVal = false;
        mmadParams.cmatrixSource  = false;
    }

    while (kIter < maxKInterLoop) {
        ReduceKPreloadFmapIter(self, tempIters, updateIterByFmapTag, kIter, multiKAL1);

        if (self->ctx.loadBL1Flag || kIter % multiKBL1 == 0) {
            LoadBL1Module<Intf>(self, kIter);
            self->ctx.loadBL0Flag = true;
        }
        posK = (kIter % multiKAL1) * currentKL0;
        KStartPosition = (kIter % multiKBL1) * kStep;

        LoadL0Module<Intf>(self, kIter, currentKL0, posK, KStartPosition, kStep, mmadParams);
        kIter++;
    }

    ReduceKPreloadFmapIter(self, tempIters, updateIterByFmapTag, kIter, multiKAL1);

    if (self->ctx.loadBL1Flag || kIter % multiKBL1 == 0) {
        LoadBL1Module<Intf>(self, kIter);
        self->ctx.loadBL0Flag = true;
    }

    posK = (kIter % multiKAL1) * currentKL0;
    if constexpr (Intf::c04Flag) {
        currentKL0 = self->ctx.loadAL0Ins.GetC04KStepTail();
    } else if constexpr (Intf::k0 == Intf::k0FmapTail) {
        currentKL0 = self->ctx.kL0Tail;
    } else {
        currentKL0 = self->ctx.kAL0Tail;
    }
    
    SetFinalMmadParams(self, mmadParams, currentKL0);

    KStartPosition = (kIter % multiKBL1) * kStep;
    kStep = self->ctx.kL0Tail / Intf::k0;

    LoadL0Module<Intf>(self, kIter, currentKL0, posK, KStartPosition, kStep, mmadParams);
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::ReduceKPreloadWithWeightFullloadL0(Intf *self, MmadParams &mmadParams)
{
    // Currently, MMode is supported. HWMode MFirst will be supported in the future.
    if (self->ctx.loadAL1Flag) {
        if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
            if constexpr (Intf::isNoPad) {
                self->ctx.loadAl1Ins.SetLoad3dFMatrixNoPad(self->ctx.convTilingData->convApiTiling.orgWi);
            } else {
                self->ctx.loadAl1Ins.SetLoad3dFMatrix(self->ctx.convTilingData->convApiTiling.padLeft,
                                                    self->ctx.convTilingData->convApiTiling.padRight,
                                                    self->ctx.convTilingData->convApiTiling.orgWi);
            }
        }
        // If the current m direction iterator is not the last one, load the next fmap block to be processed.
        if (self->ctx.mAL1Iter < self->ctx.maxMAL1Iter) {
            self->ctx.al1 = self->ctx.queueAL1.template AllocTensor<typename Intf::FmapT>();
            if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
                self->ctx.loadAl1Ins.LoadAL1(0, self->ctx.mAL1Iter + 1, self->ctx.batchIter);
            }
            self->ctx.queueAL1.EnQue(self->ctx.al1);
        } else if (self->ctx.batchIter < self->ctx.ddr2l1LoopBatch - 1) {
            self->ctx.al1 = self->ctx.queueAL1.template AllocTensor<typename Intf::FmapT>();
            if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
                self->ctx.loadAl1Ins.LoadAL1(0, 0, self->ctx.batchIter + 1); // load -> pong
            }
            self->ctx.queueAL1.EnQue(self->ctx.al1);
        }
        self->ctx.al1 = self->ctx.queueAL1.template DeQue<typename Intf::FmapT>();
        self->ctx.loadAL1Flag = false;
    }
    self->ctx.loadAL0Flag = true;

    uint64_t currentKL0 = self->ctx.convTilingData->convApiTiling.kL0;
    uint64_t posK = 0;
    uint64_t KStartPosition = 0;
    uint64_t kStep = self->ctx.convTilingData->convApiTiling.kStep;
    uint64_t multiKAL1 = self->ctx.multiKAL1;
    uint64_t multiKBL1 = self->ctx.multiKBL1;

    uint64_t kIter = 0;
    uint64_t maxKInterLoop = self->ctx.ddr2l0LoopK - 1;

    SetInitialMmadParams(self, mmadParams, currentKL0);

    if (kIter < maxKInterLoop) {
        if constexpr (!Intf::isInnerBatchFlag && !Intf::isDeQuantFlag) {
            mmadParams.unitFlag = UNIT_FLAG_ENABLE_ONLY;
        }
        LoadL0Module<Intf>(self, kIter, currentKL0, posK, KStartPosition, kStep, mmadParams);
        kIter++;

        mmadParams.cmatrixInitVal = false;
        mmadParams.cmatrixSource  = false;
    }

    while (kIter < maxKInterLoop) {
        posK = (kIter % multiKAL1) * currentKL0;
        KStartPosition = (kIter % multiKBL1) * kStep;
        LoadL0Module<Intf>(self, kIter, currentKL0, posK, KStartPosition, kStep, mmadParams);
        kIter++;
    }

    posK = (kIter % multiKAL1) * currentKL0;
    if constexpr (Intf::c04Flag) {
        currentKL0 = self->ctx.loadAL0Ins.GetC04KStepTail();
    } else if constexpr (Intf::k0 == Intf::k0FmapTail) {
        currentKL0 = self->ctx.kL0Tail;
    } else {
        currentKL0 = self->ctx.kAL0Tail;
    }

    SetFinalMmadParams(self, mmadParams, currentKL0);

    KStartPosition = (kIter % multiKBL1) * kStep;
    kStep = self->ctx.kL0Tail / Intf::k0;
   
    LoadL0Module<Intf>(self, kIter, currentKL0, posK, KStartPosition, kStep, mmadParams);
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::ReduceK(Intf *self, MmadParams &mmadParams)
{
    // state
    uint64_t currentKL0 = self->ctx.convTilingData->convApiTiling.kL0;
    uint64_t posK = 0;
    uint64_t KStartPosition = 0;
    uint64_t kStep = self->ctx.convTilingData->convApiTiling.kStep;
    uint64_t multiKAL1 = self->ctx.multiKAL1;
    uint64_t multiKBL1 = self->ctx.multiKBL1;

    uint64_t kIter = 0;
    const uint64_t maxKInterLoop = self->ctx.ddr2l0LoopK - 1;

    SetInitialMmadParams(self, mmadParams, currentKL0);

    if (self->ctx.loadAL1Flag || (!self->ctx.kAL1fullload)) {
        LoadAL1Module<Intf>(self, kIter);
        if constexpr (!Intf::preFusionFlag) {
            self->ctx.al1 = self->ctx.queueAL1.template DeQue<typename Intf::FmapT>();
        }
        self->ctx.loadAL0Flag = true;
    }
    if constexpr (Intf::weightUbTrans) {
        LoadBL1Module<Intf>(self, kIter);
        self->ctx.loadBL0Flag = true;
    } else {
        if (self->ctx.loadBL1Flag || (!self->ctx.kBL1fullload)) {
            LoadBL1Module<Intf>(self, kIter);
            self->ctx.bl1 = self->ctx.queueBL1.template DeQue<typename Intf::WeightT>();
            self->ctx.loadBL0Flag = true;
        }
    }

    if (kIter < maxKInterLoop) {
        if constexpr (!Intf::isInnerBatchFlag && !Intf::isDeQuantFlag) {
            mmadParams.unitFlag = UNIT_FLAG_ENABLE_ONLY;
        }
        LoadL0Module<Intf>(self, kIter, currentKL0, posK, KStartPosition, kStep, mmadParams);
        kIter++;
        posK+=currentKL0;
        KStartPosition+=kStep;

        mmadParams.cmatrixInitVal = false;
        mmadParams.cmatrixSource  = false;
    }

    uint64_t loadAkIter = self->ctx.kAL1fullload ? self->ctx.ddr2l0LoopK : multiKAL1;
    uint64_t loadBkIter = self->ctx.kBL1fullload ? self->ctx.ddr2l0LoopK : multiKBL1;
    if constexpr (Intf::weightUbTrans) { loadBkIter = multiKBL1; }

    while (kIter < maxKInterLoop) {
        if (kIter == loadAkIter) {
            if constexpr (!Intf::preFusionFlag) {
                self->ctx.queueAL1.FreeTensor(self->ctx.al1);
            }
            LoadAL1Module<Intf>(self, kIter);
            if constexpr (!Intf::preFusionFlag) {
                self->ctx.al1 = self->ctx.queueAL1.template DeQue<typename Intf::FmapT>();
            }
            loadAkIter += multiKAL1;
            self->ctx.loadAL0Flag = true;
            posK = 0;
        }
        if (kIter == loadBkIter) {
            if constexpr (!Intf::weightUbTrans) {
                self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
            }
            LoadBL1Module<Intf>(self, kIter);
            if constexpr (!Intf::weightUbTrans) {
                self->ctx.bl1 = self->ctx.queueBL1.template DeQue<typename Intf::WeightT>();
            }
            loadBkIter += multiKBL1;
            self->ctx.loadBL0Flag = true;
            KStartPosition = 0;
        }

        uint64_t segmentEnd = loadAkIter < loadBkIter ? (loadAkIter < maxKInterLoop ? loadAkIter : maxKInterLoop) : (loadBkIter < maxKInterLoop ? loadBkIter : maxKInterLoop);
        while (kIter < segmentEnd) {
            LoadL0Module<Intf>(self, kIter, currentKL0, posK, KStartPosition, kStep, mmadParams);
            posK+=currentKL0;
            KStartPosition+=kStep;
            kIter++;
        }
    }

    posK = (kIter % multiKAL1) * currentKL0;
    if constexpr (Intf::c04Flag) {
        currentKL0 = self->ctx.loadAL0Ins.GetC04KStepTail();
    } else if constexpr (Intf::k0 == Intf::k0FmapTail) {
        currentKL0 = self->ctx.kL0Tail;
    } else {
        currentKL0 = self->ctx.kAL0Tail;
    }

    SetFinalMmadParams(self, mmadParams, currentKL0);

    KStartPosition = (kIter % multiKBL1) * kStep;
    kStep = self->ctx.kL0Tail / Intf::k0;

    if (kIter == loadAkIter) {
        if constexpr (!Intf::preFusionFlag) {
            self->ctx.queueAL1.FreeTensor(self->ctx.al1);
        }
        LoadAL1Module<Intf>(self, kIter);
        if constexpr (!Intf::preFusionFlag) {
            self->ctx.al1 = self->ctx.queueAL1.template DeQue<typename Intf::FmapT>();
        }
        self->ctx.loadAL0Flag = true;
        posK = 0;
    }
    if (kIter == loadBkIter) {
        if constexpr (!Intf::weightUbTrans) {
            self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
        }
        LoadBL1Module<Intf>(self, kIter);
        if constexpr (!Intf::weightUbTrans) {
            self->ctx.bl1 = self->ctx.queueBL1.template DeQue<typename Intf::WeightT>();
        }
        self->ctx.loadBL0Flag = true;
        KStartPosition = 0;
    }

    LoadL0Module<Intf>(self, kIter, currentKL0, posK, KStartPosition, kStep, mmadParams);

    if constexpr (Intf::preFusionFlag) {
        if (self->ctx.updateL1Flag) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE3>(self->ctx.eventIdMte1ToMte3);
        }
    }
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::UpdateNextGroupOptIters(Intf *self, TempIters& tempIters)
{
    tempIters.batchIter = self->ctx.batchIter;
    tempIters.groupOptIter = self->ctx.groupOptIter;
    if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
        tempIters.mAL1Iter = self->ctx.mAL1Iter + 1;
        if (tempIters.mAL1Iter <= self->ctx.maxMAL1Iter) {
            return;
        }
        tempIters.mAL1Iter = 0;
    } else {
        tempIters.woAL1Iter = self->ctx.woAL1Iter + 1;
        if (tempIters.woAL1Iter <= self->ctx.maxWoL1Iter) {
            return;
        }
        tempIters.woAL1Iter = 0;
        tempIters.hoAL1Iter = self->ctx.hoAL1Iter + 1;
        if (tempIters.hoAL1Iter <= self->ctx.maxHoL1Iter) {
            return;
        }
        tempIters.hoAL1Iter = 0;
    }

    tempIters.batchIter = self->ctx.batchIter + 1;
    if (tempIters.batchIter < self->ctx.ddr2l1LoopBatch) {
        return;
    }
    tempIters.batchIter = 0;

    tempIters.groupOptIter = self->ctx.groupOptIter + 1;
    if (tempIters.groupOptIter == self->ctx.singleGroupOpt) {
        tempIters.groupOptIter = 0;
        tempIters.endTag = true;
    } else if (tempIters.groupOptIter == self->ctx.singleGroupOpt - 1 && self->ctx.updateSingleCoOpt == 0) {
        tempIters.endTag = true;
    }
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::ReduceGroupOptFmapPreload(Intf *self, MmadParams &mmadParams)
{
    self->ctx.loadAl1Ins.SetLoad3dFMatrixForOptPreload();
    TempIters tempIters;
    UpdateNextGroupOptIters(self, tempIters);
    if (self->ctx.loadAL1Flag && !tempIters.endTag) {
        self->ctx.kAL1Iter = 0;
        self->ctx.mAL1UpdateFlag = true;
        LoadAL1BaseModule<Intf>(self, tempIters);
    }
    self->ctx.al1 = self->ctx.queueAL1.template DeQue<typename Intf::FmapT>();
    self->ctx.loadAL0Flag = true;
    if (self->ctx.loadBL1Flag) {
        LoadBL1Module<Intf>(self, 0);
        self->ctx.loadBL0Flag = true;
    }

    // state
    uint64_t currentKL0 = self->ctx.convTilingData->convApiTiling.kL0;
    uint64_t posK = 0;
    uint64_t KStartPosition = 0;
    uint64_t kStep = self->ctx.convTilingData->convApiTiling.kStep;
    uint64_t multiKAL1 = self->ctx.multiKAL1;
    uint64_t multiKBL1 = self->ctx.multiKBL1;
 
    uint64_t kIter = 0;
    uint64_t ddr2l0LoopK = self->ctx.ddr2l0LoopK;
    uint64_t maxKInterLoop = ddr2l0LoopK - 1;

    SetInitialMmadParams(self, mmadParams, currentKL0);
    if constexpr (!Intf::isInnerBatchFlag && !Intf::isDeQuantFlag) {
        mmadParams.unitFlag = UNIT_FLAG_ENABLE_ONLY;
    }

    // 预先计算循环不变的条件
    const bool ddr2l0LoopKExactDivideMultiKAL1 = (ddr2l0LoopK % multiKAL1 == 0);
    const bool ddr2l0LoopKExactDivideMultiKBL1 = (ddr2l0LoopK % multiKBL1 == 0);

    if (kIter < maxKInterLoop) {
        LoadL0Module<Intf>(self, kIter, currentKL0, posK, KStartPosition, kStep, mmadParams);
        kIter++;
        mmadParams.cmatrixInitVal = false;
        mmadParams.cmatrixSource  = false;
    }

    uint16_t al0PingPongFlag = 0;
    uint16_t bl0PingPongFlag = 0;
    while (kIter < maxKInterLoop) {
        posK = (kIter % multiKAL1) * currentKL0;
        KStartPosition = (kIter % multiKBL1) * kStep;
        LoadL0Module<Intf>(self, kIter, currentKL0, posK, KStartPosition, kStep, mmadParams);
        kIter++;
    }

    posK = (kIter % multiKAL1) * currentKL0;
    if constexpr (Intf::c04Flag) {
        currentKL0 = self->ctx.loadAL0Ins.GetC04KStepTail();
    } else if constexpr (Intf::k0 == Intf::k0FmapTail) {
        currentKL0 = self->ctx.kL0Tail;
    } else {
        currentKL0 = self->ctx.kAL0Tail;
    }

    SetFinalMmadParams(self, mmadParams, currentKL0);
 
    KStartPosition = (kIter % multiKBL1) * kStep;
    kStep = self->ctx.kL0Tail / Intf::k0;
   
    LoadL0Module<Intf>(self, kIter, currentKL0, posK, KStartPosition, kStep, mmadParams);
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::IterateK(Intf *self)
{
    MmadParams mmadParams;
    SetMNBeforeIterateK<Intf>(self, mmadParams);
    SetFirstBeforeIterateK<Intf>(self);
    if constexpr (!Intf::isDeQuantFlag) {
        IterateBiasScale(self);
    }

    if constexpr (Intf::isInnerBatchFlag || Intf::isDeQuantFlag) {
        self->ctx.cl0 = self->ctx.queueCL0.template AllocTensor<typename Intf::L0cT>();
    } else {
        if ((self->ctx.convTilingData->convApiTiling.pBufferFlag & 0x04) >> 2) { // cl0 db
            self->ctx.cl0 =
                self->ctx.wholeCl0Tensor[(self->ctx.cl0PingPongFlag & 0x1) * L0C_HALF_SIZE / Intf::sizeOfL0c];
        } else {
            self->ctx.cl0 = self->ctx.wholeCl0Tensor;
        }
    }

    // reduceK priority: 1.KL0FullLoad 2.L1DB Preload 3. ordinary reduceK
    if constexpr (Intf::groupOptPreloadFlag) {
        ReduceGroupOptFmapPreload(self, mmadParams);
    } else if constexpr (Intf::kl0FullLoadFlag) {
        ReduceOneK(self, mmadParams);
    } else if constexpr (Intf::isMPreLoad) {
        ReduceKPreloadWithWeightFullloadL0(self, mmadParams);
    } else if constexpr (Intf::kPreLoadFlag) {
        if constexpr (Intf::weightUbTrans) {
            ReduceKFmapPreload(self, mmadParams);
        } else {
            ReduceKPreload(self, mmadParams);
        }
    } else {
        ReduceK(self, mmadParams);
    }

    FreeL1Tensor<Intf>(self);

    if constexpr (Intf::isInnerBatchFlag || Intf::isDeQuantFlag) {
        self->ctx.queueCL0.EnQue(self->ctx.cl0);
        self->ctx.cl0 = self->ctx.queueCL0.template DeQue<typename Intf::L0cT>();
    } else {
        self->ctx.cl0PingPongFlag++;
    }
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::IterateBiasScale(Intf *self)
{
    if (self->ctx.enableBias) {
        if (!self->ctx.convTilingData->convApiTiling.biasFullLoadFlag) {
            self->ctx.biasL1 = self->ctx.queueBiasL1.template AllocTensor<typename Intf::BiasT>();
            self->ctx.loadBiasL1Ins.LoadChannelWiseL1(self->ctx.biasL1, self->ctx.biasgm);
            self->ctx.queueBiasL1.EnQue(self->ctx.biasL1);
            self->ctx.biasL1 = self->ctx.queueBiasL1.template DeQue<typename Intf::BiasT>();
        }
        self->ctx.biasBT = self->ctx.queueBiasBT.template AllocTensor<typename Intf::L0cT>();
        self->ctx.loadBiasBTIns.LoadBiasBt(self->ctx.biasBT, self->ctx.biasL1);
        self->ctx.queueBiasBT.EnQue(self->ctx.biasBT);
        self->ctx.biasBT = self->ctx.queueBiasBT.template DeQue<typename Intf::L0cT>();
    }

    if (!self->ctx.convTilingData->convApiTiling.fixpParamsFullLoadFlag) {
        if (self->ctx.enableVectorQuant || self->ctx.enableVectorRelu) {
            if constexpr (Intf::groupType != static_cast<int8_t>(ConvGroupType::NORMAL_CONV)) {
                event_t eventId = static_cast<event_t>(self->ctx.pipe.FetchEventID(HardEvent::FIX_MTE2));
                SetFlag<HardEvent::FIX_MTE2>(eventId);
                WaitFlag<HardEvent::FIX_MTE2>(eventId);
            }
        }
        if (self->ctx.enableVectorQuant) {
            self->ctx.scaleL1 = self->ctx.queueScaleL1.template AllocTensor<typename Intf::ScaleT>();
            if constexpr (Intf::isExtendConv2d) {
                if (self->ctx.convTilingData->convApiTiling.quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                    self->ctx.loadScaleL1Ins.LoadChannelWiseL1(self->ctx.scaleL1, self->ctx.scalegm);
                }
                if (self->ctx.convTilingData->convApiTiling.dualOutput &&
                    self->ctx.convTilingData->convApiTiling.quantMode1 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                    self->ctx.loadScaleL1Ins.LoadChannelWiseL1(self->ctx.scaleL1[self->ctx.scale1L1offset],
                        self->ctx.scale1gm);
                }
            } else {
                self->ctx.loadScaleL1Ins.LoadChannelWiseL1(self->ctx.scaleL1, self->ctx.scalegm);
            }
            self->ctx.queueScaleL1.EnQue(self->ctx.scaleL1);
            self->ctx.scaleL1 = self->ctx.queueScaleL1.template DeQue<typename Intf::ScaleT>();
        }
        if constexpr (Intf::isExtendConv2d) {
            if (self->ctx.enableVectorRelu) {
                self->ctx.reluWeightL1 = self->ctx.queueReluWeightL1.template AllocTensor<typename Intf::ReluWeightT>();
                if (self->ctx.convTilingData->convApiTiling.reluMode0 == static_cast<uint8_t>(ReluMode::VECTOR_RELU)) {
                    self->ctx.loadReluWeightL1Ins.LoadChannelWiseL1(self->ctx.reluWeightL1, self->ctx.reluWeightGM);
                }
                if (self->ctx.convTilingData->convApiTiling.dualOutput &&
                    self->ctx.convTilingData->convApiTiling.reluMode1 == static_cast<uint8_t>(ReluMode::VECTOR_RELU)) {
                    self->ctx.loadReluWeightL1Ins.LoadChannelWiseL1(self->ctx.reluWeightL1[self->ctx.reluWeight1L1offset],
                        self->ctx.reluWeight1GM);
                }
                self->ctx.queueReluWeightL1.EnQue(self->ctx.reluWeightL1);
                self->ctx.reluWeightL1 = self->ctx.queueReluWeightL1.template DeQue<typename Intf::ReluWeightT>();
            }
        }
    }
}

template <class Intf, uint32_t ImplType>
__aicore__ bool Iterate<Intf, ImplType>::IterateImplHWMode(Intf *self, bool enPartialSum)
{
    if (self->ctx.isFirstIterate) {
        FirstIterateImplHWMode<Intf>(self);
    } else if constexpr (Intf::iterateMFirstFlag) {
        if (IterateMFirstHWMode<Intf>(self) == false) {
            return false;
        }
    } else if constexpr (Intf::iterateNFirstFlag) {
        if (IterateNFirstHWMode<Intf>(self) == false) {
            return false;
        }
    }

    UpdateHoL0WoL0<Intf>(self);

    IterateK(self);

    return true;
}

template <class Intf, uint32_t ImplType>
__aicore__ bool Iterate<Intf, ImplType>::IterateImplMMode(Intf *self, bool enPartialSum)
{
    if (self->ctx.isFirstIterate) {
        FirstIterateImplMMode(self);
    } else if constexpr (Intf::iterateMFirstFlag) {
        if (IterateMFirstMMode(self) == false) {
            return false;
        }
    } else if constexpr (Intf::iterateNFirstFlag) {
        if (IterateNFirstMMode(self) == false) {
            return false;
        }
    }

    if constexpr (Intf::preFusionFlag) {
        ConfigUpdateL1Flag(self);
    }
    IterateK(self);

    return true;
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::ConfigUpdateL1Flag(Intf *self)
{
    if constexpr (Intf::preFusionFlag) {
        if constexpr (Intf::hasNL1IterFlag) {
            if ((self->ctx.batchIter + 1 <= self->ctx.ddr2l1LoopBatch) && (self->ctx.mAL1Iter + 1 <= self->ctx.ddr2l1LoopM)) {
                if ((self->ctx.mL0Iter + 1 == self->ctx.l12l0LoopM) && (self->ctx.nBL1Iter + 1 == self->ctx.ddr2l1LoopN)) {
                    self->ctx.updateL1Flag = true;
                }
                if ((self->ctx.batchIter + 1 == self->ctx.ddr2l1LoopBatch) && (self->ctx.mAL1Iter + 1 == self->ctx.ddr2l1LoopM)) {
                    self->ctx.updateL1Flag = false;
                }
            }
        } else {
            if ((self->ctx.batchIter + 1 <= self->ctx.ddr2l1LoopBatch) && (self->ctx.mAL1Iter + 1 <= self->ctx.ddr2l1LoopM)) {
                if (self->ctx.mL0Iter + 1 == self->ctx.l12l0LoopM) {
                    self->ctx.updateL1Flag = true;
                }
                if ((self->ctx.batchIter + 1 == self->ctx.ddr2l1LoopBatch) && (self->ctx.mAL1Iter + 1 == self->ctx.ddr2l1LoopM)) {
                    self->ctx.updateL1Flag = false;
                }
            }
        }
    }
}

};


#endif // CONV_ITERATE_IMPL_H