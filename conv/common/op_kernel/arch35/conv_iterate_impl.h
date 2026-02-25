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

template <class Intf, uint32_t ImplType>
struct Iterate {
    template <bool sync = true>
    static __aicore__ inline bool call(Intf *self, bool enPartialSum = false)
    {
        if ASCEND_IS_AIC_CONV {
            if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
                return IterateImplMMode(self, enPartialSum);
            } else {
                return IterateImplHWMode(self, enPartialSum);
            }
        }
        if ASCEND_IS_AIV_CONV {
            if constexpr (Intf::groupOptNDFlag) {
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

    static __aicore__ inline bool IterateImplHWMode(Intf *self, bool enPartialSum);
    static __aicore__ inline bool IterateImplMMode(Intf *self, bool enPartialSum);
    static __aicore__ inline void IterateK(Intf *self);
    static __aicore__ inline void IterateBiasScale(Intf *self);
    static __aicore__ inline void ReduceK(Intf *self);
    static __aicore__ inline void ReduceOneK(Intf *self);
    static __aicore__ inline void ReduceKPreloadFmapIter(Intf *self, TempIters& tempIters, bool updateIterByFmapTag);
    static __aicore__ inline void ReduceKPreloadWeightIter(Intf *self, TempIters& tempIters, bool updateIterByFmapTag);
    static __aicore__ inline void ReduceKPreload(Intf *self);
    static __aicore__ inline void ReduceKFmapPreload(Intf *self);
    static __aicore__ inline bool UpdateItersByFmap(Intf *self, TempIters& tempIters, bool updateIterByFmapTag);
    static __aicore__ inline bool UpdateItersByWeight(Intf *self, TempIters& tempIters, bool updateIterByFmapTag);
    static __aicore__ inline bool UpdateCommonIters(Intf *self, TempIters& tempIters);
};

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::ReduceOneK(Intf *self)
{
    if (unlikely(self->ctx.loadAL1Flag)) {
        LoadAL1Moudle<Intf>(self);
        self->ctx.al1 = self->ctx.queueAL1.template DeQue<typename Intf::FmapT>();
    }
    if (unlikely(self->ctx.loadBL1Flag)) {
        LoadBL1Moudle<Intf>(self);
        self->ctx.bl1 = self->ctx.queueBL1.template DeQue<typename Intf::WeightT>();
    }

    LoadL0Moudle<Intf>(self, self->ctx.kL0FullLoadAl0PingPongFlag, self->ctx.kL0FullLoadBl0PingPongFlag, true);
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
__aicore__ bool Iterate<Intf, ImplType>::UpdateItersByFmap(Intf *self, TempIters& tempIters, bool updateIterByFmapTag)
{
    if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::HW_MODE)) {
        tempIters.woAL1Iter = self->ctx.woAL1Iter;
        tempIters.hoAL1Iter = self->ctx.hoAL1Iter;
    } else {
        tempIters.mAL1Iter = self->ctx.mAL1Iter;
    }
    tempIters.batchIter = self->ctx.batchIter;
    self->ctx.kAL1Iter = self->ctx.kIter / self->ctx.multiKAL1 + 1;
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
__aicore__ bool Iterate<Intf, ImplType>::UpdateItersByWeight(Intf *self, TempIters& tempIters, bool updateIterByFmapTag)
{
    tempIters.nBL1Iter = self->ctx.nBL1Iter;
    self->ctx.kBL1Iter = self->ctx.kIter / self->ctx.multiKBL1 + 1;
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
__aicore__ void Iterate<Intf, ImplType>::ReduceKPreloadFmapIter(Intf *self, TempIters& tempIters,
                                                                bool updateIterByFmapTag)
{
    if (self->ctx.loadAL1Flag || !self->ctx.kAL1fullload) {
        if (self->ctx.kIter % self->ctx.multiKAL1 == 0 && self->ctx.kIter < self->ctx.lastLoopKAL1StartPos) {
            if (self->ctx.kIter != 0) {
                self->ctx.queueAL1.FreeTensor(self->ctx.al1);
            }
            if (!UpdateItersByFmap(self, tempIters, updateIterByFmapTag)) {
                LoadAL1BaseMoudle<Intf>(self, tempIters);
            }
        }
        if ((self->ctx.ddr2l0LoopK % self->ctx.multiKAL1 == 0 &&
            self->ctx.kIter == self->ctx.lastLoopKAL1StartPos) ||
            (self->ctx.ddr2l0LoopK % self->ctx.multiKAL1 != 0 &&
            self->ctx.kIter == self->ctx.lastLoopKAL1StartPosTail)) {
            self->ctx.queueAL1.FreeTensor(self->ctx.al1); // last iter of multiKAL1 only Free AL1 Tensor
        }
        if (self->ctx.kIter % self->ctx.multiKAL1 == 0) {
            self->ctx.al1 = self->ctx.queueAL1.template DeQue<typename Intf::FmapT>();
            self->ctx.loadAL0Flag = true;
        }
    }
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::ReduceKPreloadWeightIter(Intf *self, TempIters& tempIters,
                                                                  bool updateIterByFmapTag)
{
    if (self->ctx.loadBL1Flag || !self->ctx.kBL1fullload) {
        if (self->ctx.kIter % self->ctx.multiKBL1 == 0 && self->ctx.kIter < self->ctx.lastLoopKBL1StartPos) {
            if (self->ctx.kIter != 0) {
                self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
            }
            if (!UpdateItersByWeight(self, tempIters, updateIterByFmapTag)) {
                LoadBL1BaseMoudle<Intf>(self, tempIters);
            }
        }
        if ((self->ctx.ddr2l0LoopK % self->ctx.multiKBL1 == 0 &&
            self->ctx.kIter == self->ctx.lastLoopKBL1StartPos) ||
            (self->ctx.ddr2l0LoopK % self->ctx.multiKBL1 != 0 &&
            self->ctx.kIter == self->ctx.lastLoopKBL1StartPosTail)) {
            self->ctx.queueBL1.FreeTensor(self->ctx.bl1); // last iter of multiKBL1 only Free BL1 Tensor
        }
        if (self->ctx.kIter % self->ctx.multiKBL1 == 0) {
            self->ctx.bl1 = self->ctx.queueBL1.template DeQue<typename Intf::WeightT>();
            self->ctx.loadBL0Flag = true;
        }
    }
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::ReduceKPreload(Intf *self)
{
    // updateIterByFmapTag is true means fm update; false means weight update
    bool updateIterByFmapTag = self->ctx.convTiling->kAL1 > self->ctx.convTiling->kBL1;
    if (self->ctx.kAL1fullload && !self->ctx.kBL1fullload) {
        updateIterByFmapTag = false;
    } else if (!self->ctx.kAL1fullload && self->ctx.kBL1fullload) {
        updateIterByFmapTag = true;
    }
    if (self->ctx.loadAL1Flag || !self->ctx.kAL1fullload) {
        self->ctx.kAL1Iter = 0;
        LoadAL1BaseMoudle<Intf>(self);
        if (self->ctx.kAL1fullload) {
            self->ctx.al1 = self->ctx.queueAL1.template DeQue<typename Intf::FmapT>();
            self->ctx.loadAL0Flag = true;
        }
    }
    if (self->ctx.loadBL1Flag || !self->ctx.kBL1fullload) {
        self->ctx.kBL1Iter = 0;
        LoadBL1BaseMoudle<Intf>(self);
        if (self->ctx.kBL1fullload) {
            self->ctx.bl1 = self->ctx.queueBL1.template DeQue<typename Intf::WeightT>();
            self->ctx.loadBL0Flag = true;
        }
    }

    // state
    uint16_t al0PingPongFlag = 0;
    uint16_t bl0PingPongFlag = 0;
    bool isFirst = true;
    TempIters tempIters;
    while (self->ctx.kIter < self->ctx.ddr2l0LoopK) {
        ReduceKPreloadFmapIter(self, tempIters, updateIterByFmapTag);
        ReduceKPreloadWeightIter(self, tempIters, updateIterByFmapTag);

        LoadL0Moudle<Intf>(self, al0PingPongFlag, bl0PingPongFlag, isFirst);
        isFirst = false;
    }
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::ReduceKFmapPreload(Intf *self)
{
    // updateIterByFmapTag is true means fm update; false means weight update
    bool updateIterByFmapTag = self->ctx.convTiling->kAL1 > self->ctx.convTiling->kBL1;
    if (self->ctx.kAL1fullload && !self->ctx.kBL1fullload) {
        updateIterByFmapTag = false;
    } else if (!self->ctx.kAL1fullload && self->ctx.kBL1fullload) {
        updateIterByFmapTag = true;
    }
    if (self->ctx.loadAL1Flag || !self->ctx.kAL1fullload) {
        self->ctx.kAL1Iter = 0;
        LoadAL1BaseMoudle<Intf>(self);
        if (self->ctx.kAL1fullload) {
            self->ctx.al1 = self->ctx.queueAL1.template DeQue<typename Intf::FmapT>();
            self->ctx.loadAL0Flag = true;
        }
    }

    // state
    uint16_t al0PingPongFlag = 0;
    uint16_t bl0PingPongFlag = 0;
    bool isFirst = true;
    TempIters tempIters;
    while (self->ctx.kIter < self->ctx.ddr2l0LoopK) {
        ReduceKPreloadFmapIter(self, tempIters, updateIterByFmapTag);

        if (self->ctx.loadBL1Flag || self->ctx.kIter % self->ctx.multiKBL1 == 0) {
            LoadBL1Moudle<Intf>(self);
            self->ctx.loadBL0Flag = true;
        }

        LoadL0Moudle<Intf>(self, al0PingPongFlag, bl0PingPongFlag, isFirst);
        isFirst = false;
    }
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::ReduceK(Intf *self)
{
    // state
    uint16_t al0PingPongFlag = 0;
    uint16_t bl0PingPongFlag = 0;
    bool isFirst = true;

    while (self->ctx.kIter < self->ctx.ddr2l0LoopK) {
        if (self->ctx.loadAL1Flag || (!self->ctx.kAL1fullload && self->ctx.kIter % self->ctx.multiKAL1 == 0)) {
            if (self->ctx.kIter != 0) {
                self->ctx.queueAL1.FreeTensor(self->ctx.al1);
            }
            LoadAL1Moudle<Intf>(self);
            self->ctx.al1 = self->ctx.queueAL1.template DeQue<typename Intf::FmapT>();
            self->ctx.loadAL0Flag = true;
        }
        if constexpr (Intf::weightUbTrans) {
            if (self->ctx.loadBL1Flag || self->ctx.kIter % self->ctx.multiKBL1 == 0) {
                LoadBL1Moudle<Intf>(self);
                self->ctx.loadBL0Flag = true;
            }
        } else {
            if (self->ctx.loadBL1Flag || (!self->ctx.kBL1fullload && self->ctx.kIter % self->ctx.multiKBL1 == 0)) {
                if (self->ctx.kIter != 0) {
                    self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
                }
                LoadBL1Moudle<Intf>(self);
                self->ctx.bl1 = self->ctx.queueBL1.template DeQue<typename Intf::WeightT>();
                self->ctx.loadBL0Flag = true;
            }
        }

        LoadL0Moudle<Intf>(self, al0PingPongFlag, bl0PingPongFlag, isFirst);
        isFirst = false;
    }
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::IterateK(Intf *self)
{
    SetMNBeforeIterateK<Intf>(self);
    if constexpr (!Intf::isDeQuantFlag) {
        IterateBiasScale(self);
    }

    if constexpr (Intf::isInnerBatchFlag || Intf::isDeQuantFlag) {
        self->ctx.cl0 = self->ctx.queueCL0.template AllocTensor<typename Intf::L0cT>();
    } else {
        if ((self->ctx.convTiling->pBufferFlag & 0x04) >> 2) {
            self->ctx.cl0 =
                self->ctx.wholeCl0Tensor[(self->ctx.cl0PingPongFlag & 0x1) * L0C_HALF_SIZE / Intf::sizeOfL0c];
        } else {
            self->ctx.cl0 = self->ctx.wholeCl0Tensor;
        }
    }

    // reduceK priority: 1.KL0FullLoad 2.L1DB Preload 3. ordinary reduceK
    if constexpr (Intf::kl0FullLoadFlag) {
        ReduceOneK(self);
    } else if constexpr (Intf::kPreLoadFlag) {
        if constexpr (Intf::weightUbTrans) {
            ReduceKFmapPreload(self);
        } else {
            ReduceKPreload(self);
        }
    } else {
        ReduceK(self);
    }

    FreeL1Tensor<Intf>(self);

    if constexpr (Intf::isInnerBatchFlag || Intf::isDeQuantFlag) {
        self->ctx.queueCL0.EnQue(self->ctx.cl0);
        self->ctx.cl0 = self->ctx.queueCL0.template DeQue<typename Intf::L0cT>();
    } else {
        self->ctx.cl0PingPongFlag++;
    }

    self->ctx.kIter = 0;
}

template <class Intf, uint32_t ImplType>
__aicore__ void Iterate<Intf, ImplType>::IterateBiasScale(Intf *self)
{
    if (self->ctx.enableBias) {
        if (!self->ctx.convTiling->biasFullLoadFlag) {
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
    if (self->ctx.enableVectorQuant && !self->ctx.convTiling->fixpParamsFullLoadFlag) {
        if constexpr (Intf::groupType != static_cast<int8_t>(ConvGroupType::NORMAL_CONV)) {
            event_t eventId = static_cast<event_t>(self->ctx.pipe.FetchEventID(HardEvent::FIX_MTE2));
            SetFlag<HardEvent::FIX_MTE2>(eventId);
            WaitFlag<HardEvent::FIX_MTE2>(eventId);
        }
        self->ctx.scaleL1 = self->ctx.queueScaleL1.template AllocTensor<typename Intf::ScaleT>();
        if constexpr (Intf::isExtendConv2d) {
            if (self->ctx.convTiling->quantMode0 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                self->ctx.loadScaleL1Ins.LoadChannelWiseL1(self->ctx.scaleL1, self->ctx.scalegm);
            }
            if (self->ctx.convTiling->dualOutput && self->ctx.convTiling->quantMode1 == static_cast<uint8_t>(QuantModeType::VECTOR_QUANT)) {
                self->ctx.loadScaleL1Ins.LoadChannelWiseL1(self->ctx.scaleL1[self->ctx.scale1L1offset],
                    self->ctx.scale1gm);
            }
        } else {
            self->ctx.loadScaleL1Ins.LoadChannelWiseL1(self->ctx.scaleL1, self->ctx.scalegm);
        }
        self->ctx.queueScaleL1.EnQue(self->ctx.scaleL1);
        self->ctx.scaleL1 = self->ctx.queueScaleL1.template DeQue<typename Intf::ScaleT>();
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

    IterateK(self);

    return true;
}

};


#endif // CONV_ITERATE_IMPL_H