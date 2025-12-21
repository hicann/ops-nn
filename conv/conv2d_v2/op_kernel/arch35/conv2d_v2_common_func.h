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
 * \file conv2d_v2_common_func.h
 * \brief
 */

#ifndef CONV2D_V2_COMMON_FUNC_H
#define CONV2D_V2_COMMON_FUNC_H

#include "../../common/arch35/conv_common_func.h"
#include "../../common/arch35/conv_framework_util.h"
#include "../../common/arch35/conv_instr.h"
#include "conv2d_v2_c04_impl.h"
#include "conv2d_v2_config.h"
#include "conv2d_v2_dma_impl.h"
#include "conv2d_v2_util.h"
#include "conv2d_v2_weight_ub_trans_impl.h"
#include "conv2d_v2_instr.h"

namespace Conv2dFunc {
using namespace ConvFunc;
using namespace conv;

CONV_DECLARE_REG_IMPL(Init);
CONV_DECLARE_REG_IMPL(SetOrgFmapShape);
CONV_DECLARE_REG_IMPL(SetOrgWeightShape);
CONV_DECLARE_REG_IMPL(SetOrgOutputShape);
CONV_DECLARE_REG_IMPL(SetSingleFmapShape);
CONV_DECLARE_REG_IMPL(SetSingleOutputShape);
CONV_DECLARE_REG_IMPL(SetFmapStartPosition);
CONV_DECLARE_REG_IMPL(SetOptGroupParams);

using TypeFalse = struct {
    __uint128_t _[1024];
};

template <class Intf>
static __aicore__ inline void InitKDirectionValueC04(Intf *self)
{
    size_t multiKL1 = self->ctx.convTiling->kAL1 / self->ctx.convTiling->kL0;
    self->ctx.maxKAL1Iter = 0;
    self->ctx.maxKBL1Iter = 0;
    self->ctx.maxKL0Iter = multiKL1 - 1;
    self->ctx.ddr2l0LoopK = multiKL1;
    self->ctx.kAL1Tail = self->ctx.convTiling->kAL1;
    self->ctx.kBL1Tail = self->ctx.convTiling->kBL1;
    self->ctx.kL0Tail = self->ctx.convTiling->kL0;
    self->ctx.multiKAL1 = multiKL1;
    self->ctx.multiKBL1 = multiKL1;
    self->ctx.kAL1fullload = true;
    self->ctx.kBL1fullload = true;
}

template <class Intf>
static __aicore__ inline void InitKDirectionValue(Intf *self)
{
    // K direction variable calculate
    if constexpr (Intf::c04Flag) {
        InitKDirectionValueC04(self);
        return;
    }
    uint64_t totalKAlignK0 = AlignB(self->ctx.singleCoreCi, Intf::k0) * self->ctx.convTiling->kernelHxkernelW;
    self->ctx.maxKAL1Iter = CeilDiv(totalKAlignK0, self->ctx.convTiling->kAL1) - 1;
    self->ctx.maxKBL1Iter = CeilDiv(totalKAlignK0, self->ctx.convTiling->kBL1) - 1;
    self->ctx.ddr2l0LoopK = CeilDiv(totalKAlignK0, self->ctx.convTiling->kL0);
    self->ctx.maxKL0Iter = self->ctx.ddr2l0LoopK - 1;
    self->ctx.kAL1Tail = (self->ctx.singleCoreCi * self->ctx.convTiling->kernelHxkernelW) % self->ctx.convTiling->kAL1;
    self->ctx.kAL1Tail = self->ctx.kAL1Tail == 0 ? self->ctx.convTiling->kAL1 : self->ctx.kAL1Tail;
    self->ctx.kBL1Tail = (self->ctx.singleCoreCi * self->ctx.convTiling->kernelHxkernelW) % self->ctx.convTiling->kBL1;
    self->ctx.kBL1Tail = self->ctx.kBL1Tail == 0 ? self->ctx.convTiling->kBL1 : self->ctx.kBL1Tail;
    self->ctx.kL0Tail = totalKAlignK0 % self->ctx.convTiling->kL0;
    if constexpr (Intf::k0 != Intf::k0FmapTail) {
        self->ctx.kAL0Tail = AlignB(self->ctx.singleCoreCi, Intf::k0FmapTail) * self->ctx.convTiling->kernelHxkernelW %
            self->ctx.convTiling->kL0;
        self->ctx.kAL0Tail = self->ctx.kAL0Tail == 0 ? self->ctx.convTiling->kL0 : self->ctx.kAL0Tail;
    }
    self->ctx.kL0Tail = self->ctx.kL0Tail == 0 ? self->ctx.convTiling->kL0 : self->ctx.kL0Tail;
    self->ctx.multiKAL1 = CeilDiv(self->ctx.convTiling->kAL1, self->ctx.convTiling->kL0);
    self->ctx.multiKBL1 = CeilDiv(self->ctx.convTiling->kBL1, self->ctx.convTiling->kL0);

    if constexpr (Intf::kPreLoadFlag || Intf::WEIGHT_NZ_FLAG) {
        self->ctx.kBL1AlignK0Tail = totalKAlignK0 % self->ctx.convTiling->kBL1;
        self->ctx.kBL1AlignK0Tail =
            self->ctx.kBL1AlignK0Tail == 0 ? self->ctx.convTiling->kBL1 : self->ctx.kBL1AlignK0Tail;
    }
    if constexpr (Intf::kPreLoadFlag) {
        self->ctx.kAL1AlignK0Tail = totalKAlignK0 % self->ctx.convTiling->kAL1;
        self->ctx.kAL1AlignK0Tail =
            self->ctx.kAL1AlignK0Tail == 0 ? self->ctx.convTiling->kAL1 : self->ctx.kAL1AlignK0Tail;
        self->ctx.lastLoopKAL1StartPos = self->ctx.ddr2l0LoopK - self->ctx.multiKAL1;
        self->ctx.lastLoopKBL1StartPos = self->ctx.ddr2l0LoopK - self->ctx.multiKBL1;
        self->ctx.lastLoopKAL1StartPosTail =
            self->ctx.ddr2l0LoopK - CeilDiv(self->ctx.kAL1AlignK0Tail, self->ctx.convTiling->kL0);
        self->ctx.lastLoopKBL1StartPosTail =
            self->ctx.ddr2l0LoopK - CeilDiv(self->ctx.kBL1AlignK0Tail, self->ctx.convTiling->kL0);
    }
    if constexpr (Intf::isDmaFlag) {
        self->ctx.cinBL1 = self->ctx.convTiling->kBL1 / (self->ctx.convTiling->khL1 * self->ctx.convTiling->kwL1);
        self->ctx.cinBL1LoopTimes = CeilDiv(self->ctx.singleCoreCi, self->ctx.cinBL1);
        self->ctx.maxCinBL1Iter = self->ctx.cinBL1LoopTimes - 1;
    }
}

template <class Intf, uint32_t ImplType>
struct Init {
    static __aicore__ inline void call(Intf *self, const void *__restrict convTiling)
    {
        self->ctx.convTiling = (TConv2DTiling *)convTiling;
        self->ctx.singleCoreBatch = self->ctx.convTiling->singleCoreBatch;
        InitBatchDirectionValue<Intf>(self);
        if ASCEND_IS_AIC_CONV {
            InitBaseValue(self);

            if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
                InitValueMMode(self);
            } else {
                InitValueHWMode(self);
            }

            InitBuffer<Intf>(self);
            InitKDirectionValue<Intf>(self);
            InitCoDirectionValue<Intf>(self);
            InitHf32Mode<Intf>(self);
            InitSubApiParams<Intf>(self);

            if constexpr (Intf::groupOptFlag) {
                self->ctx.ciPerGroup = self->ctx.convTiling->orgCi / self->ctx.convTiling->groups;
                self->ctx.singleGroupOpt = self->ctx.convTiling->singleCoreGroupOpt;
            } else if constexpr (Intf::c04NDFlag) {
                C04InitIterValue<Intf>(self);
            } else if constexpr (Intf::weightUbTrans) {
                WeightUbTransCalcBL1LoadTimes<Intf>(self);
            } else if constexpr (Intf::isDmaFlag) {
                DmaCalcAL1LoadTimes<Intf>(self);
                DmaCubeInit(self);
            }
        }
        if ASCEND_IS_AIV_CONV {
            if constexpr (Intf::groupOptNDFlag) {
                OptGroupVecInit<Intf>(self);
            } else if constexpr (Intf::c04NDFlag) {
                C04VecInit<Intf>(self);
            } else if constexpr (Intf::weightUbTrans) {
                WeightUbTransVecInit<Intf>(self);
            } else if constexpr (Intf::isDmaFlag) {
                DmaVecInit<Intf>(self);
            }
        }
    }

    static __aicore__ inline void InitBaseValue(Intf *self)
    {
        self->ctx.orgCi = self->ctx.convTiling->orgCi;
        self->ctx.orgHi = self->ctx.convTiling->orgHi;
        self->ctx.orgWi = self->ctx.convTiling->orgWi;
        self->ctx.orgCo = self->ctx.convTiling->orgCo;
        self->ctx.orgHo = self->ctx.convTiling->orgHo;
        self->ctx.orgWo = self->ctx.convTiling->orgWo;
        self->ctx.kernelH = self->ctx.convTiling->kernelH;
        self->ctx.kernelW = self->ctx.convTiling->kernelW;
        self->ctx.dilatedKernelH = 1 + (self->ctx.kernelH - 1) * self->ctx.convTiling->dilationH;
        self->ctx.dilatedKernelW = 1 + (self->ctx.kernelW - 1) * self->ctx.convTiling->dilationW;
        self->ctx.singleCoreCi = self->ctx.convTiling->singleCoreCi; // Cin Size in Single Core
        uint64_t alignCinKhKw =
            AlignB(self->ctx.convTiling->singleCoreCi, Intf::k0) * self->ctx.convTiling->kernelHxkernelW;
        self->ctx.kAL1fullload = alignCinKhKw == self->ctx.convTiling->kAL1;
        self->ctx.kBL1fullload = alignCinKhKw == self->ctx.convTiling->kBL1;
        self->ctx.singleCoreCo = self->ctx.convTiling->singleCoreCo; // Cout Size in Single Core
        self->ctx.fmapOneBatchSize = self->ctx.convTiling->orgCi * self->ctx.convTiling->orgHi *
                                     self->ctx.convTiling->orgWi;
        self->ctx.outputOneBatchSize = self->ctx.convTiling->orgCo * self->ctx.convTiling->orgHo *
                                       self->ctx.convTiling->orgWo;
    }

    static __aicore__ inline void InitValueMMode(Intf *self)
    {
        self->ctx.mAL1 = self->ctx.convTiling->hoL1;
        self->ctx.mL0 = self->ctx.convTiling->hoL0;
        self->ctx.singleCoreM = self->ctx.convTiling->singleCoreHo;
        InitMDirectionValue<Intf>(self);
    }

    static __aicore__ inline void InitValueHWMode(Intf *self)
    {
        self->ctx.singleCoreHo = self->ctx.convTiling->singleCoreHo; // Ho Size in Single Core
        self->ctx.singleCoreWo = self->ctx.convTiling->singleCoreWo; // Wo Size in Single Core
        InitHoDirectionValue<Intf>(self);
        InitWoDirectionValue<Intf>(self);
    }
};

template <class Intf, uint32_t ImplType>
struct SetOrgFmapShape {
    static __aicore__ inline void call(Intf *self, uint64_t orgCi, uint64_t orgHi, uint64_t orgWi)
    {
        self->ctx.orgCi = orgCi;
        self->ctx.orgHi = orgHi;
        self->ctx.orgWi = orgWi;
    }
};

template <class Intf, uint32_t ImplType>
struct SetOrgWeightShape {
    static __aicore__ inline void call(Intf *self, uint64_t orgCo, uint64_t orgCi, uint64_t orgKh, uint64_t orgKw)
    {
        self->ctx.orgCo = orgCo;
        self->ctx.orgCi = orgCi;
        self->ctx.kernelH = orgKh;
        self->ctx.kernelW = orgKw;
    }
};

template <class Intf, uint32_t ImplType>
struct SetOrgOutputShape {
    static __aicore__ inline void call(Intf *self, uint64_t orgCo, uint64_t orgHo, uint64_t orgWo)
    {
        self->ctx.orgCo = orgCo;
        self->ctx.orgHo = orgHo;
        self->ctx.orgWo = orgWo;
    }
};

template <class Intf, uint32_t ImplType>
struct SetSingleFmapShape {
    static __aicore__ inline void call(Intf *self, uint64_t singleCi, uint64_t singleHi, uint64_t singleWi)
    {
        self->ctx.singleCoreCi = singleCi;
    }
};

template <class Intf, uint32_t ImplType>
struct SetSingleOutputShape {
    static __aicore__ inline void call(Intf *self, uint64_t singleCo, uint64_t singleHo, uint64_t singleWo,
        uint64_t singleCoreBatch)
    {
        if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::HW_MODE)) {
            self->ctx.singleCoreCo = singleCo;
            self->ctx.singleCoreHo = singleHo;
            self->ctx.singleCoreWo = singleWo;
            self->ctx.singleCoreBatch = singleCoreBatch;
            InitBatchDirectionValue<Intf>(self);
            if ASCEND_IS_AIC_CONV {
                InitCoDirectionValue<Intf>(self);
                InitHoDirectionValue<Intf>(self);
                InitWoDirectionValue<Intf>(self);
                if constexpr (Intf::groupOptFlag) {
                    OptGroupCalcBL1LoadTimes<Intf>(self);
                } else if constexpr (Intf::c04NDFlag) {
                    C04InitIterValue<Intf>(self);
                } else if constexpr (Intf::weightUbTrans) {
                    WeightUbTransCalcBL1LoadTimes<Intf>(self);
                } else if constexpr (Intf::isDmaFlag) {
                    DmaCalcAL1LoadTimes<Intf>(self);
                }
            }
            if ASCEND_IS_AIV_CONV {
                if constexpr (Intf::groupOptNDFlag) {
                    OptGroupInitNValue<Intf>(self);
                    OptGroupInitIterValue<Intf>(self);
                    if (self->ctx.vecId == 1) {
                        OptGroupCalcBL1LoadTimes<Intf>(self);
                    }
                } else if constexpr (Intf::c04NDFlag) {
                    C04InitNValue<Intf>(self);
                    C04InitIterValue<Intf>(self);
                } else if constexpr (Intf::weightUbTrans) {
                    WeightUbTransInitNValue<Intf>(self);
                    WeightUbTransInitIterValue<Intf>(self);
                    WeightUbTransCalcBL1LoadTimes<Intf>(self);
                } else if constexpr (Intf::isDmaFlag) {
                    DmaInitHWValue<Intf>(self);
                    DmaInitIterValue<Intf>(self);
                }
            }
        }
    }
 
    static __aicore__ inline void call(Intf *self, uint64_t singleCo, uint64_t singleM, uint64_t singleCoreBatch)
    {
        if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
            self->ctx.singleCoreCo = singleCo;
            self->ctx.singleCoreM = singleM;
            self->ctx.singleCoreBatch = singleCoreBatch;
            InitBatchDirectionValue<Intf>(self);
            if ASCEND_IS_AIC_CONV {
                InitCoDirectionValue<Intf>(self);
                InitMDirectionValue<Intf>(self);
                if constexpr (Intf::groupOptFlag) {
                    OptGroupCalcBL1LoadTimes<Intf>(self);
                } else if constexpr (Intf::c04NDFlag) {
                    C04InitIterValue<Intf>(self);
                } else if constexpr (Intf::weightUbTrans) {
                    WeightUbTransCalcBL1LoadTimes<Intf>(self);
                }
            }
            if ASCEND_IS_AIV_CONV {
                if constexpr (Intf::groupOptNDFlag) {
                    OptGroupInitNValue<Intf>(self);
                    OptGroupInitIterValue<Intf>(self);
                    if (self->ctx.vecId == 1) {
                        OptGroupCalcBL1LoadTimes<Intf>(self);
                    }
                } else if constexpr (Intf::c04NDFlag) {
                    C04InitNValue<Intf>(self);
                    C04InitIterValue<Intf>(self);
                } else if constexpr (Intf::weightUbTrans) {
                    WeightUbTransInitNValue<Intf>(self);
                    WeightUbTransInitIterValue<Intf>(self);
                    WeightUbTransCalcBL1LoadTimes<Intf>(self);
                }
            }
        }
    }
};

template <class Intf, uint32_t ImplType>
struct SetFmapStartPosition {
    static __aicore__ inline void call(Intf *self, int64_t hiStartPos, int64_t wiStartPos)
    {
        if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::HW_MODE)) {
            self->ctx.hiStartPos = hiStartPos;
            self->ctx.wiStartPos = wiStartPos;
        }
    }

    static __aicore__ inline void call(Intf *self, int64_t mStartPos)
    {
        if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
            if ASCEND_IS_AIC_CONV {
                self->ctx.mStartPos = mStartPos;
            }
        }
    }
};

template <class Intf, uint32_t ImplType>
struct SetOptGroupParams {
    static __aicore__ inline void call(Intf *self, uint64_t singleGroups, uint64_t singleGroupOpt)
    {
        self->ctx.singleGroups = singleGroups;
        self->ctx.singleGroupOpt = singleGroupOpt;
        self->ctx.singleCoreCi = singleGroups * self->ctx.ciPerGroup;

        if ASCEND_IS_AIC_CONV {
            InitKDirectionValue<Intf>(self);
            if constexpr (Intf::groupOptFlag) {
                OptGroupCalcBL1LoadTimes<Intf>(self);
            }
        } 
        if ASCEND_IS_AIV_CONV {
            if constexpr (Intf::groupOptNDFlag) {
                OptGroupInitKValue<Intf>(self);
            }
        }
    }
};

}  // namespace Conv2dFunc
#endif // CONV2D_V2_COMMON_FUNC_H