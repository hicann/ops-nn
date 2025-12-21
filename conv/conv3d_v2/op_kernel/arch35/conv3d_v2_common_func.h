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
 * \file conv3d_v2_common_func.h
 * \brief
 */

#ifndef CONV3D_V2_COMMON_FUNC_H
#define CONV3D_V2_COMMON_FUNC_H

#include "../../common/arch35/conv_common_func.h"
#include "../../common/arch35/conv_framework_util.h"
#include "../../common/arch35/conv_instr.h"
#include "conv3d_v2_config.h"
#include "conv3d_v2_util.h"
#include "conv3d_v2_instr.h"

namespace Conv3dFunc {
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

using TypeFalse = const struct {
    __uint128_t _[1024];
};

template <class Intf>
__aicore__ inline void InitKDirectionValue(Intf *self)
{
    // K方向变量计算
    uint64_t alignCinKhKwKd =
        AlignB(self->ctx.singleCoreCi, Intf::k0) * self->ctx.convTiling->kernelHxkernelWxkernelD;
    self->ctx.maxKAL1Iter = CeilDiv(alignCinKhKwKd, self->ctx.convTiling->kAL1) - 1;

    self->ctx.ddr2l0LoopK = CeilDiv(alignCinKhKwKd, self->ctx.convTiling->kL0);
    self->ctx.maxKL0Iter = self->ctx.ddr2l0LoopK - 1;

    self->ctx.maxKBL1Iter = CeilDiv(alignCinKhKwKd, self->ctx.convTiling->kBL1) - 1;
    self->ctx.multiKBL1 = self->ctx.convTiling->kBL1 / self->ctx.convTiling->kL0;

    self->ctx.kL0Tail = alignCinKhKwKd % self->ctx.convTiling->kL0;
    self->ctx.kL0Tail = self->ctx.kL0Tail == 0 ? self->ctx.convTiling->kL0 : self->ctx.kL0Tail;
    self->ctx.multiKAL1 = self->ctx.convTiling->kAL1 / self->ctx.convTiling->kL0;

    self->ctx.aL1CinTail = self->ctx.singleCoreCi % self->ctx.aL1Cin;
    self->ctx.aL1CinTail = self->ctx.aL1CinTail == 0 ? self->ctx.aL1Cin : self->ctx.aL1CinTail;
    self->ctx.aL1CinLoadNum = CeilDiv(self->ctx.singleCoreCi, self->ctx.aL1Cin);
}

template <class Intf>
__aicore__ inline void InitDoDirectionValue(Intf *self)
{
    // D方向变量计算
    self->ctx.ddr2l1LoopD = self->ctx.singleCoreDo;
}

template <class Intf, uint32_t ImplType>
struct Init {
    static __aicore__ inline void call(Intf *self, const void *__restrict tiling)
    {
        self->ctx.convTiling = (TConv3DTiling *)tiling;
        self->ctx.ddr2l1LoopBatch = self->ctx.convTiling->singleCoreBatch;
        if ASCEND_IS_AIC_CONV {
            InitTilingData(self, tiling);
            InitL1LoadParams(self);
            self->ctx.dilatedKernelH = 1 + (self->ctx.kernelH - 1) * self->ctx.convTiling->dilationH;
            self->ctx.dilatedKernelW = 1 + (self->ctx.kernelW - 1) * self->ctx.convTiling->dilationW;
            self->ctx.dilatedKernelD = 1 + (self->ctx.kernelD - 1) * self->ctx.convTiling->dilationD;

            InitBuffer<Intf>(self);
            InitKDirectionValue<Intf>(self);
            if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::HW_MODE)) {
                InitHoDirectionValue<Intf>(self);
                InitWoDirectionValue<Intf>(self);
            } else {
                InitMDirectionValue<Intf>(self);
            }

            InitCoDirectionValue<Intf>(self);
            InitDoDirectionValue<Intf>(self);
            InitHf32Mode<Intf>(self);
            InitSubApiParams<Intf>(self);

            if constexpr (Intf::groupOptFlag) {
                self->ctx.ciPerGroup = self->ctx.convTiling->orgCi / self->ctx.convTiling->groups;
                self->ctx.singleGroupOpt = self->ctx.convTiling->singleCoreGroupOpt;
                OptGroupCalcBL1LoadTimes<Intf>(self);
            }
        } 
        if ASCEND_IS_AIV_CONV {
            if constexpr (Intf::groupOptFlag) {
                OptGroupVecInit<Intf>(self);
            }
        }
    }

    static __aicore__ inline void InitTilingData(Intf *self, const void *__restrict tiling)
    {
        self->ctx.kernelH = self->ctx.convTiling->kernelH;
        self->ctx.kernelW = self->ctx.convTiling->kernelW;
        self->ctx.kernelD = self->ctx.convTiling->kernelD;
        self->ctx.orgHi = self->ctx.convTiling->orgHi;
        self->ctx.orgWi = self->ctx.convTiling->orgWi;
        self->ctx.orgDi = self->ctx.convTiling->orgDi;
        self->ctx.orgHo = self->ctx.convTiling->orgHo;
        self->ctx.orgWo = self->ctx.convTiling->orgWo;
        self->ctx.orgCo = self->ctx.convTiling->orgCo;
        self->ctx.orgCi = self->ctx.convTiling->orgCi;
        self->ctx.orgDo = self->ctx.convTiling->orgDo;
        self->ctx.singleCoreCo = self->ctx.convTiling->singleCoreCo;
        self->ctx.singleCoreDo = self->ctx.convTiling->singleCoreDo;
        if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
            self->ctx.singleCoreM = self->ctx.convTiling->singleCoreHo;
            self->ctx.mAL1 = self->ctx.convTiling->hoL1;
            self->ctx.mL0 = self->ctx.convTiling->hoL0;
        } else {
            self->ctx.singleCoreHo = self->ctx.convTiling->singleCoreHo;
            self->ctx.singleCoreWo = self->ctx.convTiling->singleCoreWo;
        }
        self->ctx.singleCoreCi = self->ctx.convTiling->singleCoreCi;
        uint64_t alignCinKhKwKd = 
            AlignB(self->ctx.convTiling->singleCoreCi, Intf::k0) * self->ctx.convTiling->kernelHxkernelWxkernelD;
        self->ctx.kBL1fullload = alignCinKhKwKd == self->ctx.convTiling->kBL1;
        self->ctx.kAL1fullload = alignCinKhKwKd == self->ctx.convTiling->kAL1;
        self->ctx.fmapOneBatchSize = self->ctx.orgCi * self->ctx.orgHi * self->ctx.orgWi * self->ctx.orgDi;
        self->ctx.outputOneBatchSize = self->ctx.orgCo * self->ctx.orgHo * self->ctx.orgWo * self->ctx.orgDo;
    }

    static __aicore__ inline void InitL1LoadParams(Intf *self)
    {
        self->ctx.cin1xcin0 = AlignB(self->ctx.singleCoreCi, Intf::k0);

        self->ctx.aL1Dk = self->ctx.convTiling->cinAInCore <= self->ctx.cin1xcin0 ?
                            1 : self->ctx.convTiling->cinAInCore / self->ctx.cin1xcin0;
        self->ctx.aL1DkTail = self->ctx.kernelD % self->ctx.aL1Dk;
        self->ctx.aL1DkTail = self->ctx.aL1DkTail == 0 ? self->ctx.aL1Dk : self->ctx.aL1DkTail;
        self->ctx.aL1Cin = self->ctx.convTiling->cinAInCore / self->ctx.aL1Dk;

        self->ctx.bL1Dk = self->ctx.convTiling->cinBInCore <= self->ctx.cin1xcin0 ?
                            1 : self->ctx.convTiling->cinBInCore / self->ctx.cin1xcin0;
        self->ctx.bL1DkTail = self->ctx.kernelD % self->ctx.bL1Dk;
        self->ctx.bL1DkTail = self->ctx.bL1DkTail == 0 ? self->ctx.bL1Dk : self->ctx.bL1DkTail;
        self->ctx.bL1Cin = self->ctx.convTiling->cinBInCore / self->ctx.bL1Dk;
        self->ctx.bL1CinTail = self->ctx.bL1Dk > 1 ? self->ctx.bL1Cin : self->ctx.singleCoreCi % self->ctx.bL1Cin;
        self->ctx.bL1CinTail = self->ctx.bL1CinTail == 0 ? self->ctx.bL1Cin : self->ctx.bL1CinTail;
        self->ctx.bL1CinLoadNum = CeilDiv(self->ctx.singleCoreCi, self->ctx.bL1Cin);
    }
};

template <class Intf, uint32_t ImplType>
struct SetOrgFmapShape {
    static __aicore__ inline void call(Intf *self, uint64_t orgCi, uint64_t orgDi, uint64_t orgHi, uint64_t orgWi)
    {
        self->ctx.orgCi = orgCi;
        self->ctx.orgDi = orgDi;
        self->ctx.orgHi = orgHi;
        self->ctx.orgWi = orgWi;
    }
};

template <class Intf, uint32_t ImplType>
struct SetOrgWeightShape {
    static __aicore__ inline void call(
        Intf *self, uint64_t orgCo, uint64_t orgCi, uint64_t orgKd, uint64_t orgKh, uint64_t orgKw)
    {
        self->ctx.orgCo = orgCo;
        self->ctx.orgCi = orgCi;
        self->ctx.kernelD = orgKd;
        self->ctx.kernelH = orgKh;
        self->ctx.kernelW = orgKw;
    }
};

template <class Intf, uint32_t ImplType>
struct SetOrgOutputShape {
    static __aicore__ inline void call(Intf *self, uint64_t orgCo, uint64_t orgDo, uint64_t orgHo, uint64_t orgWo)
    {
        self->ctx.orgCo = orgCo;
        self->ctx.orgDo = orgDo;
        self->ctx.orgHo = orgHo;
        self->ctx.orgWo = orgWo;
    }
};

template <class Intf, uint32_t ImplType>
struct SetSingleFmapShape {
    static __aicore__ inline void call(
        Intf *self, uint64_t singleCi, uint64_t singleDi, uint64_t singleHi, uint64_t singleWi)
    {
        self->ctx.singleCoreCi = singleCi;
    }
};

template <class Intf, uint32_t ImplType>
struct SetSingleOutputShape {
    static __aicore__ inline void call(
        Intf *self, uint64_t singleCo, uint64_t singleDo, uint64_t singleHo, uint64_t singleWo, uint64_t singleCoreBatch)
    {
        if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::HW_MODE)) {
            self->ctx.singleCoreCo = singleCo;
            self->ctx.singleCoreDo = singleDo;
            self->ctx.singleCoreHo = singleHo;
            self->ctx.ddr2l1LoopBatch = singleCoreBatch;
            if ASCEND_IS_AIC_CONV {
                InitCoDirectionValue<Intf>(self);
                InitHoDirectionValue<Intf>(self);
                InitDoDirectionValue<Intf>(self);
                if constexpr (Intf::groupOptFlag) {
                    OptGroupCalcBL1LoadTimes<Intf>(self);
                }
            }
            if ASCEND_IS_AIV_CONV {
                if constexpr (Intf::groupOptFlag) {
                    OptGroupInitNValue<Intf>(self);
                    OptGroupInitIterValue<Intf>(self);
                }
            }
        }
    }

    static __aicore__ inline void call(Intf *self, uint64_t singleCo, uint64_t singleDo, uint64_t singleM, uint64_t singleCoreBatch)
    {
        if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
            self->ctx.singleCoreCo = singleCo;
            self->ctx.singleCoreDo = singleDo;
            self->ctx.singleCoreM = singleM;
            self->ctx.ddr2l1LoopBatch = singleCoreBatch;
            if ASCEND_IS_AIC_CONV {
                InitCoDirectionValue<Intf>(self);
                InitDoDirectionValue<Intf>(self);
                InitMDirectionValue<Intf>(self);
                if constexpr (Intf::groupOptFlag) {
                    OptGroupCalcBL1LoadTimes<Intf>(self);
                }
            } 
            if ASCEND_IS_AIV_CONV {
                if constexpr (Intf::groupOptFlag) {
                    OptGroupInitNValue<Intf>(self);
                    OptGroupInitIterValue<Intf>(self);
                }
            }
        }
    }
};

template <class Intf, uint32_t ImplType>
struct SetFmapStartPosition {
    static __aicore__ inline void call(
        Intf *self, int64_t diStartPos, int64_t hiStartPos, int64_t wiStartPos, int64_t ciStartPos)
    {
        if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::HW_MODE)) {
            if ASCEND_IS_AIC_CONV {
                self->ctx.diStartPos = diStartPos;
                self->ctx.wiStartPos = wiStartPos;
                self->ctx.hiStartPos = hiStartPos;
            }
        }
    }

    static __aicore__ inline void call(Intf *self, int64_t diStartPos, int64_t mStartPos, int64_t ciStartPos)
    {
        if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
            if ASCEND_IS_AIC_CONV {
                self->ctx.diStartPos = diStartPos;
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
            if constexpr (Intf::groupOptFlag) {
                OptGroupInitKValue<Intf>(self);
            }
        }
    }
};

}  // namespace Conv3dFunc

#endif // CONV3D_V2_COMMON_FUNC_H