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
 * \file conv_opt_group_init_impl.h
 * \brief
 */

#ifndef CONV_OPT_GROUP_INIT_IMPL_H
#define CONV_OPT_GROUP_INIT_IMPL_H

#include "conv_config.h"
#include "conv_framework_util.h"
#include "conv_util.h"

namespace ConvFunc {
using namespace AscendC;
using namespace conv;

template <class Intf>
__aicore__ inline void OptGroupCalcBL1LoadTimesHWMode(Intf *self)
{
    if (!self->ctx.kBL1fullload) {
        self->ctx.ddr2l1LoopKB = self->ctx.maxKBL1Iter + 1;
        uint64_t ddr2l0LoopH = CeilDiv(self->ctx.singleCoreHo, self->ctx.convTiling->hoL0);
        uint64_t ddr2l0LoopW = 0;
        if constexpr (Intf::hasWL0IterFlag) {
            if (self->ctx.woL1SmallTail > 0) {
                ddr2l0LoopW = (self->ctx.ddr2l1LoopW - W_TAIL_NUM) *
                    CeilDiv(self->ctx.convTiling->woL1, self->ctx.convTiling->woL0) +
                    CeilDiv(self->ctx.woAL1Tail, self->ctx.convTiling->woL0) +
                    CeilDiv(self->ctx.woL1SmallTail, self->ctx.convTiling->woL0);
            } else {
                ddr2l0LoopW = CeilDiv(self->ctx.singleCoreWo, self->ctx.convTiling->woL0);
            }
        } else {
            ddr2l0LoopW = CeilDiv(self->ctx.singleCoreWo, self->ctx.convTiling->woL0);
        }
        self->ctx.bL1LoadTimes = ddr2l0LoopH * ddr2l0LoopW * self->ctx.ddr2l1LoopN * self->ctx.l12l0LoopN *
                                 self->ctx.ddr2l1LoopBatch * self->ctx.ddr2l1LoopKB;
        if constexpr (Intf::isConv3D) {
            self->ctx.bL1LoadTimes *= self->ctx.singleCoreDo;
        }
    } else {
        if constexpr (Intf::iterateMFirstFlag) {
            self->ctx.bL1LoadTimes = self->ctx.ddr2l1LoopN;
        } else {
            self->ctx.bL1LoadTimes = self->ctx.ddr2l1LoopH * self->ctx.ddr2l1LoopW * self->ctx.ddr2l1LoopN *
                                     self->ctx.ddr2l1LoopBatch;
            if constexpr (Intf::isConv3D) {
                self->ctx.bL1LoadTimes *= self->ctx.singleCoreDo;
            }
        }
    }
}

template <class Intf>
__aicore__ inline void OptGroupCalcBL1LoadTimesMMode(Intf *self)
{
    if (!self->ctx.kBL1fullload) {
        uint64_t ddr2l0LoopM = CeilDiv(self->ctx.singleCoreM, self->ctx.mL0);
        self->ctx.ddr2l1LoopKB = self->ctx.maxKBL1Iter + 1;
        self->ctx.bL1LoadTimes = ddr2l0LoopM * self->ctx.ddr2l1LoopN * self->ctx.l12l0LoopN *
                                 self->ctx.ddr2l1LoopBatch * self->ctx.ddr2l1LoopKB;
        if constexpr (Intf::isConv3D) {
            self->ctx.bL1LoadTimes *= self->ctx.singleCoreDo;
        }
    } else {
        if constexpr (Intf::iterateMFirstFlag) {
            self->ctx.bL1LoadTimes = self->ctx.ddr2l1LoopN;
        } else {
            self->ctx.bL1LoadTimes = self->ctx.ddr2l1LoopN * self->ctx.ddr2l1LoopM * self->ctx.ddr2l1LoopBatch;
            if constexpr (Intf::isConv3D) {
                self->ctx.bL1LoadTimes *= self->ctx.singleCoreDo;
            }
        }
    }
}

template <class Intf>
__aicore__ inline void OptGroupCalcBL1LoadTimes(Intf *self)
{
    if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::HW_MODE)) {
        OptGroupCalcBL1LoadTimesHWMode<Intf>(self);
    } else if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
        OptGroupCalcBL1LoadTimesMMode<Intf>(self);
    }
}

template <class Intf>
__aicore__ inline void OptGroupInitIterValueMfirstHWMode(Intf *self)
{
    if (!self->ctx.kBL1fullload) {
        uint64_t ddr2l0LoopH = CeilDiv(self->ctx.singleCoreHo, self->ctx.convTiling->hoL0);
        uint64_t ddr2l0LoopW = 0;
        if constexpr (Intf::hasWL0IterFlag) {
            self->ctx.woAL1Tail = self->ctx.singleCoreWo % self->ctx.convTiling->woL1;
            self->ctx.woAL1Tail = self->ctx.woAL1Tail == 0 ?  self->ctx.convTiling->woL1 : self->ctx.woAL1Tail;
            if (self->ctx.convTiling->hoL0 > 1 && self->ctx.woAL1Tail % BLOCK_L0_N > 0 &&
                CeilDiv(self->ctx.woAL1Tail, self->ctx.convTiling->woL0) > 1) {
                self->ctx.woL1SmallTail = self->ctx.woAL1Tail % self->ctx.convTiling->woL0;
                self->ctx.woAL1Tail = (self->ctx.woAL1Tail / self->ctx.convTiling->woL0) * self->ctx.convTiling->woL0;
            }

            self->ctx.ddr2l1LoopW = CeilDiv(self->ctx.singleCoreWo, self->ctx.convTiling->woL1);
            ddr2l0LoopW = 
                (self->ctx.ddr2l1LoopW - 1) * CeilDiv(self->ctx.convTiling->woL1, self->ctx.convTiling->woL0) +
                CeilDiv(self->ctx.woAL1Tail, self->ctx.convTiling->woL0);

            if (self->ctx.woL1SmallTail > 0) {
                ddr2l0LoopW += CeilDiv(self->ctx.woL1SmallTail, self->ctx.convTiling->woL0);
            }
        } else {
            ddr2l0LoopW = CeilDiv(self->ctx.singleCoreWo, self->ctx.convTiling->woL0);
        }

        self->ctx.ddr2l1LoopTmp = ddr2l0LoopH * ddr2l0LoopW * self->ctx.ddr2l1LoopBatch;
        if constexpr (Intf::isConv3D) {
            self->ctx.ddr2l1LoopTmp *= self->ctx.singleCoreDo;
        }
    }
}

template <class Intf>
__aicore__ inline void OptGroupInitIterValueNfirstHWMode(Intf *self)
{
    if constexpr (!Intf::hasHL1IterFlag) {
        self->ctx.ddr2l1LoopH = 1;
        self->ctx.maxHoL1Iter = 0;
    } else {
        self->ctx.ddr2l1LoopH = CeilDiv(self->ctx.singleCoreHo, self->ctx.convTiling->hoL1);
        self->ctx.maxHoL1Iter = self->ctx.ddr2l1LoopH - 1;
    }

    if constexpr (Intf::hasWL1IterFlag) {
        self->ctx.ddr2l1LoopW = CeilDiv(self->ctx.singleCoreWo, self->ctx.convTiling->woL1);
    } else {
        self->ctx.ddr2l1LoopW = 1;
    }

    self->ctx.woAL1Tail = self->ctx.singleCoreWo % self->ctx.convTiling->woL1;
    self->ctx.woAL1Tail = self->ctx.woAL1Tail == 0 ?  self->ctx.convTiling->woL1 : self->ctx.woAL1Tail;
    if constexpr (Intf::hasWL0IterFlag) {
        if (self->ctx.convTiling->hoL0 > 1 && self->ctx.woAL1Tail % BLOCK_L0_N > 0 &&
            CeilDiv(self->ctx.woAL1Tail, self->ctx.convTiling->woL0) > 1) {
            self->ctx.woL1SmallTail = self->ctx.woAL1Tail % self->ctx.convTiling->woL0;
            self->ctx.woAL1Tail = (self->ctx.woAL1Tail / self->ctx.convTiling->woL0) * self->ctx.convTiling->woL0;
        }

        if (self->ctx.woL1SmallTail > 0) {
            self->ctx.ddr2l1LoopW += 1;
        }
    }

    if (!self->ctx.kBL1fullload) {
        self->ctx.hoAL1Tail = self->ctx.singleCoreHo % self->ctx.convTiling->hoL1;
        self->ctx.hoAL1Tail = self->ctx.hoAL1Tail == 0 ? self->ctx.convTiling->hoL1 : self->ctx.hoAL1Tail;
        self->ctx.currentHoL1 = self->ctx.hoAL1Tail;
        self->ctx.currentWoL1 = self->ctx.woAL1Tail;
        self->ctx.maxWoL1Iter = self->ctx.ddr2l1LoopW - 1;
    }

    self->ctx.ddr2l1LoopOuter = self->ctx.ddr2l1LoopH * self->ctx.ddr2l1LoopW * self->ctx.ddr2l1LoopBatch;
    if constexpr (Intf::isConv3D) {
        self->ctx.ddr2l1LoopOuter *= self->ctx.singleCoreDo;
    }
}

template <class Intf>
__aicore__ inline void OptGroupInitIterValueMfirstMMode(Intf *self)
{
    if (!self->ctx.kBL1fullload) {
        uint64_t ddr2l0LoopM = CeilDiv(self->ctx.singleCoreM, self->ctx.mL0);
        self->ctx.ddr2l1LoopTmp = ddr2l0LoopM * self->ctx.ddr2l1LoopBatch;
        if constexpr (Intf::isConv3D) {
            self->ctx.ddr2l1LoopTmp *= self->ctx.singleCoreDo;
        }
    }
}

template <class Intf>
__aicore__ inline void OptGroupInitIterValueNfirstMMode(Intf *self)
{
    self->ctx.ddr2l1LoopM = CeilDiv(self->ctx.singleCoreM, self->ctx.mAL1);

    if (!self->ctx.kBL1fullload) {
        self->ctx.mAL1Tail = self->ctx.singleCoreM % self->ctx.mAL1;
        self->ctx.mAL1Tail = self->ctx.mAL1Tail == 0 ? self->ctx.mAL1 : self->ctx.mAL1Tail;
        self->ctx.l12l0LoopM = CeilDiv(self->ctx.mAL1, self->ctx.mL0);
        self->ctx.maxMAL1Iter = self->ctx.ddr2l1LoopM - 1;
    }

    self->ctx.ddr2l1LoopOuter = self->ctx.ddr2l1LoopM * self->ctx.ddr2l1LoopBatch;
    if constexpr (Intf::isConv3D) {
        self->ctx.ddr2l1LoopOuter *= self->ctx.singleCoreDo;
    }
}

template <class Intf>
__aicore__ inline void OptGroupInitIterValue(Intf *self)
{
    if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::HW_MODE)) {
        if constexpr (Intf::iterateMFirstFlag) {
            OptGroupInitIterValueMfirstHWMode<Intf>(self);
        } else {
            OptGroupInitIterValueNfirstHWMode<Intf>(self);
        }
    } else if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
        if constexpr (Intf::iterateMFirstFlag) {
            OptGroupInitIterValueMfirstMMode<Intf>(self);
        } else {
            OptGroupInitIterValueNfirstMMode<Intf>(self);
        }
    }
}

template <class Intf>
__aicore__ inline void OptGroupInitKValue(Intf *self)
{
    self->ctx.ddr2l1LoopKB =
        CeilDiv(AlignB(self->ctx.singleCoreCi, Intf::k0) * self->ctx.convTiling->kernelHxkernelWxkernelD,
                self->ctx.convTiling->kBL1);
    self->ctx.maxKBL1Iter = self->ctx.ddr2l1LoopKB - 1;
    self->ctx.kBL1fullload = self->ctx.ddr2l1LoopKB == 1;

    self->ctx.ci1Opt = CeilDiv(self->ctx.singleCoreCi, Intf::k0);

    if constexpr (Intf::isConv3D) {
        self->ctx.bL1Cin = self->ctx.convTiling->cinBInCore / self->ctx.bL1Dk;
        self->ctx.bL1CinTail = self->ctx.bL1Dk > 1 ? self->ctx.bL1Cin : self->ctx.singleCoreCi % self->ctx.bL1Cin;
        self->ctx.bL1CinTail = self->ctx.bL1CinTail == 0 ? self->ctx.bL1Cin : self->ctx.bL1CinTail;
        self->ctx.bL1CinLoadNum = CeilDiv(self->ctx.singleCoreCi, self->ctx.bL1Cin);
    }
}

template <class Intf>
__aicore__ inline void OptGroupInitNValue(Intf *self)
{
    self->ctx.nBL1Tail = self->ctx.singleCoreCo % self->ctx.convTiling->nBL1;
    self->ctx.nBL1Tail = self->ctx.nBL1Tail == 0 ? self->ctx.convTiling->nBL1 : self->ctx.nBL1Tail;

    if constexpr (!Intf::hasNL1IterFlag) {
        self->ctx.ddr2l1LoopN = 1;
        self->ctx.maxNBL1Iter = 0;
    } else {
        self->ctx.ddr2l1LoopN = CeilDiv(self->ctx.singleCoreCo, self->ctx.convTiling->nBL1);
        self->ctx.maxNBL1Iter = self->ctx.ddr2l1LoopN  - 1;
    }

    if constexpr (!Intf::hasNL0IterFlag) {
        self->ctx.l12l0LoopN = 1;
    } else {
        self->ctx.l12l0LoopN = self->ctx.convTiling->multiNBL1;
    }
}

template <class Intf>
__aicore__ inline void OptGroupInitBuf(Intf *self)
{
    self->ctx.ubBufSize = self->ctx.ci1Opt * self->ctx.convTiling->kernelHxkernelWxkernelD * self->ctx.co1Opt *
                          Intf::k0 * BLOCK_L0_N;
    self->ctx.pipe.InitBuffer(self->ctx.ndUbBuf, self->ctx.ubBufSize * Intf::sizeOfWeight);
    self->ctx.pipe.InitBuffer(self->ctx.nzUbBuf, self->ctx.ubBufSize * Intf::sizeOfWeight);
    self->ctx.pipe.InitBuffer(self->ctx.indexUbBuf, REG_SIZE);

    self->ctx.nzTensor = self->ctx.nzUbBuf.template Get<typename Intf::WeightT>();

    int8_t al1db = (self->ctx.convTiling->pBufferFlag & AL1_DB_IDX) >> AL1_DB_OFFSET;
    uint32_t aL1SpaceSize = self->ctx.convTiling->aL1SpaceSize;
    if (al1db) {
        aL1SpaceSize *= DOUBLE_BUF;
    }
    self->ctx.bL1SpaceSize = self->ctx.convTiling->nBL1 * self->ctx.convTiling->kBL1;

    if constexpr (Intf::bL1DBFlag) {
        self->ctx.pipe.InitBuffer(self->ctx.bL1TBuf,
            aL1SpaceSize + self->ctx.bL1SpaceSize * DOUBLE_BUF * Intf::sizeOfWeight);
    } else {
        self->ctx.pipe.InitBuffer(self->ctx.bL1TBuf,
            aL1SpaceSize + self->ctx.bL1SpaceSize * Intf::sizeOfWeight);
    }
    self->ctx.bl1 = self->ctx.bL1TBuf.template Get<typename Intf::WeightT>()[
        aL1SpaceSize / Intf::sizeOfFmap];
}

template <class Intf>
__aicore__ inline void OptGroupVecInit(Intf *self)
{
    self->ctx.vecId = GetBlockIdx();

    self->ctx.singleCoreCi = self->ctx.convTiling->singleCoreCi;
    self->ctx.singleCoreCo = self->ctx.convTiling->singleCoreCo;
    self->ctx.singleGroups = self->ctx.convTiling->singleCoreGroups;
    self->ctx.singleGroupOpt = self->ctx.convTiling->singleCoreGroupOpt;

    self->ctx.ciPerGroup = self->ctx.convTiling->orgCi / self->ctx.convTiling->groups;
    self->ctx.coPerGroup = self->ctx.convTiling->orgCo / self->ctx.convTiling->groups;
    self->ctx.ciOpt = self->ctx.ciPerGroup * self->ctx.convTiling->enlarge;
    self->ctx.ci1Opt = CeilDiv(self->ctx.ciOpt, Intf::k0);
    self->ctx.ciOptAlign = self->ctx.ci1Opt * Intf::k0;
    self->ctx.kUbSize = self->ctx.ciOptAlign * self->ctx.convTiling->kernelHxkernelWxkernelD;
    self->ctx.coOpt = self->ctx.coPerGroup * self->ctx.convTiling->enlarge;
    self->ctx.co1Opt = CeilDiv(self->ctx.coOpt, BLOCK_L0_N);
    self->ctx.coOptAlign = self->ctx.co1Opt * BLOCK_L0_N;

    if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
        self->ctx.singleCoreM = self->ctx.convTiling->singleCoreHo;
        self->ctx.mAL1 = self->ctx.convTiling->hoL1;
        self->ctx.mL0 = self->ctx.convTiling->hoL0;
    } else {
        self->ctx.singleCoreHo = self->ctx.convTiling->singleCoreHo;
        self->ctx.singleCoreWo = self->ctx.convTiling->singleCoreWo;
    }

    if constexpr (Intf::isConv3D) {
        self->ctx.singleCoreDo = self->ctx.convTiling->singleCoreDo;
        self->ctx.cin1xcin0 = AlignB(self->ctx.singleCoreCi, Intf::k0);
        self->ctx.bL1Dk = self->ctx.convTiling->cinBInCore <= self->ctx.cin1xcin0 ?
                            1 : self->ctx.convTiling->cinBInCore / self->ctx.cin1xcin0;
        self->ctx.bL1DkTail = self->ctx.kernelD % self->ctx.bL1Dk;
        self->ctx.bL1DkTail = self->ctx.bL1DkTail == 0 ? self->ctx.bL1Dk : self->ctx.bL1DkTail;
    }

    OptGroupInitBuf<Intf>(self);
    OptGroupInitKValue<Intf>(self);
    OptGroupInitNValue<Intf>(self);
    OptGroupInitIterValue<Intf>(self);
    if (self->ctx.vecId == 1) {
        OptGroupCalcBL1LoadTimes<Intf>(self);
    }

    self->ctx.optGroupLoadGm2UBTools.SetParams(self);
    self->ctx.optGroupTransND2NZTools.SetParams(self);
    self->ctx.optGroupLoadUB2L1Tools.SetParams(self);
}

};

#endif // CONV_OPT_GROUP_INIT_IMPL_H