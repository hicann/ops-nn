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
 * \file conv2d_v2_weight_ub_trans_impl.h
 * \brief
 */

#ifndef CONV2D_V2_WEIGHT_UB_TRANS_IMPL_H
#define CONV2D_V2_WEIGHT_UB_TRANS_IMPL_H

#include "conv2d_v2_config.h"
#include "conv2d_v2_util.h"
#include "conv2d_v2_instr.h"

namespace Conv2dFunc {
using namespace ConvFunc;
using namespace conv;

template <class Intf>
__aicore__ inline void WeightUbTransSyncSet(Intf *self)
{
    if (!((self->ctx.kIter == self->ctx.ddr2l0LoopK - 1) ||
          ((self->ctx.kIter + 1) % self->ctx.multiKBL1 == 0))) {
        return;
    }

    if (self->ctx.loadUB2L1Iter < self->ctx.bL1LoadTimes) {
        CrossCoreSetFlag<CV_ENHANCE_MODE, PIPE_MTE1>(self->ctx.pingPongFlag * VEC_ID_MAX + CV_SYNC_ID_MTE1_MTE3);
    }
    self->ctx.pingPongFlag ^= 1;
}

template <class Intf>
__aicore__ inline void WeightUbTransCalcBL1LoadTimes(Intf *self)
{
    if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
        uint64_t ddr2l0LoopM = CeilDiv(self->ctx.singleCoreM, self->ctx.mL0);
        self->ctx.ddr2l1LoopKB = self->ctx.maxKBL1Iter + 1;
        self->ctx.bL1LoadTimes = ddr2l0LoopM * self->ctx.ddr2l1LoopN * self->ctx.l12l0LoopN *
                                self->ctx.ddr2l1LoopBatch * self->ctx.ddr2l1LoopKB;
    } else {
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
    }
    self->ctx.bL1LoadTimes -= 1;
}

template <class Intf>
__aicore__ inline void WeightUbTransInitIterValueMfirstHWMode(Intf *self)
{
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
}

template <class Intf>
__aicore__ inline void WeightUbTransInitIterValueNfirstHWMode(Intf *self)
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
    self->ctx.woAL1Tail = self->ctx.woAL1Tail == 0 ? self->ctx.convTiling->woL1 : self->ctx.woAL1Tail;
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

    self->ctx.hoAL1Tail = self->ctx.singleCoreHo % self->ctx.convTiling->hoL1;
    self->ctx.hoAL1Tail = self->ctx.hoAL1Tail == 0 ? self->ctx.convTiling->hoL1 : self->ctx.hoAL1Tail;
    self->ctx.currentHoL1 = self->ctx.hoAL1Tail;
    self->ctx.currentWoL1 = self->ctx.woAL1Tail;
    self->ctx.maxWoL1Iter = self->ctx.ddr2l1LoopW - 1;

    self->ctx.ddr2l1LoopOuter = self->ctx.ddr2l1LoopH * self->ctx.ddr2l1LoopW * self->ctx.ddr2l1LoopBatch;
}

template <class Intf>
__aicore__ inline void WeightUbTransInitIterValue(Intf *self)
{
    if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
        if constexpr (Intf::iterateMFirstFlag) {
            uint64_t ddr2l0LoopM = CeilDiv(self->ctx.singleCoreM, self->ctx.mL0);
            self->ctx.ddr2l1LoopTmp = ddr2l0LoopM * self->ctx.ddr2l1LoopBatch;
        } else {
            self->ctx.ddr2l1LoopM = CeilDiv(self->ctx.singleCoreM, self->ctx.mAL1);

            self->ctx.mAL1Tail = self->ctx.singleCoreM % self->ctx.mAL1;
            self->ctx.mAL1Tail = self->ctx.mAL1Tail == 0 ? self->ctx.mAL1 : self->ctx.mAL1Tail;
            self->ctx.l12l0LoopM = CeilDiv(self->ctx.mAL1, self->ctx.mL0);
            self->ctx.maxMAL1Iter = self->ctx.ddr2l1LoopM - 1;

            self->ctx.ddr2l1LoopOuter = self->ctx.ddr2l1LoopM * self->ctx.ddr2l1LoopBatch;
        }
    } else {
        if constexpr (Intf::iterateMFirstFlag) {
            WeightUbTransInitIterValueMfirstHWMode<Intf>(self);
        } else {
            WeightUbTransInitIterValueNfirstHWMode<Intf>(self);
        }
    }
}

template <class Intf>
__aicore__ inline void WeightUbTransInitKValue(Intf *self)
{
    self->ctx.kBL1Tail = (self->ctx.singleCoreCi * self->ctx.convTiling->kernelHxkernelW) % self->ctx.convTiling->kBL1;
    self->ctx.kBL1Tail = self->ctx.kBL1Tail == 0 ? self->ctx.convTiling->kBL1 : self->ctx.kBL1Tail;
    self->ctx.kUbTail = self->ctx.kBL1Tail % self->ctx.convTiling->bUbKStep;
    self->ctx.kUbTail = self->ctx.kUbTail == 0 ? self->ctx.convTiling->bUbKStep : self->ctx.kUbTail;

    self->ctx.ddr2l1LoopKB = CeilDiv(AlignB(self->ctx.singleCoreCi, Intf::k0) * self->ctx.convTiling->kernelHxkernelW,
        self->ctx.convTiling->kBL1);
    self->ctx.maxKBL1Iter = self->ctx.ddr2l1LoopKB - 1;

    self->ctx.vecKLoopTimes = CeilDiv(self->ctx.convTiling->kBL1, self->ctx.convTiling->bUbKStep);
    self->ctx.maxVecKIter = self->ctx.vecKLoopTimes - 1;
}

template <class Intf>
__aicore__ inline void WeightUbTransInitNValue(Intf *self)
{
    self->ctx.nBL1Tail = self->ctx.singleCoreCo % self->ctx.convTiling->nBL1;
    self->ctx.nBL1Tail = self->ctx.nBL1Tail == 0 ? self->ctx.convTiling->nBL1 : self->ctx.nBL1Tail;
    self->ctx.nUbTail = self->ctx.nBL1Tail % self->ctx.convTiling->bUbNStep;
    self->ctx.nUbTail = self->ctx.nUbTail == 0 ? self->ctx.convTiling->bUbNStep : self->ctx.nUbTail;

    self->ctx.ddr2l1LoopN = CeilDiv(self->ctx.singleCoreCo, self->ctx.convTiling->nBL1);
    self->ctx.maxNBL1Iter = self->ctx.ddr2l1LoopN  - 1;
    self->ctx.l12l0LoopN = self->ctx.convTiling->multiNBL1;
}

template <class Intf>
__aicore__ inline void WeightUbTransInitBuf(Intf *self)
{
    self->ctx.ubBufSize = self->ctx.convTiling->bUbKStep * self->ctx.convTiling->bUbNStep;
    self->ctx.pipe.InitBuffer(self->ctx.ndUbBuf, self->ctx.ubBufSize * Intf::sizeOfWeight);
    self->ctx.pipe.InitBuffer(self->ctx.nzUbBuf, self->ctx.ubBufSize * Intf::sizeOfWeight);
    self->ctx.pipe.InitBuffer(self->ctx.indexUbBuf, REG_SIZE);

    self->ctx.ndTensor = self->ctx.ndUbBuf.template Get<typename Intf::WeightT>();
    self->ctx.nzTensor = self->ctx.nzUbBuf.template Get<typename Intf::WeightT>();

    int8_t al1db = (self->ctx.convTiling->pBufferFlag & AL1_DB_IDX) >> AL1_DB_OFFSET;
    uint32_t aL1SpaceSize = self->ctx.convTiling->aL1SpaceSize;
    if (al1db) {
        aL1SpaceSize *= DOUBLE_BUF;
    }
    self->ctx.bL1SpaceSize = self->ctx.convTiling->nBL1 * self->ctx.convTiling->kBL1;

    self->ctx.pipe.InitBuffer(self->ctx.bL1TBuf,
        aL1SpaceSize + self->ctx.bL1SpaceSize * DOUBLE_BUF * Intf::sizeOfWeight);

    self->ctx.bl1 = self->ctx.bL1TBuf.template Get<typename Intf::WeightT>()[
        aL1SpaceSize / Intf::sizeOfFmap];
}

template <class Intf>
__aicore__ inline void WeightUbTransVecInit(Intf *self)
{
    self->ctx.vecId = get_subblockid();

    self->ctx.singleCoreCi = self->ctx.convTiling->singleCoreCi;
    self->ctx.singleCoreCo = self->ctx.convTiling->singleCoreCo;

    if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
        self->ctx.singleCoreM = self->ctx.convTiling->singleCoreHo;
        self->ctx.mAL1 = self->ctx.convTiling->hoL1;
        self->ctx.mL0 = self->ctx.convTiling->hoL0;
    } else {
        self->ctx.singleCoreHo = self->ctx.convTiling->singleCoreHo;
        self->ctx.singleCoreWo = self->ctx.convTiling->singleCoreWo;
    }

    WeightUbTransInitBuf<Intf>(self);
    WeightUbTransInitKValue<Intf>(self);
    WeightUbTransInitNValue<Intf>(self);
    WeightUbTransInitIterValue<Intf>(self);
    WeightUbTransCalcBL1LoadTimes<Intf>(self);

    self->ctx.weightUbLoadGM2UBTools.SetParams(self);
    self->ctx.weightUbTransND2NZTools.SetParams(self);
    self->ctx.weightUbLoadUB2L1Tools.SetParams(self);
}

template <class Intf>
__aicore__ inline void WeightUbTransUpdateKValue(Intf *self)
{
    if (self->ctx.kBL1Iter == self->ctx.maxKBL1Iter && self->ctx.vecKIter == self->ctx.maxVecKIter) {
        self->ctx.currentUbKStep = self->ctx.kUbTail;
        self->ctx.currentKLoopRpSize = self->ctx.convTiling->bUbKStep - self->ctx.kUbTail;
    } else {
        self->ctx.currentUbKStep = self->ctx.convTiling->bUbKStep;
        self->ctx.currentKLoopRpSize = 0;
    }
}

template <class Intf>
__aicore__ inline void WeightUbTransUpdateNValue(Intf *self)
{
    self->ctx.currentNBL1 = self->ctx.nBL1Iter == self->ctx.maxNBL1Iter ?
        self->ctx.nBL1Tail : self->ctx.convTiling->nBL1;
    self->ctx.l12l0LoopN = CeilDiv(self->ctx.currentNBL1, self->ctx.convTiling->nL0);
    self->ctx.ddr2l1LoopInner = self->ctx.ddr2l1LoopTmp * self->ctx.l12l0LoopN;

    self->ctx.vecNLoopTimes = CeilDiv(self->ctx.currentNBL1, self->ctx.convTiling->bUbNStep);
    self->ctx.maxVecNIter = self->ctx.vecNLoopTimes - 1;

    if (self->ctx.nBL1Iter == self->ctx.maxNBL1Iter && self->ctx.vecNIter == self->ctx.maxVecNIter) {
        self->ctx.currentUbNStep = self->ctx.nUbTail;
        self->ctx.currentUbNStepAilgn = AlignB(self->ctx.currentUbNStep, BLOCK_L0_N);
        self->ctx.currentNLoopRpSize = self->ctx.currentUbNStepAilgn - self->ctx.currentUbNStep;
    } else {
        self->ctx.currentUbNStep = self->ctx.convTiling->bUbNStep;
        self->ctx.currentUbNStepAilgn = self->ctx.convTiling->bUbNStep;
        self->ctx.currentNLoopRpSize = 0;
    }
}

template <class Intf>
__aicore__ inline void WeightUbTransUpdateLoopInner(Intf *self) {
    if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
        self->ctx.ddr2l1LoopTmp = self->ctx.outerIter == self->ctx.maxMAL1Iter ?
            CeilDiv(self->ctx.mAL1Tail, self->ctx.mL0) : self->ctx.l12l0LoopM;
    } else {
        if constexpr (Intf::hasWL1IterFlag) {
            self->ctx.woAL1Iter = self->ctx.outerIter % self->ctx.ddr2l1LoopW;
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
            self->ctx.l12l0LoopW = CeilDiv(self->ctx.currentWoL1, self->ctx.convTiling->woL0);
        }

        if constexpr (Intf::hasHL1IterFlag) {
            self->ctx.hoAL1Iter = (self->ctx.outerIter / self->ctx.ddr2l1LoopW) % self->ctx.ddr2l1LoopH;
            self->ctx.currentHoL1 = self->ctx.hoAL1Iter == self->ctx.maxHoL1Iter ?
                self->ctx.hoAL1Tail : self->ctx.convTiling->hoL1;
        }
        if constexpr (Intf::hasHL0IterFlag) {
            self->ctx.l12l0LoopH = CeilDiv(self->ctx.currentHoL1, self->ctx.convTiling->hoL0);
        }

        self->ctx.ddr2l1LoopTmp = self->ctx.l12l0LoopW * self->ctx.l12l0LoopH;
    }
    self->ctx.ddr2l1LoopInner = self->ctx.ddr2l1LoopTmp * self->ctx.l12l0LoopN;
}


template <class Intf>
__aicore__ inline void WeightUbTransIterInit(Intf *self)
{
    self->ctx.kBL1Iter = 0;
    self->ctx.nBL1Iter = 0;
    self->ctx.innerIter = 0;
    self->ctx.outerIter = 0;
    self->ctx.pingPongFlag = 0;

    self->ctx.vecKIter = 0;
    self->ctx.vecNIter = 0;

    self->ctx.isFirstIterate = true;

    if constexpr (Intf::iterateNFirstFlag) {
        WeightUbTransUpdateLoopInner<Intf>(self);
    }
    self->ctx.ddr2l1LoopInner = self->ctx.ddr2l1LoopTmp * self->ctx.l12l0LoopN;
}

template <class Intf>
__aicore__ inline bool WeightUbTransUbIterUpdate(Intf *self)
{
    if (self->ctx.pingPongFlag == self->ctx.vecId) {
        self->ctx.vecKIter++;
        if (self->ctx.vecKIter == self->ctx.vecKLoopTimes) {
            self->ctx.vecKIter = 0;
        } else {
            return true;
        }

        self->ctx.vecNIter++;
        if (self->ctx.vecNIter == self->ctx.vecNLoopTimes) {
            self->ctx.vecNIter = 0;
        } else {
            return true;
        }
    }

    self->ctx.pingPongFlag ^= 1;

    return false;
}

template <class Intf>
__aicore__ inline bool WeightUbTransIterUpdate(Intf *self)
{
    if (unlikely(self->ctx.isFirstIterate && self->ctx.vecId == 0)) {
        return true;
    }

    if (WeightUbTransUbIterUpdate<Intf>(self)) {
        return true;
    }

    self->ctx.kBL1Iter++;
    if (self->ctx.kBL1Iter == self->ctx.ddr2l1LoopKB) {
        self->ctx.kBL1Iter = 0;
    } else {
        return true;
    }

    // Total iter merge before nBL1Iter.
    self->ctx.innerIter++;
    if (self->ctx.innerIter == self->ctx.ddr2l1LoopInner) {
        self->ctx.innerIter = 0;
    } else {
        return true;
    }

    self->ctx.nBL1Iter++;
    if (self->ctx.nBL1Iter == self->ctx.ddr2l1LoopN) {
        self->ctx.nBL1Iter = 0;
    } else {
        return true;
    }

    if constexpr (Intf::iterateNFirstFlag) {
        // Total iter merge after nBL1Iter((M | HW) * batch).
        self->ctx.outerIter++;
        WeightUbTransUpdateLoopInner<Intf>(self);
        if (self->ctx.outerIter != self->ctx.ddr2l1LoopOuter) {
            return true;
        }
    }

    return false;
}

template <class Intf>
__aicore__ inline bool WeightUbTransVecImpl(Intf *self)
{
    WeightUbTransIterInit<Intf>(self);
    while (WeightUbTransIterUpdate<Intf>(self)) {
        if (self->ctx.pingPongFlag != self->ctx.vecId) {
            continue;
        }

        WeightUbTransUpdateKValue<Intf>(self);
        WeightUbTransUpdateNValue<Intf>(self);

        // For next gm2ub wait vgather2
        event_t eventId = static_cast<event_t>(self->ctx.pipe.FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
        self->ctx.weightUbLoadGM2UBTools.LoadGM2UB();

        // For vgather2 wait gm2ub
        eventId = static_cast<event_t>(self->ctx.pipe.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        // For next vgather2 wait ub2l1
        eventId = static_cast<event_t>(self->ctx.pipe.FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
        self->ctx.weightUbTransND2NZTools.TransND2NZ();

        if (self->ctx.vecKIter == 0 && self->ctx.vecNIter == 0 && !self->ctx.isFirstIterate) {
            CrossCoreWaitFlag<CV_ENHANCE_MODE, PIPE_MTE3>(CV_SYNC_ID_MTE1_MTE3);
        }

        // For ub2l1 wait vgather2
        eventId = static_cast<event_t>(self->ctx.pipe.FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        self->ctx.weightUbLoadUB2L1Tools.LoadUB2L1();

        if (self->ctx.vecKIter == self->ctx.maxVecKIter && self->ctx.vecNIter == self->ctx.maxVecNIter) {
            CrossCoreSetFlag<CV_ENHANCE_MODE, PIPE_MTE3>(CV_SYNC_ID_MTE3_MTE1);
        }

        self->ctx.isFirstIterate = false;
    }

    return false;
}

}

#endif // CONV2D_V2_WEIGHT_UB_TRANS_IMPL_H