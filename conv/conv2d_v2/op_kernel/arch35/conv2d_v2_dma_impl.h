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
 * \file conv2d_v2_dma_impl.h
 * \brief
 */
 
#ifndef CONV2D_V2_DMA_IMPL_H
#define CONV2D_V2_DMA_IMPL_H
 
#include "conv2d_v2_config.h"
#include "conv2d_v2_util.h"
#include "conv2d_v2_instr.h"
 
namespace Conv2dFunc {
using namespace ConvFunc;
using namespace conv;
 
template <class Intf>
__aicore__ inline bool DmaUpdateAL1(Intf *self)
{
    if (self->ctx.loadUB2L1Iter >= self->ctx.aL1LoadTimes) {
        return false;
    }
 
    if (!((self->ctx.kIter == self->ctx.ddr2l0LoopK - 1) || ((self->ctx.kIter + 1) % self->ctx.multiKAL1 == 0))) {
        return false;
    }
 
    // In !kAL1fullload scene, aL1 update in kReduce.
    if (!self->ctx.kAL1fullload) {
        return true;
    }
 
    // In kAL1fullload scene, aL1 update when woL1 update.
    bool updateAL1Flag = false;
    if constexpr (Intf::iterateMFirstFlag) {
        updateAL1Flag = self->ctx.woL0Iter == self->ctx.maxWoL0Iter &&
                        self->ctx.hoL0Iter == self->ctx.maxHoL0Iter &&
                        self->ctx.nL0Iter == self->ctx.maxNL0Iter;
    } else {
        updateAL1Flag = self->ctx.nL0Iter == self->ctx.maxNL0Iter &&
                        self->ctx.woL0Iter == self->ctx.maxWoL0Iter &&
                        self->ctx.hoL0Iter == self->ctx.maxHoL0Iter &&
                        self->ctx.nBL1Iter == self->ctx.maxNBL1Iter;
    }

    return updateAL1Flag;
}
 
template <class Intf>
__aicore__ inline void DmaSyncSet(Intf *self)
{
    if (!DmaUpdateAL1<Intf>(self)) {
        return;
    }
 
    CrossCoreSetFlag<CV_ENHANCE_MODE, PIPE_MTE1>(CV_SYNC_ID_MTE1_MTE3);
    if (self->ctx.convTiling->cinAInCore > Intf::k0) {
        CrossCoreSetFlag<CV_ENHANCE_MODE, PIPE_MTE1>(VEC_ID_MAX + CV_SYNC_ID_MTE1_MTE3);
    }
}
 
template <class Intf>
__aicore__ inline void DmaSyncWait(Intf *self)
{
    CrossCoreWaitFlag<CV_ENHANCE_MODE, PIPE_MTE1>(CV_SYNC_ID_MTE3_MTE1);
    if (self->ctx.convTiling->cinAInCore > Intf::k0) {
        CrossCoreWaitFlag<CV_ENHANCE_MODE, PIPE_MTE1>(VEC_ID_MAX + CV_SYNC_ID_MTE3_MTE1);
    }
    self->ctx.loadUB2L1Iter++;
}
 
template <class Intf>
__aicore__ inline void DmaCalcAL1LoadTimes(Intf *self)
{
    if (!self->ctx.kAL1fullload) {
        uint64_t ddr2L0LoopN = CeilDiv(self->ctx.singleCoreCo, self->ctx.convTiling->nL0);
        uint64_t ddr2L0LoopH = CeilDiv(self->ctx.singleCoreHo, self->ctx.convTiling->hoL0);
        uint64_t ddr2L0LoopW = CeilDiv(self->ctx.singleCoreWo, self->ctx.convTiling->woL0);
        self->ctx.ddr2l1LoopKA = self->ctx.maxKAL1Iter + 1;
 
        self->ctx.aL1LoadTimes = ddr2L0LoopN * ddr2L0LoopH * ddr2L0LoopW * self->ctx.ddr2l1LoopKA *
                                 self->ctx.ddr2l1LoopBatch;
    } else {
        if constexpr (Intf::iterateMFirstFlag) {
            self->ctx.aL1LoadTimes = self->ctx.ddr2l1LoopH * self->ctx.ddr2l1LoopW * self->ctx.ddr2l1LoopN *
                                     self->ctx.ddr2l1LoopBatch;
        } else {
            self->ctx.aL1LoadTimes = self->ctx.ddr2l1LoopH * self->ctx.ddr2l1LoopW * self->ctx.ddr2l1LoopBatch;
        }
    }
}
 
template <class Intf>
__aicore__ inline void DmaUpdateHWValue(Intf *self)
{
    self->ctx.currentHoL1 = self->ctx.hoAL1Iter == self->ctx.maxHoL1Iter ?
        self->ctx.hoAL1Tail : self->ctx.convTiling->hoL1;
 
    self->ctx.currentWoL1 = self->ctx.woAL1Iter == self->ctx.maxWoL1Iter ?
        self->ctx.woAL1Tail : self->ctx.convTiling->woL1;
 
    self->ctx.currentHoL1xWoL1Align = self->ctx.currentHoL1 * AlignB(self->ctx.currentWoL1, BLOCK_L0_M);
 
    if (!self->ctx.kAL1fullload) {
        self->ctx.l12l0LoopH = CeilDiv(self->ctx.currentHoL1, self->ctx.convTiling->hoL0);
        self->ctx.l12l0LoopW = CeilDiv(self->ctx.currentWoL1, self->ctx.convTiling->woL0);
        self->ctx.ddr2l1LoopInner = self->ctx.ddr2l1LoopTmp * self->ctx.l12l0LoopH * self->ctx.l12l0LoopW;
    }
}
 
template <class Intf>
__aicore__ inline void DmaInitKValue(Intf *self)
{
    self->ctx.kAL1Tail = (AlignB(self->ctx.singleCoreCi, Intf::k0) * self->ctx.convTiling->kernelHxkernelW) % self->ctx.convTiling->kAL1;
    self->ctx.kAL1Tail = self->ctx.kAL1Tail == 0 ? self->ctx.convTiling->kAL1 : self->ctx.kAL1Tail;
    if (self->ctx.vecId == 0 && CeilDiv(self->ctx.convTiling->cinATailInCore, Intf::k0) > 1) {
        self->ctx.ciTail = Intf::k0;
    } else {
        self->ctx.ciTail = self->ctx.singleCoreCi % Intf::k0;
        self->ctx.ciTail = self->ctx.ciTail == 0 ? Intf::k0 : self->ctx.ciTail;
    }
 
    self->ctx.ddr2l1LoopKA = CeilDiv(AlignB(self->ctx.singleCoreCi, Intf::k0) * self->ctx.convTiling->kernelHxkernelW,
        self->ctx.convTiling->kAL1);
    self->ctx.maxKAL1Iter = self->ctx.ddr2l1LoopKA - 1;
    self->ctx.kAL1fullload = self->ctx.ddr2l1LoopKA == 1;
 
    uint32_t currentCi1 = self->ctx.convTiling->cinAInCore / Intf::k0;
    self->ctx.currentCi1Ub = CeilDiv(currentCi1, VEC_NUM);
 
    if (self->ctx.vecId == 1) {
        self->ctx.currentVec0Ci = self->ctx.currentCi1Ub * Intf::k0;
        self->ctx.currentCi1Ub = currentCi1 - self->ctx.currentCi1Ub;
        self->ctx.vec0TotalSize = self->ctx.convTiling->hoL1 * self->ctx.convTiling->woL1 * self->ctx.currentVec0Ci *
                                  self->ctx.convTiling->khL1 * self->ctx.convTiling->kwL1;
    }
    self->ctx.maxVecCi1Iter = self->ctx.currentCi1Ub - 1;
}
 
template <class Intf>
__aicore__ inline void DmaInitHWValue(Intf *self)
{
    self->ctx.woAL1Tail = self->ctx.singleCoreWo % self->ctx.convTiling->woL1;
    self->ctx.woAL1Tail = self->ctx.woAL1Tail == 0 ?  self->ctx.convTiling->woL1 : self->ctx.woAL1Tail;
 
    self->ctx.ddr2l1LoopW = CeilDiv(self->ctx.singleCoreWo, self->ctx.convTiling->woL1);
    self->ctx.maxWoL1Iter = self->ctx.ddr2l1LoopW  - 1;
 
    self->ctx.hoAL1Tail = self->ctx.singleCoreHo % self->ctx.convTiling->hoL1;
    self->ctx.hoAL1Tail = self->ctx.hoAL1Tail == 0 ? self->ctx.convTiling->hoL1 : self->ctx.hoAL1Tail;
    self->ctx.ddr2l1LoopH = CeilDiv(self->ctx.singleCoreHo, self->ctx.convTiling->hoL1);
    self->ctx.maxHoL1Iter = self->ctx.ddr2l1LoopH - 1;
 
    DmaUpdateHWValue<Intf>(self);
}
 
template <class Intf>
__aicore__ inline void DmaInitIterValue(Intf *self)
{
    if constexpr (Intf::iterateMFirstFlag) {
        self->ctx.nBL1Tail = self->ctx.singleCoreCo % self->ctx.convTiling->nBL1;
        self->ctx.nBL1Tail = self->ctx.nBL1Tail == 0 ? self->ctx.convTiling->nBL1 : self->ctx.nBL1Tail;
 
        self->ctx.ddr2l1LoopN = CeilDiv(self->ctx.singleCoreCo, self->ctx.convTiling->nBL1);
        self->ctx.maxNBL1Iter = self->ctx.ddr2l1LoopN - 1;
 
        if (!self->ctx.kAL1fullload) {
            self->ctx.ddr2l1LoopTmp = self->ctx.convTiling->multiNBL1;
        }
    } else {
        if (!self->ctx.kAL1fullload) {
            self->ctx.ddr2l1LoopTmp = CeilDiv(self->ctx.singleCoreCo, self->ctx.convTiling->nL0);
        }
    }
}
 
template <class Intf>
__aicore__ inline void DmaInitBuf(Intf *self)
{
    self->ctx.ubBufSize = self->ctx.convTiling->hoL1 * AlignB(self->ctx.convTiling->woL1, BLOCK_L0_M) *
                          self->ctx.convTiling->khUb * self->ctx.convTiling->kwUb * Intf::k0;
 
    self->ctx.pipe.InitBuffer(self->ctx.ubBuf, self->ctx.ubBufSize * Intf::sizeOfFmap);
    self->ctx.img2ColTensor = self->ctx.ubBuf.template Get<typename Intf::FmapT>();
 
    self->ctx.pipe.InitBuffer(self->ctx.aL1TBuf, self->ctx.convTiling->aL1SpaceSize);
    self->ctx.al1 = self->ctx.aL1TBuf.template Get<typename Intf::FmapT>();
}
 
template <class Intf>
__aicore__ inline void DmaVecInit(Intf *self)
{
    self->ctx.vecId = GetSubBlockIdx();
 
    self->ctx.orgCi = self->ctx.convTiling->orgCi;
    self->ctx.orgHi = self->ctx.convTiling->orgHi;
    self->ctx.orgWi = self->ctx.convTiling->orgWi;
    self->ctx.singleCoreCi = self->ctx.convTiling->singleCoreCi;
    self->ctx.singleCoreCo = self->ctx.convTiling->singleCoreCo;
    self->ctx.singleCoreHo = self->ctx.convTiling->singleCoreHo;
    self->ctx.singleCoreWo = self->ctx.convTiling->singleCoreWo;
    self->ctx.vecKhLoopTimes = CeilDiv(self->ctx.convTiling->khL1, self->ctx.convTiling->khUb);
    self->ctx.maxVecKhIter = self->ctx.vecKhLoopTimes - 1;
    self->ctx.vecKwLoopTimes = CeilDiv(self->ctx.convTiling->kwL1, self->ctx.convTiling->kwUb);
    self->ctx.maxVecKwIter = self->ctx.vecKwLoopTimes - 1;
    self->ctx.ddr2L1LoopKh = CeilDiv(self->ctx.convTiling->kernelH, self->ctx.convTiling->khL1);
    self->ctx.maxKhAL1Iter = self->ctx.ddr2L1LoopKh - 1;
    self->ctx.ddr2L1LoopKw = CeilDiv(self->ctx.convTiling->kernelW, self->ctx.convTiling->kwL1);
    self->ctx.maxKwAL1Iter = self->ctx.ddr2L1LoopKw - 1;
    self->ctx.cinAL1 = self->ctx.convTiling->kAL1 / (self->ctx.convTiling->khL1 * self->ctx.convTiling->kwL1);
    self->ctx.cinAL1LoopTimes = CeilDiv(self->ctx.singleCoreCi, self->ctx.cinAL1);
    self->ctx.maxCinAL1Iter = self->ctx.cinAL1LoopTimes - 1;
 
    self->ctx.fmapOneBatchSize = self->ctx.convTiling->orgCi * self->ctx.convTiling->orgHixWi;
 
    DmaInitBuf<Intf>(self);
    DmaInitKValue<Intf>(self);
    DmaInitHWValue<Intf>(self);
    DmaInitIterValue<Intf>(self);
 
    self->ctx.dmaLoadGM2UBTools.SetParams(self);
    self->ctx.dmaLoadUB2L1Tools.SetParams(self);
}

template <class Intf>
__aicore__ inline void DmaCubeInit(Intf *self)
{
    self->ctx.ddr2L1LoopKh = CeilDiv(self->ctx.convTiling->kernelH, self->ctx.convTiling->khL1);
    self->ctx.maxKhAL1Iter = self->ctx.ddr2L1LoopKh - 1;
    self->ctx.ddr2L1LoopKw = CeilDiv(self->ctx.convTiling->kernelW, self->ctx.convTiling->kwL1);
    self->ctx.maxKwAL1Iter = self->ctx.ddr2L1LoopKw - 1;
    self->ctx.cinBL1 = CeilDiv(self->ctx.convTiling->kBL1, self->ctx.convTiling->khL1 * self->ctx.convTiling->kwL1);
    self->ctx.cinBL1LoopTimes = CeilDiv(self->ctx.convTiling->singleCoreCi, self->ctx.cinBL1);
}
 
template <class Intf>
__aicore__ inline void DmaUpdateLoopInner(Intf *self) {
    if (self->ctx.kAL1fullload) {
        return;
    }
 
    if constexpr (Intf::iterateMFirstFlag) {
        self->ctx.ddr2l1LoopTmp = self->ctx.nBL1Iter == self->ctx.maxNBL1Iter ? 
            CeilDiv(self->ctx.nBL1Tail, self->ctx.convTiling->nL0) : self->ctx.convTiling->multiNBL1;
    }
 
    self->ctx.ddr2l1LoopInner = self->ctx.ddr2l1LoopTmp * self->ctx.l12l0LoopH * self->ctx.l12l0LoopW;
}
 
template <class Intf>
__aicore__ inline void DmaIterInit(Intf *self)
{
    self->ctx.kAL1Iter = 0;
    self->ctx.hoAL1Iter = 0;
    self->ctx.woAL1Iter = 0;
    self->ctx.innerIter = 0;
    self->ctx.batchIter = 0;
    self->ctx.nBL1Iter = 0;
 
    self->ctx.vecCi1Iter = 0;
    self->ctx.vecKhIter = 0;
    self->ctx.vecKwIter = 0;
 
    self->ctx.isFirstIterate = true;
 
    DmaUpdateLoopInner<Intf>(self);
}
 
template <class Intf>
__aicore__ inline bool DmaUbIterUpdate(Intf *self)
{
    self->ctx.vecKwIter++;
    if (self->ctx.vecKwIter == self->ctx.vecKwLoopTimes) {
        self->ctx.vecKwIter = 0;
    } else {
        return true;
    }
 
    self->ctx.vecKhIter++;
    if (self->ctx.vecKhIter == self->ctx.vecKhLoopTimes) {
        self->ctx.vecKhIter = 0;
    } else {
        return true;
    }
 
    self->ctx.vecCi1Iter++;
    if (self->ctx.vecCi1Iter == self->ctx.currentCi1Ub) {
        self->ctx.vecCi1Iter = 0;
    } else {
        return true;
    }
 
    return false;
}
 
template <class Intf>
__aicore__ inline bool DmaIterUpdate(Intf *self)
{
    if (unlikely(self->ctx.isFirstIterate)) {
        return true;
    }
 
    if (DmaUbIterUpdate<Intf>(self)) {
        return true;
    }
 
    if (!self->ctx.kAL1fullload) {
        self->ctx.kAL1Iter++;
        if (self->ctx.kAL1Iter == self->ctx.ddr2l1LoopKA) {
            self->ctx.kAL1Iter = 0;
        } else {
            return true;
        }
 
        // Total iter merge before woAL1Iter
        // MFirst：loopWoL0 * loopHoL0 * loopNL0 | NFirst：loopWoL0 * loopHoL0 * loopNL0 * loopNBL1.
        self->ctx.innerIter++;
        if (self->ctx.innerIter == self->ctx.ddr2l1LoopInner) {
            self->ctx.innerIter = 0;
        } else {
            return true;
        }
    }
 
    self->ctx.woAL1Iter++;
    if (self->ctx.woAL1Iter == self->ctx.ddr2l1LoopW) {
        self->ctx.woAL1Iter = 0;
    } else {
        return true;
    }
 
    self->ctx.hoAL1Iter++;
    if (self->ctx.hoAL1Iter == self->ctx.ddr2l1LoopH) {
        self->ctx.hoAL1Iter = 0;
    } else {
        return true;
    }
 
    self->ctx.batchIter++;
    if (self->ctx.batchIter == self->ctx.ddr2l1LoopBatch) {
        self->ctx.batchIter = 0;
    } else {
        return true;
    }
 
    if constexpr (Intf::iterateMFirstFlag) {
        self->ctx.nBL1Iter++;
        DmaUpdateLoopInner<Intf>(self);
        if (self->ctx.nBL1Iter != self->ctx.ddr2l1LoopN) {
            return true;
        }
    }
 
    return false;
}
 
template <class Intf>
__aicore__ inline bool DmaVecImpl(Intf *self)
{
    if (self->ctx.currentCi1Ub == 0) {
        return false;
    }
 
    DmaIterInit<Intf>(self);
    while (DmaIterUpdate<Intf>(self)) {
        DmaUpdateHWValue<Intf>(self);
 
        // For next vdup wait ub2l1
        event_t eventId = static_cast<event_t>(self->ctx.pipe.FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
        self->ctx.dmaLoadGM2UBTools.LoadGM2UB();
 
        if (self->ctx.vecCi1Iter == 0 && self->ctx.vecKhIter == 0 && self->ctx.vecKwIter == 0 &&
            !self->ctx.isFirstIterate) {
            CrossCoreWaitFlag<CV_ENHANCE_MODE, PIPE_MTE3>(CV_SYNC_ID_MTE1_MTE3);
        }
 
        // For ub2l1 wait gm2ub
        eventId = static_cast<event_t>(self->ctx.pipe.FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(eventId);
        WaitFlag<HardEvent::MTE2_MTE3>(eventId);
        self->ctx.dmaLoadUB2L1Tools.LoadUB2L1();
 
        if (self->ctx.vecCi1Iter == self->ctx.maxVecCi1Iter && self->ctx.vecKhIter == self->ctx.maxVecKhIter &&
            self->ctx.vecKwIter == self->ctx.maxVecKwIter) {
            CrossCoreSetFlag<CV_ENHANCE_MODE, PIPE_MTE3>(CV_SYNC_ID_MTE3_MTE1);
        }
 
        self->ctx.isFirstIterate = false;
    }
 
    return false;
}
 
}; // namespace Conv2dFunc

#endif // CONV2D_V2_DMA_IMPL_H
