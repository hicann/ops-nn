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
 * \file conv2d_v2_c04_impl.h
 * \brief
 */

#ifndef CONV2D_V2_C04_IMPL_H
#define CONV2D_V2_C04_IMPL_H

#include "conv2d_v2_config.h"
#include "conv2d_v2_util.h"
#include "conv2d_v2_instr.h"

namespace Conv2dFunc {
using namespace ConvFunc;
using namespace conv;

template <class Intf>
class C04ProcessTools {
public:
    __aicore__ inline C04ProcessTools() {}

    __aicore__ inline bool C04UpdateBL1(Intf *self)
    {
        if (!((self->ctx.kIter == self->ctx.ddr2l0LoopK - 1) ||
            ((self->ctx.kIter + 1) % self->ctx.multiKBL1 == 0))) {
            return false;
        }

        // In kBL1fullload scene, bL1 update when nBL1 update.
        bool updateBL1Flag = false;
        if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::HW_MODE)) {
            if constexpr (Intf::iterateMFirstFlag) {
                updateBL1Flag = self->ctx.woL0Iter == self->ctx.maxWoL0Iter &&
                                self->ctx.hoL0Iter == self->ctx.maxHoL0Iter &&
                                self->ctx.nL0Iter == self->ctx.maxNL0Iter &&
                                self->ctx.woAL1Iter == self->ctx.maxWoL1Iter &&
                                self->ctx.hoAL1Iter == self->ctx.maxHoL1Iter &&
                                self->ctx.batchIter == self->ctx.ddr2l1LoopBatch - 1;
            } else {
                updateBL1Flag = self->ctx.nL0Iter == self->ctx.maxNL0Iter &&
                                self->ctx.woL0Iter == self->ctx.maxWoL0Iter &&
                                self->ctx.hoL0Iter == self->ctx.maxHoL0Iter;
            }
        } else if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
            if constexpr (Intf::iterateMFirstFlag) {
                updateBL1Flag = self->ctx.mL0Iter == self->ctx.maxML0Iter &&
                                self->ctx.nL0Iter == self->ctx.maxNL0Iter &&
                                self->ctx.mAL1Iter == self->ctx.maxMAL1Iter &&
                                self->ctx.batchIter == self->ctx.ddr2l1LoopBatch - 1;
            } else {
                updateBL1Flag = self->ctx.nL0Iter == self->ctx.maxNL0Iter &&
                                self->ctx.mL0Iter == self->ctx.maxML0Iter;
            }
        }

        return updateBL1Flag;
    }

    __aicore__ inline void C04SyncSet(Intf *self)
    {
        if (!C04UpdateBL1(self)) {
            return;
        }

        if (self->ctx.loadUB2L1Iter < self->ctx.bL1LoadTimes) {
            if constexpr (Intf::bL1DBFlag) {
                CrossCoreSetFlag<CV_ENHANCE_MODE, PIPE_MTE1>(self->ctx.pingPongFlag * VEC_ID_MAX + CV_SYNC_ID_MTE1_MTE3);
            } else {
                CrossCoreSetFlag<CV_ENHANCE_MODE, PIPE_MTE1>(CV_SYNC_ID_MTE1_MTE3);
                // Each core deal with half of nBL1 at least n0, when nBL1 <= n0, vec1 doesn't have data to deal.
                if (self->ctx.convTiling->nBL1 > BLOCK_L0_N) {
                    CrossCoreSetFlag<CV_ENHANCE_MODE, PIPE_MTE1>(VEC_ID_MAX + CV_SYNC_ID_MTE1_MTE3);
                }
            }
        }

        if constexpr (Intf::bL1DBFlag) {
            self->ctx.pingPongFlag ^= 1;
        }
    }

    __aicore__ inline void C04InitIterValue(Intf *self)
    {
        if constexpr (Intf::iterateMFirstFlag) {
            self->ctx.bL1LoadTimes = self->ctx.ddr2l1LoopN;
        } else {
            if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::HW_MODE)) {
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

                self->ctx.ddr2l1LoopOuter = self->ctx.ddr2l1LoopBatch * self->ctx.ddr2l1LoopH * self->ctx.ddr2l1LoopW;
            } else if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
                self->ctx.ddr2l1LoopM = CeilDiv(self->ctx.singleCoreM, self->ctx.mAL1);
                self->ctx.ddr2l1LoopOuter = self->ctx.ddr2l1LoopBatch * self->ctx.ddr2l1LoopM;
            }

            if ASCEND_IS_AIC_CONV {
                self->ctx.bL1LoadTimes = self->ctx.ddr2l1LoopOuter * self->ctx.ddr2l1LoopN;
                if constexpr (Intf::bL1DBFlag) {
                    // In DB scenc, set_intra_block need sub 1.
                    self->ctx.bL1LoadTimes -= 1;
                }
            }
        }
    }

    __aicore__ inline void C04InitNValue(Intf *self)
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
    }

    __aicore__ inline void C04InitBuf(Intf *self)
    {
        self->ctx.ubBufSize = self->ctx.convTiling->bUbNStep * self->ctx.convTiling->kBL1;
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

    __aicore__ inline void C04VecInit(Intf *self)
    {
        self->ctx.vecId = GetSubBlockIdx();

        self->ctx.singleCoreCo = self->ctx.convTiling->singleCoreCo;
        self->ctx.kSizeC04 = self->ctx.convTiling->kernelHxkernelW * C04_CIN_SIZE;
        self->ctx.ddr2l1LoopBatch = self->ctx.convTiling->singleCoreBatch;

        if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
            self->ctx.singleCoreM = self->ctx.convTiling->singleCoreHo;
            self->ctx.mAL1 = self->ctx.convTiling->hoL1;
            self->ctx.mL0 = self->ctx.convTiling->hoL0;
        } else {
            self->ctx.singleCoreHo = self->ctx.convTiling->singleCoreHo;
            self->ctx.singleCoreWo = self->ctx.convTiling->singleCoreWo;
        }

        C04InitBuf(self);
        C04InitNValue(self);
        C04InitIterValue(self);

        self->ctx.c04DupTools.SetParams(self);
        self->ctx.c04LoadGm2UBTools.SetParams(self);
        self->ctx.c04TransFractalZC04Tools.SetParams(self);
        self->ctx.c04LoadUB2L1Tools.SetParams(self);
    }

    __aicore__ inline void C04UpdateNUbValue(Intf *self)
    {
        self->ctx.currentUbNStep = self->ctx.vecNIter == self->ctx.maxVecNIter ?
                self->ctx.bUbNTailStep : self->ctx.convTiling->bUbNStep;
        self->ctx.currentUbNStepAilgn = AlignB(self->ctx.currentUbNStep, BLOCK_L0_N);
    }

    __aicore__ inline void C04UpdateNL1Value(Intf *self)
    {
        self->ctx.currentNBL1 = self->ctx.nBL1Iter == self->ctx.maxNBL1Iter ?
            self->ctx.nBL1Tail : self->ctx.convTiling->nBL1;
        self->ctx.currentNBL1Align = AlignB(self->ctx.currentNBL1, BLOCK_L0_N);

        if constexpr (!Intf::bL1DBFlag) {
            if (self->ctx.currentNBL1 <= BLOCK_L0_N) {
                if (self->ctx.vecId == 1) {
                    self->ctx.currentNBL1 = 0;
                    return;
                }
            } else {
                self->ctx.nBL1Vec0 = AlignB(self->ctx.currentNBL1 / VEC_NUM, BLOCK_L0_N);
                self->ctx.currentNBL1 = self->ctx.vecId == 0 ?
                    self->ctx.nBL1Vec0 : self->ctx.currentNBL1 - self->ctx.nBL1Vec0;
            }
        }

        self->ctx.vecNLoopTimes = CeilDiv(self->ctx.currentNBL1, self->ctx.convTiling->bUbNStep);
        self->ctx.maxVecNIter = self->ctx.vecNLoopTimes - 1;
        self->ctx.bUbNTailStep = self->ctx.currentNBL1 % self->ctx.convTiling->bUbNStep;
        self->ctx.bUbNTailStep = self->ctx.bUbNTailStep == 0 ? self->ctx.convTiling->bUbNStep : self->ctx.bUbNTailStep;

        self->ctx.currentNLoopRpSize =
            self->ctx.nBL1Iter == self->ctx.maxNBL1Iter && self->ctx.vecNIter == self->ctx.maxVecNIter ?
            AlignB(self->ctx.singleCoreCo, BLOCK_L0_N) - self->ctx.singleCoreCo : 0;
    }

    __aicore__ inline void C04UB2L1IterInit(Intf *self)
    {
        if constexpr (Intf::hasNL1IterFlag) {
            self->ctx.nBL1Iter = 0;
        }

        if constexpr (Intf::iterateNFirstFlag) {
            self->ctx.outerIter = 0;
        }

        if constexpr (Intf::bL1DBFlag) {
            self->ctx.pingPongFlag = 0;
        }

        self->ctx.vecNIter = 0;
        self->ctx.isFirstIterate = true;

        C04UpdateNL1Value(self);
    }

    __aicore__ inline bool C04UB2L1Iter(Intf *self)
    {
        if constexpr (Intf::bL1DBFlag) {
            if (unlikely(self->ctx.isFirstIterate && self->ctx.vecId == 0)) {
                return true;
            }
        } else {
            if (unlikely(self->ctx.isFirstIterate)) {
                return true;
            }
        }

        self->ctx.vecNIter++;
        if (self->ctx.vecNIter == self->ctx.vecNLoopTimes) {
            self->ctx.vecNIter = 0;
        } else {
            return true;
        }

        if constexpr (Intf::bL1DBFlag) {
            self->ctx.pingPongFlag ^= 1;
        }

        if constexpr (Intf::hasNL1IterFlag) {
            self->ctx.nBL1Iter++;
            if (self->ctx.nBL1Iter == self->ctx.ddr2l1LoopN) {
                self->ctx.nBL1Iter = 0;
                C04UpdateNL1Value(self);
            } else {
                C04UpdateNL1Value(self);
                return true;
            }
        }

        if constexpr (Intf::iterateNFirstFlag) {
            // Total iter merge after nBL1Iter((M | HW) * batch).
            self->ctx.outerIter++;
            if (self->ctx.outerIter != self->ctx.ddr2l1LoopOuter) {
                return true;
            }
        }

        return false;
    }

    __aicore__ inline void C04UBImpl(Intf *self)
    {
        // For first nddma wait vdup and next nddma wait nd2nz
        event_t eventId = static_cast<event_t>(self->ctx.pipe.FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
        self->ctx.c04LoadGm2UBTools.LoadGM2UB();

        // For nd2nz wait nddma
        eventId = static_cast<event_t>(self->ctx.pipe.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        // For next nd2nz wait ub2l1
        eventId = static_cast<event_t>(self->ctx.pipe.FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
        self->ctx.c04TransFractalZC04Tools.TransFractalZC04();

        if (self->ctx.vecNIter == 0 && !self->ctx.isFirstIterate) {
            CrossCoreWaitFlag<CV_ENHANCE_MODE, PIPE_MTE3>(CV_SYNC_ID_MTE1_MTE3);
        }

        // For ub2l1 wait nd2nz
        eventId = static_cast<event_t>(self->ctx.pipe.FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        self->ctx.c04LoadUB2L1Tools.LoadUB2L1();

        if (self->ctx.vecNIter == self->ctx.maxVecNIter) {
            CrossCoreSetFlag<CV_ENHANCE_MODE, PIPE_MTE3>(CV_SYNC_ID_MTE3_MTE1);
        }
    }

    __aicore__ inline bool C04VecImpl(Intf *self)
    {
        if constexpr (!Intf::bL1DBFlag) {
            if (self->ctx.convTiling->nBL1 <= BLOCK_L0_N && self->ctx.vecId == 1) {
                return false;
            }
        }
        self->ctx.c04DupTools.KAlignDup();

        C04UB2L1IterInit(self);
        while (C04UB2L1Iter(self)) {
            if constexpr (Intf::bL1DBFlag) {
                if (self->ctx.pingPongFlag != self->ctx.vecId) {
                    continue;
                }
            } else {
                if (unlikely(self->ctx.currentNBL1 == 0 && self->ctx.vecNIter == 0)) {
                    if (!self->ctx.isFirstIterate) {
                        CrossCoreWaitFlag<CV_ENHANCE_MODE, PIPE_MTE3>(CV_SYNC_ID_MTE1_MTE3);
                    }
                    CrossCoreSetFlag<CV_ENHANCE_MODE, PIPE_MTE3>(CV_SYNC_ID_MTE3_MTE1);
                    continue;
                }
            }

            C04UpdateNUbValue(self);

            C04UBImpl(self);

            self->ctx.isFirstIterate = false;
        }

        return false;
    }
};


}; // namespace Conv2dFunc

#endif // CONV2D_V2_C04_IMPL_H