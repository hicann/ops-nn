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
 * \file conv_iterate_base_impl.h
 * \brief
 */

#ifndef CONV_ITERATE_BASE_IMPL_H
#define CONV_ITERATE_BASE_IMPL_H

#include "conv_config.h"
#include "conv_opt_group_impl.h"
#include "conv_util.h"
#include "../../conv2d_v2/arch35/conv2d_v2_c04_impl.h"
#include "../../conv2d_v2/arch35/conv2d_v2_dma_impl.h"
#include "../../conv2d_v2/arch35/conv2d_v2_weight_ub_trans_impl.h"

namespace ConvFunc {
using namespace AscendC;
using namespace conv;

struct TempIters {
    uint64_t kAL1Iter = 0;
    uint64_t kBL1Iter = 0;
    uint64_t mAL1Iter = 0; // m mode
    uint64_t hoAL1Iter = 0; // hw mode
    uint64_t woAL1Iter = 0; // hw mode
    uint64_t nBL1Iter = 0;
    uint64_t batchIter = 0;
    bool endTag = false;
};

template <class Intf>
static __aicore__ inline void FirstIterateImplVec(Intf *self)
{
    if constexpr (Intf::groupOptNDFlag) {
        self->ctx.loadUB2L1Iter = 0;
    } else if constexpr (Intf::c04NDFlag) {
        self->ctx.loadUB2L1Iter = 0;
        if constexpr (Intf::bL1DBFlag) {
            self->ctx.pingPongFlag = 0;
        }
    } else if constexpr (Intf::weightUbTrans) {
        self->ctx.loadUB2L1Iter = 0;
        self->ctx.pingPongFlag = 0;
    } else if constexpr (Intf::isDmaFlag) {
        self->ctx.loadUB2L1Iter = 0;
    }
}

template <class Intf>
static __aicore__ inline void InitBatchDirectionValue(Intf *self)
{
    if constexpr (Intf::isInnerBatchFlag) {
        self->ctx.ddr2l1LoopBatch = CeilDiv(self->ctx.singleCoreBatch, self->ctx.convTiling->innerBatch);
        self->ctx.maxBatchIter = self->ctx.ddr2l1LoopBatch - 1;
        self->ctx.innerBatchTail = self->ctx.singleCoreBatch % self->ctx.convTiling->innerBatch;
        self->ctx.innerBatchTail = self->ctx.innerBatchTail == 0 ?
            self->ctx.convTiling->innerBatch : self->ctx.innerBatchTail;
    } else {
        self->ctx.ddr2l1LoopBatch = self->ctx.singleCoreBatch;
    }
}

template <class Intf>
static __aicore__ inline void InitCoDirectionValue(Intf *self)
{
    self->ctx.nBL1Tail = self->ctx.singleCoreCo % self->ctx.convTiling->nBL1;
    self->ctx.nBL1Tail = self->ctx.nBL1Tail == 0 ? self->ctx.convTiling->nBL1 : self->ctx.nBL1Tail;
    self->ctx.nL0Tail = self->ctx.nBL1Tail % self->ctx.convTiling->nL0;
    self->ctx.nL0Tail = self->ctx.nL0Tail == 0 ? self->ctx.convTiling->nL0 : self->ctx.nL0Tail;

    if constexpr (!Intf::hasNL1IterFlag) {
        self->ctx.maxNBL1Iter = 0;
        self->ctx.ddr2l1LoopN = 1;
    } else {
        self->ctx.ddr2l1LoopN = CeilDiv(self->ctx.singleCoreCo, self->ctx.convTiling->nBL1);
        self->ctx.maxNBL1Iter = self->ctx.ddr2l1LoopN - 1;
    }

    if constexpr (!Intf::hasNL0IterFlag) {
        self->ctx.l12l0LoopN = 1;
        self->ctx.maxNL0Iter = 0;
    } else if constexpr (Intf::groupOptFlag || Intf::weightUbTrans) {
        self->ctx.l12l0LoopN = self->ctx.nBL1Iter == self->ctx.maxNBL1Iter ?
            CeilDiv(self->ctx.nBL1Tail, self->ctx.convTiling->nL0) : self->ctx.convTiling->multiNBL1;
    }

    if constexpr (Intf::WEIGHT_NZ_FLAG) {
        self->ctx.nBL1Tail = AlignB(self->ctx.nBL1Tail, BLOCK_L0_N);
    }
    self->ctx.currentNBL1 = self->ctx.nBL1Tail;
}

template <class Intf>
__aicore__ inline void CalcCoDirectionVar(Intf *self)
{
    if constexpr (Intf::hasNL0IterFlag) {
        self->ctx.l12l0LoopN = self->ctx.nBL1Iter == self->ctx.maxNBL1Iter ?
            CeilDiv(self->ctx.nBL1Tail, self->ctx.convTiling->nL0) : self->ctx.convTiling->multiNBL1;
        self->ctx.maxNL0Iter = self->ctx.l12l0LoopN - 1;
    }
}

template <class Intf>
__aicore__ inline uint64_t CalcCurrentNL0(Intf *self)
{
    bool isNL0Tail = self->ctx.nBL1Iter == self->ctx.maxNBL1Iter && self->ctx.nL0Iter == self->ctx.maxNL0Iter;
    uint64_t currentNL0 = isNL0Tail ? self->ctx.nL0Tail : self->ctx.convTiling->nL0;
    self->ctx.currentNL0Align = isNL0Tail ? AlignB(currentNL0, BLOCK_L0_N) : currentNL0;

    return currentNL0;
}

template <class Intf>
__aicore__ inline uint64_t CalcCurrentML0MMode(Intf *self)
{
    bool isML0Tail = self->ctx.mAL1Iter == self->ctx.maxMAL1Iter && self->ctx.mL0Iter == self->ctx.maxML0Iter;
    uint64_t currentML0 = isML0Tail ? self->ctx.mAL0Tail : self->ctx.mL0;
    self->ctx.currentML0Align = isML0Tail ? AlignB(currentML0, BLOCK_L0_M) : currentML0;

    return currentML0;
}

template <class Intf>
__aicore__ inline uint64_t CalcCurrentML0HWMode(Intf *self)
{
    if constexpr (Intf::isDmaFlag) {
        uint64_t currentML0 = self->ctx.currentHoL0 * AlignB(self->ctx.currentWoL0, BLOCK_L0_M);
        self->ctx.currentML0Align = currentML0;
 
        return currentML0;
    } else {
        uint64_t currentML0 = self->ctx.currentHoL0 * self->ctx.currentWoL0;
        if (self->ctx.currentWoL0 == self->ctx.convTiling->woL0 &&
            self->ctx.currentHoL0 == self->ctx.convTiling->hoL0) {
            self->ctx.currentML0Align = self->ctx.convTiling->mStep;
        } else {
            self->ctx.currentML0Align = AlignB(currentML0, BLOCK_L0_N);
        }
 
        return currentML0;
    }
}

template <class Intf>
__aicore__ inline void LoadAL1BaseMoudle(Intf *self)
{
    self->ctx.al1 = self->ctx.queueAL1.template AllocTensor<typename Intf::FmapT>();
    if constexpr (Intf::isDmaFlag) {
        Conv2dFunc::DmaSyncWait<Intf>(self);
    } else {
        self->ctx.loadAl1Ins.LoadAL1();
    }
 
    self->ctx.queueAL1.EnQue(self->ctx.al1);
    self->ctx.loadAL1Flag = false;  // Only k directrion can be reloaded in LoopK
}

template <class Intf>
__aicore__ inline void LoadAL1BaseMoudle(Intf *self, TempIters& tempIters)
{
    self->ctx.al1 = self->ctx.queueAL1.template AllocTensor<typename Intf::FmapT>();
    if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
        self->ctx.loadAl1Ins.LoadAL1(tempIters.kAL1Iter, tempIters.mAL1Iter, tempIters.batchIter);
    } else {
        self->ctx.loadAl1Ins.LoadAL1(tempIters.kAL1Iter, tempIters.woAL1Iter, tempIters.hoAL1Iter, tempIters.batchIter);
    }
    self->ctx.queueAL1.EnQue(self->ctx.al1);
    self->ctx.loadAL1Flag = false;  // Only k directrion can be reloaded in LoopK
}

template <class Intf>
__aicore__ inline void LoadBL1BaseMoudle(Intf *self)
{
    if constexpr (Intf::weightUbTrans) {
        self->ctx.bl1 = self->ctx.bL1Tensor[self->ctx.pingPongFlag * self->ctx.bL1SpaceSize];
        CrossCoreWaitFlag<CV_ENHANCE_MODE, PIPE_MTE1>(self->ctx.pingPongFlag * VEC_ID_MAX + CV_SYNC_ID_MTE3_MTE1);
        self->ctx.loadUB2L1Iter++;
    } else {
        self->ctx.bl1 = self->ctx.queueBL1.template AllocTensor<typename Intf::WeightT>();
        if constexpr (Intf::groupOptNDFlag) {
            CrossCoreWaitFlag<CV_ENHANCE_MODE, PIPE_MTE1>(self->ctx.vecId + CV_SYNC_ID_MTE3_MTE1);
            self->ctx.loadUB2L1Iter++;
        } else if constexpr (Intf::c04NDFlag) {
            if constexpr (Intf::bL1DBFlag) {
                CrossCoreWaitFlag<CV_ENHANCE_MODE, PIPE_MTE1>(self->ctx.pingPongFlag * VEC_ID_MAX + CV_SYNC_ID_MTE3_MTE1);
            } else {
                CrossCoreWaitFlag<CV_ENHANCE_MODE, PIPE_MTE1>(CV_SYNC_ID_MTE3_MTE1);
                if (self->ctx.convTiling->nBL1 > BLOCK_L0_N) {
                    CrossCoreWaitFlag<CV_ENHANCE_MODE, PIPE_MTE1>(VEC_ID_MAX + CV_SYNC_ID_MTE3_MTE1);
                }
            }
            self->ctx.loadUB2L1Iter++;
        } else {
            self->ctx.loadBL1Ins.LoadBL1();
        }
        self->ctx.queueBL1.EnQue(self->ctx.bl1);
    }

    self->ctx.loadBL1Flag = false;  // Only k directrion can be reloaded in LoopK
}

template <class Intf>
__aicore__ inline void LoadBL1BaseMoudle(Intf *self, TempIters& tempIters)
{
    self->ctx.bl1 = self->ctx.queueBL1.template AllocTensor<typename Intf::WeightT>();
    if constexpr (Intf::c04NDFlag) {
        if constexpr (Intf::bL1DBFlag) {
            CrossCoreWaitFlag<CV_ENHANCE_MODE, PIPE_MTE1>(self->ctx.pingPongFlag * VEC_ID_MAX + CV_SYNC_ID_MTE3_MTE1);
        } else {
            CrossCoreWaitFlag<CV_ENHANCE_MODE, PIPE_MTE1>(CV_SYNC_ID_MTE3_MTE1);
            if (self->ctx.convTiling->nBL1 > BLOCK_L0_N) {
                CrossCoreWaitFlag<CV_ENHANCE_MODE, PIPE_MTE1>(VEC_ID_MAX + CV_SYNC_ID_MTE3_MTE1);
            }
        }
        self->ctx.loadUB2L1Iter++;
    } else {
        self->ctx.loadBL1Ins.LoadBL1(tempIters.kBL1Iter, tempIters.nBL1Iter);
    }
    self->ctx.queueBL1.EnQue(self->ctx.bl1);
    self->ctx.loadBL1Flag = false;  // Only k directrion can be reloaded in LoopK
}

template <class Intf>
__aicore__ inline void LoadAL1Moudle(Intf *self)
{
    self->ctx.kAL1Iter = self->ctx.kIter / self->ctx.multiKAL1;
    LoadAL1BaseMoudle<Intf>(self);
}

template <class Intf>
__aicore__ inline void LoadBL1Moudle(Intf *self)
{
    self->ctx.kBL1Iter = self->ctx.kIter / self->ctx.multiKBL1;
    if constexpr (Intf::isDmaFlag) {
        self->ctx.kwBL1Iter = self->ctx.kBL1Iter % self->ctx.ddr2L1LoopKw;
        self->ctx.khBL1Iter = (self->ctx.kBL1Iter / self->ctx.ddr2L1LoopKw) % self->ctx.ddr2L1LoopKh;
        self->ctx.cinBL1Iter = (self->ctx.kBL1Iter / (self->ctx.ddr2L1LoopKw * self->ctx.ddr2L1LoopKh)) %
                            self->ctx.cinBL1LoopTimes;
    }
    LoadBL1BaseMoudle<Intf>(self);
}

template <class Intf>
__aicore__ inline void ReduceKCloseL0PingPong(Intf *self, bool isFirst)
{
    event_t eventID = static_cast<event_t>(L0_SYBC_DB_CLOSE);
    wait_flag(PIPE_M, PIPE_MTE1, eventID);
    if (!((self->ctx.convTiling->kAL1 == self->ctx.convTiling->kL0) && (!self->ctx.loadAL0Flag))) {
        self->ctx.al0 = self->ctx.wholeAl0Tensor;
        self->ctx.kAL0Iter = self->ctx.kIter % self->ctx.multiKAL1;
        self->ctx.loadAL0Ins.LoadAL0(isFirst);
        if constexpr (Intf::isDmaFlag) {
            Conv2dFunc::DmaSyncSet<Intf>(self);
        }
    }
    if (!((self->ctx.convTiling->kBL1 == self->ctx.convTiling->kL0) && (!self->ctx.loadBL0Flag))) {
        self->ctx.bl0 = self->ctx.wholeBl0Tensor;
        self->ctx.kBL0Iter = self->ctx.kIter % self->ctx.multiKBL1;
        self->ctx.loadBL0Ins.LoadBL0(isFirst);
        if constexpr (Intf::groupOptNDFlag) {
            OptGroupSyncSet<Intf>(self);
        } else if constexpr (Intf::c04NDFlag) {
            Conv2dFunc::C04SyncSet<Intf>(self);
        } else if constexpr (Intf::weightUbTrans) {
            Conv2dFunc::WeightUbTransSyncSet<Intf>(self);
        }
    }
    set_flag(PIPE_MTE1, PIPE_M, eventID);
    wait_flag(PIPE_MTE1, PIPE_M, eventID);
    self->ctx.madIns.Mad();
    set_flag(PIPE_M, PIPE_MTE1, eventID);
    self->ctx.kIter++;
}

template <class Intf>
__aicore__ inline void ReduceKOpenL0APingPong(Intf *self, uint16_t& al0PingPongFlag, bool isFirst)
{
    event_t eventID = static_cast<event_t>(al0PingPongFlag);
    wait_flag(PIPE_M, PIPE_MTE1, eventID);
    self->ctx.al0 =
        self->ctx.wholeAl0Tensor[(al0PingPongFlag) * L0A_HALF_SIZE / Intf::sizeOfFmap];
    self->ctx.kAL0Iter = self->ctx.kIter % self->ctx.multiKAL1;
    self->ctx.loadAL0Ins.LoadAL0(isFirst);
    if constexpr (Intf::isDmaFlag) {
        Conv2dFunc::DmaSyncSet<Intf>(self);
    }

    if (!((self->ctx.convTiling->kBL1 == self->ctx.convTiling->kL0) && (!self->ctx.loadBL0Flag))) {
        self->ctx.bl0 = self->ctx.wholeBl0Tensor;
        self->ctx.kBL0Iter = self->ctx.kIter % self->ctx.multiKBL1;
        // BL0 wait MMAD
        event_t e = static_cast<event_t>(al0PingPongFlag ^ 1);
        wait_flag(PIPE_M, PIPE_MTE1, e);
        set_flag(PIPE_M, PIPE_MTE1, e);
        self->ctx.loadBL0Ins.LoadBL0(isFirst);
        if constexpr (Intf::groupOptNDFlag) {
            OptGroupSyncSet<Intf>(self);
        } else if constexpr (Intf::c04NDFlag) {
            Conv2dFunc::C04SyncSet<Intf>(self);
        } else if constexpr (Intf::weightUbTrans) {
            Conv2dFunc::WeightUbTransSyncSet<Intf>(self);
        }
    }
    set_flag(PIPE_MTE1, PIPE_M, eventID);
    wait_flag(PIPE_MTE1, PIPE_M, eventID);
    self->ctx.madIns.Mad();
    set_flag(PIPE_M, PIPE_MTE1, eventID);
    al0PingPongFlag ^= 1;
    self->ctx.kIter++;
}

template <class Intf>
__aicore__ inline void ReduceKOpenL0BPingPong(Intf *self, uint16_t& bl0PingPongFlag, bool isFirst)
{
    event_t eventID = static_cast<event_t>(bl0PingPongFlag);
    wait_flag(PIPE_M, PIPE_MTE1, eventID);
 
    self->ctx.bl0 =
        self->ctx.wholeBl0Tensor[(bl0PingPongFlag) * L0B_HALF_SIZE / Intf::sizeOfWeight];
    self->ctx.kBL0Iter = self->ctx.kIter % self->ctx.multiKBL1;
    self->ctx.loadBL0Ins.LoadBL0(isFirst);
    if constexpr (Intf::groupOptNDFlag) {
        OptGroupSyncSet<Intf>(self);
    } else if constexpr (Intf::c04NDFlag) {
        Conv2dFunc::C04SyncSet<Intf>(self);
    } else if constexpr (Intf::weightUbTrans) {
        Conv2dFunc::WeightUbTransSyncSet<Intf>(self);
    }

    if (!((self->ctx.convTiling->kAL1 == self->ctx.convTiling->kL0) && (!self->ctx.loadAL0Flag))) {
        self->ctx.al0 = self->ctx.wholeAl0Tensor;
        self->ctx.kAL0Iter = self->ctx.kIter % self->ctx.multiKAL1;
        // AL0 wait MMAD
        event_t e = static_cast<event_t>((bl0PingPongFlag ^ 1));
        wait_flag(PIPE_M, PIPE_MTE1, e);
        set_flag(PIPE_M, PIPE_MTE1, e);
        self->ctx.loadAL0Ins.LoadAL0(isFirst);
        if constexpr (Intf::isDmaFlag) {
            Conv2dFunc::DmaSyncSet<Intf>(self);
        }
    }

    set_flag(PIPE_MTE1, PIPE_M, eventID);
    wait_flag(PIPE_MTE1, PIPE_M, eventID);
    self->ctx.madIns.Mad();
    set_flag(PIPE_M, PIPE_MTE1, eventID);
    bl0PingPongFlag ^= 1;
    self->ctx.kIter++;
}

template <class Intf>
__aicore__ inline void ReduceKOpenL0AL0BPingPong(Intf *self, uint16_t& al0PingPongFlag, bool isFirst)
{
    self->ctx.al0 =
        self->ctx.wholeAl0Tensor[(al0PingPongFlag) * L0A_HALF_SIZE / Intf::sizeOfFmap];
    self->ctx.bl0 =
        self->ctx.wholeBl0Tensor[(al0PingPongFlag) * L0B_HALF_SIZE / Intf::sizeOfWeight];
    event_t eventID = static_cast<event_t>(al0PingPongFlag);
    wait_flag(PIPE_M, PIPE_MTE1, eventID);
    self->ctx.kAL0Iter = self->ctx.kIter % self->ctx.multiKAL1;
    self->ctx.loadAL0Ins.LoadAL0(isFirst);
    if constexpr (Intf::isDmaFlag) {
        Conv2dFunc::DmaSyncSet<Intf>(self);
    }

    self->ctx.kBL0Iter = self->ctx.kIter % self->ctx.multiKBL1;
    self->ctx.loadBL0Ins.LoadBL0(isFirst);
    if constexpr (Intf::groupOptNDFlag) {
        OptGroupSyncSet<Intf>(self);
    } else if constexpr (Intf::c04NDFlag) {
        Conv2dFunc::C04SyncSet<Intf>(self);
    } else if constexpr (Intf::weightUbTrans) {
        Conv2dFunc::WeightUbTransSyncSet<Intf>(self);
    }
    set_flag(PIPE_MTE1, PIPE_M, eventID);
    wait_flag(PIPE_MTE1, PIPE_M, eventID);
    self->ctx.madIns.Mad();
    set_flag(PIPE_M, PIPE_MTE1, eventID);
    al0PingPongFlag ^= 1;
    self->ctx.kIter++;
}

template <class Intf>
__aicore__ inline void LoadL0Moudle(Intf *self, uint16_t &al0PingPongFlag, uint16_t &bl0PingPongFlag, bool isFirst)
{
    if constexpr (Intf::l0PingPong == static_cast<int32_t>(ConvL0PingPong::ALL_CLOSE)) {
        ReduceKCloseL0PingPong<Intf>(self, isFirst);
    } else if constexpr (Intf::l0PingPong == static_cast<int32_t>(ConvL0PingPong::AL0_OPEN)) {
        ReduceKOpenL0APingPong<Intf>(self, al0PingPongFlag, isFirst);
    } else if constexpr (Intf::l0PingPong == static_cast<int32_t>(ConvL0PingPong::BL0_OPEN)) {
        ReduceKOpenL0BPingPong<Intf>(self, bl0PingPongFlag, isFirst);
    } else if constexpr (Intf::l0PingPong == static_cast<int32_t>(ConvL0PingPong::ALL_OPEN)) {
        ReduceKOpenL0AL0BPingPong<Intf>(self, al0PingPongFlag, isFirst);
    }
}

template <class Intf>
__aicore__ inline void SetMNBeforeIterateK(Intf *self)
{
    if constexpr (Intf::isInnerBatchFlag) {
        self->ctx.innerBatch = self->ctx.batchIter == self->ctx.maxBatchIter ?
            self->ctx.innerBatchTail : self->ctx.convTiling->innerBatch;
    }

    uint64_t currentML0 = 0;
    if constexpr (Intf::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE)) {
        currentML0 = CalcCurrentML0MMode<Intf>(self);
    } else {
        currentML0 = CalcCurrentML0HWMode<Intf>(self);
    }
    uint64_t currentNL0 = CalcCurrentNL0<Intf>(self);
    self->ctx.loadAL0Ins.SetM(self->ctx.currentML0Align, currentML0);
    self->ctx.loadBL0Ins.SetN(self->ctx.currentNL0Align);
    self->ctx.loadBiasL1Ins.SetN(currentNL0);
    self->ctx.loadBiasBTIns.SetN(self->ctx.currentNL0Align);
    self->ctx.loadScaleL1Ins.SetN(currentNL0);
    if constexpr (Intf::ConvParam::innerBatch == static_cast<int8_t>(ConvInnerBatch::KERNEL_1X1_MULTI_BATCH)) {
        self->ctx.madIns.SetMN(currentML0, self->ctx.currentNL0Align);
    } else {
        self->ctx.madIns.SetMN(self->ctx.currentML0Align, self->ctx.currentNL0Align);
    }

    self->ctx.copyOutIns.SetMN(currentML0, currentNL0);
    if constexpr (Intf::isExtendConv2d) {
        self->ctx.copyOutIns1.SetMN(currentML0, currentNL0);
    }
}

template <class Intf>
__aicore__ inline void FreeL1Tensor(Intf *self)
{
    if (!self->ctx.kAL1fullload) {
        self->ctx.queueAL1.FreeTensor(self->ctx.al1);
    }

    if constexpr (!Intf::weightUbTrans) {
        if (!self->ctx.kBL1fullload) {
            self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
        }
    }

    if (self->ctx.enableBias) {
        if (!self->ctx.convTiling->biasFullLoadFlag) {
            self->ctx.queueBiasL1.FreeTensor(self->ctx.biasL1);
        }

        self->ctx.queueBiasBT.FreeTensor(self->ctx.biasBT);
    }
}

template <class Intf>
__aicore__ inline bool CheckReduceOneKNotSupportDBCase(Intf *self)
{
    // when only L0A pingpong and L0B not pingpong, L0B tensor need full load for template ReduceKOpenL0APingPong
    if constexpr (Intf::l0PingPong == 1) {
        if (self->ctx.convTiling->nL0 < self->ctx.singleCoreCo) {
            return true;
        }
    }
    // when only L0B pingpong and L0A not pingpong, L0A tensor need full load for template ReduceKOpenL0BPingPong
    if constexpr (Intf::l0PingPong == 2) {
        if (self->ctx.convTiling->hoL0 < self->ctx.convTiling->singleCoreHo ||
            self->ctx.convTiling->woL0 < self->ctx.convTiling->singleCoreWo) {
            return true;
        }
    }

    if (self->ctx.ddr2l1LoopBatch > 1) {
        return true;
    }

    if constexpr (Intf::formatOutput == ConvFormat::NCDHW || Intf::formatOutput == ConvFormat::NDHWC) {
        if (self->ctx.ddr2l1LoopD > 1) {
            return true;
        }
    }

    if constexpr (Intf::groupOptFlag) {
        if (self->ctx.singleGroupOpt > 1) {
            return true;
        }
    }

    return false;
}

};

#endif // CONV_ITERATE_BASE_IMPL_H