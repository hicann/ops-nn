/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file conv_iterate_base_impl_arch51.h
 * \brief
 */

#ifndef CONV_ITERATE_BASE_IMPL_5162_H
#define CONV_ITERATE_BASE_IMPL_5162_H

#include "../arch35/conv_iterate_base_impl.h"
#include "conv_instr_impl_arch51.h"

template <class Intf>
__aicore__ inline void CalcCoDirectionVar(Intf* self, IterateParam& iterParam)
{
    if constexpr (Intf::hasNL0IterFlag) {
        iterParam.l12l0LoopN = iterParam.nBL1Iter == iterParam.maxNBL1Iter ?
                                   CeilDiv(iterParam.nBL1Tail, self->ctx.convTilingData->nL0) :
                                   self->ctx.convTilingData->multiNBL1;
        iterParam.maxNL0Iter = iterParam.l12l0LoopN - 1;
    }
}

template <class Intf>
__aicore__ inline void LoadAL1BaseModule(Intf* self, IterateParam& iterParam)
{
    self->ctx.al1 = self->ctx.queueAL1.template AllocTensor<typename Intf::FmapT>();
    if constexpr (Intf::kl0FullLoadFlag ||
                  Intf::convToL1Mode == static_cast<int8_t>(ConvToL1::OPEN_WEIGHT_L1_WITH_CUT_KERNEL)) {
        event_t eventId = static_cast<event_t>(self->ctx.pipe.FetchEventID(AscendC::HardEvent::M_MTE2));
        AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(eventId);
    }
    self->ctx.loadAl1Ins.LoadAL1(iterParam);
    self->ctx.queueAL1.EnQue(self->ctx.al1);
    self->ctx.loadAL1Flag = false;
}

template <class Intf>
__aicore__ inline void LoadBL1BaseModule(Intf* self, IterateParam& iterParam)
{
    self->ctx.bl1 = self->ctx.queueBL1.template AllocTensor<typename Intf::WeightT>();
    self->ctx.loadBL1Ins.LoadBL1(iterParam);
    self->ctx.queueBL1.EnQue(self->ctx.bl1);
    self->ctx.loadBL1Flag = false;
}

template <class Intf>
__aicore__ inline void LoadAL1Module(Intf* self, IterateParam& iterParam)
{
    iterParam.kAL1Iter = iterParam.kIter / iterParam.multiKAL1;
    LoadAL1BaseModule<Intf>(self, iterParam);
}

template <class Intf>
__aicore__ inline void LoadBL1Module(Intf* self, IterateParam& iterParam)
{
    iterParam.kBL1Iter = iterParam.kIter / iterParam.multiKBL1;
    LoadBL1BaseModule<Intf>(self, iterParam);
}

template <class Intf>
__aicore__ inline uint64_t CalcCurrentNL0(Intf* self, IterateParam& iterParam)
{
    if constexpr (Intf::isKL1NL0FullLoad) {
        iterParam.currentNL0Align = AlignB(iterParam.nL0Tail, BLOCK_L0_N);
        return iterParam.nL0Tail;
    } else {
        bool isNL0Tail = iterParam.nBL1Iter == iterParam.maxNBL1Iter && iterParam.nL0Iter == iterParam.maxNL0Iter;
        uint64_t currentNL0 = isNL0Tail ? iterParam.nL0Tail : self->ctx.convTilingData->nL0;
        iterParam.currentNL0Align = isNL0Tail ? AlignB(currentNL0, BLOCK_L0_N) : currentNL0;
        return currentNL0;
    }
}

template <class Intf>
__aicore__ inline void ReduceKCloseL0PingPongFor5162(Intf* self, const uint64_t& KStartPosition, const uint64_t& kStep,
                                                     const ConvFixBitModeParams& convFixParams, IterateParam& iterParam)
{
    event_t eventID = static_cast<event_t>(L0_SYBC_DB_CLOSE);
    constexpr bool isConvToL1WeightByPass = Intf::convToL1Mode == static_cast<int8_t>(ConvToL1::OPEN_WEIGHT_BY_PASS);
    if constexpr (!isConvToL1WeightByPass) {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventID);
    }

    bool aL0ProcessFlag = (self->ctx.convTilingData->kAL1 != self->ctx.convTilingData->kL0) || self->ctx.loadAL0Flag ||
                          Intf::convToL1Mode == static_cast<int8_t>(ConvToL1::OPEN_WEIGHT_L1_WITH_CUT_KERNEL);
    uint64_t kAL0Iter = 0;
    if (aL0ProcessFlag) {
        kAL0Iter = iterParam.kIter % iterParam.multiKAL1;
    }

    LocalTensor<typename Intf::WeightT> bl0;
    if constexpr (Intf::isL0BFullLoadable) {
        bl0 = self->ctx.wholeBl0Tensor[iterParam.kIter * iterParam.currentNL0Align * self->ctx.convTilingData->kL0];
    } else {
        bl0 = self->ctx.wholeBl0Tensor;
    }
    bool bL0ProcessFlag = (self->ctx.convTilingData->kBL1 != self->ctx.convTilingData->kL0) || self->ctx.loadBL0Flag ||
                          Intf::convToL1Mode == static_cast<int8_t>(ConvToL1::OPEN_WEIGHT_L1_WITH_CUT_KERNEL);
    if (bL0ProcessFlag) {
        iterParam.kBL0Iter = iterParam.kIter % iterParam.multiKBL1;
        uint64_t bgmOffset = 0;
        if constexpr (isConvToL1WeightByPass) {
            bgmOffset = self->ctx.loadBL0Ins.CalcOffsetForWeightByPass(iterParam);
        }
        self->ctx.loadBL0Ins.LoadBL0(KStartPosition, kStep, bl0, bgmOffset);
    }

    if constexpr (!isConvToL1WeightByPass) {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(eventID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(eventID);
    }
    uint32_t tempL1CinOffset = kAL0Iter * self->ctx.convTilingData->kL0 / self->ctx.convTilingData->kernelHxkernelW *
                               iterParam.currentL1Hin;
    const LocalTensor<typename Intf::FmapT>& al0 = self->ctx.al1[tempL1CinOffset];
    self->ctx.madIns.Mad(al0, bl0, convFixParams, iterParam);
    if constexpr (!isConvToL1WeightByPass) {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(eventID);
    }
}

template <class Intf>
__aicore__ inline void ReduceKOpenL0BPingPongFor5162(Intf* self, const uint16_t& bl0PingPongFlag,
                                                     const uint64_t& KStartPosition, const uint64_t& kStep,
                                                     const ConvFixBitModeParams& convFixParams, IterateParam& iterParam)
{
    event_t eventID = static_cast<event_t>(bl0PingPongFlag);
    LocalTensor<typename Intf::WeightT> bl0;
    constexpr bool isConvToL1WeightByPass = Intf::convToL1Mode == static_cast<int8_t>(ConvToL1::OPEN_WEIGHT_BY_PASS);
    if constexpr (!isConvToL1WeightByPass) {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(eventID);
    }

    bool aL0ProcessFlag = (self->ctx.convTilingData->kAL1 != self->ctx.convTilingData->kL0) || self->ctx.loadAL0Flag ||
                          Intf::convToL1Mode == static_cast<int8_t>(ConvToL1::OPEN_WEIGHT_L1_WITH_CUT_KERNEL);
    uint64_t kAL0Iter = 0;
    if (aL0ProcessFlag) {
        kAL0Iter = iterParam.kIter % iterParam.multiKAL1;
    }

    if constexpr (Intf::isL0BFullLoadable) {
        bl0 = self->ctx.wholeBl0Tensor[iterParam.kIter * iterParam.currentNL0Align * self->ctx.convTilingData->kL0];
    } else {
        bl0 = self->ctx.wholeBl0Tensor[(bl0PingPongFlag)*L0B_HALF_SIZE / Intf::sizeOfWeight];
        uint64_t bgmOffset = 0;
        if constexpr (isConvToL1WeightByPass) {
            bgmOffset = self->ctx.loadBL0Ins.CalcOffsetForWeightByPass(iterParam);
        }
        self->ctx.loadBL0Ins.LoadBL0(KStartPosition, kStep, bl0, bgmOffset);
    }

    if constexpr (!isConvToL1WeightByPass) {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(eventID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(eventID);
    }
    uint32_t tempL1CinOffset = kAL0Iter * self->ctx.convTilingData->kL0 / self->ctx.convTilingData->kernelHxkernelW *
                               iterParam.currentL1Hin;
    const LocalTensor<typename Intf::FmapT>& al0 = self->ctx.al1[tempL1CinOffset];
    self->ctx.madIns.Mad(al0, bl0, convFixParams, iterParam);
    if constexpr (!isConvToL1WeightByPass) {
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(eventID);
    }
}

template <class Intf>
__aicore__ inline void LoadL0ModuleFor5162(Intf* self, const uint64_t& KStartPosition, const uint64_t& kStep,
                                           const ConvFixBitModeParams& convFixParams, IterateParam& iterParam)
{
    if constexpr (Intf::l0PingPong == static_cast<int32_t>(ConvL0PingPong::AL0_OPEN) ||
                  Intf::l0PingPong == static_cast<int32_t>(ConvL0PingPong::ALL_CLOSE)) {
        ReduceKCloseL0PingPongFor5162<Intf>(self, KStartPosition, kStep, convFixParams, iterParam);
    } else {
        ReduceKOpenL0BPingPongFor5162<Intf>(self, iterParam.kIter & 0x1, KStartPosition, kStep, convFixParams,
                                            iterParam);
    }
}

template <class Intf>
__aicore__ inline uint64_t CalcCurrentML0HWMode(Intf* self, IterateParam& iterParam)
{
    uint64_t currentML0 = iterParam.currentHoL0 * iterParam.currentWoL0;
    if (iterParam.currentWoL0 == self->ctx.convTilingData->woL0 &&
        iterParam.currentHoL0 == self->ctx.convTilingData->hoL0) {
        iterParam.currentML0Align = self->ctx.convTilingData->mStep;
    } else {
        iterParam.currentML0Align = AlignB(currentML0, BLOCK_L0_N);
    }

    return currentML0;
}

template <class Intf>
__aicore__ inline void SetMNBeforeIterateK(Intf* self, IterateParam& iterParam)
{
    uint64_t currentML0 = CalcCurrentML0HWMode<Intf>(self, iterParam);
    uint64_t currentNL0 = CalcCurrentNL0<Intf>(self, iterParam);
    self->ctx.loadBL0Ins.SetN(iterParam);
    self->ctx.loadBiasL1Ins.SetN(currentNL0);
    self->ctx.loadBiasBTIns.SetN(iterParam.currentNL0Align);
    self->ctx.loadScaleL1Ins.SetN(currentNL0);
    self->ctx.loadReluWeightL1Ins.SetN(currentNL0);
    self->ctx.copyOutIns.SetMN(currentML0, currentNL0);

    if constexpr (Intf::isExtendConv2d) {
        self->ctx.copyOutIns1.SetMN(currentML0, currentNL0);
    }
}

template <class Intf>
__aicore__ inline void SetFirstBeforeIterateK(Intf* self, IterateParam& iterParam)
{
    if constexpr (!Intf::isL0BFullLoadable) {
        self->ctx.loadBL0Ins.SetFirst(iterParam);
    }
}

template <class Intf>
__aicore__ inline void CalcWoDirectionVar(Intf* self, IterateParam& iterParam)
{
    if constexpr (Intf::hasWL1IterFlag) {
        if (iterParam.woL1SmallTail > 0) {
            if (iterParam.woAL1Iter == iterParam.maxWoL1Iter) {
                iterParam.currentWoL1 = iterParam.woL1SmallTail;
            } else if (iterParam.woAL1Iter == iterParam.maxWoL1Iter - 1) {
                iterParam.currentWoL1 = iterParam.woAL1Tail;
            } else {
                iterParam.currentWoL1 = self->ctx.convTilingData->woL1;
            }
        } else {
            iterParam.currentWoL1 = iterParam.woAL1Iter == iterParam.maxWoL1Iter ? iterParam.woAL1Tail :
                                                                                   self->ctx.convTilingData->woL1;
        }
    }
    if constexpr (Intf::hasWL0IterFlag) {
        iterParam.woL0Tail = iterParam.currentWoL1 % self->ctx.convTilingData->woL0;
        iterParam.woL0Tail = iterParam.woL0Tail == 0 ? self->ctx.convTilingData->woL0 : iterParam.woL0Tail;
        iterParam.maxWoL0Iter = CeilDiv(iterParam.currentWoL1, self->ctx.convTilingData->woL0) - 1;
        iterParam.l12l0LoopW = iterParam.maxWoL0Iter + 1;
    }
}

template <class Intf>
__aicore__ inline void CalcHoDirectionVar(Intf* self, IterateParam& iterParam)
{
    if constexpr (Intf::hasHL1IterFlag) {
        iterParam.currentHoL1 = iterParam.hoAL1Iter == iterParam.maxHoL1Iter ? iterParam.hoAL1Tail :
                                                                               self->ctx.convTilingData->hoL1;
    }

    if constexpr (Intf::hasHL0IterFlag) {
        iterParam.hoL0Tail = iterParam.currentHoL1 % self->ctx.convTilingData->hoL0;
        iterParam.hoL0Tail = iterParam.hoL0Tail == 0 ? self->ctx.convTilingData->hoL0 : iterParam.hoL0Tail;
        iterParam.maxHoL0Iter = CeilDiv(iterParam.currentHoL1, self->ctx.convTilingData->hoL0) - 1;
        iterParam.l12l0LoopH = iterParam.maxHoL0Iter + 1;
    }
}

template <class Intf>
__aicore__ inline void UpdateHoL0WoL0(Intf* self, IterateParam& iterParam)
{
    iterParam.currentHoL0 = iterParam.hoL0Iter == iterParam.maxHoL0Iter ? iterParam.hoL0Tail :
                                                                          self->ctx.convTilingData->hoL0;
    iterParam.currentWoL0 = iterParam.woL0Iter == iterParam.maxWoL0Iter ? iterParam.woL0Tail :
                                                                          self->ctx.convTilingData->woL0;
}

template <class Intf>
__aicore__ inline void FirstIterateImplHWMode(Intf* self, IterateParam& iterParam)
{
    iterParam.nL0Iter = 0;
    iterParam.hoAL1Iter = 0;
    iterParam.woAL1Iter = 0;
    iterParam.nBL1Iter = 0;
    iterParam.woL0Iter = 0;
    iterParam.hoL0Iter = 0;

    iterParam.batchIter = 0;
    self->ctx.loadAL1Flag = true;
    self->ctx.loadBL1Flag = true;
    self->ctx.loadAL0Flag = true;
    self->ctx.loadBL0Flag = true;
    self->ctx.isFirstIterate = false;

    CalcWoDirectionVar<Intf>(self, iterParam);
    CalcHoDirectionVar<Intf>(self, iterParam);
    CalcCoDirectionVar<Intf>(self, iterParam);
}

template <class Intf>
__aicore__ inline bool IterateHWL1(Intf* self, IterateParam& iterParam)
{
    if constexpr (Intf::hasWL1IterFlag) {
        iterParam.woAL1Iter++;
        self->ctx.loadAL1Flag = true;
        if (iterParam.woAL1Iter == iterParam.ddr2l1LoopW) {
            iterParam.woAL1Iter = 0;
            CalcWoDirectionVar<Intf>(self, iterParam);
        } else {
            CalcWoDirectionVar<Intf>(self, iterParam);
            if (self->iterateMFirstRT && !Intf::hasNL0IterFlag) {
                self->ctx.loadBL0Flag = false;
            }
            return true;
        }
    }

    if constexpr (Intf::hasHL1IterFlag) {
        iterParam.hoAL1Iter++;
        if (iterParam.hoAL1Iter == iterParam.ddr2l1LoopH) {
            iterParam.hoAL1Iter = 0;
            CalcHoDirectionVar<Intf>(self, iterParam);
        } else {
            CalcHoDirectionVar<Intf>(self, iterParam);
            if (self->iterateMFirstRT && !Intf::hasNL0IterFlag) {
                self->ctx.loadBL0Flag = false;
            }
            return true;
        }
    }

    return false;
}

template <class Intf>
__aicore__ inline bool IterateL0MFirstHWMode(Intf* self, IterateParam& iterParam)
{
    if constexpr (Intf::hasWL0IterFlag) {
        iterParam.woL0Iter++;
        if (iterParam.woL0Iter == iterParam.l12l0LoopW) {
            iterParam.woL0Iter = 0;
        } else {
            self->ctx.loadBL0Flag = false;
            if ASCEND_IS_AIC_CONV {
                if constexpr (Intf::kl0FullLoadFlag) {
                    self->ctx.kL0FullLoadAl0PingPongFlag = ConvFunc::CheckReduceOneKNotSupportDBCase<Intf>(self) ?
                                                               0 :
                                                               self->ctx.kL0FullLoadAl0PingPongFlag;
                }
            }
            return true;
        }
    }

    if constexpr (Intf::hasHL0IterFlag) {
        iterParam.hoL0Iter++;
        if (iterParam.hoL0Iter == iterParam.l12l0LoopH) {
            iterParam.hoL0Iter = 0;
        } else {
            self->ctx.loadBL0Flag = false;
            if ASCEND_IS_AIC_CONV {
                if constexpr (Intf::kl0FullLoadFlag) {
                    self->ctx.kL0FullLoadAl0PingPongFlag = ConvFunc::CheckReduceOneKNotSupportDBCase<Intf>(self) ?
                                                               0 :
                                                               self->ctx.kL0FullLoadAl0PingPongFlag;
                }
            }
            return true;
        }
    }

    if ASCEND_IS_AIC_CONV {
        if constexpr (Intf::kl0FullLoadFlag) {
            self->ctx.kL0FullLoadAl0PingPongFlag = 0;
        }
    }

    if constexpr (Intf::hasNL0IterFlag) {
        iterParam.nL0Iter++;
        self->ctx.loadBL0Flag = true;
        if (iterParam.nL0Iter == iterParam.l12l0LoopN) {
            iterParam.nL0Iter = 0;
        } else {
            return true;
        }
    }
    return false;
}

template <class Intf>
__aicore__ inline bool IterateMFirstHWMode(Intf* self, IterateParam& iterParam)
{
    if (IterateL0MFirstHWMode<Intf>(self, iterParam)) {
        return true;
    }

    if ASCEND_IS_AIC_CONV {
        if (self->ctx.kAL1fullload) {
            self->ctx.queueAL1.FreeTensor(self->ctx.al1);
        }
    }

    if (IterateHWL1<Intf>(self, iterParam)) {
        return true;
    }

    iterParam.batchIter++;
    if (iterParam.batchIter == iterParam.ddr2l1LoopBatch) {
        iterParam.batchIter = 0;
    } else {
        self->ctx.loadAL1Flag = true;
        self->ctx.loadBL1Flag = false;
        return true;
    }

    if ASCEND_IS_AIC_CONV {
        if constexpr (Intf::convToL1Mode != static_cast<int8_t>(ConvToL1::OPEN_WEIGHT_BY_PASS)) {
            if (self->ctx.kBL1fullload) {
                self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
            }
        }
    }

    if constexpr (Intf::hasNL1IterFlag) {
        iterParam.nBL1Iter++;
        CalcCoDirectionVar<Intf>(self, iterParam);
        self->ctx.loadBL1Flag = true;
        if (likely(iterParam.nBL1Iter != iterParam.ddr2l1LoopN)) {
            return true;
        }
    }
    return false;
}

template <class Intf>
__aicore__ inline bool IterateL0NFirstHWMode(Intf* self, IterateParam& iterParam)
{
    if constexpr (Intf::hasNL0IterFlag) {
        iterParam.nL0Iter++;
        if (iterParam.nL0Iter == iterParam.l12l0LoopN) {
            iterParam.nL0Iter = 0;
        } else {
            self->ctx.loadAL0Flag = false;
            if ASCEND_IS_AIC_CONV {
                if constexpr (Intf::kl0FullLoadFlag) {
                    self->ctx.kL0FullLoadBl0PingPongFlag = ConvFunc::CheckReduceOneKNotSupportDBCase<Intf>(self) ?
                                                               0 :
                                                               self->ctx.kL0FullLoadBl0PingPongFlag;
                }
            }
            return true;
        }
    }
    if ASCEND_IS_AIC_CONV {
        if constexpr (Intf::kl0FullLoadFlag) {
            self->ctx.kL0FullLoadBl0PingPongFlag = 0;
        }
    }
    if constexpr (Intf::hasWL0IterFlag) {
        iterParam.woL0Iter++;
        self->ctx.loadAL0Flag = true;
        if (iterParam.woL0Iter == iterParam.l12l0LoopW) {
            iterParam.woL0Iter = 0;
        } else {
            return true;
        }
    }
    if constexpr (Intf::hasHL0IterFlag) {
        iterParam.hoL0Iter++;
        self->ctx.loadAL0Flag = true;
        if (iterParam.hoL0Iter == iterParam.l12l0LoopH) {
            iterParam.hoL0Iter = 0;
        } else {
            return true;
        }
    }

    return false;
}

template <class Intf>
__aicore__ inline bool IterateNFirstHWMode(Intf* self, IterateParam& iterParam)
{
    if (IterateL0NFirstHWMode<Intf>(self, iterParam)) {
        return true;
    }

    if ASCEND_IS_AIC_CONV {
        if constexpr (Intf::convToL1Mode != static_cast<int8_t>(ConvToL1::OPEN_WEIGHT_BY_PASS)) {
            if (self->ctx.kBL1fullload) {
                self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
            }
        }
    }

    if constexpr (Intf::hasNL1IterFlag) {
        iterParam.nBL1Iter++;
        self->ctx.loadBL1Flag = true;
        if (iterParam.nBL1Iter == iterParam.ddr2l1LoopN) {
            iterParam.nBL1Iter = 0;
            CalcCoDirectionVar<Intf>(self, iterParam);
        } else {
            CalcCoDirectionVar<Intf>(self, iterParam);
            return true;
        }
    }

    if ASCEND_IS_AIC_CONV {
        if (self->ctx.kAL1fullload) {
            self->ctx.queueAL1.FreeTensor(self->ctx.al1);
        }
    }

    if (IterateHWL1<Intf>(self, iterParam)) {
        return true;
    }

    iterParam.batchIter++;
    if (likely(iterParam.batchIter != iterParam.ddr2l1LoopBatch)) {
        self->ctx.loadAL1Flag = true;
        self->ctx.loadBL1Flag = true;
        return true;
    }
    return false;
}

#endif // CONV_ITERATE_BASE_IMPL_5162_H
