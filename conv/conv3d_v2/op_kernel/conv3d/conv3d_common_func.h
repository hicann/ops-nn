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
 * \file conv3d_common_func.h
 * \brief
 */

#ifndef CONV3D_COMMON_FUNC_H
#define CONV3D_COMMON_FUNC_H

#include "conv3d_common_init_func.h"
#include "conv3d_common_set_func.h"
#include "quant_conv3d_common_func.h"

namespace Conv3dFunc {

CONV_DECLARE_REG_IMPL(Init);
CONV_DECLARE_REG_IMPL(SetOrgFmapShape);
CONV_DECLARE_REG_IMPL(SetOrgWeightShape);
CONV_DECLARE_REG_IMPL(SetOrgOutputShape);
CONV_DECLARE_REG_IMPL(SetSingleFmapShape);
CONV_DECLARE_REG_IMPL(SetSingleOutputShape);
CONV_DECLARE_REG_IMPL(SetFmapStartPosition);
CONV_DECLARE_REG_IMPL(SetGroupOptInfo);
CONV_DECLARE_REG_IMPL(Iterate);
CONV_DECLARE_REG_IMPL(GetTensorC);
CONV_DECLARE_REG_IMPL(IterateAll);

using TypeFalse = const struct {
    __uint128_t _[1024];
};

template <class Intf, uint32_t ImplType>
struct Iterate {
    template <bool sync = true>
    static __aicore__ inline bool call(Intf *self, bool enPartialSum = false)
    {
        return IterateImpl(self, enPartialSum);
    }

    static __aicore__ inline uint64_t CalcL0CurrentN(Intf *self)
    {
        uint64_t n = (self->ctx.nBL1Iter == self->ctx.maxNBL1Iter && self->ctx.nBL0Iter == self->ctx.maxNL0Iter)
                         ? self->ctx.nL0Tail
                         : self->ctx.conv3dTiling->nL0;
        return n;
    }

    static __aicore__ inline uint64_t CalcL0CurrentM(Intf *self)
    {
        uint64_t m = (self->ctx.mAL1Iter == self->ctx.maxMAL1Iter && self->ctx.mAL0Iter == self->ctx.maxML0Iter)
                         ? self->ctx.mAL0Tail
                         : self->ctx.conv3dTiling->mL0;
        return m;
    }

    static __aicore__ void inline FirstIterateImpl(Intf *self)
    {
        // 先更新index再load，就需要加第一次处理。
        self->ctx.nBL0Iter = 0;
        self->ctx.mAL0Iter = 0;
        self->ctx.mAL1Iter = 0;
        self->ctx.nBL1Iter = 0;
        self->ctx.dOutIter = 0;
        self->ctx.loadAL1Flag = true;
        self->ctx.loadBL1Flag = !Intf::bl1bypass;
        self->ctx.loadAL0Flag = true;
        self->ctx.loadBL0Flag = true;
        self->ctx.isFirstIterate = false;
        if constexpr (Intf::formatType != conv::ConvFormat::NCDHW) {
            self->ctx.loadAl1Ins.SetLoadData3DParams();
            if (self->ctx.conv3dTiling->mL0 % self->ctx.orgWo == 0) {
                self->ctx.mL0IsDivisibleByWo = true;
            }
        }
        if constexpr (Intf::outputOrder) {
            self->ctx.hoL1Iter = 0;
        }
    }

    static __aicore__ void inline UpdateIterators(Intf *self)
    {
        if constexpr (Intf::outputOrder) {
            if (self->ctx.mAL1Iter == self->ctx.ddr2l1LoopM) {
                self->ctx.mAL1Iter = 0;
                self->ctx.hoL1Iter++;
            }
            if (self->ctx.hoL1Iter == self->ctx.ddr2l1LoopHo) {
                self->ctx.hoL1Iter = 0;
                self->ctx.dOutIter++;
            }
        } else {
            if (self->ctx.mAL1Iter == self->ctx.ddr2l1LoopM) {
                self->ctx.mAL1Iter = 0;
                self->ctx.dOutIter++;
            }
        }
    }

    static __aicore__ bool inline IterateMFirst(Intf *self)
    {
        // ReorderN: 先往M轴方向偏移再往N轴方向偏移。Fmap复用Weight。
        //    M
        //    |
        //    |
        //    |----------N-------->
        // ==================L0========================
        self->ctx.mAL0Iter++;
        if (self->ctx.mAL0Iter == self->ctx.l12l0LoopM) {
            self->ctx.mAL0Iter = 0;
            self->ctx.nBL0Iter++;
        }
        if (self->ctx.nBL0Iter == self->ctx.l12l0LoopN) {
            self->ctx.nBL0Iter = 0;
            self->ctx.mAL1Iter++;
            self->ctx.loadAL1Flag = true;
        }
        UpdateIterators(self);
        if (self->ctx.dOutIter == self->ctx.ddr2l1LoopD) {
            self->ctx.dOutIter = 0;
            self->ctx.nBL1Iter++;
            self->ctx.loadBL1Flag = true;
        }
        if (self->ctx.nBL1Iter == self->ctx.ddr2l1LoopN) {
            return false;
        }
        return true;
    }

    static __aicore__ bool inline IterateNFirst(Intf *self)
    {
        // ReorderM: 先往N轴方向偏移再往M轴方向偏移。Weight复用Fmap。
        //    ----------N-------->
        //                       |
        //                       |
        //                       M
        //                       |
        //                       |
        // ==================L0========================
        self->ctx.nBL0Iter++;
        if (self->ctx.nBL0Iter == self->ctx.l12l0LoopN) {
            self->ctx.nBL0Iter = 0;
            self->ctx.mAL0Iter++;
        }
        if (self->ctx.mAL0Iter == self->ctx.l12l0LoopM) {
            self->ctx.mAL0Iter = 0;
            self->ctx.nBL1Iter++;
            self->ctx.loadBL1Flag = true;
        }
        if (self->ctx.nBL1Iter == self->ctx.ddr2l1LoopN) {
            self->ctx.nBL1Iter = 0;
            self->ctx.mAL1Iter++;
            self->ctx.loadAL1Flag = true;
        }
        UpdateIterators(self);
        if (self->ctx.dOutIter == self->ctx.ddr2l1LoopD) {
            return false;
        }
        return true;
    }

    static __aicore__ void inline ReduceKNoPingPongBL1ByPass(Intf *self) {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(event_t::EVENT_ID0);
        self->ctx.kAL0Iter = self->ctx.kIter % self->ctx.multiKAL1;
        self->ctx.loadAL0Ins.LoadAL0();
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(event_t::EVENT_ID0);

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(event_t::EVENT_ID0);
        self->ctx.kBL0Iter = self->ctx.kIter % self->ctx.multiKBL1;
        self->ctx.loadBL0Ins.LoadBL0();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(event_t::EVENT_ID1);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(event_t::EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(event_t::EVENT_ID1);

        self->ctx.madIns.Mad();
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(event_t::EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(event_t::EVENT_ID0);
    }

    static __aicore__ void inline ReduceKNoPingPongBL1NoByPass(Intf *self) {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(event_t::EVENT_ID0);
        self->ctx.kAL0Iter = self->ctx.kIter % self->ctx.multiKAL1;
        self->ctx.loadAL0Ins.LoadAL0();

        self->ctx.kBL0Iter = self->ctx.kIter % self->ctx.multiKBL1;
        self->ctx.loadBL0Ins.LoadBL0();
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(event_t::EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(event_t::EVENT_ID0);

        self->ctx.madIns.Mad();
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(event_t::EVENT_ID0);
    }

    static __aicore__ void inline ReduceKL0APingPongBL1ByPass(Intf *self, const uint16_t& l0aFlag) {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0aFlag);
        self->ctx.al0 = l0aFlag ? self->ctx.al0Pong : self->ctx.al0Ping;
        self->ctx.kAL0Iter = self->ctx.kIter % self->ctx.multiKAL1;
        self->ctx.loadAL0Ins.LoadAL0();
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0aFlag);

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(event_t::EVENT_ID2);
        if (self->ctx.loadBL0Flag) {
            self->ctx.kBL0Iter = self->ctx.kIter % self->ctx.multiKBL1;
            self->ctx.loadBL0Ins.LoadBL0();
            AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(event_t::EVENT_ID2);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0aFlag);
        if (self->ctx.loadBL0Flag) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(event_t::EVENT_ID2);
        }
        self->ctx.madIns.Mad();
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0aFlag);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(event_t::EVENT_ID2);
    }

    static __aicore__ void inline ReduceKL0APingPongBL1NoByPass(Intf *self, const uint16_t& l0aFlag) {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0aFlag);
        self->ctx.al0 = l0aFlag ? self->ctx.al0Pong : self->ctx.al0Ping;
        self->ctx.kAL0Iter = self->ctx.kIter % self->ctx.multiKAL1;
        self->ctx.loadAL0Ins.LoadAL0();

        if (self->ctx.loadBL0Flag) {
            self->ctx.kBL0Iter = self->ctx.kIter % self->ctx.multiKBL1;
            self->ctx.loadBL0Ins.LoadBL0();
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0aFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0aFlag);
        self->ctx.madIns.Mad();
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0aFlag);
    }

    static __aicore__ void inline ReduceKL0BPingPongBL1ByPass(Intf *self, const uint16_t& l0bFlag) {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(event_t::EVENT_ID2);
        if (self->ctx.loadAL0Flag) {
            self->ctx.kAL0Iter = self->ctx.kIter % self->ctx.multiKAL1;
            self->ctx.loadAL0Ins.LoadAL0();
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(event_t::EVENT_ID2);
        }

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(l0bFlag);
        self->ctx.bl0 = l0bFlag ? self->ctx.bl0Pong : self->ctx.bl0Ping;
        self->ctx.kBL0Iter = self->ctx.kIter % self->ctx.multiKBL1;
        self->ctx.loadBL0Ins.LoadBL0();
        AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(l0bFlag);

        if (self->ctx.loadAL0Flag) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(event_t::EVENT_ID2);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(l0bFlag);

        self->ctx.madIns.Mad();
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(event_t::EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(l0bFlag);
    }

    static __aicore__ void inline ReduceKL0BPingPongBL1NoByPass(Intf *self, const uint16_t& l0bFlag) {
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(event_t::EVENT_ID2);
        if (self->ctx.loadAL0Flag) {
            self->ctx.kAL0Iter = self->ctx.kIter % self->ctx.multiKAL1;

            self->ctx.loadAL0Ins.LoadAL0();
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(event_t::EVENT_ID2);
        }

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0bFlag);
        self->ctx.bl0 = l0bFlag ? self->ctx.bl0Pong : self->ctx.bl0Ping;
        self->ctx.kBL0Iter = self->ctx.kIter % self->ctx.multiKBL1;
        self->ctx.loadBL0Ins.LoadBL0();
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0bFlag);

        if (self->ctx.loadAL0Flag) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(event_t::EVENT_ID2);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0bFlag);

        self->ctx.madIns.Mad();
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(event_t::EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0bFlag);
    }

    static __aicore__ void inline ReduceKL0AL0BPingPong(Intf *self, const uint16_t& l0abFlag) {
        if (l0abFlag) {
            self->ctx.al0 = self->ctx.al0Pong;
            self->ctx.bl0 = self->ctx.bl0Pong;
        } else {
            self->ctx.al0 = self->ctx.al0Ping;
            self->ctx.bl0 = self->ctx.bl0Ping;
        }

        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0abFlag);

        self->ctx.kAL0Iter = self->ctx.kIter % self->ctx.multiKAL1;
        self->ctx.loadAL0Ins.LoadAL0();

        if constexpr (Intf::bl1bypass) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(l0abFlag);
        }
        self->ctx.kBL0Iter = self->ctx.kIter % self->ctx.multiKBL1;
        self->ctx.loadBL0Ins.LoadBL0();
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0abFlag);
        if constexpr (Intf::bl1bypass) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_M>(l0abFlag);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0abFlag);
        if constexpr (Intf::bl1bypass) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_M>(l0abFlag);
        }

        AscendC::PipeBarrier<PIPE_M>();
        self->ctx.madIns.Mad();
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0abFlag);
        if constexpr (Intf::bl1bypass) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(l0abFlag);
        }
    }

    static __aicore__ void inline LoadAL1Process(Intf *self, uint64_t kAL1Iter)
    {
        self->ctx.al1 = self->ctx.queueAL1.template AllocTensor<typename Intf::FmapT>();
        self->ctx.kAL1Iter = kAL1Iter;
        self->ctx.loadAl1Ins.LoadAL1();
        self->ctx.queueAL1.EnQue(self->ctx.al1);
        self->ctx.al1 = self->ctx.queueAL1.template DeQue<typename Intf::FmapT>();
        self->ctx.loadAL1Flag = false;  // LoopK中只有K方向可能重新载入。
        self->ctx.freeAL1TensorFlag = true;
    }

    static __aicore__ void inline LoadBL1Process(Intf *self, uint64_t kBL1Iter)
    {
        self->ctx.bl1 = self->ctx.queueBL1.template AllocTensor<typename Intf::WeightT>();
        self->ctx.kBL1Iter = kBL1Iter;
        self->ctx.loadBL1Ins.LoadBL1();
        self->ctx.queueBL1.EnQue(self->ctx.bl1);
        self->ctx.bl1 = self->ctx.queueBL1.template DeQue<typename Intf::WeightT>();
        self->ctx.loadBL1Flag = false;  // LoopK中只有K方向可能重新载入。
        self->ctx.freeBL1TensorFlag = true;
    }

    static __aicore__ void inline LoadAL1PreloadProcess(Intf *self, uint64_t kAL1Iter)
    {
        self->ctx.al1 = self->ctx.queueAL1.template AllocTensor<typename Intf::FmapT>();
        self->ctx.kAL1Iter = kAL1Iter;
        self->ctx.loadAl1Ins.LoadAL1();
        self->ctx.queueAL1.EnQue(self->ctx.al1);
        self->ctx.loadAL1Flag = true;
        self->ctx.freeAL1TensorFlag = false;
    }

    static __aicore__ void inline LoadBL1PreloadProcess(Intf *self, uint64_t kBL1Iter)
    {
        self->ctx.bl1 = self->ctx.queueBL1.template AllocTensor<typename Intf::WeightT>();
        self->ctx.kBL1Iter = kBL1Iter;
        self->ctx.loadBL1Ins.LoadBL1();
        self->ctx.queueBL1.EnQue(self->ctx.bl1);
        self->ctx.loadBL1Flag = true;
        self->ctx.freeBL1TensorFlag = false;
    }

    // K方向迭代首次(iter==0)加载L0A, L0B
    static __aicore__ void inline ReduceKFirstIterLoadL0(Intf *self)
    {
        if constexpr(Intf::l0pingpong == static_cast<int8_t>(conv3d::ConvL0PingPong::ALL_CLOSE)) {
            // for L0PingPong::ALL_CLOSE, BL1ByPass is always ON
            self->ctx.al0 = self->ctx.al0Ping;
            self->ctx.bl0 = self->ctx.bl0Ping;
            if constexpr (Intf::bl1bypass) {
                AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(event_t::EVENT_ID0);
                ReduceKNoPingPongBL1ByPass(self);
            } else {
                ReduceKNoPingPongBL1NoByPass(self);
            }
        } else if constexpr(Intf::l0pingpong == static_cast<int8_t>(conv3d::ConvL0PingPong::L0A_OPEN)) {
            // for L0PingPong::L0A_OPEN, BL1ByPass is always ON
            self->ctx.bl0 = self->ctx.bl0Ping;
            if constexpr (Intf::bl1bypass) {
                AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(event_t::EVENT_ID2);
                ReduceKL0APingPongBL1ByPass(self, event_t::EVENT_ID0);
            } else {
                ReduceKL0APingPongBL1NoByPass(self, event_t::EVENT_ID0);
            }
        } else if constexpr(Intf::l0pingpong == static_cast<int8_t>(conv3d::ConvL0PingPong::L0B_OPEN)) {
            self->ctx.al0 = self->ctx.al0Ping;
            if constexpr (Intf::bl1bypass) {
                AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(event_t::EVENT_ID0);
                if (self->ctx.ddr2l1LoopD > 1) {
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(event_t::EVENT_ID1);
                }
                ReduceKL0BPingPongBL1ByPass(self, event_t::EVENT_ID0);
            } else {
                ReduceKL0BPingPongBL1NoByPass(self, event_t::EVENT_ID0);
            }
        } else if constexpr(Intf::l0pingpong == static_cast<int8_t>(conv3d::ConvL0PingPong::ALL_OPEN)) {
            if constexpr (Intf::bl1bypass) {
                AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(event_t::EVENT_ID0);
                AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(event_t::EVENT_ID1);
                if (self->ctx.ddr2l1LoopD > 1) {
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE2>(event_t::EVENT_ID4);
                }
            }
            ReduceKL0AL0BPingPong(self, event_t::EVENT_ID0);
        }
    }

    // K方向迭代(iter>0)加载L0A, L0B
    static __aicore__ void inline ReduceKIterLoadL0(Intf *self, const uint16_t& isOdd)
    {
        if constexpr (Intf::l0pingpong == static_cast<int8_t>(conv3d::ConvL0PingPong::ALL_CLOSE)) {
            if constexpr (Intf::bl1bypass) {
                ReduceKNoPingPongBL1ByPass(self);
            } else {
                ReduceKNoPingPongBL1NoByPass(self);
            }
        } else if constexpr(Intf::l0pingpong == static_cast<int8_t>(conv3d::ConvL0PingPong::L0A_OPEN)) {
            if constexpr (Intf::bl1bypass) {
                ReduceKL0APingPongBL1ByPass(self, isOdd);
            } else {
                ReduceKL0APingPongBL1NoByPass(self, isOdd);
            }
        } else if constexpr(Intf::l0pingpong == static_cast<int8_t>(conv3d::ConvL0PingPong::L0B_OPEN)) {
            if constexpr (Intf::bl1bypass) {
                ReduceKL0BPingPongBL1ByPass(self, isOdd);
            } else {
                ReduceKL0BPingPongBL1NoByPass(self, isOdd);
            }
        } else if constexpr(Intf::l0pingpong == static_cast<int8_t>(conv3d::ConvL0PingPong::ALL_OPEN)) {
            ReduceKL0AL0BPingPong(self, isOdd);
        }
    }

    static __aicore__ void inline ReduceKIterLoadL1(Intf *self)
    {
        if (self->ctx.loadAL1Flag || (!self->ctx.kAL1fullload && self->ctx.kIter % self->ctx.multiKAL1 == 0)) {
            self->ctx.queueAL1.FreeTensor(self->ctx.al1);
            self->ctx.freeAL1TensorFlag = false;
            LoadAL1Process(self, self->ctx.kIter / self->ctx.multiKAL1);
        }
        if constexpr (!Intf::bl1bypass) {
            if (self->ctx.loadBL1Flag || (!self->ctx.kBL1fullload && self->ctx.kIter % self->ctx.multiKBL1 == 0)) {
                self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
                self->ctx.freeBL1TensorFlag = false;
                LoadBL1Process(self, self->ctx.kIter / self->ctx.multiKBL1);
            }
        }
    }

    // K方向迭代的后处理, 当前只对bl1bypass需要添加wait_flag
    static __aicore__ void inline ReduceKPostProcessLoadL0(Intf *self)
    {
        if constexpr((Intf::l0pingpong == static_cast<int8_t>(conv3d::ConvL0PingPong::ALL_CLOSE)) && Intf::bl1bypass) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(event_t::EVENT_ID0);
        } else if constexpr((Intf::l0pingpong == static_cast<int8_t>(conv3d::ConvL0PingPong::L0A_OPEN)) && Intf::bl1bypass) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(event_t::EVENT_ID2);
        } else if constexpr((Intf::l0pingpong == static_cast<int8_t>(conv3d::ConvL0PingPong::L0B_OPEN)) && Intf::bl1bypass) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(event_t::EVENT_ID0);
            if (self->ctx.ddr2l1LoopD > 1) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(event_t::EVENT_ID1);
            }
        } else if constexpr((Intf::l0pingpong == static_cast<int8_t>(conv3d::ConvL0PingPong::ALL_OPEN)) && Intf::bl1bypass) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(event_t::EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(event_t::EVENT_ID1);
            if (self->ctx.ddr2l1LoopD > 1) {
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE2>(event_t::EVENT_ID4);
            }
        }
    }

    static __aicore__ void inline ReduceK(Intf *self)
    {
        ASC_OP_LOGD("no preload in ReduceK: loadAl1Flag: %d, kAL1fullload: %d, freeAL1TensorFlag: %d\n",
                    self->ctx.loadAL1Flag, self->ctx.kAL1fullload, self->ctx.freeAL1TensorFlag);

        if (self->ctx.loadAL1Flag || !(self->ctx.kAL1fullload)) {
            if (self->ctx.freeAL1TensorFlag) {
                self->ctx.queueAL1.FreeTensor(self->ctx.al1);
                self->ctx.freeAL1TensorFlag = false;
            }
            LoadAL1Process(self, 0);
        }
        if constexpr (!Intf::bl1bypass) {
            if (self->ctx.loadBL1Flag || !(self->ctx.kBL1fullload)) {
                if (self->ctx.freeBL1TensorFlag) {
                    self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
                    self->ctx.freeBL1TensorFlag = false;
                }
                LoadBL1Process(self, 0);
            }
        }
        ReduceKFirstIterLoadL0(self);

        self->ctx.kIter = 1;
        uint16_t isOdd = 1;
        while (self->ctx.kIter < self->ctx.ddr2l0LoopK) {
            ReduceKIterLoadL1(self);
            ReduceKIterLoadL0(self, isOdd);
            self->ctx.kIter++;
            isOdd = self->ctx.kIter & 0x1;
        }

        ReduceKPostProcessLoadL0(self);
    }

    static __aicore__ void inline ReduceKPreloadDbAllLoadL1(Intf *self, const uint64_t& maxKAL1PreloadIter, const uint64_t& maxKBL1PreloadIter)
    {
        if (self->ctx.kIter == maxKAL1PreloadIter) {
            self->ctx.queueAL1.FreeTensor(self->ctx.al1);
            self->ctx.al1 = self->ctx.queueAL1.template DeQue<typename Intf::FmapT>();
        } else if (self->ctx.kIter < maxKAL1PreloadIter &&
            (self->ctx.loadAL1Flag || (!self->ctx.kAL1fullload && self->ctx.kIter % self->ctx.multiKAL1 == 0))) {
            self->ctx.queueAL1.FreeTensor(self->ctx.al1);
            LoadAL1Process(self, (self->ctx.kIter / self->ctx.multiKAL1) + 1);
        }

        if (self->ctx.kIter == maxKBL1PreloadIter) {
            self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
            self->ctx.bl1 = self->ctx.queueBL1.template DeQue<typename Intf::WeightT>();
        } else if (self->ctx.kIter < maxKBL1PreloadIter &&
            (self->ctx.loadBL1Flag || (!self->ctx.kBL1fullload && self->ctx.kIter % self->ctx.multiKBL1 == 0))) {
            self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
            LoadBL1Process(self, (self->ctx.kIter / self->ctx.multiKBL1) + 1);
        }
    }

    static __aicore__ void inline ReduceKPreloadDbAll(Intf *self)
    {
        ASC_OP_LOGD("AL1 and BL1 db case, preload reduce k\n");

        if (self->ctx.loadAL1Flag || !(self->ctx.kAL1fullload)) {
            if (self->ctx.freeAL1TensorFlag) {
                self->ctx.queueAL1.FreeTensor(self->ctx.al1);
            }
            LoadAL1PreloadProcess(self, 0);
        }

        if (self->ctx.loadBL1Flag || !(self->ctx.kBL1fullload)) {
            if (self->ctx.freeBL1TensorFlag) {
                self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
            }
            LoadBL1PreloadProcess(self, 0);
        }

        LoadAL1Process(self, 1);
        LoadBL1Process(self, 1);
        ReduceKFirstIterLoadL0(self);

        self->ctx.kIter = 1;
        uint16_t isOdd = 1;
        uint64_t maxKAL1PreloadIter = self->ctx.ddr2l0LoopK - self->ctx.multiKAL1;
        uint64_t maxKBL1PreloadIter = self->ctx.ddr2l0LoopK - self->ctx.multiKBL1;
        while (self->ctx.kIter < self->ctx.ddr2l0LoopK) {
            ReduceKPreloadDbAllLoadL1(self, maxKAL1PreloadIter, maxKBL1PreloadIter);
            ReduceKIterLoadL0(self, isOdd);
            self->ctx.kIter++;
            isOdd = self->ctx.kIter & 0x1;
        }
    }

    static __aicore__ void inline ReduceKPreloadDbFmapLoadL1(Intf *self,  const uint64_t& maxKAL1PreloadIter)
    {
        if (self->ctx.kIter == maxKAL1PreloadIter) {
            self->ctx.queueAL1.FreeTensor(self->ctx.al1);
            self->ctx.al1 = self->ctx.queueAL1.template DeQue<typename Intf::FmapT>();
        } else if (self->ctx.kIter < maxKAL1PreloadIter && self->ctx.kIter % self->ctx.multiKAL1 == 0) {
            self->ctx.queueAL1.FreeTensor(self->ctx.al1);
            LoadAL1Process(self, (self->ctx.kIter / self->ctx.multiKAL1) + 1);
        }

        if constexpr (!Intf::bl1bypass) {
            if (self->ctx.loadBL1Flag || (!self->ctx.kBL1fullload && self->ctx.kIter % self->ctx.multiKBL1 == 0)) {
                self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
                LoadBL1Process(self, self->ctx.kIter / self->ctx.multiKBL1);
            }
        }
    }

    static __aicore__ void inline ReduceKPreloadDbFmap(Intf *self)
    {
        ASC_OP_LOGD("AL1 db case, preload reduce k\n");

        if (self->ctx.freeAL1TensorFlag) {
            self->ctx.queueAL1.FreeTensor(self->ctx.al1);
        }
        LoadAL1PreloadProcess(self, 0);
        LoadAL1Process(self, 1);

        if constexpr (!Intf::bl1bypass) {
            if (self->ctx.loadBL1Flag || !(self->ctx.kBL1fullload)) {
                if (self->ctx.freeBL1TensorFlag) {
                    self->ctx.queueBL1.FreeTensor(self->ctx.bl1);
                }
                LoadBL1Process(self, 0);
            }
        }

        ReduceKFirstIterLoadL0(self);

        self->ctx.kIter = 1;
        uint16_t isOdd = 1;
        uint64_t maxKAL1PreloadIter = self->ctx.ddr2l0LoopK - self->ctx.multiKAL1;
        while (self->ctx.kIter < self->ctx.ddr2l0LoopK) {
            ReduceKPreloadDbFmapLoadL1(self, maxKAL1PreloadIter);
            ReduceKIterLoadL0(self, isOdd);
            self->ctx.kIter++;
            isOdd = self->ctx.kIter & 0x1;
        }

        ReduceKPostProcessLoadL0(self);
    }

    static __aicore__ void inline CalcBias(Intf *self)
    {
        if constexpr(Intf::l0pingpong == static_cast<int8_t>(conv3d::ConvL0PingPong::L0B_OPEN)) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(event_t::EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(event_t::EVENT_ID2);
            self->ctx.loadBiasBTIns.LoadBiasL0WithBroadcast();
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(event_t::EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(event_t::EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(event_t::EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(event_t::EVENT_ID2);
            self->ctx.madIns.MadBias();
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(event_t::EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(event_t::EVENT_ID2);
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(event_t::EVENT_ID0);
            self->ctx.loadBiasBTIns.LoadBiasL0WithBroadcast();
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(event_t::EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(event_t::EVENT_ID0);
            self->ctx.madIns.MadBias();
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(event_t::EVENT_ID0);
        }
    }

    static __aicore__ void inline InitBiasWithPointWise(Intf *self, uint64_t m, uint64_t n)
    {
        if (self->ctx.enableBias) {
            self->ctx.loadBiasL1Ins.SetN(AlignB(n, BLOCK_L0_M));
            self->ctx.loadBiasBTIns.SetMN(AlignB(n, BLOCK_L0_M), AlignB(m, BLOCK_L0_N));
            if (!self->ctx.biasFullLoadFlag) {
                self->ctx.biasL1 = self->ctx.queueBiasL1.template AllocTensor<typename Intf::BiasT>();
                self->ctx.loadBiasL1Ins.LoadChannelWiseL1(self->ctx.biasL1, self->ctx.biasgm);
                self->ctx.queueBiasL1.EnQue(self->ctx.biasL1);
                self->ctx.biasL1 = self->ctx.queueBiasL1.template DeQue<typename Intf::BiasT>();
            }
            CalcBias(self);
        }
    }

    static __aicore__ void inline InitBiasWithNormal(Intf *self, uint64_t m, uint64_t n)
    {
        if (self->ctx.enableBias) {
            self->ctx.loadBiasL1Ins.SetN(n);
            self->ctx.loadBiasBTIns.SetN(AlignB(n, BLOCK_L0_N));
            if (!self->ctx.biasFullLoadFlag) {
                self->ctx.biasL1 = self->ctx.queueBiasL1.template AllocTensor<typename Intf::BiasT>();
                self->ctx.loadBiasL1Ins.LoadChannelWiseL1(self->ctx.biasL1, self->ctx.biasgm);
                self->ctx.queueBiasL1.EnQue(self->ctx.biasL1);
                self->ctx.biasL1 = self->ctx.queueBiasL1.template DeQue<typename Intf::BiasT>();
            }
            self->ctx.biasBT = self->ctx.queueBiasBT.template AllocTensor<typename Intf::L0cT>();
            self->ctx.loadBiasBTIns.LoadBiasBt();
            self->ctx.queueBiasBT.EnQue(self->ctx.biasBT);
            self->ctx.biasBT = self->ctx.queueBiasBT.template DeQue<typename Intf::L0cT>();
        }
    }

    static __aicore__ void inline IterateK(Intf *self)
    {
        // in each iterate k, cal current m,n value
        uint64_t n = CalcL0CurrentN(self);
        uint64_t m = CalcL0CurrentM(self);
        if ASCEND_IS_AIC {
            self->ctx.cl0 = self->ctx.queueCL0.template AllocTensor<typename Intf::L0cT>();
        }

        self->ctx.loadAL0Ins.SetM(AlignB(m, BLOCK_L0_N));
        self->ctx.loadBL0Ins.SetN(AlignB(n, BLOCK_L0_M));
        if constexpr (Intf::formatType == conv::ConvFormat::NCDHW) {
            self->ctx.madIns.SetMN(AlignB(n, BLOCK_L0_M), AlignB(m, BLOCK_L0_N));
            self->ctx.copyOutIns.SetMN(n, m);
            if ASCEND_IS_AIC {
                InitBiasWithPointWise(self, m, n);
            }
        } else {
            self->ctx.madIns.SetMN(AlignB(m, BLOCK_L0_M), AlignB(n, BLOCK_L0_N));
            self->ctx.copyOutIns.SetMN(m, AlignB(n, self->ctx.cout0));
            if ASCEND_IS_AIC {
                InitBiasWithNormal(self, m, n);
            }
        }
        if ASCEND_IS_AIC {
            if (self->ctx.preloadABL1DbFlag) {
                ReduceKPreloadDbAll(self);
            } else if (self->ctx.preloadAL1DbFlag) {
                ReduceKPreloadDbFmap(self);
            } else {
                ReduceK(self);
            }
            self->ctx.queueCL0.EnQue(self->ctx.cl0);
            self->ctx.cl0 = self->ctx.queueCL0.template DeQue<typename Intf::L0cT>();
            self->ctx.kIter = 0;
        }
    }

    static __aicore__ void inline UpdateL1TailLoop(Intf *self)
    {
        self->ctx.l12l0LoopM = self->ctx.mAL1Iter == self->ctx.maxMAL1Iter
                                   ? CeilDIV(self->ctx.mAL1Tail, self->ctx.conv3dTiling->mL0)
                                   : self->ctx.conv3dTiling->mAL1DivmL0;
        self->ctx.maxML0Iter = self->ctx.l12l0LoopM - 1;

        if constexpr (Intf::bl1bypass) {
            return;
        }
        self->ctx.l12l0LoopN = self->ctx.nBL1Iter == self->ctx.maxNBL1Iter
                                   ? CeilDIV(self->ctx.nBL1Tail, self->ctx.conv3dTiling->nL0)
                                   : self->ctx.conv3dTiling->nBL1DivnL0;
        self->ctx.maxNL0Iter = self->ctx.l12l0LoopN - 1;
    }

    static __aicore__ bool inline IterateImpl(Intf *self, bool enPartialSum)
    {
        if (self->ctx.isFirstIterate) {
            FirstIterateImpl(self);
        } else if (likely(self->ctx.conv3dTiling->iterateMNOrder == static_cast<int>(IterateMNOrder::ORDER_MTERFIRST))) {
            if (IterateMFirst(self) == false) {
                return false;
            }
        } else if (likely(self->ctx.conv3dTiling->iterateMNOrder == static_cast<int>(IterateMNOrder::ORDER_NTERFIRST))) {
            if (IterateNFirst(self) == false) {
                return false;
            }
        }
        UpdateL1TailLoop(self);
        IterateK(self);
        return true;
    }
};

template <class Intf, uint32_t ImplType>
struct GetTensorC {
    template <bool sync = true>
    static __aicore__ inline bool call(
        Intf *self, const GlobalTensor<typename Intf::OutputT> &output, bool enSequentialWrite = false)
    {
        if constexpr (Intf::quantType == static_cast<int8_t>(QuantType::PER_CHANNEL_NO_OFFSET)) {
            CrossCoreWaitFlag(self->ctx.V2CEvent + self->ctx.workspaceDbFlag);
            self->ctx.copyOutIns.CopyOut2Workspace(self->ctx.workspacegm);
            CrossCoreSetFlag<0x2, PIPE_FIX>(self->ctx.C2VEvent + self->ctx.workspaceDbFlag);
        } else {
            self->ctx.copyOutIns.CopyOut(output);
        }
        self->ctx.queueCL0.FreeTensor(self->ctx.cl0);
        if (self->ctx.enableBias) {
            if (!self->ctx.biasFullLoadFlag) {
                self->ctx.queueBiasL1.FreeTensor(self->ctx.biasL1);
            }
            if constexpr (Intf::formatType != conv::ConvFormat::NCDHW) {
                self->ctx.queueBiasBT.FreeTensor(self->ctx.biasBT);
            }
        }
        ASC_OP_LOGD("[GetTensorC] GetTensorC Success! \n\n");
        return false;
    }
};

template <class Intf, uint32_t ImplType>
struct IterateAll {
    template <bool sync = true>
    static __aicore__ inline bool call(
        Intf *self, const GlobalTensor<typename Intf::OutputT> &output, bool enPartialSum = false)
    {
        self->ctx.loadBiasL1Ins.SetParams(self);
        self->ctx.loadBL1Ins.SetParams(self);
        self->ctx.loadAl1Ins.SetParams(self);
        self->ctx.loadBL0Ins.SetParams(self);
        self->ctx.madIns.SetParams(self);
        self->ctx.copyOutIns.SetParams(self);
        self->ctx.loadBiasBTIns.SetParams(self);
        if constexpr (Intf::formatType == conv::ConvFormat::NCDHW) {
            self->ctx.loadAL0Ins.SetParams(self);
        } else {
            self->ctx.loadAL0Ins.SetParams(self, &self->ctx.loadAl1Ins);
        }
        if constexpr (Intf::groupConvType) {
            IterateAllWithGroups(self, output, enPartialSum);
        } else {
            IterateAllBase(self, output, enPartialSum);
        }
        return false;
    }

    static __aicore__ void inline IterateAllBase(
        Intf *self, const GlobalTensor<typename Intf::OutputT> &output, bool enPartialSum = false)
    {
        if ASCEND_IS_AIC {
            if (self->ctx.biasFullLoadFlag && self->ctx.enableBias) {
                self->ctx.biasL1 = self->ctx.queueBiasL1.template AllocTensor<typename Intf::BiasT>();
                self->ctx.loadBiasL1Ins.LoadChannelWiseL1(self->ctx.biasL1, self->ctx.biasgm);
                self->ctx.queueBiasL1.EnQue(self->ctx.biasL1);
                self->ctx.biasL1 = self->ctx.queueBiasL1.template DeQue<typename Intf::BiasT>();
            }
        }

        while (Iterate<Intf, ImplType>::call(self, enPartialSum)) {
            if ASCEND_IS_AIC {
                GetTensorC<Intf, ImplType>::call(self, output);
                if constexpr (Intf::formatType != conv::ConvFormat::NCDHW) {
                    if (self->ctx.enableBias) {
                        self->ctx.queueBiasBT.FreeAllEvent();
                    }
                }
            }

            if constexpr (Intf::quantType == static_cast<int8_t>(QuantType::PER_CHANNEL_NO_OFFSET)) {
                if ASCEND_IS_AIV {
                    VecCompute<Intf, ImplType>::call(self, output);
                }
            }
            self->ctx.workspaceDbFlag = (self->ctx.workspaceDbFlag + 1) & 0x03;
        }
        if ASCEND_IS_AIC {
            if (self->ctx.biasFullLoadFlag && self->ctx.enableBias) {
                self->ctx.queueBiasL1.FreeTensor(self->ctx.biasL1);
            }
        }
        self->ctx.isFirstIterate = true;
        self->ctx.nBL0Iter = 0;
        self->ctx.nBL1Iter = 0;
    }

    static __aicore__ void inline ReCalculationKTilingWithGroups(
        Intf *self, uint64_t &updateKAL1, uint64_t &updateKBL1, uint64_t &updateKL0)
    {
        // Update kaL1/kbL1/kL0 when singleCoreCin changes.
        uint64_t curKAL1Kd = GetCurrentKD(
            self->ctx.conv3dTiling->kAL1, AlignB(self->ctx.orgCi, self->ctx.cin0), self->ctx.kernelHxkernelW);
        uint64_t curKBL1Kd = GetCurrentKD(
            self->ctx.conv3dTiling->kBL1, AlignB(self->ctx.orgCi, self->ctx.cin0), self->ctx.kernelHxkernelW);
        uint64_t curCinxKhxKw = AlignB(self->ctx.singleCoreCin, self->ctx.cin0) * self->ctx.kernelHxkernelW;
        updateKAL1 = curCinxKhxKw > self->ctx.conv3dTiling->kAL1 ? 0 : curCinxKhxKw;
        updateKBL1 = curCinxKhxKw > self->ctx.conv3dTiling->kBL1 ? 0 : curCinxKhxKw;
        if (curKAL1Kd > 1) {
            // The kAL1/kBL1 is calculated by multiplying the new cin by the kd of the previous tiling decision.
            updateKAL1 = curKAL1Kd * curCinxKhxKw;
        }
        if (updateKAL1 == 0) {
            // To ensure that kAL1/kBL1 is the factor of cin1, 1 is used as kAL1, which can be optimized in the future.
            updateKAL1 =
                curCinxKhxKw % self->ctx.conv3dTiling->kAL1 == 0 ? 0 : self->ctx.cin0 * self->ctx.kernelHxkernelW;
        }
        if (curKBL1Kd > 1) {
            // The kAL1/kBL1 is calculated by multiplying the new cin by the kd of the previous tiling decision.
            updateKBL1 = curKBL1Kd * curCinxKhxKw;
        }
        if (updateKBL1 == 0) {
            // To ensure that kAL1/kBL1 is the factor of cin1, 1 is used as kAL1, which can be optimized in the future.
            updateKBL1 =
                curCinxKhxKw % self->ctx.conv3dTiling->kBL1 == 0 ? 0 : self->ctx.cin0 * self->ctx.kernelHxkernelW;
        }
        if (updateKAL1 % self->ctx.conv3dTiling->kL0 != 0 || updateKBL1 % self->ctx.conv3dTiling->kL0 != 0) {
            // To ensure that kL0 is the factor of kAL1/kBL1, cin0 is used as kL0, which can be optimized in the future.
            updateKL0 = self->ctx.cin0;
        }
    }

    static __aicore__ void inline PreProcessGroupOptDimTail(Intf *self)
    {
        if (!self->ctx.isGroupOptDimTail) {
            return;
        }

        if (self->ctx.singleCoreCinTail != 0) {
            ASC_OP_LOGD("[IterateAllWithGroups] singleCoreCin %d update to %d \n",
                self->ctx.singleCoreCin,
                self->ctx.singleCoreCinTail);
            self->ctx.singleCoreCin = self->ctx.singleCoreCinTail;
            uint64_t updateKAL1 = 0;
            uint64_t updateKBL1 = 0;
            uint64_t updateKL0 = 0;
            ReCalculationKTilingWithGroups(self, updateKAL1, updateKBL1, updateKL0);
            InitKDirectionBaseValue<Intf>(self, updateKAL1, updateKBL1, updateKL0);
            self->ctx.preloadAL1DbFlag = false;
            self->ctx.preloadABL1DbFlag = false;
            ASC_OP_LOGD("[IterateAllWithGroups] updateKAL1 %d updateKBL1 %d updateKL0 %d \n",
                updateKAL1,
                updateKBL1,
                updateKL0);
        }
        if (self->ctx.singleCoreCoutTail != 0) {
            ASC_OP_LOGD("[IterateAllWithGroups] singleCoreCo %d update to %d \n",
                self->ctx.singleCoreCo,
                self->ctx.singleCoreCoutTail);
            self->ctx.singleCoreCo = self->ctx.singleCoreCoutTail;
            InitCoutDirectionBaseValue<Intf>(self);
        }
    }

    static __aicore__ void inline PostProcessGroupOptDimTail(Intf *self, const uint64_t &tmpSingleCoreCo,
        const uint8_t &tmpPreloadAL1DbFlag, const uint8_t &tmpPreloadABL1DbFlag)
    {
        if (!self->ctx.isGroupOptDimTail) {
            return;
        }

        if (self->ctx.singleCoreCin != self->ctx.conv3dTiling->cinOpt) {
            self->ctx.singleCoreCin = self->ctx.conv3dTiling->cinOpt;
            InitKDirectionBaseValue<Intf>(self);
            self->ctx.preloadAL1DbFlag = tmpPreloadAL1DbFlag;
            self->ctx.preloadABL1DbFlag = tmpPreloadABL1DbFlag;
        }
        if (self->ctx.singleCoreCo != tmpSingleCoreCo) {
            self->ctx.singleCoreCo = tmpSingleCoreCo;
            InitCoutDirectionBaseValue<Intf>(self);
        }
        self->ctx.isGroupOptDimTail = false;
    }

    static __aicore__ void inline IterateAllWithGroups(
        Intf *self, const GlobalTensor<typename Intf::OutputT> &output, bool enPartialSum = false)
    {
        uint64_t weightOneGroupOptSize =
            self->ctx.conv3dTiling->cinOpt * self->ctx.kernelHxkernelWxkernelD * self->ctx.conv3dTiling->coutOpt;
        while (self->ctx.groupOptIter < self->ctx.maxGroupOptIter - 1) {
            IterateAllBase(self, output, enPartialSum);
            self->SetWeight(self->ctx.bgm[weightOneGroupOptSize]);
            if (self->ctx.enableBias) {
                self->SetBias(self->ctx.biasgm[self->ctx.conv3dTiling->coutOpt]);
            }
            self->ctx.groupOptIter++;
        }
        uint64_t tmpSingleCoreCo = self->ctx.singleCoreCo;
        uint8_t tmpPreloadAL1DbFlag = self->ctx.preloadAL1DbFlag;
        uint8_t tmpPreloadABL1DbFlag = self->ctx.preloadABL1DbFlag;
        PreProcessGroupOptDimTail(self);
        IterateAllBase(self, output, enPartialSum);
        PostProcessGroupOptDimTail(self, tmpSingleCoreCo, tmpPreloadAL1DbFlag, tmpPreloadABL1DbFlag);
        self->ctx.groupOptIter = 0;
    }
};

}  // namespace Conv3dFunc

#endif
