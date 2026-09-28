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
 * \file sync_helper.h
 * \brief All synchronization for the dav_3510 LSTM kernels. Both ends of each cross-core protocol
 *        are kept adjacent: what breaks is the pairing, not either side on its own.
 */
#ifndef OPS_RNN_SINGLE_LAYER_LSTM_SYNC_HELPER_H
#define OPS_RNN_SINGLE_LAYER_LSTM_SYNC_HELPER_H

#include "kernel_operator.h"

namespace SingleLayerLstmSync {

// Intra-core: Set immediately followed by Wait, a full stop rather than overlap.
template <AscendC::HardEvent E>
__aicore__ inline void PipeWait(event_t id = EVENT_ID0)
{
    AscendC::SetFlag<E>(id);
    AscendC::WaitFlag<E>(id);
}

__aicore__ inline void WaitMte2ToMte1() { PipeWait<AscendC::HardEvent::MTE2_MTE1>(); } // GM->L1 before L1->L0
__aicore__ inline void WaitMte1ToM() { PipeWait<AscendC::HardEvent::MTE1_M>(); }       // L1->L0/BT before Mmad
__aicore__ inline void WaitMToFix() { PipeWait<AscendC::HardEvent::M_FIX>(); }         // Mmad before the drain
__aicore__ inline void WaitFixToM() { PipeWait<AscendC::HardEvent::FIX_M>(); }         // drain read before L0C reuse
__aicore__ inline void WaitVToMte3() { PipeWait<AscendC::HardEvent::V_MTE3>(); }       // vector math before UB->GM/L1

/* The cross-core flag is raised from the FIX pipe and orders only FIX work, so an L1 -> UB push on
 * MTE1 needs this first. Measured bad=4035/4096 without it. */
__aicore__ inline void WaitMte1ToFix() { PipeWait<AscendC::HardEvent::MTE1_FIX>(); }

constexpr uint16_t FLAG_C2V = 8;
constexpr uint16_t FLAG_V2C = 9;

// CUBE -> VECTOR: the cube raises it from FIX once the drain has landed; both AIVs wait.
__aicore__ inline void CubeSignalVec() { AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(FLAG_C2V); }
__aicore__ inline void VecWaitCube() { AscendC::CrossCoreWaitFlag(FLAG_C2V); }

/* VECTOR -> CUBE is a barrier, not a counting semaphore. A wrong pairing deadlocks, so process exit
 * is the measurement: AIV0 sets + cube waits once deadlocks; both AIVs set + cube waits once
 * completes; both AIVs set + cube waits twice deadlocks. The counting rules of the other direction
 * do not carry over.
 *
 * The cube waits on M but reads the fed-back data on MTE2/MTE1, and an instruction on another pipe
 * can issue past the wait -- hence the full barrier. In a recurrent loop the pair also stops the
 * step-(t+1) drain from overwriting the UB tile while the AIVs still read step t. */
__aicore__ inline void VecSignalCube() { AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(FLAG_V2C); }

__aicore__ inline void CubeWaitVec()
{
    AscendC::CrossCoreWaitFlag<0x2, PIPE_M>(FLAG_V2C);
    AscendC::PipeBarrier<PIPE_ALL>();
}

} // namespace SingleLayerLstmSync

#endif // OPS_RNN_SINGLE_LAYER_LSTM_SYNC_HELPER_H
