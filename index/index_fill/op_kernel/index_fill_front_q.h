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
 * \file index_fill_front_q.h
 * \brief front axis and Q >= 1024. core will split Q.
 */
#ifndef INDEX_FILL_FRONT_Q_H
#define INDEX_FILL_FRONT_Q_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "index_fill_base.h"

namespace IndexFillNS {
using namespace AscendC;

constexpr uint32_t REPEAT_TIMES = 4;
constexpr uint32_t BUFFER_NUM = 2;

template <typename T, typename U>
class IndexFillFrontQ : public IndexFillBase<T, U>{
public:
  __aicore__ inline void Process() {
    this->ExecIndicesTask();
    FrontAxisQ();
  }

  __aicore__ inline void FrontAxisQ() {
    // x.shape = [P, N, Q]
    uint64_t PCoreNum = this->coreNum / this->P;
    PCoreNum = PCoreNum == 0 ? 1 : PCoreNum;
    uint64_t NCoreNum = PCoreNum / this->N;
    NCoreNum = NCoreNum == 0 ? 1 : NCoreNum;
    // UB布局：wsLocal[WS_UB_LENGTH * sizeof(half)], xLocal[X_UB_LENGTH * sizeof(T)]，最后是valueLocal
    this->wsLocal = this->allUbLocal.template ReinterpretCast<half>();
    this->xLocal = this->allUbLocal[WS_UB_LENGTH * sizeof(half)].template ReinterpretCast<T>();
    uint64_t valueLocalStartUb = WS_UB_LENGTH * sizeof(half) + X_UB_LENGTH * sizeof(T);
    this->valueLocal = this->allUbLocal[valueLocalStartUb].template ReinterpretCast<T>();
    this->CopyIn(this->valueGm, 0, valueLocalStartUb, sizeof(T));
    this->PIPE_MTE2_S();
    RepeatValue();

    uint64_t wsBlocks = this->N / WS_UB_LENGTH; // 标记数组已脱离indicesNum, 完全和N对齐了
    uint64_t wsLeft = this->N - wsBlocks * WS_UB_LENGTH;
    for (uint64_t p = 0; p < this->P; p++) {
        for(uint64_t i = 0; i <= wsBlocks; i++) {
            if (i == wsBlocks && wsLeft == 0) {
                break;
            }
            uint64_t wsNums = (i == wsBlocks ? wsLeft : WS_UB_LENGTH);
            uint64_t gmStartHalf = WS_UB_LENGTH * i;
            uint64_t mteNumsHalf = wsNums;
            this->CopyInWsGm(gmStartHalf, this->wsLocal, mteNumsHalf);
            this->PIPE_MTE2_S();
            for (uint64_t j = 0; j < wsNums; j++) {
                float wsValue = this->wsLocal.GetValue(j);
                if (wsValue > 0) {
                    CopyOutQ(p, i * WS_UB_LENGTH + j, NCoreNum, true);
                } else {
                    CopyOutQ(p, i * WS_UB_LENGTH + j, NCoreNum, false);
                }
            }
        }
    }
    this->PIPE_MTE3_S();
  }

  __aicore__ inline void CopyOutQ(uint64_t p, uint64_t n, uint64_t NCoreNum, bool update) {
    // 用于多核搬出一个Q
    uint64_t frontCoreNums = 0;
    uint64_t tailCoreNums = 0;
    uint64_t frontCoreQNums = 0;
    uint64_t tailCoreQNums = 0;
    this->coreSplit(NCoreNum, this->Q, frontCoreNums, tailCoreNums, frontCoreQNums, tailCoreQNums);

    uint64_t coreQStart = 0;
    uint64_t coreQNums = 0;
    if (GetBlockIdx() - this->corePtr < frontCoreNums) {
        coreQStart = (GetBlockIdx() - this->corePtr) * frontCoreQNums;
        coreQNums = frontCoreQNums;
    } else if(tailCoreNums > 0){
        coreQStart = frontCoreNums * frontCoreQNums + (GetBlockIdx() - this->corePtr - frontCoreNums) * tailCoreQNums;
        coreQNums = tailCoreQNums;
    }

    bool willWork = this->WillWorkForTask(frontCoreNums + tailCoreNums);
    if (willWork) {
        uint64_t copyTimes = coreQNums / (X_UB_LENGTH / 2);
        uint64_t leftQ = coreQNums - copyTimes * (X_UB_LENGTH / 2);
        for (uint64_t i = 0; i <= copyTimes; i++) {
            if(i == copyTimes && leftQ == 0) {
                break;
            }
            uint64_t QNums = (i == copyTimes ? leftQ : (X_UB_LENGTH / 2));
            uint64_t gmStartInt8 = (p * (this->N * this->Q) + n * this->Q + coreQStart + i * (X_UB_LENGTH / 2)) * sizeof(T);
            uint64_t mteNumsInt8 = QNums * sizeof(T);
            if (update) {
                uint64_t ubStartInt8 = WS_UB_LENGTH * sizeof(half);
                this->CopyOut(this->yGm, gmStartInt8, ubStartInt8, mteNumsInt8);
            } else {
                uint64_t ubStartInt8 = WS_UB_LENGTH * sizeof(half) + (X_UB_LENGTH / 2) * sizeof(T);
                this->CopyIn(this->xGm, gmStartInt8, ubStartInt8, mteNumsInt8);
                this->PIPE_MTE2_S();
                this->CopyOut(this->yGm, gmStartInt8, ubStartInt8, mteNumsInt8);
                this->PIPE_MTE3_S();
            }
        }
    }
  }

  __aicore__ inline void RepeatValue() {
    // 用于核内将Value复制X_UB_LENGTH个，不用dup是为了兼容更多类型
    // 前32个直接setValue，然后倍增到128，后面按128重复
    T value = this->valueLocal.GetValue(0);
    for (uint64_t i = 0; i < INT8_ALIGNED_NUM; i++) {
        this->xLocal.SetValue(i, value);
    }
    PIPE_S_V();
    for (uint64_t i = 1; i < REPEAT_TIMES; i++) {
        DataCopy(this->xLocal[i * INT8_ALIGNED_NUM], this->xLocal, INT8_ALIGNED_NUM);
        PipeBarrier<PIPE_V>();
    }
    for (uint64_t i = 1; i < X_UB_LENGTH / BUFFER_NUM / COMPARE_ALIGNED; i++) {
        if (i * COMPARE_ALIGNED > this->Q) {
            break;
        }
        DataCopy(this->xLocal[i * COMPARE_ALIGNED], this->xLocal, COMPARE_ALIGNED);
        PipeBarrier<PIPE_V>();
    }
    this->PIPE_V_S();
  }

  __aicore__ inline void PIPE_S_V() {
    event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIDSToV);
    WaitFlag<HardEvent::S_V>(eventIDSToV);
  }
};

template <typename T, typename U>
__aicore__ void index_fill_front_q(GM_ADDR x, GM_ADDR indices, GM_ADDR value, GM_ADDR y, GM_ADDR workspace,
                                   GM_ADDR tiling, const IndexFillTilingData* tilingData, TPipe* tPipe) {
    if (sizeof(T) == sizeof(half)) {
        IndexFillFrontQ<half, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    } else if (sizeof(T) == sizeof(float)) {
        IndexFillFrontQ<float, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    } else if (sizeof(T) == 1) {
        IndexFillFrontQ<uint8_t, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    } else {
        IndexFillFrontQ<uint64_t, U> op;
        op.Init(x, indices, value, y, workspace, tiling, tilingData, tPipe);
        op.Process();
    }
}
}
#endif