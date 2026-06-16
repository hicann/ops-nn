/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 /* !
 * \file sync_batch_norm_gather_stats_fused_compute.h
 * \brief
 */

#ifndef SYNC_BATCH_NORM_GATHER_STATS_FUSED_RUNNING_COMPUTE
#define SYNC_BATCH_NORM_GATHER_STATS_FUSED_RUNNING_COMPUTE

#include "kernel_operator.h"

namespace SyncBatchNormGatherStatsFused {
using namespace AscendC;
static constexpr int64_t DOUBLE_BUFFER = 2;
// static constexpr int64_t ROW_TEMPLATE = 180;
static constexpr int64_t COL_TEMPLATE = 64; // 因为Add mask的原因不太好调整
static constexpr int64_t UB_SIZE = 180 * 1024 + 2 * DOUBLE_BUFFER * COL_TEMPLATE * sizeof(float);
static constexpr int64_t FLOAT_ALIGN = 8;
static constexpr int64_t HALF_ALIGN = 16;

template <typename T>
class SyncBatchNormGatherStatsFusedRunningCompute {
public:
    __aicore__ inline SyncBatchNormGatherStatsFusedRunningCompute(){};
    __aicore__ inline void initBuffer(
        TPipe& pipe, GlobalTensor<T>& runningVarOutTensorGM, GlobalTensor<T>& invstdAllOutTensorGM,
        GlobalTensor<T>& runningVarInTensorGM, GlobalTensor<float>& workspaceGM, float countSum, float momentum,
        float eps, int64_t workspaceNum)
    {
        pipe_ = pipe;
        pipe_.InitBuffer(queueRunningVarIn_, DOUBLE_BUFFER, COL_TEMPLATE * sizeof(float));
        pipe_.InitBuffer(queueRunningVarOut_, DOUBLE_BUFFER, COL_TEMPLATE * sizeof(float));
        pipe_.InitBuffer(queueVarAllIn_, DOUBLE_BUFFER, COL_TEMPLATE * sizeof(float));
        pipe_.InitBuffer(queueInvstdAllOut_, DOUBLE_BUFFER, COL_TEMPLATE * sizeof(float));
        runningVarOutTensorGM_ = runningVarOutTensorGM;
        invstdAllOutTensorGM_ = invstdAllOutTensorGM;
        runningVarInTensorGM_ = runningVarInTensorGM;
        workspaceGM_ = workspaceGM;
        workspaceNum_ = workspaceNum;
        countSum_ = countSum;
        unbiasCountsSum_ = countSum - 1.0f;
        momentum_ = momentum;
        eps_ = eps;
    }

    __aicore__ inline void FinalComputeProcess(int64_t tcolAlignV, int64_t rowOfWorkspace, int64_t tcol) //
    {
        colAlignV_ = tcolAlignV;
        row_ = rowOfWorkspace;
        col_ = tcol;
        buffer1_ = queueVarAllIn_.AllocTensor<float>();
        buffer2_ = queueRunningVarIn_.AllocTensor<float>();
        buffer3_ = queueRunningVarOut_.AllocTensor<float>();
        buffer4_ = queueInvstdAllOut_.AllocTensor<float>();

        int64_t colcycleCount = (colAlignV_ + COL_TEMPLATE - 1) / COL_TEMPLATE;
        int64_t colcyclePerBlockCount = (colcycleCount + GetBlockNum() - 1) / GetBlockNum();
        int64_t colSize = COL_TEMPLATE;
        int64_t taskId = 0;
        for (int64_t blocktaskId = 0; blocktaskId < colcyclePerBlockCount; blocktaskId++) {
            taskId = blocktaskId * GetBlockNum() + GetBlockIdx();
            if (taskId < colcycleCount) {
                if (taskId == colcycleCount - 1) {
                    colSize = col_ - COL_TEMPLATE * taskId;
                }
                copyIn(taskId, colSize);
                compute(taskId, colSize);
                copyOut(taskId, colSize);
            } else {
                break;
            }
        }
        queueVarAllIn_.FreeTensor(buffer1_);
        queueRunningVarIn_.FreeTensor(buffer2_);
        queueRunningVarOut_.FreeTensor(buffer3_);
        queueInvstdAllOut_.FreeTensor(buffer4_);
    }

    __aicore__ inline void copyIn(int64_t colIndex, int64_t colSize)
    {
        uint8_t rightPad = 0;
        bool isPad = false;
        int64_t colSizeMod = colSize % FLOAT_ALIGN;
        if (colSizeMod != 0) {
            rightPad = FLOAT_ALIGN - colSizeMod;
            isPad = true;
        }
        DataCopyPadExtParams<float> padParams{isPad, 0, rightPad, 0};
        DataCopyExtParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = colSize * sizeof(float);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(event);
        WaitFlag<HardEvent::MTE3_MTE2>(event);

        event_t event1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(event1);
        WaitFlag<HardEvent::V_MTE2>(event1);

        int64_t offset = colIndex * COL_TEMPLATE;
        DataCopyPad(buffer1_, workspaceGM_[offset], intriParams, padParams); // 搬入VarAll
        PipeBarrier<PIPE_MTE2>();

        intriParams.blockLen = colSize * sizeof(T);
        int64_t colSize_16Mod = colSize % HALF_ALIGN;
        if (!IsSameType<T, float>::value && colSize_16Mod != 0) {
            rightPad = HALF_ALIGN - colSize_16Mod;
            isPad = true;
        }
        DataCopyPadExtParams<T> padParamsT{isPad, 0, rightPad, 0};
        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(buffer2_, runningVarInTensorGM_[offset], intriParams, padParamsT);
        } else {
            DataCopyPad(
                buffer2_.ReinterpretCast<T>()[colSize + rightPad], runningVarInTensorGM_[offset], intriParams,
                padParamsT);
        }

        if constexpr (!IsSameType<T, float>::value) {
            event_t event2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(event2);
            WaitFlag<HardEvent::MTE2_V>(event2);
            Cast(buffer2_, buffer2_.ReinterpretCast<T>()[colSize + rightPad], RoundMode::CAST_NONE, colSize + rightPad);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void compute(int64_t colIndex, int64_t colSize)
    {
        int64_t colSizeMod = colSize % FLOAT_ALIGN;
        int64_t colSizeAlign = colSize;
        // 尾核补齐对齐
        if (colSizeMod != 0) {
            colSizeAlign += FLOAT_ALIGN - colSizeMod;
        }
        event_t event1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(event1);
        WaitFlag<HardEvent::MTE3_V>(event1);

        event_t event2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(event2);
        WaitFlag<HardEvent::MTE2_V>(event2);

        Muls(buffer2_, buffer2_, 1 - momentum_, colSize);
        PipeBarrier<PIPE_V>();

        Duplicate(buffer3_, unbiasCountsSum_, colSize);
        PipeBarrier<PIPE_V>();

        Div(buffer3_, buffer1_, buffer3_, colSize);
        PipeBarrier<PIPE_V>();

        Muls(buffer3_, buffer3_, momentum_, colSize);
        PipeBarrier<PIPE_V>();

        Add(buffer3_, buffer3_, buffer2_, colSize);
        PipeBarrier<PIPE_V>();

        Duplicate(buffer2_, countSum_, colSize);
        PipeBarrier<PIPE_V>();

        Div(buffer4_, buffer1_, buffer2_, colSize);
        PipeBarrier<PIPE_V>();

        Adds(buffer4_, buffer4_, eps_, colSize);
        PipeBarrier<PIPE_V>();

        Sqrt(buffer4_, buffer4_, colSize);
        PipeBarrier<PIPE_V>();

        Duplicate(buffer2_, 1.0f, colSize);
        PipeBarrier<PIPE_V>();

        Div(buffer4_, buffer2_, buffer4_, colSize);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void copyOut(int64_t colIndex, int64_t colSize)
    {
        if constexpr (!IsSameType<T, float>::value) {
            Cast(buffer3_.ReinterpretCast<T>(), buffer3_, RoundMode::CAST_RINT, colSize);
            PipeBarrier<PIPE_V>();
            Cast(buffer4_.ReinterpretCast<T>(), buffer4_, RoundMode::CAST_RINT, colSize);
            PipeBarrier<PIPE_V>();
        }
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = colSize * sizeof(T);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        event_t event0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event0);
        WaitFlag<HardEvent::V_MTE3>(event0);
        int64_t offset = colIndex * COL_TEMPLATE;
        DataCopyPad(runningVarOutTensorGM_[offset], buffer3_.ReinterpretCast<T>(), intriParams);
        DataCopyPad(invstdAllOutTensorGM_[offset], buffer4_.ReinterpretCast<T>(), intriParams);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> queueRunningVarOut_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> queueInvstdAllOut_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> queueRunningVarIn_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> queueVarAllIn_;
    LocalTensor<float> buffer1_;
    LocalTensor<float> buffer2_;
    LocalTensor<float> buffer3_;
    LocalTensor<float> buffer4_;

    GlobalTensor<T> runningVarOutTensorGM_;
    GlobalTensor<T> invstdAllOutTensorGM_;
    GlobalTensor<T> runningVarInTensorGM_;
    GlobalTensor<float> workspaceGM_;

    int64_t workspaceNum_;
    int64_t colAlignV_;
    int64_t row_;
    int64_t col_;
    float countSum_;
    float unbiasCountsSum_;
    float momentum_;
    float eps_;
};
} // namespace SyncBatchNormGatherStatsFused
#endif