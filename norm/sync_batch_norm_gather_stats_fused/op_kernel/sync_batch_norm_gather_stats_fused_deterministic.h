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
 * \file sync_batch_norm_gather_stats_fused_deterministic.h
 * \brief
 */

#ifndef SYNC_BATCH_NORM_GATHER_STATS_FUSED_DETERMINSTIC_COMPUTE
#define SYNC_BATCH_NORM_GATHER_STATS_FUSED_DETERMINSTIC_COMPUTE

#include "kernel_operator.h"

namespace SyncBatchNormGatherStatsFused {
using namespace AscendC;
static constexpr int64_t DETERMINISTIC_DOUBLE_BUFFER = 2;
static constexpr int64_t DETERMINISTIC_ROW_TEMPLATE = 180; // 改成了双倍的size
static constexpr int64_t DETERMINISTIC_COL_TEMPLATE = 64;  // 因为Add mask的原因不太好调整
static constexpr int64_t DETERMINISTIC_UB_SIZE =
    180 * 1024 + 2 * DETERMINISTIC_DOUBLE_BUFFER * DETERMINISTIC_COL_TEMPLATE * sizeof(float);
static constexpr int64_t FLOAT_ALIGN_SIZE = 8;
static constexpr int64_t HALF_ALIGN_SIZE = 16;

template <typename T>
class SyncBatchNormGatherStatsFusedDeterminsticCompute {
public:
    __aicore__ inline SyncBatchNormGatherStatsFusedDeterminsticCompute(){};
    __aicore__ inline void initBuffer1(
        TPipe& pipe, GlobalTensor<float>& reduceOutTensorGM, GlobalTensor<float>& workspaceGM, int64_t workspaceNum)
    {
        pipe_ = pipe;
        pipe_.InitBuffer(
            queueIn_, DETERMINISTIC_DOUBLE_BUFFER,
            DETERMINISTIC_ROW_TEMPLATE * DETERMINISTIC_COL_TEMPLATE * sizeof(float));
        pipe_.InitBuffer(queueOut_, DETERMINISTIC_DOUBLE_BUFFER, DETERMINISTIC_COL_TEMPLATE * sizeof(float));

        reduceOutTensorGM_ = reduceOutTensorGM;
        workspaceGM_ = workspaceGM;
        workspaceNum_ = workspaceNum;
    }

    __aicore__ inline void initBuffer2(
        TPipe& pipe, GlobalTensor<float>& reduceOutTensorGM, GlobalTensor<T>& resultOutTensorGM,
        GlobalTensor<T>& runningMean, GlobalTensor<T>& runningMeanOut, GlobalTensor<float>& workspaceGM,
        int64_t workspaceNum, float momentum)
    {
        pipe_ = pipe;
        pipe_.InitBuffer(
            queueIn_, DETERMINISTIC_DOUBLE_BUFFER,
            DETERMINISTIC_ROW_TEMPLATE * DETERMINISTIC_COL_TEMPLATE * sizeof(float));
        pipe_.InitBuffer(queueOut_, DETERMINISTIC_DOUBLE_BUFFER, DETERMINISTIC_COL_TEMPLATE * sizeof(float));
        pipe_.InitBuffer(runningMeanOut_, DETERMINISTIC_DOUBLE_BUFFER, DETERMINISTIC_COL_TEMPLATE * sizeof(float));
        resultOutTensorGM_ = resultOutTensorGM;
        reduceOutTensorGM_ = reduceOutTensorGM;
        runningMeanGM_ = runningMean;
        runningMeanOutGM_ = runningMeanOut;
        workspaceGM_ = workspaceGM;
        workspaceNum_ = workspaceNum;
        momentum_ = momentum;
    }

    __aicore__ inline void InvstdAllDeterministic(int64_t tcolAlignV, int64_t tblockNum, int64_t tcol)
    {
        colAlignV_ = tcolAlignV;
        row_ = tblockNum;
        col_ = tcol;

        buffer1_ = queueIn_.AllocTensor<float>();
        buffer2_ = queueOut_.AllocTensor<float>();
        int64_t colcycleCount_ = (colAlignV_ + DETERMINISTIC_COL_TEMPLATE - 1) / DETERMINISTIC_COL_TEMPLATE;
        int64_t colcyclePerBlockCount_ = (colcycleCount_ + GetBlockNum() - 1) / GetBlockNum();
        int64_t rowcycleCount_ = (row_ + DETERMINISTIC_ROW_TEMPLATE - 1) / DETERMINISTIC_ROW_TEMPLATE;
        int64_t colSize = DETERMINISTIC_COL_TEMPLATE;
        int64_t rowSize = DETERMINISTIC_ROW_TEMPLATE;
        int64_t taskId = 0;
        for (int64_t blocktaskId = 0; blocktaskId < colcyclePerBlockCount_; blocktaskId++) {
            taskId = blocktaskId * GetBlockNum() + GetBlockIdx();
            if (taskId < colcycleCount_) {
                if (taskId == colcycleCount_ - 1) {
                    colSize = col_ - DETERMINISTIC_COL_TEMPLATE * taskId;
                }
                for (int64_t i = 0; i < rowcycleCount_; i++) {
                    if (i == rowcycleCount_ - 1) {
                        rowSize = row_ - DETERMINISTIC_ROW_TEMPLATE * i;
                    }
                    copyIn(taskId, i, colSize, rowSize);
                    compute(taskId, i, colSize, rowSize);
                }
                copyReduceResultOut(taskId, colSize);
                rowSize = DETERMINISTIC_ROW_TEMPLATE;
            } else {
                break;
            }
        }
        queueIn_.FreeTensor(buffer1_);
        queueOut_.FreeTensor(buffer2_);
    }

    __aicore__ inline void MeanAllDeterministicWithRunningMeanUpdate(
        int64_t tcolAlignV, int64_t tblockNum, int64_t tcol)
    {
        colAlignV_ = tcolAlignV;
        row_ = tblockNum;
        col_ = tcol;

        buffer1_ = queueIn_.AllocTensor<float>();
        buffer2_ = queueOut_.AllocTensor<float>();
        buffer3_ = runningMeanOut_.AllocTensor<float>();
        int64_t colcycleCount = (colAlignV_ + DETERMINISTIC_COL_TEMPLATE - 1) / DETERMINISTIC_COL_TEMPLATE;
        int64_t colcyclePerBlockCount = (colcycleCount + GetBlockNum() - 1) / GetBlockNum();
        int64_t rowcycleCount = (row_ + DETERMINISTIC_ROW_TEMPLATE - 1) / DETERMINISTIC_ROW_TEMPLATE;
        int64_t colSize = DETERMINISTIC_COL_TEMPLATE;
        int64_t rowSize = DETERMINISTIC_ROW_TEMPLATE;
        int64_t taskId = 0;
        for (int64_t blocktaskId = 0; blocktaskId < colcyclePerBlockCount; blocktaskId++) {
            taskId = blocktaskId * GetBlockNum() + GetBlockIdx();
            if (taskId < colcycleCount) {
                if (taskId == colcycleCount - 1) {
                    colSize = col_ - DETERMINISTIC_COL_TEMPLATE * taskId;
                }
                for (int64_t i = 0; i < rowcycleCount; i++) {
                    if (i == rowcycleCount - 1) {
                        rowSize = row_ - DETERMINISTIC_ROW_TEMPLATE * i;
                    }
                    copyIn(taskId, i, colSize, rowSize);
                    compute(taskId, i, colSize, rowSize);
                }
                updateRunningMean(taskId, colSize);
                copyReduceResultOut(taskId, colSize);
                copyReduceResultOutWithRunningMeanOut(taskId, colSize);
                rowSize = DETERMINISTIC_ROW_TEMPLATE;
            } else {
                break;
            }
        }
        queueIn_.FreeTensor(buffer1_);
        queueOut_.FreeTensor(buffer2_);
        runningMeanOut_.FreeTensor(buffer3_);
    }

    __aicore__ inline void copyIn(int64_t colIndex, int64_t rowIndex, int64_t colSize, int64_t rowSize)
    {
        uint8_t rightPad = 0;
        bool isPad = false;
        int64_t colSizeMod = colSize % FLOAT_ALIGN_SIZE;
        // 尾核补齐对齐
        if (colSizeMod != 0) {
            rightPad = FLOAT_ALIGN_SIZE - colSizeMod;
            isPad = true;
        }
        DataCopyPadExtParams<float> padParams{isPad, 0, rightPad, 0};
        DataCopyExtParams intriParams;
        intriParams.blockCount = rowSize;
        intriParams.blockLen = colSize * sizeof(float);
        intriParams.srcStride = (colAlignV_ * workspaceNum_ - colSize) * sizeof(float);
        intriParams.dstStride = (DETERMINISTIC_COL_TEMPLATE - (colSize + rightPad)) / FLOAT_ALIGN_SIZE;
        event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(event);
        WaitFlag<HardEvent::MTE3_MTE2>(event);

        event_t event1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(event1);
        WaitFlag<HardEvent::V_MTE2>(event1);

        int64_t offset = (colIndex * DETERMINISTIC_COL_TEMPLATE) +
                         rowIndex * colAlignV_ * workspaceNum_ * DETERMINISTIC_ROW_TEMPLATE;
        DataCopyPad(buffer1_, workspaceGM_[offset], intriParams, padParams);
    }

    __aicore__ inline void compute(int64_t colIndex, int64_t rowIndex, int64_t colSize, int64_t rowSize)
    {
        int64_t colSizeMod = colSize % FLOAT_ALIGN_SIZE;
        int64_t colSizeAlign = colSize;
        // 尾核补齐对齐
        if (colSizeMod != 0) {
            colSizeAlign += FLOAT_ALIGN_SIZE - colSizeMod;
        }
        event_t event1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(event1);
        WaitFlag<HardEvent::MTE3_V>(event1);

        event_t event2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(event2);
        WaitFlag<HardEvent::MTE2_V>(event2);

        Duplicate(buffer2_, static_cast<float>(0.0), DETERMINISTIC_COL_TEMPLATE);
        PipeBarrier<PIPE_V>();
        uint64_t mask = colSize;
        uint8_t repeatTimes = MAX_REPEAT_TIMES;
        BinaryRepeatParams binaryRepeatParams;
        binaryRepeatParams.dstBlkStride = 1;
        binaryRepeatParams.src0BlkStride = 1;
        binaryRepeatParams.src1BlkStride = 1;
        binaryRepeatParams.dstRepStride = 0;
        binaryRepeatParams.src0RepStride = FLOAT_ALIGN_SIZE;
        binaryRepeatParams.src1RepStride = 0;
        int64_t rowRepeatTimes = (rowSize + MAX_REPEAT_TIMES - 1) / MAX_REPEAT_TIMES;
        for (int64_t i = 0; i < rowRepeatTimes; i++) {
            if (i == rowRepeatTimes - 1) {
                repeatTimes = rowSize - (rowRepeatTimes - 1) * MAX_REPEAT_TIMES;
            }
            Add(buffer2_, buffer1_[i * MAX_REPEAT_TIMES], buffer2_, mask, repeatTimes, binaryRepeatParams);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void updateRunningMean(int64_t colIndex, int64_t colSize)
    {
        uint8_t rightPad = 0;
        bool isPad = false;

        int64_t alignment = IsSameType<T, float>::value ? FLOAT_ALIGN_SIZE : HALF_ALIGN_SIZE;
        int64_t remainder = colSize % alignment;
        if (remainder != 0) {
            rightPad = alignment - remainder;
            isPad = true;
        }
        DataCopyPadExtParams<T> padParams{isPad, 0, rightPad, 0};
        DataCopyExtParams intriParams;
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        intriParams.blockCount = 1;
        intriParams.blockLen = colSize * sizeof(T);

        event_t event1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(event1);
        WaitFlag<HardEvent::V_MTE2>(event1);

        int64_t offset = colIndex * DETERMINISTIC_COL_TEMPLATE;

        if constexpr (IsSameType<T, float>::value) {
            DataCopyPad(buffer1_.ReinterpretCast<T>(), runningMeanGM_[offset], intriParams, padParams);
        } else {
            DataCopyPad(
                buffer1_.ReinterpretCast<T>()[colSize + rightPad], runningMeanGM_[offset], intriParams, padParams);
        }
        event_t event2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(event2);
        WaitFlag<HardEvent::MTE2_V>(event2);
        if constexpr (!IsSameType<T, float>::value) {
            Cast(buffer1_, buffer1_.ReinterpretCast<T>()[colSize + rightPad], RoundMode::CAST_NONE, colSize + rightPad);
            PipeBarrier<PIPE_V>();
        }
        Muls(buffer1_, buffer1_, 1.0f - momentum_, colSize);
        PipeBarrier<PIPE_V>();
        Muls(buffer3_, buffer2_, momentum_, colSize);
        PipeBarrier<PIPE_V>();
        Add(buffer3_, buffer3_, buffer1_, colSize);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void copyReduceResultOut(int64_t colIndex, int64_t colSize)
    {
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = colSize * sizeof(float);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event);
        WaitFlag<HardEvent::V_MTE3>(event);

        int64_t offset = colIndex * DETERMINISTIC_COL_TEMPLATE;

        DataCopyPad(reduceOutTensorGM_[offset], buffer2_, intriParams);
        PipeBarrier<PIPE_MTE3>();
    }
    __aicore__ inline void copyReduceResultOutWithRunningMeanOut(int64_t colIndex, int64_t colSize)
    {
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = colSize * sizeof(T);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;
        int64_t offset = colIndex * DETERMINISTIC_COL_TEMPLATE;
        if constexpr (!IsSameType<T, float>::value) {
            event_t event1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(event1);
            WaitFlag<HardEvent::MTE3_V>(event1);
            Cast(buffer2_.ReinterpretCast<T>(), buffer2_, RoundMode::CAST_RINT, colSize);
            PipeBarrier<PIPE_V>();
            Cast(buffer3_.ReinterpretCast<T>(), buffer3_, RoundMode::CAST_RINT, colSize);
        }

        event_t event2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event2);
        WaitFlag<HardEvent::V_MTE3>(event2);
        DataCopyPad(resultOutTensorGM_[offset], buffer2_.ReinterpretCast<T>(), intriParams);
        DataCopyPad(runningMeanOutGM_[offset], buffer3_.ReinterpretCast<T>(), intriParams);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECOUT, DETERMINISTIC_DOUBLE_BUFFER> queueOut_;
    TQue<QuePosition::VECOUT, DETERMINISTIC_DOUBLE_BUFFER> runningMeanOut_;
    TQue<QuePosition::VECIN, DETERMINISTIC_DOUBLE_BUFFER> queueIn_;
    LocalTensor<float> buffer1_;
    LocalTensor<float> buffer2_;
    LocalTensor<float> buffer3_;

    GlobalTensor<T> resultOutTensorGM_;
    GlobalTensor<T> runningMeanGM_;
    GlobalTensor<T> runningMeanOutGM_;
    GlobalTensor<float> reduceOutTensorGM_;
    GlobalTensor<float> workspaceGM_;

    int64_t workspaceNum_;
    int64_t colAlignV_;
    int64_t row_;
    int64_t col_;
    float momentum_;
};
} // namespace SyncBatchNormGatherStatsFused
#endif