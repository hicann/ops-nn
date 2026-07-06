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
 * \file sync_batch_norm_gather_stats_fused_counts_sum.h
 * \brief
 */

#ifndef SYNC_BATCH_NORM_GATHER_STATS_FUSED_COUNTS_SUM
#define SYNC_BATCH_NORM_GATHER_STATS_FUSED_COUNTS_SUM

#include "kernel_operator.h"

namespace SyncBatchNormGatherStatsFused {
using namespace AscendC;
static constexpr int64_t DOUBLE_BUFFER_NUM = 2;
static constexpr int64_t BUFFER_COL_NUM = 64;
static constexpr int64_t EIGHT_ALIGN = 8;
static constexpr int64_t SIXTEEN_ALIGN = 16;

template <typename T>
class SyncBatchNormGatherStatsFusedCountsSum {
public:
    __aicore__ inline SyncBatchNormGatherStatsFusedCountsSum(){};
    __aicore__ inline void initBuffer(TPipe& pipe, GlobalTensor<float>& counts,
                                      GlobalTensor<float>& reduceresultworkspace)
    {
        pipe_ = pipe;
        pipe_.InitBuffer(queueIn_, DOUBLE_BUFFER_NUM, BUFFER_COL_NUM * sizeof(float));
        pipe_.InitBuffer(queueOut_, DOUBLE_BUFFER_NUM, sizeof(float));
        countsGM_ = counts;
        workspaceGM_ = reduceresultworkspace;
    }

    __aicore__ inline void FinalComputeProcess(int64_t tcolAlignV, int64_t tcol, float& conutsSum) //
    {
        colAlignV_ = tcolAlignV;
        col_ = tcol;

        buffer1_ = queueIn_.AllocTensor<float>();
        buffer2_ = queueOut_.AllocTensor<float>();

        int64_t colcycleCount = (colAlignV_ + BUFFER_COL_NUM - 1) / BUFFER_COL_NUM;
        int64_t colcyclePerBlockCount = (colcycleCount + GetBlockNum() - 1) / GetBlockNum();
        int64_t colSize = BUFFER_COL_NUM;
        int64_t taskId = 0;
        for (int64_t blocktaskId = 0; blocktaskId < colcyclePerBlockCount; blocktaskId++) {
            taskId = blocktaskId * GetBlockNum() + GetBlockIdx();
            if (taskId < colcycleCount) {
                if (taskId == colcycleCount - 1) {
                    colSize = col_ - BUFFER_COL_NUM * taskId;
                }
                copyIn(taskId, colSize);
                compute(colSize);
                copyOut();
            } else {
                break;
            }
        }
        queueIn_.FreeTensor(buffer1_);
        queueOut_.FreeTensor(buffer2_);
        SyncAll();

        buffer1_ = queueIn_.AllocTensor<float>();
        buffer2_ = queueOut_.AllocTensor<float>();

        finalReduce(GetBlockNum(), conutsSum);

        queueIn_.FreeTensor(buffer1_);
        queueOut_.FreeTensor(buffer2_);
    }

    __aicore__ inline void finalReduce(int64_t colSize, float& countsSum)
    {
        uint8_t rightPad = 0;
        bool isPad = false;
        int64_t remainder = colSize % EIGHT_ALIGN;
        if (remainder != 0) {
            rightPad = EIGHT_ALIGN - remainder;
            isPad = true;
        }
        DataCopyPadExtParams<float> padParams{isPad, 0, rightPad, 0};
        DataCopyExtParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = colSize * sizeof(float);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        DataCopyPad(buffer1_, workspaceGM_, intriParams, padParams);

        uint64_t mask = colSize;
        event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(event);
        WaitFlag<HardEvent::MTE2_V>(event);

        WholeReduceSum(buffer2_, buffer1_, mask, 1, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        countsSum = buffer2_.GetValue(0);
    }

    __aicore__ inline void copyIn(int64_t colIndex, int64_t colSize)
    {
        uint8_t rightPad_ = 0;
        bool isPad = false;

        int64_t alignment = EIGHT_ALIGN;
        int64_t remainder = colSize % alignment;
        if (remainder != 0) {
            rightPad_ = alignment - remainder;
            isPad = true;
        }
        DataCopyPadExtParams<float> padParams{isPad, 0, rightPad_, 0};
        DataCopyExtParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = colSize * sizeof(float);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        event_t event_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(event_);
        WaitFlag<HardEvent::MTE3_MTE2>(event_);

        event_t event1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(event1);
        WaitFlag<HardEvent::V_MTE2>(event1);

        int64_t offset = colIndex * BUFFER_COL_NUM;

        DataCopyPad(buffer1_, countsGM_[offset], intriParams, padParams);
    }

    __aicore__ inline void compute(int64_t colSize)
    {
        int64_t colSizeMod = colSize % EIGHT_ALIGN;
        int64_t colSizeAlign = colSize;
        // 尾核补齐对齐
        if (colSizeMod != 0) {
            colSizeAlign += EIGHT_ALIGN - colSizeMod;
        }
        event_t event1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(event1);
        WaitFlag<HardEvent::MTE3_V>(event1);

        event_t event2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(event2);
        WaitFlag<HardEvent::MTE2_V>(event2);
        uint64_t mask = colSize;
        WholeReduceSum(buffer2_, buffer1_, mask, 1, 1, 1, 8);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void copyOut()
    {
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = sizeof(float);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        event_t event0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event0);
        WaitFlag<HardEvent::V_MTE3>(event0);
        int64_t offset = GetBlockIdx();
        SetAtomicAdd<float>();
        DataCopyPad(workspaceGM_[offset], buffer2_, intriParams);
        SetAtomicNone();
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> queueIn_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> queueOut_;

    LocalTensor<float> buffer1_;
    LocalTensor<float> buffer2_;

    GlobalTensor<float> countsGM_;
    GlobalTensor<float> workspaceGM_;

    int64_t colAlignV_;
    int64_t col_;
};
} // namespace SyncBatchNormGatherStatsFused
#endif