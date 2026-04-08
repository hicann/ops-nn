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
 * \file ADA_LAYER_NORM_GRAD_determinstic_compute.h
 * \brief
 */

#ifndef ADA_LAYER_NORM_GRAD_DETERMINSTIC_COMPUTE
#define ADA_LAYER_NORM_GRAD_DETERMINSTIC_COMPUTE

#include "kernel_operator.h"

namespace AdaLayerNormGrad {
using namespace AscendC;
static constexpr int64_t DOUBLE_BUFFER = 2;
static constexpr int64_t ROW_TEMPLATE = 180;
static constexpr int64_t COL_TEMPLATE = 64; // 因为Add mask的原因不太好调整
static constexpr int64_t UB_SIZE = 180 * 1024 + 2 * DOUBLE_BUFFER * COL_TEMPLATE * sizeof(float);
static constexpr int64_t FLOAT_ALIGN = 8;
static constexpr int64_t MAX_REPEAT_TIMES = 255;

template <typename T>
class AdaLayerNormGradDeterminsticCompute {
public:
    __aicore__ inline AdaLayerNormGradDeterminsticCompute(){};
    __aicore__ inline void initBuffer(
        TPipe& pipe, GlobalTensor<T>& pdGammaOutTensorGM, GlobalTensor<T>& pdBetaOutTensorGM,
        GlobalTensor<float>& workspaceGM, int64_t workspaceNum)
    {
        pipe_ = pipe;
        pipe_.InitBuffer(queueGammaIn_, DOUBLE_BUFFER, ROW_TEMPLATE * COL_TEMPLATE * sizeof(float));
        pipe_.InitBuffer(queueGammaOut_, DOUBLE_BUFFER, COL_TEMPLATE * sizeof(float));
        pipe_.InitBuffer(queueBetaIn_, DOUBLE_BUFFER, ROW_TEMPLATE * COL_TEMPLATE * sizeof(float));
        pipe_.InitBuffer(queueBetaOut_, DOUBLE_BUFFER, COL_TEMPLATE * sizeof(float));
        pdGammaOutTensorGM_ = pdGammaOutTensorGM;
        pdBetaOutTensorGM_ = pdBetaOutTensorGM;
        workspaceGM_ = workspaceGM;
        workspaceNum_ = workspaceNum;
    }

    __aicore__ inline void BatchFinalProcessDeterministic(
        int64_t batch, int64_t colAlignV, int64_t blockNum, int64_t col, GlobalTensor<T> pdScaleOutGM,
        GlobalTensor<T> pdShiftOutGM)
    {
        colAlignV_ = colAlignV;
        row_ = blockNum;
        col_ = col;

        // 提前 Alloc Tensor，避免在 batch 循环中频繁申请释放
        buffer1_ = queueGammaIn_.AllocTensor<float>();
        buffer2_ = queueGammaOut_.AllocTensor<float>();
        buffer3_ = queueBetaIn_.AllocTensor<float>();
        buffer4_ = queueBetaOut_.AllocTensor<float>();

        // 保存原始 workspace 基地址，防止循环内偏移叠加
        GlobalTensor<float> workspaceBase = workspaceGM_;

        for (int64_t i = 0; i < batch; ++i) {
            // 1. 更新当前 batch 的偏移基地址
            int64_t offset = (2 + 2 * i) * colAlignV_;
            workspaceGM_ = workspaceBase[offset];

            // 根据原调用处的逻辑更新 Tensor1(Gamma) 和 Tensor2(Beta)
            if (2 * i < batch) {
                pdGammaOutTensorGM_ = pdScaleOutGM[2 * i * col_];
            } else {
                pdGammaOutTensorGM_ = pdShiftOutGM[(2 * i - batch) * col_];
            }

            if ((2 * i + 1) < batch) {
                pdBetaOutTensorGM_ = pdScaleOutGM[(2 * i + 1) * col_];
            } else {
                pdBetaOutTensorGM_ = pdShiftOutGM[(2 * i + 1 - batch) * col_];
            }

            // 2. 执行原有核心计算逻辑 (由原本的 FinalProcessDeterministic 拆解而来)
            int64_t colcycleCount = (colAlignV_ + COL_TEMPLATE - 1) / COL_TEMPLATE;
            int64_t colcyclePerBlockCount = (colcycleCount + GetBlockNum() - 1) / GetBlockNum();
            int64_t rowcycleCount = (row_ + ROW_TEMPLATE - 1) / ROW_TEMPLATE;

            for (int64_t blocktaskId = 0; blocktaskId < colcyclePerBlockCount; blocktaskId++) {
                int64_t taskId = blocktaskId * GetBlockNum() + GetBlockIdx();
                if (taskId < colcycleCount) {
                    int64_t colSize = (taskId == colcycleCount - 1) ? (col_ - COL_TEMPLATE * taskId) : COL_TEMPLATE;

                    for (int64_t r = 0; r < rowcycleCount; r++) {
                        int64_t rowSize = (r == rowcycleCount - 1) ? (row_ - ROW_TEMPLATE * r) : ROW_TEMPLATE;

                        // 调用类内原有的 copyIn 和 compute
                        copyIn(taskId, r, colSize, rowSize);
                        compute(taskId, r, colSize, rowSize);
                    }
                    copyOut(taskId, colSize);
                }
            }
            // 确保当前 batch 写入 GM 完成
            PipeBarrier<PIPE_MTE3>();
        }

        // 3. 循环结束后统一释放
        queueGammaIn_.FreeTensor(buffer1_);
        queueGammaOut_.FreeTensor(buffer2_);
        queueBetaIn_.FreeTensor(buffer3_);
        queueBetaOut_.FreeTensor(buffer4_);
    }

    __aicore__ inline void FinalProcessDeterministic(int64_t tcolAlignV, int64_t tblockNum, int64_t tcol)
    {
        colAlignV_ = tcolAlignV;
        row_ = tblockNum;
        col_ = tcol;
        buffer1_ = queueGammaIn_.AllocTensor<float>();
        buffer2_ = queueGammaOut_.AllocTensor<float>();
        buffer3_ = queueBetaIn_.AllocTensor<float>();
        buffer4_ = queueBetaOut_.AllocTensor<float>();
        int64_t colcycleCount = (colAlignV_ + COL_TEMPLATE - 1) / COL_TEMPLATE;
        int64_t colcyclePerBlockCount = (colcycleCount + GetBlockNum() - 1) / GetBlockNum();
        int64_t rowcycleCount = (row_ + ROW_TEMPLATE - 1) / ROW_TEMPLATE;
        int64_t colSize = COL_TEMPLATE;
        int64_t rowSize = ROW_TEMPLATE;
        int64_t taskId = 0;
        for (int64_t blocktaskId = 0; blocktaskId < colcyclePerBlockCount; blocktaskId++) {
            taskId = blocktaskId * GetBlockNum() + GetBlockIdx();
            if (taskId < colcycleCount) {
                if (taskId == colcycleCount - 1) {
                    colSize = col_ - COL_TEMPLATE * taskId;
                }
                for (int64_t i = 0; i < rowcycleCount; i++) {
                    if (i == rowcycleCount - 1) {
                        rowSize = row_ - ROW_TEMPLATE * i;
                    }
                    copyIn(taskId, i, colSize, rowSize);
                    compute(taskId, i, colSize, rowSize);
                }
                copyOut(taskId, colSize);
                rowSize = ROW_TEMPLATE;
            } else {
                break;
            }
        }
        queueGammaIn_.FreeTensor(buffer1_);
        queueGammaOut_.FreeTensor(buffer2_);
        queueBetaIn_.FreeTensor(buffer3_);
        queueBetaOut_.FreeTensor(buffer4_);
    }

    __aicore__ inline void copyIn(int64_t colIndex, int64_t rowIndex, int64_t colSize, int64_t rowSize)
    {
        uint8_t rightPad = 0;
        bool isPad = false;
        int64_t colSizeMod = colSize % FLOAT_ALIGN;
        // 尾核补齐对齐
        if (colSizeMod != 0) {
            rightPad = FLOAT_ALIGN - colSizeMod;
            isPad = true;
        }
        DataCopyPadExtParams<float> padParams{isPad, 0, rightPad, 0};
        DataCopyExtParams intriParams;
        intriParams.blockCount = rowSize;
        intriParams.blockLen = colSize * sizeof(float);
        intriParams.srcStride = (colAlignV_ * workspaceNum_ - colSize) * sizeof(float);
        intriParams.dstStride = (COL_TEMPLATE - (colSize + rightPad)) / FLOAT_ALIGN;
        TEventID eventID = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
        SetFlag<HardEvent::V_MTE2>(eventID);
        WaitFlag<HardEvent::V_MTE2>(eventID);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventID);
        int64_t offset = (colIndex * COL_TEMPLATE) + rowIndex * colAlignV_ * workspaceNum_ * ROW_TEMPLATE;
        DataCopyPad(buffer1_, workspaceGM_[offset], intriParams, padParams);
        DataCopyPad(buffer3_, workspaceGM_[offset + colAlignV_], intriParams, padParams);
    }

    __aicore__ inline void compute(int64_t colIndex, int64_t rowIndex, int64_t colSize, int64_t rowSize)
    {
        int64_t colSizeMod = colSize % FLOAT_ALIGN;
        int64_t colSizeAlign = colSize;
        // 尾核补齐对齐
        if (colSizeMod != 0) {
            colSizeAlign += FLOAT_ALIGN - colSizeMod;
        }
        TEventID eventID = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
        SetFlag<HardEvent::MTE2_V>(eventID);
        WaitFlag<HardEvent::MTE2_V>(eventID);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventID);
        TEventID eventID1 = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
        SetFlag<HardEvent::MTE3_V>(eventID1);
        WaitFlag<HardEvent::MTE3_V>(eventID1);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventID1);
        Duplicate(buffer2_, static_cast<float>(0.0), COL_TEMPLATE);
        Duplicate(buffer4_, static_cast<float>(0.0), COL_TEMPLATE);
        PipeBarrier<PIPE_V>();
        uint64_t mask = colSize;
        uint8_t repeatTimes = MAX_REPEAT_TIMES;
        BinaryRepeatParams binaryRepeatParams;
        binaryRepeatParams.dstBlkStride = 1;
        binaryRepeatParams.src0BlkStride = 1;
        binaryRepeatParams.src1BlkStride = 1;
        binaryRepeatParams.dstRepStride = 0;
        binaryRepeatParams.src0RepStride = FLOAT_ALIGN;
        binaryRepeatParams.src1RepStride = 0;
        int64_t rowRepeatTimes = (rowSize + MAX_REPEAT_TIMES - 1) / MAX_REPEAT_TIMES;
        for (int64_t i = 0; i < rowRepeatTimes; i++) {
            if (i == rowRepeatTimes - 1) {
                repeatTimes = rowSize - (rowRepeatTimes - 1) * MAX_REPEAT_TIMES;
            }
            Add(buffer2_, buffer1_[i * MAX_REPEAT_TIMES], buffer2_, mask, repeatTimes, binaryRepeatParams);
            Add(buffer4_, buffer3_[i * MAX_REPEAT_TIMES], buffer4_, mask, repeatTimes, binaryRepeatParams);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void copyOut(int64_t colIndex, int64_t colSize)
    {
        if constexpr (!IsSameType<T, float>::value) {
            Cast(buffer2_.ReinterpretCast<T>(), buffer2_, RoundMode::CAST_RINT, colSize);
            PipeBarrier<PIPE_V>();
            Cast(buffer4_.ReinterpretCast<T>(), buffer4_, RoundMode::CAST_RINT, colSize);
            PipeBarrier<PIPE_V>();
        }
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = colSize * sizeof(T);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        TEventID eventID = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
        SetFlag<HardEvent::V_MTE3>(eventID);
        WaitFlag<HardEvent::V_MTE3>(eventID);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventID);
        int64_t offset = colIndex * COL_TEMPLATE;
        DataCopyPad(pdGammaOutTensorGM_[offset], buffer2_.ReinterpretCast<T>(), intriParams);
        DataCopyPad(pdBetaOutTensorGM_[offset], buffer4_.ReinterpretCast<T>(), intriParams);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> queueGammaOut_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> queueBetaOut_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> queueGammaIn_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> queueBetaIn_;
    LocalTensor<float> buffer1_;
    LocalTensor<float> buffer2_;
    LocalTensor<float> buffer3_;
    LocalTensor<float> buffer4_;

    GlobalTensor<T> pdGammaOutTensorGM_;
    GlobalTensor<T> pdBetaOutTensorGM_;
    GlobalTensor<float> workspaceGM_;

    int64_t workspaceNum_;
    int64_t colAlignV_;
    int64_t row_;
    int64_t col_;
};
} // namespace AdaLayerNormGrad
#endif