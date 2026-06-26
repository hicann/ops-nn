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
 * \file swiglu_group_perf.h
 * \brief
 */

#ifndef SWIGLU_GROUP_PERF_H
#define SWIGLU_GROUP_PERF_H

#include "kernel_operator.h"
#include "swiglu_group_base.h"

namespace SwigluGroup {
using namespace AscendC;
template <typename T>
class SwigluGroupPerf {
public:
    __aicore__ inline SwigluGroupPerf()
    {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y,
        GM_ADDR workspace, const SwigluGroupTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;

        xGm.SetGlobalBuffer((__gm__ T*)x);
        yGm.SetGlobalBuffer((__gm__ T*)y);

        pipe->InitBufPool(tBufPool, tilingData->ubSize);
        ProcessGroupIndexTiling(groupIndex, tilingData, tBufPool, groupIndexQue, groupIndexSumBuf, groupIndexGm,
            groupSumLocal, hasGroupIndex_, usedCoreNums, rowOfFormerBlock, rowOfTailBlock, rowLoopOfFormerBlock,
            rowLoopOfTailBlock, tailRowFactorOfFormerBlock, tailRowFactorOfTailBlock);

        if (weight != nullptr) {
            hasWeight_ = true;
            weightGm.SetGlobalBuffer((__gm__ float*)weight);
            tBufPool.InitBuffer(weightQue, DOUBLE_BUFFER_NUM, RoundUp<float>(tilingData->rowFactor) * sizeof(float));
        }

        tBufPool.InitBuffer(
            x0Que, DOUBLE_BUFFER_NUM, tilingData->rowFactor * RoundUp<T>(tilingData->dFactor) * sizeof(T));
        tBufPool.InitBuffer(
            x1Que, DOUBLE_BUFFER_NUM, tilingData->rowFactor * RoundUp<T>(tilingData->dFactor) * sizeof(T));
        tBufPool.InitBuffer(
            yQue, DOUBLE_BUFFER_NUM, tilingData->rowFactor * RoundUp<T>(tilingData->dFactor) * sizeof(T));
        hasClampValue_ = (tilingData->hasClampLimit == 1);
        clampValue_ = tilingData->clampLimit;
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() >= usedCoreNums) {
            return;
        }
        int64_t curBlockIdx = GetBlockIdx();
        int64_t rowOuterLoop =
            (curBlockIdx == usedCoreNums - 1) ? rowLoopOfTailBlock : rowLoopOfFormerBlock;
        int64_t tailRowFactor = (curBlockIdx == usedCoreNums - 1) ? tailRowFactorOfTailBlock :
                                                                     tailRowFactorOfFormerBlock;
        int64_t x0GmBaseOffset = curBlockIdx * rowOfFormerBlock * tilingData->d;
        int64_t x1GmBaseOffset = x0GmBaseOffset + tilingData->splitD;
        int64_t yGmBaseOffset = curBlockIdx * rowOfFormerBlock * tilingData->splitD;
        int64_t weightGmBaseOffset = curBlockIdx * rowOfFormerBlock;
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx++) {
            int64_t curRowFactor = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : tilingData->rowFactor;
            if (hasWeight_) {
                weightLocal = weightQue.template AllocTensor<float>();
                CopyIn(weightGm[weightGmBaseOffset + rowOuterIdx * tilingData->rowFactor],
                    weightLocal, 1, curRowFactor);
                weightQue.template EnQue(weightLocal);
                weightLocal = weightQue.template DeQue<float>();
            }

            for (int64_t dLoopIdx = 0; dLoopIdx < tilingData->dLoop; dLoopIdx++) {
                int64_t curDFactor =
                    (dLoopIdx == tilingData->dLoop - 1) ? tilingData->tailDFactor : tilingData->dFactor;
                int64_t xBaseOffset =
                    rowOuterIdx * tilingData->rowFactor * tilingData->d + dLoopIdx * tilingData->dFactor;
                x0Local = x0Que.template AllocTensor<T>();
                CopyIn(
                    xGm[x0GmBaseOffset + xBaseOffset],
                    x0Local, curRowFactor, curDFactor, tilingData->d - curDFactor);
                x0Que.template EnQue(x0Local);
                x0Local = x0Que.template DeQue<T>();

                x1Local = x1Que.template AllocTensor<T>();
                CopyIn(
                    xGm[x1GmBaseOffset + xBaseOffset],
                    x1Local, curRowFactor, curDFactor, tilingData->d - curDFactor);
                x1Que.template EnQue(x1Local);
                x1Local = x1Que.template DeQue<T>();

                yLocal = yQue.template AllocTensor<T>();

                int32_t maskBit = (hasClampValue_ << 1) | hasWeight_;
                SwigluGroupDispatcher<T>(yLocal, x0Local, x1Local, weightLocal, clampValue_,
                    curRowFactor, curDFactor, maskBit);

                x0Que.template FreeTensor(x0Local);
                x1Que.template FreeTensor(x1Local);

                yQue.template EnQue(yLocal);
                yLocal = yQue.template DeQue<T>();
                CopyOut(yLocal, yGm[yGmBaseOffset + rowOuterIdx * tilingData->rowFactor *
                    tilingData->splitD + dLoopIdx * tilingData->dFactor],
                    curRowFactor, curDFactor, tilingData->splitD - curDFactor);
                yQue.template FreeTensor(yLocal);
            }
            if (hasWeight_) {
                weightQue.template FreeTensor(weightLocal);
            }
        }
    }

private:
    TPipe* pipe;
    const SwigluGroupTilingData* tilingData;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> weightGm;
    GlobalTensor<int64_t> groupIndexGm;

    TQue<QuePosition::VECIN, 1> x0Que;
    TQue<QuePosition::VECIN, 1> x1Que;
    TQue<QuePosition::VECOUT, 1> yQue;
    TQue<QuePosition::VECIN, 1> weightQue;

    TQue<QuePosition::VECIN, 1> groupIndexQue;
    TBuf<QuePosition::VECCALC> groupIndexSumBuf;
    TBufPool<QuePosition::VECCALC, 12> tBufPool;

    LocalTensor<T> x0Local;
    LocalTensor<T> x1Local;
    LocalTensor<T> yLocal;
    LocalTensor<float> weightLocal;

    LocalTensor<int64_t> groupIndexLocal;
    LocalTensor<int64_t> groupSumLocal;

    float clampValue_ = 448.0f;
    bool hasWeight_ = false;
    bool hasClampValue_ = false;

    bool hasGroupIndex_ = false;
    int64_t tailRowFactorOfTailBlock = 0;
    int64_t tailRowFactorOfFormerBlock = 0;
    int64_t rowLoopOfTailBlock = 0;
    int64_t rowLoopOfFormerBlock = 0;
    int64_t usedCoreNums = 0;
    int64_t rowOfFormerBlock = 0;
    int64_t rowOfTailBlock = 0;
};

} // namespace SwigluGroup

#endif
