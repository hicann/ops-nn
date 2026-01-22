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
 * \file glu_small_shape.h
 * \brief
 */
#ifndef GLU_SMALL_SHAPE_H
#define GLU_SMALL_SHAPE_H

#include "kernel_operator.h"

namespace Glu {
using namespace AscendC;

template <typename T>
class GluSmallShape
{
public:
    __aicore__ inline GluSmallShape(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const GluTilingData* tilingData);
    __aicore__ inline void Process();

    constexpr static int32_t bufferNum = 2;
    constexpr static int64_t bufferSize = 8 * 1024;
    constexpr static int32_t BLOCK_BYTES = 32;

private:
    __aicore__ inline void CopyIn(const int64_t& index, const int64_t& blockCount);
    __aicore__ inline void ComputeSigmoidAndMul(const int64_t& ub_num);
    __aicore__ inline void CopyOut(const int64_t& index, const int64_t& ub_num, const int64_t& group);
    __aicore__ inline void ProcessPerCore();
    __aicore__ inline void ProcessLastCore();

private:
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    int32_t blockIdx = 0;
    int64_t gmXOffset = 0;
    int64_t gmYOffset = 0;
    int64_t group_ub_num = 0;
    int64_t nlast_tail_ub_num = 0;
    int64_t last_tail_ub_num = 0;
    int64_t one_process_out_stride = 0;
    int64_t one_process_in_stride = 0;

    int64_t splitSize = 0;
    int64_t blockSize = 0;
    int64_t group = 0;
    int64_t realCoreNum = 0;
    int64_t loopNum = 0;
    int64_t tailLoopNum = 0;
    int64_t nLastTailGroup = 0;
    int64_t lastTailGroup = 0;

    bool isLastCore;

    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> inQueueX1;
    TQue<QuePosition::VECIN, bufferNum> inQueueX2;
    TQue<QuePosition::VECOUT, bufferNum> outQueue;

    TBuf<QuePosition::VECCALC> sigmoidTempBuf;
};

template <typename T>
__aicore__ inline void GluSmallShape<T>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const GluTilingData* tilingData)
{
    blockIdx = GetBlockIdx();

    splitSize = tilingData->splitSize;
    blockSize = tilingData->blockSize;
    group = tilingData->group;
    realCoreNum = tilingData->realCoreNum;
    loopNum = tilingData->loopNum;
    tailLoopNum = tilingData->tailLoopNum;
    nLastTailGroup = tilingData->nLastTailGroup;
    lastTailGroup = tilingData->lastTailGroup;

    gmXOffset = blockIdx * tilingData->numPerCore * splitSize * 2;
    gmYOffset = blockIdx * tilingData->numPerCore * splitSize;
    xGm.SetGlobalBuffer((__gm__ T*)x);
    yGm.SetGlobalBuffer((__gm__ T*)y);

    if (splitSize % blockSize == 0) {
        group_ub_num = group * splitSize;
        nlast_tail_ub_num = nLastTailGroup * splitSize;
        last_tail_ub_num = lastTailGroup * splitSize;
    } else {
        int64_t splitSizeAlign = (splitSize + blockSize - 1) / blockSize * blockSize;
        group_ub_num = group * splitSizeAlign;
        nlast_tail_ub_num = nLastTailGroup * splitSizeAlign;
        last_tail_ub_num = lastTailGroup * splitSizeAlign;
    }

    one_process_in_stride = group * splitSize * 2;
    one_process_out_stride = group * splitSize;

    isLastCore = (this->blockIdx == realCoreNum - 1) && (tailLoopNum != 0 || lastTailGroup != 0);
    
    pipe.InitBuffer(inQueueX1, bufferNum, bufferSize * sizeof(float));
    pipe.InitBuffer(inQueueX2, bufferNum, bufferSize * sizeof(float));
    pipe.InitBuffer(outQueue, 1, bufferSize * sizeof(float));

    pipe.InitBuffer(sigmoidTempBuf, bufferSize * sizeof(float));
}

template <typename T>
__aicore__ inline void GluSmallShape<T>::Process()
{
    if (this->blockIdx >= this->realCoreNum) {
        return;
    }

    if (this->isLastCore) { // process last core
        ProcessLastCore();
    } else {
        ProcessPerCore();
    }
}

template <typename T>
__aicore__ inline void GluSmallShape<T>::ProcessPerCore()
{
    // process core
    for (int64_t idx = 0; idx < this->loopNum; idx++) {
        CopyIn(idx, this->group);
        ComputeSigmoidAndMul(this->group_ub_num);
        CopyOut(idx, this->group_ub_num, this->group);
    }

    if (this->nLastTailGroup > 0) {
        CopyIn(this->loopNum, this->nLastTailGroup);
        ComputeSigmoidAndMul(this->nlast_tail_ub_num);
        CopyOut(this->loopNum, this->nlast_tail_ub_num, this->nLastTailGroup);
    }
}

template <typename T>
__aicore__ inline void GluSmallShape<T>::ProcessLastCore()
{
    for (int64_t idx = 0; idx < this->tailLoopNum; idx++) {
        CopyIn(idx, this->group);
        ComputeSigmoidAndMul(this->group_ub_num);
        CopyOut(idx, this->group_ub_num, this->group);
    }
    if (this->lastTailGroup > 0) {
        CopyIn(this->tailLoopNum, this->lastTailGroup);
        ComputeSigmoidAndMul(this->last_tail_ub_num);
        CopyOut(this->tailLoopNum, this->last_tail_ub_num, this->lastTailGroup);
    }
}

template <typename T>
__aicore__ inline void GluSmallShape<T>::CopyIn(const int64_t& index, const int64_t& blockCount)
{
    LocalTensor<T> ubX1 = inQueueX1.AllocTensor<T>();
    LocalTensor<T> ubX2 = inQueueX2.AllocTensor<T>();

    DataCopyParams intriParams;
    intriParams.blockCount = blockCount;
    intriParams.dstStride = 0;

    if (splitSize % blockSize == 0) {
        // align case
        intriParams.blockLen = splitSize / blockSize;
        intriParams.srcStride = splitSize / blockSize;
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            DataCopy(ubX1[bufferSize], xGm[gmXOffset + index * one_process_in_stride], intriParams);
            DataCopy(ubX2[bufferSize], xGm[gmXOffset + index * one_process_in_stride + splitSize], intriParams);
        } else {
            DataCopy(ubX1, xGm[gmXOffset + index * one_process_in_stride], intriParams);
            DataCopy(ubX2, xGm[gmXOffset + index * one_process_in_stride + splitSize], intriParams);
        }
    } else {
        // not align case
        intriParams.blockLen = splitSize * sizeof(T);
        intriParams.srcStride = splitSize * sizeof(T);
        DataCopyPadParams intriPadParams;
        intriPadParams.isPad = true;
        intriPadParams.leftPadding = 0;
        intriPadParams.rightPadding = BLOCK_BYTES / sizeof(T) - splitSize % blockSize;
        intriPadParams.paddingValue = 1;
        if constexpr (std::is_same_v<T, bfloat16_t>) {
            DataCopyPad(ubX1[bufferSize], xGm[gmXOffset + index * one_process_in_stride], intriParams, intriPadParams);
            DataCopyPad(ubX2[bufferSize], xGm[gmXOffset + index * one_process_in_stride + splitSize], intriParams, intriPadParams);
        } else {
            DataCopyPad(ubX1, xGm[gmXOffset + index * one_process_in_stride], intriParams, intriPadParams);
            DataCopyPad(ubX2, xGm[gmXOffset + index * one_process_in_stride + splitSize], intriParams, intriPadParams);
        }
    }

    inQueueX1.EnQue(ubX1);
    inQueueX2.EnQue(ubX2);
}

template <typename T>
__aicore__ inline void GluSmallShape<T>::ComputeSigmoidAndMul(const int64_t& ub_num)
{
    LocalTensor<T> ubX1 = inQueueX1.DeQue<T>();
    LocalTensor<T> ubX2 = inQueueX2.DeQue<T>();
    LocalTensor<T> out = outQueue.AllocTensor<T>();
    LocalTensor<uint8_t> tmpBuf = sigmoidTempBuf.Get<uint8_t>();

    if constexpr (std::is_same_v<T, bfloat16_t>) {
        LocalTensor<float> ubX1Fp32 = ubX1.template ReinterpretCast<float>();
        LocalTensor<float> ubX2Fp32 = ubX2.template ReinterpretCast<float>();
        LocalTensor<float> outFp32 = out.template ReinterpretCast<float>();

        Cast(ubX1Fp32, ubX1[bufferSize], RoundMode::CAST_NONE, ub_num);
        Cast(ubX2Fp32, ubX2[bufferSize], RoundMode::CAST_NONE, ub_num);
        PipeBarrier<PIPE_V>();

        Sigmoid(outFp32, ubX2Fp32, tmpBuf, ub_num);
        PipeBarrier<PIPE_V>();

        Mul(outFp32, outFp32, ubX1Fp32, ub_num);
        PipeBarrier<PIPE_V>();

        Cast(out, outFp32, RoundMode::CAST_RINT, ub_num);
        PipeBarrier<PIPE_V>();
    } else {
        Sigmoid(out, ubX2, tmpBuf, ub_num);
        PipeBarrier<PIPE_V>();

        Mul(out, out, ubX1, ub_num);
        PipeBarrier<PIPE_V>();
    }

    outQueue.EnQue(out);
    inQueueX1.FreeTensor(ubX1);
    inQueueX2.FreeTensor(ubX2);
}

template <typename T>
__aicore__ inline void GluSmallShape<T>::CopyOut(
    const int64_t& index, const int64_t& ub_num, const int64_t& group)
{
    LocalTensor<T> outLocal = outQueue.DeQue<T>();
    if (splitSize % blockSize == 0) {
        DataCopy(yGm[gmYOffset + index * one_process_out_stride], outLocal, ub_num);
    } else {
        DataCopyParams intriParams;
        intriParams.blockCount = group;
        intriParams.dstStride = 0;
        intriParams.srcStride = 0;
        intriParams.blockLen = splitSize * sizeof(T);
        DataCopyPad(yGm[gmYOffset + index * one_process_out_stride], outLocal, intriParams);
    }
    outQueue.FreeTensor(outLocal);
}
} // namespace Glu
#endif // GLU_SMALL_SHAPE_H