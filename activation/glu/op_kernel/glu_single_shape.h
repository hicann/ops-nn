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
 * \file glu_single_shape.h
 * \brief
 */
#ifndef GLU_SINGLE_SHAPE_H
#define GLU_SINGLE_SHAPE_H

#include "kernel_operator.h"

namespace Glu {
using namespace AscendC;

template <typename T>
class GluSingleShape
{
public:
    __aicore__ inline GluSingleShape(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const GluTilingData* tilingData);
    __aicore__ inline void Process();

    constexpr static int32_t bufferNum = 2;
    constexpr static int64_t bufferSize = 6 * 1024;

private:
    __aicore__ inline void CopyIn(const int64_t& index, const int64_t& blockCount);
    __aicore__ inline void ComputeSigmoidAndMul(const int64_t& count);
    __aicore__ inline void CopyOut(const int64_t& index, const int64_t& count, const int64_t& group);
    __aicore__ inline void ProcessPerCore();
    __aicore__ inline void ProcessLastCore();

private:
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    int32_t blockIdx = 0;
    int64_t gmXOffset = 0;
    int64_t gmYOffset = 0;
    int64_t one_process_out_stride = 0;
    int64_t one_process_in_stride = 0;
    int64_t group_ub_num = 0;
    int64_t nlast_tail_ub_num = 0;
    int64_t last_tail_ub_num = 0;

    int64_t splitSize = 0;
    int64_t blockSize = 0;
    int64_t group = 0;
    int64_t realCoreNum = 0;
    int64_t nLastTailGroup = 0;
    int64_t lastTailGroup = 0;
    int64_t loopNum = 0;
    int64_t tailLoopNum = 0;

    bool isLastCore;

    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> inQueueX;
    TQue<QuePosition::VECOUT, bufferNum> outQueue;

    TBuf<QuePosition::VECCALC> resultTempBuf1;
    TBuf<QuePosition::VECCALC> resultTempBuf2;
    TBuf<QuePosition::VECCALC> sigmoidTempBuf;
};

template <typename T>
__aicore__ inline void GluSingleShape<T>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const GluTilingData* tilingData)
{
    blockIdx = GetBlockIdx();

    splitSize = tilingData->splitSize;
    blockSize = tilingData->blockSize;
    group = tilingData->group;
    realCoreNum = tilingData->realCoreNum;
    nLastTailGroup = tilingData->nLastTailGroup;
    lastTailGroup = tilingData->lastTailGroup;
    loopNum = tilingData->loopNum;
    tailLoopNum = tilingData->tailLoopNum;

    xGm.SetGlobalBuffer((__gm__ T*)x);
    yGm.SetGlobalBuffer((__gm__ T*)y);
    gmXOffset = blockIdx * tilingData->numPerCore * splitSize * 2;
    gmYOffset = blockIdx * tilingData->numPerCore * splitSize;

    one_process_in_stride = group * splitSize * 2;
    one_process_out_stride = group * splitSize;

    group_ub_num = group * splitSize;
    nlast_tail_ub_num = nLastTailGroup * splitSize;
    last_tail_ub_num = lastTailGroup * splitSize;

    isLastCore = (this->blockIdx == realCoreNum - 1) && (tailLoopNum != 0 || lastTailGroup != 0);

    pipe.InitBuffer(inQueueX, bufferNum, 2 * bufferSize * sizeof(float));
    pipe.InitBuffer(outQueue, 1, bufferSize * sizeof(float));

    pipe.InitBuffer(resultTempBuf1, bufferSize * sizeof(float));
    pipe.InitBuffer(resultTempBuf2, bufferSize * sizeof(float));
    pipe.InitBuffer(sigmoidTempBuf, bufferSize * sizeof(float));
}

template <typename T>
__aicore__ inline void GluSingleShape<T>::Process()
{
    if (this->blockIdx >= this->realCoreNum) {
        return;
    }

    if (!this->isLastCore) { // process last core
        ProcessPerCore();
    } else {
        ProcessLastCore();
    }
}

template <typename T>
__aicore__ inline void GluSingleShape<T>::ProcessPerCore()
{
    // process core
    for (int64_t i = 0; i < this->loopNum; i++) {
        CopyIn(i, this->group);
        ComputeSigmoidAndMul(this->group_ub_num);
        CopyOut(i, this->group_ub_num, this->group);
    }

    if (this->nLastTailGroup > 0) {
        CopyIn(this->loopNum, this->nLastTailGroup);
        ComputeSigmoidAndMul(this->nlast_tail_ub_num);
        CopyOut(this->loopNum, this->nlast_tail_ub_num, this->nLastTailGroup);
    }
}

template <typename T>
__aicore__ inline void GluSingleShape<T>::ProcessLastCore()
{
    for (int64_t i = 0; i < this->tailLoopNum; i++) {
        CopyIn(i, this->group);
        ComputeSigmoidAndMul(this->group_ub_num);
        CopyOut(i, this->group_ub_num, this->group);
    }
    if (this->lastTailGroup > 0) {
        CopyIn(this->tailLoopNum, this->lastTailGroup);
        ComputeSigmoidAndMul(this->last_tail_ub_num);
        CopyOut(this->tailLoopNum, this->last_tail_ub_num, this->lastTailGroup);
    }
}

template <typename T>
__aicore__ inline void GluSingleShape<T>::CopyIn(const int64_t& index, const int64_t& blockCount)
{
    LocalTensor<T> ubX = inQueueX.AllocTensor<T>();
    int64_t one_process_total_num = blockCount * this->splitSize * 2;
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    intriParams.blockLen = one_process_total_num * sizeof(T);
    DataCopyPadParams intriPadParams;
    intriPadParams.isPad = false;
    DataCopyPad(ubX, xGm[gmXOffset + index * one_process_in_stride], intriParams, intriPadParams);
    inQueueX.EnQue(ubX);
}

template <typename T>
__aicore__ inline void GluSingleShape<T>::ComputeSigmoidAndMul(const int64_t& count)
{
    LocalTensor<T> ubX = inQueueX.DeQue<T>();
    LocalTensor<T> ubX1 = resultTempBuf1.Get<T>();
    LocalTensor<T> ubX2 = resultTempBuf2.Get<T>();
    LocalTensor<T> out = outQueue.AllocTensor<T>();
    LocalTensor<uint8_t> tmpBuf = sigmoidTempBuf.Get<uint8_t>();

    uint64_t rsvdCnt = 0;
    uint64_t total_vreduce_num = count * 2;
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        uint16_t repeat = (total_vreduce_num + 63) / 64;
        GatherMask(ubX1[bufferSize], ubX, 1, false, 0, {1, repeat, 8, 0}, rsvdCnt);
        GatherMask(ubX2[bufferSize], ubX, 2, false, 0, {1, repeat, 8, 0}, rsvdCnt);
        PipeBarrier<PIPE_V>();
        inQueueX.FreeTensor(ubX);

        LocalTensor<float> ubX1Fp32 = ubX1.template ReinterpretCast<float>();
        LocalTensor<float> ubX2Fp32 = ubX2.template ReinterpretCast<float>();
        LocalTensor<float> outFp32 = out.template ReinterpretCast<float>();

        Cast(ubX1Fp32, ubX1[bufferSize], RoundMode::CAST_NONE, count);
        Cast(ubX2Fp32, ubX2[bufferSize], RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();

        Sigmoid(outFp32, ubX2Fp32, tmpBuf, count);
        PipeBarrier<PIPE_V>();

        Mul(outFp32, outFp32, ubX1Fp32, count);
        PipeBarrier<PIPE_V>();

        Cast(out, outFp32, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
    } else {
        uint16_t repeat = (total_vreduce_num + 63) / 64;
        GatherMask(ubX1, ubX, 1, false, 0, {1, repeat, 8, 0}, rsvdCnt);
        GatherMask(ubX2, ubX, 2, false, 0, {1, repeat, 8, 0}, rsvdCnt);
        PipeBarrier<PIPE_V>();
        inQueueX.FreeTensor(ubX);

        Sigmoid(out, ubX2, tmpBuf, count);
        PipeBarrier<PIPE_V>();

        Mul(out, out, ubX1, count);
        PipeBarrier<PIPE_V>();
    }
    outQueue.EnQue(out);
}

template <typename T>
__aicore__ inline void GluSingleShape<T>::CopyOut(
    const int64_t& index, const int64_t& count, const int64_t& group)
{
    LocalTensor<T> outLocal = outQueue.DeQue<T>();
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    intriParams.blockLen = count * sizeof(T);
    DataCopyPad(yGm[gmYOffset + index * one_process_out_stride], outLocal, intriParams);
    outQueue.FreeTensor(outLocal);
}
} // namespace Glu
#endif // GLU_SINGLE_SHAPE_H