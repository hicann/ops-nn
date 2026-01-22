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
 * \file glu_big_shape.h
 * \brief
 */
#ifndef GLU_BIG_SHAPE_H
#define GLU_BIG_SHAPE_H

#include "kernel_operator.h"

namespace Glu {
using namespace AscendC;

template <typename T>
class GluBigShape
{
public:
    __aicore__ inline GluBigShape(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const GluTilingData* tilingData);
    __aicore__ inline void Process();

    constexpr static int32_t bufferNum = 2;
    constexpr static int64_t bufferSize = 8 * 1024;

private:
    __aicore__ inline void CopyIn(const int64_t& idx_x, const int64_t& idx_y, const int64_t& length);
    __aicore__ inline void ComputeSigmoidAndMul(const int64_t& ub_num);
    __aicore__ inline void CopyOut(const int64_t& idx_x, const int64_t& idx_y, const int64_t& length);

private:
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    int32_t blockIdx = 0;
    int64_t one_process_out_stride = 0;
    int64_t one_process_in_stride = 0;

    int64_t splitSize = 0;
    int64_t blockSize = 0;
    int64_t group = 0;
    int64_t realCoreNum = 0;
    int64_t loopNum = 0;
    int64_t tailLoopNum = 0;
    int64_t ny = 0;

    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> inQueueX1;
    TQue<QuePosition::VECIN, bufferNum> inQueueX2;
    TQue<QuePosition::VECOUT, bufferNum> outQueue;

    TBuf<QuePosition::VECCALC> sigmoidTempBuf;
};

template <typename T>
__aicore__ inline void GluBigShape<T>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const GluTilingData* tilingData)
{
    blockIdx = GetBlockIdx();

    splitSize = tilingData->splitSize;
    blockSize = tilingData->blockSize;
    group = tilingData->group;
    realCoreNum = tilingData->realCoreNum;
    loopNum = tilingData->loopNum;
    tailLoopNum = tilingData->tailLoopNum;
    ny = tilingData->ny;

    xGm.SetGlobalBuffer((__gm__ T*)x);
    yGm.SetGlobalBuffer((__gm__ T*)y);

    one_process_in_stride = ny * 2;
    one_process_out_stride = ny;
    
    pipe.InitBuffer(inQueueX1, bufferNum, bufferSize * sizeof(float));
    pipe.InitBuffer(inQueueX2, bufferNum, bufferSize * sizeof(float));
    pipe.InitBuffer(outQueue, 1, bufferSize * sizeof(float));

    pipe.InitBuffer(sigmoidTempBuf, bufferSize * sizeof(float));
}

template <typename T>
__aicore__ inline void GluBigShape<T>::Process()
{
    if (this->blockIdx >= this->realCoreNum) {
        return;
    }

    int64_t actualLoopNum = (this->loopNum + this->realCoreNum - 1) / this->realCoreNum;
    int64_t tailLoopIdx = this->loopNum % this->realCoreNum;

    for (int64_t idx = 0; idx < actualLoopNum; idx++) {
        if (tailLoopIdx != 0 && idx == actualLoopNum - 1) {
            if (this->blockIdx >= tailLoopIdx) {
                return;
            }
        }
        int64_t z = this->realCoreNum * idx + this->blockIdx;
        int64_t idx_x = 0;
        int64_t idx_y = 0;
        int64_t length = 0;
        if (this->tailLoopNum != 0) {
            idx_x = z / (this->group + 1);
            idx_y = z % (this->group + 1);
            length = idx_y == this->group ? this->tailLoopNum : this->splitSize;
        } else {
            idx_x = z / this->group;
            idx_y = z % this->group;
            length = this->splitSize;
        }

        CopyIn(idx_x, idx_y, length);
        ComputeSigmoidAndMul(length);
        CopyOut(idx_x, idx_y, length);
    }
}

template <typename T>
__aicore__ inline void GluBigShape<T>::CopyIn(
    const int64_t& idx_x, const int64_t& idx_y, const int64_t& length)
{
    LocalTensor<T> ubX1 = inQueueX1.AllocTensor<T>();
    LocalTensor<T> ubX2 = inQueueX2.AllocTensor<T>();
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    intriParams.blockLen = length * sizeof(T);
    DataCopyPadParams intriPadParams;
    intriPadParams.isPad = false;

    int64_t offset_a = idx_x * one_process_in_stride + idx_y * this->splitSize;
    int64_t offset_b = idx_x * one_process_in_stride + this->ny + idx_y * this->splitSize;

    if constexpr (std::is_same_v<T, bfloat16_t>) {
        DataCopyPad(ubX1[bufferSize], xGm[offset_a], intriParams, intriPadParams);
        DataCopyPad(ubX2[bufferSize], xGm[offset_b], intriParams, intriPadParams);
    } else {
        DataCopyPad(ubX1, xGm[offset_a], intriParams, intriPadParams);
        DataCopyPad(ubX2, xGm[offset_b], intriParams, intriPadParams);
    }

    inQueueX1.EnQue(ubX1);
    inQueueX2.EnQue(ubX2);
}

template <typename T>
__aicore__ inline void GluBigShape<T>::ComputeSigmoidAndMul(const int64_t& length)
{
    LocalTensor<T> ubX1 = inQueueX1.DeQue<T>();
    LocalTensor<T> ubX2 = inQueueX2.DeQue<T>();
    LocalTensor<T> out = outQueue.AllocTensor<T>();
    LocalTensor<uint8_t> tmpBuf = sigmoidTempBuf.Get<uint8_t>();

    if constexpr (std::is_same_v<T, bfloat16_t>) {
        LocalTensor<float> ubX1Fp32 = ubX1.template ReinterpretCast<float>();
        LocalTensor<float> ubX2Fp32 = ubX2.template ReinterpretCast<float>();
        LocalTensor<float> outFp32 = out.template ReinterpretCast<float>();

        Cast(ubX1Fp32, ubX1[bufferSize], RoundMode::CAST_NONE, length);
        Cast(ubX2Fp32, ubX2[bufferSize], RoundMode::CAST_NONE, length);
        PipeBarrier<PIPE_V>();

        Sigmoid(outFp32, ubX2Fp32, tmpBuf, length);
        PipeBarrier<PIPE_V>();

        Mul(outFp32, outFp32, ubX1Fp32, length);
        PipeBarrier<PIPE_V>();

        Cast(out, outFp32, RoundMode::CAST_RINT, length);
        PipeBarrier<PIPE_V>();
    } else {
        Sigmoid(out, ubX2, tmpBuf, length);
        PipeBarrier<PIPE_V>();

        Mul(out, out, ubX1, length);
        PipeBarrier<PIPE_V>();
    }

    outQueue.EnQue(out);
    inQueueX1.FreeTensor(ubX1);
    inQueueX2.FreeTensor(ubX2);
}

template <typename T>
__aicore__ inline void GluBigShape<T>::CopyOut(
    const int64_t& idx_x, const int64_t& idx_y, const int64_t& length)
{
    LocalTensor<T> outLocal = outQueue.DeQue<T>();
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    intriParams.blockLen = length * sizeof(T);

    int64_t offset = idx_x * one_process_out_stride + idx_y * this->splitSize;
    DataCopyPad(yGm[offset], outLocal, intriParams);
    outQueue.FreeTensor(outLocal);
}
} // namespace Glu
#endif // GLU_BIG_SHAPE_H