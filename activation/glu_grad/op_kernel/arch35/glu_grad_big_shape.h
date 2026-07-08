/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GLU_GRAD_BIG_SHAPE_H
#define GLU_GRAD_BIG_SHAPE_H

#include "kernel_operator.h"
#include "glu_grad_common.h"

#ifdef __CCE_AICORE__
#include "op_kernel/platform_util.h"
#endif

namespace GluGrad {
using namespace AscendC;
using namespace Common;
using namespace Ops::Base;

template <typename T>
class GluGradBigShape {
public:
    __aicore__ inline GluGradBigShape(){};
    __aicore__ inline void Init(
        GM_ADDR gradOut, GM_ADDR self, GM_ADDR out, GM_ADDR workspace, const GluGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(const int64_t& batchIdx, const int64_t& splitIdx, const int64_t& length);
    __aicore__ inline void ComputeGluGrad(const int64_t& count);
    __aicore__ inline void CopyOut(const int64_t& batchIdx, const int64_t& splitIdx, const int64_t& length);

private:
    GlobalTensor<T> gradOutGm;
    GlobalTensor<T> selfGm;
    GlobalTensor<T> outGm;
    int32_t blockIdx = 0;
    int64_t self_stride = 0;
    int64_t grad_stride = 0;

    int64_t splitSize = 0;
    int64_t group = 0;
    int64_t realCoreNum = 0;
    int64_t loopNum = 0;
    int64_t tailLoopNum = 0;
    int64_t ny = 0;
    int32_t bufLenPerBuf = 0;

    TPipe pipe;
    TQue<QuePosition::VECIN, NUM_ONE> inQueueA;
    TQue<QuePosition::VECIN, NUM_ONE> inQueueB;
    TQue<QuePosition::VECIN, NUM_ONE> inQueueGrad;
    TQue<QuePosition::VECOUT, NUM_ONE> outQueueA;
    TQue<QuePosition::VECOUT, NUM_ONE> outQueueB;


};

template <typename T>
__aicore__ inline void GluGradBigShape<T>::Init(
    GM_ADDR gradOut, GM_ADDR self, GM_ADDR out, GM_ADDR workspace, const GluGradTilingData* tilingData)
{
    blockIdx = GetBlockIdx();
    if (blockIdx >= tilingData->realCoreNum) {
        return;
    }

    splitSize = tilingData->splitSize;
    group = tilingData->group;
    realCoreNum = tilingData->realCoreNum;
    loopNum = tilingData->loopNum;
    tailLoopNum = tilingData->tailLoopNum;
    ny = tilingData->ny;

    SetGlobalBuffers(gradOutGm, selfGm, outGm, gradOut, self, out);

    self_stride = ny * 2;
    grad_stride = ny;

    bufLenPerBuf = static_cast<int32_t>(splitSize * sizeof(T));
    if (tailLoopNum > 0 && tailLoopNum > splitSize) {
        bufLenPerBuf = static_cast<int32_t>(tailLoopNum * sizeof(T));
    }

    pipe.InitBuffer(inQueueA, BUFFER_NUM, bufLenPerBuf);
    pipe.InitBuffer(inQueueB, BUFFER_NUM, bufLenPerBuf);
    pipe.InitBuffer(inQueueGrad, BUFFER_NUM, bufLenPerBuf);
    pipe.InitBuffer(outQueueA, BUFFER_NUM, bufLenPerBuf);
    pipe.InitBuffer(outQueueB, BUFFER_NUM, bufLenPerBuf);
}

template <typename T>
__aicore__ inline void GluGradBigShape<T>::Process()
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
        int64_t globalLoopIdx = this->realCoreNum * idx + this->blockIdx;
        int64_t batchIdx = 0;
        int64_t splitIdx = 0;
        int64_t length = 0;
        if (this->tailLoopNum != 0) {
            batchIdx = globalLoopIdx / (this->group + 1);
            splitIdx = globalLoopIdx % (this->group + 1);
            length = splitIdx == this->group ? this->tailLoopNum : this->splitSize;
        } else {
            batchIdx = globalLoopIdx / this->group;
            splitIdx = globalLoopIdx % this->group;
            length = this->splitSize;
        }

        CopyIn(batchIdx, splitIdx, length);
        ComputeGluGrad(length);
        CopyOut(batchIdx, splitIdx, length);
    }
}

template <typename T>
__aicore__ inline void GluGradBigShape<T>::CopyIn(
    const int64_t& batchIdx, const int64_t& splitIdx, const int64_t& length)
{
    LocalTensor<T> ubA = inQueueA.AllocTensor<T>();
    LocalTensor<T> ubB = inQueueB.AllocTensor<T>();
    LocalTensor<T> ubGrad = inQueueGrad.AllocTensor<T>();

    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    intriParams.blockLen = length * sizeof(T);

    DataCopyPadParams intriPadParams;
    intriPadParams.isPad = false;

    int64_t offset_a = batchIdx * self_stride + splitIdx * this->splitSize;
    int64_t offset_b = batchIdx * self_stride + this->ny + splitIdx * this->splitSize;
    int64_t offset_grad = batchIdx * grad_stride + splitIdx * this->splitSize;

    DataCopyPad(ubA, selfGm[offset_a], intriParams, intriPadParams);
    DataCopyPad(ubB, selfGm[offset_b], intriParams, intriPadParams);
    DataCopyPad(ubGrad, gradOutGm[offset_grad], intriParams, intriPadParams);

    inQueueA.EnQue(ubA);
    inQueueB.EnQue(ubB);
    inQueueGrad.EnQue(ubGrad);
}

template <typename T>
__aicore__ inline void GluGradBigShape<T>::ComputeGluGrad(const int64_t& count)
{
    LocalTensor<T> aLocal = inQueueA.DeQue<T>();
    LocalTensor<T> bLocal = inQueueB.DeQue<T>();
    LocalTensor<T> gradLocal = inQueueGrad.DeQue<T>();
    LocalTensor<T> outALocal = outQueueA.AllocTensor<T>();
    LocalTensor<T> outBLocal = outQueueB.AllocTensor<T>();

#ifdef __CCE_AICORE__
    __local_mem__ T* aLocalPtr = (__local_mem__ T*)aLocal.GetPhyAddr();
    __local_mem__ T* bLocalPtr = (__local_mem__ T*)bLocal.GetPhyAddr();
    __local_mem__ T* gradLocalPtr = (__local_mem__ T*)gradLocal.GetPhyAddr();
    __local_mem__ T* outALocalPtr = (__local_mem__ T*)outALocal.GetPhyAddr();
    __local_mem__ T* outBLocalPtr = (__local_mem__ T*)outBLocal.GetPhyAddr();

    ComputeGluGradImpl<T>(aLocalPtr, bLocalPtr, gradLocalPtr, outALocalPtr, outBLocalPtr, count);
#endif

    inQueueA.FreeTensor(aLocal);
    inQueueB.FreeTensor(bLocal);
    inQueueGrad.FreeTensor(gradLocal);
    outQueueA.EnQue(outALocal);
    outQueueB.EnQue(outBLocal);
}

template <typename T>
__aicore__ inline void GluGradBigShape<T>::CopyOut(
    const int64_t& batchIdx, const int64_t& splitIdx, const int64_t& length)
{
    LocalTensor<T> outALocal = outQueueA.DeQue<T>();
    LocalTensor<T> outBLocal = outQueueB.DeQue<T>();

    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    intriParams.blockLen = length * sizeof(T);

    int64_t offset_a = batchIdx * self_stride + splitIdx * this->splitSize;
    int64_t offset_b = batchIdx * self_stride + this->ny + splitIdx * this->splitSize;

    DataCopyPad(outGm[offset_a], outALocal, intriParams);
    DataCopyPad(outGm[offset_b], outBLocal, intriParams);

    outQueueA.FreeTensor(outALocal);
    outQueueB.FreeTensor(outBLocal);
}

} // namespace GluGrad
#endif // GLU_GRAD_BIG_SHAPE_H
