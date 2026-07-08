/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GLU_GRAD_SMALL_SHAPE_H
#define GLU_GRAD_SMALL_SHAPE_H

#include "glu_grad_common.h"
#include "kernel_operator.h"

#ifdef __CCE_AICORE__
#include "op_kernel/platform_util.h"
#endif

namespace GluGrad {
using namespace AscendC;
using namespace Ops::Base;
using namespace Common;

template <typename T>
class GluGradSmallShape {
public:
    __aicore__ inline GluGradSmallShape(){};
    __aicore__ inline void Init(
        GM_ADDR gradOut, GM_ADDR self, GM_ADDR out, GM_ADDR workspace, const GluGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(const int64_t& index, const int64_t& blockCount);
    __aicore__ inline void ComputeGluGrad(const int64_t& ub_num);
    __aicore__ inline void CopyOut(const int64_t& index, const int64_t& ub_num, const int64_t& group);
    __aicore__ inline void ProcessPerCore();
    __aicore__ inline void ProcessLastCore();

private:
    GlobalTensor<T> gradOutGm;
    GlobalTensor<T> selfGm;
    GlobalTensor<T> outGm;
    int32_t blockIdx = 0;
    int64_t gmSelfOffset = 0;
    int64_t gmGradOffset = 0;
    int64_t gmOutOffset = 0;
    int64_t group_ub_num = 0;
    int64_t nlast_tail_ub_num = 0;
    int64_t last_tail_ub_num = 0;
    int64_t self_stride = 0;
    int64_t grad_stride = 0;

    int64_t splitSize = 0;
    int64_t blockSize = 0;
    int64_t group = 0;
    int64_t realCoreNum = 0;
    int64_t loopNum = 0;
    int64_t tailLoopNum = 0;
    int64_t nLastTailGroup = 0;
    int64_t lastTailGroup = 0;
    int32_t bufLenPerBuf = 0;

    bool isLastCore = false;

    TPipe pipe;
    TQue<QuePosition::VECIN, NUM_ONE> inQueueA;
    TQue<QuePosition::VECIN, NUM_ONE> inQueueB;
    TQue<QuePosition::VECIN, NUM_ONE> inQueueGrad;
    TQue<QuePosition::VECOUT, NUM_ONE> outQueueA;
    TQue<QuePosition::VECOUT, NUM_ONE> outQueueB;
};

template <typename T>
__aicore__ inline void GluGradSmallShape<T>::Init(
    GM_ADDR gradOut, GM_ADDR self, GM_ADDR out, GM_ADDR workspace, const GluGradTilingData* tilingData)
{
    blockIdx = GetBlockIdx();
    if (blockIdx >= tilingData->realCoreNum) {
        return;
    }

    splitSize = tilingData->splitSize;
    blockSize = tilingData->blockSize;
    group = tilingData->group;
    realCoreNum = tilingData->realCoreNum;
    loopNum = tilingData->loopNum;
    tailLoopNum = tilingData->tailLoopNum;
    nLastTailGroup = tilingData->nLastTailGroup;
    lastTailGroup = tilingData->lastTailGroup;

    gmSelfOffset = blockIdx * tilingData->numPerCore * splitSize * 2;
    gmGradOffset = blockIdx * tilingData->numPerCore * splitSize;
    gmOutOffset = blockIdx * tilingData->numPerCore * splitSize * 2;

    SetGlobalBuffers(gradOutGm, selfGm, outGm, gradOut, self, out);

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

    self_stride = group * splitSize * 2;
    grad_stride = group * splitSize;

    bufLenPerBuf = static_cast<int32_t>(group_ub_num * sizeof(T));
    if (nlast_tail_ub_num > group_ub_num) {
        bufLenPerBuf = static_cast<int32_t>(nlast_tail_ub_num * sizeof(T));
    }
    if (last_tail_ub_num > group_ub_num) {
        bufLenPerBuf = static_cast<int32_t>(last_tail_ub_num * sizeof(T));
    }

    isLastCore = (this->blockIdx == realCoreNum - 1) && (tailLoopNum != 0 || lastTailGroup != 0);

    pipe.InitBuffer(inQueueA, BUFFER_NUM, bufLenPerBuf);
    pipe.InitBuffer(inQueueB, BUFFER_NUM, bufLenPerBuf);
    pipe.InitBuffer(inQueueGrad, BUFFER_NUM, bufLenPerBuf);
    pipe.InitBuffer(outQueueA, BUFFER_NUM, bufLenPerBuf);
    pipe.InitBuffer(outQueueB, BUFFER_NUM, bufLenPerBuf);
}

template <typename T>
__aicore__ inline void GluGradSmallShape<T>::Process()
{
    if (this->blockIdx >= this->realCoreNum) {
        return;
    }

    if (this->isLastCore) {
        ProcessLastCore();
    } else {
        ProcessPerCore();
    }
}

template <typename T>
__aicore__ inline void GluGradSmallShape<T>::ProcessPerCore()
{
    for (int64_t idx = 0; idx < this->loopNum; idx++) {
        CopyIn(idx, this->group);
        ComputeGluGrad(this->group_ub_num);
        CopyOut(idx, this->group_ub_num, this->group);
    }

    if (this->nLastTailGroup > 0) {
        CopyIn(this->loopNum, this->nLastTailGroup);
        ComputeGluGrad(this->nlast_tail_ub_num);
        CopyOut(this->loopNum, this->nlast_tail_ub_num, this->nLastTailGroup);
    }
}

template <typename T>
__aicore__ inline void GluGradSmallShape<T>::ProcessLastCore()
{
    for (int64_t idx = 0; idx < this->tailLoopNum; idx++) {
        CopyIn(idx, this->group);
        ComputeGluGrad(this->group_ub_num);
        CopyOut(idx, this->group_ub_num, this->group);
    }
    if (this->lastTailGroup > 0) {
        CopyIn(this->tailLoopNum, this->lastTailGroup);
        ComputeGluGrad(this->last_tail_ub_num);
        CopyOut(this->tailLoopNum, this->last_tail_ub_num, this->lastTailGroup);
    }
}

template <typename T>
__aicore__ inline void GluGradSmallShape<T>::CopyIn(const int64_t& index, const int64_t& blockCount)
{
    LocalTensor<T> ubA = inQueueA.AllocTensor<T>();
    LocalTensor<T> ubB = inQueueB.AllocTensor<T>();
    LocalTensor<T> ubGrad = inQueueGrad.AllocTensor<T>();

    DataCopyParams intriParams;
    intriParams.blockCount = blockCount;
    intriParams.dstStride = 0;

    if (splitSize % blockSize == 0) {
        intriParams.blockLen = splitSize / blockSize;
        intriParams.srcStride = splitSize / blockSize;
        DataCopy(ubA, selfGm[gmSelfOffset + index * self_stride], intriParams);
        DataCopy(ubB, selfGm[gmSelfOffset + index * self_stride + splitSize], intriParams);
        intriParams.srcStride = 0;
        DataCopy(ubGrad, gradOutGm[gmGradOffset + index * grad_stride], intriParams);
    } else {
        intriParams.blockLen = splitSize * sizeof(T);
        intriParams.srcStride = splitSize * sizeof(T);
        DataCopyPadParams intriPadParams;
        intriPadParams.isPad = true;
        intriPadParams.leftPadding = 0;
        intriPadParams.rightPadding = BLOCK_BYTES / sizeof(T) - splitSize % blockSize;
        intriPadParams.paddingValue = 1;
        DataCopyPad(ubA, selfGm[gmSelfOffset + index * self_stride], intriParams, intriPadParams);
        DataCopyPad(ubB, selfGm[gmSelfOffset + index * self_stride + splitSize], intriParams, intriPadParams);
        intriParams.srcStride = 0;
        DataCopyPad(ubGrad, gradOutGm[gmGradOffset + index * grad_stride], intriParams, intriPadParams);
    }

    inQueueA.EnQue(ubA);
    inQueueB.EnQue(ubB);
    inQueueGrad.EnQue(ubGrad);
}

template <typename T>
__aicore__ inline void GluGradSmallShape<T>::ComputeGluGrad(const int64_t& count)
{
    LocalTensor<T> aLocal = inQueueA.DeQue<T>();
    LocalTensor<T> bLocal = inQueueB.DeQue<T>();
    LocalTensor<T> gradLocal = inQueueGrad.DeQue<T>();
    LocalTensor<T> outALocal = outQueueA.AllocTensor<T>();
    LocalTensor<T> outBLocal = outQueueB.AllocTensor<T>();

    __local_mem__ T* aLocalPtr = (__local_mem__ T*)aLocal.GetPhyAddr();
    __local_mem__ T* bLocalPtr = (__local_mem__ T*)bLocal.GetPhyAddr();
    __local_mem__ T* gradLocalPtr = (__local_mem__ T*)gradLocal.GetPhyAddr();
    __local_mem__ T* outALocalPtr = (__local_mem__ T*)outALocal.GetPhyAddr();
    __local_mem__ T* outBLocalPtr = (__local_mem__ T*)outBLocal.GetPhyAddr();

    ComputeGluGradImpl<T>(aLocalPtr, bLocalPtr, gradLocalPtr, outALocalPtr, outBLocalPtr, count);

    inQueueA.FreeTensor(aLocal);
    inQueueB.FreeTensor(bLocal);
    inQueueGrad.FreeTensor(gradLocal);
    outQueueA.EnQue(outALocal);
    outQueueB.EnQue(outBLocal);
}

template <typename T>
__aicore__ inline void GluGradSmallShape<T>::CopyOut(
    const int64_t& index, const int64_t& ub_num, const int64_t& group)
{
    LocalTensor<T> outALocal = outQueueA.DeQue<T>();
    LocalTensor<T> outBLocal = outQueueB.DeQue<T>();

    if (splitSize % blockSize == 0) {
        DataCopyParams outParams;
        outParams.blockCount = group;
        outParams.blockLen = splitSize / blockSize;
        outParams.srcStride = 0;
        outParams.dstStride = splitSize / blockSize;
        DataCopy(outGm[gmOutOffset + index * self_stride], outALocal, outParams);
        DataCopy(outGm[gmOutOffset + index * self_stride + splitSize], outBLocal, outParams);
    } else {
        DataCopyParams intriParams;
        intriParams.blockCount = group;
        intriParams.dstStride = splitSize * sizeof(T);
        intriParams.srcStride = 0;
        intriParams.blockLen = splitSize * sizeof(T);
        DataCopyPad(outGm[gmOutOffset + index * self_stride], outALocal, intriParams);
        DataCopyPad(outGm[gmOutOffset + index * self_stride + splitSize], outBLocal, intriParams);
    }

    outQueueA.FreeTensor(outALocal);
    outQueueB.FreeTensor(outBLocal);
}

} // namespace GluGrad
#endif // GLU_GRAD_SMALL_SHAPE_H
