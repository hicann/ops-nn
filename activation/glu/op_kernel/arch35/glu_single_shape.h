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
 * \brief GLU single shape kernel
 */

#ifndef GLU_SINGLE_SHAPE_H
#define GLU_SINGLE_SHAPE_H

#include "kernel_operator.h"
#include "glu_common.h"

#ifdef __CCE_AICORE__
#include "op_kernel/platform_util.h"
#endif

namespace Glu {
using namespace Common;
using namespace Ops::Base;
using namespace AscendC;

template <typename T>
class GluSingleShape {
public:
    __aicore__ inline GluSingleShape(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const GluTilingData* tilingData);
    __aicore__ inline void Process();

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
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
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

    SetGlobalBufferForGlu(xGm, yGm, x, y);
    
    gmXOffset = blockIdx * tilingData->numPerCore * splitSize * 2;
    gmYOffset = blockIdx * tilingData->numPerCore * splitSize;

    one_process_in_stride = group * splitSize * 2;
    one_process_out_stride = group * splitSize;

    group_ub_num = group * splitSize;
    nlast_tail_ub_num = nLastTailGroup * splitSize;
    last_tail_ub_num = lastTailGroup * splitSize;

    isLastCore = (this->blockIdx == realCoreNum - 1) && (tailLoopNum != 0 || lastTailGroup != 0);

    pipe.InitBuffer(inQueueX, BUFFER_NUM, 2 * BUFFER_SIZE * sizeof(float));
    pipe.InitBuffer(outQueue, BUFFER_NUM, BUFFER_SIZE * sizeof(float));
}

template <typename T>
__aicore__ inline void GluSingleShape<T>::Process()
{
    if (this->blockIdx >= this->realCoreNum) {
        return;
    }

    if (!this->isLastCore) {
        ProcessPerCore();
    } else {
        ProcessLastCore();
    }
}

template <typename T>
__aicore__ inline void GluSingleShape<T>::ProcessPerCore()
{
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
    
    DataCopyExtParams intriParams;
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    intriParams.blockLen = one_process_total_num * sizeof(T);
    
    DataCopyPadExtParams<T> intriPadParams{false, 0, 0, 0};
    DataCopyPad(ubX, xGm[gmXOffset + index * one_process_in_stride], intriParams, intriPadParams);
    inQueueX.EnQue(ubX);
}

template <typename T>
__aicore__ inline void GluSingleShape<T>::ComputeSigmoidAndMul(const int64_t& count)
{
#ifdef __CCE_AICORE__
    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
    
    __local_mem__ T* xLocalPtr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ T* outLocalPtr = (__local_mem__ T*)outLocal.GetPhyAddr();

    ComputeSigmoidAndMulWithDeInterleave<T>(xLocalPtr, outLocalPtr, count);
    
    inQueueX.FreeTensor(xLocal);
    outQueue.EnQue(outLocal);
#endif
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