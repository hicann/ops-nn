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
 * \brief GLU big shape kernel
 */

#ifndef GLU_BIG_SHAPE_H
#define GLU_BIG_SHAPE_H

#include "kernel_operator.h"
#include "glu_common.h"

#ifdef __CCE_AICORE__
#include "op_kernel/platform_util.h"
#endif

namespace Glu {
using namespace AscendC;
using namespace Common;
using namespace Ops::Base;

template <typename T>
class GluBigShape {
public:
    __aicore__ inline GluBigShape(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const GluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(const int64_t& batchIdx, const int64_t& splitIdx, const int64_t& length);
    __aicore__ inline void ComputeSigmoidAndMul(const int64_t& ub_num);
    __aicore__ inline void CopyOut(const int64_t& batchIdx, const int64_t& splitIdx, const int64_t& length);

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
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX1;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
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

    SetGlobalBufferForGlu(xGm, yGm, x, y);

    one_process_in_stride = ny * 2;
    one_process_out_stride = ny;

    pipe.InitBuffer(inQueueX1, BUFFER_NUM, BUFFER_SIZE * sizeof(float));
    pipe.InitBuffer(inQueueX2, BUFFER_NUM, BUFFER_SIZE * sizeof(float));
    pipe.InitBuffer(outQueue, BUFFER_NUM, BUFFER_SIZE * sizeof(float));
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
        ComputeSigmoidAndMul(length);
        CopyOut(batchIdx, splitIdx, length);
    }
}

template <typename T>
__aicore__ inline void GluBigShape<T>::CopyIn(
    const int64_t& batchIdx, const int64_t& splitIdx, const int64_t& length)
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

    int64_t offset_a = batchIdx * one_process_in_stride + splitIdx * this->splitSize;
    int64_t offset_b = batchIdx * one_process_in_stride + this->ny + splitIdx * this->splitSize;

    DataCopyPad(ubX1, xGm[offset_a], intriParams, intriPadParams);
    DataCopyPad(ubX2, xGm[offset_b], intriParams, intriPadParams);

    inQueueX1.EnQue(ubX1);
    inQueueX2.EnQue(ubX2);
}

template <typename T>
__aicore__ inline void GluBigShape<T>::ComputeSigmoidAndMul(const int64_t& count)
{
#ifdef __CCE_AICORE__
    LocalTensor<T> x1Local = inQueueX1.DeQue<T>();
    LocalTensor<T> x2Local = inQueueX2.DeQue<T>();
    LocalTensor<T> outLocal = outQueue.AllocTensor<T>();
    
    __local_mem__ T* x1LocalPtr = (__local_mem__ T*)x1Local.GetPhyAddr();
    __local_mem__ T* x2LocalPtr = (__local_mem__ T*)x2Local.GetPhyAddr();
    __local_mem__ T* outLocalPtr = (__local_mem__ T*)outLocal.GetPhyAddr();
    
    ComputeSigmoidAndMulImpl<T>(x1LocalPtr, x2LocalPtr, outLocalPtr, count);
    
    inQueueX1.FreeTensor(x1Local);
    inQueueX2.FreeTensor(x2Local);
    outQueue.EnQue(outLocal);
#endif
}

template <typename T>
__aicore__ inline void GluBigShape<T>::CopyOut(
    const int64_t& batchIdx, const int64_t& splitIdx, const int64_t& length)
{
    LocalTensor<T> outLocal = outQueue.DeQue<T>();
    
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;
    intriParams.blockLen = length * sizeof(T);

    int64_t offset = batchIdx * one_process_out_stride + splitIdx * this->splitSize;
    DataCopyPad(yGm[offset], outLocal, intriParams);
    outQueue.FreeTensor(outLocal);
}

} // namespace Glu
#endif // GLU_BIG_SHAPE_H
