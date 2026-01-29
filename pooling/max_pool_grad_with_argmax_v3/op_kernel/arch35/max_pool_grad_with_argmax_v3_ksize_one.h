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
 * \file max_pool_grad_with_argmax_v3_ksize_one.h
 * \brief
 */

#ifndef MAX_POOL_GRAD_WITH_ARGMAX_V3_KSIZE_ONE_H_
#define MAX_POOL_GRAD_WITH_ARGMAX_V3_KSIZE_ONE_H_

#include "kernel_operator.h"

namespace MaxPoolGradWithArgmaxV3KsizeOneNameSpace {
using namespace AscendC;

constexpr int64_t DB_BUFFER = 2;

template <typename T>
class MaxPoolGradWithArgmaxV3KsizeOne {
public:
    __aicore__ inline MaxPoolGradWithArgmaxV3KsizeOne(
        const MaxPoolGradWithArgmaxV3KSizeOneTilingData &tilingData, TPipe& pipe):
        tilingData_(tilingData), pipe_(pipe) {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y);
    __aicore__ inline void Process();
private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t dataLen);
    __aicore__ inline void CopyOut(int64_t offset, int64_t dataLen);
private:
    TPipe &pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DB_BUFFER> dataQueue_;
    GlobalTensor<T> gradGm_;
    GlobalTensor<T> yGm_;
    const MaxPoolGradWithArgmaxV3KSizeOneTilingData &tilingData_;
    int64_t blockIdx_ = 0;
};

template <typename T>
__aicore__ inline void MaxPoolGradWithArgmaxV3KsizeOne<T>::Init(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y)
{
    blockIdx_ = GetBlockIdx();
    int64_t blockOffset = blockIdx_ * tilingData_.blockFactor;
    gradGm_.SetGlobalBuffer((__gm__ T *)(grad) + blockOffset);
    yGm_.SetGlobalBuffer((__gm__ T *)(y) + blockOffset);

    int64_t bufferSize = tilingData_.ubFactor  * sizeof(T);
    pipe_.InitBuffer(dataQueue_, DB_BUFFER, bufferSize);
}

template <typename T>
__aicore__ inline void MaxPoolGradWithArgmaxV3KsizeOne<T>::CopyIn(int64_t offset, int64_t dataLen)
{
    DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = dataLen * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    DataCopyPadExtParams<T> padParams = { false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0) };
    LocalTensor<T> gradLocal = dataQueue_.AllocTensor<T>();
    DataCopyPad(gradLocal, gradGm_[offset], extParams, padParams);
    dataQueue_.EnQue(gradLocal);
}

template <typename T>
__aicore__ inline void MaxPoolGradWithArgmaxV3KsizeOne<T>::CopyOut(int64_t offset, int64_t dataLen)
{
    DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = dataLen * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    LocalTensor<T> yLocal = dataQueue_.DeQue<T>();
    DataCopyPad(yGm_[offset], yLocal, extParams);
    dataQueue_.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void MaxPoolGradWithArgmaxV3KsizeOne<T>::Process()
{
    if (blockIdx_ >= tilingData_.usedCoreNum) {
        return;
    }
    int64_t loopSize = tilingData_.coreLoop;
    int64_t tailUbFactor = tilingData_.tailUbFactor;
    if (blockIdx_ == tilingData_.usedCoreNum - 1) {
        loopSize = tilingData_.tailCoreLoop;
    }
    int64_t offset = 0;
    int64_t dataLen = tilingData_.ubFactor;
    for (int64_t idx = 0; idx < loopSize - 1; idx++) {
        CopyIn(offset, dataLen);
        CopyOut(offset, dataLen);
        offset += dataLen;
    }

    dataLen = tailUbFactor;
    if (blockIdx_ == tilingData_.usedCoreNum - 1) {
        dataLen = tilingData_.tailCoreTailUbFactor;
    }
    CopyIn(offset, dataLen);
    CopyOut(offset, dataLen);
}

}  // namespace MaxPoolGradWithArgmaxV3KsizeOneNameSpace

#endif  // MAX_POOL_GRAD_WITH_ARGMAX_V3_KSIZE_ONE_H_
