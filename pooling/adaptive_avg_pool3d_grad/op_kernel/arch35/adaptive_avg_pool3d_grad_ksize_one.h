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
 * \file adaptive_avg_pool3d_grad_ksize_one.h
 * \brief pure-copy kernel for grad shape == x shape (kernel=1x1x1)
 */

#ifndef ADAPTIVE_AVG_POOL3D_GRAD_KSIZE_ONE_H_
#define ADAPTIVE_AVG_POOL3D_GRAD_KSIZE_ONE_H_

#include "kernel_operator.h"

namespace AdaptiveAvgPool3dGradOp {
using namespace AscendC;

constexpr int64_t COPY_DB_BUFFER = 2;

template <typename T>
class AdaptiveAvgPool3dGradKsizeOne {
public:
    __aicore__ inline AdaptiveAvgPool3dGradKsizeOne(TPipe* pipe,
                                                    const AdaptiveAvgPool3dGradKsizeOneTilingDataV35* tilingData)
        : pipe_(pipe), tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR y_grad, GM_ADDR x_grad);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t offset, int64_t dataLen);
    __aicore__ inline void CopyOut(int64_t offset, int64_t dataLen);

private:
    TPipe* pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, COPY_DB_BUFFER> dataQueue_;
    GlobalTensor<T> gradGm_;
    GlobalTensor<T> xGradGm_;
    const AdaptiveAvgPool3dGradKsizeOneTilingDataV35* tilingData_;
    int64_t blockIdx_ = 0;
};

template <typename T>
__aicore__ inline void AdaptiveAvgPool3dGradKsizeOne<T>::Init(GM_ADDR y_grad, GM_ADDR x_grad)
{
    blockIdx_ = GetBlockIdx();
    int64_t blockOffset = blockIdx_ * tilingData_->blockFactor;
    gradGm_.SetGlobalBuffer((__gm__ T*)(y_grad) + blockOffset);
    xGradGm_.SetGlobalBuffer((__gm__ T*)(x_grad) + blockOffset);

    int64_t bufferSize = tilingData_->ubFactor * sizeof(T);
    pipe_->InitBuffer(dataQueue_, COPY_DB_BUFFER, bufferSize);
}

template <typename T>
__aicore__ inline void AdaptiveAvgPool3dGradKsizeOne<T>::CopyIn(int64_t offset, int64_t dataLen)
{
    DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = dataLen * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    DataCopyPadExtParams<T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0)};
    LocalTensor<T> gradLocal = dataQueue_.AllocTensor<T>();
    DataCopyPad(gradLocal, gradGm_[offset], extParams, padParams);
    dataQueue_.EnQue(gradLocal);
}

template <typename T>
__aicore__ inline void AdaptiveAvgPool3dGradKsizeOne<T>::CopyOut(int64_t offset, int64_t dataLen)
{
    DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = dataLen * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    LocalTensor<T> xGradLocal = dataQueue_.DeQue<T>();
    DataCopyPad(xGradGm_[offset], xGradLocal, extParams);
    dataQueue_.FreeTensor(xGradLocal);
}

template <typename T>
__aicore__ inline void AdaptiveAvgPool3dGradKsizeOne<T>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }
    int64_t loopSize = tilingData_->coreLoop;
    int64_t tailUbFactor = tilingData_->tailUbFactor;
    if (blockIdx_ == tilingData_->usedCoreNum - 1) {
        loopSize = tilingData_->tailCoreLoop;
    }
    int64_t offset = 0;
    int64_t dataLen = tilingData_->ubFactor;
    for (int64_t idx = 0; idx < loopSize - 1; idx++) {
        CopyIn(offset, dataLen);
        CopyOut(offset, dataLen);
        offset += dataLen;
    }

    dataLen = tailUbFactor;
    if (blockIdx_ == tilingData_->usedCoreNum - 1) {
        dataLen = tilingData_->tailCoreTailUbFactor;
    }
    CopyIn(offset, dataLen);
    CopyOut(offset, dataLen);
}

} // namespace AdaptiveAvgPool3dGradOp

#endif // ADAPTIVE_AVG_POOL3D_GRAD_KSIZE_ONE_H_
