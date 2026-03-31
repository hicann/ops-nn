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
 * \file reverse_v2_tensor_move.h
 * \brief
 */

#ifndef REVERSE_V2_TENSOR_MOVE_H_
#define REVERSE_V2_TENSOR_MOVE_H_

#include "kernel_operator.h"

namespace ReverseV2 {
using namespace AscendC;

constexpr int64_t DB_BUFFER = 2;

template <typename T>
class ReverseV2TensorMoveKernel {
public:
    __aicore__ inline ReverseV2TensorMoveKernel(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const TensorMoveTilingData &tilingData);
    __aicore__ inline void Process();
private:
    __aicore__ inline void ParseTilingData(const TensorMoveTilingData &tilingData);
    __aicore__ inline void CopyIn(int64_t offset, int64_t dataLen);
    __aicore__ inline void CopyOut(int64_t offset, int64_t dataLen);
private:
    TPipe pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DB_BUFFER> dataQueue_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    int64_t blockIdx_ = 0;
    int64_t blockOffset_ = 0;
    int64_t totalCoreNum_;
    int64_t usedCoreNum_;
    int64_t ubFactor_;
    int64_t tailBlockTailUbFactor_;
    int64_t blockFactor_;
    int64_t tailBlockFactor_;
    int64_t bufferSize_;
};

template <typename T>
__aicore__ inline void ReverseV2TensorMoveKernel<T>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const TensorMoveTilingData &tilingData)
{
    blockIdx_ = GetBlockIdx();
    ParseTilingData(tilingData);
    blockOffset_ = GetBlockIdx() * blockFactor_ * ubFactor_;
    xGm_.SetGlobalBuffer((__gm__ T *)(x) + blockOffset_);
    yGm_.SetGlobalBuffer((__gm__ T *)(y) + blockOffset_);

    bufferSize_ = ubFactor_  * sizeof(T);
    pipe_.InitBuffer(dataQueue_, DB_BUFFER, bufferSize_);
}

template <typename T>
__aicore__ inline void ReverseV2TensorMoveKernel<T>::ParseTilingData(const TensorMoveTilingData &tilingData)
{
    totalCoreNum_ = tilingData.totalCoreNum;
    usedCoreNum_ = tilingData.usedCoreNum;
    ubFactor_ = tilingData.ubFactor;
    tailBlockTailUbFactor_ = tilingData.tailBlockTailUbFactor;
    blockFactor_ = tilingData.blockFactor;
    tailBlockFactor_ = tilingData.tailBlockFactor;
}

template <typename T>
__aicore__ inline void ReverseV2TensorMoveKernel<T>::CopyIn(int64_t offset, int64_t dataLen)
{
  DataCopyExtParams inParams = {
    static_cast<uint16_t>(1),
    static_cast<uint32_t>(dataLen * sizeof(T)),
    static_cast<uint32_t>(0),
    static_cast<uint32_t>(0),
    static_cast<uint32_t>(0)
  };
  DataCopyPadExtParams<T> padParams = { false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0) };
  LocalTensor<T> xLocal = dataQueue_.AllocTensor<T>();
  DataCopyPad(xLocal, xGm_[offset], inParams, padParams);
  dataQueue_.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void ReverseV2TensorMoveKernel<T>::CopyOut(int64_t offset, int64_t dataLen)
{
  DataCopyExtParams outParams = {
    static_cast<uint16_t>(1),
    static_cast<uint32_t>(dataLen * sizeof(T)),
    static_cast<uint32_t>(0),
    static_cast<uint32_t>(0),
    static_cast<uint32_t>(0)
  };
  LocalTensor<T> yLocal = dataQueue_.DeQue<T>();
  DataCopyPad(yGm_[offset], yLocal, outParams);
  dataQueue_.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void ReverseV2TensorMoveKernel<T>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }
    int64_t loopSize = blockFactor_;
    if (blockIdx_ == usedCoreNum_ - 1) {
        loopSize = tailBlockFactor_;
    }
    int64_t offset = 0;
    for (int64_t idx = 0; idx < loopSize - 1; idx++) {
        offset = idx * ubFactor_;
        CopyIn(offset, ubFactor_);
        CopyOut(offset, ubFactor_);
    }

    offset = (loopSize - 1) * ubFactor_;
    int64_t dataLen = ubFactor_;
    if (blockIdx_ == usedCoreNum_ - 1) {
        dataLen = tailBlockTailUbFactor_;
    }
    CopyIn(offset, dataLen);
    CopyOut(offset, dataLen);
}

}  // namespace TensorMove

#endif  // REVERSE_V2_TENSOR_MOVE_H_
