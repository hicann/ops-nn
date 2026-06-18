/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MAX_POOL3D_WITH_ARGMAX_V2_KSIZE_ONE_H_
#define MAX_POOL3D_WITH_ARGMAX_V2_KSIZE_ONE_H_

#include "max_pool3d_with_argmax_v2_base.h"
#include "max_pool3d_with_argmax_v2_tiling_struct.h"

namespace MaxPool3DWithArgmaxV2KsizeOneNameSpace {
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;

template <typename T1, typename T2>
class MaxPool3DWithArgmaxV2KsizeOneKernel {
public:
    __aicore__ inline MaxPool3DWithArgmaxV2KsizeOneKernel(
        TPipe& pipeIn, const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2KsizeOneTilingData& tilingData)
        : pipe_(pipeIn), tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR argmax);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessTile(int64_t globalElemOffset, int64_t tileLen);
    __aicore__ inline void DirectCopy(int64_t srcGmOffset, int64_t dstGmOffset, int64_t dataLen);
    __aicore__ inline void BatchArgmax(int64_t dstGmOffset, int64_t dataLen, int64_t baseIdx);

    TPipe& pipe_;
    const MaxPool3DWithArgmaxV2Tiling::MaxPool3DWithArgmaxV2KsizeOneTilingData& tilingData_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> dataQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> argmaxQue_;

    GlobalTensor<T1> xGm_;
    GlobalTensor<T1> yGm_;
    GlobalTensor<T2> argmaxGm_;

    uint32_t blockIdx_ = 0;
};

template <typename T1, typename T2>
__aicore__ inline void MaxPool3DWithArgmaxV2KsizeOneKernel<T1, T2>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR argmax)
{
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= tilingData_.usedCoreNum) {
        return;
    }
    xGm_.SetGlobalBuffer((__gm__ T1*)x);
    yGm_.SetGlobalBuffer((__gm__ T1*)y);
    argmaxGm_.SetGlobalBuffer((__gm__ T2*)argmax);

    pipe_.InitBuffer(dataQueue_, BUFFER_NUM, tilingData_.inputBufferSize);
    pipe_.InitBuffer(argmaxQue_, BUFFER_NUM, tilingData_.argmaxBufferSize);
}

template <typename T1, typename T2>
__aicore__ inline void MaxPool3DWithArgmaxV2KsizeOneKernel<T1, T2>::Process()
{
    if (blockIdx_ >= tilingData_.usedCoreNum) {
        return;
    }

    bool isTailCore = (blockIdx_ == tilingData_.usedCoreNum - 1);
    int64_t coreLoop = isTailCore ? tilingData_.tailCoreLoop : tilingData_.coreLoop;
    int64_t tailTileSize = isTailCore ? tilingData_.tailCoreTailUbFactor : tilingData_.tailUbFactor;

    int64_t globalOffset = blockIdx_ * tilingData_.blockFactor;

    for (int64_t loop = 0; loop < coreLoop; loop++) {
        int64_t tileLen = (loop < coreLoop - 1) ? tilingData_.ubFactor : tailTileSize;
        if (tileLen <= 0) {
            break;
        }
        int64_t globalElemOffset = globalOffset + loop * tilingData_.ubFactor;
        ProcessTile(globalElemOffset, tileLen);
    }
}

template <typename T1, typename T2>
__aicore__ inline void MaxPool3DWithArgmaxV2KsizeOneKernel<T1, T2>::ProcessTile(
    int64_t globalElemOffset, int64_t tileLen)
{
    DirectCopy(globalElemOffset, globalElemOffset, tileLen);

    int64_t outDHW = tilingData_.dOutput * tilingData_.hOutput * tilingData_.wOutput;
    int64_t ncPlaneOffset = globalElemOffset - (globalElemOffset / outDHW) * outDHW;
    int64_t dstOffset = globalElemOffset;
    int64_t remaining = tileLen;

    while (remaining > 0) {
        int64_t elemsInThisNc = outDHW - ncPlaneOffset;
        int64_t chunk = (remaining < elemsInThisNc) ? remaining : elemsInThisNc;
        BatchArgmax(dstOffset, chunk, ncPlaneOffset);
        dstOffset += chunk;
        remaining -= chunk;
        ncPlaneOffset = 0;
    }
}

template <typename T1, typename T2>
__aicore__ inline void MaxPool3DWithArgmaxV2KsizeOneKernel<T1, T2>::DirectCopy(
    int64_t srcGmOffset, int64_t dstGmOffset, int64_t dataLen)
{
    LocalTensor<T1> dataLocal = dataQueue_.AllocTensor<T1>();
    DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = dataLen * sizeof(T1);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    DataCopyPadExtParams<T1> padExtParams = {false, 0, 0, 0};
    DataCopyPad(dataLocal, xGm_[srcGmOffset], extParams, padExtParams);
    dataQueue_.EnQue(dataLocal);

    LocalTensor<T1> outLocal = dataQueue_.DeQue<T1>();
    DataCopyExtParams outParams;
    outParams.blockCount = 1;
    outParams.blockLen = dataLen * sizeof(T1);
    outParams.srcStride = 0;
    outParams.dstStride = 0;
    DataCopyPad(yGm_[dstGmOffset], outLocal, outParams);
    dataQueue_.FreeTensor(outLocal);
}

template <typename T1, typename T2>
__aicore__ inline void MaxPool3DWithArgmaxV2KsizeOneKernel<T1, T2>::BatchArgmax(
    int64_t dstGmOffset, int64_t dataLen, int64_t baseIdx)
{
    LocalTensor<T2> argmaxLocal = argmaxQue_.AllocTensor<T2>();

    AscendC::Arange(argmaxLocal, static_cast<T2>(baseIdx), static_cast<T2>(1), static_cast<int32_t>(dataLen));

    argmaxQue_.EnQue(argmaxLocal);
    LocalTensor<T2> argmaxOut = argmaxQue_.DeQue<T2>();
    DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = dataLen * sizeof(T2);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    DataCopyPad(argmaxGm_[dstGmOffset], argmaxOut, extParams);
    argmaxQue_.FreeTensor(argmaxOut);
}

} // namespace MaxPool3DWithArgmaxV2KsizeOneNameSpace
#endif // MAX_POOL3D_WITH_ARGMAX_V2_KSIZE_ONE_H_