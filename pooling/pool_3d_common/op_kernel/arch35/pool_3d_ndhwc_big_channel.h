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
 * \file pool_3d_ndhwc_big_channel.h
 * \brief
 */
#ifndef MAX_POOL_V3_NDHWC_BIG_CHANNEL_H_
#define MAX_POOL_V3_NDHWC_BIG_CHANNEL_H_

#include "copy_pad_impl.h"
#include "pool_3d_common.h"
#include "../inc/platform.h"
#include "kernel_operator.h"
#include "../inc/kernel_utils.h"

namespace Pool3D
{

using namespace AscendC;

template <typename T, int32_t OP_TYPE>
class Pool3dNDHWCBigChannel
{
public:
    __aicore__ inline Pool3dNDHWCBigChannel(TPipe* pipe, const Pool3DSmallKernelNDHWCTilingData* __restrict tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    template <typename M, typename U>
    __aicore__ inline void BaseCompute();
    __aicore__ inline void CopyInMultiChannels(int64_t offset, TensorDescInfo& inputInfo);
    __aicore__ inline void CopyOutMultiChannels(int64_t offset, int64_t n, int64_t deps, int64_t rows, int64_t cols,
                                                int64_t channels);
    template <typename M, typename U>
    __aicore__ inline void ComputeSingleChannels(int64_t n, int64_t inDeps, int64_t inRows, int64_t cols,
                                                 int64_t outDeps, int64_t outRows, int64_t outCols,
                                                 int64_t alignChannels);
    TPipe* pipe_;
    // 输入队列
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    // 输出ub
    TQue<QuePosition::VECOUT, BUFFER_NUM> maxUBOutput_;
    TBuf<QuePosition::VECCALC> indexBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> maxGm_;
    const Pool3DSmallKernelNDHWCTilingData* tilingData_;
};

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3dNDHWCBigChannel<T, OP_TYPE>::Init(GM_ADDR x, GM_ADDR y)
{
    // GM
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    maxGm_.SetGlobalBuffer((__gm__ T*)y);

    pipe_->InitBuffer(inputQue_, BUFFER_NUM, tilingData_->inUbSize * sizeof(T));
    pipe_->InitBuffer(maxUBOutput_, BUFFER_NUM, tilingData_->outUbSize * sizeof(T));
    pipe_->InitBuffer(indexBuf_, INDEX_SIZE);
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3dNDHWCBigChannel<T, OP_TYPE>::Process()
{
    using indiceType = typename IndexTypeGet<T>::type;
    using computType = typename GetComputeType<T>::type;

    BaseCompute<computType, indiceType>();
}

template <typename T, int32_t OP_TYPE>
template <typename M, typename U>
__aicore__ inline void Pool3dNDHWCBigChannel<T, OP_TYPE>::BaseCompute()
{
    int64_t sW = tilingData_->sW;
    int64_t sH = tilingData_->sH;
    int64_t sD = tilingData_->sD;
    uint32_t channels = tilingData_->channels;

    uint32_t outUbFactorW = tilingData_->outUbFactorW;
    uint32_t outUbFactorH = tilingData_->outUbFactorH;
    uint32_t outUbFactorD = tilingData_->outUbFactorD;
    uint32_t effectiveKd = (tilingData_->kD - 1) * tilingData_->dD + 1;
    uint32_t effectiveKh = (tilingData_->kH - 1) * tilingData_->dH + 1;
    uint32_t effectiveKw = (tilingData_->kW - 1) * tilingData_->dW + 1;
    int64_t startIdx = 0;
    int64_t endIdx = 0;
    if (GetBlockIdx() < tilingData_->blockTail) {
        startIdx = GetBlockIdx() * (tilingData_->blockFactor + 1);
        endIdx = startIdx + tilingData_->blockFactor + 1;
    } else {
        startIdx = GetBlockIdx() * tilingData_->blockFactor + tilingData_->blockTail;
        endIdx = startIdx + tilingData_->blockFactor;
    }
    uint32_t alignChannels = ops::Aligned(channels, static_cast<uint32_t>(platform::GetUbBlockSize() / sizeof(T)));

    for (int64_t idx = startIdx; idx < endIdx; idx++) {
        int64_t nIdx = idx / (tilingData_->dLoop * tilingData_->hLoop * tilingData_->wLoop);
        int64_t dIdx = (idx - nIdx * tilingData_->dLoop * tilingData_->hLoop * tilingData_->wLoop) /
                       (tilingData_->hLoop * tilingData_->wLoop);
        int64_t hIdx = (idx - nIdx * tilingData_->dLoop * tilingData_->hLoop * tilingData_->wLoop -
                        dIdx * tilingData_->hLoop * tilingData_->wLoop) /
                       tilingData_->wLoop;
        int64_t wIdx = idx % tilingData_->wLoop;
        uint32_t n = nIdx == tilingData_->nLoop - 1 ? tilingData_->nOutDim - nIdx * tilingData_->ubFactorN
                                                    : tilingData_->ubFactorN;
        int64_t deps = dIdx == tilingData_->dLoop - 1 ? tilingData_->dOutDim - dIdx * tilingData_->outUbFactorD
                                                      : tilingData_->outUbFactorD;
        int64_t rows = hIdx == tilingData_->hLoop - 1 ? tilingData_->hOutDim - hIdx * outUbFactorH : outUbFactorH;
        int64_t cols = wIdx == tilingData_->wLoop - 1 ? tilingData_->wOutDim - wIdx * outUbFactorW : outUbFactorW;
        int64_t depStart = dIdx * sD * outUbFactorD;
        int64_t rowStart = hIdx * sH * outUbFactorH;
        int64_t colStart = wIdx * sW * outUbFactorW;

        int64_t srcOffset =
            nIdx * tilingData_->ubFactorN * tilingData_->dInDim * tilingData_->hInDim * tilingData_->wInDim * channels +
            depStart * tilingData_->hInDim * tilingData_->wInDim * channels +
            rowStart * tilingData_->wInDim * channels + colStart * channels;
        int64_t dstOffset = nIdx * tilingData_->ubFactorN * tilingData_->dOutDim * tilingData_->hOutDim *
                                tilingData_->wOutDim * channels +
                            dIdx * outUbFactorD * tilingData_->hOutDim * tilingData_->wOutDim * channels +
                            hIdx * outUbFactorH * tilingData_->wOutDim * channels + wIdx * outUbFactorW * channels;
        uint32_t inDeps =
            tilingData_->splitMode == SPLIT_BATCHS ? tilingData_->dInDim : (deps - 1) * sD + effectiveKd;
        uint32_t inRows = tilingData_->splitMode == SPLIT_DEPTHS || tilingData_->splitMode == SPLIT_BATCHS
                              ? tilingData_->hInDim
                              : (rows - 1) * sH + effectiveKh;
        uint32_t inCols =
            tilingData_->splitMode != SPLIT_COLS ? tilingData_->wInDim : (cols - 1) * sW + effectiveKw;

        uint32_t oneRowElements = inCols * alignChannels;
        uint32_t onePlaneElements = oneRowElements * inRows;
        uint32_t oneBatchElements = onePlaneElements * inDeps;
        TensorDescInfo inputInfo = {
            {channels, inCols, inRows, inDeps, n},
            {1, alignChannels, oneRowElements, onePlaneElements, oneBatchElements},
            {1, channels, tilingData_->wInDim * channels, tilingData_->wInDim * tilingData_->hInDim * channels,
             tilingData_->dInDim * tilingData_->hInDim * tilingData_->wInDim * channels}};
        CopyInMultiChannels(srcOffset, inputInfo);
        ComputeSingleChannels<M, U>(n, inDeps, inRows, inCols, deps, rows, cols, alignChannels);
        CopyOutMultiChannels(dstOffset, n, deps, rows, cols, channels);
    }
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3dNDHWCBigChannel<T, OP_TYPE>::CopyInMultiChannels(int64_t offset, TensorDescInfo& inputInfo)
{
    LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();
    DataCopyPadExtParams<T> padExtParams;
    padExtParams.isPad = false;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = 0;
    padExtParams.paddingValue = 0;
    uint32_t dstStride = 0;
    DataCopyExtParams extParams;
    extParams.blockCount = inputInfo.size[1];
    extParams.blockLen = inputInfo.size[0] * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = dstStride;
    // 切w 跳h 跳d
    if (tilingData_->splitMode == SPLIT_COLS) {
        extParams.blockCount = inputInfo.size[1];
        LoopModeParams loopParams;
        loopParams.loop2Size = inputInfo.size[3];
        loopParams.loop2SrcStride = inputInfo.srcStride[3] * sizeof(T);
        loopParams.loop2DstStride = inputInfo.dstStride[3] * sizeof(T);
        loopParams.loop1Size = inputInfo.size[2];
        loopParams.loop1SrcStride = inputInfo.srcStride[2] * sizeof(T);
        loopParams.loop1DstStride = inputInfo.dstStride[2] * sizeof(T);
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPad<T>(xLocal, xGm_[offset], extParams, padExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    } else if (tilingData_->splitMode == SPLIT_ROWS) {  // 切h 跳d
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = inputInfo.size[3];
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = inputInfo.srcStride[3] * sizeof(T);
        loopParams.loop1DstStride = inputInfo.dstStride[3] * sizeof(T);
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        extParams.blockCount = inputInfo.size[1] * inputInfo.size[2];
        DataCopyPad<T>(xLocal, xGm_[offset], extParams, padExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    } else {  // 切d n 不跳
        extParams.blockCount = inputInfo.size[1] * inputInfo.size[2] * inputInfo.size[3] * inputInfo.size[4];
        DataCopyPad<T>(xLocal, xGm_[offset], extParams, padExtParams);
    }
    inputQue_.EnQue(xLocal);
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3dNDHWCBigChannel<T, OP_TYPE>::CopyOutMultiChannels(int64_t offset, int64_t n, int64_t deps,
                                                                               int64_t rows, int64_t cols,
                                                                               int64_t channels)
{
    LocalTensor<T> maxOutLocal = maxUBOutput_.DeQue<T>();
    DataCopyExtParams extParams;
    extParams.blockCount = n * deps * rows * cols;
    extParams.blockLen = channels * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    DataCopyPad<T>(maxGm_[offset], maxOutLocal, extParams);
    maxUBOutput_.FreeTensor<T>(maxOutLocal);
}

template <typename T, int32_t OP_TYPE>
template <typename M, typename U>
__aicore__ inline void Pool3dNDHWCBigChannel<T, OP_TYPE>::ComputeSingleChannels(int64_t n, int64_t inDeps,
                                                                                int64_t inRows, int64_t inCols,
                                                                                int64_t outDeps, int64_t outRows,
                                                                                int64_t outCols, int64_t alignChannels)
{
    LocalTensor<M> maxOutLocal = maxUBOutput_.AllocTensor<M>();
    LocalTensor<M> xLocal = inputQue_.DeQue<M>();
    __local_mem__ M* xLocalAddr = (__local_mem__ M*)xLocal.GetPhyAddr();
    __local_mem__ M* dstLocalAddr = (__local_mem__ M*)maxOutLocal.GetPhyAddr();

    uint16_t kH = tilingData_->kH;
    uint16_t kW = tilingData_->kW;
    uint16_t kD = tilingData_->kD;
    uint16_t sH = tilingData_->sH;
    uint16_t sW = tilingData_->sW;
    uint16_t sD = tilingData_->sD;
    uint32_t outUbFactorD = outDeps;
    uint32_t outUbFactorH = outRows;
    uint32_t outUbFactorW = outCols;
    uint16_t loopN = tilingData_->ubFactorN;
    uint16_t loopH = outRows;
    uint16_t loopW = outCols;
    uint16_t loopD = outDeps;
    uint32_t dalitionW = tilingData_->dW;
    uint32_t dalitionH = tilingData_->dH;
    uint32_t dalitionD = tilingData_->dD;
    U batchStride = static_cast<U>(inDeps * inRows * inCols * alignChannels);
    U colStride = static_cast<U>(alignChannels * dalitionW);
    uint32_t rowStride = static_cast<uint32_t>(inCols * alignChannels * dalitionH);
    uint32_t depStride = static_cast<uint32_t>(inRows * inCols * alignChannels * dalitionD);
    // dalition 如何添加
    U oneLoopStrideD = static_cast<U>(sD * inRows * inCols * alignChannels);
    U oneLoopStrideH = static_cast<U>(sH * inCols * alignChannels);
    U oneLoopStrideW = static_cast<U>(sW * alignChannels);
    U oneChannelOutElements = static_cast<U>(outUbFactorD * outUbFactorH * outUbFactorW * alignChannels);
    U outLoopStrideH = static_cast<U>(outCols * alignChannels);
    U outLoopStrideD = static_cast<U>(outRows * outCols * alignChannels);
    constexpr uint32_t repeatElm = platform::GetVRegSize() / sizeof(U);
    uint16_t cLoop = alignChannels / repeatElm;
    uint16_t tailNum = alignChannels - cLoop * repeatElm;
    float32_t divisor = tilingData_->divisor;

    for (uint16_t i = 0; i < loopN; i++) {
        for (uint16_t d = 0; d < loopD; d++) {
            for (uint16_t j = 0; j < loopH; j++) {
                for (uint16_t k = 0; k < loopW; k++) {
                    auto srcAddr =
                        xLocalAddr + i * batchStride + d * oneLoopStrideD + j * oneLoopStrideH + k * oneLoopStrideW;
                    auto dstAddr = dstLocalAddr + i * oneChannelOutElements + d * outLoopStrideD + j * outLoopStrideH +
                                   k * alignChannels;
                    __VEC_SCOPE__
                    {
                        for (uint16_t m = 0; m < cLoop; m++) {
                            auto curSrcAddr = srcAddr + m * repeatElm;
                            auto curDstAddr = dstAddr + m * repeatElm;
                            if constexpr (OP_TYPE == Pool3D::OP_TYPE_MAX_POOL_3D) {
                                MaxPoolSingleChannel<M, U>(curDstAddr, curSrcAddr, kD, kH, kW, depStride, rowStride,
                                                           colStride, repeatElm);
                            } else {
                                AvgPoolSingleChannelImpl<M, U>(curDstAddr, curSrcAddr, kD, kH, kW, depStride, rowStride,
                                                               colStride, repeatElm, divisor);
                            }
                        }
                        auto curSrcAddr = srcAddr + cLoop * repeatElm;
                        auto curDstAddr = dstAddr + cLoop * repeatElm;
                        if constexpr (OP_TYPE == Pool3D::OP_TYPE_MAX_POOL_3D) {
                            MaxPoolSingleChannel<M, U>(curDstAddr, curSrcAddr, kD, kH, kW, depStride, rowStride,
                                                       colStride, tailNum);
                        } else {
                            AvgPoolSingleChannelImpl<M, U>(curDstAddr, curSrcAddr, kD, kH, kW, depStride, rowStride,
                                                           colStride, tailNum, divisor);
                        }
                    }
                }
            }
        }
    }
    inputQue_.FreeTensor<M>(xLocal);
    maxUBOutput_.EnQue<M>(maxOutLocal);
}

}  // namespace Pool3D
#endif  // MAX_POOL_V3_NDHWC_BIG_CHANNEL_H_
