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
 * \file pool_3d_ndhwc_small_kernel.h
 * \brief
 */
#ifndef MAX_POOL_V3_NDHWC_SMALL_KERNEL_H_
#define MAX_POOL_V3_NDHWC_SMALL_KERNEL_H_

#include "copy_pad_impl.h"
#include "pool_3d_common.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "gather_index_impl.h"

namespace Pool3D
{
using namespace AscendC;

template <typename T, int32_t OP_TYPE>
class Pool3dNDHWCSmallKernel
{
public:
    __aicore__ inline Pool3dNDHWCSmallKernel(TPipe* pipe, const Pool3DSmallKernelNDHWCTilingData* __restrict tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    template <typename M, typename U, int32_t GATHER_MODE>
    __aicore__ inline void BaseCompute();
    __aicore__ inline void CopyInMultiRows(int64_t offset, const TensorDescInfo& inputInfo);
    __aicore__ inline void CopyMaxOut(int64_t offset, int64_t n, int64_t blockCount, int64_t blockLen,
                                      int64_t channels);
    template <typename M, typename U, bool USE_TRAIT_TWO>
    __aicore__ inline void ComputeMultiRow(const TensorDescInfo& inputInfo, const ShapeInfo& outInfo,
                                           const Pool3dParam& paramInfo);
    template <typename M, typename U, bool USE_TRAIT_TWO>
    __aicore__ inline void ComputeSingleRow(const TensorDescInfo& inputInfo, const ShapeInfo& outInfo,
                                            const Pool3dParam& paramInfo);
    template <typename M, typename U, bool USE_TRAIT_TWO>
    __aicore__ inline void ComputeMultiDep(const TensorDescInfo& inputInfo, const ShapeInfo& outInfo,
                                           const Pool3dParam& paramInfo);
    template <typename M, typename U, bool USE_TRAIT_TWO>
    __aicore__ inline void ComputeMultiBatch(const TensorDescInfo& inputInfo, const ShapeInfo& outInfo,
                                             const Pool3dParam& paramInfo);
    template <typename M, typename U, int32_t GATHER_MODE, bool USE_TRAIT_TWO>
    __aicore__ inline void Compute(const TensorDescInfo &inputInfo, const ShapeInfo &outInfo, const Pool3dParam &paramInfo);
    template <typename U, int32_t GATHER_MODE>
    __aicore__ inline void GenGatherIndex(const GatherIndexImpl::ShapeInfo& param, LocalTensor<U>& indexLocal,
                                          uint16_t loopNum = 1);
    __aicore__ inline int64_t min(int64_t a, int64_t b)
    {
        return (a > b) ? b : a;
    }

    TPipe* pipe_;
    // 输入队列
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    // 输出ub
    TQue<QuePosition::VECOUT, BUFFER_NUM> maxUBOutput_;
    TBuf<QuePosition::VECCALC> indexBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> maxGm_;
    const Pool3DSmallKernelNDHWCTilingData* tilingData_;
    uint32_t eleInBlk_ = static_cast<uint32_t>(platform::GetUbBlockSize() / sizeof(T));
    uint32_t oneRepeatNum_ = platform::GetVRegSize();
    bool needDivisorBuf_ = false;
};

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3dNDHWCSmallKernel<T, OP_TYPE>::Init(GM_ADDR x, GM_ADDR y)
{
    // GM
    if (tilingData_->useTraiTwo) {
        oneRepeatNum_ = 2 * platform::GetVRegSize();
    }
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    maxGm_.SetGlobalBuffer((__gm__ T*)y);

    pipe_->InitBuffer(inputQue_, BUFFER_NUM, tilingData_->inUbSize * sizeof(T));
    pipe_->InitBuffer(maxUBOutput_, BUFFER_NUM, tilingData_->outUbSize * sizeof(T));
    pipe_->InitBuffer(indexBuf_, 2 * INDEX_SIZE);
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3dNDHWCSmallKernel<T, OP_TYPE>::Process()
{
    using indiceType = typename IndexTypeGet<T>::type;
    using computType = typename GetComputeType<T>::type;

    if (tilingData_->gatherMode == GATHER_SINGLE_ROW) {
        BaseCompute<computType, indiceType, GATHER_SINGLE_ROW>();
    } else if (tilingData_->gatherMode == GATHER_MULTI_ROW) {
        BaseCompute<computType, indiceType, GATHER_MULTI_ROW>();
    } else if (tilingData_->gatherMode == GATHER_MULTI_PLANE) {
        BaseCompute<computType, indiceType, GATHER_MULTI_PLANE>();
    } else {
        BaseCompute<computType, indiceType, GATHER_MULTI_BATCH>();
    }
}

template <typename T, int32_t OP_TYPE>
template <typename M, typename U, int32_t GATHER_MODE>
__aicore__ inline void Pool3dNDHWCSmallKernel<T, OP_TYPE>::BaseCompute()
{
    uint32_t sW = tilingData_->sW;
    uint32_t sH = tilingData_->sH;
    uint32_t sD = tilingData_->sD;
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

    uint32_t oneRowElements = ((outUbFactorW - 1) * sW + effectiveKw) * channels;
    uint32_t onePlaneElements = oneRowElements * ((outUbFactorH - 1) * sH + effectiveKh);
    uint32_t oneBatchElements = onePlaneElements * ((outUbFactorD - 1) * sD + effectiveKd);
    if (tilingData_->splitMode == SPLIT_COLS) {
        onePlaneElements = ops::Aligned(onePlaneElements, eleInBlk_);
        oneBatchElements = onePlaneElements * ((outUbFactorD - 1) * sD + effectiveKd);
    } else if (tilingData_->splitMode == SPLIT_ROWS) {
        oneRowElements = tilingData_->wInDim * channels;
        onePlaneElements = oneRowElements * ((outUbFactorH - 1) * sH + effectiveKh);
        oneBatchElements = onePlaneElements * ((outUbFactorD - 1) * sD + effectiveKd);
    } else if (tilingData_->splitMode == SPLIT_DEPTHS) {
        oneRowElements = tilingData_->wInDim * channels;
        onePlaneElements = oneRowElements * tilingData_->hInDim;
        oneBatchElements = onePlaneElements * ((outUbFactorD - 1) * sD + effectiveKd);
    } else {
        oneRowElements = tilingData_->wInDim * channels;
        onePlaneElements = oneRowElements * tilingData_->hInDim;
        oneBatchElements = onePlaneElements * tilingData_->dInDim;
    }

    GatherIndexImpl::ShapeInfo info = {
        {1, channels, oneRowElements, onePlaneElements, oneBatchElements},  // input stride
        {channels, outUbFactorW, outUbFactorH, outUbFactorD, static_cast<uint32_t>(tilingData_->ubFactorN)}, // gather size
        {1, sW, sH, sD, 1}  // stride
    };

    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    GenGatherIndex<U, GATHER_MODE>(info, indexLocal, 2);
    Pool3dParam paramInfo = {
        {static_cast<uint16_t>(tilingData_->kW), static_cast<uint16_t>(tilingData_->kH),
         static_cast<uint16_t>(tilingData_->kD)},
        {static_cast<uint16_t>(tilingData_->sW), static_cast<uint16_t>(tilingData_->sH),
         static_cast<uint16_t>(tilingData_->sD)},
        {static_cast<uint16_t>(tilingData_->dW), static_cast<uint16_t>(tilingData_->dH),
         static_cast<uint16_t>(tilingData_->dD)},
        static_cast<float32_t>(tilingData_->divisor),
    };

    for (int64_t idx = startIdx; idx < endIdx; idx++) {
        int64_t nIdx = idx / (tilingData_->dLoop * tilingData_->hLoop * tilingData_->wLoop);
        int64_t dIdx = (idx - nIdx * tilingData_->dLoop * tilingData_->hLoop * tilingData_->wLoop) /
                       (tilingData_->hLoop * tilingData_->wLoop);
        int64_t hIdx = (idx - nIdx * tilingData_->dLoop * tilingData_->hLoop * tilingData_->wLoop -
                        dIdx * tilingData_->hLoop * tilingData_->wLoop) / tilingData_->wLoop;
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
        int64_t inDeps =
            tilingData_->splitMode == SPLIT_BATCHS ? tilingData_->dInDim : (deps - 1) * sD + effectiveKd;
        int64_t inRows = tilingData_->splitMode == SPLIT_DEPTHS || tilingData_->splitMode == SPLIT_BATCHS
                              ? tilingData_->hInDim
                              : (rows - 1) * sH + effectiveKh;
        int64_t inCols =
            tilingData_->splitMode != SPLIT_COLS ? tilingData_->wInDim : (cols - 1) * sW + effectiveKw;
        oneRowElements = inCols * channels;
        onePlaneElements = oneRowElements * inRows;
        oneBatchElements = onePlaneElements * inDeps;
        if (tilingData_->splitMode == SPLIT_COLS) {
            onePlaneElements = ops::Aligned(onePlaneElements, eleInBlk_);
            oneBatchElements = onePlaneElements * inDeps;
        }
        TensorDescInfo inputInfo = {
            {static_cast<uint32_t>(inCols * channels), static_cast<uint32_t>(inRows), static_cast<uint32_t>(inDeps), static_cast<uint32_t>(n)},
            {1, oneRowElements, onePlaneElements, oneBatchElements},
            {1, tilingData_->wInDim * channels, tilingData_->wInDim * tilingData_->hInDim * channels,
             tilingData_->dInDim * tilingData_->hInDim * tilingData_->wInDim * channels}};
        CopyInMultiRows(srcOffset, inputInfo);
        ShapeInfo outInfo = {static_cast<uint16_t>(n), static_cast<uint16_t>(deps), static_cast<uint16_t>(rows),
                             static_cast<uint16_t>(cols), static_cast<uint16_t>(channels)};
        if (tilingData_->useTraiTwo) {
            Compute<M, U, GATHER_MODE, true>(inputInfo, outInfo, paramInfo);
        } else {
            Compute<M, U, GATHER_MODE, false>(inputInfo, outInfo, paramInfo);
        }
        CopyMaxOut(dstOffset, n * deps, rows, cols, channels);
    }
}

template <typename T, int32_t OP_TYPE>
template <typename M, typename U, int32_t GATHER_MODE, bool USE_TRAIT_TWO>
__aicore__ inline void Pool3dNDHWCSmallKernel<T, OP_TYPE>::Compute(const TensorDescInfo &inputInfo, const ShapeInfo &outInfo, const Pool3dParam &paramInfo) {
    if constexpr (GATHER_MODE == GATHER_SINGLE_ROW) {
        ComputeSingleRow<M, U, USE_TRAIT_TWO>(inputInfo, outInfo, paramInfo);
    } else if constexpr (GATHER_MODE == GATHER_MULTI_ROW) {
        ComputeMultiRow<M, U, USE_TRAIT_TWO>(inputInfo, outInfo, paramInfo);
    } else if constexpr (GATHER_MODE == GATHER_MULTI_PLANE) {
        ComputeMultiDep<M, U, USE_TRAIT_TWO>(inputInfo, outInfo, paramInfo);
    } else {
        ComputeMultiBatch<M, U, USE_TRAIT_TWO>(inputInfo, outInfo, paramInfo);
    }
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3dNDHWCSmallKernel<T, OP_TYPE>::CopyInMultiRows(int64_t offset,
                                                                           const TensorDescInfo& inputInfo)
{
    LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();
    DataCopyPadExtParams<T> padExtParams;
    padExtParams.isPad = false;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = 0;
    padExtParams.paddingValue = 0;
    LoopModeParams loopParams;
    DataCopyExtParams extParams;
    if (tilingData_->splitMode == SPLIT_COLS) {
        extParams.blockCount = inputInfo.size[1];
        extParams.blockLen = inputInfo.size[0] * sizeof(T);
        extParams.srcStride = (inputInfo.srcStride[1] - inputInfo.size[0]) * sizeof(T);
        extParams.dstStride = 0;
        loopParams.loop2Size = inputInfo.size[3];
        loopParams.loop1Size = inputInfo.size[2];
        loopParams.loop2SrcStride = inputInfo.srcStride[3] * sizeof(T);
        loopParams.loop2DstStride = inputInfo.dstStride[3] * sizeof(T);
        loopParams.loop1SrcStride = inputInfo.srcStride[2] * sizeof(T);
        loopParams.loop1DstStride = inputInfo.dstStride[2] * sizeof(T);
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPad<T, PaddingMode::Compact>(xLocal, xGm_[offset], extParams, padExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    } else if (tilingData_->splitMode == SPLIT_ROWS) {
        extParams.blockCount = inputInfo.size[2];
        extParams.blockLen = inputInfo .size[0] * inputInfo.size[1] * sizeof(T);
        extParams.srcStride = (inputInfo.srcStride[2] - inputInfo.size[0] * inputInfo.size[1]) * sizeof(T);
        extParams.dstStride = ((inputInfo.dstStride[2] - inputInfo.size[0] * inputInfo.size[1]) * sizeof(T)) / platform::GetUbBlockSize();
        loopParams.loop2Size = 1;
        loopParams.loop1Size = inputInfo.size[3];
        loopParams.loop1SrcStride = inputInfo.srcStride[3] * sizeof(T);
        loopParams.loop1DstStride = inputInfo.dstStride[3] * sizeof(T);
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPad<T, PaddingMode::Compact>(xLocal, xGm_[offset], extParams, padExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    } else if (tilingData_->splitMode == SPLIT_DEPTHS) {
        extParams.blockCount = inputInfo.size[3];
        extParams.blockLen = inputInfo.size[0] * inputInfo.size[1] * inputInfo.size[2] * sizeof(T);
        extParams.srcStride =
            (inputInfo.srcStride[3] - inputInfo.size[0] * inputInfo.size[1] * inputInfo.size[2]) * sizeof(T);
        extParams.dstStride = 0;
        DataCopyPad<T, PaddingMode::Compact>(xLocal, xGm_[offset], extParams, padExtParams);
    } else {
        DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = inputInfo.size[0] * inputInfo.size[1] * inputInfo.size[2] * inputInfo.size[3] * sizeof(T);
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        DataCopyPad<T, PaddingMode::Compact>(xLocal, xGm_[offset], extParams, padExtParams);
    }
    inputQue_.EnQue(xLocal);
}

template <typename T, int32_t OP_TYPE>
__aicore__ inline void Pool3dNDHWCSmallKernel<T, OP_TYPE>::CopyMaxOut(int64_t offset, int64_t n, int64_t blockCount,
                                                                      int64_t blockLen, int64_t channels)
{
    LocalTensor<T> maxOutLocal = maxUBOutput_.DeQue<T>();
    DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = (n * blockCount * blockLen * channels) * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    DataCopyPad<T>(maxGm_[offset], maxOutLocal, extParams);
    maxUBOutput_.FreeTensor<T>(maxOutLocal);
}

template <typename T, int32_t OP_TYPE>
template <typename U, int32_t GATHER_MODE>
__aicore__ inline void Pool3dNDHWCSmallKernel<T, OP_TYPE>::GenGatherIndex(const GatherIndexImpl::ShapeInfo& param,
                                                                          LocalTensor<U>& indexLocal,
                                                                          uint16_t loopNum)
{
    if constexpr (GATHER_MODE == GATHER_SINGLE_ROW) {
        GatherIndexImpl::GenGatherIndex<U, GatherIndexImpl::TWO>(param, indexLocal, loopNum);
    } else if constexpr (GATHER_MODE == GATHER_MULTI_ROW) {
        GatherIndexImpl::GenGatherIndex<U, GatherIndexImpl::THREE>(param, indexLocal, loopNum);
    } else if constexpr (GATHER_MODE == GATHER_MULTI_PLANE) {
        GatherIndexImpl::GenGatherIndex<U, GatherIndexImpl::FOUR>(param, indexLocal, loopNum);
    } else {
        GatherIndexImpl::GenGatherIndex<U, GatherIndexImpl::FIVE>(param, indexLocal, loopNum);
    }
}

template <typename T, int32_t OP_TYPE>
template <typename M, typename U, bool USE_TRAIT_TWO>
__aicore__ inline void Pool3dNDHWCSmallKernel<T, OP_TYPE>::ComputeMultiBatch(const TensorDescInfo& inputInfo,
                                                                             const ShapeInfo& outInfo,
                                                                             const Pool3dParam& paramInfo)
{
    LocalTensor<M> maxOutLocal = maxUBOutput_.AllocTensor<M>();
    LocalTensor<M> xLocal = inputQue_.DeQue<M>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __local_mem__ M* xLocalAddr = (__local_mem__ M*)xLocal.GetPhyAddr();
    __local_mem__ M* dstLocalAddr = (__local_mem__ M*)maxOutLocal.GetPhyAddr();
    uint32_t repeatElm = oneRepeatNum_ / sizeof(U);
    uint32_t channels = tilingData_->channels;
    uint16_t nFactor = static_cast<uint16_t>(repeatElm / (outInfo.depth * outInfo.width * outInfo.height * outInfo.channel));
    uint16_t loopN = static_cast<uint16_t>(outInfo.n / nFactor);
    uint16_t tailN = static_cast<uint16_t>(outInfo.n - loopN * nFactor);

    uint16_t kD = paramInfo.kSize[2];
    uint16_t kH = paramInfo.kSize[1];
    uint16_t kW = paramInfo.kSize[0];
    U oneLoopStride = static_cast<U>(nFactor * inputInfo.dstStride[3]);
    U oneLoopElements = static_cast<U>(nFactor * outInfo.depth * outInfo.width * outInfo.height * outInfo.channel);
    U tailLoopElements = static_cast<U>(tailN * outInfo.depth * outInfo.width * outInfo.height * outInfo.channel);

    U depStrideInUb = inputInfo.dstStride[2] * paramInfo.dilation[2];
    U rowStrideInUb = inputInfo.dstStride[1] * paramInfo.dilation[1];
    U colStrideInUb = static_cast<U>(tilingData_->channels * paramInfo.dilation[0]);
    Pool3DWithOneLoop<T, U, T, OP_TYPE, false, USE_TRAIT_TWO>(dstLocalAddr, xLocalAddr, indexAddr, kD, kH, kW, depStrideInUb,
                                             rowStrideInUb, colStrideInUb, oneLoopElements, tailLoopElements,
                                             oneLoopStride, loopN, paramInfo.divisor);
    inputQue_.FreeTensor<M>(xLocal);
    maxUBOutput_.EnQue<M>(maxOutLocal);
}

template <typename T, int32_t OP_TYPE>
template <typename M, typename U, bool USE_TRAIT_TWO>
__aicore__ inline void Pool3dNDHWCSmallKernel<T, OP_TYPE>::ComputeMultiDep(const TensorDescInfo& inputInfo,
                                                                           const ShapeInfo& outInfo,
                                                                           const Pool3dParam& paramInfo)
{
    LocalTensor<M> maxOutLocal = maxUBOutput_.AllocTensor<M>();
    LocalTensor<M> xLocal = inputQue_.DeQue<M>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __local_mem__ M* xLocalAddr = (__local_mem__ M*)xLocal.GetPhyAddr();
    __local_mem__ M* dstLocalAddr = (__local_mem__ M*)maxOutLocal.GetPhyAddr();
    uint32_t repeatElm = oneRepeatNum_ / sizeof(U);

    uint16_t channels = tilingData_->channels;
    uint16_t loopN = outInfo.n;
    uint16_t dFactor = static_cast<uint16_t>(repeatElm / (outInfo.width * outInfo.height * outInfo.channel));
    uint16_t loopD = static_cast<uint16_t>(outInfo.depth / dFactor);
    uint16_t tailD = static_cast<uint16_t>(outInfo.depth - loopD * dFactor);
    uint16_t kD = paramInfo.kSize[2];
    uint16_t kH = paramInfo.kSize[1];
    uint16_t kW = paramInfo.kSize[0];

    U oneLoopStride = dFactor * paramInfo.stride[2] * inputInfo.dstStride[2];
    U oneLoopElements = static_cast<U>(dFactor * outInfo.width * outInfo.height * outInfo.channel);
    U tailLoopElements = static_cast<U>(tailD * outInfo.width * outInfo.height * outInfo.channel);

    U depStrideInUb = inputInfo.dstStride[2] * paramInfo.dilation[2];
    U rowStrideInUb = inputInfo.dstStride[1] * paramInfo.dilation[1];
    U colStrideInUb = static_cast<U>(tilingData_->channels * paramInfo.dilation[0]);

    for (int i = 0; i < outInfo.n; i++) {
        auto srcAddr = xLocalAddr + i * inputInfo.dstStride[3];
        auto dstAddr = dstLocalAddr + i * outInfo.depth * outInfo.height * outInfo.width;
        Pool3DWithOneLoop<T, U, T, OP_TYPE, false, USE_TRAIT_TWO>(dstAddr, xLocalAddr, indexAddr, kD, kH, kW, depStrideInUb,
                                                 rowStrideInUb, colStrideInUb, oneLoopElements, tailLoopElements,
                                                 oneLoopStride, loopD, paramInfo.divisor);
    }

    inputQue_.FreeTensor<M>(xLocal);
    maxUBOutput_.EnQue<M>(maxOutLocal);
}

template <typename T, int32_t OP_TYPE>
template <typename M, typename U, bool USE_TRAIT_TWO>
__aicore__ inline void Pool3dNDHWCSmallKernel<T, OP_TYPE>::ComputeMultiRow(const TensorDescInfo& inputInfo,
                                                                           const ShapeInfo& outInfo,
                                                                           const Pool3dParam& paramInfo)
{
    LocalTensor<M> maxOutLocal = maxUBOutput_.AllocTensor<M>();
    LocalTensor<M> xLocal = inputQue_.DeQue<M>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __local_mem__ M* xLocalAddr = (__local_mem__ M*)xLocal.GetPhyAddr();
    __local_mem__ M* dstLocalAddr = (__local_mem__ M*)maxOutLocal.GetPhyAddr();

    uint16_t channels = tilingData_->channels;
    uint32_t repeatElm = oneRepeatNum_ / sizeof(U);
    uint16_t hFactor = static_cast<uint16_t>(repeatElm / (outInfo.width * channels));
    uint16_t loopH = static_cast<uint16_t>(outInfo.height / hFactor);
    uint16_t tailH = static_cast<uint16_t>(outInfo.height - loopH * hFactor);
    uint16_t kD = paramInfo.kSize[2];
    uint16_t kH = paramInfo.kSize[1];
    uint16_t kW = paramInfo.kSize[0];

    U oneBatchElements = inputInfo.dstStride[3];
    U oneBatchOutElements = outInfo.depth * outInfo.height * outInfo.width * channels;
    U oneDepOutElements = outInfo.height * outInfo.width * channels;
    U oneLoopStrideD = paramInfo.stride[2] * inputInfo.dstStride[2];
    U oneLoopStrideH = static_cast<U>(hFactor * paramInfo.stride[1] * inputInfo.dstStride[1]);
    U oneLoopElements = static_cast<U>(hFactor * outInfo.width * channels);

    U depStrideInUb = inputInfo.dstStride[2] * paramInfo.dilation[2];
    U rowStrideInUb = inputInfo.dstStride[1] * paramInfo.dilation[1];
    U colStrideInUb = static_cast<U>(tilingData_->channels * paramInfo.dilation[0]);
    uint32_t tailLoopElements = static_cast<uint32_t>(tailH * outInfo.width * channels);
    for (uint16_t nIdx = 0; nIdx < outInfo.n; nIdx++) {
        for (uint16_t dIdx = 0; dIdx < outInfo.depth; dIdx++) {
            auto srcAddr = xLocalAddr + nIdx * oneBatchElements + dIdx * oneLoopStrideD;
            auto dstAddr = dstLocalAddr + nIdx * oneBatchOutElements + dIdx * oneDepOutElements;
            Pool3DWithOneLoop<T, U, T, OP_TYPE, false, USE_TRAIT_TWO>(dstAddr, srcAddr, indexAddr, kD, kH, kW, depStrideInUb,
                                                     rowStrideInUb, colStrideInUb, oneLoopElements,
                                                     tailLoopElements, oneLoopStrideH, loopH, paramInfo.divisor);
        }
    }
    inputQue_.FreeTensor<M>(xLocal);
    maxUBOutput_.EnQue<M>(maxOutLocal);
}

template <typename T, int32_t OP_TYPE>
template <typename M, typename U, bool USE_TRAIT_TWO>
__aicore__ inline void Pool3dNDHWCSmallKernel<T, OP_TYPE>::ComputeSingleRow(const TensorDescInfo& inputInfo,
                                                                            const ShapeInfo& outInfo,
                                                                            const Pool3dParam& paramInfo)
{
    LocalTensor<M> maxOutLocal = maxUBOutput_.AllocTensor<M>();
    LocalTensor<M> xLocal = inputQue_.DeQue<M>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __local_mem__ M* xLocalAddr = (__local_mem__ M*)xLocal.GetPhyAddr();
    __local_mem__ M* dstLocalAddr = (__local_mem__ M*)maxOutLocal.GetPhyAddr();
    uint32_t repeatElm = oneRepeatNum_ / sizeof(U);
    uint16_t channels = tilingData_->channels;
    uint16_t wFactor = repeatElm / outInfo.channel;
    uint16_t loopW = static_cast<uint16_t>(outInfo.width / wFactor);
    uint16_t tailW = static_cast<uint16_t>(outInfo.width - loopW * wFactor);
    uint16_t kD = paramInfo.kSize[2];
    uint16_t kH = paramInfo.kSize[1];
    uint16_t kW = paramInfo.kSize[0];

    U depStrideInUb = inputInfo.dstStride[2] * paramInfo.dilation[2];
    U rowStrideInUb = inputInfo.dstStride[1] * paramInfo.dilation[1];
    U colStrideInUb = static_cast<U>(tilingData_->channels * paramInfo.dilation[0]);
    U oneLoopElements = static_cast<U>(wFactor * outInfo.channel);
    U oneBatchOutElements = static_cast<U>(outInfo.depth * outInfo.height * outInfo.width * outInfo.channel);
    U oneDepOutElements = static_cast<U>(outInfo.height * outInfo.width * outInfo.channel);
    U oneLoopStrideW = static_cast<U>(wFactor * paramInfo.stride[0] * channels);
    uint32_t tailLoopElements = tailW * outInfo.channel;
    for (uint16_t i = 0; i < outInfo.n; i++) {
        for (uint16_t dIdx = 0; dIdx < outInfo.depth; dIdx++) {
            for (uint16_t hIdx = 0; hIdx < outInfo.height; hIdx++) {
                __local_mem__ M* srcAddr = xLocalAddr + i * inputInfo.dstStride[3] +
                                           dIdx * paramInfo.stride[2] * inputInfo.dstStride[2] +
                                           hIdx * paramInfo.stride[1] * inputInfo.dstStride[1];
                __local_mem__ M* dstAddr =
                    dstLocalAddr + i * oneBatchOutElements + dIdx * oneDepOutElements + hIdx * outInfo.width * outInfo.channel;
                Pool3DWithOneLoop<T, U, T, OP_TYPE, false, USE_TRAIT_TWO>(
                    dstAddr, srcAddr, indexAddr, kD, kH, kW, depStrideInUb, rowStrideInUb, colStrideInUb,
                    oneLoopElements, tailLoopElements, oneLoopStrideW, loopW, paramInfo.divisor);
            }
        }
    }
    inputQue_.FreeTensor<M>(xLocal);
    maxUBOutput_.EnQue<M>(maxOutLocal);
}

}  // namespace Pool3D
#endif  // MAX_POOL_V3_NDHWC_SMALL_KERNEL_H_
