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
 * \file pool_3d_ncdhw_small_kernel_pad.h
 * \brief
 */
#ifndef POOL_3D_SMALL_KERNEL_PAD_H_
#define POOL_3D_SMALL_KERNEL_PAD_H_

#include "pool_3d_common.h"
#include "gather_index_impl.h"
#include "scatter_index_impl.h"
#include "copy_pad_impl.h"

#include "../inc/platform.h"
#include "../inc/kernel_utils.h"

namespace Pool3D
{
using namespace AscendC;

template <typename T, int32_t OP_TYPE, bool OUT_DIV=false>
class Pool3DNcdhwSmallKernelPad
{
public:
    __aicore__ inline Pool3DNcdhwSmallKernelPad(TPipe* pipe, const Pool3DNcdhwSmallKernelTilingData* __restrict tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    template <typename U, int32_t GATHER_MODE>
    __aicore__ inline void BaseCompute();
    __aicore__ inline void InitDivisor();
    __aicore__ inline void CopyInSingleRow(int64_t offset, int64_t blockLen);
    __aicore__ inline void CopyInMultiRows(int64_t offset, const TensorDescInfo &inputInfo, int32_t splitMode);
    __aicore__ inline void CopyMaxOut(int64_t offset, int64_t blockLen);
    template <typename U, bool USE_TRAIT_TWO>
    __aicore__ inline void ComputeMultiBatch(const ShapeInfo &outInfo, const Pool3dParam &paramInfo, const CopyPad::CopyPadShapeInfo &padInfo);
    template <typename U, bool USE_TRAIT_TWO>
    __aicore__ inline void ComputeMultiDepth(const ShapeInfo &outInfo, const Pool3dParam &paramInfo, const CopyPad::CopyPadShapeInfo &padInfo);
    template <typename U, bool USE_TRAIT_TWO>
    __aicore__ inline void ComputeMultiRow(const ShapeInfo &outInfo, const Pool3dParam &paramInfo, const CopyPad::CopyPadShapeInfo &padInfo);
    template <typename U, bool USE_TRAIT_TWO>
    __aicore__ inline void ComputeSingleRow(const ShapeInfo &outInfo, const Pool3dParam &paramInfo, const CopyPad::CopyPadShapeInfo &padInfo);
    template <typename U, int32_t GATHER_MODE, bool USE_TRAIT_TWO>
    __aicore__ inline void Compute(const ShapeInfo &outInfo, const Pool3dParam &paramInfo, const CopyPad::CopyPadShapeInfo &padInfo);
    template <typename U, int32_t GATHER_MODE>
    __aicore__ inline void GenGatherIndex(const GatherIndexImpl::ShapeInfo &param, LocalTensor<U>& indexLocal, uint16_t loopNum=1);
    __aicore__ inline void ComputeDivisor(int64_t start, int64_t num) ;
    __aicore__ inline void DivCompute(__local_mem__ T* dstAddr, __local_mem__ float32_t* srcAddr, uint32_t num) ;
    template <typename U, int32_t GATHER_MODE>
    __aicore__ inline void GenIndexInfo(CopyPad::CopyPadShapeInfo &padInfo, uint32_t &realOneRowElements, uint32_t &realOnePlaneElements, uint32_t &realOneBatchElements);
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
    TBuf<QuePosition::VECCALC> tmpBuf_;
    TBuf<QuePosition::VECCALC> scatterIndexBuf_;
    TBuf<QuePosition::VECCALC> divisorBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> maxGm_;
    const Pool3DNcdhwSmallKernelTilingData* tilingData_;
    uint32_t eleInBlk_ = static_cast<uint32_t>(platform::GetUbBlockSize() / sizeof(T));
    uint32_t oneRepeatNum_ = platform::GetVRegSize();
    uint32_t effectiveKd_ = 1;
    uint32_t effectiveKh_ = 1;
    uint32_t effectiveKw_ = 1;
    int32_t copyMode_ = 0;
    int64_t curOffsetInBatch_ = 0;
    bool needDivisorBuf_ = false;
};

template <typename T, int32_t OP_TYPE, bool OUT_DIV>
__aicore__ inline void Pool3DNcdhwSmallKernelPad<T, OP_TYPE, OUT_DIV>::Init(GM_ADDR x, GM_ADDR y)
{
    // GM
    if (tilingData_->useTraiTwo) {
        oneRepeatNum_ = 2 * platform::GetVRegSize();
    }
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    maxGm_.SetGlobalBuffer((__gm__ T*)y);
    needDivisorBuf_ = OP_TYPE == OP_TYPE_AVG_POOL_3D && tilingData_->divisor == 0;
    pipe_->InitBuffer(inputQue_, BUFFER_NUM, tilingData_->inUbSize * sizeof(T));
    pipe_->InitBuffer(maxUBOutput_, BUFFER_NUM, tilingData_->outUbSize * sizeof(T));
    pipe_->InitBuffer(tmpBuf_, tilingData_->inUbSize * sizeof(T));
    pipe_->InitBuffer(indexBuf_, 2 * INDEX_SIZE);
    pipe_->InitBuffer(scatterIndexBuf_, INDEX_SIZE);
    if constexpr (OUT_DIV) {
        pipe_->InitBuffer(divisorBuf_, tilingData_->divisorUbSize);
    }
    effectiveKd_ = (tilingData_->kD - 1) * tilingData_->dD + 1;
    effectiveKh_ = (tilingData_->kH - 1) * tilingData_->dH + 1;
    effectiveKw_ = (tilingData_->kW - 1) * tilingData_->dW + 1;
}

template <typename T, int32_t OP_TYPE, bool OUT_DIV>
__aicore__ inline void Pool3DNcdhwSmallKernelPad<T, OP_TYPE, OUT_DIV>::InitDivisor()
{   
    if (OUT_DIV && tilingData_->realCalcDivisor == 0 && needDivisorBuf_) {
        int32_t oneVL = platform::GetVRegSize() / sizeof(float32_t);
        int32_t oneBatchNum = tilingData_->dOutDim * tilingData_->hOutDim * tilingData_->wOutDim;
        ComputeDivisor(0, max(oneVL, oneBatchNum));
    }
}

template <typename T, int32_t OP_TYPE, bool OUT_DIV>
__aicore__ inline void Pool3DNcdhwSmallKernelPad<T, OP_TYPE, OUT_DIV>::Process()
{
    using indiceType = typename IndexTypeGet<T>::type;
    InitDivisor();
    if (tilingData_->gatherMode == GATHER_SINGLE_ROW) {
        BaseCompute<indiceType, GATHER_SINGLE_ROW>();
    } else if (tilingData_->gatherMode == GATHER_MULTI_ROW) {
        BaseCompute<indiceType, GATHER_MULTI_ROW>();
    } else if (tilingData_->gatherMode == GATHER_MULTI_PLANE) {
        BaseCompute<indiceType, GATHER_MULTI_PLANE>();
    }  else {
        BaseCompute<indiceType, GATHER_MULTI_BATCH>();
    }
}

template <typename T, int32_t OP_TYPE, bool OUT_DIV>
template <typename U, int32_t GATHER_MODE>
__aicore__ inline void Pool3DNcdhwSmallKernelPad<T, OP_TYPE, OUT_DIV>::GenIndexInfo(CopyPad::CopyPadShapeInfo &padInfo, uint32_t &realOneRowElements, uint32_t &realOnePlaneElements, uint32_t &realOneBatchElements)
{
    uint32_t sW = tilingData_->sW;
    uint32_t sH = tilingData_->sH;
    uint32_t sD = tilingData_->sD;
    uint32_t outUbFactorW = tilingData_->outUbFactorW;
    uint32_t outUbFactorH = tilingData_->outUbFactorH;
    uint32_t outUbFactorD = tilingData_->outUbFactorD;
    uint32_t inDim0Size = static_cast<uint32_t>(tilingData_->wInDim);
    uint32_t inDim1Size = static_cast<uint32_t>(tilingData_->hInDim);
    uint32_t inDim2Size = static_cast<uint32_t>(tilingData_->dInDim);
    uint32_t inDim3Size = static_cast<uint32_t>(tilingData_->ubFactorN);

    CopyPad::CopyPadShapeInfo tmpPadInfo = {
        {inDim0Size, inDim1Size, inDim2Size, inDim3Size, 1}};
    copyMode_ = CopyPad::GetCopyMode<T>(tmpPadInfo, tilingData_->splitMode);
    
    uint32_t expectOneRowElements = (outUbFactorW - 1) * sW + effectiveKw_;
    uint32_t expectOnePlaneElements = realOneRowElements * ((outUbFactorH - 1) * sH + effectiveKh_);
    uint32_t expectOneBatchElements = realOnePlaneElements * ((outUbFactorD - 1) * sD + effectiveKd_);
    if (tilingData_->splitMode == SPLIT_COLS || copyMode_ == CopyPad::COPY_SINGLE_ROW || copyMode_ == CopyPad::SCATTER_ONE_DIM) {
        expectOneRowElements = realOneRowElements;
        uint32_t maxColsElement = min(static_cast<int64_t>(realOneRowElements), tilingData_->wInDim);
        realOneRowElements = ops::Aligned(maxColsElement, eleInBlk_);
        expectOnePlaneElements = expectOneRowElements * ((outUbFactorH - 1) * sH + effectiveKh_);
        realOnePlaneElements = realOneRowElements * ((outUbFactorH - 1) * sH + effectiveKh_);
        expectOneBatchElements = expectOnePlaneElements * ((outUbFactorD - 1) * sD + effectiveKd_);
        realOneBatchElements = realOnePlaneElements * ((outUbFactorD - 1) * sD + effectiveKd_);
    } else if (tilingData_->splitMode == SPLIT_ROWS) {
        expectOneRowElements = max(expectOneRowElements, static_cast<uint32_t>(tilingData_->wInDim + tilingData_->leftPad));
        realOneRowElements = tilingData_->wInDim;
        expectOnePlaneElements = expectOneRowElements * ((outUbFactorH - 1) * sH + effectiveKh_);
        realOnePlaneElements = realOneRowElements * ((outUbFactorH - 1) * sH + effectiveKh_);
        realOnePlaneElements = ops::Aligned(realOnePlaneElements, eleInBlk_);
        expectOneBatchElements = expectOnePlaneElements * ((outUbFactorD - 1) * sD + effectiveKd_);
        realOneBatchElements = realOnePlaneElements * ((outUbFactorD - 1) * sD + effectiveKd_);
    } else if (tilingData_->splitMode == SPLIT_DEPTHS) {
        expectOneRowElements = max(expectOneRowElements, static_cast<uint32_t>(tilingData_->wInDim + tilingData_->leftPad));
        realOneRowElements = tilingData_->wInDim;
        expectOnePlaneElements = expectOneRowElements * max(static_cast<uint32_t>((outUbFactorH - 1) * sH + effectiveKh_), 
                                                            static_cast<uint32_t>(tilingData_->hInDim + tilingData_->topPad));
        realOnePlaneElements = tilingData_->wInDim * tilingData_->hInDim;
        expectOneBatchElements = expectOnePlaneElements * ((outUbFactorD - 1) * sD + effectiveKd_);
        realOneBatchElements = realOnePlaneElements * ((outUbFactorD - 1) * sD + effectiveKd_);
        realOneBatchElements = ops::Aligned(realOneBatchElements, eleInBlk_);
    } else {
        expectOneRowElements = max(expectOneRowElements, static_cast<uint32_t>(tilingData_->wInDim + tilingData_->leftPad));
        realOneRowElements = tilingData_->wInDim;
        expectOnePlaneElements = expectOneRowElements * max(static_cast<uint32_t>((outUbFactorH - 1) * sH + effectiveKh_), 
                                                    static_cast<uint32_t>(tilingData_->hInDim + tilingData_->topPad));
        realOnePlaneElements = tilingData_->wInDim * tilingData_->hInDim;
        expectOneBatchElements = expectOnePlaneElements * max(static_cast<uint32_t>((outUbFactorD - 1) * sD + effectiveKd_), 
                                                        static_cast<uint32_t>(tilingData_->dInDim + tilingData_->frontPad));
        realOneBatchElements = realOnePlaneElements * tilingData_->dInDim;
    }

    GatherIndexImpl::ShapeInfo info = {
        {1, expectOneRowElements, expectOnePlaneElements, expectOneBatchElements, 1},     //input stride
        {outUbFactorW, outUbFactorH, outUbFactorD, static_cast<uint32_t>(tilingData_->ubFactorN), 1},    // gather size
        {sW, sH, sD, 1, 1}   //stride
    };

    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    GenGatherIndex<U, GATHER_MODE>(info, indexLocal, 2);

    ScatterIndexImpl::ScatterShapeInfo shapeInfo = {
        {inDim0Size, inDim1Size, inDim2Size, inDim3Size, 1},
        {1, expectOneRowElements, expectOnePlaneElements, expectOneBatchElements, 1},
    };
    LocalTensor<U> scatterLocal = scatterIndexBuf_.Get<U>();
    if (copyMode_ == CopyPad::SCATTER_ONE_DIM) {
        ScatterIndexImpl::GenScatterIndex<U, ScatterIndexImpl::ONE>(shapeInfo, scatterLocal);
    } else if (copyMode_ == CopyPad::SCATTER_TWO_DIMS) {
        ScatterIndexImpl::GenScatterIndex<U, ScatterIndexImpl::TWO>(shapeInfo, scatterLocal);
    } else if (copyMode_ == CopyPad::SCATTER_THREE_DIMS) {
        ScatterIndexImpl::GenScatterIndex<U, ScatterIndexImpl::THREE>(shapeInfo, scatterLocal);
    } else if (copyMode_ == CopyPad::SCATTER_FOUR_DIMS) {
        ScatterIndexImpl::GenScatterIndex<U, ScatterIndexImpl::FOUR>(shapeInfo, scatterLocal);
    }
    padInfo.srcStride[0] = 1;
    padInfo.srcStride[1] = realOneRowElements;
    padInfo.srcStride[2] = realOnePlaneElements;
    padInfo.srcStride[3] = realOneBatchElements;

    padInfo.dstStride[0] = 1;
    padInfo.dstStride[1] = expectOneRowElements;
    padInfo.dstStride[2] = expectOnePlaneElements;
    padInfo.dstStride[3] = expectOneBatchElements;
    padInfo.totalSize = tilingData_->ubFactorN * expectOneBatchElements;    
}

template <typename T, int32_t OP_TYPE, bool OUT_DIV>
template <typename U, int32_t GATHER_MODE>
__aicore__ inline void Pool3DNcdhwSmallKernelPad<T, OP_TYPE, OUT_DIV>::BaseCompute()
{
    uint32_t sW = tilingData_->sW;
    uint32_t sH = tilingData_->sH;
    uint32_t sD = tilingData_->sD;
    uint32_t outUbFactorW = tilingData_->outUbFactorW;
    uint32_t outUbFactorH = tilingData_->outUbFactorH;
    uint32_t outUbFactorD = tilingData_->outUbFactorD;
    int64_t startIdx = 0;
    int64_t endIdx = 0;
    if (GetBlockIdx() < tilingData_->blockTail) {
        startIdx = GetBlockIdx() * (tilingData_->blockFactor + 1);
        endIdx = startIdx + tilingData_->blockFactor + 1;
    } else {
        startIdx = GetBlockIdx() * tilingData_->blockFactor + tilingData_->blockTail;
        endIdx = startIdx + tilingData_->blockFactor;
    }
    Pool3dParam paramInfo = {
        {static_cast<uint16_t>(tilingData_->kW), static_cast<uint16_t>(tilingData_->kH), static_cast<uint16_t>(tilingData_->kD)},
        {static_cast<uint16_t>(tilingData_->sW), static_cast<uint16_t>(tilingData_->sH), static_cast<uint16_t>(tilingData_->sD)},
        {static_cast<uint16_t>(tilingData_->dW), static_cast<uint16_t>(tilingData_->dH), static_cast<uint16_t>(tilingData_->dD)},
        static_cast<float32_t>(tilingData_->divisor),
    };
    CopyPad::CopyPadShapeInfo padInfo;
    uint32_t realOneRowElements = (outUbFactorW - 1) * sW + effectiveKw_;
    uint32_t realOnePlaneElements = realOneRowElements * ((outUbFactorH - 1) * sH + effectiveKh_);
    uint32_t realOneBatchElements = realOnePlaneElements * ((outUbFactorD - 1) * sD + effectiveKd_);
    GenIndexInfo<U, GATHER_MODE>(padInfo, realOneRowElements, realOnePlaneElements, realOneBatchElements);
    bool needRealCol = tilingData_->splitMode == SPLIT_COLS || copyMode_ == CopyPad::COPY_SINGLE_ROW || copyMode_ == CopyPad::SCATTER_ONE_DIM;
    // effectiveKd_
    // effectiveKh_ 
    // effectiveKw_    
    for (int64_t idx = startIdx; idx < endIdx; idx++) {
        int64_t nIdx = idx / (tilingData_->dLoop * tilingData_->hLoop * tilingData_->wLoop);
        int64_t tmpIdx = idx - nIdx * tilingData_->dLoop * tilingData_->hLoop * tilingData_->wLoop;
        int64_t dIdx = tmpIdx / (tilingData_->hLoop * tilingData_->wLoop);
        int64_t hIdx = (tmpIdx - dIdx * tilingData_->hLoop * tilingData_->wLoop) / tilingData_->wLoop;
        int64_t wIdx = tmpIdx % tilingData_->wLoop;
        uint32_t n = nIdx == tilingData_->nLoop - 1 ? tilingData_->nOutDim - nIdx * tilingData_->ubFactorN
                                                   : tilingData_->ubFactorN;
        int64_t depths = dIdx == tilingData_->dLoop - 1 ? tilingData_->dOutDim - dIdx * outUbFactorD : outUbFactorD;                                           
        int64_t rows = hIdx == tilingData_->hLoop - 1 ? tilingData_->hOutDim - hIdx * outUbFactorH : outUbFactorH;
        int64_t cols = wIdx == tilingData_->wLoop - 1 ? tilingData_->wOutDim - wIdx * outUbFactorW : outUbFactorW;
        int64_t expectDepthStart = dIdx * sD * outUbFactorD; 
        int64_t expectRowStart = hIdx * sH * outUbFactorH;
        int64_t expectColStart = wIdx * sW * outUbFactorW;
        int64_t dUpper = min(expectDepthStart + (depths - 1) * sD + effectiveKd_ - tilingData_->frontPad, tilingData_->dInDim);
        int64_t hUpper = min(expectRowStart + (rows - 1) * sH + effectiveKh_ - tilingData_->topPad, tilingData_->hInDim);
        int64_t wUpper = min(expectColStart + (cols - 1) * sW + effectiveKw_ - tilingData_->leftPad, tilingData_->wInDim);

        int64_t dLower = max(dIdx * sD * outUbFactorD - tilingData_->frontPad, (int64_t)0);
        int64_t hLower = max(hIdx * sH * outUbFactorH - tilingData_->topPad, (int64_t)0);
        int64_t wLower = max(wIdx * sW * outUbFactorW - tilingData_->leftPad, (int64_t)0);

        uint32_t realDepths = tilingData_->splitMode == SPLIT_BATCHS ? tilingData_->dInDim : dUpper - dLower;
        uint32_t realRows = tilingData_->splitMode == SPLIT_BATCHS ||  tilingData_->splitMode == SPLIT_DEPTHS ? tilingData_->hInDim : hUpper - hLower;
        uint32_t realCols = needRealCol ? wUpper - wLower : tilingData_->wInDim;

        int64_t srcOffset = nIdx * tilingData_->ubFactorN * tilingData_->dInDim * tilingData_->hInDim * tilingData_->wInDim +
                            dLower * tilingData_->hInDim * tilingData_->wInDim + hLower * tilingData_->wInDim + wLower;
        int64_t dstOffset = nIdx * tilingData_->ubFactorN * tilingData_->dOutDim * tilingData_->hOutDim * tilingData_->wOutDim +
                            + dIdx * outUbFactorD * tilingData_->hOutDim * tilingData_->wOutDim 
                            + hIdx * outUbFactorH * tilingData_->wOutDim + wIdx * outUbFactorW;
        curOffsetInBatch_ = dIdx * outUbFactorD * tilingData_->hOutDim * tilingData_->wOutDim 
                            + hIdx * outUbFactorH * tilingData_->wOutDim + wIdx * outUbFactorW;
        int64_t expectDepths = (depths - 1) * sD + effectiveKd_;
        int64_t expectRows = (rows - 1) * sH + effectiveKh_;
        int64_t expectCols = (cols - 1) * sW + effectiveKw_;

        int64_t depthStart = expectDepthStart >= tilingData_->frontPad ? 0 : tilingData_->frontPad - expectDepthStart;
        int64_t rowStart = expectRowStart >= tilingData_->topPad ? 0 : tilingData_->topPad - expectRowStart;
        int64_t colStart = expectColStart >= tilingData_->leftPad ? 0 : tilingData_->leftPad - expectColStart;
        padInfo.inSize[0] = realCols;
        padInfo.inSize[1] = realRows;
        padInfo.inSize[2] = realDepths;
        padInfo.inSize[3] = n;
        padInfo.inSize[4] = 1;
        padInfo.expectStart[0] = colStart;
        padInfo.expectStart[1] = rowStart;
        padInfo.expectStart[2] = depthStart;
        padInfo.expectStart[3] = 0;
        padInfo.expectStart[4] = 0;
        TensorDescInfo inputInfo = {
            {realCols, realRows, realDepths, n},
            {1, realOneRowElements, realOnePlaneElements, realOneBatchElements},
            {1, tilingData_->wInDim, tilingData_->wInDim * tilingData_->hInDim, tilingData_->dInDim * tilingData_->hInDim * tilingData_->wInDim}
        };

        CopyInMultiRows(srcOffset, inputInfo, tilingData_->splitMode);

        ShapeInfo outInfo = {static_cast<uint16_t>(n), static_cast<uint16_t>(depths),static_cast<uint16_t>(rows),static_cast<uint16_t>(cols)};
        if (tilingData_->useTraiTwo) {
            Compute<U, GATHER_MODE, true>(outInfo, paramInfo, padInfo);
        } else {
            Compute<U, GATHER_MODE, false>(outInfo, paramInfo, padInfo);
        }
        CopyMaxOut(dstOffset, n*depths*rows*cols);
    }
}

template <typename T, int32_t OP_TYPE, bool OUT_DIV>
__aicore__ inline void Pool3DNcdhwSmallKernelPad<T, OP_TYPE, OUT_DIV>::CopyInSingleRow(int64_t offset, int64_t blockLen)
{
    LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();

    DataCopyPadExtParams<T> padExtParams;
    padExtParams.isPad = false;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = 0;
    padExtParams.paddingValue = 0;

    DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = blockLen * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    DataCopyPad(xLocal, xGm_[offset], extParams, padExtParams);
    inputQue_.EnQue(xLocal);
}

template <typename T, int32_t OP_TYPE, bool OUT_DIV>
__aicore__ inline void Pool3DNcdhwSmallKernelPad<T, OP_TYPE, OUT_DIV>::CopyInMultiRows(int64_t offset, const TensorDescInfo &inputInfo, int32_t splitMode)
{
    LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();
    DataCopyPadExtParams<T> padExtParams;
    padExtParams.isPad = false;
    padExtParams.leftPadding = 0;
    padExtParams.rightPadding = 0;
    padExtParams.paddingValue = 0;
    LoopModeParams loopParams;
    DataCopyExtParams extParams;
    if (tilingData_->splitMode == SPLIT_COLS || copyMode_ == CopyPad::COPY_SINGLE_ROW || copyMode_ == CopyPad::SCATTER_ONE_DIM) {    
        extParams.blockCount = inputInfo.size[1];
        extParams.blockLen = inputInfo.size[0] * sizeof(T);
        extParams.srcStride = (inputInfo.srcStride[1] - inputInfo.size[0]) * sizeof(T);
        extParams.dstStride = (inputInfo.dstStride[1] - inputInfo.size[0]) * sizeof(T) / platform::GetUbBlockSize();
        loopParams.loop2Size = inputInfo.size[3];
        loopParams.loop1Size = inputInfo.size[2];
        loopParams.loop2SrcStride = inputInfo.srcStride[3] * sizeof(T);
        loopParams.loop2DstStride = inputInfo.dstStride[3] * sizeof(T);
        loopParams.loop1SrcStride = inputInfo.srcStride[2] * sizeof(T);
        loopParams.loop1DstStride = inputInfo.dstStride[2] * sizeof(T);
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPad<T>(xLocal, xGm_[offset], extParams, padExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    } else if (tilingData_->splitMode == SPLIT_ROWS) {
        extParams.blockCount = inputInfo.size[2];
        extParams.blockLen = inputInfo.size[0] * inputInfo.size[1] * sizeof(T);
        extParams.srcStride = (inputInfo.srcStride[2] - inputInfo.size[0] * inputInfo.size[1] ) * sizeof(T);
        extParams.dstStride = (inputInfo.dstStride[2] - inputInfo.size[0] * inputInfo.size[1]) * sizeof(T) / platform::GetUbBlockSize();;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = inputInfo.size[3];
        loopParams.loop1SrcStride = inputInfo.srcStride[3] * sizeof(T);
        loopParams.loop1DstStride = inputInfo.dstStride[3] * sizeof(T);
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPad<T>(xLocal, xGm_[offset], extParams, padExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);        
    } else if (tilingData_->splitMode == SPLIT_DEPTHS) {
        extParams.blockCount = inputInfo.size[3];
        extParams.blockLen = inputInfo.size[0] * inputInfo.size[1] * inputInfo.size[2] * sizeof(T);
        extParams.srcStride = (inputInfo.srcStride[3] - inputInfo.size[0] * inputInfo.size[1] * inputInfo.size[2] ) * sizeof(T);
        extParams.dstStride = (inputInfo.dstStride[3] - inputInfo.size[0] * inputInfo.size[1] * inputInfo.size[2] ) * sizeof(T) / platform::GetUbBlockSize();
        DataCopyPad<T>(xLocal, xGm_[offset], extParams, padExtParams);
    }  else {
        DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = inputInfo.size[0] * inputInfo.size[1] * inputInfo.size[2] * inputInfo.size[3] * sizeof(T);
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        DataCopyPad<T>(xLocal, xGm_[offset], extParams, padExtParams);
    }
    inputQue_.EnQue(xLocal);
}

template <typename T, int32_t OP_TYPE, bool OUT_DIV>
__aicore__ inline void Pool3DNcdhwSmallKernelPad<T, OP_TYPE, OUT_DIV>::CopyMaxOut(int64_t offset, int64_t blockLen)
{
    LocalTensor<T> maxOutLocal = maxUBOutput_.DeQue<T>();
    DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = blockLen * sizeof(T);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    DataCopyPad<T>(maxGm_[offset], maxOutLocal, extParams);
    maxUBOutput_.FreeTensor<T>(maxOutLocal);
}

template <typename T, int32_t OP_TYPE, bool OUT_DIV>
template <typename U, int32_t GATHER_MODE>
__aicore__ inline void Pool3DNcdhwSmallKernelPad<T, OP_TYPE, OUT_DIV>::GenGatherIndex(const GatherIndexImpl::ShapeInfo &param,
                                                                LocalTensor<U>& indexLocal, uint16_t loopNum)
{
    if constexpr (GATHER_MODE == GATHER_SINGLE_ROW) {
        GatherIndexImpl::GenGatherIndex<U, 1>(param, indexLocal, loopNum);
    } else if constexpr (GATHER_MODE == GATHER_MULTI_ROW) {
        GatherIndexImpl::GenGatherIndex<U, 2>(param, indexLocal, loopNum);
    } else if constexpr (GATHER_MODE == GATHER_MULTI_PLANE) {
        GatherIndexImpl::GenGatherIndex<U, 3>(param, indexLocal, loopNum);
    } else {
        GatherIndexImpl::GenGatherIndex<U, 4>(param, indexLocal, loopNum);
    }
}

template <typename T, int32_t OP_TYPE, bool OUT_DIV>
template <typename U, int32_t GATHER_MODE, bool USE_TRAIT_TWO>
__aicore__ inline void Pool3DNcdhwSmallKernelPad<T, OP_TYPE, OUT_DIV>::Compute(const ShapeInfo &outInfo, const Pool3dParam &paramInfo, const CopyPad::CopyPadShapeInfo &padInfo)
{
    if constexpr (GATHER_MODE == GATHER_SINGLE_ROW) {
        ComputeSingleRow<U, USE_TRAIT_TWO>(outInfo, paramInfo, padInfo);
    } else if constexpr (GATHER_MODE == GATHER_MULTI_ROW) {
        ComputeMultiRow<U, USE_TRAIT_TWO>(outInfo, paramInfo, padInfo);
    } else if constexpr (GATHER_MODE == GATHER_MULTI_PLANE) {
        ComputeMultiDepth<U, USE_TRAIT_TWO>(outInfo, paramInfo, padInfo);
    } else {
        ComputeMultiBatch<U, USE_TRAIT_TWO>(outInfo, paramInfo, padInfo);
    }
}

template <typename T, int32_t OP_TYPE, bool OUT_DIV>
template <typename U, bool USE_TRAIT_TWO>
__aicore__ inline void Pool3DNcdhwSmallKernelPad<T, OP_TYPE, OUT_DIV>::ComputeMultiBatch(const ShapeInfo &outInfo, const Pool3dParam &paramInfo, const CopyPad::CopyPadShapeInfo &padInfo)
{
    LocalTensor<T> maxOutLocal = maxUBOutput_.AllocTensor<T>();
    LocalTensor<T> inLocal = inputQue_.DeQue<T>();
    LocalTensor<T> xLocal = tmpBuf_.Get<T>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();    
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __local_mem__ T* xLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    using Z = typename std::conditional<sizeof(T)==B16 && OUT_DIV && OP_TYPE == OP_TYPE_AVG_POOL_3D, float32_t,
                                    T>::type;
    __local_mem__ Z* dstLocalAddr = (__local_mem__ Z*)(maxOutLocal.template ReinterpretCast<Z>().GetPhyAddr());
    
    
    LocalTensor<U> scatterLocal = scatterIndexBuf_.Get<U>();
    if constexpr(OP_TYPE == Pool3D::OP_TYPE_MAX_POOL_3D) {
        CopyPad::CopyAndPad<T, U, CopyPad::DEFAULT_MODE_NEG_INF>(inLocal, xLocal, scatterLocal, padInfo, copyMode_);
    } else {
        CopyPad::CopyAndPad<T, U, CopyPad::DEFAULT_MODE_ZERO>(inLocal, xLocal, scatterLocal, padInfo, copyMode_);
    }
    uint32_t repeatElm = oneRepeatNum_ / sizeof(U);
    uint16_t nFactor = static_cast<uint16_t>(repeatElm / (outInfo.depth * outInfo.width * outInfo.height));
    uint16_t loopN = static_cast<uint16_t>(outInfo.n / nFactor);
    uint16_t tailN = static_cast<uint16_t>(outInfo.n - loopN * nFactor);

    U oneLoopStrideBtach = static_cast<U>(nFactor * padInfo.dstStride[3]);
    U oneLoopOutElements = static_cast<U>(nFactor * outInfo.depth * outInfo.width * outInfo.height);

    uint32_t tailLoopOutElements = static_cast<uint32_t>(tailN * outInfo.depth * outInfo.width * outInfo.height);

    U depthStrideInub = padInfo.dstStride[2] * paramInfo.dilation[2];
    U rowStrideInub = padInfo.dstStride[1] * paramInfo.dilation[1];
    U colStrideInub = paramInfo.dilation[0];
    uint16_t kD = paramInfo.kSize[2];
    uint16_t kH = paramInfo.kSize[1];
    uint16_t kW = paramInfo.kSize[0];
    Pool3DWithOneLoop<T, U, Z, OP_TYPE, OUT_DIV, USE_TRAIT_TWO>(dstLocalAddr, xLocalAddr, indexAddr, kD, kH,
                                kW, depthStrideInub, rowStrideInub, colStrideInub,
                                oneLoopOutElements, tailLoopOutElements, oneLoopStrideBtach, loopN, paramInfo.divisor);
    if constexpr (OUT_DIV && OP_TYPE == OP_TYPE_AVG_POOL_3D) {
        __local_mem__ T* newDstAddr = (__local_mem__ T*)maxOutLocal.GetPhyAddr();
        uint32_t totalOut = outInfo.n * outInfo.depth * outInfo.width * outInfo.height;
        DivCompute(newDstAddr, dstLocalAddr, totalOut);
    }
    inputQue_.FreeTensor<T>(inLocal);
    maxUBOutput_.EnQue<T>(maxOutLocal);
}

template <typename T, int32_t OP_TYPE, bool OUT_DIV>
template <typename U, bool USE_TRAIT_TWO>
__aicore__ inline void Pool3DNcdhwSmallKernelPad<T, OP_TYPE, OUT_DIV>::ComputeMultiDepth(const ShapeInfo &outInfo, const Pool3dParam &paramInfo, const CopyPad::CopyPadShapeInfo &padInfo)
{
    LocalTensor<T> maxOutLocal = maxUBOutput_.AllocTensor<T>();
    LocalTensor<T> inLocal = inputQue_.DeQue<T>();
    LocalTensor<T> xLocal = tmpBuf_.Get<T>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __local_mem__ T* srcAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    using Z = typename std::conditional<sizeof(T)==B16 && OUT_DIV && OP_TYPE == OP_TYPE_AVG_POOL_3D, float32_t,
                                    T>::type;   
    __local_mem__ Z* orgDstAddr = (__local_mem__ Z*)(maxOutLocal.template ReinterpretCast<Z>().GetPhyAddr());
    
    LocalTensor<U> scatterLocal = scatterIndexBuf_.Get<U>();
    if constexpr(OP_TYPE == Pool3D::OP_TYPE_MAX_POOL_3D) {
        CopyPad::CopyAndPad<T, U, CopyPad::DEFAULT_MODE_NEG_INF>(inLocal, xLocal, scatterLocal, padInfo, copyMode_);
    } else {
        CopyPad::CopyAndPad<T, U, CopyPad::DEFAULT_MODE_ZERO>(inLocal, xLocal, scatterLocal, padInfo, copyMode_);
    }
    uint32_t repeatElm = oneRepeatNum_ / sizeof(U);
    uint16_t dFactor = static_cast<uint16_t>(repeatElm / (outInfo.width * outInfo.height));
    uint16_t loopD = static_cast<uint16_t>(outInfo.depth / dFactor);
    uint16_t tailD = static_cast<uint16_t>(outInfo.depth - loopD * dFactor);

    U oneLoopStrideDepth = static_cast<U>(dFactor * paramInfo.stride[2] * padInfo.dstStride[2]);
    U oneLoopOutElements = static_cast<U>(dFactor * outInfo.width * outInfo.height);

    uint32_t tailLoopOutElements = static_cast<uint32_t>(tailD * outInfo.width * outInfo.height);

    U depthStrideInub = padInfo.dstStride[2] * paramInfo.dilation[2];
    U rowStrideInub = padInfo.dstStride[1] * paramInfo.dilation[1];
    U colStrideInub = paramInfo.dilation[0];
    uint16_t kD = paramInfo.kSize[2];
    uint16_t kH = paramInfo.kSize[1];
    uint16_t kW = paramInfo.kSize[0];
    for (uint16_t i = 0; i < outInfo.n; i ++) {
        __local_mem__ T* srcLocalAddr = srcAddr + i * padInfo.dstStride[3] ;
        __local_mem__ Z* dstLocalAddr = orgDstAddr + i * outInfo.depth * outInfo.height * outInfo.width;
        Pool3DWithOneLoop<T, U, Z, OP_TYPE, OUT_DIV, USE_TRAIT_TWO>(dstLocalAddr, srcLocalAddr, indexAddr, kD, kH,
                                    kW, depthStrideInub, rowStrideInub, colStrideInub,
                                    oneLoopOutElements, tailLoopOutElements, oneLoopStrideDepth, loopD, paramInfo.divisor);        
    }

    if constexpr (OUT_DIV && OP_TYPE == OP_TYPE_AVG_POOL_3D) {
        __local_mem__ T* newDstAddr = (__local_mem__ T*)maxOutLocal.GetPhyAddr();
        uint32_t totalOut = outInfo.n * outInfo.depth * outInfo.width * outInfo.height;
        DivCompute(newDstAddr, orgDstAddr, totalOut);
    }
    inputQue_.FreeTensor<T>(inLocal);
    maxUBOutput_.EnQue<T>(maxOutLocal);
}

template <typename T, int32_t OP_TYPE, bool OUT_DIV>
template <typename U, bool USE_TRAIT_TWO>
__aicore__ inline void Pool3DNcdhwSmallKernelPad<T, OP_TYPE, OUT_DIV>::ComputeMultiRow(const ShapeInfo &outInfo, const Pool3dParam &paramInfo, const CopyPad::CopyPadShapeInfo &padInfo)
{
    LocalTensor<T> maxOutLocal = maxUBOutput_.AllocTensor<T>();
    LocalTensor<T> inLocal = inputQue_.DeQue<T>();
    LocalTensor<T> xLocal = tmpBuf_.Get<T>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __local_mem__ T* srcAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    using Z = typename std::conditional<sizeof(T)==B16 && OUT_DIV && OP_TYPE == OP_TYPE_AVG_POOL_3D, float32_t,
                                    T>::type;   
    __local_mem__ Z* dstAddr = (__local_mem__ Z*)(maxOutLocal.template ReinterpretCast<Z>().GetPhyAddr());
    
    LocalTensor<U> scatterLocal = scatterIndexBuf_.Get<U>();
    if constexpr(OP_TYPE == Pool3D::OP_TYPE_MAX_POOL_3D) {
        CopyPad::CopyAndPad<T, U, CopyPad::DEFAULT_MODE_NEG_INF>(inLocal, xLocal, scatterLocal, padInfo, copyMode_);
    } else {
        CopyPad::CopyAndPad<T, U, CopyPad::DEFAULT_MODE_ZERO>(inLocal, xLocal, scatterLocal, padInfo, copyMode_);
    }
    uint32_t repeatElm = oneRepeatNum_ / sizeof(U);
    uint16_t hFactor = static_cast<uint16_t>(repeatElm / outInfo.width);
    uint16_t loopH = static_cast<uint16_t>(outInfo.height / hFactor);
    uint16_t tailH = static_cast<uint16_t>(outInfo.height - loopH * hFactor);

    U oneLoopStrideH = static_cast<U>(hFactor * paramInfo.stride[1] * padInfo.dstStride[1]);
    U oneLoopOutElements = static_cast<U>(hFactor * outInfo.width);
    uint32_t tailLoopOutElements = static_cast<uint32_t>(tailH * outInfo.width);  
    U depthStrideInub = padInfo.dstStride[2] * paramInfo.dilation[2];
    U rowStrideInub = padInfo.dstStride[1] * paramInfo.dilation[1];
    U colStrideInub = paramInfo.dilation[0];
    uint16_t kD = paramInfo.kSize[2];
    uint16_t kH = paramInfo.kSize[1];
    uint16_t kW = paramInfo.kSize[0];
    for (uint16_t i = 0; i < outInfo.n; i++) {
        for (uint16_t idxD = 0; idxD < outInfo.depth; idxD++) {
            __local_mem__ T* srcLocalAddr = srcAddr + i * padInfo.dstStride[3] + idxD * paramInfo.stride[2]  * padInfo.dstStride[2];
            __local_mem__ Z* dstLocalAddr = dstAddr + i * outInfo.depth * outInfo.height * outInfo.width + idxD * outInfo.height * outInfo.width;
            Pool3DWithOneLoop<T, U, Z, OP_TYPE, OUT_DIV, USE_TRAIT_TWO>(dstLocalAddr, srcLocalAddr, indexAddr, kD, kH,
                                         kW, depthStrideInub, rowStrideInub, colStrideInub,
                                         oneLoopOutElements, tailLoopOutElements, oneLoopStrideH, loopH, paramInfo.divisor);
        }
    }
    if constexpr (OUT_DIV && OP_TYPE == OP_TYPE_AVG_POOL_3D) {
        __local_mem__ T* newDstAddr = (__local_mem__ T*)maxOutLocal.GetPhyAddr();
        uint32_t totalOut = outInfo.n * outInfo.depth * outInfo.width * outInfo.height;
        DivCompute(newDstAddr, dstAddr, totalOut);
    }    
    inputQue_.FreeTensor<T>(inLocal);
    maxUBOutput_.EnQue<T>(maxOutLocal);
}

template <typename T, int32_t OP_TYPE, bool OUT_DIV>
template <typename U, bool USE_TRAIT_TWO>
__aicore__ inline void Pool3DNcdhwSmallKernelPad<T, OP_TYPE, OUT_DIV>::ComputeSingleRow(const ShapeInfo &outInfo, const Pool3dParam &paramInfo, const CopyPad::CopyPadShapeInfo &padInfo)
{
    LocalTensor<T> maxOutLocal = maxUBOutput_.AllocTensor<T>();
    LocalTensor<T> inLocal = inputQue_.DeQue<T>();
    LocalTensor<T> xLocal = tmpBuf_.Get<T>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __local_mem__ T* srcAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    using Z = typename std::conditional<sizeof(T)==B16 && OUT_DIV && OP_TYPE == OP_TYPE_AVG_POOL_3D, float32_t,
                                    T>::type;   
    __local_mem__ Z* dstAddr = (__local_mem__ Z*)(maxOutLocal.template ReinterpretCast<Z>().GetPhyAddr());
    
    LocalTensor<U> scatterLocal = scatterIndexBuf_.Get<U>();
    if constexpr(OP_TYPE == Pool3D::OP_TYPE_MAX_POOL_3D) {
        CopyPad::CopyAndPad<T, U, CopyPad::DEFAULT_MODE_NEG_INF>(inLocal, xLocal, scatterLocal, padInfo, copyMode_);
    } else {
        CopyPad::CopyAndPad<T, U, CopyPad::DEFAULT_MODE_ZERO>(inLocal, xLocal, scatterLocal, padInfo, copyMode_);
    }
    uint32_t repeatElm = oneRepeatNum_ / sizeof(U);
    uint16_t wFactor = static_cast<uint16_t>(repeatElm);
    uint16_t loopW = static_cast<uint16_t>(outInfo.width / wFactor);
    uint16_t tailW = static_cast<uint16_t>(outInfo.width - loopW * wFactor);

    U oneLoopStrideW = static_cast<U>(wFactor * paramInfo.stride[0] * padInfo.dstStride[0]);
    U oneLoopOutElements = static_cast<U>(wFactor);
    uint32_t tailLoopOutElements = static_cast<uint32_t>(tailW);  
    U depthStrideInub = padInfo.dstStride[2] * paramInfo.dilation[2];
    U rowStrideInub = padInfo.dstStride[1] * paramInfo.dilation[1];
    U colStrideInub = paramInfo.dilation[0];
    uint16_t kD = paramInfo.kSize[2];
    uint16_t kH = paramInfo.kSize[1];
    uint16_t kW = paramInfo.kSize[0];
    
    for (uint16_t i = 0; i < outInfo.n; i++) {
        for (uint16_t idxD = 0; idxD < outInfo.depth; idxD++) {
            for (uint16_t idxH = 0; idxH < outInfo.height; idxH++) {
                __local_mem__ T* srcLocalAddr = srcAddr + i * padInfo.dstStride[3] + idxD * paramInfo.stride[2] * padInfo.dstStride[2] + idxH * paramInfo.stride[1]  * padInfo.dstStride[1];
                __local_mem__ Z* dstLocalAddr = dstAddr + i * outInfo.depth * outInfo.height * outInfo.width + idxD * outInfo.height * outInfo.width + idxH * outInfo.width;
                Pool3DWithOneLoop<T, U, Z, OP_TYPE, OUT_DIV, USE_TRAIT_TWO>(dstLocalAddr, srcLocalAddr, indexAddr, kD, kH,
                                  kW, depthStrideInub, rowStrideInub, colStrideInub,
                                  oneLoopOutElements, tailLoopOutElements, oneLoopStrideW, loopW, paramInfo.divisor);
            }
        }
    }
    if constexpr (OUT_DIV && OP_TYPE == OP_TYPE_AVG_POOL_3D) {
        __local_mem__ T* newDstAddr = (__local_mem__ T*)maxOutLocal.GetPhyAddr();
        uint32_t totalOut = outInfo.n * outInfo.depth * outInfo.width * outInfo.height;
        DivCompute(newDstAddr, dstAddr, totalOut);
    }    
    inputQue_.FreeTensor<T>(inLocal);
    maxUBOutput_.EnQue<T>(maxOutLocal);
}

template <typename T, int32_t OP_TYPE, bool OUT_DIV>
__aicore__ inline void Pool3DNcdhwSmallKernelPad<T, OP_TYPE, OUT_DIV>::ComputeDivisor(int64_t start, int64_t num)   {

    Pool3D::CalcDivisorParam param = {
        tilingData_->kD, tilingData_->kH, tilingData_->kW,
        tilingData_->sD, tilingData_->sH, tilingData_->sW,
        tilingData_->frontPad, tilingData_->backendPad, tilingData_->topPad,
        tilingData_->bottomPad, tilingData_->leftPad, tilingData_->rightPad,
        tilingData_->dOutDim, tilingData_->hOutDim, tilingData_->wOutDim,
        tilingData_->dInDim, tilingData_->hInDim, tilingData_->wInDim
    };
    LocalTensor<float> divisorLocal = divisorBuf_.Get<float>();
    auto dstAddr = (__local_mem__ float*)divisorLocal.GetPhyAddr();
    // 0b000  -> (int32/int64, includepad/no_include, need_clac_multi_batch/no_need)
    ComputeDivisorCommon(tilingData_->divisorMode, dstAddr, param, start, num);
}


template <typename T, int32_t OP_TYPE, bool OUT_DIV>
__aicore__ inline void Pool3DNcdhwSmallKernelPad<T, OP_TYPE, OUT_DIV>::DivCompute(__local_mem__ T* dstAddr, __local_mem__ float32_t* srcAddr, uint32_t num)  
{
    if constexpr (OUT_DIV && OP_TYPE == OP_TYPE_AVG_POOL_3D) {
        LocalTensor<float> divisorLocal = divisorBuf_.Get<float>();
        auto divAddr = (__local_mem__ float*)divisorLocal.GetPhyAddr();
        if (tilingData_->splitMode == SPLIT_BATCHS) {
            uint32_t batchElement = tilingData_->dOutDim * tilingData_->hOutDim * tilingData_->wOutDim;
            uint32_t oneVL = platform::GetVRegSize() / sizeof(float32_t);
            uint32_t batch = num / batchElement;
            AvgPoolDivMultiBatch(dstAddr, srcAddr, divAddr, batch, batchElement);
        } else if (tilingData_->realCalcDivisor == 0) {
            auto curDivAddr = divAddr + curOffsetInBatch_;
            AvgPoolDivNorm(dstAddr, srcAddr, curDivAddr, num);
        } else {
            ComputeDivisor(curOffsetInBatch_, num);
            AvgPoolDivNorm(dstAddr, srcAddr, divAddr, num);
        }
    }
}
}  // namespace Pool3D
#endif  // POOL_3D_SMALL_KERNEL_PAD_H_
