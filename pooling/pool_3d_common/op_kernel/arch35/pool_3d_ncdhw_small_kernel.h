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
 * \file pool_3d_small_kernel.h
 * \brief
 */
#ifndef POOL_3D_NCDHW_SMALL_KERNEL_H_
#define POOL_3D_NCDHW_SMALL_KERNEL_H_

#include "pool_3d_common.h"
#include "gather_index_impl.h"

#include "../inc/platform.h"
#include "../inc/kernel_utils.h"

namespace Pool3D
{
using namespace AscendC;

template <typename T, int32_t OP_TYPE, bool IS_SPARSE=false>
class Pool3DNcdhwSmallKernel
{
public:
    __aicore__ inline Pool3DNcdhwSmallKernel(TPipe* pipe, const Pool3DNcdhwSmallKernelTilingData* __restrict tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y);
    __aicore__ inline void Process();

private:
    template <typename U, int32_t GATHER_MODE>
    __aicore__ inline void BaseCompute();
    __aicore__ inline void SparseCopyInMultiRows(int64_t offset, int64_t nout, int64_t dout, int64_t hout, int64_t wout);
    __aicore__ inline void CopyInMultiRows(int64_t offset, const TensorDescInfo &inputInfo, int32_t splitMode);
    __aicore__ inline void CopyMaxOut(int64_t offset, int64_t blockLen);
    template <typename U, bool USE_TRAIT_TWO>
    __aicore__ inline void ComputeMultiBatch(const TensorDescInfo &inputInfo, const ShapeInfo &outInfo, const Pool3dParam &paramInfo);
    template <typename U, bool USE_TRAIT_TWO>
    __aicore__ inline void ComputeMultiDepth(const TensorDescInfo &inputInfo, const ShapeInfo &outInfo, const Pool3dParam &paramInfo);
    template <typename U, bool USE_TRAIT_TWO>
    __aicore__ inline void ComputeMultiRow(const TensorDescInfo &inputInfo, const ShapeInfo &outInfo, const Pool3dParam &paramInfo);
    template <typename U, bool USE_TRAIT_TWO>
    __aicore__ inline void ComputeSingleRow(const TensorDescInfo &inputInfo, const ShapeInfo &outInfo, const Pool3dParam &paramInfo);
    template <typename U, int32_t GATHER_MODE, bool USE_TRAIT_TWO>
    __aicore__ inline void Compute(const TensorDescInfo &inputInfo, const ShapeInfo &outInfo, const Pool3dParam &paramInfo);
    template <typename U, int32_t GATHER_MODE>
    __aicore__ inline void GenGatherIndex(const GatherIndexImpl::ShapeInfo &param, LocalTensor<U>& indexLocal, uint16_t loopNum=1);
    __aicore__ inline void GenUbArrange();
    __aicore__ inline void GenSparseUbArrange(uint32_t &ubStrideW, uint32_t &ubStrideH, uint32_t &ubStrideD);
    __aicore__ inline void SparseHDCopyParam(MultiCopyLoopInfo<Pool3D::FIVE> &loopInfo, int64_t nout, int64_t dout, int64_t hout, int64_t wout);
    __aicore__ inline void SparseWHCopyParam(MultiCopyLoopInfo<Pool3D::FIVE> &loopInfo, int64_t nout, int64_t dout, int64_t hout, int64_t wout);
    __aicore__ inline void SparseWDCopyParam(MultiCopyLoopInfo<Pool3D::FIVE> &loopInfo, int64_t nout, int64_t dout, int64_t hout, int64_t wout);
    __aicore__ inline void SparseWCopyParam(MultiCopyLoopInfo<Pool3D::FIVE> &loopInfo, int64_t nout, int64_t dout, int64_t hout, int64_t wout);
    __aicore__ inline void SparseHCopyParam(MultiCopyLoopInfo<Pool3D::FIVE> &loopInfo, int64_t nout, int64_t dout, int64_t hout, int64_t wout);
    __aicore__ inline void SparseDCopyParam(MultiCopyLoopInfo<Pool3D::FIVE> &loopInfo, int64_t nout, int64_t dout, int64_t hout, int64_t wout);
    __aicore__ inline void SetSparseParam(MultiCopyLoopInfo<Pool3D::FIVE> &loopInfo, int64_t nout, int64_t dout, int64_t hout, int64_t wout);
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
    const Pool3DNcdhwSmallKernelTilingData* tilingData_;
    uint32_t eleInBlk_ = static_cast<uint32_t>(platform::GetUbBlockSize() / sizeof(T));
    uint32_t oneRepeatNum_ = platform::GetVRegSize();
    uint32_t effectiveKd_ = 1;
    uint32_t effectiveKh_ = 1;
    uint32_t effectiveKw_ = 1;
    bool isSparseW_ = false;
    bool isSparseH_ = false;
    bool isSparseD_ = false;
    uint32_t oneRowElements_ = 1;
    uint32_t onePlaneElements_ = 1;
    uint32_t oneBatchElements_ = 1;
    uint32_t channel_ = 1;
};

template <typename T, int32_t OP_TYPE, bool IS_SPARSE>
__aicore__ inline void Pool3DNcdhwSmallKernel<T, OP_TYPE, IS_SPARSE>::Init(GM_ADDR x, GM_ADDR y)
{
    // GM
    if (tilingData_->useTraiTwo) {
        oneRepeatNum_ = 2 * platform::GetVRegSize();
    }
    channel_ = tilingData_->channel; 
    oneRepeatNum_ = oneRepeatNum_ / channel_;
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    maxGm_.SetGlobalBuffer((__gm__ T*)y);

    pipe_->InitBuffer(inputQue_, BUFFER_NUM, tilingData_->inUbSize * sizeof(T));
    pipe_->InitBuffer(maxUBOutput_, BUFFER_NUM, tilingData_->outUbSize * sizeof(T));
    pipe_->InitBuffer(indexBuf_, 2 * INDEX_SIZE);
    effectiveKd_ = (tilingData_->kD - 1) * tilingData_->dD + 1;
    effectiveKh_ = (tilingData_->kH - 1) * tilingData_->dH + 1;
    effectiveKw_ = (tilingData_->kW - 1) * tilingData_->dW + 1;
    isSparseW_ = tilingData_->sparseMode & SPARSE_W;
    isSparseH_ = tilingData_->sparseMode & SPARSE_H;
    isSparseD_ = tilingData_->sparseMode & SPARSE_D;
}

template <typename T, int32_t OP_TYPE, bool IS_SPARSE>
__aicore__ inline void Pool3DNcdhwSmallKernel<T, OP_TYPE, IS_SPARSE>::Process()
{
    using indiceType = typename IndexTypeGet<T>::type;
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

template <typename T, int32_t OP_TYPE, bool IS_SPARSE>
__aicore__ inline void Pool3DNcdhwSmallKernel<T, OP_TYPE, IS_SPARSE>::GenUbArrange()
{
    uint32_t sW = tilingData_->sW;
    uint32_t sH = tilingData_->sH;
    uint32_t sD = tilingData_->sD;
    uint32_t outUbFactorW = tilingData_->outUbFactorW;
    uint32_t outUbFactorH = tilingData_->outUbFactorH;
    uint32_t outUbFactorD = tilingData_->outUbFactorD;
    oneRowElements_ = ((outUbFactorW - 1) * sW + effectiveKw_) * channel_;
    onePlaneElements_ = oneRowElements_ * ((outUbFactorH - 1) * sH + effectiveKh_);
    oneBatchElements_ = onePlaneElements_ * ((outUbFactorD - 1) * sD + effectiveKd_);
    if (tilingData_->splitMode == SPLIT_COLS) {
        onePlaneElements_ = ops::Aligned(onePlaneElements_, eleInBlk_);
        oneBatchElements_ = onePlaneElements_ * ((outUbFactorD - 1) * sD + effectiveKd_);
    } else if (tilingData_->splitMode == SPLIT_ROWS) {
        oneRowElements_ = tilingData_->wInDim * channel_;
        onePlaneElements_ = oneRowElements_ * ((outUbFactorH - 1) * sH + effectiveKh_);
        oneBatchElements_ = ops::Aligned(onePlaneElements_ * ((outUbFactorD - 1) * sD + effectiveKd_), eleInBlk_);
    } else if (tilingData_->splitMode == SPLIT_DEPTHS) {
        oneRowElements_ = tilingData_->wInDim * channel_;
        onePlaneElements_ = oneRowElements_ * tilingData_->hInDim;
        oneBatchElements_ = onePlaneElements_ * ((outUbFactorD - 1) * sD + effectiveKd_);
    } else {
        oneRowElements_ = tilingData_->wInDim * channel_;
        onePlaneElements_ = oneRowElements_ * tilingData_->hInDim;
        oneBatchElements_ = onePlaneElements_ * tilingData_->dInDim;
    }
}

template <typename T, int32_t OP_TYPE, bool IS_SPARSE>
__aicore__ inline void Pool3DNcdhwSmallKernel<T, OP_TYPE, IS_SPARSE>::GenSparseUbArrange(uint32_t &ubStrideW, uint32_t &ubStrideH, uint32_t &ubStrideD)
{
    uint32_t sW = tilingData_->sW;
    uint32_t sH = tilingData_->sH;
    uint32_t sD = tilingData_->sD;
    uint32_t outUbFactorW = tilingData_->outUbFactorW;
    uint32_t outUbFactorH = tilingData_->outUbFactorH;
    uint32_t outUbFactorD = tilingData_->outUbFactorD;
    if (isSparseH_ && isSparseD_) {
        // sparseH sparseD
        ubStrideH = effectiveKh_;
        ubStrideD = effectiveKd_;
        oneRowElements_ = tilingData_->wInDim * channel_;
        onePlaneElements_ = oneRowElements_ * tilingData_->outUbFactorH * effectiveKh_;
        oneBatchElements_ = onePlaneElements_ * tilingData_->outUbFactorD * effectiveKd_;
        return;        
    }
    uint32_t inputW = (outUbFactorW - 1) * sW + effectiveKw_;
    uint32_t inputH = (outUbFactorH - 1) * sH + effectiveKh_;
    uint32_t inputD = (outUbFactorD - 1) * sD + effectiveKd_;
    if (isSparseW_) {
        ubStrideW = effectiveKw_;
        inputW = tilingData_->outUbFactorW * effectiveKw_;        
    }

    if (isSparseH_) {
        ubStrideH = effectiveKh_;
        inputH = tilingData_->outUbFactorH * effectiveKh_;       
    }

    if (isSparseD_) {
        ubStrideD = effectiveKd_;
        inputD = tilingData_->outUbFactorD * effectiveKd_;      
    }

    oneRowElements_ = inputW * channel_;
    onePlaneElements_ = inputH * inputW * channel_;
    oneBatchElements_ = inputD * inputH * inputW * channel_;
}

template <typename T, int32_t OP_TYPE, bool IS_SPARSE>
template <typename U, int32_t GATHER_MODE>
__aicore__ inline void Pool3DNcdhwSmallKernel<T, OP_TYPE, IS_SPARSE>::BaseCompute()
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

    uint32_t ubStrideW = tilingData_->sW;
    uint32_t ubStrideH = tilingData_->sH;
    uint32_t ubStrideD = tilingData_->sD;
    if constexpr (IS_SPARSE) {
        GenSparseUbArrange(ubStrideW, ubStrideH, ubStrideD);
    } else {
        GenUbArrange();
    }
    GatherIndexImpl::ShapeInfo info = {
        {1, channel_, oneRowElements_, onePlaneElements_, oneBatchElements_},     //input stride
        {channel_, outUbFactorW, outUbFactorH, outUbFactorD, static_cast<uint32_t>(tilingData_->ubFactorN)},    // gather size
        {1, ubStrideW, ubStrideH, ubStrideD, 1}   //stride
    };

    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    GenGatherIndex<U, GATHER_MODE>(info, indexLocal, 2);
    Pool3dParam paramInfo = {
        {static_cast<uint16_t>(tilingData_->kW), static_cast<uint16_t>(tilingData_->kH), static_cast<uint16_t>(tilingData_->kD)},
        {static_cast<uint16_t>(ubStrideW), static_cast<uint16_t>(ubStrideH), static_cast<uint16_t>(ubStrideD)},
        {static_cast<uint16_t>(tilingData_->dW), static_cast<uint16_t>(tilingData_->dH), static_cast<uint16_t>(tilingData_->dD)},
        static_cast<float32_t>(tilingData_->divisor),
    };

    int64_t wInDimWithChannel = tilingData_->wInDim * channel_;
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
        int64_t depthStart = dIdx * sD * outUbFactorD; 
        int64_t rowStart = hIdx * sH * outUbFactorH;
        int64_t colStart = wIdx * sW * outUbFactorW;

        int64_t srcOffset = nIdx * tilingData_->ubFactorN * tilingData_->dInDim * tilingData_->hInDim * tilingData_->wInDim +
                            depthStart * tilingData_->hInDim * tilingData_->wInDim + rowStart * tilingData_->wInDim + colStart;
        int64_t dstOffset = nIdx * tilingData_->ubFactorN * tilingData_->dOutDim * tilingData_->hOutDim * tilingData_->wOutDim +
                            + dIdx * outUbFactorD  * tilingData_->hOutDim * tilingData_->wOutDim 
                            + hIdx * outUbFactorH * tilingData_->wOutDim + wIdx * outUbFactorW;
               
        uint32_t inDepths = tilingData_->splitMode == SPLIT_BATCHS ? tilingData_->dInDim : (depths - 1) * sD + effectiveKd_;
        uint32_t inRows = (tilingData_->splitMode == SPLIT_BATCHS ||  tilingData_->splitMode == SPLIT_DEPTHS) ? tilingData_->hInDim : (rows - 1) * sH + effectiveKh_;
        uint32_t inCols = (tilingData_->splitMode != SPLIT_COLS ? tilingData_->wInDim : (cols - 1) * sW + effectiveKw_) * channel_;
        if constexpr (!IS_SPARSE) {
            if (tilingData_->splitMode == SPLIT_ROWS) {
                onePlaneElements_ = inRows * inCols;
            }else if (tilingData_->splitMode == SPLIT_DEPTHS) {
                oneBatchElements_ = inDepths * inRows * inCols;
            }        
        }  
        TensorDescInfo inputInfo = {
            {inCols, inRows, inDepths, n},
            {1, inCols, onePlaneElements_, oneBatchElements_},
            {1, wInDimWithChannel, wInDimWithChannel * tilingData_->hInDim, tilingData_->dInDim * tilingData_->hInDim * wInDimWithChannel}
        };
        if constexpr (IS_SPARSE) {
            SparseCopyInMultiRows(srcOffset * channel_, n, depths, rows, cols);
            inputInfo.dstStride[1] = oneRowElements_;
            inputInfo.dstStride[2] = onePlaneElements_;
            inputInfo.dstStride[3] = oneBatchElements_;
        } else {
            CopyInMultiRows(srcOffset * channel_, inputInfo, tilingData_->splitMode);
        }        
        
        ShapeInfo outInfo = {static_cast<uint16_t>(n), static_cast<uint16_t>(depths),static_cast<uint16_t>(rows),static_cast<uint16_t>(cols)};
        if (tilingData_->useTraiTwo) {
            Compute<U, GATHER_MODE, true>(inputInfo, outInfo, paramInfo);
        } else {
            Compute<U, GATHER_MODE, false>(inputInfo, outInfo, paramInfo);
        }
        CopyMaxOut(dstOffset*channel_, n*depths*rows*cols*channel_);
    }
}

template <typename T, int32_t OP_TYPE, bool IS_SPARSE>
__aicore__ inline void Pool3DNcdhwSmallKernel<T, OP_TYPE, IS_SPARSE>::SparseCopyInMultiRows(int64_t offset, int64_t nout, int64_t dout, int64_t hout, int64_t wout)
{
    LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();

    MultiCopyLoopInfo<Pool3D::FIVE> loopInfo;

    SetSparseParam(loopInfo, nout, dout, hout, wout);

    static constexpr MultiCopyConfig config = {false};
    MultiCopyParams<T, Pool3D::FIVE> paramsMain = {loopInfo};
    DataCopy<T, Pool3D::FIVE, config>(xLocal, xGm_[offset], paramsMain);
    inputQue_.EnQue(xLocal);
}

template <typename T, int32_t OP_TYPE, bool IS_SPARSE>
__aicore__ inline void Pool3DNcdhwSmallKernel<T, OP_TYPE, IS_SPARSE>::CopyInMultiRows(int64_t offset, const TensorDescInfo &inputInfo, int32_t splitMode)
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
        extParams.dstStride = (inputInfo.dstStride[1] - inputInfo.size[0]) * sizeof(T) / platform::GetUbBlockSize();
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
        extParams.blockLen = inputInfo.size[0] * inputInfo.size[1] * sizeof(T);
        extParams.srcStride = (inputInfo.srcStride[2] - inputInfo.size[0] * inputInfo.size[1] ) * sizeof(T);
        extParams.dstStride = 0;
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
        extParams.srcStride = (inputInfo.srcStride[3] - inputInfo.size[0] * inputInfo.size[1] * inputInfo.size[2] ) * sizeof(T);
        extParams.dstStride = 0;
        DataCopyPad<T, PaddingMode::Compact>(xLocal, xGm_[offset], extParams, padExtParams);
    }  else {
        DataCopyExtParams extParams;
        extParams.blockCount = 1;
        extParams.blockLen = inputInfo.size[0] * inputInfo.size[1] * inputInfo.size[2] * inputInfo.size[3] * sizeof(T);
        extParams.srcStride = 0;
        extParams.dstStride = 0;
        DataCopyPad<T, PaddingMode::Compact>(xLocal, xGm_[offset], extParams, padExtParams);
    }
    inputQue_.EnQue(xLocal);
}

template <typename T, int32_t OP_TYPE, bool IS_SPARSE>
__aicore__ inline void Pool3DNcdhwSmallKernel<T, OP_TYPE, IS_SPARSE>::CopyMaxOut(int64_t offset, int64_t blockLen)
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

template <typename T, int32_t OP_TYPE, bool IS_SPARSE>
template <typename U, int32_t GATHER_MODE>
__aicore__ inline void Pool3DNcdhwSmallKernel<T, OP_TYPE, IS_SPARSE>::GenGatherIndex(const GatherIndexImpl::ShapeInfo &param,
                                                                LocalTensor<U>& indexLocal, uint16_t loopNum)
{
    if constexpr (GATHER_MODE == GATHER_SINGLE_ROW) {
        GatherIndexImpl::GenGatherIndex<U, 2>(param, indexLocal, loopNum);
    } else if constexpr (GATHER_MODE == GATHER_MULTI_ROW) {
        GatherIndexImpl::GenGatherIndex<U, 3>(param, indexLocal, loopNum);
    } else if constexpr (GATHER_MODE == GATHER_MULTI_PLANE) {
        GatherIndexImpl::GenGatherIndex<U, 4>(param, indexLocal, loopNum);
    } else {
        GatherIndexImpl::GenGatherIndex<U, 5>(param, indexLocal, loopNum);
    }
}

template <typename T, int32_t OP_TYPE, bool IS_SPARSE>
template <typename U, int32_t GATHER_MODE, bool USE_TRAIT_TWO>
__aicore__ inline void Pool3DNcdhwSmallKernel<T, OP_TYPE, IS_SPARSE>::Compute(const TensorDescInfo &inputInfo, const ShapeInfo &outInfo, const Pool3dParam &paramInfo)
{
    if constexpr (GATHER_MODE == GATHER_SINGLE_ROW) {
        ComputeSingleRow<U, USE_TRAIT_TWO>(inputInfo, outInfo, paramInfo);
    } else if constexpr (GATHER_MODE == GATHER_MULTI_ROW) {
        ComputeMultiRow<U, USE_TRAIT_TWO>(inputInfo, outInfo, paramInfo);
    } else if constexpr (GATHER_MODE == GATHER_MULTI_PLANE) {
        ComputeMultiDepth<U, USE_TRAIT_TWO>(inputInfo, outInfo, paramInfo);
    } else {
        ComputeMultiBatch<U, USE_TRAIT_TWO>(inputInfo, outInfo, paramInfo);
    }
}

template <typename T, int32_t OP_TYPE, bool IS_SPARSE>
template <typename U, bool USE_TRAIT_TWO>
__aicore__ inline void Pool3DNcdhwSmallKernel<T, OP_TYPE, IS_SPARSE>::ComputeMultiBatch(const TensorDescInfo &inputInfo, const ShapeInfo &outInfo, const Pool3dParam &paramInfo)
{
    LocalTensor<T> maxOutLocal = maxUBOutput_.AllocTensor<T>();
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __local_mem__ T* xLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ T* dstLocalAddr = (__local_mem__ T*)maxOutLocal.GetPhyAddr();

    uint32_t repeatElm = oneRepeatNum_ / sizeof(U);
    uint16_t nFactor = static_cast<uint16_t>(repeatElm / (outInfo.depth * outInfo.width * outInfo.height));
    uint16_t loopN = static_cast<uint16_t>(outInfo.n / nFactor);
    uint16_t tailN = static_cast<uint16_t>(outInfo.n - loopN * nFactor);

    U oneLoopStrideBtach = static_cast<U>(nFactor * inputInfo.dstStride[3]);
    U oneLoopOutElements = static_cast<U>(nFactor * outInfo.depth * outInfo.width * outInfo.height * channel_);

    uint32_t tailLoopOutElements = static_cast<uint32_t>(tailN * outInfo.depth * outInfo.width * outInfo.height * channel_);

    U depthStrideInub = inputInfo.dstStride[2] * paramInfo.dilation[2];
    U rowStrideInub = inputInfo.dstStride[1] * paramInfo.dilation[1];
    U colStrideInub = paramInfo.dilation[0] * channel_;
    uint16_t kD = paramInfo.kSize[2];
    uint16_t kH = paramInfo.kSize[1];
    uint16_t kW = paramInfo.kSize[0];
    Pool3DWithOneLoop<T, U, T, OP_TYPE, false, USE_TRAIT_TWO>(dstLocalAddr, xLocalAddr, indexAddr, kD, kH,
                                kW, depthStrideInub, rowStrideInub, colStrideInub,
                                oneLoopOutElements, tailLoopOutElements, oneLoopStrideBtach, loopN, paramInfo.divisor);

    inputQue_.FreeTensor<T>(xLocal);
    maxUBOutput_.EnQue<T>(maxOutLocal);
}

template <typename T, int32_t OP_TYPE, bool IS_SPARSE>
template <typename U, bool USE_TRAIT_TWO>
__aicore__ inline void Pool3DNcdhwSmallKernel<T, OP_TYPE, IS_SPARSE>::ComputeMultiDepth(const TensorDescInfo &inputInfo, const ShapeInfo &outInfo, const Pool3dParam &paramInfo)
{
    LocalTensor<T> maxOutLocal = maxUBOutput_.AllocTensor<T>();
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __local_mem__ T* srcAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ T* dstAddr = (__local_mem__ T*)maxOutLocal.GetPhyAddr();

    uint32_t repeatElm = oneRepeatNum_ / sizeof(U);
    uint16_t dFactor = static_cast<uint16_t>(repeatElm / (outInfo.width * outInfo.height));
    uint16_t loopD = static_cast<uint16_t>(outInfo.depth / dFactor);
    uint16_t tailD = static_cast<uint16_t>(outInfo.depth - loopD * dFactor);

    U oneLoopStrideDepth = static_cast<U>(dFactor * paramInfo.stride[2] * inputInfo.dstStride[2]);
    U oneLoopOutElements = static_cast<U>(dFactor * outInfo.width * outInfo.height * channel_);

    uint32_t tailLoopOutElements = static_cast<uint32_t>(tailD * outInfo.width * outInfo.height * channel_);

    U depthStrideInub = inputInfo.dstStride[2] * paramInfo.dilation[2];
    U rowStrideInub = inputInfo.dstStride[1] * paramInfo.dilation[1];
    U colStrideInub = paramInfo.dilation[0] * channel_;
    uint16_t kD = paramInfo.kSize[2];
    uint16_t kH = paramInfo.kSize[1];
    uint16_t kW = paramInfo.kSize[0];
    for (uint16_t i = 0; i < outInfo.n; i ++) {
        __local_mem__ T* srcLocalAddr = srcAddr + i * inputInfo.dstStride[3] ;
        __local_mem__ T* dstLocalAddr = dstAddr + i * outInfo.depth * outInfo.height * outInfo.width * channel_;
        Pool3DWithOneLoop<T, U, T, OP_TYPE, false, USE_TRAIT_TWO>(dstLocalAddr, srcLocalAddr, indexAddr, kD, kH,
                                    kW, depthStrideInub, rowStrideInub, colStrideInub,
                                    oneLoopOutElements, tailLoopOutElements, oneLoopStrideDepth, loopD, paramInfo.divisor);        
    }


    inputQue_.FreeTensor<T>(xLocal);
    maxUBOutput_.EnQue<T>(maxOutLocal);
}

template <typename T, int32_t OP_TYPE, bool IS_SPARSE>
template <typename U, bool USE_TRAIT_TWO>
__aicore__ inline void Pool3DNcdhwSmallKernel<T, OP_TYPE, IS_SPARSE>::ComputeMultiRow(const TensorDescInfo &inputInfo, const ShapeInfo &outInfo, const Pool3dParam &paramInfo)
{
    LocalTensor<T> maxOutLocal = maxUBOutput_.AllocTensor<T>();
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __local_mem__ T* srcAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ T* dstAddr = (__local_mem__ T*)maxOutLocal.GetPhyAddr();
    uint32_t repeatElm = oneRepeatNum_ / sizeof(U);
    uint16_t hFactor = static_cast<uint16_t>(repeatElm / outInfo.width);
    uint16_t loopH = static_cast<uint16_t>(outInfo.height / hFactor);
    uint16_t tailH = static_cast<uint16_t>(outInfo.height - loopH * hFactor);

    U oneLoopStrideH = static_cast<U>(hFactor * paramInfo.stride[1] * inputInfo.dstStride[1]);
    U oneLoopOutElements = static_cast<U>(hFactor * outInfo.width * channel_);
    uint32_t tailLoopOutElements = static_cast<uint32_t>(tailH * outInfo.width * channel_);  
    U depthStrideInub = inputInfo.dstStride[2] * paramInfo.dilation[2];
    U rowStrideInub = inputInfo.dstStride[1] * paramInfo.dilation[1];
    U colStrideInub = paramInfo.dilation[0] * channel_;
    uint16_t kD = paramInfo.kSize[2];
    uint16_t kH = paramInfo.kSize[1];
    uint16_t kW = paramInfo.kSize[0];
    for (uint16_t i = 0; i < outInfo.n; i++) {
        for (uint16_t idxD = 0; idxD < outInfo.depth; idxD++) {
            __local_mem__ T* srcLocalAddr = srcAddr + i * inputInfo.dstStride[3] + idxD * paramInfo.stride[2]  * inputInfo.dstStride[2];
            __local_mem__ T* dstLocalAddr = dstAddr + (i * outInfo.depth * outInfo.height * outInfo.width + idxD * outInfo.height * outInfo.width) * channel_;
            Pool3DWithOneLoop<T, U, T, OP_TYPE, false, USE_TRAIT_TWO>(dstLocalAddr, srcLocalAddr, indexAddr, kD, kH,
                                         kW, depthStrideInub, rowStrideInub, colStrideInub,
                                         oneLoopOutElements, tailLoopOutElements, oneLoopStrideH, loopH, paramInfo.divisor);
        }
    }
    inputQue_.FreeTensor<T>(xLocal);
    maxUBOutput_.EnQue<T>(maxOutLocal);
}

template <typename T, int32_t OP_TYPE, bool IS_SPARSE>
template <typename U, bool USE_TRAIT_TWO>
__aicore__ inline void Pool3DNcdhwSmallKernel<T, OP_TYPE, IS_SPARSE>::ComputeSingleRow(const TensorDescInfo &inputInfo, const ShapeInfo &outInfo, const Pool3dParam &paramInfo)
{
    LocalTensor<T> maxOutLocal = maxUBOutput_.AllocTensor<T>();
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    LocalTensor<U> indexLocal = indexBuf_.Get<U>();
    auto indexAddr = (__ubuf__ U*)indexLocal.GetPhyAddr();
    __local_mem__ T* srcAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ T* dstAddr = (__local_mem__ T*)maxOutLocal.GetPhyAddr();
    uint32_t repeatElm = oneRepeatNum_ / sizeof(U);
    uint16_t wFactor = static_cast<uint16_t>(repeatElm);
    uint16_t loopW = static_cast<uint16_t>(outInfo.width / wFactor);
    uint16_t tailW = static_cast<uint16_t>(outInfo.width - loopW * wFactor);

    U oneLoopStrideW = static_cast<U>(wFactor * channel_ * paramInfo.stride[0] * inputInfo.dstStride[0]);
    U oneLoopOutElements = static_cast<U>(wFactor * channel_);
    uint32_t tailLoopOutElements = static_cast<uint32_t>(tailW * channel_);  
    U depthStrideInub = inputInfo.dstStride[2] * paramInfo.dilation[2];
    U rowStrideInub = inputInfo.dstStride[1] * paramInfo.dilation[1];
    U colStrideInub = paramInfo.dilation[0] * channel_;
    uint16_t kD = paramInfo.kSize[2];
    uint16_t kH = paramInfo.kSize[1];
    uint16_t kW = paramInfo.kSize[0];
    for (uint16_t i = 0; i < outInfo.n; i++) {
        for (uint16_t idxD = 0; idxD < outInfo.depth; idxD++) {
            for (uint16_t idxH = 0; idxH < outInfo.height; idxH++) {
                __local_mem__ T* srcLocalAddr = srcAddr + i * inputInfo.dstStride[3] + idxD * paramInfo.stride[2] * inputInfo.dstStride[2] + idxH * paramInfo.stride[1] * inputInfo.dstStride[1];
                __local_mem__ T* dstLocalAddr = dstAddr + (i * outInfo.depth * outInfo.height * outInfo.width + idxD * outInfo.height * outInfo.width + idxH * outInfo.width) * channel_;
                Pool3DWithOneLoop<T, U, T, OP_TYPE, false, USE_TRAIT_TWO>(dstLocalAddr, srcLocalAddr, indexAddr, kD, kH,
                                  kW, depthStrideInub, rowStrideInub, colStrideInub,
                                  oneLoopOutElements, tailLoopOutElements, oneLoopStrideW, loopW, paramInfo.divisor);
            }
        }
    }
    inputQue_.FreeTensor<T>(xLocal);
    maxUBOutput_.EnQue<T>(maxOutLocal);
}

template <typename T, int32_t OP_TYPE, bool IS_SPARSE>
__aicore__ inline void Pool3DNcdhwSmallKernel<T, OP_TYPE, IS_SPARSE>::SparseHDCopyParam(MultiCopyLoopInfo<Pool3D::FIVE> &loopInfo, int64_t n, int64_t dout, int64_t hout, int64_t wout)
{
    uint32_t inputCols = tilingData_->wInDim * channel_;
    loopInfo.loopSize[ZERO] = inputCols * effectiveKh_;
    loopInfo.loopSize[ONE] = hout;
    loopInfo.loopSize[TWO] = effectiveKd_;
    loopInfo.loopSize[THREE] = dout;
    loopInfo.loopSize[FOUR] = n;
    loopInfo.loopSrcStride[ZERO] = 1;
    loopInfo.loopSrcStride[ONE] = tilingData_->sH * inputCols;
    loopInfo.loopSrcStride[TWO] = tilingData_->hInDim * inputCols;
    loopInfo.loopSrcStride[THREE] = tilingData_->sD * tilingData_->hInDim * inputCols;
    loopInfo.loopSrcStride[FOUR] = tilingData_->dInDim * tilingData_->hInDim * inputCols;
    loopInfo.loopDstStride[ZERO] = 1;
    loopInfo.loopDstStride[ONE] = inputCols * effectiveKh_;
    loopInfo.loopDstStride[TWO] = inputCols * effectiveKh_ * tilingData_->outUbFactorH;
    loopInfo.loopDstStride[THREE] = inputCols * effectiveKh_ * tilingData_->outUbFactorH * effectiveKd_; 
    loopInfo.loopDstStride[FOUR] = inputCols * effectiveKh_ * tilingData_->outUbFactorH * effectiveKd_ * tilingData_->outUbFactorD; 
}

template <typename T, int32_t OP_TYPE, bool IS_SPARSE>
__aicore__ inline void Pool3DNcdhwSmallKernel<T, OP_TYPE, IS_SPARSE>::SparseWHCopyParam(MultiCopyLoopInfo<Pool3D::FIVE> &loopInfo, int64_t n, int64_t dout, int64_t hout, int64_t wout)
{
    loopInfo.loopSize[ZERO] = effectiveKw_ * channel_;
    loopInfo.loopSize[ONE] = wout;
    loopInfo.loopSize[TWO] = effectiveKh_;
    loopInfo.loopSize[THREE] = hout;
    loopInfo.loopSize[FOUR] = effectiveKd_;
    loopInfo.loopSrcStride[ZERO] = 1;
    loopInfo.loopSrcStride[ONE] = tilingData_->sW * channel_;
    loopInfo.loopSrcStride[TWO] = tilingData_->wInDim * channel_;
    loopInfo.loopSrcStride[THREE] = tilingData_->sH * tilingData_->wInDim * channel_;
    loopInfo.loopSrcStride[FOUR] = tilingData_->hInDim * tilingData_->wInDim * channel_;
    loopInfo.loopDstStride[ZERO] = 1;
    loopInfo.loopDstStride[ONE] = effectiveKw_ * channel_;
    loopInfo.loopDstStride[TWO] = oneRowElements_;
    loopInfo.loopDstStride[THREE] = oneRowElements_ * effectiveKh_; 
    loopInfo.loopDstStride[FOUR] = onePlaneElements_; 
}

template <typename T, int32_t OP_TYPE, bool IS_SPARSE>
__aicore__ inline void Pool3DNcdhwSmallKernel<T, OP_TYPE, IS_SPARSE>::SparseWDCopyParam(MultiCopyLoopInfo<Pool3D::FIVE> &loopInfo, int64_t n, int64_t dout, int64_t hout, int64_t wout)
{
    loopInfo.loopSize[ZERO] = effectiveKw_ * channel_;
    loopInfo.loopSize[ONE] = wout;
    loopInfo.loopSize[TWO] = (hout - 1) * tilingData_->sH + effectiveKh_;
    loopInfo.loopSize[THREE] = effectiveKd_;
    loopInfo.loopSize[FOUR] = dout;
    loopInfo.loopSrcStride[ZERO] = 1;
    loopInfo.loopSrcStride[ONE] = tilingData_->sW * channel_;
    loopInfo.loopSrcStride[TWO] = tilingData_->wInDim * channel_;
    loopInfo.loopSrcStride[THREE] = tilingData_->hInDim * tilingData_->wInDim * channel_;
    loopInfo.loopSrcStride[FOUR] = tilingData_->sD * tilingData_->hInDim * tilingData_->wInDim * channel_;
    loopInfo.loopDstStride[ZERO] = 1;
    loopInfo.loopDstStride[ONE] = effectiveKw_ * channel_;
    loopInfo.loopDstStride[TWO] = oneRowElements_;
    loopInfo.loopDstStride[THREE] = onePlaneElements_; 
    loopInfo.loopDstStride[FOUR] = onePlaneElements_ * effectiveKd_; 
}

template <typename T, int32_t OP_TYPE, bool IS_SPARSE>
__aicore__ inline void Pool3DNcdhwSmallKernel<T, OP_TYPE, IS_SPARSE>::SparseWCopyParam(MultiCopyLoopInfo<Pool3D::FIVE> &loopInfo, int64_t n, int64_t dout, int64_t hout, int64_t wout)
{
    loopInfo.loopSize[ZERO] = effectiveKw_ * channel_;
    loopInfo.loopSize[ONE] = wout;
    loopInfo.loopSize[TWO] = (hout - 1) * tilingData_->sH + effectiveKh_;
    loopInfo.loopSize[THREE] = (dout - 1) * tilingData_->sD + effectiveKd_;
    loopInfo.loopSize[FOUR] = n;
    loopInfo.loopSrcStride[ZERO] = 1;
    loopInfo.loopSrcStride[ONE] = tilingData_->sW * channel_;
    loopInfo.loopSrcStride[TWO] = tilingData_->wInDim * channel_;
    loopInfo.loopSrcStride[THREE] = tilingData_->hInDim * tilingData_->wInDim * channel_;
    loopInfo.loopSrcStride[FOUR] = tilingData_->dInDim * tilingData_->hInDim * tilingData_->wInDim * channel_;
    loopInfo.loopDstStride[ZERO] = 1;
    loopInfo.loopDstStride[ONE] = effectiveKw_ * channel_;
    loopInfo.loopDstStride[TWO] = oneRowElements_;
    loopInfo.loopDstStride[THREE] = onePlaneElements_; 
    loopInfo.loopDstStride[FOUR] = oneBatchElements_; 
}

template <typename T, int32_t OP_TYPE, bool IS_SPARSE>
__aicore__ inline void Pool3DNcdhwSmallKernel<T, OP_TYPE, IS_SPARSE>::SparseHCopyParam(MultiCopyLoopInfo<Pool3D::FIVE> &loopInfo, int64_t n, int64_t dout, int64_t hout, int64_t wout)
{
    loopInfo.loopSize[ZERO] = ((wout - 1) * tilingData_->sW + effectiveKw_) * channel_;
    loopInfo.loopSize[ONE] = effectiveKh_;
    loopInfo.loopSize[TWO] = hout;
    loopInfo.loopSize[THREE] = (dout - 1) * tilingData_->sD + effectiveKd_;
    loopInfo.loopSize[FOUR] = n;
    loopInfo.loopSrcStride[ZERO] = 1;
    loopInfo.loopSrcStride[ONE] = tilingData_->wInDim * channel_;
    loopInfo.loopSrcStride[TWO] = tilingData_->wInDim * tilingData_->sH * channel_;
    loopInfo.loopSrcStride[THREE] = tilingData_->hInDim * tilingData_->wInDim * channel_;
    loopInfo.loopSrcStride[FOUR] = tilingData_->dInDim * tilingData_->hInDim * tilingData_->wInDim * channel_;
    loopInfo.loopDstStride[ZERO] = 1;
    loopInfo.loopDstStride[ONE] = oneRowElements_;
    loopInfo.loopDstStride[TWO] = oneRowElements_ * effectiveKh_;
    loopInfo.loopDstStride[THREE] = onePlaneElements_; 
    loopInfo.loopDstStride[FOUR] = oneBatchElements_; 
}

template <typename T, int32_t OP_TYPE, bool IS_SPARSE>
__aicore__ inline void Pool3DNcdhwSmallKernel<T, OP_TYPE, IS_SPARSE>::SparseDCopyParam(MultiCopyLoopInfo<Pool3D::FIVE> &loopInfo, int64_t n, int64_t dout, int64_t hout, int64_t wout)
{
    loopInfo.loopSize[ZERO] = ((wout - 1) * tilingData_->sW + effectiveKw_) * channel_;
    loopInfo.loopSize[ONE] = (hout - 1) * tilingData_->sH + effectiveKh_;
    loopInfo.loopSize[TWO] = effectiveKd_;
    loopInfo.loopSize[THREE] = dout;
    loopInfo.loopSize[FOUR] = n;
    loopInfo.loopSrcStride[ZERO] = 1;
    loopInfo.loopSrcStride[ONE] = tilingData_->wInDim * channel_;
    loopInfo.loopSrcStride[TWO] = tilingData_->hInDim * tilingData_->wInDim * channel_;
    loopInfo.loopSrcStride[THREE] = tilingData_->hInDim * tilingData_->wInDim * tilingData_->sD * channel_;
    loopInfo.loopSrcStride[FOUR] = tilingData_->dInDim * tilingData_->hInDim * tilingData_->wInDim * channel_;
    loopInfo.loopDstStride[ZERO] = 1;
    loopInfo.loopDstStride[ONE] = oneRowElements_;
    loopInfo.loopDstStride[TWO] = onePlaneElements_;
    loopInfo.loopDstStride[THREE] = onePlaneElements_ * effectiveKd_; 
    loopInfo.loopDstStride[FOUR] = oneBatchElements_; 
}

template <typename T, int32_t OP_TYPE, bool IS_SPARSE>
__aicore__ inline void Pool3DNcdhwSmallKernel<T, OP_TYPE, IS_SPARSE>::SetSparseParam(MultiCopyLoopInfo<Pool3D::FIVE> &loopInfo, int64_t nout, int64_t dout, int64_t hout, int64_t wout)
{
    int32_t sparseMode = tilingData_->sparseMode;
    switch (sparseMode) {
        case Pool3D::SPARSE_W :
            SparseWCopyParam(loopInfo, nout, dout, hout, wout);
            break;
        case Pool3D::SPARSE_H : 
            SparseHCopyParam(loopInfo, nout, dout, hout, wout);
            break;
        case Pool3D::SPARSE_D :
            SparseDCopyParam(loopInfo, nout, dout, hout, wout);
            break;
        case Pool3D::SPARSE_WH : 
            SparseWHCopyParam(loopInfo, nout, dout, hout, wout);
            break;
        case Pool3D::SPARSE_WD :
            SparseWDCopyParam(loopInfo, nout, dout, hout, wout);
            break;
        case Pool3D::SPARSE_HD : 
            SparseHDCopyParam(loopInfo, nout, dout, hout, wout);
            break;
    }
}

}  // namespace Pool3D
#endif  // POOL_3D_NCDHW_SMALL_KERNEL_H_
