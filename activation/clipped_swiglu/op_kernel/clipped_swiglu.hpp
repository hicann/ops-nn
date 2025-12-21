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
 * \file clipped_swiglu.hpp
 * \brief
 */
#ifndef OPP_CLIPPED_SWIGLU_HPP
#define OPP_CLIPPED_SWIGLU_HPP
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace ClippedSwigluOps {
using namespace AscendC;
constexpr static int64_t DB_BUFFER = 2;
constexpr static int64_t BLOCK_SIZE = 32;
constexpr static int64_t BLOCK_ELEM = BLOCK_SIZE / sizeof(float);
constexpr static int64_t SWI_FACTOR = 2;
constexpr static int64_t DTYPE_FACTOR = 2;

template <typename inType>
class ClippedSwigluBase {
public:
    __aicore__ inline ClippedSwigluBase(TPipe* pipe)
    {
        pipe_ = pipe;
    };
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR groupIndex, GM_ADDR y, const ClippedSwigluTilingData* tilingData);
    __aicore__ inline int64_t AlignBytes(int64_t number);
    __aicore__ inline void Process();
    __aicore__ inline void CalTilingParam();
    __aicore__ inline void ProcessSingleLoop(int64_t xOffset, int64_t yOffset);
    __aicore__ inline void CopyIn(int64_t xOffset);
    __aicore__ inline void CopyInHalfShortH(LocalTensor<inType>& xDTypeLocal, int64_t xOffset);
    __aicore__ inline void CopyInHalfLongH(LocalTensor<inType>& xDTypeLocal, int64_t xOffset);
    __aicore__ inline void CopyInInterLeaved(LocalTensor<inType>& xDTypeLocal, int64_t xOffset);
    __aicore__ inline void Compute(
        LocalTensor<float>& xFloatLocal, LocalTensor<float>& tmpUbF32, LocalTensor<float>& yFloatLocal);
    __aicore__ inline void GetAB(LocalTensor<float>& tmpA, LocalTensor<float>& tmpB, LocalTensor<float>& xFloatLocal);
    __aicore__ inline void CopyOut(int64_t yOffset);

protected:
    /* global memory address */
    GlobalTensor<inType> xGm_;
    GlobalTensor<int64_t> groupIndexGm_;
    GlobalTensor<inType> yGm_;

    /* ascendc variable */
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, DB_BUFFER> xQueue_;
    TQue<QuePosition::VECIN, DB_BUFFER> groupQueue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;
    TBuf<TPosition::VECCALC> tmpBuf1_;
    TBuf<TPosition::VECCALC> tmpBuf2_;

    uint32_t blockIdx_ = GetBlockIdx();
    uint32_t usedCoreNum_ = 0;
    int64_t realBatchSize_ = 0; // 用于计算的BatchSize数
    int64_t blockOffset_ = 0;   // 输入x的核间切分偏移量（单位：个元素，例如1个float元素）
    int64_t loopOffset_ = 0;    // 输入x的核内切分偏移量（单位：个元素）
    int64_t loopTime_ = 0;
    int64_t pairFrontLoop_ = 0;
    int64_t pairLastLoop_ = 0;
    int64_t pairNum_ = 0;
    int64_t batchPreBlock_ = 0; // 当前核处理batch的数量（仅前后排列场景使用）
    int64_t dimH_ = 0;
    int64_t ubMaxPair_ = 0;
    int64_t xLocalOffset1_ = 0; // copyin时，由于数据类型不同产生的目的地址偏移量
    int64_t xLocalOffset2_ = 0; // copyin时，前后排列中搬运第二次的目的地址偏移量
    int64_t xQueSpace_ = 0; // xQue、tmpBuf1_、tmpBuf2_所需存储空间
    int64_t calPairFrontLoop_ = 0; // 用于计算的pair数，对于前后排列小2H场景，需要32字节对齐
    int64_t calPairLastLoop_ = 0;
    int64_t calPairNum_ = 0;
    const ClippedSwigluTilingData* tl_ = nullptr;
};

template <typename inType>
__aicore__ inline void ClippedSwigluBase<inType>::Init(
    GM_ADDR x, GM_ADDR groupIndex, GM_ADDR y, const ClippedSwigluTilingData* tilingData)
{
    tl_ = tilingData;
    ubMaxPair_ = tl_->ubMaxPair;
    dimH_ = tl_->dim2H / SWI_FACTOR;
    // 32字节对齐的标准为dimH * sizeof(bf16)必须32字节对齐，DTYPE_FACTOR = sizeof(float) / sizeof(half)
    xQueSpace_ = SWI_FACTOR * DTYPE_FACTOR * AlignBytes(ubMaxPair_ * static_cast<int64_t>(sizeof(bfloat16_t))); 
    xGm_.SetGlobalBuffer((__gm__ inType*)x);
    yGm_.SetGlobalBuffer((__gm__ inType*)y);
    if (tl_->isGroup == 1) {
        groupIndexGm_.SetGlobalBuffer((__gm__ int64_t*)groupIndex);
        pipe_->InitBuffer(groupQueue_, DB_BUFFER, AlignBytes(tl_->groupNum * static_cast<int64_t>(sizeof(int64_t))));
    }
    pipe_->InitBuffer(xQueue_, DB_BUFFER, xQueSpace_);
    pipe_->InitBuffer(yQueue_, 1, DTYPE_FACTOR * AlignBytes(ubMaxPair_ * static_cast<int64_t>(sizeof(bfloat16_t))));
    pipe_->InitBuffer(tmpBuf1_, xQueSpace_);
    pipe_->InitBuffer(tmpBuf2_, xQueSpace_);
}

template <typename inType>
__aicore__ inline int64_t ClippedSwigluBase<inType>::AlignBytes(int64_t number)
{
    return (number + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
}

template <typename inType>
__aicore__ inline void ClippedSwigluBase<inType>::Process()
{
    // 计算realBatchSize_
    if (tl_->isGroup == 0) {
        realBatchSize_ = tl_->dimBatchSize;
    } else {
        LocalTensor<int64_t> groupLocal = groupQueue_.AllocTensor<int64_t>();
        DataCopyPad(groupLocal, groupIndexGm_, {1, static_cast<uint16_t>(tl_->groupNum * sizeof(int64_t)), 0, 0}, {false, 0, 0, 0});
        groupQueue_.EnQue<int64_t>(groupLocal);
        groupLocal = groupQueue_.DeQue<int64_t>();
        LocalTensor<float> groupFloatLocal = groupLocal.template ReinterpretCast<float>();
        Cast(groupFloatLocal, groupLocal, RoundMode::CAST_RINT, tl_->groupNum);
        LocalTensor<float> groupSumLocal = tmpBuf1_.Get<float>(); 
        ReduceSum<float>(groupSumLocal, groupFloatLocal, groupFloatLocal, tl_->groupNum);
        float groupSum = groupSumLocal.GetValue(0);
        realBatchSize_ = static_cast<int64_t>(groupSum);
        groupQueue_.FreeTensor(groupLocal);
        realBatchSize_ = realBatchSize_ < tl_->dimBatchSize ? realBatchSize_ : tl_->dimBatchSize;
    }
    // 三种情况下计算切分策略以及单次核间核内元素偏移量、每loop处理pair数量
    CalTilingParam();

    // 切分
    if (blockIdx_ < usedCoreNum_) { // 前后排列 大2H场景；两次循环，外层循环batch内层循环dim2H
        int64_t xOffset = 0;
        int64_t yOffset = 0;
        if (tl_->isInterleaved == 0 && tl_->isLongH == 1) {
            for (int64_t batchIdx = 0; batchIdx < batchPreBlock_; ++batchIdx) {
                xOffset = blockIdx_ * blockOffset_ + batchIdx * tl_->dim2H;
                yOffset = blockIdx_ * blockOffset_ / SWI_FACTOR + batchIdx * dimH_;
                for (int64_t loopIdx = 0; loopIdx < loopTime_; ++loopIdx) {
                    pairNum_ = loopIdx == (loopTime_ - 1) ? pairLastLoop_ : pairFrontLoop_;
                    calPairNum_ = loopIdx == (loopTime_ - 1) ? calPairLastLoop_ : calPairFrontLoop_;
                    ProcessSingleLoop(xOffset, yOffset);
                    xOffset += loopOffset_;
                    yOffset += loopOffset_;
                }
            }
        } else { // 前后排列 小2H场景；奇偶排列场景：一次循环
            xOffset = blockIdx_ * blockOffset_;
            yOffset = blockIdx_ * blockOffset_ / SWI_FACTOR;
            for (int64_t loopIdx = 0; loopIdx < loopTime_; ++loopIdx) {
                pairNum_ = loopIdx == (loopTime_ - 1) ? pairLastLoop_ : pairFrontLoop_;
                calPairNum_ = loopIdx == (loopTime_ - 1) ? calPairLastLoop_ : calPairFrontLoop_;
                ProcessSingleLoop(xOffset, yOffset);
                xOffset += loopOffset_;
                yOffset += loopOffset_ / SWI_FACTOR;
            }
        }
    }
}

template <typename inType>
__aicore__ inline void ClippedSwigluBase<inType>::CalTilingParam()
{
    // 前后排列
    if (tl_->isInterleaved == 0) {
        // 核间batch切分
        int64_t batchFrontBlock = (realBatchSize_ + tl_->coreNumAll - 1) / tl_->coreNumAll;
        usedCoreNum_ = (realBatchSize_ + batchFrontBlock - 1) / batchFrontBlock;
        int64_t batchLastBlock = realBatchSize_ - batchFrontBlock * (usedCoreNum_ - 1); 
        batchPreBlock_ = blockIdx_ == (usedCoreNum_ - 1) ? batchLastBlock : batchFrontBlock;
        blockOffset_ = batchFrontBlock * tl_->dim2H;
        if (tl_->isLongH == 0) { // dim2H较小时，核内batch切分
            int64_t batchSpace = SWI_FACTOR * DTYPE_FACTOR * AlignBytes(dimH_ * static_cast<int64_t>(sizeof(bfloat16_t))); // 一个batch32字节对齐所需空间
            int64_t ubMaxBatch = xQueSpace_ / batchSpace; // UB最多可以处理的batch数，即头loop处理batch数
            loopTime_ = (batchPreBlock_ + ubMaxBatch - 1) / ubMaxBatch;
            int64_t batchLastLoop = batchPreBlock_ - ubMaxBatch * (loopTime_ - 1);
            pairFrontLoop_ = ubMaxBatch * dimH_;
            pairLastLoop_ = batchLastLoop * dimH_;
            loopOffset_ = ubMaxBatch * tl_->dim2H;
            calPairFrontLoop_ = ubMaxBatch * batchSpace / SWI_FACTOR / sizeof(float);
            calPairLastLoop_ = batchLastLoop * batchSpace / SWI_FACTOR / sizeof(float);

        } else { // dim2H较大时，核内dim2H切分
            loopTime_ = (dimH_ + ubMaxPair_ - 1) / ubMaxPair_;
            pairLastLoop_ = dimH_ - ubMaxPair_ * (loopTime_ - 1);
            pairFrontLoop_ = ubMaxPair_;
            loopOffset_ = ubMaxPair_;
            calPairFrontLoop_ = pairFrontLoop_;
            calPairLastLoop_ = pairLastLoop_;
        }
    }
    // 奇偶排列
    else {
        // 核间pair切分：
        int64_t pairTotal = tl_->dim2H * realBatchSize_ / SWI_FACTOR; // 需要计算的总pair数
        int64_t pairFrontBlock = (pairTotal + tl_->coreNumAll - 1) / tl_->coreNumAll;
        usedCoreNum_ = (pairTotal + pairFrontBlock - 1) / pairFrontBlock;
        int64_t pairLastBlock = pairTotal - pairFrontBlock * (usedCoreNum_ - 1);
        int64_t pairPreBlock = blockIdx_ == (usedCoreNum_ - 1) ? pairLastBlock : pairFrontBlock;
        blockOffset_ = pairFrontBlock * SWI_FACTOR;
        // 核内pair切分：
        loopTime_ = (pairPreBlock + ubMaxPair_ - 1) / ubMaxPair_; 
        pairLastLoop_ = pairPreBlock - ubMaxPair_ * (loopTime_ - 1);
        pairFrontLoop_ = ubMaxPair_;
        loopOffset_ = SWI_FACTOR * ubMaxPair_;
        calPairFrontLoop_ = pairFrontLoop_;
        calPairLastLoop_ = pairLastLoop_;
    }
}

template <typename inType>
__aicore__ inline void ClippedSwigluBase<inType>::ProcessSingleLoop(int64_t xOffset, int64_t yOffset)
{
    CopyIn(xOffset);
    LocalTensor<inType> xDTypeLocal = xQueue_.DeQue<inType>();
    LocalTensor<float> xFloatLocal = xDTypeLocal.template ReinterpretCast<float>();
#if (ORIG_DTYPE_X != DT_FLOAT)
    if (tl_->isInterleaved == 0) {
        Cast(xFloatLocal, xDTypeLocal[xLocalOffset1_], RoundMode::CAST_NONE, calPairNum_);
        PipeBarrier<PIPE_V>();
        Cast(xFloatLocal[xQueSpace_ / sizeof(float) / SWI_FACTOR], xDTypeLocal[xLocalOffset1_ + xLocalOffset2_], RoundMode::CAST_NONE, calPairNum_);
    }
    else {
        Cast(xFloatLocal, xDTypeLocal[xLocalOffset1_], RoundMode::CAST_NONE, calPairNum_ * SWI_FACTOR);
    }

#endif
    LocalTensor<float> tmpUbF32 = tmpBuf1_.Get<float>(); // 用于装A\B矩阵
    LocalTensor<float> yFloatLocal = yQueue_.AllocTensor<float>();
    Compute(xFloatLocal, tmpUbF32, yFloatLocal);
    LocalTensor<inType> yDTypeLocal = yFloatLocal.template ReinterpretCast<inType>();
#if (ORIG_DTYPE_X == DT_BF16)
    Cast(yDTypeLocal, yFloatLocal, RoundMode::CAST_RINT, calPairNum_);
#endif
#if (ORIG_DTYPE_X == DT_FLOAT16)
    Cast(yDTypeLocal, yFloatLocal, RoundMode::CAST_NONE, calPairNum_);
#endif
    yQueue_.EnQue<inType>(yDTypeLocal);
    CopyOut(yOffset);
}

template <typename inType>
__aicore__ inline void ClippedSwigluBase<inType>::CopyIn(int64_t xOffset)
{
#if (ORIG_DTYPE_X != DT_FLOAT) // 16位的数据DataCopy的时候放在buffer的后面一半，防止后续Cast出错
    xLocalOffset1_ = xQueSpace_ / SWI_FACTOR / static_cast<int64_t>(sizeof(bfloat16_t));
    xLocalOffset2_ = xLocalOffset1_ / SWI_FACTOR; 
#endif
#if (ORIG_DTYPE_X == DT_FLOAT)
    xLocalOffset1_ = 0;
    xLocalOffset2_ = xQueSpace_ / static_cast<int64_t>(sizeof(float)) / SWI_FACTOR;
#endif
    LocalTensor<inType> xDTypeLocal = xQueue_.AllocTensor<inType>();
    if (tl_->isInterleaved == 0 && tl_->isLongH == 0) {
        CopyInHalfShortH(xDTypeLocal, xOffset);
    }
    else if (tl_->isInterleaved == 0 && tl_->isLongH == 1) {
        CopyInHalfLongH(xDTypeLocal, xOffset);
    }
    else {
        CopyInInterLeaved(xDTypeLocal, xOffset);
    }
    xQueue_.EnQue(xDTypeLocal);
}

template <typename inType>
__aicore__ inline void ClippedSwigluBase<inType>::CopyInHalfShortH(LocalTensor<inType>& xDTypeLocal, int64_t xOffset)
{
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams dataCopyXParams;
    dataCopyXParams.blockCount = pairNum_ / dimH_; // 该loop处理的BatchSize数量（已32字节对齐）
    dataCopyXParams.blockLen = dimH_ * sizeof(inType); // 一次搬运一半
    dataCopyXParams.srcStride = dimH_ * sizeof(inType);
    dataCopyXParams.dstStride = 0;
    DataCopyPad(xDTypeLocal[xLocalOffset1_], xGm_[xOffset], dataCopyXParams, padParams);
    DataCopyPad(xDTypeLocal[xLocalOffset1_ + xLocalOffset2_], xGm_[xOffset + dimH_], dataCopyXParams, padParams);
}

template <typename inType>
__aicore__ inline void ClippedSwigluBase<inType>::CopyInHalfLongH(LocalTensor<inType>& xDTypeLocal, int64_t xOffset)
{
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams dataCopyXParams;
    dataCopyXParams.blockCount = 1;
    dataCopyXParams.blockLen = pairNum_ * sizeof(inType);
    dataCopyXParams.srcStride = 0;
    dataCopyXParams.dstStride = 0;
    DataCopyPad(xDTypeLocal[xLocalOffset1_], xGm_[xOffset], dataCopyXParams, padParams);
    DataCopyPad(xDTypeLocal[xLocalOffset1_ + xLocalOffset2_], xGm_[xOffset + dimH_], dataCopyXParams, padParams);
}

template <typename inType>
__aicore__ inline void ClippedSwigluBase<inType>::CopyInInterLeaved(LocalTensor<inType>& xDTypeLocal, int64_t xOffset)
{
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams dataCopyXParams;
    dataCopyXParams.blockCount = 1;
    dataCopyXParams.blockLen = SWI_FACTOR * pairNum_ * sizeof(inType);
    dataCopyXParams.srcStride = 0;
    dataCopyXParams.dstStride = 0;
    DataCopyPad(xDTypeLocal[xLocalOffset1_], xGm_[xOffset], dataCopyXParams, padParams);
}

template <typename inType>
__aicore__ inline void ClippedSwigluBase<inType>::Compute(
    LocalTensor<float>& xFloatLocal, LocalTensor<float>& tmpUbF32, LocalTensor<float>& yFloatLocal)
{
    LocalTensor<float> tmpA = tmpUbF32;
    LocalTensor<float> tmpB = tmpUbF32[xQueSpace_ / sizeof(float) / SWI_FACTOR];
    GetAB(tmpA, tmpB, xFloatLocal);

    Mins(tmpB, tmpB, tl_->gluLimit, calPairNum_);
    PipeBarrier<PIPE_V>();
    Maxs(tmpB, tmpB, -1 * tl_->gluLimit, calPairNum_);
    PipeBarrier<PIPE_V>();
    Adds(tmpB, tmpB, tl_->gluBias, calPairNum_);
    PipeBarrier<PIPE_V>();

    Mins(tmpA, tmpA, tl_->gluLimit, calPairNum_);
    PipeBarrier<PIPE_V>();
    Muls(xFloatLocal, tmpA, -1 * tl_->gluAlpha, calPairNum_);
    PipeBarrier<PIPE_V>();
    Exp(xFloatLocal, xFloatLocal, calPairNum_);
    PipeBarrier<PIPE_V>();
    Adds(xFloatLocal, xFloatLocal, (float)1.0, calPairNum_);
    PipeBarrier<PIPE_V>();
    Div(tmpA, tmpA, xFloatLocal, calPairNum_);
    PipeBarrier<PIPE_V>();
    xQueue_.FreeTensor(xFloatLocal);
    Mul(yFloatLocal, tmpB, tmpA, calPairNum_);
    PipeBarrier<PIPE_V>();
}

template <typename inType>
__aicore__ inline void ClippedSwigluBase<inType>::GetAB(
    LocalTensor<float>& tmpA, LocalTensor<float>& tmpB, LocalTensor<float>& xFloatLocal)
{
    if (tl_->isInterleaved == 0) {
        SetMaskCount();
        SetVectorMask<float, MaskMode::COUNTER>(calPairNum_);
        Copy<float, false>(
            tmpA, xFloatLocal, AscendC::MASK_PLACEHOLDER, 1,
            {1, 1, 0, 0});
        Copy<float, false>(
            tmpB, xFloatLocal[xQueSpace_ / sizeof(float) / SWI_FACTOR], AscendC::MASK_PLACEHOLDER, 1,
            {1, 1, 0, 0});
        SetMaskNorm();
        ResetMask();
    }
    else {
        LocalTensor<int32_t> xOffsetLocalI32 =
            tmpBuf2_.Get<int32_t>();
        ArithProgression(
            xOffsetLocalI32, static_cast<int32_t>(0), static_cast<int32_t>(sizeof(float) * SWI_FACTOR),
            static_cast<int32_t>(ubMaxPair_));
        PipeBarrier<PIPE_V>();
        LocalTensor<uint32_t> xOffsetLocalU32 = xOffsetLocalI32.template ReinterpretCast<uint32_t>();
        Gather(tmpB, xFloatLocal, xOffsetLocalU32, static_cast<uint32_t>(4), pairNum_);
        Gather(tmpA, xFloatLocal, xOffsetLocalU32, static_cast<uint32_t>(0), pairNum_);
        PipeBarrier<PIPE_V>();
    }
}

template <typename inType>
__aicore__ inline void ClippedSwigluBase<inType>::CopyOut(int64_t yOffset)
{
    LocalTensor<inType> yDTypeLocal = yQueue_.DeQue<inType>();
    DataCopyParams dataCopyParams;
    if(tl_->isInterleaved == 0 && tl_->isLongH == 0) { // 前后排列 小2H场景
      dataCopyParams.blockCount = pairNum_ / dimH_;
      dataCopyParams.blockLen = dimH_ * sizeof(inType);
      dataCopyParams.srcStride = 0;
      dataCopyParams.dstStride = 0;
    }
    else { // 奇偶排列；前后排列 大2H场景
      dataCopyParams.blockCount = 1;
      dataCopyParams.blockLen = pairNum_ * sizeof(inType);
      dataCopyParams.srcStride = 0;
      dataCopyParams.dstStride = 0;
    }
    DataCopyPad(yGm_[yOffset], yDTypeLocal, dataCopyParams);
    yQueue_.FreeTensor(yDTypeLocal);
}

} // namespace ClippedSwigluOps
#endif // OPP_CLIPPED_SWIGLU_HPP
