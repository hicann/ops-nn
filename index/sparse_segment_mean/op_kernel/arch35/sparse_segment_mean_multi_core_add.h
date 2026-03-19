/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_segment_mean_multi_core_add.h
 * \brief
 */

#ifndef SPARSE_SEGMENT_MEAN_MULTI_CORE_KERNEL_H_
#define SPARSE_SEGMENT_MEAN_MULTI_CORE_KERNEL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "sparse_segment_mean_struct.h"

namespace SparseSegmentMeanNameSpace
{
constexpr uint32_t DOUBLE_NUM = 2;
constexpr int64_t FLOAT32_SIZE = 4;
constexpr uint32_t GM_NUM_OFFSET_BYTE = 128;

using namespace AscendC;
template <typename T1>
class SparseSegmentMeanMultiCoreKernel
{
public:
    __aicore__ inline SparseSegmentMeanMultiCoreKernel(void){};
    __aicore__ inline void Init(GM_ADDR y, GM_ADDR workspace, AscendC::TPipe& pipeIn,
    const SparseSegmentMeanSimdTilingData* tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void CopyIn(LocalTensor<float>& xLocal, int32_t burstLen, int64_t localOffset, int64_t gmOffset);
    __aicore__ inline void Compute(uint32_t rSize, uint32_t aSize);
    __aicore__ inline void ProcessSegment(int64_t &segmentOffset, 
    int64_t segmentIdsLength);
    __aicore__ inline void CopyOut(int64_t count, int64_t yGmOffset);

    const SparseSegmentMeanSimdTilingData* tilingData_; 

    int32_t blockIdx_;
    int64_t currentCoreElements_ = 0;
    int64_t xQueBlockNum_;

    GlobalTensor<float> sumResultWorkspace_;
    GlobalTensor<int64_t> numResultWorkspace_;
    GlobalTensor<int64_t> segmentIdWorkspace_;

    GlobalTensor<T1> yGm_;
    TQue<QuePosition::VECIN,  1> xQue_;
    TQue<QuePosition::VECOUT,  1> yQue_;
    TBuf<QuePosition::VECCALC> sumTBuf_;
    TBuf<QuePosition::VECCALC> sharedTBuf_;

    constexpr static int32_t BLOCK_SIZE = platform::GetUbBlockSize();
    constexpr static int64_t EIGHT_ALIGN = BLOCK_SIZE / sizeof(float);
};

template <typename T1>
__aicore__ inline void SparseSegmentMeanMultiCoreKernel<T1>::Init(
    GM_ADDR y, GM_ADDR workspace, AscendC::TPipe& pipeIn,
    const SparseSegmentMeanSimdTilingData* tilingData)
{
    tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= tilingData_->usedCoreNumForMulCore) {
        return;
    }
    currentCoreElements_ = tilingData_->perCoreInnerElements;
    if (blockIdx_ == tilingData_->usedCoreNumForMulCore - 1) {
        currentCoreElements_ = tilingData_->tailCoreInnerElements;
    }

    xQueBlockNum_ = tilingData_->inBufferSize / BLOCK_SIZE;
    
    yGm_.SetGlobalBuffer((__gm__ T1*)y + blockIdx_ * tilingData_->perCoreInnerElements);
    sumResultWorkspace_.SetGlobalBuffer((__gm__ float*)workspace + blockIdx_ * tilingData_->perCoreInnerElements);
    numResultWorkspace_.SetGlobalBuffer((__gm__ int64_t*)workspace + tilingData_->indicesOuter * DOUBLE * tilingData_->innerSize / DOUBLE + GM_NUM_OFFSET_BYTE / sizeof(int64_t)); // 局部
    segmentIdWorkspace_.SetGlobalBuffer((__gm__ int64_t*)workspace + tilingData_->indicesOuter * DOUBLE * tilingData_->innerSize / DOUBLE + GM_NUM_OFFSET_BYTE / sizeof(int64_t) + DOUBLE * tilingData_->indicesOuter * (GM_NUM_OFFSET_BYTE / sizeof(int64_t))); // 局部
    
    pipeIn.InitBuffer(xQue_, 1, tilingData_->inBufferSize);
    pipeIn.InitBuffer(yQue_, 1, tilingData_->outBufferSize);
    pipeIn.InitBuffer(sumTBuf_, tilingData_->outBufferSize);
    pipeIn.InitBuffer(sharedTBuf_, tilingData_->sharedTmpBufferSize);
}

template <typename T1>
__aicore__ inline void SparseSegmentMeanMultiCoreKernel<T1>::Compute(uint32_t rSize, uint32_t aSize)
{
    LocalTensor<float> xLocal = xQue_.DeQue<float>();
    LocalTensor<float> sumLocal = sumTBuf_.Get<float>();
    LocalTensor<uint8_t> sharedTmpLocal = sharedTBuf_.Get<uint8_t>();
    uint32_t shape[] = {rSize, aSize};
    constexpr bool isReuse = true;
    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, isReuse>(sumLocal, xLocal, sharedTmpLocal, shape, true);
    xQue_.FreeTensor(xLocal);
}

template <typename T1>
__aicore__ inline void SparseSegmentMeanMultiCoreKernel<T1>::ProcessSegment(int64_t &segmentOffset, 
    int64_t segmentIdsLength)
{
    constexpr int64_t numOffset = GM_NUM_OFFSET_BYTE / sizeof(int64_t);
    int64_t curCopyInNum = 1;
    int64_t segmentId = segmentIdWorkspace_.GetValue(segmentOffset * numOffset);
    if (segmentId == -1) {
        segmentOffset++;
        if (segmentOffset >= segmentIdsLength) {
            return;
        }

        segmentId = segmentIdWorkspace_.GetValue(segmentOffset * numOffset);
    }
    int64_t segmentOffsetStart = segmentOffset;
    int64_t numResultTotal = numResultWorkspace_.GetValue(segmentOffset * numOffset);

    segmentOffset++;
    while(segmentOffset < segmentIdsLength) {
        int64_t temp = segmentIdWorkspace_.GetValue(segmentOffset * numOffset);
        if (temp == segmentId) {
            numResultTotal += numResultWorkspace_.GetValue(segmentOffset * numOffset);
            segmentOffset++;
            curCopyInNum++;
        } else if (temp == -1) {
            segmentOffset++;
        } else {
            break;
        }
    }
    
    int64_t perInnerBlockNum = xQueBlockNum_ / curCopyInNum;
    int64_t perInnerNum = perInnerBlockNum * EIGHT_ALIGN;
    perInnerNum = perInnerNum > currentCoreElements_ ? currentCoreElements_ : perInnerNum;
    perInnerNum = tilingData_->outBufferSize / sizeof(float) < perInnerNum ? tilingData_->outBufferSize / sizeof(float) : perInnerNum;
    int64_t nomalPerInnerNum = perInnerNum;
    int64_t nomalPerInnerNumAligned = (nomalPerInnerNum + EIGHT_ALIGN - 1) / EIGHT_ALIGN * EIGHT_ALIGN;

    int64_t repeatInnerTimes = (currentCoreElements_ + nomalPerInnerNum - 1) / nomalPerInnerNum;
    int64_t tailPerInnerNum = currentCoreElements_ - (repeatInnerTimes - 1) * nomalPerInnerNum;

    for (int64_t i=0; i < repeatInnerTimes; ++i) {
        int64_t curLoopInnerNum = (i + 1) == repeatInnerTimes ? tailPerInnerNum : nomalPerInnerNum;
        LocalTensor<float> xLocal = xQue_.AllocTensor<float>();
        LocalTensor<float> yLocal = yQue_.AllocTensor<float>();
        Duplicate<float>(yLocal, 0, curLoopInnerNum);
        int64_t xLocalOffset = 0;
        int64_t jumpNum = 0;
        for (int64_t j = 0; j < curCopyInNum; j++) {
            int64_t gmOffset = (segmentOffsetStart + jumpNum) * tilingData_->innerSize + i * nomalPerInnerNum;
            CopyIn(xLocal, curLoopInnerNum, xLocalOffset, gmOffset);
            xLocalOffset += nomalPerInnerNumAligned;
            if (segmentIdWorkspace_.GetValue((segmentOffsetStart + jumpNum + 1) * numOffset) == -1) {
                jumpNum += 2;
            } else {
                jumpNum++;
            }
        }
        xQue_.EnQue(xLocal);
        if (curCopyInNum > 1) {
            Compute(curCopyInNum, nomalPerInnerNumAligned);
            LocalTensor<float> sumLocal = sumTBuf_.Get<float>();
            AscendC::Add(yLocal, yLocal, sumLocal, curLoopInnerNum);
        } else {
            LocalTensor<float> xLocal = xQue_.DeQue<float>();
            AscendC::Add(yLocal, yLocal, xLocal, curLoopInnerNum);
            xQue_.FreeTensor(xLocal);
        }

        float scalar = float(1)/float(numResultTotal);
        AscendC::Muls(yLocal, yLocal, scalar, curLoopInnerNum); 
        if constexpr (std::negation<std::is_same<T1, float>>::value) {
            AscendC::Cast(yLocal.template ReinterpretCast<T1>(), yLocal, AscendC::RoundMode::CAST_RINT, curLoopInnerNum);
        }
        yQue_.EnQue(yLocal);
        
        int64_t yGmOffset = segmentId * tilingData_->innerSize + i * nomalPerInnerNum;
        CopyOut(curLoopInnerNum, yGmOffset);
    }
}

template <typename T1>
__aicore__ inline void SparseSegmentMeanMultiCoreKernel<T1>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNumForMulCore) {
        return;
    }
    int64_t segmentIdsLength = tilingData_->indicesOuter * DOUBLE_NUM;
    int64_t segmentOffset = 0;
    while(segmentOffset < segmentIdsLength) {
        ProcessSegment(segmentOffset, segmentIdsLength);
    }
}

template <typename T1>
__aicore__ inline void SparseSegmentMeanMultiCoreKernel<T1>::CopyIn(
    LocalTensor<float>& xLocal, int32_t burstLen, int64_t localOffset, int64_t gmOffset)
{
    DataCopyPadExtParams<float> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = 1;
    dataCoptExtParams.blockLen = burstLen * sizeof(float);
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(xLocal[localOffset], sumResultWorkspace_[gmOffset], dataCoptExtParams, dataCopyPadExtParams);
}

template <typename T1>
__aicore__ inline void SparseSegmentMeanMultiCoreKernel<T1>::CopyOut(int64_t count, int64_t yGmOffset)
{
    LocalTensor<T1> yLocal = yQue_.DeQue<T1>();
    DataCopyExtParams copyOutParamT1 = {static_cast<uint16_t>(1),
                                        static_cast<uint32_t>(count * sizeof(T1)),
                                        static_cast<uint32_t>(0),
                                        static_cast<uint32_t>(0),
                                        static_cast<uint32_t>(0)};

    DataCopyPad(yGm_[yGmOffset], yLocal, copyOutParamT1);
    yQue_.FreeTensor(yLocal);
}
}
#endif //SPARSE_SEGMENT_MEAN_MULTI_CORE_KERNEL_H_