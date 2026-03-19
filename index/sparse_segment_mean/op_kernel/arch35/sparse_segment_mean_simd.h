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
 * \file sparse_segment_mean_simd.h
 * \brief
 */

#ifndef SPARSE_SEGMENT_MEAN_SIMD_KERNEL_H_
#define SPARSE_SEGMENT_MEAN_SIMD_KERNEL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "sparse_segment_mean_struct.h"

namespace SparseSegmentMeanNameSpace {
constexpr uint32_t DOUBLE = 2;
constexpr uint32_t MIN_BATCH_BINAEY_ACC = 64;
constexpr uint32_t GM_NUM_OFFSET = 128;

using namespace AscendC;
template <typename T1, typename T2, typename T3>
class SparseSegmentMeanSimdKernel
{
public:
    __aicore__ inline SparseSegmentMeanSimdKernel(void){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR indices, GM_ADDR segmentIds, GM_ADDR y, GM_ADDR workspace, AscendC::TPipe& pipeIn,
        const SparseSegmentMeanSimdTilingData* tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void CopyInX(LocalTensor<T1>& xLocal, int32_t burstLen, int64_t localOffset, int64_t gmOffset, bool isLegal);
    __aicore__ inline void ComputeSum(uint32_t rSize, uint32_t aSize, uint32_t xLocalOffsetStart);
    __aicore__ inline void ProcessSingleSegment(int64_t& segmentOffset);
    __aicore__ inline void CopyOutY(int64_t segmentId, int64_t innersizeSingleLoopPro, int64_t innerOffset);
    __aicore__ inline void CopyOutWorkspace(int32_t writePostion, int64_t innersizeSingleLoopPro, int64_t innerOffset);
    __aicore__ inline void CopyOutNumResultWorkspace(int64_t offset, LocalTensor<int64_t>& local, uint32_t blockLen);
    __aicore__ inline void CopyOutSegmentIdWorkspace(int64_t offset, LocalTensor<int64_t>& local, uint32_t blockLen);
    __aicore__ inline void CopyInSegmentIds(LocalTensor<T3>& segmentIds, int32_t batchSize);

    TQue<QuePosition::VECIN, 1> xQue_;
    TQue<QuePosition::VECOUT, 1> yQue_;
    TBuf<QuePosition::VECCALC> sumTBuf_;
    TBuf<QuePosition::VECCALC> sharedTBuf_;
    TBuf<QuePosition::VECCALC> outBuf_;
    TBuf<QuePosition::VECCALC> outBuf2_;

    const SparseSegmentMeanSimdTilingData* tilingData_;

    GlobalTensor<T1> xGm_;
    GlobalTensor<T1> yGm_;
    GlobalTensor<T2> indicesGm_;
    GlobalTensor<T3> segmentIdsGm_;

    GlobalTensor<float> sumResultWorkspace_;
    GlobalTensor<int64_t> numResultWorkspace_;
    GlobalTensor<int64_t> segmentIdWorkspace_;

    uint32_t blockIdx_ = 0;

    int64_t curCoreIndicesNum_;
    int64_t curCoreInnerNum_;

    int64_t innerGmOffset_;
    int64_t indicesGmOffset_;

    int64_t indicesAxisIndex_;
    int64_t innerAxisIndex_;

    int64_t xQueBlockNum_;

    constexpr static int32_t BLOCK_SIZE = platform::GetUbBlockSize();
    constexpr static int32_t blockNumT1_ = BLOCK_SIZE / sizeof(T1);
};

template <typename T1, typename T2, typename T3>
__aicore__ inline void SparseSegmentMeanSimdKernel<T1, T2, T3>::Init(
    GM_ADDR x, GM_ADDR indices, GM_ADDR segmentIds, GM_ADDR y, GM_ADDR workspace, AscendC::TPipe& pipeIn,
    const SparseSegmentMeanSimdTilingData* tilingData)
{
    tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();

    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    // 先分核在分UB
    indicesAxisIndex_ = blockIdx_ / tilingData_->innerOuter;
    innerAxisIndex_ = blockIdx_ % tilingData_->innerOuter;

    innerGmOffset_ = innerAxisIndex_ * tilingData_->normalCoreInnerNum;
    indicesGmOffset_ = indicesAxisIndex_ * tilingData_->normalCoreIndicesNum;

    curCoreIndicesNum_ = (indicesAxisIndex_ + 1 == tilingData_->indicesOuter) ? tilingData_->tailCoreIndicesNum :
                                                                                tilingData_->normalCoreIndicesNum;
    curCoreInnerNum_ = (innerAxisIndex_ + 1 == tilingData_->innerOuter) ? tilingData_->tailCoreInnerNum :
                                                                          tilingData_->normalCoreInnerNum;

    xQueBlockNum_ = tilingData_->xBufferSize / BLOCK_SIZE;
    xGm_.SetGlobalBuffer((__gm__ T1*)x);
    indicesGm_.SetGlobalBuffer((__gm__ T2*)indices + indicesGmOffset_);
    segmentIdsGm_.SetGlobalBuffer((__gm__ T3*)segmentIds + indicesGmOffset_);
    yGm_.SetGlobalBuffer((__gm__ T1*)y);

    // workspace 组成： sumResultWorkspace_ [innerSize * indicesOuter * 2]   numResultWorkspace_ [indicesOuter * 2 int64 个点  点与点之间偏移128B 跟前面偏移128B]    
    //    segmentIdWorkspace_  同numResultWorkspace_
    sumResultWorkspace_.SetGlobalBuffer((__gm__ float*)workspace + indicesAxisIndex_ * DOUBLE * tilingData_->innerSize); // 局部
    numResultWorkspace_.SetGlobalBuffer((__gm__ int64_t*)workspace + tilingData_->indicesOuter * DOUBLE * tilingData_->innerSize / DOUBLE + GM_NUM_OFFSET / sizeof(int64_t)); // 局部
    segmentIdWorkspace_.SetGlobalBuffer((__gm__ int64_t*)workspace + tilingData_->indicesOuter * DOUBLE * tilingData_->innerSize / DOUBLE + GM_NUM_OFFSET / sizeof(int64_t) + DOUBLE * tilingData_->indicesOuter * (GM_NUM_OFFSET / sizeof(int64_t))); // 局部

    pipeIn.InitBuffer(xQue_, 1, tilingData_->xBufferSize); // 需要满足double  block对齐
    pipeIn.InitBuffer(yQue_, 1, tilingData_->yBufferSize);
    pipeIn.InitBuffer(sumTBuf_, tilingData_->yBufferSize);
    pipeIn.InitBuffer(sharedTBuf_, tilingData_->sharedTmpBufferSize);

    pipeIn.InitBuffer(outBuf_,  tilingData_->workspaceBufferSize);
    pipeIn.InitBuffer(outBuf2_, tilingData_->workspaceBufferSize);
}

template <typename T1, typename T2, typename T3>
__aicore__ inline void SparseSegmentMeanSimdKernel<T1, T2, T3>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    int64_t segmentOffset = 0;
    while (segmentOffset < curCoreIndicesNum_) {
        ProcessSingleSegment(segmentOffset);
    }
}

template <typename T1, typename T2, typename T3>
__aicore__ inline void SparseSegmentMeanSimdKernel<T1, T2, T3>::CopyInX(
    LocalTensor<T1>& xLocal, int32_t burstLen, int64_t localOffset, int64_t gmOffset, bool isLegal)
{
    if (isLegal) {
        DataCopyPadExtParams<T1> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;

        DataCopyExtParams dataCoptExtParams;
        dataCoptExtParams.blockCount = 1;
        dataCoptExtParams.blockLen = burstLen * sizeof(T1);
        dataCoptExtParams.srcStride = 0;
        dataCoptExtParams.dstStride = 0;
        DataCopyPad(xLocal[localOffset], xGm_[gmOffset], dataCoptExtParams, dataCopyPadExtParams);
    } else {
        Duplicate<T1>(xLocal[localOffset], 0, burstLen);
    }
}


template <typename T1, typename T2, typename T3>
__aicore__ inline void SparseSegmentMeanSimdKernel<T1, T2, T3>::ComputeSum(
    uint32_t rSize, uint32_t aSize, uint32_t xLocalOffsetStart)
{
    LocalTensor<T1> xLocal = xQue_.DeQue<T1>();
    LocalTensor<float> sumLocal = sumTBuf_.Get<float>();
    LocalTensor<uint8_t> sharedTmpLocal = sharedTBuf_.Get<uint8_t>();
    uint32_t shape[] = {rSize, aSize};
    constexpr bool isReuse = true;

    if constexpr (std::negation<std::is_same<T1, float>>::value) {
        uint32_t count = rSize * aSize;
        AscendC::Cast(
            xLocal.template ReinterpretCast<float>(), xLocal[xLocalOffsetStart], AscendC::RoundMode::CAST_NONE, count);
    }
    AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, isReuse>(
        sumLocal, xLocal.template ReinterpretCast<float>(), sharedTmpLocal, shape, true);
    xQue_.FreeTensor(xLocal);
}

template <typename T1, typename T2, typename T3>
__aicore__ inline void SparseSegmentMeanSimdKernel<T1, T2, T3>::ProcessSingleSegment(int64_t& segmentOffset)
{
    bool isFirst = false;
    bool isLast = false;
    if (segmentOffset == 0) {
        isFirst = true;
    }

    int64_t indicesStart = segmentOffset;
    int64_t curSegmentIdsNum = 1;
    int64_t segmentId = segmentIdsGm_.GetValue(segmentOffset);

    segmentOffset++;
    while (segmentOffset < curCoreIndicesNum_) {
        if (segmentId == segmentIdsGm_.GetValue(segmentOffset)) {
            curSegmentIdsNum++;
            segmentOffset++;
        } else {
            break;
        }
    }

    if (indicesStart + curSegmentIdsNum >= curCoreIndicesNum_) {
        isLast = true;
    }

    int64_t indicesSingleLoopPro = curSegmentIdsNum >= MIN_BATCH_BINAEY_ACC ? MIN_BATCH_BINAEY_ACC : curSegmentIdsNum;
    int64_t indicesRepeatTimes = (curSegmentIdsNum + indicesSingleLoopPro - 1) / indicesSingleLoopPro;
    int64_t indicesSingleLoopProTail = curSegmentIdsNum - (indicesRepeatTimes - 1) * indicesSingleLoopPro;

    int64_t blockNumsPerIndices = xQueBlockNum_ / indicesSingleLoopPro; // xQueBlockNum_ 远远大于64  结果不会为0
    if constexpr (std::negation<std::is_same<T1, float>>::value) {
        blockNumsPerIndices = blockNumsPerIndices / DOUBLE;
    }

    int64_t innersizeSingleLoopPro = blockNumsPerIndices * blockNumT1_;
    innersizeSingleLoopPro = innersizeSingleLoopPro > curCoreInnerNum_ ? curCoreInnerNum_ : innersizeSingleLoopPro;
    innersizeSingleLoopPro = tilingData_->yBufferSize / sizeof(float) < innersizeSingleLoopPro ? tilingData_->yBufferSize / sizeof(float) : innersizeSingleLoopPro;
    int64_t innerRepeatTimes = (curCoreInnerNum_ + innersizeSingleLoopPro - 1) / innersizeSingleLoopPro;
    int64_t innersizeSingleLoopProTail = curCoreInnerNum_ - (innerRepeatTimes - 1) * innersizeSingleLoopPro;
    int64_t innersizeSingleLoopProAligned = (innersizeSingleLoopPro + blockNumT1_ - 1) / blockNumT1_ * blockNumT1_;

    int64_t xLocalOffsetStart = 0;
    if constexpr (std::negation<std::is_same<T1, float>>::value) {
        xLocalOffsetStart = xQueBlockNum_ / DOUBLE * blockNumT1_;
    }

    for (int64_t i = 0; i < innerRepeatTimes; i++) {
        int64_t curLoopInnerNum = (i + 1) == innerRepeatTimes ? innersizeSingleLoopProTail : innersizeSingleLoopPro;
        int64_t innerOffset = i * innersizeSingleLoopPro + innerGmOffset_;
        LocalTensor<float> yLocal = yQue_.AllocTensor<float>();
        Duplicate<float>(yLocal, 0, curLoopInnerNum);
        int64_t indicesOffset = indicesStart;
        for (int64_t j = 0; j < indicesRepeatTimes; j++) {
            int64_t curLoopIndicesNum = (j + 1) == indicesRepeatTimes ? indicesSingleLoopProTail : indicesSingleLoopPro;
            LocalTensor<T1> xLocal = xQue_.AllocTensor<T1>();
            int64_t xLocalOffset = xLocalOffsetStart;
            for (int64_t k = 0; k < curLoopIndicesNum; k++) {
                int64_t indices = indicesGm_.GetValue(indicesOffset);
                int64_t gmOffset = indices * tilingData_->innerSize + innerOffset;
                bool isLegal = true;
                if (indices < 0 || indices >= tilingData_->gatherSize) {
                    isLegal = false;
                }

                CopyInX(xLocal, curLoopInnerNum, xLocalOffset, gmOffset, isLegal);
                xLocalOffset += innersizeSingleLoopProAligned;
                indicesOffset++;
            }
            xQue_.EnQue(xLocal);
            ComputeSum(curLoopIndicesNum, innersizeSingleLoopProAligned, xLocalOffsetStart);
            LocalTensor<float> sumLocal = sumTBuf_.Get<float>();
            AscendC::Add(yLocal, yLocal, sumLocal, curLoopInnerNum);
        }

        if (!isFirst && !isLast) { // 非头非尾
            float scalar = float(1) / float(curSegmentIdsNum);
            AscendC::Muls(yLocal, yLocal, scalar, curLoopInnerNum);
            if constexpr (std::negation<std::is_same<T1, float>>::value) {
                AscendC::Cast(
                    yLocal.template ReinterpretCast<T1>(), yLocal, AscendC::RoundMode::CAST_RINT, curLoopInnerNum);
            }
            yQue_.EnQue(yLocal);
            CopyOutY(segmentId, curLoopInnerNum, innerOffset);
        } else { // 需要写workspace
            yQue_.EnQue(yLocal);
            int64_t writePostion = isFirst ? 0 : 1;
            CopyOutWorkspace(writePostion, curLoopInnerNum, innerOffset); // 既是头又是尾的情况只写头
            if (innerAxisIndex_ == 0) {  // 只有第一列的核需要写num 和segmentid
                constexpr int64_t numOffset = GM_NUM_OFFSET / sizeof(int64_t);
                int64_t resultWorkspaceOffset = DOUBLE * indicesAxisIndex_ * numOffset;
                if (isFirst && isLast) { // 既是头又是尾的情况需要写头，尾填-1
                    event_t eventIdMte3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
                    SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
                    WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);

                    LocalTensor<int64_t> numResultLocal = outBuf_.Get<int64_t>();
                    numResultLocal.SetValue(0, curSegmentIdsNum);
                    numResultLocal.SetValue(numOffset, -1);

                    LocalTensor<int64_t> segmentIdLocal = outBuf2_.Get<int64_t>();
                    segmentIdLocal.SetValue(0, segmentId);
                    segmentIdLocal.SetValue(numOffset, -1);

                    event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
                    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
                    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
                    CopyOutNumResultWorkspace(resultWorkspaceOffset, numResultLocal, tilingData_->workspaceBufferSize);
                    CopyOutSegmentIdWorkspace(resultWorkspaceOffset, segmentIdLocal, tilingData_->workspaceBufferSize);
                } else {
                    event_t eventIdMte3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
                    SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
                    WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);

                    LocalTensor<int64_t> numResultLocal = outBuf_.Get<int64_t>();
                    numResultLocal.SetValue(0, curSegmentIdsNum);
                   
                    LocalTensor<int64_t> segmentIdLocal = outBuf2_.Get<int64_t>();
                    segmentIdLocal.SetValue(0, segmentId);
                   
                    event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
                    SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
                    WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);

                    CopyOutNumResultWorkspace(resultWorkspaceOffset + writePostion * numOffset, numResultLocal, sizeof(int64_t));
                    CopyOutSegmentIdWorkspace(resultWorkspaceOffset + writePostion * numOffset, segmentIdLocal, sizeof(int64_t));
                }
            }
        }
    }
}

template <typename T1, typename T2, typename T3>
__aicore__ inline void SparseSegmentMeanSimdKernel<T1, T2, T3>::CopyOutY(
    int64_t segmentId, int64_t innersizeSingleLoopPro, int64_t innerOffset)
{
    LocalTensor<T1> yLocal = yQue_.DeQue<T1>();
    DataCopyExtParams copyOutParamT1 = {
        static_cast<uint16_t>(1), static_cast<uint32_t>(innersizeSingleLoopPro * sizeof(T1)), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPad(yGm_[segmentId * tilingData_->innerSize + innerOffset], yLocal, copyOutParamT1);
    yQue_.FreeTensor(yLocal);
}

template <typename T1, typename T2, typename T3>
__aicore__ inline void SparseSegmentMeanSimdKernel<T1, T2, T3>::CopyOutWorkspace(
    int32_t writePostion, int64_t innersizeSingleLoopPro, int64_t innerOffset)
{
    // writePostion  头是0  尾是1
    LocalTensor<float> yLocal = yQue_.DeQue<float>();
    DataCopyExtParams copyOutParamFloat = {
        static_cast<uint16_t>(1), static_cast<uint32_t>(innersizeSingleLoopPro * sizeof(float)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPad(sumResultWorkspace_[writePostion * tilingData_->innerSize + innerOffset], yLocal, copyOutParamFloat);
    yQue_.FreeTensor(yLocal);
}


template <typename T1, typename T2, typename T3>
__aicore__ inline void SparseSegmentMeanSimdKernel<T1, T2, T3>::CopyOutSegmentIdWorkspace(
    int64_t offset, LocalTensor<int64_t>& segmentIdLocal, uint32_t blockLen)
{
    DataCopyExtParams copyOutParam = {
        static_cast<uint16_t>(1), blockLen, static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPad(segmentIdWorkspace_[offset], segmentIdLocal, copyOutParam);
}

template <typename T1, typename T2, typename T3>
__aicore__ inline void SparseSegmentMeanSimdKernel<T1, T2, T3>::CopyOutNumResultWorkspace(
    int64_t offset, LocalTensor<int64_t>& numResultLocal, uint32_t blockLen)
{
    DataCopyExtParams copyOutParam = {
        static_cast<uint16_t>(1), blockLen,  static_cast<uint32_t>(0),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPad(numResultWorkspace_[offset], numResultLocal, copyOutParam);
}

} // namespace SparseSegmentMeanNameSpace
#endif // SPARSE_SEGMENT_MEAN_SIMD_KERNEL_H_