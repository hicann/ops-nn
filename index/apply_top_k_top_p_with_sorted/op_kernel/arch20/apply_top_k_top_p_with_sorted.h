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
 * \file apply_top_k_top_p_with_sorted.h
 * \brief 310P (arch20) kernel for ApplyTopKTopPWithSorted.
 *
 * 310P DataCopy consumer contract:
 *   310P bans DataCopyPad. All GM->UB copies use DataCopy with AlignUp(elem,
 *   BLOCK_BYTES/sizeof(T)), which may over-read the last 32B block into UB.
 *   Consumers MUST only process the original (non-aligned) element count in
 *   subsequent vector ops (Cast, Compare, ReduceSum, etc.). The over-read
 *   bytes in UB are garbage and must not enter computation results.
 *   Currently safe because all call sites pass dataNumInit_ or tailUbFactorElement_
 *   (non-aligned) to vector ops, never the aligned copy length.
 */
#ifndef APPLY_TOP_K_TOP_P_WITH_SORTED_H_KERNEL
#define APPLY_TOP_K_TOP_P_WITH_SORTED_H_KERNEL

#include "kernel_operator.h"

#if __CCE_AICORE__ == 200
#ifndef bfloat16_t
#define bfloat16_t int16_t
#endif
#endif

using namespace AscendC;
namespace ApplyTopKTopPWithSortedOp {
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint16_t FLOAT16_NEG_INF = 0xFC00;    // -inf 64512
constexpr uint16_t BF16_NEG_INF = 0xFF80;       // -inf 65408
constexpr int32_t FLOAT32_NEG_INF = 0xFF800000; // -inf -2139095040

constexpr uint32_t BLOCK_BYTES = 32;
constexpr uint32_t DATA_PER_BLOCK_B32 = 8;
constexpr uint32_t DATA_PER_REPEAT_B32 = 64;
constexpr uint32_t K_MAX = 1024;
constexpr uint32_t LARGE_VOCAB_THRESHOLD = 8 * K_MAX;
constexpr uint64_t MASK_64 = 64;
constexpr uint32_t RESERVE_CAL_BUFFER_SIZE = 1024;
constexpr CumSumConfig CUMSUM_CONFIG{true, true, false};

template <typename inputT, typename calT, typename outputT>
class ApplyTopKTopPWithSorted {
public:
    __aicore__ inline ApplyTopKTopPWithSorted(){};
    __aicore__ inline void SetMode(uint32_t m) { mode_ = m; }
    __aicore__ inline void InitTilingData(const ApplyTopKTopPWithSortedTilingData& __restrict tilingData,
                                          GM_ADDR sorted_value, GM_ADDR sorted_indices, GM_ADDR p, GM_ADDR k,
                                          GM_ADDR out, GM_ADDR workSpace);
    __aicore__ inline void InitBuffer(TPipe* inputPipe);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessTopK();

private:
    __aicore__ inline void InitCopyIn(uint32_t loopBatch, int64_t currentGmIdx);
    __aicore__ inline void InitProcess(uint32_t loopBatch);
    __aicore__ inline void ProcessKLtKMax(uint32_t loopBatch);
    __aicore__ inline void ScatterCumtomImpl(uint32_t loopBatch, uint32_t loopProbNum, uint32_t offset);
    __aicore__ inline void ProcessRemain(uint32_t loopBatch);
    __aicore__ inline void ProcessLargeVocab();
    __aicore__ inline void ProcessOneBatchLarge(uint32_t batchIdx);
    __aicore__ inline void ProcessRemainderPass();
    __aicore__ inline void ScatterOne(int32_t gmIndex, outputT value);
    __aicore__ inline void FillNegInf(uint32_t batchIdx);
    __aicore__ inline void ProcessLargeVocabScatter();
    __aicore__ inline void ScatterTopKSuffix();
    __aicore__ inline void GetKthResult(uint32_t loopBatch, uint32_t offset, uint8_t repeatTimes);
    __aicore__ inline void GetFirstKLoop(uint32_t loopBatch, int32_t& firstKLoop);
    __aicore__ inline void ScatterFromFirstKLoop(uint32_t loopBatch, int32_t firstKLoop, float& cumsumData);
    __aicore__ inline void ReduceSumWithAddsAndExpImpl(uint32_t offset, uint32_t loopDataNum);
    __aicore__ inline void CumSumWithAddsAndExpImpl(uint32_t offset, uint32_t loopDataNum, uint32_t cumsumInner,
                                                    float cumsumData);
    __aicore__ inline void SetCumsumGTIndex(uint32_t loopBatch, int32_t index);
    __aicore__ inline void ProcessScatter();
    __aicore__ inline void ProcessResScatter();
    // topk func
    __aicore__ inline void InitProcessTopK(uint32_t loopBatch);
    __aicore__ inline void ProcessKLtKMaxTopK(uint32_t loopBatch);
    __aicore__ inline void ProcessRemainTopK(uint32_t loopBatch);
    __aicore__ inline void GetFirstKLoopTopK(uint32_t loopBatch, int32_t& firstKLoop);
    __aicore__ inline void ScatterFromFirstKLoopTopK(uint32_t loopBatch, int32_t firstKLoop);
    __aicore__ inline void ScatterCumtomImplTopK(uint32_t loopBatch, uint32_t loopProbNum, uint32_t offset);
    __aicore__ inline void SToMTE3Sync()
    {
        event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    }
    __aicore__ inline void MTE3ToSSync()
    {
        event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
        WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
    }
    __aicore__ inline void VToSSync()
    {
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
    }
    __aicore__ inline void MTE2ToVSync()
    {
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    }
    __aicore__ inline void MTE2ToSSync()
    {
        event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    }
    __aicore__ inline void MTE3ToVSync()
    {
        event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    }
    __aicore__ inline void SToMTE2Sync()
    {
        event_t eventIDSToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
        WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
    }
    __aicore__ inline void VToMTE3Sync()
    {
        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    }
    __aicore__ inline void SToVSync()
    {
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
    }
    __aicore__ inline void CopyOutToGM(GlobalTensor<outputT> gmDst, LocalTensor<outputT> ubSrc, uint32_t count)
    {
        constexpr uint32_t blockElems = BLOCK_BYTES / sizeof(outputT);
        uint32_t alignedCount = count / blockElems * blockElems;
        uint32_t tailCount = count - alignedCount;
        if (alignedCount > 0) {
            VToMTE3Sync();
            DataCopy(gmDst, ubSrc, alignedCount);
            MTE3ToSSync();
        }
        if (tailCount > 0) {
            VToSSync();
            for (uint32_t i = 0; i < tailCount; i++) {
                gmDst.SetValue(alignedCount + i, ubSrc.GetValue(alignedCount + i));
            }
        }
    }
    template <typename T>
    __aicore__ inline T Min(T a, T b)
    {
        return a > b ? b : a;
    }

    template <typename T>
    __aicore__ inline T Max(T a, T b)
    {
        return a > b ? a : b;
    }

private:
    TPipe* pipe_;
    // create queues for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> sortedValueInQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> sortedIndicesInQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> pInQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> kInQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue_;
    TBuf<TPosition::VECCALC> calBuf_;

    // tilingData
    uint32_t batchSize_ = 0;
    uint32_t vocabSize_ = 0;
    uint32_t batchPerCore_ = 0;
    uint32_t tailBatch_ = 0;
    uint32_t blockNum_ = 0;
    uint32_t dataNumInit_ = 0;
    uint32_t dataNumInitAligned_ = 0;
    uint32_t ubFactorElement_ = 0;
    uint32_t ubFactorElementAligned_ = 0;
    uint32_t tailUbFactorElement_ = 0;
    uint32_t tailUbFactorElementAligned_ = 0;
    uint32_t calUbSize_ = 0;

    uint32_t blockIdx_ = 0;
    uint32_t loopBatch_ = 0;
    uint32_t batchOffset_ = 0;
    uint32_t bufOffsetLoop = 0;
    uint32_t loopInner_ = 0;
    uint32_t loopInnerOnlyP_ = 0;
    int64_t baseGmIdx_ = 0;
    uint32_t numGroups_ = 0;
    uint32_t remBatches_ = 0;
    uint32_t remOffset_ = 0;
    uint32_t outOffset_ = 0;
    uint32_t lastGroupCore_ = 0;
    uint32_t mode_ = 0;
    uint32_t keepStart_ = 0;

    GlobalTensor<inputT> mGmSortedValue_;
    GlobalTensor<int32_t> mGmSortedIndices_;
    GlobalTensor<inputT> mGmP_;
    GlobalTensor<int32_t> mGmK_;
    GlobalTensor<outputT> mGmOut_;
    GlobalTensor<int32_t> mGmCumsumGTIndex;
    GlobalTensor<outputT> mGmRemVals;

    LocalTensor<int32_t> kLocal;
    LocalTensor<inputT> pLocal;
    LocalTensor<outputT> outTensor;
    LocalTensor<inputT> sortedValueLocal;
    LocalTensor<int32_t> sortedIndicesLocal;

    LocalTensor<float> sortedValueLocalFp32;
    LocalTensor<float> negInfLocal;

    LocalTensor<float> calLocalFp32;
    LocalTensor<float> kthValueLocal;
    LocalTensor<float> tmpLocal;
    LocalTensor<float> cumSumRes;
    LocalTensor<float> cumSumTmp;
    LocalTensor<float> reduceLocal;
    LocalTensor<float> softMaxRes;
    LocalTensor<inputT> scatterTensor;
    LocalTensor<uint8_t> sharedTmpBuffer;

    LocalTensor<int32_t> scatterIdxTensor;
    LocalTensor<inputT> scatterSortedValueLocal;
    LocalTensor<int32_t> scatterSortedIndicesLocal;
    LocalTensor<inputT> scatterValueTensor;

    float kthValue = 0;
    float pValue = 0;
    float maxValue = 0;
    float reduceSumValueInvert = 0;
    float reduceSumValue = 0;
    inputT kthTopKValue = 0;
    bool hadGreaterCumsumP = false;
    bool hadGreaterK = false;
    bool hadGreaterKFirstLoop = false;
    uint32_t scatterTensorNums_ = 1;
    BinaryRepeatParams repeatParams = {1, 0, 1, 8, 0, 8};
};

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::InitTilingData(
    const ApplyTopKTopPWithSortedTilingData& __restrict tilingData, GM_ADDR sorted_value, GM_ADDR sorted_indices,
    GM_ADDR p, GM_ADDR k, GM_ADDR out, GM_ADDR workSpace)
{
    batchSize_ = tilingData.batchSize;
    vocabSize_ = tilingData.vocabSize;
    batchPerCore_ = tilingData.batchPerCore;
    tailBatch_ = tilingData.tailBatch;
    blockNum_ = tilingData.blockNum;
    dataNumInit_ = tilingData.dataNumInit;
    dataNumInitAligned_ = AscendC::AlignUp(dataNumInit_, DATA_PER_BLOCK_B32);
    ubFactorElement_ = tilingData.ubFactorElement;
    ubFactorElementAligned_ = tilingData.ubFactorElementAligned;
    tailUbFactorElement_ = tilingData.tailUbFactorElement;
    tailUbFactorElementAligned_ = tilingData.tailUbFactorElementAligned;
    calUbSize_ = tilingData.calUbSize;
    blockIdx_ = GetBlockIdx();

    constexpr uint32_t GROUP_SIZE = BLOCK_BYTES / sizeof(outputT);
    numGroups_ = batchSize_ / GROUP_SIZE;
    remBatches_ = batchSize_ % GROUP_SIZE;
    remOffset_ = numGroups_ * GROUP_SIZE;
    uint32_t groupsPerCore = numGroups_ / blockNum_;
    uint32_t extraGroups = numGroups_ % blockNum_;
    uint32_t groupsOnCore = groupsPerCore + (blockIdx_ < extraGroups ? 1 : 0);
    batchOffset_ = (blockIdx_ * groupsPerCore + (blockIdx_ < extraGroups ? blockIdx_ : extraGroups)) * GROUP_SIZE;
    loopBatch_ = groupsOnCore * GROUP_SIZE;
    lastGroupCore_ = 0;
    if (numGroups_ > 0) {
        uint32_t acc = 0;
        for (uint32_t c = 0; c < blockNum_; c++) {
            uint32_t goc = groupsPerCore + (c < extraGroups ? 1 : 0);
            if (goc > 0 && acc + goc >= numGroups_) {
                lastGroupCore_ = c;
                break;
            }
            acc += goc;
        }
    }
    loopInner_ = (vocabSize_ - dataNumInit_ + ubFactorElementAligned_ - 1) / ubFactorElementAligned_;
    loopInnerOnlyP_ = (vocabSize_ + ubFactorElementAligned_ - 1) / ubFactorElementAligned_;
    mGmSortedValue_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT*>(sorted_value));
    mGmSortedIndices_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sorted_indices));
    mGmP_.SetGlobalBuffer(reinterpret_cast<__gm__ inputT*>(p));
    mGmK_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(k));
    mGmOut_.SetGlobalBuffer(reinterpret_cast<__gm__ outputT*>(out));
    mGmCumsumGTIndex.SetGlobalBuffer((__gm__ int32_t*)workSpace, batchSize_ * DATA_PER_BLOCK_B32);
    mGmRemVals.SetGlobalBuffer(reinterpret_cast<__gm__ outputT*>(reinterpret_cast<__gm__ int32_t*>(workSpace) +
                                                                 batchSize_ * DATA_PER_BLOCK_B32),
                               batchSize_ * (BLOCK_BYTES / sizeof(outputT)));
}

// init used buffer
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::InitBuffer(TPipe* inputPipe)
{
    pipe_ = inputPipe;
    pipe_->InitBuffer(sortedValueInQueue_, BUFFER_NUM, sizeof(inputT) * (ubFactorElementAligned_ + K_MAX));
    pipe_->InitBuffer(sortedIndicesInQueue_, BUFFER_NUM, sizeof(int32_t) * (ubFactorElementAligned_ + K_MAX));
    pipe_->InitBuffer(pInQueue_, BUFFER_NUM, BLOCK_BYTES);
    pipe_->InitBuffer(kInQueue_, BUFFER_NUM, BLOCK_BYTES);
    pipe_->InitBuffer(outQueue_, BUFFER_NUM, BLOCK_BYTES * dataNumInit_);
    pipe_->InitBuffer(calBuf_, calUbSize_);
    if constexpr (!IsSameType<inputT, float>::value) {
        sortedValueLocalFp32 = calBuf_.GetWithOffset<float>(ubFactorElementAligned_ + K_MAX, bufOffsetLoop);
        bufOffsetLoop = bufOffsetLoop + (ubFactorElementAligned_ + K_MAX) * sizeof(float);
    }
    kthValueLocal = calBuf_.GetWithOffset<float>(DATA_PER_BLOCK_B32, bufOffsetLoop);
    bufOffsetLoop = bufOffsetLoop + BLOCK_BYTES;

    negInfLocal = calBuf_.GetWithOffset<float>(DATA_PER_BLOCK_B32, bufOffsetLoop);
    bufOffsetLoop = bufOffsetLoop + BLOCK_BYTES;

    tmpLocal = calBuf_.GetWithOffset<float>(ubFactorElementAligned_, bufOffsetLoop);
    bufOffsetLoop = bufOffsetLoop + ubFactorElementAligned_ * sizeof(float);
    cumSumRes = calBuf_.GetWithOffset<float>(ubFactorElementAligned_, bufOffsetLoop);
    bufOffsetLoop = bufOffsetLoop + ubFactorElementAligned_ * sizeof(float);
    cumSumTmp = calBuf_.GetWithOffset<float>(ubFactorElementAligned_, bufOffsetLoop);
    bufOffsetLoop = bufOffsetLoop + ubFactorElementAligned_ * sizeof(float);
    reduceLocal = calBuf_.GetWithOffset<float>(ubFactorElementAligned_ * BLOCK_BYTES, bufOffsetLoop);
    bufOffsetLoop = bufOffsetLoop + ubFactorElementAligned_ * BLOCK_BYTES * sizeof(float);

    softMaxRes = tmpLocal.template ReinterpretCast<float>();
    scatterTensor = reduceLocal.template ReinterpretCast<inputT>();
    sharedTmpBuffer = reduceLocal.template ReinterpretCast<uint8_t>();

    scatterIdxTensor = scatterTensor.template ReinterpretCast<int32_t>();
    scatterTensorNums_ = (calUbSize_ - RESERVE_CAL_BUFFER_SIZE) / (sizeof(float) + sizeof(int32_t)) / BLOCK_BYTES *
                         BLOCK_BYTES;
    scatterSortedValueLocal = calBuf_.GetWithOffset<inputT>(scatterTensorNums_, 0);
    scatterSortedIndicesLocal = calBuf_.GetWithOffset<int32_t>(scatterTensorNums_, scatterTensorNums_ * sizeof(inputT));
    scatterValueTensor = calBuf_.GetWithOffset<inputT>(BLOCK_BYTES / sizeof(inputT),
                                                       scatterTensorNums_ * (sizeof(inputT) + sizeof(int32_t)));
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::Process()
{
    kLocal = kInQueue_.AllocTensor<int32_t>();
    pLocal = pInQueue_.AllocTensor<inputT>();
    outTensor = outQueue_.AllocTensor<outputT>();
    sortedValueLocal = sortedValueInQueue_.AllocTensor<inputT>();
    sortedIndicesLocal = sortedIndicesInQueue_.AllocTensor<int32_t>();
    Duplicate(negInfLocal.template ReinterpretCast<int32_t>(), FLOAT32_NEG_INF, DATA_PER_BLOCK_B32);
    if constexpr (IsSameType<inputT, float>::value) {
        calLocalFp32 = sortedValueLocal;
        Duplicate(outTensor.template ReinterpretCast<int32_t>(), FLOAT32_NEG_INF, ubFactorElementAligned_);
    } else if constexpr (IsSameType<inputT, half>::value) {
        calLocalFp32 = sortedValueLocalFp32;
        Duplicate(outTensor.template ReinterpretCast<uint16_t>(), FLOAT16_NEG_INF, ubFactorElementAligned_);
    } else {
        calLocalFp32 = sortedValueLocalFp32;
        Duplicate(outTensor.template ReinterpretCast<uint16_t>(), BF16_NEG_INF, ubFactorElementAligned_);
    }
    VToSSync();
    if (loopInner_ == 0) {
        constexpr uint32_t GROUP_SIZE = BLOCK_BYTES / sizeof(outputT);
        for (uint32_t g = 0; g < loopBatch_; g += GROUP_SIZE) {
            if constexpr (IsSameType<inputT, float>::value) {
                Duplicate(outTensor.template ReinterpretCast<int32_t>(), FLOAT32_NEG_INF, GROUP_SIZE * dataNumInit_);
            } else if constexpr (IsSameType<inputT, half>::value) {
                Duplicate(outTensor.template ReinterpretCast<uint16_t>(), FLOAT16_NEG_INF, GROUP_SIZE * dataNumInit_);
            } else {
                Duplicate(outTensor.template ReinterpretCast<uint16_t>(), BF16_NEG_INF, GROUP_SIZE * dataNumInit_);
            }
            VToSSync();
            uint32_t groupStartBatch = batchOffset_ + g;
            for (uint32_t b = 0; b < GROUP_SIZE; b++) {
                outOffset_ = b * dataNumInit_;
                baseGmIdx_ = (groupStartBatch + b) * vocabSize_;
                hadGreaterKFirstLoop = false;
                hadGreaterK = false;
                hadGreaterCumsumP = false;
                uint32_t loopBatch = g + b;
                InitProcess(loopBatch);
                if (calLocalFp32.GetValue(ubFactorElementAligned_) < kthValue) {
                    ProcessKLtKMax(loopBatch);
                } else {
                    ProcessRemain(loopBatch);
                }
                SToMTE2Sync();
            }
            SToMTE3Sync();
            DataCopy(mGmOut_[groupStartBatch * vocabSize_], outTensor, GROUP_SIZE * dataNumInit_);
            MTE3ToVSync();
        }
        if (blockIdx_ == lastGroupCore_ && remBatches_ > 0) {
            if constexpr (IsSameType<inputT, float>::value) {
                Duplicate(outTensor.template ReinterpretCast<int32_t>(), FLOAT32_NEG_INF, GROUP_SIZE * dataNumInit_);
            } else if constexpr (IsSameType<inputT, half>::value) {
                Duplicate(outTensor.template ReinterpretCast<uint16_t>(), FLOAT16_NEG_INF, GROUP_SIZE * dataNumInit_);
            } else {
                Duplicate(outTensor.template ReinterpretCast<uint16_t>(), BF16_NEG_INF, GROUP_SIZE * dataNumInit_);
            }
            VToSSync();
            for (uint32_t b = 0; b < remBatches_; b++) {
                outOffset_ = b * dataNumInit_;
                baseGmIdx_ = (remOffset_ + b) * vocabSize_;
                hadGreaterKFirstLoop = false;
                hadGreaterK = false;
                hadGreaterCumsumP = false;
                uint32_t loopBatch = remOffset_ + b - batchOffset_;
                InitProcess(loopBatch);
                if (calLocalFp32.GetValue(ubFactorElementAligned_) < kthValue) {
                    ProcessKLtKMax(loopBatch);
                } else {
                    ProcessRemain(loopBatch);
                }
                SToMTE2Sync();
            }
            uint32_t remTotal = remBatches_ * dataNumInit_;
            uint32_t aligned = (remTotal / GROUP_SIZE) * GROUP_SIZE;
            SToMTE3Sync();
            DataCopy(mGmOut_[remOffset_ * vocabSize_], outTensor, aligned);
            MTE3ToSSync();
            for (uint32_t i = aligned; i < remTotal; i++) {
                mGmOut_.SetValue(remOffset_ * vocabSize_ + i, outTensor.GetValue(i));
            }
            SToMTE3Sync();
            MTE3ToVSync();
        }
    } else {
        if (vocabSize_ > LARGE_VOCAB_THRESHOLD) {
            ProcessLargeVocabScatter();
        } else {
            ProcessLargeVocab();
            if (vocabSize_ % (BLOCK_BYTES / sizeof(outputT)) != 0) {
                SyncAll();
                ProcessRemainderPass();
            }
        }
    }
    kInQueue_.FreeTensor(kLocal);
    pInQueue_.FreeTensor(pLocal);
    sortedValueInQueue_.FreeTensor(sortedValueLocal);
    sortedIndicesInQueue_.FreeTensor(sortedIndicesLocal);
    outQueue_.FreeTensor(outTensor);
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::ProcessLargeVocab()
{
    for (uint32_t b = 0; b < loopBatch_; b++) {
        ProcessOneBatchLarge(batchOffset_ + b);
    }
    if (blockIdx_ == lastGroupCore_ && remBatches_ > 0) {
        for (uint32_t b = 0; b < remBatches_; b++) {
            ProcessOneBatchLarge(remOffset_ + b);
        }
    }
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::ProcessOneBatchLarge(uint32_t batchIdx)
{
    baseGmIdx_ = static_cast<int64_t>(batchIdx) * vocabSize_;
    outOffset_ = 0;
    hadGreaterKFirstLoop = false;
    hadGreaterK = false;
    hadGreaterCumsumP = false;
    uint32_t loopBatch = batchIdx - batchOffset_;
    if (vocabSize_ > LARGE_VOCAB_THRESHOLD) {
        if (mode_ == 1) {
            InitProcessTopK(loopBatch);
        } else {
            InitProcess(loopBatch);
        }
        uint32_t remainder = vocabSize_ % (BLOCK_BYTES / sizeof(outputT));
        if (remainder > 0) {
            uint32_t tailEnd = vocabSize_ - remainder;
            for (uint32_t i = 0; i < remainder; i++) {
                mGmOut_.SetValue(baseGmIdx_ + tailEnd + i, outTensor.GetValue(0));
            }
        }
        if (mode_ == 1) {
            ScatterTopKSuffix();
        } else {
            if (calLocalFp32.GetValue(ubFactorElementAligned_) < kthValue) {
                ProcessKLtKMax(loopBatch);
            } else {
                ProcessRemain(loopBatch);
            }
        }
        DataCacheCleanAndInvalid<outputT, CacheLine::ENTIRE_DATA_CACHE>(mGmOut_);
        return;
    }
    if constexpr (IsSameType<inputT, float>::value) {
        Duplicate(outTensor.template ReinterpretCast<int32_t>(), FLOAT32_NEG_INF, vocabSize_);
    } else if constexpr (IsSameType<inputT, half>::value) {
        Duplicate(outTensor.template ReinterpretCast<uint16_t>(), FLOAT16_NEG_INF, vocabSize_);
    } else {
        Duplicate(outTensor.template ReinterpretCast<uint16_t>(), BF16_NEG_INF, vocabSize_);
    }
    VToSSync();
    if (mode_ == 1) {
        InitProcessTopK(loopBatch);
        ScatterTopKSuffix();
    } else {
        InitProcess(loopBatch);
        if (calLocalFp32.GetValue(ubFactorElementAligned_) < kthValue) {
            ProcessKLtKMax(loopBatch);
        } else {
            ProcessRemain(loopBatch);
        }
    }
    SToMTE3Sync();
    constexpr uint32_t outBlockElems = BLOCK_BYTES / sizeof(outputT);
    uint32_t numOutChunks = vocabSize_ / ubFactorElementAligned_;
    for (uint32_t c = 0; c < numOutChunks; c++) {
        DataCopy(mGmOut_[baseGmIdx_ + c * ubFactorElementAligned_], outTensor[c * ubFactorElementAligned_],
                 ubFactorElementAligned_);
    }
    uint32_t tailCount = vocabSize_ - numOutChunks * ubFactorElementAligned_;
    uint32_t tailAligned = (tailCount / outBlockElems) * outBlockElems;
    if (tailAligned > 0) {
        DataCopy(mGmOut_[baseGmIdx_ + numOutChunks * ubFactorElementAligned_],
                 outTensor[numOutChunks * ubFactorElementAligned_], tailAligned);
    }
    uint32_t remainder = vocabSize_ % outBlockElems;
    if (remainder > 0) {
        uint32_t tailEnd = vocabSize_ - remainder;
        DataCopy(mGmRemVals[batchIdx * outBlockElems], outTensor[tailEnd], outBlockElems);
    }
    MTE3ToVSync();
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::ProcessRemainderPass()
{
    constexpr uint32_t outBlockElems = BLOCK_BYTES / sizeof(outputT);
    uint32_t remainder = vocabSize_ % outBlockElems;
    uint32_t tailEnd = vocabSize_ - remainder;
    for (uint32_t b = 0; b < loopBatch_; b++) {
        uint32_t batchIdx = batchOffset_ + b;
        int64_t baseGm = static_cast<int64_t>(batchIdx) * vocabSize_;
        DataCacheCleanAndInvalid<outputT, CacheLine::ENTIRE_DATA_CACHE>(mGmOut_);
        for (uint32_t i = 0; i < remainder; i++) {
            outputT v = mGmRemVals.GetValue(batchIdx * outBlockElems + i);
            mGmOut_.SetValue(baseGm + tailEnd + i, v);
        }
        DataCacheCleanAndInvalid<outputT, CacheLine::ENTIRE_DATA_CACHE>(mGmOut_);
    }
    if (blockIdx_ == lastGroupCore_ && remBatches_ > 0) {
        for (uint32_t b = 0; b < remBatches_; b++) {
            uint32_t batchIdx = remOffset_ + b;
            int64_t baseGm = static_cast<int64_t>(batchIdx) * vocabSize_;
            DataCacheCleanAndInvalid<outputT, CacheLine::ENTIRE_DATA_CACHE>(mGmOut_);
            for (uint32_t i = 0; i < remainder; i++) {
                outputT v = mGmRemVals.GetValue(batchIdx * outBlockElems + i);
                mGmOut_.SetValue(baseGm + tailEnd + i, v);
            }
            DataCacheCleanAndInvalid<outputT, CacheLine::ENTIRE_DATA_CACHE>(mGmOut_);
        }
    }
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::ScatterOne(int32_t gmIndex, outputT value)
{
    if (vocabSize_ > LARGE_VOCAB_THRESHOLD) {
        mGmOut_.SetValue(baseGmIdx_ + gmIndex, value);
    } else {
        outTensor.SetValue(outOffset_ + gmIndex, value);
    }
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::FillNegInf(uint32_t batchIdx)
{
    int64_t baseGm = static_cast<int64_t>(batchIdx) * vocabSize_;
    if constexpr (IsSameType<inputT, float>::value) {
        Duplicate(outTensor.template ReinterpretCast<int32_t>(), FLOAT32_NEG_INF, ubFactorElementAligned_);
    } else if constexpr (IsSameType<inputT, half>::value) {
        Duplicate(outTensor.template ReinterpretCast<uint16_t>(), FLOAT16_NEG_INF, ubFactorElementAligned_);
    } else {
        Duplicate(outTensor.template ReinterpretCast<uint16_t>(), BF16_NEG_INF, ubFactorElementAligned_);
    }
    VToMTE3Sync();
    constexpr uint32_t outBlockElems = BLOCK_BYTES / sizeof(outputT);
    uint32_t numChunks = vocabSize_ / ubFactorElementAligned_;
    for (uint32_t c = 0; c < numChunks; c++) {
        DataCopy(mGmOut_[baseGm + c * ubFactorElementAligned_], outTensor, ubFactorElementAligned_);
    }
    uint32_t tailCount = vocabSize_ - numChunks * ubFactorElementAligned_;
    uint32_t tailAligned = (tailCount / outBlockElems) * outBlockElems;
    if (tailAligned > 0) {
        DataCopy(mGmOut_[baseGm + numChunks * ubFactorElementAligned_], outTensor, tailAligned);
    }
    MTE3ToVSync();
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::ProcessLargeVocabScatter()
{
    for (uint32_t b = 0; b < loopBatch_; b++) {
        FillNegInf(batchOffset_ + b);
    }
    if (blockIdx_ == lastGroupCore_ && remBatches_ > 0) {
        for (uint32_t b = 0; b < remBatches_; b++) {
            FillNegInf(remOffset_ + b);
        }
    }
    SyncAll();
    for (uint32_t phase = 0; phase < 2; phase++) {
        for (uint32_t b = 0; b < loopBatch_; b++) {
            uint32_t batchIdx = batchOffset_ + b;
            if ((batchIdx & 1) == phase) {
                ProcessOneBatchLarge(batchIdx);
            }
        }
        if (blockIdx_ == lastGroupCore_ && remBatches_ > 0) {
            for (uint32_t b = 0; b < remBatches_; b++) {
                uint32_t batchIdx = remOffset_ + b;
                if ((batchIdx & 1) == phase) {
                    ProcessOneBatchLarge(batchIdx);
                }
            }
        }
        SyncAll();
    }
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::ProcessScatter()
{
    uint32_t batchBlocks = batchSize_ / blockNum_;
    for (uint32_t count = 0; count < batchBlocks; count++) {
        int64_t currentBatch = static_cast<int64_t>(blockIdx_) * batchBlocks + count;
        int32_t cumsumGTIndex = mGmCumsumGTIndex.GetValue(currentBatch * DATA_PER_BLOCK_B32);
        int64_t currentBatchIdx = currentBatch * vocabSize_;
        int64_t currentBatchEndIdx = currentBatchIdx + vocabSize_ - 1;
        int32_t scatterLength = currentBatchEndIdx - (currentBatchIdx + cumsumGTIndex) + 1;
        if (scatterLength <= 0) {
            continue;
        }

        int32_t scatterLengthBlocks = (scatterLength + scatterTensorNums_ - 1) / scatterTensorNums_;
        int32_t resScatterLength = scatterLength - (scatterLengthBlocks - 1) * scatterTensorNums_;
        for (int32_t scatterLengthCount = 0; scatterLengthCount < scatterLengthBlocks; scatterLengthCount++) {
            int64_t currentGmIdx = currentBatchIdx + cumsumGTIndex + scatterLengthCount * scatterTensorNums_;
            int32_t dataNums = scatterTensorNums_;
            if (scatterLengthCount == scatterLengthBlocks - 1) {
                dataNums = resScatterLength;
            }
            DataCopy(scatterSortedValueLocal, mGmSortedValue_[currentGmIdx],
                     AscendC::AlignUp(dataNums, BLOCK_BYTES / sizeof(inputT)));
            DataCopy(scatterSortedIndicesLocal, mGmSortedIndices_[currentGmIdx],
                     AscendC::AlignUp(dataNums, DATA_PER_BLOCK_B32));
            MTE2ToSSync();
            for (int32_t loopProb = 0; loopProb < dataNums; loopProb++) {
                scatterValueTensor.SetValue(0, scatterSortedValueLocal.GetValue(loopProb));
                int32_t gmIndex = scatterSortedIndicesLocal.GetValue(loopProb);
                mGmOut_.SetValue(currentBatchIdx + gmIndex,
                                 scatterValueTensor.template ReinterpretCast<outputT>().GetValue(0));
            }
            SToMTE2Sync();
        }
    }
    ProcessResScatter();
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::ProcessResScatter()
{
    uint32_t batchBlocks = batchSize_ / blockNum_;
    uint32_t resbatch = batchSize_ % blockNum_;
    if (resbatch == 0) {
        return;
    }
    uint32_t batchCoreNum = blockNum_ / resbatch;
    if (blockIdx_ > batchCoreNum * resbatch - 1) {
        return;
    }
    int64_t currentBatch = static_cast<int64_t>(batchBlocks) * blockNum_ + blockIdx_ / batchCoreNum;
    int64_t currentBatchIdx = currentBatch * vocabSize_;
    int64_t currentBatchEndIdx = currentBatchIdx + vocabSize_ - 1;
    int32_t cumsumGTIndex = mGmCumsumGTIndex.GetValue(currentBatch * DATA_PER_BLOCK_B32);
    uint32_t batchCoreIdx = blockIdx_ % batchCoreNum;
    int32_t scatterLength = currentBatchEndIdx - (currentBatchIdx + cumsumGTIndex) + 1;
    int32_t scatterCoreLength = (scatterLength + batchCoreNum - 1) / batchCoreNum;
    int64_t currentCoreStartIdx = currentBatchIdx + cumsumGTIndex + scatterCoreLength * batchCoreIdx;
    currentCoreStartIdx = Max(currentCoreStartIdx, currentBatchIdx + cumsumGTIndex + batchCoreIdx);
    scatterCoreLength = Min(scatterCoreLength, static_cast<int32_t>(currentBatchEndIdx - currentCoreStartIdx + 1));
    if (scatterLength <= 0 || currentCoreStartIdx > currentBatchEndIdx) {
        return;
    }

    int32_t scatterLengthBlocks = (scatterCoreLength + scatterTensorNums_ - 1) / scatterTensorNums_;
    int32_t resScatterLength = scatterCoreLength - (scatterLengthBlocks - 1) * scatterTensorNums_;
    for (int32_t scatterLengthCount = 0; scatterLengthCount < scatterLengthBlocks; scatterLengthCount++) {
        int64_t currentGmIdx = currentCoreStartIdx + scatterLengthCount * scatterTensorNums_;
        int32_t dataNums = scatterTensorNums_;
        if (scatterLengthCount == scatterLengthBlocks - 1) {
            dataNums = resScatterLength;
        }
        DataCopy(scatterSortedValueLocal, mGmSortedValue_[currentGmIdx],
                 AscendC::AlignUp(dataNums, BLOCK_BYTES / sizeof(inputT)));
        DataCopy(scatterSortedIndicesLocal, mGmSortedIndices_[currentGmIdx],
                 AscendC::AlignUp(dataNums, DATA_PER_BLOCK_B32));
        MTE2ToSSync();
        for (int32_t loopProb = 0; loopProb < dataNums; loopProb++) {
            scatterValueTensor.SetValue(0, scatterSortedValueLocal.GetValue(loopProb));
            int32_t gmIndex = scatterSortedIndicesLocal.GetValue(loopProb);
            mGmOut_.SetValue(currentBatchIdx + gmIndex,
                             scatterValueTensor.template ReinterpretCast<outputT>().GetValue(0));
        }
    }
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::InitCopyIn(uint32_t loopBatch,
                                                                                  int64_t currentGmIdx)
{
    scatterIdxTensor.SetValue(0, static_cast<int32_t>(vocabSize_));
    SToMTE3Sync();
    DataCopy(mGmCumsumGTIndex[(batchOffset_ + loopBatch) * DATA_PER_BLOCK_B32], scatterIdxTensor, DATA_PER_BLOCK_B32);
    MTE3ToSSync();
    DataCopy(sortedValueLocal[ubFactorElementAligned_], mGmSortedValue_[currentGmIdx],
             AscendC::AlignUp(dataNumInit_, BLOCK_BYTES / sizeof(inputT)));
    DataCopy(pLocal, mGmP_[batchOffset_ + loopBatch], BLOCK_BYTES / sizeof(inputT));
    if constexpr (!IsSameType<inputT, float>::value) {
        MTE2ToVSync();
        Cast(sortedValueLocalFp32[ubFactorElementAligned_], sortedValueLocal[ubFactorElementAligned_],
             RoundMode::CAST_NONE, dataNumInit_);
        Cast(tmpLocal, pLocal, RoundMode::CAST_NONE, DATA_PER_BLOCK_B32);
    }
    DataCopy(sortedIndicesLocal[ubFactorElementAligned_], mGmSortedIndices_[currentGmIdx],
             AscendC::AlignUp(dataNumInit_, DATA_PER_BLOCK_B32));
    if (mode_ != 2) {
        DataCopy(kLocal, mGmK_[batchOffset_ + loopBatch], DATA_PER_BLOCK_B32);
    }
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::GetKthResult(uint32_t loopBatch, uint32_t offset,
                                                                                    uint8_t repeatTimes)
{
    Compare(tmpLocal.template ReinterpretCast<uint8_t>(), kthValueLocal, calLocalFp32[offset], CMPMODE::GT, MASK_64,
            repeatTimes, repeatParams);
    PipeBarrier<PIPE_V>();
    Select(calLocalFp32[offset], tmpLocal.template ReinterpretCast<uint8_t>(), negInfLocal, calLocalFp32[offset],
           SELMODE::VSEL_TENSOR_TENSOR_MODE, MASK_64, repeatTimes, repeatParams);
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::ReduceSumWithAddsAndExpImpl(uint32_t offset,
                                                                                                   uint32_t loopDataNum)
{
    Adds(softMaxRes, calLocalFp32[offset], maxValue, loopDataNum);
    PipeBarrier<PIPE_V>();
    Exp(softMaxRes, softMaxRes, loopDataNum);
    PipeBarrier<PIPE_V>();
    ReduceSum(reduceLocal, softMaxRes, reduceLocal, loopDataNum);
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::InitProcess(uint32_t loopBatch)
{
    int64_t initGmIdx = baseGmIdx_ + vocabSize_ - dataNumInit_;
    InitCopyIn(loopBatch, initGmIdx);
    MTE2ToSSync();

    if (mode_ == 2) {
        kthValue = -1e30f;
    } else {
        int32_t kValue = kLocal.GetValue(0);
        if constexpr (IsSameType<inputT, float>::value) {
            kthValue = mGmSortedValue_[baseGmIdx_ + vocabSize_ - kValue].GetValue(0);
        } else {
            kthValue = static_cast<float>(mGmSortedValue_[baseGmIdx_ + vocabSize_ - kValue].GetValue(0));
        }
    }
    if constexpr (IsSameType<inputT, float>::value) {
        pValue = float(1.0) - pLocal.GetValue(0);
    } else {
        pValue = float(1.0) - tmpLocal.GetValue(0);
    }
    maxValue = -calLocalFp32[ubFactorElementAligned_].GetValue(dataNumInit_ - 1);
    Duplicate(kthValueLocal, kthValue, 8);
    PipeBarrier<PIPE_V>();

    uint8_t repeatTimes = (dataNumInit_ + DATA_PER_REPEAT_B32 - 1) / DATA_PER_REPEAT_B32;
    GetKthResult(loopBatch, ubFactorElementAligned_, repeatTimes);
    PipeBarrier<PIPE_V>();

    ReduceSumWithAddsAndExpImpl(ubFactorElementAligned_, dataNumInit_);
    VToSSync();
    reduceSumValue = reduceLocal.GetValue(0);
    reduceSumValueInvert = 1 / reduceSumValue;
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::ProcessKLtKMax(uint32_t loopBatch)
{
    Muls(softMaxRes, softMaxRes, reduceSumValueInvert, dataNumInit_);
    PipeBarrier<PIPE_V>();
    const CumSumInfo cumSumInfo{1, dataNumInitAligned_};
    CumSum<float, CUMSUM_CONFIG>(cumSumRes, cumSumTmp, softMaxRes, sharedTmpBuffer, cumSumInfo);
    VToSSync();
    int32_t loopProb = dataNumInit_ - 1;
    scatterTensor.SetValue(0, sortedValueLocal[ubFactorElementAligned_].GetValue(loopProb));
    int32_t gmIndex = sortedIndicesLocal[ubFactorElementAligned_].GetValue(loopProb);
    ScatterOne(gmIndex, scatterTensor.template ReinterpretCast<outputT>().GetValue(0));
    loopProb = loopProb - 1;
    for (; loopProb >= 0; loopProb--) {
        float cumsumData = cumSumRes.GetValue(loopProb);
        if (cumsumData <= pValue) {
            break;
        }
        scatterTensor.SetValue(0, sortedValueLocal[ubFactorElementAligned_].GetValue(loopProb));
        gmIndex = sortedIndicesLocal[ubFactorElementAligned_].GetValue(loopProb);
        ScatterOne(gmIndex, scatterTensor.template ReinterpretCast<outputT>().GetValue(0));
    }
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::ScatterCumtomImpl(uint32_t loopBatch,
                                                                                         uint32_t loopProbNum,
                                                                                         uint32_t offset)
{
    for (int32_t loopProb = 0; loopProb < static_cast<int32_t>(loopProbNum); loopProb++) {
        float cumsumDataTmp = cumSumRes.GetValue(loopProb);
        if (cumsumDataTmp <= pValue && !hadGreaterCumsumP) {
            continue;
        }
        scatterTensor.SetValue(0, sortedValueLocal[offset].GetValue(loopProb));
        int32_t gmIndex = sortedIndicesLocal[offset].GetValue(loopProb);
        ScatterOne(gmIndex, scatterTensor.template ReinterpretCast<outputT>().GetValue(0));
    }
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::GetFirstKLoop(uint32_t loopBatch,
                                                                                     int32_t& firstKLoop)
{
    uint8_t repeatTimes = (dataNumInit_ + DATA_PER_REPEAT_B32 - 1) / DATA_PER_REPEAT_B32;
    uint32_t loopDataNum = ubFactorElementAligned_;
    for (int32_t loopInner = 0; loopInner < loopInner_; loopInner++) {
        int64_t currentGmIdx = baseGmIdx_ + loopInner * ubFactorElementAligned_;
        if (loopInner == (loopInner_ - 1)) {
            repeatTimes = ((tailUbFactorElement_) + DATA_PER_REPEAT_B32 - 1) / DATA_PER_REPEAT_B32;
            loopDataNum = tailUbFactorElement_;
        }
        DataCopy(sortedValueLocal.template ReinterpretCast<inputT>(), mGmSortedValue_[currentGmIdx],
                 AscendC::AlignUp(loopDataNum, BLOCK_BYTES / sizeof(inputT)));
        if constexpr (!IsSameType<inputT, float>::value) {
            MTE2ToVSync();
            Cast(sortedValueLocalFp32, sortedValueLocal, RoundMode::CAST_NONE, loopDataNum);
            VToSSync();
        } else {
            MTE2ToSSync();
        }
        if (calLocalFp32.GetValue(loopDataNum - 1) < kthValue) {
            firstKLoop += 1;
            continue;
        }
        SToVSync();
        if (!hadGreaterKFirstLoop) {
            GetKthResult(loopBatch, 0, repeatTimes);
            PipeBarrier<PIPE_V>();
            hadGreaterKFirstLoop = true;
        }
        ReduceSumWithAddsAndExpImpl(0, loopDataNum);
        VToSSync();
        reduceSumValue += reduceLocal.GetValue(0);
    }
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::CumSumWithAddsAndExpImpl(uint32_t offset,
                                                                                                uint32_t loopDataNum,
                                                                                                uint32_t cumsumInner,
                                                                                                float cumsumData)
{
    Adds(softMaxRes, calLocalFp32[offset], maxValue, loopDataNum);
    PipeBarrier<PIPE_V>();
    Exp(softMaxRes, softMaxRes, loopDataNum);
    PipeBarrier<PIPE_V>();
    Muls(softMaxRes, softMaxRes, reduceSumValueInvert, loopDataNum);
    PipeBarrier<PIPE_V>();
    const CumSumInfo cumSumInfo{1, cumsumInner};
    CumSum<float, CUMSUM_CONFIG>(cumSumRes, cumSumTmp, softMaxRes, sharedTmpBuffer, cumSumInfo);
    PipeBarrier<PIPE_V>();
    Adds(cumSumRes, cumSumRes, cumsumData, loopDataNum);
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::ProcessRemain(uint32_t loopBatch)
{
    int32_t firstKLoop = 0;
    GetFirstKLoop(loopBatch, firstKLoop);
    reduceSumValueInvert = 1 / reduceSumValue;
    float cumsumData = 0;
    ScatterFromFirstKLoop(loopBatch, firstKLoop, cumsumData);
    uint32_t loopProb = dataNumInit_ - 1;
    scatterTensor.SetValue(0, sortedValueLocal[ubFactorElementAligned_].GetValue(loopProb));
    int32_t gmIndex = sortedIndicesLocal[ubFactorElementAligned_].GetValue(loopProb);
    ScatterOne(gmIndex, scatterTensor.template ReinterpretCast<outputT>().GetValue(0));
    if (!hadGreaterCumsumP) {
        CumSumWithAddsAndExpImpl(ubFactorElementAligned_, dataNumInit_, dataNumInitAligned_, cumsumData);
        VToSSync();
    }
    ScatterCumtomImpl(loopBatch, dataNumInit_ - 1, ubFactorElementAligned_);
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::ScatterFromFirstKLoop(uint32_t loopBatch,
                                                                                             int32_t firstKLoop,
                                                                                             float& cumsumData)
{
    uint32_t loopDataNum = ubFactorElementAligned_;
    uint32_t cumsumInner = ubFactorElementAligned_;
    uint8_t repeatTimes = ((ubFactorElementAligned_) + DATA_PER_REPEAT_B32 - 1) / DATA_PER_REPEAT_B32;
    for (int32_t loopInner = firstKLoop; loopInner < loopInner_; loopInner++) {
        int64_t currentGmIdx = baseGmIdx_ + loopInner * ubFactorElementAligned_;
        if (loopInner == (loopInner_ - 1)) {
            repeatTimes = (tailUbFactorElement_ + DATA_PER_REPEAT_B32 - 1) / DATA_PER_REPEAT_B32;
            loopDataNum = tailUbFactorElement_;
            cumsumInner = tailUbFactorElementAligned_;
        }
        DataCopy(sortedValueLocal.template ReinterpretCast<inputT>(), mGmSortedValue_[currentGmIdx],
                 AscendC::AlignUp(loopDataNum, BLOCK_BYTES / sizeof(inputT)));
        DataCopy(sortedIndicesLocal, mGmSortedIndices_[currentGmIdx],
                 AscendC::AlignUp(loopDataNum, DATA_PER_BLOCK_B32));
        if constexpr (!IsSameType<inputT, float>::value) {
            MTE2ToVSync();
            Cast(sortedValueLocalFp32, sortedValueLocal, RoundMode::CAST_NONE, loopDataNum);
            PipeBarrier<PIPE_V>();
        } else {
            MTE2ToVSync();
        }

        if (!hadGreaterK) {
            GetKthResult(loopBatch, 0, repeatTimes);
            PipeBarrier<PIPE_V>();
            hadGreaterK = true;
        }

        if (!hadGreaterCumsumP) {
            CumSumWithAddsAndExpImpl(0, loopDataNum, cumsumInner, cumsumData);
            VToSSync();
            float cumsumDataTmp = cumSumRes.GetValue(loopDataNum - 1);
            cumsumData = cumsumDataTmp;
            if (cumsumDataTmp <= pValue) {
                continue;
            }
            SetCumsumGTIndex(loopBatch, static_cast<int32_t>(loopInner * ubFactorElementAligned_ + loopDataNum));
        } else {
            MTE2ToSSync();
        }
        ScatterCumtomImpl(loopBatch, loopDataNum, 0);
        hadGreaterCumsumP = true;
    }
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::SetCumsumGTIndex(uint32_t loopBatch,
                                                                                        int32_t index)
{
    scatterIdxTensor.SetValue(0, index);
    SToMTE3Sync();
    DataCopy(mGmCumsumGTIndex[(batchOffset_ + loopBatch) * DATA_PER_BLOCK_B32], scatterIdxTensor, DATA_PER_BLOCK_B32);
    MTE3ToSSync();
}

#if __CCE_AICORE__ == 200
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::ProcessTopK()
{
    mode_ = 1;
    outTensor = outQueue_.AllocTensor<outputT>();
    sortedValueLocal = sortedValueInQueue_.AllocTensor<inputT>();
    sortedIndicesLocal = sortedIndicesInQueue_.AllocTensor<int32_t>();
    if (vocabSize_ > LARGE_VOCAB_THRESHOLD) {
        ProcessLargeVocabScatter();
    } else {
        ProcessLargeVocab();
        if (vocabSize_ % (BLOCK_BYTES / sizeof(outputT)) != 0) {
            SyncAll();
            ProcessRemainderPass();
        }
    }
    outQueue_.FreeTensor(outTensor);
    sortedValueInQueue_.FreeTensor(sortedValueLocal);
    sortedIndicesInQueue_.FreeTensor(sortedIndicesLocal);
}
#else
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::ProcessTopK()
{
    kLocal = kInQueue_.AllocTensor<int32_t>();
    outTensor = outQueue_.AllocTensor<outputT>();
    sortedValueLocal = sortedValueInQueue_.AllocTensor<inputT>();
    sortedIndicesLocal = sortedIndicesInQueue_.AllocTensor<int32_t>();
    Duplicate(negInfLocal.template ReinterpretCast<int32_t>(), FLOAT32_NEG_INF, DATA_PER_BLOCK_B32);
    if constexpr (IsSameType<inputT, float>::value) {
        calLocalFp32 = sortedValueLocal;
        Duplicate(outTensor.template ReinterpretCast<int32_t>(), FLOAT32_NEG_INF, ubFactorElementAligned_);
    } else if constexpr (IsSameType<inputT, half>::value) {
        calLocalFp32 = sortedValueLocalFp32;
        Duplicate(outTensor.template ReinterpretCast<uint16_t>(), FLOAT16_NEG_INF, ubFactorElementAligned_);
    } else {
        calLocalFp32 = sortedValueLocalFp32;
        Duplicate(outTensor.template ReinterpretCast<uint16_t>(), BF16_NEG_INF, ubFactorElementAligned_);
    }
    VToSSync();
    for (uint32_t loopBatch = 0; loopBatch < loopBatch_; loopBatch++) {
        baseGmIdx_ = batchOffset_ * vocabSize_ + loopBatch * vocabSize_;
        hadGreaterKFirstLoop = false;
        hadGreaterK = false;
        InitProcessTopK(loopBatch);
        if (calLocalFp32.GetValue(ubFactorElementAligned_) < kthValue) {
            ProcessKLtKMaxTopK(loopBatch);
        } else {
            ProcessRemainTopK(loopBatch);
        }
    }
    SyncAll();
    ProcessScatter();
    kInQueue_.FreeTensor(kLocal);
    sortedValueInQueue_.FreeTensor(sortedValueLocal);
    sortedIndicesInQueue_.FreeTensor(sortedIndicesLocal);
    outQueue_.FreeTensor(outTensor);
}
#endif

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::ProcessRemainTopK(uint32_t loopBatch)
{
    int32_t firstKLoop = 0;
    GetFirstKLoopTopK(loopBatch, firstKLoop);
    // Start the scatter calculation from the first loop in the row where the value is >= kthValue.
    ScatterFromFirstKLoopTopK(loopBatch, firstKLoop);
    /* Perform scatter calculation on the maximum number of ubFactorElementAligned_,
       which does not overlap with the previous ones.*/
    uint32_t loopProb = dataNumInit_ - 1;
    scatterTensor.SetValue(0, sortedValueLocal[ubFactorElementAligned_].GetValue(loopProb));
    int32_t gmIndex = sortedIndicesLocal[ubFactorElementAligned_].GetValue(loopProb);
    mGmOut_.SetValue(baseGmIdx_ + gmIndex, scatterTensor.template ReinterpretCast<outputT>().GetValue(0));
    if (hadGreaterK) {
        return;
    }
    ScatterCumtomImplTopK(loopBatch, dataNumInit_ - 1, ubFactorElementAligned_);
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::GetFirstKLoopTopK(uint32_t loopBatch,
                                                                                         int32_t& firstKLoop)
{
    uint8_t repeatTimes = (dataNumInit_ + DATA_PER_REPEAT_B32 - 1) / DATA_PER_REPEAT_B32;
    uint32_t loopDataNum = ubFactorElementAligned_;
    for (int32_t loopInner = 0; loopInner < loopInner_; loopInner++) {
        int64_t currentGmIdx = baseGmIdx_ + loopInner * ubFactorElementAligned_;
        if (loopInner == (loopInner_ - 1)) {
            repeatTimes = ((tailUbFactorElement_) + DATA_PER_REPEAT_B32 - 1) / DATA_PER_REPEAT_B32;
            loopDataNum = tailUbFactorElement_;
        }
        CopyOutToGM(mGmOut_[currentGmIdx], outTensor, loopDataNum);
        if (hadGreaterKFirstLoop) {
            continue;
        }
        DataCopy(sortedValueLocal.template ReinterpretCast<inputT>(), mGmSortedValue_[currentGmIdx],
                 AscendC::AlignUp(loopDataNum, BLOCK_BYTES / sizeof(inputT)));
        MTE2ToSSync();
        float rightVlaue = 0;
        // Make a judgment on the rightmost value of each loop to filter the data.
        if constexpr (IsSameType<inputT, bfloat16_t>::value) {
            rightVlaue = static_cast<float>(sortedValueLocal.GetValue(loopDataNum - 1));
        } else {
            rightVlaue = static_cast<float>(sortedValueLocal.GetValue(loopDataNum - 1));
        }
        SToMTE2Sync();
        if (rightVlaue < kthValue) {
            firstKLoop += 1;
            continue;
        }
        hadGreaterKFirstLoop = true;
    }
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::ScatterFromFirstKLoopTopK(uint32_t loopBatch,
                                                                                                 int32_t firstKLoop)
{
    uint32_t loopDataNum = ubFactorElementAligned_;
    uint32_t cumsumInner = ubFactorElementAligned_;
    uint8_t repeatTimes = ((ubFactorElementAligned_) + DATA_PER_REPEAT_B32 - 1) / DATA_PER_REPEAT_B32;
    for (int32_t loopInner = firstKLoop; loopInner < loopInner_; loopInner++) {
        int64_t currentGmIdx = baseGmIdx_ + loopInner * ubFactorElementAligned_;
        if (loopInner == (loopInner_ - 1)) {
            repeatTimes = (tailUbFactorElement_ + DATA_PER_REPEAT_B32 - 1) / DATA_PER_REPEAT_B32;
            loopDataNum = tailUbFactorElement_;
            cumsumInner = tailUbFactorElementAligned_;
        }
        DataCopy(sortedValueLocal.template ReinterpretCast<inputT>(), mGmSortedValue_[currentGmIdx],
                 AscendC::AlignUp(loopDataNum, BLOCK_BYTES / sizeof(inputT)));
        if constexpr (!IsSameType<inputT, float>::value) {
            MTE2ToVSync();
            Cast(sortedValueLocalFp32, sortedValueLocal, RoundMode::CAST_NONE, loopDataNum);
            VToSSync();
        }
        DataCopy(sortedIndicesLocal, mGmSortedIndices_[currentGmIdx],
                 AscendC::AlignUp(loopDataNum, DATA_PER_BLOCK_B32));
        MTE2ToSSync();

        SetCumsumGTIndex(loopBatch, static_cast<int32_t>(loopInner * ubFactorElementAligned_ + loopDataNum));
        ScatterCumtomImplTopK(loopBatch, loopDataNum, 0);
        hadGreaterK = true;
        break;
    }
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::ScatterCumtomImplTopK(uint32_t loopBatch,
                                                                                             uint32_t loopProbNum,
                                                                                             uint32_t offset)
{
    // Reverse traversal, returning early to improve performance.
    for (int32_t loopProb = static_cast<int32_t>(loopProbNum) - 1; loopProb >= 0; loopProb--) {
        float curValue = calLocalFp32[offset].GetValue(loopProb);
        if (curValue < kthValue) {
            break;
        }
        scatterTensor.SetValue(0, sortedValueLocal[offset].GetValue(loopProb));
        int32_t gmIndex = sortedIndicesLocal[offset].GetValue(loopProb);
        mGmOut_.SetValue(baseGmIdx_ + gmIndex, scatterTensor.template ReinterpretCast<outputT>().GetValue(0));
    }
}

#if __CCE_AICORE__ == 200
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::InitProcessTopK(uint32_t loopBatch)
{
    int32_t kValue = mGmK_.GetValue(batchOffset_ + loopBatch);
    if (kValue < 0) {
        kValue = 0;
    }
    if (kValue > static_cast<int32_t>(vocabSize_)) {
        kValue = static_cast<int32_t>(vocabSize_);
    }
    uint32_t kEff = static_cast<uint32_t>(kValue);
    if (kEff == 0) {
        kthValue = 1e30f;
    } else {
        if constexpr (IsSameType<inputT, float>::value) {
            kthValue = mGmSortedValue_[baseGmIdx_ + vocabSize_ - kEff].GetValue(0);
        } else {
            kthValue = static_cast<float>(mGmSortedValue_[baseGmIdx_ + vocabSize_ - kEff].GetValue(0));
        }
    }
}

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::ScatterTopKSuffix()
{
    bool started = false;
    uint32_t totalChunks = (vocabSize_ + ubFactorElementAligned_ - 1) / ubFactorElementAligned_;
    for (uint32_t c = 0; c < totalChunks; c++) {
        uint32_t chunkBase = c * ubFactorElementAligned_;
        uint32_t chunkLen = Min(ubFactorElementAligned_, vocabSize_ - chunkBase);
        DataCopy(sortedValueLocal.template ReinterpretCast<inputT>(), mGmSortedValue_[baseGmIdx_ + chunkBase],
                 AscendC::AlignUp(chunkLen, BLOCK_BYTES / sizeof(inputT)));
        DataCopy(sortedIndicesLocal, mGmSortedIndices_[baseGmIdx_ + chunkBase],
                 AscendC::AlignUp(chunkLen, DATA_PER_BLOCK_B32));
        if constexpr (!IsSameType<inputT, float>::value) {
            MTE2ToVSync();
            Cast(sortedValueLocalFp32, sortedValueLocal, RoundMode::CAST_NONE, chunkLen);
            VToSSync();
        } else {
            MTE2ToSSync();
        }
        for (uint32_t i = 0; i < chunkLen; i++) {
            float val;
            if constexpr (IsSameType<inputT, float>::value) {
                val = sortedValueLocal.GetValue(i);
            } else {
                val = sortedValueLocalFp32.GetValue(i);
            }
            if (!started && val < kthValue) {
                continue;
            }
            started = true;
            scatterTensor.SetValue(0, sortedValueLocal.GetValue(i));
            int32_t gmIndex = sortedIndicesLocal.GetValue(i);
            ScatterOne(gmIndex, scatterTensor.template ReinterpretCast<outputT>().GetValue(0));
        }
        SToMTE2Sync();
    }
}
#else
template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::InitProcessTopK(uint32_t loopBatch)
{
    scatterIdxTensor.SetValue(0, static_cast<int32_t>(vocabSize_));
    SToMTE3Sync();
    DataCopy(mGmCumsumGTIndex[(batchOffset_ + loopBatch) * DATA_PER_BLOCK_B32], scatterIdxTensor, DATA_PER_BLOCK_B32);
    MTE3ToSSync();
    int64_t initGmIdx = baseGmIdx_ + vocabSize_ - dataNumInit_;
    CopyOutToGM(mGmOut_[initGmIdx], outTensor, dataNumInit_);
    DataCopy(sortedValueLocal[ubFactorElementAligned_], mGmSortedValue_[initGmIdx],
             AscendC::AlignUp(dataNumInit_, BLOCK_BYTES / sizeof(inputT)));
    if constexpr (!IsSameType<inputT, float>::value) {
        MTE2ToVSync();
        Cast(sortedValueLocalFp32[ubFactorElementAligned_], sortedValueLocal[ubFactorElementAligned_],
             RoundMode::CAST_NONE, dataNumInit_);
    }
    DataCopy(sortedIndicesLocal[ubFactorElementAligned_], mGmSortedIndices_[initGmIdx],
             AscendC::AlignUp(dataNumInit_, DATA_PER_BLOCK_B32));
    DataCopy(kLocal, mGmK_[batchOffset_ + loopBatch], DATA_PER_BLOCK_B32);
    MTE2ToSSync();
    int32_t kValue = mGmK_.GetValue(batchOffset_ + loopBatch);
    maxValue = -calLocalFp32[ubFactorElementAligned_].GetValue(dataNumInit_ - 1);
    if constexpr (IsSameType<inputT, float>::value) {
        kthValue = mGmSortedValue_[baseGmIdx_ + vocabSize_ - kValue].GetValue(0);
    } else if constexpr (IsSameType<inputT, half>::value) {
        kthValue = static_cast<float>(mGmSortedValue_[baseGmIdx_ + vocabSize_ - kValue].GetValue(0));
    } else {
        kthValue = static_cast<float>(mGmSortedValue_[baseGmIdx_ + vocabSize_ - kValue].GetValue(0));
    }
}
#endif

template <typename inputT, typename calT, typename outputT>
__aicore__ inline void ApplyTopKTopPWithSorted<inputT, calT, outputT>::ProcessKLtKMaxTopK(uint32_t loopBatch)
{
    // Move out -infinity to fill GM
    for (int32_t loopInner = 0; loopInner < loopInner_; loopInner++) {
        int64_t currentGmIdxInner = baseGmIdx_ + loopInner * ubFactorElementAligned_;
        if (loopInner == loopInner_ - 1) {
            CopyOutToGM(mGmOut_[currentGmIdxInner], outTensor, tailUbFactorElement_);
        } else {
            CopyOutToGM(mGmOut_[currentGmIdxInner], outTensor, ubFactorElementAligned_);
        }
    }
    // Scatter calculation
    int32_t loopProb = dataNumInit_ - 1;
    scatterTensor.SetValue(0, sortedValueLocal[ubFactorElementAligned_].GetValue(loopProb));
    int32_t gmIndex = sortedIndicesLocal[ubFactorElementAligned_].GetValue(loopProb);
    mGmOut_.SetValue(baseGmIdx_ + gmIndex, scatterTensor.template ReinterpretCast<outputT>().GetValue(0));
    loopProb = loopProb - 1;

    for (; loopProb >= 0; loopProb--) {
        float curValue = calLocalFp32[ubFactorElementAligned_].GetValue(loopProb);
        if (curValue < kthValue) {
            break;
        }
        scatterTensor.SetValue(0, sortedValueLocal[ubFactorElementAligned_].GetValue(loopProb));
        gmIndex = sortedIndicesLocal[ubFactorElementAligned_].GetValue(loopProb);
        mGmOut_.SetValue(baseGmIdx_ + gmIndex, scatterTensor.template ReinterpretCast<outputT>().GetValue(0));
    }
}

} // namespace ApplyTopKTopPWithSortedOp

#endif // APPLY_TOP_K_TOP_P_WITH_SORTED_H_KERNEL
