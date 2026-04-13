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
 * \file uss_deterministic_big_innerdim.h
 * \brief uss_deterministic_big_innerdim
 */

#ifndef ASCENDC_USS_DETERMINISTIC_BIG_INNERDIM_H_
#define ASCENDC_USS_DETERMINISTIC_BIG_INNERDIM_H_

#include "kernel_operator.h"
#include "unsorted_segment_base.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace UnsortedSegmentSum {
using namespace AscendC;

constexpr uint32_t UB_AGLIN = 32;
constexpr uint32_t DOUBLE_BUFFER = 2;

template <typename X_T, typename IDS_T>
class USSKernelDeterministicBigInnerDim
{
public:
    __aicore__ inline USSKernelDeterministicBigInnerDim(const UnsortedSegmentSumDeterministicBigInnerDimTilingData* tiling, TPipe* pipe)
        : td_(tiling), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output);
    __aicore__ inline void Process();
    __aicore__ inline void CopyInData(uint64_t blockOffset, uint64_t sLoop, uint64_t aLoop, 
        uint64_t cols, uint64_t rows, uint64_t colsAlign, int64_t localOffset);

private:
    TPipe* pipe_ = nullptr;
    const UnsortedSegmentSumDeterministicBigInnerDimTilingData* td_;
    GlobalTensor<X_T> xGm_;
    GlobalTensor<X_T> yGm_;
    AscendC::GlobalTensor<X_T> yGmInit_;
    GlobalTensor<IDS_T> idsGm_;
    TQue<QuePosition::VECIN, 1> xQueue_;
    TQue<QuePosition::VECIN, 1> segmentIdQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> yQueue_;  
    TBuf<QuePosition::VECCALC> sharedTmpBuf;
    TBuf<QuePosition::VECCALC> sortedIdBuf;
    TBuf<QuePosition::VECCALC> sortedIdIndexBuf;

    uint32_t blockId_;
    uint32_t innerDim_{1};
    uint64_t inputOuterDim_{0};
    uint64_t outputOuterDim_{0};
    uint64_t normalCoreProcessCols_{0};
    uint64_t tailCoreProcessCols_{0};
    uint64_t baseS_{1};
    uint64_t baseA_{1};
    uint32_t sortSharedBufSize_{0};
    uint64_t initPerCore_{0};
    uint64_t initCoreReal_{0};
    int64_t xQueSize_{0};
};

template <typename X_T, typename IDS_T>
__aicore__ inline void USSKernelDeterministicBigInnerDim<X_T, IDS_T>::Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output)
{
    blockId_ = GetBlockIdx();
    innerDim_ = td_->innerDim;
    inputOuterDim_ = td_->inputOuterDim;
    outputOuterDim_ = td_->outputOuterDim;
    normalCoreProcessCols_ = td_->normalCoreProcessCols;
    tailCoreProcessCols_ = td_->tailCoreProcessCols;
    baseS_ = td_->baseS;
    baseA_ = td_->baseA;
    sortSharedBufSize_ = td_->sortSharedBufSize;

    xGm_.SetGlobalBuffer((__gm__ X_T*)(x));
    idsGm_.SetGlobalBuffer((__gm__ IDS_T*)(segmentIds));
    yGm_.SetGlobalBuffer((__gm__ X_T*)(output));
    
    // yGm分多核清零
    uint32_t coreNum = GetBlockNum();
    initPerCore_ = outputOuterDim_ * innerDim_ / coreNum;
    uint32_t initTailCore = outputOuterDim_ * innerDim_ - (coreNum - 1) * initPerCore_;
    initCoreReal_ = blockId_ == (coreNum - 1) ? initTailCore : initPerCore_;
    uint64_t yGmOffset = blockId_ * initPerCore_;
    yGmInit_.SetGlobalBuffer((__gm__ X_T*)(output) + yGmOffset);
    InitGlobalMemory(yGmInit_, initCoreReal_, (X_T)0);
    SyncAll();
    pipe_->Reset();

    uint32_t xQueueSize = Ops::Base::CeilAlign(static_cast<uint32_t>(baseS_* baseA_ * sizeof(float)), UB_AGLIN);
    uint32_t segmentIdQueueSize = Ops::Base::CeilAlign(static_cast<uint32_t>(baseS_* sizeof(IDS_T)), UB_AGLIN);
    uint32_t sortedIdIndexBufSize = Ops::Base::CeilAlign(static_cast<uint32_t>(baseS_* sizeof(uint32_t)), UB_AGLIN);
    xQueSize_ = xQueueSize;
    pipe_->InitBuffer(xQueue_, 1, xQueueSize);
    pipe_->InitBuffer(segmentIdQueue_, 1, segmentIdQueueSize);
    pipe_->InitBuffer(yQueue_, DOUBLE_BUFFER, baseA_ * sizeof(float));
    pipe_->InitBuffer(sharedTmpBuf, sortSharedBufSize_);
    pipe_->InitBuffer(sortedIdBuf, segmentIdQueueSize);
    pipe_->InitBuffer(sortedIdIndexBuf, sortedIdIndexBufSize);
}

template <typename X_T, typename IDS_T>
__aicore__ inline void USSKernelDeterministicBigInnerDim<X_T, IDS_T>::CopyInData(uint64_t blockOffset, uint64_t sLoop, 
    uint64_t aLoop, uint64_t cols, uint64_t rows, uint64_t colsAlign, int64_t localOffset)
{
    // copy in x 
    LocalTensor<X_T> xLocal = xQueue_.AllocTensor<X_T>();
    uint64_t inputOffset = blockOffset + sLoop * baseS_ * innerDim_ + aLoop * baseA_;  
    DataCopyExtParams dataCopyParam{(uint16_t)rows, (uint32_t)(cols * sizeof(X_T)), (uint32_t)((innerDim_ - cols) * sizeof(X_T)), 
        (uint32_t)((colsAlign - cols) * sizeof(X_T) / ONE_BLOCK_SIZE), 0};
    DataCopyPadExtParams<X_T> dataPadParam{false, 0, 0, 0};
    DataCopyPad(xLocal[localOffset], xGm_[inputOffset], dataCopyParam, dataPadParam);
    xQueue_.EnQue<X_T>(xLocal);

    // copy in segmentId 
    LocalTensor<IDS_T> segmentIdLocal = segmentIdQueue_.AllocTensor<IDS_T>();
    DataCopyExtParams idCopyParam{(uint16_t)1, (uint32_t)(rows * sizeof(IDS_T)), 0, 0, 0};
    DataCopyPadExtParams<IDS_T> idPadParam{false, 0, 0, 0};
    DataCopyPad(segmentIdLocal, idsGm_[sLoop * baseS_], idCopyParam, idPadParam);
    segmentIdQueue_.EnQue(segmentIdLocal);
}

template <typename X_T, typename IDS_T>
__aicore__ inline void USSKernelDeterministicBigInnerDim<X_T, IDS_T>::Process()
{
    if (GetBlockIdx() >= GetBlockNum()) {
        return;
    }

    int64_t xLocalOffsetStart = 0;
    if constexpr (std::negation<std::is_same<X_T, float>>::value) {
        xLocalOffsetStart = xQueSize_ / sizeof(X_T) / DOUBLE;
    }

    uint64_t curCoreCols = GetBlockIdx() != (GetBlockNum() - 1) ? normalCoreProcessCols_ : tailCoreProcessCols_;
    uint64_t blockOffset = GetBlockIdx() * normalCoreProcessCols_;  
    uint64_t aLoopNum = Ops::Base::CeilDiv(curCoreCols, baseA_);
    uint64_t sLoopNum = Ops::Base::CeilDiv(inputOuterDim_, baseS_);

    for (uint64_t aLoop = 0; aLoop < aLoopNum; aLoop++) {
        uint64_t cols = (aLoop == aLoopNum - 1) ? (curCoreCols - aLoop * baseA_) : baseA_;
        uint64_t colsAlign = Ops::Base::CeilAlign(static_cast<uint64_t>(cols * sizeof(X_T)), ONE_BLOCK_SIZE) / sizeof(X_T);
        for (uint64_t sLoop = 0; sLoop < sLoopNum; sLoop++) {
            uint64_t rows = (sLoop == sLoopNum - 1) ? (inputOuterDim_ - sLoop * baseS_) : baseS_;
            CopyInData(blockOffset, sLoop, aLoop, cols, rows, colsAlign, xLocalOffsetStart);
            LocalTensor<X_T> xLocal = xQueue_.DeQue<X_T>();

            LocalTensor<float> xLocalFloat = xLocal.template ReinterpretCast<float>();
            if constexpr (std::negation<std::is_same<X_T, float>>::value) {
                AscendC::Cast(xLocalFloat, xLocal[xLocalOffsetStart], AscendC::RoundMode::CAST_NONE, rows * colsAlign);
            }
            LocalTensor<IDS_T> segmentIdTensor = segmentIdQueue_.DeQue<IDS_T>();

            LocalTensor<IDS_T> sortedIdTensor = sortedIdBuf.Get<IDS_T>();
            LocalTensor<uint32_t> sortedIdIndexTensor = sortedIdIndexBuf.Get<uint32_t>();
            LocalTensor<uint8_t> sharedTmpTensor = sharedTmpBuf.Get<uint8_t>();
            AscendC::Sort<IDS_T, false, sortConfig>(sortedIdTensor, sortedIdIndexTensor, segmentIdTensor, sharedTmpTensor,
                                                static_cast<uint32_t>(rows));
            
            LocalTensor<float> yLocal = yQueue_.AllocTensor<float>();
            Duplicate(yLocal, static_cast<float>(0), colsAlign);

            event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventIdVToS);
            WaitFlag<HardEvent::V_S>(eventIdVToS);

            int64_t startSegId = -1;
            for (uint64_t i = 0; i < rows; i++) {
                int64_t segId = static_cast<int64_t>(sortedIdTensor(i));
                if (segId >= 0 && segId < static_cast<int64_t>(outputOuterDim_)) {
                    startSegId = segId;
                    break;
                }
            }

            if (startSegId < 0) {
                xQueue_.FreeTensor(xLocal);
                segmentIdQueue_.FreeTensor(segmentIdTensor);
                yQueue_.FreeTensor(yLocal);
                continue;
            }

            for (uint64_t i = 0; i < rows; i++) {
                int64_t curSegId = static_cast<int64_t>(sortedIdTensor(i));
                if (curSegId < 0 || curSegId >= static_cast<int64_t>(outputOuterDim_)) {
                    if (i == rows - 1) {
                        SetAtomicAdd<X_T>();
                        uint64_t yGmOffset = blockId_ * normalCoreProcessCols_ + aLoop * baseA_ +
                                             static_cast<uint64_t>(startSegId) * innerDim_;
                        LocalTensor<X_T> yLocalOutput = yLocal.template ReinterpretCast<X_T>();
                        if constexpr (std::negation<std::is_same<X_T, float>>::value) {
                            AscendC::Cast(yLocalOutput, yLocal, AscendC::RoundMode::CAST_RINT, cols);
                        }
                        yQueue_.EnQue<X_T>(yLocalOutput);
                        yLocalOutput = yQueue_.DeQue<X_T>();
                        CopyOut(yGm_, yLocalOutput, yGmOffset, 1, cols);
                        SetAtomicNone();
                    }
                    continue;
                }

                if (curSegId == startSegId) {
                    Add(yLocal, yLocal, xLocalFloat[sortedIdIndexTensor(i) * colsAlign], cols);
                } else {
                    SetAtomicAdd<X_T>();
                    uint64_t yGmOffset = blockId_ * normalCoreProcessCols_ + aLoop * baseA_ +
                                         static_cast<uint64_t>(startSegId) * innerDim_;
                    LocalTensor<X_T> yLocalOutput = yLocal.template ReinterpretCast<X_T>();
                    if constexpr (std::negation<std::is_same<X_T, float>>::value) {
                        AscendC::Cast(yLocalOutput, yLocal, AscendC::RoundMode::CAST_RINT, cols);
                    }
                    yQueue_.EnQue<X_T>(yLocalOutput);
                    yLocalOutput = yQueue_.DeQue<X_T>();
                    CopyOut(yGm_, yLocalOutput, yGmOffset, 1, cols);
                    SetAtomicNone();

                    event_t eventIdMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                    SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);
                    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);

                    startSegId = curSegId;
                    Duplicate(yLocal, static_cast<float>(0), colsAlign);
                    Add(yLocal, yLocal, xLocalFloat[sortedIdIndexTensor(i) * colsAlign], cols);
                }

                if (i == rows - 1) {
                    SetAtomicAdd<X_T>();
                    uint64_t yGmOffset = blockId_ * normalCoreProcessCols_ + aLoop * baseA_ +
                                         static_cast<uint64_t>(startSegId) * innerDim_;
                    LocalTensor<X_T> yLocalOutput = yLocal.template ReinterpretCast<X_T>();
                    if constexpr (std::negation<std::is_same<X_T, float>>::value) {
                        AscendC::Cast(yLocalOutput, yLocal, AscendC::RoundMode::CAST_RINT, cols);
                    }
                    yQueue_.EnQue<X_T>(yLocalOutput);
                    yLocalOutput = yQueue_.DeQue<X_T>();
                    CopyOut(yGm_, yLocalOutput, yGmOffset, 1, cols);
                    SetAtomicNone();
                    break;
                }
            }
            xQueue_.FreeTensor(xLocal);
            segmentIdQueue_.FreeTensor(segmentIdTensor);
            yQueue_.FreeTensor(yLocal);
        }
    }
}
} // namespace UnsortedSegmentSum
#endif