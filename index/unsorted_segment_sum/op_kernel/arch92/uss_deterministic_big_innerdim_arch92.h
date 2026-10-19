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

#ifndef ASCENDC_USS_DETERMINISTIC_BIG_INNERDIM_ARCH92_H_
#define ASCENDC_USS_DETERMINISTIC_BIG_INNERDIM_ARCH92_H_

#include "kernel_operator.h"
#include "unsorted_segment_base_arch92.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace UnsortedSegmentSum {
using namespace AscendC;

constexpr uint32_t UB_AGLIN = 32;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint64_t BIGINNERDIM_OFFSET_TYPESIZE = 8; // int64_t offset for ScatterDataCopy

template <typename IDS_T>
__simd_callee__ inline void ComputeScatterOffsetsBigInnerDim(__local_mem__ int64_t* offsetAddr,
                                                             __local_mem__ IDS_T* dstSegIdAddr, uint32_t arNum,
                                                             uint64_t outputOuterDim, uint64_t innerDim,
                                                             uint64_t ubOffset, uint64_t valueTypeBytes)
{
    constexpr uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(int64_t);
    uint32_t remaining = arNum;
    int64_t rowStrideBytes = static_cast<int64_t>(innerDim * valueTypeBytes);
    int64_t colOffsetBytes = static_cast<int64_t>(ubOffset * valueTypeBytes);
    int64_t maxIdx = static_cast<int64_t>(outputOuterDim);
    uint16_t loopCnt = static_cast<uint16_t>((arNum + vfLen - 1) / vfLen);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<IDS_T> invalidReg;
        MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> offsetReg;
        MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> tmpReg;
        MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> keyI64Reg;
        MicroAPI::RegTensor<int32_t> castReg32;
        MicroAPI::MaskReg pregLoop;
        MicroAPI::MaskReg cmpMask;
        MicroAPI::MaskReg pregLoopI64;
        static constexpr MicroAPI::CastTrait castTraitB322B64 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                                 MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

        MicroAPI::Duplicate(invalidReg, static_cast<IDS_T>(-1));

        for (uint16_t i = 0; i < loopCnt; i++) {
            uint32_t remainingForKey = remaining;
            uint32_t remainingForI64 = remaining;
            pregLoop = MicroAPI::UpdateMask<IDS_T>(remainingForKey);
            pregLoopI64 = MicroAPI::UpdateMask<int64_t, MicroAPI::RegTraitNumTwo>(remainingForI64);

            if constexpr (IsSameType<IDS_T, int32_t>::value) {
                MicroAPI::RegTensor<int32_t> keyReg;
                MicroAPI::DataCopy<int32_t, MicroAPI::LoadDist::DIST_NORM>(keyReg, dstSegIdAddr + i * vfLen);
                MicroAPI::Compares<int32_t, CMPMODE::GE>(cmpMask, keyReg, maxIdx, pregLoop);
                MicroAPI::Select<int32_t>(keyReg, invalidReg, keyReg, cmpMask);
                MicroAPI::Cast<int64_t, int32_t, castTraitB322B64>(keyI64Reg, keyReg, pregLoopI64);
            } else {
                MicroAPI::DataCopy<int64_t>(keyI64Reg, reinterpret_cast<__ubuf__ int64_t*>(dstSegIdAddr + i * vfLen));
                MicroAPI::Compares<int64_t, CMPMODE::GE>(cmpMask, keyI64Reg, maxIdx, pregLoopI64);
                MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> invalidI64Reg;
                MicroAPI::Duplicate<int64_t>(invalidI64Reg, static_cast<int64_t>(-1));
                MicroAPI::Select<int64_t>(keyI64Reg, invalidI64Reg, keyI64Reg, cmpMask);
            }
            MicroAPI::Muls(tmpReg, keyI64Reg, rowStrideBytes, pregLoopI64);
            MicroAPI::Adds(offsetReg, tmpReg, colOffsetBytes, pregLoopI64);
            MicroAPI::DataCopy(offsetAddr + i * vfLen, offsetReg, pregLoopI64);
            remaining = remaining > vfLen ? remaining - vfLen : 0;
        }
    }
}

template <typename X_T, typename IDS_T>
class USSKernelDeterministicBigInnerDim {
public:
    __aicore__ inline USSKernelDeterministicBigInnerDim(
        const UnsortedSegmentSumDeterministicBigInnerDimTilingData* tiling, TPipe* pipe)
        : td_(tiling), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output);
    __aicore__ inline void Process();
    __aicore__ inline void CopyInData(uint64_t blockOffset, uint64_t sLoop, uint64_t aLoop, uint64_t cols,
                                      uint64_t rows, uint64_t colsAlign, int64_t localOffset);
    __aicore__ inline void ProcessScatter(uint64_t blockOffset, uint64_t aLoop, uint64_t sLoop, uint64_t cols,
                                          uint64_t rows, uint64_t colsAlign, int64_t xLocalOffsetStart);
    __aicore__ inline void ProcessNonScatter(uint64_t blockOffset, uint64_t aLoop, uint64_t sLoop, uint64_t cols,
                                             uint64_t rows, uint64_t colsAlign, int64_t xLocalOffsetStart);
    __aicore__ inline void WriteBackRow(LocalTensor<float>& yLocal, int64_t startSegId, uint64_t aLoop, uint64_t cols);
    __aicore__ inline int64_t FindFirstValidSegment(LocalTensor<IDS_T>& sortedIdTensor, uint64_t rows);
    __aicore__ inline void ScatterWriteBack(LocalTensor<X_T>& srcLocal, LocalTensor<int64_t>& offsetLocal,
                                            uint32_t arNum, uint64_t cols, uint64_t colsAlign);

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
    TBuf<QuePosition::VECCALC> resBuf;
    TBuf<QuePosition::VECCALC> resCastBuf;
    TBuf<QuePosition::VECCALC> dstSegIdBuf;
    TBuf<QuePosition::VECCALC> offsetBuf;

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
    uint64_t useScatterDMA_{0};
};

template <typename X_T, typename IDS_T>
__aicore__ inline void USSKernelDeterministicBigInnerDim<X_T, IDS_T>::Init(GM_ADDR x, GM_ADDR segmentIds,
                                                                           GM_ADDR output)
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
    useScatterDMA_ = td_->useScatterDMA;

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

    uint32_t xQueueSize = Ops::Base::CeilAlign(static_cast<uint32_t>(baseS_ * baseA_ * sizeof(float)), UB_AGLIN);
    uint32_t segmentIdQueueSize = Ops::Base::CeilAlign(static_cast<uint32_t>(baseS_ * sizeof(IDS_T)), UB_AGLIN);
    uint32_t sortedIdIndexBufSize = Ops::Base::CeilAlign(static_cast<uint32_t>(baseS_ * sizeof(uint32_t)), UB_AGLIN);
    xQueSize_ = xQueueSize;
    pipe_->InitBuffer(xQueue_, 1, xQueueSize);
    pipe_->InitBuffer(segmentIdQueue_, 1, segmentIdQueueSize);

    if (useScatterDMA_ == 0) {
        pipe_->InitBuffer(yQueue_, DOUBLE_BUFFER, baseA_ * sizeof(float));
    }
    pipe_->InitBuffer(sharedTmpBuf, sortSharedBufSize_);
    pipe_->InitBuffer(sortedIdBuf, segmentIdQueueSize);
    pipe_->InitBuffer(sortedIdIndexBuf, sortedIdIndexBufSize);

    if (useScatterDMA_ != 0) {
        uint32_t resBufSize = Ops::Base::CeilAlign(static_cast<uint32_t>(baseS_ * baseA_ * sizeof(float)), UB_AGLIN);
        pipe_->InitBuffer(resBuf, resBufSize);

        if constexpr (std::negation<std::is_same<X_T, float>>::value) {
            uint32_t resCastBufSize = Ops::Base::CeilAlign(static_cast<uint32_t>(baseS_ * baseA_ * sizeof(X_T)),
                                                           UB_AGLIN);
            pipe_->InitBuffer(resCastBuf, resCastBufSize);
        }
        uint32_t dstSegIdBufSize = Ops::Base::CeilAlign(static_cast<uint32_t>(baseS_ * sizeof(IDS_T)), UB_AGLIN);
        pipe_->InitBuffer(dstSegIdBuf, dstSegIdBufSize);

        uint32_t offsetBufSize = Ops::Base::CeilAlign(static_cast<uint32_t>(baseS_ * BIGINNERDIM_OFFSET_TYPESIZE),
                                                      UB_AGLIN);
        pipe_->InitBuffer(offsetBuf, offsetBufSize);
    }
}

template <typename X_T, typename IDS_T>
__aicore__ inline void USSKernelDeterministicBigInnerDim<X_T, IDS_T>::CopyInData(uint64_t blockOffset, uint64_t sLoop,
                                                                                 uint64_t aLoop, uint64_t cols,
                                                                                 uint64_t rows, uint64_t colsAlign,
                                                                                 int64_t localOffset)
{
    // copy in x
    LocalTensor<X_T> xLocal = xQueue_.AllocTensor<X_T>();
    uint64_t inputOffset = blockOffset + sLoop * baseS_ * innerDim_ + aLoop * baseA_;
    DataCopyExtParams dataCopyParam{(uint16_t)rows, (uint32_t)(cols * sizeof(X_T)),
                                    (uint32_t)((innerDim_ - cols) * sizeof(X_T)),
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
__aicore__ inline int64_t USSKernelDeterministicBigInnerDim<X_T, IDS_T>::FindFirstValidSegment(
    LocalTensor<IDS_T>& sortedIdTensor, uint64_t rows)
{
    int64_t startSegId = -1;
    for (uint64_t i = 0; i < rows; i++) {
        int64_t segId = static_cast<int64_t>(sortedIdTensor(i));
        if (segId >= 0 && segId < static_cast<int64_t>(outputOuterDim_)) {
            startSegId = segId;
            break;
        }
    }
    return startSegId;
}

template <typename X_T, typename IDS_T>
__aicore__ inline void USSKernelDeterministicBigInnerDim<X_T, IDS_T>::ScatterWriteBack(
    LocalTensor<X_T>& srcLocal, LocalTensor<int64_t>& offsetLocal, uint32_t arNum, uint64_t cols, uint64_t colsAlign)
{
    SetAtomicAdd<X_T>();
    ScatterParams scatterParams;
    scatterParams.offsetNum = static_cast<uint16_t>(arNum);
    scatterParams.burstLen = cols * sizeof(X_T);
    scatterParams.srcStride = colsAlign * sizeof(X_T);
    ScatterDataCopy(yGm_, srcLocal, offsetLocal, scatterParams);
    SetAtomicNone();
}

template <typename X_T, typename IDS_T>
__aicore__ inline void USSKernelDeterministicBigInnerDim<X_T, IDS_T>::WriteBackRow(LocalTensor<float>& yLocal,
                                                                                   int64_t startSegId, uint64_t aLoop,
                                                                                   uint64_t cols)
{
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

template <typename X_T, typename IDS_T>
__aicore__ inline void USSKernelDeterministicBigInnerDim<X_T, IDS_T>::ProcessNonScatter(uint64_t blockOffset,
                                                                                        uint64_t aLoop, uint64_t sLoop,
                                                                                        uint64_t cols, uint64_t rows,
                                                                                        uint64_t colsAlign,
                                                                                        int64_t xLocalOffsetStart)
{
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

    int64_t startSegId = FindFirstValidSegment(sortedIdTensor, rows);

    if (startSegId < 0) {
        xQueue_.FreeTensor(xLocal);
        segmentIdQueue_.FreeTensor(segmentIdTensor);
        yQueue_.FreeTensor(yLocal);
        return;
    }

    for (uint64_t i = 0; i < rows; i++) {
        int64_t curSegId = static_cast<int64_t>(sortedIdTensor(i));
        if (curSegId < 0 || curSegId >= static_cast<int64_t>(outputOuterDim_)) {
            if (i == rows - 1) {
                WriteBackRow(yLocal, startSegId, aLoop, cols);
            }
            continue;
        }

        if (curSegId == startSegId) {
            Add(yLocal, yLocal, xLocalFloat[sortedIdIndexTensor(i) * colsAlign], cols);
        } else {
            WriteBackRow(yLocal, startSegId, aLoop, cols);

            event_t eventIdMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);

            startSegId = curSegId;
            Duplicate(yLocal, static_cast<float>(0), colsAlign);
            Add(yLocal, yLocal, xLocalFloat[sortedIdIndexTensor(i) * colsAlign], cols);
        }

        if (i == rows - 1) {
            WriteBackRow(yLocal, startSegId, aLoop, cols);
            break;
        }
    }
    xQueue_.FreeTensor(xLocal);
    segmentIdQueue_.FreeTensor(segmentIdTensor);
    yQueue_.FreeTensor(yLocal);
}

template <typename X_T, typename IDS_T>
__aicore__ inline void USSKernelDeterministicBigInnerDim<X_T, IDS_T>::ProcessScatter(uint64_t blockOffset,
                                                                                     uint64_t aLoop, uint64_t sLoop,
                                                                                     uint64_t cols, uint64_t rows,
                                                                                     uint64_t colsAlign,
                                                                                     int64_t xLocalOffsetStart)
{
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

    LocalTensor<float> resLocal = resBuf.Get<float>();
    LocalTensor<IDS_T> dstSegIdLocal = dstSegIdBuf.Get<IDS_T>();

    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);

    // Find first valid segment id.
    int64_t startSegId = FindFirstValidSegment(sortedIdTensor, rows);

    if (startSegId < 0) {
        xQueue_.FreeTensor(xLocal);
        segmentIdQueue_.FreeTensor(segmentIdTensor);
        return;
    }

    uint32_t segRowIdx = 0;
    dstSegIdLocal.SetValue(0, static_cast<IDS_T>(startSegId));
    Duplicate(resLocal, static_cast<float>(0), colsAlign);

    bool hasResult = false;
    for (uint64_t i = 0; i < rows; i++) {
        int64_t curSegId = static_cast<int64_t>(sortedIdTensor(i));
        if (curSegId < 0 || curSegId >= static_cast<int64_t>(outputOuterDim_)) {
            continue;
        }

        if (curSegId != startSegId) {
            segRowIdx++;
            startSegId = curSegId;
            dstSegIdLocal.SetValue(segRowIdx, static_cast<IDS_T>(startSegId));
            Duplicate(resLocal[segRowIdx * colsAlign], static_cast<float>(0), colsAlign);
        }
        Add(resLocal[segRowIdx * colsAlign], resLocal[segRowIdx * colsAlign],
            xLocalFloat[sortedIdIndexTensor(i) * colsAlign], cols);
        hasResult = true;
    }

    if (!hasResult) {
        xQueue_.FreeTensor(xLocal);
        segmentIdQueue_.FreeTensor(segmentIdTensor);
        return;
    }

    uint32_t arNum = segRowIdx + 1;

    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    LocalTensor<int64_t> offsetLocal = offsetBuf.Get<int64_t>();
    __local_mem__ IDS_T* dstSegIdAddr = (__local_mem__ IDS_T*)dstSegIdLocal.GetPhyAddr();
    __local_mem__ int64_t* offsetAddr = (__local_mem__ int64_t*)offsetLocal.GetPhyAddr();
    uint64_t ubOffset = blockId_ * normalCoreProcessCols_ + aLoop * baseA_;
    ComputeScatterOffsetsBigInnerDim<IDS_T>(offsetAddr, dstSegIdAddr, arNum, outputOuterDim_, innerDim_, ubOffset,
                                            sizeof(X_T));

    LocalTensor<X_T> srcLocal;
    if constexpr (std::is_same<X_T, float>::value) {
        srcLocal = resLocal.template ReinterpretCast<X_T>();
    } else {
        LocalTensor<X_T> resCastLocal = resCastBuf.Get<X_T>();
        AscendC::Cast(resCastLocal, resLocal, AscendC::RoundMode::CAST_RINT, arNum * colsAlign);
        srcLocal = resCastLocal;
    }

    event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

    ScatterWriteBack(srcLocal, offsetLocal, arNum, cols, colsAlign);

    xQueue_.FreeTensor(xLocal);
    segmentIdQueue_.FreeTensor(segmentIdTensor);
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
        uint64_t colsAlign = Ops::Base::CeilAlign(static_cast<uint64_t>(cols * sizeof(X_T)), ONE_BLOCK_SIZE) /
                             sizeof(X_T);
        for (uint64_t sLoop = 0; sLoop < sLoopNum; sLoop++) {
            uint64_t rows = (sLoop == sLoopNum - 1) ? (inputOuterDim_ - sLoop * baseS_) : baseS_;
            if (useScatterDMA_ != 0) {
                ProcessScatter(blockOffset, aLoop, sLoop, cols, rows, colsAlign, xLocalOffsetStart);
            } else {
                ProcessNonScatter(blockOffset, aLoop, sLoop, cols, rows, colsAlign, xLocalOffsetStart);
            }
        }
    }
}
} // namespace UnsortedSegmentSum
#endif
