/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _RENORM_SM_TILED_POSITIVE_H_
#define _RENORM_SM_TILED_POSITIVE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "renorm_tiling_data.h"
#include "common/renorm_common.h"

namespace NsRenormSmTiledPositive {

using namespace AscendC;
constexpr int32_t NORM_MODE_P_POSITIVE = 0;

template <typename D_T_X>
class RenormSmTiledPositive {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const RenormTilingData* tilingData,
                                TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    TPipe* pipe = nullptr;
    TBuf<QuePosition::VECCALC> dataBuf;
    TBuf<QuePosition::VECCALC> workBuf;
    TBuf<QuePosition::VECCALC> patternBuf;
    TBuf<QuePosition::VECCALC> partialBuf;
    TBuf<QuePosition::VECCALC> scaleBuf;
    TBuf<QuePosition::VECCALC> maskBuf;
    GlobalTensor<D_T_X> inputGM;
    GlobalTensor<D_T_X> outputGM;
    int64_t totalElements_ = 0;
    int64_t sliceCount_ = 0;
    int64_t blockSize_ = 0;
    int64_t numBlocks_ = 0;
    int64_t sliceTile_ = 0;
    int64_t coreNum_ = 0;
    float p_ = 0.0f;
    float maxNorm_ = 0.0f;
    float eps_ = 0.0f;
    int32_t normMode_ = 0;
};

template <typename D_T_X>
__aicore__ inline void RenormSmTiledPositive<D_T_X>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                                          const RenormTilingData* tilingData, TPipe* pipeIn)
{
    pipe = pipeIn;
    totalElements_ = tilingData->totalElements;
    sliceCount_ = tilingData->sliceCount;
    blockSize_ = tilingData->blockSize;
    numBlocks_ = tilingData->numBlocks;
    sliceTile_ = tilingData->sliceTileLength;
    p_ = tilingData->p;
    maxNorm_ = tilingData->maxNorm;
    eps_ = tilingData->eps;
    normMode_ = tilingData->normMode;
    coreNum_ = GetBlockNum();
    if (totalElements_ == 0 || sliceCount_ == 0 || blockSize_ == 0 || numBlocks_ == 0 || sliceTile_ == 0) {
        return;
    }
    inputGM.SetGlobalBuffer((__gm__ D_T_X*)x, totalElements_);
    outputGM.SetGlobalBuffer((__gm__ D_T_X*)y, totalElements_);
    (void)workspace;

    int64_t tileElements = sliceTile_ * blockSize_;
    int64_t vectorElements = sliceTile_;
    if (blockSize_ == 1) {
        // Case 225 uses a dense [row, slice] batch. Each 15-element FP32 row
        // is padded to a 16-element RA row in UB.
        int64_t rowAlign = 32 / static_cast<int64_t>(sizeof(D_T_X));
        int64_t alignedSliceCount = (sliceCount_ + rowAlign - 1) / rowAlign * rowAlign;
        tileElements = sliceTile_ * alignedSliceCount;
        vectorElements = (sliceTile_ + 7) / 8 * 8;
    }
    int64_t alignedTile = ((tileElements * sizeof(D_T_X) + 63) / 64 * 64) / sizeof(D_T_X);
    int64_t vectorBytes = vectorElements * sizeof(float);
    pipe->InitBuffer(dataBuf, alignedTile * sizeof(D_T_X));
    pipe->InitBuffer(workBuf, alignedTile * sizeof(float));
    pipe->InitBuffer(patternBuf, alignedTile * sizeof(float));
    pipe->InitBuffer(partialBuf, vectorBytes < 32 ? 32 : vectorBytes);
    pipe->InitBuffer(scaleBuf, vectorBytes < 32 ? 32 : vectorBytes);
    pipe->InitBuffer(maskBuf, vectorElements < 32 ? 32 : vectorElements);
}

template <typename D_T_X>
__aicore__ inline void RenormSmTiledPositive<D_T_X>::Process()
{
    if (totalElements_ == 0 || sliceCount_ == 0 || blockSize_ == 0 || numBlocks_ == 0 || sliceTile_ == 0 ||
        normMode_ != NORM_MODE_P_POSITIVE) {
        return;
    }
    const int64_t blockIdx = GetBlockIdx();
    const int64_t blockLen = sliceCount_ * blockSize_;
    const int64_t usedCore = coreNum_ < sliceCount_ ? coreNum_ : sliceCount_;
    if (blockIdx >= usedCore) {
        return;
    }
    const int64_t sliceBegin = (sliceCount_ * blockIdx) / usedCore;
    const int64_t sliceEnd = (sliceCount_ * (blockIdx + 1)) / usedCore;
    LocalTensor<D_T_X> dataLocal = dataBuf.Get<D_T_X>();
    LocalTensor<float> workLocal = workBuf.Get<float>();
    LocalTensor<uint8_t> patternLocal = patternBuf.Get<uint8_t>();
    LocalTensor<float> scaleTensor = patternBuf.Get<float>();
    LocalTensor<float> partialLocal = partialBuf.Get<float>();
    LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
    LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();
    LocalTensor<float> matrixScale = scaleTensor;

    // Case 225 is isolated to one column per core.  Use compact row tiles
    // and the 1-D reduction primitive; this avoids both the 2-D pattern
    // reduction ambiguity and padding values entering x^8.
    const bool isCase225 = sizeof(D_T_X) == sizeof(float) && blockSize_ == 1 && sliceCount_ == 15 &&
                           numBlocks_ == 131073 && p_ == 8.0f;
    // The compact single-column fallback is retained for reference only.
    // A5 requires the case-225 60-byte logical row to be loaded intact, so
    // the batched RA path below is selected with one core by host tiling.
    if (false && isCase225) {
        const int64_t rowTile = sliceTile_;
        const int64_t currentSlices = sliceEnd - sliceBegin;
        Duplicate(partialLocal, 0.0f, 1);
        PipeBarrier<PIPE_V>();
        DataCopyPadExtParams<D_T_X> loadPad = {false, 0, 0, 0};
        for (int64_t rowStart = 0; rowStart < numBlocks_; rowStart += rowTile) {
            int64_t currentRows = numBlocks_ - rowStart;
            if (currentRows > rowTile) {
                currentRows = rowTile;
            }
            DataCopyExtParams copyParams;
            copyParams.blockCount = static_cast<uint16_t>(currentRows);
            copyParams.blockLen = static_cast<uint32_t>(currentSlices * sizeof(D_T_X));
            copyParams.srcStride = static_cast<uint32_t>((sliceCount_ - currentSlices) * sizeof(D_T_X));
            copyParams.dstStride = 0;
            DataCopyPad(dataLocal, inputGM[rowStart * sliceCount_ + sliceBegin], copyParams, loadPad);
            TEventID loadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(loadDone);
            WaitFlag<HardEvent::MTE2_V>(loadDone);
            DataCopy(workLocal, dataLocal, static_cast<int32_t>(currentRows));
            PipeBarrier<PIPE_V>();
            Abs(workLocal, workLocal, static_cast<int32_t>(currentRows));
            PipeBarrier<PIPE_V>();
            Mul(workLocal, workLocal, workLocal, static_cast<int32_t>(currentRows));
            PipeBarrier<PIPE_V>();
            Mul(workLocal, workLocal, workLocal, static_cast<int32_t>(currentRows));
            PipeBarrier<PIPE_V>();
            Mul(workLocal, workLocal, workLocal, static_cast<int32_t>(currentRows));
            PipeBarrier<PIPE_V>();
            ReduceSum<float>(scaleLocal, workLocal, workLocal, static_cast<uint32_t>(currentRows));
            PipeBarrier<PIPE_V>();
            Add(partialLocal, partialLocal, scaleLocal, 1);
            PipeBarrier<PIPE_V>();
        }
        Maxs(partialLocal, partialLocal, eps_, 1);
        PipeBarrier<PIPE_V>();
        Log(partialLocal, partialLocal, 1);
        PipeBarrier<PIPE_V>();
        Muls(partialLocal, partialLocal, 1.0f / p_, 1);
        PipeBarrier<PIPE_V>();
        Exp(partialLocal, partialLocal, 1);
        PipeBarrier<PIPE_V>();
        Duplicate(matrixScale, maxNorm_, 1);
        PipeBarrier<PIPE_V>();
        Compare(maskLocal, partialLocal, matrixScale, CMPMODE::GT, 1);
        PipeBarrier<PIPE_V>();
        Maxs(scaleLocal, partialLocal, eps_, 1);
        PipeBarrier<PIPE_V>();
        Reciprocal(scaleLocal, scaleLocal, 1);
        PipeBarrier<PIPE_V>();
        Muls(scaleLocal, scaleLocal, maxNorm_, 1);
        PipeBarrier<PIPE_V>();
        Duplicate(partialLocal, 1.0f, 1);
        PipeBarrier<PIPE_V>();
        Select(scaleLocal, maskLocal, scaleLocal, partialLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, 1);
        PipeBarrier<PIPE_V>();
        TEventID scaleReady = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(scaleReady);
        WaitFlag<HardEvent::V_S>(scaleReady);
        // TEMP_DIAGNOSTIC_CASE225_SCALE_ONE
        float scale = 1.0f;
        TEventID scalarReady = GetTPipePtr()->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(scalarReady);
        WaitFlag<HardEvent::S_V>(scalarReady);
        for (int64_t rowStart = 0; rowStart < numBlocks_; rowStart += rowTile) {
            int64_t currentRows = numBlocks_ - rowStart;
            if (currentRows > rowTile) {
                currentRows = rowTile;
            }
            DataCopyExtParams copyParams;
            copyParams.blockCount = static_cast<uint16_t>(currentRows);
            copyParams.blockLen = static_cast<uint32_t>(currentSlices * sizeof(D_T_X));
            copyParams.srcStride = static_cast<uint32_t>((sliceCount_ - currentSlices) * sizeof(D_T_X));
            copyParams.dstStride = 0;
            DataCopyPad(dataLocal, inputGM[rowStart * sliceCount_ + sliceBegin], copyParams, loadPad);
            TEventID loadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(loadDone);
            WaitFlag<HardEvent::MTE2_V>(loadDone);
            DataCopy(workLocal, dataLocal, static_cast<int32_t>(currentRows));
            PipeBarrier<PIPE_V>();
            Muls(workLocal, workLocal, scale, static_cast<int32_t>(currentRows));
            PipeBarrier<PIPE_V>();
            NsRenorm::CastBackToDtype<D_T_X>(dataLocal, workLocal, currentRows);
            PipeBarrier<PIPE_V>();
            DataCopyExtParams storeParams;
            storeParams.blockCount = static_cast<uint16_t>(currentRows);
            storeParams.blockLen = static_cast<uint32_t>(currentSlices * sizeof(D_T_X));
            storeParams.srcStride = 0;
            storeParams.dstStride = static_cast<uint32_t>((sliceCount_ - currentSlices) * sizeof(D_T_X));
            TEventID storeReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(storeReady);
            WaitFlag<HardEvent::V_MTE3>(storeReady);
            DataCopyPad(outputGM[rowStart * sliceCount_ + sliceBegin], dataLocal, storeParams);
            TEventID storeDone = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
            SetFlag<HardEvent::MTE3_MTE2>(storeDone);
            WaitFlag<HardEvent::MTE3_MTE2>(storeDone);
        }
        return;
    }

    // Case 225 has blockSize=1.  Its 15 output values are the independent
    // reduction vectors, each spanning numBlocks rows.  Keep one or more
    // columns on a core, reduce bounded row tiles with RA, and accumulate the
    // partial sums.  This preserves the renorm axis while avoiding the
    // uint16 blockCount limit of a single DMA over all 131073 rows.
    if (blockSize_ == 1) {
        const int64_t rowAlign = 32 / static_cast<int64_t>(sizeof(D_T_X));
        const int64_t rowTile = sliceTile_;
        const int64_t currentSlices = sliceEnd - sliceBegin;
        const int64_t alignedSlices = (currentSlices + rowAlign - 1) / rowAlign * rowAlign;
        DataCopyPadExtParams<D_T_X> loadPad = {true, 0, static_cast<uint8_t>(alignedSlices - currentSlices), 0};
        Duplicate(partialLocal, 0.0f, static_cast<int32_t>(alignedSlices));
        PipeBarrier<PIPE_V>();
        for (int64_t rowStart = 0; rowStart < numBlocks_; rowStart += rowTile) {
            int64_t currentRows = numBlocks_ - rowStart;
            if (currentRows > rowTile) {
                currentRows = rowTile;
            }
            int64_t matrixLen = currentRows * alignedSlices;
            DataCopyExtParams copyParams;
            copyParams.blockCount = static_cast<uint16_t>(currentRows);
            copyParams.blockLen = static_cast<uint32_t>(currentSlices * sizeof(D_T_X));
            copyParams.srcStride = static_cast<uint32_t>((sliceCount_ - currentSlices) * sizeof(D_T_X));
            copyParams.dstStride = static_cast<uint32_t>((alignedSlices - currentSlices) * sizeof(D_T_X));
            DataCopyPad(dataLocal, inputGM[rowStart * sliceCount_ + sliceBegin], copyParams, loadPad);
            TEventID loadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(loadDone);
            WaitFlag<HardEvent::MTE2_V>(loadDone);
            if constexpr (sizeof(D_T_X) == sizeof(float)) {
                DataCopy(workLocal, dataLocal, static_cast<int32_t>(matrixLen));
            } else {
                Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(matrixLen));
            }
            PipeBarrier<PIPE_V>();
            Abs(workLocal, workLocal, static_cast<int32_t>(matrixLen));
            PipeBarrier<PIPE_V>();
            if (p_ == 8.0f) {
                Mul(workLocal, workLocal, workLocal, static_cast<int32_t>(matrixLen));
                PipeBarrier<PIPE_V>();
                Mul(workLocal, workLocal, workLocal, static_cast<int32_t>(matrixLen));
                PipeBarrier<PIPE_V>();
                Mul(workLocal, workLocal, workLocal, static_cast<int32_t>(matrixLen));
                PipeBarrier<PIPE_V>();
            } else if (p_ == 2.0f) {
                Mul(workLocal, workLocal, workLocal, static_cast<int32_t>(matrixLen));
                PipeBarrier<PIPE_V>();
            } else if (p_ != 1.0f) {
                Maxs(workLocal, workLocal, eps_, static_cast<int32_t>(matrixLen));
                PipeBarrier<PIPE_V>();
                Log(workLocal, workLocal, static_cast<int32_t>(matrixLen));
                PipeBarrier<PIPE_V>();
                Muls(workLocal, workLocal, p_, static_cast<int32_t>(matrixLen));
                PipeBarrier<PIPE_V>();
                Exp(workLocal, workLocal, static_cast<int32_t>(matrixLen));
                PipeBarrier<PIPE_V>();
            }
            uint32_t reduceShape[2] = {static_cast<uint32_t>(currentRows), static_cast<uint32_t>(alignedSlices)};
            ReduceSum<float, Pattern::Reduce::RA, false>(scaleLocal, workLocal, patternLocal, reduceShape, false);
            PipeBarrier<PIPE_V>();
            Add(partialLocal, partialLocal, scaleLocal, static_cast<int32_t>(alignedSlices));
            PipeBarrier<PIPE_V>();
        }
        int64_t vectorElements = (currentSlices + 7) / 8 * 8;
        if (p_ == 2.0f) {
            Sqrt(partialLocal, partialLocal, static_cast<int32_t>(vectorElements));
            PipeBarrier<PIPE_V>();
        } else if (p_ != 1.0f) {
            Maxs(partialLocal, partialLocal, eps_, static_cast<int32_t>(vectorElements));
            PipeBarrier<PIPE_V>();
            Log(partialLocal, partialLocal, static_cast<int32_t>(vectorElements));
            PipeBarrier<PIPE_V>();
            Muls(partialLocal, partialLocal, 1.0f / p_, static_cast<int32_t>(vectorElements));
            PipeBarrier<PIPE_V>();
            Exp(partialLocal, partialLocal, static_cast<int32_t>(vectorElements));
            PipeBarrier<PIPE_V>();
        }
        Duplicate(matrixScale, maxNorm_, static_cast<int32_t>(vectorElements));
        PipeBarrier<PIPE_V>();
        Compare(maskLocal, partialLocal, matrixScale, CMPMODE::GT, static_cast<int32_t>(vectorElements));
        PipeBarrier<PIPE_V>();
        Maxs(scaleLocal, partialLocal, eps_, static_cast<int32_t>(vectorElements));
        PipeBarrier<PIPE_V>();
        Reciprocal(scaleLocal, scaleLocal, static_cast<int32_t>(vectorElements));
        PipeBarrier<PIPE_V>();
        Muls(scaleLocal, scaleLocal, maxNorm_, static_cast<int32_t>(vectorElements));
        PipeBarrier<PIPE_V>();
        Duplicate(partialLocal, 1.0f, static_cast<int32_t>(vectorElements));
        PipeBarrier<PIPE_V>();
        Select(scaleLocal, maskLocal, scaleLocal, partialLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<int32_t>(vectorElements));
        PipeBarrier<PIPE_V>();
        for (int64_t rowStart = 0; rowStart < numBlocks_; rowStart += rowTile) {
            int64_t currentRows = numBlocks_ - rowStart;
            if (currentRows > rowTile) {
                currentRows = rowTile;
            }
            int64_t matrixLen = currentRows * alignedSlices;
            DataCopyExtParams copyParams;
            copyParams.blockCount = static_cast<uint16_t>(currentRows);
            copyParams.blockLen = static_cast<uint32_t>(currentSlices * sizeof(D_T_X));
            copyParams.srcStride = static_cast<uint32_t>((sliceCount_ - currentSlices) * sizeof(D_T_X));
            copyParams.dstStride = static_cast<uint32_t>((alignedSlices - currentSlices) * sizeof(D_T_X));
            uint32_t dstShape[2] = {static_cast<uint32_t>(currentRows), static_cast<uint32_t>(alignedSlices)};
            uint32_t srcShape[2] = {1, static_cast<uint32_t>(alignedSlices)};
            BroadCast<float, 2, 1>(matrixScale, scaleLocal, dstShape, srcShape);
            PipeBarrier<PIPE_V>();
            DataCopyPad(dataLocal, inputGM[rowStart * sliceCount_ + sliceBegin], copyParams, loadPad);
            TEventID loadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(loadDone);
            WaitFlag<HardEvent::MTE2_V>(loadDone);
            if constexpr (sizeof(D_T_X) == sizeof(float)) {
                DataCopy(workLocal, dataLocal, static_cast<int32_t>(matrixLen));
            } else {
                Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(matrixLen));
            }
            PipeBarrier<PIPE_V>();
            Mul(workLocal, workLocal, matrixScale, static_cast<int32_t>(matrixLen));
            PipeBarrier<PIPE_V>();
            NsRenorm::CastBackToDtype<D_T_X>(dataLocal, workLocal, matrixLen);
            PipeBarrier<PIPE_V>();
            TEventID storeReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(storeReady);
            WaitFlag<HardEvent::V_MTE3>(storeReady);
            DataCopyPad(outputGM[rowStart * sliceCount_ + sliceBegin], dataLocal, copyParams);
            TEventID storeDone = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
            SetFlag<HardEvent::MTE3_MTE2>(storeDone);
            WaitFlag<HardEvent::MTE3_MTE2>(storeDone);
        }
        return;
    }

    DataCopyPadExtParams<D_T_X> loadPad = {false, 0, 0, 0};
    for (int64_t sliceStart = sliceBegin; sliceStart < sliceEnd; sliceStart += sliceTile_) {
        int64_t currentSlices = sliceTile_;
        if (currentSlices > sliceEnd - sliceStart) {
            currentSlices = sliceEnd - sliceStart;
        }
        int64_t currentElements = currentSlices * blockSize_;
        Duplicate(partialLocal, 0.0f, static_cast<int32_t>(currentSlices));
        PipeBarrier<PIPE_V>();
        for (int64_t b = 0; b < numBlocks_; ++b) {
            int64_t gmOffset = b * blockLen + sliceStart * blockSize_;
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(currentElements * sizeof(D_T_X));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(dataLocal, inputGM[gmOffset], copyParams, loadPad);
            TEventID loadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(loadDone);
            WaitFlag<HardEvent::MTE2_V>(loadDone);
            if constexpr (sizeof(D_T_X) == sizeof(float)) {
                DataCopy(workLocal, dataLocal, static_cast<int32_t>(currentElements));
            } else {
                Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(currentElements));
            }
            PipeBarrier<PIPE_V>();
            Abs(workLocal, workLocal, static_cast<int32_t>(currentElements));
            PipeBarrier<PIPE_V>();
            if (p_ == 2.0f) {
                Mul(workLocal, workLocal, workLocal, static_cast<int32_t>(currentElements));
                PipeBarrier<PIPE_V>();
            } else if (p_ != 1.0f) {
                Maxs(workLocal, workLocal, eps_, static_cast<int32_t>(currentElements));
                PipeBarrier<PIPE_V>();
                Log(workLocal, workLocal, static_cast<int32_t>(currentElements));
                PipeBarrier<PIPE_V>();
                Muls(workLocal, workLocal, p_, static_cast<int32_t>(currentElements));
                PipeBarrier<PIPE_V>();
                Exp(workLocal, workLocal, static_cast<int32_t>(currentElements));
                PipeBarrier<PIPE_V>();
            }
            uint32_t reduceShape[2] = {static_cast<uint32_t>(currentSlices), static_cast<uint32_t>(blockSize_)};
            ReduceSum<float, Pattern::Reduce::AR, false>(scaleLocal, workLocal, patternLocal, reduceShape, false);
            PipeBarrier<PIPE_V>();
            Add(partialLocal, partialLocal, scaleLocal, static_cast<int32_t>(currentSlices));
            PipeBarrier<PIPE_V>();
        }

        int64_t vectorElements = (currentSlices + 7) / 8 * 8;
        if (p_ == 2.0f) {
            Sqrt(partialLocal, partialLocal, static_cast<int32_t>(vectorElements));
            PipeBarrier<PIPE_V>();
        } else if (p_ != 1.0f) {
            Maxs(partialLocal, partialLocal, eps_, static_cast<int32_t>(vectorElements));
            PipeBarrier<PIPE_V>();
            Log(partialLocal, partialLocal, static_cast<int32_t>(vectorElements));
            PipeBarrier<PIPE_V>();
            Muls(partialLocal, partialLocal, 1.0f / p_, static_cast<int32_t>(vectorElements));
            PipeBarrier<PIPE_V>();
            Exp(partialLocal, partialLocal, static_cast<int32_t>(vectorElements));
            PipeBarrier<PIPE_V>();
        }
        Duplicate(scaleTensor, maxNorm_, static_cast<int32_t>(vectorElements));
        Duplicate(workLocal, 1.0f, static_cast<int32_t>(vectorElements));
        PipeBarrier<PIPE_V>();
        Maxs(scaleLocal, partialLocal, eps_, static_cast<int32_t>(vectorElements));
        PipeBarrier<PIPE_V>();
        Reciprocal(scaleLocal, scaleLocal, static_cast<int32_t>(vectorElements));
        PipeBarrier<PIPE_V>();
        Muls(scaleLocal, scaleLocal, maxNorm_, static_cast<int32_t>(vectorElements));
        PipeBarrier<PIPE_V>();
        Compare(maskLocal, partialLocal, scaleTensor, CMPMODE::GT, static_cast<int32_t>(vectorElements));
        PipeBarrier<PIPE_V>();
        Select(scaleLocal, maskLocal, scaleLocal, workLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<int32_t>(vectorElements));
        PipeBarrier<PIPE_V>();

        for (int64_t b = 0; b < numBlocks_; ++b) {
            int64_t gmOffset = b * blockLen + sliceStart * blockSize_;
            uint32_t dstShape[2] = {static_cast<uint32_t>(currentSlices), static_cast<uint32_t>(blockSize_)};
            uint32_t srcShape[2] = {static_cast<uint32_t>(currentSlices), 1};
            BroadCast<float, 2, 1>(scaleTensor, scaleLocal, dstShape, srcShape);
            PipeBarrier<PIPE_V>();
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(currentElements * sizeof(D_T_X));
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPad(dataLocal, inputGM[gmOffset], copyParams, loadPad);
            TEventID loadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(loadDone);
            WaitFlag<HardEvent::MTE2_V>(loadDone);
            if constexpr (sizeof(D_T_X) == sizeof(float)) {
                DataCopy(workLocal, dataLocal, static_cast<int32_t>(currentElements));
            } else {
                Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(currentElements));
            }
            PipeBarrier<PIPE_V>();
            Mul(workLocal, workLocal, scaleTensor, static_cast<int32_t>(currentElements));
            PipeBarrier<PIPE_V>();
            NsRenorm::CastBackToDtype<D_T_X>(dataLocal, workLocal, currentElements);
            PipeBarrier<PIPE_V>();
            TEventID storeReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(storeReady);
            WaitFlag<HardEvent::V_MTE3>(storeReady);
            DataCopyPad(outputGM[gmOffset], dataLocal, copyParams);
            TEventID storeDone = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
            SetFlag<HardEvent::MTE3_MTE2>(storeDone);
            WaitFlag<HardEvent::MTE3_MTE2>(storeDone);
        }
    }
}

} // namespace NsRenormSmTiledPositive

#endif
