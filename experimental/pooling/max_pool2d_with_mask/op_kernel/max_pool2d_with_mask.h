/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MAX_POOL2D_WITH_MASK_H
#define MAX_POOL2D_WITH_MASK_H

#include "kernel_operator.h"
#include "max_pool2d_with_mask_tiling_data.h"

#if !defined(__CCE_AICORE__) || __CCE_AICORE__ >= 220
#include "max_pool2d_with_mask_vector.h"
#endif

namespace MaxPool2dWithMask {

using namespace AscendC;

constexpr int32_t DOUBLE_BUFFER = 2;
constexpr uint64_t ALIGN_BYTES = 32;
constexpr uint64_t VECTOR_ZERO_BUFFER_BYTES = 64 * 1024;

#if !defined(__CCE_AICORE__) || __CCE_AICORE__ >= 220
class IndexTailClearKernel {
public:
    __aicore__ inline void Init(GM_ADDR indices, TPipe* pipe, const MaxPool2dWithMaskTilingData* tilingData)
    {
        pipe_ = pipe;
        tiling_ = tilingData;
        indexBytesGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(indices), tiling_->containerBytes);
        pipe_->InitBuffer(zeroQueue_, 1, VECTOR_ZERO_BUFFER_BYTES);
    }

    __aicore__ inline void Process()
    {
        const int64_t tailBytes = tiling_->containerBytes - tiling_->validIndexBytes;
        int64_t firstByte = GetBlockIdx() * tiling_->zeroBytesPerCore;
        if (firstByte >= tailBytes) {
            return;
        }
        int64_t lastByte = firstByte + tiling_->zeroBytesPerCore;
        if (lastByte > tailBytes) {
            lastByte = tailBytes;
        }

        LocalTensor<uint8_t> zeroLocal = zeroQueue_.AllocTensor<uint8_t>();
        const int64_t firstCopyBytes = lastByte - firstByte < VECTOR_ZERO_BUFFER_BYTES ? lastByte - firstByte :
                                                                                         VECTOR_ZERO_BUFFER_BYTES;
        const uint64_t alignedBytes = (static_cast<uint64_t>(firstCopyBytes) + ALIGN_BYTES - 1) / ALIGN_BYTES *
                                      ALIGN_BYTES;
        Duplicate(zeroLocal.ReinterpretCast<int32_t>(), 0, alignedBytes / sizeof(int32_t));
        zeroQueue_.EnQue(zeroLocal);
        zeroLocal = zeroQueue_.DeQue<uint8_t>();
        // Every DMA reads the same immutable zeros. Keep the buffer alive until
        // all copies have been issued; the caller orders reuse before pipe.Reset.
        while (firstByte < lastByte) {
            int64_t copyBytes = VECTOR_ZERO_BUFFER_BYTES;
            if (firstByte + copyBytes > lastByte) {
                copyBytes = lastByte - firstByte;
            }
            DataCopyExtParams copyParams{};
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(copyBytes);
            DataCopyPad(indexBytesGlobal_[tiling_->validIndexBytes + firstByte], zeroLocal, copyParams);
            firstByte += copyBytes;
        }
        zeroQueue_.FreeTensor(zeroLocal);
    }

private:
    TPipe* pipe_ = nullptr;
    const MaxPool2dWithMaskTilingData* tiling_ = nullptr;
    TQue<QuePosition::VECOUT, 1> zeroQueue_;
    GlobalTensor<uint8_t> indexBytesGlobal_;
};
#endif

template <typename T>
__aicore__ inline float ToFloat(T value)
{
#if !defined(__CCE_AICORE__) || __CCE_AICORE__ >= 220
    if constexpr (AscendC::IsSameType<T, bfloat16_t>::value) {
        return AscendC::ToFloat(value);
    } else {
        return static_cast<float>(value);
    }
#else
    return static_cast<float>(value);
#endif
}

template <typename T, bool Direct = false>
class Kernel {
public:
    __aicore__ inline Kernel() = default;
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR out, GM_ADDR indices, const MaxPool2dWithMaskTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline uint64_t AlignUp(uint64_t value, uint64_t alignment) const;
    __aicore__ inline void ProcessSlice(int64_t sliceIndex);
    __aicore__ inline void ComputeDirect(int64_t sliceIndex);
    __aicore__ inline void ComputeLegacy(int64_t sliceIndex, int64_t outputRowStart, int64_t outputRowCount);
    __aicore__ inline void CopyInput(int64_t sliceIndex, int64_t outputRowStart, int64_t outputRowCount);
    __aicore__ inline void Compute(int64_t sliceIndex, int64_t outputRowStart, int64_t outputRowCount);
    __aicore__ inline void CopyOutput(int64_t sliceIndex, int64_t outputRowStart, int64_t outputRowCount);
    __aicore__ inline void ClearIndexTail();
    __aicore__ inline void ComputeSeparable(const LocalTensor<uint16_t>& input, const LocalTensor<uint16_t>& output,
                                            const LocalTensor<int32_t>& indices, int64_t outputRowStart,
                                            int64_t outputRowCount, int64_t firstInputRow, int64_t inputPrefix);

    __aicore__ inline void ComputeVectorInterior(const LocalTensor<uint16_t>& input,
                                                 const LocalTensor<uint16_t>& output,
                                                 const LocalTensor<int32_t>& indices, int64_t outputRowStart,
                                                 int64_t outputRowCount, int64_t firstInputRow, int64_t inputPrefix);

    TPipe pipe_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> inputQueue_;
    TQue<QuePosition::VECOUT, 1> outputQueue_;
    TQue<QuePosition::VECOUT, 1> indexQueue_;
    TQue<QuePosition::VECOUT, 1> zeroQueue_;
    TBuf<QuePosition::VECCALC> columnBuffer_;
    TBuf<QuePosition::VECCALC> vectorBuffer_;

    GlobalTensor<T> inputGlobal_;
    GlobalTensor<uint16_t> inputHalfWordsGlobal_;
    GlobalTensor<T> outputGlobal_;
    GlobalTensor<uint16_t> outputHalfWordsGlobal_;
    GlobalTensor<int32_t> indexGlobal_;
    GlobalTensor<uint8_t> indexBytesGlobal_;

    const MaxPool2dWithMaskTilingData* tiling_ = nullptr;
    int64_t blockIndex_ = 0;
    int64_t firstSlice_ = 0;
    int64_t lastSlice_ = 0;
};

template <typename T, bool Direct>
__aicore__ inline uint64_t Kernel<T, Direct>::AlignUp(uint64_t value, uint64_t alignment) const
{
    return (value + alignment - 1) / alignment * alignment;
}

template <typename T, bool Direct>
__aicore__ inline void Kernel<T, Direct>::Init(GM_ADDR x, GM_ADDR out, GM_ADDR indices,
                                               const MaxPool2dWithMaskTilingData* tilingData)
{
    tiling_ = tilingData;
    blockIndex_ = GetBlockIdx();
    firstSlice_ = blockIndex_ * tiling_->slicesPerCore;
    lastSlice_ = firstSlice_ + tiling_->slicesPerCore;
    if (lastSlice_ > tiling_->totalSlices) {
        lastSlice_ = tiling_->totalSlices;
    }

    const int64_t inputElements = tiling_->totalSlices * tiling_->inputHeight * tiling_->inputWidth;
    const int64_t outputElements = tiling_->totalSlices * tiling_->outputHeight * tiling_->outputWidth;
    inputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), inputElements);
    outputGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(out), outputElements);
#if ORIG_DTYPE_X == DT_FLOAT
    inputHalfWordsGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(x), inputElements * 2);
    outputHalfWordsGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(out), outputElements * 2);
#endif
    indexGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(indices), tiling_->containerBytes / sizeof(int32_t));
    indexBytesGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(indices), tiling_->containerBytes);

    if constexpr (Direct) {
        inputHalfWordsGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(x), inputElements * sizeof(T) / 2);
        outputHalfWordsGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(out), outputElements * sizeof(T) / 2);
        pipe_.InitBuffer(zeroQueue_, 1, tiling_->zeroBufferBytes);
        return;
    }

    const int64_t inputRows = (tiling_->outputRowsPerTile - 1) * tiling_->strideH +
                              tiling_->dilationH * (tiling_->kernelH - 1) + 1;
    uint64_t inputBufferBytes = AlignUp(static_cast<uint64_t>(inputRows) * tiling_->inputWidthAligned * sizeof(T),
                                        ALIGN_BYTES);
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220 && ORIG_DTYPE_X == DT_FLOAT
    // CopyInput rounds the GM start down to a 32-byte boundary and keeps that prefix in UB.
    inputBufferBytes += ALIGN_BYTES;
#endif
    const uint64_t outputElementsPerTile = static_cast<uint64_t>(tiling_->outputRowsPerTile) * tiling_->outputWidth;
    const uint64_t outputBufferBytes = AlignUp(outputElementsPerTile * sizeof(T), ALIGN_BYTES);
    const uint64_t indexBufferBytes = AlignUp(outputElementsPerTile * sizeof(int32_t), ALIGN_BYTES);

    pipe_.InitBuffer(inputQueue_, DOUBLE_BUFFER, inputBufferBytes);
    pipe_.InitBuffer(outputQueue_, 1, outputBufferBytes);
    pipe_.InitBuffer(indexQueue_, 1, indexBufferBytes);
    pipe_.InitBuffer(zeroQueue_, 1, tiling_->zeroBufferBytes);
    if (tiling_->vectorBufferElements != 0) {
        pipe_.InitBuffer(vectorBuffer_, tiling_->vectorBufferElements * 6 * sizeof(uint32_t) + ALIGN_BYTES);
    }
    if (tiling_->columnBufferElements != 0) {
        pipe_.InitBuffer(columnBuffer_, tiling_->columnBufferElements * 2 * sizeof(uint32_t));
    }
}

template <typename T, bool Direct>
__aicore__ inline void Kernel<T, Direct>::Process()
{
    for (int64_t slice = firstSlice_; slice < lastSlice_; ++slice) {
        if constexpr (Direct) {
            ComputeDirect(slice);
        } else {
            ProcessSlice(slice);
        }
    }
    ClearIndexTail();
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
    // M200 scalar GM writes use the data cache; flush the final dirty cache lines before kernel exit.
    PipeBarrier<PIPE_ALL>();
    DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE>(indexGlobal_);
#endif
}

template <typename T, bool Direct>
__aicore__ inline void Kernel<T, Direct>::ProcessSlice(int64_t sliceIndex)
{
    for (int64_t row = 0; row < tiling_->outputHeight; row += tiling_->outputRowsPerTile) {
        int64_t rowCount = tiling_->outputRowsPerTile;
        if (row + rowCount > tiling_->outputHeight) {
            rowCount = tiling_->outputHeight - row;
        }
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
#if ORIG_DTYPE_X == DT_FLOAT
        CopyInput(sliceIndex, row, rowCount);
        Compute(sliceIndex, row, rowCount);
        CopyOutput(sliceIndex, row, rowCount);
#else
        ComputeLegacy(sliceIndex, row, rowCount);
#endif
#else
        CopyInput(sliceIndex, row, rowCount);
        Compute(sliceIndex, row, rowCount);
        CopyOutput(sliceIndex, row, rowCount);
#endif
    }
}

template <typename T, bool Direct>
__aicore__ inline void Kernel<T, Direct>::ComputeDirect(int64_t sliceIndex)
{
    // Compare IEEE representations as unsigned ordered keys. This preserves
    // subnormals, infinities and the original signed-zero bits on M200.
    constexpr int64_t words = sizeof(T) / sizeof(uint16_t);
    constexpr uint32_t sign = sizeof(T) == 4 ? 0x80000000U : 0x8000U;
    constexpr uint32_t bitsMask = sizeof(T) == 4 ? 0xFFFFFFFFU : 0xFFFFU;
    const int64_t inputBase = sliceIndex * tiling_->inputHeight * tiling_->inputWidth;
    const int64_t outputBase = sliceIndex * tiling_->outputHeight * tiling_->outputWidth;
    for (int64_t row = 0; row < tiling_->outputHeight; ++row) {
        for (int64_t column = 0; column < tiling_->outputWidth; ++column) {
            bool initialized = false;
            uint32_t maximumKey = 0;
            uint32_t maximumBits = 0;
            int32_t maximumIndex = 0;
            for (int64_t kh = 0; kh < tiling_->kernelH; ++kh) {
                const int64_t ih = row * tiling_->strideH - tiling_->padH + kh * tiling_->dilationH;
                if (ih < 0 || ih >= tiling_->inputHeight) {
                    continue;
                }
                for (int64_t kw = 0; kw < tiling_->kernelW; ++kw) {
                    const int64_t iw = column * tiling_->strideW - tiling_->padW + kw * tiling_->dilationW;
                    if (iw < 0 || iw >= tiling_->inputWidth) {
                        continue;
                    }
                    const int64_t index = ih * tiling_->inputWidth + iw;
                    const int64_t offset = (inputBase + index) * words;
                    uint32_t bits = inputHalfWordsGlobal_.GetValue(offset);
                    if constexpr (sizeof(T) == 4) {
                        bits |= static_cast<uint32_t>(inputHalfWordsGlobal_.GetValue(offset + 1)) << 16;
                    }
                    uint32_t key = (bits & sign) != 0 ? (~bits & bitsMask) : (bits ^ sign);
                    if ((bits & (sign - 1)) == 0) {
                        key = sign;
                    }
                    if (!initialized || key > maximumKey) {
                        initialized = true;
                        maximumKey = key;
                        maximumBits = bits;
                        maximumIndex = static_cast<int32_t>(index);
                    }
                }
            }
            const int64_t offset = outputBase + row * tiling_->outputWidth + column;
            outputHalfWordsGlobal_.SetValue(offset * words, static_cast<uint16_t>(maximumBits));
            if constexpr (sizeof(T) == 4) {
                outputHalfWordsGlobal_.SetValue(offset * words + 1, static_cast<uint16_t>(maximumBits >> 16));
            }
            indexGlobal_.SetValue(offset, maximumIndex);
        }
    }
}

template <typename T, bool Direct>
__aicore__ inline void Kernel<T, Direct>::ComputeLegacy(int64_t sliceIndex, int64_t outputRowStart,
                                                        int64_t outputRowCount)
{
    const int64_t inputSliceOffset = sliceIndex * tiling_->inputHeight * tiling_->inputWidth;
    const int64_t outputSliceOffset = sliceIndex * tiling_->outputHeight * tiling_->outputWidth;

    for (int64_t localRow = 0; localRow < outputRowCount; ++localRow) {
        const int64_t outputRow = outputRowStart + localRow;
        for (int64_t outputColumn = 0; outputColumn < tiling_->outputWidth; ++outputColumn) {
            const int64_t outputOffset = outputSliceOffset + outputRow * tiling_->outputWidth + outputColumn;
            bool initialized = false;
            T maximum{};
            float maximumFloat = 0.0f;
            int32_t maximumIndex = 0;

            for (int32_t kernelRow = 0; kernelRow < tiling_->kernelH; ++kernelRow) {
                const int64_t inputRow = outputRow * tiling_->strideH - tiling_->padH + kernelRow * tiling_->dilationH;
                if (inputRow < 0 || inputRow >= tiling_->inputHeight) {
                    continue;
                }
                for (int32_t kernelColumn = 0; kernelColumn < tiling_->kernelW; ++kernelColumn) {
                    const int64_t inputColumn = outputColumn * tiling_->strideW - tiling_->padW +
                                                kernelColumn * tiling_->dilationW;
                    if (inputColumn < 0 || inputColumn >= tiling_->inputWidth) {
                        continue;
                    }
                    const int32_t inputIndex = static_cast<int32_t>(inputRow * tiling_->inputWidth + inputColumn);
                    const T candidate = inputGlobal_.GetValue(inputSliceOffset + inputIndex);
                    const float candidateFloat = ToFloat(candidate);
                    if (!initialized || candidateFloat > maximumFloat) {
                        initialized = true;
                        maximum = candidate;
                        maximumFloat = candidateFloat;
                        maximumIndex = inputIndex;
                    }
                }
            }

            outputGlobal_.SetValue(outputOffset, maximum);
            indexGlobal_.SetValue(outputOffset, maximumIndex);
        }
    }
}

template <typename T, bool Direct>
__aicore__ inline void Kernel<T, Direct>::CopyInput(int64_t sliceIndex, int64_t outputRowStart, int64_t outputRowCount)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220 && ORIG_DTYPE_X == DT_FLOAT
    LocalTensor<uint16_t> inputLocal = inputQueue_.AllocTensor<uint16_t>();
#else
    LocalTensor<T> inputLocal = inputQueue_.AllocTensor<T>();
#endif
    const int64_t inputRowStart = outputRowStart * tiling_->strideH - tiling_->padH;
    const int64_t inputRows = (outputRowCount - 1) * tiling_->strideH + tiling_->dilationH * (tiling_->kernelH - 1) + 1;
    const int64_t sliceOffset = sliceIndex * tiling_->inputHeight * tiling_->inputWidth;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220 && ORIG_DTYPE_X == DT_FLOAT
    constexpr int64_t elementsPerBlock = ALIGN_BYTES / sizeof(T);
    const int64_t firstInputRow = inputRowStart < 0 ? 0 : inputRowStart;
    int64_t lastInputRow = inputRowStart + inputRows;
    if (lastInputRow > tiling_->inputHeight) {
        lastInputRow = tiling_->inputHeight;
    }
    const int64_t firstElement = sliceOffset + firstInputRow * tiling_->inputWidth;
    const int64_t lastElement = sliceOffset + lastInputRow * tiling_->inputWidth;
    const int64_t alignedFirstElement = firstElement / elementsPerBlock * elementsPerBlock;
    const int64_t alignedCopyEndElement = lastElement / elementsPerBlock * elementsPerBlock;
    if (alignedCopyEndElement > alignedFirstElement) {
        DataCopy(inputLocal, inputHalfWordsGlobal_[alignedFirstElement * 2],
                 static_cast<uint32_t>((alignedCopyEndElement - alignedFirstElement) * 2));
    }
    for (int64_t element = alignedCopyEndElement; element < lastElement; ++element) {
        const int64_t localHalfWord = (element - alignedFirstElement) * 2;
        inputLocal.SetValue(localHalfWord, inputHalfWordsGlobal_.GetValue(element * 2));
        inputLocal.SetValue(localHalfWord + 1, inputHalfWordsGlobal_.GetValue(element * 2 + 1));
    }
#else
    DataCopyParams copyParams{};
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(tiling_->inputWidth * sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadParams padParams{false, 0, 0, 0};

    for (int64_t row = 0; row < inputRows; ++row) {
        const int64_t inputRow = inputRowStart + row;
        if (inputRow < 0 || inputRow >= tiling_->inputHeight) {
            continue;
        }
        const int64_t localOffset = row * tiling_->inputWidthAligned;
        const int64_t globalOffset = sliceOffset + inputRow * tiling_->inputWidth;
        DataCopyPad(inputLocal[localOffset], inputGlobal_[globalOffset], copyParams, padParams);
    }
#endif
    inputQueue_.EnQue(inputLocal);
}

template <typename T, bool Direct>
__aicore__ inline void Kernel<T, Direct>::Compute(int64_t sliceIndex, int64_t outputRowStart, int64_t outputRowCount)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220 && ORIG_DTYPE_X == DT_FLOAT
    {
        LocalTensor<uint16_t> scalarInput = inputQueue_.DeQue<uint16_t>();
        LocalTensor<uint16_t> scalarOutput = outputQueue_.AllocTensor<uint16_t>();
        LocalTensor<int32_t> scalarIndices = indexQueue_.AllocTensor<int32_t>();
        const int64_t sliceOffset = sliceIndex * tiling_->inputHeight * tiling_->inputWidth;
        const int64_t firstInputRow = outputRowStart * tiling_->strideH - tiling_->padH < 0 ?
                                          0 :
                                          outputRowStart * tiling_->strideH - tiling_->padH;
        constexpr int64_t elementsPerBlock = ALIGN_BYTES / sizeof(T);
        const int64_t firstElement = sliceOffset + firstInputRow * tiling_->inputWidth;
        const int64_t alignedFirstElement = firstElement / elementsPerBlock * elementsPerBlock;
        const int64_t inputPrefixElements = firstElement - alignedFirstElement;
        event_t eventMte2ToScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventMte2ToScalar);
        WaitFlag<HardEvent::MTE2_S>(eventMte2ToScalar);

        if (tiling_->columnBufferElements != 0) {
            ComputeSeparable(scalarInput, scalarOutput, scalarIndices, outputRowStart, outputRowCount, firstInputRow,
                             inputPrefixElements);
            outputQueue_.EnQue(scalarOutput);
            indexQueue_.EnQue(scalarIndices);
            inputQueue_.FreeTensor(scalarInput);
            return;
        }

        int64_t interiorStart = 0;
        int64_t interiorEnd = 0;
        if (tiling_->vectorBufferElements != 0) {
            ComputeVectorInterior(scalarInput, scalarOutput, scalarIndices, outputRowStart, outputRowCount,
                                  firstInputRow, inputPrefixElements);
            interiorStart = (tiling_->padW + tiling_->strideW - 1) / tiling_->strideW;
            interiorEnd = (tiling_->inputWidth + tiling_->padW - tiling_->kernelW) / tiling_->strideW + 1;
            if (interiorEnd > tiling_->outputWidth) {
                interiorEnd = tiling_->outputWidth;
            }
        }
        for (int64_t localRow = 0; localRow < outputRowCount; ++localRow) {
            const int64_t outputRow = outputRowStart + localRow;
            for (int64_t outputColumn = 0; outputColumn < tiling_->outputWidth; ++outputColumn) {
                if (outputColumn >= interiorStart && outputColumn < interiorEnd) {
                    continue;
                }
                bool initialized = false;
                uint16_t maximumLow = 0;
                uint16_t maximumHigh = 0;
                uint32_t maximumKey = 0;
                int32_t maximumIndex = 0;

                for (int32_t kernelRow = 0; kernelRow < tiling_->kernelH; ++kernelRow) {
                    const int64_t inputRow = outputRow * tiling_->strideH - tiling_->padH +
                                             static_cast<int64_t>(kernelRow) * tiling_->dilationH;
                    if (inputRow < 0 || inputRow >= tiling_->inputHeight) {
                        continue;
                    }
                    for (int32_t kernelColumn = 0; kernelColumn < tiling_->kernelW; ++kernelColumn) {
                        const int64_t inputColumn = outputColumn * tiling_->strideW - tiling_->padW +
                                                    static_cast<int64_t>(kernelColumn) * tiling_->dilationW;
                        if (inputColumn < 0 || inputColumn >= tiling_->inputWidth) {
                            continue;
                        }
                        const int64_t localElement = inputPrefixElements +
                                                     (inputRow - firstInputRow) * tiling_->inputWidth + inputColumn;
                        const uint16_t candidateLow = scalarInput.GetValue(localElement * 2);
                        const uint16_t candidateHigh = scalarInput.GetValue(localElement * 2 + 1);
                        const uint32_t candidateBits = static_cast<uint32_t>(candidateLow) |
                                                       (static_cast<uint32_t>(candidateHigh) << 16);
                        uint32_t candidateKey = (candidateBits & 0x80000000U) != 0 ? ~candidateBits :
                                                                                     candidateBits ^ 0x80000000U;
                        if ((candidateBits & 0x7FFFFFFFU) == 0) {
                            candidateKey = 0x80000000U;
                        }
                        if (!initialized || candidateKey > maximumKey) {
                            initialized = true;
                            maximumLow = candidateLow;
                            maximumHigh = candidateHigh;
                            maximumKey = candidateKey;
                            maximumIndex = static_cast<int32_t>(inputRow * tiling_->inputWidth + inputColumn);
                        }
                    }
                }

                const int64_t localOutput = localRow * tiling_->outputWidth + outputColumn;
                scalarOutput.SetValue(localOutput * 2, maximumLow);
                scalarOutput.SetValue(localOutput * 2 + 1, maximumHigh);
                scalarIndices.SetValue(localOutput, maximumIndex);
            }
        }

        outputQueue_.EnQue(scalarOutput);
        indexQueue_.EnQue(scalarIndices);
        inputQueue_.FreeTensor(scalarInput);
        return;
    }

#else
    LocalTensor<T> inputLocal = inputQueue_.DeQue<T>();
    LocalTensor<T> outputLocal = outputQueue_.AllocTensor<T>();
    LocalTensor<int32_t> indexLocal = indexQueue_.AllocTensor<int32_t>();
    const int64_t inputRowStart = outputRowStart * tiling_->strideH - tiling_->padH;

    for (int64_t localRow = 0; localRow < outputRowCount; ++localRow) {
        const int64_t outputRow = outputRowStart + localRow;
        for (int64_t outputColumn = 0; outputColumn < tiling_->outputWidth; ++outputColumn) {
            bool initialized = false;
            T maximum{};
            float maximumFloat = 0.0f;
            int32_t maximumIndex = 0;

            for (int32_t kernelRow = 0; kernelRow < tiling_->kernelH; ++kernelRow) {
                const int64_t inputRow = outputRow * tiling_->strideH - tiling_->padH + kernelRow * tiling_->dilationH;
                if (inputRow < 0 || inputRow >= tiling_->inputHeight) {
                    continue;
                }
                for (int32_t kernelColumn = 0; kernelColumn < tiling_->kernelW; ++kernelColumn) {
                    const int64_t inputColumn = outputColumn * tiling_->strideW - tiling_->padW +
                                                kernelColumn * tiling_->dilationW;
                    if (inputColumn < 0 || inputColumn >= tiling_->inputWidth) {
                        continue;
                    }
                    const int64_t localInputRow = inputRow - inputRowStart;
                    const int64_t localOffset = localInputRow * tiling_->inputWidthAligned + inputColumn;
                    const T candidate = inputLocal.GetValue(localOffset);
                    const float candidateFloat = ToFloat(candidate);
                    if (!initialized || candidateFloat > maximumFloat) {
                        initialized = true;
                        maximum = candidate;
                        maximumFloat = candidateFloat;
                        maximumIndex = static_cast<int32_t>(inputRow * tiling_->inputWidth + inputColumn);
                    }
                }
            }

            const int64_t localOutputOffset = localRow * tiling_->outputWidth + outputColumn;
            outputLocal.SetValue(localOutputOffset, maximum);
            indexLocal.SetValue(localOutputOffset, maximumIndex);
        }
    }

    outputQueue_.EnQue(outputLocal);
    indexQueue_.EnQue(indexLocal);
    inputQueue_.FreeTensor(inputLocal);
#endif
}

template <typename T, bool Direct>
__aicore__ inline void Kernel<T, Direct>::ComputeVectorInterior(const LocalTensor<uint16_t>& input,
                                                                const LocalTensor<uint16_t>& output,
                                                                const LocalTensor<int32_t>& indices,
                                                                int64_t outputRowStart, int64_t outputRowCount,
                                                                int64_t firstInputRow, int64_t inputPrefix)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220 && ORIG_DTYPE_X == DT_FLOAT
    constexpr uint32_t lanes = 64;
    auto scratch = vectorBuffer_.Get<uint32_t>();
    auto maximum = scratch.template ReinterpretCast<float>();
    auto candidate = scratch[lanes].template ReinterpretCast<float>();
    auto positions = scratch[2 * lanes].template ReinterpretCast<int32_t>();
    auto candidatePositions = scratch[3 * lanes].template ReinterpretCast<int32_t>();
    auto positionTemplate = scratch[4 * lanes].template ReinterpretCast<int32_t>();
    auto byteOffsets = scratch[5 * lanes];
    auto mask = scratch[6 * lanes].template ReinterpretCast<uint8_t>();
    auto values = input.template ReinterpretCast<float>();
    for (uint32_t lane = 0; lane < lanes; ++lane) {
        positionTemplate.SetValue(lane, lane * tiling_->strideW);
        byteOffsets.SetValue(lane, lane * tiling_->strideW * sizeof(float));
    }
    const int64_t interiorStart = (tiling_->padW + tiling_->strideW - 1) / tiling_->strideW;
    int64_t interiorEnd = (tiling_->inputWidth + tiling_->padW - tiling_->kernelW) / tiling_->strideW + 1;
    if (interiorEnd > tiling_->outputWidth) {
        interiorEnd = tiling_->outputWidth;
    }
    for (int64_t row = 0; row < outputRowCount; ++row) {
        int64_t firstRow = (outputRowStart + row) * tiling_->strideH - tiling_->padH;
        int64_t lastRow = firstRow + tiling_->kernelH;
        if (firstRow < 0) {
            firstRow = 0;
        }
        if (lastRow > tiling_->inputHeight) {
            lastRow = tiling_->inputHeight;
        }
        for (int64_t column = interiorStart; column < interiorEnd; column += lanes) {
            const uint32_t count = interiorEnd - column < lanes ? interiorEnd - column : lanes;
            const int64_t firstColumn = column * tiling_->strideW - tiling_->padW;
            // The next vector iteration must not overwrite scratch still read by scalar stores.
            event_t scalarToVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(scalarToVector);
            WaitFlag<HardEvent::S_V>(scalarToVector);
            Duplicate(maximum.template ReinterpretCast<int32_t>(), 0, lanes);
            Duplicate(candidate.template ReinterpretCast<int32_t>(), 0, lanes);
            PipeBarrier<PIPE_V>();
            uint32_t offset = (inputPrefix + (firstRow - firstInputRow) * tiling_->inputWidth + firstColumn) *
                              sizeof(float);
            Gather(maximum, values, byteOffsets, offset, static_cast<uint64_t>(count), 1, 8);
            Adds(positions, positionTemplate, static_cast<int32_t>(firstRow * tiling_->inputWidth + firstColumn),
                 lanes);
            PipeBarrier<PIPE_V>();
            for (int64_t inputRow = firstRow; inputRow < lastRow; ++inputRow) {
                for (int64_t kw = 0; kw < tiling_->kernelW; ++kw) {
                    if (inputRow == firstRow && kw == 0) {
                        continue;
                    }
                    offset = (inputPrefix + (inputRow - firstInputRow) * tiling_->inputWidth + firstColumn + kw) *
                             sizeof(float);
                    Gather(candidate, values, byteOffsets, offset, static_cast<uint64_t>(count), 1, 8);
                    Adds(candidatePositions, positionTemplate,
                         static_cast<int32_t>(inputRow * tiling_->inputWidth + firstColumn + kw), lanes);
                    PipeBarrier<PIPE_V>();
                    // Compare writes a full 64-bit mask; Select only consumes the valid lanes.
                    Compare(mask, candidate, maximum, CMPMODE::GT, static_cast<uint64_t>(lanes), 1, {1, 1, 1, 8, 8, 8});
                    PipeBarrier<PIPE_V>();
                    Select(maximum, mask, candidate, maximum, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                           static_cast<uint64_t>(count), 1, {1, 1, 1, 8, 8, 8});
                    Select(positions.template ReinterpretCast<float>(), mask,
                           candidatePositions.template ReinterpretCast<float>(),
                           positions.template ReinterpretCast<float>(), SELMODE::VSEL_TENSOR_TENSOR_MODE,
                           static_cast<uint64_t>(count), 1, {1, 1, 1, 8, 8, 8});
                    PipeBarrier<PIPE_V>();
                }
            }
            event_t vectorToScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(vectorToScalar);
            WaitFlag<HardEvent::V_S>(vectorToScalar);
            auto halfWords = maximum.template ReinterpretCast<uint16_t>();
            for (uint32_t lane = 0; lane < count; ++lane) {
                const int64_t destination = row * tiling_->outputWidth + column + lane;
                output.SetValue(destination * 2, halfWords.GetValue(lane * 2));
                output.SetValue(destination * 2 + 1, halfWords.GetValue(lane * 2 + 1));
                indices.SetValue(destination, positions.GetValue(lane));
            }
        }
    }
#endif
}

template <typename T, bool Direct>
__aicore__ inline void Kernel<T, Direct>::ComputeSeparable(const LocalTensor<uint16_t>& input,
                                                           const LocalTensor<uint16_t>& output,
                                                           const LocalTensor<int32_t>& indices, int64_t outputRowStart,
                                                           int64_t outputRowCount, int64_t firstInputRow,
                                                           int64_t inputPrefix)
{
    LocalTensor<uint32_t> keys = columnBuffer_.Get<uint32_t>();
    LocalTensor<int32_t> positions = keys[tiling_->columnBufferElements].template ReinterpretCast<int32_t>();
    for (int64_t row = 0; row < outputRowCount; ++row) {
        const int64_t rowStart = (outputRowStart + row) * tiling_->strideH - tiling_->padH;
        const int64_t rowEnd = rowStart + tiling_->kernelH < tiling_->inputHeight ? rowStart + tiling_->kernelH :
                                                                                    tiling_->inputHeight;
        const int64_t validStart = rowStart < 0 ? 0 : rowStart;
        // Reduce each column once. Index -1 also represents a window containing only padding.
        for (int64_t column = 0; column < tiling_->inputWidth; ++column) {
            uint32_t maximumKey = 0;
            int32_t maximumIndex = -1;
            for (int64_t inputRow = validStart; inputRow < rowEnd; ++inputRow) {
                const int64_t element = inputPrefix + (inputRow - firstInputRow) * tiling_->inputWidth + column;
                const uint32_t bits = static_cast<uint32_t>(input.GetValue(element * 2)) |
                                      (static_cast<uint32_t>(input.GetValue(element * 2 + 1)) << 16);
                uint32_t key = (bits & 0x80000000U) != 0 ? ~bits : bits ^ 0x80000000U;
                if ((bits & 0x7FFFFFFFU) == 0) {
                    key = 0x80000000U;
                }
                if (maximumIndex < 0 || key > maximumKey) {
                    maximumKey = key;
                    maximumIndex = static_cast<int32_t>(inputRow * tiling_->inputWidth + column);
                }
            }
            keys.SetValue(column, maximumKey);
            positions.SetValue(column, maximumIndex);
        }
        for (int64_t column = 0; column < tiling_->outputWidth; ++column) {
            const int64_t columnStart = column * tiling_->strideW - tiling_->padW;
            const int64_t columnEnd = columnStart + tiling_->kernelW < tiling_->inputWidth ?
                                          columnStart + tiling_->kernelW :
                                          tiling_->inputWidth;
            uint32_t maximumKey = 0;
            int32_t maximumIndex = -1;
            for (int64_t inputColumn = columnStart < 0 ? 0 : columnStart; inputColumn < columnEnd; ++inputColumn) {
                const uint32_t key = keys.GetValue(inputColumn);
                const int32_t index = positions.GetValue(inputColumn);
                // Column traversal differs from row-major order: explicitly retain the first original index.
                if (index >= 0 &&
                    (maximumIndex < 0 || key > maximumKey || (key == maximumKey && index < maximumIndex))) {
                    maximumKey = key;
                    maximumIndex = index;
                }
            }
            const int64_t localOutput = row * tiling_->outputWidth + column;
            uint16_t low = 0;
            uint16_t high = 0;
            if (maximumIndex >= 0) {
                const int64_t element = inputPrefix + maximumIndex - firstInputRow * tiling_->inputWidth;
                low = input.GetValue(element * 2);
                high = input.GetValue(element * 2 + 1);
            }
            output.SetValue(localOutput * 2, low);
            output.SetValue(localOutput * 2 + 1, high);
            indices.SetValue(localOutput, maximumIndex < 0 ? 0 : maximumIndex);
        }
    }
}

template <typename T, bool Direct>
__aicore__ inline void Kernel<T, Direct>::CopyOutput(int64_t sliceIndex, int64_t outputRowStart, int64_t outputRowCount)
{
#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220 && ORIG_DTYPE_X == DT_FLOAT
    LocalTensor<uint16_t> outputLocal = outputQueue_.DeQue<uint16_t>();
#else
    LocalTensor<T> outputLocal = outputQueue_.DeQue<T>();
#endif
    LocalTensor<int32_t> indexLocal = indexQueue_.DeQue<int32_t>();
    const int64_t outputOffset = sliceIndex * tiling_->outputHeight * tiling_->outputWidth +
                                 outputRowStart * tiling_->outputWidth;
    const int64_t outputElements = outputRowCount * tiling_->outputWidth;

#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220 && ORIG_DTYPE_X == DT_FLOAT
    constexpr int64_t cacheLineElements = 64 / sizeof(float);
    constexpr int64_t blockElements = ALIGN_BYTES / sizeof(float);
    const bool alignedSource = outputOffset % blockElements == 0;
    const int64_t dmaFirst = alignedSource ? AlignUp(outputOffset, cacheLineElements) - outputOffset : outputElements;
    const int64_t dmaLast = alignedSource ?
                                (outputOffset + outputElements) / cacheLineElements * cacheLineElements - outputOffset :
                                outputElements;
    // DMA owns complete cache lines. Scalar boundary stores cannot overwrite these
    // lines when the M200 scalar data cache is cleaned at the end of the kernel.
    for (int64_t element = 0; element < outputElements; ++element) {
        if (element >= dmaFirst && element < dmaLast) {
            continue;
        }
        outputHalfWordsGlobal_.SetValue((outputOffset + element) * 2, outputLocal.GetValue(element * 2));
        outputHalfWordsGlobal_.SetValue((outputOffset + element) * 2 + 1, outputLocal.GetValue(element * 2 + 1));
        indexGlobal_.SetValue(outputOffset + element, indexLocal.GetValue(element));
    }
    if (dmaFirst < dmaLast) {
        event_t scalarToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(scalarToMte3);
        WaitFlag<HardEvent::S_MTE3>(scalarToMte3);
        DataCopy(outputHalfWordsGlobal_[(outputOffset + dmaFirst) * 2], outputLocal[dmaFirst * 2],
                 static_cast<uint32_t>((dmaLast - dmaFirst) * 2));
        DataCopy(indexGlobal_[outputOffset + dmaFirst], indexLocal[dmaFirst],
                 static_cast<uint32_t>(dmaLast - dmaFirst));
    }
#else
    DataCopyParams outputCopyParams{};
    outputCopyParams.blockCount = 1;
    outputCopyParams.blockLen = static_cast<uint32_t>(outputElements * sizeof(T));
    outputCopyParams.srcStride = 0;
    outputCopyParams.dstStride = 0;
    DataCopyPad(outputGlobal_[outputOffset], outputLocal, outputCopyParams);

    DataCopyParams indexCopyParams{};
    indexCopyParams.blockCount = 1;
    indexCopyParams.blockLen = static_cast<uint32_t>(outputElements * sizeof(int32_t));
    indexCopyParams.srcStride = 0;
    indexCopyParams.dstStride = 0;
    DataCopyPad(indexGlobal_[outputOffset], indexLocal, indexCopyParams);
#endif

    outputQueue_.FreeTensor(outputLocal);
    indexQueue_.FreeTensor(indexLocal);
}

template <typename T, bool Direct>
__aicore__ inline void Kernel<T, Direct>::ClearIndexTail()
{
    const int64_t tailBytes = tiling_->containerBytes - tiling_->validIndexBytes;
    int64_t firstByte = blockIndex_ * tiling_->zeroBytesPerCore;
    if (firstByte >= tailBytes) {
        return;
    }
    int64_t lastByte = firstByte + tiling_->zeroBytesPerCore;
    if (lastByte > tailBytes) {
        lastByte = tailBytes;
    }

#if defined(__CCE_AICORE__) && __CCE_AICORE__ < 220
    int64_t wordFirstByte = firstByte;
    while (wordFirstByte < lastByte &&
           (tiling_->validIndexBytes + wordFirstByte) % static_cast<int64_t>(sizeof(int32_t)) != 0) {
        indexBytesGlobal_.SetValue(tiling_->validIndexBytes + wordFirstByte, 0);
        ++wordFirstByte;
    }
    if (wordFirstByte == lastByte) {
        return;
    }

    const int64_t wordLastByte = lastByte -
                                 (tiling_->validIndexBytes + lastByte) % static_cast<int64_t>(sizeof(int32_t));
    // A scalar store can dirty its entire 64-byte cache line on M200. Restrict DMA to
    // complete cache lines so the final cache clean cannot overwrite DMA-written zeros.
    constexpr int64_t cacheLineBytes = 64;
    const int64_t absoluteStart = tiling_->validIndexBytes + wordFirstByte;
    const int64_t absoluteEnd = tiling_->validIndexBytes + wordLastByte;
    const int64_t dmaStart = static_cast<int64_t>(AlignUp(absoluteStart, cacheLineBytes));
    const int64_t dmaEnd = absoluteEnd / cacheLineBytes * cacheLineBytes;
    int64_t byte = absoluteStart;
    for (; byte < absoluteEnd && byte < dmaStart; byte += sizeof(int32_t)) {
        indexGlobal_.SetValue(byte / sizeof(int32_t), 0);
    }
    if (dmaStart < dmaEnd) {
        LocalTensor<int32_t> zeroLocal = zeroQueue_.AllocTensor<int32_t>();
        Duplicate(zeroLocal.ReinterpretCast<int16_t>(), static_cast<int16_t>(0),
                  tiling_->zeroBufferBytes / sizeof(int16_t));
        zeroQueue_.EnQue(zeroLocal);
        zeroLocal = zeroQueue_.DeQue<int32_t>();
        for (; byte < dmaEnd;) {
            const int64_t copyBytes = dmaEnd - byte < tiling_->zeroBufferBytes ? dmaEnd - byte :
                                                                                 tiling_->zeroBufferBytes;
            DataCopy(indexGlobal_[byte / sizeof(int32_t)], zeroLocal,
                     static_cast<uint32_t>(copyBytes / sizeof(int32_t)));
            byte += copyBytes;
        }
        zeroQueue_.FreeTensor(zeroLocal);
    }
    for (; byte < absoluteEnd; byte += sizeof(int32_t)) {
        indexGlobal_.SetValue(byte / sizeof(int32_t), 0);
    }
    for (int64_t byte = wordLastByte; byte < lastByte; ++byte) {
        indexBytesGlobal_.SetValue(tiling_->validIndexBytes + byte, 0);
    }
#else
    while (firstByte < lastByte) {
        int64_t copyBytes = tiling_->zeroBufferBytes;
        if (firstByte + copyBytes > lastByte) {
            copyBytes = lastByte - firstByte;
        }
        LocalTensor<uint8_t> zeroLocal = zeroQueue_.AllocTensor<uint8_t>();
        const uint64_t alignedBytes = AlignUp(static_cast<uint64_t>(copyBytes), ALIGN_BYTES);
        Duplicate(zeroLocal.ReinterpretCast<int32_t>(), 0, alignedBytes / sizeof(int32_t));
        zeroQueue_.EnQue(zeroLocal);
        zeroLocal = zeroQueue_.DeQue<uint8_t>();

        DataCopyParams copyParams{};
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(copyBytes);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(indexBytesGlobal_[tiling_->validIndexBytes + firstByte], zeroLocal, copyParams);
        zeroQueue_.FreeTensor(zeroLocal);
        firstByte += copyBytes;
    }
#endif
}

} // namespace MaxPool2dWithMask

#endif // MAX_POOL2D_WITH_MASK_H
