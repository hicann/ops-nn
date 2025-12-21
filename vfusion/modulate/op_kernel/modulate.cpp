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
 * \file modulate.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace NormModulate {
using namespace AscendC;

constexpr int64_t MIN_DLENGTH = 2048 * 4;
constexpr int64_t ALIGNED_SIZE = 32;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t HALF_UBLENGTH = 2;
constexpr int64_t SCALE_AND_SHIFT = 0;
constexpr int64_t NO_SCALE = 1;
constexpr int64_t NO_SHIFT = 2;

template <typename T>
class ModulateBase {
protected:
    __aicore__ inline ModulateBase(TPipe* pipe, const ModulateTilingData* tiling)
    {
        this->pipe = pipe;
        this->inputB = tiling->inputB;
        this->inputL = tiling->inputL;
        this->inputD = tiling->inputD;
        this->ubLength = tiling->ubLength;
        this->frontNum = tiling->frontNum;
        this->frontLength = tiling->frontLength;
        this->tailNum = tiling->tailNum;
        this->tailLength = tiling->tailLength;
        this->useDTiling = tiling->useDTiling;
        this->parameterStatus = tiling->parameterStatus;

        this->pipe->InitBuffer(this->QueueX, DOUBLE_BUFFER, this->ubLength * sizeof(T));
        this->pipe->InitBuffer(this->QueueY, DOUBLE_BUFFER, this->ubLength * sizeof(T));
        if(this->parameterStatus != NO_SCALE) {
            this->pipe->InitBuffer(this->QueueScale, 1, this->ubLength * sizeof(T));
        }
        if(this->parameterStatus != NO_SHIFT) {
            this->pipe->InitBuffer(this->QueueShift, 1, this->ubLength * sizeof(T));
        }
        this->ubLength = std::is_same<T, bfloat16_t>::value ? this->ubLength / HALF_UBLENGTH : this->ubLength;
    }

protected:
    __aicore__ inline int64_t min(int64_t a, int64_t b)
    {
        return a < b ? a : b;
    }

    __aicore__ inline int64_t GetNum()
    {
        return GetBlockIdx() >= this->frontNum ? this->tailLength : this->frontLength;
    }

    __aicore__ inline int64_t GetBatchEndId()
    {
        int64_t batchEndId = (this->frontLength * (GetBlockIdx() + 1) - 1) / this->inputL;
        if (GetBlockIdx() >= this->frontNum) {
            batchEndId =
                (this->frontLength * this->frontNum + this->tailLength * (GetBlockIdx() + 1 - this->frontNum) - 1) /
                this->inputL;
        }
        return batchEndId;
    }

    __aicore__ inline int64_t GetSecondFirstL()
    {
        int64_t secondFirstL = (this->batchStartId + 1) * this->inputL - this->frontLength * GetBlockIdx();
        if (GetBlockIdx() >= this->frontNum) {
            secondFirstL = (this->batchStartId + 1) * this->inputL -
                           (this->frontLength * this->frontNum + this->tailLength * (GetBlockIdx() - this->frontNum));
        }
        return secondFirstL;
    }

    __aicore__ inline int64_t CalcOffsetForScaleShift(
        int64_t BId, int64_t DId, int64_t DLength, int64_t BlockId, int64_t BlockLength)
    {
        int64_t offset = BId * this->inputD + DId * DLength + BlockId * BlockLength;
        if (this->useDTiling) {
            if (DLength == this->frontLength) {
                offset = BId * this->inputD + DId * this->frontLength + BlockId * BlockLength;
            } else {
                offset = BId * this->inputD + this->frontLength * this->frontNum +
                         this->tailLength * (DId - this->frontNum) + BlockId * BlockLength;
            }
        }
        return offset;
    }

    __aicore__ inline int64_t CalcOffset(
        int64_t BId, int64_t LId, int64_t DId, int64_t DLength, int64_t BlockId, int64_t BlockLength)
    {
        int64_t offset = BId * this->inputL * this->inputD + LId * this->inputD + DId * DLength + BlockId * BlockLength;
        if (this->useDTiling) {
            if (DLength == this->frontLength) {
                offset = BId * this->inputL * this->inputD + LId * this->inputD + DId * this->frontLength +
                         BlockId * BlockLength;
            } else {
                offset = BId * this->inputL * this->inputD + LId * this->inputD + this->frontLength * this->frontNum +
                         this->tailLength * (DId - this->frontNum) + BlockId * BlockLength;
            }
        }
        return offset;
    }

    __aicore__ inline void CopyCommom(int64_t offset, int64_t opCopyLength, LocalTensor<T> &localData, GlobalTensor<T> &gmData)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(opCopyLength * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            DataCopyPad(localData[this->ubLength], gmData[offset], copyParams, padParams);
        } else {
            DataCopyPad(localData, gmData[offset], copyParams, padParams);
        }
    }

    __aicore__ inline void CopyScaleIn(
        int64_t BId, int64_t DId, int64_t BlockId, int64_t opCopyLength, int64_t DLength, int64_t BlockLength)
    {
        LocalTensor<T> scaleLocal = QueueScale.template AllocTensor<T>();
        int64_t offset = CalcOffsetForScaleShift(BId, DId, DLength, BlockId, BlockLength);
        CopyCommom(offset, opCopyLength, scaleLocal, scaleGm);
        QueueScale.template EnQue(scaleLocal);
    }

    __aicore__ inline void CopyShiftIn(
        int64_t BId, int64_t DId, int64_t BlockId, int64_t opCopyLength, int64_t DLength, int64_t BlockLength)
    {
        LocalTensor<T> shiftLocal = QueueShift.template AllocTensor<T>();
        int64_t offset = CalcOffsetForScaleShift(BId, DId, DLength, BlockId, BlockLength);
        CopyCommom(offset, opCopyLength, shiftLocal, shiftGm);
        QueueShift.template EnQue(shiftLocal);
    }

    __aicore__ inline void CopyIn(int64_t offset, int64_t opCopyLength, int64_t handleL)
    {
        LocalTensor<T> xLocal = QueueX.template AllocTensor<T>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(handleL), static_cast<uint32_t>(opCopyLength * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            DataCopyPad(xLocal[this->ubLength], this->xGm[offset], copyParams, padParams);
        } else {
            DataCopyPad(xLocal, this->xGm[offset], copyParams, padParams);
        }
        QueueX.template EnQue(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset, int64_t opCopyLength, int64_t handleL)
    {
        LocalTensor<T> yLocal = QueueY.template DeQue<T>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(handleL), static_cast<uint32_t>(opCopyLength * sizeof(T)), 0, 0, 0};
        DataCopyPad(this->yGm[offset], yLocal, copyParams);
        QueueY.template FreeTensor(yLocal);
    }

    __aicore__ inline void PreProcess(
        LocalTensor<T>& scaleLocal, LocalTensor<T>& shiftLocal, LocalTensor<float>& scaleLocalFp32,
        LocalTensor<float>& shiftLocalFp32, int64_t opCopyLength)
    { 
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            Cast(scaleLocalFp32, scaleLocal[this->ubLength], RoundMode::CAST_NONE, opCopyLength);
            PipeBarrier<PIPE_V>();
            float oneValue = 1;
            Adds(scaleLocalFp32, scaleLocalFp32, oneValue, opCopyLength);
            PipeBarrier<PIPE_V>();
            Cast(shiftLocalFp32, shiftLocal[this->ubLength], RoundMode::CAST_NONE, opCopyLength);
            PipeBarrier<PIPE_V>();
        }else {
            T oneValue = 1;
            Adds(scaleLocal, scaleLocal, oneValue, opCopyLength);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void Compute(
        int64_t BId, int64_t DId, int64_t BlockId, int64_t opCopyLength, int64_t DLength, int64_t BlockLength,
        int64_t loopLStart, int64_t loopLEnd)
    {
        int64_t alignedLength = ALIGNED_SIZE / sizeof(T);
        this->alignedD = (opCopyLength + alignedLength - 1) & ~(alignedLength - 1);
        bool isDSmall = (!this->useDTiling) && (DId == 0) && (opCopyLength * sizeof(T) < MIN_DLENGTH); // 判断一行的个数是否太小，太小则一次搬入多行
        int64_t handleL = isDSmall ? min(loopLEnd - loopLStart, this->ubLength / this->alignedD) : 1; // 一次处理的行数
        LocalTensor<T> scaleLocal = this->QueueScale.template DeQue<T>();
        LocalTensor<T> shiftLocal = this->QueueShift.template DeQue<T>();
        LocalTensor<float> scaleLocalFp32 = scaleLocal.template ReinterpretCast<float>();
        LocalTensor<float> shiftLocalFp32 = shiftLocal.template ReinterpretCast<float>();
        PreProcess(scaleLocal, shiftLocal, scaleLocalFp32, shiftLocalFp32, opCopyLength);
        for (int64_t curL = loopLStart; curL < loopLEnd; ++curL) {
            handleL = (curL + handleL > loopLEnd) ? loopLEnd - curL : handleL;
            int64_t offset = CalcOffset(BId, curL, DId, DLength, BlockId, BlockLength);
            CopyIn(offset, opCopyLength, handleL);
            LocalTensor<T> xLocal = this->QueueX.template DeQue<T>();
            LocalTensor<T> yLocal = this->QueueY.template AllocTensor<T>();
            if constexpr (std::is_same<T, bfloat16_t>::value) {
                auto xLocalFp32 = xLocal.template ReinterpretCast<float>();
                auto yLocalFp32 = yLocal.template ReinterpretCast<float>();
                Cast(xLocalFp32, xLocal[this->ubLength], RoundMode::CAST_NONE, handleL * this->alignedD);
                PipeBarrier<PIPE_V>();
                for (int64_t jL = 0; jL < handleL; ++jL) {
                    Mul(xLocalFp32[jL * this->alignedD], xLocalFp32[jL * this->alignedD], scaleLocalFp32, opCopyLength);
                    PipeBarrier<PIPE_V>();
                    Add(yLocalFp32[jL * this->alignedD], xLocalFp32[jL * this->alignedD], shiftLocalFp32, opCopyLength);
                }
                PipeBarrier<PIPE_V>();
                Cast(yLocal, yLocalFp32, RoundMode::CAST_RINT, handleL * this->alignedD);
            } else {
                for (int64_t jL = 0; jL < handleL; ++jL) {
                    Mul(xLocal[jL * this->alignedD], xLocal[jL * this->alignedD], scaleLocal, opCopyLength);
                    PipeBarrier<PIPE_V>();
                    Add(yLocal[jL * this->alignedD], xLocal[jL * this->alignedD], shiftLocal, opCopyLength);
                }
            }
            this->QueueX.template FreeTensor(xLocal);
            this->QueueY.template EnQue(yLocal);
            CopyOut(offset, opCopyLength, handleL);
            curL += handleL - 1;
        }
        this->QueueScale.template FreeTensor(scaleLocal);
        this->QueueShift.template FreeTensor(shiftLocal);
    }

    __aicore__ inline void ComputeWithoutScale(
        int64_t BId, int64_t DId, int64_t BlockId, int64_t opCopyLength, int64_t DLength, int64_t BlockLength,
        int64_t loopLStart, int64_t loopLEnd)
    {
        int64_t alignedLength = ALIGNED_SIZE / sizeof(T);
        this->alignedD = (opCopyLength + alignedLength - 1) & ~(alignedLength - 1);
        bool isDSmall = (!this->useDTiling) && (DId == 0) && (opCopyLength * sizeof(T) < MIN_DLENGTH); // 判断一行的个数是否太小，太小则一次搬入多行
        int64_t handleL = isDSmall ? min(loopLEnd - loopLStart, this->ubLength / this->alignedD) : 1; // 一次处理的行数
        LocalTensor<T> shiftLocal = this->QueueShift.template DeQue<T>();
        LocalTensor<float> shiftLocalFp32 = shiftLocal.template ReinterpretCast<float>();
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            Cast(shiftLocalFp32, shiftLocal[this->ubLength], RoundMode::CAST_NONE, opCopyLength);
            PipeBarrier<PIPE_V>();
        }
        for (int64_t currL = loopLStart; currL < loopLEnd; ++currL) {
            handleL = (currL + handleL > loopLEnd) ? loopLEnd - currL : handleL;
            int64_t offset = CalcOffset(BId, currL, DId, DLength, BlockId, BlockLength);
            CopyIn(offset, opCopyLength, handleL);
            LocalTensor<T> xLocal = this->QueueX.template DeQue<T>();
            LocalTensor<T> yLocal = this->QueueY.template AllocTensor<T>();
            if constexpr (std::is_same<T, bfloat16_t>::value) {
                auto xLocalFp32 = xLocal.template ReinterpretCast<float>();
                auto yLocalFp32 = yLocal.template ReinterpretCast<float>();
                Cast(xLocalFp32, xLocal[this->ubLength], RoundMode::CAST_NONE, handleL * this->alignedD);
                PipeBarrier<PIPE_V>();
                for (int64_t kL = 0; kL < handleL; ++kL) {
                    Add(yLocalFp32[kL * this->alignedD], xLocalFp32[kL * this->alignedD], shiftLocalFp32, opCopyLength);
                }
                PipeBarrier<PIPE_V>();
                Cast(yLocal, yLocalFp32, RoundMode::CAST_RINT, handleL * this->alignedD);
            } else {
                for (int64_t kL = 0; kL < handleL; ++kL) {
                    Add(yLocal[kL * this->alignedD], xLocal[kL * this->alignedD], shiftLocal, opCopyLength);
                }
            }
            this->QueueX.template FreeTensor(xLocal);
            this->QueueY.template EnQue(yLocal);
            CopyOut(offset, opCopyLength, handleL);
            currL += handleL - 1;
        }
        this->QueueShift.template FreeTensor(shiftLocal);
    }

    __aicore__ inline void ComputeWithoutShift(
        int64_t BId, int64_t DId, int64_t BlockId, int64_t opCopyLength, int64_t DLength, int64_t BlockLength,
        int64_t loopLStart, int64_t loopLEnd)
    {
        int64_t alignedLength = ALIGNED_SIZE / sizeof(T);
        this->alignedD = (opCopyLength + alignedLength - 1) & ~(alignedLength - 1);
        bool isDSmall = (!this->useDTiling) && (DId == 0) && (opCopyLength * sizeof(T) < MIN_DLENGTH); // 判断一行的个数是否太小，太小则一次搬入多行
        int64_t handleL = isDSmall ? min(loopLEnd - loopLStart, this->ubLength / this->alignedD) : 1; // 一次处理的行数
        LocalTensor<T> scaleLocal = this->QueueScale.template DeQue<T>();
        LocalTensor<float> scaleLocalFp32 = scaleLocal.template ReinterpretCast<float>();
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            Cast(scaleLocalFp32, scaleLocal[this->ubLength], RoundMode::CAST_NONE, opCopyLength);
            PipeBarrier<PIPE_V>();
            float oneValue = 1;
            Adds(scaleLocalFp32, scaleLocalFp32, oneValue, opCopyLength);
            PipeBarrier<PIPE_V>();
        }else {
            T oneValue = 1;
            Adds(scaleLocal, scaleLocal, oneValue, opCopyLength);
            PipeBarrier<PIPE_V>();
        }
        for (int64_t i = loopLStart; i < loopLEnd; ++i) {
            handleL = (i + handleL > loopLEnd) ? loopLEnd - i : handleL;
            int64_t offset = CalcOffset(BId, i, DId, DLength, BlockId, BlockLength);
            CopyIn(offset, opCopyLength, handleL);
            LocalTensor<T> xLocal = this->QueueX.template DeQue<T>();
            LocalTensor<T> yLocal = this->QueueY.template AllocTensor<T>();
            if constexpr (std::is_same<T, bfloat16_t>::value) {
                auto xLocalFp32 = xLocal.template ReinterpretCast<float>();
                auto yLocalFp32 = yLocal.template ReinterpretCast<float>();
                Cast(xLocalFp32, xLocal[this->ubLength], RoundMode::CAST_NONE, handleL * this->alignedD);
                PipeBarrier<PIPE_V>();
                for (int64_t lL = 0; lL < handleL; ++lL) {
                    Mul(yLocalFp32[lL * this->alignedD], xLocalFp32[lL * this->alignedD], scaleLocalFp32, opCopyLength);
                }
                PipeBarrier<PIPE_V>();
                Cast(yLocal, yLocalFp32, RoundMode::CAST_RINT, handleL * this->alignedD);
            } else {
                for (int64_t lL = 0; lL < handleL; ++lL) {
                    Mul(yLocal[lL * this->alignedD], xLocal[lL * this->alignedD], scaleLocal, opCopyLength);
                }
            }
            this->QueueX.template FreeTensor(xLocal);
            this->QueueY.template EnQue(yLocal);
            CopyOut(offset, opCopyLength, handleL);
            i += handleL - 1;
        }
        this->QueueScale.template FreeTensor(scaleLocal);
    }

    __aicore__ inline void CopyAndCompute(
        int64_t BId, int64_t DId, int64_t BlockId, int64_t opCopyLength, int64_t DLength, int64_t BlockLength,
        int64_t loopLStart, int64_t loopLEnd, bool isTilingL)
    {
        switch (this->parameterStatus) {
            case SCALE_AND_SHIFT:
                CopyScaleIn(BId, DId, BlockId, opCopyLength, DLength, BlockLength);
                CopyShiftIn(BId, DId, BlockId, opCopyLength, DLength, BlockLength);
                Compute(isTilingL ? 0 : BId, DId, BlockId, opCopyLength, DLength, BlockLength, loopLStart, loopLEnd);
                break;
            case NO_SCALE:
                CopyShiftIn(BId, DId, BlockId, opCopyLength, DLength, BlockLength);
                ComputeWithoutScale(
                    isTilingL ? 0 : BId, DId, BlockId, opCopyLength, DLength, BlockLength, loopLStart, loopLEnd);
                break;
            case NO_SHIFT:
                CopyScaleIn(BId, DId, BlockId, opCopyLength, DLength, BlockLength);
                ComputeWithoutShift(
                    isTilingL ? 0 : BId, DId, BlockId, opCopyLength, DLength, BlockLength, loopLStart, loopLEnd);
                break;
            default:
                break;
        }
    }

protected:
    int64_t inputB;
    int64_t inputL;
    int64_t inputD;
    int64_t ubLength;
    int64_t alignedD;
    int64_t frontNum;
    int64_t frontLength;
    int64_t tailNum;
    int64_t tailLength;
    int64_t batchStartId;
    int64_t useDTiling;
    int64_t parameterStatus;

    TPipe* pipe;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> QueueX;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> QueueY;
    TQue<QuePosition::VECIN, 1> QueueScale;
    TQue<QuePosition::VECIN, 1> QueueShift;

    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> scaleGm;
    GlobalTensor<T> shiftGm;
};

template <typename T>
class ModulateB : public NormModulate::ModulateBase<T> {
public:
    __aicore__ inline ModulateB(TPipe* pipe, const ModulateTilingData* tiling)
        : NormModulate::ModulateBase<T>(pipe, tiling)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR shift, GM_ADDR y)
    {
        int64_t blockIdx = GetBlockIdx();
        int64_t currentLength = (blockIdx < this->frontNum) ? this->frontLength : this->tailLength;
        int64_t baseOffset;
        int64_t scaleShiftOffset;
        if(blockIdx < this->frontNum) {
            baseOffset = this->frontLength * this->inputL * this->inputD * blockIdx;
            scaleShiftOffset = this->frontLength * this->inputD * blockIdx;
        }else {
            int64_t relativeIdx = blockIdx - this->frontNum;
            baseOffset = this->frontLength * this->inputL * this->inputD * this->frontNum +
                        this->tailLength * this->inputL * this->inputD * relativeIdx;
            scaleShiftOffset = this->frontLength * this->inputD * this->frontNum +
                        this->tailLength * this->inputD * relativeIdx;
        }
        int64_t baseBufferSize = currentLength * this->inputL * this->inputD;
        int64_t scaleShiftBufferSize = currentLength * this->inputD;
        this->xGm.SetGlobalBuffer((__gm__ T*)x + baseOffset, baseBufferSize);
        this->yGm.SetGlobalBuffer((__gm__ T*)y + baseOffset, baseBufferSize);
        this->scaleGm.SetGlobalBuffer((__gm__ T*)scale + scaleShiftOffset, scaleShiftBufferSize);
        this->shiftGm.SetGlobalBuffer((__gm__ T*)shift + scaleShiftOffset, scaleShiftBufferSize);
    }

    __aicore__ inline void Process()
    {
        int64_t BNum = this->GetNum();
        bool isOutOfUbLength = (this->inputD > this->ubLength);
        int64_t D = isOutOfUbLength ? (this->inputD / this->ubLength) : 0;
        int64_t remainLength = isOutOfUbLength ? (this->inputD % this->ubLength) : 0;

        for (int64_t i = 0; i < BNum; ++i) {
            if (isOutOfUbLength) {
                for (int64_t j = 0; j < D; ++j) {
                    this->CopyAndCompute(i, j, 0, this->ubLength, this->ubLength, 0, 0, this->inputL, false);
                }
                if (remainLength != 0) {
                    this->CopyAndCompute(i, D, 0, remainLength, this->ubLength, 0, 0, this->inputL, false);
                }
            } else {
                this->CopyAndCompute(i, 0, 0, this->inputD, 0, 0, 0, this->inputL, false);
            }
        }
    }
};

template <typename T>
class ModulateL : public NormModulate::ModulateBase<T> {
public:
    __aicore__ inline ModulateL(TPipe* pipe, const ModulateTilingData* tiling)
        : NormModulate::ModulateBase<T>(pipe, tiling)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR shift, GM_ADDR y)
    {
        int64_t blockIdx = GetBlockIdx();
        int64_t currentLength; 
        int64_t baseOffset;
        if(blockIdx < this->frontNum) {
            currentLength = this->frontLength;
            baseOffset = this->frontLength * this->inputD * blockIdx;
            this->batchStartId = (this->frontLength * blockIdx) / this->inputL;
        }else {
            currentLength = this->tailLength;
            int64_t relativeIdx = blockIdx - this->frontNum;
            baseOffset = this->frontLength * this->inputD * this->frontNum +
                        this->tailLength * this->inputD * relativeIdx;
            this->batchStartId = (this->frontLength * this->frontNum + 
                                this->tailLength * relativeIdx) / this->inputL;
        }
        int64_t bufferSize = currentLength * this->inputD;
        int64_t scaleShiftOffset = this->batchStartId * this->inputD;
        this->xGm.SetGlobalBuffer((__gm__ T*)x + baseOffset, bufferSize);
        this->yGm.SetGlobalBuffer((__gm__ T*)y + baseOffset, bufferSize);
        this->scaleGm.SetGlobalBuffer((__gm__ T*)scale + scaleShiftOffset, this->inputD);
        this->shiftGm.SetGlobalBuffer((__gm__ T*)shift + scaleShiftOffset, this->inputD);
    }

    __aicore__ inline void Process()
    {
        int64_t LNum = this->GetNum();
        int64_t batchEndId = this->GetBatchEndId();
        int64_t secondFirstL = this->GetSecondFirstL();
        bool isSingleBatch = (this->batchStartId == batchEndId);
        bool isOutOfUbLength = (this->inputD > this->ubLength);
        int64_t D = isOutOfUbLength ? (this->inputD / this->ubLength) : 0;
        int64_t remainLength = isOutOfUbLength ? (this->inputD % this->ubLength) : 0;
        constexpr int64_t SINGLE_BATCH = 1;
        constexpr int64_t DOUBLE_BATCH = 2;
        int64_t batchCount = isSingleBatch ? SINGLE_BATCH : DOUBLE_BATCH;
        for (int64_t i = 0; i < batchCount; ++i) {
            int64_t startL = isSingleBatch ? 0 : (i == 0 ? 0 : secondFirstL);
            int64_t endL = isSingleBatch ? LNum : (i == 0 ? secondFirstL : LNum);
            if (isOutOfUbLength) {
                for (int64_t j = 0; j < D; ++j) {
                    this->CopyAndCompute(i, j, 0, this->ubLength, this->ubLength, 0, startL, endL, true);
                }
                if (remainLength != 0) {
                    this->CopyAndCompute(i, D, 0, remainLength, this->ubLength, 0, startL, endL, true);
                }
            } else {
                this->CopyAndCompute(i, 0, 0, this->inputD, 0, 0, startL, endL, true);
            }
        }
    }
};

template <typename T>
class ModulateD : public NormModulate::ModulateBase<T> {
public:
    __aicore__ inline ModulateD(TPipe* pipe, const ModulateTilingData* tiling)
        : NormModulate::ModulateBase<T>(pipe, tiling)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR shift, GM_ADDR y)
    {
        this->xGm.SetGlobalBuffer((__gm__ T*)x);
        this->yGm.SetGlobalBuffer((__gm__ T*)y);
        this->scaleGm.SetGlobalBuffer((__gm__ T*)scale);
        this->shiftGm.SetGlobalBuffer((__gm__ T*)shift);
    }

    __aicore__ inline void Process()
    {
        int64_t dNum = this->GetNum();

        for (int64_t i = 0; i < this->inputB; ++i) {
            if (dNum > this->ubLength) {
                int64_t BlockNum = dNum / this->ubLength;
                for (int64_t j = 0; j < BlockNum; ++j) {
                    this->CopyAndCompute(
                        i, GetBlockIdx(), j, this->ubLength, dNum, this->ubLength, 0, this->inputL, false);
                }
                int64_t remainLength = dNum % this->ubLength;
                if (remainLength != 0) {
                    this->CopyAndCompute(
                        i, GetBlockIdx(), BlockNum, remainLength, dNum, this->ubLength, 0, this->inputL, false);
                }
            } else {
                this->CopyAndCompute(i, GetBlockIdx(), 0, dNum, dNum, 0, 0, this->inputL, false);
            }
        }
    }
};
} // namespace NormModulate

extern "C" __global__ __aicore__ void modulate(
    GM_ADDR x, GM_ADDR scale, GM_ADDR shift, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);
#if (defined(DTYPE_X))
    if (TILING_KEY_IS(0)) {
        NormModulate::ModulateB<DTYPE_X> op(&pipe, &tilingData);
        op.Init(x, scale, shift, y);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        NormModulate::ModulateL<DTYPE_X> op(&pipe, &tilingData);
        op.Init(x, scale, shift, y);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        NormModulate::ModulateD<DTYPE_X> op(&pipe, &tilingData);
        op.Init(x, scale, shift, y);
        op.Process();
    }
#endif
}