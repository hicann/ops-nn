/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file add_rms_norm_regbase.h
 * \brief
 */
#ifndef ADD_RMS_NORM_REGBASE_H
#define ADD_RMS_NORM_REGBASE_H
#include "add_rms_norm_regbase_common.h"
#include "../../rms_norm/rms_norm_base.h"
namespace AddRmsNorm {
using namespace AscendC;
constexpr uint64_t ALIGN_32_FACTOR = 32;
constexpr int32_t CONST_FACTOR_2 = 2;
constexpr int32_t NDDMA_DIM = 5;
constexpr int32_t UNROLL_NUM = 2;
using RmsNorm::DataCopyCustom;
using RmsNorm::DataCopyImpl;

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}
template <typename T>
class KernelAddRmsNormRegBase {
public:
    __aicore__ inline KernelAddRmsNormRegBase(TPipe* pipe)
    {
        pPipe = pipe;
    }
    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, GM_ADDR x,
        const AddRMSNormRegbaseTilingData* tiling)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");
        numRow = tiling->numRow;
        numCol = tiling->numCol;
        blockFactor = tiling->blockFactor;
        ubFactor = tiling->ubFactor;
        ubLoop = tiling->ubLoop;
        rowFactor = tiling->rowFactor;
        epsilon = tiling->epsilon;
        numColAlign = tiling->numColAlign;
        colBufferLength = tiling->colBuferLength;
        multiNNum = tiling->multiNNum;
        avgFactor = tiling->avgFactor;
        isNddma = tiling->isNddma;
        rowWork = (GetBlockIdx() < GetBlockNum() - 1) ? blockFactor : numRow - (GetBlockNum() - 1) * blockFactor;
        xGm1.SetGlobalBuffer((__gm__ T*)x1 + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);
        xGm2.SetGlobalBuffer((__gm__ T*)x2 + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma, numCol);
        yGm.SetGlobalBuffer((__gm__ T*)y + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + GetBlockIdx() * blockFactor, blockFactor);
        xOutGm.SetGlobalBuffer((__gm__ T*)x + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);
        pPipe->InitBuffer(inQueueX1, DOUBLE_BUFFER_NUM, colBufferLength * sizeof(T) * multiNNum);
        pPipe->InitBuffer(inQueueX2, DOUBLE_BUFFER_NUM, colBufferLength * sizeof(T) * multiNNum);
        pPipe->InitBuffer(inQueueGamma, DOUBLE_BUFFER_NUM, colBufferLength * sizeof(T));
        pPipe->InitBuffer(outQueueY, DOUBLE_BUFFER_NUM, colBufferLength * sizeof(T) * multiNNum);
        pPipe->InitBuffer(outQueueX, DOUBLE_BUFFER_NUM, colBufferLength * sizeof(T) * multiNNum);
        pPipe->InitBuffer(outQueueRstd, DOUBLE_BUFFER_NUM, rowFactor * sizeof(float));
        pPipe->InitBuffer(xFp32Buf, numColAlign * sizeof(float) * multiNNum);
        pPipe->InitBuffer(workLocalBuf, ONCE_VECTOR_SIZE * UNROLL_NUM * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        CopyInGamma();
        LocalTensor<T> gammaLocal = inQueueGamma.DeQue<T>();
        uint32_t repeatTimes = CeilDiv(rowWork, rowFactor);
        for (uint32_t repeat = 0; repeat < repeatTimes; repeat++) {
            uint32_t remain = rowWork - repeat * rowFactor;
            uint32_t calRowNum = Min(remain, rowFactor);
            SubProcess(repeat, calRowNum, gammaLocal);
        }
        inQueueGamma.FreeTensor(gammaLocal);
    }

    __aicore__ inline void SubProcess(uint32_t rowRepeat, uint32_t calRowNum, LocalTensor<T>& gammaLocal)
    {
        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
        for (uint32_t row = 0; row < calRowNum; row += multiNNum) {
            uint64_t offset = (rowRepeat * rowFactor + row) * numCol;
            uint32_t curRows = calRowNum - row > multiNNum ? multiNNum : (calRowNum - row);
            uint32_t curCols = CeilAlign((uint64_t)(numCol * sizeof(T)), ALIGN_32_FACTOR) / sizeof(T);
            if (isNddma) {
                CopyInXMutiNddma(offset, curRows);
            } else {
                CopyInXMutiMoveAlign(offset, curCols, curRows);
            }
            Compute(row, gammaLocal, rstdLocal, curCols, curRows);
            CopyOutX(offset, curRows, curCols);
            CopyOutY(offset, curRows, curCols);
        }
        outQueueRstd.EnQue<float>(rstdLocal);
        CopyOutRstd(rowRepeat, calRowNum);
    }

private:
    __aicore__ inline void CopyInXMutiNddma(uint64_t offset, uint32_t curRows = 0)
    {
        // load x1
        LocalTensor<T> xLocal1 = inQueueX1.AllocTensor<T>();
        dmaParam_ = {
            {{1, 1, 1, (uint32_t)numCol, 1},                 // SrcStride
             {1, 1, 1, (uint32_t)numColAlign, 1},            // DstStride
             {1, 1, 1, (uint32_t)curRows, (uint32_t)numCol}, // LoopSize
             {0, 0, 0, 0, 0},                                // FrontPad
             {0, 0, 0, 0, (uint8_t)(numColAlign - numCol)}},
            0}; // pad value
        DataCopy(xLocal1, xGm1[offset], dmaParam_);
        inQueueX1.EnQue(xLocal1);
        // load x2
        LocalTensor<T> xLocal2 = inQueueX2.AllocTensor<T>();
        DataCopy(xLocal2, xGm2[offset], dmaParam_);
        inQueueX2.EnQue(xLocal2);
    }

    __aicore__ inline void CopyInXMutiMoveAlign(uint64_t offset, uint32_t curCols, uint32_t curRows = 0)
    {
        LocalTensor<T> xLocal1 = inQueueX1.AllocTensor<T>();
        LocalTensor<T> xLocal2 = inQueueX2.AllocTensor<T>();
        DataCopyExtParams extParams{
            static_cast<uint16_t>(curRows),                                               // blockCount
            static_cast<uint32_t>(numCol * sizeof(T)),                                    // blockLen
            static_cast<uint32_t>(0),                                                     // srcStride
            static_cast<uint32_t>((numColAlign - curCols) * sizeof(T) / ALIGN_32_FACTOR), // dstStride
            0                                                                             // rsv
        };
        DataCopyPadExtParams<T> padParams{
            true,                                   // isPad
            static_cast<uint8_t>(0),                // leftPadding
            static_cast<uint8_t>(curCols - numCol), // rightPadding
            static_cast<T>(0.0)                     // paddingValue
        };
        DataCopyPad(xLocal1, xGm1[offset], extParams, padParams);
        DataCopyPad(xLocal2, xGm2[offset], extParams, padParams);
        inQueueX1.EnQue(xLocal1);
        inQueueX2.EnQue(xLocal2);
    }

    __aicore__ inline void CopyInGamma()
    {
        LocalTensor<T> gammaLocal = inQueueGamma.AllocTensor<T>();
        DataCopyCustom<T>(gammaLocal, gammaGm, numCol);
        inQueueGamma.EnQue(gammaLocal);
    }

    __aicore__ inline void Compute(
        uint32_t curRow, LocalTensor<T> gammaLocal, LocalTensor<float> rstdLocal, uint32_t curCols, uint32_t curRows)
    {
        LocalTensor<T> xLocal1 = inQueueX1.DeQue<T>();
        LocalTensor<T> xLocal2 = inQueueX2.DeQue<T>();
        LocalTensor<float> xFp32 = xFp32Buf.Get<float>();
        LocalTensor<float> workLocal = workLocalBuf.Get<float>();
        LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        LocalTensor<T> xOutLocal = outQueueX.AllocTensor<T>();

        uint64_t dupLen = numColAlign - curCols;
        if (dupLen > 0 && !isNddma) {
            for (uint32_t rowIdx = 0; rowIdx < curRows; rowIdx++) {
                Duplicate(xLocal1[rowIdx * numColAlign + curCols], (T)0.0, dupLen);
                Duplicate(xLocal2[rowIdx * numColAlign + curCols], (T)0.0, dupLen);
            }
            PipeBarrier<PIPE_V>();
        }

        ComputeFormerImplV1MultiN(
            xLocal1, xLocal2, xFp32, workLocal, rstdLocal, avgFactor, epsilon, curRow, numColAlign, ubFactor, curRows);

        inQueueX1.FreeTensor(xLocal1);
        inQueueX2.FreeTensor(xLocal2);
        ComputeYMultiN(xFp32, gammaLocal, yLocal, rstdLocal, curRow, numColAlign, xOutLocal, curRows);

        outQueueY.EnQue<T>(yLocal);
        outQueueX.EnQue<T>(xOutLocal);
    }

    __aicore__ inline void CopyOutY(uint64_t offset, uint32_t curRows, uint32_t colAlign)
    {
        LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        uint32_t srcStride = (numColAlign - colAlign) * sizeof(T) / ALIGN_32_FACTOR;
        DataCopyImpl<T>(yGm[offset], yLocal, curRows, numCol, srcStride, 0);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutRstd(uint64_t offset, uint32_t num)
    {
        LocalTensor<float> rstdLocal = outQueueRstd.DeQue<float>();
        DataCopyImpl<float>(rstdGm[offset * rowFactor], rstdLocal, 1, num, 0, 0);
        outQueueRstd.FreeTensor(rstdLocal);
    }

    __aicore__ inline void CopyOutX(uint64_t offset, uint32_t curRows, uint32_t colAlign)
    {
        LocalTensor<T> xLocal = outQueueX.DeQue<T>();
        uint32_t srcStride = (numColAlign - colAlign) * sizeof(T) / ALIGN_32_FACTOR;
        DataCopyImpl<T>(xOutGm[offset], xLocal, curRows, numCol, srcStride, 0);
        outQueueX.FreeTensor(xLocal);
    }

private:
    TPipe* pPipe = nullptr;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> inQueueX1;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> inQueueX2;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> inQueueGamma;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueRstd;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueX;
    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> workLocalBuf;
    MultiCopyParams<T, NDDMA_DIM> dmaParam_;
    GlobalTensor<T> xGm1;
    GlobalTensor<T> xGm2;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> yGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<T> xOutGm;
    uint32_t numRow;
    uint32_t numCol;
    uint32_t numColAlign;
    uint32_t colBufferLength;
    uint32_t blockFactor;
    uint32_t rowFactor;
    uint32_t ubFactor;
    uint32_t ubLoop;
    float epsilon;
    float avgFactor;
    uint32_t rowWork{1};
    uint32_t multiNNum;
    uint32_t isNddma;
};
} // namespace AddRmsNorm
#endif // _ADD_RMS_NORM_REGBASE_H