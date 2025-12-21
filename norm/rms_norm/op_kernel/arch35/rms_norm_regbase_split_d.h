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
 * \file rms_norm_regbase_split_d.h
 * \brief RmsNorm regbase split D
 */
#ifndef OPS_BUILT_IN_TBE_IMPL_ASCENDC_RMS_NORM_REGBASE_SPLIT_D_H
#define OPS_BUILT_IN_TBE_IMPL_ASCENDC_RMS_NORM_REGBASE_SPLIT_D_H
#include "rms_norm_regbase_common.h"

namespace RmsNorm {
using namespace AscendC;

template <typename DX, typename DG>
class KernelRmsNormRegBaseSplitD {
public:
    __aicore__ inline KernelRmsNormRegBaseSplitD()
    {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, const RMSNormTilingData* tiling)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        numRow = tiling->num_row;
        numCol = tiling->num_col;
        blockFactor = tiling->block_factor;
        ubFactor = tiling->ub_factor;
        ubLoop = tiling->ub_loop;
        rowFactor = tiling->row_factor;
        epsilon = tiling->epsilon;
        numColAlign = tiling->num_col_align;
        colBufferLength = tiling->col_buffer_length;
        workLocalLen = colBufferLength / (CONST_FACTOR_2 * V_LENGTH);
        avgFactor = numCol != 0 ? (float)1.0 / numCol : 0;
        rowWork = (GetBlockIdx() < GetBlockNum() - 1) ? blockFactor : numRow - (GetBlockNum() - 1) * blockFactor;
        xGm.SetGlobalBuffer((__gm__ DX*)x + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);
        gammaGm.SetGlobalBuffer((__gm__ DG*)gamma, numCol);
        yGm.SetGlobalBuffer((__gm__ DX*)y + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + GetBlockIdx() * blockFactor, blockFactor);
        pipe.InitBuffer(inQueueX, DOUBLE_BUFFER_NUM, colBufferLength * sizeof(DX));
        pipe.InitBuffer(inQueueGamma, DOUBLE_BUFFER_NUM, colBufferLength * sizeof(DG));
        pipe.InitBuffer(outQueueY, DOUBLE_BUFFER_NUM, colBufferLength * sizeof(DX));
        pipe.InitBuffer(outQueueRstd, DOUBLE_BUFFER_NUM, rowFactor * sizeof(float));
        pipe.InitBuffer(xFp32Buf, ubFactor * sizeof(float));
        pipe.InitBuffer(workLocalBuf, ONCE_VECTOR_SIZE * sizeof(float));
        pipe.InitBuffer(level1Buf, ONCE_VECTOR_SIZE * sizeof(float));
        pipe.InitBuffer(level2Buf, ONCE_VECTOR_SIZE * sizeof(float));
        pipe.InitBuffer(level3Buf, ONCE_VECTOR_SIZE * sizeof(float));
        pipe.InitBuffer(tempBuf, V_LENGTH * sizeof(float));
    }

    __aicore__ inline void SubProcess(uint32_t rowRepeat, uint32_t calRowNum)
    {
        uint32_t colRepeats = CeilDiv(numCol, (uint64_t)ubFactor);

        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
        Duplicate(rstdLocal, (float)0.0, rowFactor);

        for (uint32_t row = 0; row < calRowNum; row++) {
            uint32_t split = ubLoop * ubFactor;
            uint32_t colTail = numCol - split;
            uint32_t tail = colTail % ubFactor;
            uint32_t tailLoop = colTail / ubFactor;
            uint32_t masterLoop = tail != 0 ? 1 : 0;
            masterLoop = ubLoop - tailLoop - masterLoop;
            uint64_t offsets = (rowRepeat * rowFactor + row) * numCol;
            ComputeFormer(offsets, rstdLocal, row, masterLoop, tailLoop, tail);
        }

        ComputeRstd(rstdLocal, epsilon, avgFactor, calRowNum);

        for (uint32_t repeat = 0; repeat < colRepeats; repeat++) {
            uint32_t remain = numCol - repeat * ubFactor;
            uint32_t calColNum = Min(remain, ubFactor);
            ComputeLatter(rowRepeat, calRowNum, repeat, rstdLocal, calColNum);
        }

        outQueueRstd.EnQue<float>(rstdLocal);
        CopyOutRstd(rowRepeat, calRowNum);
    }

    __aicore__ inline void Process()
    {
        uint32_t repeatTimes = CeilDiv(rowWork, rowFactor);
        for (uint32_t repeat = 0; repeat < repeatTimes; repeat++) {
            uint32_t remain = rowWork - repeat * rowFactor;
            uint32_t calRowNum = Min(remain, rowFactor);
            SubProcess(repeat, calRowNum);
        }
    }

private:
    __aicore__ inline void CopyInX(uint64_t offset, uint32_t count, uint32_t left = 0, uint32_t right = 0)
    {
        LocalTensor<DX> xLocal = inQueueX.AllocTensor<DX>();
        DataCopyPadExtParams<DX> padParams{
            true,                        // isPad
            static_cast<uint8_t>(left),  // leftPadding
            static_cast<uint8_t>(right), // rightPadding
            static_cast<DX>(0.0)         // paddingValue
        };
        DataCopyImpl<DX>(xLocal, xGm[offset], 1, count, 0, 0, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyInGamma(uint32_t colRepeat, uint32_t calColNum)
    {
        LocalTensor<DG> gammaLocal = inQueueGamma.AllocTensor<DG>();
        DataCopyImpl<DG>(gammaLocal, gammaGm[colRepeat * ubFactor], 1, calColNum, 0, 0);
        inQueueGamma.EnQue(gammaLocal);
    }

    __aicore__ inline void ComputeFormerHandle(
        LocalTensor<float>& dstLocal, uint64_t srcOffset, uint64_t dstOffset, uint32_t count, uint32_t power)
    {
        uint32_t calCount = CeilAlign((uint64_t)(count * sizeof(DX)), ALIGN_32_FACTOR) / sizeof(DX);
        CopyInX(srcOffset, count, 0, calCount - count);
        LocalTensor<DX> xLocal = inQueueX.DeQue<DX>();
        LocalTensor<float> xFp32 = xFp32Buf.Get<float>();
        LocalTensor<float> workLocal = workLocalBuf.Get<float>();
        uint32_t calNum = CeilAlign((uint64_t)(count * sizeof(DX)), ALIGN_512_FACTOR) / sizeof(DX);
        if (calNum - calCount > 0) {
            Duplicate(xLocal[calCount], (DX)0.0, calNum - calCount);
        }
        ComputeFormerImplV2(dstLocal, xLocal, workLocal, dstOffset, calNum, power);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeFormer(
        uint32_t curRow, LocalTensor<float> dstLocal, uint32_t position, uint32_t masterLoop, uint32_t tailLoop,
        uint32_t tail)
    {
        uint64_t offset{curRow};
        uint32_t loop = 0;
        uint32_t level1{0};
        uint32_t level2{0};
        uint32_t level3{0};
        LocalTensor<float> level1Local = level1Buf.Get<float>();
        LocalTensor<float> level2Local = level2Buf.Get<float>();
        LocalTensor<float> level3Local = level3Buf.Get<float>();
        LocalTensor<float> tempLocal = tempBuf.Get<float>();
        Duplicate(level1Local, (float)0.0, ONCE_VECTOR_SIZE);
        Duplicate(level2Local, (float)0.0, ONCE_VECTOR_SIZE);
        Duplicate(level3Local, (float)0.0, ONCE_VECTOR_SIZE);
        // stage1: 处理整尾块逻辑
        for (uint32_t repeat = 0; repeat < tailLoop; repeat++) {
            ComputeFormerHandle(tempLocal, offset, 0, ubFactor, ubFactor);
            offset += ubFactor;

            ComputeFormerHandle(tempLocal, offset, 1, ubFactor, ubFactor);
            offset += ubFactor;

            ComputeSum(level1Local, tempLocal, level1, SUM_COUNT);
            level1 += 1;
            ComputeMultiLevelReduce(level1Local, level2Local, level3Local, level1, level2, level3);
            loop += 1;
        }
        // stage2: 处理非整尾块逻辑
        if (tail > 0 && tail <= ubFactor / CONST_FACTOR_2) {
            ComputeFormerHandle(tempLocal, offset, 0, ubFactor / CONST_FACTOR_2 + tail, ubFactor / CONST_FACTOR_2);
            offset += ubFactor / CONST_FACTOR_2 + tail;

            ComputeFormerHandle(tempLocal, offset, 1, ubFactor / CONST_FACTOR_2, ubFactor / CONST_FACTOR_2);
            offset += ubFactor / CONST_FACTOR_2;

            ComputeSum(level1Local, tempLocal, level1, SUM_COUNT);
        } else if (tail > ubFactor / CONST_FACTOR_2) {
            ComputeFormerHandle(tempLocal, offset, 0, ubFactor, ubFactor);
            offset += ubFactor;

            ComputeFormerHandle(tempLocal, offset, 1, tail, ubFactor / CONST_FACTOR_2);
            offset += tail;

            ComputeSum(level1Local, tempLocal, level1, SUM_COUNT);
        }
        level1 += 1;
        ComputeMultiLevelReduce(level1Local, level2Local, level3Local, level1, level2, level3);
        loop += 1;
        // stage3: 处理主块逻辑
        for (uint32_t repeat = 0; repeat < masterLoop; repeat++) {
            ComputeFormerHandle(level1Local, offset, level1, ubFactor, ubFactor);
            offset += ubFactor;
            level1 += 1;
            ComputeMultiLevelReduce(level1Local, level2Local, level3Local, level1, level2, level3);
            loop += 1;
        }
        ComputeMultiLevelRstd<false>(dstLocal, position, level1Local, level2Local, level3Local, level1, level2, level3);
    }

    __aicore__ inline void ComputeLatter(
        uint32_t rowRepeat, uint32_t calRowNum, uint32_t colRepeat, LocalTensor<float>& rstdLocal, uint32_t calColNum)
    {
        CopyInGamma(colRepeat, calColNum);
        LocalTensor<DG> gammaLocal = inQueueGamma.DeQue<DG>();
        for (uint32_t row = 0; row < calRowNum; row++) {
            uint64_t offset = (rowRepeat * rowFactor + row) * numCol + colRepeat * ubFactor;
            CopyInX(offset, calColNum);
            LocalTensor<DX> xLocal = inQueueX.DeQue<DX>();
            LocalTensor<DX> yLocal = outQueueY.AllocTensor<DX>();
            uint32_t calCount = CeilAlign((uint64_t)(calColNum * sizeof(DX)), ALIGN_512_FACTOR) / sizeof(DX);
            ComputeLatterY<DX, DG>(xLocal, gammaLocal, yLocal, rstdLocal, row, calCount);
            inQueueX.FreeTensor(xLocal);
            outQueueY.EnQue<DX>(yLocal);
            CopyOutY(rowRepeat * rowFactor + row, colRepeat, calColNum);
        }
        inQueueGamma.FreeTensor(gammaLocal);
    }

    __aicore__ inline void CopyOutY(uint32_t curRow, uint32_t curCol, uint32_t calColNum)
    {
        LocalTensor<DX> yLocal = outQueueY.DeQue<DX>();
        DataCopyImpl<DX>(yGm[curRow * numCol + curCol * ubFactor], yLocal, 1, calColNum, 0, 0);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutRstd(uint32_t rowRepeat, uint32_t calRowNum)
    {
        LocalTensor<float> rstdLocal = outQueueRstd.DeQue<float>();
        DataCopyImpl<float>(rstdGm[rowRepeat * rowFactor], rstdLocal, 1, calRowNum, 0, 0);
        outQueueRstd.FreeTensor(rstdLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> inQueueGamma;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueRstd;
    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> workLocalBuf;
    TBuf<TPosition::VECCALC> level1Buf;
    TBuf<TPosition::VECCALC> level2Buf;
    TBuf<TPosition::VECCALC> level3Buf;
    TBuf<TPosition::VECCALC> tempBuf;
    GlobalTensor<DX> xGm;
    GlobalTensor<DG> gammaGm;
    GlobalTensor<DX> yGm;
    GlobalTensor<float> rstdGm;
    uint64_t numRow;
    uint64_t numCol;
    uint64_t numColAlign;
    uint64_t colBufferLength;
    uint32_t blockFactor;
    uint32_t rowFactor;
    uint32_t ubFactor;
    uint32_t ubLoop;
    float epsilon;
    float avgFactor;
    uint32_t rowWork{1};
    uint32_t workLocalLen{0};
};
} // namespace RmsNorm
#endif // OPS_BUILT_IN_TBE_IMPL_ASCENDC_RMS_NORM_REGBASE_SPLIT_D_H