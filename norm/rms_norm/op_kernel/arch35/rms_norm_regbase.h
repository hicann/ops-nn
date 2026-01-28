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
 * \file rms_norm_regbase.h
 * \brief RmsNorm regbase
 */
#ifndef OPS_BUILT_IN_TBE_IMPL_ASCENDC_RMS_NORM_REGBASE_H
#define OPS_BUILT_IN_TBE_IMPL_ASCENDC_RMS_NORM_REGBASE_H
#include "rms_norm_regbase_common.h"

namespace RmsNorm {
using namespace AscendC;

template <typename DX, typename DG>
class KernelRmsNormRegBase {
public:
    __aicore__ inline KernelRmsNormRegBase()
    {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, const RMSNormArch35TilingData* tiling)
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
        multiNNum = tiling->multi_n_num;
        colBufferLength = tiling->col_buffer_length;
        isNddma = tiling->is_nddma;
        workLocalLen = colBufferLength / (CONST_FACTOR_2 * V_LENGTH);
        avgFactor = numCol != 0 ? (float)1.0 / numCol : 0;
        rowWork = (GetBlockIdx() < GetBlockNum() - 1) ? blockFactor : numRow - (GetBlockNum() - 1) * blockFactor;
        xGm.SetGlobalBuffer((__gm__ DX*)x + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);
        gammaGm.SetGlobalBuffer((__gm__ DG*)gamma, numCol);
        yGm.SetGlobalBuffer((__gm__ DX*)y + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + GetBlockIdx() * blockFactor, blockFactor);
        pipe.InitBuffer(inQueueX, DOUBLE_BUFFER_NUM, colBufferLength * sizeof(DX) * multiNNum);
        pipe.InitBuffer(inQueueGamma, DOUBLE_BUFFER_NUM, colBufferLength * sizeof(DG));
        pipe.InitBuffer(outQueueY, DOUBLE_BUFFER_NUM, colBufferLength * sizeof(DX) * multiNNum);

        pipe.InitBuffer(outQueueRstd, DOUBLE_BUFFER_NUM, rowFactor * sizeof(float));
        if constexpr (!is_same<DX, float>::value) {
            pipe.InitBuffer(xFp32Buf, numColAlign * sizeof(float) * multiNNum);
        }
        pipe.InitBuffer(workLocalBuf, ONCE_VECTOR_SIZE * UNROLL_NUM * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        CopyInGamma();
        LocalTensor<DG> gammaLocal = inQueueGamma.DeQue<DG>();
        uint32_t repeatTimes = CeilDiv(rowWork, rowFactor);
        for (uint32_t repeat = 0; repeat < repeatTimes; repeat++) {
            uint32_t remain = rowWork - repeat * rowFactor;
            uint32_t calRowNum = Min(remain, rowFactor);
            SubProcess(repeat, calRowNum, gammaLocal);
        }
        inQueueGamma.FreeTensor(gammaLocal);
    }

    __aicore__ inline void SubProcess(uint32_t rowRepeat, uint32_t calRowNum, LocalTensor<DG>& gammaLocal)
    {
        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
        for (uint32_t row = 0; row < calRowNum; row += multiNNum) {
            uint64_t offset = (rowRepeat * rowFactor + row) * numCol;
            uint32_t curRows = calRowNum - row > multiNNum ? multiNNum : (calRowNum - row);
            uint32_t curCols = (CeilAlign((uint64_t)(numCol * sizeof(DX)), ALIGN_32_FACTOR) / sizeof(DX));
            if (isNddma) {
                CopyInMutiNddma(offset, numCol, 0, curCols - numCol, curRows);
            } else {
                CopyInMutiMoveAlign(offset, curCols, curRows);
            }
            Compute(row, gammaLocal, rstdLocal, curCols, curRows);
            CopyOutY(offset, curRows, curCols);
        }
        outQueueRstd.EnQue<float>(rstdLocal);
        CopyOutRstd(rowRepeat, calRowNum);
    }

private:
    __aicore__ inline void CopyInMutiNddma(
        uint64_t offset, uint32_t count, uint32_t left = 0, uint32_t right = 0, uint32_t curRows = 0)
    {
        LocalTensor<DX> xLocal = inQueueX.AllocTensor<DX>();
        dmaParam_ = {
            {{1, 1, 1, (uint32_t)numCol, 1},                 // SrcStride
             {1, 1, 1, (uint32_t)numColAlign, 1},            // DstStride
             {1, 1, 1, (uint32_t)curRows, (uint32_t)numCol}, // LoopSize
             {0, 0, 0, 0, 0},                                // FrontPad
             {0, 0, 0, 0, (uint8_t)(numColAlign - numCol)}},
            0}; // pad value
        DataCopy(xLocal, xGm[offset], dmaParam_);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyInMutiMoveAlign(uint64_t offset, uint32_t curCols, uint32_t curRows = 0)
    {
        LocalTensor<DX> xLocal = inQueueX.AllocTensor<DX>();
        DataCopyExtParams extParams{
            static_cast<uint16_t>(curRows),                                                // blockCount
            static_cast<uint32_t>(numCol * sizeof(DX)),                                    // blockLen
            static_cast<uint32_t>(0),                                                      // srcStride
            static_cast<uint32_t>((numColAlign - curCols) * sizeof(DX) / ALIGN_32_FACTOR), // dstStride
            0                                                                              // rsv
        };
        DataCopyPadExtParams<DX> padParams{
            true,                                   // isPad
            static_cast<uint8_t>(0),                // leftPadding
            static_cast<uint8_t>(curCols - numCol), // rightPadding
            static_cast<DX>(0.0)                    // paddingValue
        };
        DataCopyPad(xLocal, xGm[offset], extParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void CopyInGamma()
    {
        LocalTensor<DG> gammaLocal = inQueueGamma.AllocTensor<DG>();
        DataCopyCustom<DG>(gammaLocal, gammaGm, numCol);
        inQueueGamma.EnQue(gammaLocal);
    }

    __aicore__ inline void Compute(
        uint32_t curRow, LocalTensor<DG> gammaLocal, LocalTensor<float> rstdLocal, uint32_t curCols, uint32_t curRows)
    {
        LocalTensor<DX> xLocal = inQueueX.DeQue<DX>();
        LocalTensor<float> xFp32;
        if constexpr (!is_same<DX, float>::value) {
            xFp32 = xFp32Buf.Get<float>();
        }
        LocalTensor<float> workLocal = workLocalBuf.Get<float>();
        LocalTensor<DX> yLocal = outQueueY.AllocTensor<DX>();

        uint64_t dupLen = numColAlign - curCols;
        if (dupLen > 0) {
            for (uint32_t rowIdx = 0; rowIdx < curRows; rowIdx++) {
                Duplicate(xLocal[rowIdx * numColAlign + curCols], (DX)0.0, dupLen);
            }
            PipeBarrier<PIPE_V>();
        }

        ComputeFormerImplV1MultiN(
            xLocal, xFp32, workLocal, rstdLocal, avgFactor, epsilon, curRow, numColAlign, ubFactor, curRows);

        if constexpr (is_same<DX, half>::value || is_same<DX, bfloat16_t>::value) {
            inQueueX.FreeTensor(xLocal);
            ComputeYMultiN<DX, DG>(xFp32, gammaLocal, yLocal, rstdLocal, curRow, numColAlign, curRows);
        } else {
            ComputeYMultiN<DX, DG>(xLocal, gammaLocal, yLocal, rstdLocal, curRow, numColAlign, curRows);
            inQueueX.FreeTensor(xLocal);
        }
        outQueueY.EnQue<DX>(yLocal);
    }

    __aicore__ inline void CopyOutY(uint64_t offset, uint32_t curRows, uint32_t colAlign)
    {
        LocalTensor<DX> yLocal = outQueueY.DeQue<DX>();
        uint32_t srcStride = (numColAlign - colAlign) * sizeof(DX) / ALIGN_32_FACTOR;
        DataCopyImpl<DX>(yGm[offset], yLocal, curRows, numCol, srcStride, 0);
        outQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOutRstd(uint64_t offset, uint32_t num)
    {
        LocalTensor<float> rstdLocal = outQueueRstd.DeQue<float>();
        DataCopyImpl<float>(rstdGm[offset * rowFactor], rstdLocal, 1, num, 0, 0);
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
    MultiCopyParams<DX, NDDMA_DIM> dmaParam_;
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
    uint32_t multiNNum;
    uint32_t isNddma;
    float epsilon;
    float avgFactor;
    uint32_t rowWork{1};
    uint32_t workLocalLen{0};
};
} // namespace RmsNorm
#endif // OPS_BUILT_IN_TBE_IMPL_ASCENDC_RMS_NORM_REGBASE_H