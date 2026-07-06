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
 * \file rms_norm_regbase_perf.h
 * \brief RmsNorm regbase perf schedule
 */
#ifndef OPS_BUILT_IN_TBE_IMPL_ASCENDC_RMS_NORM_REGBASE_PERF_H
#define OPS_BUILT_IN_TBE_IMPL_ASCENDC_RMS_NORM_REGBASE_PERF_H
#include "rms_norm_regbase_common.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace RmsNorm {
using namespace AscendC;
using NormCommon::NormCommonRegbase::LoadRegForDtype;
using NormCommon::NormCommonRegbase::StoreRegForDtype;

template <typename DX, typename DG>
class KernelRmsNormRegBasePerf {
public:
    __aicore__ inline KernelRmsNormRegBasePerf() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd,
                                const RMSNormArch35TilingData* tiling)
    {
        // Tilingdata init
        is_gemma = tiling->is_gemma;
        curBlockIdx = GetBlockIdx();
        coreNum = GetBlockNum();
        numRow = tiling->num_row;
        numCol = tiling->num_col;
        numColAlign = tiling->num_col_align;
        ubFactor = tiling->ub_factor;
        blockFactor = tiling->block_factor;
        blockFactorTail = tiling->last_block_factor;

        epsilon = tiling->epsilon;
        avgFactor = tiling->avg_factor;
        colFlodFactor = tiling->col_flod_factor;

        uint64_t rstdBlockCount = blockFactor;
        uint64_t xBlockCount = rstdBlockCount * numCol;
        uint64_t curRstdBlockCount = rstdBlockCount;
        uint64_t curXBlockCount = xBlockCount;
        curBlockFactor = blockFactor;
        if (curBlockIdx == (coreNum - 1)) {
            curBlockFactor = blockFactorTail;
            curRstdBlockCount = blockFactorTail;
            curXBlockCount = curRstdBlockCount * numCol;
        }

        curBlockLoops = (curBlockFactor + ubFactor - 1) / ubFactor;
        curUbTails = curBlockFactor - (curBlockLoops - 1) * ubFactor;

        // init gm buffer
        xGm.SetGlobalBuffer((__gm__ DX*)x + curBlockIdx * xBlockCount, curXBlockCount);
        gammaGm.SetGlobalBuffer((__gm__ DG*)gamma, numCol);
        yGm.SetGlobalBuffer((__gm__ DX*)y + curBlockIdx * xBlockCount, curXBlockCount);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + curBlockIdx * rstdBlockCount, curRstdBlockCount);

        // init que buffer
        pipe.InitBuffer(inQueueGamma, 1, numColAlign * sizeof(DG));
        pipe.InitBuffer(inQueueX, DOUBLE_BUFFER_NUM, ubFactor * numColAlign * sizeof(DX));
        pipe.InitBuffer(outQueueY, DOUBLE_BUFFER_NUM, ubFactor * numColAlign * sizeof(DX));
        uint64_t rstdUbSizeAlign = (ubFactor + blockSizeB32 - 1) / blockSizeB32 * blockSizeB32 *
                                   sizeof(float); // vector length align
        pipe.InitBuffer(outQueueRstd, DOUBLE_BUFFER_NUM, rstdUbSizeAlign);
        // init tmp buffer
        uint64_t firstVcaddResult = ubFactor *
                                    (((colFlodFactor + VectorLenB32 - 1) / VectorLenB32 + blockSizeB32 - 1) /
                                     blockSizeB32 * blockSizeB32) *
                                    sizeof(float);
        pipe.InitBuffer(xTmpBuf, firstVcaddResult);
        // branch 4 need 512B tmpBuffer
        pipe.InitBuffer(xReduceTmpBuf, rstdUbSizeAlign);
    }

    __aicore__ inline void Process()
    {
        // gamma copy params
        DataCopyPadExtParams<DG> dataCopyPadExtParamsGamma;
        dataCopyPadExtParamsGamma.isPad = false;
        dataCopyPadExtParamsGamma.leftPadding = 0;
        dataCopyPadExtParamsGamma.rightPadding = 0;
        dataCopyPadExtParamsGamma.paddingValue = 0;
        DataCopyExtParams copyInParamsGamma;
        copyInParamsGamma.blockCount = 1;
        copyInParamsGamma.blockLen = numCol * sizeof(DG);
        copyInParamsGamma.srcStride = 0;
        copyInParamsGamma.dstStride = 0;

        LocalTensor<DG> gammaLocal = inQueueGamma.AllocTensor<DG>();
        DataCopyPad(gammaLocal, gammaGm, copyInParamsGamma, dataCopyPadExtParamsGamma);

        inQueueGamma.EnQue(gammaLocal);
        inQueueGamma.DeQue<DG>();

        for (uint64_t i = 0; i < curBlockLoops; i++) {
            curUbFactor = (i == (curBlockLoops - 1)) ? curUbTails : ubFactor;

            DataCopyPadExtParams<DX> dataCopyPadExtParamsX;
            dataCopyPadExtParamsX.isPad = false;
            dataCopyPadExtParamsX.leftPadding = 0;
            dataCopyPadExtParamsX.rightPadding = 0;
            dataCopyPadExtParamsX.paddingValue = 0;
            DataCopyExtParams copyInParamsX;
            copyInParamsX.blockCount = curUbFactor;
            copyInParamsX.blockLen = numCol * sizeof(DX);
            copyInParamsX.srcStride = 0;
            copyInParamsX.dstStride = (numColAlign - numCol) * sizeof(DX) / blockSize;
            LocalTensor<DX> xLocal = inQueueX.AllocTensor<DX>();
            DataCopyPad(xLocal, xGm[i * ubFactor * numCol], copyInParamsX, dataCopyPadExtParamsX);
            inQueueX.EnQue(xLocal);
            inQueueX.DeQue<DX>();
            LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
            LocalTensor<float> xTmpLocal = xTmpBuf.Get<float>();
            LocalTensor<float> xReduceTmpLocal = xReduceTmpBuf.Get<float>();
            NormCommon::NormCommonRegbase::CalculateSquareReduceSum<DX>(
                xLocal, xReduceTmpLocal, xTmpLocal, static_cast<uint16_t>(curUbFactor),
                static_cast<uint32_t>(numColAlign), static_cast<uint32_t>(numCol), static_cast<uint32_t>(colFlodFactor),
                static_cast<uint32_t>(blockSizeB32), static_cast<uint32_t>(numColAlign));
            NormCommon::ComputeRstdNewtonRaphson<false, true>(
                xReduceTmpLocal, rstdLocal, static_cast<uint32_t>(curUbFactor), epsilon, avgFactor, VectorLenB32);

            outQueueRstd.EnQue(rstdLocal);
            outQueueRstd.DeQue<float>();

            DataCopyExtParams copyInParamsRstd;
            copyInParamsRstd.blockCount = 1;
            copyInParamsRstd.blockLen = curUbFactor * sizeof(float);
            copyInParamsRstd.srcStride = 0;
            copyInParamsRstd.dstStride = 0;
            DataCopyPad(rstdGm[i * ubFactor], rstdLocal, copyInParamsRstd);

            LocalTensor<DX> yLocal = outQueueY.AllocTensor<DX>();
            ComputeY(xLocal, gammaLocal, rstdLocal, yLocal, curUbFactor);
            inQueueX.FreeTensor(xLocal);
            outQueueRstd.FreeTensor(rstdLocal);
            outQueueY.EnQue(yLocal);
            outQueueY.DeQue<DX>();

            DataCopyExtParams copyInParamsY;
            copyInParamsY.blockCount = curUbFactor;
            copyInParamsY.blockLen = numCol * sizeof(DX);
            copyInParamsY.srcStride = (numColAlign - numCol) * sizeof(DX) / blockSize;
            copyInParamsY.dstStride = 0;
            DataCopyPad(yGm[i * ubFactor * numCol], yLocal, copyInParamsY);
            outQueueY.FreeTensor(yLocal);
        }
        inQueueGamma.FreeTensor(gammaLocal);
    }

private:
    __aicore__ inline void ComputeY(LocalTensor<DX> xLocal, LocalTensor<DG> gammaLocal, LocalTensor<float> rstdLocal,
                                    LocalTensor<DX> yLocal, uint64_t curUbFactor)
    {
        __local_mem__ DX* xLocalAddr = (__local_mem__ DX*)xLocal.GetPhyAddr();
        __local_mem__ DG* gammaLocalUbAddr = (__local_mem__ DG*)gammaLocal.GetPhyAddr();
        __local_mem__ float* rstdLocalUbAddr = (__local_mem__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ DX* yLocalUbAddr = (__local_mem__ DX*)yLocal.GetPhyAddr();

        uint32_t colNum = static_cast<uint32_t>(numCol);
        uint16_t curAloops = static_cast<uint16_t>(curUbFactor);
        uint16_t colLoops = static_cast<uint16_t>((numColAlign + VectorLenB32 - 1) / VectorLenB32);
        uint32_t colNumAlign = static_cast<uint32_t>(numColAlign);
        __VEC_SCOPE__
        {
            RegTensor<float> RstdReg;
            RegTensor<float> gammaReg;

            RegTensor<float> xReg;
            RegTensor<float> mul1Reg;
            RegTensor<float> mul2Reg;

            for (uint16_t i = 0; i < curAloops; i++) {
                uint32_t sregElewiseNum = numCol;
                DataCopy<float, LoadDist::DIST_BRC_B32>(RstdReg, rstdLocalUbAddr + i);
                for (uint16_t j = 0; j < colLoops; j++) {
                    MaskReg pregCurLoop = UpdateMask<float>(sregElewiseNum);
                    LoadRegForDtype(xLocalAddr, xReg, pregCurLoop, (i * colNumAlign + j * VectorLenB32));
                    Mul(mul1Reg, xReg, RstdReg, pregCurLoop);
                    LoadRegForDtype(gammaLocalUbAddr, gammaReg, pregCurLoop, (j * VectorLenB32));
                    if (is_gemma) {
                        RegTensor<float> gammaAddReg;
                        Adds(gammaAddReg, gammaReg, 1.0f, pregCurLoop);
                        Mul(mul2Reg, mul1Reg, gammaAddReg, pregCurLoop);
                    } else {
                        Mul(mul2Reg, mul1Reg, gammaReg, pregCurLoop);
                    }
                    StoreRegForDtype(yLocalUbAddr, mul2Reg, pregCurLoop, (i * colNumAlign + j * VectorLenB32));
                }
            }
        }
    }

private:
    TPipe pipe;
    // QUE
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, 1> inQueueGamma;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueY;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueRstd;
    TBuf<TPosition::VECCALC> xTmpBuf;
    TBuf<TPosition::VECCALC> xReduceTmpBuf;

    // GM tensor
    GlobalTensor<DX> xGm;
    GlobalTensor<DG> gammaGm;
    GlobalTensor<DX> yGm;
    GlobalTensor<float> rstdGm;

    uint64_t curBlockIdx;
    uint64_t coreNum;
    uint64_t numRow;
    uint64_t numCol;
    uint64_t numColAlign;
    uint64_t blockFactor;
    uint64_t ubFactor;
    uint64_t colFlodFactor;
    uint64_t blockFactorTail;
    float epsilon;
    float avgFactor;
    uint64_t curBlockFactor;
    uint64_t curUbFactor;
    uint64_t curBlockLoops;
    uint64_t curUbTails;
    uint32_t blockSize = GetUbBlockSize();
    uint32_t blockSizeB32 = GetUbBlockSize() / sizeof(float);
    static constexpr uint32_t VectorLenB32 = GetVRegSize() / sizeof(float);
    static constexpr float RMS_POS_INF = 3.40282366920938E+38;
    static constexpr float RMS_ZERO = 0.0f;
    static constexpr int32_t NUM_ONE = 1;
    static constexpr int32_t NUM_TWO = 2;
    uint8_t is_gemma = 0;
};
} // namespace RmsNorm
#endif // OPS_BUILT_IN_TBE_IMPL_ASCENDC_RMS_NORM_REGBASE_PERF_H
