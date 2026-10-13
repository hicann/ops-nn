/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ada_layer_norm_grad_cut_n_arch35.h
 * \brief small-col kernels: cutN + batch-split (KernelCutNBS) / cutN + row-split (KernelCutNRow).
 *        Both share one implementation templated on the TILING_KEY tier (1xx / 2xx).
 */
#ifndef ADA_LAYER_NORM_GRAD_CUT_N_ARCH35_H
#define ADA_LAYER_NORM_GRAD_CUT_N_ARCH35_H

#include "ada_layer_norm_grad_common_arch35.h"
#include "ada_layer_norm_grad_determinstic_compute_arch35.h"

namespace AdaLayerNormGrad {
using namespace AscendC;
using namespace AscendC::Reg;

template <typename T, typename U, int TILING_KEY>
class KernelCutN {
public:
    __aicore__ inline KernelCutN() {}

    // b16 output present: scale/shift (T) or gamma/beta (U) is not fp32. Cross-core reduction outputs
    // (gamma/beta always; scale/shift in row-split) must accumulate in fp32 workspace and cast once at
    // the end, otherwise per-partial b16 rounding across cores breaks the precision budget.
    static constexpr bool B16_OUTPUT = !std::is_same_v<T, float> || !std::is_same_v<U, float>;
    static constexpr bool IS_BATCH_SPLIT = (TILING_KEY / 100) == 1;
    // scale/shift must go through the fp32 workspace when cross-core reduction is needed (row-split) or
    // when the output is b16: in batch-split b16 the per-row partial would otherwise be cast to b16 and
    // atomically added once per row, accumulating b16 rounding instead of a single end cast.
    static constexpr bool WS_SCALE_SHIFT = ((TILING_KEY / 100) == 2) || (IS_BATCH_SPLIT && B16_OUTPUT);

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR scale, GM_ADDR gamma,
                                GM_ADDR beta, GM_ADDR pd_x, GM_ADDR pd_scale, GM_ADDR pd_shift, GM_ADDR pd_gamma,
                                GM_ADDR pd_beta, const AdaLayerNormGradTilingDataArch35 tiling, GM_ADDR workspace)
    {
        selfTiling = tiling;
        isDeterministicKey = tiling.isDeterministicKey != 0;
        colAlignV = tiling.colAlignV;
        colAlignM = tiling.colAlignM;
        workspaceNum = (IS_BATCH_SPLIT && !B16_OUTPUT) ? 2 : (2 + 2 * tiling.batch);
        scaleRowsMax = (tiling.nAvailInUb + tiling.seq - 1) / tiling.seq + 1;

        myRows = (GetBlockIdx() != tiling.blockNum - 1) ? tiling.blockFormer : tiling.blockTail;
        myStartRow = GetBlockIdx() * tiling.blockFormer;
        isComputedCore = (GetBlockIdx() < tiling.blockNum);

        dyGm.SetGlobalBuffer((__gm__ T*)dy + static_cast<uint64_t>(myStartRow) * tiling.col,
                             static_cast<uint64_t>(myRows) * tiling.col);
        xGm.SetGlobalBuffer((__gm__ T*)x + static_cast<uint64_t>(myStartRow) * tiling.col,
                            static_cast<uint64_t>(myRows) * tiling.col);
        pdXGm.SetGlobalBuffer((__gm__ T*)pd_x + static_cast<uint64_t>(myStartRow) * tiling.col,
                              static_cast<uint64_t>(myRows) * tiling.col);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + myStartRow, myRows);
        meanGm.SetGlobalBuffer((__gm__ float*)mean + myStartRow, myRows);
        scaleGm.SetGlobalBuffer((__gm__ T*)scale, static_cast<uint64_t>(tiling.batch) * tiling.col);
        gammaGm.SetGlobalBuffer((__gm__ U*)gamma, tiling.col);
        betaGm.SetGlobalBuffer((__gm__ U*)beta, tiling.col);
        pdScaleGm.SetGlobalBuffer((__gm__ T*)pd_scale, static_cast<uint64_t>(tiling.batch) * tiling.col);
        pdShiftGm.SetGlobalBuffer((__gm__ T*)pd_shift, static_cast<uint64_t>(tiling.batch) * tiling.col);
        pdGammaGm.SetGlobalBuffer((__gm__ U*)pd_gamma, tiling.col);
        pdBetaGm.SetGlobalBuffer((__gm__ U*)pd_beta, tiling.col);

        if (isDeterministicKey || B16_OUTPUT) {
            deterministicWorkSpaceSize = static_cast<uint64_t>(workspaceNum) * colAlignV * tiling.blockNum;
            workspaceGMOri.SetGlobalBuffer((__gm__ float*)workspace, deterministicWorkSpaceSize);
            workspaceGmGamma.SetGlobalBuffer((__gm__ float*)workspace + GetBlockIdx() * workspaceNum * colAlignV,
                                             colAlignV);
            workspaceGmBeta.SetGlobalBuffer((__gm__ float*)workspace + (GetBlockIdx() * workspaceNum + 1) * colAlignV,
                                            colAlignV);
        }

        InitQueue();
        InitOutputQueue();

        InitGmAccumulator();
        if (isComputedCore) {
            InitInputQueue();
            InitAccumulatorBuffer();
        }
        SyncAll();
    }

    __aicore__ inline void Process()
    {
        if (isComputedCore) {
            CopyInGammaBeta();
            LocalTensor<U> inputGamma = gammaQue.DeQue<U>();
            LocalTensor<U> inputBeta = betaQue.DeQue<U>();
            int64_t ubLoopCount = (GetBlockIdx() != selfTiling.blockNum - 1) ? selfTiling.ubLoopOfFormerBlock :
                                                                               selfTiling.ubLoopOfTailBlock;
            int64_t ubTailRows = (GetBlockIdx() != selfTiling.blockNum - 1) ? selfTiling.ubTailOfFormerBlock :
                                                                              selfTiling.ubTailOfTailBlock;
            for (int64_t i = 0; i < ubLoopCount; ++i) {
                int64_t nInOnceUb = (i != ubLoopCount - 1) ? selfTiling.nAvailInUb : ubTailRows;
                int64_t startRow = myStartRow + i * selfTiling.nAvailInUb;
                CopyInTile(startRow, nInOnceUb);
                ComputeTile(startRow, nInOnceUb, inputGamma, inputBeta);
                FlushPdX(i, nInOnceUb);
                FlushScaleShift(startRow, nInOnceUb);
            }
            gammaQue.FreeTensor(inputGamma);
            betaQue.FreeTensor(inputBeta);
            FlushGammaBeta();
        }
        if (isDeterministicKey || B16_OUTPUT) {
            SyncAll();
            pipe.Reset();
            AdaLayerNormGradDeterminsticCompute<T, U> op;
            op.initBuffer(pipe, pdGammaGm, pdBetaGm, pdScaleGm, pdShiftGm, workspaceGMOri, workspaceNum,
                          WS_SCALE_SHIFT);
            op.FinalProcessDeterministic(colAlignV, selfTiling.blockNum, selfTiling.col);
            if (WS_SCALE_SHIFT) {
                op.BatchFinalProcessDeterministic(selfTiling.batch, colAlignV, selfTiling.blockNum, selfTiling.col);
            }
            op.FinalProcessFree();
        }
    }

private:
    __aicore__ inline void InitQueue()
    {
        pipe.InitBuffer(gammaQue, BUFFER_NUM, colAlignM * sizeof(U));
        pipe.InitBuffer(betaQue, BUFFER_NUM, colAlignM * sizeof(U));
        pipe.InitBuffer(pdGammaBuf, colAlignV * sizeof(float));
        pipe.InitBuffer(pdBetaBuf, colAlignV * sizeof(float));
        pipe.InitBuffer(dyGStageBuf, colAlignV * sizeof(float));
        pipe.InitBuffer(yStageBuf, colAlignV * sizeof(float));
        pipe.InitBuffer(tmpBuf, V_LENGTH * sizeof(float) * 2);
        pipe.InitBuffer(scaleOutTmp, colAlignM * sizeof(T));
        pipe.InitBuffer(shiftOutTmp, colAlignM * sizeof(T));
        pipe.InitBuffer(gammaOutTmp, colAlignM * sizeof(U));
    }

    __aicore__ inline void InitOutputQueue()
    {
        pipe.InitBuffer(pdXQue, BUFFER_NUM, selfTiling.nAvailInUb * colAlignM * sizeof(T));
        pipe.InitBuffer(dScaleBuf, scaleRowsMax * colAlignV * sizeof(float));
        pipe.InitBuffer(dShiftBuf, scaleRowsMax * colAlignV * sizeof(float));
    }

    __aicore__ inline void InitInputQueue()
    {
        pipe.InitBuffer(dyQue, BUFFER_NUM, selfTiling.nAvailInUb * colAlignM * sizeof(T));
        pipe.InitBuffer(xQue, BUFFER_NUM, selfTiling.nAvailInUb * colAlignM * sizeof(T));
        pipe.InitBuffer(rstdQue, BUFFER_NUM, selfTiling.nAvailInUb * roundUp1Dtype * sizeof(float));
        pipe.InitBuffer(meanQue, BUFFER_NUM, selfTiling.nAvailInUb * roundUp1Dtype * sizeof(float));
        pipe.InitBuffer(scaleQue, BUFFER_NUM, scaleRowsMax * colAlignM * sizeof(T));
    }

    __aicore__ inline void InitAccumulatorBuffer()
    {
        LocalTensor<float> dGammaLocal = pdGammaBuf.Get<float>();
        LocalTensor<float> dBetaLocal = pdBetaBuf.Get<float>();
        Duplicate(dGammaLocal, 0.0f, colAlignV);
        Duplicate(dBetaLocal, 0.0f, colAlignV);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitGmAccumulator()
    {
        if (isDeterministicKey || B16_OUTPUT) {
            LocalTensor<float> tempLocal = pdGammaBuf.Get<float>();
            InitGmData(workspaceGmGamma, workspaceGmBeta, colAlignV, tempLocal, colAlignV);
            if (WS_SCALE_SHIFT) {
                int64_t coreBase = GetBlockIdx() * workspaceNum * colAlignV;
                for (int64_t i = 0; i < selfTiling.batch; ++i) {
                    InitGmData(workspaceGMOri[coreBase + (2 + 2 * i) * colAlignV],
                               workspaceGMOri[coreBase + (3 + 2 * i) * colAlignV], colAlignV, tempLocal, colAlignV);
                }
            } else if (GetBlockIdx() == 0) {
                // batch-split: pd_scale/pd_shift accumulate on GM via in-core atomic adds -> zero once first
                LocalTensor<T> tempLocalT = pdXQue.AllocTensor<T>();
                InitSingleData<T>(pdScaleGm, tempLocalT, selfTiling.batch * selfTiling.col);
                InitSingleData<T>(pdShiftGm, tempLocalT, selfTiling.batch * selfTiling.col);
                pdXQue.FreeTensor(tempLocalT);
            }
        } else if (GetBlockIdx() == 0) {
            LocalTensor<U> tempLocalU = gammaOutTmp.Get<U>();
            InitSingleData<U>(pdGammaGm, tempLocalU, selfTiling.col);
            InitSingleData<U>(pdBetaGm, tempLocalU, selfTiling.col);
            LocalTensor<T> tempLocalT = pdXQue.AllocTensor<T>();
            InitSingleData<T>(pdScaleGm, tempLocalT, selfTiling.batch * selfTiling.col);
            InitSingleData<T>(pdShiftGm, tempLocalT, selfTiling.batch * selfTiling.col);
            pdXQue.FreeTensor(tempLocalT);
        }
    }

    __aicore__ inline void CopyInGammaBeta()
    {
        LocalTensor<U> gammaLocal = gammaQue.AllocTensor<U>();
        LocalTensor<U> betaLocal = betaQue.AllocTensor<U>();
        // rightPadding fills only the trailing 32B block and must be < block elems of the loaded dtype:
        // colAlignM is aligned for T, so for a narrower U (e.g. fp32 gamma under b16 dy) derive the pad
        // from the U block alignment instead (otherwise pad >= 8 corrupts the MTE2 load).
        DataCopyParams copyParams{1, (uint16_t)(selfTiling.col * sizeof(U)), 0, 0};
        int64_t colAlignU = ROUND_UP(selfTiling.col, BLOCK_ALIGN_B / sizeof(U));
        DataCopyPadParams padParams{true, 0, (uint8_t)(colAlignU - selfTiling.col), 0};
        DataCopyPad(gammaLocal, gammaGm[0], copyParams, padParams);
        DataCopyPad(betaLocal, betaGm[0], copyParams, padParams);
        gammaQue.EnQue(gammaLocal);
        betaQue.EnQue(betaLocal);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyInTile(const int64_t startRow, const int64_t nInOnceUb)
    {
        LocalTensor<T> dyLocal = dyQue.AllocTensor<T>();
        LocalTensor<T> xLocal = xQue.AllocTensor<T>();
        LocalTensor<float> rstdLocal = rstdQue.AllocTensor<float>();
        LocalTensor<float> meanLocal = meanQue.AllocTensor<float>();
        LocalTensor<T> scaleLocal = scaleQue.AllocTensor<T>();
        DataCopyParams rowParams{1, (uint16_t)(selfTiling.col * sizeof(T)), 0, 0};
        DataCopyPadParams rowPadParams{true, 0, (uint8_t)(colAlignM - selfTiling.col), 0};
        DataCopyParams scalarParams{1, (uint16_t)sizeof(float), 0, 0};
        DataCopyPadParams scalarPadParams{true, 0, (uint8_t)(roundUp1Dtype - 1), 0};

        for (int64_t rowIdx = 0; rowIdx < nInOnceUb; ++rowIdx) {
            uint64_t gmRow = static_cast<uint64_t>(startRow - myStartRow) + rowIdx;
            DataCopyPad(dyLocal[rowIdx * colAlignM], dyGm[gmRow * selfTiling.col], rowParams, rowPadParams);
            DataCopyPad(xLocal[rowIdx * colAlignM], xGm[gmRow * selfTiling.col], rowParams, rowPadParams);
            DataCopyPad(rstdLocal[rowIdx * roundUp1Dtype], rstdGm[gmRow], scalarParams, scalarPadParams);
            DataCopyPad(meanLocal[rowIdx * roundUp1Dtype], meanGm[gmRow], scalarParams, scalarPadParams);
        }
        int64_t startBatch = startRow / selfTiling.seq;
        int64_t endBatch = (startRow + nInOnceUb - 1) / selfTiling.seq;
        int64_t scaleRows = endBatch - startBatch + 1;
        for (int64_t i = 0; i < scaleRows; ++i) {
            DataCopyPad(scaleLocal[i * colAlignM], scaleGm[(startBatch + i) * selfTiling.col], rowParams, rowPadParams);
        }

        dyQue.EnQue(dyLocal);
        xQue.EnQue(xLocal);
        rstdQue.EnQue(rstdLocal);
        meanQue.EnQue(meanLocal);
        scaleQue.EnQue(scaleLocal);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ComputeTile(const int64_t startRow, const int64_t nInOnceUb,
                                       const LocalTensor<U>& inputGamma, const LocalTensor<U>& inputBeta)
    {
        LocalTensor<T> inputDy = dyQue.DeQue<T>();
        LocalTensor<T> inputX = xQue.DeQue<T>();
        LocalTensor<float> inputRstd = rstdQue.DeQue<float>();
        LocalTensor<float> inputMean = meanQue.DeQue<float>();
        LocalTensor<T> inputScale = scaleQue.DeQue<T>();
        LocalTensor<T> outputPdX = pdXQue.AllocTensor<T>();

        __ubuf__ T* dyTAddr = (__ubuf__ T*)inputDy.GetPhyAddr();
        __ubuf__ T* xTAddr = (__ubuf__ T*)inputX.GetPhyAddr();
        __ubuf__ float* rstdAddr = (__ubuf__ float*)inputRstd.GetPhyAddr();
        __ubuf__ float* meanAddr = (__ubuf__ float*)inputMean.GetPhyAddr();
        __ubuf__ T* scaleTAddr = (__ubuf__ T*)inputScale.GetPhyAddr();
        __ubuf__ U* gammaTAddr = (__ubuf__ U*)inputGamma.GetPhyAddr();
        __ubuf__ U* betaTAddr = (__ubuf__ U*)inputBeta.GetPhyAddr();
        __ubuf__ T* pdXTAddr = (__ubuf__ T*)outputPdX.GetPhyAddr();

        LocalTensor<float> dGammaLocal = pdGammaBuf.Get<float>();
        LocalTensor<float> dBetaLocal = pdBetaBuf.Get<float>();
        LocalTensor<float> dScaleLocal = dScaleBuf.Get<float>();
        LocalTensor<float> dShiftLocal = dShiftBuf.Get<float>();
        LocalTensor<float> dyGStageLocal = dyGStageBuf.Get<float>();
        LocalTensor<float> yStageLocal = yStageBuf.Get<float>();
        LocalTensor<float> tmpLocal = tmpBuf.Get<float>();

        __ubuf__ float* dGammaAddr = (__ubuf__ float*)dGammaLocal.GetPhyAddr();
        __ubuf__ float* dBetaAddr = (__ubuf__ float*)dBetaLocal.GetPhyAddr();
        __ubuf__ float* dScaleAddr = (__ubuf__ float*)dScaleLocal.GetPhyAddr();
        __ubuf__ float* dShiftAddr = (__ubuf__ float*)dShiftLocal.GetPhyAddr();
        __ubuf__ float* dyGStageAddr = (__ubuf__ float*)dyGStageLocal.GetPhyAddr();
        __ubuf__ float* yStageAddr = (__ubuf__ float*)yStageLocal.GetPhyAddr();
        __ubuf__ float* tmpAddr = (__ubuf__ float*)tmpLocal.GetPhyAddr();

        int64_t startBatch = startRow / selfTiling.seq;
        int64_t endBatch = (startRow + nInOnceUb - 1) / selfTiling.seq;
        int64_t scaleRows = endBatch - startBatch + 1;
        Duplicate(dScaleLocal, 0.0f, scaleRows * colAlignV);
        Duplicate(dShiftLocal, 0.0f, scaleRows * colAlignV);
        PipeBarrier<PIPE_V>();

        uint16_t colLoopTimes = static_cast<uint16_t>((selfTiling.col + V_LENGTH - 1) / V_LENGTH);
        int64_t numLastDim = selfTiling.col;
        int64_t roundUpNumLastDim = colAlignM;
        float coff = (numLastDim != 0) ? 1.0f / numLastDim : 0.0f;

        for (int64_t rowIdx = 0; rowIdx < nInOnceUb; ++rowIdx) {
            int64_t batchLocal = ((startRow + rowIdx) / selfTiling.seq) - startBatch;
            uint32_t rowTBase = rowIdx * colAlignM;
            uint32_t rowFBase = rowIdx * roundUp1Dtype;

            __VEC_SCOPE__
            {
                RegTensor<float> meanReg;
                RegTensor<float> rstdReg;
                RegTensor<float> dy;
                RegTensor<float> x;
                RegTensor<float> gamma;
                RegTensor<float> beta;
                RegTensor<float> scale;
                RegTensor<float> scaleP1;
                RegTensor<float> dyScaleP1;
                RegTensor<float> pdBetaAcc;
                RegTensor<float> xCentered;
                RegTensor<float> y;
                RegTensor<float> dyGY;
                RegTensor<float> pdGammaAcc;
                RegTensor<float> yGamma;
                RegTensor<float> yGammaBeta;
                RegTensor<float> dyYGB;
                RegTensor<float> pdScaleAcc;
                RegTensor<float> pdShiftAcc;
                RegTensor<float> dyG;

                MaskReg pregLoop;
                DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg, meanAddr + rowFBase);
                DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + rowFBase);

                for (uint16_t j = 0; j < colLoopTimes; ++j) {
                    uint32_t remainElem = numLastDim - j * V_LENGTH;
                    pregLoop = UpdateMask<float>(remainElem);
                    uint32_t ubOffsetT = rowTBase + j * V_LENGTH;

                    LoadTensor(dy, dyTAddr + ubOffsetT, pregLoop);
                    LoadTensor(x, xTAddr + ubOffsetT, pregLoop);
                    LoadTensor(gamma, gammaTAddr + j * V_LENGTH, pregLoop);
                    LoadTensor(beta, betaTAddr + j * V_LENGTH, pregLoop);
                    LoadTensor(scale, scaleTAddr + batchLocal * colAlignM + j * V_LENGTH, pregLoop);

                    Adds(scaleP1, scale, 1.0f, pregLoop);
                    Mul(dyScaleP1, dy, scaleP1, pregLoop);

                    LoadTensor(pdBetaAcc, dBetaAddr + j * V_LENGTH, pregLoop);
                    Add(pdBetaAcc, pdBetaAcc, dyScaleP1, pregLoop);
                    CopyToTensor(dBetaAddr + j * V_LENGTH, pdBetaAcc, pregLoop);

                    Sub(xCentered, x, meanReg, pregLoop);
                    Mul(y, xCentered, rstdReg, pregLoop);

                    Mul(dyGY, dyScaleP1, y, pregLoop);
                    LoadTensor(pdGammaAcc, dGammaAddr + j * V_LENGTH, pregLoop);
                    Add(pdGammaAcc, pdGammaAcc, dyGY, pregLoop);
                    CopyToTensor(dGammaAddr + j * V_LENGTH, pdGammaAcc, pregLoop);

                    Mul(yGamma, y, gamma, pregLoop);
                    Add(yGammaBeta, yGamma, beta, pregLoop);
                    Mul(dyYGB, dy, yGammaBeta, pregLoop);
                    LoadTensor(pdScaleAcc, dScaleAddr + batchLocal * colAlignV + j * V_LENGTH, pregLoop);
                    Add(pdScaleAcc, pdScaleAcc, dyYGB, pregLoop);
                    CopyToTensor(dScaleAddr + batchLocal * colAlignV + j * V_LENGTH, pdScaleAcc, pregLoop);

                    LoadTensor(pdShiftAcc, dShiftAddr + batchLocal * colAlignV + j * V_LENGTH, pregLoop);
                    Add(pdShiftAcc, pdShiftAcc, dy, pregLoop);
                    CopyToTensor(dShiftAddr + batchLocal * colAlignV + j * V_LENGTH, pdShiftAcc, pregLoop);

                    Mul(dyG, dyScaleP1, gamma, pregLoop);
                    CopyToTensor(dyGStageAddr + j * V_LENGTH, dyG, pregLoop);
                    CopyToTensor(yStageAddr + j * V_LENGTH, y, pregLoop);
                }
            }

            PipeBarrier<PIPE_V>();
            AscendC::Mul(yStageLocal, dyGStageLocal, yStageLocal, numLastDim);
            PipeBarrier<PIPE_V>();
            float reduce3 = ReduceSumCustom(yStageLocal, numLastDim);
            float reduce2 = ReduceSumCustom(dyGStageLocal, numLastDim);
            reduce2 = reduce2 * coff;
            reduce3 = reduce3 * coff;

            __VEC_SCOPE__
            {
                RegTensor<float> meanReg;
                RegTensor<float> rstdReg;
                RegTensor<float> reduce2Reg;
                RegTensor<float> reduce3Reg;
                RegTensor<float> dy;
                RegTensor<float> x;
                RegTensor<float> gamma;
                RegTensor<float> scale;
                RegTensor<float> scaleP1;
                RegTensor<float> dyG;
                RegTensor<float> xCentered;
                RegTensor<float> y;
                RegTensor<float> pdX;
                RegTensor<float> tmp;

                MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
                MaskReg pregLoop;
                DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg, meanAddr + rowFBase);
                DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + rowFBase);
                Duplicate(reduce2Reg, reduce2, pregFull);
                Duplicate(reduce3Reg, reduce3, pregFull);

                for (uint16_t j = 0; j < colLoopTimes; ++j) {
                    uint32_t remainElem = numLastDim - j * V_LENGTH;
                    pregLoop = UpdateMask<float>(remainElem);
                    uint32_t ubOffsetT = rowTBase + j * V_LENGTH;

                    LoadTensor(dy, dyTAddr + ubOffsetT, pregLoop);
                    LoadTensor(x, xTAddr + ubOffsetT, pregLoop);
                    LoadTensor(gamma, gammaTAddr + j * V_LENGTH, pregLoop);
                    LoadTensor(scale, scaleTAddr + batchLocal * colAlignM + j * V_LENGTH, pregLoop);

                    Adds(scaleP1, scale, 1.0f, pregLoop);
                    Mul(dyG, dy, scaleP1, pregLoop);
                    Mul(dyG, dyG, gamma, pregLoop);
                    Sub(xCentered, x, meanReg, pregLoop);
                    Mul(y, xCentered, rstdReg, pregLoop);
                    Sub(pdX, dyG, reduce2Reg, pregLoop);
                    Mul(tmp, y, reduce3Reg, pregLoop);
                    Sub(pdX, pdX, tmp, pregLoop);
                    Mul(pdX, pdX, rstdReg, pregLoop);
                    CopyToTensor(pdXTAddr + ubOffsetT, pdX, pregLoop);
                }
            }
            PipeBarrier<PIPE_V>();
        }

        dyQue.FreeTensor(inputDy);
        xQue.FreeTensor(inputX);
        rstdQue.FreeTensor(inputRstd);
        meanQue.FreeTensor(inputMean);
        scaleQue.FreeTensor(inputScale);
        pdXQue.EnQue(outputPdX);
    }

    __aicore__ inline void FlushPdX(const int64_t outerIdx, const int64_t nInOnceUb)
    {
        LocalTensor<T> outputPdX = pdXQue.DeQue<T>();
        PipeBarrier<PIPE_ALL>();
        DataCopyCustom<T>(pdXGm, outputPdX, selfTiling.col, outerIdx * selfTiling.nAvailInUb * selfTiling.col,
                          std::is_same_v<T, float>, (uint16_t)nInOnceUb);
        pdXQue.FreeTensor(outputPdX);
    }

    __aicore__ inline void FlushScaleShift(const int64_t startRow, const int64_t nInOnceUb)
    {
        LocalTensor<float> dScaleLocal = dScaleBuf.Get<float>();
        LocalTensor<float> dShiftLocal = dShiftBuf.Get<float>();
        int64_t startBatch = startRow / selfTiling.seq;
        int64_t endBatch = (startRow + nInOnceUb - 1) / selfTiling.seq;
        int64_t scaleRows = endBatch - startBatch + 1;

        event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event);
        WaitFlag<HardEvent::V_MTE3>(event);

        if (WS_SCALE_SHIFT && (isDeterministicKey || B16_OUTPUT)) {
            int64_t coreBase = GetBlockIdx() * workspaceNum * colAlignV;
            for (int64_t i = 0; i < scaleRows; ++i) {
                SetAtomicAdd<float>();
                DataCopyAtomicAdd(workspaceGMOri[coreBase + (2 + 2 * (startBatch + i)) * colAlignV],
                                  dScaleLocal[i * colAlignV], colAlignV, 0, (uint16_t)1);
                DataCopyAtomicAdd(workspaceGMOri[coreBase + (3 + 2 * (startBatch + i)) * colAlignV],
                                  dShiftLocal[i * colAlignV], colAlignV, 0, (uint16_t)1);
                SetAtomicNone();
            }
        } else {
            for (int64_t i = 0; i < scaleRows; ++i) {
                if constexpr (std::is_same_v<T, float>) {
                    SetAtomicAdd<float>();
                    DataCopyPad(pdScaleGm[(startBatch + i) * selfTiling.col], dScaleLocal[i * colAlignV],
                                DataCopyParams{1, (uint16_t)(selfTiling.col * sizeof(T)), 0, 0});
                    DataCopyPad(pdShiftGm[(startBatch + i) * selfTiling.col], dShiftLocal[i * colAlignV],
                                DataCopyParams{1, (uint16_t)(selfTiling.col * sizeof(T)), 0, 0});
                    SetAtomicNone();
                } else {
                    LocalTensor<T> scaleOut = scaleOutTmp.Get<T>();
                    LocalTensor<T> shiftOut = shiftOutTmp.Get<T>();
                    Cast(scaleOut, dScaleLocal[i * colAlignV], RoundMode::CAST_RINT, colAlignV);
                    PipeBarrier<PIPE_V>();
                    SetAtomicAdd<T>();
                    DataCopyPad(pdScaleGm[(startBatch + i) * selfTiling.col], scaleOut,
                                DataCopyParams{1, (uint16_t)(selfTiling.col * sizeof(T)), 0, 0});
                    SetAtomicNone();
                    Cast(shiftOut, dShiftLocal[i * colAlignV], RoundMode::CAST_RINT, colAlignV);
                    PipeBarrier<PIPE_V>();
                    SetAtomicAdd<T>();
                    DataCopyPad(pdShiftGm[(startBatch + i) * selfTiling.col], shiftOut,
                                DataCopyParams{1, (uint16_t)(selfTiling.col * sizeof(T)), 0, 0});
                    SetAtomicNone();
                }
            }
        }
    }

    __aicore__ inline void FlushGammaBeta()
    {
        LocalTensor<float> dGammaLocal = pdGammaBuf.Get<float>();
        LocalTensor<float> dBetaLocal = pdBetaBuf.Get<float>();
        event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event);
        WaitFlag<HardEvent::V_MTE3>(event);

        if (isDeterministicKey || B16_OUTPUT) {
            DataCopyPad(workspaceGmGamma, dGammaLocal, DataCopyParams{1, (uint16_t)(colAlignV * sizeof(float)), 0, 0});
            DataCopyPad(workspaceGmBeta, dBetaLocal, DataCopyParams{1, (uint16_t)(colAlignV * sizeof(float)), 0, 0});
            PipeBarrier<PIPE_ALL>();
        } else {
            if constexpr (std::is_same_v<U, float>) {
                SetAtomicAdd<float>();
                DataCopyPad(pdGammaGm, dGammaLocal,
                            DataCopyParams{1, (uint16_t)(selfTiling.col * sizeof(float)), 0, 0});
                DataCopyPad(pdBetaGm, dBetaLocal, DataCopyParams{1, (uint16_t)(selfTiling.col * sizeof(float)), 0, 0});
                SetAtomicNone();
            } else {
                LocalTensor<U> castTmp = gammaOutTmp.Get<U>();
                Cast(castTmp, dGammaLocal, RoundMode::CAST_RINT, colAlignV);
                PipeBarrier<PIPE_V>();
                SetAtomicAdd<U>();
                DataCopyPad(pdGammaGm, castTmp, DataCopyParams{1, (uint16_t)(selfTiling.col * sizeof(U)), 0, 0});
                SetAtomicNone();
                PipeBarrier<PIPE_MTE3>();
                Cast(castTmp, dBetaLocal, RoundMode::CAST_RINT, colAlignV);
                PipeBarrier<PIPE_V>();
                SetAtomicAdd<U>();
                DataCopyPad(pdBetaGm, castTmp, DataCopyParams{1, (uint16_t)(selfTiling.col * sizeof(U)), 0, 0});
                SetAtomicNone();
            }
        }
    }

private:
    TPipe pipe;
    AdaLayerNormGradTilingDataArch35 selfTiling;
    TQue<QuePosition::VECIN, BUFFER_NUM> dyQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> rstdQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> meanQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> scaleQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> gammaQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> betaQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> pdXQue;
    TBuf<TPosition::VECCALC> pdGammaBuf;
    TBuf<TPosition::VECCALC> pdBetaBuf;
    TBuf<TPosition::VECCALC> dScaleBuf;
    TBuf<TPosition::VECCALC> dShiftBuf;
    TBuf<TPosition::VECCALC> dyGStageBuf;
    TBuf<TPosition::VECCALC> yStageBuf;
    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> scaleOutTmp;
    TBuf<TPosition::VECCALC> shiftOutTmp;
    TBuf<TPosition::VECCALC> gammaOutTmp;

    GlobalTensor<T> dyGm;
    GlobalTensor<T> xGm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<float> meanGm;
    GlobalTensor<T> scaleGm;
    GlobalTensor<U> gammaGm;
    GlobalTensor<U> betaGm;
    GlobalTensor<T> pdXGm;
    GlobalTensor<T> pdScaleGm;
    GlobalTensor<T> pdShiftGm;
    GlobalTensor<U> pdGammaGm;
    GlobalTensor<U> pdBetaGm;
    GlobalTensor<float> workspaceGMOri;
    GlobalTensor<float> workspaceGmGamma;
    GlobalTensor<float> workspaceGmBeta;

    int64_t colAlignV = 0;
    int64_t colAlignM = 0;
    int64_t workspaceNum = 0;
    int64_t scaleRowsMax = 0;
    uint32_t roundUp1Dtype = BLOCK_ALIGN_B / sizeof(float);
    int64_t myRows = 0;
    int64_t myStartRow = 0;
    uint64_t deterministicWorkSpaceSize = 0;
    bool isComputedCore = false;
    bool isDeterministicKey = false;
};

} // namespace AdaLayerNormGrad

#endif // ADA_LAYER_NORM_GRAD_CUT_N_ARCH35_H
