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
 * \file ada_layer_norm_grad_cut_d_arch35.h
 * \brief large-col kernels: cutD + batch-split (KernelCutDBS) / cutD + row-split (KernelCutDRow).
 *        Two-pass per row: Pass1 accumulates reduce2/reduce3 scalars and flushes the four
 *        reduction-output chunk contributions; Pass2 recomputes and writes pd_x.
 */
#ifndef ADA_LAYER_NORM_GRAD_CUT_D_ARCH35_H
#define ADA_LAYER_NORM_GRAD_CUT_D_ARCH35_H

#include "ada_layer_norm_grad_common_arch35.h"
#include "ada_layer_norm_grad_determinstic_compute_arch35.h"

namespace AdaLayerNormGrad {
using namespace AscendC;
using namespace AscendC::Reg;

template <typename T, typename U, int TILING_KEY>
class KernelCutD {
public:
    __aicore__ inline KernelCutD() {}

    // b16 output present: scale/shift (T) or gamma/beta (U) is not fp32. Cross-core reduction outputs
    // (gamma/beta always; scale/shift in row-split) must accumulate in fp32 workspace and cast once at
    // the end, otherwise per-partial b16 rounding across cores breaks the precision budget.
    static constexpr bool B16_OUTPUT = !std::is_same_v<T, float> || !std::is_same_v<U, float>;
    static constexpr bool IS_BATCH_SPLIT = (TILING_KEY / 100) == 3;
    // scale/shift must go through the fp32 workspace when cross-core reduction is needed (row-split) or
    // when the output is b16: in batch-split b16 the per-chunk partial would otherwise be cast to b16 and
    // atomically added once per row, accumulating b16 rounding instead of a single end cast.
    static constexpr bool WS_SCALE_SHIFT = ((TILING_KEY / 100) == 4) || (IS_BATCH_SPLIT && B16_OUTPUT);

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR mean, GM_ADDR scale, GM_ADDR gamma,
                                GM_ADDR beta, GM_ADDR pd_x, GM_ADDR pd_scale, GM_ADDR pd_shift, GM_ADDR pd_gamma,
                                GM_ADDR pd_beta, const AdaLayerNormGradTilingDataArch35 tiling, GM_ADDR workspace)
    {
        selfTiling = tiling;
        isDeterministicKey = tiling.isDeterministicKey != 0;
        colAlignV = tiling.colAlignV;
        colAlignM = tiling.colAlignM;
        workspaceNum = (IS_BATCH_SPLIT && !B16_OUTPUT) ? 2 : (2 + 2 * tiling.batch);

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
        if (isComputedCore) {
            InitInputQueue();
        }
        InitGmAccumulator();
        SyncAll();
    }

    __aicore__ inline void Process()
    {
        if (isComputedCore) {
            int64_t dInnerLengthChunk = selfTiling.dInnerLength;
            int64_t dOuterLength = selfTiling.dOuterLength;
            float coff = (selfTiling.col != 0) ? 1.0f / selfTiling.col : 0.0f;

            for (int64_t rowIndex = 0; rowIndex < myRows; ++rowIndex) {
                int64_t batch = (myStartRow + rowIndex) / selfTiling.seq;
                LoadRowScalars(rowIndex);
                LocalTensor<float> inputRstd = rstdQue.DeQue<float>();
                LocalTensor<float> inputMean = meanQue.DeQue<float>();

                LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
                tmpLocal.SetValue(0, 0.0f);
                tmpLocal.SetValue(1, 0.0f);
                PipeBarrier<PIPE_V>();

                // Pass 1: accumulate reduce2/reduce3 partials and flush the four chunk accumulators
                for (int64_t dOuterUbIndex = 0; dOuterUbIndex < dOuterLength; ++dOuterUbIndex) {
                    int64_t dInOnceUb = (dOuterUbIndex != dOuterLength - 1) ? dInnerLengthChunk :
                                                                              selfTiling.dInnerLengthTail;
                    int64_t offsetUbXY = rowIndex * selfTiling.col + dOuterUbIndex * dInnerLengthChunk;
                    int64_t offsetUbGamma = dOuterUbIndex * dInnerLengthChunk;
                    CopyInChunk(offsetUbXY, offsetUbGamma, batch, dInOnceUb, true);
                    ComputePass1Chunk(offsetUbGamma, dInOnceUb, inputRstd, inputMean);
                    FlushChunkAcc(offsetUbGamma, batch, dInOnceUb);
                }

                float reduce2Partial = tmpLocal.GetValue(0);
                float reduce3Partial = tmpLocal.GetValue(1);
                float reduce2 = reduce2Partial * coff;
                float reduce3 = reduce3Partial * coff;
                tmpLocal.SetValue(0, reduce2);
                tmpLocal.SetValue(1, reduce3);
                PipeBarrier<PIPE_V>();

                // Pass 2: recompute dy_g / y and write pd_x
                for (int64_t dOuterUbIndex = 0; dOuterUbIndex < dOuterLength; ++dOuterUbIndex) {
                    int64_t dInOnceUb = (dOuterUbIndex != dOuterLength - 1) ? dInnerLengthChunk :
                                                                              selfTiling.dInnerLengthTail;
                    int64_t offsetUbXY = rowIndex * selfTiling.col + dOuterUbIndex * dInnerLengthChunk;
                    int64_t offsetUbGamma = dOuterUbIndex * dInnerLengthChunk;
                    CopyInChunk(offsetUbXY, offsetUbGamma, batch, dInOnceUb, false);
                    ComputePass2Chunk(rowIndex, offsetUbXY, dInOnceUb, inputRstd, inputMean);
                }
                rstdQue.FreeTensor(inputRstd);
                meanQue.FreeTensor(inputMean);
            }
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
        pipe.InitBuffer(pdGammaBuf, ROUND_UP(selfTiling.dInnerLength, FLOAT_BLOCK_ELEM) * sizeof(float));
        pipe.InitBuffer(pdBetaBuf, ROUND_UP(selfTiling.dInnerLength, FLOAT_BLOCK_ELEM) * sizeof(float));
        pipe.InitBuffer(dScaleBuf, ROUND_UP(selfTiling.dInnerLength, FLOAT_BLOCK_ELEM) * sizeof(float));
        pipe.InitBuffer(dShiftBuf, ROUND_UP(selfTiling.dInnerLength, FLOAT_BLOCK_ELEM) * sizeof(float));
        pipe.InitBuffer(dyGStageBuf, ROUND_UP(selfTiling.dInnerLength, FLOAT_BLOCK_ELEM) * sizeof(float));
        pipe.InitBuffer(yStageBuf, ROUND_UP(selfTiling.dInnerLength, FLOAT_BLOCK_ELEM) * sizeof(float));
        pipe.InitBuffer(tmpBuf, V_LENGTH * sizeof(float) * 2 + 2 * sizeof(float));
        // cutD serves unbounded col; these cast-out temp buffers only ever hold dInOnceUb (<= dInnerLength)
        // elements (FlushChunkAcc / FlushScaleShiftGm all size their Cast/DataCopyPad by dInOnceUb), so they
        // must be sized by dInnerLength, not colAlignM (colAlignM has no upper bound in cutD and would
        // overflow UB). Sized to the per-dtype block alignment like the input/output queues.
        pipe.InitBuffer(scaleOutTmp, ROUND_UP(selfTiling.dInnerLength, blockNumTdtype) * sizeof(T));
        pipe.InitBuffer(shiftOutTmp, ROUND_UP(selfTiling.dInnerLength, blockNumTdtype) * sizeof(T));
        pipe.InitBuffer(gammaOutTmp, ROUND_UP(selfTiling.dInnerLength, blockNumUdtype) * sizeof(U));
    }

    __aicore__ inline void InitOutputQueue()
    {
        pipe.InitBuffer(pdXQue, BUFFER_NUM, ROUND_UP(selfTiling.dInnerLength, blockNumTdtype) * sizeof(T));
    }

    __aicore__ inline void InitInputQueue()
    {
        pipe.InitBuffer(dyQue, BUFFER_NUM, ROUND_UP(selfTiling.dInnerLength, blockNumTdtype) * sizeof(T));
        pipe.InitBuffer(xQue, BUFFER_NUM, ROUND_UP(selfTiling.dInnerLength, blockNumTdtype) * sizeof(T));
        pipe.InitBuffer(scaleQue, BUFFER_NUM, ROUND_UP(selfTiling.dInnerLength, blockNumTdtype) * sizeof(T));
        pipe.InitBuffer(gammaQue, BUFFER_NUM, ROUND_UP(selfTiling.dInnerLength, blockNumUdtype) * sizeof(U));
        pipe.InitBuffer(betaQue, BUFFER_NUM, ROUND_UP(selfTiling.dInnerLength, blockNumUdtype) * sizeof(U));
        pipe.InitBuffer(rstdQue, BUFFER_NUM, roundUp1Dtype * sizeof(float));
        pipe.InitBuffer(meanQue, BUFFER_NUM, roundUp1Dtype * sizeof(float));
    }

    __aicore__ inline void InitGmAccumulator()
    {
        // cutD serves unbounded col: every GM zero-init below walks the col-length (or batch*col) output in
        // dInnerLength-sized chunks, using a dInnerLength-sized UB scratch, so no col-unbounded write can
        // overflow the scratch buffer (same class of overflow as the colAlignM InitBuffer issue).
        if (isDeterministicKey || B16_OUTPUT) {
            // workspace gamma/beta (colAlignV long) and, in row-split, the per-batch scale/shift slices:
            // InitGmData chunks internally when given the scratch capacity as elemWithDInUBFp32.
            const uint32_t scratchElemFp32 = ROUND_UP(selfTiling.dInnerLength, FLOAT_BLOCK_ELEM);
            LocalTensor<float> tempLocal = pdGammaBuf.Get<float>();
            InitGmData(workspaceGmGamma, workspaceGmBeta, colAlignV, tempLocal, scratchElemFp32);
            if (WS_SCALE_SHIFT) {
                int64_t coreBase = GetBlockIdx() * workspaceNum * colAlignV;
                for (int64_t i = 0; i < selfTiling.batch; ++i) {
                    InitGmData(workspaceGMOri[coreBase + (2 + 2 * i) * colAlignV],
                               workspaceGMOri[coreBase + (3 + 2 * i) * colAlignV], colAlignV, tempLocal,
                               scratchElemFp32);
                }
            } else if (GetBlockIdx() == 0) {
                // batch-split: pd_scale/pd_shift accumulate on GM via in-core atomic adds -> zero once first
                LocalTensor<T> tempLocalT = pdXQue.AllocTensor<T>();
                InitGmZeroChunked<T>(pdScaleGm, tempLocalT, selfTiling.batch * selfTiling.col);
                InitGmZeroChunked<T>(pdShiftGm, tempLocalT, selfTiling.batch * selfTiling.col);
                pdXQue.FreeTensor(tempLocalT);
            }
        } else if (GetBlockIdx() == 0) {
            LocalTensor<U> tempLocalU = gammaOutTmp.Get<U>();
            InitGmZeroChunked<U>(pdGammaGm, tempLocalU, selfTiling.col);
            InitGmZeroChunked<U>(pdBetaGm, tempLocalU, selfTiling.col);
            LocalTensor<T> tempLocalT = pdXQue.AllocTensor<T>();
            InitGmZeroChunked<T>(pdScaleGm, tempLocalT, selfTiling.batch * selfTiling.col);
            InitGmZeroChunked<T>(pdShiftGm, tempLocalT, selfTiling.batch * selfTiling.col);
            pdXQue.FreeTensor(tempLocalT);
        }
    }

    // zero dstGm[0..totalElem) in dInnerLength-sized chunks so the dInnerLength-sized scratch is never
    // written beyond its capacity, even though the output region is col/batch*col long (unbounded in cutD).
    template <typename Dt>
    __aicore__ inline void InitGmZeroChunked(GlobalTensor<Dt> dstGm, LocalTensor<Dt> scratch, const uint32_t totalElem)
    {
        const uint32_t chunkElem = ROUND_UP(selfTiling.dInnerLength, BLOCK_ALIGN_B / sizeof(Dt));
        for (uint32_t off = 0; off < totalElem; off += chunkElem) {
            InitSingleData<Dt>(dstGm[off], scratch, MIN(chunkElem, totalElem - off));
        }
    }

    __aicore__ inline void LoadRowScalars(const int64_t rowIndex)
    {
        LocalTensor<float> rstdLocal = rstdQue.AllocTensor<float>();
        LocalTensor<float> meanLocal = meanQue.AllocTensor<float>();
        DataCopyPad(rstdLocal, rstdGm[rowIndex], DataCopyParams{1, (uint16_t)sizeof(float), 0, 0},
                    DataCopyPadParams{true, 0, (uint8_t)(roundUp1Dtype - 1), 0});
        DataCopyPad(meanLocal, meanGm[rowIndex], DataCopyParams{1, (uint16_t)sizeof(float), 0, 0},
                    DataCopyPadParams{true, 0, (uint8_t)(roundUp1Dtype - 1), 0});
        rstdQue.EnQue(rstdLocal);
        meanQue.EnQue(meanLocal);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyInChunk(const int64_t offsetUbXY, const int64_t offsetUbGamma, const int64_t batch,
                                       const int64_t dInOnceUb, const bool withBeta)
    {
        LocalTensor<T> dyLocal = dyQue.AllocTensor<T>();
        LocalTensor<T> xLocal = xQue.AllocTensor<T>();
        LocalTensor<T> scaleLocal = scaleQue.AllocTensor<T>();
        LocalTensor<U> gammaLocal = gammaQue.AllocTensor<U>();
        LocalTensor<U> betaLocal;
        if (withBeta) {
            betaLocal = betaQue.AllocTensor<U>();
        }
        DataCopyParams copyParams{1, (uint16_t)(dInOnceUb * sizeof(T)), 0, 0};
        uint8_t padRight = ROUND_UP(dInOnceUb, blockNumTdtype) - dInOnceUb;
        DataCopyPadParams padParams{true, 0, padRight, 0};
        // gamma/beta carry dtype U, not the dy dtype T: using the T block length/alignment truncates the
        // load (e.g. fp32 gamma under b16 dy) and can push rightPadding past a full 32B block (MTE2 fault).
        DataCopyParams copyParamsU{1, (uint16_t)(dInOnceUb * sizeof(U)), 0, 0};
        uint8_t padRightU = ROUND_UP(dInOnceUb, blockNumUdtype) - dInOnceUb;
        DataCopyPadParams padParamsU{true, 0, padRightU, 0};
        DataCopyPad(dyLocal, dyGm[offsetUbXY], copyParams, padParams);
        DataCopyPad(xLocal, xGm[offsetUbXY], copyParams, padParams);
        DataCopyPad(scaleLocal, scaleGm[batch * selfTiling.col + offsetUbGamma], copyParams, padParams);
        DataCopyPad(gammaLocal, gammaGm[offsetUbGamma], copyParamsU, padParamsU);
        if (withBeta) {
            DataCopyPad(betaLocal, betaGm[offsetUbGamma], copyParamsU, padParamsU);
        }
        dyQue.EnQue(dyLocal);
        xQue.EnQue(xLocal);
        scaleQue.EnQue(scaleLocal);
        gammaQue.EnQue(gammaLocal);
        if (withBeta) {
            betaQue.EnQue(betaLocal);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void ComputePass1Chunk(const int64_t offsetUbGamma, const int64_t dInOnceUb,
                                             const LocalTensor<float>& inputRstd, const LocalTensor<float>& inputMean)
    {
        LocalTensor<T> inputDy = dyQue.DeQue<T>();
        LocalTensor<T> inputX = xQue.DeQue<T>();
        LocalTensor<T> inputScale = scaleQue.DeQue<T>();
        LocalTensor<U> inputGamma = gammaQue.DeQue<U>();
        LocalTensor<U> inputBeta = betaQue.DeQue<U>();

        __ubuf__ T* dyTAddr = (__ubuf__ T*)inputDy.GetPhyAddr();
        __ubuf__ T* xTAddr = (__ubuf__ T*)inputX.GetPhyAddr();
        __ubuf__ T* scaleTAddr = (__ubuf__ T*)inputScale.GetPhyAddr();
        __ubuf__ U* gammaTAddr = (__ubuf__ U*)inputGamma.GetPhyAddr();
        __ubuf__ U* betaTAddr = (__ubuf__ U*)inputBeta.GetPhyAddr();
        __ubuf__ float* rstdAddr = (__ubuf__ float*)inputRstd.GetPhyAddr();
        __ubuf__ float* meanAddr = (__ubuf__ float*)inputMean.GetPhyAddr();

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

        uint16_t colLoopTimes = static_cast<uint16_t>((dInOnceUb + V_LENGTH - 1) / V_LENGTH);

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
            RegTensor<float> xCentered;
            RegTensor<float> y;
            RegTensor<float> dyGY;
            RegTensor<float> yGamma;
            RegTensor<float> yGammaBeta;
            RegTensor<float> dyYGB;
            RegTensor<float> dyG;

            MaskReg pregLoop;
            DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg, meanAddr);
            DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr);

            for (uint16_t j = 0; j < colLoopTimes; ++j) {
                uint32_t remainElem = dInOnceUb - j * V_LENGTH;
                pregLoop = UpdateMask<float>(remainElem);
                uint32_t chunkBase = j * V_LENGTH;

                LoadTensor(dy, dyTAddr + chunkBase, pregLoop);
                LoadTensor(x, xTAddr + chunkBase, pregLoop);
                LoadTensor(gamma, gammaTAddr + chunkBase, pregLoop);
                LoadTensor(beta, betaTAddr + chunkBase, pregLoop);
                LoadTensor(scale, scaleTAddr + chunkBase, pregLoop);

                Adds(scaleP1, scale, 1.0f, pregLoop);
                Mul(dyScaleP1, dy, scaleP1, pregLoop);

                CopyToTensor(dBetaAddr + chunkBase, dyScaleP1, pregLoop);

                Sub(xCentered, x, meanReg, pregLoop);
                Mul(y, xCentered, rstdReg, pregLoop);

                Mul(dyGY, dyScaleP1, y, pregLoop);
                CopyToTensor(dGammaAddr + chunkBase, dyGY, pregLoop);

                Mul(yGamma, y, gamma, pregLoop);
                Add(yGammaBeta, yGamma, beta, pregLoop);
                Mul(dyYGB, dy, yGammaBeta, pregLoop);
                CopyToTensor(dScaleAddr + chunkBase, dyYGB, pregLoop);

                CopyToTensor(dShiftAddr + chunkBase, dy, pregLoop);

                Mul(dyG, dyScaleP1, gamma, pregLoop);
                CopyToTensor(dyGStageAddr + chunkBase, dyG, pregLoop);
                CopyToTensor(yStageAddr + chunkBase, y, pregLoop);
            }
        }

        PipeBarrier<PIPE_V>();
        AscendC::Mul(yStageLocal, dyGStageLocal, yStageLocal, dInOnceUb);
        PipeBarrier<PIPE_V>();
        float p3 = ReduceSumCustom(yStageLocal, dInOnceUb);
        float p2 = ReduceSumCustom(dyGStageLocal, dInOnceUb);
        float acc2 = tmpLocal.GetValue(0) + p2;
        float acc3 = tmpLocal.GetValue(1) + p3;
        tmpLocal.SetValue(0, acc2);
        tmpLocal.SetValue(1, acc3);
        PipeBarrier<PIPE_V>();

        dyQue.FreeTensor(inputDy);
        xQue.FreeTensor(inputX);
        scaleQue.FreeTensor(inputScale);
        gammaQue.FreeTensor(inputGamma);
        betaQue.FreeTensor(inputBeta);
    }

    __aicore__ inline void FlushChunkAcc(const int64_t offsetUbGamma, const int64_t batch, const int64_t dInOnceUb)
    {
        LocalTensor<float> dGammaLocal = pdGammaBuf.Get<float>();
        LocalTensor<float> dBetaLocal = pdBetaBuf.Get<float>();
        LocalTensor<float> dScaleLocal = dScaleBuf.Get<float>();
        LocalTensor<float> dShiftLocal = dShiftBuf.Get<float>();

        event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event);
        WaitFlag<HardEvent::V_MTE3>(event);

        if (isDeterministicKey || B16_OUTPUT) {
            SetAtomicAdd<float>();
            DataCopyAtomicAdd(workspaceGmGamma[offsetUbGamma], dGammaLocal, dInOnceUb, 0, (uint16_t)1);
            DataCopyAtomicAdd(workspaceGmBeta[offsetUbGamma], dBetaLocal, dInOnceUb, 0, (uint16_t)1);
            SetAtomicNone();
            if (WS_SCALE_SHIFT) {
                int64_t coreBase = GetBlockIdx() * workspaceNum * colAlignV;
                SetAtomicAdd<float>();
                DataCopyAtomicAdd(workspaceGMOri[coreBase + (2 + 2 * batch) * colAlignV + offsetUbGamma], dScaleLocal,
                                  dInOnceUb, 0, (uint16_t)1);
                DataCopyAtomicAdd(workspaceGMOri[coreBase + (3 + 2 * batch) * colAlignV + offsetUbGamma], dShiftLocal,
                                  dInOnceUb, 0, (uint16_t)1);
                SetAtomicNone();
            } else {
                FlushScaleShiftGm(offsetUbGamma, batch, dInOnceUb, dScaleLocal, dShiftLocal);
            }
        } else {
            if constexpr (std::is_same_v<U, float>) {
                SetAtomicAdd<float>();
                DataCopyAtomicAdd(pdGammaGm[offsetUbGamma], dGammaLocal, dInOnceUb, 0, (uint16_t)1);
                DataCopyAtomicAdd(pdBetaGm[offsetUbGamma], dBetaLocal, dInOnceUb, 0, (uint16_t)1);
                SetAtomicNone();
            } else {
                LocalTensor<U> castTmp = gammaOutTmp.Get<U>();
                Cast(castTmp, dGammaLocal, RoundMode::CAST_RINT, dInOnceUb);
                PipeBarrier<PIPE_V>();
                SetAtomicAdd<U>();
                DataCopyPad(pdGammaGm[offsetUbGamma], castTmp,
                            DataCopyParams{1, (uint16_t)(dInOnceUb * sizeof(U)), 0, 0});
                SetAtomicNone();
                PipeBarrier<PIPE_MTE3>();
                Cast(castTmp, dBetaLocal, RoundMode::CAST_RINT, dInOnceUb);
                PipeBarrier<PIPE_V>();
                SetAtomicAdd<U>();
                DataCopyPad(pdBetaGm[offsetUbGamma], castTmp,
                            DataCopyParams{1, (uint16_t)(dInOnceUb * sizeof(U)), 0, 0});
                SetAtomicNone();
            }
            FlushScaleShiftGm(offsetUbGamma, batch, dInOnceUb, dScaleLocal, dShiftLocal);
        }
    }

    __aicore__ inline void FlushScaleShiftGm(const int64_t offsetUbGamma, const int64_t batch, const int64_t dInOnceUb,
                                             const LocalTensor<float>& dScaleLocal,
                                             const LocalTensor<float>& dShiftLocal)
    {
        if constexpr (std::is_same_v<T, float>) {
            SetAtomicAdd<float>();
            DataCopyPad(pdScaleGm[batch * selfTiling.col + offsetUbGamma], dScaleLocal,
                        DataCopyParams{1, (uint16_t)(dInOnceUb * sizeof(T)), 0, 0});
            DataCopyPad(pdShiftGm[batch * selfTiling.col + offsetUbGamma], dShiftLocal,
                        DataCopyParams{1, (uint16_t)(dInOnceUb * sizeof(T)), 0, 0});
            SetAtomicNone();
        } else {
            PipeBarrier<PIPE_MTE3>();
            LocalTensor<T> scaleOut = scaleOutTmp.Get<T>();
            LocalTensor<T> shiftOut = shiftOutTmp.Get<T>();
            Cast(scaleOut, dScaleLocal, RoundMode::CAST_RINT, dInOnceUb);
            Cast(shiftOut, dShiftLocal, RoundMode::CAST_RINT, dInOnceUb);
            PipeBarrier<PIPE_V>();
            SetAtomicAdd<T>();
            DataCopyPad(pdScaleGm[batch * selfTiling.col + offsetUbGamma], scaleOut,
                        DataCopyParams{1, (uint16_t)(dInOnceUb * sizeof(T)), 0, 0});
            DataCopyPad(pdShiftGm[batch * selfTiling.col + offsetUbGamma], shiftOut,
                        DataCopyParams{1, (uint16_t)(dInOnceUb * sizeof(T)), 0, 0});
            SetAtomicNone();
        }
    }

    __aicore__ inline void ComputePass2Chunk(const int64_t rowIndex, const int64_t offsetUbXY, const int64_t dInOnceUb,
                                             const LocalTensor<float>& inputRstd, const LocalTensor<float>& inputMean)
    {
        LocalTensor<T> inputDy = dyQue.DeQue<T>();
        LocalTensor<T> inputX = xQue.DeQue<T>();
        LocalTensor<T> inputScale = scaleQue.DeQue<T>();
        LocalTensor<U> inputGamma = gammaQue.DeQue<U>();

        __ubuf__ T* dyTAddr = (__ubuf__ T*)inputDy.GetPhyAddr();
        __ubuf__ T* xTAddr = (__ubuf__ T*)inputX.GetPhyAddr();
        __ubuf__ T* scaleTAddr = (__ubuf__ T*)inputScale.GetPhyAddr();
        __ubuf__ U* gammaTAddr = (__ubuf__ U*)inputGamma.GetPhyAddr();
        __ubuf__ float* rstdAddr = (__ubuf__ float*)inputRstd.GetPhyAddr();
        __ubuf__ float* meanAddr = (__ubuf__ float*)inputMean.GetPhyAddr();

        LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
        float reduce2 = tmpLocal.GetValue(0);
        float reduce3 = tmpLocal.GetValue(1);

        LocalTensor<T> outputPdX = pdXQue.AllocTensor<T>();
        __ubuf__ T* pdXTAddr = (__ubuf__ T*)outputPdX.GetPhyAddr();

        uint16_t colLoopTimes = static_cast<uint16_t>((dInOnceUb + V_LENGTH - 1) / V_LENGTH);

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
            DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg, meanAddr);
            DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr);
            Duplicate(reduce2Reg, reduce2, pregFull);
            Duplicate(reduce3Reg, reduce3, pregFull);

            for (uint16_t j = 0; j < colLoopTimes; ++j) {
                uint32_t remainElem = dInOnceUb - j * V_LENGTH;
                pregLoop = UpdateMask<float>(remainElem);
                uint32_t chunkBase = j * V_LENGTH;

                LoadTensor(dy, dyTAddr + chunkBase, pregLoop);
                LoadTensor(x, xTAddr + chunkBase, pregLoop);
                LoadTensor(gamma, gammaTAddr + chunkBase, pregLoop);
                LoadTensor(scale, scaleTAddr + chunkBase, pregLoop);

                Adds(scaleP1, scale, 1.0f, pregLoop);
                Mul(dyG, dy, scaleP1, pregLoop);
                Mul(dyG, dyG, gamma, pregLoop);
                Sub(xCentered, x, meanReg, pregLoop);
                Mul(y, xCentered, rstdReg, pregLoop);
                Sub(pdX, dyG, reduce2Reg, pregLoop);
                Mul(tmp, y, reduce3Reg, pregLoop);
                Sub(pdX, pdX, tmp, pregLoop);
                Mul(pdX, pdX, rstdReg, pregLoop);
                CopyToTensor(pdXTAddr + chunkBase, pdX, pregLoop);
            }
        }
        PipeBarrier<PIPE_V>();

        dyQue.FreeTensor(inputDy);
        xQue.FreeTensor(inputX);
        scaleQue.FreeTensor(inputScale);
        gammaQue.FreeTensor(inputGamma);
        pdXQue.EnQue(outputPdX);
        CopyOutPdX(rowIndex, offsetUbXY, dInOnceUb);
    }

    __aicore__ inline void CopyOutPdX(const int64_t rowIndex, const int64_t offsetUbXY, const int64_t dInOnceUb)
    {
        LocalTensor<T> outputPdX = pdXQue.DeQue<T>();
        DataCopyPad(pdXGm[offsetUbXY], outputPdX, DataCopyParams{1, (uint16_t)(dInOnceUb * sizeof(T)), 0, 0});
        PipeBarrier<PIPE_ALL>();
        pdXQue.FreeTensor(outputPdX);
    }

private:
    TPipe pipe;
    AdaLayerNormGradTilingDataArch35 selfTiling;
    TQue<QuePosition::VECIN, BUFFER_NUM> dyQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> scaleQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> gammaQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> betaQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> rstdQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> meanQue;
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
    uint32_t roundUp1Dtype = BLOCK_ALIGN_B / sizeof(float);
    uint32_t blockNumTdtype = BLOCK_ALIGN_B / sizeof(T);
    uint32_t blockNumUdtype = BLOCK_ALIGN_B / sizeof(U);
    int64_t myRows = 0;
    int64_t myStartRow = 0;
    uint64_t deterministicWorkSpaceSize = 0;
    bool isComputedCore = false;
    bool isDeterministicKey = false;
};

} // namespace AdaLayerNormGrad

#endif // ADA_LAYER_NORM_GRAD_CUT_D_ARCH35_H
