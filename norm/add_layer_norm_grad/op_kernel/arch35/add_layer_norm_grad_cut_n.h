/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ADD_LAYER_NORM_GRAD_CUT_N_A35
#define ADD_LAYER_NORM_GRAD_CUT_N_A35

#include "add_layer_norm_grad_common.h"
#include "../add_layer_norm_determinstic_compute.h"

namespace AddLayerNormGrad {
using namespace AscendC;
using namespace AscendC::Reg;

template <typename T, int TILING_KEY>
class KernelAddLayerNormGradA35 {
#define HAS_ADDITIONAL_INPUT ((TILING_KEY % 10) == 1)
public:
    __aicore__ inline KernelAddLayerNormGradA35() {}

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x_1, GM_ADDR x_2, GM_ADDR rstd, GM_ADDR mean, GM_ADDR gamma,
                                GM_ADDR dsum, GM_ADDR d_x, GM_ADDR d_gamma, GM_ADDR d_beta,
                                const AddLayerNormGradTilingData tiling, GM_ADDR workspace)
    {
        selfTiling = tiling;
        isDeterministicKey = tiling.isDeterministicKey;
        roundUpNumLastDimFloatLen = selfTiling.roundUpNumLastDimFloat / sizeof(float);

        if (GetBlockIdx() != selfTiling.numCore - 1) {
            nInOneCore = tiling.nInOneCoreNorm;
            gmOneCoreElemXY = tiling.gmOneCoreElemXYNorm;
            nAvailInUb = tiling.nAvailInUbNorm;
            nMiddleCount = tiling.nMiddleCountNorm;
            nInUbTotalTail = tiling.nInUbTotalNormTail;
        } else {
            nInOneCore = tiling.nInOneCoreTail;
            gmOneCoreElemXY = tiling.gmOneCoreElemXYTail;
            nAvailInUb = tiling.nAvailInUbTail;
            nMiddleCount = tiling.nMiddleCountTail;
            nInUbTotalTail = tiling.nInUbTotalTailTail;
        }

        blockNumber = BLOCK_AlIGN / sizeof(float);
        if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
            blockNumberTdtype = BLOCK_AlIGN / sizeof(half);
        } else {
            blockNumberTdtype = BLOCK_AlIGN / sizeof(float);
        }

        if (GetBlockIdx() < selfTiling.numCore) {
            isComputedCore = true;
        }

        if (isDeterministicKey) {
            deterministicWorkSpaceSize = roundUpNumLastDimFloatLen * CONSTANT_TWO * selfTiling.numCore;
            workspaceGMOri.SetGlobalBuffer((__gm__ float*)workspace, deterministicWorkSpaceSize);
        }

        if (selfTiling.numLastDim < BLOCK_AlIGN / sizeof(T)) {
            if (GetBlockIdx() == 0) {
                GlobalTensor<T> dXGmAll;
                dXGmAll.SetGlobalBuffer((__gm__ T*)d_x, selfTiling.numLastDim * selfTiling.numFirstDim);
                uint32_t fullAlign = selfTiling.numFirstDim * selfTiling.numLastDim * sizeof(T) / FULL_ALIGN_BLOCK *
                                     FULL_ALIGN_BLOCK / sizeof(T);
                if (fullAlign != 0) {
                    InitGlobalMemory(dXGmAll, fullAlign, static_cast<T>(0.0));
                }
                for (uint32_t i = 0; i < selfTiling.numLastDim * selfTiling.numFirstDim - fullAlign; i++) {
                    dXGmAll.SetValue(i + fullAlign, static_cast<T>(0.0));
                }
                DataCacheCleanAndInvalid<T, AscendC::CacheLine::SINGLE_CACHE_LINE>(dXGmAll[fullAlign]);
                PipeBarrier<PIPE_ALL>();
            }
        }

        InitOutputQueue();
        InitOuputGmBuffer(d_gamma, d_beta);

        if (isComputedCore) {
            if (isDeterministicKey) {
                InitWorkspaceGmBuffer(workspace);
            }
            InitInputGmBuffer(dy, x_1, x_2, rstd, mean, gamma, dsum);
            InitInputQueue();
            dXGm.SetGlobalBuffer((__gm__ T*)d_x + static_cast<uint64_t>(GetBlockIdx()) * selfTiling.ndInOneCoreLength,
                                 gmOneCoreElemXY);
            InitTmpBuffer();
        }
        SyncAll();
    }

    __aicore__ inline void InitInputGmBuffer(GM_ADDR dy, GM_ADDR x_1, GM_ADDR x_2, GM_ADDR rstd, GM_ADDR mean,
                                             GM_ADDR gamma, GM_ADDR dsum)
    {
        dyGm.SetGlobalBuffer((__gm__ T*)dy + static_cast<uint64_t>(GetBlockIdx()) * selfTiling.ndInOneCoreLength,
                             gmOneCoreElemXY);
        x1Gm.SetGlobalBuffer((__gm__ T*)x_1 + static_cast<uint64_t>(GetBlockIdx()) * selfTiling.ndInOneCoreLength,
                             gmOneCoreElemXY);
        x2Gm.SetGlobalBuffer((__gm__ T*)x_2 + static_cast<uint64_t>(GetBlockIdx()) * selfTiling.ndInOneCoreLength,
                             gmOneCoreElemXY);
        if (selfTiling.rstdElemNum == 1) {
            rstdGm.SetGlobalBuffer((__gm__ float*)rstd, 1);
            meanGm.SetGlobalBuffer((__gm__ float*)mean, 1);
        } else {
            rstdGm.SetGlobalBuffer((__gm__ float*)rstd + GetBlockIdx() * selfTiling.nInOneCoreLength, nInOneCore);
            meanGm.SetGlobalBuffer((__gm__ float*)mean + GetBlockIdx() * selfTiling.nInOneCoreLength, nInOneCore);
        }
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma, selfTiling.numLastDim);
        if constexpr (HAS_ADDITIONAL_INPUT) {
            dSumGm.SetGlobalBuffer(
                (__gm__ T*)dsum + static_cast<uint64_t>(GetBlockIdx()) * selfTiling.ndInOneCoreLength, gmOneCoreElemXY);
        }
    }

    __aicore__ inline void InitWorkspaceGmBuffer(GM_ADDR workspace)
    {
        workspaceGmGamma.SetGlobalBuffer(
            (__gm__ float*)workspace + GetBlockIdx() * CONSTANT_TWO * roundUpNumLastDimFloatLen,
            roundUpNumLastDimFloatLen);
        workspaceGmBeta.SetGlobalBuffer(
            (__gm__ float*)workspace + (1 + GetBlockIdx() * CONSTANT_TWO) * roundUpNumLastDimFloatLen,
            roundUpNumLastDimFloatLen);
        LocalTensor<float> tempLocalTensor = dGammaQue.AllocTensor<float>();
        InitGmData(workspaceGmGamma, workspaceGmBeta, roundUpNumLastDimFloatLen, tempLocalTensor,
                   selfTiling.roundUpNumLastDimFloat);
        dGammaQue.FreeTensor(tempLocalTensor);
    }

    __aicore__ inline void InitOuputGmBuffer(GM_ADDR d_gamma, GM_ADDR d_beta)
    {
        dGammaGm.SetGlobalBuffer((__gm__ float*)d_gamma, selfTiling.numLastDim);
        dBetaGm.SetGlobalBuffer((__gm__ float*)d_beta, selfTiling.numLastDim);
        if (GetBlockIdx() == 0) {
            LocalTensor<float> tempLocalTensor = dGammaQue.AllocTensor<float>();
            InitGmData(dGammaGm, dBetaGm, selfTiling.numLastDim, tempLocalTensor, selfTiling.roundUpNumLastDimFloat);
            dGammaQue.FreeTensor(tempLocalTensor);
        }
    }

    __aicore__ inline void InitInputQueue()
    {
        pipe.InitBuffer(dyQue, BUFFER_NUM, selfTiling.ndRoundUpDtypeNorm);
        pipe.InitBuffer(x1Que, BUFFER_NUM, selfTiling.ndRoundUpDtypeNorm);
        pipe.InitBuffer(x2Que, BUFFER_NUM, selfTiling.ndRoundUpDtypeNorm);
        pipe.InitBuffer(rstdQue, BUFFER_NUM, selfTiling.n1RoundUpFloatNorm);
        pipe.InitBuffer(meanQue, BUFFER_NUM, selfTiling.n1RoundUpFloatNorm);
        pipe.InitBuffer(gammaQue, BUFFER_NUM, selfTiling.roundUpNumLastDimDtype);
        if constexpr (HAS_ADDITIONAL_INPUT) {
            pipe.InitBuffer(dSumQue, BUFFER_NUM, selfTiling.ndRoundUpDtypeNorm);
        }
    }

    __aicore__ inline void InitOutputQueue()
    {
        pipe.InitBuffer(dXQue, BUFFER_NUM, selfTiling.ndRoundUpDtypeNorm);
        pipe.InitBuffer(dGammaQue, BUFFER_NUM, selfTiling.roundUpNumLastDimFloat);
        pipe.InitBuffer(dBetaQue, BUFFER_NUM, selfTiling.roundUpNumLastDimFloat);
    }

    __aicore__ inline void InitTmpBuffer()
    {
        // xCentered/dyGamma 各 roundUpNumLastDimFloat（容纳完整 numLastDim）
        // pdVar/pdMean 标量各 V_LENGTH（BroadcastScalar 读取用）
        pipe.InitBuffer(tmpBuf,
                        selfTiling.roundUpNumLastDimFloat * CONSTANT_TWO + V_LENGTH * sizeof(float) * CONSTANT_TWO);
        if constexpr (!is_same<T, float>::value) {
            pipe.InitBuffer(dyFp32Buf, selfTiling.roundUpNumLastDimFloat);
            pipe.InitBuffer(x1Fp32Buf, selfTiling.roundUpNumLastDimFloat);
            pipe.InitBuffer(x2Fp32Buf, selfTiling.roundUpNumLastDimFloat);
            pipe.InitBuffer(dXFp32Buf, selfTiling.roundUpNumLastDimFloat);
            if constexpr (HAS_ADDITIONAL_INPUT) {
                pipe.InitBuffer(dSumFp32Buf, selfTiling.roundUpNumLastDimFloat);
            }
        }
    }

    __aicore__ inline void CutNProcess()
    {
        if (!isComputedCore) {
            goto DETERMINISTIC;
        }
        {
            CopyInGamma(selfTiling.numLastDim, selfTiling.dyPadRight);
            LocalTensor<T> inputGamma = gammaQue.DeQue<T>();
            LocalTensor<float> gammaFp32Local;
            if constexpr (!is_same<T, float>::value) {
                gammaFp32Local = dyFp32Buf.Get<float>();
                Cast(gammaFp32Local, inputGamma, RoundMode::CAST_NONE, selfTiling.numLastDim);
                PipeBarrier<PIPE_V>();
            }

            float reduceAxisSize = (selfTiling.numLastDim != 0) ? 1.0f / selfTiling.numLastDim : 0.0f;

            for (int32_t NOuterUbIndex = 0; NOuterUbIndex < nMiddleCount; ++NOuterUbIndex) {
                uint32_t nInOnceUb = (NOuterUbIndex != nMiddleCount - 1) ? nAvailInUb : nInUbTotalTail;
                uint32_t offsetUbXY = NOuterUbIndex * nAvailInUb * selfTiling.numLastDim;
                uint32_t offsetUbMeanVar = NOuterUbIndex * nAvailInUb;

                CopyIn(offsetUbXY, offsetUbMeanVar, selfTiling.numLastDim, 1, nInOnceUb, selfTiling.dyPadRight,
                       selfTiling.rstdPadRight);

                LocalTensor<T> inputDy = dyQue.DeQue<T>();
                LocalTensor<T> inputX1 = x1Que.DeQue<T>();
                LocalTensor<T> inputX2 = x2Que.DeQue<T>();
                LocalTensor<float> inputRstd = rstdQue.DeQue<float>();
                LocalTensor<float> inputMean = meanQue.DeQue<float>();
                LocalTensor<T> inputDx;
                if constexpr (HAS_ADDITIONAL_INPUT) {
                    inputDx = dSumQue.DeQue<T>();
                }

                LocalTensor<T> outputDx = dXQue.AllocTensor<T>();
                LocalTensor<float> outputDgamma = dGammaQue.AllocTensor<float>();
                LocalTensor<float> outputDbeta = dBetaQue.AllocTensor<float>();
                Duplicate<float>(outputDgamma, 0.0f, selfTiling.numLastDim);
                Duplicate<float>(outputDbeta, 0.0f, selfTiling.numLastDim);

                LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
                uint32_t roundUpNumLastDimFp32 = selfTiling.roundUpNumLastDimFloat / sizeof(float);
                __ubuf__ float* xCenteredAddr = (__ubuf__ float*)tmpLocal.GetPhyAddr();
                // xCentered 区域占 roundUpNumLastDimFp32，dyGamma 紧随其后（与 InitTmpBuffer 分配一致）
                __ubuf__ float* dyGammaAddr = xCenteredAddr + roundUpNumLastDimFp32;
                __ubuf__ float* gammaAddr;
                __ubuf__ T* gammaTAddr;
                if constexpr (is_same<T, float>::value) {
                    gammaAddr = (__ubuf__ float*)inputGamma.GetPhyAddr();
                } else {
                    gammaAddr = (__ubuf__ float*)gammaFp32Local.GetPhyAddr();
                }
                gammaTAddr = (__ubuf__ T*)inputGamma.GetPhyAddr();

                __ubuf__ float* rstdAddr = (__ubuf__ float*)inputRstd.GetPhyAddr();
                __ubuf__ float* meanAddr = (__ubuf__ float*)inputMean.GetPhyAddr();
                __ubuf__ float* dgammaAddr = (__ubuf__ float*)outputDgamma.GetPhyAddr();
                __ubuf__ float* dbetaAddr = (__ubuf__ float*)outputDbeta.GetPhyAddr();

                __ubuf__ T* dyTAddr = (__ubuf__ T*)inputDy.GetPhyAddr();
                __ubuf__ T* x1TAddr = (__ubuf__ T*)inputX1.GetPhyAddr();
                __ubuf__ T* x2TAddr = (__ubuf__ T*)inputX2.GetPhyAddr();
                __ubuf__ T* dxTAddr = (__ubuf__ T*)outputDx.GetPhyAddr();
                __ubuf__ T* dsumTAddr;
                if constexpr (HAS_ADDITIONAL_INPUT) {
                    dsumTAddr = (__ubuf__ T*)inputDx.GetPhyAddr();
                }

                uint32_t roundUpNumLastDim = selfTiling.roundUpNumLastDim;
                uint32_t numLastDim = selfTiling.numLastDim;
                uint16_t colLoopTimes = static_cast<uint16_t>((numLastDim + V_LENGTH - 1) / V_LENGTH);

                __ubuf__ float* pdVarAddr = dyGammaAddr + roundUpNumLastDimFp32;
                __ubuf__ float* pdMeanAddr = pdVarAddr + V_LENGTH;
                LocalTensor<float> pdVarReduce = tmpLocal[roundUpNumLastDimFp32 * 2];
                LocalTensor<float> pdMeanReduce = tmpLocal[roundUpNumLastDimFp32 * 2 + V_LENGTH];
                LocalTensor<float> pdVarFull;
                LocalTensor<float> pdMeanFull;
                if constexpr (!is_same<T, float>::value) {
                    pdVarFull = x1Fp32Buf.Get<float>();
                    pdMeanFull = x2Fp32Buf.Get<float>();
                }

                for (int32_t nInnerIndex = 0; nInnerIndex < nInOnceUb; ++nInnerIndex) {
                    float meanNum = inputMean.GetValue(nInnerIndex * selfTiling.roundUp1Dtype);
                    float rstdNum = inputRstd.GetValue(nInnerIndex * selfTiling.roundUp1Dtype);
                    float rstd3Num = rstdNum * rstdNum * rstdNum;

                    __VEC_SCOPE__
                    {
                        RegTensor<float> dy;
                        RegTensor<float> x1;
                        RegTensor<float> x2;
                        RegTensor<float> gamma;
                        RegTensor<float> xSum;
                        RegTensor<float> dyGamma;
                        RegTensor<float> xCentered;
                        RegTensor<float> tmp;
                        RegTensor<float> meanReg;
                        RegTensor<float> rstdReg;
                        RegTensor<float> rstd3Reg;

                        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
                        MaskReg pregMerge = CreateMask<float, MaskPattern::VL1>();
                        MaskReg pregLoop;

                        DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg,
                                                                meanAddr + nInnerIndex * selfTiling.roundUp1Dtype);
                        DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg,
                                                                rstdAddr + nInnerIndex * selfTiling.roundUp1Dtype);
                        Duplicate(rstd3Reg, rstd3Num, pregFull);

                        for (uint16_t j = 0; j < colLoopTimes; j++) {
                            uint32_t remainElem = numLastDim - j * V_LENGTH;
                            pregLoop = UpdateMask<float>(remainElem);
                            uint32_t ubOffsetT = nInnerIndex * roundUpNumLastDim + j * V_LENGTH;

                            LoadTensor(dy, dyTAddr + ubOffsetT, pregLoop);
                            LoadTensor(x1, x1TAddr + ubOffsetT, pregLoop);
                            LoadTensor(x2, x2TAddr + ubOffsetT, pregLoop);
                            LoadTensor(gamma, gammaAddr + j * V_LENGTH, pregLoop);

                            Add(xSum, x1, x2, pregLoop);
                            Mul(dyGamma, dy, gamma, pregLoop);
                            Adds(xCentered, xSum, meanNum * (-1.0f), pregLoop);

                            CopyToTensor(xCenteredAddr + j * V_LENGTH, xCentered, pregLoop);
                            CopyToTensor(dyGammaAddr + j * V_LENGTH, dyGamma, pregLoop);

                            Muls(tmp, xCentered, rstd3Num, pregLoop);
                            Mul(tmp, dyGamma, tmp, pregLoop);
                            if constexpr (is_same<T, float>::value) {
                                CopyToTensor(x1TAddr + ubOffsetT, tmp, pregLoop);
                            } else {
                                __ubuf__ float* pdVarFullAddr = (__ubuf__ float*)pdVarFull.GetPhyAddr();
                                CopyToTensor(pdVarFullAddr + j * V_LENGTH, tmp, pregLoop);
                            }

                            // dgamma
                            Muls(tmp, xCentered, rstdNum, pregLoop);
                            Mul(tmp, dy, tmp, pregLoop);
                            RegTensor<float> dgammaAcc;
                            LoadTensor(dgammaAcc, (__ubuf__ float*)(dgammaAddr + j * V_LENGTH), pregLoop);
                            Add(dgammaAcc, dgammaAcc, tmp, pregLoop);
                            CopyToTensor(dgammaAddr + j * V_LENGTH, dgammaAcc, pregLoop);

                            // dbeta
                            RegTensor<float> dbetaAcc;
                            LoadTensor(dbetaAcc, (__ubuf__ float*)(dbetaAddr + j * V_LENGTH), pregLoop);
                            Add(dbetaAcc, dbetaAcc, dy, pregLoop);
                            CopyToTensor(dbetaAddr + j * V_LENGTH, dbetaAcc, pregLoop);

                            Muls(tmp, dyGamma, rstdNum, pregLoop);
                            Muls(tmp, tmp, -1.0f, pregLoop);
                            if constexpr (is_same<T, float>::value) {
                                CopyToTensor(x2TAddr + ubOffsetT, tmp, pregLoop);
                            } else {
                                __ubuf__ float* pdMeanFullAddr = (__ubuf__ float*)pdMeanFull.GetPhyAddr();
                                CopyToTensor(pdMeanFullAddr + j * V_LENGTH, tmp, pregLoop);
                            }
                        }
                    }

                    PipeBarrier<PIPE_V>();
                    float pdVar;
                    float pdMean;
                    if constexpr (is_same<T, float>::value) {
                        pdVar = ReduceSumCustom(inputX1[nInnerIndex * roundUpNumLastDim], numLastDim);
                        pdMean = ReduceSumCustom(inputX2[nInnerIndex * roundUpNumLastDim], numLastDim);
                    } else {
                        pdVar = ReduceSumCustom(pdVarFull, numLastDim);
                        pdMean = ReduceSumCustom(pdMeanFull, numLastDim);
                    }
                    pdVarReduce.SetValue(0, pdVar);
                    pdMeanReduce.SetValue(0, pdMean);
                    float pdVarScale = pdVar * (-reduceAxisSize);
                    float pdMeanScale = pdMean * reduceAxisSize;

                    __VEC_SCOPE__
                    {
                        RegTensor<float> dyGamma;
                        RegTensor<float> xCentered;
                        RegTensor<float> tmp;
                        RegTensor<float> scalarReg;
                        RegTensor<float> meanReg;
                        RegTensor<float> rstdReg;
                        RegTensor<float> reduceAxisReg;

                        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
                        MaskReg pregLoop;

                        DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg,
                                                                meanAddr + nInnerIndex * selfTiling.roundUp1Dtype);
                        DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg,
                                                                rstdAddr + nInnerIndex * selfTiling.roundUp1Dtype);

                        Duplicate(scalarReg, pdVarScale, pregFull);

                        RegTensor<float> pdMeanBroad;
                        Duplicate(pdMeanBroad, pdMeanScale, pregFull);

                        for (uint16_t j = 0; j < colLoopTimes; j++) {
                            uint32_t remainElem = numLastDim - j * V_LENGTH;
                            pregLoop = UpdateMask<float>(remainElem);
                            uint32_t ubOffsetT = nInnerIndex * roundUpNumLastDim + j * V_LENGTH;

                            // 从 UB 直接加载第一个 VEC_SCOPE 已保存的 xCentered/dyGamma
                            LoadTensor(xCentered, xCenteredAddr + j * V_LENGTH, pregLoop);
                            LoadTensor(dyGamma, dyGammaAddr + j * V_LENGTH, pregLoop);

                            Muls(tmp, xCentered, pdVarScale, pregLoop);
                            Muls(dyGamma, dyGamma, rstdNum, pregLoop);
                            Add(tmp, tmp, dyGamma, pregLoop);
                            Adds(tmp, tmp, pdMeanScale, pregLoop);

                            if constexpr (HAS_ADDITIONAL_INPUT) {
                                RegTensor<float> dSum;
                                LoadTensor(dSum, dsumTAddr + ubOffsetT, pregLoop);
                                Add(tmp, tmp, dSum, pregLoop);
                            }

                            CopyToTensor(dxTAddr + ubOffsetT, tmp, pregLoop);
                        }
                    }
                }

                PipeBarrier<PIPE_V>();

                dyQue.FreeTensor(inputDy);
                x1Que.FreeTensor(inputX1);
                x2Que.FreeTensor(inputX2);
                rstdQue.FreeTensor(inputRstd);
                meanQue.FreeTensor(inputMean);
                if constexpr (HAS_ADDITIONAL_INPUT) {
                    dSumQue.FreeTensor(inputDx);
                }
                dXQue.EnQue(outputDx);
                dGammaQue.EnQue(outputDgamma);
                dBetaQue.EnQue(outputDbeta);
                CopyOut(offsetUbXY, selfTiling.numLastDim, nInOnceUb);
            }
            gammaQue.FreeTensor(inputGamma);
        }
    DETERMINISTIC:
        if (isDeterministicKey) {
            SyncAll();
            pipe.Reset();
            AddLayerNormGradDeterminsticCompute op;
            op.initBuffer(pipe, dGammaGm, dBetaGm, workspaceGMOri, CONSTANT_TWO);
            op.FinalProcessDeterministic(roundUpNumLastDimFloatLen, selfTiling.numCore, selfTiling.numLastDim);
        }
    }

private:
    __aicore__ inline void CopyInGamma(const int32_t dYInUb, const int32_t dyPadRight)
    {
        LocalTensor<T> gammaLocal = gammaQue.AllocTensor<T>();
        DataCopyParams gammaDataCopyParams = {1, (uint16_t)(dYInUb * sizeof(T)), 0, 0};
        DataCopyPadParams dyPadParams{true, 0, (uint8_t)dyPadRight, 0};
        DataCopyPad(gammaLocal, gammaGm[0], gammaDataCopyParams, dyPadParams);
        gammaQue.EnQue(gammaLocal);
    }

    __aicore__ inline void CopyIn(const int32_t offsetUbXY, const int32_t offsetUbMeanVar, const int32_t dYInUb,
                                  const int32_t DRstdInUb, const int32_t nInOnceUb, const int32_t dyPadRight,
                                  const int32_t rstdPadRight)
    {
        LocalTensor<T> dyLocal = dyQue.AllocTensor<T>();
        LocalTensor<T> x1Local = x1Que.AllocTensor<T>();
        LocalTensor<T> x2Local = x2Que.AllocTensor<T>();
        LocalTensor<float> rstdLocal = rstdQue.AllocTensor<float>();
        LocalTensor<float> meanLocal = meanQue.AllocTensor<float>();
        LocalTensor<T> dSumLocal;
        if constexpr (HAS_ADDITIONAL_INPUT) {
            dSumLocal = dSumQue.AllocTensor<T>();
        }

        DataCopyParams dyDataCopyParams{1, (uint16_t)(dYInUb * sizeof(T)), 0, 0};
        DataCopyPadParams dyPadParams{true, 0, (uint8_t)dyPadRight, 0};
        DataCopyParams rstdDataCopyParams{1, (uint16_t)(DRstdInUb * sizeof(float)), 0, 0};
        DataCopyPadParams rstdPadParams{true, 0, (uint8_t)rstdPadRight, 0};

        // Keep aligned copies on the fast path and use padding only when an aligned read would cross the GM boundary.
        uint32_t blockNumelT = BLOCK_AlIGN / sizeof(T);
        uint32_t roundUpDy = ROUND_UP(dYInUb, blockNumelT);
        uint32_t blockNumelFloat = BLOCK_AlIGN / sizeof(float);
        uint32_t roundUpRstd = ROUND_UP(DRstdInUb, blockNumelFloat);
        bool rstdIsScalar = (selfTiling.rstdElemNum == 1);
        for (int32_t idx = 0; idx < nInOnceUb; idx++) {
            uint64_t dyOffset = static_cast<uint64_t>(offsetUbXY) + static_cast<uint64_t>(idx) * dYInUb;
            if (dyOffset + roundUpDy <= gmOneCoreElemXY) {
                DataCopy(dyLocal[idx * roundUpDy], dyGm[dyOffset], roundUpDy);
                DataCopy(x1Local[idx * roundUpDy], x1Gm[dyOffset], roundUpDy);
                DataCopy(x2Local[idx * roundUpDy], x2Gm[dyOffset], roundUpDy);
            } else {
                DataCopyPad(dyLocal[idx * roundUpDy], dyGm[dyOffset], dyDataCopyParams, dyPadParams);
                DataCopyPad(x1Local[idx * roundUpDy], x1Gm[dyOffset], dyDataCopyParams, dyPadParams);
                DataCopyPad(x2Local[idx * roundUpDy], x2Gm[dyOffset], dyDataCopyParams, dyPadParams);
            }
            uint32_t meanVarOffset = rstdIsScalar ? offsetUbMeanVar : (offsetUbMeanVar + idx * DRstdInUb);
            if (!rstdIsScalar && meanVarOffset + roundUpRstd <= nInOneCore) {
                DataCopy(rstdLocal[idx * roundUpRstd], rstdGm[meanVarOffset], roundUpRstd);
                DataCopy(meanLocal[idx * roundUpRstd], meanGm[meanVarOffset], roundUpRstd);
            } else {
                uint32_t scalarOffset = rstdIsScalar ? 0 : meanVarOffset;
                DataCopyPad(rstdLocal[idx * roundUpRstd], rstdGm[scalarOffset], rstdDataCopyParams, rstdPadParams);
                DataCopyPad(meanLocal[idx * roundUpRstd], meanGm[scalarOffset], rstdDataCopyParams, rstdPadParams);
            }
            if constexpr (HAS_ADDITIONAL_INPUT) {
                if (dyOffset + roundUpDy <= gmOneCoreElemXY) {
                    DataCopy(dSumLocal[idx * roundUpDy], dSumGm[dyOffset], roundUpDy);
                } else {
                    DataCopyPad(dSumLocal[idx * roundUpDy], dSumGm[dyOffset], dyDataCopyParams, dyPadParams);
                }
            }
        }

        PipeBarrier<PIPE_ALL>();
        dyQue.EnQue(dyLocal);
        x1Que.EnQue(x1Local);
        x2Que.EnQue(x2Local);
        rstdQue.EnQue(rstdLocal);
        meanQue.EnQue(meanLocal);
        if constexpr (HAS_ADDITIONAL_INPUT) {
            dSumQue.EnQue(dSumLocal);
        }
    }

    __aicore__ inline void CopyOut(const int32_t offsetUbXY, const int32_t dYInUb, const int32_t nInOnceUb)
    {
        LocalTensor<T> outputDx = dXQue.DeQue<T>();
        LocalTensor<float> outputDgamma = dGammaQue.DeQue<float>();
        LocalTensor<float> outputDbeta = dBetaQue.DeQue<float>();
        PipeBarrier<PIPE_ALL>();

        // 与 A2 一致：用 DataCopyCustom 处理非对齐小 shape（numLastDim < blockNumel）
        DataCopyCustom<T>(dXGm, outputDx, dYInUb, offsetUbXY, std::is_same_v<T, float>, (uint16_t)nInOnceUb);

        SetAtomicAdd<float>();
        if (isDeterministicKey) {
            DataCopyAutomicAdd(workspaceGmBeta, outputDbeta, selfTiling.numLastDim, 0, (uint16_t)1);
        } else {
            DataCopyAutomicAdd(dBetaGm, outputDbeta, selfTiling.numLastDim, 0, (uint16_t)1);
        }
        SetAtomicNone();

        SetAtomicAdd<float>();
        if (isDeterministicKey) {
            DataCopyAutomicAdd(workspaceGmGamma, outputDgamma, selfTiling.numLastDim, 0, (uint16_t)1);
        } else {
            DataCopyAutomicAdd(dGammaGm, outputDgamma, selfTiling.numLastDim, 0, (uint16_t)1);
        }
        PipeBarrier<PIPE_ALL>();
        SetAtomicNone();

        dXQue.FreeTensor(outputDx);
        dGammaQue.FreeTensor(outputDgamma);
        dBetaQue.FreeTensor(outputDbeta);
    }

private:
    TPipe pipe;
    AddLayerNormGradTilingData selfTiling;
    TQue<QuePosition::VECIN, BUFFER_NUM> dyQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> x1Que;
    TQue<QuePosition::VECIN, BUFFER_NUM> x2Que;
    TQue<QuePosition::VECIN, BUFFER_NUM> rstdQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> meanQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> gammaQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> dSumQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dXQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dGammaQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dBetaQue;

    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> dyFp32Buf;
    TBuf<TPosition::VECCALC> x1Fp32Buf;
    TBuf<TPosition::VECCALC> x2Fp32Buf;
    TBuf<TPosition::VECCALC> dXFp32Buf;
    TBuf<TPosition::VECCALC> dSumFp32Buf;
    GlobalTensor<float> rstdGm;
    GlobalTensor<float> meanGm;
    GlobalTensor<float> dGammaGm;
    GlobalTensor<float> dBetaGm;
    GlobalTensor<float> workspaceGmGamma;
    GlobalTensor<float> workspaceGmBeta;
    GlobalTensor<float> workspaceGMOri;
    GlobalTensor<T> dyGm;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> gammaGm;
    GlobalTensor<T> dXGm;
    GlobalTensor<T> dSumGm;

    uint64_t roundUpNumLastDimFloatLen;
    uint32_t nInOneCore;
    uint32_t gmOneCoreElemXY;
    uint32_t nAvailInUb;
    uint32_t nMiddleCount;
    uint32_t nInUbTotalTail;
    uint32_t blockNumber;
    uint32_t blockNumberTdtype;
    uint64_t deterministicWorkSpaceSize = 0;
    bool isComputedCore = false;
    bool isDeterministicKey = false;
};
} // namespace AddLayerNormGrad

#endif // ADD_LAYER_NORM_GRAD_CUT_N_A35
