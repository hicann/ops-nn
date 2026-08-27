/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ADD_LAYER_NORM_GRAD_CUT_D_A35
#define ADD_LAYER_NORM_GRAD_CUT_D_A35

#include "add_layer_norm_grad_common.h"
#include "../add_layer_norm_determinstic_compute.h"

namespace AddLayerNormGrad {
using namespace AscendC;
using namespace AscendC::Reg;

template <typename T, int TILING_KEY>
class KernelAddLayerNormGradLargeA35 {
#define HAS_ADDITIONAL_INPUT ((TILING_KEY % 10) == 1)
public:
    __aicore__ inline KernelAddLayerNormGradLargeA35() {}

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x_1, GM_ADDR x_2, GM_ADDR rstd, GM_ADDR mean, GM_ADDR gamma,
                                GM_ADDR dsum, GM_ADDR d_x, GM_ADDR d_gamma, GM_ADDR d_beta,
                                const AddLayerNormGradTilingData tiling, GM_ADDR workspace)
    {
        selfTiling = tiling;
        roundUpNumLastDimFloatLen = selfTiling.roundUpNumLastDimFloat / sizeof(float);

        nInOneCore = (GetBlockIdx() != selfTiling.numCore - 1) ? selfTiling.nInOneCoreLength :
                                                                 selfTiling.nInOneCoreLengthTail;
        gmOneCoreElemXY = static_cast<uint32_t>(static_cast<uint64_t>(nInOneCore) * selfTiling.numLastDim);

        blockNumber = BLOCK_AlIGN / sizeof(float);
        if constexpr (!is_same<T, float>::value) {
            blockNumberTdtype = BLOCK_AlIGN / sizeof(half);
        } else {
            blockNumberTdtype = BLOCK_AlIGN / sizeof(float);
        }

        if (GetBlockIdx() < selfTiling.numCore) {
            isComputedCore = true;
        }

        if (selfTiling.isDeterministicKey) {
            deterministicWorkSpaceSize = roundUpNumLastDimFloatLen * CONSTANT_TWO * selfTiling.numCore;
            workspaceGMOri.SetGlobalBuffer((__gm__ float*)workspace, deterministicWorkSpaceSize);
        }
        InitOutputQueue();
        InitOuputGmBuffer(d_gamma, d_beta);

        if (isComputedCore) {
            if (selfTiling.isDeterministicKey) {
                InitWorkspaceGmBuffer(workspace);
            }
            InitInputGmBuffer(dy, x_1, x_2, rstd, mean, gamma, dsum);
            InitInputQueue();
            InitTmpBuffer();
            dXGm.SetGlobalBuffer((__gm__ T*)d_x + static_cast<uint64_t>(GetBlockIdx()) * selfTiling.nInOneCoreLength *
                                                      selfTiling.numLastDim,
                                 gmOneCoreElemXY);
        }
        SyncAll();
    }

    __aicore__ inline void InitInputGmBuffer(GM_ADDR dy, GM_ADDR x_1, GM_ADDR x_2, GM_ADDR rstd, GM_ADDR mean,
                                             GM_ADDR gamma, GM_ADDR dsum)
    {
        dyGm.SetGlobalBuffer(
            (__gm__ T*)dy + static_cast<uint64_t>(GetBlockIdx()) * selfTiling.nInOneCoreLength * selfTiling.numLastDim,
            gmOneCoreElemXY);
        x1Gm.SetGlobalBuffer(
            (__gm__ T*)x_1 + static_cast<uint64_t>(GetBlockIdx()) * selfTiling.nInOneCoreLength * selfTiling.numLastDim,
            gmOneCoreElemXY);
        x2Gm.SetGlobalBuffer(
            (__gm__ T*)x_2 + static_cast<uint64_t>(GetBlockIdx()) * selfTiling.nInOneCoreLength * selfTiling.numLastDim,
            gmOneCoreElemXY);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + GetBlockIdx() * selfTiling.nInOneCoreLength, nInOneCore);
        meanGm.SetGlobalBuffer((__gm__ float*)mean + GetBlockIdx() * selfTiling.nInOneCoreLength, nInOneCore);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma, selfTiling.numLastDim);
        if constexpr (HAS_ADDITIONAL_INPUT) {
            dSumGm.SetGlobalBuffer((__gm__ T*)dsum + static_cast<uint64_t>(GetBlockIdx()) *
                                                         selfTiling.nInOneCoreLength * selfTiling.numLastDim,
                                   gmOneCoreElemXY);
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
                   ROUND_UP(selfTiling.dInnerLength, blockNumber));
        dGammaQue.FreeTensor(tempLocalTensor);
    }

    __aicore__ inline void InitOuputGmBuffer(GM_ADDR d_gamma, GM_ADDR d_beta)
    {
        dGammaGm.SetGlobalBuffer((__gm__ float*)d_gamma, selfTiling.numLastDim);
        dBetaGm.SetGlobalBuffer((__gm__ float*)d_beta, selfTiling.numLastDim);
        if (GetBlockIdx() == 0) {
            LocalTensor<float> tempLocalTensor = dGammaQue.AllocTensor<float>();
            InitGmData(dGammaGm, dBetaGm, selfTiling.numLastDim, tempLocalTensor,
                       ROUND_UP(selfTiling.dInnerLength, blockNumber));
            dGammaQue.FreeTensor(tempLocalTensor);
        }
    }

    __aicore__ inline void InitInputQueue()
    {
        pipe.InitBuffer(dyQue, BUFFER_NUM, ROUND_UP(selfTiling.dInnerLength, blockNumberTdtype) * sizeof(T));
        pipe.InitBuffer(x1Que, BUFFER_NUM, ROUND_UP(selfTiling.dInnerLength, blockNumberTdtype) * sizeof(T));
        pipe.InitBuffer(x2Que, BUFFER_NUM, ROUND_UP(selfTiling.dInnerLength, blockNumberTdtype) * sizeof(T));
        pipe.InitBuffer(rstdQue, BUFFER_NUM, ROUND_UP(1, blockNumber) * sizeof(float));
        pipe.InitBuffer(meanQue, BUFFER_NUM, ROUND_UP(1, blockNumber) * sizeof(float));
        pipe.InitBuffer(gammaQue, BUFFER_NUM, ROUND_UP(selfTiling.dInnerLength, blockNumberTdtype) * sizeof(T));
        if constexpr (HAS_ADDITIONAL_INPUT) {
            pipe.InitBuffer(dSumQue, BUFFER_NUM, ROUND_UP(selfTiling.dInnerLength, blockNumberTdtype) * sizeof(T));
        }
    }

    __aicore__ inline void InitOutputQueue()
    {
        pipe.InitBuffer(dXQue, BUFFER_NUM, ROUND_UP(selfTiling.dInnerLength, blockNumberTdtype) * sizeof(T));
        pipe.InitBuffer(dGammaQue, BUFFER_NUM, ROUND_UP(selfTiling.dInnerLength, blockNumber) * sizeof(float));
        pipe.InitBuffer(dBetaQue, BUFFER_NUM, ROUND_UP(selfTiling.dInnerLength, blockNumber) * sizeof(float));
    }

    __aicore__ inline void InitTmpBuffer()
    {
        pipe.InitBuffer(tmpMeanPdBuf, ROUND_UP(1, blockNumber) * sizeof(float));
        pipe.InitBuffer(tmpVarPdBuf, ROUND_UP(1, blockNumber) * sizeof(float));
        if constexpr (!is_same<T, float>::value) {
            pipe.InitBuffer(dyFp32Buf, ROUND_UP(selfTiling.dInnerLength, blockNumber) * sizeof(float));
            pipe.InitBuffer(x1Fp32Buf, ROUND_UP(selfTiling.dInnerLength, blockNumber) * sizeof(float));
            pipe.InitBuffer(x2Fp32Buf, ROUND_UP(selfTiling.dInnerLength, blockNumber) * sizeof(float));
            pipe.InitBuffer(dgammaFp32Buf, ROUND_UP(selfTiling.dInnerLength, blockNumber) * sizeof(float));
            pipe.InitBuffer(dXFp32Buf, ROUND_UP(selfTiling.dInnerLength, blockNumber) * sizeof(float));
            if constexpr (HAS_ADDITIONAL_INPUT) {
                pipe.InitBuffer(dSumFp32Buf, ROUND_UP(selfTiling.dInnerLength, blockNumber) * sizeof(float));
            }
        }
    }

    __aicore__ inline void CutDProcess()
    {
        if (!isComputedCore) {
            goto DETERMINISTIC;
        }
        {
            LocalTensor<float> tmpMeanPdLocal = tmpMeanPdBuf.Get<float>();
            LocalTensor<float> tmpVarPdLocal = tmpVarPdBuf.Get<float>();
            LocalTensor<float> dyFp32Local;
            LocalTensor<float> x1Fp32Local;
            LocalTensor<float> x2Fp32Local;
            LocalTensor<float> gammaFp32Local;
            LocalTensor<float> dXLocal;
            LocalTensor<float> dSumFp32Local;
            if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
                dyFp32Local = dyFp32Buf.Get<float>();
                x1Fp32Local = x1Fp32Buf.Get<float>();
                x2Fp32Local = x2Fp32Buf.Get<float>();
                gammaFp32Local = dgammaFp32Buf.Get<float>();
                dXLocal = dXFp32Buf.Get<float>();
                if constexpr (HAS_ADDITIONAL_INPUT) {
                    dSumFp32Local = dSumFp32Buf.Get<float>();
                }
            }

            for (int32_t nInnerIndex = 0; nInnerIndex < nInOneCore; ++nInnerIndex) {
                Duplicate(tmpVarPdLocal, 0.0f, blockNumber);
                Duplicate(tmpMeanPdLocal, 0.0f, blockNumber);
                PipeBarrier<PIPE_V>();

                for (int32_t DOuterUbIndex = 0; DOuterUbIndex < selfTiling.dOuterLength; ++DOuterUbIndex) {
                    uint32_t DInOnceUb = (DOuterUbIndex != selfTiling.dOuterLength - 1) ? selfTiling.dInnerLength :
                                                                                          selfTiling.dInnerLengthTail;
                    uint32_t offsetUbXY = DOuterUbIndex * selfTiling.dInnerLength + nInnerIndex * selfTiling.numLastDim;
                    uint32_t offsetUbMeanVar = nInnerIndex;
                    uint32_t offsetUbGamma = DOuterUbIndex * selfTiling.dInnerLength;
                    uint32_t elemCoutXYUb = DInOnceUb;

                    CopyIn(offsetUbXY, offsetUbMeanVar, elemCoutXYUb, 1, selfTiling.nAvailInUb, offsetUbGamma);
                    event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                    SetFlag<HardEvent::MTE2_V>(eventMte2V);
                    WaitFlag<HardEvent::MTE2_V>(eventMte2V);
                    LocalTensor<T> inputDy = dyQue.DeQue<T>();
                    LocalTensor<T> inputX1 = x1Que.DeQue<T>();
                    LocalTensor<T> inputX2 = x2Que.DeQue<T>();
                    LocalTensor<float> inputRstd = rstdQue.DeQue<float>();
                    LocalTensor<float> inputMean = meanQue.DeQue<float>();
                    LocalTensor<T> inputGamma = gammaQue.DeQue<T>();

                    LocalTensor<T> outputDx = dXQue.AllocTensor<T>();
                    LocalTensor<float> outputDgamma = dGammaQue.AllocTensor<float>();
                    LocalTensor<float> outputDbeta = dBetaQue.AllocTensor<float>();

                    if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
                        Cast(dyFp32Local, inputDy, RoundMode::CAST_NONE, elemCoutXYUb);
                        Cast(x1Fp32Local, inputX1, RoundMode::CAST_NONE, elemCoutXYUb);
                        Cast(x2Fp32Local, inputX2, RoundMode::CAST_NONE, elemCoutXYUb);
                        Cast(gammaFp32Local, inputGamma, RoundMode::CAST_NONE, elemCoutXYUb);
                        PipeBarrier<PIPE_V>();
                        MicroComputeFirstPart(dyFp32Local, x1Fp32Local, x2Fp32Local, inputRstd, inputMean,
                                              gammaFp32Local, outputDgamma, outputDbeta, dXLocal, tmpVarPdLocal,
                                              tmpMeanPdLocal, elemCoutXYUb);
                    } else {
                        MicroComputeFirstPart(inputDy, inputX1, inputX2, inputRstd, inputMean, inputGamma, outputDgamma,
                                              outputDbeta, outputDx, tmpVarPdLocal, tmpMeanPdLocal, elemCoutXYUb);
                    }

                    event_t eventV2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                    SetFlag<HardEvent::V_S>(eventV2S);
                    WaitFlag<HardEvent::V_S>(eventV2S);
                    float pdVarPartial;
                    float pdMeanPartial;
                    if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
                        pdVarPartial = dXLocal.GetValue(0);
                        pdMeanPartial = dXLocal.GetValue(1);
                    } else {
                        pdVarPartial = outputDx.GetValue(0);
                        pdMeanPartial = outputDx.GetValue(1);
                    }
                    Adds(tmpVarPdLocal, tmpVarPdLocal, pdVarPartial, blockNumber);
                    Adds(tmpMeanPdLocal, tmpMeanPdLocal, pdMeanPartial, blockNumber);
                    PipeBarrier<PIPE_V>();

                    dyQue.FreeTensor(inputDy);
                    x1Que.FreeTensor(inputX1);
                    x2Que.FreeTensor(inputX2);
                    rstdQue.FreeTensor(inputRstd);
                    meanQue.FreeTensor(inputMean);
                    gammaQue.FreeTensor(inputGamma);
                    dXQue.FreeTensor(outputDx);
                    dGammaQue.EnQue(outputDgamma);
                    dBetaQue.EnQue(outputDbeta);
                    event_t eventVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                    SetFlag<HardEvent::V_MTE3>(eventVMte3);
                    WaitFlag<HardEvent::V_MTE3>(eventVMte3);
                    CopyOutBetaGamma(offsetUbGamma, elemCoutXYUb);
                    PipeBarrier<PIPE_ALL>();
                }

                PipeBarrier<PIPE_V>();

                for (int32_t DOuterUbIndex = 0; DOuterUbIndex < selfTiling.dOuterLength; ++DOuterUbIndex) {
                    uint32_t DInOnceUb = (DOuterUbIndex != selfTiling.dOuterLength - 1) ? selfTiling.dInnerLength :
                                                                                          selfTiling.dInnerLengthTail;
                    uint32_t offsetUbXY = DOuterUbIndex * selfTiling.dInnerLength + nInnerIndex * selfTiling.numLastDim;
                    uint32_t offsetUbMeanVar = nInnerIndex;
                    uint32_t offsetUbGamma = DOuterUbIndex * selfTiling.dInnerLength;
                    uint32_t elemCoutXYUb = DInOnceUb;

                    CopyIn(offsetUbXY, offsetUbMeanVar, elemCoutXYUb, 1, selfTiling.nAvailInUb, offsetUbGamma, true);
                    event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                    SetFlag<HardEvent::MTE2_V>(eventMte2V);
                    WaitFlag<HardEvent::MTE2_V>(eventMte2V);
                    LocalTensor<T> inputDy = dyQue.DeQue<T>();
                    LocalTensor<T> inputX1 = x1Que.DeQue<T>();
                    LocalTensor<T> inputX2 = x2Que.DeQue<T>();
                    LocalTensor<float> inputRstd = rstdQue.DeQue<float>();
                    LocalTensor<float> inputMean = meanQue.DeQue<float>();
                    LocalTensor<T> inputGamma = gammaQue.DeQue<T>();
                    LocalTensor<T> inputDx;
                    LocalTensor<T> outputDx = dXQue.AllocTensor<T>();
                    if constexpr (HAS_ADDITIONAL_INPUT) {
                        inputDx = dSumQue.DeQue<T>();
                    }
                    if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
                        Cast(dyFp32Local, inputDy, RoundMode::CAST_NONE, elemCoutXYUb);
                        Cast(x1Fp32Local, inputX1, RoundMode::CAST_NONE, elemCoutXYUb);
                        Cast(x2Fp32Local, inputX2, RoundMode::CAST_NONE, elemCoutXYUb);
                        Cast(gammaFp32Local, inputGamma, RoundMode::CAST_NONE, elemCoutXYUb);
                        if constexpr (HAS_ADDITIONAL_INPUT) {
                            Cast(dSumFp32Local, inputDx, RoundMode::CAST_NONE, elemCoutXYUb);
                        }
                        PipeBarrier<PIPE_V>();
                        MicroComputeSecondPart(dyFp32Local, x1Fp32Local, x2Fp32Local, inputRstd, inputMean,
                                               gammaFp32Local, dSumFp32Local, dXLocal, tmpVarPdLocal, tmpMeanPdLocal,
                                               elemCoutXYUb);
                        if constexpr (is_same<T, half>::value) {
                            Cast(outputDx, dXLocal, RoundMode::CAST_NONE, elemCoutXYUb);
                        } else {
                            Cast(outputDx, dXLocal, RoundMode::CAST_RINT, elemCoutXYUb);
                        }
                        PipeBarrier<PIPE_V>();
                    } else {
                        MicroComputeSecondPart(inputDy, inputX1, inputX2, inputRstd, inputMean, inputGamma, inputDx,
                                               outputDx, tmpVarPdLocal, tmpMeanPdLocal, elemCoutXYUb);
                    }

                    dyQue.FreeTensor(inputDy);
                    x1Que.FreeTensor(inputX1);
                    x2Que.FreeTensor(inputX2);
                    rstdQue.FreeTensor(inputRstd);
                    meanQue.FreeTensor(inputMean);
                    gammaQue.FreeTensor(inputGamma);
                    if constexpr (HAS_ADDITIONAL_INPUT) {
                        dSumQue.FreeTensor(inputDx);
                    }
                    dXQue.EnQue(outputDx);
                    event_t eventVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                    SetFlag<HardEvent::V_MTE3>(eventVMte3);
                    WaitFlag<HardEvent::V_MTE3>(eventVMte3);
                    CopyOutX(offsetUbXY, elemCoutXYUb, selfTiling.nAvailInUb);
                }
            }
        }
    DETERMINISTIC:
        if (selfTiling.isDeterministicKey) {
            SyncAll();
            pipe.Reset();
            AddLayerNormGradDeterminsticCompute op;
            op.initBuffer(pipe, dGammaGm, dBetaGm, workspaceGMOri, CONSTANT_TWO);
            op.FinalProcessDeterministic(roundUpNumLastDimFloatLen, selfTiling.numCore, selfTiling.numLastDim);
        }
    }

private:
    __aicore__ inline void CopyIn(const int32_t offsetUbXY, const int32_t offsetUbMeanVar, const int32_t dYInUb,
                                  const int32_t DRstdInUb, const int32_t nInOnceUb, const int32_t offsetUbGamma,
                                  const bool hasDsum = false)
    {
        LocalTensor<T> dyLocal = dyQue.AllocTensor<T>();
        LocalTensor<T> x1Local = x1Que.AllocTensor<T>();
        LocalTensor<T> x2Local = x2Que.AllocTensor<T>();
        LocalTensor<float> rstdLocal = rstdQue.AllocTensor<float>();
        LocalTensor<float> meanLocal = meanQue.AllocTensor<float>();
        LocalTensor<T> gammaLocal = gammaQue.AllocTensor<T>();
        LocalTensor<T> dSumLocal;

        DataCopyParams dyDataCopyParams{(uint16_t)nInOnceUb, (uint16_t)(dYInUb * sizeof(T)), 0, 0};
        uint8_t dyPadRight = ROUND_UP(dYInUb, blockNumberTdtype) - dYInUb;
        DataCopyPadParams dyPadParams{true, 0, dyPadRight, 0};
        DataCopyParams rstdDataCopyParams{(uint16_t)nInOnceUb, (uint16_t)(DRstdInUb * sizeof(float)), 0, 0};
        uint8_t rstdPadRight = ROUND_UP(DRstdInUb, blockNumber) - DRstdInUb;
        DataCopyPadParams rstdPadParams{true, 0, rstdPadRight, 0};

        DataCopyPad(dyLocal, dyGm[offsetUbXY], dyDataCopyParams, dyPadParams);
        DataCopyPad(x1Local, x1Gm[offsetUbXY], dyDataCopyParams, dyPadParams);
        DataCopyPad(x2Local, x2Gm[offsetUbXY], dyDataCopyParams, dyPadParams);
        DataCopyPad(rstdLocal, rstdGm[offsetUbMeanVar], rstdDataCopyParams, rstdPadParams);
        DataCopyPad(meanLocal, meanGm[offsetUbMeanVar], rstdDataCopyParams, rstdPadParams);
        DataCopyParams gammaDataCopyParams = {1, (uint16_t)(dYInUb * sizeof(T)), 0, 0};
        DataCopyPad(gammaLocal, gammaGm[offsetUbGamma], gammaDataCopyParams, dyPadParams);
        if (HAS_ADDITIONAL_INPUT && hasDsum) {
            dSumLocal = dSumQue.AllocTensor<T>();
            DataCopyPad(dSumLocal, dSumGm[offsetUbXY], dyDataCopyParams, dyPadParams);
        }

        PipeBarrier<PIPE_ALL>();
        dyQue.EnQue(dyLocal);
        x1Que.EnQue(x1Local);
        x2Que.EnQue(x2Local);
        rstdQue.EnQue(rstdLocal);
        meanQue.EnQue(meanLocal);
        gammaQue.EnQue(gammaLocal);
        if (HAS_ADDITIONAL_INPUT && hasDsum) {
            dSumQue.EnQue(dSumLocal);
        }
    }

    __aicore__ inline void MicroComputeFirstPart(
        const LocalTensor<float>& inputDy, const LocalTensor<float>& inputX1, const LocalTensor<float>& inputX2,
        const LocalTensor<float>& inputRstd, const LocalTensor<float>& inputMean, const LocalTensor<float>& inputGamma,
        const LocalTensor<float>& outputDgamma, const LocalTensor<float>& outputDbeta,
        const LocalTensor<float>& outputDx, const LocalTensor<float>& tmpVarPdLocal,
        const LocalTensor<float>& tmpMeanPdLocal, const uint32_t elemCoutDXy)
    {
        __ubuf__ float* dyAddr = (__ubuf__ float*)inputDy.GetPhyAddr();
        __ubuf__ float* x1Addr = (__ubuf__ float*)inputX1.GetPhyAddr();
        __ubuf__ float* x2Addr = (__ubuf__ float*)inputX2.GetPhyAddr();
        __ubuf__ float* rstdAddr = (__ubuf__ float*)inputRstd.GetPhyAddr();
        __ubuf__ float* meanAddr = (__ubuf__ float*)inputMean.GetPhyAddr();
        __ubuf__ float* gammaAddr = (__ubuf__ float*)inputGamma.GetPhyAddr();
        __ubuf__ float* dgammaAddr = (__ubuf__ float*)outputDgamma.GetPhyAddr();
        __ubuf__ float* dbetaAddr = (__ubuf__ float*)outputDbeta.GetPhyAddr();
        __ubuf__ float* tmpVarAddr = (__ubuf__ float*)tmpVarPdLocal.GetPhyAddr();
        __ubuf__ float* tmpMeanAddr = (__ubuf__ float*)tmpMeanPdLocal.GetPhyAddr();
        __ubuf__ float* partialAddr = (__ubuf__ float*)outputDx.GetPhyAddr();

        uint16_t colLoopTimes = static_cast<uint16_t>((elemCoutDXy + V_LENGTH - 1) / V_LENGTH);
        float rstdNum = rstdAddr[0];
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

            DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg, meanAddr);
            DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr);
            Duplicate(rstd3Reg, rstd3Num, pregFull);

            for (uint16_t j = 0; j < colLoopTimes; j++) {
                uint32_t remainElem = elemCoutDXy - j * V_LENGTH;
                pregLoop = UpdateMask<float>(remainElem);

                DataCopy(dy, dyAddr + j * V_LENGTH);
                DataCopy(x1, x1Addr + j * V_LENGTH);
                DataCopy(x2, x2Addr + j * V_LENGTH);
                DataCopy(gamma, gammaAddr + j * V_LENGTH);

                Add(xSum, x1, x2, pregLoop);
                Mul(dyGamma, dy, gamma, pregLoop);

                Sub(xCentered, xSum, meanReg, pregLoop);
                Mul(tmp, xCentered, rstd3Reg, pregLoop);
                Mul(tmp, dyGamma, tmp, pregLoop);
                // Keep the per-element sequence intact; ReduceSumCustom below must see
                // the same contiguous values as the generic CutD implementation.
                CopyToTensor(x1Addr + j * V_LENGTH, tmp, pregLoop);

                Mul(tmp, xCentered, rstdReg, pregLoop);
                Mul(tmp, dy, tmp, pregLoop);
                CopyToTensor(dgammaAddr + j * V_LENGTH, tmp, pregLoop);

                CopyToTensor(dbetaAddr + j * V_LENGTH, dy, pregLoop);

                Mul(tmp, dyGamma, rstdReg, pregLoop);
                Neg(tmp, tmp, pregLoop);
                CopyToTensor(x2Addr + j * V_LENGTH, tmp, pregLoop);
            }
        }
        PipeBarrier<PIPE_V>();
        float pdVar = ReduceSumCustom(inputX1, elemCoutDXy);
        float pdMean = ReduceSumCustom(inputX2, elemCoutDXy);
        partialAddr[0] = pdVar;
        partialAddr[1] = pdMean;
    }

    __aicore__ inline void MicroComputeSecondPart(
        const LocalTensor<float>& inputDy, const LocalTensor<float>& inputX1, const LocalTensor<float>& inputX2,
        const LocalTensor<float>& inputRstd, const LocalTensor<float>& inputMean, const LocalTensor<float>& inputGamma,
        const LocalTensor<float>& inputDx, const LocalTensor<float>& outputDx, const LocalTensor<float>& tmpVarPdLocal,
        const LocalTensor<float>& tmpMeanPdLocal, const uint32_t elemCoutDXy)
    {
        __ubuf__ float* dyAddr = (__ubuf__ float*)inputDy.GetPhyAddr();
        __ubuf__ float* x1Addr = (__ubuf__ float*)inputX1.GetPhyAddr();
        __ubuf__ float* x2Addr = (__ubuf__ float*)inputX2.GetPhyAddr();
        __ubuf__ float* rstdAddr = (__ubuf__ float*)inputRstd.GetPhyAddr();
        __ubuf__ float* meanAddr = (__ubuf__ float*)inputMean.GetPhyAddr();
        __ubuf__ float* gammaAddr = (__ubuf__ float*)inputGamma.GetPhyAddr();
        __ubuf__ float* dxAddr = (__ubuf__ float*)outputDx.GetPhyAddr();
        __ubuf__ float* dsumAddr = (__ubuf__ float*)inputDx.GetPhyAddr();
        __ubuf__ float* pdVarAddr = (__ubuf__ float*)tmpVarPdLocal.GetPhyAddr();
        __ubuf__ float* pdMeanAddr = (__ubuf__ float*)tmpMeanPdLocal.GetPhyAddr();

        float reduceAxisSize = (selfTiling.numLastDim != 0) ? 1.0f / selfTiling.numLastDim : 0.0f;

        uint16_t colLoopTimes = static_cast<uint16_t>((elemCoutDXy + V_LENGTH - 1) / V_LENGTH);
        float pdVarScale = pdVarAddr[0] * (-reduceAxisSize);
        float pdMeanScale = pdMeanAddr[0] * reduceAxisSize;

        __VEC_SCOPE__
        {
            RegTensor<float> dy;
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> gamma;
            RegTensor<float> xSum;
            RegTensor<float> dyGamma;
            RegTensor<float> xCentered;
            RegTensor<float> dxResult;
            RegTensor<float> tmp;
            RegTensor<float> meanReg;
            RegTensor<float> rstdReg;
            RegTensor<float> pdVarReg;
            RegTensor<float> pdMeanReg;
            RegTensor<float> reduceAxisReg;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregLoop;

            DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg, meanAddr);
            DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr);
            // Match the generic scalar expression order exactly.
            Duplicate(pdMeanReg, pdMeanScale, pregFull);

            for (uint16_t j = 0; j < colLoopTimes; j++) {
                uint32_t remainElem = elemCoutDXy - j * V_LENGTH;
                pregLoop = UpdateMask<float>(remainElem);

                DataCopy(dy, dyAddr + j * V_LENGTH);
                DataCopy(x1, x1Addr + j * V_LENGTH);
                DataCopy(x2, x2Addr + j * V_LENGTH);
                DataCopy(gamma, gammaAddr + j * V_LENGTH);

                Add(xSum, x1, x2, pregLoop);
                Mul(dyGamma, dy, gamma, pregLoop);
                Sub(xCentered, xSum, meanReg, pregLoop);

                Muls(tmp, xCentered, pdVarScale, pregLoop);
                Mul(dxResult, dyGamma, rstdReg, pregLoop);
                Add(dxResult, tmp, dxResult, pregLoop);
                Add(dxResult, dxResult, pdMeanReg, pregLoop);

                if constexpr (HAS_ADDITIONAL_INPUT) {
                    RegTensor<float> dSum;
                    DataCopy(dSum, dsumAddr + j * V_LENGTH);
                    Add(dxResult, dxResult, dSum, pregLoop);
                }

                DataCopy(dxAddr + j * V_LENGTH, dxResult, pregLoop);
            }
        }
    }

    __aicore__ inline void CopyOutBetaGamma(const int32_t offsetUbGamma, const int32_t elemCoutDXy)
    {
        LocalTensor<float> dGammaLocal = dGammaQue.DeQue<float>();
        LocalTensor<float> dBetaLocal = dBetaQue.DeQue<float>();

        SetAtomicAdd<float>();
        if (selfTiling.isDeterministicKey) {
            DataCopyAutomicAdd(workspaceGmGamma, dGammaLocal, elemCoutDXy, offsetUbGamma, (uint16_t)1);
        } else {
            DataCopyAutomicAdd(dGammaGm, dGammaLocal, elemCoutDXy, offsetUbGamma, (uint16_t)1);
        }
        PipeBarrier<PIPE_ALL>();
        SetAtomicNone();

        SetAtomicAdd<float>();
        if (selfTiling.isDeterministicKey) {
            DataCopyAutomicAdd(workspaceGmBeta, dBetaLocal, elemCoutDXy, offsetUbGamma, (uint16_t)1);
        } else {
            DataCopyAutomicAdd(dBetaGm, dBetaLocal, elemCoutDXy, offsetUbGamma, (uint16_t)1);
        }
        PipeBarrier<PIPE_ALL>();
        SetAtomicNone();

        dGammaQue.FreeTensor(dGammaLocal);
        dBetaQue.FreeTensor(dBetaLocal);
    }

    __aicore__ inline void CopyOutX(const int32_t offsetUbXY, const int32_t dYInUb, const int32_t nInOnceUb)
    {
        LocalTensor<T> dXLocal = dXQue.DeQue<T>();
        DataCopyParams dxCopyParams{(uint16_t)nInOnceUb, (uint16_t)(dYInUb * sizeof(T)), 0, 0};
        DataCopyPad(dXGm[offsetUbXY], dXLocal, dxCopyParams);
        PipeBarrier<PIPE_ALL>();
        dXQue.FreeTensor(dXLocal);
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

    TBuf<TPosition::VECCALC> tmpMeanPdBuf;
    TBuf<TPosition::VECCALC> tmpVarPdBuf;
    TBuf<TPosition::VECCALC> dyFp32Buf;
    TBuf<TPosition::VECCALC> x1Fp32Buf;
    TBuf<TPosition::VECCALC> x2Fp32Buf;
    TBuf<TPosition::VECCALC> dgammaFp32Buf;
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
    GlobalTensor<T> dSumGm;
    GlobalTensor<T> dXGm;

    uint64_t roundUpNumLastDimFloatLen;
    uint32_t nInOneCore;
    uint32_t gmOneCoreElemXY;
    uint32_t blockNumber;
    uint32_t blockNumberTdtype;
    uint64_t deterministicWorkSpaceSize = 0;
    bool isComputedCore = false;
};
} // namespace AddLayerNormGrad

#endif // ADD_LAYER_NORM_GRAD_CUT_D_A35
