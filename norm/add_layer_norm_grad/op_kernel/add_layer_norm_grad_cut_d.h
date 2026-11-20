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
 * \file add_layer_norm_grad_cut_d.h
 * \brief
 */
#ifndef OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_LAYER_NORM_GRAD_ADD_LAYER_NORM_GRAD_CUT_D
#define OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_LAYER_NORM_GRAD_ADD_LAYER_NORM_GRAD_CUT_D
#include "add_layer_norm_grad_utils.h"
#include "add_layer_norm_determinstic_compute.h"

namespace AddLayerNormGrad {
using namespace AscendC;

template <typename T, int TILING_KEY>
class KernelAddLayerNormGradLarge {
#define HAS_ADDITIONAL_INPUT ((TILING_KEY % 10) == 1)
public:
    __aicore__ inline KernelAddLayerNormGradLarge() {}
    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x_1, GM_ADDR x_2, GM_ADDR rstd, GM_ADDR mean, GM_ADDR gamma,
                                GM_ADDR dsum, GM_ADDR d_x, GM_ADDR d_gamma, GM_ADDR d_beta,
                                const AddLayerNormGradTilingData tiling, GM_ADDR workspace)
    {
        InitVar(tiling);
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
            InitTmpBuffer(workspace);
            dXGm.SetGlobalBuffer((__gm__ T*)d_x + GetBlockIdx() * selfTiling.nInOneCoreLength * selfTiling.numLastDim,
                                 gmOneCoreElemXY);
        }
#if __CCE_AICORE__ == 220 || __CCE_AICORE__ == 310
        SyncAll();
#else
        uint32_t each_core_handle_num = BLOCK_AlIGN / sizeof(int32_t);
        syncGlobal_.SetGlobalBuffer(
            (__gm__ int32_t*)workspace + selfTiling.isDeterministicKey * deterministicWorkSpaceSize,
            GetBlockNum() * FLOAT_BLOCK_ELEM);
        syncGlobal1_.SetGlobalBuffer((__gm__ int32_t*)workspace +
                                         selfTiling.isDeterministicKey * deterministicWorkSpaceSize +
                                         GetBlockNum() * FLOAT_BLOCK_ELEM,
                                     GetBlockNum() * FLOAT_BLOCK_ELEM);

        LocalTensor<int32_t> tmp_init_buf = dBetaQue.AllocTensor<int32_t>();
        Duplicate(tmp_init_buf, 0, each_core_handle_num);
        DataCopy(syncGlobal_[each_core_handle_num * GetBlockIdx()], tmp_init_buf, each_core_handle_num);
        DataCopy(syncGlobal1_[each_core_handle_num * GetBlockIdx()], tmp_init_buf, each_core_handle_num);
        LocalTensor<int32_t> workLocal = dGammaQue.AllocTensor<int32_t>();
        SyncAll(syncGlobal_, workLocal);
        dGammaQue.FreeTensor(workLocal);
        dBetaQue.FreeTensor(tmp_init_buf);
#endif
    }
    __aicore__ inline void InitVar(AddLayerNormGradTilingData tiling)
    {
        selfTiling = tiling;
        roundUpNumLastDimFloatLen = selfTiling.roundUpNumLastDimFloat / sizeof(float);

        nInOneCore = (GetBlockIdx() != selfTiling.numCore - 1) ? selfTiling.nInOneCoreLength :
                                                                 selfTiling.nInOneCoreLengthTail;
        gmOneCoreElemXY = nInOneCore * selfTiling.numLastDim;

        blockNumber = BLOCK_AlIGN / sizeof(float);
        if constexpr (!is_same<T, float>::value) {
            blockNumberTdtype = BLOCK_AlIGN / sizeof(half);
        } else {
            blockNumberTdtype = BLOCK_AlIGN / sizeof(float);
        }

        if (GetBlockIdx() < selfTiling.numCore) {
            isComputedCore = true;
        }
    }

    __aicore__ inline void InitInputGmBuffer(GM_ADDR dy, GM_ADDR x_1, GM_ADDR x_2, GM_ADDR rstd, GM_ADDR mean,
                                             GM_ADDR gamma, GM_ADDR dsum)
    {
        dyGm.SetGlobalBuffer((__gm__ T*)dy + GetBlockIdx() * selfTiling.nInOneCoreLength * selfTiling.numLastDim,
                             gmOneCoreElemXY);
        x1Gm.SetGlobalBuffer((__gm__ T*)x_1 + GetBlockIdx() * selfTiling.nInOneCoreLength * selfTiling.numLastDim,
                             gmOneCoreElemXY);
        x2Gm.SetGlobalBuffer((__gm__ T*)x_2 + GetBlockIdx() * selfTiling.nInOneCoreLength * selfTiling.numLastDim,
                             gmOneCoreElemXY);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + GetBlockIdx() * selfTiling.nInOneCoreLength, nInOneCore);
        meanGm.SetGlobalBuffer((__gm__ float*)mean + GetBlockIdx() * selfTiling.nInOneCoreLength, nInOneCore);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma, selfTiling.numLastDim);
        if constexpr (HAS_ADDITIONAL_INPUT) {
            dSumGm.SetGlobalBuffer(
                (__gm__ T*)dsum + GetBlockIdx() * selfTiling.nInOneCoreLength * selfTiling.numLastDim, gmOneCoreElemXY);
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
        LocalTensor<float> temp_local_tensor = dGammaQue.AllocTensor<float>();
        InitGmData(workspaceGmGamma, workspaceGmBeta, roundUpNumLastDimFloatLen, temp_local_tensor,
                   ROUND_UP(selfTiling.dInnerLength, blockNumber));
        dGammaQue.FreeTensor(temp_local_tensor);
    }

    __aicore__ inline void InitOuputGmBuffer(GM_ADDR d_gamma, GM_ADDR d_beta)
    {
        dGammaGm.SetGlobalBuffer((__gm__ float*)d_gamma, selfTiling.numLastDim);
        dBetaGm.SetGlobalBuffer((__gm__ float*)d_beta, selfTiling.numLastDim);
        if (GetBlockIdx() == 0) {
            LocalTensor<float> temp_local_tensor = dGammaQue.AllocTensor<float>();
            InitGmData(dGammaGm, dBetaGm, selfTiling.numLastDim, temp_local_tensor,
                       ROUND_UP(selfTiling.dInnerLength, blockNumber));
            dGammaQue.FreeTensor(temp_local_tensor);
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

    __aicore__ inline void InitTmpBuffer(GM_ADDR workspace)
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
        if (isComputedCore) {
            LocalTensor<float> tmpMeanPdLocal = tmpMeanPdBuf.Get<float>();
            LocalTensor<float> tmpVarPdLocal = tmpVarPdBuf.Get<float>();
            LocalTensor<float> dyFp32Local;
            LocalTensor<float> x1Fp32Local;
            LocalTensor<float> x2Fp32Local;
            LocalTensor<float> gammaFp32Local;
            LocalTensor<float> dXLocal;
            LocalTensor<float> dSumFp32Local;
#if __CCE_AICORE__ == 220 || __CCE_AICORE__ == 310
            if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
#else
            if constexpr (is_same<T, half>::value) {
#endif
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
                Duplicate(tmpMeanPdLocal, 0.0f, blockNumber);
                Duplicate(tmpVarPdLocal, 0.0f, blockNumber);
                PipeBarrier<PIPE_V>();
                for (int32_t DOuterUbIndex = 0; DOuterUbIndex < selfTiling.dOuterLength; ++DOuterUbIndex) {
                    uint32_t DInOnceUb = (DOuterUbIndex != selfTiling.dOuterLength - 1) ? selfTiling.dInnerLength :
                                                                                          selfTiling.dInnerLengthTail;
                    uint32_t offsetUbXY = DOuterUbIndex * selfTiling.dInnerLength + nInnerIndex * selfTiling.numLastDim;
                    uint32_t offsetUbMeanVar = nInnerIndex;
                    uint32_t offsetUbGamma = DOuterUbIndex * selfTiling.dInnerLength;
                    uint32_t elemCoutXYUb = DInOnceUb;

                    CopyIn(offsetUbXY, offsetUbMeanVar, elemCoutXYUb, 1, selfTiling.nAvailInUb, offsetUbGamma);
                    event_t event_mte2_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                    SetFlag<HardEvent::MTE2_V>(event_mte2_v);
                    WaitFlag<HardEvent::MTE2_V>(event_mte2_v);
                    LocalTensor<T> inputDy = dyQue.DeQue<T>();
                    LocalTensor<T> inputX1 = x1Que.DeQue<T>();
                    LocalTensor<T> inputX2 = x2Que.DeQue<T>();
                    LocalTensor<float> inputRstd = rstdQue.DeQue<float>();
                    LocalTensor<float> inputMean = meanQue.DeQue<float>();
                    LocalTensor<T> inputGamma = gammaQue.DeQue<T>();

                    LocalTensor<T> outputDx = dXQue.AllocTensor<T>();
                    LocalTensor<float> outputDgamma = dGammaQue.AllocTensor<float>();
                    LocalTensor<float> outputDbeta = dBetaQue.AllocTensor<float>();
#if __CCE_AICORE__ == 220 || __CCE_AICORE__ == 310
                    if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
#else
                    if constexpr (is_same<T, half>::value) {
#endif
                        Cast(dyFp32Local, inputDy, RoundMode::CAST_NONE, elemCoutXYUb);
                        Cast(x1Fp32Local, inputX1, RoundMode::CAST_NONE, elemCoutXYUb);
                        Cast(x2Fp32Local, inputX2, RoundMode::CAST_NONE, elemCoutXYUb);
                        Cast(gammaFp32Local, inputGamma, RoundMode::CAST_NONE, elemCoutXYUb);
                        MainComputeFirstPart(dyFp32Local, x1Fp32Local, x2Fp32Local, inputRstd, inputMean,
                                             gammaFp32Local, outputDgamma, outputDbeta, dXLocal, tmpVarPdLocal,
                                             tmpMeanPdLocal, elemCoutXYUb);
                    } else {
                        MainComputeFirstPart(inputDy, inputX1, inputX2, inputRstd, inputMean, inputGamma, outputDgamma,
                                             outputDbeta, outputDx, tmpVarPdLocal, tmpMeanPdLocal, elemCoutXYUb);
                    }
                    dyQue.FreeTensor(inputDy);
                    x1Que.FreeTensor(inputX1);
                    x2Que.FreeTensor(inputX2);
                    rstdQue.FreeTensor(inputRstd);
                    meanQue.FreeTensor(inputMean);
                    gammaQue.FreeTensor(inputGamma);
                    dXQue.FreeTensor(outputDx);
                    dGammaQue.EnQue(outputDgamma);
                    dBetaQue.EnQue(outputDbeta);
                    event_t event_v_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                    SetFlag<HardEvent::V_MTE3>(event_v_mte3);
                    WaitFlag<HardEvent::V_MTE3>(event_v_mte3);
                    CopyOutBetaGamma(offsetUbGamma, elemCoutXYUb);
                    PipeBarrier<PIPE_ALL>();
                }

                for (int32_t DOuterUbIndex = 0; DOuterUbIndex < selfTiling.dOuterLength; ++DOuterUbIndex) {
                    uint32_t DInOnceUb = (DOuterUbIndex != selfTiling.dOuterLength - 1) ? selfTiling.dInnerLength :
                                                                                          selfTiling.dInnerLengthTail;
                    uint32_t offsetUbXY = DOuterUbIndex * selfTiling.dInnerLength + nInnerIndex * selfTiling.numLastDim;
                    uint32_t offsetUbMeanVar = nInnerIndex;
                    uint32_t offsetUbGamma = DOuterUbIndex * selfTiling.dInnerLength;
                    uint32_t elemCoutXYUb = DInOnceUb;

                    CopyIn(offsetUbXY, offsetUbMeanVar, elemCoutXYUb, 1, selfTiling.nAvailInUb, offsetUbGamma, true);
                    event_t event_mte2_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                    SetFlag<HardEvent::MTE2_V>(event_mte2_v);
                    WaitFlag<HardEvent::MTE2_V>(event_mte2_v);
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
#if __CCE_AICORE__ == 220 || __CCE_AICORE__ == 310
                    if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
#else
                    if constexpr (is_same<T, half>::value) {
#endif
                        Cast(dyFp32Local, inputDy, RoundMode::CAST_NONE, elemCoutXYUb);
                        Cast(x1Fp32Local, inputX1, RoundMode::CAST_NONE, elemCoutXYUb);
                        Cast(x2Fp32Local, inputX2, RoundMode::CAST_NONE, elemCoutXYUb);
                        Cast(gammaFp32Local, inputGamma, RoundMode::CAST_NONE, elemCoutXYUb);
                        if constexpr (HAS_ADDITIONAL_INPUT) {
                            Cast(dSumFp32Local, inputDx, RoundMode::CAST_NONE, elemCoutXYUb);
                        }
                        PipeBarrier<PIPE_V>();
                        MainComputeSecondPart(dyFp32Local, x1Fp32Local, x2Fp32Local, inputRstd, inputMean,
                                              gammaFp32Local, dSumFp32Local, dXLocal, tmpVarPdLocal, tmpMeanPdLocal,
                                              elemCoutXYUb);
                        if constexpr (is_same<T, half>::value) {
                            Cast(outputDx, dXLocal, RoundMode::CAST_NONE, elemCoutXYUb);
                            PipeBarrier<PIPE_V>();
                        } else {
                            Cast(outputDx, dXLocal, RoundMode::CAST_RINT, elemCoutXYUb);
                            PipeBarrier<PIPE_V>();
                        }
                    } else {
                        MainComputeSecondPart(inputDy, inputX1, inputX2, inputRstd, inputMean, inputGamma, inputDx,
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
                    event_t event_v_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                    SetFlag<HardEvent::V_MTE3>(event_v_mte3);
                    WaitFlag<HardEvent::V_MTE3>(event_v_mte3);
                    CopyOutX(offsetUbXY, elemCoutXYUb, selfTiling.nAvailInUb);
                }
            }
        }
        if (selfTiling.isDeterministicKey) {
#if __CCE_AICORE__ == 220 || __CCE_AICORE__ == 310
            SyncAll();
#else
            LocalTensor<int32_t> workLocal = dGammaQue.AllocTensor<int32_t>();
            SyncAll(syncGlobal1_, workLocal);
            dGammaQue.FreeTensor(workLocal);
#endif
            pipe.Reset();
            AddLayerNormGradDeterminsticCompute op;
            op.initBuffer(pipe, dGammaGm, dBetaGm, workspaceGMOri, CONSTANT_TWO);
            op.FinalProcessDeterministic(roundUpNumLastDimFloatLen, selfTiling.numCore, selfTiling.numLastDim);
        }
    }

private:
#include "add_layer_norm_grad_cut_d_part1.h"

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
    GlobalTensor<int32_t> syncGlobal_;
    GlobalTensor<int32_t> syncGlobal1_;

    uint64_t roundUpNumLastDimFloatLen;

    uint32_t nInOneCore;
    uint32_t gmOneCoreElemXY;
    uint32_t blockNumber;
    uint32_t blockNumberTdtype;
    uint64_t deterministicWorkSpaceSize = 0;
    bool isComputedCore = false;
};
} // namespace AddLayerNormGrad

#endif // OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_LAYER_NORM_GRAD_ADD_LAYER_NORM_GRAD_CUT_D
