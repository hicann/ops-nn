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
 * \file add_layer_norm_grad_cut_n.h
 * \brief
 */
#ifndef OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_LAYER_NORM_GRAD_ADD_LAYER_NORM_GRAD_CUT_N
#define OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_LAYER_NORM_GRAD_ADD_LAYER_NORM_GRAD_CUT_N
#include "add_layer_norm_grad_utils.h"
#include "add_layer_norm_determinstic_compute.h"

namespace AddLayerNormGrad {
using namespace AscendC;

template <typename T, int TILING_KEY>
class KernelAddLayerNormGrad {
#define HAS_ADDITIONAL_INPUT ((TILING_KEY % 10) == 1)
public:
    __aicore__ inline KernelAddLayerNormGrad()
    {}

    __aicore__ inline void Init(
        GM_ADDR dy, GM_ADDR x_1, GM_ADDR x_2, GM_ADDR rstd, GM_ADDR mean, GM_ADDR gamma, GM_ADDR dsum, GM_ADDR d_x,
        GM_ADDR d_gamma, GM_ADDR d_beta, const AddLayerNormGradTilingData tiling, GM_ADDR workspace)
    {
        InitVar(tiling);
        if(isDeterministicKey) {
            deterministicWorkSpaceSize = roundUpNumLastDimFloatLen * CONSTANT_TWO * selfTiling.numCore;
            workspaceGMOri.SetGlobalBuffer((__gm__ float*)workspace, deterministicWorkSpaceSize);
        }
#if __CCE_AICORE__ != 220 && __CCE_AICORE__ != 310
        if (selfTiling.numLastDim < BLOCK_AlIGN / sizeof(T)) {
            if (GetBlockIdx() == 0) {
                GlobalTensor<T> dXGmAll;
                dXGmAll.SetGlobalBuffer((__gm__ T *)d_x, selfTiling.numLastDim * selfTiling.numFirstDim);
                uint32_t fullAlign = selfTiling.numFirstDim * selfTiling.numLastDim * sizeof(T) / FULL_ALIGN_BLOCK * FULL_ALIGN_BLOCK / sizeof(T);
                if(fullAlign != 0) {
                    InitGlobalMemory(dXGmAll, fullAlign, static_cast<T>(0.0));
                }
                for(uint32_t i = 0; i < selfTiling.numLastDim * selfTiling.numFirstDim - fullAlign; i++) {
                    dXGmAll.SetValue(i + fullAlign, static_cast<T>(0.0));
                }
                DataCacheCleanAndInvalid<T, AscendC::CacheLine::SINGLE_CACHE_LINE>(dXGmAll[fullAlign]);
                PipeBarrier<PIPE_ALL>();
            }
        }
#endif
        InitOutputQueue();
        InitOuputGmBuffer(d_gamma, d_beta);
        if(isComputedCore) {
            if(isDeterministicKey) {
                InitWorkspaceGmBuffer(workspace);
            }
            InitInputGmBuffer(dy, x_1, x_2, rstd, mean, gamma, dsum);
            InitInputQueue();
            dXGm.SetGlobalBuffer((__gm__ T *)d_x + GetBlockIdx() * selfTiling.ndInOneCoreLength, gmOneCoreElemXY);
            InitTmpBuffer(workspace);
        }
#if __CCE_AICORE__ == 220 || __CCE_AICORE__ == 310
        SyncAll();
#else
        uint32_t each_core_handle_num = BLOCK_AlIGN / sizeof(int32_t);
        syncGlobal1_.SetGlobalBuffer((__gm__ int32_t*)workspace + isDeterministicKey * deterministicWorkSpaceSize + GetBlockNum() * FLOAT_BLOCK_ELEM, GetBlockNum() * FLOAT_BLOCK_ELEM);
        syncGlobal_.SetGlobalBuffer((__gm__ int32_t*)workspace + isDeterministicKey * deterministicWorkSpaceSize, GetBlockNum() * FLOAT_BLOCK_ELEM);

        LocalTensor<int32_t> tmp_init_buf = dBetaQue.AllocTensor<int32_t>();
        Duplicate(tmp_init_buf, 0, each_core_handle_num);
        DataCopy(syncGlobal1_[each_core_handle_num * GetBlockIdx()], tmp_init_buf, each_core_handle_num);
        DataCopy(syncGlobal_[each_core_handle_num * GetBlockIdx()], tmp_init_buf, each_core_handle_num);
        LocalTensor<int32_t> workLocal = dGammaQue.AllocTensor<int32_t>();
        SyncAll(syncGlobal_, workLocal);
        dBetaQue.FreeTensor(tmp_init_buf);
        dGammaQue.FreeTensor(workLocal);
#endif
    }

    __aicore__ inline void InitVar(AddLayerNormGradTilingData tiling)
    {
        selfTiling = tiling;
        roundUpNumLastDimFloatLen = selfTiling.roundUpNumLastDimFloat / sizeof(float);
        isDeterministicKey = tiling.isDeterministicKey;
        if (GetBlockIdx() != selfTiling.numCore - 1) {
            nInOneCore = tiling.nInOneCoreNorm;
            gmOneCoreElemXY = tiling.gmOneCoreElemXYNorm;
            nAvailInUb = tiling.nAvailInUbNorm;
            nMiddleCount = tiling.nMiddleCountNorm;
            nInUbTotalTail = tiling.nInUbTotalNormTail;
            nDRoundUpDtype = tiling.ndRoundUpDtypeNorm;
            n1RoundUpFloat = tiling.n1RoundUpFloatNorm;
        } else {
            nInOneCore = tiling.nInOneCoreTail;
            gmOneCoreElemXY = tiling.gmOneCoreElemXYTail;
            nAvailInUb = tiling.nAvailInUbTail;
            nMiddleCount = tiling.nMiddleCountTail;
            nInUbTotalTail = tiling.nInUbTotalTailTail;
            nDRoundUpDtype = tiling.ndRoundUpDtypeTail;
            n1RoundUpFloat = tiling.n1RoundUpFloatTail;
        }

        blockNumber = BLOCK_AlIGN / sizeof(float);

#if __CCE_AICORE__ == 220 || __CCE_AICORE__ == 310
        if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
#else
        if constexpr (is_same<T, half>::value) {
#endif
            blockNumberTdtype1 = BLOCK_AlIGN / sizeof(half);
        } else {
            blockNumberTdtype1 = BLOCK_AlIGN / sizeof(float);
        }
        eachCoreHandleNum = BLOCK_AlIGN / sizeof(int32_t);

        if(GetBlockIdx() < selfTiling.numCore) {
            isComputedCore = true;
        }
    }

    __aicore__ inline void InitInputGmBuffer(
        GM_ADDR dy, GM_ADDR x_1, GM_ADDR x_2, GM_ADDR rstd, GM_ADDR mean, GM_ADDR gamma, GM_ADDR dsum)
    {
        dyGm.SetGlobalBuffer((__gm__ T*)dy + GetBlockIdx() * selfTiling.ndInOneCoreLength, gmOneCoreElemXY);
        x1Gm.SetGlobalBuffer((__gm__ T*)x_1 + GetBlockIdx() * selfTiling.ndInOneCoreLength, gmOneCoreElemXY);
        x2Gm.SetGlobalBuffer((__gm__ T*)x_2 + GetBlockIdx() * selfTiling.ndInOneCoreLength, gmOneCoreElemXY);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + GetBlockIdx() * selfTiling.nInOneCoreLength, nInOneCore);
        meanGm.SetGlobalBuffer((__gm__ float*)mean + GetBlockIdx() * selfTiling.nInOneCoreLength, nInOneCore);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma, selfTiling.numLastDim);
        if constexpr (HAS_ADDITIONAL_INPUT) {
            dSumGm.SetGlobalBuffer((__gm__ T*)dsum + GetBlockIdx() * selfTiling.ndInOneCoreLength, gmOneCoreElemXY);
        }
    }

    __aicore__ inline void InitWorkspaceGmBuffer(GM_ADDR workspace) {
        workspaceGmGamma.SetGlobalBuffer((__gm__ float*)workspace + GetBlockIdx() * CONSTANT_TWO * roundUpNumLastDimFloatLen, roundUpNumLastDimFloatLen);
        workspaceGmBeta.SetGlobalBuffer((__gm__ float*)workspace + (1 + GetBlockIdx() * CONSTANT_TWO) * roundUpNumLastDimFloatLen, roundUpNumLastDimFloatLen);
        LocalTensor<float> temp_local_tensor = dGammaQue.AllocTensor<float>();
        InitGmData(workspaceGmGamma, workspaceGmBeta, roundUpNumLastDimFloatLen, temp_local_tensor, selfTiling.roundUpNumLastDimFloat);
        dGammaQue.FreeTensor(temp_local_tensor);
    }

    __aicore__ inline void InitOuputGmBuffer(GM_ADDR d_gamma, GM_ADDR d_beta)
    {
        dGammaGm.SetGlobalBuffer((__gm__ float*)d_gamma, selfTiling.numLastDim);
        dBetaGm.SetGlobalBuffer((__gm__ float*)d_beta, selfTiling.numLastDim);
        if (GetBlockIdx() == 0) {
            LocalTensor<float> temp_local_tensor = dGammaQue.AllocTensor<float>();
            InitGmData(dGammaGm, dBetaGm, selfTiling.numLastDim, temp_local_tensor, selfTiling.roundUpNumLastDimFloat);
            dGammaQue.FreeTensor(temp_local_tensor);
        }
    }

    __aicore__ inline void InitInputQueue()
    {
        pipe.InitBuffer(dyQue, BUFFER_NUM, nDRoundUpDtype);
        pipe.InitBuffer(x1Que, BUFFER_NUM, nDRoundUpDtype);
        pipe.InitBuffer(x2Que, BUFFER_NUM, nDRoundUpDtype);
        pipe.InitBuffer(rstdQue, BUFFER_NUM, n1RoundUpFloat);
        pipe.InitBuffer(meanQue, BUFFER_NUM, n1RoundUpFloat);
        pipe.InitBuffer(GammaQue, BUFFER_NUM, selfTiling.roundUpNumLastDimDtype);
        if constexpr (HAS_ADDITIONAL_INPUT) {
            pipe.InitBuffer(dSumQue, BUFFER_NUM, nDRoundUpDtype);
        }
    }

    __aicore__ inline void InitOutputQueue()
    {
        pipe.InitBuffer(dXQue, BUFFER_NUM, nDRoundUpDtype);
        pipe.InitBuffer(dGammaQue, BUFFER_NUM, selfTiling.roundUpNumLastDimFloat);
        pipe.InitBuffer(dBetaQue, BUFFER_NUM, selfTiling.roundUpNumLastDimFloat);
    }

    __aicore__ inline void InitTmpBuffer(GM_ADDR workspace)
    {
        if constexpr (is_same<T, float>::value) {
        } else {
            pipe.InitBuffer(dyFp32Buf, selfTiling.roundUpNumLastDimFloat);
            pipe.InitBuffer(x1Fp32Buf, selfTiling.roundUpNumLastDimFloat);
            pipe.InitBuffer(x2Fp32Buf, selfTiling.roundUpNumLastDimFloat);
            pipe.InitBuffer(gammaFp32Buf, selfTiling.roundUpNumLastDimFloat);
            pipe.InitBuffer(dXFp32Buf, selfTiling.roundUpNumLastDimFloat);
            if constexpr (HAS_ADDITIONAL_INPUT) {
                pipe.InitBuffer(dSumFp32Buf, selfTiling.roundUpNumLastDimFloat);
            }
        }
    }

    __aicore__ inline void CutNProcess()
    {
        if(isComputedCore){
            CopyInGamma(selfTiling.numLastDim, selfTiling.dyPadRight);
            LocalTensor<T> inputGamma = GammaQue.DeQue<T>();
            float reduceAxisSize = (float)1.0 / selfTiling.numLastDim;

            for (int32_t NOuterUbIndex = 0; NOuterUbIndex < nMiddleCount; ++NOuterUbIndex) {
                uint32_t nInOnceUb = (NOuterUbIndex != nMiddleCount - 1) ? nAvailInUb : nInUbTotalTail;
                uint32_t offsetUbXY = NOuterUbIndex * nAvailInUb * selfTiling.numLastDim;
                uint32_t offsetUbMeanVar = NOuterUbIndex * nAvailInUb;
                uint32_t DRstdInUb = 1;

                CopyIn(offsetUbXY, offsetUbMeanVar, selfTiling.numLastDim, DRstdInUb, nInOnceUb, selfTiling.dyPadRight, selfTiling.rstdPadRight);
                PrecisionCompute(nInOnceUb, inputGamma, reduceAxisSize);
                CopyOut(offsetUbXY, selfTiling.numLastDim, nInOnceUb);
            }
            GammaQue.FreeTensor(inputGamma);
        }
        if(isDeterministicKey) {
#if __CCE_AICORE__ == 220 || __CCE_AICORE__ == 310
            SyncAll();
#else
            LocalTensor<int32_t> workLocal1 = dGammaQue.AllocTensor<int32_t>();
            SyncAll(syncGlobal1_, workLocal1);
            dGammaQue.FreeTensor(workLocal1);
#endif
            pipe.Reset();
            AddLayerNormGradDeterminsticCompute op1;
            op1.initBuffer(pipe, dGammaGm, dBetaGm, workspaceGMOri, CONSTANT_TWO);
            op1.FinalProcessDeterministic(roundUpNumLastDimFloatLen, selfTiling.numCore, selfTiling.numLastDim);
        }
    }

private:
    __aicore__ inline void CopyInGamma(const int32_t d_y_in_ub, const int32_t dyPadRight)
    {
        LocalTensor<T> gammaLocal = GammaQue.AllocTensor<T>();
#if __CCE_AICORE__ == 220 || __CCE_AICORE__ == 310
        DataCopyParams gamma_data_copy_params = {1, (uint16_t)(d_y_in_ub * sizeof(T)), 0, 0};
        DataCopyPadParams dy_pad_params{true, 0, (uint8_t)selfTiling.dyPadRight, 0};
        DataCopyPad(gammaLocal, gammaGm[0], gamma_data_copy_params, dy_pad_params);
#else
        DataCopy(gammaLocal, gammaGm[0], ROUND_UP(d_y_in_ub, blockNumberTdtype1));
#endif
        GammaQue.EnQue(gammaLocal);
    }

    __aicore__ inline void CopyIn(
        const int32_t offsetUbXY, const int32_t offsetUbMeanVar, const int32_t d_y_in_ub, const int32_t DRstdInUb,
        const int32_t nInOnceUb, const int32_t dyPadRight, const int32_t rstdPadRight)
    {
        // AllocTensor
        LocalTensor<T> dyLocal = dyQue.AllocTensor<T>();
        LocalTensor<T> x1Local = x1Que.AllocTensor<T>();
        LocalTensor<T> x2Local = x2Que.AllocTensor<T>();
        LocalTensor<float> rstdLocal = rstdQue.AllocTensor<float>();
        LocalTensor<float> meanLocal = meanQue.AllocTensor<float>();
        LocalTensor<T> dSumLocal;
        if constexpr (HAS_ADDITIONAL_INPUT) {
            dSumLocal = dSumQue.AllocTensor<T>();
        }
#if __CCE_AICORE__ == 220 || __CCE_AICORE__ == 310
        DataCopyParams dy_data_copy_params{(uint16_t)nInOnceUb, (uint16_t)(d_y_in_ub * sizeof(T)), 0, 0};
        DataCopyPadParams dy_pad_params{true, 0, (uint8_t)selfTiling.dyPadRight, 0};
        DataCopyParams rstd_data_copy_params{(uint16_t)nInOnceUb, (uint16_t)(DRstdInUb * sizeof(float)), 0, 0};
        DataCopyPadParams rstd_pad_params{true, 0, (uint8_t)selfTiling.rstdPadRight, 0};

        DataCopyPad(dyLocal, dyGm[offsetUbXY], dy_data_copy_params, dy_pad_params);
        DataCopyPad(x1Local, x1Gm[offsetUbXY], dy_data_copy_params, dy_pad_params);
        DataCopyPad(x2Local, x2Gm[offsetUbXY], dy_data_copy_params, dy_pad_params);

        DataCopyPad(rstdLocal, rstdGm[offsetUbMeanVar], rstd_data_copy_params, rstd_pad_params);
        DataCopyPad(meanLocal, meanGm[offsetUbMeanVar], rstd_data_copy_params, rstd_pad_params);

        if constexpr (HAS_ADDITIONAL_INPUT) {
            DataCopyPad(dSumLocal, dSumGm[offsetUbXY], dy_data_copy_params, dy_pad_params);
        }
#else
        for (uint32_t idx = 0; idx < nInOnceUb; idx++) {
            DataCopy(
                dyLocal[idx * ROUND_UP(d_y_in_ub, blockNumberTdtype1)], dyGm[offsetUbXY + idx * d_y_in_ub],
                ROUND_UP(d_y_in_ub, blockNumberTdtype1));
            DataCopy(
                x1Local[idx * ROUND_UP(d_y_in_ub, blockNumberTdtype1)], x1Gm[offsetUbXY + idx * d_y_in_ub],
                ROUND_UP(d_y_in_ub, blockNumberTdtype1));
            DataCopy(
                x2Local[idx * ROUND_UP(d_y_in_ub, blockNumberTdtype1)], x2Gm[offsetUbXY + idx * d_y_in_ub],
                ROUND_UP(d_y_in_ub, blockNumberTdtype1));

            DataCopy(
                rstdLocal[idx * ROUND_UP(DRstdInUb, blockNumber)], rstdGm[offsetUbMeanVar + idx * DRstdInUb],
                ROUND_UP(DRstdInUb, blockNumber));
            DataCopy(
                meanLocal[idx * ROUND_UP(DRstdInUb, blockNumber)], meanGm[offsetUbMeanVar + idx * DRstdInUb],
                ROUND_UP(DRstdInUb, blockNumber));

            if (HAS_ADDITIONAL_INPUT) {
                DataCopy(
                    dSumLocal[idx * ROUND_UP(d_y_in_ub, blockNumberTdtype1)], dSumGm[offsetUbXY + idx * d_y_in_ub],
                    ROUND_UP(d_y_in_ub, blockNumberTdtype1));
            }
        }
#endif

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

    __aicore__ inline void PrecisionCompute(
        const uint32_t nInOnceUb, const LocalTensor<T> inputGamma, const float reduceAxisSize)
    {
        LocalTensor<T> inputDy = dyQue.DeQue<T>();
        LocalTensor<T> inputX1 = x1Que.DeQue<T>();
        LocalTensor<T> inputX2 = x2Que.DeQue<T>();
        LocalTensor<float> inputRstd = rstdQue.DeQue<float>();
        LocalTensor<float> inputMean = meanQue.DeQue<float>();
        LocalTensor<T> dSumLocal;
        if constexpr (HAS_ADDITIONAL_INPUT) {
            dSumLocal = dSumQue.DeQue<T>();
        }

        LocalTensor<T> outputDx = dXQue.AllocTensor<T>();
        LocalTensor<float> outputDgamma = dGammaQue.AllocTensor<float>();
        LocalTensor<float> outputDbeta = dBetaQue.AllocTensor<float>();
#if __CCE_AICORE__ == 220 || __CCE_AICORE__ == 310
        Duplicate<float>(outputDgamma, 0.0, selfTiling.numLastDim);
        Duplicate<float>(outputDbeta, 0.0, selfTiling.numLastDim);
#else
        Duplicate<float>(outputDgamma, 0.0, ROUND_UP(selfTiling.numLastDim, blockNumber));
        Duplicate<float>(outputDbeta, 0.0, ROUND_UP(selfTiling.numLastDim, blockNumber));
#endif

        for (int32_t nInnerIndex = 0; nInnerIndex < nInOnceUb; ++nInnerIndex) {
            uint32_t offsetDXY = nInnerIndex * selfTiling.roundUpNumLastDim;
            uint32_t offsetDMeanVar = nInnerIndex * selfTiling.roundUp1Dtype;
#if __CCE_AICORE__ == 220 || __CCE_AICORE__ == 310
            if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
#else
            if constexpr (is_same<T, half>::value) {
#endif
                LocalTensor<float> dyFp32Local = dyFp32Buf.Get<float>();
                LocalTensor<float> x1Fp32Local = x1Fp32Buf.Get<float>();
                LocalTensor<float> x2Fp32Local = x2Fp32Buf.Get<float>();
                LocalTensor<float> gammaFp32Local = gammaFp32Buf.Get<float>();
                LocalTensor<float> dXLocal = dXFp32Buf.Get<float>();
                LocalTensor<float> dSumFp32Local;
                if constexpr (HAS_ADDITIONAL_INPUT) {
                    dSumFp32Local = dSumFp32Buf.Get<float>();
                    Cast(dSumFp32Local, dSumLocal[offsetDXY], RoundMode::CAST_NONE, selfTiling.numLastDim);
                }
                Cast(dyFp32Local, inputDy[offsetDXY], RoundMode::CAST_NONE, selfTiling.numLastDim);
                Cast(x1Fp32Local, inputX1[offsetDXY], RoundMode::CAST_NONE, selfTiling.numLastDim);
                Cast(x2Fp32Local, inputX2[offsetDXY], RoundMode::CAST_NONE, selfTiling.numLastDim);
                Cast(gammaFp32Local, inputGamma, RoundMode::CAST_NONE, selfTiling.numLastDim);
                PipeBarrier<PIPE_V>();

                MainCompute(
                    dyFp32Local, x1Fp32Local, x2Fp32Local, inputRstd[offsetDMeanVar], inputMean[offsetDMeanVar],
                    gammaFp32Local, dSumFp32Local, dXLocal, outputDgamma, outputDbeta, selfTiling.numLastDim, selfTiling.numLastDim,
                    reduceAxisSize);

                if constexpr (is_same<T, half>::value) {
                    Cast(outputDx[offsetDXY], dXLocal, RoundMode::CAST_NONE, selfTiling.numLastDim);
                } else {
                    Cast(outputDx[offsetDXY], dXLocal, RoundMode::CAST_RINT, selfTiling.numLastDim);
                }
            } else {
                MainCompute(
                    inputDy[offsetDXY], inputX1[offsetDXY], inputX2[offsetDXY], inputRstd[offsetDMeanVar],
                    inputMean[offsetDMeanVar], inputGamma, dSumLocal[offsetDXY], outputDx[offsetDXY], outputDgamma,
                    outputDbeta, selfTiling.numLastDim, selfTiling.numLastDim, reduceAxisSize);
            }
        }
        dyQue.FreeTensor(inputDy);
        x1Que.FreeTensor(inputX1);
        x2Que.FreeTensor(inputX2);
        rstdQue.FreeTensor(inputRstd);
        meanQue.FreeTensor(inputMean);
        if constexpr (HAS_ADDITIONAL_INPUT) {
            dSumQue.FreeTensor(dSumLocal);
        }
        dXQue.EnQue(outputDx);
        dGammaQue.EnQue(outputDgamma);
        dBetaQue.EnQue(outputDbeta);
    }

    __aicore__ inline void MainCompute(
        const LocalTensor<float>& inputDy, const LocalTensor<float>& inputX1, const LocalTensor<float>& inputX2,
        const LocalTensor<float>& inputRstd, const LocalTensor<float>& inputMean, const LocalTensor<float>& inputGamma,
        const LocalTensor<float>& input_dx, const LocalTensor<float>& outputDx, const LocalTensor<float>& outputDgamma,
        const LocalTensor<float>& outputDbeta, const uint32_t elem_cout_d_x_y, const uint32_t numLastDim,
        const float reduceAxisSize)
    {
        Add(inputX1, inputX1, inputX2, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();
        // 1. x1Tensor = inputDy * inputGamma
        Mul(inputX2, inputDy, inputGamma, elem_cout_d_x_y);

        // 2. x2Tensor = inputX - inputMean
        event_t event_mte2_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(event_mte2_s);
        WaitFlag<HardEvent::MTE2_S>(event_mte2_s);
        float inputMeanNum = inputMean.GetValue(0);
        float inputRstdNum = inputRstd.GetValue(0);
        float tmpLocalNum = inputRstdNum * inputRstdNum * inputRstdNum;
        event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);
        WaitFlag<HardEvent::S_V>(event_s_v);
        Adds(outputDx, inputX1, inputMeanNum * (-1.0f), elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();

        // 3. pd_var = np.sum((-0.5) * x1Tensor * x2Tensor * np.power(inputRstd, 3))
        // 3.1. duplicate
        Muls(inputX1, outputDx, tmpLocalNum, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();

        Mul(inputX1, inputX2, inputX1, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();

        // 3.2. pd_var = np.sum(res1)
        auto aveLocalTemp = ReduceSumCustom(inputX1, selfTiling.numLastDim);
        inputMeanNum = aveLocalTemp * -reduceAxisSize;
        event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);

        // 4. pd_mean = np.sum((-1.0) * x1Tensor * inputRstd)
        // output: gamma = x2Tensor * rstd * inputDy
        Muls(inputX1, outputDx, inputRstdNum, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();

        Mul(inputX1, inputX1, inputDy, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();
        Add(outputDgamma, outputDgamma, inputX1, elem_cout_d_x_y);
        Add(outputDbeta, outputDbeta, inputDy, elem_cout_d_x_y);

        // 4.1. res1 = (-1.0) * x1Tensor * rstd
        WaitFlag<HardEvent::S_V>(event_s_v);
        Muls(inputX1, outputDx, inputMeanNum, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();
        Muls(outputDx, inputX2, inputRstdNum, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();
        Muls(inputDy, outputDx, -1.0f, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();

        // 4.2. pd_mean = np.sum(res1)
        Add(inputX2, inputX1, outputDx, elem_cout_d_x_y);
        aveLocalTemp = ReduceSumCustom(inputDy, elem_cout_d_x_y);
        inputMeanNum = aveLocalTemp * reduceAxisSize;
        event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(event_s_v);

        // 5. d_x = x1Tensor * inputRstd +
        //           pd_var * (2.0 / reduceAxisSize) * x2Tensor +
        //           pd_mean * (1.0 / reduceAxisSize)
        // 5.1. res0 = x1Tensor * np.power((inputVariace + EPSLON), (-0.5)), already store in resForGamma
        // 5.2. res1 = pd_var*(2.0 / reduceAxisSize)*(x2Tensor)
        // 5.3. res2 = pd_mean*(1.0 / reduceAxisSize)
        // 5.4. d_x = res0 + res1 + res2
        WaitFlag<HardEvent::S_V>(event_s_v);
        Adds(outputDx, inputX2, inputMeanNum, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();
        if constexpr (HAS_ADDITIONAL_INPUT) {
            Add(outputDx, outputDx, input_dx, elem_cout_d_x_y);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CopyOut(const int32_t offsetUbXY, const int32_t d_y_in_ub, const int32_t nInOnceUb)
    {
        LocalTensor<T> outputDx = dXQue.DeQue<T>();
        LocalTensor<float> outputDgamma = dGammaQue.DeQue<float>();
        LocalTensor<float> outputDbeta = dBetaQue.DeQue<float>();
        PipeBarrier<PIPE_ALL>();
        DataCopyCustom<T>(dXGm, outputDx, d_y_in_ub, offsetUbXY, false, (uint16_t)nInOnceUb);

        SetAtomicAdd<float>();
        if(isDeterministicKey){
            DataCopyAutomicAdd(workspaceGmBeta, outputDbeta, selfTiling.numLastDim, 0, (uint16_t)1);
        } else {
            DataCopyAutomicAdd(dBetaGm, outputDbeta, selfTiling.numLastDim, 0, (uint16_t)1);
        }
        SetAtomicNone();

        SetAtomicAdd<float>();
        if(isDeterministicKey){
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
    TQue<QuePosition::VECIN, BUFFER_NUM> GammaQue;
    TQue<QuePosition::VECIN, BUFFER_NUM> dSumQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dXQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dGammaQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dBetaQue;

    TBuf<TPosition::VECCALC> dyFp32Buf;
    TBuf<TPosition::VECCALC> x1Fp32Buf;
    TBuf<TPosition::VECCALC> x2Fp32Buf;
    TBuf<TPosition::VECCALC> gammaFp32Buf;
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
    GlobalTensor<T> dXGmAll;
    GlobalTensor<T> dSumGm;
    GlobalTensor<int32_t> syncGlobal_;
    GlobalTensor<int32_t> syncGlobal1_;

    uint32_t nAvailInUb;
    uint32_t roundUpNumLastDimFloatLen;

    uint32_t nInOneCore;
    uint32_t nMiddleCount;
    uint32_t nInUbTotalTail;
    uint32_t nDRoundUpDtype;
    uint32_t n1RoundUpFloat;
    uint32_t gmOneCoreElemXY;
    uint32_t blockNumber;
    uint32_t blockNumberTdtype1;
    uint32_t eachCoreHandleNum;
    uint32_t isDeterministicKey;
    uint32_t deterministicWorkSpaceSize = 0;
    bool isComputedCore = false;
};
}

#endif // OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_LAYER_NORM_GRAD_ADD_LAYER_NORM_GRAD_CUT_N