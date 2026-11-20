/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Cut D gradient computation and tensor copies.
 * Include only from add_layer_norm_grad_cut_d.h. */

__aicore__ inline void CopyIn(const int32_t offsetUbXY, const int32_t offsetUbMeanVar, const int32_t d_y_in_ub,
                              const int32_t DRstdInUb, const int32_t n_in_once_ub, const int32_t offsetUbGamma,
                              const bool has_dsum = false)
{
    // AllocTensor
    LocalTensor<T> dyLocal = dyQue.AllocTensor<T>();
    LocalTensor<T> x1Local = x1Que.AllocTensor<T>();
    LocalTensor<T> x2Local = x2Que.AllocTensor<T>();
    LocalTensor<float> rstdLocal = rstdQue.AllocTensor<float>();
    LocalTensor<float> meanLocal = meanQue.AllocTensor<float>();
    LocalTensor<T> gammaLocal = gammaQue.AllocTensor<T>();
    LocalTensor<T> dSumLocal;
#if __CCE_AICORE__ == 220 || __CCE_AICORE__ == 310
    DataCopyParams dy_data_copy_params{(uint16_t)n_in_once_ub, (uint16_t)(d_y_in_ub * sizeof(T)), 0, 0};
    uint8_t dyPadRight = ROUND_UP(d_y_in_ub, blockNumberTdtype) - d_y_in_ub;
    DataCopyPadParams dy_pad_params{true, 0, dyPadRight, 0};
    DataCopyParams rstd_data_copy_params{(uint16_t)n_in_once_ub, (uint16_t)(DRstdInUb * sizeof(float)), 0, 0};
    uint8_t rstdPadRight = ROUND_UP(DRstdInUb, blockNumber) - DRstdInUb;
    DataCopyPadParams rstd_pad_params{true, 0, rstdPadRight, 0};

    DataCopyPad(dyLocal, dyGm[offsetUbXY], dy_data_copy_params, dy_pad_params);
    DataCopyPad(x1Local, x1Gm[offsetUbXY], dy_data_copy_params, dy_pad_params);
    DataCopyPad(x2Local, x2Gm[offsetUbXY], dy_data_copy_params, dy_pad_params);

    DataCopyPad(rstdLocal, rstdGm[offsetUbMeanVar], rstd_data_copy_params, rstd_pad_params);
    DataCopyPad(meanLocal, meanGm[offsetUbMeanVar], rstd_data_copy_params, rstd_pad_params);
    DataCopyParams gamma_data_copy_params = {1, (uint16_t)(d_y_in_ub * sizeof(T)), 0, 0};
    DataCopyPad(gammaLocal, gammaGm[offsetUbGamma], gamma_data_copy_params, dy_pad_params);
    if (HAS_ADDITIONAL_INPUT && has_dsum) {
        dSumLocal = dSumQue.AllocTensor<T>();
        DataCopyPad(dSumLocal, dSumGm[offsetUbXY], dy_data_copy_params, dy_pad_params);
    }
#else
    for (uint32_t idx = 0; idx < n_in_once_ub; idx++) {
        DataCopy(dyLocal[idx * ROUND_UP(d_y_in_ub, blockNumberTdtype)], dyGm[offsetUbXY + idx * d_y_in_ub],
                 ROUND_UP(d_y_in_ub, blockNumberTdtype));
        DataCopy(x1Local[idx * ROUND_UP(d_y_in_ub, blockNumberTdtype)], x1Gm[offsetUbXY + idx * d_y_in_ub],
                 ROUND_UP(d_y_in_ub, blockNumberTdtype));
        DataCopy(x2Local[idx * ROUND_UP(d_y_in_ub, blockNumberTdtype)], x2Gm[offsetUbXY + idx * d_y_in_ub],
                 ROUND_UP(d_y_in_ub, blockNumberTdtype));

        DataCopy(rstdLocal[idx * ROUND_UP(DRstdInUb, blockNumber)], rstdGm[offsetUbMeanVar + idx * DRstdInUb],
                 ROUND_UP(DRstdInUb, blockNumber));
        DataCopy(meanLocal[idx * ROUND_UP(DRstdInUb, blockNumber)], meanGm[offsetUbMeanVar + idx * DRstdInUb],
                 ROUND_UP(DRstdInUb, blockNumber));

        if (HAS_ADDITIONAL_INPUT && has_dsum) {
            dSumLocal = dSumQue.AllocTensor<T>();
            DataCopy(dSumLocal[idx * ROUND_UP(d_y_in_ub, blockNumberTdtype)], dSumGm[offsetUbXY + idx * d_y_in_ub],
                     ROUND_UP(d_y_in_ub, blockNumberTdtype));
        }
    }
    DataCopy(gammaLocal, gammaGm[offsetUbGamma], ROUND_UP(d_y_in_ub, blockNumberTdtype));

#endif

    PipeBarrier<PIPE_ALL>();
    dyQue.EnQue(dyLocal);
    x1Que.EnQue(x1Local);
    x2Que.EnQue(x2Local);
    rstdQue.EnQue(rstdLocal);
    meanQue.EnQue(meanLocal);
    gammaQue.EnQue(gammaLocal);
    if (HAS_ADDITIONAL_INPUT && has_dsum) {
        dSumQue.EnQue(dSumLocal);
    }
}

__aicore__ inline void MainComputeFirstPart(const LocalTensor<float>& inputDy, const LocalTensor<float>& inputX1,
                                            const LocalTensor<float>& inputX2, const LocalTensor<float>& inputRstd,
                                            const LocalTensor<float>& inputMean, const LocalTensor<float>& inputGamma,
                                            const LocalTensor<float>& outputDgamma,
                                            const LocalTensor<float>& outputDbeta, const LocalTensor<float>& outputDx,
                                            const LocalTensor<float>& tmpVarPdLocal,
                                            const LocalTensor<float>& tmpMeanPdLocal, const uint32_t elem_cout_d_x_y)
{
    Adds(outputDbeta, inputDy, 0.0f, elem_cout_d_x_y);
    // 1. x1Tensor = inputDy * inputGamma
    Add(inputX1, inputX1, inputX2, elem_cout_d_x_y);
    event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(event_v_s);
    Mul(inputGamma, inputDy, inputGamma, elem_cout_d_x_y);
    // 2. x2Tensor = inputX - inputMean
    WaitFlag<HardEvent::V_S>(event_v_s);
    float inputMeanNum = inputMean.GetValue(0);
    float inputRstdNum = inputRstd.GetValue(0);
    Duplicate<float>(outputDx, inputMeanNum, elem_cout_d_x_y);
    Duplicate<float>(inputX2, inputRstdNum, elem_cout_d_x_y);
    PipeBarrier<PIPE_V>();
    float tmpLocalNum = inputRstdNum * inputRstdNum * inputRstdNum;
    event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(event_s_v);
    WaitFlag<HardEvent::S_V>(event_s_v);
    Sub(outputDx, inputX1, outputDx, elem_cout_d_x_y);

    // 3. pd_var = np.sum((-0.5) * x1Tensor * x2Tensor * np.power(inputRstd, 3))
    // 3.1. duplicate
    Duplicate<float>(inputX1, tmpLocalNum, elem_cout_d_x_y);
    PipeBarrier<PIPE_V>();

    // 3.2. res1 = (-0.5) * x1Tensor * (x2Tensor) * res
    Mul(inputX1, outputDx, inputX1, elem_cout_d_x_y);
    PipeBarrier<PIPE_V>();
    Mul(inputX1, inputGamma, inputX1, elem_cout_d_x_y);
    PipeBarrier<PIPE_V>();

    // 3.3. pd_var = np.sum(res1)
    auto aveLocalTemp = ReduceSumCustom(inputX1, elem_cout_d_x_y);

    // 3.4. pd_mean = np.sum((-1.0) * x1Tensor * inputRstd)
    // gamma = x2Tensor * rstd * inputDy
    Mul(inputX1, outputDx, inputX2, elem_cout_d_x_y);
    PipeBarrier<PIPE_V>();
    Mul(outputDgamma, inputX1, inputDy, elem_cout_d_x_y);

    // 5.1. res1 = (-1.0) * x1Tensor * rstd
    Mul(inputX1, inputGamma, inputX2, elem_cout_d_x_y);
    PipeBarrier<PIPE_V>();
    Muls(inputX2, inputX1, -1.0f, elem_cout_d_x_y);
    PipeBarrier<PIPE_V>();

    // 5.2. pd_mean = np.sum(res1)
    auto aveLocalTemp2 = ReduceSumCustom(inputX2, elem_cout_d_x_y);

    Adds(tmpVarPdLocal, tmpVarPdLocal, aveLocalTemp, blockNumber);
    Adds(tmpMeanPdLocal, tmpMeanPdLocal, aveLocalTemp2, blockNumber);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void MainComputeSecondPart(const LocalTensor<float>& inputDy, const LocalTensor<float>& inputX1,
                                             const LocalTensor<float>& inputX2, const LocalTensor<float>& inputRstd,
                                             const LocalTensor<float>& inputMean, const LocalTensor<float>& inputGamma,
                                             const LocalTensor<float>& inputDx, const LocalTensor<float>& outputDx,
                                             const LocalTensor<float>& tmpVarPdLocal,
                                             const LocalTensor<float>& tmpMeanPdLocal, const uint32_t elem_cout_d_x_y)
{
    // 1. x1Tensor = inputDy * inputGamma
    Add(inputX1, inputX1, inputX2, elem_cout_d_x_y);
    Mul(inputGamma, inputDy, inputGamma, elem_cout_d_x_y);
    // 2. x2Tensor = inputX - inputMean
    PipeBarrier<PIPE_V>();
    Adds(inputX1, inputX1, -1.0f * inputMean.GetValue(0), elem_cout_d_x_y);
    Muls(inputDy, inputGamma, inputRstd.GetValue(0), elem_cout_d_x_y);
    PipeBarrier<PIPE_V>();

    // 5. d_x = x1Tensor * inputRstd +
    //           pd_var * (2.0 / reduceAxisSize) * x2Tensor +
    //           pd_mean * (1.0 / reduceAxisSize)
    float reduceAxisSize = 1.0f;
    if (selfTiling.numLastDim != 0) {
        reduceAxisSize = 1.0f / selfTiling.numLastDim;
    }
    float inputRstdNum = tmpVarPdLocal.GetValue(0) * -reduceAxisSize;
    Muls(inputX2, inputX1, inputRstdNum, elem_cout_d_x_y);
    // 5.3. res2 = pd_mean*(1.0 / reduceAxisSize)
    float inputMeanNum = tmpMeanPdLocal.GetValue(0) * reduceAxisSize;
    Duplicate<float>(inputGamma, inputMeanNum, elem_cout_d_x_y);
    PipeBarrier<PIPE_V>();
    event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(event_s_v);
    WaitFlag<HardEvent::S_V>(event_s_v);

    // 5.4. d_x = res0 + res1 + res2
    Add(inputX2, inputGamma, inputX2, elem_cout_d_x_y);
    PipeBarrier<PIPE_V>();
    Add(outputDx, inputX2, inputDy, elem_cout_d_x_y);
    PipeBarrier<PIPE_V>();
    if constexpr (HAS_ADDITIONAL_INPUT) {
        Add(outputDx, outputDx, inputDx, elem_cout_d_x_y);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void CopyOutBetaGamma(const int32_t offsetUbGamma, const int32_t elem_cout_d_x_y)
{
    LocalTensor<float> d_gammaLocal = dGammaQue.DeQue<float>();
    LocalTensor<float> d_beta_local = dBetaQue.DeQue<float>();

    SetAtomicAdd<float>();
    if (selfTiling.isDeterministicKey) {
        DataCopyAutomicAdd(workspaceGmGamma, d_gammaLocal, elem_cout_d_x_y, offsetUbGamma, (uint16_t)1);
    } else {
        DataCopyAutomicAdd(dGammaGm, d_gammaLocal, elem_cout_d_x_y, offsetUbGamma, (uint16_t)1);
    }
    PipeBarrier<PIPE_ALL>();
    SetAtomicNone();

    SetAtomicAdd<float>();
    if (selfTiling.isDeterministicKey) {
        DataCopyAutomicAdd(workspaceGmBeta, d_beta_local, elem_cout_d_x_y, offsetUbGamma, (uint16_t)1);
    } else {
        DataCopyAutomicAdd(dBetaGm, d_beta_local, elem_cout_d_x_y, offsetUbGamma, (uint16_t)1);
    }
    PipeBarrier<PIPE_ALL>();
    SetAtomicNone();

    dGammaQue.FreeTensor(d_gammaLocal);
    dBetaQue.FreeTensor(d_beta_local);
}

__aicore__ inline void CopyOutX(const int32_t offsetUbXY, const int32_t d_y_in_ub, const int32_t n_in_once_ub)
{
    LocalTensor<T> dXLocal = dXQue.DeQue<T>();

    DataCopyCustom<T>(dXGm, dXLocal, d_y_in_ub, offsetUbXY, false, (uint16_t)n_in_once_ub);
    PipeBarrier<PIPE_ALL>();

    dXQue.FreeTensor(dXLocal);
}
