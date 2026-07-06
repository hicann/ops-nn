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
 * \file optimized_transducer.h
 * \brief
 */
#ifndef __OPTIMIZED_TRANSDUCER_H__
#define __OPTIMIZED_TRANSDUCER_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "optimized_transducer_tiling_data.h"
#include "optimized_transducer_tiling_key.h"

namespace NsOptimizedTransducer {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class OptimizedTransducer {
public:
    __aicore__ inline OptimizedTransducer(){};

    __aicore__ inline void Init(GM_ADDR logits, GM_ADDR targets, GM_ADDR logitLengths, GM_ADDR targetLengths,
                                GM_ADDR loss, GM_ADDR grad, GM_ADDR workspace, uint64_t maxT, uint64_t maxU, uint64_t V,
                                uint64_t batch_size, uint64_t bigCoreNum, uint64_t bigCoreProcessNum,
                                uint64_t smallCoreProcessNum, int blank, float clamp, bool fused_log_softmax,
                                bool requires_grad);
    __aicore__ inline void Process();

private:
    __aicore__ inline float exp(float a);
    __aicore__ inline float logsumexp(LocalTensor<float>& a, int32_t count);
    __aicore__ inline void LogAdd(LocalTensor<float>& a, LocalTensor<float>& b, int32_t count);
    __aicore__ inline void ComputeLogProbs(uint64_t idx);
    __aicore__ inline void ComputeLogProbsForLogSoftmax(uint64_t idx);
    __aicore__ inline void ComputeAlpha(uint64_t idx);
    __aicore__ inline void ComputeBeta(uint64_t idx);
    __aicore__ inline void ComputeGrad(uint64_t idx);
    __aicore__ inline void ComputeGradForLogSoftmax(uint64_t idx);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> Qin_logits;
    TQue<QuePosition::VECOUT, BUFFER_NUM> Qout_loss, Qout_grad;
    TBuf<QuePosition::VECCALC> B1, B2, B3, B4, B_cast;
    GlobalTensor<T> logitsGm, lossGm, gradGm;
    GlobalTensor<int32_t> targetsGm, logitLengthsGm, targetLengthsGm;
    GlobalTensor<float> probsGm, denGm, alphaGm, betaGm;
    uint64_t maxT, maxU, V, processNum, globalBufferIndex;
    int blank;
    float clamp;
    bool fused_log_softmax, requires_grad;
};

template <typename T>
__aicore__ inline float OptimizedTransducer<T>::exp(float a)
{
    LocalTensor<float> buf1 = B1.Get<float>();
    buf1.SetValue(0, a);
    Exp(buf1, buf1, 1);
    return buf1.GetValue(0);
}

template <typename T>
__aicore__ inline float OptimizedTransducer<T>::logsumexp(LocalTensor<float>& a, int32_t count)
{
    LocalTensor<float> buf1 = B1.Get<float>();
    ReduceMax<float>(buf1, a, buf1, count, false);
    float maxv = buf1.GetValue(0);
    Adds(buf1, a, -maxv, count);
    Exp(buf1, buf1, count);
    ReduceSum<float>(buf1, buf1, buf1, count);
    Log(buf1, buf1, 1);
    return maxv + buf1.GetValue(0);
}

template <typename T>
__aicore__ inline void OptimizedTransducer<T>::LogAdd(LocalTensor<float>& a, LocalTensor<float>& b, int32_t count)
{
    LocalTensor<float> maxv = B4.Get<float>();
    Max(maxv, a, b, count);
    Min(b, a, b, count);
    Sub(b, b, maxv, count);
    Exp(b, b, count);
    Adds(b, b, (float)1, count);
    Log(b, b, count);
    Add(a, b, maxv, count);
}

template <typename T>
__aicore__ inline void OptimizedTransducer<T>::Init(GM_ADDR logits, GM_ADDR targets, GM_ADDR logitLengths,
                                                    GM_ADDR targetLengths, GM_ADDR loss, GM_ADDR grad,
                                                    GM_ADDR workspace, uint64_t maxT, uint64_t maxU, uint64_t V,
                                                    uint64_t batch_size, uint64_t bigCoreNum,
                                                    uint64_t bigCoreProcessNum, uint64_t smallCoreProcessNum, int blank,
                                                    float clamp, bool fused_log_softmax, bool requires_grad)
{
    uint64_t coreIdx = GetBlockIdx();
    this->maxT = maxT;
    this->maxU = maxU;
    this->V = V;
    if (blank >= 0) {
        this->blank = blank;
    } else {
        this->blank = V + blank;
    }
    this->clamp = clamp;
    this->fused_log_softmax = fused_log_softmax;
    this->requires_grad = requires_grad;
    this->globalBufferIndex = coreIdx * bigCoreProcessNum;
    uint64_t TU = (maxT * maxU + 15) / 16 * 16;
    if (coreIdx < bigCoreNum) {
        this->processNum = bigCoreProcessNum;
    } else {
        this->processNum = smallCoreProcessNum;
        this->globalBufferIndex -= (bigCoreProcessNum - smallCoreProcessNum) * (coreIdx - bigCoreNum);
    }
    logitsGm.SetGlobalBuffer((__gm__ T*)logits + globalBufferIndex * maxT * maxU * V, processNum * maxT * maxU * V);
    targetsGm.SetGlobalBuffer((__gm__ int32_t*)targets + globalBufferIndex * (maxU - 1), processNum * (maxU - 1));
    logitLengthsGm.SetGlobalBuffer((__gm__ int32_t*)logitLengths + globalBufferIndex, processNum);
    targetLengthsGm.SetGlobalBuffer((__gm__ int32_t*)targetLengths + globalBufferIndex, processNum);
    lossGm.SetGlobalBuffer((__gm__ T*)loss + globalBufferIndex, processNum);
    gradGm.SetGlobalBuffer((__gm__ T*)grad + globalBufferIndex * maxT * maxU * V, processNum * maxT * maxU * V);
    probsGm.SetGlobalBuffer((__gm__ float*)workspace + globalBufferIndex * TU * 2, processNum * TU * 2);
    denGm.SetGlobalBuffer((__gm__ float*)workspace + batch_size * TU * 2 + globalBufferIndex * TU, processNum * TU);
    alphaGm.SetGlobalBuffer((__gm__ float*)workspace + batch_size * TU * 3 + globalBufferIndex * TU, processNum * TU);
    betaGm.SetGlobalBuffer((__gm__ float*)workspace + batch_size * TU * 4 + globalBufferIndex * TU, processNum * TU);
    pipe.InitBuffer(Qin_logits, BUFFER_NUM, V * sizeof(T));
    pipe.InitBuffer(Qout_loss, BUFFER_NUM, 32);
    pipe.InitBuffer(Qout_grad, BUFFER_NUM, V * sizeof(T));
    pipe.InitBuffer(B1, V * sizeof(float));
    uint64_t buffer_size = (maxT > maxU ? maxT : maxU) * sizeof(float);
    pipe.InitBuffer(B2, buffer_size);
    pipe.InitBuffer(B3, buffer_size);
    pipe.InitBuffer(B4, buffer_size);
    pipe.InitBuffer(B_cast, V * sizeof(float));
}

template <typename T>
__aicore__ inline void OptimizedTransducer<T>::Process()
{
    if (fused_log_softmax) {
        for (uint64_t i = 0; i < processNum; i++) {
            ComputeLogProbs(i);
            ComputeAlpha(i);
            ComputeBeta(i);
            if (requires_grad) {
                ComputeGrad(i);
            }
        }
    } else {
        for (uint64_t i = 0; i < processNum; i++) {
            ComputeLogProbsForLogSoftmax(i);
            ComputeAlpha(i);
            ComputeBeta(i);
            if (requires_grad) {
                ComputeGradForLogSoftmax(i);
            }
        }
    }
}

template <typename T>
__aicore__ inline void OptimizedTransducer<T>::ComputeLogProbs(uint64_t idx)
{
    int32_t len_T = logitLengthsGm.GetValue(idx), len_U = targetLengthsGm.GetValue(idx) + 1;
    uint64_t idx_probs = 0, offset = idx * maxT * maxU;
    LocalTensor<T> logits = Qin_logits.AllocTensor<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(V * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    if constexpr (std::is_same_v<T, float>) {
        for (int32_t t = 0; t != len_T; t++) {
            for (int32_t u = 0; u != len_U; u++, idx_probs += 2) {
                uint64_t idx_logits = idx * maxT * maxU * V + t * maxU * V + u * V;
                DataCopyPad(logits, logitsGm[idx_logits], copyParams, padParams);
                Qin_logits.EnQue(logits);
                logits = Qin_logits.DeQue<T>();
                float res = logsumexp(logits, (int32_t)V);
                probsGm.SetValue(offset * 2 + idx_probs, logits.GetValue(blank) - res);
                denGm.SetValue(offset + idx_probs / 2, res);
                if (u < len_U - 1) {
                    probsGm.SetValue(offset * 2 + idx_probs + 1,
                                     logits.GetValue(targetsGm.GetValue(idx * (maxU - 1) + u)) - res);
                }
            }
        }
    } else {
        LocalTensor<float> logits_float = B_cast.Get<float>();
        for (int32_t t = 0; t != len_T; t++) {
            for (int32_t u = 0; u != len_U; u++, idx_probs += 2) {
                uint64_t idx_logits = idx * maxT * maxU * V + t * maxU * V + u * V;
                DataCopyPad(logits, logitsGm[idx_logits], copyParams, padParams);
                Qin_logits.EnQue(logits);
                logits = Qin_logits.DeQue<T>();
                Cast(logits_float, logits, RoundMode::CAST_NONE, (int32_t)V);
                float res = logsumexp(logits_float, (int32_t)V);
                probsGm.SetValue(offset * 2 + idx_probs, logits_float.GetValue(blank) - res);
                denGm.SetValue(offset + idx_probs / 2, res);
                if (u < len_U - 1) {
                    probsGm.SetValue(offset * 2 + idx_probs + 1,
                                     logits_float.GetValue(targetsGm.GetValue(idx * (maxU - 1) + u)) - res);
                }
            }
        }
    }
    Qin_logits.FreeTensor(logits);
}

template <typename T>
__aicore__ inline void OptimizedTransducer<T>::ComputeLogProbsForLogSoftmax(uint64_t idx)
{
    int32_t len_T = logitLengthsGm.GetValue(idx), len_U = targetLengthsGm.GetValue(idx) + 1;
    uint64_t idx_probs = 0, offset = idx * maxT * maxU;
    if constexpr (std::is_same_v<T, float>) {
        for (int32_t t = 0; t != len_T; t++) {
            for (int32_t u = 0; u != len_U; u++, idx_probs += 2) {
                uint64_t offset_logits = offset * V + t * maxU * V + u * V;
                probsGm.SetValue(offset * 2 + idx_probs, logitsGm.GetValue(offset_logits + blank));
                if (u < len_U - 1) {
                    probsGm.SetValue(offset * 2 + idx_probs + 1,
                                     logitsGm.GetValue(offset_logits + targetsGm.GetValue(idx * (maxU - 1) + u)));
                }
            }
        }
    } else {
        for (int32_t t = 0; t != len_T; t++) {
            for (int32_t u = 0; u != len_U; u++, idx_probs += 2) {
                uint64_t offset_logits = offset * V + t * maxU * V + u * V;
                probsGm.SetValue(offset * 2 + idx_probs, (float)logitsGm.GetValue(offset_logits + blank));
                if (u < len_U - 1) {
                    probsGm.SetValue(
                        offset * 2 + idx_probs + 1,
                        (float)logitsGm.GetValue(offset_logits + targetsGm.GetValue(idx * (maxU - 1) + u)));
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void OptimizedTransducer<T>::ComputeAlpha(uint64_t idx)
{
    int32_t len_T = logitLengthsGm.GetValue(idx), len_U = targetLengthsGm.GetValue(idx) + 1;
    uint64_t offset = idx * maxT * maxU;
    alphaGm.SetValue(offset, (float)0);
    for (int32_t t = 1; t != len_T; t++) {
        alphaGm.SetValue(offset + t * len_U, alphaGm.GetValue(offset + (t - 1) * len_U) +
                                                 probsGm.GetValue(offset * 2 + (t - 1) * len_U * 2));
    }
    for (int32_t u = 1; u != len_U; u++) {
        alphaGm.SetValue(offset + u, alphaGm.GetValue(offset + u - 1) + probsGm.GetValue(offset * 2 + (u - 1) * 2 + 1));
    }
    LocalTensor<float> buf2 = B2.Get<float>();
    LocalTensor<float> buf3 = B3.Get<float>();
    if (len_T < len_U) {
        for (int32_t n = 2; n < len_T; n++) {
            for (int32_t t = 1; t < n; t++) {
                int u = n - t;
                buf2.SetValue(t - 1, alphaGm.GetValue(offset + (t - 1) * len_U + u) +
                                         probsGm.GetValue(offset * 2 + ((t - 1) * len_U + u) * 2));
                buf3.SetValue(t - 1, alphaGm.GetValue(offset + t * len_U + u - 1) +
                                         probsGm.GetValue(offset * 2 + (t * len_U + u - 1) * 2 + 1));
            }
            LogAdd(buf2, buf3, n - 1);
            for (int32_t t = 1; t < n; t++) {
                int u = n - t;
                alphaGm.SetValue(offset + t * len_U + u, buf2.GetValue(t - 1));
            }
        }
        for (int32_t n = len_T; n < len_U; n++) {
            for (int32_t t = 1; t < len_T; t++) {
                int u = n - t;
                buf2.SetValue(t - 1, alphaGm.GetValue(offset + (t - 1) * len_U + u) +
                                         probsGm.GetValue(offset * 2 + ((t - 1) * len_U + u) * 2));
                buf3.SetValue(t - 1, alphaGm.GetValue(offset + t * len_U + u - 1) +
                                         probsGm.GetValue(offset * 2 + (t * len_U + u - 1) * 2 + 1));
            }
            LogAdd(buf2, buf3, len_T - 1);
            for (int32_t t = 1; t < len_T; t++) {
                int u = n - t;
                alphaGm.SetValue(offset + t * len_U + u, buf2.GetValue(t - 1));
            }
        }
        for (int32_t n = len_U; n < len_T + len_U - 1; n++) {
            int32_t count = 0;
            for (int32_t t = n - len_U + 1; t < len_T; t++, count++) {
                int u = n - t;
                buf2.SetValue(count, alphaGm.GetValue(offset + (t - 1) * len_U + u) +
                                         probsGm.GetValue(offset * 2 + ((t - 1) * len_U + u) * 2));
                buf3.SetValue(count, alphaGm.GetValue(offset + t * len_U + u - 1) +
                                         probsGm.GetValue(offset * 2 + (t * len_U + u - 1) * 2 + 1));
            }
            LogAdd(buf2, buf3, count);
            for (int32_t t = n - len_U + 1, j = 0; t < len_T; t++, j++) {
                int u = n - t;
                alphaGm.SetValue(offset + t * len_U + u, buf2.GetValue(j));
            }
        }
    } else {
        for (int32_t n = 2; n < len_U; n++) {
            for (int32_t t = 1; t < n; t++) {
                int u = n - t;
                buf2.SetValue(t - 1, alphaGm.GetValue(offset + (t - 1) * len_U + u) +
                                         probsGm.GetValue(offset * 2 + ((t - 1) * len_U + u) * 2));
                buf3.SetValue(t - 1, alphaGm.GetValue(offset + t * len_U + u - 1) +
                                         probsGm.GetValue(offset * 2 + (t * len_U + u - 1) * 2 + 1));
            }
            LogAdd(buf2, buf3, n - 1);
            for (int32_t t = 1; t < n; t++) {
                int u = n - t;
                alphaGm.SetValue(offset + t * len_U + u, buf2.GetValue(t - 1));
            }
        }
        for (int32_t n = len_U; n < len_T; n++) {
            for (int32_t u = 1; u < len_U; u++) {
                int t = n - u;
                buf2.SetValue(u - 1, alphaGm.GetValue(offset + (t - 1) * len_U + u) +
                                         probsGm.GetValue(offset * 2 + ((t - 1) * len_U + u) * 2));
                buf3.SetValue(u - 1, alphaGm.GetValue(offset + t * len_U + u - 1) +
                                         probsGm.GetValue(offset * 2 + (t * len_U + u - 1) * 2 + 1));
            }
            LogAdd(buf2, buf3, len_U - 1);
            for (int32_t u = 1; u < len_U; u++) {
                int t = n - u;
                alphaGm.SetValue(offset + t * len_U + u, buf2.GetValue(u - 1));
            }
        }
        for (int32_t n = len_T; n < len_T + len_U - 1; n++) {
            int32_t count = 0;
            for (int32_t u = n - len_T + 1; u < len_U; u++, count++) {
                int t = n - u;
                buf2.SetValue(count, alphaGm.GetValue(offset + (t - 1) * len_U + u) +
                                         probsGm.GetValue(offset * 2 + ((t - 1) * len_U + u) * 2));
                buf3.SetValue(count, alphaGm.GetValue(offset + t * len_U + u - 1) +
                                         probsGm.GetValue(offset * 2 + (t * len_U + u - 1) * 2 + 1));
            }
            LogAdd(buf2, buf3, count);
            for (int32_t u = n - len_T + 1, j = 0; u < len_U; u++, j++) {
                int t = n - u;
                alphaGm.SetValue(offset + t * len_U + u, buf2.GetValue(j));
            }
        }
    }
}

template <typename T>
__aicore__ inline void OptimizedTransducer<T>::ComputeBeta(uint64_t idx)
{
    int len_T = logitLengthsGm.GetValue(idx), len_U = targetLengthsGm.GetValue(idx) + 1;
    uint64_t offset = idx * maxT * maxU;
    betaGm.SetValue(offset + (len_T - 1) * len_U + len_U - 1, probsGm.GetValue(offset * 2 + 2 * len_T * len_U - 2));
    for (int32_t t = len_T - 2; t >= 0; t--) {
        betaGm.SetValue(offset + t * len_U + len_U - 1, betaGm.GetValue(offset + (t + 1) * len_U + len_U - 1) +
                                                            probsGm.GetValue(offset * 2 + (t + 1) * len_U * 2 - 2));
    }
    for (int32_t u = len_U - 2; u >= 0; u--) {
        betaGm.SetValue(offset + (len_T - 1) * len_U + u,
                        betaGm.GetValue(offset + (len_T - 1) * len_U + u + 1) +
                            probsGm.GetValue(offset * 2 + ((len_T - 1) * len_U + u) * 2 + 1));
    }
    LocalTensor<float> buf2 = B2.Get<float>();
    LocalTensor<float> buf3 = B3.Get<float>();
    if (len_T < len_U) {
        for (int32_t n = len_T + len_U - 4, count = 1; n >= len_U - 1; n--, count++) {
            for (int32_t t = len_T - 2, j = 0; j < count; j++, t--) {
                int u = n - t;
                buf2.SetValue(j, betaGm.GetValue(offset + (t + 1) * len_U + u) +
                                     probsGm.GetValue(offset * 2 + (t * len_U + u) * 2));
                buf3.SetValue(j, betaGm.GetValue(offset + t * len_U + u + 1) +
                                     probsGm.GetValue(offset * 2 + (t * len_U + u) * 2 + 1));
            }
            LogAdd(buf2, buf3, count);
            for (int32_t t = len_T - 2, j = 0; j < count; j++, t--) {
                int u = n - t;
                betaGm.SetValue(offset + t * len_U + u, buf2.GetValue(j));
            }
        }
        for (int32_t n = len_U - 2; n >= len_T - 1; n--) {
            for (int32_t t = 0; t < len_T - 1; t++) {
                int u = n - t;
                buf2.SetValue(t, betaGm.GetValue(offset + (t + 1) * len_U + u) +
                                     probsGm.GetValue(offset * 2 + (t * len_U + u) * 2));
                buf3.SetValue(t, betaGm.GetValue(offset + t * len_U + u + 1) +
                                     probsGm.GetValue(offset * 2 + (t * len_U + u) * 2 + 1));
            }
            LogAdd(buf2, buf3, len_T - 1);
            for (int32_t t = 0; t < len_T - 1; t++) {
                int u = n - t;
                betaGm.SetValue(offset + t * len_U + u, buf2.GetValue(t));
            }
        }
        for (int32_t n = len_T - 2; n >= 0; n--) {
            for (int32_t t = 0; t <= n; t++) {
                int u = n - t;
                buf2.SetValue(t, betaGm.GetValue(offset + (t + 1) * len_U + u) +
                                     probsGm.GetValue(offset * 2 + (t * len_U + u) * 2));
                buf3.SetValue(t, betaGm.GetValue(offset + t * len_U + u + 1) +
                                     probsGm.GetValue(offset * 2 + (t * len_U + u) * 2 + 1));
            }
            LogAdd(buf2, buf3, n + 1);
            for (int32_t t = 0; t <= n; t++) {
                int u = n - t;
                betaGm.SetValue(offset + t * len_U + u, buf2.GetValue(t));
            }
        }
    } else {
        for (int32_t n = len_T + len_U - 4, count = 1; n >= len_T - 1; n--, count++) {
            for (int32_t t = len_T - 2, j = 0; j < count; j++, t--) {
                int u = n - t;
                buf2.SetValue(j, betaGm.GetValue(offset + (t + 1) * len_U + u) +
                                     probsGm.GetValue(offset * 2 + (t * len_U + u) * 2));
                buf3.SetValue(j, betaGm.GetValue(offset + t * len_U + u + 1) +
                                     probsGm.GetValue(offset * 2 + (t * len_U + u) * 2 + 1));
            }
            LogAdd(buf2, buf3, count);
            for (int32_t t = len_T - 2, j = 0; j < count; j++, t--) {
                int u = n - t;
                betaGm.SetValue(offset + t * len_U + u, buf2.GetValue(j));
            }
        }
        for (int32_t n = len_T - 2; n >= len_U - 1; n--) {
            for (int32_t u = 0; u < len_U - 1; u++) {
                int t = n - u;
                buf2.SetValue(u, betaGm.GetValue(offset + (t + 1) * len_U + u) +
                                     probsGm.GetValue(offset * 2 + (t * len_U + u) * 2));
                buf3.SetValue(u, betaGm.GetValue(offset + t * len_U + u + 1) +
                                     probsGm.GetValue(offset * 2 + (t * len_U + u) * 2 + 1));
            }
            LogAdd(buf2, buf3, len_U - 1);
            for (int32_t u = 0; u < len_U - 1; u++) {
                int t = n - u;
                betaGm.SetValue(offset + t * len_U + u, buf2.GetValue(u));
            }
        }
        for (int32_t n = len_U - 2; n >= 0; n--) {
            for (int32_t t = 0; t <= n; t++) {
                int u = n - t;
                buf2.SetValue(t, betaGm.GetValue(offset + (t + 1) * len_U + u) +
                                     probsGm.GetValue(offset * 2 + (t * len_U + u) * 2));
                buf3.SetValue(t, betaGm.GetValue(offset + t * len_U + u + 1) +
                                     probsGm.GetValue(offset * 2 + (t * len_U + u) * 2 + 1));
            }
            LogAdd(buf2, buf3, n + 1);
            for (int32_t t = 0; t <= n; t++) {
                int u = n - t;
                betaGm.SetValue(offset + t * len_U + u, buf2.GetValue(t));
            }
        }
    }
    LocalTensor<T> loss = Qout_loss.AllocTensor<T>();
    if constexpr (std::is_same_v<T, float>) {
        loss.SetValue(0, -betaGm.GetValue(offset));
    } else {
        LocalTensor<float> beta = B2.Get<float>();
        beta.SetValue(0, -betaGm.GetValue(offset));
        Cast(loss, beta, RoundMode::CAST_RINT, 1);
    }
    DataCopyExtParams copyParams{1, (uint32_t)(sizeof(T)), 0, 0, 0};
    DataCopyPad(lossGm[idx], loss, copyParams);
    Qout_loss.FreeTensor(loss);
}

template <typename T>
__aicore__ inline void OptimizedTransducer<T>::ComputeGrad(uint64_t idx)
{
    int len_T = logitLengthsGm.GetValue(idx), len_U = targetLengthsGm.GetValue(idx) + 1;
    uint64_t offset = idx * maxT * maxU;
    float loss = -1 * betaGm.GetValue(offset);
    LocalTensor<T> logits = Qin_logits.AllocTensor<T>();
    LocalTensor<T> grad = Qout_grad.AllocTensor<T>();
    DataCopyExtParams copyParams{1, (uint32_t)(V * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    if constexpr (std::is_same_v<T, float>) {
        for (int32_t t = 0; t != len_T; t++) {
            for (int32_t u = 0; u != len_U; u++) {
                float c = alphaGm.GetValue(offset + t * len_U + u) + loss - denGm.GetValue(offset + t * len_U + u);
                int32_t target_u = (u < len_U - 1) ? targetsGm.GetValue(idx * (maxU - 1) + u) : -1;
                uint64_t idx_logits = idx * maxT * maxU * V + t * maxU * V + u * V;
                DataCopyPad(logits, logitsGm[idx_logits], copyParams, padParams);
                Qin_logits.EnQue(logits);
                logits = Qin_logits.DeQue<T>();
                float g_blank = logits.GetValue(blank) + c;
                float g_target;
                if (target_u != -1) {
                    g_target = logits.GetValue(target_u) + c;
                }
                Adds(logits, logits, c + betaGm.GetValue(offset + t * len_U + u), (int32_t)V);
                Exp(grad, logits, (int32_t)V);
                if (t == len_T - 1 && u == len_U - 1) {
                    grad.SetValue(blank, grad.GetValue(blank) - exp(g_blank));
                }
                if (t < len_T - 1) {
                    grad.SetValue(blank,
                                  grad.GetValue(blank) - exp(g_blank + betaGm.GetValue(offset + (t + 1) * len_U + u)));
                }
                if (u < len_U - 1) {
                    grad.SetValue(target_u, grad.GetValue(target_u) -
                                                exp(g_target + betaGm.GetValue(offset + t * len_U + u + 1)));
                }
                if (clamp > 0) {
                    Mins(grad, grad, clamp, (int32_t)V);
                    Maxs(grad, grad, -clamp, (int32_t)V);
                }
                Qout_grad.EnQue(grad);
                grad = Qout_grad.DeQue<T>();
                DataCopyPad(gradGm[idx_logits], grad, copyParams);
                int32_t eventIDMTE3ToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            }
        }
    } else {
        LocalTensor<float> logits_float = B_cast.Get<float>();
        for (int32_t t = 0; t != len_T; t++) {
            for (int32_t u = 0; u != len_U; u++) {
                float c = alphaGm.GetValue(offset + t * len_U + u) + loss - denGm.GetValue(offset + t * len_U + u);
                int32_t target_u = (u < len_U - 1) ? targetsGm.GetValue(idx * (maxU - 1) + u) : -1;
                uint64_t idx_logits = idx * maxT * maxU * V + t * maxU * V + u * V;
                DataCopyPad(logits, logitsGm[idx_logits], copyParams, padParams);
                Qin_logits.EnQue(logits);
                logits = Qin_logits.DeQue<T>();
                Cast(logits_float, logits, RoundMode::CAST_NONE, (int32_t)V);
                float g_blank = logits_float.GetValue(blank) + c;
                float g_target;
                if (target_u != -1) {
                    g_target = logits_float.GetValue(target_u) + c;
                }
                Adds(logits_float, logits_float, c + betaGm.GetValue(offset + t * len_U + u), (int32_t)V);
                Exp(logits_float, logits_float, (int32_t)V);
                if (t == len_T - 1 && u == len_U - 1) {
                    logits_float.SetValue(blank, logits_float.GetValue(blank) - exp(g_blank));
                }
                if (t < len_T - 1) {
                    logits_float.SetValue(blank, logits_float.GetValue(blank) -
                                                     exp(g_blank + betaGm.GetValue(offset + (t + 1) * len_U + u)));
                }
                if (u < len_U - 1) {
                    logits_float.SetValue(target_u, logits_float.GetValue(target_u) -
                                                        exp(g_target + betaGm.GetValue(offset + t * len_U + u + 1)));
                }
                if (clamp > 0) {
                    Mins(logits_float, logits_float, clamp, (int32_t)V);
                    Maxs(logits_float, logits_float, -clamp, (int32_t)V);
                }
                Cast(grad, logits_float, RoundMode::CAST_RINT, (int32_t)V);
                Qout_grad.EnQue(grad);
                grad = Qout_grad.DeQue<T>();
                DataCopyPad(gradGm[idx_logits], grad, copyParams);
                int32_t eventIDMTE3ToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
            }
        }
    }
    Qin_logits.FreeTensor(logits);
    Qout_grad.FreeTensor(grad);
}

template <typename T>
__aicore__ inline void OptimizedTransducer<T>::ComputeGradForLogSoftmax(uint64_t idx)
{
    int len_T = logitLengthsGm.GetValue(idx), len_U = targetLengthsGm.GetValue(idx) + 1;
    uint64_t offset = idx * maxT * maxU;
    float loss = -1 * betaGm.GetValue(offset);
    if (clamp >= 0) {
        for (int32_t t = 0; t != len_T; t++) {
            for (int32_t u = 0; u != len_U; u++) {
                float c = alphaGm.GetValue(offset + t * len_U + u) + loss;
                int32_t target_u = (u < len_U - 1) ? targetsGm.GetValue(idx * (maxU - 1) + u) : -1;
                uint64_t idx_logits = idx * maxT * maxU * V + t * maxU * V + u * V;
                float g_blank = (float)logitsGm.GetValue(idx_logits + blank) + c;
                float g_target, res;
                if (target_u != -1) {
                    g_target = (float)logitsGm.GetValue(idx_logits + target_u) + c;
                }
                if (t == len_T - 1 && u == len_U - 1) {
                    res = -exp(g_blank);
                    if (res < -clamp) {
                        res = -clamp;
                    }
                    gradGm.SetValue(idx_logits + blank, res);
                }
                if (t < len_T - 1) {
                    res = -exp(g_blank + betaGm.GetValue(offset + (t + 1) * len_U + u));
                    if (res < -clamp) {
                        res = -clamp;
                    }
                    gradGm.SetValue(idx_logits + blank, res);
                }
                if (u < len_U - 1) {
                    res = -exp(g_target + betaGm.GetValue(offset + t * len_U + u + 1));
                    if (res < -clamp) {
                        res = -clamp;
                    }
                    gradGm.SetValue(idx_logits + target_u, res);
                }
            }
        }
    } else {
        for (int32_t t = 0; t != len_T; t++) {
            for (int32_t u = 0; u != len_U; u++) {
                float c = alphaGm.GetValue(offset + t * len_U + u) + loss;
                int32_t target_u = (u < len_U - 1) ? targetsGm.GetValue(idx * (maxU - 1) + u) : -1;
                uint64_t idx_logits = idx * maxT * maxU * V + t * maxU * V + u * V;
                float g_blank = (float)logitsGm.GetValue(idx_logits + blank) + c;
                float g_target;
                if (target_u != -1) {
                    g_target = (float)logitsGm.GetValue(idx_logits + target_u) + c;
                }
                if (t == len_T - 1 && u == len_U - 1) {
                    gradGm.SetValue(idx_logits + blank, -exp(g_blank));
                }
                if (t < len_T - 1) {
                    gradGm.SetValue(idx_logits + blank, -exp(g_blank + betaGm.GetValue(offset + (t + 1) * len_U + u)));
                }
                if (u < len_U - 1) {
                    gradGm.SetValue(idx_logits + target_u,
                                    -exp(g_target + betaGm.GetValue(offset + t * len_U + u + 1)));
                }
            }
        }
    }
}

} // namespace NsOptimizedTransducer
#endif // __OPTIMIZED_TRANSDUCER_H__
