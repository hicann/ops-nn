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
 * \file ctc_loss_v2_grad.h
 * \brief
 */
#ifndef CTC_LOSS_V2_GRAD_H
#define CTC_LOSS_V2_GRAD_H

#include <cmath>
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "kernel_tiling/kernel_tiling.h"
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"

namespace CTCLossV2GradNS {
using namespace AscendC;

constexpr int32_t THREAD_NUM_1024 = 1024;
constexpr int32_t THREAD_NUM_512 = 512;
constexpr int32_t INT_SIZE_32 = 4;
constexpr int32_t INT_SIZE_64 = 8;
constexpr int32_t ONE = 1;
constexpr int32_t ALIGN_SIZE = 32;

template <typename T, typename DataType, typename ThreadType>
class CTCLossV2Grad {
public:
    __aicore__ inline CTCLossV2Grad(){};

    __aicore__ inline void Init(
        GM_ADDR grad_out, GM_ADDR log_probs, GM_ADDR targets, GM_ADDR input_lengths, GM_ADDR target_lengths,
        GM_ADDR neg_log_likelihood, GM_ADDR log_alpha, GM_ADDR grad, GM_ADDR workspace,
        const CTCLossV2GradTilingData4AscnedC* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void InitGlobalGm();

private:
    // tiling data
    const CTCLossV2GradTilingData4AscnedC* tilingData_;
    // 输入参数gradOut
    GlobalTensor<T> gradOutGm;
    // 输入参数LogProbs
    GlobalTensor<T> logProbsGm;
    // 输入参数targets
    GlobalTensor<DataType> targetsGm;
    // 输入参数inputLengths
    GlobalTensor<DataType> inputLengthsGm;
    // 输入参数targetLengths
    GlobalTensor<DataType> targetLengthsGm;
    // 输入参数negLogLikehood
    GlobalTensor<T> negLogLikelihoodGm;
    // 输入参数logAlpha
    GlobalTensor<T> logAlphaGm;
    // 输出参数grad
    GlobalTensor<T> gradGm;
    // 中间计算参数logBeta，大小为（N, T, S）
    GlobalTensor<float> logBetaGm;
    // 保存临时的计算结果，否则会有精度损失
    GlobalTensor<float> tempGradGm;

    int32_t blockInx;
}; // CTCLossV2Grad

template <typename T, typename DataType, typename ThreadType>
__aicore__ inline void CTCLossV2Grad<T, DataType, ThreadType>::InitGlobalGm()
{
    if (blockInx >= tilingData_->initGradGmStartBlock && blockInx < tilingData_->initGradGmEndBlock) {
        AscendC::InitOutput<T>(
            gradGm[blockInx * tilingData_->initGradGmSizePerBlock], tilingData_->initGradGmSizePerBlock, 0);
    }
    if (blockInx == tilingData_->initGradGmEndBlock) {
        AscendC::InitOutput<T>(
            gradGm[blockInx * tilingData_->initGradGmSizePerBlock], tilingData_->initGradGmSizeLastBlock, 0);
    }
    if (blockInx >= tilingData_->initLogBetaGmStartBlock && blockInx < tilingData_->initLogBetaGmEndBlock) {
        AscendC::InitOutput<float>(
            logBetaGm[(blockInx - tilingData_->initLogBetaGmStartBlock) * tilingData_->initLogBetaGmSizePerBlock],
            tilingData_->initLogBetaGmSizePerBlock, -INFINITY);
    }
    if (blockInx == tilingData_->initLogBetaGmEndBlock) {
        AscendC::InitOutput<float>(
            logBetaGm[(blockInx - tilingData_->initLogBetaGmStartBlock) * tilingData_->initLogBetaGmSizePerBlock],
            tilingData_->initLogBetaGmSizeLastBlock, -INFINITY);
    }
    if (blockInx >= tilingData_->initTempGradGmStartBlock && blockInx < tilingData_->initTempGradGmEndBlock) {
        AscendC::InitOutput<float>(
            tempGradGm[(blockInx - tilingData_->initTempGradGmStartBlock) * tilingData_->initTempGradGmSizePerBlock],
            tilingData_->initTempGradGmSizePerBlock, -INFINITY);
    }
    if (blockInx == tilingData_->initTempGradGmEndBlock) {
        AscendC::InitOutput<float>(
            tempGradGm[(blockInx - tilingData_->initTempGradGmStartBlock) * tilingData_->initTempGradGmSizePerBlock],
            tilingData_->initTempGradGmSizeEndBlock, -INFINITY);
    }
}

template <typename T, typename DataType, typename ThreadType>
__aicore__ inline void CTCLossV2Grad<T, DataType, ThreadType>::Init(
    GM_ADDR gradOut, GM_ADDR logProbs, GM_ADDR targets, GM_ADDR inputLengths, GM_ADDR targetLengths,
    GM_ADDR negLogLikelihood, GM_ADDR logAlpha, GM_ADDR grad, GM_ADDR workspace,
    const CTCLossV2GradTilingData4AscnedC* tilingData)
{
    tilingData_ = tilingData;
    gradOutGm.SetGlobalBuffer((__gm__ T*)(gradOut));
    logProbsGm.SetGlobalBuffer((__gm__ T*)(logProbs));
    targetsGm.SetGlobalBuffer((__gm__ DataType*)(targets));
    inputLengthsGm.SetGlobalBuffer((__gm__ DataType*)(inputLengths));
    targetLengthsGm.SetGlobalBuffer((__gm__ DataType*)(targetLengths));
    negLogLikelihoodGm.SetGlobalBuffer((__gm__ T*)(negLogLikelihood));
    logAlphaGm.SetGlobalBuffer((__gm__ T*)(logAlpha));
    gradGm.SetGlobalBuffer((__gm__ T*)(grad));
    ThreadType batchSize = tilingData_->batchSize;
    ThreadType maxInputLength = tilingData_->maxInputLength;
    ThreadType alphaLength = tilingData_->alphaLength;
    ThreadType symbolSet = tilingData_->symbolSet;
    logBetaGm.SetGlobalBuffer((__gm__ float*)(workspace), batchSize * maxInputLength * alphaLength);
    // 为了保证精度，申请临时空间
    tempGradGm.SetGlobalBuffer(
        (__gm__ float*)(workspace) + batchSize * maxInputLength * alphaLength, batchSize * maxInputLength * symbolSet);
    blockInx = GetBlockIdx();
    InitGlobalGm();
    SyncAll();
}

template <typename T, typename DataType, typename ThreadType>
__aicore__ __attribute__((always_inline)) inline ThreadType ProcessTgBatchOffsets(
    ThreadType idx, __gm__ DataType* targetLengthsGm, ThreadType targetsDimNum, ThreadType sDimRange)
{
    if (targetsDimNum == 1) {
        ThreadType sum = 0;
        for (ThreadType i = 0; i < idx; i++) {
            sum += targetLengthsGm[i];
        }
        return sum;
    } else {
        return sDimRange * idx;
    }
}

template <typename T, typename DataType, typename ThreadType>
__aicore__ __attribute__((always_inline)) inline ThreadType GetTargetPrime(
    __gm__ DataType* targetsGm, ThreadType offset, ThreadType stride, ThreadType idx, ThreadType blank)
{
    if ((idx & 1) == 0) {
        return blank;
    } else {
        ThreadType targetNum = targetsGm[offset + stride * (idx >> 1)];
        return targetNum;
    }
}

template <typename T, typename DataType, typename ThreadType, int32_t THREAD_NUM>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM) __aicore__ void CalGradCompute(
    __gm__ T* gradOutGm, __gm__ T* logProbsGm, __gm__ DataType* targetsGm, __gm__ DataType* inputLengthsGm,
    __gm__ DataType* targetLengthsGm, __gm__ T* negLogLikelihoodGm, __gm__ T* logAlphaGm, __gm__ T* gradGm,
    __gm__ float* logBetaGm, __gm__ float* tempGradGm, ThreadType maxInputLength, ThreadType batchSize,
    ThreadType symbolSet, ThreadType zeroInfinity)
{
    ThreadType threadIdx = AscendC::Simt::GetThreadIdx<0>();
    ThreadType blockDimX = AscendC::Simt::GetThreadNum<0>();
    ThreadType gradBatchOffset = batchSize * symbolSet;
    ThreadType length = maxInputLength * batchSize * symbolSet;
    for (ThreadType index = threadIdx + block_idx * blockDimX; index < length; index += block_num * blockDimX) {
        ThreadType t = index / gradBatchOffset;
        ThreadType offset = index % gradBatchOffset;
        ThreadType b = offset / symbolSet;
        ThreadType c = offset % symbolSet;
        if (t >= maxInputLength || b >= batchSize || c >= symbolSet) {
            continue;
        }
        ThreadType inputLength = inputLengthsGm[b];
        float nll = negLogLikelihoodGm[b];
        float gr = gradOutGm[b];
        ThreadType batchOffset = b * symbolSet;
        ThreadType inputOffset = t * gradBatchOffset;
        if (t < inputLength && (!zeroInfinity || nll != INFINITY)) {
            float res = tempGradGm[batchOffset + inputOffset + c];
            float lp = logProbsGm[batchOffset + inputOffset + c];
            gradGm[batchOffset + inputOffset + c] = (__expf(lp) - __expf(res + nll - lp)) * gr;
        }
    }
}

template <typename T, typename DataType, typename ThreadType, int32_t THREAD_NUM>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM) __aicore__ void UpdateLcabCompute(
    __gm__ T* gradOutGm, __gm__ T* logProbsGm, __gm__ DataType* targetsGm, __gm__ DataType* inputLengthsGm,
    __gm__ DataType* targetLengthsGm, __gm__ T* negLogLikelihoodGm, __gm__ T* logAlphaGm, __gm__ volatile T* gradGm,
    __gm__ volatile float* logBetaGm, __gm__ volatile float* tempGradGm, ThreadType maxInputLength,
    ThreadType batchSize, ThreadType symbolSet, ThreadType zeroInfinity, ThreadType blank, ThreadType logAlphaT,
    ThreadType alphaLength, ThreadType targetsDimNum, ThreadType sDimRange)
{
    ThreadType threadIdx = AscendC::Simt::GetThreadIdx<0>();
    ThreadType blockDimX = AscendC::Simt::GetThreadNum<0>();
    ThreadType length = maxInputLength * batchSize;
    for (ThreadType index = threadIdx + block_idx * blockDimX; index < length; index += block_num * blockDimX) {
        ThreadType b = index / maxInputLength;
        ThreadType t = index % maxInputLength;
        if ((t >= maxInputLength) || (b >= batchSize)) {
            continue;
        }
        constexpr float neginf = -INFINITY;
        ThreadType inputLength = inputLengthsGm[b];
        // begin: 新增判断逻辑，减少计算量
        float nll = negLogLikelihoodGm[b];
        if (t >= inputLength || (zeroInfinity && nll == INFINITY)) {
            return;
        }
        // end: 新增判断逻辑，减少计算量
        ThreadType targetLength = targetLengthsGm[b];
        ThreadType batchOffset = b * symbolSet;
        ThreadType inputBatchOffset = t * symbolSet * batchSize;
        ThreadType logAlphaBatchOffset = b * logAlphaT * alphaLength;
        ThreadType logAlphaInputOffset = t * alphaLength;
        ThreadType targetBatchOffset =
            ProcessTgBatchOffsets<T, DataType, ThreadType>(b, targetLengthsGm, targetsDimNum, sDimRange);
        ThreadType currentTargetPrime;
        // 和CPU保持一致，但是跟GPU不一致
        if (t == inputLength - 1) {
            currentTargetPrime = GetTargetPrime<T, DataType, ThreadType>(
                targetsGm, targetBatchOffset, static_cast<ThreadType>(1), 2 * targetLength, blank);
            tempGradGm[inputBatchOffset + batchOffset + currentTargetPrime] =
                static_cast<float>(logAlphaGm[logAlphaBatchOffset + logAlphaInputOffset + 2 * targetLength]) +
                static_cast<float>(logProbsGm[batchOffset + inputBatchOffset + currentTargetPrime]);

            if (targetLength > 0) {
                currentTargetPrime = GetTargetPrime<T, DataType, ThreadType>(
                    targetsGm, targetBatchOffset, static_cast<ThreadType>(1), 2 * targetLength - 1, blank);
                tempGradGm[inputBatchOffset + batchOffset + currentTargetPrime] =
                    static_cast<float>(logAlphaGm[logAlphaBatchOffset + logAlphaInputOffset + 2 * targetLength - 1]) +
                    static_cast<float>(logProbsGm[batchOffset + inputBatchOffset + currentTargetPrime]);
            }
        }
        for (ThreadType s = 0; s < alphaLength; s++) {
            if (s < 2 * targetLength + 1 && t != inputLength - 1) {
                currentTargetPrime = GetTargetPrime<T, DataType, ThreadType>(
                    targetsGm, targetBatchOffset, static_cast<ThreadType>(1), s, blank);
                float logAlphaBetaSum = static_cast<float>(logAlphaGm[logAlphaBatchOffset + logAlphaInputOffset + s]) +
                                        logBetaGm[logAlphaBatchOffset + logAlphaInputOffset + s];
                float lcab = tempGradGm[inputBatchOffset + batchOffset + currentTargetPrime];
                if (lcab == neginf) {
                    lcab = logAlphaBetaSum;
                    tempGradGm[inputBatchOffset + batchOffset + currentTargetPrime] = lcab;
                } else {
                    float max = lcab > logAlphaBetaSum ? lcab : logAlphaBetaSum;
                    tempGradGm[inputBatchOffset + batchOffset + currentTargetPrime] =
                        __logf(__expf(lcab - max) + __expf(logAlphaBetaSum - max)) + max;
                }
            }
        }
    }
}

template <typename T, typename DataType, typename ThreadType, int32_t THREAD_NUM>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM) __aicore__ void LogBetaCompute(
    __gm__ T* gradOutGm, __gm__ T* logProbsGm, __gm__ DataType* targetsGm, __gm__ DataType* inputLengthsGm,
    __gm__ DataType* targetLengthsGm, __gm__ T* negLogLikelihoodGm, __gm__ T* logAlphaGm, __gm__ T* gradGm,
    __gm__ float* logBetaGm, __gm__ float* tempGradGm, ThreadType maxInputLength, ThreadType batchSize,
    ThreadType symbolSet, ThreadType zeroInfinity, ThreadType blank, ThreadType logAlphaT, ThreadType alphaLength,
    ThreadType targetsDimNum, ThreadType sDimRange)
{
    constexpr float neginf = -INFINITY;
    ThreadType threadIdy = AscendC::Simt::GetThreadIdx<1>();
    ThreadType threadIdx = AscendC::Simt::GetThreadIdx<0>();
    ThreadType blockDimx = AscendC::Simt::GetThreadNum<0>();
    ThreadType blockDimy = AscendC::Simt::GetThreadNum<1>();
    for (ThreadType index = threadIdy + block_idx * blockDimy; index < batchSize; index += block_num * blockDimy) {
        ThreadType b = index;
        ThreadType inputLength = inputLengthsGm[b];
        if (inputLength == 0) {
            continue;
        }
        ThreadType targetLength = targetLengthsGm[b];
        ThreadType logProbsBatchOffset = b * symbolSet;
        ThreadType logBetaBatchOffset = b * logAlphaT * alphaLength;
        ThreadType targetBatchOffset =
            ProcessTgBatchOffsets<T, DataType, ThreadType>(b, targetLengthsGm, targetsDimNum, sDimRange);
        for (ThreadType block_s = alphaLength - 1 - ((alphaLength - 1) % blockDimx); block_s >= 0;
             block_s -= blockDimx) {
            ThreadType s = threadIdx + block_s;
            float lb;
            if (s == 2 * targetLength) {
                lb = logProbsGm[logProbsBatchOffset + (inputLength - 1) * batchSize * symbolSet + blank];
            } else if (s == 2 * targetLength - 1) {
                ThreadType currentTargetPrime = GetTargetPrime<T, DataType, ThreadType>(
                    targetsGm, targetBatchOffset, static_cast<ThreadType>(1), s, blank);
                lb = logProbsGm[logProbsBatchOffset + (inputLength - 1) * batchSize * symbolSet + currentTargetPrime];
            } else {
                lb = neginf;
            }
            if (s < alphaLength) {
                logBetaGm[logBetaBatchOffset + (inputLength - 1) * alphaLength + s] = lb;
            }
        }
        for (ThreadType block_s = alphaLength - 1 - ((alphaLength - 1) % blockDimx); block_s >= 0;
             block_s -= blockDimx) {
            ThreadType s = threadIdx + block_s;
            ThreadType currentTargetPrime;
            bool haveThree;
            if (s < 2 * targetLength + 1 && targetLength > 0) {
                currentTargetPrime = GetTargetPrime<T, DataType, ThreadType>(
                    targetsGm, targetBatchOffset, static_cast<ThreadType>(1), s, blank);
                haveThree =
                    ((s < 2 * targetLength - 1) && (GetTargetPrime<T, DataType, ThreadType>(
                                                        targetsGm, targetBatchOffset, static_cast<ThreadType>(1), s + 2,
                                                        blank) != currentTargetPrime));
            } else {
                currentTargetPrime = blank;
                haveThree = false;
            }
            for (ThreadType t = maxInputLength - 2; t >= 0; t--) {
                Simt::ThreadBarrier();
                if ((t < inputLength - 1) && (s < 2 * targetLength + 1)) {
                    float lb1 = logBetaGm[logBetaBatchOffset + (t + 1) * alphaLength + s];
                    float lbmax = lb1;
                    float lb2, lb3;
                    if (s < 2 * targetLength) {
                        lb2 = logBetaGm[logBetaBatchOffset + alphaLength * (t + 1) + (s + 1)];
                        if (lb2 > lbmax) {
                            lbmax = lb2;
                        }
                    } else {
                        lb2 = neginf;
                    }
                    if (haveThree) {
                        lb3 = logBetaGm[logBetaBatchOffset + alphaLength * (t + 1) + (s + 2)];
                        if (lb3 > lbmax) {
                            lbmax = lb3;
                        }
                    } else {
                        lb3 = neginf;
                    }
                    if (lbmax == neginf) {
                        lbmax = 0;
                    }
                    float lb =
                        __logf(__expf(lb1 - lbmax) + __expf(lb2 - lbmax) + __expf(lb3 - lbmax)) + lbmax +
                        static_cast<float>(logProbsGm[b * symbolSet + t * batchSize * symbolSet + currentTargetPrime]);
                    logBetaGm[logBetaBatchOffset + alphaLength * t + s] = lb;
                }
            }
        }
    }
}

template <typename T, typename DataType, typename ThreadType>
__aicore__ inline void CTCLossV2Grad<T, DataType, ThreadType>::Process()
{
    ThreadType blockDimX = tilingData_->blockDimX;
    ThreadType blockDimY = tilingData_->blockDimY;
    if constexpr (sizeof(ThreadType) == INT_SIZE_32) {
        Simt::VF_CALL<LogBetaCompute<T, DataType, ThreadType, THREAD_NUM_1024>>(
            Simt::Dim3(blockDimX, blockDimY), (__gm__ T*)(gradOutGm.GetPhyAddr()), (__gm__ T*)(logProbsGm.GetPhyAddr()),
            (__gm__ DataType*)(targetsGm.GetPhyAddr()), (__gm__ DataType*)(inputLengthsGm.GetPhyAddr()),
            (__gm__ DataType*)(targetLengthsGm.GetPhyAddr()), (__gm__ T*)(negLogLikelihoodGm.GetPhyAddr()),
            (__gm__ T*)(logAlphaGm.GetPhyAddr()), (__gm__ T*)(gradGm.GetPhyAddr()),
            (__gm__ float*)(logBetaGm.GetPhyAddr()), (__gm__ float*)(tempGradGm.GetPhyAddr()),
            tilingData_->maxInputLength, tilingData_->batchSize, tilingData_->symbolSet, tilingData_->zeroInfinity,
            tilingData_->BLANK, tilingData_->logAlphaT, tilingData_->alphaLength, tilingData_->targetsDimNum,
            tilingData_->sDimRange);

        SyncAll();
        Simt::VF_CALL<UpdateLcabCompute<T, DataType, ThreadType, THREAD_NUM_1024>>(
            Simt::Dim3(tilingData_->updateLcabThreadNum, 1), (__gm__ T*)(gradOutGm.GetPhyAddr()),
            (__gm__ T*)(logProbsGm.GetPhyAddr()), (__gm__ DataType*)(targetsGm.GetPhyAddr()),
            (__gm__ DataType*)(inputLengthsGm.GetPhyAddr()), (__gm__ DataType*)(targetLengthsGm.GetPhyAddr()),
            (__gm__ T*)(negLogLikelihoodGm.GetPhyAddr()), (__gm__ T*)(logAlphaGm.GetPhyAddr()),
            (__gm__ volatile T*)(gradGm.GetPhyAddr()), (__gm__ volatile float*)(logBetaGm.GetPhyAddr()),
            (__gm__ volatile float*)(tempGradGm.GetPhyAddr()), tilingData_->maxInputLength, tilingData_->batchSize,
            tilingData_->symbolSet, tilingData_->zeroInfinity, tilingData_->BLANK, tilingData_->logAlphaT,
            tilingData_->alphaLength, tilingData_->targetsDimNum, tilingData_->sDimRange);
        SyncAll();
        Simt::VF_CALL<CalGradCompute<T, DataType, ThreadType, THREAD_NUM_1024>>(
            Simt::Dim3(tilingData_->calGradThreadNum, 1), (__gm__ T*)(gradOutGm.GetPhyAddr()),
            (__gm__ T*)(logProbsGm.GetPhyAddr()), (__gm__ DataType*)(targetsGm.GetPhyAddr()),
            (__gm__ DataType*)(inputLengthsGm.GetPhyAddr()), (__gm__ DataType*)(targetLengthsGm.GetPhyAddr()),
            (__gm__ T*)(negLogLikelihoodGm.GetPhyAddr()), (__gm__ T*)(logAlphaGm.GetPhyAddr()),
            (__gm__ T*)(gradGm.GetPhyAddr()), (__gm__ float*)(logBetaGm.GetPhyAddr()),
            (__gm__ float*)(tempGradGm.GetPhyAddr()), tilingData_->maxInputLength, tilingData_->batchSize,
            tilingData_->symbolSet, tilingData_->zeroInfinity);
    }

    if constexpr (sizeof(ThreadType) == INT_SIZE_64) {
        Simt::VF_CALL<LogBetaCompute<T, DataType, ThreadType, THREAD_NUM_512>>(
            Simt::Dim3(blockDimX, blockDimY), (__gm__ T*)(gradOutGm.GetPhyAddr()), (__gm__ T*)(logProbsGm.GetPhyAddr()),
            (__gm__ DataType*)(targetsGm.GetPhyAddr()), (__gm__ DataType*)(inputLengthsGm.GetPhyAddr()),
            (__gm__ DataType*)(targetLengthsGm.GetPhyAddr()), (__gm__ T*)(negLogLikelihoodGm.GetPhyAddr()),
            (__gm__ T*)(logAlphaGm.GetPhyAddr()), (__gm__ T*)(gradGm.GetPhyAddr()),
            (__gm__ float*)(logBetaGm.GetPhyAddr()), (__gm__ float*)(tempGradGm.GetPhyAddr()),
            tilingData_->maxInputLength, tilingData_->batchSize, tilingData_->symbolSet, tilingData_->zeroInfinity,
            tilingData_->BLANK, tilingData_->logAlphaT, tilingData_->alphaLength, tilingData_->targetsDimNum,
            tilingData_->sDimRange);

        SyncAll();
        Simt::VF_CALL<UpdateLcabCompute<T, DataType, ThreadType, THREAD_NUM_512>>(
            Simt::Dim3(tilingData_->updateLcabThreadNum, 1), (__gm__ T*)(gradOutGm.GetPhyAddr()),
            (__gm__ T*)(logProbsGm.GetPhyAddr()), (__gm__ DataType*)(targetsGm.GetPhyAddr()),
            (__gm__ DataType*)(inputLengthsGm.GetPhyAddr()), (__gm__ DataType*)(targetLengthsGm.GetPhyAddr()),
            (__gm__ T*)(negLogLikelihoodGm.GetPhyAddr()), (__gm__ T*)(logAlphaGm.GetPhyAddr()),
            (__gm__ volatile T*)(gradGm.GetPhyAddr()), (__gm__ volatile float*)(logBetaGm.GetPhyAddr()),
            (__gm__ volatile float*)(tempGradGm.GetPhyAddr()), tilingData_->maxInputLength, tilingData_->batchSize,
            tilingData_->symbolSet, tilingData_->zeroInfinity, tilingData_->BLANK, tilingData_->logAlphaT,
            tilingData_->alphaLength, tilingData_->targetsDimNum, tilingData_->sDimRange);
        SyncAll();
        Simt::VF_CALL<CalGradCompute<T, DataType, ThreadType, THREAD_NUM_512>>(
            Simt::Dim3(tilingData_->calGradThreadNum, 1), (__gm__ T*)(gradOutGm.GetPhyAddr()),
            (__gm__ T*)(logProbsGm.GetPhyAddr()), (__gm__ DataType*)(targetsGm.GetPhyAddr()),
            (__gm__ DataType*)(inputLengthsGm.GetPhyAddr()), (__gm__ DataType*)(targetLengthsGm.GetPhyAddr()),
            (__gm__ T*)(negLogLikelihoodGm.GetPhyAddr()), (__gm__ T*)(logAlphaGm.GetPhyAddr()),
            (__gm__ T*)(gradGm.GetPhyAddr()), (__gm__ float*)(logBetaGm.GetPhyAddr()),
            (__gm__ float*)(tempGradGm.GetPhyAddr()), tilingData_->maxInputLength, tilingData_->batchSize,
            tilingData_->symbolSet, tilingData_->zeroInfinity);
    }
}
} // namespace CTCLossV2GradNS
#endif // CTC_LOSS_V2_GRAD_INT32