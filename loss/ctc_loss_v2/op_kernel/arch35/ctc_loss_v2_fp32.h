/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file ctc_loss_v2_fp32.h
 * \brief ctc_loss_v2_fp32
 */

#ifndef CTC_LOSS_V2_FP32
#define CTC_LOSS_V2_FP32

#include "kernel_operator.h"
#include "../inc/kernel_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "ctc_loss_v2_base.h"
namespace CTCLossV2 {
using namespace AscendC;

template <typename T, typename DataType, typename ThreadType>
class CTCLossV2FP32 : public CTCLossV2Base<T, DataType, ThreadType> {
public:
    __aicore__ inline CTCLossV2FP32(){};
    __aicore__ inline void Init(GM_ADDR log_probs, GM_ADDR targets, GM_ADDR input_lengths, GM_ADDR target_lengths,
                                GM_ADDR neg_log_likelihood, GM_ADDR log_alpha, GM_ADDR workspace,
                                const CTCLossV2TilingData4AscendC* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    LocalTensor<DataType> targetsLengthsTensor;
    TQue<QuePosition::VECIN, 1> targetsLengthsQue;
    TBuf<QuePosition::VECCALC> targetOffsetQue;
    LocalTensor<DataType> targetOffsetTensor;
    TPipe* pipe;
};

template <typename T, typename DataType, typename ThreadType>
__aicore__ inline void CTCLossV2FP32<T, DataType, ThreadType>::Init(
    GM_ADDR log_probs, GM_ADDR targets, GM_ADDR input_lengths, GM_ADDR target_lengths, GM_ADDR neg_log_likelihood,
    GM_ADDR log_alpha, GM_ADDR workspace, const CTCLossV2TilingData4AscendC* tilingData, TPipe* inputPipe)
{
    this->BaseInit(log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, workspace,
                   tilingData);
    ThreadType batchSize = this->tdPtr->batchSize;
    pipe = inputPipe;
    uint32_t bsAlign = ops::CeilAlign(static_cast<int64_t>(batchSize * sizeof(DataType)), int64_t(BSALIGNSIZE));
    pipe->InitBuffer(targetsLengthsQue, 1, bsAlign);
    targetsLengthsTensor = targetsLengthsQue.AllocTensor<DataType>();
    DataCopyPadExtParams<DataType> padParams = {false, 0, 0, 0};
    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(batchSize * sizeof(DataType)), 0, 0, 0};
    DataCopyPad(targetsLengthsTensor, this->targetLengthsGm, copyParams, padParams);
    this->MTE2ToSSync();
    pipe->InitBuffer(targetOffsetQue, bsAlign);
    targetOffsetTensor = targetOffsetQue.AllocTensor<DataType>();
    targetOffsetTensor.SetValue(0, 0);
    DataType offset = 0;
    DataType targetsDim = this->tdPtr->targetsDim;
    if (targetsDim == 1) {
        for (int32_t b = 0; b < batchSize - 1; b++) {
            offset += targetsLengthsTensor.GetValue(b);
            targetOffsetTensor.SetValue(b + 1, offset);
        }
    }
}

template <typename DataType>
__simt_callee__ __aicore__ inline DataType GetTargetPrime(__gm__ DataType* target, int64_t offset, int64_t stride,
                                                          int32_t idx, int64_t blank)
{
    return ((idx & 1) == 0) ? blank : target[offset + stride * (idx >> 1)];
}

template <typename DataType>
__simt_callee__ __aicore__ inline DataType ProcessTgBatchOffsets(__ubuf__ DataType* tensor, int64_t targetsDim,
                                                                 int64_t tgBatchStride, int32_t idx)
{
    return (targetsDim == 1) ? tensor[idx] : (tgBatchStride * idx);
}

template <typename T, typename DataType, typename ThreadType>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void CalcLogAlpha(
    int32_t batchSize, int32_t laInputStride, ThreadType laBatchStride, ThreadType lpBatchStride,
    int32_t maxInputLength, ThreadType lpInputStride, int32_t targetsDim, int32_t tgBatchStride, int32_t blank,
    int32_t tgTargetStride, __gm__ T* logProbsGm, __gm__ DataType* targetsGm, __gm__ DataType* inputLengthsGm,
    __gm__ DataType* targetLengthsGm, __gm__ T* negLogLikelihoodGm, __gm__ T* logAlphaGm, __ubuf__ DataType* tensor)
{
    int32_t threadIdy = threadIdx.y;
    int32_t thread_idx = threadIdx.x;
    int32_t blockDimx = blockDim.x;
    int32_t blockDimy = blockDim.y;
    int32_t blkIdx = blockIdx.x;
    int32_t blockNum = gridDim.x;
    for (int32_t b = threadIdy + blkIdx * blockDimy; b < batchSize; b += blockNum * blockDimy) {
        ThreadType inputLength = inputLengthsGm[b];
        ThreadType targetLength = targetLengthsGm[b];
        ThreadType lpBatchOffset = b * lpBatchStride;
        ThreadType laBatchOffset = b * laBatchStride;
        ThreadType tgBatchOffset = ProcessTgBatchOffsets<DataType>(tensor, targetsDim, tgBatchStride, b);

        if (inputLength == 0) {
            if (thread_idx == 0) {
                float log_likelihood = targetLength == 0 ? 0 : neginf;
                negLogLikelihoodGm[b] = -log_likelihood;
            }
            continue;
        }
        for (int32_t block_s = 0; block_s < laInputStride; block_s += blockDimx) {
            int32_t s = thread_idx + block_s;
            float la = 0;
            if (s == 0) {
                la = logProbsGm[lpBatchOffset + blank];
            } else if (s == 1) {
                la = targetLength == 0 ?
                         neginf :
                         logProbsGm[lpBatchOffset + GetTargetPrime<DataType>(targetsGm, tgBatchOffset, tgTargetStride,
                                                                             int32_t(1), blank)];
            } else {
                la = neginf;
            }
            if (s < laInputStride) {
                logAlphaGm[laBatchOffset + s] = la;
            }
        }

        for (int32_t block_s = 0; block_s < laInputStride; block_s += blockDimx) {
            int32_t s = thread_idx + block_s;
            int32_t currentChar;
            bool haveThree;
            if (s < 2 * targetLength + 1 && targetLength > 0) {
                currentChar = GetTargetPrime<DataType>(targetsGm, tgBatchOffset, tgTargetStride, s, blank);
                haveThree = ((s > 1) && (GetTargetPrime<DataType>(targetsGm, tgBatchOffset, tgTargetStride, s - 2,
                                                                  blank) != currentChar));
            } else {
                currentChar = blank;
                haveThree = false;
            }
            for (int32_t t = 1; t < maxInputLength; t++) {
                asc_syncthreads();
                if ((t < inputLength) && (s < 2 * targetLength + 1)) {
                    float x = logProbsGm[lpBatchOffset + t * lpInputStride + currentChar];
                    float la1 = logAlphaGm[laBatchOffset + laInputStride * (t - 1) + s];
                    float lamax = la1;
                    float la2 = 0;
                    float la3 = 0;
                    la2 = (s > 0) ? logAlphaGm[laBatchOffset + laInputStride * (t - 1) + (s - 1)] : neginf;
                    lamax = (s > 0) ? (fmaxf(la2, lamax)) : lamax;
                    la3 = (haveThree == true) ? logAlphaGm[laBatchOffset + laInputStride * (t - 1) + (s - 2)] : neginf;
                    lamax = (haveThree == true) ? (fmaxf(la3, lamax)) : lamax;
                    lamax = (lamax == neginf) ? 0 : lamax;
                    logAlphaGm[laBatchOffset + laInputStride * t + s] = log1pf(expf(la1 - lamax) + expf(la2 - lamax) +
                                                                               expf(la3 - lamax) - 1) +
                                                                        lamax + x;
                }
            }
        }
        asc_syncthreads();

        // compute the loss
        if (thread_idx == 0) {
            float l1 = logAlphaGm[laBatchOffset + laInputStride * (inputLength - 1) + (targetLength * 2)];
            float l2 = targetLength > 0 ?
                           logAlphaGm[laBatchOffset + laInputStride * (inputLength - 1) + (targetLength * 2 - 1)] :
                           neginf;
            float m = ((l1 > l2) ? l1 : l2);
            m = ((m == neginf) ? 0 : m);
            float log_likelihood = log1pf(expf(l1 - m) + expf(l2 - m) - 1) + m;
            negLogLikelihoodGm[b] = -log_likelihood;
        }
    }
}

template <typename T, typename DataType, typename ThreadType>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_1024) __aicore__
    void SimtComputeINT32(int32_t batchSize, int32_t laInputStride, ThreadType laBatchStride, ThreadType lpBatchStride,
                          int32_t maxInputLength, ThreadType lpInputStride, int32_t targetsDim, int32_t tgBatchStride,
                          int32_t blank, int32_t tgTargetStride, __gm__ T* logProbsGm, __gm__ DataType* targetsGm,
                          __gm__ DataType* inputLengthsGm, __gm__ DataType* targetLengthsGm,
                          __gm__ T* negLogLikelihoodGm, __gm__ T* logAlphaGm, __ubuf__ DataType* tensor)
{
    CalcLogAlpha<T, DataType, ThreadType>(batchSize, laInputStride, laBatchStride, lpBatchStride, maxInputLength,
                                          lpInputStride, targetsDim, tgBatchStride, blank, tgTargetStride, logProbsGm,
                                          targetsGm, inputLengthsGm, targetLengthsGm, negLogLikelihoodGm, logAlphaGm,
                                          tensor);
}

template <typename T, typename DataType, typename ThreadType>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_512) __aicore__
    void SimtCompute(int32_t batchSize, int32_t laInputStride, ThreadType laBatchStride, ThreadType lpBatchStride,
                     int32_t maxInputLength, ThreadType lpInputStride, int32_t targetsDim, int32_t tgBatchStride,
                     int32_t blank, int32_t tgTargetStride, __gm__ T* logProbsGm, __gm__ DataType* targetsGm,
                     __gm__ DataType* inputLengthsGm, __gm__ DataType* targetLengthsGm, __gm__ T* negLogLikelihoodGm,
                     __gm__ T* logAlphaGm, __ubuf__ DataType* tensor)
{
    CalcLogAlpha<T, DataType, ThreadType>(batchSize, laInputStride, laBatchStride, lpBatchStride, maxInputLength,
                                          lpInputStride, targetsDim, tgBatchStride, blank, tgTargetStride, logProbsGm,
                                          targetsGm, inputLengthsGm, targetLengthsGm, negLogLikelihoodGm, logAlphaGm,
                                          tensor);
}

template <typename T, typename DataType, typename ThreadType>
__aicore__ inline void CTCLossV2FP32<T, DataType, ThreadType>::Process()
{
    int32_t maxInputLength = this->tdPtr->maxInputLength;
    int32_t maxTargetLength = this->tdPtr->maxTargetLength;
    ThreadType lpInputStride = this->tdPtr->lpInputStride;
    ThreadType lpBatchStride = this->tdPtr->lpBatchStride;
    ThreadType laBatchStride = this->tdPtr->laBatchStride;
    int32_t laInputStride = this->tdPtr->laInputStride;
    int32_t tgTargetStride = this->tdPtr->tgTargetStride;
    int32_t batchSize = this->tdPtr->batchSize;
    int32_t blank = this->tdPtr->blank;
    int32_t blockDimX = this->tdPtr->blockDimX;
    int32_t blockDimY = this->tdPtr->blockDimY;
    int32_t targetsDim = this->tdPtr->targetsDim;
    int32_t tgBatchStride = this->tdPtr->tgBatchStride;
    int32_t gridY = this->tdPtr->gridY;

    if constexpr (sizeof(ThreadType) == sizeof(int32_t)) {
        asc_vf_call<SimtComputeINT32<T, DataType, ThreadType>>(
            dim3(blockDimX, blockDimY), batchSize, laInputStride, laBatchStride, lpBatchStride, maxInputLength,
            lpInputStride, targetsDim, tgBatchStride, blank, tgTargetStride,
            (__gm__ T*)(this->logProbsDataGm.GetPhyAddr()), (__gm__ DataType*)(this->targetsDataGm.GetPhyAddr()),
            (__gm__ DataType*)(this->inputLengthsGm.GetPhyAddr()),
            (__gm__ DataType*)(this->targetLengthsGm.GetPhyAddr()),
            (__gm__ T*)(this->negLogLikelihoodDataGm.GetPhyAddr()), (__gm__ T*)(this->logAlphaDataGm.GetPhyAddr()),
            (__ubuf__ DataType*)(targetOffsetTensor.GetPhyAddr()));
    }
    if constexpr (sizeof(ThreadType) == sizeof(int64_t)) {
        asc_vf_call<SimtCompute<T, DataType, ThreadType>>(
            dim3(blockDimX, blockDimY), batchSize, laInputStride, laBatchStride, lpBatchStride, maxInputLength,
            lpInputStride, targetsDim, tgBatchStride, blank, tgTargetStride,
            (__gm__ T*)(this->logProbsDataGm.GetPhyAddr()), (__gm__ DataType*)(this->targetsDataGm.GetPhyAddr()),
            (__gm__ DataType*)(this->inputLengthsGm.GetPhyAddr()),
            (__gm__ DataType*)(this->targetLengthsGm.GetPhyAddr()),
            (__gm__ T*)(this->negLogLikelihoodDataGm.GetPhyAddr()), (__gm__ T*)(this->logAlphaDataGm.GetPhyAddr()),
            (__ubuf__ DataType*)(targetOffsetTensor.GetPhyAddr()));
    }
}
} // namespace CTCLossV2
#endif // CTC_LOSS_V2