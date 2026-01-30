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
 * \file ctc_loss_v2_fp16.h
 * \brief ctc_loss_v2_fp16
 */

#ifndef CTC_LOSS_V2_FP16
#define CTC_LOSS_V2_FP16

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "ctc_loss_v2_base.h"

namespace CTCLossV2 {
using namespace AscendC;

template <typename T, typename DataType, typename ThreadType> class CTCLossV2FP16 : public CTCLossV2Base<T, DataType, ThreadType> {
public:
    __aicore__ inline CTCLossV2FP16(){};
    __aicore__ inline void Init(GM_ADDR log_probs, GM_ADDR targets, GM_ADDR input_lengths, GM_ADDR target_lengths,
        GM_ADDR neg_log_likelihood, GM_ADDR log_alpha, GM_ADDR workspace,
        const CTCLossV2TilingData4AscendC *tilingData);

    __aicore__ inline void Process();

private:
    GlobalTensor<float> tmpFloatData;
};

template <typename T, typename DataType, typename ThreadType>
__aicore__ inline void CTCLossV2FP16<T, DataType, ThreadType>::Init(GM_ADDR log_probs, GM_ADDR targets, GM_ADDR input_lengths,
    GM_ADDR target_lengths, GM_ADDR neg_log_likelihood, GM_ADDR log_alpha, GM_ADDR workspace,
    const CTCLossV2TilingData4AscendC *tilingData)
{
    this->BaseInit(log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, workspace,
        tilingData);
    tmpFloatData.SetGlobalBuffer((__gm__ float *)(workspace));
}

template <typename DataType, typename ThreadType>
__aicore__ inline DataType ProcessTgBatchOffsetsFp16(__gm__ DataType *targetLengthsGm, int32_t targetsDim,
    int32_t tgBatchStride, int32_t idx)
{
    if (targetsDim == 1) {
        ThreadType sum = 0;
        for (int32_t i = 0; i < idx; i++) {
            sum += targetLengthsGm[i];
        }
        return sum;
    } else {
        return tgBatchStride * idx;
    }
}

template <typename T, typename DataType, typename ThreadType>
__aicore__ __attribute__((always_inline)) inline void CalcLogAlphaFp16(int32_t batchSize, int32_t laInputStride, 
    ThreadType laBatchStride, ThreadType lpBatchStride, int32_t maxInputLength, ThreadType lpInputStride, int32_t targetsDim, 
    int32_t tgBatchStride, int32_t blank, int32_t tgTargetStride, __gm__ T *logProbsGm, __gm__ DataType *targetsGm, 
    __gm__ DataType *inputLengthsGm, __gm__ DataType *targetLengthsGm, __gm__ T *negLogLikelihoodGm, __gm__ T *logAlphaGm, __gm__ float *tmpDataGm)
{
    int32_t threadIdy = Simt::GetThreadIdx<1>();
    int32_t threadIdx = Simt::GetThreadIdx<0>();
    int32_t blockDimx = Simt::GetThreadNum<0>();
    int32_t blockDimy = Simt::GetThreadNum<1>();
    int32_t blkIdx = Simt::GetBlockIdx();
    int32_t blockNum = Simt::GetBlockNum();
    for (int32_t b = threadIdy + blkIdx * blockDimy; b < batchSize; b += blockNum * blockDimy) {
        ThreadType inputLength = inputLengthsGm[b];
        ThreadType targetLength = targetLengthsGm[b];
        ThreadType lpBatchOffset = b * lpBatchStride;
        ThreadType laBatchOffset = b * laBatchStride;
        ThreadType tgBatchOffset = ProcessTgBatchOffsetsFp16<DataType, ThreadType>(targetLengthsGm, targetsDim, tgBatchStride, b);

        if (inputLength == 0) {
            if (Simt::GetThreadIdx<0>() == 0) {
                float log_likelihood = targetLength == 0 ? 0 : neginf;
                negLogLikelihoodGm[b] = -static_cast<T>(log_likelihood);
            }
            continue;
        }

        for (int32_t block_s = 0; block_s < laInputStride; block_s += blockDimx) {
            int32_t s = threadIdx + block_s;
            float la = 0;
            switch (s) {
                case 0:
                    la = static_cast<float>(logProbsGm[lpBatchOffset + blank]);
                    break;
                case 1:
                    la = targetLength == 0 ? neginf :
                                             static_cast<float>(logProbsGm[lpBatchOffset +
                        GetTargetPrime<DataType>(targetsGm, tgBatchOffset, tgTargetStride, int32_t(1), blank)]);
                    break;
                default:
                    la = neginf;
            }
            if (s < laInputStride) {
                tmpDataGm[laBatchOffset + s] = la;
                logAlphaGm[laBatchOffset + s] = static_cast<T>(la);
            }
        }
        for (int32_t block_s = 0; block_s < laInputStride; block_s += blockDimx) {
            int32_t s = threadIdx + block_s;
            int32_t currentChar;
            bool haveThree;
            if (s < 2 * targetLength + 1 && targetLength > 0) {
                currentChar = GetTargetPrime<DataType>(targetsGm, tgBatchOffset, tgTargetStride, s, blank);
                haveThree = ((s > 1) &&
                    (GetTargetPrime<DataType>(targetsGm, tgBatchOffset, tgTargetStride, s - 2, blank) != currentChar));
            } else {
                currentChar = blank;
                haveThree = false;
            }
            for (int32_t t = 1; t < maxInputLength; t++) {
                Simt::ThreadBarrier();
                if ((t < inputLength) && (s < 2 * targetLength + 1)) {
                    float la1 = tmpDataGm[laBatchOffset + laInputStride * (t - 1) + s];
                    float lamax = la1;
                    float la2 = 0;
                    float la3 = 0;
                    if (s > 0) {
                        la2 = tmpDataGm[laBatchOffset + laInputStride * (t - 1) + (s - 1)];
                        if (la2 > lamax) {
                            lamax = la2;
                        }
                    } else {
                        la2 = neginf;
                    }
                    if (haveThree) {
                        la3 = tmpDataGm[laBatchOffset + laInputStride * (t - 1) + (s - 2)];
                        if (la3 > lamax) {
                            lamax = la3;
                        }
                    } else {
                        la3 = neginf;
                    }
                    if (lamax == neginf) {
                        lamax = 0;
                    }
                    tmpDataGm[laBatchOffset + laInputStride * t + s] =
                        Simt::Log1p(Simt::Exp(la1 - lamax) + Simt::Exp(la2 - lamax) + Simt::Exp(la3 - lamax) - 1) +
                        lamax + static_cast<float>(logProbsGm[lpBatchOffset + t * lpInputStride + currentChar]);
                    logAlphaGm[laBatchOffset + laInputStride * t + s] =
                        static_cast<T>(tmpDataGm[laBatchOffset + laInputStride * t + s]);
                }
            }
        }
        Simt::ThreadBarrier();

        // compute the loss
        if (threadIdx == 0) {
            float l1 = tmpDataGm[laBatchOffset + laInputStride * (inputLength - 1) + (targetLength * 2)];
            float l2 = targetLength > 0 ?
                tmpDataGm[laBatchOffset + laInputStride * (inputLength - 1) + (targetLength * 2 - 1)] :
                neginf;
            float m = ((l1 > l2) ? l1 : l2);
            m = ((m == neginf) ? 0 : m);
            float log_likelihood = Simt::Log1p(Simt::Exp(l1 - m) + Simt::Exp(l2 - m) - 1) + m;
            negLogLikelihoodGm[b] = -static_cast<T>(log_likelihood);
        }
    }
}

template <typename T, typename DataType, typename ThreadType>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_512)__aicore__
    void SimtComputeFp16(int32_t batchSize, int32_t laInputStride, ThreadType laBatchStride, ThreadType lpBatchStride,
    int32_t maxInputLength, ThreadType lpInputStride, int32_t targetsDim, int32_t tgBatchStride, int32_t blank,
    int32_t tgTargetStride, __gm__ T *logProbsGm, __gm__ DataType *targetsGm, __gm__ DataType *inputLengthsGm,
    __gm__ DataType *targetLengthsGm, __gm__ T *negLogLikelihoodGm, __gm__ T *logAlphaGm, __gm__ float *tmpDataGm) 
    {
        CalcLogAlphaFp16<T, DataType, ThreadType>(batchSize, laInputStride, laBatchStride, lpBatchStride, maxInputLength, lpInputStride, targetsDim, tgBatchStride, 
                        blank, tgTargetStride, logProbsGm, targetsGm, inputLengthsGm, targetLengthsGm, negLogLikelihoodGm, logAlphaGm, tmpDataGm);
    }

template <typename T, typename DataType, typename ThreadType>
__simt_vf__ LAUNCH_BOUND(THREAD_NUM_1024)__aicore__
    void SimtComputeFp16Int32(int32_t batchSize, int32_t laInputStride, ThreadType laBatchStride, ThreadType lpBatchStride,
    int32_t maxInputLength, ThreadType lpInputStride, int32_t targetsDim, int32_t tgBatchStride, int32_t blank,
    int32_t tgTargetStride, __gm__ T *logProbsGm, __gm__ DataType *targetsGm, __gm__ DataType *inputLengthsGm,
    __gm__ DataType *targetLengthsGm, __gm__ T *negLogLikelihoodGm, __gm__ T *logAlphaGm, __gm__ float *tmpDataGm) 
    {
        CalcLogAlphaFp16<T, DataType, ThreadType>(batchSize, laInputStride, laBatchStride, lpBatchStride, maxInputLength, lpInputStride, targetsDim, tgBatchStride, 
                        blank, tgTargetStride, logProbsGm, targetsGm, inputLengthsGm, targetLengthsGm, negLogLikelihoodGm, logAlphaGm, tmpDataGm);
    }

template <typename T, typename DataType, typename ThreadType> __aicore__ inline void CTCLossV2FP16<T, DataType, ThreadType>::Process()
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
        Simt::VF_CALL<SimtComputeFp16Int32<T, DataType, ThreadType>>(Simt::Dim3(blockDimX, blockDimY), batchSize, laInputStride, laBatchStride,
            lpBatchStride, maxInputLength, lpInputStride, targetsDim, tgBatchStride, blank, tgTargetStride,
            (__gm__ T *)(this->logProbsDataGm.GetPhyAddr()), (__gm__ DataType *)(this->targetsDataGm.GetPhyAddr()),
            (__gm__ DataType *)(this->inputLengthsGm.GetPhyAddr()), (__gm__ DataType *)(this->targetLengthsGm.GetPhyAddr()),
            (__gm__ T *)(this->negLogLikelihoodDataGm.GetPhyAddr()), (__gm__ T *)(this->logAlphaDataGm.GetPhyAddr()),
            (__gm__ float *)(tmpFloatData.GetPhyAddr()));
    }
    if constexpr (sizeof(ThreadType) == sizeof(int64_t)) {
        Simt::VF_CALL<SimtComputeFp16<T, DataType, ThreadType>>(Simt::Dim3(blockDimX, blockDimY), batchSize, laInputStride, laBatchStride,
            lpBatchStride, maxInputLength, lpInputStride, targetsDim, tgBatchStride, blank, tgTargetStride,
            (__gm__ T *)(this->logProbsDataGm.GetPhyAddr()), (__gm__ DataType *)(this->targetsDataGm.GetPhyAddr()),
            (__gm__ DataType *)(this->inputLengthsGm.GetPhyAddr()), (__gm__ DataType *)(this->targetLengthsGm.GetPhyAddr()),
            (__gm__ T *)(this->negLogLikelihoodDataGm.GetPhyAddr()), (__gm__ T *)(this->logAlphaDataGm.GetPhyAddr()),
            (__gm__ float *)(tmpFloatData.GetPhyAddr()));
    }
}
} // namespace CTCLossV2
#endif // CTC_LOSS_V2_FP16