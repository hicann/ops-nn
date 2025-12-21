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
 * \file batch_norm_grad_v3_common.h
 * \brief
 */
#ifndef __BATCH_NORM_GRAD_V3_BASE_COMMON_H__
#define __BATCH_NORM_GRAD_V3_BASE_COMMON_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

namespace BatchNormGradV3 {
using namespace AscendC;

#define CALC_TWO_TENSOR(FUNC_IMPL, RESP_TENSOR, CAL_0_TENSOR, CAL_1_TENSOR, N_LENGTH, L_LENGTH, FUNC_NAME)             \
    do {                                                                                                               \
        int64_t r0ForLoopNum = L_LENGTH / ELEM_PER_REP_FP32;                                                           \
        int64_t r0ForRemainNum = L_LENGTH % ELEM_PER_REP_FP32;                                                         \
        if ((r0ForLoopNum < N_LENGTH) && (L_LENGTH < (UINT8_MAX_NUM * B32_BLOCK_ALIGN_NUM))) {                         \
            uint8_t repStride = L_LENGTH / B32_BLOCK_ALIGN_NUM;                                                        \
            for (int64_t i = 0; i < r0ForLoopNum; i++) {                                                               \
                FUNC_IMPL(                                                                                             \
                    (RESP_TENSOR)[i * ELEM_PER_REP_FP32], (CAL_0_TENSOR)[i * ELEM_PER_REP_FP32],                       \
                    (CAL_1_TENSOR)[i * ELEM_PER_REP_FP32], ELEM_PER_REP_FP32, N_LENGTH,                                \
                    {1, 1, 1, repStride, repStride, repStride});                                                       \
            }                                                                                                          \
            if (r0ForRemainNum > 0) {                                                                                  \
                int64_t repeatForLoopNum = N_LENGTH / UINT8_MAX_NUM;                                                   \
                for (int64_t i = 0; i < repeatForLoopNum; i++) {                                                       \
                    FUNC_IMPL(                                                                                         \
                        (RESP_TENSOR)[r0ForLoopNum * ELEM_PER_REP_FP32 + i * UINT8_MAX_NUM * L_LENGTH],                \
                        (CAL_0_TENSOR)[r0ForLoopNum * ELEM_PER_REP_FP32 + i * UINT8_MAX_NUM * L_LENGTH],               \
                        (CAL_1_TENSOR)[r0ForLoopNum * ELEM_PER_REP_FP32 + i * UINT8_MAX_NUM * L_LENGTH],               \
                        r0ForRemainNum, UINT8_MAX_NUM, {1, 1, 1, repStride, repStride, repStride});                    \
                }                                                                                                      \
                if ((N_LENGTH % UINT8_MAX_NUM) > 0) {                                                                  \
                    FUNC_IMPL(                                                                                         \
                        (RESP_TENSOR)[r0ForLoopNum * ELEM_PER_REP_FP32 + repeatForLoopNum * UINT8_MAX_NUM * L_LENGTH], \
                        (CAL_0_TENSOR)                                                                                 \
                            [r0ForLoopNum * ELEM_PER_REP_FP32 + repeatForLoopNum * UINT8_MAX_NUM * L_LENGTH],          \
                        (CAL_1_TENSOR)                                                                                 \
                            [r0ForLoopNum * ELEM_PER_REP_FP32 + repeatForLoopNum * UINT8_MAX_NUM * L_LENGTH],          \
                        r0ForRemainNum, (N_LENGTH % UINT8_MAX_NUM), {1, 1, 1, repStride, repStride, repStride});       \
                }                                                                                                      \
            }                                                                                                          \
        } else {                                                                                                       \
            for (int64_t i = 0; i < N_LENGTH; i++) {                                                                   \
                FUNC_IMPL(                                                                                             \
                    (RESP_TENSOR)[L_LENGTH * i], (CAL_0_TENSOR)[L_LENGTH * i], (CAL_1_TENSOR)[L_LENGTH * i],           \
                    L_LENGTH);                                                                                         \
            }                                                                                                          \
        }                                                                                                              \
        PipeBarrier<PIPE_V>();                                                                                         \
    } while (0)

#define CALC_TWO_TENSOR_REPEAT(FUNC_IMPL, SCALAR_FUNC_IMPL, CAL_0_TENSOR, CAL_1_TENSOR, FUNC_TYPE, FUNC_NAME)     \
    do {                                                                                                          \
        int64_t bAlignLength = this->b1DimAlign * this->b0Dim;                                                    \
        int64_t r0ForLoopNum = bAlignLength / ELEM_PER_REP_FP32;                                                  \
        int64_t r0ForRemainNum = bAlignLength % ELEM_PER_REP_FP32;                                                \
        if ((r0ForLoopNum < this->cBlockLength) && (bAlignLength < (UINT8_MAX_NUM * B32_BLOCK_ALIGN_NUM))) {      \
            uint8_t repStride = bAlignLength / B32_BLOCK_ALIGN_NUM;                                               \
            for (int64_t i = 0; i < r0ForLoopNum; i++) {                                                          \
                FUNC_IMPL(                                                                                        \
                    (CAL_0_TENSOR)[i * ELEM_PER_REP_FP32], (CAL_0_TENSOR)[i * ELEM_PER_REP_FP32], CAL_1_TENSOR,   \
                    ELEM_PER_REP_FP32, this->cBlockLength, {1, 1, 0, repStride, repStride, 1});                   \
            }                                                                                                     \
            if (r0ForRemainNum > 0) {                                                                             \
                int64_t repeatForLoopNum = this->cBlockLength / UINT8_MAX_NUM;                                    \
                for (int64_t i = 0; i < repeatForLoopNum; i++) {                                                  \
                    FUNC_IMPL(                                                                                    \
                        (CAL_0_TENSOR)[r0ForLoopNum * ELEM_PER_REP_FP32 + i * UINT8_MAX_NUM * bAlignLength],      \
                        (CAL_0_TENSOR)[r0ForLoopNum * ELEM_PER_REP_FP32 + i * UINT8_MAX_NUM * bAlignLength],      \
                        (CAL_1_TENSOR)[i * UINT8_MAX_NUM * 8], r0ForRemainNum, UINT8_MAX_NUM,                     \
                        {1, 1, 0, repStride, repStride, 1});                                                      \
                }                                                                                                 \
                if ((this->cBlockLength % UINT8_MAX_NUM) > 0) {                                                   \
                    FUNC_IMPL(                                                                                    \
                        (CAL_0_TENSOR)                                                                            \
                            [r0ForLoopNum * ELEM_PER_REP_FP32 + repeatForLoopNum * UINT8_MAX_NUM * bAlignLength], \
                        (CAL_0_TENSOR)                                                                            \
                            [r0ForLoopNum * ELEM_PER_REP_FP32 + repeatForLoopNum * UINT8_MAX_NUM * bAlignLength], \
                        (CAL_1_TENSOR)[repeatForLoopNum * UINT8_MAX_NUM * 8], r0ForRemainNum,                     \
                        (this->cBlockLength % UINT8_MAX_NUM), {1, 1, 0, repStride, repStride, 1});                \
                }                                                                                                 \
            }                                                                                                     \
        } else {                                                                                                  \
            for (int64_t i = 0; i < this->cBlockLength; i++) {                                                    \
                if ((int64_t)(FUNC_TYPE) == -1) {                                                                 \
                    SCALAR_FUNC_IMPL(                                                                             \
                        (CAL_0_TENSOR)[bAlignLength * i], (CAL_0_TENSOR)[bAlignLength * i],                       \
                        -1 * (CAL_1_TENSOR).GetValue(i * 8), bAlignLength);                                       \
                } else if ((int64_t)(FUNC_TYPE) == -2) {                                                          \
                    SCALAR_FUNC_IMPL(                                                                             \
                        (CAL_0_TENSOR)[bAlignLength * i], (CAL_0_TENSOR)[bAlignLength * i],                       \
                        1 / (CAL_1_TENSOR).GetValue(i * 8), bAlignLength);                                        \
                } else {                                                                                          \
                    SCALAR_FUNC_IMPL(                                                                             \
                        (CAL_0_TENSOR)[bAlignLength * i], (CAL_0_TENSOR)[bAlignLength * i],                       \
                        (CAL_1_TENSOR).GetValue(i * 8), bAlignLength);                                            \
                }                                                                                                 \
            }                                                                                                     \
        }                                                                                                         \
        PipeBarrier<PIPE_V>();                                                                                    \
    } while (0)

constexpr uint32_t ONE = 1;
constexpr uint32_t TWO = 2;
constexpr uint32_t THREE = 3;
constexpr uint32_t FOUR = 4;
constexpr uint32_t ONE_BLK_SIZE = 32U;
constexpr uint32_t TWO_BLK_SIZE = ONE_BLK_SIZE * TWO;
constexpr uint32_t ELEM_PER_REP_FP32 = 64;
constexpr uint32_t UINT8_MAX_NUM = 255;
constexpr int64_t B32_BLOCK_ALIGN_NUM = 8;
constexpr int64_t B16_BLOCK_ALIGN_NUM = 16;
constexpr int32_t CALC_DIV_FLAG = -2;
constexpr int32_t CALC_SUB_FLAG = -1;
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t BUFFER_DEPTH = 1;
constexpr int64_t BLOCK_SIZE = 32U;

struct WeightMeanVarStruct {
    LocalTensor<float> weightTensor;
    LocalTensor<float> meanTensor;
    LocalTensor<float> varTensor;
    LocalTensor<float> weightBrcbTensor;
    LocalTensor<float> meanBrcbTensor;
    LocalTensor<float> varBrcbTensor;

    __aicore__ inline WeightMeanVarStruct()
    {}

    __aicore__ inline WeightMeanVarStruct(
        LocalTensor<float>& weightTensor, LocalTensor<float>& meanTensor, LocalTensor<float>& varTensor,
        LocalTensor<float>& weightBrcbTensor, LocalTensor<float>& meanBrcbTensor, LocalTensor<float>& varBrcbTensor)
        : weightTensor(weightTensor),
          meanTensor(meanTensor),
          varTensor(varTensor),
          weightBrcbTensor(weightBrcbTensor),
          meanBrcbTensor(meanBrcbTensor),
          varBrcbTensor(varBrcbTensor)
    {}
};

struct GMStruct {
    GM_ADDR dy;
    GM_ADDR x;
    GM_ADDR weight;
    GM_ADDR mean;
    GM_ADDR var;
    GM_ADDR dx;
    GM_ADDR dWeight;
    GM_ADDR dBias;
    GM_ADDR usrWorkspace;

    __aicore__ inline GMStruct()
    {}

    __aicore__ inline GMStruct(
        GM_ADDR dy, GM_ADDR x, GM_ADDR weight, GM_ADDR mean, GM_ADDR var, GM_ADDR dx, GM_ADDR dWeight, GM_ADDR dBias,
        GM_ADDR usrWorkspace)
        : dy(dy),
          x(x),
          weight(weight),
          mean(mean),
          var(var),
          dx(dx),
          dWeight(dWeight),
          dBias(dBias),
          usrWorkspace(usrWorkspace)
    {}
};

template <HardEvent event>
__aicore__ inline void TPipeSetWaitFlag()
{
    auto eventID = GetTPipePtr()->FetchEventID(event);
    SetFlag<event>(eventID);
    WaitFlag<event>(eventID);
}

__aicore__ inline uint32_t CeilDiv(uint32_t x, uint32_t y)
{
    if (y > 0) {
        return (x + y - 1) / y;
    }
    return 0;
}

__aicore__ inline uint32_t RoundUpOneBlock(uint32_t x)
{
    return (x + ONE_BLK_SIZE - 1) / ONE_BLK_SIZE * ONE_BLK_SIZE;
}

__aicore__ inline uint32_t RoundUpTwoBlock(uint32_t x)
{
    return (x + TWO_BLK_SIZE - 1) / TWO_BLK_SIZE * TWO_BLK_SIZE;
}

template <typename T>
__aicore__ inline int64_t GetAlignValue(int64_t value)
{
    if constexpr (IsSameType<T, float>::value) {
        return (value + B32_BLOCK_ALIGN_NUM - 1) / B32_BLOCK_ALIGN_NUM * B32_BLOCK_ALIGN_NUM;
    } else {
        return (value + B16_BLOCK_ALIGN_NUM - 1) / B16_BLOCK_ALIGN_NUM * B16_BLOCK_ALIGN_NUM;
    }
}

template <typename T>
__aicore__ inline void LoadOneTensor(
    GlobalTensor<T> input, LocalTensor<float> dst, DataCopyExtParams& dataCopyExtParams,
    DataCopyPadExtParams<T>& dataCopyPadExtParams, uint32_t number)
{
    PipeBarrier<PIPE_MTE2>();;
    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(dst, input, dataCopyExtParams, dataCopyPadExtParams);
    } else {
        LocalTensor<T> dstT = dst.ReinterpretCast<T>();
        int64_t offset = GetAlignValue<T>(number);
        DataCopyPad(dstT[offset], input, dataCopyExtParams, dataCopyPadExtParams);
        TPipeSetWaitFlag<HardEvent::MTE2_V>();
        Cast(dst, dstT[offset], RoundMode::CAST_NONE, number);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void StoreOneTensor(
    GlobalTensor<T> output, LocalTensor<float> src, DataCopyExtParams dataCopyExtParams, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        TPipeSetWaitFlag<HardEvent::V_MTE3>();
        DataCopyPad(output, src, dataCopyExtParams);
    } else {
        LocalTensor<T> srcT = src.ReinterpretCast<T>();
        Cast(srcT, src, RoundMode::CAST_RINT, offset);
        TPipeSetWaitFlag<HardEvent::V_MTE3>();
        DataCopyPad(output, srcT, dataCopyExtParams);
    }
    TPipeSetWaitFlag<HardEvent::MTE3_V>();
}

enum BNGV3SplitMode : int
{
    R0_SPLIT_MODE = 0,
    R1_SPLIT_MODE = 1,
};

template <typename T1, typename T2, typename T3>
class BatchNormGradV3Base {
public:
    __aicore__ inline BatchNormGradV3Base()
    {}

protected:
    int64_t b1Dim;
    int64_t aDim;
    int64_t b0Dim;
    int64_t bAlign;
    float epsilon;

    int64_t coreChannelNum;
    int64_t coreChannelNumTail;
    int64_t cUbBlock;
    int64_t needCoreNum;

    int64_t totalChannel;
    int64_t cBlockLength;
    int64_t cBlockLengthAlign;
    int64_t cBlockLengthT1Align;
    int64_t cBlockLengthT2Align;
    int64_t cBlockLengthT3Align;

    int64_t b1DimAlign;

    int64_t blockIdx;

    GlobalTensor<T1> dyGm_;
    GlobalTensor<T1> xGm_;
    GlobalTensor<T2> weightGm_;
    GlobalTensor<T3> meanGm_;
    GlobalTensor<T3> varGm_;

    GlobalTensor<T1> dxGm_;
    GlobalTensor<T2> dWeightGm_;
    GlobalTensor<T2> dBiasGm_;
    GlobalTensor<float> workSpaceGm_;

protected:
    /**
     * 由[c]广播为[c, 8]。
     */
    __aicore__ inline void BroadCastWeightMeanVar(WeightMeanVarStruct& weightMeanVarStruct)
    {
        int64_t repeatTime = CeilDiv(cBlockLength, B32_BLOCK_ALIGN_NUM);
        Adds(weightMeanVarStruct.varTensor, weightMeanVarStruct.varTensor, epsilon, cBlockLength);
        PipeBarrier<PIPE_V>();
        Sqrt(weightMeanVarStruct.varTensor, weightMeanVarStruct.varTensor, cBlockLength);
        PipeBarrier<PIPE_V>();
        Brcb(weightMeanVarStruct.weightBrcbTensor, weightMeanVarStruct.weightTensor, repeatTime, {1, 8});
        Brcb(weightMeanVarStruct.varBrcbTensor, weightMeanVarStruct.varTensor, repeatTime, {1, 8});
        Brcb(weightMeanVarStruct.meanBrcbTensor, weightMeanVarStruct.meanTensor, repeatTime, {1, 8});
        PipeBarrier<PIPE_V>();
    }

    /**
     * shape[nLength, length]累加到[nLength]
     */
    __aicore__ inline void DoNormalReduce(
        LocalTensor<float> inputTensor, LocalTensor<float> reduceTensor, LocalTensor<float> outputTensor,
        int64_t nLength, int64_t length, int64_t lengthAlign)
    {
        int64_t tmpLength = length;
        int64_t offset = 0;
        int64_t loopNum = length / ELEM_PER_REP_FP32;
        int64_t repStride = CeilDiv(length, ELEM_PER_REP_FP32) * 8;
        int64_t modNum = length % ELEM_PER_REP_FP32;

        // 1. 如果length较小，可以while循环每次Add nLength * 64个数，直到每行数量小于64时，用WholeRuduceSum累加
        // 2. 如果length超过跳转最大值，如果nLength比较小，可以循环nLength，使用Add减少单行的计算量，然后走1中逻辑
        // 3. 否则，逐行使用ReduceSum累加。
        // 由于开发时间问题，直接走3中逻辑
        for (int64_t i = 0; i < nLength; i++) {
            ReduceSum(
                reduceTensor[i * B32_BLOCK_ALIGN_NUM], inputTensor[i * lengthAlign], reduceTensor[i * lengthAlign],
                length);
        }
        TPipeSetWaitFlag<HardEvent::V_S>();
        for (int64_t i = 0; i < nLength; i++) {
            outputTensor.SetValue(i, reduceTensor.GetValue(i * B32_BLOCK_ALIGN_NUM));
        }
        TPipeSetWaitFlag<HardEvent::S_V>();
    }

    /**
     * shape[nLength, length]累加到[length]
     */
    __aicore__ inline void DoNormalReduce2(
        LocalTensor<float> inputTensor, LocalTensor<float> reduceTensor, LocalTensor<float> outputTensor,
        int64_t nLength, int64_t length)
    {
        int64_t tmpLength = 0;
        if (nLength > 1) {
            tmpLength = (nLength) / TWO * length;

            if (nLength % TWO == 0) {
                Add(outputTensor, inputTensor, inputTensor[tmpLength], tmpLength);
            } else {
                Add(outputTensor, inputTensor, inputTensor[tmpLength + length], tmpLength);
            }
            PipeBarrier<PIPE_V>();
            nLength = (nLength + 1) / TWO;
        }
        while (nLength > 1) {
            tmpLength = (nLength) / TWO * length;
            if (nLength % TWO == 0) {
                Add(outputTensor, outputTensor, outputTensor[tmpLength], tmpLength);
            } else {
                Add(outputTensor, outputTensor, outputTensor[tmpLength + length], tmpLength);
            }

            PipeBarrier<PIPE_V>();
            nLength = (nLength + 1) / TWO;
        }
    }
};
} // namespace BatchNormGradV3
#endif
