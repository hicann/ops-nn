/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sigmoid_cross_entropy_with_logits_grad_v2.h"
#include <type_traits>

using namespace AscendC;

template <typename T, bool HAS_WEIGHT, bool HAS_POS_WEIGHT>
__aicore__ inline void KernelSigmoidCrossEntropyWithLogitsGradV2<T, HAS_WEIGHT, HAS_POS_WEIGHT>::Init(
    GM_ADDR predict, GM_ADDR target, GM_ADDR dout, GM_ADDR gradient, GM_ADDR weight, GM_ADDR pos_weight,
    const SigmoidCrossEntropyWithLogitsGradV2TilingData& tiling)
{
    this->tiling = tiling;

    uint64_t blockIdx = GetBlockIdx();
    uint64_t globalBufferIndex = tiling.bigCoreDataNum * blockIdx;
    this->tileDataNum = tiling.tileDataNum;

    if (blockIdx < tiling.tailBlockNum) {
        this->coreDataNum = tiling.bigCoreDataNum;
        this->tileNum = tiling.finalBigTileNum;
        this->tailDataNum = tiling.bigTailDataNum;
    } else {
        this->coreDataNum = tiling.smallCoreDataNum;
        this->tileNum = tiling.finalSmallTileNum;
        this->tailDataNum = tiling.smallTailDataNum;
        globalBufferIndex -= (tiling.bigCoreDataNum - tiling.smallCoreDataNum) * (blockIdx - tiling.tailBlockNum);
    }

    predictGm.SetGlobalBuffer((__gm__ T*)predict + globalBufferIndex, this->coreDataNum);
    targetGm.SetGlobalBuffer((__gm__ T*)target + globalBufferIndex, this->coreDataNum);
    doutGm.SetGlobalBuffer((__gm__ T*)dout + globalBufferIndex, this->coreDataNum);
    gradGm.SetGlobalBuffer((__gm__ T*)gradient + globalBufferIndex, this->coreDataNum);

    if constexpr (HAS_WEIGHT) {
        weightGm.SetGlobalBuffer((__gm__ T*)weight + globalBufferIndex, this->coreDataNum);
    }
    if constexpr (HAS_POS_WEIGHT) {
        posWeightGm.SetGlobalBuffer((__gm__ T*)pos_weight + globalBufferIndex, this->coreDataNum);
    }

    uint64_t tileBytes = this->tileDataNum * sizeof(T);
    pipe.InitBuffer(inQueuePredict, BUFFER_NUM, tileBytes);
    pipe.InitBuffer(inQueueTarget, BUFFER_NUM, tileBytes);
    pipe.InitBuffer(inQueueDout, BUFFER_NUM, tileBytes);
    if constexpr (HAS_WEIGHT) {
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, tileBytes);
    }
    if constexpr (HAS_POS_WEIGHT) {
        pipe.InitBuffer(inQueuePosWeight, BUFFER_NUM, tileBytes);
    }
    pipe.InitBuffer(outQueueGrad, BUFFER_NUM, tileBytes);
    if constexpr (std::is_same<T, float>::value) {
        uint64_t fp32BufNum = (tiling.fp32_lite_mode == 1) ? 3 : FP32_BUF_NUM;
        pipe.InitBuffer(calcBuf, fp32BufNum * this->tileDataNum * sizeof(float));
    } else {
        pipe.InitBuffer(calcBuf, FP32_BUF_NUM * this->tileDataNum * sizeof(float));
    }
}

template <typename T, bool HAS_WEIGHT, bool HAS_POS_WEIGHT>
__aicore__ inline void KernelSigmoidCrossEntropyWithLogitsGradV2<T, HAS_WEIGHT, HAS_POS_WEIGHT>::SetProcessDataNum(
    int32_t index, int32_t loopCount)
{
    if (index == loopCount - 1) {
        this->processDataNum = this->tailDataNum;
    } else {
        this->processDataNum = this->tileDataNum;
    }
}

template <typename T, bool HAS_WEIGHT, bool HAS_POS_WEIGHT>
__aicore__ inline void KernelSigmoidCrossEntropyWithLogitsGradV2<T, HAS_WEIGHT, HAS_POS_WEIGHT>::Process()
{
    int32_t loopCount = static_cast<int32_t>(this->tileNum);
    if (loopCount <= 0) {
        return;
    }

    for (int32_t i = 0; i < loopCount; ++i) {
        SetProcessDataNum(i, loopCount);
        CopyIn(i);

        Compute(i);
        CopyOut(i);
    }
}

template <typename T, bool HAS_WEIGHT, bool HAS_POS_WEIGHT>
__aicore__ inline void KernelSigmoidCrossEntropyWithLogitsGradV2<T, HAS_WEIGHT, HAS_POS_WEIGHT>::CopyIn(
    int32_t progress)
{
    bool fullTile = (this->processDataNum == this->tileDataNum);
    bool usePadCopy = !fullTile;
    if constexpr (std::is_same<T, float>::value) {
        // For tiny fp32 tiles, pad copy overhead can exceed scalar fallback benefits.
        usePadCopy = usePadCopy && (this->tileDataNum >= 1024);
    }
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->processDataNum * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    LocalTensor<T> predictLocal = inQueuePredict.AllocTensor<T>();
    if (!usePadCopy) {
        DataCopy(predictLocal, predictGm[progress * this->tileDataNum], this->processDataNum);
    } else {
        DataCopyPad(predictLocal, predictGm[progress * this->tileDataNum], copyParams, padParams);
    }
    inQueuePredict.EnQue(predictLocal);

    LocalTensor<T> targetLocal = inQueueTarget.AllocTensor<T>();
    if (!usePadCopy) {
        DataCopy(targetLocal, targetGm[progress * this->tileDataNum], this->processDataNum);
    } else {
        DataCopyPad(targetLocal, targetGm[progress * this->tileDataNum], copyParams, padParams);
    }
    inQueueTarget.EnQue(targetLocal);

    LocalTensor<T> doutLocal = inQueueDout.AllocTensor<T>();
    if (!usePadCopy) {
        DataCopy(doutLocal, doutGm[progress * this->tileDataNum], this->processDataNum);
    } else {
        DataCopyPad(doutLocal, doutGm[progress * this->tileDataNum], copyParams, padParams);
    }
    inQueueDout.EnQue(doutLocal);

    if constexpr (HAS_WEIGHT) {
        LocalTensor<T> weightLocal = inQueueWeight.AllocTensor<T>();
        if (!usePadCopy) {
            DataCopy(weightLocal, weightGm[progress * this->tileDataNum], this->processDataNum);
        } else {
            DataCopyPad(weightLocal, weightGm[progress * this->tileDataNum], copyParams, padParams);
        }
        inQueueWeight.EnQue(weightLocal);
    }

    if constexpr (HAS_POS_WEIGHT) {
        LocalTensor<T> posWLocal = inQueuePosWeight.AllocTensor<T>();
        if (!usePadCopy) {
            DataCopy(posWLocal, posWeightGm[progress * this->tileDataNum], this->processDataNum);
        } else {
            DataCopyPad(posWLocal, posWeightGm[progress * this->tileDataNum], copyParams, padParams);
        }
        inQueuePosWeight.EnQue(posWLocal);
    }
}

template <typename T, bool HAS_WEIGHT, bool HAS_POS_WEIGHT>
__aicore__ inline void KernelSigmoidCrossEntropyWithLogitsGradV2<T, HAS_WEIGHT, HAS_POS_WEIGHT>::Compute(
    int32_t progress)
{
    LocalTensor<T> xLocal = inQueuePredict.DeQue<T>();
    LocalTensor<T> yLocal = inQueueTarget.DeQue<T>();
    LocalTensor<T> doutLocal = inQueueDout.DeQue<T>();
    LocalTensor<T> wLocal, posWLocal;

    if constexpr (HAS_WEIGHT) {
        wLocal = inQueueWeight.DeQue<T>();
    }
    if constexpr (HAS_POS_WEIGHT) {
        posWLocal = inQueuePosWeight.DeQue<T>();
    }

    LocalTensor<float> calc_fp32 = calcBuf.Get<float>();
    LocalTensor<float> x_fp32;
    LocalTensor<float> y_fp32;
    LocalTensor<float> tmp1;
    LocalTensor<float> tmp2;
    LocalTensor<float> tmp3;

    uint64_t calCount = this->processDataNum;
    uint64_t procCount = calCount;
    if constexpr (std::is_same<T, float>::value) {
        if (this->tileDataNum >= 1024) {
            procCount = (calCount + 15U) & (~15U);
            if (procCount > this->tileDataNum) {
                procCount = this->tileDataNum;
            }
        }
    }
    if (procCount == 0) {
        return;
    }

    if constexpr (std::is_same<T, float>::value) {
        x_fp32 = xLocal;
        y_fp32 = yLocal;
        if (tiling.fp32_lite_mode == 1) {
            tmp1 = calc_fp32[0];
            tmp2 = calc_fp32[this->tileDataNum];
            tmp3 = calc_fp32[2 * this->tileDataNum];
        } else {
            tmp1 = calc_fp32[2 * this->tileDataNum];
            tmp2 = calc_fp32[3 * this->tileDataNum];
            tmp3 = calc_fp32[4 * this->tileDataNum];
        }
    } else {
        x_fp32 = calc_fp32;
        y_fp32 = calc_fp32[this->tileDataNum];
        tmp1 = calc_fp32[2 * this->tileDataNum];
        tmp2 = calc_fp32[3 * this->tileDataNum];
        tmp3 = calc_fp32[4 * this->tileDataNum];
        Cast(x_fp32, xLocal, RoundMode::CAST_NONE, calCount);
        Cast(y_fp32, yLocal, RoundMode::CAST_NONE, calCount);
    }

    if constexpr (HAS_POS_WEIGHT) {
        LocalTensor<float> pos_w_vec_fp32;
        if constexpr (std::is_same<T, float>::value) {
            pos_w_vec_fp32 = posWLocal;
        } else {
            pos_w_vec_fp32 = tmp2;
            Cast(pos_w_vec_fp32, posWLocal, RoundMode::CAST_NONE, procCount);
        }

        // Keep formula same as TBE while aligning instruction order closer to builtin trace.
        // log_weight = pos_weight * target
        // weight_sub = log_weight + 1 - target
        Mul(tmp1, pos_w_vec_fp32, y_fp32, procCount); // tmp1 = log_weight
        Adds(tmp3, tmp1, 1.0f, procCount);            // tmp3 = log_weight + 1
        Sub(tmp3, tmp3, y_fp32, procCount);           // tmp3 = log_weight + 1 - target

        // sigmoid(x) = 1 / (1 + exp(-x))
        Muls(tmp2, x_fp32, -1.0f, procCount); // tmp2 = -x
        Exp(tmp2, tmp2, procCount);           // tmp2 = exp(-x)
        Adds(y_fp32, tmp2, 1.0f, procCount);  // y_fp32 = 1 + exp(-x)
        Duplicate(tmp2, 1.0f, procCount);
        Div(x_fp32, tmp2, y_fp32, procCount); // x_fp32 = sigmoid(x)

        // grad_base = ((log_weight + 1 - target) * sigmoid(x) - log_weight)
        Mul(tmp3, tmp3, x_fp32, procCount); // tmp3 = (log_weight + 1 - target) * sigmoid(x)
        Sub(x_fp32, tmp3, tmp1, procCount); // x_fp32 = grad_base
    } else {
        // Align with TBE path: sigmoid(x) = 1 / (1 + exp(-x)).
        Muls(tmp1, x_fp32, -1.0f, procCount); // tmp1 = -x
        Exp(tmp1, tmp1, procCount);           // tmp1 = exp(-x)
        Adds(tmp3, tmp1, 1.0f, procCount);    // tmp3 = 1 + exp(-x)
        Duplicate(tmp2, 1.0f, procCount);
        Div(x_fp32, tmp2, tmp3, procCount); // x_fp32 = sigmoid(x)

        Sub(x_fp32, x_fp32, y_fp32, procCount); // grad_base = p - y
    }

    LocalTensor<float> dout_fp32;
    if constexpr (std::is_same<T, float>::value) {
        dout_fp32 = doutLocal;
    } else {
        dout_fp32 = y_fp32;
        Cast(dout_fp32, doutLocal, RoundMode::CAST_NONE, procCount);
    }
    Mul(x_fp32, x_fp32, dout_fp32, procCount);

    if constexpr (HAS_WEIGHT) {
        LocalTensor<float> w_vec_fp32;
        if constexpr (std::is_same<T, float>::value) {
            w_vec_fp32 = wLocal;
        } else {
            w_vec_fp32 = tmp1;
            Cast(w_vec_fp32, wLocal, RoundMode::CAST_NONE, procCount);
        }
        Mul(x_fp32, x_fp32, w_vec_fp32, procCount);
    }

    if (tiling.globalScale != 1.0f) {
        Muls(x_fp32, x_fp32, tiling.globalScale, procCount);
    }

    LocalTensor<T> outLocal = outQueueGrad.AllocTensor<T>();
    if constexpr (std::is_same<T, float>::value) {
        DataCopy(outLocal, x_fp32, procCount);
    } else if constexpr (std::is_same<T, bfloat16_t>::value) {
        Cast(outLocal, x_fp32, RoundMode::CAST_RINT, procCount);
    } else {
        Cast(outLocal, x_fp32, RoundMode::CAST_NONE, procCount);
    }

    outQueueGrad.EnQue(outLocal);

    if constexpr (HAS_WEIGHT) {
        inQueueWeight.FreeTensor(wLocal);
    }
    if constexpr (HAS_POS_WEIGHT) {
        inQueuePosWeight.FreeTensor(posWLocal);
    }
    inQueuePredict.FreeTensor(xLocal);
    inQueueTarget.FreeTensor(yLocal);
    inQueueDout.FreeTensor(doutLocal);
}

template <typename T, bool HAS_WEIGHT, bool HAS_POS_WEIGHT>
__aicore__ inline void KernelSigmoidCrossEntropyWithLogitsGradV2<T, HAS_WEIGHT, HAS_POS_WEIGHT>::CopyOut(
    int32_t progress)
{
    LocalTensor<T> outLocal = outQueueGrad.DeQue<T>();
    bool usePadCopy = (this->processDataNum != this->tileDataNum);
    if constexpr (std::is_same<T, float>::value) {
        usePadCopy = usePadCopy && (this->tileDataNum >= 1024);
    }

    if (!usePadCopy) {
        DataCopy(gradGm[progress * this->tileDataNum], outLocal, this->processDataNum);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->processDataNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(gradGm[progress * this->tileDataNum], outLocal, copyParams);
    }
    outQueueGrad.FreeTensor(outLocal);
}

#define DISPATCH_SIGMOID_GRAD(TYPE)                                                \
    if (tiling_data.has_weight && tiling_data.has_pos_weight) {                    \
        KernelSigmoidCrossEntropyWithLogitsGradV2<TYPE, true, true> op;            \
        op.Init(predict, target, dout, gradient, weight, pos_weight, tiling_data); \
        op.Process();                                                              \
    } else if (tiling_data.has_weight && !tiling_data.has_pos_weight) {            \
        KernelSigmoidCrossEntropyWithLogitsGradV2<TYPE, true, false> op;           \
        op.Init(predict, target, dout, gradient, weight, pos_weight, tiling_data); \
        op.Process();                                                              \
    } else if (!tiling_data.has_weight && tiling_data.has_pos_weight) {            \
        KernelSigmoidCrossEntropyWithLogitsGradV2<TYPE, false, true> op;           \
        op.Init(predict, target, dout, gradient, weight, pos_weight, tiling_data); \
        op.Process();                                                              \
    } else {                                                                       \
        KernelSigmoidCrossEntropyWithLogitsGradV2<TYPE, false, false> op;          \
        op.Init(predict, target, dout, gradient, weight, pos_weight, tiling_data); \
        op.Process();                                                              \
    }

extern "C" __global__ __aicore__ void sigmoid_cross_entropy_with_logits_grad_v2(
    GM_ADDR predict, GM_ADDR target, GM_ADDR dout, GM_ADDR weight, GM_ADDR pos_weight, GM_ADDR gradient,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SigmoidCrossEntropyWithLogitsGradV2TilingData);
    GET_TILING_DATA(tiling_data, tiling);

    using InputT = DTYPE_PREDICT;
    DISPATCH_SIGMOID_GRAD(InputT)
}
