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
 * \file sigmoid_cross_entropy_with_logits_v2.cpp
 * \brief
 */

#include "sigmoid_cross_entropy_with_logits_v2.h"
#include "sigmoid_cross_entropy_with_logits_v2_tiling_key.h"
#include <type_traits>

using namespace AscendC;

// -------------------------------------------------------------------------
// 类成员函数实现
// -------------------------------------------------------------------------

template <typename T>
__aicore__ inline void KernelSigmoidCrossEntropyWithLogitsV2<T>::Init(
    GM_ADDR predict, GM_ADDR target, GM_ADDR weight, GM_ADDR posWeight, GM_ADDR loss,
    SigmoidCrossEntropyWithLogitsV2TilingData tiling)
{
    this->tiling = tiling;
    coreOffset = tiling.tileNum * tiling.tileLength * GetBlockIdx();
    uint32_t coreCapacity = tiling.tileNum * tiling.tileLength;
    if (coreOffset >= tiling.totalLength) {
        coreLength = 0;
    } else {
        uint32_t remaining = tiling.totalLength - coreOffset;
        coreLength = (remaining < coreCapacity) ? remaining : coreCapacity;
    }

    predictGm.SetGlobalBuffer((__gm__ T*)predict + coreOffset, coreLength);
    targetGm.SetGlobalBuffer((__gm__ T*)target + coreOffset, coreLength);
    lossGm.SetGlobalBuffer((__gm__ float*)loss + coreOffset, coreLength);

    if (tiling.has_weight) {
        weightGm.SetGlobalBuffer((__gm__ T*)weight + coreOffset, coreLength);
    }

    if (tiling.has_pos_weight) {
        posWeightGm.SetGlobalBuffer((__gm__ T*)posWeight + coreOffset, coreLength);
    }

    pipe.InitBuffer(inQueuePredict, BUFFER_NUM, tiling.tileLength * sizeof(T));
    pipe.InitBuffer(inQueueTarget, BUFFER_NUM, tiling.tileLength * sizeof(T));
    if (tiling.has_weight) {
        pipe.InitBuffer(inQueueWeight, BUFFER_NUM, tiling.tileLength * sizeof(T));
    }
    if (tiling.has_pos_weight) {
        pipe.InitBuffer(inQueuePosWeight, BUFFER_NUM, tiling.tileLength * sizeof(T));
    }
    pipe.InitBuffer(outQueueLoss, BUFFER_NUM, tiling.tileLength * sizeof(float));

    pipe.InitBuffer(calcBuf, FP32_BUF_NUM * tiling.tileLength * sizeof(float));
}

template <typename T>
__aicore__ inline void KernelSigmoidCrossEntropyWithLogitsV2<T>::Process()
{
    if (coreLength == 0) {
        return;
    }
    for (int32_t i = 0; i < static_cast<int32_t>(tiling.tileNum); i++) {
        uint32_t curLen = GetCurTileLength(i);
        if (curLen == 0) {
            break;
        }
        CopyIn(i, curLen);
        Compute(i, curLen);
        CopyOut(i, curLen);
    }
}

template <typename T>
__aicore__ inline uint32_t KernelSigmoidCrossEntropyWithLogitsV2<T>::GetCurTileLength(int32_t index) const
{
    uint32_t offset = static_cast<uint32_t>(index) * tiling.tileLength;
    if (offset >= coreLength) {
        return 0;
    }
    uint32_t remain = coreLength - offset;
    return (remain < tiling.tileLength) ? remain : tiling.tileLength;
}

template <typename T>
__aicore__ inline void KernelSigmoidCrossEntropyWithLogitsV2<T>::CopyIn(int32_t index, uint32_t curLen)
{
    bool fullTile = (curLen == tiling.tileLength);
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLen * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    LocalTensor<T> predictLocal = inQueuePredict.AllocTensor<T>();
    if (fullTile) {
        DataCopy(predictLocal, predictGm[index * tiling.tileLength], curLen);
    } else {
        DataCopyPad(predictLocal, predictGm[index * tiling.tileLength], copyParams, padParams);
    }
    inQueuePredict.EnQue(predictLocal);

    LocalTensor<T> targetLocal = inQueueTarget.AllocTensor<T>();
    if (fullTile) {
        DataCopy(targetLocal, targetGm[index * tiling.tileLength], curLen);
    } else {
        DataCopyPad(targetLocal, targetGm[index * tiling.tileLength], copyParams, padParams);
    }
    inQueueTarget.EnQue(targetLocal);

    if (tiling.has_weight) {
        LocalTensor<T> weightLocal = inQueueWeight.AllocTensor<T>();
        if (fullTile) {
            DataCopy(weightLocal, weightGm[index * tiling.tileLength], curLen);
        } else {
            DataCopyPad(weightLocal, weightGm[index * tiling.tileLength], copyParams, padParams);
        }
        inQueueWeight.EnQue(weightLocal);
    }

    if (tiling.has_pos_weight) {
        LocalTensor<T> posWLocal = inQueuePosWeight.AllocTensor<T>();
        if (fullTile) {
            DataCopy(posWLocal, posWeightGm[index * tiling.tileLength], curLen);
        } else {
            DataCopyPad(posWLocal, posWeightGm[index * tiling.tileLength], copyParams, padParams);
        }
        inQueuePosWeight.EnQue(posWLocal);
    }
}

template <typename T>
__aicore__ inline void KernelSigmoidCrossEntropyWithLogitsV2<T>::Compute(int32_t index, uint32_t curLen)
{
    uint32_t procLen = (curLen + 15U) & (~15U);
    if (procLen > tiling.tileLength) {
        procLen = tiling.tileLength;
    }
    if (procLen == 0) {
        return;
    }

    LocalTensor<T> predictLocal = inQueuePredict.DeQue<T>();
    LocalTensor<T> targetLocal = inQueueTarget.DeQue<T>();

    LocalTensor<float> calc_fp32 = calcBuf.Get<float>();
    LocalTensor<float> x_fp32 = calc_fp32;
    LocalTensor<float> y_fp32 = calc_fp32[tiling.tileLength];
    LocalTensor<float> tmp1 = calc_fp32[2 * tiling.tileLength];
    LocalTensor<float> tmp2 = calc_fp32[3 * tiling.tileLength];
    LocalTensor<float> tmp3 = calc_fp32[4 * tiling.tileLength];
    LocalTensor<float> tmp4 = calc_fp32[5 * tiling.tileLength];

    if constexpr (std::is_same<T, float>::value) {
        DataCopy(x_fp32, predictLocal, procLen);
        DataCopy(y_fp32, targetLocal, procLen);
    } else {
        Cast(x_fp32, predictLocal, RoundMode::CAST_NONE, procLen);
        Cast(y_fp32, targetLocal, RoundMode::CAST_NONE, procLen);
    }

    // max_val = max(-predict, 0)
    Muls(tmp1, x_fp32, -1.0f, procLen);
    Maxs(tmp2, tmp1, 0.0f, procLen);

    // max_val + ln(exp(-max_val) + exp(-predict - max_val))
    Muls(tmp3, tmp2, -1.0f, procLen);
    Exp(tmp4, tmp3, procLen);
    Sub(tmp3, tmp3, x_fp32, procLen);
    Exp(tmp3, tmp3, procLen);
    Add(tmp4, tmp4, tmp3, procLen);
    Ln(tmp4, tmp4, procLen);
    Add(tmp4, tmp4, tmp2, procLen);

    // (1 - target) * predict
    Muls(tmp1, y_fp32, -1.0f, procLen);
    Adds(tmp1, tmp1, 1.0f, procLen);
    Mul(tmp1, tmp1, x_fp32, procLen);

    LocalTensor<T> posWLocal;
    bool hasPosWeight = (tiling.has_pos_weight != 0);
    if (hasPosWeight) {
        posWLocal = inQueuePosWeight.DeQue<T>();
        LocalTensor<float> pos_w_vec_fp32 = tmp3;
        if constexpr (std::is_same<T, float>::value) {
            DataCopy(pos_w_vec_fp32, posWLocal, procLen);
        } else {
            Cast(pos_w_vec_fp32, posWLocal, RoundMode::CAST_NONE, procLen);
        }
        // Keep arithmetic order consistent with golden:
        // (pos_w * target + (1 - target)) * log_term
        Mul(pos_w_vec_fp32, pos_w_vec_fp32, y_fp32, procLen);
        Muls(tmp2, y_fp32, -1.0f, procLen);
        Adds(tmp2, tmp2, 1.0f, procLen);
        Add(pos_w_vec_fp32, pos_w_vec_fp32, tmp2, procLen);
        Mul(pos_w_vec_fp32, pos_w_vec_fp32, tmp4, procLen);
        Add(tmp1, tmp1, pos_w_vec_fp32, procLen);
    } else {
        Add(tmp1, tmp1, tmp4, procLen);
    }

    LocalTensor<T> wLocal;
    bool hasWeight = (tiling.has_weight != 0);
    if (hasWeight) {
        wLocal = inQueueWeight.DeQue<T>();
        LocalTensor<float> w_vec_fp32 = tmp2;
        if constexpr (std::is_same<T, float>::value) {
            DataCopy(w_vec_fp32, wLocal, procLen);
        } else {
            Cast(w_vec_fp32, wLocal, RoundMode::CAST_NONE, procLen);
        }
        Mul(tmp1, tmp1, w_vec_fp32, procLen);
    }

    LocalTensor<float> outLocal = outQueueLoss.AllocTensor<float>();
    DataCopy(outLocal, tmp1, procLen);
    outQueueLoss.EnQue(outLocal);

    if (hasPosWeight) {
        inQueuePosWeight.FreeTensor(posWLocal);
    }
    if (hasWeight) {
        inQueueWeight.FreeTensor(wLocal);
    }

    inQueuePredict.FreeTensor(predictLocal);
    inQueueTarget.FreeTensor(targetLocal);
}

template <typename T>
__aicore__ inline void KernelSigmoidCrossEntropyWithLogitsV2<T>::CopyOut(int32_t index, uint32_t curLen)
{
    LocalTensor<float> outLocal = outQueueLoss.DeQue<float>();
    if (curLen == tiling.tileLength) {
        DataCopy(lossGm[index * tiling.tileLength], outLocal, curLen);
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLen * sizeof(float)), 0, 0, 0};
        DataCopyPad(lossGm[index * tiling.tileLength], outLocal, copyParams);
    }
    outQueueLoss.FreeTensor(outLocal);
}

// -------------------------------------------------------------------------
// 入口函数：使用 DTYPE_PREDICT 进行模板实例化
// -------------------------------------------------------------------------
extern "C" __global__ __aicore__ void sigmoid_cross_entropy_with_logits_v2(
    GM_ADDR predict, GM_ADDR target, GM_ADDR weight, GM_ADDR posWeight, GM_ADDR loss, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SigmoidCrossEntropyWithLogitsV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(SigmoidCrossEntropyWithLogitsV2TilingData, tiling_data, tiling);

    KernelSigmoidCrossEntropyWithLogitsV2<DTYPE_PREDICT> op;
    op.Init(predict, target, weight, posWeight, loss, tiling_data);
    op.Process();
}
