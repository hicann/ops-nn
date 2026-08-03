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
 * \file nll_loss_grad.h
 * \brief NllLossGrad 算子 kernel 类定义
 */

#ifndef NLLLOSSGRAD_H
#define NLLLOSSGRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "nll_loss_grad_tiling_data.h"
#include "nll_loss_grad_tiling_key.h"

namespace NsNllLossGrad {

using namespace AscendC;

constexpr int64_t BLOCK_FP32 = 8; // 32B / 4B
constexpr int64_t REDUCTION_NONE = 0;
constexpr int64_t REDUCTION_SUM = 1;
constexpr int64_t REDUCTION_MEAN = 2;

template <typename T, typename TargetT>
class NllLossGrad {
public:
    __aicore__ inline NllLossGrad(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y_grad, GM_ADDR target, GM_ADDR weight, GM_ADDR total_weight,
                                GM_ADDR x_grad, const NllLossGradTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessNormal();
    __aicore__ inline void ProcessBigWeight();
    __aicore__ inline float LoadScalarF(const GlobalTensor<T>& gm, int64_t idx);
    __aicore__ inline float ComputeScale();

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> weightBuf_;     // float [cAlign] 常驻
    TBuf<TPosition::VECCALC> weightTBuf_;    // T [cAlign] (bf16/half weight load)
    TBuf<TPosition::VECCALC> outFloatBuf_;   // float scatter 构造缓冲(低精度/BigWeight)
    TBuf<TPosition::VECCALC> outTBuf_;       // T 输出缓冲(BigWeight 低精度)
    TBuf<TPosition::VECCALC> yGradFloatBuf_; // float [lineTile] (none & 低精度 cast)
    TBuf<TPosition::VECCALC> scalarBuf_;     // scratch for scalar loads

    // NormalWeight 主路径双缓冲队列
    TQue<TPosition::VECIN, 2> inTargetQue_; // TargetT [lineTile]
    TQue<TPosition::VECIN, 2> inYGradQue_;  // T [lineTile] (reduction=none)
    TQue<TPosition::VECOUT, 2> outQue_;     // T [lineTile*cDim]

    GlobalTensor<T> yGradGm_;
    GlobalTensor<TargetT> targetGm_;
    GlobalTensor<T> weightGm_;
    GlobalTensor<T> totalWeightGm_;
    GlobalTensor<T> xGradGm_;

    NllLossGradTilingData tiling_;
    int64_t startLine_ = 0;
    int64_t lineCount_ = 0;
};

template <typename T, typename TargetT>
__aicore__ inline void NllLossGrad<T, TargetT>::Init(GM_ADDR x, GM_ADDR y_grad, GM_ADDR target, GM_ADDR weight,
                                                     GM_ADDR total_weight, GM_ADDR x_grad,
                                                     const NllLossGradTilingData* tilingData)
{
    tiling_ = *tilingData;
    yGradGm_.SetGlobalBuffer((__gm__ T*)y_grad);
    targetGm_.SetGlobalBuffer((__gm__ TargetT*)target);
    weightGm_.SetGlobalBuffer((__gm__ T*)weight);
    totalWeightGm_.SetGlobalBuffer((__gm__ T*)total_weight);
    xGradGm_.SetGlobalBuffer((__gm__ T*)x_grad);

    int64_t blockIdx = GetBlockIdx();
    if (blockIdx < tiling_.redundantLine) {
        startLine_ = blockIdx * tiling_.maxLine;
        lineCount_ = tiling_.maxLine;
    } else {
        startLine_ = tiling_.redundantLine * tiling_.maxLine + (blockIdx - tiling_.redundantLine) * tiling_.lowerLine;
        lineCount_ = tiling_.lowerLine;
    }

    int64_t tgtAlignBytes = (tiling_.lineTile * (int64_t)sizeof(TargetT) + 31) / 32 * 32;
    if (tgtAlignBytes < 32) {
        tgtAlignBytes = 32;
    }
    int64_t yGradAlignBytes = ((tiling_.lineTile + BLOCK_FP32 - 1) / BLOCK_FP32 * BLOCK_FP32) * (int64_t)sizeof(T);
    if (yGradAlignBytes < 32) {
        yGradAlignBytes = 32;
    }
    if (tiling_.bigWeight == 0) {
        pipe_.InitBuffer(weightBuf_, tiling_.cAlign * sizeof(float));
        pipe_.InitBuffer(inTargetQue_, 2, tgtAlignBytes);
        pipe_.InitBuffer(outQue_, 2, tiling_.outUbSize * sizeof(T));
        if (tiling_.reduction == REDUCTION_NONE) {
            pipe_.InitBuffer(inYGradQue_, 2, yGradAlignBytes);
        }
        if (sizeof(T) != sizeof(float)) {
            pipe_.InitBuffer(weightTBuf_, tiling_.cAlign * sizeof(T));
            pipe_.InitBuffer(outFloatBuf_, tiling_.outUbSize * sizeof(float));
            pipe_.InitBuffer(yGradFloatBuf_,
                             ((tiling_.lineTile + BLOCK_FP32 - 1) / BLOCK_FP32 * BLOCK_FP32) * sizeof(float));
        }
    } else {
        int64_t colAlign = (tiling_.colTile + BLOCK_FP32 - 1) / BLOCK_FP32 * BLOCK_FP32;
        pipe_.InitBuffer(outFloatBuf_, colAlign * sizeof(float));
        if (sizeof(T) != sizeof(float)) {
            pipe_.InitBuffer(outTBuf_, colAlign * sizeof(T));
        }
    }
    pipe_.InitBuffer(scalarBuf_, 32 * 4);
}

template <typename T, typename TargetT>
__aicore__ inline float NllLossGrad<T, TargetT>::LoadScalarF(const GlobalTensor<T>& gm, int64_t idx)
{
    LocalTensor<T> tmpT = scalarBuf_.Get<T>();
    DataCopyExtParams params{1, (uint32_t)sizeof(T), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(tmpT, gm[idx], params, padParams);
    pipe_barrier(PIPE_ALL);
    if (sizeof(T) != sizeof(float)) {
        LocalTensor<float> tmpF = scalarBuf_.GetWithOffset<float>(8, 64);
        Cast(tmpF, tmpT, RoundMode::CAST_NONE, 8);
        pipe_barrier(PIPE_ALL);
        return tmpF.GetValue(0);
    }
    return (float)tmpT.GetValue(0);
}

template <typename T, typename TargetT>
__aicore__ inline float NllLossGrad<T, TargetT>::ComputeScale()
{
    // returns scale so that grad = -weight[t] * scale
    if (tiling_.reduction == REDUCTION_SUM) {
        return LoadScalarF(yGradGm_, 0);
    }
    // mean: div_no_nan(y_grad, total_weight)
    float y = LoadScalarF(yGradGm_, 0);
    float tw = LoadScalarF(totalWeightGm_, 0);
    float absTw = tw < 0 ? -tw : tw;
    if (absTw > 0) {
        return y / tw;
    }
    return 0.0f;
}

template <typename T, typename TargetT>
__aicore__ inline void NllLossGrad<T, TargetT>::Process()
{
    if (lineCount_ <= 0) {
        return;
    }
    if (tiling_.bigWeight == 0) {
        ProcessNormal();
    } else {
        ProcessBigWeight();
    }
}

template <typename T, typename TargetT>
__aicore__ inline void NllLossGrad<T, TargetT>::ProcessNormal()
{
    int64_t cDim = tiling_.cDim;
    int64_t cAlign = tiling_.cAlign;
    LocalTensor<float> weightF = weightBuf_.Get<float>();

    // 常驻预取 weight[0:C] 并 cast 到 float
    if constexpr (sizeof(T) != sizeof(float)) {
        LocalTensor<T> weightT = weightTBuf_.Get<T>();
        DataCopyExtParams wp{1, (uint32_t)(cDim * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padT{false, 0, 0, 0};
        DataCopyPad(weightT, weightGm_, wp, padT);
        pipe_barrier(PIPE_ALL);
        Cast(weightF, weightT, RoundMode::CAST_NONE, cAlign);
    } else {
        DataCopyExtParams wp{1, (uint32_t)(cDim * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padF{false, 0, 0, 0};
        DataCopyPad(weightF, weightGm_, wp, padF);
    }
    pipe_barrier(PIPE_ALL);

    // sum/mean：每 core 预乘 -scale；none：仅取负
    float preScale = -1.0f;
    if (tiling_.reduction != REDUCTION_NONE) {
        preScale = -ComputeScale();
    }
    Muls(weightF, weightF, preScale, cAlign);
    pipe_barrier(PIPE_ALL);

    constexpr bool lowPrec = (sizeof(T) != sizeof(float));
    int64_t lineTile = tiling_.lineTile;
    int64_t done = 0;
    while (done < lineCount_) {
        int64_t cur = lineCount_ - done;
        if (cur > lineTile) {
            cur = lineTile;
        }
        int64_t rowBase = startLine_ + done;
        int64_t outLen = cur * cDim;
        int64_t outLenAlign = (outLen + BLOCK_FP32 - 1) / BLOCK_FP32 * BLOCK_FP32;

        // ---- CopyIn（MTE2，双缓冲）----
        LocalTensor<TargetT> targetL = inTargetQue_.AllocTensor<TargetT>();
        DataCopyExtParams tp{1, (uint32_t)(cur * sizeof(TargetT)), 0, 0, 0};
        DataCopyPadExtParams<TargetT> padI{false, 0, 0, 0};
        DataCopyPad(targetL, targetGm_[rowBase], tp, padI);
        inTargetQue_.EnQue(targetL);

        LocalTensor<T> yGradTile;
        if (tiling_.reduction == REDUCTION_NONE) {
            yGradTile = inYGradQue_.AllocTensor<T>();
            DataCopyExtParams yp{1, (uint32_t)(cur * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padY{false, 0, 0, 0};
            DataCopyPad(yGradTile, yGradGm_[rowBase], yp, padY);
            inYGradQue_.EnQue(yGradTile);
        }

        // ---- Compute（Vector/Scalar）----
        LocalTensor<TargetT> targetC = inTargetQue_.DeQue<TargetT>();
        LocalTensor<float> yGradF;
        LocalTensor<T> yGradC;
        if (tiling_.reduction == REDUCTION_NONE) {
            yGradC = inYGradQue_.DeQue<T>();
            if constexpr (lowPrec) {
                yGradF = yGradFloatBuf_.Get<float>();
                Cast(yGradF, yGradC, RoundMode::CAST_NONE, (cur + BLOCK_FP32 - 1) / BLOCK_FP32 * BLOCK_FP32);
                pipe_barrier(PIPE_ALL);
            } else {
                yGradF = yGradC.template ReinterpretCast<float>();
            }
        }

        LocalTensor<T> outTile = outQue_.AllocTensor<T>();
        LocalTensor<float> outF = lowPrec ? outFloatBuf_.Get<float>() : outTile.template ReinterpretCast<float>();
        Duplicate(outF, 0.0f, outLenAlign);
        pipe_barrier(PIPE_ALL);

        for (int64_t i = 0; i < cur; i++) {
            int64_t t = (int64_t)targetC.GetValue(i);
            if (t >= 0 && t < cDim && t != tiling_.ignoreIndex) {
                float grad = weightF.GetValue(t);
                if (tiling_.reduction == REDUCTION_NONE) {
                    grad = grad * yGradF.GetValue(i);
                }
                outF.SetValue(i * cDim + t, grad);
            }
        }
        inTargetQue_.FreeTensor(targetC);
        if (tiling_.reduction == REDUCTION_NONE) {
            inYGradQue_.FreeTensor(yGradC);
        }
        if constexpr (lowPrec) {
            pipe_barrier(PIPE_ALL);
            Cast(outTile, outF, RoundMode::CAST_RINT, outLenAlign);
            pipe_barrier(PIPE_ALL);
        } else {
            pipe_barrier(PIPE_ALL);
        }
        outQue_.EnQue(outTile);

        // ---- CopyOut（MTE3，双缓冲，与下一 tile 的 MTE2/Vector 重叠）----
        LocalTensor<T> outC = outQue_.DeQue<T>();
        DataCopyExtParams op{1, (uint32_t)(outLen * sizeof(T)), 0, 0, 0};
        DataCopyPad(xGradGm_[rowBase * cDim], outC, op);
        outQue_.FreeTensor(outC);

        done += cur;
    }
}

template <typename T, typename TargetT>
__aicore__ inline void NllLossGrad<T, TargetT>::ProcessBigWeight()
{
    int64_t cDim = tiling_.cDim;
    int64_t colTile = tiling_.colTile;
    LocalTensor<float> outF = outFloatBuf_.Get<float>();

    float scale = 0.0f;
    if (tiling_.reduction != REDUCTION_NONE) {
        scale = ComputeScale();
    }

    // 单块路径：整行可放入一个 colTile。清零一次，逐行仅写/复位一个位置。
    if (tiling_.moveOutTime == 1) {
        int64_t cAlignLocal = (cDim + BLOCK_FP32 - 1) / BLOCK_FP32 * BLOCK_FP32;
        Duplicate(outF, 0.0f, cAlignLocal);
        pipe_barrier(PIPE_ALL);
        for (int64_t r = 0; r < lineCount_; r++) {
            int64_t row = startLine_ + r;
            int64_t t = (int64_t)targetGm_.GetValue(row);
            bool valid = (t >= 0 && t < cDim && t != tiling_.ignoreIndex);
            if (valid) {
                float w = LoadScalarF(weightGm_, t);
                float s = scale;
                if (tiling_.reduction == REDUCTION_NONE) {
                    s = LoadScalarF(yGradGm_, row);
                }
                outF.SetValue(t, -w * s);
                pipe_barrier(PIPE_ALL);
            }
            DataCopyExtParams op{1, (uint32_t)(cDim * sizeof(T)), 0, 0, 0};
            if constexpr (sizeof(T) != sizeof(float)) {
                LocalTensor<T> outT = outTBuf_.Get<T>();
                Cast(outT, outF, RoundMode::CAST_RINT, cAlignLocal);
                pipe_barrier(PIPE_ALL);
                DataCopyPad(xGradGm_[row * cDim], outT, op);
            } else {
                DataCopyPad(xGradGm_[row * cDim], outF, op);
            }
            pipe_barrier(PIPE_ALL);
            if (valid) {
                outF.SetValue(t, 0.0f); // 复位，保持清零供下一行复用
                pipe_barrier(PIPE_ALL);
            }
        }
        return;
    }

    // 多块路径（超大 C 兜底）：按列 tile 清零 + 写单个有效位置
    for (int64_t r = 0; r < lineCount_; r++) {
        int64_t row = startLine_ + r;
        int64_t t = (int64_t)targetGm_.GetValue(row);
        bool valid = (t >= 0 && t < cDim && t != tiling_.ignoreIndex);
        float grad = 0.0f;
        if (valid) {
            float w = LoadScalarF(weightGm_, t);
            float s = scale;
            if (tiling_.reduction == REDUCTION_NONE) {
                s = LoadScalarF(yGradGm_, row);
            }
            grad = -w * s;
        }

        for (int64_t colBase = 0; colBase < cDim; colBase += colTile) {
            int64_t colLen = cDim - colBase;
            if (colLen > colTile) {
                colLen = colTile;
            }
            Duplicate(outF, 0.0f, (colLen + BLOCK_FP32 - 1) / BLOCK_FP32 * BLOCK_FP32);
            pipe_barrier(PIPE_ALL);
            if (valid && t >= colBase && t < colBase + colLen) {
                outF.SetValue(t - colBase, grad);
            }
            pipe_barrier(PIPE_ALL);
            DataCopyExtParams op{1, (uint32_t)(colLen * sizeof(T)), 0, 0, 0};
            if constexpr (sizeof(T) != sizeof(float)) {
                LocalTensor<T> outT = outTBuf_.Get<T>();
                Cast(outT, outF, RoundMode::CAST_RINT, (colLen + BLOCK_FP32 - 1) / BLOCK_FP32 * BLOCK_FP32);
                pipe_barrier(PIPE_ALL);
                DataCopyPad(xGradGm_[row * cDim + colBase], outT, op);
            } else {
                DataCopyPad(xGradGm_[row * cDim + colBase], outF, op);
            }
            pipe_barrier(PIPE_ALL);
        }
    }
}

} // namespace NsNllLossGrad
#endif // NLLLOSSGRAD_H
