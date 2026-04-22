/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/*!
 * \file apply_proximal_gradient_descent.h
 * \brief ApplyProximalGradientDescent Kernel 实现（arch35）
 *
 * 公式：
 *   prox = var - alpha * delta
 *   varOut = sign(prox) * max(|prox| - alpha*l1, 0) / (1 + alpha*l2)
 *
 * 模板参数：
 *   - T: 数据类型（float / half）
 *   - BUFFER_MODE: 缓冲模式（0=单缓冲, 1=双缓冲）
 */
#ifndef APPLY_PROXIMAL_GRADIENT_DESCENT_H
#define APPLY_PROXIMAL_GRADIENT_DESCENT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "apply_proximal_gradient_descent_tiling_data.h"
#include "apply_proximal_gradient_descent_tiling_key.h"

namespace NsApplyProximalGradientDescent {

using namespace AscendC;

// scalar UB buffer 预留 32B / sizeof(T) 元素，保证 DataCopy 32B 对齐
constexpr int32_t SCALAR_BYTES = 32;

template <typename T, int BUFFER_MODE>
class ApplyProximalGradientDescent {
    static constexpr int32_t BUFFER_NUM = BUFFER_MODE ? 2 : 1;
    static constexpr bool IS_HALF = std::is_same<T, half>::value;

public:
    __aicore__ inline ApplyProximalGradientDescent() {}

    __aicore__ inline void Init(
        GM_ADDR var, GM_ADDR alpha, GM_ADDR l1, GM_ADDR l2, GM_ADDR delta,
        GM_ADDR varOut, const ApplyProximalGradientDescentTilingData* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void LoadScalars(GM_ADDR alpha, GM_ADDR l1, GM_ADDR l2);
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueVar;
    TQue<QuePosition::VECIN,  BUFFER_NUM> inQueueDelta;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueVar;

    // FP32 中间量（FP16 场景也是 FP32 buffer）
    TBuf<QuePosition::VECCALC> tmpProxBuf;
    TBuf<QuePosition::VECCALC> tmpAbsBuf;
    TBuf<QuePosition::VECCALC> tmpSignBuf;
    // FP16 场景额外的 FP32 Cast 副本
    TBuf<QuePosition::VECCALC> tmpVarF32Buf;
    TBuf<QuePosition::VECCALC> tmpDelF32Buf;
    // 标量 UB buffer（各 32B）
    TBuf<QuePosition::VECCALC> scalarAlphaBuf;
    TBuf<QuePosition::VECCALC> scalarL1Buf;
    TBuf<QuePosition::VECCALC> scalarL2Buf;

    GlobalTensor<T> varGm;
    GlobalTensor<T> deltaGm;
    GlobalTensor<T> varOutGm;

    int64_t blockLength_ = 0;
    int64_t ubLength_    = 0;

    // 组合后的 host-like 标量（FP32 计算口径）
    float alphaS_   = 0.f;
    float alphaL1_  = 0.f;
    float invScale_ = 1.f;
};

// ------------------------------ Init ------------------------------

template <typename T, int BUFFER_MODE>
__aicore__ inline void ApplyProximalGradientDescent<T, BUFFER_MODE>::LoadScalars(
    GM_ADDR alpha, GM_ADDR l1, GM_ADDR l2)
{
    // 将 3 个标量分别搬到 UB，GetValue 读取，再组合成 host-like scalar
    GlobalTensor<T> alphaGm;
    GlobalTensor<T> l1Gm;
    GlobalTensor<T> l2Gm;
    alphaGm.SetGlobalBuffer((__gm__ T*)alpha, 1);
    l1Gm.SetGlobalBuffer((__gm__ T*)l1, 1);
    l2Gm.SetGlobalBuffer((__gm__ T*)l2, 1);

    AscendC::LocalTensor<T> alphaUb = scalarAlphaBuf.Get<T>();
    AscendC::LocalTensor<T> l1Ub    = scalarL1Buf.Get<T>();
    AscendC::LocalTensor<T> l2Ub    = scalarL2Buf.Get<T>();

    AscendC::DataCopyParams scalarParams;
    scalarParams.blockCount = 1;
    scalarParams.blockLen   = sizeof(T);   // 读 1 个元素
    scalarParams.srcStride  = 0;
    scalarParams.dstStride  = 0;
    AscendC::DataCopyPad(alphaUb, alphaGm, scalarParams, {false, 0, 0, 0});
    AscendC::DataCopyPad(l1Ub,    l1Gm,    scalarParams, {false, 0, 0, 0});
    AscendC::DataCopyPad(l2Ub,    l2Gm,    scalarParams, {false, 0, 0, 0});

    // 等待 DMA 完成
    AscendC::PipeBarrier<PIPE_ALL>();

    float alphaF;
    float l1F;
    float l2F;
    if constexpr (IS_HALF) {
        alphaF = static_cast<float>(alphaUb.GetValue(0));
        l1F    = static_cast<float>(l1Ub.GetValue(0));
        l2F    = static_cast<float>(l2Ub.GetValue(0));
    } else {
        alphaF = alphaUb.GetValue(0);
        l1F    = l1Ub.GetValue(0);
        l2F    = l2Ub.GetValue(0);
    }
    alphaS_   = alphaF;
    alphaL1_  = alphaF * l1F;
    invScale_ = 1.0f / (1.0f + alphaF * l2F);
}

template <typename T, int BUFFER_MODE>
__aicore__ inline void ApplyProximalGradientDescent<T, BUFFER_MODE>::Init(
    GM_ADDR var, GM_ADDR alpha, GM_ADDR l1, GM_ADDR l2, GM_ADDR delta,
    GM_ADDR varOut, const ApplyProximalGradientDescentTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    ubLength_ = tilingData->ubFactor;

    // 空 tensor / 本核无数据：直接返回
    if (blockLength_ == 0 || ubLength_ == 0) {
        return;
    }

    varGm.SetGlobalBuffer((__gm__ T*)var + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    deltaGm.SetGlobalBuffer((__gm__ T*)delta + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    varOutGm.SetGlobalBuffer((__gm__ T*)varOut + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    pipe.InitBuffer(inQueueVar,   BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(inQueueDelta, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outQueueVar,  BUFFER_NUM, ubLength_ * sizeof(T));

    pipe.InitBuffer(tmpProxBuf, ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpAbsBuf,  ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpSignBuf, ubLength_ * sizeof(float));
    if constexpr (IS_HALF) {
        pipe.InitBuffer(tmpVarF32Buf, ubLength_ * sizeof(float));
        pipe.InitBuffer(tmpDelF32Buf, ubLength_ * sizeof(float));
    }

    // 标量 UB buffer：每个 32B（容纳 FP16 或 FP32 单元素）
    pipe.InitBuffer(scalarAlphaBuf, SCALAR_BYTES);
    pipe.InitBuffer(scalarL1Buf,    SCALAR_BYTES);
    pipe.InitBuffer(scalarL2Buf,    SCALAR_BYTES);

    LoadScalars(alpha, l1, l2);
}

// ------------------------------ CopyIn ------------------------------

template <typename T, int BUFFER_MODE>
__aicore__ inline void ApplyProximalGradientDescent<T, BUFFER_MODE>::CopyIn(
    int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> varLocal   = inQueueVar.template AllocTensor<T>();
    AscendC::LocalTensor<T> deltaLocal = inQueueDelta.template AllocTensor<T>();

    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen   = currentNum * sizeof(T);
    copyParams.srcStride  = 0;
    copyParams.dstStride  = 0;
    AscendC::DataCopyPad(varLocal,   varGm[progress * ubLength_],   copyParams, {false, 0, 0, 0});
    AscendC::DataCopyPad(deltaLocal, deltaGm[progress * ubLength_], copyParams, {false, 0, 0, 0});

    inQueueVar.EnQue(varLocal);
    inQueueDelta.EnQue(deltaLocal);
}

// ------------------------------ Compute ------------------------------

template <typename T, int BUFFER_MODE>
__aicore__ inline void ApplyProximalGradientDescent<T, BUFFER_MODE>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> varLocal   = inQueueVar.template DeQue<T>();
    AscendC::LocalTensor<T> deltaLocal = inQueueDelta.template DeQue<T>();
    AscendC::LocalTensor<T> outLocal   = outQueueVar.template AllocTensor<T>();

    AscendC::LocalTensor<float> tmpProx = tmpProxBuf.Get<float>();
    AscendC::LocalTensor<float> tmpAbs  = tmpAbsBuf.Get<float>();
    AscendC::LocalTensor<float> tmpSign = tmpSignBuf.Get<float>();

    // 准备 FP32 源
    AscendC::LocalTensor<float> srcVarF32;
    AscendC::LocalTensor<float> srcDelF32;
    if constexpr (IS_HALF) {
        AscendC::LocalTensor<float> varF32 = tmpVarF32Buf.Get<float>();
        AscendC::LocalTensor<float> delF32 = tmpDelF32Buf.Get<float>();
        AscendC::Cast(varF32, varLocal,   AscendC::RoundMode::CAST_NONE, currentNum);
        AscendC::Cast(delF32, deltaLocal, AscendC::RoundMode::CAST_NONE, currentNum);
        srcVarF32 = varF32;
        srcDelF32 = delF32;
    } else {
        // FP32：直接 reinterpret（T == float）
        srcVarF32 = varLocal.template ReinterpretCast<float>();
        srcDelF32 = deltaLocal.template ReinterpretCast<float>();
    }

    // prox = var - alpha * delta
    AscendC::Muls(tmpProx, srcDelF32, alphaS_, currentNum);       // α·δ
    AscendC::Sub (tmpProx, srcVarF32, tmpProx, currentNum);       // prox = var - α·δ

    // |prox|
    AscendC::Abs(tmpAbs, tmpProx, currentNum);

    // |prox| - α·l1
    AscendC::Adds(tmpAbs, tmpAbs, -alphaL1_, currentNum);

    // relu
    AscendC::Maxs(tmpAbs, tmpAbs, 0.0f, currentNum);

    // sign(prox)
    AscendC::Sign(tmpSign, tmpProx, currentNum);

    // sign * relu
    AscendC::Mul(tmpProx, tmpSign, tmpAbs, currentNum);

    // * invScale，最终结果直接写入 outLocal（避免 UB->UB DataCopy 的对齐/调度问题）
    if constexpr (IS_HALF) {
        AscendC::Muls(tmpProx, tmpProx, invScale_, currentNum);
        AscendC::Cast(outLocal, tmpProx, AscendC::RoundMode::CAST_RINT, currentNum);
    } else {
        AscendC::LocalTensor<float> outF32 = outLocal.template ReinterpretCast<float>();
        AscendC::Muls(outF32, tmpProx, invScale_, currentNum);
    }

    outQueueVar.template EnQue<T>(outLocal);
    inQueueVar.FreeTensor(varLocal);
    inQueueDelta.FreeTensor(deltaLocal);
}

// ------------------------------ CopyOut ------------------------------

template <typename T, int BUFFER_MODE>
__aicore__ inline void ApplyProximalGradientDescent<T, BUFFER_MODE>::CopyOut(
    int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> outLocal = outQueueVar.template DeQue<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen   = currentNum * sizeof(T);
    copyParams.srcStride  = 0;
    copyParams.dstStride  = 0;
    AscendC::DataCopyPad(varOutGm[progress * ubLength_], outLocal, copyParams);
    outQueueVar.FreeTensor(outLocal);
}

// ------------------------------ Process ------------------------------

template <typename T, int BUFFER_MODE>
__aicore__ inline void ApplyProximalGradientDescent<T, BUFFER_MODE>::Process()
{
    if (blockLength_ == 0 || ubLength_ == 0) {
        return;
    }
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsApplyProximalGradientDescent

#endif // APPLY_PROXIMAL_GRADIENT_DESCENT_H
