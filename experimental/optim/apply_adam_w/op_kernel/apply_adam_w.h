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
 * \file apply_adam_w.h
 * \brief
 */
#ifndef APPLY_ADAM_W_H
#define APPLY_ADAM_W_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "apply_adam_w_tiling_data.h"
#include "apply_adam_w_tiling_key.h"

namespace MyApplyAdamW {

using namespace AscendC;

template <typename TYPE_X, uint64_t BUFFER_NUM>
class KernelApplyAdamW {
public:
    __aicore__ inline KernelApplyAdamW(){};

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR m, GM_ADDR v, GM_ADDR beta1_power, GM_ADDR beta2_power, GM_ADDR lr,
                                GM_ADDR weight_decay, GM_ADDR beta1, GM_ADDR beta2, GM_ADDR epsilon, GM_ADDR grad,
                                GM_ADDR max_grad_norm, GM_ADDR var_out, GM_ADDR m_out, GM_ADDR v_out,
                                const ApplyAdamWTilingData* tilingData, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void DoRunOp(int32_t offset);
    // fp16 与 bf16 共用低精度计算路径：先将 half/bfloat16 数据 cast 到 float 计算，再 cast 回原类型
    __aicore__ inline void DoRunOpLowPrecision(int32_t offset);

private:
    TPipe* pipe;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> bindQue;
    TBuf<QuePosition::VECCALC> tmpBuf;
    TBuf<QuePosition::VECCALC> scalarTmpBuf, scalarCastBuf;
    GlobalTensor<TYPE_X> varGm, mGm, vGm, gradGm, mgnGm;
    GlobalTensor<TYPE_X> varOutGm, mOutGm, vOutGm;
    GlobalTensor<TYPE_X> beta1PowGm, beta2PowGm, lrGm, wdGm, beta1Gm, beta2Gm, epsGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    float beta1PowValue;
    float beta1Value;
    float beta2PowValue;
    float beta2Value;
    float lrValue, wdValue;
    float epsilonValue;
    float negBeta1S1;
    float negBeta2S1;
    float varTP;
    float negBeta2PowOutS1;
    float lrDBeta1PowOutS1;
    uint32_t amsgrad;
    uint32_t maximize;
    int32_t eventIDMTE2ToV;
    int32_t eventIDVToMTE3;
};

template <typename TYPE_X, uint64_t BUFFER_NUM>
__aicore__ inline void KernelApplyAdamW<TYPE_X, BUFFER_NUM>::Init(GM_ADDR var, GM_ADDR m, GM_ADDR v,
                                                                  GM_ADDR beta1_power, GM_ADDR beta2_power, GM_ADDR lr,
                                                                  GM_ADDR weight_decay, GM_ADDR beta1, GM_ADDR beta2,
                                                                  GM_ADDR epsilon, GM_ADDR grad, GM_ADDR max_grad_norm,
                                                                  GM_ADDR var_out, GM_ADDR m_out, GM_ADDR v_out,
                                                                  const ApplyAdamWTilingData* tilingData, TPipe* pipeIn)
{
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    uint32_t coreNum = GetBlockIdx();
    uint32_t globalBufferIndex = tilingData->bigCoreDataNum * GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    this->pipe = pipeIn;
    if (coreNum < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) *
                             (GetBlockIdx() - tilingData->tailBlockNum);
    }

    varGm.SetGlobalBuffer((__gm__ TYPE_X*)var + globalBufferIndex, this->coreDataNum);
    mGm.SetGlobalBuffer((__gm__ TYPE_X*)m + globalBufferIndex, this->coreDataNum);
    vGm.SetGlobalBuffer((__gm__ TYPE_X*)v + globalBufferIndex, this->coreDataNum);
    gradGm.SetGlobalBuffer((__gm__ TYPE_X*)grad + globalBufferIndex, this->coreDataNum);
    mgnGm.SetGlobalBuffer((__gm__ TYPE_X*)max_grad_norm + globalBufferIndex, this->coreDataNum);
    varOutGm.SetGlobalBuffer((__gm__ TYPE_X*)var_out + globalBufferIndex, this->coreDataNum);
    mOutGm.SetGlobalBuffer((__gm__ TYPE_X*)m_out + globalBufferIndex, this->coreDataNum);
    vOutGm.SetGlobalBuffer((__gm__ TYPE_X*)v_out + globalBufferIndex, this->coreDataNum);

    beta1PowGm.SetGlobalBuffer((__gm__ TYPE_X*)beta1_power, 16);
    beta2PowGm.SetGlobalBuffer((__gm__ TYPE_X*)beta2_power, 16);
    lrGm.SetGlobalBuffer((__gm__ TYPE_X*)lr, 16);
    wdGm.SetGlobalBuffer((__gm__ TYPE_X*)weight_decay, 16);
    beta1Gm.SetGlobalBuffer((__gm__ TYPE_X*)beta1, 16);
    beta2Gm.SetGlobalBuffer((__gm__ TYPE_X*)beta2, 16);
    epsGm.SetGlobalBuffer((__gm__ TYPE_X*)epsilon, 16);

    if constexpr (IsSameType<TYPE_X, bfloat16_t>::value || IsSameType<TYPE_X, half>::value) {
        pipe->InitBuffer(scalarTmpBuf, 16 * sizeof(TYPE_X));
        pipe->InitBuffer(scalarCastBuf, 16 * sizeof(float));
        LocalTensor<TYPE_X> castLocal = scalarTmpBuf.Get<TYPE_X>();
        LocalTensor<float> castLocalFp = scalarCastBuf.Get<float>();
        int32_t eventIDSToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        int32_t eventIDVToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        castLocal.SetValue(0, beta1PowGm.GetValue(0));
        castLocal.SetValue(1, beta2PowGm.GetValue(0));
        castLocal.SetValue(2, lrGm.GetValue(0));
        castLocal.SetValue(3, wdGm.GetValue(0));
        castLocal.SetValue(4, beta1Gm.GetValue(0));
        castLocal.SetValue(5, beta2Gm.GetValue(0));
        castLocal.SetValue(6, epsGm.GetValue(0));
        SetFlag<HardEvent::S_V>(eventIDSToV);
        WaitFlag<HardEvent::S_V>(eventIDSToV);
        Cast(castLocalFp, castLocal, RoundMode::CAST_NONE, 16);
        SetFlag<HardEvent::V_S>(eventIDVToS);
        WaitFlag<HardEvent::V_S>(eventIDVToS);
        this->beta1PowValue = castLocalFp.GetValue(0);
        this->beta2PowValue = castLocalFp.GetValue(1);
        this->lrValue = castLocalFp.GetValue(2);
        this->wdValue = castLocalFp.GetValue(3);
        this->beta1Value = castLocalFp.GetValue(4);
        this->beta2Value = castLocalFp.GetValue(5);
        this->epsilonValue = castLocalFp.GetValue(6);
        pipe->Reset();
    }
    if constexpr (IsSameType<TYPE_X, float>::value) {
        this->beta1PowValue = beta1PowGm.GetValue(0);
        this->beta2PowValue = beta2PowGm.GetValue(0);
        this->lrValue = lrGm.GetValue(0);
        this->wdValue = wdGm.GetValue(0);
        this->beta1Value = beta1Gm.GetValue(0);
        this->beta2Value = beta2Gm.GetValue(0);
        this->epsilonValue = epsGm.GetValue(0);
    }
    this->negBeta1S1 = 1.0f - this->beta1Value;
    this->negBeta2S1 = 1.0f - this->beta2Value;
    this->varTP = 1.0f - this->lrValue * this->wdValue;
    this->negBeta2PowOutS1 = 1.0f / (1.0f - this->beta2PowValue * this->beta2Value);
    this->lrDBeta1PowOutS1 = this->lrValue / (this->beta1PowValue * this->beta1Value - 1.0f);

    this->amsgrad = tilingData->amsgrad;
    this->maximize = tilingData->maximize;
    this->eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    this->eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    if (this->amsgrad) {
        pipe->InitBuffer(bindQue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X) * 5);
    } else {
        pipe->InitBuffer(bindQue, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X) * 4);
    }
    if constexpr (IsSameType<TYPE_X, bfloat16_t>::value || IsSameType<TYPE_X, half>::value) {
        // tmpBuf 布局（float，每段 processDataNum 个元素）：
        // [0] varLocalFp [1] mLocalFp [2] gradLocalFp [3] vLocalFp [4] tmpLocalFp（amsgrad 时 mgnLocalFp 复用该段）
        pipe->InitBuffer(tmpBuf, this->tileDataNum * sizeof(float) * 5);
    }
    if constexpr (IsSameType<TYPE_X, float>::value) {
        pipe->InitBuffer(tmpBuf, this->tileDataNum * sizeof(float));
    }
}

template <typename TYPE_X, uint64_t BUFFER_NUM>
__aicore__ inline void KernelApplyAdamW<TYPE_X, BUFFER_NUM>::DoRunOp(int32_t offset)
{
    LocalTensor<TYPE_X> varLocal = bindQue.template AllocTensor<TYPE_X>();
    LocalTensor<TYPE_X> mLocal = varLocal[this->processDataNum];
    LocalTensor<TYPE_X> vLocal = mLocal[this->processDataNum];
    LocalTensor<TYPE_X> gradLocal = vLocal[this->processDataNum];
    LocalTensor<TYPE_X> mgnLocal;
    LocalTensor<TYPE_X> tmpLocal = tmpBuf.Get<TYPE_X>();

    DataCopy(varLocal, varGm[offset], this->processDataNum);
    DataCopy(mLocal, mGm[offset], this->processDataNum);
    DataCopy(vLocal, vGm[offset], this->processDataNum);
    DataCopy(gradLocal, gradGm[offset], this->processDataNum);
    if (amsgrad) {
        mgnLocal = gradLocal[this->processDataNum];
        DataCopy(mgnLocal, mgnGm[offset], this->processDataNum);
    }
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);

    if (maximize) {
        Muls(gradLocal, gradLocal, (TYPE_X)-1.0f, this->processDataNum);
    }
    Muls(tmpLocal, gradLocal, (TYPE_X)this->negBeta1S1, this->processDataNum);
    Muls(mLocal, mLocal, (TYPE_X)this->beta1Value, this->processDataNum);
    Add(mLocal, mLocal, tmpLocal, this->processDataNum);

    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    DataCopy(mOutGm[offset], mLocal, this->processDataNum);

    Mul(gradLocal, gradLocal, gradLocal, this->processDataNum);
    Muls(gradLocal, gradLocal, (TYPE_X)this->negBeta2S1, this->processDataNum);
    Muls(vLocal, vLocal, (TYPE_X)this->beta2Value, this->processDataNum);
    Add(vLocal, vLocal, gradLocal, this->processDataNum);

    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    DataCopy(vOutGm[offset], vLocal, this->processDataNum);

    Muls(varLocal, varLocal, (TYPE_X)this->varTP, this->processDataNum);
    if (amsgrad) {
        Max(tmpLocal, mgnLocal, vLocal, this->processDataNum);
        Muls(tmpLocal, tmpLocal, (TYPE_X)this->negBeta2PowOutS1, this->processDataNum);
    } else {
        Muls(tmpLocal, vLocal, (TYPE_X)this->negBeta2PowOutS1, this->processDataNum);
    }
    Sqrt(tmpLocal, tmpLocal, this->processDataNum);
    Adds(tmpLocal, tmpLocal, (TYPE_X)this->epsilonValue, this->processDataNum);

    Div(tmpLocal, mLocal, tmpLocal, this->processDataNum);
    Muls(tmpLocal, tmpLocal, (TYPE_X)this->lrDBeta1PowOutS1, this->processDataNum);
    Add(varLocal, varLocal, tmpLocal, this->processDataNum);

    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    DataCopy(varOutGm[offset], varLocal, this->processDataNum);
    bindQue.template FreeTensor(varLocal);
}

template <typename TYPE_X, uint64_t BUFFER_NUM>
__aicore__ inline void KernelApplyAdamW<TYPE_X, BUFFER_NUM>::DoRunOpLowPrecision(int32_t offset)
{
    LocalTensor<TYPE_X> mLocal = bindQue.template AllocTensor<TYPE_X>();
    LocalTensor<TYPE_X> vLocal = mLocal[this->processDataNum];
    LocalTensor<TYPE_X> varLocal = vLocal[this->processDataNum];
    LocalTensor<TYPE_X> gradLocal = varLocal[this->processDataNum];
    LocalTensor<TYPE_X> mgnLocal = gradLocal[this->processDataNum];
    LocalTensor<float> varLocalFp = tmpBuf.Get<float>();
    LocalTensor<float> mLocalFp = varLocalFp[this->processDataNum];
    LocalTensor<float> gradLocalFp = mLocalFp[this->processDataNum];
    LocalTensor<float> vLocalFp = gradLocalFp[this->processDataNum];
    LocalTensor<float> tmpLocalFp = vLocalFp[this->processDataNum];
    // mgnLocalFp 与 tmpLocalFp 复用同一段空间：Max(tmpLocalFp, mgnLocalFp, vLocalFp) 为逐元素原地运算，
    // 且 mgnLocalFp 在 Max 之后不再被使用，无读写冲突（仅 amsgrad 时使用 mgnLocalFp）
    LocalTensor<float> mgnLocalFp = tmpLocalFp;

    DataCopy(varLocal, varGm[offset], this->processDataNum);
    DataCopy(mLocal, mGm[offset], this->processDataNum);
    DataCopy(vLocal, vGm[offset], this->processDataNum);
    DataCopy(gradLocal, gradGm[offset], this->processDataNum);
    if (amsgrad) {
        DataCopy(mgnLocal, mgnGm[offset], this->processDataNum);
    }
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    Cast(mLocalFp, mLocal, RoundMode::CAST_NONE, this->processDataNum);
    Cast(gradLocalFp, gradLocal, RoundMode::CAST_NONE, this->processDataNum);

    if (maximize) {
        Muls(gradLocalFp, gradLocalFp, (float)-1.0f, this->processDataNum);
    }
    Muls(varLocalFp, gradLocalFp, (float)this->negBeta1S1, this->processDataNum);
    Muls(mLocalFp, mLocalFp, (float)this->beta1Value, this->processDataNum);
    Add(mLocalFp, mLocalFp, varLocalFp, this->processDataNum);
    Cast(mLocal, mLocalFp, RoundMode::CAST_RINT, this->processDataNum);

    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    DataCopy(mOutGm[offset], mLocal, this->processDataNum);

    Cast(vLocalFp, vLocal, RoundMode::CAST_NONE, this->processDataNum);
    Mul(gradLocalFp, gradLocalFp, gradLocalFp, this->processDataNum);
    Muls(gradLocalFp, gradLocalFp, (float)this->negBeta2S1, this->processDataNum);
    Muls(vLocalFp, vLocalFp, (float)this->beta2Value, this->processDataNum);
    Add(vLocalFp, vLocalFp, gradLocalFp, this->processDataNum);
    Cast(vLocal, vLocalFp, RoundMode::CAST_RINT, this->processDataNum);

    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    DataCopy(vOutGm[offset], vLocal, this->processDataNum);

    Cast(varLocalFp, varLocal, RoundMode::CAST_NONE, this->processDataNum);
    Muls(varLocalFp, varLocalFp, (float)this->varTP, this->processDataNum);

    if (amsgrad) {
        Cast(mgnLocalFp, mgnLocal, RoundMode::CAST_NONE, this->processDataNum);
        Max(tmpLocalFp, mgnLocalFp, vLocalFp, this->processDataNum);
        Muls(tmpLocalFp, tmpLocalFp, (float)this->negBeta2PowOutS1, this->processDataNum);
    } else {
        Muls(tmpLocalFp, vLocalFp, (float)this->negBeta2PowOutS1, this->processDataNum);
    }
    Sqrt(tmpLocalFp, tmpLocalFp, this->processDataNum);
    Adds(tmpLocalFp, tmpLocalFp, (float)this->epsilonValue, this->processDataNum);

    Div(tmpLocalFp, mLocalFp, tmpLocalFp, this->processDataNum);
    Muls(tmpLocalFp, tmpLocalFp, (float)this->lrDBeta1PowOutS1, this->processDataNum);
    Add(varLocalFp, varLocalFp, tmpLocalFp, this->processDataNum);
    Cast(varLocal, varLocalFp, RoundMode::CAST_RINT, this->processDataNum);
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    DataCopy(varOutGm[offset], varLocal, this->processDataNum);
    bindQue.template FreeTensor(mLocal);
}

template <typename TYPE_X, uint64_t BUFFER_NUM>
__aicore__ inline void KernelApplyAdamW<TYPE_X, BUFFER_NUM>::Process()
{
    int32_t loopCount = this->tileNum - 1;
    this->processDataNum = this->tileDataNum;
    int32_t offset = 0;
    for (int32_t i = 0; i < loopCount; i++, offset += this->tileDataNum) {
        if constexpr (IsSameType<TYPE_X, bfloat16_t>::value) {
            DoRunOpLowPrecision(offset);
        }
        if constexpr (IsSameType<TYPE_X, float>::value) {
            DoRunOp(offset);
        }
        if constexpr (IsSameType<TYPE_X, half>::value) {
            DoRunOpLowPrecision(offset);
        }
    }
    this->processDataNum = this->tailDataNum;
    if constexpr (IsSameType<TYPE_X, bfloat16_t>::value) {
        DoRunOpLowPrecision(offset);
    }
    if constexpr (IsSameType<TYPE_X, float>::value) {
        DoRunOp(offset);
    }
    if constexpr (IsSameType<TYPE_X, half>::value) {
        DoRunOpLowPrecision(offset);
    }
}

} // namespace MyApplyAdamW
#endif // APPLY_ADAM_W_H
