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
 * \file apply_fused_ema_adam_f_bf16.h
 * \brief
 */

#ifndef APPLY_FUSED_EMA_ADAM_F_BF_16_H
#define APPLY_FUSED_EMA_ADAM_F_BF_16_H

#include "apply_fused_ema_adam_base.h"

namespace FusedEmaAdam {
using namespace AscendC;

template <typename T>
class FusedEmaAdamF16 : public FusedEmaAdamBase<T> {
public:
    __aicore__ inline FusedEmaAdamF16(){};
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR var, GM_ADDR m, GM_ADDR v, GM_ADDR s, GM_ADDR step,
                                GM_ADDR var_ref, GM_ADDR m_ref, GM_ADDR v_ref, GM_ADDR s_ref,
                                const ApplyFusedEmaAdamTilingData& tiling, TPipe* pipe);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void Compute(const uint64_t index, const uint64_t dataCount);

    TQue<QuePosition::VECIN, BUFFER_NUM> inQueF16;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueF16;

    TBuf<QuePosition::VECCALC> inCastBuf, outCastBuf, powTBuf1, powTBuf2;

    GlobalTensor<T> gmGrad, gmVar, gmM, gmV, gmS;
    GlobalTensor<int64_t> gmStep;
    GlobalTensor<T> gmVarRef, gmMRef, gmVRef, gmSRef;

    float step_ = 0;
    int32_t INPUT_NUM_F16 = 5;
    int32_t OUTPUT_NUM_F16 = 4;
    uint32_t blockIdxF16;
    uint64_t blockOffsetF16;
    int64_t mOffsetF16;
    int64_t vOffsetF16;
    int64_t varOffsetF16;
    int64_t sOffsetF16;
    int64_t gradOffsetF16;
};

template <typename T>
__aicore__ inline void FusedEmaAdamF16<T>::Init(GM_ADDR grad, GM_ADDR var, GM_ADDR m, GM_ADDR v, GM_ADDR s,
                                                GM_ADDR step, GM_ADDR var_ref, GM_ADDR m_ref, GM_ADDR v_ref,
                                                GM_ADDR s_ref, const ApplyFusedEmaAdamTilingData& tiling, TPipe* pipe)
{
    this->InitData(tiling);
    blockIdxF16 = GetBlockIdx();

    if (blockIdxF16 < this->frontCoreNum) {
        blockOffsetF16 = this->coreCalcNum * blockIdxF16;
    } else if (this->coreCalcNum - 1 != 0) {
        blockOffsetF16 = this->coreCalcNum * this->frontCoreNum +
                         (blockIdxF16 - this->frontCoreNum) * (this->coreCalcNum - 1);
    }

    gmGrad.SetGlobalBuffer((__gm__ T*)grad + blockOffsetF16);
    gmVar.SetGlobalBuffer((__gm__ T*)var + blockOffsetF16);
    gmM.SetGlobalBuffer((__gm__ T*)m + blockOffsetF16);
    gmV.SetGlobalBuffer((__gm__ T*)v + blockOffsetF16);
    gmS.SetGlobalBuffer((__gm__ T*)s + blockOffsetF16);

    gmVarRef.SetGlobalBuffer((__gm__ T*)var_ref + blockOffsetF16);
    gmMRef.SetGlobalBuffer((__gm__ T*)m_ref + blockOffsetF16);
    gmVRef.SetGlobalBuffer((__gm__ T*)v_ref + blockOffsetF16);
    gmSRef.SetGlobalBuffer((__gm__ T*)s_ref + blockOffsetF16);

    if (this->mode == 1) {
        INPUT_NUM_F16 -= 1;
    }
    pipe->InitBuffer(inQueF16, BUFFER_NUM, this->coreCalcMax * sizeof(T) * INPUT_NUM_F16);
    pipe->InitBuffer(outQueF16, BUFFER_NUM, this->coreCalcMax * sizeof(T) * OUTPUT_NUM_F16);

    pipe->InitBuffer(inCastBuf, this->coreCalcMax * sizeof(float) * INPUT_NUM_F16);
    pipe->InitBuffer(outCastBuf, this->coreCalcMax * sizeof(float) * OUTPUT_NUM_F16);

    varOffsetF16 = this->coreCalcMax * INDEX_VAR;
    mOffsetF16 = this->coreCalcMax * INDEX_M;
    vOffsetF16 = this->coreCalcMax * INDEX_V;
    sOffsetF16 = this->coreCalcMax * INDEX_S;
    gradOffsetF16 = this->mode == 0 ? this->coreCalcMax * INDEX_GRAD : varOffsetF16;

    pipe->InitBuffer(powTBuf1, BYTE_ONE_BLOCK);
    pipe->InitBuffer(powTBuf2, BYTE_ONE_BLOCK);

    gmStep.SetGlobalBuffer((__gm__ int64_t*)step, 1);
    step_ = static_cast<float>(gmStep.GetValue(0));
    if (this->biasCorrection == 1) {
        this->beta1Correction = 1.0f - this->ScalarPow(powTBuf1, powTBuf2, inQueF16, this->beta1, step_);
        this->beta2Correction = 1.0f - this->ScalarPow(powTBuf1, powTBuf2, inQueF16, this->beta2, step_);
    }
}

template <typename T>
__aicore__ inline void FusedEmaAdamF16<T>::Compute(const uint64_t index, const uint64_t dataCount)
{
    uint64_t offset = index * this->coreCalcMax;
    DataCopyParams copyParams = {1, static_cast<uint16_t>(dataCount * sizeof(T)), 0, 0};
    DataCopyPadParams padParams = {false, 0, 0, 0};

    LocalTensor<T> inLocal = inQueF16.AllocTensor<T>();
    LocalTensor<T> outLocal = outQueF16.AllocTensor<T>();
    LocalTensor<float> inLocalC = inCastBuf.Get<float>();
    LocalTensor<float> outLocalC = outCastBuf.Get<float>();

    // grad = grad [+ weight_decay*var if mode == 0]
    if (this->mode == 0) {
        DataCopyPad(inLocal[varOffsetF16], gmVar[offset], copyParams, padParams);
        this->PipeM2V();
        Cast(inLocalC[varOffsetF16], inLocal[varOffsetF16], RoundMode::CAST_NONE, dataCount);
        Muls(outLocalC[mOffsetF16], inLocalC[varOffsetF16], this->weightDecay, dataCount);
    }
    DataCopyPad(inLocal[gradOffsetF16], gmGrad[offset], copyParams, padParams);
    this->PipeM2V();
    Cast(inLocalC[gradOffsetF16], inLocal[gradOffsetF16], RoundMode::CAST_NONE, dataCount);
    if (this->mode == 0) {
        Add(inLocalC[gradOffsetF16], outLocalC[mOffsetF16], inLocalC[gradOffsetF16], dataCount);
    }

    // m = beta1*m + (1-beta1)*grad, next_m = m/beta1_correction
    DataCopyPad(inLocal[mOffsetF16], gmM[offset], copyParams, padParams);
    this->PipeM2V();
    Cast(inLocalC[mOffsetF16], inLocal[mOffsetF16], RoundMode::CAST_NONE, dataCount);
    Muls(inLocalC[mOffsetF16], inLocalC[mOffsetF16], this->beta1, dataCount);
    Muls(outLocalC[mOffsetF16], inLocalC[gradOffsetF16], 1 - this->beta1, dataCount);
    Add(outLocalC[mOffsetF16], outLocalC[mOffsetF16], inLocalC[mOffsetF16], dataCount);
    Cast(outLocal[mOffsetF16], outLocalC[mOffsetF16], RoundMode::CAST_RINT, dataCount);
    this->PipeVM3();
    DataCopyPad(gmMRef[offset], outLocal[mOffsetF16], copyParams);
    Muls(inLocalC[mOffsetF16], outLocalC[mOffsetF16], 1 / this->beta1Correction, dataCount);

    // v = beta2*v + (1-beta2)*grad*grad, next_v = v/beta2_correction
    DataCopyPad(inLocal[vOffsetF16], gmV[offset], copyParams, padParams);
    this->PipeM2V();
    Cast(inLocalC[vOffsetF16], inLocal[vOffsetF16], RoundMode::CAST_NONE, dataCount);
    Muls(inLocalC[vOffsetF16], inLocalC[vOffsetF16], this->beta2, dataCount);
    Mul(outLocalC[vOffsetF16], inLocalC[gradOffsetF16], inLocalC[gradOffsetF16], dataCount);
    if (this->mode == 1) {
        this->PipeVM2();
    }
    Muls(outLocalC[vOffsetF16], outLocalC[vOffsetF16], 1 - this->beta2, dataCount);
    Add(outLocalC[vOffsetF16], outLocalC[vOffsetF16], inLocalC[vOffsetF16], dataCount);
    Cast(outLocal[vOffsetF16], outLocalC[vOffsetF16], RoundMode::CAST_RINT, dataCount);
    this->PipeVM3();
    DataCopyPad(gmVRef[offset], outLocal[vOffsetF16], copyParams);
    Muls(inLocalC[vOffsetF16], outLocalC[vOffsetF16], 1 / this->beta2Correction, dataCount);

    // denom = sqrt(next_v) + eps, update = next_m/denom [+ weight_decay*var if mode == 1]
    Sqrt(inLocalC[vOffsetF16], inLocalC[vOffsetF16], dataCount);
    Adds(inLocalC[vOffsetF16], inLocalC[vOffsetF16], this->eps, dataCount);
    Div(inLocalC[mOffsetF16], inLocalC[mOffsetF16], inLocalC[vOffsetF16], dataCount);
    if (this->mode == 1) {
        DataCopyPad(inLocal[varOffsetF16], gmVar[offset], copyParams, padParams);
        this->PipeM2V();
        Cast(inLocalC[varOffsetF16], inLocal[varOffsetF16], RoundMode::CAST_NONE, dataCount);
        Muls(inLocalC[vOffsetF16], inLocalC[varOffsetF16], this->weightDecay, dataCount);
        Add(inLocalC[mOffsetF16], inLocalC[mOffsetF16], inLocalC[vOffsetF16], dataCount);
    }

    // var = var - lr*update
    Muls(inLocalC[mOffsetF16], inLocalC[mOffsetF16], this->lr, dataCount);
    Sub(outLocalC[varOffsetF16], inLocalC[varOffsetF16], inLocalC[mOffsetF16], dataCount);
    Cast(outLocal[varOffsetF16], outLocalC[varOffsetF16], RoundMode::CAST_RINT, dataCount);
    this->PipeVM3();
    DataCopyPad(gmVarRef[offset], outLocal[varOffsetF16], copyParams);
    Muls(inLocalC[varOffsetF16], outLocalC[varOffsetF16], 1 - this->emaDecay, dataCount);

    // s = ema_decay*s + (1-ema_decay)*var
    DataCopyPad(inLocal[sOffsetF16], gmS[offset], copyParams, padParams);
    this->PipeM2V();
    Cast(inLocalC[sOffsetF16], inLocal[sOffsetF16], RoundMode::CAST_NONE, dataCount);
    Muls(inLocalC[sOffsetF16], inLocalC[sOffsetF16], this->emaDecay, dataCount);
    Add(outLocalC[sOffsetF16], inLocalC[sOffsetF16], inLocalC[varOffsetF16], dataCount);
    Cast(outLocal[sOffsetF16], outLocalC[sOffsetF16], RoundMode::CAST_RINT, dataCount);
    this->PipeVM3();
    DataCopyPad(gmSRef[offset], outLocal[sOffsetF16], copyParams);

    inQueF16.FreeTensor(inLocal);
    outQueF16.FreeTensor(outLocal);
}

template <typename T>
__aicore__ inline void FusedEmaAdamF16<T>::Process()
{
    for (uint64_t n = 0; n < this->loopNum - 1; n++) {
        Compute(n, this->coreCalcMax);
    }
    if (blockIdxF16 < this->frontCoreNum) {
        Compute(this->loopNum - 1, this->frontCalcExtra);
    } else if (this->tailCalcExtra != 0) {
        Compute(this->loopNum - 1, this->tailCalcExtra);
    }
}

} // namespace FusedEmaAdam

#endif // APPLY_FUSED_EMA_ADAM_F_BF_16_H
