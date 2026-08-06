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
 * \file sync_batch_norm_backward_elemt.h
 * \brief
 */
#ifndef SYNC_BATCH_NORM_BACKWARD_ELEMT_H_
#define SYNC_BATCH_NORM_BACKWARD_ELEMT_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "sync_batch_norm_backward_elemt_tilingdata.h"
#include "sync_batch_norm_backward_elemt_tiling_key.h"

namespace NsSyncBatchNormBackwardElemt {

using namespace AscendC;

template <typename T, typename T1>
class KernelSyncBatchNormBackwardElemt {
public:
    __aicore__ inline KernelSyncBatchNormBackwardElemt(){};

    __aicore__ inline void Init(GM_ADDR grad_output, GM_ADDR save_input, GM_ADDR mean, GM_ADDR invstd, GM_ADDR weight,
                                GM_ADDR mean_dy, GM_ADDR mean_dy_xmu, GM_ADDR grad_input,
                                const SyncBatchNormBackwardElemtTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t progress);
    __aicore__ inline void CopyOut(int32_t progress);
    __aicore__ inline void Compute(int32_t progress);
    __aicore__ inline void CalculateFp(AscendC::LocalTensor<float>& gradInput, AscendC::LocalTensor<float>& gradOut,
                                       AscendC::LocalTensor<float>& saveInput, AscendC::LocalTensor<float>& mean,
                                       AscendC::LocalTensor<float>& invstd, AscendC::LocalTensor<float>& weight,
                                       AscendC::LocalTensor<float>& meanDy, AscendC::LocalTensor<float>& meanDyXmu,
                                       uint32_t length);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueIN;
    TQue<QuePosition::VECIN, 1> inQueueIN1;
    TQue<QuePosition::VECOUT, 1> outQueueOUT;

    TBuf<QuePosition::VECCALC> t_tmpBuf;
    TBuf<QuePosition::VECCALC> u_tmpBuf;

    GlobalTensor<T> grad_outputGm;
    GlobalTensor<T> save_inputGm;
    GlobalTensor<T1> meanGm;
    GlobalTensor<T1> invstdGm;
    GlobalTensor<T1> weightGm;
    GlobalTensor<T1> mean_dyGm;
    GlobalTensor<T1> mean_dy_xmuGm;
    GlobalTensor<T> grad_inputGm;

    LocalTensor<float> tTmp;
    LocalTensor<float> uTmp;

    uint64_t coreDataNum = 0;
    uint64_t tileNum = 0;
    uint64_t tileDataNum = 0;
    uint64_t tailDataNum = 0;
    uint64_t processDataNum = 0;
    int32_t bufferNum = 2;
};

template <typename T, typename T1>
__aicore__ inline void KernelSyncBatchNormBackwardElemt<T, T1>::Init(
    GM_ADDR grad_output, GM_ADDR save_input, GM_ADDR mean, GM_ADDR invstd, GM_ADDR weight, GM_ADDR mean_dy,
    GM_ADDR mean_dy_xmu, GM_ADDR grad_input, const SyncBatchNormBackwardElemtTilingData* tilingData)
{
    uint64_t coreId = AscendC::GetBlockIdx();
    uint64_t globalBufferIndex = tilingData->bigCoreDataNum * AscendC::GetBlockIdx();
    this->tileDataNum = tilingData->tileDataNum;
    if (coreId < tilingData->tailBlockNum) {
        this->coreDataNum = tilingData->bigCoreDataNum;
        this->tileNum = tilingData->finalBigTileNum;
        this->tailDataNum = tilingData->bigTailDataNum;
    } else {
        this->coreDataNum = tilingData->smallCoreDataNum;
        this->tileNum = tilingData->finalSmallTileNum;
        this->tailDataNum = tilingData->smallTailDataNum;
        globalBufferIndex -= (tilingData->bigCoreDataNum - tilingData->smallCoreDataNum) *
                             (AscendC::GetBlockIdx() - tilingData->tailBlockNum);
    }

    this->bufferNum = 1;
    if (static_cast<int32_t>(tilingData->usedDb) == 1) {
        this->bufferNum = 2;
    }
    grad_outputGm.SetGlobalBuffer((__gm__ T*)grad_output + globalBufferIndex, this->coreDataNum);
    save_inputGm.SetGlobalBuffer((__gm__ T*)save_input + globalBufferIndex, this->coreDataNum);

    meanGm.SetGlobalBuffer((__gm__ T1*)mean + globalBufferIndex, this->coreDataNum);
    invstdGm.SetGlobalBuffer((__gm__ T1*)invstd + globalBufferIndex, this->coreDataNum);
    weightGm.SetGlobalBuffer((__gm__ T1*)weight + globalBufferIndex, this->coreDataNum);
    mean_dyGm.SetGlobalBuffer((__gm__ T1*)mean_dy + globalBufferIndex, this->coreDataNum);
    mean_dy_xmuGm.SetGlobalBuffer((__gm__ T1*)mean_dy_xmu + globalBufferIndex, this->coreDataNum);

    grad_inputGm.SetGlobalBuffer((__gm__ T*)grad_input + globalBufferIndex, this->coreDataNum);

    pipe.InitBuffer(inQueueIN, this->bufferNum, 2 * this->tileDataNum * sizeof(T));
    pipe.InitBuffer(inQueueIN1, this->bufferNum, 5 * this->tileDataNum * sizeof(T1));
    pipe.InitBuffer(outQueueOUT, this->bufferNum, this->tileDataNum * sizeof(T));

    if constexpr ((AscendC::Std::is_same<T, bfloat16_t>::value) && (AscendC::Std::is_same<T1, bfloat16_t>::value)) {
        pipe.InitBuffer(t_tmpBuf, 3 * this->tileDataNum * sizeof(float));
        pipe.InitBuffer(u_tmpBuf, 5 * this->tileDataNum * sizeof(float));
    } else if constexpr ((AscendC::Std::is_same<T, half>::value) && (AscendC::Std::is_same<T1, float>::value)) {
        pipe.InitBuffer(t_tmpBuf, 3 * this->tileDataNum * sizeof(float));
    }
}

template <typename T, typename T1>
__aicore__ inline void KernelSyncBatchNormBackwardElemt<T, T1>::CopyIn(int32_t progress)
{
    AscendC::LocalTensor<T> inLocal = inQueueIN.AllocTensor<T>();
    AscendC::LocalTensor<T1> in1Local = inQueueIN1.AllocTensor<T1>();

    AscendC::DataCopy(inLocal, grad_outputGm[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(inLocal[this->tileDataNum], save_inputGm[progress * this->tileDataNum], this->processDataNum);

    AscendC::DataCopy(in1Local, meanGm[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(in1Local[this->tileDataNum], invstdGm[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(in1Local[2 * this->tileDataNum], weightGm[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(in1Local[3 * this->tileDataNum], mean_dyGm[progress * this->tileDataNum], this->processDataNum);
    AscendC::DataCopy(in1Local[4 * this->tileDataNum], mean_dy_xmuGm[progress * this->tileDataNum],
                      this->processDataNum);

    inQueueIN.EnQue(inLocal);
    inQueueIN1.EnQue(in1Local);
}

template <typename T, typename T1>
__aicore__ inline void KernelSyncBatchNormBackwardElemt<T, T1>::CopyOut(int32_t progress)
{
    AscendC::LocalTensor<T> outLocal = outQueueOUT.DeQue<T>();

    AscendC::DataCopy(grad_inputGm[progress * this->tileDataNum], outLocal, this->processDataNum);

    outQueueOUT.FreeTensor(outLocal);
}

template <typename T, typename T1>
__aicore__ inline void KernelSyncBatchNormBackwardElemt<T, T1>::Compute(int32_t progress)
{
    LocalTensor<T> inLocal = inQueueIN.DeQue<T>();
    LocalTensor<T1> in1Local = inQueueIN1.DeQue<T1>();

    LocalTensor<T> outLocal = outQueueOUT.AllocTensor<T>();

    if constexpr ((AscendC::Std::is_same<T, half>::value && AscendC::Std::is_same<T1, half>::value) ||
                  (AscendC::Std::is_same<T, float>::value && AscendC::Std::is_same<T1, float>::value)) { // 直接计算
        LocalTensor<T> gradOut = inLocal;
        LocalTensor<T> saveInput = inLocal[this->tileDataNum];

        LocalTensor<T> mean = in1Local;
        LocalTensor<T> invstd = in1Local[this->tileDataNum];
        LocalTensor<T> weight = in1Local[2 * this->tileDataNum];
        LocalTensor<T> meanDy = in1Local[3 * this->tileDataNum];
        LocalTensor<T> meanDyXmu = in1Local[4 * this->tileDataNum];

        // gradInput = ({gradOut} - {meanDy}) - ((input - mean) * (invstd^{2} *   {meanDyXmu})) * invstd * weight
        AscendC::Sub(outLocal, gradOut, meanDy, this->processDataNum);
        AscendC::Sub(saveInput, saveInput, mean, this->processDataNum);
        AscendC::Mul(mean, invstd, invstd, this->processDataNum);
        AscendC::Mul(mean, mean, meanDyXmu, this->processDataNum);
        AscendC::Mul(saveInput, saveInput, mean, this->processDataNum);
        AscendC::Sub(outLocal, outLocal, saveInput, this->processDataNum);
        AscendC::Mul(outLocal, outLocal, invstd, this->processDataNum);
        AscendC::Mul(outLocal, outLocal, weight, this->processDataNum);
    } else if constexpr ((AscendC::Std::is_same<T, bfloat16_t>::value) &&
                         (AscendC::Std::is_same<T1, bfloat16_t>::value)) {
        this->tTmp = t_tmpBuf.Get<float>();
        AscendC::Cast(this->tTmp, inLocal, AscendC::RoundMode::CAST_NONE, 2 * this->tileDataNum);
        this->uTmp = u_tmpBuf.Get<float>();
        AscendC::Cast(this->uTmp, in1Local, AscendC::RoundMode::CAST_NONE, 5 * this->tileDataNum);
        auto gradInputRef = this->tTmp[2 * this->tileDataNum];
        auto inputRef1 = this->tTmp;
        auto inputRef2 = this->tTmp[this->tileDataNum];

        auto param1 = this->uTmp;
        auto param2 = this->uTmp[this->tileDataNum];
        auto param3 = this->uTmp[2 * this->tileDataNum];
        auto param4 = this->uTmp[3 * this->tileDataNum];
        auto param5 = this->uTmp[4 * this->tileDataNum];

        CalculateFp(gradInputRef, inputRef1, inputRef2, param1, param2, param3, param4, param5, this->processDataNum);
        AscendC::Cast(outLocal, this->tTmp[2 * this->tileDataNum], AscendC::RoundMode::CAST_RINT, this->processDataNum);

    } else if constexpr ((AscendC::Std::is_same<T, half>::value) && (AscendC::Std::is_same<T1, float>::value)) {
        this->tTmp = t_tmpBuf.Get<float>();
        AscendC::Cast(this->tTmp, inLocal, AscendC::RoundMode::CAST_NONE, 2 * this->tileDataNum);

        auto gradInputRef = this->tTmp[2 * this->tileDataNum];
        auto inputRef1 = this->tTmp;
        auto inputRef2 = this->tTmp[this->tileDataNum];

        auto param1 = in1Local;
        auto param2 = in1Local[this->tileDataNum];
        auto param3 = in1Local[2 * this->tileDataNum];
        auto param4 = in1Local[3 * this->tileDataNum];
        auto param5 = in1Local[4 * this->tileDataNum];
        CalculateFp(gradInputRef, inputRef1, inputRef2, param1, param2, param3, param4, param5, this->processDataNum);
        AscendC::Cast(outLocal, this->tTmp[2 * this->tileDataNum], AscendC::RoundMode::CAST_RINT, this->processDataNum);
    }

    outQueueOUT.EnQue(outLocal);
    inQueueIN.FreeTensor(inLocal);
    inQueueIN1.FreeTensor(in1Local);
}

template <typename T, typename T1>
__aicore__ inline void KernelSyncBatchNormBackwardElemt<T, T1>::CalculateFp(
    AscendC::LocalTensor<float>& gradInput, AscendC::LocalTensor<float>& gradOut,
    AscendC::LocalTensor<float>& saveInput, AscendC::LocalTensor<float>& mean, AscendC::LocalTensor<float>& invstd,
    AscendC::LocalTensor<float>& weight, AscendC::LocalTensor<float>& meanDy, AscendC::LocalTensor<float>& meanDyXmu,
    uint32_t length)
{
    // gradInput = ({gradOut} - {meanDy}) - ((input - mean) * (invstd^{2} *   {meanDyXmu})) * invstd * weight
    AscendC::Sub(gradInput, gradOut, meanDy, length);
    AscendC::Sub(saveInput, saveInput, mean, length);
    AscendC::Mul(mean, invstd, invstd, length);
    AscendC::Mul(mean, mean, meanDyXmu, length);
    AscendC::Mul(saveInput, saveInput, mean, length);
    AscendC::Sub(gradInput, gradInput, saveInput, length);
    AscendC::Mul(gradInput, gradInput, invstd, length);
    AscendC::Mul(gradInput, gradInput, weight, length);
}

template <typename T, typename T1>
__aicore__ inline void KernelSyncBatchNormBackwardElemt<T, T1>::Process()
{
    int32_t loopCount = this->tileNum;
    this->processDataNum = this->tileDataNum;
    for (int32_t i = 0; i < loopCount - 1; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
    this->processDataNum = this->tailDataNum;
    CopyIn(loopCount - 1);
    Compute(loopCount - 1);
    CopyOut(loopCount - 1);
}

} // namespace NsSyncBatchNormBackwardElemt
#endif // SYNC_BATCH_NORM_BACKWARD_ELEMT_H
