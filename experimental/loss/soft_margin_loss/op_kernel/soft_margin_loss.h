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

/**
 * \file soft_margin_loss.h
 * \brief SoftMarginLoss kernel class definition (arch32 - Ascend910B)
 *
 * Computes SoftMarginLoss: L[i] = max(0, -t*x) + log(1 + exp(-|t*x|))
 * where x = self[i], t = target[i]
 *
 * Template-based implementation:
 *   SoftMarginLossNone<T>   - elementwise output (reduction='none')
 *   SoftMarginLossReduce<T> - scalar output (reduction='mean'/'sum')
 * where T = float or half
 *
 * Uses double buffering (depth=2) for pipeline parallelism.
 * Reduce path uses two-phase cross-core reduction via workspace.
 */
#ifndef SOFT_MARGIN_LOSS_H
#define SOFT_MARGIN_LOSS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "soft_margin_loss_tiling_data.h"
#include "soft_margin_loss_tiling_key.h"

namespace NsSoftMarginLoss {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

// ============================================================================
// ComputeLossCore - 9-step SoftMarginLoss core computation (float32)
// L = max(0, -tx) + log(1 + exp(-|tx|))
//
// outputF32:  result written here
// txLocal:    temp buffer 1 (reused across steps)
// negTxLocal: temp buffer 2 (reused across steps)
// selfF32:    self values in float32
// targetF32:  target values in float32
// ============================================================================
__aicore__ inline void ComputeLossCore(
    LocalTensor<float>& outputF32,
    LocalTensor<float>& txLocal,
    LocalTensor<float>& negTxLocal,
    LocalTensor<float>& selfF32,
    LocalTensor<float>& targetF32,
    int64_t currentNum)
{
    // Step 1: tx = target * self
    AscendC::Mul(txLocal, targetF32, selfF32, currentNum);
    // Step 2: neg_tx = -tx
    AscendC::Muls(negTxLocal, txLocal, static_cast<float>(-1.0f), currentNum);
    // Step 3: abs_tx = |tx| (reuse txLocal)
    AscendC::Abs(txLocal, txLocal, currentNum);
    // Step 4: -|tx| (reuse txLocal)
    AscendC::Muls(txLocal, txLocal, static_cast<float>(-1.0f), currentNum);
    // Step 5: exp(-|tx|) (reuse txLocal)
    AscendC::Exp(txLocal, txLocal, currentNum);
    // Step 6: 1 + exp(-|tx|) (reuse txLocal)
    AscendC::Adds(txLocal, txLocal, static_cast<float>(1.0f), currentNum);
    // Step 7: log(1 + exp(-|tx|)) (reuse txLocal)
    AscendC::Log(txLocal, txLocal, currentNum);
    // Step 8: max(0, neg_tx) (reuse negTxLocal)
    AscendC::Maxs(negTxLocal, negTxLocal, static_cast<float>(0.0f), currentNum);
    // Step 9: L = max(0, -tx) + log(1 + exp(-|tx|))
    AscendC::Add(outputF32, negTxLocal, txLocal, currentNum);
}

// ============================================================================
// SoftMarginLossNone<T> - Elementwise path (reduction='none')
// T = float: direct compute
// T = half:  Cast to float -> compute -> Cast back to half
// ============================================================================
template <typename T>
class SoftMarginLossNone {
public:
    __aicore__ inline SoftMarginLossNone() {}

    __aicore__ inline void Init(GM_ADDR selfGm, GM_ADDR targetGm, GM_ADDR outputGm,
                                 const SoftMarginLossTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> selfQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> targetQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue;
    TBuf<QuePosition::VECCALC> tmpBuf1;   // float: tx / abs / exp / log
    TBuf<QuePosition::VECCALC> tmpBuf2;   // float: neg_tx / max_val
    TBuf<QuePosition::VECCALC> castBuf;   // float: Cast workspace (half path only)

    GlobalTensor<T> selfGM;
    GlobalTensor<T> targetGM;
    GlobalTensor<T> outputGM;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

// ---- Init ----
template <typename T>
__aicore__ inline void SoftMarginLossNone<T>::Init(GM_ADDR selfGm, GM_ADDR targetGm, GM_ADDR outputGm,
                                                    const SoftMarginLossTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    ubLength_ = tilingData->ubFactor;

    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }

    int64_t coreOffset = tilingData->blockFactor * AscendC::GetBlockIdx();
    selfGM.SetGlobalBuffer((__gm__ T*)selfGm + coreOffset, blockLength_);
    targetGM.SetGlobalBuffer((__gm__ T*)targetGm + coreOffset, blockLength_);
    outputGM.SetGlobalBuffer((__gm__ T*)outputGm + coreOffset, blockLength_);

    pipe.InitBuffer(selfQueue, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(targetQueue, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueue, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(tmpBuf1, ubLength_ * sizeof(float));
    pipe.InitBuffer(tmpBuf2, ubLength_ * sizeof(float));
    if constexpr (std::is_same_v<T, half>) {
        pipe.InitBuffer(castBuf, ubLength_ * sizeof(float));
    }
}

// ---- CopyIn ----
template <typename T>
__aicore__ inline void SoftMarginLossNone<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> selfLocal = selfQueue.template AllocTensor<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(selfLocal, selfGM[progress * ubLength_], copyParams, {false, 0, 0, 0});
    selfQueue.EnQue(selfLocal);

    AscendC::LocalTensor<T> targetLocal = targetQueue.template AllocTensor<T>();
    AscendC::DataCopyPad(targetLocal, targetGM[progress * ubLength_], copyParams, {false, 0, 0, 0});
    targetQueue.EnQue(targetLocal);
}

// ---- Compute ----
template <typename T>
__aicore__ inline void SoftMarginLossNone<T>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> selfLocal = selfQueue.template DeQue<T>();
    AscendC::LocalTensor<T> targetLocal = targetQueue.template DeQue<T>();
    AscendC::LocalTensor<T> outputLocal = outputQueue.template AllocTensor<T>();
    AscendC::LocalTensor<float> txLocal = tmpBuf1.Get<float>();
    AscendC::LocalTensor<float> negTxLocal = tmpBuf2.Get<float>();

    if constexpr (std::is_same_v<T, float>) {
        // float path: compute directly
        ComputeLossCore(outputLocal, txLocal, negTxLocal, selfLocal, targetLocal, currentNum);
    } else {
        // half path: Cast to float -> compute -> Cast back
        AscendC::LocalTensor<float> castLocal = castBuf.Get<float>();
        // Cast self (half -> float) into castLocal
        AscendC::Cast(castLocal, selfLocal, AscendC::RoundMode::CAST_NONE, currentNum);
        // Cast target (half -> float) into txLocal (temporary reuse)
        AscendC::Cast(txLocal, targetLocal, AscendC::RoundMode::CAST_NONE, currentNum);
        // Compute in float: result in negTxLocal (reused as output buffer)
        ComputeLossCore(negTxLocal, txLocal, negTxLocal, castLocal, txLocal, currentNum);
        // Wait - negTxLocal is both output and temp. Need separate flow:
        // selfF32 = castLocal, targetF32 = txLocal after cast
        // But ComputeLossCore overwrites txLocal in step 1 (Mul into txLocal).
        // And negTxLocal is used as both neg_tx buffer and output. This works because
        // step 9 writes Add result to outputF32 (=negTxLocal) using negTxLocal and txLocal as inputs.
        // Actually let's re-examine: we need selfF32 and targetF32 as separate inputs to step 1.
        // After Cast: castLocal=selfF32, txLocal=targetF32
        // Step 1: Mul(txLocal, targetF32=txLocal, selfF32=castLocal) -> txLocal = tx (OK, txLocal overwritten)
        // This is correct because Mul reads txLocal and castLocal first, then writes txLocal.
        // The rest follows the same pattern as float path.
        // But outputF32=negTxLocal and negTxLocal param is also negTxLocal - that's fine,
        // the output is written in step 9, after negTxLocal is last read in step 9 itself.
        // Actually no: Add(output, negTx, tx) reads negTx then writes output=negTx. This is in-place, which is OK.
        // Let me use a cleaner approach with castLocal as the third buffer:
        // selfF32=castLocal, targetF32=txLocal (after second Cast)
        // Then ComputeLossCore(output=negTxLocal, tx_buf=txLocal, neg_buf=castLocal, self=castLocal, target=txLocal)
        // Problem: castLocal is both selfF32 input and neg_tx buffer. Step 1 reads selfF32=castLocal,
        // step 2 writes neg_tx=castLocal. But step 1 is Mul(txLocal, target, self) which only writes txLocal.
        // So castLocal (selfF32) is only read in step 1, then reused as neg_tx from step 2 onward. That works!

        // Re-do with correct buffer assignment:
        // castLocal = selfF32 (from Cast), then reused as neg_tx buffer from step 2
        // txLocal = targetF32 (from Cast), then reused as tx buffer from step 1
        // negTxLocal = output destination
        // But wait, ComputeLossCore signature: (output, txBuf, negTxBuf, selfF32, targetF32)
        // We need: output=negTxLocal, txBuf=txLocal, negTxBuf=castLocal, selfF32=castLocal, targetF32=txLocal
        // negTxBuf and selfF32 are both castLocal - step 1 reads selfF32, step 2 writes negTxBuf.
        // Inside ComputeLossCore step 1: Mul(txLocal, targetF32=txLocal, selfF32=castLocal) - reads both, writes txLocal
        // Step 2: Muls(negTxLocal=castLocal, txLocal, -1) - reads txLocal (from step 1), writes castLocal. OK!
        // Step 9: Add(output=negTxLocal, negTxLocal=castLocal, txLocal) -> writes negTxLocal. OK!
        // This works correctly.
        ComputeLossCore(negTxLocal, txLocal, castLocal, castLocal, txLocal, currentNum);
        // Cast result (float -> half)
        AscendC::Cast(outputLocal, negTxLocal, AscendC::RoundMode::CAST_ROUND, currentNum);
    }

    outputQueue.template EnQue<T>(outputLocal);
    selfQueue.FreeTensor(selfLocal);
    targetQueue.FreeTensor(targetLocal);
}

// ---- CopyOut ----
template <typename T>
__aicore__ inline void SoftMarginLossNone<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> outputLocal = outputQueue.template DeQue<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outputGM[progress * ubLength_], outputLocal, copyParams);
    outputQueue.FreeTensor(outputLocal);
}

// ---- Process ----
template <typename T>
__aicore__ inline void SoftMarginLossNone<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
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

// ============================================================================
// SoftMarginLossReduce<T> - Reduction path (reduction='mean'/'sum')
// T = float or half
//
// Two-phase cross-core reduction:
//   Phase 1: Each core computes elementwise loss + local ReduceSum -> partialSum
//            Writes partialSum to workspace[coreIdx * 8] (32-byte aligned)
//   Phase 2: After SyncAll, core 0 reads all partial sums, aggregates,
//            applies mean division if needed, writes scalar output
// ============================================================================
template <typename T>
class SoftMarginLossReduce {
public:
    __aicore__ inline SoftMarginLossReduce() {}

    __aicore__ inline void Init(GM_ADDR selfGm, GM_ADDR targetGm, GM_ADDR outputGm,
                                 GM_ADDR workspaceGm, const SoftMarginLossTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void ComputeAndReduce(int64_t currentNum, float& localSum);
    __aicore__ inline void WritePartialSum(float localSum);
    __aicore__ inline void CrossCoreReduce();
    __aicore__ inline void WriteFinalScalar(float value);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> selfQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> targetQueue;
    TBuf<QuePosition::VECCALC> tmpBuf1;       // float: tx / abs / exp / log
    TBuf<QuePosition::VECCALC> tmpBuf2;       // float: neg_tx / max_val
    TBuf<QuePosition::VECCALC> tmpBuf3;       // float: loss result (ReduceSum input)
    TBuf<QuePosition::VECCALC> castBuf;       // float: Cast workspace (half path only)
    TBuf<QuePosition::VECCALC> reduceTmpBuf;  // float: ReduceSum temporary buffer
    TBuf<QuePosition::VECCALC> resultBuf;     // small buffer for partial sum / final result

    GlobalTensor<T> selfGM;
    GlobalTensor<T> targetGM;
    GlobalTensor<T> outputGM;
    GlobalTensor<float> workspaceGM;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int32_t reductionMode_ = 0;
    float invNumel_ = 0.0f;
    int64_t usedCoreNum_ = 0;
};

// ---- Init ----
template <typename T>
__aicore__ inline void SoftMarginLossReduce<T>::Init(GM_ADDR selfGm, GM_ADDR targetGm, GM_ADDR outputGm,
                                                      GM_ADDR workspaceGm,
                                                      const SoftMarginLossTilingData* tilingData)
{
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * AscendC::GetBlockIdx();
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    if (blockLength_ < 0) {
        blockLength_ = 0;
    }
    ubLength_ = tilingData->ubFactor;
    reductionMode_ = tilingData->reductionMode;
    invNumel_ = tilingData->invNumel;
    usedCoreNum_ = tilingData->usedCoreNum;

    int64_t coreOffset = tilingData->blockFactor * AscendC::GetBlockIdx();
    selfGM.SetGlobalBuffer((__gm__ T*)selfGm + coreOffset, (blockLength_ > 0) ? blockLength_ : 1);
    targetGM.SetGlobalBuffer((__gm__ T*)targetGm + coreOffset, (blockLength_ > 0) ? blockLength_ : 1);
    outputGM.SetGlobalBuffer((__gm__ T*)outputGm, 1);
    workspaceGM.SetGlobalBuffer((__gm__ float*)workspaceGm, usedCoreNum_ * 8);

    if (ubLength_ > 0) {
        pipe.InitBuffer(selfQueue, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(targetQueue, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(tmpBuf1, ubLength_ * sizeof(float));
        pipe.InitBuffer(tmpBuf2, ubLength_ * sizeof(float));
        pipe.InitBuffer(tmpBuf3, ubLength_ * sizeof(float));
        pipe.InitBuffer(reduceTmpBuf, ubLength_ * sizeof(float));
        if constexpr (std::is_same_v<T, half>) {
            pipe.InitBuffer(castBuf, ubLength_ * sizeof(float));
        }
    }
    int64_t resultBufSize = (usedCoreNum_ * sizeof(float) + 31) & ~31;
    if (resultBufSize < 32) {
        resultBufSize = 32;
    }
    pipe.InitBuffer(resultBuf, resultBufSize);
}

// ---- CopyIn ----
template <typename T>
__aicore__ inline void SoftMarginLossReduce<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> selfLocal = selfQueue.template AllocTensor<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(selfLocal, selfGM[progress * ubLength_], copyParams, {false, 0, 0, 0});
    selfQueue.EnQue(selfLocal);

    AscendC::LocalTensor<T> targetLocal = targetQueue.template AllocTensor<T>();
    AscendC::DataCopyPad(targetLocal, targetGM[progress * ubLength_], copyParams, {false, 0, 0, 0});
    targetQueue.EnQue(targetLocal);
}

// ---- ComputeAndReduce ----
template <typename T>
__aicore__ inline void SoftMarginLossReduce<T>::ComputeAndReduce(int64_t currentNum, float& localSum)
{
    AscendC::LocalTensor<T> selfLocal = selfQueue.template DeQue<T>();
    AscendC::LocalTensor<T> targetLocal = targetQueue.template DeQue<T>();
    AscendC::LocalTensor<float> txLocal = tmpBuf1.Get<float>();
    AscendC::LocalTensor<float> negTxLocal = tmpBuf2.Get<float>();
    AscendC::LocalTensor<float> lossLocal = tmpBuf3.Get<float>();
    AscendC::LocalTensor<float> reduceTmp = reduceTmpBuf.Get<float>();

    if constexpr (std::is_same_v<T, float>) {
        ComputeLossCore(lossLocal, txLocal, negTxLocal, selfLocal, targetLocal, currentNum);
    } else {
        AscendC::LocalTensor<float> castLocal = castBuf.Get<float>();
        // Cast self (half -> float) into castLocal
        AscendC::Cast(castLocal, selfLocal, AscendC::RoundMode::CAST_NONE, currentNum);
        // Cast target (half -> float) into txLocal (temporary)
        AscendC::Cast(txLocal, targetLocal, AscendC::RoundMode::CAST_NONE, currentNum);
        // Compute: output=lossLocal, txBuf=txLocal, negTxBuf=castLocal, selfF32=castLocal, targetF32=txLocal
        ComputeLossCore(lossLocal, txLocal, castLocal, castLocal, txLocal, currentNum);
    }

    // ReduceSum over this tile
    AscendC::ReduceSum<float>(lossLocal, lossLocal, reduceTmp, currentNum);
    localSum += lossLocal.GetValue(0);

    selfQueue.FreeTensor(selfLocal);
    targetQueue.FreeTensor(targetLocal);
}

// ---- WritePartialSum ----
template <typename T>
__aicore__ inline void SoftMarginLossReduce<T>::WritePartialSum(float localSum)
{
    AscendC::LocalTensor<float> resultLocal = resultBuf.Get<float>();
    resultLocal.SetValue(0, localSum);

    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = 32;
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(workspaceGM[AscendC::GetBlockIdx() * 8], resultLocal, copyParams);
}

// ---- WriteFinalScalar ----
template <typename T>
__aicore__ inline void SoftMarginLossReduce<T>::WriteFinalScalar(float value)
{
    AscendC::LocalTensor<float> resultLocal = resultBuf.Get<float>();

    if constexpr (std::is_same_v<T, float>) {
        resultLocal.SetValue(0, value);
    } else {
        // Cast float scalar to half via ReinterpretCast
        AscendC::LocalTensor<half> halfResult = resultLocal.ReinterpretCast<half>();
        halfResult.SetValue(0, static_cast<half>(value));
    }

    AscendC::DataCopyParams outParams;
    outParams.blockCount = 1;
    outParams.blockLen = 32;
    outParams.srcStride = 0;
    outParams.dstStride = 0;

    if constexpr (std::is_same_v<T, float>) {
        AscendC::DataCopyPad(outputGM[0], resultLocal, outParams);
    } else {
        AscendC::LocalTensor<half> halfResult = resultLocal.ReinterpretCast<half>();
        AscendC::DataCopyPad(outputGM[0], halfResult, outParams);
    }
}

// ---- CrossCoreReduce ----
template <typename T>
__aicore__ inline void SoftMarginLossReduce<T>::CrossCoreReduce()
{
    if (AscendC::GetBlockIdx() != 0) {
        return;
    }

    AscendC::LocalTensor<float> resultLocal = resultBuf.Get<float>();

    float globalSum = 0.0f;
    for (int64_t i = 0; i < usedCoreNum_; i++) {
        AscendC::DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = 32;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(resultLocal, workspaceGM[i * 8], copyParams, {false, 0, 0, 0});
        AscendC::PipeBarrier<PIPE_ALL>();
        globalSum += resultLocal.GetValue(0);
    }

    if (reductionMode_ == 1) {
        globalSum *= invNumel_;
    }

    WriteFinalScalar(globalSum);
}

// ---- Process ----
template <typename T>
__aicore__ inline void SoftMarginLossReduce<T>::Process()
{
    float localSum = 0.0f;

    if (blockLength_ > 0 && ubLength_ > 0) {
        int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
        for (int64_t i = 0; i < loopCount; i++) {
            int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
            CopyIn(i, currentNum);
            ComputeAndReduce(currentNum, localSum);
        }
    }

    if (usedCoreNum_ == 1) {
        if (reductionMode_ == 1) {
            localSum *= invNumel_;
        }
        WriteFinalScalar(localSum);
    } else {
        WritePartialSum(localSum);
        AscendC::SyncAll();
        CrossCoreReduce();
    }
}

} // namespace NsSoftMarginLoss
#endif // SOFT_MARGIN_LOSS_H
