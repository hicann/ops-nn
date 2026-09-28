/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_RNN_DYNAMIC_AUGRU_OP_KERNEL_ARCH35_DYNAMIC_AUGRU_BASE_H_
#define OPS_RNN_DYNAMIC_AUGRU_OP_KERNEL_ARCH35_DYNAMIC_AUGRU_BASE_H_

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "dynamic_augru_tiling_data.h"
#include "lib/matmul_intf.h"

namespace DynamicAUGRU {
using namespace AscendC;

// On Ascend 950 (3510), hardware guarantees PIPE_V-to-PIPE_V ordering.
// Cross-pipeline memory dependencies still require the event synchronization below.

constexpr uint32_t kSequenceNone = 0;
constexpr uint32_t kSequenceLength = 1;
constexpr uint32_t kSequenceMask = 2;
constexpr uint32_t kGateOrderZrh = 0;
constexpr uint32_t kFpBufferCount = 10;
// Bound independent dot products; merge FP32 partials with regbase Kahan accumulation.
constexpr int64_t kProjectionGroup = 16;
// Keep short input reductions in smaller official Matmul chunks.  This uses
// the existing regbase kernel and tiling key; it only bounds the partial-dot
// product before the existing Kahan merge.
constexpr int64_t kShortProjectionGroup = 8;
constexpr int64_t kHiddenProjectionGroup = 8;

constexpr MatmulConfig kMatmulConfig = GetMDLConfig(true, false, 0, false, false, false, true);
// Preserve near-zero accuracy with the official FP32 regbase algorithm.
constexpr TanhConfig kTanhConfig = {TanhAlgo::SUBSECTION_COMPENSATION};

// Keep each arithmetic chain in registers. Separate Add/Sub operations are
// intentional: Kahan and TwoDiff depend on their individual FP32 roundings.
__simd_vf__ inline void AccumulatePartialVF(__ubuf__ float* total, __ubuf__ float* correction, __ubuf__ float* partial,
                                            __ubuf__ float* sum, uint32_t count)
{
    constexpr uint32_t lanes = GetVecLen() / sizeof(float);
    const uint16_t repeats = (count + lanes - 1) / lanes;
    Reg::RegTensor<float> partialProjection, accumulatedSum, roundingCorrection, adjusted, updatedSum, zero;
    Reg::MaskReg active, valid;
    for (uint16_t vectorIndex = 0; vectorIndex < repeats; ++vectorIndex) {
        active = Reg::UpdateMask<float>(count);
        Reg::LoadAlign(partialProjection, partial + vectorIndex * lanes);
        Reg::LoadAlign(accumulatedSum, sum + vectorIndex * lanes);
        Reg::LoadAlign(roundingCorrection, correction + vectorIndex * lanes);
        Reg::Sub(adjusted, partialProjection, roundingCorrection, active);
        Reg::Add(updatedSum, accumulatedSum, adjusted, active);
        Reg::Sub(roundingCorrection, updatedSum, accumulatedSum, active);
        Reg::Sub(roundingCorrection, roundingCorrection, adjusted, active);
        Reg::Compare<float, CMPMODE::EQ>(valid, roundingCorrection, roundingCorrection, active);
        Reg::Duplicate(zero, 0.0F);
        Reg::Select(roundingCorrection, roundingCorrection, zero, valid);
        Reg::StoreAlign(total + vectorIndex * lanes, updatedSum, active);
        Reg::StoreAlign(correction + vectorIndex * lanes, roundingCorrection, active);
    }
}

template <bool DirectCenter>
__simd_vf__ inline void CandidatePreactivationVF(__ubuf__ float* dst, __ubuf__ float* gateX, __ubuf__ float* gateH,
                                                 __ubuf__ float* reset, __ubuf__ float* halfReset,
                                                 __ubuf__ float* centeredTanh, uint32_t count)
{
    constexpr uint32_t lanes = GetVecLen() / sizeof(float);
    const uint16_t repeats = (count + lanes - 1) / lanes;
    Reg::RegTensor<float> inputProjection, hiddenProjection, resetGate, halfR, tanhR, halfH, center, centerReset,
        original;
    Reg::MaskReg active, centered, valid;
    for (uint16_t vectorIndex = 0; vectorIndex < repeats; ++vectorIndex) {
        active = Reg::UpdateMask<float>(count);
        Reg::LoadAlign(inputProjection, gateX + vectorIndex * lanes);
        Reg::LoadAlign(hiddenProjection, gateH + vectorIndex * lanes);
        Reg::LoadAlign(resetGate, reset + vectorIndex * lanes);
        Reg::LoadAlign(halfR, halfReset + vectorIndex * lanes);
        Reg::LoadAlign(tanhR, centeredTanh + vectorIndex * lanes);
        Reg::Move(original, resetGate, active);
        Reg::FusedMulDstAdd(original, hiddenProjection, inputProjection, active);
        if constexpr (DirectCenter) {
            // For short input projections, form sigmoid(s) through the official
            // tanh identity and fuse the final product/add.  This avoids the
            // cancellation in x + h/2 while keeping all arithmetic in FP32.
            Reg::Muls(centerReset, tanhR, 0.5F, active);
            Reg::Adds(centerReset, centerReset, 0.5F, active);
            Reg::FusedMulDstAdd(centerReset, hiddenProjection, inputProjection, active);
            Reg::Move(center, centerReset, active);
        } else {
            Reg::Muls(halfH, hiddenProjection, 0.5F, active);
            Reg::Add(center, inputProjection, halfH, active);
            Reg::FusedMulDstAdd(halfH, tanhR, center, active);
            Reg::Move(center, halfH, active);
        }
        Reg::Abs(halfR, halfR, active);
        Reg::Compares<float, CMPMODE::LE>(centered, halfR, 0.5F, active);
        if constexpr (DirectCenter) {
            Reg::Select(center, center, original, centered);
        } else {
            Reg::Select(center, halfH, original, centered);
        }
        // Keep the original IEEE expression if regrouping produced NaN.
        Reg::Compare<float, CMPMODE::EQ>(valid, center, center, active);
        Reg::Select(center, center, original, valid);
        Reg::StoreAlign(dst + vectorIndex * lanes, center, active);
    }
}

__simd_vf__ inline void UpdateStateVF(__ubuf__ float* result, __ubuf__ float* updateAttention, __ubuf__ float* update,
                                      __ubuf__ float* previous, __ubuf__ float* candidate, float attentionScale,
                                      uint32_t count)
{
    constexpr uint32_t lanes = GetVecLen() / sizeof(float);
    const uint16_t repeats = (count + lanes - 1) / lanes;
    Reg::RegTensor<float> attentionUpdateGate, previousState, candidateState, diff, virtualN, recoveredP, residual,
        base, corrected;
    Reg::MaskReg active, valid;
    for (uint16_t vectorIndex = 0; vectorIndex < repeats; ++vectorIndex) {
        active = Reg::UpdateMask<float>(count);
        Reg::LoadAlign(attentionUpdateGate, update + vectorIndex * lanes);
        Reg::LoadAlign(previousState, previous + vectorIndex * lanes);
        Reg::LoadAlign(candidateState, candidate + vectorIndex * lanes);
        Reg::Muls(attentionUpdateGate, attentionUpdateGate, attentionScale, active);
        Reg::Sub(diff, previousState, candidateState, active);
        Reg::Move(base, attentionUpdateGate, active);
        Reg::FusedMulDstAdd(base, diff, candidateState, active);
        // TwoDiff: retain the baseline operation order and FP32 rounding points.
        Reg::Sub(virtualN, previousState, diff, active);
        Reg::Add(recoveredP, diff, virtualN, active);
        Reg::Sub(virtualN, virtualN, candidateState, active);
        Reg::Sub(recoveredP, previousState, recoveredP, active);
        Reg::Add(residual, recoveredP, virtualN, active);
        Reg::Move(corrected, attentionUpdateGate, active);
        Reg::FusedMulDstAdd(corrected, residual, base, active);
        Reg::Compare<float, CMPMODE::EQ>(valid, corrected, corrected, active);
        Reg::Select(corrected, corrected, base, valid);
        Reg::StoreAlign(updateAttention + vectorIndex * lanes, attentionUpdateGate, active);
        Reg::StoreAlign(result + vectorIndex * lanes, corrected, active);
    }
}

template <typename StateT, uint32_t SequenceMode>
class DynamicAUGRUBase {
public:
    TPipe pipe;
    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, half>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, half>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, StateT>, kMatmulConfig>
        inputMM;
    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>, kMatmulConfig>
        hiddenMM;

    __aicore__ inline DynamicAUGRUBase() = default;

    __aicore__ inline void Init(GM_ADDR inputSequence, GM_ADDR weightInput, GM_ADDR weightHidden, GM_ADDR weightAtt,
                                GM_ADDR biasInput, GM_ADDR biasHidden, GM_ADDR sequenceLength, GM_ADDR initH,
                                GM_ADDR outputSequence, GM_ADDR outputH, GM_ADDR update, GM_ADDR updateAtt,
                                GM_ADDR reset, GM_ADDR newState, GM_ADDR hiddenNew, GM_ADDR workspace,
                                const DynamicAUGRUTilingData* tilingData)
    {
        tilingData_ = tilingData;
        const int64_t cores = tilingData_->usedAicCoreNum;
        const int64_t core = GetBlockIdx();
        batchStart_ = tilingData_->batchSize * core / cores;
        batchEnd_ = tilingData_->batchSize * (core + 1) / cores;
        const int64_t rows = tilingData_->timeSize * tilingData_->batchSize;
        inputRowStart_ = rows * core / cores;
        inputRowCount_ = rows * (core + 1) / cores - inputRowStart_;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(inputSequence));
        weightInputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(weightInput));
        weightHiddenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(weightHidden));
        attentionGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(weightAtt));
        if (tilingData_->hasBiasInput != 0U) {
            biasInputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ StateT*>(biasInput));
        }
        if (tilingData_->hasBiasHidden != 0U) {
            biasHiddenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ StateT*>(biasHidden));
        }
        if constexpr (SequenceMode == kSequenceLength) {
            sequenceLengthGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sequenceLength));
        } else if constexpr (SequenceMode == kSequenceMask) {
            sequenceMaskGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(sequenceLength));
        }
        if (tilingData_->hasInitH != 0U) {
            initHGm_.SetGlobalBuffer(reinterpret_cast<__gm__ StateT*>(initH));
        }

        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ StateT*>(outputSequence));
        outputHGm_.SetGlobalBuffer(reinterpret_cast<__gm__ StateT*>(outputH));
        updateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ StateT*>(update));
        updateAttGm_.SetGlobalBuffer(reinterpret_cast<__gm__ StateT*>(updateAtt));
        resetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ StateT*>(reset));
        newGm_.SetGlobalBuffer(reinterpret_cast<__gm__ StateT*>(newState));
        hiddenNewGm_.SetGlobalBuffer(reinterpret_cast<__gm__ StateT*>(hiddenNew));

        inputProjectionGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(workspace + tilingData_->inputProjectionOffset));
        hiddenProjectionGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(workspace + tilingData_->hiddenProjectionOffset));
        const uint64_t hiddenBytes = ((tilingData_->batchSize * 3 * tilingData_->hiddenSize * sizeof(float) + 511) /
                                      512) *
                                     512;
        hiddenSumGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(workspace + tilingData_->hiddenProjectionOffset + hiddenBytes));
        hiddenCorrectionGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(workspace + tilingData_->hiddenProjectionOffset + 2 * hiddenBytes));
        stateFp32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + tilingData_->stateFp32Offset));
        weightHiddenFp32Gm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(workspace + tilingData_->weightHiddenFp32Offset));
        biasHiddenFp32Gm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(workspace + tilingData_->weightHiddenFp32Offset) +
            3 * tilingData_->hiddenSize * tilingData_->hiddenSize);

        const uint64_t fpBytes = static_cast<uint64_t>(kFpBufferCount) * tilingData_->tileHidden * sizeof(float);
        const uint64_t stateBytes = static_cast<uint64_t>(2) * tilingData_->tileHidden * sizeof(StateT);
        const uint64_t halfBytes = static_cast<uint64_t>(2) * tilingData_->tileHidden * sizeof(half);
        pipe.InitBuffer(fpBuffer_, fpBytes);
        pipe.InitBuffer(stateBuffer_, stateBytes);
        pipe.InitBuffer(halfBuffer_, halfBytes);
    }

    __aicore__ inline void Process()
    {
        ComputeInputProjection();
        SyncAll();
        InitializeState();
        // Convert invariant hidden weights/bias once, not at every time step.
        if (GetBlockIdx() == 0) {
            ConvertToFp32(weightHiddenFp32Gm_, weightHiddenGm_, 3 * tilingData_->hiddenSize * tilingData_->hiddenSize);
            if (tilingData_->hasBiasHidden != 0U) {
                ConvertToFp32(biasHiddenFp32Gm_, biasHiddenGm_, 3 * tilingData_->hiddenSize);
            }
        }
        SyncAll();

        for (int64_t timeStepIndex = 0; timeStepIndex < tilingData_->timeSize; ++timeStepIndex) {
            ComputeHiddenProjection();
            SyncAll();
            ComputeOneTimeStep(timeStepIndex);
            SyncAll();
        }
    }

private:
    __aicore__ inline void ComputeInputProjection()
    {
        // Reuse the per-core projection scratch before recurrent computation starts.
        const int64_t width = 3 * tilingData_->hiddenSize;
        const int64_t scratchRows = batchEnd_ - batchStart_;
        const int64_t end = inputRowStart_ + inputRowCount_;
        const int64_t projectionGroup = tilingData_->inputSize <= 64 ? kShortProjectionGroup : kProjectionGroup;
        for (int64_t row = inputRowStart_; row < end; row += scratchRows) {
            const int64_t rows = Min(scratchRows, end - row);
            for (int64_t reductionOffset = 0; reductionOffset < tilingData_->inputSize;
                 reductionOffset += projectionGroup) {
                inputMM.SetTensorA(xGm_[row * tilingData_->inputSize + reductionOffset]);
                inputMM.SetTensorB(weightInputGm_[reductionOffset * width]);
                inputMM.SetTail(rows, width, Min(projectionGroup, tilingData_->inputSize - reductionOffset));
                inputMM.DisableBias();
                inputMM.IterateAll(hiddenProjectionGm_[batchStart_ * width], false);
                inputMM.End();
                AccumulateProjectionPartial<true>(inputProjectionGm_, row * width, rows, reductionOffset == 0,
                                                  reductionOffset + projectionGroup >= tilingData_->inputSize);
            }
        }
    }

    __aicore__ inline void ComputeHiddenProjection()
    {
        // HF32 truncates the mantissa; recurrence must retain FP32 precision.
        hiddenMM.SetHF32(false, 0);
        const int64_t width = 3 * tilingData_->hiddenSize;
        for (int64_t reductionOffset = 0; reductionOffset < tilingData_->hiddenSize;
             reductionOffset += kHiddenProjectionGroup) {
            hiddenMM.SetTensorA(stateFp32Gm_[batchStart_ * tilingData_->hiddenSize + reductionOffset]);
            hiddenMM.SetTensorB(weightHiddenFp32Gm_[reductionOffset * width]);
            hiddenMM.SetTail(batchEnd_ - batchStart_, width,
                             Min(kHiddenProjectionGroup, tilingData_->hiddenSize - reductionOffset));
            hiddenMM.DisableBias();
            hiddenMM.IterateAll(hiddenProjectionGm_[batchStart_ * width], false);
            hiddenMM.End();
            AccumulateProjectionPartial<false>(hiddenProjectionGm_, batchStart_ * width, batchEnd_ - batchStart_,
                                               reductionOffset == 0,
                                               reductionOffset + kHiddenProjectionGroup >= tilingData_->hiddenSize);
        }
    }

    template <bool IsInput>
    __aicore__ inline void AccumulateProjectionPartial(const GlobalTensor<float>& output, int64_t outputOffset,
                                                       int64_t rows, bool first, bool last)
    {
        LocalTensor<float> fp = fpBuffer_.Get<float>();
        const int64_t stride = tilingData_->tileHidden;
        LocalTensor<float> partial = fp;
        LocalTensor<float> sum = fp[stride];
        LocalTensor<float> correction = fp[2 * stride];
        LocalTensor<float> adjusted = fp[3 * stride];
        LocalTensor<float> total = fp[4 * stride];
        const int64_t begin = batchStart_ * 3 * tilingData_->hiddenSize;
        const int64_t end = begin + rows * 3 * tilingData_->hiddenSize;
        const int64_t width = 3 * tilingData_->hiddenSize;
        for (int64_t offset = begin; offset < end;) {
            const int64_t biasOffset = offset % width;
            const uint32_t count = static_cast<uint32_t>(Min(stride, Min(end - offset, width - biasOffset)));
            VToMTE2Sync();
            CopyIn(partial, hiddenProjectionGm_[offset], count);
            if (!first) {
                CopyIn(sum, hiddenSumGm_[offset], count);
                CopyIn(correction, hiddenCorrectionGm_[offset], count);
            }
            MTE2ToVSync();
            if (first) {
                Adds(total, partial, 0.0F, count);
                Duplicate(correction, 0.0F, count);
            } else {
                AccumulatePartialVF((__ubuf__ float*)total.GetPhyAddr(), (__ubuf__ float*)correction.GetPhyAddr(),
                                    (__ubuf__ float*)partial.GetPhyAddr(), (__ubuf__ float*)sum.GetPhyAddr(), count);
            }
            if (last && (IsInput ? tilingData_->hasBiasInput : tilingData_->hasBiasHidden) != 0U) {
                // Include bias before discarding the accumulated residual. Rounding the
                // dot product first and then adding bias introduces an extra rounding.
                LocalTensor<float> bias = fp[6 * stride];
                if constexpr (IsInput && std::is_same<StateT, half>::value) {
                    LocalTensor<half> halfBias = halfBuffer_.Get<half>();
                    CopyIn(halfBias, biasInputGm_[biasOffset], count);
                    MTE2ToVSync();
                    Cast(bias, halfBias, RoundMode::CAST_NONE, count);
                } else if constexpr (IsInput) {
                    CopyIn(bias, biasInputGm_[biasOffset], count);
                    MTE2ToVSync();
                } else {
                    CopyIn(bias, biasHiddenFp32Gm_[biasOffset], count);
                    MTE2ToVSync();
                }
                Sub(adjusted, bias, correction, count);
                Add(total, total, adjusted, count);
            }
            VToMTE3Sync();
            CopyOut(last ? output[outputOffset + offset - begin] : hiddenSumGm_[offset], total, count);
            CopyOut(hiddenCorrectionGm_[offset], correction, count);
            MTE3ToVSync();
            MTE3ToMTE2Sync();
            offset += count;
        }
    }

    __aicore__ inline void InitializeState()
    {
        LocalTensor<float> fp = fpBuffer_.Get<float>();
        LocalTensor<float> hPrev = fp;
        LocalTensor<StateT> stateIn = stateBuffer_.Get<StateT>();
        const int64_t hidden = tilingData_->hiddenSize;
        for (int64_t batchIndex = batchStart_; batchIndex < batchEnd_; ++batchIndex) {
            for (int64_t hiddenOffset = 0; hiddenOffset < hidden; hiddenOffset += tilingData_->tileHidden) {
                const uint32_t count = static_cast<uint32_t>(Min(tilingData_->tileHidden, hidden - hiddenOffset));
                const int64_t offset = batchIndex * hidden + hiddenOffset;
                if (tilingData_->hasInitH == 0U) {
                    Duplicate(hPrev, 0.0F, count);
                } else if constexpr (std::is_same<StateT, float>::value) {
                    CopyIn(hPrev, initHGm_[offset], count);
                    MTE2ToVSync();
                } else {
                    CopyIn(stateIn, initHGm_[offset], count);
                    MTE2ToVSync();
                    Cast(hPrev, stateIn, RoundMode::CAST_NONE, count);
                }
                VToMTE3Sync();
                CopyOut(stateFp32Gm_[offset], hPrev, count);
                MTE3ToVSync();
                MTE3ToMTE2Sync();
            }
        }
    }

    template <typename T>
    __aicore__ inline void ConvertToFp32(const GlobalTensor<float>& dst, const GlobalTensor<T>& src, int64_t elements)
    {
        LocalTensor<float> value = fpBuffer_.Get<float>();
        LocalTensor<T> input = halfBuffer_.Get<T>();
        for (int64_t offset = 0; offset < elements; offset += tilingData_->tileHidden) {
            const uint32_t count = static_cast<uint32_t>(Min(tilingData_->tileHidden, elements - offset));
            if constexpr (std::is_same<T, float>::value) {
                CopyIn(value, src[offset], count);
                MTE2ToVSync();
            } else {
                CopyIn(input, src[offset], count);
                MTE2ToVSync();
                Cast(value, input, RoundMode::CAST_NONE, count);
            }
            VToMTE3Sync();
            CopyOut(dst[offset], value, count);
            MTE3ToVSync();
            MTE3ToMTE2Sync();
        }
    }

    __aicore__ inline void ComputeOneTimeStep(int64_t timeIndex)
    {
        LocalTensor<float> fp = fpBuffer_.Get<float>();
        const int64_t stride = tilingData_->tileHidden;
        LocalTensor<float> gateX = fp;
        LocalTensor<float> hiddenNew = fp[stride];
        LocalTensor<float> hPrev = fp[2 * stride];
        LocalTensor<float> update = fp[3 * stride];
        LocalTensor<float> updateAttention = fp[4 * stride];
        LocalTensor<float> reset = fp[5 * stride];
        LocalTensor<float> candidate = fp[6 * stride];
        LocalTensor<float> tmp = fp[7 * stride];
        LocalTensor<float> result = fp[8 * stride];
        LocalTensor<float> mask = fp[9 * stride];
        LocalTensor<half> halfInOut = halfBuffer_.Get<half>();

        const int64_t hidden = tilingData_->hiddenSize;
        const int64_t gateWidth = 3 * hidden;
        const int64_t zGate = tilingData_->gateOrder == kGateOrderZrh ? 0 : 1;
        const int64_t rGate = tilingData_->gateOrder == kGateOrderZrh ? 1 : 0;
        for (int64_t batchIndex = batchStart_; batchIndex < batchEnd_; ++batchIndex) {
            const float attention = static_cast<float>(
                attentionGm_.GetValue(timeIndex * tilingData_->batchSize + batchIndex));
            bool batchValid = true;
            if constexpr (SequenceMode == kSequenceLength) {
                batchValid = timeIndex < static_cast<int64_t>(sequenceLengthGm_.GetValue(batchIndex));
            }
            for (int64_t hiddenOffset = 0; hiddenOffset < hidden; hiddenOffset += tilingData_->tileHidden) {
                const uint32_t count = static_cast<uint32_t>(Min(tilingData_->tileHidden, hidden - hiddenOffset));
                const int64_t projectionRow = (timeIndex * tilingData_->batchSize + batchIndex) * gateWidth;
                const int64_t hiddenRow = batchIndex * gateWidth;
                const int64_t stateOffset = batchIndex * hidden + hiddenOffset;

                LoadProjectionGate(gateX, hiddenNew, projectionRow, hiddenRow, zGate, hiddenOffset, count);
                Add(update, gateX, hiddenNew, count);
                Sigmoid(update, update, count);

                LoadProjectionGate(gateX, hiddenNew, projectionRow, hiddenRow, rGate, hiddenOffset, count);
                Add(reset, gateX, hiddenNew, count);
                // Near zero, keep the small displacement from sigmoid(0) = 1/2
                // until candidate evaluation instead of rounding it into the reset gate.
                Muls(hPrev, reset, 0.5F, count);
                Tanh<float, false, kTanhConfig>(mask, hPrev, count);
                Sigmoid(reset, reset, count);

                // Candidate hidden projection is retained as the hidden_new output.
                LoadProjectionGate(gateX, hiddenNew, projectionRow, hiddenRow, 2, hiddenOffset, count);
                if (tilingData_->inputSize <= 64) {
                    CandidatePreactivationVF<true>(
                        (__ubuf__ float*)tmp.GetPhyAddr(), (__ubuf__ float*)gateX.GetPhyAddr(),
                        (__ubuf__ float*)hiddenNew.GetPhyAddr(), (__ubuf__ float*)reset.GetPhyAddr(),
                        (__ubuf__ float*)hPrev.GetPhyAddr(), (__ubuf__ float*)mask.GetPhyAddr(), count);
                } else {
                    CandidatePreactivationVF<false>(
                        (__ubuf__ float*)tmp.GetPhyAddr(), (__ubuf__ float*)gateX.GetPhyAddr(),
                        (__ubuf__ float*)hiddenNew.GetPhyAddr(), (__ubuf__ float*)reset.GetPhyAddr(),
                        (__ubuf__ float*)hPrev.GetPhyAddr(), (__ubuf__ float*)mask.GetPhyAddr(), count);
                }
                Tanh<float, false, kTanhConfig>(candidate, tmp, count);

                // hPrev was vector scratch above; finish writes before DMA reloads it.
                VToMTE2Sync();
                CopyIn(hPrev, stateFp32Gm_[stateOffset], count);
                MTE2ToVSync();
                UpdateStateVF((__ubuf__ float*)result.GetPhyAddr(), (__ubuf__ float*)updateAttention.GetPhyAddr(),
                              (__ubuf__ float*)update.GetPhyAddr(), (__ubuf__ float*)hPrev.GetPhyAddr(),
                              (__ubuf__ float*)candidate.GetPhyAddr(), 1.0F - attention, count);

                if (SequenceMode == kSequenceLength && !batchValid) {
                    Adds(result, hPrev, 0.0F, count);
                } else if (SequenceMode == kSequenceMask) {
                    const int64_t maskOffset = (timeIndex * tilingData_->batchSize + batchIndex) * hidden +
                                               hiddenOffset;
                    CopyIn(halfInOut, sequenceMaskGm_[maskOffset], count);
                    MTE2ToVSync();
                    Cast(mask, halfInOut, RoundMode::CAST_NONE, count);
                    Sub(tmp, result, hPrev, count);
                    Mul(tmp, tmp, mask, count);
                    Add(result, hPrev, tmp, count);
                }

                const int64_t outputOffset = (timeIndex * tilingData_->batchSize + batchIndex) * hidden + hiddenOffset;
                StoreOutputs(outputOffset, stateOffset, count, result, update, updateAttention, reset, candidate,
                             hiddenNew);
            }
        }
    }

    __aicore__ inline void LoadProjectionGate(const LocalTensor<float>& gateX, const LocalTensor<float>& gateH,
                                              int64_t projectionRow, int64_t hiddenRow, int64_t gate,
                                              int64_t hiddenOffset, uint32_t count)
    {
        // gateX/gateH are reused by each gate.  Wait for the vector pipeline to
        // finish consuming the previous gate before MTE2 overwrites them.
        VToMTE2Sync();
        CopyIn(gateX, inputProjectionGm_[projectionRow + gate * tilingData_->hiddenSize + hiddenOffset], count);
        CopyIn(gateH, hiddenProjectionGm_[hiddenRow + gate * tilingData_->hiddenSize + hiddenOffset], count);
        MTE2ToVSync();
    }

    __aicore__ inline void StoreOutputs(int64_t outputOffset, int64_t stateOffset, uint32_t count,
                                        const LocalTensor<float>& result, const LocalTensor<float>& update,
                                        const LocalTensor<float>& updateAttention, const LocalTensor<float>& reset,
                                        const LocalTensor<float>& candidate, const LocalTensor<float>& hiddenNew)
    {
        LocalTensor<half> halfOut = halfBuffer_.Get<half>();
        VToMTE3Sync();
        // Preserve the unrounded recurrence for both output dtypes.
        CopyOut(stateFp32Gm_[stateOffset], result, count);
        if constexpr (std::is_same<StateT, float>::value) {
            CopyOut(yGm_[outputOffset], result, count);
            CopyOut(outputHGm_[outputOffset], result, count);
            CopyOut(updateGm_[outputOffset], update, count);
            CopyOut(updateAttGm_[outputOffset], updateAttention, count);
            CopyOut(resetGm_[outputOffset], reset, count);
            CopyOut(newGm_[outputOffset], candidate, count);
            CopyOut(hiddenNewGm_[outputOffset], hiddenNew, count);
            MTE3ToVSync();
        } else {
            CastAndCopy(yGm_[outputOffset], result, halfOut, count);
            CopyOut(outputHGm_[outputOffset], halfOut, count);
            MTE3ToVSync();
            CastAndCopy(updateGm_[outputOffset], update, halfOut, count);
            MTE3ToVSync();
            CastAndCopy(updateAttGm_[outputOffset], updateAttention, halfOut, count);
            MTE3ToVSync();
            CastAndCopy(resetGm_[outputOffset], reset, halfOut, count);
            MTE3ToVSync();
            CastAndCopy(newGm_[outputOffset], candidate, halfOut, count);
            MTE3ToVSync();
            CastAndCopy(hiddenNewGm_[outputOffset], hiddenNew, halfOut, count);
            MTE3ToVSync();
        }
        MTE3ToMTE2Sync();
    }

    __aicore__ inline void CastAndCopy(const GlobalTensor<half>& dst, const LocalTensor<float>& src,
                                       const LocalTensor<half>& tmp, uint32_t count)
    {
        Cast(tmp, src, RoundMode::CAST_RINT, count);
        VToMTE3Sync();
        CopyOut(dst, tmp, count);
    }

    template <typename T>
    __aicore__ inline void CopyIn(const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint32_t count)
    {
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(dst, src, params, padParams);
    }

    template <typename T>
    __aicore__ inline void CopyOut(const GlobalTensor<T>& dst, const LocalTensor<T>& src, uint32_t count)
    {
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPad(dst, src, params);
    }

    template <typename T>
    __aicore__ inline T Min(T lhs, T rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

    __aicore__ inline void VToMTE3Sync()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void MTE2ToVSync()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
    }

    __aicore__ inline void VToMTE2Sync()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
    }

    __aicore__ inline void MTE3ToVSync()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
    }

    __aicore__ inline void MTE3ToMTE2Sync()
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);
    }

private:
    const DynamicAUGRUTilingData* tilingData_ = nullptr;
    int64_t batchStart_ = 0;
    int64_t batchEnd_ = 0;
    int64_t inputRowStart_ = 0;
    int64_t inputRowCount_ = 0;
    TBuf<TPosition::VECCALC> fpBuffer_;
    TBuf<TPosition::VECCALC> stateBuffer_;
    TBuf<TPosition::VECCALC> halfBuffer_;

    GlobalTensor<half> xGm_;
    GlobalTensor<half> weightInputGm_;
    GlobalTensor<half> weightHiddenGm_;
    GlobalTensor<half> attentionGm_;
    GlobalTensor<StateT> biasInputGm_;
    GlobalTensor<StateT> biasHiddenGm_;
    GlobalTensor<int32_t> sequenceLengthGm_;
    GlobalTensor<half> sequenceMaskGm_;
    GlobalTensor<StateT> initHGm_;

    GlobalTensor<StateT> yGm_;
    GlobalTensor<StateT> outputHGm_;
    GlobalTensor<StateT> updateGm_;
    GlobalTensor<StateT> updateAttGm_;
    GlobalTensor<StateT> resetGm_;
    GlobalTensor<StateT> newGm_;
    GlobalTensor<StateT> hiddenNewGm_;

    GlobalTensor<float> inputProjectionGm_;
    GlobalTensor<float> hiddenProjectionGm_;
    GlobalTensor<float> hiddenSumGm_;
    GlobalTensor<float> hiddenCorrectionGm_;
    GlobalTensor<float> stateFp32Gm_;
    GlobalTensor<float> weightHiddenFp32Gm_;
    GlobalTensor<float> biasHiddenFp32Gm_;
};
} // namespace DynamicAUGRU
#endif // OPS_RNN_DYNAMIC_AUGRU_OP_KERNEL_ARCH35_DYNAMIC_AUGRU_BASE_H_
