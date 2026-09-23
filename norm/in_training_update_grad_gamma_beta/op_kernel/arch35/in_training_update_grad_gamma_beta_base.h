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
 * \file in_training_update_grad_gamma_beta_base.h
 * \brief Ascend 950 kernel for two deterministic axis-zero reductions.
 */

#ifndef IN_TRAINING_UPDATE_GRAD_GAMMA_BETA_BASE_H_
#define IN_TRAINING_UPDATE_GRAD_GAMMA_BETA_BASE_H_

#include "kernel_operator.h"
#include "in_training_update_grad_gamma_beta_common.h"
#include "in_training_update_grad_gamma_beta_tiling_data.h"

namespace NsINTrainingUpdateGradGammaBeta {
using namespace AscendC;

class INTrainingUpdateGradGammaBetaKernel {
public:
    __aicore__ inline INTrainingUpdateGradGammaBetaKernel() = default;

    __aicore__ inline void Init(GM_ADDR resGamma, GM_ADDR resBeta, GM_ADDR pdGamma, GM_ADDR pdBeta,
                                const INTrainingUpdateGradGammaBetaTilingData* tilingData, TPipe* pipe)
    {
        pipe_ = pipe;
        reduceCount_ = tilingData->reduceCount;
        outputElements_ = tilingData->outputElements;
        baseBlocksPerCore_ = tilingData->baseBlocksPerCore;
        tileElements_ = tilingData->tileElements;
        reduceRowsPerTile_ = tilingData->reduceRowsPerTile;
        extraBlockCoreCount_ = tilingData->extraBlockCoreCount;
        blockElements_ = tilingData->blockElements;
        usedCoreNum_ = tilingData->usedCoreNum;
        safeMagnitude_ = tilingData->safeMagnitude;
        inputScale_ = tilingData->inputScale;
        outputScale_ = tilingData->outputScale;
        partialCount_ = reduceCount_ < static_cast<int64_t>(IN_TRAINING_UPDATE_GRAD_GAMMA_BETA_MAX_PARTIALS) ?
                            static_cast<uint32_t>(reduceCount_ > 0 ? reduceCount_ : 1) :
                            IN_TRAINING_UPDATE_GRAD_GAMMA_BETA_MAX_PARTIALS;
        stateSlotCount_ = reduceCount_ <= 2 ? 1U : (partialCount_ < 4U ? 4U : partialCount_);

        blockIndex_ = static_cast<uint32_t>(GetBlockIdx());
        const int64_t precedingExtraBlocks = blockIndex_ < extraBlockCoreCount_ ?
                                                 static_cast<int64_t>(blockIndex_) :
                                                 static_cast<int64_t>(extraBlockCoreCount_);
        const int64_t startBlock = static_cast<int64_t>(blockIndex_) * baseBlocksPerCore_ + precedingExtraBlocks;
        const int64_t ownedBlocks = baseBlocksPerCore_ + static_cast<int64_t>(blockIndex_ < extraBlockCoreCount_);
        coreOffset_ = startBlock * static_cast<int64_t>(blockElements_);
        coreElements_ = 0;
        if (blockIndex_ < usedCoreNum_ && coreOffset_ < outputElements_) {
            const int64_t remaining = outputElements_ - coreOffset_;
            const int64_t ownedElements = ownedBlocks * static_cast<int64_t>(blockElements_);
            coreElements_ = remaining < ownedElements ? remaining : ownedElements;
        }

        const int64_t inputElements = reduceCount_ * outputElements_;
        resGammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(resGamma), inputElements);
        resBetaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(resBeta), inputElements);
        pdGammaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pdGamma), outputElements_);
        pdBetaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(pdBeta), outputElements_);

        const uint32_t accumulatorElements = (tileElements_ + VL_FP32 - 1U) / VL_FP32 * VL_FP32;
        accumulatorStride_ = accumulatorElements;
        const uint32_t accumulatorBytes = accumulatorElements * static_cast<uint32_t>(sizeof(float));
        const uint32_t tileBytes = tileElements_ * static_cast<uint32_t>(sizeof(float));
        const uint32_t inputTailSlackBytes = (VL_FP32 - blockElements_) * static_cast<uint32_t>(sizeof(float));
        const uint32_t inputTileBytes = reduceRowsPerTile_ * tileBytes + inputTailSlackBytes;
        pipe_->InitBuffer(inputQueue_, 1, inputTileBytes);
        pipe_->InitBuffer(accumulatorBuffer_, accumulatorBytes * stateSlotCount_);
        if (reduceCount_ > 2) {
            pipe_->InitBuffer(lengthBuffer_, accumulatorBytes);
            pipe_->InitBuffer(magnitudeBuffer_, accumulatorBytes);
        }
    }

    __aicore__ inline void Process()
    {
        if (coreElements_ == 0) {
            return;
        }
        const event_t vectorToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        const event_t mte3ToVector = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        const event_t vectorToScalar = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        for (int64_t progress = 0; progress < coreElements_; progress += static_cast<int64_t>(tileElements_)) {
            const int64_t remaining = coreElements_ - progress;
            const uint32_t currentElements = static_cast<uint32_t>(
                remaining < static_cast<int64_t>(tileElements_) ? remaining : static_cast<int64_t>(tileElements_));
            const int64_t outputOffset = coreOffset_ + progress;
            ReduceInput(resGammaGm_, pdGammaGm_, outputOffset, currentElements, vectorToMte3, mte3ToVector,
                        vectorToScalar);
            ReduceInput(resBetaGm_, pdBetaGm_, outputOffset, currentElements, vectorToMte3, mte3ToVector,
                        vectorToScalar);
        }
    }

private:
    __aicore__ inline void ReduceInput(const GlobalTensor<float>& inputGm, GlobalTensor<float>& outputGm,
                                       int64_t outputOffset, uint32_t currentElements, event_t vectorToMte3,
                                       event_t mte3ToVector, event_t vectorToScalar)
    {
        LocalTensor<float> accumulator = accumulatorBuffer_.Get<float>();
        if (reduceCount_ <= 2) {
            ReduceInputDirect(inputGm, outputGm, outputOffset, currentElements, accumulator, vectorToMte3,
                              mte3ToVector);
            return;
        }

        LocalTensor<float> lengths = lengthBuffer_.Get<float>();
        LocalTensor<float> magnitude = magnitudeBuffer_.Get<float>();
        asc_vf_call<&ZeroFloatBuffer>(reinterpret_cast<__ubuf__ float*>(magnitude.GetPhyAddr()), currentElements);

        bool retryWithExpansion = reduceCount_ > FAST_PATH_MAX_REDUCE_COUNT;
        if (!retryWithExpansion) {
            asc_vf_call<&ZeroFastReductionState>(reinterpret_cast<__ubuf__ float*>(accumulator.GetPhyAddr()),
                                                 currentElements, accumulatorStride_);
            for (int64_t reduceStart = 0; reduceStart < reduceCount_;
                 reduceStart += static_cast<int64_t>(reduceRowsPerTile_)) {
                const int64_t remainingRows = reduceCount_ - reduceStart;
                const uint16_t currentRows = static_cast<uint16_t>(remainingRows <
                                                                           static_cast<int64_t>(reduceRowsPerTile_) ?
                                                                       remainingRows :
                                                                       static_cast<int64_t>(reduceRowsPerTile_));
                const int64_t inputOffset = reduceStart * outputElements_ + outputOffset;
                LoadAndAccumulateFast(inputGm, inputOffset, currentElements, currentRows, accumulator, magnitude);
            }

            asc_vf_call<&FinalizeFastAndMarkRetry>(reinterpret_cast<__ubuf__ float*>(accumulator.GetPhyAddr()),
                                                   reinterpret_cast<const __ubuf__ float*>(magnitude.GetPhyAddr()),
                                                   currentElements, accumulatorStride_, safeMagnitude_);
            PipeBarrier<PIPE_V>();
            LocalTensor<float> retryFlags = accumulator[2U * accumulatorStride_];
            LocalTensor<float> retryWork = accumulator[3U * accumulatorStride_];
            ReduceMax<float>(retryWork, retryFlags, retryWork, currentElements, false);
            SetFlag<HardEvent::V_S>(vectorToScalar);
            WaitFlag<HardEvent::V_S>(vectorToScalar);
            retryWithExpansion = retryWork.GetValue(0) != 0.0F;
            if (!retryWithExpansion) {
                CopyOutput(outputGm, outputOffset, currentElements, accumulator, vectorToMte3, mte3ToVector);
                return;
            }
        } else {
            for (int64_t reduceStart = 0; reduceStart < reduceCount_;
                 reduceStart += static_cast<int64_t>(reduceRowsPerTile_)) {
                const int64_t remainingRows = reduceCount_ - reduceStart;
                const uint16_t currentRows = static_cast<uint16_t>(remainingRows <
                                                                           static_cast<int64_t>(reduceRowsPerTile_) ?
                                                                       remainingRows :
                                                                       static_cast<int64_t>(reduceRowsPerTile_));
                const int64_t inputOffset = reduceStart * outputElements_ + outputOffset;
                LoadAndUpdateMagnitude(inputGm, inputOffset, currentElements, currentRows, magnitude);
            }
        }

        asc_vf_call<&ZeroExpansionState>(reinterpret_cast<__ubuf__ float*>(accumulator.GetPhyAddr()),
                                         reinterpret_cast<__ubuf__ float*>(lengths.GetPhyAddr()), currentElements,
                                         accumulatorStride_, partialCount_);
        for (int64_t reduceStart = 0; reduceStart < reduceCount_;
             reduceStart += static_cast<int64_t>(reduceRowsPerTile_)) {
            const int64_t remainingRows = reduceCount_ - reduceStart;
            const uint16_t currentRows = static_cast<uint16_t>(
                remainingRows < static_cast<int64_t>(reduceRowsPerTile_) ? remainingRows :
                                                                           static_cast<int64_t>(reduceRowsPerTile_));
            const int64_t inputOffset = reduceStart * outputElements_ + outputOffset;
            LoadAndAccumulateExpansion(inputGm, inputOffset, currentElements, currentRows, accumulator, lengths,
                                       magnitude);
        }

        asc_vf_call<&FinalizeReduction>(reinterpret_cast<__ubuf__ float*>(accumulator.GetPhyAddr()),
                                        reinterpret_cast<const __ubuf__ float*>(lengths.GetPhyAddr()),
                                        reinterpret_cast<const __ubuf__ float*>(magnitude.GetPhyAddr()),
                                        currentElements, accumulatorStride_, partialCount_, safeMagnitude_,
                                        outputScale_);

        CopyOutput(outputGm, outputOffset, currentElements, accumulator, vectorToMte3, mte3ToVector);
    }

    __aicore__ inline void ReduceInputDirect(const GlobalTensor<float>& inputGm, GlobalTensor<float>& outputGm,
                                             int64_t outputOffset, uint32_t currentElements,
                                             const LocalTensor<float>& accumulator, event_t vectorToMte3,
                                             event_t mte3ToVector)
    {
        asc_vf_call<&ZeroFloatBuffer>(reinterpret_cast<__ubuf__ float*>(accumulator.GetPhyAddr()), currentElements);
        for (int64_t reduceStart = 0; reduceStart < reduceCount_;
             reduceStart += static_cast<int64_t>(reduceRowsPerTile_)) {
            const int64_t remainingRows = reduceCount_ - reduceStart;
            const uint16_t currentRows = static_cast<uint16_t>(
                remainingRows < static_cast<int64_t>(reduceRowsPerTile_) ? remainingRows :
                                                                           static_cast<int64_t>(reduceRowsPerTile_));
            const int64_t inputOffset = reduceStart * outputElements_ + outputOffset;
            LoadAndAccumulateDirect(inputGm, inputOffset, currentElements, currentRows, accumulator);
        }
        CopyOutput(outputGm, outputOffset, currentElements, accumulator, vectorToMte3, mte3ToVector);
    }

    __aicore__ inline void CopyOutput(GlobalTensor<float>& outputGm, int64_t outputOffset, uint32_t currentElements,
                                      const LocalTensor<float>& accumulator, event_t vectorToMte3, event_t mte3ToVector)
    {
        SetFlag<HardEvent::V_MTE3>(vectorToMte3);
        WaitFlag<HardEvent::V_MTE3>(vectorToMte3);
        const DataCopyExtParams copyParams{1U, currentElements * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(outputGm[outputOffset], accumulator, copyParams);
        SetFlag<HardEvent::MTE3_V>(mte3ToVector);
        WaitFlag<HardEvent::MTE3_V>(mte3ToVector);
    }

    __aicore__ inline void LoadAndUpdateMagnitude(const GlobalTensor<float>& inputGm, int64_t inputOffset,
                                                  uint32_t currentElements, uint16_t currentRows,
                                                  const LocalTensor<float>& magnitude)
    {
        LocalTensor<float> input = LoadRows(inputGm, inputOffset, currentElements, currentRows);
        const uint32_t rowPitchElements = (currentElements + blockElements_ - 1U) / blockElements_ * blockElements_;
        asc_vf_call<&UpdateMagnitudeRows>(reinterpret_cast<const __ubuf__ float*>(input.GetPhyAddr()),
                                          reinterpret_cast<__ubuf__ float*>(magnitude.GetPhyAddr()), currentElements,
                                          rowPitchElements, currentRows);
        inputQueue_.FreeTensor(input);
    }

    __aicore__ inline void LoadAndAccumulateDirect(const GlobalTensor<float>& inputGm, int64_t inputOffset,
                                                   uint32_t currentElements, uint16_t currentRows,
                                                   const LocalTensor<float>& accumulator)
    {
        LocalTensor<float> input = LoadRows(inputGm, inputOffset, currentElements, currentRows);
        const uint32_t rowPitchElements = (currentElements + blockElements_ - 1U) / blockElements_ * blockElements_;
        asc_vf_call<&AccumulateDirectRows>(reinterpret_cast<const __ubuf__ float*>(input.GetPhyAddr()),
                                           reinterpret_cast<__ubuf__ float*>(accumulator.GetPhyAddr()), currentElements,
                                           rowPitchElements, currentRows);
        inputQueue_.FreeTensor(input);
    }

    __aicore__ inline void LoadAndAccumulateFast(const GlobalTensor<float>& inputGm, int64_t inputOffset,
                                                 uint32_t currentElements, uint16_t currentRows,
                                                 const LocalTensor<float>& accumulator,
                                                 const LocalTensor<float>& magnitude)
    {
        LocalTensor<float> input = LoadRows(inputGm, inputOffset, currentElements, currentRows);
        const uint32_t rowPitchElements = (currentElements + blockElements_ - 1U) / blockElements_ * blockElements_;
        asc_vf_call<&AccumulateFastRows>(reinterpret_cast<const __ubuf__ float*>(input.GetPhyAddr()),
                                         reinterpret_cast<__ubuf__ float*>(accumulator.GetPhyAddr()),
                                         reinterpret_cast<__ubuf__ float*>(magnitude.GetPhyAddr()), currentElements,
                                         rowPitchElements, currentRows, accumulatorStride_);
        inputQueue_.FreeTensor(input);
    }

    __aicore__ inline void LoadAndAccumulateExpansion(const GlobalTensor<float>& inputGm, int64_t inputOffset,
                                                      uint32_t currentElements, uint16_t currentRows,
                                                      const LocalTensor<float>& accumulator,
                                                      const LocalTensor<float>& lengths,
                                                      const LocalTensor<float>& magnitude)
    {
        LocalTensor<float> input = LoadRows(inputGm, inputOffset, currentElements, currentRows);
        const uint32_t rowPitchElements = (currentElements + blockElements_ - 1U) / blockElements_ * blockElements_;
        asc_vf_call<&AccumulateScaledRows>(reinterpret_cast<const __ubuf__ float*>(input.GetPhyAddr()),
                                           reinterpret_cast<__ubuf__ float*>(accumulator.GetPhyAddr()),
                                           reinterpret_cast<__ubuf__ float*>(lengths.GetPhyAddr()),
                                           reinterpret_cast<const __ubuf__ float*>(magnitude.GetPhyAddr()),
                                           currentElements, rowPitchElements, currentRows, accumulatorStride_,
                                           partialCount_, safeMagnitude_, inputScale_);
        inputQueue_.FreeTensor(input);
    }

    __aicore__ inline LocalTensor<float> LoadRows(const GlobalTensor<float>& inputGm, int64_t inputOffset,
                                                  uint32_t currentElements, uint16_t currentRows)
    {
        LocalTensor<float> input = inputQueue_.AllocTensor<float>();
        const uint32_t rowPitchElements = (currentElements + blockElements_ - 1U) / blockElements_ * blockElements_;
        const int64_t sourceStrideBytes = currentRows > 1U ? (outputElements_ - static_cast<int64_t>(currentElements)) *
                                                                 static_cast<int64_t>(sizeof(float)) :
                                                             0;
        const DataCopyExtParams copyParams{currentRows, currentElements * static_cast<uint32_t>(sizeof(float)),
                                           sourceStrideBytes, 0, 0};
        const uint8_t rightPadding = static_cast<uint8_t>(rowPitchElements - currentElements);
        const DataCopyPadExtParams<float> padParams{true, 0U, rightPadding, 0.0F};
        DataCopyPad(input, inputGm[inputOffset], copyParams, padParams);
        inputQueue_.EnQue(input);
        return inputQueue_.DeQue<float>();
    }

private:
    TPipe* pipe_{nullptr};
    TQue<QuePosition::VECIN, 1> inputQueue_;
    TBuf<TPosition::VECCALC> accumulatorBuffer_;
    TBuf<TPosition::VECCALC> lengthBuffer_;
    TBuf<TPosition::VECCALC> magnitudeBuffer_;

    GlobalTensor<float> resGammaGm_;
    GlobalTensor<float> resBetaGm_;
    GlobalTensor<float> pdGammaGm_;
    GlobalTensor<float> pdBetaGm_;

    int64_t reduceCount_{0};
    int64_t outputElements_{0};
    int64_t baseBlocksPerCore_{0};
    int64_t coreOffset_{0};
    int64_t coreElements_{0};
    uint32_t tileElements_{0};
    uint32_t accumulatorStride_{0};
    uint32_t partialCount_{1};
    uint32_t stateSlotCount_{1};
    uint32_t reduceRowsPerTile_{0};
    uint32_t extraBlockCoreCount_{0};
    uint32_t blockElements_{0};
    uint32_t usedCoreNum_{0};
    uint32_t blockIndex_{0};
    float safeMagnitude_{0.0F};
    float inputScale_{1.0F};
    float outputScale_{1.0F};
};
} // namespace NsINTrainingUpdateGradGammaBeta

#endif // IN_TRAINING_UPDATE_GRAD_GAMMA_BETA_BASE_H_
