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
 * \file embedding_bag_regbase_1d_sum.h
 * \brief embedding_bag_regbase_1d_sum
 */

#ifndef EMBEDDING_BAG_1D_SUM_H
#define EMBEDDING_BAG_1D_SUM_H

#include <cstdint>
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "embedding_bag_regbase_common.h"

namespace EmbeddingBag {

using namespace AscendC;
using namespace ops;

template <typename T, typename U, typename E, typename I>
class EmbeddingBagRegBaseIndx1dSum : public EmbeddingBagRegBaseIndx1dBase<T, U, E, I> {
public:
    __aicore__ inline EmbeddingBagRegBaseIndx1dSum(const EmbeddingBagTilingData& tilingData, TPipe& pipe)
        : pipe_(pipe), tiling_(tilingData){};

    __aicore__ inline void Init(GM_ADDR gmParam[TENSOR_COUNT])
    {
        this->weightGm_.SetGlobalBuffer((__gm__ T*)(gmParam[WEIGHT_INPUT_IDX]));
        this->indicesGm_.SetGlobalBuffer((__gm__ U*)(gmParam[INDICES_INPUT_IDX]));
        this->perSampleWeightsGm_.SetGlobalBuffer((__gm__ T*)(gmParam[PERSAMPWEIGHT_INPUT_IDX]));
        this->offsetsGm_.SetGlobalBuffer((__gm__ E*)(gmParam[OFFSETS_INPUT_IDX]));
        this->yGm_.SetGlobalBuffer((__gm__ T*)(gmParam[Y_OUTPUT_IDX]));
        this->offset2bagGm_.SetGlobalBuffer((__gm__ I*)(gmParam[OFFSET2BAG_OUTPUT_IDX]));
        this->bagSizeGm_.SetGlobalBuffer((__gm__ I*)(gmParam[BAGSIZE_OUTPUT_IDX]));
        if (GetBlockIdx() >= tiling_.usedCoreNum) {
            return;
        }
        this->weightCoreOfset_ = (GetBlockIdx() % tiling_.colTileNum) * tiling_.colNormalNum;
        this->offsetStart_ = GetBlockIdx() / tiling_.colTileNum * tiling_.rowNormalNum;
        this->curCoreBag_ =
            (GetBlockIdx() / tiling_.colTileNum == tiling_.rowTileNum - 1) ? tiling_.rowTailNum : tiling_.rowNormalNum;
        this->curCoreEmbedDim_ =
            (GetBlockIdx() % tiling_.colTileNum == tiling_.colTileNum - 1) ? tiling_.colTailNum : tiling_.colNormalNum;
        this->weightLoop_ = ops::CeilDiv(this->curCoreEmbedDim_, tiling_.weightDimFactor);
        this->tailLoopWeightNmber_ = this->curCoreEmbedDim_ - (this->weightLoop_ - 1) * tiling_.weightDimFactor;
        this->embeddingDim_ = tiling_.embeddingDim;
        this->weightDimFactor_ = tiling_.weightDimFactor;
        if (tiling_.isNeedSampleWeight == 1) {
            pipe_.InitBuffer(this->inQueueperSampleWeights_, 1, tiling_.indicesFactor * sizeof(T));
            pipe_.InitBuffer(this->perSampleWeightsBuf_, (tiling_.weightRowFactor) * sizeof(float));
        }
        pipe_.InitBuffer(this->inQueueWeight_, 1, tiling_.weightRowFactor * tiling_.weightDimFactor * sizeof(T));
        pipe_.InitBuffer(this->inQueueOffsets_, 1, (tiling_.offsetsFactor + 1) * sizeof(E));
        pipe_.InitBuffer(this->inQueueIndices_, 1, tiling_.indicesFactor * sizeof(U));
        pipe_.InitBuffer(this->outQueueY_, 1, tiling_.weightDimFactor * sizeof(T));
        pipe_.InitBuffer(this->outQueueBagSize_, 1, (tiling_.offsetsFactor + 1) * sizeof(I));
        pipe_.InitBuffer(this->outQueueOffset2bag_, 1, tiling_.weightRowFactor * sizeof(I));
    }

    __aicore__ inline void HandleBigBagSumNoPerSample(
        int64_t curBagIndiceNumber, int64_t bagIndiceStart, int64_t curOffsetStart, int64_t bagIdx,
        LocalTensor<I> bagSizeLocal)
    {
        int64_t indicesCopyLoop = ops::CeilDiv(curBagIndiceNumber, tiling_.indicesFactor);
        int64_t tailLoopIndicesCopyNumber = curBagIndiceNumber - (indicesCopyLoop - 1) * tiling_.indicesFactor;
        int64_t paddingCount = 0;
        int64_t weightOfset = this->weightCoreOfset_;
        int32_t eventIDMTE2ToS1 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS1);
        WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS1);
        LocalTensor<U> indicesLocal_ = this->inQueueIndices_.template DeQue<U>();
        for (uint64_t weightLoopIdx = 0; weightLoopIdx < this->weightLoop_; ++weightLoopIdx) {
            int32_t eventIDMTE3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
            int32_t eventIDMTE3ToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
            WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
            int64_t indiceStart = bagIndiceStart;
            int64_t validIndicesFactorNumber = curBagIndiceNumber;
            int64_t curWeightNumber =
                weightLoopIdx == this->weightLoop_ - 1 ? this->tailLoopWeightNmber_ : tiling_.weightDimFactor;
            LocalTensor<T> outYLocal_ = this->outQueueY_.template AllocTensor<T>();
            Duplicate(outYLocal_, static_cast<T>(0), static_cast<int32_t>(tiling_.weightDimFactor));
            for (uint64_t indicesCopyLoopIdx = 0; indicesCopyLoopIdx < indicesCopyLoop; ++indicesCopyLoopIdx) {
                int64_t curIndicesCopyNumber =
                    indicesCopyLoopIdx == indicesCopyLoop - 1 ? tailLoopIndicesCopyNumber : tiling_.indicesFactor;
                int64_t indicesLoop = ops::CeilDiv(curIndicesCopyNumber, tiling_.weightRowFactor);
                int64_t tailLoopIndicesNumber = curIndicesCopyNumber - (indicesLoop - 1) * tiling_.weightRowFactor;
                this->CopyIndicesFromGm(indiceStart, curIndicesCopyNumber);
                LocalTensor<U> indicesLocal_ = this->inQueueIndices_.template DeQue<U>();
                this->copyIndicesStart_ = indiceStart;
                this->copyIndicesEnd_ = this->copyIndicesStart_ + curIndicesCopyNumber;
                int32_t eventIDMTE2ToS1 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
                SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS1);
                WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS1);
                for (uint64_t indicesLoopIdx = 0; indicesLoopIdx < indicesLoop; ++indicesLoopIdx) {
                    int64_t curIndicesNumber =
                        indicesLoopIdx == indicesLoop - 1 ? tailLoopIndicesNumber : tiling_.weightRowFactor;
                    if (weightLoopIdx == 0) {
                        this->CopyOffset2bagToGm(indiceStart, curIndicesNumber, curOffsetStart);
                    }
                    int64_t validCopyIndices = curIndicesNumber;
                    int64_t weightLocalOffset = 0;
                    LocalTensor<T> weightLocal_ = this->inQueueWeight_.template AllocTensor<T>();
                    for (uint64_t indicesIdx = 0; indicesIdx < curIndicesNumber; ++indicesIdx) {
                        int64_t indiceLocalIdx = indiceStart + indicesIdx - this->copyIndicesStart_;
                        U weightIndex = indicesLocal_(indiceLocalIdx);
                        if (weightIndex == tiling_.paddingIdx) {
                            paddingCount = weightLoopIdx == 0 ? paddingCount + 1 : paddingCount;
                            validIndicesFactorNumber = validIndicesFactorNumber - 1;
                            validCopyIndices = validCopyIndices - 1;
                            continue;
                        }
                        int64_t weightOffset = weightIndex * tiling_.embeddingDim + weightOfset;
                        this->CopyWeightNoPerSampleFromGm(
                            weightOffset, curWeightNumber, weightLocal_[weightLocalOffset]);
                        weightLocalOffset = weightLocalOffset + tiling_.weightDimFactor;
                    }
                    this->ComputeAdd(curWeightNumber, (uint16_t)validCopyIndices, outYLocal_, weightLocal_);
                    this->inQueueWeight_.template FreeTensor(weightLocal_);
                    indiceStart = indiceStart + curIndicesNumber;
                }
                this->inQueueIndices_.template EnQue(indicesLocal_);
            }
            int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            int64_t outYOfset = curOffsetStart * tiling_.embeddingDim + weightOfset;
            this->CopyYToGm(outYOfset, curWeightNumber, outYLocal_);
            this->outQueueY_.template FreeTensor(outYLocal_);
            weightOfset = weightOfset + curWeightNumber;
        }
        bagSizeLocal(bagIdx) = bagSizeLocal(bagIdx) - paddingCount;
    }

    __aicore__ inline void HandleBagSumNoPerSample(
        int64_t curBagIndiceNumber, int64_t bagIndiceStart, int64_t curOffsetStart, int64_t bagIdx,
        LocalTensor<I> bagSizeLocal)
    {
        int64_t indicesLoop = ops::CeilDiv(curBagIndiceNumber, tiling_.weightRowFactor);
        int64_t tailLoopIndicesNumber = curBagIndiceNumber - (indicesLoop - 1) * tiling_.weightRowFactor;
        int64_t weightOfset = this->weightCoreOfset_;
        int64_t paddingCount = 0;
        LocalTensor<U> indicesLocal_ = this->inQueueIndices_.template DeQue<U>();
        int32_t eventIDMTE2ToS1 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS1);
        WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS1);
        for (uint64_t weightLoopIdx = 0; weightLoopIdx < this->weightLoop_; ++weightLoopIdx) {
            int32_t eventIDMTE3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
            int64_t curWeightNumber =
                weightLoopIdx == this->weightLoop_ - 1 ? this->tailLoopWeightNmber_ : tiling_.weightDimFactor;
            LocalTensor<T> outYLocal_ = this->outQueueY_.template AllocTensor<T>();
            Duplicate(outYLocal_, static_cast<T>(0), static_cast<int32_t>(tiling_.weightDimFactor));
            int32_t eventIDMTE3ToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
            WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
            int64_t validIndicesFactorNumber = curBagIndiceNumber;
            int64_t indiceStart = bagIndiceStart;
            for (uint64_t indicesLoopIdx = 0; indicesLoopIdx < indicesLoop; ++indicesLoopIdx) {
                int64_t curIndicesNumber =
                    indicesLoopIdx == indicesLoop - 1 ? tailLoopIndicesNumber : tiling_.indicesFactor;
                // Offset2bagweight
                if (weightLoopIdx == 0) {
                    this->CopyOffset2bagToGm(indiceStart, curIndicesNumber, curOffsetStart);
                }
                int64_t weightLocalOffset = 0;
                LocalTensor<T> weightLocal_ = this->inQueueWeight_.template AllocTensor<T>();
                for (uint64_t indicesIdx = 0; indicesIdx < curIndicesNumber; ++indicesIdx) {
                    int64_t indiceLocalIdx = indiceStart + indicesIdx - this->copyIndicesStart_;
                    U weightIndex = indicesLocal_(indiceLocalIdx);
                    if (weightIndex == tiling_.paddingIdx) {
                        paddingCount = weightLoopIdx == 0 ? paddingCount + 1 : paddingCount;
                        validIndicesFactorNumber = validIndicesFactorNumber - 1;
                        continue;
                    }
                    int64_t weightOffset = weightIndex * tiling_.embeddingDim + weightOfset;
                    this->CopyWeightNoPerSampleFromGm(weightOffset, curWeightNumber, weightLocal_[weightLocalOffset]);
                    weightLocalOffset = weightLocalOffset + tiling_.weightDimFactor;
                }
                this->ComputeAdd(curWeightNumber, (uint16_t)validIndicesFactorNumber, outYLocal_, weightLocal_);
                this->inQueueWeight_.template FreeTensor(weightLocal_);
                indiceStart = indiceStart + curIndicesNumber;
            }
            int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            int64_t outYOfset = curOffsetStart * tiling_.embeddingDim + weightOfset;
            this->CopyYToGm(outYOfset, curWeightNumber, outYLocal_);
            this->outQueueY_.template FreeTensor(outYLocal_);
            weightOfset = weightOfset + curWeightNumber;
        }
        bagSizeLocal(bagIdx) = bagSizeLocal(bagIdx) - paddingCount;
    }

    __aicore__ inline void ProcessNoPerSample()
    {
        LocalTensor<U> indicesLocal = this->inQueueIndices_.template AllocTensor<U>();
        int64_t curOffsetStart = this->offsetStart_;
        int64_t outBagSizeStart = this->offsetStart_;
        this->inQueueIndices_.template EnQue(indicesLocal);
        int64_t offsetsLoop = ops::CeilDiv(this->curCoreBag_, tiling_.offsetsFactor);
        int64_t tailOffsetsFactor = this->curCoreBag_ - (offsetsLoop - 1) * tiling_.offsetsFactor;
        for (uint64_t offsetsLoopIdx = 0; offsetsLoopIdx < offsetsLoop; ++offsetsLoopIdx) {
            int64_t curLoopOffsetsNumber =
                offsetsLoopIdx == offsetsLoop - 1 ? tailOffsetsFactor : tiling_.offsetsFactor;
            if (GetBlockIdx() / tiling_.colTileNum == tiling_.rowTileNum - 1 && offsetsLoopIdx == offsetsLoop - 1) {
                this->CopyOffsetsFromGm(curOffsetStart, curLoopOffsetsNumber);
                LocalTensor<E> offsetsLocal_ = this->inQueueOffsets_.template DeQue<E>();
                int32_t eventIDMTE2ToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
                SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
                WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
                offsetsLocal_(curLoopOffsetsNumber) = tiling_.indicesNumel;
                this->inQueueOffsets_.template EnQue(offsetsLocal_);
            } else {
                this->CopyOffsetsFromGm(curOffsetStart, curLoopOffsetsNumber + 1);
            }
            LocalTensor<E> offsetsLocal = this->inQueueOffsets_.template DeQue<E>();
            this->ComputeBagSize(curLoopOffsetsNumber + 1, offsetsLocal);
            LocalTensor<I> bagSizeLocal = this->outQueueBagSize_.template DeQue<I>();
            for (uint64_t bagIdx = 0; bagIdx < curLoopOffsetsNumber; ++bagIdx) {
                int64_t curBagIndiceNumber = (int64_t)bagSizeLocal(bagIdx);
                if (curBagIndiceNumber == 0) {
                    uint64_t dimNumber = (uint64_t)this->curCoreEmbedDim_;
                    int64_t outYofset = curOffsetStart * tiling_.embeddingDim + this->weightCoreOfset_;
                    GlobalTensor<T> yOutGm_ = this->yGm_[outYofset];
                    InitGlobalMemory(yOutGm_, dimNumber, (T)(-1));
                    curOffsetStart = curOffsetStart + 1;
                    continue;
                }
                int64_t indiceStart = offsetsLocal(bagIdx);
                if (indiceStart >= this->copyIndicesStart_ &&
                    indiceStart + curBagIndiceNumber < this->copyIndicesEnd_) {
                    HandleBagSumNoPerSample(curBagIndiceNumber, indiceStart, curOffsetStart, bagIdx, bagSizeLocal);
                } else if (curBagIndiceNumber > tiling_.indicesFactor) {
                    HandleBigBagSumNoPerSample(curBagIndiceNumber, indiceStart, curOffsetStart, bagIdx, bagSizeLocal);
                } else {
                    this->CopyIndicesFromGm(indiceStart, tiling_.indicesFactor);
                    this->copyIndicesStart_ = indiceStart;
                    this->copyIndicesEnd_ = indiceStart + tiling_.indicesFactor;
                    HandleBagSumNoPerSample(curBagIndiceNumber, indiceStart, curOffsetStart, bagIdx, bagSizeLocal);
                }
                curOffsetStart = curOffsetStart + 1;
            }
            int32_t eventIDSToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            this->inQueueOffsets_.template FreeTensor(offsetsLocal);
            DataCopyExtParams copyParam{1, static_cast<uint32_t>(curLoopOffsetsNumber * sizeof(I)), 0, 0, 0};
            DataCopyPad(this->bagSizeGm_[outBagSizeStart], bagSizeLocal, copyParam);
            this->outQueueBagSize_.template FreeTensor(bagSizeLocal);
            outBagSizeStart = outBagSizeStart + curLoopOffsetsNumber;
        }
        LocalTensor<U> indicesLocal_ = this->inQueueIndices_.template DeQue<U>();
        this->inQueueIndices_.template FreeTensor(indicesLocal_);
    }

    __aicore__ inline void HandleBagSumPerSample(
        int64_t curBagIndiceNumber, int64_t bagIndiceStart, int64_t curOffsetStart, int64_t bagIdx,
        LocalTensor<I> bagSizeLocal)
    {
        int64_t indicesLoop = ops::CeilDiv(curBagIndiceNumber, tiling_.weightRowFactor);
        int64_t tailLoopIndicesNumber = curBagIndiceNumber - (indicesLoop - 1) * tiling_.weightRowFactor;
        int64_t paddingCount = 0;
        int64_t weightOfset = this->weightCoreOfset_;
        int32_t eventIDMTE2ToS1 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS1);
        WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS1);
        LocalTensor<U> indicesLocal_ = this->inQueueIndices_.template DeQue<U>();
        LocalTensor<T> perSampleWeightsLocal_ = this->inQueueperSampleWeights_.template DeQue<T>();
        for (uint64_t weightLoopIdx = 0; weightLoopIdx < this->weightLoop_; ++weightLoopIdx) {
            int32_t eventIDMTE3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
            int64_t curWeightNumber =
                weightLoopIdx == this->weightLoop_ - 1 ? this->tailLoopWeightNmber_ : tiling_.weightDimFactor;
            LocalTensor<T> outYLocal_ = this->outQueueY_.template AllocTensor<T>();
            Duplicate(outYLocal_, static_cast<T>(0), static_cast<int32_t>(tiling_.weightDimFactor));
            int64_t validIndicesFactorNumber = curBagIndiceNumber;
            int64_t indiceStart = bagIndiceStart;
            int32_t eventIDMTE3ToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
            WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
            for (uint64_t indicesLoopIdx = 0; indicesLoopIdx < indicesLoop; ++indicesLoopIdx) {
                int64_t curIndicesNumber =
                    indicesLoopIdx == indicesLoop - 1 ? tailLoopIndicesNumber : tiling_.indicesFactor;
                // Offset2bagweight
                if (weightLoopIdx == 0) {
                    this->CopyOffset2bagToGm(indiceStart, curIndicesNumber, curOffsetStart);
                }
                int64_t indicesIndex = 0;
                int64_t weightLocalOffset = 0;
                LocalTensor<T> weightLocal_ = this->inQueueWeight_.template AllocTensor<T>();
                LocalTensor<float> perSampleWeightCountLocal = this->perSampleWeightsBuf_.template Get<float>();
                for (uint64_t indicesIdx = 0; indicesIdx < curIndicesNumber; ++indicesIdx) {
                    int64_t indiceLocalIdx = indiceStart + indicesIdx - this->copyIndicesStart_;
                    U weightIndex = indicesLocal_(indiceLocalIdx);
                    if (weightIndex == tiling_.paddingIdx) {
                        paddingCount = weightLoopIdx == 0 ? paddingCount + 1 : paddingCount;
                        validIndicesFactorNumber = validIndicesFactorNumber - 1;
                        continue;
                    }
                    if (indiceStart + indicesIdx <= tiling_.sampleWeightNum - 1) {
                        T perSample = perSampleWeightsLocal_(indiceLocalIdx);
                        if constexpr (IsSameType<T, float>::value) {
                            perSampleWeightCountLocal(indicesIndex) = perSample;
                        } else if constexpr (IsSameType<T, half>::value) {
                            perSampleWeightCountLocal(indicesIndex) = static_cast<float>(perSample);
                        } else {
                            perSampleWeightCountLocal(indicesIndex) = ToFloat(perSample);
                        }
                    } else {
                        perSampleWeightCountLocal(indicesIndex) = static_cast<float>(0);
                    }
                    int64_t weightOffset = weightIndex * tiling_.embeddingDim + weightOfset;
                    this->CopyWeightNoPerSampleFromGm(weightOffset, curWeightNumber, weightLocal_[weightLocalOffset]);
                    indicesIndex = indicesIndex + 1;
                    weightLocalOffset = weightLocalOffset + tiling_.weightDimFactor;
                }
                this->ComputeAddPerSample(
                    curWeightNumber, (uint16_t)validIndicesFactorNumber, outYLocal_, weightLocal_,
                    perSampleWeightCountLocal);
                this->inQueueWeight_.template FreeTensor(weightLocal_);
                indiceStart = indiceStart + curIndicesNumber;
            }
            int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            int64_t outYOfset = curOffsetStart * tiling_.embeddingDim + weightOfset;
            this->CopyYToGm(outYOfset, curWeightNumber, outYLocal_);
            weightOfset = weightOfset + curWeightNumber;
            this->outQueueY_.template FreeTensor(outYLocal_);
        }
        bagSizeLocal(bagIdx) = bagSizeLocal(bagIdx) - paddingCount;
    }

    __aicore__ inline void HandleBigBagSumPerSample(
        int64_t curBagIndiceNumber, int64_t bagIndiceStart, int64_t curOffsetStart, int64_t bagIdx,
        LocalTensor<I> bagSizeLocal)
    {
        int64_t indicesCopyLoop = ops::CeilDiv(curBagIndiceNumber, tiling_.indicesFactor);
        int64_t tailLoopIndicesCopyNumber = curBagIndiceNumber - (indicesCopyLoop - 1) * tiling_.indicesFactor;
        int64_t weightOfset = this->weightCoreOfset_;
        int64_t paddingCount = 0;
        int32_t eventIDMTE2ToS1 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS1);
        WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS1);
        for (uint64_t weightLoopIdx = 0; weightLoopIdx < this->weightLoop_; ++weightLoopIdx) {
            int64_t curWeightNumber =
                weightLoopIdx == this->weightLoop_ - 1 ? this->tailLoopWeightNmber_ : tiling_.weightDimFactor;
            int64_t indiceStart = bagIndiceStart;
            int64_t validIndicesFactorNumber = curBagIndiceNumber;
            int32_t eventIDMTE3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
            LocalTensor<T> outYLocal_ = this->outQueueY_.template AllocTensor<T>();
            Duplicate(outYLocal_, static_cast<T>(0), static_cast<int32_t>(tiling_.weightDimFactor));
            int32_t eventIDMTE3ToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
            WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
            for (uint64_t indicesCopyLoopIdx = 0; indicesCopyLoopIdx < indicesCopyLoop; ++indicesCopyLoopIdx) {
                int64_t curIndicesCopyNumber =
                    indicesCopyLoopIdx == indicesCopyLoop - 1 ? tailLoopIndicesCopyNumber : tiling_.indicesFactor;
                int64_t indicesLoop = ops::CeilDiv(curIndicesCopyNumber, tiling_.weightRowFactor);
                int64_t tailLoopIndicesNumber = curIndicesCopyNumber - (indicesLoop - 1) * tiling_.weightRowFactor;
                this->CopyIndicesFromGm(indiceStart, curIndicesCopyNumber);
                this->CopyPerSampleWeightsFromGm(indiceStart, curIndicesCopyNumber);
                this->copyIndicesStart_ = indiceStart;
                this->copyIndicesEnd_ = this->copyIndicesStart_ + curIndicesCopyNumber;
                LocalTensor<U> indicesLocal_ = this->inQueueIndices_.template DeQue<U>();
                LocalTensor<T> perSampleWeightsLocal_ = this->inQueueperSampleWeights_.template DeQue<T>();
                int32_t eventIDMTE2ToS1 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
                SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS1);
                WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS1);
                for (uint64_t indicesLoopIdx = 0; indicesLoopIdx < indicesLoop; ++indicesLoopIdx) {
                    int64_t curIndicesNumber =
                        indicesLoopIdx == indicesLoop - 1 ? tailLoopIndicesNumber : tiling_.weightRowFactor;
                    // Offset2bagweight
                    if (weightLoopIdx == 0) {
                        this->CopyOffset2bagToGm(indiceStart, curIndicesNumber, curOffsetStart);
                    }
                    int64_t weightLocalOffset = 0;
                    int64_t validCopyIndices = curIndicesNumber;
                    int64_t indicesIndex = 0;
                    LocalTensor<float> perSampleWeightCountLocal = this->perSampleWeightsBuf_.template Get<float>();
                    LocalTensor<T> weightLocal_ = this->inQueueWeight_.template AllocTensor<T>();
                    for (uint64_t indicesIdx = 0; indicesIdx < curIndicesNumber; ++indicesIdx) {
                        int64_t indiceLocalIdx = indiceStart + indicesIdx - this->copyIndicesStart_;
                        U weightIndex = indicesLocal_(indiceLocalIdx);
                        if (weightIndex == tiling_.paddingIdx) {
                            validIndicesFactorNumber = validIndicesFactorNumber - 1;
                            paddingCount = weightLoopIdx == 0 ? paddingCount + 1 : paddingCount;
                            validCopyIndices = validCopyIndices - 1;
                            continue;
                        }
                        if (tiling_.sampleWeightNum - 1 >= indiceStart + indicesIdx) {
                            T perSample = perSampleWeightsLocal_(indiceLocalIdx);
                            if constexpr (IsSameType<T, float>::value) {
                                perSampleWeightCountLocal(indicesIndex) = perSample;
                            } else if constexpr (IsSameType<T, half>::value) {
                                perSampleWeightCountLocal(indicesIndex) = static_cast<float>(perSample);
                            } else {
                                perSampleWeightCountLocal(indicesIndex) = ToFloat(perSample);
                            }
                        } else {
                            perSampleWeightCountLocal(indicesIndex) = 0;
                        }
                        indicesIndex = indicesIndex + 1;
                        int64_t weightOffset = weightIndex * tiling_.embeddingDim + weightOfset;

                        this->CopyWeightNoPerSampleFromGm(
                            weightOffset, curWeightNumber, weightLocal_[weightLocalOffset]);
                        weightLocalOffset = weightLocalOffset + tiling_.weightDimFactor;
                    }
                    this->ComputeAddPerSample(
                        curWeightNumber, (uint16_t)validCopyIndices, outYLocal_, weightLocal_,
                        perSampleWeightCountLocal);
                    this->inQueueWeight_.template FreeTensor(weightLocal_);
                    indiceStart = indiceStart + curIndicesNumber;
                }
                this->inQueueIndices_.template EnQue(indicesLocal_);
                this->inQueueperSampleWeights_.template EnQue(perSampleWeightsLocal_);
            }
            int64_t outYOfset = curOffsetStart * tiling_.embeddingDim + weightOfset;
            int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            this->CopyYToGm(outYOfset, curWeightNumber, outYLocal_);
            this->outQueueY_.template FreeTensor(outYLocal_);
            weightOfset = weightOfset + curWeightNumber;
        }
        bagSizeLocal(bagIdx) = bagSizeLocal(bagIdx) - paddingCount;
    }

    __aicore__ inline void ProcessPerSample()
    {
        int64_t curOffsetStart = this->offsetStart_;
        int64_t outBagSizeStart = this->offsetStart_;
        int64_t offsetsLoop = ops::CeilDiv(this->curCoreBag_, tiling_.offsetsFactor);
        int64_t tailOffsetsFactor = this->curCoreBag_ - (offsetsLoop - 1) * tiling_.offsetsFactor;
        LocalTensor<U> indicesLocal = this->inQueueIndices_.template AllocTensor<U>();
        this->inQueueIndices_.template EnQue(indicesLocal);
        LocalTensor<T> perSampleWeightsLocal = this->inQueueperSampleWeights_.template AllocTensor<T>();
        this->inQueueperSampleWeights_.template EnQue(perSampleWeightsLocal);
        for (uint64_t offsetsLoopIdx = 0; offsetsLoopIdx < offsetsLoop; ++offsetsLoopIdx) {
            int64_t curLoopOffsetsNumber =
                offsetsLoopIdx == offsetsLoop - 1 ? tailOffsetsFactor : tiling_.offsetsFactor;
            // copy offset indices
            if (GetBlockIdx() / tiling_.colTileNum == tiling_.rowTileNum - 1 && offsetsLoopIdx == offsetsLoop - 1) {
                this->CopyOffsetsFromGm(curOffsetStart, curLoopOffsetsNumber);
                int32_t eventIDMTE2ToS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
                SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
                WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
                LocalTensor<E> offsetsLocal_ = this->inQueueOffsets_.template DeQue<E>();
                offsetsLocal_(curLoopOffsetsNumber) = static_cast<E>(tiling_.indicesNumel);
                this->inQueueOffsets_.template EnQue(offsetsLocal_);
            } else {
                this->CopyOffsetsFromGm(curOffsetStart, curLoopOffsetsNumber + 1);
            }
            /* compute bagsize */
            LocalTensor<E> offsetsLocal = this->inQueueOffsets_.template DeQue<E>();
            int64_t computeLen = curLoopOffsetsNumber + 1;
            this->ComputeBagSize(computeLen, offsetsLocal);
            LocalTensor<I> bagSizeLocal = this->outQueueBagSize_.template DeQue<I>();
            for (uint64_t bagIdx = 0; bagIdx < curLoopOffsetsNumber; ++bagIdx) {
                int64_t curBagIndiceNumber = (int64_t)bagSizeLocal(bagIdx);
                /* bag size is 0 */
                if (curBagIndiceNumber == 0) {
                    int64_t outYofset = curOffsetStart * tiling_.embeddingDim + this->weightCoreOfset_;
                    uint64_t dimNumbers = (uint64_t)this->curCoreEmbedDim_;
                    GlobalTensor<T> yOutGm_ = this->yGm_[outYofset];
                    InitGlobalMemory(yOutGm_, dimNumbers, (T)(-1));
                    curOffsetStart = curOffsetStart + 1;
                    continue;
                }
                int64_t indiceStart = offsetsLocal(bagIdx);
                if (indiceStart >= this->copyIndicesStart_ &&
                    indiceStart + curBagIndiceNumber < this->copyIndicesEnd_) {
                    HandleBagSumPerSample(curBagIndiceNumber, indiceStart, curOffsetStart, bagIdx, bagSizeLocal);
                } else if (curBagIndiceNumber > tiling_.indicesFactor) {
                    HandleBigBagSumPerSample(curBagIndiceNumber, indiceStart, curOffsetStart, bagIdx, bagSizeLocal);
                } else {
                    this->CopyIndicesFromGm(indiceStart, tiling_.indicesFactor);
                    int64_t sampleWeightsDataLen = tiling_.indicesFactor;
                    if (indiceStart + curBagIndiceNumber > tiling_.sampleWeightNum) {
                        sampleWeightsDataLen = tiling_.sampleWeightNum - indiceStart;
                    }
                    this->CopyPerSampleWeightsFromGm(indiceStart, sampleWeightsDataLen);
                    this->copyIndicesStart_ = indiceStart;
                    this->copyIndicesEnd_ = indiceStart + tiling_.indicesFactor;
                    HandleBagSumPerSample(curBagIndiceNumber, indiceStart, curOffsetStart, bagIdx, bagSizeLocal);
                }
                curOffsetStart = curOffsetStart + 1;
            }
            int32_t eventIDSToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            DataCopyExtParams copyParam{1, static_cast<uint32_t>(curLoopOffsetsNumber * sizeof(I)), 0, 0, 0};
            DataCopyPad(this->bagSizeGm_[outBagSizeStart], bagSizeLocal, copyParam);
            outBagSizeStart = outBagSizeStart + curLoopOffsetsNumber;
            this->outQueueBagSize_.template FreeTensor(bagSizeLocal);
            this->inQueueOffsets_.template FreeTensor(offsetsLocal);
        }
        LocalTensor<U> indicesLocal_ = this->inQueueIndices_.template DeQue<U>();
        this->inQueueIndices_.template FreeTensor(indicesLocal_);
        LocalTensor<T> perSampleWeightsLocal_ = this->inQueueperSampleWeights_.template DeQue<T>();
        this->inQueueperSampleWeights_.template FreeTensor(perSampleWeightsLocal_);
    }

    __aicore__ inline void Process()
    {
        if (tiling_.nBags <= 0 || GetBlockIdx() >= tiling_.usedCoreNum) {
            return;
        }
        if (tiling_.isNeedSampleWeight == 1) {
            ProcessPerSample();
        } else {
            ProcessNoPerSample();
        }
    }

private:
    TPipe& pipe_;
    const EmbeddingBagTilingData& tiling_;
};
} // namespace EmbeddingBag
#endif
;