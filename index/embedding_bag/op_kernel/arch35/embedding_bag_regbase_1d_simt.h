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
 * \file embedding_bag_regbase_1d_simt.h
 * \brief
 */

#ifndef EMBEDDING_BAG_REGBASE_1D_SIMT_H
#define EMBEDDING_BAG_REGBASE_1D_SIMT_H

#include "kernel_operator.h"
#include "../inc/kernel_utils.h"
#include "../inc/platform.h"
#include "embedding_bag_regbase_common.h"

namespace EmbeddingBag {
using namespace AscendC;

template <typename W, typename I,  typename O, typename P, typename COMP_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD_NUM) inline void SimtComputeSumWithPerSample1D(
    __gm__ W* weights, __gm__ I* indices, __gm__ O* offsets, __gm__ W* perSampleWeights,
    __gm__ W* y, __gm__ P* offset2bag, __gm__ P* bagSize,
    COMP_T numBags, COMP_T numIndices,  COMP_T chunkPerBag, COMP_T magic, COMP_T shift,
    COMP_T embeddingDimSize, COMP_T paddingIdx)
{
    COMP_T chunkStride = Simt::GetBlockNum() * Simt::GetThreadNum<1>();
    COMP_T numChunks = numBags * chunkPerBag;
    COMP_T chunkOffset = Simt::GetBlockIdx() * Simt::GetThreadNum<1>() + Simt::GetThreadIdx<1>();

    for (COMP_T chunk = chunkOffset; chunk < numChunks; chunk += chunkStride) {
        COMP_T quotient = Simt::UintDiv(chunk, magic, shift);
        COMP_T reminder = chunk - quotient * chunkPerBag;
        COMP_T featureDim = Simt::GetThreadIdx<0>() + reminder * Simt::GetThreadNum<0>();
        if (featureDim < embeddingDimSize) {
            COMP_T bag = Simt::UintDiv(chunk, magic, shift);
            COMP_T eachBagSize = 0;
            COMP_T begin = (bag == 0 ? 0 : offsets[bag]);
            COMP_T end = (bag < numBags - 1 ? offsets[bag + 1] : numIndices);
            float weightSum = 0;
            for (COMP_T embedIdx = begin; embedIdx < end; embedIdx++) {
                bool pad = (indices[embedIdx] == paddingIdx);
                COMP_T weightOffset = indices[embedIdx] * embeddingDimSize;
                W weightValue = weights[featureDim + weightOffset];
                float scaleWeightBy = static_cast<float>(perSampleWeights[embedIdx]);
                if (featureDim == 0) {
                    offset2bag[embedIdx] = static_cast<P>(bag);
                }
                weightValue = pad ? static_cast<W>(0) : weightValue;
                weightSum += scaleWeightBy * static_cast<float>(weightValue);
                eachBagSize += pad ? 0 : 1;
            }
            y[bag * embeddingDimSize + featureDim] = static_cast<W>(weightSum);
            bagSize[bag] = static_cast<P>(eachBagSize);
        }
    }
}

template <typename W, typename I,  typename O, typename P, typename COMP_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD_NUM) inline void SimtComputeSum1D(
    __gm__ W* weights, __gm__ I* indices, __gm__ O* offsets,
    __gm__ W* y, __gm__ P* offset2bag, __gm__ P* bagSize,
    COMP_T numBags, COMP_T numIndices,  COMP_T chunkPerBag, COMP_T magic, COMP_T shift,
    COMP_T embeddingDimSize, COMP_T paddingIdx)
{
    COMP_T chunkStride = Simt::GetBlockNum() * Simt::GetThreadNum<1>();
    COMP_T chunkOffset = Simt::GetBlockIdx() * Simt::GetThreadNum<1>() + Simt::GetThreadIdx<1>();
    COMP_T numChunks = chunkPerBag * numBags;
    for (COMP_T chunk = chunkOffset; chunk < numChunks; chunk += chunkStride) {
        COMP_T quotient = Simt::UintDiv(chunk, magic, shift);
        COMP_T reminder = chunk - quotient * chunkPerBag;
        COMP_T featureDim = reminder * Simt::GetThreadNum<0>() + Simt::GetThreadIdx<0>();
        if (featureDim < embeddingDimSize) {
            float weightSum = 0;
            COMP_T eachBagSize = 0;
            COMP_T bag = Simt::UintDiv(chunk, magic, shift);
            COMP_T begin = (bag == 0 ? 0 : offsets[bag]);
            COMP_T end = (bag < numBags - 1 ? offsets[bag + 1] : numIndices);
            for (COMP_T embedIdx = begin; embedIdx < end; embedIdx++) {
                bool pad = (indices[embedIdx] == paddingIdx);
                COMP_T weightOffset = indices[embedIdx] * embeddingDimSize;
                W weightValue = weights[weightOffset + featureDim];
                weightValue = pad ? static_cast<W>(0) : weightValue;
                weightSum += static_cast<float>(weightValue);
                eachBagSize += pad ? 0 : 1;
                if (featureDim == 0) {
                    offset2bag[embedIdx] = static_cast<P>(bag);
                }
            }
            bagSize[bag] = static_cast<P>(eachBagSize);
            y[bag * embeddingDimSize + featureDim] = static_cast<W>(weightSum);
        }
    }
}

template <typename W, typename I,  typename O, typename P, typename COMP_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD_NUM) inline void SimtComputeMean1D(
    __gm__ W* weights, __gm__ I* indices, __gm__ O* offsets,
    __gm__ W* y, __gm__ P* offset2bag, __gm__ P* bagSize,
    COMP_T numBags, COMP_T numIndices,  COMP_T chunkPerBag, COMP_T magic, COMP_T shift,
    COMP_T embeddingDimSize, COMP_T paddingIdx)
{
    COMP_T chunkOffset = Simt::GetBlockIdx() * Simt::GetThreadNum<1>() + Simt::GetThreadIdx<1>();
    COMP_T chunkStride = Simt::GetThreadNum<1>() * Simt::GetBlockNum();
    COMP_T numChunks = numBags * chunkPerBag;
    for (COMP_T chunk = chunkOffset; chunk < numChunks; chunk += chunkStride) {
        COMP_T quotient = Simt::UintDiv(chunk, magic, shift);
        COMP_T reminder = chunk - quotient * chunkPerBag;
        COMP_T featureDim = reminder * Simt::GetThreadNum<0>() + Simt::GetThreadIdx<0>();
        if (featureDim < embeddingDimSize) {
            COMP_T bag = Simt::UintDiv(chunk, magic, shift);
            COMP_T begin = (bag == 0 ? 0 : offsets[bag]);
            COMP_T end = (bag < numBags - 1 ? offsets[bag + 1] : numIndices);
            COMP_T eachBagSize = 0;
            float weightSum = 0;
            for (COMP_T embedIdx = begin; embedIdx < end; embedIdx++) {
                bool pad = (indices[embedIdx] == paddingIdx);
                COMP_T weightOffset = indices[embedIdx] * embeddingDimSize;
                eachBagSize += pad ? 0 : 1;
                W weightValue = weights[weightOffset + featureDim];
                weightValue = pad ? static_cast<W>(0) : weightValue;
                weightSum += static_cast<float>(weightValue);
                if (featureDim == 0) {
                    offset2bag[embedIdx] = static_cast<P>(bag);
                }
            }
            if (eachBagSize != 0) {
                weightSum = weightSum / static_cast<float>(eachBagSize);
            }
            bagSize[bag] = static_cast<P>(eachBagSize);
            y[bag * embeddingDimSize + featureDim] = static_cast<W>(weightSum);
        }
    }
}

template <typename W, typename I,  typename O, typename P, typename COMP_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD_NUM) inline void SimtComputeMax1D(
    __gm__ W* weights, __gm__ I* indices, __gm__ O* offsets,
    __gm__ W* y, __gm__ P* offset2bag, __gm__ P* bagSize, __gm__ P* maxIndices,
    COMP_T numBags, COMP_T numIndices,  COMP_T chunkPerBag, COMP_T magic, COMP_T shift,
    COMP_T embeddingDimSize, COMP_T paddingIdx)
{
    COMP_T numChunks = chunkPerBag * numBags;
    COMP_T chunkOffset = Simt::GetBlockIdx() * Simt::GetThreadNum<1>() + Simt::GetThreadIdx<1>();
    COMP_T chunkStride = Simt::GetBlockNum() * Simt::GetThreadNum<1>();

    for (COMP_T chunk = chunkOffset; chunk < numChunks; chunk += chunkStride) {
        COMP_T quotient = Simt::UintDiv(chunk, magic, shift);
        COMP_T reminder = chunk - quotient * chunkPerBag;
        COMP_T featureDim = reminder * Simt::GetThreadNum<0>() + Simt::GetThreadIdx<0>();
        if (featureDim < embeddingDimSize) {
            COMP_T bag = Simt::UintDiv(chunk, magic, shift);
            COMP_T begin = (bag == 0 ? 0 : offsets[bag]);
            COMP_T end = (bag < numBags - 1 ? offsets[bag + 1] : numIndices);
            COMP_T eachBagSize = 0;
            P maxIdx = static_cast<P>(-1);
            W maxVal = 0;
            for (COMP_T embedIdx = begin; embedIdx < end; embedIdx++) {
                bool pad = (indices[embedIdx] == paddingIdx);
                COMP_T weightOffset = embeddingDimSize * indices[embedIdx];
                W weightValue = weights[weightOffset + featureDim];
                weightValue = pad ? static_cast<W>(0) : weightValue;
                if (featureDim == 0) {
                    offset2bag[embedIdx] = static_cast<P>(bag);
                }
                if (eachBagSize == 0 || weightValue > maxVal) {
                    if (!pad) {
                        maxIdx = static_cast<P>(indices[embedIdx]);
                        maxVal = weightValue;
                    }
                }
                eachBagSize += pad ? 0 : 1;
            }
            COMP_T outPutOffset = bag * embeddingDimSize + featureDim;
            bagSize[bag] = static_cast<P>(eachBagSize);
            y[outPutOffset] = maxVal;
            maxIndices[outPutOffset] = maxIdx;
        }
    }
}

template<typename W, typename I,  typename O, typename P, typename COMP_T>
class EmbeddingBagRegBaseSimt1D {
public:
    __aicore__ inline EmbeddingBagRegBaseSimt1D(const EmbeddingBagTilingData &tilingData, TPipe &pipe) :
        tilingData_(tilingData), pipe_(pipe) {};
    __aicore__ inline void Init(GM_ADDR gmParam[TENSOR_COUNT]);
    __aicore__ inline void Process();

private:
    TPipe& pipe_;
    GlobalTensor<W> weightGm_;
    GlobalTensor<I> indicesGm_;
    GlobalTensor<O> offsetsGm_;
    GlobalTensor<W> perSampleWeightsGm_;
    GlobalTensor<W> yGm_;
    GlobalTensor<P> offset2bagGm_;
    GlobalTensor<P> bagSizeGm_;
    GlobalTensor<P> maxIndicesGm_;
    const EmbeddingBagTilingData& tilingData_;
};

template<typename W, typename I,  typename O, typename P, typename COMP_T>
__aicore__ inline void EmbeddingBagRegBaseSimt1D<W, I, O, P, COMP_T>::Init(GM_ADDR gmParam[TENSOR_COUNT])
{
    weightGm_.SetGlobalBuffer((__gm__ W*)(gmParam[WEIGHT_INPUT_IDX]));
    indicesGm_.SetGlobalBuffer((__gm__ I*)(gmParam[INDICES_INPUT_IDX]));
    offsetsGm_.SetGlobalBuffer((__gm__ O*)(gmParam[OFFSETS_INPUT_IDX]));
    perSampleWeightsGm_.SetGlobalBuffer((__gm__ W*)(gmParam[PERSAMPWEIGHT_INPUT_IDX]));
    yGm_.SetGlobalBuffer((__gm__ W*)(gmParam[Y_OUTPUT_IDX]));
    offset2bagGm_.SetGlobalBuffer((__gm__ P*)(gmParam[OFFSET2BAG_OUTPUT_IDX]));
    bagSizeGm_.SetGlobalBuffer((__gm__ P*)(gmParam[BAGSIZE_OUTPUT_IDX]));
    maxIndicesGm_.SetGlobalBuffer((__gm__ P*)(gmParam[MAXINDICES_OUTPUT_IDX]));
}

template<typename W, typename I,  typename O, typename P, typename COMP_T>
__aicore__ inline void EmbeddingBagRegBaseSimt1D<W, I, O, P, COMP_T>::Process()
{
    int64_t mode = tilingData_.mode;
    int64_t isNeedSampleWeight = tilingData_.isNeedSampleWeight;
    COMP_T numBags = tilingData_.nBags;
    COMP_T numIndices = tilingData_.indicesNumel;
    COMP_T embeddingDimSize = tilingData_.embeddingDim;
    COMP_T paddingIdx = tilingData_.paddingIdx;

    COMP_T magic = 0;
    COMP_T shift = 0;
    COMP_T chunkPerBag = ops::CeilDiv(embeddingDimSize, static_cast<COMP_T>(BLOCK_DIM_0));
    GetUintDivMagicAndShift(magic, shift, chunkPerBag);

    if (mode == MODE_MAX) {
        Simt::VF_CALL<SimtComputeMax1D<W, I, O, P, COMP_T>>(Simt::Dim3{BLOCK_DIM_0, BLOCK_DIM_1}, 
                (__gm__ W*)(weightGm_.GetPhyAddr()), 
                (__gm__ I*)(indicesGm_.GetPhyAddr()),
                (__gm__ O*)(offsetsGm_.GetPhyAddr()),
                (__gm__ W*)(yGm_.GetPhyAddr()),
                (__gm__ P*)(offset2bagGm_.GetPhyAddr()),
                (__gm__ P*)(bagSizeGm_.GetPhyAddr()),
                (__gm__ P*)(maxIndicesGm_.GetPhyAddr()),
                numBags, numIndices, chunkPerBag, magic, shift,
                embeddingDimSize, paddingIdx);
    } else if (mode == MODE_MEAN) {
        Simt::VF_CALL<SimtComputeMean1D<W, I, O, P, COMP_T>>(Simt::Dim3{BLOCK_DIM_0, BLOCK_DIM_1}, 
                (__gm__ W*)(weightGm_.GetPhyAddr()), 
                (__gm__ I*)(indicesGm_.GetPhyAddr()),
                (__gm__ O*)(offsetsGm_.GetPhyAddr()),
                (__gm__ W*)(yGm_.GetPhyAddr()),
                (__gm__ P*)(offset2bagGm_.GetPhyAddr()),
                (__gm__ P*)(bagSizeGm_.GetPhyAddr()),
                numBags, numIndices, chunkPerBag, magic, shift,
                embeddingDimSize, paddingIdx);
    } else if (mode == MODE_SUM && isNeedSampleWeight == 0) {
        Simt::VF_CALL<SimtComputeSum1D<W, I, O, P, COMP_T>>(Simt::Dim3{BLOCK_DIM_0, BLOCK_DIM_1}, 
                (__gm__ W*)(weightGm_.GetPhyAddr()), 
                (__gm__ I*)(indicesGm_.GetPhyAddr()),
                (__gm__ O*)(offsetsGm_.GetPhyAddr()),
                (__gm__ W*)(yGm_.GetPhyAddr()),
                (__gm__ P*)(offset2bagGm_.GetPhyAddr()),
                (__gm__ P*)(bagSizeGm_.GetPhyAddr()),
                numBags, numIndices, chunkPerBag, magic, shift,
                embeddingDimSize, paddingIdx);
    } else {
        Simt::VF_CALL<SimtComputeSumWithPerSample1D<W, I, O, P, COMP_T>>(Simt::Dim3{BLOCK_DIM_0, BLOCK_DIM_1}, 
                (__gm__ W*)(weightGm_.GetPhyAddr()), 
                (__gm__ I*)(indicesGm_.GetPhyAddr()),
                (__gm__ O*)(offsetsGm_.GetPhyAddr()),
                (__gm__ W*)(perSampleWeightsGm_.GetPhyAddr()),
                (__gm__ W*)(yGm_.GetPhyAddr()),
                (__gm__ P*)(offset2bagGm_.GetPhyAddr()),
                (__gm__ P*)(bagSizeGm_.GetPhyAddr()),
                numBags, numIndices, chunkPerBag, magic, shift,
                embeddingDimSize, paddingIdx);
    }
}
}

#endif // EMBEDDING_BAG_H_REGBASE_1D_SIMT_H