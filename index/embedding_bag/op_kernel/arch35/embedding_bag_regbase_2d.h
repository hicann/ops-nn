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
 * \file embedding_bag_regbase_2d.h
 * \brief
 */

#ifndef EMBEDDING_BAG_REGBASE_2D_H
#define EMBEDDING_BAG_REGBASE_2D_H

#include "kernel_operator.h"
#include "../inc/kernel_utils.h"
#include "../inc/platform.h"
#include "embedding_bag_regbase_common.h"

namespace EmbeddingBag {
using namespace AscendC;

template<typename W, typename I, typename P>
class EmbeddingBagRegBaseIndx2d {
public:
    __aicore__ inline EmbeddingBagRegBaseIndx2d(const EmbeddingBagTilingData &tilingData, TPipe &pipe) :
        tilingData_(tilingData), pipe_(pipe) {};
    __aicore__ inline void Init(GM_ADDR gmParam[TENSOR_COUNT]);
    __aicore__ inline void CopyInIndices(int64_t indiceLpIdx, int64_t indicesNum);
    __aicore__ inline float PromoteType(W sampleValue);
    __aicore__ inline void CopyInWeightAndSum(
        int64_t indiceLpIdx, int64_t weightLpIdx, int64_t indicesLen, int64_t weightDimLen);
    __aicore__ inline void CopyInWeightAndMax(
        int64_t indiceLpIdx, int64_t weightLpIdx, int64_t indicesLen, int64_t weightDimLen);
    __aicore__ inline void CopyInWeightAndMean(
        int64_t indiceLpIdx, int64_t weightLpIdx, int64_t indicesLen, int64_t weightDimLen);
    __aicore__ inline void TensorAlloc();
    __aicore__ inline void TensorFree();
    __aicore__ inline void Process();

private:
    TPipe& pipe_;
    GlobalTensor<W> weightGm_;
    GlobalTensor<I> indicesGm_;
    GlobalTensor<I> offsetsGm_;
    GlobalTensor<W> perSampleWeightsGm_;
    GlobalTensor<W> yGm_;
    GlobalTensor<P> offset2bagGm_;
    GlobalTensor<P> bagSizeGm_;
    GlobalTensor<P> maxIndicesGm_;
    const EmbeddingBagTilingData& tilingData_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> weightQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> weightOutQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> indicesQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> perSamplWeightQue_;

    TBuf<QuePosition::VECCALC> offset2BagBuf_;
    TBuf<QuePosition::VECCALC> bagSizeBuf_;
    TBuf<QuePosition::VECCALC> maxIndicesOutBuf_;
    TBuf<QuePosition::VECCALC> maxIndicesCalcBuf_;
    TBuf<QuePosition::VECCALC> compareMaskBuf_;

    LocalTensor<W> perSampleWeightLocal_;

    int64_t indiceStart_{0};
    int64_t weightDimStart_{0};
    int64_t curCoreIndices_{0};
    int64_t curCoreEmbedDim_{0};
    int64_t weightRowLoop_{0};
    int64_t weightRowMainLen_{0};
    int64_t weightRowTailLen_{0};
};

template<typename W, typename I, typename P>
__aicore__ inline void EmbeddingBagRegBaseIndx2d<W, I, P>::Init(GM_ADDR gmParam[TENSOR_COUNT])
{
    weightGm_.SetGlobalBuffer((__gm__ W*)(gmParam[WEIGHT_INPUT_IDX]));
    indicesGm_.SetGlobalBuffer((__gm__ I*)(gmParam[INDICES_INPUT_IDX]));
    offsetsGm_.SetGlobalBuffer((__gm__ I*)(gmParam[OFFSETS_INPUT_IDX]));
    perSampleWeightsGm_.SetGlobalBuffer((__gm__ W*)(gmParam[PERSAMPWEIGHT_INPUT_IDX]));
    yGm_.SetGlobalBuffer((__gm__ W*)(gmParam[Y_OUTPUT_IDX]));
    offset2bagGm_.SetGlobalBuffer((__gm__ P*)(gmParam[OFFSET2BAG_OUTPUT_IDX]));
    bagSizeGm_.SetGlobalBuffer((__gm__ P*)(gmParam[BAGSIZE_OUTPUT_IDX]));
    maxIndicesGm_.SetGlobalBuffer((__gm__ P*)(gmParam[MAXINDICES_OUTPUT_IDX]));

    pipe_.InitBuffer(weightQue_, DOUBLE_BUFFER,
                    ops::CeilAlign((tilingData_.weightRowFactor * tilingData_.weightDimFactor) * sizeof(W), UB_AGLIN_VALUE));
    pipe_.InitBuffer(weightOutQue_, DOUBLE_BUFFER,
                    ops::CeilAlign((tilingData_.weightDimFactor) * sizeof(W), UB_AGLIN_VALUE));
    pipe_.InitBuffer(indicesQue_, DOUBLE_BUFFER,
                    ops::CeilAlign((tilingData_.indicesFactor) * tilingData_.indiceSize * sizeof(I), UB_AGLIN_VALUE));
       
    if (tilingData_.isNeedSampleWeight == 1) {
        pipe_.InitBuffer(perSamplWeightQue_, DOUBLE_BUFFER,
                        ops::CeilAlign((tilingData_.indicesFactor) * tilingData_.indiceSize * sizeof(W), UB_AGLIN_VALUE));
    }

    /* output que */
    pipe_.InitBuffer(offset2BagBuf_, ops::CeilAlign(tilingData_.indiceSize * sizeof(P), UB_AGLIN_VALUE));
    pipe_.InitBuffer(bagSizeBuf_, ops::CeilAlign((tilingData_.indicesFactor) * sizeof(P), UB_AGLIN_VALUE));

    if (tilingData_.mode == MODE_MAX) {
        pipe_.InitBuffer(maxIndicesOutBuf_, ops::CeilAlign(tilingData_.weightDimFactor * sizeof(P), UB_AGLIN_VALUE));
        pipe_.InitBuffer(maxIndicesCalcBuf_, ops::CeilAlign(tilingData_.weightDimFactor * sizeof(P), UB_AGLIN_VALUE));
        pipe_.InitBuffer(compareMaskBuf_, ops::CeilAlign(tilingData_.weightDimFactor * sizeof(uint8_t), UB_AGLIN_VALUE));
    }

    /* 计算搬运的起始偏移 */
    indiceStart_ = GetBlockIdx() / tilingData_.colTileNum * tilingData_.rowNormalNum;
    weightDimStart_ = GetBlockIdx() % tilingData_.colTileNum * tilingData_.colNormalNum;
    curCoreIndices_ = (GetBlockIdx() / tilingData_.colTileNum == tilingData_.rowTileNum - 1) ?
                      tilingData_.rowTailNum : tilingData_.rowNormalNum;
    curCoreEmbedDim_ = (GetBlockIdx() % tilingData_.colTileNum == tilingData_.colTileNum - 1) ?
                      tilingData_.colTailNum : tilingData_.colNormalNum;
}

template<typename W, typename I, typename P>
__aicore__ inline float EmbeddingBagRegBaseIndx2d<W, I, P>::PromoteType(W sampleValue)
{
    if constexpr (IsSameType<W, half>::value) {
        return static_cast<float>(sampleValue);
    } else if constexpr (IsSameType<W, bfloat16_t>::value) {
        return ToFloat(sampleValue);
    } else {
        return sampleValue;
    }
}

template<typename W, typename I, typename P>
__aicore__ inline void EmbeddingBagRegBaseIndx2d<W, I, P>::CopyInIndices(int64_t loopIdx, int64_t indicesNum)
{
    LocalTensor<I> inidcesLocal = indicesQue_.DeQue<I>();

    int64_t indicesLen = static_cast<uint32_t>(indicesNum * tilingData_.indiceSize);
    DataCopyExtParams copyParams = {static_cast<uint16_t>(1), 
                                    static_cast<uint32_t>(indicesLen * sizeof(I)),
                                    static_cast<uint32_t>(0),
                                    static_cast<uint32_t>(0),
                                    static_cast<uint32_t>(0)};
    DataCopyPadExtParams<I> padParams = {
        false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<I>(0)};

    auto indicesOfset = (indiceStart_ + loopIdx * tilingData_.indicesFactor) * tilingData_.indiceSize;
    DataCopyPad(inidcesLocal, indicesGm_[indicesOfset], copyParams, padParams);
    if (tilingData_.isNeedSampleWeight == 1) {
        perSampleWeightLocal_ = perSamplWeightQue_.DeQue<W>();
        copyParams.blockLen = static_cast<uint32_t>(indicesLen * sizeof(W));
        DataCopyPadExtParams<W> samplePadParams = {
            false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<W>(0)};
        DataCopyPad(perSampleWeightLocal_, perSampleWeightsGm_[indicesOfset], copyParams, samplePadParams);
        perSamplWeightQue_.EnQue(perSampleWeightLocal_);
    }
    indicesQue_.EnQue(inidcesLocal);
}

template<typename W, typename I, typename P>
__aicore__ inline void EmbeddingBagRegBaseIndx2d<W, I, P>::CopyInWeightAndSum(
    int64_t indiceLpIdx, int64_t weightLpIdx, int64_t indicesLen, int64_t weightDimLen)
{
    LocalTensor<I> inidcesLocal = indicesQue_.DeQue<I>();
    LocalTensor<W> weightLocalOut = weightOutQue_.DeQue<W>();
    LocalTensor<W> weightLocal = weightQue_.DeQue<W>();

    LocalTensor<P> bagSizeBuf = bagSizeBuf_.Get<P>();
    LocalTensor<P> offset2BagBuf = offset2BagBuf_.Get<P>();

    DataCopyExtParams copyParams = {static_cast<uint16_t>(1), 
                                    static_cast<uint32_t>(weightDimLen * sizeof(W)),
                                    static_cast<uint32_t>(0),
                                    static_cast<uint32_t>(0),
                                    static_cast<uint32_t>(0)};
    DataCopyPadExtParams<W> padParams = {
        false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<W>(0)};

    int64_t weightDimOfset = weightDimStart_ + weightLpIdx * tilingData_.weightDimFactor;
    int64_t indicesOutOfset = indiceStart_ + indiceLpIdx * tilingData_.indicesFactor;

    /* mode为sum的情况 */
    if (tilingData_.isNeedSampleWeight == 1) {
        perSampleWeightLocal_ = perSamplWeightQue_.DeQue<W>();
        for (int64_t bagIdx = 0; bagIdx < indicesLen; bagIdx++) {
            auto padNum = 0;
            int64_t indiceOfset = bagIdx * tilingData_.indiceSize;
            event_t eventIdMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);
            Duplicate(weightLocalOut, static_cast<W>(0), static_cast<int32_t>(weightDimLen));
            for (auto idx = 0; idx < tilingData_.indiceSize; idx++) {
                auto weightIdx = inidcesLocal(indiceOfset + idx);
                if (weightIdx == tilingData_.paddingIdx) {
                    padNum += 1;
                    continue;
                }
                auto weightStart = weightIdx * tilingData_.embeddingDim + weightDimOfset;
                auto sampleVale = perSampleWeightLocal_(indiceOfset + idx);

                event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
                WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
                copyParams.blockLen = static_cast<uint32_t>(weightDimLen * sizeof(W));
                DataCopyPad(weightLocal, weightGm_[weightStart], copyParams, padParams);

                event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
                WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
                ComputeSumWithWeight<W>(weightLocal, weightLocalOut, PromoteType(sampleVale), 1, weightDimLen, tilingData_.weightDimFactor);
            }
            event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            copyParams.blockLen = static_cast<uint32_t>(weightDimLen * sizeof(W));
            DataCopyPad(yGm_[(indicesOutOfset + bagIdx) * tilingData_.embeddingDim + weightDimOfset], weightLocalOut, copyParams);

            Duplicate(offset2BagBuf, static_cast<P>(indicesOutOfset + bagIdx), static_cast<int32_t>(tilingData_.indiceSize));
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            copyParams.blockLen = static_cast<uint32_t>(tilingData_.indiceSize * sizeof(P));
            DataCopyPad(offset2bagGm_[(indicesOutOfset + bagIdx) * tilingData_.indiceSize], offset2BagBuf, copyParams);
            bagSizeBuf(bagIdx) = tilingData_.indiceSize - padNum;
        }
    } else {
        for (int64_t bagIdx = 0; bagIdx < indicesLen; bagIdx++) {
            int64_t padNum = 0;
            int64_t indiceOfset = bagIdx * tilingData_.indiceSize;
            int64_t outRowIdx = indicesOutOfset + bagIdx;
            event_t eventIdMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);
            Duplicate(weightLocalOut, static_cast<W>(0), static_cast<int32_t>(weightDimLen));
            for (auto loopIdx = 0; loopIdx < weightRowLoop_; loopIdx++) {
                int64_t weightRowNum = (loopIdx == weightRowLoop_ - 1) ? weightRowTailLen_ : weightRowMainLen_;
                int64_t padNumInLoop = 0;
                event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
                for (auto weightRow = 0; weightRow < weightRowNum; weightRow++) {
                    int64_t weightIdx = inidcesLocal(indiceOfset + loopIdx * weightRowMainLen_ + weightRow);
                    if (weightIdx == tilingData_.paddingIdx) {
                        padNumInLoop += 1;
                        continue;
                    }
                    int64_t weightStart = weightIdx * tilingData_.embeddingDim + weightDimOfset;
                    copyParams.blockLen = static_cast<uint32_t>(weightDimLen * sizeof(W));
                    int64_t weightLocalDstIdx = (weightRow - padNumInLoop) * tilingData_.weightDimFactor;
                    DataCopyPad(weightLocal[weightLocalDstIdx], weightGm_[weightStart], copyParams, padParams);
                }
                event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
                WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
                ComputeSum<W>(weightLocal, weightLocalOut, weightRowNum - padNumInLoop, weightDimLen, tilingData_.weightDimFactor);
                padNum += padNumInLoop;
            }
            event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            copyParams.blockLen = static_cast<uint32_t>(weightDimLen * sizeof(W));
            DataCopyPad(yGm_[(outRowIdx) * tilingData_.embeddingDim + weightDimOfset], weightLocalOut, copyParams);

            Duplicate(offset2BagBuf, static_cast<P>(outRowIdx), static_cast<int32_t>(tilingData_.indiceSize));
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            copyParams.blockLen = static_cast<uint32_t>(tilingData_.indiceSize * sizeof(P));
            DataCopyPad(offset2bagGm_[(outRowIdx) * tilingData_.indiceSize], offset2BagBuf, copyParams);
            bagSizeBuf(bagIdx) = tilingData_.indiceSize - padNum;
        }
    }
    copyParams.blockLen = static_cast<uint32_t>(indicesLen * sizeof(P));
    DataCopyPad(bagSizeGm_[indicesOutOfset], bagSizeBuf, copyParams);

    indicesQue_.EnQue(inidcesLocal);
    weightOutQue_.EnQue(weightLocalOut);
    weightQue_.EnQue(weightLocal);
}

template<typename W, typename I, typename P>
__aicore__ inline void EmbeddingBagRegBaseIndx2d<W, I, P>::CopyInWeightAndMax(
    int64_t indiceLpIdx, int64_t weightLpIdx, int64_t indicesLen, int64_t weightDimLen)
{
    LocalTensor<I> inidcesLocal = indicesQue_.DeQue<I>();
    LocalTensor<W> weightLocalOut = weightOutQue_.DeQue<W>();
    LocalTensor<W> weightLocal = weightQue_.DeQue<W>();

    LocalTensor<P> offset2BagBuf = offset2BagBuf_.Get<P>();
    LocalTensor<P> bagSizeBuf = bagSizeBuf_.Get<P>();
    LocalTensor<P> maxIndicesOutBuf = maxIndicesOutBuf_.Get<P>();
    LocalTensor<P> maxIndicesCalcBuf = maxIndicesCalcBuf_.Get<P>();
    LocalTensor<uint8_t> compareMaskBuf = compareMaskBuf_.Get<uint8_t>();

    DataCopyExtParams copyParams = {static_cast<uint16_t>(1), 
                                    static_cast<uint32_t>(weightDimLen * sizeof(W)),
                                    static_cast<uint32_t>(0),
                                    static_cast<uint32_t>(0),
                                    static_cast<uint32_t>(0)};
    DataCopyPadExtParams<W> padParams = {
        false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<W>(0)};

    /* mode为max的情况 */
    int64_t weightDimOfset = weightDimStart_ + weightLpIdx * tilingData_.weightDimFactor;
    int64_t indicesOutOfset = indiceStart_ + indiceLpIdx * tilingData_.indicesFactor;
    for (int64_t bagIdx = 0; bagIdx < indicesLen; bagIdx++) {
        auto padNum = 0;
        int64_t indiceOfset = bagIdx * tilingData_.indiceSize;
        event_t eventIdMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);
        DuplicateNegInf<W>(weightLocalOut, static_cast<int32_t>(weightDimLen));
        for (auto idx = 0; idx < tilingData_.indiceSize; idx++) {
            auto weightIdx = inidcesLocal(indiceOfset + idx);
            if (weightIdx == tilingData_.paddingIdx) {
                padNum += 1;
                continue;
            }
            auto weightStart = weightIdx * tilingData_.embeddingDim + weightDimOfset;
            Duplicate(maxIndicesCalcBuf, static_cast<P>(weightIdx), static_cast<int32_t>(weightDimLen));
            event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);

            copyParams.blockLen = static_cast<uint32_t>(weightDimLen * sizeof(W));
            DataCopyPad(weightLocal, weightGm_[weightStart], copyParams, padParams);

            event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
            Compare(compareMaskBuf, weightLocal, weightLocalOut, CMPMODE::GT, weightDimLen);
            Select(maxIndicesOutBuf, compareMaskBuf, maxIndicesCalcBuf, maxIndicesOutBuf, SELMODE::VSEL_TENSOR_TENSOR_MODE, weightDimLen);
            Max(weightLocalOut, weightLocalOut, weightLocal, weightDimLen);
        }
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        copyParams.blockLen = static_cast<uint32_t>(weightDimLen * sizeof(W));
        DataCopyPad(yGm_[(indicesOutOfset + bagIdx) * tilingData_.embeddingDim + weightDimOfset], weightLocalOut, copyParams);

        Duplicate(offset2BagBuf, static_cast<P>(indicesOutOfset + bagIdx), static_cast<int32_t>(tilingData_.indiceSize));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        copyParams.blockLen = static_cast<uint32_t>(tilingData_.indiceSize * sizeof(P));
        DataCopyPad(offset2bagGm_[(indicesOutOfset + bagIdx) * tilingData_.indiceSize], offset2BagBuf, copyParams);

        copyParams.blockLen = static_cast<uint32_t>(weightDimLen * sizeof(P));
        DataCopyPad(maxIndicesGm_[(indicesOutOfset + bagIdx) * tilingData_.embeddingDim + weightDimOfset], maxIndicesOutBuf, copyParams);
        bagSizeBuf(bagIdx) = tilingData_.indiceSize - padNum;
    }
    copyParams.blockLen = static_cast<uint32_t>(indicesLen * sizeof(P));
    DataCopyPad(bagSizeGm_[indicesOutOfset], bagSizeBuf, copyParams);

    indicesQue_.EnQue(inidcesLocal);
    weightOutQue_.EnQue(weightLocalOut);
    weightQue_.EnQue(weightLocal);
}

template<typename W, typename I, typename P>
__aicore__ inline void EmbeddingBagRegBaseIndx2d<W, I, P>::CopyInWeightAndMean(
    int64_t indiceLpIdx, int64_t weightLpIdx, int64_t indicesLen, int64_t weightDimLen)
{
    LocalTensor<I> inidcesLocal = indicesQue_.DeQue<I>();
    LocalTensor<W> weightLocalOut = weightOutQue_.DeQue<W>();
    LocalTensor<W> weightLocal = weightQue_.DeQue<W>();

    LocalTensor<P> bagSizeBuf = bagSizeBuf_.Get<P>();
    LocalTensor<P> offset2BagBuf = offset2BagBuf_.Get<P>();

    DataCopyExtParams copyParams = {static_cast<uint16_t>(1), 
                                    static_cast<uint32_t>(weightDimLen * sizeof(W)),
                                    static_cast<uint32_t>(0),
                                    static_cast<uint32_t>(0),
                                    static_cast<uint32_t>(0)};
    DataCopyPadExtParams<W> padParams = {
        false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<W>(0)};

    int64_t weightDimOfset = weightDimStart_ + weightLpIdx * tilingData_.weightDimFactor;
    int64_t indicesOutOfset = indiceStart_ + indiceLpIdx * tilingData_.indicesFactor;

    /* mode为mean的情况 */
    for (int64_t bagIdx = 0; bagIdx < indicesLen; bagIdx++) {
        auto padNum = 0;
        int64_t indiceOfset = bagIdx * tilingData_.indiceSize;
        event_t eventIdMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE3ToV);
        Duplicate(weightLocalOut, static_cast<W>(0), static_cast<int32_t>(weightDimLen));
        for (auto loopIdx = 0; loopIdx < weightRowLoop_; loopIdx++) {
            int64_t weightRowNum = (loopIdx == weightRowLoop_ - 1) ? weightRowTailLen_ : weightRowMainLen_;
            int64_t padNumInLoop = 0;
            event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            for (auto weightRow = 0; weightRow < weightRowNum; weightRow++) {
                int64_t weightIdx = inidcesLocal(indiceOfset + loopIdx * weightRowMainLen_ + weightRow);
                if (weightIdx == tilingData_.paddingIdx) {
                    padNumInLoop += 1;
                    continue;
                }
                int64_t weightStart = weightIdx * tilingData_.embeddingDim + weightDimOfset;
                copyParams.blockLen = static_cast<uint32_t>(weightDimLen * sizeof(W));
                int64_t weightLocalDstIdx = (weightRow - padNumInLoop) * tilingData_.weightDimFactor;
                DataCopyPad(weightLocal[weightLocalDstIdx], weightGm_[weightStart], copyParams, padParams);
            }
            event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
            ComputeSum<W>(weightLocal, weightLocalOut, weightRowNum - padNumInLoop, weightDimLen, tilingData_.weightDimFactor);
            padNum += padNumInLoop;
        }
        DivForMean<W>(weightLocalOut, tilingData_.indiceSize - padNum, weightDimLen);
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        copyParams.blockLen = static_cast<uint32_t>(weightDimLen * sizeof(W));
        DataCopyPad(yGm_[(indicesOutOfset + bagIdx) * tilingData_.embeddingDim + weightDimOfset], weightLocalOut, copyParams);

        Duplicate(offset2BagBuf, static_cast<P>(indicesOutOfset + bagIdx), static_cast<int32_t>(tilingData_.indiceSize));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        copyParams.blockLen = static_cast<uint32_t>(tilingData_.indiceSize * sizeof(P));
        DataCopyPad(offset2bagGm_[(indicesOutOfset + bagIdx) * tilingData_.indiceSize], offset2BagBuf, copyParams);
        bagSizeBuf(bagIdx) = tilingData_.indiceSize - padNum;
    }
    
    copyParams.blockLen = static_cast<uint32_t>(indicesLen * sizeof(P));
    DataCopyPad(bagSizeGm_[indicesOutOfset], bagSizeBuf, copyParams);

    indicesQue_.EnQue(inidcesLocal);
    weightOutQue_.EnQue(weightLocalOut);
    weightQue_.EnQue(weightLocal);
}

template<typename W, typename I, typename P>
__aicore__ inline void EmbeddingBagRegBaseIndx2d<W, I, P>::TensorAlloc()
{
    LocalTensor<I> inidcesLocal = indicesQue_.AllocTensor<I>();
    LocalTensor<W> weightLocalOut = weightOutQue_.AllocTensor<W>();
    LocalTensor<W> weightLocal = weightQue_.AllocTensor<W>();
    if (tilingData_.isNeedSampleWeight == 1) {
        LocalTensor<W> perSampleWeightLocal = perSamplWeightQue_.AllocTensor<W>();
        perSamplWeightQue_.EnQue(perSampleWeightLocal);
    }

    indicesQue_.EnQue(inidcesLocal);
    weightOutQue_.EnQue(weightLocalOut);
    weightQue_.EnQue(weightLocal);
}

template<typename W, typename I, typename P>
__aicore__ inline void EmbeddingBagRegBaseIndx2d<W, I, P>::TensorFree()
{
    LocalTensor<I> inidcesLocal = indicesQue_.DeQue<I>();
    LocalTensor<W> weightLocalOut = weightOutQue_.DeQue<W>();
    LocalTensor<W> weightLocal = weightQue_.DeQue<W>();
    if (tilingData_.isNeedSampleWeight == 1) {
        LocalTensor<W> perSampleWeightLocal = perSamplWeightQue_.DeQue<W>();
        perSamplWeightQue_.FreeTensor(perSampleWeightLocal);
    }

    indicesQue_.FreeTensor(inidcesLocal);
    weightOutQue_.FreeTensor(weightLocalOut);
    weightQue_.FreeTensor(weightLocal);
}

template<typename W, typename I, typename P>
__aicore__ inline void EmbeddingBagRegBaseIndx2d<W, I, P>::Process()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNum) {
        return;
    }

    /* offsets和indices能全载，tiling里做判断 */
    int64_t indicesLoop = ops::CeilDiv(curCoreIndices_, tilingData_.indicesFactor);
    int64_t indicesMainLen =  tilingData_.indicesFactor;
    int64_t indicesTailLen =  curCoreIndices_ - tilingData_.indicesFactor * (indicesLoop - 1);

    int64_t weightDimLoop = ops::CeilDiv(curCoreEmbedDim_, tilingData_.weightDimFactor);
    int64_t weightDimMainLen =  tilingData_.weightDimFactor;
    int64_t weightDimTailLen =  curCoreEmbedDim_ - tilingData_.weightDimFactor * (weightDimLoop - 1);

    weightRowLoop_ = ops::CeilDiv(tilingData_.indiceSize, tilingData_.weightRowFactor);
    weightRowMainLen_ =  tilingData_.weightRowFactor;
    weightRowTailLen_ =  tilingData_.indiceSize - tilingData_.weightRowFactor * (weightRowLoop_ - 1);

    TensorAlloc();
    for (int64_t indiceLpIdx = 0; indiceLpIdx < indicesLoop; indiceLpIdx++) {
        auto handleIndices = (indiceLpIdx == indicesLoop - 1) ? indicesTailLen : indicesMainLen;
        CopyInIndices(indiceLpIdx, handleIndices);

        for (int64_t weightLpIdx = 0; weightLpIdx < weightDimLoop; weightLpIdx++) {
            auto handleWeightDim = (weightLpIdx == weightDimLoop - 1) ? weightDimTailLen : weightDimMainLen;
            if (tilingData_.mode == MODE_SUM) {
                CopyInWeightAndSum(indiceLpIdx, weightLpIdx, handleIndices, handleWeightDim);
            } else if (tilingData_.mode == MODE_MAX) {
                CopyInWeightAndMax(indiceLpIdx, weightLpIdx, handleIndices, handleWeightDim);
            } else if (tilingData_.mode == MODE_MEAN) {
                CopyInWeightAndMean(indiceLpIdx, weightLpIdx, handleIndices, handleWeightDim);
            }
        }
    }
    TensorFree();
}
}

#endif // EMBEDDING_BAG_H_REGBASE_2D_H