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
 * \file ada_layer_norm_grad_common_phase.inl
 * \brief
 */


#ifndef ADA_LAYER_NORM_GRAD_COMMON_PHASE_INL
#define ADA_LAYER_NORM_GRAD_COMMON_PHASE_INL

namespace AdaLayerNormGrad {

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::Process(
    const AdaLayerNormGradTilingDataCommon* tilingData)
{
    if (GetBlockIdx() < tilingData->blockNum) {
        int64_t ubLoopCount = (GetBlockIdx() == tilingData->blockNum - 1)
                                  ? tilingData->ubLoopOfTailBlock
                                  : tilingData->ubLoopOfFormerBlock;
        int64_t tailRowsNum = (GetBlockIdx() == tilingData->blockNum - 1)
                                  ? tilingData->ubTailOfTailBlock
                                  : tilingData->ubTailOfFormerBlock;
        int64_t blockStartRow = GetBlockIdx() * tilingData->blockFormer;
        blockStartBatch_ = blockStartRow / tilingData->seq;

        CopyInGamma(tilingData);
        CopyInBeta(tilingData);
        for (int64_t i = 0; i < ubLoopCount - 1; i++) {
            int64_t startRow = blockStartRow + i * tilingData->ubFormer;
            CopyInPhase0(tilingData, tilingData->ubFormer, startRow);
            ComputePhase0(tilingData);
            CopyInPhase1(i, tilingData->ubFormer, tilingData);
            ComputePhase1(i, tilingData->ubFormer, startRow, tilingData);
            CopyInPhase2(i, tilingData->ubFormer, tilingData);
            ComputePhase2(i, tilingData->ubFormer, tilingData);
            CopyInPhase3(i, tilingData->ubFormer, tilingData);
            ComputePhase3(i, tilingData->ubFormer, tilingData);
            CopyInPhase4(i, tilingData->ubFormer, tilingData);
            ComputePhase4(i, tilingData->ubFormer, startRow, tilingData);
            CopyOutPhase0(i, tilingData->ubFormer, tilingData);
            CopyOutPhase1(startRow, tilingData);
        }

        int64_t startRow = blockStartRow + (ubLoopCount - 1) * tilingData->ubFormer;
        CopyInPhase0(tilingData, tailRowsNum, startRow);
        ComputePhase0(tilingData);
        CopyInPhase1(ubLoopCount - 1, tailRowsNum, tilingData);
        ComputePhase1(ubLoopCount - 1, tailRowsNum, startRow, tilingData);
        CopyInPhase2(ubLoopCount - 1, tailRowsNum, tilingData);
        ComputePhase2(ubLoopCount - 1, tailRowsNum, tilingData);
        CopyInPhase3(ubLoopCount - 1, tailRowsNum, tilingData);
        ComputePhase3(ubLoopCount - 1, tailRowsNum, tilingData);
        CopyInPhase4(ubLoopCount - 1, tailRowsNum, tilingData);
        ComputePhase4(ubLoopCount - 1, tailRowsNum, startRow, tilingData);
        CopyOutPhase0(ubLoopCount - 1, tailRowsNum, tilingData);
        CopyOutPhase1(startRow, tilingData);
        CopyOutPhase1Deterministic(tilingData);

        queue3_.FreeTensor(buffer3_);
        queue4_.FreeTensor(buffer4_);
    }

    if constexpr (isDeterministic) {
        CopyOutPhase2Deterministic(tilingData);
    } else if (GetBlockIdx() < tilingData->blockNum) {
        CrossCoreWaitFlag(SYNC_AIV_ONLY_ALL);
        CopyOutPhase2(tilingData);
    } else {
        CrossCoreWaitFlag(SYNC_AIV_ONLY_ALL);
    }
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::CopyInBeta(
    const AdaLayerNormGradTilingDataCommon* tilingData)
{
    buffer4_ = queue4_.AllocTensor<float>();
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = tilingData->col * sizeof(U);
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;

    if constexpr (IsSameType<U, float>::value) {
        DataCopyPad(buffer4_.ReinterpretCast<U>(), betaInTensorGM_, intriParams, padParams);
    } else {
        DataCopyPad(buffer4_.ReinterpretCast<U>()[tilingData->colAlignM], betaInTensorGM_, intriParams, padParams);
    }

    event_t event1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(event1);
    WaitFlag<HardEvent::MTE2_V>(event1);

    if constexpr (!IsSameType<U, float>::value) {
        Cast(
            buffer4_, buffer4_.ReinterpretCast<U>()[tilingData->colAlignM], RoundMode::CAST_NONE,
            tilingData->colAlignV);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::CopyInGamma(
    const AdaLayerNormGradTilingDataCommon* tilingData)
{
    buffer3_ = queue3_.AllocTensor<float>();
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = tilingData->col * sizeof(U);
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;

    if constexpr (IsSameType<U, float>::value) {
        DataCopyPad(buffer3_.ReinterpretCast<U>(), gammaInTensorGM_, intriParams, padParams);
    } else {
        DataCopyPad(buffer3_.ReinterpretCast<U>()[tilingData->colAlignM], gammaInTensorGM_, intriParams, padParams);
    }

    event_t event1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(event1);
    WaitFlag<HardEvent::MTE2_V>(event1);

    if constexpr (!IsSameType<U, float>::value) {
        Cast(
            buffer3_, buffer3_.ReinterpretCast<U>()[tilingData->colAlignM], RoundMode::CAST_NONE,
            tilingData->colAlignV);
        PipeBarrier<PIPE_V>();
    }

    buffer9_ = queue9_.AllocTensor<float>();
    Duplicate(buffer9_, static_cast<float>(0.0), tilingData->colAlignV);

    buffer10_ = queue10_.AllocTensor<float>();
    Duplicate(buffer10_, static_cast<float>(0.0), tilingData->colAlignV);

    buffer11_ = tmpTensor0_.Get<float>();
    buffer12_ = tmpTensor1_.Get<float>();
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::CopyInPhase0(
    const AdaLayerNormGradTilingDataCommon* tilingData, const int64_t curRowsNum, const int64_t startRow)
{
    buffer5_ = queue5_.AllocTensor<float>();
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams intriParams;

    int64_t cpStartBatch = startRow / tilingData->seq;
    int64_t cpEndBatch = (startRow + curRowsNum) / tilingData->seq;
    ScaleReRowNum_ = cpEndBatch - cpStartBatch + 1;

    ScaleReEleNum_ = ScaleReRowNum_ * tilingData->col;

    if (likely(tilingData->colAlignV == tilingData->col)) {
        intriParams.blockCount = 1;
        intriParams.blockLen = ScaleReEleNum_ * sizeof(T);
    } else {
        intriParams.blockCount = ScaleReRowNum_;
        intriParams.blockLen = tilingData->col * sizeof(T);
        padParams.isPad = true;
        padParams.rightPadding = tilingData->colAlignM - tilingData->col;
    }
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;

    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(buffer5_.ReinterpretCast<T>(), scaleInTensorGM_[(cpStartBatch - blockStartBatch_) * tilingData->col], intriParams, padParams);
    } else {
        DataCopyPad(
            buffer5_.ReinterpretCast<T>()[tilingData->blockFormerScaleBufferElemNums],
            scaleInTensorGM_[(cpStartBatch - blockStartBatch_) * tilingData->col], intriParams, padParams);
    }

    queue5_.EnQue(buffer5_);
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::ComputePhase0(
    const AdaLayerNormGradTilingDataCommon* tilingData)
{
    buffer5_ = queue5_.DeQue<float>();

    if constexpr (!IsSameType<T, float>::value) {
        CastToFloat(
            buffer5_, buffer11_, ScaleReRowNum_, tilingData, tilingData->blockFormerScaleBufferElemNums);
        PipeBarrier<PIPE_V>();
    }

    buffer7_ = queue7_.AllocTensor<float>();
    Duplicate(buffer7_, static_cast<float>(0.0), tilingData->blockFormerScaleBufferElemNums);

    buffer8_ = queue8_.AllocTensor<float>();
    Duplicate(buffer8_, static_cast<float>(0.0), tilingData->blockFormerScaleBufferElemNums);

    PipeBarrier<PIPE_V>();
    event_t event1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(event1);
    WaitFlag<HardEvent::MTE2_V>(event1);

    Adds(buffer5_, buffer5_, static_cast<float>(1.0), ScaleReRowNum_ * tilingData->colAlignV);
    PipeBarrier<PIPE_V>();
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::CopyInPhase1(
    const int64_t outerIdx, const int64_t curRowsNum, const AdaLayerNormGradTilingDataCommon* tilingData)
{
    buffer1_ = queue1_.AllocTensor<float>();
    DataCopyPadParams padParams_dy{false, 0, 0, 0};
    DataCopyParams intriParams_dy;

    if (likely(tilingData->colAlignV == tilingData->col)) {
        intriParams_dy.blockCount = 1;
        intriParams_dy.blockLen = curRowsNum * tilingData->col * sizeof(T);
    } else {
        intriParams_dy.blockCount = curRowsNum;
        intriParams_dy.blockLen = tilingData->col * sizeof(T);
        padParams_dy.isPad = true;
        padParams_dy.rightPadding = tilingData->colAlignM - tilingData->col;
    }
    intriParams_dy.srcStride = 0;
    intriParams_dy.dstStride = 0;

    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(
            buffer1_.ReinterpretCast<T>(), dyInTensorGM_[tilingData->ubFormer * tilingData->col * outerIdx],
            intriParams_dy, padParams_dy);
    } else {
        DataCopyPad(
            buffer1_.ReinterpretCast<T>()[tilingData->wholeBufferElemNums],
            dyInTensorGM_[tilingData->ubFormer * tilingData->col * outerIdx], intriParams_dy, padParams_dy);
    }

    queue1_.EnQue(buffer1_);
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::ComputePhase1(
    const int64_t outerIdx, const int64_t curRowsNum, const int64_t startRow,
    const AdaLayerNormGradTilingDataCommon* tilingData)
{
    buffer1_ = queue1_.DeQue<float>();
    if constexpr (!IsSameType<T, float>::value) {
        CastToFloat(buffer1_, buffer11_, curRowsNum, tilingData, tilingData->wholeBufferElemNums);
    }
    PipeBarrier<PIPE_V>();
    NlastBatchReduceSum(buffer8_, buffer1_, curRowsNum, startRow, tilingData);
    PipeBarrier<PIPE_V>();
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::CopyInPhase2(
    const int64_t outerIdx, const int64_t curRowsNum, const AdaLayerNormGradTilingDataCommon* tilingData)
{
    buffer2_ = queue2_.AllocTensor<float>();
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = curRowsNum * sizeof(float);
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;
    DataCopyPad(buffer2_, meanInTensorGM_[tilingData->ubFormer * outerIdx], intriParams, padParams);
    queue2_.EnQue(buffer2_);
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::ComputePhase2(
    const int64_t outerIdx, const int64_t curRowsNum, const AdaLayerNormGradTilingDataCommon* tilingData)
{
    buffer2_ = queue2_.DeQue<float>();
    BlockBroadcast<float>(buffer12_, buffer2_, curRowsNum);
    queue2_.FreeTensor(buffer2_);
    PipeBarrier<PIPE_V>();
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::CopyInPhase3(
    const int64_t outerIdx, const int64_t curRowsNum, const AdaLayerNormGradTilingDataCommon* tilingData)
{
    buffer0_ = queue0_.AllocTensor<float>();
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams intriParams;

    if (likely(tilingData->colAlignV == tilingData->col)) {
        intriParams.blockCount = 1;
        intriParams.blockLen = curRowsNum * tilingData->col * sizeof(T);
    } else {
        intriParams.blockCount = curRowsNum;
        intriParams.blockLen = tilingData->col * sizeof(T);
        padParams.isPad = true;
        padParams.rightPadding = tilingData->colAlignM - tilingData->col;
    }
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;

    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(
            buffer0_.ReinterpretCast<T>(), xInTensorGM_[tilingData->ubFormer * tilingData->col * outerIdx],
            intriParams, padParams);
    } else {
        DataCopyPad(
            buffer0_.ReinterpretCast<T>()[tilingData->wholeBufferElemNums],
            xInTensorGM_[tilingData->ubFormer * tilingData->col * outerIdx], intriParams, padParams);
    }

    queue0_.EnQue(buffer0_);
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::ComputePhase3(
    const int64_t outerIdx, const int64_t curRowsNum, const AdaLayerNormGradTilingDataCommon* tilingData)
{
    buffer0_ = queue0_.DeQue<float>();

    if constexpr (!IsSameType<T, float>::value) {
        CastToFloat(buffer0_, buffer11_, curRowsNum, tilingData, tilingData->wholeBufferElemNums);
    }
    PipeBarrier<PIPE_V>();

    BinElemWithInlinedLastBrcFP32(buffer0_, buffer0_, buffer12_, curRowsNum, tilingData, Sub);
    PipeBarrier<PIPE_V>();
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::CopyInPhase4(
    const int64_t outerIdx, const int64_t curRowsNum, const AdaLayerNormGradTilingDataCommon* tilingData)
{
    buffer2_ = queue2_.AllocTensor<float>();
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = curRowsNum * sizeof(float);
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;
    DataCopyPad(buffer2_, rstdInTensorGM_[tilingData->ubFormer * outerIdx], intriParams, padParams);
    queue2_.EnQue(buffer2_);
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::ComputePhase4(
    const int64_t outerIdx, const int64_t curRowsNum, const int64_t startRow,
    const AdaLayerNormGradTilingDataCommon* tilingData)
{
    buffer6_ = queue6_.AllocTensor<float>();

    NlastBatchMul(buffer6_, buffer1_, buffer5_, curRowsNum, startRow, tilingData);
    PipeBarrier<PIPE_V>();
    queue5_.FreeTensor(buffer5_);

    NlastReduceSum(buffer10_, buffer6_, curRowsNum, tilingData);
    PipeBarrier<PIPE_V>();

    buffer2_ = queue2_.DeQue<float>();
    BlockBroadcast<float>(buffer12_, buffer2_, curRowsNum);
    PipeBarrier<PIPE_V>();

    BinElemWithInlinedLastBrcFP32(buffer0_, buffer0_, buffer12_, curRowsNum, tilingData, Mul);
    PipeBarrier<PIPE_V>();

    Mul(buffer11_, buffer6_, buffer0_, curRowsNum * tilingData->colAlignV);
    PipeBarrier<PIPE_V>();

    NlastReduceSum(buffer9_, buffer11_, curRowsNum, tilingData);
    PipeBarrier<PIPE_V>();

    BinElemWithInlinedNLastBrcFP32(buffer11_, buffer0_, buffer3_, curRowsNum, tilingData, Mul);
    PipeBarrier<PIPE_V>();

    BinElemWithInlinedNLastBrcFP32(buffer11_, buffer11_, buffer4_, curRowsNum, tilingData, Add);
    PipeBarrier<PIPE_V>();

    Mul(buffer11_, buffer1_, buffer11_, curRowsNum * tilingData->colAlignV);
    PipeBarrier<PIPE_V>();

    NlastBatchReduceSum(buffer7_, buffer11_, curRowsNum, startRow, tilingData);
    PipeBarrier<PIPE_V>();

    BinElemWithInlinedNLastBrcFP32(buffer6_, buffer6_, buffer3_, curRowsNum, tilingData, Mul);
    PipeBarrier<PIPE_V>();

    LastReduceSum(buffer2_, buffer6_, buffer11_, curRowsNum, tilingData);
    PipeBarrier<PIPE_V>();

    Muls(buffer2_, buffer2_, coff_, curRowsNum);
    PipeBarrier<PIPE_V>();

    Mul(buffer1_, buffer6_, buffer0_, curRowsNum * tilingData->colAlignV);
    PipeBarrier<PIPE_V>();

    BlockBroadcast<float>(buffer11_, buffer2_, curRowsNum);
    PipeBarrier<PIPE_V>();

    BinElemWithInlinedLastBrcFP32(buffer6_, buffer6_, buffer11_, curRowsNum, tilingData, Sub);
    PipeBarrier<PIPE_V>();

    LastReduceSum(buffer2_, buffer1_, buffer11_, curRowsNum, tilingData);
    PipeBarrier<PIPE_V>();
    queue1_.FreeTensor(buffer1_);

    Muls(buffer2_, buffer2_, coff_, curRowsNum);
    PipeBarrier<PIPE_V>();

    BlockBroadcast<float>(buffer11_, buffer2_, curRowsNum);
    queue2_.FreeTensor(buffer2_);
    PipeBarrier<PIPE_V>();

    BinElemWithInlinedLastBrcFP32(buffer0_, buffer0_, buffer11_, curRowsNum, tilingData, Mul);
    PipeBarrier<PIPE_V>();

    Sub(buffer6_, buffer6_, buffer0_, curRowsNum * tilingData->colAlignV);
    PipeBarrier<PIPE_V>();
    queue0_.FreeTensor(buffer0_);

    BinElemWithInlinedLastBrcFP32(buffer6_, buffer6_, buffer12_, curRowsNum, tilingData, Mul);
    PipeBarrier<PIPE_V>();

    if constexpr (!IsSameType<T, float>::value) {
        CastToB16(buffer6_, buffer11_, curRowsNum, tilingData);
    }

    queue6_.EnQue(buffer6_);
}

} // namespace AdaLayerNormGrad

#endif