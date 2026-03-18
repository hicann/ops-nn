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
 * \file ada_layer_norm_grad_common_output.inl
 * \brief
 */

#ifndef ADA_LAYER_NORM_GRAD_COMMON_OUTPUT_INL
#define ADA_LAYER_NORM_GRAD_COMMON_OUTPUT_INL

namespace AdaLayerNormGrad {

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::CopyOutPhase0(
    const int64_t outerIdx, const int64_t curRowsNum, const AdaLayerNormGradTilingDataCommon* tilingData)
{
    event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(event);
    WaitFlag<HardEvent::V_MTE3>(event);

    buffer6_ = queue6_.DeQue<float>();
    DataCopyParams intriParams;
    if (likely(tilingData->colAlignV == tilingData->col)) {
        intriParams.blockCount = 1;
        intriParams.blockLen = curRowsNum * tilingData->col * sizeof(T);
    } else {
        intriParams.blockCount = curRowsNum;
        intriParams.blockLen = tilingData->col * sizeof(T);
    }
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;

    DataCopyPad(
        pdXOutTensorGM_[tilingData->ubFormer * tilingData->col * outerIdx], buffer6_.ReinterpretCast<T>(),
        intriParams);
    queue6_.FreeTensor(buffer6_);
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::CopyOutPhase1(
    const int64_t startRow, const AdaLayerNormGradTilingDataCommon* tilingData)
{
    event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(event);
    WaitFlag<HardEvent::V_MTE3>(event);

    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = ScaleReRowNum_ * tilingData->colAlignV * sizeof(float);
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;
    SetAtomicAdd<float>();

    int64_t cpOutBatchIdx = startRow / tilingData->seq;
    int64_t cpOutOffset = (cpOutBatchIdx - blockStartBatch_) * tilingData->colAlignV;

    queue7_.EnQue(buffer7_);
    buffer7_ = queue7_.DeQue<float>();
    DataCopyPad(dScaleWorkspaceGM[cpOutOffset], buffer7_, intriParams);
    queue7_.FreeTensor(buffer7_);

    queue8_.EnQue(buffer8_);
    buffer8_ = queue8_.DeQue<float>();
    DataCopyPad(dShiftWorkspaceGM[cpOutOffset], buffer8_, intriParams);
    queue8_.FreeTensor(buffer8_);

    SetAtomicNone();
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::CopyOutPhase1Deterministic(
    const AdaLayerNormGradTilingDataCommon* tilingData){
        int64_t BatchRowsNum =
            ((GetBlockIdx() != tilingData->blockNum - 1) ? tilingData->blockFormer : tilingData->blockTail) / tilingData->seq;
        int64_t rowOfBuffer = ((tilingData->ubFormer + tilingData->seq - 1) / tilingData->seq + (tilingData->seq == 1 ? 0 : 1));
        int64_t loopOfCp = BatchRowsNum / rowOfBuffer;
        int64_t remainderRow = BatchRowsNum - loopOfCp * rowOfBuffer;
        for(int i = 0; i < loopOfCp; ++i){
            CastWithCopyOutSWT(i, rowOfBuffer, rowOfBuffer, tilingData);
        }
        if(likely(remainderRow != 0)){
        CastWithCopyOutSWT(loopOfCp, rowOfBuffer, remainderRow, tilingData);
        }
    }

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::CastWithCopyOutSWT(
    const int64_t outIdx, const int64_t rowOffset, const int64_t rowOfCp, const AdaLayerNormGradTilingDataCommon* tilingData)
{
    buffer0_ = queue0_.AllocTensor<float>();
    buffer1_ = queue1_.AllocTensor<float>();
    buffer7_ = queue7_.AllocTensor<float>();
    buffer8_ = queue8_.AllocTensor<float>();

    DataCopyExtParams intriParamsIn = {1, static_cast<uint32_t>(rowOfCp * tilingData->colAlignV * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams padParams = {false, 0, 0, 0.0f};
    int64_t offset = outIdx * rowOffset * tilingData->colAlignV;

    event_t event0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(event0);
    WaitFlag<HardEvent::V_MTE2>(event0);

    DataCopyPad(buffer0_, dScaleWorkspaceGM[offset], intriParamsIn, padParams);
    queue0_.EnQue(buffer0_);
    DataCopyPad(buffer1_, dShiftWorkspaceGM[offset], intriParamsIn, padParams);
    queue1_.EnQue(buffer1_);

    event_t event1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(event1);
    WaitFlag<HardEvent::MTE2_V>(event1);
    buffer0_ = queue0_.DeQue<float>();
    buffer1_ = queue1_.DeQue<float>();

    Adds(buffer7_, buffer0_, 0.0f, rowOfCp * tilingData->colAlignV);
    PipeBarrier<PIPE_V>();
    Adds(buffer8_, buffer1_, 0.0f, rowOfCp * tilingData->colAlignV);
    PipeBarrier<PIPE_V>();

    if constexpr (!IsSameType<T, float>::value) {
        PipeBarrier<PIPE_V>();
        CastToB16(buffer7_, buffer11_, rowOfCp, tilingData);
        PipeBarrier<PIPE_V>();
        CastToB16(buffer8_, buffer11_, rowOfCp, tilingData);
    }

    event_t event2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(event2);
    WaitFlag<HardEvent::V_MTE3>(event2);

    DataCopyParams intriParams;
    if (likely(tilingData->colAlignV == tilingData->col)) {
        intriParams.blockCount = 1;
        intriParams.blockLen = rowOfCp * tilingData->colAlignV * sizeof(T);
    } else {
        intriParams.blockCount = rowOfCp;
        intriParams.blockLen = tilingData->col * sizeof(T);
    }
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;

    queue7_.EnQue(buffer7_);
    buffer7_ = queue7_.DeQue<float>();
    DataCopyPad(pdScaleOutTensorGM[(outIdx * rowOffset) * tilingData->col], buffer7_.ReinterpretCast<T>(), intriParams);

    queue8_.EnQue(buffer8_);
    buffer8_ = queue8_.DeQue<float>();
    DataCopyPad(pdShiftOutTensorGM[(outIdx * rowOffset) * tilingData->col], buffer8_.ReinterpretCast<T>(), intriParams);
    queue0_.FreeTensor(buffer0_);
    queue1_.FreeTensor(buffer1_);
    queue7_.FreeTensor(buffer7_);
    queue8_.FreeTensor(buffer8_);
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::CopyOutPhase2(
    const AdaLayerNormGradTilingDataCommon* tilingData)
{
    if constexpr (!IsSameType<U, float>::value) {
        PipeBarrier<PIPE_V>();
        CastToB16(buffer9_, buffer11_, 1, tilingData);
        PipeBarrier<PIPE_V>();
        CastToB16(buffer10_, buffer11_, 1, tilingData);
    }

    event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(event);
    WaitFlag<HardEvent::V_MTE3>(event);

    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = tilingData->col * sizeof(U);
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;
    SetAtomicAdd<float>();

    queue9_.EnQue(buffer9_);
    buffer9_ = queue9_.DeQue<float>();
    DataCopyPad(pdGammaOutTensorGM, buffer9_.ReinterpretCast<U>(), intriParams);
    queue9_.FreeTensor(buffer9_);

    queue10_.EnQue(buffer10_);
    buffer10_ = queue10_.DeQue<float>();
    DataCopyPad(pdBetaOutTensorGM, buffer10_.ReinterpretCast<U>(), intriParams);
    queue10_.FreeTensor(buffer10_);

    SetAtomicNone();
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::CopyOutPhase2Deterministic(
    const AdaLayerNormGradTilingDataCommon* tilingData)
{
    event_t event = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(event);
    WaitFlag<HardEvent::V_MTE3>(event);

    if (GetBlockIdx() < tilingData->blockNum) {
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = tilingData->colAlignV * sizeof(float);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        queue9_.EnQue(buffer9_);
        buffer9_ = queue9_.DeQue<float>();
        DataCopyPad(workspaceGM, buffer9_, intriParams);
        queue9_.FreeTensor(buffer9_);

        queue10_.EnQue(buffer10_);
        buffer10_ = queue10_.DeQue<float>();
        DataCopyPad(workspaceGM[tilingData->colAlignV], buffer10_, intriParams);
        queue10_.FreeTensor(buffer10_);
    }

    PipeBarrier<PIPE_ALL>();
    pipe.Reset();
    SyncAll();

    AdaLayerNormGradDeterminsticCompute<U> op;
    int64_t curWorkspaceRowsNum = 2 * tilingData->blockFormer / tilingData->seq + COMMON_CONSTANT_TWO;
    op.initBuffer(pipe, pdGammaOutTensorGM, pdBetaOutTensorGM, workspaceGMOri, curWorkspaceRowsNum);
    op.FinalProcessDeterministic(tilingData->colAlignV, tilingData->blockNum, tilingData->col);
}

} // namespace AdaLayerNormGrad

#endif