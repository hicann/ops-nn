/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file ada_layer_norm_grad_bs_common_output.inl
 * \brief
 */


#ifndef ADA_LAYER_NORM_GRAD_MERGE_BS_COMMON_OUTPUT_INL
#define ADA_LAYER_NORM_GRAD_MERGE_BS_COMMON_OUTPUT_INL

namespace AdaLayerNormGrad {

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradMergeBSCommon<T, U, isDeterministic>::CopyOutPhase0(
    const int64_t outerIdx, const int64_t curRowsNum, const AdaLayerNormGradTilingDataMergeBSCommon* tilingData)
{
    event_t event0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(event0);
    WaitFlag<HardEvent::V_MTE3>(event0);

    buffer6_ = queue6_.DeQue<float>();
    DataCopyParams intriParams;
    if (likely(tilingData->colAlignV == tilingData->col)) {
        intriParams.blockCount = 1;
        intriParams.blockLen = curRowsNum * tilingData->col * sizeof(T);
    } else {
        intriParams.blockLen = tilingData->col * sizeof(T);
        intriParams.blockCount = curRowsNum;
    }
    intriParams.dstStride = 0;
    intriParams.srcStride = 0;

    DataCopyPad(
        pdXOutTensorGM_[tilingData->ubFormer * tilingData->col * outerIdx], buffer6_.ReinterpretCast<T>(), intriParams);
    queue6_.FreeTensor(buffer6_);
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradMergeBSCommon<T, U, isDeterministic>::CopyOutPhase1(
    const int64_t startRow, const AdaLayerNormGradTilingDataMergeBSCommon* tilingData)
{
    event_t event0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(event0);
    WaitFlag<HardEvent::V_MTE3>(event0);

    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = ScaleReRowNum_ * tilingData->colAlignV * sizeof(float);
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;
    SetAtomicAdd<float>();

    int64_t cpOutOffset = startRow / tilingData->seq * tilingData->colAlignV;

    queue8_.EnQue(buffer8_);
    buffer8_ = queue8_.DeQue<float>();
    DataCopyPad(dShiftWorkspaceGM[cpOutOffset], buffer8_, intriParams);
    queue8_.FreeTensor(buffer8_);

    queue7_.EnQue(buffer7_);
    buffer7_ = queue7_.DeQue<float>();
    DataCopyPad(dScaleWorkspaceGM[cpOutOffset], buffer7_, intriParams);
    queue7_.FreeTensor(buffer7_);

    SetAtomicNone();
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradMergeBSCommon<T, U, isDeterministic>::CopyOutPhase1Deterministic(
    const AdaLayerNormGradTilingDataMergeBSCommon* tilingData)
{
        int64_t curWorkspaceRowsNum = 2 * tilingData->batch + COMMON_CONSTANT_TWO;

        AdaLayerNormGradDeterminsticCompute<T> op;
        // initBuffer 依然按照原来的方式调用，传入初始的基地址
        op.initBuffer(pipe, pdScaleOutTensorGM, pdShiftOutTensorGM, workspaceGMOri, curWorkspaceRowsNum);

        // 调用新写的 Batch 处理函数
        op.BatchFinalProcessDeterministic(
            tilingData->batch, tilingData->colAlignV, tilingData->blockNum, tilingData->col, pdScaleOutTensorGM,
            pdShiftOutTensorGM);
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradMergeBSCommon<T, U, isDeterministic>::CopyOutPhase2(
    const AdaLayerNormGradTilingDataMergeBSCommon* tilingData)
{
    if constexpr (!IsSameType<U, float>::value) {
        PipeBarrier<PIPE_V>();
        CastToB16(buffer10_, buffer11_, 1, tilingData);
        PipeBarrier<PIPE_V>();
        CastToB16(buffer9_, buffer11_, 1, tilingData);
    }

    event_t event0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(event0);
    WaitFlag<HardEvent::V_MTE3>(event0);

    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = tilingData->col * sizeof(U);
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;
    SetAtomicAdd<float>();

    queue10_.EnQue(buffer10_);
    buffer10_ = queue10_.DeQue<float>();
    DataCopyPad(pdBetaOutTensorGM, buffer10_.ReinterpretCast<U>(), intriParams);
    queue10_.FreeTensor(buffer10_);

    queue9_.EnQue(buffer9_);
    buffer9_ = queue9_.DeQue<float>();
    DataCopyPad(pdGammaOutTensorGM, buffer9_.ReinterpretCast<U>(), intriParams);
    queue9_.FreeTensor(buffer9_);

    SetAtomicNone();
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradMergeBSCommon<T, U, isDeterministic>::CopyOutPhase2Deterministic(
    const AdaLayerNormGradTilingDataMergeBSCommon* tilingData)
{
    event_t event0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(event0);
    WaitFlag<HardEvent::V_MTE3>(event0);

    if (GetBlockIdx() < tilingData->blockNum) {
        DataCopyParams intriParams;
        intriParams.blockCount = 1;
        intriParams.blockLen = tilingData->colAlignV * sizeof(float);
        intriParams.srcStride = 0;
        intriParams.dstStride = 0;

        queue10_.EnQue(buffer10_);
        buffer10_ = queue10_.DeQue<float>();
        DataCopyPad(workspaceGM[tilingData->colAlignV], buffer10_, intriParams);
        queue10_.FreeTensor(buffer10_);

        queue9_.EnQue(buffer9_);
        buffer9_ = queue9_.DeQue<float>();
        DataCopyPad(workspaceGM, buffer9_, intriParams);
        queue9_.FreeTensor(buffer9_);

    }

    PipeBarrier<PIPE_ALL>();
    pipe.Reset();
    SyncAll();

    AdaLayerNormGradDeterminsticCompute<U> op;
    int64_t curWorkspaceRowsNum = 2 * tilingData->batch + COMMON_CONSTANT_TWO;
    op.initBuffer(pipe, pdGammaOutTensorGM, pdBetaOutTensorGM, workspaceGMOri, curWorkspaceRowsNum);
    op.FinalProcessDeterministic(tilingData->colAlignV, tilingData->blockNum, tilingData->col);
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradMergeBSCommon<T, U, isDeterministic>::CastToFloatLargeStride(
    const LocalTensor<float>& dst, const LocalTensor<float>& src, const int64_t curRowsNum, 
    const AdaLayerNormGradTilingDataMergeBSCommon* tilingData, RoundMode castMode)
{
    UnaryRepeatParams intriParamsOne;
    intriParamsOne.dstBlkStride = 1;
    intriParamsOne.srcBlkStride = 1;
    intriParamsOne.dstRepStride = 0;
    intriParamsOne.srcRepStride = 0;

    for (int64_t i = 0; i < curRowsNum; ++i) {
        int64_t srcOffset = i * tilingData->colAlignM;
        int64_t dstOffset = i * tilingData->colAlignV;

        int64_t formerColLoops = tilingData->colAlignV / COMMON_B32_REPEAT_SIZE;
        int64_t remainderCol = tilingData->colAlignV % COMMON_B32_REPEAT_SIZE;

        for (int64_t j = 0; j < formerColLoops; ++j) {
            int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
            Cast(
                dst[dstOffset + colOffset], src.ReinterpretCast<T>()[srcOffset + colOffset],
                castMode, COMMON_B32_REPEAT_SIZE, 1, intriParamsOne);
        }
        if (remainderCol != 0) {
            int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
            Cast(
                dst[dstOffset + colOffset], src.ReinterpretCast<T>()[srcOffset + colOffset],
                castMode, remainderCol, 1, intriParamsOne);
        }
    }
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradMergeBSCommon<T, U, isDeterministic>::CastToFloat(
    const LocalTensor<float>& buffer, const LocalTensor<float>& tmpBuffer, const int64_t curRowsNum,
    const AdaLayerNormGradTilingDataMergeBSCommon* tilingData, const int64_t bufferElemNums)
{
    if (tilingData->colAlignM == tilingData->colAlignV || tilingData->colAlignV == tilingData->col) {
        Cast(
            buffer, buffer.ReinterpretCast<T>()[bufferElemNums], RoundMode::CAST_NONE,
            curRowsNum * tilingData->colAlignV);
    } else {
        DataCopyParams copyIntriParam;
        copyIntriParam.blockCount = 1;
        copyIntriParam.blockLen = curRowsNum * tilingData->colAlignM / COMMON_CONSTANT_SIXTEEN;
        copyIntriParam.dstStride = 0;
        copyIntriParam.srcStride = 0;
        DataCopy(tmpBuffer.ReinterpretCast<T>(), buffer.ReinterpretCast<T>()[bufferElemNums], copyIntriParam);
        PipeBarrier<PIPE_V>();

        if (tilingData->colAlignV / COMMON_CONSTANT_EIGHT > 255 || tilingData->colAlignM / COMMON_CONSTANT_SIXTEEN > 255) {
            CastToFloatLargeStride(buffer, tmpBuffer, curRowsNum, tilingData, RoundMode::CAST_NONE);
        } else {
            int64_t repeatLoops = curRowsNum / COMMON_MAX_REPEAT;
            int64_t remainderRepeat = curRowsNum - repeatLoops * COMMON_MAX_REPEAT;
            int64_t formerColLoops = tilingData->colAlignV / COMMON_B32_REPEAT_SIZE;
            int64_t remainderCol = tilingData->colAlignV - formerColLoops * COMMON_B32_REPEAT_SIZE;

            UnaryRepeatParams intriParam;
            intriParam.dstBlkStride = 1;
            intriParam.srcBlkStride = 1;
            intriParam.dstRepStride = tilingData->colAlignV / COMMON_CONSTANT_EIGHT;
            intriParam.srcRepStride = tilingData->colAlignM / COMMON_CONSTANT_SIXTEEN;

            for (int64_t i = 0; i < repeatLoops; i++) {
                int64_t dstRepOffset = i * COMMON_MAX_REPEAT * tilingData->colAlignV;
                int64_t srcRepOffset = i * COMMON_MAX_REPEAT * tilingData->colAlignM;
                for (int64_t j = 0; j < formerColLoops; j++) {
                    int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                    Cast(
                        buffer[dstRepOffset + colOffset],
                        tmpBuffer.ReinterpretCast<T>()[srcRepOffset + colOffset], RoundMode::CAST_NONE,
                        COMMON_B32_REPEAT_SIZE, COMMON_MAX_REPEAT, intriParam);
                }
                if (likely(remainderCol != 0)) {
                    int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
                    Cast(
                        buffer[dstRepOffset + colOffset],
                        tmpBuffer.ReinterpretCast<T>()[srcRepOffset + colOffset], RoundMode::CAST_NONE, remainderCol,
                        COMMON_MAX_REPEAT, intriParam);
                }
            }

            if (likely(remainderRepeat != 0)) {
                int64_t dstRepOffset = repeatLoops * COMMON_MAX_REPEAT * tilingData->colAlignV;
                int64_t srcRepOffset = repeatLoops * COMMON_MAX_REPEAT * tilingData->colAlignM;
                for (int64_t j = 0; j < formerColLoops; j++) {
                    int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;

                    Cast(
                        buffer[dstRepOffset + colOffset],
                        tmpBuffer.ReinterpretCast<T>()[srcRepOffset + colOffset], RoundMode::CAST_NONE,
                        COMMON_B32_REPEAT_SIZE, remainderRepeat, intriParam);
                }
                if (likely(remainderCol != 0)) {
                    int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
                    Cast(
                        buffer[dstRepOffset + colOffset],
                        tmpBuffer.ReinterpretCast<T>()[srcRepOffset + colOffset], RoundMode::CAST_NONE, remainderCol,
                        remainderRepeat, intriParam);
                }
            }
        }
    }
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradMergeBSCommon<T, U, isDeterministic>::CastTo16LargeStride(
    const LocalTensor<float>& dst, const LocalTensor<float>& src, const int64_t curRowsNum, 
    const AdaLayerNormGradTilingDataMergeBSCommon* tilingData, RoundMode castMode)
{
    UnaryRepeatParams intriParamsOneRow;
    intriParamsOneRow.dstBlkStride = 1;
    intriParamsOneRow.srcBlkStride = 1;
    intriParamsOneRow.dstRepStride = 0;
    intriParamsOneRow.srcRepStride = 0;

    for (int64_t i = 0; i < curRowsNum; ++i) {

        int64_t formerColLoops = tilingData->colAlignV / COMMON_B32_REPEAT_SIZE;
        int64_t remainderCol = tilingData->colAlignV % COMMON_B32_REPEAT_SIZE;
        int64_t srcRowOffset = i * tilingData->colAlignV;
        int64_t dstRowOffset = i * tilingData->colAlignM;

        for (int64_t j = 0; j < formerColLoops; ++j) {
            int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
            Cast(
                dst.ReinterpretCast<T>()[dstRowOffset + colOffset], src[srcRowOffset + colOffset],
                castMode, COMMON_B32_REPEAT_SIZE, 1, intriParamsOneRow);
        }
        if (remainderCol != 0) {
            int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
            Cast(
                dst.ReinterpretCast<T>()[dstRowOffset + colOffset], src[srcRowOffset + colOffset],
                castMode, remainderCol, 1, intriParamsOneRow);
        }
    }
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradMergeBSCommon<T, U, isDeterministic>::CastToB16(
    const LocalTensor<float>& buffer, const LocalTensor<float>& tmpBuffer, const int64_t curRowsNum,
    const AdaLayerNormGradTilingDataMergeBSCommon* tilingData)
{
    RoundMode b16RoundMode = RoundMode::CAST_RINT;
    if (tilingData->colAlignM == tilingData->colAlignV || tilingData->colAlignV == tilingData->col) {
        Cast(buffer.ReinterpretCast<T>(), buffer, b16RoundMode, curRowsNum * tilingData->colAlignV);
    } else {
        DataCopyParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = curRowsNum * tilingData->colAlignV / COMMON_CONSTANT_EIGHT;
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopy(tmpBuffer, buffer, copyInParams);
        PipeBarrier<PIPE_V>();

        if (tilingData->colAlignV / COMMON_CONSTANT_EIGHT > 255 ||
            tilingData->colAlignM / COMMON_CONSTANT_SIXTEEN > 255) {
            CastTo16LargeStride(buffer, tmpBuffer, curRowsNum, tilingData, b16RoundMode);
        } else {
            int64_t repeatLoops = curRowsNum / COMMON_MAX_REPEAT;
            int64_t remainderRepeat = curRowsNum - repeatLoops * COMMON_MAX_REPEAT;
            int64_t formerColLoops = tilingData->colAlignV / COMMON_B32_REPEAT_SIZE;
            int64_t remainderCol = tilingData->colAlignV - formerColLoops * COMMON_B32_REPEAT_SIZE;

            UnaryRepeatParams intriParams;
            intriParams.dstBlkStride = 1;
            intriParams.srcBlkStride = 1;
            intriParams.dstRepStride = tilingData->colAlignM / COMMON_CONSTANT_SIXTEEN;
            intriParams.srcRepStride = tilingData->colAlignV / COMMON_CONSTANT_EIGHT;

            for (int64_t i = 0; i < repeatLoops; i++) {
                int64_t dstRepOffset = i * COMMON_MAX_REPEAT * tilingData->colAlignM;
                int64_t srcRepOffset = i * COMMON_MAX_REPEAT * tilingData->colAlignV;
                for (int64_t j = 0; j < formerColLoops; j++) {
                    int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                    Cast(
                        buffer.ReinterpretCast<T>()[dstRepOffset + colOffset],
                        tmpBuffer[srcRepOffset + colOffset], b16RoundMode, COMMON_B32_REPEAT_SIZE, COMMON_MAX_REPEAT,
                        intriParams);
                }
                if (likely(remainderCol != 0)) {
                    int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
                    Cast(
                        buffer.ReinterpretCast<T>()[dstRepOffset + colOffset],
                        tmpBuffer[srcRepOffset + colOffset], b16RoundMode, remainderCol, COMMON_MAX_REPEAT,
                        intriParams);
                }
            }

            if (likely(remainderRepeat != 0)) {
                int64_t srcRepOffset = repeatLoops * COMMON_MAX_REPEAT * tilingData->colAlignV;
                int64_t dstRepOffset = repeatLoops * COMMON_MAX_REPEAT * tilingData->colAlignM;
                for (int64_t j = 0; j < formerColLoops; j++) {
                    int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                    Cast(
                        buffer.ReinterpretCast<T>()[dstRepOffset + colOffset],
                        tmpBuffer[srcRepOffset + colOffset], b16RoundMode, COMMON_B32_REPEAT_SIZE, remainderRepeat,
                        intriParams);
                }
                if (likely(remainderCol != 0)) {
                    int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
                    Cast(
                        buffer.ReinterpretCast<T>()[dstRepOffset + colOffset],
                        tmpBuffer[srcRepOffset + colOffset], b16RoundMode, remainderCol, remainderRepeat,
                        intriParams);
                }
            }
        }
    }
}


} // namespace AdaLayerNormGrad

#endif