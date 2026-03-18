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
 * \file ada_layer_norm_grad_common_utils.inl
 * \brief
 */


#ifndef ADA_LAYER_NORM_GRAD_COMMON_UTILS_INL
#define ADA_LAYER_NORM_GRAD_COMMON_UTILS_INL

namespace AdaLayerNormGrad {

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::CastToFloat(
    const LocalTensor<float>& buffer, const LocalTensor<float>& tmpBuffer, const int64_t curRowsNum,
    const AdaLayerNormGradTilingDataCommon* tilingData, const int64_t bufferElemNums)
{
    if (tilingData->colAlignM == tilingData->colAlignV || tilingData->colAlignV == tilingData->col) {
        Cast(
            buffer, buffer.ReinterpretCast<T>()[bufferElemNums], RoundMode::CAST_NONE,
            curRowsNum * tilingData->colAlignV);
    } else {
        DataCopyParams copyIntriParams;
        copyIntriParams.blockCount = 1;
        copyIntriParams.blockLen = curRowsNum * tilingData->colAlignM / COMMON_CONSTANT_SIXTEEN;
        copyIntriParams.srcStride = 0;
        copyIntriParams.dstStride = 0;
        DataCopy(
            tmpBuffer.ReinterpretCast<T>(), buffer.ReinterpretCast<T>()[bufferElemNums], copyIntriParams);
        PipeBarrier<PIPE_V>();

        int64_t formerColLoops = tilingData->colAlignV / COMMON_B32_REPEAT_SIZE;
        int64_t remainderCol = tilingData->colAlignV - formerColLoops * COMMON_B32_REPEAT_SIZE;
        int64_t repeatLoops = curRowsNum / COMMON_MAX_REPEAT;
        int64_t remainderRepeat = curRowsNum - repeatLoops * COMMON_MAX_REPEAT;

        UnaryRepeatParams intriParams;
        intriParams.dstBlkStride = 1;
        intriParams.srcBlkStride = 1;
        intriParams.dstRepStride = tilingData->colAlignV / COMMON_CONSTANT_EIGHT;
        intriParams.srcRepStride = tilingData->colAlignM / COMMON_CONSTANT_SIXTEEN;

        for (int64_t i = 0; i < repeatLoops; i++) {
            int64_t srcRepeatOffset = i * COMMON_MAX_REPEAT * tilingData->colAlignM;
            int64_t dstRepeatOffset = i * COMMON_MAX_REPEAT * tilingData->colAlignV;
            for (int64_t j = 0; j < formerColLoops; j++) {
                int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                Cast(
                    buffer[dstRepeatOffset + colOffset],
                    tmpBuffer.ReinterpretCast<T>()[srcRepeatOffset + colOffset], RoundMode::CAST_NONE,
                    COMMON_B32_REPEAT_SIZE, COMMON_MAX_REPEAT, intriParams);
            }
            if (likely(remainderCol != 0)) {
                int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
                Cast(
                    buffer[dstRepeatOffset + colOffset],
                    tmpBuffer.ReinterpretCast<T>()[srcRepeatOffset + colOffset], RoundMode::CAST_NONE,
                    remainderCol, COMMON_MAX_REPEAT, intriParams);
            }
        }

        if (likely(remainderRepeat != 0)) {
            int64_t srcRepeatOffset = repeatLoops * COMMON_MAX_REPEAT * tilingData->colAlignM;
            int64_t dstRepeatOffset = repeatLoops * COMMON_MAX_REPEAT * tilingData->colAlignV;
            for (int64_t j = 0; j < formerColLoops; j++) {
                int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                Cast(
                    buffer[dstRepeatOffset + colOffset],
                    tmpBuffer.ReinterpretCast<T>()[srcRepeatOffset + colOffset], RoundMode::CAST_NONE,
                    COMMON_B32_REPEAT_SIZE, remainderRepeat, intriParams);
            }
            if (likely(remainderCol != 0)) {
                int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
                Cast(
                    buffer[dstRepeatOffset + colOffset],
                    tmpBuffer.ReinterpretCast<T>()[srcRepeatOffset + colOffset], RoundMode::CAST_NONE,
                    remainderCol, remainderRepeat, intriParams);
            }
        }
    }
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::CastToB16(
    const LocalTensor<float>& buffer, const LocalTensor<float>& tmpBuffer, const int64_t curRowsNum,
    const AdaLayerNormGradTilingDataCommon* tilingData)
{
    RoundMode b16RoundMode = IsSameType<T, bfloat16_t>::value ? RoundMode::CAST_ROUND : RoundMode::CAST_NONE;
    if (tilingData->colAlignM == tilingData->colAlignV || tilingData->colAlignV == tilingData->col) {
        Cast(buffer.ReinterpretCast<T>(), buffer, b16RoundMode, curRowsNum * tilingData->colAlignV);
    } else {
        DataCopyParams copyIntriParams;
        copyIntriParams.blockCount = 1;
        copyIntriParams.blockLen = curRowsNum * tilingData->colAlignV / COMMON_CONSTANT_EIGHT;
        copyIntriParams.srcStride = 0;
        copyIntriParams.dstStride = 0;
        DataCopy(tmpBuffer, buffer, copyIntriParams);
        PipeBarrier<PIPE_V>();

        int64_t formerColLoops = tilingData->colAlignV / COMMON_B32_REPEAT_SIZE;
        int64_t remainderCol = tilingData->colAlignV - formerColLoops * COMMON_B32_REPEAT_SIZE;
        int64_t repeatLoops = curRowsNum / COMMON_MAX_REPEAT;
        int64_t remainderRepeat = curRowsNum - repeatLoops * COMMON_MAX_REPEAT;

        UnaryRepeatParams intriParams;
        intriParams.dstBlkStride = 1;
        intriParams.srcBlkStride = 1;
        intriParams.dstRepStride = tilingData->colAlignM / COMMON_CONSTANT_SIXTEEN;
        intriParams.srcRepStride = tilingData->colAlignV / COMMON_CONSTANT_EIGHT;

        for (int64_t i = 0; i < repeatLoops; i++) {
            int64_t srcRepeatOffset = i * COMMON_MAX_REPEAT * tilingData->colAlignV;
            int64_t dstRepeatOffset = i * COMMON_MAX_REPEAT * tilingData->colAlignM;
            for (int64_t j = 0; j < formerColLoops; j++) {
                int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                Cast(
                    buffer.ReinterpretCast<T>()[dstRepeatOffset + colOffset],
                    tmpBuffer[srcRepeatOffset + colOffset], b16RoundMode, COMMON_B32_REPEAT_SIZE,
                    COMMON_MAX_REPEAT, intriParams);
            }
            if (likely(remainderCol != 0)) {
                int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
                Cast(
                    buffer.ReinterpretCast<T>()[dstRepeatOffset + colOffset],
                    tmpBuffer[srcRepeatOffset + colOffset], b16RoundMode, remainderCol,
                    COMMON_MAX_REPEAT, intriParams);
            }
        }

        if (likely(remainderRepeat != 0)) {
            int64_t srcRepeatOffset = repeatLoops * COMMON_MAX_REPEAT * tilingData->colAlignV;
            int64_t dstRepeatOffset = repeatLoops * COMMON_MAX_REPEAT * tilingData->colAlignM;
            for (int64_t j = 0; j < formerColLoops; j++) {
                int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                Cast(
                    buffer.ReinterpretCast<T>()[dstRepeatOffset + colOffset],
                    tmpBuffer[srcRepeatOffset + colOffset], b16RoundMode, COMMON_B32_REPEAT_SIZE,
                    remainderRepeat, intriParams);
            }
            if (likely(remainderCol != 0)) {
                int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
                Cast(
                    buffer.ReinterpretCast<T>()[dstRepeatOffset + colOffset],
                    tmpBuffer[srcRepeatOffset + colOffset], b16RoundMode, remainderCol,
                    remainderRepeat, intriParams);
            }
        }
    }
}

template <typename T, typename U, bool isDeterministic>
template <typename dType>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::BlockBroadcast(
    const LocalTensor<dType>& dst, const LocalTensor<dType>& src, const int64_t curRowsNum)
{
    Brcb(dst, src, (curRowsNum + COMMON_CONSTANT_EIGHT - 1) / COMMON_CONSTANT_EIGHT, {1, COMMON_CONSTANT_EIGHT});
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::BinElemWithInlinedLastBrcFP32(
    const LocalTensor<float>& output, const LocalTensor<float>& input0, const LocalTensor<float>& input1,
    const int64_t rows, const AdaLayerNormGradTilingDataCommon* tiling,
    void (*func)(
        const LocalTensor<float>&, const LocalTensor<float>&, const LocalTensor<float>&, uint64_t, uint8_t,
        const BinaryRepeatParams&))
{
    int64_t colMain = tiling->colAlignV / COMMON_B32_REPEAT_SIZE;
    int64_t colRemain = tiling->colAlignV - colMain * COMMON_B32_REPEAT_SIZE;
    int64_t rowMain = rows / COMMON_MAX_REPEAT;
    int64_t rowRemain = rows - rowMain * COMMON_MAX_REPEAT;

    BinaryRepeatParams params;
    params.dstBlkStride = 1;
    params.src0BlkStride = 1;
    params.src1BlkStride = 0;
    params.dstRepStride = tiling->colAlignV / COMMON_CONSTANT_EIGHT;
    params.src0RepStride = tiling->colAlignV / COMMON_CONSTANT_EIGHT;
    params.src1RepStride = 1;

    for (int64_t i = 0; i < rowMain; i++) {
        int64_t rowOffset = i * COMMON_MAX_REPEAT * tiling->colAlignV;
        for (int64_t j = 0; j < colMain; j++) {
            int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
            func(
                output[rowOffset + colOffset], input0[rowOffset + colOffset],
                input1[i * COMMON_MAX_REPEAT * COMMON_B32_BLOCK_SIZE], COMMON_B32_REPEAT_SIZE,
                COMMON_MAX_REPEAT, params);
        }
        if (likely(colRemain != 0)) {
            int64_t colOffset = colMain * COMMON_B32_REPEAT_SIZE;
            func(
                output[rowOffset + colOffset], input0[rowOffset + colOffset],
                input1[i * COMMON_MAX_REPEAT * COMMON_B32_BLOCK_SIZE], colRemain,
                COMMON_MAX_REPEAT, params);
        }
    }

    if (likely(rowRemain != 0)) {
        int64_t rowOffset = rowMain * COMMON_MAX_REPEAT * tiling->colAlignV;
        for (int64_t j = 0; j < colMain; j++) {
            int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
            func(
                output[rowOffset + colOffset], input0[rowOffset + colOffset],
                input1[rowMain * COMMON_MAX_REPEAT * COMMON_B32_BLOCK_SIZE], COMMON_B32_REPEAT_SIZE,
                rowRemain, params);
        }
        if (likely(colRemain != 0)) {
            int64_t colOffset = colMain * COMMON_B32_REPEAT_SIZE;
            func(
                output[rowOffset + colOffset], input0[rowOffset + colOffset],
                input1[rowMain * COMMON_MAX_REPEAT * COMMON_B32_BLOCK_SIZE], colRemain,
                rowRemain, params);
        }
    }
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::BinElemWithInlinedNLastBrcFP32(
    const LocalTensor<float>& dst, const LocalTensor<float>& src0, const LocalTensor<float>& src1,
    const int64_t curRowsNum, const AdaLayerNormGradTilingDataCommon* tilingData,
    void (*func)(
        const LocalTensor<float>&, const LocalTensor<float>&, const LocalTensor<float>&, uint64_t, uint8_t,
        const BinaryRepeatParams&))
{
    int64_t formerColLoops = tilingData->colAlignV / COMMON_B32_REPEAT_SIZE;
    int64_t remainderCol = tilingData->colAlignV - formerColLoops * COMMON_B32_REPEAT_SIZE;
    int64_t repeatLoops = curRowsNum / COMMON_MAX_REPEAT;
    int64_t remainderRepeat = curRowsNum - repeatLoops * COMMON_MAX_REPEAT;

    BinaryRepeatParams intriParams;
    intriParams.dstBlkStride = 1;
    intriParams.src0BlkStride = 1;
    intriParams.src1BlkStride = 1;
    intriParams.dstRepStride = tilingData->colAlignV / COMMON_CONSTANT_EIGHT;
    intriParams.src0RepStride = tilingData->colAlignV / COMMON_CONSTANT_EIGHT;
    intriParams.src1RepStride = 0;

    for (int64_t i = 0; i < repeatLoops; i++) {
        int64_t repeatOffset = i * COMMON_MAX_REPEAT * tilingData->colAlignV;
        for (int64_t j = 0; j < formerColLoops; j++) {
            int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
            func(
                dst[repeatOffset + colOffset], src0[repeatOffset + colOffset], src1[colOffset],
                COMMON_B32_REPEAT_SIZE, COMMON_MAX_REPEAT, intriParams);
        }
        if (likely(remainderCol != 0)) {
            int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
            func(
                dst[repeatOffset + colOffset], src0[repeatOffset + colOffset], src1[colOffset],
                remainderCol, COMMON_MAX_REPEAT, intriParams);
        }
    }

    if (likely(remainderRepeat != 0)) {
        int64_t repeatOffset = repeatLoops * COMMON_MAX_REPEAT * tilingData->colAlignV;
        for (int64_t j = 0; j < formerColLoops; j++) {
            int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
            func(
                dst[repeatOffset + colOffset], src0[repeatOffset + colOffset], src1[colOffset],
                COMMON_B32_REPEAT_SIZE, remainderRepeat, intriParams);
        }
        if (likely(remainderCol != 0)) {
            int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
            func(
                dst[repeatOffset + colOffset], src0[repeatOffset + colOffset], src1[colOffset],
                remainderCol, remainderRepeat, intriParams);
        }
    }
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::NlastBatchReduceSum(
    const LocalTensor<float>& dst, const LocalTensor<float>& src, const int64_t curRowsNum,
    const int64_t startRow, const AdaLayerNormGradTilingDataCommon* tilingData)
{
    int64_t computeStartBatch = startRow / tilingData->seq;

    for (int64_t i = 0; i < curRowsNum; i++) {
        int64_t curBatch = (startRow + i) / tilingData->seq;
        int64_t batchOffset = curBatch - computeStartBatch;
        int64_t dstOffset = batchOffset * tilingData->colAlignV;
        int64_t srcOffset = i * tilingData->colAlignV;
        
        Add(dst[dstOffset], src[srcOffset], dst[dstOffset], tilingData->colAlignV);
    }
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::NlastBatchMul(
    const LocalTensor<float>& dst, const LocalTensor<float>& src0, const LocalTensor<float>& src1,
    const int64_t curRowsNum, const int64_t startRow, const AdaLayerNormGradTilingDataCommon* tilingData)
{
    int64_t computeStartBatch = startRow / tilingData->seq;

    for (int64_t i = 0; i < curRowsNum; i++) {
        int64_t curBatch = (startRow + i) / tilingData->seq;
        int64_t batchOffset = curBatch - computeStartBatch;
        int64_t srcOffset = batchOffset * tilingData->colAlignV;
        int64_t dstOffset = i * tilingData->colAlignV;

        Mul(dst[dstOffset], src0[dstOffset], src1[srcOffset], tilingData->colAlignV);
    }
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::NlastReduceSum(
    const LocalTensor<float>& dst, const LocalTensor<float>& src, const int64_t curRowsNum,
    const AdaLayerNormGradTilingDataCommon* tilingData)
{
    int64_t formerColLoops = tilingData->colAlignV / COMMON_B32_REPEAT_SIZE;
    int64_t remainderCol = tilingData->colAlignV - formerColLoops * COMMON_B32_REPEAT_SIZE;
    int64_t repeatLoops = curRowsNum / COMMON_MAX_REPEAT;
    int64_t remainderRepeat = curRowsNum - repeatLoops * COMMON_MAX_REPEAT;

    BinaryRepeatParams intriParams;
    intriParams.dstBlkStride = 1;
    intriParams.src0BlkStride = 1;
    intriParams.src1BlkStride = 1;
    intriParams.dstRepStride = 0;
    intriParams.src0RepStride = tilingData->colAlignV / COMMON_CONSTANT_EIGHT;
    intriParams.src1RepStride = 0;

    for (int64_t i = 0; i < repeatLoops; i++) {
        int64_t repeatOffset = i * COMMON_MAX_REPEAT * tilingData->colAlignV;
        for (int64_t j = 0; j < formerColLoops; j++) {
            int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
            Add(
                dst[colOffset], src[repeatOffset + colOffset], dst[colOffset],
                COMMON_B32_REPEAT_SIZE, COMMON_MAX_REPEAT, intriParams);
        }
        if (likely(remainderCol != 0)) {
            int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
            Add(
                dst[colOffset], src[repeatOffset + colOffset], dst[colOffset],
                remainderCol, COMMON_MAX_REPEAT, intriParams);
        }
        PipeBarrier<PIPE_V>();
    }

    if (likely(remainderRepeat != 0)) {
        int64_t repeatOffset = repeatLoops * COMMON_MAX_REPEAT * tilingData->colAlignV;
        for (int64_t j = 0; j < formerColLoops; j++) {
            int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
            Add(
                dst[colOffset], src[repeatOffset + colOffset], dst[colOffset],
                COMMON_B32_REPEAT_SIZE, remainderRepeat, intriParams);
        }
        if (likely(remainderCol != 0)) {
            int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
            Add(
                dst[colOffset], src[repeatOffset + colOffset], dst[colOffset],
                remainderCol, remainderRepeat, intriParams);
        }
    }
}

template <typename T, typename U, bool isDeterministic>
__aicore__ inline void AdaLayerNormGradCommon<T, U, isDeterministic>::LastReduceSum(
    const LocalTensor<float>& dst, const LocalTensor<float>& src, const LocalTensor<float>& tmp,
    const int64_t curRowsNum, const AdaLayerNormGradTilingDataCommon* tilingData)
{
    if (tilingData->colAlignV <= COMMON_B32_REPEAT_SIZE) {
        int64_t repeatLoops = curRowsNum / COMMON_VC_MAX_REPEAT;
        int64_t remainderRepeat = curRowsNum - repeatLoops * COMMON_VC_MAX_REPEAT;

        for (int64_t i = 0; i < repeatLoops; i++) {
            WholeReduceSum(
                dst[i * COMMON_VC_MAX_REPEAT],
                src[i * COMMON_VC_MAX_REPEAT * tilingData->colAlignV],
                tilingData->colAlignV, COMMON_VC_MAX_REPEAT, 1, 1,
                tilingData->colAlignV / COMMON_CONSTANT_EIGHT);
        }

        if (likely(remainderRepeat != 0)) {
            WholeReduceSum(
                dst[repeatLoops * COMMON_VC_MAX_REPEAT],
                src[repeatLoops * COMMON_VC_MAX_REPEAT * tilingData->colAlignV],
                tilingData->colAlignV, remainderRepeat, 1, 1,
                tilingData->colAlignV / COMMON_CONSTANT_EIGHT);
        }
    } else {
        DataCopyParams copyIntriParams;
        copyIntriParams.blockCount = curRowsNum;
        copyIntriParams.blockLen = COMMON_CONSTANT_EIGHT;
        copyIntriParams.srcStride = tilingData->colAlignV / COMMON_CONSTANT_EIGHT - COMMON_CONSTANT_EIGHT;
        copyIntriParams.dstStride = 0;

        DataCopy(tmp, src, copyIntriParams);
        PipeBarrier<PIPE_V>();

        int64_t formerColLoops = tilingData->colAlignV / COMMON_B32_REPEAT_SIZE;
        int64_t remainderCol = tilingData->colAlignV - formerColLoops * COMMON_B32_REPEAT_SIZE;

        BinaryRepeatParams intriParams;
        intriParams.dstBlkStride = 1;
        intriParams.src0BlkStride = 1;
        intriParams.src1BlkStride = 1;
        intriParams.dstRepStride = COMMON_CONSTANT_EIGHT;
        intriParams.src0RepStride = COMMON_CONSTANT_EIGHT;
        intriParams.src1RepStride = tilingData->colAlignV / COMMON_CONSTANT_EIGHT;

        for (int64_t i = 1; i < formerColLoops; i++) {
            Add(tmp, tmp, src[i * COMMON_B32_REPEAT_SIZE], COMMON_B32_REPEAT_SIZE, curRowsNum, intriParams);
            PipeBarrier<PIPE_V>();
        }
        if (likely(remainderCol != 0)) {
            Add(tmp, tmp, src[formerColLoops * COMMON_B32_REPEAT_SIZE], remainderCol, curRowsNum, intriParams);
        }
        PipeBarrier<PIPE_V>();

        WholeReduceSum(dst, tmp, COMMON_B32_REPEAT_SIZE, curRowsNum, 1, 1, COMMON_CONSTANT_EIGHT);
    }
}

} // namespace AdaLayerNormGrad

#endif