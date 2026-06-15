/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sync_batch_norm_gather_stats_fused_common.inl
 * \brief Inline implementation for SyncBatchNormGatherStatsFusedCommon helper functions
 */

namespace SyncBatchNormGatherStatsFused {
template <typename T>
template <typename dType>
__aicore__ inline void SyncBatchNormGatherStatsFusedCommon<T>::BlockBroadcast(
    const LocalTensor<dType>& dst, const LocalTensor<dType>& src, const int64_t curRowsNum)
{
    constexpr int64_t MAX_REPEAT = 255;
    constexpr int64_t BLOCK_SIZE = COMMON_CONSTANT_EIGHT;

    int64_t totalBlocks = (curRowsNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int64_t processedBlocks = 0;

    while (processedBlocks < totalBlocks) {
        int64_t remainingBlocks = totalBlocks - processedBlocks;
        int64_t curRepeat = (remainingBlocks < MAX_REPEAT) ? remainingBlocks : MAX_REPEAT;
        int64_t srcOffset = processedBlocks * BLOCK_SIZE;

        Brcb(dst[srcOffset * BLOCK_SIZE], src[srcOffset], curRepeat, {1, BLOCK_SIZE});
        processedBlocks += curRepeat;
    }
}

template <typename T>
__aicore__ inline void SyncBatchNormGatherStatsFusedCommon<T>::BinElemNLastBrcLargeStride(
    const LocalTensor<float>& dst, const LocalTensor<float>& src0, const LocalTensor<float>& src1,
    const int64_t curRowsNum,
    void (*func)(const LocalTensor<float>&, const LocalTensor<float>&, const LocalTensor<float>&, uint64_t, uint8_t, const BinaryRepeatParams&))
{
    BinaryRepeatParams paramsOne;
    paramsOne.dstRepStride = 0;
    paramsOne.src0RepStride = 0;
    paramsOne.src1RepStride = 0;
    paramsOne.dstBlkStride = 1;
    paramsOne.src0BlkStride = 1;
    paramsOne.src1BlkStride = 1;

    int64_t formerColLoops = cAlign / COMMON_B32_REPEAT_SIZE;
    int64_t remainderCol = cAlign % COMMON_B32_REPEAT_SIZE;

    for (int64_t i = 0; i < curRowsNum; i++) {
        int64_t rowOffset = i * cAlign;
        for (int64_t j = 0; j < formerColLoops; j++) {
            int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
            func(dst[rowOffset + colOffset], src0[rowOffset + colOffset], src1[colOffset], COMMON_B32_REPEAT_SIZE, 1, paramsOne);
        }
        if (remainderCol != 0) {
            int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
            func(dst[rowOffset + colOffset], src0[rowOffset + colOffset], src1[colOffset], remainderCol, 1, paramsOne);
        }
    }
}

template <typename T>
__aicore__ inline void SyncBatchNormGatherStatsFusedCommon<T>::BinElemWithInlinedNLastBrcFP32(
    const LocalTensor<float>& dst, const LocalTensor<float>& src0, const LocalTensor<float>& src1,
    const int64_t curRowsNum,
    void (*func)(const LocalTensor<float>&, const LocalTensor<float>&, const LocalTensor<float>&, uint64_t, uint8_t, const BinaryRepeatParams&))
{
    if (unlikely(cAlign / COMMON_CONSTANT_EIGHT > 255)) {
        BinElemNLastBrcLargeStride(dst, src0, src1, curRowsNum, func);
    } else {
        int64_t formerColLoops = cAlign / COMMON_B32_REPEAT_SIZE;
        int64_t remainderCol = cAlign - formerColLoops * COMMON_B32_REPEAT_SIZE;
        int64_t repeatLoops = curRowsNum / COMMON_MAX_REPEAT;
        int64_t remainderRepeat = curRowsNum - repeatLoops * COMMON_MAX_REPEAT;

        BinaryRepeatParams intriParams;
        intriParams.dstBlkStride = 1;
        intriParams.src0BlkStride = 1;
        intriParams.src1BlkStride = 1;
        intriParams.dstRepStride = cAlign / COMMON_CONSTANT_EIGHT;
        intriParams.src0RepStride = cAlign / COMMON_CONSTANT_EIGHT;
        intriParams.src1RepStride = 0;

        for (int64_t i = 0; i < repeatLoops; i++) {
            int64_t repeatOffset = i * COMMON_MAX_REPEAT * cAlign;
            for (int64_t j = 0; j < formerColLoops; j++) {
                int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                func(dst[repeatOffset + colOffset], src0[repeatOffset + colOffset], src1[colOffset], COMMON_B32_REPEAT_SIZE, COMMON_MAX_REPEAT, intriParams);
            }
            if (likely(remainderCol != 0)) {
                int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
                func(dst[repeatOffset + colOffset], src0[repeatOffset + colOffset], src1[colOffset], remainderCol, COMMON_MAX_REPEAT, intriParams);
            }
        }

        if (likely(remainderRepeat != 0)) {
            int64_t repeatOffset = repeatLoops * COMMON_MAX_REPEAT * cAlign;
            for (int64_t j = 0; j < formerColLoops; j++) {
                int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                func(dst[repeatOffset + colOffset], src0[repeatOffset + colOffset], src1[colOffset], COMMON_B32_REPEAT_SIZE, remainderRepeat, intriParams);
            }
            if (likely(remainderCol != 0)) {
                int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
                func(dst[repeatOffset + colOffset], src0[repeatOffset + colOffset], src1[colOffset], remainderCol, remainderRepeat, intriParams);
            }
        }
    }
}

template <typename T>
__aicore__ inline void SyncBatchNormGatherStatsFusedCommon<T>::LastBrcFP32LargeStride(
    const LocalTensor<float>& output, const LocalTensor<float>& input0, const LocalTensor<float>& input1,
    const int64_t curRowsNum,
    void (*func)(const LocalTensor<float>&, const LocalTensor<float>&, const LocalTensor<float>&, uint64_t, uint8_t, const BinaryRepeatParams&))
{
    BinaryRepeatParams paramsOneRow;
    paramsOneRow.dstBlkStride = 1;
    paramsOneRow.src0BlkStride = 1;
    paramsOneRow.src1BlkStride = 0;
    paramsOneRow.dstRepStride = 0;
    paramsOneRow.src0RepStride = 0;
    paramsOneRow.src1RepStride = 0;

    int64_t colMain = cAlign / COMMON_B32_REPEAT_SIZE;
    int64_t colRemain = cAlign % COMMON_B32_REPEAT_SIZE;

    for (int64_t i = 0; i < curRowsNum; i++) {
        int64_t src1Offset = i * 8;
        int64_t rowOffset = i * cAlign;

        for (int64_t j = 0; j < colMain; j++) {
            int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
            func(output[rowOffset + colOffset], input0[rowOffset + colOffset], input1[src1Offset], COMMON_B32_REPEAT_SIZE, 1, paramsOneRow);
        }
        if (colRemain != 0) {
            int64_t colOffset = colMain * COMMON_B32_REPEAT_SIZE;
            func(output[rowOffset + colOffset], input0[rowOffset + colOffset], input1[src1Offset], colRemain, 1, paramsOneRow);
        }
    }
}

template <typename T>
__aicore__ inline void SyncBatchNormGatherStatsFusedCommon<T>::BinElemWithInlinedLastBrcFP32(
    const LocalTensor<float>& output, const LocalTensor<float>& input0, const LocalTensor<float>& input1,
    const int64_t rows,
    void (*func)(const LocalTensor<float>&, const LocalTensor<float>&, const LocalTensor<float>&, uint64_t, uint8_t, const BinaryRepeatParams&))
{
    int64_t rowMain = rows / COMMON_MAX_REPEAT;
    int64_t rowRemain = rows - rowMain * COMMON_MAX_REPEAT;
    int64_t colMain = cAlign / COMMON_B32_REPEAT_SIZE;
    int64_t colRemain = cAlign - colMain * COMMON_B32_REPEAT_SIZE;

    if (cAlign / COMMON_CONSTANT_EIGHT > 255) {
        LastBrcFP32LargeStride(output, input0, input1, rows, func);
    } else {
        BinaryRepeatParams params;
        params.src0BlkStride = 1;
        params.src1BlkStride = 0;
        params.dstBlkStride = 1;
        params.src0RepStride = cAlign / COMMON_CONSTANT_EIGHT;
        params.src1RepStride = 1;
        params.dstRepStride = cAlign / COMMON_CONSTANT_EIGHT;

        for (int64_t i = 0; i < rowMain; i++) {
            int64_t repeatOffset = i * COMMON_MAX_REPEAT * cAlign;
            for (int64_t j = 0; j < colMain; j++) {
                int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                func(output[repeatOffset + colOffset], input0[repeatOffset + colOffset], input1[i * COMMON_MAX_REPEAT * COMMON_B32_BLOCK_SIZE], COMMON_B32_REPEAT_SIZE, COMMON_MAX_REPEAT, params);
            }
            if (likely(colRemain != 0)) {
                int64_t colOffset = colMain * COMMON_B32_REPEAT_SIZE;
                func(output[repeatOffset + colOffset], input0[repeatOffset + colOffset], input1[i * COMMON_MAX_REPEAT * COMMON_B32_BLOCK_SIZE], colRemain, COMMON_MAX_REPEAT, params);
            }
        }

        if (likely(rowRemain != 0)) {
            int64_t repeatOffset = rowMain * COMMON_MAX_REPEAT * cAlign;
            for (int64_t j = 0; j < colMain; j++) {
                int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                func(output[repeatOffset + colOffset], input0[repeatOffset + colOffset], input1[rowMain * COMMON_MAX_REPEAT * COMMON_B32_BLOCK_SIZE], COMMON_B32_REPEAT_SIZE, rowRemain, params);
            }
            if (likely(colRemain != 0)) {
                int64_t colOffset = colMain * COMMON_B32_REPEAT_SIZE;
                func(output[repeatOffset + colOffset], input0[repeatOffset + colOffset], input1[rowMain * COMMON_MAX_REPEAT * COMMON_B32_BLOCK_SIZE], colRemain, rowRemain, params);
            }
        }
    }
}

template <typename T>
__aicore__ inline void SyncBatchNormGatherStatsFusedCommon<T>::NlastReduceSumLargeStride(
    const LocalTensor<float>& dst, const LocalTensor<float>& src, const int64_t curRowsNum)
{
    BinaryRepeatParams paramsOne;
    paramsOne.dstBlkStride = 1;
    paramsOne.src0BlkStride = 1;
    paramsOne.src1BlkStride = 1;
    paramsOne.dstRepStride = 0;
    paramsOne.src0RepStride = 0;
    paramsOne.src1RepStride = 0;

    int64_t formerColLoops = cAlign / COMMON_B32_REPEAT_SIZE;
    int64_t remainderCol = cAlign % COMMON_B32_REPEAT_SIZE;

    for (int64_t i = 0; i < curRowsNum; i++) {
        int64_t srcRowOffset = i * cAlign;

        for (int64_t j = 0; j < formerColLoops; j++) {
            int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
            Add(dst[colOffset], src[srcRowOffset + colOffset], dst[colOffset], COMMON_B32_REPEAT_SIZE, 1, paramsOne);
        }

        if (remainderCol != 0) {
            int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
            Add(dst[colOffset], src[srcRowOffset + colOffset], dst[colOffset], remainderCol, 1, paramsOne);
        }
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void SyncBatchNormGatherStatsFusedCommon<T>::NlastReduceSum(
    const LocalTensor<float>& dst, const LocalTensor<float>& src, const int64_t curRowsNum)
{
    if (cAlign / COMMON_CONSTANT_EIGHT > 255) {
        NlastReduceSumLargeStride(dst, src, curRowsNum);
    } else {
        int64_t formerColLoops = cAlign / COMMON_B32_REPEAT_SIZE;
        int64_t remainderCol = cAlign - formerColLoops * COMMON_B32_REPEAT_SIZE;
        int64_t repeatLoops = curRowsNum / COMMON_MAX_REPEAT;
        int64_t remainderRepeat = curRowsNum - repeatLoops * COMMON_MAX_REPEAT;

        BinaryRepeatParams intriParams;
        intriParams.dstBlkStride = 1;
        intriParams.src0BlkStride = 1;
        intriParams.src1BlkStride = 1;
        intriParams.dstRepStride = 0;
        intriParams.src0RepStride = cAlign / COMMON_CONSTANT_EIGHT;
        intriParams.src1RepStride = 0;

        for (int64_t i = 0; i < repeatLoops; i++) {
            int64_t repeatOffset = i * COMMON_MAX_REPEAT * cAlign;
            for (int64_t j = 0; j < formerColLoops; j++) {
                int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                Add(dst[colOffset], src[repeatOffset + colOffset], dst[colOffset], COMMON_B32_REPEAT_SIZE, COMMON_MAX_REPEAT, intriParams);
            }
            if (likely(remainderCol != 0)) {
                int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
                Add(dst[colOffset], src[repeatOffset + colOffset], dst[colOffset], remainderCol, COMMON_MAX_REPEAT, intriParams);
            }
            PipeBarrier<PIPE_V>();
        }

        if (likely(remainderRepeat != 0)) {
            int64_t repeatOffset = repeatLoops * COMMON_MAX_REPEAT * cAlign;
            for (int64_t j = 0; j < formerColLoops; j++) {
                int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
                Add(dst[colOffset], src[repeatOffset + colOffset], dst[colOffset], COMMON_B32_REPEAT_SIZE, remainderRepeat, intriParams);
            }
            if (likely(remainderCol != 0)) {
                int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
                Add(dst[colOffset], src[repeatOffset + colOffset], dst[colOffset], remainderCol, remainderRepeat, intriParams);
            }
        }
    }
}

template <typename T>
__aicore__ inline void SyncBatchNormGatherStatsFusedCommon<T>::LastReduceSumLargeStride(
    const LocalTensor<float>& tmp, const LocalTensor<float>& src, const int64_t curRowsNum, const int64_t curColNum)
{
    BinaryRepeatParams intriParamsOne;
    intriParamsOne.dstRepStride = 0;
    intriParamsOne.src0RepStride = 0;
    intriParamsOne.src1RepStride = 0;
    intriParamsOne.dstBlkStride = 1;
    intriParamsOne.src0BlkStride = 1;
    intriParamsOne.src1BlkStride = 1;

    int64_t formerColLoops = curColNum / COMMON_B32_REPEAT_SIZE;
    int64_t remainderCol = curColNum % COMMON_B32_REPEAT_SIZE;

    for (int64_t i = 0; i < curRowsNum; i++) {
        int64_t srcOffset = i * curColNum;
        int64_t tmpOffset = i * COMMON_CONSTANT_EIGHT * 8;

        for (int64_t j = 1; j < formerColLoops; j++) {
            int64_t colOffset = j * COMMON_B32_REPEAT_SIZE;
            Add(tmp[tmpOffset], tmp[tmpOffset], src[srcOffset + colOffset], COMMON_B32_REPEAT_SIZE, 1, intriParamsOne);
        }

        if (likely(remainderCol != 0)) {
            int64_t colOffset = formerColLoops * COMMON_B32_REPEAT_SIZE;
            Add(tmp[tmpOffset], tmp[tmpOffset], src[srcOffset + colOffset], remainderCol, 1, intriParamsOne);
        }
    }
}

template <typename T>
__aicore__ inline void SyncBatchNormGatherStatsFusedCommon<T>::LastReduceSum(
    const LocalTensor<float>& dst, const LocalTensor<float>& src, const LocalTensor<float>& tmp,
    const int64_t curRowsNum, const int64_t curColNum)
{
    if (curColNum <= COMMON_B32_REPEAT_SIZE) {
        int64_t repLoops = curRowsNum / COMMON_VC_MAX_REPEAT;
        int64_t remainderRepeat = curRowsNum - repLoops * COMMON_VC_MAX_REPEAT;

        for (int64_t i = 0; i < repLoops; i++) {
            WholeReduceSum(dst[i * COMMON_VC_MAX_REPEAT], src[i * COMMON_VC_MAX_REPEAT * curColNum], curColNum, COMMON_VC_MAX_REPEAT, 1, 1, curColNum / COMMON_CONSTANT_EIGHT);
        }

        if (likely(remainderRepeat != 0)) {
            WholeReduceSum(dst[repLoops * COMMON_VC_MAX_REPEAT], src[repLoops * COMMON_VC_MAX_REPEAT * curColNum], curColNum, remainderRepeat, 1, 1, curColNum / COMMON_CONSTANT_EIGHT);
        }
    } else {
        DataCopyParams copyIntriParams;
        copyIntriParams.blockCount = curRowsNum;
        copyIntriParams.blockLen = COMMON_CONSTANT_EIGHT;
        copyIntriParams.dstStride = 0;
        copyIntriParams.srcStride = curColNum / COMMON_CONSTANT_EIGHT - COMMON_CONSTANT_EIGHT;

        DataCopy(tmp, src, copyIntriParams);
        PipeBarrier<PIPE_V>();

        int64_t formerColLoops = curColNum / COMMON_B32_REPEAT_SIZE;
        int64_t remainderCol = curColNum - formerColLoops * COMMON_B32_REPEAT_SIZE;
        if (curColNum / COMMON_CONSTANT_EIGHT > 255) {
            LastReduceSumLargeStride(tmp, src, curRowsNum, curColNum);
        } else {
            BinaryRepeatParams computeParams;
            computeParams.dstBlkStride = 1;
            computeParams.src0BlkStride = 1;
            computeParams.src1BlkStride = 1;
            computeParams.dstRepStride = COMMON_CONSTANT_EIGHT;
            computeParams.src0RepStride = COMMON_CONSTANT_EIGHT;
            computeParams.src1RepStride = curColNum / COMMON_CONSTANT_EIGHT;

            for (int64_t i = 1; i < formerColLoops; i++) {
                Add(tmp, tmp, src[i * COMMON_B32_REPEAT_SIZE], COMMON_B32_REPEAT_SIZE, curRowsNum, computeParams);
                PipeBarrier<PIPE_V>();
            }
            if (likely(remainderCol != 0)) {
                Add(tmp, tmp, src[formerColLoops * COMMON_B32_REPEAT_SIZE], remainderCol, curRowsNum, computeParams);
            }
        }
        PipeBarrier<PIPE_V>();

        WholeReduceSum(dst, tmp, COMMON_B32_REPEAT_SIZE, curRowsNum, 1, 1, COMMON_CONSTANT_EIGHT);
    }
}

} // namespace SyncBatchNormGatherStatsFused