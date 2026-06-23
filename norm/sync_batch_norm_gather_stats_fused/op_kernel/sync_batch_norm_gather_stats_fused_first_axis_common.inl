/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file sync_batch_norm_gather_stats_fused_first_axis_common.inl
 * \brief Inline implementation for SyncBatchNormGatherStatsFusedFirstAxisCommon helper functions
 */

namespace SyncBatchNormGatherStatsFused {

template <typename T>
__aicore__ inline void SyncBatchNormGatherStatsFusedFirstAxisCommon<T>::CopyMeanData(
    const int64_t ubIdx, const int64_t curRowNum)
{
    buffer0_ = queue0_.AllocTensor<float>();
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyExtParams copyIntriParam;

    copyIntriParam.srcStride = 0;
    copyIntriParam.dstStride = 0;

    if (likely(cAlign == cLength)) {
        copyIntriParam.blockCount = 1;
        copyIntriParam.blockLen = curRowNum * cLength * sizeof(T);
    } else {
        copyIntriParam.blockCount = curRowNum;
        copyIntriParam.blockLen = cLength * sizeof(T);
        padParams.isPad = true;
        padParams.rightPadding = cAlign - cLength;
    }

    int64_t offset = (blockNStart + ubIdx * ubFormer) * cLength;

    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(buffer0_.ReinterpretCast<T>(), meanGm[offset], copyIntriParam, padParams);
    } else {
        DataCopyPad(buffer0_.ReinterpretCast<T>()[wholeBufferElemNums], meanGm[offset], copyIntriParam, padParams);
    }

    queue0_.EnQue(buffer0_);
    buffer0_ = queue0_.DeQue<float>();

    if constexpr (!IsSameType<T, float>::value) {
        Cast(buffer0_, buffer0_.ReinterpretCast<T>()[wholeBufferElemNums], RoundMode::CAST_NONE, wholeBufferElemNums);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void SyncBatchNormGatherStatsFusedFirstAxisCommon<T>::CopyInvstdData(
    const int64_t ubIdx, const int64_t curRowNum)
{
    buffer1_ = queue1_.AllocTensor<float>();
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyExtParams copyIntriParam;

    if (likely(cAlign == cLength)) {
        copyIntriParam.blockCount = 1;
        copyIntriParam.blockLen = curRowNum * cLength * sizeof(T);
    } else {
        copyIntriParam.blockCount = curRowNum;
        copyIntriParam.blockLen = cLength * sizeof(T);
        padParams.isPad = true;
        padParams.rightPadding = cAlign - cLength;
    }
    copyIntriParam.srcStride = 0;
    copyIntriParam.dstStride = 0;

    int64_t offset = (blockNStart + ubIdx * ubFormer) * cLength;

    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(buffer1_.ReinterpretCast<T>(), invstdGm[offset], copyIntriParam, padParams);
    } else {
        DataCopyPad(buffer1_.ReinterpretCast<T>()[wholeBufferElemNums], invstdGm[offset], copyIntriParam, padParams);
    }

    queue1_.EnQue(buffer1_);
    buffer1_ = queue1_.DeQue<float>();
    if constexpr (!IsSameType<T, float>::value) {
        Cast(buffer1_, buffer1_.ReinterpretCast<T>()[wholeBufferElemNums], RoundMode::CAST_NONE, wholeBufferElemNums);
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void SyncBatchNormGatherStatsFusedFirstAxisCommon<T>::CopyCount(
    const int64_t ubIdx, const int64_t curRowNum)
{
    buffer2_ = queue2_.AllocTensor<float>();
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyExtParams copyIntriParam;
    copyIntriParam.blockCount = 1;
    copyIntriParam.blockLen = curRowNum * sizeof(float);
    copyIntriParam.srcStride = 0;
    copyIntriParam.dstStride = 0;

    DataCopyPad(buffer2_, countsGm[blockNStart + ubIdx * ubFormer], copyIntriParam, padParams);
    queue2_.EnQue(buffer2_);
    buffer2_ = queue2_.DeQue<float>();
}

template <typename T>
__aicore__ inline void SyncBatchNormGatherStatsFusedFirstAxisCommon<T>::ComputeGlobalMean(
    const int64_t ubIdx, const int64_t curRowsNum)
{
    CopyMeanData(ubIdx, curRowsNum);
    CopyCount(ubIdx, curRowsNum);

    BlockBroadcast<float>(buffer5_, buffer2_, curRowsNum);
    PipeBarrier<PIPE_V>();

    BinElemWithInlinedLastBrcFP32(buffer0_, buffer0_, buffer5_, curRowsNum, Mul);
    PipeBarrier<PIPE_V>();

    buffer1_ = queue1_.AllocTensor<float>();
    Duplicate(buffer1_, countsSumScalar, curRowsNum * cAlign);
    PipeBarrier<PIPE_V>();

    Div<float, divConfig>(buffer0_, buffer0_, buffer1_, curRowsNum * cAlign);
    PipeBarrier<PIPE_V>();

    NlastReduceSum(buffer3_, buffer0_, curRowsNum);
    PipeBarrier<PIPE_V>();
    queue0_.FreeTensor(buffer0_);
    queue1_.FreeTensor(buffer1_);
    queue2_.FreeTensor(buffer2_);
    if (ubIdx == ubLoop - 1) {
        DataCopyExtParams copyIntriParam = {1, static_cast<uint32_t>(cAlign * sizeof(float)), 0, 0, 0};
        queue3_.EnQue(buffer3_);
        buffer3_ = queue3_.DeQue<float>();
        event_t event1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event1);
        WaitFlag<HardEvent::V_MTE3>(event1);
        SetAtomicAdd<float>();
        DataCopyPad(reduceWorkspaceGM, buffer3_, copyIntriParam);
        SetAtomicNone();
        queue3_.FreeTensor(buffer3_);
    }
}

template <typename T>
__aicore__ inline void SyncBatchNormGatherStatsFusedFirstAxisCommon<T>::ComputeGlobalVar(
    const int64_t ubIdx, const int64_t curRowsNum)
{
    CopyCount(ubIdx, curRowsNum);
    CopyInvstdData(ubIdx, curRowsNum);

    buffer0_ = queue0_.AllocTensor<float>();
    Duplicate(buffer0_, static_cast<float>(1.0f), curRowsNum * cAlign);
    PipeBarrier<PIPE_V>();

    Div<float, divConfig>(buffer1_, buffer0_, buffer1_, curRowsNum * cAlign);
    PipeBarrier<PIPE_V>();

    queue0_.FreeTensor(buffer0_);
    CopyMeanData(ubIdx, curRowsNum);
    PipeBarrier<PIPE_V>();

    Mul(buffer1_, buffer1_, buffer1_, curRowsNum * cAlign);
    PipeBarrier<PIPE_V>();

    Adds(buffer1_, buffer1_, -eps, curRowsNum * cAlign);
    PipeBarrier<PIPE_V>();

    BinElemWithInlinedNLastBrcFP32(buffer0_, buffer0_, buffer3_, curRowsNum, Sub);
    PipeBarrier<PIPE_V>();

    Mul(buffer0_, buffer0_, buffer0_, curRowsNum * cAlign);
    PipeBarrier<PIPE_V>();

    Add(buffer0_, buffer0_, buffer1_, curRowsNum * cAlign);
    PipeBarrier<PIPE_V>();

    BlockBroadcast<float>(buffer5_, buffer2_, curRowsNum);
    PipeBarrier<PIPE_V>();

    BinElemWithInlinedLastBrcFP32(buffer0_, buffer0_, buffer5_, curRowsNum, Mul);
    PipeBarrier<PIPE_V>();

    NlastReduceSum(buffer4_, buffer0_, curRowsNum);
    PipeBarrier<PIPE_V>();

    queue0_.FreeTensor(buffer0_);
    queue1_.FreeTensor(buffer1_);
    queue2_.FreeTensor(buffer2_);
    if (ubIdx == ubLoop - 1) {
        DataCopyExtParams copyIntriParam = {1, static_cast<uint32_t>(cAlign * sizeof(float)), 0, 0, 0};
        queue4_.EnQue(buffer4_);
        buffer4_ = queue4_.DeQue<float>();

        event_t event1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event1);
        WaitFlag<HardEvent::V_MTE3>(event1);
        SetAtomicAdd<float>();
        DataCopyPad(reduceWorkspaceGM, buffer4_, copyIntriParam);
        SetAtomicNone();

        queue3_.FreeTensor(buffer3_);
        queue4_.FreeTensor(buffer4_);
    }
}

template <typename T>
template <typename dType>
__aicore__ inline void SyncBatchNormGatherStatsFusedFirstAxisCommon<T>::BlockBroadcast(
    const LocalTensor<dType>& dst, const LocalTensor<dType>& src, const int64_t curRowsNum)
{
    constexpr int64_t MAX_REPEAT_BRCB = 255;
    constexpr int64_t BLOCK_SIZE = FA_COMMON_CONSTANT_EIGHT;

    int64_t totalBlocks = (curRowsNum + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int64_t processedBlocks = 0;

    while (processedBlocks < totalBlocks) {
        int64_t remainingBlocks = totalBlocks - processedBlocks;
        int64_t curRepeat = (remainingBlocks < MAX_REPEAT_BRCB) ? remainingBlocks : MAX_REPEAT_BRCB;
        int64_t srcOffset = processedBlocks * BLOCK_SIZE;

        Brcb(dst[srcOffset * BLOCK_SIZE], src[srcOffset], curRepeat, {1, BLOCK_SIZE});
        processedBlocks += curRepeat;
    }
}

template <typename T>
__aicore__ inline void SyncBatchNormGatherStatsFusedFirstAxisCommon<T>::BinElemNLastBrcLargeStride(
    const LocalTensor<float>& dst, const LocalTensor<float>& src0, const LocalTensor<float>& src1,
    const int64_t curRowsNum,
    void (*func)(
        const LocalTensor<float>&, const LocalTensor<float>&, const LocalTensor<float>&, uint64_t, uint8_t,
        const BinaryRepeatParams&))
{
    BinaryRepeatParams params;
    params.dstRepStride = 0;
    params.src0RepStride = 0;
    params.src1RepStride = 0;
    params.dstBlkStride = 1;
    params.src0BlkStride = 1;
    params.src1BlkStride = 1;

    int64_t formerColLoops = cAlign / FA_COMMON_B32_REPEAT_SIZE;
    int64_t remainderCol = cAlign % FA_COMMON_B32_REPEAT_SIZE;

    for (int64_t i = 0; i < curRowsNum; i++) {
        int64_t rowOffset = i * cAlign;
        for (int64_t j = 0; j < formerColLoops; j++) {
            int64_t colOffset = j * FA_COMMON_B32_REPEAT_SIZE;
            func(
                dst[rowOffset + colOffset], src0[rowOffset + colOffset], src1[colOffset], FA_COMMON_B32_REPEAT_SIZE, 1,
                params);
        }
        if (remainderCol != 0) {
            int64_t colOffset = formerColLoops * FA_COMMON_B32_REPEAT_SIZE;
            func(dst[rowOffset + colOffset], src0[rowOffset + colOffset], src1[colOffset], remainderCol, 1, params);
        }
    }
}

template <typename T>
__aicore__ inline void SyncBatchNormGatherStatsFusedFirstAxisCommon<T>::BinElemWithInlinedNLastBrcFP32(
    const LocalTensor<float>& dst, const LocalTensor<float>& src0, const LocalTensor<float>& src1,
    const int64_t curRowsNum,
    void (*func)(
        const LocalTensor<float>&, const LocalTensor<float>&, const LocalTensor<float>&, uint64_t, uint8_t,
        const BinaryRepeatParams&))
{
    if (unlikely(cAlign / FA_COMMON_CONSTANT_EIGHT > 255)) {
        BinElemNLastBrcLargeStride(dst, src0, src1, curRowsNum, func);
    } else {
        int64_t formerColLoops = cAlign / FA_COMMON_B32_REPEAT_SIZE;
        int64_t remainderCol = cAlign - formerColLoops * FA_COMMON_B32_REPEAT_SIZE;
        int64_t repeatLoops = curRowsNum / FA_COMMON_MAX_REPEAT_BRCB;
        int64_t remainderRepeat = curRowsNum - repeatLoops * FA_COMMON_MAX_REPEAT_BRCB;

        BinaryRepeatParams copyIntriParam;
        copyIntriParam.dstBlkStride = 1;
        copyIntriParam.src0BlkStride = 1;
        copyIntriParam.src1BlkStride = 1;
        copyIntriParam.dstRepStride = cAlign / FA_COMMON_CONSTANT_EIGHT;
        copyIntriParam.src0RepStride = cAlign / FA_COMMON_CONSTANT_EIGHT;
        copyIntriParam.src1RepStride = 0;

        for (int64_t i = 0; i < repeatLoops; i++) {
            int64_t repeatOffset = i * FA_COMMON_MAX_REPEAT_BRCB * cAlign;
            for (int64_t j = 0; j < formerColLoops; j++) {
                int64_t colOffset = j * FA_COMMON_B32_REPEAT_SIZE;
                func(
                    dst[repeatOffset + colOffset], src0[repeatOffset + colOffset], src1[colOffset],
                    FA_COMMON_B32_REPEAT_SIZE, FA_COMMON_MAX_REPEAT_BRCB, copyIntriParam);
            }
            if (likely(remainderCol != 0)) {
                int64_t colOffset = formerColLoops * FA_COMMON_B32_REPEAT_SIZE;
                func(
                    dst[repeatOffset + colOffset], src0[repeatOffset + colOffset], src1[colOffset], remainderCol,
                    FA_COMMON_MAX_REPEAT_BRCB, copyIntriParam);
            }
        }

        if (likely(remainderRepeat != 0)) {
            int64_t repeatOffset = repeatLoops * FA_COMMON_MAX_REPEAT_BRCB * cAlign;
            for (int64_t j = 0; j < formerColLoops; j++) {
                int64_t colOffset = j * FA_COMMON_B32_REPEAT_SIZE;
                func(
                    dst[repeatOffset + colOffset], src0[repeatOffset + colOffset], src1[colOffset],
                    FA_COMMON_B32_REPEAT_SIZE, remainderRepeat, copyIntriParam);
            }
            if (likely(remainderCol != 0)) {
                int64_t colOffset = formerColLoops * FA_COMMON_B32_REPEAT_SIZE;
                func(
                    dst[repeatOffset + colOffset], src0[repeatOffset + colOffset], src1[colOffset], remainderCol,
                    remainderRepeat, copyIntriParam);
            }
        }
    }
}

template <typename T>
__aicore__ inline void SyncBatchNormGatherStatsFusedFirstAxisCommon<T>::LastBrcFP32LargeStride(
    const LocalTensor<float>& output, const LocalTensor<float>& input0, const LocalTensor<float>& input1,
    const int64_t curRowsNum,
    void (*func)(
        const LocalTensor<float>&, const LocalTensor<float>&, const LocalTensor<float>&, uint64_t, uint8_t,
        const BinaryRepeatParams&))
{
    BinaryRepeatParams paramsRow;
    paramsRow.dstBlkStride = 1;
    paramsRow.src0BlkStride = 1;
    paramsRow.src1BlkStride = 0;
    paramsRow.dstRepStride = 0;
    paramsRow.src0RepStride = 0;
    paramsRow.src1RepStride = 0;

    int64_t colMain = cAlign / FA_COMMON_B32_REPEAT_SIZE;
    int64_t colRemain = cAlign % FA_COMMON_B32_REPEAT_SIZE;

    for (int64_t i = 0; i < curRowsNum; i++) {
        int64_t src1Offset = i * 8;
        int64_t rowOffset = i * cAlign;

        for (int64_t j = 0; j < colMain; j++) {
            int64_t colOffset = j * FA_COMMON_B32_REPEAT_SIZE;
            func(
                output[rowOffset + colOffset], input0[rowOffset + colOffset], input1[src1Offset],
                FA_COMMON_B32_REPEAT_SIZE, 1, paramsRow);
        }
        if (colRemain != 0) {
            int64_t colOffset = colMain * FA_COMMON_B32_REPEAT_SIZE;
            func(
                output[rowOffset + colOffset], input0[rowOffset + colOffset], input1[src1Offset], colRemain, 1,
                paramsRow);
        }
    }
}

template <typename T>
__aicore__ inline void SyncBatchNormGatherStatsFusedFirstAxisCommon<T>::BinElemWithInlinedLastBrcFP32(
    const LocalTensor<float>& output, const LocalTensor<float>& input0, const LocalTensor<float>& input1,
    const int64_t rows,
    void (*func)(
        const LocalTensor<float>&, const LocalTensor<float>&, const LocalTensor<float>&, uint64_t, uint8_t,
        const BinaryRepeatParams&))
{
    int64_t rowMain = rows / FA_COMMON_MAX_REPEAT_BRCB;
    int64_t rowRemain = rows - rowMain * FA_COMMON_MAX_REPEAT_BRCB;
    int64_t colMain = cAlign / FA_COMMON_B32_REPEAT_SIZE;
    int64_t colRemain = cAlign - colMain * FA_COMMON_B32_REPEAT_SIZE;
    if (cAlign / FA_COMMON_CONSTANT_EIGHT > 255) {
        LastBrcFP32LargeStride(output, input0, input1, rows, func);
    } else {
        BinaryRepeatParams params;
        params.src0BlkStride = 1;
        params.src1BlkStride = 0;
        params.dstBlkStride = 1;
        params.src0RepStride = cAlign / FA_COMMON_CONSTANT_EIGHT;
        params.src1RepStride = 1;
        params.dstRepStride = cAlign / FA_COMMON_CONSTANT_EIGHT;
        for (int64_t i = 0; i < rowMain; i++) {
            int64_t repeatOffset = i * FA_COMMON_MAX_REPEAT_BRCB * cAlign;
            for (int64_t j = 0; j < colMain; j++) {
                int64_t colOffset = j * FA_COMMON_B32_REPEAT_SIZE;
                func(output[repeatOffset + colOffset], input0[repeatOffset + colOffset], input1[i * FA_COMMON_MAX_REPEAT_BRCB * FA_COMMON_B32_BLOCK_SIZE], FA_COMMON_B32_REPEAT_SIZE, FA_COMMON_MAX_REPEAT_BRCB, params);
            }
            if (likely(colRemain != 0)) {
                int64_t colOffset = colMain * FA_COMMON_B32_REPEAT_SIZE;
                func(output[repeatOffset + colOffset], input0[repeatOffset + colOffset], input1[i * FA_COMMON_MAX_REPEAT_BRCB * FA_COMMON_B32_BLOCK_SIZE], colRemain, FA_COMMON_MAX_REPEAT_BRCB, params);
            }
        }
        if (likely(rowRemain != 0)) {
            int64_t repeatOffset = rowMain * FA_COMMON_MAX_REPEAT_BRCB * cAlign;
            for (int64_t j = 0; j < colMain; j++) {
                int64_t colOffset = j * FA_COMMON_B32_REPEAT_SIZE;
                func(output[repeatOffset + colOffset], input0[repeatOffset + colOffset], input1[rowMain * FA_COMMON_MAX_REPEAT_BRCB * FA_COMMON_B32_BLOCK_SIZE], FA_COMMON_B32_REPEAT_SIZE,
                    rowRemain, params);
            }
            if (likely(colRemain != 0)) {
                int64_t colOffset = colMain * FA_COMMON_B32_REPEAT_SIZE;
                func(output[repeatOffset + colOffset], input0[repeatOffset + colOffset], input1[rowMain * FA_COMMON_MAX_REPEAT_BRCB * FA_COMMON_B32_BLOCK_SIZE], colRemain, rowRemain,
                    params);
            }
        }
    }
}

template <typename T>
__aicore__ inline void SyncBatchNormGatherStatsFusedFirstAxisCommon<T>::NlastReduceSumLargeStride(
    const LocalTensor<float>& dst, const LocalTensor<float>& src, const int64_t curRowsNum)
{
    BinaryRepeatParams params;
    params.dstBlkStride = 1;
    params.src0BlkStride = 1;
    params.src1BlkStride = 1;
    params.dstRepStride = 0;
    params.src0RepStride = 0;
    params.src1RepStride = 0;

    int64_t formerColLoops = cAlign / FA_COMMON_B32_REPEAT_SIZE;
    int64_t remainderCol = cAlign % FA_COMMON_B32_REPEAT_SIZE;

    for (int64_t i = 0; i < curRowsNum; i++) {
        int64_t srcRowOffset = i * cAlign;

        for (int64_t j = 0; j < formerColLoops; j++) {
            int64_t colOffset = j * FA_COMMON_B32_REPEAT_SIZE;
            Add(dst[colOffset], src[srcRowOffset + colOffset], dst[colOffset], FA_COMMON_B32_REPEAT_SIZE, 1, params);
        }

        if (remainderCol != 0) {
            int64_t colOffset = formerColLoops * FA_COMMON_B32_REPEAT_SIZE;
            Add(dst[colOffset], src[srcRowOffset + colOffset], dst[colOffset], remainderCol, 1, params);
        }
        PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void SyncBatchNormGatherStatsFusedFirstAxisCommon<T>::NlastReduceSum(
    const LocalTensor<float>& dst, const LocalTensor<float>& src, const int64_t curRowsNum)
{
    if (cAlign / FA_COMMON_CONSTANT_EIGHT > 255) {
        NlastReduceSumLargeStride(dst, src, curRowsNum);
    } else {
        int64_t formerColLoops = cAlign / FA_COMMON_B32_REPEAT_SIZE;
        int64_t remainderCol = cAlign - formerColLoops * FA_COMMON_B32_REPEAT_SIZE;
        int64_t repeatLoops = curRowsNum / FA_COMMON_MAX_REPEAT_BRCB;
        int64_t remainderRepeat = curRowsNum - repeatLoops * FA_COMMON_MAX_REPEAT_BRCB;

        BinaryRepeatParams copyIntriParam;
        copyIntriParam.dstBlkStride = 1;
        copyIntriParam.src0BlkStride = 1;
        copyIntriParam.src1BlkStride = 1;
        copyIntriParam.dstRepStride = 0;
        copyIntriParam.src0RepStride = cAlign / FA_COMMON_CONSTANT_EIGHT;
        copyIntriParam.src1RepStride = 0;

        for (int64_t i = 0; i < repeatLoops; i++) {
            int64_t repeatOffset = i * FA_COMMON_MAX_REPEAT_BRCB * cAlign;
            for (int64_t j = 0; j < formerColLoops; j++) {
                int64_t colOffset = j * FA_COMMON_B32_REPEAT_SIZE;
                Add(dst[colOffset], src[repeatOffset + colOffset], dst[colOffset], FA_COMMON_B32_REPEAT_SIZE,
                    FA_COMMON_MAX_REPEAT_BRCB, copyIntriParam);
            }
            if (likely(remainderCol != 0)) {
                int64_t colOffset = formerColLoops * FA_COMMON_B32_REPEAT_SIZE;
                Add(dst[colOffset], src[repeatOffset + colOffset], dst[colOffset], remainderCol,
                    FA_COMMON_MAX_REPEAT_BRCB, copyIntriParam);
            }
            PipeBarrier<PIPE_V>();
        }

        if (likely(remainderRepeat != 0)) {
            int64_t repeatOffset = repeatLoops * FA_COMMON_MAX_REPEAT_BRCB * cAlign;
            for (int64_t j = 0; j < formerColLoops; j++) {
                int64_t colOffset = j * FA_COMMON_B32_REPEAT_SIZE;
                Add(dst[colOffset], src[repeatOffset + colOffset], dst[colOffset], FA_COMMON_B32_REPEAT_SIZE, remainderRepeat, copyIntriParam);
            }
            if (likely(remainderCol != 0)) {
                int64_t colOffset = formerColLoops * FA_COMMON_B32_REPEAT_SIZE;
                Add(dst[colOffset], src[repeatOffset + colOffset], dst[colOffset], remainderCol, remainderRepeat, copyIntriParam);
            }
        }
    }
}

template <typename T>
__aicore__ inline void SyncBatchNormGatherStatsFusedFirstAxisCommon<T>::LastReduceSumLargeStride(
    const LocalTensor<float>& tmp, const LocalTensor<float>& src, const int64_t curRowsNum, const int64_t curColNum)
{
    BinaryRepeatParams copyIntriParam;
    copyIntriParam.dstRepStride = 0;
    copyIntriParam.src0RepStride = 0;
    copyIntriParam.src1RepStride = 0;
    copyIntriParam.dstBlkStride = 1;
    copyIntriParam.src0BlkStride = 1;
    copyIntriParam.src1BlkStride = 1;

    int64_t formerColLoops = curColNum / FA_COMMON_B32_REPEAT_SIZE;
    int64_t remainderCol = curColNum % FA_COMMON_B32_REPEAT_SIZE;

    for (int64_t i = 0; i < curRowsNum; i++) {
        int64_t srcOffset = i * curColNum;
        int64_t tmpOffset = i * FA_COMMON_CONSTANT_EIGHT * 8;

        for (int64_t j = 1; j < formerColLoops; j++) {
            int64_t colOffset = j * FA_COMMON_B32_REPEAT_SIZE;
            Add(tmp[tmpOffset], tmp[tmpOffset], src[srcOffset + colOffset], FA_COMMON_B32_REPEAT_SIZE, 1,
                copyIntriParam);
        }

        if (likely(remainderCol != 0)) {
            int64_t colOffset = formerColLoops * FA_COMMON_B32_REPEAT_SIZE;
            Add(tmp[tmpOffset], tmp[tmpOffset], src[srcOffset + colOffset], remainderCol, 1, copyIntriParam);
        }
    }
}

template <typename T>
__aicore__ inline void SyncBatchNormGatherStatsFusedFirstAxisCommon<T>::LastReduceSum(
    const LocalTensor<float>& dst, const LocalTensor<float>& src, const LocalTensor<float>& tmp,
    const int64_t curRowsNum, const int64_t curColNum)
{
    if (curColNum <= FA_COMMON_B32_REPEAT_SIZE) {
        int64_t repLoops = curRowsNum / FA_COMMON_VC_MAX_REPEAT_BRCB;
        int64_t remainderRepeat = curRowsNum - repLoops * FA_COMMON_VC_MAX_REPEAT_BRCB;

        for (int64_t i = 0; i < repLoops; i++) {
            WholeReduceSum(
                dst[i * FA_COMMON_VC_MAX_REPEAT_BRCB], src[i * FA_COMMON_VC_MAX_REPEAT_BRCB * curColNum], curColNum,
                FA_COMMON_VC_MAX_REPEAT_BRCB, 1, 1, curColNum / FA_COMMON_CONSTANT_EIGHT);
        }

        if (likely(remainderRepeat != 0)) {
            WholeReduceSum(
                dst[repLoops * FA_COMMON_VC_MAX_REPEAT_BRCB], src[repLoops * FA_COMMON_VC_MAX_REPEAT_BRCB * curColNum],
                curColNum, remainderRepeat, 1, 1, curColNum / FA_COMMON_CONSTANT_EIGHT);
        }
    } else {
        DataCopyParams copycopyIntriParam;
        copycopyIntriParam.blockCount = curRowsNum;
        copycopyIntriParam.blockLen = FA_COMMON_CONSTANT_EIGHT;
        copycopyIntriParam.dstStride = 0;
        copycopyIntriParam.srcStride = curColNum / FA_COMMON_CONSTANT_EIGHT - FA_COMMON_CONSTANT_EIGHT;

        DataCopy(tmp, src, copycopyIntriParam);
        PipeBarrier<PIPE_V>();

        int64_t formerColLoops = curColNum / FA_COMMON_B32_REPEAT_SIZE;
        int64_t remainderCol = curColNum - formerColLoops * FA_COMMON_B32_REPEAT_SIZE;
        if (curColNum / FA_COMMON_CONSTANT_EIGHT > 255) {
            LastReduceSumLargeStride(tmp, src, curRowsNum, curColNum);
        } else {
            BinaryRepeatParams computeParams;
            computeParams.dstBlkStride = 1;
            computeParams.src0BlkStride = 1;
            computeParams.src1BlkStride = 1;
            computeParams.dstRepStride = FA_COMMON_CONSTANT_EIGHT;
            computeParams.src0RepStride = FA_COMMON_CONSTANT_EIGHT;
            computeParams.src1RepStride = curColNum / FA_COMMON_CONSTANT_EIGHT;

            for (int64_t i = 1; i < formerColLoops; i++) {
                Add(tmp, tmp, src[i * FA_COMMON_B32_REPEAT_SIZE], FA_COMMON_B32_REPEAT_SIZE, curRowsNum, computeParams);
                PipeBarrier<PIPE_V>();
            }
            if (likely(remainderCol != 0)) {
                Add(tmp, tmp, src[formerColLoops * FA_COMMON_B32_REPEAT_SIZE], remainderCol, curRowsNum, computeParams);
            }
        }
        PipeBarrier<PIPE_V>();

        WholeReduceSum(dst, tmp, FA_COMMON_B32_REPEAT_SIZE, curRowsNum, 1, 1, FA_COMMON_CONSTANT_EIGHT);
    }
}

} // namespace SyncBatchNormGatherStatsFused