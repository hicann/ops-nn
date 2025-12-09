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
 * \file advance_step_spec.h
 * \brief
 */

#ifndef ADVANCE_STEP_SPEC_H
#define ADVANCE_STEP_SPEC_H

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "advance_step_common.h"

namespace AdvanceStepNs {
using namespace AscendC;

template <typename T>
class KernelAdvanceStepSpec
{
public:
    __aicore__ inline KernelAdvanceStepSpec(){};
    __aicore__ inline void Init(
        GM_ADDR input_tokens, GM_ADDR sampled_token_ids, GM_ADDR input_positions, GM_ADDR seq_lens,
        GM_ADDR slot_mapping, GM_ADDR block_tables, GM_ADDR spec_token, GM_ADDR accepted_num, GM_ADDR workspace,
        const AdvanceStepTilingData* tilingData, TPipe* tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void CopyIn(int64_t loop, int64_t seqsNum);
    __aicore__ inline void Compute(int64_t loop, int64_t seqsNum);
    __aicore__ inline void CopyOut(int64_t loop, int64_t seqsNum);
    __aicore__ inline void ComputeSlotMapping(int64_t loop, int64_t seqsNum);
    __aicore__ inline void CopyOutSlotMapping(int64_t loop, int64_t seqsNum);
    __aicore__ inline void CopyInInputTokens(int64_t loop, int64_t seqsNum);
    __aicore__ inline void ComputeInputTokens(int64_t loop, int64_t seqsNum);
    __aicore__ inline void CopyOutInputTokens(int64_t loop, int64_t seqsNum);

private:
    TPipe* pipe_;
    GlobalTensor<T> inputTokensGm_;
    GlobalTensor<T> sampledTokenIdsGm_;
    GlobalTensor<T> inputPositionsGm_;
    GlobalTensor<T> seqLensGm_;
    GlobalTensor<T> slotMappingGm_;
    GlobalTensor<T> blockTablesGm_;
    GlobalTensor<T> specTokenGm_;
    GlobalTensor<T> acceptedNumGm_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> buffer1;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> buffer2;
    TQue<QuePosition::VECIN, 1> buffer3;
    TQue<QuePosition::VECIN, 1> tempQue_;

    TBuf<QuePosition::VECCALC> sampledTokenIdsBuf_;
    TBuf<QuePosition::VECCALC> outputInt32Buf_;
    int64_t blockIdx_ = 0;
    int64_t blockTablesStride_ = 0;
    int64_t numSeqs_ = 0;
    int64_t numQueries_ = 0;
    int64_t blockSize_ = 0;
    int64_t specNum_ = 0;
    int64_t curCoreSeqs_ = 0;
    int64_t curCoreLoopNum_ = 0;
    int64_t perLoopSeqs_ = 0;
    int64_t lastLoopSeqs_ = 0;
    int64_t tokenEachReqs_ = 0;
    bool align = false;
};

template <typename T>
__aicore__ inline void KernelAdvanceStepSpec<T>::Init(
    GM_ADDR input_tokens, GM_ADDR sampled_token_ids, GM_ADDR input_positions, GM_ADDR seq_lens, GM_ADDR slot_mapping,
    GM_ADDR block_tables, GM_ADDR spec_token, GM_ADDR accepted_num, GM_ADDR workspace,
    const AdvanceStepTilingData* tilingData, TPipe* tPipe)
{
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    // tiling中获取本核处理的seqs数量
    if (blockIdx_ == tilingData->needCoreNum - 1) {
        curCoreSeqs_ = tilingData->lastCoreSeqs;
    } else {
        curCoreSeqs_ = tilingData->perCoreSeqs;
    }
    specNum_ = tilingData->specNum;
    blockSize_ = tilingData->blockSize;
    numSeqs_ = tilingData->numSeqs;
    blockTablesStride_ = tilingData->blockTablesStride;

    // 计算每个loop需要处理的seqs数量、需要多少个loop、以及最后一个loop处理的seqs数量
    perLoopSeqs_ = Min(tilingData->perLoopMaxSeqs, curCoreSeqs_);
    curCoreLoopNum_ = curCoreSeqs_ / perLoopSeqs_;
    lastLoopSeqs_ = curCoreSeqs_ - curCoreLoopNum_ * perLoopSeqs_;

    int64_t seqOffset = blockIdx_ * tilingData->perCoreSeqs;
    tokenEachReqs_ = specNum_ + 1;
    int64_t elementByte32 = AlignBytes(tokenEachReqs_, sizeof(T)) / 2;
    if (elementByte32 % BLOCK_BYTES != 0) {
        align = true;
    }

    inputTokensGm_.SetGlobalBuffer((__gm__ T*)input_tokens + seqOffset * tokenEachReqs_);
    sampledTokenIdsGm_.SetGlobalBuffer((__gm__ T*)sampled_token_ids + seqOffset * tokenEachReqs_);
    inputPositionsGm_.SetGlobalBuffer((__gm__ T*)input_positions + seqOffset * tokenEachReqs_);
    seqLensGm_.SetGlobalBuffer((__gm__ T*)seq_lens + seqOffset * tokenEachReqs_);
    slotMappingGm_.SetGlobalBuffer((__gm__ T*)slot_mapping + seqOffset * tokenEachReqs_);
    blockTablesGm_.SetGlobalBuffer((__gm__ T*)block_tables + seqOffset * blockTablesStride_);
    specTokenGm_.SetGlobalBuffer((__gm__ T*)spec_token + seqOffset * specNum_);
    acceptedNumGm_.SetGlobalBuffer((__gm__ T*)accepted_num + seqOffset);

    pipe_->InitBuffer(buffer1, 1, (AlignBytes(tokenEachReqs_, sizeof(T)) + BLOCK_BYTES) * perLoopSeqs_);
    pipe_->InitBuffer(buffer2, 1, (AlignBytes(tokenEachReqs_, sizeof(T)) + BLOCK_BYTES) * perLoopSeqs_);
    pipe_->InitBuffer(buffer3, 1, (AlignBytes(tokenEachReqs_, sizeof(T)) + BLOCK_BYTES) * perLoopSeqs_);
    pipe_->InitBuffer(tempQue_, 1, (AlignBytes(tokenEachReqs_, sizeof(T)) + BLOCK_BYTES) * perLoopSeqs_);

    pipe_->InitBuffer(sampledTokenIdsBuf_, AlignBytes(tokenEachReqs_ + 1, sizeof(float)));
    pipe_->InitBuffer(outputInt32Buf_, (AlignBytes(tokenEachReqs_, sizeof(int32_t)) + BLOCK_BYTES) * perLoopSeqs_);
}

template <typename T>
__aicore__ inline void KernelAdvanceStepSpec<T>::CopyIn(int64_t loop, int64_t seqsNum)
{
    LocalTensor<T> inputPositionsLocal = buffer1.AllocTensor<T>();
    LocalTensor<T> seqLensLocal = buffer2.AllocTensor<T>();
    LocalTensor<T> acceptedNumLocal = buffer3.AllocTensor<T>();
    DataCopyExtParams dataCopyParams{
        static_cast<uint16_t>(seqsNum), static_cast<uint32_t>(tokenEachReqs_ * sizeof(T)), 0, 0, 0};
    if (align) {
        dataCopyParams.dstStride = 1;
    }
    DataCopyPadExtParams<T> dataCopyPadParams{false, 0, 0, 0};
    int64_t offset = loop * perLoopSeqs_ * tokenEachReqs_;
    DataCopyPad(inputPositionsLocal, inputPositionsGm_[offset], dataCopyParams, dataCopyPadParams);
    DataCopyPad(seqLensLocal, seqLensGm_[offset], dataCopyParams, dataCopyPadParams);
    DataCopyExtParams acceptedNumDataCopyParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(seqsNum * sizeof(T)), 0, 0, 0};
    offset = loop * perLoopSeqs_;
    DataCopyPad(acceptedNumLocal, acceptedNumGm_[offset], acceptedNumDataCopyParams, dataCopyPadParams);

    buffer1.EnQue(inputPositionsLocal);
    buffer2.EnQue(seqLensLocal);
    buffer3.EnQue(acceptedNumLocal);
}

template <typename T>
__aicore__ inline void KernelAdvanceStepSpec<T>::Compute(int64_t loop, int64_t seqsNum)
{
    LocalTensor<T> inputPositionsLocalInt64 = buffer1.DeQue<T>();
    LocalTensor<T> seqLensLocalInt64 = buffer2.DeQue<T>();
    LocalTensor<T> acceptedNumLocalInt64 = buffer3.DeQue<T>();
    LocalTensor<int32_t> bufferLocal = tempQue_.AllocTensor<int32_t>();
    LocalTensor<int32_t> seqLensLocal = outputInt32Buf_.Get<int32_t>();
    tempQue_.EnQue(bufferLocal);
    bufferLocal = tempQue_.DeQue<int32_t>();

    int bufOffset = 0;
    int elementNum = seqsNum * Align(tokenEachReqs_, sizeof(T));
    if (align) {
        elementNum = seqsNum * (Align(tokenEachReqs_, sizeof(T)) + INT64_ONE_BLOCK_NUM);
    }

    Cast(bufferLocal, inputPositionsLocalInt64, RoundMode::CAST_NONE, elementNum);
    Cast(seqLensLocal, seqLensLocalInt64, RoundMode::CAST_NONE, elementNum);
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

    // 对acceptedNum进行repeat_interval，每个元素tokenEachReqs_次
    int64_t index = 0;
    for (uint64_t i = 0; i < seqsNum; i++) {
        int32_t acceptedNumI = static_cast<int32_t>(acceptedNumLocalInt64.GetValue(i));
        Adds(
            bufferLocal[i * elementNum / seqsNum], bufferLocal[i * elementNum / seqsNum], acceptedNumI, tokenEachReqs_);
    }
    SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
    PipeBarrier<PIPE_V>();
    // inputPositions += 1
    Adds(bufferLocal, bufferLocal, 1, elementNum);
    PipeBarrier<PIPE_V>();
    // seqLens = inputPositions + 1
    Adds(seqLensLocal, bufferLocal, 1, elementNum);
    PipeBarrier<PIPE_V>();
    Cast(inputPositionsLocalInt64, bufferLocal, RoundMode::CAST_NONE, elementNum);
    Cast(seqLensLocalInt64, seqLensLocal, RoundMode::CAST_NONE, elementNum);
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);

    buffer1.EnQue(inputPositionsLocalInt64);
    buffer2.EnQue(seqLensLocalInt64);
    tempQue_.EnQue(bufferLocal);
    buffer3.EnQue(acceptedNumLocalInt64);
}

template <typename T>
__aicore__ inline void KernelAdvanceStepSpec<T>::CopyOut(int64_t loop, int64_t seqsNum)
{
    LocalTensor<T> inputPositionsLocal = buffer1.DeQue<T>();
    LocalTensor<T> seqLensLocal = buffer2.DeQue<T>();
    // 搬运inputPositions、seqLens、slotMapping，shape相同，用同一个dataCopyParams
    DataCopyExtParams dataCopyParams{
        static_cast<uint16_t>(seqsNum), static_cast<uint32_t>(tokenEachReqs_ * sizeof(T)), 0, 0, 0};
    if (align) {
        dataCopyParams.srcStride = 1;
    }
    int64_t offset = loop * perLoopSeqs_ * tokenEachReqs_;
    DataCopyPad(inputPositionsGm_[offset], inputPositionsLocal, dataCopyParams);
    DataCopyPad(seqLensGm_[offset], seqLensLocal, dataCopyParams);
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    buffer1.EnQue(inputPositionsLocal);
    buffer2.EnQue(seqLensLocal);
}

template <typename T>
__aicore__ inline void KernelAdvanceStepSpec<T>::ComputeSlotMapping(int64_t loop, int64_t seqsNum)
{
    LocalTensor<T> slotMappingLocalInt64 = buffer2.DeQue<T>();
    LocalTensor<int32_t> inputPositionsLocal = tempQue_.DeQue<int32_t>();
    LocalTensor<int32_t> bufferLocal = buffer3.DeQue<int32_t>();

    int bufOffset = 0;
    int elementNum = seqsNum * Align(tokenEachReqs_, sizeof(T));
    if (align) {
        elementNum = seqsNum * (Align(tokenEachReqs_, sizeof(T)) + INT64_ONE_BLOCK_NUM);
    }
    LocalTensor<int32_t> slotMappingLocal = outputInt32Buf_.Get<int32_t>();
    Cast(slotMappingLocal, slotMappingLocalInt64, RoundMode::CAST_NONE, elementNum);
    PipeBarrier<PIPE_V>();

    // 生成blockTables行偏移，存放在slotMappingLocal
    ArithProgression<int32_t>(bufferLocal, static_cast<int32_t>(0), blockTablesStride_, seqsNum);
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    PipeBarrier<PIPE_V>();

    for (uint64_t i = 0; i < seqsNum; i++) {
        int32_t bufferLocalI = static_cast<int32_t>(bufferLocal.GetValue(i));
        Duplicate(slotMappingLocal[i * elementNum / seqsNum], bufferLocalI, tokenEachReqs_);
    }
    PipeBarrier<PIPE_V>();
    // 计算inputPositions/blockSize，存放在bufferLocal，保证blockSize_不为0
    for (uint32_t i = 0; i < elementNum; i++) {
        bufferLocal.SetValue(i, inputPositionsLocal.GetValue(i) / static_cast<int32_t>(blockSize_));
    }

    SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
    // 计算需要取出的blockTable的下标偏移，即上两项相加，inputPositions/blockSize + blockTablesStride*index
    Add(bufferLocal, slotMappingLocal, bufferLocal, elementNum);
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    PipeBarrier<PIPE_V>();
    // gather获取偏移中的值
    int64_t blockTablesOffset = loop * perLoopSeqs_ * blockTablesStride_;
    for (int i = 0; i < seqsNum; ++i) {
        for (int j = 0; j < tokenEachReqs_; ++j) {
            int64_t index = i * elementNum / seqsNum + j;
            int offset = bufferLocal.GetValue(index);
            int v = blockTablesGm_[blockTablesOffset].GetValue(offset);
            slotMappingLocal.SetValue(index, v);
        }
    }
    SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
    // 计算blockTable[...] * blockSize
    Muls(slotMappingLocal, slotMappingLocal, static_cast<int32_t>(blockSize_), elementNum);
    PipeBarrier<PIPE_V>();
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    // 计算inputPositions % blockSize
    for (uint32_t i = 0; i < elementNum; i++) {
        bufferLocal.SetValue(i, inputPositionsLocal.GetValue(i) % static_cast<int32_t>(blockSize_));
    }
    SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
    // 计算slotMapping最终值
    Add(slotMappingLocal, bufferLocal, slotMappingLocal, elementNum);
    PipeBarrier<PIPE_V>();
    Cast(slotMappingLocalInt64, slotMappingLocal, RoundMode::CAST_NONE, elementNum);
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);

    buffer2.EnQue(slotMappingLocalInt64);
    buffer3.EnQue(bufferLocal);
    tempQue_.FreeTensor(inputPositionsLocal);
}

template <typename T>
__aicore__ inline void KernelAdvanceStepSpec<T>::CopyOutSlotMapping(int64_t loop, int64_t seqsNum)
{
    LocalTensor<T> slotMappingLocal = buffer2.DeQue<T>();
    int elementNum = seqsNum * tokenEachReqs_;
    DataCopyExtParams dataCopyParams{
        static_cast<uint16_t>(seqsNum), static_cast<uint32_t>(tokenEachReqs_ * sizeof(T)), 0, 0, 0};
    if (align) {
        dataCopyParams.srcStride = 1;
    }
    int64_t offset = loop * perLoopSeqs_ * tokenEachReqs_;
    DataCopyPad(slotMappingGm_[offset], slotMappingLocal, dataCopyParams);
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    buffer2.EnQue(slotMappingLocal);
}

template <typename T>
__aicore__ inline void KernelAdvanceStepSpec<T>::CopyInInputTokens(int64_t loop, int64_t seqsNum)
{
    LocalTensor<T> sampledTokenIdsLocal = buffer1.DeQue<T>();
    LocalTensor<T> specTokenLocal = buffer2.DeQue<T>();

    // 搬运sampledTokenIds，其第2维是specNum+1
    int64_t offset = loop * perLoopSeqs_ * tokenEachReqs_;
    DataCopyExtParams sampledTokenIdsDataCopyParams{
        static_cast<uint16_t>(seqsNum), static_cast<uint32_t>(tokenEachReqs_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> sampledTokenIdsDataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(
        sampledTokenIdsLocal, sampledTokenIdsGm_[offset], sampledTokenIdsDataCopyParams,
        sampledTokenIdsDataCopyPadParams);

    // 搬运specToken，其第2维是specNum
    offset = loop * perLoopSeqs_ * specNum_;
    DataCopyExtParams specTokenDataCopyParams{
        static_cast<uint16_t>(seqsNum), static_cast<uint32_t>(specNum_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> specTokenDataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(specTokenLocal, specTokenGm_[offset], specTokenDataCopyParams, specTokenDataCopyPadParams);

    buffer1.EnQue(sampledTokenIdsLocal);
    buffer2.EnQue(specTokenLocal);
}

template <typename T>
__aicore__ inline void KernelAdvanceStepSpec<T>::ComputeInputTokens(int64_t loop, int64_t seqsNum)
{
    LocalTensor<T> sampledTokenIdsLocalInt64 = buffer1.DeQue<T>();
    LocalTensor<T> lastTokensInt64 = buffer3.DeQue<T>();

    LocalTensor<float> sampledTokenIdsLocalFp32 = sampledTokenIdsBuf_.Get<float>();
    uint64_t stride = Align(tokenEachReqs_, sizeof(T));
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    for (uint64_t i = 0; i < seqsNum; ++i) {
        // sampledTokenIds末尾补-1
        Cast(sampledTokenIdsLocalFp32, sampledTokenIdsLocalInt64[i * stride], RoundMode::CAST_RINT, tokenEachReqs_);
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        sampledTokenIdsLocalFp32.SetValue(tokenEachReqs_, -1);
        LocalTensor<float> temp = sampledTokenIdsLocalFp32;
        SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
        // 取出sampledTokenIds一行的第一个-1的索引，将lastTokens存放在bufferLocal中
        ReduceMin(temp, temp, temp, specNum_ + 2, true);
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        float tempIndex = temp.GetValue(1);
        uint32_t minIndex = *reinterpret_cast<uint32_t*>(&tempIndex);
        // bufferLocal存放的是lastTokens
        int64_t lastToken = sampledTokenIdsLocalInt64.GetValue(i * stride + minIndex - 1);
        lastTokensInt64.SetValue(Align(1, sizeof(T)) * i, lastToken);
        PipeBarrier<PIPE_V>();
    }
    buffer3.EnQue(lastTokensInt64);
    buffer1.FreeTensor(sampledTokenIdsLocalInt64);
}

template <typename T>
__aicore__ inline void KernelAdvanceStepSpec<T>::CopyOutInputTokens(int64_t loop, int64_t seqsNum)
{
    LocalTensor<T> specTokenLocal = buffer2.DeQue<T>();
    LocalTensor<T> lastTokensLocal = buffer3.DeQue<T>();
    // 搬运lastToken到inputTokens的第1列
    int64_t offset = loop * perLoopSeqs_ * tokenEachReqs_;
    DataCopyExtParams lastTokenDataCopyParams{
        static_cast<uint16_t>(seqsNum), static_cast<uint32_t>(sizeof(T)), 0,
        static_cast<uint32_t>((specNum_) * sizeof(T)), 0};
    DataCopyPad(inputTokensGm_[offset], lastTokensLocal, lastTokenDataCopyParams);

    // // 搬运specToke到inputTokens的后specNum列
    DataCopyExtParams specTokenDataCopyParams{
        static_cast<uint16_t>(seqsNum), static_cast<uint32_t>(specNum_ * sizeof(T)), 0,
        static_cast<uint32_t>(sizeof(T)), 0};
    DataCopyPad(inputTokensGm_[offset + 1], specTokenLocal, specTokenDataCopyParams);

    buffer2.FreeTensor(specTokenLocal);
    buffer3.FreeTensor(lastTokensLocal);
}

template <typename T>
__aicore__ inline void KernelAdvanceStepSpec<T>::Process()
{
    // 每个循环处理perLoopSeqs_行
    int64_t i = 0;
    for (; i < curCoreLoopNum_; i++) {
        CopyIn(i, perLoopSeqs_);
        Compute(i, perLoopSeqs_);
        CopyOut(i, perLoopSeqs_);
        ComputeSlotMapping(i, perLoopSeqs_);
        CopyOutSlotMapping(i, perLoopSeqs_);
        CopyInInputTokens(i, perLoopSeqs_);
        ComputeInputTokens(i, perLoopSeqs_);
        CopyOutInputTokens(i, perLoopSeqs_);
    }
    if (lastLoopSeqs_ > 0) {
        CopyIn(i, lastLoopSeqs_);
        Compute(i, lastLoopSeqs_);
        CopyOut(i, lastLoopSeqs_);
        ComputeSlotMapping(i, lastLoopSeqs_);
        CopyOutSlotMapping(i, lastLoopSeqs_);
        CopyInInputTokens(i, lastLoopSeqs_);
        ComputeInputTokens(i, lastLoopSeqs_);
        CopyOutInputTokens(i, lastLoopSeqs_);
    }
}
} // namespace AdvanceStepNs
#endif // ADVANCE_STEP_SPEC_H