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
 * \file embedding_dense_grad_v2_small_dim.h
 * \brief
 */

#ifndef EMBEDDING_DENSE_GRAD_V2_H_SMALL_DIM_H
#define EMBEDDING_DENSE_GRAD_V2_H_SMALL_DIM_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

constexpr int64_t SORT_QUE_NUM = 3;

namespace AscendC {
template<typename MT, typename CT>
class EmbeddingDenseGradV2SmallDimKernel {
public:
__aicore__ inline EmbeddingDenseGradV2SmallDimKernel() = delete;
__aicore__ inline EmbeddingDenseGradV2SmallDimKernel(GM_ADDR grad, GM_ADDR indices, GM_ADDR backProps, GM_ADDR workSpace,
                                                     const EmbeddingDenseGradV2TilingData &tiling, TPipe &pipe)
{
    // 1. 初始化tiling参数
    InitParams(tiling);
    // 2. 初始化que
    InitBuffers(pipe);
    // 3. 设置gm地址
    SetGmAddr(grad, indices, backProps, workSpace, tiling);
}
__aicore__ inline void Process()
{
    for (int i = 0; i < copyTime_ - 1; i++) {
        CopyIn(i, true);
        ComputeAndCopyOut(true);
    }
    CopyIn(copyTime_ - 1, false);
    ComputeAndCopyOut(false);
    PIPE_MTE3_S();
    SyncAll();
    if (isDifferentDtype_) {
        SplitOutCasted();
        for (int i = 0; i < formLoopNum_; ++i) {
            uint64_t offset = startOutId_ + i * gradAlignNum_;
            CopyCastedOut(offset, gradAlignNum_);
        }
        if (tailLoopLength_ != 0) {
            uint64_t offset = startOutId_ + formLoopNum_ * gradAlignNum_;
            CopyCastedOut(offset, tailLoopLength_);
        }
    }
}

private:
__aicore__ inline void InitParams(const EmbeddingDenseGradV2TilingData &tiling)
{
    coreIdx_ = GetBlockIdx();
    ResetScaleCount();
    numWeights_ = tiling.params.numWeights;
    embeddingDim_ = tiling.params.embeddingDim;
    scaleGradByFreq_ = tiling.params.scaleGradByFreq;
    paddingIdx_ = tiling.params.paddingIdx;
    coreNum_ = tiling.params.coreNum;
    maxRowInUb_ = tiling.smallDimTiling.maxRowInUb;
    partNum_ = tiling.smallDimTiling.partNum;
    scaleWorkspaceLength_ = tiling.params.scaleWorkspaceLength;
    outCastedWorkspaceLength_ = tiling.params.outCastedWorkspaceLength;
    if (coreIdx_ < GetBlockNum() - 1) {
        rowNum_ = tiling.smallDimTiling.formerCopyRow;
        copyTime_ = tiling.smallDimTiling.formerCopyTime;
        lastRowNum_ = tiling.smallDimTiling.formerLastRow;
    } else {
        rowNum_ = tiling.smallDimTiling.tailCopyRow;
        copyTime_ = tiling.smallDimTiling.tailCopyTime;
        lastRowNum_ = tiling.smallDimTiling.tailLastRow;
    }
    copyRowNum_ = maxRowInUb_ > rowNum_ ? rowNum_ : maxRowInUb_;
    isDifferentDtype_ = !std::is_same<MT, CT>::value;
}

__aicore__ inline void InitBuffers(TPipe &pipe)
{
    uint64_t alignNum = BLOCK_SIZE_GRAD / sizeof(CT);
    gradAlignNum_ = BLOCK_SIZE / sizeof(MT);
    embeddingDimAlign_ = (embeddingDim_ + gradAlignNum_ - 1) / gradAlignNum_ * gradAlignNum_;
    gradAlignNum_ = (embeddingDimAlign_ * copyRowNum_ + alignNum - 1) / alignNum * alignNum;
    idxAlignNum_ = (copyRowNum_ + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE; // 32个数align
    pipe.InitBuffer(indicesQue_, BUFFER_NUM, idxAlignNum_ * sizeof(int));
    pipe.InitBuffer(idxBuf_, idxAlignNum_ * sizeof(int));
    pipe.InitBuffer(tmp2Que_, BUFFER_NUM, idxAlignNum_ * sizeof(CT));
    pipe.InitBuffer(tmpQue_, BUFFER_NUM, idxAlignNum_ * SORT_QUE_NUM * sizeof(CT));
    pipe.InitBuffer(sortIndicesQue_, BUFFER_NUM, idxAlignNum_ * SORT_QUE_NUM * sizeof(int));
    pipe.InitBuffer(gradQue_, BUFFER_NUM, gradAlignNum_ * sizeof(CT));
    pipe.InitBuffer(addResQue_, BUFFER_NUM, embeddingDim_ * sizeof(CT));
}

__aicore__ inline void SetGmAddr(GM_ADDR grad, GM_ADDR indices, GM_ADDR backProps, GM_ADDR workSpace,
                                 const EmbeddingDenseGradV2TilingData &tiling)
{
    uint64_t indicesAddrOffset = tiling.smallDimTiling.formerCopyRow * coreIdx_;
    uint64_t gradAddrOffset = tiling.smallDimTiling.formerCopyRow * embeddingDim_ * coreIdx_;
    gradGm_.SetGlobalBuffer((__gm__ MT*)grad + gradAddrOffset);
    indicesGm_.SetGlobalBuffer((__gm__ int*)indices + indicesAddrOffset);
    outputGm_.SetGlobalBuffer((__gm__ MT*)backProps);
    idxNumGm_.SetGlobalBuffer((__gm__ float*)workSpace);
    if (isDifferentDtype_) {
        outCastedGm_.SetGlobalBuffer((__gm__ CT*)workSpace + scaleWorkspaceLength_);
        if (coreIdx_ == 0) {
            InitOutput(outCastedGm_, outCastedWorkspaceLength_, (CT)(0.0f));
        }
        SyncAll();
        PIPE_S_MTE3();
    }
}

__aicore__ inline void CopyIn(const uint32_t progress, bool formerFlag)
{
    LocalTensor<int> indicesLocal = indicesQue_.AllocTensor<int>();
    LocalTensor<CT> gradLocal = gradQue_.AllocTensor<CT>();
    auto gradLocalCasted = gradLocal.template ReinterpretCast<MT>();
    uint64_t idxAddrOffset = copyRowNum_ * progress;
    uint64_t gradAddrOffset = copyRowNum_ * embeddingDim_ * progress;
    uint64_t copyRow = formerFlag ? copyRowNum_ : lastRowNum_;
    DataCopyExtParams idxCopyParams{1, static_cast<uint32_t>(copyRow * sizeof(int)), 0, 0, 0};
    DataCopyExtParams gradCopyParams{static_cast<uint16_t>(copyRow), static_cast<uint32_t>(embeddingDim_ * sizeof(MT)), 0, 0, 0};
    DataCopyPadExtParams idxpadParams{true, 0, 0, 0};
    DataCopyPadExtParams gradPadParams{true, 0, static_cast<uint8_t>(embeddingDimAlign_ - embeddingDim_), (MT)(0.0f)};
    DataCopyPad(indicesLocal, indicesGm_[idxAddrOffset], idxCopyParams, idxpadParams);
    if(isDifferentDtype_) {
        uint64_t gradLocalCastedOffset = gradLocalCasted.GetSize() / 2;
        DataCopyPad(gradLocalCasted[gradLocalCastedOffset], gradGm_[gradAddrOffset], gradCopyParams, gradPadParams);
        PIPE_MTE2_V();
        Cast(gradLocal, gradLocalCasted[gradLocalCastedOffset], RoundMode::CAST_NONE, gradLocalCastedOffset);
    }else {
        DataCopyPad(gradLocalCasted, gradGm_[gradAddrOffset], gradCopyParams, gradPadParams);
    }
    indicesQue_.EnQue<int>(indicesLocal);
    gradQue_.EnQue<CT>(gradLocal);
}

__aicore__ inline void SortIndices(bool formerFlag)
{
    LocalTensor<int> indicesLocal = indicesQue_.DeQue<int>();
    LocalTensor<int32_t> idxLocal = idxBuf_.Get<int32_t>();
    LocalTensor<CT> tmpLocal = tmpQue_.AllocTensor<CT>();
    LocalTensor<CT> tmp2Local = tmp2Que_.AllocTensor<CT>();
    LocalTensor<CT> sortResLocal = sortIndicesQue_.AllocTensor<CT>();

    uint32_t idxNum = formerFlag ? copyRowNum_ : lastRowNum_;
    uint32_t idxAlign32 = (idxNum + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    uint32_t sortRepeatTimes = idxAlign32 / BLOCK_SIZE;
    // 1. cast indices to fp32
    Duplicate<CT>(tmp2Local, -1, idxAlign32);
    Cast(tmp2Local, indicesLocal, RoundMode::CAST_ROUND, idxNum);
    // 2. create posIdx
    CreateVecIndex<int32_t>(idxLocal, 0U, idxNum);
    LocalTensor<uint32_t> idxULocal = idxLocal.ReinterpretCast<uint32_t>();
    // 3. sort indices
    Sort<float, true>(sortResLocal, tmp2Local, idxULocal, tmpLocal, sortRepeatTimes);
    // 4. extract res
    Extract(tmp2Local, idxULocal, sortResLocal, sortRepeatTimes);
    // 5. cast sort res to int
    Cast(indicesLocal, tmp2Local, RoundMode::CAST_ROUND, idxNum);
    indicesQue_.EnQue<int>(indicesLocal);
    tmpQue_.FreeTensor<CT>(tmpLocal);
    tmp2Que_.FreeTensor<CT>(tmp2Local);
    sortIndicesQue_.FreeTensor<CT>(sortResLocal);
}

__aicore__ inline void ResetAddQue()
{
    LocalTensor<CT> addLocal = addResQue_.AllocTensor<CT>();
    Duplicate<CT>(addLocal, 0.0, embeddingDim_);
    addResQue_.EnQue<CT>(addLocal);
}

__aicore__ inline void FreeAddQue()
{
    LocalTensor<CT> addLocal = addResQue_.DeQue<CT>();
    addResQue_.FreeTensor<CT>(addLocal);
}

__aicore__ inline void ResetScaleCount()
{
    scaleCount_ = 0;
}

__aicore__ inline void SplitOutCasted()
{
    uint64_t tailLength = numWeights_ * embeddingDim_ / coreNum_;
    uint64_t formLength = tailLength + 1UL;
    uint64_t formBlockNum = numWeights_ * embeddingDim_ % coreNum_;
    uint64_t tailBlockNum = coreNum_ - formBlockNum;

    if (coreIdx_ < formBlockNum) {
        formLoopNum_ = formLength / gradAlignNum_;
        tailLoopLength_ = formLength % gradAlignNum_;
        startOutId_ = coreIdx_ * formLength;
    }else {
        formLoopNum_ = tailLength / gradAlignNum_;
        tailLoopLength_ = tailLength % gradAlignNum_;
        startOutId_ = formBlockNum * formLength + (coreIdx_ - formBlockNum) * tailLength;
    }
}

__aicore__ inline void AtomicAddInUb(LocalTensor<CT> &gradLocal, const int addrOffset)
{
    ++scaleCount_;
    int offset  = addrOffset * embeddingDimAlign_;
    LocalTensor<CT> addLocal = addResQue_.DeQue<CT>();
    Add(addLocal, addLocal, gradLocal[offset], embeddingDim_);
    addResQue_.EnQue<CT>(addLocal);
}

__aicore__ inline void ComputeAndCopyOut(bool formerFlag)
{
    // 1. sort
    SortIndices(formerFlag);
    // 2. process one row
    LocalTensor<int> indicesLocal = indicesQue_.DeQue<int>();
    LocalTensor<int> idxLocal = idxBuf_.Get<int>();
    LocalTensor<CT> gradLocal = gradQue_.DeQue<CT>();
    uint64_t processRowNum = formerFlag ? copyRowNum_ : lastRowNum_;
    ResetAddQue();
    ResetScaleCount();
    int rightPtr;
    for (int i = 0; i < processRowNum;) {
        int posIdx = idxLocal.GetValue(i);
        int currentId = indicesLocal.GetValue(i);
        if (currentId == paddingIdx_) {
            ++i;
            continue;
        }
        AtomicAddInUb(gradLocal, posIdx);
        for (rightPtr = i + 1; rightPtr < processRowNum; rightPtr++) {
            int nextId = indicesLocal.GetValue(rightPtr);
            if (currentId != nextId) {
                break;
            }
            int nextPos = idxLocal.GetValue(rightPtr);
            AtomicAddInUb(gradLocal, nextPos);
        }
        i = rightPtr;
        CopyOut(currentId);
    }
    FreeAddQue();
    indicesQue_.FreeTensor<int>(indicesLocal);
    gradQue_.FreeTensor<CT>(gradLocal);
}

__aicore__ inline void CopyOut(int indice)
{
    LocalTensor<CT> addLocal = addResQue_.DeQue<CT>();
    auto addLocalCasted = addLocal.template ReinterpretCast<MT>();
    uint32_t gmAddrOffset = indice * embeddingDim_;
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(embeddingDim_ * sizeof(CT)), 0, 0, 0};
    SetAtomicAdd<CT>();
    if (isDifferentDtype_) {
        DataCopyPad(outCastedGm_[gmAddrOffset], addLocal, copyParams);
    }else {
        DataCopyPad(outputGm_[gmAddrOffset], addLocalCasted, copyParams);
    }
    SetAtomicNone();
    if (scaleGradByFreq_) {
        LocalTensor<float> tmpLocal = tmpQue_.AllocTensor<float>();
        tmpLocal.SetValue(0, static_cast<float>((int)scaleCount_));
        PIPE_S_MTE3();
        DataCopyExtParams scaleCopyParams{1, sizeof(uint32_t), 0, 0, 0};
        SetAtomicAdd<float>();
        DataCopyPad(idxNumGm_[indice], tmpLocal, scaleCopyParams);
        SetAtomicNone();
        tmpQue_.FreeTensor<float>(tmpLocal);
    }
    addResQue_.FreeTensor<CT>(addLocal);
    ResetAddQue();
    ResetScaleCount();
}

__aicore__ inline void CopyCastedOut(uint64_t offset, uint64_t opLength)
{
    LocalTensor<CT> gradLocal = gradQue_.AllocTensor<CT>();
    auto gradLocalCasted = gradLocal.template ReinterpretCast<MT>();
    DataCopyExtParams copyInParams{1, static_cast<uint32_t>(opLength * sizeof(CT)), 0, 0, 0};
    DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(opLength * sizeof(MT)), 0, 0, 0};
    DataCopyPadExtParams<CT> padParams{true, 0, 0, 0};
    DataCopyPad(gradLocal, outCastedGm_[offset], copyInParams, padParams);
    PIPE_MTE2_V();
    Cast(gradLocalCasted, gradLocal, RoundMode::CAST_RINT, opLength);
    PIPE_V_MTE3();
    DataCopyPad(outputGm_[offset], gradLocalCasted, copyOutParams);
    PIPE_MTE3_S();
    gradQue_.FreeTensor<CT>(gradLocal);
}

__aicore__ inline void PIPE_MTE3_S()
{
    event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
}

__aicore__ inline void PIPE_S_MTE3()
{
    event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
}

__aicore__ inline void PIPE_MTE2_V()
{
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
}

__aicore__ inline void PIPE_V_MTE3()
{
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
}

private:
uint64_t coreIdx_;
uint64_t partNum_;
uint64_t rowNum_;
uint64_t copyTime_;
uint64_t lastRowNum_;
uint64_t maxRowInUb_;
uint64_t copyRowNum_;
int64_t paddingIdx_;

uint64_t embeddingDim_;
uint64_t numWeights_;
bool scaleGradByFreq_;
uint64_t idxAlignNum_;
uint32_t scaleCount_;
uint64_t embeddingDimAlign_;
uint64_t coreNum_;
bool isDifferentDtype_;
uint64_t scaleWorkspaceLength_;
uint64_t outCastedWorkspaceLength_;
uint64_t gradAlignNum_;
uint64_t formLoopNum_;
uint64_t tailLoopLength_;
uint64_t startOutId_;

int lastIndices_;
bool switchId_;

private:
GlobalTensor<MT> gradGm_;
GlobalTensor<MT> outputGm_;
GlobalTensor<CT> outCastedGm_;
GlobalTensor<int> indicesGm_;
GlobalTensor<float> idxNumGm_;

TQue<TPosition::VECIN, BUFFER_NUM> gradQue_;
TQue<TPosition::VECIN, BUFFER_NUM> indicesQue_;
TQue<TPosition::VECIN, BUFFER_NUM> sortIndicesQue_;
TQue<TPosition::VECIN, BUFFER_NUM> tmpQue_;
TQue<TPosition::VECIN, BUFFER_NUM> tmp2Que_;
TQue<TPosition::VECOUT, BUFFER_NUM> addResQue_;
TBuf<TPosition::VECCALC> idxBuf_;
};
}

#endif // EMBEDDING_DENSE_GRAD_V2_H_SMALL_DIM_H