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
 * \file embedding_dense_grad_v2.h
 * \brief
 */

#ifndef EMBEDDING_DENSE_GRAD_V2_H
#define EMBEDDING_DENSE_GRAD_V2_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

constexpr uint64_t BUFFER_NUM = 1;
constexpr uint64_t DOUBLE_BUFFER = 1;
constexpr uint64_t INIT_PARAM = 0;
constexpr int64_t INDICE_INIT_PARAM = -1;
constexpr uint64_t LIMIT_COUNT_NUM = 10;
constexpr uint64_t COPY_ROW_NUM = 1;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t BLOCK_SIZE_GRAD = 64;
constexpr uint64_t DOUBLE_PRE_CORE = 2;

struct AddParam {
    uint64_t mask;
    uint64_t repeatTime;
    uint64_t computeFormerNum;
    uint64_t computeTailNum;
    int64_t lastIndices;
    bool switchId;
    uint64_t tailRepeatTime;
    uint64_t tailComputeFormerNum;
    uint64_t tailComputeTailNum;
    uint64_t formerRepeatTime;
    uint64_t formerComputeFormerNum;
    uint64_t formerComputeTailNum;
};

namespace AscendC {
template<typename MT, typename CT>
class EmbeddingDenseGradV2Kernel {
public:
    __aicore__ inline EmbeddingDenseGradV2Kernel() = delete;
    __aicore__ inline EmbeddingDenseGradV2Kernel(GM_ADDR grad, GM_ADDR sortIndices, GM_ADDR posIdx, GM_ADDR backProps,
                                                 GM_ADDR workSpace, const EmbeddingDenseGradV2TilingData &tiling, TPipe &pipe)
    {
        InitParams(tiling);
        InitBuffers(pipe);
        SetGmAddr(grad, sortIndices, posIdx, backProps, workSpace, tiling);
    }
    __aicore__ inline void Process()
    {
        InitQue();
        firstIndex_ = indiceGm_.GetValue(0);
        endIndex_ = indiceGm_.GetValue(rowNum_ - 1);
        for (uint64_t dimJ = 0; dimJ <= formerDimRepTime_; dimJ++) {
            UpdateParams(dimJ);
            if (curEmbeddingDim_ == 0) continue;
            for (uint64_t i = 0; i < rowNum_; i++) {
                CopyIn(i, dimJ);
                if (isDifferentDtype_) {
                    if (paddingIdx_ == firstIndex_) {
                        CopyToOutStage(paddingIdx_, false);
                    }
                    if (paddingIdx_ == endIndex_) {
                        CopyToOutStage(paddingIdx_, true);
                    }
                    ComputeUbCache(i, dimJ);
                }else {
                    ComputeAndCopyOut(i, dimJ);
                }
            }
            if (isDifferentDtype_) {
                PIPE_MTE3_S();
                SyncAll();
                ProcessGmData(dimJ);
                SyncAll();
            }
        }
        FreeQue();
    }

private:
    __aicore__ inline void UpdateParams(const uint64_t dimJ)
    {
        if (dimJ == formerDimRepTime_) {
            curEmbeddingDim_ = tailEmbeddingDim_;
            addParam_.repeatTime = addParam_.tailRepeatTime;
            addParam_.computeFormerNum = addParam_.tailComputeFormerNum;
            addParam_.computeTailNum = addParam_.tailComputeTailNum;
        } else {
            curEmbeddingDim_ = formerEmbeddingDim_;
            addParam_.repeatTime = addParam_.formerRepeatTime;
            addParam_.computeFormerNum = addParam_.formerComputeFormerNum;
            addParam_.computeTailNum = addParam_.formerComputeTailNum;
        }
    }

    __aicore__ inline void InitParams(const EmbeddingDenseGradV2TilingData &tiling)
    {
        blockIdx_ = GetBlockIdx();
        if (blockIdx_ >= tiling.params.formerRowRepTime) {
            rowNum_ = tiling.params.tailRowNum;
        } else {
            rowNum_ = tiling.params.formerRowNum;
        }

        addParam_.mask = tiling.params.computeMask;
        addParam_.formerRepeatTime = tiling.params.formerComputeRepTime;
        addParam_.formerComputeFormerNum = tiling.params.formerComputeFormerNum;
        addParam_.formerComputeTailNum = tiling.params.formerComputeTailNum;
        addParam_.lastIndices = INDICE_INIT_PARAM;
        addParam_.switchId = false;
        addParam_.tailRepeatTime = tiling.params.tailComputeRepTime;
        addParam_.tailComputeFormerNum = tiling.params.tailComputeFormerNum;
        addParam_.tailComputeTailNum = tiling.params.tailComputeTailNum;

        addCount_[0] = INIT_PARAM;
        addCount_[1] = INIT_PARAM;

        numWeights_ = tiling.params.numWeights;
        embeddingDim_ = tiling.params.embeddingDim;
        paddingIdx_ = tiling.params.paddingIdx;
        scaleGradByFreq_ = tiling.params.scaleGradByFreq;

        formerDimRepTime_ = tiling.params.formerDimRepTime;
        formerEmbeddingDim_ = tiling.params.formerEmbeddingDim;
        tailEmbeddingDim_ = tiling.params.tailEmbeddingDim;
        curEmbeddingDim_ = formerEmbeddingDim_;

        coreNum_ = tiling.params.coreNum;
        isDifferentDtype_ = !std::is_same<MT, CT>::value;
        scaleWorkspaceLength_ = tiling.params.scaleWorkspaceLength;
        outStageWorkspaceLength_ = tiling.params.outStageWorkspaceLength;
        outIndexWorkspaceLength_ = tiling.params.outIndexWorkspaceLength;
    }
    __aicore__ inline void InitBuffers(TPipe &pipe)
    {
        uint64_t idxAlignNum = BLOCK_SIZE / sizeof(int);
        uint64_t gradAlignNum = BLOCK_SIZE_GRAD / sizeof(CT);
        gradAlignNum = ((formerEmbeddingDim_ + gradAlignNum - 1) / gradAlignNum) * gradAlignNum;
        pipe.InitBuffer(indiceQue_, DOUBLE_BUFFER, idxAlignNum * COPY_ROW_NUM * sizeof(int));
        pipe.InitBuffer(posIdxQue_, DOUBLE_BUFFER, idxAlignNum * COPY_ROW_NUM * sizeof(int));
        pipe.InitBuffer(gradQue_, DOUBLE_BUFFER, gradAlignNum * sizeof(CT));
        pipe.InitBuffer(tmpQue_, BUFFER_NUM, idxAlignNum * COPY_ROW_NUM * sizeof(int));
        pipe.InitBuffer(addResQue_[0], BUFFER_NUM, gradAlignNum * sizeof(CT));
        pipe.InitBuffer(addResQue_[1], BUFFER_NUM, gradAlignNum * sizeof(CT));
        if (isDifferentDtype_) {
            pipe.InitBuffer(cacheQue_, BUFFER_NUM, gradAlignNum * sizeof(CT));
        }
    }

    __aicore__ inline void SetGmAddr(GM_ADDR grad, GM_ADDR sortIndices, GM_ADDR posIdx, GM_ADDR backProps, GM_ADDR workSpace, const EmbeddingDenseGradV2TilingData &tiling)
    {
        uint64_t formerRowNumLoops = blockIdx_ < tiling.params.formerRowRepTime ? blockIdx_ : tiling.params.formerRowRepTime;
        uint64_t tailRowNumLoops = blockIdx_ < tiling.params.formerRowRepTime ? 0 : blockIdx_ - tiling.params.formerRowRepTime;
        uint64_t indiceAddrOffset = tiling.params.formerRowNum * formerRowNumLoops + tiling.params.tailRowNum * tailRowNumLoops;
        gradGm_.SetGlobalBuffer((__gm__ MT*)grad);
        indiceGm_.SetGlobalBuffer((__gm__ uint32_t*)sortIndices + indiceAddrOffset);
        posIdxGm_.SetGlobalBuffer((__gm__ uint32_t*)posIdx + indiceAddrOffset);
        outputGm_.SetGlobalBuffer((__gm__ MT*)backProps);
        idxNumGm_.SetGlobalBuffer((__gm__ float*)workSpace, scaleWorkspaceLength_);
        if (isDifferentDtype_) {
            outIndexGm_.SetGlobalBuffer((__gm__ uint32_t*)workSpace + scaleWorkspaceLength_, outIndexWorkspaceLength_);
            outStageGm_.SetGlobalBuffer((__gm__ CT*)workSpace + scaleWorkspaceLength_ + outIndexWorkspaceLength_, outStageWorkspaceLength_);
        }
    }

    __aicore__ inline void InitQue()
    {
        LocalTensor<CT> addResLocal1 = addResQue_[0].AllocTensor<CT>();
        LocalTensor<CT> addResLocal2 = addResQue_[1].AllocTensor<CT>();
        ResetQue(addResLocal1);
        ResetQue(addResLocal2);
        addResQue_[0].EnQue<CT>(addResLocal1);
        addResQue_[1].EnQue<CT>(addResLocal2);
        
        if (isDifferentDtype_) {
            LocalTensor<CT> cacheLocal = cacheQue_.AllocTensor<CT>();
            ResetQue(cacheLocal);
            cacheQue_.EnQue<CT>(cacheLocal);
        }
    }

    __aicore__ inline void FreeQue()
    {
        LocalTensor<CT> addResLocal1 = addResQue_[0].DeQue<CT>();
        LocalTensor<CT> addResLocal2 = addResQue_[1].DeQue<CT>();
        addResQue_[0].FreeTensor<CT>(addResLocal1);
        addResQue_[1].FreeTensor<CT>(addResLocal2);

        if (isDifferentDtype_) {
            LocalTensor<CT> cacheLocal = cacheQue_.DeQue<CT>();
            cacheQue_.FreeTensor<CT>(cacheLocal);
        }
    }

    __aicore__ inline void ResetQue(LocalTensor<CT> &queData)
    {
        Duplicate<CT>(queData, 0.0, queData.GetSize());
        PIPE_V_S();
    }

    __aicore__ inline void CopyIn(const uint64_t progress, const uint64_t dimJ)
    {
        LocalTensor<CT> gradLocal = gradQue_.AllocTensor<CT>();
        auto gradLocalCasted = gradLocal.template ReinterpretCast<MT>();
        LocalTensor<uint32_t> indiceLocal = indiceQue_.AllocTensor<uint32_t>();
        LocalTensor<uint32_t> posIdxLocal = posIdxQue_.AllocTensor<uint32_t>();
        uint64_t gradAddrOffset = 0;
        uint64_t indicesOffset = progress;
        DataCopyParams gradCopyParams{1, static_cast<uint16_t>(curEmbeddingDim_ * sizeof(MT)), 0, 0};
        DataCopyParams indiceCopyParams{1, static_cast<uint16_t>(sizeof(uint32_t)), 0, 0};
        DataCopyPadParams padParams{true, 0, 0, 0};
        DataCopyPad(posIdxLocal, posIdxGm_[indicesOffset], indiceCopyParams, padParams);
        DataCopyPad(indiceLocal, indiceGm_[indicesOffset], indiceCopyParams, padParams);
        PIPE_MTE2_S();
        gradAddrOffset = posIdxLocal.GetValue(0) * embeddingDim_ + formerEmbeddingDim_ * dimJ;
        if(isDifferentDtype_) {
            uint64_t gradLocalCastedOffset = gradLocalCasted.GetSize() / 2;
            DataCopyPad(gradLocalCasted[gradLocalCastedOffset], gradGm_[gradAddrOffset], gradCopyParams, padParams);
            PIPE_MTE2_V();
            Cast(gradLocal, gradLocalCasted[gradLocalCastedOffset], RoundMode::CAST_NONE, curEmbeddingDim_);
        }else {
            DataCopyPad(gradLocalCasted, gradGm_[gradAddrOffset], gradCopyParams, padParams);
        }
        gradQue_.EnQue<CT>(gradLocal);
        indiceQue_.EnQue<uint32_t>(indiceLocal);
        posIdxQue_.FreeTensor<uint32_t>(posIdxLocal);
    }

    __aicore__ inline void ComputeAndCopyOut(const uint64_t progress, const uint64_t dimJ)
    {
        LocalTensor<uint32_t> indiceLocal = indiceQue_.DeQue<uint32_t>();
        LocalTensor<CT> gradLocal = gradQue_.DeQue<CT>();
        uint64_t currentId = indiceLocal.GetValue(0);
        bool isNeedSwitch = false;
        bool isLastRow = progress == rowNum_ - 1;
        if (currentId != paddingIdx_) {
            // 1. decided change atomic add que
            isNeedSwitch = CheckIsNeedSwitchAddQue(currentId);
            bool isLimite = addCount_[addParam_.switchId] == LIMIT_COUNT_NUM;
            // 2. atomic add in add que
            AtomicAddInUb(gradLocal);
            // 3. check is copyout
            if (isLimite || isLastRow) {
                CopyOut(addParam_.switchId, currentId, dimJ);
            }
            if (isNeedSwitch) {
                CopyOut(!addParam_.switchId, addParam_.lastIndices, dimJ);
            }
            addParam_.lastIndices = currentId;
        } else if (addParam_.lastIndices != INDICE_INIT_PARAM) {
            CopyOut(addParam_.switchId, addParam_.lastIndices, dimJ);
            addParam_.lastIndices = INDICE_INIT_PARAM;
        }
        gradQue_.FreeTensor<CT>(gradLocal);
        indiceQue_.FreeTensor<uint32_t>(indiceLocal);
    }

    __aicore__ inline void CopyOut(const uint64_t addAddrOffset, const uint64_t indice, const uint64_t dimJ)
    {
        // 1. copy out to releative grad row
        LocalTensor<CT> addResLocal = addResQue_[addAddrOffset].DeQue<CT>();
        auto addResLocalCasted = addResLocal.template ReinterpretCast<MT>();
        uint64_t gmAddrOffset = indice * embeddingDim_ + formerEmbeddingDim_ * dimJ;
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curEmbeddingDim_ * sizeof(MT)), 0, 0, 0};
        SetAtomicAdd<MT>();
        DataCopyPad(outputGm_[gmAddrOffset], addResLocalCasted, copyParams);
        SetAtomicNone();
        // 2. copy out indices counts
        AddFreq(addAddrOffset, indice, dimJ);
        PipeBarrier<PIPE_ALL>();
        ResetQue(addResLocal);
        ResetAddCount(addAddrOffset);
        addResQue_[addAddrOffset].EnQue<CT>(addResLocal);
    }

    __aicore__ inline void ResetAddCount(const uint64_t id)
    {
        addCount_[id] = INIT_PARAM;
    }

    __aicore__ inline bool CheckIsNeedSwitchAddQue(const uint64_t currentId)
    {
        // 1. indice is not equal to last indice
        if (addParam_.lastIndices != INDICE_INIT_PARAM && currentId != addParam_.lastIndices) {
            // reset
            addParam_.switchId = !addParam_.switchId;
            ResetAddCount(addParam_.switchId);
            return true;
        }
        return false;
    }

    __aicore__ inline void AtomicAddInUb(LocalTensor<CT> &gradLocal)
    {
        LocalTensor<CT> addLocal = addResQue_[static_cast<uint8_t>(addParam_.switchId)].DeQue<CT>();
        if (addParam_.computeFormerNum > 0) {
            Add(addLocal, addLocal, gradLocal, addParam_.mask, addParam_.repeatTime, {1, 1, 1, 8, 8, 8});
        }
        if (addParam_.computeTailNum > 0) {
            Add(addLocal[addParam_.computeFormerNum], addLocal[addParam_.computeFormerNum],
                gradLocal[addParam_.computeFormerNum], addParam_.computeTailNum, 1, {1, 1, 1, 0, 0, 0});
        }
        PIPE_V_S();
        addResQue_[static_cast<uint8_t>(addParam_.switchId)].EnQue<CT>(addLocal);
        addCount_[addParam_.switchId]++;
    }

    __aicore__ inline void GetFreeCore(uint64_t &coreId)
    {
        coreId += 1;
        if (coreId == coreNum_) {
            coreId = 0;
        }
    }

    __aicore__ inline void ComputeUbCache(const uint64_t progress, const uint64_t dimJ)
    {
        LocalTensor<uint32_t> indiceLocal = indiceQue_.DeQue<uint32_t>();
        LocalTensor<CT> gradLocal = gradQue_.DeQue<CT>();
        uint64_t currentId = indiceLocal.GetValue(0);
        bool isNeedSwitch = false;
        bool isLastRow = (progress == rowNum_ - 1);
        if (currentId != paddingIdx_) {
            // 1. decided change atomic ub
            isNeedSwitch = CheckIsNeedSwitchAddQue(currentId);
            bool isLimite = addCount_[addParam_.switchId] == LIMIT_COUNT_NUM;
            // 2. atomic add in ub
            AtomicAddInUb(gradLocal);
            // 3. check is copyout
            if (isLimite) {
                CopyToUbCache(addParam_.switchId, currentId, dimJ);
            }
            if (isNeedSwitch) {
                CopyToUbCache(!addParam_.switchId, addParam_.lastIndices, dimJ);
                CopyUbCacheOut(addParam_.lastIndices, dimJ);
            }
            if (isLastRow) {
                CopyToUbCache(addParam_.switchId, currentId, dimJ);
            }
            addParam_.lastIndices = currentId;
        } else if (addParam_.lastIndices != INDICE_INIT_PARAM) {
            CopyToUbCache(addParam_.switchId, addParam_.lastIndices, dimJ);
            CopyUbCacheOut(addParam_.lastIndices, dimJ);
            addParam_.lastIndices = INDICE_INIT_PARAM;
        }
        if (isLastRow) {
            CopyUbCacheOut(currentId, dimJ);
        }
        gradQue_.FreeTensor<CT>(gradLocal);
        indiceQue_.FreeTensor<uint32_t>(indiceLocal);
    }

    __aicore__ inline void CopyToUbCache(const uint64_t addAddrOffset, const uint64_t indice, const uint64_t dimJ)
    {
        LocalTensor<CT> addResLocal = addResQue_[addAddrOffset].DeQue<CT>();
        LocalTensor<CT> cacheLocal = cacheQue_.DeQue<CT>();
        if (addParam_.computeFormerNum > 0) {
            Add(cacheLocal, cacheLocal, addResLocal, addParam_.mask, addParam_.repeatTime, {1, 1, 1, 8, 8, 8});
        }
        if (addParam_.computeTailNum > 0) {
            Add(cacheLocal[addParam_.computeFormerNum], cacheLocal[addParam_.computeFormerNum],
                addResLocal[addParam_.computeFormerNum], addParam_.computeTailNum, 1, {1, 1, 1, 0, 0, 0});
        }
        AddFreq(addAddrOffset, indice, dimJ);
        PipeBarrier<PIPE_ALL>();
        cacheQue_.EnQue<CT>(cacheLocal);
        ResetQue(addResLocal);
        ResetAddCount(addAddrOffset);
        addResQue_[addAddrOffset].EnQue<CT>(addResLocal);
    }

    __aicore__ inline void CopyUbCacheOut(const uint64_t indice, const uint64_t dimJ)
    {
        if (indice == paddingIdx_) {
            LocalTensor<CT> cacheLocal = cacheQue_.DeQue<CT>();
            ResetQue(cacheLocal);
            cacheQue_.EnQue<CT>(cacheLocal);
            return;
        }
        if (firstIndex_ == endIndex_) {
            CopyToOutStage(indice, false);
            CopyToOutStage(indice, indice == endIndex_);
        }else if(indice == firstIndex_ || indice == endIndex_) {
            CopyToOutStage(indice, indice == endIndex_);
        }else {
            CopyToOutPut(indice, dimJ);
        }
    }

    __aicore__ inline void AddFreq(const uint64_t addAddrOffset, const uint64_t indice, const uint64_t dimJ)
    {
        if (scaleGradByFreq_ && dimJ == 0) {
            LocalTensor<float> tmpLocal = tmpQue_.AllocTensor<float>();
            tmpLocal.SetValue(0, static_cast<float>((int)addCount_[addAddrOffset]));
            PIPE_S_MTE3();
            DataCopyExtParams scaleCopyParams{1, sizeof(uint32_t), 0, 0, 0};
            SetAtomicAdd<float>();
            DataCopyPad(idxNumGm_[indice], tmpLocal, scaleCopyParams);
            SetAtomicNone();
            tmpQue_.FreeTensor<float>(tmpLocal);
        }
    }

    __aicore__ inline void CopyToOutStage(const uint64_t indice, const bool isEndIndex)
    {
        LocalTensor<uint32_t> tmpLocal = tmpQue_.AllocTensor<uint32_t>();
        LocalTensor<CT> cacheLocal = cacheQue_.DeQue<CT>();
        tmpLocal.SetValue(0, static_cast<uint32_t>((int)indice));
        PIPE_S_MTE3();
        DataCopyExtParams outStageCopyParams{1, static_cast<uint32_t>(curEmbeddingDim_ * sizeof(CT)), 0, 0, 0};
        DataCopyExtParams outIndexCopyParams{1, sizeof(uint32_t), 0, 0, 0};
        uint64_t outStageOffset = blockIdx_ * DOUBLE_PRE_CORE * formerEmbeddingDim_ + isEndIndex * formerEmbeddingDim_;
        uint64_t outIndexOffset = blockIdx_ * DOUBLE_PRE_CORE + isEndIndex;
        DataCopyPad(outStageGm_[outStageOffset], cacheLocal, outStageCopyParams);
        DataCopyPad(outIndexGm_[outIndexOffset], tmpLocal, outIndexCopyParams);
        PIPE_MTE3_S();
        tmpQue_.FreeTensor<uint32_t>(tmpLocal);
        ResetQue(cacheLocal);
        cacheQue_.EnQue<CT>(cacheLocal);
    }

    __aicore__ inline void CopyToOutPut(const uint64_t indice, const uint64_t dimJ) 
    {
        LocalTensor<CT> cacheLocal = cacheQue_.DeQue<CT>();
        auto cacheLocalCasted = cacheLocal.template ReinterpretCast<MT>();
        uint64_t gmAddrOffset = indice * embeddingDim_ + formerEmbeddingDim_ * dimJ;
        Cast(cacheLocalCasted, cacheLocal, RoundMode::CAST_RINT, curEmbeddingDim_);
        PIPE_V_MTE3();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(curEmbeddingDim_ * sizeof(MT)), 0, 0, 0};
        DataCopyPad(outputGm_[gmAddrOffset], cacheLocalCasted, copyParams);
        PIPE_MTE3_S();
        ResetQue(cacheLocal);
        cacheQue_.EnQue<CT>(cacheLocal);
    }

        __aicore__ inline void ProcessGmData(const uint64_t dimJ)
    {
        uint64_t pre = 0;
        uint64_t coreId = 0;
        uint32_t preIndex = outIndexGm_.GetValue(0);
        uint64_t loopNum = DOUBLE_PRE_CORE * coreNum_;
        for(uint64_t i = 1; i < loopNum; ++i) {
            uint32_t curIndex = outIndexGm_.GetValue(i);
            if(curIndex != preIndex) {
                GetFreeCore(coreId);
                if(blockIdx_ == coreId) {
                    ProcessSameIndex(pre, i, preIndex, dimJ);
                }
                pre = i;
                preIndex = curIndex;
            }
            if(i == loopNum - 1) {
                GetFreeCore(coreId);
                if(blockIdx_ == coreId) {
                    ProcessSameIndex(pre, i + 1, preIndex, dimJ);
                }
            }
        }
    }

    __aicore__ inline void ProcessSameIndex(const uint64_t startIndexId, const uint64_t endIndexId, const uint64_t handleIndex, const uint64_t dimJ)
    {
        for(uint64_t i = startIndexId; i < endIndexId; ++i) {
            CopyStageIn(i);
            ComputeStage(handleIndex, dimJ, i == endIndexId - 1);
        }
    }

     __aicore__ inline void CopyStageIn(const uint64_t indexId)
     {
        LocalTensor<CT> gradLocal = gradQue_.AllocTensor<CT>();
        DataCopyParams gradCopyParams{1, static_cast<uint16_t>(curEmbeddingDim_ * sizeof(CT)), 0, 0};
        DataCopyPadParams padParams{true, 0, 0, 0};
        DataCopyPad(gradLocal, outStageGm_[indexId * formerEmbeddingDim_], gradCopyParams, padParams);
        gradQue_.EnQue<CT>(gradLocal);
     }

    __aicore__ inline void ComputeStage(const uint64_t handleIndex, const uint64_t dimJ, const bool isEndIndexId)
    {
        LocalTensor<CT> gradLocal = gradQue_.DeQue<CT>();
        uint64_t currentId = handleIndex;
        bool isLimite = addCount_[addParam_.switchId] == LIMIT_COUNT_NUM;
        AtomicAddInUb(gradLocal);
        if (isLimite) {
            CopyToUbCache(addParam_.switchId, currentId, dimJ);
        }
        if (isEndIndexId) {
            CopyToUbCache(addParam_.switchId, currentId, dimJ);
            CopyToOutPut(currentId, dimJ);
        }
        gradQue_.FreeTensor<CT>(gradLocal);
    }

    __aicore__ inline void PIPE_S_MTE3()
    {
        event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    }

    __aicore__ inline void PIPE_V_S()
    {
        event_t eventIDVToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIDVToV);
        WaitFlag<HardEvent::V_S>(eventIDVToV);
    }

    __aicore__ inline void PIPE_MTE3_S()
    {
        event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
        WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
    }

    __aicore__ inline void PIPE_MTE2_V()
    {
        event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    }

    __aicore__ inline void PIPE_MTE2_S()
    {
        event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
    }

    __aicore__ inline void PIPE_V_MTE3()
    {
        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    }

private:
    GlobalTensor<MT> gradGm_;
    GlobalTensor<MT> outputGm_;
    GlobalTensor<uint32_t> indiceGm_;
    GlobalTensor<uint32_t> posIdxGm_;
    GlobalTensor<float> idxNumGm_;
    GlobalTensor<CT> outStageGm_;
    GlobalTensor<uint32_t> outIndexGm_;

    TQue<TPosition::VECIN, DOUBLE_BUFFER> gradQue_;
    TQue<TPosition::VECIN, DOUBLE_BUFFER> indiceQue_;
    TQue<TPosition::VECIN, DOUBLE_BUFFER> posIdxQue_;
    TQue<TPosition::VECOUT, DOUBLE_BUFFER> cacheQue_;
    TQue<TPosition::VECIN, BUFFER_NUM> tmpQue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> addResQue_[2];

    AddParam addParam_;
    uint64_t rowNum_;
    uint64_t addCount_[2];

    uint64_t blockIdx_;
    // attr
    uint64_t numWeights_;
    uint64_t embeddingDim_;
    uint64_t paddingIdx_;
    bool scaleGradByFreq_;

    // big shape
    uint64_t formerDimRepTime_;
    uint64_t formerEmbeddingDim_;
    uint64_t tailEmbeddingDim_;
    uint64_t curEmbeddingDim_;

    uint64_t firstIndex_;
    uint64_t endIndex_;
    uint64_t coreNum_;
    bool isDifferentDtype_;
    uint64_t scaleWorkspaceLength_;
    uint64_t outStageWorkspaceLength_;
    uint64_t outIndexWorkspaceLength_;
};
}

#endif // EMBEDDING_DENSE_GRAD_V2_H