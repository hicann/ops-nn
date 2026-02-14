/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_nd_add_deterministic.h
 * \brief deterministic kernel of scatter_nd_add
 */

#ifndef SCATTER_ND_ADD_SIMT_SORT_H
#define SCATTER_ND_ADD_SIMT_SORT_H

#include "scatter_nd_add_common.h"
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"

namespace ScatterNdAdd {
using namespace AscendC;

template <typename INDICES_T, typename PARAMS_T, typename TYPE_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void SimtOut(
    __gm__ PARAMS_T* varGmAddr, 
    __ubuf__ INDICES_T* indiceLocalAddr, 
    __ubuf__ PARAMS_T* updateLocalAddr,
    int64_t colLen,
    uint32_t uniqueIdNum,
    int64_t updateOfset,
    int64_t afterAxisFactor,
    int64_t afterAxis,
    int64_t varInAxis,
    TYPE_T magic, 
    TYPE_T shift)
{
    for (TYPE_T index = Simt::GetThreadIdx(); index < afterAxisFactor * uniqueIdNum; index += Simt::GetThreadNum()) {
        
        TYPE_T quotient = Simt::UintDiv(index, magic, shift);
        TYPE_T updateAxisIdx = index - quotient * afterAxisFactor;
        if (updateAxisIdx < colLen)
        {
            INDICES_T varRowIndex = indiceLocalAddr[quotient];
        
            if (varRowIndex >= 0 && varRowIndex < varInAxis) {
                int64_t varRowOfest = varRowIndex * afterAxis;
                int64_t varColOfest = updateOfset + updateAxisIdx;

                int64_t varOfset = varRowOfest + varColOfest;
                if constexpr (IsSameType<PARAMS_T, bool>::value) {
                    PARAMS_T value = updateLocalAddr[index];
                    if (value > 0) {
                        varGmAddr[varOfset] = value;
                    }
                } else {
                    Simt::AtomicAdd(varGmAddr + varOfset, updateLocalAddr[index]);
                }
            }
        }
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename CAST_T, uint32_t castType>
class ScatterNdAddSimtSort : public ScatterNdAddBase<PARAMS_T, INDICES_T, CAST_T, castType>
{
public:
    __aicore__ inline ScatterNdAddSimtSort(const ScatterNdAddRegBaseTilingData& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void ProcessSplitIndices();
    __aicore__ inline void Process();
    __aicore__ inline void SimtOutUpdate(int64_t rowLen, int64_t colLen, int64_t updateOfset);
    __aicore__ inline void SimtOutUpdateNonSort(int64_t rowLen, int64_t colLen, int64_t updateOfset);

private:
    TPipe& pipe_;
    const ScatterNdAddRegBaseTilingData& tilingData_;

    int64_t curCoreIndexCount_{0};
    uint64_t strideList[MAX_RANK_COUNT];
};

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename CAST_T, uint32_t castType>
__aicore__ inline void ScatterNdAddSimtSort<PARAMS_T, INDICES_T, TYPE_T, CAST_T, castType>::Init(
    GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace)
{
    this->indicesFactor_ = tilingData_.indicesFactor;
    this->afterAxis_ = tilingData_.afterAxis;
    this->afterAxisFactor_ = tilingData_.afterAxisFactor;
    this->indexRankSize_ = tilingData_.indexRankSize;
    this->eachCoreAfterAxisCount_ = tilingData_.eachCoreAfterAxisCount;
    this->eachCoreIndexCount_ = tilingData_.eachCoreIndexCount;
    this->InitBaseBuffer(pipe_, tilingData_.indicesFactor, indices, updates, y);

    curCoreIndexCount_ =
        (GetBlockIdx() != (tilingData_.usedCoreNumBefore - 1) ? tilingData_.eachCoreIndexCount :
                                                                tilingData_.tailCoreIndexCount);
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename CAST_T, uint32_t castType>
__aicore__ inline void ScatterNdAddSimtSort<PARAMS_T, INDICES_T, TYPE_T, CAST_T, castType>::SimtOutUpdate(int64_t rowLen, int64_t colLen, int64_t updateOfset){
    LocalTensor<INDICES_T> updateSumIdxLocal = this->updateSumIdxQue_.template DeQue<INDICES_T>();
    LocalTensor<PARAMS_T> updatesLocal = this->dataQueue_.template DeQue<PARAMS_T>();
    
    TYPE_T magic = 0;
    TYPE_T shift = 0;
    int64_t afterAxisFactor = tilingData_.afterAxisFactor;
    int64_t afterAxis = tilingData_.afterAxis;
    int64_t varInAxis = tilingData_.varInAxis;
    uint32_t uniqueIdNum = this->uniqueIdNum_;

    GetUintDivMagicAndShift(magic, shift, static_cast<TYPE_T>(afterAxisFactor));
    
    if constexpr (IsSameType<PARAMS_T, half>::value || IsSameType<PARAMS_T, bfloat16_t>::value) {
        LocalTensor<float> updateSumLocal = this->updateSumQue_.template DeQue<float>();
        this->ExecConvertAlign(updatesLocal, updateSumLocal, rowLen, colLen);
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        
        AscendC::Simt::VF_CALL<SimtOut<INDICES_T, PARAMS_T, TYPE_T>>(
        Simt::Dim3(THREAD_NUM), 
        (__gm__ PARAMS_T*)(this->yGm_.GetPhyAddr()), 
        (__ubuf__ INDICES_T*)updateSumIdxLocal.GetPhyAddr(),
        (__ubuf__ PARAMS_T*)updatesLocal.GetPhyAddr(),
        colLen, 
        uniqueIdNum,
        updateOfset, 
        afterAxisFactor,
        afterAxis,
        varInAxis,
        magic,
        shift);
        this->updateSumQue_.template FreeTensor(updateSumLocal);
    } else {
        LocalTensor<PARAMS_T> updateSumLocal = this->updateSumQue_.template DeQue<PARAMS_T>();

        AscendC::Simt::VF_CALL<SimtOut<INDICES_T, PARAMS_T, TYPE_T>>(
        Simt::Dim3(THREAD_NUM), 
        (__gm__ PARAMS_T*)(this->yGm_.GetPhyAddr()), 
        (__ubuf__ INDICES_T*)updateSumIdxLocal.GetPhyAddr(),
        (__ubuf__ PARAMS_T*)updateSumLocal.GetPhyAddr(),
        colLen, 
        uniqueIdNum,
        updateOfset,
        afterAxisFactor,
        afterAxis,
        varInAxis,
        magic,
        shift);
        this->updateSumQue_.template FreeTensor(updateSumLocal);
    }
    
    this->dataQueue_.template FreeTensor(updatesLocal);
    this->updateSumIdxQue_.template EnQue(updateSumIdxLocal);
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename CAST_T, uint32_t castType>
__aicore__ inline void ScatterNdAddSimtSort<PARAMS_T, INDICES_T, TYPE_T, CAST_T, castType>::SimtOutUpdateNonSort(
    int64_t rowLen, int64_t colLen, int64_t updateOfset)
{
    LocalTensor<INDICES_T> outOfstLocal = this->outOfstBuf_.template Get<INDICES_T>();
    LocalTensor<PARAMS_T> updatesLocal = this->dataQueue_.template DeQue<PARAMS_T>();

    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    TYPE_T magic = 0;
    TYPE_T shift = 0;
    int64_t afterAxisFactor = tilingData_.afterAxisFactor;
    int64_t afterAxis = tilingData_.afterAxis;
    int64_t varInAxis = tilingData_.varInAxis;
    uint32_t uniqueIdNum = static_cast<uint32_t>(rowLen);

    GetUintDivMagicAndShift(magic, shift, static_cast<TYPE_T>(afterAxisFactor));

    AscendC::Simt::VF_CALL<SimtOut<INDICES_T, PARAMS_T, TYPE_T>>(
        Simt::Dim3(THREAD_NUM), (__gm__ PARAMS_T*)(this->yGm_.GetPhyAddr()),
        (__ubuf__ INDICES_T*)outOfstLocal.GetPhyAddr(), (__ubuf__ PARAMS_T*)updatesLocal.GetPhyAddr(), colLen,
        uniqueIdNum, updateOfset, afterAxisFactor, afterAxis, varInAxis, magic, shift);

    this->dataQueue_.template FreeTensor(updatesLocal);
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename CAST_T, uint32_t castType>
__aicore__ inline void ScatterNdAddSimtSort<PARAMS_T, INDICES_T, TYPE_T, CAST_T, castType>::ProcessSplitIndices()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNumBefore) {
        return;
    }
    int64_t rowLoopNum = ops::CeilDiv(curCoreIndexCount_, tilingData_.indicesFactor);
    int64_t rowMainDataLen = tilingData_.indicesFactor;
    int64_t rowTailDataLen = curCoreIndexCount_ - tilingData_.indicesFactor * (rowLoopNum - 1);
    int64_t colLoopNum = tilingData_.updateLoopSize;
    int64_t colMainDataLen = tilingData_.afterAxisFactor;
    int64_t colTailDataLen = tilingData_.updateTailNum;
    for (int64_t rowIdx = 0; rowIdx < rowLoopNum; rowIdx++) {
        int64_t rowDataLen = (rowIdx == rowLoopNum - 1) ? rowTailDataLen : rowMainDataLen;
        this->CopyIndiceInSplitIndices(rowIdx, rowDataLen);
        LocalTensor<INDICES_T> outOfstLocal = this->outOfstBuf_.template Get<INDICES_T>();
        if (this->maxScore_ > SORT_HIST_THRESHOLD) {
            this->SortIndices(outOfstLocal, rowDataLen);
            for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
                int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
                this->CopyUpdatesInSplitIndices(rowIdx, colIdx, rowDataLen, colDataLen);
                event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                this->ComputeUpdateSum(rowDataLen, colDataLen);
                int64_t updateOfset = colIdx * tilingData_.afterAxisFactor;
                SimtOutUpdate(rowDataLen, colDataLen, updateOfset);
            }
            LocalTensor<int32_t> uniqueIdCountLocal = this->uniqueIdCountQue_.template DeQue<int32_t>();
            LocalTensor<uint32_t> updatesOriginIdexLocal = this->updatesOriginIdexQue_.template DeQue<uint32_t>();
            LocalTensor<INDICES_T> updateSumIdxLocal = this->updateSumIdxQue_.template DeQue<INDICES_T>();
            this->uniqueIdCountQue_.template FreeTensor(uniqueIdCountLocal);
            this->updatesOriginIdexQue_.template FreeTensor(updatesOriginIdexLocal);
            this->updateSumIdxQue_.template FreeTensor(updateSumIdxLocal);
        } else {
            for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
                int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
                this->CopyUpdatesInSplitIndices(rowIdx, colIdx, rowDataLen, colDataLen);
                int64_t updateOfset = colIdx * tilingData_.afterAxisFactor;
                SimtOutUpdateNonSort(rowDataLen, colDataLen, updateOfset);
            }
        }
    }
}

template <typename PARAMS_T, typename INDICES_T, typename TYPE_T, typename CAST_T, uint32_t castType>
__aicore__ inline void ScatterNdAddSimtSort<PARAMS_T, INDICES_T, TYPE_T, CAST_T, castType>::Process()
{
    LocalTensor<INDICES_T> strideLocal = this->strideBuf_.template Get<INDICES_T>();
    for (int32_t i = 0; i < MAX_RANK_COUNT; i++) {
        strideLocal(i) = tilingData_.strideList[i];
    }
    ProcessSplitIndices();
}
} // namespace ScatterNdAdd
#endif // SCATTER_ND_ADD_SIMD_H