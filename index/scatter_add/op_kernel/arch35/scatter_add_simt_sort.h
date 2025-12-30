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
 * \file scatter_add_simt_sort.h
 * \brief scatter_add
 */
#ifndef ASCENDC_SCATTER_ADD_SORT_H_
#define ASCENDC_SCATTER_ADD_SORT_H_

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"

namespace ScatterAdd
{
using namespace AscendC;
using namespace ScatterAddCommon;


#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM_SORT = 256;
#else
constexpr uint32_t THREAD_NUM_SORT = 1024;
#endif

template <typename IDX_T, typename VAR_T, typename CAST_T, typename ADDR_T, bool isUpdateScalar, uint32_t castType>
class ScatterAddSimtSort
{
public:
    __aicore__ inline ScatterAddSimtSort(const ScatterAddTilingData& tilingData, TPipe& pipe)
        : td_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace);
    __aicore__ inline void CopyInIndicesUpdates(uint32_t loopIdx, uint32_t dataCount);
    __aicore__ inline void Compute(uint32_t loopIdx, uint32_t indicesCount, VAR_T updateScalarValue);
    __aicore__ inline void Process();
    __aicore__ inline void ParseTilingData();

private:
    GlobalTensor<VAR_T> var_;
    GlobalTensor<IDX_T> indices_;
    GlobalTensor<VAR_T> updates_;

    TQue<QuePosition::VECIN, 1> indicesInQueue_;
    TQue<QuePosition::VECIN, 1> updatesInQueue_;
    TBuf<QuePosition::VECCALC> castIndicesQue_;
    TBuf<QuePosition::VECCALC> castTmpIndicesQue_;
    TBuf<QuePosition::VECCALC> sortIndicesQue_;
    TBuf<QuePosition::VECCALC> updatesOriginIdexQue_;
    TBuf<QuePosition::VECCALC> uniqueIdCountQue_;
    TPipe& pipe_;
    const ScatterAddTilingData& td_;

    ADDR_T blockIdx_{0};
    ADDR_T blockNum_{0};
    uint64_t ubTailLoopSize_ = 0;        // 当前coreUB尾循环搬运数据量，排序使用
    uint64_t currLoopCount_ = 0;         // 当前core循环搬运数据次数，排序使用
    static constexpr uint32_t shiftOffset_ = platform::GetUbBlockSize() / sizeof(CAST_T);
};

template <typename IDX_T, typename VAR_T, typename CAST_T, typename ADDR_T, bool isUpdateScalar, uint32_t castType>
__aicore__ inline void ScatterAddSimtSort<IDX_T, VAR_T, CAST_T, ADDR_T, isUpdateScalar, castType>::ParseTilingData()
{
    if (blockIdx_ == td_.sortCoreNum - 1) {
        currLoopCount_ = td_.tailBlockLoop;
        ubTailLoopSize_ = td_.tailBlockTail;
    } else {
        currLoopCount_ = td_.normBlockLoop;
        ubTailLoopSize_ = td_.indicesFactor;
    }
}

template <typename IDX_T, typename VAR_T, typename CAST_T, typename ADDR_T, bool isUpdateScalar, uint32_t castType>
__aicore__ inline void ScatterAddSimtSort<IDX_T, VAR_T, CAST_T, ADDR_T, isUpdateScalar, castType>::Init(
                                  GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace)
{
    blockIdx_ = GetBlockIdx();
    blockNum_ = GetBlockNum();
    ParseTilingData();
    var_.SetGlobalBuffer((__gm__ VAR_T*)(var));
    indices_.SetGlobalBuffer((__gm__ IDX_T*)(indices));
    updates_.SetGlobalBuffer((__gm__ VAR_T*)(updates));

    pipe_.InitBuffer(indicesInQueue_, 1, ops::CeilAlign(td_.indicesFactor * sizeof(IDX_T), UB_AGLIN_VALUE));
    pipe_.InitBuffer(updatesInQueue_, 1, ops::CeilAlign(td_.indicesFactor * td_.varShape[1] * sizeof(VAR_T), UB_AGLIN_VALUE));
    pipe_.InitBuffer(updatesOriginIdexQue_, ops::CeilAlign(td_.indicesFactor * sizeof(uint32_t), UB_AGLIN_VALUE));
    pipe_.InitBuffer(uniqueIdCountQue_, ops::CeilAlign(td_.indicesFactor * sizeof(int32_t), UB_AGLIN_VALUE) + SORT_PAD_NUM * UB_AGLIN_VALUE);
    if constexpr (castType == CAST_0) {
        pipe_.InitBuffer(sortIndicesQue_, ops::CeilAlign(td_.indicesFactor * sizeof(IDX_T), UB_AGLIN_VALUE) + SORT_PAD_NUM * UB_AGLIN_VALUE);
    } else if constexpr (castType == CAST_3) {
        pipe_.InitBuffer(sortIndicesQue_, ops::CeilAlign(td_.indicesFactor * sizeof(CAST_T), UB_AGLIN_VALUE) + SORT_PAD_NUM * UB_AGLIN_VALUE);
        pipe_.InitBuffer(castIndicesQue_, ops::CeilAlign(td_.indicesFactor * sizeof(CAST_T), UB_AGLIN_VALUE));
        pipe_.InitBuffer(castTmpIndicesQue_, ops::CeilAlign(td_.indicesFactor * sizeof(int32_t), UB_AGLIN_VALUE));
    } else {
        pipe_.InitBuffer(sortIndicesQue_, ops::CeilAlign(td_.indicesFactor * sizeof(CAST_T), UB_AGLIN_VALUE) + SORT_PAD_NUM * UB_AGLIN_VALUE);
        pipe_.InitBuffer(castIndicesQue_, ops::CeilAlign(td_.indicesFactor * sizeof(CAST_T), UB_AGLIN_VALUE));
    }
}

template <typename IDX_T, typename VAR_T, typename CAST_T, typename ADDR_T, bool isUpdateScalar>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_SORT) inline void ScatterAddSimtSortCompute(
    ADDR_T outputOuterDimSize, __gm__ VAR_T* outputAddr, __local_mem__ VAR_T* inputAddr,
    __local_mem__ CAST_T* sortedAddr, __local_mem__ uint32_t* sortedOriginIndexAddr, __local_mem__ int32_t* cumSumAddr,
    uint32_t uniqueIndexNum, ADDR_T lastDim, VAR_T updateScalarValue)
{
    int32_t blockIdx = Simt::GetThreadIdx<1>();
    int32_t blockNum = Simt::GetThreadNum<1>();
    int32_t innerOffset = Simt::GetThreadIdx<0>();
    for(int32_t i = blockIdx; i < uniqueIndexNum; i += blockNum) {
        if (sortedAddr[cumSumAddr[i]] < 0 || sortedAddr[cumSumAddr[i]] >= outputOuterDimSize) {
            continue;
        }
        VAR_T result = 0;
        for (int32_t tid = 0; tid < cumSumAddr[i + 1] - cumSumAddr[i]; tid++) {
            int32_t srcOffset = sortedOriginIndexAddr[cumSumAddr[i] + tid] * lastDim + innerOffset;
            result += inputAddr[srcOffset];
        }
        int64_t gmDstOffset = sortedAddr[cumSumAddr[i]] * lastDim + innerOffset;
        Simt::AtomicAdd(outputAddr + gmDstOffset, result);
    }
}

template <typename IDX_T, typename VAR_T, typename CAST_T, typename ADDR_T, bool isUpdateScalar, uint32_t castType>
__aicore__ inline void ScatterAddSimtSort<IDX_T, VAR_T, CAST_T, ADDR_T, isUpdateScalar, castType>::CopyInIndicesUpdates(
                                                                         uint32_t loopIdx, uint32_t indicesCount)
{
    LocalTensor<IDX_T> indicesLocal = indicesInQueue_.AllocTensor<IDX_T>();
    LocalTensor<VAR_T> updatesLocal = updatesInQueue_.AllocTensor<VAR_T>();

    DataCopyExtParams indicesCopyParams { 1, (uint32_t)(indicesCount * sizeof(IDX_T)), 0, 0, 0 };
    DataCopyPadExtParams<IDX_T> indicesPadParams { false, 0, 0, 0 };
    DataCopyPad(indicesLocal, indices_[blockIdx_ * td_.normBlockIndices + loopIdx * td_.indicesFactor],
                                                                    indicesCopyParams, indicesPadParams);
    DataCopyExtParams updatesCopyParams { 1, (uint32_t)(indicesCount * td_.varShape[1] * sizeof(VAR_T)), 0, 0, 0 };
    DataCopyPadExtParams<VAR_T> updatesPadParams { false, 0, 0, 0 };
    DataCopyPad(updatesLocal, updates_[(blockIdx_ * td_.normBlockIndices + loopIdx * td_.indicesFactor) * td_.varShape[1]],
                                                                    updatesCopyParams, updatesPadParams);
    indicesInQueue_.EnQue<IDX_T>(indicesLocal);
    updatesInQueue_.EnQue<VAR_T>(updatesLocal);
}

template <typename IDX_T, typename VAR_T, typename CAST_T, typename ADDR_T, bool isUpdateScalar, uint32_t castType>
__aicore__ inline void ScatterAddSimtSort<IDX_T, VAR_T, CAST_T, ADDR_T, isUpdateScalar, castType>::Compute(
                                    uint32_t loopIdx, uint32_t indicesCount, VAR_T updateScalarValue)
{
    ADDR_T totalCol = static_cast<ADDR_T>(td_.varShape[1]);
    ADDR_T varFirstDimSize = static_cast<ADDR_T>(td_.varShape[0]);
    ADDR_T magic = 0;
    ADDR_T shift = 0;
    GetUintDivMagicAndShift(magic, shift, totalCol);
    
    LocalTensor<IDX_T> indicesLocal = indicesInQueue_.DeQue<IDX_T>();
    LocalTensor<VAR_T> updatesLocal = updatesInQueue_.DeQue<VAR_T>();
    LocalTensor<uint32_t> updatesOriginIdxLocal = updatesOriginIdexQue_.Get<uint32_t>();
    LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.Get<int32_t>();

    uint32_t uniqueIdNum = 0;
    LocalTensor<CAST_T> indicesSortedLocal = sortIndicesQue_.Get<CAST_T>();
    if constexpr (castType == CAST_0) {
        uniqueIdNum = SortAndComputeUniqueIdx<IDX_T>(
            indicesCount, indicesLocal, indicesSortedLocal, uniqueIdCountLocal, updatesOriginIdxLocal);
    } else if constexpr (castType == CAST_3) {
        LocalTensor<CAST_T> indicesCastLocal = castIndicesQue_.Get<CAST_T>();
        LocalTensor<int32_t> indicesCastTmpLocal = castTmpIndicesQue_.Get<int32_t>();
        Cast<int32_t, IDX_T>(indicesCastTmpLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
        Cast<CAST_T, int32_t>(indicesCastLocal, indicesCastTmpLocal, RoundMode::CAST_NONE, indicesCount);
        uniqueIdNum = SortAndComputeUniqueIdx<CAST_T>(
            indicesCount, indicesCastLocal, indicesSortedLocal, uniqueIdCountLocal, updatesOriginIdxLocal);
    } else {
        LocalTensor<CAST_T> indicesCastLocal = castIndicesQue_.Get<CAST_T>();
        Cast<CAST_T, IDX_T>(indicesCastLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
        uniqueIdNum = SortAndComputeUniqueIdx<CAST_T>(
            indicesCount, indicesCastLocal, indicesSortedLocal, uniqueIdCountLocal, updatesOriginIdxLocal);
    }

    __local_mem__ CAST_T* indicesSortedPtr = (__local_mem__ CAST_T*)(indicesSortedLocal.GetPhyAddr()) + shiftOffset_;
    __local_mem__ VAR_T* updatesLocalPtr = (__local_mem__ VAR_T*)(updatesLocal.GetPhyAddr());
    uint32_t currentMaxThread = uniqueIdNum * totalCol >= THREAD_NUM_SORT ? THREAD_NUM_SORT : uniqueIdNum * totalCol;
    int32_t threadBlock = currentMaxThread / totalCol;
    threadBlock = threadBlock < uniqueIdNum ? threadBlock : uniqueIdNum;

    Simt::VF_CALL<ScatterAddSimtSortCompute<IDX_T, VAR_T, CAST_T, ADDR_T, isUpdateScalar>>(
        Simt::Dim3({static_cast<uint32_t>(totalCol), static_cast<uint32_t>(threadBlock)}),
        varFirstDimSize, (__gm__ VAR_T*)(var_.GetPhyAddr()), updatesLocalPtr, indicesSortedPtr,
        (__local_mem__ uint32_t*)(updatesOriginIdxLocal.GetPhyAddr()),
        (__local_mem__ int32_t*)(uniqueIdCountLocal.GetPhyAddr()), uniqueIdNum, totalCol, updateScalarValue);

    indicesInQueue_.FreeTensor(indicesLocal);
    updatesInQueue_.FreeTensor(updatesLocal);
}

template <typename IDX_T, typename VAR_T, typename CAST_T, typename ADDR_T, bool isUpdateScalar, uint32_t castType>
__aicore__ inline void ScatterAddSimtSort<IDX_T, VAR_T, CAST_T, ADDR_T, isUpdateScalar, castType>::Process()
{
    uint32_t indicesCount = 0;
    VAR_T updateScalarValue = ((__gm__ VAR_T*)(updates_.GetPhyAddr()))[0];
    for (uint64_t idx = 0; idx < currLoopCount_; idx++) {
        indicesCount = (idx == currLoopCount_ - 1) ?
            static_cast<uint32_t>(ubTailLoopSize_) : static_cast<uint32_t>(td_.indicesFactor);
        CopyInIndicesUpdates(idx, indicesCount);
        Compute(idx, indicesCount, updateScalarValue);
    }
}
}  // namespace ScatterAdd

#endif