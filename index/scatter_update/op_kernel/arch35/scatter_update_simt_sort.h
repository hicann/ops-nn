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
 * \file scatter_update_simt.h
 * \brief scatter_update
 */
#ifndef ASCENDC_SCATTER_UPDATE_SIMT_SORT_H_
#define ASCENDC_SCATTER_UPDATE_SIMT_SORT_H_

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "scatter_update_common.h"
#include "scatter_update_struct.h"

namespace ScatterUpdate
{
using namespace AscendC;

#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM_SORT = 256;
#else
constexpr uint32_t THREAD_NUM_SORT = 2048;
#endif
constexpr float SIMT_SORT_HIST_THRESHOLD =  0.01f;

template <typename IDX_T, typename VAR_T, typename CAST_T, typename ADDR_T, bool isUpdateScalar, uint32_t castType>
class ScatterUpdateSimtSort
{
public:
    __aicore__ inline ScatterUpdateSimtSort(const ScatterUpdateTilingData& tilingData) : td_(tilingData){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInIndicds(uint32_t loopIdx, uint32_t dataCount);
    __aicore__ inline void Compute(uint32_t loopIdx, uint32_t indicesCount, VAR_T updateScalarValue);
    __aicore__ inline void ParseTilingData();

private:
    TPipe pipe_;
    GlobalTensor<VAR_T> var_;
    GlobalTensor<IDX_T> indices_;
    GlobalTensor<VAR_T> updates_;
    TQue<QuePosition::VECIN, 1> indicesInQueue_;
    TQue<QuePosition::VECCALC, 1> sortIndicesQue_;
    TQue<QuePosition::VECCALC, 1> updatesOriginIdexQue_;
    TQue<QuePosition::VECCALC, 1> uniqueIdCountQue_;
    TBuf<QuePosition::VECCALC> hashBuffer_;
    TBuf<QuePosition::VECCALC> castIndicesQue_;

    const ScatterUpdateTilingData& td_;

    ADDR_T blockIdx_{0};
    ADDR_T blockNum_{0};
    uint64_t ubTailLoopSize_ = 0;        // 当前coreUB尾循环搬运数据量
    uint64_t currLoopCount_ = 0;         // 当前core循环搬运数据次数
    static constexpr uint32_t shiftOffset_ = platform::GetUbBlockSize() / sizeof(IDX_T);
};

template <typename IDX_T, typename VAR_T, typename CAST_T, typename ADDR_T, bool isUpdateScalar, uint32_t castType>
__aicore__ inline void ScatterUpdateSimtSort<IDX_T, VAR_T, CAST_T, ADDR_T, isUpdateScalar, castType>::ParseTilingData()
{
    if (blockIdx_ == td_.usedCoreNum - 1) {
        currLoopCount_ = td_.tailBlockLoop;
        ubTailLoopSize_ = td_.tailBlockTail;
    } else {
        currLoopCount_ = td_.normBlockLoop;
        ubTailLoopSize_ = td_.indicesFactor;
    }
}

template <typename IDX_T, typename VAR_T, typename CAST_T, typename ADDR_T, bool isUpdateScalar, uint32_t castType>
__aicore__ inline void ScatterUpdateSimtSort<IDX_T, VAR_T, CAST_T, ADDR_T, isUpdateScalar, castType>::Init(GM_ADDR var, GM_ADDR indices,
                                                                                     GM_ADDR updates, GM_ADDR workspace)
{
    blockIdx_ = GetBlockIdx();
    blockNum_ = GetBlockNum();
    ParseTilingData();
    var_.SetGlobalBuffer((__gm__ VAR_T*)(var));
    indices_.SetGlobalBuffer((__gm__ IDX_T*)(indices));
    updates_.SetGlobalBuffer((__gm__ VAR_T*)(updates));
    pipe_.InitBuffer(indicesInQueue_, 1, ops::CeilAlign(td_.indicesFactor * sizeof(IDX_T), UB_AGLIN_VALUE));
    if constexpr (castType == CAST_NOT_CAST) {
        pipe_.InitBuffer(sortIndicesQue_, 1,
            ops::CeilAlign(td_.indicesFactor * sizeof(IDX_T), UB_AGLIN_VALUE) + SORT_PAD_NUM * UB_AGLIN_VALUE);
    } else {
        pipe_.InitBuffer(sortIndicesQue_, 1,
            ops::CeilAlign(td_.indicesFactor * sizeof(CAST_T), UB_AGLIN_VALUE) + SORT_PAD_NUM * UB_AGLIN_VALUE);
        pipe_.InitBuffer(castIndicesQue_, ops::CeilAlign(td_.indicesFactor * sizeof(CAST_T), UB_AGLIN_VALUE));
    }
    pipe_.InitBuffer(updatesOriginIdexQue_, 1, ops::CeilAlign(td_.indicesFactor * sizeof(uint32_t), UB_AGLIN_VALUE));
    pipe_.InitBuffer(uniqueIdCountQue_, 1, ops::CeilAlign((td_.indicesFactor + 1) * sizeof(int32_t), UB_AGLIN_VALUE));
    pipe_.InitBuffer(hashBuffer_, HASH_BUCKER_BUFFER_SIZE);
}

template <typename IDX_T, typename VAR_T, typename CAST_T, typename ADDR_T, bool isUpdateScalar, uint32_t castType>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_SORT) inline void ScatterUpdateSimtSortCompute(
    ADDR_T varFirstDimSize, ADDR_T magic, ADDR_T shift, __gm__ VAR_T* var, __gm__ VAR_T* currCalcUpdates,
    __local_mem__ IDX_T* indecesSorted, __local_mem__ uint32_t* updateOriginIdx, __local_mem__ int32_t* uniqueIdCount,
    ADDR_T blockIdx, uint32_t uniqueIdNum, ADDR_T totalCol, VAR_T updateScalarValue, ADDR_T varStride)
{
    ADDR_T totalSizeCurr = uniqueIdNum * totalCol;

    for (ADDR_T i = Simt::GetThreadIdx(); i < totalSizeCurr; i += Simt::GetThreadNum()) {
        ADDR_T indiceRow = Simt::UintDiv(i, magic, shift);        // 当前线程对应当前分块indices行
        ADDR_T tailRowIdx = i - indiceRow * totalCol;             // 获取当前indices对应updates行中的数，在当前updates行中的索引
        
        int32_t indicesIdx = uniqueIdCount[indiceRow];     // 不重复index的第一个排序后列表的索引位置
        IDX_T varIdx = indecesSorted[indicesIdx];          // index值
        uint32_t updatesIdx = updateOriginIdx[indicesIdx];  // 未排序索引时的位置

        if (!(varIdx >= 0 && varIdx < varFirstDimSize)) {
            continue;
        }

        if constexpr (isUpdateScalar) {
            var[varIdx * varStride + tailRowIdx] = updateScalarValue;
        } else {
            var[varIdx * varStride + tailRowIdx] = currCalcUpdates[updatesIdx * totalCol + tailRowIdx];
        }
    }
}

template <typename IDX_T, typename VAR_T, typename CAST_T, typename ADDR_T, bool isUpdateScalar, uint32_t castType>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ScatterUpdateSimtNoSortCompute(
    ADDR_T totalCol, ADDR_T currIndicesSize, ADDR_T varFirstDimSize, ADDR_T magic, ADDR_T shift, __gm__ VAR_T* var,
    __local_mem__ IDX_T* indices, __gm__ VAR_T* currCalcUpdates, VAR_T updateScalarValue, ADDR_T varStride)
{
    ADDR_T totalSizeCurr = currIndicesSize * totalCol;

    for (ADDR_T i = Simt::GetThreadIdx(); i < totalSizeCurr; i += Simt::GetThreadNum()) {
        ADDR_T indiceRow = Simt::UintDiv(i, magic, shift);
        ADDR_T tailRowIdx = i - indiceRow * totalCol;
        ADDR_T varRow = static_cast<ADDR_T>(indices[indiceRow]);
        ADDR_T varDataIdx = varRow * varStride + tailRowIdx;
        if (!(varRow >= 0 && varRow < varFirstDimSize)) {
            continue;
        }

        if constexpr (isUpdateScalar) {
            var[varDataIdx] = updateScalarValue;
        } else {
            var[varDataIdx] = currCalcUpdates[i];
        }
    }
}

template <typename IDX_T, typename VAR_T, typename CAST_T, typename ADDR_T, bool isUpdateScalar, uint32_t castType>
__aicore__ inline void ScatterUpdateSimtSort<IDX_T, VAR_T, CAST_T, ADDR_T, isUpdateScalar, castType>::CopyInIndicds(
                                                           uint32_t loopIdx, uint32_t indicesCount)
{
    LocalTensor<IDX_T> indicesLocal = indicesInQueue_.AllocTensor<IDX_T>();

    DataCopyExtParams indicesCopyParams { 1, (uint32_t)(indicesCount * sizeof(IDX_T)), 0, 0, 0 };
    DataCopyPadExtParams<IDX_T> indicesPadParams { false, 0, 0, 0 };
    DataCopyPad(indicesLocal, indices_[blockIdx_ * td_.normBlockIndices + loopIdx * td_.indicesFactor],
                                                                    indicesCopyParams, indicesPadParams);
    indicesInQueue_.EnQue<IDX_T>(indicesLocal);
}

template <typename IDX_T, typename VAR_T, typename CAST_T, typename ADDR_T, bool isUpdateScalar, uint32_t castType>
__aicore__ inline void ScatterUpdateSimtSort<IDX_T, VAR_T, CAST_T, ADDR_T, isUpdateScalar, castType>::Compute(
                                    uint32_t loopIdx, uint32_t indicesCount, VAR_T updateScalarValue)
{
    ADDR_T totalCol = static_cast<ADDR_T>(td_.varShape[1]);
    ADDR_T varFirstDimSize = static_cast<ADDR_T>(td_.varShape[0]);
    ADDR_T varStride = static_cast<ADDR_T>(td_.varStride);
    ADDR_T magic = 0;
    ADDR_T shift = 0;

    __gm__ VAR_T* currCalcUpdates = (__gm__ VAR_T*)(updates_.GetPhyAddr()) +
            td_.normBlockIndices * totalCol * blockIdx_ + td_.indicesFactor * totalCol * loopIdx;
    
    GetUintDivMagicAndShift(magic, shift, totalCol);
    LocalTensor<IDX_T> indicesLocal = indicesInQueue_.DeQue<IDX_T>();
    LocalTensor<CAST_T> indicesSortedLocal = sortIndicesQue_.AllocTensor<CAST_T>();
    LocalTensor<uint32_t> updatesOriginIdxLocal = updatesOriginIdexQue_.AllocTensor<uint32_t>();
    LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.AllocTensor<int32_t>();
    LocalTensor<float> hashLocal = hashBuffer_.Get<float>();
    float maxScore = 0.0f;
    if constexpr (IsSameType<IDX_T, int32_t>::value) {
        IndexStatisticInt32(indicesLocal, hashLocal, maxScore, indicesCount, totalCol);
    } else {
        IndexStatisticInt64(indicesLocal, hashLocal, maxScore, indicesCount, totalCol);
    }

    if (maxScore > SIMT_SORT_HIST_THRESHOLD) {
        __local_mem__ IDX_T* indicesSortedPtr = (__local_mem__ IDX_T*)(indicesSortedLocal.GetPhyAddr()) + shiftOffset_;
        uint32_t uniqueIdNum = 0;
        if constexpr (castType == CAST_NOT_CAST) {
            uniqueIdNum = SortAndComputeUniqueIdx<IDX_T>(
                            indicesCount, indicesLocal, indicesSortedLocal, uniqueIdCountLocal, updatesOriginIdxLocal);
        } else {
            LocalTensor<CAST_T> indicesCastLocal = castIndicesQue_.Get<CAST_T>();
 	        IndicesSortCast<IDX_T, CAST_T, castType>(indicesLocal, indicesCastLocal, uniqueIdCountLocal, indicesCount);
 	        uniqueIdNum = SortAndComputeUniqueIdx<CAST_T>(
 	                        indicesCount, indicesCastLocal, indicesSortedLocal, uniqueIdCountLocal, updatesOriginIdxLocal);
        }
         
        Simt::VF_CALL<ScatterUpdateSimtSortCompute<IDX_T, VAR_T, CAST_T, ADDR_T, isUpdateScalar, castType>>(Simt::Dim3(THREAD_NUM_SORT),
            varFirstDimSize, magic, shift, (__gm__ VAR_T*)(var_.GetPhyAddr()), currCalcUpdates, indicesSortedPtr,
            (__local_mem__ uint32_t*)(updatesOriginIdxLocal.GetPhyAddr()), (__local_mem__ int32_t*)(uniqueIdCountLocal.GetPhyAddr()),
            blockIdx_, uniqueIdNum, totalCol, updateScalarValue, varStride);
    } else {
        Simt::VF_CALL<ScatterUpdateSimtNoSortCompute<IDX_T, VAR_T, CAST_T, ADDR_T, isUpdateScalar, castType>>(Simt::Dim3(THREAD_NUM), 
            totalCol, indicesCount, varFirstDimSize, magic, shift, (__gm__ VAR_T*)(var_.GetPhyAddr()),
            (__local_mem__ IDX_T*)(indicesLocal.GetPhyAddr()), currCalcUpdates, updateScalarValue, varStride);
    }

    indicesInQueue_.FreeTensor(indicesLocal);
    sortIndicesQue_.FreeTensor(indicesSortedLocal);
    updatesOriginIdexQue_.FreeTensor(updatesOriginIdxLocal);
    uniqueIdCountQue_.FreeTensor(uniqueIdCountLocal);
}

template <typename IDX_T, typename VAR_T, typename CAST_T, typename ADDR_T, bool isUpdateScalar, uint32_t castType>
__aicore__ inline void ScatterUpdateSimtSort<IDX_T, VAR_T, CAST_T, ADDR_T, isUpdateScalar, castType>::Process()
{
    uint32_t indicesCount = 0;
    VAR_T updateScalarValue = static_cast<VAR_T>(((__gm__ VAR_T*)(updates_.GetPhyAddr()))[0]);
    for (uint64_t idx = 0; idx < currLoopCount_; idx++) {
        indicesCount = (idx == currLoopCount_ - 1) ?
            static_cast<uint32_t>(ubTailLoopSize_) : static_cast<uint32_t>(td_.indicesFactor);
        CopyInIndicds(idx, indicesCount);
        Compute(idx, indicesCount, updateScalarValue);
    }
}
}  // namespace ScatterUpdate

#endif