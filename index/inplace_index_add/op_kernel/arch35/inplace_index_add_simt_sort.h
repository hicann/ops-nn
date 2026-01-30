/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_index_add_simt_sort.h
 * \brief inplace_index_add
 */
#ifndef ASCENDC_INPLACE_INDEX_ADD_SIMT_SORT_H_
#define ASCENDC_INPLACE_INDEX_ADD_SIMT_SORT_H_

#include "kernel_operator.h"
#include "../inc/kernel_utils.h"
#include "inplace_index_add_common.h"

namespace InplaceIndexAdd
{
using namespace AscendC;

#ifdef __DAV_FPGA__
constexpr uint32_t USED_THREAD_SORT = 256;
#else
constexpr uint32_t USED_THREAD_SORT = 2048;
#endif
constexpr float SIMT_SORT_HIST_THRESHOLD =  0.01f;

template <typename T>
__aicore__ inline void CopyIn(
    LocalTensor<T> dstLocal, GlobalTensor<T> srcGm, uint64_t offset, uint32_t nBurst, uint32_t copyLen,
    uint32_t srcStride = 0, uint32_t dstStride = 0)
{
    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;                // 连续传输块个数
    dataCoptExtParams.blockLen = copyLen * sizeof(T);     // 每块大小
    dataCoptExtParams.srcStride = srcStride * sizeof(T);  // 源地址相邻块间隔
    dataCoptExtParams.dstStride = dstStride * sizeof(T);  // 目的地址相邻块间隔
    DataCopyPad(dstLocal, srcGm[offset], dataCoptExtParams, dataCopyPadExtParams);
}

template <typename T>
__aicore__ inline void CopyOut(
    GlobalTensor<T> dstGm, LocalTensor<T> srcLocal, uint64_t offset, uint32_t nBurst, uint32_t copyLen,
    uint32_t srcStride = 0, uint32_t dstStride = 0)
{
    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen * sizeof(T);
    dataCoptExtParams.srcStride = srcStride * sizeof(T);
    dataCoptExtParams.dstStride = dstStride * sizeof(T);
    DataCopyPad(dstGm[offset], srcLocal, dataCoptExtParams);
}

template <typename VAR_T, typename IDX_T, typename COMP_T, bool WITH_ALPHA, bool IS_CONTIGUOUS>
class InplaceIndexAddSimtSort : InplaceIndexAddBase<VAR_T, IDX_T>
{
public:
    __aicore__ inline InplaceIndexAddSimtSort(const InplaceIndexAddSimtSortTilingData& tilingData, TPipe& pipe)
        : td_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR alpha, GM_ADDR workspace);
    __aicore__ inline void CopyInIndices(int64_t loopIdx, uint32_t indicesCount);
    __aicore__ inline void CopyInUpdates(int64_t loopIdx, int64_t preIdx, uint32_t indicesCount, uint32_t updatesPreNum);
    __aicore__ inline void Compute(uint32_t uniqueIdNum, int64_t preIdx, uint32_t indicesCount, uint32_t updatesPreNum);
    __aicore__ inline void ComputeNoSort(int64_t loopIdx, uint32_t indicesCount, LocalTensor<IDX_T> indicesLocal);
    __aicore__ inline void Process();

private:
    static __simt_vf__ __aicore__ inline void SimtCompute(COMP_T varInAxis, uint32_t afterAxis, uint32_t uniqueIdNum, uint32_t updatesPreNum, uint32_t updatesStride,
                                           int64_t preIdx, __local_mem__ IDX_T* sortedAddr, __local_mem__ uint32_t* updatesOriginIdxAddr,
                                           __local_mem__ int32_t* uniqueIdCountAddr, __local_mem__ VAR_T* updatesLocalAddr, __gm__ VAR_T* varAddr, 
                                           __gm__ VAR_T* alpha, uint32_t m0, uint32_t shift0);
    static __simt_vf__ __aicore__ inline void SimtComputeNoSort(COMP_T varInAxis, uint32_t afterAxis, COMP_T preAxis, COMP_T updatesInAxis, COMP_T indicesOffset,
                                           uint32_t indicesCount, __local_mem__ IDX_T* indicesLocalAddr, __gm__ VAR_T* varAddr,
                                           __gm__ VAR_T* updatesAddr, __gm__ VAR_T* alpha, uint32_t m1, uint32_t shift1);

private:
    GlobalTensor<VAR_T> var_;
    GlobalTensor<IDX_T> indices_;
    GlobalTensor<VAR_T> updates_;
    GlobalTensor<VAR_T> alpha_;
    
    TQue<QuePosition::VECIN, 1> indicesInQueue_;
    TQue<QuePosition::VECIN, 1> updatesInQueue_;
    TBuf<QuePosition::VECCALC> sortIndicesQue_;
    TBuf<QuePosition::VECCALC> updatesOriginIdexQue_;
    TBuf<QuePosition::VECCALC> uniqueIdCountQue_;
    TBuf<QuePosition::VECCALC> sharedTmpBuf_;
    TBuf<QuePosition::VECCALC> hashBuffer_;
    TPipe& pipe_;
    const InplaceIndexAddSimtSortTilingData& td_;

    COMP_T blockIdx_;
    COMP_T blockNum_;
    int64_t normBlockData_{0};
    int64_t usedCoreNum_{0};
    int64_t loopNum_{0};
    int64_t indicesTailLoopLength_{0};
    int64_t indicesLoopCount_{0};
    static constexpr uint32_t shiftOffset_ = platform::GetUbBlockSize() / sizeof(IDX_T);
};

template <typename VAR_T, typename IDX_T, typename COMP_T, bool WITH_ALPHA, bool IS_CONTIGUOUS>
__aicore__ inline void InplaceIndexAddSimtSort<VAR_T, IDX_T, COMP_T, WITH_ALPHA, IS_CONTIGUOUS>::Init(GM_ADDR var, GM_ADDR indices,
                                                                                       GM_ADDR updates,
                                                                                       GM_ADDR alpha,
                                                                                       GM_ADDR workspace)
{
    blockIdx_ = GetBlockIdx();
    var_.SetGlobalBuffer((__gm__ VAR_T*)(var));
    indices_.SetGlobalBuffer((__gm__ IDX_T*)(indices));
    updates_.SetGlobalBuffer((__gm__ VAR_T*)(updates));
    alpha_.SetGlobalBuffer((__gm__ VAR_T*)(alpha));

    pipe_.InitBuffer(indicesInQueue_, 1, ops::CeilAlign(td_.indicesUbFactor * sizeof(IDX_T), UB_AGLIN_VALUE));
    pipe_.InitBuffer(updatesInQueue_, 1,
        ops::CeilAlign(td_.indicesUbFactor * td_.afterAxis * sizeof(VAR_T), UB_AGLIN_VALUE) * td_.normalUpdatesPreNum);
    pipe_.InitBuffer(sortIndicesQue_, ops::CeilAlign(td_.indicesUbFactor * sizeof(IDX_T), UB_AGLIN_VALUE) + SORT_PAD_NUM * UB_AGLIN_VALUE);
    pipe_.InitBuffer(updatesOriginIdexQue_, ops::CeilAlign(td_.indicesUbFactor * sizeof(uint32_t), UB_AGLIN_VALUE));
    pipe_.InitBuffer(uniqueIdCountQue_, ops::CeilAlign(td_.indicesUbFactor * sizeof(int32_t), UB_AGLIN_VALUE) + SORT_PAD_NUM * UB_AGLIN_VALUE);
    pipe_.InitBuffer(sharedTmpBuf_, ops::Aligned(static_cast<uint64_t>(td_.sortShareBufSize), UB_AGLIN_VALUE));
    pipe_.InitBuffer(hashBuffer_, HASH_SCORE_BUF_SIZE * sizeof(float));

    if (blockIdx_ == td_.usedCoreNum - 1) {
        indicesLoopCount_ = td_.tailBlockIndicesLoopSize;
        indicesTailLoopLength_ = td_.indiceAxisTailNum;
    } else {
        indicesLoopCount_ = td_.indicesLoopSize;
        indicesTailLoopLength_ = td_.indicesUbFactor;
    }
}

template <typename VAR_T, typename IDX_T, typename COMP_T, bool WITH_ALPHA, bool IS_CONTIGUOUS>
__aicore__ inline void InplaceIndexAddSimtSort<VAR_T, IDX_T, COMP_T, WITH_ALPHA, IS_CONTIGUOUS>::CopyInIndices(
                                                               int64_t loopIdx, uint32_t indicesCount)
{
    LocalTensor<IDX_T> indicesLocal = indicesInQueue_.AllocTensor<IDX_T>();
    uint64_t offset = blockIdx_ * td_.eachCoreIndexCount + loopIdx * td_.indicesUbFactor;
    if constexpr (IS_CONTIGUOUS) {
        CopyIn<IDX_T>(indicesLocal, indices_, offset, 1, indicesCount);
    } else {
        CopyInNoContiguous<IDX_T>(indicesLocal, indices_[offset * td_.indicesStride], indicesCount, 1, td_.indicesStride - 1);
    }                                                          
    indicesInQueue_.EnQue<IDX_T>(indicesLocal);
}

template <typename VAR_T, typename IDX_T, typename COMP_T, bool WITH_ALPHA, bool IS_CONTIGUOUS>
__aicore__ inline void InplaceIndexAddSimtSort<VAR_T, IDX_T, COMP_T, WITH_ALPHA, IS_CONTIGUOUS>::CopyInUpdates(
                                int64_t loopIdx, int64_t preIdx, uint32_t indicesCount, uint32_t updatesPreNum)
{
    LocalTensor<VAR_T> updatesLocal = updatesInQueue_.AllocTensor<VAR_T>();
    uint64_t offset = preIdx * td_.updatesInAxis * td_.afterAxis +
                      blockIdx_ * td_.eachCoreIndexCount * td_.afterAxis +
                      loopIdx * td_.indicesUbFactor * td_.afterAxis;
    uint32_t copyLen = indicesCount * td_.afterAxis;
    uint32_t srcStride = td_.updatesInAxis * td_.afterAxis - indicesCount * td_.afterAxis;
    CopyIn<VAR_T>(updatesLocal, updates_, offset, updatesPreNum, copyLen, srcStride, 0);                                                                   
    updatesInQueue_.EnQue<VAR_T>(updatesLocal);
}

template <typename VAR_T, typename IDX_T, typename COMP_T, bool WITH_ALPHA, bool IS_CONTIGUOUS>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD_SORT) inline void InplaceIndexAddSimtSort<VAR_T, IDX_T, COMP_T, WITH_ALPHA, IS_CONTIGUOUS>::SimtCompute(
    COMP_T varInAxis, uint32_t afterAxis, uint32_t uniqueIdNum, uint32_t updatesPreNum, uint32_t updatesStride, int64_t preIdx,
    __local_mem__ IDX_T* sortedAddr, __local_mem__ uint32_t* updatesOriginIdxAddr, __local_mem__ int32_t* uniqueIdCountAddr,
    __local_mem__ VAR_T* updatesLocalAddr, __gm__ VAR_T* varAddr, __gm__ VAR_T* alpha, uint32_t m0, uint32_t shift0)
{
    COMP_T varStride = varInAxis * afterAxis;
    VAR_T alphaValue = 1;
    if constexpr (WITH_ALPHA) {
        alphaValue = alpha[0];
    }
    float alphaValueFloat = static_cast<float>(alphaValue);

    uint32_t threadBlockIdx = Simt::GetThreadIdx<1>();
    uint32_t threadBlockNum = Simt::GetThreadNum<1>();
    uint32_t innerOffset = Simt::GetThreadIdx<0>();
    
    for(uint32_t i = threadBlockIdx; i < uniqueIdNum * updatesPreNum; i += threadBlockNum) {  // 每个循环对应多个pre中其中一行updates
        uint32_t preInternalIdx = Simt::UintDiv(i, m0, shift0);  // 计算当前线程处于当前pre循环第几个pre
        uint32_t idx = i - uniqueIdNum * preInternalIdx;         // 计算当前线程对应排序后无重复indices的第几个
        if (sortedAddr[uniqueIdCountAddr[idx]] < 0 || sortedAddr[uniqueIdCountAddr[idx]] >= varInAxis) {
            continue;
        }

        VAR_T result = 0;
        float reslut_f = 0;
        for (int32_t tid = 0; tid < uniqueIdCountAddr[idx + 1] - uniqueIdCountAddr[idx]; tid++) {  // 每个循环遍历重复的indices
            int32_t srcOffset = updatesOriginIdxAddr[uniqueIdCountAddr[idx] + tid] * afterAxis + innerOffset;  // 获取当前update在当前pre的偏移
            srcOffset = srcOffset + updatesStride * preInternalIdx;

            if constexpr (WITH_ALPHA) {
                if constexpr (IsSameType<VAR_T, bfloat16_t>::value || IsSameType<VAR_T, half>::value) {
                    reslut_f += (static_cast<float>(updatesLocalAddr[srcOffset]) * alphaValueFloat);
                } else {
                    result += (updatesLocalAddr[srcOffset] * alphaValue);
                }
            } else {
                result += (updatesLocalAddr[srcOffset]);
            }
        }

        int64_t gmDstOffset = varStride * (preIdx + preInternalIdx) + sortedAddr[uniqueIdCountAddr[idx]] * afterAxis + innerOffset;
        if constexpr (IsSameType<VAR_T, bfloat16_t>::value || IsSameType<VAR_T, half>::value) {
            Simt::AtomicAdd(varAddr + gmDstOffset, static_cast<VAR_T>(reslut_f));
        } else {
            Simt::AtomicAdd(varAddr + gmDstOffset, result);
        }
    }
}

template <typename VAR_T, typename IDX_T, typename COMP_T, bool WITH_ALPHA, bool IS_CONTIGUOUS>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD_SORT) inline void InplaceIndexAddSimtSort<VAR_T, IDX_T, COMP_T, WITH_ALPHA, IS_CONTIGUOUS>::SimtComputeNoSort(
            COMP_T varInAxis, uint32_t afterAxis, COMP_T preAxis, COMP_T updatesInAxis, COMP_T indicesOffset, uint32_t indicesCount,
            __local_mem__ IDX_T* indicesLocalAddr, __gm__ VAR_T* varAddr, __gm__ VAR_T* updatesAddr, __gm__ VAR_T* alpha, uint32_t m1, uint32_t shift1)
{
    COMP_T varStride = varInAxis * afterAxis;
    COMP_T updatesStride = updatesInAxis * afterAxis;
    VAR_T alphaValue = 1;
    if constexpr (WITH_ALPHA) {
        alphaValue = alpha[0];
    }
    float alphaValueFloat = static_cast<float>(alphaValue);

    uint32_t threadBlockIdx = Simt::GetThreadIdx<1>();
    uint32_t threadBlockNum = Simt::GetThreadNum<1>();
    uint32_t innerOffset = Simt::GetThreadIdx<0>();

    for(uint32_t i = threadBlockIdx; i < indicesCount * preAxis; i += threadBlockNum) {  // 每个循环对应其中一行updates
        uint32_t preIdx = Simt::UintDiv(i, m1, shift1);                                      // 计算当前线程处于第几个pre
        uint32_t indicesLocalIdx = i - indicesCount * preIdx;     // 计算当前线程对应第几个indices

        if (indicesLocalAddr[indicesLocalIdx] < 0 || indicesLocalAddr[indicesLocalIdx] >= varInAxis) {
            continue;
        }
        VAR_T result = 0;
        float reslut_f = 0;

        int32_t srcOffset = (indicesLocalIdx + indicesOffset) * afterAxis + innerOffset;  // 获取当前updates在当前pre的偏移
        srcOffset = srcOffset + updatesStride * preIdx;                                   // 当前updates总偏移
        if constexpr (WITH_ALPHA) {
            if constexpr (IsSameType<VAR_T, bfloat16_t>::value || IsSameType<VAR_T, half>::value) {
                reslut_f += (static_cast<float>(updatesAddr[srcOffset]) * alphaValueFloat);
            } else {
                result += (updatesAddr[srcOffset] * alphaValue);
            }
        } else {
            result += (updatesAddr[srcOffset]);
        }
        

        int64_t gmDstOffset = varStride * preIdx + indicesLocalAddr[indicesLocalIdx] * afterAxis + innerOffset;
        if constexpr (IsSameType<VAR_T, bfloat16_t>::value || IsSameType<VAR_T, half>::value) {
            Simt::AtomicAdd(varAddr + gmDstOffset, static_cast<VAR_T>(reslut_f));
        } else {
            Simt::AtomicAdd(varAddr + gmDstOffset, result);
        }
    }
}

template <typename VAR_T, typename IDX_T, typename COMP_T, bool WITH_ALPHA, bool IS_CONTIGUOUS>
__aicore__ inline void InplaceIndexAddSimtSort<VAR_T, IDX_T, COMP_T, WITH_ALPHA, IS_CONTIGUOUS>::Compute(
                    uint32_t uniqueIdNum, int64_t preIdx, uint32_t indicesCount, uint32_t updatesPreNum)
{
    uint32_t afterAxis = td_.afterAxis;
    COMP_T varInAxis = td_.varInAxis;
    uint32_t updatesLocalStride = ops::CeilAlign(indicesCount * afterAxis, static_cast<uint32_t>(UB_AGLIN_VALUE / sizeof(VAR_T)));  // 每次处理的indices数量对应1个pre的updates对齐的数量
    
    LocalTensor<VAR_T> updatesLocal = updatesInQueue_.DeQue<VAR_T>();
    LocalTensor<IDX_T> indicesSortedLocal = sortIndicesQue_.Get<IDX_T>();
    LocalTensor<uint32_t> updatesOriginIdxLocal = updatesOriginIdexQue_.Get<uint32_t>();
    LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.Get<int32_t>();
    LocalTensor<uint8_t> sharedTmpBuffer = sharedTmpBuf_.Get<uint8_t>();

    __local_mem__ IDX_T* indicesSortedPtr = (__local_mem__ IDX_T*)(indicesSortedLocal.GetPhyAddr()) + shiftOffset_;
    __local_mem__ VAR_T* updatesLocalPtr = (__local_mem__ VAR_T*)(updatesLocal.GetPhyAddr());
    __local_mem__ uint32_t* updatesOriginIdxPtr = (__local_mem__ uint32_t*)(updatesOriginIdxLocal.GetPhyAddr());
    __local_mem__ int32_t* uniqueIdCountPtr = (__local_mem__ int32_t*)(uniqueIdCountLocal.GetPhyAddr());

    uint32_t m0 = 1;
    uint32_t shift0 = 1;
    GetUintDivMagicAndShift(m0, shift0, uniqueIdNum);
    COMP_T calcCount = afterAxis * uniqueIdNum * updatesPreNum;
    uint32_t currentMaxThread = calcCount >= USED_THREAD_SORT ? USED_THREAD_SORT : calcCount;
    uint32_t threadBlock = currentMaxThread / afterAxis;
    threadBlock = threadBlock < uniqueIdNum * updatesPreNum ? threadBlock : uniqueIdNum * updatesPreNum;

    Simt::VF_CALL<InplaceIndexAddSimtSort<VAR_T, IDX_T, COMP_T, WITH_ALPHA, IS_CONTIGUOUS>::SimtCompute>(
        Simt::Dim3({afterAxis, threadBlock}), varInAxis, afterAxis, uniqueIdNum, updatesPreNum, updatesLocalStride,
        preIdx, indicesSortedPtr, updatesOriginIdxPtr, uniqueIdCountPtr, updatesLocalPtr,
        (__gm__ VAR_T*)(var_.GetPhyAddr()), (__gm__ VAR_T*)(alpha_.GetPhyAddr()), m0, shift0);

    updatesInQueue_.FreeTensor(updatesLocal);
}

template <typename VAR_T, typename IDX_T, typename COMP_T, bool WITH_ALPHA, bool IS_CONTIGUOUS>
__aicore__ inline void InplaceIndexAddSimtSort<VAR_T, IDX_T, COMP_T, WITH_ALPHA, IS_CONTIGUOUS>::ComputeNoSort(
                                   int64_t loopIdx, uint32_t indicesCount, LocalTensor<IDX_T> indicesLocal)
{
    uint32_t afterAxis = td_.afterAxis;
    COMP_T preAxis = td_.preAxis;
    COMP_T varInAxis = td_.varInAxis;
    COMP_T updatesInAxis = td_.updatesInAxis;
    COMP_T indicesOffset = loopIdx * td_.indicesUbFactor;
    
    __local_mem__ IDX_T* indicesLocalPtr = (__local_mem__ IDX_T*)(indicesLocal.GetPhyAddr());

    uint32_t m1 = 1;
    uint32_t shift1 = 1;
    GetUintDivMagicAndShift(m1, shift1, indicesCount);

    COMP_T calaUpdatesCount = indicesCount * preAxis * afterAxis;
    uint32_t currentMaxThread = calaUpdatesCount >= USED_THREAD_SORT ? USED_THREAD_SORT : calaUpdatesCount;
    uint32_t threadBlock = currentMaxThread / afterAxis;
    threadBlock = threadBlock < indicesCount * preAxis ? threadBlock : indicesCount * preAxis ;

    Simt::VF_CALL<InplaceIndexAddSimtSort<VAR_T, IDX_T, COMP_T, WITH_ALPHA, IS_CONTIGUOUS>::SimtComputeNoSort>(Simt::Dim3({afterAxis, threadBlock}),
        varInAxis, afterAxis, preAxis, updatesInAxis, indicesOffset, indicesCount, indicesLocalPtr,
        (__gm__ VAR_T*)(var_.GetPhyAddr()), (__gm__ VAR_T*)(updates_.GetPhyAddr()), (__gm__ VAR_T*)(alpha_.GetPhyAddr()), m1, shift1);
}

template <typename VAR_T, typename IDX_T, typename COMP_T, bool WITH_ALPHA, bool IS_CONTIGUOUS>
__aicore__ inline void InplaceIndexAddSimtSort<VAR_T, IDX_T, COMP_T, WITH_ALPHA, IS_CONTIGUOUS>::Process()
{
    uint32_t indicesCount = 0;
    for (int64_t idx = 0; idx < indicesLoopCount_; idx++) {
        indicesCount = (idx == indicesLoopCount_ - 1) ?
            static_cast<uint32_t>(indicesTailLoopLength_) : static_cast<uint32_t>(td_.indicesUbFactor);
        CopyInIndices(idx, indicesCount);
        LocalTensor<IDX_T> indicesLocal = indicesInQueue_.DeQue<IDX_T>();
        LocalTensor<float> hashLocal = hashBuffer_.Get<float>();
        float maxScore = 0.0f;
        if constexpr (IsSameType<IDX_T, int32_t>::value) {
            IndexStatisticInt32<IDX_T>(indicesLocal, hashLocal, maxScore, static_cast<int64_t>(indicesCount), td_.afterAxis);
        } else {
            IndexStatisticInt64<IDX_T>(indicesLocal, hashLocal, maxScore, static_cast<int64_t>(indicesCount), td_.afterAxis);
        }

        if (maxScore > SIMT_SORT_HIST_THRESHOLD) {
            LocalTensor<IDX_T> indicesSortedLocal = sortIndicesQue_.Get<IDX_T>();
            LocalTensor<uint32_t> updatesOriginIdxLocal = updatesOriginIdexQue_.Get<uint32_t>();
            LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.Get<int32_t>();
            LocalTensor<uint8_t> sharedTmpBuffer = sharedTmpBuf_.Get<uint8_t>();
            uint32_t uniqueIdNum = this->SortAndComputeUniqueIdx(
                indicesCount, indicesLocal, indicesSortedLocal, uniqueIdCountLocal, updatesOriginIdxLocal, sharedTmpBuffer);
            uint32_t updatesPreNum = 0;
            for (int64_t preLoopIdx = 0; preLoopIdx < td_.updatesPreLoop; preLoopIdx++) {
                updatesPreNum = (preLoopIdx == td_.updatesPreLoop - 1) ?
                    static_cast<uint32_t>(td_.tailUpdatesPreNum) : static_cast<uint32_t>(td_.normalUpdatesPreNum);
                CopyInUpdates(idx, preLoopIdx * td_.normalUpdatesPreNum, indicesCount, updatesPreNum);
                Compute(uniqueIdNum, preLoopIdx * td_.normalUpdatesPreNum, indicesCount, updatesPreNum);
            }
        } else {
            ComputeNoSort(idx, indicesCount, indicesLocal);
        }

        indicesInQueue_.FreeTensor(indicesLocal);
    }
}
}  // namespace InplaceIndexAdd

#endif