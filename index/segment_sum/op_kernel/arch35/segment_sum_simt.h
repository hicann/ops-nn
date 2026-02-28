/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SEGMENT_SUM_SIMT_H
#define SEGMENT_SUM_SIMT_H

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "simt_api/common_functions.h"
#include "simt_api/asc_simt.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_bf16.h"
#include "segment_sum_struct.h"


namespace SegmentSum
{
constexpr uint32_t TMP_ROWS_PER_CORE = 2;
constexpr uint32_t TMP_ROWS_TOTAL = 128;
constexpr uint32_t MAX_THREAD_NUM = 2048;
constexpr uint32_t UINT32_MAX_VALUE = 4294967295;
using namespace AscendC;

template <typename TX, typename Index>
class SegmentSumSimt
{
public:
    __aicore__ inline SegmentSumSimt(const SegmentSumSimtTilingData* __restrict tilingData, TPipe* pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output, GM_ADDR workspace);
    __aicore__ inline void Process();
    __aicore__ inline void CopySegmentIdsIn(LocalTensor<Index> segmentIdsLocal, int64_t offset, int32_t length);
    __aicore__ inline int32_t GetUniqueSegIdCount(uint32_t segmentIdsNum);
    __aicore__ inline void ProcessEachLoop(uint32_t segmentIdsNum, uint64_t baseOffset);
    __aicore__ inline void ProcessEachLoopForDeterminstic(uint32_t segmentIdsNum, uint64_t baseOffset);

private:
    GlobalTensor<TX> xGm, outputGm, outputGmInit, tmpRowWs;
    GlobalTensor<Index> segmentIdsGm, tmpIdWs;
    TBuf<TPosition::VECCALC> uniqueIdPosBuf_;
    TBuf<TPosition::VECCALC> segmentIdsBuf_;
    TPipe* pipe_;
    const SegmentSumSimtTilingData* tilingData_;
    int32_t loopTimes;
    uint32_t segIds, tailSegIds;  // 整/尾循环处理的id个数
    uint32_t threadNum, threadBlock;
    LocalTensor<int32_t> uniqueIdPosLocal;
    LocalTensor<Index> segmentIdsLocalShift, segmentIdsLocal;
    int32_t coreId;
    int32_t coreNum;
};

template <typename TX, typename Index>
__aicore__ inline void SegmentSumSimt<TX, Index>::Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output, GM_ADDR workspace)
{
    // common
    coreId = GetBlockIdx();
    coreNum = GetBlockNum();
    xGm.SetGlobalBuffer((__gm__ TX*)(x));
    segmentIdsGm.SetGlobalBuffer((__gm__ Index*)(segmentIds));
    outputGm.SetGlobalBuffer((__gm__ TX*)(output));
    pipe_->InitBuffer(uniqueIdPosBuf_, tilingData_->maxSegIdsInUb * sizeof(Index));
    pipe_->InitBuffer(segmentIdsBuf_, tilingData_->maxSegIdsInUb * sizeof(Index) + platform::GetUbBlockSize());
     // determinstic
    if (tilingData_->isDeterministic == 1) {
        tmpIdWs.SetGlobalBuffer((__gm__ Index*)(workspace));
        tmpRowWs.SetGlobalBuffer((__gm__ TX*)((__gm__ Index*)workspace + TMP_ROWS_TOTAL));   
    }
    // clear
    uint64_t initCoreReal = coreId == (coreNum - 1) ? 
                                                tilingData_->initNumTailCore : tilingData_->initNumPerCore;
    uint64_t outputGmOffset = coreId * tilingData_->initNumPerCore;
    outputGmInit.SetGlobalBuffer((__gm__ TX *)(output) + outputGmOffset);
    InitGlobalMemory(outputGmInit, initCoreReal, static_cast<TX>(0));
    SyncAll();
}

template <typename TX, typename Index>
__aicore__ inline void SegmentSumSimt<TX, Index>::CopySegmentIdsIn(
    LocalTensor<Index> segmentIdsLocal, int64_t offset, int32_t length)
{
    DataCopyPadExtParams<Index> dataCopyPadExtParams = {false, 0, 0, 0};
    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = 1;
    dataCoptExtParams.blockLen = length * sizeof(Index);
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;
    
    DataCopyPad(segmentIdsLocal, segmentIdsGm[offset], dataCoptExtParams, dataCopyPadExtParams);
    return;
}

template <typename TX, typename Index>
__aicore__ inline int32_t SegmentSumSimt<TX, Index>::GetUniqueSegIdCount(uint32_t segmentIdsNum)
{
    __ubuf__ Index* segmentIdsAddr = (__ubuf__ Index*)segmentIdsLocalShift.GetPhyAddr();
    __ubuf__ int32_t* segIdsPosAddr = (__ubuf__ int32_t*)uniqueIdPosLocal.GetPhyAddr();
    uint32_t vl = platform::GetVRegSize() / sizeof(Index);
    uint16_t loopCnt = (uint16_t)(ops::CeilDiv(static_cast<uint32_t>(segmentIdsNum), vl));
    uint32_t maskCount = segmentIdsNum;
    uint32_t offset = platform::GetUbBlockSize() / sizeof(Index);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> orderReg;
        AscendC::MicroAPI::RegTensor<int32_t> selReg;
        AscendC::MicroAPI::RegTensor<Index> indicesReg;
        AscendC::MicroAPI::RegTensor<Index> indicesShiftOneReg;
        AscendC::MicroAPI::MaskReg cmpMask;
        AscendC::MicroAPI::MaskReg maskRegUpdate;
        AscendC::MicroAPI::UnalignReg u0;
        MicroAPI::UnalignReg ureg;
        AscendC::MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();
        int32_t vciStart = 0;
        for (uint16_t i = 0; i < loopCnt; ++i) {
            vciStart = i * vl;
            auto segIdsOffset = segmentIdsAddr + offset + i * vl;
            AscendC::MicroAPI::Arange(orderReg, vciStart);
            maskRegUpdate = AscendC::MicroAPI::UpdateMask<Index>(maskCount);
            AscendC::MicroAPI::DataCopy(indicesReg, segIdsOffset);
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, segIdsOffset - 1);
            AscendC::MicroAPI::DataCopyUnAlign<Index>(indicesShiftOneReg, u0, segIdsOffset - 1);
            AscendC::MicroAPI::Compare<Index, CMPMODE::NE>(cmpMask, indicesReg, indicesShiftOneReg, maskRegUpdate);

            if constexpr (IsSameType<Index, int64_t>::value) {
                AscendC::MicroAPI::MaskReg maskHalf;
                AscendC::MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskHalf, cmpMask);
                // vSQZ
                AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(
                    selReg, orderReg, maskHalf);
            } else {
                // vSQZ
                AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(
                    selReg, orderReg, cmpMask);
            }
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                segIdsPosAddr, selReg, ureg);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(segIdsPosAddr, ureg);
    }
    return ((AscendC::MicroAPI::GetSpr<AscendC::SpecialPurposeReg::AR>()) / sizeof(int32_t));
}

template <typename TX, typename Index>
__simt_vf__ __launch_bounds__(MAX_THREAD_NUM) inline void Compute(
    __gm__ TX* xAddr, __gm__ Index* segmentIdsAddr, __gm__ TX* outputAddr, __local_mem__ uint32_t* uniqueIdPosAddr,
    uint32_t colSize, uint32_t uniqueIdNum, uint32_t baseOffset, uint32_t idsNum)
{
    for (uint32_t i = threadIdx.y; i < uniqueIdNum; i += blockDim.y) {
        TX res = 0;
        uint32_t nextIdPos = (i != uniqueIdNum - 1) ? uniqueIdPosAddr[i+1] : idsNum;
        for (int32_t j = uniqueIdPosAddr[i]; j < nextIdPos; j++) {
            int32_t offset = (baseOffset + j) * colSize + threadIdx.x;
            res += xAddr[offset];
        }
        int32_t outputOffset = segmentIdsAddr[baseOffset + uniqueIdPosAddr[i]] * colSize + threadIdx.x;
        if (i == 0 || i == uniqueIdNum - 1) {
            asc_atomic_add(outputAddr + outputOffset, res);
        } else {
            outputAddr[outputOffset] = res;
        }
    }
    return;
}

template <typename TX, typename Index>
__simt_vf__ __launch_bounds__(MAX_THREAD_NUM) inline void ComputeForDeterminstic(
    __gm__ TX* xAddr, __gm__ Index* segmentIdsAddr, __gm__ TX* outputAddr, __local_mem__ uint32_t* uniqueIdPosAddr,
    __gm__ TX* tmpRowWs, __gm__ Index* tmpIdWs, uint32_t colSize, uint32_t uniqueIdNum, uint32_t baseOffset, uint32_t idsNum)
{
    if (threadIdx.y == 0) {   // use thread(0/1, 0) clear segids workspace
        for (uint32_t i = threadIdx.x; i < TMP_ROWS_PER_CORE; i++) {
            tmpIdWs[get_block_idx() * TMP_ROWS_PER_CORE + i] = static_cast<Index>(-1);
        }
    }
    for (uint32_t i = threadIdx.y; i < uniqueIdNum; i += blockDim.y) {
        TX res = 0;
        uint32_t nextIdPos = (i != uniqueIdNum - 1) ? uniqueIdPosAddr[i+1] : idsNum;
        for (int32_t j = uniqueIdPosAddr[i]; j < nextIdPos; j++) {
            int32_t offset = (baseOffset + j) * colSize + threadIdx.x;
            res += xAddr[offset];
        }
        if (i == 0 || i == uniqueIdNum - 1) {   // 如果uniqueid个数是1，那么只搬首行，那么尾行在ws就是随机值+（-1）id
            int32_t bias = (i == 0) ? 0 : 1;
            int32_t offsetInWs = get_block_idx() * TMP_ROWS_PER_CORE + bias;
            tmpRowWs[offsetInWs * colSize + threadIdx.x] = res;
            if (threadIdx.x == 0) {
                tmpIdWs[offsetInWs] = segmentIdsAddr[baseOffset + uniqueIdPosAddr[i]];
            }
        } else {
            outputAddr[segmentIdsAddr[baseOffset + uniqueIdPosAddr[i]] * colSize + threadIdx.x] = res;
        }
    }
    return;
}

template <typename TX, typename Index>
__simt_vf__ __launch_bounds__(MAX_THREAD_NUM) inline void ComputeInWs(__gm__ TX* tmpRowWs, __gm__ Index* segmentIdsWs, 
                                                                      __gm__ TX* outputAddr, uint32_t colSize)
{
    for (uint32_t i = threadIdx.y; i < TMP_ROWS_TOTAL; i += blockDim.y) {
        if (segmentIdsWs[i] == -1) {
            continue;
        }
        if (i > 0) {
            int32_t prevId = (segmentIdsWs[i - 1] == -1) ? (i - 2) : (i - 1);
            if (segmentIdsWs[i] == segmentIdsWs[prevId]) {  // 如果当前线程组对应id非0且跟前一行id相同，则跳过
                continue;
            }
        }
        TX res = tmpRowWs[i * colSize + threadIdx.x];  // 走到这里要么是第0组的线程，要么是当前线程组对应的id跟上一个非-1 id不同，就是一个新id的起始
        for (Index j = i + 1; j < TMP_ROWS_TOTAL; j++) {  // 从当前组线程开始往后加，知道id跟自己对应的id不同
            if (segmentIdsWs[j] == -1) {
                continue;
            }
            if (segmentIdsWs[i] != segmentIdsWs[j]) {
                break;
            }
            res += tmpRowWs[j * colSize + threadIdx.x];
        }
        asc_atomic_add(outputAddr + segmentIdsWs[i] * colSize + threadIdx.x, res);
    }   
}    


template <typename TX, typename Index>
__aicore__ inline void SegmentSumSimt<TX, Index>::ProcessEachLoop(uint32_t segmentIdsNum, uint64_t baseOffset)
{
    auto vectorWaitMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(vectorWaitMTE2);
    WaitFlag<HardEvent::MTE2_V>(vectorWaitMTE2);
    Duplicate(segmentIdsLocalShift, static_cast<Index>(-1), platform::GetUbBlockSize() / sizeof(Index));
    uint32_t uniqueIdNum = GetUniqueSegIdCount(segmentIdsNum);
    asc_vf_call<Compute<TX, Index>>(dim3{threadNum, threadBlock}, 
                                    (__gm__ TX*)xGm.GetPhyAddr(),
                                    (__gm__ Index*)segmentIdsGm.GetPhyAddr(),
                                    (__gm__ TX*)outputGm.GetPhyAddr(),
                                    (__local_mem__ uint32_t*)(uniqueIdPosLocal.GetPhyAddr()),
                                    tilingData_->innerDim, uniqueIdNum, baseOffset, segmentIdsNum);
}

template <typename TX, typename Index>
__aicore__ inline void SegmentSumSimt<TX, Index>::ProcessEachLoopForDeterminstic(uint32_t segmentIdsNum, uint64_t baseOffset)
{
    auto vectorWaitMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(vectorWaitMTE2);
    WaitFlag<HardEvent::MTE2_V>(vectorWaitMTE2);
    Duplicate(segmentIdsLocalShift, static_cast<Index>(-1), platform::GetUbBlockSize() / sizeof(Index));
    uint32_t uniqueIdNum = GetUniqueSegIdCount(segmentIdsNum);
    asc_vf_call<ComputeForDeterminstic<TX, Index>>(dim3{threadNum, threadBlock},
                                                (__gm__ TX*)xGm.GetPhyAddr(), 
                                                (__gm__ Index*)segmentIdsGm.GetPhyAddr(), 
                                                (__gm__ TX*)outputGm.GetPhyAddr(), 
                                                (__local_mem__ uint32_t*)(uniqueIdPosLocal.GetPhyAddr()),
                                                (__gm__ TX*)tmpRowWs.GetPhyAddr(), 
                                                (__gm__ Index*)tmpIdWs.GetPhyAddr(), 
                                                tilingData_->innerDim, uniqueIdNum, baseOffset, segmentIdsNum);
    SyncAll();  // wait for all core write ws
    if (coreId > 0) {
        return;
    }
    asc_vf_call<ComputeInWs<TX, Index>>(dim3{threadNum, threadBlock},
                                        (__gm__ TX*)tmpRowWs.GetPhyAddr(), 
                                        (__gm__ Index*)tmpIdWs.GetPhyAddr(), 
                                        (__gm__ TX*)outputGm.GetPhyAddr(),
                                        tilingData_->innerDim);  
}

template <typename TX, typename Index>
__aicore__ inline void SegmentSumSimt<TX, Index>::Process()
{
    uniqueIdPosLocal = uniqueIdPosBuf_.Get<int32_t>();
    segmentIdsLocalShift = segmentIdsBuf_.Get<Index>();
    segmentIdsLocal = segmentIdsLocalShift[platform::GetUbBlockSize() / sizeof(Index)];

    threadNum = MAX_THREAD_NUM > tilingData_->innerDim ? tilingData_->innerDim : MAX_THREAD_NUM;
    threadBlock = MAX_THREAD_NUM / threadNum;

    loopTimes = coreId == (coreNum - 1) ? tilingData_->loopTimesTailCore : tilingData_->loopTimes;
    segIds = coreId == (coreNum - 1) ? tilingData_->segIdsPerLoopTailCore : tilingData_->segIdsPerLoop;
    tailSegIds = coreId == (coreNum - 1) ? tilingData_->segIdsTailLoopTailCore : tilingData_->segIdsTailLoop;
    
    uint64_t segIdsPerCore = tilingData_->outerDim / coreNum;
    uint64_t baseOffset = coreId * segIdsPerCore;
    if (tilingData_->isDeterministic == 1) {
        for (int32_t i = 0; i < loopTimes - 1; i++){
            CopySegmentIdsIn(segmentIdsLocal, baseOffset + i * segIds, segIds);
            ProcessEachLoopForDeterminstic(segIds, baseOffset + i * segIds);
        }
        CopySegmentIdsIn(segmentIdsLocal, baseOffset + (loopTimes - 1) * segIds, tailSegIds);
        ProcessEachLoopForDeterminstic(tailSegIds, baseOffset + (loopTimes - 1) * segIds);
    } else {
        for (int32_t i = 0; i < loopTimes - 1; i++){
            CopySegmentIdsIn(segmentIdsLocal, baseOffset + i * segIds, segIds);
            ProcessEachLoop(segIds, baseOffset + i * segIds);
        }
        CopySegmentIdsIn(segmentIdsLocal, baseOffset + (loopTimes - 1) * segIds, tailSegIds);
        ProcessEachLoop(tailSegIds, baseOffset + (loopTimes - 1) * segIds);
    }
}
}
#endif