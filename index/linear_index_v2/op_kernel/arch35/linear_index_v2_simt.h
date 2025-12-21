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
 * \file linear_index_v2_simt.h
 * \brief
 */
#ifndef LINEAR_INDEX_V2_SIMT_H
#define LINEAR_INDEX_V2_SIMT_H
#include "kernel_operator.h"
#include "linear_index_v2_common.h"
using namespace AscendC;

constexpr int64_t THREAD_DIM = 1024;
constexpr int64_t UB_THRES_INT64 = 27648; // 216K / 8Byte
constexpr int64_t UB_THRES_INT32 = 55296; // 216K / 4Byte
constexpr uint64_t VALID_INDICES = 2;
constexpr uint64_t INVALID_INDICES = 1;

template <typename T>
__simt_vf__ LAUNCH_BOUND(THREAD_DIM)
__aicore__ inline void SimtLinearIndexProcess(
    int tensorId, __gm__ int* strideAddr, __gm__ int* valueSizeAddr, __gm__ T* indexAddr, __ubuf__ T* outputUbAddr,
    uint64_t calcNum)
{
    for (int64_t i = Simt::GetThreadIdx(); i < calcNum; i = i + Simt::GetThreadNum()) {
        T indexValue = indexAddr[i];
        int stride = strideAddr[tensorId];
        int valueSize = valueSizeAddr[tensorId];
        outputUbAddr[i] = outputUbAddr[i] + static_cast<T>(((indexValue % valueSize + valueSize) % valueSize) * stride);
    }
}

template <typename T>
class LinearIndexSimtKernelV2
{
public:
    __aicore__ inline LinearIndexSimtKernelV2() = delete;
    __aicore__ inline LinearIndexSimtKernelV2(
        GM_ADDR indexList, GM_ADDR stride, GM_ADDR valueSize, GM_ADDR output, GM_ADDR workSpace,
        const LinearIndexV2TilingData& tiling, TPipe& pipe)
    {
        InitParams(tiling, indexList);
        SetGmAddr(stride, valueSize, output, tiling);
        InitBuffers(pipe);
    }

    __aicore__ inline void Process()
    {
        for (int i = 0; i < loopTime_; i++) {
            ProcessMain(i, ubThres_);
        }
        if (loopTail_ != 0) {
            ProcessMain(loopTime_, loopTail_);
        }
    }

private:
    __aicore__ inline void ProcessMain(int loopIdx, uint64_t calcNum)
    {
        __gm__ int* strideAddr = (__gm__ int*)strideGm_.GetPhyAddr();
        __gm__ int* valueSizeAddr = (__gm__ int*)valueSizeGm_.GetPhyAddr();
        LocalTensor<T> outputUbBuf = outputUb_.Get<T>();
        Duplicate(outputUbBuf, static_cast<T>(0), ubThres_);
        PipeBarrier<PIPE_V>();
        __ubuf__ T* outputUbAddr = (__ubuf__ T*)outputUbBuf.GetPhyAddr();

        for (int i = 0; i < tensorId_; i++) {
            // if indices is invalid, continue
            if (indicesMask_[i] == 0) {
                continue;
            }
            indexGm_.SetGlobalBuffer(GetTensorAddr(indexList_, i) + idxAddrOffset_ + loopIdx * ubThres_);
            __gm__ T* indexAddr = (__gm__ T*)indexGm_.GetPhyAddr();
            Simt::VF_CALL<SimtLinearIndexProcess<T>>(
                Simt::Dim3(THREAD_DIM), i, strideAddr, valueSizeAddr, indexAddr, outputUbAddr, calcNum);
        }
        auto mte3WaitVEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(mte3WaitVEventID);
        WaitFlag<HardEvent::V_MTE3>(mte3WaitVEventID);
        DataCopyExtParams idxCopyParams{1, static_cast<uint32_t>(calcNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGm_[loopIdx * ubThres_], outputUbBuf, idxCopyParams);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void InitParams(const LinearIndexV2TilingData& tiling, GM_ADDR indexList)
    {
        indexList_ = indexList;
        blockIdx_ = GetBlockIdx();
        formerCoreNum_ = tiling.params.formerCoreNum;
        tensorId_ = tiling.params.tensorId;
        indicesMask_ = tiling.params.indicesMask;
        if (blockIdx_ < formerCoreNum_) {
            dataNum_ = tiling.params.formerCoreDataNum;
            idxAddrOffset_ = tiling.params.formerCoreDataNum * blockIdx_;
        } else {
            dataNum_ = tiling.params.tailCoreDataNum;
            idxAddrOffset_ = tiling.params.formerCoreDataNum * formerCoreNum_ +
                             tiling.params.tailCoreDataNum * (blockIdx_ - formerCoreNum_);
        }
        ubThres_ = IS_CAST_INT ? UB_THRES_INT64 : UB_THRES_INT32;
        loopTime_ = dataNum_ / ubThres_;
        loopTail_ = dataNum_ % ubThres_;
    }

    __aicore__ inline void InitBuffers(TPipe& pipe)
    {
        pipe.InitBuffer(outputUb_, ubThres_ * sizeof(T));
    }

    __aicore__ inline void SetGmAddr(
        GM_ADDR stride, GM_ADDR valueSize, GM_ADDR output, const LinearIndexV2TilingData& tiling)
    {
        valueSizeGm_.SetGlobalBuffer((__gm__ int*)valueSize);
        strideGm_.SetGlobalBuffer((__gm__ int*)stride);
        outputGm_.SetGlobalBuffer((__gm__ T*)output + idxAddrOffset_);
        InitGlobalMemory(outputGm_, dataNum_, static_cast<T>(0));
        SyncAll();
    }

    __aicore__ inline __gm__ T* GetTensorAddr(GM_ADDR indexListPtr, const int offset)
    {
        // 获取tensorList中tensor的首地址
        __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(indexListPtr);
        uint64_t tensorPtrOffset = *dataAddr;
        __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
        return reinterpret_cast<__gm__ T*>(*(tensorPtr + offset));
    }

private:
    GM_ADDR indexList_;
    GlobalTensor<T> indexGm_;
    GlobalTensor<int> valueSizeGm_;
    GlobalTensor<int> strideGm_;
    GlobalTensor<T> outputGm_;
    TBuf<TPosition::VECCALC> outputUb_;

    uint64_t idxAddrOffset_;
    uint64_t blockIdx_;
    uint64_t tensorId_;
    uint64_t dataNum_;
    uint64_t formerCoreNum_;
    uint64_t formerDataNum_;
    uint64_t tailDataNum_;
    uint64_t formerTime_;
    uint64_t tailTime_;
    uint64_t loopTime_;
    uint64_t loopTail_;
    uint64_t ubThres_;
    const uint64_t* indicesMask_;
};

#endif // LINEAR_INDEX_V2_SIMT_H