/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file uss_deterministic.h
 * \brief uss_deterministic
 */
#ifndef USS_DETERMINISTIC_H_
#define USS_DETERMINISTIC_H_

#include <cmath>
#include <cstdint>

#include "unsorted_segment_base.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace UnsortedSegmentSum {

using namespace AscendC;

constexpr uint32_t MAX_THREAD = 1024;
constexpr uint32_t UB_AGLIN_VALUE = 32;
constexpr uint32_t USED_THREAD_NUMS = 1024;
constexpr uint32_t KB_SIZE = 1024;

static constexpr SortConfig sortConfig = {SortType::RADIX_SORT, false};

template <typename P>
__aicore__ inline P MIN(P x, P y) {
    return x < y ? x : y;
}

template <typename T, typename U>
class KernelUSSDeterministic {
public:
    __aicore__ inline KernelUSSDeterministic(const UnsortedSegmentSumDetermTilingData& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output, GM_ADDR workSpace);
    __aicore__ inline void Process();
    __aicore__ inline void FirstUbProcess(uint64_t curProcessRowsNum, uint64_t segmentIdOffset);
    __aicore__ inline void SecondUbProcess(uint64_t curProcessRowsNum, uint64_t segmentIdOffset);
    __aicore__ inline void ThirdUbProcess();
    __aicore__ inline void CopyInData(uint32_t rowNums, uint64_t segmentIdOffset);

private:
    TPipe& pipe_;
    const UnsortedSegmentSumDetermTilingData& tilingData_;
    TQue<QuePosition::VECIN, 1> xQueue_;
    TQue<QuePosition::VECIN, 1> segmentIdQueue_;
    TBuf<QuePosition::VECCALC> sharedTmpBuf;
    TBuf<QuePosition::VECCALC> sortedIdBuf;
    TBuf<QuePosition::VECCALC> sortedIdIndexBuf;

    GlobalTensor<T> xGm_, yGm_;
    GlobalTensor<U> segmentIds_;
    GlobalTensor<int32_t> workspaceOutput_;
    GlobalTensor<uint64_t> workspaceNValue_;
    GlobalTensor<float> workspaceMValue_;

    uint32_t blockId_;
    uint32_t innerDim_{1};
    uint32_t normalCoreProcessNum_{0};
    uint32_t tailCoreProcessNum_{0};
    uint32_t usedCoreNum_{1};
    uint64_t segmentNum_{1};
    uint32_t tmpBufferSize_{0};
    uint64_t curCoreProcessNum_{0};
    uint64_t rowsNumInUb_{0};
};

template <typename T, typename U>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD_NUMS) inline void SimtAtomicComputeNValueAndMValue(
    uint32_t curProcessRowsNum, uint32_t innerDim_, uint64_t segmentNum, __gm__ float* workspaceMValue_,
    __gm__ uint64_t* workspaceNValue_, __local_mem__ T* xTensor, __local_mem__ U* sortedIdTensor,
    __local_mem__ uint32_t* sortedIdIndexTensor)
{
    for (int32_t j = Simt::GetThreadIdx(); j < innerDim_; j += Simt::GetThreadNum()) {
        // 找第一个有效的 startSegId
        int64_t startSegId = -1;
        for (uint32_t k = 0; k < curProcessRowsNum; k++) {
            int64_t segId = static_cast<int64_t>(sortedIdTensor[k]);
            if (segId >= 0 && segId < static_cast<int64_t>(segmentNum)) {
                startSegId = segId;
                break;
            }
        }

        if (startSegId < 0) {
            continue;
        }

        float maxValue = 0.0f;        
        uint64_t num = 0;

        for (uint32_t i = 0; i < curProcessRowsNum; i++) { 
            int64_t curSegId = static_cast<int64_t>(sortedIdTensor[i]);

            // 跳过无效的 segmentId
            if (curSegId < 0 || curSegId >= static_cast<int64_t>(segmentNum)) {
                if (i == curProcessRowsNum - 1 && num > 0) {
                    if (j == 0) {
                        Simt::AtomicAdd(workspaceNValue_ + static_cast<uint64_t>(startSegId), num);
                    }
                    Simt::AtomicMax(workspaceMValue_ + static_cast<uint64_t>(startSegId) * innerDim_ + j, maxValue);
                }
                continue;
            }

            int32_t tmpXIndex = sortedIdIndexTensor[i] * innerDim_ + j;
            float tmpValue = Simt::Abs(static_cast<float>(xTensor[tmpXIndex]));

            if (curSegId == startSegId) {
                num++;
                maxValue = maxValue > tmpValue ? maxValue : tmpValue;  
            } else {
                if (j == 0) {
                    Simt::AtomicAdd(workspaceNValue_ + static_cast<uint64_t>(startSegId), num);
                }

                Simt::AtomicMax(workspaceMValue_ + static_cast<uint64_t>(startSegId) * innerDim_ + j, maxValue);
                startSegId = curSegId;
                num = 1;
                maxValue = tmpValue;
            }

            if (i == curProcessRowsNum - 1) {
                if (j == 0) {
                    Simt::AtomicAdd(workspaceNValue_ + static_cast<uint64_t>(startSegId), num);
                }
                Simt::AtomicMax(workspaceMValue_ + static_cast<uint64_t>(startSegId) * innerDim_ + j, maxValue);
                break;
            }
        }
    }	
}

template <typename T, typename U>
__aicore__ inline void KernelUSSDeterministic<T, U>::Init(GM_ADDR x, GM_ADDR segmentIds, GM_ADDR output, GM_ADDR workSpace)
{
    blockId_ = GetBlockIdx();
    innerDim_ = tilingData_.innerDim; 
    normalCoreProcessNum_ = tilingData_.normalCoreProcessNum;
    tailCoreProcessNum_ = tilingData_.tailCoreProcessNum;
    usedCoreNum_ = tilingData_.usedCoreNum;
    tmpBufferSize_ = tilingData_.tmpBufferSize;
    rowsNumInUb_ = tilingData_.rowsNumInUB;
    segmentNum_ = tilingData_.outputOuterDim;      

    workspaceNValue_.SetGlobalBuffer((__gm__ uint64_t*)workSpace);
    workspaceMValue_.SetGlobalBuffer((__gm__ float*)workSpace + segmentNum_ * 2 + KB_SIZE / sizeof(float));
    workspaceOutput_.SetGlobalBuffer((__gm__ int32_t*)workSpace + segmentNum_ * 2 + segmentNum_ * innerDim_ + KB_SIZE * 2 /sizeof(int32_t));
    
    xGm_.SetGlobalBuffer((__gm__ T*)x + blockId_ * normalCoreProcessNum_ * innerDim_);
    yGm_.SetGlobalBuffer((__gm__ T*)output);
    segmentIds_.SetGlobalBuffer((__gm__ U*)segmentIds + blockId_ * normalCoreProcessNum_);
    
    if (blockId_ == 0) {
        InitGlobalMemory(workspaceNValue_, segmentNum_, (uint64_t)(0));
        InitGlobalMemory(workspaceMValue_, segmentNum_ * innerDim_, (float)(0));
        InitGlobalMemory(yGm_, segmentNum_ * innerDim_, (T)0);
        InitGlobalMemory(workspaceOutput_, segmentNum_ * innerDim_, (int32_t)0);
    }

    PipeBarrier<PIPE_ALL>();
    SyncAll();
    pipe_.Reset();

    curCoreProcessNum_ = blockId_ == (usedCoreNum_ - 1) ? tailCoreProcessNum_ : normalCoreProcessNum_;   
    uint32_t xQueueSize = Ops::Base::CeilAlign(static_cast<uint32_t>(rowsNumInUb_* innerDim_ * sizeof(T)), UB_AGLIN_VALUE);
    uint32_t segmentIdQueueSize = Ops::Base::CeilAlign(static_cast<uint32_t>(rowsNumInUb_* sizeof(U)), UB_AGLIN_VALUE);
    uint32_t sortedIdIndexBufSize = Ops::Base::CeilAlign(static_cast<uint32_t>(rowsNumInUb_* sizeof(uint32_t)), UB_AGLIN_VALUE);

    pipe_.InitBuffer(xQueue_, 1, xQueueSize);        
    pipe_.InitBuffer(segmentIdQueue_, 1, segmentIdQueueSize);        
    pipe_.InitBuffer(sharedTmpBuf, tmpBufferSize_); 
    pipe_.InitBuffer(sortedIdBuf, segmentIdQueueSize); 
    pipe_.InitBuffer(sortedIdIndexBuf, sortedIdIndexBufSize); 
}

template <typename T, typename U>
__aicore__ inline void KernelUSSDeterministic<T, U>::Process()
{
    if (blockId_ >= tilingData_.usedCoreNum) {
        return;
    }
    uint32_t loopCnt = Ops::Base::CeilDiv(curCoreProcessNum_, rowsNumInUb_);

    for (uint32_t loop = 0; loop < loopCnt; ++loop) {
        uint64_t curProcessRowsNum = loop == loopCnt - 1 ? curCoreProcessNum_ - (loopCnt - 1) * rowsNumInUb_ : rowsNumInUb_;
        uint64_t segmentIdOffset = loop * rowsNumInUb_;
        FirstUbProcess(curProcessRowsNum, segmentIdOffset);
    }
    
    PipeBarrier<PIPE_ALL>();
    SyncAll();
    for (uint32_t loop = 0; loop < loopCnt; ++loop) {
        uint64_t curProcessRowsNum = loop == loopCnt - 1 ? curCoreProcessNum_ - (loopCnt - 1) * rowsNumInUb_ : rowsNumInUb_;
        uint64_t segmentIdOffset = loop * rowsNumInUb_;
        SecondUbProcess(curProcessRowsNum, segmentIdOffset);  
    }

    PipeBarrier<PIPE_ALL>();
    SyncAll();
    ThirdUbProcess();
    PipeBarrier<PIPE_ALL>();
    SyncAll();
}

template <typename T, typename U>
__aicore__ inline void KernelUSSDeterministic<T, U>::CopyInData(uint32_t rowNums, uint64_t segmentIdOffset)
{
    // copy in x
    LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();
    uint64_t inputOffset = segmentIdOffset * innerDim_;
    DataCopyExtParams dataCopyParam{(uint16_t)1, (uint32_t)(rowNums * innerDim_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> dataPadParam{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm_[inputOffset], dataCopyParam, dataPadParam);
    xQueue_.EnQue(xLocal);

    // copy in id
    LocalTensor<U> segmentIdLocal = segmentIdQueue_.AllocTensor<U>();
    DataCopyExtParams idCopyParam{(uint16_t)1, (uint32_t)(rowNums * sizeof(U)), 0, 0, 0};
    DataCopyPadExtParams<U> idPadParam{false, 0, 0, 0};
    DataCopyPad(segmentIdLocal, segmentIds_[segmentIdOffset], idCopyParam, idPadParam);
    segmentIdQueue_.EnQue(segmentIdLocal);
}

template <typename T, typename U>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD_NUMS) inline void QuantizeAndCopyOut(
    __gm__ float* mValueWorkSpace, __gm__ uint64_t* nValueWorkSpace, __gm__ int32_t* workspaceOutput32,
    __local_mem__ T* xTensor, __local_mem__ U* sortedIdTensor, __local_mem__ uint32_t* sortedIdIndexTensor,
    uint32_t curProcessRowsNum, uint32_t innerDim, uint64_t segmentNum)
{
    float scaling = static_cast<float>(1L << 30);

    for (uint32_t i = Simt::GetThreadIdx(); i < innerDim; i += Simt::GetThreadNum()) {
        // 找第一个有效的 startSegId
        int64_t startSegId = -1;
        for (uint32_t k = 0; k < curProcessRowsNum; k++) {
            int64_t segId = static_cast<int64_t>(sortedIdTensor[k]);
            if (segId >= 0 && segId < static_cast<int64_t>(segmentNum)) {
                startSegId = segId;
                break;
            }
        }

        if (startSegId < 0) {
            continue;
        }

        float sumValue = 0.0f;

        for (uint32_t j = 0; j < curProcessRowsNum; j++) {    
            int64_t curSegId = static_cast<int64_t>(sortedIdTensor[j]);

            // 跳过无效的 segmentId
            if (curSegId < 0 || curSegId >= static_cast<int64_t>(segmentNum)) {
                if (j == curProcessRowsNum - 1 && sumValue != 0.0f) {
                    uint32_t tmpOffset = static_cast<uint64_t>(startSegId) * innerDim + i;
                    float mValueFloat = static_cast<float>(mValueWorkSpace[tmpOffset]);
                    float nValueFloat = static_cast<float>(nValueWorkSpace[static_cast<uint64_t>(startSegId)]);
                    sumValue = sumValue / (mValueFloat * nValueFloat) * scaling;
                    int32_t quantizedResInt =
                        AscendC::Simt::Cast<int32_t, float, RoundMode::CAST_RINT, AscendC::Simt::SatMode::SAT>(
                            sumValue);
                    Simt::AtomicAdd(workspaceOutput32 + tmpOffset, quantizedResInt);
                }
                continue;
            }

            uint32_t tmpXIndex = sortedIdIndexTensor[j] * innerDim + i;
            float tmpValue = static_cast<float>(xTensor[tmpXIndex]);

            if (curSegId == startSegId) {
                sumValue += tmpValue;
            } else {
                uint32_t tmpOffset = static_cast<uint64_t>(startSegId) * innerDim + i;
                float mValueFloat = static_cast<float>(mValueWorkSpace[tmpOffset]);
                float nValueFloat = static_cast<float>(nValueWorkSpace[static_cast<uint64_t>(startSegId)]);

                sumValue = sumValue / (mValueFloat * nValueFloat) * scaling;
                int32_t quantizedResInt = AscendC::Simt::Cast<int32_t, float, RoundMode::CAST_RINT, AscendC::Simt::SatMode::SAT>(sumValue);
                Simt::AtomicAdd(workspaceOutput32 + tmpOffset, quantizedResInt);

                startSegId = curSegId;
                sumValue = tmpValue;
            }

            if (j == curProcessRowsNum - 1) {
                uint32_t tmpOffset = static_cast<uint64_t>(startSegId) * innerDim + i;
                float mValueFloat = static_cast<float>(mValueWorkSpace[tmpOffset]);
                float nValueFloat = static_cast<float>(nValueWorkSpace[static_cast<uint64_t>(startSegId)]);
                sumValue = sumValue / (mValueFloat * nValueFloat) * scaling;
                int32_t quantizedResInt = AscendC::Simt::Cast<int32_t, float, RoundMode::CAST_RINT, AscendC::Simt::SatMode::SAT>(sumValue);
                Simt::AtomicAdd(workspaceOutput32 + tmpOffset, quantizedResInt);
                break;
            }
        }
    }
}

template <typename T, typename U>
__aicore__ inline void KernelUSSDeterministic<T, U>::FirstUbProcess(uint64_t curProcessRowsNum, uint64_t segmentIdOffset)
{
    CopyInData(curProcessRowsNum, segmentIdOffset);
    LocalTensor<T> xLocal = xQueue_.DeQue<T>();
    LocalTensor<U> segmentIdTensor = segmentIdQueue_.DeQue<U>();
    LocalTensor<U> sortedIdTensor = sortedIdBuf.Get<U>();
    LocalTensor<uint32_t> sortedIdIndexTensor = sortedIdIndexBuf.Get<uint32_t>();

    LocalTensor<uint8_t> sharedTmpTensor = sharedTmpBuf.Get<uint8_t>();
    AscendC::Sort<U, false, sortConfig>(sortedIdTensor, sortedIdIndexTensor, segmentIdTensor, sharedTmpTensor,
                                        static_cast<uint32_t>(curProcessRowsNum));

    AscendC::Simt::VF_CALL<SimtAtomicComputeNValueAndMValue<T, U>>(
        Simt::Dim3(MAX_THREAD), curProcessRowsNum, innerDim_, segmentNum_,
        (__gm__ float*)(workspaceMValue_.GetPhyAddr()), (__gm__ uint64_t*)(workspaceNValue_.GetPhyAddr()),
        (__local_mem__ T*)(xLocal.GetPhyAddr()), (__local_mem__ U*)(sortedIdTensor.GetPhyAddr()),
        (__local_mem__ uint32_t*)(sortedIdIndexTensor.GetPhyAddr()));

    xQueue_.FreeTensor(xLocal);
    segmentIdQueue_.FreeTensor(segmentIdTensor);
}

template <typename T, typename U>
__aicore__ inline void KernelUSSDeterministic<T, U>::SecondUbProcess(uint64_t curProcessRowsNum, uint64_t segmentIdOffset)
{
    CopyInData(curProcessRowsNum, segmentIdOffset);
    LocalTensor<T> xLocal = xQueue_.DeQue<T>();
    LocalTensor<U> segmentIdTensor = segmentIdQueue_.DeQue<U>();
    LocalTensor<U> sortedIdTensor = sortedIdBuf.Get<U>();
    LocalTensor<uint32_t> sortedIdIndexTensor = sortedIdIndexBuf.Get<uint32_t>();

    LocalTensor<uint8_t> sharedTmpTensor = sharedTmpBuf.Get<uint8_t>();
    AscendC::Sort<U, false, sortConfig>(sortedIdTensor, sortedIdIndexTensor, segmentIdTensor, sharedTmpTensor,
                                        static_cast<uint32_t>(curProcessRowsNum));

    AscendC::Simt::VF_CALL<QuantizeAndCopyOut<T, U>>(
        Simt::Dim3(MAX_THREAD), (__gm__ float*)(workspaceMValue_.GetPhyAddr()),
        (__gm__ uint64_t*)(workspaceNValue_.GetPhyAddr()), (__gm__ int32_t*)(workspaceOutput_.GetPhyAddr()),
        (__local_mem__ T*)(xLocal.GetPhyAddr()), (__local_mem__ U*)(sortedIdTensor.GetPhyAddr()),
        (__local_mem__ uint32_t*)(sortedIdIndexTensor.GetPhyAddr()), curProcessRowsNum, innerDim_, segmentNum_);

    xQueue_.FreeTensor(xLocal);
    segmentIdQueue_.FreeTensor(segmentIdTensor);
}

template <typename T, typename U>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD_NUMS) inline void Dequantize(
    __gm__ float* mValueWorkSpace, __gm__ uint64_t* nValueWorkSpace, __gm__ int32_t* workspaceOutput, __gm__ T* yGm, 
    uint32_t segmentNum, uint32_t innerDim, uint32_t blockId, uint32_t blockNum)
{
    float scaling = static_cast<float>(1L << 30);
    
    for (uint32_t i = blockId * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < segmentNum * innerDim; 
        i += blockNum * Simt::GetThreadNum()) {  
        uint32_t row = i / innerDim;
        uint32_t col = i % innerDim;

        uint32_t tmpOffset = row * innerDim + col;
        float mValueFloat = static_cast<float>(mValueWorkSpace[tmpOffset]);
        float nValueFloat = static_cast<float>(nValueWorkSpace[row]);
        float quantizedResFloat = AscendC::Simt::Cast<float, int32_t, RoundMode::CAST_RINT, AscendC::Simt::SatMode::SAT>(workspaceOutput[tmpOffset]); 
        float result = quantizedResFloat * mValueFloat * nValueFloat / scaling;
        
        yGm[tmpOffset] = static_cast<T>(result);
    }
}

template <typename T, typename U>
__aicore__ inline void KernelUSSDeterministic<T, U>::ThirdUbProcess()
{
    Simt::VF_CALL<Dequantize<T,U>>(Simt::Dim3(MAX_THREAD), (__gm__ float*)(workspaceMValue_.GetPhyAddr()), 
        (__gm__ uint64_t*)(workspaceNValue_.GetPhyAddr()), (__gm__ int32_t*)(workspaceOutput_.GetPhyAddr()),
        (__gm__ T*)(yGm_.GetPhyAddr()), segmentNum_, innerDim_, blockId_, usedCoreNum_);
}

} // namespace UnsortedSegmentSum
#endif