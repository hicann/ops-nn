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
 * \file fused_linear_cross_entropy_loss_grad_mem_friendly.h
 * \brief
 */
#ifndef FUSED_LINEAR_CROSS_ENTROPY_LOSS_GRAD_MEM_FRIENDLY_H
#define FUSED_LINEAR_CROSS_ENTROPY_LOSS_GRAD_MEM_FRIENDLY_H

#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "fused_linear_cross_entropy_loss_grad_config.h"
#ifdef __CCE_KT_TEST__
#include "fused_linear_cross_entropy_loss_grad_tiling_data_def.h"
#endif  // __CCE_KT_TEST__

namespace FusedLinearCrossEntropyLossGradNS {
using namespace AscendC;

class SlidingWindowBlockIterator {
private:
    // 输入参数
    uint32_t BT_, V_;
    uint32_t baseBT_, baseV_;
    uint32_t baseBTPerWindow_, baseVPerWindow_;
    
    // 计算得到的参数
    uint32_t totalRowBlocks_, totalColBlocks_;
    
    // 迭代状态
    uint32_t windowRowStart_, windowColStart_;
    uint32_t colHead_, offset_;
    uint32_t currentWindowRowBlocks_, currentWindowColBlocks_;
    uint32_t currentTaskCnt_;
    bool finished_;

    // 缓存相关成员
    uint32_t cachedRowIdx_[30];  // 固定大小的缓存数组
    uint32_t cachedColIdx_[30];  // 固定大小的缓存数组
    uint32_t cacheSize_;         // 实际缓存大小
    uint32_t cacheStartIdx_;     // 当前缓存块的起始索引
    uint32_t cacheEndIdx_;       // 当前缓存块的结束索引+1

public:
    __aicore__ inline SlidingWindowBlockIterator() 
        : cacheSize_(0), cacheStartIdx_(0), cacheEndIdx_(0) {};
    
    __aicore__ inline  void init(
        uint32_t BT, uint32_t V, 
        uint32_t baseBT, uint32_t baseV,
        uint32_t baseBTPerWindow, uint32_t baseVPerWindow)
    {
        BT_ = BT;
        V_ = V;
        baseBT_ = Std::max(baseBT, 1U);
        baseV_ = Std::max(baseV, 1U);
        baseBTPerWindow_ = baseBTPerWindow;
        baseVPerWindow_ = baseVPerWindow;
        finished_ = false;
        // 计算总块数
        totalRowBlocks_ = (BT_ + baseBT_ - 1U) / baseBT_;
        totalColBlocks_ = (V_ + baseV_ - 1U) / baseV_;
        // 初始化迭代状态
        reset();
    }

    // 新增：设置缓存大小
    __aicore__ inline void setCacheSize(uint32_t size) {
        // 确保缓存大小不超过30
        cacheSize_ = Std::min(size, 30U);
        cacheStartIdx_ = 0;
        cacheEndIdx_ = 0;
    }

    // 获取指定索引的位置
    __aicore__ inline bool getAtIndex(uint32_t index, uint32_t& rowIdx, uint32_t& colIdx) {
        // 如果索引在当前缓存范围内，直接返回
        if (index >= cacheStartIdx_ && index < cacheEndIdx_) {
            uint32_t cachePos = index - cacheStartIdx_;
            rowIdx = cachedRowIdx_[cachePos];
            colIdx = cachedColIdx_[cachePos];
            return true;
        }
        
        // 计算新的缓存块起始位置
        uint32_t newCacheStart = (index / cacheSize_) * cacheSize_;
        
        // 填充新的缓存块
        fillCacheBlock(newCacheStart);
        
        // 再次检查索引是否在缓存范围内
        if (index >= cacheStartIdx_ && index < cacheEndIdx_) {
            uint32_t cachePos = index - cacheStartIdx_;
            rowIdx = cachedRowIdx_[cachePos];
            colIdx = cachedColIdx_[cachePos];
            return true;
        }
        
        // 索引超出范围
        return false;
    }

    // 填充缓存块
    __aicore__ inline void fillCacheBlock(uint32_t startIndex) {
        if (cacheSize_ == 0) {
            return; // 没有设置缓存大小
        }
        
        // 保存当前状态
        uint32_t savedWindowRowStart = windowRowStart_;
        uint32_t savedWindowColStart = windowColStart_;
        uint32_t savedColHead = colHead_;
        uint32_t savedOffset = offset_;
        uint32_t savedCurrentTaskCnt = currentTaskCnt_;
        bool savedFinished = finished_;
        
        // 重置到起始位置
        resetToIndex(startIndex);
        
        // 填充缓存
        uint32_t count = 0;
        uint32_t rowIdx, colIdx;
        
        while (count < cacheSize_ && next(rowIdx, colIdx)) {
            cachedRowIdx_[count] = rowIdx;
            cachedColIdx_[count] = colIdx;
            count++;
        }
        
        // 更新缓存范围
        cacheStartIdx_ = startIndex;
        cacheEndIdx_ = startIndex + count;
        
        // 恢复原始状态
        windowRowStart_ = savedWindowRowStart;
        windowColStart_ = savedWindowColStart;
        colHead_ = savedColHead;
        offset_ = savedOffset;
        currentTaskCnt_ = savedCurrentTaskCnt;
        finished_ = savedFinished;
        updateWindowParameters();
    }

    // 重置到指定索引
    __aicore__ inline void resetToIndex(uint32_t index) {
        // 完全重置迭代器
        reset();
        
        // 如果index为0，直接返回
        if (index == 0) {
            return;
        }
        
        // 迭代到指定索引
        uint32_t rowIdx, colIdx;
        for (uint32_t i = 0; i < index; i++) {
            if (!next(rowIdx, colIdx)) {
                break;
            }
        }
    }

    // 重置迭代器（原有代码）
    __aicore__ inline void reset() {
        windowRowStart_ = 0;
        windowColStart_ = 0;
        colHead_ = 0;
        offset_ = 0;
        currentTaskCnt_ = 0;
        finished_ = false;
        // 计算当前窗口参数
        updateWindowParameters();
        
        // 清空缓存范围
        cacheStartIdx_ = 0;
        cacheEndIdx_ = 0;
    }

    // 获取总块数信息（原有代码）
    __aicore__ inline void getBlockCounts(uint32_t& baseBTCnt, uint32_t& baseVCnt, uint32_t& baseTaskCnt) const {
        baseBTCnt = totalRowBlocks_;
        baseVCnt = totalColBlocks_;
        baseTaskCnt = calculateTotalTasks();
    }

    // 获取下一个分块位置（原有代码）
    __aicore__ inline bool next(uint32_t& rowIdx, uint32_t& colIdx) {
        if (finished_) {
            return false;
        }
        // 计算当前分块位置
        rowIdx = offset_ + windowRowStart_;
        colIdx = (colHead_ + offset_) % currentWindowColBlocks_ + windowColStart_;
        // 移动到下一个位置
        if (!moveToNextPosition()) {
            finished_ = true;
        }
        
        return true;
    }

    // 判断是否还有更多分块（原有代码）
    __aicore__ inline bool hasNext() const {
        return !finished_;
    }

    // 获取当前分块计数（原有代码）
    __aicore__ inline uint32_t getCurrentTaskCount() const {
        return currentTaskCnt_;
    }

private:
    // 计算总任务数（原有代码）
    __aicore__ inline uint32_t calculateTotalTasks() const {
        uint32_t totalTasks = 0;
        
        for (uint32_t wrStart = 0; wrStart < totalRowBlocks_; wrStart += baseBTPerWindow_) {
            uint32_t currentWRBlocks = Std::min(baseBTPerWindow_, totalRowBlocks_ - wrStart);
            
            for (uint32_t wcStart = 0; wcStart < totalColBlocks_; wcStart += baseVPerWindow_) {
                uint32_t currentWCBlocks = Std::min(baseVPerWindow_, totalColBlocks_ - wcStart);
                
                for (uint32_t ch = 0; ch < currentWCBlocks; ch++) {
                    for (uint32_t off = 0; off < currentWRBlocks; off++) {
                        totalTasks++;
                    }
                }
            }
        }
        
        return totalTasks;
    }

    // 更新当前窗口参数（原有代码）
    __aicore__ inline void updateWindowParameters() {
        currentWindowRowBlocks_ = Std::min(baseBTPerWindow_, totalRowBlocks_ - windowRowStart_);
        currentWindowColBlocks_ = Std::min(baseVPerWindow_, totalColBlocks_ - windowColStart_);
    }

    // 移动到下一个位置（原有代码）
    __aicore__ inline bool moveToNextPosition() {
        currentTaskCnt_++;
        // 在当前窗口内移动
        offset_++;
        if (offset_ < currentWindowRowBlocks_) {
            return true;
        }
        // 移动到下一列头
        offset_ = 0;
        colHead_++;
        if (colHead_ < currentWindowColBlocks_) {
            return true;
        }
        // 移动到下一个窗口列
        colHead_ = 0;
        windowColStart_ += baseVPerWindow_;
        if (windowColStart_ < totalColBlocks_) {
            updateWindowParameters();
            return true;
        }
        // 移动到下一个窗口行
        windowColStart_ = 0;
        windowRowStart_ += baseBTPerWindow_;
        if (windowRowStart_ < totalRowBlocks_) {
            updateWindowParameters();
            return true;
        }
        // 所有分块都已处理完毕
        return false;
    }
};

template <typename HalfType, typename TargetMaskType, typename MaskedTargetType, typename MM0, typename MM1, typename MM2>
class FusedLinearCrossEntropyLossGradMemFriendly {
public:

__aicore__ inline FusedLinearCrossEntropyLossGradMemFriendly() {}
__aicore__ inline void Init(
    GM_ADDR grad, GM_ADDR input, GM_ADDR weight, GM_ADDR target_mask, GM_ADDR masked_target,
    GM_ADDR logits_max, GM_ADDR sum_exp_logits, GM_ADDR input_grad_out, GM_ADDR weight_grad_out,
    GM_ADDR workspace, const __gm__ FusedLinearCrossEntropyLossGradMemFriendlyTilingData *tiling_data_ptr)
{
    tdp = tiling_data_ptr;
    if ASCEND_IS_AIV {
        aicBlockIdx = blockIdx / 2;
        pipe.Init();
    } else {
        aicBlockIdx = blockIdx;
        COPY_TILING_WITH_STRUCT(tiling::TCubeTiling, &tdp->mm0Tiling, mm0Tiling);
        COPY_TILING_WITH_STRUCT(tiling::TCubeTiling, &tdp->mm1Tiling, mm1Tiling);
        COPY_TILING_WITH_STRUCT(tiling::TCubeTiling, &tdp->mm2Tiling, mm2Tiling);
        mm0.Init(mm0Tiling, &pipe);
        mm1.Init(mm1Tiling, &pipe);
        mm2.Init(mm2Tiling, &pipe);
    }

    uint64_t allocatedWSSize = 0;  // 以float计
    // 只取当前核心需要的ws
    mm0ResultWS[0].SetGlobalBuffer((__gm__ float *)workspace + aicBlockIdx * tdp->mm0ResultSize, tdp->mm0ResultSize);
    mm0ResultWS[1].SetGlobalBuffer((__gm__ float *)workspace + tdp->mm0ResultSizePerEpoch + aicBlockIdx * tdp->mm0ResultSize, tdp->mm0ResultSize);
    mm0ResultHalfWS[0].SetGlobalBuffer((__gm__ HalfType *)mm0ResultWS[0].GetPhyAddr(), tdp->mm0ResultSize);
    mm0ResultHalfWS[1].SetGlobalBuffer((__gm__ HalfType *)mm0ResultWS[1].GetPhyAddr(), tdp->mm0ResultSize);
    allocatedWSSize += tdp->mm0ResultSizePerEpoch * SIZE_2;
    // 分核取全部，以便V2使用
    for (uint8_t wsIdx = 0; wsIdx < SIZE_2; wsIdx++) {
        for (uint32_t idx = 0; idx < tdp->aicNum; idx++) {
            mm1ResultWS[wsIdx][idx].SetGlobalBuffer((__gm__ float *)workspace + allocatedWSSize + idx * tdp->mm1ResultSize, tdp->mm1ResultSize);
        }
        allocatedWSSize += tdp->mm1ResultSizePerEpoch;
    }
    for (uint8_t wsIdx = 0; wsIdx < SIZE_2; wsIdx++) {
        for (uint32_t idx = 0; idx < tdp->aicNum; idx++) {
            mm2ResultWS[wsIdx][idx].SetGlobalBuffer((__gm__ float *)workspace + allocatedWSSize + idx * tdp->mm2ResultSize, tdp->mm2ResultSize);
        }
        allocatedWSSize += tdp->mm2ResultSizePerEpoch;
    }

    if ASCEND_IS_AIV {
        logitsMaxGM.SetGlobalBuffer((__gm__ float *)logits_max, tdp->BT);
        sumExpLogitsGM.SetGlobalBuffer((__gm__ float *)sum_exp_logits, tdp->BT);
        targetMaskGM.SetGlobalBuffer((__gm__ TargetMaskType *)target_mask, tdp->targetMaskSize);
        maskedTargetGM.SetGlobalBuffer((__gm__ MaskedTargetType *)masked_target, tdp->BT);
        gradGM.SetGlobalBuffer((__gm__ float *)grad, tdp->BT);
        inputGradOutGM.SetGlobalBuffer((__gm__ HalfType *)input_grad_out, (uint64_t)tdp->BT * tdp->H);
        weightGradOutGM.SetGlobalBuffer((__gm__ HalfType *)weight_grad_out, (uint64_t)tdp->V * tdp->H);
        mm1AccumWS.SetGlobalBuffer((__gm__ float *)workspace + allocatedWSSize, (uint64_t)tdp->BT * tdp->H);
        allocatedWSSize += (uint64_t)tdp->BT * tdp->H;
        mm2AccumWS.SetGlobalBuffer((__gm__ float *)workspace + allocatedWSSize, (uint64_t)tdp->V * tdp->H);
        allocatedWSSize += (uint64_t)tdp->V * tdp->H;
        if (blockIdx == 0) {
            InitGlobalMemory(mm1AccumWS, (uint64_t)tdp->BT * tdp->H, 0.0f);
        }
        if (blockIdx == 1) {
            InitGlobalMemory(mm2AccumWS, (uint64_t)tdp->V * tdp->H, 0.0f);
        }
        // UB
        pipe.InitBufPool(V1Pool, tdp->poolByteSize);
        pipe.InitBufPool(V2Pool, tdp->poolByteSize, V1Pool);
        pipe.InitBufPool(castMainPool, tdp->poolByteSize, V1Pool);
        V1Pool.InitBuffer(V1MainQ, 1, tdp->V1MainSize * IN_BYTE_SIZE);
        V1Pool.InitBuffer(divisorBuf, tdp->baseV * IN_BYTE_SIZE);
        V2Pool.InitBuffer(V2TargetQ, SIZE_2, tdp->V2MainSize * IN_BYTE_SIZE);
        V2Pool.InitBuffer(V2AdditionQ, SIZE_2, tdp->V2MainSize * IN_BYTE_SIZE);
        castMainPool.InitBuffer(castMainQ, SIZE_2, tdp->castMainSize * IN_BYTE_SIZE);
        divisorUB = divisorBuf.Get<float>();
    } else {
        inputGM.SetGlobalBuffer((__gm__ HalfType *)input, (uint64_t)tdp->BT * tdp->H);
        weightGM.SetGlobalBuffer((__gm__ HalfType *)weight, (uint64_t)tdp->V * tdp->H);
    }
    // 对角线分块准备
    iterator1.init(tdp->BT, tdp->V, tdp->baseBT, tdp->baseV, SIZE_8, SIZE_8);
    iterator1.setCacheSize(tdp->aicNum);
    iterator2.init(tdp->BT, tdp->V, tdp->baseBT, tdp->baseV, SIZE_8, SIZE_8);
    iterator2.setCacheSize(tdp->aicNum);
    iterator1.getBlockCounts(baseBTCnt, baseVCnt, baseTaskCnt);
    baseTaskEpochCnt = static_cast<uint32_t>((baseTaskCnt + tdp->aicNum - 1) / tdp->aicNum);
}

__aicore__ inline void Process()
{
    if ASCEND_IS_AIV {
        ProcessVec();
    } else {
        ProcessCube();
    }
}

__aicore__ inline void InitSyncCube()
{
    evtIDFixToMTE2 = GetTPipePtr()->AllocEventID<HardEvent::FIX_MTE2>();
}

__aicore__ inline void CloseSyncCube()
{
    GetTPipePtr()->ReleaseEventID<HardEvent::FIX_MTE2>(evtIDFixToMTE2);
}

__aicore__ inline void SyncFixPipe()
{
    SetFlag<HardEvent::FIX_MTE2>(evtIDFixToMTE2);
    WaitFlag<HardEvent::FIX_MTE2>(evtIDFixToMTE2);
}

__aicore__ inline void InitSyncVec()
{
    evtIDVToS = GetTPipePtr()->AllocEventID<HardEvent::V_S>();
    evtIDSToV = GetTPipePtr()->AllocEventID<HardEvent::S_V>();
}

__aicore__ inline void CloseSyncVec()
{
    GetTPipePtr()->ReleaseEventID<HardEvent::V_S>(evtIDVToS);
    GetTPipePtr()->ReleaseEventID<HardEvent::S_V>(evtIDSToV);
}

__aicore__ inline void SyncAllVecMTE3()
{
    CrossCoreSetFlag<0, PIPE_MTE3>(VecMTE3FlagId);
    CrossCoreWaitFlag(VecMTE3FlagId);
}

__aicore__ inline void SyncGroupVecMTE2()
{
    CrossCoreSetFlag<1, PIPE_MTE2>(VecMTE2FlagId);
    CrossCoreWaitFlag(VecMTE2FlagId);
}

__aicore__ inline void ProcessCube()
{
    InitSyncCube();

    // 逐步启动
    for (uint32_t step = 0; step < baseTaskEpochCnt + SIZE_3; step++) {
        if (step >= SIZE_2 and step < baseTaskEpochCnt + SIZE_2) {
            ProcessC2(step);
        }
        if (step < baseTaskEpochCnt) {
            ProcessC1(step);
        }
    }

    CloseSyncCube();
}

__aicore__ inline void ProcessC1(uint32_t step)
{
    // 对当前cube的当前任务轮次，启动mm0。根据轮次的奇偶选择WS输出
    baseTaskEpoch = step;
    baseTaskIdx = baseTaskEpoch * tdp->aicNum + blockIdx;
    // baseTaskIdx超越边界时，仅保留核间同步
    if (baseTaskIdx >= baseTaskCnt) {
        return;
    }
    iterator1.getAtIndex(baseTaskIdx, baseTaskRowIdx, baseTaskColIdx);
    WSIdx = baseTaskEpoch % SIZE_2;

    ProcessMM0();
    CrossCoreSetFlag<CVModeId, PIPE_FIX>(C1ToV1FlagId);
}

__aicore__ inline void ProcessMM0()
{
    LaunchMM0();
    mm0.End();
}

__aicore__ inline void LaunchMM0()
{
    auto gmA = inputGM[(uint64_t)baseTaskRowIdx * tdp->baseBT * tdp->H];  // [baseBT, H]
    auto gmB = weightGM[(uint64_t)baseTaskColIdx * tdp->baseV * tdp->H];  // [baseV, H].T
    auto& gmC = mm0ResultWS[WSIdx];  // [baseBT, baseV]
    int singleCoreM = (baseTaskRowIdx == baseBTCnt - 1) ? tdp->baseBTTail : tdp->mm0Tiling.singleCoreM;
    int singleCoreN = (baseTaskColIdx == baseVCnt - 1) ? tdp->baseVTail : tdp->mm0Tiling.singleCoreN;
    mm0.SetOrgShape(singleCoreM, singleCoreN, tdp->mm0Tiling.singleCoreK);
    mm0.SetSingleShape(singleCoreM, singleCoreN, tdp->mm0Tiling.singleCoreK);
    mm0.SetTensorA(gmA, false);
    mm0.SetTensorB(gmB, true);
    #ifndef __CCE_KT_TEST__
    // kernel ut暂不支持此接口
    mm0.IterateAll(gmC);
    #endif
}

__aicore__ inline void ProcessC2(uint32_t step)
{
    // 对当前cube的当前任务轮次，启动mm1、mm2。根据轮次的奇偶选择WS
    baseTaskEpoch = step - 2U;
    baseTaskIdx = baseTaskEpoch * tdp->aicNum + blockIdx;
    if (baseTaskIdx >= baseTaskCnt) {
        return;
    }
    iterator2.getAtIndex(baseTaskIdx, baseTaskRowIdx, baseTaskColIdx);
    WSIdx = baseTaskEpoch % SIZE_2;

    CrossCoreWaitFlag(V1ToC2FlagId);
    ProcessMM1();
    ProcessMM2();
    CrossCoreSetFlag<CVModeId, PIPE_FIX>(C2ToV2FlagId);
}

__aicore__ inline void ProcessMM1()
{
    LaunchMM1();
    mm1.End();
}

__aicore__ inline void LaunchMM1()
{
    auto& gmA = mm0ResultHalfWS[WSIdx];  // [baseBT, baseV]
    auto gmB = weightGM[(uint64_t)baseTaskColIdx * tdp->baseV * tdp->H];  // [baseV, H]
    auto& gmC = mm1ResultWS[WSIdx][blockIdx];  // [baseBT, H]
    int singleCoreM = (baseTaskRowIdx == baseBTCnt - 1) ? tdp->baseBTTail : tdp->mm1Tiling.singleCoreM;
    int singleCoreK = (baseTaskColIdx == baseVCnt - 1) ? tdp->baseVTail : tdp->mm1Tiling.singleCoreK;
    mm1.SetOrgShape(singleCoreM, tdp->mm1Tiling.singleCoreN, singleCoreK);
    mm1.SetSingleShape(singleCoreM, tdp->mm1Tiling.singleCoreN, singleCoreK);
    mm1.SetTensorA(gmA, false);
    mm1.SetTensorB(gmB, false);
    #ifndef __CCE_KT_TEST__
    mm1.IterateAll(gmC);
    #endif
}

__aicore__ inline void ProcessMM2()
{
    LaunchMM2();
    mm2.End();
}

__aicore__ inline void LaunchMM2()
{   
    auto& gmA = mm0ResultHalfWS[WSIdx];  // [baseBT, baseV].T
    auto gmB = inputGM[(uint64_t)baseTaskRowIdx * tdp->baseBT * tdp->H];  // [baseBT, H]
    auto& gmC = mm2ResultWS[WSIdx][blockIdx];  // [baseV, H]
    int singleCoreK = (baseTaskRowIdx == baseBTCnt - 1) ? tdp->baseBTTail : tdp->mm2Tiling.singleCoreK;
    int singleCoreM = (baseTaskColIdx == baseVCnt - 1) ? tdp->baseVTail : tdp->mm2Tiling.singleCoreM;
    mm2.SetOrgShape(singleCoreM, tdp->mm2Tiling.singleCoreN, singleCoreK);
    mm2.SetSingleShape(singleCoreM, tdp->mm2Tiling.singleCoreN, singleCoreK);
    mm2.SetTensorA(gmA, true);
    mm2.SetTensorB(gmB, false);
    #ifndef __CCE_KT_TEST__
    mm2.IterateAll(gmC);
    #endif
}

__aicore__ inline void ProcessVec()
{
    InitSyncVec();

    // 逐步启动
    for (uint32_t step = 0; step < baseTaskEpochCnt + SIZE_3; step++) {
        if (step >= SIZE_3) {
            ProcessV2(step);
        }
        if (step >= 1 and step < baseTaskEpochCnt + 1) {
            ProcessV1(step);
        }
    }
    ProcessFinalCast();

    CloseSyncVec();
}

/*
  对当前vec的当前任务轮次，启动C2前的矢量处理。根据轮次的奇偶选择WS
  两个vec平分当前任务，采用vec和标量运算接口，复用UB，可以一次性搬入UB
  vec被掩盖的情形下，不必db
*/
__aicore__ inline void ProcessV1(uint32_t step)
{
    baseTaskEpoch = step - 1U;
    baseTaskIdx = baseTaskEpoch * tdp->aicNum + aicBlockIdx;
    if (baseTaskIdx >= baseTaskCnt) {
        SyncAllVecMTE3();
        return;
    }
    iterator1.getAtIndex(baseTaskIdx, baseTaskRowIdx, baseTaskColIdx);
    WSIdx = baseTaskEpoch % SIZE_2;
    if (baseTaskRowIdx != baseBTCnt - 1) {
        V1MainRowCnt = tdp->baseBT / SIZE_2;  // baseBT一定为偶数
        if (not isVec0) {
            lastV1MainRowCnt = V1MainRowCnt;
        }
    } else {
        V1MainRowCnt = tdp->baseBTTail / SIZE_2;
        if (isVec0) {
            V1MainRowCnt = tdp->baseBTTail - V1MainRowCnt;
        } else {
            lastV1MainRowCnt = tdp->baseBTTail - V1MainRowCnt;
        }
    }
    if (baseTaskColIdx != baseVCnt - 1) {
        V1MainColCnt = tdp->baseV;
        V1InUBStride = 0U;
        V1OutUBStride = 0U;
    } else {
        V1MainColCnt = tdp->baseVTail;
        V1InUBStride = tdp->V1InUBStride;
        V1OutUBStride = tdp->V1OutUBStride;
    }

    SyncAllVecMTE3();  // 等V2或InitGlobalMemory
    CrossCoreWaitFlag(C1ToV1FlagId);
    if (V1MainRowCnt) {
        CopyInV1();
        ComputeV1();
        SyncGroupVecMTE2();
        CopyOutV1();
    } else {
        SyncGroupVecMTE2();
    }
    CrossCoreSetFlag<CVModeId, PIPE_MTE3>(V1ToC2FlagId);
}

__aicore__ inline void CopyInV1()
{
    auto mainUB = V1MainQ.AllocTensor<float>();
    DataCopyExtParams copyParams{V1MainRowCnt, (uint32_t)(V1MainColCnt * IN_BYTE_SIZE),
                                 0,
                                 V1InUBStride,
                                 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(
        mainUB,
        mm0ResultWS[WSIdx][(isVec0) ? 0 : (lastV1MainRowCnt * V1MainColCnt)],
        copyParams,
        padParams
    );
    V1MainQ.EnQue<TPosition::GM, TPosition::VECIN, float>(mainUB);
}

__aicore__ inline void ComputeV1()
{
    uint32_t mainStartRow = baseTaskRowIdx * tdp->baseBT + ((isVec0) ? 0 : lastV1MainRowCnt);  // 当前处理数据的起始行号
    uint32_t mainStartCol = baseTaskColIdx * tdp->baseV;  // 当前处理数据的起始列号
    auto mainUB = V1MainQ.DeQue<TPosition::GM, TPosition::VECIN, float>();
    auto mainHalfUB = mainUB.ReinterpretCast<HalfType>();
    SubV1(mainStartRow, mainUB);

    PipeBarrier<PIPE_V>();
    ExpV1(mainUB);

    PipeBarrier<PIPE_V>();
    DivV1(mainStartRow, mainUB);

    SetFlag<HardEvent::V_S>(evtIDVToS);
    WaitFlag<HardEvent::V_S>(evtIDVToS);
    UpdateV1(mainStartRow, mainStartCol, mainUB);
    
    SetFlag<HardEvent::S_V>(evtIDSToV);
    WaitFlag<HardEvent::S_V>(evtIDSToV);
    MulV1(mainStartRow, mainUB);

    PipeBarrier<PIPE_V>();
    CastV1(mainUB, mainHalfUB);
    V1MainQ.EnQue<TPosition::VECOUT, TPosition::GM, float>(mainUB);
}

__aicore__ inline void SubV1(uint32_t mainStartRow, LocalTensor<float>& mainUB)
{
    for (uint32_t row = 0; row < V1MainRowCnt; row++) {
        auto logits_max = NEG * logitsMaxGM.GetValue(mainStartRow + row);
        Adds(mainUB[row * tdp->baseV], mainUB[row * tdp->baseV], logits_max, V1MainColCnt);
    }
}

__aicore__ inline void ExpV1(LocalTensor<float>& mainUB)
{
    Exp(mainUB, mainUB, (int32_t)tdp->V1MainSize);
}

__aicore__ inline void DivV1(uint32_t mainStartRow, LocalTensor<float>& mainUB)
{
    for (uint32_t row = 0; row < V1MainRowCnt; row++) {
        auto sum_exp_logits = sumExpLogitsGM.GetValue(mainStartRow + row);
        auto currentRowUB = mainUB[row * tdp->baseV];
        PipeBarrier<PIPE_V>();
        Duplicate(divisorUB, sum_exp_logits, V1MainColCnt);
        PipeBarrier<PIPE_V>();
        Div(currentRowUB, currentRowUB, divisorUB, V1MainColCnt);
    }
}

__aicore__ inline bool GetBoolFromUint8Tensor(uint32_t bitIndex)
{
    // 计算目标位所在的字节索引和位偏移
    size_t byteIndex = bitIndex / 8;
    size_t bitOffset = bitIndex % 8;
    uint8_t cachedByte = targetMaskGM.GetValue(byteIndex);
    return (cachedByte >> bitOffset) & 0x01;
}

__aicore__ inline void UpdateV1(uint32_t mainStartRow, uint32_t mainStartCol, LocalTensor<float>& mainUB)
{
    uint32_t mainEndCol = mainStartCol + V1MainColCnt;
    for (uint32_t row = 0; row < V1MainRowCnt; row++) {
        bool targetMask;
        uint32_t rowIdx = mainStartRow + row;
        if constexpr (std::is_same_v<DTYPE_TARGET_MASK, uint8_t>) {
            targetMask = GetBoolFromUint8Tensor(rowIdx);
        } else {
            targetMask = targetMaskGM.GetValue(rowIdx);
        }
        if (!targetMask) {
            auto maskedTarget = maskedTargetGM.GetValue(rowIdx);
            if (maskedTarget >= mainStartCol and maskedTarget < mainEndCol) {
                uint32_t targetOffset = row * tdp->baseV + maskedTarget - mainStartCol;
                mainUB.SetValue(targetOffset, mainUB.GetValue(targetOffset) - 1.0f);
            }
        }
    }
}

__aicore__ inline void MulV1(uint32_t mainStartRow, LocalTensor<float>& mainUB)
{
    for (uint32_t row = 0; row < V1MainRowCnt; row++) {
        auto grad = gradGM.GetValue(mainStartRow + row);
        Muls(mainUB[row * tdp->baseV], mainUB[row * tdp->baseV], grad, V1MainColCnt);
    }
}

__aicore__ inline void CastV1(LocalTensor<float>& mainUB, LocalTensor<HalfType>& mainHalfUB)
{
    Cast(mainHalfUB, mainUB, RoundMode::CAST_RINT, tdp->V1MainSize);
}

__aicore__ inline void CopyOutV1()
{
    auto mainUB = V1MainQ.DeQue<TPosition::VECOUT, TPosition::GM, float>();
    auto mainHalfUB = mainUB.ReinterpretCast<HalfType>();
    DataCopyExtParams copyParams{V1MainRowCnt, (uint32_t)(V1MainColCnt * OUT_BYTE_SIZE),
                                 V1OutUBStride,
                                 0,
                                 0};
    DataCopyPad(
        mm0ResultHalfWS[WSIdx][(isVec0) ? 0 : (lastV1MainRowCnt * V1MainColCnt)],
        mainHalfUB,
        copyParams
    );
    V1MainQ.FreeTensor(mainUB);
}

/*
  对当前vec的当前任务轮次，启动C2后的矢量处理。根据轮次的奇偶选择WS
  单vec处理当前轮次内同输出地址的累加任务（称为同组），每个vec0处理mm1的结果，vec1处理mm2的结果
  每个vec先明确自己的组，然后从当轮内第0个任务遍历到自己，如果自身为组内第一个核心，则负责处理，否则空闲
*/
__aicore__ inline void ProcessV2(uint32_t step)
{
    baseTaskEpoch = step - 3U;
    baseTaskIdx = baseTaskEpoch * tdp->aicNum + aicBlockIdx;
    if (baseTaskIdx >= baseTaskCnt) {
        SyncAllVecMTE3();
        return;
    }
    iterator2.getAtIndex(baseTaskIdx, baseTaskRowIdx, baseTaskColIdx);
    WSIdx = baseTaskEpoch % SIZE_2;

    // 判断任务
    bool handlingMM1 = true;
    bool handlingMM2 = true;
    for (uint32_t taskIdxOffset = 0; taskIdxOffset < aicBlockIdx; taskIdxOffset++) {
        uint32_t currIdx = baseTaskEpoch * tdp->aicNum + taskIdxOffset;
        uint32_t currRowIdx, currColIdx;
        iterator2.getAtIndex(currIdx, currRowIdx, currColIdx);
        if (handlingMM1) {
            if (currRowIdx == baseTaskRowIdx) {
                handlingMM1 = false;
            }
        }
        if (handlingMM2) {
            if (currColIdx == baseTaskColIdx) {
                handlingMM2 = false;
            }
        }
    }

    CrossCoreWaitFlag(C2ToV2FlagId);
    SyncAllVecMTE3();  // 代码位置同步 & 等V1
    if (handlingMM1 and isVec0) {
        ProcessV2MM1();
    }
    if (handlingMM2 and !isVec0) {
        ProcessV2MM2();
    }
}

__aicore__ inline void ProcessV2MM1()
{
    // 统计组内任务
    grpTaskCnt = 0U;
    for (uint32_t taskIdxOffset = aicBlockIdx; taskIdxOffset < tdp->aicNum; taskIdxOffset++) {
        uint32_t currIdx = baseTaskEpoch * tdp->aicNum + taskIdxOffset;
        uint32_t currRowIdx, currColIdx;
        iterator2.getAtIndex(currIdx, currRowIdx, currColIdx);
        if (currRowIdx != baseTaskRowIdx or currIdx >= baseTaskCnt) continue;
        grpTaskIdxOffset[grpTaskCnt] = taskIdxOffset;
        grpTaskCnt += 1;
    }
    uint32_t baseBT = (baseTaskRowIdx == baseBTCnt - 1) ? tdp->baseBTTail : tdp->baseBT;

    // 按行数启动组内累加
    for (uint32_t rowOffset = 0; rowOffset < baseBT; rowOffset += tdp->mmAccumRowOnetime) {
        uint32_t rowCnt = tdp->mmAccumRowOnetime;
        if (rowOffset + rowCnt > baseBT) {
            rowCnt = baseBT - rowOffset;
        }
        DoV2MM1Group(rowOffset, rowCnt);  // 参数为baseTask内的行偏移与行数
    }
}

__aicore__ inline void DoV2MM1Group(uint32_t rowOffset, uint32_t rowCnt)
{
    // 使用UB逐套循环读入目标ws的结果组内的mm1Result，完成累加
    if (tdp->mmAccumRowOnetime >= SIZE_2 or tdp->mmAccumCyclePerRow == 1) {
        // 把目标ws的结果读取到UB
        CopyInTargetV2MM1(rowOffset, rowCnt);
        auto targetUB = V2TargetQ.DeQue<TPosition::GM, TPosition::VECIN, float>();
        // 遍历组内任务
        for (uint32_t idx = 0; idx < grpTaskCnt; idx++) {
            uint32_t taskIdxOffset = grpTaskIdxOffset[idx];
            // 搬入
            CopyInAdditionV2MM1(rowOffset, rowCnt, taskIdxOffset);
            // 累加
            ComputeV2MM1(targetUB);
        }
        // 搬出累加结果到目标ws
        V2TargetQ.EnQue<TPosition::VECOUT, TPosition::GM, float>(targetUB);
        CopyOutV2MM1(rowOffset, rowCnt);
    } else {
        // 遍历cycle
        for (uint32_t cycleIdx = 0; cycleIdx < tdp->mmAccumCyclePerRow; cycleIdx++) {
            // 把目标ws的结果读取到UB
            uint32_t accumCnt = (cycleIdx == tdp->mmAccumCyclePerRow - 1) ? tdp->mmAccumCntLastCycle : tdp->mmAccumCntPerCycle;
            CopyInTargetV2MM1Large(rowOffset, cycleIdx, accumCnt);
            auto targetUB = V2TargetQ.DeQue<TPosition::GM, TPosition::VECIN, float>();
            // 遍历组内任务
            for (uint32_t idx = 0; idx < grpTaskCnt; idx++) {
                uint32_t taskIdxOffset = grpTaskIdxOffset[idx];
                // 搬入
                CopyInAdditionV2MM1Large(rowOffset, cycleIdx, taskIdxOffset, accumCnt);
                // 累加
                ComputeV2MM1(targetUB);
            }
            V2TargetQ.EnQue<TPosition::VECOUT, TPosition::GM, float>(targetUB);
            CopyOutV2MM1Large(rowOffset, cycleIdx, accumCnt);
        }
    }
}

// 多行处理场景
__aicore__ inline void CopyInTargetV2MM1(uint32_t rowOffset, uint32_t rowCnt)
{
    auto targetUB = V2TargetQ.AllocTensor<float>();
    DataCopyExtParams copyParams{1, (uint32_t)(rowCnt * tdp->H * IN_BYTE_SIZE),
                                 0,
                                 0,
                                 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(
        targetUB,
        mm1AccumWS[(uint64_t)baseTaskRowIdx * tdp->baseBT * tdp->H + rowOffset * tdp->H],
        copyParams,
        padParams
    );
    V2TargetQ.EnQue<TPosition::GM, TPosition::VECIN, float>(targetUB);
}

// 不到单行处理场景
__aicore__ inline void CopyInTargetV2MM1Large(uint32_t rowOffset, uint32_t cycleIdx, uint32_t accumCnt)
{
    auto targetUB = V2TargetQ.AllocTensor<float>();
    DataCopyExtParams copyParams{1, (uint32_t)(accumCnt * IN_BYTE_SIZE),
                                 0,
                                 0,
                                 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(
        targetUB,
        mm1AccumWS[(uint64_t)baseTaskRowIdx * tdp->baseBT * tdp->H + rowOffset * tdp->H + cycleIdx * tdp->mmAccumCntPerCycle],
        copyParams,
        padParams
    );
    V2TargetQ.EnQue<TPosition::GM, TPosition::VECIN, float>(targetUB);
}

__aicore__ inline void CopyInAdditionV2MM1(uint32_t rowOffset, uint32_t rowCnt, uint32_t taskIdxOffset)
{
    auto additionUB = V2AdditionQ.AllocTensor<float>();
    DataCopyExtParams copyParams{1, (uint32_t)(rowCnt * tdp->H * IN_BYTE_SIZE),
                                 0,
                                 0,
                                 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(
        additionUB,
        mm1ResultWS[WSIdx][taskIdxOffset][(uint64_t)rowOffset * tdp->H],
        copyParams,
        padParams
    );
    V2AdditionQ.EnQue(additionUB);
}

__aicore__ inline void CopyInAdditionV2MM1Large(uint32_t rowOffset, uint32_t cycleIdx, uint32_t taskIdxOffset, uint32_t accumCnt)
{
    auto additionUB = V2AdditionQ.AllocTensor<float>();
    DataCopyExtParams copyParams{1, (uint32_t)(accumCnt * IN_BYTE_SIZE),
                                 0,
                                 0,
                                 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(
        additionUB,
        mm1ResultWS[WSIdx][taskIdxOffset][(uint64_t)rowOffset * tdp->H + cycleIdx * tdp->mmAccumCntPerCycle],
        copyParams,
        padParams
    );
    V2AdditionQ.EnQue(additionUB);
}

__aicore__ inline void ComputeV2MM1(LocalTensor<float>& targetUB)
{
    auto additionUB = V2AdditionQ.DeQue<float>();
    targetUB = targetUB + additionUB;
    V2AdditionQ.FreeTensor(additionUB);
}

__aicore__ inline void CopyOutV2MM1(uint32_t rowOffset, uint32_t rowCnt)
{
    auto targetUB = V2TargetQ.DeQue<TPosition::VECOUT, TPosition::GM, float>();
    DataCopyExtParams copyParams{1, (uint32_t)(rowCnt * tdp->H * IN_BYTE_SIZE),
                                 0,
                                 0,
                                 0};
    DataCopyPad(
        mm1AccumWS[(uint64_t)baseTaskRowIdx * tdp->baseBT * tdp->H + rowOffset * tdp->H],
        targetUB,
        copyParams
    );
    V2TargetQ.FreeTensor(targetUB);
}

__aicore__ inline void CopyOutV2MM1Large(uint32_t rowOffset, uint32_t cycleIdx, uint32_t accumCnt)
{
    auto targetUB = V2TargetQ.DeQue<TPosition::VECOUT, TPosition::GM, float>();
    DataCopyExtParams copyParams{1, (uint32_t)(accumCnt * IN_BYTE_SIZE),
                                 0,
                                 0,
                                 0};
    DataCopyPad(
        mm1AccumWS[(uint64_t)baseTaskRowIdx * tdp->baseBT * tdp->H + rowOffset * tdp->H + cycleIdx * tdp->mmAccumCntPerCycle],
        targetUB,
        copyParams
    );
    V2TargetQ.FreeTensor(targetUB);
}

__aicore__ inline void ProcessV2MM2()
{
    // 统计组内任务
    grpTaskCnt = 0U;
    for (uint32_t taskIdxOffset = aicBlockIdx; taskIdxOffset < tdp->aicNum; taskIdxOffset++) {
        uint32_t currIdx = baseTaskEpoch * tdp->aicNum + taskIdxOffset;
        uint32_t currRowIdx, currColIdx;
        iterator2.getAtIndex(currIdx, currRowIdx, currColIdx);
        if (currColIdx != baseTaskColIdx or currIdx >= baseTaskCnt) continue;
        grpTaskIdxOffset[grpTaskCnt] = taskIdxOffset;
        grpTaskCnt += 1;
    }
    uint32_t baseV = (baseTaskColIdx == baseVCnt - 1) ? tdp->baseVTail : tdp->baseV;

    // 按行数启动组内累加
    for (uint32_t rowOffset = 0; rowOffset < baseV; rowOffset += tdp->mmAccumRowOnetime) {
        uint32_t rowCnt = tdp->mmAccumRowOnetime;
        if (rowOffset + rowCnt > baseV) {
            rowCnt = baseV - rowOffset;
        }
        DoV2MM2Group(rowOffset, rowCnt);  // 参数为baseTask内的行偏移与行数
    }
}

__aicore__ inline void DoV2MM2Group(uint32_t rowOffset, uint32_t rowCnt)
{
    // 使用UB逐套循环读入目标ws的结果组内的mm1Result，完成累加
    if (tdp->mmAccumRowOnetime >= SIZE_2 or tdp->mmAccumCyclePerRow == 1) {
        // 把目标ws的结果读取到UB
        CopyInTargetV2MM2(rowOffset, rowCnt);
        auto targetUB = V2TargetQ.DeQue<TPosition::GM, TPosition::VECIN, float>();
        // 遍历组内任务
        for (uint32_t idx = 0; idx < grpTaskCnt; idx++) {
            uint32_t taskIdxOffset = grpTaskIdxOffset[idx];
            // 搬入
            CopyInAdditionV2MM2(rowOffset, rowCnt, taskIdxOffset);
            // 累加
            ComputeV2MM2(targetUB);
        }
        // 搬出累加结果到目标ws
        V2TargetQ.EnQue<TPosition::VECOUT, TPosition::GM, float>(targetUB);
        CopyOutV2MM2(rowOffset, rowCnt);
    } else {
        // 遍历cycle
        for (uint32_t cycleIdx = 0; cycleIdx < tdp->mmAccumCyclePerRow; cycleIdx++) {
            // 把目标ws的结果读取到UB
            uint32_t accumCnt = (cycleIdx == tdp->mmAccumCyclePerRow - 1) ? tdp->mmAccumCntLastCycle : tdp->mmAccumCntPerCycle;
            CopyInTargetV2MM2Large(rowOffset, cycleIdx, accumCnt);
            auto targetUB = V2TargetQ.DeQue<TPosition::GM, TPosition::VECIN, float>();
            // 遍历组内任务
            for (uint32_t idx = 0; idx < grpTaskCnt; idx++) {
                uint32_t taskIdxOffset = grpTaskIdxOffset[idx];
                // 搬入
                CopyInAdditionV2MM2Large(rowOffset, cycleIdx, taskIdxOffset, accumCnt);
                // 累加
                ComputeV2MM2(targetUB);
            }
            V2TargetQ.EnQue<TPosition::VECOUT, TPosition::GM, float>(targetUB);
            CopyOutV2MM2Large(rowOffset, cycleIdx, accumCnt);
        }
    }
}

// 多行处理场景
__aicore__ inline void CopyInTargetV2MM2(uint32_t rowOffset, uint32_t rowCnt)
{
    auto targetUB = V2TargetQ.AllocTensor<float>();
    DataCopyExtParams copyParams{1, (uint32_t)(rowCnt * tdp->H * IN_BYTE_SIZE),
                                 0,
                                 0,
                                 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(
        targetUB,
        mm2AccumWS[(uint64_t)baseTaskColIdx * tdp->baseV * tdp->H + rowOffset * tdp->H],
        copyParams,
        padParams
    );
    V2TargetQ.EnQue<TPosition::GM, TPosition::VECIN, float>(targetUB);
}

// 不到单行处理场景
__aicore__ inline void CopyInTargetV2MM2Large(uint32_t rowOffset, uint32_t cycleIdx, uint32_t accumCnt)
{
    auto targetUB = V2TargetQ.AllocTensor<float>();
    DataCopyExtParams copyParams{1, (uint32_t)(accumCnt * IN_BYTE_SIZE),
                                 0,
                                 0,
                                 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(
        targetUB,
        mm2AccumWS[(uint64_t)baseTaskColIdx * tdp->baseV * tdp->H + rowOffset * tdp->H + cycleIdx * tdp->mmAccumCntPerCycle],
        copyParams,
        padParams
    );
    V2TargetQ.EnQue<TPosition::GM, TPosition::VECIN, float>(targetUB);
}

__aicore__ inline void CopyInAdditionV2MM2(uint32_t rowOffset, uint32_t rowCnt, uint32_t taskIdxOffset)
{
    auto additionUB = V2AdditionQ.AllocTensor<float>();
    DataCopyExtParams copyParams{1, (uint32_t)(rowCnt * tdp->H * IN_BYTE_SIZE),
                                 0,
                                 0,
                                 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(
        additionUB,
        mm2ResultWS[WSIdx][taskIdxOffset][(uint64_t)rowOffset * tdp->H],
        copyParams,
        padParams
    );
    V2AdditionQ.EnQue(additionUB);
}

__aicore__ inline void CopyInAdditionV2MM2Large(uint32_t rowOffset, uint32_t cycleIdx, uint32_t taskIdxOffset, uint32_t accumCnt)
{
    auto additionUB = V2AdditionQ.AllocTensor<float>();
    DataCopyExtParams copyParams{1, (uint32_t)(accumCnt * IN_BYTE_SIZE),
                                 0,
                                 0,
                                 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(
        additionUB,
        mm2ResultWS[WSIdx][taskIdxOffset][(uint64_t)rowOffset * tdp->H + cycleIdx * tdp->mmAccumCntPerCycle],
        copyParams,
        padParams
    );
    V2AdditionQ.EnQue(additionUB);
}

__aicore__ inline void ComputeV2MM2(LocalTensor<float>& targetUB)
{
    auto additionUB = V2AdditionQ.DeQue<float>();
    targetUB = targetUB + additionUB;
    V2AdditionQ.FreeTensor(additionUB);
}

__aicore__ inline void CopyOutV2MM2(uint32_t rowOffset, uint32_t rowCnt)
{
    auto targetUB = V2TargetQ.DeQue<TPosition::VECOUT, TPosition::GM, float>();
    DataCopyExtParams copyParams{1, (uint32_t)(rowCnt * tdp->H * IN_BYTE_SIZE),
                                 0,
                                 0,
                                 0};
    DataCopyPad(
        mm2AccumWS[(uint64_t)baseTaskColIdx * tdp->baseV * tdp->H + rowOffset * tdp->H],
        targetUB,
        copyParams
    );
    V2TargetQ.FreeTensor(targetUB);
}

__aicore__ inline void CopyOutV2MM2Large(uint32_t rowOffset, uint32_t cycleIdx, uint32_t accumCnt)
{
    auto targetUB = V2TargetQ.DeQue<TPosition::VECOUT, TPosition::GM, float>();
    DataCopyExtParams copyParams{1, (uint32_t)(accumCnt * IN_BYTE_SIZE),
                                 0,
                                 0,
                                 0};
    DataCopyPad(
        mm2AccumWS[(uint64_t)baseTaskColIdx * tdp->baseV * tdp->H + rowOffset * tdp->H + cycleIdx * tdp->mmAccumCntPerCycle],
        targetUB,
        copyParams
    );
    V2TargetQ.FreeTensor(targetUB);
}

__aicore__ inline void ProcessFinalCast()
{
    SyncAllVecMTE3();  // 等V2
    ProcessFinalCastMM1();
    ProcessFinalCastMM2();
}

__aicore__ inline void ProcessFinalCastMM1()
{
    uint32_t rowCurrVec;  // 当前核心要处理的行数
    uint32_t rowStart;  // 起始行在全局的偏移
    if (blockIdx < tdp->mm1CastExtraRowCnt) {
        rowCurrVec = tdp->mm1CastRowPerVec + 1U;
        rowStart = tdp->mm1CastRowPerVec * blockIdx + blockIdx; 
    } else {
        rowCurrVec = tdp->mm1CastRowPerVec;
        rowStart = tdp->mm1CastRowPerVec * blockIdx + tdp->mm1CastExtraRowCnt; 
    }
    
    // 按行数启动cast
    for (uint32_t rowOffset = 0; rowOffset < rowCurrVec; rowOffset += tdp->mmCastRowOnetime) {
        uint32_t rowCnt = tdp->mmCastRowOnetime;
        if (rowOffset + rowCnt > rowCurrVec) {
            rowCnt = rowCurrVec - rowOffset;
        }
        uint32_t rowIdx = rowStart + rowOffset;
        if (tdp->mmCastRowOnetime >= SIZE_2 or tdp->mmCastCyclePerRow == 1) {
            CopyInFinalCastMM1(rowIdx, rowCnt);
            ComputeFinalCast();
            CopyOutFinalCastMM1(rowIdx, rowCnt);
        } else {
            // 遍历cycle
            for (uint32_t cycleIdx = 0; cycleIdx < tdp->mmCastCyclePerRow; cycleIdx++) {
                uint32_t castCnt = (cycleIdx == tdp->mmCastCyclePerRow - 1) ? tdp->mmCastCntLastCycle : tdp->mmCastCntPerCycle;
                CopyInFinalCastMM1Large(rowIdx, cycleIdx, castCnt);
                ComputeFinalCast();
                CopyOutFinalCastMM1Large(rowIdx, cycleIdx, castCnt);
            }
        }
    }
}

__aicore__ inline void CopyInFinalCastMM1(uint32_t rowIdx, uint32_t rowCnt)
{
    auto castUB = castMainQ.AllocTensor<float>();
    DataCopyExtParams copyParams{1, (uint32_t)(rowCnt * tdp->H * IN_BYTE_SIZE),
                                 0,
                                 0,
                                 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(
        castUB,
        mm1AccumWS[(uint64_t)rowIdx * tdp->H],
        copyParams,
        padParams
    );
    castMainQ.EnQue<TPosition::GM, TPosition::VECIN, float>(castUB);
}

__aicore__ inline void CopyInFinalCastMM1Large(uint32_t rowIdx, uint32_t cycleIdx, uint32_t castCnt)
{
    auto castUB = castMainQ.AllocTensor<float>();
    DataCopyExtParams copyParams{1, (uint32_t)(castCnt * IN_BYTE_SIZE),
                                 0,
                                 0,
                                 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(
        castUB,
        mm1AccumWS[(uint64_t)rowIdx * tdp->H + cycleIdx * tdp->mmCastCntPerCycle],
        copyParams,
        padParams
    );
    castMainQ.EnQue<TPosition::GM, TPosition::VECIN, float>(castUB);
}

__aicore__ inline void ComputeFinalCast()
{
    auto castUB = castMainQ.DeQue<TPosition::GM, TPosition::VECIN, float>();
    auto castHalfUB = castUB.ReinterpretCast<HalfType>();
    Cast(castHalfUB, castUB, RoundMode::CAST_RINT, tdp->castMainSize);
    castMainQ.EnQue<TPosition::VECOUT, TPosition::GM, float>(castUB);
}

__aicore__ inline void CopyOutFinalCastMM1(uint32_t rowIdx, uint32_t rowCnt)
{
    auto castUB = castMainQ.DeQue<TPosition::VECOUT, TPosition::GM, float>();
    auto castHalfUB = castUB.ReinterpretCast<HalfType>();
    DataCopyExtParams copyParams{1, (uint32_t)(rowCnt * tdp->H * OUT_BYTE_SIZE),
                                 0,
                                 0,
                                 0};
    DataCopyPad(
        inputGradOutGM[(uint64_t)rowIdx * tdp->H],
        castHalfUB,
        copyParams
    );
    castMainQ.FreeTensor(castUB);
}

__aicore__ inline void CopyOutFinalCastMM1Large(uint32_t rowIdx, uint32_t cycleIdx, uint32_t castCnt)
{
    auto castUB = castMainQ.DeQue<TPosition::VECOUT, TPosition::GM, float>();
    auto castHalfUB = castUB.ReinterpretCast<HalfType>();
    DataCopyExtParams copyParams{1, (uint32_t)(castCnt * OUT_BYTE_SIZE),
                                 0,
                                 0,
                                 0};
    DataCopyPad(
        inputGradOutGM[(uint64_t)rowIdx * tdp->H + cycleIdx * tdp->mmCastCntPerCycle],
        castHalfUB,
        copyParams
    );
    castMainQ.FreeTensor(castUB);
}

__aicore__ inline void ProcessFinalCastMM2()
{
    uint32_t rowCurrVec;  // 当前核心要处理的行数
    uint32_t rowStart;  // 起始行在全局的偏移
    if (blockIdx < tdp->mm2CastExtraRowCnt) {
        rowCurrVec = tdp->mm2CastRowPerVec + 1U;
        rowStart = tdp->mm2CastRowPerVec * blockIdx + blockIdx; 
    } else {
        rowCurrVec = tdp->mm2CastRowPerVec;
        rowStart = tdp->mm2CastRowPerVec * blockIdx + tdp->mm2CastExtraRowCnt; 
    }
    
    // 按行数启动cast
    for (uint32_t rowOffset = 0; rowOffset < rowCurrVec; rowOffset += tdp->mmCastRowOnetime) {
        uint32_t rowCnt = tdp->mmCastRowOnetime;
        if (rowOffset + rowCnt > rowCurrVec) {
            rowCnt = rowCurrVec - rowOffset;
        }
        uint32_t rowIdx = rowStart + rowOffset;
        if (tdp->mmCastRowOnetime >= SIZE_2 or tdp->mmCastCyclePerRow == 1) {
            CopyInFinalCastMM2(rowIdx, rowCnt);
            ComputeFinalCast();
            CopyOutFinalCastMM2(rowIdx, rowCnt);
        } else {
            // 遍历cycle
            for (uint32_t cycleIdx = 0; cycleIdx < tdp->mmCastCyclePerRow; cycleIdx++) {
                uint32_t castCnt = (cycleIdx == tdp->mmCastCyclePerRow - 1) ? tdp->mmCastCntLastCycle : tdp->mmCastCntPerCycle;
                CopyInFinalCastMM2Large(rowIdx, cycleIdx, castCnt);
                ComputeFinalCast();
                CopyOutFinalCastMM2Large(rowIdx, cycleIdx, castCnt);
            }
        }
    }
}

__aicore__ inline void CopyInFinalCastMM2(uint32_t rowIdx, uint32_t rowCnt)
{
    auto castUB = castMainQ.AllocTensor<float>();
    DataCopyExtParams copyParams{1, (uint32_t)(rowCnt * tdp->H * IN_BYTE_SIZE),
                                 0,
                                 0,
                                 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(
        castUB,
        mm2AccumWS[(uint64_t)rowIdx * tdp->H],
        copyParams,
        padParams
    );
    castMainQ.EnQue<TPosition::GM, TPosition::VECIN, float>(castUB);
}

__aicore__ inline void CopyInFinalCastMM2Large(uint32_t rowIdx, uint32_t cycleIdx, uint32_t castCnt)
{
    auto castUB = castMainQ.AllocTensor<float>();
    DataCopyExtParams copyParams{1, (uint32_t)(castCnt * IN_BYTE_SIZE),
                                 0,
                                 0,
                                 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(
        castUB,
        mm2AccumWS[(uint64_t)rowIdx * tdp->H + cycleIdx * tdp->mmCastCntPerCycle],
        copyParams,
        padParams
    );
    castMainQ.EnQue<TPosition::GM, TPosition::VECIN, float>(castUB);
}

__aicore__ inline void CopyOutFinalCastMM2(uint32_t rowIdx, uint32_t rowCnt)
{
    auto castUB = castMainQ.DeQue<TPosition::VECOUT, TPosition::GM, float>();
    auto castHalfUB = castUB.ReinterpretCast<HalfType>();
    DataCopyExtParams copyParams{1, (uint32_t)(rowCnt * tdp->H * OUT_BYTE_SIZE),
                                 0,
                                 0,
                                 0};
    DataCopyPad(
        weightGradOutGM[(uint64_t)rowIdx * tdp->H],
        castHalfUB,
        copyParams
    );
    castMainQ.FreeTensor(castUB);
}

__aicore__ inline void CopyOutFinalCastMM2Large(uint32_t rowIdx, uint32_t cycleIdx, uint32_t castCnt)
{
    auto castUB = castMainQ.DeQue<TPosition::VECOUT, TPosition::GM, float>();
    auto castHalfUB = castUB.ReinterpretCast<HalfType>();
    DataCopyExtParams copyParams{1, (uint32_t)(castCnt * OUT_BYTE_SIZE),
                                 0,
                                 0,
                                 0};
    DataCopyPad(
        weightGradOutGM[(uint64_t)rowIdx * tdp->H + cycleIdx * tdp->mmCastCntPerCycle],
        castHalfUB,
        copyParams
    );
    castMainQ.FreeTensor(castUB);
}

public:
// 内存与流水
TPipe pipe;
TEventID evtIDFixToMTE2;  // mm间搬入等搬出
TEventID evtIDVToS;  // vec标量等矢量
TEventID evtIDSToV;  // vec矢量等标量
GlobalTensor<HalfType> inputGM;  // [BT, H]
GlobalTensor<HalfType> weightGM;  // [V, H]
GlobalTensor<float> mm0ResultWS[SIZE_2];  // [baseBT, baseV] * aicNum
GlobalTensor<HalfType> mm0ResultHalfWS[SIZE_2];  // [baseBT, baseV] * tdp->aicNum
GlobalTensor<float> logitsMaxGM;  // [BT]
GlobalTensor<float> sumExpLogitsGM;  // [BT]
GlobalTensor<TargetMaskType> targetMaskGM;  // [BT]
GlobalTensor<MaskedTargetType> maskedTargetGM;  // [BT]
GlobalTensor<float> gradGM;  // [BT]
GlobalTensor<float> mm1ResultWS[SIZE_2][SIZE_MAX_AIC_NUM];  // [baseBT, H] * aicNum
GlobalTensor<float> mm2ResultWS[SIZE_2][SIZE_MAX_AIC_NUM];  // [baseV, H] * aicNum
GlobalTensor<float> mm1AccumWS;  // [BT, H]
GlobalTensor<float> mm2AccumWS;  // [V, H]
GlobalTensor<HalfType> inputGradOutGM;  // [BT, H]
GlobalTensor<HalfType> weightGradOutGM;  // [V, H]
TBufPool<TPosition::VECCALC> V1Pool;
TBufPool<TPosition::VECCALC> V2Pool;
TBufPool<TPosition::VECCALC> castMainPool;
TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> V1MainQ;
TBuf<TPosition::VECCALC> divisorBuf;
TQueBind<TPosition::VECIN, TPosition::VECOUT, SIZE_2> V2TargetQ;
TQue<TPosition::VECIN, SIZE_2> V2AdditionQ;
TQueBind<TPosition::VECIN, TPosition::VECOUT, SIZE_2> castMainQ;
LocalTensor<float> divisorUB;
// 全局变量
MM0 mm0;
MM1 mm1;
MM2 mm2;
uint32_t aicBlockIdx;
uint16_t V1MainRowCnt;  // V1单核需要处理的行数
uint16_t V1MainColCnt;  // V1单核需要处理的列数
uint32_t V1MainSize;  // V1单核需要处理的元素数
uint16_t lastV1MainRowCnt;  // 仅供vec1使用
uint32_t V1InUBStride;  // V1搬入参数
uint32_t V1OutUBStride;  // V1搬出参数
uint32_t baseTaskEpoch;  // 当前任务块所在轮次
uint32_t baseTaskIdx;  // 当前任务块序号
uint32_t baseTaskRowIdx;  // 当前任务块行序号
uint32_t baseTaskColIdx;  // 当前任务块列序号
uint8_t WSIdx;  // 当前使用的ws位置
uint32_t grpTaskCnt;  // 组内任务数量
uint32_t grpTaskIdxOffset[SIZE_MAX_AIC_NUM];  // 组内任务在ResultWS的块偏移
uint32_t baseTaskCnt;  // 任务块数量
uint32_t baseBTCnt;  // 行方向任务块数量
uint32_t baseVCnt;  // 列方向任务块数量
uint32_t baseTaskEpochCnt;  // 任务轮数
SlidingWindowBlockIterator iterator1;  // C1 / V1
SlidingWindowBlockIterator iterator2;  // C2 / V2
// 全局常量
const __gm__ FusedLinearCrossEntropyLossGradMemFriendlyTilingData *tdp;
const uint32_t blockIdx = GetBlockIdx();
const bool isVec0 = blockIdx % SIZE_2 == 0;
const uint16_t C1ToV1FlagId = 0;  // fix
const uint16_t V1ToC2FlagId = 1;  // mte3
const uint16_t C2ToV2FlagId = 2;  // fix
const uint16_t VecMTE3FlagId = 3;  // mte3
const uint16_t VecMTE2FlagId = 4;  // mte3
};

}  // namespace FusedLinearCrossEntropyLossGradNS

#endif  // FUSED_LINEAR_CROSS_ENTROPY_LOSS_GRAD_MEM_FRIENDLY_H