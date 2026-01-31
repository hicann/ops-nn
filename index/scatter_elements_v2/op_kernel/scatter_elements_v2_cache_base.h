/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_elements_v2_cache_base.h
 * \brief
 */

#ifndef SCATTER_ELEMENTS_CACHE_BASE_H
#define SCATTER_ELEMENTS_CACHE_BASE_H
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace ScatterElementsV2NS {
using namespace AscendC;
constexpr uint64_t X_LOCAL_LENGTH = 40000;
constexpr uint64_t INDICES_LOCAL_LENGTH = 2048; // index is int32
constexpr uint64_t UINT_32_MAX = 4294967295;
constexpr uint64_t UINT_16_MAX = 65535;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint64_t ALL_UB_SIZE = 196000;
constexpr uint64_t MTE_UPDATES_MODE = 2; // updates dimvalue超过indices，但仍批量搬运

template <typename T, typename U>
class CacheXScatterAddBase {
public:
    __aicore__ inline CacheXScatterAddBase() {}

    __aicore__ inline void Init(GlobalTensor<T>& x, GlobalTensor<U>& indices,
                                GlobalTensor<T>& updates, LocalTensor<uint8_t>& allUbLocal) {
        this->xGm = x;
        this->indicesGm = indices;
        this->updatesGm = updates;
        // ub内存分配
        this->xLocalTensor = allUbLocal.ReinterpretCast<T>();
        this->indicesLocalTensor = allUbLocal[X_LOCAL_LENGTH * sizeof(T)].ReinterpretCast<U>();
        this->updatesLocalTensor = allUbLocal[X_LOCAL_LENGTH * sizeof(T) + INDICES_LOCAL_LENGTH * sizeof(U)].ReinterpretCast<T>();
    }

    __aicore__ inline void SetXInfo(uint64_t xDim0, uint64_t xDim1) {
        this->xDim0 = xDim0;
        this->xDim1 = xDim1;
    }

    __aicore__ inline void SetIndicesInfo(uint64_t indicesDim0, uint64_t indicesDim1) {
        this->indicesDim0 = indicesDim0;
        this->indicesDim1 = indicesDim1;
    }

    __aicore__ inline void SetupdatesInfo(uint64_t updatesDim0, uint64_t updatesDim1) {
        this->updatesDim0 = updatesDim0;
        this->updatesDim1 = updatesDim1;
    }

    __aicore__ inline void SetCoreNums(int32_t coreNums) {
        this->coreNums = coreNums;
    }

    __aicore__ inline void SetTranspose(bool needTranspose) {
        this->needTranspose = needTranspose;
    }

protected:
    __aicore__ inline void GetCoreTasks() {
        uint32_t frontCoreNum = 0;
        uint32_t tailCoreNum = 0;
        uint32_t frontCoreTaskNum = 0;
        uint32_t tailCoreTaskNum = 0;
        uint32_t tasks = this->needTranspose ? this->indicesDim1 : this->indicesDim0; // 需要transpose时任务数为列数。
        tailCoreTaskNum = tasks / this->coreNums;

        if (tailCoreTaskNum == 0) {
            frontCoreTaskNum = 1;
            frontCoreNum = tasks;
            tailCoreNum = 0;
        } else if (tasks % this->coreNums == 0) {
            frontCoreTaskNum = tailCoreTaskNum;
            frontCoreNum = this->coreNums;
            tailCoreTaskNum = 0;
            tailCoreNum = 0;
        } else {
            frontCoreTaskNum = tailCoreTaskNum + 1;
            frontCoreNum = tasks - tailCoreTaskNum * this->coreNums;
            tailCoreNum = this->coreNums - frontCoreNum;
        }

        if (GetBlockIdx() < frontCoreNum) {
            this->coreTasks = frontCoreTaskNum;
            this->coreStartTask = GetBlockIdx() * frontCoreTaskNum;
        } else {
            this->coreTasks = tailCoreTaskNum;
            this->coreStartTask = frontCoreNum * frontCoreTaskNum + (GetBlockIdx() - frontCoreNum) * tailCoreTaskNum;
        }
    }

    __aicore__ inline void CopyInupdatesRows(uint64_t mteStart, uint64_t nums, uint64_t rowMteMode) {
        if (rowMteMode == 0) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(nums * this->updatesDim1 * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
            DataCopyPad(this->updatesLocalTensor, this->updatesGm[mteStart], copyParams, padParams);
        } else if (rowMteMode == 1) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->indicesDim1 * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
            uint64_t aligned = BLOCK_SIZE / sizeof(T);
            uint64_t indicesDim1Aligned = ((this->indicesDim1 + aligned - 1) / aligned) * aligned;
            for (uint64_t i = 0; i < nums; i++) {
                DataCopyPad(this->updatesLocalTensor[i * indicesDim1Aligned], this->updatesGm[mteStart + i * this->updatesDim1], copyParams, padParams);
            }
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(nums * this->updatesDim1 * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
            DataCopyPad(this->updatesLocalTensor, this->updatesGm[mteStart], copyParams, padParams);
        }
    }

    __aicore__ inline void CopyInXCols(uint64_t startTask, uint64_t tasks) {
        if (this->xDim1 * sizeof(T) < UINT_32_MAX && this->xDim0 < UINT_16_MAX) { // 可跳搬
            DataCopyExtParams copyParams{static_cast<uint16_t>(this->xDim0), static_cast<uint32_t>(tasks * sizeof(T)),
                                        static_cast<uint32_t>(this->xDim1 * sizeof(T) - tasks * sizeof(T)), 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
            DataCopyPad(this->xLocalTensor, this->xGm[startTask], copyParams, padParams);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(tasks * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
            uint64_t aligned = BLOCK_SIZE / sizeof(T);
            uint64_t tasksAligned = ((tasks + aligned - 1) / aligned) * aligned;
            if (tasks == this->aggTasks) {
                tasksAligned = tasks;
            }

            for (uint64_t i = 0; i < this->xDim0; i++) {
                uint64_t src = i * this->xDim1 + startTask;
                uint64_t dst = i * tasksAligned;
                DataCopyPad(this->xLocalTensor[dst], this->xGm[src], copyParams, padParams);
            }
        }
        this->PIPE_MTE2_S();
    }

    __aicore__ inline void CopyInXRows(uint64_t startTask, uint64_t tasks) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tasks * this->xDim1 * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        uint64_t src = startTask * this->xDim1;
        DataCopyPad(this->xLocalTensor, this->xGm[src], copyParams, padParams);
        this->PIPE_MTE2_S();
    }

    __aicore__ inline void CopyOutXCols(uint64_t startTask, uint64_t tasks) {
        if (this->xDim1 * sizeof(T) < UINT_32_MAX && this->xDim0 < UINT_16_MAX) { // 可跳搬
            DataCopyExtParams copyParams{static_cast<uint16_t>(this->xDim0), static_cast<uint32_t>(tasks * sizeof(T)),
                                        0, static_cast<uint32_t>(this->xDim1 * sizeof(T) - tasks * sizeof(T)), 0};
            DataCopyPad(this->xGm[startTask], this->xLocalTensor, copyParams);
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(tasks * sizeof(T)), 0, 0, 0};
            uint64_t aligned = BLOCK_SIZE / sizeof(T);
            uint64_t tasksAligned = ((tasks + aligned - 1) / aligned) * aligned;
            if (tasks == this->aggTasks) {
                tasksAligned = tasks;
            }

            for (uint64_t i = 0; i < this->xDim0; i++) {
                uint64_t dst = i * this->xDim1 + startTask;
                uint64_t src = i * tasksAligned;
                DataCopyPad(this->xGm[dst], this->xLocalTensor[src], copyParams);
            }
        }
        this->PIPE_MTE3_S();
    }

    __aicore__ inline void CopyOutXRows(uint64_t startTask, uint64_t tasks) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tasks * this->xDim1 * sizeof(T)), 0, 0, 0};
        uint64_t dst = startTask * this->xDim1;
        DataCopyPad(this->xGm[dst], this->xLocalTensor, copyParams);
        this->PIPE_MTE3_S();
    }

    __aicore__ inline void CopyInIndices(uint64_t mteStart, uint64_t mteNums) { // 入参单位都是元素
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(mteNums * sizeof(U)), 0, 0, 0};
        DataCopyPadExtParams<U> padParams{true, 0, 0, 0};
        DataCopyPad(this->indicesLocalTensor, this->indicesGm[mteStart], copyParams, padParams);
    }

    __aicore__ inline void CopyInupdates(uint64_t mteStart, uint16_t mteNums) { // 入参单位都是元素
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(mteNums * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        DataCopyPad(this->updatesLocalTensor, this->updatesGm[mteStart], copyParams, padParams);
    }

    __aicore__ inline void PIPE_V_S() {
        event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIDVToS);
        WaitFlag<HardEvent::V_S>(eventIDVToS);
    }

    __aicore__ inline void PIPE_MTE2_S() {
        event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
    }

    __aicore__ inline void PIPE_MTE3_S() {
        event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
        WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
    }

    __aicore__ inline void PIPE_S_MTE3() {
        event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    }

    __aicore__ inline void PIPE_S_MTE2() {
        event_t eventIDSToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
        WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
    }

    __aicore__ inline void GetColsXAggParams() {
        // 由xDim0，决定一次处理多少任务。任务数由索引决定。
        uint64_t aligned = BLOCK_SIZE / sizeof(T);
        this->aggTasks = ((X_LOCAL_LENGTH / this->xDim0) / aligned) * aligned; // task要向下对齐，因为要作为1块搬进来。向上对齐会超X_LOCAL_LENGTH
        this->aggTasks = (this->aggTasks > INDICES_LOCAL_LENGTH ? INDICES_LOCAL_LENGTH : this->aggTasks);
        this->aggTimes = this->coreTasks / this->aggTasks;
        this->aggLeftTasks = this->coreTasks - this->aggTimes * this->aggTasks; // left搬入的时候可以向上对齐
    }

    __aicore__ inline void GetRowsXAggParams() {
        this->aggTasks = X_LOCAL_LENGTH / this->xDim1;
        this->aggTimes = this->coreTasks / this->aggTasks;
        this->aggLeftTasks = this->coreTasks - this->aggTimes * this->aggTasks;
    }

    __aicore__ inline void GetRowsIndicesAggParams(uint64_t& mteNums, uint64_t& rowMteMode) {
        if (this->indicesDim1 == this->updatesDim1) {
            // 都批量搬运
            mteNums = INDICES_LOCAL_LENGTH / this->indicesDim1;
        } else {
            if (this->updatesDim1 > BLOCK_SIZE) {
                // updates循环搬运，每次搬indicesDim1长, 在ub上是对齐的indicesDim1Aligned
                uint64_t aligned = BLOCK_SIZE / sizeof(T);
                uint64_t indicesDim1Aligned = ((this->indicesDim1 + aligned - 1) / aligned) * aligned;
                mteNums = INDICES_LOCAL_LENGTH / indicesDim1Aligned;
                rowMteMode = 1;
            } else {
                // updates批量搬运
                mteNums = INDICES_LOCAL_LENGTH / this->updatesDim1;
                rowMteMode = MTE_UPDATES_MODE;
            }
        }
    }
protected:
    LocalTensor<U> indicesLocalTensor;
    LocalTensor<T> updatesLocalTensor;
    LocalTensor<T> xLocalTensor;

    GlobalTensor<U> indicesGm;
    GlobalTensor<T> updatesGm;
    GlobalTensor<T> xGm;

    int32_t coreNums = 0; // 传入的可用核数
    uint64_t xDim0 = 0; // x.shape[0]
    uint64_t xDim1 = 0; // x.shape[1]
    uint64_t indicesDim0 = 0; // indices.shape[0]
    uint64_t indicesDim1 = 0; // indices.shape[1]
    uint64_t updatesDim0 = 0; // updates.shape[0]
    uint64_t updatesDim1 = 0; // updates.shape[1]
    uint64_t coreStartTask = 0; // 核分到的起始任务，每个任务为1行或1列
    uint64_t coreTasks = 0; // 核分到的行数
    int32_t needTranspose = 0;
    // 索引多行一起搬入时的参数:
    uint64_t aggTasks = 0; // 一次搬入的任务
    uint64_t aggTimes = 0; // coreTasks行，需要聚集多少次
    uint64_t aggLeftTasks = 0; // 剩余的任务行
};
}
#endif