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
#include "common.h"

namespace ScatterElementsV2NS {
using namespace AscendC;
using namespace std;

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

    SCATTER_ELEMENTS_SET_INFO_METHODS()

protected:
    __aicore__ inline void GetCoreTasks() {
        uint32_t frontCoreNum = 0;
        uint32_t tailCoreNum = 0;
        uint32_t frontCoreTaskNum = 0;
        uint32_t tailCoreTaskNum = 0;
        uint32_t tasks = this->indicesDim0; // 任务数为行数。
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

    __aicore__ inline void CopyInXRows(uint64_t startTask, uint64_t tasks) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tasks * this->xDim1 * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        uint64_t src = startTask * this->xDim1;
        DataCopyPad(this->xLocalTensor, this->xGm[src], copyParams, padParams);
        PIPE_MTE2_S();
    }

    __aicore__ inline void CopyOutXRows(uint64_t startTask, uint64_t tasks) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tasks * this->xDim1 * sizeof(T)), 0, 0, 0};
        uint64_t dst = startTask * this->xDim1;
        DataCopyPad(this->xGm[dst], this->xLocalTensor, copyParams);
        PIPE_MTE3_S();
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
    // 索引多行一起搬入时的参数:
    uint64_t aggTasks = 0; // 一次搬入的任务
    uint64_t aggTimes = 0; // coreTasks行，需要聚集多少次
    uint64_t aggLeftTasks = 0; // 剩余的任务行
};

#define LOAD_CACHE_8(cache, address, offset) \
    cache[0] = *(address + offset + 0); \
    cache[1] = *(address + offset + 1); \
    cache[2] = *(address + offset + 2); \
    cache[3] = *(address + offset + 3); \
    cache[4] = *(address + offset + 4); \
    cache[5] = *(address + offset + 5); \
    cache[6] = *(address + offset + 6); \
    cache[7] = *(address + offset + 7);

#define LOAD_CACHE_8_WITH_BASE(cache, address, base, offset) \
    cache[0] = *(address + base + offset + 0); \
    cache[1] = *(address + base + offset + 1); \
    cache[2] = *(address + base + offset + 2); \
    cache[3] = *(address + base + offset + 3); \
    cache[4] = *(address + base + offset + 4); \
    cache[5] = *(address + base + offset + 5); \
    cache[6] = *(address + base + offset + 6); \
    cache[7] = *(address + base + offset + 7);

}
#endif