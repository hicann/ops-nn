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
 * \file scatter_elements_v2_cache_scatter.h
 * \brief
 */

#ifndef SCATTER_ELEMENTS_CACHE_SCATTER_H
#define SCATTER_ELEMENTS_CACHE_SCATTER_H
#include "scatter_elements_v2_cache_base.h"
#include "scatter_elements_v2_cache_scatter_add.h"
namespace ScatterElementsV2NS {
using namespace AscendC;

template <typename T, typename U>
class CacheXScatter : public CacheXScatterAddBase<T, U> {
public:
    __aicore__ inline void Process() {
        this->DoScatter();
    }

private:
    __aicore__ inline void DoScatter() {
        this->GetCoreTasks();
        if (this->needTranspose) {
            this->DoColsScatter();
        } else {
            this->DoRowsScatter();
        }
    }

    __aicore__ inline void DoColsScatter() {
        this->GetColsXAggParams();
        for (uint64_t i = 0; i <= this->aggTimes; i++) {
            if (i == this->aggTimes && this->aggLeftTasks == 0) {
                break;
            }
            uint64_t nowTasks = (i == this->aggTimes ? this->aggLeftTasks : this->aggTasks);
            uint64_t startTask = this->coreStartTask + i * this->aggTasks;
            this->CopyInXCols(startTask, nowTasks);
            for (uint64_t j = 0; j < this->indicesDim0; j++) {
                // 搬入一行indices及其对应的updates
                uint64_t mte2Start = j * this->indicesDim1 + startTask;
                this->CopyInIndices(mte2Start, nowTasks);
                mte2Start = j * this->updatesDim1 + startTask;
                this->CopyInupdates(mte2Start, nowTasks);
                this->PIPE_MTE2_S();
                this->ColsScatterOneByOne(nowTasks);
                this->PIPE_S_MTE2(); // 搬运等待SetValue
            }
            this->PIPE_S_MTE3();
            this->CopyOutXCols(startTask, nowTasks);
        }
    }

    __aicore__ inline void ColsScatterOneByOne(uint64_t tasks) {
        uint64_t aligned = BLOCK_SIZE / sizeof(T);
        uint64_t strides = ((tasks + aligned - 1) / aligned) * aligned;
        for (uint64_t i = 0; i < tasks; i++) {
            int32_t indexValue = this->indicesLocalTensor.GetValue(i);
            T value = this->updatesLocalTensor.GetValue(i);
            int32_t dst = indexValue * strides + i;
            this->xLocalTensor.SetValue(dst, value);
        }
    }

    __aicore__ inline void DoRowsScatter() {
        this->GetRowsXAggParams();
        for (uint64_t i = 0; i <= this->aggTimes; i++) {
            if (i == this->aggTimes && this->aggLeftTasks == 0) {
                break;
            }
            uint64_t tasks = (i == this->aggTimes ? this->aggLeftTasks : this->aggTasks);
            uint64_t startTask = this->coreStartTask + i * this->aggTasks;
            this->CopyInXRows(startTask, tasks);
            if (this->indicesDim1 > INDICES_LOCAL_LENGTH) {
                this->DoRowsScatterSliceIndices(startTask, tasks);
            } else {
                this->DoRowsScatterAggIndices(startTask, tasks);
            }
            this->PIPE_S_MTE3();
            this->CopyOutXRows(startTask, tasks);
        }
    }

    __aicore__ inline void DoRowsScatterSliceIndices(uint64_t startTask, uint64_t tasks) {
        // 行内切块
        uint64_t sliceBlockNums = this->indicesDim1 / INDICES_LOCAL_LENGTH;
        uint64_t leftIndicesNums = this->indicesDim1 - sliceBlockNums * INDICES_LOCAL_LENGTH;
        for (uint64_t i = 0; i <= sliceBlockNums; i++) {
            if (i == sliceBlockNums && leftIndicesNums == 0) {
                break;
            }
            uint64_t nums = (i == sliceBlockNums ? leftIndicesNums : INDICES_LOCAL_LENGTH);
            // 处理tasks
            for (uint64_t j = 0; j < tasks; j++) {
                uint64_t mteStart = (startTask + j) * this->indicesDim1 + i * INDICES_LOCAL_LENGTH;
                this->CopyInIndices(mteStart, nums);
                mteStart = (startTask + j) * this->updatesDim1 + i * INDICES_LOCAL_LENGTH;
                this->CopyInupdates(mteStart, nums);
                this->PIPE_MTE2_S();
                // 处理每个索引
                this->DoRowsScatterSliceOneByOne(nums, j);
                this->PIPE_S_MTE2();
            }
        }
    }

    __aicore__ inline void DoRowsScatterSliceOneByOne(uint64_t indicesNums, uint64_t taskId) {
        auto dstBase = taskId * this->xDim1;
        for (uint64_t i = 0; i < indicesNums; i++) {
            int32_t indexValue = this->indicesLocalTensor.GetValue(i);
            T value = this->updatesLocalTensor.GetValue(i);
            uint32_t dst = dstBase + indexValue;
            this->xLocalTensor.SetValue(dst, value);
        }
    }

    __aicore__ inline void DoRowsScatterAggIndices(uint64_t startTask, uint64_t tasks) {
        uint64_t mteNums = 0; // 一次搬入多少行
        uint64_t rowMteMode = 0;
        this->GetRowsIndicesAggParams(mteNums, rowMteMode);

        // 对tasks做分块，一次搬入mteNums行。
        uint64_t mteTimes = tasks / mteNums;
        uint64_t leftNums = tasks - mteTimes * mteNums;
        for (uint64_t i = 0; i <= mteTimes; i++) {
            if (i == mteTimes && leftNums == 0) {
                break;
            }
            uint64_t taskNums = (i == mteTimes ? leftNums : mteNums);
            uint64_t mteStart = (startTask + i * mteNums) * this->indicesDim1;
            this->CopyInIndices(mteStart, taskNums * this->indicesDim1);
            mteStart = (startTask + i * mteNums) * this->updatesDim1;
            this->CopyInupdatesRows(mteStart, taskNums, rowMteMode);
            this->PIPE_MTE2_S();
            // 处理搬入的索引行
            this->ScatterRowsInUb(taskNums, rowMteMode, i * mteNums);
            this->PIPE_S_MTE2();
        }
    }

    __aicore__ inline void ScatterRowsInUb(uint64_t rowNums, uint64_t rowMteMode, uint64_t xStartRow) {
        // indices updates must be one to one
        this->DoRowsScatterAggOneByOne(rowNums, this->updatesDim1, xStartRow);
    }

    __aicore__ inline void DoRowsScatterAggOneByOne(uint64_t rowNums, uint64_t updatesBlockLengh, uint64_t xStartRow) {
        for (uint64_t i = 0; i < rowNums; i++) {
            auto indexBase = i * this->indicesDim1;
            auto updatesBase = i * updatesBlockLengh;
            auto dstBase = (xStartRow + i) * this->xDim1;
            for (uint64_t j = 0; j < this->indicesDim1; j++) {
                int32_t indexValue = this->indicesLocalTensor.GetValue(indexBase + j);
                T value = this->updatesLocalTensor.GetValue(updatesBase + j);
                uint32_t dst = dstBase + indexValue;
                this->xLocalTensor.SetValue(dst, value);
            }
        }
    }
};

template <typename T, typename U>
class CacheXScatterValue : public CacheXScatterAddBase<T, U> {
public:
    __aicore__ inline void Process() {
        this->DoScatterValue();
    }

private:
    __aicore__ inline void DoScatterValue() {
        this->updatesValue = this->updatesGm.GetValue(0);
        this->GetCoreTasks();
        if (this->needTranspose) {
            this->DoColsScatterValue();
        } else {
            this->DoRowsScatterValue();
        }
    }

    __aicore__ inline void DoColsScatterValue() {
        this->GetColsXAggParams();
        for (uint64_t i = 0; i <= this->aggTimes; i++) {
            if (i == this->aggTimes && this->aggLeftTasks == 0) {
                break;
            }
            uint64_t taskNums = (i == this->aggTimes ? this->aggLeftTasks : this->aggTasks);
            uint64_t startTask = this->coreStartTask + i * this->aggTasks;
            this->CopyInXCols(startTask, taskNums);
            for (uint64_t j = 0; j < this->indicesDim0; j++) {
                // 搬入一行indices及其对应的updates
                uint64_t mteStart = j * this->indicesDim1 + startTask;
                this->CopyInIndices(mteStart, taskNums);
                this->PIPE_MTE2_S();
                this->ColsScatterOneByOne(taskNums);
                this->PIPE_S_MTE2(); // 搬运等待SetValue
            }
            this->PIPE_S_MTE3();
            this->CopyOutXCols(startTask, taskNums);
        }
    }

    __aicore__ inline void ColsScatterOneByOne(uint64_t tasks) {
        uint64_t aligned = BLOCK_SIZE / sizeof(T);
        uint64_t strides = ((tasks + aligned - 1) / aligned) * aligned;
        for (uint64_t i = 0; i < tasks; i++) {
            int32_t indexValue = this->indicesLocalTensor.GetValue(i);
            int32_t dst = indexValue * strides + i;
            this->xLocalTensor.SetValue(dst, this->updatesValue);
        }
    }

    __aicore__ inline void DoRowsScatterValue() {
        this->GetRowsXAggParams();
        for (uint64_t i = 0; i <= this->aggTimes; i++) {
            if (i == this->aggTimes && this->aggLeftTasks == 0) {
                break;
            }
            uint64_t tasks = (i == this->aggTimes ? this->aggLeftTasks : this->aggTasks);
            uint64_t startTask = this->coreStartTask + i * this->aggTasks;
            this->CopyInXRows(startTask, tasks);
            if (this->indicesDim1 > INDICES_LOCAL_LENGTH) {
                this->DoRowsScatterValueSliceIndices(startTask, tasks);
            } else {
                this->DoRowsScatterValueAggIndices(startTask, tasks);
            }
            this->PIPE_S_MTE3();
            this->CopyOutXRows(startTask, tasks);
        }
    }

    __aicore__ inline void DoRowsScatterValueSliceIndices(uint64_t startTask, uint64_t tasks) {
        // 行内切块
        uint64_t sliceBlockNums = this->indicesDim1 / INDICES_LOCAL_LENGTH;
        uint64_t leftIndicesNums = this->indicesDim1 - sliceBlockNums * INDICES_LOCAL_LENGTH;
        for (uint64_t i = 0; i <= sliceBlockNums; i++) {
            if (i == sliceBlockNums && leftIndicesNums == 0) {
                break;
            }
            uint64_t nowNums = (i == sliceBlockNums ? leftIndicesNums : INDICES_LOCAL_LENGTH);
            // 处理tasks
            for (uint64_t j = 0; j < tasks; j++) {
                uint64_t mteStart = (startTask + j) * this->indicesDim1 + i * INDICES_LOCAL_LENGTH;
                this->CopyInIndices(mteStart, nowNums);
                this->PIPE_MTE2_S();
                // 处理每个索引
                this->DoRowsScatterSliceOneByOne(nowNums, j);
                this->PIPE_S_MTE2();
            }
        }
    }

    __aicore__ inline void DoRowsScatterSliceOneByOne(uint64_t indicesNums, uint64_t taskId) {
        auto dstBase = taskId * this->xDim1;
        for (uint64_t i = 0; i < indicesNums; i++) {
            int32_t indexValue = this->indicesLocalTensor.GetValue(i);
            uint32_t dst = dstBase + indexValue;
            this->xLocalTensor.SetValue(dst, this->updatesValue);
        }
    }

    __aicore__ inline void DoRowsScatterValueAggIndices(uint64_t startTask, uint64_t tasks) {
        uint64_t mteNums = 0; // 一次搬入多少行
        uint64_t rowMteMode = 0;
        this->GetRowsIndicesAggParams(mteNums, rowMteMode);

        // 对tasks做分块，一次搬入mteNums行。
        uint64_t mteTimes = tasks / mteNums;
        uint64_t leftNums = tasks - mteTimes * mteNums;
        for (uint64_t i = 0; i <= mteTimes; i++) {
            if (i == mteTimes && leftNums == 0) {
                break;
            }
            uint64_t trueNums = (i == mteTimes ? leftNums : mteNums);
            uint64_t mteStart = (startTask + i * mteNums) * this->indicesDim1;
            this->CopyInIndices(mteStart, trueNums * this->indicesDim1);
            this->PIPE_MTE2_S();
            // 处理搬入的索引行
            this->ScatterRowsInUb(trueNums, rowMteMode, i * mteNums);
            this->PIPE_S_MTE2();
        }
    }

    __aicore__ inline void ScatterRowsInUb(uint64_t rowNums, uint64_t rowMteMode, uint64_t xStartRow) {
        this->DoRowsScatterValueAggOneByOne(rowNums, xStartRow);
    }

    __aicore__ inline void DoRowsScatterValueAggOneByOne(uint64_t rowNums, uint64_t xStartRow) {
        for (uint64_t i = 0; i < rowNums; i++) {
            auto indexBase = i * this->indicesDim1;
            auto dstBase = (xStartRow + i) * this->xDim1;
            for (uint64_t j = 0; j < this->indicesDim1; j++) {
                int32_t indexValue = this->indicesLocalTensor.GetValue(indexBase + j);
                uint32_t dst = dstBase + indexValue;
                this->xLocalTensor.SetValue(dst, this->updatesValue);
            }
        }
    }
private:
    T updatesValue = 0;
};

template <typename T, typename U>
class ScatterElementsTwoDims {
public:
    __aicore__ inline ScatterElementsTwoDims() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR updates,
                                const ScatterElementsV2TilingData* tilingData, TPipe* tPipe) {
        this->needTranspose = tilingData->needTranspose;
        this->mode = tilingData->mode;
        this->coreNums = tilingData->coreNums;
        this->updatesIsScalar = tilingData->updatesIsScalar;
        this->xDim0 = tilingData->xDim0;
        this->xDim1 = tilingData->xDim1;
        this->indicesDim0 = tilingData->indicesDim0;
        this->indicesDim1 = tilingData->indicesDim1;
        this->updatesDim0 = tilingData->updatesDim0;
        this->updatesDim1 = tilingData->updatesDim1;

        this->xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        this->updatesGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(updates));
        this->indicesGm.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(indices));

        TPipe* pipe = tPipe;
        TBuf<TPosition::VECCALC> allUbBuf;
        pipe->InitBuffer(allUbBuf, ALL_UB_SIZE);
        this->allUbLocal = allUbBuf.Get<uint8_t>();
    }

    __aicore__ inline void Process() {
        this->DoScatterElements();
    }
private:
    __aicore__ inline void DoScatterElements() {
        if (this->mode == 0 && this->updatesIsScalar) {
            CacheXScatterValue<T, U> op;
            op.Init(this->xGm, this->indicesGm, this->updatesGm, this->allUbLocal);
            op.SetXInfo(this->xDim0, this->xDim1);
            op.SetIndicesInfo(this->indicesDim0, this->indicesDim1);
            op.SetupdatesInfo(this->updatesDim0, this->updatesDim1);
            op.SetCoreNums(this->coreNums);
            op.SetTranspose(this->needTranspose);
            op.Process();
        }
        if (this->mode == 1 && this->updatesIsScalar) {
            CacheXScatterAddValue<T, U> op;
            op.Init(this->xGm, this->indicesGm, this->updatesGm, this->allUbLocal);
            op.SetXInfo(this->xDim0, this->xDim1);
            op.SetIndicesInfo(this->indicesDim0, this->indicesDim1);
            op.SetupdatesInfo(this->updatesDim0, this->updatesDim1);
            op.SetCoreNums(this->coreNums);
            op.SetTranspose(this->needTranspose);
            op.Process();
        }
        if (this->mode == 0 && !this->updatesIsScalar) {
            CacheXScatter<T, U> op;
            op.Init(this->xGm, this->indicesGm, this->updatesGm, this->allUbLocal);
            op.SetXInfo(this->xDim0, this->xDim1);
            op.SetIndicesInfo(this->indicesDim0, this->indicesDim1);
            op.SetupdatesInfo(this->updatesDim0, this->updatesDim1);
            op.SetCoreNums(this->coreNums);
            op.SetTranspose(this->needTranspose);
            op.Process();
        }
        if (this->mode == 1 && !this->updatesIsScalar) {
            CacheXScatterAdd<T, U> op;
            op.Init(this->xGm, this->indicesGm, this->updatesGm, this->allUbLocal);
            op.SetXInfo(this->xDim0, this->xDim1);
            op.SetIndicesInfo(this->indicesDim0, this->indicesDim1);
            op.SetupdatesInfo(this->updatesDim0, this->updatesDim1);
            op.SetCoreNums(this->coreNums);
            op.SetTranspose(this->needTranspose);
            op.Process();
        }
    }

private:
    LocalTensor<uint8_t> allUbLocal;
    GlobalTensor<U> indicesGm;
    GlobalTensor<T> updatesGm;
    GlobalTensor<T> xGm;
    bool needTranspose = false;
    int32_t mode = 0;
    int32_t coreNums = 0;
    bool updatesIsScalar = false;
    uint64_t xDim0 = 0;
    uint64_t xDim1 = 0;
    uint64_t indicesDim0 = 0;
    uint64_t indicesDim1 = 0;
    uint64_t updatesDim0 = 0;
    uint64_t updatesDim1 = 0;
};
}
#endif