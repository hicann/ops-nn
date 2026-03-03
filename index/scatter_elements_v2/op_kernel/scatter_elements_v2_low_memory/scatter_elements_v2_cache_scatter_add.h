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
 * \file scatter_elements_v2_cache_scatter_add.h
 * \brief
 */

#ifndef SCATTER_ELEMENTS_CACHE_SCATTER_ADD_H
#define SCATTER_ELEMENTS_CACHE_SCATTER_ADD_H
#include "scatter_elements_v2_cache_base.h"

namespace ScatterElementsV2NS {
using namespace AscendC;
using namespace std;

template <typename T, typename U>
class CacheXScatterAdd : public CacheXScatterAddBase<T, U> {
public:
    __aicore__ inline void Process() {
        this->DoScatterAdd();
    }

private:
    __aicore__ inline void DoScatterAdd() {
        this->GetCoreTasks();
        this->DoRowsScatterAdd();
    }

    __aicore__ inline void DoRowsScatterAdd() {
        this->GetRowsXAggParams();
        for (uint64_t i = 0; i <= this->aggTimes; i++) {
            if (i == this->aggTimes && this->aggLeftTasks == 0) {
                break;
            }
            uint64_t tasks = (i == this->aggTimes ? this->aggLeftTasks : this->aggTasks);
            uint64_t startTask = this->coreStartTask + i * this->aggTasks;
            this->CopyInXRows(startTask, tasks);
            if (this->indicesDim1 > INDICES_LOCAL_LENGTH) {
                this->DoRowsScatterAddSliceIndices(startTask, tasks);
            } else {
                this->DoRowsScatterAddAggIndices(startTask, tasks);
            }
            PIPE_V_S();
            PIPE_S_MTE3();
            this->CopyOutXRows(startTask, tasks);
        }
        PIPE_MTE3_S();
    }

    __aicore__ inline void DoRowsScatterAddSliceIndices(uint64_t startTask, uint64_t tasks) {
        // 行内切块
        uint64_t sliceBlockNums = this->indicesDim1 / INDICES_LOCAL_LENGTH;
        uint64_t leftIndicesNums = this->indicesDim1 - sliceBlockNums * INDICES_LOCAL_LENGTH;
        for (uint64_t i = 0; i <= sliceBlockNums; i++) {
            if (i == sliceBlockNums && leftIndicesNums == 0) {
                break;
            }
            uint64_t indicesNums = (i == sliceBlockNums ? leftIndicesNums : INDICES_LOCAL_LENGTH);
            // 处理tasks
            for (uint64_t j = 0; j < tasks; j++) {
                uint64_t mteStart = (startTask + j) * this->indicesDim1 + i * INDICES_LOCAL_LENGTH;
                this->CopyInIndices(mteStart, indicesNums);
                mteStart = (startTask + j) * this->updatesDim1 + i * INDICES_LOCAL_LENGTH;
                this->CopyInupdates(mteStart, indicesNums);
                PIPE_MTE2_S();
                // 处理每个索引
                this->DoRowsScatterAddSliceOneByOne(indicesNums, j);
                PIPE_S_MTE2();
            }
        }
    }

    __aicore__ inline void DoRowsScatterAddSliceOneByOne(uint64_t indicesNums, uint64_t taskId) {
        if constexpr (std::is_same<T, float>::value || std::is_same<T, int32_t>::value || std::is_same<T, half>::value) {
            uint64_t mask[1];
            uint64_t aligned = BLOCK_SIZE / sizeof(T);
            for (uint64_t i = 0; i < indicesNums; i++) {
                int32_t indexValue = this->indicesLocalTensor.GetValue(i);
                T value = this->updatesLocalTensor.GetValue(i);
                uint32_t dst = taskId * this->xDim1 + indexValue;
                int32_t dstAligned = (dst / aligned) * aligned;
                mask[0] = 0b1 << (dst - dstAligned);
                PipeBarrier<PIPE_V>();
                Adds(this->xLocalTensor[dstAligned], this->xLocalTensor[dstAligned], value, mask, 1, {1,1,8,8});
                PipeBarrier<PIPE_V>();
            }
        } else {
            for (uint64_t i = 0; i < indicesNums; i++) {
                int32_t indexValue = this->indicesLocalTensor.GetValue(i);
                T value = this->updatesLocalTensor.GetValue(i);
                uint32_t dst = taskId * this->xDim1 + indexValue;
                value += this->xLocalTensor.GetValue(dst);
                this->xLocalTensor.SetValue(dst, value);
            }
        }
    }

    __aicore__ inline void DoRowsScatterAddAggIndices(uint64_t startTask, uint64_t tasks) {
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
            uint64_t nums = (i == mteTimes ? leftNums : mteNums);
            uint64_t mteStart = (startTask + i * mteNums) * this->indicesDim1;
            this->CopyInIndices(mteStart, nums * this->indicesDim1);
            mteStart = (startTask + i * mteNums) * this->updatesDim1;
            this->CopyInupdatesRows(mteStart, nums, rowMteMode);
            PIPE_MTE2_S();
            // 处理搬入的索引行
            this->ScatterAddRowsInUb(nums, rowMteMode, i * mteNums);
            PIPE_S_MTE2();
        }
    }

    __aicore__ inline void ScatterAddRowsInUb(uint64_t rowNums, uint64_t rowMteMode, uint64_t xStartRow) {
        if (rowMteMode == 0 || rowMteMode == MTE_UPDATES_MODE) {
            // indices updates one to one
            this->DoRowsScatterAddAggOneByOne(rowNums, this->updatesDim1, xStartRow);
        } else if (rowMteMode == 1) {
            // updates是循环搬运上来的，中间有气泡
            uint64_t aligned = 32 / sizeof(T); // todo: 对齐到32字节，之前是按照8字节对齐，bool有个精度问题，需要验证下是否已经解决
            uint64_t indicesDim1Aligned = ((this->indicesDim1 + aligned - 1) / aligned) * aligned;
            this->DoRowsScatterAddAggOneByOne(rowNums, indicesDim1Aligned, xStartRow);
        }
    }

    __aicore__ inline void DoRowsScatterAddAggOneByOne(uint64_t rowNums, uint64_t updatesBlockLengh, uint64_t xStartRow) {
        if constexpr (std::is_same<T, float>::value || std::is_same<T, int32_t>::value || std::is_same<T, half>::value) {
            uint64_t mask[1];
            uint64_t aligned = BLOCK_SIZE / sizeof(T);
            for (uint64_t i = 0; i < rowNums; i++) {
                for (uint64_t j = 0; j < this->indicesDim1; j++) {
                    int32_t indexValue = this->indicesLocalTensor.GetValue(i * this->indicesDim1 + j);
                    T value = this->updatesLocalTensor.GetValue(i * updatesBlockLengh + j);
                    uint32_t dst = (xStartRow + i) * this->xDim1 + indexValue;
                    int32_t dstAligned = (dst / aligned) * aligned;
                    mask[0] = 0b1 << (dst - dstAligned);
                    PipeBarrier<PIPE_V>();
                    Adds(this->xLocalTensor[dstAligned], this->xLocalTensor[dstAligned], value, mask, 1, {1,1,8,8});
                    PipeBarrier<PIPE_V>();
                }
            }
        } else {
            for (uint64_t i = 0; i < rowNums; i++) {
                for (uint64_t j = 0; j < this->indicesDim1; j++) {
                    int32_t indexValue = this->indicesLocalTensor.GetValue(i * this->indicesDim1 + j);
                    T value = this->updatesLocalTensor.GetValue(i * updatesBlockLengh + j);
                    uint32_t dst = (xStartRow + i) * this->xDim1 + indexValue;
                    value += this->xLocalTensor.GetValue(dst);
                    this->xLocalTensor.SetValue(dst, value);
                }
            }
        }
    }
};

template <typename T, typename U>
class CacheXScatterAddValue : public CacheXScatterAddBase<T, U> {
public:
    __aicore__ inline void Process() {
        this->DoScatterAddValue();
    }

private:
    __aicore__ inline void DoScatterAddValue() {
        this->updatesValue = this->updatesGm.GetValue(0);
        this->GetCoreTasks();
        this->DoRowsScatterAddValue();
    }

    __aicore__ inline void DoRowsScatterAddValue() {
        this->GetRowsXAggParams();
        for (uint64_t i = 0; i <= this->aggTimes; i++) {
            if (i == this->aggTimes && this->aggLeftTasks == 0) {
                break;
            }
            uint64_t tasks = (i == this->aggTimes ? this->aggLeftTasks : this->aggTasks);
            uint64_t startTask = this->coreStartTask + i * this->aggTasks;
            this->CopyInXRows(startTask, tasks);
            if (this->indicesDim1 > INDICES_LOCAL_LENGTH) {
                this->DoRowsScatterAddValueSliceIndices(startTask, tasks);
            } else {
                this->DoRowsScatterAddValueAggIndices(startTask, tasks);
            }
            PIPE_V_S();
            PIPE_S_MTE3();
            this->CopyOutXRows(startTask, tasks);
        }
        PIPE_MTE3_S();
    }

    __aicore__ inline void DoRowsScatterAddValueSliceIndices(uint64_t startTask, uint64_t tasks) {
        // 行内切块
        uint64_t sliceBlockNums = this->indicesDim1 / INDICES_LOCAL_LENGTH;
        uint64_t leftIndicesNums = this->indicesDim1 - sliceBlockNums * INDICES_LOCAL_LENGTH;
        for (uint64_t i = 0; i <= sliceBlockNums; i++) {
            if (i == sliceBlockNums && leftIndicesNums == 0) {
                break;
            }
            uint64_t tasksNums = (i == sliceBlockNums ? leftIndicesNums : INDICES_LOCAL_LENGTH);
            // 处理tasks
            for (uint64_t j = 0; j < tasks; j++) {
                uint64_t mteStart = (startTask + j) * this->indicesDim1 + i * INDICES_LOCAL_LENGTH;
                this->CopyInIndices(mteStart, tasksNums);
                PIPE_MTE2_S();
                // 处理每个索引
                this->DoRowsScatterAddValueSliceOneByOne(tasksNums, j);
                PIPE_S_MTE2();
            }
        }
    }

    __aicore__ inline void DoRowsScatterAddValueSliceOneByOne(uint64_t indicesNums, uint64_t taskId) {
        if constexpr (std::is_same<T, float>::value || std::is_same<T, int32_t>::value || std::is_same<T, half>::value) {
            uint64_t mask[1];
            uint64_t aligned = BLOCK_SIZE / sizeof(T);
            for (uint64_t i = 0; i < indicesNums; i++) {
                int32_t indexValue = this->indicesLocalTensor.GetValue(i);
                uint32_t dst = taskId * this->xDim1 + indexValue;
                int32_t dstAligned = (dst / aligned) * aligned;
                mask[0] = 0b1 << (dst - dstAligned);
                PipeBarrier<PIPE_V>();
                Adds(this->xLocalTensor[dstAligned], this->xLocalTensor[dstAligned], this->updatesValue, mask, 1, {1,1,8,8});
                PipeBarrier<PIPE_V>();
            }
        } else {
            for (uint64_t i = 0; i < indicesNums; i++) {
                int32_t indexValue = this->indicesLocalTensor.GetValue(i);
                uint32_t dst = taskId * this->xDim1 + indexValue;
                T value = this->xLocalTensor.GetValue(dst) + this->updatesValue;
                this->xLocalTensor.SetValue(dst, value);
            }
        }
    }

    __aicore__ inline void DoRowsScatterAddValueAggIndices(uint64_t startTask, uint64_t tasks) {
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
            uint64_t nowNums = (i == mteTimes ? leftNums : mteNums);
            uint64_t mteStart = (startTask + i * mteNums) * this->indicesDim1;
            this->CopyInIndices(mteStart, nowNums * this->indicesDim1);
            PIPE_MTE2_S();
            // 处理搬入的索引行
            this->ScatterAddValueRowsInUb(nowNums, rowMteMode, i * mteNums);
            PIPE_S_MTE2();
        }
    }

    __aicore__ inline void ScatterAddValueRowsInUb(uint64_t rowNums, uint64_t rowMteMode, uint64_t xStartRow) {
        if (rowMteMode == 0 || rowMteMode == MTE_UPDATES_MODE) {
            // indices updates one to one
            this->DoRowsScatterAddValueAggOneByOne(rowNums, this->updatesDim1, xStartRow);
        } else if (rowMteMode == 1) {
            // updates是循环搬运上来的，中间有气泡
            uint64_t indicesDim1Aligned = ((this->indicesDim1 + 8 - 1) / 8) * 8;
            this->DoRowsScatterAddValueAggOneByOne(rowNums, indicesDim1Aligned, xStartRow);
        }
    }

    __aicore__ inline void DoRowsScatterAddValueAggOneByOne(uint64_t rowNums, uint64_t updatesBlockLengh, uint64_t xStartRow) {
        if constexpr (std::is_same<T, float>::value || std::is_same<T, int32_t>::value || std::is_same<T, half>::value) {
            uint64_t mask[1];
            uint64_t aligned = BLOCK_SIZE / sizeof(T);
            for (uint64_t i = 0; i < rowNums; i++) {
                for (uint64_t j = 0; j < this->indicesDim1; j++) {
                    int32_t indexValue = this->indicesLocalTensor.GetValue(i * this->indicesDim1 + j);
                    uint32_t dst = (xStartRow + i) * this->xDim1 + indexValue;
                    int32_t dstAligned = (dst / aligned) * aligned;
                    mask[0] = 0b1 << (dst - dstAligned);
                    PipeBarrier<PIPE_V>();
                    Adds(this->xLocalTensor[dstAligned], this->xLocalTensor[dstAligned], this->updatesValue, mask, 1, {1,1,8,8});
                    PipeBarrier<PIPE_V>();
                }
            }
        } else {
            for (uint64_t i = 0; i < rowNums; i++) {
                for (uint64_t j = 0; j < this->indicesDim1; j++) {
                    int32_t indexValue = this->indicesLocalTensor.GetValue(i * this->indicesDim1 + j);
                    uint32_t dst = (xStartRow + i) * this->xDim1 + indexValue;
                    T value = this->xLocalTensor.GetValue(dst) + this->updatesValue;
                    this->xLocalTensor.SetValue(dst, value);
                }
            }
        }
    }
private:
    T updatesValue = 0;
};
}
#endif