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
using namespace std;

template <typename T, typename U>
class CacheXScatter : public CacheXScatterAddBase<T, U> {
public:
    __aicore__ inline void Process() {
        this->DoScatter();
    }

private:
    __aicore__ inline void DoScatter() {
        this->GetCoreTasks();
        this->DoRowsScatter();
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
            PIPE_S_MTE3();
            this->CopyOutXRows(startTask, tasks);
        }
        PIPE_MTE3_S();
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
                PIPE_MTE2_S();
                // 处理每个索引
                this->DoRowsScatterSliceOneByOne(nums, j);
                PIPE_S_MTE2();
            }
        }
    }

    __aicore__ inline void DoRowsScatterSliceOneByOne(uint64_t indicesNums, uint64_t taskId) {
        auto indicesUbAddress = reinterpret_cast<__ubuf__ U *>(this->indicesLocalTensor.GetPhyAddr(0));
        auto updatesUbAddress = reinterpret_cast<__ubuf__ T *>(this->updatesLocalTensor.GetPhyAddr(0));
        auto xUbAddress = reinterpret_cast<__ubuf__ T *>(this->xLocalTensor.GetPhyAddr(0));
        U indicesCache[8];
        T updatesCache[8];
        
        uint64_t staticTimes = indicesNums / 8;
        
        // 处理8的倍数部分（向量化处理）
        for (uint64_t i = 0; i < staticTimes; i++) {
            uint64_t baseOffset = i * 8;
            LOAD_CACHE_8(indicesCache, indicesUbAddress, baseOffset)
            LOAD_CACHE_8(updatesCache, updatesUbAddress, baseOffset)
            
            for (uint64_t k = 0; k < 8; k++) {
                int32_t indexValue = indicesCache[k];
                T value = updatesCache[k];
                uint32_t dst = taskId * this->xDim1 + indexValue;
                *(xUbAddress + dst) = value;
            }
        }
        
        // 处理剩余部分（少于8个元素）
        for (uint64_t i = staticTimes * 8; i < indicesNums; i++) {
            int32_t indexValue = *(indicesUbAddress + i);
            T value = *(updatesUbAddress + i);
            uint32_t dst = taskId * this->xDim1 + indexValue;
            *(xUbAddress + dst) = value;
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
            PIPE_MTE2_S();
            // 处理搬入的索引行
            this->ScatterRowsInUb(taskNums, rowMteMode, i * mteNums);
            PIPE_S_MTE2();
        }
    }

    __aicore__ inline void ScatterRowsInUb(uint64_t rowNums, uint64_t rowMteMode, uint64_t xStartRow) {
        if (rowMteMode == 0 || rowMteMode == MTE_UPDATES_MODE) {
            // indices updates one to one
            this->DoRowsScatterAggOneByOne(rowNums, this->updatesDim1, xStartRow);
        } else if (rowMteMode == 1) {
            // updates是循环搬运上来的，中间有气泡
            uint64_t aligned = 32 / sizeof(T); // todo: 对齐到32字节，之前是按照8字节对齐，bool有个精度问题，需要验证下是否已经解决
            uint64_t indicesDim1Aligned = ((this->indicesDim1 + aligned - 1) / aligned) * aligned;
            this->DoRowsScatterAggOneByOne(rowNums, indicesDim1Aligned, xStartRow);
        }
    }

    __aicore__ inline void DoRowsScatterAggOneByOne(uint64_t rowNums, uint64_t updatesBlockLengh, uint64_t xStartRow) {
        auto indicesUbAddress = reinterpret_cast<__ubuf__ U *>(this->indicesLocalTensor.GetPhyAddr(0));
        auto updatesUbAddress = reinterpret_cast<__ubuf__ T *>(this->updatesLocalTensor.GetPhyAddr(0));
        auto xUbAddress = reinterpret_cast<__ubuf__ T *>(this->xLocalTensor.GetPhyAddr(0));
        U indicesCache[8];
        T updatesCache[8];
        for (uint64_t i = 0; i < rowNums; i++) {
            auto indexBase = i * this->indicesDim1;
            auto updatesBase = i * updatesBlockLengh;
            auto dstBase = (xStartRow + i) * this->xDim1;
            uint64_t staticTimes = this->indicesDim1 / 8;
            for (uint64_t j = 0; j < staticTimes; j++) {
                uint64_t baseOffset = j * 8;
                LOAD_CACHE_8_WITH_BASE(indicesCache, indicesUbAddress, indexBase, baseOffset)
                LOAD_CACHE_8_WITH_BASE(updatesCache, updatesUbAddress, updatesBase, baseOffset)
                for (uint64_t k = 0; k < 8; k++) {
                    int32_t indexValue = indicesCache[k];
                    T value = updatesCache[k];
                    uint32_t dst = dstBase + indexValue;
                    *(xUbAddress + dst) = value;
                }
            }
            for (uint64_t j = staticTimes * 8; j < this->indicesDim1; j++) {
                int32_t indexValue = *(indicesUbAddress + indexBase + j);
                T value = *(updatesUbAddress + updatesBase + j);
                uint32_t dst = dstBase + indexValue;
                *(xUbAddress + dst) = value;
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
        this->DoRowsScatterValue();
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
            PIPE_S_MTE3();
            this->CopyOutXRows(startTask, tasks);
        }
        PIPE_MTE3_S();
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
                PIPE_MTE2_S();
                // 处理每个索引
                this->DoRowsScatterSliceOneByOne(nowNums, j);
                PIPE_S_MTE2();
            }
        }
    }

    __aicore__ inline void DoRowsScatterSliceOneByOne(uint64_t indicesNums, uint64_t taskId) {
        auto indicesUbAddress = reinterpret_cast<__ubuf__ U *>(this->indicesLocalTensor.GetPhyAddr(0));
        auto xUbAddress = reinterpret_cast<__ubuf__ T *>(this->xLocalTensor.GetPhyAddr(0));
        U indicesCache[8];
        T updateValue = this->updatesValue;
        auto dstBase = taskId * this->xDim1;
        
        uint64_t staticTimes = indicesNums / 8;
        
        // 处理8的倍数部分（向量化处理）
        for (uint64_t i = 0; i < staticTimes; i++) {
            uint64_t baseOffset = i * 8;
            LOAD_CACHE_8(indicesCache, indicesUbAddress, baseOffset)
            
            // 批量设置相同的值
            for (uint64_t k = 0; k < 8; k++) {
                int32_t indexValue = indicesCache[k];
                uint32_t dst = dstBase + indexValue;
                *(xUbAddress + dst) = updateValue;
            }
        }
        
        // 处理剩余部分（少于8个元素）
        for (uint64_t i = staticTimes * 8; i < indicesNums; i++) {
            int32_t indexValue = *(indicesUbAddress + i);
            uint32_t dst = dstBase + indexValue;
            *(xUbAddress + dst) = updateValue;
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
            PIPE_MTE2_S();
            // 处理搬入的索引行
            this->ScatterRowsInUb(trueNums, rowMteMode, i * mteNums);
            PIPE_S_MTE2();
        }
    }

    __aicore__ inline void ScatterRowsInUb(uint64_t rowNums, uint64_t rowMteMode, uint64_t xStartRow) {
        this->DoRowsScatterValueAggOneByOne(rowNums, xStartRow);
    }

    __aicore__ inline void DoRowsScatterValueAggOneByOne(uint64_t rowNums, uint64_t xStartRow) {
        auto indicesUbAddress = reinterpret_cast<__ubuf__ U *>(this->indicesLocalTensor.GetPhyAddr(0));
        auto xUbAddress = reinterpret_cast<__ubuf__ T *>(this->xLocalTensor.GetPhyAddr(0));
        U indicesCache[8];
        T updateValue = this->updatesValue;
        
        for (uint64_t i = 0; i < rowNums; i++) {
            auto indexBase = i * this->indicesDim1;
            auto dstBase = (xStartRow + i) * this->xDim1;
            uint64_t staticTimes = this->indicesDim1 / 8;
            
            // 处理8的倍数部分（向量化处理）
            for (uint64_t j = 0; j < staticTimes; j++) {
                uint64_t baseOffset = j * 8;
                LOAD_CACHE_8_WITH_BASE(indicesCache, indicesUbAddress, indexBase, baseOffset)
                
                // 批量设置相同的值
                for (uint64_t k = 0; k < 8; k++) {
                    int32_t indexValue = indicesCache[k];
                    uint32_t dst = dstBase + indexValue;
                    *(xUbAddress + dst) = updateValue;
                }
            }
            
            // 处理剩余部分（少于8个元素）
            for (uint64_t j = staticTimes * 8; j < this->indicesDim1; j++) {
                int32_t indexValue = *(indicesUbAddress + indexBase + j);
                uint32_t dst = dstBase + indexValue;
                *(xUbAddress + dst) = updateValue;
            }
        }
    }
private:
    T updatesValue = 0;
};
}
#endif