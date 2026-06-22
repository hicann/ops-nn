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
 * \file scatter_elements_cache.h
 * \brief 支持x尾轴长度<40000的尾轴场景的scatterElements操作。支持int32/int64索引，并且支持负数索引。
 *        当x的dtype为fp16/bf16且是累加场景时，会升精度运算。其他场景均按原dtype运算。
 */

#ifndef SCATTER_ELEMENTS_CACHE_H
#define SCATTER_ELEMENTS_CACHE_H
#include "common.h"

namespace ScatterElementsV2NS {
using namespace AscendC;
using namespace std;

/**
 * \brief ScatterElementsCacheOp
 * \tparam T: dtype of x
 * \tparam U: dtype of indices
 * \tparam V: dtype of x in ub
 * \tparam MODE: scatter_elements mode, 0: scatter_elements, 1: scatter_add_elements
 * \tparam IsScalar: whether updates is scalar
 *  MODE = 1
 *  T: half/bfloat16 V: float
 *  T: float32/int32/uint8/int8/bool V: T
 *  MODE = 0
 *  T = V
 */
template <typename T, typename U, typename V, const uint32_t MODE, const bool IsScalar>
class ScatterElementsCacheOp {
public:
    __aicore__ inline ScatterElementsCacheOp() {}

    __aicore__ inline void Init(GlobalTensor<T>& x, GlobalTensor<U>& indices,
                                GlobalTensor<T>& updates, LocalTensor<uint8_t>& allUbLocal, GM_ADDR workspace) {
        this->xGm = x;
        this->indicesGm = indices;
        this->updatesGm = updates;
        this->workspace = workspace;
        // ub内存分配
        this->xLocalTensor = allUbLocal.ReinterpretCast<V>();
        this->indicesLocalTensor = allUbLocal[X_LOCAL_LENGTH * sizeof(V)].ReinterpretCast<U>();
        this->updatesLocalTensor = allUbLocal[X_LOCAL_LENGTH * sizeof(V) + INDICES_LOCAL_LENGTH * sizeof(int64_t)].ReinterpretCast<V>();
        this->aggIndicesOffset = allUbLocal[X_LOCAL_LENGTH * sizeof(V) + INDICES_LOCAL_LENGTH * sizeof(int64_t) +
                                            INDICES_LOCAL_LENGTH * sizeof(int32_t)].template ReinterpretCast<int32_t>();
        this->aggUpdatesOffset = allUbLocal[X_LOCAL_LENGTH * sizeof(V) + INDICES_LOCAL_LENGTH * sizeof(int64_t) + 
                                            INDICES_LOCAL_LENGTH * sizeof(int32_t) + AGG_INDICES_NUM * sizeof(int32_t)].template ReinterpretCast<uint32_t>();
        
    }

    __aicore__ inline void SetXInfo(uint64_t xDim0, uint64_t xDim1) {
        this->xDim0 = xDim0;
        this->xDim1 = xDim1;
    }

    __aicore__ inline void SetIndicesInfo(uint64_t indicesDim0, uint64_t indicesDim1) {
        this->indicesDim0 = indicesDim0;
        this->indicesDim1 = indicesDim1;
    }

    __aicore__ inline void SetUpdatesInfo(uint64_t updatesDim0, uint64_t updatesDim1) {
        this->updatesDim0 = updatesDim0;
        this->updatesDim1 = updatesDim1;
    }

    __aicore__ inline void SetCoreNums(int32_t coreNums) {
        this->coreNums = coreNums;
    }

    __aicore__ inline void Process() {
        if (this->indicesDim1 <= BLOCK_SIZE) {
            this->ReadAggIndicesOffset();
        }
        if constexpr (IsScalar) {
            this->ReadUpdatesValue();
        } else {
            if (this->indicesDim1 <= BLOCK_SIZE && this->updatesDim1 != this->indicesDim1) {
                this->ReadAggUpdatesOffset();
            } 
        }
        this->GetCoreTasks();
        this->DoRowsScatterElements();
    }

private:
    __aicore__ inline void ReadAggIndicesOffset() {
        GlobalTensor<int32_t> offsetGmTensor;
        offsetGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(this->workspace));

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(AGG_INDICES_NUM * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
        DataCopyPad(this->aggIndicesOffset, offsetGmTensor, copyParams, padParams);
        PIPE_MTE2_S();
    }

    __aicore__ inline void ReadAggUpdatesOffset() {
        GlobalTensor<uint32_t> offsetGmTensor;
        offsetGmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(this->workspace + AGG_INDICES_NUM * sizeof(int32_t)));

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(AGG_INDICES_NUM * sizeof(uint32_t)), 0, 0, 0};
        DataCopyPadExtParams<uint32_t> padParams{true, 0, 0, 0};
        DataCopyPad(this->aggUpdatesOffset, offsetGmTensor, copyParams, padParams);
        PIPE_MTE2_S();
    }

    __aicore__ inline void ReadUpdatesValue() {
        LocalTensor<T> srcLocal = this->xLocalTensor.template ReinterpretCast<T>();
        LocalTensor<V> dstLocal = this->xLocalTensor[BLOCK_SIZE].template ReinterpretCast<V>();

        DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        DataCopyPad(srcLocal[0], this->updatesGm[0], copyParams, padParams);
        PIPE_MTE2_S();
        if constexpr (std::is_same<T, V>::value) {
            this->updatesValue = srcLocal.GetValue(0);
        } else {
            Cast(dstLocal[0], srcLocal[0], RoundMode::CAST_NONE, 1);
            PIPE_V_S();
            this->updatesValue = dstLocal.GetValue(0);
        }
    }
    /*
     * \brief 按索引行数分核，最终计算出每个核分到的任务数和起始任务
     */
    __aicore__ inline void GetCoreTasks() {
        uint32_t tasks = this->indicesDim0;
        uint32_t baseTaskNum = tasks / this->coreNums;
        uint32_t remainder = tasks % this->coreNums;
        uint32_t blockIdx = GetBlockIdx();
        if (blockIdx < remainder) {
            this->coreTasks = baseTaskNum + 1;
            this->coreStartTask = blockIdx * (baseTaskNum + 1);
        } else {
            this->coreTasks = baseTaskNum;
            this->coreStartTask = remainder * (baseTaskNum + 1) + (blockIdx - remainder) * baseTaskNum;
        }
    }

    __aicore__ inline void CopyInupdatesRows(uint64_t mteStart, uint64_t nums, uint64_t rowMteMode) {
        LocalTensor<T> dstUbLocal;
        if constexpr (std::is_same<T, V>::value) {
            dstUbLocal = this->updatesLocalTensor;
        } else {
            dstUbLocal = this->updatesLocalTensor[INDICES_LOCAL_LENGTH / 2].template ReinterpretCast<T>();
        }

        if (rowMteMode == 1) {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->indicesDim1 * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
            uint64_t aligned = BLOCK_SIZE / sizeof(T);
            uint64_t indicesDim1Aligned = ((this->indicesDim1 + aligned - 1) / aligned) * aligned;
            for (uint64_t i = 0; i < nums; i++) {
                DataCopyPad(dstUbLocal[i * indicesDim1Aligned], this->updatesGm[mteStart + i * this->updatesDim1], copyParams, padParams);
            }
        } else {
            DataCopyExtParams copyParams{1, static_cast<uint32_t>(nums * this->updatesDim1 * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
            DataCopyPad(dstUbLocal, this->updatesGm[mteStart], copyParams, padParams);
        }

        PIPE_MTE2_S();
        if constexpr (!std::is_same<T, V>::value) {
            if (rowMteMode == 1) {
                uint64_t aligned = BLOCK_SIZE / sizeof(T);
                uint64_t indicesDim1Aligned = ((this->indicesDim1 + aligned - 1) / aligned) * aligned;
                Cast(this->updatesLocalTensor, dstUbLocal, RoundMode::CAST_NONE, nums * indicesDim1Aligned);
            } else {
                Cast(this->updatesLocalTensor, dstUbLocal, RoundMode::CAST_NONE, nums * this->updatesDim1);
            }
            PIPE_V_S();
        }
    }

    __aicore__ inline void CopyInXRows(uint64_t startTask, uint64_t tasks) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tasks * this->xDim1 * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        uint64_t src = startTask * this->xDim1;
        LocalTensor<T> dstUbLocal;
        if constexpr (std::is_same<T, V>::value) {
            dstUbLocal = this->xLocalTensor;
        } else {
            dstUbLocal = this->xLocalTensor[X_LOCAL_LENGTH / 2].template ReinterpretCast<T>();
        }
        DataCopyPad(dstUbLocal, this->xGm[src], copyParams, padParams);
        PIPE_MTE2_S();
        if constexpr (!std::is_same<T, V>::value) {
            Cast(this->xLocalTensor, dstUbLocal, RoundMode::CAST_NONE, tasks * this->xDim1);
            PIPE_V_S();
        }
    }

    __aicore__ inline void CopyOutXRows(uint64_t startTask, uint64_t tasks) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(tasks * this->xDim1 * sizeof(T)), 0, 0, 0};
        uint64_t dst = startTask * this->xDim1;
        LocalTensor<T> srcUbLocal;
        if constexpr (std::is_same<T, V>::value) {
            srcUbLocal = this->xLocalTensor;
        } else {
            srcUbLocal = this->xLocalTensor.template ReinterpretCast<T>();
            Cast(srcUbLocal, this->xLocalTensor, RoundMode::CAST_RINT, tasks * this->xDim1);
            PIPE_V_S();
        }

        DataCopyPad(this->xGm[dst], srcUbLocal, copyParams);
        PIPE_MTE3_S();
    }

    __aicore__ inline void CopyInIndices(uint64_t mteStart, uint64_t mteNums) { // 入参单位都是元素
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(mteNums * sizeof(U)), 0, 0, 0};
        DataCopyPadExtParams<U> padParams{true, 0, 0, 0};
        DataCopyPad(this->indicesLocalTensor, this->indicesGm[mteStart], copyParams, padParams);
        // linear indices
        PIPE_MTE2_S();
        LocalTensor<int32_t> indicesInt32;
        if constexpr (std::is_same<U, int64_t>::value) {
            // int32类型索引放在后半部分
            Cast(this->indicesLocalTensor.template ReinterpretCast<int32_t>(), this->indicesLocalTensor, RoundMode::CAST_NONE, mteNums);
            PipeBarrier<PIPE_V>();
            indicesInt32 = this->indicesLocalTensor.template ReinterpretCast<int32_t>()[INDICES_LOCAL_LENGTH];
            auto aligned = BLOCK_SIZE / sizeof(int32_t);
            auto mteNumsAligned = ((mteNums + aligned - 1) / aligned) * aligned;
            DataCopy(indicesInt32, this->indicesLocalTensor.template ReinterpretCast<int32_t>(), mteNumsAligned); // 接口仅支持元素个数对齐
            PipeBarrier<PIPE_V>();
        } else {
            // int32类型索引放在前半部分
            indicesInt32 = this->indicesLocalTensor;
        }
        // 以下代码用于负数索引转正数
        LocalTensor<int32_t> indicesTemp;
        if constexpr (std::is_same<U, int32_t>::value) {
            // temp放在后半部分
            indicesTemp = this->indicesLocalTensor[INDICES_LOCAL_LENGTH];
        } else {
            // temp放在前半部分
            indicesTemp = this->indicesLocalTensor.template ReinterpretCast<int32_t>();
        }
        // 右移31位，负数索引结果为-1，正数索引结果为0
        ShiftRight(indicesTemp, indicesInt32, 31, static_cast<int>(mteNums));
        PipeBarrier<PIPE_V>();
        // 乘以边界值
        Muls(indicesTemp, indicesTemp, static_cast<int>(this->xDim1), static_cast<int>(mteNums));
        PipeBarrier<PIPE_V>();
        // 负数索引加上边界值
        Sub(indicesInt32, indicesInt32, indicesTemp, static_cast<int>(mteNums));
        PipeBarrier<PIPE_V>();
        if constexpr (std::is_same<U, int64_t>::value) {
            Cast(this->indicesLocalTensor, indicesInt32, RoundMode::CAST_NONE, mteNums);
        }
        PIPE_V_S();
    }

    __aicore__ inline void CopyInupdates(uint64_t mteStart, uint16_t mteNums) { // 入参单位都是元素
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(mteNums * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        LocalTensor<T> updatesUbLocal;
        if constexpr (std::is_same<T, V>::value) {
            updatesUbLocal = this->updatesLocalTensor;
        } else {
            updatesUbLocal = this->updatesLocalTensor[INDICES_LOCAL_LENGTH / 2].template ReinterpretCast<T>();
        }

        DataCopyPad(updatesUbLocal, this->updatesGm[mteStart], copyParams, padParams);
        PIPE_MTE2_S();
        if constexpr (!std::is_same<T, V>::value) {
            Cast(this->updatesLocalTensor, updatesUbLocal, RoundMode::CAST_NONE, mteNums);
            PIPE_V_S();
        }
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

    __aicore__ inline void DoRowsScatterElements() {
        this->aggTasks = X_LOCAL_LENGTH / this->xDim1;
        this->aggTimes = this->coreTasks / this->aggTasks;
        this->aggLeftTasks = this->coreTasks - this->aggTimes * this->aggTasks;

        for (uint64_t i = 0; i <= this->aggTimes; i++) {
            if (i == this->aggTimes && this->aggLeftTasks == 0) {
                break;
            }
            uint64_t tasks = (i == this->aggTimes ? this->aggLeftTasks : this->aggTasks);
            uint64_t startTask = this->coreStartTask + i * this->aggTasks;
            this->CopyInXRows(startTask, tasks);
            if (this->indicesDim1 > INDICES_LOCAL_LENGTH) {
                this->DoRowsScatterElementsSliceIndices(startTask, tasks);
            } else {
                this->DoRowsScatterElementsAggIndices(startTask, tasks);
            }
            PIPE_V_S();
            PIPE_S_MTE3();
            PIPE_S_V();
            this->CopyOutXRows(startTask, tasks);
        }
        PIPE_MTE3_S();
    }

    __aicore__ inline void DoRowsScatterElementsSliceIndices(uint64_t startTask, uint64_t tasks) {
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
                if constexpr (!IsScalar) {
                    this->CopyInupdates(mteStart, indicesNums);
                }
                PIPE_MTE2_S();
                // 处理每个索引
                this->DoScatterElementsOneByOne(0, indicesNums, 0, j * this->xDim1);
                PIPE_S_MTE2();
            }
        }
    }

    template <typename DataType>
    __aicore__ inline void LoadCache8(__ubuf__ DataType* ubAddress, uint64_t baseOffset, DataType* cache) {
        uint32_t idx = 0, off = 0;
        cache[idx++] = *(ubAddress + baseOffset + off++);
        cache[idx++] = *(ubAddress + baseOffset + off++);
        cache[idx++] = *(ubAddress + baseOffset + off++);
        cache[idx++] = *(ubAddress + baseOffset + off++);
        cache[idx++] = *(ubAddress + baseOffset + off++);
        cache[idx++] = *(ubAddress + baseOffset + off++);
        cache[idx++] = *(ubAddress + baseOffset + off++);
        cache[idx++] = *(ubAddress + baseOffset + off++);
    }

    __aicore__ inline void DoScatterElementsOneByOne(uint64_t indicesStart, uint64_t indicesNums, uint64_t updatesStart, uint64_t xStart) {
        auto indicesUbAddress = reinterpret_cast<__ubuf__ U *>(this->indicesLocalTensor.GetPhyAddr(indicesStart));
        auto updatesUbAddress = reinterpret_cast<__ubuf__ V *>(this->updatesLocalTensor.GetPhyAddr(updatesStart));
        auto xUbAddress = reinterpret_cast<__ubuf__ V *>(this->xLocalTensor.GetPhyAddr(xStart));
        U indicesCache[LOOP_UNROLL_SIZE];
        V updatesCache[LOOP_UNROLL_SIZE];
        uint64_t staticTimes = indicesNums / LOOP_UNROLL_SIZE;
        for (uint64_t i = 0; i < staticTimes; i++) {
            uint64_t baseOffset = i * LOOP_UNROLL_SIZE;
            this->LoadCache8(indicesUbAddress, baseOffset, indicesCache);
            if constexpr (!IsScalar) {
                this->LoadCache8(updatesUbAddress, baseOffset, updatesCache);
            }
            for (uint64_t k = 0; k < LOOP_UNROLL_SIZE; k++) {
                int32_t indexValue = indicesCache[k];
                V value = this->updatesValue;
                if constexpr (!IsScalar) {
                    value = updatesCache[k];
                }
                if constexpr (MODE == 0) {
                    *(xUbAddress + indexValue) = value;
                } else {
                    *(xUbAddress + indexValue) += value;
                }
            }
        }
        
        for (uint64_t i = staticTimes * LOOP_UNROLL_SIZE; i < indicesNums; i++) {
            int32_t indexValue = *(indicesUbAddress + i);
            V value = this->updatesValue;
            if constexpr (!IsScalar) {
                value = *(updatesUbAddress + i);
            }
            if constexpr (MODE == 0) {
                *(xUbAddress + indexValue) = value;
            } else {
                *(xUbAddress + indexValue) += value;
            }
        }
    }

    __aicore__ inline void DoRowsScatterElementsAggIndices(uint64_t startTask, uint64_t tasks) {
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
            if constexpr (!IsScalar) {
                this->CopyInupdatesRows(mteStart, nums, rowMteMode);
            }
            PIPE_MTE2_S();
            this->ScatterElementsRowsInUb(nums, rowMteMode, i * mteNums);
            PIPE_S_MTE2();
        }
    }

    __aicore__ inline void AggregateIndices(uint64_t indicesStart, uint64_t aggRowNums) {
        if constexpr(std::is_same<U, int32_t>::value) {
            Add(this->indicesLocalTensor[indicesStart], this->indicesLocalTensor[indicesStart], this->aggIndicesOffset, aggRowNums * this->indicesDim1);
        } else {
            auto indicesLocalInt32 = this->indicesLocalTensor[indicesStart].template ReinterpretCast<int32_t>();
            Cast(indicesLocalInt32, this->indicesLocalTensor[indicesStart], RoundMode::CAST_NONE, aggRowNums * this->indicesDim1);
            PipeBarrier<PIPE_V>();
            auto indicesLocalInt32Left = this->indicesLocalTensor[indicesStart + HALF_BYTE_ALIGNMENT * this->indicesDim1].template ReinterpretCast<int32_t>();
            DataCopy(indicesLocalInt32Left, indicesLocalInt32, aggRowNums * this->indicesDim1);
            PipeBarrier<PIPE_V>();
            Add(indicesLocalInt32Left, indicesLocalInt32Left, this->aggIndicesOffset, aggRowNums * this->indicesDim1);
            PipeBarrier<PIPE_V>();
            Cast(this->indicesLocalTensor[indicesStart], indicesLocalInt32Left, RoundMode::CAST_NONE, aggRowNums * this->indicesDim1);
        }
        PIPE_V_S();
    }

    __aicore__ inline void ProcessAggRows(uint64_t rowNums, uint64_t xStartRow, uint64_t updatesStride) {
        constexpr uint64_t aggRowNums = BLOCK_SIZE;
        uint32_t aggTimes = rowNums / aggRowNums;
        for (uint32_t i = 0; i < aggTimes; i++) {
            uint64_t indicesStart = i * aggRowNums * this->indicesDim1;
            this->AggregateIndices(indicesStart, aggRowNums);
            
            if constexpr (!IsScalar) {
                if (this->indicesDim1 != this->updatesDim1) {
                    uint64_t updatesStart = i * aggRowNums * updatesStride;
                    Gather(this->updatesLocalTensor[updatesStart], this->updatesLocalTensor[updatesStart], this->aggUpdatesOffset, 0, aggRowNums * this->indicesDim1);
                    PIPE_V_S();
                }
            }
            this->DoScatterElementsOneByOne(indicesStart, aggRowNums * this->indicesDim1,
                                            i * aggRowNums * updatesStride, (xStartRow + i * aggRowNums) * this->xDim1);
        }
        
        uint32_t aggLeft = rowNums % aggRowNums;
        for (uint64_t i = 0; i < aggLeft; i++) {
            this->DoScatterElementsOneByOne((aggTimes * aggRowNums + i) * this->indicesDim1, this->indicesDim1,
                                            (aggTimes * aggRowNums + i) * updatesStride, (xStartRow + aggTimes * aggRowNums + i) * this->xDim1);
        }
    }

    __aicore__ inline void ProcessSimpleRows(uint64_t rowNums, uint64_t xStartRow, uint64_t updatesStride) {
        for (uint64_t i = 0; i < rowNums; i++) {
            this->DoScatterElementsOneByOne(i * this->indicesDim1, this->indicesDim1,
                                            i * updatesStride, (xStartRow + i) * this->xDim1);
        }
    }

    __aicore__ inline void ScatterElementsRowsInUb(uint64_t rowNums, uint64_t rowMteMode, uint64_t xStartRow) {
        if (rowMteMode == 0 || rowMteMode == MTE_UPDATES_MODE) {
            if constexpr (std::is_same<T, float>::value || std::is_same<T, int32_t>::value ||
                          std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
                if (this->indicesDim1 <= BLOCK_SIZE) {
                    this->ProcessAggRows(rowNums, xStartRow, this->updatesDim1);
                    return;
                }
            }
            this->ProcessSimpleRows(rowNums, xStartRow, this->updatesDim1);
        } else if (rowMteMode == 1) {
            uint64_t aligned = BLOCK_SIZE / sizeof(T);
            uint64_t updatesUbSize = ((this->indicesDim1 + aligned - 1) / aligned) * aligned;

            if constexpr (std::is_same<T, float>::value || std::is_same<T, int32_t>::value ||
                          std::is_same<T, bfloat16_t>::value || std::is_same<T, half>::value) {
                if (this->indicesDim1 <= BLOCK_SIZE) {
                    this->ProcessAggRows(rowNums, xStartRow, updatesUbSize);
                    return;
                }
            }
            this->ProcessSimpleRows(rowNums, xStartRow, updatesUbSize);
        }
    }

private:
    LocalTensor<U> indicesLocalTensor;
    LocalTensor<V> updatesLocalTensor;
    LocalTensor<V> xLocalTensor;
    LocalTensor<int32_t> aggIndicesOffset;
    LocalTensor<uint32_t> aggUpdatesOffset;

    GlobalTensor<U> indicesGm;
    GlobalTensor<T> updatesGm;
    GlobalTensor<T> xGm;
    GM_ADDR workspace;

    int32_t coreNums = 0; // 传入的可用核数
    uint64_t xDim0 = 0; // x.shape[0]
    uint64_t xDim1 = 0; // x.shape[1]
    uint64_t indicesDim0 = 0; // indices.shape[0]
    uint64_t indicesDim1 = 0; // indices.shape[1]
    uint64_t updatesDim0 = 0; // updates.shape[0]
    uint64_t updatesDim1 = 0; // updates.shape[1]
    uint64_t coreStartTask = 0; // 核分到的起始任务，每个任务为1行或1列
    uint64_t coreTasks = 0; // 核分到的行数
    uint64_t aggTasks = 0; // 一次搬入的任务
    uint64_t aggTimes = 0; // coreTasks行，需要聚集多少次
    uint64_t aggLeftTasks = 0; // 剩余的任务行

    V updatesValue = 0;
};
}
#endif