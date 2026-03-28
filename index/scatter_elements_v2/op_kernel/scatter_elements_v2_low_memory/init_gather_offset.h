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
 * \file init_gather_offset.h
 * \brief transpose时搬入搬出转连续使用的是gather重构内存，这里初始化Gather接口所用的offset
 */

#ifndef INIT_GATHER_OFFSET_H_
#define INIT_GATHER_OFFSET_H_
#include "common.h"

namespace ScatterElementsV2NS {
using namespace AscendC;
using namespace std;


template <typename T, const uint32_t MODE=0>
class InitGatherOffset {
public:
    __aicore__ inline InitGatherOffset() {}

    __aicore__ inline void Init(GlobalTensor<int32_t>& offsetGmTensor, LocalTensor<uint8_t>& allUbLocal, uint64_t dimValue, uint64_t dimValueAlign) {
        this->offsetGmTensor = offsetGmTensor;
        this->allUbLocal = allUbLocal;
        this->dimValue = dimValue;
        this->dimValueAlign = dimValueAlign;
    }

    __aicore__ inline void SetCoreNums(int32_t coreNums) {
        this->coreNums = coreNums;
    }

    __aicore__ inline void ProcessAggIndices(uint64_t xDim1) {
        // 对indices进行聚合，多行索引变成一行
        uint32_t coreId = 0;
        for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
            coreId = this->GetNextCore(coreId);
            if (GetBlockIdx() == coreId) {
                auto offsetUbLocal = this->allUbLocal.template ReinterpretCast<int32_t>();
                Duplicate(offsetUbLocal, (int32_t)(xDim1 * i), this->dimValue);
                PIPE_V_S();
                DataCopyExtParams dstCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->dimValue * sizeof(int32_t)), 0, 0, 0};
                DataCopyPad(this->offsetGmTensor[i * this->dimValue], offsetUbLocal, dstCopyParams);
                PIPE_MTE3_S();
            }
        }
    }
    
    __aicore__ inline void ProcessAggUpdates(uint64_t xDim1, uint64_t updatesDim1) {
        // T为updates数据类型
        auto updatesRowLengthUb = updatesDim1;
        if (this->dimValue != updatesDim1 && updatesDim1 > BLOCK_SIZE) {
            // updates循环搬运，每次搬indicesDim1长, 在ub上是对齐的indicesDim1Aligned
            uint64_t aligned = BLOCK_SIZE / sizeof(T);
            updatesRowLengthUb = ((this->dimValue + aligned - 1) / aligned) * aligned;
        }
        uint32_t coreId = 0;
         for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
            coreId = this->GetNextCore(coreId);
            if (GetBlockIdx() == coreId) {
                auto offsetUbLocal = this->allUbLocal.template ReinterpretCast<int32_t>();
                CreateVecIndex(offsetUbLocal, (int32_t)(i * updatesRowLengthUb), this->dimValue);
                PipeBarrier<PIPE_V>();
                if constexpr ((std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) && MODE == 1) {
                    Muls(offsetUbLocal, offsetUbLocal, (int32_t)(sizeof(float)), this->dimValue);
                } else {
                    Muls(offsetUbLocal, offsetUbLocal, (int32_t)(sizeof(T)), this->dimValue);
                }
                PIPE_V_S();
                DataCopyExtParams dstCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(this->dimValue * sizeof(int32_t)), 0, 0, 0};
                DataCopyPad(this->offsetGmTensor[i * this->dimValue], offsetUbLocal, dstCopyParams);
                PIPE_MTE3_S();
            }
        }
    }

    // [32, dimValue] -> [32, dimValueAlign]
    __aicore__ inline void ProcessPad() {
        uint32_t coreId = 0;
        for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
            coreId = this->GetNextCore(coreId);
            if (GetBlockIdx() == coreId) {
                this->Compute(i, true);
                PIPE_V_S();
                this->CopyOut(i, true);
                PIPE_MTE3_S();
            }
        }
    }
    // [32, dimValueAlign] -> [32, dimValue]
    __aicore__ inline void ProcessUnPad() {
        uint32_t coreId = 0;
        for (uint32_t i = 0; i < BLOCK_SIZE; i++) {
            coreId = this->GetNextCore(coreId);
            if (GetBlockIdx() == coreId) {
                this->Compute(i, false);
                PIPE_V_S();
                this->CopyOut(i, false);
                PIPE_MTE3_S();
            }
        }
    }
    
    __aicore__ inline void ProcessTransposeForward() {
        // 转置前[dimValue, weight]，转置后[weight, dimValue]
        uint32_t multiple = BASE_TILE_SIZE / this->dimValueAlign;
        uint32_t weight = BASE_TILE_SIZE * multiple;
        auto offsetUbLocalBase = this->allUbLocal.template ReinterpretCast<int32_t>();
        CreateVecIndex(offsetUbLocalBase, (int32_t)(0), this->dimValueAlign);
        PipeBarrier<PIPE_V>();
        Muls(offsetUbLocalBase, offsetUbLocalBase, (int32_t)(weight), this->dimValueAlign);
        PIPE_V_S();

        auto tasks = weight / TRANSPOSE_TASK_UNIT;
        auto taskLeft = weight % TRANSPOSE_TASK_UNIT;
        uint32_t coreId = 0;
        for (uint32_t i = 0; i <= tasks; i++) {
            if (i == tasks && taskLeft == 0) {
                break;
            }
            coreId = this->GetNextCore(coreId);
            if (GetBlockIdx() == coreId) {
                auto wStart = i * TRANSPOSE_TASK_UNIT;
                auto wNums = (i == tasks ? taskLeft : TRANSPOSE_TASK_UNIT);
                auto offsetUbLocal = this->allUbLocal[CACHE_CAPACITY * sizeof(int32_t)].template ReinterpretCast<int32_t>();
                for (uint32_t j = 0; j < wNums; j++) {
                    Adds(offsetUbLocal[j * this->dimValueAlign], offsetUbLocalBase,(int32_t)(wStart + j), this->dimValueAlign);
                    PipeBarrier<PIPE_V>();
                    Muls(offsetUbLocal[j * this->dimValueAlign], offsetUbLocal[j * this->dimValueAlign], (int32_t)(sizeof(T)), this->dimValueAlign);
                    PipeBarrier<PIPE_V>();
                }
                PIPE_V_S();
                uint32_t srcStride = (this->dimValueAlign - this->dimValue) / (BLOCK_SIZE / sizeof(int32_t));
                DataCopyExtParams dstCopyParams{static_cast<uint16_t>(wNums), static_cast<uint32_t>(dimValue * sizeof(int32_t)), srcStride, 0, 0};
                DataCopyPad(this->offsetGmTensor[wStart * dimValue], offsetUbLocal, dstCopyParams);
                PIPE_MTE3_S();
            }
        }
    }

    __aicore__ inline void ProcessTransposeBackward() {
        // 转置前[hight, dimValue]，转置后[dimValue, hight]
        uint32_t multiple = BASE_TILE_SIZE / this->dimValueAlign;
        uint32_t hight = BASE_TILE_SIZE * multiple;
        auto offsetUbLocalBase = this->allUbLocal.template ReinterpretCast<int32_t>();
        CreateVecIndex(offsetUbLocalBase, (int32_t)(0), hight);
        PipeBarrier<PIPE_V>();
        Muls(offsetUbLocalBase, offsetUbLocalBase, (int32_t)(this->dimValue), hight);
        PIPE_V_S();

        auto tasks = this->dimValue / TRANSPOSE_WEIGHT_UNIT;
        auto taskLeft = this->dimValue % TRANSPOSE_WEIGHT_UNIT;
        uint32_t coreId = 0;
        for (uint32_t i = 0; i <= tasks; i++) {
            if (i == tasks && taskLeft == 0) {
                break;
            }
            coreId = this->GetNextCore(coreId);
            if (GetBlockIdx() == coreId) {
                auto hStart = i * TRANSPOSE_WEIGHT_UNIT;
                auto hNums = (i == tasks ? taskLeft : TRANSPOSE_WEIGHT_UNIT);
                auto offsetUbLocal = this->allUbLocal[CACHE_CAPACITY * sizeof(int32_t)].template ReinterpretCast<int32_t>();
                for (uint32_t j = 0; j < hNums; j++) {
                    Adds(offsetUbLocal[j * hight], offsetUbLocalBase,(int32_t)(hStart + j), hight);
                    PipeBarrier<PIPE_V>();
                    Muls(offsetUbLocal[j * hight], offsetUbLocal[j * hight], (int32_t)(sizeof(T)), hight);
                    PipeBarrier<PIPE_V>();
                }
                PIPE_V_S();
                DataCopyExtParams dstCopyParams{static_cast<uint16_t>(hNums), static_cast<uint32_t>(hight * sizeof(int32_t)), 0, 0, 0};
                DataCopyPad(this->offsetGmTensor[hStart * hight], offsetUbLocal, dstCopyParams);
                PIPE_MTE3_S();
            }
        }
    }

private:
    __aicore__ inline void Compute(uint32_t batch, bool isPad) {
        auto offsetUbLocal = this->allUbLocal.template ReinterpretCast<int32_t>();
        if (isPad) {
            CreateVecIndex(offsetUbLocal, (int32_t)(batch * dimValue), dimValueAlign);
        } else {
            CreateVecIndex(offsetUbLocal, (int32_t)(batch * dimValueAlign), dimValueAlign);
        }
        PipeBarrier<PIPE_V>();
        Muls(offsetUbLocal, offsetUbLocal, (int32_t)(sizeof(T)), dimValueAlign);
    }

    __aicore__ inline void CopyOut(uint32_t batch, bool isPad) {
        auto offsetUbLocal = this->allUbLocal.template ReinterpretCast<int32_t>();
        if (isPad) {
            // [32, dimValue] -> [32, dimValueAlign]
            DataCopyExtParams dstCopyParams{1, static_cast<uint32_t>(dimValueAlign * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(this->offsetGmTensor[batch * dimValueAlign], offsetUbLocal, dstCopyParams);
        } else {
            // [32, dimValueAlign] -> [32, dimValue]
            DataCopyExtParams dstCopyParams{1, static_cast<uint32_t>(dimValue * sizeof(int32_t)), 0, 0, 0};
            DataCopyPad(this->offsetGmTensor[batch * dimValue], offsetUbLocal, dstCopyParams);
        }
    }

    __aicore__ inline uint32_t GetNextCore(uint32_t coreId) {
        coreId += 1;
        if (coreId == this->coreNums) {
            coreId = 0;
        }
        return coreId;
    }

    LocalTensor<uint8_t> allUbLocal;
    GlobalTensor<int32_t> offsetGmTensor;
    int32_t coreNums = 0;
    uint64_t dimValue = 0;
    uint64_t dimValueAlign = 0;
};
}
#endif