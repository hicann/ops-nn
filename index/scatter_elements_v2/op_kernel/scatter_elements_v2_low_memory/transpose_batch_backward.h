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
 * \file transpose_batch_backward.h
 * \brief 完成srcGm -> dstGm的transpose与cast功能，支持以下几种类型的转换
 * T: float U: float
 * T: float U: half
 * T: float U: bfloat16_t
 * T: int32_t U: int32_t
 * T: half U: uint8
 * T: half U: int8
 * T: half U: bool
 */

#ifndef TRANSPOSE_BATCH_BACKWARD_H
#define TRANSPOSE_BATCH_BACKWARD_H
#include "transpose_tile_forward.h"
#include "transpose_batch_base.h"

namespace ScatterElementsV2NS {
using namespace AscendC;
using namespace std;

template <typename T, typename U>
class TransposeBatchBackward : public TransposeBatchBase<T, U> {
public:
    __aicore__ inline TransposeBatchBackward() {}

    __aicore__ inline void Init(GlobalTensor<T>& srcGm, GlobalTensor<U>& dstGm, LocalTensor<uint8_t>& allUbLocal) {
        this->srcGm = srcGm;
        this->dstGm = dstGm;
        this->allUbLocal = allUbLocal;
        // 分配ub空间
        this->srcUbLocal = this->allUbLocal.template ReinterpretCast<T>(); // 在有需要时原地cast
        this->dstUbLocal = this->allUbLocal[CACHE_CAPACITY * sizeof(T)].template ReinterpretCast<T>();
    }

    TRANSPOSE_BATCH_PROCESS()
    TRANSPOSE_BATCH_SET_SHAPE()
    TRANSPOSE_BATCH_SET_CORE_ID()
    TRANSPOSE_BATCH_SET_CORE_NUMS()

private:
    TRANSPOSE_BATCH_PROCESS_BY_COLUMNS()
    TRANSPOSE_BATCH_PROCESS_BY_ROWS()
    TRANSPOSE_BATCH_GET_NEXT_CORE()
    TRANSPOSE_BATCH_PROCESS_TASK()

    __aicore__ inline void DoTranspose() {
        if constexpr (std::is_same<T, float>::value || std::is_same<T, int32_t>::value) {
            LocalTensor<float> dstLocal = this->dstUbLocal.template ReinterpretCast<float>();
            LocalTensor<float> srcLocal = this->srcUbLocal.template ReinterpretCast<float>();
            TransposeFloat(dstLocal, srcLocal, BASE_TILE_SIZE, BASE_TILE_SIZE);
        } else if constexpr (std::is_same<T, half>::value) {
            LocalTensor<half> dstLocal = this->dstUbLocal.template ReinterpretCast<half>();
            LocalTensor<half> srcLocal = this->srcUbLocal.template ReinterpretCast<half>();
            TransposeHalf(dstLocal, srcLocal, BASE_TILE_SIZE, BASE_TILE_SIZE);
        }
    }

    __aicore__ inline void StoreDataToGm(uint32_t dstOffset, uint32_t rowSize, uint32_t colSize) {
        auto dstUbLocalU = this->dstUbLocal.template ReinterpretCast<U>();
        if constexpr (std::is_same<U, half>::value || std::is_same<U, bfloat16_t>::value) {
            Cast(dstUbLocalU, this->dstUbLocal, RoundMode::CAST_RINT, colSize * BASE_TILE_SIZE);
            PIPE_V_S();
        }
        if constexpr (std::is_same<U, uint8_t>::value || std::is_same<U, int8_t>::value) {
            Cast(dstUbLocalU, this->dstUbLocal, RoundMode::CAST_NONE, CACHE_CAPACITY);
            PIPE_V_S();
        }
        if constexpr (std::is_same<U, bool>::value) {
            // 非0值需要先归1
            // 1. dstUbLocal与0比较大小，结果存入maskLocal
            // 2. dstUbLocal置1
            // 3. dstUbLocal根据maskLocal赋值为0
            // 4. dstUbLocal转为bool类型
            auto compareMaskLocal = this->srcUbLocal.template ReinterpretCast<uint8_t>();
            CompareScalar(compareMaskLocal, this->dstUbLocal, static_cast<half>(1), CMPMODE::LT, CACHE_CAPACITY);
            PipeBarrier<PIPE_V>();
            Duplicate(this->dstUbLocal, static_cast<half>(0), CACHE_CAPACITY);
            PipeBarrier<PIPE_V>();
            Select(this->dstUbLocal, compareMaskLocal, this->dstUbLocal, static_cast<half>(1), SELMODE::VSEL_TENSOR_SCALAR_MODE, CACHE_CAPACITY);
            PipeBarrier<PIPE_V>();
            Cast(this->dstUbLocal.template ReinterpretCast<uint8_t>(), this->dstUbLocal, RoundMode::CAST_NONE, CACHE_CAPACITY);
            PIPE_V_S();
        }

        uint32_t aliged = BYTE_ALIGNMENT / sizeof(U);
        uint32_t rowSizeAligned = (rowSize + aliged - 1) / aliged * aliged;
        uint32_t srcStride = (BASE_TILE_SIZE - rowSizeAligned) / aliged;
        DataCopyExtParams dstCopyParams{
            static_cast<uint16_t>(colSize),
            static_cast<uint32_t>(rowSize * sizeof(U)),
            srcStride, // 源数据前一块数据的尾，到下一块数据的头的距离，单位为32字节
            static_cast<uint32_t>(this->rows * sizeof(U) - rowSize * sizeof(U)), // 目的数据前一块数据的尾，到下一块数据的头的距离，单位为字节
            0
        };
        DataCopyPad(this->dstGm[dstOffset], dstUbLocalU, dstCopyParams);
    }

    __aicore__ inline void LoadDataToUb(uint32_t srcOffset, uint32_t rowSize, uint32_t colSize) {
        uint32_t aliged = BYTE_ALIGNMENT / sizeof(T);
        uint32_t colSizeAligned = (colSize + aliged - 1) / aliged * aliged;
        uint32_t dstGmStride = (BASE_TILE_SIZE - colSizeAligned) / aliged;
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(rowSize),
            static_cast<uint32_t>(colSize * sizeof(T)),
            static_cast<uint32_t>(this->cols * sizeof(T) - colSize * sizeof(T)), // 源数据上前一块数据的尾，到下一块数据的头的字节数
            dstGmStride, // 目的数据前一块数据的尾，到下一块数据的头的距离，单位为32字节
            0
        };
        DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
        DataCopyPad(this->srcUbLocal[0], this->srcGm[srcOffset], copyParams, padParams);
    }

private:
    LocalTensor<T> srcUbLocal;
    LocalTensor<T> dstUbLocal;
};
}
#endif
