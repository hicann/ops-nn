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
 * \file transpose_batch_forward.h
 * \brief 完成transpose与cast功能
 * T: float U: float
 * T: int32_t U: int32_t
 * T: half U: float
 * T: bf16 U: float
 * T: uint8 U: half
 * T: int8 U: half
 * T: bool U: half
 * T: int64 U: int32
 */

#ifndef TRANSPOSE_BATCH_FORWARD_H
#define TRANSPOSE_BATCH_FORWARD_H
#include "transpose_tile_forward.h"
#include "transpose_batch_base.h"

namespace ScatterElementsV2NS {
using namespace AscendC;
using namespace std;

template <typename T, typename U>
class TransposeBatchForward : public TransposeBatchBase<T, U> {
public:
    __aicore__ inline TransposeBatchForward() {}

    __aicore__ inline void Init(GlobalTensor<T>& srcGm, GlobalTensor<U>& dstGm, LocalTensor<uint8_t>& allUbLocal) {
        this->srcGm = srcGm;
        this->dstGm = dstGm;
        this->allUbLocal = allUbLocal;
        // 分配ub空间
        this->srcUbLocal = this->allUbLocal.template ReinterpretCast<U>(); // 在有需要时原地cast
        this->dstUbLocal = this->allUbLocal[CACHE_CAPACITY * sizeof(U)].template ReinterpretCast<U>();
    }

    TRANSPOSE_BATCH_SET_SHAPE()
    TRANSPOSE_BATCH_SET_CORE_ID()
    TRANSPOSE_BATCH_SET_CORE_NUMS()
    TRANSPOSE_BATCH_PROCESS()

private:
    TRANSPOSE_BATCH_PROCESS_BY_COLUMNS()
    TRANSPOSE_BATCH_PROCESS_BY_ROWS()
    TRANSPOSE_BATCH_GET_NEXT_CORE()
    TRANSPOSE_BATCH_PROCESS_TASK()

    __aicore__ inline void LoadDataToUb(uint32_t srcOffset, uint32_t rowSize, uint32_t colSize) {
        
        uint32_t aliged = BYTE_ALIGNMENT / sizeof(T);
        uint32_t colSizeAligned = (colSize + aliged - 1) / aliged * aliged;
        uint32_t dstStride = (BASE_TILE_SIZE - colSizeAligned) / aliged;
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(rowSize),
            static_cast<uint32_t>(colSize * sizeof(T)),
            static_cast<uint32_t>(this->cols * sizeof(T) - colSize * sizeof(T)), // 源数据上前一块数据的尾，到下一块数据的头的字节数
            dstStride, // 目的数据前一块数据的尾，到下一块数据的头的距离，单位为32字节
            0
        };
        if constexpr (std::is_same<T, float>::value || std::is_same<T, int32_t>::value) {
            // T与U相同
            DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
            DataCopyPad(this->srcUbLocal[0], this->srcGm[srcOffset], copyParams, padParams);
        }
        if constexpr (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value ||
                    std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value) {
            // T->U
            DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
            auto srcUbLocalT = this->srcUbLocal.template ReinterpretCast<T>()[CACHE_CAPACITY];
            DataCopyPad(srcUbLocalT, this->srcGm[srcOffset], copyParams, padParams);
            PIPE_MTE2_S();
            Cast(this->srcUbLocal, srcUbLocalT, RoundMode::CAST_NONE, CACHE_CAPACITY);
            PIPE_V_S();
        }
        if constexpr (std::is_same<T, bool>::value) {
            // T->uint8->U
            DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
            auto srcUbLocalT = this->srcUbLocal.template ReinterpretCast<T>()[CACHE_CAPACITY];
            DataCopyPad(srcUbLocalT, this->srcGm[srcOffset], copyParams, padParams);
            PIPE_MTE2_S();
            Cast(this->srcUbLocal, srcUbLocalT.template ReinterpretCast<uint8_t>(), RoundMode::CAST_NONE, CACHE_CAPACITY);
            PIPE_V_S();
        }
        if constexpr (std::is_same<T, int64_t>::value) {
            // T->U
            DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
            auto srcUbLocalT = this->srcUbLocal.template ReinterpretCast<T>();
            DataCopyPad(srcUbLocalT, this->srcGm[srcOffset], copyParams, padParams);
            PIPE_MTE2_S();
            Cast(this->srcUbLocal, srcUbLocalT, RoundMode::CAST_NONE, CACHE_CAPACITY);
            PIPE_V_S();
        }
    }

    __aicore__ inline void StoreDataToGm(uint32_t dstOffset, uint32_t rowSize, uint32_t colSize) {
        uint32_t aliged = BYTE_ALIGNMENT / sizeof(U);
        uint32_t rowSizeAligned = (rowSize + aliged - 1) / aliged * aliged;
        uint32_t srcStrideUb = (BASE_TILE_SIZE - rowSizeAligned) / aliged;
        DataCopyExtParams dstCopyParams{
            static_cast<uint16_t>(colSize),
            static_cast<uint32_t>(rowSize * sizeof(U)),
            srcStrideUb, // 源数据前一块数据的尾，到下一块数据的头的距离，单位为32字节
            static_cast<uint32_t>(this->rows * sizeof(U) - rowSize * sizeof(U)), // 目的数据前一块数据的尾，到下一块数据的头的距离，单位为字节
            0
        };
        DataCopyPad(this->dstGm[dstOffset], this->dstUbLocal[0], dstCopyParams);
    }

    __aicore__ inline void DoTranspose() {
        if constexpr (std::is_same<U, float>::value || std::is_same<U, int32_t>::value) {
            LocalTensor<float> dstUbLocal = this->dstUbLocal.template ReinterpretCast<float>();
            LocalTensor<float> srcUbLocal = this->srcUbLocal.template ReinterpretCast<float>();
            TransposeFloat(dstUbLocal, srcUbLocal, BASE_TILE_SIZE, BASE_TILE_SIZE);
        } else if constexpr (std::is_same<U, half>::value) {
            LocalTensor<half> dstUbLocal = this->dstUbLocal.template ReinterpretCast<half>();
            LocalTensor<half> srcUbLocal = this->srcUbLocal.template ReinterpretCast<half>();
            TransposeHalf(dstUbLocal, srcUbLocal, BASE_TILE_SIZE, BASE_TILE_SIZE);
        }
    }

private:
    LocalTensor<U> srcUbLocal;
    LocalTensor<U> dstUbLocal;
};
}
#endif
