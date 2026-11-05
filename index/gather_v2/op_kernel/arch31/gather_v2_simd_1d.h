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
 * \file gather_v2_simd_1d.h
 * \brief KirinX90 纯 SIMD 1D 小 shape 模板（替代原 Gatherv2 simt 实现，去除 simt_api 依赖）
 */
#ifndef GATHER_V2_SIMD_1D_H
#define GATHER_V2_SIMD_1D_H

#include "kernel_operator.h"
// 已删除: simt_api/asc_simt.h、basic_api/kernel_vec_intf.h（KirinX90 硬件不支持 simt）

namespace gatherv2 {
using namespace AscendC;

template <typename X_T, typename INDICES_T, typename INDEX_SIZE_T>
class Gatherv2Simd1D {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR axis, GM_ADDR y,
                                const GatherV2TilingData* tilingData)
    {
        tilingData_ = tilingData;
        xGm_.SetGlobalBuffer((__gm__ X_T*)x);
        indicesGm_.SetGlobalBuffer((__gm__ INDICES_T*)indices);
        yGm_.SetGlobalBuffer((__gm__ X_T*)y);
    }

    __aicore__ inline void Process()
    {
        int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());
        int32_t needCoreNum = static_cast<int32_t>(tilingData_->needCoreNum);
        if (blockIdx >= needCoreNum) {
            return;
        }

        INDEX_SIZE_T outerSize = static_cast<INDEX_SIZE_T>(tilingData_->outerSize);
        INDEX_SIZE_T gatherDimSize = static_cast<INDEX_SIZE_T>(tilingData_->gatherDimSize);
        INDEX_SIZE_T gatherSize = static_cast<INDEX_SIZE_T>(tilingData_->gatherSize);
        INDEX_SIZE_T innerSize = static_cast<INDEX_SIZE_T>(tilingData_->innerSize);
        bool nis = static_cast<bool>(tilingData_->negativeIndexSupport);

        INDEX_SIZE_T currentCoreElements = static_cast<INDEX_SIZE_T>(tilingData_->perCoreElements);
        if (blockIdx == tilingData_->needCoreNum - 1) {
            currentCoreElements = static_cast<INDEX_SIZE_T>(tilingData_->lastCoreElements);
        }
        INDEX_SIZE_T yIndexBase = static_cast<INDEX_SIZE_T>(blockIdx) * tilingData_->perCoreElements;

        __gm__ X_T* xPtr = (__gm__ X_T*)xGm_.GetPhyAddr();
        __gm__ INDICES_T* indicesPtr = (__gm__ INDICES_T*)indicesGm_.GetPhyAddr();
        __gm__ volatile X_T* yPtr = (__gm__ volatile X_T*)yGm_.GetPhyAddr();

        if (nis) {
            GatherCore<true>(xPtr, indicesPtr, yPtr, yIndexBase, currentCoreElements, gatherSize, outerSize, innerSize,
                             gatherDimSize);
        } else {
            GatherCore<false>(xPtr, indicesPtr, yPtr, yIndexBase, currentCoreElements, gatherSize, outerSize, innerSize,
                              gatherDimSize);
        }
    }

private:
    template <const bool NIS>
    __aicore__ inline void GatherCore(__gm__ X_T* x, __gm__ INDICES_T* indices, __gm__ volatile X_T* y,
                                      INDEX_SIZE_T yIndexBase, INDEX_SIZE_T currentCoreElements,
                                      INDEX_SIZE_T gatherSize, INDEX_SIZE_T outerSize, INDEX_SIZE_T innerSize,
                                      INDEX_SIZE_T gatherDimSize)
    {
        // 原 GatherSimt 计算逻辑原样保留；线程级(threadIdx/asc_vf_call)→核内串行循环；
        // UintDiv 快速除法→普通除法，去除 simt 依赖
        for (INDEX_SIZE_T index = 0; index < currentCoreElements; index++) {
            INDEX_SIZE_T yIndex = yIndexBase + index;
            INDEX_SIZE_T tmp = yIndex / innerSize; // 原 UintDiv(yIndex, m0, shift0)
            INDEX_SIZE_T tmp1 = tmp / gatherSize;  // 原 UintDiv(tmp, m1, shift1)
            INDEX_SIZE_T gatherI = tmp - tmp1 * gatherSize;
            INDEX_SIZE_T batchI = tmp1 / outerSize; // 原 UintDiv(tmp1, m2, shift2)
            INDEX_SIZE_T innerI = yIndex - tmp * innerSize;
            INDEX_SIZE_T indicesIndex = batchI * gatherSize + gatherI;

            INDICES_T indicesValue = indices[indicesIndex];
            if constexpr (NIS) {
                if (unlikely(indicesValue < 0)) {
                    indicesValue += static_cast<INDICES_T>(gatherDimSize);
                }
            }
            INDEX_SIZE_T indicesValueI = static_cast<INDEX_SIZE_T>(indicesValue);
            INDEX_SIZE_T xIndex = (tmp1 * gatherDimSize + indicesValueI) * innerSize + innerI;
            bool oob = indicesValue < 0 || indicesValueI >= gatherDimSize;
            y[yIndex] = oob ? static_cast<X_T>(0) : x[xIndex];
        }
    }

private:
    GlobalTensor<X_T> xGm_;
    GlobalTensor<INDICES_T> indicesGm_;
    GlobalTensor<X_T> yGm_;
    const GatherV2TilingData* tilingData_ = nullptr;
};
} // namespace gatherv2
#endif // GATHER_V2_SIMD_1D_H
