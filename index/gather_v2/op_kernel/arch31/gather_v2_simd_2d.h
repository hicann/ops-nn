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
 * \file gather_v2_simd_2d.h
 * \brief KirinX90 纯 SIMD 2D 小 shape 模板（替代原 Gatherv2SimtTwoDim simt 实现，去除 simt_api 依赖）
 */
#ifndef GATHER_V2_SIMD_2D_H
#define GATHER_V2_SIMD_2D_H

#include "kernel_operator.h"
// 已删除: simt_api/asc_simt.h、basic_api/kernel_vec_intf.h（KirinX90 硬件不支持 simt）

namespace gatherv2 {
using namespace AscendC;

template <typename X_T, typename INDICES_T, typename INDEX_SIZE_T>
class Gatherv2Simd2D {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR axis, GM_ADDR y,
                                const GatherV2TilingDataSimtTwoDim* tilingData)
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

        INDEX_SIZE_T gatherDimSize = static_cast<INDEX_SIZE_T>(tilingData_->gatherDimSize);
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
            GatherCore<true>(xPtr, indicesPtr, yPtr, yIndexBase, currentCoreElements, innerSize, gatherDimSize);
        } else {
            GatherCore<false>(xPtr, indicesPtr, yPtr, yIndexBase, currentCoreElements, innerSize, gatherDimSize);
        }
    }

private:
    template <const bool NIS>
    __aicore__ inline void GatherCore(__gm__ X_T* x, __gm__ INDICES_T* indices, __gm__ volatile X_T* y,
                                      INDEX_SIZE_T yIndexBase, INDEX_SIZE_T currentCoreElements, INDEX_SIZE_T innerSize,
                                      INDEX_SIZE_T gatherDimSize)
    {
        // 2D: batch==1 && outer==1，yIndex 只分解成 gatherI/innerI 两维
        for (INDEX_SIZE_T index = 0; index < currentCoreElements; index++) {
            INDEX_SIZE_T yIndex = yIndexBase + index;
            INDEX_SIZE_T gatherI = yIndex / innerSize; // 原 UintDiv(yIndex, m0, shift0)
            INDEX_SIZE_T innerI = yIndex - gatherI * innerSize;

            INDICES_T indicesValue = indices[gatherI];
            if constexpr (NIS) {
                if (unlikely(indicesValue < 0)) {
                    indicesValue += static_cast<INDICES_T>(gatherDimSize);
                }
            }
            INDEX_SIZE_T indicesValueI = static_cast<INDEX_SIZE_T>(indicesValue);
            INDEX_SIZE_T xIndex = indicesValueI * innerSize + innerI;
            bool oob = indicesValue < 0 || indicesValueI >= gatherDimSize;
            y[yIndex] = oob ? static_cast<X_T>(0) : x[xIndex];
        }
    }

private:
    GlobalTensor<X_T> xGm_;
    GlobalTensor<INDICES_T> indicesGm_;
    GlobalTensor<X_T> yGm_;
    const GatherV2TilingDataSimtTwoDim* tilingData_ = nullptr;
};
} // namespace gatherv2
#endif // GATHER_V2_SIMD_2D_H
