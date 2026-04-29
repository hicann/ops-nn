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
 * \file inplace_index_fill_simt.h
 * \brief
 */
#ifndef INPLACE_INDEX_FILL_SIMT_H_
#define INPLACE_INDEX_FILL_SIMT_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "inplace_index_fill_struct.h"
#include "simt_api/asc_simt.h"

namespace InplaceIndexFillSimt {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 2048;

template <typename T_X, typename INDEX_TYPE, typename COM_T>
class InplaceIndexFillSimtImpl {
public:
    __aicore__ inline InplaceIndexFillSimtImpl(const InplaceIndexFill::InplaceIndexFillSimtTilingData* tilingData):tilingData_(tilingData) {};
    __aicore__ inline void Init(GM_ADDR value);
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR indices);

private:
    AscendC::GlobalTensor<T_X> valueGm_;
    T_X fillValue;
    const InplaceIndexFill::InplaceIndexFillSimtTilingData* tilingData_;
};

// 初始化
template <typename T_X, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void InplaceIndexFillSimtImpl<T_X, INDEX_TYPE, COM_T>::Init(GM_ADDR value)
{
    // 入参直接再simt核心处理中进行
    valueGm_.SetGlobalBuffer((__gm__ T_X*)(value));
    fillValue = valueGm_.GetValue(0);
}

// 核心操作
template <typename T_X, typename INDEX_TYPE, typename COM_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void InplaceIndexFillSimtCompute(
    COM_T curOffset, COM_T perBlockData, COM_T dimSize,
    COM_T magic0, COM_T shift0, COM_T magic1, COM_T shift1,
    COM_T indicesNum, COM_T postDimProduct,
    __gm__ INDEX_TYPE* indices, T_X fillValue, __gm__ T_X* x)
{
    uint32_t thread_idx = threadIdx.x;
    uint32_t threadNum = blockDim.x;

    for (COM_T i = thread_idx; i < perBlockData; i += threadNum) {
        COM_T curIdx = i + curOffset;
        COM_T pIdx = Simt::UintDiv(curIdx, magic0, shift0);
        COM_T mIdx = Simt::UintDiv(static_cast<COM_T>(curIdx - pIdx * (indicesNum * postDimProduct)), magic1, shift1);
        COM_T qIdx = (curIdx - pIdx * (indicesNum * postDimProduct)) - mIdx * postDimProduct;
        int64_t curNIdx = indices[mIdx] >= 0 ? indices[mIdx] : indices[mIdx] + dimSize;
        // indices[mIdx] 越界，跳过不做处理
        if (curNIdx < 0 || curNIdx >= dimSize) {
            continue;
        }
        COM_T nIdx = static_cast<COM_T>(curNIdx);
        COM_T offset = pIdx * dimSize * postDimProduct + nIdx * postDimProduct + qIdx;
        x[offset] = fillValue;
    }
}
// 主函数
template <typename T_X, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void InplaceIndexFillSimtImpl<T_X, INDEX_TYPE, COM_T>::Process(GM_ADDR x, GM_ADDR indices)
{
    // 空tensor
    COM_T preDimProduct = tilingData_->preDimProduct;
    COM_T postDimProduct = tilingData_->postDimProduct;
    COM_T dimSize = tilingData_->dimSize;
    uint32_t curThreadNum = tilingData_->threadNum;
    COM_T indicesNum = tilingData_->indicesNum;
    COM_T perBlockData = tilingData_->perBlockData;

    COM_T temp = indicesNum * postDimProduct;
    COM_T magic0 = 0;
    COM_T shift0 = 0;
    GetUintDivMagicAndShift(magic0, shift0, temp);
    COM_T magic1 = 0;
    COM_T shift1 = 0;
    GetUintDivMagicAndShift(magic1, shift1, postDimProduct);

    COM_T usedCoreNum = tilingData_->usedCoreNum;
    COM_T blockIdx = GetBlockIdx();

    // 执行inplace_index_fill核心逻辑
    COM_T curOffset = GetBlockIdx() * perBlockData;
    if (blockIdx == usedCoreNum - 1) {
        perBlockData = tilingData_->tailBlockData;
    }

    asc_vf_call<InplaceIndexFillSimtCompute<T_X, INDEX_TYPE, COM_T>>(
        dim3{curThreadNum}, curOffset, perBlockData,
        dimSize, magic0, shift0, magic1, shift1,
        indicesNum, postDimProduct,
        (__gm__ INDEX_TYPE*)(indices), fillValue, (__gm__ T_X*)(x));
}

} // namespace InplaceIndexFillSimt

#endif // INPLACE_INDEX_FILL_SIMT_H_