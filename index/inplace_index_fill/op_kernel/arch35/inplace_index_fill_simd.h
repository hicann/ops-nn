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
 * \file inplace_index_fill_simd.h
 * \brief SIMD kernel (mask-bitmap based, in-place)
 */
#ifndef INPLACE_INDEX_FILL_SIMD_H
#define INPLACE_INDEX_FILL_SIMD_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "inplace_index_fill_struct.h"
#include "inplace_index_fill_common.h"
#include "simt_api/asc_simt.h"

namespace InplaceIndexFill {
using namespace AscendC;

// SIMT 核函数: 遍历 indices, 归一化后将 mask[nIdx] 置 1
template <typename INDICES_T, typename COM_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void InplaceIndexFillSimdMaskFill(
    uint64_t blockIdx, uint64_t blockNum, __gm__ INDICES_T* indices, __gm__ int8_t* mask, COM_T indicesNum, uint64_t n)
{
    COM_T threadIdx = static_cast<COM_T>(blockIdx * Simt::GetThreadNum() + Simt::GetThreadIdx());
    COM_T threadNum = static_cast<COM_T>(blockNum * Simt::GetThreadNum());
    for (COM_T idx = threadIdx; idx < indicesNum; idx += threadNum) {
        INDICES_T nIdx = static_cast<INDICES_T>(indices[idx]);
        nIdx = nIdx >= 0 ? nIdx : nIdx + static_cast<INDICES_T>(n);
        if (nIdx < 0 || nIdx >= static_cast<INDICES_T>(n)) {
            continue;
        }
        mask[nIdx] = static_cast<int8_t>(1);
    }
}

template <typename X_T, typename INDICES_T, typename COM_T>
class InplaceIndexFillSimd {
public:
    __aicore__ inline InplaceIndexFillSimd(const InplaceIndexFillSimdTilingData& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe)
    {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR value, GM_ADDR workspace)
    {
        xGm_.SetGlobalBuffer((__gm__ X_T*)(x));
        indicesGm_.SetGlobalBuffer((__gm__ INDICES_T*)(indices));
        valueGm_.SetGlobalBuffer((__gm__ X_T*)(value));
        blockIdx_ = GetBlockIdx();
        blockNum_ = GetBlockNum();
        fillValue_ = valueGm_.GetValue(0);
        workspace_ = workspace;
    }

    __aicore__ inline void Process()
    {
        COM_T p = static_cast<COM_T>(tilingData_.preDimProduct);
        COM_T q = static_cast<COM_T>(tilingData_.postDimProduct);
        COM_T n = static_cast<COM_T>(tilingData_.dimSize);

        // 1) 多核并行清零 mask 位图(workspace 上 N 字节) + 核间同步
        int64_t nLen = tilingData_.dimSize;
        InplaceIndexFillCommon::InitMaskZero(maskGm_, workspace_, nLen, blockIdx_, blockNum_);

        // 2) SIMT 遍历 indices, 置 mask[nIdx]=1
        COM_T indicesNum = static_cast<COM_T>(tilingData_.indicesNum);
        Simt::VF_CALL<InplaceIndexFillSimdMaskFill<INDICES_T, COM_T>>(
            Simt::Dim3(SIMT_THREAD_NUM), blockIdx_, blockNum_, (__gm__ INDICES_T*)(indicesGm_.GetPhyAddr()),
            (__gm__ int8_t*)(maskGm_.GetPhyAddr()), indicesNum, static_cast<uint64_t>(nLen));
        SyncAll();

        // 3) 按 P*N*Q 遍历, mask==1 的位置原地写 value (复用 common 设施)
        InplaceIndexFillCommon::CallMaskBasedFill<X_T, INDICES_T, COM_T>(
            (__gm__ X_T*)xGm_.GetPhyAddr(), (__gm__ int8_t*)maskGm_.GetPhyAddr(), fillValue_, n, p, q);
    }

private:
    GlobalTensor<X_T> xGm_; // in-place: 直接写回 x
    GlobalTensor<INDICES_T> indicesGm_;
    GlobalTensor<X_T> valueGm_;
    GlobalTensor<int8_t> maskGm_;
    const InplaceIndexFillSimdTilingData& tilingData_;
    TPipe& pipe_;
    GM_ADDR workspace_ = nullptr;
    uint32_t blockIdx_ = 0;
    uint32_t blockNum_ = 0;
    X_T fillValue_ = 0;
};

} // namespace InplaceIndexFill
#endif // INPLACE_INDEX_FILL_SIMD_H