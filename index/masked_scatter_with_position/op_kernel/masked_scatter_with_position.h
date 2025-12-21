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
 * \file masked_scatter_with_position.h
 * \brief
 */
#ifndef __MASKED_SCATTER_WITH_POSITION_H__
#define __MASKED_SCATTER_WITH_POSITION_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "masked_scatter_with_position_tiling_struct.h"

namespace MaskedScatterWithPosition {

using namespace AscendC;
constexpr uint32_t USED_THREAD_NUMS = 1024;
constexpr uint32_t THREAD_LAUNCH = 1024;
constexpr uint32_t PATTERN_BA = 0;
constexpr uint32_t PATTERN_AB = 1;

template <typename T, typename U, const uint32_t PATTERN_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_LAUNCH) inline void MaskedScatterWithPositionSimtAB(
    __gm__ T* xGm, __gm__ bool* maskGm, __gm__ int64_t* positionGm, __gm__ T* updatesGm, __gm__ T* yGm, U magic, U shift, U xNum, U xInner
);

template <typename T, typename U, const uint32_t PATTERN_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_LAUNCH) inline void MaskedScatterWithPositionSimtBA(
    __gm__ T* xGm, __gm__ bool* maskGm, __gm__ int64_t* positionGm, __gm__ T* updatesGm, __gm__ T* yGm, U magic, U shift, U xNum, U xInner
);

template <typename T, typename U, const uint32_t PATTERN_TYPE>
class MaskedScatterWithPositionSimt {
public:
    __aicore__ inline MaskedScatterWithPositionSimt (
        const MaskedScatterWithPositionTilingData* tilingData):tilingData_(tilingData){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR mask, GM_ADDR position, GM_ADDR updates, GM_ADDR y);
    __aicore__ inline void Process();

private:
    const MaskedScatterWithPositionTilingData* tilingData_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<bool> maskGm_;
    AscendC::GlobalTensor<int64_t> positionGm_;
    AscendC::GlobalTensor<T> updatesGm_;
    AscendC::GlobalTensor<T> yGm_;
};

template <typename T, typename U, const uint32_t PATTERN_TYPE>
__aicore__ inline void MaskedScatterWithPositionSimt<T, U, PATTERN_TYPE>::Init(GM_ADDR x, GM_ADDR mask, GM_ADDR position, GM_ADDR updates, GM_ADDR y)
{
    xGm_.SetGlobalBuffer((__gm__ T *)(x));
    maskGm_.SetGlobalBuffer((__gm__ bool *)(mask));
    positionGm_.SetGlobalBuffer((__gm__ int64_t *)(position));
    updatesGm_.SetGlobalBuffer((__gm__ T *)(updates));
    yGm_.SetGlobalBuffer((__gm__ T *)(y));
}

template <typename T, typename U, const uint32_t PATTERN_TYPE>
__aicore__ inline void MaskedScatterWithPositionSimt<T, U, PATTERN_TYPE>::Process()
{
    U xNum = tilingData_->xNum;
    U xInner = tilingData_->xInner;
    U xOutter = xNum / xInner;
    auto positionGm = (__gm__ int64_t*)(positionGm_.GetPhyAddr());
    U maskTrueNum = PATTERN_TYPE == PATTERN_AB ? (positionGm[xOutter - 1] * xInner) : (positionGm[xInner - 1] * xOutter);
    assert(maskTrueNum <= tilingData_->updatesEleNums,
        "The num of true in mask is larger than the num of elements in update.");
    U magic = 1;
    U shift = 1;
    GetUintDivMagicAndShift(magic, shift, static_cast<U>(xInner));

    if constexpr (PATTERN_TYPE == PATTERN_AB) {  // AB
        Simt::VF_CALL<MaskedScatterWithPositionSimtAB<T, U, PATTERN_TYPE>>(Simt::Dim3(USED_THREAD_NUMS),
        (__gm__ T*)(xGm_.GetPhyAddr()), (__gm__ bool*)(maskGm_.GetPhyAddr()), (__gm__ int64_t*)(positionGm_.GetPhyAddr()), (__gm__ T*)(updatesGm_.GetPhyAddr()),
        (__gm__ T*)(yGm_.GetPhyAddr()), magic, shift, xNum, xInner);
    } else {  // BA
        Simt::VF_CALL<MaskedScatterWithPositionSimtBA<T, U, PATTERN_TYPE>>(Simt::Dim3(USED_THREAD_NUMS),
        (__gm__ T*)(xGm_.GetPhyAddr()), (__gm__ bool*)(maskGm_.GetPhyAddr()), (__gm__ int64_t*)(positionGm_.GetPhyAddr()), (__gm__ T*)(updatesGm_.GetPhyAddr()),
        (__gm__ T*)(yGm_.GetPhyAddr()), magic, shift, xNum, xInner);
    }
}

template <typename T, typename U, const uint32_t PATTERN_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_LAUNCH) inline void MaskedScatterWithPositionSimtAB(
     __gm__ T* xGm, __gm__ bool* maskGm, __gm__ int64_t* positionGm, __gm__ T* updatesGm, __gm__ T* yGm, U magic, U shift, U xNum, U xInner
     )
{
    for (U i = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < xNum; i += Simt::GetBlockNum() * Simt::GetThreadNum()) {
        U rowidx = Simt::UintDiv(i, magic, shift);
        U colidx = i - rowidx * xInner;
        if (maskGm[rowidx] == true) {
            U prefixSum = (positionGm[rowidx] - 1) * xInner + colidx;
            yGm[i] = updatesGm[prefixSum];
        } else {
            yGm[i] = xGm[i];
        }
    }
}

template <typename T, typename U, const uint32_t PATTERN_TYPE>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_LAUNCH) inline void MaskedScatterWithPositionSimtBA(
     __gm__ T* xGm, __gm__ bool* maskGm, __gm__ int64_t* positionGm, __gm__ T* updatesGm, __gm__ T* yGm, U magic, U shift, U xNum, U xInner
     )
{
    for (U i = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < xNum; i += Simt::GetBlockNum() * Simt::GetThreadNum()) {
        U rowidx = Simt::UintDiv(i, magic, shift);
        U colidx = i - rowidx * xInner;
        if (maskGm[i % xInner] == true) {
            U prefixSum = positionGm[colidx] - 1 + rowidx * positionGm[xInner - 1];
            yGm[i] = updatesGm[prefixSum];
        } else {
            yGm[i] = xGm[i];
        }
    }
}
} // namespace MaskedScatterWithPosition
#endif // MASKED_SCATTER_WITH_POSITION_H