/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gather_elements.h
 * \brief
 */
#ifndef GATHER_ELEMENTS_H
#define GATHER_ELEMENTS_H

#include "kernel_operator.h"
#include "gather_elements_common.h"

namespace GatherElements {
using namespace AscendC;

template <typename X_T, typename INDEX_T, typename COM_T, int32_t DIM_NUM, int32_t AXIS>
class GatherElementsKernel {
public:
    __aicore__ inline GatherElementsKernel(TPipe &pipe) : pipe_(pipe) {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR index, GM_ADDR y, const GatherElementsTilingData* tilingData);
    __aicore__ inline void ProcessOptim(__gm__ X_T* xAddr, __gm__ INDEX_T* indexAddr, __gm__ volatile X_T* yAddr,
        __gm__ const GatherElementsTilingData* tilingData);
    __aicore__ inline void Process();
private:
    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<INDEX_T> indexGm_;
    AscendC::GlobalTensor<X_T> yGm_;
    TPipe &pipe_;
    const GatherElementsTilingData *tilingData_;
    TBuf<TPosition::VECCALC> magicAndShiftBuf_;

    COM_T shift_[M_SHIFT_OFFSET];
    COM_T m_[M_SHIFT_OFFSET];

    constexpr static uint32_t SIMT_BUF_SIZE = 512; // actually use 2 * 7 * 8 + 2 * 8 * 8 = 254
    constexpr static uint32_t SIMT_BUF_B64_NUM = 32;
    constexpr static uint32_t OFFSET7_B64_NUM = 16;
    constexpr static uint32_t UINT64_SIZE_BYTE = 8;
    constexpr static uint32_t OFFSET8_B64_NUM = 16;
};

template <typename X_T, typename INDEX_T, typename COM_T, int32_t DIM_NUM, int32_t AXIS>
__aicore__ inline void GatherElementsKernel<X_T, INDEX_T, COM_T, DIM_NUM, AXIS>::Init(GM_ADDR x, GM_ADDR index,
    GM_ADDR y, const GatherElementsTilingData* tilingData)
{
    tilingData_ = tilingData;
    xGm_.SetGlobalBuffer((__gm__ X_T *)x);
    indexGm_.SetGlobalBuffer((__gm__ INDEX_T *)index);
    yGm_.SetGlobalBuffer((__gm__ X_T *)y);
    pipe_.InitBuffer(magicAndShiftBuf_, SIMT_BUF_SIZE);
}

__aicore__ inline uint32_t ConvertDimIdx(uint32_t dimSize, uint32_t curDim)
{
    return curDim + (MAX_DIM_LEN - dimSize);
}

template <typename X_T, typename INDEX_T, typename U>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM_2048) inline void GatherDim1Compute(__gm__ X_T* xAddr, __gm__ INDEX_T* indexAddr, __gm__ volatile X_T* yAddr,
    __gm__ const GatherElementsTilingData* tiling)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
    if (Simt::GetBlockIdx() >= tilingData->usedCore) {
        return;
    }
    U coreOffset = static_cast<U>(Simt::GetBlockIdx() * tilingData->perCoreNum);
    U batchNum = static_cast<U>((Simt::GetBlockIdx() == tilingData->usedCore - 1) ? tilingData->tailCoreNum :
        tilingData->perCoreNum) + coreOffset;
    for (U i = Simt::GetThreadIdx() + coreOffset; i < batchNum; i += Simt::GetThreadNum()) {
        yAddr[i] = xAddr[indexAddr[i]];
    }
}

template <typename X_T, typename INDEX_T, typename U, int32_t AXIS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM_2048) inline void GatherDim2Compute(__gm__ X_T* xAddr, __gm__ INDEX_T* indexAddr, __gm__ volatile X_T* yAddr,
    __gm__ const GatherElementsTilingData* tiling)
{
    GET_TILING_DATA_PTR_WITH_STRUCT(GatherElementsTilingData, tilingData, tiling);
    if (Simt::GetBlockIdx() >= tilingData->usedCore) {
        return;
    }
    U coreOffset = static_cast<U>(Simt::GetBlockIdx() * tilingData->perCoreNum);
    U batchNum = static_cast<U>((Simt::GetBlockIdx() == tilingData->usedCore - 1) ? tilingData->tailCoreNum :
        tilingData->perCoreNum) + coreOffset;
    U magic = tilingData->magic[MS_IDX6];
    U shift = tilingData->shift[MS_IDX6];
    U indexStride = tilingData->indexStrideArr[MS_IDX6];
    U xStride = tilingData->xStrideArr[MS_IDX6];
    for (U i = Simt::GetThreadIdx() + coreOffset; i < batchNum; i += Simt::GetThreadNum()) {
        U indexVal = indexAddr[i];

        U dim0Index = Simt::UintDiv(i, magic, shift);
        U dim1Index = i - dim0Index * indexStride;
        if constexpr (AXIS == 0) {
            yAddr[i] = xAddr[indexVal * xStride + dim1Index];
        } else {
            yAddr[i] = xAddr[dim0Index * xStride + indexVal];
        }
    }
}

template <typename X_T, typename INDEX_T, typename U, int32_t AXIS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM_2048) inline void GatherDim3Compute(__gm__ INDEX_T* indexGm,
    __gm__ X_T* yGm, __gm__ X_T* xGm, U m, U shift, U m1, U shift1, U indexStride, U xStridex,
    U indexStride1, U xStridex1, U batchNum, U coreOffset)
{
    for (U i = Simt::GetThreadIdx() + coreOffset; i < batchNum; i += Simt::GetThreadNum()) {
        U indexVal = indexGm[i];

        U dim0Index = Simt::UintDiv(i, m, shift);
        U newIdx2 = i - dim0Index * indexStride;
        U dim1Index = Simt::UintDiv(newIdx2, m1, shift1);
        U dim2Index = newIdx2 - dim1Index * indexStride1;
        if constexpr (AXIS == 0) {
            yGm[i] = xGm[indexVal * xStridex + dim1Index * xStridex1 + dim2Index];
        } else if constexpr (AXIS == 1) {
            yGm[i] = xGm[dim0Index * xStridex + indexVal * xStridex1 + dim2Index];
        } else {
            yGm[i] = xGm[dim0Index * xStridex + dim1Index * xStridex1 + indexVal];
        }
    }
}

template <typename X_T, typename INDEX_T, typename U, int32_t AXIS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM_1024) inline void GatherDim4Compute(__gm__ INDEX_T* indexGm,
    __gm__ X_T* yGm, __gm__ X_T* xGm, __local_mem__ U* mAndShiftAddr, __local_mem__ U* strideAddr,
    U batchNum, U coreOffset)
{
    for (U i = Simt::GetThreadIdx() + coreOffset; i < batchNum; i += Simt::GetThreadNum()) {
        U indexVal = indexGm[i];

        U dim0Index = Simt::UintDiv(i, mAndShiftAddr[MS_IDX4], mAndShiftAddr[M_SHIFT_OFFSET + MS_IDX4]);
        U newIdx3 = i - dim0Index * strideAddr[MS_IDX4];

        U dim1Index = Simt::UintDiv(newIdx3, mAndShiftAddr[MS_IDX5], mAndShiftAddr[M_SHIFT_OFFSET + MS_IDX5]);
        U newIdx2 = newIdx3 - dim1Index * strideAddr[MS_IDX5];

        U dim2Index = Simt::UintDiv(newIdx2, mAndShiftAddr[MS_IDX6], mAndShiftAddr[M_SHIFT_OFFSET + MS_IDX6]);
        U dim3Index = newIdx2 - dim2Index * strideAddr[MS_IDX6];

        if constexpr (AXIS == 0) {
            yGm[i] = xGm[indexVal * strideAddr[M_SHIFT_OFFSET + MS_IDX4] +
                dim1Index * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                dim2Index * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + dim3Index];
        } else if constexpr (AXIS == 1) {
            yGm[i] = xGm[dim0Index * strideAddr[M_SHIFT_OFFSET + MS_IDX4] +
                indexVal * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                dim2Index * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + dim3Index];
        } else if constexpr (AXIS == DIM2) {
            yGm[i] = xGm[dim0Index * strideAddr[M_SHIFT_OFFSET + MS_IDX4] +
                dim1Index * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                indexVal * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + dim3Index];
        } else {
            yGm[i] = xGm[dim0Index * strideAddr[M_SHIFT_OFFSET + MS_IDX4] +
                dim1Index * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                dim2Index * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + indexVal];
        }
    }
}

template <typename X_T, typename INDEX_T, typename U, int32_t AXIS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM_1024) inline void GatherDim5Compute(__gm__ INDEX_T* indexGm,
    __gm__ X_T* yGm, __gm__ X_T* xGm, __local_mem__ U* mAndShiftAddr, __local_mem__ U* strideAddr,
    U batchNum, U coreOffset)
{
    for (U i = Simt::GetThreadIdx() + coreOffset; i < batchNum; i += Simt::GetThreadNum()) {
        U indexVal = indexGm[i];

        U dim0Index = Simt::UintDiv(i, mAndShiftAddr[MS_IDX3], mAndShiftAddr[M_SHIFT_OFFSET + MS_IDX3]);
        U newIdx4 = i - dim0Index * strideAddr[MS_IDX3];

        U dim1Index = Simt::UintDiv(newIdx4, mAndShiftAddr[MS_IDX4], mAndShiftAddr[M_SHIFT_OFFSET + MS_IDX4]);
        U newIdx3 = newIdx4 - dim1Index * strideAddr[MS_IDX4];

        U dim2Index = Simt::UintDiv(newIdx3, mAndShiftAddr[MS_IDX5], mAndShiftAddr[M_SHIFT_OFFSET + MS_IDX5]);
        U newIdx2 = newIdx3 - dim2Index * strideAddr[MS_IDX5];

        U dim3Index = Simt::UintDiv(newIdx2, mAndShiftAddr[MS_IDX6], mAndShiftAddr[M_SHIFT_OFFSET + MS_IDX6]);
        U dim4Index = newIdx2 - dim3Index * strideAddr[MS_IDX6];

       if constexpr (AXIS == 0) {
            yGm[i] = xGm[indexVal * strideAddr[M_SHIFT_OFFSET + MS_IDX3] +
                dim1Index * strideAddr[M_SHIFT_OFFSET + MS_IDX4] + dim2Index * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                dim3Index * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + dim4Index];
        } else if constexpr (AXIS == 1) {
            yGm[i] = xGm[dim0Index * strideAddr[M_SHIFT_OFFSET + MS_IDX3] +
                indexVal * strideAddr[M_SHIFT_OFFSET + MS_IDX4] + dim2Index * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                dim3Index * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + dim4Index];
        } else if constexpr (AXIS == DIM2) {
            yGm[i] = xGm[dim0Index * strideAddr[M_SHIFT_OFFSET + MS_IDX3] +
                dim1Index * strideAddr[M_SHIFT_OFFSET + MS_IDX4] + indexVal * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                dim3Index * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + dim4Index];
        } else if constexpr (AXIS == DIM3) {
            yGm[i] = xGm[dim0Index * strideAddr[M_SHIFT_OFFSET + MS_IDX3] +
                dim1Index * strideAddr[M_SHIFT_OFFSET + MS_IDX4] + dim2Index * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                indexVal * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + dim4Index];
        } else {
            yGm[i] = xGm[dim0Index * strideAddr[M_SHIFT_OFFSET + MS_IDX3] +
                dim1Index * strideAddr[M_SHIFT_OFFSET + MS_IDX4] + dim2Index * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                dim3Index * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + indexVal];
        }
    }
}

template <typename X_T, typename INDEX_T, typename U, int32_t AXIS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM_512) inline void GatherDim6Compute(__gm__ INDEX_T* indexGm,
    __gm__ X_T* yGm, __gm__ X_T* xGm, __local_mem__ U* mAndShiftAddr, __local_mem__ U* strideAddr,
    U batchNum, U coreOffset)
{
    for (U i = Simt::GetThreadIdx() + coreOffset; i < batchNum; i += Simt::GetThreadNum()) {
        U indexVal = indexGm[i];

        U dim0Index = Simt::UintDiv(i, mAndShiftAddr[MS_IDX2], mAndShiftAddr[M_SHIFT_OFFSET + MS_IDX2]);
        U newIdx5 = i - dim0Index * strideAddr[MS_IDX2];

        U dim1Index = Simt::UintDiv(newIdx5, mAndShiftAddr[MS_IDX3], mAndShiftAddr[M_SHIFT_OFFSET + MS_IDX3]);
        U newIdx4 = newIdx5 - dim1Index * strideAddr[MS_IDX3];

        U dim2Index = Simt::UintDiv(newIdx4, mAndShiftAddr[MS_IDX4], mAndShiftAddr[M_SHIFT_OFFSET + MS_IDX4]);
        U newIdx3 = newIdx4 - dim2Index * strideAddr[MS_IDX4];

        U dim3Index = Simt::UintDiv(newIdx3, mAndShiftAddr[MS_IDX5], mAndShiftAddr[M_SHIFT_OFFSET + MS_IDX5]);
        U newIdx2 = newIdx3 - dim3Index * strideAddr[MS_IDX5];

        U dim4Index = Simt::UintDiv(newIdx2, mAndShiftAddr[MS_IDX6], mAndShiftAddr[M_SHIFT_OFFSET + MS_IDX6]);
        U dim5Index = newIdx2 - dim4Index * strideAddr[MS_IDX6];
        if constexpr (AXIS == 0) {
            yGm[i] = xGm[indexVal * strideAddr[M_SHIFT_OFFSET + MS_IDX2] +
                dim1Index * strideAddr[M_SHIFT_OFFSET + MS_IDX3] + dim2Index * strideAddr[M_SHIFT_OFFSET + MS_IDX4] +
                dim3Index * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                dim4Index * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + dim5Index];
        } else if constexpr (AXIS == 1) {
            yGm[i] = xGm[dim0Index * strideAddr[M_SHIFT_OFFSET + MS_IDX2] +
                indexVal * strideAddr[M_SHIFT_OFFSET + MS_IDX3] + dim2Index * strideAddr[M_SHIFT_OFFSET + MS_IDX4] +
                dim3Index * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                dim4Index * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + dim5Index];
        } else if constexpr (AXIS == DIM2) {
            yGm[i] = xGm[dim0Index * strideAddr[M_SHIFT_OFFSET + MS_IDX2] +
                dim1Index * strideAddr[M_SHIFT_OFFSET + MS_IDX3] + indexVal * strideAddr[M_SHIFT_OFFSET + MS_IDX4] +
                dim3Index * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                dim4Index * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + dim5Index];
        } else if constexpr (AXIS == DIM3) {
            yGm[i] = xGm[dim0Index * strideAddr[M_SHIFT_OFFSET + MS_IDX2] +
                dim1Index * strideAddr[M_SHIFT_OFFSET + MS_IDX3] + dim2Index * strideAddr[M_SHIFT_OFFSET + MS_IDX4] +
                indexVal * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                dim4Index * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + dim5Index];
        } else if constexpr (AXIS == DIM4) {
            yGm[i] = xGm[dim0Index * strideAddr[M_SHIFT_OFFSET + MS_IDX2] +
                dim1Index * strideAddr[M_SHIFT_OFFSET + MS_IDX3] + dim2Index * strideAddr[M_SHIFT_OFFSET + MS_IDX4] +
                dim3Index * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                indexVal * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + dim5Index];
        } else {
            yGm[i] = xGm[dim0Index * strideAddr[M_SHIFT_OFFSET + MS_IDX2] +
                dim1Index * strideAddr[M_SHIFT_OFFSET + MS_IDX3] + dim2Index * strideAddr[M_SHIFT_OFFSET + MS_IDX4] +
                dim3Index * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                dim4Index * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + indexVal];
        }
    }
}

template <typename X_T, typename INDEX_T, typename U, int32_t AXIS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM_512) inline void GatherDim7Compute(__gm__ INDEX_T* indexGm,
    __gm__ X_T* yGm, __gm__ X_T* xGm, __local_mem__ U* mAndShiftAddr, __local_mem__ U* strideAddr,
    U batchNum, U coreOffset)
{
    for (U i = Simt::GetThreadIdx() + coreOffset; i < batchNum; i += Simt::GetThreadNum()) {
        U indexVal = indexGm[i];

        U dim0Index = Simt::UintDiv(i, mAndShiftAddr[MS_IDX1], mAndShiftAddr[M_SHIFT_OFFSET + MS_IDX1]);
        U newIdx6 = i - dim0Index * strideAddr[DIM6_INDEX];

        U dim1Index = Simt::UintDiv(newIdx6, mAndShiftAddr[MS_IDX2], mAndShiftAddr[M_SHIFT_OFFSET + MS_IDX2]);
        U newIdx5 = newIdx6 - dim1Index * strideAddr[DIM5_INDEX];

        U dim2Index = Simt::UintDiv(newIdx5, mAndShiftAddr[MS_IDX3], mAndShiftAddr[M_SHIFT_OFFSET + MS_IDX3]);
        U newIdx4 = newIdx5 - dim2Index * strideAddr[DIM4_INDEX];

        U dim3Index = Simt::UintDiv(newIdx4, mAndShiftAddr[MS_IDX4], mAndShiftAddr[M_SHIFT_OFFSET + MS_IDX4]);
        U newIdx3 = newIdx4 - dim3Index * strideAddr[DIM3_INDEX];

        U dim4Index = Simt::UintDiv(newIdx3, mAndShiftAddr[MS_IDX5], mAndShiftAddr[M_SHIFT_OFFSET + MS_IDX5]);
        U newIdx2 = newIdx3 - dim4Index * strideAddr[DIM2_INDEX];

        U dim5Index = Simt::UintDiv(newIdx2, mAndShiftAddr[MS_IDX6], mAndShiftAddr[M_SHIFT_OFFSET + MS_IDX6]);
        U dim6Index = newIdx2 - dim5Index * strideAddr[DIM1_INDEX];

        if constexpr (AXIS == 0) {
            yGm[i] = xGm[indexVal * strideAddr[M_SHIFT_OFFSET + MS_IDX1] +
                dim1Index * strideAddr[M_SHIFT_OFFSET + MS_IDX2] + dim2Index * strideAddr[M_SHIFT_OFFSET + MS_IDX3] +
                dim3Index * strideAddr[M_SHIFT_OFFSET + MS_IDX4] + dim4Index * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                dim5Index * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + dim6Index];
        } else if constexpr (AXIS == 1) {
            yGm[i] = xGm[dim0Index * strideAddr[M_SHIFT_OFFSET + MS_IDX1] +
                indexVal * strideAddr[M_SHIFT_OFFSET + MS_IDX2] + dim2Index * strideAddr[M_SHIFT_OFFSET + MS_IDX3] +
                dim3Index * strideAddr[M_SHIFT_OFFSET + MS_IDX4] + dim4Index * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                dim5Index * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + dim6Index];
        } else if constexpr (AXIS == DIM2) {
            yGm[i] = xGm[dim0Index * strideAddr[M_SHIFT_OFFSET + MS_IDX1] +
                dim1Index * strideAddr[M_SHIFT_OFFSET + MS_IDX2] + indexVal * strideAddr[M_SHIFT_OFFSET + MS_IDX3] +
                dim3Index * strideAddr[M_SHIFT_OFFSET + MS_IDX4] + dim4Index * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                dim5Index * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + dim6Index];
        } else if constexpr (AXIS == DIM3) {
            yGm[i] = xGm[dim0Index * strideAddr[M_SHIFT_OFFSET + MS_IDX1] +
                dim1Index * strideAddr[M_SHIFT_OFFSET + MS_IDX2] + dim2Index * strideAddr[M_SHIFT_OFFSET + MS_IDX3] +
                indexVal * strideAddr[M_SHIFT_OFFSET + MS_IDX4] + dim4Index * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                dim5Index * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + dim6Index];
        } else if constexpr (AXIS == DIM4) {
            yGm[i] = xGm[dim0Index * strideAddr[M_SHIFT_OFFSET + MS_IDX1] +
                dim1Index * strideAddr[M_SHIFT_OFFSET + MS_IDX2] + dim2Index * strideAddr[M_SHIFT_OFFSET + MS_IDX3] +
                dim3Index * strideAddr[M_SHIFT_OFFSET + MS_IDX4] + indexVal * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                dim5Index * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + dim6Index];
        } else if constexpr (AXIS == DIM5) {
            yGm[i] = xGm[dim0Index * strideAddr[M_SHIFT_OFFSET + MS_IDX1] +
                dim1Index * strideAddr[M_SHIFT_OFFSET + MS_IDX2] + dim2Index * strideAddr[M_SHIFT_OFFSET + MS_IDX3] +
                dim3Index * strideAddr[M_SHIFT_OFFSET + MS_IDX4] + dim4Index * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                indexVal * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + dim6Index];
        } else if constexpr (AXIS == DIM6) {
            yGm[i] = xGm[dim0Index * strideAddr[M_SHIFT_OFFSET + MS_IDX1] +
                dim1Index * strideAddr[M_SHIFT_OFFSET + MS_IDX2] + dim2Index * strideAddr[M_SHIFT_OFFSET + MS_IDX3] +
                dim3Index * strideAddr[M_SHIFT_OFFSET + MS_IDX4] + dim4Index * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                dim5Index * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + indexVal];
        }
    }
}

template <typename X_T, typename INDEX_T, typename U, int32_t AXIS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM_512) inline void GatherDim8Compute(__gm__ INDEX_T* indexGm,
    __gm__ X_T* yGm, __gm__ X_T* xGm, __local_mem__ U* mAndShiftAddr, __local_mem__ U* strideAddr,
    U batchNum, U coreOffset)
{
    for (U i = Simt::GetThreadIdx() + coreOffset; i < batchNum; i += Simt::GetThreadNum()) {
        U indexVal = indexGm[i];

        U dim0Index = Simt::UintDiv(i, mAndShiftAddr[MS_IDX0], mAndShiftAddr[M_SHIFT_OFFSET + MS_IDX0]);
        U newIdx7 = i - dim0Index * strideAddr[MS_IDX0];

        U dim1Index = Simt::UintDiv(newIdx7, mAndShiftAddr[MS_IDX1], mAndShiftAddr[M_SHIFT_OFFSET + MS_IDX1]);
        U newIdx6 = newIdx7 - dim1Index * strideAddr[MS_IDX1];

        U dim2Index = Simt::UintDiv(newIdx6, mAndShiftAddr[MS_IDX2], mAndShiftAddr[M_SHIFT_OFFSET + MS_IDX2]);
        U newIdx5 = newIdx6 - dim2Index * strideAddr[MS_IDX2];

        U dim3Index = Simt::UintDiv(newIdx5, mAndShiftAddr[MS_IDX3], mAndShiftAddr[M_SHIFT_OFFSET + MS_IDX3]);
        U newIdx4 = newIdx5 - dim3Index * strideAddr[MS_IDX3];

        U dim4Index = Simt::UintDiv(newIdx4, mAndShiftAddr[MS_IDX4], mAndShiftAddr[M_SHIFT_OFFSET + MS_IDX4]);
        U newIdx3 = newIdx4 - dim4Index * strideAddr[MS_IDX4];

        U dim5Index = Simt::UintDiv(newIdx3, mAndShiftAddr[MS_IDX5], mAndShiftAddr[M_SHIFT_OFFSET + MS_IDX5]);
        U newIdx2 = newIdx3 - dim5Index * strideAddr[MS_IDX5];

        U dim6Index = Simt::UintDiv(newIdx2, mAndShiftAddr[MS_IDX6], mAndShiftAddr[M_SHIFT_OFFSET + MS_IDX6]);
        U dim7Index = newIdx2 - dim6Index * strideAddr[MS_IDX6];

        if constexpr (AXIS == 0) {
            yGm[i] = xGm[indexVal * strideAddr[M_SHIFT_OFFSET + MS_IDX0] +
                dim1Index * strideAddr[M_SHIFT_OFFSET + MS_IDX1] + dim2Index * strideAddr[M_SHIFT_OFFSET + MS_IDX2] +
                dim3Index * strideAddr[M_SHIFT_OFFSET + MS_IDX3] + dim4Index * strideAddr[M_SHIFT_OFFSET + MS_IDX4] +
                dim5Index * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                dim6Index * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + dim7Index];
        } else if constexpr (AXIS == 1) {
            yGm[i] = xGm[dim0Index * strideAddr[M_SHIFT_OFFSET + MS_IDX0] +
                indexVal * strideAddr[M_SHIFT_OFFSET + MS_IDX1] + dim2Index * strideAddr[M_SHIFT_OFFSET + MS_IDX2] +
                dim3Index * strideAddr[M_SHIFT_OFFSET + MS_IDX3] + dim4Index * strideAddr[M_SHIFT_OFFSET + MS_IDX4] +
                dim5Index * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                dim6Index * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + dim7Index];
        } else if constexpr (AXIS == DIM2) {
            yGm[i] = xGm[dim0Index * strideAddr[M_SHIFT_OFFSET + MS_IDX0] +
                dim1Index * strideAddr[M_SHIFT_OFFSET + MS_IDX1] + indexVal * strideAddr[M_SHIFT_OFFSET + MS_IDX2] +
                dim3Index * strideAddr[M_SHIFT_OFFSET + MS_IDX3] + dim4Index * strideAddr[M_SHIFT_OFFSET + MS_IDX4] +
                dim5Index * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                dim6Index * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + dim7Index];
        } else if constexpr (AXIS == DIM3) {
            yGm[i] = xGm[dim0Index * strideAddr[M_SHIFT_OFFSET + MS_IDX0] +
                dim1Index * strideAddr[M_SHIFT_OFFSET + MS_IDX1] + dim2Index * strideAddr[M_SHIFT_OFFSET + MS_IDX2] +
                indexVal * strideAddr[M_SHIFT_OFFSET + MS_IDX3] + dim4Index * strideAddr[M_SHIFT_OFFSET + MS_IDX4] +
                dim5Index * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                dim6Index * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + dim7Index];
        } else if constexpr (AXIS == DIM4) {
            yGm[i] = xGm[dim0Index * strideAddr[M_SHIFT_OFFSET + MS_IDX0] +
                dim1Index * strideAddr[M_SHIFT_OFFSET + MS_IDX1] + dim2Index * strideAddr[M_SHIFT_OFFSET + MS_IDX2] +
                dim3Index * strideAddr[M_SHIFT_OFFSET + MS_IDX3] + indexVal * strideAddr[M_SHIFT_OFFSET + MS_IDX4] +
                dim5Index * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                dim6Index * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + dim7Index];
        } else if constexpr (AXIS == DIM5) {
            yGm[i] = xGm[dim0Index * strideAddr[M_SHIFT_OFFSET + MS_IDX0] +
                dim1Index * strideAddr[M_SHIFT_OFFSET + MS_IDX1] + dim2Index * strideAddr[M_SHIFT_OFFSET + MS_IDX2] +
                dim3Index * strideAddr[M_SHIFT_OFFSET + MS_IDX3] + dim4Index * strideAddr[M_SHIFT_OFFSET + MS_IDX4] +
                indexVal * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                dim6Index * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + dim7Index];
        } else if constexpr (AXIS == DIM6) {
            yGm[i] = xGm[dim0Index * strideAddr[M_SHIFT_OFFSET + MS_IDX0] +
                dim1Index * strideAddr[M_SHIFT_OFFSET + MS_IDX1] + dim2Index * strideAddr[M_SHIFT_OFFSET + MS_IDX2] +
                dim3Index * strideAddr[M_SHIFT_OFFSET + MS_IDX3] + dim4Index * strideAddr[M_SHIFT_OFFSET + MS_IDX4] +
                dim5Index * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                indexVal * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + dim7Index];
        } else {
            yGm[i] = xGm[dim0Index * strideAddr[M_SHIFT_OFFSET + MS_IDX0] +
                dim1Index * strideAddr[M_SHIFT_OFFSET + MS_IDX1] + dim2Index * strideAddr[M_SHIFT_OFFSET + MS_IDX2] +
                dim3Index * strideAddr[M_SHIFT_OFFSET + MS_IDX3] + dim4Index * strideAddr[M_SHIFT_OFFSET + MS_IDX4] +
                dim5Index * strideAddr[M_SHIFT_OFFSET + MS_IDX5] +
                dim6Index * strideAddr[M_SHIFT_OFFSET + MS_IDX6] + indexVal];
        }
    }
}

template <typename X_T, typename INDEX_T, typename COM_T, int32_t DIM_NUM, int32_t AXIS>
__aicore__ inline void GatherElementsKernel<X_T, INDEX_T, COM_T, DIM_NUM, AXIS>::ProcessOptim(__gm__ X_T* xAddr, __gm__ INDEX_T* indexAddr, __gm__ volatile X_T* yAddr,
    __gm__ const GatherElementsTilingData* tilingData)
{
    if constexpr (DIM_NUM == DIM1) {
        Simt::VF_CALL<GatherDim1Compute<X_T, INDEX_T, COM_T>>(Simt::Dim3(THREAD_DIM_2048),
            xAddr, indexAddr, yAddr, tilingData);
    } else if constexpr (DIM_NUM == DIM2) {
        Simt::VF_CALL<GatherDim2Compute<X_T, INDEX_T, COM_T, AXIS>>(Simt::Dim3(THREAD_DIM_2048),
            xAddr, indexAddr, yAddr, tilingData);
    }
}

template <typename X_T, typename INDEX_T, typename COM_T, int32_t DIM_NUM, int32_t AXIS>
__aicore__ inline void GatherElementsKernel<X_T, INDEX_T, COM_T, DIM_NUM, AXIS>::Process()
{
    if (GetBlockIdx() >= tilingData_->usedCore) {
        return;
    }
    LocalTensor<COM_T> mAndShiftLocal = magicAndShiftBuf_.Get<COM_T>(OFFSET7_B64_NUM);
    LocalTensor<COM_T> strideLocal = magicAndShiftBuf_.GetWithOffset<COM_T>(OFFSET8_B64_NUM, OFFSET7_B64_NUM * UINT64_SIZE_BYTE);

    for (int64_t i = MAX_DIM_LEN - DIM_NUM; i < M_SHIFT_OFFSET; i++) {
        GetUintDivMagicAndShift(m_[i], shift_[i], static_cast<COM_T>(tilingData_->indexStrideArr[i]));
        if constexpr (DIM_NUM != DIM2 && DIM_NUM != DIM3) {
            mAndShiftLocal(i) = m_[i];
            mAndShiftLocal(i + M_SHIFT_OFFSET) = shift_[i];
            strideLocal(i) = static_cast<COM_T>(tilingData_->indexStrideArr[i]);
            strideLocal(i + M_SHIFT_OFFSET) = static_cast<COM_T>(tilingData_->xStrideArr[i]);
        }
    }
    DataSyncBarrier<MemDsbT::UB>();

    COM_T coreOffset = static_cast<COM_T>(GetBlockIdx() * tilingData_->perCoreNum);
    COM_T batchNum = static_cast<COM_T>((GetBlockIdx() == tilingData_->usedCore - 1) ? tilingData_->tailCoreNum :
        tilingData_->perCoreNum) + coreOffset;
    __gm__ INDEX_T* indexAddr = (__gm__ INDEX_T*)(indexGm_.GetPhyAddr());
    __gm__ X_T* yAddr = (__gm__ X_T*)(yGm_.GetPhyAddr());
    __gm__ X_T* xAddr = (__gm__ X_T*)(xGm_.GetPhyAddr());
    __local_mem__ COM_T* mAndShiftAddr = (__local_mem__ COM_T*)(mAndShiftLocal.GetPhyAddr());
    __local_mem__ COM_T* strideAddr = (__local_mem__ COM_T*)(strideLocal.GetPhyAddr());

    if constexpr (DIM_NUM == DIM3) {
        Simt::VF_CALL<GatherDim3Compute<X_T, INDEX_T, COM_T, AXIS>>(Simt::Dim3(THREAD_DIM_2048),
            indexAddr, yAddr, xAddr, m_[MS_IDX5], shift_[MS_IDX5], m_[MS_IDX6], shift_[MS_IDX6],
            static_cast<COM_T>(tilingData_->indexStrideArr[MS_IDX5]),
            static_cast<COM_T>(tilingData_->xStrideArr[MS_IDX5]),
            static_cast<COM_T>(tilingData_->indexStrideArr[MS_IDX6]),
            static_cast<COM_T>(tilingData_->xStrideArr[MS_IDX6]), batchNum, coreOffset);
    } else if constexpr (DIM_NUM == DIM4) {
        Simt::VF_CALL<GatherDim4Compute<X_T, INDEX_T, COM_T, AXIS>>(Simt::Dim3(THREAD_DIM_1024),
            indexAddr, yAddr, xAddr, mAndShiftAddr, strideAddr, batchNum, coreOffset);
    } else if constexpr (DIM_NUM == DIM5) {
        Simt::VF_CALL<GatherDim5Compute<X_T, INDEX_T, COM_T, AXIS>>(Simt::Dim3(THREAD_DIM_1024),
            indexAddr, yAddr, xAddr, mAndShiftAddr, strideAddr, batchNum, coreOffset);
    } else if constexpr (DIM_NUM == DIM6) {
        Simt::VF_CALL<GatherDim6Compute<X_T, INDEX_T, COM_T, AXIS>>(Simt::Dim3(THREAD_DIM_512),
            indexAddr, yAddr, xAddr, mAndShiftAddr, strideAddr, batchNum, coreOffset);
    } else if constexpr (DIM_NUM == DIM7) {
        Simt::VF_CALL<GatherDim7Compute<X_T, INDEX_T, COM_T, AXIS>>(Simt::Dim3(THREAD_DIM_512),
            indexAddr, yAddr, xAddr, mAndShiftAddr, strideAddr, batchNum, coreOffset);
    } else {
        Simt::VF_CALL<GatherDim8Compute<X_T, INDEX_T, COM_T, AXIS>>(Simt::Dim3(THREAD_DIM_512),
            indexAddr, yAddr, xAddr, mAndShiftAddr, strideAddr, batchNum, coreOffset);
    }
}
} // namespace GatherElements
#endif // GATHER_ELEMENTS_H