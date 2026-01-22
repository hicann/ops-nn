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
 * \file gather_elements_no_contiguos.h
 * \brief
 */
#ifndef GATHER_ELEMENTS_NO_CONTIGUOUS_H
#define GATHER_ELEMENTS_NO_CONTIGUOUS_H

#include "kernel_operator.h"
#include "gather_elements_common.h"

namespace GatherElements {
using namespace AscendC;

template <typename X_T, typename INDEX_T, typename COM_T, int32_t DIM_NUM, int32_t AXIS, bool IS_SPECIAL=false>
class GatherElementsKernelNoContiguous {
public:
    __aicore__ inline GatherElementsKernelNoContiguous(TPipe &pipe) : pipe_(pipe) {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR index, GM_ADDR y, const GatherElementsNoContiguousTilingData* tilingData);
    __aicore__ inline void ProcessOptim(__gm__ X_T* xAddr, __gm__ INDEX_T* indexAddr, __gm__ volatile X_T* yAddr,
        __gm__ const GatherElementsNoContiguousTilingData* tilingData);
        __aicore__ inline void Process();
    __aicore__ inline void ProcessCommon();
    __aicore__ inline void ProcessSpecial();
private:
    AscendC::GlobalTensor<X_T> xGm_;
    AscendC::GlobalTensor<INDEX_T> indexGm_;
    AscendC::GlobalTensor<X_T> yGm_;
    TPipe &pipe_;
    const GatherElementsNoContiguousTilingData *tilingData_;
    TBuf<TPosition::VECCALC> strideBuf_;

    COM_T shift_[M_SHIFT_OFFSET];
    COM_T m_[M_SHIFT_OFFSET];

    constexpr static uint32_t SIMT_BUF_SIZE = 1024; 
};

template <typename X_T, typename INDEX_T, typename COM_T, int32_t DIM_NUM, int32_t AXIS, bool IS_SPECIAL>
__aicore__ inline void GatherElementsKernelNoContiguous<X_T, INDEX_T, COM_T, DIM_NUM, AXIS, IS_SPECIAL>::Init(GM_ADDR x, GM_ADDR index,
    GM_ADDR y, const GatherElementsNoContiguousTilingData* tilingData)
{
    tilingData_ = tilingData;
    xGm_.SetGlobalBuffer((__gm__ X_T *)x);
    indexGm_.SetGlobalBuffer((__gm__ INDEX_T *)index);
    yGm_.SetGlobalBuffer((__gm__ X_T *)y);
    pipe_.InitBuffer(strideBuf_, SIMT_BUF_SIZE);
}

template <typename X_T, typename INDEX_T, typename U>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM_2048) inline void GatherDim1NoContiguousCompute(__gm__ INDEX_T* indexGm, __gm__ X_T* yGm,
    __gm__ X_T* xGm, U indexStride, U xStridex,  U batchNum, U coreOffset)
{
    for (U i = Simt::GetThreadIdx() + coreOffset; i < batchNum; i += Simt::GetThreadNum()) {
        U indexVal = indexGm[i * indexStride];
        yGm[i] = xGm[indexVal * xStridex];
    }
}

template <typename X_T, typename INDEX_T, typename U, int32_t AXIS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM_2048) inline void GatherDim2NoContiguousCompute(__gm__ INDEX_T* indexGm, __gm__ X_T* yGm,
    __gm__ X_T* xGm, U m, U shift, U indexStride, U xStridex, U indexContiguousStride, U indexStride1, U xStridex1, U batchNum,
    U coreOffset)
{
    for (U i = Simt::GetThreadIdx() + coreOffset; i < batchNum; i += Simt::GetThreadNum()) {
        U dim0Index = Simt::UintDiv(i, m, shift);
        U dim1Index = i - dim0Index * indexContiguousStride;
        U indexVal = indexGm[dim0Index * indexStride + dim1Index * indexStride1];

        if constexpr (AXIS == 0) {
            yGm[i] = xGm[indexVal * xStridex + dim1Index * xStridex1];
        } else {
            yGm[i] = xGm[dim0Index * xStridex + indexVal * xStridex1];
        } 
    }
}

template <typename X_T, typename INDEX_T, typename U, int32_t AXIS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM_2048) inline void GatherDim3NoContiguousCompute(__gm__ INDEX_T* indexGm, __gm__ X_T* yGm,
    __gm__ X_T* xGm, U m, U shift, U m1, U shift1, U indexStride, U xStridex, U indexContiguousStride, U indexStride1, U xStridex1,
    U indexContiguousStride1, U indexStride2, U xStridex2, U batchNum, U coreOffset)
{
    for (U i = Simt::GetThreadIdx() + coreOffset; i < batchNum; i += Simt::GetThreadNum()) {
        U dim0Index = Simt::UintDiv(i, m, shift);
        U newIdx2 = i - dim0Index * indexContiguousStride;
        U dim1Index = Simt::UintDiv(newIdx2, m1, shift1);
        U dim2Index = newIdx2 - dim1Index * indexContiguousStride1;
        U indexVal = indexGm[dim0Index * indexStride + dim1Index * indexStride1 + dim2Index * indexStride2];

        if constexpr (AXIS == 0) {
            yGm[i] = xGm[indexVal * xStridex + dim1Index * xStridex1 + dim2Index * xStridex2];
        } else if constexpr (AXIS == 1) {
            yGm[i] = xGm[dim0Index * xStridex + indexVal * xStridex1 + dim2Index * xStridex2];
        } else {
            yGm[i] = xGm[dim0Index * xStridex + dim1Index * xStridex1 + indexVal * xStridex2];
        }
    }
}

template <typename X_T, typename INDEX_T, typename U, int32_t AXIS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM_2048) inline void GatherDim3NoContiguousComputeB64(__gm__ INDEX_T* indexGm, __gm__ X_T* yGm,
    __gm__ X_T* xGm, __ubuf__ U* stride, U m, U shift, U m1, U shift1, U batchNum, U coreOffset)
{
    for (U i = Simt::GetThreadIdx() + coreOffset; i < batchNum; i += Simt::GetThreadNum()) {
        U dim0Index = Simt::UintDiv(i, m, shift);
        U newIdx2 = i - dim0Index * stride[0];
        U dim1Index = Simt::UintDiv(newIdx2, m1, shift1);
        U dim2Index = newIdx2 - dim1Index * stride[1];
        U indexVal = indexGm[dim0Index * stride[3] + dim1Index * stride[4]  + dim2Index * stride[5]];

        if constexpr (AXIS == 0) {
            yGm[i] = xGm[indexVal * stride[6] + dim1Index * stride[7] + dim2Index * stride[8]];
        } else if constexpr (AXIS == 1) {
            yGm[i] = xGm[dim0Index * stride[6] + indexVal * stride[7] + dim2Index * stride[8]];
        } else {
            yGm[i] = xGm[dim0Index * stride[6] + dim1Index * stride[7] + indexVal * stride[8]];
        }
    }
}

template <typename X_T, typename INDEX_T, typename U, int32_t AXIS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM_1024) inline void GatherDim4NoContiguousCompute(__gm__ INDEX_T* indexGm, __gm__ X_T* yGm, 
    __gm__ X_T* xGm, U m, U shift, U m1, U shift1, U m2, U shift2, U indexStride, U xStridex, U indexContiguousStride,
    U indexStride1, U xStridex1, U indexContiguousStride1, U indexStride2, U xStridex2, U indexContiguousStride2, U indexStride3, U xStridex3,
    U batchNum, U coreOffset)
{
    for (U i = Simt::GetThreadIdx() + coreOffset; i < batchNum; i += Simt::GetThreadNum()) {
        U dim0Index = Simt::UintDiv(i, m, shift);
        U newIdx2 = i - dim0Index * indexContiguousStride;
        U dim1Index = Simt::UintDiv(newIdx2, m1, shift1);
        U newIdx3 = newIdx2 - dim1Index * indexContiguousStride1;
        U dim2Index = Simt::UintDiv(newIdx3, m2, shift2);
        U dim3Index = newIdx3 - dim2Index * indexContiguousStride2;
        U indexVal = indexGm[dim0Index * indexStride + dim1Index * indexStride1 + dim2Index * indexStride2 + dim3Index * indexStride3];

        if constexpr (AXIS == 0) {
            yGm[i] = xGm[indexVal * xStridex + dim1Index * xStridex1 + dim2Index * xStridex2 + dim3Index * xStridex3];
        } else if constexpr (AXIS == 1) {
            yGm[i] = xGm[dim0Index * xStridex + indexVal * xStridex1 + dim2Index * xStridex2 + dim3Index * xStridex3];
        } else if constexpr (AXIS == 2) {
            yGm[i] = xGm[dim0Index * xStridex + dim1Index * xStridex1 + indexVal * xStridex2 + dim3Index * xStridex3];
        }  else {
            yGm[i] = xGm[dim0Index * xStridex + dim1Index * xStridex1 + dim2Index * xStridex2 + indexVal * xStridex3];
        }
    }
}

template <typename X_T, typename INDEX_T, typename U, int32_t AXIS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM_1024) inline void GatherDim4NoContiguousComputeB64(__gm__ INDEX_T* indexGm, __gm__ X_T* yGm, 
    __gm__ X_T* xGm, __ubuf__ U* stride, U m, U shift, U m1, U shift1, U m2, U shift2, U batchNum, U coreOffset)
{
    for (U i = Simt::GetThreadIdx() + coreOffset; i < batchNum; i += Simt::GetThreadNum()) {
        U dim0Index = Simt::UintDiv(i, m, shift);
        U newIdx2 = i - dim0Index * stride[0];
        U dim1Index = Simt::UintDiv(newIdx2, m1, shift1);
        U newIdx3 = newIdx2 - dim1Index * stride[1];
        U dim2Index = Simt::UintDiv(newIdx3, m2, shift2);
        U dim3Index = newIdx3 - dim2Index * stride[2];
        U indexVal = indexGm[dim0Index * stride[4] + dim1Index * stride[5]  + dim2Index * stride[6]  + dim3Index * stride[7]];

        if constexpr (AXIS == 0) {
            yGm[i] = xGm[indexVal * stride[8] + dim1Index * stride[9] + dim2Index * stride[10] + dim3Index * stride[11]];
        } else if constexpr (AXIS == 1) {
            yGm[i] = xGm[dim0Index * stride[8] + indexVal * stride[9] + dim2Index * stride[10] + dim3Index * stride[11]];
        } else if constexpr (AXIS == 2) {
            yGm[i] = xGm[dim0Index * stride[8] + dim1Index * stride[9] + indexVal * stride[10] + dim3Index * stride[11]];
        }  else {
            yGm[i] = xGm[dim0Index * stride[8] + dim1Index * stride[9] + dim2Index * stride[10] + indexVal * stride[11]];
        }
    }
}

template <typename X_T, typename INDEX_T, typename U, int32_t AXIS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM_1024) inline void GatherDim5NoContiguousCompute(__gm__ INDEX_T* indexGm, __gm__ X_T* yGm, 
    __gm__ X_T* xGm, U m, U shift, U m1, U shift1, U m2, U shift2, U m3, U shift3, U indexStride, U xStridex, U indexContiguousStride,
    U indexStride1, U xStridex1, U indexContiguousStride1, U indexStride2, U xStridex2, U indexContiguousStride2,  
    U indexStride3, U xStridex3, U indexContiguousStride3, U indexStride4, U xStridex4, U batchNum, U coreOffset)
{
    for (U i = Simt::GetThreadIdx() + coreOffset; i < batchNum; i += Simt::GetThreadNum()) {
        U dim0Index = Simt::UintDiv(i, m, shift);
        U newIdx2 = i - dim0Index * indexContiguousStride;
        U dim1Index = Simt::UintDiv(newIdx2, m1, shift1);
        U newIdx3 = newIdx2 - dim1Index * indexContiguousStride1;
        U dim2Index = Simt::UintDiv(newIdx3, m2, shift2);
        U newIdx4 = newIdx3 - dim2Index * indexContiguousStride2;
        U dim3Index = Simt::UintDiv(newIdx4, m3, shift3);
        U dim4Index = newIdx4 - dim3Index * indexContiguousStride3;
        U indexVal = indexGm[dim0Index * indexStride + dim1Index * indexStride1 + dim2Index * indexStride2 + dim3Index * indexStride3 + 
            dim4Index * indexStride4];

        if constexpr (AXIS == 0) {
            yGm[i] = xGm[indexVal * xStridex + dim1Index * xStridex1 + dim2Index * xStridex2 + dim3Index * xStridex3 + dim4Index * xStridex4];
        } else if constexpr (AXIS == 1) {
            yGm[i] = xGm[dim0Index * xStridex + indexVal * xStridex1 + dim2Index * xStridex2 + dim3Index * xStridex3 + dim4Index * xStridex4];
        } else if constexpr (AXIS == 2) {
            yGm[i] = xGm[dim0Index * xStridex + dim1Index * xStridex1 + indexVal * xStridex2 + dim3Index * xStridex3 + dim4Index * xStridex4];
        } else if constexpr (AXIS == 3) {
            yGm[i] = xGm[dim0Index * xStridex + dim1Index * xStridex1 + dim2Index * xStridex2 + indexVal * xStridex3 + dim4Index * xStridex4];
        } else {
            yGm[i] = xGm[dim0Index * xStridex + dim1Index * xStridex1 + dim2Index * xStridex2 + dim3Index * xStridex3 + indexVal * xStridex4];
        }
    }
}

template <typename X_T, typename INDEX_T, typename U, int32_t AXIS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM_512) inline void GatherDim6NoContiguousCompute(__gm__ INDEX_T* indexGm, __gm__ X_T* yGm, 
    __gm__ X_T* xGm, U m, U shift, U m1, U shift1, U m2, U shift2, U m3, U shift3, U m4, U shift4, U indexStride, U xStridex, U indexContiguousStride,
    U indexStride1, U xStridex1, U indexContiguousStride1, U indexStride2, U xStridex2, U indexContiguousStride2,  
    U indexStride3, U xStridex3, U indexContiguousStride3, U indexStride4, U xStridex4, U indexContiguousStride4,
    U indexStride5, U xStridex5, U batchNum, U coreOffset)
{
    for (U i = Simt::GetThreadIdx() + coreOffset; i < batchNum; i += Simt::GetThreadNum()) {
        U dim0Index = Simt::UintDiv(i, m, shift);
        U newIdx2 = i - dim0Index * indexContiguousStride;
        U dim1Index = Simt::UintDiv(newIdx2, m1, shift1);
        U newIdx3 = newIdx2 - dim1Index * indexContiguousStride1;
        U dim2Index = Simt::UintDiv(newIdx3, m2, shift2);
        U newIdx4 = newIdx3 - dim2Index * indexContiguousStride2;
        U dim3Index = Simt::UintDiv(newIdx4, m3, shift3);
        U newIdx5 = newIdx4 - dim3Index * indexContiguousStride3;
        U dim4Index = Simt::UintDiv(newIdx5, m4, shift4);
        U dim5Index = newIdx5 - dim4Index * indexContiguousStride4;
        U indexVal = indexGm[dim0Index * indexStride + dim1Index * indexStride1 + dim2Index * indexStride2 + dim3Index * indexStride3 + 
            dim4Index * indexStride4 + dim5Index * indexStride5];

        if constexpr (AXIS == 0) {
            yGm[i] = xGm[indexVal * xStridex + dim1Index * xStridex1 + dim2Index * xStridex2 + dim3Index * xStridex3 + dim4Index * xStridex4 + 
                dim5Index * xStridex5];
        } else if constexpr (AXIS == 1) {
            yGm[i] = xGm[dim0Index * xStridex + indexVal * xStridex1 + dim2Index * xStridex2 + dim3Index * xStridex3 + dim4Index * xStridex4 + 
                dim5Index * xStridex5];
        } else if constexpr (AXIS == 2) {
            yGm[i] = xGm[dim0Index * xStridex + dim1Index * xStridex1 + indexVal * xStridex2 + dim3Index * xStridex3 + dim4Index * xStridex4 + 
                dim5Index * xStridex5];
        } else if constexpr (AXIS == 3) {
            yGm[i] = xGm[dim0Index * xStridex + dim1Index * xStridex1 + dim2Index * xStridex2 + indexVal * xStridex3 + dim4Index * xStridex4 + 
                dim5Index * xStridex5];
        } else if constexpr (AXIS == 4) {
            yGm[i] = xGm[dim0Index * xStridex + dim1Index * xStridex1 + dim2Index * xStridex2 + dim3Index * xStridex3 + indexVal * xStridex4 + 
                dim5Index * xStridex5];
        } else {
            yGm[i] = xGm[dim0Index * xStridex + dim1Index * xStridex1 + dim2Index * xStridex2 + dim3Index * xStridex3 + dim4Index * xStridex4 + 
                indexVal * xStridex5];
        }
    }
}

template <typename X_T, typename INDEX_T, typename U, int32_t AXIS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM_512) inline void GatherDim7NoContiguousCompute(__gm__ INDEX_T* indexGm, __gm__ X_T* yGm, 
    __gm__ X_T* xGm, U m, U shift, U m1, U shift1, U m2, U shift2, U m3, U shift3, U m4, U shift4, U m5, U shift5,
    U indexStride, U xStridex, U indexContiguousStride, U indexStride1, U xStridex1, U indexContiguousStride1,
    U indexStride2, U xStridex2, U indexContiguousStride2, U indexStride3, U xStridex3, U indexContiguousStride3,
    U indexStride4, U xStridex4, U indexContiguousStride4, U indexStride5, U xStridex5, U indexContiguousStride5,
    U indexStride6, U xStridex6, U batchNum, U coreOffset)
{
    for (U i = Simt::GetThreadIdx() + coreOffset; i < batchNum; i += Simt::GetThreadNum()) {
        U dim0Index = Simt::UintDiv(i, m, shift);
        U newIdx2 = i - dim0Index * indexContiguousStride;
        U dim1Index = Simt::UintDiv(newIdx2, m1, shift1);
        U newIdx3 = newIdx2 - dim1Index * indexContiguousStride1;
        U dim2Index = Simt::UintDiv(newIdx3, m2, shift2);
        U newIdx4 = newIdx3 - dim2Index * indexContiguousStride2;
        U dim3Index = Simt::UintDiv(newIdx4, m3, shift3);
        U newIdx5 = newIdx4 - dim3Index * indexContiguousStride3;
        U dim4Index = Simt::UintDiv(newIdx5, m4, shift4);
        U newIdx6 = newIdx5 - dim4Index * indexContiguousStride4;
        U dim5Index = Simt::UintDiv(newIdx6, m5, shift5);
        U dim6Index = newIdx6 - dim5Index * indexContiguousStride5;
        U indexVal = indexGm[dim0Index * indexStride + dim1Index * indexStride1 + dim2Index * indexStride2 + dim3Index * indexStride3 + 
            dim4Index * indexStride4 + dim5Index * indexStride5 + dim6Index * indexStride6];

        if constexpr (AXIS == 0) {
            yGm[i] = xGm[indexVal * xStridex + dim1Index * xStridex1 + dim2Index * xStridex2 + dim3Index * xStridex3 + dim4Index * xStridex4 + 
                dim5Index * xStridex5 + dim6Index * xStridex6];
        } else if constexpr (AXIS == 1) {
            yGm[i] = xGm[dim0Index * xStridex + indexVal * xStridex1 + dim2Index * xStridex2 + dim3Index * xStridex3 + dim4Index * xStridex4 + 
                dim5Index * xStridex5 + dim6Index * xStridex6];
        } else if constexpr (AXIS == 2) {
            yGm[i] = xGm[dim0Index * xStridex + dim1Index * xStridex1 + indexVal * xStridex2 + dim3Index * xStridex3 + dim4Index * xStridex4 + 
                dim5Index * xStridex5 + dim6Index * xStridex6];
        } else if constexpr (AXIS == 3) {
            yGm[i] = xGm[dim0Index * xStridex + dim1Index * xStridex1 + dim2Index * xStridex2 + indexVal * xStridex3 + dim4Index * xStridex4 + 
                dim5Index * xStridex5 + dim6Index * xStridex6];
        } else if constexpr (AXIS == 4) {
            yGm[i] = xGm[dim0Index * xStridex + dim1Index * xStridex1 + dim2Index * xStridex2 + dim3Index * xStridex3 + indexVal * xStridex4 + 
                dim5Index * xStridex5 + dim6Index * xStridex6];
        } else if constexpr (AXIS == 5) {
            yGm[i] = xGm[dim0Index * xStridex + dim1Index * xStridex1 + dim2Index * xStridex2 + dim3Index * xStridex3 + dim4Index * xStridex4 + 
                indexVal * xStridex5 + dim6Index * xStridex6];
        } else {
            yGm[i] = xGm[dim0Index * xStridex + dim1Index * xStridex1 + dim2Index * xStridex2 + dim3Index * xStridex3 + dim4Index * xStridex4 + 
                dim5Index * xStridex5 + indexVal * xStridex6];
        }
    }
}

template <typename X_T, typename INDEX_T, typename U, int32_t AXIS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM_512) inline void GatherDim8NoContiguousCompute(__gm__ INDEX_T* indexGm, __gm__ X_T* yGm, 
    __gm__ X_T* xGm, U m, U shift, U m1, U shift1, U m2, U shift2, U m3, U shift3, U m4, U shift4, U m5, U shift5, U m6, U shift6,
    U indexStride, U xStridex, U indexContiguousStride, U indexStride1, U xStridex1, U indexContiguousStride1,
    U indexStride2, U xStridex2, U indexContiguousStride2, U indexStride3, U xStridex3, U indexContiguousStride3,
    U indexStride4, U xStridex4, U indexContiguousStride4, U indexStride5, U xStridex5, U indexContiguousStride5,
    U indexStride6, U xStridex6, U indexContiguousStride6, U indexStride7, U xStridex7, U batchNum, U coreOffset)
{
    for (U i = Simt::GetThreadIdx() + coreOffset; i < batchNum; i += Simt::GetThreadNum()) {
        U dim0Index = Simt::UintDiv(i, m, shift);
        U newIdx2 = i - dim0Index * indexContiguousStride;
        U dim1Index = Simt::UintDiv(newIdx2, m1, shift1);
        U newIdx3 = newIdx2 - dim1Index * indexContiguousStride1;
        U dim2Index = Simt::UintDiv(newIdx3, m2, shift2);
        U newIdx4 = newIdx3 - dim2Index * indexContiguousStride2;
        U dim3Index = Simt::UintDiv(newIdx4, m3, shift3);
        U newIdx5 = newIdx4 - dim3Index * indexContiguousStride3;
        U dim4Index = Simt::UintDiv(newIdx5, m4, shift4);
        U newIdx6 = newIdx5 - dim4Index * indexContiguousStride4;
        U dim5Index = Simt::UintDiv(newIdx6, m5, shift5);
        U newIdx7 = newIdx6 - dim5Index * indexContiguousStride5;
        U dim6Index = Simt::UintDiv(newIdx7, m6, shift6);
        U dim7Index = newIdx7 - dim6Index * indexContiguousStride6;
        U indexVal = indexGm[dim0Index * indexStride + dim1Index * indexStride1 + dim2Index * indexStride2 + dim3Index * indexStride3 + 
            dim4Index * indexStride4 + dim5Index * indexStride5 + dim6Index * indexStride6 + dim7Index * indexStride7];

        if constexpr (AXIS == 0) {
            yGm[i] = xGm[indexVal * xStridex + dim1Index * xStridex1 + dim2Index * xStridex2 + dim3Index * xStridex3 + dim4Index * xStridex4 + 
                dim5Index * xStridex5 + dim6Index * xStridex6 + dim7Index * xStridex7];
        } else if constexpr (AXIS == 1) {
            yGm[i] = xGm[dim0Index * xStridex + indexVal * xStridex1 + dim2Index * xStridex2 + dim3Index * xStridex3 + dim4Index * xStridex4 + 
                dim5Index * xStridex5 + dim6Index * xStridex6 + dim7Index * xStridex7];
        } else if constexpr (AXIS == 2) {
            yGm[i] = xGm[dim0Index * xStridex + dim1Index * xStridex1 + indexVal * xStridex2 + dim3Index * xStridex3 + dim4Index * xStridex4 + 
                dim5Index * xStridex5 + dim6Index * xStridex6 + dim7Index * xStridex7];
        } else if constexpr (AXIS == 3) {
            yGm[i] = xGm[dim0Index * xStridex + dim1Index * xStridex1 + dim2Index * xStridex2 + indexVal * xStridex3 + dim4Index * xStridex4 + 
                dim5Index * xStridex5 + dim6Index * xStridex6 + dim7Index * xStridex7];
        } else if constexpr (AXIS == 4) {
            yGm[i] = xGm[dim0Index * xStridex + dim1Index * xStridex1 + dim2Index * xStridex2 + dim3Index * xStridex3 + indexVal * xStridex4 + 
                dim5Index * xStridex5 + dim6Index * xStridex6 + dim7Index * xStridex7];
        } else if constexpr (AXIS == 5) {
            yGm[i] = xGm[dim0Index * xStridex + dim1Index * xStridex1 + dim2Index * xStridex2 + dim3Index * xStridex3 + dim4Index * xStridex4 + 
                indexVal * xStridex5 + dim6Index * xStridex6 + dim7Index * xStridex7];
        } else if constexpr (AXIS == 6) {
            yGm[i] = xGm[dim0Index * xStridex + dim1Index * xStridex1 + dim2Index * xStridex2 + dim3Index * xStridex3 + dim4Index * xStridex4 + 
                dim5Index * xStridex5 + indexVal * xStridex6 + dim7Index * xStridex7];
        } else {
            yGm[i] = xGm[dim0Index * xStridex + dim1Index * xStridex1 + dim2Index * xStridex2 + dim3Index * xStridex3 + dim4Index * xStridex4 + 
                dim5Index * xStridex5 + dim6Index * xStridex6 + indexVal * xStridex7];
        }
    }
}

template <typename X_T, typename INDEX_T, typename U, int32_t AXIS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM_2048) inline void GatherDim3ComputeTransPose(__gm__ INDEX_T* indexGm, __gm__ X_T* yGm,
    __gm__ X_T* xGm, U m, U shift, U m1, U shift1, U indexStride, U xStridex, U indexContiguousStride, U yStride, U indexStride1,
    U xStridex1, U indexContiguousStride1, U yStride1, U indexStride2, U xStridex2, U batchNum, U coreOffset)
{
    for (U i = Simt::GetThreadIdx() + coreOffset; i < batchNum; i += Simt::GetThreadNum()) {
        //  [a, b, c]  -> [b, a, c]
        U dim1Index = Simt::UintDiv(i, m, shift);
        U newIdx2 = i - dim1Index * indexContiguousStride;
        U dim0Index = Simt::UintDiv(newIdx2, m1, shift1);
        U dim2Index = newIdx2 - dim0Index * indexContiguousStride1;
        U indexVal = indexGm[dim0Index * indexStride + dim1Index * indexStride1 + dim2Index * indexStride2];
        U yIndex = dim0Index * yStride + dim1Index * yStride1 + dim2Index;
        if constexpr (AXIS == 0) {
            yGm[yIndex] = xGm[indexVal * xStridex + dim1Index * xStridex1 + dim2Index * xStridex2];
        } else if constexpr (AXIS == 1) {
            yGm[yIndex] = xGm[dim0Index * xStridex + indexVal * xStridex1 + dim2Index * xStridex2];
        } else {
            yGm[yIndex] = xGm[dim0Index * xStridex + dim1Index * xStridex1 + indexVal * xStridex2];
        }
    }
}

template <typename X_T, typename INDEX_T, typename U, int32_t AXIS>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_DIM_2048) inline void GatherDim3ComputeTransPoseB64(__gm__ INDEX_T* indexGm, __gm__ X_T* yGm,
    __gm__ X_T* xGm, __ubuf__ U* stride, U m, U shift, U m1, U shift1, U batchNum, U coreOffset)
{
    for (U i = Simt::GetThreadIdx() + coreOffset; i < batchNum; i += Simt::GetThreadNum()) {
        //  [a, b, c]  -> [b, a, c]
        U dim1Index = Simt::UintDiv(i, m, shift);
        U newIdx2 = i - dim1Index * stride[0];
        U dim0Index = Simt::UintDiv(newIdx2, m1, shift1);
        U dim2Index = newIdx2 - dim0Index * stride[1];
        U indexVal = indexGm[dim0Index * stride[3] + dim1Index * stride[4] + dim2Index * stride[5]];
        U yIndex = dim0Index * stride[6] + dim1Index * stride[7] + dim2Index;
        if constexpr (AXIS == 0) {
            yGm[yIndex] = xGm[indexVal * stride[9] + dim1Index * stride[10] + dim2Index * stride[11]];
        } else if constexpr (AXIS == 1) {
            yGm[yIndex] = xGm[dim0Index * stride[9] + indexVal * stride[10] + dim2Index * stride[11]];
        } else {
            yGm[yIndex] = xGm[dim0Index * stride[9] + dim1Index * stride[10] + indexVal * stride[11]];
        }
    }
}

template <typename X_T, typename INDEX_T, typename COM_T, int32_t DIM_NUM, int32_t AXIS, bool IS_SPECIAL>
__aicore__ inline void GatherElementsKernelNoContiguous<X_T, INDEX_T, COM_T, DIM_NUM, AXIS, IS_SPECIAL>::ProcessCommon()
{
    if (GetBlockIdx() >= tilingData_->usedCore) {
        return;
    }
    if constexpr (!IS_SPECIAL) {
        int64_t indexContigousStride[MAX_DIM_LEN] = {1, 1, 1, 1, 1, 1, 1, 1};
        for (int i = MAX_DIM_LEN - 2; i >= 0; --i) {
            indexContigousStride[i] = indexContigousStride[i + 1] * tilingData_->indexShape[i + 1];
        }
        for (uint32_t i = MAX_DIM_LEN - DIM_NUM; i < M_SHIFT_OFFSET; i++) {
            GetUintDivMagicAndShift(m_[i], shift_[i], static_cast<COM_T>(indexContigousStride[i]));
        }
        COM_T coreOffset = static_cast<COM_T>(GetBlockIdx() * tilingData_->perCoreNum);
        COM_T batchNum = static_cast<COM_T>((GetBlockIdx() == tilingData_->usedCore - 1) ? tilingData_->tailCoreNum :
            tilingData_->perCoreNum) + coreOffset;
        __gm__ INDEX_T* indexAddr = (__gm__ INDEX_T*)(indexGm_.GetPhyAddr());
        __gm__ X_T* yAddr = (__gm__ X_T*)(yGm_.GetPhyAddr());
        __gm__ X_T* xAddr = (__gm__ X_T*)(xGm_.GetPhyAddr());

        if constexpr (DIM_NUM == DIM1) {
            Simt::VF_CALL<GatherDim1NoContiguousCompute<X_T, INDEX_T, COM_T>>(Simt::Dim3(THREAD_DIM_2048),
                indexAddr, yAddr, xAddr,   
                static_cast<COM_T>(tilingData_->indexStride[MS_IDX7]),
                static_cast<COM_T>(tilingData_->xStride[MS_IDX7]),      
                batchNum, 
                coreOffset);
        } else if constexpr (DIM_NUM == DIM2) {
            Simt::VF_CALL<GatherDim2NoContiguousCompute<X_T, INDEX_T, COM_T, AXIS>>(Simt::Dim3(THREAD_DIM_2048),
                indexAddr, yAddr, xAddr, m_[MS_IDX6], shift_[MS_IDX6],
                static_cast<COM_T>(tilingData_->indexStride[MS_IDX6]),
                static_cast<COM_T>(tilingData_->xStride[MS_IDX6]),
                static_cast<COM_T>(indexContigousStride[MS_IDX6]),  
                static_cast<COM_T>(tilingData_->indexStride[MS_IDX7]),
                static_cast<COM_T>(tilingData_->xStride[MS_IDX7]),      
                batchNum, 
                coreOffset);       
        } else if constexpr (DIM_NUM == DIM3) {
            if constexpr (sizeof(COM_T) == B32) {
                Simt::VF_CALL<GatherDim3NoContiguousCompute<X_T, INDEX_T, COM_T, AXIS>>(Simt::Dim3(THREAD_DIM_2048),
                indexAddr, yAddr, xAddr, m_[MS_IDX5], shift_[MS_IDX5], m_[MS_IDX6], shift_[MS_IDX6],
                static_cast<COM_T>(tilingData_->indexStride[MS_IDX5]),
                static_cast<COM_T>(tilingData_->xStride[MS_IDX5]),
                static_cast<COM_T>(indexContigousStride[MS_IDX5]),
                static_cast<COM_T>(tilingData_->indexStride[MS_IDX6]),
                static_cast<COM_T>(tilingData_->xStride[MS_IDX6]),
                static_cast<COM_T>(indexContigousStride[MS_IDX6]),  
                static_cast<COM_T>(tilingData_->indexStride[MS_IDX7]),
                static_cast<COM_T>(tilingData_->xStride[MS_IDX7]),      
                batchNum, 
                coreOffset);
            } else {
                LocalTensor<COM_T> strideLocal = strideBuf_.Get<COM_T>();
                uint16_t preFillNum = MAX_DIM_LEN - DIM3;
                for (uint16_t i = 0; i < DIM3; i++) {
                    strideLocal(i) = static_cast<COM_T>(indexContigousStride[preFillNum + i]);
                    strideLocal(i + DIM3) = static_cast<COM_T>(tilingData_->indexStride[preFillNum + i]);
                    strideLocal(i + TWO * DIM3) = static_cast<COM_T>(tilingData_->xStride[preFillNum + i]);
                }
                DataSyncBarrier<MemDsbT::UB>();
                __local_mem__ COM_T* strideAddr = (__local_mem__ COM_T*)(strideLocal.GetPhyAddr());
                Simt::VF_CALL<GatherDim3NoContiguousComputeB64<X_T, INDEX_T, COM_T, AXIS>>(Simt::Dim3(THREAD_DIM_2048),
                indexAddr, yAddr, xAddr, strideAddr, 
                m_[MS_IDX5], shift_[MS_IDX5], m_[MS_IDX6], shift_[MS_IDX6],
                batchNum, 
                coreOffset);
            }
        } else if constexpr (DIM_NUM == DIM4) {
            if constexpr (sizeof(COM_T) == B32) {
                Simt::VF_CALL<GatherDim4NoContiguousCompute<X_T, INDEX_T, COM_T, AXIS>>(Simt::Dim3(THREAD_DIM_1024),
                indexAddr, yAddr, xAddr, m_[MS_IDX4], shift_[MS_IDX4],
                m_[MS_IDX5], shift_[MS_IDX5], m_[MS_IDX6], shift_[MS_IDX6],
                static_cast<COM_T>(tilingData_->indexStride[MS_IDX4]),
                static_cast<COM_T>(tilingData_->xStride[MS_IDX4]),
                static_cast<COM_T>(indexContigousStride[MS_IDX4]),
                static_cast<COM_T>(tilingData_->indexStride[MS_IDX5]),
                static_cast<COM_T>(tilingData_->xStride[MS_IDX5]),
                static_cast<COM_T>(indexContigousStride[MS_IDX5]),
                static_cast<COM_T>(tilingData_->indexStride[MS_IDX6]),
                static_cast<COM_T>(tilingData_->xStride[MS_IDX6]),
                static_cast<COM_T>(indexContigousStride[MS_IDX6]),  
                static_cast<COM_T>(tilingData_->indexStride[MS_IDX7]),
                static_cast<COM_T>(tilingData_->xStride[MS_IDX7]),      
                batchNum, 
                coreOffset);
            } else {
                LocalTensor<COM_T> strideLocal = strideBuf_.Get<COM_T>();
                uint16_t preFillNum = MAX_DIM_LEN - DIM4;
                for (uint16_t i = 0; i < DIM4; i++) {
                    strideLocal(i) = static_cast<COM_T>(indexContigousStride[preFillNum + i]);
                    strideLocal(i + DIM4) = static_cast<COM_T>(tilingData_->indexStride[preFillNum + i]);
                    strideLocal(i + TWO * DIM4) = static_cast<COM_T>(tilingData_->xStride[preFillNum + i]);
                }
                DataSyncBarrier<MemDsbT::UB>();
                __local_mem__ COM_T* strideAddr = (__local_mem__ COM_T*)(strideLocal.GetPhyAddr());
                Simt::VF_CALL<GatherDim4NoContiguousComputeB64<X_T, INDEX_T, COM_T, AXIS>>(Simt::Dim3(THREAD_DIM_1024),
                indexAddr, yAddr, xAddr, strideAddr, m_[MS_IDX4], shift_[MS_IDX4],
                m_[MS_IDX5], shift_[MS_IDX5], m_[MS_IDX6], shift_[MS_IDX6],   
                batchNum, 
                coreOffset);
            }
        } else if constexpr (DIM_NUM == DIM5) {
            Simt::VF_CALL<GatherDim5NoContiguousCompute<X_T, INDEX_T, COM_T, AXIS>>(Simt::Dim3(THREAD_DIM_1024),
            indexAddr, yAddr, xAddr, m_[MS_IDX3], shift_[MS_IDX3], m_[MS_IDX4], shift_[MS_IDX4],
            m_[MS_IDX5], shift_[MS_IDX5], m_[MS_IDX6], shift_[MS_IDX6],
            static_cast<COM_T>(tilingData_->indexStride[MS_IDX3]),
            static_cast<COM_T>(tilingData_->xStride[MS_IDX3]),
            static_cast<COM_T>(indexContigousStride[MS_IDX3]),
            static_cast<COM_T>(tilingData_->indexStride[MS_IDX4]),
            static_cast<COM_T>(tilingData_->xStride[MS_IDX4]),
            static_cast<COM_T>(indexContigousStride[MS_IDX4]),
            static_cast<COM_T>(tilingData_->indexStride[MS_IDX5]),
            static_cast<COM_T>(tilingData_->xStride[MS_IDX5]),
            static_cast<COM_T>(indexContigousStride[MS_IDX5]),
            static_cast<COM_T>(tilingData_->indexStride[MS_IDX6]),
            static_cast<COM_T>(tilingData_->xStride[MS_IDX6]),
            static_cast<COM_T>(indexContigousStride[MS_IDX6]),  
            static_cast<COM_T>(tilingData_->indexStride[MS_IDX7]),
            static_cast<COM_T>(tilingData_->xStride[MS_IDX7]),      
            batchNum, 
            coreOffset);
        } else if constexpr (DIM_NUM == DIM6) {
            Simt::VF_CALL<GatherDim6NoContiguousCompute<X_T, INDEX_T, COM_T, AXIS>>(Simt::Dim3(THREAD_DIM_512),
            indexAddr, yAddr, xAddr, m_[MS_IDX2], shift_[MS_IDX2], m_[MS_IDX3], shift_[MS_IDX3], m_[MS_IDX4], shift_[MS_IDX4],
            m_[MS_IDX5], shift_[MS_IDX5], m_[MS_IDX6], shift_[MS_IDX6],
            static_cast<COM_T>(tilingData_->indexStride[MS_IDX2]),
            static_cast<COM_T>(tilingData_->xStride[MS_IDX2]),
            static_cast<COM_T>(indexContigousStride[MS_IDX2]),
            static_cast<COM_T>(tilingData_->indexStride[MS_IDX3]),
            static_cast<COM_T>(tilingData_->xStride[MS_IDX3]),
            static_cast<COM_T>(indexContigousStride[MS_IDX3]),
            static_cast<COM_T>(tilingData_->indexStride[MS_IDX4]),
            static_cast<COM_T>(tilingData_->xStride[MS_IDX4]),
            static_cast<COM_T>(indexContigousStride[MS_IDX4]),
            static_cast<COM_T>(tilingData_->indexStride[MS_IDX5]),
            static_cast<COM_T>(tilingData_->xStride[MS_IDX5]),
            static_cast<COM_T>(indexContigousStride[MS_IDX5]),
            static_cast<COM_T>(tilingData_->indexStride[MS_IDX6]),
            static_cast<COM_T>(tilingData_->xStride[MS_IDX6]),
            static_cast<COM_T>(indexContigousStride[MS_IDX6]),  
            static_cast<COM_T>(tilingData_->indexStride[MS_IDX7]),
            static_cast<COM_T>(tilingData_->xStride[MS_IDX7]),      
            batchNum, 
            coreOffset);
        } else if constexpr (DIM_NUM == DIM7) {
            Simt::VF_CALL<GatherDim7NoContiguousCompute<X_T, INDEX_T, COM_T, AXIS>>(Simt::Dim3(THREAD_DIM_512),
            indexAddr, yAddr, xAddr, m_[MS_IDX1], shift_[MS_IDX1], m_[MS_IDX2], shift_[MS_IDX2], m_[MS_IDX3], shift_[MS_IDX3], 
            m_[MS_IDX4], shift_[MS_IDX4], m_[MS_IDX5], shift_[MS_IDX5], m_[MS_IDX6], shift_[MS_IDX6],
            static_cast<COM_T>(tilingData_->indexStride[MS_IDX1]),
            static_cast<COM_T>(tilingData_->xStride[MS_IDX1]),
            static_cast<COM_T>(indexContigousStride[MS_IDX1]),
            static_cast<COM_T>(tilingData_->indexStride[MS_IDX2]),
            static_cast<COM_T>(tilingData_->xStride[MS_IDX2]),
            static_cast<COM_T>(indexContigousStride[MS_IDX2]),
            static_cast<COM_T>(tilingData_->indexStride[MS_IDX3]),
            static_cast<COM_T>(tilingData_->xStride[MS_IDX3]),
            static_cast<COM_T>(indexContigousStride[MS_IDX3]),
            static_cast<COM_T>(tilingData_->indexStride[MS_IDX4]),
            static_cast<COM_T>(tilingData_->xStride[MS_IDX4]),
            static_cast<COM_T>(indexContigousStride[MS_IDX4]),
            static_cast<COM_T>(tilingData_->indexStride[MS_IDX5]),
            static_cast<COM_T>(tilingData_->xStride[MS_IDX5]),
            static_cast<COM_T>(indexContigousStride[MS_IDX5]),
            static_cast<COM_T>(tilingData_->indexStride[MS_IDX6]),
            static_cast<COM_T>(tilingData_->xStride[MS_IDX6]),
            static_cast<COM_T>(indexContigousStride[MS_IDX6]),  
            static_cast<COM_T>(tilingData_->indexStride[MS_IDX7]),
            static_cast<COM_T>(tilingData_->xStride[MS_IDX7]),      
            batchNum, 
            coreOffset);
        } else if constexpr (DIM_NUM == DIM8) {
            Simt::VF_CALL<GatherDim8NoContiguousCompute<X_T, INDEX_T, COM_T, AXIS>>(Simt::Dim3(THREAD_DIM_512),
            indexAddr, yAddr, xAddr, m_[MS_IDX0], shift_[MS_IDX0], m_[MS_IDX1], shift_[MS_IDX1], m_[MS_IDX2], shift_[MS_IDX2], 
            m_[MS_IDX3], shift_[MS_IDX3], m_[MS_IDX4], shift_[MS_IDX4], m_[MS_IDX5], shift_[MS_IDX5], m_[MS_IDX6], shift_[MS_IDX6],
            static_cast<COM_T>(tilingData_->indexStride[MS_IDX0]),
            static_cast<COM_T>(tilingData_->xStride[MS_IDX0]),
            static_cast<COM_T>(indexContigousStride[MS_IDX0]),
            static_cast<COM_T>(tilingData_->indexStride[MS_IDX1]),
            static_cast<COM_T>(tilingData_->xStride[MS_IDX1]),
            static_cast<COM_T>(indexContigousStride[MS_IDX1]),
            static_cast<COM_T>(tilingData_->indexStride[MS_IDX2]),
            static_cast<COM_T>(tilingData_->xStride[MS_IDX2]),
            static_cast<COM_T>(indexContigousStride[MS_IDX2]),
            static_cast<COM_T>(tilingData_->indexStride[MS_IDX3]),
            static_cast<COM_T>(tilingData_->xStride[MS_IDX3]),
            static_cast<COM_T>(indexContigousStride[MS_IDX3]),
            static_cast<COM_T>(tilingData_->indexStride[MS_IDX4]),
            static_cast<COM_T>(tilingData_->xStride[MS_IDX4]),
            static_cast<COM_T>(indexContigousStride[MS_IDX4]),
            static_cast<COM_T>(tilingData_->indexStride[MS_IDX5]),
            static_cast<COM_T>(tilingData_->xStride[MS_IDX5]),
            static_cast<COM_T>(indexContigousStride[MS_IDX5]),
            static_cast<COM_T>(tilingData_->indexStride[MS_IDX6]),
            static_cast<COM_T>(tilingData_->xStride[MS_IDX6]),
            static_cast<COM_T>(indexContigousStride[MS_IDX6]),  
            static_cast<COM_T>(tilingData_->indexStride[MS_IDX7]),
            static_cast<COM_T>(tilingData_->xStride[MS_IDX7]),      
            batchNum, 
            coreOffset);
        }
    }
}

template <typename X_T, typename INDEX_T, typename COM_T, int32_t DIM_NUM, int32_t AXIS, bool IS_SPECIAL>
__aicore__ inline void GatherElementsKernelNoContiguous<X_T, INDEX_T, COM_T, DIM_NUM, AXIS, IS_SPECIAL>::ProcessSpecial()
{
    if (GetBlockIdx() >= tilingData_->usedCore) {
        return;
    }
    if constexpr (IS_SPECIAL) {
        int64_t indexContigousStride[MAX_DIM_LEN] = {1, 1, 1, 1, 1, tilingData_->indexShape[MS_IDX5] * tilingData_->indexShape[MS_IDX7], tilingData_->indexShape[MS_IDX7], 1};
        int64_t yStride[MAX_DIM_LEN] = {1, 1, 1, 1, 1, tilingData_->indexShape[MS_IDX6] * tilingData_->indexShape[MS_IDX7], tilingData_->indexShape[MS_IDX7], 1};
        for (uint32_t i = MAX_DIM_LEN - DIM_NUM; i < M_SHIFT_OFFSET; i++) {
            GetUintDivMagicAndShift(m_[i], shift_[i], static_cast<COM_T>(indexContigousStride[i]));
        }
        DataSyncBarrier<MemDsbT::UB>();
    
        COM_T coreOffset = static_cast<COM_T>(GetBlockIdx() * tilingData_->perCoreNum);
        COM_T batchNum = static_cast<COM_T>((GetBlockIdx() == tilingData_->usedCore - 1) ? tilingData_->tailCoreNum :
            tilingData_->perCoreNum) + coreOffset;
        __gm__ INDEX_T* indexAddr = (__gm__ INDEX_T*)(indexGm_.GetPhyAddr());
        __gm__ X_T* yAddr = (__gm__ X_T*)(yGm_.GetPhyAddr());
        __gm__ X_T* xAddr = (__gm__ X_T*)(xGm_.GetPhyAddr());
    
        if constexpr (DIM_NUM == DIM3) {
            if constexpr (sizeof(COM_T) == B32) {
                Simt::VF_CALL<GatherDim3ComputeTransPose<X_T, INDEX_T, COM_T, AXIS>>(Simt::Dim3(THREAD_DIM_2048),
                    indexAddr, yAddr, xAddr, m_[MS_IDX5], shift_[MS_IDX5], m_[MS_IDX6], shift_[MS_IDX6],
                    static_cast<COM_T>(tilingData_->indexStride[MS_IDX5]),
                    static_cast<COM_T>(tilingData_->xStride[MS_IDX5]),
                    static_cast<COM_T>(indexContigousStride[MS_IDX5]),
                    static_cast<COM_T>(yStride[MS_IDX5]),
                    static_cast<COM_T>(tilingData_->indexStride[MS_IDX6]),
                    static_cast<COM_T>(tilingData_->xStride[MS_IDX6]),
                    static_cast<COM_T>(indexContigousStride[MS_IDX6]),
                    static_cast<COM_T>(yStride[MS_IDX6]),     
                    static_cast<COM_T>(tilingData_->indexStride[MS_IDX7]),
                    static_cast<COM_T>(tilingData_->xStride[MS_IDX7]),      
                    batchNum, 
                    coreOffset);
            } else {
                LocalTensor<COM_T> strideLocal = strideBuf_.Get<COM_T>();
                uint16_t preFillNum = MAX_DIM_LEN - DIM3;
                for (uint16_t i = 0; i < DIM3; i++) {
                    strideLocal(i) = static_cast<COM_T>(indexContigousStride[preFillNum + i]);
                    strideLocal(i + DIM3) = static_cast<COM_T>(tilingData_->indexStride[preFillNum + i]);
                    strideLocal(i + TWO * DIM3) = static_cast<COM_T>(yStride[preFillNum + i]);
                    strideLocal(i + THREE * DIM3) = static_cast<COM_T>(tilingData_->xStride[preFillNum + i]);
                }
                DataSyncBarrier<MemDsbT::UB>();    
                __local_mem__ COM_T* strideAddr = (__local_mem__ COM_T*)(strideLocal.GetPhyAddr()); 
                Simt::VF_CALL<GatherDim3ComputeTransPoseB64<X_T, INDEX_T, COM_T, AXIS>>(Simt::Dim3(THREAD_DIM_2048),
                    indexAddr, yAddr, xAddr, strideAddr, m_[MS_IDX5], shift_[MS_IDX5], m_[MS_IDX6], shift_[MS_IDX6],
                    batchNum, 
                    coreOffset);
            }
        }
    }
}

template <typename X_T, typename INDEX_T, typename COM_T, int32_t DIM_NUM, int32_t AXIS, bool IS_SPECIAL>
__aicore__ inline void GatherElementsKernelNoContiguous<X_T, INDEX_T, COM_T, DIM_NUM, AXIS, IS_SPECIAL>::Process()
{
    if constexpr (IS_SPECIAL) {
        ProcessSpecial();
    } else {
        ProcessCommon();
    }
}

} // namespace GatherElements
#endif // GATHER_ELEMENTS_NO_CONTIGUOUS_H