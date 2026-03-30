/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 /*!
 * \file sorted_sparse_segment_mean_grad_simt_base.h
 * \brief
 */

#ifndef SORTED_SPARSE_SEGMENT_MEAN_GRAD_SIMT_BASE_H
#define SORTED_SPARSE_SEGMENT_MEAN_GRAD_SIMT_BASE_H

#include "kernel_operator.h"
#include "../inc/platform.h"

namespace SparseSegmentMeanGradNameSpace
{
using namespace AscendC;

#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 512;
constexpr uint32_t SIMPLE_THREAD_NUM_LAUNCH_BOUND = 512;
#else
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 1024;
constexpr uint32_t SIMPLE_THREAD_NUM_LAUNCH_BOUND = 2048;
#endif


template <typename SEGMENTIDS_T>
__simt_callee__ __aicore__ inline SEGMENTIDS_T Clip(SEGMENTIDS_T id, SEGMENTIDS_T segmentNum)
{
    return min(
        static_cast<SEGMENTIDS_T>(max(SEGMENTIDS_T(-1), id)),
        static_cast<SEGMENTIDS_T>(segmentNum)
    );
}

template <typename SEGMENTIDS_T, typename OUTTER_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMPLE_THREAD_NUM_LAUNCH_BOUND) inline void SimtGetSegmentOffset(uint32_t blockId, OUTTER_T outterSize, uint32_t blockNums, SEGMENTIDS_T segmentNum,
                                                                                              __gm__ OUTTER_T* segment_offset, __gm__ SEGMENTIDS_T* segment_ids)
{
    for (OUTTER_T i = blockId * Simt::GetThreadNum<1>() + Simt::GetThreadIdx<1>(); i < outterSize + 1;
            i = i + blockNums * Simt::GetThreadNum<1>()) {
        
        const SEGMENTIDS_T curId = (i < outterSize) ? Clip(segment_ids[i], segmentNum) : segmentNum;
        const SEGMENTIDS_T prevId = (i == 0) ? SEGMENTIDS_T(-1) : Clip(segment_ids[i - 1], segmentNum);

        for (SEGMENTIDS_T id = prevId + 1 + Simt::GetThreadIdx<0>(); id <= curId; id = id + Simt::GetThreadNum<0>()) {
            segment_offset[id] = i;
        }
    }
}

template <typename SEGMENTIDS_T, typename OUTTER_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMPLE_THREAD_NUM_LAUNCH_BOUND) inline void SimtCalcWeight(uint32_t blockId, uint32_t blockNums, SEGMENTIDS_T segmentNum,
                                                                                        __gm__ OUTTER_T* segment_offset, __gm__ float* weight)
{
    if (Simt::GetThreadIdx() == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
    __syncthreads();
    for (SEGMENTIDS_T i = blockId * Simt::GetThreadNum() + Simt::GetThreadIdx() + 1; i < segmentNum + 1;
            i = i + blockNums * Simt::GetThreadNum()) {
        OUTTER_T seg_size = segment_offset[i] - segment_offset[i - 1];
        seg_size = (seg_size > 0) ? seg_size : 1;
        weight[i - 1] = static_cast<float>(1.0) / static_cast<float>(seg_size);
    }
}

template <typename X_T, typename LOCATION_T, typename SEGMENTIDS_T, typename OUTTER_T, typename INNER_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void SimtLargeInnerComputer(uint32_t indicesBase, uint32_t curCoreIndices, uint32_t threadNumY,
                                                                                                INNER_T innerSize, SEGMENTIDS_T segmentNum, __gm__ X_T* x, __gm__ volatile X_T* y,
                                                                                                __gm__ OUTTER_T* indices_offset, __gm__ SEGMENTIDS_T* segmentIds,
                                                                                                __gm__ LOCATION_T* location, __gm__ float* weight, uint32_t outputDim0,
                                                                                                INNER_T srcColOffset, INNER_T processCols)
{
    if (Simt::GetThreadIdx<0>() + Simt::GetThreadIdx<1>() == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
    __syncthreads();
    uint32_t index = 0;
    for (; index < curCoreIndices; index += threadNumY) {
        uint32_t globalIndex = indicesBase + index;
        if (globalIndex >= outputDim0) {
            continue;
        }
        OUTTER_T begin = static_cast<OUTTER_T>(indices_offset[globalIndex]);
        OUTTER_T end = static_cast<OUTTER_T>(indices_offset[globalIndex + 1]);
        INNER_T colOffset = Simt::GetThreadIdx<0>() + srcColOffset;
        INNER_T colStride = Simt::GetThreadNum<0>();
        for (; colOffset < processCols; colOffset += colStride) {
            float res = 0;
            for (OUTTER_T locaOffset = begin; locaOffset < end; locaOffset += 1) {
                SEGMENTIDS_T seg = segmentIds[location[locaOffset]];
                bool segValid = (seg >= 0) && (seg < segmentNum);
                int64_t inputIdx = seg * static_cast<int64_t>(innerSize) + static_cast<int64_t>(colOffset);
                float value = segValid ? static_cast<float>(x[inputIdx]) : float(0);
                res += segValid ? (value * weight[seg]) : float(0);
            }
            bool empty = (begin >= end);
            res = empty ? static_cast<X_T>(0) : res;
            int64_t outputIdx = static_cast<int64_t>(globalIndex) * static_cast<int64_t>(innerSize) + static_cast<int64_t>(colOffset);
            y[outputIdx] = static_cast<X_T>(res);
        }
    }
}

template <typename X_T, typename LOCATION_T, typename SEGMENTIDS_T, typename OUTTER_T, typename INNER_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void SimtLargeMinInnerComputer(uint32_t indicesBase, uint32_t curCoreIndices, INNER_T innerSize, 
                                                                                                  SEGMENTIDS_T segmentNum, __gm__ X_T* x, __gm__ volatile X_T* y,
                                                                                                  __gm__ OUTTER_T* indices_offset, __gm__ SEGMENTIDS_T* segmentIds,
                                                                                                  __gm__ LOCATION_T* location, __gm__ float* weight, uint32_t outputDim0,
                                                                                                  INNER_T srcColOffset)
{
    if (Simt::GetThreadIdx<0>() + Simt::GetThreadIdx<1>() == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
    __syncthreads();
    for (uint32_t index = Simt::GetThreadIdx<1>(); index < curCoreIndices; index += Simt::GetThreadNum<1>()) {
        uint32_t globalIndex = indicesBase + index;
        if (globalIndex >= outputDim0) {
            continue;
        }
        OUTTER_T begin = static_cast<OUTTER_T>(indices_offset[globalIndex]);
        OUTTER_T end = static_cast<OUTTER_T>(indices_offset[globalIndex + 1]);
        float res = 0;
        for (OUTTER_T locaOffset = begin; locaOffset < end; locaOffset += 1) {
            SEGMENTIDS_T seg = segmentIds[location[locaOffset]];
            bool segValid = (seg >= 0) && (seg < segmentNum);
            int64_t inputIdx = seg * static_cast<int64_t>(innerSize) + static_cast<int64_t>(Simt::GetThreadIdx<0>()) + static_cast<int64_t>(srcColOffset);
            float value = segValid ? static_cast<float>(x[inputIdx]) : float(0);
            res += segValid ? (value * weight[seg]) : float(0);
        }
        bool empty = (begin >= end);
        res = empty ? static_cast<X_T>(0) : res;
        int64_t outputIdx = static_cast<int64_t>(globalIndex) * static_cast<int64_t>(innerSize) + static_cast<int64_t>(Simt::GetThreadIdx<0>()) + static_cast<int64_t>(srcColOffset);
        y[outputIdx] = static_cast<X_T>(res);
    }
}

template <typename X_T, typename LOCATION_T, typename SEGMENTIDS_T, typename OUTTER_T, typename INNER_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void SimtLargeUBInnerComputer(uint32_t indicesBase, uint32_t curCoreIndices,
                                                                                                INNER_T innerSize, SEGMENTIDS_T segmentNum, __gm__ X_T* x, __gm__ volatile X_T* y,
                                                                                                __gm__ OUTTER_T* indices_offset, __gm__ SEGMENTIDS_T* segmentIds,
                                                                                                __gm__ LOCATION_T* location, __gm__ float* weight, uint32_t outputDim0,
                                                                                                __local_mem__ float* tmpLocal, INNER_T srcColOffset, INNER_T processCols)
{
    if (Simt::GetThreadIdx<0>() + Simt::GetThreadIdx<1>() == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
    __syncthreads();
    for (uint32_t index = 0; index < curCoreIndices; index++) {
        uint32_t globalIndex = indicesBase + index;
        if (globalIndex >= outputDim0) {
            continue;
        }
        OUTTER_T begin = static_cast<OUTTER_T>(indices_offset[globalIndex]);
        OUTTER_T end = static_cast<OUTTER_T>(indices_offset[globalIndex + 1]);
        for (OUTTER_T locaOffset = begin; locaOffset < end; locaOffset += 1) {
            SEGMENTIDS_T seg = segmentIds[location[locaOffset]];
            bool segValid = (seg >= 0) && (seg < segmentNum);
            INNER_T colOffset = Simt::GetThreadIdx<0>() + srcColOffset;
            INNER_T colStride = Simt::GetThreadNum<0>();
            for (; colOffset < processCols; colOffset += colStride) {
                int64_t inputIdx = seg * static_cast<int64_t>(innerSize) + static_cast<int64_t>(colOffset);
                float value = segValid ? static_cast<float>(x[inputIdx]) : float(0);
                tmpLocal[colOffset] += segValid ? (value * weight[seg]) : float(0);
            }
        }
        bool empty = (begin >= end);
        INNER_T colOffset = Simt::GetThreadIdx<0>() + srcColOffset;
        INNER_T colStride = Simt::GetThreadNum<0>();
        if (empty) {
            for (; colOffset < processCols; colOffset += colStride) {
                int64_t outputIdx = static_cast<int64_t>(globalIndex) * static_cast<int64_t>(innerSize) + static_cast<int64_t>(colOffset);
                y[outputIdx] = static_cast<X_T>(0);
            }
        } else {
            for (; colOffset < processCols; colOffset += colStride) {
                int64_t outputIdx = static_cast<int64_t>(globalIndex) * static_cast<int64_t>(innerSize) + static_cast<int64_t>(colOffset);
                y[outputIdx] = static_cast<X_T>(tmpLocal[colOffset]);
                tmpLocal[colOffset] = 0.0f;
            }
        }
    }
}

// 这里进行二分累加，实际是按y线程数量按照 inner逐列累加，按 0与(numY/2)处依次累加
__simt_callee__ __aicore__ inline float BinaryAdd(float value, uint32_t threadNumY, bool valid, uint32_t threadIdxY,
                                  uint32_t threadIdxX, uint32_t threadNumX, __local_mem__ float* tmpLocal)
{
    // Binary add in the thread_y
    for (uint32_t k = threadNumY / 2; k > 0; k /= 2) {
        if (valid && threadIdxY < 2 * k) {
            tmpLocal[threadIdxY * threadNumX + threadIdxX] = value;
        }
        Simt::ThreadBarrier();

        if (valid && threadIdxY < k) {
            value += tmpLocal[(threadIdxY + k) * threadNumX + threadIdxX];
            tmpLocal[(threadIdxY + k) * threadNumX + threadIdxX] = 0;
        }
        Simt::ThreadBarrier();
    }
    return value;
}

template <typename X_T, typename LOCATION_T, typename SEGMENTIDS_T, typename OUTTER_T, typename INNER_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void SimtSmallInnerComputer(uint32_t indicesBase, uint32_t curCoreIndices, uint32_t threadNumY, INNER_T innerSize,
                                                                                                SEGMENTIDS_T segmentNum, uint32_t threadNumX, __local_mem__ float* tmpLocal, __gm__ X_T* x, 
                                                                                                __gm__ volatile X_T* y, __gm__ OUTTER_T* indices_offset, __gm__ SEGMENTIDS_T* segmentIds,
                                                                                                __gm__ LOCATION_T* location, __gm__ float* weight, uint32_t outputDim0)
{
    if (Simt::GetThreadIdx<0>() + Simt::GetThreadIdx<1>() == 0) {
        __builtin_cce_dcci(nullptr, 1, 0);
    }
    __syncthreads();
    uint32_t threadIdxY = Simt::GetThreadIdx<1>();
    uint32_t threadIdxX = Simt::GetThreadIdx<0>();
    bool xValid = (threadIdxX < innerSize);
    for (uint32_t index = 0; index < curCoreIndices; index += 1) {
        uint32_t globalIndex = indicesBase + index;
        if (globalIndex >= outputDim0) {
            continue;
        }
        OUTTER_T begin = static_cast<OUTTER_T>(indices_offset[globalIndex]);
        OUTTER_T end = static_cast<OUTTER_T>(indices_offset[globalIndex + 1]);
        float res = 0;
        for (OUTTER_T yOffset = begin; yOffset < end; yOffset += threadNumY) {
            OUTTER_T locaOffset = yOffset + threadIdxY;
            bool yValid = (locaOffset < end);
            SEGMENTIDS_T seg = yValid ? segmentIds[location[locaOffset]] : 0;
            bool segValid = (seg >= 0) && (seg < segmentNum);
            int64_t inputIdx = seg * static_cast<int64_t>(innerSize) + static_cast<int64_t>(threadIdxX);
            bool valid = xValid && yValid && segValid;
            float value = valid ? static_cast<float>(x[inputIdx]) * weight[seg] : float(0);

            value = BinaryAdd(value, threadNumY, valid, threadIdxY, threadIdxX, threadNumX, tmpLocal);

            // thready_0 is the reduce sum
            if (threadIdxY == 0 && xValid) {
                res += value;
            }
        }
        if (threadIdxY == 0 && xValid) {
            bool empty = (begin >= end);
            res = empty ? 0 : res;
            int64_t outputIdx = static_cast<int64_t>(globalIndex) * static_cast<int64_t>(innerSize) + static_cast<int64_t>(threadIdxX);
            y[outputIdx] = static_cast<X_T>(res);
        }
    }
}

}  // namespace SparseSegmentMeanGradNameSpace
#endif  // SORTED_SPARSE_SEGMENT_MEAN_GRAD_SIMT_BASE_H