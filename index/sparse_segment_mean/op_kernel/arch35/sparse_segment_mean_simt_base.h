/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 /*!
 * \file sparse_segment_mean_simt_base.h
 * \brief
 */

#ifndef SPARSE_SEGMENT_MEAN_SIMT_BASE_H
#define SPARSE_SEGMENT_MEAN_SIMT_BASE_H

#include "kernel_operator.h"
#include "../inc/platform.h"

namespace SparseSegmentMeanNameSpace
{
using namespace AscendC;

#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 512;
#else
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 2048;
#endif

constexpr uint32_t THREAD_NUM_LAUNCH_BOUND_SMALL = 1024;

template <typename SEGMENTIDS_T>
__aicore__ inline SEGMENTIDS_T Clip(SEGMENTIDS_T id, uint32_t segmentNum)
{
    return min(
        static_cast<SEGMENTIDS_T>(max(SEGMENTIDS_T(-1), id)),
        static_cast<SEGMENTIDS_T>(segmentNum)
    );
}

template <typename SEGMENTIDS_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void SimtGetSegmentOffset(uint32_t blockId, int64_t outterSize, uint32_t blockNums, uint32_t segmentNum,
                                                                                              __gm__ uint32_t* segment_offset, __gm__ SEGMENTIDS_T* segment_ids)
{
    for (int64_t i = blockId * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < outterSize + 1;
            i = i + blockNums * Simt::GetThreadNum()) {
        
        const SEGMENTIDS_T curId = (i < outterSize) ? Clip(segment_ids[i], segmentNum) : segmentNum;
        const SEGMENTIDS_T prevId = (i == 0) ? SEGMENTIDS_T(-1) : Clip(segment_ids[i - 1], segmentNum);

        for (SEGMENTIDS_T id = prevId + 1; id <= curId; ++id) {
            segment_offset[id] = i;
        }
    }
}


template <typename X_T, typename INDICES_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void SimtLargeInnerComputer(int64_t segOffsetBase, int64_t curCoreSegments,
                                                                                                uint32_t innerSize, int64_t xDim0, __gm__ X_T* x, __gm__ volatile X_T* y,
                                                                                                __gm__ uint32_t* segment_offset, __gm__ INDICES_T* indices)
{
    for (int64_t seg = Simt::GetThreadIdx<1>(); seg < curCoreSegments; seg += Simt::GetThreadNum<1>())
    {
        int64_t globalSeg = segOffsetBase + seg;
        uint32_t begin = segment_offset[globalSeg];
        uint32_t end = segment_offset[globalSeg + 1];
        float res = 0;
        for (uint32_t idxOffset = begin; idxOffset < end; idxOffset += 1) {
            INDICES_T idx = indices[idxOffset];
            bool idxValid = (idx >= 0) && (idx < xDim0);
            int64_t inputIdx = idx * innerSize + Simt::GetThreadIdx<0>();
            float value = idxValid ? x[inputIdx] : float(0);
            res += value;
        }
        bool empty = (begin >= end);
        res = empty ? 0 : (res / (end - begin));
        int64_t outputIdx = globalSeg * innerSize + Simt::GetThreadIdx<0>();
        y[outputIdx] = static_cast<X_T>(res);
    }
}

template <typename X_T, typename INDICES_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void SimtLoopComputer(int64_t segOffsetBase, int64_t curCoreSegments,
                                                                                                uint32_t innerSize, int64_t xDim0, __gm__ X_T* x, __gm__ volatile X_T* y,
                                                                                                __gm__ uint32_t* segment_offset, __gm__ INDICES_T* indices)
{
    uint32_t threadIdxX = Simt::GetThreadIdx<0>();
    uint32_t threadIdxY = Simt::GetThreadIdx<1>();
    for (int64_t seg = threadIdxY; seg < curCoreSegments; seg += Simt::GetThreadNum<1>())
    {
        for (uint32_t curXIdx = threadIdxX; curXIdx < innerSize; curXIdx += Simt::GetThreadNum<0>()) {
            int64_t globalSeg = segOffsetBase + seg;
            uint32_t begin = segment_offset[globalSeg];
            uint32_t end = segment_offset[globalSeg + 1];
            float res = 0;
            for (uint32_t idxOffset = begin; idxOffset < end; idxOffset += 1) {
                INDICES_T idx = indices[idxOffset];
                bool idxValid = (idx >= 0) && (idx < xDim0);
                int64_t inputIdx = idx * innerSize + curXIdx;
                float value = idxValid ? x[inputIdx] : float(0);
                res += value;
            }
            bool empty = (begin >= end);
            res = empty ? 0 : (res / (end - begin));
            int64_t outputIdx = globalSeg * innerSize + curXIdx;
            y[outputIdx] = static_cast<X_T>(res);
        }
    }
}


__aicore__ inline float BinaryAdd(float value, uint32_t threadNumY, bool valid, uint32_t threadIdxY,
                                  uint32_t threadIdxX, uint32_t threadNumX, __local_mem__ float* tmpLocal, uint32_t threadIdxZ)
{
    // Binary add in the thread_y
    for (uint32_t k = threadNumY / 2; k > 0; k /= 2) {
        if (valid && threadIdxY < 2 * k) {
            tmpLocal[threadIdxZ * threadNumY * threadNumX + threadIdxY * threadNumX + threadIdxX] = value;
        }
        Simt::ThreadBarrier();

        if (valid && threadIdxY < k) {
            value += tmpLocal[threadIdxZ * threadNumY * threadNumX + (threadIdxY + k) * threadNumX + threadIdxX];
            tmpLocal[threadIdxZ * threadNumY * threadNumX + (threadIdxY + k) * threadNumX + threadIdxX] = 0;
        }
        Simt::ThreadBarrier();
    }
    return value;
}

template <typename X_T, typename INDICES_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND_SMALL) inline void SimtSmallInnerComputer(int64_t segOffsetBase, int64_t curCoreSegments, uint32_t innerSize,
                                                                                                __local_mem__ float* tmpLocal, __gm__ X_T* x, __gm__ volatile X_T* y,
                                                                                                __gm__ uint32_t* segment_offset, __gm__ INDICES_T* indices)
{
    uint32_t threadIdxX = Simt::GetThreadIdx<0>();
    uint32_t threadIdxY = Simt::GetThreadIdx<1>();
    uint32_t threadIdxZ = Simt::GetThreadIdx<2>();
    bool xValid = (threadIdxX < innerSize);
    for (int64_t seg = 0; seg < curCoreSegments; seg += Simt::GetThreadNum<2>())
    {
        int64_t globalSeg = segOffsetBase + seg + threadIdxZ;
        uint32_t begin = segment_offset[globalSeg];
        uint32_t end = segment_offset[globalSeg + 1];
        float res = 0;
        for (uint32_t yOffset = begin; yOffset < end; yOffset += Simt::GetThreadNum<1>()) {
            uint32_t idxOffset = yOffset + threadIdxY;
            bool yValid = (idxOffset < end);
            INDICES_T idx = yValid ? indices[idxOffset] : 0;
            int64_t inputIdx = idx * innerSize + threadIdxX;
            bool valid = xValid && yValid;
            float value = valid ? x[inputIdx] : float(0);

            value = BinaryAdd(value, Simt::GetThreadNum<1>(), valid, threadIdxY, threadIdxX, Simt::GetThreadNum<0>(), tmpLocal, threadIdxZ);

            // thready_0 is the reduce sum
            if (threadIdxY == 0 && xValid) {
                res += value;
            }
        }
        if (threadIdxY == 0 && xValid) {
            bool empty = (begin >= end);
            res = empty ? 0 : (res / (end - begin));
            int64_t outputIdx = globalSeg * innerSize + threadIdxX;
            y[outputIdx] = static_cast<X_T>(res);
        }
    }
}


}  // namespace SparseSegmentMeanNameSpace
#endif  // SPARSE_SEGMENT_MEAN_SIMT_BASE_H