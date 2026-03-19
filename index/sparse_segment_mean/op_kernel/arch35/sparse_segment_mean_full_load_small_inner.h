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
 * \file sparse_segment_mean_full_load_small_inner.h
 * \brief
 */

#ifndef SPARSE_SEGMENT_MEAN_FULL_LOAD_SMALL_INNER_H
#define SPARSE_SEGMENT_MEAN_FULL_LOAD_SMALL_INNER_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "sparse_segment_mean_struct.h"
#include "sparse_segment_mean_simt_base.h"
#include "kernel_operator.h"

namespace SparseSegmentMeanNameSpace {
using namespace AscendC;

#ifdef __DAV_FPGA__
constexpr uint32_t FULL_LOAD_SMALL_INNER_THREAD_NUM = 512;
#else
constexpr uint32_t FULL_LOAD_SMALL_INNER_THREAD_NUM = 1024;
#endif

constexpr uint32_t TBUF_SIZE = 4096;

template <typename X_T, typename INDICES_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(FULL_LOAD_SMALL_INNER_THREAD_NUM) inline void FullLoadBinaryAddComputer(
    int64_t segOffsetBase, int64_t curCoreSegments, uint32_t innerSize,
    __local_mem__ float* tmpLocal, __local_mem__ X_T* xLocal, __gm__ volatile X_T* y, __gm__ uint32_t* segment_offset,
    __local_mem__ INDICES_T* indicesTensor)
{
    uint32_t threadIdxX = Simt::GetThreadIdx<0>();
    uint32_t threadIdxY = Simt::GetThreadIdx<1>();
    uint32_t threadIdxZ = Simt::GetThreadIdx<2>();
    bool xValid = (threadIdxX < innerSize);
    for (int64_t seg = 0; seg < curCoreSegments; seg += Simt::GetThreadNum<2>()) {
        int64_t globalSeg = segOffsetBase + seg + threadIdxZ;
        uint32_t begin = segment_offset[globalSeg];
        uint32_t end = segment_offset[globalSeg + 1];
        float res = 0;
        for (uint32_t yOffset = begin; yOffset < end; yOffset += Simt::GetThreadNum<1>()) {
            uint32_t idxOffset = yOffset + threadIdxY;
            bool yValid = (idxOffset < end);
            INDICES_T idx = yValid ? indicesTensor[idxOffset] : 0;
            int64_t inputIdx = idx * innerSize + threadIdxX;
            bool valid = xValid && yValid;
            float value = valid ? xLocal[inputIdx] : float(0);

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


template <typename X_T, typename INDICES_T, typename SEGMENTIDS_T>
class SparseSegmentMeanFullLoadSmallInner
{
public:
    __aicore__ inline SparseSegmentMeanFullLoadSmallInner(const SparseSegmentMeanFullLoadTilingData& tilingData, TPipe& pipe) :
        tilingData_(tilingData), pipe_(pipe) {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR segment_ids, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void Process();
    __aicore__ inline void CopyInX(LocalTensor<X_T>& xLocal);
    __aicore__ inline void CopyInIndices(LocalTensor<INDICES_T>& indicesLocal);

private:
    TPipe& pipe_;
    const SparseSegmentMeanFullLoadTilingData& tilingData_;

    GlobalTensor<X_T> xGm_;
    GlobalTensor<INDICES_T> indicesGm_;
    GlobalTensor<SEGMENTIDS_T> segmentIdsGm_;
    GlobalTensor<X_T> yGm_;
    GlobalTensor<uint32_t> workspaceSegmentOffset_;

    TBuf<QuePosition::VECCALC> xBuf_;
    TBuf<QuePosition::VECCALC> indicesBuf_;
    TBuf<QuePosition::VECCALC> tmpBuf_;

    int64_t segOffsetBase_ = 0;
    int64_t curCoreSegments_ = 0;
    uint32_t segmentNum_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t blockNums_ = 0;
    uint32_t threadNumZ_ = 0;
};

template <typename X_T, typename INDICES_T, typename SEGMENTIDS_T>
__aicore__ inline void SparseSegmentMeanFullLoadSmallInner<X_T, INDICES_T, SEGMENTIDS_T>::Init(
    GM_ADDR x, GM_ADDR indices, GM_ADDR segment_ids, GM_ADDR y, GM_ADDR workspace)
{
    xGm_.SetGlobalBuffer((__gm__ X_T*)x);
    indicesGm_.SetGlobalBuffer((__gm__ INDICES_T*)indices);
    segmentIdsGm_.SetGlobalBuffer((__gm__ SEGMENTIDS_T*)segment_ids);
    yGm_.SetGlobalBuffer((__gm__ X_T*)y);
    workspaceSegmentOffset_.SetGlobalBuffer((__gm__ uint32_t*)workspace);
    pipe_.InitBuffer(xBuf_, tilingData_.xBufferSize);
    pipe_.InitBuffer(indicesBuf_, tilingData_.indicesBufferSize);
    pipe_.InitBuffer(tmpBuf_, TBUF_SIZE);

    segmentNum_ = static_cast<uint32_t>(tilingData_.segmentNum);
    blockIdx_ = static_cast<uint32_t>(GetBlockIdx());
    blockNums_ = static_cast<uint32_t>(GetBlockNum());

    // calc core offset
    if (tilingData_.specialBlockTiling) {
        if (tilingData_.secondToLastCoreSegmentNum == 0) {
            segOffsetBase_ = tilingData_.normalCoreSegmentNum * blockIdx_;
            curCoreSegments_ = blockIdx_ == tilingData_.needCoreNum - 1
                ? tilingData_.lastCoreSegmentNum
                : tilingData_.normalCoreSegmentNum;
        } else {
            if (blockIdx_ == tilingData_.needCoreNum - 1) {
                segOffsetBase_ = tilingData_.normalCoreSegmentNum * (blockIdx_ - 1) + tilingData_.secondToLastCoreSegmentNum;
                curCoreSegments_ = tilingData_.lastCoreSegmentNum;
            } else {
                segOffsetBase_ = tilingData_.normalCoreSegmentNum * blockIdx_;
                curCoreSegments_ = blockIdx_ == tilingData_.needCoreNum - 2
                    ? tilingData_.secondToLastCoreSegmentNum
                    : tilingData_.normalCoreSegmentNum;
            }
        }
        threadNumZ_ = blockIdx_ == tilingData_.needCoreNum - 1 ? tilingData_.lastCoreSegmentNum : 16;
    } else {
        if (blockIdx_ < tilingData_.resSegmentNum) {
            segOffsetBase_ = (tilingData_.perCoreSegmentNum + 1) * blockIdx_;
            curCoreSegments_ = tilingData_.perCoreSegmentNum + 1;
        } else {
            segOffsetBase_ = tilingData_.perCoreSegmentNum * blockIdx_ + tilingData_.resSegmentNum;
            curCoreSegments_ = tilingData_.perCoreSegmentNum;
        }
        threadNumZ_ = curCoreSegments_;
    }
}

template <typename X_T, typename INDICES_T, typename SEGMENTIDS_T>
__aicore__ inline void SparseSegmentMeanFullLoadSmallInner<X_T, INDICES_T, SEGMENTIDS_T>::CopyInX(LocalTensor<X_T>& xLocal)
{
    DataCopyPadExtParams<X_T> dataCopyPadExtParams = {false, 0, 0, 0};
    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = 1;
    dataCoptExtParams.blockLen = tilingData_.xBufferSize;
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;

    DataCopyPad(xLocal, xGm_, dataCoptExtParams, dataCopyPadExtParams);
}

template <typename X_T, typename INDICES_T, typename SEGMENTIDS_T>
__aicore__ inline void SparseSegmentMeanFullLoadSmallInner<X_T, INDICES_T, SEGMENTIDS_T>::CopyInIndices(
    LocalTensor<INDICES_T>& indicesLocal)
{
    DataCopyPadExtParams<INDICES_T> dataCopyPadExtParams = {false, 0, 0, 0};
    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = 1;
    dataCoptExtParams.blockLen = tilingData_.indicesBufferSize;
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;

    DataCopyPad(indicesLocal, indicesGm_, dataCoptExtParams, dataCopyPadExtParams);
}

template <typename X_T, typename INDICES_T, typename SEGMENTIDS_T>
__aicore__ inline void SparseSegmentMeanFullLoadSmallInner<X_T, INDICES_T, SEGMENTIDS_T>::Process()
{
    if (blockIdx_ >= blockNums_) {
        return;
    }

    int64_t outterSize = tilingData_.outterSize;
    uint32_t threadNumX = static_cast<uint32_t>(tilingData_.threadNumX);
    uint32_t threadNumY = static_cast<uint32_t>(tilingData_.threadNumY);
    uint32_t innerSize = static_cast<uint32_t>(tilingData_.innerSize);

    AscendC::Simt::VF_CALL<SimtGetSegmentOffset<SEGMENTIDS_T>>(
        Simt::Dim3(MAX_THREAD_NUM), blockIdx_, outterSize, blockNums_, segmentNum_,
        (__gm__ uint32_t*)(workspaceSegmentOffset_.GetPhyAddr()), (__gm__ SEGMENTIDS_T*)(segmentIdsGm_.GetPhyAddr()));

    SyncAll();

    LocalTensor<X_T> xLocal = xBuf_.Get<X_T>();
    CopyInX(xLocal);

    LocalTensor<INDICES_T> indicesTensor = indicesBuf_.Get<INDICES_T>();
    CopyInIndices(indicesTensor);

    event_t eventIdMTE2toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2toV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2toV);

    LocalTensor<float> tmpLocal = tmpBuf_.Get<float>();
    Duplicate<float>(tmpLocal, 0, TBUF_SIZE / sizeof(float));

    AscendC::Simt::VF_CALL<FullLoadBinaryAddComputer<X_T, INDICES_T>>(
        Simt::Dim3{threadNumX, threadNumY, threadNumZ_}, segOffsetBase_, curCoreSegments_, innerSize,
        (__local_mem__ float*)(tmpLocal.GetPhyAddr()), (__local_mem__ X_T*)(xLocal.GetPhyAddr()),
        (__gm__ volatile X_T*)(yGm_.GetPhyAddr()), (__gm__ uint32_t*)(workspaceSegmentOffset_.GetPhyAddr()),
        (__local_mem__ INDICES_T*)(indicesTensor.GetPhyAddr()));
}

} // namespace SparseSegmentMeanNameSpace
#endif // SPARSE_SEGMENT_MEAN_FULL_LOAD_SMALL_INNER_H