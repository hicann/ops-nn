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
 * \file gather_elements.h
 * \brief
 */
#ifndef GATHER_ELEMENTS_KERNEL_H_
#define GATHER_ELEMENTS_KERNEL_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "gather_elements_v2_tiling_data.h"

#include <type_traits>

constexpr int64_t Tiling_MODE_X_SMALL_INDICES_LARGE = 2;
constexpr int64_t Tiling_MODE_X_SLICE_INDICES_LARGE = 3;
constexpr int64_t Tiling_MODE_X_SMALL_INDICES_LARGE_DIFF = 5;
constexpr int64_t Tiling_MODE_X_SLICE_INDICES_LARGE_DIFF = 6;
constexpr int64_t Tiling_MODE_FOR_LAST_AXIS = 7;
constexpr int64_t Tiling_MODE_FOR_LAST_AXIS_GATHER = 8;
constexpr int64_t Tiling_MODE_FOR_LAST_AXIS_DIFF_SHAPE = 9;
constexpr int64_t Tiling_MODE_FOR_LAST_AXIS_CUT_GATHER = 10;

constexpr int32_t BLOCK_SIZE = 32;

struct GatherElementsTilingData {
    int64_t tilingMode;
    int64_t axis;
    int64_t params_pre;
    int64_t params_axis;
    int64_t params_row;
    int64_t params_total;
    int64_t need_core_num;
    int64_t indices_num;
    int64_t indices_axis;
    int64_t indices_num_each_core;
    int64_t indices_num_remaining;
    int64_t indices_loop_num;
    int64_t indices_row_num_once;
    int64_t indices_row_num_last;
    int64_t remaining_block_remain;
    int64_t remaining_block_num;
    int64_t slice_thickness_once;
    int64_t slice_num;
    int64_t slice_thickness_last;
    int64_t indices_slice_thickness_dim1;
    int64_t indices_slice_thickness_dim1_last;
    int64_t indices_slice_num_dim1;
    int64_t params_shape[8];
    int64_t indices_shape[8];
    int64_t dims;
    int64_t repeat_per_core;
    int64_t rounds;
    int64_t rounds_tail;
    int64_t dbFlag;
    int64_t useV2;
    int64_t v2Mode;
    GatherElementsV2TilingData v2Data;
};

template <typename T>
__aicore__ inline constexpr bool IsGatherSupportedDtype()
{
    return std::is_same<T, int16_t>::value || std::is_same<T, uint16_t>::value || std::is_same<T, half>::value ||
           std::is_same<T, bfloat16_t>::value || std::is_same<T, int32_t>::value || std::is_same<T, uint32_t>::value ||
           std::is_same<T, float>::value;
}

template <typename T>
__aicore__ inline T DivCeil(T val, T div)
{
    return (val + div - 1) / div;
}

namespace AscendC {
template <typename X_T, typename INDEX_T>
class GatherElementsKernel {
public:
    __aicore__ inline GatherElementsKernel(GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR workspace,
                                           const GatherElementsTilingData& tiling, TPipe& pipe)
    {
        pipe_ = &pipe;
        InitParams(tiling);
        SetGmAddr(x, index, y, workspace);
    }

    __aicore__ inline void Process()
    {
        if (tilingMode_ >= Tiling_MODE_FOR_LAST_AXIS && tilingMode_ <= Tiling_MODE_FOR_LAST_AXIS_CUT_GATHER) {
            ProcessLastAxis();
            return;
        }
        ProcessBatch();
    }

private:
    GlobalTensor<X_T> xGm_;
    GlobalTensor<INDEX_T> indexGm_;
    GlobalTensor<X_T> yGm_;
    TPipe* pipe_;
    int64_t tilingMode_ = 0;
    int64_t axis_ = 0;
    int64_t paramsPre_ = 0;
    int64_t paramsAxis_ = 0;
    int64_t paramsRow_ = 0;
    int64_t paramsTotal_ = 0;
    int64_t needCoreNum_ = 0;
    int64_t indicesNum_ = 0;
    int64_t indicesAxis_ = 0;
    int64_t indicesNumEachCore_ = 0;
    int64_t indicesNumRemaining_ = 0;
    int64_t indicesLoopNum_ = 0;
    int64_t indicesRowNumOnce_ = 0;
    int64_t indicesRowNumLast_ = 0;
    int64_t remainingBlockRemain_ = 0;
    int64_t remainingBlockNum_ = 0;
    int64_t sliceThicknessOnce_ = 0;
    int64_t sliceNum_ = 0;
    int64_t sliceThicknessLast_ = 0;
    int64_t indicesSliceThicknessDim1_ = 0;
    int64_t indicesSliceThicknessDim1Last_ = 0;
    int64_t indicesSliceNumDim1_ = 0;
    int64_t paramsShape_[8] = {0};
    int64_t indicesShape_[8] = {0};
    int64_t dims_ = 0;
    int64_t repeatPerCore_ = 0;
    int64_t rounds_ = 0;
    int64_t roundsTail_ = 0;
    int64_t dbFlag_ = 0;

    __aicore__ inline void InitParams(const GatherElementsTilingData& tiling)
    {
        tilingMode_ = tiling.tilingMode;
        axis_ = tiling.axis;
        paramsPre_ = tiling.params_pre;
        paramsAxis_ = tiling.params_axis;
        paramsRow_ = tiling.params_row;
        paramsTotal_ = tiling.params_total;
        needCoreNum_ = tiling.need_core_num;
        indicesNum_ = tiling.indices_num;
        indicesAxis_ = tiling.indices_axis;
        indicesNumEachCore_ = tiling.indices_num_each_core;
        indicesNumRemaining_ = tiling.indices_num_remaining;
        indicesLoopNum_ = tiling.indices_loop_num;
        indicesRowNumOnce_ = tiling.indices_row_num_once;
        indicesRowNumLast_ = tiling.indices_row_num_last;
        remainingBlockRemain_ = tiling.remaining_block_remain;
        remainingBlockNum_ = tiling.remaining_block_num;
        sliceThicknessOnce_ = tiling.slice_thickness_once;
        sliceNum_ = tiling.slice_num;
        sliceThicknessLast_ = tiling.slice_thickness_last;
        indicesSliceThicknessDim1_ = tiling.indices_slice_thickness_dim1;
        indicesSliceThicknessDim1Last_ = tiling.indices_slice_thickness_dim1_last;
        indicesSliceNumDim1_ = tiling.indices_slice_num_dim1;
        for (int32_t i = 0; i < 8; i++) {
            paramsShape_[i] = tiling.params_shape[i];
            indicesShape_[i] = tiling.indices_shape[i];
        }
        dims_ = tiling.dims;
        repeatPerCore_ = tiling.repeat_per_core;
        rounds_ = tiling.rounds;
        roundsTail_ = tiling.rounds_tail;
        dbFlag_ = tiling.dbFlag;
    }

    __aicore__ inline void SetGmAddr(GM_ADDR x, GM_ADDR index, GM_ADDR y, GM_ADDR workspace)
    {
        (void)workspace;
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_T*>(x));
        indexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ INDEX_T*>(index));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ X_T*>(y));
    }

    __aicore__ inline void SyncVtoM2()
    {
        event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
    }

    __aicore__ inline void SyncVtoM3()
    {
        event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void SyncM3toV()
    {
        event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
    }

    __aicore__ inline void ProcessBatch()
    {
        bool isDiffShape = (tilingMode_ > Tiling_MODE_X_SLICE_INDICES_LARGE);
        int64_t paramsAxis = paramsAxis_;
        int64_t paramsRow = paramsRow_;
        int64_t indicesAxis = indicesAxis_;

        if (tilingMode_ == Tiling_MODE_X_SMALL_INDICES_LARGE || tilingMode_ == Tiling_MODE_X_SMALL_INDICES_LARGE_DIFF) {
            ProcessSmallX(paramsAxis, paramsRow, indicesAxis, isDiffShape);
        } else if (tilingMode_ == Tiling_MODE_X_SLICE_INDICES_LARGE ||
                   tilingMode_ == Tiling_MODE_X_SLICE_INDICES_LARGE_DIFF) {
            ProcessSlicedX(paramsAxis, paramsRow, indicesAxis, isDiffShape);
        } else {
            ProcessLargeX(paramsAxis, paramsRow, indicesAxis, isDiffShape);
        }
    }

    __aicore__ inline void ProcessSmallX(int64_t paramsAxis, int64_t paramsRow, int64_t indicesAxis, bool isDiffShape)
    {
        int32_t blockIdx = GetBlockIdx();

        int64_t xTotal = paramsTotal_;
        uint32_t xAligned = DivCeil(xTotal, BLOCK_SIZE / static_cast<int32_t>(sizeof(X_T))) *
                            (BLOCK_SIZE / static_cast<int32_t>(sizeof(X_T)));

        TBuf<TPosition::VECCALC> xBuf;
        pipe_->InitBuffer(xBuf, xAligned * sizeof(X_T));
        auto xUb = xBuf.Get<X_T>();
        DataCopyPadExtParams<X_T> padParams = {true, 0, 0, static_cast<X_T>(0)};
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(xAligned * sizeof(X_T)), 0, 0, 0};
        DataCopyPad(xUb, xGm_, copyParams, padParams);
        PipeBarrier<PIPE_ALL>();

        int64_t paramSmallerThanIndices = sizeof(INDEX_T) / sizeof(X_T) > 0 ? sizeof(INDEX_T) / sizeof(X_T) : 1;
        int64_t indicesBlockNumLarge = paramSmallerThanIndices * (BLOCK_SIZE / static_cast<int64_t>(sizeof(INDEX_T)));

        int64_t maxChunkCount = indicesRowNumOnce_ > indicesBlockNumLarge ? indicesRowNumOnce_ : indicesBlockNumLarge;
        int64_t blockSizeIdx = BLOCK_SIZE / static_cast<int32_t>(sizeof(INDEX_T));
        int64_t blockSizeX = BLOCK_SIZE / static_cast<int32_t>(sizeof(X_T));
        uint32_t idxAlignedMax = DivCeil(maxChunkCount, blockSizeIdx) * blockSizeIdx;
        uint32_t resAlignedMax = DivCeil(maxChunkCount, blockSizeX) * blockSizeX;
        TBuf<TPosition::VECCALC> idxBuf;
        TBuf<TPosition::VECCALC> resBuf;
        pipe_->InitBuffer(idxBuf, idxAlignedMax * sizeof(INDEX_T));
        pipe_->InitBuffer(resBuf, resAlignedMax * sizeof(X_T));
        auto idxUb = idxBuf.Get<INDEX_T>();
        auto resUb = resBuf.Get<X_T>();

        uint32_t idxRemAligned = DivCeil(indicesBlockNumLarge, blockSizeIdx) * blockSizeIdx;
        uint32_t resRemAligned = DivCeil(indicesBlockNumLarge, blockSizeX) * blockSizeX;
        TBuf<TPosition::VECCALC> idxRemBuf;
        TBuf<TPosition::VECCALC> resRemBuf;
        pipe_->InitBuffer(idxRemBuf, idxRemAligned * sizeof(INDEX_T));
        pipe_->InitBuffer(resRemBuf, resRemAligned * sizeof(X_T));
        auto idxRemUb = idxRemBuf.Get<INDEX_T>();
        auto resRemUb = resRemBuf.Get<X_T>();

        int64_t coreBase = blockIdx * indicesNumEachCore_;
        for (int64_t loop = 0; loop < indicesLoopNum_; loop++) {
            ProcessSmallXChunk(xUb, idxUb, resUb, paramsAxis, paramsRow, indicesAxis, isDiffShape,
                               coreBase + loop * indicesRowNumOnce_, indicesRowNumOnce_);
        }
        if (indicesRowNumLast_ > 0) {
            ProcessSmallXChunk(xUb, idxUb, resUb, paramsAxis, paramsRow, indicesAxis, isDiffShape,
                               coreBase + indicesLoopNum_ * indicesRowNumOnce_, indicesRowNumLast_);
        }

        if (indicesNumRemaining_ > 0 && blockIdx < remainingBlockNum_) {
            int64_t off = indicesNumEachCore_ * needCoreNum_ + blockIdx * indicesBlockNumLarge;
            ProcessSmallXChunk(xUb, idxRemUb, resRemUb, paramsAxis, paramsRow, indicesAxis, isDiffShape, off,
                               indicesBlockNumLarge);
        }
        if (indicesNumRemaining_ > 0 && blockIdx == remainingBlockNum_ && remainingBlockRemain_ > 0) {
            int64_t off = indicesNumEachCore_ * needCoreNum_ + blockIdx * indicesBlockNumLarge;
            ProcessSmallXChunk(xUb, idxRemUb, resRemUb, paramsAxis, paramsRow, indicesAxis, isDiffShape, off,
                               remainingBlockRemain_);
        }
    }

    __aicore__ inline void ProcessSmallXChunk(const LocalTensor<X_T>& xUb, const LocalTensor<INDEX_T>& idxUb,
                                              const LocalTensor<X_T>& resUb, int64_t paramsAxis, int64_t paramsRow,
                                              int64_t indicesAxis, bool isDiffShape, int64_t startOffset, int64_t count)
    {
        if (count <= 0)
            return;

        PipeBarrier<PIPE_ALL>();
        {
            DataCopyPadExtParams<INDEX_T> padParams = {true, 0, 0, static_cast<INDEX_T>(0)};
            DataCopyExtParams copyParams = {1, static_cast<uint32_t>(count * sizeof(INDEX_T)), 0, 0, 0};
            DataCopyPad(idxUb, indexGm_[startOffset], copyParams, padParams);
        }
        PipeBarrier<PIPE_ALL>();

        for (int64_t i = 0; i < count; i++) {
            int64_t idx = static_cast<int64_t>(idxUb.GetValue(i));
            idx = (idx + paramsAxis) % paramsAxis;
            int64_t p = startOffset + i;
            int64_t pre = p / (indicesAxis * GetIndicesRow());
            int64_t row = MapAfterAxisOffset(p);
            int64_t srcOffset;
            if (isDiffShape) {
                srcOffset = MapDiffShapeOffset(p, pre) * paramsAxis * paramsRow + idx * paramsRow + row;
            } else {
                srcOffset = pre * paramsAxis * paramsRow + idx * paramsRow + row;
            }
            resUb.SetValue(i, xUb.GetValue(srcOffset));
        }

        SyncVtoM3();
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(count * sizeof(X_T)), 0, 0, 0};
        DataCopyPad<X_T>(yGm_[startOffset], resUb, copyParams);
        SyncM3toV();
    }

    __aicore__ inline void ProcessSlicedX(int64_t paramsAxis, int64_t paramsRow, int64_t indicesAxis, bool isDiffShape)
    {
        int32_t blockIdx = GetBlockIdx();

        int64_t paramSmallerThanIndices = sizeof(INDEX_T) / sizeof(X_T) > 0 ? sizeof(INDEX_T) / sizeof(X_T) : 1;
        int64_t indicesBlockNumLarge = paramSmallerThanIndices * (BLOCK_SIZE / static_cast<int64_t>(sizeof(INDEX_T)));

        int64_t sliceOnce = sliceThicknessOnce_;
        int64_t blockSizeX = BLOCK_SIZE / static_cast<int32_t>(sizeof(X_T));
        int64_t blockSizeIdx = BLOCK_SIZE / static_cast<int32_t>(sizeof(INDEX_T));

        uint32_t xSliceAligned = DivCeil(sliceOnce, blockSizeX) * blockSizeX;
        int64_t maxChunkCount = indicesRowNumOnce_ > indicesBlockNumLarge ? indicesRowNumOnce_ : indicesBlockNumLarge;
        uint32_t idxAlignedMax = DivCeil(maxChunkCount, blockSizeIdx) * blockSizeIdx;
        uint32_t resAlignedMax = DivCeil(maxChunkCount, blockSizeX) * blockSizeX;
        TBuf<TPosition::VECCALC> xBuf;
        TBuf<TPosition::VECCALC> idxBuf;
        TBuf<TPosition::VECCALC> resBuf;
        pipe_->InitBuffer(xBuf, xSliceAligned * sizeof(X_T));
        pipe_->InitBuffer(idxBuf, idxAlignedMax * sizeof(INDEX_T));
        pipe_->InitBuffer(resBuf, resAlignedMax * sizeof(X_T));
        auto xUb = xBuf.Get<X_T>();
        auto idxUb = idxBuf.Get<INDEX_T>();
        auto resUb = resBuf.Get<X_T>();

        uint32_t idxRemAligned = DivCeil(indicesBlockNumLarge, blockSizeIdx) * blockSizeIdx;
        uint32_t resRemAligned = DivCeil(indicesBlockNumLarge, blockSizeX) * blockSizeX;
        TBuf<TPosition::VECCALC> idxRemBuf;
        TBuf<TPosition::VECCALC> resRemBuf;
        pipe_->InitBuffer(idxRemBuf, idxRemAligned * sizeof(INDEX_T));
        pipe_->InitBuffer(resRemBuf, resRemAligned * sizeof(X_T));
        auto idxRemUb = idxRemBuf.Get<INDEX_T>();
        auto resRemUb = resRemBuf.Get<X_T>();

        int64_t coreBase = blockIdx * indicesNumEachCore_;
        for (int64_t loop = 0; loop < indicesLoopNum_; loop++) {
            ProcessSlicedXChunk(xUb, idxUb, resUb, paramsAxis, paramsRow, indicesAxis, isDiffShape,
                                coreBase + loop * indicesRowNumOnce_, indicesRowNumOnce_);
        }
        if (indicesRowNumLast_ > 0) {
            ProcessSlicedXChunk(xUb, idxUb, resUb, paramsAxis, paramsRow, indicesAxis, isDiffShape,
                                coreBase + indicesLoopNum_ * indicesRowNumOnce_, indicesRowNumLast_);
        }

        if (indicesNumRemaining_ > 0 && blockIdx < remainingBlockNum_) {
            int64_t off = indicesNumEachCore_ * needCoreNum_ + blockIdx * indicesBlockNumLarge;
            ProcessSlicedXChunk(xUb, idxRemUb, resRemUb, paramsAxis, paramsRow, indicesAxis, isDiffShape, off,
                                indicesBlockNumLarge);
        }
        if (indicesNumRemaining_ > 0 && blockIdx == remainingBlockNum_ && remainingBlockRemain_ > 0) {
            int64_t off = indicesNumEachCore_ * needCoreNum_ + blockIdx * indicesBlockNumLarge;
            ProcessSlicedXChunk(xUb, idxRemUb, resRemUb, paramsAxis, paramsRow, indicesAxis, isDiffShape, off,
                                remainingBlockRemain_);
        }
    }

    __aicore__ inline void ProcessSlicedXChunk(const LocalTensor<X_T>& xUb, const LocalTensor<INDEX_T>& idxUb,
                                               const LocalTensor<X_T>& resUb, int64_t paramsAxis, int64_t paramsRow,
                                               int64_t indicesAxis, bool isDiffShape, int64_t startOffset,
                                               int64_t count)
    {
        if (count <= 0)
            return;
        int64_t sliceOnce = sliceThicknessOnce_;
        int64_t sliceLast = sliceThicknessLast_;
        int64_t sliceNum = sliceNum_;
        int64_t blockSizeX = BLOCK_SIZE / static_cast<int32_t>(sizeof(X_T));

        PipeBarrier<PIPE_ALL>();
        {
            DataCopyPadExtParams<INDEX_T> padParams = {true, 0, 0, static_cast<INDEX_T>(0)};
            DataCopyExtParams copyParams = {1, static_cast<uint32_t>(count * sizeof(INDEX_T)), 0, 0, 0};
            DataCopyPad(idxUb, indexGm_[startOffset], copyParams, padParams);
        }
        PipeBarrier<PIPE_ALL>();

        for (int64_t s = 0; s < sliceNum; s++) {
            int64_t curThick = (s == sliceNum - 1) ? sliceLast : sliceOnce;
            int64_t sliceOffset = s * sliceOnce;

            PipeBarrier<PIPE_ALL>();
            {
                DataCopyPadExtParams<X_T> padParams = {true, 0, 0, static_cast<X_T>(0)};
                DataCopyExtParams copyParams = {1, static_cast<uint32_t>(curThick * sizeof(X_T)), 0, 0, 0};
                DataCopyPad(xUb, xGm_[sliceOffset], copyParams, padParams);
            }
            PipeBarrier<PIPE_ALL>();

            for (int64_t i = 0; i < count; i++) {
                int64_t idx = static_cast<int64_t>(idxUb.GetValue(i));
                idx = (idx + paramsAxis) % paramsAxis;
                int64_t p = startOffset + i;
                int64_t pre = p / (indicesAxis * GetIndicesRow());
                int64_t row = MapAfterAxisOffset(p);
                int64_t gmOffset;
                if (isDiffShape) {
                    gmOffset = MapDiffShapeOffset(p, pre) * paramsAxis * paramsRow + idx * paramsRow + row;
                } else {
                    gmOffset = pre * paramsAxis * paramsRow + idx * paramsRow + row;
                }
                if (gmOffset >= sliceOffset && gmOffset < sliceOffset + curThick) {
                    resUb.SetValue(i, xUb.GetValue(gmOffset - sliceOffset));
                }
            }
        }

        SyncVtoM3();
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(count * sizeof(X_T)), 0, 0, 0};
        DataCopyPad<X_T>(yGm_[startOffset], resUb, copyParams);
        SyncM3toV();
    }

    __aicore__ inline void ProcessLargeX(int64_t paramsAxis, int64_t paramsRow, int64_t indicesAxis, bool isDiffShape)
    {
        int32_t blockIdx = GetBlockIdx();

        int64_t paramSmallerThanIndices = sizeof(INDEX_T) / sizeof(X_T) > 0 ? sizeof(INDEX_T) / sizeof(X_T) : 1;
        int64_t indicesBlockNumLarge = paramSmallerThanIndices * (BLOCK_SIZE / static_cast<int64_t>(sizeof(INDEX_T)));

        int64_t maxChunkCount = indicesRowNumOnce_ > indicesBlockNumLarge ? indicesRowNumOnce_ : indicesBlockNumLarge;
        int64_t blockSizeIdx = BLOCK_SIZE / static_cast<int32_t>(sizeof(INDEX_T));
        uint32_t idxAlignedMax = DivCeil(maxChunkCount, blockSizeIdx) * blockSizeIdx;
        uint32_t resAlignedMax = DivCeil(maxChunkCount, BLOCK_SIZE / static_cast<int32_t>(sizeof(X_T))) *
                                 (BLOCK_SIZE / static_cast<int32_t>(sizeof(X_T)));
        TBuf<TPosition::VECCALC> idxBuf;
        TBuf<TPosition::VECCALC> resBuf;
        pipe_->InitBuffer(idxBuf, idxAlignedMax * sizeof(INDEX_T));
        pipe_->InitBuffer(resBuf, resAlignedMax * sizeof(X_T));
        auto idxUb = idxBuf.Get<INDEX_T>();
        auto resUb = resBuf.Get<X_T>();

        uint32_t idxRemAligned = DivCeil(indicesBlockNumLarge, blockSizeIdx) * blockSizeIdx;
        uint32_t resRemAligned = DivCeil(indicesBlockNumLarge, BLOCK_SIZE / static_cast<int32_t>(sizeof(X_T))) *
                                 (BLOCK_SIZE / static_cast<int32_t>(sizeof(X_T)));
        TBuf<TPosition::VECCALC> idxRemBuf;
        TBuf<TPosition::VECCALC> resRemBuf;
        pipe_->InitBuffer(idxRemBuf, idxRemAligned * sizeof(INDEX_T));
        pipe_->InitBuffer(resRemBuf, resRemAligned * sizeof(X_T));
        auto idxRemUb = idxRemBuf.Get<INDEX_T>();
        auto resRemUb = resRemBuf.Get<X_T>();

        int64_t coreBase = blockIdx * indicesNumEachCore_;
        for (int64_t loop = 0; loop < indicesLoopNum_; loop++) {
            ProcessLargeXChunk(idxUb, resUb, paramsAxis, paramsRow, indicesAxis, isDiffShape,
                               coreBase + loop * indicesRowNumOnce_, indicesRowNumOnce_);
        }
        if (indicesRowNumLast_ > 0) {
            ProcessLargeXChunk(idxUb, resUb, paramsAxis, paramsRow, indicesAxis, isDiffShape,
                               coreBase + indicesLoopNum_ * indicesRowNumOnce_, indicesRowNumLast_);
        }

        if (indicesNumRemaining_ > 0 && blockIdx < remainingBlockNum_) {
            int64_t off = indicesNumEachCore_ * needCoreNum_ + blockIdx * indicesBlockNumLarge;
            ProcessLargeXChunk(idxRemUb, resRemUb, paramsAxis, paramsRow, indicesAxis, isDiffShape, off,
                               indicesBlockNumLarge);
        }
        if (indicesNumRemaining_ > 0 && blockIdx == remainingBlockNum_ && remainingBlockRemain_ > 0) {
            int64_t off = indicesNumEachCore_ * needCoreNum_ + blockIdx * indicesBlockNumLarge;
            ProcessLargeXChunk(idxRemUb, resRemUb, paramsAxis, paramsRow, indicesAxis, isDiffShape, off,
                               remainingBlockRemain_);
        }
    }

    __aicore__ inline void ProcessLargeXChunk(const LocalTensor<INDEX_T>& idxUb, const LocalTensor<X_T>& resUb,
                                              int64_t paramsAxis, int64_t paramsRow, int64_t indicesAxis,
                                              bool isDiffShape, int64_t startOffset, int64_t count)
    {
        if (count <= 0)
            return;

        PipeBarrier<PIPE_ALL>();
        {
            DataCopyPadExtParams<INDEX_T> padParams = {true, 0, 0, static_cast<INDEX_T>(0)};
            DataCopyExtParams copyParams = {1, static_cast<uint32_t>(count * sizeof(INDEX_T)), 0, 0, 0};
            DataCopyPad(idxUb, indexGm_[startOffset], copyParams, padParams);
        }
        PipeBarrier<PIPE_ALL>();

        for (int64_t i = 0; i < count; i++) {
            int64_t idx = static_cast<int64_t>(idxUb.GetValue(i));
            idx = (idx + paramsAxis) % paramsAxis;
            int64_t p = startOffset + i;
            int64_t pre = p / (indicesAxis * GetIndicesRow());
            int64_t row = MapAfterAxisOffset(p);
            int64_t srcOffset;
            if (isDiffShape) {
                srcOffset = MapDiffShapeOffset(p, pre) * paramsAxis * paramsRow + idx * paramsRow + row;
            } else {
                srcOffset = pre * paramsAxis * paramsRow + idx * paramsRow + row;
            }
            resUb.SetValue(i, xGm_.GetValue(srcOffset));
        }

        SyncVtoM3();
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(count * sizeof(X_T)), 0, 0, 0};
        DataCopyPad<X_T>(yGm_[startOffset], resUb, copyParams);
        SyncM3toV();
    }

    __aicore__ inline int64_t MapDiffShapeOffset(int64_t flatOffset, int64_t preIdx)
    {
        int64_t axis = axis_;
        int64_t acc = indicesNum_;
        int64_t result = 0;
        for (int64_t d = 0; d < axis; d++) {
            int64_t dimSize = indicesShape_[d];
            acc /= dimSize;
            int64_t coord = (flatOffset / acc) % dimSize;
            int64_t xDim = paramsShape_[d];
            if (coord >= xDim)
                coord = xDim - 1;
            result = result * xDim + coord;
        }
        return result;
    }

    __aicore__ inline int64_t GetIndicesRow() const
    {
        int64_t row = 1;
        for (int64_t d = axis_ + 1; d < dims_; d++) {
            row *= indicesShape_[d];
        }
        return row;
    }

    __aicore__ inline int64_t MapAfterAxisOffset(int64_t flatOffset) const
    {
        int64_t acc = 1;
        for (int64_t d = axis_ + 1; d < dims_; d++) {
            acc *= indicesShape_[d];
        }
        if (acc == 0) {
            return 0;
        }
        int64_t rem = flatOffset % acc;
        int64_t result = 0;
        for (int64_t d = axis_ + 1; d < dims_; d++) {
            int64_t dimSize = indicesShape_[d];
            acc /= dimSize;
            int64_t coord = (rem / acc) % dimSize;
            int64_t xDim = paramsShape_[d];
            if (coord >= xDim)
                coord = xDim - 1;
            result = result * xDim + coord;
        }
        return result;
    }

    template <typename T, typename DstT, typename SrcT>
    __aicore__ inline void BatchDataCopyPad(const DstT& dst, const SrcT& src, int64_t elemCnt)
    {
        constexpr int64_t kMaxBurstBytes = 65520;
        const int64_t dsize = static_cast<int64_t>(sizeof(T));
        const int64_t alignElem = BLOCK_SIZE / dsize;
        const int64_t maxElem = (kMaxBurstBytes / dsize / alignElem) * alignElem;
        int64_t done = 0;
        while (done < elemCnt) {
            int64_t cur = elemCnt - done;
            if (cur > maxElem) {
                cur = maxElem;
            }
            DataCopyExtParams copyParams = {1, static_cast<uint32_t>(cur * dsize), 0, 0, 0};
            if constexpr (std::is_same<DstT, GlobalTensor<T>>::value) {
                DataCopyPad(dst[done], src[done], copyParams);
            } else {
                DataCopyPadExtParams<T> padParams = {true, 0, 0, static_cast<T>(0)};
                DataCopyPad(dst[done], src[done], copyParams, padParams);
            }
            done += cur;
        }
    }

    __aicore__ inline void ProcessLastAxis()
    {
        int32_t blockIdx = GetBlockIdx();
        int64_t paramsAxis = paramsAxis_;
        int64_t indicesAxis = indicesAxis_;
        int64_t numTasks = rounds_;
        int64_t coreNum = needCoreNum_;
        if (numTasks <= 0)
            return;

        int64_t rowsPerTask = repeatPerCore_ > 0 ? repeatPerCore_ : 1;

        int64_t blockSizeX = BLOCK_SIZE / static_cast<int32_t>(sizeof(X_T));
        int64_t blockSizeIdx = BLOCK_SIZE / static_cast<int32_t>(sizeof(INDEX_T));
        int64_t blockSizeIdx32 = BLOCK_SIZE / static_cast<int32_t>(sizeof(int32_t));

        bool isCut = (tilingMode_ == Tiling_MODE_FOR_LAST_AXIS_CUT_GATHER);
        bool isDiffShape = (tilingMode_ == Tiling_MODE_FOR_LAST_AXIS_DIFF_SHAPE);
        bool useGather = (tilingMode_ == Tiling_MODE_FOR_LAST_AXIS_GATHER);
        int64_t idxDim = isCut ? sliceThicknessOnce_ : indicesAxis;
        int64_t sliceNum = isCut ? sliceNum_ : 1;

        uint32_t xAligned = DivCeil(paramsAxis, blockSizeX) * blockSizeX;
        uint32_t idxAligned = DivCeil(idxDim, blockSizeIdx) * blockSizeIdx;
        uint32_t resAligned = DivCeil(idxDim, blockSizeX) * blockSizeX;
        uint32_t idx32Aligned = DivCeil(idxDim, blockSizeIdx32) * blockSizeIdx32;

        uint32_t xBatchAligned = xAligned * static_cast<uint32_t>(rowsPerTask);
        uint32_t idxBatchAligned = idxAligned * static_cast<uint32_t>(rowsPerTask);
        uint32_t resBatchAligned = resAligned * static_cast<uint32_t>(rowsPerTask);
        uint32_t idx32BatchAligned = idx32Aligned * static_cast<uint32_t>(rowsPerTask);

        TBuf<TPosition::VECCALC> xBuf;
        TBuf<TPosition::VECCALC> idxBuf;
        TBuf<TPosition::VECCALC> resBuf;
        TBuf<TPosition::VECCALC> idx32Buf;
        TBuf<TPosition::VECCALC> rampBuf;
        pipe_->InitBuffer(xBuf, xBatchAligned * sizeof(X_T));
        pipe_->InitBuffer(idxBuf, idxBatchAligned * sizeof(INDEX_T));
        pipe_->InitBuffer(resBuf, resBatchAligned * sizeof(X_T));
        pipe_->InitBuffer(idx32Buf, idx32BatchAligned * 2 * sizeof(int32_t));
        LocalTensor<int32_t> rampUb;
        if (!isCut) {
            pipe_->InitBuffer(
                rampBuf, static_cast<uint64_t>(rowsPerTask) * static_cast<uint64_t>(indicesAxis) * sizeof(int32_t));
            rampUb = rampBuf.Get<int32_t>();
            const int32_t rampRowStride = static_cast<int32_t>(paramsAxis * static_cast<int64_t>(sizeof(X_T)));
            for (int64_t r = 0; r < rowsPerTask; r++) {
                for (int64_t c = 0; c < indicesAxis; c++) {
                    rampUb.SetValue(r * indicesAxis + c, static_cast<int32_t>(r * rampRowStride));
                }
            }
        }

        for (int64_t t = blockIdx; t < numTasks; t += coreNum) {
            int64_t rowsThisTask = rowsPerTask;
            if ((t == rounds_ - 1) && (roundsTail_ > 0)) {
                rowsThisTask = roundsTail_;
            }
            int64_t rowBase = t * rowsPerTask;

            auto xUb = xBuf.Get<X_T>();
            auto idxUb = idxBuf.Get<INDEX_T>();
            auto idx32Ub = idx32Buf.Get<int32_t>();
            auto resUb = resBuf.Get<X_T>();

            if (isCut) {
                ProcessLastAxisCut(xUb, idxUb, idx32Ub, resUb, paramsAxis, indicesAxis, rowBase, rowsThisTask, xAligned,
                                   idxAligned, idx32Aligned, resAligned, idxDim, sliceNum);
                continue;
            }

            SyncVtoM2();
            BatchDataCopyPad<INDEX_T>(idxUb, indexGm_[rowBase * indicesAxis], rowsThisTask * indicesAxis);
            if (!isDiffShape) {
                SyncVtoM2();
                BatchDataCopyPad<X_T>(xUb, xGm_[rowBase * paramsAxis], rowsThisTask * paramsAxis);
            } else {
                for (int64_t r = 0; r < rowsThisTask; r++) {
                    int64_t preIdx = MapLastAxisPreOffset(rowBase + r);
                    SyncVtoM2();
                    DataCopyPadExtParams<X_T> xPadParams = {true, 0, 0, static_cast<X_T>(0)};
                    DataCopyExtParams xCopyParams = {1, static_cast<uint32_t>(paramsAxis * sizeof(X_T)), 0, 0, 0};
                    DataCopyPad(xUb[r * xAligned], xGm_[preIdx * paramsAxis], xCopyParams, xPadParams);
                }
            }
            PipeBarrier<PIPE_ALL>();

            if (useGather && IsGatherSupportedDtype<X_T>()) {
                const int32_t paramsAxisI32 = static_cast<int32_t>(paramsAxis);
                const int32_t dsizeI32 = static_cast<int32_t>(sizeof(X_T));
                const uint32_t nElem = static_cast<uint32_t>(rowsThisTask * indicesAxis);
                const uint32_t nElemMax = static_cast<uint32_t>(rowsPerTask * indicesAxis);
                LocalTensor<int32_t> idx32View;
                LocalTensor<int32_t> signTmp;
                if constexpr (std::is_same<INDEX_T, int64_t>::value) {
                    Cast<int32_t>(idx32Ub, idxUb, RoundMode::CAST_NONE, nElem);
                    idx32View = idx32Ub;
                    signTmp = idx32Ub[nElemMax];
                } else {
                    idx32View = idxUb.template ReinterpretCast<int32_t>();
                    signTmp = idx32Ub;
                }

                ShiftRight<int32_t>(signTmp, idx32View, 31, nElem);
                Muls<int32_t>(signTmp, signTmp, -paramsAxisI32, nElem);
                Add<int32_t>(idx32View, idx32View, signTmp, nElem);
                Muls<int32_t>(signTmp, idx32View, -1, nElem);
                Adds<int32_t>(signTmp, signTmp, paramsAxisI32 - 1, nElem);
                ShiftRight<int32_t>(signTmp, signTmp, 31, nElem);
                Muls<int32_t>(signTmp, signTmp, paramsAxisI32, nElem);
                Add<int32_t>(idx32View, idx32View, signTmp, nElem);
                Muls<int32_t>(idx32View, idx32View, dsizeI32, nElem);
                Add<int32_t>(idx32View, idx32View, rampUb, nElem);

                AscendC::Gather<X_T>(resUb, xUb, idx32View.ReinterpretCast<uint32_t>(), 0, nElem);
            } else if (paramsAxis == 1) {
                const int64_t xRowStride = isDiffShape ? static_cast<int64_t>(xAligned) : 1;
                for (int64_t r = 0; r < rowsThisTask; r++) {
                    const X_T v = xUb.GetValue(r * xRowStride);
                    const int64_t base = r * indicesAxis;
                    for (int64_t c = 0; c < indicesAxis; c++) {
                        resUb.SetValue(base + c, v);
                    }
                }
            } else {
                const int64_t xRowStride = isDiffShape ? static_cast<int64_t>(xAligned) : paramsAxis;
                const int64_t pAxis = paramsAxis;
                const int64_t pm1 = paramsAxis - 1;
                for (int64_t r = 0; r < rowsThisTask; r++) {
                    const int64_t base = r * indicesAxis;
                    const int64_t xRowOff = r * xRowStride;
                    int64_t c = 0;
                    for (; c + 4 <= indicesAxis; c += 4) {
                        int64_t i0 = static_cast<int64_t>(idxUb.GetValue(base + c));
                        i0 += (i0 >> 63) & pAxis;
                        i0 -= ((pm1 - i0) >> 63) & pAxis;
                        int64_t i1 = static_cast<int64_t>(idxUb.GetValue(base + c + 1));
                        i1 += (i1 >> 63) & pAxis;
                        i1 -= ((pm1 - i1) >> 63) & pAxis;
                        int64_t i2 = static_cast<int64_t>(idxUb.GetValue(base + c + 2));
                        i2 += (i2 >> 63) & pAxis;
                        i2 -= ((pm1 - i2) >> 63) & pAxis;
                        int64_t i3 = static_cast<int64_t>(idxUb.GetValue(base + c + 3));
                        i3 += (i3 >> 63) & pAxis;
                        i3 -= ((pm1 - i3) >> 63) & pAxis;
                        resUb.SetValue(base + c, xUb.GetValue(xRowOff + i0));
                        resUb.SetValue(base + c + 1, xUb.GetValue(xRowOff + i1));
                        resUb.SetValue(base + c + 2, xUb.GetValue(xRowOff + i2));
                        resUb.SetValue(base + c + 3, xUb.GetValue(xRowOff + i3));
                    }
                    for (; c < indicesAxis; c++) {
                        int64_t iv = static_cast<int64_t>(idxUb.GetValue(base + c));
                        iv += (iv >> 63) & pAxis;
                        iv -= ((pm1 - iv) >> 63) & pAxis;
                        resUb.SetValue(base + c, xUb.GetValue(xRowOff + iv));
                    }
                }
            }

            SyncVtoM3();
            BatchDataCopyPad<X_T>(yGm_[rowBase * indicesAxis], resUb, rowsThisTask * indicesAxis);
            SyncM3toV();
        }
    }

    __aicore__ inline void ProcessLastAxisCut(const LocalTensor<X_T>& xUb, const LocalTensor<INDEX_T>& idxUb,
                                              const LocalTensor<int32_t>& idx32Ub, const LocalTensor<X_T>& resUb,
                                              int64_t paramsAxis, int64_t indicesAxis, int64_t rowBase,
                                              int64_t rowsThisTask, uint32_t xAligned, uint32_t idxAligned,
                                              uint32_t idx32Aligned, uint32_t resAligned, int64_t idxDim,
                                              int64_t sliceNum)
    {
        for (int64_t r = 0; r < rowsThisTask; r++) {
            int64_t row = rowBase + r;
            int64_t indicesGmOff = row * indicesAxis;
            int64_t preIdx = row;
            if (tilingMode_ == Tiling_MODE_FOR_LAST_AXIS_DIFF_SHAPE) {
                preIdx = MapLastAxisPreOffset(row);
            }
            int64_t xGmOff = preIdx * paramsAxis;

            SyncVtoM2();
            {
                DataCopyPadExtParams<X_T> xPadParams = {true, 0, 0, static_cast<X_T>(0)};
                DataCopyExtParams xCopyParams = {1, static_cast<uint32_t>(paramsAxis * sizeof(X_T)), 0, 0, 0};
                DataCopyPad(xUb[r * xAligned], xGm_[xGmOff], xCopyParams, xPadParams);
            }
            PipeBarrier<PIPE_ALL>();

            for (int64_t s = 0; s < sliceNum; s++) {
                int64_t curThick = (s == sliceNum - 1) ? sliceThicknessLast_ : idxDim;
                int64_t sliceOff = s * sliceThicknessOnce_;
                int64_t curIdxGmOff = indicesGmOff + sliceOff;

                SyncVtoM2();
                {
                    DataCopyPadExtParams<INDEX_T> idxPadParams = {true, 0, 0, static_cast<INDEX_T>(0)};
                    DataCopyExtParams idxCopyParams = {1, static_cast<uint32_t>(curThick * sizeof(INDEX_T)), 0, 0, 0};
                    DataCopyPad(idxUb[r * idxAligned], indexGm_[curIdxGmOff], idxCopyParams, idxPadParams);
                }
                PipeBarrier<PIPE_ALL>();

                GatherRow(xUb[r * xAligned], idxUb[r * idxAligned], idx32Ub[r * idx32Aligned], resUb[r * resAligned],
                          paramsAxis, curThick, tilingMode_ == Tiling_MODE_FOR_LAST_AXIS_GATHER);

                SyncVtoM3();
                DataCopyExtParams copyParams = {1, static_cast<uint32_t>(curThick * sizeof(X_T)), 0, 0, 0};
                DataCopyPad<X_T>(yGm_[curIdxGmOff], resUb[r * resAligned], copyParams);
                SyncM3toV();
            }
        }
    }

    __aicore__ inline int64_t MapLastAxisPreOffset(int64_t preIdx)
    {
        int64_t axis = axis_;
        int64_t acc = 1;
        for (int64_t d = 0; d < axis; d++) {
            acc *= indicesShape_[d];
        }
        int64_t result = 0;
        int64_t remaining = preIdx;
        for (int64_t d = 0; d < axis; d++) {
            acc /= indicesShape_[d];
            int64_t coord = remaining / acc;
            remaining = remaining % acc;
            int64_t xDim = paramsShape_[d];
            if (coord >= xDim)
                coord = xDim - 1;
            int64_t xAcc = 1;
            for (int64_t k = d + 1; k < axis; k++) {
                xAcc *= paramsShape_[k];
            }
            result += coord * xAcc;
        }
        return result;
    }

    __aicore__ inline void GatherRow(const LocalTensor<X_T>& xUb, const LocalTensor<INDEX_T>& idxUb,
                                     const LocalTensor<int32_t>& idx32Ub, const LocalTensor<X_T>& resUb,
                                     int64_t paramsAxis, int64_t indicesAxis, bool useGather)
    {
        if constexpr (IsGatherSupportedDtype<X_T>()) {
            if (useGather) {
                for (int64_t j = 0; j < indicesAxis; j++) {
                    int64_t idxVal = static_cast<int64_t>(idxUb.GetValue(j));
                    idxVal = (idxVal + paramsAxis) % paramsAxis;
                    idx32Ub.SetValue(j, static_cast<int32_t>(idxVal * sizeof(X_T)));
                }
                AscendC::Gather<X_T>(resUb, xUb, idx32Ub.template ReinterpretCast<uint32_t>(), 0,
                                     static_cast<uint32_t>(indicesAxis));
                return;
            }
        }
        for (int64_t j = 0; j < indicesAxis; j++) {
            int64_t idxVal = static_cast<int64_t>(idxUb.GetValue(j));
            idxVal = (idxVal + paramsAxis) % paramsAxis;
            resUb.SetValue(j, xUb.GetValue(idxVal));
        }
    }
};
} // namespace AscendC

#endif // GATHER_ELEMENTS_KERNEL_H
