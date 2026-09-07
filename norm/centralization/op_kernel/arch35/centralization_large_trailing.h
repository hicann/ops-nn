/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_NORM_CENTRALIZATION_LARGE_TRAILING_H_
#define OPS_NORM_CENTRALIZATION_LARGE_TRAILING_H_
#include "kernel_operator.h"
#include "centralization_tiling_data.h"

namespace Centralization {
using namespace AscendC;

template <typename T>
class LargeTrailing {
public:
    __aicore__ inline LargeTrailing(TPipe* pipe, const CentralizationTilingData* tiling) : pipe_(pipe), tiling_(tiling)
    {}

    __aicore__ inline void Init(__gm__ uint8_t* x, __gm__ uint8_t* y, __gm__ uint8_t* workspace)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        workspaceGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
        pipe_->InitBuffer(inQueue_, 2, tiling_->largeTrailingInputBufferBytes);
        pipe_->InitBuffer(outQueue_, 2, tiling_->largeTrailingOutputBufferBytes);
        pipe_->InitBuffer(calcBuf_, tiling_->largeTrailingCalcBufferBytes);
        pipe_->InitBuffer(meanBuf_, tiling_->largeTrailingMeanBufferBytes);
        pipe_->InitBuffer(sumBuf_, 128);
        eventMte2ToV_ = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
        eventVToS_ = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_S));
        eventSToMte3_ = static_cast<event_t>(pipe_->FetchEventID(HardEvent::S_MTE3));
        eventMte3ToMte2_ = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_MTE2));
        eventMte2ToS_ = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_S));
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = GetBlockIdx();
        const uint32_t cores = GetBlockNum();
        if (tiling_->rowParallel != 0) {
            ProcessRowParallel(core, cores);
            return;
        }
        ProcessMultiCoreReduce(core, cores);
    }

private:
    static constexpr int64_t kTileElements = 1024;
    static constexpr int64_t kReduceLayerElements = 256;
    static constexpr int64_t kWorkspaceElements = 32 / sizeof(float);
    static constexpr bool kUseCompensatedReduce = IsSameType<T, float>::value || IsSameType<T, half>::value;

    struct CompensatedSum {
        float sum;
        float correction;
    };

    __aicore__ inline int64_t Min(int64_t lhs, int64_t rhs) const { return lhs < rhs ? lhs : rhs; }

    __aicore__ inline int64_t AlignToBlock(int64_t elements) const
    {
        constexpr int64_t blockElements = 32 / sizeof(T);
        return (elements + blockElements - 1) / blockElements * blockElements;
    }

    __aicore__ inline int64_t WorkspaceStride() const
    {
        return tiling_->alignedCoresPerRow > 0 ? tiling_->alignedCoresPerRow : kWorkspaceElements;
    }

    __aicore__ inline uint64_t WorkspaceOffset(int64_t row, int64_t slot) const
    {
        return static_cast<uint64_t>((row * WorkspaceStride() + slot) * kWorkspaceElements);
    }

    __aicore__ inline float Abs(float value) const { return value < 0.0f ? -value : value; }

    __aicore__ inline void AddCompensated(CompensatedSum& acc, float value) const
    {
        const float next = acc.sum + value;
        if (Abs(acc.sum) >= Abs(value)) {
            acc.correction += (acc.sum - next) + value;
        } else {
            acc.correction += (value - next) + acc.sum;
        }
        acc.sum = next;
    }

    __aicore__ inline float TotalCompensated(const CompensatedSum& acc) const { return acc.sum + acc.correction; }

    __aicore__ inline int64_t RowBase(int64_t row) const { return row * tiling_->reduceCount; }

    __aicore__ inline void CopyIn(LocalTensor<T>& input, int64_t offset, int64_t validElements)
    {
        const int64_t alignedElements = AlignToBlock(validElements);
        DataCopyExtParams copy{1, static_cast<uint32_t>(validElements * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(alignedElements - validElements), static_cast<T>(0)};
        DataCopyPad(input, xGm_[static_cast<uint64_t>(offset)], copy, pad);
    }

    __aicore__ inline void CopyInReduceTile(int64_t offset, int64_t validElements)
    {
        LocalTensor<T> input = inQueue_.AllocTensor<T>();
        CopyIn(input, offset, validElements);
        inQueue_.EnQue(input);
    }

    __aicore__ inline float ComputeReduceTile(int64_t validElements)
    {
        const int64_t alignedElements = AlignToBlock(validElements);
        LocalTensor<T> input = inQueue_.DeQue<T>();
        LocalTensor<float> calc = calcBuf_.Get<float>();
        if constexpr (IsSameType<T, float>::value) {
            DataCopy(calc, input, static_cast<uint32_t>(alignedElements));
            SetFlag<HardEvent::MTE2_V>(eventMte2ToV_);
            WaitFlag<HardEvent::MTE2_V>(eventMte2ToV_);
        } else {
            Cast(calc, input, RoundMode::CAST_NONE, static_cast<uint32_t>(alignedElements));
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<float> sum = sumBuf_.Get<float>();
        uint32_t srcShape[2] = {1, static_cast<uint32_t>(alignedElements)};
        ReduceSum<float, Pattern::Reduce::AR, true>(sum, calc, srcShape, true);
        SetFlag<HardEvent::V_S>(eventVToS_);
        WaitFlag<HardEvent::V_S>(eventVToS_);
        const float value = sum.GetValue(0);
        inQueue_.FreeTensor(input);
        return value;
    }

    __aicore__ inline float ReduceTile(int64_t offset, int64_t validElements)
    {
        CopyInReduceTile(offset, validElements);
        return ComputeReduceTile(validElements);
    }

    __aicore__ inline CompensatedSum ComputeReduceTileCompensated(int64_t validElements)
    {
        const int64_t alignedElements = AlignToBlock(validElements);
        LocalTensor<T> input = inQueue_.DeQue<T>();
        LocalTensor<float> calc = calcBuf_.Get<float>();
        if constexpr (IsSameType<T, float>::value) {
            DataCopy(calc, input, static_cast<uint32_t>(alignedElements));
            SetFlag<HardEvent::MTE2_V>(eventMte2ToV_);
            WaitFlag<HardEvent::MTE2_V>(eventMte2ToV_);
        } else {
            Cast(calc, input, RoundMode::CAST_NONE, static_cast<uint32_t>(alignedElements));
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<float> sum = sumBuf_.Get<float>();
        CompensatedSum partial{0.0f, 0.0f};
        for (int64_t layer = 0; layer < validElements; layer += kReduceLayerElements) {
            const int64_t layerElements = Min(kReduceLayerElements, validElements - layer);
            const int64_t alignedLayerElements = AlignToBlock(layerElements);
            uint32_t srcShape[2] = {1, static_cast<uint32_t>(alignedLayerElements)};
            ReduceSum<float, Pattern::Reduce::AR, true>(sum, calc[static_cast<uint32_t>(layer)], srcShape, true);
            SetFlag<HardEvent::V_S>(eventVToS_);
            WaitFlag<HardEvent::V_S>(eventVToS_);
            AddCompensated(partial, sum.GetValue(0));
        }
        inQueue_.FreeTensor(input);
        return partial;
    }

    __aicore__ inline CompensatedSum ReduceTileCompensated(int64_t offset, int64_t validElements)
    {
        CopyInReduceTile(offset, validElements);
        return ComputeReduceTileCompensated(validElements);
    }

    __aicore__ inline float ReduceRowSingleCore(int64_t row)
    {
        const int64_t chunkCount = (tiling_->reduceCount + kTileElements - 1) / kTileElements;
        const int64_t rowBase = RowBase(row);
        float partial = 0.0f;
        for (int64_t chunk = 0; chunk < chunkCount; ++chunk) {
            const int64_t col = chunk * kTileElements;
            const int64_t validElements = Min(kTileElements, tiling_->reduceCount - col);
            partial += ReduceTile(rowBase + col, validElements);
        }
        return partial / static_cast<float>(tiling_->reduceCount);
    }

    __aicore__ inline float ReduceRowSingleCoreCompensated(int64_t row)
    {
        const int64_t chunkCount = (tiling_->reduceCount + kTileElements - 1) / kTileElements;
        const int64_t rowBase = RowBase(row);
        CompensatedSum partial{0.0f, 0.0f};
        for (int64_t chunk = 0; chunk < chunkCount; ++chunk) {
            const int64_t col = chunk * kTileElements;
            const int64_t validElements = Min(kTileElements, tiling_->reduceCount - col);
            const CompensatedSum tilePartial = ReduceTileCompensated(rowBase + col, validElements);
            AddCompensated(partial, tilePartial.sum);
            AddCompensated(partial, tilePartial.correction);
        }
        return TotalCompensated(partial) / static_cast<float>(tiling_->reduceCount);
    }

    __aicore__ inline void CopyInTile(int64_t offset, int64_t validElements)
    {
        LocalTensor<T> input = inQueue_.AllocTensor<T>();
        CopyIn(input, offset, validElements);
        inQueue_.EnQue(input);
    }

    __aicore__ inline void ComputeTile(int64_t validElements, float mean)
    {
        const int64_t alignedElements = AlignToBlock(validElements);
        LocalTensor<T> input = inQueue_.DeQue<T>();

        LocalTensor<float> calc = calcBuf_.Get<float>();
        if constexpr (IsSameType<T, float>::value) {
            DataCopy(calc, input, static_cast<uint32_t>(alignedElements));
            SetFlag<HardEvent::MTE2_V>(eventMte2ToV_);
            WaitFlag<HardEvent::MTE2_V>(eventMte2ToV_);
        } else {
            Cast(calc, input, RoundMode::CAST_NONE, static_cast<uint32_t>(alignedElements));
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<float> meanTensor = meanBuf_.Get<float>();
        Duplicate(meanTensor, mean, static_cast<int32_t>(alignedElements));

        PipeBarrier<PIPE_V>();

        Sub(calc, calc, meanTensor, static_cast<uint32_t>(alignedElements));
        PipeBarrier<PIPE_V>();

        LocalTensor<T> output = outQueue_.AllocTensor<T>();
        if constexpr (IsSameType<T, float>::value) {
            DataCopy(output, calc, static_cast<uint32_t>(alignedElements));
        } else {
            Cast(output, calc, RoundMode::CAST_ROUND, static_cast<uint32_t>(alignedElements));
        }
        PipeBarrier<PIPE_V>();
        outQueue_.EnQue(output);
        inQueue_.FreeTensor(input);
    }

    __aicore__ inline void CopyOutTile(int64_t offset, int64_t validElements)
    {
        LocalTensor<T> output = outQueue_.DeQue<T>();

        DataCopyExtParams copy{1, static_cast<uint32_t>(validElements * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[static_cast<uint64_t>(offset)], output, copy);
        outQueue_.FreeTensor(output);
    }

    __aicore__ inline void SubtractRowSingleCore(int64_t row, float mean)
    {
        const int64_t chunkCount = (tiling_->reduceCount + kTileElements - 1) / kTileElements;
        const int64_t rowBase = RowBase(row);
        for (int64_t stage = 0; stage < chunkCount + 2; ++stage) {
            if (stage < chunkCount) {
                const int64_t chunk = stage;
                const int64_t col = chunk * kTileElements;
                const int64_t validElements = Min(kTileElements, tiling_->reduceCount - col);
                CopyInTile(rowBase + col, validElements);
            }
            if (stage >= 1 && (stage - 1) < chunkCount) {
                const int64_t chunk = stage - 1;
                const int64_t col = chunk * kTileElements;
                const int64_t validElements = Min(kTileElements, tiling_->reduceCount - col);
                ComputeTile(validElements, mean);
            }
            if (stage >= 2 && (stage - 2) < chunkCount) {
                const int64_t chunk = stage - 2;
                const int64_t col = chunk * kTileElements;
                const int64_t validElements = Min(kTileElements, tiling_->reduceCount - col);
                CopyOutTile(rowBase + col, validElements);
            }
        }
    }

    __aicore__ inline float PartialReduceRow(int64_t row, uint32_t localCore, uint32_t coresPerRow)
    {
        const int64_t chunkCount = (tiling_->reduceCount + kTileElements - 1) / kTileElements;
        const int64_t rowBase = RowBase(row);
        float partial = 0.0f;
        for (int64_t chunk = static_cast<int64_t>(localCore); chunk < chunkCount;
             chunk += static_cast<int64_t>(coresPerRow)) {
            const int64_t col = chunk * kTileElements;
            const int64_t validElements = Min(kTileElements, tiling_->reduceCount - col);
            partial += ReduceTile(rowBase + col, validElements);
        }
        return partial;
    }

    __aicore__ inline CompensatedSum PartialReduceRowCompensated(int64_t row, uint32_t localCore, uint32_t coresPerRow)
    {
        const int64_t chunkCount = (tiling_->reduceCount + kTileElements - 1) / kTileElements;
        const int64_t rowBase = RowBase(row);
        CompensatedSum partial{0.0f, 0.0f};
        for (int64_t chunk = static_cast<int64_t>(localCore); chunk < chunkCount;
             chunk += static_cast<int64_t>(coresPerRow)) {
            const int64_t col = chunk * kTileElements;
            const int64_t validElements = Min(kTileElements, tiling_->reduceCount - col);
            const CompensatedSum tilePartial = ReduceTileCompensated(rowBase + col, validElements);
            AddCompensated(partial, tilePartial.sum);
            AddCompensated(partial, tilePartial.correction);
        }
        return partial;
    }

    __aicore__ inline void StoreWorkspaceValue(int64_t row, int64_t slot, float value)
    {
        LocalTensor<float> output = outQueue_.AllocTensor<float>();
        Duplicate(output, static_cast<float>(0.0f), static_cast<int32_t>(kWorkspaceElements));
        SetFlag<HardEvent::V_S>(eventVToS_);
        WaitFlag<HardEvent::V_S>(eventVToS_);
        output.SetValue(0, value);
        outQueue_.EnQue(output);
        output = outQueue_.DeQue<float>();
        SetFlag<HardEvent::S_MTE3>(eventSToMte3_);
        WaitFlag<HardEvent::S_MTE3>(eventSToMte3_);
        DataCopy(workspaceGm_[WorkspaceOffset(row, slot)], output, static_cast<uint32_t>(kWorkspaceElements));
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
        outQueue_.FreeTensor(output);
    }

    __aicore__ inline void StoreWorkspaceCompensated(int64_t row, int64_t slot, CompensatedSum value)
    {
        LocalTensor<float> output = outQueue_.AllocTensor<float>();
        Duplicate(output, static_cast<float>(0.0f), static_cast<int32_t>(kWorkspaceElements));
        SetFlag<HardEvent::V_S>(eventVToS_);
        WaitFlag<HardEvent::V_S>(eventVToS_);
        output.SetValue(0, value.sum);
        output.SetValue(1, value.correction);
        outQueue_.EnQue(output);
        output = outQueue_.DeQue<float>();
        SetFlag<HardEvent::S_MTE3>(eventSToMte3_);
        WaitFlag<HardEvent::S_MTE3>(eventSToMte3_);
        DataCopy(workspaceGm_[WorkspaceOffset(row, slot)], output, static_cast<uint32_t>(kWorkspaceElements));
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
        outQueue_.FreeTensor(output);
    }

    __aicore__ inline LocalTensor<float> LoadWorkspaceTensor(int64_t row, int64_t slot)
    {
        LocalTensor<float> input = inQueue_.AllocTensor<float>();
        DataCopy(input, workspaceGm_[WorkspaceOffset(row, slot)], static_cast<uint32_t>(kWorkspaceElements));
        inQueue_.EnQue(input);
        input = inQueue_.DeQue<float>();
        SetFlag<HardEvent::MTE2_S>(eventMte2ToS_);
        WaitFlag<HardEvent::MTE2_S>(eventMte2ToS_);
        return input;
    }

    __aicore__ inline float LoadWorkspaceValue(int64_t row, int64_t slot = 0)
    {
        LocalTensor<float> input = LoadWorkspaceTensor(row, slot);
        const float value = input.GetValue(0);
        inQueue_.FreeTensor(input);
        return value;
    }

    __aicore__ inline CompensatedSum LoadWorkspaceCompensated(int64_t row, int64_t slot)
    {
        LocalTensor<float> input = LoadWorkspaceTensor(row, slot);
        const CompensatedSum value{input.GetValue(0), input.GetValue(1)};
        inQueue_.FreeTensor(input);
        return value;
    }

    __aicore__ inline void FinalizeMeanRow(int64_t row)
    {
        const int64_t partialElements = tiling_->coresPerRow * kWorkspaceElements;
        LocalTensor<float> partials = calcBuf_.Get<float>();
        DataCopy(partials, workspaceGm_[WorkspaceOffset(row, 0)], static_cast<uint32_t>(partialElements));
        SetFlag<HardEvent::MTE2_V>(eventMte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(eventMte2ToV_);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> sum = sumBuf_.Get<float>();
        uint32_t srcShape[2] = {1, static_cast<uint32_t>(partialElements)};
        ReduceSum<float, Pattern::Reduce::AR, true>(sum, partials, srcShape, true);
        SetFlag<HardEvent::V_S>(eventVToS_);
        WaitFlag<HardEvent::V_S>(eventVToS_);
        const float mean = sum.GetValue(0) / static_cast<float>(tiling_->reduceCount);
        StoreWorkspaceValue(row, 0, mean);
    }

    __aicore__ inline void FinalizeMeanRowCompensated(int64_t row)
    {
        CompensatedSum partial{0.0f, 0.0f};
        for (int64_t slot = 0; slot < tiling_->coresPerRow; ++slot) {
            const CompensatedSum slotPartial = LoadWorkspaceCompensated(row, slot);
            AddCompensated(partial, slotPartial.sum);
            AddCompensated(partial, slotPartial.correction);
        }
        const float mean = TotalCompensated(partial) / static_cast<float>(tiling_->reduceCount);
        StoreWorkspaceValue(row, 0, mean);
    }

    __aicore__ inline void ProcessRowParallel(uint32_t core, uint32_t cores)
    {
        for (int64_t row = core; row < tiling_->groupCount; row += cores) {
            if constexpr (kUseCompensatedReduce) {
                const float mean = ReduceRowSingleCoreCompensated(row);
                SubtractRowSingleCore(row, mean);
            } else {
                const float mean = ReduceRowSingleCore(row);
                SubtractRowSingleCore(row, mean);
            }
        }
    }

    __aicore__ inline void ProcessMultiCoreReduce(uint32_t core, uint32_t /*cores*/)
    {
        const uint32_t coresPerRow = static_cast<uint32_t>(tiling_->coresPerRow);
        const uint32_t row = core / coresPerRow;
        if (row >= static_cast<uint32_t>(tiling_->groupCount)) {
            return;
        }
        const uint32_t localCore = core % coresPerRow;
        if constexpr (kUseCompensatedReduce) {
            const CompensatedSum partial = PartialReduceRowCompensated(static_cast<int64_t>(row), localCore,
                                                                       coresPerRow);
            StoreWorkspaceCompensated(static_cast<int64_t>(row), static_cast<int64_t>(localCore), partial);
            SyncAll();
            if (localCore == 0) {
                FinalizeMeanRowCompensated(static_cast<int64_t>(row));
            }
            SyncAll();
            SubtractRow(static_cast<int64_t>(row), localCore, coresPerRow);
        } else {
            const float partial = PartialReduceRow(static_cast<int64_t>(row), localCore, coresPerRow);
            StoreWorkspaceValue(static_cast<int64_t>(row), static_cast<int64_t>(localCore), partial);
            SyncAll();
            if (localCore == 0) {
                FinalizeMeanRow(static_cast<int64_t>(row));
            }
            SyncAll();
            SubtractRow(static_cast<int64_t>(row), localCore, coresPerRow);
        }
    }

    __aicore__ inline void SubtractRow(int64_t row, uint32_t localCore, uint32_t coresPerRow)
    {
        const int64_t chunkCount = (tiling_->reduceCount + kTileElements - 1) / kTileElements;
        const float mean = LoadWorkspaceValue(row);
        const int64_t rowBase = RowBase(row);
        const int64_t tileCount = (chunkCount - static_cast<int64_t>(localCore) + static_cast<int64_t>(coresPerRow) -
                                   1) /
                                  static_cast<int64_t>(coresPerRow);
        if (tileCount <= 0) {
            return;
        }
        for (int64_t stage = 0; stage < tileCount + 2; ++stage) {
            if (stage < tileCount) {
                const int64_t localTile = stage;
                const int64_t chunk = static_cast<int64_t>(localCore) + localTile * static_cast<int64_t>(coresPerRow);
                const int64_t col = chunk * kTileElements;
                const int64_t validElements = Min(kTileElements, tiling_->reduceCount - col);
                CopyInTile(rowBase + col, validElements);
            }
            if (stage >= 1 && (stage - 1) < tileCount) {
                const int64_t localTile = stage - 1;
                const int64_t chunk = static_cast<int64_t>(localCore) + localTile * static_cast<int64_t>(coresPerRow);
                const int64_t col = chunk * kTileElements;
                const int64_t validElements = Min(kTileElements, tiling_->reduceCount - col);
                ComputeTile(validElements, mean);
            }
            if (stage >= 2 && (stage - 2) < tileCount) {
                const int64_t localTile = stage - 2;
                const int64_t chunk = static_cast<int64_t>(localCore) + localTile * static_cast<int64_t>(coresPerRow);
                const int64_t col = chunk * kTileElements;
                const int64_t validElements = Min(kTileElements, tiling_->reduceCount - col);
                CopyOutTile(rowBase + col, validElements);
            }
        }
    }

    TPipe* pipe_;
    const CentralizationTilingData* tiling_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<float> workspaceGm_;
    TQue<QuePosition::VECIN, 2> inQueue_;
    TQue<QuePosition::VECOUT, 2> outQueue_;
    TBuf<TPosition::VECCALC> calcBuf_;
    TBuf<TPosition::VECCALC> meanBuf_;
    TBuf<TPosition::VECCALC> sumBuf_;
    event_t eventMte2ToV_{};
    event_t eventVToS_{};
    event_t eventSToMte3_{};
    event_t eventMte3ToMte2_{};
    event_t eventMte2ToS_{};
};
} // namespace Centralization
#endif
