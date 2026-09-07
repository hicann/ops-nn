/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_NORM_CENTRALIZATION_GENERIC_LARGE_IMPL_H_
#define OPS_NORM_CENTRALIZATION_GENERIC_LARGE_IMPL_H_

__aicore__ inline void ProcessLarge(uint32_t core, uint32_t cores)
{
    if (tiling_->rowParallel != 0) {
        ProcessLargeRowParallel(core, cores);
        return;
    }
    ProcessLargeMultiCoreReduce(core, cores);
}

__aicore__ inline void ProcessLargeRowParallel(uint32_t core, uint32_t cores)
{
    for (int64_t group = core; group < tiling_->groupCount; group += cores) {
        const int64_t base = GroupBase(group);
        const float mean = CalculateMeanFromBase(base);
        for (int64_t r = 0; r < tiling_->reduceCount; ++r) {
            const int64_t offset = base + ReduceOffset(r);
            StoreOutput(offset, LoadInput(offset) - mean);
        }
    }
}

__aicore__ inline void ProcessLargeMultiCoreReduce(uint32_t core, uint32_t /*cores*/)
{
    const uint32_t coresPerRow = static_cast<uint32_t>(tiling_->coresPerRow);
    const uint32_t group = core / coresPerRow;
    if (group >= static_cast<uint32_t>(tiling_->groupCount)) {
        return;
    }
    const uint32_t localCore = core % coresPerRow;
    const int64_t base = GroupBase(static_cast<int64_t>(group));
    float partial = 0.0f;
    for (int64_t r = static_cast<int64_t>(localCore); r < tiling_->reduceCount;
         r += static_cast<int64_t>(coresPerRow)) {
        partial += LoadInput(base + ReduceOffset(r));
    }
    StoreWorkspace(WorkspaceSlot(static_cast<int64_t>(group), static_cast<int64_t>(localCore)), partial);
    SyncAll();
    if (localCore == 0) {
        float sum = 0.0f;
        for (uint32_t c = 0; c < coresPerRow; ++c) {
            sum += LoadWorkspace(WorkspaceSlot(static_cast<int64_t>(group), static_cast<int64_t>(c)));
        }
        StoreWorkspace(WorkspaceSlot(static_cast<int64_t>(group), 0), sum / static_cast<float>(tiling_->reduceCount));
    }
    SyncAll();
    ProcessOutputBlocks(core, GetBlockNum(), true);
}

__aicore__ inline float LoadInput(int64_t offset)
{
    constexpr int64_t kBlockBytes = 32;
    constexpr int64_t kBlockElements = kBlockBytes / sizeof(T);
    const int64_t alignedOffset = offset / kBlockElements * kBlockElements;
    const int64_t lane = offset - alignedOffset;
    const int64_t validElements = (tiling_->totalCount - alignedOffset < kBlockElements) ?
                                      tiling_->totalCount - alignedOffset :
                                      kBlockElements;
    LocalTensor<T> input = inQueue_.AllocTensor<T>();
    if (validElements == kBlockElements) {
        DataCopy(input, xGm_[static_cast<uint64_t>(alignedOffset)], static_cast<uint32_t>(kBlockElements));
    } else {
        DataCopyExtParams copy{1, static_cast<uint32_t>(validElements * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(kBlockElements - validElements), static_cast<T>(0)};
        DataCopyPad(input, xGm_[static_cast<uint64_t>(alignedOffset)], copy, pad);
    }
    inQueue_.EnQue(input);
    input = inQueue_.DeQue<T>();
    SetFlag<HardEvent::MTE2_S>(eventMte2ToS_);
    WaitFlag<HardEvent::MTE2_S>(eventMte2ToS_);
    const float value = static_cast<float>(input.GetValue(lane));
    inQueue_.FreeTensor(input);
    return value;
}

__aicore__ inline float CalculateMean(int64_t group)
{
    const int64_t base = GroupBase(group);
    return CalculateMeanFromBase(base);
}

__aicore__ inline float CalculateMeanFromBase(int64_t base)
{
    float sum = 0.0f;
    for (int64_t r = 0; r < tiling_->reduceCount; ++r) {
        sum += LoadInput(base + ReduceOffset(r));
    }
    return sum / static_cast<float>(tiling_->reduceCount);
}

__aicore__ inline int64_t GroupFromOffset(int64_t offset) const
{
    int64_t strides[8] = {};
    int64_t stride = 1;
    const int64_t rank = tiling_->keepRank + tiling_->reduceRank;
    for (int64_t i = rank - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= tiling_->dims[i];
    }

    int64_t group = 0;
    int64_t value = offset;
    int64_t keepIndex = 0;
    for (int64_t i = 0; i < rank; ++i) {
        const int64_t coord = value / strides[i];
        value %= strides[i];
        if (tiling_->reduceMask[i] == 0) {
            group += coord * tiling_->keepIndexStrides[keepIndex++];
        }
    }
    return group;
}

__aicore__ inline int64_t BuildMeanCache(int64_t blockOffset, int64_t validElements, uint32_t /*cores*/,
                                         bool useWorkspace, int64_t* cachedGroups, float* cachedMeans)
{
    constexpr int64_t kBlockElements = 32 / sizeof(T);
    int64_t cachedCount = 0;
    for (int64_t lane = 0; lane < validElements; ++lane) {
        const int64_t group = GroupFromOffset(blockOffset + lane);
        bool found = false;
        for (int64_t i = 0; i < cachedCount; ++i) {
            if (cachedGroups[i] == group) {
                found = true;
                break;
            }
        }
        if (!found && cachedCount < kBlockElements) {
            cachedGroups[cachedCount] = group;
            cachedMeans[cachedCount++] = useWorkspace ? LoadWorkspace(WorkspaceSlot(group, 0)) : CalculateMean(group);
        }
    }
    return cachedCount;
}

__aicore__ inline void StoreOutput(int64_t offset, float value)
{
    LocalTensor<T> output = outQueue_.AllocTensor<T>();
    output.SetValue(0, static_cast<T>(value));
    outQueue_.EnQue(output);
    output = outQueue_.DeQue<T>();
    DataCopyExtParams copyOut{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    SetFlag<HardEvent::S_MTE3>(eventSToMte3_);
    WaitFlag<HardEvent::S_MTE3>(eventSToMte3_);
    DataCopyPad(yGm_[static_cast<uint64_t>(offset)], output, copyOut);
    outQueue_.FreeTensor(output);
}

__aicore__ inline void ProcessBlock(int64_t blockOffset, int64_t validElements, uint32_t cores, bool useWorkspace)
{
    constexpr int64_t kBlockElements = 32 / sizeof(T);
    int64_t cachedGroups[kBlockElements] = {};
    float cachedMeans[kBlockElements] = {};
    const int64_t cachedCount = BuildMeanCache(blockOffset, validElements, cores, useWorkspace, cachedGroups,
                                               cachedMeans);

    LocalTensor<T> input = inQueue_.AllocTensor<T>();
    DataCopyExtParams copy{1, static_cast<uint32_t>(validElements * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(kBlockElements - validElements), static_cast<T>(0)};
    DataCopyPad(input, xGm_[static_cast<uint64_t>(blockOffset)], copy, pad);
    inQueue_.EnQue(input);
    input = inQueue_.DeQue<T>();
    SetFlag<HardEvent::MTE2_S>(eventMte2ToS_);
    WaitFlag<HardEvent::MTE2_S>(eventMte2ToS_);
    LocalTensor<T> output = outQueue_.AllocTensor<T>();
    for (int64_t lane = 0; lane < validElements; ++lane) {
        const int64_t group = GroupFromOffset(blockOffset + lane);
        float mean = 0.0f;
        for (int64_t i = 0; i < cachedCount; ++i) {
            if (cachedGroups[i] == group) {
                mean = cachedMeans[i];
                break;
            }
        }
        output.SetValue(lane, static_cast<T>(static_cast<float>(input.GetValue(lane)) - mean));
    }
    outQueue_.EnQue(output);
    output = outQueue_.DeQue<T>();
    DataCopyExtParams copyOut{1, static_cast<uint32_t>(validElements * sizeof(T)), 0, 0, 0};
    SetFlag<HardEvent::S_MTE3>(eventSToMte3_);
    WaitFlag<HardEvent::S_MTE3>(eventSToMte3_);
    DataCopyPad(yGm_[static_cast<uint64_t>(blockOffset)], output, copyOut);
    SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
    outQueue_.FreeTensor(output);
    inQueue_.FreeTensor(input);
}

__aicore__ inline void ProcessOutputBlocks(uint32_t core, uint32_t cores, bool useWorkspace)
{
    constexpr int64_t kBlockElements = 32 / sizeof(T);
    const int64_t blockCount = (tiling_->totalCount + kBlockElements - 1) / kBlockElements;
    for (int64_t block = core; block < blockCount; block += cores) {
        const int64_t blockOffset = block * kBlockElements;
        const int64_t validElements = (tiling_->totalCount - blockOffset < kBlockElements) ?
                                          tiling_->totalCount - blockOffset :
                                          kBlockElements;
        ProcessBlock(blockOffset, validElements, cores, useWorkspace);
    }
}

__aicore__ inline float LoadWorkspace(int64_t offset)
{
    constexpr int64_t kWorkspaceElements = 32 / sizeof(float);
    LocalTensor<float> input = inQueue_.AllocTensor<float>();
    DataCopy(input, workspaceGm_[static_cast<uint64_t>(offset * kWorkspaceElements)],
             static_cast<uint32_t>(kWorkspaceElements));
    inQueue_.EnQue(input);
    input = inQueue_.DeQue<float>();
    SetFlag<HardEvent::MTE2_S>(eventMte2ToS_);
    WaitFlag<HardEvent::MTE2_S>(eventMte2ToS_);
    const float value = input.GetValue(0);
    inQueue_.FreeTensor(input);
    return value;
}

__aicore__ inline void StoreWorkspace(int64_t offset, float value)
{
    constexpr int64_t kWorkspaceElements = 32 / sizeof(float);
    LocalTensor<float> output = outQueue_.AllocTensor<float>();
    for (int64_t i = 0; i < kWorkspaceElements; ++i) {
        output.SetValue(i, 0.0f);
    }
    output.SetValue(0, value);
    outQueue_.EnQue(output);
    output = outQueue_.DeQue<float>();
    SetFlag<HardEvent::S_MTE3>(eventSToMte3_);
    WaitFlag<HardEvent::S_MTE3>(eventSToMte3_);
    DataCopy(workspaceGm_[static_cast<uint64_t>(offset * kWorkspaceElements)], output,
             static_cast<uint32_t>(kWorkspaceElements));
    SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2_);
    outQueue_.FreeTensor(output);
}

#endif
