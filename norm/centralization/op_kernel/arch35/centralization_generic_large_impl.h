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
    // 先按 group 并行算出 mean 存 workspace，再统一按 32B block 并行写输出。
    // 不能逐元素写：非连续/带 stride 的 reduce axes 下，相邻元素常属于不同 group，
    // 不同核并发写同一 cache line 会丢写（表现为偶发、非确定的错误元素）。
    for (int64_t group = core; group < tiling_->groupCount; group += cores) {
        const int64_t base = GroupBase(group);
        const MeanHiLo m = CalculateMeanFromBase(base);
        StoreWorkspace(WorkspaceSlot(group, 0), m.hi);
        StoreWorkspace(WorkspaceSlot(group, 1), m.lo);
    }
    SyncAll();
    ProcessOutputBlocks(core, cores, true);
}

__aicore__ inline void ProcessLargeMultiCoreReduce(uint32_t core, uint32_t /*cores*/)
{
    const uint32_t coresPerRow = static_cast<uint32_t>(tiling_->coresPerRow);
    const int64_t group = static_cast<int64_t>(core) / static_cast<int64_t>(coresPerRow);
    if (group >= tiling_->groupCount) {
        return;
    }
    const uint32_t localCore = core % coresPerRow;
    const int64_t base = GroupBase(group);
    LargeCompensatedSum partial{0.0f, 0.0f};
    for (int64_t r = static_cast<int64_t>(localCore); r < tiling_->reduceCount;
         r += static_cast<int64_t>(coresPerRow)) {
        AddLargeCompensated(partial, LoadInput(base + ReduceOffset(r)));
    }
    StoreWorkspace(WorkspaceSlot(static_cast<int64_t>(group), static_cast<int64_t>(localCore)),
                   TotalLargeCompensated(partial));
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

struct LargeCompensatedSum {
    float sum;
    float correction;
};

struct MeanHiLo {
    float hi;
    float lo;
};

// Kahan/Neumaier 补偿求和：超大 reduce 时抑制 fp32 累加的线性误差增长
// （对齐 bn3d_training_reduce / large_trailing 的补偿归约）。
__aicore__ inline void AddLargeCompensated(LargeCompensatedSum& acc, float value) const
{
    const float next = acc.sum + value;
    const float absSum = acc.sum < 0.0f ? -acc.sum : acc.sum;
    const float absValue = value < 0.0f ? -value : value;
    // inf/NaN 保护：非有限值不做 Neumaier 修正，避免 inf-inf=NaN
    constexpr float kMaxFinite = 3.4028235e38f;
    if (!(absSum <= kMaxFinite && absValue <= kMaxFinite)) {
        acc.sum = next;
        return;
    }
    if (absSum >= absValue) {
        acc.correction += (acc.sum - next) + value;
    } else {
        acc.correction += (value - next) + acc.sum;
    }
    acc.sum = next;
}

__aicore__ inline float TotalLargeCompensated(const LargeCompensatedSum& acc) const { return acc.sum + acc.correction; }

// Dekker 拆分：a = hi + lo，hi 保留高 12 位尾数。
__aicore__ inline void SplitFloat(float a, float& hi, float& lo) const
{
    // Dekker 拆分 (2^12+1) 会临时把指数抬高 12 位；|a| > FLT_MAX/4097 时 4097*a 会溢出成 inf，
    // 进而 hi = inf-inf = NaN。先按 2 的幂缩放到安全区间，拆分后再精确放大回去（缩放不引入误差）。
    constexpr float kSplitSafe = 8.0e34f; // < FLT_MAX/4097
    float scale = 1.0f;
    float value = a;
    float absValue = value < 0.0f ? -value : value;
    while (absValue > kSplitSafe) {
        value *= 0.5f;
        absValue *= 0.5f;
        scale *= 2.0f;
    }
    volatile float c = 4097.0f * value; // 2^12 + 1
    const float h = c - (c - value);
    hi = h * scale;
    lo = (value - h) * scale;
}

// 精确计算 acc - a*b（Dekker 两积）。设备端无 fmaf，用它替代 FMA 得到无舍入的乘法。
__aicore__ inline float ExactSubProduct(float a, float b, float acc) const
{
    float aHi = 0.0f;
    float aLo = 0.0f;
    float bHi = 0.0f;
    float bLo = 0.0f;
    SplitFloat(a, aHi, aLo);
    SplitFloat(b, bHi, bLo);
    volatile float p = a * b;
    volatile float t1 = aHi * bHi - p;
    volatile float t2 = aHi * bLo;
    volatile float t3 = aLo * bHi;
    volatile float t4 = aLo * bLo;
    const float e = ((t1 + t2) + t3) + t4;
    // a*b == p + e（无舍入）
    return (acc - p) - e;
}

// 双 FP32 均值：sum + correction ≈ Σx（Neumaier）。
//   mean  = sum / N
//   r     = sum - mean*N（精确，Dekker） // 除法本身的舍入残差
//   meanLow = (r + correction) / N      // 除法残差 + 求和补偿，一起保留
__aicore__ inline MeanHiLo FinalizeMeanHiLo(const LargeCompensatedSum& acc, int64_t count) const
{
    const float n = static_cast<float>(count);
    MeanHiLo out;
    out.hi = acc.sum / n;
    // mean 非有限（组内含 inf/NaN）时除法残差会产生 NaN；置 0 保持 y = x - mean
    constexpr float kMaxFinite = 3.4028235e38f;
    if (!(out.hi <= kMaxFinite && out.hi >= -kMaxFinite)) {
        out.lo = 0.0f;
        return out;
    }
    const float r = ExactSubProduct(out.hi, n, acc.sum);
    out.lo = (r + acc.correction) / n;
    return out;
}

__aicore__ inline MeanHiLo CalculateMeanFromBase(int64_t base)
{
    const uint16_t numSlots = static_cast<uint16_t>(
        (tiling_->numAccSlots > 0 && tiling_->numAccSlots <= kMaxAccSlots) ? tiling_->numAccSlots : 1);
    LargeCompensatedSum slots[kMaxAccSlots];
    for (uint16_t i = 0; i < kMaxAccSlots; ++i) {
        slots[i] = LargeCompensatedSum{0.0f, 0.0f};
    }
    uint16_t slot = 0;
    for (int64_t r = 0; r < tiling_->reduceCount; ++r) {
        AddLargeCompensated(slots[slot], LoadInput(base + ReduceOffset(r)));
        slot = static_cast<uint16_t>((slot + 1 == numSlots) ? 0 : (slot + 1));
    }
    for (uint16_t active = numSlots; active > 1; active = static_cast<uint16_t>(active / 2)) {
        const uint16_t half = static_cast<uint16_t>(active / 2);
        for (uint16_t g = 0; g < half; ++g) {
            AddLargeCompensated(slots[g], slots[g + half].sum);
            AddLargeCompensated(slots[g], slots[g + half].correction);
        }
    }
    return FinalizeMeanHiLo(slots[0], tiling_->reduceCount);
}

__aicore__ inline MeanHiLo CalculateMean(int64_t group) { return CalculateMeanFromBase(GroupBase(group)); }

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
                                         bool useWorkspace, int64_t* cachedGroups, float* cachedMeans,
                                         float* cachedMeansLow)
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
            if (tiling_->rowParallel != 0) {
                // 行并行：slot0/1 分别保存 hi/lo，完整保留低位
                cachedMeans[cachedCount] = LoadWorkspace(WorkspaceSlot(group, 0));
                cachedMeansLow[cachedCount] = LoadWorkspace(WorkspaceSlot(group, 1));
            } else {
                // 多核归约：workspace 只存 hi
                cachedMeans[cachedCount] = useWorkspace ? LoadWorkspace(WorkspaceSlot(group, 0)) :
                                                          CalculateMean(group).hi;
                cachedMeansLow[cachedCount] = 0.0f;
            }
            ++cachedCount;
        }
    }
    return cachedCount;
}

__aicore__ inline void ProcessBlock(int64_t blockOffset, int64_t validElements, uint32_t cores, bool useWorkspace)
{
    constexpr int64_t kBlockElements = 32 / sizeof(T);
    int64_t cachedGroups[kBlockElements] = {};
    float cachedMeans[kBlockElements] = {};
    float cachedMeansLow[kBlockElements] = {};
    const int64_t cachedCount = BuildMeanCache(blockOffset, validElements, cores, useWorkspace, cachedGroups,
                                               cachedMeans, cachedMeansLow);

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
        float meanLow = 0.0f;
        for (int64_t i = 0; i < cachedCount; ++i) {
            if (cachedGroups[i] == group) {
                mean = cachedMeans[i];
                meanLow = cachedMeansLow[i];
                break;
            }
        }
        // 两步相减，避免均值低阶在单次减法中被丢弃
        output.SetValue(lane, static_cast<T>((static_cast<float>(input.GetValue(lane)) - mean) - meanLow));
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
