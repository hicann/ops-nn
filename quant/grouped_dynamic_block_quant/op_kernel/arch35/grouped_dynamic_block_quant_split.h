/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file grouped_dynamic_block_quant_split.h
 * \brief GroupedDynamicBlockQuant 本地使用的切分基类。
 *        相对 quant_common 的 GroupedSplit，本实现把 batch 维也折叠进基本块空间，
 *        使得 (batch × group × 行块 × 列块) 整体跨核并行，避免 batch 维在核内串行导致
 *        batch 大而单 batch 块数少的场景并行度骤降。
 */

#ifndef GROUPED_DYNAMIC_BLOCK_QUANT_SPLIT_H
#define GROUPED_DYNAMIC_BLOCK_QUANT_SPLIT_H

namespace GroupedDynamicBlockQuant {
template <typename Derived>
class GroupedSplitLocal {
public:
    __aicore__ inline GroupedSplitLocal(){};
    __aicore__ inline void ProcessBase(const int64_t totalCoreNum, const int64_t blockIdx, const int64_t groupNum,
                                       const int64_t blockColSize, const int64_t blockRowSize,
                                       const int64_t blockRowTailSize, const int64_t blockRowCount,
                                       const int64_t batchNum);
    // 列带流式调度：仅并行 N 轴/batch/group，M 行在核内按 maxUbRow 流式，
    // 避免 N 轴块数已充足时把 M 切碎跨核造成浅切片与跨 group 跳地址导致的搬移/局部性劣化。
    __aicore__ inline void ProcessColBand(const int64_t usedCoreNum, const int64_t coreIdx, const int64_t groupNum,
                                          const int64_t uo, const int64_t batchNum, const int64_t maxUbRow,
                                          const int64_t blockFactor, const int64_t tailBlockFactor);

protected:
    static constexpr int64_t DEFAULT_BATCH_NUM = 1;
    __aicore__ inline void InitGroup(GM_ADDR groupIndex);
    // 具体算子具体实现
    __aicore__ inline void ProcessOneLoop(const int64_t bIdx, const int64_t curBlockRowSize,
                                          const int64_t curBlockColSize, const int64_t blockRowIdx,
                                          const int64_t blockColIdx, const int64_t groupStart,
                                          const int64_t groupIdx) {};

protected:
    AscendC::GlobalTensor<int32_t> groupIndexGm_;
};

template <typename Derived>
__aicore__ inline void GroupedSplitLocal<Derived>::InitGroup(GM_ADDR groupIndex)
{
    groupIndexGm_.SetGlobalBuffer((__gm__ int32_t*)(groupIndex));
}

template <typename Derived>
__aicore__ inline void GroupedSplitLocal<Derived>::ProcessBase(const int64_t totalCoreNum, const int64_t coreIdx,
                                                               const int64_t groupNum, const int64_t blockColSize,
                                                               const int64_t blockRowSize,
                                                               const int64_t blockRowTailSize,
                                                               const int64_t blockRowCount, const int64_t batchNum)
{
    // batchNum<=0按单batch处理，兼容无需batch维的调用
    int64_t realBatchNum = (batchNum > 0) ? batchNum : DEFAULT_BATCH_NUM;
    int64_t coreRotateOffset = 0;
    for (int64_t groupIdx = 0; groupIdx < groupNum; groupIdx++) {
        int64_t groupStart = (groupIdx > 0) ? groupIndexGm_.GetValue(groupIdx - 1) : 0;
        int64_t groupEnd = groupIndexGm_.GetValue(groupIdx);
        int64_t groupSize = groupEnd - groupStart;
        if (groupSize <= 0) {
            continue;
        }

        int64_t blockColCount = ops::CeilDiv(groupSize, blockColSize);
        int64_t blockCount = blockColCount * blockRowCount;
        // 该group叠加batch维后的总基本块数，使batch维也能跨核并行
        int64_t groupTotalBlocks = blockCount * realBatchNum;

        int64_t loopPerCore = 0;
        int64_t blockOffset = 0;

        // 当前group所用核数
        int64_t curUsedCoreNum = (groupTotalBlocks < totalCoreNum) ? groupTotalBlocks : totalCoreNum;
        // 当前是处理这个group的第几个核
        int64_t curCoreIdxInGroup = coreIdx - coreRotateOffset;
        if (curCoreIdxInGroup < 0) {
            curCoreIdxInGroup += totalCoreNum;
        }

        if (curCoreIdxInGroup < curUsedCoreNum) {
            int64_t headCoreNum = groupTotalBlocks % curUsedCoreNum;
            int64_t blockPerHeadCore = ops::CeilDiv(groupTotalBlocks, curUsedCoreNum);
            int64_t blockPerTailCore = groupTotalBlocks / curUsedCoreNum;
            if (curCoreIdxInGroup < headCoreNum) {
                loopPerCore = blockPerHeadCore;
                blockOffset = curCoreIdxInGroup * loopPerCore;
            } else {
                loopPerCore = blockPerTailCore;
                blockOffset = headCoreNum * blockPerHeadCore + (curCoreIdxInGroup - headCoreNum) * loopPerCore;
            }
        }

        coreRotateOffset = (coreRotateOffset + groupTotalBlocks) % totalCoreNum;
        if (loopPerCore == 0) {
            continue;
        }

        int64_t blockColTailSize = groupSize % blockColSize == 0 ? blockColSize : groupSize % blockColSize;

        for (int64_t i = 0; i < loopPerCore; i++) {
            int64_t blockInGroup = blockOffset + i;
            // 解码batch维与行列块
            int64_t bIdx = blockInGroup / blockCount;
            int64_t blockInBatchGroup = blockInGroup % blockCount;
            int64_t blockRowIdx = blockInBatchGroup % blockRowCount;
            int64_t blockColIdx = blockInBatchGroup / blockRowCount;

            int64_t curBlockRowSize = (blockRowIdx == blockRowCount - 1) ? blockRowTailSize : blockRowSize;
            int64_t curBlockColSize = (blockColIdx == blockColCount - 1) ? blockColTailSize : blockColSize;

            static_cast<Derived*>(this)->ProcessOneLoop(bIdx, curBlockRowSize, curBlockColSize, blockRowIdx,
                                                        blockColIdx, groupStart, groupIdx);
        }
    }
}

template <typename Derived>
__aicore__ inline void GroupedSplitLocal<Derived>::ProcessColBand(const int64_t usedCoreNum, const int64_t coreIdx,
                                                                  const int64_t groupNum, const int64_t uo,
                                                                  const int64_t batchNum, const int64_t maxUbRow,
                                                                  const int64_t blockFactor,
                                                                  const int64_t tailBlockFactor)
{
    // 列带流式调度：并行单元仅为 (batch × group × N轴块)，每个单元在核内对 group 的全部行按 maxUbRow 串行流式。
    // 相比 ProcessBase 把 M 行切碎跨核，这里 M 保持纵深流式，减少搬移次数、保持大粒度 DMA，
    // 且各核内连续访问同一列带的相邻行，L2/TLB 局部性更好；仅在 N 轴块数不足以填满核数时才改用 2D 切分补并行。
    constexpr int64_t DEFAULT_BATCH = 1;
    int64_t realBatchNum = (batchNum > 0) ? batchNum : DEFAULT_BATCH;
    const int64_t totalUnits = realBatchNum * groupNum * uo;
    if (coreIdx >= usedCoreNum || totalUnits <= 0) {
        return;
    }
    const int64_t curUsedCoreNum = (totalUnits < usedCoreNum) ? totalUnits : usedCoreNum;
    if (coreIdx >= curUsedCoreNum) {
        return;
    }
    const int64_t headCoreNum = totalUnits % curUsedCoreNum;
    const int64_t unitPerHeadCore = ops::CeilDiv(totalUnits, curUsedCoreNum);
    const int64_t unitPerTailCore = totalUnits / curUsedCoreNum;
    int64_t loopPerCore = 0;
    int64_t unitOffset = 0;
    if (coreIdx < headCoreNum) {
        loopPerCore = unitPerHeadCore;
        unitOffset = coreIdx * loopPerCore;
    } else {
        loopPerCore = unitPerTailCore;
        unitOffset = headCoreNum * unitPerHeadCore + (coreIdx - headCoreNum) * loopPerCore;
    }

    for (int64_t u = 0; u < loopPerCore; u++) {
        const int64_t unit = unitOffset + u;
        const int64_t bIdx = unit / (groupNum * uo);
        const int64_t rest = unit % (groupNum * uo);
        const int64_t nIdx = rest / groupNum;
        const int64_t gIdx = rest % groupNum;
        const int64_t dataLen = (nIdx == uo - 1) ? tailBlockFactor : blockFactor;

        const int64_t groupStart = (gIdx > 0) ? groupIndexGm_.GetValue(gIdx - 1) : 0;
        const int64_t groupEnd = groupIndexGm_.GetValue(gIdx);
        const int64_t groupSize = groupEnd - groupStart;
        if (groupSize <= 0) {
            continue;
        }
        const int64_t inLoopNum = ops::CeilDiv(groupSize, maxUbRow);
        for (int64_t i = 0; i < inLoopNum; i++) {
            const int64_t blockCount = (i == inLoopNum - 1) ? groupSize - i * maxUbRow : maxUbRow;
            static_cast<Derived*>(this)->ProcessOneLoop(bIdx, dataLen, blockCount, nIdx, i, groupStart, gIdx);
        }
    }
}

} // namespace GroupedDynamicBlockQuant

#endif // GROUPED_DYNAMIC_BLOCK_QUANT_SPLIT_H
