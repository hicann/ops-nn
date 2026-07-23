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
 * \file grouped_split_base.h
 * \brief
 */

#ifndef GROUPED_SPLIT_BASE_H
#define GROUPED_SPLIT_BASE_H

namespace GroupedSplitBase {
template <typename Derived>
class GroupedSplit {
public:
    __aicore__ inline GroupedSplit(){};
    __aicore__ inline void ProcessBase(const int64_t totalCoreNum, const int64_t blockIdx, const int64_t groupNum,
                                       const int64_t blockColSize, const int64_t blockRowSize,
                                       const int64_t blockRowTailSize, const int64_t blockRowCount);

protected:
    __aicore__ inline void InitGroup(GM_ADDR groupIndex);
    // 具体算子具体实现
    __aicore__ inline void ProcessOneLoop(const int64_t curBlockRowSize, const int64_t curBlockColSize,
                                          const int64_t blockRowIdx, const int64_t blockColIdx,
                                          const int64_t groupStart, const int64_t groupIdx) {};

protected:
    AscendC::GlobalTensor<int32_t> groupIndexGm_;
};

template <typename Derived>
__aicore__ inline void GroupedSplit<Derived>::InitGroup(GM_ADDR groupIndex)
{
    groupIndexGm_.SetGlobalBuffer((__gm__ int32_t*)(groupIndex));
}

template <typename Derived>
__aicore__ inline void GroupedSplit<Derived>::ProcessBase(const int64_t totalCoreNum, const int64_t coreIdx,
                                                          const int64_t groupNum, const int64_t blockColSize,
                                                          const int64_t blockRowSize, const int64_t blockRowTailSize,
                                                          const int64_t blockRowCount)
{
    // 所有group的总基本块数
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

        int64_t loopPerCore = 0;
        int64_t blockOffset = 0;

        // 当前group所用核数
        int64_t curUsedCoreNum = (blockCount < totalCoreNum) ? blockCount : totalCoreNum;
        // 当前是处理这个group的第几个核
        int64_t curCoreIdxInGroup = coreIdx - coreRotateOffset;
        if (curCoreIdxInGroup < 0) {
            curCoreIdxInGroup += totalCoreNum;
        }

        if (curCoreIdxInGroup < curUsedCoreNum) {
            int64_t headCoreNum = blockCount % curUsedCoreNum;
            int64_t blockPerHeadCore = ops::CeilDiv(blockCount, curUsedCoreNum);
            int64_t blockPerTailCore = blockCount / curUsedCoreNum;
            if (curCoreIdxInGroup < headCoreNum) {
                loopPerCore = blockPerHeadCore;
                blockOffset = curCoreIdxInGroup * loopPerCore;
            } else {
                loopPerCore = blockPerTailCore;
                blockOffset = headCoreNum * blockPerHeadCore + (curCoreIdxInGroup - headCoreNum) * loopPerCore;
            }
        }

        coreRotateOffset = (coreRotateOffset + blockCount) % totalCoreNum;
        if (loopPerCore == 0) {
            continue;
        }

        int64_t blockColTailSize = groupSize % blockColSize == 0 ? blockColSize : groupSize % blockColSize;

        for (int64_t i = 0; i < loopPerCore; i++) {
            int64_t blockInGroup = blockOffset + i;
            int64_t blockRowIdx = blockInGroup % blockRowCount;
            int64_t blockColIdx = blockInGroup / blockRowCount;

            int64_t curBlockRowSize = (blockRowIdx == blockRowCount - 1) ? blockRowTailSize : blockRowSize;
            int64_t curBlockColSize = (blockColIdx == blockColCount - 1) ? blockColTailSize : blockColSize;

            static_cast<Derived*>(this)->ProcessOneLoop(curBlockRowSize, curBlockColSize, blockRowIdx, blockColIdx,
                                                        groupStart, groupIdx);
        }
    }
}

} // namespace GroupedSplitBase

#endif // GROUPED_SPLIT_BASE_H
