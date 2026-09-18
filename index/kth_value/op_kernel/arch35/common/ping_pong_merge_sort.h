/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PING_PONG_MERGE_SORT_H
#define PING_PONG_MERGE_SORT_H

#include "kernel_operator.h"
#include "merge_sort_constants.h"

namespace PingPongMergeSortCommon {
using namespace AscendC;

using MergeSortConstants::DEALING_SORT_NUM_ONCE;
using MergeSortConstants::MERGE_LIST_MAX_NUM;

// One round of 4-way merge: src has sorted runs of length queueLength, merge every 4 adjacent runs into dst.
// Full groups (4 complete runs) are batched in one MrgSort call; an incomplete final group is handled separately.
// Current arch35 callers keep both proposal buffers in UB (>=8 bytes/element
// each). Even allowing all 256 KiB, totalElements <=16384. Queue lengths are
// 32*4^s; while queueCount > 1 the largest is 8192, so uint16 counts do not
// truncate. Recheck this bound if larger UB or a different proposal layout is used.
template <typename T>
__aicore__ inline void MergeStage(LocalTensor<T> dst, LocalTensor<T> src, uint32_t totalElements, uint32_t queueLength,
                                  uint32_t queueCount)
{
    constexpr uint16_t allListsValid = (1U << MERGE_LIST_MAX_NUM) - 1U;
    // Full groups: each group merges 4 runs of queueLength elements.
    uint32_t groupSpan = queueLength * MERGE_LIST_MAX_NUM;
    uint32_t fullGroupCount = queueCount / MERGE_LIST_MAX_NUM;
    uint32_t tailQueueCount = queueCount % MERGE_LIST_MAX_NUM;
    // A divisible queue count can still end in a partial run. Leave that final
    // group to the tail path so its last element count is described exactly.
    if (tailQueueCount == 0U && totalElements != queueCount * queueLength) {
        --fullGroupCount;
        tailQueueCount = MERGE_LIST_MAX_NUM;
    }
    if (fullGroupCount > 0U) {
        uint16_t elementCounts[MERGE_LIST_MAX_NUM] = {
            static_cast<uint16_t>(queueLength), static_cast<uint16_t>(queueLength), static_cast<uint16_t>(queueLength),
            static_cast<uint16_t>(queueLength)};
        uint32_t sourceStride = GetSortOffset<T>(queueLength);
        MrgSortSrcList<T> sourceList(src, src[sourceStride], src[sourceStride * 2U], src[sourceStride * 3U]);
        uint32_t sortedCounts[MERGE_LIST_MAX_NUM];
        // false disables exhausted-list suspension: consume every valid list.
        MrgSort<T, false>(dst, sourceList, elementCounts, sortedCounts, allListsValid,
                          static_cast<int32_t>(fullGroupCount));
    }

    // Tail: 1-4 remaining runs; a fourth run can be shorter than queueLength.
    uint32_t tailOffset = fullGroupCount * groupSpan;
    uint32_t tailElements = totalElements - tailOffset;
    if (tailElements == 0U) {
        return;
    }
    uint32_t proposalTailOffset = GetSortOffset<T>(tailOffset);
    if (tailQueueCount == 1U) {
        DataCopy(dst[proposalTailOffset], src[proposalTailOffset], GetSortOffset<T>(tailElements));
        return;
    }

    uint16_t elementCounts[MERGE_LIST_MAX_NUM] = {0, 0, 0, 0};
    uint32_t sourceOffsets[MERGE_LIST_MAX_NUM] = {tailOffset, tailOffset, tailOffset, tailOffset};
    for (uint32_t list = 0; list < tailQueueCount; ++list) {
        uint32_t sourceOffset = tailOffset + list * queueLength;
        uint32_t sourceRemaining = totalElements - sourceOffset;
        sourceOffsets[list] = sourceOffset;
        elementCounts[list] = static_cast<uint16_t>(sourceRemaining < queueLength ? sourceRemaining : queueLength);
    }
    MrgSortSrcList<T> sourceList(src[GetSortOffset<T>(sourceOffsets[0])], src[GetSortOffset<T>(sourceOffsets[1])],
                                 src[GetSortOffset<T>(sourceOffsets[2])], src[GetSortOffset<T>(sourceOffsets[3])]);
    uint32_t sortedCounts[MERGE_LIST_MAX_NUM];
    uint16_t validBit = static_cast<uint16_t>((1U << tailQueueCount) - 1U);
    MrgSort<T, false>(dst[proposalTailOffset], sourceList, elementCounts, sortedCounts, validBit, 1);
}

// Ping-pong merge sort in UB:
//   1. Sort32 produces repeatTimes sorted runs of DEALING_SORT_NUM_ONCE elements into ping.
//   2. Iteratively 4-way merge adjacent runs, alternating src/dst between ping and pong each round.
//      After each round, run length grows by 4x until only one sorted run remains.
// Returns true when the final sorted result is in ping, false when in pong.
template <typename T>
__aicore__ inline bool SortToProposal(LocalTensor<T> ping, LocalTensor<T> pong, LocalTensor<T> srcValue,
                                      LocalTensor<uint32_t> baseIndex, uint32_t repeatTimes)
{
    Sort32<T>(ping, srcValue, baseIndex, repeatTimes);
    if (repeatTimes <= 1U) {
        return true;
    }
    uint32_t totalElements = repeatTimes * DEALING_SORT_NUM_ONCE;
    uint32_t queueLength = DEALING_SORT_NUM_ONCE;
    uint32_t queueCount = repeatTimes;
    bool sourceIsPing = true;
    while (queueCount > 1U) {
        if (sourceIsPing) {
            MergeStage(pong, ping, totalElements, queueLength, queueCount);
        } else {
            MergeStage(ping, pong, totalElements, queueLength, queueCount);
        }
        sourceIsPing = !sourceIsPing;
        queueLength *= MERGE_LIST_MAX_NUM;
        queueCount = (queueCount + MERGE_LIST_MAX_NUM - 1U) / MERGE_LIST_MAX_NUM;
    }
    return sourceIsPing;
}

} // namespace PingPongMergeSortCommon

#endif // PING_PONG_MERGE_SORT_H
