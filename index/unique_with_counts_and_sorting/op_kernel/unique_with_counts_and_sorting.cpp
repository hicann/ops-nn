/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <type_traits>

#include "op_kernel/math_util.h"

#include "arch35/unique_with_counts_and_sorting_tiling_data.h"
#include "arch35/unique_with_counts_and_sorting_tiling_key.h"

#include "arch35/sort/unique_merge_sort_big_size.h"
#include "arch35/sort/unique_sort_axis_one_copy.h"
#include "arch35/sort/unique_sort_merge_intra_core.h"
#include "arch35/sort/unique_sort_merge_sort.h"
#include "arch35/sort/unique_sort_radix_sort_more_core.h"
#include "arch35/sort/unique_sort_radix_sort_one_core.h"
#include "arch35/sort/unique_sort_small_axis_insertion.h"
#include "arch35/sort/unique_sort_small_axis_two_stage.h"
#include "arch35/uniqueconsecutive/unique_consecutive_kernel.h"

using namespace AscendC;
template <typename Op>
__aicore__ inline void RunUniqueSort(GM_ADDR x, GM_ADDR values, GM_ADDR indices, GM_ADDR scratch,
                                     const UniqueSortRegBaseTilingData* data, TPipe* pipe)
{
    Op sort;
    sort.Init(x, values, indices, scratch, data, pipe);
    sort.Process();
}
template <uint64_t schedule, bool useInt64Counters>
__global__ __aicore__ void unique_with_counts_and_sorting(GM_ADDR x, GM_ADDR y, GM_ADDR inverse, GM_ADDR counts,
                                                          GM_ADDR shape_out, GM_ADDR workspace, GM_ADDR tiling)
{
    // Every fused route crosses a SyncAll phase boundary; multi-core UC also synchronizes internally.
    // Cross-core scheduling is independent of whether the selected Sort template needs SIMT UB scratch.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(UniqueWithCountsAndSortingTilingData);
    GET_TILING_DATA_WITH_STRUCT(UniqueWithCountsAndSortingTilingData, data, tiling);
    using Value = DTYPE_X;
    using Counter = std::conditional_t<useInt64Counters, int64_t, uint32_t>;
    using Bits = std::conditional_t<
        sizeof(Value) == 1, uint8_t,
        std::conditional_t<sizeof(Value) == 2, uint16_t, std::conditional_t<sizeof(Value) == 4, uint32_t, uint64_t>>>;
    using Origin = std::conditional_t<
        sizeof(Value) == 1, int8_t,
        std::conditional_t<sizeof(Value) == 2, int16_t, std::conditional_t<sizeof(Value) == 4, int32_t, int64_t>>>;
    using Convert = std::conditional_t<std::is_same_v<Value, bfloat16_t>, float, Value>;
    TPipe pipe;
    GM_ADDR values = GetUserWorkspace(workspace);
    GM_ADDR metadata = values + data.valuesBytes;
    GM_ADDR indices = data.indicesBytes == 0 ? y : metadata;
    GM_ADDR scratch = metadata + data.indicesBytes;
    if constexpr (schedule == UNIQUE_SORT_RADIX_ONE_CORE) {
        RunUniqueSort<UniqueSort::SortRadixOneCore<Value, int32_t, false, false>>(x, values, y, scratch, &data.sort,
                                                                                  &pipe);
    } else if constexpr (schedule == UNIQUE_SORT_RADIX_MORE_CORE) {
        RunUniqueSort<UniqueSort::SortRadixMoreCore<Value, int64_t, Bits, Counter, false, false, false, true>>(
            x, values, y, scratch, &data.sort, &pipe);
    } else if constexpr (schedule == UNIQUE_SORT_MERGE_ONE_CORE || schedule == UNIQUE_SORT_MERGE_32_SMALL_AXIS) {
        if constexpr (std::is_same_v<Value, float> || std::is_same_v<Value, half> ||
                      std::is_same_v<Value, bfloat16_t>) {
            RunUniqueSort<UniqueSort::MergeSort<Value, int32_t, Convert, false>>(x, values, indices, scratch,
                                                                                 &data.sort, &pipe);
        }
    } else if constexpr (schedule == UNIQUE_SORT_MERGE_MORE_CORE || schedule == UNIQUE_SORT_MERGE_DIRECT) {
        if constexpr (std::is_same_v<Value, float>) {
            RunUniqueSort<
                UniqueSort::MergeSortBigSize<Value, Value, false, int32_t, schedule == UNIQUE_SORT_MERGE_DIRECT>>(
                x, values, indices, scratch, &data.sort, &pipe);
        }
    } else if constexpr (schedule == UNIQUE_SORT_MERGE_INTRA_CORE) {
        if constexpr (std::is_same_v<Value, float>) {
            RunUniqueSort<UniqueSort::SortMergeIntraCore<Value, int32_t, false>>(x, values, indices, scratch,
                                                                                 &data.sort, &pipe);
        }
    } else if constexpr (schedule == UNIQUE_SORT_SMALL_AXIS_INSERTION) {
        RunUniqueSort<UniqueSort::SortSmallAxisInsertion<Value, Convert, int32_t, false>>(x, values, indices, scratch,
                                                                                          &data.sort, &pipe);
    } else if constexpr (schedule == UNIQUE_SORT_SMALL_AXIS_TWO_STAGE) {
        RunUniqueSort<UniqueSort::SortSmallAxisTwoStage<Value, int32_t, false>>(x, values, indices, scratch, &data.sort,
                                                                                &pipe);
    } else if constexpr (schedule == UNIQUE_SORT_AXIS_ONE_COPY) {
        RunUniqueSort<UniqueSort::SortAxisOneCopy<Value, int32_t>>(x, values, indices, scratch, &data.sort, &pipe);
    }
    SyncAll();
    pipe.Reset();
    if (data.uniqueSingleCore != 0) {
        if (GetBlockIdx() != 0) {
            return;
        }
        UniqueConsecutiveSingleCoreKerenl<Origin, Value, int64_t, false, false> unique(&pipe);
        unique.Init(values, y, inverse, counts, shape_out, metadata, &data.unique);
        unique.Process();
    } else {
        UniqueConsecutiveMutilCoreKerenl<Origin, Value, int64_t, false, false> unique(&pipe);
        unique.Init(values, y, inverse, counts, shape_out, metadata, &data.unique);
        unique.Process();
    }
}
