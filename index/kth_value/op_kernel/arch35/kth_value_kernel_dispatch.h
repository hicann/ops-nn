/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KTH_VALUE_KERNEL_DISPATCH_H
#define KTH_VALUE_KERNEL_DISPATCH_H

#include "kth_value_axis_one_copy.h"
#include "kth_value_median_utils.h"
#include "kth_value_merge_intra_core.h"
#include "kth_value_merge_sort_one_core.h"
#include "kth_value_merge_sort_more_core.h"
#include "kth_value_non_last_small_axis.h"
#include "kth_value_radix_more_core.h"
#include "kth_value_radix_one_core.h"
#include "kth_value_radix_select.h"
#include "kth_value_small_axis_insertion.h"
#include "kth_value_small_axis_short_rank_select.h"
#include "kth_value_small_axis_two_stage.h"
#include "kth_value_tiling_data.h"
#include "kth_value_tiling_key.h"

namespace KthValue {
using namespace AscendC;

template <bool EnableMedian, uint64_t schId>
__aicore__ inline void RunMergeSortRoute(GM_ADDR x, GM_ADDR y1, GM_ADDR y2, KthValueTilingData* tilingData, TPipe* pipe)
{
    constexpr uint64_t isSort32SmallAxis = (schId == KTH_VALUE_SCHID_SORT32_SMALL_AXIS);
    if constexpr (IsSameType<bfloat16_t, DTYPE_X>::value) {
        KthValueMergeSortOneCore<DTYPE_X, float, isSort32SmallAxis, EnableMedian> op;
        op.Init(x, y1, y2, tilingData, pipe);
        op.Process();
    } else if constexpr (IsSameType<float, DTYPE_X>::value || IsSameType<half, DTYPE_X>::value) {
        KthValueMergeSortOneCore<DTYPE_X, DTYPE_X, isSort32SmallAxis, EnableMedian> op;
        op.Init(x, y1, y2, tilingData, pipe);
        op.Process();
    }
}

template <bool EnableMedian, uint64_t isInt32>
__aicore__ inline void RunRadixMoreCoreRoute(GM_ADDR x, GM_ADDR y1, GM_ADDR y2, GM_ADDR workspace,
                                             KthValueTilingData* tilingData, TPipe* pipe)
{
    constexpr bool enableNanMode = EnableMedian && IS_MEDIAN_FLOAT_TYPE<DTYPE_X>;
    if constexpr (sizeof(DTYPE_X) == sizeof(uint8_t)) {
        if constexpr (isInt32 == 1) {
            KthValueRadixMoreCore<DTYPE_X, uint32_t, uint8_t, enableNanMode> op;
            op.Init(x, y1, y2, workspace, tilingData, pipe);
            op.Process();
        } else {
            KthValueRadixMoreCore<DTYPE_X, int64_t, uint8_t, enableNanMode> op;
            op.Init(x, y1, y2, workspace, tilingData, pipe);
            op.Process();
        }
    } else if constexpr (sizeof(DTYPE_X) == sizeof(uint16_t)) {
        if constexpr (isInt32 == 1) {
            KthValueRadixMoreCore<DTYPE_X, uint32_t, uint16_t, enableNanMode> op;
            op.Init(x, y1, y2, workspace, tilingData, pipe);
            op.Process();
        } else {
            KthValueRadixMoreCore<DTYPE_X, int64_t, uint16_t, enableNanMode> op;
            op.Init(x, y1, y2, workspace, tilingData, pipe);
            op.Process();
        }
    } else if constexpr (sizeof(DTYPE_X) == sizeof(uint32_t)) {
        if constexpr (isInt32 == 1) {
            KthValueRadixMoreCore<DTYPE_X, uint32_t, uint32_t, enableNanMode> op;
            op.Init(x, y1, y2, workspace, tilingData, pipe);
            op.Process();
        } else {
            KthValueRadixMoreCore<DTYPE_X, int64_t, uint32_t, enableNanMode> op;
            op.Init(x, y1, y2, workspace, tilingData, pipe);
            op.Process();
        }
    } else if constexpr (sizeof(DTYPE_X) == sizeof(uint64_t)) {
        if constexpr (isInt32 == 1) {
            KthValueRadixMoreCore<DTYPE_X, uint32_t, uint64_t, enableNanMode> op;
            op.Init(x, y1, y2, workspace, tilingData, pipe);
            op.Process();
        } else {
            KthValueRadixMoreCore<DTYPE_X, int64_t, uint64_t, enableNanMode> op;
            op.Init(x, y1, y2, workspace, tilingData, pipe);
            op.Process();
        }
    }
}

template <bool EnableMedian>
__aicore__ inline void RunSmallAxisInsertionRoute(GM_ADDR x, GM_ADDR y1, GM_ADDR y2, KthValueTilingData* tilingData,
                                                  TPipe* pipe)
{
    constexpr bool enableNanMode = EnableMedian && IS_MEDIAN_FLOAT_TYPE<DTYPE_X>;
    if constexpr (IsSameType<bfloat16_t, DTYPE_X>::value) {
        KthValueSmallAxisInsertion<DTYPE_X, float, enableNanMode> op;
        op.Init(x, y1, y2, tilingData, pipe);
        op.Process();
    } else {
        KthValueSmallAxisInsertion<DTYPE_X, DTYPE_X, enableNanMode> op;
        op.Init(x, y1, y2, tilingData, pipe);
        op.Process();
    }
}

template <bool EnableMedian>
__aicore__ inline void RunMergeMoreCoreRoute(GM_ADDR x, GM_ADDR y1, GM_ADDR y2, GM_ADDR workspace,
                                             KthValueTilingData* tilingData, TPipe* pipe)
{
    if constexpr (IsSameType<float, DTYPE_X>::value) {
        KthValueMergeSortMoreCore<DTYPE_X, DTYPE_X, false, int64_t, EnableMedian> op;
        op.Init(x, y1, y2, workspace, tilingData, pipe);
        op.Process();
    }
}

template <bool EnableMedian>
__aicore__ inline void RunMergeIntraCoreRoute(GM_ADDR x, GM_ADDR y1, GM_ADDR y2, GM_ADDR workspace,
                                              KthValueTilingData* tilingData, TPipe* pipe)
{
    if constexpr (IsSameType<float, DTYPE_X>::value) {
        KthValueMergeIntraCore<DTYPE_X, int64_t, false, EnableMedian> op;
        op.Init(x, y1, y2, workspace, tilingData, pipe);
        op.Process();
    }
}

template <bool EnableMedian, bool useMergeSort>
__aicore__ inline void RunNonLastSmallAxisRoute(GM_ADDR x, GM_ADDR y1, GM_ADDR y2, GM_ADDR workspace,
                                                KthValueTilingData* tilingData, TPipe* pipe)
{
    constexpr bool enableNanMode = EnableMedian && IS_MEDIAN_FLOAT_TYPE<DTYPE_X>;
    if constexpr (useMergeSort) {
        if constexpr (IsSameType<DTYPE_X, float>::value || IsSameType<DTYPE_X, half>::value ||
                      IsSameType<DTYPE_X, bfloat16_t>::value) {
            KthValueNonLastSmallAxis<DTYPE_X, false, true, enableNanMode> op;
            op.Init(x, y1, y2, workspace, tilingData, pipe);
            op.Process();
        }
    } else {
        KthValueNonLastSmallAxis<DTYPE_X, false, false, enableNanMode> op;
        op.Init(x, y1, y2, workspace, tilingData, pipe);
        op.Process();
    }
}

__aicore__ inline void RunAxisOneCopyRoute(GM_ADDR x, GM_ADDR y1, GM_ADDR y2, GM_ADDR workspace,
                                           KthValueTilingData* tilingData, TPipe* pipe)
{
    KthValueAxisOneCopy<DTYPE_X> op;
    op.Init(x, y1, y2, workspace, tilingData, pipe);
    op.Process();
}

template <bool EnableMedian>
__aicore__ inline void RunRadixOneCoreRoute(GM_ADDR x, GM_ADDR y1, GM_ADDR y2, KthValueTilingData* tilingData,
                                            TPipe* pipe)
{
    constexpr bool enableNanMode = EnableMedian && IS_MEDIAN_FLOAT_TYPE<DTYPE_X>;
    KthValueRadixOneCore<DTYPE_X, enableNanMode> op;
    op.Init(x, y1, y2, tilingData, pipe);
    op.Process();
}

template <bool EnableMedian>
__aicore__ inline void RunRadixSelectRoute(GM_ADDR x, GM_ADDR y1, GM_ADDR y2, GM_ADDR workspace,
                                           KthValueTilingData* tilingData, TPipe* pipe)
{
    constexpr bool enableNanMode = EnableMedian && IS_MEDIAN_FLOAT_TYPE<DTYPE_X>;
    if constexpr (sizeof(DTYPE_X) == 1) {
        KthValueRadixSelect<DTYPE_X, uint8_t, enableNanMode> op;
        op.Init(x, y1, y2, workspace, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == 2) {
        KthValueRadixSelect<DTYPE_X, uint16_t, enableNanMode> op;
        op.Init(x, y1, y2, workspace, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == 4) {
        KthValueRadixSelect<DTYPE_X, uint32_t, enableNanMode> op;
        op.Init(x, y1, y2, workspace, tilingData, pipe);
        op.Process();
    } else if constexpr (sizeof(DTYPE_X) == 8) {
        KthValueRadixSelect<DTYPE_X, uint64_t, enableNanMode> op;
        op.Init(x, y1, y2, workspace, tilingData, pipe);
        op.Process();
    }
}

template <bool EnableMedian>
__aicore__ inline void RunSmallAxisTwoStageRoute(GM_ADDR x, GM_ADDR y1, GM_ADDR y2, KthValueTilingData* tilingData,
                                                 TPipe* pipe)
{
    constexpr bool enableNanMode = EnableMedian && IS_MEDIAN_FLOAT_TYPE<DTYPE_X>;
    KthValueSmallAxisTwoStage<DTYPE_X, enableNanMode> op;
    op.Init(x, y1, y2, tilingData, pipe);
    op.Process();
}

__aicore__ inline void RunSmallAxisShortRankSelectRoute(GM_ADDR x, GM_ADDR y1, GM_ADDR y2,
                                                        KthValueTilingData* tilingData, TPipe* pipe)
{
    if constexpr (sizeof(DTYPE_X) == 8) {
        KthValueSmallAxisShortRankSelect<DTYPE_X> op;
        op.Init(x, y1, y2, tilingData, pipe);
        op.Process();
    }
}

template <bool EnableMedian, uint64_t schId>
__aicore__ inline bool TryRunSmallAxisRoute(GM_ADDR x, GM_ADDR y1, GM_ADDR y2, KthValueTilingData* tilingData,
                                            TPipe* pipe)
{
    if constexpr (schId == KTH_VALUE_SCHID_SMALL_AXIS_INSERTION) {
        RunSmallAxisInsertionRoute<EnableMedian>(x, y1, y2, tilingData, pipe);
        return true;
    }
    if constexpr (schId == KTH_VALUE_SCHID_SMALL_AXIS_TWO_STAGE) {
        RunSmallAxisTwoStageRoute<EnableMedian>(x, y1, y2, tilingData, pipe);
        return true;
    }
    if constexpr (schId == KTH_VALUE_SCHID_SMALL_AXIS_SHORT_RANK_SELECT) {
        RunSmallAxisShortRankSelectRoute(x, y1, y2, tilingData, pipe);
        return true;
    }
    return false;
}

// EnableMedian selects the shared Median/NanMedian behavior. tilingData->medianMode
// distinguishes propagate-NaN Median from ignore-NaN NanMedian at runtime.
template <bool EnableMedian, uint64_t schId, uint64_t isInt32>
__aicore__ inline void Dispatch(GM_ADDR x, GM_ADDR y1, GM_ADDR y2, GM_ADDR workspace, KthValueTilingData* tilingData,
                                TPipe* pipe)
{
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if constexpr (schId == KTH_VALUE_SCHID_AXIS_ONE_COPY) {
        RunAxisOneCopyRoute(x, y1, y2, userWorkspace, tilingData, pipe);
    } else if constexpr (schId == KTH_VALUE_SCHID_MERGE_SORT || schId == KTH_VALUE_SCHID_SORT32_SMALL_AXIS) {
        RunMergeSortRoute<EnableMedian, schId>(x, y1, y2, tilingData, pipe);
    } else if constexpr (schId == KTH_VALUE_SCHID_RADIX_ONE_CORE) {
        RunRadixOneCoreRoute<EnableMedian>(x, y1, y2, tilingData, pipe);
    } else if constexpr (schId == KTH_VALUE_SCHID_RADIX_MORE_CORE) {
        RunRadixMoreCoreRoute<EnableMedian, isInt32>(x, y1, y2, userWorkspace, tilingData, pipe);
    } else if constexpr (schId == KTH_VALUE_SCHID_RADIX_SELECT) {
        RunRadixSelectRoute<EnableMedian>(x, y1, y2, userWorkspace, tilingData, pipe);
    } else if constexpr (schId == KTH_VALUE_SCHID_SMALL_AXIS_INSERTION ||
                         schId == KTH_VALUE_SCHID_SMALL_AXIS_TWO_STAGE ||
                         schId == KTH_VALUE_SCHID_SMALL_AXIS_SHORT_RANK_SELECT) {
        TryRunSmallAxisRoute<EnableMedian, schId>(x, y1, y2, tilingData, pipe);
    } else if constexpr (schId == KTH_VALUE_SCHID_MERGE_MORE_CORE) {
        RunMergeMoreCoreRoute<EnableMedian>(x, y1, y2, userWorkspace, tilingData, pipe);
    } else if constexpr (schId == KTH_VALUE_SCHID_MERGE_INTRA_CORE) {
        RunMergeIntraCoreRoute<EnableMedian>(x, y1, y2, userWorkspace, tilingData, pipe);
    } else if constexpr (schId == KTH_VALUE_SCHID_NON_LAST_SMALL_AXIS) {
        RunNonLastSmallAxisRoute<EnableMedian, true>(x, y1, y2, userWorkspace, tilingData, pipe);
    } else if constexpr (schId == KTH_VALUE_SCHID_NON_LAST_SMALL_AXIS_RADIX) {
        RunNonLastSmallAxisRoute<EnableMedian, false>(x, y1, y2, userWorkspace, tilingData, pipe);
    }
}

} // namespace KthValue

#endif
