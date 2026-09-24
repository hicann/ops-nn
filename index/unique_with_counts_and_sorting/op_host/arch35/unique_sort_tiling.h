/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file unique_sort_tiling.h
 * \brief sort ac tiling impl
 */
#pragma once
#include "register/op_impl_registry.h"
#include "../../op_kernel/arch35/sort/unique_sort_tiling_data.h"
namespace unique_with_counts_and_sorting_sort {
struct UniqueSortPlan {
    UniqueSortRegBaseTilingData tiling{};
    uint64_t schedule = 0;
    uint64_t workspaceBytes = 0;
    uint32_t cores = 0;
    uint32_t usableUb = 0;
    bool needsIndices = false;
};
ge::graphStatus PlanUniqueSort(gert::TilingContext* context, int64_t count, ge::DataType dtype, uint32_t ub,
                               uint32_t cores, UniqueSortPlan& result, int requestedSchedule = -1);
} // namespace unique_with_counts_and_sorting_sort
