/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UNIQUE_WITH_COUNTS_AND_SORTING_TILING_KEY_H
#define UNIQUE_WITH_COUNTS_AND_SORTING_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

// Flattened ascending values-only Unique; only multi-core radix needs 64-bit global counters.
#define UNIQUE_SORT_MERGE_ONE_CORE 0
#define UNIQUE_SORT_RADIX_ONE_CORE 1
#define UNIQUE_SORT_RADIX_MORE_CORE 2
#define UNIQUE_SORT_MERGE_MORE_CORE 3
#define UNIQUE_SORT_MERGE_INTRA_CORE 4
#define UNIQUE_SORT_SMALL_AXIS_INSERTION 5
#define UNIQUE_SORT_SMALL_AXIS_TWO_STAGE 6
#define UNIQUE_SORT_AXIS_ONE_COPY 7
#define UNIQUE_SORT_MERGE_32_SMALL_AXIS 8
#define UNIQUE_SORT_MERGE_DIRECT 12

ASCENDC_TPL_ARGS_DECL(UniqueWithCountsAndSorting,
                      // Range encoding preserves Sort's schedule numbers. Selection below excludes non-last-axis 9-11.
                      ASCENDC_TPL_UINT_DECL(schedule, ASCENDC_TPL_8_BW, ASCENDC_TPL_UI_RANGE, 1,
                                            UNIQUE_SORT_MERGE_ONE_CORE, UNIQUE_SORT_MERGE_DIRECT),
                      ASCENDC_TPL_BOOL_DECL(useInt64Counters, 0, 1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schedule, ASCENDC_TPL_UI_LIST, UNIQUE_SORT_MERGE_ONE_CORE,
                                                          UNIQUE_SORT_RADIX_ONE_CORE, UNIQUE_SORT_RADIX_MORE_CORE,
                                                          UNIQUE_SORT_MERGE_MORE_CORE, UNIQUE_SORT_MERGE_INTRA_CORE,
                                                          UNIQUE_SORT_SMALL_AXIS_INSERTION,
                                                          UNIQUE_SORT_SMALL_AXIS_TWO_STAGE, UNIQUE_SORT_AXIS_ONE_COPY,
                                                          UNIQUE_SORT_MERGE_32_SMALL_AXIS, UNIQUE_SORT_MERGE_DIRECT),
                                     ASCENDC_TPL_BOOL_SEL(useInt64Counters, 0)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_UINT_SEL(schedule, ASCENDC_TPL_UI_LIST, UNIQUE_SORT_RADIX_MORE_CORE),
                                     ASCENDC_TPL_BOOL_SEL(useInt64Counters, 1)));

#endif
