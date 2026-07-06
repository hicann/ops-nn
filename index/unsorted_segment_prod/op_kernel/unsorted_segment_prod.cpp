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
 * \file unsorted_segment_prod.cpp
 * \brief unsorted_segment_prod kernel
 */

#include "../unsorted_segment_common/arch35/unsorted_segment_struct.h"
#include "./unsorted_segment_prod.h"

using namespace AscendC;

#define TEMPLATE_SIMT_TILING_KEY 1000

extern "C" __global__ __aicore__ void unsorted_segment_prod(GM_ADDR x, GM_ADDR segment_ids, GM_ADDR num_segments,
                                                            GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    TPipe pipe;
    REGISTER_TILING_DEFAULT(UnsortedSegment::UnsortedSegmentSimtTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);

    if (TILING_KEY_IS(TEMPLATE_SIMT_TILING_KEY)) {
        REGISTER_TILING_FOR_TILINGKEY("TILING_KEY_VAR == 1000", UnsortedSegment::UnsortedSegmentSimtTilingData);
        GET_TILING_DATA_WITH_STRUCT(UnsortedSegment::UnsortedSegmentSimtTilingData, tilingData, tiling);
        UnsortedSegmentProd::KernelUnsortedSegmentProd<DTYPE_X, DTYPE_SEGMENT_IDS> op(&tilingData, &pipe);
        op.Init(x, segment_ids, output);
        op.Process();
    }
}
