/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file index_put_with_sort.cpp
 * \brief
 */

#include "index_put_with_sort_gather_data.h"

extern "C" __global__ __aicore__ void index_put_with_sort(GM_ADDR self, GM_ADDR linear_index, GM_ADDR pos_idx, 
                                                          GM_ADDR values, GM_ADDR self_ref, GM_ADDR workspace, GM_ADDR tiling) {
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    GET_TILING_DATA(tiling_data, tiling);
    AscendC::TPipe tpipe;
# if (defined(DTYPE_SELF))
    if (TILING_KEY_IS(0)) {
        IndexPutWithSort::ScatterDataBetweenKernelOp<DTYPE_SELF, float> op;
        op.Init(self, linear_index, pos_idx, values, userWorkspace, &tiling_data, &tpipe);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        IndexPutWithSort::ScatterDataInKernelOp<DTYPE_SELF, float> op;
        op.Init(self, linear_index, pos_idx, values, userWorkspace, &tiling_data, &tpipe);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        IndexPutWithSort::GatherDataOp<DTYPE_SELF, float> op;
        op.Init(self, linear_index, pos_idx, values, userWorkspace, &tiling_data, &tpipe);
        op.Process();
    }
#endif
}