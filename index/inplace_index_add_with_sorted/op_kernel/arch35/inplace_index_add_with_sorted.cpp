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
 * \file inplace_index_add_with_sorted.cpp
 * \brief A5 (ascend950) kernel entry — 模板参数派发（单默认调度模式 SORTED_SCH_MODE_DEFAULT）
 *
 *   入口路由到负载均衡模板 InplaceIndexAddWithSortedLoadBalance（StageA 核内累加分流 +
 *   StageB 跨核合并），输入 dtype 由编译框架按 def 输入名 var 生成 DTYPE_VAR 宏实例化。
 *   老 InplaceIndexAddWithSortedFix 类保留在 fix.h，入口不再路由，
 *   仅作回退参考（如需回退，取消下方注释并改回 Init/Process 调用）。
 */
#include "inplace_index_add_with_sorted_struct.h"
#include "inplace_index_add_with_sorted_tiling_key.h"
#include "inplace_index_add_with_sorted_load_balance.h"
// #include "inplace_index_add_with_sorted_fix.h"  // 老 Fix 类：保留不路由

template <uint32_t MODE>
__global__ __aicore__ void inplace_index_add_with_sorted(GM_ADDR var, GM_ADDR value, GM_ADDR sorted_indices,
                                                         GM_ADDR pos, GM_ADDR alpha, GM_ADDR output, GM_ADDR workspace,
                                                         GM_ADDR tiling)
{
    if (workspace == nullptr) {
        return;
    }
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    if (userWorkspace == nullptr) {
        return;
    }

    AscendC::TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_NONE_TILING;

    if constexpr (MODE == SORTED_SCH_MODE_DEFAULT) {
        GET_TILING_DATA_WITH_STRUCT(InplaceIndexAddWithSortedTilingData, tilingData, tiling);
        InplaceIndexAddWithSortedLoadBalance<DTYPE_VAR> op(&pipe, &tilingData);
        op.Init(var, value, sorted_indices, pos, alpha, userWorkspace);
        op.Process();
    }
}
