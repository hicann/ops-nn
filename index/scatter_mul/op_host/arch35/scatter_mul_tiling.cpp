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
 * \file scatter_mul_tiling.cpp
 * \brief ScatterMul tiling: delegates to the shared scatter_reduce_common tiling.
 */
#include "register/op_impl_registry.h"
#include "../../../scatter_reduce_common/op_host/arch35/scatter_reduce_common_tiling.h"

namespace optiling {
static ge::graphStatus Tiling4ScatterMul(gert::TilingContext* context)
{
    // kernel 经 scatter_reduce_common 的 sort 归约路径使用 SyncAll 全核同步，需设置 BatchMode 保证所有核同批启动
    context->SetScheduleMode(1);
    return ScatterReduceCommonTiling(context);
}

IMPL_OP_OPTILING(ScatterMul)
    .Tiling(Tiling4ScatterMul)
    .TilingParse<ScatterReduceCompileInfo>(TilingPrepareForScatterReduce);
} // namespace optiling
