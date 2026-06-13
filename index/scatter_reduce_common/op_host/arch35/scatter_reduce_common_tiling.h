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
 * \file scatter_reduce_common_tiling.h
 * \brief Shared tiling entry for the non-nd scatter reduce ops (ScatterMax/Min/...). Direct-struct tiling.
 */
#ifndef SCATTER_REDUCE_COMMON_TILING_H
#define SCATTER_REDUCE_COMMON_TILING_H

#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
struct ScatterReduceCompileInfo {
    uint64_t coreNum = 0;
};

ge::graphStatus TilingPrepareForScatterReduce(gert::TilingParseContext* context);
ge::graphStatus ScatterReduceCommonTiling(gert::TilingContext* context);
}  // namespace optiling

#endif  // SCATTER_REDUCE_COMMON_TILING_H
