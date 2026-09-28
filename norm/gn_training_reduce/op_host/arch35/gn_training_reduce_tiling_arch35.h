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
 * \file gn_training_reduce_tiling_arch35.h
 * \brief GNTrainingReduce host tiling entry points for ascend950 (arch35).
 */

#ifndef GN_TRAINING_REDUCE_TILING_ARCH35_H
#define GN_TRAINING_REDUCE_TILING_ARCH35_H

#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "graph/types.h"

namespace optiling {

// CompileInfo 载体：TilingParse 在编译期把 coreNum / ubSize 填进来。
struct GNTrainingReduceCompileInfo {
    int64_t coreNum; // Vector 核数
    int64_t ubSize;  // UB 字节数
};

// 运行期 tiling 入口（CANN 框架在 kernel launch 前调用一次）。
ge::graphStatus TilingForGNTrainingReduce(gert::TilingContext* context);

// 编译期准备：解析 platform 并把编译期常量写入 CompileInfo。
ge::graphStatus TilingPrepareForGNTrainingReduce(gert::TilingParseContext* context);

} // namespace optiling

#endif
