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
 * \file scatter_list_arch35_tiling.h
 * \brief arch35(A5) 独立 tiling 入口声明
 */
#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_SCATTER_LIST_ARCH35_TILING_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_SCATTER_LIST_ARCH35_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "graph/types.h"

namespace optiling {

// A5(regbase) 专用 tiling: 自带校验与 tiling 结构, 与其余芯片的实现互不依赖。
// 由共享入口在识别到 regbase SoC 时转交, 其余芯片不进入本函数。
ge::graphStatus Tiling4ScatterListArch35(gert::TilingContext* context);

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_SCATTER_LIST_ARCH35_TILING_H_
