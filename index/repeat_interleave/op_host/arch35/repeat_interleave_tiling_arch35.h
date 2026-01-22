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
 * \file repeat_interleave_tiling_arch35.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_REPEAT_INTERLEAVE_TILING_H_
#define OPS_BUILT_IN_OP_TILING_RUNTIME_REPEAT_INTERLEAVE_TILING_H_

#include "register/op_impl_registry.h"

namespace optiling {
struct RepeatInterleaveCompileInfo {
    int64_t coreNumAscendc;
    int64_t ubSizeAscendc;
};
ge::graphStatus RepeatInterleaveTilingForAscendC(gert::TilingContext* context);
ge::graphStatus TilingPrepareRepeatInterleaveForAscendC(gert::TilingParseContext* context);
} // namespace optiling

#endif // OPS_BUILT_IN_OP_TILING_RUNTIME_REPEAT_INTERLEAVE_TILING_H_
