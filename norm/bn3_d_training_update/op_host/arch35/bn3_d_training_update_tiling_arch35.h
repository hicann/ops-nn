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
 * \file bn3_d_training_update_tiling_arch35.h
 * \brief
 */
#ifndef BN3_D_TRAINING_UPDATE_TILING_ARCH35_H
#define BN3_D_TRAINING_UPDATE_TILING_ARCH35_H

#include "exe_graph/runtime/tiling_context.h" // gert::TilingContext
#include "graph/types.h"                      // ge::graphStatus

namespace optiling {

struct BN3DTrainingUpdateCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

ge::graphStatus TilingFuncBN3DTrainingUpdate(gert::TilingContext* context);
ge::graphStatus TilingPrepareForBN3DTrainingUpdate(gert::TilingParseContext* context);

} // namespace optiling

#endif
