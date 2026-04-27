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
 * \file grouped_dynamic_mx_quant_tiling_arch35.h
 * \brief
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_DYNAMIC_MX_QUANT_H
#define AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_DYNAMIC_MX_QUANT_H

#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "op_host/tiling_util.h"
#include "graph/types.h"
#include "register/op_impl_registry.h"
#include "quant/grouped_dynamic_mx_quant/op_kernel/arch35/grouped_dynamic_mx_quant_tilingdata.h"

namespace optiling {
struct GroupedDynamicMxQuantCompileInfo {
    int64_t coreNum = 0;
    int64_t ubSize = 0;
};

struct GroupedDynamicMxQuantTilingParam {
    int64_t totalCoreNum = 0;
    int64_t usedCoreNum = 0;
    int64_t blockFactor = 0;
    int64_t tailBlockFactor = 0;
    int64_t uo = 1;
    int64_t maxUbCol = 1;
    int64_t ubFactor = 0;
    int64_t tailUbFactor = 0;
    int64_t blockSize = 0;
    int64_t scaleAlg = 0;
    int64_t preAxisSize = 0;
    int64_t postAxisSize = 1;
    float dstTypeMax = 0.0;
    bool isTailAxis = false;
    int64_t ubSize = 0;
    uint32_t vfLen = 0;
    int64_t tilingKey = 0;
    int64_t groupSize = 1;
    ge::DataType inDtype = ge::DT_FLOAT16;
    ge::DataType outDtype = ge::DT_FLOAT8_E4M3FN;
};

} // namespace optiling
#endif // AIR_CXX_RUNTIME_V2_OP_IMPL_GROUPED_DYNAMIC_MX_QUANT_H