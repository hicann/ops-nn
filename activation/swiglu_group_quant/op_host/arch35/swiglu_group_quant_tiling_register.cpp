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
 * \file swiglu_group_quant_tiling_registe.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "exe_graph/runtime/tiling_context.h"
#include "log/log.h"
#include "swiglu_group_quant_tiling.h"
#include "swiglu_group_quant_hifp8_tiling.h"

using namespace ge;
namespace optiling {
namespace {
constexpr int64_t BLOCK_QUANT = 0;
constexpr int64_t STATIC_HIFP8_QUANT = 2;
constexpr int64_t DYNAMIC_HIFP8_QUANT = 3;
constexpr size_t ATTR_INDEX_QUANT_MODE = 1;
}

ge::graphStatus TilingPrepareForSwigluGroupQuant(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingForSwigluGroupQuant(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("SwigluGroupQuant", "Tiling context is null"),
               return ge::GRAPH_FAILED);

    auto* attrs = context->GetAttrs();
    int64_t quantMode = BLOCK_QUANT;
    if (attrs != nullptr) {
        auto quantModeAttr = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_MODE);
        if (quantModeAttr != nullptr) {
            quantMode = *quantModeAttr;
        }
    }

    if (quantMode == DYNAMIC_HIFP8_QUANT || quantMode == STATIC_HIFP8_QUANT) {
        SwigluGroupQuantHifp8Tiling tilingObj(context);
        return tilingObj.DoOpTiling();
    }

    SwigluGroupQuantTiling tilingObj(context);
    return tilingObj.DoOpTiling();
}

IMPL_OP_OPTILING(SwigluGroupQuant)
    .Tiling(TilingForSwigluGroupQuant)
    .TilingParse<SwigluGroupQuantCompileInfo>(TilingPrepareForSwigluGroupQuant);
}  // namespace optiling
