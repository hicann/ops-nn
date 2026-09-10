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
 * \file foreach_add_list_tiling_arch35.cpp
 * \brief Operator-owned validation plus shared flat RegBase tiling for ForeachAddList.
 */
#include "register/op_impl_registry.h"
#include "../../../foreach_utils/op_host/foreach_flat_regbase_tiling.h"
#include "../../../foreach_utils/op_host/foreach_flat_regbase_validator.h"

namespace optiling {
struct ForeachAddListCompileInfo {};

static bool IsForeachAddListDtype(ge::DataType dtype)
{
    return ForeachFlatRegbaseValidation::IsFloatFamily(dtype) || dtype == ge::DT_INT32 || dtype == ge::DT_INT16 ||
           dtype == ge::DT_INT8 || dtype == ge::DT_UINT8;
}

// alpha scalar tensor dtype follows DtypeScalarToTensor2:
// fp16->fp16, fp32->fp32, int32->int32, bf16->fp32, int16/int8/uint8->int32
static ge::DataType GetExpectedAlphaDtype(ge::DataType dtype)
{
    if (dtype == ge::DT_BF16) {
        return ge::DT_FLOAT;
    }
    if (dtype == ge::DT_INT16 || dtype == ge::DT_INT8 || dtype == ge::DT_UINT8) {
        return ge::DT_INT32;
    }
    return dtype;
}

static ge::graphStatus ValidateForeachAddList(gert::TilingContext* context)
{
    if (ForeachFlatRegbaseValidation::ValidateHomogeneous<2>(context, IsForeachAddListDtype) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto xDesc = context->GetDynamicInputDesc(0, 0);
    auto alphaDesc = context->GetRequiredInputDesc(2);
    auto alphaShape = context->GetRequiredInputShape(2);
    if (xDesc == nullptr || alphaDesc == nullptr || alphaShape == nullptr) {
        OP_LOGE(context, "x1 or alpha metadata is null");
        return ge::GRAPH_FAILED;
    }
    if (alphaDesc->GetDataType() != GetExpectedAlphaDtype(xDesc->GetDataType())) {
        OP_LOGE(context, "alpha dtype does not match the scalar mapping of x1 dtype");
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& alphaStorageShape = alphaShape->GetStorageShape();
    if (ForeachFlatRegbaseValidation::ValidateShape(context, alphaStorageShape, "alpha", 0) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (alphaStorageShape.GetShapeSize() != 1) {
        OP_LOGE(context, "alpha must contain exactly one element");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ForeachAddListTilingFunc(gert::TilingContext* context)
{
    if (ValidateForeachAddList(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ForeachFlatRegbaseTiling::Build<2>(context);
}

static ge::graphStatus TilingParseForForeachAddList([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ForeachAddList)
    .Tiling(ForeachAddListTilingFunc)
    .TilingParse<ForeachAddListCompileInfo>(TilingParseForForeachAddList);
} // namespace optiling
