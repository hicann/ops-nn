/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file update_tensor_desc_tiling_arch35.cpp
 * \brief UpdateTensorDesc host 侧 tiling（arch35 / Ascend 950）。
 *   固定形状极简 tiling：单核 RMW（blockDim=1），tilingKey 恒 0（无 TPL 模板参数），
 *   无 GM workspace。校验链顺序：dtype → format → 维度 → attr → shape，
 *   任一失败 return GRAPH_FAILED，不进入 kernel。
 */

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "../../op_kernel/arch35/update_tensor_desc_tiling_data.h"
#include "update_tensor_desc_tiling_arch35.h"

#include <set>
#include <sstream>
#include <string>

namespace optiling {

static const std::set<ge::DataType> SUPPORTED_X_DTYPES = {ge::DT_BOOL,  ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_DOUBLE,
                                                          ge::DT_INT8,  ge::DT_INT16,   ge::DT_INT32,  ge::DT_INT64,
                                                          ge::DT_UINT8, ge::DT_UINT16,  ge::DT_UINT32, ge::DT_UINT64};

constexpr int64_t MAX_RANK_X = 8;

static std::string JoinInt64(const int64_t* arr, int64_t n)
{
    std::ostringstream oss;
    oss << "[";
    for (int64_t i = 0; i < n; ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << arr[i];
    }
    oss << "]";
    return oss.str();
}

static ge::graphStatus TilingFuncUpdateTensorDesc(gert::TilingContext* context)
{
    const gert::CompileTimeTensorDesc* xDesc = context->GetInputDesc(0);
    const gert::CompileTimeTensorDesc* yDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const gert::StorageShape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const gert::TypedContinuousVector<int64_t>* shapeAttr = attrs->GetListInt(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, shapeAttr);

    const char* nodeName = context->GetNodeName();

    if (SUPPORTED_X_DTYPES.find(xDesc->GetDataType()) == SUPPORTED_X_DTYPES.end()) {
        OP_LOGE(nodeName, "x dtype(%d) not in the supported 12-dtype set", static_cast<int>(xDesc->GetDataType()));
        return ge::GRAPH_FAILED;
    }
    if (yDesc->GetDataType() != ge::DT_INT64) {
        OP_LOGE(nodeName, "y dtype(%d) must be DT_INT64", static_cast<int>(yDesc->GetDataType()));
        return ge::GRAPH_FAILED;
    }
    if (xDesc->GetStorageFormat() != ge::FORMAT_ND || yDesc->GetStorageFormat() != ge::FORMAT_ND) {
        OP_LOGE(nodeName, "format must be ND, x(%d) y(%d)", static_cast<int>(xDesc->GetStorageFormat()),
                static_cast<int>(yDesc->GetStorageFormat()));
        return ge::GRAPH_FAILED;
    }
    const int64_t xRank = static_cast<int64_t>(xShape->GetStorageShape().GetDimNum());
    if (xRank > MAX_RANK_X) {
        OP_LOGE(nodeName, "rank(x)=%ld exceeds max supported dim 8", xRank);
        return ge::GRAPH_FAILED;
    }
    const int64_t shapeNum = static_cast<int64_t>(shapeAttr->GetSize());
    if (shapeNum < 1 || shapeNum > kMaxRank) {
        OP_LOGE(nodeName, "rank(attr shape)=%ld out of range [1, %ld]", shapeNum, kMaxRank);
        return ge::GRAPH_FAILED;
    }
    const int64_t* shapeVals = shapeAttr->GetData();
    int64_t numel = 1;
    for (int64_t i = 0; i < shapeNum; i++) {
        if (shapeVals[i] < 0) {
            OP_LOGE(nodeName, "attr shape[%ld]=%ld is negative", i, shapeVals[i]);
            return ge::GRAPH_FAILED;
        }
        numel *= shapeVals[i];
    }
    if (numel < kDescSize) {
        OP_LOGE(nodeName, "numel(attr shape)=%ld below %ld", numel, kDescSize);
        return ge::GRAPH_FAILED;
    }
    const int64_t yNumel = yShape->GetStorageShape().GetShapeSize();
    if (yNumel < kDescSize) {
        OP_LOGE(nodeName, "numel(y TensorDesc)=%ld below %ld", yNumel, kDescSize);
        return ge::GRAPH_FAILED;
    }

    auto* tiling = context->GetTilingData<UpdateTensorDescTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->rank = shapeNum;
    for (int64_t i = 0; i < shapeNum; i++) {
        tiling->shape[i] = shapeVals[i];
    }
    context->SetBlockDim(1);
    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = 0;
    OP_LOGI(nodeName, "tilingData: rank=%ld, shape=%s, blockDim=1, tilingKey=0(default), workspace=0", tiling->rank,
            JoinInt64(tiling->shape, tiling->rank).c_str());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForUpdateTensorDesc(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UpdateTensorDesc)
    .Tiling(TilingFuncUpdateTensorDesc)
    .TilingParse<UpdateTensorDescCompileInfo>(TilingParseForUpdateTensorDesc);

} // namespace optiling
