/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../foreach_utils/op_host/foreach_tiling_class.h"

namespace optiling {

constexpr int64_t SMALL_INT_SCALAR_MAX_DATA_COUNT = 64;
constexpr uint64_t FOREACH_MUL_LIST_INT8_SCALAR_TILING_KEY = 17;

static bool SetSmallIntScalarTiling(gert::TilingContext* context)
{
    auto inputDesc = context->GetInputDesc(0);
    if (inputDesc == nullptr) {
        return false;
    }
    auto dataType = inputDesc->GetDataType();
    if (dataType != ge::DT_INT8) {
        return false;
    }

    int64_t totalDataCount = 0;
    uint16_t tensorCount = 0;
    int64_t tensorDataCountList[MAX_TENSOR_CONT] = {0};
    uint16_t tensorStartList[MAX_CORE_CONT] = {0};
    uint16_t tensorEndList[MAX_CORE_CONT] = {0};
    int64_t tensorStartOffsetList[MAX_CORE_CONT] = {0};
    int64_t tensorEndOffsetList[MAX_CORE_CONT] = {0};
    for (uint32_t i = 0; i < MAX_TENSOR_CONT; ++i) {
        auto inputTensor = context->GetDynamicInputTensor(0, i);
        if (inputTensor == nullptr) {
            break;
        }
        int64_t dataCount = inputTensor->GetStorageShape().GetShapeSize();
        if (dataCount <= 0 || i >= MAX_CORE_CONT) {
            return false;
        }
        tensorDataCountList[i] = dataCount;
        tensorStartList[i] = static_cast<uint16_t>(i);
        tensorEndList[i] = static_cast<uint16_t>(i);
        tensorEndOffsetList[i] = dataCount - 1;
        totalDataCount += dataCount;
        ++tensorCount;
        if (totalDataCount > SMALL_INT_SCALAR_MAX_DATA_COUNT) {
            return false;
        }
    }
    if (totalDataCount <= 0 || tensorCount == 0) {
        return false;
    }

    ForeachCommonTilingData tinyTilingData;
    tinyTilingData.set_inputsTensorUbSize(SMALL_INT_SCALAR_MAX_DATA_COUNT);
    tinyTilingData.set_tensorDataCountList(tensorDataCountList);
    tinyTilingData.set_tensorStartList(tensorStartList);
    tinyTilingData.set_tensorEndList(tensorEndList);
    tinyTilingData.set_tensorStartOffsetList(tensorStartOffsetList);
    tinyTilingData.set_tensorEndOffsetList(tensorEndOffsetList);
    tinyTilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tinyTilingData.GetDataSize());
    context->SetTilingKey(FOREACH_MUL_LIST_INT8_SCALAR_TILING_KEY);
    context->SetBlockDim(tensorCount);
    return true;
}

static ge::graphStatus Tiling4ForeachMulListTiling(gert::TilingContext* context)
{
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(BINARY_LIST_OP_CODE) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto status = tilingObject.RunBigKernelTiling();
    if (status != ge::GRAPH_SUCCESS) {
        return status;
    }

    SetSmallIntScalarTiling(context);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4ForeachTiling([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ForeachMulList)
    .Tiling(Tiling4ForeachMulListTiling)
    .TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachTiling);

} // namespace optiling
