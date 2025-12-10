/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file trans_quant_param_v2_tiling.cpp
 * \brief
 */
#include "trans_quant_param_v2_tiling.h"

namespace optiling {

void GetChipFeature(
    fe::PlatFormInfos& platform_info, const string& lable, const string& platform_info_key, const string& true_value,
    bool& value)
{
    std::string temp_str;
    if (platform_info.GetPlatformRes(lable, platform_info_key, temp_str)) {
        OP_LOGD("NO_OP_NAME", "PLATFORM INFO %s: %s", platform_info_key.c_str(), temp_str.c_str());
        value = (temp_str.find(true_value) != string::npos);
    } else {
        OP_LOGW("NO_OP_NAME", "NO PLATFORM INFO for %s", platform_info_key.c_str());
        value = false;
    }
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    TransQuantParamV2TilingData tiling;
    OP_TILING_CHECK(
        tiling.GetDataSize() > context->GetRawTilingData()->GetCapacity(),
        OP_LOGE(
            context->GetNodeName(), "actual tiling data size %zu > actual tiling data size %zu", tiling.GetDataSize(),
            context->GetRawTilingData()->GetCapacity()),
        return ge::GRAPH_FAILED);
    tiling.SetDataPtr(context->GetRawTilingData()->GetData());
    const gert::StorageShape* scaleShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);
    uint32_t scaleLength = scaleShape->GetStorageShape().GetShapeSize();
    auto offsetShape = context->GetOptionalInputShape(1);
    uint32_t offsetLength = (offsetShape == nullptr) ? 0 : offsetShape->GetStorageShape().GetShapeSize();
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    uint32_t roundMode = (attrs->GetInt(0) == nullptr) ? 0 : static_cast<uint32_t>(*(attrs->GetInt(0)));
    OP_TILING_CHECK(
        roundMode != 0 && roundMode != 1,
        OP_LOGE(context->GetNodeName(), "round mode only support 0 or 1, actual mode is %u", roundMode),
        return ge::GRAPH_FAILED);
    tiling.set_scaleLength(scaleLength);
    tiling.set_offsetLength(offsetLength);
    tiling.set_roundMode(roundMode);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    bool supportL12btBf16 = false;
    GetChipFeature(
        *platformInfo, "AICoreintrinsicDtypeMap", "Intrinsic_data_move_l12bt", "bf16", supportL12btBf16);
    auto scaleInputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleInputShape);
    auto scaleOriShape = scaleInputShape->GetOriginShape();
    bool isScaleValid = (scaleOriShape.GetDimNum() == 1 && scaleOriShape.GetDim(0) > 0) ||
                        // 2 is the dimension of scale dim, shape is (1, n) at this time
                        (scaleOriShape.GetDimNum() == 2 && scaleOriShape.GetDim(0) == 1 && scaleOriShape.GetDim(1) > 0);
    OP_TILING_CHECK(
        !supportL12btBf16 && !isScaleValid,
        OP_LOGE(
            context->GetNodeName(), "shape of scale only support (n, ) and (1, n), but current is %s",
            Ops::Base::ToString(scaleOriShape).c_str()),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    size_t* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    workspace[0] = ascendcPlatform.GetLibApiWorkSpaceSize();
    context->SetBlockDim(1);
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

struct TransQuantParamV2CompileInfo {};
static ge::graphStatus TilingParseForTransQuantParamV2(gert::TilingParseContext* /* context */)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TransQuantParamV2)
    .Tiling(TilingFunc)
    .TilingParse<TransQuantParamV2CompileInfo>(TilingParseForTransQuantParamV2);
} // namespace optiling