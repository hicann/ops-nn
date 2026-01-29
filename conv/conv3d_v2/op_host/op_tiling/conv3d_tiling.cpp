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
 * \file conv3d_tiling.cpp
 * \brief
 */

#include "conv3d_base_tiling.h"
#include "tiling_base/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "error_util.h"
#include "conv/common/op_host/op_tiling/conv_tiling_templates_registry.h"
#include "platform/platform_infos_def.h"

using namespace optiling::Conv3dOpsTiling;
using namespace optiling::conv_ops_tiling;

namespace optiling {

CONV_REGISTER_TILING_TEMPLATE(Conv3DV2, Conv3dBaseTiling,
    static_cast<int32_t>(NpuArch::DAV_2201), 0);

static ge::graphStatus Conv3DV2TilingFunc(gert::TilingContext* context)
{
    OP_TILING_CHECK(context == nullptr,
                    CUBE_INNER_ERR_REPORT("Conv3DV2", "context is null"),
    return ge::GRAPH_FAILED);
    OP_LOGD(context->GetNodeType(), "Begin process Conv3DV2TilingFunc");
    return ConvTilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForConv3DV2(gert::TilingParseContext *context) {
    OP_TILING_CHECK(context == nullptr,
                    CUBE_INNER_ERR_REPORT("Conv3DV2", "context is null"),
    return ge::GRAPH_FAILED);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr,
                    CUBE_INNER_ERR_REPORT(context->GetNodeName(), "platformInfoPtr is null"),
    return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<optiling::Conv3DTilingParseInfo>();
    OP_TILING_CHECK(compileInfoPtr == nullptr,
                    CUBE_INNER_ERR_REPORT(context->GetNodeName(), "compileInfoPtr is null"),
    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    platformInfo->GetPlatformRes("version", "SoC_version", compileInfoPtr->socVersion);
    compileInfoPtr->aicoreNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::BT, compileInfoPtr->btSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);
    ascendcPlatform.GetCoreMemBw(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Rate);
    OP_LOGD(context->GetNodeName(),
            "compileInfoPtr->socVersion %s"
            " l1Size:%lu, l2Size:%lu, coreNum:%u, "
            "l0aSize:%lu, l0bSize:%lu, l0cSize:%lu, "
            "l2Rate:%lu, btSize:%lu",
            compileInfoPtr->socVersion.c_str(),
            compileInfoPtr->l1Size,
            compileInfoPtr->l2Size,
            compileInfoPtr->aicoreNum,
            compileInfoPtr->l0aSize,
            compileInfoPtr->l0bSize,
            compileInfoPtr->l0cSize,
            compileInfoPtr->l2Rate,
            compileInfoPtr->btSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Conv3DV2)
.Tiling(Conv3DV2TilingFunc)
.TilingParse<optiling::Conv3DTilingParseInfo>(TilingPrepareForConv3DV2);

} // namespace optiling