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

static ge::graphStatus Tiling4ForeachSubScalarTiling(gert::TilingContext* context)
{
    ForeachCommonTiling tilingObject(context);
    if (tilingObject.Init(FOREACH_SUB_SCALAR_OP_CODE, ForeachInputType::TYPE_SCALAR) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return tilingObject.RunBigScalarKernelTiling();
}

static ge::graphStatus TilingPrepare4ForeachScalarTiling(gert::TilingParseContext* context)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, OP_LOGE(context, "platformInfoPtr is null"), return ge::GRAPH_FAILED);

    auto compileInfoPtr = context->GetCompiledInfo<ForeachCompileInfo>();
    OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context, "compileInfoPtr is null"), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfoPtr->coreNum = ascendcPlatform.GetCoreNum();
    compileInfoPtr->aivCoreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->aicCoreNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ForeachSubScalar)
    .Tiling(Tiling4ForeachSubScalarTiling)
    .TilingParse<ForeachCompileInfo>(TilingPrepare4ForeachScalarTiling);

} // namespace optiling
