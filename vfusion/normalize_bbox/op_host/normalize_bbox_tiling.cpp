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
 * \file normalize_bbox_tiling.cpp
 * \brief IMPL_OP_OPTILING entry + TilingParse (Ascend950 / arch35 only)
 */

#include "normalize_bbox_tiling.h"
#include "normalize_bbox_regbase_tiling.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "util/math_util.h"

namespace optiling {

static ge::graphStatus NormalizeBBoxTilingFunc(gert::TilingContext* context)
{
    OP_LOGD(context, "NormalizeBBoxTiling start.");
    NormalizeBBoxTilingForRegbase tilingOp(context);
    auto ret = tilingOp.DoTiling();
    OP_CHECK_IF((ret == ge::GRAPH_FAILED), OP_LOGE(context, "NormalizeBBoxTiling failed."), return ge::GRAPH_FAILED);
    OP_LOGD(context, "NormalizeBBoxTiling end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForNormalizeBBox(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForNormalizeBBox start.");
    auto compileInfo = context->GetCompiledInfo<NormalizeBBoxCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->totalCoreNum == 0), OP_LOGE(context, "coreNum is 0."), return ge::GRAPH_FAILED);
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
    OP_LOGD(context, "ub_size_platform is %lu.", compileInfo->ubSizePlatForm);
    OP_LOGD(context, "TilingPrepareForNormalizeBBox end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(NormalizeBBox)
    .Tiling(NormalizeBBoxTilingFunc)
    .TilingParse<NormalizeBBoxCompileInfo>(TilingPrepareForNormalizeBBox);
} // namespace optiling
