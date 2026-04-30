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
 * \file glu_tiling.cpp
 * \brief
 */

#include "glu_tiling.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "op_host/tiling_util.h"
#include "glu_tiling_arch35.h"
#include "glu_tiling_common.h"

namespace optiling {

using Ops::NN::OpTiling::IsRegbaseSocVersion;
using namespace glu_common;

static const int64_t COMMON_BUFFER_SIZE = 8 * 1024;
static const int64_t SINGLE_BUFFER_SIZE = 6 * 1024;

static ge::graphStatus TilingPrepare4Glu(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4Glu enter.");

    auto compileInfo = context->GetCompiledInfo<GluCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_LOGD(context, "Tiling totalCoreNum: %d", compileInfo->totalCoreNum);
    OP_CHECK_IF(
        (compileInfo->totalCoreNum <= 0), OP_LOGE(context, "TilingPrepare4Glu fail to get core num."),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0), OP_LOGE(context, "TilingPrepare4Glu fail to get ub size."),
        return ge::GRAPH_FAILED);

    OP_LOGD(
        context, "TilingPrepare4Glu exit. coreNum: %d ubSize: %lu", compileInfo->totalCoreNum,
        compileInfo->ubSizePlatForm);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4Glu(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling4Glu enter.");

    if (IsRegbaseSocVersion(context)) {
        static thread_local GluRegbaseTiling tilingRegbase;
        auto ret = tilingRegbase.RunFusionKernelTiling(context);
        return ret;
    }

    TilingParam tilingParam;
    if (GetTilingParam(context, tilingParam) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context, "Get Tiling Param Failed.");
        return ge::GRAPH_FAILED;
    }

    size_t workspaceSize = WORK_SPACE_SIZE + tilingParam.coreNum * BYTES_ONE_BLOCK * FP32_DTYPE_BYTES;
    return RunCommonTiling(context, COMMON_BUFFER_SIZE, SINGLE_BUFFER_SIZE, workspaceSize);
}

IMPL_OP_OPTILING(GLU).Tiling(Tiling4Glu).TilingParse<GluCompileInfo>(TilingPrepare4Glu);
} // namespace optiling