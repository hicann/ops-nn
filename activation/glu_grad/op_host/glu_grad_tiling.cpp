/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "glu_grad_tiling.h"
#include "register/tilingdata_base.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/math_util.h"
#include "glu_grad_tiling_common.h"

namespace optiling {

using namespace glu_grad_common;

static const int64_t BUFFER_SIZE_FACTOR = 10;
static const int64_t BUFFER_SIZE_FACTOR_LOW_PREC = 14;
constexpr int64_t RESERVED_SIZE_8K = 8 * 1024;

static ge::graphStatus TilingPrepare4GluGrad(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepare4GluGrad enter.");

    auto compileInfo = context->GetCompiledInfo<GluGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    OP_LOGD(context, "Tiling totalCoreNum: %d", compileInfo->totalCoreNum);
    OP_CHECK_IF(
        (compileInfo->totalCoreNum <= 0), OP_LOGE(context, "TilingPrepare4GluGrad fail to get core num."),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = static_cast<int64_t>(ubSizePlatForm);
    OP_CHECK_IF(
        (compileInfo->ubSizePlatForm <= 0), OP_LOGE(context, "TilingPrepare4GluGrad fail to get ub size."),
        return ge::GRAPH_FAILED);

    OP_LOGD(
        context, "TilingPrepare4GluGrad exit. coreNum: %d ubSize: %lu", compileInfo->totalCoreNum,
        compileInfo->ubSizePlatForm);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4GluGrad(gert::TilingContext* context)
{
    OP_LOGD(context, "Tiling4GluGrad enter.");

    auto compileInfo = context->GetCompileInfo<GluGradCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);

    int64_t ubSizePlatForm = compileInfo->ubSizePlatForm - RESERVED_SIZE_8K;
    OP_CHECK_IF(
        (ubSizePlatForm <= 0), OP_LOGE(context, "Tiling4GluGrad fail to get ub size."), return ge::GRAPH_FAILED);

    int64_t bufferSizeFactor = BUFFER_SIZE_FACTOR;
    auto inputDesc = context->GetInputDesc(SELF_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDesc);
    if (inputDesc->GetDataType() == ge::DT_BF16 || inputDesc->GetDataType() == ge::DT_FLOAT16) {
        bufferSizeFactor = BUFFER_SIZE_FACTOR_LOW_PREC;
    }

    int64_t commonBufferSize = ubSizePlatForm / (bufferSizeFactor * sizeof(float));

    OP_LOGD(context, "Tiling4GluGrad ubSize: %ld, commonBufferSize: %ld", ubSizePlatForm, commonBufferSize);
    return RunCommonTiling(context, commonBufferSize, commonBufferSize, 0);
}

IMPL_OP_OPTILING(GLUGrad).Tiling(Tiling4GluGrad).TilingParse<GluGradCompileInfo>(TilingPrepare4GluGrad);
} // namespace optiling
