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
 * \file matmul_compress_tiling.cpp
 * \brief
 */

#include "matmul_compress_tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"
#include "error_util.h"

using namespace optiling::matmul_compress;

namespace optiling {

static ge::graphStatus MatmulCompressTilingFunc(gert::TilingContext* context)
{
    OP_TILING_CHECK(
        context == nullptr, CUBE_INNER_ERR_REPORT("MatmulCompress", "context is null"), return ge::GRAPH_FAILED);
    auto ret = MatmulCompressTilingBase(context).DoTiling();
    OP_TILING_CHECK(
        ret == ge::GRAPH_FAILED, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "do tiling failed!"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForMatmulCompress(gert::TilingParseContext* context)
{
    OP_TILING_CHECK(
        context == nullptr, CUBE_INNER_ERR_REPORT("MatmulCompress", "context is null"), return ge::GRAPH_FAILED);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_TILING_CHECK(
        platformInfo == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "platformInfoPtr is null"),
        return ge::GRAPH_FAILED);

    auto hardwareInfoPtr = context->GetCompiledInfo<optiling::pp_matmul::HardwareInfo>();
    OP_TILING_CHECK(
        hardwareInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "hardwareInfoPtr is null"),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    hardwareInfoPtr->coreNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, hardwareInfoPtr->l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, hardwareInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, hardwareInfoPtr->l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, hardwareInfoPtr->l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, hardwareInfoPtr->l0cSize);

    OP_LOGI(
        context->GetNodeName(),
        "parse hardware info success coreNum:%lu, l2Size:%lu, l1Size:%lu, l0aSize:%lu, l0bSize:%lu, \
            l0cSize:%lu, hbmBandWidth:%lu, l2BandWidth:%lu",
        hardwareInfoPtr->coreNum, hardwareInfoPtr->l2Size, hardwareInfoPtr->l1Size, hardwareInfoPtr->l0aSize,
        hardwareInfoPtr->l0bSize, hardwareInfoPtr->l0cSize, hardwareInfoPtr->hbmBandWidth,
        hardwareInfoPtr->l2BandWidth);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MatmulCompress)
    .Tiling(MatmulCompressTilingFunc)
    .TilingParse<optiling::pp_matmul::HardwareInfo>(TilingPrepareForMatmulCompress);
} // namespace optiling