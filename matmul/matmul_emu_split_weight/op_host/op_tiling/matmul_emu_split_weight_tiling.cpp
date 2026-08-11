/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "arch35/matmul_emu_split_weight_tiling.h"
#include "arch35/matmul_emu_split_weight_compile_info.h"

#include "register/op_impl_registry.h"
#include "error_util.h"
#include "op_host/tiling_templates_registry.h"

using optiling::matmul_emu_split_weight::MatmulEmuSplitWeightTiling;

namespace {
static const int32_t MATMUL_EMU_SPLIT_WEIGHT_TILING_PRIORITY = 0;
} // namespace

namespace optiling {

REGISTER_TILING_TEMPLATE("MatmulEmuSplitWeight", MatmulEmuSplitWeightTiling, MATMUL_EMU_SPLIT_WEIGHT_TILING_PRIORITY);

static ge::graphStatus MatmulEmuSplitWeightTilingFunc(gert::TilingContext* context)
{
    OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("MatmulEmuSplitWeight", "context is null"),
                    return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, CUBE_INNER_ERR_REPORT("MatmulEmuSplitWeight", "platformInfo is null"),
                    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto socVersion = ascendcPlatform.GetSocVersion();
    OP_TILING_CHECK(socVersion != platform_ascendc::SocVersion::ASCEND950,
                    CUBE_INNER_ERR_REPORT("MatmulEmuSplitWeight",
                                          "MatmulEmuSplitWeight only supports Ascend950, current socVersion is %d",
                                          static_cast<int32_t>(socVersion)),
                    return ge::GRAPH_FAILED);
    return MatmulEmuSplitWeightTiling(context).DoTiling();
}

static ge::graphStatus TilingPrepareForMatmulEmuSplitWeight(gert::TilingParseContext* context)
{
    OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("MatmulEmuSplitWeight", "context is null"),
                    return ge::GRAPH_FAILED);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, CUBE_INNER_ERR_REPORT("MatmulEmuSplitWeight", "platformInfo is null"),
                    return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompiledInfo<MatmulEmuSplitWeightCompileInfo>();
    OP_TILING_CHECK(compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT("MatmulEmuSplitWeight", "compileInfoPtr is null"),
                    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MatmulEmuSplitWeight)
    .Tiling(MatmulEmuSplitWeightTilingFunc)
    .TilingParse<MatmulEmuSplitWeightCompileInfo>(TilingPrepareForMatmulEmuSplitWeight);
} // namespace optiling
