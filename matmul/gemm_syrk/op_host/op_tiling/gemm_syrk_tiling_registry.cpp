/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file gemm_syrk_tiling_registry.cpp
 * \brief GemmSyrk tiling registry: entry func (Ascend 950 dispatch), tiling
 * parse (fills GemmSyrkCompileInfo from the platform) and the simplified key.
 */

#include "arch35/gemm_syrk_compile_info.h"
#include "arch35/gemm_syrk_tiling.h"

#include "register/op_impl_registry.h"
#include "error_util.h"

using optiling::gemm_syrk::GemmSyrkTiling;

namespace optiling {

static ge::graphStatus GemmSyrkTilingFunc(gert::TilingContext* context)
{
    OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("GemmSyrk", "context is null"), return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, CUBE_INNER_ERR_REPORT("GemmSyrk", "platformInfo is null"),
                    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto socVersion = ascendcPlatform.GetSocVersion();
    if (socVersion != platform_ascendc::SocVersion::ASCEND950) {
        CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                              "GemmSyrk only supports Ascend 950 / 350 (DAV_3510), current socVersion is %d",
                              static_cast<int32_t>(socVersion));
        return ge::GRAPH_FAILED;
    }
    return GemmSyrkTiling(context).DoTiling();
}

static ge::graphStatus TilingPrepareForGemmSyrk(gert::TilingParseContext* context)
{
    OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("GemmSyrk", "context is null"), return ge::GRAPH_FAILED);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, CUBE_INNER_ERR_REPORT("GemmSyrk", "platformInfo is null"),
                    return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompiledInfo<GemmSyrkCompileInfo>();
    OP_TILING_CHECK(compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT("GemmSyrk", "compileInfoPtr is null"),
                    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0aSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0bSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GemmSyrk).Tiling(GemmSyrkTilingFunc).TilingParse<GemmSyrkCompileInfo>(TilingPrepareForGemmSyrk);
// No GenSimplifiedKey registration: the runtime falls back to the default
// simplified-key derivation (input/output dtype+format matrix), which is what
// the def-config-generated binary json matches against.
} // namespace optiling
