/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file fused_mat_mul_tiling.cpp
 * \brief FusedMatMul tiling entry: routes to built-in tiling (BatchMatMulV3-based).
 */
#include "fused_mat_mul_tiling.h"

#include "fused_mat_mul_builtin_tiling.h"
#include "fused_mat_mul_common.h"
#include "fused_mat_mul_simplifiedkey.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_compile_info_advanced.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_compile_info.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_platform_common.h"
#include "op_cache_tiling.h"
#include "op_host/tiling_templates_registry.h"
#include "register/op_impl_registry.h"
#include "matmul/common/op_host/log_format_util.h"
#include "op_common/op_host/util/op_const_def.h"

namespace {
using namespace optiling;
using Ops::NN::TilingPrepareForOpCache;
using Ops::NN::Optiling::TilingRegistry;

// NpuArch -> supported opTypes
const std::unordered_map<NpuArch, std::vector<std::string>> NpuArchFusedOpSupport = {
    {Ops::Base::DAV_3510, {"", "relu", "add", "mul", "16cast32", "gelu_erf", "gelu_tanh", "sigmoid", "scale_add"}},
    {NpuArch::DAV_9201, {"", "relu", "16cast32"}},
    {NpuArch::DAV_9202, {"", "relu", "16cast32"}},
    {NpuArch::DAV_RESV, {"relu", "quant", "relu_quant"}},
};

std::string JoinOpTypes(const std::vector<std::string>& ops)
{
    std::ostringstream oss;
    oss << "[";
    bool first = true;
    for (const auto& str : ops) {
        if (!first) {
            oss << ",";
        }
        if (str.empty()) {
            oss << "\"\"";
        } else {
            oss << str;
        }
        first = false;
    }
    oss << "]";
    return oss.str();
}

ge::graphStatus TilingPrepareForFusedMatMulA2(gert::TilingParseContext* context)
{
    OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("FusedMatMul", "context is null"),
                    return ge::GRAPH_FAILED);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "platformInfoPtr is null"),
                    return ge::GRAPH_FAILED);
    auto compileInfoPtr = context->GetCompiledInfo<MatmulV3CompileInfo>();
    OP_TILING_CHECK(compileInfoPtr == nullptr, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "compileInfoPtr is null"),
                    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    platformInfo->GetPlatformRes("version", "SoC_version", compileInfoPtr->socVersionStr);
    std::string val;
    std::string dataMoveL12Bt;
    platformInfo->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_fix_pipe_l0c2out", val);
    platformInfo->GetPlatformRes("AICoreintrinsicDtypeMap", "Intrinsic_data_move_l12bt", dataMoveL12Bt);
    compileInfoPtr->supportL0c2out = !val.empty();
    compileInfoPtr->supportL12BtBf16 = (dataMoveL12Bt.find("bf16") != std::string::npos);
    compileInfoPtr->aicNum = ascendcPlatform.GetCoreNumAic();
    compileInfoPtr->aivNum = ascendcPlatform.GetCoreNumAiv();
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->npuArch = ascendcPlatform.GetCurNpuArch();
    compileInfoPtr->btSize = compileInfoPtr->supportL0c2out ? 1024UL : 0UL;
    compileInfoPtr->btSize = compileInfoPtr->supportL12BtBf16 ? 4096UL : compileInfoPtr->btSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);
    TilingPrepareForOpCache(context);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedMatMulTilingFunc(gert::TilingContext* context)
{
    OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("FusedMatMul", "context is null"),
                    return ge::GRAPH_FAILED);
    if (!IsAdvancedSocVersion(context)) {
        NpuArch npuArch = NpuArch::DAV_RESV;
        OP_TILING_CHECK(GetSocVersion(context, npuArch) == ge::GRAPH_FAILED,
                        CUBE_INNER_ERR_REPORT("FusedMatMul", "fail to get npu arch"), return ge::GRAPH_FAILED);
        if (npuArch != NpuArch::DAV_2201) {
            OP_LOGE("FusedMatMul", "does not support npu arch");
            return ge::GRAPH_FAILED;
        }
        auto attrs = context->GetAttrs();
        OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
        OPS_CHECK_NULL_WITH_CONTEXT(context, attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX));
        std::string fusedOpType = attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);
        OP_TILING_CHECK(fusedOpType != "relu",
                        CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                                              "Atlas A2/A3 FusedMatMul only supports fusedOpType relu, current: %s",
                                              fusedOpType.c_str()),
                        return ge::GRAPH_FAILED);
        return TilingRegistry::GetInstance().DoTilingImpl(context);
    }

    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX));
    std::string fusedOpType = attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);

    NpuArch npuArch;
    OP_TILING_CHECK(GetSocVersion(context, npuArch) == ge::GRAPH_FAILED,
                    CUBE_INNER_ERR_REPORT(context->GetNodeName(), "fail to get npu arch"), return ge::GRAPH_FAILED);
    auto it = NpuArchFusedOpSupport.find(npuArch);
    OP_TILING_CHECK(it == NpuArchFusedOpSupport.end(),
                    CUBE_INNER_ERR_REPORT(context->GetNodeName(), "unsupported platform(impossible situation)"),
                    return ge::GRAPH_FAILED);

    const auto& supportedOps = it->second;
    bool useBuiltInTiling = std::find(supportedOps.begin(), supportedOps.end(), fusedOpType) != supportedOps.end();
    OP_TILING_CHECK(!useBuiltInTiling,
                    CUBE_INNER_ERR_REPORT(context->GetNodeName(), "unsupported fusedOpType: %s, supported: %s",
                                          fusedOpType.c_str(), JoinOpTypes(supportedOps).c_str()),
                    return ge::GRAPH_FAILED);

    return fused_matmul::FusedMatMulBuiltInTiling(context).DoTiling();
}

} // namespace

namespace optiling {
static ge::graphStatus TilingPrepareForFusedMatMul(gert::TilingParseContext* context)
{
    if (context != nullptr && !IsAdvancedSocVersion(context)) {
        return TilingPrepareForFusedMatMulA2(context);
    }
    return matmul_v3_advanced::InitCompileInfo(context);
}

IMPL_OP_OPTILING(FusedMatMul)
    .Tiling(FusedMatMulTilingFunc)
    .TilingParse<MatmulV3CompileInfo>(TilingPrepareForFusedMatMul)
    .GenSimplifiedKey(fused_matmul::GenSimplifiedKey);
} // namespace optiling
