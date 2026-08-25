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
 * \file fused_matmul_tiling.cpp
 * \brief FusedMatMul tiling entry: routes to built-in tiling (BatchMatMulV3-based).
 */
#include "fused_matmul_tiling.h"

#include "fused_matmul_builtin_tiling.h"
#include "fused_matmul_common.h"
#include "fused_matmul_simplifiedkey.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_compile_info_advanced.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_platform_common.h"
#include "register/op_impl_registry.h"
#include "matmul/common/op_host/log_format_util.h"

namespace {
using namespace optiling;

// NpuArch -> supported opTypes
const std::unordered_map<NpuArch, std::vector<std::string>> NpuArchFusedOpSupport = {
    {NpuArch::DAV_3510, {"", "relu", "add", "mul", "16cast32", "gelu_erf", "gelu_tanh", "scale_add"}},
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

ge::graphStatus FusedMatMulTilingFunc(gert::TilingContext* context)
{
    OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("FusedMatMul", "context is null"),
                    return ge::GRAPH_FAILED);
    if (!IsAdvancedSocVersion(context)) {
        OP_LOGE("FusedMatMul", "not support npu arch");
        return ge::GRAPH_FAILED;
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
                    CUBE_INNER_ERR_REPORT(context->GetNodeName(), "unsupported fused op type: %s, supported: %s",
                                          fusedOpType.c_str(), JoinOpTypes(supportedOps).c_str()),
                    return ge::GRAPH_FAILED);

    return fused_matmul::FusedMatMulBuiltInTiling(context).DoTiling();
}

} // namespace

namespace optiling {
IMPL_OP_OPTILING(FusedMatMul)
    .Tiling(FusedMatMulTilingFunc)
    .TilingParse<MatmulV3CompileInfo>(matmul_v3_advanced::InitCompileInfo)
    .GenSimplifiedKey(fused_matmul::GenSimplifiedKey);
} // namespace optiling
