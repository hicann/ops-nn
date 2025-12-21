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
 * \file fused_matmul_builtin_tiling.cpp
 * \brief
 */

#include "fused_matmul_builtin_tiling.h"

#include "fused_matmul_common.h"
#include "fused_matmul_builtin_tiling_strategy.h"
#include "fused_matmul_tiling_key.h"
#include "matmul/batch_mat_mul_v3/op_host/op_tiling/arch35/batch_matmul_v3_tiling_strategy.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_compile_info_advanced.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_platform_common.h"
#include "register/op_def_registry.h"
#include "tiling_base/tiling_templates_registry.h"

namespace {
using namespace optiling;
using namespace optiling::fused_matmul;

static const std::vector<std::vector<ge::DataType>> DTYPE_SUPPORT_LIST_RESERVED = {
    // x1,              x2,             y,              bias
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
};

static const std::vector<std::vector<ge::DataType>> DTYPE_SUPPORT_LIST_91095 = {
    // x1,              x2,             y,               bias
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
    {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT},
    {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16},
    {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT},
    {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}};

inline void GetDtype(const gert::TilingContext& context, MatMulV3Args& args, platform_ascendc::SocVersion socVersion)
{
    args.aType = context.GetInputDesc(0)->GetDataType();
    args.bType = context.GetInputDesc(1)->GetDataType();
    args.cType = context.GetOutputDesc(0)->GetDataType();
    if (args.hasBias) {
        args.biasType = context.GetInputDesc(INPUT_BIAS_IDX)->GetDataType();
    }
    // op_impl_mode_enum: 0x1: default 0x2: high_performance 0x4: high_precision 0x8: super_performance
    // 0x10: support_of_bound_index 0x20: enable_float_32_execution 0x40: enable_hi_float_32_execution
    args.isHf32 = *((context.GetAttrs())->GetAttrPointer<bool>(ATTR_ENABLE_HF32_IDX));
    args.aDtypeSize = ge::GetSizeByDataType(args.aType);
    args.bDtypeSize = ge::GetSizeByDataType(args.bType);
    if (args.isHf32 && socVersion != platform_ascendc::SocVersion::ASCEND910_95) {
        OP_LOGW(args.opName, "Hf32 flag is: %d, which is not support yet", args.isHf32);
    }
}

ge::graphStatus IsValidDtype(const MatMulV3Args& args, platform_ascendc::SocVersion socVersion)
{
    std::vector<ge::DataType> dtype = {args.aType, args.bType, args.cType};
    if (args.hasBias) {
        dtype.push_back(args.biasType);
    }

    // check dtype
    auto supportList = socVersion == platform_ascendc::SocVersion::ASCEND910_95 ? DTYPE_SUPPORT_LIST_91095 :
                                                                                  DTYPE_SUPPORT_LIST_RESERVED;
    for (auto& supported : supportList) {
        if (std::equal(dtype.begin(), dtype.end(), supported.begin())) {
            return ge::GRAPH_SUCCESS;
        }
    }

    if (args.hasBias) {
        OP_LOGE(
            args.opName, "Unsupported data type: x1[%s], x2[%s], y[%s], bias[%s]", std::to_string(args.aType).c_str(),
            std::to_string(args.bType).c_str(), std::to_string(args.cType).c_str(),
            std::to_string(args.biasType).c_str());
        return ge::GRAPH_FAILED;
    } else {
        OP_LOGE(
            args.opName, "Unsupported data type: x1[%s], x2[%s], y[%s]", std::to_string(args.aType).c_str(),
            std::to_string(args.bType).c_str(), std::to_string(args.cType).c_str());
        return ge::GRAPH_FAILED;
    }
}

ge::graphStatus OpSpecificCheck(
    const gert::TilingContext& context, const MatMulV3Args& args, platform_ascendc::SocVersion socVersion)
{
    // check bias
    if (args.hasBias) {
        const gert::Shape& biasShape = context.GetInputShape(INPUT_BIAS_IDX)->GetOriginShape();
        const gert::Shape& cShape = context.GetOutputShape(0)->GetOriginShape();
        const int64_t biasValue = biasShape[biasShape.GetDimNum() - 1];
        const int64_t nOriValue = cShape[cShape.GetDimNum() - 1];
        if (biasValue != nOriValue) {
            OP_LOGE(args.opName, "illegal value: bias[%ld], n[%ld]", biasValue, nOriValue);
            return ge::GRAPH_FAILED;
        }
    }

    // dtype check
    return IsValidDtype(args, socVersion);
}
} // namespace

namespace optiling {
namespace fused_matmul {
ge::graphStatus FusedMatMulBuiltInTiling::DoTiling()
{
    if (GetShapeAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    MatMulV3BatchInfo tempBatchInfo;
    OP_TILING_CHECK(
        (GetBatchInfo(*context_, args_, tempBatchInfo) != ge::GRAPH_SUCCESS),
        CUBE_INNER_ERR_REPORT(args_.opName, "GetBatchInfo failed"), return ge::GRAPH_FAILED);
    args_.batchInfo = &tempBatchInfo;
    FusedMatmulTilingKey fusedMatmulTilingKey;
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    std::string opType = attrs->GetAttrPointer<char>(ATTR_OP_TYPE_IDX);
    fusedMatmulTilingKey.SetFusedOpType(FUSED_OP_TYPE_MAP.at(opType));
    MatMulTilingCfg tilingCfg(
        false, context_->GetCompileInfo(), reinterpret_cast<void*>(&args_), &fusedMatmulTilingKey);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, tilingCfg.compileInfo);

    MMRegisterCfg registerCfg{"FusedMatMul", socVersion_, strategy::GetFusedMatMulPriorities(socVersion_)};
    return MMTilingRegistry::GetInstance().DoTilingImpl(context_, tilingCfg, registerCfg);
}

ge::graphStatus FusedMatMulBuiltInTiling::GetArgs()
{
    GetFormat();
    GetDtype(*context_, args_, socVersion_);
    if (GetShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return OpSpecificCheck(*context_, args_, socVersion_);
}

ge::graphStatus FusedMatMulBuiltInTiling::CheckArgs()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(idx));
    idx++;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(idx));
    idx++;

    if (context_->GetInputDesc(idx) != nullptr) {
        args_.hasBias = true;
    }
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(ATTR_ENABLE_HF32_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(ATTR_OP_TYPE_IDX));

    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(0));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(0));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedMatMulBuiltInTiling::GetShapeAttrsInfo()
{
    OP_TILING_CHECK(
        GetSocVersion(context_, socVersion_) == ge::GRAPH_FAILED,
        CUBE_INNER_ERR_REPORT("FusedMatMul", "fail to get soc version"), return ge::GRAPH_FAILED);
    return MatMulV3Tiling::GetShapeAttrsInfo();
}
} // namespace fused_matmul
} // namespace optiling