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
 * \file transpose_batch_mat_mul_tiling.cc
 * \brief
 */
#include "transpose_batch_mat_mul_tiling.h"

#include <type_traits>

#include "matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_platform_common.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_simplifiedkey.h"
#include "./arch35/transpose_batch_mat_mul_tiling_advanced.h"
#include "transpose_batch_mat_mul_base_tiling.h"
#include "transpose_batch_mat_mul_einsum_tiling.h"
#include "platform/platform_infos_def.h"
#include "op_cache_tiling.h"

#include "op_host/tiling_templates_registry.h"
#include "register/op_def_registry.h"

using namespace optiling::transpose_batch_mat_mul;
using namespace optiling::matmul_v3;
using Ops::NN::Optiling::TilingRegistry;
using Ops::NN::TilingPrepareForOpCache;

namespace optiling {

REGISTER_TILING_TEMPLATE("TransposeBatchMatMul", TransposeBatchMatMulBaseTiling, 0);

static ge::graphStatus IsEinsumMode(gert::TilingContext* context)
{
    constexpr size_t ALLOW_DIM = 3;
    constexpr size_t ATTR_NUM = 5;
    constexpr bool EINSUM_SUPPORT_ENABLE_HF32 = false;
    constexpr int32_t EINSUM_SUPPORT_BATCHSPLIT_FACTOR = 1;
    size_t idx = 0;
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(idx));
    idx++;
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(idx));
    idx++;
    OP_TILING_CHECK((context->GetOptionalInputShape(idx)!= nullptr ||
        context->GetOptionalInputShape(idx + 1)!= nullptr),
        OP_LOGI(context->GetNodeName(), "Einsum mode not support bias or scale"),
        return ge::GRAPH_FAILED);
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetOutputDesc(0));
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    idx = static_cast<size_t>(0);
    auto aPermList_ = attrs->GetAttrPointer<gert::ContinuousVector>(idx);
    const int64_t* perm_x1 = reinterpret_cast<const int64_t*>(aPermList_->GetData());
    OP_TILING_CHECK((PermDecode(perm_x1, aPermList_->GetSize()) != 213L),
        OP_LOGI(context->GetNodeName(), "Einsum mode only support permA={1,0,2}"),
        return ge::GRAPH_FAILED);
    if (attrs->GetAttrNum() >= ATTR_NUM) {
        idx++;
        OP_TILING_CHECK(
            (*(attrs->GetAttrPointer<int32_t>(ATTR_NUM - idx)) != EINSUM_SUPPORT_BATCHSPLIT_FACTOR),
            OP_LOGI(context->GetNodeName(), "Einsum mode only support batch_split_factor=1"),
            return ge::GRAPH_FAILED);
        idx++;
        OP_TILING_CHECK(
            (*(attrs->GetAttrPointer<bool>(ATTR_NUM - idx)) != EINSUM_SUPPORT_ENABLE_HF32),
            OP_LOGI(context->GetNodeName(), "Einsum mode only support ENABLE_HF32=false"),
            return ge::GRAPH_FAILED);
    }
    const gert::Shape &aShape = context->GetInputShape(0)->GetOriginShape();
    const gert::Shape &bShape = context->GetInputShape(1)->GetOriginShape();
    const size_t aDimNum = aShape.GetDimNum();
    const size_t bDimNum = bShape.GetDimNum();
    if ((aDimNum == ALLOW_DIM) && (bDimNum == ALLOW_DIM)) {
        if ((context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT) &&
            (context->GetInputDesc(1)->GetDataType() == ge::DT_FLOAT) &&
            (context->GetOutputDesc(0)->GetDataType() == ge::DT_FLOAT)) {
            return ge::GRAPH_SUCCESS;
        }
    }
    return ge::GRAPH_FAILED;
}

static ge::graphStatus TbmmEinsumTilingFunc(gert::TilingContext* context)
{
    OP_TILING_CHECK(context == nullptr,
        CUBE_INNER_ERR_REPORT("TbmmEinsum", "context is null"), return ge::GRAPH_FAILED);
    size_t sysWorkspaceSize = static_cast<size_t>(24 * 1024 * 1024);  // 24M same as ppmatmul tiling
    size_t* currentWorkSpace = context->GetWorkspaceSizes(1);
    currentWorkSpace[0] = sysWorkspaceSize;
    OP_LOGI(context->GetNodeName(), "TbmmEinsum Tiling start.");
    TransposeBatchMatMulEinsumTiling tbmmEinsumTiling(context);
    tbmmEinsumTiling.DoTiling();
    auto res = tbmmEinsumTiling.PostTiling();
    OP_LOGI(context->GetNodeName(), "TbmmEinsum Tiling end.");
    return res;
}

static ge::graphStatus TransposeBatchMatMulTilingFunc(gert::TilingContext* context) {
    OP_TILING_CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("TransposeBatchMatMul", "context is null"),
                    return ge::GRAPH_FAILED);
    if (IsAdvancedSocVersion(context)) {
        return transpose_batch_mat_mul_advanced::TransposeBatchMatMulTiling(context).DoTiling();
    }
    if (IsEinsumMode(context) == ge::GRAPH_SUCCESS) {
        return TbmmEinsumTilingFunc(context);
    }
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForTransposeBatchMatMul(gert::TilingParseContext *context) {
    OP_TILING_CHECK(context == nullptr,
                    CUBE_INNER_ERR_REPORT("TransposeBatchMatMul", "context is null"),
                    return ge::GRAPH_FAILED);
    fe::PlatFormInfos* platformInfo = context->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr,
                CUBE_INNER_ERR_REPORT(context->GetNodeName(), "platformInfoPtr is null"),
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
    compileInfoPtr->socVersion = ascendcPlatform.GetSocVersion();
    compileInfoPtr->npuArch = ascendcPlatform.GetCurNpuArch();
    compileInfoPtr->btSize = compileInfoPtr->supportL0c2out ? 1024UL : 0UL;                       // 1024 is btSize
    compileInfoPtr->btSize = compileInfoPtr->supportL12BtBf16 ? 4096UL : compileInfoPtr->btSize;  // 4096 is btSize
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfoPtr->ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfoPtr->l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, compileInfoPtr->l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, compileInfoPtr->l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfoPtr->l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfoPtr->l2Size);

    if(!TilingPrepareForOpCache(context)){
        return ge::GRAPH_SUCCESS;
    }
    OP_LOGI(
        context->GetNodeName(),
        "compile info success soc:%d, npu arch:%u, l1Size:%lu, l2Size:%lu, coreNum:%lu, supportL0c2out:%d, supportL12BtBf16:%d",
        static_cast<int>(compileInfoPtr->socVersion), compileInfoPtr->npuArch, compileInfoPtr->l1Size, compileInfoPtr->l2Size,
        compileInfoPtr->aicNum, compileInfoPtr->supportL0c2out, compileInfoPtr->supportL12BtBf16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TransposeBatchMatMul)
    .Tiling(TransposeBatchMatMulTilingFunc)
    .TilingParse<MatmulV3CompileInfo>(TilingPrepareForTransposeBatchMatMul)
    .GenSimplifiedKey(GenSimplifiedKey);
}