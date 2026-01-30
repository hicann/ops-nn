/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file transpose_quant_batch_mat_mul_tiling_advanced.cc
 * \brief
 */
#include "transpose_quant_batch_mat_mul_tiling_advanced.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "transpose_quant_batch_mat_mul_tiling_strategy.h"
#include "transpose_quant_batch_mat_mul_common.h"
#include "register/op_def_registry.h"
#include "common/op_host/op_tiling/debug_tiling.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_compile_info_advanced.h"

namespace {
using namespace optiling;
using namespace optiling::transpose_quant_batch_mat_mul_advanced;
using namespace optiling::matmul_v3_advanced;
static inline void TQBMMGetDtype(const gert::TilingContext& context, MatMulV3Args& args)
{
    args.aType = context.GetInputDesc(0)->GetDataType();
    args.bType = context.GetInputDesc(1)->GetDataType();
    args.cType = context.GetOutputDesc(0)->GetDataType();
    args.aDtypeSize = ge::GetSizeByDataType(args.aType);
    args.bDtypeSize = ge::GetSizeByDataType(args.bType);
}

ge::graphStatus TQBMMGetShapeMKN(const gert::Shape& aShape, const gert::Shape& bShape,
                                 const gert::ContinuousVector* aPermList, const gert::ContinuousVector* bPermList,
                                 MatMulV3Args& args)
{
    const int64_t* aPerm = static_cast<const int64_t*>(aPermList->GetData());
    const int64_t* bPerm = static_cast<const int64_t*>(bPermList->GetData());
    int64_t m = aShape[aPerm[M_IDX]];
    int64_t kA = aShape[aPerm[KA_IDX]];
    int64_t kB = bShape[bPerm[KB_IDX]];
    int64_t n = bShape[bPerm[N_IDX]];
    args.isBTrans = bPerm[KB_IDX] > bPerm[N_IDX];
    args.isATrans = aPerm[M_IDX] > aPerm[KA_IDX];
    OP_TILING_CHECK(kA != kB, CUBE_INNER_ERR_REPORT(args.opName, "unequal input kDim values"), return ge::GRAPH_FAILED);

    if ((m <= 0) || (kA <= 0) || (n <= 0)) {
        OP_LOGE(args.opName, "illegal value: m[%ld], k[%ld], n[%ld]", m, kA, n);
        return ge::GRAPH_FAILED;
    }

    if (kA != TQBMM_VALID_K || n != TQBMM_VALID_N) {
        OP_LOGE(args.opName, "The shape of the x2 is not supported, now K are %ld and N are %ld", kA, n);
        return ge::GRAPH_FAILED;
    }
    args.mValue = static_cast<uint64_t>(m);
    args.kValue = static_cast<uint64_t>(kA);
    args.nValue = static_cast<uint64_t>(n);
    args.mOriValue = args.mValue;
    args.nOriValue = args.nValue;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQBMMGetShape(const gert::TilingContext& context, MatMulV3Args& args)
{
    const gert::Shape& aShape = context.GetInputShape(0)->GetOriginShape();
    const gert::Shape& bShape = context.GetInputShape(1)->GetOriginShape();
    auto attrs = context.GetAttrs();
    const gert::ContinuousVector* aPermList = attrs->GetAttrPointer<gert::ContinuousVector>(PERM_X1_IDX);
    const gert::ContinuousVector* bPermList = attrs->GetAttrPointer<gert::ContinuousVector>(PERM_X2_IDX);
    const gert::ContinuousVector* yPermList = attrs->GetAttrPointer<gert::ContinuousVector>(PERM_Y_IDX);
    // perm_x1
    const int64_t* aPerm = static_cast<const int64_t*>(aPermList->GetData());
    bool aPermCheck = ((aPerm[BATCH_IDX] == 1L) && (aPerm[M_IDX] == 0L) && (aPerm[KA_IDX] == 2L)); // aPerm is [1,0,2]
    // perm_x2
    OP_TILING_CHECK(!aPermCheck, CUBE_INNER_ERR_REPORT(args.opName, "unsupport aPerm value"), return ge::GRAPH_FAILED);
    const int64_t* bPerm = static_cast<const int64_t*>(bPermList->GetData());
    bool bPermCheck = ((bPerm[BATCH_IDX] == 0L) && (bPerm[KB_IDX] == 1L) && (bPerm[N_IDX] == 2L)); // bPerm is [0,1,2]
    OP_TILING_CHECK(!bPermCheck, CUBE_INNER_ERR_REPORT(args.opName, "unsupport bPerm value"), return ge::GRAPH_FAILED);
    // perm_y
    const int64_t* yPerm = static_cast<const int64_t*>(yPermList->GetData());
    bool yPermCheck = (yPerm[BATCH_IDX] == 1L) && (yPerm[M_IDX] == 0L) && (yPerm[N_IDX] == 2L); // yPerm is [1,0,2]
    OP_TILING_CHECK(!yPermCheck, CUBE_INNER_ERR_REPORT(args.opName, "unsupport yPerm value"), return ge::GRAPH_FAILED);

    OP_TILING_CHECK((TQBMMGetShapeMKN(aShape, bShape, aPermList, bPermList, args) != ge::GRAPH_SUCCESS),
                    CUBE_INNER_ERR_REPORT(args.opName, "get m/k/n failed"), return ge::GRAPH_FAILED);

    if (attrs->GetAttrNum() >= ATTR_NUM) {
        int32_t batchSplitFactor = *(attrs->GetAttrPointer<int32_t>(ATTR_NUM - 1));
        bool batchSplitFactorPermCheck = (batchSplitFactor == 1);
        OP_TILING_CHECK(!batchSplitFactorPermCheck,
                        CUBE_INNER_ERR_REPORT(args.opName, "batch_split_factor is not supported"),
                        return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IsValidDtype(const gert::TilingContext& context, const MatMulV3Args& args)
{
    auto scaleX1Desc = context.GetOptionalInputDesc(SCALE_X1_IDX);
    auto scaleX2Desc = context.GetOptionalInputDesc(SCALE_X2_IDX);
    if (scaleX1Desc == nullptr || scaleX2Desc == nullptr) {
        OP_LOGE(args.opName, "scale cannot be nullptr");
        return ge::GRAPH_FAILED;
    }
    auto scaleX1Dtype = scaleX1Desc->GetDataType();
    auto scaleX2Dtype = scaleX2Desc->GetDataType();
    std::vector<ge::DataType> dtype = {args.aType, args.bType, scaleX1Dtype, scaleX2Dtype, args.cType};
    const std::vector<std::vector<ge::DataType>> dtypeSuportList = {
        // x1,              x2,                   scale_x1,      scale_x2        y,
        {ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16},
        {ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16},
        {ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16},
        {ge::DT_FLOAT8_E5M2, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16},
        {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16},
        {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16},
        {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT16},
        {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16}};
    for (auto& supported : dtypeSuportList) {
        if (std::equal(dtype.begin(), dtype.end(), supported.begin())) {
            return ge::GRAPH_SUCCESS;
        }
    }
    OP_LOGE(args.opName, "Unsupported data type: x1[%s], x2[%s], scale_x1[%s], scale_x2[%s], y[%s]",
            Ops::Base::ToString(args.aType).c_str(), Ops::Base::ToString(args.bType).c_str(),
            Ops::Base::ToString(scaleX1Dtype).c_str(), Ops::Base::ToString(scaleX2Dtype).c_str(),
            Ops::Base::ToString(args.cType).c_str());
    return ge::GRAPH_FAILED;
}
} // namespace

namespace optiling {
namespace transpose_quant_batch_mat_mul_advanced {
ge::graphStatus TransposeQuantBatchMatMulTiling::GetArgs()
{
    TQBMMGetDtype(*context_, args_);
    if (TQBMMGetShape(*context_, args_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return IsValidDtype(*context_, args_);
}
ge::graphStatus TransposeQuantBatchMatMulTiling::CheckScale(const int64_t m, const int64_t n) const
{
    // scale
    auto scaleX1ShapePtr = context_->GetOptionalInputShape(SCALE_X1_IDX);
    auto scaleX2ShapePtr = context_->GetOptionalInputShape(SCALE_X2_IDX);
    if (scaleX1ShapePtr == nullptr || scaleX2ShapePtr == nullptr) {
        OP_LOGE(args_.opName, "scale cannot be nullptr");
        return ge::GRAPH_FAILED;
    }
    auto scaleX1DimNum = scaleX1ShapePtr->GetStorageShape().GetDimNum();
    int64_t scaleX1Dim0 = scaleX1ShapePtr->GetStorageShape().GetDim(0);
    if (scaleX1DimNum != EXPECTED_SCALE_DIM || scaleX1Dim0 != m) {
        OP_LOGE(args_.opName, "Dim of x1Scale != 1 or x1Scale dim 0 != M");
        return ge::GRAPH_FAILED;
    }
    auto scaleX2DimNum = scaleX2ShapePtr->GetStorageShape().GetDimNum();
    int64_t scaleX2Dim0 = scaleX2ShapePtr->GetStorageShape().GetDim(0);
    if (scaleX2DimNum != EXPECTED_SCALE_DIM || scaleX2Dim0 != n) {
        OP_LOGE(args_.opName, "Dim of x2Scale != 1 or x2Scale dim 0 != N");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeQuantBatchMatMulTiling::CheckArgs()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(X1_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(X1_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(X2_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(X2_IDX));
    const gert::ContinuousVector* aPermList = attrs->GetAttrPointer<gert::ContinuousVector>(PERM_X1_IDX);
    const gert::ContinuousVector* bPermList = attrs->GetAttrPointer<gert::ContinuousVector>(PERM_X2_IDX);
    const gert::ContinuousVector* yPermList = attrs->GetAttrPointer<gert::ContinuousVector>(PERM_Y_IDX);
    if (aPermList == nullptr || bPermList == nullptr || yPermList == nullptr) {
        OP_LOGE(args_.opName, "PermX1 and permX2 and permY should not be nullptr");
        return ge::GRAPH_FAILED;
    }
    if (aPermList->GetSize() != ALLOW_DIM || bPermList->GetSize() != ALLOW_DIM || yPermList->GetSize() != ALLOW_DIM) {
        OP_LOGE(args_.opName, "The dims of the perm intArray should be 3");
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& aShape = context_->GetInputShape(X1_IDX)->GetOriginShape();
    const gert::Shape& bShape = context_->GetInputShape(X2_IDX)->GetOriginShape();
    const gert::Shape& cShape = context_->GetOutputShape(0)->GetOriginShape();
    const size_t aDimNum = aShape.GetDimNum();
    const size_t bDimNum = bShape.GetDimNum();
    const size_t cDimNum = cShape.GetDimNum();
    if ((aDimNum != ALLOW_DIM) || (bDimNum != ALLOW_DIM) || (cDimNum != ALLOW_DIM)) {
        OP_LOGE(args_.opName, "invalid input/output dim num");
        return ge::GRAPH_FAILED;
    }
    const int64_t* aPerm = static_cast<const int64_t*>(aPermList->GetData());
    const int64_t* bPerm = static_cast<const int64_t*>(bPermList->GetData());
    int64_t m = aShape[aPerm[M_IDX]];
    int64_t n = bShape[bPerm[N_IDX]];
    //scale
    OP_TILING_CHECK((CheckScale(m, n) != ge::GRAPH_SUCCESS), CUBE_INNER_ERR_REPORT(args_.opName, "invalid scale"),
                    return ge::GRAPH_FAILED);
    // bias
    if (context_->GetOptionalInputShape(BIAS_IDX) != nullptr) {
        OP_LOGE(args_.opName, "bias is not supported");
        return ge::GRAPH_FAILED;
    }
    if (attrs->GetAttrNum() >= ATTR_NUM) {
        OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<int32_t>(ATTR_NUM - 1));
    }
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(0));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeQuantBatchMatMulTiling::GetShapeAttrsInfo()
{
    args_.opName = context_->GetNodeName();
    OP_TILING_CHECK(args_.opName == nullptr, CUBE_INNER_ERR_REPORT("TransposeQuantBatchMatMul", "get op name invalid"),
                    return ge::GRAPH_FAILED);
    OP_LOGI(args_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    OP_TILING_CHECK((CheckArgs() != ge::GRAPH_SUCCESS) || (GetArgs() != ge::GRAPH_SUCCESS),
                    CUBE_INNER_ERR_REPORT(args_.opName, "invalid context"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeQuantBatchMatMulTiling::DoTiling()
{
    if (GetShapeAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    MatMulV3BatchInfo tempBatchInfo;
    OP_TILING_CHECK((GetBatchInfo(*context_, args_, tempBatchInfo) != ge::GRAPH_SUCCESS),
                    CUBE_INNER_ERR_REPORT(args_.opName, "GetBatchInfo failed"), return ge::GRAPH_FAILED);
    args_.batchInfo = &tempBatchInfo;
    MatMulTilingCfg tilingCfg(false, context_->GetCompileInfo(), static_cast<void*>(&args_));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, tilingCfg.compileInfo);
    platform_ascendc::SocVersion socVersion =
        static_cast<const MatmulV3CompileInfo*>(tilingCfg.compileInfo)->socVersion;
    MMRegisterCfg registerCfg{"TransposeQuantBatchMatMul", socVersion,
                              strategy::GetTransposeQuantBatchMatMulPriorities(socVersion)};
    return MMTilingRegistry::GetInstance().DoTilingImpl(context_, tilingCfg, registerCfg);
}

ge::graphStatus TransposeQuantBatchMatMulTiling::GetBatchInfo(const gert::TilingContext& context, MatMulV3Args& args,
                                                              MatMulV3BatchInfo& batchInfo) const
{
    const gert::Shape& aShape = context.GetInputShape(0)->GetOriginShape();
    const gert::Shape& bShape = context.GetInputShape(1)->GetOriginShape();

    auto attrs = context.GetAttrs();
    const gert::ContinuousVector* aPermList = attrs->GetAttrPointer<gert::ContinuousVector>(PERM_X1_IDX);
    const gert::ContinuousVector* bPermList = attrs->GetAttrPointer<gert::ContinuousVector>(PERM_X2_IDX);

    uint64_t batchA = static_cast<uint64_t>(aShape[BATCH_IDX]);
    uint64_t batchB = static_cast<uint64_t>(bShape[BATCH_IDX]);
    const int64_t* aPerm = static_cast<const int64_t*>(aPermList->GetData());
    batchA = aShape[aPerm[BATCH_IDX]];
    const int64_t* bPerm = static_cast<const int64_t*>(bPermList->GetData());
    batchB = bShape[bPerm[BATCH_IDX]];
    OP_TILING_CHECK(batchA != batchB, CUBE_INNER_ERR_REPORT(args.opName, "unequal input batch values"),
                    return ge::GRAPH_FAILED);
    batchInfo.batchA3 = batchA;
    batchInfo.batchB3 = batchA;
    batchInfo.batchC3 = batchA;
    batchInfo.batchA = batchA;
    batchInfo.batchB = batchA;
    batchInfo.batchC = batchA;
    return ge::GRAPH_SUCCESS;
}

} // namespace transpose_quant_batch_mat_mul_advanced
} // namespace optiling