/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include "matmul/common/op_host/math_util.h"
#include "matmul/common/op_host/log_format_util.h"

namespace {
using namespace optiling;
using namespace optiling::transpose_quant_batch_mat_mul_advanced;
using namespace optiling::matmul_v3_advanced;
static inline void TQBMMGetDtype(const gert::TilingContext& context, MatMulV3Args& args)
{
    args.aType = context.GetInputDesc(X1_IDX)->GetDataType();
    args.bType = context.GetInputDesc(X2_IDX)->GetDataType();
    args.cType = context.GetOutputDesc(0)->GetDataType();
    args.aDtypeSize = ge::GetSizeByDataType(args.aType);
    args.bDtypeSize = ge::GetSizeByDataType(args.bType);
}

static inline void TQBMMGetFormat(const gert::TilingContext& context, MatMulV3Args& args)
{
    ge::Format formatA =
        static_cast<ge::Format>(ge::GetPrimaryFormat(context.GetInputDesc(X1_IDX)->GetStorageFormat()));
    ge::Format formatB =
        static_cast<ge::Format>(ge::GetPrimaryFormat(context.GetInputDesc(X2_IDX)->GetStorageFormat()));
    ge::Format formatOut = static_cast<ge::Format>(ge::GetPrimaryFormat(context.GetOutputDesc(0)->GetStorageFormat()));
    args.aFormat = (formatA != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatA;
    args.bFormat = (formatB != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatB;
    args.outFormat = (formatOut != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatOut;
}

static inline ge::graphStatus CheckWeightNz(const gert::TilingContext& context, MatMulV3Args& args, bool isMxFp)
{
    if (!isMxFp) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
            args.opName, "x2", "FRACTAL_NZ",
            Ops::NN::FormatString("In %s case, the format of %s cannot be %s",
                                  "non-mxfp8 mode", "x2", "FRACTAL_NZ").c_str());
        return ge::GRAPH_FAILED;
    }
    auto bShape = context.GetInputShape(X2_IDX)->GetOriginShape();
    size_t bDims = bShape.GetDimNum();
    OP_TILING_CHECK(
        bDims == EXPECTED_NZ_DIM,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            args.opName, "x2", Ops::NN::FormatString("%zu", bDims).c_str(),
            Ops::NN::FormatString("The storage shape dim of %s must be %d", "x2",
                                  static_cast<int>(EXPECTED_NZ_DIM)).c_str()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus CheckNotNull(const gert::TilingContext* context)
{
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetAttrs());
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(X1_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(X1_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(X2_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(X2_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetOutputDesc(0));
    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus IsValidGroupSize(const int64_t groupSize, bool isMxFp, MatMulV3Args& args)
{
    uint64_t groupSizeK = static_cast<uint64_t>(groupSize) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeN = (static_cast<uint64_t>(groupSize) >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeM = (static_cast<uint64_t>(groupSize) >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;
    if (isMxFp && !((groupSizeM == 0 || groupSizeM == 1) && (groupSizeN == 0 || groupSizeN == 1) &&
                    groupSizeK == SUPPORTED_GROUP_SIZE)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            args.opName, "groupSizeM, groupSizeN, groupSizeK",
            Ops::NN::FormatString("%lu, %lu, %lu", groupSizeM, groupSizeN, groupSizeK).c_str(),
            Ops::NN::FormatString(
                "The values of %s must be %s, and the value of %s must be %d",
                "groupSizeM and groupSizeN", "0 or 1", "groupSizeK", static_cast<int>(SUPPORTED_GROUP_SIZE))
                .c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQBMMGetShapeMKN(
    const gert::Shape& aShape, const gert::Shape& bShape, const gert::ContinuousVector* aPermList,
    const gert::ContinuousVector* bPermList, MatMulV3Args& args, bool isMxfp8, bool isHIFP8)
{
    const int64_t* aPerm = static_cast<const int64_t*>(aPermList->GetData());
    const int64_t* bPerm = static_cast<const int64_t*>(bPermList->GetData());
    int64_t m = aShape[aPerm[M_IDX]];
    int64_t kA = aShape[aPerm[KA_IDX]];
    int64_t kB = bShape[bPerm[KB_IDX]];
    int64_t n = bShape[bPerm[N_IDX]];
    args.isBTrans = bPerm[KB_IDX] > bPerm[N_IDX];
    args.isATrans = aPerm[M_IDX] > aPerm[KA_IDX];
    if (kA != kB) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            args.opName, "x1, x2",
            Ops::NN::FormatString("%s, %s", Ops::Base::ToString(aShape).c_str(), Ops::Base::ToString(bShape).c_str())
                .c_str(),
            Ops::NN::FormatString("%s of %s must be equal", "K-axis", "x1, x2").c_str());
        return ge::GRAPH_FAILED;
    }

    if ((m <= 0) || (kA <= 0) || (n <= 0)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            args.opName, "x1, x2",
            Ops::NN::FormatString("%s, %s", Ops::Base::ToString(aShape).c_str(), Ops::Base::ToString(bShape).c_str())
                .c_str(),
            Ops::NN::FormatString("%s of %s must be positive numbers", "m, k, n", "x1, x2").c_str());
        return ge::GRAPH_FAILED;
    }
    if (isMxfp8 && (kA % K_ALIGNMENT64 != 0)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            args.opName, "x1", Ops::Base::ToString(aShape).c_str(),
            Ops::NN::FormatString("%s of %s must be exactly divisible by %lu", "K-axis", "x1", K_ALIGNMENT64).c_str());
        return ge::GRAPH_FAILED;
    }
    if (!isMxfp8 && !isHIFP8 && (kA != TQBMM_VALID_K || n != TQBMM_VALID_N)) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            args.opName, "x2", Ops::Base::ToString(bShape).c_str(),
            Ops::NN::FormatString("The k-axis and n-axis of %s must be %d and %d", "x2", TQBMM_VALID_K, TQBMM_VALID_N)
                .c_str());
        return ge::GRAPH_FAILED;
    }
    args.mValue = static_cast<uint64_t>(m);
    args.kValue = static_cast<uint64_t>(kA);
    args.nValue = static_cast<uint64_t>(n);
    args.mOriValue = args.mValue;
    args.nOriValue = args.nValue;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQBMMCheckPerm(const gert::ContinuousVector* aPermList, const gert::ContinuousVector* bPermList,
                               const gert::ContinuousVector* yPermList, MatMulV3Args& args, bool isMxfp8, bool isHIFP8)
{
    // perm_x1
    const int64_t* aPerm = static_cast<const int64_t*>(aPermList->GetData());
    bool aPermCheck = ((aPerm[BATCH_IDX] == 1L) && (aPerm[M_IDX] == 0L) && (aPerm[KA_IDX] == 2L)); // aPerm is [1,0,2]
    if (!aPermCheck) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            args.opName, "permX1",
            Ops::NN::FormatString("[%ld, %ld, %ld]", aPerm[BATCH_IDX], aPerm[M_IDX], aPerm[KA_IDX]).c_str(),
            Ops::NN::FormatString("The value of %s must be %s", "permX1", "[1,0,2]").c_str());
        return ge::GRAPH_FAILED;
    }
    // perm_x2
    const int64_t* bPerm = static_cast<const int64_t*>(bPermList->GetData());
    bool bPermCheck = ((bPerm[BATCH_IDX] == 0L) && (bPerm[KB_IDX] == 1L) && (bPerm[N_IDX] == 2L)); // bPerm is [0,1,2]
    if (isMxfp8 || isHIFP8) {
        bPermCheck = bPermCheck || ((bPerm[BATCH_IDX] == 0L) && (bPerm[KB_IDX] == 2L) && (bPerm[N_IDX] == 1L));
    }
    if (!bPermCheck) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            args.opName, "permX2",
            Ops::NN::FormatString("[%ld, %ld, %ld]", bPerm[BATCH_IDX], bPerm[KB_IDX], bPerm[N_IDX]).c_str(),
            Ops::NN::FormatString("The value of %s must be in %s", "permX2",
                                  isMxfp8 ? "{[0,1,2], [0,2,1]}" : "{[0,1,2]}")
                .c_str());
        return ge::GRAPH_FAILED;
    }
    // perm_y
    const int64_t* yPerm = static_cast<const int64_t*>(yPermList->GetData());
    bool yPermCheck = (yPerm[BATCH_IDX] == 1L) && (yPerm[M_IDX] == 0L) && (yPerm[N_IDX] == 2L); // yPerm is [1,0,2]
    if (!yPermCheck) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            args.opName, "permY",
            Ops::NN::FormatString("[%ld, %ld, %ld]", yPerm[BATCH_IDX], yPerm[M_IDX], yPerm[N_IDX]).c_str(),
            Ops::NN::FormatString("The value of %s must be %s", "permY", "[1,0,2]").c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TQBMMGetShape(const gert::TilingContext& context, MatMulV3Args& args, bool isMXFP8, bool isHIFP8)
{
    const gert::Shape& aShape = context.GetInputShape(0)->GetOriginShape();
    const gert::Shape& bShape = context.GetInputShape(1)->GetOriginShape();
    auto attrs = context.GetAttrs();
    const gert::ContinuousVector* aPermList = attrs->GetAttrPointer<gert::ContinuousVector>(PERM_X1_IDX);
    const gert::ContinuousVector* bPermList = attrs->GetAttrPointer<gert::ContinuousVector>(PERM_X2_IDX);
    const gert::ContinuousVector* yPermList = attrs->GetAttrPointer<gert::ContinuousVector>(PERM_Y_IDX);
    OP_TILING_CHECK((TQBMMCheckPerm(aPermList, bPermList, yPermList, args, isMXFP8, isHIFP8) != ge::GRAPH_SUCCESS),
                    CUBE_INNER_ERR_REPORT(args.opName, "check perm failed"), return ge::GRAPH_FAILED);
    OP_TILING_CHECK(
        (TQBMMGetShapeMKN(aShape, bShape, aPermList, bPermList, args, isMXFP8, isHIFP8) != ge::GRAPH_SUCCESS),
        CUBE_INNER_ERR_REPORT(args.opName, "get m/k/n failed"), return ge::GRAPH_FAILED);
    if (attrs->GetAttrNum() >= ATTR_NUM) {
        int32_t batchSplitFactor = *(attrs->GetAttrPointer<int32_t>(ATTR_NUM - 1));
        bool batchSplitFactorPermCheck = (batchSplitFactor == 1);
        if (!batchSplitFactorPermCheck) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                args.opName, "batchSplitFactor", Ops::NN::FormatString("%d", batchSplitFactor).c_str(),
                Ops::NN::FormatString("The value of %s must be %d", "batchSplitFactor", 1).c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus IsValidDtype(const gert::TilingContext& context, const MatMulV3Args& args, bool isHIFP8)
{
    auto scaleX1Desc = context.GetOptionalInputDesc(SCALE_X1_IDX);
    auto scaleX2Desc = context.GetOptionalInputDesc(SCALE_X2_IDX);
    if ((!isHIFP8 && scaleX1Desc == nullptr) || scaleX2Desc == nullptr) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            args.opName, "x1Scale, x2Scale", "nullptr",
            Ops::NN::FormatString("The values of %s cannot be %s", "x1Scale, x2Scale", "null").c_str());
        return ge::GRAPH_FAILED;
    }
    auto scaleX1Dtype = isHIFP8 ? ge::DT_UINT64 : scaleX1Desc->GetDataType();
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
        {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_BF16},
        {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT16},
        {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E8M0, ge::DT_FLOAT8_E8M0, ge::DT_BF16},
        {ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_UINT64, ge::DT_UINT64, ge::DT_FLOAT16},
        {ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_UINT64, ge::DT_UINT64, ge::DT_BF16},
        {ge::DT_HIFLOAT8, ge::DT_HIFLOAT8, ge::DT_UINT64, ge::DT_UINT64, ge::DT_HIFLOAT8}};
    for (auto& supported : dtypeSuportList) {
        if (std::equal(dtype.begin(), dtype.end(), supported.begin())) {
            return ge::GRAPH_SUCCESS;
        }
    }
    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
        args.opName, "x1, x2, x1Scale, x2Scale, y",
        Ops::NN::FormatString(
            "%s, %s, %s, %s, %s", Ops::Base::ToString(args.aType).c_str(), Ops::Base::ToString(args.bType).c_str(),
            Ops::Base::ToString(scaleX1Dtype).c_str(), Ops::Base::ToString(scaleX2Dtype).c_str(),
            Ops::Base::ToString(args.cType).c_str())
            .c_str(),
        Ops::NN::FormatString("The dtypes of %s must be in the supported combinations", "x1, x2, x1Scale, x2Scale, y")
            .c_str());
    return ge::GRAPH_FAILED;
}

ge::graphStatus IsValidFormat(const gert::TilingContext& context, const MatMulV3Args& args, bool isMXFP8)
{
    if (args.aFormat == ge::FORMAT_FRACTAL_NZ || args.outFormat == ge::FORMAT_FRACTAL_NZ) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
            args.opName, "x1, y",
            Ops::NN::FormatString(
                "%s, %s", ge::TypeUtils::FormatToSerialString(args.aFormat).c_str(),
                ge::TypeUtils::FormatToSerialString(args.outFormat).c_str())
                .c_str(),
            Ops::NN::FormatString("The formats of %s cannot be %s", "x1, y", "FRACTAL_NZ").c_str());
        return ge::GRAPH_FAILED;
    }
    if (args.bFormat == ge::FORMAT_FRACTAL_NZ && !isMXFP8) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
            args.opName, "x2", "FRACTAL_NZ",
            Ops::NN::FormatString("In %s case, the format of %s cannot be %s",
                                  "non-mxfp8 mode", "x2", "FRACTAL_NZ").c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace

namespace optiling {
namespace transpose_quant_batch_mat_mul_advanced {
ge::graphStatus TransposeQuantBatchMatMulTiling::GetArgs()
{
    TQBMMGetFormat(*context_, args_);
    TQBMMGetDtype(*context_, args_);
    if (TQBMMGetShape(*context_, args_, isMXFP8_, isHIFP8_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (IsValidFormat(*context_, args_, isMXFP8_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return IsValidDtype(*context_, args_, isHIFP8_);
}

ge::graphStatus TransposeQuantBatchMatMulTiling::CheckScale(
    const int64_t b, const int64_t m, const int64_t n, const int64_t k, const int64_t* bPerm) const
{
    // scale
    auto scaleX1ShapePtr = context_->GetOptionalInputShape(SCALE_X1_IDX);
    auto scaleX2ShapePtr = context_->GetOptionalInputShape(SCALE_X2_IDX);
    // scaleX1 can be nullptr in  hifp8 mode
    if ((!isHIFP8_ && scaleX1ShapePtr == nullptr) || scaleX2ShapePtr == nullptr) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            args_.opName, "x1Scale, x2Scale", "nullptr",
            Ops::NN::FormatString("The values of %s cannot be %s", "x1Scale, x2Scale", "null").c_str());
        return ge::GRAPH_FAILED;
    }
    auto scaleX1DimNum = isHIFP8_ ? 1 : scaleX1ShapePtr->GetStorageShape().GetDimNum();
    auto scaleX2DimNum = scaleX2ShapePtr->GetStorageShape().GetDimNum();
    if (isMXFP8_) {
        int64_t numGroup = ops::CeilDiv(ops::CeilDiv(k, SUPPORTED_GROUP_SIZE), NUM_TWO);
        bool invalidScaleX1 =
            scaleX1DimNum != EXPECTED_MX_SCALE_DIM || scaleX1ShapePtr->GetStorageShape().GetDim(0) != m ||
            scaleX1ShapePtr->GetStorageShape().GetDim(1) != b ||
            scaleX1ShapePtr->GetStorageShape().GetDim(NUM_TWO) != numGroup ||
            scaleX1ShapePtr->GetStorageShape().GetDim(NUM_THREE) != NUM_TWO;
        if (invalidScaleX1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                args_.opName, "x1Scale", Ops::Base::ToString(scaleX1ShapePtr->GetStorageShape()).c_str(),
                Ops::NN::FormatString(
                    "In %s case, the shape of %s must be %s", "MXFp8", "x1Scale",
                    Ops::NN::FormatString("[%ld, %ld, %ld, 2]", m, b, numGroup).c_str())
                    .c_str());
            return ge::GRAPH_FAILED;
        }
        int64_t scaleN = scaleX2ShapePtr->GetStorageShape().GetDim(bPerm[NUM_TWO]);
        int64_t scaleGroupNum = scaleX2ShapePtr->GetStorageShape().GetDim(bPerm[1]);
        bool invalidScaleX2 =
            scaleX2DimNum != EXPECTED_MX_SCALE_DIM || scaleX2ShapePtr->GetStorageShape().GetDim(0) != b ||
            scaleN != n || scaleGroupNum != numGroup ||
            scaleX2ShapePtr->GetStorageShape().GetDim(NUM_THREE) != NUM_TWO;
        if (invalidScaleX2) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                args_.opName, "x2Scale", Ops::Base::ToString(scaleX2ShapePtr->GetStorageShape()).c_str(),
                Ops::NN::FormatString(
                    "In %s case, the shape of %s must be %s", "MXFp8", "x2Scale", "[b, numGroup, n, 2] after permX2")
                    .c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        if (!isHIFP8_) {
            if (scaleX1DimNum != EXPECTED_SCALE_DIM) {
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    args_.opName, "x1Scale", Ops::NN::FormatString("%zu", scaleX1DimNum).c_str(),
                    Ops::NN::FormatString("The shape dim of %s must be %zu", "x1Scale", EXPECTED_SCALE_DIM).c_str());
                return ge::GRAPH_FAILED;
            }
            if (scaleX1ShapePtr->GetStorageShape().GetDim(0) != m) {
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    args_.opName, "x1Scale",
                    Ops::NN::FormatString("%ld", scaleX1ShapePtr->GetStorageShape().GetDim(0)).c_str(),
                    Ops::NN::FormatString(
                        "%s of %s must be equal to %s of %s (%ld)", "Shape[0]", "x1Scale", "m-axis", "x1", m)
                        .c_str());
                return ge::GRAPH_FAILED;
            }
        }
        if (scaleX2DimNum != EXPECTED_SCALE_DIM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                args_.opName, "x2Scale", Ops::NN::FormatString("%zu", scaleX2DimNum).c_str(),
                Ops::NN::FormatString("The shape dim of %s must be %zu", "x2Scale", EXPECTED_SCALE_DIM).c_str());
            return ge::GRAPH_FAILED;
        }
        if (scaleX2ShapePtr->GetStorageShape().GetDim(0) != n) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                args_.opName, "x2Scale",
                Ops::NN::FormatString("%ld", scaleX2ShapePtr->GetStorageShape().GetDim(0)).c_str(),
                Ops::NN::FormatString(
                    "%s of %s must be equal to %s of %s (%ld)", "Shape[0]", "x2Scale", "n-axis", "x2", n)
                    .c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeQuantBatchMatMulTiling::CheckArgs()
{
    OP_TILING_CHECK(
        (CheckNotNull(context_) != ge::GRAPH_SUCCESS),
        CUBE_INNER_ERR_REPORT(args_.opName, "parameter should not be nullptr"), return ge::GRAPH_FAILED);
    auto attrs = context_->GetAttrs();
    const gert::ContinuousVector* aPermList = attrs->GetAttrPointer<gert::ContinuousVector>(PERM_X1_IDX);
    const gert::ContinuousVector* bPermList = attrs->GetAttrPointer<gert::ContinuousVector>(PERM_X2_IDX);
    const gert::ContinuousVector* yPermList = attrs->GetAttrPointer<gert::ContinuousVector>(PERM_Y_IDX);
    if (aPermList == nullptr || bPermList == nullptr || yPermList == nullptr) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            args_.opName, "permX1, permX2, permY", "nullptr",
            Ops::NN::FormatString("The values of %s cannot be %s", "permX1, permX2, permY", "null").c_str());
        return ge::GRAPH_FAILED;
    }
    if (aPermList->GetSize() != ALLOW_DIM) {
        OP_LOGE_FOR_INVALID_LISTSIZE(
            args_.opName, "permX1",
            std::to_string(aPermList->GetSize()).c_str(), std::to_string(ALLOW_DIM).c_str());
        return ge::GRAPH_FAILED;
    }
    if (bPermList->GetSize() != ALLOW_DIM) {
        OP_LOGE_FOR_INVALID_LISTSIZE(
            args_.opName, "permX2",
            std::to_string(bPermList->GetSize()).c_str(), std::to_string(ALLOW_DIM).c_str());
        return ge::GRAPH_FAILED;
    }
    if (yPermList->GetSize() != ALLOW_DIM) {
        OP_LOGE_FOR_INVALID_LISTSIZE(
            args_.opName, "permY",
            std::to_string(yPermList->GetSize()).c_str(), std::to_string(ALLOW_DIM).c_str());
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& aShape = context_->GetInputShape(X1_IDX)->GetOriginShape();
    const gert::Shape& bShape = context_->GetInputShape(X2_IDX)->GetOriginShape();
    const gert::Shape& cShape = context_->GetOutputShape(0)->GetOriginShape();
    const size_t aDimNum = aShape.GetDimNum();
    const size_t bDimNum = bShape.GetDimNum();
    const size_t cDimNum = cShape.GetDimNum();
    if ((aDimNum != ALLOW_DIM) || (bDimNum != ALLOW_DIM) || (cDimNum != ALLOW_DIM)) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            args_.opName, "x1, x2, y", Ops::NN::FormatString("%zu, %zu, %zu", aDimNum, bDimNum, cDimNum).c_str(),
            Ops::NN::FormatString("The shape dims of %s must be %d", "x1, x2, y", static_cast<int>(ALLOW_DIM))
                .c_str());
        return ge::GRAPH_FAILED;
    }
    const int64_t* aPerm = static_cast<const int64_t*>(aPermList->GetData());
    const int64_t* bPerm = static_cast<const int64_t*>(bPermList->GetData());
    // check scale
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOptionalInputDesc(SCALE_X2_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOptionalInputShape(SCALE_X2_IDX));
    isHIFP8_ = IsHIFP8(context_->GetInputDesc(X1_IDX), context_->GetInputDesc(X2_IDX));
    isMXFP8_ = IsMicroScaling(context_->GetOptionalInputDesc(SCALE_X1_IDX),
                              context_->GetOptionalInputDesc(SCALE_X2_IDX));
    OP_TILING_CHECK((CheckScale(
        aShape[aPerm[BATCH_IDX]], aShape[aPerm[M_IDX]], bShape[bPerm[N_IDX]], aShape[aPerm[KA_IDX]], bPerm) !=
        ge::GRAPH_SUCCESS), CUBE_INNER_ERR_REPORT(args_.opName, "invalid scale"), return ge::GRAPH_FAILED);
    // check group_size
    const int64_t* groupSize = attrs->GetAttrPointer<int64_t>(GROUP_SIZE_IDX);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, groupSize);
    OP_TILING_CHECK(
        (IsValidGroupSize(*groupSize, isMXFP8_, args_) != ge::GRAPH_SUCCESS),
        CUBE_INNER_ERR_REPORT(args_.opName, "invalid group_size"), return ge::GRAPH_FAILED);
    // check format
    if (ge::GetPrimaryFormat(context_->GetInputDesc(1)->GetStorageFormat()) == ge::FORMAT_FRACTAL_NZ) {
        OP_TILING_CHECK(
            (CheckWeightNz(*context_, args_, isMXFP8_) != ge::GRAPH_SUCCESS),
            CUBE_INNER_ERR_REPORT(args_.opName, "invalid weightNz"), return ge::GRAPH_FAILED);
    }
    // bias
    OP_TILING_CHECK(
        (context_->GetOptionalInputShape(BIAS_IDX) != nullptr),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            args_.opName, "bias", "not nullptr",
            Ops::NN::FormatString("The value of %s must be %s", "bias", "null").c_str()),
        return ge::GRAPH_FAILED);
    if (attrs->GetAttrNum() >= ATTR_NUM) {
        OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<int32_t>(ATTR_NUM - 1));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeQuantBatchMatMulTiling::GetShapeAttrsInfo()
{
    args_.opName = context_->GetNodeName();
    OP_TILING_CHECK(
        args_.opName == nullptr, CUBE_INNER_ERR_REPORT("TransposeQuantBatchMatMul", "get op name invalid"),
        return ge::GRAPH_FAILED);
    OP_LOGI(args_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    OP_TILING_CHECK(
        (CheckArgs() != ge::GRAPH_SUCCESS) || (GetArgs() != ge::GRAPH_SUCCESS),
        CUBE_INNER_ERR_REPORT(args_.opName, "invalid context"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeQuantBatchMatMulTiling::DoTiling()
{
    if (GetShapeAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    MatMulV3BatchInfo tempBatchInfo;
    OP_TILING_CHECK(
        (GetBatchInfo(*context_, args_, tempBatchInfo) != ge::GRAPH_SUCCESS),
        CUBE_INNER_ERR_REPORT(args_.opName, "GetBatchInfo failed"), return ge::GRAPH_FAILED);
    args_.batchInfo = &tempBatchInfo;
    MatMulTilingCfg tilingCfg(false, context_->GetCompileInfo(), static_cast<void*>(&args_));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, tilingCfg.compileInfo);
    NpuArch npuArch = static_cast<const MatmulV3CompileInfo*>(tilingCfg.compileInfo)->npuArch;
    MMRegisterCfg registerCfg{
        "TransposeQuantBatchMatMul", npuArch, strategy::GetTransposeQuantBatchMatMulPriorities(npuArch)};
    return MMTilingRegistry::GetInstance().DoTilingImpl(context_, tilingCfg, registerCfg);
}

ge::graphStatus TransposeQuantBatchMatMulTiling::GetBatchInfo(
    const gert::TilingContext& context, MatMulV3Args& args, MatMulV3BatchInfo& batchInfo) const
{
    const gert::Shape& aShape = context.GetInputShape(X1_IDX)->GetOriginShape();
    const gert::Shape& bShape = context.GetInputShape(X2_IDX)->GetOriginShape();

    auto attrs = context.GetAttrs();
    const gert::ContinuousVector* aPermList = attrs->GetAttrPointer<gert::ContinuousVector>(PERM_X1_IDX);
    const gert::ContinuousVector* bPermList = attrs->GetAttrPointer<gert::ContinuousVector>(PERM_X2_IDX);

    uint64_t batchA = static_cast<uint64_t>(aShape[BATCH_IDX]);
    uint64_t batchB = static_cast<uint64_t>(bShape[BATCH_IDX]);
    const int64_t* aPerm = static_cast<const int64_t*>(aPermList->GetData());
    batchA = aShape[aPerm[BATCH_IDX]];
    const int64_t* bPerm = static_cast<const int64_t*>(bPermList->GetData());
    batchB = bShape[bPerm[BATCH_IDX]];
    if (batchA != batchB) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            args.opName, "x1, x2",
            Ops::NN::FormatString("%s, %s", Ops::Base::ToString(aShape).c_str(), Ops::Base::ToString(bShape).c_str())
                .c_str(),
            Ops::NN::FormatString("%s of %s must be equal", "Batch-axis", "x1, x2").c_str());
        return ge::GRAPH_FAILED;
    }
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