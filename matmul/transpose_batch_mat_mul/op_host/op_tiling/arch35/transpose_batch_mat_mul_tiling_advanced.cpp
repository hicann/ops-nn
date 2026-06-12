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
 * \file transpose_batch_mat_mul_tiling_advanced.cc
 * \brief
 */

#include "transpose_batch_mat_mul_tiling_advanced.h"

#include "register/op_def_registry.h"
#include "common/op_host/op_tiling/debug_tiling.h"

#include "transpose_batch_mat_mul_tiling_strategy.h"
#include "transpose_batch_mat_mul_common.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_cfg.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_compile_info_advanced.h"
#include "matmul/common/op_host/log_format_util.h"

namespace {
using namespace optiling;
using namespace optiling::transpose_batch_mat_mul_advanced;
using namespace optiling::matmul_v3_advanced;

constexpr uint64_t ONE_BATCH_DIM = 1UL;

static inline ge::graphStatus TBMMCheckNotNull(const gert::TilingContext* context)
{
    auto attrs = context->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const gert::ContinuousVector* aPermList = attrs->GetAttrPointer<gert::ContinuousVector>(0);
    const gert::ContinuousVector* bPermList = attrs->GetAttrPointer<gert::ContinuousVector>(1);
    const gert::ContinuousVector* yPermList = attrs->GetAttrPointer<gert::ContinuousVector>(NUM_TWO);
    OPS_CHECK_NULL_WITH_CONTEXT(context, aPermList);
    OPS_CHECK_NULL_WITH_CONTEXT(context, bPermList);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yPermList);
    OPS_CHECK_NULL_WITH_CONTEXT(context, aPermList->GetData());
    OPS_CHECK_NULL_WITH_CONTEXT(context, bPermList->GetData());
    OPS_CHECK_NULL_WITH_CONTEXT(context, yPermList->GetData());
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(0));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(0));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputDesc(1));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetInputShape(1));
    OPS_CHECK_NULL_WITH_CONTEXT(context, context->GetOutputDesc(0));
    if (attrs->GetAttrNum() >= HF32_ATTR_NUM) {
        OPS_CHECK_NULL_WITH_CONTEXT(context, attrs->GetAttrPointer<bool>(HF32_ATTR_INDEX));
    }
    if (attrs->GetAttrNum() >= ATTR_NUM) {
        OPS_CHECK_NULL_WITH_CONTEXT(context, attrs->GetAttrPointer<int32_t>(ATTR_NUM - 1));
    }
    return ge::GRAPH_SUCCESS;
}

static inline ge::graphStatus TBMMCheckPermAndShapeDim(const gert::TilingContext& context, const char* opName)
{
    const gert::Shape& aShape = context.GetInputShape(0)->GetOriginShape();
    const gert::Shape& bShape = context.GetInputShape(1)->GetOriginShape();
    const gert::Shape& cShape = context.GetOutputShape(0)->GetOriginShape();
    if ((aShape.GetDimNum() != ALLOW_DIM) || (bShape.GetDimNum() != ALLOW_DIM) || (cShape.GetDimNum() != ALLOW_DIM)) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            opName, "x1, x2, out",
            Ops::NN::FormatString("%zu, %zu, %zu", aShape.GetDimNum(), bShape.GetDimNum(), cShape.GetDimNum()).c_str(),
            Ops::NN::FormatString("The shape dims of %s must be %d", "x1, x2, out", ALLOW_DIM).c_str());
        return ge::GRAPH_FAILED;
    }
    auto attrs = context.GetAttrs();
    if ((attrs->GetAttrPointer<gert::ContinuousVector>(0)->GetSize() != ALLOW_DIM) ||
        (attrs->GetAttrPointer<gert::ContinuousVector>(1)->GetSize() != ALLOW_DIM) ||
        (attrs->GetAttrPointer<gert::ContinuousVector>(NUM_TWO)->GetSize() != ALLOW_DIM)) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            opName, "perm_x1, perm_x2, perm_y",
            Ops::NN::FormatString("%zu, %zu, %zu", attrs->GetAttrPointer<gert::ContinuousVector>(0)->GetSize(),
                                  attrs->GetAttrPointer<gert::ContinuousVector>(1)->GetSize(),
                                  attrs->GetAttrPointer<gert::ContinuousVector>(NUM_TWO)->GetSize())
                .c_str(),
            Ops::NN::FormatString("The shape dims of %s must be %d", "perm_x1, perm_x2, perm_y", ALLOW_DIM).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static inline void TBMMGetFormat(const gert::TilingContext &context, MatMulV3Args &args)
{
    ge::Format formatA = static_cast<ge::Format>(ge::GetPrimaryFormat(context.GetInputDesc(0)->GetStorageFormat()));
    ge::Format formatB = static_cast<ge::Format>(ge::GetPrimaryFormat(context.GetInputDesc(1)->GetStorageFormat()));
    ge::Format formatOut = static_cast<ge::Format>(ge::GetPrimaryFormat(context.GetOutputDesc(0)->GetStorageFormat()));
    args.aFormat = (formatA != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatA;
    args.bFormat = (formatB != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatB;
    args.outFormat = (formatOut != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatOut;
}

static inline void TBMMGetDtype(const gert::TilingContext &context, MatMulV3Args &args)
{
    args.aType = context.GetInputDesc(0)->GetDataType();
    args.bType = context.GetInputDesc(1)->GetDataType();
    args.cType = context.GetOutputDesc(0)->GetDataType();
    // op_impl_mode_enum: 0x1: default 0x2: high_performance 0x4: high_precision 0x8: super_performance
    // 0x10: support_of_bound_index 0x20: enable_float_32_execution 0x40: enable_hi_float_32_execution
    if (context.GetAttrs()->GetAttrNum() >= HF32_ATTR_NUM) {
        args.isHf32 = *context.GetAttrs()->GetAttrPointer<bool>(HF32_ATTR_INDEX);
    }
    args.aDtypeSize = ge::GetSizeByDataType(args.aType);
    args.bDtypeSize = ge::GetSizeByDataType(args.bType);
    OP_LOGD(args.opName, "MatMulV3 Hf32 flag is: %d", args.isHf32);
}

ge::graphStatus IsValidDtype(const MatMulV3Args &args)
{
    std::vector<ge::DataType> dtype = { args.aType, args.bType, args.cType };
    const std::vector<std::vector<ge::DataType>> dtypeSuportList = {
        // x1,              x2,             y,              bias
        { ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16 },
        { ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT },
        { ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT},
        { ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT },
        { ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT },
        { ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16 } // david supports bias-bf16
    };
    for (auto &supported : dtypeSuportList) {
        if (std::equal(dtype.begin(), dtype.end(), supported.begin())) {
            return ge::GRAPH_SUCCESS;
        }
    }
    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
        args.opName, "x1, x2, out",
        Ops::NN::FormatString(
            "%s, %s, %s", Ops::Base::ToString(args.aType).c_str(), Ops::Base::ToString(args.bType).c_str(),
            Ops::Base::ToString(args.cType).c_str())
            .c_str(),
        Ops::NN::FormatString(
            "The dtypes of %s must be within the range %s", "x1, x2, out",
            "(FLOAT16,FLOAT16,FLOAT16), (FLOAT16,FLOAT16,INT8), (FLOAT,FLOAT,FLOAT), (BF16,BF16,BF16)")
            .c_str());
    return ge::GRAPH_FAILED;
}

ge::graphStatus TBMMOpSpecificCheck(MatMulV3Args &args)
{
    // format check
    if ((args.aFormat == ge::FORMAT_FRACTAL_NZ) || (args.outFormat == ge::FORMAT_FRACTAL_NZ)) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
            args.opName, "x1, out",
            Ops::NN::FormatString(
                "%s, %s", (args.aFormat == ge::FORMAT_FRACTAL_NZ) ? "FRACTAL_NZ" : "ND",
                (args.outFormat == ge::FORMAT_FRACTAL_NZ) ? "FRACTAL_NZ" : "ND")
                .c_str(),
            Ops::NN::FormatString("The formats of %s must be %s", "x1, out", "ND").c_str());
        return ge::GRAPH_FAILED;
    }

    if (args.hasBias) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            args.opName, "bias", "", Ops::NN::FormatString("The value of %s must be %s", "bias", "null").c_str());
        return ge::GRAPH_FAILED;
    }

    // dtype check
    return IsValidDtype(args);
}

ge::graphStatus TBMMGetShapeMKN(const gert::Shape &aShape, const gert::Shape &bShape,
                                const gert::ContinuousVector *aPermList, const gert::ContinuousVector *bPermList,
                                MatMulV3Args &args)
{
    const int64_t* aPerm = static_cast<const int64_t*>(aPermList->GetData());
    int64_t m = aShape[aPerm[M_IDX]];
    int64_t kA = aShape[aPerm[KA_IDX]];
    args.isATrans = aPerm[M_IDX] > aPerm[KA_IDX];

    const int64_t* bPerm = static_cast<const int64_t*>(bPermList->GetData());
    int64_t kB = bShape[bPerm[KB_IDX]];
    int64_t n = bShape[bPerm[N_IDX]];
    args.isBTrans = bPerm[KB_IDX] > bPerm[N_IDX];

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
    args.mValue = static_cast<uint64_t>(m);
    args.kValue = static_cast<uint64_t>(kA);
    args.nValue = static_cast<uint64_t>(n);
    args.mOriValue = args.mValue;
    args.nOriValue = args.nValue;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TBMMGetShape(const gert::TilingContext &context, MatMulV3Args &args)
{
    const gert::Shape& aShape = context.GetInputShape(0)->GetOriginShape();
    const gert::Shape& bShape = context.GetInputShape(1)->GetOriginShape();

    auto attrs = context.GetAttrs();
    size_t idx = 0;
    const gert::ContinuousVector *aPermList = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    const gert::ContinuousVector *bPermList = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    const gert::ContinuousVector *yPermList = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    const int64_t* aPerm = static_cast<const int64_t*>(aPermList->GetData());
    bool aPermCheck =
        ((aPerm[BATCH_IDX] == 1L) && (aPerm[M_IDX] == 0L) && (aPerm[KA_IDX] == 2L)) ||
        ((aPerm[BATCH_IDX] == 0L) && (aPerm[M_IDX] == 1L) && (aPerm[KA_IDX] == 2L)); // aPerm is [1,0,2] or [0,1,2]
    if (!aPermCheck) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            args.opName, "permX1",
            Ops::NN::FormatString("[%ld, %ld, %ld]", aPerm[BATCH_IDX], aPerm[M_IDX], aPerm[KA_IDX]).c_str(),
            Ops::NN::FormatString("The value of %s must be in %s", "permX1", "{[1,0,2], [0,1,2]}").c_str());
        return ge::GRAPH_FAILED;
    }
    const int64_t* bPerm = static_cast<const int64_t*>(bPermList->GetData());
    bool bPermCheck = ((bPerm[BATCH_IDX] == 0L) && (bPerm[KB_IDX] == 1L) && (bPerm[N_IDX] == 2L)) || // bPerm 012 or
                      ((bPerm[BATCH_IDX] == 0L) && (bPerm[KB_IDX] == 2L) && (bPerm[N_IDX] == 1L));   // bPerm 021
    if (!bPermCheck) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            args.opName, "permX2",
            Ops::NN::FormatString("[%ld, %ld, %ld]", bPerm[BATCH_IDX], bPerm[KB_IDX], bPerm[N_IDX]).c_str(),
            Ops::NN::FormatString("The value of %s must be in %s", "permX2", "{[0,1,2], [0,2,1]}").c_str());
        return ge::GRAPH_FAILED;
    }
    const int64_t* yPerm = static_cast<const int64_t*>(yPermList->GetData());
    bool yPermCheck = (yPerm[BATCH_IDX] == 1L) && (yPerm[M_IDX] == 0L) && (yPerm[N_IDX] == 2L); // yPerm is [1,0,2]
    if (!yPermCheck) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            args.opName, "permY",
            Ops::NN::FormatString("[%ld, %ld, %ld]", yPerm[BATCH_IDX], yPerm[M_IDX], yPerm[N_IDX]).c_str(),
            Ops::NN::FormatString("The value of %s must be %s", "permY", "[1,0,2]").c_str());
        return ge::GRAPH_FAILED;
    }

    OP_TILING_CHECK((TBMMGetShapeMKN(aShape, bShape, aPermList, bPermList, args) != ge::GRAPH_SUCCESS),
                    CUBE_INNER_ERR_REPORT(args.opName, "get m/k/n failed"), return ge::GRAPH_FAILED);

    if (attrs->GetAttrNum() >= ATTR_NUM) {
        uint32_t batchSplitFactor = std::max(*(attrs->GetAttrPointer<int32_t>(ATTR_NUM - 1)), 1);
        bool batchSplitFactorPermCheck =
            aShape[static_cast<const int64_t*>(aPermList->GetData())[0]] % batchSplitFactor == 0;
        if (!batchSplitFactorPermCheck) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                args.opName, "x1", Ops::Base::ToString(aShape).c_str(),
                Ops::NN::FormatString("%s of %s must be exactly divisible by attribute %s (%u)", "Batch-axis", "x1",
                                      "batch_split_factor", batchSplitFactor)
                    .c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}
}

namespace optiling {
namespace transpose_batch_mat_mul_advanced {

ge::graphStatus TransposeBatchMatMulTiling::GetArgs()
{
    TBMMGetFormat(*context_, args_);
    TBMMGetDtype(*context_, args_);
    if (TBMMGetShape(*context_, args_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return TBMMOpSpecificCheck(args_);
}

ge::graphStatus TransposeBatchMatMulTiling::CheckScale(const gert::Shape& shape_scale) const
{
    const gert::Shape& bShape = context_->GetInputShape(1)->GetOriginShape();
    auto attrs = context_->GetAttrs();
    const gert::ContinuousVector* bPermList = attrs->GetAttrPointer<gert::ContinuousVector>(1);
    if (attrs->GetAttrNum() >= ATTR_NUM) {
        uint32_t batchSplitFactor = std::max(*(attrs->GetAttrPointer<int32_t>(ATTR_NUM - 1)), 1);
        if (batchSplitFactor != 1) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                args_.opName, "batchSplitFactor", Ops::NN::FormatString("%u", batchSplitFactor).c_str(),
                Ops::NN::FormatString(
                    "When optional parameter %s exists, the value of %s must be %d", "scale", "batchSplitFactor", 1)
                    .c_str());
            return ge::GRAPH_FAILED;
        }
    }
    const int64_t* bPerm = static_cast<const int64_t*>(bPermList->GetData());
    int64_t n = bShape[bPerm[N_IDX]];
    int64_t b = bShape[bPerm[BATCH_IDX]];
    if (shape_scale.GetDim(0) != b * n) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            args_.opName, "scale", Ops::NN::FormatString("%ld", shape_scale.GetDim(0)).c_str(),
            Ops::NN::FormatString("%s of %s must be equal to %s of %s multiplied by %s of %s (%ld)", "Shape[0]",
                                  "scale", "batch-axis", "x2", "n-axis", "x2", b * n)
                .c_str());
        return ge::GRAPH_FAILED;
    }
    auto tensor_x1 = context_->GetInputDesc(0);
    auto tensor_x2 = context_->GetInputDesc(1);
    ge::DataType dtype_x1 = tensor_x1->GetDataType();
    ge::DataType dtype_x2 = tensor_x2->GetDataType();
    if (dtype_x2 != ge::DT_FLOAT16 || dtype_x1 != ge::DT_FLOAT16) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            args_.opName, "x1, x2",
            Ops::NN::FormatString(
                "%s, %s", Ops::Base::ToString(dtype_x1).c_str(), Ops::Base::ToString(dtype_x2).c_str())
                .c_str(),
            Ops::NN::FormatString(
                "When optional parameter %s exists, the dtypes of %s must be %s", "scale", "x1, x2", "FLOAT16")
                .c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeBatchMatMulTiling::CheckArgs()
{
    if (TBMMCheckNotNull(context_) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (TBMMCheckPermAndShapeDim(*context_, args_.opName) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (context_->GetOptionalInputShape(BIAS_IDX) != nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            args_.opName, "bias", "", Ops::NN::FormatString("The value of %s must be %s", "bias", "null").c_str());
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape* storageShape_scale = context_->GetOptionalInputShape(SCALE_IDX);
    if (storageShape_scale != nullptr) {
        const gert::Shape shape_scale = storageShape_scale->GetOriginShape();
        OP_TILING_CHECK((CheckScale(shape_scale) != ge::GRAPH_SUCCESS),
                        CUBE_INNER_ERR_REPORT(args_.opName, "scale condition not satisfied"), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeBatchMatMulTiling::GetShapeAttrsInfo() // TBMM manages shape attrs itself
{
    args_.opName = context_->GetNodeName();
    OP_TILING_CHECK(args_.opName == nullptr, CUBE_INNER_ERR_REPORT("TransposeBatchMatMul", "get op name invalid"),
                    return ge::GRAPH_FAILED);
    OP_LOGI(args_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    OP_TILING_CHECK((CheckArgs() != ge::GRAPH_SUCCESS) || (GetArgs() != ge::GRAPH_SUCCESS),
                    CUBE_INNER_ERR_REPORT(args_.opName, "invalid context"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeBatchMatMulTiling::DoTiling()
{
    if (GetShapeAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    MatMulV3BatchInfo tempBatchInfo;
    OP_TILING_CHECK((GetBatchInfo(*context_, args_, tempBatchInfo) != ge::GRAPH_SUCCESS),
                    CUBE_INNER_ERR_REPORT(args_.opName, "GetBatchInfo failed"), return ge::GRAPH_FAILED);
    args_.batchInfo = &tempBatchInfo;
    if (context_->GetOptionalInputShape(SCALE_IDX) != nullptr) {
        args_.hasScale = true;
        OP_LOGI(args_.opName, "Quant function is activated.");
    }
    MatMulTilingCfg tilingCfg(false, context_->GetCompileInfo(), static_cast<void*>(&args_));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, tilingCfg.compileInfo);
    NpuArch npuArch =
        static_cast<const MatmulV3CompileInfo*>(tilingCfg.compileInfo)->npuArch;
    MMRegisterCfg registerCfg{"TransposeBatchMatMul", npuArch,
                              strategy::GetTransposeBatchMatMulPriorities(npuArch)};
    return MMTilingRegistry::GetInstance().DoTilingImpl(context_, tilingCfg, registerCfg);
}

ge::graphStatus TransposeBatchMatMulTiling::GetBatchInfo(const gert::TilingContext &context, MatMulV3Args& args,
                                                         MatMulV3BatchInfo& batchInfo) const
{
    const gert::Shape &aShape = context.GetInputShape(0)->GetOriginShape();
    const gert::Shape &bShape = context.GetInputShape(1)->GetOriginShape();

    auto attrs = context.GetAttrs();
    size_t idx = 0;
    const gert::ContinuousVector *aPermList = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    const gert::ContinuousVector *bPermList = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);

    const int64_t* aPerm = static_cast<const int64_t*>(aPermList->GetData());
    uint64_t batchA = aShape[aPerm[BATCH_IDX]];
    const int64_t* bPerm = static_cast<const int64_t*>(bPermList->GetData());
    uint64_t batchB = bShape[bPerm[BATCH_IDX]];
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
}
}