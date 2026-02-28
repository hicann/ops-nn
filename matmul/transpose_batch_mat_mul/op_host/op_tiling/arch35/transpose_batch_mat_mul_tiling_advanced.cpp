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

namespace {
using namespace optiling;
using namespace optiling::transpose_batch_mat_mul_advanced;
using namespace optiling::matmul_v3_advanced;

constexpr uint64_t ONE_BATCH_DIM = 1UL;

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
    OP_LOGE(args.opName,
        "Unsupported data type: x1[%s], x2[%s], y[%s], input dtype of x1 and x2 and output dtype must be same, "
        "only support[DT_FLOAT16, DT_FLOAT, DT_BF16]",
        Ops::Base::ToString(args.aType).c_str(), Ops::Base::ToString(args.bType).c_str(), Ops::Base::ToString(args.cType).c_str());
    return ge::GRAPH_FAILED;
}

ge::graphStatus TBMMOpSpecificCheck(MatMulV3Args &args)
{
    // dtype check
    return IsValidDtype(args);
}

ge::graphStatus TBMMGetShapeMKN(const gert::Shape &aShape, const gert::Shape &bShape,
                                const gert::ContinuousVector *aPermList, const gert::ContinuousVector *bPermList,
                                MatMulV3Args &args)
{
    int64_t m = aShape[M_IDX];
    int64_t kA = aShape[KA_IDX];
    int64_t kB = bShape[KB_IDX];
    int64_t n = bShape[N_IDX];

    if(aPermList != nullptr && aPermList->GetSize() == ALLOW_DIM) {
        const int64_t *aPerm = static_cast<const int64_t *>(aPermList->GetData());
        m = aShape[aPerm[M_IDX]];
        kA = aShape[aPerm[KA_IDX]];
        args.isATrans = aPerm[M_IDX] > aPerm[KA_IDX];
    }
    if(bPermList != nullptr && bPermList->GetSize() == ALLOW_DIM) {
        const int64_t *bPerm = static_cast<const int64_t *>(bPermList->GetData());
        kB = bShape[bPerm[KB_IDX]];
        n = bShape[bPerm[N_IDX]];
        args.isBTrans = bPerm[KB_IDX] > bPerm[N_IDX];
    }
    OP_TILING_CHECK(kA != kB, CUBE_INNER_ERR_REPORT(args.opName, "unequal input kDim values"), return ge::GRAPH_FAILED);

    if ((m <= 0) || (kA <= 0) || (n <= 0)) {
        OP_LOGE(args.opName, "illegal value: m[%ld], k[%ld], n[%ld]", m, kA, n);
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
    const gert::Shape &aShape = context.GetInputShape(0)->GetOriginShape();
    const gert::Shape &bShape = context.GetInputShape(1)->GetOriginShape();
    const gert::Shape &cShape = context.GetOutputShape(0)->GetOriginShape();
    const size_t aDimNum = aShape.GetDimNum();
    const size_t bDimNum = bShape.GetDimNum();
    const size_t cDimNum = cShape.GetDimNum();
    OP_TILING_CHECK((aDimNum != ALLOW_DIM) || (bDimNum != ALLOW_DIM) || (cDimNum != ALLOW_DIM),
                    CUBE_INNER_ERR_REPORT(args.opName, "invalid input/output dim num"), return ge::GRAPH_FAILED);

    auto attrs = context.GetAttrs();
    size_t idx = 0;
    const gert::ContinuousVector *aPermList = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    const gert::ContinuousVector *bPermList = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    const gert::ContinuousVector *yPermList = attrs->GetAttrPointer<gert::ContinuousVector>(idx++);
    if(aPermList != nullptr && aPermList->GetSize() == ALLOW_DIM) {
        const int64_t *aPerm = static_cast<const int64_t *>(aPermList->GetData());
        bool aPermCheck =
            ((aPerm[BATCH_IDX] == 1L) && (aPerm[M_IDX] == 0L) && (aPerm[KA_IDX] == 2L)) ||
            ((aPerm[BATCH_IDX] == 0L) && (aPerm[M_IDX] == 1L) && (aPerm[KA_IDX] == 2L)); // aPerm is [1,0,2] or [0,1,2]
        OP_TILING_CHECK(!aPermCheck,
            CUBE_INNER_ERR_REPORT(args.opName, "unsupport aPerm value"), return ge::GRAPH_FAILED);
    }
    if(bPermList != nullptr && bPermList->GetSize() == ALLOW_DIM) {
        const int64_t *bPerm = static_cast<const int64_t *>(bPermList->GetData());
        bool bPermCheck = ((bPerm[BATCH_IDX] == 0L) && (bPerm[KB_IDX] == 1L) && (bPerm[N_IDX] == 2L)) || // bPerm 012 or
                          ((bPerm[BATCH_IDX] == 0L) && (bPerm[KB_IDX] == 2L) && (bPerm[N_IDX] == 1L)); // bPerm 021
        OP_TILING_CHECK(!bPermCheck,
            CUBE_INNER_ERR_REPORT(args.opName, "unsupport bPerm value"), return ge::GRAPH_FAILED);
    }
    if(yPermList != nullptr && yPermList->GetSize() == ALLOW_DIM) {
        const int64_t *yPerm = static_cast<const int64_t *>(yPermList->GetData());
        bool yPermCheck = (yPerm[BATCH_IDX] == 1L) && (yPerm[M_IDX] == 0L) && (yPerm[N_IDX] == 2L); // yPerm is [1,0,2]
        OP_TILING_CHECK(!yPermCheck,
            CUBE_INNER_ERR_REPORT(args.opName, "unsupport yPerm value"), return ge::GRAPH_FAILED);
    }

    OP_TILING_CHECK((TBMMGetShapeMKN(aShape, bShape, aPermList, bPermList, args) != ge::GRAPH_SUCCESS),
                    CUBE_INNER_ERR_REPORT(args.opName, "get m/k/n failed"), return ge::GRAPH_FAILED);

    if (aPermList != nullptr && attrs->GetAttrNum() >= ATTR_NUM) {
        uint32_t batchSplitFactor = std::max(*(attrs->GetAttrPointer<int32_t>(ATTR_NUM - 1)), 1);
        bool batchSplitFactorPermCheck =
            aShape[static_cast<const int64_t*>(aPermList->GetData())[0]] % batchSplitFactor == 0;
        OP_TILING_CHECK(!batchSplitFactorPermCheck,
                        CUBE_INNER_ERR_REPORT(args.opName, "batch_split_factor is not supported."),
                        return ge::GRAPH_FAILED);
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
        OP_TILING_CHECK(batchSplitFactor != 1,
                        CUBE_INNER_ERR_REPORT(args_.opName, "batchSplitFactor should be 1 when the scale is not null."),
                        return ge::GRAPH_FAILED);
    }
    if (bPermList != nullptr && bPermList->GetSize() == ALLOW_DIM) {
        const int64_t* bPerm = static_cast<const int64_t*>(bPermList->GetData());
        int64_t n = bShape[bPerm[N_IDX]];
        int64_t b = bShape[bPerm[BATCH_IDX]];
        OP_TILING_CHECK(shape_scale.GetDim(0) != b * n,
                        CUBE_INNER_ERR_REPORT(args_.opName,
                                              "The dimension of n mul b [%ld] and scale [%ld] tensors must be the same",
                                              b * n, shape_scale.GetDim(0)),
                        return ge::GRAPH_FAILED);
    }
    auto tensor_x1 = context_->GetInputDesc(0);
    auto tensor_x2 = context_->GetInputDesc(1);
    ge::DataType dtype_x1 = tensor_x1->GetDataType();
    ge::DataType dtype_x2 = tensor_x2->GetDataType();
    OP_TILING_CHECK(dtype_x2 != ge::DT_FLOAT16 || dtype_x1 != ge::DT_FLOAT16,
                    CUBE_INNER_ERR_REPORT(args_.opName,
                                          "the dtype of input is only supported FLOAT16 when the scale takes effect."),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TransposeBatchMatMulTiling::CheckArgs()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    size_t idx = 0;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(idx));
    idx++;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(idx));
    idx++;
    if (context_->GetOptionalInputShape(BIAS_IDX) != nullptr) {
        OP_LOGE(args_.opName, "bias is not supported now");
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape* storageShape_scale = context_->GetOptionalInputShape(SCALE_IDX);
    if (storageShape_scale != nullptr) {
        const gert::Shape shape_scale = storageShape_scale->GetOriginShape();
        OP_TILING_CHECK((CheckScale(shape_scale) != ge::GRAPH_SUCCESS),
                        CUBE_INNER_ERR_REPORT(args_.opName, "scale condition not satisfied"), return ge::GRAPH_FAILED);
    }
    if (attrs->GetAttrNum() >= HF32_ATTR_NUM) {
        OPS_CHECK_NULL_WITH_CONTEXT(context_,
                                    attrs->GetAttrPointer<int32_t>(HF32_ATTR_INDEX - 1)); // 检查倒数第2个属性
        OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(HF32_ATTR_INDEX));
    }
    if (attrs->GetAttrNum() >= ATTR_NUM) {
        OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<int32_t>(ATTR_NUM - 1));
    }
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(0));
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

    uint64_t batchA = static_cast<uint64_t>(aShape[BATCH_IDX]);
    uint64_t batchB = static_cast<uint64_t>(bShape[BATCH_IDX]);
    if (aPermList != nullptr && aPermList->GetSize() == ALLOW_DIM) {
        const int64_t *aPerm = static_cast<const int64_t *>(aPermList->GetData());
        batchA = aShape[aPerm[BATCH_IDX]];
    }
    if (bPermList != nullptr && bPermList->GetSize() == ALLOW_DIM) {
        const int64_t *bPerm = static_cast<const int64_t *>(bPermList->GetData());
        batchB = bShape[bPerm[BATCH_IDX]];
    }
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
}
}