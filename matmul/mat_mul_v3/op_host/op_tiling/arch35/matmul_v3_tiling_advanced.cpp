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
 * \file matmul_v3_tiling_advanced.cc
 * \brief Refactored implementation: each method has single responsibility
 *        (Extract / Validate / Detect). Old virtual methods delegate to
 *        the new clean methods for backward compatibility.
 */

#include "matmul_v3_tiling_advanced.h"

#include "register/op_def_registry.h"
#include "matmul/common/op_host/math_util.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include "./matmul_tiling_registry.h"
#include "./matmul_tiling_cfg.h"
#include "matmul_v3_compile_info_advanced.h"
#include "matmul_v3_tiling_strategy.h"
#include "matmul/common/op_host/log_format_util.h"

namespace {
using namespace optiling;
using namespace optiling::matmul_v3_advanced;

constexpr uint64_t ONE_BATCH_DIM = 1UL;
constexpr uint64_t TWO_BATCH_DIM = 2UL;
constexpr uint64_t THREE_BATCH_DIM = 3UL;
constexpr uint64_t FOUR_BATCH_DIM = 4UL;
constexpr size_t OFFSET_X_ATTR_NUM = 3UL;
constexpr size_t OFFSET_X_ATTR_INDEX = 2UL;
constexpr int64_t OFFSET_X_AVOID_TENSOR_API = 0x80;
constexpr size_t HF32_ATTR_NUM = 4UL;
constexpr size_t HF32_ATTR_INDEX = 3UL;
constexpr size_t OP_IMPL_MODE_ATTR_NUM = 4UL;
constexpr size_t OP_IMPL_MODE_ATTR_INDEX = 3UL;
constexpr size_t BIAS_IDX = 2UL;

ge::graphStatus InvalidDtypeErrorMsg(const MatMulV3Args& args)
{
    if (args.hasBias) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            args.opName, "a, b, c, bias",
            Ops::NN::FormatString("%s, %s, %s, %s", Ops::Base::ToString(args.aType).c_str(),
                                  Ops::Base::ToString(args.bType).c_str(), Ops::Base::ToString(args.cType).c_str(),
                                  Ops::Base::ToString(args.biasType).c_str())
                .c_str(),
            Ops::NN::FormatString(
                "The dtypes of %s must be the same and within the range %s or when the dtypes of %s is %s, the dtypes "
                "of "
                "%s must be %s",
                "a, b, c, bias", "{FLOAT16, FLOAT, BF16}", "a, b", "FLOAT16 or BF16", "c, bias", "FLOAT")
                .c_str());
        return ge::GRAPH_FAILED;
    } else {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            args.opName, "a, b, c",
            Ops::NN::FormatString("%s, %s, %s", Ops::Base::ToString(args.aType).c_str(),
                                  Ops::Base::ToString(args.bType).c_str(), Ops::Base::ToString(args.cType).c_str())
                .c_str(),
            Ops::NN::FormatString(
                "The dtypes of %s must be the same and within the range %s or when the dtypes of %s is %s, the dtype "
                "of "
                "%s must be %s",
                "a, b, c", "{FLOAT16, FLOAT, BF16}", "a, b", "FLOAT16 or BF16", "c", "FLOAT")
                .c_str());
        return ge::GRAPH_FAILED;
    }
}

ge::graphStatus InvalidDtypeErrorMsgForResv(const MatMulV3Args& args)
{
    if (args.hasBias) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            args.opName, "a, b, c, bias",
            Ops::NN::FormatString("%s, %s, %s, %s", Ops::Base::ToString(args.aType).c_str(),
                                  Ops::Base::ToString(args.bType).c_str(), Ops::Base::ToString(args.cType).c_str(),
                                  Ops::Base::ToString(args.biasType).c_str())
                .c_str(),
            Ops::NN::FormatString("The dtypes of %s must be the same and within the range %s ", "a, b, c, bias",
                                  "{FLOAT16}")
                .c_str());
        return ge::GRAPH_FAILED;
    } else {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            args.opName, "a, b, c",
            Ops::NN::FormatString("%s, %s, %s", Ops::Base::ToString(args.aType).c_str(),
                                  Ops::Base::ToString(args.bType).c_str(), Ops::Base::ToString(args.cType).c_str())
                .c_str(),
            Ops::NN::FormatString("The dtypes of %s must be the same and within the range %s ", "a, b, c", "{FLOAT16}")
                .c_str());
        return ge::GRAPH_FAILED;
    }
}
} // namespace

namespace optiling {
namespace matmul_v3_advanced {

// ====== Phase 1: Context initialization ======
ge::graphStatus MatMulV3Tiling::InitContext()
{
    args_.opName = context_->GetNodeName();
    OP_TILING_CHECK(args_.opName == nullptr, CUBE_INNER_ERR_REPORT("matmul", "get op name invalid context"),
                    return ge::GRAPH_FAILED);
    OP_LOGI(args_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    auto compileInfo = context_->GetCompileInfo();
    OP_TILING_CHECK(compileInfo == nullptr, CUBE_INNER_ERR_REPORT(args_.opName, "compileInfo is nullptr"),
                    return ge::GRAPH_FAILED);
    arch_ = reinterpret_cast<const MatmulV3CompileInfo*>(compileInfo)->npuArch;
    return ge::GRAPH_SUCCESS;
}

// ====== Phase 2: Input null-check ======
ge::graphStatus MatMulV3Tiling::ValidateInputsNotNull()
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
    if (attrs->GetAttrNum() >= HF32_ATTR_NUM) {
        OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<int64_t>(HF32_ATTR_INDEX - 1));
        OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(HF32_ATTR_INDEX));
    }
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(0));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(0));
    return ge::GRAPH_SUCCESS;
}

// ====== Phase 3: Optional input detection ======
void MatMulV3Tiling::DetectOptionalInputs()
{
    size_t idx = 2;
    if (context_->GetInputDesc(idx) != nullptr && context_->GetInputDesc(idx + 1) == nullptr) {
        args_.hasBias = true;
    }
}

// ====== Phase 4: Format extraction ======
void MatMulV3Tiling::ExtractFormat()
{
    ge::Format formatA = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(0)->GetStorageFormat()));
    ge::Format formatB = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(1)->GetStorageFormat()));
    ge::Format formatOut = static_cast<ge::Format>(
        ge::GetPrimaryFormat(context_->GetOutputDesc(0)->GetStorageFormat()));
    args_.aFormat = (formatA != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatA;
    args_.bFormat = (formatB != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatB;
    args_.outFormat = (formatOut != ge::FORMAT_FRACTAL_NZ) ? ge::FORMAT_ND : formatOut;
}

// ====== Phase 5: Dtype & attr-flag extraction ======
void MatMulV3Tiling::ExtractDtype()
{
    args_.aType = context_->GetInputDesc(0)->GetDataType();
    args_.bType = context_->GetInputDesc(1)->GetDataType();
    args_.cType = context_->GetOutputDesc(0)->GetDataType();
    if (args_.hasBias) {
        args_.biasType = context_->GetInputDesc(BIAS_IDX)->GetDataType();
    }
    args_.aDtypeSize = ge::GetSizeByDataType(args_.aType);
    args_.bDtypeSize = ge::GetSizeByDataType(args_.bType);
}

void MatMulV3Tiling::ExtractAttrFlags()
{
    // offsetX: 0x80 means avoid tensor API, unified across all op types
    if (context_->GetAttrs()->GetAttrNum() >= OFFSET_X_ATTR_NUM) {
        int64_t offsetX = *context_->GetAttrs()->GetAttrPointer<int64_t>(OFFSET_X_ATTR_INDEX);
        args_.isAvoidTensorApi = offsetX == OFFSET_X_AVOID_TENSOR_API;
    }
    // op_impl_mode_enum: 0x1: default 0x2: high_performance 0x4: high_precision 0x8: super_performance
    // 0x10: support_of_bound_index 0x20: enable_float_32_execution 0x40: enable_hi_float_32_execution
    if (strcmp(context_->GetNodeType(), "MatMulV3") == 0) {
        if (context_->GetAttrs()->GetAttrNum() >= OP_IMPL_MODE_ATTR_NUM) {
            args_.isHf32 = *context_->GetAttrs()->GetAttrPointer<int64_t>(OP_IMPL_MODE_ATTR_INDEX) == 0x40;
            args_.isForceGrpAccForFp32 = *context_->GetAttrs()->GetAttrPointer<int64_t>(OP_IMPL_MODE_ATTR_INDEX) == 0x4;
        }
    } else {
        if (context_->GetAttrs()->GetAttrNum() >= HF32_ATTR_NUM) {
            args_.isHf32 = *((context_->GetAttrs())->GetAttrPointer<bool>(HF32_ATTR_INDEX));
        }
    }
    OP_LOGD(args_.opName, "Hf32 flag is: %d, isAvoidTensorApi flag is: %d", args_.isHf32, args_.isAvoidTensorApi);
}

// ====== Phase 6: Shape extraction ======
ge::graphStatus MatMulV3Tiling::ExtractTranspose()
{
    args_.isATrans = *((context_->GetAttrs())->GetAttrPointer<bool>(0));
    args_.isBTrans = *((context_->GetAttrs())->GetAttrPointer<bool>(1));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatMulV3Tiling::ExtractMKN()
{
    int64_t mkDims[TWO_BATCH_DIM];
    int64_t knDims[TWO_BATCH_DIM];

    if (!ExtractNonContiguousDims(mkDims, knDims)) {
        return ge::GRAPH_FAILED;
    }

    uint64_t kIdxA = args_.isATrans ? 0ULL : 1ULL;
    uint64_t kIdxB = args_.isBTrans ? 1ULL : 0ULL;
    uint64_t mIdx = args_.isATrans ? 1ULL : 0ULL;
    uint64_t nIdx = args_.isBTrans ? 0ULL : 1ULL;
    args_.mValue = static_cast<uint64_t>(mkDims[mIdx]);
    args_.kValue = static_cast<uint64_t>(mkDims[kIdxA]);
    kBValue_ = knDims[kIdxB];
    args_.nValue = static_cast<uint64_t>(knDims[nIdx]);
    return ge::GRAPH_SUCCESS;
}

// ====== Phase 7: Validation ======
ge::graphStatus MatMulV3Tiling::ValidateFormat()
{
    // DAV_RESV芯片隔离：仅支持ND格式（a/b/c全部）
    if (arch_ == NpuArch::DAV_RESV &&
        (args_.aFormat != ge::FORMAT_ND || args_.bFormat != ge::FORMAT_ND || args_.outFormat != ge::FORMAT_ND)) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
            args_.opName, "a, b, c",
            Ops::NN::FormatString("%s, %s, %s", (args_.aFormat == ge::FORMAT_ND) ? "ND" : "FRACTAL_NZ",
                                  (args_.bFormat == ge::FORMAT_ND) ? "ND" : "FRACTAL_NZ",
                                  (args_.outFormat == ge::FORMAT_ND) ? "ND" : "FRACTAL_NZ")
                .c_str(),
            Ops::NN::FormatString("The formats of %s must be %s", "a, b, c", "ND").c_str());
        return ge::GRAPH_FAILED;
    }
    // 非DAV_RESV：a和out不能为FRACTAL_NZ
    if (arch_ != NpuArch::DAV_RESV &&
        (args_.aFormat == ge::FORMAT_FRACTAL_NZ || args_.outFormat == ge::FORMAT_FRACTAL_NZ)) {
        OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
            args_.opName, "a, c",
            Ops::NN::FormatString("%s, %s", (args_.aFormat == ge::FORMAT_FRACTAL_NZ) ? "FRACTAL_NZ" : "ND",
                                  (args_.outFormat == ge::FORMAT_FRACTAL_NZ) ? "FRACTAL_NZ" : "ND")
                .c_str(),
            Ops::NN::FormatString("The formats of %s must be %s", "a, c", "ND").c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatMulV3Tiling::ValidateShape()
{
    if (static_cast<int64_t>(args_.kValue) != kBValue_) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            args_.opName, "a, b",
            Ops::NN::FormatString("%s, %s", Ops::Base::ToString(context_->GetInputShape(0)->GetOriginShape()).c_str(),
                                  Ops::Base::ToString(context_->GetInputShape(1)->GetOriginShape()).c_str())
                .c_str(),
            Ops::NN::FormatString("%s of %s must be equal", "K-axis", "a, b").c_str());
        return ge::GRAPH_FAILED;
    }
    if (args_.kValue == 0UL && args_.hasBias) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            args_.opName, "a", Ops::Base::ToString(context_->GetInputShape(0)->GetOriginShape()).c_str(),
            Ops::NN::FormatString("When optional parameter %s exists, %s of %s must be a positive number", "bias",
                                  "k-axis", "a")
                .c_str());
        return ge::GRAPH_FAILED;
    }
    auto isValidDimValue = [](int64_t dim) -> bool { return (dim > 0) && (dim <= INT32_MAX); };
    auto isValidDimValueK = [](int64_t dim) -> bool { return (dim >= 0) && (dim <= INT32_MAX); };
    if (!isValidDimValue(args_.mValue) || !isValidDimValueK(args_.kValue) || !isValidDimValue(args_.nValue)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            args_.opName, "a, b",
            Ops::NN::FormatString("%s, %s", Ops::Base::ToString(context_->GetInputShape(0)->GetOriginShape()).c_str(),
                                  Ops::Base::ToString(context_->GetInputShape(1)->GetOriginShape()).c_str())
                .c_str(),
            Ops::NN::FormatString("%s of %s must be within the range %s, %s of %s must be within the range %s", "m, n",
                                  "a, b", "(0, INT32_MAX]", "k", "a, b", "[0, INT32_MAX]")
                .c_str());
        return ge::GRAPH_FAILED;
    }
    const gert::Shape& cShape = context_->GetOutputShape(0)->GetOriginShape();
    const size_t cDimNum = cShape.GetDimNum();
    if (cDimNum < TWO_BATCH_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            args_.opName, "c", Ops::NN::FormatString("%zu", cDimNum).c_str(),
            Ops::NN::FormatString("The shape dim of %s must be at least %llu", "c", TWO_BATCH_DIM).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatMulV3Tiling::ValidateBias()
{
    if (!args_.hasBias) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::Shape& biasShape = context_->GetInputShape(BIAS_IDX)->GetOriginShape();
    const gert::Shape& cShape = context_->GetOutputShape(0)->GetOriginShape();
    const int64_t biasValue = biasShape[biasShape.GetDimNum() - 1];
    const int64_t nOriValue = cShape[cShape.GetDimNum() - 1];
    if (biasValue != nOriValue) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(args_.opName, "bias", Ops::Base::ToString(biasShape).c_str(),
                                              Ops::NN::FormatString("%s of %s must be equal to %s of %s (%ld)",
                                                                    "Shape[-1]", "bias", "Shape[-1]", "c", nOriValue)
                                                  .c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MatMulV3Tiling::ValidateDtype()
{
    std::vector<ge::DataType> dtype = {args_.aType, args_.bType, args_.cType};
    if (args_.hasBias) {
        dtype.push_back(args_.biasType);
    }
    // DAV_RESV架构：仅支持FLOAT16
    if (arch_ == NpuArch::DAV_RESV) {
        const std::vector<std::vector<ge::DataType>> dtypeSuportListForResv = {
            {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}};
        for (auto& supported : dtypeSuportListForResv) {
            if (std::equal(dtype.begin(), dtype.end(), supported.begin())) {
                return ge::GRAPH_SUCCESS;
            }
        }
        return InvalidDtypeErrorMsgForResv(args_);
    }
    // 其他架构：使用GetDtypeSupportList()
    auto supportList = GetDtypeSupportList();
    for (auto& supported : supportList) {
        if (std::equal(dtype.begin(), dtype.end(), supported.begin())) {
            return ge::GRAPH_SUCCESS;
        }
    }
    return InvalidDtypeErrorMsg(args_);
}

ge::graphStatus MatMulV3Tiling::ValidateOpSpecific()
{
    const bool isMatMulV3 = (strcmp(context_->GetNodeType(), "MatMulV3") == 0);
    const bool isBatchMatMulV3 = (strcmp(context_->GetNodeType(), "BatchMatMulV3") == 0);
    if (!isBatchMatMulV3 && !isMatMulV3) {
        return ge::GRAPH_SUCCESS;
    }

    if (isMatMulV3) {
        auto isMatrix = [this](const gert::Shape& oriShape) {
            if (isSelfSlice_) {
                return oriShape.GetDimNum() == TWO_BATCH_DIM || oriShape.GetDimNum() == THREE_BATCH_DIM;
            } else {
                return oriShape.GetDimNum() == TWO_BATCH_DIM;
            }
        };
        if (!isMatrix(context_->GetInputShape(0)->GetOriginShape()) ||
            !isMatrix(context_->GetInputShape(1)->GetOriginShape())) {
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                args_.opName, "a, b",
                Ops::NN::FormatString("%zu, %zu", context_->GetInputShape(0)->GetOriginShape().GetDimNum(),
                                      context_->GetInputShape(1)->GetOriginShape().GetDimNum())
                    .c_str(),
                Ops::NN::FormatString("The shape dims of %s must all be %s", "a, b", "2 or 3").c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// ====== Phase 10: Registry delegation hooks ======
std::vector<int32_t> MatMulV3Tiling::GetRegistryPriorities(NpuArch npuArch) const
{
    return strategy::GetMatMulV3Priorities(npuArch);
}

// ====== Dtype support list ======
std::vector<std::vector<ge::DataType>> MatMulV3Tiling::GetDtypeSupportList() const
{
    return {// x1,              x2,             y,              bias
            {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16},
            {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT},
            {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT16},
            {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT},
            {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT},
            {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT},
            {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::DT_BF16},
            {ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16},
            {ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_FLOAT}};
}

// ====== DoTiling: orchestrates all phases ======
ge::graphStatus MatMulV3Tiling::DoTiling()
{
    if (GetShapeAttrsInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ExtractBatchInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (PostBatchInfoCheck() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    MatMulTilingCfg tilingCfg(false, context_->GetCompileInfo(), reinterpret_cast<void*>(&args_), GetTilingKeyObj());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, tilingCfg.compileInfo);
    NpuArch npuArch = reinterpret_cast<const MatmulV3CompileInfo*>(tilingCfg.compileInfo)->npuArch;
    MMRegisterCfg registerCfg{GetRegistryOpType(), npuArch, GetRegistryPriorities(npuArch)};
    return MMTilingRegistry::GetInstance().DoTilingImpl(context_, tilingCfg, registerCfg);
}

// ====== Old interface: GetShapeAttrsInfo (delegates to new phases) ======
ge::graphStatus MatMulV3Tiling::GetShapeAttrsInfo()
{
    if (InitContext() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_TILING_CHECK((CheckArgs() != ge::GRAPH_SUCCESS) || (GetArgs() != ge::GRAPH_SUCCESS),
                    CUBE_INNER_ERR_REPORT(args_.opName, "invalid context"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ====== Old interface: CheckArgs (delegates to Validate + Detect) ======
ge::graphStatus MatMulV3Tiling::CheckArgs()
{
    if (ValidateInputsNotNull() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    DetectOptionalInputs();
    return ge::GRAPH_SUCCESS;
}

// ====== Old interface: GetArgs (delegates to Extract phases + old GetShape/BaseOpSpecificCheck) ======
ge::graphStatus MatMulV3Tiling::GetArgs()
{
    ExtractFormat();
    ExtractDtype();
    ExtractAttrFlags();
    if (GetShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return BaseOpSpecificCheck();
}

// ====== Old interface: GetFormat (delegates to ExtractFormat) ======
void MatMulV3Tiling::GetFormat() { ExtractFormat(); }

// ====== Old interface: GetShape (delegates to Extract + Validate phases) ======
ge::graphStatus MatMulV3Tiling::GetShape()
{
    if (ExtractTranspose() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidateFormat() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ExtractMKN() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ValidateShape();
}

// ====== Old interface: BaseOpSpecificCheck (delegates to Validate phases) ======
ge::graphStatus MatMulV3Tiling::BaseOpSpecificCheck()
{
    if (ValidateOpSpecific() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidateBias() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ValidateDtype();
}

// ====== Private: slice dims extraction ======
ge::graphStatus MatMulV3Tiling::ExtractSliceDims(int64_t (&dims)[TWO_BATCH_DIM])
{
    auto selfShape = context_->GetInputShape(0)->GetOriginShape();
    if (args_.isATrans) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(args_.opName, "transposeA",
                                              Ops::NN::FormatString("%s", args_.isATrans ? "true" : "false").c_str(),
                                              Ops::NN::FormatString("In %s case, the value of %s cannot be %s",
                                                                    "non-contiguous a-slice", "transposeA", "true")
                                                  .c_str());
        return ge::GRAPH_FAILED;
    }
    dims[0] = selfShape[0] * selfShape[1]; // m = batch * sliceM
    dims[1] = selfShape[2];                // k = viewShape[2]
    isSelfSlice_ = true;
    return ge::GRAPH_SUCCESS;
}

// ====== Private: transpose dims extraction ======
ge::graphStatus MatMulV3Tiling::ExtractTransposeDims(int64_t (&dims)[TWO_BATCH_DIM], int64_t idx)
{
    auto inputViewShape = context_->GetInputShape(idx)->GetOriginShape();
    const size_t oriDimNum = inputViewShape.GetDimNum();
    const char* paramName = (idx == 0) ? "a" : "b";
    if (oriDimNum != THREE_BATCH_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            args_.opName, paramName, Ops::NN::FormatString("%zu", oriDimNum).c_str(),
            Ops::NN::FormatString("In %s scene, the shape dim of %s must be %llu", "non-contiguous transpose",
                                  paramName, THREE_BATCH_DIM)
                .c_str());
        return ge::GRAPH_FAILED;
    }
    dims[0] = inputViewShape[oriDimNum - TWO_BATCH_DIM];
    dims[1] = inputViewShape[oriDimNum - ONE_BATCH_DIM];
    return ge::GRAPH_SUCCESS;
}

// ====== Private: non-contiguous dims extraction routing ======
bool MatMulV3Tiling::ExtractNonContiguousDims(int64_t (&mkDims)[TWO_BATCH_DIM], int64_t (&knDims)[TWO_BATCH_DIM])
{
    auto selfShape = context_->GetInputShape(0)->GetOriginShape();
    auto mat2Shape = context_->GetInputShape(1)->GetOriginShape();
    auto selfStorageShape = context_->GetInputShape(0)->GetStorageShape();
    auto mat2StorageShape = context_->GetInputShape(1)->GetStorageShape();
    size_t selfDimNum = selfShape.GetDimNum();
    size_t mat2DimNum = mat2Shape.GetDimNum();
    bool isANonContiguous = context_->InputIsView(0) && (selfStorageShape.GetDimNum() == 1) &&
                            (selfDimNum == THREE_BATCH_DIM);
    bool isBTransposeNonContiguous = context_->InputIsView(1) && (mat2StorageShape.GetDimNum() == 1) &&
                                     (mat2DimNum == THREE_BATCH_DIM);
    bool isASliceNonContiguous = isANonContiguous && mat2DimNum == 2;
    bool isATransposeNonContiguous = isANonContiguous && mat2DimNum == 3;
    if (isASliceNonContiguous) {
        if (ExtractSliceDims(mkDims) != ge::GRAPH_SUCCESS) {
            return false;
        }
    } else if (isATransposeNonContiguous) {
        if (ExtractTransposeDims(mkDims, 0) != ge::GRAPH_SUCCESS) {
            return false;
        }
    } else {
        if (ExtractNormalDims(selfStorageShape, selfShape, args_.aDtypeSize, args_.aFormat, mkDims, "a") !=
            ge::GRAPH_SUCCESS) {
            OP_LOGE(args_.opName, "invalid input dim num for a");
            return false;
        }
    }

    if (isBTransposeNonContiguous) {
        if (ExtractTransposeDims(knDims, 1) != ge::GRAPH_SUCCESS) {
            return false;
        }
    } else {
        if (ExtractNormalDims(mat2StorageShape, mat2Shape, args_.bDtypeSize, args_.bFormat, knDims, "b") !=
            ge::GRAPH_SUCCESS) {
            OP_LOGE(args_.opName, "invalid input dim num for b");
            return false;
        }
    }
    return true;
}

// ====== Private: normal dims extraction (was free function GetInputDims) ======
ge::graphStatus MatMulV3Tiling::ExtractNormalDims(const gert::Shape& storageShape, const gert::Shape& oriShape,
                                                  uint64_t dtypeSize, ge::Format format, int64_t (&dims)[TWO_BATCH_DIM],
                                                  const char* paramName)
{
    const size_t dimNum = storageShape.GetDimNum();
    const size_t oriDimNum = oriShape.GetDimNum();
    dims[0] = oriShape[oriDimNum - TWO_BATCH_DIM];
    dims[1] = oriShape[oriDimNum - ONE_BATCH_DIM];
    if (format == ge::FORMAT_ND) {
        if (dimNum < TWO_BATCH_DIM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                "MatMulV3", paramName, Ops::NN::FormatString("%zu", dimNum).c_str(),
                Ops::NN::FormatString("When the format of %s is %s, the shape dim of %s must be at least %llu",
                                      paramName, "ND", paramName, TWO_BATCH_DIM)
                    .c_str());
            return ge::GRAPH_FAILED;
        }
    } else {
        if (dimNum < FOUR_BATCH_DIM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                "MatMulV3", paramName, Ops::NN::FormatString("%zu", dimNum).c_str(),
                Ops::NN::FormatString("When the format of %s is %s, the shape dim of %s must be at least %llu",
                                      paramName, "FRACTAL_NZ", paramName, FOUR_BATCH_DIM)
                    .c_str());
            return ge::GRAPH_FAILED;
        }
        int64_t storageShape0 = storageShape[dimNum - THREE_BATCH_DIM] * storageShape[dimNum - TWO_BATCH_DIM];
        int64_t storageShape1 = storageShape[dimNum - FOUR_BATCH_DIM] * storageShape[dimNum - ONE_BATCH_DIM];
        if (ops::CeilAlign(dims[0], static_cast<int64_t>(BASIC_BLOCK_SIZE_16)) != storageShape0 ||
            ops::CeilAlign(dims[1], static_cast<int64_t>(BLOCK_BYTE_SIZE / dtypeSize)) != storageShape1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                "MatMulV3", paramName, Ops::NN::FormatString("%s", Ops::Base::ToString(oriShape).c_str()).c_str(),
                Ops::NN::FormatString("The NZ aligned %s of %s must be equal to %s of %s", "oriShape", paramName,
                                      "storageShape", paramName)
                    .c_str());
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace matmul_v3_advanced
} // namespace optiling
