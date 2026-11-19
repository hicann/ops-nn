/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gemm_syrk_tiling.h"

#include <cstdint>
#include <string>
#include <vector>

#include "base/err_msg.h"
#include "gemm_syrk_tiling_key.h"
#include "gemm_syrk_tiling_strategy.h"
#include "error_util.h"
#include "matmul/mat_mul_v3/op_host/op_tiling/arch35/matmul_tiling_registry.h"

namespace {
static constexpr uint32_t INDEX_A = 0;
static constexpr uint32_t INDEX_C = 1;
static constexpr uint32_t INDEX_ATTR_ALPHA = 0;
static constexpr uint32_t INDEX_ATTR_BETA = 1;
static constexpr uint32_t INDEX_ATTR_TRANSPOSE_X = 2;
static constexpr uint32_t INDEX_ATTR_FILL_MODE = 3;
static constexpr uint32_t INDEX_ATTR_OP_IMPL_MODE = 4;
static constexpr int64_t OP_IMPL_MODE_HF32 = 0x40;
static constexpr uint64_t NUM_TWO = 2UL;
static constexpr uint64_t INT32_MAX_LIMIT = static_cast<uint64_t>(INT32_MAX);
static constexpr size_t MIN_SHAPE_DIM = 2;
static constexpr size_t MAX_SHAPE_DIM = 6;
static const char* FILL_MODE_FULL = "full";
static const char* SYRK_OP_NAME = "GemmSyrk";

// needUpdate = true: the ASW basic template hands its TilingResult back
// through Update() instead of writing the tiling context.
class GemmSyrkTilingResultCfg : public optiling::MatMulTilingCfg {
public:
    GemmSyrkTilingResultCfg(const void* compileInfo, const void* args) : MatMulTilingCfg(true, compileInfo, args) {}

    ge::graphStatus Update(const optiling::TilingResult& result) override
    {
        result_ = result;
        return ge::GRAPH_SUCCESS;
    }

    const optiling::TilingResult& GetResult() const { return result_; }

private:
    optiling::TilingResult result_{};
};
} // namespace

namespace optiling {
namespace gemm_syrk {
ge::graphStatus GemmSyrkTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "platformInfo is null"),
                    return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    auto socVersion = ascendcPlatform.GetSocVersion();
    if (socVersion != platform_ascendc::SocVersion::ASCEND950) {
        const char* nodeName = context_->GetNodeName();
        const std::string opName = (nodeName != nullptr) ? nodeName : SYRK_OP_NAME;
        OP_LOGE_WITHOUT_REPORT(opName.c_str(),
                               "GemmSyrk only supports Ascend 950 / 350 (DAV_3510), current socVersion is %d",
                               static_cast<int32_t>(socVersion));
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1009", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{opName.c_str(), "only Ascend 950 / 350 (DAV_3510) is supported"}));
        return ge::GRAPH_FAILED;
    }
    // Feed the matmul compile info consumed by the basic tiling strategy.
    mmCompileInfo_ = {};
    mmCompileInfo_.aicNum = ascendcPlatform.GetCoreNumAic();
    mmCompileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();
    mmCompileInfo_.npuArch = ascendcPlatform.GetCurNpuArch();
    mmCompileInfo_.socVersion = socVersion;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, mmCompileInfo_.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, mmCompileInfo_.l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, mmCompileInfo_.l0BSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, mmCompileInfo_.l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, mmCompileInfo_.l2Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, mmCompileInfo_.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::BT, mmCompileInfo_.btSize);
    if (mmCompileInfo_.aivNum != mmCompileInfo_.aicNum * NUM_TWO) {
        const char* nodeName = context_->GetNodeName();
        const std::string opName = (nodeName != nullptr) ? nodeName : SYRK_OP_NAME;
        OP_LOGE_WITHOUT_REPORT(opName.c_str(), "GemmSyrk requires aicNum:aivNum = 1:2 (MIX), aicNum=%lu, aivNum=%lu",
                               mmCompileInfo_.aicNum, mmCompileInfo_.aivNum);
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1009", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{opName.c_str(), "the kernel requires the MIX aicNum:aivNum = 1:2 ratio"}));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ====== Extract phases ======

ge::graphStatus GemmSyrkTiling::ValidateInputsNotNull()
{
    // GetInputDesc/GetInputShape/GetOutputDesc return nullptr when the index is
    // invalid or the compute node info is missing; dereferencing them below
    // would crash the tiling process.
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(INDEX_A));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(INDEX_C));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INDEX_A));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INDEX_C));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(0));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GemmSyrkTiling::ExtractAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "attrs is null"),
                    return ge::GRAPH_FAILED);

    const auto* alpha = attrs->GetAttrPointer<float>(INDEX_ATTR_ALPHA);
    alpha_ = (alpha != nullptr) ? *alpha : 1.0F;
    const auto* beta = attrs->GetAttrPointer<float>(INDEX_ATTR_BETA);
    beta_ = (beta != nullptr) ? *beta : 1.0F;

    const auto* transX = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANSPOSE_X);
    transX_ = (transX != nullptr) ? *transX : false;

    const char* fillMode = attrs->GetAttrPointer<char>(INDEX_ATTR_FILL_MODE);
    fillMode_ = (fillMode != nullptr) ? std::string(fillMode) : std::string(FILL_MODE_FULL);

    const auto* opImplMode = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_OP_IMPL_MODE);
    opImplMode_ = (opImplMode != nullptr) ? *opImplMode : 0x1L;

    aType_ = context_->GetInputDesc(INDEX_A)->GetDataType();
    dtypeSize_ = ge::GetSizeByDataType(aType_);
    // FP32 inputs only support the HF32 (TF32-mantissa) cube mode; the
    // op_impl_mode 0x40 bit must be set explicitly. The 16-bit element types
    // keep the full-precision cube path.
    isHf32_ = (aType_ == ge::DT_FLOAT && (opImplMode_ & OP_IMPL_MODE_HF32) != 0) ? 1U : 0U;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GemmSyrkTiling::ExtractShape()
{
    const gert::Shape& aShape = context_->GetInputShape(INDEX_A)->GetOriginShape();
    const gert::Shape& cShape = context_->GetInputShape(INDEX_C)->GetOriginShape();
    const size_t aDimNum = aShape.GetDimNum();
    const size_t cDimNum = cShape.GetDimNum();
    OP_TILING_CHECK(aDimNum < MIN_SHAPE_DIM || aDimNum > MAX_SHAPE_DIM || aDimNum != cDimNum,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context_->GetNodeName(), "a, c",
                        std::string("a=") + Ops::Base::ToString(aShape) + ", c=" + Ops::Base::ToString(cShape),
                        "the rank of a and c must be identical and within [2, 6]"),
                    return ge::GRAPH_FAILED);
    // transpose_x = false: a is [..., m, k]; transpose_x = true: a is the
    // transposed [..., k, m] storage, so m is a's last dim and k its second
    // to last. Syrk: N == M, read from the square c for consistency.
    if (transX_) {
        k_ = static_cast<uint64_t>(aShape[aDimNum - NUM_TWO]);
        m_ = static_cast<uint64_t>(aShape[aDimNum - 1]);
    } else {
        m_ = static_cast<uint64_t>(aShape[aDimNum - NUM_TWO]);
        k_ = static_cast<uint64_t>(aShape[aDimNum - 1]);
    }
    n_ = static_cast<uint64_t>(cShape[cDimNum - 1]);
    batch_ = 1UL;
    for (size_t i = 0; i + NUM_TWO < aDimNum; ++i) {
        const int64_t batchDim = aShape.GetDim(i);
        OP_TILING_CHECK(
            batchDim < 0 || static_cast<uint64_t>(batchDim) > INT32_MAX_LIMIT,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context_->GetNodeName(), "a", Ops::Base::ToString(aShape),
                                                  "the batch axes of a must be non-negative and within [0, INT32_MAX]"),
            return ge::GRAPH_FAILED);
        const uint64_t dimValue = static_cast<uint64_t>(batchDim);
        OP_TILING_CHECK(dimValue != 0UL && batch_ > INT32_MAX_LIMIT / dimValue,
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                            context_->GetNodeName(), "a, c",
                            std::string("a=") + Ops::Base::ToString(aShape) + ", c=" + Ops::Base::ToString(cShape),
                            "the product of the batch axes must be within [0, INT32_MAX]"),
                        return ge::GRAPH_FAILED);
        batch_ *= dimValue;
    }
    return ge::GRAPH_SUCCESS;
}

// ====== Validate phases ======

ge::graphStatus GemmSyrkTiling::ValidateDtype() const
{
    ge::DataType cType = context_->GetInputDesc(INDEX_C)->GetDataType();
    ge::DataType outType = context_->GetOutputDesc(0)->GetDataType();
    const std::string dtypeText = std::string("a=") + Ops::Base::ToString(aType_) +
                                  ", c=" + Ops::Base::ToString(cType) + ", out=" + Ops::Base::ToString(outType);
    OP_TILING_CHECK((aType_ != ge::DT_FLOAT16 && aType_ != ge::DT_BF16 && aType_ != ge::DT_FLOAT) || cType != aType_ ||
                        outType != aType_,
                    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                        context_->GetNodeName(), "a, c, out", dtypeText,
                        "the dtypes of a, c and out must be identical and be FLOAT16, BF16 or FLOAT (HF32)"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(aType_ == ge::DT_FLOAT && (opImplMode_ & OP_IMPL_MODE_HF32) == 0,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "op_impl_mode", std::to_string(opImplMode_),
                                              "0x40 (HF32 enabled)"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GemmSyrkTiling::ValidateFormat() const
{
    auto aFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(INDEX_A)->GetStorageFormat()));
    auto cFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetInputDesc(INDEX_C)->GetStorageFormat()));
    auto outFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->GetOutputDesc(0)->GetStorageFormat()));
    const std::string formatText = std::string("a=") + Ops::Base::ToString(aFormat) +
                                   ", c=" + Ops::Base::ToString(cFormat) + ", out=" + Ops::Base::ToString(outFormat);
    OP_TILING_CHECK(aFormat != ge::FORMAT_ND || cFormat != ge::FORMAT_ND || outFormat != ge::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(context_->GetNodeName(), "a, c, out", formatText,
                                                            "only ND format is supported"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GemmSyrkTiling::ValidateShape() const
{
    const gert::Shape& aShape = context_->GetInputShape(INDEX_A)->GetOriginShape();
    const gert::Shape& cShape = context_->GetInputShape(INDEX_C)->GetOriginShape();
    const size_t aDimNum = aShape.GetDimNum();
    // c must be square and consistent with a: a is [..., m, k] (or the
    // transposed [..., k, m] with transpose_x), c is [..., m, m]. The batch
    // axis must match (in-place update does not support broadcast).
    const std::string shapeText = std::string("a=") + Ops::Base::ToString(aShape) +
                                  ", c=" + Ops::Base::ToString(cShape);
    OP_TILING_CHECK(static_cast<uint64_t>(cShape[aDimNum - NUM_TWO]) != m_ ||
                        static_cast<uint64_t>(cShape[aDimNum - 1]) != m_ || n_ != m_,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context_->GetNodeName(), "a, c", shapeText,
                        "a must be [..., m, k] (or [..., k, m] with transpose_x) and c must be the square "
                        "matrix [..., m, m]"),
                    return ge::GRAPH_FAILED);
    for (size_t i = 0; i + NUM_TWO < aDimNum; ++i) {
        OP_TILING_CHECK(aShape.GetDim(i) != cShape.GetDim(i),
                        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                            context_->GetNodeName(), "a, c", shapeText,
                            "the batch axes of a and c must be identical, in-place update does not support broadcast"),
                        return ge::GRAPH_FAILED);
    }
    auto isValidDimValue = [](uint64_t dim) -> bool { return dim <= INT32_MAX_LIMIT; };
    // The batch value is the flattened product of all leading axes (up to 4
    // axes with 6D inputs) and is narrowed into the uint32 GemmSyrkTilingData
    // field, so it needs its own upper bound next to the per-dim checks.
    OP_TILING_CHECK(
        !isValidDimValue(m_) || !isValidDimValue(k_) || !isValidDimValue(n_) || !isValidDimValue(batch_),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "m, k, n, batch",
                                               std::string("m=") + std::to_string(m_) + ", k=" + std::to_string(k_) +
                                                   ", n=" + std::to_string(n_) + ", batch=" + std::to_string(batch_),
                                               "must be within the range [0, INT32_MAX]"),
        return ge::GRAPH_FAILED);
    // The Blaze syrk path accumulates over k; k == 0 never reaches this kernel
    // via aclnnGemmSyrk (routed to an elementwise beta * C scaling), and the
    // empty-tensor no-op path in DoTiling tolerates any k, so k == 0 is only
    // rejected when the kernel would actually run.
    OP_TILING_CHECK(k_ == 0UL && m_ > 0UL && batch_ > 0UL,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "k", std::to_string(k_),
                                              "a positive number for the GemmSyrk kernel"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GemmSyrkTiling::ValidateAttrs() const
{
    // fill_mode is declared as "full"/"up"/"low" in the prototype; only the
    // complete-symmetric-matrix "full" mode is implemented by this kernel.
    OP_TILING_CHECK(fillMode_ != FILL_MODE_FULL,
                    OP_LOGE_FOR_INVALID_VALUE(context_->GetNodeName(), "fill_mode", fillMode_,
                                              "\"full\" (the \"up\"/\"low\" modes are not implemented)"),
                    return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ====== GetShapeAttrsInfo: orchestrates extract + validate phases ======

ge::graphStatus GemmSyrkTiling::GetShapeAttrsInfo()
{
    if (ValidateInputsNotNull() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ExtractAttrs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ExtractShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidateDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidateFormat() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidateShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidateAttrs() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ====== Tiling phases ======

void GemmSyrkTiling::BuildMatmulArgs()
{
    // Map the syrk problem onto the matmul arg model: n == m, the B operand
    // mirrors a (the kernel reuses a's address as the transposed B), the
    // layout (op(A) = A, op(B) = A^T) is resolved inside the kernel, and the
    // in-place addend c always carries the full batch.
    mmArgs_ = {};
    mmArgs_.opName = SYRK_OP_NAME;
    mmArgs_.aType = aType_;
    mmArgs_.bType = aType_; // B == A
    mmArgs_.cType = context_->GetOutputDesc(0)->GetDataType();
    mmArgs_.x3Type = context_->GetInputDesc(INDEX_C)->GetDataType();
    mmArgs_.aDtypeSize = dtypeSize_;
    mmArgs_.bDtypeSize = dtypeSize_;
    mmArgs_.mValue = m_;
    mmArgs_.kValue = k_;
    mmArgs_.nValue = n_;
    mmArgs_.isATrans = false;
    mmArgs_.isBTrans = false;
    mmArgs_.hasBias = false;
    mmArgs_.batchX3 = batch_;
    mmBatchInfo_ = {};
    mmBatchInfo_.batchA3 = batch_;
    mmBatchInfo_.batchB3 = batch_; // B == A
    mmBatchInfo_.batchC3 = batch_;
    mmBatchInfo_.batchA = batch_;
    mmBatchInfo_.batchB = batch_;
    mmBatchInfo_.batchC = batch_;
    mmArgs_.batchInfo = &mmBatchInfo_;
}

ge::graphStatus GemmSyrkTiling::RunBasicTilingByStrategy()
{
    // Dispatch through the tiling strategy priorities to the registered
    // GemmSyrkBaseTiling, whose DoTiling runs IsCapable (MIX ratio plus the
    // mirrored batch/type constraints) and DoOpTiling (the BatchMatMulV3 ASW
    // basic computation followed by the syrk symmetric square clamps).
    // needUpdate = true hands the TilingResult back through Update() instead
    // of writing the tiling context.
    GemmSyrkTilingResultCfg cfg(&mmCompileInfo_, &mmArgs_);
    MMRegisterCfg registerCfg{SYRK_OP_NAME, mmCompileInfo_.npuArch,
                              strategy::GetGemmSyrkPriorities(mmCompileInfo_.npuArch)};
    const ge::graphStatus ret = MMTilingRegistry::GetInstance().DoTilingImpl(context_, cfg, registerCfg);
    if (ret != ge::GRAPH_SUCCESS) {
        const char* nodeName = context_->GetNodeName();
        const std::string opName = (nodeName != nullptr) ? nodeName : SYRK_OP_NAME;
        OP_LOGE_WITHOUT_REPORT(opName.c_str(), "the GemmSyrk basic tiling strategy failed, ret=%d",
                               static_cast<int32_t>(ret));
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{opName.c_str(), "the GemmSyrk basic tiling strategy failed"}));
        return ge::GRAPH_FAILED;
    }
    const TilingResult& result = cfg.GetResult();
    OP_TILING_CHECK(result.tilingData == nullptr || result.tilingDataSize != sizeof(BatchMatMulV3BasicTilingData),
                    CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "unexpected basic tiling data size"),
                    return ge::GRAPH_FAILED);
    bmmTiling_ = *static_cast<const BatchMatMulV3BasicTilingData*>(result.tilingData.get());
    return ge::GRAPH_SUCCESS;
}

void GemmSyrkTiling::SetTilingData(const BatchMatMulV3BasicTilingData& bmmTiling)
{
    // The clamped ASW result satisfies the symmetric square contract
    // (baseM == baseN == mL1 == nL1), so the flat struct carries one size.
    tilingData_.m = static_cast<uint32_t>(m_);
    tilingData_.n = static_cast<uint32_t>(n_);
    tilingData_.k = static_cast<uint32_t>(k_);
    tilingData_.batch = static_cast<uint32_t>(batch_);
    tilingData_.baseBlock = bmmTiling.matMulTilingData.baseM;
    tilingData_.baseK = bmmTiling.matMulTilingData.baseK;
    tilingData_.kL1 = bmmTiling.matMulTilingData.kL1;
    tilingData_.usedCoreNum = bmmTiling.matMulTilingData.usedCoreNum;
    tilingData_.isHf32 = isHf32_;
    tilingData_.alpha = alpha_;
    tilingData_.beta = beta_;
}

ge::graphStatus GemmSyrkTiling::DoTiling()
{
    // Orchestrator: extract/validate the syrk inputs, map them onto the
    // matmul arg model, dispatch through the tiling strategy to the basic
    // tiling (IsCapable + the ASW basic computation + syrk clamps run
    // there), then persist the flat tiling data, tiling key and block dim.
    OP_TILING_CHECK(GetShapeAttrsInfo() != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "GetShapeAttrsInfo failed"),
                    return ge::GRAPH_FAILED);
    OP_TILING_CHECK(GetPlatformInfo() != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "GetPlatformInfo failed"), return ge::GRAPH_FAILED);
    // 空张量 no-op：m == 0 或 batch == 0 时输出无元素（与 aclnnGemmSyrk 的空张量
    // 路径一致），上三角槽位数为 0，kernel 各核在访问任何 GM 前直接返回。tiling
    // 数据退化为 m == n == 0 的安全几何 —— batch == 0 时调度器会把 batch 钳位到
    // 1，不能依赖 batch 本身让槽位数归零。ASW 基础 tiling 的成本模型不定义
    // m/batch 为 0 的行为，一并跳过。
    if (m_ == 0UL || batch_ == 0UL) {
        tilingData_ = {};
        tilingData_.k = static_cast<uint32_t>(k_);
        tilingData_.batch = static_cast<uint32_t>(batch_);
        tilingData_.baseBlock = static_cast<uint32_t>(BASIC_BLOCK_SIZE_16);
        tilingData_.baseK = static_cast<uint32_t>(BASIC_BLOCK_SIZE_16);
        tilingData_.kL1 = static_cast<uint32_t>(BASIC_BLOCK_SIZE_16);
        tilingData_.usedCoreNum = 1U;
        tilingData_.isHf32 = isHf32_;
        tilingData_.alpha = alpha_;
        tilingData_.beta = beta_;
        context_->SetTilingKey(GetTilingKey());
        return PostTiling();
    }
    BuildMatmulArgs();
    OP_TILING_CHECK(RunBasicTilingByStrategy() != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "RunBasicTilingByStrategy failed"),
                    return ge::GRAPH_FAILED);
    SetTilingData(bmmTiling_);
    context_->SetTilingKey(GetTilingKey());
    return PostTiling();
}

ge::graphStatus GemmSyrkTiling::PostTiling()
{
    const size_t sizeTilingData = sizeof(GemmSyrkTilingData);
    OP_TILING_CHECK(sizeTilingData % sizeof(uint64_t) != 0,
                    OP_LOGE(context_->GetNodeName(), "tiling data size[%zu] is not aligned to 8", sizeTilingData),
                    return ge::GRAPH_FAILED);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    context_->GetRawTilingData()->SetDataSize(sizeTilingData);
    context_->SetBlockDim(tilingData_.usedCoreNum);

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_TILING_CHECK(workspaces == nullptr, CUBE_INNER_ERR_REPORT(context_->GetNodeName(), "workspaces is null"),
                    return ge::GRAPH_FAILED);
    workspaces[0] = 0;

    auto tilingPtr = static_cast<GemmSyrkTilingData*>(context_->GetRawTilingData()->GetData());
    *tilingPtr = tilingData_;
    return ge::GRAPH_SUCCESS;
}

uint64_t GemmSyrkTiling::GetTilingKey() const { return GemmSyrkTilingKey().SetTrans(transX_).GetTilingKey(); }

} // namespace gemm_syrk
} // namespace optiling
