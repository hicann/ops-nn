/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_batch_matmul_v4_weight_quant_mx_cube_tiling.cpp
 * \brief
 */
#include "quant_batch_matmul_v4_weight_quant_mx_cube_tiling.h"

#include <algorithm>
#include <cstdint>
#include <utility>

#include "../../../../quant_batch_matmul_v3/op_host/op_tiling/arch35/base_block_calculator.h"
#include "../../../../quant_batch_matmul_v3/op_host/op_tiling/arch35/l1_tiling_data_calculator.h"
#include "../../../op_kernel/arch35/quant_batch_matmul_v4_tiling_key.h"
#include "error_util.h"
#include "graph/utils/type_utils.h"
#include "log/log.h"
#include "matmul/common/op_host/log_format_util.h"
#include "matmul/common/op_host/op_tiling/weight_quant_batch_matmul_tiling_tool.h"
#include "quant_batch_matmul_v4_basic_block_tiling.h"
#include "quant_batch_matmul_v4_tiling.h"

namespace optiling {
using matmul_v4::BLOCK_CUBE;
using matmul_v4::DIM_INDEX_0;
using matmul_v4::DIM_INDEX_1;
using matmul_v4::GROUP_SIZE_INDEX;
using matmul_v4::MATMUL_SHAPE_DIM_NUM;
using matmul_v4::MX_GROUP_SIZE;
using matmul_v4::VALID_MX_SCALE_LAST_DIM;
using matmul_v4::Y_OUTPUT_INDEX;
using qmmv3_tiling_const::DATA_SIZE_L0C;
using qmmv3_tiling_const::DOUBLE_BUFFER_NUM;
using qmmv3_tiling_const::L1_FOUR_BUFFER;
using qmmv3_tiling_const::L1_TWO_BUFFER;

namespace {
constexpr uint64_t GROUP_SIZE_K_MASK = 0xFFFFUL; // K-direction group size occupies the low 16 bits.

bool IsPositiveInt32(int64_t value)
{
    return value >= matmul_v4::MIN_SHAPE_SIZE && value <= static_cast<int64_t>(matmul_v4::MAX_SHAPE_DIM);
}
} // namespace

ge::graphStatus AdaptiveSlidingWindowWeightQuantMxCubeTiling::GetShapeAttrsInfo()
{
    inputParams_.Reset();
    inputParams_.opName = GetDefaultOpName();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_);
    auto* rawTilingData = context_->GetRawTilingData();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData->GetData());
    adaptiveWin_ = {};
    return AdaptiveSlidingWindowTiling::GetShapeAttrsInfo();
}

bool AdaptiveSlidingWindowWeightQuantMxCubeTiling::SetPlatformInfoForTiling()
{
    auto platformInfo = context_->GetPlatformInfo();
    OP_TILING_CHECK(platformInfo == nullptr, OP_LOGE(inputParams_.opName, "Platform info is null."), return false);
    compileInfo_ = {};
    QuantBatchMatmulV3TilingBase::InitCompileInfo();
    // InitCompileInfo会在supportMmadS8S4时将npuArch设为DAV_RESV，需修正为实际值
    compileInfo_.npuArch = platform_ascendc::PlatformAscendC(platformInfo).GetCurNpuArch();
    OP_TILING_CHECK(compileInfo_.aicNum == 0 || compileInfo_.l1Size == 0 || compileInfo_.l2Size == 0 ||
                        compileInfo_.l0aSize == 0 || compileInfo_.l0bSize == 0 || compileInfo_.l0cSize == 0,
                    OP_LOGE(inputParams_.opName, "Nonzero AIC, L1, L2 and L0 resources are required."), return false);
    aicoreParams_.aicNum = compileInfo_.aicNum;
    aicoreParams_.ubSize = compileInfo_.ubSize;
    aicoreParams_.l1Size = compileInfo_.l1Size;
    aicoreParams_.l0aSize = compileInfo_.l0aSize;
    aicoreParams_.l0bSize = compileInfo_.l0bSize;
    aicoreParams_.l0cSize = compileInfo_.l0cSize;
    inputParams_.libApiWorkSpaceSize = compileInfo_.workspaceNum;
    return true;
}

bool AdaptiveSlidingWindowWeightQuantMxCubeTiling::AnalyzeAttrs()
{
    if (!AnalyzeGroupSize() || !AnalyzeTranspose()) {
        return false;
    }
    inputParams_.outDtype = *context_->GetAttrs()->GetAttrPointer<int64_t>(matmul_v4::DTYPE_INDEX);
    return true;
}

bool AdaptiveSlidingWindowWeightQuantMxCubeTiling::AnalyzeTranspose()
{
    auto attrs = context_->GetAttrs();
    auto transposeX1 = attrs->GetAttrPointer<bool>(matmul_v4::TRANSPOSE_X1_INDEX);
    auto transposeX2 = attrs->GetAttrPointer<bool>(matmul_v4::TRANSPOSE_X2_INDEX);
    if (transposeX1 == nullptr || *transposeX1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "transpose_x1", transposeX1 ? "true" : "null",
                                              "transpose_x1 must be false");
        return false;
    }
    if (transposeX2 == nullptr || !*transposeX2) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "transpose_x2", transposeX2 ? "false" : "null",
                                              "transpose_x2 must be true");
        return false;
    }
    inputParams_.transA = *transposeX1;
    inputParams_.transB = *transposeX2;
    return true;
}

bool AdaptiveSlidingWindowWeightQuantMxCubeTiling::AnalyzeGroupSize()
{
    auto attrs = context_->GetAttrs();
    auto groupSize = attrs->GetAttrNum() > GROUP_SIZE_INDEX ? attrs->GetAttrPointer<int64_t>(GROUP_SIZE_INDEX) :
                                                              nullptr;
    if (groupSize == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "group_size", "null", "group_size must be provided");
        return false;
    }
    const auto x2ScaleDtype = context_->GetOptionalInputDesc(GetScaleIdx())->GetDataType();
    if (x2ScaleDtype == Ops::NN::DT_HIFLOAT4_SCALE) {
        // MxFp8HiF4场景A B矩阵groupSize不同，groupSize要求传入-1，内部按固定值处理
        if (*groupSize != -1) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                inputParams_.opName, "group_size", std::to_string(*groupSize).c_str(),
                "when the dtype of x2_scale is HIFLOAT4_SCALE, group_size must be -1");
            return false;
        }
    } else if (x2ScaleDtype == ge::DT_FLOAT8_E8M0) {
        const uint64_t groupSizeK = static_cast<uint64_t>(*groupSize) & GROUP_SIZE_K_MASK;
        if (groupSizeK != MX_GROUP_SIZE) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                inputParams_.opName, "group_size", std::to_string(*groupSize).c_str(),
                "when the dtype of x2_scale is FLOAT8_E8M0, the low 16 bits of group_size must be 32");
            return false;
        }
    }

    inputParams_.groupSizeM = 1UL;
    inputParams_.groupSizeN = 1UL;
    inputParams_.groupSizeK = x2ScaleDtype == Ops::NN::DT_HIFLOAT4_SCALE ? matmul_v4::HIF4_GROUP_SIZE : MX_GROUP_SIZE;
    inputParams_.groupSize = inputParams_.groupSizeK;
    return true;
}

bool AdaptiveSlidingWindowWeightQuantMxCubeTiling::CheckDtype() const
{
    return CheckOperandDtypes() && CheckOutputAndBiasDtypes();
}

bool AdaptiveSlidingWindowWeightQuantMxCubeTiling::CheckOperandDtypes() const
{
    OP_TILING_CHECK(
        context_->GetOptionalInputDesc(GetPertokenIdx()) == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "x1_scale", "null", "x1_scale must be provided"),
        return false);
    if (inputParams_.aDtype != ge::DT_FLOAT8_E4M3FN || inputParams_.perTokenScaleDtype != ge::DT_FLOAT8_E8M0) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            inputParams_.opName, "x1, x1_scale",
            Ops::NN::FormatString("%s, %s", ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                                  ge::TypeUtils::DataTypeToSerialString(inputParams_.perTokenScaleDtype).c_str())
                .c_str(),
            "the dtypes of x1 and x1_scale must be (FLOAT8_E4M3FN, FLOAT8_E8M0)");
        return false;
    }
    const bool isFp4 = inputParams_.bDtype == ge::DT_FLOAT4_E2M1 && inputParams_.scaleDtype == ge::DT_FLOAT8_E8M0;
    const bool isHif4 = inputParams_.bDtype == Ops::NN::DT_HIFLOAT4 &&
                        inputParams_.scaleDtype == Ops::NN::DT_HIFLOAT4_SCALE;
    if (!isFp4 && !isHif4) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            inputParams_.opName, "x2, x2_scale",
            Ops::NN::FormatString("%s, %s", ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str(),
                                  ge::TypeUtils::DataTypeToSerialString(inputParams_.scaleDtype).c_str())
                .c_str(),
            "the dtypes of x2 and x2_scale must be (FLOAT4_E2M1, FLOAT8_E8M0) or (HIFLOAT4, HIFLOAT4_SCALE)");
        return false;
    }
    return true;
}

bool AdaptiveSlidingWindowWeightQuantMxCubeTiling::CheckOutputAndBiasDtypes() const
{
    if (inputParams_.cDtype != ge::DT_FLOAT16 && inputParams_.cDtype != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams_.opName, "y",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str(),
                                              "the dtype of y must be FLOAT16 or BF16");
        return false;
    }
    auto biasDesc = context_->GetOptionalInputDesc(GetBiasIdx());
    auto biasShape = context_->GetOptionalInputShape(GetBiasIdx());
    if ((biasDesc == nullptr) != (biasShape == nullptr)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "bias", biasDesc ? "descriptor only" : "shape only",
                                              "bias must have both a descriptor and a shape when provided");
        return false;
    }
    if (biasDesc != nullptr && inputParams_.biasDtype != inputParams_.cDtype) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            inputParams_.opName, "bias, y",
            Ops::NN::FormatString("%s, %s", ge::TypeUtils::DataTypeToSerialString(inputParams_.biasDtype).c_str(),
                                  ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str())
                .c_str(),
            "the dtype of bias must be the same as that of y");
        return false;
    }
    return true;
}

bool AdaptiveSlidingWindowWeightQuantMxCubeTiling::CheckDisabledInputs() const
{
    constexpr std::pair<uint32_t, const char*> disabledInputs[] = {{Y_SCALE_INDEX_V4, "y_scale"},
                                                                   {X1_OFFSET_INDEX_V4, "x1_offset"},
                                                                   {X2_OFFSET_INDEX_V4, "x2_offset"},
                                                                   {Y_OFFSET_INDEX_V4, "y_offset"},
                                                                   {X2_TABLE_INDEX_V4, "x2_table"}};
    for (const auto& input : disabledInputs) {
        const auto* shape = context_->GetOptionalInputShape(input.first);
        if (shape != nullptr && shape->GetStorageShape().GetShapeSize() != 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, input.second,
                                                  std::to_string(shape->GetStorageShape().GetShapeSize()).c_str(),
                                                  "the tensor must be null or have a shape size of 0");
            return false;
        }
    }
    return true;
}

bool AdaptiveSlidingWindowWeightQuantMxCubeTiling::AnalyzeInputs()
{
    const auto x1Shape = GetX1Shape(GetX1Idx());
    const auto x2Shape = GetX2Shape(GetX2Idx());
    const auto pertokenShape = GetPertokenShape(GetPertokenIdx());
    const auto biasShape = GetBiasShape(GetBiasIdx());
    OP_TILING_CHECK(pertokenShape == nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "x1_scale", "null shape",
                                                          "x1_scale must be provided with its shape"),
                    return false);
    inputParams_.isMxA8W4 = true;
    inputParams_.isMxPerGroup = inputParams_.perTokenScaleDtype == ge::DT_FLOAT8_E8M0 &&
                                inputParams_.scaleDtype == ge::DT_FLOAT8_E8M0;
    inputParams_.isHif4PerGroup = false;
    inputParams_.isPertoken = false;
    inputParams_.hasBias = biasShape != nullptr;
    inputParams_.batchBias = 1UL;
    inputParams_.batchA = 1UL;
    inputParams_.batchB = 1UL;
    inputParams_.batchC = 1UL;
    if (!CheckMatrixShape("x1", x1Shape) || !CheckMatrixShape("x2", x2Shape)) {
        return false;
    }
    inputParams_.mSize = static_cast<uint64_t>(x1Shape.GetDim(DIM_INDEX_0));
    inputParams_.kSize = static_cast<uint64_t>(x1Shape.GetDim(DIM_INDEX_1));
    inputParams_.nSize = static_cast<uint64_t>(x2Shape.GetDim(DIM_INDEX_0));
    AnalyzeBatchInfo(x1Shape, x2Shape);
    return CheckDisabledInputs() && CheckMxShapes();
}

bool AdaptiveSlidingWindowWeightQuantMxCubeTiling::CheckMatrixShape(const char* name, const gert::Shape& shape) const
{
    if (shape.GetDimNum() != MATMUL_SHAPE_DIM_NUM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(inputParams_.opName, name,
                                                 Ops::NN::FormatString("%zuD", shape.GetDimNum()).c_str(),
                                                 "the dimNum of the matrix must be 2");
        return false;
    }
    for (size_t i = 0; i < MATMUL_SHAPE_DIM_NUM; ++i) {
        if (!IsPositiveInt32(shape.GetDim(i))) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                inputParams_.opName, Ops::NN::FormatString("%s[%zu]", name, i).c_str(),
                std::to_string(shape.GetDim(i)).c_str(), "the dimension must be in range [1, INT32_MAX]");
            return false;
        }
    }
    return true;
}

bool AdaptiveSlidingWindowWeightQuantMxCubeTiling::CheckMxShapes() const
{
    if (inputParams_.kSize % matmul_v4::K_ALIGN_SIZE_MX != 0 || inputParams_.nSize % matmul_v4::N_ALIGN_SIZE_8 != 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            inputParams_.opName, "K, N",
            (std::to_string(inputParams_.kSize) + ", " + std::to_string(inputParams_.nSize)).c_str(),
            "K and N must both be multiples of 8");
        return false;
    }
    const int64_t m = static_cast<int64_t>(inputParams_.mSize);
    const int64_t n = static_cast<int64_t>(inputParams_.nSize);
    const int64_t k = static_cast<int64_t>(inputParams_.kSize);
    const int64_t scaleKGroupCount = ops::CeilDiv(k, MX_GROUP_SIZE * VALID_MX_SCALE_LAST_DIM);
    const int64_t bScaleLastDim = inputParams_.bDtype == Ops::NN::DT_HIFLOAT4 ? matmul_v4::VALID_MXHIF4_SCALE_LAST_DIM :
                                                                                VALID_MX_SCALE_LAST_DIM;
    const gert::Shape aShape({m, k});
    const gert::Shape bShape({n, k});
    const gert::Shape yShape({m, n});
    const int64_t nzK0 = static_cast<int64_t>(GetBlockAlignSizeByDataType(inputParams_.bDtype));
    const gert::Shape bStorageShape({ops::CeilDiv(k, nzK0), ops::CeilDiv(n, BLOCK_CUBE), BLOCK_CUBE, nzK0});
    const gert::Shape aScaleShape({m, scaleKGroupCount, VALID_MX_SCALE_LAST_DIM});
    const gert::Shape bScaleShape({n, scaleKGroupCount, bScaleLastDim});
    if (!CheckTensorLayout("x1", context_->GetInputDesc(GetX1Idx()), context_->GetInputShape(GetX1Idx()), ge::FORMAT_ND,
                           aShape, aShape) ||
        !CheckTensorLayout("x2", context_->GetInputDesc(GetX2Idx()), context_->GetInputShape(GetX2Idx()),
                           ge::FORMAT_FRACTAL_NZ, bShape, bStorageShape) ||
        !CheckTensorLayout("y", context_->GetOutputDesc(Y_OUTPUT_INDEX), context_->GetOutputShape(Y_OUTPUT_INDEX),
                           ge::FORMAT_ND, yShape, yShape) ||
        !CheckTensorLayout("x1_scale", context_->GetOptionalInputDesc(GetPertokenIdx()),
                           context_->GetOptionalInputShape(GetPertokenIdx()), ge::FORMAT_ND, aScaleShape,
                           aScaleShape) ||
        !CheckTensorLayout("x2_scale", context_->GetOptionalInputDesc(GetScaleIdx()),
                           context_->GetOptionalInputShape(GetScaleIdx()), ge::FORMAT_ND, bScaleShape, bScaleShape)) {
        return false;
    }
    if (!inputParams_.hasBias) {
        return true;
    }
    const gert::Shape biasShape({1, n});
    return CheckTensorLayout("bias", context_->GetOptionalInputDesc(GetBiasIdx()),
                             context_->GetOptionalInputShape(GetBiasIdx()), ge::FORMAT_ND, biasShape, biasShape);
}

bool AdaptiveSlidingWindowWeightQuantMxCubeTiling::CheckTensorLayout(const char* name,
                                                                     const gert::CompileTimeTensorDesc* desc,
                                                                     const gert::StorageShape* shape, ge::Format format,
                                                                     const gert::Shape& originShape,
                                                                     const gert::Shape& storageShape) const
{
    const auto actualFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(desc->GetStorageFormat()));
    if (actualFormat != format) {
        OP_LOGE_FOR_INVALID_FORMAT(inputParams_.opName, name, Ops::Base::ToString(actualFormat),
                                   Ops::Base::ToString(format));
        return false;
    }
    if (shape->GetOriginShape() != originShape || shape->GetStorageShape() != storageShape) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            inputParams_.opName, Ops::NN::FormatString("%s originShape, %s storageShape", name, name).c_str(),
            Ops::NN::FormatString("%s, %s", Ops::Base::ToString(shape->GetOriginShape()).c_str(),
                                  Ops::Base::ToString(shape->GetStorageShape()).c_str())
                .c_str(),
            Ops::NN::FormatString("the origin shape must be %s and the storage shape must be %s",
                                  Ops::Base::ToString(originShape).c_str(), Ops::Base::ToString(storageShape).c_str())
                .c_str());
        return false;
    }
    return true;
}

bool AdaptiveSlidingWindowWeightQuantMxCubeTiling::IsCapable() { return true; }

uint64_t AdaptiveSlidingWindowWeightQuantMxCubeTiling::GetTilingKey() const
{
    const uint64_t kernelType = isAFullLoad_ ? QBMMV4_MX_BLAZE_A_FULL_LOAD : QBMMV4_MX_BLAZE;
    return GET_TPL_TILING_KEY(QBMMV4_B_TRANS, QBMMV4_MX, QBMMV4_OPTION_ATTR_NONE, QBMMV4_IS_WEIGHT_NZ, kernelType);
}

void AdaptiveSlidingWindowWeightQuantMxCubeTiling::ResetTilingData()
{
    AdaptiveSlidingWindowTiling::ResetTilingData();
    withoutBatchTilingData_ = {};
}

const void* AdaptiveSlidingWindowWeightQuantMxCubeTiling::GetTilingData() const { return &withoutBatchTilingData_; }

bool AdaptiveSlidingWindowWeightQuantMxCubeTiling::CalcBasicBlock()
{
    BaseBlockCalculator calculator(inputParams_, compileInfo_, GetBatchCoreCnt(), context_->GetPlatformInfo());
    if (!calculator.Compute(BaseBlockMode::LOAD_BALANCE)) {
        return false;
    }
    const auto& baseBlockRes = calculator.GetOutput();
    adaptiveWin_.baseM = baseBlockRes.baseM;
    adaptiveWin_.baseN = baseBlockRes.baseN;
    adaptiveWin_.baseK = baseBlockRes.baseK;
    adaptiveWin_.useTailWinLogic = baseBlockRes.useTailWinLogic;
    return true;
}

void AdaptiveSlidingWindowWeightQuantMxCubeTiling::UpdateAFullLoadStatus()
{
    if (adaptiveWin_.mBlockCnt == 0UL) {
        isAFullLoad_ = false;
        return;
    }
    const uint64_t singleCoreASize = GetSizeWithDataType(adaptiveWin_.baseM * inputParams_.kSize, inputParams_.aDtype);
    const uint64_t windowLen = GetAswWindowLen(compileInfo_.npuArch, aicoreParams_.aicNum);
    isAFullLoad_ = singleCoreASize <= aicoreParams_.l1Size / qmmv3_tiling_const::AFULLLOAD_SINGLE_CORE_A_SCALER &&
                   adaptiveWin_.mBlockCnt < windowLen && aicoreParams_.aicNum % adaptiveWin_.mBlockCnt == 0UL &&
                   adaptiveWin_.totalBlockCnt > aicoreParams_.aicNum && inputParams_.batchC == 1UL;
}

void AdaptiveSlidingWindowWeightQuantMxCubeTiling::AnalyseFullLoadInfo()
{
    isABFullLoad_ = false;
    isBFullLoad_ = false;
    UpdateAFullLoadStatus();
}

void AdaptiveSlidingWindowWeightQuantMxCubeTiling::CalcTailRoundBasicBlockSplit()
{
    if (!adaptiveWin_.useTailWinLogic) {
        return;
    }
    if (isAFullLoad_) {
        CalcTailBasicBlockAfullLoad();
    } else {
        CalcTailBasicBlock();
    }
}

bool AdaptiveSlidingWindowWeightQuantMxCubeTiling::CalL1Tiling()
{
    basicTiling_.usedCoreNum = CalUsedCoreNum();
    OP_LOGD(inputParams_.opName, "CoreNum: %u", basicTiling_.usedCoreNum);
    basicTiling_.baseM = adaptiveWin_.baseM;
    basicTiling_.baseN = adaptiveWin_.baseN;
    basicTiling_.baseK = adaptiveWin_.baseK;
    basicTiling_.stepM = 1U;
    basicTiling_.stepN = 1U;
    basicTiling_.singleCoreM = std::min(inputParams_.mSize, static_cast<uint64_t>(basicTiling_.baseM));
    basicTiling_.singleCoreN = std::min(inputParams_.nSize, static_cast<uint64_t>(basicTiling_.baseN));
    basicTiling_.singleCoreK = inputParams_.kSize;

    basicTiling_.iterateOrder = 0U;
    basicTiling_.dbL0c = basicTiling_.baseM * basicTiling_.baseN * DATA_SIZE_L0C * DOUBLE_BUFFER_NUM <=
                                 aicoreParams_.l0cSize ?
                             DOUBLE_BUFFER_NUM :
                             1U;

    L1TilingMode mode = isAFullLoad_ ? L1TilingMode::A_L1_FULL_LOAD : L1TilingMode::DEFAULT;
    L1TilingDataCalculator l1Calculator(inputParams_, compileInfo_, basicTiling_.baseM, basicTiling_.baseN,
                                        basicTiling_.baseK);
    if (!l1Calculator.Compute(mode)) {
        return false;
    }
    const L1TilingData& l1TilingData = l1Calculator.GetOutput();
    basicTiling_.depthA1 = static_cast<uint32_t>(l1TilingData.depthKa_);
    basicTiling_.depthB1 = static_cast<uint32_t>(l1TilingData.depthKb_);
    basicTiling_.stepKa = static_cast<uint32_t>(l1TilingData.stepKa_);
    basicTiling_.stepKb = static_cast<uint32_t>(l1TilingData.stepKb_);
    basicTiling_.scaleFactorA = static_cast<uint32_t>(l1TilingData.scaleFactorA_);
    basicTiling_.scaleFactorB = static_cast<uint32_t>(l1TilingData.scaleFactorB_);
    return true;
}

uint64_t AdaptiveSlidingWindowWeightQuantMxCubeTiling::DeriveScaleKL1(uint32_t scaleFactorA, uint64_t stepKa,
                                                                      uint32_t scaleFactorB, uint64_t stepKb,
                                                                      uint64_t baseK) const
{
    return std::min(static_cast<uint64_t>(scaleFactorA) * stepKa * baseK,
                    static_cast<uint64_t>(scaleFactorB) * stepKb * baseK);
}

uint64_t AdaptiveSlidingWindowWeightQuantMxCubeTiling::CalcMxFullKLoadSize(uint64_t outerSize, ge::DataType dataDtype,
                                                                           ge::DataType scaleDtype) const
{
    const uint64_t kAligned = ops::CeilAlign(inputParams_.kSize, qmmv3_tiling_const::MXFP_DIVISOR_SIZE);
    const uint64_t scaleK = GetScaleBaseK(scaleDtype, inputParams_.kSize);
    return GetSizeWithDataType(outerSize * kAligned, dataDtype) + GetSizeWithDataType(outerSize * scaleK, scaleDtype);
}

uint64_t AdaptiveSlidingWindowWeightQuantMxCubeTiling::CalcUsedL1Size(const MxL1EstimateParams& params,
                                                                      uint32_t l1BufferNum) const
{
    uint64_t usedL1Size = GetSizeWithDataType(params.baseN * params.kL1, inputParams_.bDtype) * l1BufferNum;
    // B-side scale follows the scaleKL1 window and is double-buffered separately from B data.
    usedL1Size += GetSizeWithDataType(
                      params.baseN * ops::CeilDiv(params.scaleKL1, GetScaleGroupSize(inputParams_.scaleDtype)),
                      inputParams_.scaleDtype) *
                  L1_TWO_BUFFER;
    if (inputParams_.hasBias) {
        usedL1Size += GetSizeWithDataType(params.baseN, inputParams_.biasDtype) * L1_TWO_BUFFER;
    }
    if (params.isAFullLoad) {
        usedL1Size += CalcMxFullKLoadSize(params.baseM, inputParams_.aDtype, inputParams_.perTokenScaleDtype);
    } else {
        usedL1Size += GetSizeWithDataType(params.baseM * params.kL1, inputParams_.aDtype) * l1BufferNum;
        usedL1Size += GetSizeWithDataType(
                          params.baseM *
                              ops::CeilDiv(params.scaleKL1, GetScaleGroupSize(inputParams_.perTokenScaleDtype)),
                          inputParams_.perTokenScaleDtype) *
                      L1_TWO_BUFFER;
    }
    return usedL1Size;
}

void AdaptiveSlidingWindowWeightQuantMxCubeTiling::CalculateNBufferNum()
{
    const uint32_t stepK = std::min(basicTiling_.stepKa, basicTiling_.stepKb);
    const uint64_t currentKL1 = static_cast<uint64_t>(stepK) * basicTiling_.baseK;
    const MxL1EstimateParams params = {currentKL1, withoutBatchTilingData_.scaleKL1, basicTiling_.baseM,
                                       basicTiling_.baseN, isAFullLoad_};
    const uint64_t usedL1Size = CalcUsedL1Size(params, L1_FOUR_BUFFER);
    withoutBatchTilingData_.kAL1 = static_cast<uint32_t>(currentKL1);
    withoutBatchTilingData_.kBL1 = withoutBatchTilingData_.kAL1;
    withoutBatchTilingData_.nBufferNum = static_cast<uint8_t>(usedL1Size < aicoreParams_.l1Size ? L1_FOUR_BUFFER :
                                                                                                  L1_TWO_BUFFER);
}

void AdaptiveSlidingWindowWeightQuantMxCubeTiling::SetTilingData()
{
    withoutBatchTilingData_.m = static_cast<uint32_t>(inputParams_.mSize);
    withoutBatchTilingData_.n = static_cast<uint32_t>(inputParams_.nSize);
    withoutBatchTilingData_.k = static_cast<uint32_t>(inputParams_.kSize);
    withoutBatchTilingData_.baseM = static_cast<uint16_t>(basicTiling_.baseM);
    withoutBatchTilingData_.baseN = static_cast<uint16_t>(basicTiling_.baseN);
    withoutBatchTilingData_.baseK = static_cast<uint16_t>(basicTiling_.baseK);
    withoutBatchTilingData_.groupSizeM = static_cast<uint16_t>(inputParams_.groupSizeM);
    withoutBatchTilingData_.groupSizeN = static_cast<uint16_t>(inputParams_.groupSizeN);
    withoutBatchTilingData_.groupSizeK = static_cast<uint16_t>(inputParams_.groupSizeK);
    withoutBatchTilingData_.mTailTile = static_cast<uint16_t>(adaptiveWin_.mTailTile);
    withoutBatchTilingData_.nTailTile = static_cast<uint16_t>(adaptiveWin_.nTailTile);
    withoutBatchTilingData_.mBaseTailSplitCnt = static_cast<uint16_t>(adaptiveWin_.mBaseTailSplitCnt);
    withoutBatchTilingData_.nBaseTailSplitCnt = static_cast<uint16_t>(adaptiveWin_.nBaseTailSplitCnt);
    withoutBatchTilingData_.mTailMain = static_cast<uint16_t>(adaptiveWin_.mTailMain);
    withoutBatchTilingData_.nTailMain = static_cast<uint16_t>(adaptiveWin_.nTailMain);
    withoutBatchTilingData_.x1QuantMode = static_cast<uint8_t>(GetPergroupQuantMode(inputParams_.perTokenScaleDtype));
    withoutBatchTilingData_.x2QuantMode = static_cast<uint8_t>(GetPergroupQuantMode(inputParams_.scaleDtype));
    withoutBatchTilingData_.isBias = inputParams_.hasBias ? 1U : 0U;
    withoutBatchTilingData_.biasDtype = static_cast<uint8_t>(inputParams_.biasDtype);
    withoutBatchTilingData_.dbL0C = static_cast<uint8_t>(basicTiling_.dbL0c);
    withoutBatchTilingData_.weightMustHitL2 = static_cast<uint8_t>(IsWeightMustHitL2(inputParams_, basicTiling_.baseM));
    withoutBatchTilingData_.windowLen = static_cast<uint32_t>(
        GetAswWindowLen(compileInfo_.npuArch, aicoreParams_.aicNum));
}

ge::graphStatus AdaptiveSlidingWindowWeightQuantMxCubeTiling::DoLibApiTiling()
{
    withoutBatchTilingData_.scaleKL1 = static_cast<uint32_t>(
        DeriveScaleKL1(basicTiling_.scaleFactorA, basicTiling_.stepKa, basicTiling_.scaleFactorB, basicTiling_.stepKb,
                       basicTiling_.baseK));
    CalculateNBufferNum();
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

void AdaptiveSlidingWindowWeightQuantMxCubeTiling::PrintTilingData() const
{
    const auto& tiling = withoutBatchTilingData_;
    OP_LOGD(inputParams_.opName,
            "tilingData: m=%u, n=%u, k=%u, kAL1=%u, kBL1=%u, scaleKL1=%u, "
            "baseM=%u, baseN=%u, baseK=%u, groupSizeM=%u, groupSizeN=%u, groupSizeK=%u, "
            "mTailTile=%u, nTailTile=%u, mBaseTailSplitCnt=%u, nBaseTailSplitCnt=%u, mTailMain=%u, nTailMain=%u, "
            "x1QuantMode=%u, x2QuantMode=%u, isBias=%u, biasDtype=%u, nBufferNum=%u, dbL0C=%u, "
            "weightMustHitL2=%u, windowLen=%u",
            tiling.m, tiling.n, tiling.k, tiling.kAL1, tiling.kBL1, tiling.scaleKL1,
            static_cast<uint32_t>(tiling.baseM), static_cast<uint32_t>(tiling.baseN),
            static_cast<uint32_t>(tiling.baseK), static_cast<uint32_t>(tiling.groupSizeM),
            static_cast<uint32_t>(tiling.groupSizeN), static_cast<uint32_t>(tiling.groupSizeK),
            static_cast<uint32_t>(tiling.mTailTile), static_cast<uint32_t>(tiling.nTailTile),
            static_cast<uint32_t>(tiling.mBaseTailSplitCnt), static_cast<uint32_t>(tiling.nBaseTailSplitCnt),
            static_cast<uint32_t>(tiling.mTailMain), static_cast<uint32_t>(tiling.nTailMain),
            static_cast<uint32_t>(tiling.x1QuantMode), static_cast<uint32_t>(tiling.x2QuantMode),
            static_cast<uint32_t>(tiling.isBias), static_cast<uint32_t>(tiling.biasDtype),
            static_cast<uint32_t>(tiling.nBufferNum), static_cast<uint32_t>(tiling.dbL0C),
            static_cast<uint32_t>(tiling.weightMustHitL2), tiling.windowLen);
}
} // namespace optiling
