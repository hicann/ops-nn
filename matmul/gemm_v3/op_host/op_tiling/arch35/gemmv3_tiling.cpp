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
 * \file gemmv3_tiling.cpp
 * \brief GemmV3 tiling implementation.
 */

#include "gemmv3_tiling.h"

#include "gemmv3_tiling_strategy.h"
#include "error_util.h"
#include "matmul/common/op_host/log_format_util.h"

namespace {

constexpr size_t HF32_ATTR_NUM = 5;
constexpr size_t HF32_ATTR_INDEX = 4;
constexpr size_t ALLOW_DIM = 2;
constexpr size_t A_IDX = 0;
constexpr size_t B_IDX = 1;
constexpr size_t C_IDX = 2;
constexpr size_t Y_IDX = 0;
constexpr size_t ALPHA_ATTR_INDEX = 0;
constexpr size_t BETA_ATTR_INDEX = 1;
constexpr size_t TRANSA_ATTR_INDEX = 2;
constexpr size_t TRANSB_ATTR_INDEX = 3;
constexpr float EPSILON = 1e-6f;
} // namespace

namespace optiling {
namespace gemmv3 {

ge::graphStatus GemmV3Tiling::ValidateInputsNotNull()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(A_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(A_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(B_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(B_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(Y_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputShape(Y_IDX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<float>(ALPHA_ATTR_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<float>(BETA_ATTR_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(TRANSA_ATTR_INDEX));
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(TRANSB_ATTR_INDEX));
    if (attrs->GetAttrNum() >= HF32_ATTR_NUM) {
        OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs->GetAttrPointer<bool>(HF32_ATTR_INDEX));
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GemmV3Tiling::DetectOptionalInputs()
{
    if (context_->GetOptionalInputDesc(C_IDX) != nullptr) {
        OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOptionalInputShape(C_IDX));
    }
    return ge::GRAPH_SUCCESS;
}

void GemmV3Tiling::ExtractAttrFlags()
{
    if (context_->GetAttrs()->GetAttrNum() >= HF32_ATTR_NUM) {
        args_.isHf32 = *context_->GetAttrs()->GetAttrPointer<bool>(HF32_ATTR_INDEX);
    }
    OP_LOGD(args_.opName, "GemmV3 Hf32 flag is: %d", args_.isHf32);
}

ge::graphStatus GemmV3Tiling::ExtractTranspose()
{
    args_.isATrans = *context_->GetAttrs()->GetAttrPointer<bool>(TRANSA_ATTR_INDEX);
    args_.isBTrans = *context_->GetAttrs()->GetAttrPointer<bool>(TRANSB_ATTR_INDEX);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GemmV3Tiling::ExtractMKN()
{
    const gert::Shape& aShape = context_->GetInputShape(A_IDX)->GetOriginShape();
    const gert::Shape& bShape = context_->GetInputShape(B_IDX)->GetOriginShape();
    int64_t kA = aShape[args_.isATrans ? 0 : 1];
    int64_t kB = bShape[args_.isBTrans ? 1 : 0];
    args_.mValue = static_cast<uint64_t>(aShape[args_.isATrans ? 1 : 0]);
    args_.kValue = static_cast<uint64_t>(kA);
    kBValue_ = kB;
    args_.nValue = static_cast<uint64_t>(bShape[args_.isBTrans ? 0 : 1]);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GemmV3Tiling::ValidateShape()
{
    const gert::Shape& aShape = context_->GetInputShape(A_IDX)->GetOriginShape();
    const gert::Shape& bShape = context_->GetInputShape(B_IDX)->GetOriginShape();
    const gert::Shape& yShape = context_->GetOutputShape(Y_IDX)->GetOriginShape();
    const size_t aDimNum = aShape.GetDimNum();
    const size_t bDimNum = bShape.GetDimNum();
    const size_t yDimNum = yShape.GetDimNum();
    if (aDimNum != ALLOW_DIM || bDimNum != ALLOW_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            args_.opName, "A, B", Ops::NN::FormatString("[%zu, %zu]", aDimNum, bDimNum).c_str(),
            Ops::NN::FormatString("The shape dims of %s must be %zu", "A, B", ALLOW_DIM).c_str());
        return ge::GRAPH_FAILED;
    }
    if (yDimNum != ALLOW_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            args_.opName, "out", std::to_string(yDimNum).c_str(),
            Ops::NN::FormatString("The shape dim of %s must be %zu", "out", ALLOW_DIM).c_str());
        return ge::GRAPH_FAILED;
    }
    if (args_.kValue != static_cast<uint64_t>(kBValue_)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            args_.opName, "A, B",
            Ops::NN::FormatString("%s, %s", Ops::Base::ToString(aShape).c_str(), Ops::Base::ToString(bShape).c_str())
                .c_str(),
            Ops::NN::FormatString("%s of %s must be equal", "K-axis", "A, B").c_str());
        return ge::GRAPH_FAILED;
    }
    auto isValidDimValue = [](uint64_t dim) -> bool { return dim > 0UL && dim <= static_cast<uint64_t>(INT32_MAX); };
    if (!isValidDimValue(args_.mValue) || !isValidDimValue(args_.kValue) || !isValidDimValue(args_.nValue)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            args_.opName, "A, B",
            Ops::NN::FormatString("%s, %s", Ops::Base::ToString(aShape).c_str(), Ops::Base::ToString(bShape).c_str())
                .c_str(),
            Ops::NN::FormatString("%s of %s must be within the range %s", "m, k, n", "A, B", "(0, INT32_MAX]").c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GemmV3Tiling::ValidateBias()
{
    if (context_->GetOptionalInputDesc(C_IDX) == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    const gert::Shape& cShape = context_->GetOptionalInputShape(C_IDX)->GetOriginShape();
    const gert::Shape& yShape = context_->GetOutputShape(Y_IDX)->GetOriginShape();
    const size_t cDimNum = cShape.GetDimNum();
    if (cDimNum != ALLOW_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            args_.opName, "C", std::to_string(cDimNum).c_str(),
            Ops::NN::FormatString("The shape dim of %s must be %zu", "C", ALLOW_DIM).c_str());
        return ge::GRAPH_FAILED;
    }
    if (cShape[0] != yShape[0] || cShape[1] != yShape[1]) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            args_.opName, "C, out",
            Ops::NN::FormatString("%s, %s", Ops::Base::ToString(cShape).c_str(), Ops::Base::ToString(yShape).c_str())
                .c_str(),
            Ops::NN::FormatString("The shapes of %s must be the same", "C, out").c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GemmV3Tiling::ValidateOpSpecific()
{
    auto attrs = context_->GetAttrs();
    const float* alphaValue = attrs->GetAttrPointer<float>(ALPHA_ATTR_INDEX);
    const float* betaValue = attrs->GetAttrPointer<float>(BETA_ATTR_INDEX);
    if (std::abs(*alphaValue - 1.0f) > EPSILON || std::abs(*betaValue - 1.0f) > EPSILON) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            args_.opName, "alpha, beta", Ops::NN::FormatString("%f, %f", *alphaValue, *betaValue).c_str(),
            Ops::NN::FormatString("The values of %s must be %s", "alpha, beta", "1.0").c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

std::vector<std::vector<ge::DataType>> GemmV3Tiling::GetDtypeSupportList() const
{
    return {{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT},
            {ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT},
            {ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT}};
}

ge::graphStatus GemmV3Tiling::ValidateDtype()
{
    std::vector<ge::DataType> dtype = {args_.aType, args_.bType, args_.cType};
    auto supportList = GetDtypeSupportList();
    for (auto& supported : supportList) {
        if (std::equal(dtype.begin(), dtype.end(), supported.begin())) {
            return ge::GRAPH_SUCCESS;
        }
    }
    std::string incorrectDtypes = Ops::Base::ToString(args_.aType) + ", " + Ops::Base::ToString(args_.bType) + ", " +
                                  Ops::Base::ToString(args_.cType);
    OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
        args_.opName, "A, B, out", incorrectDtypes.c_str(),
        Ops::NN::FormatString("The dtypes of %s must be within the range %s", "A, B, out",
                              "(FLOAT16,FLOAT16,FLOAT), (BF16,BF16,FLOAT), (FLOAT,FLOAT,FLOAT)")
            .c_str());
    return ge::GRAPH_FAILED;
}

std::vector<int32_t> GemmV3Tiling::GetRegistryPriorities(NpuArch npuArch) const
{
    return strategy::GetGemmV3Priorities(npuArch);
}

MatMulV3TilingKey* GemmV3Tiling::GetTilingKeyObj() { return &gemmV3TilingKey_; }
} // namespace gemmv3
} // namespace optiling
