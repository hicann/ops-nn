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
 * \file quant_matmul_checker.cpp
 * \brief
 */
#include "quant_matmul_checker.h"
#include "log/log.h"
#include "matmul/common/op_host/log_format_util.h"

using namespace op;
using namespace ge;
using Ops::NN::FormatString;

namespace {
static constexpr int INDEX_X1_IN_INPUT_TUPLE = 0;
static constexpr int INDEX_X2_IN_INPUT_TUPLE = 1;
static constexpr int INDEX_Y_OFFSET_IN_QUANT_TUPLE = 5;
static constexpr int INDEX_BIAS_IN_QUANT_TUPLE = 6;
static constexpr int INDEX_GROUP_SIZE_IN_QUANT_TUPLE = 7;
static constexpr int INDEX_INTERFACE_TYPE_IN_QUANT_TUPLE = 8;
static constexpr int INDEX_X1_SCALE_IN_QUANT_TUPLE = 0;
static constexpr int INDEX_X2_SCALE_IN_QUANT_TUPLE = 1;
static constexpr int INDEX_Y_SCALE_IN_QUANT_TUPLE = 2;
static constexpr int INDEX_X1_OFFSET_IN_QUANT_TUPLE = 3;
static constexpr int INDEX_X2_OFFSET_IN_QUANT_TUPLE = 4;

static const int X2_FIXED_DIM_NUM_A4W4 = 2;
static const int64_t INT4_NUMS_IN_INT8 = 2;
static const int64_t MIN_DIM_NUM_ND = 2;
static const size_t MX_SCALE_DIM = 3;
static const size_t MX_SCALE_MIN_DIM = 3;
static const size_t PENULTIMATE_DIM = 2;
static const size_t BATCH_TAILENDER_DIM = 3;
static const size_t NZ_K1_INDEX = 3;
static const size_t NZ_K1_INDEX_TRANS = 4;
static const int64_t NZ_K0_VALUE_INT8_INT4 = 16;
static const int64_t NZ_K0_VALUE_INT8_TRANS = 32;
static const int64_t NZ_K0_VALUE_INT4_TRANS = 64;
static constexpr int64_t OUTPUT_INFER_FAIL = -1L;
static const int64_t PERGROUP_GROUP_SIZE = 32;
static const int64_t PERGROUP_GROUPSIZEK_SIZE = 256;
static const int64_t MXFP_DIVISOR_SIZE = 64;
static const int64_t MXFP_MULTI_BASE_SIZE = 2;
static const uint64_t PERBLOCK_BLOCK_SIZE = 128;
static const int32_t GROUP_M_OFFSET = 32;
static const int32_t GROUP_N_OFFSET = 16;
static const uint64_t GROUP_MNK_BIT_SIZE = 0xFFFF;
static const int64_t MICRO_SCALING_ALIGN_NUM = 2;

const std::map<int64_t, std::string> X1_NAME{{3, "x1"}, {4, "x1"}, {5, "x1"}};

const std::map<int64_t, std::string> X2_NAME{{3, "x2"}, {4, "x2"}, {5, "x2"}};

const std::map<int64_t, std::string> OUT_NAME{{3, "out"}, {4, "out"}, {5, "out"}};

const std::map<int64_t, std::string> BIAS_NAME{{3, "bias"}, {4, "bias"}, {5, "bias"}};

const std::map<int64_t, std::string> X1SCALE_NAME{
    {3, "pertokenScaleOptional"}, {4, "pertokenScaleOptional"}, {5, "x1Scale(pertokenScale)"}};

const std::map<int64_t, std::string> X2SCALE_NAME{{3, "scale"}, {4, "scale"}, {5, "x2Scale(scale)"}};

const std::map<int64_t, std::string> X2OFFSET_NAME{{3, "offset"}, {4, "offset"}, {5, "x2Offset(offset)"}};

static const std::initializer_list<op::DataType> X1_X2_L0C2OUT_PERTOKEN_SUPPORT_LIST = {op::DataType::DT_INT4,
                                                                                        op::DataType::DT_INT8};
static const std::initializer_list<op::DataType> UINT64_X2_SCALE_TYPE_SUPPORT_LIST = {op::DataType::DT_UINT64,
                                                                                      op::DataType::DT_INT64};
static const std::initializer_list<op::DataType> X2_INT8_X2_SCALE_TYPE_SUPPORT_LIST = {
    op::DataType::DT_UINT64, op::DataType::DT_INT64, op::DataType::DT_FLOAT, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> INT8_OUT_TYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8, op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> HIF8_OUT_TYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> FP8_OUT_TYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> PERTOKEN_FP16_OUT_BIAS_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> PERTOKEN_BF16_OUT_X2SCALE_SUPPORT_LIST = {op::DataType::DT_BF16,
                                                                                           op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> PERTOKEN_BF16_OUT_BIAS_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_BF16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> PERTOKEN_OUT_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16,
                                                                                   op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> BF16_OUT_X2SCALE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                                  op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> BIAS_TYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_BF16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> FP8_AND_HIF8_COMMON_OUT_TYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> FOUR_BIT_FLOAT_INPUT_LIST = {op::DataType::DT_FLOAT4_E2M1};
static const std::initializer_list<op::DataType> EIGHT_BIT_FLOAT_INPUT_LIST = {op::DataType::DT_FLOAT8_E4M3FN,
                                                                               op::DataType::DT_FLOAT8_E5M2};
static const std::initializer_list<op::DataType> INT32_OUT_X2SCALE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                                   op::DataType::DT_BF16};
static inline std::string GetInputName(const std::map<int64_t, std::string>& inputNameMap, int64_t interfaceType)
{
    std::string inputName;
    auto iter = inputNameMap.find(interfaceType);
    if (iter != inputNameMap.end()) {
        inputName = iter->second;
    }
    return inputName;
}

static inline bool OpCheckDtypeNotSupport(int64_t interfaceType, const std::map<int64_t, std::string>& inputNameMap,
                                          const aclTensor* tensor,
                                          const std::initializer_list<op::DataType>& supportList, const char* apiName)
{
    if (!CheckType(tensor->GetDataType(), supportList)) {
        const auto inputName = GetInputName(inputNameMap, interfaceType);
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(apiName, inputName.c_str(),
                                              op::ToString(tensor->GetDataType()).GetString(),
                                              FormatString("the dtype of %s must be in dtype support list %s",
                                                           inputName.c_str(), op::ToString(supportList).GetString())
                                                  .c_str());
        return false;
    }
    return true;
}

static inline bool OpCheckDtypeNotMatch(int64_t interfaceType, const std::map<int64_t, std::string>& inputNameMap,
                                        const aclTensor* tensor, DataType expectedDtype, const char* apiName)
{
    if (tensor->GetDataType() != expectedDtype) {
        const auto inputName = GetInputName(inputNameMap, interfaceType);
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            apiName, inputName.c_str(), op::ToString(tensor->GetDataType()).GetString(),
            FormatString("the dtype of %s must be %s", inputName.c_str(), op::ToString(expectedDtype).GetString())
                .c_str());
        return false;
    }
    return true;
}

static inline bool OpCheckWrongDimension(int64_t interfaceType, const std::map<int64_t, std::string>& inputNameMap,
                                         const aclTensor* tensor, size_t expectedDimNum, const char* apiName)
{
    if (tensor->GetViewShape().GetDimNum() != expectedDimNum) {
        const auto inputName = GetInputName(inputNameMap, interfaceType);
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            apiName, inputName.c_str(), FormatString("%zuD", tensor->GetViewShape().GetDimNum()).c_str(),
            FormatString("the shape dim of %s must be %zuD", inputName.c_str(), expectedDimNum).c_str());
        return false;
    }
    return true;
}

static inline std::string RemoveDtInDtype(std::string str)
{
    std::string target = "DT_";
    std::string replacement = "";
    size_t pos = str.find(target);
    while (pos != std::string::npos) {
        str.replace(pos, target.length(), replacement);
        pos = str.find(target, pos + replacement.length());
    }
    return str;
}

static inline bool IsInt8Input(const aclTensor* x1, const aclTensor* x2)
{
    return x1->GetDataType() == op::DataType::DT_INT8 && x2->GetDataType() == op::DataType::DT_INT8;
}

static inline bool IsInt4Input(const aclTensor* x1, const aclTensor* x2)
{
    return x1->GetDataType() == op::DataType::DT_INT4 && x2->GetDataType() == op::DataType::DT_INT4;
}

static inline bool IsInt32Input(const aclTensor* x1, const aclTensor* x2)
{
    return x1->GetDataType() == op::DataType::DT_INT32 && x2->GetDataType() == op::DataType::DT_INT32;
}

static inline bool IsIntInput(const aclTensor* x1, const aclTensor* x2)
{
    return IsInt8Input(x1, x2) || IsInt4Input(x1, x2) || IsInt32Input(x1, x2);
}

static inline bool IsHif8Input(const aclTensor* x1, const aclTensor* x2)
{
    return x1->GetDataType() == op::DataType::DT_HIFLOAT8 && x2->GetDataType() == op::DataType::DT_HIFLOAT8;
}

static inline bool IsMicroScaling(const aclTensor* x1Scale, const aclTensor* x2Scale)
{
    if (x1Scale == nullptr) {
        return false;
    }
    return x1Scale->GetDataType() == op::DataType::DT_FLOAT8_E8M0 &&
           x2Scale->GetDataType() == op::DataType::DT_FLOAT8_E8M0;
}

struct GroupSizeMNK {
    uint64_t m = 0;
    uint64_t n = 0;
    uint64_t k = 0;
};

static inline GroupSizeMNK DecodeGroupSizeMnk(int64_t groupSize)
{
    GroupSizeMNK groupSizeMnk;
    groupSizeMnk.k = static_cast<uint64_t>(groupSize) & GROUP_MNK_BIT_SIZE;
    groupSizeMnk.n = (static_cast<uint64_t>(groupSize) >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    groupSizeMnk.m = (static_cast<uint64_t>(groupSize) >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;
    return groupSizeMnk;
}

static inline int64_t EncodeGroupSizeMnk(const GroupSizeMNK& groupSizeMnk)
{
    return static_cast<int64_t>((groupSizeMnk.m << GROUP_M_OFFSET) | (groupSizeMnk.n << GROUP_N_OFFSET) |
                                groupSizeMnk.k);
}

static inline bool CheckMxGroupSize(int64_t groupSize, const GroupSizeMNK& groupSizeMnk, const char* apiName)
{
    if (groupSizeMnk.k != static_cast<uint64_t>(PERGROUP_GROUP_SIZE) || groupSizeMnk.m != 1ULL ||
        groupSizeMnk.n != 1ULL) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            apiName, "groupSize, groupSizeM, groupSizeN, groupSizeK",
            FormatString("%ld, %lu, %lu, %lu", groupSize, groupSizeMnk.m, groupSizeMnk.n, groupSizeMnk.k).c_str(),
            "when the quantization mode is mx, groupSize must be 4295032864 and Torch API group_sizes must be [1, 1, "
            "32]");
        return false;
    }
    return true;
}

static inline bool CheckA4W4PergroupNonSymmetricGroupSize(const GroupSizeMNK& groupSizeMnk, const char* apiName)
{
    if (groupSizeMnk.k != static_cast<uint64_t>(PERGROUP_GROUPSIZEK_SIZE)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            apiName, "groupSizeK", std::to_string(groupSizeMnk.k).c_str(),
            "in A4W4 pertoken-pergroup non-symmetric scenario, groupSizeK must be 256");
        return false;
    }
    return true;
}

static inline bool CheckPerblockGroupSize(const GroupSizeMNK& groupSizeMnk, const char* apiName)
{
    if (groupSizeMnk.k != PERBLOCK_BLOCK_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "groupSizeK", std::to_string(groupSizeMnk.k).c_str(),
                                              "when the quantization mode is G-B or B-B, groupSizeK must be 128");
        return false;
    }
    if (groupSizeMnk.n != PERBLOCK_BLOCK_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "groupSizeN", std::to_string(groupSizeMnk.n).c_str(),
                                              "when the quantization mode is G-B or B-B, groupSizeN must be 128");
        return false;
    }
    if (groupSizeMnk.m != PERBLOCK_BLOCK_SIZE && groupSizeMnk.m != 1UL) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "groupSizeM", std::to_string(groupSizeMnk.m).c_str(),
                                              "when the quantization mode is G-B or B-B, groupSizeM must be 128 or 1");
        return false;
    }
    return true;
}

static inline bool IsFp8Input(const aclTensor* x1, const aclTensor* x2)
{
    return (std::find(EIGHT_BIT_FLOAT_INPUT_LIST.begin(), EIGHT_BIT_FLOAT_INPUT_LIST.end(), x1->GetDataType()) !=
            EIGHT_BIT_FLOAT_INPUT_LIST.end()) &&
           (std::find(EIGHT_BIT_FLOAT_INPUT_LIST.begin(), EIGHT_BIT_FLOAT_INPUT_LIST.end(), x2->GetDataType()) !=
            EIGHT_BIT_FLOAT_INPUT_LIST.end());
}

static inline bool IsPerblock(const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale,
                              const aclTensor* x2Scale)
{
    if (x1Scale == nullptr || x2Scale == nullptr) {
        return false;
    }
    return ((IsInt8Input(x1, x2) || IsFp8Input(x1, x2) || IsHif8Input(x1, x2)) &&
            x1Scale->GetDataType() == op::DataType::DT_FLOAT && x1Scale->GetViewShape().GetDimNum() > 1 &&
            x2Scale->GetDataType() == op::DataType::DT_FLOAT && x2Scale->GetViewShape().GetDimNum() > 1);
}

static inline bool IsFp4Input(const aclTensor* x1, const aclTensor* x2)
{
    return (std::find(FOUR_BIT_FLOAT_INPUT_LIST.begin(), FOUR_BIT_FLOAT_INPUT_LIST.end(), x1->GetDataType()) !=
            FOUR_BIT_FLOAT_INPUT_LIST.end()) &&
           (std::find(FOUR_BIT_FLOAT_INPUT_LIST.begin(), FOUR_BIT_FLOAT_INPUT_LIST.end(), x2->GetDataType()) !=
            FOUR_BIT_FLOAT_INPUT_LIST.end());
}

template <typename T>
static inline auto CeilDiv(T x, T y) -> T
{
    if (y != 0 && x != 0) {
        const T quotient = x / y;
        return (x % y != 0) ? (quotient + 1) : quotient;
    }
    return x;
}

template <typename T>
static inline auto CeilAlign(T x, T y) -> T
{
    return CeilDiv(x, y) * y;
}

template <typename T>
static inline bool IsAligned(T num, T factor, const char* apiName)
{
    if (factor == 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName, "factor", std::to_string(factor).c_str(),
                                              "the divisor can not be zero");
        return false;
    }
    return num % factor == 0 && num > 0;
}
} // namespace

void QuantMatmulChecker::Init()
{
    socVersion_ = GetCurrentPlatformInfo().GetSocVersion();
    npuArch_ = op::GetCurrentPlatformInfo().GetCurNpuArch();
    x1_ = std::get<INDEX_X1_IN_INPUT_TUPLE>(inputTensors_);
    x2_ = std::get<INDEX_X2_IN_INPUT_TUPLE>(inputTensors_);
    x1Scale_ = std::get<INDEX_X1_SCALE_IN_QUANT_TUPLE>(quantTensors_);
    x2Scale_ = std::get<INDEX_X2_SCALE_IN_QUANT_TUPLE>(quantTensors_);
    yScale_ = std::get<INDEX_Y_SCALE_IN_QUANT_TUPLE>(quantTensors_);
    x1Offset_ = std::get<INDEX_X1_OFFSET_IN_QUANT_TUPLE>(quantTensors_);
    x2Offset_ = std::get<INDEX_X2_OFFSET_IN_QUANT_TUPLE>(quantTensors_);
    yOffset_ = std::get<INDEX_Y_OFFSET_IN_QUANT_TUPLE>(quantTensors_);
    bias_ = std::get<INDEX_BIAS_IN_QUANT_TUPLE>(quantTensors_);
    groupSize_ = std::get<INDEX_GROUP_SIZE_IN_QUANT_TUPLE>(quantTensors_);
    transposeX1_ = std::get<INDEX_X1_IN_INPUT_TUPLE>(boolsTrans_);
    transposeX2_ = std::get<INDEX_X2_IN_INPUT_TUPLE>(boolsTrans_);
    interfaceType_ = std::get<INDEX_INTERFACE_TYPE_IN_QUANT_TUPLE>(quantTensors_);
    if (out_ != nullptr) {
        outType_ = out_->GetDataType();
    }
    if (x2Scale_ != nullptr) {
        x2ScaleType_ = x2Scale_->GetDataType();
    }
    if (x1_ != nullptr && x2_ != nullptr) {
        GetX1X2DimValue();
        isA4W4_ = IsInt4Input(x1_, x2_);
    }
}

bool QuantMatmulChecker::IsA4W4PergroupNonSymmetric(const uint64_t groupSizeK) const
{
    if (x1Scale_ == nullptr || x2Scale_ == nullptr || x2Offset_ == nullptr) {
        return false;
    }

    if (x1Scale_->GetDataType() != op::DataType::DT_FLOAT || x1Scale_->GetViewShape().GetDimNum() <= 1) {
        return false;
    }

    if (x2Scale_->GetDataType() != op::DataType::DT_FLOAT || x2Scale_->GetViewShape().GetDimNum() <= 1) {
        return false;
    }

    if (x2Offset_->GetDataType() != op::DataType::DT_FLOAT16 || x2Offset_->GetViewShape().GetDimNum() <= 1) {
        return false;
    }

    auto bias = std::get<INDEX_BIAS_IN_QUANT_TUPLE>(quantTensors_);
    if (bias != nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName_, "bias", "not null",
                                              "in A4W4 pergroup non-symmetric scenario, bias must be null");
        return false;
    }

    int64_t m = (transposeX1_ ? x1_->GetViewShape().GetDim(1) : x1_->GetViewShape().GetDim(0));
    int64_t k = (transposeX2_ ? x2_->GetViewShape().GetDim(1) : x2_->GetViewShape().GetDim(0));
    int64_t n = (transposeX2_ ? x2_->GetViewShape().GetDim(0) : x2_->GetViewShape().GetDim(1));

    int32_t ALIGN1024 = 1024;
    int32_t ALIGN256 = 256;
    if (k % ALIGN1024 != 0 || n % ALIGN256 != 0) {
        return false;
    }

    return (IsInt4Input(x1_, x2_) && (x1Scale_->GetViewShape().GetDim(0) == m) &&
            (x1Scale_->GetViewShape().GetDim(1) == 1) &&
            (x2Scale_->GetViewShape().GetDim(0) == CeilDiv(k, static_cast<int64_t>(groupSizeK))) &&
            (x2Scale_->GetViewShape().GetDim(1) == n) &&
            (x2Offset_->GetViewShape().GetDim(0) == CeilDiv(k, static_cast<int64_t>(groupSizeK))) &&
            (x2Offset_->GetViewShape().GetDim(1) == n));
}

std::string QuantMatmulChecker::GetX1ScaleName() const { return GetInputName(X1SCALE_NAME, interfaceType_); }

std::string QuantMatmulChecker::GetX2ScaleName() const { return GetInputName(X2SCALE_NAME, interfaceType_); }

std::string QuantMatmulChecker::GetX2OffsetName() const { return GetInputName(X2OFFSET_NAME, interfaceType_); }

bool QuantMatmulChecker::CheckEmptyTensor() const
{
    // scale, out和可选参数已在CheckShape函数校验，无需再次校验空tensor场景。
    if (x1_->IsEmpty() || x2_->IsEmpty()) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            apiName_, "x1, x2",
            FormatString("%s, %s", x1_->IsEmpty() ? "empty" : "not empty", x2_->IsEmpty() ? "empty" : "not empty")
                .c_str(),
            "x1 and x2 can not be empty");
        return false;
    }
    return true;
}

void QuantMatmulChecker::GetX1X2DimValue()
{
    auto x1DimNum = x1_->GetViewShape().GetDimNum();
    auto x2DimNum = x2_->GetViewShape().GetDimNum();
    if (x1DimNum >= MIN_DIM_NUM_ND && x2DimNum >= MIN_DIM_NUM_ND) {
        const op::Shape x1Shape = x1_->GetViewShape();
        const op::Shape x2Shape = x2_->GetViewShape();
        x1KDim_ = transposeX1_ ? x1Shape[x1DimNum - PENULTIMATE_DIM] : x1Shape[x1DimNum - 1];
        x1MDim_ = transposeX1_ ? x1Shape[x1DimNum - 1] : x1Shape[x1DimNum - PENULTIMATE_DIM];
        x2KDim_ = transposeX2_ ? x2Shape[x2DimNum - 1] : x2Shape[x2DimNum - PENULTIMATE_DIM];
        x2NDim_ = transposeX2_ ? x2Shape[x2DimNum - PENULTIMATE_DIM] : x2Shape[x2DimNum - 1];
    }
}

bool QuantMatmulChecker::CheckShapeForWeightNz() const
{
    const op::Shape x2Shape = x2_->GetStorageShape();
    auto x2DimNum = x2_->GetStorageShape().GetDimNum();
    int64_t x2K1Dim = transposeX2_ ? x2Shape[x2DimNum - NZ_K1_INDEX_TRANS] : x2Shape[x2DimNum - NZ_K1_INDEX];
    int64_t nz_k0_value_trans = (x1_->GetDataType() == op::DataType::DT_INT32 ||
                                 x1_->GetDataType() == op::DataType::DT_INT4 ||
                                 x1_->GetDataType() == op::DataType::DT_FLOAT4_E2M1) ?
                                    NZ_K0_VALUE_INT4_TRANS :
                                    NZ_K0_VALUE_INT8_TRANS;
    int64_t aligneValue = transposeX2_ ? nz_k0_value_trans : NZ_K0_VALUE_INT8_INT4;
    int64_t alignedX1K = ((x1KDim_ + aligneValue - 1) / aligneValue) * aligneValue;
    if (alignedX1K != x2K1Dim * aligneValue) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            apiName_, "alignedX1K, x2K1Dim * alignedValue",
            FormatString("%ld, %ld", alignedX1K, x2K1Dim * aligneValue).c_str(),
            "when the format of x2 is FRACTAL_NZ, alignedX1K must be equal to x2K1Dim multiplied by alignedValue");
        return false;
    }
    return true;
}

bool QuantMatmulChecker::CheckMXFP4FP8ParamsNDOrNZ() const
{
    // fp4 内轴偶数校验
    auto x1DimNum = x1_->GetViewShape().GetDimNum();
    auto x1InnerAxis = x1_->GetViewShape().GetDim(x1DimNum - 1);
    auto x2DimNum = x2_->GetViewShape().GetDimNum();
    auto x2InnerAxis = x2_->GetViewShape().GetDim(x2DimNum - 1);
    if (x1InnerAxis % MICRO_SCALING_ALIGN_NUM != 0 || x2InnerAxis % MICRO_SCALING_ALIGN_NUM != 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            apiName_, "x1 inner axis, x2 inner axis", FormatString("%ld, %ld", x1InnerAxis, x2InnerAxis).c_str(),
            "when the quantization mode is mx and x1 and x2 are FLOAT4_E2M1, the inner axes of x1 and x2 must be even");
        return false;
    }
    // fp4 k轴大于2校验
    if (x1KDim_ <= 2 || x2KDim_ <= 2) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(apiName_, "x1 K, x2 K",
                                               FormatString("%ld, %ld", x1KDim_, x2KDim_).c_str(),
                                               "when the quantization mode is mx and x1 and x2 are FLOAT4_E2M1, the K "
                                               "dimension of x1 and x2 must be greater than 2");
        return false;
    }
    return true;
}

bool QuantMatmulChecker::CheckDimValueMicroScaling() const
{
    size_t x1ScaleBatchDimNum = x1Scale_->GetViewShape().GetDimNum() - MX_SCALE_DIM;
    size_t x2ScaleBatchDimNum = x2Scale_->GetViewShape().GetDimNum() - MX_SCALE_DIM;
    auto x1ScaleMDimIndex = x1ScaleBatchDimNum + (transposeX1_ ? 1 : 0);
    auto x1ScaleKDimIndex = x1ScaleBatchDimNum + (transposeX1_ ? 0 : 1);
    auto x2ScaleNDimIndex = x2ScaleBatchDimNum + (transposeX2_ ? 0 : 1);
    auto x2ScaleKDimIndex = x2ScaleBatchDimNum + (transposeX2_ ? 1 : 0);
    auto x1ScaleMDim = x1Scale_->GetViewShape().GetDim(x1ScaleMDimIndex);
    auto x1ScaleKDim = x1Scale_->GetViewShape().GetDim(x1ScaleKDimIndex);
    auto x2ScaleNDim = x2Scale_->GetViewShape().GetDim(x2ScaleNDimIndex);
    auto x2ScaleKDim = x2Scale_->GetViewShape().GetDim(x2ScaleKDimIndex);
    // shape里有1时,x和scale的转置可以不一致,放开转置一致性校验
    bool x1ScaleHasOne = (x1ScaleMDim == 1 && x1ScaleKDim == x1MDim_) || (x1ScaleMDim == x1MDim_ && x1ScaleKDim == 1) ||
                         (x1ScaleMDim == 1 && x1ScaleKDim == CeilDiv(x1KDim_, MXFP_DIVISOR_SIZE)) ||
                         (x1ScaleMDim == CeilDiv(x1KDim_, MXFP_DIVISOR_SIZE) && x1ScaleKDim == 1);
    bool x2ScaleHasOne = (x2ScaleNDim == 1 && x2ScaleKDim == x2NDim_) || (x2ScaleNDim == x2NDim_ && x2ScaleKDim == 1) ||
                         (x2ScaleNDim == 1 && x2ScaleKDim == CeilDiv(x2KDim_, MXFP_DIVISOR_SIZE)) ||
                         (x2ScaleNDim == CeilDiv(x2KDim_, MXFP_DIVISOR_SIZE) && x2ScaleKDim == 1);
    if ((!x1ScaleHasOne && x1ScaleMDim != x1MDim_) || (!x2ScaleHasOne && x2ScaleNDim != x2NDim_)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            apiName_,
            FormatString("x1 M, %s M, x2 N, %s N", GetX1ScaleName().c_str(), GetX2ScaleName().c_str()).c_str(),
            FormatString("%ld, %ld, %ld, %ld", x1MDim_, x1ScaleMDim, x2NDim_, x2ScaleNDim).c_str(),
            FormatString(
                "when the quantization mode is mx, the M dimension of x1 and %s must be equal, and the N dimension "
                "of x2 and %s must be equal",
                GetX1ScaleName().c_str(), GetX2ScaleName().c_str())
                .c_str());
        return false;
    }
    if ((!x1ScaleHasOne && (CeilDiv(x1KDim_, MXFP_DIVISOR_SIZE) != x1ScaleKDim)) ||
        (!x2ScaleHasOne && (CeilDiv(x2KDim_, MXFP_DIVISOR_SIZE) != x2ScaleKDim))) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            apiName_,
            FormatString("x1 K, %s K, x2 K, %s K", GetX1ScaleName().c_str(), GetX2ScaleName().c_str()).c_str(),
            FormatString("%ld, %ld, %ld, %ld", x1KDim_, x1ScaleKDim, x2KDim_, x2ScaleKDim).c_str(),
            FormatString(
                "when the quantization mode is mx, the K dimension of %s must be equal to the K dimension of x1 "
                "ceildivided by 64, and the K dimension of %s must be equal to the K dimension of x2 ceildivided by 64",
                GetX1ScaleName().c_str(), GetX2ScaleName().c_str())
                .c_str());
        return false;
    }
    if (x1Scale_->GetViewShape().GetDim(x1Scale_->GetViewShape().GetDimNum() - 1) != MXFP_MULTI_BASE_SIZE ||
        x2Scale_->GetViewShape().GetDim(x2Scale_->GetViewShape().GetDimNum() - 1) != MXFP_MULTI_BASE_SIZE) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            apiName_,
            FormatString("%s last dimension, %s last dimension", GetX1ScaleName().c_str(), GetX2ScaleName().c_str())
                .c_str(),
            FormatString("%ld, %ld", x1Scale_->GetViewShape().GetDim(x1Scale_->GetViewShape().GetDimNum() - 1),
                         x2Scale_->GetViewShape().GetDim(x2Scale_->GetViewShape().GetDimNum() - 1))
                .c_str(),
            FormatString("when the quantization mode is mx, the last dimension of %s and %s must be 2",
                         GetX1ScaleName().c_str(), GetX2ScaleName().c_str())
                .c_str());
        return false;
    }
    if (IsFp4Input(x1_, x2_)) {
        CHECK_RET(CheckMXFP4FP8ParamsNDOrNZ(), false);
    }
    return true;
}

bool QuantMatmulChecker::CheckGroupSizeShape(uint64_t groupSizeM) const
{
    auto x1DimNum = x1_->GetViewShape().GetDimNum();
    auto x2DimNum = x2_->GetViewShape().GetDimNum();
    auto x1Shape = x1_->GetViewShape();
    auto x1ScaShape = x1Scale_->GetViewShape();
    auto x2Shape = x2_->GetViewShape();
    auto x2ScaShape = x2Scale_->GetViewShape();
    auto mDimNum = x1DimNum - (transposeX1_ ? 1 : PENULTIMATE_DIM);
    auto x1KDimNum = x1DimNum - (transposeX1_ ? PENULTIMATE_DIM : 1);
    auto nDimNum = x2DimNum - (transposeX2_ ? PENULTIMATE_DIM : 1);
    auto x2KDimNum = x2DimNum - (transposeX2_ ? 1 : PENULTIMATE_DIM);
    // m dim
    if (CeilDiv(x1Shape.GetDim(mDimNum), static_cast<int64_t>(groupSizeM)) != x1ScaShape.GetDim(mDimNum)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            apiName_, FormatString("x1 m, %s m, groupSizeM", GetX1ScaleName().c_str()).c_str(),
            FormatString("%ld, %ld, %lu", x1Shape.GetDim(mDimNum), x1ScaShape.GetDim(mDimNum), groupSizeM).c_str(),
            FormatString("when the quantization mode is G-B or B-B, the m dimension of %s must be equal to "
                         "ceil(m_x1 / groupSizeM)",
                         GetX1ScaleName().c_str())
                .c_str());
        return false;
    }
    if (CeilDiv(x1Shape.GetDim(x1KDimNum), static_cast<int64_t>(PERBLOCK_BLOCK_SIZE)) != x1ScaShape.GetDim(x1KDimNum)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            apiName_, FormatString("x1 k, %s k", GetX1ScaleName().c_str()).c_str(),
            FormatString("%ld, %ld", x1Shape.GetDim(x1KDimNum), x1ScaShape.GetDim(x1KDimNum)).c_str(),
            FormatString("when the quantization mode is G-B or B-B, the k dimension of %s must be equal to "
                         "ceil(k_x1 / 128)",
                         GetX1ScaleName().c_str())
                .c_str());
        return false;
    }
    if (CeilDiv(x2Shape.GetDim(nDimNum), static_cast<int64_t>(PERBLOCK_BLOCK_SIZE)) != x2ScaShape.GetDim(nDimNum)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            apiName_, FormatString("x2 n, %s n", GetX2ScaleName().c_str()).c_str(),
            FormatString("%ld, %ld", x2Shape.GetDim(nDimNum), x2ScaShape.GetDim(nDimNum)).c_str(),
            FormatString("when the quantization mode is G-B or B-B, the n dimension of %s must be equal to "
                         "ceil(n_x2 / 128)",
                         GetX2ScaleName().c_str())
                .c_str());
        return false;
    }
    if (CeilDiv(x2Shape.GetDim(x2KDimNum), static_cast<int64_t>(PERBLOCK_BLOCK_SIZE)) != x2ScaShape.GetDim(x2KDimNum)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            apiName_, FormatString("x2 k, %s k", GetX2ScaleName().c_str()).c_str(),
            FormatString("%ld, %ld", x2Shape.GetDim(x2KDimNum), x2ScaShape.GetDim(x2KDimNum)).c_str(),
            FormatString("when the quantization mode is G-B or B-B, the k dimension of %s must be equal to "
                         "ceil(k_x2 / 128)",
                         GetX2ScaleName().c_str())
                .c_str());
        return false;
    }
    return true;
}

bool QuantMatmulChecker::ReCalcGroupSize(int64_t inputSize, int64_t scaleSize, uint64_t& groupSize,
                                         const char* dimensionName) const
{
    if (scaleSize == 0ULL) {
        std::string scaleName = strcmp(dimensionName, "n") == 0 ? "x2Scale(scale)" : "x1Scale(pertokenScale)";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            apiName_, FormatString("%s dimension of %s", dimensionName, scaleName.c_str()).c_str(),
            std::to_string(scaleSize).c_str(),
            FormatString("the %s dimension of %s must be positive", dimensionName, scaleName.c_str()).c_str());
        return false;
    }
    if (groupSize == 0ULL) {
        if (inputSize % scaleSize != 0UL) {
            std::string scaleName = "x1Scale(pertokenScale)";
            std::string inputName = "x1";
            if (strcmp(dimensionName, "n") == 0) {
                scaleName = "x2Scale(scale)";
                inputName = "x2";
            }
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                apiName_,
                FormatString("groupSize, %s dimension of %s, %s dimension of %s", dimensionName, inputName.c_str(),
                             dimensionName, scaleName.c_str())
                    .c_str(),
                FormatString("%lu, %ld, %ld", groupSize, inputSize, scaleSize).c_str(),
                FormatString("when groupSize in %s dimension is 0, the %s dimension of %s must be divisible by the "
                             "%s dimension of %s",
                             dimensionName, dimensionName, inputName.c_str(), dimensionName, scaleName.c_str())
                    .c_str());
            return false;
        }
        groupSize = inputSize / scaleSize;
    }
    return true;
}

bool QuantMatmulChecker::InferGroupSizeM(const aclTensor* x1, const aclTensor* x1Scale, const aclTensor* x2Scale,
                                         bool transX1, uint64_t& groupSizeM) const
{
    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x1ScaleDimNum = x1Scale->GetViewShape().GetDimNum();
    auto inputSizeM = transX1 ? x1->GetViewShape().GetDim(x1DimNum - 1) :
                                x1->GetViewShape().GetDim(x1DimNum - PENULTIMATE_DIM);
    auto scaleSizeM = 0L;
    if (IsMicroScaling(x1Scale, x2Scale)) {
        size_t batchDimNum = x1Scale->GetViewShape().GetDimNum() - MX_SCALE_DIM;
        scaleSizeM = x1Scale->GetViewShape().GetDim(batchDimNum + (transX1 ? 1 : 0));
    } else {
        scaleSizeM = transX1 ? x1Scale->GetViewShape().GetDim(x1ScaleDimNum - 1) :
                               x1Scale->GetViewShape().GetDim(x1ScaleDimNum - PENULTIMATE_DIM);
    }
    return ReCalcGroupSize(inputSizeM, scaleSizeM, groupSizeM, "m");
}

bool QuantMatmulChecker::InferGroupSizeK(const aclTensor* x1, const aclTensor* x1Scale, const aclTensor* x2Scale,
                                         bool transX1, uint64_t& groupSizeK) const
{
    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x1ScaleDimNum = x1Scale->GetViewShape().GetDimNum();
    auto inputSizeK = transX1 ? x1->GetViewShape().GetDim(x1DimNum - PENULTIMATE_DIM) :
                                x1->GetViewShape().GetDim(x1DimNum - 1);
    auto scaleSizeK = 0L;
    if (IsMicroScaling(x1Scale, x2Scale)) {
        // when scale type is e8m0, scalex1 shape is [m, k/2, 2] or [k/2, m, 2]
        size_t batchDimNum = x1Scale->GetViewShape().GetDimNum() - MX_SCALE_DIM;
        scaleSizeK = x1Scale->GetViewShape().GetDim(batchDimNum + (transX1 ? 0 : 1)) * MICRO_SCALING_ALIGN_NUM;
    } else {
        scaleSizeK = transX1 ? x1Scale->GetViewShape().GetDim(x1ScaleDimNum - PENULTIMATE_DIM) :
                               x1Scale->GetViewShape().GetDim(x1ScaleDimNum - 1);
    }
    return ReCalcGroupSize(inputSizeK, scaleSizeK, groupSizeK, "k");
}

bool QuantMatmulChecker::InferGroupSizeN(const aclTensor* x2, const aclTensor* x1Scale, const aclTensor* x2Scale,
                                         bool transX2, uint64_t groupSizeK, uint64_t& groupSizeN) const
{
    auto x2DimNum = x2->GetViewShape().GetDimNum();
    auto x2ScaleDimNum = x2Scale->GetViewShape().GetDimNum();
    auto inputSizeN = transX2 ? x2->GetViewShape().GetDim(x2DimNum - PENULTIMATE_DIM) :
                                x2->GetViewShape().GetDim(x2DimNum - 1);
    auto scaleSizeN = 0L;
    if (IsMicroScaling(x1Scale, x2Scale)) {
        size_t batchDimNum = x2Scale->GetViewShape().GetDimNum() - MX_SCALE_DIM;
        scaleSizeN = x2Scale->GetViewShape().GetDim(batchDimNum + (transX2 ? 0 : 1));
        auto scaleSizeDim0 = x2Scale->GetViewShape().GetDim(batchDimNum);
        auto scaleSizeDim1 = x2Scale->GetViewShape().GetDim(batchDimNum + 1);
        if ((scaleSizeDim0 == 1 && scaleSizeDim1 == inputSizeN) ||
            (scaleSizeDim0 == inputSizeN && scaleSizeDim1 == 1) ||
            (inputSizeN == 1 && (scaleSizeDim0 == 1 || scaleSizeDim1 == 1))) {
            scaleSizeN = inputSizeN;
        }
    } else {
        if (IsA4W4PergroupNonSymmetric(groupSizeK)) {
            scaleSizeN = transX2 ? x2Scale->GetViewShape().GetDim(x2ScaleDimNum - 1) :
                                   x2Scale->GetViewShape().GetDim(x2ScaleDimNum - PENULTIMATE_DIM);
        } else {
            scaleSizeN = transX2 ? x2Scale->GetViewShape().GetDim(x2ScaleDimNum - PENULTIMATE_DIM) :
                                   x2Scale->GetViewShape().GetDim(x2ScaleDimNum - 1);
        }
    }
    return ReCalcGroupSize(inputSizeN, scaleSizeN, groupSizeN, "n");
}

bool QuantMatmulChecker::InferGroupSize(int64_t& groupSize)
{
    auto& x1 = std::get<INDEX_X1_IN_INPUT_TUPLE>(inputTensors_);
    auto& x2 = std::get<INDEX_X2_IN_INPUT_TUPLE>(inputTensors_);
    auto& x1Scale = std::get<INDEX_X1_SCALE_IN_QUANT_TUPLE>(quantTensors_);
    auto& x2Scale = std::get<INDEX_X2_SCALE_IN_QUANT_TUPLE>(quantTensors_);
    // when x1Scale and x2Scale dim num is less than 2, groupsize not used
    if (x1Scale == nullptr || x1Scale->GetViewShape().GetDimNum() < 2 || x2Scale->GetViewShape().GetDimNum() < 2) {
        return true;
    }
    auto transX1 = std::get<INDEX_X1_IN_INPUT_TUPLE>(boolsTrans_);
    auto transX2 = std::get<INDEX_X2_IN_INPUT_TUPLE>(boolsTrans_);
    auto groupSizeMnk = DecodeGroupSizeMnk(groupSize);

    CHECK_RET(InferGroupSizeM(x1, x1Scale, x2Scale, transX1, groupSizeMnk.m), false);
    CHECK_RET(InferGroupSizeK(x1, x1Scale, x2Scale, transX1, groupSizeMnk.k), false);
    CHECK_RET(InferGroupSizeN(x2, x1Scale, x2Scale, transX2, groupSizeMnk.k, groupSizeMnk.n), false);

    OP_LOGD("Inferred groupSize: groupSizeM: %lu, groupSizeN: %lu, groupSizeK: %lu.", groupSizeMnk.m, groupSizeMnk.n,
            groupSizeMnk.k);
    groupSize = EncodeGroupSizeMnk(groupSizeMnk);
    return true;
}

bool QuantMatmulChecker::CheckDimValuePerblock() const
{
    auto x1DimNum = x1_->GetViewShape().GetDimNum();
    auto x2DimNum = x2_->GetViewShape().GetDimNum();
    auto x1ScaleDimNum = x1Scale_->GetViewShape().GetDimNum();
    auto x2ScaleDimNum = x2Scale_->GetViewShape().GetDimNum();
    uint64_t groupSizeM = std::max((static_cast<uint64_t>(groupSize_) >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE, 1UL);
    if (x1ScaleDimNum != x1DimNum || x2ScaleDimNum != x2DimNum) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            apiName_, FormatString("x1, %s, x2, %s", GetX1ScaleName().c_str(), GetX2ScaleName().c_str()).c_str(),
            FormatString("%zuD, %zuD, %zuD, %zuD", x1DimNum, x1ScaleDimNum, x2DimNum, x2ScaleDimNum).c_str(),
            FormatString(
                "when the quantization mode is G-B or B-B, the shape dim of %s must be the same as x1, and the "
                "shape dim of %s must be the same as x2",
                GetX1ScaleName().c_str(), GetX2ScaleName().c_str())
                .c_str());
        return false;
    }
    for (int64_t index = x1DimNum - BATCH_TAILENDER_DIM; index >= 0; --index) {
        if (x1Scale_->GetViewShape().GetDim(index) != x1_->GetViewShape().GetDim(index)) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                apiName_, FormatString("x1 batch dimension, %s batch dimension", GetX1ScaleName().c_str()).c_str(),
                FormatString("%ld, %ld", x1_->GetViewShape().GetDim(index), x1Scale_->GetViewShape().GetDim(index))
                    .c_str(),
                FormatString("when the quantization mode is G-B or B-B, the batch dimensions of %s must be equal to x1",
                             GetX1ScaleName().c_str())
                    .c_str());
            return false;
        }
    }
    for (int64_t index = x2DimNum - BATCH_TAILENDER_DIM; index >= 0; --index) {
        if (x2Scale_->GetViewShape().GetDim(index) != x2_->GetViewShape().GetDim(index)) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                apiName_, FormatString("x2 batch dimension, %s batch dimension", GetX2ScaleName().c_str()).c_str(),
                FormatString("%ld, %ld", x2_->GetViewShape().GetDim(index), x2Scale_->GetViewShape().GetDim(index))
                    .c_str(),
                FormatString("when the quantization mode is G-B or B-B, the batch dimensions of %s must be equal to x2",
                             GetX2ScaleName().c_str())
                    .c_str());
            return false;
        }
    }
    return CheckGroupSizeShape(groupSizeM);
}

bool QuantMatmulChecker::CheckDimValuePertokenDoubleScale() const
{
    auto x1ScaleDim0Size = x1Scale_->GetViewShape().GetDim(0);
    if (x1MDim_ == 1) {
        if (x1ScaleDim0Size != 1) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                apiName_, GetX1ScaleName().c_str(), std::to_string(x1ScaleDim0Size).c_str(),
                FormatString("when %s and %s are 1D and the M dimension of x1 is 1, the shape of %s must be [1]",
                             GetX1ScaleName().c_str(), GetX2ScaleName().c_str(), GetX1ScaleName().c_str())
                    .c_str());
            return false;
        }
    } else {
        if (x1ScaleDim0Size == 1 && !IsInt8Input(x1_, x2_)) { // double scale
            if (x2Scale_->GetViewShape().GetDim(0) != 1 && x2Scale_->GetViewShape().GetDim(0) != x2NDim_) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    apiName_, GetX2ScaleName().c_str(), std::to_string(x2Scale_->GetViewShape().GetDim(0)).c_str(),
                    FormatString("when the shape of %s is [1], the shape of %s must be [1] or [%ld]",
                                 GetX1ScaleName().c_str(), GetX2ScaleName().c_str(), x2NDim_)
                        .c_str());
                return false;
            }
        } else {
            if (x1ScaleDim0Size != x1MDim_) { // pertoken
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    apiName_, GetX1ScaleName().c_str(), std::to_string(x1ScaleDim0Size).c_str(),
                    FormatString("when %s and %s are 1D, the shape of %s must be [%ld]", GetX1ScaleName().c_str(),
                                 GetX2ScaleName().c_str(), GetX1ScaleName().c_str(), x1MDim_)
                        .c_str());
                return false;
            }
        }
    }
    return true;
}

bool QuantMatmulChecker::CheckDimValue() const
{
    if (x1Scale_ != nullptr) {
        if (IsMicroScaling(x1Scale_, x2Scale_)) {
            CHECK_RET(CheckDimValueMicroScaling(), false);
            return true;
        } else if (IsPerblock(x1_, x2_, x1Scale_, x2Scale_)) {
            CHECK_RET(CheckDimValuePerblock(), false);
            return true;
        } else {
            CHECK_RET(CheckDimValuePertokenDoubleScale(), false);
        }
    }
    if (ge::GetPrimaryFormat(x2_->GetStorageFormat()) == Format::FORMAT_FRACTAL_NZ && (x2KDim_ == 1 || x2NDim_ == 1)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            apiName_, "x2 K, x2 N", FormatString("%ld, %ld", x2KDim_, x2NDim_).c_str(),
            "when the format of x2 is FRACTAL_NZ, the last two dimensions of x2 can not be 1");
        return false;
    }

    const uint64_t groupSizeK = static_cast<uint64_t>(groupSize_ & GROUP_MNK_BIT_SIZE);
    bool isA4W4PergroupNonSymmetric = IsA4W4PergroupNonSymmetric(groupSizeK);
    if (!isA4W4PergroupNonSymmetric && x2Scale_->GetViewShape().GetDim(0) != x2NDim_ &&
        x2Scale_->GetViewShape().GetDim(0) != 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            apiName_, GetX2ScaleName().c_str(), std::to_string(x2Scale_->GetViewShape().GetDim(0)).c_str(),
            FormatString(
                "when the quantization mode of x2 is pertensor or perchannel, the last dimension of %s must be "
                "%ld or 1",
                GetX2ScaleName().c_str(), x2NDim_)
                .c_str());
        return false;
    }

    if (x2Offset_ != nullptr) {
        if (!isA4W4PergroupNonSymmetric) {
            CHECK_RET(OpCheckWrongDimension(interfaceType_, X2OFFSET_NAME, x2Offset_, 1UL, apiName_), false);
            if (x2Offset_->GetViewShape().GetDim(0) != x2NDim_ && x2Offset_->GetViewShape().GetDim(0) != 1) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    apiName_, GetX2OffsetName().c_str(), std::to_string(x2Offset_->GetViewShape().GetDim(0)).c_str(),
                    FormatString("the first dimension of %s must be %ld or 1", GetX2OffsetName().c_str(), x2NDim_)
                        .c_str());
                return false;
            }
        }
    }
    return true;
}

int64_t QuantMatmulChecker::InferOutputShape(std::vector<int64_t>& batchRecord) const
{
    int64_t inferredOutbatchValue = 1;
    auto x1DimNum = x1_->GetViewShape().GetDimNum();
    auto x2DimNum = x2_->GetViewShape().GetDimNum();
    auto outDimNum = std::max(x1DimNum, x2DimNum);
    size_t validOffset = outDimNum - std::min(x1DimNum, x2DimNum);
    auto& longShapeTensor = x1DimNum > x2DimNum ? x1_ : x2_;
    auto& shortShapeTensor = x1DimNum > x2DimNum ? x2_ : x1_;
    for (size_t i = 0; i < outDimNum - PENULTIMATE_DIM; i++) {
        auto longDimVal = longShapeTensor->GetViewShape().GetDim(i);
        auto shortDimVal = i < validOffset ? 1 : shortShapeTensor->GetViewShape().GetDim(i - validOffset);
        if (shortDimVal > 1 && longDimVal > 1 && shortDimVal != longDimVal) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(apiName_, "short batch dimension, long batch dimension",
                                                   FormatString("%ld, %ld", shortDimVal, longDimVal).c_str(),
                                                   "the batch dimensions of x1 and x2 must be broadcastable");
            return OUTPUT_INFER_FAIL;
        }
        int64_t curBatchValue = static_cast<int64_t>(std::max(shortDimVal, longDimVal));
        inferredOutbatchValue = inferredOutbatchValue * curBatchValue;
        batchRecord.push_back(curBatchValue);
    }
    return inferredOutbatchValue;
}

bool QuantMatmulChecker::CheckBiasShape(const std::vector<int64_t>& batchRecord, int64_t inferredOutbatchValue) const
{
    auto biasDimNum = bias_->GetViewShape().GetDimNum();
    // 3 is bias with batch dim-num
    if (biasDimNum != 3 && biasDimNum != 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(apiName_, "bias", FormatString("%zuD", biasDimNum).c_str(),
                                                 "the shape dim of bias must be 1D or 3D");
        return false;
    }
    auto biasFirstDim = bias_->GetViewShape().GetDim(0);
    if (biasDimNum == 1) {
        OP_CHECK(
            biasFirstDim == x2NDim_,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(apiName_, "bias", std::to_string(biasFirstDim).c_str(),
                                                  FormatString("the shape of bias must be [%ld]", x2NDim_).c_str()),
            return false);
        return true;
    }
    auto biasSecondDim = bias_->GetViewShape().GetDim(1);
    // 2 is bias last dim index
    auto biasThirdDim = bias_->GetViewShape().GetDim(2);
    // output batch need to be only 1 dim when bias dim is 3
    if (batchRecord.size() != 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            apiName_, "inferred out batch dimension", std::to_string(batchRecord.size()).c_str(),
            "when the shape dim of bias is 3D, the inferred out batch dimension must be 1");
        return false;
    }
    OP_CHECK(biasSecondDim == 1,
             OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                 apiName_, "bias second dimension", std::to_string(biasSecondDim).c_str(),
                 "when the shape dim of bias is 3D, the second dimension of bias must be 1"),
             return false);
    OP_CHECK(
        biasThirdDim == x2NDim_,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            apiName_, "bias last dimension", std::to_string(biasThirdDim).c_str(),
            FormatString("when the shape dim of bias is 3D, the last dimension of bias must be %ld", x2NDim_).c_str()),
        return false);
    OP_CHECK(
        biasFirstDim == inferredOutbatchValue,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            apiName_, "bias first dimension, inferred out batch dimension",
            FormatString("%ld, %ld", biasFirstDim, inferredOutbatchValue).c_str(),
            "when the shape dim of bias is 3D, the first dimension of bias must be equal to the inferred out batch "
            "dimension"),
        return false);
    return true;
}

bool QuantMatmulChecker::CheckOutShape(bool twoDimMatmulCaseFlag, const std::vector<int64_t>& batchRecord) const
{
    auto outDimNum = out_->GetViewShape().GetDimNum();
    int64_t outMDim = out_->GetViewShape().GetDim(outDimNum - PENULTIMATE_DIM);
    int64_t outNDim = out_->GetViewShape().GetDim(outDimNum - 1);
    size_t inferredOutDimNum = batchRecord.size() + 2;
    // x1 and x2 are 2 dim and out is 3 dim speical case
    if (outMDim == 1 && inferredOutDimNum == 2 && outDimNum == 3 && twoDimMatmulCaseFlag) {
        outDimNum -= 1;
        outMDim = out_->GetViewShape().GetDim(0);
    }
    if (inferredOutDimNum != outDimNum) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            apiName_, GetInputName(OUT_NAME, interfaceType_).c_str(), FormatString("%zuD", outDimNum).c_str(),
            FormatString("the shape dim of out must be %zuD", inferredOutDimNum).c_str());
        return false;
    }
    OP_CHECK(outMDim == x1MDim_,
             OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(apiName_, "out M, x1 M",
                                                    FormatString("%ld, %ld", outMDim, x1MDim_).c_str(),
                                                    "the M dimension of out must be equal to the M dimension of x1"),
             return false);
    OP_CHECK(outNDim == x2NDim_,
             OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(apiName_, "out N, x2 N",
                                                    FormatString("%ld, %ld", outNDim, x2NDim_).c_str(),
                                                    "the N dimension of out must be equal to the N dimension of x2"),
             return false);
    for (size_t i = 0; i < outDimNum - PENULTIMATE_DIM; i++) {
        OP_CHECK(
            out_->GetViewShape().GetDim(i) == batchRecord[i],
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                apiName_, FormatString("out batch dimension %zu, inferred output batch dimension %zu", i, i).c_str(),
                FormatString("%ld, %ld", out_->GetViewShape().GetDim(i), batchRecord[i]).c_str(),
                FormatString("the batch dimension %zu of out must be equal to the inferred output batch dimension", i)
                    .c_str()),
            return false);
    }
    return true;
}

bool QuantMatmulChecker::CheckShapeInt4() const
{
    if (!IsAligned<int64_t>(x1KDim_, INT4_NUMS_IN_INT8, apiName_)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName_, "x1", std::to_string(x1KDim_).c_str(),
                                              "in A4W4 scenario, the K dimension of x1 must be positive and even");
        return false;
    }
    if (transposeX2_ && !IsAligned<int64_t>(x2KDim_, INT4_NUMS_IN_INT8, apiName_)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            apiName_, "x2", std::to_string(x2KDim_).c_str(),
            "in A4W4 scenario and transposeX2 is true, the K dimension of x2 must be positive and even");
        return false;
    }
    if (!transposeX2_ && !IsAligned<int64_t>(x2NDim_, INT4_NUMS_IN_INT8, apiName_)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            apiName_, "x2", std::to_string(x2NDim_).c_str(),
            "in A4W4 scenario and transposeX2 is false, the N dimension of x2 must be positive and even");
        return false;
    }
    if (transposeX1_) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName_, "transposeX1", transposeX1_ ? "true" : "false",
                                              "in A4W4 scenario, transposeX1 must be false");
        return false;
    }
    if (x2_->GetViewShape().GetDimNum() != X2_FIXED_DIM_NUM_A4W4) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(apiName_, "x2",
                                                 FormatString("%zuD", x2_->GetViewShape().GetDimNum()).c_str(),
                                                 "in A4W4 scenario, the shape dim of x2 must be 2D");
        return false;
    }
    if (bias_ != nullptr && bias_->GetViewShape().GetDimNum() != 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(apiName_, "bias",
                                                 FormatString("%zuD", bias_->GetViewShape().GetDimNum()).c_str(),
                                                 "in A4W4 scenario, the shape dim of bias must be 1D");
        return false;
    }
    return true;
}

bool QuantMatmulChecker::CheckMxScaleMinDim(size_t x1ScaleDim, size_t x2ScaleDim) const
{
    if (x1ScaleDim < MX_SCALE_MIN_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            apiName_, GetX1ScaleName().c_str(), FormatString("%zuD", x1ScaleDim).c_str(),
            FormatString("when the quantization mode is mx, the shape dim of %s must be at least %zuD",
                         GetX1ScaleName().c_str(), MX_SCALE_MIN_DIM)
                .c_str());
        return false;
    }
    if (x2ScaleDim < MX_SCALE_MIN_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            apiName_, GetX2ScaleName().c_str(), FormatString("%zuD", x2ScaleDim).c_str(),
            FormatString("when the quantization mode is mx, the shape dim of %s must be at least %zuD",
                         GetX2ScaleName().c_str(), MX_SCALE_MIN_DIM)
                .c_str());
        return false;
    }
    return true;
}

bool QuantMatmulChecker::CheckMxScaleBatchDimMatch(size_t x1ScaleDim, size_t x2ScaleDim) const
{
    // batch dim count alignment: scale removes MX fixed 3 dims, x removes last 2 dims
    size_t x1DimNum = x1_->GetViewShape().GetDimNum();
    size_t x2DimNum = x2_->GetViewShape().GetDimNum();
    size_t x1BatchDimNum = x1ScaleDim - MX_SCALE_DIM;
    size_t x2BatchDimNum = x2ScaleDim - MX_SCALE_DIM;
    size_t x1InputBatchDimNum = (x1DimNum > MIN_DIM_NUM_ND) ? (x1DimNum - MIN_DIM_NUM_ND) : 0;
    size_t x2InputBatchDimNum = (x2DimNum > MIN_DIM_NUM_ND) ? (x2DimNum - MIN_DIM_NUM_ND) : 0;
    if (x1BatchDimNum != x1InputBatchDimNum) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            apiName_, "pertokenBatchDimNum, x1BatchDimNum",
            FormatString("%zu, %zu", x1BatchDimNum, x1InputBatchDimNum).c_str(),
            "when the quantization mode is mx, the batch dimension num of x1Scale must be equal to that of x1");
        return false;
    }
    if (x2BatchDimNum != x2InputBatchDimNum) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            apiName_, "scaleBatchDimNum, x2BatchDimNum",
            FormatString("%zu, %zu", x2BatchDimNum, x2InputBatchDimNum).c_str(),
            "when the quantization mode is mx, the batch dimension num of x2Scale must be equal to that of x2");
        return false;
    }
    // per-batch dim value match
    const auto& x1Shape = x1_->GetViewShape();
    const auto& x2Shape = x2_->GetViewShape();
    const auto& x1ScaleShape = x1Scale_->GetViewShape();
    const auto& x2ScaleShape = x2Scale_->GetViewShape();
    for (size_t i = 0; i < x1BatchDimNum; ++i) {
        if (x1ScaleShape.GetDim(i) != x1Shape.GetDim(i)) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                apiName_, "dimIndex, x1Batch, pertokenBatch",
                FormatString("%zu, %ld, %ld", i, x1Shape.GetDim(i), x1ScaleShape.GetDim(i)).c_str(),
                "when the quantization mode is mx, the batch dimension of x1Scale must be equal to that of x1");
            return false;
        }
    }
    for (size_t i = 0; i < x2BatchDimNum; ++i) {
        if (x2ScaleShape.GetDim(i) != x2Shape.GetDim(i)) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                apiName_, "dimIndex, x2Batch, scaleBatch",
                FormatString("%zu, %ld, %ld", i, x2Shape.GetDim(i), x2ScaleShape.GetDim(i)).c_str(),
                "when the quantization mode is mx, the batch dimension of x2Scale must be equal to that of x2");
            return false;
        }
    }
    return true;
}

bool QuantMatmulChecker::CheckMxScaleDimRange(size_t x1ScaleDim, size_t x2ScaleDim) const
{
    if (!CheckMxScaleMinDim(x1ScaleDim, x2ScaleDim)) {
        return false;
    }
    if (!CheckMxScaleBatchDimMatch(x1ScaleDim, x2ScaleDim)) {
        return false;
    }
    return true;
}

bool QuantMatmulChecker::CheckNormalScaleDimRange(size_t x1ScaleDim, size_t x2ScaleDim) const
{
    bool isPerblock = IsPerblock(x1_, x2_, x1Scale_, x2Scale_);
    if ((isPerblock && x1ScaleDim != x1_->GetViewShape().GetDimNum()) || (!isPerblock && x1ScaleDim != 1)) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            apiName_, FormatString("%s, x1", GetX1ScaleName().c_str()).c_str(),
            FormatString("%zuD, %zuD", x1ScaleDim, x1_->GetViewShape().GetDimNum()).c_str(),
            FormatString("when the quantization mode is G-B or B-B, the shape dim of %s must be the same "
                         "as the shape dim of x1; otherwise, the shape dim of %s must be 1D",
                         GetX1ScaleName().c_str(), GetX1ScaleName().c_str())
                .c_str());
        return false;
    }
    if ((isPerblock && x2ScaleDim != x2_->GetViewShape().GetDimNum()) || (!isPerblock && x2ScaleDim != 1)) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            apiName_, FormatString("%s, x2", GetX2ScaleName().c_str()).c_str(),
            FormatString("%zuD, %zuD", x2ScaleDim, x2_->GetViewShape().GetDimNum()).c_str(),
            FormatString("when the quantization mode is G-B or B-B, the shape dim of %s must be the same "
                         "as the shape dim of x2; otherwise, the shape dim of %s must be 1D",
                         GetX2ScaleName().c_str(), GetX2ScaleName().c_str())
                .c_str());
        return false;
    }
    return true;
}

bool QuantMatmulChecker::CheckScaleDimRange() const
{
    if (x1Scale_ == nullptr) { // pertensor / perchannel
        OP_CHECK_WRONG_DIMENSION(x2Scale_, 1, return false);
        OP_LOGD("QuantMatmul check dimension range success.");
        return true;
    }

    auto x1ScaleDim = x1Scale_->GetViewShape().GetDimNum();
    auto x2ScaleDim = x2Scale_->GetViewShape().GetDimNum();
    const uint64_t groupSizeK = static_cast<uint64_t>(groupSize_ & GROUP_MNK_BIT_SIZE);
    if (IsMicroScaling(x1Scale_, x2Scale_)) {
        CHECK_RET(CheckMxScaleDimRange(x1ScaleDim, x2ScaleDim), false);
    } else if (IsA4W4PergroupNonSymmetric(groupSizeK)) {
        return true;
    } else {
        CHECK_RET(CheckNormalScaleDimRange(x1ScaleDim, x2ScaleDim), false);
    }
    OP_LOGD("QuantMatmul check dimension range success.");
    return true;
}

bool QuantMatmulChecker::CheckGroupSize() const
{
    if (npuArch_ != NpuArch::DAV_3510) {
        return true;
    }
    const GroupSizeMNK groupSizeMnk = DecodeGroupSizeMnk(groupSize_);
    if (IsA4W4PergroupNonSymmetric(groupSizeMnk.k)) {
        CHECK_RET(CheckA4W4PergroupNonSymmetricGroupSize(groupSizeMnk, apiName_), false);
    } else if (IsMicroScaling(x1Scale_, x2Scale_)) {
        CHECK_RET(CheckMxGroupSize(groupSize_, groupSizeMnk, apiName_), false);
    } else if (IsPerblock(x1_, x2_, x1Scale_, x2Scale_)) {
        CHECK_RET(CheckPerblockGroupSize(groupSizeMnk, apiName_), false);
    } else if (groupSize_ != 0UL) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            apiName_, "groupSize, groupSizeM, groupSizeN, groupSizeK",
            FormatString("%ld, %lu, %lu, %lu", groupSize_, groupSizeMnk.m, groupSizeMnk.n, groupSizeMnk.k).c_str(),
            "when the quantization mode is other than G-B, B-B, or mx, groupSize must be 0 and Torch API "
            "group_sizes must be [0, 0, 0] or None");
        return false;
    }
    OP_LOGD("QuantMatmul check group_size success.");
    return true;
}

bool QuantMatmulChecker::CheckShape() const
{
    auto x1DimNum = x1_->GetViewShape().GetDimNum();
    auto x2DimNum = x2_->GetViewShape().GetDimNum();

    CHECK_RET(CheckScaleDimRange(), false);
    CHECK_RET(CheckGroupSize(), false);
    if (isA4W4_ && !CheckShapeInt4()) {
        return false;
    }
    OP_CHECK(x1KDim_ == x2KDim_,
             OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(apiName_, "x1 K, x2 K",
                                                    FormatString("%ld, %ld", x1KDim_, x2KDim_).c_str(),
                                                    "the K dimension of x1 and x2 must be equal"),
             return false);

    if (ge::GetPrimaryFormat(x2_->GetStorageFormat()) == Format::FORMAT_FRACTAL_NZ) {
        CHECK_RET(CheckShapeForWeightNz(), false);
    }
    CHECK_RET(CheckDimValue(), false);

    std::vector<int64_t> batchRecord;
    int64_t inferredOutbatchValue = InferOutputShape(batchRecord);
    if (inferredOutbatchValue == OUTPUT_INFER_FAIL) {
        return false;
    }
    if (bias_ != nullptr) {
        if (!CheckBiasShape(batchRecord, inferredOutbatchValue)) {
            return false;
        }
    }
    bool twoDimMatmulCaseFlag = x1DimNum == x2DimNum && x2DimNum == 2;
    CHECK_RET(CheckOutShape(twoDimMatmulCaseFlag, batchRecord), false);
    return true;
}

bool QuantMatmulChecker::CheckFormatInt4() const
{
    if (x1_->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(apiName_, "x1", op::ToString(x1_->GetStorageFormat()).GetString(),
                                               "in A4W4 scenario, the format of x1 must be ND");
        return false;
    }
    if (isWeightNz_ && x2_->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
            apiName_, "x2", op::ToString(x2_->GetStorageFormat()).GetString(),
            "in A4W4 scenario and weight_nz_flag is true, the format of x2 must be FRACTAL_NZ");
        return false;
    } else if (!isWeightNz_ && x2_->GetStorageFormat() != op::Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(apiName_, "x2", op::ToString(x2_->GetStorageFormat()).GetString(),
                                               "in A4W4 scenario, the format of x2 must be ND");
        return false;
    }
    return true;
}

bool QuantMatmulChecker::CheckWeightNzAbDtypes() const
{
    if (!(x1_->GetDataType() == op::DataType::DT_INT8 || IsHif8Input(x1_, x2_) ||
          (x1_->GetDataType() == op::DataType::DT_FLOAT8_E4M3FN &&
           x2_->GetDataType() == op::DataType::DT_FLOAT8_E4M3FN) ||
          (x1_->GetDataType() == op::DataType::DT_FLOAT4_E2M1 && x2_->GetDataType() == op::DataType::DT_FLOAT4_E2M1))) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(apiName_, "x1, x2",
                                               FormatString("%s, %s", op::ToString(x1_->GetDataType()).GetString(),
                                                            op::ToString(x2_->GetDataType()).GetString())
                                                   .c_str(),
                                               "when the format of x2 is FRACTAL_NZ, the dtypes of x1 and x2 must be "
                                               "INT8/HIFLOAT8/FLOAT8_E4M3FN/FLOAT4_E2M1");
        return false;
    }
    return true;
}

bool QuantMatmulChecker::CheckWeightNzDtype4Fp8E4M3() const
{
    bool isStaticX2Scale = x1Scale_ == nullptr && x2Scale_ != nullptr &&
                           (x2Scale_->GetDataType() == op::DataType::DT_UINT64 ||
                            x2Scale_->GetDataType() == op::DataType::DT_INT64);
    bool isPerblockFloatScale = IsPerblock(x1_, x2_, x1Scale_, x2Scale_);
    bool isFloatScale = x1Scale_ != nullptr && x2Scale_ != nullptr &&
                        x1Scale_->GetDataType() == op::DataType::DT_FLOAT &&
                        x2Scale_->GetDataType() == op::DataType::DT_FLOAT;
    if (!IsMicroScaling(x1Scale_, x2Scale_) && !isFloatScale && !isStaticX2Scale && !isPerblockFloatScale) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            apiName_, FormatString("%s, %s", GetX1ScaleName().c_str(), GetX2ScaleName().c_str()).c_str(),
            FormatString("%s, %s", x1Scale_ == nullptr ? "null" : op::ToString(x1Scale_->GetDataType()).GetString(),
                         op::ToString(x2Scale_->GetDataType()).GetString())
                .c_str(),
            "when the format of x2 is FRACTAL_NZ and input dtype is FLOAT8_E4M3FN, x1Scale and x2Scale must be "
            "FLOAT8_E8M0 for mx quantization, or both be FLOAT for G-B/B-B/K-C/K-T quantization, or x1Scale must be "
            "null "
            "and x2Scale must exist and its dtype must be UINT64/INT64");
        return false;
    }
    return true;
}

bool QuantMatmulChecker::CheckWeightNzDtype4Fp4() const
{
    if (!IsMicroScaling(x1Scale_, x2Scale_)) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            apiName_, FormatString("%s, %s", GetX1ScaleName().c_str(), GetX2ScaleName().c_str()).c_str(),
            FormatString("%s, %s", x1Scale_ == nullptr ? "null" : op::ToString(x1Scale_->GetDataType()).GetString(),
                         op::ToString(x2Scale_->GetDataType()).GetString())
                .c_str(),
            "when the format of x2 is FRACTAL_NZ and input dtype is FLOAT4_E2M1, the dtypes of x1Scale and x2Scale "
            "must be FLOAT8_E8M0");
        return false;
    }
    return true;
}

bool QuantMatmulChecker::CheckWeightNzDtype4Hifloat8() const
{
    bool isStaticX2Scale = x1Scale_ == nullptr && x2Scale_ != nullptr &&
                           (x2Scale_->GetDataType() == op::DataType::DT_UINT64 ||
                            x2Scale_->GetDataType() == op::DataType::DT_INT64);
    bool isPerblockFloatScale = IsPerblock(x1_, x2_, x1Scale_, x2Scale_);
    bool isFloatScale = x1Scale_ != nullptr && x2Scale_ != nullptr &&
                        x1Scale_->GetDataType() == op::DataType::DT_FLOAT &&
                        x2Scale_->GetDataType() == op::DataType::DT_FLOAT;
    if (isPerblockFloatScale || isFloatScale) {
        return true;
    }
    if (x1Scale_ != nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            apiName_, GetX1ScaleName().c_str(), "not null",
            "when the format of x2 is FRACTAL_NZ and input dtype is HIFLOAT8, x1Scale must be null "
            "for static quantization, or x1Scale and x2Scale must both be FLOAT for G-B/B-B/K-C/K-T quantization");
        return false;
    }
    if (!isStaticX2Scale) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            apiName_, GetX2ScaleName().c_str(), op::ToString(x2Scale_->GetDataType()).GetString(),
            "when the format of x2 is FRACTAL_NZ and input dtype is HIFLOAT8, the dtype of x2Scale must be "
            "UINT64 or INT64 for static quantization, or FLOAT for G-B/B-B quantization");
        return false;
    }
    return true;
}

bool QuantMatmulChecker::CheckDtype4WeightNz() const
{
    if (!CheckWeightNzAbDtypes()) {
        return false;
    }

    if (IsHif8Input(x1_, x2_)) {
        return CheckWeightNzDtype4Hifloat8();
    }

    if (x1_->GetDataType() == op::DataType::DT_FLOAT4_E2M1 && x2_->GetDataType() == op::DataType::DT_FLOAT4_E2M1) {
        return CheckWeightNzDtype4Fp4();
    }

    if (x1_->GetDataType() == op::DataType::DT_FLOAT8_E4M3FN && x2_->GetDataType() == op::DataType::DT_FLOAT8_E4M3FN) {
        return CheckWeightNzDtype4Fp8E4M3();
    }

    return true;
}

bool QuantMatmulChecker::CheckFormat() const
{
    auto x2StorageFormat = ge::GetPrimaryFormat(x2_->GetStorageFormat());
    if (x1Scale_ != nullptr && npuArch_ == NpuArch::DAV_2002) {
        OP_LOGD("QuantMatmul pertoken mode in Arch 2002: x1 should be FORMAT_FRACTAL_NZ");
    } else {
        CHECK_RET(x1_->GetStorageFormat() == op::Format::FORMAT_ND, false);
    }
    if (npuArch_ == NpuArch::DAV_2002) {
        CHECK_RET(x2_->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ, false);
    } else {
        CHECK_RET(x2StorageFormat == op::Format::FORMAT_ND || x2StorageFormat == op::Format::FORMAT_FRACTAL_NZ, false);
    }
    if (x1Scale_ != nullptr) {
        CHECK_RET(x1Scale_->GetStorageFormat() == op::Format::FORMAT_ND, false);
    }
    CHECK_RET(x2Scale_->GetStorageFormat() == op::Format::FORMAT_ND, false);
    if (x2Offset_ != nullptr) {
        CHECK_RET(x2Offset_->GetStorageFormat() == op::Format::FORMAT_ND, false);
    }
    if (bias_ != nullptr) {
        CHECK_RET(bias_->GetStorageFormat() == op::Format::FORMAT_ND, false);
    }
    if (isA4W4_) {
        CHECK_RET(CheckFormatInt4(), false);
    }
    return true;
}

bool QuantMatmulChecker::CheckL0c2outPertensorPerchannel() const
{
    if (!(x2Scale_->GetDataType() == op::DataType::DT_FLOAT || x2Scale_->GetDataType() == op::DataType::DT_BF16)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            apiName_, GetX2ScaleName().c_str(), op::ToString(x2Scale_->GetDataType()).GetString(),
            FormatString("when the dtype of out is INT32, the dtype of %s must be FLOAT or BFLOAT16",
                         GetX2ScaleName().c_str())
                .c_str());
        return false;
    }
    if (bias_ != nullptr && bias_->GetDataType() != op::DataType::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(apiName_, GetInputName(BIAS_NAME, interfaceType_).c_str(),
                                              op::ToString(bias_->GetDataType()).GetString(),
                                              "when the dtype of out is INT32, the dtype of bias must be INT32");
        return false;
    }
    return true;
}

bool QuantMatmulChecker::CheckFloat16OutBiasAndOffset() const
{
    if (bias_ != nullptr) {
        CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, BIAS_NAME, bias_, op::DataType::DT_INT32, apiName_), false);
    }
    if (x2Offset_ != nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            apiName_, GetX2OffsetName().c_str(), "not null",
            FormatString("when the dtype of out is FLOAT16, %s must be null", GetX2OffsetName().c_str()).c_str());
        return false;
    }
    return true;
}

bool QuantMatmulChecker::CheckL0c2outAndL0c2ubPertensorPerchannel() const
{
    CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X1_NAME, x1_, op::DataType::DT_INT8, apiName_), false);
    CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X2_NAME, x2_, op::DataType::DT_INT8, apiName_), false);
    if (outType_ == op::DataType::DT_INT32) {
        CHECK_RET(CheckL0c2outPertensorPerchannel(), false);
    } else if (outType_ == op::DataType::DT_BF16) {
        CHECK_RET(
            OpCheckDtypeNotSupport(interfaceType_, X2SCALE_NAME, x2Scale_, BF16_OUT_X2SCALE_SUPPORT_LIST, apiName_),
            false);
        if (groupSize_ != 0UL) {
            CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X2SCALE_NAME, x2Scale_, op::DataType::DT_BF16, apiName_),
                      false);
        }
        if (bias_ != nullptr) {
            CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, BIAS_NAME, bias_, BIAS_TYPE_SUPPORT_LIST, apiName_),
                      false);
        }
        if (x2Offset_ != nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                apiName_, GetX2OffsetName().c_str(), "not null",
                FormatString("when the dtype of out is BFLOAT16, %s must be null", GetX2OffsetName().c_str()).c_str());
            return false;
        }
    } else if (outType_ == op::DataType::DT_INT8) {
        CHECK_RET(
            OpCheckDtypeNotSupport(interfaceType_, X2SCALE_NAME, x2Scale_, UINT64_X2_SCALE_TYPE_SUPPORT_LIST, apiName_),
            false);
        if (bias_ != nullptr) {
            CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, BIAS_NAME, bias_, op::DataType::DT_INT32, apiName_), false);
        }
        if (x2Offset_ != nullptr) {
            CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X2OFFSET_NAME, x2Offset_, op::DataType::DT_FLOAT, apiName_),
                      false);
        }
    } else if (outType_ == op::DataType::DT_FLOAT16) {
        CHECK_RET(
            OpCheckDtypeNotSupport(interfaceType_, X2SCALE_NAME, x2Scale_, UINT64_X2_SCALE_TYPE_SUPPORT_LIST, apiName_),
            false);
        CHECK_RET(CheckFloat16OutBiasAndOffset(), false);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(apiName_, GetInputName(OUT_NAME, interfaceType_).c_str(),
                                              RemoveDtInDtype(op::ToString(outType_).GetString()).c_str(),
                                              "the dtype of out must be INT32, BFLOAT16, INT8 or FLOAT16");
        return false;
    }
    return true;
}

bool QuantMatmulChecker::CheckOnlyL0c2outPertoken() const
{
    CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, X2_NAME, x2_, X1_X2_L0C2OUT_PERTOKEN_SUPPORT_LIST, apiName_),
              false);
    CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X1SCALE_NAME, x1Scale_, op::DataType::DT_FLOAT, apiName_), false);
    if (outType_ == op::DataType::DT_BF16) {
        CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, X2SCALE_NAME, x2Scale_, PERTOKEN_BF16_OUT_X2SCALE_SUPPORT_LIST,
                                         apiName_),
                  false);
        if (bias_ != nullptr) {
            CHECK_RET(
                OpCheckDtypeNotSupport(interfaceType_, BIAS_NAME, bias_, PERTOKEN_BF16_OUT_BIAS_SUPPORT_LIST, apiName_),
                false);
        }
    } else if (outType_ == op::DataType::DT_FLOAT16) {
        CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X2SCALE_NAME, x2Scale_, op::DataType::DT_FLOAT, apiName_),
                  false);
        if (bias_ != nullptr) {
            CHECK_RET(
                OpCheckDtypeNotSupport(interfaceType_, BIAS_NAME, bias_, PERTOKEN_FP16_OUT_BIAS_SUPPORT_LIST, apiName_),
                false);
        }
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            apiName_, GetInputName(OUT_NAME, interfaceType_).c_str(),
            RemoveDtInDtype(op::ToString(outType_).GetString()).c_str(),
            FormatString("when x1/x2 are INT8/INT4 and %s, %s are FLOAT, the dtype of out must be BFLOAT16 or FLOAT16",
                         GetX1ScaleName().c_str(), GetX2ScaleName().c_str())
                .c_str());
        return false;
    }
    if (x2Offset_ != nullptr) {
        if (x1_->GetDataType() == op::DataType::DT_INT8 && x2_->GetDataType() == op::DataType::DT_INT8 &&
            x1Scale_->GetDataType() != op::DataType::DT_FLOAT && x2Scale_->GetDataType() != op::DataType::DT_FLOAT) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                apiName_, GetX2OffsetName().c_str(), "not null",
                FormatString("when x1 and x2 are INT8 and the dtypes of %s and %s are outside FLOAT, %s must be null",
                             GetX1ScaleName().c_str(), GetX2ScaleName().c_str(), GetX2OffsetName().c_str())
                    .c_str());
            return false;
        }
    }
    return true;
}

bool QuantMatmulChecker::CheckDtypeValidOnOnlyL0c2outForA4W4() const
{
    if (x1_->GetDataType() != op::DataType::DT_INT4 || x2_->GetDataType() != op::DataType::DT_INT4) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(apiName_, "x1, x2",
                                               FormatString("%s, %s", op::ToString(x1_->GetDataType()).GetString(),
                                                            op::ToString(x2_->GetDataType()).GetString())
                                                   .c_str(),
                                               "in A4W4 scenario, the dtypes of x1 and x2 must be INT4");
        return false;
    }
    if (x2Scale_->GetDataType() != op::DataType::DT_UINT64 && x2Scale_->GetDataType() != op::DataType::DT_INT64) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            apiName_, GetX2ScaleName().c_str(), op::ToString(x2Scale_->GetDataType()).GetString(),
            FormatString("in A4W4 scenario and %s is null, the dtype of %s must be UINT64 or INT64",
                         GetX1ScaleName().c_str(), GetX2ScaleName().c_str())
                .c_str());
        return false;
    }
    if (bias_ != nullptr && bias_->GetDataType() != op::DataType::DT_INT32) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            apiName_, GetInputName(BIAS_NAME, interfaceType_).c_str(), op::ToString(bias_->GetDataType()).GetString(),
            FormatString("in A4W4 scenario and %s is null, the dtype of bias must be INT32", GetX1ScaleName().c_str())
                .c_str());
        return false;
    }
    if (out_->GetDataType() != op::DataType::DT_FLOAT16 && out_->GetDataType() != op::DataType::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(apiName_, GetInputName(OUT_NAME, interfaceType_).c_str(),
                                              op::ToString(out_->GetDataType()).GetString(),
                                              "in A4W4 scenario, the dtype of out must be FLOAT16 or BFLOAT16");
        return false;
    }
    return true;
}

aclnnStatus QuantMatmulChecker::CheckDtypeOnlyL0c2ub() const
{
    if (isA4W4_) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "QuantBatchMatmul support for %s is not implemented in a4w4 scenario.",
                op::ToString(socVersion_).GetString());
    }
    if (x1Scale_ != nullptr) {
        CHECK_RET(CheckOnlyL0c2ubPertoken(), ACLNN_ERR_PARAM_INVALID);
    } else {
        CHECK_RET(CheckL0c2outAndL0c2ubPertensorPerchannel(), ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus QuantMatmulChecker::CheckDtypeOnlyL0c2out() const
{
    if (isA4W4_ && x1Scale_ == nullptr) { // pertensor/perchannel
        CHECK_RET(CheckDtypeValidOnOnlyL0c2outForA4W4(), ACLNN_ERR_PARAM_INVALID);
    } else if (!isA4W4_ && x1Scale_ == nullptr) { // pertensor/perchannel
        CHECK_RET(CheckL0c2outAndL0c2ubPertensorPerchannel(), ACLNN_ERR_PARAM_INVALID);
    } else if (x1_->GetDataType() == op::DataType::DT_INT8 || (isA4W4_ && x1Scale_ != nullptr)) { // pertoken
        CHECK_RET(CheckOnlyL0c2outPertoken(), ACLNN_ERR_PARAM_INVALID);
    } else {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            apiName_, FormatString("x1, x2, %s, %s, out", GetX1ScaleName().c_str(), GetX2ScaleName().c_str()).c_str(),
            FormatString("%s, %s, %s, %s, %s", op::ToString(x1_->GetDataType()).GetString(),
                         op::ToString(x2_->GetDataType()).GetString(),
                         x1Scale_ == nullptr ? "null" : op::ToString(x1Scale_->GetDataType()).GetString(),
                         x2Scale_ == nullptr ? "null" : op::ToString(x2Scale_->GetDataType()).GetString(),
                         op::ToString(outType_).GetString())
                .c_str(),
            "the quantization mode determined by x1, x2, x1Scale, x2Scale and out must be supported");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

bool QuantMatmulChecker::CheckL0c2outOrL0c2ubPertensorPerchannel4Int8Input() const
{
    if (outType_ == op::DataType::DT_BF16) {
        CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, X2SCALE_NAME, x2Scale_, X2_INT8_X2_SCALE_TYPE_SUPPORT_LIST,
                                         apiName_),
                  false);
    } else if (outType_ == op::DataType::DT_INT8 || outType_ == op::DataType::DT_FLOAT16) {
        CHECK_RET(
            OpCheckDtypeNotSupport(interfaceType_, X2SCALE_NAME, x2Scale_, UINT64_X2_SCALE_TYPE_SUPPORT_LIST, apiName_),
            false);
    } else if (outType_ == op::DataType::DT_INT32) {
        CHECK_RET(
            OpCheckDtypeNotSupport(interfaceType_, X2SCALE_NAME, x2Scale_, INT32_OUT_X2SCALE_SUPPORT_LIST, apiName_),
            false);
    }
    if (bias_ != nullptr) {
        if (outType_ == op::DataType::DT_INT32 && bias_->GetDataType() != op::DataType::DT_INT32) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(apiName_, GetInputName(BIAS_NAME, interfaceType_).c_str(),
                                                  op::ToString(bias_->GetDataType()).GetString(),
                                                  "when the dtype of out is INT32, the dtype of bias must be INT32");
            return false;
        }
        if (outType_ == op::DataType::DT_BF16 &&
            (x2ScaleType_ == op::DataType::DT_FLOAT || x2ScaleType_ == op::DataType::DT_BF16)) {
            CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, BIAS_NAME, bias_, BIAS_TYPE_SUPPORT_LIST, apiName_),
                      false);
        }
        if ((outType_ == op::DataType::DT_INT8 || outType_ == op::DataType::DT_FLOAT16 ||
             outType_ == op::DataType::DT_BF16) &&
            (x2ScaleType_ == op::DataType::DT_UINT64 || x2ScaleType_ == op::DataType::DT_INT64)) {
            CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, BIAS_NAME, bias_, op::DataType::DT_INT32, apiName_), false);
        }
    }
    if (x2Offset_ != nullptr) {
        if (outType_ != op::DataType::DT_INT8) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                apiName_, GetX2OffsetName().c_str(), "not null",
                FormatString("when x1 and x2 are INT8 and the dtype of out is outside INT8, %s must be null",
                             GetX2OffsetName().c_str())
                    .c_str());
            return false;
        }
        CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X2OFFSET_NAME, x2Offset_, op::DataType::DT_FLOAT, apiName_),
                  false);
    }
    CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, OUT_NAME, out_, INT8_OUT_TYPE_SUPPORT_LIST, apiName_), false);
    return true;
}

bool QuantMatmulChecker::CheckL0c2outOrL0c2ubPertensorPerchannel() const
{
    if (outType_ == op::DataType::DT_INT32 &&
        (x1_->GetDataType() != op::DataType::DT_INT8 || x2_->GetDataType() != op::DataType::DT_INT8)) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(apiName_, "x1, x2",
                                               FormatString("%s, %s", op::ToString(x1_->GetDataType()).GetString(),
                                                            op::ToString(x2_->GetDataType()).GetString())
                                                   .c_str(),
                                               "when the dtype of out is INT32, the dtypes of x1 and x2 must be INT8");
        return false;
    }
    if (IsInt8Input(x1_, x2_) || IsInt4Input(x1_, x2_)) {
        CHECK_RET(CheckL0c2outOrL0c2ubPertensorPerchannel4Int8Input(), false);
    } else if (IsHif8Input(x1_, x2_) || IsFp8Input(x1_, x2_)) {
        CHECK_RET(
            OpCheckDtypeNotSupport(interfaceType_, X2SCALE_NAME, x2Scale_, UINT64_X2_SCALE_TYPE_SUPPORT_LIST, apiName_),
            false);
        if (bias_ != nullptr) {
            CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, BIAS_NAME, bias_, op::DataType::DT_FLOAT, apiName_), false);
        }
        if (x2Offset_ != nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                apiName_, GetX2OffsetName().c_str(), "not null",
                FormatString("when x1 and x2 are FLOAT8 or HIFLOAT8, %s must be null", GetX2OffsetName().c_str())
                    .c_str());
            return false;
        }
        if (IsHif8Input(x1_, x2_)) {
            CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, OUT_NAME, out_, HIF8_OUT_TYPE_SUPPORT_LIST, apiName_),
                      false);
        } else {
            CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, OUT_NAME, out_, FP8_OUT_TYPE_SUPPORT_LIST, apiName_),
                      false);
        }
    } else {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            apiName_, "x1, x2",
            FormatString("%s, %s", RemoveDtInDtype(op::ToString(x1_->GetDataType()).GetString()).c_str(),
                         RemoveDtInDtype(op::ToString(x2_->GetDataType()).GetString()).c_str())
                .c_str(),
            FormatString("when %s is null, the dtypes of x1 and x2 must be the same and must be in "
                         "[INT8, FLOAT8, HIFLOAT8]",
                         GetX1ScaleName().c_str())
                .c_str());
        return false;
    }
    return true;
}

bool QuantMatmulChecker::CheckMicroScaling() const
{
    if (!IsFp4Input(x1_, x2_) && !IsFp8Input(x1_, x2_)) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            apiName_, "x1, x2",
            FormatString("%s, %s", RemoveDtInDtype(op::ToString(x1_->GetDataType()).GetString()).c_str(),
                         RemoveDtInDtype(op::ToString(x2_->GetDataType()).GetString()).c_str())
                .c_str(),
            FormatString("when %s and %s are FLOAT8_E8M0, the dtypes of x1 and x2 must be FLOAT4_E2M1 or FLOAT8",
                         GetX1ScaleName().c_str(), GetX2ScaleName().c_str())
                .c_str());
        return false;
    }
    if (bias_ != nullptr) {
        CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, BIAS_NAME, bias_, op::DataType::DT_FLOAT, apiName_), false);
    }
    CHECK_RET(
        OpCheckDtypeNotSupport(interfaceType_, OUT_NAME, out_, FP8_AND_HIF8_COMMON_OUT_TYPE_SUPPORT_LIST, apiName_),
        false);
    if (x2Offset_ != nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            apiName_, GetX2OffsetName().c_str(), "not null",
            FormatString("when %s and %s are FLOAT8_E8M0, %s must be null", GetX1ScaleName().c_str(),
                         GetX2ScaleName().c_str(), GetX2OffsetName().c_str())
                .c_str());
        return false;
    }
    // MXFP4 NZ场景下，transposeX1必须为false
    if (IsFp4Input(x1_, x2_) && ge::GetPrimaryFormat(x2_->GetStorageFormat()) == Format::FORMAT_FRACTAL_NZ &&
        transposeX1_) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            apiName_, "transposeX1", transposeX1_ ? "true" : "false",
            FormatString("when %s and %s are FLOAT8_E8M0, x1 and x2 are FLOAT4_E2M1 and x2 is FRACTAL_NZ, "
                         "transposeX1 must be false",
                         GetX1ScaleName().c_str(), GetX2ScaleName().c_str())
                .c_str());
        return false;
    }
    return true;
}

bool QuantMatmulChecker::CheckL0c2outOrL0c2ubPertoken() const
{
    CHECK_RET(
        OpCheckDtypeNotSupport(interfaceType_, X1_NAME, x1_, {op::DataType::DT_INT4, op::DataType::DT_INT8}, apiName_),
        false);
    CHECK_RET(
        OpCheckDtypeNotSupport(interfaceType_, X2_NAME, x2_, {op::DataType::DT_INT4, op::DataType::DT_INT8}, apiName_),
        false);
    CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, OUT_NAME, out_, PERTOKEN_OUT_TYPE_SUPPORT_LIST, apiName_), false);
    CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X1SCALE_NAME, x1Scale_, op::DataType::DT_FLOAT, apiName_), false);

    if (x2Offset_ != nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName_, GetX2OffsetName().c_str(), "not null",
                                              FormatString("when x1 and x2 are INT8 with %s, %s must be null",
                                                           GetX1ScaleName().c_str(), GetX2OffsetName().c_str())
                                                  .c_str());
        return false;
    }

    if (outType_ == op::DataType::DT_BF16) {
        CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, X2SCALE_NAME, x2Scale_, PERTOKEN_BF16_OUT_X2SCALE_SUPPORT_LIST,
                                         apiName_),
                  false);
        if (bias_ != nullptr) {
            CHECK_RET(
                OpCheckDtypeNotSupport(interfaceType_, BIAS_NAME, bias_, PERTOKEN_BF16_OUT_BIAS_SUPPORT_LIST, apiName_),
                false);
        }
    } else if (outType_ == op::DataType::DT_FLOAT16) {
        CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X2SCALE_NAME, x2Scale_, op::DataType::DT_FLOAT, apiName_),
                  false);
        if (bias_ != nullptr) {
            CHECK_RET(
                OpCheckDtypeNotSupport(interfaceType_, BIAS_NAME, bias_, PERTOKEN_FP16_OUT_BIAS_SUPPORT_LIST, apiName_),
                false);
        }
    }

    return true;
}

bool QuantMatmulChecker::CheckL0C2outOrL0C2ubPertokenPergroup() const
{
    if (x1_ == nullptr || x2_ == nullptr || out_ == nullptr || x1Scale_ == nullptr || x2Scale_ == nullptr ||
        x2Offset_ == nullptr) {
        return false;
    }
    CHECK_RET(
        OpCheckDtypeNotSupport(interfaceType_, X1_NAME, x1_, {op::DataType::DT_INT8, op::DataType::DT_INT4}, apiName_),
        false);
    CHECK_RET(
        OpCheckDtypeNotSupport(interfaceType_, X2_NAME, x2_, {op::DataType::DT_INT8, op::DataType::DT_INT4}, apiName_),
        false);
    CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, OUT_NAME, out_, {op::DataType::DT_FLOAT16, op::DataType::DT_BF16},
                                     apiName_),
              false);
    CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, X1SCALE_NAME, x1Scale_, {op::DataType::DT_FLOAT}, apiName_),
              false);
    CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, X2SCALE_NAME, x2Scale_, {op::DataType::DT_FLOAT}, apiName_),
              false);
    CHECK_RET(OpCheckDtypeNotSupport(interfaceType_, X2OFFSET_NAME, x2Offset_, {op::DataType::DT_FLOAT16}, apiName_),
              false);

    if (transposeX1_ != false || transposeX2_ != true) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            apiName_, "transposeX1, transposeX2",
            FormatString("%s, %s", transposeX1_ ? "true" : "false", transposeX2_ ? "true" : "false").c_str(),
            "in A4W4 scenario, transposeX1 must be false and transposeX2 must be true");
        return false;
    }
    return true;
}

bool QuantMatmulChecker::CheckDoubleScaleAndFp8Hif8PertokenPerblock() const
{
    CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X1SCALE_NAME, x1Scale_, op::DataType::DT_FLOAT, apiName_), false);
    CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X2SCALE_NAME, x2Scale_, op::DataType::DT_FLOAT, apiName_), false);
    CHECK_RET(
        OpCheckDtypeNotSupport(interfaceType_, OUT_NAME, out_, FP8_AND_HIF8_COMMON_OUT_TYPE_SUPPORT_LIST, apiName_),
        false);
    if (IsPerblock(x1_, x2_, x1Scale_, x2Scale_)) {
        if (bias_ != nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(apiName_, GetInputName(BIAS_NAME, interfaceType_).c_str(), "not null",
                                                  "when the quantization mode is G-B or B-B, bias must be null");
            return false;
        }
    }
    if (bias_ != nullptr) {
        CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, BIAS_NAME, bias_, op::DataType::DT_FLOAT, apiName_), false);
    }
    if (x2Offset_ != nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            apiName_, GetX2OffsetName().c_str(), "not null",
            FormatString("when x1 and x2 are FLOAT8 or HIFLOAT8 and %s and %s are FLOAT, %s must be null",
                         GetX1ScaleName().c_str(), GetX2ScaleName().c_str(), GetX2OffsetName().c_str())
                .c_str());
        return false;
    }
    return true;
}

aclnnStatus QuantMatmulChecker::CheckDtypeL0c2outOrL0c2ub() const
{
    if (ge::GetPrimaryFormat(x2_->GetStorageFormat()) == op::Format::FORMAT_FRACTAL_NZ && !CheckDtype4WeightNz()) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (outType_ == op::DataType::DT_INT32 && x1Scale_ != nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            apiName_, GetX1ScaleName().c_str(), "not null",
            "when the dtype of out is INT32, pertokenScaleOptional/x1Scale(pertokenScale) must be null");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x1Scale_ == nullptr) { // pertensor/perchannel
        CHECK_RET(CheckL0c2outOrL0c2ubPertensorPerchannel(), ACLNN_ERR_PARAM_INVALID);
    } else if (IsMicroScaling(x1Scale_, x2Scale_)) { // micro scaling
        CHECK_RET(CheckMicroScaling(), ACLNN_ERR_PARAM_INVALID);
    } else if (IsInt4Input(x1_, x2_) && x2Offset_ != nullptr) { // pertoken
        CHECK_RET(CheckL0C2outOrL0C2ubPertokenPergroup(), ACLNN_ERR_PARAM_INVALID);
    } else if (IsInt8Input(x1_, x2_) || IsInt4Input(x1_, x2_)) { // pertoken
        CHECK_RET(CheckL0c2outOrL0c2ubPertoken(), ACLNN_ERR_PARAM_INVALID);
    } else if (IsHif8Input(x1_, x2_) ||
               IsFp8Input(x1_, x2_)) { // double scale, pertensor-perchannel, fp8/hif8 pertoken, perblock/pertile
        CHECK_RET(CheckDoubleScaleAndFp8Hif8PertokenPerblock(), ACLNN_ERR_PARAM_INVALID);
    } else {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            apiName_, "x1, x2",
            FormatString("%s, %s", RemoveDtInDtype(op::ToString(x1_->GetDataType()).GetString()).c_str(),
                         RemoveDtInDtype(op::ToString(x2_->GetDataType()).GetString()).c_str())
                .c_str(),
            FormatString("the quantization mode determined by x1, x2 and %s must be supported",
                         GetX1ScaleName().c_str())
                .c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

bool QuantMatmulChecker::CheckOnlyL0c2ubPertoken() const
{
    CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X1_NAME, x1_, op::DataType::DT_INT8, apiName_), false);
    CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X2_NAME, x2_, op::DataType::DT_INT8, apiName_), false);
    CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X1SCALE_NAME, x1Scale_, op::DataType::DT_FLOAT, apiName_), false);
    if (outType_ == op::DataType::DT_FLOAT16) {
        CHECK_RET(OpCheckDtypeNotMatch(interfaceType_, X2SCALE_NAME, x2Scale_, op::DataType::DT_FLOAT, apiName_),
                  false);
        CHECK_RET(CheckFloat16OutBiasAndOffset(), false);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(apiName_, GetInputName(OUT_NAME, interfaceType_).c_str(),
                                              RemoveDtInDtype(op::ToString(outType_).GetString()).c_str(),
                                              "when pertokenScaleOptional exists, the dtype of out must be FLOAT16");
        return false;
    }
    return true;
}

aclnnStatus QuantMatmulChecker::CheckDtype() const
{
    // 5 represents the aclnnQuantMatmulV5 interface
    if (!IsIntInput(x1_, x2_) && interfaceType_ != 5) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            apiName_, "x1, x2",
            FormatString("%s, %s", op::ToString(x1_->GetDataType()).GetString(),
                         op::ToString(x2_->GetDataType()).GetString())
                .c_str(),
            "when the interface is aclnnQuantMatmulV3 or aclnnQuantMatmulV4, the dtypes of x1 and x2 must be "
            "INT8, INT4 or INT32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    switch (npuArch_) {
        case NpuArch::DAV_2201:
            CHECK_RET(CheckDtypeOnlyL0c2out() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
            break;
        case NpuArch::DAV_3510:
            CHECK_RET(CheckDtypeL0c2outOrL0c2ub() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
            break;
        case NpuArch::DAV_2002:
            CHECK_RET(CheckDtypeOnlyL0c2ub() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
            break;
        default:
            OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "AclnnQuantMatmul do not support this platform: %s.",
                    op::ToString(socVersion_).GetString());
            return ACLNN_ERR_RUNTIME_ERROR;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus QuantMatmulChecker::CheckParams() const
{
    // 1. 分平台校验dtype
    aclnnStatus dtypeRet = CheckDtype();
    CHECK_RET(dtypeRet == ACLNN_SUCCESS, dtypeRet);

    // 2. 检查shape是否符合要求
    CHECK_RET(CheckShape(), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查format是否符合要求
    CHECK_RET(CheckFormat(), ACLNN_ERR_PARAM_INVALID);

    // 4. 空Tensor处理逻辑
    CHECK_RET(CheckEmptyTensor(), ACLNN_ERR_PARAM_INVALID);

    OP_LOGD("QuantMatmul check params success.");
    return ACLNN_SUCCESS;
}
