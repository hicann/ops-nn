/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file weight_quant_batch_matmul_v2_checker_for_920r1.cpp
 * \brief
 */

#include "weight_quant_batch_matmul_v2_checker_for_920r1.h"
#include "common/inc/op_host/op_const_def_nn.h"
#include "graph/types.h"
#include "matmul/common/op_host/math_util_nn.h"
#include "matmul/common/op_host/log_format_util.h"
#include "matmul/common/op_host/op_tiling/debug_tiling.h"
#include <unordered_set>

namespace {
constexpr size_t ANTIQUANT_OFFSET_INDEX = 3UL;
constexpr size_t BIAS_INDEX = 6UL;
constexpr size_t MM_SHAPE_LEN_ND = 2UL;
constexpr size_t MX_SCALE_SHAPE_LEN = 3UL;
constexpr size_t MM_SHAPE_LEN_NZ = 4UL;
constexpr uint64_t MAX_SHAPE_DIM = 65535UL;
constexpr uint64_t MIN_GROUP_SIZE = 32UL;
constexpr uint64_t MAX_INT32 = 2147483647UL;
constexpr uint64_t MX_GROUP_SIZE = 32UL;
constexpr uint64_t HIFLOAT4_GROUP_SIZE = 64UL;
constexpr size_t MX_SCALE_SHAPE_INNER_DIM_LEN = 2UL;
} // namespace

namespace optiling {
static const std::unordered_set<uint64_t> SUPPORT_GROUP_SIZE = {0UL, 32UL, 64UL, 128UL, 256UL};
static const std::unordered_set<uint64_t> A16W4_GROUP_SIZE = {64UL, 128UL, 256UL};
static const std::unordered_set<uint64_t> A16W4_GROUP_SIZE_GENERAL = {32UL, 64UL, 128UL, 256UL};
static const std::unordered_set<uint64_t> A16W8_GROUP_SIZE = {64UL, 128UL};
static const std::unordered_set<uint64_t> A16W8_GROUP_SIZE_GENERAL = {32UL, 64UL, 128UL, 256UL};

ge::graphStatus WeightQuantBatchMatmulV2CheckerFor920R1::CheckContext()
{
    // check Raw TilingData
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
    // check the Required input and output desc
    size_t idx = 0;
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx));
    inputParams_.aDtype = context_->GetInputDesc(idx++)->GetDataType();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx));
    inputParams_.bDtype = context_->GetInputDesc(idx++)->GetDataType();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(idx));
    inputParams_.antiQuantScaleDtype = context_->GetInputDesc(idx++)->GetDataType();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, context_->GetOutputDesc(0));
    inputParams_.cDtype = context_->GetOutputDesc(0)->GetDataType();
    inputParams_.opName = context_->GetNodeName();
    OPS_LOG_I(inputParams_.opName, "TilingContext: %s", Ops::NN::DebugTilingContext(context_).c_str());
    return ge::GRAPH_SUCCESS;
}

bool CheckBiasDtype(gert::TilingContext* context, WeightQuantBatchMatmulInfo inputParams)
{
    bool isA16MxFp4 = inputParams.bDtype == ge::DT_FLOAT4_E2M1 && inputParams.antiQuantScaleDtype == ge::DT_FLOAT8_E8M0;
    bool isA16HiF4 = inputParams.bDtype == Ops::NN::DT_HIFLOAT4;
    bool isA16S4 = inputParams.bDtype == ge::DT_INT4;
    bool isA16Fp4 = inputParams.bDtype == ge::DT_FLOAT4_E2M1 && !isA16MxFp4;
    bool isA16S8 = inputParams.bDtype == ge::DT_INT8;
    bool isA16Fp8 = inputParams.bDtype == ge::DT_FLOAT8_E4M3FN;

    auto biasDesc = context->GetOptionalInputDesc(BIAS_INDEX);
    if (biasDesc == nullptr) {
        return true;
    }

    auto biasDtype = biasDesc->GetDataType();
    OP_TILING_CHECK(inputParams.aDtype == ge::DT_FLOAT16 && biasDtype != ge::DT_FLOAT16,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        inputParams.opName, "bias", ge::TypeUtils::DataTypeToSerialString(biasDtype).c_str(),
                        "When the dtype of x is FLOAT16, the dtype of bias must be FLOAT16"),
                    return false);
    if (isA16S8 || isA16S4) {
        OP_TILING_CHECK(inputParams.aDtype == ge::DT_BF16 && biasDtype != ge::DT_BF16 && biasDtype != ge::DT_FLOAT,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(inputParams.opName, "bias",
                                                              ge::TypeUtils::DataTypeToSerialString(biasDtype).c_str(),
                                                              "When the dtype of x is BF16 and the dtype of weight is "
                                                              "INT4 or INT8, the dtype of bias must be BF16 or FLOAT"),
                        return false);
    }
    if (isA16Fp4 || isA16Fp8 || isA16MxFp4 || isA16HiF4) {
        OP_TILING_CHECK(inputParams.aDtype == ge::DT_BF16 && biasDtype != ge::DT_BF16,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            inputParams.opName, "bias", ge::TypeUtils::DataTypeToSerialString(biasDtype).c_str(),
                            "When the dtype of x is BF16 and the dtype of weight is FLOAT4_E2M1, FLOAT8_E4M3FN or "
                            "HIFLOAT4, the dtype of bias must be BF16"),
                        return false);
    }

    return true;
}

/*
The function is check the dtype limit:
    1. Input x dtype should be fp16/bf16, weight should be int8/int4
    2. Output dtype should be same with x dtype without quant
    3. antiquant scale dtype should be uint64 when weight's dtype is int8
    4. antiquant scale dtype should be same with x when weight's dtype is int4
    4. ouput dtype should be same with x dtype
*/
bool WeightQuantBatchMatmulV2CheckerFor920R1::CheckDtype()
{
    OP_TILING_CHECK(inputParams_.aDtype != ge::DT_FLOAT16 && inputParams_.aDtype != ge::DT_BF16,
                    OP_LOGE_FOR_INVALID_DTYPE(inputParams_.opName, "x",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str(),
                                              "FLOAT16 or BF16"),
                    return false);

    OP_TILING_CHECK(inputParams_.bDtype != ge::DT_FLOAT8_E4M3FN && inputParams_.bDtype != ge::DT_INT8 &&
                        inputParams_.bDtype != ge::DT_INT4 && inputParams_.bDtype != ge::DT_FLOAT4_E2M1 &&
                        inputParams_.bDtype != Ops::NN::DT_HIFLOAT4 && inputParams_.bDtype != ge::DT_HIFLOAT8,
                    OP_LOGE_FOR_INVALID_DTYPE(inputParams_.opName, "weight",
                                              ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str(),
                                              "INT8, FLOAT8_E4M3FN, HIFLOAT8, INT4, FLOAT4_E2M1 or HIFLOAT4"),
                    return false);

    if (inputParams_.antiQuantType == QuantType::PER_GROUP) {
        OP_TILING_CHECK(
            inputParams_.antiQuantScaleDtype != inputParams_.aDtype,
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                inputParams_.opName, "antiquantScale",
                ge::TypeUtils::DataTypeToSerialString(inputParams_.antiQuantScaleDtype).c_str(),
                "When the quantization mode is per-group, the dtype of antiquantScale must equal the dtype of x"),
            return false);
    } else if (inputParams_.antiQuantType == QuantType::PER_CHANNEL ||
               inputParams_.antiQuantType == QuantType::PER_TENSOR) {
        bool w8Nd = inputParams_.bFormat == ge::Format::FORMAT_ND &&
                    (inputParams_.bDtype == ge::DT_INT8 || inputParams_.bDtype == ge::DT_HIFLOAT8);
        bool w4Nz = inputParams_.bFormat == ge::Format::FORMAT_FRACTAL_NZ && inputParams_.bDtype == ge::DT_INT4 &&
                    !inputParams_.transB;
        if (w4Nz || w8Nd) {
            OP_TILING_CHECK(inputParams_.antiQuantScaleDtype != inputParams_.aDtype,
                            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                                inputParams_.opName, "antiquantScale",
                                ge::TypeUtils::DataTypeToSerialString(inputParams_.antiQuantScaleDtype).c_str(),
                                "When the dtype of weight is INT8 or HIFLOAT8 with ND format, or INT4 with "
                                "non-transposed NZ format, the dtype of antiquantScale must equal the dtype of x"),
                            return false);
        } else {
            OP_TILING_CHECK(
                inputParams_.antiQuantScaleDtype != ge::DT_UINT64,
                OP_LOGE_FOR_INVALID_DTYPE(
                    inputParams_.opName, "antiquantScale",
                    ge::TypeUtils::DataTypeToSerialString(inputParams_.antiQuantScaleDtype).c_str(), "UINT64"),
                return false);
        }
    } else if (isA16HiF4_) {
        OP_TILING_CHECK(
            inputParams_.antiQuantScaleDtype != Ops::NN::DT_HIFLOAT4_SCALE,
            OP_LOGE_FOR_INVALID_DTYPE(inputParams_.opName, "antiquantScale",
                                      ge::TypeUtils::DataTypeToSerialString(inputParams_.antiQuantScaleDtype).c_str(),
                                      "HIFLOAT4_SCALE"),
            return false);
    } else {
        OP_TILING_CHECK(
            !isA16MxFp4_,
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                inputParams_.opName, "weight, antiquantScale",
                Ops::NN::FormatString("%s, %s", ge::TypeUtils::DataTypeToSerialString(inputParams_.bDtype).c_str(),
                                      ge::TypeUtils::DataTypeToSerialString(inputParams_.antiQuantScaleDtype).c_str())
                    .c_str(),
                "When the quantization mode is MX FP4, the dtype of weight must be FLOAT4_E2M1 and the dtype of "
                "antiquantScale must be FLOAT8_E8M0"),
            return false);
    }
    auto antiQuantOffsetDesc = context_->GetOptionalInputDesc(ANTIQUANT_OFFSET_INDEX);
    if (antiQuantOffsetDesc != nullptr) {
        OP_TILING_CHECK(antiQuantOffsetDesc->GetDataType() != inputParams_.aDtype,
                        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                            inputParams_.opName, "antiquantOffset",
                            ge::TypeUtils::DataTypeToSerialString(antiQuantOffsetDesc->GetDataType()).c_str(),
                            "the dtype of antiquantOffset must equal the dtype of x"),
                        return false);
    }
    OP_TILING_CHECK(
        inputParams_.cDtype != inputParams_.aDtype,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            inputParams_.opName, "y, x",
            Ops::NN::FormatString("%s, %s", ge::TypeUtils::DataTypeToSerialString(inputParams_.cDtype).c_str(),
                                  ge::TypeUtils::DataTypeToSerialString(inputParams_.aDtype).c_str())
                .c_str(),
            "the dtype of y must equal the dtype of x"),
        return false);
    OP_TILING_CHECK(!CheckBiasDtype(context_, inputParams_),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Check bias dtype failed."), return false);
    return true;
}

/*
The function is check the attr limit:
    1. transposeX should be false
    2. group size should be 0 when per channel
       group size should be 64/128/256 when per group
       group size should be 32/64 when mx
    3. innerPrecise only support 0 or 1
*/
bool WeightQuantBatchMatmulV2CheckerFor920R1::CheckAttr()
{
    auto attrs = context_->GetAttrs();
    OPS_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    size_t idx = 0;
    auto transposeX = attrs->GetAttrPointer<bool>(idx++);
    auto transposeWeight = attrs->GetAttrPointer<bool>(idx++);

    const int64_t* groupSizePtr = nullptr;
    if (attrs->GetAttrNum() > idx) {
        groupSizePtr = attrs->GetAttrPointer<int64_t>(idx++);
    }
    idx++; // 跳过dtype属性

    const int64_t* innerPrecisePtr = nullptr;
    if (attrs->GetAttrNum() > idx) {
        innerPrecisePtr = attrs->GetAttrPointer<int64_t>(idx++);
    }

    inputParams_.transA = transposeX != nullptr && *transposeX;
    inputParams_.transB = transposeWeight != nullptr && *transposeWeight;

    OP_TILING_CHECK(
        inputParams_.transA,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "transA", "true", "transA must be false"),
        return false);
    if (groupSizePtr != nullptr) {
        OP_TILING_CHECK(SUPPORT_GROUP_SIZE.find(*groupSizePtr) == SUPPORT_GROUP_SIZE.end(),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "groupSize",
                                                              std::to_string(*groupSizePtr).c_str(),
                                                              "groupSize must be 0, 32, 64, 128 or 256"),
                        return false);
        inputParams_.groupSize = static_cast<uint64_t>(*groupSizePtr);
    }
    if (innerPrecisePtr != nullptr) {
        OP_TILING_CHECK((*innerPrecisePtr != 0) && (*innerPrecisePtr != 1),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "innerPrecise",
                                                              std::to_string(*innerPrecisePtr).c_str(),
                                                              "innerPrecise must be 0 or 1"),
                        return false);
        inputParams_.innerPrecise = static_cast<uint64_t>(*innerPrecisePtr);
    }
    return true;
}

bool CheckBiasShape(WeightQuantBatchMatmulInfo inputParams, const gert::StorageShape* biasShape)
{
    if (biasShape == nullptr) {
        return true;
    }

    auto biasShapeDimNum = static_cast<uint64_t>(biasShape->GetStorageShape().GetDimNum());
    if (biasShapeDimNum == 1) {
        OP_TILING_CHECK(biasShape->GetStorageShape().GetDim(0) != 0 &&
                            static_cast<uint64_t>(biasShape->GetStorageShape().GetDim(0)) != inputParams.nSize,
                        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                            inputParams.opName, "bias", std::to_string(biasShape->GetStorageShape().GetDim(0)).c_str(),
                            Ops::NN::FormatString("When bias is 1D, its element count must be 0 or %llu",
                                                  static_cast<unsigned long long>(inputParams.nSize))
                                .c_str()),
                        return false);
    } else if (biasShapeDimNum == MM_SHAPE_LEN_ND) {
        OP_TILING_CHECK(
            biasShape->GetStorageShape().GetDim(0) != 1 ||
                static_cast<uint64_t>(biasShape->GetStorageShape().GetDim(1)) != inputParams.nSize,
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                inputParams.opName, "bias",
                Ops::NN::FormatString("%lld,%lld", static_cast<long long>(biasShape->GetStorageShape().GetDim(0)),
                                      static_cast<long long>(biasShape->GetStorageShape().GetDim(1)))
                    .c_str(),
                Ops::NN::FormatString("When bias is 2D, its shape must be [1,%llu]",
                                      static_cast<unsigned long long>(inputParams.nSize))
                    .c_str()),
            return false);
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM(
            inputParams.opName, "bias",
            Ops::NN::FormatString("%lluD", static_cast<unsigned long long>(biasShapeDimNum)).c_str(), "1D or 2D");
        return false;
    }
    return true;
}

/*
The function is check the shape limit:
    1. the input x shape dims should be 2(ND), weight shape dims should be 2(ND)
    2. the k of x and weight must be same
    3. antiquant shape must be:
        per_channel: not trans_n: (1, n) or (n); trans_b: (n, 1) or (n)
        per_tensor: (1,)
    4. quant shape must be empty tensor
    5. bias shape must be empty tensor
    6. nk must <= 65535, m <= 65535(trans_a) or int32_max(not trans_a);
*/
bool WeightQuantBatchMatmulV2CheckerFor920R1::CheckShape()
{
    size_t idx = 0;
    auto xShape = context_->GetInputShape(idx++);
    auto weightShape = context_->GetInputShape(idx++);
    auto antiQuantScaleShape = context_->GetInputShape(idx++);
    auto antiQuantOffsetShape = context_->GetOptionalInputShape(idx++);
    auto quantScaleShape = context_->GetOptionalInputShape(idx++);
    auto quantOffsetShape = context_->GetOptionalInputShape(idx++);
    auto biasShape = context_->GetOptionalInputShape(idx++);
    auto outputShape = context_->GetOutputShape(0);

    OPS_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, weightShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, antiQuantScaleShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context_, outputShape);

    // not yet support empty tensor for input
    OP_TILING_CHECK(
        xShape->GetStorageShape().GetShapeSize() == 0 || weightShape->GetStorageShape().GetShapeSize() == 0 ||
            antiQuantScaleShape->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            inputParams_.opName, "x, weight, antiquantScale",
            Ops::NN::FormatString("%lld, %lld, %lld", static_cast<long long>(xShape->GetStorageShape().GetShapeSize()),
                                  static_cast<long long>(weightShape->GetStorageShape().GetShapeSize()),
                                  static_cast<long long>(antiQuantScaleShape->GetStorageShape().GetShapeSize()))
                .c_str(),
            "the element count of x, weight and antiquantScale must be greater than 0"),
        return false);

    // check x, weight
    inputParams_.aFormat = GetInputStorageFormat(context_, 0);
    inputParams_.bFormat = GetInputStorageFormat(context_, 1);
    OP_TILING_CHECK(!CheckInputShape(xShape, weightShape),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Check input x and weight shape failed"),
                    return false);
    // check antiquant scale, antiquant offset, judge antiquant type
    OP_TILING_CHECK(!CheckAntiQuantShape(antiQuantScaleShape, antiQuantOffsetShape),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Check antiquant shape failed"), return false);
    // check quant scale, quant offset
    OP_TILING_CHECK(quantScaleShape != nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "quantScale", "present",
                                                          "quantScale can not be provided"),
                    return false);
    OP_TILING_CHECK(quantOffsetShape != nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "quantOffset", "present",
                                                          "quantOffset can not be provided"),
                    return false);
    // check bias
    OP_TILING_CHECK(!CheckBiasShape(inputParams_, biasShape),
                    VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Check bias shape failed"), return false);
    if (!CheckShapeAdvanced()) {
        return false;
    }

    return true;
}

bool WeightQuantBatchMatmulV2CheckerFor920R1::CheckInputDimNum(const gert::StorageShape* xShape,
                                                               const gert::StorageShape* weightShape)
{
    size_t xOriDimNum = xShape->GetOriginShape().GetDimNum();
    size_t weightOriDimNum = weightShape->GetOriginShape().GetDimNum();
    size_t xDimNum = xShape->GetStorageShape().GetDimNum();
    size_t weightDimNum = weightShape->GetStorageShape().GetDimNum();

    OP_TILING_CHECK(
        xOriDimNum != MM_SHAPE_LEN_ND,
        OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opName, "x", Ops::NN::FormatString("%zuD", xOriDimNum).c_str(), "2D"),
        return false);

    OP_TILING_CHECK(weightOriDimNum != MM_SHAPE_LEN_ND,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opName, "weight",
                                                 Ops::NN::FormatString("%zuD", weightOriDimNum).c_str(), "2D"),
                    return false);

    OP_TILING_CHECK(
        xDimNum != MM_SHAPE_LEN_ND,
        OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opName, "x", Ops::NN::FormatString("%zuD", xDimNum).c_str(), "2D"),
        return false);

    OP_TILING_CHECK(inputParams_.bFormat == ge::Format::FORMAT_ND && weightDimNum != MM_SHAPE_LEN_ND,
                    OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opName, "weight",
                                                 Ops::NN::FormatString("%zuD", weightDimNum).c_str(), "2D"),
                    return false);

    OP_TILING_CHECK(inputParams_.bFormat == ge::Format::FORMAT_FRACTAL_NZ && weightDimNum != MM_SHAPE_LEN_NZ,
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        inputParams_.opName, "weight", Ops::NN::FormatString("%zuD", weightDimNum).c_str(),
                        "When the format of weight is FRACTAL_NZ, its rank must be 4D"),
                    return false);

    return true;
}

bool WeightQuantBatchMatmulV2CheckerFor920R1::CheckInputSize(const gert::StorageShape* xShape,
                                                             const gert::StorageShape* weightShape)
{
    uint64_t weightLastDim = weightShape->GetOriginShape().GetDim(1);

    inputParams_.mSize = static_cast<uint64_t>(inputParams_.transA ? xShape->GetOriginShape().GetDim(1) :
                                                                     xShape->GetOriginShape().GetDim(0));

    inputParams_.kSize = static_cast<uint64_t>(inputParams_.transA ? xShape->GetOriginShape().GetDim(0) :
                                                                     xShape->GetOriginShape().GetDim(1));
    auto kBSize = static_cast<uint64_t>(inputParams_.transB ? weightLastDim : weightShape->GetOriginShape().GetDim(0));
    OP_TILING_CHECK(inputParams_.kSize != kBSize,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        inputParams_.opName, "x, weight",
                        Ops::NN::FormatString("%llu, %llu", static_cast<unsigned long long>(inputParams_.kSize),
                                              static_cast<unsigned long long>(kBSize))
                            .c_str(),
                        "the K dimension of x must equal the K dimension of weight"),
                    return false);

    isA16MxFp4_ = inputParams_.bDtype == ge::DT_FLOAT4_E2M1 && inputParams_.antiQuantScaleDtype == ge::DT_FLOAT8_E8M0;
    isA16HiF4_ = inputParams_.bDtype == Ops::NN::DT_HIFLOAT4;
    inputParams_.nSize = static_cast<uint64_t>(inputParams_.transB ? weightShape->GetOriginShape().GetDim(0) :
                                                                     weightLastDim);
    auto weightStorageShape = weightShape->GetStorageShape();
    if (inputParams_.bFormat == ge::Format::FORMAT_FRACTAL_NZ && inputParams_.transB) {
        uint64_t n0 = weightStorageShape.GetDim(2);
        uint64_t n1 = weightStorageShape.GetDim(1);
        OP_TILING_CHECK(!isA16MxFp4_ && !isA16HiF4_ && n0 * n1 != inputParams_.nSize,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            inputParams_.opName, "weight",
                            Ops::NN::FormatString("%llu, %llu", static_cast<unsigned long long>(n0 * n1),
                                                  static_cast<unsigned long long>(inputParams_.nSize))
                                .c_str(),
                            "When the format of weight is FRACTAL_NZ, the product of N0 and N1 must equal N"),
                        return false);
    }
    // 3: NZ第四维是C0 16：C016
    isNZC016_ = weightStorageShape.GetDim(3) == 16 && inputParams_.bFormat == ge::Format::FORMAT_FRACTAL_NZ ? true :
                                                                                                              false;
    OP_TILING_CHECK(isNZC016_ && inputParams_.transB,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        inputParams_.opName, "transB", "true",
                        "When the format of weight is FRACTAL_NZ and C0 is 16, transB must be false"),
                    return false);

    return true;
}

bool WeightQuantBatchMatmulV2CheckerFor920R1::CheckInputShape(const gert::StorageShape* xShape,
                                                              const gert::StorageShape* weightShape)
{
    OP_TILING_CHECK(inputParams_.aFormat != ge::Format::FORMAT_ND,
                    OP_LOGE_FOR_INVALID_FORMAT(inputParams_.opName, "x",
                                               ge::TypeUtils::FormatToSerialString(inputParams_.aFormat).c_str(), "ND"),
                    return false);

    if (!CheckInputDimNum(xShape, weightShape)) {
        return false;
    }

    if (!CheckInputSize(xShape, weightShape)) {
        return false;
    }

    return true;
}

bool WeightQuantBatchMatmulV2CheckerFor920R1::CheckAntiQuantShape(const gert::StorageShape* antiQuantScaleShape,
                                                                  const gert::StorageShape* antiQuantOffsetShape)
{
    size_t antiQuantScaleDimNum = antiQuantScaleShape->GetStorageShape().GetDimNum();
    isA16HiF4_ = inputParams_.bDtype == Ops::NN::DT_HIFLOAT4;
    size_t antiQuantScaleShapeSize = static_cast<size_t>(antiQuantScaleShape->GetStorageShape().GetShapeSize());
    const size_t maxAntiQuantScaleShapeLen = (isA16MxFp4_ || isA16HiF4_) ? MX_SCALE_SHAPE_LEN : MM_SHAPE_LEN_ND;
    if (antiQuantScaleDimNum > maxAntiQuantScaleShapeLen) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            inputParams_.opName, "antiquantScale", Ops::NN::FormatString("%zuD", antiQuantScaleDimNum).c_str(),
            Ops::NN::FormatString("the rank of antiquantScale must be at most %zuD", maxAntiQuantScaleShapeLen)
                .c_str());
        return false;
    }

    if (isA16MxFp4_ && antiQuantScaleDimNum != MX_SCALE_SHAPE_LEN) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(inputParams_.opName, "antiquantScale",
                                     Ops::NN::FormatString("%zuD", antiQuantScaleDimNum).c_str(), "3D");
        return false;
    }

    isA16W4C016PerG_ = (inputParams_.bDtype == ge::DT_INT4 || inputParams_.bDtype == ge::DT_FLOAT4_E2M1) && isNZC016_ &&
                       inputParams_.groupSize > 0;
    bool isA16W8C016PerG = inputParams_.bDtype == ge::DT_INT8 && isNZC016_ && inputParams_.groupSize > 0;
    if ((inputParams_.bDtype == ge::DT_FLOAT4_E2M1 && isNZC016_) ||
        (inputParams_.bFormat == ge::Format::FORMAT_FRACTAL_NZ && !isNZC016_) ||
        ((inputParams_.bDtype == ge::DT_FLOAT8_E4M3FN || inputParams_.bDtype == ge::DT_HIFLOAT8) &&
         inputParams_.bFormat == ge::Format::FORMAT_ND)) {
        OP_TILING_CHECK(antiQuantOffsetShape != nullptr,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "antiquantOffset", "present",
                                                              "antiquantOffset can not be provided"),
                        return false);
    }

    if (antiQuantScaleShapeSize != 1) {
        if (antiQuantScaleDimNum == maxAntiQuantScaleShapeLen) {
            uint64_t kNum = 1;
            gert::Shape expectShape;
            if (inputParams_.groupSize == 0) {
                // PerChannel 量化
                if (inputParams_.transB) {
                    expectShape.AppendDim(static_cast<int64_t>(inputParams_.nSize));
                    expectShape.AppendDim(static_cast<int64_t>(kNum));
                } else {
                    expectShape.AppendDim(static_cast<int64_t>(kNum));
                    expectShape.AppendDim(static_cast<int64_t>(inputParams_.nSize));
                }
                inputParams_.antiQuantType = QuantType::PER_CHANNEL;
            } else if (inputParams_.groupSize > 0) {
                if (isA16MxFp4_) {
                    // Mx 量化 MxFp4
                    OP_TILING_CHECK(inputParams_.groupSize != MX_GROUP_SIZE,
                                    OP_LOGE_FOR_INVALID_VALUE(inputParams_.opName, "groupSize",
                                                              std::to_string(inputParams_.groupSize).c_str(), "32"),
                                    return false);
                    // mx scale是3维的：(n, ceildiv(k, 32 *2), 2)
                    expectShape.AppendDim(static_cast<int64_t>(inputParams_.nSize));
                    expectShape.AppendDim(static_cast<int64_t>(
                        ops::CeilDiv(inputParams_.kSize, MX_GROUP_SIZE * MX_SCALE_SHAPE_INNER_DIM_LEN)));
                    expectShape.AppendDim(static_cast<int64_t>(MX_SCALE_SHAPE_INNER_DIM_LEN));
                    inputParams_.antiQuantType = QuantType::MX;
                } else if (isA16HiF4_) {
                    // Mx 量化 HiF4
                    OP_TILING_CHECK(inputParams_.groupSize != HIFLOAT4_GROUP_SIZE,
                                    OP_LOGE_FOR_INVALID_VALUE(inputParams_.opName, "groupSize",
                                                              std::to_string(inputParams_.groupSize).c_str(), "64"),
                                    return false);
                    expectShape.AppendDim(static_cast<int64_t>(inputParams_.nSize));
                    expectShape.AppendDim(static_cast<int64_t>(ops::CeilDiv(inputParams_.kSize, HIFLOAT4_GROUP_SIZE)));
                    expectShape.AppendDim(static_cast<int64_t>(1));
                    inputParams_.antiQuantType = QuantType::MX;
                } else {
                    // PerGroup 量化
                    bool isA16W4PerG = (inputParams_.bDtype == ge::DT_INT4 ||
                                        inputParams_.bDtype == ge::DT_FLOAT4_E2M1) &&
                                       !isNZC016_;
                    bool isA16W8PerG = (inputParams_.bDtype == ge::DT_INT8 ||
                                        inputParams_.bDtype == ge::DT_FLOAT8_E4M3FN) &&
                                       !isNZC016_;
                    OP_TILING_CHECK(
                        isA16W4PerG && A16W4_GROUP_SIZE.find(inputParams_.groupSize) == A16W4_GROUP_SIZE.end(),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "groupSize",
                                                              std::to_string(inputParams_.groupSize).c_str(),
                                                              "groupSize must be 64, 128 or 256"),
                        return false);
                    OP_TILING_CHECK(
                        isA16W4C016PerG_ &&
                            A16W4_GROUP_SIZE_GENERAL.find(inputParams_.groupSize) == A16W4_GROUP_SIZE_GENERAL.end(),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "groupSize",
                                                              std::to_string(inputParams_.groupSize).c_str(),
                                                              "groupSize must be 32, 64, 128 or 256"),
                        return false);

                    OP_TILING_CHECK(
                        isA16W8PerG && A16W8_GROUP_SIZE.find(inputParams_.groupSize) == A16W8_GROUP_SIZE.end(),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "groupSize",
                                                              std::to_string(inputParams_.groupSize).c_str(),
                                                              "groupSize must be 64 or 128"),
                        return false);
                    OP_TILING_CHECK(
                        isA16W8C016PerG &&
                            A16W8_GROUP_SIZE_GENERAL.find(inputParams_.groupSize) == A16W8_GROUP_SIZE_GENERAL.end(),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(inputParams_.opName, "groupSize",
                                                              std::to_string(inputParams_.groupSize).c_str(),
                                                              "groupSize must be 32, 64, 128 or 256"),
                        return false);
                    kNum = ops::CeilDiv(inputParams_.kSize, inputParams_.groupSize);
                    expectShape.AppendDim(static_cast<int64_t>(kNum));
                    expectShape.AppendDim(static_cast<int64_t>(inputParams_.nSize));
                    inputParams_.antiQuantType = QuantType::PER_GROUP;
                }
                OP_TILING_CHECK(
                    !isA16MxFp4_ && !isA16W4C016PerG_ && !isA16W8C016PerG &&
                        inputParams_.kSize % inputParams_.groupSize != 0,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        inputParams_.opName, "weight, groupSize",
                        Ops::NN::FormatString("%llu, %llu", static_cast<unsigned long long>(inputParams_.kSize),
                                              static_cast<unsigned long long>(inputParams_.groupSize))
                            .c_str(),
                        "When the quantization mode is per-group, K must be divisible by groupSize"),
                    return false);
            }
            OP_TILING_CHECK(expectShape != antiQuantScaleShape->GetStorageShape(),
                            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                inputParams_.opName, "antiquantScale",
                                Ops::Base::ToString(antiQuantScaleShape->GetStorageShape())
                                    .substr(1, Ops::Base::ToString(antiQuantScaleShape->GetStorageShape()).size() - 2)
                                    .c_str(),
                                Ops::NN::FormatString("the shape of antiquantScale must be %s",
                                                      Ops::Base::ToString(expectShape).c_str())
                                    .c_str()),
                            return false);
            if (antiQuantOffsetShape != nullptr) {
                OP_TILING_CHECK(
                    expectShape != antiQuantOffsetShape->GetStorageShape(),
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        inputParams_.opName, "antiquantOffset",
                        Ops::Base::ToString(antiQuantOffsetShape->GetStorageShape())
                            .substr(1, Ops::Base::ToString(antiQuantOffsetShape->GetStorageShape()).size() - 2)
                            .c_str(),
                        Ops::NN::FormatString("the shape of antiquantOffset must be %s",
                                              Ops::Base::ToString(expectShape).c_str())
                            .c_str()),
                    return false);
            }
        } else {
            // PerChannel 量化
            OP_TILING_CHECK(antiQuantScaleShapeSize != inputParams_.nSize,
                            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                                inputParams_.opName, "antiquantScale", std::to_string(antiQuantScaleShapeSize).c_str(),
                                Ops::NN::FormatString("When the quantization mode is per-channel, the element count of "
                                                      "antiquantScale must equal N (%llu)",
                                                      static_cast<unsigned long long>(inputParams_.nSize))
                                    .c_str()),
                            return false);
            inputParams_.antiQuantType = QuantType::PER_CHANNEL;
        }
    } else {
        // PerTensor 量化
        inputParams_.antiQuantType = QuantType::PER_TENSOR;
    }

    return true;
}

bool WeightQuantBatchMatmulV2CheckerFor920R1::CheckShapeAdvanced()
{
    if ((inputParams_.bDtype == ge::DT_INT4 && (inputParams_.antiQuantType == QuantType::PER_CHANNEL ||
                                                inputParams_.antiQuantType == QuantType::PER_TENSOR)) ||
        isA16W4C016PerG_) {
        // A16W4 输入 PerChannel/PerTensor weight ND 和 PerGroup weight NZ非转置 支持非对齐，仅要求 n 和 k 是偶数
        OP_TILING_CHECK(
            // 2 对齐
            inputParams_.nSize % 2 != 0 || inputParams_.kSize % 2 != 0,
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                inputParams_.opName, "x, weight",
                Ops::NN::FormatString("%llu, %llu", static_cast<unsigned long long>(inputParams_.kSize),
                                      static_cast<unsigned long long>(inputParams_.nSize))
                    .c_str(),
                "the K and N dimensions must be divisible by 2"),
            return false);
    } else if (inputParams_.bDtype == ge::DT_FLOAT4_E2M1) {
        OP_TILING_CHECK(
            // 64对齐
            !isA16MxFp4_ && (inputParams_.nSize % 64 != 0 || inputParams_.kSize % 64 != 0),
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                inputParams_.opName, "x, weight",
                Ops::NN::FormatString("%llu, %llu", static_cast<unsigned long long>(inputParams_.kSize),
                                      static_cast<unsigned long long>(inputParams_.nSize))
                    .c_str(),
                "the K and N dimensions must be divisible by 64"),
            return false);
    }

    // A16W8 仅支持 n 和 k 对 32 对齐，A16S8ND支持非对齐
    if ((inputParams_.bDtype == ge::DT_INT8 && inputParams_.bFormat != ge::Format::FORMAT_ND) ||
        inputParams_.bDtype == ge::DT_FLOAT8_E4M3FN) {
        OP_TILING_CHECK(
            // 32 对齐
            inputParams_.nSize % 32 != 0 || inputParams_.kSize % 32 != 0,
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                inputParams_.opName, "x, weight",
                Ops::NN::FormatString("%llu, %llu", static_cast<unsigned long long>(inputParams_.kSize),
                                      static_cast<unsigned long long>(inputParams_.nSize))
                    .c_str(),
                "the K and N dimensions must be divisible by 32"),
            return false);
    }

    bool w4NzNoTrans = inputParams_.bFormat == ge::Format::FORMAT_FRACTAL_NZ && inputParams_.bDtype == ge::DT_INT4 &&
                       !inputParams_.transB;
    if ((inputParams_.bFormat == ge::Format::FORMAT_ND &&
         (inputParams_.bDtype == ge::DT_INT4 || inputParams_.bDtype == ge::DT_FLOAT8_E4M3FN) &&
         (inputParams_.antiQuantType == QuantType::PER_CHANNEL ||
          inputParams_.antiQuantType == QuantType::PER_TENSOR)) ||
        isA16W4C016PerG_ || (inputParams_.bDtype == ge::DT_INT8 && isNZC016_) || w4NzNoTrans) {
        // A16W4/A16Fp8 PerChannel/PerTensor ND 和 A16W4/A16W8 PerGroup NZ C016 支持 n、k 大于 65535
        OP_TILING_CHECK(inputParams_.kSize > MAX_INT32 || inputParams_.nSize > MAX_INT32,
                        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                            inputParams_.opName, "x, weight",
                            Ops::NN::FormatString("%llu, %llu", static_cast<unsigned long long>(inputParams_.kSize),
                                                  static_cast<unsigned long long>(inputParams_.nSize))
                                .c_str(),
                            "the K and N dimensions must each be at most 2147483647"),
                        return false);
    } else {
        OP_TILING_CHECK(inputParams_.kSize > MAX_SHAPE_DIM || inputParams_.nSize > MAX_SHAPE_DIM,
                        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                            inputParams_.opName, "x, weight",
                            Ops::NN::FormatString("%llu, %llu", static_cast<unsigned long long>(inputParams_.kSize),
                                                  static_cast<unsigned long long>(inputParams_.nSize))
                                .c_str(),
                            "the K and N dimensions must each be at most 65535"),
                        return false);
    }
    uint64_t batchMax = inputParams_.transA ? MAX_SHAPE_DIM : MAX_INT32;
    OP_TILING_CHECK(
        inputParams_.mSize > batchMax,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            inputParams_.opName, "x", std::to_string(inputParams_.mSize).c_str(),
            Ops::NN::FormatString("the M dimension must be at most %llu", static_cast<unsigned long long>(batchMax))
                .c_str()),
        return false);
    OP_TILING_CHECK(
        (inputParams_.bDtype != Ops::NN::DT_HIFLOAT4 && inputParams_.groupSize >= inputParams_.kSize) ||
            inputParams_.groupSize % MIN_GROUP_SIZE != 0,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            inputParams_.opName, "groupSize, weight",
            Ops::NN::FormatString("%llu, %llu", static_cast<unsigned long long>(inputParams_.groupSize),
                                  static_cast<unsigned long long>(inputParams_.kSize))
                .c_str(),
            "groupSize must be divisible by 32 and must be less than K unless the dtype of weight is HIFLOAT4"),
        return false);

    return true;
}

ge::graphStatus WeightQuantBatchMatmulV2CheckerFor920R1::Check()
{
    auto ret = CheckContext();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    // check attrs: transA, transB, groupSize, innerPrecise
    OP_TILING_CHECK(!CheckAttr(), VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Check attr failed"),
                    return ge::GRAPH_FAILED);
    // check the input and output shape: all
    OP_TILING_CHECK(!CheckShape(), VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Check shape failed"),
                    return ge::GRAPH_FAILED);

    // check the input and output dtype: x, weight, antiquant_scale, y.
    OP_TILING_CHECK(!CheckDtype(), VECTOR_INNER_ERR_REPORT_TILIING(inputParams_.opName, "Check input dtype failed"),
                    return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
