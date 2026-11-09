/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_ada_layer_norm_backward.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "level0/fill.h"
#include "ada_layer_norm_grad.h"
#include "level0/squeeze.h"
#include "norm/norm_common/op_host/op_api/norm_tensor_util.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "log/log.h"
#include "op_api/aclnn_util.h"

using namespace op;

namespace {
// aclnn op_api层空指针失败出口统一上报EZ1011(nnopbase注册码, Arglist={functionName,paraName})。
// op_api编译单元带-DOP_LOG_LIBOPAPI_ONLY, log/log.h内OP_LOGE_WITHOUT_REPORT(opName版)不可见,
// 日志走始终可见的OP_LOGE_LIBOPAPI_REPORT。
void ReportNullPointerError(const char* functionName, const char* paraName)
{
    std::string funcName = functionName;
    std::string paramName = paraName;
    std::string msg = funcName + " failed because " + paramName + " cannot be a NULL pointer.";
    OP_LOGE_LIBOPAPI_REPORT(funcName.c_str(), "%s", msg.c_str());
    const std::vector<const char*> msgKey = {"functionName", "paraName"};
    const std::vector<const char*> msgValue = {funcName.c_str(), paramName.c_str()};
    REPORT_PREDEFINED_ERR_MSG("EZ1011", msgKey, msgValue);
}
} // namespace
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_DIM_LEN = 8;
constexpr size_t GRAD_INPUT_INDEX = 0;
constexpr size_t GRAD_SCALE_INDEX = 1;
constexpr size_t GRAD_SHIFT_INDEX = 2;
constexpr size_t GRAD_WEIGHT_INDEX = 3;
constexpr size_t GRAD_BIAS_INDEX = 4;
constexpr size_t NORMALIZED_SHAPE_LEN = 1;
constexpr size_t GRAD_OUT_NUM = 5;
constexpr size_t X_OUT_NUM = 2;
constexpr size_t BETA_GAMMAX_NUM = 2;
constexpr int64_t MIN_GRAD_REDUCE_AXIS = 1024;
constexpr int64_t MAX_GRAD_REDUCE_AXIS = 4096;

// 根据API定义，列出所能支持的所有dtype
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16, DataType::DT_FLOAT,
                                                                   DataType::DT_BF16};

inline static bool CheckNotNull(const aclTensor* gradOut, const aclTensor* input, const aclIntArray* normalizedShape,
                                const aclTensor* mean, const aclTensor* rstd, const aclTensor* scale,
                                const aclTensor* shift)
{
    // 校验输入是否为空指针
    OP_CHECK(gradOut != nullptr, ReportNullPointerError(__func__, "gradOut"), return false);
    OP_CHECK(input != nullptr, ReportNullPointerError(__func__, "input"), return false);
    OP_CHECK(normalizedShape != nullptr, ReportNullPointerError(__func__, "normalizedShape"), return false);
    OP_CHECK(mean != nullptr, ReportNullPointerError(__func__, "mean"), return false);
    OP_CHECK(rstd != nullptr, ReportNullPointerError(__func__, "rstd"), return false);
    OP_CHECK(scale != nullptr, ReportNullPointerError(__func__, "scale"), return false);
    OP_CHECK(shift != nullptr, ReportNullPointerError(__func__, "shift"), return false);
    return true;
}

inline static bool CheckOptionalOutDtype(const aclTensor* input, const aclTensor* weightOptional,
                                         const aclTensor* biasOptional, const aclTensor* gradWeightOut,
                                         const aclTensor* gradBiasOut)
{
    if (gradWeightOut) {
        if (weightOptional) {
            OP_CHECK_DTYPE_NOT_SAME(gradWeightOut, weightOptional, return false);
        } else if (biasOptional) {
            OP_CHECK_DTYPE_NOT_SAME(gradWeightOut, biasOptional, return false);
        } else {
            if (gradWeightOut->GetDataType() != DataType::DT_FLOAT) {
                OP_CHECK_DTYPE_NOT_SAME(gradWeightOut, input, return false);
            }
        }
    }

    if (gradBiasOut) {
        if (weightOptional) {
            OP_CHECK_DTYPE_NOT_SAME(gradBiasOut, weightOptional, return false);
        } else if (biasOptional) {
            OP_CHECK_DTYPE_NOT_SAME(gradBiasOut, biasOptional, return false);
        } else {
            if (gradBiasOut->GetDataType() != DataType::DT_FLOAT) {
                OP_CHECK_DTYPE_NOT_SAME(gradBiasOut, input, return false);
            }
        }
    }
    return true;
}

inline static bool CheckFormat(const aclTensor* gradOut, const aclTensor* input, const aclTensor* rstd,
                               const aclTensor* mean, const aclTensor* scale, const aclTensor* shift,
                               const aclTensor* weightOptional, const aclTensor* biasOptional,
                               const aclTensor* gradInputOut, const aclTensor* gradScaleOut,
                               const aclTensor* gradShiftOut, const aclTensor* gradWeightOut,
                               const aclTensor* gradBiasOut)
{
    // 全部入/出参仅支持 ND；私有格式会被当成连续数据计算，结果错且不报错。
    struct TensorName {
        const aclTensor* tensor;
        const char* name;
    };
    const TensorName tensors[] = {{gradOut, "gradOut"},
                                  {input, "input"},
                                  {rstd, "rstd"},
                                  {mean, "mean"},
                                  {scale, "scale"},
                                  {shift, "shift"},
                                  {weightOptional, "weightOptional"},
                                  {biasOptional, "biasOptional"},
                                  {gradInputOut, "gradInputOut"},
                                  {gradScaleOut, "gradScaleOut"},
                                  {gradShiftOut, "gradShiftOut"},
                                  {gradWeightOut, "gradWeightOut"},
                                  {gradBiasOut, "gradBiasOut"}};
    for (const auto& item : tensors) {
        if (item.tensor == nullptr) {
            continue;
        }
        auto format = item.tensor->GetStorageFormat();
        if (format != op::Format::FORMAT_ND) {
            OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON("aclnnAdaLayerNormBackward", item.name,
                                                   op::ToString(format).GetString(), "only ND format is supported");
            return false;
        }
    }
    return true;
}

static bool CheckDtype(const aclTensor* gradOut, const aclTensor* input, const aclTensor* rstd, const aclTensor* mean,
                       const aclTensor* scale, const aclTensor* shift, const aclTensor* weightOptional,
                       const aclTensor* biasOptional, const aclTensor* gradInputOut, const aclTensor* gradScaleOut,
                       const aclTensor* gradShiftOut, const aclTensor* gradWeightOut, const aclTensor* gradBiasOut)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(gradOut, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SAME(input, gradOut, return false);
    OP_CHECK_DTYPE_NOT_SAME(rstd, gradOut, return false);
    OP_CHECK_DTYPE_NOT_SAME(mean, gradOut, return false);
    OP_CHECK_DTYPE_NOT_SAME(scale, gradOut, return false);
    OP_CHECK_DTYPE_NOT_SAME(shift, gradOut, return false);

    if (weightOptional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(weightOptional, DTYPE_SUPPORT_LIST, return false);
        if (weightOptional->GetDataType() != DataType::DT_FLOAT) {
            OP_CHECK_DTYPE_NOT_SAME(weightOptional, input, return false);
        }
    }
    if (biasOptional) {
        OP_CHECK_DTYPE_NOT_SUPPORT(biasOptional, DTYPE_SUPPORT_LIST, return false);
        if (biasOptional->GetDataType() != DataType::DT_FLOAT) {
            OP_CHECK_DTYPE_NOT_SAME(biasOptional, input, return false);
        }
    }
    if (biasOptional != nullptr && weightOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SAME(weightOptional, biasOptional, return false);
    }
    // 检查输出类型
    if (gradInputOut) {
        OP_CHECK_DTYPE_NOT_SAME(gradInputOut, input, return false);
    }
    if (gradScaleOut) {
        OP_CHECK_DTYPE_NOT_SAME(gradScaleOut, scale, return false);
    }
    if (gradShiftOut) {
        OP_CHECK_DTYPE_NOT_SAME(gradShiftOut, shift, return false);
    }
    if (!CheckOptionalOutDtype(input, weightOptional, biasOptional, gradWeightOut, gradBiasOut)) {
        return false;
    }

    return true;
}

static bool CheckShapeEqual(const aclTensor* input, const aclIntArray* normalizedShape)
{
    auto inputShape = input->GetViewShape();
    size_t inputLen = inputShape.GetDimNum();
    size_t normLen = normalizedShape->Size();
    size_t axis = inputLen - normLen;

    for (size_t i = 0; i < normLen; i++) {
        int64_t normDim = *(normalizedShape->GetData() + i);
        int64_t inputDim = inputShape.GetDim(i + axis);
        if (normDim == inputDim)
            continue;

        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnAdaLayerNormBackward", "normalizedShape",
                                                 std::to_string(normDim).c_str(),
                                                 "dims of normalizedShape must match the trailing dims of input");
        return false;
    }
    return true;
}

static bool CheckShapeCompatibility(const aclTensor* weightOptional, const aclTensor* biasOptional,
                                    const aclIntArray* normalizedShape)
{
    if (weightOptional) {
        OP_CHECK_MAX_DIM(weightOptional, MAX_DIM_LEN, return false);
        OP_CHECK_WRONG_DIMENSION(weightOptional, normalizedShape->Size(), return false);
    }
    if (biasOptional) {
        OP_CHECK_MAX_DIM(biasOptional, MAX_DIM_LEN, return false);
        OP_CHECK_WRONG_DIMENSION(biasOptional, normalizedShape->Size(), return false);
    }
    return true;
}

static bool ValidateScaleShiftShapes(const aclTensor* input, const aclTensor* scale, const aclTensor* shift)
{
    auto inputShape = input->GetViewShape();
    int64_t inputDim = inputShape.GetDimNum();
    int64_t B = 1, S = inputShape[inputDim - 2], H = inputShape[inputDim - 1];

    op::Shape expectShape1, expectShape2;
    for (int64_t i = 0; i < inputDim - 2; i++) {
        expectShape1.AppendDim(inputShape[i]);
        expectShape2.AppendDim(inputShape[i]);
        B *= inputShape[i];
    }
    expectShape1.AppendDim(1).AppendDim(H);
    expectShape2.AppendDim(H);

    if (B <= 0 || S <= 0 || H <= 0) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("aclnnAdaLayerNormBackward", "input",
                                              op::ToString(input->GetViewShape()).GetString(),
                                              "input sizes should be greater than 0");
        return false;
    }

    auto scaleShape = scale->GetViewShape();
    auto shiftShape = shift->GetViewShape();

    auto isShapeValid = [&](const op::Shape& shape) { return shape == expectShape1 || shape == expectShape2; };

    if (!isShapeValid(scaleShape) || !isShapeValid(shiftShape)) {
        auto msg = ToString(scale->GetViewShape()).GetString();
        auto expect1 = ToString(expectShape1).GetString();
        auto expect2 = ToString(expectShape2).GetString();
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("aclnnAdaLayerNormBackward", "scale/shift", msg,
                                              (std::string("shape should be ") + expect1 + " or " + expect2).c_str());
        return false;
    }
    return true;
}

static inline bool CheckAllTensorsMaxDim(const aclTensor* gradOut, const aclTensor* input, const aclTensor* mean,
                                         const aclTensor* rstd, const aclTensor* scale, const aclTensor* shift,
                                         const aclTensor* gradInputOut, const aclTensor* gradScaleOut,
                                         const aclTensor* gradShiftOut, const aclTensor* gradWeightOut,
                                         const aclTensor* gradBiasOut)
{
    OP_CHECK_MAX_DIM(gradOut, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(input, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(mean, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(rstd, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(scale, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(shift, MAX_DIM_LEN, return false);

    if (gradInputOut) {
        OP_CHECK_MAX_DIM(gradInputOut, MAX_DIM_LEN, return false);
    }
    if (gradScaleOut) {
        OP_CHECK_MAX_DIM(gradScaleOut, MAX_DIM_LEN, return false);
    }
    if (gradShiftOut) {
        OP_CHECK_MAX_DIM(gradShiftOut, MAX_DIM_LEN, return false);
    }
    if (gradWeightOut) {
        OP_CHECK_MAX_DIM(gradWeightOut, MAX_DIM_LEN, return false);
    }
    if (gradBiasOut) {
        OP_CHECK_MAX_DIM(gradBiasOut, MAX_DIM_LEN, return false);
    }

    return true;
}

static bool CheckOutputShapes(const aclTensor* gradInputOut, const aclTensor* input, const aclTensor* gradScaleOut,
                              const aclTensor* scale, const aclTensor* gradShiftOut, const aclTensor* shift,
                              const aclTensor* gradWeightOut, const aclTensor* gradBiasOut, int64_t N)
{
    struct OutputCheck {
        const aclTensor* out;
        const aclTensor* ref;
        int64_t expectedSize;
        const char* name;
    };

    OutputCheck checks[] = {{gradInputOut, input, 0, "gradInputOut"},
                            {gradScaleOut, scale, 0, "gradScaleOut"},
                            {gradShiftOut, shift, 0, "gradShiftOut"},
                            {gradWeightOut, nullptr, N, "gradWeightOut"},
                            {gradBiasOut, nullptr, N, "gradBiasOut"}};

    for (const auto& check : checks) {
        if (check.out == nullptr)
            continue;

        if (check.ref != nullptr) {
            OP_CHECK(check.out->GetViewShape() == check.ref->GetViewShape(),
                     OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                         "aclnnAdaLayerNormBackward", "check.out,check.ref",
                         (std::string(op::ToString(check.out->GetViewShape()).GetString()) + ", " +
                          op::ToString(check.ref->GetViewShape()).GetString())
                             .c_str(),
                         "shapes of check.out and check.ref should be the same"),
                     return false);
        } else if (check.out->GetViewShape().GetShapeSize() != check.expectedSize) {
            OP_LOGE_FOR_INVALID_SHAPESIZE("aclnnAdaLayerNormBackward", check.name,
                                          ToString(check.out->GetViewShape()).GetString(),
                                          std::to_string(check.expectedSize).c_str());
            return false;
        }
    }
    return true;
}

static bool CheckShape(const aclTensor* gradOut, const aclTensor* input, const aclIntArray* normalizedShape,
                       const aclTensor* rstd, const aclTensor* mean, const aclTensor* scale, const aclTensor* shift,
                       const aclTensor* weightOptional, const aclTensor* biasOptional, const aclTensor* gradInputOut,
                       const aclTensor* gradScaleOut, const aclTensor* gradShiftOut, const aclTensor* gradWeightOut,
                       const aclTensor* gradBiasOut, int64_t N)
{
    // 检查基本张量维度

    if (!CheckAllTensorsMaxDim(gradOut, input, mean, rstd, scale, shift, gradInputOut, gradScaleOut, gradShiftOut,
                               gradWeightOut, gradBiasOut))
        return false;

    OP_CHECK(gradOut->GetViewShape() == input->GetViewShape(),
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("aclnnAdaLayerNormBackward", "gradOut,input",
                                                    (std::string(op::ToString(gradOut->GetViewShape()).GetString()) +
                                                     ", " + op::ToString(input->GetViewShape()).GetString())
                                                        .c_str(),
                                                    "shapes of gradOut and input should be the same"),
             return false);
    OP_CHECK(mean->GetViewShape() == rstd->GetViewShape(),
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("aclnnAdaLayerNormBackward", "mean,rstd",
                                                    (std::string(op::ToString(mean->GetViewShape()).GetString()) +
                                                     ", " + op::ToString(rstd->GetViewShape()).GetString())
                                                        .c_str(),
                                                    "shapes of mean and rstd should be the same"),
             return false);

    // 检查normalizedShape
    if (normalizedShape->Size() != NORMALIZED_SHAPE_LEN) {
        OP_LOGE_FOR_INVALID_LISTSIZE("aclnnAdaLayerNormBackward", "normalizedShape",
                                     std::to_string(normalizedShape->Size()).c_str(),
                                     std::to_string(NORMALIZED_SHAPE_LEN).c_str());
        return false;
    }

    // 检查维度约束
    OP_CHECK_MIN_DIM(input, NORMALIZED_SHAPE_LEN + 1, return false);

    // 检查shape相等性
    if (!CheckShapeEqual(input, normalizedShape))
        return false;

    // 检查weight和bias
    if (!CheckShapeCompatibility(weightOptional, biasOptional, normalizedShape))
        return false;
    if (weightOptional != nullptr) {
        CheckShapeEqual(weightOptional, normalizedShape);
    }
    if (biasOptional != nullptr) {
        CheckShapeEqual(biasOptional, normalizedShape);
    }

    // 检查scale/shift

    if (!ValidateScaleShiftShapes(input, scale, shift))
        return false;

    // 检查mean shape
    auto inputShape = input->GetViewShape();
    int64_t inputDim = inputShape.GetDimNum();
    inputShape[inputDim - 1] = 1;
    if (mean->GetViewShape() != inputShape) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            "aclnnAdaLayerNormBackward", "mean", ToString(mean->GetViewShape()).GetString(),
            (std::string("shape of mean should be ") + ToString(inputShape).GetString()).c_str());
        return false;
    }

    // 检查输出shapes
    return CheckOutputShapes(gradInputOut, input, gradScaleOut, scale, gradShiftOut, shift, gradWeightOut, gradBiasOut,
                             N);
}

static aclnnStatus CheckParams(const aclTensor* gradOut, const aclTensor* input, const aclIntArray* normalizedShape,
                               const aclTensor* rstd, const aclTensor* mean, const aclTensor* scale,
                               const aclTensor* shift, const aclTensor* weightOptional, const aclTensor* biasOptional,
                               aclTensor* gradInputOut, aclTensor* gradScaleOut, aclTensor* gradShiftOut,
                               aclTensor* gradWeightOut, aclTensor* gradBiasOut, int64_t N)
{
    // 0. 格式校验：私有格式（如 NZ）会被当连续数据处理，必须前置拒绝
    CHECK_RET(CheckFormat(gradOut, input, rstd, mean, scale, shift, weightOptional, biasOptional, gradInputOut,
                          gradScaleOut, gradShiftOut, gradWeightOut, gradBiasOut),
              ACLNN_ERR_PARAM_INVALID);
    // 1. 检查数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtype(gradOut, input, rstd, mean, scale, shift, weightOptional, biasOptional, gradInputOut,
                         gradScaleOut, gradShiftOut, gradWeightOut, gradBiasOut),
              ACLNN_ERR_PARAM_INVALID);
    // 2. 检查入参间的shape关系
    CHECK_RET(CheckShape(gradOut, input, normalizedShape, rstd, mean, scale, shift, weightOptional, biasOptional,
                         gradInputOut, gradScaleOut, gradShiftOut, gradWeightOut, gradBiasOut, N),
              ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus GenOutWithMask(const aclTensor* tempOut, const aclTensor* output, bool mask, aclOpExecutor* executor)
{
    if (mask) {
        OP_LOGD("Entering into GenOutWithMask Func.");
        CHECK_RET(tempOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto tempCast = l0op::Cast(tempOut, output->GetDataType(), executor);
        CHECK_RET(tempCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto outputViewCopy = l0op::ViewCopy(tempCast, output, executor);
        CHECK_RET(outputViewCopy != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

static std::pair<int64_t, int64_t> CalculateMN(const aclTensor* input, const aclIntArray* normalizedShape)
{
    auto inputShape = input->GetViewShape();
    int64_t inputDim = inputShape.GetDimNum();
    int64_t normDim = normalizedShape->Size();
    int64_t beginAxis = inputDim - normDim;
    int64_t M = 1, N = 1;

    for (int64_t i = 0; i < inputDim; i++) {
        (i < beginAxis ? M : N) *= inputShape.GetDim(i);
    }
    return {M, N};
}

ge::DataType infer_dtype(const aclTensor* weightOptional, const aclTensor* biasOptional, aclTensor* gradWeightOut,
                         aclTensor* gradBiasOut)
{
    if (weightOptional) {
        return weightOptional->GetDataType();
    } else if (biasOptional) {
        return biasOptional->GetDataType();
    } else if (gradWeightOut) {
        return gradWeightOut->GetDataType();
    } else if (gradBiasOut) {
        return gradBiasOut->GetDataType();
    } else {
        return DataType::DT_FLOAT;
    }
}

static aclnnStatus CreateContiguousTensorsAndComputeGrad(
    const aclTensor* gradOut, const aclTensor* input, const aclTensor* mean, const aclTensor* rstd,
    const aclTensor* scale, const aclTensor* shift, const aclTensor* weightOptional, const aclTensor* biasOptional,
    const aclIntArray* normalizedShape, aclTensor* gradInputOut, aclTensor* gradScaleOut, aclTensor* gradShiftOut,
    aclTensor* gradWeightOut, aclTensor* gradBiasOut, aclOpExecutor* executor)
{
    auto gradOutContiguous = l0op::Contiguous(gradOut, executor);
    CHECK_RET(gradOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto inputContiguous = l0op::Contiguous(input, executor);
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto meanContiguous = l0op::Contiguous(mean, executor);
    CHECK_RET(meanContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto rstdContiguous = l0op::Contiguous(rstd, executor);
    CHECK_RET(rstdContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto scaleContiguous = l0op::Contiguous(scale, executor);
    CHECK_RET(scaleContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto shiftContiguous = l0op::Contiguous(shift, executor);
    CHECK_RET(shiftContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* weightContiguous = nullptr;
    if (weightOptional) {
        weightContiguous = l0op::Contiguous(weightOptional, executor);
    } else {
        auto dtype = infer_dtype(weightOptional, biasOptional, gradWeightOut, gradBiasOut);
        auto weightTensor = executor->ConvertToTensor(normalizedShape, DataType::DT_INT64);
        aclScalar* scalarOne = executor->AllocScalar(1);
        auto oneTensor = executor->ConvertToTensor(scalarOne, dtype);
        weightContiguous = l0op::Fill(weightTensor, oneTensor, normalizedShape, executor);
    }

    CHECK_RET(weightContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* biasContiguous = nullptr;
    if (biasOptional) {
        biasContiguous = l0op::Contiguous(biasOptional, executor);
    } else {
        auto dtype = infer_dtype(weightOptional, biasOptional, gradWeightOut, gradBiasOut);
        auto biasTensor = executor->ConvertToTensor(normalizedShape, DataType::DT_INT64);
        aclScalar* scalarZero = executor->AllocScalar(0);
        auto zeroTensor = executor->ConvertToTensor(scalarZero, dtype);
        biasContiguous = l0op::Fill(biasTensor, zeroTensor, normalizedShape, executor);
    }

    CHECK_RET(biasContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    bool gradCompute = curArch == NpuArch::DAV_2201 || Ops::NN::AclnnUtil::IsRegbase(curArch);
    if (gradCompute) {
        OP_LOGD("Entering into ada_layer_norm_grad Func.");
        // AdaLayerNormGrad只支持fp32 rstd mean输入，如果不是fp32先转fp32
        auto rstdContiguousFp32 = l0op::Cast(rstdContiguous, DataType::DT_FLOAT, executor);
        CHECK_RET(rstdContiguousFp32 != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto meanContiguousFp32 = l0op::Cast(meanContiguous, DataType::DT_FLOAT, executor);
        CHECK_RET(meanContiguousFp32 != nullptr, ACLNN_ERR_INNER_NULLPTR);

        std::array<aclTensor*, GRAD_OUT_NUM> gradRes = l0op::AdaLayerNormGrad(
            gradOutContiguous, inputContiguous, rstdContiguousFp32, meanContiguousFp32, scaleContiguous,
            shiftContiguous, weightContiguous, biasContiguous, executor);

        GenOutWithMask(gradRes[GRAD_INPUT_INDEX], gradInputOut, gradInputOut != nullptr, executor);
        GenOutWithMask(gradRes[GRAD_SCALE_INDEX], gradScaleOut, gradScaleOut != nullptr, executor);
        GenOutWithMask(gradRes[GRAD_SHIFT_INDEX], gradShiftOut, gradShiftOut != nullptr, executor);
        GenOutWithMask(gradRes[GRAD_WEIGHT_INDEX], gradWeightOut, gradWeightOut != nullptr, executor);
        GenOutWithMask(gradRes[GRAD_BIAS_INDEX], gradBiasOut, gradBiasOut != nullptr, executor);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAdaLayerNormBackwardGetWorkspaceSize(
    const aclTensor* gradOut, const aclTensor* input, const aclIntArray* normalizedShape, const aclTensor* rstd,
    const aclTensor* mean, const aclTensor* scale, const aclTensor* shift, const aclTensor* weightOptional,
    const aclTensor* biasOptional, aclTensor* gradInputOut, aclTensor* gradScaleOut, aclTensor* gradShiftOut,
    aclTensor* gradWeightOut, aclTensor* gradBiasOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnAdaLayerNormBackward,
                   DFX_IN(gradOut, input, normalizedShape, rstd, mean, scale, shift, weightOptional, biasOptional),
                   DFX_OUT(gradInputOut, gradScaleOut, gradShiftOut, gradWeightOut, gradBiasOut));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // 提前检查参数是否为空指针，防止input为空指针时获取input的shape报错
    CHECK_RET(CheckNotNull(gradOut, input, normalizedShape, mean, rstd, scale, shift), ACLNN_ERR_PARAM_NULLPTR);

    // 根据input_shape和normalizedShape的关系进行校验和后处理

    auto [M, N] = CalculateMN(input, normalizedShape);

    // 固定写法，参数检查
    auto ret = CheckParams(gradOut, input, normalizedShape, rstd, mean, scale, shift, weightOptional, biasOptional,
                           gradInputOut, gradScaleOut, gradShiftOut, gradWeightOut, gradBiasOut, N);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (gradInputOut == nullptr && gradWeightOut == nullptr && gradBiasOut == nullptr && gradScaleOut == nullptr &&
        gradShiftOut == nullptr) {
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 空tensor场景处理
    if (M <= 0 || N <= 0) {
        if (gradWeightOut != nullptr && M <= 0) {
            ret = op::ProcessEmptyTensorWithValue(gradWeightOut, 0, uniqueExecutor.get());
            CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
        }

        if (gradBiasOut != nullptr && M <= 0) {
            ret = op::ProcessEmptyTensorWithValue(gradBiasOut, 0, uniqueExecutor.get());
            CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
        }

        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    ret = CreateContiguousTensorsAndComputeGrad(gradOut, input, mean, rstd, scale, shift, weightOptional, biasOptional,
                                                normalizedShape, gradInputOut, gradScaleOut, gradShiftOut,
                                                gradWeightOut, gradBiasOut, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAdaLayerNormBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnAdaLayerNormBackward);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
