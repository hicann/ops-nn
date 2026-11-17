/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_dequant_situ_quant.h"

#include <string>
#include <vector>
#include "aclnnInner_dequant_situ_quant.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "base/err_msg.h"
#include "op_api/aclnn_util.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr size_t MIN_DIM_NUM_INT8 = 2;
constexpr size_t MAX_DIM_NUM = 8;
constexpr size_t DIM_NUM_NOT_INT8 = 2;
constexpr int64_t NUM_TWO = 2;
constexpr double BETA_ZERO = 0.0;
const char* const ACLNN_OP_NAME = "aclnnDequantSituQuant";

static const std::initializer_list<op::DataType> X_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT8, op::DataType::DT_INT32, op::DataType::DT_BF16, op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> FLOAT_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> Y_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT8};
static const std::initializer_list<op::DataType> GROUP_INDEX_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT64};

enum class QuantType : int32_t { STATIC = 0, DYNAMIC = 1 };

static void ReportInvalidValue(const char* paramName, const std::string& incorrectValue, const std::string& reason)
{
    OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
    REPORT_PREDEFINED_ERR_MSG(
        "EZ0026", (std::vector<const char*>{"param_name", "op_name", "incorrect_value", "reason"}),
        (std::vector<const char*>{paramName, ACLNN_OP_NAME, incorrectValue.c_str(), reason.c_str()}));
}

static void ReportInvalidShape(const char* paramNames, const std::string& incorrectShapes, const std::string& reason)
{
    OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
    REPORT_PREDEFINED_ERR_MSG(
        "EZ0010", (std::vector<const char*>{"param_names", "op_name", "incorrect_shapes", "reason"}),
        (std::vector<const char*>{paramNames, ACLNN_OP_NAME, incorrectShapes.c_str(), reason.c_str()}));
}

static bool HasUnknownDim(const op::Shape& shape)
{
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        if (shape.GetDim(i) < 0) {
            return true;
        }
    }
    return false;
}

// 1. 检查必选输入/输出是否为空指针
static bool CheckNotNull(const aclTensor* x, const aclTensor* yOut, const aclTensor* yScaleOut)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(yOut, return false);
    OP_CHECK_NULL(yScaleOut, return false);
    return true;
}

// 2. 检查条件必选输入是否为空指针：INT8/INT32必选weightScaleOptional，INT32必选activationScaleOptional，
// static量化必选quantScaleOptional
static bool CheckConditionalInputNotNull(const aclTensor* x, const aclTensor* weightScaleOptional,
                                         const aclTensor* activationScaleOptional, const aclTensor* quantScaleOptional,
                                         QuantType quantType)
{
    const op::DataType xDtype = x->GetDataType();
    if (xDtype == op::DataType::DT_INT8 || xDtype == op::DataType::DT_INT32) {
        OP_CHECK_NULL(weightScaleOptional, return false);
    }
    if (xDtype == op::DataType::DT_INT32) {
        OP_CHECK_NULL(activationScaleOptional, return false);
    }
    if (quantType == QuantType::STATIC) {
        OP_CHECK_NULL(quantScaleOptional, return false);
    }
    return true;
}

// 3. 检查quantTypeOptional取值，nullptr按默认值dynamic处理，取值大小写不敏感
static bool CheckQuantType(char* quantTypeOptional, QuantType& quantType)
{
    quantType = QuantType::DYNAMIC;
    if (quantTypeOptional == nullptr) {
        return true;
    }
    std::string quantTypeStr{quantTypeOptional};
    for (size_t i = 0; i < quantTypeStr.size(); i++) {
        if (quantTypeStr[i] >= 'A' && quantTypeStr[i] <= 'Z') {
            quantTypeStr[i] = static_cast<char>(quantTypeStr[i] - 'A' + 'a');
        }
    }
    if (quantTypeStr == "static") {
        quantType = QuantType::STATIC;
        return true;
    }
    if (quantTypeStr == "dynamic") {
        quantType = QuantType::DYNAMIC;
        return true;
    }
    const std::string reason = (("quantTypeOptional only supports \"static\" or \"dynamic\", but got " +
                                 std::string(quantTypeOptional) + "."));
    ReportInvalidValue("quantTypeOptional", std::string(quantTypeOptional), reason);
    return false;
}

// 4. 检查x、yOut、yScaleOut是否为空tensor
static bool CheckEmptyTensor(const aclTensor* x, const aclTensor* yOut, const aclTensor* yScaleOut)
{
    if (x->IsEmpty() || yOut->IsEmpty() || yScaleOut->IsEmpty()) {
        const std::string reason = (("x, yOut and yScaleOut do not support empty tensor, x: " +
                                     std::string(op::ToString(x->GetViewShape()).GetString()) + ", yOut: " +
                                     std::string(op::ToString(yOut->GetViewShape()).GetString()) + ", yScaleOut: " +
                                     std::string(op::ToString(yScaleOut->GetViewShape()).GetString()) + "."));
        ReportInvalidShape("x,yOut,yScaleOut",
                           (std::string(op::ToString(x->GetViewShape()).GetString()) + "," +
                            op::ToString(yOut->GetViewShape()).GetString() + "," +
                            op::ToString(yScaleOut->GetViewShape()).GetString()),
                           reason);
        return false;
    }
    return true;
}

// 5. 检查beta是否为0
static bool CheckBeta(double beta)
{
    if (beta == BETA_ZERO) {
        const std::string reason = "beta must not be 0.";
        ReportInvalidValue("beta", std::to_string(beta), reason);
        return false;
    }
    return true;
}

// 6. 检查输入/输出数据类型是否在支持范围内
static bool CheckDtypeValid(const aclTensor* x, const aclTensor* weightScaleOptional,
                            const aclTensor* activationScaleOptional, const aclTensor* biasOptional,
                            const aclTensor* quantScaleOptional, const aclTensor* quantOffsetOptional,
                            const aclTensor* groupIndexOptional, const aclTensor* yOut, const aclTensor* yScaleOut)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, X_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(yOut, Y_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(yScaleOut, FLOAT_DTYPE_SUPPORT_LIST, return false);
    if (weightScaleOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(weightScaleOptional, FLOAT_DTYPE_SUPPORT_LIST, return false);
    }
    if (activationScaleOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(activationScaleOptional, FLOAT_DTYPE_SUPPORT_LIST, return false);
    }
    if (biasOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(biasOptional, FLOAT_DTYPE_SUPPORT_LIST, return false);
    }
    if (quantScaleOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(quantScaleOptional, FLOAT_DTYPE_SUPPORT_LIST, return false);
    }
    if (quantOffsetOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(quantOffsetOptional, FLOAT_DTYPE_SUPPORT_LIST, return false);
    }
    if (groupIndexOptional != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(groupIndexOptional, GROUP_INDEX_DTYPE_SUPPORT_LIST, return false);
    }
    return true;
}

// 7. 检查数据格式是否为线性格式（ND/NCL/NCHW/NCDHW），分形等非线性存储格式拦截
static bool IsLinearFormat(op::Format format)
{
    return format == op::Format::FORMAT_ND || format == op::Format::FORMAT_NCL || format == op::Format::FORMAT_NCHW ||
           format == op::Format::FORMAT_NCDHW;
}

static bool CheckFormat(const aclTensor* x, const aclTensor* yOut, const aclTensor* yScaleOut)
{
    if (!IsLinearFormat(x->GetStorageFormat()) || !IsLinearFormat(yOut->GetStorageFormat()) ||
        !IsLinearFormat(yScaleOut->GetStorageFormat())) {
        const std::string reason = ((
            "x, yOut and yScaleOut only support linear formats (ND/NCL/NCHW/NCDHW), but got x: " +
            std::string(op::ToString(x->GetStorageFormat()).GetString()) +
            ", yOut: " + std::string(op::ToString(yOut->GetStorageFormat()).GetString()) +
            ", yScaleOut: " + std::string(op::ToString(yScaleOut->GetStorageFormat()).GetString()) + "."));
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
        REPORT_PREDEFINED_ERR_MSG(
            "EZ0018", (std::vector<const char*>{"param_names", "op_name", "incorrect_formats", "reason"}),
            (std::vector<const char*>{"x,yOut,yScaleOut", ACLNN_OP_NAME,
                                      (std::string(op::ToString(x->GetStorageFormat()).GetString()) + "," +
                                       op::ToString(yOut->GetStorageFormat()).GetString() + "," +
                                       op::ToString(yScaleOut->GetStorageFormat()).GetString())
                                          .c_str(),
                                      reason.c_str()}));
        return false;
    }
    return true;
}

// 8. 检查x的维度和最后一维是否为2的倍数
static bool CheckXShape(const aclTensor* x)
{
    const op::Shape& xShape = x->GetViewShape();
    const size_t dimNum = xShape.GetDimNum();
    if (x->GetDataType() == op::DataType::DT_INT8) {
        OP_CHECK_MIN_DIM(x, MIN_DIM_NUM_INT8, return false);
        OP_CHECK_MAX_DIM(x, MAX_DIM_NUM, return false);
    } else if (dimNum != DIM_NUM_NOT_INT8) {
        const std::string reason = (("x shape rank must be 2 for int32/bfloat16/float16, but got " +
                                     std::to_string(dimNum) + "."));
        ReportInvalidShape("x", op::ToString(xShape).GetString(), reason);
        return false;
    }

    // 含未知维(-1)的动态shape无法在host侧确定最后一维，交由执行阶段校验
    if (HasUnknownDim(xShape)) {
        return true;
    }
    const int64_t lastDim = xShape.GetDim(dimNum - 1);
    if (lastDim % NUM_TWO != 0) {
        const std::string reason = (("The last dimension of x must be divisible by 2, but got " +
                                     std::to_string(lastDim) + "."));
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
        REPORT_PREDEFINED_ERR_MSG(
            "EZ0015", (std::vector<const char*>{"param_name", "op_name", "incorrect_size", "reason"}),
            (std::vector<const char*>{("x"), ACLNN_OP_NAME, (std::to_string(lastDim)).c_str(), reason.c_str()}));
        return false;
    }
    return true;
}

// 9. 检查输出shape的维度与x匹配
static bool CheckOutShape(const aclTensor* x, const aclTensor* yOut, const aclTensor* yScaleOut)
{
    const size_t xDimNum = x->GetViewShape().GetDimNum();
    const size_t expectedYDimNum = (x->GetDataType() == op::DataType::DT_INT8) ? xDimNum : DIM_NUM_NOT_INT8;
    const size_t expectedYScaleDimNum = (x->GetDataType() == op::DataType::DT_INT8) ? (xDimNum - 1) :
                                                                                      (DIM_NUM_NOT_INT8 - 1);
    if (yOut->GetViewShape().GetDimNum() != expectedYDimNum) {
        const std::string reason = (("yOut shape rank must be " + std::to_string(expectedYDimNum) + ", but got " +
                                     std::to_string(yOut->GetViewShape().GetDimNum()) + "."));
        ReportInvalidShape("yOut", op::ToString(yOut->GetViewShape()).GetString(), reason);
        return false;
    }
    if (yScaleOut->GetViewShape().GetDimNum() != expectedYScaleDimNum) {
        const std::string reason = (("yScaleOut shape rank must be " + std::to_string(expectedYScaleDimNum) +
                                     ", but got " + std::to_string(yScaleOut->GetViewShape().GetDimNum()) + "."));
        ReportInvalidShape("yScaleOut", op::ToString(yScaleOut->GetViewShape()).GetString(), reason);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* weightScaleOptional,
                               const aclTensor* activationScaleOptional, const aclTensor* biasOptional,
                               const aclTensor* quantScaleOptional, const aclTensor* quantOffsetOptional,
                               const aclTensor* groupIndexOptional, double beta, char* quantTypeOptional,
                               const aclTensor* yOut, const aclTensor* yScaleOut)
{
    // 1. 检查必选输入/输出是否为空指针
    CHECK_RET(CheckNotNull(x, yOut, yScaleOut), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查quantTypeOptional取值
    QuantType quantType = QuantType::DYNAMIC;
    CHECK_RET(CheckQuantType(quantTypeOptional, quantType), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查条件必选输入是否为空指针
    CHECK_RET(
        CheckConditionalInputNotNull(x, weightScaleOptional, activationScaleOptional, quantScaleOptional, quantType),
        ACLNN_ERR_PARAM_NULLPTR);

    // 4. 检查x、yOut、yScaleOut是否为空tensor
    CHECK_RET(CheckEmptyTensor(x, yOut, yScaleOut), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查host侧标量参数beta
    CHECK_RET(CheckBeta(beta), ACLNN_ERR_PARAM_INVALID);

    // 6. 检查输入/输出数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(x, weightScaleOptional, activationScaleOptional, biasOptional, quantScaleOptional,
                              quantOffsetOptional, groupIndexOptional, yOut, yScaleOut),
              ACLNN_ERR_PARAM_INVALID);

    // 7. 检查format是否符合要求
    CHECK_RET(CheckFormat(x, yOut, yScaleOut), ACLNN_ERR_PARAM_INVALID);

    // 8. 检查x和输出的shape维度是否符合约束
    CHECK_RET(CheckXShape(x), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckOutShape(x, yOut, yScaleOut), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnDequantSituQuantGetWorkspaceSize(
    const aclTensor* x, const aclTensor* weightScaleOptional, const aclTensor* activationScaleOptional,
    const aclTensor* biasOptional, const aclTensor* quantScaleOptional, const aclTensor* quantOffsetOptional,
    const aclTensor* groupIndexOptional, double beta, double linearBeta, bool activateLeft, char* quantTypeOptional,
    const aclTensor* yOut, const aclTensor* yScaleOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    auto ret = CheckParams(x, weightScaleOptional, activationScaleOptional, biasOptional, quantScaleOptional,
                           quantOffsetOptional, groupIndexOptional, beta, quantTypeOptional, yOut, yScaleOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return aclnnInnerDequantSituQuantGetWorkspaceSize(x, weightScaleOptional, activationScaleOptional, biasOptional,
                                                      quantScaleOptional, quantOffsetOptional, groupIndexOptional, beta,
                                                      linearBeta, activateLeft, quantTypeOptional, yOut, yScaleOut,
                                                      workspaceSize, executor);
}

aclnnStatus aclnnDequantSituQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    return aclnnInnerDequantSituQuant(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
