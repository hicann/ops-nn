/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "rms_norm_quant.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/op_log.h"
#include "log/log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/platform.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_rms_norm_quant.h"
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

namespace {
static const size_t DIMS_ONE_NUMS = 1;
static const size_t DIMS_TWO_NUMS = 2;
static const size_t MAX_DIM_LEN = 8;
static constexpr int64_t INT4_NUMS_IN_INT32_SPACE = 8;
static constexpr size_t kRmsNormQuantResultNum = 2U; // output tensor count of RmsNormQuantV2
static constexpr int32_t IDX_0 = 0;

static const std::initializer_list<op::DataType> EXTEND_ATB_DTYPE_SUPPORT_LIST_X = {op::DataType::DT_FLOAT16,
                                                                                    op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> ASCEND310_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT8, op::DataType::DT_INT4,
                                                                           op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> OUT_ASCEND310_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT8};
static const std::initializer_list<DataType> EMPTY_LIST = {};
static const std::initializer_list<op::DataType> IN_REG_BASE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> OUT_REG_BASE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_HIFLOAT8,
    op::DataType::DT_INT8,          op::DataType::DT_INT4,        op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> ZP_REG_BASE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16,
    op::DataType::DT_INT8};

struct RmsNormQuantInputTensor {
    const aclTensor* x;
    const aclTensor* gamma;
    const aclTensor* beta;
    const aclTensor* scale;
    const aclTensor* offset;
};

static bool CheckNotNull(RmsNormQuantInputTensor& inputTensor, aclTensor* y)
{
    OP_CHECK(inputTensor.x != nullptr, ReportNullPointerError(__func__, "inputTensor.x"), return false);
    OP_CHECK(inputTensor.gamma != nullptr, ReportNullPointerError(__func__, "inputTensor.gamma"), return false);
    OP_CHECK(inputTensor.beta != nullptr, ReportNullPointerError(__func__, "inputTensor.beta"), return false);
    OP_CHECK(inputTensor.scale != nullptr, ReportNullPointerError(__func__, "inputTensor.scale"), return false);
    OP_CHECK(inputTensor.offset != nullptr, ReportNullPointerError(__func__, "inputTensor.offset"), return false);
    OP_CHECK(y != nullptr, ReportNullPointerError(__func__, "y"), return false);
    return true;
}

static bool CheckShapeBeforeReshape(const aclTensor* tensor, const char* tensorName)
{
    if (tensor != nullptr && tensor->GetViewShape().GetDimNum() == DIMS_TWO_NUMS &&
        tensor->GetViewShape().GetDim(0) != 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnRmsNormQuant", tensorName,
                                                 std::to_string(tensor->GetViewShape().GetDim(0)).c_str(),
                                                 "the first dimension of a 2D tensor must be 1");
        return false;
    }
    return true;
}

static const std::initializer_list<DataType>& GetInDtypeSupportList()
{
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        return IN_REG_BASE_DTYPE_SUPPORT_LIST;
    }
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    switch (socVersion) {
        case SocVersion::ASCEND910_93:
        case SocVersion::ASCEND910B: {
            return DTYPE_SUPPORT_LIST;
        }
        case SocVersion::ASCEND310P:
        case SocVersion::ASCEND310B: {
            return ASCEND310_DTYPE_SUPPORT_LIST;
        }
        default: {
            const std::vector<const char*> ez3002Key = {"optype", "opskernel", "reason"};
            // 先落具名局部变量, 避免临时std::string在vector声明语句结束析构导致.c_str()悬垂
            const std::string ez3002Reason = "the current SoC " + std::string(op::ToString(socVersion).GetString()) +
                                             " is not supported by this implementation; use a supported SoC";
            const std::vector<const char*> ez3002Value = {"RmsNormQuant", "AiCore", ez3002Reason.c_str()};
            REPORT_PREDEFINED_ERR_MSG("EZ3002", ez3002Key, ez3002Value);
            return EMPTY_LIST;
        }
    }
}

static const std::initializer_list<DataType>& GetOutDtypeSupportList()
{
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        return OUT_REG_BASE_DTYPE_SUPPORT_LIST;
    }
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    switch (socVersion) {
        case SocVersion::ASCEND910_93:
        case SocVersion::ASCEND910B: {
            return OUT_DTYPE_SUPPORT_LIST;
        }
        case SocVersion::ASCEND310P:
        case SocVersion::ASCEND310B: {
            return OUT_ASCEND310_DTYPE_SUPPORT_LIST;
        }
        default: {
            const std::vector<const char*> ez3002Key = {"optype", "opskernel", "reason"};
            // 先落具名局部变量, 避免临时std::string在vector声明语句结束析构导致.c_str()悬垂
            const std::string ez3002Reason = "the current SoC " + std::string(op::ToString(socVersion).GetString()) +
                                             " is not supported for this output dtype list; use a supported SoC";
            const std::vector<const char*> ez3002Value = {"RmsNormQuant", "AiCore", ez3002Reason.c_str()};
            REPORT_PREDEFINED_ERR_MSG("EZ3002", ez3002Key, ez3002Value);
            return EMPTY_LIST;
        }
    }
}

static bool CheckDtypeValid(RmsNormQuantInputTensor& inputTensor, const aclTensor* y)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor.x, GetInDtypeSupportList(), return false);
    OP_CHECK_DTYPE_NOT_SAME(inputTensor.gamma, inputTensor.x, return false);
    OP_CHECK_DTYPE_NOT_SAME(inputTensor.beta, inputTensor.x, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(y, GetOutDtypeSupportList(), return false);
    if (!Ops::NN::AclnnUtil::IsRegbase()) {
        OP_CHECK_DTYPE_NOT_SAME(inputTensor.scale, inputTensor.x, return false);
        OP_CHECK_DTYPE_NOT_MATCH(inputTensor.offset, op::DataType::DT_INT8, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(inputTensor.offset, ZP_REG_BASE_DTYPE_SUPPORT_LIST, return false);
        if (inputTensor.x->GetDataType() == op::DataType::DT_FLOAT) {
            OP_CHECK(inputTensor.scale->GetDataType() == op::DataType::DT_FLOAT &&
                         inputTensor.offset->GetDataType() == op::DataType::DT_FLOAT,
                     OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                         "aclnnRmsNormQuant", "scale,offset",
                         (std::string(op::ToString(inputTensor.scale->GetDataType()).GetString()) + ", " +
                          op::ToString(inputTensor.offset->GetDataType()).GetString())
                             .c_str(),
                         "if xType is float32, scaleType and offsetType must be float32"),
                     return false);
        }
        if (inputTensor.x->GetDataType() == op::DataType::DT_FLOAT16 ||
            inputTensor.x->GetDataType() == op::DataType::DT_BF16) {
            if (inputTensor.scale->GetDataType() == op::DataType::DT_FLOAT) {
                OP_CHECK(
                    inputTensor.offset->GetDataType() == op::DataType::DT_FLOAT ||
                        inputTensor.offset->GetDataType() == op::DataType::DT_INT32,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        "aclnnRmsNormQuant", "offset", op::ToString(inputTensor.offset->GetDataType()).GetString(),
                        "if xType is float16/bfloat16 and scaleType is float, offsetType must be float32 or int32"),
                    return false);
            } else {
                OP_CHECK(inputTensor.x->GetDataType() == inputTensor.scale->GetDataType(),
                         OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                             "aclnnRmsNormQuant", "scale", op::ToString(inputTensor.scale->GetDataType()).GetString(),
                             "when xType is float16/bfloat16, scaleType should equal xType if it is not float"),
                         return false);
                OP_CHECK(inputTensor.scale->GetDataType() == inputTensor.offset->GetDataType() ||
                             inputTensor.offset->GetDataType() == op::DataType::DT_INT8,
                         OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                             "aclnnRmsNormQuant", "offset", op::ToString(inputTensor.offset->GetDataType()).GetString(),
                             "if scaleType is the same with xType, offsetType should be equal to scaleType or int8"),
                         return false);
            }
        }
    }
    return true;
}

aclnnStatus PreDealData(RmsNormQuantInputTensor& inputTensor, aclOpExecutor* executor)
{
    if (inputTensor.gamma != nullptr && inputTensor.gamma->GetViewShape().GetDimNum() == DIMS_TWO_NUMS) {
        auto gammaCont = l0op::Contiguous(inputTensor.gamma, executor);
        CHECK_RET(gammaCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto gammaReshape = l0op::Reshape(gammaCont, {gammaCont->GetViewShape()[1]}, executor);
        CHECK_RET(gammaReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        inputTensor.gamma = gammaReshape;
    }
    if (inputTensor.beta != nullptr && inputTensor.beta->GetViewShape().GetDimNum() == DIMS_TWO_NUMS) {
        auto betaCont = l0op::Contiguous(inputTensor.beta, executor);
        CHECK_RET(betaCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto betaReshape = l0op::Reshape(betaCont, {betaCont->GetViewShape()[1]}, executor);
        CHECK_RET(betaReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        inputTensor.beta = betaReshape;
    }
    return ACLNN_SUCCESS;
}

bool CheckShapeDimWithFrontN(const op::Shape& shape1, const op::Shape& shape2, size_t front_n)
{
    size_t x1n = shape1.GetDimNum();
    size_t x2n = shape2.GetDimNum();
    if (x1n != x2n) {
        return false;
    }
    for (size_t i = 0; i < x1n && i < front_n; ++i) {
        if (shape1.GetDim(i) != shape2.GetDim(i)) {
            return false;
        }
    }
    return true;
}

static bool CheckShapeDim(RmsNormQuantInputTensor& inputTensor, const aclTensor* y)
{
    OP_CHECK_MAX_DIM(inputTensor.x, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(inputTensor.gamma, DIMS_TWO_NUMS, return false);
    OP_CHECK_MAX_DIM(inputTensor.beta, DIMS_TWO_NUMS, return false);
    OP_CHECK_MAX_DIM(y, MAX_DIM_LEN, return false);

    OP_CHECK_MIN_DIM(inputTensor.x, DIMS_ONE_NUMS, return false);
    OP_CHECK_MIN_DIM(inputTensor.gamma, DIMS_ONE_NUMS, return false);
    OP_CHECK_MIN_DIM(inputTensor.beta, DIMS_ONE_NUMS, return false);
    OP_CHECK_MIN_DIM(y, DIMS_ONE_NUMS, return false);

    int64_t scaleDimNum = static_cast<int64_t>(inputTensor.scale->GetViewShape().GetDimNum());
    if (scaleDimNum != 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM("aclnnRmsNormQuant", "scale", std::to_string(scaleDimNum).c_str(), "1");
        return false;
    }
    int64_t gammaDimNum = static_cast<int64_t>(inputTensor.gamma->GetViewShape().GetDimNum());
    int64_t gammaLastDim = inputTensor.gamma->GetViewShape().GetDim(gammaDimNum - 1);
    int64_t xDimNum = static_cast<int64_t>(inputTensor.x->GetViewShape().GetDimNum());
    int64_t xLastDim = inputTensor.x->GetViewShape().GetDim(xDimNum - 1);
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        int64_t scaleLength = inputTensor.scale->GetViewShape().GetDim(0);
        OP_CHECK(
            scaleLength == 1 || scaleLength == xLastDim,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                "aclnnRmsNormQuant", "scale", std::to_string(scaleLength).c_str(),
                (std::string("the length of scale must be 1 or the last dim of x (") + std::to_string(xLastDim) + ")")
                    .c_str()),
            return false);
    } else {
        int64_t scaleLength = inputTensor.scale->GetViewShape().GetDim(0);
        OP_CHECK(
            scaleLength == 1,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnRmsNormQuant", "scale", std::to_string(scaleLength).c_str(),
                                                     "the length of scale must be 1 on this platform"),
            return false);
    }
    if (gammaLastDim != xLastDim) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnRmsNormQuant", "x", std::to_string(xLastDim).c_str(),
                                                 "the last dim of x must be the same as gamma and beta");
        return false;
    }
    //  check last dim
    auto outDtype = y->GetDataType();
    int64_t yDimNum = static_cast<int64_t>(y->GetViewShape().GetDimNum());
    int64_t outLastDim = y->GetViewShape().GetDim(yDimNum - 1);
    if (outDtype == op::DataType::DT_INT32) {
        OP_CHECK(CheckShapeDimWithFrontN(inputTensor.x->GetViewShape(), y->GetViewShape(), xDimNum - 1),
                 OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                     "aclnnRmsNormQuant", "x,y",
                     (std::string(op::ToString(inputTensor.x->GetViewShape()).GetString()) + ", " +
                      op::ToString(y->GetViewShape()).GetString())
                         .c_str(),
                     "the first n-1 dims of x and y should be the same"),
                 return false);
        OP_CHECK(
            xLastDim == outLastDim * INT4_NUMS_IN_INT32_SPACE,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnRmsNormQuant", "y", std::to_string(outLastDim).c_str(),
                                                     "if outType is int32, the out last dim must be 1/8 of x last dim"),
            return false);
    } else if (Ops::NN::AclnnUtil::IsRegbase() && outDtype == op::DataType::DT_INT4) {
        OP_CHECK(CheckShapeDimWithFrontN(inputTensor.x->GetViewShape(), y->GetViewShape(), xDimNum),
                 OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                     "aclnnRmsNormQuant", "x,y",
                     (std::string(op::ToString(inputTensor.x->GetViewShape()).GetString()) + ", " +
                      op::ToString(y->GetViewShape()).GetString())
                         .c_str(),
                     "x shape should be the same as y shape"),
                 return false);
        OP_CHECK(xLastDim % 2 == 0,
                 OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnRmsNormQuant", "x", std::to_string(xLastDim).c_str(),
                                                          "if outType is int4 on regbase, x last dim should be even"),
                 return false);
    } else {
        if (Ops::NN::AclnnUtil::IsRegbase()) {
            OP_CHECK(CheckShapeDimWithFrontN(inputTensor.x->GetViewShape(), y->GetViewShape(), xDimNum),
                     OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                         "aclnnRmsNormQuant", "x,y",
                         (std::string(op::ToString(inputTensor.x->GetViewShape()).GetString()) + ", " +
                          op::ToString(y->GetViewShape()).GetString())
                             .c_str(),
                         "x shape should be the same as y shape"),
                     return false);
        } else {
            // A2/A3：INT4 量化结果须以 INT32 打包张量声明（y 最后一维为 x 最后一维的 1/8），
            // 不支持直接以 INT4 数据类型声明输出。
            OP_CHECK(outDtype != op::DataType::DT_INT4,
                     OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                         "aclnnRmsNormQuant", "y", op::ToString(outDtype).GetString(),
                         "on this platform int4 output must be declared as int32 with the last dim of y equal to "
                         "1/8 of the last dim of x"),
                     return false);
            OP_CHECK(CheckShapeDimWithFrontN(inputTensor.x->GetViewShape(), y->GetViewShape(), xDimNum),
                     OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                         "aclnnRmsNormQuant", "x,y",
                         (std::string(op::ToString(inputTensor.x->GetViewShape()).GetString()) + ", " +
                          op::ToString(y->GetViewShape()).GetString())
                             .c_str(),
                         "x shape should be the same as y shape"),
                     return false);
            OP_CHECK(xLastDim == outLastDim,
                     OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                         "aclnnRmsNormQuant", "x", std::to_string(xLastDim).c_str(),
                         "if outType is int8, x last dim must be equal with out last dim"),
                     return false);
        }
    }

    if (Ops::NN::AclnnUtil::IsRegbase()) {
        // Validate [H]/[1, H] before preprocessing, including H=0.
        const auto& betaShape = inputTensor.beta->GetViewShape();
        OP_CHECK(betaShape.GetDim(betaShape.GetDimNum() - 1) == gammaLastDim,
                 OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                     "aclnnRmsNormQuant", "beta", std::to_string(betaShape.GetDim(betaShape.GetDimNum() - 1)).c_str(),
                     "the last dimension of beta must match the last dimension of gamma"),
                 return false);
    } else {
        OP_CHECK(inputTensor.gamma->GetViewShape() == inputTensor.beta->GetViewShape(),
                 OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                     "aclnnRmsNormQuant", "inputTensor.gamma,inputTensor.beta",
                     (std::string(op::ToString(inputTensor.gamma->GetViewShape()).GetString()) + ", " +
                      op::ToString(inputTensor.beta->GetViewShape()).GetString())
                         .c_str(),
                     "shapes of inputTensor.gamma and inputTensor.beta should be the same"),
                 return false);
    }
    OP_CHECK(inputTensor.scale->GetViewShape() == inputTensor.offset->GetViewShape(),
             OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                 "aclnnRmsNormQuant", "inputTensor.scale,inputTensor.offset",
                 (std::string(op::ToString(inputTensor.scale->GetViewShape()).GetString()) + ", " +
                  op::ToString(inputTensor.offset->GetViewShape()).GetString())
                     .c_str(),
                 "shapes of inputTensor.scale and inputTensor.offset should be the same"),
             return false);
    return true;
}

static aclnnStatus CheckParams(RmsNormQuantInputTensor& inputTensor, aclTensor* y)
{
    // 1. 检查必选输入/输出是否为空指针
    CHECK_RET(CheckNotNull(inputTensor, y), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入/输出的数据类型是否合法
    CHECK_RET(CheckDtypeValid(inputTensor, y), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入/输出的shape大小
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        CHECK_RET(
            CheckShapeBeforeReshape(inputTensor.gamma, "gamma") && CheckShapeBeforeReshape(inputTensor.beta, "beta"),
            ACLNN_ERR_PARAM_INVALID);
    }
    CHECK_RET(CheckShapeDim(inputTensor, y), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus Int42Int32PackedTensor(const aclTensor* y, const aclTensor*& outTensor, aclOpExecutor* executor)
{
    // if outType is int32, pack output
    auto viewShape = y->GetViewShape();
    auto viewShapeDim = viewShape.GetDimNum();
    viewShape[viewShapeDim - 1] /= INT4_NUMS_IN_INT32_SPACE;
    auto outTemp = executor->CreateView(y, viewShape, y->GetViewOffset());
    CHECK_RET(outTemp != nullptr, ACLNN_ERR_INNER_NULLPTR);

    outTemp->SetDataType(DataType::DT_INT32);
    outTensor = outTemp;
    OP_LOGD("aclnnRmsNormQuant output real dtype is int4, pack to int32 to out.");

    return ACLNN_SUCCESS;
}

static aclnnStatus RunRmsNormQuantKernel(const aclTensor* xCont, const aclTensor* scaleCont,
                                         const aclTensor* offsetCont, const aclTensor* gamma, const aclTensor* beta,
                                         const aclTensor* regbaseGamma, const aclTensor* regbaseBeta, double epsilon,
                                         int32_t yType, aclTensor* y, aclOpExecutor* executor)
{
    aclnnStatus ret = ACLNN_SUCCESS;
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        bool divMode = true;
        auto gammaCont = l0op::Contiguous(regbaseGamma, executor);
        auto betaCont = l0op::Contiguous(regbaseBeta, executor);
        CHECK_RET(gammaCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(betaCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
        std::array<aclTensor*, kRmsNormQuantResultNum> addRmsNormQuantOuts = l0op::RmsNormQuantV2(
            xCont, gammaCont, scaleCont, nullptr, offsetCont, nullptr, betaCont, epsilon, divMode, yType, executor);
        aclTensor* resultTensor = std::get<IDX_0>(addRmsNormQuantOuts);
        CHECK_RET(resultTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        const aclTensor* outTensor = resultTensor;
        if (yType == op::DataType::DT_INT4 && y->GetDataType() == op::DataType::DT_INT32) {
            ret = Int42Int32PackedTensor(resultTensor, outTensor, executor);
            CHECK_RET(ret == ACLNN_SUCCESS, ret);
            auto viewCopyY = l0op::ViewCopy(outTensor, y, executor);
            CHECK_RET(viewCopyY != nullptr, ACLNN_ERR_INNER_NULLPTR);
        } else {
            auto viewCopyY = l0op::ViewCopy(resultTensor, y, executor);
            CHECK_RET(viewCopyY != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    } else {
        auto gammaCont = l0op::Contiguous(gamma, executor);
        auto betaCont = l0op::Contiguous(beta, executor);
        CHECK_RET(gammaCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(betaCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto resultTensor = l0op::RmsNormQuant(xCont, gammaCont, betaCont, scaleCont, offsetCont, epsilon, yType,
                                               executor);
        const aclTensor* outTensor = resultTensor;
        if (yType == op::DataType::DT_INT4) {
            ret = Int42Int32PackedTensor(resultTensor, outTensor, executor);
            CHECK_RET(ret == ACLNN_SUCCESS, ret);
            auto viewCopyY = l0op::ViewCopy(outTensor, y, executor);
            CHECK_RET(viewCopyY != nullptr, ACLNN_ERR_INNER_NULLPTR);
        } else {
            auto viewCopyY = l0op::ViewCopy(resultTensor, y, executor);
            CHECK_RET(viewCopyY != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRmsNormQuantGetWorkspaceSize(const aclTensor* x, const aclTensor* gamma, const aclTensor* beta,
                                              const aclTensor* scale, const aclTensor* offset, double epsilon,
                                              aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnRmsNormQuant, DFX_IN(x, gamma, beta, scale, offset, epsilon), DFX_OUT(y));
    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    // 参数检查
    RmsNormQuantInputTensor inputTensorOri = {x, gamma, beta, scale, offset};
    const bool isRegbase = Ops::NN::AclnnUtil::IsRegbase();
    if (isRegbase) {
        const auto paramRet = CheckParams(inputTensorOri, y);
        CHECK_RET(paramRet == ACLNN_SUCCESS, paramRet);
        if (x->IsEmpty()) {
            // Shape validation guarantees an empty y. No preprocessing or kernel task is needed.
            *workspaceSize = 0;
            uniqueExecutor.ReleaseTo(executor);
            return ACLNN_SUCCESS;
        }
    }
    CHECK_RET(PreDealData(inputTensorOri, uniqueExecutor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    if (!isRegbase) {
        const auto paramRet = CheckParams(inputTensorOri, y);
        CHECK_RET(paramRet == ACLNN_SUCCESS, paramRet);
        if (x->IsEmpty()) {
            // Shape validation guarantees an empty y. 空Tensor在所有平台均按成功空转处理：
            // 返回0字节workspace，第二段不执行计算。
            *workspaceSize = 0;
            uniqueExecutor.ReleaseTo(executor);
            return ACLNN_SUCCESS;
        }
    }
    int32_t yType = y->GetDataType();
    if (yType == op::DataType::DT_INT32) {
        yType = op::DataType::DT_INT4;
        OP_LOGD("aclnnRmsNormQuant real output is int4.");
    }
    // 固定写法，将输入转换成连续的tensor，可选输入不做判空校验
    auto xCont = l0op::Contiguous(x, uniqueExecutor.get());
    auto offsetCont = l0op::Contiguous(offset, uniqueExecutor.get());
    auto scaleCont = l0op::Contiguous(scale, uniqueExecutor.get());
    CHECK_RET(xCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(offsetCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(scaleCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const auto ret = RunRmsNormQuantKernel(xCont, scaleCont, offsetCont, gamma, beta, inputTensorOri.gamma,
                                           inputTensorOri.beta, epsilon, yType, y, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
aclnnStatus aclnnRmsNormQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRmsNormQuant);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
