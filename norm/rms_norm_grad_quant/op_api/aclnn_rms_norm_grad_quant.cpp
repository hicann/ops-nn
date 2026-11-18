/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <initializer_list>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"

#include "rms_norm_grad_quant.h"
#include "aclnn_rms_norm_grad_quant.h"
#include "log/log.h"

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

namespace RmsNormGradQuantACLNN {
constexpr int32_t IDX_DX = 0;
constexpr int32_t IDX_DGAMMA = 1;

static bool CheckNotNull(const aclTensor* dy, const aclTensor* x, const aclTensor* rstd, const aclTensor* gamma,
                         const aclTensor* scalesX, const aclTensor* dxOut, const aclTensor* dgammaOut)
{
    OP_CHECK(dy != nullptr, ReportNullPointerError(__func__, "dy"), return false);
    OP_CHECK(x != nullptr, ReportNullPointerError(__func__, "x"), return false);
    OP_CHECK(rstd != nullptr, ReportNullPointerError(__func__, "rstd"), return false);
    OP_CHECK(gamma != nullptr, ReportNullPointerError(__func__, "gamma"), return false);
    OP_CHECK(scalesX != nullptr, ReportNullPointerError(__func__, "scalesX"), return false);
    OP_CHECK(dxOut != nullptr, ReportNullPointerError(__func__, "dxOut"), return false);
    OP_CHECK(dgammaOut != nullptr, ReportNullPointerError(__func__, "dgammaOut"), return false);
    return true;
}

static bool CheckTensorDtype(const aclTensor* tensor, const char* paramName,
                             const std::initializer_list<op::DataType>& supportedTypes)
{
    const auto dtype = tensor->GetDataType();
    for (const auto supportedType : supportedTypes) {
        if (dtype == supportedType) {
            return true;
        }
    }
    OP_LOGE_FOR_INVALID_DTYPE("aclnnRmsNormGradQuant", paramName, op::ToString(dtype).GetString(),
                              op::ToString(supportedTypes).GetString());
    return false;
}

static bool CheckDtypeValid(const aclTensor* dy, const aclTensor* x, const aclTensor* rstd, const aclTensor* gamma,
                            const aclTensor* scalesX, const aclTensor* offsetXOptional, const aclTensor* dxOut,
                            const aclTensor* dgammaOut)
{
    const auto dyDtype = dy->GetDataType();
    return CheckTensorDtype(dy, "dy", {op::DataType::DT_FLOAT16, op::DataType::DT_BF16, op::DataType::DT_FLOAT}) &&
           CheckTensorDtype(x, "x", {dyDtype}) && CheckTensorDtype(rstd, "rstd", {op::DataType::DT_FLOAT}) &&
           CheckTensorDtype(gamma, "gamma", {op::DataType::DT_FLOAT, dyDtype}) &&
           CheckTensorDtype(scalesX, "scalesX", {op::DataType::DT_FLOAT, dyDtype}) &&
           (offsetXOptional == nullptr ||
            CheckTensorDtype(offsetXOptional, "offsetXOptional", {op::DataType::DT_INT32})) &&
           CheckTensorDtype(dxOut, "dxOut", {op::DataType::DT_INT8, op::DataType::DT_HIFLOAT8}) &&
           CheckTensorDtype(dgammaOut, "dgammaOut", {op::DataType::DT_FLOAT});
}

} // namespace RmsNormGradQuantACLNN

static const aclTensor* GetTensorContiguous(const aclTensor* opt, aclOpExecutor* executor)
{
    if (opt == nullptr) {
        return nullptr;
    }
    return l0op::Contiguous(opt, executor);
}

aclnnStatus aclnnRmsNormGradQuantGetWorkspaceSize(const aclTensor* dy, const aclTensor* x, const aclTensor* rstd,
                                                  const aclTensor* gamma, const aclTensor* scalesX,
                                                  const aclTensor* offsetXOptional, const char* quantMode, bool divMode,
                                                  aclTensor* dxOut, aclTensor* dgammaOut, uint64_t* workspaceSize,
                                                  aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    OP_LOGD("Enter aclnnRmsNormGradQuantGetWorkspaceSize.");
    L2_DFX_PHASE_1(aclnnRmsNormGradQuant, DFX_IN(dy, x, rstd, gamma, scalesX, offsetXOptional, quantMode, divMode),
                   DFX_OUT(dxOut, dgammaOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    CHECK_RET(RmsNormGradQuantACLNN::CheckNotNull(dy, x, rstd, gamma, scalesX, dxOut, dgammaOut),
              ACLNN_ERR_PARAM_NULLPTR);

    if (Ops::NN::AclnnUtil::IsRegbase()) {
        CHECK_RET(
            RmsNormGradQuantACLNN::CheckDtypeValid(dy, x, rstd, gamma, scalesX, offsetXOptional, dxOut, dgammaOut),
            ACLNN_ERR_PARAM_INVALID);
    }

    // a = 0, r != 0 时下放到算子进行计算，其他情况aclnn直接处理
    auto gammaShape = gamma->GetViewShape();
    auto gammaDimNum = gamma->GetViewShape().GetDimNum();
    for (int64_t i = 0; i < static_cast<int64_t>(gammaDimNum); ++i) {
        if (gammaShape.GetDim(i) == 0) {
            OP_LOGW("Got empty tensor in aclnnRmsNormGradQuant!");
            *workspaceSize = 0;
            uniqueExecutor.ReleaseTo(executor);
            return ACLNN_SUCCESS;
        }
    }

    // Convert inputs to contiguous
    auto dyCont = l0op::Contiguous(dy, uniqueExecutor.get());
    auto xCont = l0op::Contiguous(x, uniqueExecutor.get());
    auto rstdCont = l0op::Contiguous(rstd, uniqueExecutor.get());
    auto gammaCont = l0op::Contiguous(gamma, uniqueExecutor.get());
    auto scalesXCont = l0op::Contiguous(scalesX, uniqueExecutor.get());

    CHECK_RET(dyCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(xCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(rstdCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(gammaCont != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(scalesXCont != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto offsetXCont = GetTensorContiguous(offsetXOptional, uniqueExecutor.get());

    // 从dxOut的dtype推导dstType传给L0
    int32_t dstType = static_cast<int32_t>(dxOut->GetDataType());

    // 创建输出Tensor
    aclTensor* dx_output = uniqueExecutor.get()->AllocTensor(dxOut->GetViewShape(), dxOut->GetDataType(),
                                                             dxOut->GetViewFormat());
    aclTensor* dgamma_output = uniqueExecutor.get()->AllocTensor(dgammaOut->GetViewShape(), dgammaOut->GetDataType(),
                                                                 dgammaOut->GetViewFormat());

    auto outs = l0op::RmsNormGradQuant(dyCont, xCont, rstdCont, gammaCont, scalesXCont, offsetXCont, quantMode, divMode,
                                       dstType, dx_output, dgamma_output, uniqueExecutor.get());
    aclTensor* dxCompute = std::get<RmsNormGradQuantACLNN::IDX_DX>(outs);
    aclTensor* dgammaCompute = std::get<RmsNormGradQuantACLNN::IDX_DGAMMA>(outs);
    CHECK_RET(dxCompute != nullptr && dgammaCompute != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // ViewCopy results to output tensors
    auto viewCopyDx = l0op::ViewCopy(dxCompute, dxOut, uniqueExecutor.get());
    CHECK_RET(viewCopyDx != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyDgamma = l0op::ViewCopy(dgammaCompute, dgammaOut, uniqueExecutor.get());
    CHECK_RET(viewCopyDgamma != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    OP_LOGD("Finish aclnnRmsNormGradQuantGetWorkspaceSize.");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRmsNormGradQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRmsNormGradQuant);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
