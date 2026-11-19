/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_foreach_div_scalar.h"
#include "foreach_div_scalar_v2.h"
#include "../../foreach_utils/op_host/foreach_contiguous_helper.h"
#include "op_api/op_api_def_nn.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/aclnn_util.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "../../foreach_utils/op_host/foreach_null_check_helper.h"
#include <algorithm>
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> ASCEND910BC_TENSOR_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

static const std::initializer_list<DataType> SCALAR_FLOAT_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_DOUBLE};
static const std::initializer_list<DataType> SCALAR_FLOAT16_SUPPORT_LIST = {DataType::DT_FLOAT16, DataType::DT_DOUBLE};

static const std::initializer_list<DataType> EMPTY_LIST = {};

static std::string DtypesToString(const std::initializer_list<DataType>& dtypes)
{
    std::string result;
    for (const auto& dtype : dtypes) {
        if (!result.empty()) {
            result += ", ";
        }
        result += op::ToString(dtype).GetString();
    }
    return result;
}

static std::string ShapeToString(const aclTensor* tensor)
{
    std::string result = "(";
    const auto& shape = tensor->GetViewShape();
    for (int64_t i = 0; i < shape.GetDimNum(); i++) {
        if (i > 0) {
            result += ",";
        }
        result += std::to_string(shape.GetDim(i));
    }
    result += ")";
    return result;
}

static inline bool CheckNotNull(const aclTensorList* self, const aclTensor* scalar, const aclTensorList* out)
{
    OP_CHECK_NULL(out, return false);
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(scalar, return false);
    return true;
}

static inline bool CheckFormat(const aclTensorList* self, const aclTensorList* out)
{
    for (uint64_t i = 0; i < self->Size(); i++) {
        if (IsPrivateFormat((*out)[i]->GetStorageFormat()) || IsPrivateFormat((*self)[i]->GetStorageFormat())) {
            OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s",
                                   "Format only support ND, NCHW, NHWC, HWCN, NDHWC, NCDHW.");
            REPORT_PREDEFINED_ERR_MSG(
                "EZ0035", (std::vector<const char*>{"param_name", "op_name", "incorrect_format", "reason"}),
                (std::vector<const char*>{
                    std::string((IsPrivateFormat((*out)[i]->GetStorageFormat()) ? "out" : "x")).c_str(),
                    "aclnnForeachDivScalar",
                    (std::to_string(static_cast<int>((IsPrivateFormat((*out)[i]->GetStorageFormat()) ?
                                                          (*out)[i]->GetStorageFormat() :
                                                          (*self)[i]->GetStorageFormat()))))
                        .c_str(),
                    "Format only support ND, NCHW, NHWC, HWCN, NDHWC, NCDHW."}));
            return false;
        }
    }
    return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    auto socArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (socArch == NpuArch::DAV_2201 || Ops::NN::AclnnUtil::IsRegbase(socArch)) {
        return ASCEND910BC_TENSOR_DTYPE_DTYPE_SUPPORT_LIST;
    } else {
        return EMPTY_LIST;
    }
}

static inline bool CheckDtypeValid(const aclTensorList* self, const aclTensor* scalar, const aclTensorList* out)
{
    if (self->Size() == 0) {
        return true;
    }

    const auto& dtypeSupportList = GetDtypeSupportList();
    if (dtypeSupportList.size() == 0) {
        const std::string reason = (("support for " +
                                     std::string(op::ToString(GetCurrentPlatformInfo().GetSocVersion()).GetString()) +
                                     " is not implemented"));
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ3002", (std::vector<const char*>{"optype", "opskernel", "reason"}),
                                  (std::vector<const char*>{"ForeachDivScalar", "AiCore", reason.c_str()}));
        return false;
    }

    auto selfDtype = (*self)[0]->GetDataType();
    // dtype 不在支持列表：EZ0019 精确上报（原泛化宏统一报 EZ1001）
    if (std::find(dtypeSupportList.begin(), dtypeSupportList.end(), selfDtype) == dtypeSupportList.end()) {
        const std::string expected = DtypesToString(dtypeSupportList);
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "The dtype of x must be one of [%s], but got %s.",
                               expected.c_str(), op::ToString(selfDtype).GetString());
        REPORT_PREDEFINED_ERR_MSG(
            "EZ0019", (std::vector<const char*>{"param_name", "op_name", "incorrect_dtype", "correct_dtype"}),
            (std::vector<const char*>{"x", "aclnnForeachDivScalar", op::ToString(selfDtype).GetString(),
                                      expected.c_str()}));
        return false;
    }

    if (selfDtype == DataType::DT_BF16 || selfDtype == DataType::DT_FLOAT) {
        if (std::find(SCALAR_FLOAT_SUPPORT_LIST.begin(), SCALAR_FLOAT_SUPPORT_LIST.end(), scalar->GetDataType()) ==
            SCALAR_FLOAT_SUPPORT_LIST.end()) {
            const std::string expected = DtypesToString(SCALAR_FLOAT_SUPPORT_LIST);
            OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "The dtype of scalar must be one of [%s], but got %s.",
                                   expected.c_str(), op::ToString(scalar->GetDataType()).GetString());
            REPORT_PREDEFINED_ERR_MSG(
                "EZ0019", (std::vector<const char*>{"param_name", "op_name", "incorrect_dtype", "correct_dtype"}),
                (std::vector<const char*>{"scalar", "aclnnForeachDivScalar",
                                          op::ToString(scalar->GetDataType()).GetString(), expected.c_str()}));
            return false;
        }
    } else if (selfDtype == DataType::DT_FLOAT16) {
        if (std::find(SCALAR_FLOAT16_SUPPORT_LIST.begin(), SCALAR_FLOAT16_SUPPORT_LIST.end(), scalar->GetDataType()) ==
            SCALAR_FLOAT16_SUPPORT_LIST.end()) {
            const std::string expected = DtypesToString(SCALAR_FLOAT16_SUPPORT_LIST);
            OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "The dtype of scalar must be one of [%s], but got %s.",
                                   expected.c_str(), op::ToString(scalar->GetDataType()).GetString());
            REPORT_PREDEFINED_ERR_MSG(
                "EZ0019", (std::vector<const char*>{"param_name", "op_name", "incorrect_dtype", "correct_dtype"}),
                (std::vector<const char*>{"scalar", "aclnnForeachDivScalar",
                                          op::ToString(scalar->GetDataType()).GetString(), expected.c_str()}));
            return false;
        }
    }

    // dtype 关系不一致：EZ0021 精确上报
    for (uint64_t i = 0; i < out->Size(); i++) {
        auto outDtype = (*out)[i]->GetDataType();
        if (outDtype != selfDtype) {
            const std::string reason = "The dtype of out[" + std::to_string(i) + "] must be the same as that of x.";
            OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
            REPORT_PREDEFINED_ERR_MSG(
                "EZ0021", (std::vector<const char*>{"param_names", "op_name", "incorrect_dtypes", "reason"}),
                (std::vector<const char*>{"out", "aclnnForeachDivScalar", op::ToString(outDtype).GetString(),
                                          reason.c_str()}));
            return false;
        }
    }

    for (uint64_t i = 0; i < self->Size(); i++) {
        auto inDtype = (*self)[i]->GetDataType();
        if (inDtype != selfDtype) {
            const std::string reason = "The dtypes of all tensors in x must be the same, but x[" + std::to_string(i) +
                                       "] is inconsistent with x[0].";
            OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
            REPORT_PREDEFINED_ERR_MSG(
                "EZ0021", (std::vector<const char*>{"param_names", "op_name", "incorrect_dtypes", "reason"}),
                (std::vector<const char*>{"x", "aclnnForeachDivScalar", op::ToString(inDtype).GetString(),
                                          reason.c_str()}));
            return false;
        }
    }
    return true;
}

static inline bool CheckShape(const aclTensorList* self, const aclTensorList* out)
{
    // rank 超限：EZ0012 精确上报（原泛化宏统一报 EZ1001）
    for (uint64_t i = 0; i < self->Size(); i++) {
        if ((*self)[i]->GetViewShape().GetDimNum() > MAX_SUPPORT_DIMS_NUMS) {
            const std::string reason = "The rank of x[" + std::to_string(i) + "] must not exceed " +
                                       std::to_string(MAX_SUPPORT_DIMS_NUMS) + ".";
            OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
            REPORT_PREDEFINED_ERR_MSG(
                "EZ0012", (std::vector<const char*>{"param_name", "op_name", "incorrect_dim", "reason"}),
                (std::vector<const char*>{"x", "aclnnForeachDivScalar",
                                          std::to_string((*self)[i]->GetViewShape().GetDimNum()).c_str(),
                                          reason.c_str()}));
            return false;
        }
    }
    for (uint64_t i = 0; i < out->Size(); i++) {
        if ((*out)[i]->GetViewShape().GetDimNum() > MAX_SUPPORT_DIMS_NUMS) {
            const std::string reason = "The rank of out[" + std::to_string(i) + "] must not exceed " +
                                       std::to_string(MAX_SUPPORT_DIMS_NUMS) + ".";
            OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
            REPORT_PREDEFINED_ERR_MSG(
                "EZ0012", (std::vector<const char*>{"param_name", "op_name", "incorrect_dim", "reason"}),
                (std::vector<const char*>{"out", "aclnnForeachDivScalar",
                                          std::to_string((*out)[i]->GetViewShape().GetDimNum()).c_str(),
                                          reason.c_str()}));
            return false;
        }
    }

    if (self->Size() != out->Size()) {
        const std::string reason = (("Tensor lists must have the same number of tensors; self=" +
                                     std::to_string(self->Size()) + ", out=" + std::to_string(out->Size()) + "."));
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ0038",
                                  (std::vector<const char*>{"param_name", "op_name", "incorrect_size", "reason"}),
                                  (std::vector<const char*>{"out", "aclnnForeachDivScalar",
                                                            (std::to_string(out->Size())).c_str(), reason.c_str()}));
        return false;
    }

    // x[i] 与 out[i] shape 关系：EZ0010 精确上报
    for (uint64_t i = 0; i < self->Size(); i++) {
        if ((*self)[i]->GetViewShape() != (*out)[i]->GetViewShape()) {
            const std::string incorrect = "x[" + std::to_string(i) + "]=" + ShapeToString((*self)[i]) + ", out[" +
                                          std::to_string(i) + "]=" + ShapeToString((*out)[i]);
            const std::string reason = "The shape of out[" + std::to_string(i) + "] must be the same as that of x[" +
                                       std::to_string(i) + "].";
            OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
            REPORT_PREDEFINED_ERR_MSG(
                "EZ0010", (std::vector<const char*>{"param_names", "op_name", "incorrect_shapes", "reason"}),
                (std::vector<const char*>{"x, out", "aclnnForeachDivScalar", incorrect.c_str(), reason.c_str()}));
            return false;
        }
    }
    return true;
}

static inline aclnnStatus CheckParams(const aclTensorList* self, const aclTensor* scalar, const aclTensorList* out)
{
    CHECK_RET(CheckNotNull(self, scalar, out), ACLNN_ERR_PARAM_NULLPTR);

    // Check every entry in tensor lists is not null, to avoid null pointer
    // dereference in CheckDtypeValid/CheckShape/CheckFormat.
    CHECK_RET(CheckTensorListNotNull(self, "self"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorListNotNull(out, "out"), ACLNN_ERR_PARAM_INVALID);

    // scalar 元素个数必须为 1：文档口径为 561002。原路径经 V2 桥接统一映射 561103，
    // 丢失 tiling 阶段语义，故在 API 层前置校验并按文档返回码上报。
    if (scalar->GetViewShape().GetShapeSize() != 1) {
        const int64_t scalarSize = scalar->GetViewShape().GetShapeSize();
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "The element count of scalar must be 1, but got %ld.",
                               scalarSize);
        REPORT_PREDEFINED_ERR_MSG(
            "EZ0014", (std::vector<const char*>{"param_name", "op_name", "incorrect_size", "correct_size"}),
            (std::vector<const char*>{"scalar", "aclnnForeachDivScalar", (std::to_string(scalarSize)).c_str(), "1"}));
        return ACLNN_ERR_PARAM_INVALID;
    }

    CHECK_RET(CheckDtypeValid(self, scalar, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(self, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecForeachDivScalarGetWorkspaceSize(const aclTensorList* x, const aclTensor* scalar,
                                                        const aclTensorList* out, uint64_t* workspaceSize,
                                                        aclOpExecutor** executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(x, scalar, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->Size() == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 输出如果非连续，需要转连续作为kernel输出buffer；连续/空则直接使用
    auto contiguousOut = ForeachMakeContiguousTensorList(out, uniqueExecutor.get());
    CHECK_RET(contiguousOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 输入如果非连续，需要转连续（空tensor直接用，保持索引一致）
    auto contiguousTensors = ForeachMakeContiguousTensorList(x, uniqueExecutor.get());
    CHECK_RET(contiguousTensors != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 复用V2桥接，scalar(aclTensor*)直接透传，输出到连续buffer
    auto result = l0op::ForeachDivScalarV2(contiguousTensors, scalar, contiguousOut, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将连续计算结果拷贝到输出out上，out可能是非连续的tensor（空/连续跳过）
    CHECK_RET(ForeachViewCopyToOutputTensorList(contiguousOut, out, uniqueExecutor.get()), ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnForeachDivScalarGetWorkspaceSize(const aclTensorList* x, const aclTensor* scalar, aclTensorList* out,
                                                  uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnForeachDivScalar, DFX_IN(x, scalar), DFX_OUT(out));
    return ExecForeachDivScalarGetWorkspaceSize(x, scalar, out, workspaceSize, executor);
}

aclnnStatus aclnnForeachDivScalar(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                  const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnForeachDivScalar);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
