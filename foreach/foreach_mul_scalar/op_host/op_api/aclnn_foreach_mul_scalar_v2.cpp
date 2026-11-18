/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_common/op_host/util/op_const_def.h"
#include <vector>
#include "aclnn_foreach_mul_scalar_v2.h"
#include "foreach_mul_scalar_v2.h"
#include "foreach_contiguous_helper.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/aclnn_util.h"
#include "op_api/op_api_def_nn.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "../../foreach_utils/op_host/foreach_null_check_helper.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> MUL_SCALAR_V2_TENSOR_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_INT32};

static const std::initializer_list<DataType> FOREACH_SCALAR_FLOAT_SUPPORT_LIST = {DataType::DT_FLOAT,
                                                                                  DataType::DT_DOUBLE};

static const std::initializer_list<DataType> FOREACH_SCALAR_FLOAT16_SUPPORT_LIST = {DataType::DT_FLOAT16,
                                                                                    DataType::DT_DOUBLE};

static const std::initializer_list<DataType> FOREACH_SCALAR_FLOAT16_SUPPORT_LIST_910D = {
    DataType::DT_FLOAT16, DataType::DT_FLOAT, DataType::DT_DOUBLE};

static const std::initializer_list<DataType> FOREACH_SCALAR_INT_SUPPORT_LIST = {DataType::DT_INT32, DataType::DT_INT64};

static const std::initializer_list<DataType> EMPTY_LIST = {};
static inline bool CheckNotNull(const aclTensorList* mulScalarV2Self, const aclScalar* scalar, const aclTensorList* out)
{
    OP_CHECK_NULL(mulScalarV2Self, return false);
    OP_CHECK_NULL(scalar, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckFormat(const aclTensorList* mulScalarV2Self, const aclTensorList* out)
{
    for (uint64_t j = 0; j < mulScalarV2Self->Size(); j++) {
        // self格式不能是私有格式
        if (IsPrivateFormat((*mulScalarV2Self)[j]->GetStorageFormat()) ||
            IsPrivateFormat((*out)[j]->GetStorageFormat())) {
            OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s",
                                   "Format only support ND, NCHW, NHWC, HWCN, NDHWC, NCDHW.");
            REPORT_PREDEFINED_ERR_MSG(
                "EZ0035", (std::vector<const char*>{"param_name", "op_name", "incorrect_format", "reason"}),
                (std::vector<const char*>{
                    std::string((IsPrivateFormat((*mulScalarV2Self)[j]->GetStorageFormat()) ? "x" : "out")).c_str(),
                    "aclnnForeachMulScalarV2",
                    (std::to_string(static_cast<int>((IsPrivateFormat((*mulScalarV2Self)[j]->GetStorageFormat()) ?
                                                          (*mulScalarV2Self)[j]->GetStorageFormat() :
                                                          (*out)[j]->GetStorageFormat()))))
                        .c_str(),
                    "Format only support ND, NCHW, NHWC, HWCN, NDHWC, NCDHW."}));
            return false;
        }
    }
    return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetCurNpuArch() == Ops::Base::DAV_2201 || Ops::NN::AclnnUtil::IsRegbase()) {
        return MUL_SCALAR_V2_TENSOR_DTYPE_SUPPORT_LIST;
    } else {
        const std::string reason = (("support for arch " +
                                     std::to_string(static_cast<uint32_t>(GetCurrentPlatformInfo().GetCurNpuArch())) +
                                     " is not implemented"));
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_RUNTIME_ERROR, "%s", reason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ3002", (std::vector<const char*>{"optype", "opskernel", "reason"}),
                                  (std::vector<const char*>{"ForeachMulScalar", "AiCore", reason.c_str()}));
        return EMPTY_LIST;
    }
}

static inline bool CheckDtype(const aclTensorList* mulScalarV2Self, const aclScalar* scalar, const aclTensorList* out)
{
    const auto& dtypeSupportList = GetDtypeSupportList();
    if (dtypeSupportList.size() == 0) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "support for arch %u is not implemented",
                               static_cast<uint32_t>(GetCurrentPlatformInfo().GetCurNpuArch()));
        return false;
    }
    if (mulScalarV2Self->Size() == 0) {
        return true;
    }

    // checkself input dtype, and check the releation of input and out
    OP_CHECK_NULL((*mulScalarV2Self)[0], return false);
    auto selfDtyte = (*mulScalarV2Self)[0]->GetDataType();
    OP_CHECK_DTYPE_NOT_SUPPORT((*mulScalarV2Self)[0], dtypeSupportList, return false);
    for (uint64_t i = 0; i < mulScalarV2Self->Size(); i++) {
        OP_CHECK_NULL((*mulScalarV2Self)[i], return false);
        OP_CHECK_DTYPE_NOT_MATCH((*mulScalarV2Self)[i], selfDtyte, return false);
    }

    for (uint64_t i = 0; i < out->Size(); i++) {
        OP_CHECK_NULL((*out)[i], return false);
        OP_CHECK_DTYPE_NOT_MATCH((*out)[i], selfDtyte, return false);
    }

    // check the releation of self and scalar
    if (selfDtyte == DataType::DT_BF16 || selfDtyte == DataType::DT_FLOAT) {
        OP_CHECK_DTYPE_NOT_SUPPORT(scalar, FOREACH_SCALAR_FLOAT_SUPPORT_LIST, return false);
    } else if (selfDtyte == DataType::DT_FLOAT16) {
        if (Ops::NN::AclnnUtil::IsRegbase()) {
            OP_CHECK_DTYPE_NOT_SUPPORT(scalar, FOREACH_SCALAR_FLOAT16_SUPPORT_LIST_910D, return false);
        } else {
            OP_CHECK_DTYPE_NOT_SUPPORT(scalar, FOREACH_SCALAR_FLOAT16_SUPPORT_LIST, return false);
        }
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(scalar, FOREACH_SCALAR_INT_SUPPORT_LIST, return false);
    }
    return true;
}

static inline bool CheckShape(const aclTensorList* mulScalarV2Self, const aclTensorList* out)
{
    // tensorlist size检查
    OP_CHECK_TENSORLIST_SIZE_EQUAL(mulScalarV2Self, out, return false);

    // tensor 维度检查
    for (uint64_t n = 0; n < mulScalarV2Self->Size(); n++) {
        OP_CHECK_MAX_DIM((*mulScalarV2Self)[n], MAX_SUPPORT_DIMS_NUMS, return false);
    }

    // self和out的shape必须一致
    for (uint64_t n = 0; n < mulScalarV2Self->Size(); n++) {
        OP_CHECK_SHAPE_NOT_EQUAL((*mulScalarV2Self)[n], (*out)[n], return false);
    }
    return true;
}

static inline aclnnStatus CheckParams(const aclTensorList* mulScalarV2Self, const aclScalar* scalar,
                                      const aclTensorList* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(mulScalarV2Self, scalar, out), ACLNN_ERR_PARAM_NULLPTR);

    // Check every entry in tensor lists is not null, to avoid null pointer
    // dereference in CheckDtype/CheckShape/CheckFormat.
    CHECK_RET(CheckTensorListNotNull(mulScalarV2Self, "self"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorListNotNull(out, "out"), ACLNN_ERR_PARAM_INVALID);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtype(mulScalarV2Self, scalar, out), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查shape是否满足约束
    CHECK_RET(CheckShape(mulScalarV2Self, out), ACLNN_ERR_PARAM_INVALID);
    // 4. 检查Format是否满足约束
    CHECK_RET(CheckFormat(mulScalarV2Self, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecForeachMulScalarV2GetWorkspaceSize(const aclTensorList* x, const aclScalar* scalar,
                                                          const aclTensorList* out, uint64_t* workspaceSize,
                                                          aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto mulScalarV2Executor = CREATE_EXECUTOR();
    CHECK_RET(mulScalarV2Executor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret_1 = CheckParams(x, scalar, out);
    CHECK_RET(ret_1 == ACLNN_SUCCESS, ret_1);

    // 空Tensorlist处理
    if (x->Size() == 0) {
        *workspaceSize = 0;
        mulScalarV2Executor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // self如果非连续，需要转连续
    auto contiguousTensors = ForeachMakeContiguousTensorList(x, mulScalarV2Executor.get());
    CHECK_RET(contiguousTensors != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Allocate temporary outputs without copying the old output data.
    std::vector<const aclTensor*> outputTensors;
    outputTensors.reserve(out->Size());
    for (size_t i = 0; i < out->Size(); ++i) {
        auto outTensor = (*out)[i];
        if (op::IsContiguous(outTensor) && outTensor->GetStorageShape() == outTensor->GetViewShape()) {
            outputTensors.push_back(outTensor);
        } else {
            auto tempOut = mulScalarV2Executor->AllocTensor(outTensor->GetViewShape(), outTensor->GetDataType());
            CHECK_RET(tempOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
            outputTensors.push_back(tempOut);
        }
    }
    auto contiguousOut = mulScalarV2Executor->AllocTensorList(outputTensors.data(), outputTensors.size());
    CHECK_RET(contiguousOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // sclar to tensor
    const aclTensor* otherTensor;
    if ((*x)[0]->GetDataType() == DataType::DT_BF16) {
        otherTensor = mulScalarV2Executor.get()->ConvertToTensor(scalar, DataType::DT_FLOAT);
    } else if (Ops::NN::AclnnUtil::IsRegbase() && (*x)[0]->GetDataType() == DataType::DT_FLOAT16 &&
               (scalar->GetDataType() == DataType::DT_FLOAT || scalar->GetDataType() == DataType::DT_DOUBLE)) {
        otherTensor = mulScalarV2Executor.get()->ConvertToTensor(scalar, DataType::DT_FLOAT);
    } else {
        otherTensor = mulScalarV2Executor.get()->ConvertToTensor(scalar, (*x)[0]->GetDataType());
    }
    CHECK_RET(otherTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0算子ForeachMulScalarV2进行计算
    auto result = l0op::ForeachMulScalarV2(contiguousTensors, otherTensor, contiguousOut, mulScalarV2Executor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_INVALID);

    // copy contiguousOut to non-contiguous out
    CHECK_RET(ForeachViewCopyToOutputTensorList(contiguousOut, out, mulScalarV2Executor.get()),
              ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = mulScalarV2Executor->GetWorkspaceSize();
    mulScalarV2Executor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnForeachMulScalarV2GetWorkspaceSize(const aclTensorList* x, const aclScalar* scalar, aclTensorList* out,
                                                    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnForeachMulScalarV2, DFX_IN(x, scalar), DFX_OUT(out));
    return ExecForeachMulScalarV2GetWorkspaceSize(x, scalar, out, workspaceSize, executor);
}

aclnnStatus aclnnForeachMulScalarV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnForeachMulScalarV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
