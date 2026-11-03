/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include "aclnn_foreach_maximum_scalar_list.h"
#include "foreach_maximum_scalar_list.h"
#include "../../foreach_utils/op_host/foreach_contiguous_helper.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/op_api_def_nn.h"
#include "op_api/aclnn_util.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"
#include "../../foreach_utils/op_host/foreach_null_check_helper.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> ASCEND910BC_TENSOR_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_INT32, DataType::DT_BF16};

static const std::initializer_list<DataType> FOREACH_SCALAR_FLOAT_SUPPORT_LIST = {DataType::DT_FLOAT,
                                                                                  DataType::DT_DOUBLE};

static const std::initializer_list<DataType> FOREACH_SCALAR_INT_SUPPORT_LIST = {DataType::DT_INT32, DataType::DT_INT64};

static const std::initializer_list<DataType> EMPTY_LIST = {};

static inline bool CheckNotNull(const aclTensorList* x, const aclScalarList* scalars, const aclTensorList* out)
{
    OP_CHECK_NULL(x, return false);
    if (scalars == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "scalars is nullptr.");
        return false;
    }
    OP_CHECK_NULL(out, return false);
    return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2201 || Ops::NN::AclnnUtil::IsRegbase(curArch)) {
        return ASCEND910BC_TENSOR_DTYPE_DTYPE_SUPPORT_LIST;
    }
    return EMPTY_LIST;
}

static inline bool CheckFormat(const aclTensorList* x, const aclTensorList* out)
{
    for (uint64_t i = 0; i < x->Size(); i++) {
        // x格式不能是私有格式
        if (IsPrivateFormat((*x)[i]->GetStorageFormat()) || IsPrivateFormat((*out)[i]->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND, NCHW, NHWC, HWCN, NDHWC, NCDHW.");
            return false;
        }
        if ((*x)[i]->GetStorageFormat() != (*out)[i]->GetStorageFormat()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format of x[%lu] and out[%lu] must be the same, x[%s], out[%s].", i, i,
                    op::ToString((*x)[i]->GetStorageFormat()).GetString(),
                    op::ToString((*out)[i]->GetStorageFormat()).GetString());
            return false;
        }
    }
    return true;
}

static inline bool CheckDtypeValid(const aclTensorList* x, const aclScalarList* scalars, const aclTensorList* out)
{
    if (x->Size() == 0) {
        return true;
    }

    if (scalars->Size() != x->Size()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The number of scalars must be the same as the number of tensors");
        return false;
    }

    const auto& dtypeSupportList = GetDtypeSupportList();
    if (dtypeSupportList.size() == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "support for %s is not implemented",
                op::ToString(GetCurrentPlatformInfo().GetSocVersion()).GetString());
        return false;
    }

    // check x input dtype, and check the relation of input and out
    auto selfDtype = (*x)[0]->GetDataType();
    OP_CHECK_DTYPE_NOT_SUPPORT((*x)[0], dtypeSupportList, return false);
    for (uint64_t i = 0; i < x->Size(); i++) {
        OP_CHECK_DTYPE_NOT_MATCH((*x)[i], selfDtype, return false);
    }
    for (uint64_t i = 0; i < out->Size(); i++) {
        OP_CHECK_DTYPE_NOT_MATCH((*out)[i], selfDtype, return false);
    }

    // check the relation of x and scalars
    for (uint64_t i = 0; i < scalars->Size(); i++) {
        OP_CHECK_NULL((*scalars)[i], return false);
    }
    if (selfDtype == DataType::DT_INT32) {
        OP_CHECK_DTYPE_NOT_SUPPORT((*scalars)[0], FOREACH_SCALAR_INT_SUPPORT_LIST, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT((*scalars)[0], FOREACH_SCALAR_FLOAT_SUPPORT_LIST, return false);
    }
    return true;
}

static inline bool CheckShape(const aclTensorList* x, const aclScalarList* scalars, const aclTensorList* out)
{
    if (x->Size() != out->Size()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensor lists must have the same number of tensors");
        return false;
    }
    if (x->Size() != scalars->Size()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The number of scalars must be the same as the number of tensors");
        return false;
    }

    // tensor 维度检查，且 out 的 view shape 必须与 x 完全一致
    for (uint64_t i = 0; i < x->Size(); i++) {
        OP_CHECK_MAX_DIM((*x)[i], MAX_SUPPORT_DIMS_NUMS, return false);
        OP_CHECK_MAX_DIM((*out)[i], MAX_SUPPORT_DIMS_NUMS, return false);
        OP_CHECK_SHAPE_NOT_EQUAL((*out)[i], (*x)[i], return false);
    }
    return true;
}

static inline aclnnStatus CheckParams(const aclTensorList* x, const aclScalarList* scalars, const aclTensorList* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x, scalars, out), ACLNN_ERR_PARAM_NULLPTR);

    // Check every entry in tensor lists is not null, to avoid null pointer
    // dereference in CheckDtypeValid/CheckShape/CheckFormat.
    CHECK_RET(CheckTensorListNotNull(x, "x"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorListNotNull(out, "out"), ACLNN_ERR_PARAM_INVALID);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(x, scalars, out), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查shape是否满足约束
    CHECK_RET(CheckShape(x, scalars, out), ACLNN_ERR_PARAM_INVALID);
    // 4. 检查Format是否满足约束
    CHECK_RET(CheckFormat(x, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static inline const aclTensor* ConvertScalarsToTensor(const aclScalarList* scalars, DataType dataType,
                                                      aclOpExecutor* executor)
{
    // scalars列表在kernel侧是一个shape为[N]的tensor，需将标量列表转换为单个tensor
    if (dataType == DataType::DT_INT64) {
        std::vector<int64_t> scalarsVec;
        scalarsVec.reserve(scalars->Size());
        for (uint64_t i = 0; i < scalars->Size(); i++) {
            scalarsVec.push_back((*scalars)[i]->ToInt64());
        }
        return executor->ConvertToTensor(scalarsVec.data(), scalarsVec.size(), dataType);
    }
    std::vector<float> scalarsVec;
    scalarsVec.reserve(scalars->Size());
    for (uint64_t i = 0; i < scalars->Size(); i++) {
        scalarsVec.push_back((*scalars)[i]->ToFloat());
    }
    return executor->ConvertToTensor(scalarsVec.data(), scalarsVec.size(), dataType);
}

static aclnnStatus ExecForeachMaximumScalarListGetWorkspaceSize(const aclTensorList* x, const aclScalarList* scalars,
                                                                const aclTensorList* out, uint64_t* workspaceSize,
                                                                aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(x, scalars, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensorlist处理
    if (x->Size() == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 输入如果非连续，需要转连续（空tensor直接用，保持索引一致）
    auto contiguousTensors = ForeachMakeContiguousTensorList(x, uniqueExecutor.get());
    CHECK_RET(contiguousTensors != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // scalars转tensor，dtype按x的dtype映射：INT32->INT64，其余->FLOAT
    auto scalarsDtype = ((*x)[0]->GetDataType() == DataType::DT_INT32) ? DataType::DT_INT64 : DataType::DT_FLOAT;
    auto scalarsTensor = ConvertScalarsToTensor(scalars, scalarsDtype, uniqueExecutor.get());
    CHECK_RET(scalarsTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 输出如果非连续，需要转连续作为kernel输出buffer；连续/空则直接使用
    auto contiguousOut = ForeachMakeContiguousTensorList(out, uniqueExecutor.get());
    CHECK_RET(contiguousOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0算子ForeachMaximumScalarList进行计算，输出到连续buffer
    auto result = l0op::ForeachMaximumScalarList(contiguousTensors, scalarsTensor, contiguousOut, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将连续计算结果拷贝到输出out上，out可能是非连续的tensor（空/连续跳过）
    CHECK_RET(ForeachViewCopyToOutputTensorList(contiguousOut, out, uniqueExecutor.get()), ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnForeachMaximumScalarListGetWorkspaceSize(const aclTensorList* x, const aclScalarList* scalars,
                                                          aclTensorList* out, uint64_t* workspaceSize,
                                                          aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnForeachMaximumScalarList, DFX_IN(x, scalars), DFX_OUT(out));
    return ExecForeachMaximumScalarListGetWorkspaceSize(x, scalars, out, workspaceSize, executor);
}

aclnnStatus aclnnForeachMaximumScalarList(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                          aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnForeachMaximumScalarList);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
