/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_foreach_div_list.h"
#include "foreach_div_list.h"
#include "../../foreach_utils/op_api/foreach_contiguous_helper.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/aclnn_util.h"
#include "op_api/op_api_def.h"
#include "opdev/op_dfx.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> ASCEND910BC_TENSOR_DTYPE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

static const std::initializer_list<DataType> EMPTY_LIST = {};

static inline bool CheckNotNull(const aclTensorList* x1, const aclTensorList* x2, const aclTensorList* out)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList()
{
    auto curNpuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curNpuArch == NpuArch::DAV_2201 || Ops::NN::AclnnUtil::IsRegbase(curNpuArch)) {
        return ASCEND910BC_TENSOR_DTYPE_DTYPE_SUPPORT_LIST;
    } else {
        return EMPTY_LIST;
    }
}

static inline bool CheckFormat(const aclTensorList* x1, const aclTensorList* x2, const aclTensorList* out)
{
    for (uint64_t i = 0; i < x1->Size(); i++) {
        // self格式不能是私有格式
        if (IsPrivateFormat((*x2)[i]->GetStorageFormat()) || IsPrivateFormat((*x1)[i]->GetStorageFormat()) ||
            IsPrivateFormat((*out)[i]->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Format only support ND、NCHW、NHWC、HWCN、NDHWC、NCDHW.");
            return false;
        }
    }
    return true;
}

static inline bool CheckDtypeValid(const aclTensorList* x1, const aclTensorList* x2, const aclTensorList* out)
{
    const auto& dtypeSupportList = GetDtypeSupportList();
    if (dtypeSupportList.size() == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "support for %s is not implemented",
                op::ToString(GetCurrentPlatformInfo().GetSocVersion()).GetString());
        return false;
    }

    if (x1->Size() == 0) {
        return true;
    }

    // check self input dtype, and check the releation of input and out
    auto selfDtype = (*x1)[0]->GetDataType();
    OP_CHECK_DTYPE_NOT_SUPPORT((*x1)[0], dtypeSupportList, return false);

    for (uint64_t i = 0; i < x2->Size(); i++) {
        OP_CHECK_DTYPE_NOT_MATCH((*x2)[i], selfDtype, return false);
    }

    for (uint64_t i = 0; i < x1->Size(); i++) {
        OP_CHECK_DTYPE_NOT_MATCH((*x1)[i], selfDtype, return false);
    }

    for (uint64_t i = 0; i < out->Size(); i++) {
        OP_CHECK_DTYPE_NOT_MATCH((*out)[i], selfDtype, return false);
    }
    return true;
}

static inline bool CheckShape(const aclTensorList* x1, const aclTensorList* x2, const aclTensorList* out)
{
    // tensorlist size检查
    if (x1->Size() != x2->Size() || x1->Size() != out->Size()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensor lists must have the same number of tensors, got %lu and %lu",
                x1->Size(), x2->Size());
        return false;
    }

    // tensorlist中tensor shape一致性检查
    for (uint64_t i = 0; i < x1->Size(); i++) {
        OP_CHECK_SHAPE_NOT_EQUAL((*x1)[i], (*x2)[i], return false);
    }

    // tensor 维度检查
    for (uint64_t i = 0; i < x1->Size(); i++) {
        OP_CHECK_MAX_DIM((*x1)[i], MAX_SUPPORT_DIMS_NUMS, return false);
    }

    // self和out的shape必须一致
    for (uint64_t i = 0; i < x1->Size(); i++) {
        OP_CHECK_SHAPE_NOT_EQUAL((*x1)[i], (*out)[i], return false);
    }
    return true;
}

static inline aclnnStatus CheckParams(const aclTensorList* x1, const aclTensorList* x2, const aclTensorList* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x1, x2, out), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(x1, x2, out), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查shape是否满足约束
    CHECK_RET(CheckShape(x1, x2, out), ACLNN_ERR_PARAM_INVALID);
    // 4. 检查Format是否满足约束
    CHECK_RET(CheckFormat(x1, x2, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecForeachDivListGetWorkspaceSize(const aclTensorList* x1, const aclTensorList* x2,
                                                      const aclTensorList* out, uint64_t* workspaceSize,
                                                      aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(x1, x2, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensorlist处理
    if (x2->Size() == 0 || x1->Size() == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 输出如果非连续，需要转连续作为kernel输出buffer；连续/空则直接使用
    auto contiguousOut = ForeachMakeContiguousTensorList(out, uniqueExecutor.get());
    CHECK_RET(contiguousOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 输入如果非连续，需要转连续（空tensor直接用，保持索引一致）
    auto contiguousTensorsX1 = ForeachMakeContiguousTensorList(x1, uniqueExecutor.get());
    CHECK_RET(contiguousTensorsX1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto contiguousTensorsX2 = ForeachMakeContiguousTensorList(x2, uniqueExecutor.get());
    CHECK_RET(contiguousTensorsX2 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0算子ForeachDivList进行计算，输出到连续buffer
    auto result = l0op::ForeachDivList(contiguousTensorsX1, contiguousTensorsX2, contiguousOut, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将连续计算结果拷贝到输出out上，out可能是非连续的tensor（空/连续跳过）
    CHECK_RET(ForeachViewCopyToOutputTensorList(contiguousOut, out, uniqueExecutor.get()), ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnForeachDivListGetWorkspaceSize(const aclTensorList* x1, const aclTensorList* x2, aclTensorList* out,
                                                uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnForeachDivList, DFX_IN(x1, x2), DFX_OUT(out));
    return ExecForeachDivListGetWorkspaceSize(x1, x2, out, workspaceSize, executor);
}

aclnnStatus aclnnForeachDivList(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnForeachDivList);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
