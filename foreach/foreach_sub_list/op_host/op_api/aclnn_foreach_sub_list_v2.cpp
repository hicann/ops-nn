/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_foreach_sub_list_v2.cpp
 * \brief
 */

#include "aclnn_foreach_sub_list_v2.h"
#include "foreach_sub_list_v2.h"
#include "../../foreach_utils/op_host/foreach_contiguous_helper.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/op_api_def_nn.h"
#include "op_api/aclnn_util.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_dfx.h"
#include "opdev/platform.h"
#include "opdev/make_op_executor.h"
#include "../../foreach_utils/op_host/foreach_null_check_helper.h"
#include <string>
#include <vector>
#include "base/err_msg.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<ge::DataType> ASCEND910BC_TENSOR_DTYPE_DTYPE_SUPPORT_LIST = {
    ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT32, ge::DT_INT16, ge::DT_INT8, ge::DT_UINT8};

static const std::initializer_list<ge::DataType> FOREACH_SCALAR_FLOAT_SUPPORT_LIST = {ge::DT_FLOAT, ge::DT_DOUBLE};

static const std::initializer_list<ge::DataType> FOREACH_SCALAR_FLOAT16_SUPPORT_LIST = {ge::DT_FLOAT16, ge::DT_DOUBLE};

static const std::initializer_list<ge::DataType> FOREACH_SCALAR_INT_SUPPORT_LIST = {ge::DT_INT32, ge::DT_INT64};

static const std::initializer_list<ge::DataType> FOREACH_SCALAR_SUPPORT_LIST = {
    ge::DT_FLOAT, ge::DT_DOUBLE, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT64};

static const std::initializer_list<ge::DataType> EMPTY_LIST = {};

static inline ge::DataType GetAlphaTensorDtype(ge::DataType selfDtype)
{
    if (selfDtype == ge::DT_BF16) {
        return ge::DT_FLOAT;
    }
    if (selfDtype == ge::DT_INT16 || selfDtype == ge::DT_INT8 || selfDtype == ge::DT_UINT8) {
        return ge::DT_INT32;
    }
    return selfDtype;
}

static inline bool CheckNotNull(const aclTensorList* subListV2Self, const aclTensorList* x2, const aclScalar* scalar,
                                const aclTensorList* out)
{
    OP_CHECK_NULL(subListV2Self, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(scalar, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckTensorListSize(const aclTensorList* subListV2Self, const aclTensorList* x2,
                                       const aclTensorList* out)
{
    if (subListV2Self->Size() != x2->Size() || subListV2Self->Size() != out->Size()) {
        bool x2Mismatch = subListV2Self->Size() != x2->Size();
        auto parameter = x2Mismatch ? "x2" : "out";
        auto actualSize = x2Mismatch ? x2->Size() : out->Size();
        std::string reason = "Tensor lists must have the same number of tensors, got x1=" +
                             std::to_string(subListV2Self->Size()) + ", x2=" + std::to_string(x2->Size()) +
                             ", out=" + std::to_string(out->Size());
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "%s", reason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ0038",
                                  (std::vector<const char*>{"param_name", "op_name", "incorrect_size", "reason"}),
                                  (std::vector<const char*>{parameter, "aclnnForeachSubListV2",
                                                            (std::to_string(actualSize)).c_str(), reason.c_str()}));
        return false;
    }
    return true;
}

static const std::initializer_list<ge::DataType>& GetDtypeSupportList()
{
    auto subListV2Arch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (subListV2Arch == NpuArch::DAV_2201 || Ops::NN::AclnnUtil::IsRegbase(subListV2Arch)) {
        return ASCEND910BC_TENSOR_DTYPE_DTYPE_SUPPORT_LIST;
    } else {
        const std::string reason = (("support for " +
                                     std::string(op::ToString(GetCurrentPlatformInfo().GetSocVersion()).GetString()) +
                                     " is not implemented"));
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_RUNTIME_ERROR, "%s", reason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ3002", (std::vector<const char*>{"optype", "opskernel", "reason"}),
                                  (std::vector<const char*>{"ForeachSubList", "AiCore", reason.c_str()}));
        return EMPTY_LIST;
    }
}

static inline bool CheckDtypeValid(const aclTensorList* subListV2Self, const aclTensorList* x2, const aclScalar* scalar,
                                   const aclTensorList* out)
{
    const auto& dtypeSupportList = GetDtypeSupportList();
    if (dtypeSupportList.size() == 0) {
        OP_LOGE_WITHOUT_REPORT(ACLNN_ERR_PARAM_INVALID, "support for %s is not implemented",
                               op::ToString(GetCurrentPlatformInfo().GetSocVersion()).GetString());
        return false;
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(scalar, FOREACH_SCALAR_SUPPORT_LIST, return false);
    if (subListV2Self->Size() == 0) {
        return true;
    }

    // checkself input dtype, and check the releation of input and out
    auto selfDtyte = (*subListV2Self)[0]->GetDataType();
    OP_CHECK_DTYPE_NOT_SUPPORT((*subListV2Self)[0], dtypeSupportList, return false);

    // check the releation of self and scalar
    if (selfDtyte == ge::DT_BF16 || selfDtyte == ge::DT_FLOAT) {
        OP_CHECK_DTYPE_NOT_SUPPORT(scalar, FOREACH_SCALAR_FLOAT_SUPPORT_LIST, return false);
    } else if (selfDtyte == ge::DT_FLOAT16) {
        OP_CHECK_DTYPE_NOT_SUPPORT(scalar, FOREACH_SCALAR_FLOAT16_SUPPORT_LIST, return false);
    } else if (selfDtyte == ge::DT_INT32 || selfDtyte == ge::DT_INT16 || selfDtyte == ge::DT_INT8 ||
               selfDtyte == ge::DT_UINT8) {
        OP_CHECK_DTYPE_NOT_SUPPORT(scalar, FOREACH_SCALAR_INT_SUPPORT_LIST, return false);
    }

    for (uint64_t i = 0; i < subListV2Self->Size(); i++) {
        OP_CHECK_DTYPE_NOT_MATCH((*subListV2Self)[i], selfDtyte, return false);
    }

    for (uint64_t i = 0; i < x2->Size(); i++) {
        OP_CHECK_DTYPE_NOT_MATCH((*x2)[i], selfDtyte, return false);
    }

    for (uint64_t i = 0; i < out->Size(); i++) {
        OP_CHECK_DTYPE_NOT_MATCH((*out)[i], selfDtyte, return false);
    }
    return true;
}

static inline bool CheckShape(const aclTensorList* subListV2Self, const aclTensorList* x2, const aclTensorList* out)
{
    // tensorlist中tensor shape一致性检查
    for (uint64_t i = 0; i < subListV2Self->Size(); i++) {
        OP_CHECK_SHAPE_NOT_EQUAL((*subListV2Self)[i], (*x2)[i], return false);
    }

    // tensor 维度检查
    for (uint64_t i = 0; i < subListV2Self->Size(); i++) {
        OP_CHECK_MAX_DIM((*subListV2Self)[i], MAX_SUPPORT_DIMS_NUMS, return false);
    }

    // self和out的shape必须一致
    for (uint64_t i = 0; i < subListV2Self->Size(); i++) {
        OP_CHECK_SHAPE_NOT_EQUAL((*subListV2Self)[i], (*out)[i], return false);
    }
    return true;
}

static inline aclnnStatus CheckParams(const aclTensorList* subListV2Self, const aclTensorList* x2,
                                      const aclScalar* scalar, const aclTensorList* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(subListV2Self, x2, scalar, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckTensorListSize(subListV2Self, x2, out), ACLNN_ERR_PARAM_INVALID);

    // Check every entry in tensor lists is not null, to avoid null pointer
    // dereference in CheckDtypeValid/CheckShape.
    CHECK_RET(CheckTensorListNotNull(subListV2Self, "self"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorListNotNull(x2, "x2"), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckTensorListNotNull(out, "out"), ACLNN_ERR_PARAM_INVALID);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(subListV2Self, x2, scalar, out), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查shape是否满足约束
    CHECK_RET(CheckShape(subListV2Self, x2, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus ExecForeachSubListV2GetWorkspaceSize(const aclTensorList* x1, const aclTensorList* x2,
                                                        const aclScalar* scalar, const aclTensorList* out,
                                                        uint64_t* workspaceSize, aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto subListV2Executor = CREATE_EXECUTOR();
    CHECK_RET(subListV2Executor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(x1, x2, scalar, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensorlist处理
    if (x1->Size() == 0 || x2->Size() == 0) {
        *workspaceSize = 0;
        subListV2Executor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 输入如果非连续，需要转连续（空tensor直接用，保持索引一致）
    auto contiguousTensorsX1 = ForeachMakeContiguousTensorList(x1, subListV2Executor.get());
    CHECK_RET(contiguousTensorsX1 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto contiguousTensorsX2 = ForeachMakeContiguousTensorList(x2, subListV2Executor.get());
    CHECK_RET(contiguousTensorsX2 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 输出如果非连续，需要转连续作为kernel输出buffer；连续/空则直接使用
    auto contiguousOut = ForeachMakeContiguousTensorList(out, subListV2Executor.get());
    CHECK_RET(contiguousOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // sclar to tensor
    const aclTensor* otherTensor = subListV2Executor.get()->ConvertToTensor(
        scalar, GetAlphaTensorDtype((*x1)[0]->GetDataType()));
    CHECK_RET(otherTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用l0算子ForeachSubListV2进行计算，输出到连续buffer
    auto result = l0op::ForeachSubListV2(contiguousTensorsX1, contiguousTensorsX2, otherTensor, contiguousOut,
                                         subListV2Executor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将连续计算结果拷贝到输出out上，out可能是非连续的tensor（空/连续跳过）
    CHECK_RET(ForeachViewCopyToOutputTensorList(contiguousOut, out, subListV2Executor.get()), ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = subListV2Executor->GetWorkspaceSize();
    subListV2Executor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnForeachSubListV2GetWorkspaceSize(const aclTensorList* x1, const aclTensorList* x2,
                                                  const aclScalar* alpha, aclTensorList* out, uint64_t* workspaceSize,
                                                  aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnForeachSubListV2, DFX_IN(x1, x2, alpha), DFX_OUT(out));
    return ExecForeachSubListV2GetWorkspaceSize(x1, x2, alpha, out, workspaceSize, executor);
}

aclnnStatus aclnnForeachSubListV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                  const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnForeachSubListV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
