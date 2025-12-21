/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "op_api/op_api_def.h"
#include "opdev/tensor_view_utils.h"
#include "level0/broadcast_to.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/mul.h"
#include "renorm.h"
#include "aclnn_renorm.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

namespace {
static const int32_t SELF_SUPPORT_MIN_DIMS = 2;
}

// 根据API定义，需要列出self和out所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static inline bool CheckNotNull(
    const aclTensor* self, const aclScalar* p, const aclScalar* maxNorm, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(p, return false);
    OP_CHECK_NULL(maxNorm, return false);
    OP_CHECK_NULL(out, return false);

    return true;
}

static inline bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    // 检查self的数据类型是否在renorm算子的输入支持列表内
    const std::initializer_list<op::DataType> currentDtypeSupportList =
        GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B ? ASCEND910B_DTYPE_SUPPORT_LIST :
                                                                             ASCEND910_DTYPE_SUPPORT_LIST;
    OP_CHECK_DTYPE_NOT_SUPPORT(self, currentDtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, currentDtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);

    return true;
}

static inline bool CheckRenormFormat(const aclTensor* self, const aclTensor* out)
{
    // 需要根据算子实际情况添加校验
    if (self->GetViewFormat() != out->GetViewFormat()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of input and output should be equal, self [%s], out [%s].",
            op::ToString(self->GetViewFormat()).GetString(), op::ToString(out->GetViewFormat()).GetString());
        return false;
    }

    return true;
}

static inline bool CheckRenormShape(const aclTensor* self, const aclTensor* out)
{
    OP_CHECK_MIN_DIM(self, SELF_SUPPORT_MIN_DIMS, return false);
    OP_CHECK_MIN_DIM(out, SELF_SUPPORT_MIN_DIMS, return false);
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(out, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);

    return true;
}

static inline bool IsSupportPValue(const aclScalar* p)
{
    float x = p->ToFloat();
    if (x < 0) {
        return false;
    }
    return true;
};

static bool CheckValue(const aclTensor* self, const aclScalar* p, int64_t dim, const aclScalar* maxNorm)
{
    // 检查p的取值是否在范围内
    if (!IsSupportPValue(p)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "p only support non-negative values, but now p is [%f].", p->ToFloat());
        return false;
    }

    // 检查dim是否在[-N, N - 1]范围内
    int64_t tensorDimSize = static_cast<int64_t>(self->GetViewShape().GetDimNum());
    auto dimMax = tensorDimSize - 1;
    auto dimMin = -1 * tensorDimSize;
    if (dim < dimMin || dim > dimMax) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim range only support [-dimNum, dimNum - 1], but now dim is [%ld].", dim);
        return false;
    }

    // 检查maxNorm是否在[0, +∞]范围内
    if (maxNorm->ToFloat() < 0.0f) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "maxNorm should be greater than 0, but now maxNorm is [%f].", maxNorm->ToFloat());
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(
    const aclTensor* self, const aclScalar* p, int64_t dim, const aclScalar* maxNorm, aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, p, maxNorm, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查数据格式是否支持
    CHECK_RET(CheckRenormFormat(self, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入shape是否超过8维
    CHECK_RET(CheckRenormShape(self, out), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查输入p, dim, maxNorm的值是否在范围内
    CHECK_RET(CheckValue(self, p, dim, maxNorm), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRenormGetWorkspaceSize(
    const aclTensor* self, const aclScalar* p, int64_t dim, const aclScalar* maxNorm, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnRenorm, DFX_IN(self, p, dim, maxNorm), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, p, dim, maxNorm, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // renorm算子的空tensor在kernel中支持
    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    size_t dimNum = self->GetViewShape().GetDimNum();
    if (dim < 0) {
        dim += dimNum;
    }

    // 进行Renorm计算
    auto renormOpOut = l0op::Renorm(selfContiguous, p->ToFloat(), dim, maxNorm->ToFloat(), uniqueExecutor.get());
    CHECK_RET(renormOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 进行broadcast_to扩维，将renorm计算的结果的shape恢复成和self一致
    int64_t sizes[dimNum];
    for (size_t i = 0U; i < dimNum; ++i) {
        sizes[i] = self->GetViewShape().GetDim(i);
    }
    aclIntArray* shapes = uniqueExecutor.get()->AllocIntArray(sizes, dimNum);

    auto broadcastOut = l0op::BroadcastTo(renormOpOut, shapes, uniqueExecutor.get());
    CHECK_RET(broadcastOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 用self和broadcastto的结果做点乘，得到输出
    auto mulOut = l0op::Mul(selfContiguous, broadcastOut, uniqueExecutor.get());
    CHECK_RET(mulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(mulOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnRenorm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRenorm);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnInplaceRenormGetWorkspaceSize(
    aclTensor* selfRef, const aclScalar* p, int64_t dim, const aclScalar* maxNorm, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    aclTensor* out = selfRef;
    return aclnnRenormGetWorkspaceSize(
        selfRef, p, dim, maxNorm, out, workspaceSize, executor);
}

aclnnStatus aclnnInplaceRenorm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceRenorm);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
