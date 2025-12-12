/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include "aclnn_softmax.h"
#include "softmax_v2_op.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static bool CheckNotNull(const aclTensor *self, aclTensor *out) {
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);

    return true;
}

// pytorch：CPU支持Bfloat、float、double，CUDA支持Bfloat、half、float、double
// ascend：AIC支持:DT_BF16, DT_FLOAT16, DT_FLOAT, AICPU支持 DT_DOUBLE
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_DOUBLE, op::DataType::DT_BF16};

static const int64_t AXIS_LIMIT = 8;  // 底层算子不支持超过8维

static inline const std::initializer_list<DataType>& GetDtypeSupportList(void) {
    if(GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
       GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
      return ASCEND910B_DTYPE_SUPPORT_LIST;
    }else {
      return ASCEND910_DTYPE_SUPPORT_LIST;
    }
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *out) {
    const std::initializer_list<DataType>& dtypeSupportList = GetDtypeSupportList();

    OP_CHECK_DTYPE_NOT_SUPPORT(self, dtypeSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, dtypeSupportList, return false);
    return true;
}

static bool CheckDim(const aclTensor *self, int64_t dim) {
    // 检查指定dim是否在self的维度范围内
    auto selfViewShape = self->GetViewShape();
    auto selfDimNum = static_cast<int64_t>(selfViewShape.GetDimNum());
    if (selfDimNum == 0) {
        selfDimNum++;
    }
    if (dim >= selfDimNum || dim < -selfDimNum) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "provided dim %ld not in the range of input size %ld.", dim, selfDimNum);
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor *self, aclTensor *out) {
    OP_CHECK_MAX_DIM(self, AXIS_LIMIT, return false);

    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);

    return true;
}

static aclnnStatus CheckParams(const aclTensor *self, int64_t dim, aclTensor* out) {
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 空tensor直接返回
    if (self->IsEmpty()) {
      return ACLNN_SUCCESS;
    }
    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查指定dim是否在self的维度范围内
    CHECK_RET(CheckDim(self, dim), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查输入shape与输出shape是否一致
    CHECK_RET(CheckShape(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSoftmaxGetWorkspaceSize(const aclTensor* self, int64_t dim, aclTensor* out,
                                         uint64_t* workspaceSize, aclOpExecutor** executor) {
    L2_DFX_PHASE_1(aclnnSoftmax, DFX_IN(self, dim), DFX_OUT(out));

    // 固定写法，参数检查
    auto ret = CheckParams(self, dim, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // softmax支持空tensor
    if (self->IsEmpty()) {
        OP_LOGD("empty input tensor");
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto self_contiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(self_contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 调用SoftmaxGrad算子kernel
    auto op_out = l0op::SoftmaxV2(self_contiguous, dim, uniqueExecutor.get());
    CHECK_RET(op_out != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto cast_out = l0op::Cast(op_out, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(cast_out != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto view_copy_result = l0op::ViewCopy(cast_out, out, uniqueExecutor.get());
    CHECK_RET(view_copy_result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSoftmax(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnSoftmax);
    OP_LOGD("Entering aclnnSoftmax");
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif