/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_prelu.h"
#include "prelu.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_errno.h"
#include "op_api/aclnn_util.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<DataType>& GetDtypeSupportList() {
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2201 || Ops::NN::AclnnUtil::IsRegbase(curArch)) {
      return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
    } else {
      return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
    }
}

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_AICORE_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT,
                                                                                         DataType::DT_FLOAT16,
                                                                                         DataType::DT_BF16};

static bool IsAiCoreSupport(const aclTensor *self) {
  auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
  if (curArch == NpuArch::DAV_2201 || Ops::NN::AclnnUtil::IsRegbase(curArch)) {
    return CheckType(self->GetDataType(), ASCEND910B_AICORE_DTYPE_SUPPORT_LIST);
  }
    return CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST);
}

static bool CheckNotNull(const aclTensor *self, const aclTensor *weight, const aclTensor *out) {
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(weight, return false);
    OP_CHECK_NULL(out, return false);

    return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *weight, const aclTensor *out) {
    // 检查self的数据类型是否在prelu算子的支持列表内
    const auto& supportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
    // 检查weight的数据类型是否在prelu算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(weight, supportList, return false);
    // 检查out的数据类型是否在prelu算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(out, supportList, return false);

    return true;
}

static bool CheckPromoteType(const aclTensor *self, const aclTensor *weight, const aclTensor *out) {
    // 检查self和weight能否做数据类型推导
    op::DataType promoteType = op::PromoteType(self->GetDataType(), weight->GetDataType());
    if (promoteType == DataType::DT_UNDEFINED) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype %s and weight dtype %s can not promote dtype.",
              op::ToString(self->GetDataType()).GetString(), op::ToString(weight->GetDataType()).GetString());
      return false;
    }

    // 检查推导后的数据类型能否转换为输出的数据类型
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);

    return true;
}

constexpr size_t MAX_DIM_LEN = 8;

static bool CheckShape(const aclTensor *self, const aclTensor *weight, const aclTensor *out) {
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(weight, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(out, MAX_DIM_LEN, return false);

    // self和out的shape必须相等
    OP_CHECK_SHAPE_NOT_EQUAL(self, out, return false);

    // self是一维，则input_channels为1
    auto input_channels = self->GetViewShape().GetDimNum() <= 1 ? 1 : self->GetViewShape().GetDim(1);
    int64_t weightSize = weight->GetViewShape().GetShapeSize();
    if (weightSize != input_channels && weightSize != 1) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID,
              "expected weight tensor size equal input channels, but found %ld does not equal %ld", weightSize,
              input_channels);
      return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *weight, const aclTensor *out) {
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, weight, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, weight, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查self和weight能否做数据类型推导以及推导的数据类型能否转换为输出数据类型
    CHECK_RET(CheckPromoteType(self, weight, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查双输入shape
    CHECK_RET(CheckShape(self, weight, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnPreluGetWorkspaceSize(const aclTensor *self, const aclTensor *weight, aclTensor *out,
                                       uint64_t *workspaceSize, aclOpExecutor **executor) {
    L2_DFX_PHASE_1(aclnnPrelu, DFX_IN(self, weight), DFX_OUT(out));

    // 固定写法，参数检查
    CHECK_RET(workspaceSize != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto ret = CheckParams(self, weight, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (self->IsEmpty() || weight->IsEmpty()) {
      *workspaceSize = 0;
      uniqueExecutor.ReleaseTo(executor);
      return ACLNN_SUCCESS;
    }

    // Prelu算子需要对self和weight两个输入做隐式数据类型转换，根据具体算子语义按需调用
    auto promoteType = op::PromoteType(self->GetDataType(), weight->GetDataType());

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入self的数据类型转换成隐式数据类型，根据具体算子语义按需调用
    auto selfCasted = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(selfCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入weight转换成连续的tensor
    auto weightContiguous = l0op::Contiguous(weight, uniqueExecutor.get());
    CHECK_RET(weightContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将输入weight的数据类型转换成隐式数据类型，根据具体算子语义按需调用
    auto weightCasted = l0op::Cast(weightContiguous, promoteType, uniqueExecutor.get());
    CHECK_RET(weightCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (selfCasted->GetViewShape().GetDimNum() > 1) {
      int64_t weightSize = weightCasted->GetViewShape().GetShapeSize();
      auto weight_new_shape = selfCasted->GetViewShape();
      for (size_t i = 0; i < selfCasted->GetViewShape().GetDimNum(); i++) {
        if (i == 1) {
          weight_new_shape.SetDim(1, weightSize);
        } else {
          weight_new_shape.SetDim(i, 1);
        }
      }
      weightCasted = l0op::Reshape(weightCasted, weight_new_shape, uniqueExecutor.get());
      CHECK_RET(weightCasted != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (!IsAiCoreSupport(selfCasted) || !IsAiCoreSupport(weightCasted)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "after promote, dtype is not valid");
      return ACLNN_ERR_PARAM_INVALID;
    }
    auto preluOpOut = l0op::PRelu(selfCasted, weightCasted, uniqueExecutor.get());
    CHECK_RET(preluOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(preluOpOut, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnPrelu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnPrelu);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
