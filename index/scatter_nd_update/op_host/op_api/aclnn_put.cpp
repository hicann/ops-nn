/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file aclnn_put.cpp
 * \brief
 */

#include "aclnn_kernels/reshape.h"
#include "level0/scatter_nd_add.h"
#include "scatter_nd_update.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_put.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

/* Put 算子的完整计算流程如下:
 * selfRef                           index                 source               accumulate
 *   |                                  |                     |                        |
 *   \                                  |                     |                       /
 * Contiguous(workspace_0)    Contiguous(workspace_2)   Contiguous(workspace_4)      /
 *      \                              |                      |                     /
 *     Reshape(workspace_1)     Reshape(workspace_3)     Reshape(workspace_5)      /
 *               \                      |                     /                  /
 *                      ScatterNdAdd(ScatterNdUpdate)(workspace_6)
 *                                      |
 *                              Reshape(workspace_7)
 *                                      |
 *                            ViewCopy(workspace_8)
 *                                      |
 *                                   selfRef
 */

constexpr size_t MAX_DIM_LEN = 8;
// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,     op::DataType::DT_INT32,      op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16,     op::DataType::DT_INT8,       op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,     op::DataType::DT_INT32,      op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16,     op::DataType::DT_INT8,       op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
    op::DataType::DT_COMPLEX64, op::DataType::DT_COMPLEX128, op::DataType::DT_BOOL, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> INDEX_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT64,
                                                                             op::DataType::DT_INT32};

static bool CheckNotNull(const aclTensor* self, const aclTensor* index, const aclTensor* source) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(index, return false);
  OP_CHECK_NULL(source, return false);
  return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList(bool accumulate) {
  if ((GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) && accumulate == false) {
    return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
  }
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* index, const aclTensor* source, bool accumulate) {
  // 检查self的数据类型是否在算子的支持列表内
  auto supportList = GetDtypeSupportList(accumulate);
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  // 检查index的数据类型是否在算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(index, INDEX_DTYPE_SUPPORT_LIST, return false);
  // self和source的数据类型要一致
  if (self->GetDataType() != source->GetDataType()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "source dtype %s should be in same with self dtype %s.",
            op::ToString(source->GetDataType()).GetString(), op::ToString(self->GetDataType()).GetString());
    return false;
  }

  return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* index, const aclTensor* source) {
  OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
  OP_CHECK_MAX_DIM(index, MAX_DIM_LEN, return false);

  if (index->GetViewShape().GetShapeSize() != source->GetViewShape().GetShapeSize()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Expect source and index to have the same number of elements,but the source has %ld,the index has %ld.",
            source->GetViewShape().GetShapeSize(), index->GetViewShape().GetShapeSize());
    return false;
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* index, const aclTensor* source, bool accumulate) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, index, source), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
  CHECK_RET(CheckDtypeValid(self, index, source, accumulate), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查输入形状是否满足
  CHECK_RET(CheckShape(self, index, source), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplacePutGetWorkspaceSize(aclTensor* selfRef, const aclTensor* index, const aclTensor* source,
                                            bool accumulate, uint64_t* workspaceSize, aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnInplacePut, DFX_IN(selfRef, index, source, accumulate), DFX_OUT(selfRef));

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(selfRef, index, source, accumulate);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  if (index->IsEmpty()) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  if (selfRef->IsEmpty() && (!index->IsEmpty())) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Can not put elements into an empty tensor");
    return ACLNN_ERR_PARAM_INVALID;
  }

  op::Shape rowVector;
  rowVector.AppendDim(-1);

  // 固定写法，将输入selfRef转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(selfRef, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto selfFlatten = l0op::Reshape(selfContiguous, rowVector, uniqueExecutor.get());
  CHECK_RET(selfFlatten != nullptr, ACLNN_ERR_INNER_NULLPTR);

  op::Shape columnVector;
  columnVector.AppendDim(-1);
  columnVector.AppendDim(1);

  // 固定写法，将输入index转换成连续的tensor
  auto indexContiguous = l0op::Contiguous(index, uniqueExecutor.get());
  CHECK_RET(indexContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto indexFlatten = l0op::Reshape(indexContiguous, columnVector, uniqueExecutor.get());
  CHECK_RET(indexFlatten != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将输入source转换成连续的tensor
  auto sourceContiguous = l0op::Contiguous(source, uniqueExecutor.get());
  CHECK_RET(sourceContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto sourceFlatten = l0op::Reshape(sourceContiguous, rowVector, uniqueExecutor.get());
  CHECK_RET(sourceFlatten != nullptr, ACLNN_ERR_INNER_NULLPTR);

  const aclTensor* selfRefReShapeOut;
  if (accumulate) {
    // 调用ScatterNdAdd算子
    auto kernelOut = l0op::ScatterNdAdd(selfFlatten, indexFlatten, sourceFlatten, false, uniqueExecutor.get());
    CHECK_RET(kernelOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    selfRefReShapeOut = l0op::Reshape(kernelOut, selfRef->GetViewShape(), uniqueExecutor.get());
  } else {
    // 调用ScatterNdUpdate算子
    auto kernelOut = l0op::ScatterNdUpdate(selfFlatten, indexFlatten, sourceFlatten, false, uniqueExecutor.get());
    CHECK_RET(kernelOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    selfRefReShapeOut = l0op::Reshape(kernelOut, selfRef->GetViewShape(), uniqueExecutor.get());
  }

  // 固定写法，将计算结果拷贝到输出selfRef上
  if (!IsContiguous(selfRef)) {
    auto viewCopyResult = l0op::ViewCopy(selfRefReShapeOut, selfRef, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplacePut(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnInplacePut);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
