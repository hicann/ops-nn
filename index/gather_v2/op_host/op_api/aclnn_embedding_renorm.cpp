/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_embedding_renorm.h"
#include "level0/arange.h"
#include "aclnn_kernels/reshape.h"
#include "index/scatter_update/op_host/op_api/scatter_update.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "gather_v2.h"
#include "level0/mul.h"
#include "level0/renorm.h"
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
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"

using namespace op;
static const int64_t EMBEDDING_RENORM_RENORM_DEFAULT_DIM = 0;
static const int64_t GATHERV2_DEFAULT_AXIS = 0;
static constexpr size_t MAX_DIM_LEN = 8;

/* embedding_renorm aten的完整计算流程:
 *                                   indices[4]([0,3,1,2])
 *                                            |
 *     self [4,6]                    Contiguous(workspace_1)
 *         |                                  |
 *    Contiguous(workspace_0)           Reshape(workspace_2) ------
 *         |       \                 /                             |
 *          \      GatherV2(workspace_3)                           |
 *            \              |   \                                 |
 *             \             |  Renorm(workspace_4)     indices-2  |
 *              \            |           \                /      /
 *               \           |         GatherV2(workspace_5)  /
 *                \          |          /                 /
 *                 \        Mul(workspace_6)          /
 *                  \        |                    /
 *                    ScatterUpdate(workspace_7)
 *                           |
 *                      ViewCopy(workspace_8)
 *                           |
 *                          self
 */

// 检查是否有空tensor
static bool HasEmptyTensor(const aclTensor *self, const aclTensor *indices) {
  if (self->IsEmpty() || indices->IsEmpty()) {
    return true;
  }
  return false;
}

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_INDICES = {
    op::DataType::DT_INT32, op::DataType::DT_INT64};

// 算子支持的维度
static const size_t DIM_SUPPORT_ONLY = 2;

// CheckDim，检查self维度是否为2维，不是则报错
static bool CheckSelfDim(const aclTensor *self) {
  if (self->GetViewShape().GetDimNum() != DIM_SUPPORT_ONLY) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID,
            "Embedding_Renorm: Except 2-dimension Tensor but got [%s]-dimension Tensor.",
            op::ToString(self->GetViewShape()).GetString());
    return false;
  }
  return true;
}

// checkIndicesDim, 检查indices的维度是否小于等于8维，不是则报错
static bool CheckIndicesDimension(const aclTensor *tensor) {
  // indices的维度不能超过8
  OP_CHECK_MAX_DIM(tensor, MAX_DIM_LEN, return false);
  return true;
}

static bool CheckNotNull(const aclTensor *self, const aclTensor *indices) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(indices, return false);
  return true;
}

static const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
      GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93) {
    return ASCEND910B_DTYPE_DTYPE_SUPPORT_LIST;
  } else {
    return ASCEND910_DTYPE_DTYPE_SUPPORT_LIST;
  }
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *indices) {
  auto supportList = GetDtypeSupportList();
  // 检查self的数据类型是否在embedding_renorm支持的算子库内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);

  // 检查indices的数据类型是否在embedding_renorm支持的算子库内
  OP_CHECK_DTYPE_NOT_SUPPORT(indices, DTYPE_SUPPORT_LIST_INDICES, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, const aclTensor *indices) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, indices), ACLNN_ERR_INNER_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
  CHECK_RET(CheckDtypeValid(self, indices), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查输入的数据类型是否为2维
  CHECK_RET(CheckSelfDim(self), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查indices的维度是否超过8
  CHECK_RET(CheckIndicesDimension(indices), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnEmbeddingRenormGetWorkspaceSize(aclTensor *selfRef,
                                                 const aclTensor *indices,
                                                 double maxNorm,
                                                 double normType,
                                                 uint64_t *workspaceSize,
                                                 aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);

  L2_DFX_PHASE_1(aclnnEmbeddingRenorm, DFX_IN(selfRef, indices, maxNorm, normType), DFX_OUT(selfRef));

  // 固定写法 参数检查
  auto ret = CheckParams(selfRef, indices);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // selfRef 和 indices支持空Tensor，则返回ACLNN_SUCCESS
  if (HasEmptyTensor(selfRef, indices)) {
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 对indices的shape进行校验，若不为1维tensor则进行reshape
  // 先对indices进行Contiguous
  auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
  CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_PARAM_NULLPTR);
  size_t indicesShapeDim = indicesContiguous->GetViewShape().GetDimNum();
  auto indicesReshape = indicesContiguous;
  // 对不为1维tensor则进行reshape
  if (indicesShapeDim > 1 || indicesShapeDim == 0) {
    int64_t indicesShapeValue[1] = {1};
    for (size_t i = 0; i < indicesShapeDim; i++) {
        indicesShapeValue[0] *= indicesContiguous->GetViewShape().GetDim(i);
    }
    aclIntArray* indicesShape = (uniqueExecutor.get())->AllocIntArray(indicesShapeValue, 1);
    indicesReshape = l0op::Reshape(indicesContiguous, indicesShape, uniqueExecutor.get());
    CHECK_RET(indicesReshape != nullptr, ACLNN_ERR_PARAM_NULLPTR);
  }

  // 对selfRef和indices进行contiguous等操作
  auto selfRefContiguous = l0op::Contiguous(selfRef, uniqueExecutor.get());
  CHECK_RET(selfRefContiguous != nullptr, ACLNN_ERR_PARAM_NULLPTR);

  // 调用l0算子GatherV2进行计算，mid_input-> firstGatherV2
  auto firstGatherV2 = l0op::GatherV2(selfRefContiguous, GATHERV2_DEFAULT_AXIS, indicesReshape, uniqueExecutor.get());
  CHECK_RET(firstGatherV2 != nullptr, ACLNN_ERR_PARAM_NULLPTR);

  if (std::isnan(maxNorm)) {
      maxNorm = std::numeric_limits<float>::infinity();
  }

  // 调用l0算子Renorm进行计算，mid_output-> embeddingRenormRenorm
  auto renorm = l0op::Renorm(firstGatherV2, static_cast<float>(normType), EMBEDDING_RENORM_RENORM_DEFAULT_DIM,
                             static_cast<float>(maxNorm), uniqueExecutor.get());
  CHECK_RET(renorm != nullptr, ACLNN_ERR_PARAM_NULLPTR);

  // 对第二次GatherV2的indices的数据进行处理
  int64_t indicesDim0 = indicesReshape->GetViewShape().GetDim(0);

  // indices size转换为tensor
  op::Shape indicesShape = {indicesDim0};
  const aclTensor* index = (uniqueExecutor.get())->AllocTensor(indicesShape, op::DataType::DT_INT32);
  CHECK_RET(index != nullptr, ACLNN_ERR_INNER_NULLPTR);
  auto start = (uniqueExecutor.get())->AllocScalar(0);
  auto end = (uniqueExecutor.get())->AllocScalar(indicesDim0);
  auto step = (uniqueExecutor.get())->AllocScalar(1);
  // 调用L0算子Arange
  auto indicesTensor = l0op::Arange(start, end, step, index, false, uniqueExecutor.get());
  CHECK_RET(indicesTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用l0算子GatherV2进行计算
  auto secondGatherV2 = l0op::GatherV2(renorm, GATHERV2_DEFAULT_AXIS, indicesTensor, uniqueExecutor.get());
  CHECK_RET(secondGatherV2 != nullptr, ACLNN_ERR_PARAM_NULLPTR);

  // 进行Mul计算
  auto update = l0op::Mul(firstGatherV2, secondGatherV2, uniqueExecutor.get());
  CHECK_RET(update != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 调用l0算子ScatterUpdate进行计算
  auto embeddingRenormScatterUpdate = l0op::ScatterUpdate(selfRefContiguous, indicesReshape, update,
                                                          uniqueExecutor.get());
  CHECK_RET(embeddingRenormScatterUpdate != nullptr, ACLNN_ERR_PARAM_NULLPTR);

  // 固定写法，将计算结果拷贝到输出selfRef上
  auto view_copy_result = l0op::ViewCopy(embeddingRenormScatterUpdate, selfRef, uniqueExecutor.get());
  CHECK_RET(view_copy_result != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);

  return ACLNN_SUCCESS;
}

aclnnStatus aclnnEmbeddingRenorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                 const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnEmbeddingRenorm);

  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
