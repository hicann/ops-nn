/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <dlfcn.h>
#include <new>
#include "aclnn_grouped_dynamic_mx_quant.h"
#include "grouped_dynamic_mx_quant.h"
#include "level0/fault_injection.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "acl/acl.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif
static constexpr int64_t X_DIM_NUM = 2;
static constexpr int64_t NUM_TWO = 2;
static constexpr int64_t SCALE_DIM_NUM = 3;
static constexpr uint64_t NUM_ZERO = 0 ;

static const std::initializer_list<op::DataType> X_DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> GROUP_INDEX_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT32};

static const std::initializer_list<op::DataType> OUTPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2};

static const std::initializer_list<op::DataType> MXSCALE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT8_E8M0};

static inline bool CheckNotNull(const aclTensor* x, const aclTensor* groupIndex, const char* roundMode, const aclTensor* y, const aclTensor* mxscale) {
  OP_CHECK_NULL(x, return false);
  OP_CHECK_NULL(groupIndex, return false);
  if (roundMode == nullptr) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "roundMode cannot be nullptr");
    return false;
  }
  OP_CHECK_NULL(y, return false);
  OP_CHECK_NULL(mxscale, return false);
  return true;
}

static bool CheckShape(const aclTensor* x, const aclTensor* groupIndex, int64_t blocksize, const aclTensor* y, const aclTensor* mxscale) {
  auto xShape = x->GetViewShape();
  auto groupShape = groupIndex->GetViewShape();
  auto yShape = y->GetViewShape();
  auto mxscaleShape = mxscale->GetViewShape();
  OP_CHECK(xShape.GetDimNum() == X_DIM_NUM,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input x Dims is %ld, should be 2D.", xShape.GetDimNum()), return false);
  OP_CHECK(groupShape.GetDimNum() == 1,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "input groupIndex Dims is %ld, should be 1D.", groupShape.GetDimNum()), return false);
  OP_CHECK(yShape.GetDimNum() == X_DIM_NUM,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "output yShape Dims is %ld, should be 2D.", yShape.GetDimNum()), return false);
  OP_CHECK(mxscaleShape.GetDimNum() == SCALE_DIM_NUM,
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "output mxscale Dims is %ld, should be 3D.", mxscaleShape.GetDimNum()), return false);
  OP_CHECK_SHAPE_NOT_EQUAL(y, x, return false);
  int64_t xDim0 = xShape.GetDim(0);
  int64_t xDim1 = xShape.GetDim(1);
  int64_t groupIndexDim0 = groupShape.GetDim(0);
  int64_t mxscaleDim0 = mxscaleShape.GetDim(0);
  int64_t mxscaleDim1 = mxscaleShape.GetDim(1);
  int64_t mxscaleDim2 = mxscaleShape.GetDim(NUM_TWO);
  int64_t mxscaleDim0Count = (xDim0/(blocksize * NUM_TWO) + groupIndexDim0);
  OP_CHECK(mxscaleDim2 == NUM_TWO,
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mxscale dim2 is %ld, should be 2.", mxscaleDim1), return false);
  OP_CHECK(
      xDim1 == mxscaleDim1,
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mxscale dim1 is %ld, should be same as x dim1 (%ld).", mxscaleDim1, xDim1),
      return false);
  OP_CHECK(mxscaleDim0 == mxscaleDim0Count,
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mxscale dim0 is %ld, should be same with mxscaleDim0Count (%ld).", mxscaleDim0, mxscaleDim0Count), return false);
  return true;
}

static bool CheckDtypeValid(const aclTensor* x, const aclTensor* groupIndex, const char* roundMode, int64_t dstType, 
                              int64_t blocksize, const aclTensor* y, const aclTensor* mxscale) {
  // 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  bool IsRegbaseSocVersion = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95;
  if (IsRegbaseSocVersion) {
    OP_CHECK_DTYPE_NOT_SUPPORT(x, X_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(groupIndex, GROUP_INDEX_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(y, OUTPUT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(mxscale, MXSCALE_DTYPE_SUPPORT_LIST, return false);
    const std::string mode = std::string(roundMode);
    OP_CHECK(mode == "rint",
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expected roundMode equals 'rint', get: %s", mode.c_str()),
            return false);
    OP_CHECK(blocksize == 32,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "blocksize only support '32' now, get: %ld", blocksize),
            return false);
    OP_CHECK(static_cast<int64_t>(y->GetDataType()) == dstType,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dstType:%ld(%s) is must be the same as y dtype[%s].",
                dstType, op::ToString(static_cast<op::DataType>(dstType)).GetString(), op::ToString(y->GetDataType()).GetString()),
                return false);
  } else {
    OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "support for %s is not implemented",
      op::ToString(GetCurrentPlatformInfo().GetSocVersion()).GetString());
    return false;
  }
  return true;
}

inline static aclnnStatus CheckParams(const aclTensor* x, const aclTensor* groupIndex, const char* roundMode, int64_t dstType, int64_t blocksize, 
                                      const aclTensor* y, const aclTensor* mxscale) {
  CHECK_RET(CheckNotNull(x, groupIndex, roundMode, y, mxscale), ACLNN_ERR_PARAM_NULLPTR);
  CHECK_RET(CheckDtypeValid(x, groupIndex, roundMode, dstType, blocksize, y, mxscale), ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(CheckShape(x, groupIndex, blocksize, y, mxscale), ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedDynamicMxQuantGetWorkspaceSize(const aclTensor* x, const aclTensor* groupIndex,
                                                const char* roundMode, int64_t dstType, int64_t blocksize, 
                                                const aclTensor* y, const aclTensor* mxscale,
                                                uint64_t* workspaceSize, aclOpExecutor** executor) {
  L2_DFX_PHASE_1(aclnnGroupedDynamicMxQuant, DFX_IN(x, groupIndex, roundMode, dstType, blocksize),
                 DFX_OUT(y, mxscale));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(x, groupIndex, roundMode, dstType, blocksize, y, mxscale);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 空Tensor处理
  if (groupIndex->IsEmpty()) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "groupIndex does not support empty values.");
    return ACLNN_ERR_PARAM_INVALID;
  }

  if (x->IsEmpty()) {
    *workspaceSize = NUM_ZERO;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // x如果非连续，需要转连续
  auto selfContiguous = l0op::Contiguous(x, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto groupIndexContiguous = l0op::Contiguous(groupIndex, uniqueExecutor.get());
  CHECK_RET(groupIndexContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto result = l0op::GroupedDynamicMxQuant(selfContiguous, groupIndexContiguous, roundMode, dstType, blocksize,
                                 uniqueExecutor.get());
  const aclTensor *yOut = std::get<0>(result);
  const aclTensor *mxscaleOut = std::get<1>(result);
  // 如果出参y是非连续Tensor，需要把计算完的连续Tensor转非连续
  auto viewCopyResult0 = l0op::ViewCopy(yOut, y, uniqueExecutor.get());
  CHECK_RET(viewCopyResult0 != nullptr, ACLNN_ERR_INNER_NULLPTR);
  if (!IsContiguous(mxscale)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mxscale must be contiguous.");
    return ACLNN_ERR_PARAM_INVALID;
  }
  auto viewCopyResult1 = l0op::ViewCopy(mxscaleOut, mxscale, uniqueExecutor.get());
  CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedDynamicMxQuant(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnGroupedDynamicMxQuant);
  auto ret = CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
  if (ret != ACLNN_SUCCESS) {
    OP_LOGE(ACLNN_ERR_INNER, "This is an error in GroupedDynamicMxQuant launch aicore");
    return ACLNN_ERR_INNER;
  }
  return ACLNN_SUCCESS;
}
#ifdef __cplusplus
}
#endif