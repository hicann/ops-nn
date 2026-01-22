/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gather_elements.cpp
 * \brief
 */

#include "gather_elements.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_util.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(GatherElements);

static const int64_t SELF_BOUND = 3000 * 1024;
static const int64_t INDEX_BOUND = 2000;
static const int64_t ONE_BYTE = 1;
static const int64_t TWO_BYTE = 2;
static const int64_t FOUR_BYTE = 4;
static const int64_t EIGHT_BYTE = 8;
static const int64_t BLOCK_SIZE = 32;
static const int64_t LEAST_REPEAT_TIME = 1;
static const int64_t SMALL_UB_SIZE = 192 * 1024;
static const int64_t UB_SIZE = 256 * 1024;
static const int64_t RESERVED_UB_SIZE = 2 * 1024;
static const int64_t INT_MAX_NUM = 2147483647;
static const int64_t HALF = 2;
static const int64_t DOUBLE_UB = 2;

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32,
    op::DataType::DT_FLOAT16, op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_UINT32,
    op::DataType::DT_INT16, op::DataType::DT_UINT16,
    op::DataType::DT_INT64, op::DataType::DT_UINT64,
    op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_WITH_BOOL = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32,
    op::DataType::DT_FLOAT16, op::DataType::DT_INT8,
    op::DataType::DT_UINT8, op::DataType::DT_UINT32,
    op::DataType::DT_INT16, op::DataType::DT_UINT16,
    op::DataType::DT_INT64, op::DataType::DT_UINT64,
    op::DataType::DT_BF16, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ONE_BYTE_DTYPE_LIST = {
    op::DataType::DT_INT8, op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> ONE_BYTE_DTYPE_LIST_910_95 = {
    op::DataType::DT_INT8, op::DataType::DT_UINT8, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> TWO_BYTE_DTYPE_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_INT16,
    op::DataType::DT_UINT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> FOUR_BYTE_DTYPE_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32,
    op::DataType::DT_UINT32};

static const std::initializer_list<op::DataType> EIGHT_BYTE_DTYPE_LIST = {
    op::DataType::DT_INT64, op::DataType::DT_UINT64};

static const std::initializer_list<op::DataType> VGATHER_DTYPE_LIST = {
    op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static int64_t GetDtypeSize(const op::DataType dtype) {
  auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
  if (Ops::NN::AclnnUtil::IsRegbase(npuArch)) {
    if (op::CheckType(dtype, ONE_BYTE_DTYPE_LIST_910_95)) {
      return ONE_BYTE;
    }
  } else {
    if (op::CheckType(dtype, ONE_BYTE_DTYPE_LIST)) {
      return ONE_BYTE;
    }
  }
  if (op::CheckType(dtype, TWO_BYTE_DTYPE_LIST)) {
    return TWO_BYTE;
  }
  if (op::CheckType(dtype, FOUR_BYTE_DTYPE_LIST)) {
    return FOUR_BYTE;
  }
  if (op::CheckType(dtype, EIGHT_BYTE_DTYPE_LIST)) {
    return EIGHT_BYTE;
  }
  return 0;
}

static int64_t GetTensorSize(const aclTensor *input) {
  const op::Shape shape = input->GetViewShape();
  const size_t dims = shape.GetDimNum();
  int64_t size = 1;
  for (size_t i = 0; i < dims; ++i) {
    size *= shape.GetDim(i);
  }
  return size;
}

static bool IsSameDimValueExceptAxis(const op::Shape x_shape, const op::Shape index_shape, const int64_t axis, const size_t dims) {
  for (int64_t i = 0; i < static_cast<int64_t>(dims); i++) {
    if ((i != axis) && (x_shape[i] != index_shape[i])) {
      return false;
    }
  }
  return true;
}

static bool IsLastAxisSupport(const aclTensor *self, const aclTensor *index, const int64_t axis) {
  const op::Shape x_shape = self->GetViewShape();
  const op::Shape index_shape = index->GetViewShape();
  const size_t dims = x_shape.GetDimNum();
  const int64_t index_dsize = GetDtypeSize(index->GetDataType());
  const int64_t x_dsize = GetDtypeSize(self->GetDataType());
  CHECK_RET(std::min(x_dsize, index_dsize) != 0, false);
  const int64_t large_num_per_block = BLOCK_SIZE / std::min(x_dsize, index_dsize);
  int64_t index_axis = index_shape[axis];
  int64_t x_axis = x_shape[axis];
  int64_t repeat_per_core = LEAST_REPEAT_TIME;
  bool if_same_dim_value_except_axis = IsSameDimValueExceptAxis(x_shape, index_shape, axis, dims);
  bool is_last_axis = (axis == static_cast<int64_t>(dims) - 1);
  auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
  bool isSupportSoc = (npuArch == NpuArch::DAV_2201);
  int64_t ubSize = isSupportSoc ? SMALL_UB_SIZE : UB_SIZE;
  int64_t available_ub_size = ubSize - RESERVED_UB_SIZE;
  int64_t all_data_size = 0;

  // Branch judgment for 910b vgather
  bool vgather910BSupport = op::CheckType(self->GetDataType(), VGATHER_DTYPE_LIST) && isSupportSoc;
  if (is_last_axis && vgather910BSupport && if_same_dim_value_except_axis) {
    all_data_size = x_axis * x_dsize + index_axis * (x_dsize + index_dsize * DOUBLE_UB);
    if (all_data_size < available_ub_size) {
      return true;
    }
  }

  // Branch judgment for cutting indices into slices
  int64_t last_dim_size = x_axis * x_dsize + index_axis * (x_dsize +index_dsize);
  bool cuttingIntoSlicesFlag = last_dim_size >= available_ub_size && if_same_dim_value_except_axis &&\
                               x_axis * x_dsize <= available_ub_size / HALF &&\
                               index_axis % large_num_per_block == 0;
  if (is_last_axis && cuttingIntoSlicesFlag) {
    return true;
  }

  // Normal branches
  if (is_last_axis && if_same_dim_value_except_axis) {
    // unaligned cases
    if (index_axis % large_num_per_block != 0) {
      while (repeat_per_core * index_axis % large_num_per_block != 0) {
        repeat_per_core++;
      }
    }
  } else if (is_last_axis && index_axis < large_num_per_block) {
    return false;
  } else if (!is_last_axis) {
    return false;
  }
  all_data_size = repeat_per_core * x_axis * x_dsize + repeat_per_core * index_axis * (x_dsize + index_dsize);
  if (all_data_size >= available_ub_size) {
    return false;
  }
  return true;
}

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCoreSupport(const aclTensor *self, const aclTensor *index, const int64_t axis) {
    auto npuArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (Ops::NN::AclnnUtil::IsRegbase(npuArch)) {
      if (!op::CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST_WITH_BOOL)) {
          return false;
      }
    } else {
      if (!op::CheckType(self->GetDataType(), AICORE_DTYPE_SUPPORT_LIST)) {
          return false;
      }
    }

    if (Ops::NN::AclnnUtil::IsRegbase(npuArch)) {
        return true;
    }

    if (GetDtypeSize(self->GetDataType()) * GetTensorSize(self) > INT_MAX_NUM) {
        return false;
    }
    if (self->GetViewShape()[axis] > (INT_MAX_NUM / HALF)) {
        return false;
    }

    if (IsLastAxisSupport(self, index, axis)) {
        return true;
    }
    if (GetDtypeSize(self->GetDataType()) * GetTensorSize(self) > SELF_BOUND && GetTensorSize(index) > INDEX_BOUND) {
        return false;
    }
    return true;
}

const aclTensor *GatherElements(const aclTensor *self, const int64_t dim,
                                const aclTensor *index, aclOpExecutor *executor) {
  L0_DFX(GatherElements, self, dim, index);
  CHECK_RET(self != nullptr, nullptr);
  CHECK_RET(index != nullptr, nullptr);

  auto out = executor->AllocTensor(index->GetViewShape(), self->GetDataType());
  auto ret = ACL_SUCCESS;
  if (IsAiCoreSupport(self, index, dim)) {
    ret = ADD_TO_LAUNCHER_LIST_AICORE(GatherElements, OP_INPUT(self, index), OP_OUTPUT(out), OP_ATTR(dim));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GatherElements AiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  } else {
    static internal::AicpuTaskSpace space("GatherElements");
    ret = ADD_TO_LAUNCHER_LIST_AICPU(GatherElements, OP_ATTR_NAMES({"dim"}), OP_INPUT(self, index),
                               OP_OUTPUT(out), OP_ATTR(dim));
    OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "GatherElements AiCpu ADD_TO_LAUNCHER_LIST_AICPU failed."), return nullptr);
  }
  return out;
}
} // l0op
