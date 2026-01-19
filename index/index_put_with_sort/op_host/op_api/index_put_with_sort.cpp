/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file index_put_with_sort.cpp
 * \brief
 */

#include "index_put_with_sort.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
using namespace op;

static const int64_t INT32_MAX_LIMIT = 2147483647;
static const int64_t CAST_MAX_NUM = 16777216;
static const int64_t INDICES_BASE = 20000;
static const int64_t INDICES_LIMIT = 100000;
static const int64_t SLICESIZE_LIMIT = 1024;
static const int64_t SLICESIZE_BASE = 128;
static const int64_t INDICES_MIN_LIMIT = 100;
static const int64_t SLICESIZE_FP16_LIMIT = 30000;
static const int64_t MEMORY_LIMIT = 250 * 1024 * 1024; // 250M
static const int64_t SIZE_OF_FLOAT = 4;
static const int64_t SIZE_OF_HALF = 2;
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_INDEX_PUT_WITH_SORT = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

namespace l0op {

OP_TYPE_REGISTER(IndexPutWithSort);

static bool CheckSocVersion() {
  if (op::GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910B &&
      op::GetCurrentPlatformInfo().GetSocVersion() != SocVersion::ASCEND910_93) {
      OP_LOGD("IndexPutWithSort Op not support this Soc Version!");
      return false;
  }
  return true;
}

static bool CheckSelfAndValuesDtype(aclTensor *selfRef, const aclTensor *values) {
  if (!CheckType(selfRef->GetDataType(), DTYPE_SUPPORT_LIST_INDEX_PUT_WITH_SORT)) {
      OP_LOGD("IndexPutWithSort Op not support this selfRef dtype.");
      return false;
  }
  if (!CheckType(values->GetDataType(), DTYPE_SUPPORT_LIST_INDEX_PUT_WITH_SORT)) {
      OP_LOGD("IndexPutWithSort Op not support this values dtype.");
      return false;
  }
  if (selfRef->GetDataType() != values->GetDataType()) {
      OP_LOGD("IndexPutWithSort Op not support selfRef dtype != values dtype.");
      return false;
  }
  return true;
}

static bool CheckIndicesWithSelf(aclTensor *selfRef, const aclTensorList *indices, const aclTensor *values) {
  int64_t indicesSize = static_cast<int64_t>(indices->Size());
  int64_t selfRefSize = selfRef->GetViewShape().GetDimNum();
  int64_t valuesSize = values->GetViewShape().GetDimNum();
  // 1 不支持没有索引，不支持self values, index为0维
  if (indicesSize < 1 || selfRefSize < 1 || valuesSize < 1 ||
      (*indices)[0]->GetViewShape().GetDimNum() < 1) {
      OP_LOGD("IndexPutWithSort Op not support no indices, not support 0 dims of selfRef or values!");
      return false;
  }
  // 2 不支持索引个数比self维度多（非法输入）
  if (indicesSize > selfRefSize) {
      OP_LOGD("IndexPutWithSort Op not support nums of indices greater than dims of selfRef!");
      return false;
  }
  // 3 索引必须从首轴开始且连续，不支持有空索引
  for (int64_t i = 0; i < indicesSize; i++) {
      if ((*indices)[i]) {
          if ((*indices)[i]->GetViewShape().GetShapeSize() == 0) {
              OP_LOGD("IndexPutWithSort Op not support none data of indices!");
              return false;
          }
      } else {
          OP_LOGD("IndexPutWithSort Op not support nullptr of indices!");
          return false;
      }
  }
  return true;
}

static bool CheckIndicesDtypeAndShape(const aclTensorList *indices) {
  int64_t indicesSize = static_cast<int64_t>(indices->Size());
  for (int64_t i = 0; i < indicesSize; i++) {
      if (i == 0) {
          if ((*indices)[0]->GetDataType() == op::DataType::DT_BOOL) {
              OP_LOGD("IndexPutWithSort Op not support bool dtype of indices!");
              return false;
          }
      } else {
          if ((*indices)[i]->GetDataType() != (*indices)[0]->GetDataType()) {
              OP_LOGD("IndexPutWithSort Op only support one dtype of indices!");
              return false;
          }
          // 索引之间需要广播不支持
          if ((*indices)[i]->GetViewShape().GetDimNum() != (*indices)[0]->GetViewShape().GetDimNum()) {
              OP_LOGD("IndexPutWithSort Op only support same dims of indices!");
              return false;
          }
          for (size_t j = 0; j < (*indices)[0]->GetViewShape().GetDimNum(); j++) {
              if ((*indices)[0]->GetViewShape().GetDim(j) != (*indices)[i]->GetViewShape().GetDim(j)) {
                  OP_LOGD("IndexPutWithSort Op only support same shape of indices!");
                  return false;
              }
          }
      }
  }
  return true;
}

static bool CheckValuesShape(const aclTensorList *indices, const aclTensor *values) {
  auto valuesSize = values->GetViewShape().GetDimNum();
  if (valuesSize < (*indices)[0]->GetViewShape().GetDimNum()) {
      OP_LOGD("IndexPutWithSort Op not support dims of values smaller than dims of indices!");
      return false;
  }
  for (size_t i = 0; i < (*indices)[0]->GetViewShape().GetDimNum(); i++) {
      if (values->GetViewShape().GetDim(i) != (*indices)[0]->GetViewShape().GetDim(i) &&
          values->GetViewShape().GetDim(i) != 1) {
          OP_LOGD("IndexPutWithSort Op not support values need broadcast!");
          return false;
      }
  }
  return true;
}

static bool CheckSliceSize(aclTensor *selfRef, const aclTensorList *indices, const aclTensor *values) {
  int64_t indicesSize = static_cast<int64_t>(indices->Size());
  int64_t selfRefSize = selfRef->GetViewShape().GetDimNum();

  int64_t SliceDims = selfRefSize - indicesSize;
  auto indexDims = (*indices)[0]->GetViewShape().GetDimNum();
  for (int64_t i = 0; i < SliceDims; i++) {
    auto selfDimValue = selfRef->GetViewShape().GetDim(indicesSize + i);
    auto valuesDimValue = values->GetViewShape().GetDim(indexDims + i);
    if (selfDimValue != valuesDimValue && valuesDimValue != 1) {
      OP_LOGD("IndexPutWithSort Op only support values shape not match selfRef!");
      return false;
    }
  }
  return true;
}

static bool CheckDataSize(aclTensor *selfRef, const aclTensorList *indices) {
  auto indicesSize = indices->Size();
  int64_t shapeProd = 1;
  for (size_t i = 0; i < indicesSize; i++) {
    if (selfRef->GetViewShape().GetDim(i) > CAST_MAX_NUM) {
      OP_LOGD("IndexPutWithSort Op not support dim value of selfRef greater than %ld!", CAST_MAX_NUM);
      return false;
    }
    shapeProd *= selfRef->GetViewShape().GetDim(i);
  }
  if (shapeProd > INT32_MAX_LIMIT) {
    OP_LOGD("IndexPutWithSort Op not support nums of tail data greater than %ld!", INT32_MAX_LIMIT);
    return false;
  }
  return true;
}

static bool IndexPutWithSortBetter(aclTensor *selfRef, const aclTensorList *indices, const aclTensor *values) {
  int64_t indicesNums = static_cast<int64_t>((*indices)[0]->GetViewShape().GetShapeSize());
  int64_t sliceSize = 1;
  auto indicesSize = indices->Size();
  auto selfRefSize = selfRef->GetViewShape().GetDimNum();
  auto selfShape = selfRef->GetViewShape();
  if (selfRefSize > indicesSize) {
      for (size_t i = indicesSize; i < selfRefSize; i++) {
          sliceSize *= static_cast<int64_t>(selfShape.GetDim(i));
      }
  }
  OP_LOGD("IndexPutWitSort Op indicesNums %ld, sliceSize %ld", indicesNums, sliceSize);
  if (sliceSize > INT32_MAX_LIMIT) {
     OP_LOGD("IndexPutWithSort Op not support sliceSize greater than %ld!", INT32_MAX_LIMIT);
     return false;
  }
  if (indicesNums >= INDICES_BASE && sliceSize <= SLICESIZE_BASE) {
    // 小尾轴场景
    return true;
  }
  if (selfRef->GetDataType() == op::DataType::DT_FLOAT &&
      indicesNums >= INDICES_LIMIT && sliceSize <= SLICESIZE_LIMIT) {
     // 输入索引多且尾轴稍大
     return true;
  }
  if (selfRef->GetDataType() == op::DataType::DT_FLOAT16 || selfRef->GetDataType() == op::DataType::DT_BF16) {
    // 索引多 或 尾轴大 或 尾轴较大且索引较多
    if (indicesNums >= INDICES_LIMIT || sliceSize >= INDICES_LIMIT ||
        (indicesNums >= INDICES_MIN_LIMIT && sliceSize >= SLICESIZE_FP16_LIMIT)) {
      return true;
    }
    // self和values升精度前后总内存超过250M
    int64_t dataNums = static_cast<int64_t>(selfRef->GetViewShape().GetShapeSize() + values->GetViewShape().GetShapeSize());
    if (dataNums * (SIZE_OF_FLOAT + SIZE_OF_HALF) > MEMORY_LIMIT) {
      return true;
    }
  }
  return false;
}

bool IsIndexPutWithSortSupport(aclTensor *selfRef, const aclTensorList *indices, const aclTensor *values,
                               int64_t deterministicValue, const bool accumulate) {
  // 1. 芯片限制
  if (!CheckSocVersion()) {
    return false;
  }
  // 2. selfRef, values数据类型限制
  if (!CheckSelfAndValuesDtype(selfRef, values)) {
    return false;
  }
  // 3. indices限制
  if (!CheckIndicesWithSelf(selfRef, indices, values)) {
    return false;
  }
  // 4. 索引不能为bool，且索引数据类型一致，且不支持索引之间广播
  if (!CheckIndicesDtypeAndShape(indices)) {
    return false;
  }
  // 5. values shape限制，values维度数不能小于索引维度数，支持values广播
  if (!CheckValuesShape(indices, values)) {
    return false;
  }
  // 6. 尾轴限制，self的尾轴和values必须相同，或可广播
  if (!CheckSliceSize(selfRef, indices, values)) {
    return false;
  }
  // 7. 数据量限制，每个索引的取值范围不超过CAST_MAX_NUM，且self尾轴个数不超过INT32_MAX_LIMIT
  if(!CheckDataSize(selfRef, indices)) {
    return false;
  }
  // 8. indexputv2优势限制
  if(deterministicValue == 0 && (!IndexPutWithSortBetter(selfRef, indices, values))) {
    return false;
  }
  // 9. 非确定性计算的替换模式不走indexputwithsort
  if (deterministicValue == 0 && (!accumulate)) {
    return false;
  }
  return true;
}

const aclTensor *IndexPutWithSort(const aclTensor *self, const aclTensor *linearIndex, const aclTensor *posIdx,
                                  const aclTensor *values, const int32_t sliceSize, const bool accumulate,
                                  aclTensor *out, aclOpExecutor *executor) {
  L0_DFX(IndexPutWithSort, self, linearIndex, posIdx, values, sliceSize, accumulate);
  
  auto ret = ADD_TO_LAUNCHER_LIST_AICORE(IndexPutWithSort,
                                         OP_INPUT(self, linearIndex, posIdx, values),
                                         OP_OUTPUT(out),
                                         OP_ATTR(sliceSize, accumulate));
  OP_CHECK(ret == ACLNN_SUCCESS, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "IndexPutWithSortAiCore ADD_TO_LAUNCHER_LIST_AICORE failed."), return nullptr);
  return out;
}
}  // namespace l0op

