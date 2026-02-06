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
 * \file aclnn_index_put_impl.cpp
 * \brief IndexPut Aclnn file
 */
#include "aclnn_index_put_impl.h"
#include "index/index_put/op_host/op_api/index_put.h"
#include "index_put_v2.h"
#include "index/linear_index_v2/op_host/op_api/linear_index_v2.h"
#include "index/index_put_with_sort_v2/op_api/index_put_with_sort_v2.h"
#include "index/index_put_with_sort/op_host/op_api/index_put_with_sort.h"
#include "level0/sort.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/cast.h"
#include "level0/broadcast_to.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/shape_utils.h"
#include "op_api/aclnn_util.h"
#include "opdev/platform.h"
#include "op_api/level2_base.h"
#if __has_include("runtime/context.h")
#include "runtime/context.h"
#else
#include "runtime/runtime/context.h"
#endif

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

//  根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
    op::DataType::DT_BOOL, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_SORT = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_910B_SUPPORT_ATOMIC = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT8, op::DataType::DT_BOOL, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_310B_SUPPORT_ATOMIC = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_FLOAT16,
    op::DataType::DT_BOOL, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> INDICES_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_BOOL};

static const int64_t MAX_INDICES_NUM = 20000;
static const int64_t MAX_SUPPORTTYPE_INDICES_NUM = 60000000; // 维度小于5时的最大索引数限制
static const int64_t MAX_SUPPORTTYPE_INDICES_NUM_DIM5 = 54247424; // 维度5时的最大索引数限制
static const int64_t MAX_SUPPORTTYPE_INDICES_NUM_DIM6 = 48611328; // 维度6时的最大索引数限制
static const int64_t MAX_SUPPORTTYPE_INDICES_NUM_DIM7 = 44384256; // 维度7时的最大索引数限制
static const int64_t MAX_SUPPORTTYPE_INDICES_NUM_DIM8 = 40861696; // 维度8时的最大索引数限制
static const int64_t MAX_RESERVE_NUM = 100;
static const int64_t MULTI_DTYPE_SUPPORT_NUM = 200;
static const int64_t MULTI_DTYPE_SUPPORT_TAIL = 128;
static const int64_t MAX_AICORE_TIAL = 70;
static const int64_t MAX_AICORE_INDICES = 50;
static const int64_t MAX_SHPAE_NUM = 5400000;
static const int64_t DIMLIMIT = 8;
static const int64_t TAIL_SIZE = 1024;
static const uint64_t INT32_INF = 2139095040;
static const uint64_t INT32_MAX_LIMIT = 2147483647;
static const uint64_t CAST_MAX_NUM = 16777216;
static constexpr size_t DIM_BOUND_NON_CONTIGUOUS = 4;

static aclIntArray* GetPerm(int64_t masksNum, int64_t indicesNum, int64_t transposeDimNum, aclOpExecutor* executor) {
  FVector<int64_t, DIMLIMIT> transposeArray;
  for (int64_t i = masksNum - indicesNum; i < transposeDimNum; i++){
    transposeArray.emplace_back(i);
  }
  for (int64_t i = 0; i < masksNum - indicesNum; i++){
    transposeArray.emplace_back(i);
  }
  auto perm = executor->AllocIntArray(transposeArray.data(), transposeDimNum);
  return perm;
}

static aclIntArray* GetPermBack(int64_t masksNum, int64_t indicesNum, int64_t transposeDimNum, aclOpExecutor* executor) {
  FVector<int64_t, DIMLIMIT> transposeArray;
  for (int64_t i = transposeDimNum - (masksNum - indicesNum); i < transposeDimNum; i++){
    transposeArray.emplace_back(i);
  }
  for (int64_t i = 0; i < transposeDimNum - (masksNum - indicesNum); i++){
    transposeArray.emplace_back(i);
  }
  auto perm = executor->AllocIntArray(transposeArray.data(), transposeDimNum);
  return perm;
}

static bool CheckDtypeEqual(const aclTensor *selfRef, const aclTensor *values) {
  // 检查selfRef和other能否做数据类型推导
  OP_CHECK_DTYPE_NOT_MATCH(selfRef, values->GetDataType(),  return false);
  return true;
}

static bool CheckNotNull(const aclTensor *self,
                        const aclTensorList *indices,
                        const aclTensor *values)
{
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(values, return false);
  OP_CHECK_NULL(indices, return false);
  for (uint64_t i = 0; i < indices->Size(); i++)
  {
    OP_CHECK_NULL((*indices)[i], return false);
  }
  return true;
}

static bool CheckDtypeValid(const aclTensor *self,
                            const aclTensor *value)
{
  if (op::GetCurrentPlatformInfo().GetSocVersion() < op::SocVersion::ASCEND910B) {
    if (self->GetDataType() == op::DataType::DT_BF16) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self not implemented for DT_BF16, when SocVersion is less than ASCEND910B.");
      return false;
    }
    if (value->GetDataType() == op::DataType::DT_BF16) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "value not implemented for DT_BF16, when SocVersion is less than ASCEND910B.");
      return false;
    }
  }
  // 检查self的数据类型是否在IndexPut算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(value, DTYPE_SUPPORT_LIST, return false);
  return true;
}

static bool IsAiCPUSupportCheckIndicesArch3510(const aclTensor *selfRef, const FVector<const aclTensor*, 8> &indices,
                                const aclTensor *value) {
  /**
   * Using AICPU when allDefinedIndices size is more than two or the type of selfRef tensor is double.
   * Otherwise AICORE, indices tensors are at the discontinuous axis position, implemented by AICPU.
   */
  size_t indices_size = static_cast<int64_t>(indices.size());
  // 不超过8维
  if (indices_size == 0 || value->GetViewShape().GetDimNum() >= DIMLIMIT) {
    return true;
  }
  // 索引不为int32/int64类型时走aicpu分支
  for (int i = 0; i < static_cast<int>(indices.size()); i++) {
      if (indices[i]->GetDataType() != op::DataType::DT_INT32 && indices[i]->GetDataType() != op::DataType::DT_INT64) {
          return true;
      }
  }
  // self为double/int16类型时走aicpu分支
  if (selfRef->GetDataType() == op::DataType::DT_DOUBLE || selfRef->GetDataType() == op::DataType::DT_INT16) {
    OP_LOGD("IndexPutV2 not support int16 or float64");
    return true;
  }
  // tiling失败走aicpu逻辑
  for (int i = 0; i < static_cast<int>(indices.size()); i++) {
    if (indices[i]->GetViewShape().GetDimNum() != 0 && indices[i]->GetViewShape().GetShapeSize() != 0) {  // Find first unscalar/unEmpty tensor
      if (indices[i]->GetViewShape().GetShapeSize() > MAX_SUPPORTTYPE_INDICES_NUM) {
        OP_LOGD("IndexPutV2 not support indices num greater than 60000000.");
        return true;
      }
      break;
    }
  }
  return false;
}

static bool IsAiCPUSupportCheckIndices(const FVector<const aclTensor*, 8>& indices,
                                       const aclTensor* value)
{
    /**
     * Using AICPU when allDefinedIndices size is more than two or the type of selfRef tensor is double.
     * Otherwise AICORE, indices tensors are at the discontinuous axis position, implemented by AICPU.
     */
    size_t indices_size = static_cast<int64_t>(indices.size());
    // 不超过8维
    if (indices_size == 0 || value->GetViewShape().GetDimNum() >= DIMLIMIT) {
        return true;
    }

    // 索引不为int32/int64类型时走aicpu分支
    if (indices[0]->GetDataType() != op::DataType::DT_INT32 && indices[0]->GetDataType() != op::DataType::DT_INT64) {
        return true;
    }
    // 索引需要广播时偶走aicpu分支
    for (size_t i = 1; i < indices_size; i++) {
        // 索引为bool类型时走aicpu分支
        if (indices[i]->GetDataType() == op::DataType::DT_BOOL) {
            return true;
        }
        // 索引之间需要广播
        if (indices[0]->GetViewShape().GetDimNum() != indices[i]->GetViewShape().GetDimNum()) {
            return true;
        }
        for (size_t j = 0; j < indices[0]->GetViewShape().GetDimNum(); j++) {
            if (indices[0]->GetViewShape().GetDim(j) != indices[i]->GetViewShape().GetDim(j)) {
                return true;
            }
        }
    }
    return false;
}

static bool IndexPutV2IndicesNumsLimit(const FVector<const aclTensor*, 8> &indices) {
  // 以下情况会在indexPutV2的tiling里被拦截，需要路由到aicpu
  auto indicesNums = indices[0]->GetViewShape().GetShapeSize();
  auto dims = indices.size();
  OP_LOGD("indices size is %ld, indices nums is %ld", dims, indicesNums);
  // aclnn中原本的拦截量
  if (indicesNums > MAX_SUPPORTTYPE_INDICES_NUM) {
    OP_LOGD("IndexPutV2 not support indices num greater than 60000000.");
    return false;
  }
  // 高维需要进一步细化拦截量
  if (dims == 5 && indicesNums > MAX_SUPPORTTYPE_INDICES_NUM_DIM5) { // 5 is dims, 54247424 is indices limits
    OP_LOGD("IndexPutV2 not support indices num greater than 54247424 when indices size is 5.");
    return false;
  }
  if (dims == 6 && indicesNums > MAX_SUPPORTTYPE_INDICES_NUM_DIM6) { // 6 is dims, 48611328 is indices limits
    OP_LOGD("IndexPutV2 not support indices num greater than 48611328 when indices size is 6.");
    return false;
  }
  if (dims == 7 && indicesNums > MAX_SUPPORTTYPE_INDICES_NUM_DIM7) { // 7 is dims, 44384256 is indices limits
    OP_LOGD("IndexPutV2 not support indices num greater than 44384256 when indices size is 7.");
    return false;
  }
  if (dims == 8 && indicesNums > MAX_SUPPORTTYPE_INDICES_NUM_DIM8) { // 8 is dims, 40861696 is indices limits
    OP_LOGD("IndexPutV2 not support indices num greater than 40861696 when indices size is 8.");
    return false;
  }
  return true;
}

// 根据芯片类型、dtype判断算子是否支持走aicore
static bool IsAiCPUSupport(const aclTensor *selfRef, const FVector<const aclTensor*, 8> &indices,
                           const aclTensor *value, const bool accumulate,
                           const FVector<int64_t, 8> masks) {
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  bool isSupportAtomic = (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93 ||
                            socVersion == SocVersion::ASCEND310B);
  bool is910BSocVersion = (socVersion == SocVersion::ASCEND910B || socVersion == SocVersion::ASCEND910_93);
  bool is310BSocVersion = (socVersion == SocVersion::ASCEND310B);
  if (IsAiCPUSupportCheckIndices(indices, value)) {
    return true;
  }
  // 索引为非连续时走aicpu
  size_t start = 0;
  for (size_t i = 0; i < masks.size(); i++) {
    if (masks[i] == 1) {
      break;
    }
    start++;
  }
  bool is_zero_in_masks = false;
  for (size_t i = start; i < masks.size(); i++) {
    if (is_zero_in_masks && masks[i] == 1) {
      return true;
    }
    if (masks[i] == 0) {
      is_zero_in_masks = true;
    }
  }
  // 非首轴连续索引，只支持下一代芯片
  if (start != 0 && isSupportAtomic == false) {
    return true;
  }

  if (selfRef->GetDataType() == op::DataType::DT_DOUBLE || selfRef->GetDataType() == op::DataType::DT_INT16) {
    OP_LOGD("IndexPutV2 not support int16 or float64");
    return true;
  }

  // tiling失败走aicpu逻辑
  if (!IndexPutV2IndicesNumsLimit(indices)) {
    return true;
  }
  int64_t tailSize = 1;
  for (size_t i = masks.size(); i < selfRef->GetViewShape().GetDimNum(); i++) {
    tailSize = tailSize * selfRef->GetViewShape().GetDim(i);
  }
  if (isSupportAtomic == false ||
      (is310BSocVersion && !CheckType(selfRef->GetDataType(), DTYPE_310B_SUPPORT_ATOMIC)) ||
      (is910BSocVersion && !CheckType(selfRef->GetDataType(), DTYPE_910B_SUPPORT_ATOMIC))) {
    if ((tailSize <= MAX_RESERVE_NUM && tailSize > MAX_AICORE_TIAL) &&
        indices[0]->GetViewShape().GetShapeSize() <= MAX_RESERVE_NUM) {
      return true;
    }
    if (tailSize <= MAX_RESERVE_NUM && indices[0]->GetViewShape().GetShapeSize() <= MAX_RESERVE_NUM) {
      if (selfRef ->GetViewShape().GetShapeSize() > MAX_SHPAE_NUM) {
        return true;
      }
    }
    if (tailSize <= TAIL_SIZE && indices[0]->GetViewShape().GetShapeSize() > MAX_INDICES_NUM)  {
      return true;
    }
  }

  // need transpose, 判断transpos后是否支持
  int64_t tailSizeTranspose = 1;
  if (start != 0 && tailSize < TAIL_SIZE && isSupportAtomic == true) {
    for (size_t i = 0; i < start; i++) {
      tailSizeTranspose = tailSizeTranspose * selfRef->GetViewShape().GetDim(i);
    }
    tailSizeTranspose = tailSizeTranspose * tailSize;
  }

  if (indices[0]->GetViewShape().GetShapeSize() > MAX_RESERVE_NUM || tailSize > MULTI_DTYPE_SUPPORT_TAIL ||
      (start != 0 && tailSize < TAIL_SIZE && tailSizeTranspose > MULTI_DTYPE_SUPPORT_TAIL)) {
    if ((selfRef->GetDataType() != op::DataType::DT_FLOAT16 && selfRef->GetDataType() != op::DataType::DT_FLOAT) && isSupportAtomic == false) {
      OP_LOGD("IndexPutV2 Indices_number > 100 and input_dtype is not float, aicore does not support.");
      return true;
    }
    if ((selfRef->GetDataType() == op::DataType::DT_FLOAT16 || selfRef->GetDataType() == op::DataType::DT_FLOAT)
        && accumulate == false && isSupportAtomic == false) {
      OP_LOGD("IndexPutV2 Indices_number > 100 and input_dtype is float16 or float, aicore does not support accumulate false when SocVersion is less than ASCEND910B.");
      return true;
    }
    if (is910BSocVersion && !CheckType(selfRef->GetDataType(), DTYPE_910B_SUPPORT_ATOMIC)) {
      OP_LOGD("IndexPutV2 Indices_number > 100 and input_dtype is uint8 or int64, aicore does not support.");
      return true;
    }
    if (is310BSocVersion && !CheckType(selfRef->GetDataType(), DTYPE_310B_SUPPORT_ATOMIC)) {
      OP_LOGD("IndexPutV2 Indices_number > 100 and input_dtype is uint8, int8 or int64, aicore does not support.");
      return true;
    }
  }
  return false;
}

static inline bool CheckShape(const aclTensor* selfRef, const aclTensorList *indices) {
  // self的维度必须小于等于8
  OP_CHECK_MAX_DIM(selfRef, MAX_SUPPORT_DIMS_NUMS, return false);
  // indices的size必须小于self DimSize
  int64_t indicesSize = static_cast<int64_t>(indices->Size());
  if (indicesSize > static_cast<int64_t>(selfRef->GetViewShape().GetDimNum())) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "indicesSize must <= self DimSize.");
    return false;
  }
  return true;
}

static aclnnStatus CheckParams(const aclTensor *selfRef, const aclTensorList *indices, const aclTensor *values) {
  // 错误码等DFX方案细化后刷新，错误日志在check接口内打印
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(selfRef, indices, values), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(selfRef, values), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查selfRef和other能否做数据类型推导以及推导的数据类型能否转换为输出数据类型
  CHECK_RET(CheckDtypeEqual(selfRef, values), ACLNN_ERR_PARAM_INVALID);

  // 4. 检查输入的数据shape是否符合条件
  CHECK_RET(CheckShape(selfRef, indices), ACLNN_ERR_PARAM_INVALID);

  return ACLNN_SUCCESS;
}

static const aclTensor* AdaptValueforBroadcast(int64_t valueDimNum, op::Shape &oriValueShape, const aclTensor *valuesCast, aclOpExecutor* executor) {
  FVector<int64_t, DIMLIMIT> valueShape;
  if (valueDimNum <= static_cast<int64_t>(oriValueShape.GetDimNum())) {
    return valuesCast;
  }
  for (int64_t i = 0; i < valueDimNum - static_cast<int64_t>(oriValueShape.GetDimNum()); i++) {
    valueShape.emplace_back(1);
  }
  for (int64_t i = 0; i < static_cast<int64_t>(oriValueShape.GetDimNum()); i++) {
    valueShape.emplace_back(oriValueShape.GetDim(i));
  }
  auto shape = executor->AllocIntArray(valueShape.data(), valueDimNum);

  return l0op::Reshape(valuesCast, shape, executor);
}

static bool CheckTensorValueSameShape(bool needBroadcast, std::vector<int64_t>& tensorShape, op::Shape& valueShape,
                                      int64_t valueDimNum)
{
    for (int32_t i = 0; i < valueDimNum; i++) {
        if (tensorShape[i] != valueShape[i]) {
            needBroadcast = true;
            break;
        }
    }
    return needBroadcast;
}

static const aclTensor* AicoreCompute(const aclTensor *selfCast, const FVector<const aclTensor*, 8> &allDefinedIndices,
                                const aclTensorList *indicesTensorList, const aclTensor *valuesCast,
                                const aclTensor *maskTensor, const FVector<int64_t, 8> masks, const bool accumulate,
                                int64_t masksNum, int64_t indicesNum, aclOpExecutor* executor) {
  auto indicesShape = allDefinedIndices[0]->GetViewShape();
  const aclTensor* valueBroadcast = valuesCast;
  bool needBroadcast = false;
  auto selfShape = selfCast->GetViewShape();
  int64_t selfDimNum = static_cast<int64_t>(selfShape.GetDimNum());
  auto valueShape = valuesCast->GetViewShape();
  int64_t indicesDimNum= allDefinedIndices[0]->GetViewShape().GetDimNum();
  int64_t valueDimNum = selfDimNum - indicesNum + indicesDimNum;
  std::vector<int64_t> tensorShape(valueDimNum);
  for (int32_t i = 0; i < masksNum - indicesNum; i++) {
    tensorShape[i] = selfShape[i];
  }
  for (int32_t i = masksNum - indicesNum; i < masksNum - indicesNum + indicesDimNum; i++) {
    tensorShape[i] = indicesShape[i - masksNum + indicesNum];
  }
  for (int32_t i = masksNum - indicesNum + indicesDimNum; i < valueDimNum; i++) {
    tensorShape[i] = selfShape[i - indicesDimNum + indicesNum];
  }
  if (valueDimNum != static_cast<int64_t>(valueShape.GetDimNum())) {
    valuesCast = AdaptValueforBroadcast(valueDimNum, valueShape, valuesCast, executor);
    needBroadcast = true;
  }
  needBroadcast = CheckTensorValueSameShape(needBroadcast, tensorShape, valueShape, valueDimNum);
  auto valueShapeBroad = executor->AllocIntArray(tensorShape.data(), valueDimNum);
  if (needBroadcast) {
    valueBroadcast = l0op::BroadcastTo(valuesCast, valueShapeBroad, executor);
  }
  int64_t tailSize = 1;
  for (size_t i = masks.size(); i < selfCast->GetViewShape().GetDimNum(); i++) {
    tailSize = tailSize * selfCast->GetViewShape().GetDim(i);
  }
  const aclTensor* indexPutOpOut;
  if (masksNum != indicesNum && tailSize < TAIL_SIZE) {
    auto perm = GetPerm(masksNum, indicesNum, selfDimNum, executor);
    auto valueperm = GetPerm(masksNum, indicesNum, valueDimNum, executor);
    selfCast = l0op::Transpose(selfCast, perm, executor);
    valueBroadcast = l0op::Transpose(valueBroadcast, valueperm, executor);
    aclTensor* out = const_cast<aclTensor*>(selfCast);
    FVector<int64_t, DIMLIMIT> masks_final;
    for (int32_t i = 0; i < indicesNum; i++) {
      masks_final.emplace_back(1);
    }
    auto maskarray = executor->AllocIntArray(masks_final.data(), indicesNum);
    auto masktensor_trans = executor->ConvertToTensor(maskarray, op::ToOpDataType(ACL_INT64));
    indexPutOpOut = l0op::IndexPutV2(selfCast, indicesTensorList, valueBroadcast, masktensor_trans,
                                     accumulate, out, executor);
    auto permback = GetPermBack(masksNum, indicesNum, selfDimNum, executor);
    indexPutOpOut = l0op::Transpose(indexPutOpOut, permback, executor);
  } else {
    aclTensor* out = const_cast<aclTensor*>(selfCast);
    indexPutOpOut = l0op::IndexPutV2(selfCast, indicesTensorList, valueBroadcast, maskTensor,
                                     accumulate, out, executor);
  }
  return indexPutOpOut;
}

void ConstructStrideAndValue(const aclTensor* self, FVector<int64_t> &valueSize, FVector<int64_t> &stride)
{
  auto selfShape = self->GetViewShape();
  valueSize[valueSize.size() - 1] = selfShape.GetDim(valueSize.size() - 1);
  for (int32_t i = valueSize.size() - 2; i >= 0; --i) {
    valueSize[i] = selfShape.GetDim(i);
    stride[i] = stride[i + 1] * valueSize[i + 1];
  }
}

void ConstructStrideAndValueWithPerm(const FVector<int64_t> &permute, FVector<int64_t> &valueSize,
                                     FVector<int64_t> &valueSizeTrans, FVector<int64_t> &stride)
{
  valueSizeTrans[valueSizeTrans.size() - 1] = valueSize[permute[valueSizeTrans.size() - 1]];
  for (int32_t i = valueSizeTrans.size() - 2; i >= 0; --i) {
    valueSizeTrans[i] = valueSize[permute[i]];
    stride[i] = stride[i + 1] * valueSizeTrans[i + 1];
  }
}

bool CheckIsDisContinueIdx(const aclTensorList* indices, int64_t &headNullNum,
                           FVector<const aclTensor*> &definedIndices, aclOpExecutor *executor)
{
  bool res = false;
  bool haveMask = false;
  int64_t indicesSize = static_cast<int64_t>(indices->Size());
  for (int i = 0; i < indicesSize; i++) {
    if ((*indices)[i] && (*indices)[i]->GetViewShape().GetShapeSize() != 0) {
      auto indicesContiguous = l0op::Contiguous((*indices)[i], executor);
      definedIndices.emplace_back(indicesContiguous);
      if (!haveMask) {
        headNullNum = i;
        haveMask = true;
      }
    } else {
      res = haveMask ? true : false;
    }
  }
  return res;
}

void GetPermute(const int64_t &selfSize, const int64_t &headNullNum, FVector<int64_t> &permute)
{
    int32_t cnt = 0;
    for (int i = selfSize - headNullNum; i < selfSize; i++) {
        permute[i] = cnt++;
    }
    for (int i = 0; i < selfSize - headNullNum; i++) {
        permute[i] = cnt++;
    }
}

void GetPermuteBack(const int64_t &selfSize, const int64_t &headNullNum, FVector<int64_t> &permuteBack)
{
    int32_t cnt = 0;
    for (int i = headNullNum; i < selfSize; i++) {
        permuteBack[i] = cnt++;
    }
    for (int i = 0; i < headNullNum; i++) {
        permuteBack[i] = cnt++;
    }
}

static const aclTensor *valuesToBroadcast(int64_t indicesSize, const aclTensor *selfCast,
                                          const aclTensorList *indices, FVector<const aclTensor*> definedIndices,
                                          const aclTensor *valuesCast, aclOpExecutor *executor) {
  int64_t indicesNum = 0;
  int64_t masksNum = 0;
  for (int32_t i = 0; i < indicesSize; i++) {
    if ((*indices)[i]) {
      if ((*indices)[i]->GetViewShape().GetShapeSize() != 0) {
        indicesNum += 1;
        masksNum += 1;
      } else {
        masksNum += 1;
      }
    } else {
        masksNum += 1;
    }
  }
  bool needBroadcast = false;
  auto selfShape = selfCast->GetViewShape();
  int64_t selfDimNum = static_cast<int64_t>(selfShape.GetDimNum());
  const aclTensor* valueBroadcast = valuesCast;
  auto valueShape = valueBroadcast->GetViewShape();
  int64_t indicesDimNum= definedIndices[0]->GetViewShape().GetDimNum();
  auto indicesShape = definedIndices[0]->GetViewShape();
  int64_t valueDimNum = selfDimNum - indicesNum + indicesDimNum;
  std::vector<int64_t> tensorShape(valueDimNum);
  for (int32_t i = 0; i < masksNum - indicesNum; i++) {
    tensorShape[i] = selfShape[i];
  }
  for (int32_t i = masksNum - indicesNum; i < masksNum - indicesNum + indicesDimNum; i++) {
    tensorShape[i] = indicesShape[i - masksNum + indicesNum];
  }
  for (int32_t i = masksNum - indicesNum + indicesDimNum; i < valueDimNum; i++) {
    tensorShape[i] = selfShape[i - indicesDimNum + indicesNum];
  }
  if (valueDimNum != static_cast<int64_t>(valueShape.GetDimNum())) {
    valueBroadcast = AdaptValueforBroadcast(valueDimNum, valueShape, valueBroadcast, executor);
    needBroadcast = true;
  }
  needBroadcast = CheckTensorValueSameShape(needBroadcast, tensorShape, valueShape, valueDimNum);
  auto valueShapeBroad = executor->AllocIntArray(tensorShape.data(), valueDimNum);
  if (needBroadcast) {
    valueBroadcast = l0op::BroadcastTo(valueBroadcast, valueShapeBroad, executor);
  }
  return valueBroadcast;
}

static void ViewDataType(const aclTensor *input, const op::DataType dtype)
{
  if (input == nullptr) {
    OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "view data type error!!");
    return;
  }
  auto tmpTensor = const_cast<aclTensor*>(input);
  tmpTensor->SetDataType(dtype);
  input = tmpTensor;
}

static std::pair<const aclTensor*, const aclTensor*> ProcessIndices(const aclTensor *linearIndex,
                                                                    aclOpExecutor *executor)
{
    if (linearIndex == nullptr) {
      return {nullptr, nullptr};
    }
    int64_t row = linearIndex->GetViewShape().GetDim(0);
    if (row == 1) { // sort在元素为1时走aicpu，耗时达到几百us，避免走sort
      FVector<int32_t> posIdxVec = {0};
      auto posIdx = executor->ConvertToTensor(posIdxVec.data(), posIdxVec.size(), DataType::DT_INT32);
      return std::make_pair(linearIndex, posIdx);
    }

    const aclTensor *indiceViewFloat = executor->CreateView(linearIndex, {row}, linearIndex->GetViewOffset());
    if (indiceViewFloat == nullptr) {
      return {nullptr, nullptr};
    }

    // 修改读取的数据类型
    if (row < static_cast<int64_t>(INT32_INF)) {
      ViewDataType(indiceViewFloat, op::DataType::DT_FLOAT);
      OP_LOGD("aclnnIndexPutImpl: indice sort by aicore");
    }
    // 对index进行sort操作
    indiceViewFloat = l0op::Reshape(indiceViewFloat, {row}, executor);
    if (indiceViewFloat == nullptr) {
      return {nullptr, nullptr};
    }
    auto sortResult = l0op::Sort(indiceViewFloat, -1, false, true, op::DataType::DT_INT32, executor);
    const aclTensor *sortIdxOut = std::get<0>(sortResult);
    const aclTensor *posIdx = std::get<1>(sortResult);
    if (sortIdxOut == nullptr || posIdx == nullptr) {
      return {nullptr, nullptr};
    }
    // 将sort的结果已INT32的数据类型来进行读取
    auto sortIndice = executor->CreateView(sortIdxOut, {row}, sortIdxOut->GetViewOffset());
    if (sortIndice == nullptr) {
      return {nullptr, nullptr};
    }
    sortIndice->SetDataType(op::DataType::DT_INT32);
    return std::make_pair(sortIndice, posIdx);
}

static const aclTensor* valuesToBroadcastArch3510(const aclTensor* selfCast,
                                                FVector<const aclTensor*> definedIndices,
                                                const aclTensor* valuesCast, const FVector<int64_t, 8> masks,
                                                bool iscontiguousIdx, aclOpExecutor* executor)
{
    auto indicesShape = definedIndices[0]->GetViewShape();
    auto selfShape = selfCast->GetViewShape();
    auto valueShape = valuesCast->GetViewShape();
    const aclTensor* valueBroadcast = valuesCast;
    bool needBroadcast = false;
    int64_t indicesNum = definedIndices.size();
    int64_t selfDimNum = static_cast<int64_t>(selfShape.GetDimNum());
    int64_t indicesDimNum = static_cast<int64_t>(indicesShape.GetDimNum());
    int64_t valueDimNum = selfDimNum - indicesNum + indicesDimNum;
    std::vector<int64_t> tensorShape(valueDimNum);
    if (iscontiguousIdx) {
        if (masks[0] == 0) {  // some case like: indices mask=[0110]
            auto selfIndex = 0;
            auto updateIndex = 0;
            while (updateIndex < valueDimNum && selfIndex < selfDimNum &&
                   masks[selfIndex] == 0) {  // add former x_shape
                tensorShape[updateIndex] = selfShape[selfIndex];
                updateIndex++;
                selfIndex++;
            }
            for (int i = 0; i < indicesDimNum; i++) {
                tensorShape[updateIndex] = indicesShape.GetDim(i);  // add indices shape
                updateIndex++;
            }
            // find next 0 index
            while (selfIndex < static_cast<int>(masks.size()) && masks[selfIndex] == 1) {
                selfIndex++;
            }
            while (updateIndex < valueDimNum && selfIndex < selfDimNum) {  // add latter x_shape
                tensorShape[updateIndex] = selfShape[selfIndex];
                updateIndex++;
                selfIndex++;
            }
        }
        if (masks[0] == 1) {  // some case like: indices mask= [1100]
            auto updateIndex = 0;
            auto selfIndex = 0;
            for (int i = 0; i < indicesDimNum; i++) {  // add indices shape
                tensorShape[updateIndex] = indicesShape.GetDim(i);
                updateIndex++;
            }
            // find next 0 index
            while (selfIndex < static_cast<int>(masks.size()) && masks[selfIndex] == 1) {
                selfIndex++;
            }
            while (updateIndex < valueDimNum && selfIndex < selfDimNum) {  // add latter x_shape
                tensorShape[updateIndex] = selfShape[selfIndex];
                updateIndex++;
                selfIndex++;
            }
        }
    } else {  // some case like: 01011, 01101, 10101
        auto updateIndex = 0;
        auto selfIndex = 0;
        for (int i = 0; i < indicesDimNum; i++) {  // add indices shape
            tensorShape[updateIndex] = indicesShape.GetDim(i);
            updateIndex++;
        }
        while (updateIndex < valueDimNum && selfIndex < selfDimNum) {  // add x_shape when mask is 0
            if (masks[selfIndex] == 0) {
                tensorShape[updateIndex] = selfShape[selfIndex];
                updateIndex++;
                selfIndex++;
            } else {
                selfIndex++;
            }
        }
    }
    for (int i = 0; i < valueDimNum; i++) {
        OP_LOGD("Broadcast Shape %d is %ld.", i, tensorShape[i]);
    }
    if (valueDimNum != static_cast<int64_t>(valueShape.GetDimNum())) {
        valueBroadcast = AdaptValueforBroadcast(valueDimNum, valueShape, valueBroadcast, executor);
        needBroadcast = true;
    }
    needBroadcast = CheckTensorValueSameShape(needBroadcast, tensorShape, valueShape, valueDimNum);
    auto valueShapeBroad = executor->AllocIntArray(tensorShape.data(), valueDimNum);
    if (needBroadcast) {
        valueBroadcast = l0op::BroadcastTo(valueBroadcast, valueShapeBroad, executor);
    }
    return valueBroadcast;
}

static std::pair<const aclTensor*, const aclTensor*> ProcessIndicesArch3510(const aclTensor* linearIndex,
                                                                          aclOpExecutor* executor)
{
    if (linearIndex == nullptr) {
        return {nullptr, nullptr};
    }
    int64_t row = linearIndex->GetViewShape().GetDim(0);
    if (row == 1) {
        FVector<int32_t> posIdxVec = {0};
        auto posIdx = executor->ConvertToTensor(posIdxVec.data(), posIdxVec.size(), linearIndex->GetDataType());
        return std::make_pair(linearIndex, posIdx);
    }
    auto sortResult = l0op::Sort(linearIndex, -1, false, true, op::DataType::DT_INT32, executor);
    const aclTensor* sortIdxOut = std::get<0>(sortResult);
    const aclTensor* posIdx = std::get<1>(sortResult);
    if (sortIdxOut == nullptr || posIdx == nullptr) {
        return {nullptr, nullptr};
    }
    return std::make_pair(sortIdxOut, posIdx);
}

bool CheckIfContiguous(const aclTensorList* indices, FVector<const aclTensor*>& definedIndices,
                       FVector<const aclTensor*>& allIndices, aclOpExecutor* executor)
{  // [], idx, idx, [], idx   false
  // [], idx = true
  // [], idx, [] = true
  // idx, [] = true
  // [], idx, [],[], idx, [] = false
  // [], idx, [], [] = true
    bool haveMask = false;
    bool haveMiddleEmptyTensor = false;
    int64_t indicesSize = static_cast<int64_t>(indices->Size());
    for (int i = 0; i < indicesSize; i++) {
        if ((*indices)[i] && (*indices)[i]->GetViewShape().GetShapeSize() != 0) {
            auto indicesContiguous = l0op::Contiguous((*indices)[i], executor);
            definedIndices.emplace_back(indicesContiguous);
            allIndices.emplace_back(indicesContiguous);
            if (!haveMask) {
                haveMask = true;
            }
        } else {
            allIndices.emplace_back((*indices)[i]);
            if (haveMask && !haveMiddleEmptyTensor) { // have indices former, and now is empty indices
              haveMiddleEmptyTensor = true;
              haveMask = false;
            }
        }
    }
    if (haveMask && haveMiddleEmptyTensor) {
        return false;
    } else {
        return true;
    }
}

static const aclTensor* AicpuProcess(const aclTensor* selfRef, const aclTensor* selfCast, const aclTensor* valuesCast,
                                     const FVector<const aclTensor*, 8>& allDefinedIndices, const bool accumulate,
                                     const FVector<int64_t, 8> masks, int64_t masksNum, aclOpExecutor* executor)
{
    bool needHighPrecision = accumulate && (selfCast->GetDataType() == op::DataType::DT_FLOAT16 ||
                                            selfRef->GetDataType() == op::DataType::DT_BF16);
    const aclTensor* selfFp32 = nullptr;
    const aclTensor* valuesFp32 = nullptr;
    if (needHighPrecision) {
        OP_LOGD("Begin IndexPutV2 cast fp16 or bf16 to fp32");
        selfFp32 = l0op::Cast(selfCast, op::DataType::DT_FLOAT, executor);
        valuesFp32 = l0op::Cast(valuesCast, op::DataType::DT_FLOAT, executor);
        CHECK_RET(selfFp32 != nullptr, nullptr);
        CHECK_RET(valuesFp32 != nullptr, nullptr);
    }
    auto indicesTensorList = executor->AllocTensorList(allDefinedIndices.data(), allDefinedIndices.size());
    auto maskArray = executor->AllocIntArray(masks.data(), masksNum);
    auto maskTensor = executor->ConvertToTensor(maskArray, op::ToOpDataType(ACL_INT64));
    auto indicesShape = allDefinedIndices[0]->GetViewShape();
    size_t dimNum = indicesShape.GetDimNum();
    FVector<int64_t, DIMLIMIT> indicesvector;
    for (size_t i = 0; i < dimNum; i++) {
        indicesvector.emplace_back(indicesShape.GetDim(i));
    }

    // 调用IndexPut算子kernel
    const aclTensor* indexPutOpOut;
    if (needHighPrecision) {
        aclTensor* out = const_cast<aclTensor*>(selfFp32);
        indexPutOpOut = l0op::IndexPut(selfFp32, indicesTensorList, valuesFp32, maskTensor, accumulate, out, executor);
    } else {
        aclTensor* out = const_cast<aclTensor*>(selfCast);
        indexPutOpOut = l0op::IndexPut(selfCast, indicesTensorList, valuesCast, maskTensor, accumulate, out, executor);
    }
    return indexPutOpOut;
}

namespace {
static bool ComputeBroadCastShape(FVector<const aclTensor*, DIMLIMIT>& allIndices, std::vector<int64_t> &tensorShape)
{
    OP_LOGD("Enter ComputeBroadCastShape Function");
    if (allIndices.size() == 0) {
        return false;
    }
    op::Shape broadcastShape = allIndices[0]->GetViewShape();
    for (size_t i = 1; i < allIndices.size(); i++) {
        if(!BroadcastInferShape(broadcastShape, allIndices[i]->GetViewShape(), broadcastShape)) {
            return false;
        }
    }
    uint64_t dimnum = broadcastShape.GetDimNum();
    // 将最终广播的shape转移到tensorShape
    for (size_t i = 0; i < dimnum; i++) {
        tensorShape.push_back(broadcastShape.GetDim(i));
    }
    return true;
}

static bool isBroadCastShape(FVector<const aclTensor*, DIMLIMIT>& allIndices, int i, std::vector<int64_t> tensorShape, uint64_t tensorShapeDim)
{
    bool needBroadcast = false;
    for (size_t j = 0; j < tensorShapeDim; j++) {
        if (tensorShape[j] != allIndices[i]->GetViewShape()[j]) {
            needBroadcast = true;
            break;
        }
    }
    return needBroadcast;
}

static bool IndicesBroadcastUndeter(FVector<const aclTensor*, DIMLIMIT>& allIndices, aclOpExecutor* executor)
{
    OP_LOGD("Enter IndicesBroadcast");
    bool needBroadcast = false;
    std::vector<int64_t> tensorShape;
    CHECK_RET(ComputeBroadCastShape(allIndices, tensorShape), ACLNN_ERR_PARAM_INVALID);
    uint64_t tensorShapeDim = tensorShape.size();
    auto dstDtype = op::DataType::DT_INT64;
    auto valueShapeBroad = executor->AllocIntArray(tensorShape.data(), tensorShapeDim);
    for (int i = 0; i < static_cast<int>(allIndices.size()); i++) {
        needBroadcast = false;
        if (tensorShapeDim == allIndices[i]->GetViewShape().GetDimNum()) {
          needBroadcast = isBroadCastShape(allIndices, i, tensorShape, tensorShapeDim);
        } else {
          needBroadcast = true;
        }
        if (!needBroadcast) {
          continue;
        }
        if (allIndices[i]->GetDataType() != dstDtype) {
            allIndices[i] = l0op::Cast(allIndices[i], op::DataType::DT_INT32, executor);
        }
        if (!allIndices[i]->IsEmpty()) {  // scalar tensor need broadcast, empty tensor not
            OP_LOGD("IndicesBroadcast start, index is %d", i);
            allIndices[i] = l0op::BroadcastTo(allIndices[i], valueShapeBroad, executor);
        }
    }
    return true;
}
} // namespace


static bool IndicesBroadcast(FVector<const aclTensor*, DIMLIMIT>& allIndices, aclOpExecutor* executor)
{
    for (int i = 0; i < static_cast<int>(allIndices.size()); i++) {
        if (allIndices[i]->GetViewShape().GetDimNum() != 0 &&
            allIndices[i]->GetViewShape().GetShapeSize() != 0) {  // Find first unscalar tensor
            auto tensorShapeDim = allIndices[i]->GetViewShape().GetDimNum();
            std::vector<int64_t> tensorShape(tensorShapeDim);
            for (int j = 0; j < static_cast<int>(tensorShapeDim); j++) {
                tensorShape[j] = allIndices[i]->GetViewShape().GetDim(j);
            }
            auto valueShapeBroad = executor->AllocIntArray(tensorShape.data(), tensorShapeDim);
            auto dstDtype = allIndices[i]->GetDataType();  // sometimes indices dtypes are not same, need cast.
            for (int j = 0; j < static_cast<int>(allIndices.size()); j++) {
                if (allIndices[j]->GetDataType() != dstDtype) {
                    allIndices[j] = l0op::Cast(allIndices[j], op::DataType::DT_INT32, executor);
                    allIndices[j] = l0op::Cast(allIndices[j], dstDtype, executor);
                }
                if (!allIndices[j]->IsEmpty()) {  // scalar tensor need broadcast, empty tensor not
                    allIndices[j] = l0op::BroadcastTo(allIndices[j], valueShapeBroad, executor);
                }
            }
            break;
        }
    }
    return true;
}

static const aclTensor* IndexPutV2Process(const aclTensor* selfCast, const aclTensor* valuesCast,
                                          const aclTensorList* indices, const bool accumulate,
                                          const FVector<int64_t, 8> masks, int64_t masksNum, const bool isNonContiguous, 
                                          const FVector<const aclTensor*, 8>& allDefinedIndices, aclOpExecutor* executor)
{
    int64_t selfSize = selfCast->GetViewShape().GetDimNum();
    FVector<int64_t, DIMLIMIT> stride(selfSize, 1);
    FVector<int64_t, DIMLIMIT> valueSize(selfSize, 0);
    FVector<const aclTensor*, DIMLIMIT> definedIndices;
    FVector<const aclTensor*, DIMLIMIT> allIndices;
    auto maskArray = executor->AllocIntArray(masks.data(), masksNum);
    auto maskTensor = executor->ConvertToTensor(maskArray, op::ToOpDataType(ACL_INT64));
    const aclTensorList *allIndicesTensorList;
    const aclTensor* valueBroadcast;
    const aclTensor* tmp = selfCast;
    if (isNonContiguous) {
        allIndicesTensorList = executor->AllocTensorList(allDefinedIndices.data(), allDefinedIndices.size());
        valueBroadcast = executor->CreateView(valuesCast, valuesCast->GetViewShape(), valuesCast->GetStorageShape(), valuesCast->GetViewStrides(),
                    valuesCast->GetViewOffset());
        selfCast = executor->CreateView(tmp, tmp->GetViewShape(), tmp->GetStorageShape(), tmp->GetViewStrides(),
                    tmp->GetViewOffset());
    } else {
        ConstructStrideAndValue(selfCast, valueSize, stride);
        bool iscontiguousIdx = CheckIfContiguous(indices, definedIndices, allIndices, executor);
        // Indices Broadcast
        auto ret = IndicesBroadcastUndeter(definedIndices, executor);
        CHECK_RET(ret, nullptr);
        allIndicesTensorList = executor->AllocTensorList(definedIndices.data(), definedIndices.size());
        // Value Broadcast
        valueBroadcast = valuesToBroadcastArch3510(selfCast, definedIndices, valuesCast, masks,
                                                      iscontiguousIdx, executor);
        CHECK_RET(valueBroadcast != nullptr, nullptr);
    }
    
    // IndexPutV2
    const aclTensor* indexPutOpOut;
    aclTensor* out = const_cast<aclTensor*>(selfCast);
    indexPutOpOut =
        l0op::IndexPutV2(selfCast, allIndicesTensorList, valueBroadcast, maskTensor, accumulate, out, executor);
    CHECK_RET(indexPutOpOut != nullptr, nullptr);
    return indexPutOpOut;
}

static const aclTensor* DeterministicProcess(const aclTensor* selfCast, const aclTensor* valuesCast,
                                             const aclTensorList* indices,
                                             const bool accumulate, const FVector<int64_t, 8> masks, int64_t masksNum,
                                             aclOpExecutor* executor)
{
    // when indice_shape = [0], input tensor will be sliced
    // if indices is contiguous:
    //     1. if x_shape is [shape0, shape1, shape2, shape3], indices shape is [idx0, idx1], when mask is 0110,
    //        value_shape should be [shape0, idx0, idx1, shape3], which is [former_idx, indices_idx, latter_idx]
    // if indices is discontiguous:
    //     2. if x_shape is [shape0, shape1, shape2, shape3], indices shape is [idx0, idx1], when mask is 0101,
    //        value_shape should be [idx0, idx1, shape0, shape3], which is [indices_idx, remain_x_idx]
    int64_t selfSize = selfCast->GetViewShape().GetDimNum();
    FVector<int64_t, DIMLIMIT> stride(selfSize, 1);
    FVector<int64_t, DIMLIMIT> valueSize(selfSize, 0);
    FVector<const aclTensor*, DIMLIMIT> definedIndices;
    FVector<const aclTensor*, DIMLIMIT> allIndices;
    ConstructStrideAndValue(selfCast, valueSize, stride);
    bool iscontiguousIdx = CheckIfContiguous(indices, definedIndices, allIndices, executor);
    auto strideTensor = executor->ConvertToTensor(stride.data(), stride.size(), DataType::DT_INT32);
    auto valueSizeTensor = executor->ConvertToTensor(valueSize.data(), valueSize.size(), DataType::DT_INT32);

    // Indices Broadcast
    auto ret = IndicesBroadcast(allIndices, executor);   // scalar tensor will be broadcast
    CHECK_RET(ret, nullptr);
    auto allIndicesTensorList = executor->AllocTensorList(allIndices.data(), allIndices.size());

    // LinearIndexV2
    auto linearIndex = l0op::LinearIndexV2(allIndicesTensorList, strideTensor, valueSizeTensor, executor);
    CHECK_RET(linearIndex != nullptr, nullptr);

    // Sort
    auto result = ProcessIndicesArch3510(linearIndex, executor);
    auto sortIdxInt = result.first;
    CHECK_RET(sortIdxInt != nullptr, nullptr);
    auto posIdx = result.second;
    CHECK_RET(posIdx != nullptr, nullptr);
    posIdx = l0op::Cast(posIdx, op::DataType::DT_INT32, executor);
    CHECK_RET(posIdx != nullptr, nullptr);

    // Value Broadcast
    auto valueBroadcast = valuesToBroadcastArch3510(selfCast, definedIndices, valuesCast, masks,
                                                  iscontiguousIdx, executor);
    CHECK_RET(valueBroadcast != nullptr, nullptr);

    // IndexPutWithSort
    const aclTensor* indexPutOpOut;
    aclTensor* out = const_cast<aclTensor*>(selfCast);
    const aclIntArray* maskArray = executor->AllocIntArray(masks.data(), masksNum);
    indexPutOpOut =
        l0op::IndexPutWithSortV2(selfCast, sortIdxInt, posIdx, valueBroadcast, maskArray, accumulate, out, executor);
    CHECK_RET(indexPutOpOut != nullptr, nullptr);
    return indexPutOpOut;
}

static const aclTensor* IndexPutProcessArch3510(aclTensor* selfRef, const aclTensor* selfCast, const aclTensor* valuesCast,
                                              const aclTensorList* indices,
                                              const FVector<const aclTensor*, 8>& allDefinedIndices,
                                              const bool accumulate, FVector<int64_t, 8> masks, int64_t masksNum, const bool isSupportAiCpu,
                                              const bool isNonContiguous, aclOpExecutor* executor)
{
    int64_t deterministicValue = 0;
    rtError_t retRts = rtCtxGetSysParamOpt(SYS_OPT_DETERMINISTIC, &deterministicValue);
    if (retRts != RT_ERROR_NONE) {
        deterministicValue = 0;
    }
    if (isSupportAiCpu) {
        OP_LOGD("Enter Aicpu Process");
        return AicpuProcess(selfRef, selfCast, valuesCast, allDefinedIndices, accumulate, masks, masksNum,
                            executor);
    } else {
        while (masks.size() < selfRef->GetViewShape().GetDimNum()) {
            masks.emplace_back(0);
        }
        for (int i = 0; i < static_cast<int>(masks.size()); i++) {
            OP_LOGD("Process arch3510 masks %d is %ld.", i, masks[i]);
        }
        if (deterministicValue == 0 && !(accumulate && selfRef->GetDataType() == op::DataType::DT_BOOL)) {
            OP_LOGD("Enter IndexPutV2 Process");
            return IndexPutV2Process(selfCast, valuesCast, indices, accumulate, masks, selfRef->GetViewShape().GetDimNum(), 
                                      isNonContiguous, allDefinedIndices, executor);
        } else {
            OP_LOGD("Enter Deterministic Process");
            return DeterministicProcess(selfCast, valuesCast, indices, accumulate, masks,
                                        selfRef->GetViewShape().GetDimNum(), executor);
        }
    }
}

static bool IsNonContiguousScene(const aclTensor* self, const aclTensorList* indices, const aclTensor *value, 
                                  FVector<const aclTensor*, DIMLIMIT>& allDefinedIndices, FVector<int64_t, DIMLIMIT>& masks,
                                  int64_t& indicesNum, int64_t& masksNum, const aclTensor*& selfRefContiguous, const aclTensor*& valuesContiguous, 
                                  aclOpExecutor* executor)
{
    size_t xDim = self->GetViewShape().GetDimNum();
    int64_t indicesSize = static_cast<int64_t>(indices->Size());
    bool overLimit = false;
    if (xDim > DIM_BOUND_NON_CONTIGUOUS || indicesSize > DIM_BOUND_NON_CONTIGUOUS) {
        overLimit = true;
    }
    bool indicesContiguous = false;
    for (int32_t i = 0; i < indicesSize; i++) {
      if ((*indices)[i]) {
        const aclTensor* curIndice = (*indices)[i];
        if (curIndice->GetViewShape().GetShapeSize() != 0) {
          auto indexNonContiguous = executor->CreateView(
              curIndice, curIndice->GetViewShape(), curIndice->GetStorageShape(), curIndice->GetViewStrides(),
              curIndice->GetViewOffset());
          allDefinedIndices.emplace_back(indexNonContiguous);
          masks.emplace_back(1);
          indicesNum += 1;
          masksNum += 1;
          if (IsContiguous(curIndice)) {
              indicesContiguous = true;
          }
        } else {
          masks.emplace_back(0);
          masksNum += 1;
        }
      } else {
          masks.emplace_back(0);
          masksNum += 1;
      }
    }
    bool selfContiguous = IsContiguous(self);
    bool valueContiguous = IsContiguous(value);
    if ((selfContiguous && valueContiguous && indicesContiguous)) {
      return false;
    }
    if ((!selfContiguous || !valueContiguous || !indicesContiguous) && overLimit) {
      return false;
    }

    if (selfContiguous == false) {
      selfRefContiguous =  executor->CreateView(
                self, self->GetViewShape(), self->GetStorageShape(), self->GetViewStrides(),
                self->GetViewOffset());
    }
    if (valueContiguous == false) {
      valuesContiguous =  executor->CreateView(
                value, value->GetViewShape(), value->GetStorageShape(), value->GetViewStrides(),
                value->GetViewOffset());
    }
    return true;
}

static bool IsIndexPutV2Scene(const aclTensor *selfRef, const bool accumulate) 
{
  if (Ops::NN::AclnnUtil::IsRegbase()) {
      int64_t deterministicValue = 0;
      rtError_t retRts = rtCtxGetSysParamOpt(SYS_OPT_DETERMINISTIC, &deterministicValue);
      if (retRts != RT_ERROR_NONE) {
          deterministicValue = 0;
      }
      if (deterministicValue == 0 && !(accumulate && selfRef->GetDataType() == op::DataType::DT_BOOL)) {
          return true;
      }
  }
  return false;
}

aclnnStatus aclnnIndexPutImplGetWorkspaceSize(aclTensor *selfRef,
                                              const aclTensorList *indices,
                                              const aclTensor *values,
                                              const bool accumulate, const bool unsafe,
                                              uint64_t *workspaceSize, aclOpExecutor **executor) {
  OP_CHECK_COMM_INPUT(workspaceSize, executor);
  L2_DFX_PHASE_1(aclnnIndexPutImpl, DFX_IN(selfRef, indices, values, accumulate, unsafe), DFX_OUT(selfRef));
  // 固定写法，创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 固定写法，参数检查
  auto ret = CheckParams(selfRef, indices, values);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);
  // IndexPut算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
  if (selfRef->IsEmpty() || values->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  FVector<int64_t, DIMLIMIT> masks;
  FVector<const aclTensor*, DIMLIMIT> allDefinedIndices;
  int64_t indicesNum = 0;
  int64_t masksNum = 0;
  const aclTensor* selfCast = selfRef;
  const aclTensor* valuesCast = values;
  const aclTensor* selfRefContiguous = selfRef;
  const aclTensor* valuesContiguous = values;
  auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
  bool isIndexPutV2 = IsIndexPutV2Scene(selfRef, accumulate);
  bool nonContiguous = IsNonContiguousScene(selfRef, indices, values, allDefinedIndices, masks, indicesNum, masksNum, 
                        selfRefContiguous, valuesContiguous,  uniqueExecutor.get());    // 是否满足非连续场景条件
  bool isSupportAiCpu = IsAiCPUSupportCheckIndicesArch3510(selfRef, allDefinedIndices, values);
  bool isHighPrecision = accumulate && (selfRef->GetDataType() == op::DataType::DT_FLOAT16 || 
                                        selfRef->GetDataType() == op::DataType::DT_BF16 ||selfRef->GetDataType() == op::DataType::DT_INT8 
                                        || selfRef->GetDataType() == op::DataType::DT_UINT8);
  bool isNonContiguous = isIndexPutV2 && nonContiguous && !isSupportAiCpu && !isHighPrecision;         // 非连续+AiCore+IndexPutV2分支
  OP_LOGI("isNonContiguous is %s", isNonContiguous ? "true" : "false");
  int64_t indicesSize = static_cast<int64_t>(indices->Size());
  int64_t selfSize = selfCast->GetViewShape().GetDimNum();
  FVector<int64_t, DIMLIMIT> stride(selfSize, 1);
  FVector<int64_t, DIMLIMIT> valueSize(selfSize, 0);
  FVector<int64_t, DIMLIMIT> permute(selfSize, 0);
  FVector<int64_t, DIMLIMIT> permuteBack(selfSize, 0);
  FVector<const aclTensor*, DIMLIMIT> definedIndices;
  int64_t headNullNum = 0;
  const aclTensor* indexPutOpOut;
  if (isNonContiguous == false) {
    // 固定写法，将输入selfRef转换成连续的tensor
    selfRefContiguous = l0op::Contiguous(selfRef, uniqueExecutor.get());
    CHECK_RET(selfRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 固定写法，将输入values转换成连续的tensor
    valuesContiguous = l0op::Contiguous(values, uniqueExecutor.get());
    CHECK_RET(valuesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    selfCast = selfRefContiguous;
    valuesCast = valuesContiguous;
    CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(valuesCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    ConstructStrideAndValue(selfCast, valueSize, stride);
    (void)CheckIsDisContinueIdx(indices, headNullNum, definedIndices, uniqueExecutor.get());
    allDefinedIndices.clear();
    for (int64_t i = 0; i < indicesSize; i++) {
      if ((*indices)[i]) {
        if ((*indices)[i]->GetViewShape().GetShapeSize() != 0) {
          auto indicesContiguous = l0op::Contiguous((*indices)[i], uniqueExecutor.get());
          allDefinedIndices.emplace_back(indicesContiguous);
        }
      }
    }
  }
  // Print mask
  OP_LOGD("masksNum is %ld.", masksNum);
  for (int i = 0; i < static_cast<int>(indices->Size()); i++) {
      OP_LOGD("masks %d is %ld.", i, masks[i]);
  }
  if (indicesNum == 0) {
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }
  if (Ops::NN::AclnnUtil::IsRegbase()) {
      int64_t deterministicValue = 0;
      rtError_t retRts = rtCtxGetSysParamOpt(SYS_OPT_DETERMINISTIC, &deterministicValue);
      if (retRts != RT_ERROR_NONE) {
          deterministicValue = 0;
      }
      bool disDeterministicHighPrecision =
          accumulate && deterministicValue == 0 &&
          (selfRef->GetDataType() == op::DataType::DT_FLOAT16 || selfRef->GetDataType() == op::DataType::DT_BF16 ||
           selfRef->GetDataType() == op::DataType::DT_INT8 || selfRef->GetDataType() == op::DataType::DT_UINT8);
      bool DeterministicHighPrecision =
          accumulate && deterministicValue == 1 &&
          (selfRef->GetDataType() == op::DataType::DT_FLOAT16 || selfRef->GetDataType() == op::DataType::DT_BF16);
      if (disDeterministicHighPrecision || DeterministicHighPrecision) {
          OP_LOGD("Begin cast fp16, bf16 to fp32");
          if (selfRef->GetDataType() == op::DataType::DT_FLOAT16 || selfRef->GetDataType() == op::DataType::DT_BF16) {
              selfCast = l0op::Cast(selfRefContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
              valuesCast = l0op::Cast(valuesContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
          }
          OP_LOGD("Begin cast int8, uint8 to int32");
          if (selfRef->GetDataType() == op::DataType::DT_INT8 || selfRef->GetDataType() == op::DataType::DT_UINT8) {
              selfCast = l0op::Cast(selfRefContiguous, op::DataType::DT_INT32, uniqueExecutor.get());
              valuesCast = l0op::Cast(valuesContiguous, op::DataType::DT_INT32, uniqueExecutor.get());
          }
          CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
          CHECK_RET(valuesCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
      }
    indexPutOpOut = IndexPutProcessArch3510(selfRef, selfCast, valuesCast, indices, allDefinedIndices, accumulate, masks, 
                                          masksNum, isSupportAiCpu, isNonContiguous, uniqueExecutor.get());
    CHECK_RET(indexPutOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(indexPutOpOut, selfRef->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(castOut, selfRef, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
  }
  int64_t deterministicValue = 0;
  rtError_t retRts = rtCtxGetSysParamOpt(SYS_OPT_DETERMINISTIC, &deterministicValue);
  if (retRts != RT_ERROR_NONE) {
      deterministicValue = 0;
  }
  bool useIndexPutWithSortSupport = l0op::IsIndexPutWithSortSupport(selfRef, indices, values,
                                                                    deterministicValue, accumulate);
  if (!useIndexPutWithSortSupport){
    if (accumulate && (selfRef->GetDataType() == op::DataType::DT_FLOAT16 ||
        selfRef->GetDataType() == op::DataType::DT_BF16)) {
      OP_LOGD("Begin IndexPutV2 cast fp16 or bf16 to fp32");
      selfCast = l0op::Cast(selfRefContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
      valuesCast = l0op::Cast(valuesContiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
      CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
      CHECK_RET(valuesCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (accumulate && (selfRef->GetDataType() == op::DataType::DT_BOOL)) {
      OP_LOGD("Begin IndexPutV2 cast bool to int8");
      selfCast = l0op::Cast(selfRefContiguous, op::DataType::DT_INT32, uniqueExecutor.get());
      valuesCast = l0op::Cast(valuesContiguous, op::DataType::DT_INT32, uniqueExecutor.get());
      CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
      CHECK_RET(valuesCast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    auto indicesTensorList = uniqueExecutor.get()->AllocTensorList(allDefinedIndices.data(), indicesNum);
    auto maskArray = uniqueExecutor.get()->AllocIntArray(masks.data(), masksNum);
    auto maskTensor = uniqueExecutor.get()->ConvertToTensor(maskArray, op::ToOpDataType(ACL_INT64));
    auto indicesShape = allDefinedIndices[0]->GetViewShape();
    size_t dimNum = indicesShape.GetDimNum();
    FVector<int64_t, DIMLIMIT> indicesvector;
    for (size_t i = 0; i < dimNum; i++) {
      indicesvector.emplace_back(indicesShape.GetDim(i));
    }

    // 调用IndexPut算子kernel
    bool isSupportAiCpu = IsAiCPUSupport(selfRef, allDefinedIndices, values, accumulate, masks);
    if (isSupportAiCpu) {
      aclTensor* out = const_cast<aclTensor*>(selfCast);
      if(deterministicValue != 0) {
          indexPutOpOut = l0op::IndexPutV3(selfCast, indicesTensorList, valuesCast, maskTensor,
                                    accumulate, true, out, uniqueExecutor.get());
      } else {
          indexPutOpOut = l0op::IndexPut(selfCast, indicesTensorList, valuesCast, maskTensor,
                                    accumulate, out, uniqueExecutor.get());
      }
    } else {
      indexPutOpOut = AicoreCompute(selfCast, allDefinedIndices, indicesTensorList, valuesCast,
                                    maskTensor, masks, accumulate, masksNum, indicesNum,
                                    uniqueExecutor.get());
    }
    CHECK_RET(indexPutOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  } else {
    // 调用linearIndexV2时，以尾轴为单位，而不是以数为单位
    indicesSize = static_cast<int64_t>(indices->Size());
    int64_t selfTailSize = indicesSize;
    int64_t selfRefSize = selfRef->GetViewShape().GetDimNum();
    if (selfRefSize > indicesSize) {
        selfTailSize += 1;
    }
    FVector<int64_t, DIMLIMIT> selfTailShape(selfTailSize, 0);
    auto selfShape = selfRef->GetViewShape();
    for (int64_t i = 0; i < indicesSize; i++) {
        selfTailShape[i] = selfShape.GetDim(i);
    }
    if (selfRefSize > indicesSize) {
        selfTailShape[indicesSize] = 1;
    }

    FVector<int64_t, DIMLIMIT> strideTail(selfTailSize, 1);
    FVector<int64_t, DIMLIMIT> valueSizeTail(selfTailSize, 0);

    valueSizeTail[selfTailSize - 1] = selfTailShape[selfTailSize - 1];
    for (int64_t i = selfTailSize - 2; i >= 0; --i) {
        valueSizeTail[i] = selfTailShape[i];
        strideTail[i] = strideTail[i + 1] * valueSizeTail[i + 1];
    }

    int32_t sliceSize = 1;
    if (selfRefSize > indicesSize) {
        for (int64_t i = indicesSize; i < selfRefSize; i++) {
            sliceSize *= selfShape.GetDim(i);
        }
    }
    // values做broadcast操作
    auto valueBroadcast = valuesToBroadcast(indicesSize, selfCast, indices, definedIndices, valuesCast, uniqueExecutor.get());

    auto strideTensor = uniqueExecutor.get()->ConvertToTensor(strideTail.data(), strideTail.size(), DataType::DT_INT32);
    auto valueSizeTensor = uniqueExecutor.get()->ConvertToTensor(valueSizeTail.data(), valueSizeTail.size(), DataType::DT_INT32);
    auto indicesTensorList = uniqueExecutor.get()->AllocTensorList(definedIndices.data(), definedIndices.size());

    auto linearIndex = l0op::LinearIndexV2(indicesTensorList, strideTensor, valueSizeTensor, uniqueExecutor.get());
    CHECK_RET(linearIndex != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = ProcessIndices(linearIndex, uniqueExecutor.get());
    auto sortIdxInt = result.first;
    CHECK_RET(sortIdxInt != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto posIdx = result.second;
    CHECK_RET(posIdx != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(valueBroadcast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    aclTensor* out = const_cast<aclTensor*>(selfCast);
    indexPutOpOut = l0op::IndexPutWithSort(selfCast, sortIdxInt, posIdx, valueBroadcast,
                                           sliceSize, accumulate, out, uniqueExecutor.get());
    CHECK_RET(indexPutOpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }

  // 固定写法，将计算结果转换成输出out的数据类型
  auto castOut = l0op::Cast(indexPutOpOut, selfRef->GetDataType(), uniqueExecutor.get());
  CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
  auto viewCopyResult = l0op::ViewCopy(castOut, selfRef, uniqueExecutor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 固定写法，获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);  // 需要把 uniqueExecutor持有executor转移给executor
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnIndexPutImpl(void *workspace, uint64_t workspaceSize,
                              aclOpExecutor *executor, aclrtStream stream) {
   L2_DFX_PHASE_2(aclnnIndexPutImpl);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif

