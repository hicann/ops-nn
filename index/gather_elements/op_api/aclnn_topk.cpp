/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#include "aclnn_topk.h"
#include "level0/topk.h"
#include "level0/sort.h"
#include "level0/arange.h"
#include "level0/concat.h"
#include "gather_elements.h"
#include "level0/mod.h"
#include "level0/split_v.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "op_api/aclnn_util.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/small_vector.h"
#include "opdev/platform.h"

#include <cstdint>
#include <cmath>

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

const int64_t MAX_AICORE_CALC_INPUTSIZE = 32768;
const int64_t PARALLEL_K = 32;
const int64_t MAX_AICORE_CALC_DIM = 8;
const int64_t CONCAT_MAX = 512; // Concat能处理的最大Tensor
const int64_t SORT_WITH_INDEX_THRESHOLD = 2000; // TopK后调用SortWithIndex的阈值
const float SORT_AND_TOP_K_THRESHOLD = 0.5; // 走先排序后取前K个值k/n的比值的阈值

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_DOUBLE};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_WITH_BF16 = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
    op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_WITH_BF16_AND_UINT = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
    op::DataType::DT_BF16, op::DataType::DT_UINT64, op::DataType::DT_UINT16, op::DataType::DT_UINT32};
static const std::initializer_list<DataType>& GetDtypeSupportList() {
  if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201) {
    return DTYPE_SUPPORT_LIST_WITH_BF16;
  }
  if (Ops::NN::AclnnUtil::IsRegbase()) {
    return DTYPE_SUPPORT_LIST_WITH_BF16_AND_UINT;
  }
  return DTYPE_SUPPORT_LIST;
}

static bool CheckNotNull(const aclTensor *self, const aclTensor *values, const aclTensor *indices) {
  OP_CHECK_NULL(self, return false);
  OP_CHECK_NULL(values, return false);
  OP_CHECK_NULL(indices, return false);
  return true;
}

static int64_t MakeWrapDim(int64_t dim, int64_t dimPostExpr) {
  // 支持0维tensor
  if (dimPostExpr <= 0) {
    dimPostExpr = 1;
  }
  if (dim < 0) {
    dim += dimPostExpr;
  }
  return dim;
}

static bool CheckParamValid(const aclTensor *self, int64_t k, int64_t dim) {
  // 检查参数dim是否合法
  auto inputShape = self->GetViewShape();
  int64_t tmpDim = static_cast<int64_t>(inputShape.GetDimNum());
  if (tmpDim == static_cast<int64_t>(0) && dim != static_cast<int64_t>(0) && dim != static_cast<int64_t>(-1)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dimension out of range (expected to be in range of [-1, 0], but got %ld)",
            dim);
    return false;
  } else if (tmpDim > 0 && (dim < -tmpDim || dim >= tmpDim)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dimension out of range (expected to be in range of [-%ld, %ld],"
            "but got %ld)", tmpDim, tmpDim - 1, dim);
    return false;
  }

  // 检查参数k是否合法
  int64_t positiveDim = MakeWrapDim(dim, tmpDim);
  int64_t tmpK = (tmpDim > 0) ? inputShape.GetDim(positiveDim) : 1;
  if (k < 0 || k > tmpK) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Selected index k out of range (max num of self.size(%ld) is %ld,"
            "but k is %ld)", tmpDim, tmpK, k);
    return false;
  }
  return true;
}

static bool CheckDtypeValid(const aclTensor *self, const aclTensor *values, const aclTensor *indices) {
  auto supportList = GetDtypeSupportList();
  // 检查self的数据类型是否在topk算子的支持列表内
  OP_CHECK_DTYPE_NOT_SUPPORT(self, supportList, return false);
  OP_CHECK_DTYPE_NOT_MATCH(values, self->GetDataType(), return false);
  OP_CHECK_DTYPE_NOT_MATCH(indices, op::DataType::DT_INT64, return false);
  return true;
}

static bool CheckShape(const aclTensor *self) {
  OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
  return true;
}

static aclnnStatus CheckParams(const aclTensor *self, int64_t k, int64_t dim,
                               const aclTensor *values, const aclTensor *indices) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(self, values, indices), ACLNN_ERR_PARAM_NULLPTR);

  // 2. 检查参数k和dim是否合法
  CHECK_RET(CheckParamValid(self, k, dim), ACLNN_ERR_PARAM_INVALID);

  // 3. 检查self、values和indices的数据类型是否合法
  CHECK_RET(CheckDtypeValid(self, values, indices), ACLNN_ERR_PARAM_INVALID);

  // 4. 查输入tensor的shape是否为异常
  CHECK_RET(CheckShape(self), ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

static const aclTensor *TopkAdaptInputZeroDimTensor(const aclTensor *self, int64_t dimNum, aclOpExecutor *executor) {
  if (dimNum != 0) {
    return self;
  }
  int64_t selfShapeValue[1] = {1};
  aclIntArray *selfShape = executor->AllocIntArray(selfShapeValue, 1);
  auto selfReshape = l0op::Reshape(self, selfShape, executor);
  return selfReshape;
}

static bool CheckCalcInAiCore(const aclTensor *self, int64_t k) {
  auto inputShape = self->GetViewShape();
  int64_t tmpDim = static_cast<int64_t>(inputShape.GetDimNum());
  int64_t inputSize = 1;
  for (int64_t i = 0; i < tmpDim; i++) {
    inputSize *= inputShape.GetDim(i);
  }
  if (inputSize > MAX_AICORE_CALC_INPUTSIZE) {
    return k >= MAX_AICORE_CALC_DIM;
  }
  return true;
}

static const aclTensor *TopkAdaptGeCastTensor(const aclTensor *self, const aclTensor *value, int64_t k,
                                              op::DataType dataType, aclOpExecutor *executor) {
  NpuArch version = GetCurrentPlatformInfo().GetCurNpuArch();
  if (version == NpuArch::DAV_1001 && CheckCalcInAiCore(self, k) && self->GetDataType() == op::DataType::DT_FLOAT) {
    return l0op::Cast(value, dataType, executor);
  }
  return value;
}

static aclIntArray *GetDimTransposeArray(int64_t dimNum, int64_t lastDim, int64_t positiveDim,
                                         aclOpExecutor *executor) {
  std::vector<int64_t> perm(dimNum, 0);
  for (int64_t i = 0; i < dimNum; i++) {
    perm[i] = i;
  }
  std::swap(perm[positiveDim], perm[lastDim]);
  return executor->AllocIntArray(perm.data(), dimNum);
}

static bool CanDealWith(const aclTensor *self, int64_t k)
{
    auto inputShape = self->GetViewShape();
    int64_t tmpDim = static_cast<int64_t>(inputShape.GetDimNum());
    int64_t inputSize = 1;
    for (int64_t i = 0; i < tmpDim; i++) {
      inputSize *= inputShape.GetDim(i);
    }
    int64_t batchNum = inputSize / k;

    if (inputSize <= INT32_MAX) {
        return true;
    }
    if (k <= INT32_MAX) {
        int64_t int32MaxBatchNum = INT32_MAX / k;
        int64_t int32Num = batchNum / int32MaxBatchNum;
        if (batchNum % int32MaxBatchNum != 0) {
            int32Num += 1;
        }
        return int32Num <= CONCAT_MAX;
    } else {
        return batchNum <= CONCAT_MAX;
    }
}

static bool IsTopKCopy(const aclTensor *self, int64_t k, int64_t sortDimValue, bool sorted)
{
  // 如果不是950,不copy
  if (!Ops::NN::AclnnUtil::IsRegbase()) {
    return false;
  }
  // 如果排序轴不等于k，不Copy
  if (sortDimValue != k) {
    return false;
  }
  // 如果索引不能生成，不Copy
  if (!CanDealWith(self, k)) {
    return false;
  }
  // k等于1或者不排序的时候，直接Copy
  return k == 1 || !sorted;
}

static aclnnStatus TopKCopy(const aclTensor *self, int64_t k, aclTensor *values, aclTensor *indices, aclOpExecutor *executor)
{
    auto viewCopyValuesResult = l0op::ViewCopy(self, values, executor);
    CHECK_RET(viewCopyValuesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto inputShape = self->GetViewShape();
    int64_t tmpDim = static_cast<int64_t>(inputShape.GetDimNum());
    int64_t inputSize = 1;
    for (int64_t i = 0; i < tmpDim; i++) {
      inputSize *= inputShape.GetDim(i);
    }

    auto start = executor->AllocScalar(0);
    auto step = executor->AllocScalar(1);

    auto kScalar = executor->AllocScalar(k);
    auto kTensor = executor->ConvertToTensor(kScalar, kScalar->GetDataType());

    const aclTensor* indiceRet = nullptr;
    int64_t batchNum = inputSize / k;
    
    /**
     * 索引生成方案总体上有2个:
     * 1、生成[0,inputSize)的索引，然后对K取余，可以得到每行的索引
     * 2、由于Mod当前不支持int64，因此对于inputSize > int32_max时，不能用Mod，因此使用Arange+ConcatD进行处理，此时又分2种情况：
     *   （1）K小于int32_max,此时int32_max能够生成N行大小为K的索引，那么要生成所有的索引就需要Batch轴 / N次循环生成，最后ConcatD
     *   （2）k大于int32_max,可以生成[0,inputSize)，然后ConcatD Batch轴个前面的索引即可
     */
    if (inputSize <= INT32_MAX) {
        auto end = executor->AllocScalar(inputSize);
        auto arangeIndiceRet = l0op::Arange(start, end, step, indices, false, executor);
        auto arangeCasted = l0op::Cast(arangeIndiceRet, op::DataType::DT_INT32, executor);
        auto kCasted = l0op::Cast(kTensor, op::DataType::DT_INT32, executor);
        indiceRet = l0op::Mod(arangeCasted, kCasted, executor);
    } else if (k <= INT32_MAX) {
        int64_t int32MaxBatchNum = INT32_MAX / k;
        auto end = executor->AllocScalar(int32MaxBatchNum * k);
        int64_t int32Num = batchNum / int32MaxBatchNum;
        auto arangeIndiceRet = l0op::Arange(start, end, step, indices, false, executor);
        auto arangeCasted = l0op::Cast(arangeIndiceRet, op::DataType::DT_INT32, executor);
        auto kCasted = l0op::Cast(kTensor, op::DataType::DT_INT32, executor);
        auto singleIndiceRet = l0op::Mod(arangeCasted, kCasted, executor);
        op::FVector<const aclTensor*> tensorFVector;
        for (int16_t i = 0; i < int32Num; i++) {
            tensorFVector.emplace_back(singleIndiceRet);
        }
        int64_t remainIn32Num = batchNum % int32MaxBatchNum;
        if (remainIn32Num != 0) {
            end = executor->AllocScalar(remainIn32Num * k);
            arangeIndiceRet = l0op::Arange(start, end, step, indices, false, executor);
            arangeCasted = l0op::Cast(arangeIndiceRet, op::DataType::DT_INT32, executor);
            singleIndiceRet = l0op::Mod(arangeCasted, kCasted, executor);
            tensorFVector.emplace_back(singleIndiceRet);
        }
        auto tensorList = executor->AllocTensorList(tensorFVector.data(), tensorFVector.size());
        indiceRet = l0op::ConcatD(tensorList, 0, executor);
    } else {
        auto end = executor->AllocScalar(inputSize);
        auto arangeIndiceRet = l0op::Arange(start, end, step, indices, false, executor);
        op::FVector<const aclTensor*> tensorFVector;
        for (int16_t i = 0; i < batchNum; i++) {
          tensorFVector.emplace_back(arangeIndiceRet);
        }
        auto tensorList = executor->AllocTensorList(tensorFVector.data(), tensorFVector.size());
        indiceRet = l0op::ConcatD(tensorList, 0, executor);
    }

    auto indicesRetCast = l0op::Cast(indiceRet, op::DataType::DT_INT64, executor);
    auto viewCopyIndicesResult = l0op::ViewCopy(indicesRetCast, indices, executor);
    CHECK_RET(viewCopyIndicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static const aclTensor *TopKSplit(aclTensor *x, int64_t k, int64_t positiveDim, int64_t sortDimValue, aclOpExecutor *executor) {
    int64_t numSplit = 2;
    op::FVector<int64_t> splitVector(numSplit, k);
    splitVector[1] = sortDimValue - k;
    aclIntArray *splitSize = executor->AllocIntArray(splitVector.data(), splitVector.size());
    if (splitSize == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SplitSize is nullptr, please check");
        return nullptr;
    }
    auto splitRes = l0op::SplitV(x, splitSize, positiveDim, executor);
    if ((splitRes == nullptr)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "splitRes is nullptr, please check");
        return nullptr;
    }
    return (*splitRes)[0];
}

static std::tuple<const aclTensor*, const aclTensor*> SortAndTopK(const aclTensor *self, int64_t k, int64_t lastDim, int64_t sortDimValue, bool largest,
                                                  op::DataType indicesType, aclOpExecutor *executor)
{
    auto sortOut = l0op::Sort(self, -1, largest, true, indicesType, executor);
    if (std::get<0>(sortOut) == nullptr ||  std::get<1>(sortOut) == nullptr) {
      OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Sort result is null, Please check Sort");
    }
    
    const aclTensor* values = TopKSplit(std::get<0>(sortOut), k, lastDim, sortDimValue, executor);
    const aclTensor* indices = TopKSplit(std::get<1>(sortOut), k, lastDim, sortDimValue, executor);
    return std::tie(values, indices);
}

static bool indicesOutNeedsCast(int64_t k, bool sorted, bool isHasCasted)
{
  if (isHasCasted) {
    return false;
  }
  if ((Ops::NN::AclnnUtil::IsRegbase()) &&
      ((k <= SORT_WITH_INDEX_THRESHOLD) || (sorted == false))) {
    return false;
  }
  return true;
}

/**
 * 判断是否走先排序后取前K个值
 *
 * @param sorted 是否排序
 * @param k TopK K的值
 * @param sortDimValue 排序轴的大小 
 * @return 是否先排序
 */
static bool IsSortAndTopK(bool sorted, int64_t k, int64_t sortDimValue) {
    // 如果不是950,不走该逻辑
    if (!Ops::NN::AclnnUtil::IsRegbase()) {
      return false;
    }
    // 如果不需要排序，不走先排序后取前K个数
    if (!sorted) {
      return false;
    }
    // 如果k / n 小于0.5，不走先排序后取前K个数
    if (k < SORT_AND_TOP_K_THRESHOLD * sortDimValue) {
      return false;
    }
    // 如果K小于等于2000，不走先排序后取前K个数
    return k > SORT_WITH_INDEX_THRESHOLD;
}

aclnnStatus aclnnTopkGetWorkspaceSize(const aclTensor *self, int64_t k, int64_t dim, bool largest, bool sorted,
                                      aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *workspaceSize,
                                      aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnTopk, DFX_IN(self, k, dim, largest, sorted), DFX_OUT(valuesOut, indicesOut));

   // 创建OpExecutor
  auto uniqueExecutor = CREATE_EXECUTOR();
  CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

  // 参数检查
  auto ret = CheckParams(self, k, dim, valuesOut, indicesOut);
  CHECK_RET(ret == ACLNN_SUCCESS, ret);

  // 支持空tensor
  if (self->IsEmpty() || valuesOut->IsEmpty() || indicesOut->IsEmpty()) {
    // 根据实际支持情况补充
    *workspaceSize = static_cast<uint64_t>(0);
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }

  // 固定写法，将输入self转换成连续的tensor
  auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
  CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

  int64_t dimNum = static_cast<int64_t>(selfContiguous->GetViewShape().GetDimNum());
  int64_t positiveDim = MakeWrapDim(dim, dimNum);
  int64_t lastDim = MakeWrapDim(static_cast<int64_t>(-1), dimNum);

  // 获取排序轴的数据个数，用于判断排序轴是否于K相等
  int64_t sortDimValue = static_cast<int64_t>(self->GetViewShape().GetDim(positiveDim));

  auto selfReshape = TopkAdaptInputZeroDimTensor(selfContiguous, dimNum, uniqueExecutor.get());
  CHECK_RET(selfReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

  const aclTensor *indicesCastInt32 = nullptr;
  const aclTensor *valuesTopkOut = nullptr;

  // 在910上,当输入fp32时,路径3的ge侧会插入cast,转换到fp16.此处修改与路径3保持一致。
  auto selfCast = TopkAdaptGeCastTensor(selfReshape, selfReshape, k, op::DataType::DT_FLOAT16, uniqueExecutor.get());
  CHECK_RET(selfCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

  auto indicesDType = indicesOut->GetDataType();
  bool isHasCasted = false;
  if (IsTopKCopy(selfCast, k, sortDimValue, sorted)) {
    TopKCopy(selfCast, k, valuesOut, indicesOut, uniqueExecutor.get());
    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }
  if (positiveDim != lastDim) {
    aclIntArray *axes = GetDimTransposeArray(dimNum, lastDim, positiveDim, uniqueExecutor.get());
    CHECK_RET(axes != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 对self进行transpose
    auto selfTranspose = l0op::Transpose(selfCast, axes, uniqueExecutor.get());
    CHECK_RET(selfTranspose != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 进行top计算
    std::tuple<const aclTensor*, const aclTensor*> topkOut(nullptr, nullptr);
    if (sortDimValue == k && Ops::NN::AclnnUtil::IsRegbase() && CanDealWith(selfTranspose, k)) {
      topkOut = l0op::Sort(selfTranspose, -1, largest, true, indicesDType, uniqueExecutor.get());
      isHasCasted = true;
    } else if (IsSortAndTopK(sorted, k, sortDimValue)) {
      topkOut = SortAndTopK(selfTranspose, k, lastDim, sortDimValue, largest, indicesDType, uniqueExecutor.get());
      isHasCasted = true;
    } else {
      topkOut = l0op::Topk(selfTranspose, k, lastDim, largest, sorted, indicesDType, uniqueExecutor.get());
    }

    CHECK_RET(std::get<0>(topkOut) != nullptr && std::get<1>(topkOut) != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将结果values_transpose进行transpose，转换成正确的shape
    valuesTopkOut = l0op::Transpose(std::get<0>(topkOut), axes, uniqueExecutor.get());

    // 将结果indices_transpose进行transpose，转换成正确的shape
    indicesCastInt32 = l0op::Transpose(std::get<1>(topkOut), axes, uniqueExecutor.get());
  } else {
    if (k > 0 && k < MAX_AICORE_CALC_DIM && !Ops::NN::AclnnUtil::IsRegbase()) {
      int64_t kFirst = std::min(PARALLEL_K,selfContiguous->GetViewShape().GetDim(positiveDim));
      auto topkOutFirst = l0op::Topk(selfCast, kFirst, positiveDim, largest, sorted, indicesDType, uniqueExecutor.get());

      valuesTopkOut = std::get<0>(topkOutFirst);
      aclTensor *indicesCastFirst = std::get<1>(topkOutFirst);
      auto topkOut = l0op::Topk(valuesTopkOut, k, positiveDim, largest, sorted, indicesDType, uniqueExecutor.get());

      valuesTopkOut = std::get<0>(topkOut);
      aclTensor *indicesCast = std::get<1>(topkOut);
      
      indicesCastInt32 = l0op::GatherElements(indicesCastFirst, positiveDim, indicesCast, uniqueExecutor.get());
    } else {
      std::tuple<const aclTensor*, const aclTensor*> topkOut(nullptr, nullptr);
      if (sortDimValue == k && Ops::NN::AclnnUtil::IsRegbase() && CanDealWith(selfCast, k)) {
        topkOut = l0op::Sort(selfCast, -1, largest, true, indicesDType, uniqueExecutor.get());
        isHasCasted = true;
      } else if (IsSortAndTopK(sorted, k, sortDimValue)) {
        topkOut = SortAndTopK(selfCast, k, positiveDim, sortDimValue, largest, indicesDType, uniqueExecutor.get());
        isHasCasted = true;
      } else {
        topkOut = l0op::Topk(selfCast, k, positiveDim, largest, sorted, indicesDType, uniqueExecutor.get());
      }
      valuesTopkOut = std::get<0>(topkOut);
      indicesCastInt32 = std::get<1>(topkOut);
    }
  }
  CHECK_RET(valuesTopkOut != nullptr && indicesCastInt32!= nullptr, ACLNN_ERR_INNER_NULLPTR);

  CHECK_RET(CheckReduceOutShape(valuesOut, valuesTopkOut), ACLNN_ERR_PARAM_INVALID);
  CHECK_RET(CheckReduceOutShape(indicesOut, indicesCastInt32), ACLNN_ERR_PARAM_INVALID);

  // 在910上，输入fp32转换到fp16计算完后需要重新转换到fp32。
  auto valuesCast = TopkAdaptGeCastTensor(selfReshape, valuesTopkOut, k, op::DataType::DT_FLOAT, uniqueExecutor.get());
  CHECK_RET(valuesCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

  // 将valuesCast结果拷贝到values上
  auto viewCopyValuesResult = l0op::ViewCopy(valuesCast, valuesOut, uniqueExecutor.get());
  CHECK_RET(viewCopyValuesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

  if (!indicesOutNeedsCast(k, sorted, isHasCasted)) {
    auto viewCopyIndicesResult = l0op::ViewCopy(indicesCastInt32, indicesOut, uniqueExecutor.get());
    CHECK_RET(viewCopyIndicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  } else {
    // 将结果indices_cast_int32进行cast，转换成int64类型
    auto indicesCastInt64 = l0op::Cast(indicesCastInt32, op::DataType::DT_INT64, uniqueExecutor.get());
    CHECK_RET(indicesCastInt64 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 将indices_cast_int64结果拷贝到values上
    auto viewCopyIndicesResult = l0op::ViewCopy(indicesCastInt64, indicesOut, uniqueExecutor.get());
    CHECK_RET(viewCopyIndicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  }
  // 获取计算过程中需要使用的workspace大小
  *workspaceSize = uniqueExecutor->GetWorkspaceSize();
  uniqueExecutor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnTopk(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnTopk);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
