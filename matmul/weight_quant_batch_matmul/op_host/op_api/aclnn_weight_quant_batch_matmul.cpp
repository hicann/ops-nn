/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_weight_quant_batch_matmul.h"

#include "op_api/op_api_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

#include "level0/add.h"
#include "level0/muls.h"
#include "level0/mul.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/dot.h"
#include "level0/fill.h"
#include "matmul/mat_mul_v3/op_host/op_api/matmul.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transdata.h"
#include "level0/unsqueeze.h"
#include "weightQuantBatchmatmul.h"

#include "matmul/common/op_host/op_api/matmul_util.h"
#include "matmul/common/op_host/op_api/batch_matmul_util.h"

using namespace std;
using namespace op;
#ifdef __cplusplus
extern "C" {
#endif
static const std::initializer_list<op::DataType> x1_SUPPORT_LIST = {DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> x2_SUPPORT_LIST = {DataType::DT_INT8};
static const std::initializer_list<op::DataType> DIAG_SUPPORT_LIST = {DataType::DT_INT8};
static const std::initializer_list<op::DataType> BIAS_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> MULADD_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> DEQSCALE_DTYPE_SUPPORT_LIST = {DataType::DT_UINT64};
static const std::initializer_list<op::DataType> DEQOFFSET_DTYPE_SUPPORT_LIST = {DataType::DT_INT32};
static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16};
static const int64_t M_THRESHOLD = 64;
static const int64_t SUPPORT_DIMS = 2;

inline static bool SelectFixPipe (int64_t M) {
    return M <= M_THRESHOLD;
}

// bias can be nullptr
inline static bool CheckNotNull(const aclTensor *x1, const aclTensor *x2,
                                const aclTensor *addOffset, const aclTensor *mulScale,
                                const aclTensor *diagonalMatrix, const aclTensor *deqOffset,
                                const aclTensor *deqScale, const aclTensor *out, bool transposeX1)
{
  (void) addOffset;
  (void) mulScale;
  int64_t dimTensor1 = x1->GetViewShape().GetDimNum();
  int64_t M = transposeX1 ?  x1->GetViewShape().GetDim(dimTensor1 - 1) : x1->GetViewShape().GetDim(dimTensor1 - 2);
  if (SelectFixPipe(M)) {
    OP_CHECK_NULL(diagonalMatrix, return false);
    OP_CHECK_NULL(deqOffset, return false);
    OP_CHECK_NULL(deqScale, return false);
  }
  OP_CHECK_NULL(x1, return false);
  OP_CHECK_NULL(x2, return false);
  OP_CHECK_NULL(out, return false);
  return true;
}

inline static bool CheckDtypeValid(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                   const aclTensor *addOffset, const aclTensor *mulScale,
                                   const aclTensor *diagonalMatrix, const aclTensor *deqOffset,
                                   const aclTensor *deqScale, const aclTensor *out, bool transposeX1) {
  int64_t dimTensor1 = x1->GetViewShape().GetDimNum();
  int64_t M = transposeX1 ?  x1->GetViewShape().GetDim(dimTensor1 - 1) : x1->GetViewShape().GetDim(dimTensor1 - 2);
  OP_CHECK_DTYPE_NOT_SUPPORT(x1, x1_SUPPORT_LIST, return false);
  OP_CHECK_DTYPE_NOT_SUPPORT(x2, x2_SUPPORT_LIST, return false);
  if (bias != nullptr){
    OP_CHECK_DTYPE_NOT_SUPPORT(bias, BIAS_DTYPE_SUPPORT_LIST, return false);
  }

  if (SelectFixPipe(M)) {
    OP_CHECK_DTYPE_NOT_SUPPORT(diagonalMatrix, DIAG_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(deqOffset, DEQOFFSET_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(deqScale, DEQSCALE_DTYPE_SUPPORT_LIST, return false);
  } else {
    if (addOffset != nullptr) {
      OP_CHECK_DTYPE_NOT_SUPPORT(addOffset, MULADD_DTYPE_SUPPORT_LIST, return false);
    }
    if (mulScale != nullptr) {
      OP_CHECK_DTYPE_NOT_SUPPORT(mulScale, MULADD_DTYPE_SUPPORT_LIST, return false);
    }
  }
  OP_CHECK_DTYPE_NOT_SUPPORT(out, OUT_DTYPE_SUPPORT_LIST, return false);
  return true;
}

static bool IsFloatEqual(float f1, float f2) {
  return std::abs(f1 - f2) <= std::numeric_limits<float>::epsilon();
}

static bool CheckOffsetAndScale(const aclTensor *antiQuantTensor, bool transposeX2, int64_t N) {
  int64_t dimAntiQuantTensor = antiQuantTensor->GetViewShape().GetDimNum();
  int64_t getCalDim = transposeX2 ? -2 : -1;
  // dim 为 1 时，不能取 -2 维度，单独处理
  if (dimAntiQuantTensor == 1) {
    if (transposeX2 && (antiQuantTensor->GetViewShape().GetDim(0) != 1)) {
      // 转置 && dim == 1, 值只能为1
      return false;
    } else if (!transposeX2 && (antiQuantTensor->GetViewShape().GetDim(0) != 1) &&
                        (antiQuantTensor->GetViewShape().GetDim(0) != N)) {
      // 非转置 && dim == 1, 值只能为1 或者 N
      return false;
    }
  } else if (dimAntiQuantTensor > 1) {
    if ((antiQuantTensor->GetViewShape().GetDim(dimAntiQuantTensor + getCalDim) != 1) &&
        (antiQuantTensor->GetViewShape().GetDim(dimAntiQuantTensor + getCalDim) != N)) {
      // broadcast维度, 值只能为1 或者 N
      return false;
    }
  }

  return true;
}

static bool CheckAntiQuantTensorValid(const aclTensor *addOffset, const aclTensor *mulScale,
                                      const aclTensor *diagonalMatrix, const aclTensor *deqOffset,
                                      const aclTensor *deqScale, bool transposeX2,
                                      int64_t M, int64_t N) {
  if (SelectFixPipe(M)) {
    if (!CheckOffsetAndScale(deqOffset, false, N)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "fixpipe Offset's broad cast dim is not equal to one or N, can't broadcast");
      return false;
    }
    if (!CheckOffsetAndScale(deqScale, false, N)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "fixpipe Scale's broad cast dim is not equal to one or N, can't broadcast");
      return false;
    }
    // diagmatrix check shape need equal to [32, 32], length = 2
    size_t diagDim = 2;
    int64_t diagSize = 32;
    if ((diagonalMatrix->GetViewShape().GetDimNum() != diagDim) ||
        (diagonalMatrix->GetViewShape().GetDim(0) != diagSize) ||
        (diagonalMatrix->GetViewShape().GetDim(1) != diagSize)) {
         OP_LOGE(ACLNN_ERR_PARAM_INVALID, "diagonalMatrix shape is not equal to [32, 32]");
         return false;
    }
  } else {
    if ((addOffset != nullptr) && !CheckOffsetAndScale(addOffset, transposeX2, N)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "single op Offset's broad cast dim is not equal to one or N, can't broadcast");
      return false;
    }
    if ((mulScale != nullptr) && !CheckOffsetAndScale(mulScale, transposeX2, N)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "single op Scale's broad cast dim is not equal to one or N, can't broadcast");
      return false;
    }
  }
  return true;
}

static bool CheckShapeValid(const aclTensor* x1, const aclTensor* x2, const aclTensor *bias,
                            const aclTensor *addOffset, const aclTensor *mulScale, const aclTensor *diagonalMatrix,
                            const aclTensor *deqOffset, const aclTensor *deqScale, const aclTensor *out,
                            bool transposeX1, bool transposeX2)
{
  op::Shape x1Shape = x1->GetViewShape();
  op::Shape x2Shape = x2->GetViewShape();
  op::Shape outShape = out->GetViewShape();
  int64_t dimTensor1 = x1Shape.GetDimNum();
  int64_t dimTensor2 = x2Shape.GetDimNum();
  int64_t dimTensorout = outShape.GetDimNum();

  // x1 and x2 shape's length must be same or larger than 2
  int64_t minDim = 2;
  if ((dimTensor1 < minDim) || (dimTensor2 < minDim)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dims of the two inputs should be same or larger than 2,"
            "now the shapes are  %s, %s", op::ToString(x1Shape).GetString(), op::ToString(x2Shape).GetString());
    return false;
  }

  int64_t M = transposeX1 ?  x1->GetViewShape().GetDim(dimTensor1 - 1) : x1->GetViewShape().GetDim(dimTensor1 - 2);
  int64_t x1KDim = transposeX1 ?  x1->GetViewShape().GetDim(dimTensor1 - 2) : x1->GetViewShape().GetDim(dimTensor1 - 1);
  int64_t x2KDim = transposeX2 ?  x2->GetViewShape().GetDim(dimTensor2 - 1) : x2->GetViewShape().GetDim(dimTensor2 - 2);
  int64_t N = transposeX2 ?  x2->GetViewShape().GetDim(dimTensor2 - 2) : x2->GetViewShape().GetDim(dimTensor2 - 1);
  // check addOffset mulScale deqOffset deqScale
  if (!CheckAntiQuantTensorValid(addOffset, mulScale, diagonalMatrix, deqOffset, deqScale,
                                 transposeX2, M, N)) {
    return false;
  }

  //check bias shape last dim = 1 or N
  if (bias != nullptr) {
    auto dimTensorBias = bias->GetViewShape().GetDimNum();
    if ((bias->GetViewShape().GetDim(dimTensorBias - 1) != 1) &&
        (bias->GetViewShape().GetDim(dimTensorBias - 1) != N)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Bias shape last dim need equal to 1 or N");
      return false;
    }
  }
  // 超出最大支持维度返回
  OP_CHECK_MAX_DIM(x1, MAX_SUPPORT_DIMS_NUMS, return false);
  OP_CHECK_MAX_DIM(x2, MAX_SUPPORT_DIMS_NUMS, return false);

  if (x1KDim != x2KDim) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The k-axis of the two inputs are different %s, %s",
            op::ToString(x1Shape).GetString(), op::ToString(x2Shape).GetString());
    return false;
  }
  // 2是矩阵的维度
  for (int i = 0; i < dimTensorout - 2; i++) {
    if (x1Shape.GetDim(i) != outShape.GetDim(i)) {
      OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The shape above 2 dims of input %s is not match output %s",
              op::ToString(x1Shape).GetString(), op::ToString(outShape).GetString());
      return false;
    }
  }
  // 2 or 1 is axis offset
  if (M != outShape.GetDim(dimTensorout - 2) || N != outShape.GetDim(dimTensorout - 1)) {
    OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The m-axis or n-axis of inputs %s, %s are not match output %s",
            op::ToString(x1Shape).GetString(), op::ToString(x2Shape).GetString(), op::ToString(outShape).GetString());
    return false;
  }
  return true;
}

inline static aclnnStatus CheckParam(const aclTensor *x1, const aclTensor *x2, const aclTensor *bias,
                                     const aclTensor *addOffset, const aclTensor *mulScale,
                                     const aclTensor *diagonalMatrix, const aclTensor *deqOffset,
                                     const aclTensor *deqScale, const aclTensor *out, bool transposeX1, bool transposeX2) {
  // 1. 检查参数是否为空指针
  CHECK_RET(CheckNotNull(x1, x2, addOffset, mulScale, diagonalMatrix, deqOffset, deqScale, out, transposeX1), ACLNN_ERR_INNER_NULLPTR);
  // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
  CHECK_RET(CheckDtypeValid(x1, x2, bias, addOffset, mulScale, diagonalMatrix, deqOffset, deqScale, out, transposeX1), ACLNN_ERR_PARAM_INVALID);
  // 3. 检查Shape是否支持
  CHECK_RET(CheckShapeValid(x1, x2, bias, addOffset, mulScale, diagonalMatrix, deqOffset, deqScale, out, transposeX1, transposeX2), ACLNN_ERR_PARAM_INVALID);
  return ACLNN_SUCCESS;
}

static const aclTensor* ProcessEmptyTensor(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                           const aclTensor* out, aclOpExecutor* executor)
{
  (void) x2;
  // 获取shape信息
  op::Shape outShape = out->GetViewShape();
  auto output = executor->AllocTensor(outShape, x1->GetDataType());
  if (output->IsEmpty()) {
    OP_LOGI("Returning an empty tensor without actually doing calculation");
    return output;
  }
  FVector<int64_t> fillShape =  Ops::NN::GetShape(output);
  const aclTensor* dims = executor->ConvertToTensor(fillShape.data(), fillShape.size(), op::DataType::DT_INT64);
  aclIntArray* shapeArray = executor->AllocIntArray(fillShape.data(), fillShape.size());
  const aclScalar* valueScalar = executor->AllocScalar(0);
  const aclTensor* valueTensor = executor->ConvertToTensor(valueScalar, out->GetDataType());
  auto fillTensor = l0op::Fill(dims, valueTensor, shapeArray, executor);
  // bias
  if (bias != nullptr) {
      bias = l0op::Cast(bias, DataType::DT_FLOAT16, executor);
      CHECK_RET(bias != nullptr, nullptr);
      fillTensor = l0op::Add(fillTensor, bias, executor);
      CHECK_RET(fillTensor != nullptr, nullptr);
      fillTensor = l0op::Cast(fillTensor, out->GetDataType(), executor);
      CHECK_RET(fillTensor != nullptr, nullptr);
  }
  return fillTensor;
}


aclnnStatus InputPreProcess(const aclTensor *inputTensor, aclOpExecutor *executor) {
  if (inputTensor != nullptr) {
    inputTensor = l0op::Contiguous(inputTensor, executor);
    OP_CHECK(inputTensor != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input perprocess failed, contiguous return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);
    inputTensor = l0op::ReFormat(inputTensor, op::Format::FORMAT_ND);
    OP_CHECK(inputTensor != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input perprocess failed, reformat return nullptr."),
             return ACLNN_ERR_INNER_NULLPTR);
  }

  return ACLNN_SUCCESS;
}

const aclTensor *WeightQuantMatmulSingleOp(const aclTensor *x1, const aclTensor *x2, const aclTensor *addOffset,
                                            const aclTensor *mulScale, const aclTensor *bias,
                                            float antiquantScale, float antiquantOffset,
                                            bool transposeX1, bool transposeX2, aclTensor *out,
                                            aclOpExecutor *executor) {
    // x1 * ((cast(x2) + antiquantOffset) * antiquantScale + addOffset) * mulScale + bias
    //   fp16      fp32              fp32             fp32             fp16        fp16     fp32
    const aclTensor *matmulOut;
    // cast
    auto antiquantX2 = l0op::Cast(x2, DataType::DT_FLOAT, executor);
    CHECK_RET(antiquantX2 != nullptr, nullptr);
    // add
    if (!IsFloatEqual(antiquantOffset, 0.0f)) {
        aclScalar* antiquantOffsetScalar = executor->AllocScalar(antiquantOffset);
        auto antiquantOffsetTensor = executor->ConvertToTensor(antiquantOffsetScalar, DataType::DT_FLOAT);
        CHECK_RET(antiquantOffsetTensor != nullptr, nullptr);
        antiquantX2 = l0op::Add(antiquantX2, antiquantOffsetTensor, executor);
        CHECK_RET(antiquantX2 != nullptr, nullptr);
    }
    // mul
    if (!IsFloatEqual(antiquantScale, 1.0f)) {
        antiquantX2 = l0op::Muls(antiquantX2, antiquantScale, executor);
        CHECK_RET(antiquantX2 != nullptr, nullptr);
    }
    // cast to 16
    antiquantX2 = l0op::Cast(antiquantX2, DataType::DT_FLOAT16, executor);
    CHECK_RET(antiquantX2 != nullptr, nullptr);

    // add
    if (addOffset != nullptr) {
        antiquantX2 = l0op::Add(antiquantX2, addOffset, executor);
        CHECK_RET(antiquantX2 != nullptr, nullptr);
    }
    // mul
    if (mulScale != nullptr) {
        antiquantX2 = l0op::Mul(antiquantX2, mulScale, executor);
        CHECK_RET(antiquantX2 != nullptr, nullptr);
    }
    // bias
    if (bias != nullptr) {
        bias = l0op::Cast(bias, DataType::DT_FLOAT16, executor);
        CHECK_RET(bias != nullptr, nullptr);
    }
    // matmul 有一个大于等于2 就为batchmatmul
    if ((x1->GetViewShape().GetDimNum() > 2) || (x2->GetViewShape().GetDimNum() > 2)) {
        matmulOut = Ops::NN::ExecBatchMatmulOpWithBiasAndAttrs(x1, antiquantX2, bias, out, transposeX1, transposeX2,
                                                      0, executor);
    } else {
        // 先直调l0 完成功能
        matmulOut = l0op::MatMulNd(x1, antiquantX2, bias, nullptr, transposeX1, transposeX2,
                                   0, 0, executor);
    }
    return matmulOut;
}

aclnnStatus aclnnWeightQuantBatchMatmulGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
                                                   const aclTensor *diagonalMatrix, const aclTensor *deqOffset,
                                                   const aclTensor *deqScale, const aclTensor *addOffset,
                                                   const aclTensor *mulScale, const aclTensor *bias,
                                                   bool transposeX1, bool transposeX2,
                                                   float antiquantScale, float antiquantOffset, aclTensor *out,
                                                   uint64_t *workspaceSize, aclOpExecutor **executor) {
  L2_DFX_PHASE_1(aclnnWeightQuantBatchMatmul, DFX_IN(x1, x2, bias, addOffset, mulScale, diagonalMatrix,
                                                                deqOffset, deqScale, transposeX1,
                                                                transposeX2, antiquantScale, antiquantOffset), DFX_OUT(out));
  // 固定写法，创建OpExecutor
  auto unique_executor = CREATE_EXECUTOR();
  CHECK_RET(unique_executor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
  // 入参检查
  if (CheckParam(x1, x2, bias, addOffset, mulScale, diagonalMatrix, deqOffset, deqScale, out, transposeX1, transposeX2) != ACLNN_SUCCESS) {
    return ACLNN_ERR_PARAM_INVALID;
  }
  const aclTensor *matmulOut;
  // 空tensor 处理
  if (x1->IsEmpty() || x2->IsEmpty()) {
    matmulOut = ProcessEmptyTensor(x1, x2, bias, out, unique_executor.get());
    CHECK_RET(matmulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(matmulOut, out, unique_executor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // 获取workspace
    *workspaceSize = unique_executor->GetWorkspaceSize();
    unique_executor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }
  // 连续性转换
  CHECK_RET(InputPreProcess(x1, unique_executor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(InputPreProcess(x2, unique_executor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(InputPreProcess(bias, unique_executor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(InputPreProcess(addOffset, unique_executor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(InputPreProcess(mulScale, unique_executor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(InputPreProcess(diagonalMatrix, unique_executor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(InputPreProcess(deqOffset, unique_executor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
  CHECK_RET(InputPreProcess(deqScale, unique_executor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
  // 构建matmul计算图
  int64_t dimTensor1 = x1->GetViewShape().GetDimNum();
  int64_t m = transposeX1 ?  x1->GetViewShape().GetDim(dimTensor1 - 1) : x1->GetViewShape().GetDim(dimTensor1 - 2);
  if (SelectFixPipe(m)) {
    auto reformatDeqScale = const_cast<aclTensor *>(deqScale);
    reformatDeqScale->SetStorageFormat(op::Format::FORMAT_NC1HWC0);
    matmulOut = l0op::WeightQuantBatchmatmul(x1, x2, bias, diagonalMatrix, deqOffset, reformatDeqScale,
                                             transposeX1, transposeX2, unique_executor.get());
  } else {
    matmulOut = WeightQuantMatmulSingleOp(x1, x2, addOffset, mulScale, bias, antiquantScale, antiquantOffset,
                                          transposeX1, transposeX2, out,unique_executor.get());
  }
  CHECK_RET(matmulOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
  if (matmulOut->IsEmpty()) {
    // 当输出为空tensor的场景，空tensor处理
    *workspaceSize = 0;
    unique_executor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
  }
  matmulOut = l0op::Cast(matmulOut, out->GetDataType(), unique_executor.get());
  auto viewCopyResult = l0op::ViewCopy(matmulOut, out, unique_executor.get());
  CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
  // 获取workspace
  *workspaceSize = unique_executor->GetWorkspaceSize();
  unique_executor.ReleaseTo(executor);
  return ACLNN_SUCCESS;
}

aclnnStatus aclnnWeightQuantBatchMatmul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                             const aclrtStream stream) {
  L2_DFX_PHASE_2(aclnnWeightQuantBatchMatmul);
  // 固定写法，调用框架能力，完成计算
  return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}


#ifdef __cplusplus
}
#endif