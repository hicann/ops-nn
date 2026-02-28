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
 * \file transpose_batch_mat_mul_infer.cpp
 * \brief
 */
#include <string>

#include "error_util.h"
#include "common/op_host/matmul_common_infershape.h"
using namespace gert;
namespace {
#define CHECK(cond, log_func, return_expr) \
    do {                                   \
        if (cond) {                        \
            log_func;                      \
            return_expr;                   \
        }                                  \
    } while (0)
const size_t kTransposeBatchMatMulScaleIdx = 3;
const size_t kBlockSize = 128;
constexpr static int64_t UNKNOWN_DIM_NUM = static_cast<int64_t>(-2);
constexpr static int64_t N_DIM_NUM = 2;

static bool TransposeShape(const Shape &src, const TypedContinuousVector<int64_t> &perm, Shape &dst) {
  if (perm.GetSize() == 0) {
    dst = src;
    return true;
  }

  if (src.GetDimNum() != perm.GetSize() || dst.GetDimNum() != 0) {
    return false;
  }

  for (size_t idx_dst = 0; idx_dst < perm.GetSize(); ++idx_dst) {
    dst.AppendDim(src.GetDim(*(perm.GetData() + idx_dst)));
  }

  return true;
}

static bool CheckIsUnknownDimNum(const gert::Shape& shape)
{
    return shape.GetDimNum() == 1 && shape.GetDim(0) == UNKNOWN_DIM_NUM;
}

static ge::graphStatus CheckScaleForTransposeBatchMatMul(const InferShapeContext* context,
                                                         const Shape& shape_x2_transposed)
{
    const Shape* shape_scale = context->GetOptionalInputShape(kTransposeBatchMatMulScaleIdx);
    if (shape_scale != nullptr && shape_scale->GetDimNum() == 1) {
        int64_t scale_dim = shape_scale->GetDimNum();
        CHECK(shape_scale->GetDim(scale_dim - 1) !=
                  shape_x2_transposed.GetDim(0) * shape_x2_transposed.GetDim(2), // scale = batch * n
              OP_LOGE(context->GetNodeName(), "The dimension of n * b [%ld] and scale [%ld] tensors must be the same",
                      shape_x2_transposed.GetDim(0) * shape_x2_transposed.GetDim(2),
                      shape_scale->GetDim(scale_dim - 1)),
              return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetShapeY(Shape &shape_y, const Shape &shape_x1_transposed, const Shape &shape_x2_transposed,
                                 const TypedContinuousVector<int64_t> &perm_y, const int32_t batch_split_factor,
                                 bool has_scale_shape)
{
  constexpr int EXPECTED_DIM = 3; // the dim of y should be 3
  shape_y.SetDimNum(EXPECTED_DIM);
  auto m_dim = shape_x1_transposed.GetDim(1);
  auto batch_dim = shape_x1_transposed.GetDim(0);
  auto n_dim = shape_x2_transposed.GetDim(2);
  shape_y.SetDim(0, m_dim);
  shape_y.SetDim(1, batch_dim);
  shape_y.SetDim(N_DIM_NUM, n_dim);

  if (has_scale_shape) {
    shape_y.SetDim(0, m_dim);
    shape_y.SetDim(1, 1);
    shape_y.SetDim(N_DIM_NUM, batch_dim * n_dim);
  }

  if (batch_split_factor > 1) {
    int64_t m = shape_y.GetDim(0);
    int64_t batch_n = shape_y.GetDim(1) * shape_y.GetDim(2);
    shape_y.SetDim(0, batch_split_factor);
    shape_y.SetDim(1, m);
    shape_y.SetDim(2, batch_n / batch_split_factor); // 2 is dim index
  }

  Shape shape_y_transposed;
  CHECK(!TransposeShape(shape_y, perm_y, shape_y_transposed),
        CUBE_INNER_ERR_REPORT("TBMM", "[InferShape] Failed to transpose shape of y"), return ge::GRAPH_FAILED);

  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckPermForTransposeBatchMatMul(const TypedContinuousVector<int64_t> &perm_x1,
                                                        const TypedContinuousVector<int64_t> &perm_x2,
                                                        const TypedContinuousVector<int64_t> &perm_y)
{
  const auto perm_x1_attr = perm_x1.GetData();
  auto check_perm_x1 = (*perm_x1_attr == 1 && *(perm_x1_attr + 1) == 0 && *(perm_x1_attr + 2) == 2) ||
                       (*perm_x1_attr == 0 && *(perm_x1_attr + 1) == 1 && *(perm_x1_attr + 2) == 2);
  CHECK(
      !check_perm_x1, CUBE_INNER_ERR_REPORT("TBMM", "[InferShape] perm_x1 should be {1, 0, 2} or {0, 1, 2}"),
      return ge::GRAPH_FAILED);

  const auto perm_x2_attr = perm_x2.GetData();
  auto check_perm_x2 = (*perm_x2_attr == 0 && *(perm_x2_attr + 1) == 1 && *(perm_x2_attr + 2) == 2) ||
                        (*perm_x2_attr == 0 && *(perm_x2_attr + 1) == 2 && *(perm_x2_attr + 2) == 1);
  CHECK(
      !check_perm_x2, CUBE_INNER_ERR_REPORT("TBMM", "[InferShape] perm_x2 should be {0, 1, 2} or {0, 2, 1}"),
      return ge::GRAPH_FAILED);

  const auto perm_y_attr = perm_y.GetData();
  auto check_perm_y = *perm_y_attr == 1 && *(perm_y_attr + 1) == 0 && *(perm_y_attr + 2) == 2;
  CHECK(
      !check_perm_y, CUBE_INNER_ERR_REPORT("TBMM", "[InferShape] perm_y should {1, 0, 2}"),
      return ge::GRAPH_FAILED);
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForTransposeBatchMatMul(InferShapeContext *context)
{
  CHECK(
      context == nullptr, CUBE_INNER_ERR_REPORT("TransposeBatchMatMul", "context is null"),
      return ge::GRAPH_FAILED);

  auto shape_x1 = context->GetInputShape(0);
  auto shape_x2 = context->GetInputShape(1);
  auto shape_y = context->GetOutputShape(0);
  auto attrs = context->GetAttrs();
  auto name_op = context->GetNodeName();
  CHECK(shape_x1 == nullptr || shape_x2 == nullptr || shape_y == nullptr || attrs == nullptr,
        CUBE_INNER_ERR_REPORT(name_op, "[Infershape]shape or attrs is null"), return ge::GRAPH_FAILED);

  const auto perm_x1 = attrs->GetListInt(0);
  const auto perm_x2 = attrs->GetListInt(1);
  const auto perm_y = attrs->GetListInt(2);
  const auto batch_split_factor = attrs->GetAttrPointer<int32_t>(4); // batch_split_factor index is 4
  CHECK(perm_x1 == nullptr || perm_x2 == nullptr || perm_y == nullptr || batch_split_factor == nullptr,
        CUBE_INNER_ERR_REPORT(name_op, "[Infershape] null"), return ge::GRAPH_FAILED);

  CHECK(CheckPermForTransposeBatchMatMul(*perm_x1, *perm_x2, *perm_y) != ge::GRAPH_SUCCESS,
        CUBE_INNER_ERR_REPORT(name_op, "[InferShape] Failed to check perm"), return ge::GRAPH_FAILED);
  Shape shape_x1_transposed;
  Shape shape_x2_transposed;
  CHECK(!TransposeShape(*shape_x1, *perm_x1, shape_x1_transposed),
        CUBE_INNER_ERR_REPORT(name_op, "[InferShape] Failed to transpose shape of x1"), return ge::GRAPH_FAILED);
  CHECK(!TransposeShape(*shape_x2, *perm_x2, shape_x2_transposed),
        CUBE_INNER_ERR_REPORT(name_op, "[InferShape] Failed to transpose shape of x2"), return ge::GRAPH_FAILED);

  CHECK(
      shape_x1_transposed.GetDim(2) != shape_x2_transposed.GetDim(1),
      CUBE_INNER_ERR_REPORT(name_op, "The k-axis of the two inputs are different."), return ge::GRAPH_FAILED);

  CHECK(
      shape_x1_transposed.GetDim(0) != shape_x2_transposed.GetDim(0),
      CUBE_INNER_ERR_REPORT(
          name_op, "[InferShape] batch must be equal, transposed shape of x1 and x2 is %s, %s.",
          Ops::Base::ToString(shape_x1_transposed).c_str(), Ops::Base::ToString(shape_x2_transposed).c_str()),
      return ge::GRAPH_FAILED);

  CHECK(
      (*batch_split_factor <= 0 || *batch_split_factor > shape_x1_transposed.GetDim(0) ||
       shape_x1_transposed.GetDim(0) % *batch_split_factor != 0),
      CUBE_INNER_ERR_REPORT(name_op, "batch_split_factor is not supported."), return ge::GRAPH_FAILED);

  auto *shape_scale = context->GetOptionalInputShape(kTransposeBatchMatMulScaleIdx);
  if (shape_scale != nullptr) {
    CHECK(
        *batch_split_factor != 1,
        CUBE_INNER_ERR_REPORT(name_op, "batchSplitFactor should be 1 when the scale is not null."),
        return ge::GRAPH_FAILED);
    CHECK(
        !CheckIsUnknownDimNum(*shape_scale) &&
            shape_scale->GetDim(0) != shape_x1_transposed.GetDim(0) * shape_x2_transposed.GetDim(2),
        CUBE_INNER_ERR_REPORT(
            name_op, "The dimension of n mul b [%ld] and scale [%ld] tensors must be the same",
            shape_x1_transposed.GetDim(0) * shape_x2_transposed.GetDim(2), shape_scale->GetDim(0)),
        return ge::GRAPH_FAILED);
    auto tensor_x1 = context->GetInputDesc(0);
    auto tensor_x2 = context->GetInputDesc(1);
    ge::DataType dtype_x1 = tensor_x1->GetDataType();
    ge::DataType dtype_x2 = tensor_x2->GetDataType();
    CHECK(dtype_x1 != ge::DT_FLOAT16, CUBE_INNER_ERR_REPORT(name_op,
         "the dtype of input is only supported FLOAT16 when the scale takes effect."), return ge::GRAPH_FAILED);
    CHECK(dtype_x2 != ge::DT_FLOAT16, CUBE_INNER_ERR_REPORT(name_op,
         "the dtype of input is only supported FLOAT16 when the scale takes effect."), return ge::GRAPH_FAILED);
  }
  ge::graphStatus ret = SetShapeY(*shape_y, shape_x1_transposed, shape_x2_transposed,
                                  *perm_y, *batch_split_factor, shape_scale != nullptr);
  OP_LOGD(name_op, "y_shape: %s", Ops::Base::ToString(*shape_y).c_str());
  CHECK(ret != ge::GRAPH_SUCCESS, CUBE_INNER_ERR_REPORT(name_op, "[InferShape] set shape_y failed"),
        return ge::GRAPH_FAILED);

  CHECK(CheckScaleForTransposeBatchMatMul(context, shape_x2_transposed) != ge::GRAPH_SUCCESS,
        CUBE_INNER_ERR_REPORT(name_op, "[InferShape] scale shape check failed"), return ge::GRAPH_FAILED);
  // no need to SetDataType in runtime
  return ge::GRAPH_SUCCESS;
}


}  // namespace gert

namespace Ops::NN::MatMul {
  IMPL_OP_INFERSHAPE(TransposeBatchMatMul)
    .InferShape(InferShapeForTransposeBatchMatMul);
}
