/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file transpose_quant_batch_mat_mul_infer.cpp
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
const size_t kX1ScaleIdx = 3;
const size_t kX2ScaleIdx = 4;
const int64_t kSupportedInnerAxis = 65536;
const size_t kBlockSize = 128;
const size_t DIM_2 = 2;
const size_t DIM_1 = 1;
const size_t DIM_0 = 0;
const size_t VALID_K = 512;
const size_t VALID_N = 128;
const size_t VALID_BATCH_SPLIT_FACTOR = 1;
constexpr static int64_t UNKNOWN_DIM_NUM = static_cast<int64_t>(-2);
constexpr static int64_t N_DIM_NUM = 2;
constexpr static size_t PERM_DIM_NUM = 3;

static bool TransposeShape(const Shape& src, const TypedContinuousVector<int64_t>& perm, Shape& dst)
{
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

static ge::graphStatus CheckScaleForTransposeQuantBatchMatMul(
    const InferShapeContext* context, const Shape* x1Scale, const Shape* x2Scale, Shape& shapeX1Transposed,
    Shape& shapeX2Transposed)
{
    if (x1Scale != nullptr && x1Scale->GetDimNum() == 1) {
        CHECK(
            x1Scale->GetDim(0) != shapeX1Transposed.GetDim(1), // scale[0] = x1[1] = m
            OP_LOGE(
                context->GetNodeName(), "The m dimension of x1 [%ld] and x1Scale [%ld] tensors must be the same",
                shapeX1Transposed.GetDim(1), x1Scale->GetDim(0)),
            return ge::GRAPH_FAILED);
    }

    if (x2Scale != nullptr && x2Scale->GetDimNum() == 1) {
        CHECK(
            x2Scale->GetDim(0) != shapeX2Transposed.GetDim(2), // scale[0] = x2[2] = n
            OP_LOGE(
                context->GetNodeName(), "The n dimension of x2 [%ld] and x2Scale [%ld] tensors must be the same",
                shapeX2Transposed.GetDim(2), x2Scale->GetDim(0)),
            return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetShapeY(
    Shape& shapeY, const Shape& shapeX1Transposed, const Shape& shapeX2Transposed,
    const TypedContinuousVector<int64_t>& permY, const int32_t batchSplitFactor)
{
    constexpr int EXPECTED_DIM = 3; // the dim of y should be 3
    shapeY.SetDimNum(EXPECTED_DIM);
    auto mDim = shapeX1Transposed.GetDim(1);
    auto batchDim = shapeX1Transposed.GetDim(0);
    auto nDim = shapeX2Transposed.GetDim(2);
    // m b n
    shapeY.SetDim(0, mDim);
    shapeY.SetDim(1, batchDim);
    shapeY.SetDim(N_DIM_NUM, nDim);

    // bs, m , bn/bs
    if (batchSplitFactor > 1) {
        int64_t m = shapeY.GetDim(0);
        int64_t batch_n = shapeY.GetDim(1) * shapeY.GetDim(2);
        shapeY.SetDim(0, batchSplitFactor);
        shapeY.SetDim(1, m);
        shapeY.SetDim(2, batch_n / batchSplitFactor); // 2 is dim index
    }

    Shape shapeY_transposed;
    CHECK(
        !TransposeShape(shapeY, permY, shapeY_transposed),
        CUBE_INNER_ERR_REPORT("TQBMM", "[InferShape] Failed to transpose shape of y"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckPermForTransposeQuantBatchMatMul(
    const TypedContinuousVector<int64_t>& permX1, const TypedContinuousVector<int64_t>& permX2,
    const TypedContinuousVector<int64_t>& permY)
{
    bool permSizeInvalid =
        permX1.GetSize() != PERM_DIM_NUM || permX2.GetSize() != PERM_DIM_NUM || permY.GetSize() != PERM_DIM_NUM;
    CHECK(
        permSizeInvalid,
        CUBE_INNER_ERR_REPORT(
            "TQBMM",
            "The dims of the permArray should be 3, now permX1 dims: %ld, permX2 dims: %ld, permY dims: %ld",
            permX1.GetSize(), permX2.GetSize(), permY.GetSize()),
        return ge::GRAPH_FAILED);
    const auto permX1_attr = permX1.GetData();
    auto check_permX1 = (*permX1_attr == 1 && *(permX1_attr + 1) == 0 && *(permX1_attr + 2) == 2);
    CHECK(
        !check_permX1, CUBE_INNER_ERR_REPORT("TQBMM", "[InferShape] permX1 should be {1, 0, 2}"),
        return ge::GRAPH_FAILED);

    const auto permX2_attr = permX2.GetData();
    auto check_permX2 = (*permX2_attr == 0 && *(permX2_attr + 1) == 1 && *(permX2_attr + 2) == 2);
    CHECK(
        !check_permX2, CUBE_INNER_ERR_REPORT("TQBMM", "[InferShape] permX2 should be {0, 1, 2}"),
        return ge::GRAPH_FAILED);

    const auto permY_attr = permY.GetData();
    auto check_permY = *permY_attr == 1 && *(permY_attr + 1) == 0 && *(permY_attr + 2) == 2;
    CHECK(!check_permY, CUBE_INNER_ERR_REPORT("TQBMM", "[InferShape] permY should {1, 0, 2}"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForTransposeQuantBatchMatMul(InferShapeContext* context)
{
    CHECK(
        context == nullptr, CUBE_INNER_ERR_REPORT("TransposeQuantBatchMatMul", "context is null"),
        return ge::GRAPH_FAILED);

    auto shapeX1 = context->GetInputShape(0);
    auto shapeX2 = context->GetInputShape(1);
    auto shapeY = context->GetOutputShape(0);
    auto attrs = context->GetAttrs();
    auto nameOp = context->GetNodeName();
    CHECK(
        shapeX1 == nullptr || shapeX2 == nullptr || shapeY == nullptr || attrs == nullptr,
        CUBE_INNER_ERR_REPORT(nameOp, "[Infershape]shape or attrs is null."), return ge::GRAPH_FAILED);

    const auto dtype = attrs->GetAttrPointer<int32_t>(0); // dtype index is 0
    CHECK(dtype == nullptr, CUBE_INNER_ERR_REPORT(nameOp, "[Infershape] attr dtype is null."), return ge::GRAPH_FAILED);

    const auto permX1 = attrs->GetListInt(2);                        // permX1 index is 2
    const auto permX2 = attrs->GetListInt(3);                        // permX2 index is 3
    const auto permY = attrs->GetListInt(4);                         // permY index is 4
    const auto batchSplitFactor = attrs->GetAttrPointer<int32_t>(5); // batchSplitFactor index is 5
    CHECK(
        permX1 == nullptr || permX2 == nullptr || permY == nullptr,
        CUBE_INNER_ERR_REPORT(nameOp, "[Infershape] attr is nullptr."), return ge::GRAPH_FAILED);
    CHECK(
        CheckPermForTransposeQuantBatchMatMul(*permX1, *permX2, *permY) != ge::GRAPH_SUCCESS,
        CUBE_INNER_ERR_REPORT(nameOp, "[InferShape] Failed to check perm value."), return ge::GRAPH_FAILED);

    Shape shapeX1Transposed;
    Shape shapeX2Transposed;
    CHECK(
        !TransposeShape(*shapeX1, *permX1, shapeX1Transposed),
        CUBE_INNER_ERR_REPORT(nameOp, "[InferShape] Failed to transpose shape of x1"), return ge::GRAPH_FAILED);
    CHECK(
        !TransposeShape(*shapeX2, *permX2, shapeX2Transposed),
        CUBE_INNER_ERR_REPORT(nameOp, "[InferShape] Failed to transpose shape of x2"), return ge::GRAPH_FAILED);

    // Check x1 and x2 shape
    CHECK(
        (shapeX1Transposed.GetDimNum() != PERM_DIM_NUM) || (shapeX2Transposed.GetDimNum() != PERM_DIM_NUM),
        CUBE_INNER_ERR_REPORT(
            nameOp, "The dims of the two inputs should be 3, now x1 dims: %ld and x2 dims: %d.", shapeX1Transposed.GetDimNum(),
            shapeX2Transposed.GetDimNum()),
        return ge::GRAPH_FAILED);
    // check x1 and x2 k-axis
    CHECK(
        shapeX1Transposed.GetDim(DIM_2) != shapeX2Transposed.GetDim(DIM_1),
        CUBE_INNER_ERR_REPORT(nameOp, "The k-axis of the two inputs are different."), return ge::GRAPH_FAILED);
    // check x1 and x2 batch-axis
    CHECK(
        shapeX1Transposed.GetDim(DIM_0) != shapeX2Transposed.GetDim(DIM_0),
        CUBE_INNER_ERR_REPORT(
            nameOp, "The batch-axis must be equal, transposed shape of x1 and x2 is %s, %s.",
            Ops::Base::ToString(shapeX1Transposed).c_str(), Ops::Base::ToString(shapeX2Transposed).c_str()),
        return ge::GRAPH_FAILED);
    // Check shape k n
    CHECK(
        shapeX2Transposed.GetDim(DIM_1) != VALID_K || shapeX2Transposed.GetDim(DIM_2) != VALID_N,
        CUBE_INNER_ERR_REPORT(
            nameOp, "The shape of the x2 is not supported, now transposed shape of is %s.",
            Ops::Base::ToString(shapeX2Transposed).c_str()),
        return ge::GRAPH_FAILED);

    auto* x1Scale = context->GetOptionalInputShape(kX1ScaleIdx);
    auto* x2Scale = context->GetOptionalInputShape(kX2ScaleIdx);
    // 当前不允许x1Scale或者x2Scale为空
    CHECK(
        x1Scale == nullptr || x2Scale == nullptr, CUBE_INNER_ERR_REPORT(nameOp, "X1Scale or x2Scale is null."),
        return ge::GRAPH_FAILED);
    // batchSplitFactor only support 1
    CHECK(
        batchSplitFactor != nullptr && *batchSplitFactor != VALID_BATCH_SPLIT_FACTOR,
        CUBE_INNER_ERR_REPORT(nameOp, "batchSplitFactor should be 1 ."), return ge::GRAPH_FAILED);

    // Set shapeY
    ge::graphStatus ret = SetShapeY(*shapeY, shapeX1Transposed, shapeX2Transposed, *permY, *batchSplitFactor);
    OP_LOGD(nameOp, "y_shape: %s", Ops::Base::ToString(*shapeY).c_str());
    CHECK(
        ret != ge::GRAPH_SUCCESS, CUBE_INNER_ERR_REPORT(nameOp, "[InferShape] set shapeY failed."),
        return ge::GRAPH_FAILED);

    CHECK(
        CheckScaleForTransposeQuantBatchMatMul(context, x1Scale, x2Scale, shapeX1Transposed, shapeX2Transposed) !=
            ge::GRAPH_SUCCESS,
        CUBE_INNER_ERR_REPORT(nameOp, "[InferShape] scale shape check failed."), return ge::GRAPH_FAILED);

    // no need to SetDataType in runtime
    return ge::GRAPH_SUCCESS;
}

} // namespace

namespace Ops::NN::MatMul {
IMPL_OP_INFERSHAPE(TransposeQuantBatchMatMul).InferShape(InferShapeForTransposeQuantBatchMatMul);
}
