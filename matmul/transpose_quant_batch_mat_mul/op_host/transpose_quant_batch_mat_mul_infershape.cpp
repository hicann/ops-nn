/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include <algorithm>

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
const std::initializer_list<ge::DataType> OUT_TYPE_LIST = {ge::DT_FLOAT16, ge::DT_BF16};
constexpr size_t ATTR_INDEX_DST_TYPE = 0;
constexpr int64_t Y_INDEX = 0;

static inline bool IsMicroScaling(const ge::DataType& dtypeX1Scale, const ge::DataType& dtypeX2Scale)
{
    return dtypeX1Scale == ge::DT_FLOAT8_E8M0 && dtypeX2Scale == ge::DT_FLOAT8_E8M0;
}

static inline bool IsHIFP8(const ge::DataType& dtypeX1, const ge::DataType& dtypeX2)
{
    return dtypeX1 == ge::DT_HIFLOAT8 && dtypeX2 == ge::DT_HIFLOAT8;
}

static bool CheckDtypeValid(const ge::DataType& dtypeX1, const ge::DataType& dtypeX2, const ge::DataType& dtypeX1Scale,
                            const ge::DataType& dtypeX2Scale)
{
    // MXFP8
    if (IsMicroScaling(dtypeX1Scale, dtypeX2Scale)) {
        CHECK(dtypeX1 != ge::DT_FLOAT8_E4M3FN || dtypeX2 != ge::DT_FLOAT8_E4M3FN,
              CUBE_INNER_ERR_REPORT("TQBMM", "the dtype of input is only supported FLOAT8_E4M3FN."), return false);
        // FP8
    } else if (!IsHIFP8(dtypeX1, dtypeX2)) {
        CHECK((dtypeX1 != ge::DT_FLOAT8_E4M3FN && dtypeX1 != ge::DT_FLOAT8_E5M2) ||
                  (dtypeX2 != ge::DT_FLOAT8_E4M3FN && dtypeX2 != ge::DT_FLOAT8_E5M2),
              CUBE_INNER_ERR_REPORT("TQBMM", "the dtype of input is only supported FLOAT8_E4M3FN or FLOAT8_E5M2."),
              return false);
    }
    // MXFP8/FP8 scale dtype check
    if (!IsHIFP8(dtypeX1, dtypeX2)) {
        CHECK((dtypeX1Scale != ge::DT_FLOAT && dtypeX1Scale != ge::DT_FLOAT8_E8M0) ||
                  (dtypeX2Scale != ge::DT_FLOAT && dtypeX2Scale != ge::DT_FLOAT8_E8M0),
              CUBE_INNER_ERR_REPORT("TQBMM", "the dtype of scale is only supported FLOAT or FLOAT8_E8M0."),
              return false);
    } else {
        CHECK(dtypeX2Scale != ge::DT_UINT64,
              CUBE_INNER_ERR_REPORT("TQBMM", "the dtype of scale is only supported UINT64."), return false);
    }
    return true;
}

static bool CheckPerm(const TypedContinuousVector<int64_t>* permX1, const TypedContinuousVector<int64_t>* permX2,
                      const TypedContinuousVector<int64_t>* permY)
{
    CHECK(permX1 == nullptr || permX2 == nullptr || permY == nullptr,
          CUBE_INNER_ERR_REPORT("TQBMM", "[Infershape] attr is nullptr."), return false);

    CHECK(permX1->GetSize() != PERM_DIM_NUM || permX2->GetSize() != PERM_DIM_NUM || permY->GetSize() != PERM_DIM_NUM,
          CUBE_INNER_ERR_REPORT("TQBMM", "[InferShape] The dims of the perm intArray should be 3"), return false);
    const auto permX1Attr = permX1->GetData();
    auto checkPermX1 = (*permX1Attr == 1 && *(permX1Attr + 1) == 0 && *(permX1Attr + 2) == 2);
    CHECK(!checkPermX1, CUBE_INNER_ERR_REPORT("TQBMM", "[InferShape] perm_x1 should be {1, 0, 2}"), return false);

    const auto permX2Attr = permX2->GetData();
    auto checkPermX2 = (*permX2Attr == 0 && *(permX2Attr + 1) == 1 && *(permX2Attr + 2) == 2) ||
                       (*permX2Attr == 0 && *(permX2Attr + 1) == 2 && *(permX2Attr + 2) == 1);
    CHECK(!checkPermX2, CUBE_INNER_ERR_REPORT("TQBMM", "[InferShape] perm_x2 should be {0, 1, 2} or {0, 2, 1}"),
          return false);

    const auto permYAttr = permY->GetData();
    auto checkPermY = *permYAttr == 1 && *(permYAttr + 1) == 0 && *(permYAttr + 2) == 2;
    CHECK(!checkPermY, CUBE_INNER_ERR_REPORT("TQBMM", "[InferShape] perm_y should {1, 0, 2}"), return false);
    return true;
}

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

static ge::graphStatus SetShapeY(Shape& shapeY, const Shape& shapeX1Transposed, const Shape& shapeX2Transposed,
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
    CHECK(!TransposeShape(shapeY, permY, shapeY_transposed),
          CUBE_INNER_ERR_REPORT("TQBMM", "[InferShape] Failed to transpose shape of y"), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForTransposeQuantBatchMatMul(InferShapeContext* context)
{
    CHECK(context == nullptr, CUBE_INNER_ERR_REPORT("TransposeQuantBatchMatMul", "context is null"),
          return ge::GRAPH_FAILED);

    auto shapeX1 = context->GetInputShape(0);
    auto shapeX2 = context->GetInputShape(1);
    auto shapeY = context->GetOutputShape(0);
    auto attrs = context->GetAttrs();
    auto nameOp = context->GetNodeName();
    CHECK(shapeX1 == nullptr || shapeX2 == nullptr || shapeY == nullptr || attrs == nullptr,
          CUBE_INNER_ERR_REPORT(nameOp, "[Infershape]shape or attrs is null."), return ge::GRAPH_FAILED);

    const auto dtype = attrs->GetAttrPointer<int64_t>(0); // dtype index is 0
    CHECK(dtype == nullptr, CUBE_INNER_ERR_REPORT(nameOp, "[Infershape] attr dtype is null."), return ge::GRAPH_FAILED);
    auto tensorX1 = context->GetInputDesc(0);
    auto tensorX2 = context->GetInputDesc(1);
    CHECK(tensorX1 == nullptr || tensorX2 == nullptr, CUBE_INNER_ERR_REPORT(nameOp, "x1 or x2 is null."),
          return ge::GRAPH_FAILED);
    ge::DataType dtypeX1 = tensorX1->GetDataType();
    ge::DataType dtypeX2 = tensorX2->GetDataType();
    bool isHIFP8 = IsHIFP8(dtypeX1, dtypeX2);
    // 当前不允许x1Scale或者x2Scale为空
    auto tensorX1Scale = context->GetOptionalInputDesc(kX1ScaleIdx);
    auto tensorX2Scale = context->GetOptionalInputDesc(kX2ScaleIdx);
    CHECK((!isHIFP8 && tensorX1Scale == nullptr) || tensorX2Scale == nullptr,
          CUBE_INNER_ERR_REPORT(nameOp, "X1Scale or x2Scale is null."), return ge::GRAPH_FAILED);

    ge::DataType dtypeX1Scale = isHIFP8 ? ge::DT_UINT64 : tensorX1Scale->GetDataType();
    ge::DataType dtypeX2Scale = tensorX2Scale->GetDataType();
    CHECK(!CheckDtypeValid(dtypeX1, dtypeX2, dtypeX1Scale, dtypeX2Scale),
          CUBE_INNER_ERR_REPORT(nameOp, "[InferShape] Failed to check dtype"), return ge::GRAPH_FAILED);

    const auto permX1 = attrs->GetListInt(2);                        // permX1 index is 2
    const auto permX2 = attrs->GetListInt(3);                        // permX2 index is 3
    const auto permY = attrs->GetListInt(4);                         // permY index is 4
    const auto batchSplitFactor = attrs->GetAttrPointer<int32_t>(5); // batchSplitFactor index is 5
    CHECK(!CheckPerm(permX1, permX2, permY), CUBE_INNER_ERR_REPORT(nameOp, "[InferShape] Failed to check perm"),
          return ge::GRAPH_FAILED);
    Shape shapeX1Transposed;
    Shape shapeX2Transposed;
    CHECK(!TransposeShape(*shapeX1, *permX1, shapeX1Transposed),
          CUBE_INNER_ERR_REPORT(nameOp, "[InferShape] Failed to transpose shape of x1"), return ge::GRAPH_FAILED);
    CHECK(!TransposeShape(*shapeX2, *permX2, shapeX2Transposed),
          CUBE_INNER_ERR_REPORT(nameOp, "[InferShape] Failed to transpose shape of x2"), return ge::GRAPH_FAILED);

    // Check x1 and x2 shape
    CHECK((shapeX1Transposed.GetDimNum() != PERM_DIM_NUM) || (shapeX2Transposed.GetDimNum() != PERM_DIM_NUM),
          CUBE_INNER_ERR_REPORT(nameOp, "The dims of the two inputs should be 3, now x1 dims: %zu and x2 dims: %zu.",
                                shapeX1Transposed.GetDimNum(), shapeX2Transposed.GetDimNum()),
          return ge::GRAPH_FAILED);
    // check x1 and x2 k-axis
    CHECK(shapeX1Transposed.GetDim(DIM_2) != shapeX2Transposed.GetDim(DIM_1),
          CUBE_INNER_ERR_REPORT(nameOp, "The k-axis of the two inputs are different."), return ge::GRAPH_FAILED);
    // check x1 and x2 batch-axis
    CHECK(shapeX1Transposed.GetDim(DIM_0) != shapeX2Transposed.GetDim(DIM_0),
          CUBE_INNER_ERR_REPORT(nameOp, "The batch-axis must be equal, transposed shape of x1 and x2 is %s, %s.",
                                Ops::Base::ToString(shapeX1Transposed).c_str(),
                                Ops::Base::ToString(shapeX2Transposed).c_str()),
          return ge::GRAPH_FAILED);

    // batchSplitFactor only support 1
    CHECK(batchSplitFactor != nullptr && *batchSplitFactor != VALID_BATCH_SPLIT_FACTOR,
          CUBE_INNER_ERR_REPORT(nameOp, "batchSplitFactor should be 1 ."), return ge::GRAPH_FAILED);

    // Set shapeY
    ge::graphStatus ret = SetShapeY(*shapeY, shapeX1Transposed, shapeX2Transposed, *permY, *batchSplitFactor);
    OP_LOGD(nameOp, "y_shape: %s", Ops::Base::ToString(*shapeY).c_str());
    CHECK(ret != ge::GRAPH_SUCCESS, CUBE_INNER_ERR_REPORT(nameOp, "[InferShape] set shapeY failed."),
          return ge::GRAPH_FAILED);
    // no need to SetDataType in runtime
    return ge::GRAPH_SUCCESS;
}

} // namespace

static ge::graphStatus TransposeQuantBatchMatMulInferDataType(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context, "TransposeQuantBatchMatMulInferDataType begin");
    auto nameOp = context->GetNodeName();
    auto* attrs = context->GetAttrs();
    CHECK(attrs == nullptr, CUBE_INNER_ERR_REPORT(nameOp, "[Infershape] attr is nullptr."), return ge::GRAPH_FAILED);
    const int32_t* dtype = attrs->GetAttrPointer<int32_t>(ATTR_INDEX_DST_TYPE);
    CHECK(dtype == nullptr, CUBE_INNER_ERR_REPORT(nameOp, "[Infershape] dtype is nullptr."), return ge::GRAPH_FAILED);
    int32_t dstDtype = *dtype;
    ge::DataType yDtype = static_cast<ge::DataType>(dstDtype);

    OP_CHECK_IF(std::find(OUT_TYPE_LIST.begin(), OUT_TYPE_LIST.end(), yDtype) == OUT_TYPE_LIST.end(),
                OP_LOGE(context, "attr dtype only support float16, bfloat16"), return ge::GRAPH_FAILED);

    context->SetOutputDataType(Y_INDEX, yDtype);

    OP_LOGD(context, "TransposeQuantBatchMatMulInferDataType end");
    return ge::GRAPH_SUCCESS;
}

namespace Ops::NN::MatMul {
IMPL_OP_INFERSHAPE(TransposeQuantBatchMatMul)
    .InferShape(InferShapeForTransposeQuantBatchMatMul)
    .InferDataType(TransposeQuantBatchMatMulInferDataType);
}
