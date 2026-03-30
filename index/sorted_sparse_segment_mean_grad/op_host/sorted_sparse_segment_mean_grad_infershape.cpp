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
 * \file sorted_sparse_segment_mean_grad_infershape.cpp
 * \brief
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"
#include "runtime/infer_shape_context.h"
#include "runtime/storage_shape.h"
#include "util/const_util.h"
#include "util/math_util.h"
#include "error_util.h"

using namespace ge;
using namespace Ops::Base;
namespace ops {

constexpr size_t kInputIndex0 = 0U;
constexpr size_t kInputIndex1 = 1U;
constexpr size_t kInputIndex2 = 2U;
constexpr size_t kInputIndex3 = 3U;
constexpr size_t kInputIndex4 = 4U;
constexpr size_t kOutputIndex0 = 0U;
constexpr size_t kRank1 = 1U;
constexpr int64_t UNKNOWN_DIM_VALUE_ = -1LL;

inline ge::graphStatus SetAllUnknownDim(const int64_t rank, gert::Shape* output_shape) {
    output_shape->SetDimNum(rank);
    for (int64_t i = 0; i < rank; ++i) {
      output_shape->SetDim(i, UNKNOWN_DIM_VALUE_);
    }
    OP_LOGD("SetAllUnknownDim", "set all dim = -1, output = %s", ToString(*output_shape).c_str());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus Concatenate(const gert::Shape* s1, const gert::Shape* s2, gert::Shape* out) {
    if (s1 == nullptr || s2 == nullptr || out == nullptr) {
      return GRAPH_FAILED;
    } 

    size_t s1_rank = s1->GetDimNum();
    size_t s2_rank = s2->GetDimNum();
    size_t rank = s1_rank + s2_rank;

    out->SetDimNum(rank);
    for (size_t i = 0; i < s1_rank; ++i) {
      out->SetDim(i, s1->GetDim(i));
    }
    for (size_t i = 0; i < s2_rank; ++i) {
      out->SetDim(s1_rank + i, s2->GetDim(i));
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus WithRankAtLeast(const gert::Shape* tensor, int64_t rank, 
                                       gert::Shape* out_shape, const std::string opName) {
    if (rank > INT32_MAX) {
      OP_LOGE(opName, "rank[%ld] cannot exceed kint32max", rank);
      return GRAPH_FAILED;
    }
    int64_t size = static_cast<int64_t>(tensor->GetDimNum());
    if ((!IsUnknownRank(*tensor)) && size < rank) {
      OP_LOGE(opName, "rank[%ld] must be at least [%ld]", size, rank);
      return GRAPH_FAILED;
    }
    *out_shape = *tensor;
    return GRAPH_SUCCESS;
}

graphStatus WithRank(const gert::Shape* tensor, int64_t rank, 
                     gert::Shape* out_shape, const std::string opName) {
    if (rank > INT32_MAX) {
      OP_LOGE(opName, "rank[%ld] cannot exceed kint32max", rank);
      return GRAPH_FAILED;
    }
    int64_t existing = static_cast<int64_t>(tensor->GetDimNum());
    if (IsUnknownRank(*tensor)) {
      SetAllUnknownDim(rank, out_shape);
      return GRAPH_SUCCESS;
    }
    if (existing != rank) {
      OP_LOGE(opName, "rank[%ld] must be [%ld]", existing, rank);
      return GRAPH_FAILED;
    }
    *out_shape = *tensor;
    return GRAPH_SUCCESS;
}

struct SubShapePara {
    int64_t start;
    int64_t end;
    int64_t stride;
    SubShapePara(int st, int e, int str) : start(st), end(e), stride(str) {}
};

graphStatus SubShapeUpdateStartAndEnd(SubShapePara& para, int64_t s_rank, const std::string opName) {
    int64_t start = para.start;
    int64_t end = para.end;
    int64_t stride = para.stride;
    if (start > s_rank) {
      start = s_rank;
    }
    if (end > s_rank) {
      end = s_rank;
    }
    if (stride < 0 && start == s_rank) {
      --start;
    }
    if (start < 0) {
      start += s_rank;
      if (start < 0) {
        OP_LOGE(opName, "invalid start[%ld] to get sub shape with rank[%ld]", 
                start - s_rank, s_rank);
        return GRAPH_FAILED;
      }
    }
    if (end < 0) {
      end += s_rank;
      if (end < 0) {
        OP_LOGE(opName, "invalid end[%ld] to get sub shape with rank[%ld]", 
                end - s_rank, s_rank);
        return GRAPH_FAILED;
      }
    }
    if (stride > 0 && start > end) {
      OP_LOGE(opName, "start[%ld] should be less than end[%ld] at positive stride[%ld]",
              start, end, stride);
      return GRAPH_FAILED;
    } else if (stride < 0 && start < end) {
      OP_LOGE(opName, "start[%ld] should be greater than end[%ld] at negative stride[%ld]",
              start, end, stride);
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

graphStatus SubShape(const gert::Shape* s,
                     SubShapePara& para, 
                     gert::Shape* out, 
                     const std::string opName) {
    int64_t &start = para.start;
    int64_t &end = para.end;
    int64_t stride = para.stride;
    int64_t s_rank = s->GetDimNum();
    if (s_rank > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
      OP_LOGE(opName, "rank[%ld] cannot exceed kint32max", s_rank);
      return GRAPH_FAILED;
    }
    if (start == 0 && stride == 1 &&
        ((!IsUnknownRank(*s) && end >= s_rank) ||
        (end == std::numeric_limits<int64_t>::max()))) {
      gert::Shape out_shape = *s;
      *out = out_shape;
      return GRAPH_SUCCESS;
    }
    OP_LOGE_IF(SubShapeUpdateStartAndEnd(para, s_rank , opName) != GRAPH_SUCCESS,
               ge::GRAPH_FAILED, opName, "update start and end failed.");
    out->SetDimNum(0);
    for (int64_t i = start; (stride > 0 ? i < end : i > end); i += stride) {
      out->AppendDim(s->GetDim(i));
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDtypeForSortedSparseSegmentMeanGrad(gert::InferDataTypeContext* context) {
    OP_LOGI(context->GetNodeName(), "Begin to do InferDtypeForSortedSparseSegmentMeanGrad");
    DataType x_data_type = context->GetInputDataType(kInputIndex0);
    context->SetOutputDataType(kOutputIndex0, x_data_type);
    return GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputAndOutputNum(gert::InferShapeContext *context) {
    OP_LOGI(context->GetNodeName(), "Begin to do CheckInputAndOutputNum");
    constexpr size_t INPUT_NUM = 5;
    constexpr size_t OUTPUT_NUM = 1;
    OP_CHECK_IF(INPUT_NUM != context->GetComputeNodeInputNum(),
        OP_LOGE(
            context->GetNodeName(),
            "[", context->GetNodeName(), "], input size should be 5, got[%zu], ", context->GetComputeNodeInputNum()),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(OUTPUT_NUM != context->GetComputeNodeOutputNum(),
        OP_LOGE(
            context->GetNodeName(),
            "[", context->GetNodeName(), "], output size should be 1, got[%zu].", context->GetComputeNodeOutputNum()),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SortedSparseSegmentMeanGradCheck(gert::InferShapeContext *context) {
    OP_LOGI(context->GetNodeName(), "Begin to do InferShapeForSortedSparseSegmentMeanGrad check");
    const gert::Shape *x_shape = context->GetInputShape(kInputIndex0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    gert::Shape unused_shape;
    OP_CHECK_IF(WithRankAtLeast(x_shape, kRank1, &unused_shape, context->GetNodeName()) != GRAPH_SUCCESS,
        OP_LOGE(
            context->GetNodeName(),
            "[", context->GetNodeName(), "], failed to call WithRankAtLeast function."),
        return ge::GRAPH_FAILED);
    const gert::Shape *indices_shape = context->GetInputShape(kInputIndex1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, indices_shape);
    OP_CHECK_IF(WithRank(indices_shape, kRank1, &unused_shape, context->GetNodeName()) != GRAPH_SUCCESS,
        OP_LOGE(
            context->GetNodeName(),
            "[", context->GetNodeName(), "], failed to call WithRank function."),
        return ge::GRAPH_FAILED);
    const gert::Shape *location_shape = context->GetInputShape(kInputIndex2);
    OPS_CHECK_NULL_WITH_CONTEXT(context, location_shape);
    OP_CHECK_IF(WithRank(location_shape, kRank1, &unused_shape, context->GetNodeName()) != GRAPH_SUCCESS,
        OP_LOGE(
            context->GetNodeName(),
            "[", context->GetNodeName(), "], failed to call WithRank function."),
        return ge::GRAPH_FAILED);
    const gert::Shape *out_dim_0_shape = context->GetInputShape(kInputIndex4);
    OPS_CHECK_NULL_WITH_CONTEXT(context, out_dim_0_shape);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForSortedSparseSegmentMeanGrad(gert::InferShapeContext *context) {
    OP_LOGI(context->GetNodeName(), "Begin to do InferShapeForSortedSparseSegmentMeanGrad");
    OP_CHECK_IF(CheckInputAndOutputNum(context) != GRAPH_SUCCESS, 
        OP_LOGE(context->GetNodeName(), "[", context->GetNodeName(), "], num check failed."), return ge::GRAPH_FAILED);
    const gert::Shape *x_shape = context->GetInputShape(kInputIndex0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    gert::Shape *y_shape = context->GetOutputShape(kOutputIndex0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, y_shape);
    gert::Shape const_shape;
    bool can_get_output_dim0 = Ops::Base::GetConstIntToShape(context, kInputIndex4, const_shape);
    if (IsUnknownRank(*x_shape)) {
        OP_LOGD(context->GetNodeName(), "Input shape is -2, set output shape to (-2, )");
        SetUnknownRank(*y_shape);
        return ge::GRAPH_SUCCESS; 
    } else if (IsUnknownShape(*x_shape) || !can_get_output_dim0) {
        y_shape->SetDimNum(x_shape->GetDimNum());
        for (uint32_t i = 0; i < x_shape->GetDimNum(); i++) {
            y_shape->SetDim(i, -1);
        }
        return ge::GRAPH_SUCCESS;
    } else {
        OP_CHECK_IF(SortedSparseSegmentMeanGradCheck(context) != GRAPH_SUCCESS, 
            OP_LOGE(context->GetNodeName(), "[", context->GetNodeName(), "], check failed."), return ge::GRAPH_FAILED);
        int64_t start = 1;
        int64_t stride = 1;
        gert::Shape sub_shape_out;
        struct SubShapePara para(start, static_cast<int64_t>(x_shape->GetDimNum()), stride);
        OP_CHECK_IF(SubShape(x_shape, para, &sub_shape_out, context->GetNodeName()) != GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "[", context->GetNodeName(), "], call SubShape function failed."), return ge::GRAPH_FAILED);
        gert::Shape dim0_shape;
        const gert::Tensor *out_dim_0_tensor = context->GetInputTensor(kInputIndex4);
        OPS_CHECK_NULL_WITH_CONTEXT(context, out_dim_0_tensor);
        int32_t dims0_data = *(out_dim_0_tensor->GetData<int32_t>());
        dim0_shape = gert::Shape({dims0_data});
        gert::Shape out;
        OP_CHECK_IF(Concatenate(&dim0_shape, &sub_shape_out, &out) != GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "[", context->GetNodeName(), "], call Concatenate function failed."), return ge::GRAPH_FAILED);
        *y_shape = out;
    }

    OP_LOGD(context->GetNodeName(), "End to do InferShapeForSortedSparseSegmentMeanGrad");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SortedSparseSegmentMeanGrad)
    .InputsDataDependency({kInputIndex4})
    .InferShape(InferShapeForSortedSparseSegmentMeanGrad)
    .InferDataType(InferDtypeForSortedSparseSegmentMeanGrad);
} // namespace ops