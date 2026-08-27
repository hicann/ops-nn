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
 * \file sparse_segment_mean_infershape.cpp
 * \brief
 */

#include <limits>
#include <string>
#include <vector>

#include "error_util.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "runtime/infer_shape_context.h"
#include "runtime/storage_shape.h"
#include "util/const_util.h"
#include "util/shape_util.h"

using namespace ge;
using namespace Ops::Base;
namespace ops {

static constexpr size_t kInputIndex0 = 0U;
static constexpr size_t kInputIndex1 = 1U;
static constexpr size_t kInputIndex2 = 2U;
static constexpr size_t kOutputIndex0 = 0U;
static constexpr int64_t kRank1 = 1;
static constexpr int64_t kNum1 = 1;
static constexpr int64_t kUnknownDim = -1;

static inline bool IsConstTensor(const gert::Tensor* inputTensor)
{
    if (inputTensor != nullptr) {
        if (inputTensor->GetAddr() == nullptr) {
            return inputTensor->GetShapeSize() == 0;
        }
        return true;
    }
    return false;
}

template <typename T>
static bool GetConstIntToShape(T* context, const int64_t constIdx, gert::Shape& constShape)
{
    const gert::Tensor* constTensor = context->GetInputTensor(constIdx);
    if (constTensor == nullptr) {
        OP_LOGE(context->GetNodeName(), "input[%ld] is null.", constIdx);
        return false;
    }
    if (!IsConstTensor(constTensor)) {
        OP_LOGW(context->GetNodeName(), "the input[%ld] is not const tensor, will return failed.", constIdx);
        return false;
    }
    return Ops::Base::GetConstIntToShape(context, constIdx, constShape);
}

namespace {
struct SubShapePara {
    SubShapePara(int64_t startInput, int64_t endInput, int64_t strideInput)
        : start(startInput), end(endInput), stride(strideInput)
    {}
    int64_t start;
    int64_t end;
    int64_t stride;
};
} // namespace

static ge::graphStatus WithRankAtLeast(const gert::Shape* tensor, int64_t rank, gert::Shape* outShape,
                                       const std::string& opName)
{
    int64_t size = static_cast<int64_t>(tensor->GetDimNum());
    if ((!IsUnknownRank(*tensor)) && size < rank) {
        OP_LOGE(opName, "rank[%ld] must be at least [%ld]", size, rank);
        return GRAPH_FAILED;
    }
    *outShape = *tensor;
    return GRAPH_SUCCESS;
}

static ge::graphStatus WithRank(const gert::Shape* tensor, int64_t rank, gert::Shape* outShape,
                                const std::string& opName)
{
    int64_t existing = static_cast<int64_t>(tensor->GetDimNum());
    if (IsUnknownRank(*tensor)) {
        outShape->SetDimNum(rank);
        for (int64_t i = 0; i < rank; ++i) {
            outShape->SetDim(i, kUnknownDim);
        }
        return GRAPH_SUCCESS;
    }
    if (existing != rank) {
        OP_LOGE(opName, "rank[%ld] must be [%ld]", existing, rank);
        return GRAPH_FAILED;
    }
    *outShape = *tensor;
    return GRAPH_SUCCESS;
}

static ge::graphStatus Merge(int64_t dim1, int64_t dim2, int64_t& out)
{
    if (dim1 == dim2) {
        out = dim1;
        return GRAPH_SUCCESS;
    }
    if (dim2 == kUnknownDim) {
        out = dim1;
        return GRAPH_SUCCESS;
    }
    if (dim1 == kUnknownDim) {
        out = dim2;
        return GRAPH_SUCCESS;
    }
    return GRAPH_FAILED;
}

static ge::graphStatus Merge(const gert::Shape* s0, const gert::Shape* s1, gert::Shape* out, const std::string& opName)
{
    if (s0 == nullptr || s1 == nullptr || out == nullptr) {
        OP_LOGE(opName, "s0 or s1 or out is nullptr, please check!");
        return GRAPH_FAILED;
    }

    if (IsUnknownRank(*s1)) {
        *out = *s0;
        return GRAPH_SUCCESS;
    }
    if (IsUnknownRank(*s0)) {
        *out = *s1;
        return GRAPH_SUCCESS;
    }
    if (s0->GetDimNum() != s1->GetDimNum()) {
        OP_LOGE(opName, "different rank of first shape %zu and the second shape rank %zu", s0->GetDimNum(),
                s1->GetDimNum());
        return GRAPH_FAILED;
    }

    const size_t rank = s0->GetDimNum();
    out->SetDimNum(rank);
    for (size_t i = 0; i < rank; ++i) {
        int64_t dim = 0;
        if (Merge(s0->GetDim(i), s1->GetDim(i), dim) != GRAPH_SUCCESS) {
            OP_LOGE(opName, "merge %zu th dim failed, first dim value%ld and the second dim value %ld", i,
                    s0->GetDim(i), s1->GetDim(i));
            return GRAPH_FAILED;
        }
        out->SetDim(i, dim);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus SubShapeUpdateStartAndEnd(SubShapePara& para, int64_t sRank, const std::string& opName)
{
    int64_t start = para.start;
    int64_t end = para.end;
    int64_t stride = para.stride;
    if (start > sRank) {
        start = sRank;
    }
    if (end > sRank) {
        end = sRank;
    }
    if (stride < 0 && start == sRank) {
        --start;
    }
    if (start < 0) {
        start += sRank;
        if (start < 0) {
            OP_LOGE(opName, "invalid start[%ld] to get sub shape with rank[%ld]", start - sRank, sRank);
            return GRAPH_FAILED;
        }
    }
    if (end < 0) {
        end += sRank;
        if (end < 0) {
            OP_LOGE(opName, "invalid end[%ld] to get sub shape with rank[%ld]", end - sRank, sRank);
            return GRAPH_FAILED;
        }
    }
    if (stride > 0 && start > end) {
        OP_LOGE(opName, "start[%ld] should be less than end[%ld] at positive stride[%ld]", start, end, stride);
        return GRAPH_FAILED;
    } else if (stride < 0 && start < end) {
        OP_LOGE(opName, "start[%ld] should be greater than end[%ld] at negative stride[%ld]", start, end, stride);
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus SubShape(const gert::Shape* s, SubShapePara& para, gert::Shape* out, const std::string& opName)
{
    int64_t& start = para.start;
    int64_t& end = para.end;
    int64_t stride = para.stride;
    int64_t sRank = static_cast<int64_t>(s->GetDimNum());
    if (sRank > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        OP_LOGE(opName, "rank[%ld] cannot exceed kint32max", sRank);
        return GRAPH_FAILED;
    }
    if (start == 0 && stride == 1 &&
        ((!IsUnknownRank(*s) && end >= sRank) || (end == std::numeric_limits<int64_t>::max()))) {
        *out = *s;
        return GRAPH_SUCCESS;
    }
    OP_LOGE_IF(SubShapeUpdateStartAndEnd(para, sRank, opName) != GRAPH_SUCCESS, ge::GRAPH_FAILED, opName,
               "update start and end failed.");
    out->SetDimNum(0);
    for (int64_t i = start; (stride > 0 ? i < end : i > end); i += stride) {
        out->AppendDim(s->GetDim(i));
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus Concatenate(const gert::Shape* s1, const gert::Shape* s2, gert::Shape* out)
{
    if (s1 == nullptr || s2 == nullptr || out == nullptr) {
        return GRAPH_FAILED;
    }

    size_t s1Rank = s1->GetDimNum();
    size_t s2Rank = s2->GetDimNum();
    size_t rank = s1Rank + s2Rank;

    out->SetDimNum(rank);
    for (size_t i = 0; i < s1Rank; ++i) {
        out->SetDim(i, s1->GetDim(i));
    }
    for (size_t i = 0; i < s2Rank; ++i) {
        out->SetDim(s1Rank + i, s2->GetDim(i));
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus SegmentMeanCheck(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do InferShapeForSparseSegmentMean check");
    gert::Shape outputUnused;
    const gert::Shape* xShape = context->GetInputShape(kInputIndex0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_LOGE_IF(WithRankAtLeast(xShape, kRank1, &outputUnused, context->GetNodeName()) != GRAPH_SUCCESS,
               ge::GRAPH_FAILED, context->GetNodeName(), "Input x should be at least 1-D.");
    const gert::Shape* indicesShape = context->GetInputShape(kInputIndex1);
    OPS_CHECK_NULL_WITH_CONTEXT(context, indicesShape);
    OP_LOGE_IF(WithRank(indicesShape, kRank1, &outputUnused, context->GetNodeName()) != GRAPH_SUCCESS, ge::GRAPH_FAILED,
               context->GetNodeName(), "Input indices must be 1-D.");
    const gert::Shape* segmentIdsShape = context->GetInputShape(kInputIndex2);
    OPS_CHECK_NULL_WITH_CONTEXT(context, segmentIdsShape);
    OP_LOGE_IF(WithRank(segmentIdsShape, kRank1, &outputUnused, context->GetNodeName()) != GRAPH_SUCCESS,
               ge::GRAPH_FAILED, context->GetNodeName(), "Input segment_ids must be 1-D.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeForSparseSegmentMean(gert::InferShapeContext* context)
{
    OP_LOGI(context->GetNodeName(), "Begin to do InferShapeForSparseSegmentMean");
    OP_LOGE_IF(SegmentMeanCheck(context) != GRAPH_SUCCESS, ge::GRAPH_FAILED, context->GetNodeName(), "Check failed.");
    const gert::Shape* xShape = context->GetInputShape(kInputIndex0);
    const gert::Shape* indicesShape = context->GetInputShape(kInputIndex1);
    const gert::Shape* segmentIdsShape = context->GetInputShape(kInputIndex2);
    gert::Shape unusedOut;
    OP_LOGE_IF(Merge(indicesShape, segmentIdsShape, &unusedOut, context->GetNodeName()) != GRAPH_SUCCESS,
               ge::GRAPH_FAILED, context->GetNodeName(), "call merge function failed.");
    int64_t start = 1;
    int64_t stride = 1;
    gert::Shape subShapeOut;
    struct SubShapePara para(start, static_cast<int64_t>(xShape->GetDimNum()), stride);
    OP_LOGE_IF(SubShape(xShape, para, &subShapeOut, context->GetNodeName()) != GRAPH_SUCCESS, ge::GRAPH_FAILED,
               context->GetNodeName(), "call SubShape function failed.");

    const gert::Tensor* segmentIdsTensor = context->GetInputTensor(kInputIndex2);
    if (IsConstTensor(segmentIdsTensor)) {
        OP_LOGE_IF(segmentIdsShape->GetShapeSize() <= 0, ge::GRAPH_FAILED, context->GetNodeName(),
                   "segment_ids must not be empty.");
    }
    gert::Shape constShape;
    bool canGetSegmentIds = GetConstIntToShape<gert::InferShapeContext>(context, kInputIndex2, constShape);
    gert::Shape dim0Shape;
    dim0Shape.SetDimNum(kNum1);
    if (canGetSegmentIds) {
        const int64_t segmentShapeSize = segmentIdsShape->GetShapeSize();
        OPS_CHECK_NULL_WITH_CONTEXT(context, segmentIdsTensor);
        if (segmentIdsTensor->GetDataType() == DT_INT32) {
            auto maxSegmentId = (segmentIdsTensor->GetData<int32_t>())[segmentShapeSize - 1];
            dim0Shape.SetDim(kOutputIndex0, maxSegmentId + 1);
        } else {
            auto maxSegmentId = (segmentIdsTensor->GetData<int64_t>())[segmentShapeSize - 1];
            dim0Shape.SetDim(kOutputIndex0, maxSegmentId + 1);
        }
    } else {
        dim0Shape.SetDim(kOutputIndex0, ge::UNKNOWN_DIM);
    }
    gert::Shape* yShape = context->GetOutputShape(kOutputIndex0);
    OPS_CHECK_NULL_WITH_CONTEXT(context, yShape);
    OP_LOGE_IF(Concatenate(&dim0Shape, &subShapeOut, yShape) != GRAPH_SUCCESS, ge::GRAPH_FAILED, context->GetNodeName(),
               "call Concatenate function failed.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SparseSegmentMean).InputsDataDependency({kInputIndex2}).InferShape(InferShapeForSparseSegmentMean);
} // namespace ops
