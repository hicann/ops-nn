/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "union_sparse_gen_contract.h"
#include "union_sparse_gen_geometry.h"
#include "union_sparse_gen_dfx.h"
#include <limits>

using namespace ge;

namespace Contract = UnionSparseGenContract;

namespace ops {
namespace {
constexpr const char* kOpName = "UnionSparseGen";

graphStatus CheckInputShape(const gert::InferShapeContext* context, size_t index, size_t rank)
{
    const gert::Shape* shape = context->GetInputShape(index);
    OP_CHECK_NULL_WITH_CONTEXT(context, shape);
    if (shape->GetDimNum() != rank) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(kOpName, UnionSparseGenDfx::kInputNames[index], std::to_string(shape->GetDimNum()),
                                     std::to_string(rank));
        return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}
} // namespace

struct InferredGeometry {
    int64_t heads = 0, groups = 0, capacity = 0, logical = 0;
    int64_t blockX = 0, blockY = 0, unionSize = 0, maskType = 0;
};

static graphStatus CheckLengthGeometry(gert::InferShapeContext* context, int64_t& batch,
                                       const gert::RuntimeAttrs*& attrs)
{
    const auto* qShape = context->GetOptionalInputShape(Contract::kCuQInput);
    const auto* kvShape = context->GetOptionalInputShape(Contract::kCuKvInput);
    const auto* usedQShape = context->GetOptionalInputShape(Contract::kUsedQInput);
    const auto* usedKvShape = context->GetOptionalInputShape(Contract::kUsedKvInput);
    for (size_t i = Contract::kFirstOptionalInput; i < Contract::kInputCount; ++i) {
        const auto* shape = context->GetOptionalInputShape(i);
        if (shape != nullptr && shape->GetDimNum() != Contract::kLengthRank) {
            OP_LOGE_FOR_INVALID_SHAPE(kOpName, UnionSparseGenDfx::kInputNames[i], UnionSparseGenDfx::ShapeText(*shape),
                                      "rank-1 length tensor");
            return GRAPH_FAILED;
        }
    }
    if ((qShape == nullptr && usedQShape == nullptr) || (kvShape == nullptr && usedKvShape == nullptr)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(kOpName, "length inputs",
                                                 "fixed Q requires seqUsedQ; KV requires cuSqlenKv or seqUsedKv");
        return GRAPH_FAILED;
    }
    attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* maxQ = attrs->GetAttrPointer<int64_t>(Contract::kMaxQ);
    const auto* maxKv = attrs->GetAttrPointer<int64_t>(Contract::kMaxKv);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxQ);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxKv);
    if (!Contract::LengthGeometry(context->GetInputShape(Contract::kSparseIdxInput)->GetDim(Contract::kQueryAxis),
                                  qShape ? qShape->GetDim(Contract::kLengthAxis) : -1,
                                  usedQShape ? usedQShape->GetDim(Contract::kLengthAxis) : -1, *maxQ, *maxKv, batch)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(kOpName, "cuSqlenQ,seqUsedQ,T",
                                                 "invalid prefix/fixed-slot batch geometry");
        return GRAPH_FAILED;
    }
    for (size_t i = Contract::kFirstOptionalInput; i < Contract::kInputCount; ++i) {
        const auto* shape = context->GetOptionalInputShape(i);
        if (shape != nullptr &&
            shape->GetDim(Contract::kLengthAxis) != batch + (i < Contract::kFirstUsedLengthInput ? 1 : 0)) {
            OP_LOGE_FOR_INVALID_SHAPE(kOpName, UnionSparseGenDfx::kInputNames[i], UnionSparseGenDfx::ShapeText(*shape),
                                      "matching batch length");
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;
}

static graphStatus ReadGeometryAttributes(gert::InferShapeContext* context, const gert::RuntimeAttrs* attrs,
                                          int64_t batch, InferredGeometry& geometry)
{
    const int64_t* maskType = attrs->GetAttrPointer<int64_t>(Contract::kMaskType);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskType);
    if (!Contract::SupportsMask(*maskType)) {
        OP_LOGE_FOR_INVALID_VALUE(kOpName, "maskType", std::to_string(*maskType), "0 (noncausal) or 1 (causal)");
        return GRAPH_FAILED;
    }
    const auto* blockShape = attrs->GetListInt(Contract::kBlockShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, blockShape);
    if (!Contract::SupportsBlockShape(blockShape->GetData(), blockShape->GetSize())) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(kOpName, "blockShape", "expected two elements [1,128]");
        return GRAPH_FAILED;
    }
    const int64_t* blockY = blockShape->GetData() + Contract::kBlockYAxis;
    const int64_t* unionSize = attrs->GetAttrPointer<int64_t>(Contract::kUnionSize);
    const int64_t* numGroups = attrs->GetAttrPointer<int64_t>(Contract::kNumGroups);
    OP_CHECK_NULL_WITH_CONTEXT(context, blockY);
    OP_CHECK_NULL_WITH_CONTEXT(context, unionSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, numGroups);
    if (*numGroups < Contract::kUnspecifiedGeometry || (batch == 0 && *numGroups > 0)) {
        OP_LOGE_FOR_INVALID_VALUE(kOpName, "totalqUnionNum", std::to_string(*numGroups),
                                  "G>=0 or -1 (infer); B=0 requires G=0 or -1");
        return GRAPH_FAILED;
    }
    geometry.blockX = blockShape->GetData()[Contract::kBlockXAxis];
    geometry.blockY = *blockY;
    geometry.unionSize = *unionSize;
    geometry.maskType = *maskType;
    geometry.groups = *numGroups;
    return GRAPH_SUCCESS;
}

static graphStatus CheckCapacityGeometry(gert::InferShapeContext* context, const gert::RuntimeAttrs* attrs,
                                         InferredGeometry& geometry)
{
    const auto* logical = attrs->GetAttrPointer<int64_t>(Contract::kMaxLogical);
    const auto* layout = attrs->GetAttrPointer<int64_t>(Contract::kLayout);
    const auto* residual = attrs->GetAttrPointer<int64_t>(Contract::kResidual);
    const auto* sorted = attrs->GetAttrPointer<bool>(Contract::kSorted);
    int64_t& capacity = geometry.capacity;
    if (!logical || !layout || !residual || !sorted ||
        !Contract::SupportsSpec(geometry.blockX, geometry.blockY, geometry.unionSize, geometry.maskType, *layout,
                                *sorted, *residual) ||
        !Contract::Capacity(context->GetInputShape(Contract::kSparseIdxInput)->GetDim(Contract::kSparseEntryAxis),
                            geometry.unionSize, *logical == Contract::kUnspecifiedGeometry ? 1 : *logical, *residual,
                            capacity) ||
        capacity > std::numeric_limits<int64_t>::max() / geometry.blockY) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            kOpName, "length/layout/capacity contract",
            "layoutSparsePattern=" + (layout ? std::to_string(*layout) : "absent") +
                ", residualBlockMode=" + (residual ? std::to_string(*residual) : "absent") + ", maxLogicalBlocks=" +
                (logical ? std::to_string(*logical) : "absent") + ", W=" + std::to_string(geometry.unionSize) + ", K=" +
                std::to_string(context->GetInputShape(Contract::kSparseIdxInput)->GetDim(Contract::kSparseEntryAxis)) +
                ", capacity=" + std::to_string(capacity) +
                "; expected X=1, Y=128, W=8, layout=4, sorted=true, residual=0, K>0, 1<=Lmax<=INT32_MAX, "
                "U=min(W*(K+residual),Lmax)");
        return GRAPH_FAILED;
    }

    geometry.logical = *logical;
    if (*logical == Contract::kUnspecifiedGeometry) {
        geometry.capacity = Contract::kUnspecifiedGeometry;
    }
    return GRAPH_SUCCESS;
}

static graphStatus ResolveInferredGeometry(gert::InferShapeContext* context, int64_t batch, InferredGeometry& geometry)
{
    if (geometry.groups >= 0 && geometry.logical > 0) {
        return GRAPH_SUCCESS;
    }
    UnionSparseGenGeometry::Result result{geometry.groups, geometry.logical};
    const auto* attrs = context->GetAttrs();
    const auto totalQ = context->GetInputShape(Contract::kSparseIdxInput)->GetDim(Contract::kQueryAxis);
    const auto maxQ = *attrs->GetInt(Contract::kMaxQ);
    if (!UnionSparseGenGeometry::Resolve(context, batch, totalQ, maxQ, geometry.unionSize, geometry.blockY, result)) {
        return GRAPH_FAILED;
    }
    geometry.groups = result.groups;
    geometry.logical = result.logical;
    if (result.logical != Contract::kUnspecifiedGeometry) {
        const auto topK = context->GetInputShape(Contract::kSparseIdxInput)->GetDim(Contract::kSparseEntryAxis);
        if (!Contract::Capacity(topK, geometry.unionSize, result.logical, Contract::kExplicitBlocks,
                                geometry.capacity)) {
            return GRAPH_FAILED;
        }
    }
    return GRAPH_SUCCESS;
}

static graphStatus SetOutputGeometry(gert::InferShapeContext* context, const InferredGeometry& geometry)
{
    gert::Shape* unionShape = context->GetOutputShape(Contract::kUnionIdxOutput);
    gert::Shape* countShape = context->GetOutputShape(Contract::kUnionCountOutput);
    gert::Shape* maskShape = context->GetOutputShape(Contract::kExactMaskOutput);
    OP_CHECK_NULL_WITH_CONTEXT(context, unionShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, countShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskShape);

    unionShape->SetDimNum(Contract::kUnionIdxRank);
    unionShape->SetDim(Contract::kHeadAxis, geometry.heads);
    unionShape->SetDim(Contract::kGroupAxis, geometry.groups);
    unionShape->SetDim(Contract::kUnionEntryAxis, geometry.capacity);
    countShape->SetDimNum(Contract::kUnionCountRank);
    countShape->SetDim(Contract::kHeadAxis, geometry.heads);
    countShape->SetDim(Contract::kGroupAxis, geometry.groups);
    maskShape->SetDimNum(Contract::kExactMaskRank);
    maskShape->SetDim(Contract::kHeadAxis, geometry.heads);
    maskShape->SetDim(Contract::kGroupAxis, geometry.groups);
    maskShape->SetDim(Contract::kMaskTokenAxis,
                      geometry.capacity < 0 ? Contract::kUnspecifiedGeometry : geometry.capacity * geometry.blockY);
    return GRAPH_SUCCESS;
}

static graphStatus InferShapeUnionSparseGen(gert::InferShapeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    if (CheckInputShape(context, Contract::kSparseIdxInput, Contract::kSparseIdxRank) != GRAPH_SUCCESS ||
        CheckInputShape(context, Contract::kSparseCountInput, Contract::kSparseCountRank) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    const gert::RuntimeAttrs* attrs = nullptr;
    int64_t batch = 0;
    InferredGeometry geometry;
    if (CheckLengthGeometry(context, batch, attrs) != GRAPH_SUCCESS ||
        ReadGeometryAttributes(context, attrs, batch, geometry) != GRAPH_SUCCESS ||
        CheckCapacityGeometry(context, attrs, geometry) != GRAPH_SUCCESS ||
        ResolveInferredGeometry(context, batch, geometry) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    const gert::Shape* sparseShape = context->GetInputShape(Contract::kSparseIdxInput);
    geometry.heads = sparseShape->GetDim(Contract::kHeadAxis);
    if (!Contract::InputFits(geometry.heads, sparseShape->GetDim(Contract::kQueryAxis),
                             sparseShape->GetDim(Contract::kSparseEntryAxis)) ||
        ((geometry.groups >= 0 && geometry.capacity >= 0) &&
         !Contract::OutputFits(geometry.heads, geometry.groups, geometry.capacity, geometry.blockY))) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            kOpName, "geometry",
            "sparseBlockIdx=" + UnionSparseGenDfx::ShapeText(*sparseShape) + ", G=" + std::to_string(geometry.groups) +
                ", U=" + std::to_string(geometry.capacity) + ", Y=" + std::to_string(geometry.blockY) +
                "; dimensions must be nonnegative and int32 byte extents must fit int64");
        return GRAPH_FAILED;
    }
    return SetOutputGeometry(context, geometry);
}

static graphStatus InferDataTypeUnionSparseGen(gert::InferDataTypeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    context->SetOutputDataType(Contract::kUnionIdxOutput, DT_INT32);
    context->SetOutputDataType(Contract::kUnionCountOutput, DT_INT32);
    context->SetOutputDataType(Contract::kExactMaskOutput, DT_INT32);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(UnionSparseGen)
    .InferShape(InferShapeUnionSparseGen)
    .InferDataType(InferDataTypeUnionSparseGen)
    .InputsDataDependency({Contract::kCuQInput, Contract::kCuKvInput, Contract::kUsedQInput, Contract::kUsedKvInput});
} // namespace ops
