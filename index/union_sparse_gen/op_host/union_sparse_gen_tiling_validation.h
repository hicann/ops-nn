/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UNION_SPARSE_GEN_TILING_VALIDATION_H
#define UNION_SPARSE_GEN_TILING_VALIDATION_H
#include <limits>
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "union_sparse_gen_contract.h"
#include "union_sparse_gen_geometry.h"
#include "union_sparse_gen_dfx.h"

namespace UnionSparseGenTilingValidation {
namespace Contract = UnionSparseGenContract;
constexpr const char* kOpName = "UnionSparseGen";

inline ge::graphStatus CheckInputDescriptors(const gert::TilingContext* context)
{
    for (size_t i = 0; i < Contract::kInputCount; ++i) {
        const auto* desc = i >= Contract::kFirstOptionalInput ? context->GetOptionalInputDesc(i) :
                                                                context->GetInputDesc(i);
        if (i >= Contract::kFirstOptionalInput && desc == nullptr) {
            continue;
        }
        OP_CHECK_NULL_WITH_CONTEXT(context, desc);
        const auto* shape = i >= Contract::kFirstOptionalInput ? context->GetOptionalInputShape(i) :
                                                                 context->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, shape);
        const auto dtype = (Contract::IsPrefixInput(i)) ? ge::DT_INT64 : ge::DT_INT32;
        if (desc->GetDataType() != dtype || shape->GetStorageShape().GetDimNum() != Contract::kInputRanks[i]) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                kOpName, UnionSparseGenDfx::kInputNames[i],
                (Contract::IsPrefixInput(i) ? "expected DT_INT64 rank " : "expected DT_INT32 rank ") +
                    std::to_string(Contract::kInputRanks[i]) + ", got " +
                    UnionSparseGenDfx::ShapeText(shape->GetStorageShape()));
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckLengthExtents(const gert::TilingContext* context, int64_t batch)
{
    for (size_t i = Contract::kFirstOptionalInput; i < Contract::kInputCount; ++i) {
        const auto* shape = context->GetOptionalInputShape(i);
        const int64_t expected = i < Contract::kFirstUsedLengthInput ? batch + 1 : batch;
        if (shape != nullptr && shape->GetStorageShape().GetDim(Contract::kLengthAxis) != expected) {
            OP_LOGE_FOR_INVALID_SHAPE(kOpName, UnionSparseGenDfx::kInputNames[i],
                                      UnionSparseGenDfx::ShapeText(shape->GetStorageShape()),
                                      "[" + std::to_string(expected) + "]");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckInputGeometry(const gert::TilingContext* context, const int64_t* maxQ, const int64_t* maxKv,
                                          int64_t& batch)
{
    const auto* sparseShape = context->GetInputShape(Contract::kSparseIdxInput);
    const auto* countShape = context->GetInputShape(Contract::kSparseCountInput);
    const auto* qShape = context->GetOptionalInputShape(Contract::kCuQInput);
    const auto* kvShape = context->GetOptionalInputShape(Contract::kCuKvInput);
    const auto* usedQShape = context->GetOptionalInputShape(Contract::kUsedQInput);
    const auto* usedKvShape = context->GetOptionalInputShape(Contract::kUsedKvInput);
    if ((qShape == nullptr && usedQShape == nullptr) || (kvShape == nullptr && usedKvShape == nullptr)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(kOpName, "cuSqlenQ,seqUsedQ,cuSqlenKv,seqUsedKv",
                                                 "fixed Q requires seqUsedQ; KV requires cuSqlenKv or seqUsedKv");
        return ge::GRAPH_FAILED;
    }
    const auto& sd = sparseShape->GetStorageShape();
    const auto& cd = countShape->GetStorageShape();
    if (!Contract::InputFits(sd.GetDim(Contract::kHeadAxis), sd.GetDim(Contract::kQueryAxis),
                             sd.GetDim(Contract::kSparseEntryAxis)) ||
        !Contract::LengthGeometry(
            sd.GetDim(Contract::kQueryAxis), qShape ? qShape->GetStorageShape().GetDim(Contract::kLengthAxis) : -1,
            usedQShape ? usedQShape->GetStorageShape().GetDim(Contract::kLengthAxis) : -1, *maxQ, *maxKv, batch)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(kOpName, "cuSqlenQ,seqUsedQ,T",
                                                 "invalid input bytes or prefix/fixed-slot batch geometry");
        return ge::GRAPH_FAILED;
    }
    if (cd.GetDim(Contract::kHeadAxis) != sd.GetDim(Contract::kHeadAxis) ||
        cd.GetDim(Contract::kQueryAxis) != sd.GetDim(Contract::kQueryAxis)) {
        OP_LOGE_FOR_INVALID_SHAPE(kOpName, "sparseBlockCount", UnionSparseGenDfx::ShapeText(cd), "[H,T]");
        return ge::GRAPH_FAILED;
    }
    return CheckLengthExtents(context, batch);
}

inline ge::graphStatus CheckScalarGeometry(const gert::TilingContext* context, int64_t blockX, int64_t blockY,
                                           int64_t unionSize, int64_t maskType, int64_t batch, int64_t numGroups,
                                           int64_t maxLogicalBlocks)
{
    const auto& sd = context->GetInputShape(Contract::kSparseIdxInput)->GetStorageShape();
    if (blockX != Contract::kSupportedBlockX || blockY != Contract::kSupportedBlockY ||
        unionSize != Contract::kSupportedUnionSize || !Contract::SupportsMask(maskType)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(kOpName, "blockX,blockY,unionSize,maskType",
                                                 "actual=" + std::to_string(blockX) + "," + std::to_string(blockY) +
                                                     "," + std::to_string(unionSize) + "," + std::to_string(maskType) +
                                                     "; expected 1,128,8,maskType=0/1");
        return ge::GRAPH_FAILED;
    }
    if ((batch == 0 && numGroups != 0) || numGroups < 0 || maxLogicalBlocks <= 0 ||
        sd.GetDim(Contract::kSparseEntryAxis) <= 0) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            kOpName, "totalqUnionNum,maxLogicalBlocks,K",
            "actual=" + std::to_string(numGroups) + "," + std::to_string(maxLogicalBlocks) + "," +
                std::to_string(sd.GetDim(Contract::kSparseEntryAxis)) + "; expected >=0,>0,>0");
        return ge::GRAPH_FAILED;
    }
    if (maxLogicalBlocks > std::numeric_limits<int32_t>::max()) {
        OP_LOGE_FOR_INVALID_VALUE(kOpName, "maxLogicalBlocks", std::to_string(maxLogicalBlocks), "[1,INT32_MAX]");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckCapacity(const gert::TilingContext* context, int64_t blockX, int64_t blockY,
                                     int64_t unionSize, int64_t maskType, int64_t numGroups, int64_t maxLogicalBlocks,
                                     int64_t& capacity)
{
    const auto& sd = context->GetInputShape(Contract::kSparseIdxInput)->GetStorageShape();
    const auto* contractAttrs = context->GetAttrs();
    const auto* residual = contractAttrs->GetAttrPointer<int64_t>(Contract::kResidual);
    const auto* layout = contractAttrs->GetAttrPointer<int64_t>(Contract::kLayout);
    const auto* sorted = contractAttrs->GetAttrPointer<bool>(Contract::kSorted);
    if (residual == nullptr || layout == nullptr || sorted == nullptr ||
        !Contract::SupportsSpec(blockX, blockY, unionSize, maskType, *layout, *sorted, *residual) ||
        !Contract::Capacity(sd.GetDim(Contract::kSparseEntryAxis), unionSize, maxLogicalBlocks, *residual, capacity) ||
        capacity > std::numeric_limits<int64_t>::max() / blockY ||
        !Contract::OutputFits(sd.GetDim(Contract::kHeadAxis), numGroups, capacity, blockY)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            kOpName, "length/layout/capacity contract",
            "layoutSparsePattern=" + (layout ? std::to_string(*layout) : "absent") + ", residualBlockMode=" +
                (residual ? std::to_string(*residual) : "absent") + ", capacity=" + std::to_string(capacity) +
                ", H=" + std::to_string(sd.GetDim(Contract::kHeadAxis)) + ", G=" + std::to_string(numGroups) +
                ", K=" + std::to_string(sd.GetDim(Contract::kSparseEntryAxis)) + ", W=" + std::to_string(unionSize) +
                ", Lmax=" + std::to_string(maxLogicalBlocks) + ", Y=" + std::to_string(blockY) +
                "; expected Y=128, W=8, layout=4, sorted=true, residual=0, capacity=min(W*(K+residual),Lmax), "
                "int64-safe bytes");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus CheckOutputShapes(const gert::TilingContext* context, int64_t blockY, int64_t numGroups,
                                         int64_t capacity)
{
    const auto& sd = context->GetInputShape(Contract::kSparseIdxInput)->GetStorageShape();
    const auto* unionOutput = context->GetOutputShape(Contract::kUnionIdxOutput);
    const auto* countOutput = context->GetOutputShape(Contract::kUnionCountOutput);
    const auto* maskOutput = context->GetOutputShape(Contract::kExactMaskOutput);
    OP_CHECK_NULL_WITH_CONTEXT(context, unionOutput);
    OP_CHECK_NULL_WITH_CONTEXT(context, countOutput);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskOutput);
    if (unionOutput->GetStorageShape().GetDimNum() != Contract::kUnionIdxRank ||
        unionOutput->GetStorageShape().GetDim(Contract::kHeadAxis) != sd.GetDim(Contract::kHeadAxis) ||
        unionOutput->GetStorageShape().GetDim(Contract::kGroupAxis) != numGroups ||
        unionOutput->GetStorageShape().GetDim(Contract::kUnionEntryAxis) != capacity ||
        countOutput->GetStorageShape().GetDimNum() != Contract::kUnionCountRank ||
        countOutput->GetStorageShape().GetDim(Contract::kHeadAxis) != sd.GetDim(Contract::kHeadAxis) ||
        countOutput->GetStorageShape().GetDim(Contract::kGroupAxis) != numGroups ||
        maskOutput->GetStorageShape().GetDimNum() != Contract::kExactMaskRank ||
        maskOutput->GetStorageShape().GetDim(Contract::kHeadAxis) != sd.GetDim(Contract::kHeadAxis) ||
        maskOutput->GetStorageShape().GetDim(Contract::kGroupAxis) != numGroups ||
        maskOutput->GetStorageShape().GetDim(Contract::kMaskTokenAxis) != capacity * blockY) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            kOpName, "output shapes",
            "unionSparseIdx=" + UnionSparseGenDfx::ShapeText(unionOutput->GetStorageShape()) +
                ", unionSparseCount=" + UnionSparseGenDfx::ShapeText(countOutput->GetStorageShape()) +
                ", exactMask=" + UnionSparseGenDfx::ShapeText(maskOutput->GetStorageShape()) +
                "; expected [H,G,U], [H,G], [H,G,U*Y] with H=" + std::to_string(sd.GetDim(Contract::kHeadAxis)) +
                ", G=" + std::to_string(numGroups) + ", U=" + std::to_string(capacity) +
                ", Y=" + std::to_string(blockY));
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus ValidateInputs(const gert::TilingContext* context, int64_t blockX, int64_t blockY,
                                      int64_t unionSize, int64_t maskType, int64_t& capacity, int64_t numGroups,
                                      int64_t maxLogicalBlocks)
{
    const auto* lengthAttrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, lengthAttrs);
    const auto* maxQ = lengthAttrs->GetAttrPointer<int64_t>(Contract::kMaxQ);
    const auto* maxKv = lengthAttrs->GetAttrPointer<int64_t>(Contract::kMaxKv);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxQ);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxKv);
    int64_t batch = 0;
    if (CheckInputDescriptors(context) != ge::GRAPH_SUCCESS ||
        CheckInputGeometry(context, maxQ, maxKv, batch) != ge::GRAPH_SUCCESS ||
        CheckScalarGeometry(context, blockX, blockY, unionSize, maskType, batch, numGroups, maxLogicalBlocks) !=
            ge::GRAPH_SUCCESS ||
        CheckCapacity(context, blockX, blockY, unionSize, maskType, numGroups, maxLogicalBlocks, capacity) !=
            ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return CheckOutputShapes(context, blockY, numGroups, capacity);
}
struct TilingSpec {
    int64_t blockX = 0, blockY = 0, unionSize = 0, maskType = 0;
    int64_t groups = 0, logicalBlocks = 0, capacity = 0;
};

inline ge::graphStatus GetAttrInt(const gert::TilingContext* context, size_t index, const char* name, int64_t& value)
{
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* attr = attrs->GetAttrPointer<int64_t>(index);
    if (attr == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(kOpName, name);
        return ge::GRAPH_FAILED;
    }
    value = *attr;
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus ResolveTilingGeometry(gert::TilingContext* context, int64_t width, int64_t blockY,
                                             int64_t& numGroups, int64_t& maxLogicalBlocks)
{
    if (numGroups != Contract::kUnspecifiedGeometry && maxLogicalBlocks != Contract::kUnspecifiedGeometry) {
        return ge::GRAPH_SUCCESS;
    }
    const auto* attrs = context->GetAttrs();
    if (CheckInputDescriptors(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // Inference has already resolved the exact output shape. Only an
    // unsaturated U identifies Lmax; saturated U still needs KV values.
    const auto* countShape = context->GetOutputShape(Contract::kUnionCountOutput);
    const auto* idxShape = context->GetOutputShape(Contract::kUnionIdxOutput);
    OP_CHECK_NULL_WITH_CONTEXT(context, countShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, idxShape);
    if (countShape->GetStorageShape().GetDimNum() != Contract::kUnionCountRank ||
        idxShape->GetStorageShape().GetDimNum() != Contract::kUnionIdxRank) {
        return ge::GRAPH_FAILED;
    }
    if (numGroups == Contract::kUnspecifiedGeometry) {
        numGroups = countShape->GetStorageShape().GetDim(Contract::kGroupAxis);
    }
    const auto
        topK = context->GetInputShape(Contract::kSparseIdxInput)->GetStorageShape().GetDim(Contract::kSparseEntryAxis);
    const auto outputCapacity = idxShape->GetStorageShape().GetDim(Contract::kUnionEntryAxis);
    if (maxLogicalBlocks == Contract::kUnspecifiedGeometry && topK > 0 &&
        topK <= std::numeric_limits<int64_t>::max() / width && outputCapacity > 0 && outputCapacity < topK * width) {
        maxLogicalBlocks = outputCapacity;
    }
    const auto* maxQ = attrs->GetInt(Contract::kMaxQ);
    const auto* maxKv = attrs->GetInt(Contract::kMaxKv);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxQ);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxKv);
    int64_t batch = 0;
    if (CheckInputGeometry(context, maxQ, maxKv, batch) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    UnionSparseGenGeometry::Result result{numGroups, maxLogicalBlocks};
    const auto
        totalQ = context->GetInputShape(Contract::kSparseIdxInput)->GetStorageShape().GetDim(Contract::kQueryAxis);
    if (!UnionSparseGenGeometry::Resolve(context, batch, totalQ, *maxQ, width, blockY, result) || result.groups < 0 ||
        result.logical <= 0) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(kOpName, "geometry",
                                                 "runtime length values or exact output shapes are unavailable");
        return ge::GRAPH_FAILED;
    }
    numGroups = result.groups;
    maxLogicalBlocks = result.logical;
    return ge::GRAPH_SUCCESS;
}

inline ge::graphStatus ReadTilingSpec(gert::TilingContext* context, TilingSpec& spec)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* blockShape = attrs->GetListInt(Contract::kBlockShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, blockShape);
    if (!Contract::SupportsBlockShape(blockShape->GetData(), blockShape->GetSize())) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(kOpName, "blockShape", "expected two elements [1,128]");
        return ge::GRAPH_FAILED;
    }
    const int64_t* blockX = blockShape->GetData();
    const int64_t* blockY = blockShape->GetData() + Contract::kBlockYAxis;
    const int64_t* unionSize = attrs->GetAttrPointer<int64_t>(Contract::kUnionSize);
    const int64_t* maskType = attrs->GetAttrPointer<int64_t>(Contract::kMaskType);
    OP_CHECK_NULL_WITH_CONTEXT(context, blockX);
    OP_CHECK_NULL_WITH_CONTEXT(context, blockY);
    OP_CHECK_NULL_WITH_CONTEXT(context, unionSize);
    OP_CHECK_NULL_WITH_CONTEXT(context, maskType);
    if (*unionSize != Contract::kSupportedUnionSize || !Contract::SupportsMask(*maskType)) {
        return ge::GRAPH_FAILED;
    }
    int64_t& capacity = spec.capacity;
    int64_t numGroups = 0;
    int64_t maxLogicalBlocks = 0;
    if (GetAttrInt(context, Contract::kNumGroups, "totalq_union_num", numGroups) != ge::GRAPH_SUCCESS ||
        GetAttrInt(context, Contract::kMaxLogical, "max_logical_blocks", maxLogicalBlocks) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ResolveTilingGeometry(context, *unionSize, *blockY, numGroups, maxLogicalBlocks) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ValidateInputs(context, *blockX, *blockY, *unionSize, *maskType, capacity, numGroups, maxLogicalBlocks) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    spec.blockX = *blockX;
    spec.blockY = *blockY;
    spec.unionSize = *unionSize;
    spec.maskType = *maskType;
    spec.groups = numGroups;
    spec.logicalBlocks = maxLogicalBlocks;
    return ge::GRAPH_SUCCESS;
}
} // namespace UnionSparseGenTilingValidation
#endif
