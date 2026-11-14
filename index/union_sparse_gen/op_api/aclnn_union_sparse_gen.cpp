/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_union_sparse_gen.h"

#include <algorithm>

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "union_sparse_gen_l0.h"
#include "../op_host/union_sparse_gen_contract.h"
#include "../op_host/union_sparse_gen_dfx.h"
#include "log/log.h"
#include "opdev/data_type_utils.h"

#include <limits>

using namespace op;

namespace Contract = UnionSparseGenContract;

namespace {
using namespace UnionSparseGenDfx;
constexpr const char* kOpName = "UnionSparseGen";

aclnnStatus CheckTensor(const aclTensor* tensor, const char* name, size_t rank, bool optional = false,
                        DataType dtype = DataType::DT_INT32)
{
    if (tensor == nullptr) {
        if (optional) {
            return ACLNN_SUCCESS;
        }
        OP_LOGE_WITH_INVALID_INPUT(kOpName, name);
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (tensor->GetDataType() != dtype) {
        OP_LOGE_FOR_INVALID_DTYPE(kOpName, name, op::ToString(tensor->GetDataType()).GetString(),
                                  op::ToString(dtype).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (IsPrivateFormat(tensor->GetStorageFormat())) {
        OP_LOGE_FOR_INVALID_FORMAT(kOpName, name, op::ToString(tensor->GetStorageFormat()).GetString(),
                                   "public dense format (ND/NCL and other public formats)");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (tensor->GetViewShape().GetDimNum() != rank) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(kOpName, name, std::to_string(tensor->GetViewShape().GetDimNum()),
                                     std::to_string(rank));
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckInputTensors(const aclTensor* const inputs[Contract::kInputCount])
{
    for (size_t i = 0; i < Contract::kInputCount; ++i) {
        const auto status = CheckTensor(inputs[i], kInputNames[i], Contract::kInputRanks[i],
                                        i >= Contract::kFirstOptionalInput,
                                        (Contract::IsPrefixInput(i)) ? DataType::DT_INT64 : DataType::DT_INT32);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckLengthInputs(const aclTensor* const inputs[Contract::kInputCount], int64_t maxQSeqlen,
                              int64_t maxKvSeqlen, int64_t& batch)
{
    const auto* sparseBlockIdx = inputs[Contract::kSparseIdxInput];
    const auto* sparseBlockCount = inputs[Contract::kSparseCountInput];
    const auto* cuSqlenQ = inputs[Contract::kCuQInput];
    const auto* cuSqlenKv = inputs[Contract::kCuKvInput];
    const auto* seqUsedQ = inputs[Contract::kUsedQInput];
    const auto* seqUsedKv = inputs[Contract::kUsedKvInput];
    if (cuSqlenQ == nullptr && seqUsedQ == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(kOpName, "cuSqlenQ,seqUsedQ,seqUsedKv",
                                                 "fixed Q slots require seqUsedQ when cuSqlenQ is null");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (cuSqlenKv == nullptr && seqUsedKv == nullptr) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(kOpName, "cuSqlenKv,seqUsedKv",
                                                 "both are null; provide cuSqlenKv or seqUsedKv");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto& sd = sparseBlockIdx->GetViewShape();
    if (!Contract::InputFits(sd.GetDim(Contract::kHeadAxis), sd.GetDim(Contract::kQueryAxis),
                             sd.GetDim(Contract::kSparseEntryAxis))) {
        OP_LOGE_FOR_INVALID_SHAPE(kOpName, "sparseBlockIdx", ShapeText(sd),
                                  "nonnegative [H,T,K] with int32 byte extent <= INT64_MAX");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!Contract::LengthGeometry(sd.GetDim(Contract::kQueryAxis), cuSqlenQ ? cuSqlenQ->GetViewShape().GetDim(0) : -1,
                                  seqUsedQ ? seqUsedQ->GetViewShape().GetDim(0) : -1, maxQSeqlen, maxKvSeqlen, batch)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            kOpName, "cuSqlenQ,seqUsedQ,T",
            "prefix requires [B+1]; fixed Q requires maxQ>0 and T=B*maxQ; max attrs must be >=-1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const Shape countShape({sd.GetDim(Contract::kHeadAxis), sd.GetDim(Contract::kQueryAxis)});
    if (sparseBlockCount->GetViewShape() != countShape) {
        OP_LOGE_FOR_INVALID_SHAPE(kOpName, "sparseBlockCount", ShapeText(sparseBlockCount->GetViewShape()),
                                  ShapeText(countShape));
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (size_t i = Contract::kCuKvInput; i < Contract::kInputCount; ++i) {
        const int64_t expected = i == Contract::kCuKvInput ? batch + 1 : batch;
        if (inputs[i] != nullptr && inputs[i]->GetViewShape().GetDim(0) != expected) {
            OP_LOGE_FOR_INVALID_SHAPE(kOpName, kInputNames[i], ShapeText(inputs[i]->GetViewShape()),
                                      "[" + std::to_string(expected) + "]");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckScalarAttributes(int64_t unionSize, int64_t maskType, int64_t layout, int64_t residual, bool sorted,
                                  int64_t groups, int64_t logical, int64_t topK, int64_t batch)
{
    const bool valid[] = {unionSize == Contract::kSupportedUnionSize,
                          Contract::SupportsMask(maskType),
                          layout == Contract::kSparsePatternLayout,
                          residual == Contract::kExplicitBlocks,
                          sorted,
                          groups >= 0 && (batch != 0 || groups == 0),
                          logical > 0 && logical <= std::numeric_limits<int32_t>::max(),
                          topK > 0};
    const char* names[] = {"unionSize",        "maskType",       "layoutSparsePattern", "residualBlockMode",
                           "isInputIdxSorted", "totalqUnionNum", "maxLogicalBlocks",    "K"};
    const int64_t values[] = {unionSize, maskType, layout, residual, sorted, groups, logical, topK};
    const char* expected[] = {"8", "0 (noncausal) or 1 (causal)", "4", "0", "true", ">= 0", "[1,INT32_MAX]", "> 0"};
    for (size_t i = 0; i < sizeof(valid) / sizeof(valid[0]); ++i) {
        if (!valid[i]) {
            OP_LOGE_FOR_INVALID_VALUE(kOpName, names[i], std::to_string(values[i]), expected[i]);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckOutputs(const Shape& sd, int64_t blockY, int64_t groups, int64_t capacity, aclTensor* unionIdx,
                         aclTensor* unionCount, aclTensor* mask)
{
    if (capacity > std::numeric_limits<int64_t>::max() / blockY ||
        !Contract::OutputFits(sd.GetDim(Contract::kHeadAxis), groups, capacity, blockY)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            kOpName, "output geometry",
            "H=" + std::to_string(sd.GetDim(Contract::kHeadAxis)) + ", G=" + std::to_string(groups) +
                ", U=" + std::to_string(capacity) + ", blockY=" + std::to_string(blockY) +
                "; all int32 output byte extents and dimensions must fit int64");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const aclTensor* outputs[] = {unionIdx, unionCount, mask};
    const char* outputNames[] = {"unionSparseIdx", "unionSparseCount", "exactMask"};
    const Shape shapes[] = {Shape({sd.GetDim(Contract::kHeadAxis), groups, capacity}),
                            Shape({sd.GetDim(Contract::kHeadAxis), groups}),
                            Shape({sd.GetDim(Contract::kHeadAxis), groups, capacity * blockY})};
    for (size_t i = 0; i < Contract::kOutputCount; ++i) {
        const auto status = CheckTensor(
            outputs[i], outputNames[i],
            i == Contract::kUnionCountOutput ? Contract::kUnionCountRank : Contract::kUnionIdxRank);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
        if (outputs[i]->GetViewShape() != shapes[i]) {
            OP_LOGE_FOR_INVALID_SHAPE(kOpName, outputNames[i], ShapeText(outputs[i]->GetViewShape()),
                                      ShapeText(shapes[i]));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const aclTensor* sparseBlockIdx, const aclTensor* sparseBlockCount, const aclTensor* cuSqlenQ,
                        const aclTensor* cuSqlenKv, const aclTensor* seqUsedQ, const aclTensor* seqUsedKv,
                        const aclIntArray* blockShape, int64_t unionSize, int64_t maskType, int64_t maxQSeqlen,
                        int64_t maxKvSeqlen, int64_t layout, int64_t residual, int64_t groups, int64_t logical,
                        bool sorted, aclTensor* unionIdx, aclTensor* unionCount, aclTensor* mask)
{
    if (blockShape == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(kOpName, "blockShape");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!Contract::SupportsBlockShape(blockShape->GetData(), blockShape->Size())) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(kOpName, "blockShape", "expected two elements [1,128]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    const int64_t blockY = blockShape->GetData()[Contract::kBlockYAxis];
    const aclTensor* inputs[] = {sparseBlockIdx, sparseBlockCount, cuSqlenQ, cuSqlenKv, seqUsedQ, seqUsedKv};
    auto status = CheckInputTensors(inputs);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    int64_t batch = 0;
    status = CheckLengthInputs(inputs, maxQSeqlen, maxKvSeqlen, batch);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    const auto& sd = sparseBlockIdx->GetViewShape();
    status = CheckScalarAttributes(unionSize, maskType, layout, residual, sorted, groups, logical,
                                   sd.GetDim(Contract::kSparseEntryAxis), batch);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    int64_t capacity = 0;
    if (!Contract::Capacity(sd.GetDim(Contract::kSparseEntryAxis), unionSize, logical, residual, capacity)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            kOpName, "K,unionSize,residualBlockMode",
            "K=" + std::to_string(sd.GetDim(Contract::kSparseEntryAxis)) + ", unionSize=" + std::to_string(unionSize) +
                ", residualBlockMode=" + std::to_string(residual) + "; unionSize*(K+residualBlockMode) must fit int64");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return CheckOutputs(sd, blockY, groups, capacity, unionIdx, unionCount, mask);
}
aclnnStatus MakeContiguous(const aclTensor* const inputs[Contract::kInputCount], aclOpExecutor* executor,
                           const aclTensor* contiguous[Contract::kInputCount])
{
    const auto* sparseBlockIdx = inputs[Contract::kSparseIdxInput];
    const auto* sparseBlockCount = inputs[Contract::kSparseCountInput];
    const auto* cuSqlenQOptional = inputs[Contract::kCuQInput];
    const auto* cuSqlenKvOptional = inputs[Contract::kCuKvInput];
    const auto* seqUsedQOptional = inputs[Contract::kUsedQInput];
    const auto* seqUsedKvOptional = inputs[Contract::kUsedKvInput];
    auto sparseIdxContiguous = l0op::Contiguous(sparseBlockIdx, executor);
    auto sparseCountContiguous = l0op::Contiguous(sparseBlockCount, executor);
    auto cuQContiguous = cuSqlenQOptional == nullptr ? nullptr : l0op::Contiguous(cuSqlenQOptional, executor);
    auto cuKvContiguous = cuSqlenKvOptional == nullptr ? nullptr : l0op::Contiguous(cuSqlenKvOptional, executor);
    CHECK_RET(sparseIdxContiguous != nullptr && sparseCountContiguous != nullptr &&
                  (cuSqlenQOptional == nullptr || cuQContiguous != nullptr) &&
                  (cuSqlenKvOptional == nullptr || cuKvContiguous != nullptr),
              ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* seqQContiguous = seqUsedQOptional;
    const aclTensor* seqKvContiguous = seqUsedKvOptional;
    if (seqQContiguous != nullptr) {
        seqQContiguous = l0op::Contiguous(seqQContiguous, executor);
        CHECK_RET(seqQContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (seqKvContiguous != nullptr) {
        seqKvContiguous = l0op::Contiguous(seqKvContiguous, executor);
        CHECK_RET(seqKvContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    contiguous[Contract::kSparseIdxInput] = sparseIdxContiguous;
    contiguous[Contract::kSparseCountInput] = sparseCountContiguous;
    contiguous[Contract::kCuQInput] = cuQContiguous;
    contiguous[Contract::kCuKvInput] = cuKvContiguous;
    contiguous[Contract::kUsedQInput] = seqQContiguous;
    contiguous[Contract::kUsedKvInput] = seqKvContiguous;
    return ACLNN_SUCCESS;
}
aclnnStatus CopyOutputs(const l0op::UnionSparseGenOutputs& outputs, aclTensor* unionSparseIdx,
                        aclTensor* unionSparseCount, aclTensor* exactMask, aclOpExecutor* executor)
{
    CHECK_RET(outputs[Contract::kUnionIdxOutput] != nullptr && outputs[Contract::kUnionCountOutput] != nullptr &&
                  outputs[Contract::kExactMaskOutput] != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(outputs[Contract::kUnionIdxOutput], unionSparseIdx, executor) != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(outputs[Contract::kUnionCountOutput], unionSparseCount, executor) != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(outputs[Contract::kExactMaskOutput], exactMask, executor) != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}
} // namespace

extern "C" {
aclnnStatus aclnnUnionSparseGenGetWorkspaceSize(const aclTensor* sparseBlockIdx, const aclTensor* sparseBlockCount,
                                                const aclTensor* cuSqlenQOptional, const aclTensor* cuSqlenKvOptional,
                                                const aclTensor* seqUsedQOptional, const aclTensor* seqUsedKvOptional,
                                                const aclIntArray* blockShape, int64_t unionSize, int64_t maskType,
                                                int64_t maxQSeqlen, int64_t maxKvSeqlen, int64_t layoutSparsePattern,
                                                bool isInputIdxSorted, int64_t residualBlockMode,
                                                int64_t totalqUnionNum, int64_t maxLogicalBlocks,
                                                aclTensor* unionSparseIdx, aclTensor* unionSparseCount,
                                                aclTensor* exactMask, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    if (workspaceSize == nullptr || executor == nullptr) {
        OP_LOGE_WITH_INVALID_INPUT(kOpName, workspaceSize == nullptr ? "workspaceSize" : "executor");
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    L2_DFX_PHASE_1(aclnnUnionSparseGen,
                   DFX_IN(sparseBlockIdx, sparseBlockCount, cuSqlenQOptional, cuSqlenKvOptional, seqUsedQOptional,
                          seqUsedKvOptional, blockShape, unionSize, maskType, maxQSeqlen, maxKvSeqlen,
                          layoutSparsePattern, isInputIdxSorted, residualBlockMode, totalqUnionNum, maxLogicalBlocks),
                   DFX_OUT(unionSparseIdx, unionSparseCount, exactMask));
    const auto ret = CheckParams(sparseBlockIdx, sparseBlockCount, cuSqlenQOptional, cuSqlenKvOptional,
                                 seqUsedQOptional, seqUsedKvOptional, blockShape, unionSize, maskType, maxQSeqlen,
                                 maxKvSeqlen, layoutSparsePattern, residualBlockMode, totalqUnionNum, maxLogicalBlocks,
                                 isInputIdxSorted, unionSparseIdx, unionSparseCount, exactMask);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    if (totalqUnionNum == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor* inputs[] = {sparseBlockIdx,    sparseBlockCount, cuSqlenQOptional,
                                 cuSqlenKvOptional, seqUsedQOptional, seqUsedKvOptional};
    const aclTensor* contiguous[Contract::kInputCount] = {};
    const auto contiguousStatus = MakeContiguous(inputs, uniqueExecutor.get(), contiguous);
    if (contiguousStatus != ACLNN_SUCCESS) {
        return contiguousStatus;
    }

    aclnnStatus launchStatus = ACLNN_SUCCESS;
    const auto outputs = l0op::UnionSparseGen(
        contiguous[Contract::kSparseIdxInput], contiguous[Contract::kSparseCountInput], contiguous[Contract::kCuQInput],
        contiguous[Contract::kCuKvInput], contiguous[Contract::kUsedQInput], contiguous[Contract::kUsedKvInput],
        blockShape, unionSize, maskType, totalqUnionNum, maxLogicalBlocks, maxQSeqlen, maxKvSeqlen, layoutSparsePattern,
        isInputIdxSorted, residualBlockMode, uniqueExecutor.get(), launchStatus);
    if (launchStatus != ACLNN_SUCCESS) {
        return launchStatus;
    }
    const auto copyStatus = CopyOutputs(outputs, unionSparseIdx, unionSparseCount, exactMask, uniqueExecutor.get());
    if (copyStatus != ACLNN_SUCCESS) {
        return copyStatus;
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUnionSparseGen(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUnionSparseGen);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
}
