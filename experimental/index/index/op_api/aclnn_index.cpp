/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_index_v2.h"

#include <algorithm>
#include <vector>

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "experimental_index_l0.h"
#include "op_api/aclnn_util.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {
constexpr size_t MAX_DIM_LEN = 8;

const std::initializer_list<DataType> VALUE_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT, DataType::DT_DOUBLE, DataType::DT_FLOAT16, DataType::DT_BF16,  DataType::DT_INT16,
    DataType::DT_INT32, DataType::DT_INT64,  DataType::DT_INT8,    DataType::DT_UINT8, DataType::DT_BOOL};

const std::initializer_list<DataType> INDEX_DTYPE_SUPPORT_LIST = {DataType::DT_INT64, DataType::DT_INT32,
                                                                  DataType::DT_BOOL};

bool CheckNotNull(const aclTensor* self, const aclTensorList* indices, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(indices, return false);
    OP_CHECK_NULL(out, return false);
    for (uint64_t i = 0; i < indices->Size(); ++i) {
        OP_CHECK_NULL((*indices)[i], return false);
    }
    return true;
}

bool CheckDtype(const aclTensor* self, const aclTensorList* indices, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(self, VALUE_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);
    for (uint64_t i = 0; i < indices->Size(); ++i) {
        if ((*indices)[i]->GetViewShape().GetShapeSize() == 0) {
            continue;
        }
        OP_CHECK_DTYPE_NOT_SUPPORT((*indices)[i], INDEX_DTYPE_SUPPORT_LIST, return false);
    }
    return true;
}

bool HasBoolIndex(const aclTensorList* indices)
{
    for (uint64_t i = 0; i < indices->Size(); ++i) {
        if ((*indices)[i]->GetDataType() == DataType::DT_BOOL) {
            return true;
        }
    }
    return false;
}

bool IsScalarSelfWithSingleIndex(const aclTensor* self, const aclTensorList* indices)
{
    return self->GetViewShape().GetDimNum() == 0 && indices->Size() == 1;
}

bool HasIndicesBeyondSelfRank(const aclTensor* self, const aclTensorList* indices)
{
    return indices->Size() > self->GetViewShape().GetDimNum();
}

bool CheckFormat(const aclTensor* self, const aclTensorList* indices, const aclTensor* out)
{
    if (IsPrivateFormat(self->GetStorageFormat()) || IsPrivateFormat(out->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Index does not support private format for self or out.");
        return false;
    }
    for (uint64_t i = 0; i < indices->Size(); ++i) {
        if (IsPrivateFormat((*indices)[i]->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Index does not support private format for indices.");
            return false;
        }
    }
    return true;
}

bool BroadcastShape(const aclTensorList* indices, ShapeVector& indexShape)
{
    size_t maxRank = 0;
    for (uint64_t i = 0; i < indices->Size(); ++i) {
        maxRank = std::max(maxRank, (*indices)[i]->GetViewShape().GetDimNum());
    }
    if (maxRank > MAX_DIM_LEN) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "indices rank must be <= %zu.", MAX_DIM_LEN);
        return false;
    }
    indexShape.assign(maxRank, 1);
    for (uint64_t tensorIdx = 0; tensorIdx < indices->Size(); ++tensorIdx) {
        const auto& shape = (*indices)[tensorIdx]->GetViewShape();
        for (size_t i = 0; i < maxRank; ++i) {
            int64_t srcIdx = static_cast<int64_t>(shape.GetDimNum()) - static_cast<int64_t>(maxRank - i);
            int64_t dim = srcIdx < 0 ? 1 : shape.GetDim(static_cast<size_t>(srcIdx));
            if (dim <= 0) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "indices shape dimension must be positive, but got %ld.", dim);
                return false;
            }
            if (indexShape[i] != 1 && dim != 1 && indexShape[i] != dim) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "indices shapes can not broadcast.");
                return false;
            }
            indexShape[i] = std::max(indexShape[i], dim);
        }
    }
    return true;
}

bool InferOutputShape(const aclTensor* self, const aclTensorList* indices, Shape& outputShape)
{
    if (self->GetViewShape().GetDimNum() > MAX_DIM_LEN || indices->Size() == 0 ||
        indices->Size() > self->GetViewShape().GetDimNum()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self rank must be <= %zu and indices size must be in [1, self rank].",
                MAX_DIM_LEN);
        return false;
    }
    ShapeVector indexShape;
    if (!BroadcastShape(indices, indexShape)) {
        return false;
    }
    ShapeVector outShape;
    for (size_t i = 0; i < indexShape.size(); ++i) {
        outShape.emplace_back(indexShape[i]);
    }
    for (size_t i = static_cast<size_t>(indices->Size()); i < self->GetViewShape().GetDimNum(); ++i) {
        outShape.emplace_back(self->GetViewShape().GetDim(i));
    }
    if (outShape.size() > MAX_DIM_LEN) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "output rank must be <= %zu, but got %zu. selfRank=%zu, indicesSize=%lu, broadcastIndexRank=%zu.",
                MAX_DIM_LEN, outShape.size(), self->GetViewShape().GetDimNum(), indices->Size(), indexShape.size());
        return false;
    }
    outputShape.SetDimNum(outShape.size());
    for (size_t i = 0; i < outShape.size(); ++i) {
        outputShape.SetDim(i, outShape[i]);
    }
    return true;
}

bool ShapeEqual(const Shape& lhs, const Shape& rhs)
{
    if (lhs.GetDimNum() != rhs.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < lhs.GetDimNum(); ++i) {
        if (lhs.GetDim(i) != rhs.GetDim(i)) {
            return false;
        }
    }
    return true;
}

bool MakeIndexedMetadata(const aclTensor* self, const aclTensorList* indices, aclOpExecutor* executor,
                         const aclTensor** indexedSizesTensor, const aclTensor** indexedStridesTensor)
{
    std::vector<int64_t> indexedSizes(self->GetViewShape().GetDimNum(), 0);
    for (uint64_t i = 0; i < indices->Size(); ++i) {
        indexedSizes[static_cast<size_t>(i)] = 1;
    }

    std::vector<int64_t> indexedStrides(self->GetViewShape().GetDimNum(), 1);
    int64_t running = 1;
    for (int64_t i = static_cast<int64_t>(self->GetViewShape().GetDimNum()) - 1; i >= 0; --i) {
        indexedStrides[static_cast<size_t>(i)] = running;
        running *= self->GetViewShape().GetDim(static_cast<size_t>(i));
    }

    auto sizesArray = executor->AllocIntArray(indexedSizes.data(), indexedSizes.size());
    auto stridesArray = executor->AllocIntArray(indexedStrides.data(), indexedStrides.size());
    OP_CHECK(sizesArray != nullptr && stridesArray != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Index alloc metadata int arrays failed."), return false);
    *indexedSizesTensor = executor->ConvertToTensor(sizesArray, ToOpDataType(ACL_INT64));
    *indexedStridesTensor = executor->ConvertToTensor(stridesArray, ToOpDataType(ACL_INT64));
    OP_CHECK(*indexedSizesTensor != nullptr && *indexedStridesTensor != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Index convert metadata tensors failed."), return false);
    return true;
}

bool MakeBoolIndexMetadata(const aclTensor* out, const aclTensorList* indices, aclOpExecutor* executor,
                           const aclTensor** indexedSizesTensor, const aclTensor** indexedStridesTensor)
{
    std::vector<int64_t> indexedSizes(indices->Size(), 1);
    std::vector<int64_t> outputShape(out->GetViewShape().GetDimNum(), 1);
    for (size_t i = 0; i < out->GetViewShape().GetDimNum(); ++i) {
        outputShape[i] = out->GetViewShape().GetDim(i);
    }

    auto sizesArray = executor->AllocIntArray(indexedSizes.data(), indexedSizes.size());
    auto stridesArray = executor->AllocIntArray(outputShape.data(), outputShape.size());
    OP_CHECK(sizesArray != nullptr && stridesArray != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Index bool alloc metadata int arrays failed."), return false);
    *indexedSizesTensor = executor->ConvertToTensor(sizesArray, ToOpDataType(ACL_INT64));
    *indexedStridesTensor = executor->ConvertToTensor(stridesArray, ToOpDataType(ACL_INT64));
    OP_CHECK(*indexedSizesTensor != nullptr && *indexedStridesTensor != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Index bool convert metadata tensors failed."), return false);
    return true;
}

aclnnStatus CheckParams(const aclTensor* self, const aclTensorList* indices, const aclTensor* out, Shape& outputShape)
{
    CHECK_RET(CheckNotNull(self, indices, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtype(self, indices, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(self, indices, out), ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return ACLNN_ERR_PARAM_INVALID);
    for (uint64_t i = 0; i < indices->Size(); ++i) {
        OP_CHECK_MAX_DIM((*indices)[i], MAX_DIM_LEN, return ACLNN_ERR_PARAM_INVALID);
    }
    if (self->IsEmpty() || out->IsEmpty()) {
        outputShape = out->GetViewShape();
        return ACLNN_SUCCESS;
    }
    if (IsScalarSelfWithSingleIndex(self, indices)) {
        outputShape = out->GetViewShape();
        return ACLNN_SUCCESS;
    }
    if (HasBoolIndex(indices) || HasIndicesBeyondSelfRank(self, indices)) {
        return ACLNN_SUCCESS;
    }
    CHECK_RET(InferOutputShape(self, indices, outputShape), ACLNN_ERR_PARAM_INVALID);
    if (!ShapeEqual(outputShape, out->GetViewShape())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out shape must match inferred Index output shape.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}
} // namespace

extern "C" {
aclnnStatus aclnnIndexGetWorkspaceSize(const aclTensor* self, const aclTensorList* indices, aclTensor* out,
                                       uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnIndex, DFX_IN(self, indices), DFX_OUT(out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    Shape outputShape;
    auto ret = CheckParams(self, indices, out, outputShape);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    const bool hasBoolIndex = HasBoolIndex(indices);
    const bool indicesBeyondSelfRank = HasIndicesBeyondSelfRank(self, indices);
    if (hasBoolIndex || indicesBeyondSelfRank) {
        outputShape = out->GetViewShape();
    }
    if (indicesBeyondSelfRank) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    if (self->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    std::vector<const aclTensor*> indexTensors;
    indexTensors.reserve(indices->Size());
    for (uint64_t i = 0; i < indices->Size(); ++i) {
        auto indexContiguous = l0op::Contiguous((*indices)[i], uniqueExecutor.get());
        CHECK_RET(indexContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        indexTensors.emplace_back(indexContiguous);
    }
    auto indicesList = uniqueExecutor->AllocTensorList(indexTensors.data(), indexTensors.size());
    CHECK_RET(indicesList != nullptr, ACLNN_ERR_INNER_NULLPTR);

    if (IsScalarSelfWithSingleIndex(selfContiguous, indicesList)) {
        auto viewCopyResult = l0op::ViewCopy(selfContiguous, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    const aclTensor* indexedSizes = nullptr;
    const aclTensor* indexedStrides = nullptr;
    if (hasBoolIndex || indicesBeyondSelfRank) {
        CHECK_RET(MakeBoolIndexMetadata(out, indicesList, uniqueExecutor.get(), &indexedSizes, &indexedStrides),
                  ACLNN_ERR_INNER_NULLPTR);
    } else {
        CHECK_RET(
            MakeIndexedMetadata(selfContiguous, indicesList, uniqueExecutor.get(), &indexedSizes, &indexedStrides),
            ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor* indexOut = (hasBoolIndex || indicesBeyondSelfRank) ?
                                    l0op::IndexAiCpu(selfContiguous, indexedSizes, indexedStrides, outputShape,
                                                     indicesList, uniqueExecutor.get()) :
                                    l0op::IndexAiCore(selfContiguous, indexedSizes, indexedStrides, outputShape,
                                                      indicesList, uniqueExecutor.get());
    CHECK_RET(indexOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(indexOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnIndex(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnIndex);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
}
