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
 * \file index_put_with_sort_v2.cpp
 * \brief
 */
#include "index_put_with_sort_v2.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
using namespace op;

namespace l0op {
OP_TYPE_REGISTER(IndexPutWithSortV2);

static bool CheckBasicTensorProperties(const aclTensor* self, const aclTensorList* indices, const aclTensor* value) {
    if (!self || !value) {
        return false;
    }
    auto selfDtype = self->GetDataType();
    auto valueDtype = value->GetDataType();
    if (selfDtype != valueDtype)
        return false;
    static constexpr int SELF_DATATYPE_COUNT = 8;
    static constexpr std::array<op::DataType, SELF_DATATYPE_COUNT> selfSupportedDtypes = {
        op::DataType::DT_INT64,   op::DataType::DT_INT32,
        op::DataType::DT_FLOAT,   op::DataType::DT_FLOAT16,
        op::DataType::DT_BF16,    op::DataType::DT_INT8,
        op::DataType::DT_UINT8,   op::DataType::DT_BOOL
    };
    auto it = std::find(selfSupportedDtypes.begin(), selfSupportedDtypes.end(), selfDtype);
    if (it == selfSupportedDtypes.end()) {
        return false;
    }
    if (!indices || indices->Size() < 1 || !(*indices)[0]) {
        return false;
    }
    auto indexDtype = (*indices)[0]->GetDataType();
    if (indexDtype != op::DataType::DT_INT32 && indexDtype != op::DataType::DT_INT64) {
        return false;
    }
    return true;
}

static bool CheckIndicesWithSelf(const aclTensor* selfRef, const aclTensorList* indices, const aclTensor* values) {
    int64_t indicesSize = static_cast<int64_t>(indices->Size());
    int64_t selfRefSize = selfRef->GetViewShape().GetDimNum();
    int64_t valuesSize = values->GetViewShape().GetDimNum();
    if (indicesSize < 1 || selfRefSize < 1 || valuesSize < 1 || (*indices)[0]->GetViewShape().GetDimNum() < 1) {
        OP_LOGD("IndexPutWithSortV2 Op not support no indices, not support 0 dims of selfRef or values!");
        return false;
    }
    if (indicesSize > selfRefSize) {
        OP_LOGD("IndexPutWithSortV2 Op not support nums of indices greater than dims of selfRef!");
        return false;
    }
    for (int64_t i = 0; i < indicesSize; i++) {
        if ((*indices)[i]) {
            if ((*indices)[i]->GetViewShape().GetShapeSize() == 0) {
                OP_LOGD("IndexPutWithSortV2 Op not support none data of indices!");
                return false;
            }
        } else {
            OP_LOGD("IndexPutWithSortV2 Op not support nullptr of indices!");
            return false;
        }
    }
    return true;
}

static bool CheckValuesShape(const aclTensorList* indices, const aclTensor* values) {
    auto valuesSize = values->GetViewShape().GetDimNum();
    if (valuesSize < (*indices)[0]->GetViewShape().GetDimNum()) {
        OP_LOGD("IndexPutWithSortV2 Op not support dims of values smaller than dims of indices!");
        return false;
    }
    for (size_t i = 0; i < (*indices)[0]->GetViewShape().GetDimNum(); i++) {
        if (values->GetViewShape().GetDim(i) != (*indices)[0]->GetViewShape().GetDim(i) &&
            values->GetViewShape().GetDim(i) != 1) {
            OP_LOGD("IndexPutWithSortV2 Op not support values need broadcast!");
            return false;
        }
    }
    return true;
}

static bool CheckSliceSize(const aclTensor* selfRef, const aclTensorList* indices, const aclTensor* values) {
    int64_t numIndexTensors = static_cast<int64_t>(indices->Size());
    int64_t selfDimNum = selfRef->GetViewShape().GetDimNum();
    int64_t numTailDims = selfDimNum - numIndexTensors; 
    auto indexDimNum = (*indices)[0]->GetViewShape().GetDimNum();
    auto valuesDimNum = values->GetViewShape().GetDimNum();
    if (valuesDimNum < indexDimNum + numTailDims) {
        OP_LOGD("IndexPutWithSortV2 Op not support values dims less than required!");
        return false;
    }
    for (int64_t i = 0; i < numTailDims; i++) {
        auto selfTailDimSize = selfRef->GetViewShape().GetDim(numIndexTensors + i);
        auto valuesTailDimSize = values->GetViewShape().GetDim(indexDimNum + i);
        if (selfTailDimSize != valuesTailDimSize && valuesTailDimSize != 1) {
            OP_LOGD("IndexPutWithSortV2 Op requires values tail shape to match selfRef or be broadcastable (1)!");
            return false;
        }
    }
    return true;
}

static bool CheckDataSize(const aclTensor* selfRef, const aclTensorList* indices) {
    static const int64_t INT32_MAX_LIMIT = 2147483647;
    static const int64_t CAST_MAX_NUM = 16777216;
    auto indicesSize = indices->Size();
    int64_t shapeProd = 1;
    for (size_t i = 0; i < indicesSize; i++) {
        if (selfRef->GetViewShape().GetDim(i) > CAST_MAX_NUM) {
            OP_LOGD("IndexPutWithSortV2 Op not support dim value of selfRef greater than %ld!", CAST_MAX_NUM);
            return false;
        }
        shapeProd *= selfRef->GetViewShape().GetDim(i);
    }
    if (shapeProd > INT32_MAX_LIMIT) {
        OP_LOGD("IndexPutWithSortV2 Op not support nums of indexed data greater than %ld!", INT32_MAX_LIMIT);
        return false;
    }
    return true;
}

static bool CheckIndicesDtypeAndShape(const aclTensorList* indices) {
    int64_t indicesSize = static_cast<int64_t>(indices->Size());
    for (int64_t i = 0; i < indicesSize; i++) {
        if (i == 0) {
            if ((*indices)[0]->GetDataType() == op::DataType::DT_BOOL) {
                OP_LOGD("IndexPutWithSortV2 Op not support bool dtype of indices!");
                return false;
            }
        } else {
            if ((*indices)[i]->GetDataType() != (*indices)[0]->GetDataType()) {
                OP_LOGD("IndexPutWithSortV2 Op only support one dtype of indices!");
                return false;
            }
            // 索引之间需要广播不支持
            if ((*indices)[i]->GetViewShape().GetDimNum() != (*indices)[0]->GetViewShape().GetDimNum()) {
                OP_LOGD("IndexPutWithSortV2 Op only support same dims of indices!");
                return false;
            }
            for (size_t j = 0; j < (*indices)[0]->GetViewShape().GetDimNum(); j++) {
                if ((*indices)[0]->GetViewShape().GetDim(j) != (*indices)[i]->GetViewShape().GetDim(j)) {
                    OP_LOGD("IndexPutWithSortV2 Op only support same shape of indices!");
                    return false;
                }
            }
        }
    }
    return true;
}

static bool IsSortV2PerformanceOptimal(const aclTensor* selfRef, 
    const aclTensorList* indices, const aclTensor* values) {
    constexpr int64_t MIN_TAIL_AXIS_ELEMENTS = 2048;
    constexpr int64_t AVG_BYTES_PER_ELEMENT = 2;
    constexpr int64_t MEMORY_LIMIT_BYTES = 64 * 1024 * 1024; // 64MB

    auto selfShape = selfRef->GetViewShape();
    auto selfDimNum = selfShape.GetDimNum();
    auto indexCount = indices->Size();
    if (selfDimNum <= indexCount) {
        return false;
    }
    // 尾轴元素个数大于等于 2048
    int64_t tailElementsCount = 1;
    for (size_t i = indexCount; i < selfDimNum; i++) {
        tailElementsCount *= static_cast<int64_t>(selfShape.GetDim(i));
    }
    if (tailElementsCount < MIN_TAIL_AXIS_ELEMENTS) {
        OP_LOGD("IndexPutWithSortV2 Opt skip: tail elements count is less than %ld!", MIN_TAIL_AXIS_ELEMENTS);
        return false;
    }
    // self和values总大小>=64M
    auto valueShape = values->GetViewShape();
    int64_t dataNums = static_cast<int64_t>(selfShape.GetShapeSize() + valueShape.GetShapeSize());
    if (dataNums < MEMORY_LIMIT_BYTES / AVG_BYTES_PER_ELEMENT) {
        OP_LOGD("IndexPutWithSortV2 Opt skip: total data size is less than %ld bytes!", MEMORY_LIMIT_BYTES);
        return false;
    }
    return true;
}

bool IsUseSortedV2OptScene(
    const bool isAiCpu, const aclTensor* self, const aclTensorList* indices, const aclTensor* values,
    const bool deterministicValue, const bool accumulate, const bool isNonContiguous) {
    // 1. 基本判断
    if (isAiCpu || deterministicValue || !accumulate || isNonContiguous) {
        return false;
    }
    // 2. self, indices和values数据类型限制
    if (!CheckBasicTensorProperties(self, indices, values)) {
        return false;
    }
    // 3. 不支持索引个数比self维度多 索引必须从首轴开始且连续
    if (!CheckIndicesWithSelf(self, indices, values)) {
        return false;
    }
    // 4. 索引不能为bool，且索引数据类型一致，且不支持索引之间广播
    if (!CheckIndicesDtypeAndShape(indices)) {
        return false;
    }
    // 5. values维度数不能小于索引维度数，支持values广播
    if (!CheckValuesShape(indices, values)) {
        return false;
    }
    // 6. 尾轴限制，self的尾轴和values必须相同，或可广播
    if (!CheckSliceSize(self, indices, values)) {
        return false;
    }
    // 7. 数据量限制，每个索引的取值范围不超过CAST_MAX_NUM，且self尾轴个数不超过INT32_MAX_LIMIT
    if (!CheckDataSize(self, indices)) {
        return false;
    }
    // 8. sortv2优化判断
    if (!IsSortV2PerformanceOptimal(self, indices, values)) {
        return false;
    }

    return true;
}

const aclTensor* IndexPutWithSortV2(
    const aclTensor* self, const aclTensor* linearIndex, const aclTensor* posIdx, const aclTensor* values,
    const aclIntArray* indexed_sizes, const bool accumulate, aclTensor* out, aclOpExecutor* executor) {
    L0_DFX(IndexPutWithSortV2, self, linearIndex, posIdx, values, indexed_sizes, accumulate);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        IndexPutWithSortV2, OP_INPUT(self, linearIndex, posIdx, values), OP_OUTPUT(out),
        OP_ATTR(indexed_sizes, accumulate));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(
        ret != ACLNN_SUCCESS, return nullptr, "IndexPutWithSortV2 ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return out;
}
} // namespace l0op
