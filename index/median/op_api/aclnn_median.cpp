/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_median.h"
#include "median_common.h"

#include "median.h"

using namespace Ops::NN::MedianCommon;

extern "C" {
aclnnStatus aclnnMedianGetWorkspaceSize(const aclTensor* self, aclTensor* valuesOut, uint64_t* workspaceSize,
                                        aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnMedian, DFX_IN(self), DFX_OUT(valuesOut));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = CheckParams(self, valuesOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 检查格式
    CheckFormat(self);

    // 空Tensor处理
    if (valuesOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    if (self->IsEmpty()) {
        ret = DealMedianEmptyTensor(self, valuesOut, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // self如果非连续，需要转连续
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto selfShapeDim = GetTensorDim(self);
    auto selfReshape = selfContiguous;
    // 对不为1维的tensor进行reshape
    if (selfShapeDim != 1) {
        selfReshape = ReduceOneDim(selfContiguous, selfShapeDim, uniqueExecutor.get());
        CHECK_RET(selfReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor* result = nullptr;
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        auto medianResult = l0op::Median(selfReshape, -1, uniqueExecutor.get());
        result = std::get<0>(medianResult);
    } else {
        auto sortValues = selfReshape;
        // 调用l0算子Sort进行计算
        if (selfReshape->GetViewShape().GetDim(DIM_ZERO) > 1) {
            bool descending = false;
            bool stable = true;
            auto sortResult = l0op::Sort(selfReshape, -1, descending, stable, op::DataType::DT_INT32,
                                         uniqueExecutor.get());
            CHECK_RET(CheckTupleNotNullptr(sortResult), ACLNN_ERR_INNER_NULLPTR);
            sortValues = std::get<0>(sortResult);
        }
        CHECK_RET(sortValues != nullptr, ACLNN_ERR_INNER_NULLPTR);

        auto medianIndex = (sortValues->GetViewShape().GetDim(DIM_ZERO) - 1) / 2;
        const aclTensor* medianIndexTensor = uniqueExecutor->ConvertToTensor(&medianIndex, 1, op::DataType::DT_INT64);
        CHECK_RET(medianIndexTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto selectResult = l0op::GatherV2(sortValues, 0, medianIndexTensor, uniqueExecutor.get());
        CHECK_RET(selectResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

        result = selectResult;
        auto FLOAT_DTYPE_LIST = GetFloatList();
        if (CheckType(self->GetDataType(), FLOAT_DTYPE_LIST)) {
            auto nanMask = GetNanMask(sortValues, 0, uniqueExecutor.get());
            CHECK_RET(nanMask != nullptr, ACLNN_ERR_INNER_NULLPTR);
            // 生成nan的Tensor
            auto nanTensor = CreateNanTensor(selectResult, uniqueExecutor.get());
            CHECK_RET(nanTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
            result = l0op::SelectV2(nanMask, nanTensor, selectResult, uniqueExecutor.get());
        }
    }
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // output shape check
    CHECK_RET(CheckReduceOutShape(result, valuesOut), ACLNN_ERR_PARAM_INVALID);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(result, valuesOut, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMedian(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMedian);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnMedianDimGetWorkspaceSize(const aclTensor* self, int64_t dim, bool keepDim, aclTensor* valuesOut,
                                           aclTensor* indicesOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnMedianDim, DFX_IN(self, dim, keepDim), DFX_OUT(valuesOut, indicesOut));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = CheckParamsDim(self, dim, keepDim, valuesOut, indicesOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 检查格式
    CheckFormat(self);

    // 空Tensor处理
    if (self->IsEmpty() || valuesOut->IsEmpty() || indicesOut->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // self如果非连续，需要转连续
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    int64_t dimSize = GetTensorDim(self);
    int64_t realDim = self->GetViewShape().IsScalar() ? 0 : dim < 0 ? dim + dimSize : dim;
    auto selfReshape = MedianAdaptInputZeroDimTensor(selfContiguous, dimSize, uniqueExecutor.get());
    CHECK_RET(selfReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    aclIntArray* selfReduceShape = GetReduceShape(selfReshape, realDim, uniqueExecutor.get());
    CHECK_RET(selfReduceShape != nullptr, ACLNN_ERR_INNER_NULLPTR);

    int64_t sortDimSize = self->GetViewShape().GetDim(static_cast<size_t>(realDim));
    const aclTensor* lastValueResult = nullptr;
    const aclTensor* lastIndicesResult = nullptr;
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        auto result = GetMedianResult(selfReshape, realDim, uniqueExecutor.get(), l0op::Median);
        lastValueResult = std::get<0>(result);
        CHECK_RET(lastValueResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        lastIndicesResult = std::get<1>(result);
        CHECK_RET(lastIndicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2002 && dimSize > 1 &&
               self->GetDataType() == op::DataType::DT_FLOAT && sortDimSize <= SMALLSORTLIMIT) {
        // small rank dim
        // Transpose
        std::vector<int64_t> perm(dimSize);
        for (int64_t i = 0; i < dimSize; i++) {
            perm[i] = i;
        }
        std::swap(perm[realDim], perm[0]);
        auto valuePerm = uniqueExecutor->AllocIntArray(perm.data(), dimSize);
        CHECK_RET(valuePerm != nullptr, ACLNN_ERR_INNER_NULLPTR);
        if (realDim != 0) {
            selfReshape = l0op::Transpose(selfReshape, valuePerm, uniqueExecutor.get());
            CHECK_RET(selfReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
        // MedianDim
        auto result = l0op::MedianDim(selfReshape, 0, true, uniqueExecutor.get());
        lastValueResult = std::get<0>(result);
        CHECK_RET(lastValueResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        lastIndicesResult = std::get<1>(result);
        CHECK_RET(lastIndicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        // Transpose back
        if (realDim != 0) {
            lastValueResult = l0op::Transpose(lastValueResult, valuePerm, uniqueExecutor.get());
            CHECK_RET(lastValueResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
            lastIndicesResult = l0op::Transpose(lastIndicesResult, valuePerm, uniqueExecutor.get());
            CHECK_RET(lastIndicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        }
    } else {
        // 获取排序结果
        auto result = GetMedianSortResult(selfReshape, realDim, uniqueExecutor.get());
        lastValueResult = std::get<0>(result);
        CHECK_RET(lastValueResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        lastIndicesResult = std::get<1>(result);
        CHECK_RET(lastIndicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // Reshape
    auto valueReshape = keepDim ? lastValueResult :
                                  l0op::Reshape(lastValueResult, selfReduceShape, uniqueExecutor.get());
    CHECK_RET(valueReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto indicesReshape = keepDim ? lastIndicesResult :
                                    l0op::Reshape(lastIndicesResult, selfReduceShape, uniqueExecutor.get());
    CHECK_RET(indicesReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Cast转换
    auto indicesCastResult = l0op::Cast(indicesReshape, indicesOut->GetDataType(), uniqueExecutor.get());
    CHECK_RET(indicesCastResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // output shape check
    CHECK_RET(CheckReduceOutShape(valueReshape, valuesOut), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckReduceOutShape(indicesCastResult, indicesOut), ACLNN_ERR_PARAM_INVALID);
    // 如果出参valuesOut是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto valueCopyResult = l0op::ViewCopy(valueReshape, valuesOut, uniqueExecutor.get());
    CHECK_RET(valueCopyResult != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // 如果出参indicesOut是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto indicesCopyResult = l0op::ViewCopy(indicesCastResult, indicesOut, uniqueExecutor.get());
    CHECK_RET(indicesCopyResult != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMedianDim(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMedianDim);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
}
