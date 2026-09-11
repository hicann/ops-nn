/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../median/op_api/aclnn_median.h"
#include "../../median/op_api/median_common.h"

#include "nan_median.h"

using namespace Ops::NN::MedianCommon;

extern "C" {
static aclnnStatus DealNanMedianEmptyTensor(aclTensor* out, aclOpExecutor* executor)
{
    auto outShape = out->GetViewShape();
    op::FVector<int64_t, op::MAX_DIM_NUM> fillDims = op::ToShapeVector(outShape);
    auto shapes = executor->AllocIntArray(fillDims.data(), outShape.GetDimNum());
    const aclTensor* dimTensor = executor->ConvertToTensor(shapes, op::DataType::DT_INT64);

    const aclScalar* valueScalar = executor->AllocScalar(NAN);
    const aclTensor* valueTensor = executor->ConvertToTensor(valueScalar, out->GetDataType());
    auto fillTensor = l0op::Fill(dimTensor, valueTensor, shapes, executor);
    CHECK_RET(fillTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto dstCopyResult = l0op::ViewCopy(fillTensor, out, executor);
    CHECK_RET(dstCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static int64_t GetTensorSize(const aclTensor* self)
{
    int64_t size = 1;
    if (self->IsEmpty()) {
        return size;
    }
    int64_t dimNum = self->GetViewShape().GetDimNum();
    for (int64_t dim = 0; dim < dimNum; dim++) {
        size *= self->GetViewShape().GetDim(dim);
    }

    return size;
}

// 获取带dim的median中位数索引
static const aclTensor* GetNanMedianDimIndexTensor(const aclTensor* selfReshape, int64_t dim, aclOpExecutor* executor)
{
    // nan的Equal结果是false，非nan的Equal结果是true
    auto selfEqualTensor = l0op::Equal(selfReshape, selfReshape, executor);
    OP_CHECK(selfEqualTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Equal return nullptr."), return nullptr);

    auto dimSize = selfReshape->GetStorageShape().GetDim(dim);

    // 将Equal的结果cast成int类型
    const aclTensor* equalCasted = nullptr;
    auto dtype = op::DataType::DT_INT32;
    auto size = GetTensorSize(selfEqualTensor);
    if (size > MAX_INT32) {
        equalCasted = l0op::Cast(selfEqualTensor, op::DataType::DT_INT64, executor);
        dtype = op::DataType::DT_INT64;
    } else {
        equalCasted = l0op::Cast(selfEqualTensor, op::DataType::DT_INT32, executor);
    }
    OP_CHECK(equalCasted != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Cast return nullptr."), return nullptr);

    // 通过l0算子ReduceSumOp求非nan元素的个数
    const int64_t reduceDim[] = {dim};
    auto reduceDimArray = executor->AllocIntArray(reduceDim, 1);
    auto reduceSumTensor = l0op::ReduceSumOp(equalCasted, reduceDimArray, true, executor);
    OP_CHECK(reduceSumTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "reducesumop return nullptr."),
             return nullptr);

    // 减1
    auto one = 1;
    const aclTensor* oneTensor = executor->ConvertToTensor(&one, 1, dtype);
    OP_CHECK(oneTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ConvertToTensor return nullptr."), return nullptr);

    auto subOneTensor = l0op::Sub(reduceSumTensor, oneTensor, executor);
    OP_CHECK(subOneTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Sub return nullptr."), return nullptr);
    // 调用l0算子将减1的结果除以2
    const aclTensor* resultTensor = nullptr;
    if (dimSize >= MAX_CONVERT_NUM) {
        int64_t rightShiftisor = 1;
        const aclTensor* otherTensor = executor->ConvertToTensor(&rightShiftisor, 1, dtype);
        OP_CHECK(otherTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ConvertToTensor return nullptr."),
                 return nullptr);

        // 调用l0算子rightShift计算中位数索引
        resultTensor = l0op::RightShift(subOneTensor, otherTensor, executor);
        OP_CHECK(resultTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "rightShift return nullptr."),
                 return nullptr);
    } else {
        int64_t Divisor = 2;
        const aclTensor* otherTensor = executor->ConvertToTensor(&Divisor, 1, dtype);
        OP_CHECK(otherTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ConvertToTensor return nullptr."),
                 return nullptr);

        // 调用l0算子rightShift计算中位数索引
        resultTensor = l0op::Div(subOneTensor, otherTensor, executor);
        OP_CHECK(resultTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Div return nullptr."), return nullptr);
    }

    return resultTensor;
}

aclnnStatus aclnnNanMedianGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                           aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnNanMedian, DFX_IN(self), DFX_OUT(out));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = CheckParams(self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor处理
    if (self->IsEmpty()) {
        DealNanMedianEmptyTensor(out, uniqueExecutor.get());
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
    if (selfShapeDim > 1) {
        int64_t selfShapeValue[1] = {1};
        for (int64_t i = 0; i < selfShapeDim; i++) {
            selfShapeValue[0] *= self->GetViewShape().GetDim(i);
        }

        aclIntArray* selfShape = uniqueExecutor->AllocIntArray(selfShapeValue, 1);
        CHECK_RET(selfShape != nullptr, ACLNN_ERR_INNER_NULLPTR);
        selfReshape = l0op::Reshape(selfContiguous, selfShape, uniqueExecutor.get());
        CHECK_RET(selfReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    const aclTensor* selectResult = nullptr;
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        auto result = l0op::NanMedian(selfReshape, -1, uniqueExecutor.get());
        selectResult = std::get<0>(result);
    } else {
        // 调用l0算子Sort进行计算
        bool descending = false;
        bool stable = true;
        auto sortResult = l0op::Sort(selfReshape, -1, descending, stable, op::DataType::DT_INT32, uniqueExecutor.get());
        CHECK_RET(CheckTupleNotNullptr(sortResult), ACLNN_ERR_INNER_NULLPTR);
        auto sortValues = std::get<0>(sortResult);

        // 获取中位数的索引
        auto medianIndexTensor = GetNanMedianDimIndexTensor(selfReshape, 0, uniqueExecutor.get());
        CHECK_RET(medianIndexTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 按照索引取出中位数
        selectResult = l0op::GatherV2(sortValues, 0, medianIndexTensor, uniqueExecutor.get(), 0, true);
    }
    CHECK_RET(selectResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // output shape check
    CHECK_RET(CheckReduceOutShape(selectResult, out), ACLNN_ERR_PARAM_INVALID);
    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(selectResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNanMedian(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnNanMedian);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

static aclnnStatus ExecZeroDimNum(const aclTensor* self, int64_t dim, aclTensor* valuesOut, aclTensor* indicesOut,
                                  aclOpExecutor* executor)
{
    CHECK_RET(CheckDimValue(self, dim), ACLNN_ERR_PARAM_INVALID);

    auto valueCopyResult = l0op::ViewCopy(self, valuesOut, executor);
    CHECK_RET(valueCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 标量的索引直接返回0
    auto indicesOutZero = l0op::ZerosLike(indicesOut, executor);
    CHECK_RET(indicesOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto indicesCopyResult = l0op::ViewCopy(indicesOutZero, indicesOut, executor);
    CHECK_RET(indicesCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNanMedianDimGetWorkspaceSize(const aclTensor* self, int64_t dim, bool keepDim, aclTensor* valuesOut,
                                              aclTensor* indicesOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnNanMedianDim, DFX_IN(self, dim, keepDim), DFX_OUT(valuesOut, indicesOut));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = CheckParamsDim(self, dim, keepDim, valuesOut, indicesOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 如果传入是标量
    auto dimNum = self->GetViewShape().GetDimNum();
    if (dimNum == 0) {
        ret = ExecZeroDimNum(self, dim, valuesOut, indicesOut, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ret);

        // 获取计算过程中需要使用的workspace大小
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    dim = (dim + dimNum) % dimNum;

    // 处理空tensor
    if (self->IsEmpty()) {
        if (self->GetViewShape().GetDim(dim) == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Expected reduction dim %ld to have non-zero size.", dim);
            CHECK_RET(false, ACLNN_ERR_PARAM_INVALID);
        }

        // 多维空tensor返回空tensor
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // self如果非连续，需要转连续
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* valueResult = nullptr;
    const aclTensor* indicesResult = nullptr;
    if (Ops::NN::AclnnUtil::IsRegbase()) {
        auto result = GetMedianResult(selfContiguous, dim, uniqueExecutor.get(), l0op::NanMedian);
        valueResult = std::get<0>(result);
        indicesResult = std::get<1>(result);
    } else {
        // 获取中位数的索引
        auto medianIndexTensor = GetNanMedianDimIndexTensor(selfContiguous, dim, uniqueExecutor.get());
        CHECK_RET(medianIndexTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // 调用Sort算子
        auto sortResult = SortProcess(selfContiguous, dim, uniqueExecutor.get());
        auto sortValues = std::get<0>(sortResult);
        CHECK_RET(sortValues != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto sortIndices = std::get<1>(sortResult);
        CHECK_RET(sortIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);

        valueResult = l0op::GatherElements(sortValues, dim, medianIndexTensor, uniqueExecutor.get());
        indicesResult = l0op::GatherElements(sortIndices, dim, medianIndexTensor, uniqueExecutor.get());
    }
    CHECK_RET(valueResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(indicesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Reshape
    aclIntArray* selfShape = GetReduceShape(self, dim, uniqueExecutor.get());
    CHECK_RET(selfShape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto ValueReshape = keepDim ? valueResult : l0op::Reshape(valueResult, selfShape, uniqueExecutor.get());
    CHECK_RET(ValueReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto IndicesReshape = keepDim ? indicesResult : l0op::Reshape(indicesResult, selfShape, uniqueExecutor.get());
    CHECK_RET(IndicesReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Cast转换
    auto indicesCastResult = l0op::Cast(IndicesReshape, indicesOut->GetDataType(), uniqueExecutor.get());
    CHECK_RET(indicesCastResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // output shape check
    CHECK_RET(CheckReduceOutShape(ValueReshape, valuesOut), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckReduceOutShape(indicesCastResult, indicesOut), ACLNN_ERR_PARAM_INVALID);
    // 如果出参valuesOut是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto valueCopyResult = l0op::ViewCopy(ValueReshape, valuesOut, uniqueExecutor.get());
    CHECK_RET(valueCopyResult != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // 如果出参indicesOut是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto indicesCopyResult = l0op::ViewCopy(indicesCastResult, indicesOut, uniqueExecutor.get());
    CHECK_RET(indicesCopyResult != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNanMedianDim(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnNanMedianDim);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
}
