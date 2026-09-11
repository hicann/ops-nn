/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "median_common.h"

namespace Ops::NN::MedianCommon {
namespace {
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_WITH_INT_AND_BF16 = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16,  op::DataType::DT_UINT8,
    op::DataType::DT_INT8,  op::DataType::DT_INT16,   op::DataType::DT_INT32, op::DataType::DT_INT64};

static const std::initializer_list<op::DataType> FLOAT_DTYPE_LIST_910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_WITH_INT = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_UINT8, op::DataType::DT_INT8,
    op::DataType::DT_INT16, op::DataType::DT_INT32,   op::DataType::DT_INT64};

static const std::initializer_list<op::DataType> FLOAT_DTYPE_LIST_910 = {op::DataType::DT_FLOAT,
                                                                         op::DataType::DT_FLOAT16};

} // namespace

const std::initializer_list<op::DataType>& GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201 || Ops::NN::AclnnUtil::IsRegbase()) {
        return DTYPE_SUPPORT_LIST_WITH_INT_AND_BF16;
    }
    return DTYPE_SUPPORT_LIST_WITH_INT;
}

const std::initializer_list<op::DataType>& GetFloatList()
{
    if (GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201 || Ops::NN::AclnnUtil::IsRegbase()) {
        return FLOAT_DTYPE_LIST_910B;
    }
    return FLOAT_DTYPE_LIST_910;
}

bool CheckNotNull(const aclTensor* self, const aclTensor* valuesOut)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(valuesOut, return false);
    return true;
}

bool CheckDtypeValid(const aclTensor* self, const aclTensor* valuesOut)
{
    auto DTYPE_SUPPORT_LIST = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(valuesOut, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_MATCH(valuesOut, self->GetDataType(), return false);
    return true;
}

bool CheckShape(const aclTensor* self, const aclTensor* valuesOut)
{
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(valuesOut, MAX_SUPPORT_DIMS_NUMS, return false);
    return true;
}

aclnnStatus CheckParams(const aclTensor* self, const aclTensor* valuesOut)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, valuesOut), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    CHECK_RET(CheckDtypeValid(self, valuesOut), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否满足约束
    CHECK_RET(CheckShape(self, valuesOut), ACLNN_ERR_PARAM_INVALID);

    // 4. 当self的数据类型为BFLOAT16时，self的最后一根轴大小不能等于1
    auto dimSize = self->GetViewShape().GetDimNum();
    if (dimSize != 0) {
        if (self->GetDataType() == op::DataType::DT_BF16 && self->GetViewShape().GetDim(dimSize - 1) == 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The last axis value is not support 1 when input type is BF16.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    return ACLNN_SUCCESS;
}

// 检查tuple <values, indices>里的元素是否为非null。true表示为非null，false表示为null
bool CheckTupleNotNullptr(std::tuple<const aclTensor*, const aclTensor*> tensorTuple)
{
    return (std::get<0>(tensorTuple) != nullptr) && (std::get<1>(tensorTuple) != nullptr);
}

// 检查dType符合预期
bool CheckDtypeValidDim(const aclTensor* self, const aclTensor* valuesOut, const aclTensor* indicesOut)
{
    if (!CheckDtypeValid(self, valuesOut)) {
        return false;
    }

    if (indicesOut->GetDataType() != op::DataType::DT_INT64) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "indicesOut dtype %s should be in dtype support list [%s].",
                op::ToString(indicesOut->GetDataType()).GetString(), "op::DataType::DT_INT64");
        return false;
    }

    return true;
}

bool CheckNotNullDim(const aclTensor* self, const aclTensor* valuesOut, const aclTensor* indicesOut)
{
    if (!CheckNotNull(self, valuesOut)) {
        return false;
    }

    if (indicesOut == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "indicesOut should not be null.");
        return false;
    }

    return true;
}

// 获得tensor的维度数
int64_t GetTensorDim(const aclTensor* self) { return static_cast<int64_t>(self->GetViewShape().GetDimNum()); }

// dim应该处于范围 [-N, N-1]中
bool CheckDimValue(const aclTensor* self, const int64_t dim)
{
    int64_t dimSize = GetTensorDim(self);
    int64_t dimMin = std::min(-1 * dimSize, dimSize - 1);
    int64_t dimMax = std::max(-1 * dimSize, dimSize - 1);
    if ((dim > dimMax) || (dim < dimMin)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim should be in range [%ld, %ld].", dimMin, dimMax);
        return false;
    }
    return true;
}

bool CheckShapeDim(const aclTensor* self, bool keepDim, const aclTensor* valuesOut, const aclTensor* indicesOut)
{
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(valuesOut, MAX_SUPPORT_DIMS_NUMS, return false);
    OP_CHECK_MAX_DIM(indicesOut, MAX_SUPPORT_DIMS_NUMS, return false);

    auto expectedDimNum = keepDim ? self->GetViewShape().GetDimNum() : self->GetViewShape().GetDimNum() - 1;
    if (self->GetViewShape().GetDimNum() == 0) {
        expectedDimNum = self->GetViewShape().GetDimNum();
    }
    OP_CHECK_WRONG_DIMENSION(valuesOut, expectedDimNum, return false);
    OP_CHECK_WRONG_DIMENSION(indicesOut, expectedDimNum, return false);
    return true;
}

// 检查参数情况
aclnnStatus CheckParamsDim(const aclTensor* self, int64_t dim, bool keepDim, aclTensor* valuesOut,
                           aclTensor* indicesOut)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNullDim(self, valuesOut, indicesOut), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查参数的数据类型是否符合预期
    CHECK_RET(CheckDtypeValidDim(self, valuesOut, indicesOut), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查dim是否为 [-N, N-1] 范围内
    CHECK_RET(CheckDimValue(self, dim), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查shape是否超出MAX_DIM
    CHECK_RET(CheckShapeDim(self, keepDim, valuesOut, indicesOut), ACLNN_ERR_PARAM_INVALID);

    // 5. 当self的数据类型为BFLOAT16时，self.shape[dim]不能等于1
    auto dimSize = self->GetViewShape().GetDimNum();
    if (dimSize != 0) {
        auto realDim = (dim + dimSize) % dimSize;
        if (self->GetDataType() == op::DataType::DT_BF16 && self->GetViewShape().GetDim(realDim) == 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim axis value is not support 1 when input type is BF16.");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }

    return ACLNN_SUCCESS;
}

// 将dim与最后一个dim进行对换
aclIntArray* GetPermResult(int64_t dim, int64_t dimSize, aclOpExecutor* executor)
{
    std::vector<int64_t> valuePerm(dimSize, 0);
    for (int64_t i = 0; i < dimSize; i++) {
        valuePerm[i] = i;
    }

    std::swap(valuePerm[dim], valuePerm[dimSize - 1]);

    return executor->AllocIntArray(valuePerm.data(), dimSize);
}

const aclTensor* MedianAdaptInputZeroDimTensor(const aclTensor* self, int64_t dimNum, aclOpExecutor* executor)
{
    if (dimNum != 0) {
        return self;
    }
    int64_t selfShapeValue[1] = {1};
    aclIntArray* selfShape = executor->AllocIntArray(selfShapeValue, 1);
    auto selfReshape = l0op::Reshape(self, selfShape, executor);
    return selfReshape;
}

const aclTensor* ReduceOneDim(const aclTensor* self, int64_t selfShapeDim, aclOpExecutor* executor)
{
    int64_t selfShapeValue[1] = {1};
    for (int64_t i = 0; i < selfShapeDim; i++) {
        selfShapeValue[0] *= self->GetViewShape().GetDim(i);
    }
    aclIntArray* selfShape = executor->AllocIntArray(selfShapeValue, 1);
    CHECK_RET(selfShape != nullptr, nullptr);
    auto selfReshape = l0op::Reshape(self, selfShape, executor);
    CHECK_RET(selfReshape != nullptr, nullptr);
    return selfReshape;
}

aclIntArray* GetReduceShape(const aclTensor* self, const int64_t dim, aclOpExecutor* executor)
{
    int64_t dimSize = GetTensorDim(self);
    int64_t selfShapeValue[dimSize - 1];
    int64_t idx = 0;
    for (int64_t i = 0; i < dimSize; i++) {
        if (i == dim) {
            continue;
        }
        selfShapeValue[idx] = self->GetViewShape().GetDim(i);
        idx++;
    }
    aclIntArray* selfShape = executor->AllocIntArray(selfShapeValue, dimSize - 1);
    CHECK_RET(selfShape != nullptr, nullptr);
    return selfShape;
}

std::tuple<const aclTensor*, const aclTensor*> SortProcess(const aclTensor* self, const int64_t dim,
                                                           aclOpExecutor* executor)
{
    int64_t dimSize = GetTensorDim(self);
    CHECK_RET(dimSize != 0, std::tuple(nullptr, nullptr));
    int64_t dimMax = std::max(-1 * dimSize, dimSize - 1);
    bool descending = false;
    bool stable = true;

    const aclTensor* sortValues;
    const aclTensor* sortIndices;

    auto realDim = (dim + dimSize) % dimSize;
    if (realDim == dimMax) {
        auto sortResult = l0op::Sort(self, -1, descending, stable, op::DataType::DT_INT32, executor);
        CHECK_RET(CheckTupleNotNullptr(sortResult), std::tuple(nullptr, nullptr));
        sortValues = std::get<0>(sortResult);
        sortIndices = std::get<1>(sortResult);
    } else {
        // dim非最后一维，需进行transpose
        auto valuePerm = GetPermResult(realDim, dimSize, executor);
        CHECK_RET(valuePerm != nullptr, std::tuple(nullptr, nullptr));
        auto transposeSelf = l0op::Transpose(self, valuePerm, executor);
        CHECK_RET(transposeSelf != nullptr, std::tuple(nullptr, nullptr));

        auto sortResult = l0op::Sort(transposeSelf, -1, descending, stable, op::DataType::DT_INT32, executor);
        CHECK_RET(CheckTupleNotNullptr(sortResult), std::tuple(nullptr, nullptr));
        sortValues = l0op::Transpose(std::get<0>(sortResult), valuePerm, executor);
        sortIndices = l0op::Transpose(std::get<1>(sortResult), valuePerm, executor);
    }

    return std::tie(sortValues, sortIndices);
}

const aclTensor* GetLast(const aclTensor* sortValues, int64_t dim, aclOpExecutor* executor)
{
    int64_t LastIndex = sortValues->GetViewShape().GetDim(dim) - 1;
    const aclTensor* LastIndexTensor = executor->ConvertToTensor(&LastIndex, 1, op::DataType::DT_INT64);
    CHECK_RET(LastIndexTensor != nullptr, nullptr);
    auto lastResult = l0op::GatherV2(sortValues, dim, LastIndexTensor, executor);
    CHECK_RET(lastResult != nullptr, nullptr);
    return lastResult;
}

const aclTensor* CreateNanTensor(const aclTensor* self, aclOpExecutor* executor)
{
    FVector<float> valVector = {NAN};
    auto nanTensor = executor->ConvertToTensor(valVector.data(), valVector.size(), self->GetDataType());
    return nanTensor;
}

const aclTensor* GetNanMask(const aclTensor* self, int64_t dim, aclOpExecutor* executor)
{
    // 判断是否为nan，nan的notEqual结果是true，非nan的notEqual结果是false
    auto lastValue = GetLast(self, dim, executor);
    CHECK_RET(lastValue != nullptr, nullptr);
    auto nanMask = l0op::NotEqual(lastValue, lastValue, executor);
    CHECK_RET(nanMask != nullptr, nullptr);
    return nanMask;
}

const aclTensor* GetFirstNanIndices(const aclTensor* sortValues, const aclTensor* sortIndices, int64_t dim,
                                    aclOpExecutor* executor)
{
    // nan的Equal结果是false，非nan的Equal结果是true
    auto equalTensor = l0op::Equal(sortValues, sortValues, executor);
    OP_CHECK(equalTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Equal return nullptr."), return nullptr);

    auto endIndices = sortValues->GetViewShape().GetDim(dim) - 1;
    const aclTensor* endTensor = executor->ConvertToTensor(executor->AllocScalar(endIndices),
                                                           sortIndices->GetDataType());
    OP_CHECK(endTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ConvertToTensor return nullptr."), return nullptr);

    auto maskedFillTensor = l0op::MaskedFill(sortIndices, equalTensor, endTensor, executor);
    OP_CHECK(maskedFillTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "MaskedFill return nullptr."),
             return nullptr);

    int64_t appendDim[1] = {dim};
    auto dimArray = executor->AllocIntArray(appendDim, 1);
    OP_CHECK(dimArray != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AllocIntArray return nullptr."), return nullptr);
    auto reduceMinTensor = l0op::ReduceMin(maskedFillTensor, dimArray, true, executor);
    OP_CHECK(reduceMinTensor != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ReduceMin return nullptr."), return nullptr);

    return reduceMinTensor;
}

std::tuple<const aclTensor*, const aclTensor*> GetMedianSortResult(const aclTensor* self, int64_t dim,
                                                                   aclOpExecutor* executor)
{
    // 调用Sort算子
    auto sortResult = SortProcess(self, dim, executor);
    auto sortValues = std::get<0>(sortResult);
    CHECK_RET(sortValues != nullptr, std::tuple(nullptr, nullptr));
    auto sortIndices = std::get<1>(sortResult);
    CHECK_RET(sortIndices != nullptr, std::tuple(nullptr, nullptr));

    // GatherV2选择中位数
    auto medianIndex = (self->GetViewShape().GetDim(dim) - 1) / 2;
    const aclTensor* medianIndexTensor = executor->ConvertToTensor(&medianIndex, 1, op::DataType::DT_INT64);
    CHECK_RET(medianIndexTensor != nullptr, std::tuple(nullptr, nullptr));
    auto valueResult = l0op::GatherV2(sortValues, dim, medianIndexTensor, executor);
    CHECK_RET(valueResult != nullptr, std::tuple(nullptr, nullptr));
    auto indicesResult = l0op::GatherV2(sortIndices, dim, medianIndexTensor, executor);
    CHECK_RET(indicesResult != nullptr, std::tuple(nullptr, nullptr));

    auto lastValueResult = valueResult;
    auto lastIndicesResult = indicesResult;
    auto FLOAT_DTYPE_LIST = GetFloatList();
    if (CheckType(self->GetDataType(), FLOAT_DTYPE_LIST)) {
        // 获取第一个nan的索引
        auto indicesResultTmp = GetFirstNanIndices(sortValues, sortIndices, dim, executor);
        CHECK_RET(indicesResultTmp != nullptr, std::tuple(nullptr, nullptr));

        auto nanMask = GetNanMask(sortValues, dim, executor);
        CHECK_RET(nanMask != nullptr, std::tuple(nullptr, nullptr));

        // 生成nan的Tensor
        auto nanTensor = CreateNanTensor(valueResult, executor);
        CHECK_RET(nanTensor != nullptr, std::tuple(nullptr, nullptr));

        int64_t dimSize = GetTensorDim(self);
        int64_t selfShapeValue[dimSize];
        int64_t idx = 0;
        for (int64_t i = 0; i < dimSize; i++) {
            if (i == dim) {
                selfShapeValue[idx] = 1;
                idx++;
                continue;
            }
            selfShapeValue[idx] = self->GetViewShape().GetDim(i);
            idx++;
        }
        aclIntArray* selfShape = executor->AllocIntArray(selfShapeValue, dimSize);
        CHECK_RET(selfShape != nullptr, std::tuple(nullptr, nullptr));

        auto selfShapeTensor = executor->ConvertToTensor(selfShape, op::ToOpDataType(ACL_INT64));
        CHECK_RET(selfShapeTensor != nullptr, std::tuple(nullptr, nullptr));
        auto nanFillTensor = l0op::Fill(selfShapeTensor, nanTensor, selfShape, executor);
        CHECK_RET(nanFillTensor != nullptr, std::tuple(nullptr, nullptr));

        lastValueResult = l0op::SelectV2(nanMask, nanFillTensor, valueResult, executor);
        lastIndicesResult = l0op::SelectV2(nanMask, indicesResultTmp, indicesResult, executor);
    }
    CHECK_RET(lastValueResult != nullptr, std::tuple(nullptr, nullptr));
    CHECK_RET(lastIndicesResult != nullptr, std::tuple(nullptr, nullptr));

    return std::tie(lastValueResult, lastIndicesResult);
}

bool CanMedianNonLastAxisDirectly(const aclTensor* self, int64_t realDim)
{
    if (!Ops::NN::AclnnUtil::IsRegbase()) {
        return false;
    }
    int64_t dimNum = static_cast<int64_t>(self->GetViewShape().GetDimNum());
    if (dimNum <= 0 || dimNum > static_cast<int64_t>(MAX_SUPPORT_DIMS_NUMS) || realDim == dimNum - 1) {
        return false;
    }

    auto selfShape = self->GetViewShape();
    int64_t axisLen = selfShape[realDim];
    if (axisLen < MEDIAN_MIN_AXIS_LEN || axisLen > MEDIAN_DIRECT_AXIS_THRESHOLD) {
        return false;
    }

    int64_t outerSize = 1;
    int64_t innerSize = 1;
    for (int64_t i = 0; i < realDim; ++i) {
        int64_t dimSize = selfShape[i];
        if (dimSize == 0) {
            outerSize = 0;
            break;
        }
        if (dimSize < 0) {
            return false;
        }
        outerSize *= dimSize;
    }
    for (int64_t i = realDim + 1; i < dimNum; ++i) {
        int64_t dimSize = selfShape[i];
        if (dimSize == 0) {
            innerSize = 0;
            break;
        }
        if (dimSize < 0) {
            return false;
        }
        innerSize *= dimSize;
    }

    int64_t dtypeSize = static_cast<int64_t>(op::TypeSize(self->GetDataType()));
    int64_t blockBytes = GetCurrentPlatformInfo().GetBlockSize();
    int64_t blockElems = Ops::Base::CeilDiv(blockBytes, dtypeSize);
    constexpr int64_t smallRowLargeOuterThreshold = 1024;
    // Small inner rows incur padding/gather overhead on the non-transpose path. With many outer slices,
    // that fixed cost outweighs the transpose traffic saved by calling Median/NanMedian directly.
    if (innerSize < blockElems && outerSize >= smallRowLargeOuterThreshold) {
        return false;
    }
    return true;
}

std::tuple<const aclTensor*, const aclTensor*> GetMedianResult(const aclTensor* self, int64_t realDim,
                                                               aclOpExecutor* executor, MedianFunction medianFunction)
{
    int64_t dimSize = GetTensorDim(self);
    if (realDim == dimSize - 1 || CanMedianNonLastAxisDirectly(self, realDim)) {
        auto result = medianFunction(self, realDim, executor);
        return std::make_tuple(std::get<0>(result), std::get<1>(result));
    }

    auto valuePerm = GetPermResult(realDim, dimSize, executor);
    CHECK_RET(valuePerm != nullptr, std::tuple(nullptr, nullptr));
    auto transposeSelf = l0op::Transpose(self, valuePerm, executor);
    CHECK_RET(transposeSelf != nullptr, std::tuple(nullptr, nullptr));

    auto result = medianFunction(transposeSelf, -1, executor);
    auto valuesNoTrans = std::get<0>(result);
    auto indicesNoTrans = std::get<1>(result);
    CHECK_RET(valuesNoTrans != nullptr && indicesNoTrans != nullptr, std::tuple(nullptr, nullptr));

    auto values = l0op::Transpose(valuesNoTrans, valuePerm, executor);
    auto indices = l0op::Transpose(indicesNoTrans, valuePerm, executor);
    return std::make_tuple(values, indices);
}

aclnnStatus DealMedianEmptyTensor(const aclTensor* self, aclTensor* out, aclOpExecutor* executor)
{
    OP_LOGI("start DealMedianEmptyTensor.");
    auto outShape = out->GetViewShape();
    op::FVector<int64_t, op::MAX_DIM_NUM> fillDims = op::ToShapeVector(outShape);
    auto shapes = executor->AllocIntArray(fillDims.data(), outShape.GetDimNum());
    const aclTensor* dimTensor = executor->ConvertToTensor(shapes, op::DataType::DT_INT64);

    const aclScalar* valueScalar;
    if (self->GetDataType() == op::DataType::DT_INT64) {
        valueScalar = executor->AllocScalar(MIN_INT64);
    } else if (self->GetDataType() == op::DataType::DT_INT32) {
        valueScalar = executor->AllocScalar(MIN_INT32);
    } else if (self->GetDataType() == op::DataType::DT_FLOAT || self->GetDataType() == op::DataType::DT_FLOAT16 ||
               self->GetDataType() == op::DataType::DT_BF16) {
        valueScalar = executor->AllocScalar(NAN);
    } else {
        valueScalar = executor->AllocScalar(0);
    }

    const aclTensor* valueTensor = executor->ConvertToTensor(valueScalar, out->GetDataType());
    auto fillTensor = l0op::Fill(dimTensor, valueTensor, shapes, executor);
    CHECK_RET(fillTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto dstCopyResult = l0op::ViewCopy(fillTensor, out, executor);
    CHECK_RET(dstCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

void CheckFormat(const aclTensor* self)
{
    ge::Format selfStorageFormat = self->GetStorageFormat();
    if (selfStorageFormat == ge::Format::FORMAT_FRACTAL_NZ) {
        OP_LOGW("aclnnMedian/aclnnMedianDim doesn't support format NZ.");
    }
}

} // namespace Ops::NN::MedianCommon
