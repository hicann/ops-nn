/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_gather.cpp
 * \brief
 */

#include "aclnn_kernels/cast.h"
#include "level0/gather_elements.h"
#include "gather_elements_v2.h"
#include "level0/gather_v2.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/slice.h"
#include "aclnn_kernels/reshape.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_gather.h"

using namespace op;

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT,  DataType::DT_INT32,  DataType::DT_FLOAT16, DataType::DT_INT8,
    DataType::DT_DOUBLE, DataType::DT_INT16,  DataType::DT_INT64,   DataType::DT_UINT64,
    DataType::DT_UINT32, DataType::DT_UINT16, DataType::DT_UINT8,   DataType::DT_BOOL};

static const std::initializer_list<DataType> DTYPE_SUPPORT_910B_LIST = {
    DataType::DT_FLOAT, DataType::DT_INT32, DataType::DT_FLOAT16, DataType::DT_INT8,   DataType::DT_DOUBLE,
    DataType::DT_INT16, DataType::DT_INT64, DataType::DT_UINT64,  DataType::DT_UINT32, DataType::DT_UINT16,
    DataType::DT_UINT8, DataType::DT_BOOL,  DataType::DT_BF16};

static const std::initializer_list<DataType> INDEX_DTYPE_SUPPORT_LIST = {DataType::DT_INT64, DataType::DT_INT32};

static const std::initializer_list<DataType> GATHER_ELEMENTS_V2_DTYPE_SUPPORT_LIST = {
    DataType::DT_INT32, DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};

static const size_t DIM_BOUND = 8;
static const size_t SIZE_FLOAT16 = 2;
static const int32_t ALIGNED_SIZE = 32;
static const int32_t UPPER_LIMIT = 122400;
static const int32_t LOWER_LIMIT = 512;
static const int64_t SHAPE_PRODUCT_LIMIT = 256;
static const int32_t MOE_DIM_NUM = 2;
static const int32_t INDEX_DIM_NUM = 1;
static const int32_t FIRST_DIM = 0;
static const int32_t SECOND_DIM = 1;
static const int64_t STRIDE_ZERO = 0;
static const int64_t STRIDE_ONE = 1;
static const int64_t INT_MAX_NUM = 2147483647;
static const int64_t ASCEND_910B_CORE_NUM = 48;
static const int64_t ASCEND_910B_UB = 196608;
static const int64_t POST_DIM_LIMIT = 32;
static const int64_t TRANS_LEN = 16;
static const int64_t UB_LIMIT = 120000;
static const int64_t CAHELINE = 512;

static bool CheckNotNull(const aclTensor* self, const aclTensor* index, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(index, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* index, const aclTensor* out)
{
    bool is910bSocVersion =
        (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
         GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
         GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95);
    const std::initializer_list<op::DataType> CURRENT_DTYPE_SUPPORT_LIST =
        is910bSocVersion ? DTYPE_SUPPORT_910B_LIST : DTYPE_SUPPORT_LIST;
    OP_CHECK_DTYPE_NOT_SUPPORT(self, CURRENT_DTYPE_SUPPORT_LIST, return false);
    if (!index->IsEmpty() && !CheckType(index->GetDataType(), INDEX_DTYPE_SUPPORT_LIST)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "aclnnGather not implemented for index dtype %s.",
            op::ToString(index->GetDataType()).GetString());
        return false;
    }
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const int64_t dim, const aclTensor* index, const aclTensor* out)
{
    auto selfDim = self->GetViewShape().GetDimNum();
    auto indexDim = index->GetViewShape().GetDimNum();
    auto selfShape = self->GetViewShape();
    auto indexShape = index->GetViewShape();
    int64_t selfDim2 = (selfDim == 0) ? 1 : static_cast<int64_t>(selfDim);
    int64_t indexDim2 = (indexDim == 0) ? 1 : static_cast<int64_t>(indexDim);

    // self与index的维度需要相同
    if (selfDim2 != indexDim2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Index tensor must have the same number of dimensions as input tensor.");
        return false;
    }

    // tensor的维度不能超过上限
    OP_CHECK_MAX_DIM(self, DIM_BOUND, return false);
    OP_CHECK_MAX_DIM(index, DIM_BOUND, return false);
    OP_CHECK_MAX_DIM(out, DIM_BOUND, return false);

    // index的shape应该和out相同
    OP_CHECK_SHAPE_NOT_EQUAL(out, index, return false);

    // dim取值范围不能超过tensor本身的维度
    if (dim < (-selfDim2) || dim >= selfDim2) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Dimension out of range (expected to be in range of [%ld, %ld], but got %ld)",
            -selfDim2, selfDim2 - 1, dim);
        return false;
    }

    // 判断index的各维shape是否符合要求
    if (index->IsEmpty()) {
        return true;
    }

    size_t noNeedCheckDim = (dim < 0) ? selfDim + dim : dim;
    for (size_t i = 0; i < selfDim; i++) {
        auto curSelfShape = selfShape.GetDim(i);
        auto curIndexShape = indexShape.GetDim(i);
        if (i != noNeedCheckDim) {
            if (curIndexShape > curSelfShape) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID,
                    "Size does not match at dimension %zu, expected index shape %ld smaller than self shape %ld", i,
                    curIndexShape, curSelfShape);
                return false;
            }
        } else {
            if (curSelfShape == 0) {
                OP_LOGE(
                    ACLNN_ERR_PARAM_INVALID, "Size does not match at dimension %zu, expected index shape to be 0", i);
                return false;
            }
        }
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const int64_t dim, const aclTensor* index, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, index, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, index, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否满足约束
    CHECK_RET(CheckShape(self, dim, index, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static bool SpecialCheck(const aclTensor* selfContiguous, const aclTensor* indexContiguous, int64_t dimSize)
{
    // special scenario check
    bool socVersionCheck = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P;
    auto indexShape = indexContiguous->GetViewShape();
    int64_t indexShapeProduct = 1;
    for (int64_t i = 0; i < dimSize - 1; i++) {
        indexShapeProduct *= static_cast<int64_t>(indexShape.GetDim(i));
    }
    auto lastDim = indexShape.GetDim(dimSize - 1);
    bool indexShapeCheck = indexShapeProduct >= SHAPE_PRODUCT_LIMIT && lastDim <= UPPER_LIMIT && lastDim >= LOWER_LIMIT;
    bool lastDimAlignedCheck = lastDim * SIZE_FLOAT16 % ALIGNED_SIZE == 0;
    bool dTypeCheck =
        indexContiguous->GetDataType() == DataType::DT_INT64 && selfContiguous->GetDataType() == DataType::DT_FLOAT16;
    return socVersionCheck && indexShapeCheck && lastDimAlignedCheck && dTypeCheck;
}

static bool MemCheck(
    const aclTensor* selfContiguous, const aclTensor* indexContiguous, const int64_t& dim, const int64_t& idxPreDim,
    const int64_t& idxPostDim, const int64_t& selfShapeProduct, const int64_t& indexShapeProduct)
{
    bool isTransCase = true;
    bool memCheck = true;
    auto indexShape = indexContiguous->GetViewShape();
    auto selfShape = selfContiguous->GetViewShape();
    int64_t idxGatherDim = indexShape.GetDim(dim);
    int64_t selfGatherDim = selfShape.GetDim(dim);
    int64_t selfDtypeSize = GetSizeByDataType(selfContiguous->GetDataType());
    int64_t idxDtypeSize = GetSizeByDataType(indexContiguous->GetDataType());
    // transpose方案
    if (selfGatherDim * selfDtypeSize < UB_LIMIT) {
        isTransCase = true;
        // 收集轴 < 16 方案不适用
        memCheck = idxGatherDim >= TRANS_LEN && selfGatherDim >= TRANS_LEN;
        // scalar方案
    } else if (selfGatherDim * selfDtypeSize > ASCEND_910B_UB) {
        isTransCase = false;
    } else {
        memCheck = false;
    }
    // 检查transpose内存占用
    if (memCheck && isTransCase) {
        // 计算申请workspace行数
        int64_t tailGroupCoreNum = std::max(static_cast<int64_t>(1), ASCEND_910B_CORE_NUM / idxPreDim);
        int64_t workspaceLen =
            std::min(CAHELINE / selfDtypeSize, (idxPostDim + tailGroupCoreNum - 1) / tailGroupCoreNum);
        // 不同分核内存占用检查
        if (idxPreDim * idxPostDim <= ASCEND_910B_CORE_NUM) {
            memCheck = selfShapeProduct > idxPreDim * idxPostDim * (idxGatherDim + selfGatherDim);
        } else {
            memCheck =
                selfShapeProduct * selfDtypeSize + indexShapeProduct * idxDtypeSize >
                indexShapeProduct * selfDtypeSize +
                    ASCEND_910B_CORE_NUM * workspaceLen * (idxGatherDim * idxDtypeSize + selfGatherDim * selfDtypeSize);
        }
    }
    return memCheck;
}

static bool IfUseGatherElementsV2(
    const aclTensor* selfContiguous, const aclTensor* indexContiguous, const int64_t& dim, const int64_t& dimSize,
    bool& dtypeAndSocCheckFlag)
{
    // soc检查
    bool socVersionCheck =
        (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
         GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93);
    // self数据类型约束
    bool selfDTypeCheck = CheckType(selfContiguous->GetDataType(), GATHER_ELEMENTS_V2_DTYPE_SUPPORT_LIST);
    dtypeAndSocCheckFlag = socVersionCheck && selfDTypeCheck;
    bool lastDimFlag = dim == dimSize - 1;
    // shape检查
    auto indexShape = indexContiguous->GetViewShape();
    auto selfShape = selfContiguous->GetViewShape();
    int64_t indexShapeProduct = 1;
    int64_t selfShapeProduct = 1;
    bool dimCheck = true;
    int64_t selfPostDim = 1;
    int64_t selfPreDim = 1;
    int64_t idxPostDim = 1;
    int64_t idxPreDim = 1;
    for (int64_t i = 0; i < dimSize; i++) {
        int64_t indexDim = indexShape.GetDim(i);
        int64_t selfDim = selfShape.GetDim(i);
        indexShapeProduct *= indexDim;
        selfShapeProduct *= selfDim;
        if (i != dim && selfDim != indexDim) {
            dtypeAndSocCheckFlag = false;
        }
        // 方案将收集轴前后进行合轴，若非0、dim、dim+1轴shape不一致，会导致无法连续搬运，因此存在dim限制
        if ((i != 0 && i != dim && i != dim + 1) && indexDim != selfDim) {
            dimCheck = false;
            break;
        }
        if (i > dim) {
            selfPostDim *= selfDim;
            idxPostDim *= indexDim;
        }
        if (i < dim) {
            selfPreDim *= selfDim;
            idxPreDim *= indexDim;
        }
    }
    if (lastDimFlag) {
        return false;
    }
    bool selfShapeCheck = selfShapeProduct < INT_MAX_NUM;
    // 合轴后尾轴 < 32 会放大跳搬影响方案不适用
    dimCheck = dimCheck && selfPostDim > POST_DIM_LIMIT;
    // 核内transpose内存占用与aclnnTranspose比较
    bool memCheck =
        MemCheck(selfContiguous, indexContiguous, dim, idxPreDim, idxPostDim, selfShapeProduct, indexShapeProduct);
    return socVersionCheck && selfDTypeCheck && dimCheck && selfShapeCheck && memCheck;
}

static const aclTensor* GatherAdaptInputZeroDimTensor(const aclTensor* self, aclOpExecutor* executor)
{
    if (self->GetViewShape().GetDimNum() != 0) {
        return self;
    }
    int64_t selfShapeValue[1] = {1};
    aclIntArray* selfShape = executor->AllocIntArray(selfShapeValue, 1);
    CHECK_RET(selfShape != nullptr, nullptr);
    auto selfReshape = l0op::Reshape(self, selfShape, executor);
    CHECK_RET(selfReshape != nullptr, nullptr);
    return selfReshape;
}

static const aclTensor* CalGather(
    const aclTensor* selfContiguous, const int64_t dimFinal, const aclTensor* index, const int64_t dimSize,
    aclOpExecutor* executor)
{
    // calculation by GatherElements
    const aclTensor* gatherElementsResult = nullptr;
    // index如果非连续，需要转连续
    auto indexContiguous = l0op::Contiguous(index, executor);
    CHECK_RET(indexContiguous != nullptr, nullptr);
    std::vector<int64_t> perm(dimSize);
    for (int64_t i = 0; i < dimSize; i++) {
        perm[i] = i;
    }
    std::swap(perm[dimFinal], perm[dimSize - 1]);
    auto valuePerm = executor->AllocIntArray(perm.data(), dimSize);
    CHECK_RET(valuePerm != nullptr, nullptr);
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    bool dtypeAndSocCheckFlag = false;
    bool needTranspose =
        !IfUseGatherElementsV2(selfContiguous, indexContiguous, dimFinal, dimSize, dtypeAndSocCheckFlag);
    if (dimFinal != dimSize - 1 && needTranspose && socVersion != SocVersion::ASCEND910_95) {
        selfContiguous = l0op::Transpose(selfContiguous, valuePerm, executor);
        CHECK_RET(selfContiguous != nullptr, nullptr);
        indexContiguous = l0op::Transpose(indexContiguous, valuePerm, executor);
        CHECK_RET(indexContiguous != nullptr, nullptr);
    }

    // special calculation on inference soc
    if (SpecialCheck(selfContiguous, indexContiguous, dimSize)) {
        const aclTensor* indexContiguousCasted = l0op::Cast(indexContiguous, DataType::DT_INT32, executor);
        CHECK_RET(indexContiguousCasted != nullptr, nullptr);
        gatherElementsResult = l0op::GatherElements(selfContiguous, dimSize - 1, indexContiguousCasted, executor);
    } else if (dimFinal != dimSize - 1 && !needTranspose) {
        const aclTensor* indexContiguousCasted = l0op::Cast(indexContiguous, DataType::DT_INT32, executor);
        CHECK_RET(indexContiguousCasted != nullptr, nullptr);
        gatherElementsResult = l0op::GatherElementsV2(selfContiguous, indexContiguousCasted, dimFinal, executor);
    } else if (dtypeAndSocCheckFlag) {
        const aclTensor* indexContiguousCasted = l0op::Cast(indexContiguous, DataType::DT_INT32, executor);
        CHECK_RET(indexContiguousCasted != nullptr, nullptr);
        gatherElementsResult = l0op::GatherElementsV2(selfContiguous, indexContiguousCasted, dimSize - 1, executor);
    } else {
        int64_t dim = socVersion == SocVersion::ASCEND910_95 ? dimFinal : dimSize - 1;
        gatherElementsResult = l0op::GatherElements(selfContiguous, dim, indexContiguous, executor);
    }
    CHECK_RET(gatherElementsResult != nullptr, nullptr);

    if (dimFinal != dimSize - 1 && needTranspose && socVersion != SocVersion::ASCEND910_95) {
        gatherElementsResult = l0op::Transpose(gatherElementsResult, valuePerm, executor);
        CHECK_RET(gatherElementsResult != nullptr, nullptr);
    }
    return gatherElementsResult;
}

static bool IfMoeUseV2(const int64_t dim, const aclTensor* self, const aclTensor* index, int64_t selfDimNum)
{
    auto strides = index->GetViewStrides();
    bool strideCheckFlag = false;
    if (strides.size() == MOE_DIM_NUM && strides[FIRST_DIM] == STRIDE_ONE && strides[SECOND_DIM] == STRIDE_ZERO &&
        index->GetStorageShape().GetDimNum() == INDEX_DIM_NUM) {
        strideCheckFlag = true;
    }
    auto inferOutputDim = self->GetViewShape().GetDim(SECOND_DIM);
    auto realOutputDim = index->GetViewShape().GetDim(SECOND_DIM);

    auto indexDimNum = index->GetViewShape().GetDimNum();
    bool dimCheckFlag = indexDimNum == MOE_DIM_NUM && selfDimNum == static_cast<int64_t>(indexDimNum) && dim == FIRST_DIM;
    return dimCheckFlag && strideCheckFlag && (inferOutputDim == realOutputDim);
}

static const aclTensor* CalGatherV2(
    const aclTensor* selfContiguous, const int64_t dimFinal, const aclTensor* index,
    aclOpExecutor* executor)
{
    // calculation by GatherV2
    const aclTensor* gatherElementsResult = nullptr;
    FVector<int64_t> offsetVector(SECOND_DIM, 0);
    aclIntArray* offsetArray = executor->AllocIntArray(offsetVector.data(), offsetVector.size());
    CHECK_RET(offsetArray != nullptr, nullptr);
    FVector<int64_t> sizeVector;
    auto indexShape = index->GetViewShape();
    sizeVector.emplace_back(indexShape[0]);
    aclIntArray* sizeArray = executor->AllocIntArray(sizeVector.data(), sizeVector.size());
    CHECK_RET(sizeArray != nullptr, nullptr);
    auto indexSlice = l0op::Slice(index, offsetArray, sizeArray, executor);
    CHECK_RET(indexSlice != nullptr, nullptr);
    // 调用l0算子GatherV2进行计算
    gatherElementsResult = l0op::GatherV2(selfContiguous, dimFinal, indexSlice, executor, 0, true);
    CHECK_RET(gatherElementsResult != nullptr, nullptr);
    return gatherElementsResult;
}

aclnnStatus aclnnGatherGetWorkspaceSize(
    const aclTensor* self, const int64_t dim, const aclTensor* index, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnGather, DFX_IN(self, dim, index), DFX_OUT(out));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数检查
    auto ret = CheckParams(self, dim, index, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空Tensor处理，因为已经通过了检查，只要index为空，out一定也为空，out的resize也已经在外部处理，直接返回
    if (index->IsEmpty() || self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // self如果非连续，需要转连续
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // process 0_d tensor
    auto selfReshapeContinuous = GatherAdaptInputZeroDimTensor(selfContiguous, uniqueExecutor.get());
    CHECK_RET(selfReshapeContinuous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto indexReshape = GatherAdaptInputZeroDimTensor(index, uniqueExecutor.get());
    CHECK_RET(indexReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);
    int64_t dimSize = (int64_t)(selfReshapeContinuous->GetViewShape().GetDimNum());
    int64_t dimFinal = (dim >= 0) ? dim : dim + dimSize;

    const aclTensor* gatherElementsResult = nullptr;
    // 调用l0算子GatherElements进行计算
    if (IfMoeUseV2(dimFinal, self, indexReshape, dimSize)) {
        gatherElementsResult =
            CalGatherV2(selfReshapeContinuous, dimFinal, indexReshape, uniqueExecutor.get());
    } else {
        gatherElementsResult = CalGather(selfReshapeContinuous, dimFinal, indexReshape, dimSize, uniqueExecutor.get());
    }
    CHECK_RET(gatherElementsResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyRes = l0op::ViewCopy(gatherElementsResult, out, uniqueExecutor.get());
    CHECK_RET(viewCopyRes != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGather(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGather);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}