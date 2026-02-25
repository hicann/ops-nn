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
 * \file aclnn_index.cpp
 * \brief index Aclnn file
 */
#include "aclnn_index.h"
#include "index.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "op_api/aclnn_util.h"
#include "index/gather_v2/op_api/gather_v2.h"

/* AdvancedIndex operator's flow as:
 *      self          indexed_sizes     indexed_strides     indices_0                 indices_1   ...
 *        |                  |                 |                |                        |
 * ToContiguous(ws_0)        |                 |                |                        |
 *        |                  |                 |                |                        |
 *  CastTo(ws_1)          (ws_2)           (ws_3)     BroadcastOperator(ws_4)   BroadcastOperator(ws_5) ...
 *         \                 |                 |               /                         /
 *          \                |                 |              /                         /
 *                                          IndexOp
 *                                             |
 *                                      CastTo(workspace_6)
 *                                             |
 *                                           result
 */

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const int64_t DATA_LIMIT_MULTI_INDEX = 500000;
static const int64_t MASK_MODE_01_SIZE = 2;
static const int64_t UB_LIMIT = 256;
static const int64_t TAIL_SIZE_LIMIT = 32;
static constexpr size_t MAX_DIM_LEN = 8;
static constexpr size_t DIM_BOUND_NON_CONTIGUOUS = 4;
struct IndicesInfo {
    FVector<int64_t, MAX_SUPPORT_DIMS_NUMS> masks;
    FVector<const aclTensor*, MAX_SUPPORT_DIMS_NUMS> allDefinedIndices;
    int64_t indicesNum = 0;
    int64_t masksNum = 0;
    int64_t dim = 0;
    size_t indicesDim = 0;
};
struct ChooseInfo {
    size_t outputDim = 0;
    size_t tailSize = 1;
    size_t permMasksNum = 0;
    op::Shape outputShape;
    bool isAicore = false;
    bool isNeedTranspose = false;
    bool isNonContiguous = false;
};

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT64, op::DataType::DT_INT32,
    op::DataType::DT_BOOL,  op::DataType::DT_INT8,    op::DataType::DT_UINT8};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_910B = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT64, op::DataType::DT_INT32,
    op::DataType::DT_BOOL,  op::DataType::DT_INT8,    op::DataType::DT_UINT8, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,  op::DataType::DT_FLOAT16, op::DataType::DT_INT64, op::DataType::DT_INT32,
    op::DataType::DT_BOOL,   op::DataType::DT_INT8,    op::DataType::DT_UINT8, op::DataType::DT_BF16,
    op::DataType::DT_DOUBLE, op::DataType::DT_INT16};

static const std::initializer_list<op::DataType> AICORE_DTYPE_SUPPORT_LIST_REGBASE = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT64, op::DataType::DT_INT32,
    op::DataType::DT_BOOL,  op::DataType::DT_INT8,    op::DataType::DT_UINT8, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> INDICES_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_INT64, op::DataType::DT_INT32, op::DataType::DT_BOOL};

static bool CheckPtrValid(const aclTensor* self, const aclTensorList* indices, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    OP_CHECK_NULL(indices, return false);
    for (uint64_t i = 0; i < indices->Size(); i++) {
        OP_CHECK_NULL((*indices)[i], return false);
    }
    return true;
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensorList* indices, const aclTensor* out)
{
    bool bf16Invalid = op::GetCurrentPlatformInfo().GetSocVersion() < op::SocVersion::ASCEND910B &&
                       self->GetDataType() == op::DataType::DT_BF16;
    if (bf16Invalid) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self not implemented for DT_BF16, when SocVersion is less than ASCEND910B.");
        return false;
    }
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);
    // 检查数据类型
    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);

    for (size_t i = 0; i < indices->Size(); i++) {
        if ((*indices)[i]->GetViewShape().GetShapeSize() != 0) {
            OP_CHECK_DTYPE_NOT_SUPPORT((*indices)[i], INDICES_DTYPE_SUPPORT_LIST, return false);
        }
    }
    return true;
}

static bool CheckFormat(const aclTensor* self, const aclTensorList* indices, const aclTensor* out)
{
    // 如果输入格式是私有格式，记录日志，直接报错
    if (op::IsPrivateFormat(self->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "self don't support private format.");
        return false;
    }

    if (op::IsPrivateFormat(out->GetStorageFormat())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out don't support private format.");
        return false;
    }

    for (size_t i = 0; i < indices->Size(); i++) {
        if (op::IsPrivateFormat((*indices)[i]->GetStorageFormat())) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "indices don't support private format.");
            return false;
        }
    }

    return true;
}

static void CheckShape(const aclTensor* self, const aclTensorList* indices)
{
    // 索引个数不能大于self维度数量
    size_t indicesSize = indices->Size();
    size_t selfDimNum = self->GetViewShape().GetDimNum();
    bool indicesSizeInvalidForZero = (selfDimNum == 0 && indicesSize > 1);
    if (indicesSizeInvalidForZero) {
        OP_LOGW("when self dim nums is 0, indices size must be 1 but got %zu", indicesSize);
    }
    bool indicesSizeInvalid = (selfDimNum > 0 && indicesSize > selfDimNum);
    if (indicesSizeInvalid) {
        OP_LOGW(
            "indices size must not greater than self dim nums, bug got indices size %zu and self dim nums %zu",
            indicesSize, selfDimNum);
    }
}

static inline bool CheckShapeDim(const aclTensor* self, const aclTensorList* indices)
{
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    int64_t indicesSize = static_cast<int64_t>(indices->Size());

    for (int64_t i = 0; i < indicesSize; i++) {
        OP_CHECK_MAX_DIM((*indices)[i], MAX_DIM_LEN, return false);
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensorList* indices, const aclTensor* out)
{
    // 错误码等DFX方案细化后刷新，错误日志在check接口内打印
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckPtrValid(self, indices, out), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, indices, out), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查格式是否支持
    CHECK_RET(CheckFormat(self, indices, out), ACLNN_ERR_PARAM_INVALID);
    // 4. 检查shape是否支持
    CheckShape(self, indices);
    CHECK_RET(CheckShapeDim(self, indices), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclIntArray* GetPerm(int64_t masksNum, int64_t indicesNum, int64_t selfDimNum, aclOpExecutor* executor)
{
    FVector<int64_t, MAX_SUPPORT_DIMS_NUMS> transposeArray;
    for (int64_t i = masksNum - indicesNum; i < masksNum; i++) {
        transposeArray.emplace_back(i);
    }
    for (int64_t i = 0; i < masksNum - indicesNum; i++) {
        transposeArray.emplace_back(i);
    }
    for (int64_t i = masksNum; i < selfDimNum; i++) {
        transposeArray.emplace_back(i);
    }
    auto perm = executor->AllocIntArray(transposeArray.data(), selfDimNum);
    return perm;
}

static aclIntArray* GetPermBack(int64_t maskZerosNum, int64_t indicesDimNum, int64_t outDimNum, aclOpExecutor* executor)
{
    FVector<int64_t, MAX_SUPPORT_DIMS_NUMS> transposeArray;
    for (int64_t i = indicesDimNum; i < indicesDimNum + maskZerosNum; i++) {
        transposeArray.emplace_back(i);
    }
    for (int64_t i = 0; i < indicesDimNum; i++) {
        transposeArray.emplace_back(i);
    }
    for (int64_t i = indicesDimNum + maskZerosNum; i < outDimNum; i++) {
        transposeArray.emplace_back(i);
    }
    auto perm = executor->AllocIntArray(transposeArray.data(), outDimNum);
    return perm;
}

static const inline std::initializer_list<DataType>& GetAicoreSupportDtypeList()
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (Ops::NN::AclnnUtil::IsRegbase(curArch)) {
        return AICORE_DTYPE_SUPPORT_LIST_REGBASE;
    } else if (curArch == NpuArch::DAV_2201) {
        return AICORE_DTYPE_SUPPORT_LIST_910B;
    }
    return AICORE_DTYPE_SUPPORT_LIST;
}

bool check_index_aicore(
    const aclTensor* self, const FVector<const aclTensor*, 8>& indices, const FVector<int64_t, 8>& masks)
{
    // indices don't support bool
    for (size_t idx = 0; idx < indices.size(); idx++) {
        if (indices[idx]->GetDataType() == op::DataType::DT_BOOL) {
            return false;
        }
    }

    // indices must have same shape
    for (size_t idx = 1; idx < indices.size(); idx++) {
        if (indices[idx]->GetViewShape() != indices[idx - 1]->GetViewShape()) {
            return false;
        }
    }
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    const auto& dTypeSupportList = GetAicoreSupportDtypeList();
    if (!CheckType(self->GetDataType(), dTypeSupportList)) {
        return false;
    }

    bool isRegbase = Ops::NN::AclnnUtil::IsRegbase(curArch);
    // indices must be continuous
    if (!isRegbase) {
        for (size_t idx = 1; idx < masks.size(); idx++) {
            if (masks[idx] - masks[idx - 1] < 0) {
                return false;
            }
        }
    }

    bool dataSizeInvalid =
        ((curArch != NpuArch::DAV_2201 &&
          (!isRegbase)) &&
         self->GetViewShape().GetDimNum() == indices.size() &&
         indices[0]->GetViewShape().GetShapeSize() > DATA_LIMIT_MULTI_INDEX);
    if (dataSizeInvalid) {
        return false;
    }
    return true;
}

static op::ShapeVector GetTransposedOutputShape(
    const FVector<const aclTensor*, 8>& allDefinedIndices, const aclTensor* self, int64_t masksNum, int64_t indicesNum)
{
    // construct output shape after transpose
    op::ShapeVector transposedOutputShape;
    for (size_t i = 0; i < allDefinedIndices[0]->GetViewShape().GetDimNum(); i++) {
        transposedOutputShape.emplace_back(allDefinedIndices[0]->GetViewShape().GetDim(i));
    }
    for (int64_t i = 0; i < masksNum - indicesNum; i++) {
        transposedOutputShape.emplace_back(self->GetViewShape().GetDim(i));
    }
    for (size_t i = masksNum; i < self->GetViewShape().GetDimNum(); i++) {
        transposedOutputShape.emplace_back(self->GetViewShape().GetDim(i));
    }
    return transposedOutputShape;
}

static bool IsUseNonContiguous(const aclTensor* self, const aclTensorList* indices)
{
    if (!(Ops::NN::AclnnUtil::IsRegbase())) {
        return false;
    }
    size_t xDim = self->GetViewShape().GetDimNum();
    int64_t indicesSize = static_cast<int64_t>(indices->Size());
    bool isDimBound = xDim > DIM_BOUND_NON_CONTIGUOUS || static_cast<size_t>(indicesSize) > DIM_BOUND_NON_CONTIGUOUS;
    if (isDimBound) {
        return false;
    }
    FVector<const aclTensor*, MAX_SUPPORT_DIMS_NUMS> allDefinedIndices;
    FVector<int64_t, MAX_SUPPORT_DIMS_NUMS> masks;
    bool existIndicesNonContiguous = false;
    for (int64_t i = 0; i < indicesSize; ++i) {
        int64_t curMask = 0;
        const aclTensor* curIndice = (*indices)[i];
        if (curIndice->GetViewShape().GetShapeSize() != 0) {
            allDefinedIndices.emplace_back(curIndice);
            curMask = 1;
            if (!IsContiguous(curIndice)) {
                existIndicesNonContiguous = true;
            }
        }
        masks.push_back(curMask);
    }
    if (allDefinedIndices.size() <= 1) {
        return false;
    }
    bool selfIsNonContiguous = !IsContiguous(self);
    bool isContiguous = !selfIsNonContiguous && !existIndicesNonContiguous;
    if (isContiguous) {
        return false;
    }
    return check_index_aicore(self, allDefinedIndices, masks);
}

IndicesInfo ProcessIndicesAndMasks(const aclTensorList* indices, aclOpExecutor* executor, bool isNonContiguous){
    IndicesInfo indicesInfo;
    int64_t indicesSize = static_cast<int64_t>(indices->Size());
    // 封装输入输出Tensor
    for (int64_t i = 0; i < indicesSize; i++) {
        const aclTensor* curIndice = (*indices)[i];
        int64_t curMask = 0;
        if (curIndice->GetViewShape().GetShapeSize() != 0) {
            indicesInfo.indicesNum += 1;
            if (isNonContiguous) {
                auto indexNonContiguous = executor->CreateView(
                    curIndice, curIndice->GetViewShape(), curIndice->GetStorageShape(), curIndice->GetViewStrides(),
                    curIndice->GetViewOffset());
                indicesInfo.allDefinedIndices.emplace_back(indexNonContiguous);
            } else {
                auto indexContiguous = l0op::Contiguous(curIndice, executor);
                indicesInfo.allDefinedIndices.emplace_back(indexContiguous);
            }
            curMask = 1;
            indicesInfo.dim = i;
        }
        indicesInfo.masks.emplace_back(curMask);
        indicesInfo.masksNum += 1;
    }

    indicesInfo.indicesDim = indicesInfo.indicesNum > 0 ? indicesInfo.allDefinedIndices[0]->GetViewShape().GetDimNum() : 0UL;
    OP_LOGI("masksNum is %zu, indicesNum is %zu", indicesInfo.masksNum, indicesInfo.indicesNum);
    return indicesInfo;
}

ChooseInfo CalculateOutputShapeAndTransposeFlag(const aclTensor* self, IndicesInfo& indicesInfo){
    ChooseInfo chooseInfo;
    size_t xDim = self->GetViewShape().GetDimNum();
    auto indicesDim = indicesInfo.indicesDim;
    auto indicesNum = indicesInfo.indicesNum;

    chooseInfo.outputDim = indicesDim + xDim - static_cast<size_t>(indicesNum);
    OP_LOGI("x dim is %zu, indices dim is %zu, so output dim is %zu", xDim, indicesDim, chooseInfo.outputDim);
    if (chooseInfo.outputDim > MAX_DIM_LEN) {
        return chooseInfo;
    }
    size_t tailSize = 1;
    for (size_t i = indicesInfo.masks.size(); i < self->GetViewShape().GetDimNum(); i++) {
        int64_t dim = self->GetViewShape().GetDim(i);
        if (dim < 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "GetDim returns negative value");
            return chooseInfo;
        }
        if (tailSize > SIZE_MAX / static_cast<size_t>(dim)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "tailSize overflow detected");
            return chooseInfo;
        }
        tailSize = tailSize * static_cast<size_t>(dim);
    }
    chooseInfo.tailSize = tailSize;

    chooseInfo.isAicore = check_index_aicore(self, indicesInfo.allDefinedIndices, indicesInfo.masks);
    // small tail size will use transpose
    bool isRegbase = Ops::NN::AclnnUtil::IsRegbase();
    chooseInfo.isNeedTranspose = chooseInfo.isAicore && indicesInfo.masksNum > indicesInfo.indicesNum && chooseInfo.tailSize < TAIL_SIZE_LIMIT && !isRegbase;
    if (chooseInfo.isNeedTranspose) {
        indicesInfo.masks.clear();
        for (int64_t i = 0; i < indicesInfo.indicesNum; i++) {
           indicesInfo.masks.emplace_back(1);
        }
    }
    return chooseInfo;
}

const aclTensor* CallKernel(const aclTensor* self, const aclTensor* out, const IndicesInfo& indicesInfo, const ChooseInfo& chooseInfo, aclOpExecutor* executor) {
    const aclTensor* selfContiguous = nullptr;
    if (!chooseInfo.isNonContiguous){
        selfContiguous = l0op::Contiguous(self, executor);
        CHECK_RET(selfContiguous != nullptr, nullptr);
    }
    auto IndicesTensorList = executor->AllocTensorList(indicesInfo.allDefinedIndices.data(), indicesInfo.indicesNum);
    auto MaskArray = executor->AllocIntArray(indicesInfo.masks.data(), chooseInfo.permMasksNum);
    auto indexedSizes = executor->ConvertToTensor(MaskArray, op::ToOpDataType(ACL_INT64));
    auto outPutShapeVector = op::ToShapeVector(chooseInfo.outputShape);
    auto outPutShapeArray = executor->AllocIntArray(outPutShapeVector.data(), outPutShapeVector.size());
    auto indexedStrides = executor->ConvertToTensor(outPutShapeArray, op::ToOpDataType(ACL_INT64));

    // 调用Index算子kernel
    const aclTensor* opOut = nullptr;
    bool aicoreNeedTranspose = (chooseInfo.isAicore && chooseInfo.isNeedTranspose == 1);
    bool aicoreNotNeedTranspose = (chooseInfo.isAicore && chooseInfo.isNeedTranspose != 1);

    if (aicoreNeedTranspose) {
        auto selfDimNum = selfContiguous->GetViewShape().GetDimNum();
        auto outDimNum = out->GetViewShape().GetDimNum();
        auto indicesDimNum = indicesInfo.allDefinedIndices[0]->GetViewShape().GetDimNum();
        auto perm = GetPerm(indicesInfo.masksNum, indicesInfo.indicesNum, selfDimNum, executor);
        selfContiguous = l0op::Transpose(selfContiguous, perm, executor);
        opOut = l0op::IndexAiCore(
            selfContiguous, indexedSizes, indexedStrides, chooseInfo.outputShape, IndicesTensorList, executor);
        auto permBack = GetPermBack(indicesInfo.masksNum - indicesInfo.indicesNum, indicesDimNum, outDimNum, executor);
        opOut = l0op::Transpose(opOut, permBack, executor);
    } else if (aicoreNotNeedTranspose) {
        bool isRegbase = Ops::NN::AclnnUtil::IsRegbase();
        bool isOutEqualSelf = (!isRegbase) && (selfContiguous->GetViewShape().GetDimNum() == 0);
        if (isRegbase && indicesInfo.indicesNum == 1) {
            opOut = l0op::GatherV2(selfContiguous, indicesInfo.dim, indicesInfo.allDefinedIndices[0], executor, 0, true);
        } else if (isOutEqualSelf) {
            opOut = selfContiguous;
        } else if (chooseInfo.isNonContiguous) {
            auto newself = executor->CreateView(
                self, self->GetViewShape(), self->GetStorageShape(), self->GetViewStrides(), self->GetViewOffset());
            opOut = l0op::IndexAiCore(
                newself, indexedSizes, indexedStrides, chooseInfo.outputShape, IndicesTensorList, executor);
        } else {
            opOut = l0op::IndexAiCore(
                selfContiguous, indexedSizes, indexedStrides, chooseInfo.outputShape, IndicesTensorList, executor);
        }
    } else {
        if (selfContiguous->GetViewShape().GetDimNum() == 0) {
            opOut = selfContiguous;
        } else {
            opOut = l0op::IndexAiCpu(
                selfContiguous, indexedSizes, indexedStrides, chooseInfo.outputShape, IndicesTensorList, executor);
        }
    }
    return opOut;
}

aclnnStatus aclnnIndexGetWorkspaceSize(
    const aclTensor* self, const aclTensorList* indices, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnIndex, DFX_IN(self, indices), DFX_OUT(out));
    // 固定写法，创建OpExecutor，参数检查
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto ret = CheckParams(self, indices, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 算子的空tensor在kernel中支持，对标竞品根据算子实际情况补充
    bool selfOrOutIsEmpty = self->IsEmpty() || out->IsEmpty();
    if (selfOrOutIsEmpty) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    bool isNonContiguous = IsUseNonContiguous(self, indices);
    OP_LOGI("isNonContiguous is %s", isNonContiguous ? "true" : "false");

    IndicesInfo indicesInfo = ProcessIndicesAndMasks(indices, uniqueExecutor.get(), isNonContiguous);

    ChooseInfo chooseInfo = CalculateOutputShapeAndTransposeFlag(self, indicesInfo);
    chooseInfo.isNonContiguous = isNonContiguous;
    if (chooseInfo.outputDim > MAX_DIM_LEN) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Outputshape Dim should be less than or equal 8, but got [%zu]", chooseInfo.outputDim);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // construct output shape after transpose
    op::ShapeVector transposedOutputShape = GetTransposedOutputShape(indicesInfo.allDefinedIndices, self, indicesInfo.masksNum, indicesInfo.indicesNum);
    if (chooseInfo.isNeedTranspose == 1) {
        ToShape(transposedOutputShape, chooseInfo.outputShape);
        chooseInfo.permMasksNum = indicesInfo.indicesNum;
    } else {
        chooseInfo.outputShape = out->GetViewShape();
        chooseInfo.permMasksNum = indicesInfo.masksNum;
    }

    const aclTensor* opOut = CallKernel(self, out, indicesInfo, chooseInfo, uniqueExecutor.get());
    CHECK_RET(opOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(opOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnIndex(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnIndex);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
