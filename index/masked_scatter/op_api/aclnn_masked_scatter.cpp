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
 * \file aclnn_masked_scatter.cpp
 * \brief
 */
#include "aclnn_masked_scatter.h"
#include "masked_scatter.h"
#include "index/masked_scatter_with_position/op_host/op_api/masked_scatter_with_position.h"
#include "level0/broadcast_to.h"
#include "level0/cumsum.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_util.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

/* MaskedScatter 算子的完整计算流程如下:
 * selfRef                               mask                    source
 *   |                                  |                          /
 *   \                                  /                         /
 * Contiguous(workspace_0)    Contiguous(workspace_1)     Contiguous(workspace_2)
 *      \                             /                       /
 *          \                 Cast(workspace_3)             /
 *             \                 /                        /
 *                      MaskedScatter(workspace_4)
 *                                 |
 *                             ViewCopy
 *                                  |
 *                                result
 */

constexpr size_t MAX_DIM_LEN = 8;
const int32_t INT32_MAX_VAL = 2147483647;
const int32_t Threshold = 128;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_SELFREF_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT,   op::DataType::DT_INT32,  op::DataType::DT_INT64,
    op::DataType::DT_FLOAT16, op::DataType::DT_INT16,  op::DataType::DT_INT8,
    op::DataType::DT_UINT8,   op::DataType::DT_DOUBLE, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> ASCEND910B_SELFREF_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_INT32, op::DataType::DT_INT64, op::DataType::DT_FLOAT16,
    op::DataType::DT_INT16, op::DataType::DT_INT8,  op::DataType::DT_UINT8, op::DataType::DT_DOUBLE,
    op::DataType::DT_BOOL,  op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> MASK_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_UINT8, op::DataType::DT_BOOL};

static bool CheckNotNull(const aclTensor* selfRef, const aclTensor* mask, const aclTensor* source)
{
    OP_CHECK_NULL(selfRef, return false);
    OP_CHECK_NULL(mask, return false);
    OP_CHECK_NULL(source, return false);

    return true;
}

static inline const std::initializer_list<op::DataType>& GetDtypeSupportList()
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    if (curArch == NpuArch::DAV_2201 || Ops::NN::AclnnUtil::IsRegbase(curArch)) {
        return ASCEND910B_SELFREF_DTYPE_SUPPORT_LIST;
    } else {
        return ASCEND910_SELFREF_DTYPE_SUPPORT_LIST;
    }
}

static bool CheckDtypeValid(const aclTensor* selfRef, const aclTensor* mask, const aclTensor* source)
{
    auto supportList = GetDtypeSupportList();
    // 检查selfRef的数据类型是否在masked_scatter算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(selfRef, supportList, return false);
    // 检查mask的数据类型是否在maskedScatter算子的支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(mask, MASK_DTYPE_SUPPORT_LIST, return false);
    // 检查selfRef和source的数据类型是否相同
    if (selfRef->GetDataType() != source->GetDataType()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "source dtype %s should be same with selfRef [%s].",
            op::ToString(source->GetDataType()).GetString(), op::ToString(selfRef->GetDataType()).GetString());
        return false;
    }

    return true;
}

static bool isMaskCanExpandToSelfRef(const aclTensor* selfRef, const aclTensor* mask, op::Shape* broadcastShape)
{
    op::Shape selfRefShape = selfRef->GetViewShape();
    size_t selfRefShapeLen = selfRefShape.GetDimNum();
    for (size_t i = 0; i < selfRefShapeLen; i++) {
        if (broadcastShape->GetDim(i) != selfRefShape.GetDim(i)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "target selfRef sizes %s ,source mask sizes %s",
                op::ToString(selfRef->GetViewShape()).GetString(), op::ToString(mask->GetViewShape()).GetString());
            return false;
        }
    }
    return true;
}

static bool CheckShape(const aclTensor* selfRef, const aclTensor* mask, const aclTensor* source)
{
    OP_CHECK_MAX_DIM(selfRef, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(mask, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(source, MAX_DIM_LEN, return false);

    Shape broadcastShape;
    OP_CHECK_BROADCAST_AND_INFER_SHAPE(selfRef, mask, broadcastShape, return false);

    if (!isMaskCanExpandToSelfRef(selfRef, mask, &broadcastShape)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "the number of dimensions in selfRef tensor must be greater or equal to the mask tensor");
        return false;
    }

    return true;
}

static aclnnStatus CheckParams(const aclTensor* selfRef, const aclTensor* mask, const aclTensor* source)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(selfRef, mask, source), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(selfRef, mask, source), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入形状是否满足
    CHECK_RET(CheckShape(selfRef, mask, source), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static std::vector<int64_t> ShapeToVector(const op::Shape& shape)
{
    std::vector<int64_t> vec;
    auto shapeDimNum = shape.GetDimNum();
    for (size_t i = 0; i < shapeDimNum; i++) {
        vec.push_back(shape.GetDim(i));
    }
    return vec;
}

static op::Shape VectorToShape(const std::vector<int64_t>& vec)
{
    op::Shape shape;
    auto vecSize = vec.size();
    shape.SetDimNum(vecSize);
    for (size_t i = 0; i < vecSize; i++) {
        shape.SetDim(i, vec[i]);
    }
    return shape;
}

static std::vector<int64_t> GetMaskAligned(const std::vector<int64_t>& maskShape, int64_t maxDim)
{
    std::vector<int64_t> alignVec(maxDim, 1);
    int64_t start = maxDim - maskShape.size();
    for (size_t i = 0; i < maskShape.size(); i++) {
        alignVec[start + i] = maskShape[i];
    }
    return alignVec;
}

static void CalcBroadcastInfo(
    const std::vector<int64_t>& selfShape, const std::vector<int64_t>& maskAligned, int64_t maxDim,
    std::vector<int64_t>& isBroadcast, int64_t& tagLeft0, int64_t& tagLeft1, int64_t& tagRight0, int64_t& tagRight1)
{
    isBroadcast.assign(maxDim, 0);
    for (int64_t i = 0; i < maxDim; i++) {
        if (maskAligned[i] == 1 && maskAligned[i] != selfShape[i]) {
            isBroadcast[i] = 1;
        }
    }
    tagLeft0 = -1;
    tagLeft1 = -1;
    tagRight0 = -1;
    tagRight1 = -1;

    for (int64_t i = 0; i < maxDim; i++) {
        if (isBroadcast[i] == 0) {
            tagLeft0 = i;
            break;
        }
    }

    for (int64_t i = 0; i < maxDim; i++) {
        if (isBroadcast[i] == 1) {
            tagLeft1 = i;
            break;
        }
    }

    for (int64_t i = maxDim - 1, idx = 0; i >= 0; i--, idx++) {
        if (isBroadcast[i] == 0) {
            tagRight0 = maxDim - 1 - idx;
            break;
        }
    }

    for (int64_t i = maxDim - 1, idx = 0; i >= 0; i--, idx++) {
        if (isBroadcast[i] == 1) {
            tagRight1 = maxDim - 1 - idx;
            break;
        }
    }
}

static bool IsLeftOverlap(int64_t tagLeft1, int64_t tagRight0)
{
    return tagLeft1 != -1 && tagLeft1 >= tagRight0;
}

static bool IsBroadcastAll(const std::vector<int64_t>& isBroadcast, int64_t end)
{
    for (int64_t i = 0; i < end; i++) {
        if (isBroadcast[i] != 1) {
            return false;
        }
    }
    return true;
}

static void AlignMaskLeft(std::vector<int64_t>& maskAligned, const std::vector<int64_t>& selfShape, int64_t end)
{
    for (int64_t i = 0; i < end; i++) {
        maskAligned[i] = selfShape[i];
    }
}

static void AlignMaskRight(std::vector<int64_t>& maskAligned, const std::vector<int64_t>& selfShape, int64_t start)
{
    for (int64_t i = maskAligned.size() - 1; i > start; i--) {
        maskAligned[i] = selfShape[i];
    }
}

static std::pair<std::vector<int64_t>, std::vector<int64_t>> MaskedScatterShapeFun(const std::vector<int64_t>& selfShape, const std::vector<int64_t>& maskShape,
    const std::vector<int64_t>& broadcastShape)
{
    int64_t maxDim = broadcastShape.size();
    if (selfShape == maskShape) {
        OP_LOGD("aclnn_masked_scatter the shapes of and mask are equal.");
        return {selfShape, maskShape};
    }

    OP_LOGD("aclnn_masked_scatter the shapes of and mask is broadcast to self.");
    std::vector<int64_t> maskAligned = GetMaskAligned(maskShape, maxDim);
    std::vector<int64_t> isBroadcast;
    int64_t tagLeft0 = -1;
    int64_t tagLeft1 = -1;
    int64_t tagRight0 = -1;
    int64_t tagRight1 = -1;
    CalcBroadcastInfo(selfShape, maskAligned, maxDim, isBroadcast, tagLeft0, tagLeft1, tagRight0, tagRight1);

    if (IsLeftOverlap(tagLeft1, tagRight0)) {
        return {selfShape, maskShape};
    }

    bool fullyBroadcast = tagLeft0 != -1 && tagRight1 != -1 && tagLeft0 > tagRight1 && IsBroadcastAll(isBroadcast, tagLeft0);
    if (fullyBroadcast) {
        return {selfShape, maskShape};
    }

    bool noLeft = isBroadcast[0] == 0;
    bool noRight = isBroadcast[maxDim - 1] == 0;
    if (noLeft && noRight) {
        return {selfShape, broadcastShape};
    }

    bool bothSide = isBroadcast[0] == 1 && isBroadcast[maxDim - 1] == 1;
    if (bothSide) {
        uint64_t leftCost = 1;
        uint64_t rightCost = 1;
        for (int64_t i = 0; i < tagLeft0; i++) {
            leftCost *= selfShape[i];
        }
        for (int64_t i = tagLeft0 + 1; i < maxDim; i++) {
            rightCost *= selfShape[i];
        }
        if (leftCost < rightCost) {
            AlignMaskLeft(maskAligned, selfShape, tagRight0);
        } else {
            AlignMaskRight(maskAligned, selfShape, tagLeft0);
        }
        return {selfShape, maskAligned};
    }

    bool leftBroadcast = isBroadcast[0] == 0 && isBroadcast[maxDim - 1] == 1;
    if (leftBroadcast && tagRight0 > tagLeft1) {
        for (int64_t i = tagLeft1; i < tagRight0; i++) {
            maskAligned[i] = selfShape[i];
        }
        return {selfShape, maskAligned};
    }

    bool rightBroadcast = isBroadcast[0] == 1 && isBroadcast[maxDim - 1] == 0;
    if (rightBroadcast && tagRight1 > tagLeft1) {
        for (int64_t i = tagLeft1 + 1; i <= tagRight1; i++) {
            maskAligned[i] = selfShape[i];
        }
        return {selfShape, maskAligned};
    }
    return {selfShape, broadcastShape};
}

// getBroadcastShape for l0op::BroadcastTo
static aclIntArray* GetShape(const op::Shape& broadcastShape, aclOpExecutor* executor)
{
    int64_t tensorSize = static_cast<int64_t>(broadcastShape.GetDimNum());
    std::vector<int64_t> tensorShape(tensorSize);
    for (int i = 0; i < tensorSize; i++) {
        tensorShape[i] = broadcastShape[i];
    }
    return executor->AllocIntArray(tensorShape.data(), tensorSize);
}

// 如果gradOutput或者self的shape与braodcast后的shape不一致，在进行反向计算前，先进行broadcasto操作。
static const aclTensor* BroadcastTensor(const aclTensor* self, const op::Shape& broadcastShape, aclOpExecutor* executor)
{
    // 如果self的shape与broadcast的不一致，进行BroadcastTo
    if (self->GetViewShape() != broadcastShape) {
        auto broadcastShapeIntArray = GetShape(broadcastShape, executor);
        if (broadcastShapeIntArray != nullptr) {
            return l0op::BroadcastTo(self, broadcastShapeIntArray, executor);
        }
    }
    return self;
}

static aclnnStatus ProcessBroadcast(
    aclTensor* selfRef, const aclTensor* mask, const aclTensor* source, const aclTensor** out, aclOpExecutor* executor)
{
    auto selfShape = ShapeToVector(selfRef->GetViewShape());
    auto maskShape = ShapeToVector(mask->GetViewShape());

    op::Shape broadcastShape;

    OP_CHECK_BROADCAST_AND_INFER_SHAPE(selfRef, mask, broadcastShape, return ACLNN_ERR_PARAM_INVALID);
    auto broadcastShapeVec = ShapeToVector(broadcastShape);

    auto [targetSelf, targetMask] = MaskedScatterShapeFun(selfShape, maskShape, broadcastShapeVec);
    op::Shape targetMaskShape = VectorToShape(targetMask);

    const aclTensor* selfRefProcessed = l0op::Contiguous(selfRef, executor);
    CHECK_RET(selfRefProcessed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* maskProcessed = BroadcastTensor(mask, targetMaskShape, executor);
    CHECK_RET(maskProcessed != nullptr, ACLNN_ERR_INNER_NULLPTR);
    maskProcessed = l0op::Contiguous(maskProcessed, executor);
    CHECK_RET(maskProcessed != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* sourceProcessed = source;
    if (!source->IsEmpty()) {
        sourceProcessed = l0op::Contiguous(source, executor);
        CHECK_RET(sourceProcessed != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    std::vector<int64_t> isBroadcast;
    int64_t tagLeft0 = -1;
    int64_t tagLeft1 = -1;
    int64_t tagRight0 = -1;
    int64_t tagRight1 = -1;
    int64_t maxDim = broadcastShapeVec.size();

    auto maskAligned = GetMaskAligned(targetMask, maxDim);
    CalcBroadcastInfo(targetSelf, maskAligned, maxDim, isBroadcast, tagLeft0, tagLeft1, tagRight0, tagRight1);
    bool judgeBroadcast = std::any_of(isBroadcast.begin(), isBroadcast.end(), [](int64_t val) { return val == 1; });
    bool isAB = judgeBroadcast && tagLeft1 != -1 && tagLeft1 >= tagRight0;
    bool isBA = judgeBroadcast && tagLeft0 != -1 && tagRight1 != -1 && tagLeft0 > tagRight1;
    for (int64_t i = 0; i < tagLeft0 && isBA; i++) {
        if (isBroadcast[i] != 1) {
            isBA = false;
            break;
        }
    }

    const aclTensor* opResult = nullptr;
    if (isAB || isBA) {
        OP_LOGD("aclnn_masked_scatter the call is MaskedScatterWithPosition.");
        const aclTensor* maskWithPosition = nullptr;
        auto maskSize = maskProcessed->GetViewShape().GetShapeSize();
        bool isInt64Mask_ = maskSize > INT32_MAX_VAL;
        if (!isInt64Mask_ && maskSize > Threshold) {
            maskWithPosition = l0op::Cast(maskProcessed, DataType::DT_INT32, executor);
        } else {
            maskWithPosition = l0op::Cast(maskProcessed, DataType::DT_INT64, executor);
        }

        op::Shape newSelfShape = {-1};
        const aclTensor* maskWithPositionReshape = l0op::Reshape(maskWithPosition, newSelfShape, executor);
        CHECK_RET(maskWithPositionReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

        FVector<int64_t> posIdxVec = {0};
        auto posIdx = executor->ConvertToTensor(posIdxVec.data(), posIdxVec.size(), DataType::DT_INT64);
        auto maskCumSum = l0op::Cumsum(maskWithPositionReshape, posIdx, executor);
        CHECK_RET(maskCumSum != nullptr, ACLNN_ERR_INNER_NULLPTR);
        if (!isInt64Mask_ && maskSize > Threshold) {
            maskCumSum = l0op::Cast(maskCumSum, DataType::DT_INT64, executor);
        }
        const aclTensor* maskCumSumReshape = l0op::Reshape(maskCumSum, maskWithPosition->GetViewShape(), executor);
        CHECK_RET(maskCumSumReshape != nullptr, ACLNN_ERR_INNER_NULLPTR);

        opResult = l0op::MaskedScatterWithPosition(selfRefProcessed, maskProcessed, maskCumSumReshape, sourceProcessed, executor);
    } else {
        OP_LOGD("aclnn_masked_scatter the call is MaskedScatter.");
        opResult = l0op::MaskedScatter(selfRefProcessed, maskProcessed, sourceProcessed, executor);
    }

    CHECK_RET(opResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *out = opResult;
    return ACLNN_SUCCESS;
}

static void CheckFormat(const aclTensor* self, const aclTensor* mask, const aclTensor* source)
{
    ge::Format selfStorageFormat = self->GetStorageFormat();
    ge::Format maskStorageFormat = mask->GetStorageFormat();
    ge::Format sourceStorageFormat = source->GetStorageFormat();
    if (selfStorageFormat != ge::Format::FORMAT_ND || maskStorageFormat != ge::Format::FORMAT_ND ||
        maskStorageFormat != ge::Format::FORMAT_ND) {
        OP_LOGW("aclnnInplaceMaskedScatter only support format ND.");
    }
}

// 得到tensor的维度数
static inline int64_t GetTensorDimNum(const aclTensor* self)
{
    return static_cast<int64_t>(self->GetViewShape().GetDimNum());
}

aclnnStatus aclnnInplaceMaskedScatterGetWorkspaceSize(
    aclTensor* selfRef, const aclTensor* mask, const aclTensor* source, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnInplaceMaskedScatter, DFX_IN(selfRef, mask, source), DFX_OUT(selfRef));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(selfRef, mask, source);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    CheckFormat(selfRef, mask, source);

    if (selfRef->IsEmpty() || mask->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入selfRef转换成连续的tensor
    auto selfRefContiguous = l0op::Contiguous(selfRef, uniqueExecutor.get());
    CHECK_RET(selfRefContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将输入mask转换成连续的tensor
    auto maskContiguous = l0op::Contiguous(mask, uniqueExecutor.get());
    CHECK_RET(maskContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // mask 0维转换维1维
    if (GetTensorDimNum(maskContiguous) == 0 && maskContiguous->Size() > 0) {
        maskContiguous = uniqueExecutor.get()->CreateView(maskContiguous, {1}, maskContiguous->GetViewOffset());
    }

    // input 0维转换维1维
    if (GetTensorDimNum(selfRefContiguous) == 0 && selfRefContiguous->Size() > 0) {
        selfRefContiguous = uniqueExecutor.get()->CreateView(selfRefContiguous, {1}, selfRefContiguous->GetViewOffset());
    } 

    // 当输入source不是空tensor时，将其换成连续的tensor
    const aclTensor* sourceContiguous = nullptr;
    if (source->IsEmpty()) {
        sourceContiguous = source;
    } else {
        sourceContiguous = l0op::Contiguous(source, uniqueExecutor.get());
        CHECK_RET(sourceContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // source 0维转换维1维
    if (GetTensorDimNum(sourceContiguous) == 0 && sourceContiguous->Size() > 0) {
        sourceContiguous = uniqueExecutor.get()->CreateView(sourceContiguous, {1}, sourceContiguous->GetViewOffset());
    }

    // 将输入mask的数据类型转换成bool数据类型
    auto maskBool = l0op::Cast(maskContiguous, DataType::DT_BOOL, uniqueExecutor.get());
    CHECK_RET(maskBool != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor* opOut = nullptr;
    if (Ops::NN::AclnnUtil::IsRegbase()) {    // 判断芯片架构
        opOut = l0op::MaskedScatter(selfRefContiguous, maskBool, sourceContiguous, uniqueExecutor.get());
    } else {
        auto retStatus = ProcessBroadcast(selfRef, maskBool, sourceContiguous, &opOut, uniqueExecutor.get());
        CHECK_RET(retStatus == ACLNN_SUCCESS, retStatus);
    }
    CHECK_RET(opOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(opOut, selfRef, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor); // 需要把 uniqueExecutor持有executor转移给executor
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnInplaceMaskedScatter(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnInplaceMaskedScatter);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif