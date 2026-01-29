/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_repeat_interleave.h"
#include "level0/arange.h"
#include "level0/broadcast_to.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "repeat_interleave.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transpose.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/op_api_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<op::DataType> ASCEND_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_UINT8, op::DataType::DT_INT8, op::DataType::DT_INT16,   op::DataType::DT_INT32,
    op::DataType::DT_INT64, op::DataType::DT_BOOL, op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT};

static const std::initializer_list<op::DataType> ASCEND950_DTYPE_SUPPORT_LIST_SELF = {
    op::DataType::DT_UINT8, op::DataType::DT_INT8,   op::DataType::DT_INT16,   op::DataType::DT_INT32,
    op::DataType::DT_INT64, op::DataType::DT_BOOL,   op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT,
    op::DataType::DT_BF16,  op::DataType::DT_UINT16, op::DataType::DT_UINT32,  op::DataType::DT_UINT64};

static const std::initializer_list<op::DataType> ASCEND910B_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_UINT8,   op::DataType::DT_INT8,  op::DataType::DT_INT16,
    op::DataType::DT_INT32,   op::DataType::DT_INT64, op::DataType::DT_BOOL,
    op::DataType::DT_FLOAT16, op::DataType::DT_FLOAT, op::DataType::DT_BF16};

static const std::initializer_list<DataType> GetDtypeSupportList()
{
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND950) {
        return ASCEND950_DTYPE_SUPPORT_LIST_SELF;
    }
    if (GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
        GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E) {
        return ASCEND910B_DTYPE_SUPPORT_LIST;
    }
    return ASCEND_DTYPE_SUPPORT_LIST;
}

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_REPEATS = {op::DataType::DT_INT64};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_REPEATS_TENSOR = {
    op::DataType::DT_INT64, op::DataType::DT_INT32};

static const std::string STR_SELF = "self";
static const std::string STR_REPEATS = "repeats";
static const std::string STR_OUT = "out";

// 得到tensor的维度数
static inline int64_t GetTensorDimNum(const aclTensor* self)
{
    return (int64_t)(self->GetViewShape().GetDimNum());
}

// 将dim转变为正整数
static inline int64_t WrapDim(int64_t dim, int64_t dimNum)
{
    return (dim < 0) ? dim + dimNum : dim;
}

// 取得tensor的元素个数
static inline int64_t GetTensorElementsNum(const aclTensor* self)
{
    int64_t selfSize = 1;
    op::Shape selfShape = self->GetViewShape();
    size_t selfDimNum = selfShape.GetDimNum();
    for (size_t idx = 0; idx < selfDimNum; idx++) {
        selfSize *= selfShape.GetDim(idx);
    }

    return selfSize;
}

// 取得flatten成1D以后的shape
static inline const aclIntArray* GetFlattenShape(const aclTensor* self, aclOpExecutor* executor)
{
    int64_t valuePerm[1] = {GetTensorElementsNum(self)};
    auto reshapePerm = executor->AllocIntArray(valuePerm, 1);
    return reshapePerm;
}

// 检查int repeats的取值是否为自然数
static bool CheckIntRepeatsRange(int64_t repeats)
{
    if (repeats < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "repeats %ld can not be negative.", repeats);
        return false;
    }
    return true;
}

// 检查tensor repeats是否为0维 / 1维 tensor
static inline bool CheckRepeatsDimRange(const aclTensor* repeats)
{
    OP_CHECK_MAX_DIM(repeats, 1, return false);
    return true;
}

// dim的取值范围为 [-N, N-1]。0维场景，不允许传入dim
static bool CheckDimRange(const aclTensor* self, int64_t dim)
{
    auto selfDim = GetTensorDimNum(self);
    // self为0维的场景，不允许传入dim
    if (selfDim == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dimension specified as %ld but tensor has no dimensions.", dim);
        return false;
    }

    auto dimMin = std::min(selfDim - 1, -selfDim);
    auto dimMax = std::max(selfDim - 1, -selfDim);
    if (dim > dimMax || dim < dimMin) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Dimension out of range (expected to be in range of [%ld, %ld], but got %ld).",
            dimMin, dimMax, dim);
        return false;
    }

    return true;
}

// tensor维度数不能超过8维
static inline bool CheckTensorDimSize(const aclTensor* self)
{
    OP_CHECK_MAX_DIM(self, MAX_SUPPORT_DIMS_NUMS, return false);
    return true;
}

// dim已经被校验过是否处于合理的范围内, repeats已经被校验过为 0 / 1维tensor
// 校验tensor repeats的大小是否允许 有dim场景
static bool CheckRepeatsSizeDim(const aclTensor* self, const aclTensor* repeats, int64_t dim)
{
    // 维度数为0的tensor为允许的
    int64_t repeatDim = GetTensorDimNum(repeats);
    if (repeatDim == 0) {
        return true;
    }

    // 维度数为1时，repeats的size必须为1 / self的dim维度的size
    int64_t selfDimNum = GetTensorDimNum(self);
    int64_t dimPositive = WrapDim(dim, selfDimNum);
    auto selfSize = (self->GetViewShape())[dimPositive];
    int64_t repeatSize = (repeats->GetViewShape())[0];
    if (repeatSize != selfSize && repeatSize != 1) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Tensor repeats must be size 1 or have the same size %ld as input along dim, but got %ld.", selfSize,
            repeatSize);
        return false;
    }
    return true;
}

// dim已经被校验过是否处于合理的范围内, repeats已经被校验过为 0 / 1维tensor
// 校验tensor repeats的大小是否允许 无dim场景
static bool CheckRepeatsSize(const aclTensor* self, const aclTensor* repeats)
{
    // 维度数为0的tensor为允许的
    int64_t repeatDim = GetTensorDimNum(repeats);
    if (repeatDim == 0) {
        return true;
    }

    // 维度数为1时，repeats的size必须为1 / self的元素个数
    int64_t selfElementsNum = GetTensorElementsNum(self);
    int64_t repeatSize = (repeats->GetViewShape())[0];
    if (repeatSize != selfElementsNum && repeatSize != 1) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Tensor repeats size %ld must be 1 or the number of elements in tensor self %ld.",
            repeatSize, selfElementsNum);
        return false;
    }
    return true;
}

// 校验self为空tensor场景时，tensor repeats必须为empty / 0维1元素场景 / 1维1元素场景
static bool CheckEmptyTensor(const aclTensor* self, const aclTensor* repeats)
{
    if (self->IsEmpty()) {
        if (repeats->IsEmpty()) {
            return true;
        }
        int64_t repeatDimNum = GetTensorDimNum(repeats);
        if (repeatDimNum == 0) {
            return true;
        } else if (repeatDimNum == 1 && GetTensorElementsNum(repeats) == 1) {
            return true;
        }
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "When tensor self is empty, repeats should be empty / 0 dim tensor / 1 dim \
tensor with only 1 elements, but got shape %s.",
            op::ToString(repeats->GetViewShape()).GetString());
        return false;
    }
    return true;
}

// 校验repeatInterleave算子计算后生成的tensor与out的shape一致
static bool CheckRepeatresOutShape(const aclTensor* repeatRes, aclTensor* out)
{
    OP_CHECK_SHAPE_NOT_EQUAL(repeatRes, out, return false);
    return true;
}

// 返回为[1]的intarray shape
static aclIntArray* getBaseShape(aclOpExecutor* executor)
{
    int64_t tensorShape[1] = {};
    tensorShape[0] = 1;
    auto res = executor->AllocIntArray(tensorShape, 1);
    return res;
}

// 如果self / repeats为0维tensor，那么转换为1维tensor。其余情况转成连续tensor
static const aclTensor* InitializeTensor(const aclTensor* x, aclOpExecutor* executor)
{
    auto xContiguous = l0op::Contiguous(x, executor);
    // 如果tensor为0维，则转换为1维tensor
    if (GetTensorDimNum(xContiguous) == 0) {
        auto baseShape = getBaseShape(executor);
        xContiguous = l0op::BroadcastTo(xContiguous, baseShape, executor);
    }
    return xContiguous;
}

// 将int64_t 转为为1维tensor
static const aclTensor* IntToTensor(int64_t repeats, aclOpExecutor* executor)
{
    auto socVersion = GetCurrentPlatformInfo().GetSocVersion();
    if (socVersion != SocVersion::ASCEND950) {
        auto repeatsScalar = executor->AllocScalar(repeats);
        auto repeatsTensor = executor->ConvertToTensor(repeatsScalar, op::DataType::DT_INT64);
        auto baseShape = getBaseShape(executor);
        repeatsTensor = l0op::BroadcastTo(repeatsTensor, baseShape, executor);
        return repeatsTensor;
    }

    FVector<int64_t> tmpVector = {static_cast<int64_t>(repeats)};
    auto repeatsTensor = executor->ConvertToTensor(tmpVector.data(), tmpVector.size(), op::DataType::DT_INT64);
    return repeatsTensor;
}

// 无dim, tensor repeats
static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* repeats, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    OP_CHECK_NULL(self, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(repeats, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(out, return ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    auto inputDtypeSupportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, inputDtypeSupportList, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(repeats, DTYPE_SUPPORT_LIST_REPEATS_TENSOR, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return ACLNN_ERR_PARAM_INVALID);

    // 3. repeats只能是 0D / 1D tensor
    CHECK_RET(CheckRepeatsDimRange(repeats), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查repeats的size是否为 0维 / 1维时size为1或self的元素个数
    CHECK_RET(CheckRepeatsSize(self, repeats), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查self不能超过8维
    CHECK_RET(CheckTensorDimSize(self), ACLNN_ERR_PARAM_INVALID);

    // 6. 校验self为空tensor场景时，tensor repeats必须为empty / 0维1元素场景 / 1维1元素场景
    CHECK_RET(CheckEmptyTensor(self, repeats), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// 有dim, tensor repeats
static aclnnStatus CheckParamsDim(const aclTensor* self, const aclTensor* repeats, int64_t dim, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    OP_CHECK_NULL(self, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(repeats, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(out, return ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    auto inputDtypeSupportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, inputDtypeSupportList, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_SUPPORT(repeats, DTYPE_SUPPORT_LIST_REPEATS_TENSOR, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return ACLNN_ERR_PARAM_INVALID);

    // 3. repeats只能是 0D / 1D tensor
    CHECK_RET(CheckRepeatsDimRange(repeats), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查dims的大小必须处于 [-N, N-1] 之间。0维场景不允许传入dims
    CHECK_RET(CheckDimRange(self, dim), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查repeats的size是否为 0维 / 1维时size为1或self的dim维度的size
    CHECK_RET(CheckRepeatsSizeDim(self, repeats, dim), ACLNN_ERR_PARAM_INVALID);

    // 6. 检查self不能超过8维
    CHECK_RET(CheckTensorDimSize(self), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// 无dim, int repeats
static aclnnStatus CheckParamsInt(const aclTensor* self, int64_t repeats, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    OP_CHECK_NULL(self, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(out, return ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    auto inputDtypeSupportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, inputDtypeSupportList, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return ACLNN_ERR_PARAM_INVALID);

    // 3. 检查repeats必须为自然数
    CHECK_RET(CheckIntRepeatsRange(repeats), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查self不能超过8维
    CHECK_RET(CheckTensorDimSize(self), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// 有dim, int repeats
static aclnnStatus CheckParamsDimInt(const aclTensor* self, int64_t repeats, int64_t dim, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    OP_CHECK_NULL(self, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(out, return ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    auto inputDtypeSupportList = GetDtypeSupportList();
    OP_CHECK_DTYPE_NOT_SUPPORT(self, inputDtypeSupportList, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(out, self->GetDataType(), return ACLNN_ERR_PARAM_INVALID);

    // 3. 检查repeats必须为自然数
    CHECK_RET(CheckIntRepeatsRange(repeats), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查dims的大小必须处于 [-N, N-1] 之间。0维场景不允许传入dims
    CHECK_RET(CheckDimRange(self, dim), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查self不能超过8维
    CHECK_RET(CheckTensorDimSize(self), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// repeatinterleave.Tensor
static aclnnStatus CheckParamsTensor(const aclTensor* repeats, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    OP_CHECK_NULL(repeats, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(out, return ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内
    OP_CHECK_DTYPE_NOT_SUPPORT(repeats, DTYPE_SUPPORT_LIST_REPEATS_TENSOR, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(out, repeats->GetDataType(), return ACLNN_ERR_PARAM_INVALID);

    // 3. repeats只能是 空tensor / 1D tensor
    bool repeatsIsEmpty = repeats->IsEmpty();
    bool repeatsIs1D = GetTensorDimNum(repeats) == 1;
    if (!(repeatsIsEmpty || repeatsIs1D)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "repeats must be an empty tensor / 1D tensor.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

// 在需要的场景下，进行cast操作
static const aclTensor* CastIfNeeded(const aclTensor* x, bool needCast, op::DataType dtype, aclOpExecutor* executor)
{
    if (needCast) {
        x = l0op::Cast(x, dtype, executor);
        if (x == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "After cast, x is nullptr.");
        }
    }
    return x;
}

// 无dim, tensor repeats
aclnnStatus aclnnRepeatInterleaveGetWorkspaceSize(
    const aclTensor* self, const aclTensor* repeats, int64_t outputSize, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnRepeatInterleave, DFX_IN(self, repeats, outputSize), DFX_OUT(out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParams(self, repeats, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 【self + repeats为空tensor】+ 【self为空tensor, repeats非空 1元素】 场景，直接返回空tensor
    if (self->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    // 【self非空，repeats空】-> CPU直接报错
    if (repeats->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "repeats must have the same size as input along dim.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto selfContiguous = InitializeTensor(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto repeatsContiguous = InitializeTensor(repeats, uniqueExecutor.get());
    CHECK_RET(repeatsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // repeats等于1，直接返回原tensor
    if (outputSize == GetTensorElementsNum(selfContiguous) && GetTensorElementsNum(repeatsContiguous) == 1) {
        auto flattenShape = GetFlattenShape(selfContiguous, uniqueExecutor.get());
        auto flattenSelf = l0op::Reshape(selfContiguous, flattenShape, uniqueExecutor.get());
        CHECK_RET(flattenSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewcopyResult = l0op::ViewCopy(flattenSelf, out, uniqueExecutor.get());
        CHECK_RET(viewcopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    bool needCast = selfContiguous->GetDataType() == op::DataType::DT_BOOL;
    selfContiguous = CastIfNeeded(selfContiguous, needCast, op::DataType::DT_INT8, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 因为没有dim, 将self flatten成1维
    auto flattenShape = GetFlattenShape(selfContiguous, uniqueExecutor.get());
    auto flattenSelf = l0op::Reshape(selfContiguous, flattenShape, uniqueExecutor.get());
    CHECK_RET(flattenSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto repeatInterleaveOut =
        l0op::RepeatInterleave(flattenSelf, repeatsContiguous, 0, outputSize, uniqueExecutor.get());
    CHECK_RET(repeatInterleaveOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckRepeatresOutShape(repeatInterleaveOut, out), ACLNN_ERR_PARAM_INVALID);

    repeatInterleaveOut = CastIfNeeded(repeatInterleaveOut, needCast, op::DataType::DT_BOOL, uniqueExecutor.get());
    CHECK_RET(repeatInterleaveOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewcopyResult = l0op::ViewCopy(repeatInterleaveOut, out, uniqueExecutor.get());
    CHECK_RET(viewcopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

// 有dim, tensor repeats
aclnnStatus aclnnRepeatInterleaveWithDimGetWorkspaceSize(
    const aclTensor* self, const aclTensor* repeats, int64_t dim, int64_t outputSize, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnRepeatInterleaveWithDim, DFX_IN(self, repeats, dim, outputSize), DFX_OUT(out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParamsDim(self, repeats, dim, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 【self + repeats为空tensor】+ 【self为空tensor, repeats非空 1元素】+
    // 【self非空但是repeats为0】场景，直接返回空tensor
    if (self->IsEmpty() || outputSize == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    // 【self非空，repeats空】-> CPU直接报错
    if (repeats->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "repeats must have the same size as input along dim.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto selfContiguous = InitializeTensor(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto repeatsContiguous = InitializeTensor(repeats, uniqueExecutor.get());
    CHECK_RET(repeatsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    bool needCast = selfContiguous->GetDataType() == op::DataType::DT_BOOL;
    selfContiguous = CastIfNeeded(selfContiguous, needCast, op::DataType::DT_INT8, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    int64_t selfDimNum = GetTensorDimNum(self);
    dim = WrapDim(dim, selfDimNum); // 将负数dim转为正数dim

    auto repeatInterleaveOut =
        l0op::RepeatInterleaveV2(selfContiguous, repeatsContiguous, dim, outputSize, uniqueExecutor.get());
    CHECK_RET(repeatInterleaveOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckRepeatresOutShape(repeatInterleaveOut, out), ACLNN_ERR_PARAM_INVALID);

    repeatInterleaveOut = CastIfNeeded(repeatInterleaveOut, needCast, op::DataType::DT_BOOL, uniqueExecutor.get());
    CHECK_RET(repeatInterleaveOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewcopyResult = l0op::ViewCopy(repeatInterleaveOut, out, uniqueExecutor.get());
    CHECK_RET(viewcopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

// 无dim, int repeats
aclnnStatus aclnnRepeatInterleaveIntGetWorkspaceSize(
    const aclTensor* self, int64_t repeats, int64_t outputSize, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnRepeatInterleaveInt, DFX_IN(self, repeats, outputSize), DFX_OUT(out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParamsInt(self, repeats, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor场景，直接返回
    if (self->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // repeats等于1，直接返回原tensor
    if (repeats == 1) {
        auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
        CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto flattenShape = GetFlattenShape(selfContiguous, uniqueExecutor.get());
        auto flattenSelf = l0op::Reshape(selfContiguous, flattenShape, uniqueExecutor.get());
        CHECK_RET(flattenSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewcopyResult = l0op::ViewCopy(flattenSelf, out, uniqueExecutor.get());
        CHECK_RET(viewcopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = InitializeTensor(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto repeatsTensor = IntToTensor(repeats, uniqueExecutor.get());
    CHECK_RET(repeatsTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    bool needCast = selfContiguous->GetDataType() == op::DataType::DT_BOOL;
    selfContiguous = CastIfNeeded(selfContiguous, needCast, op::DataType::DT_INT8, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 因为没有dim, 将self flatten成1维
    auto flattenShape = GetFlattenShape(selfContiguous, uniqueExecutor.get());
    auto flattenSelf = l0op::Reshape(selfContiguous, flattenShape, uniqueExecutor.get());
    CHECK_RET(flattenSelf != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto repeatInterleaveOut = l0op::RepeatInterleave(flattenSelf, repeatsTensor, 0, outputSize, uniqueExecutor.get());
    CHECK_RET(repeatInterleaveOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckRepeatresOutShape(repeatInterleaveOut, out), ACLNN_ERR_PARAM_INVALID);

    repeatInterleaveOut = CastIfNeeded(repeatInterleaveOut, needCast, op::DataType::DT_BOOL, uniqueExecutor.get());
    CHECK_RET(repeatInterleaveOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewcopyResult = l0op::ViewCopy(repeatInterleaveOut, out, uniqueExecutor.get());
    CHECK_RET(viewcopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

// 有dim, int repeats
aclnnStatus aclnnRepeatInterleaveIntWithDimGetWorkspaceSize(
    const aclTensor* self, int64_t repeats, int64_t dim, int64_t outputSize, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnRepeatInterleaveIntWithDim, DFX_IN(self, repeats, dim, outputSize), DFX_OUT(out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParamsDimInt(self, repeats, dim, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor场景 + 【self非空但是repeats为0】场景，直接返回
    if (self->IsEmpty() || outputSize == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // repeats等于1，直接返回原tensor
    if (repeats == 1) {
        auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
        auto viewcopyResult = l0op::ViewCopy(selfContiguous, out, uniqueExecutor.get());
        CHECK_RET(viewcopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto selfContiguous = InitializeTensor(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto repeatsTensor = IntToTensor(repeats, uniqueExecutor.get());
    CHECK_RET(repeatsTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);

    bool needCast = selfContiguous->GetDataType() == op::DataType::DT_BOOL;
    selfContiguous = CastIfNeeded(selfContiguous, needCast, op::DataType::DT_INT8, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    int64_t selfDimNum = GetTensorDimNum(self);
    dim = WrapDim(dim, selfDimNum);

    auto repeatInterleaveOut =
        l0op::RepeatInterleaveV2(selfContiguous, repeatsTensor, dim, outputSize, uniqueExecutor.get());
    CHECK_RET(repeatInterleaveOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckRepeatresOutShape(repeatInterleaveOut, out), ACLNN_ERR_PARAM_INVALID);

    repeatInterleaveOut = CastIfNeeded(repeatInterleaveOut, needCast, op::DataType::DT_BOOL, uniqueExecutor.get());
    CHECK_RET(repeatInterleaveOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewcopyResult = l0op::ViewCopy(repeatInterleaveOut, out, uniqueExecutor.get());
    CHECK_RET(viewcopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

// repeatInterleave.Tensor
aclnnStatus aclnnRepeatInterleaveTensorGetWorkspaceSize(
    const aclTensor* repeats, int64_t outputSize, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnRepeatInterleaveTensor, DFX_IN(repeats, outputSize), DFX_OUT(out));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = CheckParamsTensor(repeats, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 【repeats为空tensor】场景，直接返回空tensor
    if (repeats->IsEmpty() || out->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    bool needCastInt32 = repeats->GetDataType() == op::DataType::DT_INT32;
    repeats = CastIfNeeded(repeats, needCastInt32, op::DataType::DT_INT64, uniqueExecutor.get());
    CHECK_RET(repeats != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // repeats size转换为tensor
    int64_t repeatsSize = GetTensorElementsNum(repeats);
    op::Shape selfShape = {repeatsSize};
    const aclTensor* self = (uniqueExecutor)->AllocTensor(selfShape, repeats->GetDataType());
    auto start = (uniqueExecutor)->AllocScalar(0);
    auto end = (uniqueExecutor)->AllocScalar(repeatsSize);
    auto step = (uniqueExecutor)->AllocScalar(1);
    auto selfContiguous = l0op::Arange(start, end, step, self, false, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto repeatsContiguous = InitializeTensor(repeats, uniqueExecutor.get());
    CHECK_RET(repeatsContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto repeatInterleaveOut =
        l0op::RepeatInterleave(selfContiguous, repeatsContiguous, 0, outputSize, uniqueExecutor.get());
    CHECK_RET(repeatInterleaveOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckRepeatresOutShape(repeatInterleaveOut, out), ACLNN_ERR_PARAM_INVALID);

    repeatInterleaveOut =
        CastIfNeeded(repeatInterleaveOut, needCastInt32, op::DataType::DT_INT32, uniqueExecutor.get());
    CHECK_RET(repeatInterleaveOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewcopyResult = l0op::ViewCopy(repeatInterleaveOut, out, uniqueExecutor.get());
    CHECK_RET(viewcopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

// 无dim, tensor repeats
aclnnStatus aclnnRepeatInterleave(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRepeatInterleave);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

// 有dim, tensor repeats
aclnnStatus aclnnRepeatInterleaveWithDim(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRepeatInterleaveWithDim);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

// 无dim, int repeats
aclnnStatus aclnnRepeatInterleaveInt(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRepeatInterleaveInt);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

// 有dim, int repeats
aclnnStatus aclnnRepeatInterleaveIntWithDim(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRepeatInterleaveIntWithDim);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

// repeatInterleave.Tensor
aclnnStatus aclnnRepeatInterleaveTensor(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnRepeatInterleaveTensor);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
