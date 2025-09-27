/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <climits>
#include <cmath>
#include <bitset>
#include "level0/lp_norm_reduce_v2.h"
#include "level0/lp_norm_update_v2.h"
#include "lp_norm_v2.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/fill.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "aclnn_norm.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_MASK_LEN = 64;
constexpr size_t MAX_DIM_LEN = 8;
constexpr float INT_MAX_F = static_cast<float>(INT_MAX);
constexpr float INT_MIN_F = static_cast<float>(INT_MIN);

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static inline bool CheckNotNull(
    const aclTensor* self, const aclScalar* pScalar, const aclIntArray* dim, const aclTensor* out)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(pScalar, return false);
    OP_CHECK_NULL(dim, return false);
    OP_CHECK_NULL(out, return false);

    return true;
}

static inline bool CheckSocVersionIsSupportBf16(void)
{
    return GetCurrentPlatformInfo().GetSocVersion() >= SocVersion::ASCEND910B &&
           GetCurrentPlatformInfo().GetSocVersion() <= SocVersion::ASCEND910E;
}

static inline bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    if (!CheckSocVersionIsSupportBf16() && (self->GetDataType() == op::DataType::DT_BF16)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Self dtype DT_BF16 not support in current soc version.");
        return false;
    }

    OP_CHECK_DTYPE_NOT_SUPPORT(self, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST, return false);
    return true;
}

static inline bool CheckPromoteType(const aclTensor* self, const aclTensor* out)
{
    // 检查self和other能否做数据类型推导
    op::DataType promoteType = op::PromoteType(self->GetDataType(), out->GetDataType());
    if (promoteType == DataType::DT_UNDEFINED) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Input dtype %s and output dtype %s can not promote dtype.",
            op::ToString(self->GetDataType()).GetString(), op::ToString(out->GetDataType()).GetString());
        return false;
    }
    // 检查推导后的数据类型是否能转换为输出的数据类型
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(promoteType, out->GetDataType(), return false);
    return true;
}

static bool CheckDimValid(const aclTensor* self, const aclIntArray* dim)
{
    auto selfViewShape = self->GetViewShape();
    auto selfDimNum = static_cast<int64_t>(selfViewShape.GetDimNum());
    if (selfDimNum <= 0) {
        // self为标量时，dim范围为[-1, 0]
        selfDimNum = 1;
    }
    // 获取dim元素
    for (size_t i = 0; i < dim->Size(); i++) {
        if (dim->operator[](i) >= selfDimNum || dim->operator[](i) < (-selfDimNum)) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "provided dim %ld not in the range of input tensor size %ld.",
                dim->operator[](i), selfDimNum);
            return false;
        }
    }

    // dim值不能重复
    uint64_t dimMask[64] = {0};
    for (size_t i = 0; i < dim->Size(); i++) {
        auto dimValue = dim->operator[](i);
        // 负数维度转成正值后 在mask中判断重复
        if (dim->operator[](i) < 0) {
            dimValue = dim->operator[](i) + selfDimNum;
        }
        if (dimMask[dimValue] == 1) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim %ld appears multiple times in the list of dims", dim->operator[](i));
            return false;
        } else {
            dimMask[dimValue] = 1;
        }
    }
    return true;
}

static inline int64_t GetPosDim(int64_t dim, int64_t selfdimNum)
{
    if (selfdimNum <= 0) {
        selfdimNum = 1;
    }
    return dim >= 0 ? dim : dim + selfdimNum;
}

static void NormInferShape(const op::Shape& selfShape, const aclIntArray* dim, bool keepdim, op::Shape& expectShape)
{
    bitset<MAX_MASK_LEN> dimMask = bitset<MAX_MASK_LEN>();
    // dim为空时，所有轴都视为mask，与竞品一致
    if (dim->Size() == 0) {
        dimMask.flip();
    }
    for (size_t i = 0; i < dim->Size(); i++) {
        int64_t index = GetPosDim(dim->operator[](i), selfShape.GetDimNum());
        // 前序已校验， 此处dim不会重复
        dimMask.set(index);
    }

    for (size_t i = 0; i < selfShape.GetDimNum(); i++) {
        if (!dimMask[i]) {
            expectShape.AppendDim(selfShape.GetDim(i));
        } else if (keepdim) {
            // dim为空时，所有轴reduce
            expectShape.AppendDim(1);
        }
    }
}

static inline bool CheckShape(const aclTensor* self, const aclTensor* out, const aclIntArray* dim, bool keepdim)
{
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(out, MAX_DIM_LEN, return false);

    // 是否满足由self，dim，keepdim推导得到的shape
    op::Shape expectShape;
    NormInferShape(self->GetViewShape(), dim, keepdim, expectShape);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(out, expectShape, return false);
    return true;
}

static bool IsFloatEqual(float a, float b, float epsilon = 1e-6f)
{
    return std::fabs(a - b) < epsilon;
}

static float CalculateValP(const aclScalar* pScalar)
{
    float pVal = pScalar->ToFloat();
    if ((std::isinf(pVal) && pVal > 0) || IsFloatEqual(pVal, INT_MAX_F)) {
        return static_cast<float>(INT_MAX);
    } else if ((std::isinf(pVal) && pVal < 0) || IsFloatEqual(pVal, INT_MIN_F)) {
        return static_cast<float>(INT_MIN);
    } else {
        return static_cast<float>(pVal);
    }
}

static aclnnStatus CheckParams(
    const aclTensor* self, const aclScalar* pScalar, const aclIntArray* dim, bool keepdim, const aclTensor* out)
{
    // 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, pScalar, dim, out), ACLNN_ERR_PARAM_NULLPTR);

    // 检查输入的数据类型是否支持在API支持的数据类型范围之内， 需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 检查到输入输出dtype不同时，精度转换是否合理
    CHECK_RET(CheckPromoteType(self, out), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查reduce的轴是否超出self维度范围
    CHECK_RET(CheckDimValid(self, dim), ACLNN_ERR_PARAM_INVALID);

    // 检查数据排布是否满足要求
    CHECK_RET(CheckShape(self, out, dim, keepdim), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus FillScalar(aclTensor* out, float val, aclOpExecutor* executor)
{
    FVector<int64_t> shape;
    size_t dimNum = out->GetViewShape().GetDimNum();
    for (size_t idx = 0; idx < dimNum; idx++) {
        int64_t tmpVal = out->GetViewShape().GetDim(idx);
        shape.push_back(tmpVal);
    }
    auto dims = executor->ConvertToTensor(shape.data(), shape.size(), DataType::DT_INT64);
    auto shapeArray = executor->AllocIntArray(shape.data(), shape.size());

    FVector<float> valVector = {val};
    auto valTensor = executor->ConvertToTensor(valVector.data(), valVector.size(), out->GetDataType());
    auto fillOut = l0op::Fill(dims, valTensor, shapeArray, executor);
    CHECK_RET(fillOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(fillOut, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNormGetWorkspaceSize(
    const aclTensor* self, const aclScalar* pScalar, const aclIntArray* dim, bool keepdim, aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnNorm, DFX_IN(self, pScalar, dim, keepdim), DFX_OUT(out));

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, pScalar, dim, keepdim, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 对于空tensor在kernal中的处理，对标竞品场景进行实现
    if (self->IsEmpty()) {
        ret = FillScalar(out, 0, uniqueExecutor.get());
        CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 获取attr p的值
    auto p = CalculateValP(pScalar);
    // 设置评价精度误差(精度为0)
    auto ops = static_cast<float>(0);

    // [Contiguous] 将输入的self转换为连续
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // On 910B or later chips: self (-> cast) -> LpNormV2 -> out
    if (CheckSocVersionIsSupportBf16()) {
        // [Cast] 将计算结果转化为out的数据类型
        op::DataType promoteType = op::PromoteType(self->GetDataType(), out->GetDataType());
        auto selfContiguousCast = l0op::Cast(selfContiguous, promoteType, uniqueExecutor.get());
        CHECK_RET(selfContiguousCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // [LpNormV2] 调用LpNormV2算子kernel function(AI core算子)
        auto normOut = l0op::LpNormV2(selfContiguousCast, p, dim, keepdim, ops, uniqueExecutor.get());
        CHECK_RET(normOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // [ViewCopy] 将计算结果拷贝到输出out上，可能是非连续的tensor
        auto viewCopyResult = l0op::ViewCopy(normOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    } else {
        // On 910A and 310P chips: self -> fp32Self -> LpNormReduceV2 -> LpNormUpdateV2 -> Cast
        // 类型转换为fp32
        const aclTensor* fp32Self(selfContiguous);
        if (selfContiguous->GetDataType() != static_cast<op::DataType>(ACL_FLOAT)) {
            fp32Self = l0op::Cast(fp32Self, static_cast<op::DataType>(ACL_FLOAT), uniqueExecutor.get());
        }
        CHECK_RET(fp32Self != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // [LpNormReduceV2] 调用LpNormReduceV2算子kernel function(AI core算子)
        auto reduceOut = l0op::LpNormReduceV2(fp32Self, p, dim, keepdim, ops, uniqueExecutor.get());
        CHECK_RET(reduceOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // [LpNormUpdateV2] 调用LpNormUpdateV2算子kernel function(AI core算子)
        auto updateOut = l0op::LpNormUpdateV2(reduceOut, p, ops, uniqueExecutor.get());
        CHECK_RET(updateOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // [Cast] 将计算结果转化为out的数据类型
        auto castOut = l0op::Cast(updateOut, out->GetDataType(), uniqueExecutor.get());
        CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

        // [ViewCopy] 将计算结果拷贝到输出out上，可能是非连续的tensor
        auto viewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // 获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNorm(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnNorm);

    // 调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
