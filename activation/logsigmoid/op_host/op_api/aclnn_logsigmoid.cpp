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
 * \file aclnn_logsigmoid.cpp
 * \brief
 */

#include "aclnn_logsigmoid.h"
#include "opdev/op_dfx.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "logsigmoid.h"
#include "aclnn_kernels/reshape.h"
#include "op_api/level2_base_caculation.h"
using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> SELF_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ASCENDBF16_INPUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> OUT_BF16_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

inline static aclIntArray* GetTensorShape(const aclTensor* x, aclOpExecutor* executor) {
    auto shape = x->GetViewShape();
    size_t dimSize = x->GetViewShape().GetDimNum();

    std::vector<int64_t> valuePerm(dimSize);
    for (size_t i = 0; i < dimSize; i++) {
        valuePerm[i] = shape[i];
    }

    auto perm = executor->AllocIntArray(valuePerm.data(), dimSize);
    return perm;
}

inline static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    // 检查self和out的数据类型是否在LogSigmoid算子的支持列表内
    auto inputSupportList = GetDtypeSupportListV2(ASCENDBF16_INPUT_DTYPE_SUPPORT_LIST, SELF_DTYPE_SUPPORT_LIST);
    auto outputSupportList = GetDtypeSupportListV2(OUT_BF16_DTYPE_SUPPORT_LIST, OUT_DTYPE_SUPPORT_LIST);
    OP_CHECK_DTYPE_NOT_SUPPORT(self, inputSupportList, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, outputSupportList, return false);

    // 检查self和out数据类型是否一样
    OP_CHECK_DTYPE_NOT_MATCH(self, out->GetDataType(), return false);

    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const aclTensor* out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull2Tensor(self, out), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. ND 算子不检查格式
    // 4. 检查self和out的shape是否一致
    CHECK_RET(CheckSameShapeNotlimit1In1Out(self, out), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

static aclnnStatus ExecLogSigmoidGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                                  aclOpExecutor** executor)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(self, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor在kernel中支持，对标竞品根据算子实际情况补充
    if (self->IsEmpty()) {
        // 根据实际支持情况补充
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入self转换成连续的tensor
    auto selfContiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 当输入维度大于8时，reshape成一个长的1维tensor
    size_t dimSize = self->GetViewShape().GetDimNum();
    auto shapeOriDetial = GetTensorShapeActivation(selfContiguous, uniqueExecutor.get());
    auto reshapeSelf = ReshapeSelfValueGetActivation(self, dimSize, selfContiguous, uniqueExecutor);

    // 调用LogSigmoid算子Kernel
    auto OpOut = l0op::LogSigmoid(reshapeSelf, uniqueExecutor.get());
    CHECK_RET(OpOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 把长1维tensor reshape回原来的维度
    auto reshapeSwishGradOpOut = OpOut;
    if (dimSize > MAX_SUPPORT_DIMS_NUMS) {
        reshapeSwishGradOpOut = ReshapeLongTensorActivation(OpOut, uniqueExecutor.get(), dimSize, shapeOriDetial);
    }

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(reshapeSwishGradOpOut, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLogSigmoidGetWorkspaceSize(const aclTensor* self, aclTensor* out, uint64_t* workspaceSize,
                                            aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnLogSigmoid, DFX_IN(self), DFX_OUT(out));
    return ExecLogSigmoidGetWorkspaceSize(self, out, workspaceSize, executor);
}

aclnnStatus aclnnLogSigmoidForwardGetWorkspaceSize(const aclTensor* self, aclTensor* out, aclTensor* buffer,
                                                   uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnLogSigmoidForward, DFX_IN(self), DFX_OUT(out, buffer));
    return ExecLogSigmoidGetWorkspaceSize(self, out, workspaceSize, executor);
}

aclnnStatus aclnnLogSigmoid(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnLogSigmoid);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

aclnnStatus aclnnLogSigmoidForward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                   const aclrtStream stream)
{
    // 固定写法，调用框架能力，完成计算
    L2_DFX_PHASE_2(aclnnLogSigmoidForward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
