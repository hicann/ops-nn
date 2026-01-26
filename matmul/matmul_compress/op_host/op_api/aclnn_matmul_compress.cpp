/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_matmul_compress.h"
#include "matmul_compress.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "matmul/common/op_host/op_api/cube_util.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/make_op_executor.h"
#include "op_api/op_api_def.h"
#include "aclnn_kernels/transdata.h"

using namespace op;
namespace {
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> BIAS_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> COMPRESSINDEX_DTYPE_SUPPORT_LIST = {DataType::DT_INT8};
static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16};

inline static bool CheckNotNull(const aclTensor* x, const aclTensor* weight, const aclTensor* bias,
                                const aclTensor* compressIndex, const aclTensor *out)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(weight, return false);
    OP_CHECK_NULL(bias, return false);
    OP_CHECK_NULL(compressIndex, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

inline static bool CheckDtypeValid(const aclTensor* x, const aclTensor* weight, const aclTensor* bias,
                                   const aclTensor* compressIndex, const aclTensor *out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(weight, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(bias, BIAS_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(compressIndex, COMPRESSINDEX_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, OUT_DTYPE_SUPPORT_LIST, return false);
    return true;
}

inline static bool CheckShapeValid(const aclTensor* x, const aclTensor* weight, const aclTensor* bias,
                                   const aclTensor* compressIndex, const aclTensor *out)
{
    CHECK_RET(x->GetViewShape().GetDimNum() == 2, ACLNN_ERR_INNER_NULLPTR); // 检查x是否是2维
    CHECK_RET(weight->GetViewShape().GetDimNum() == 1, ACLNN_ERR_INNER_NULLPTR); // 检查weight是否是1维
    CHECK_RET(bias->GetViewShape().GetDimNum() == 1, ACLNN_ERR_INNER_NULLPTR); // 检查bias是否是1维
    CHECK_RET(compressIndex->GetViewShape().GetDimNum() == 1, ACLNN_ERR_INNER_NULLPTR); // 检查compressIndex是否是1维
    CHECK_RET(out->GetViewShape().GetDimNum() == 2, ACLNN_ERR_INNER_NULLPTR); // 检查out是否是2维
    CHECK_RET(out->GetViewShape().GetDim(1) == bias->GetViewShape().GetDim(0), ACLNN_ERR_INNER_NULLPTR); // 检查out最后一维是否与bias维度一致
    return true;
}

inline static aclnnStatus CheckParam(const aclTensor* x, const aclTensor* weight, const aclTensor* bias,
                                     const aclTensor* compressIndex, const aclTensor *out)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(x, weight, bias, compressIndex, out), ACLNN_ERR_PARAM_NULLPTR);
    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(x, weight, bias, compressIndex, out), ACLNN_ERR_PARAM_INVALID);
    // 3. 检查输入的数据shape是否支持
    CHECK_RET(CheckShapeValid(x, weight, bias, compressIndex, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

inline static const aclTensor *PreProcessWeight(const aclTensor *x, const op::Format& format, aclOpExecutor *executor)
{
    auto formatTensor = executor == nullptr ? const_cast<aclTensor *>(x)
                                            : executor->CreateView(x, x->GetViewShape(), x->GetViewOffset());
    formatTensor->SetViewFormat(x->GetViewFormat());
    formatTensor->SetOriginalFormat(x->GetOriginalFormat());
    formatTensor->SetStorageFormat(format);
    return formatTensor;
}

inline static const aclTensor *PreProcessX(const aclTensor *x, aclOpExecutor *executor)
{
    op::Shape newShape;
    newShape.AppendDim(x->GetStorageShape().GetDim(0)); // 第0维不变
    newShape.AppendDim(x->GetStorageShape().GetDim(1) * x->GetStorageShape().GetDim(2)); // 合并第1维和第2维
    newShape.AppendDim(x->GetStorageShape().GetDim(3)); // 第3维变为第2维
    auto reshapeTensor = executor == nullptr ? const_cast<aclTensor *>(x)
                                            : executor->CreateView(x, x->GetViewShape(), x->GetViewOffset());
    reshapeTensor->SetViewShape(x->GetViewShape());
    reshapeTensor->SetOriginalShape(x->GetOriginalShape());
    reshapeTensor->SetStorageShape(newShape);
    return reshapeTensor;
}

inline static const aclTensor *PostProcessC(const aclTensor *x, aclOpExecutor *executor)
{
    op::Shape newShape;
    newShape.AppendDim(x->GetStorageShape().GetDim(0)); // 第0维不变
    newShape.AppendDim(x->GetStorageShape().GetDim(1) / 16); // 1维分形，16为分形
    newShape.AppendDim(16); // 16为分形
    newShape.AppendDim(x->GetStorageShape().GetDim(2)); // 第2维变为第3维
    auto reshapeTensor = executor == nullptr ? const_cast<aclTensor *>(x)
                                            : executor->CreateView(x, x->GetViewShape(), x->GetViewOffset());
    reshapeTensor->SetViewShape(x->GetViewShape());
    reshapeTensor->SetOriginalShape(x->GetOriginalShape());
    reshapeTensor->SetStorageShape(newShape);
    return reshapeTensor;
}
} // namespace

aclnnStatus aclnnMatmulCompressGetWorkspaceSize(const aclTensor* x, const aclTensor* weight, const aclTensor* bias,
                                                const aclTensor* compressIndex, aclTensor* out,
                                                uint64_t* workspaceSize, aclOpExecutor** executor) {
    L2_DFX_PHASE_1(aclnnMatmulCompress, DFX_IN(x, weight, bias, compressIndex), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 入参检查
    auto ret = CheckParam(x, weight, bias, compressIndex, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (x->IsEmpty() || weight->IsEmpty() || out->IsEmpty()) {
        OP_LOGI("Returning an empty tensor without actually doing calculation");
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 调用transdata算子ND->NZ
    auto xNZ = l0op::TransData(x, Format::FORMAT_FRACTAL_NZ, 1, uniqueExecutor.get());
    CHECK_RET(xNZ != nullptr, ACLNN_ERR_INNER_NULLPTR);

    const aclTensor *xNZReshape = PreProcessX(xNZ, uniqueExecutor.get());
    const aclTensor *weightNZ = PreProcessWeight(weight, op::Format::FORMAT_FRACTAL_NZ, uniqueExecutor.get());

    // 调用MatmulCompress算子Kernel
    auto outC = l0op::MatmulCompress(xNZReshape, weightNZ, bias, compressIndex, false, true, 8, 8, uniqueExecutor.get());
    CHECK_RET(outC != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor *outCReshape = PostProcessC(outC, uniqueExecutor.get());

    // 调用transdata算子NZ->ND
    auto outND = l0op::TransData(outCReshape, Format::FORMAT_ND, 1, uniqueExecutor.get());
    CHECK_RET(outND != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto viewCopyResult = l0op::ViewCopy(outND, out, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMatmulCompress(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMatmulCompress);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}