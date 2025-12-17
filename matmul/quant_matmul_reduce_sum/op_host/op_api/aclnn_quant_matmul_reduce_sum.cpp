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
 * \file aclnn_quant_matmul_reduce_sum.cpp
 * \brief
 */
#include "aclnn_quant_matmul_reduce_sum.h"
#include "quant_matmul_reduce_sum.h"

#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"

using namespace op;

namespace {
constexpr size_t X_DIM_NUM = 3;
constexpr size_t X2_SCALE_DIM_NUM = 1;
constexpr size_t X1_SCALE_DIM_NUM = 2;

class AclnnQuantMatmulReduceSum
{
public:
    AclnnQuantMatmulReduceSum() = default;
    ~AclnnQuantMatmulReduceSum() = default;
    void InitInputSeries1(
        const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale, const aclTensor* x2Scale,
        const aclTensor* yScale)
    {
        x1_ = x1;
        x2_ = x2;
        x1Scale_ = x1Scale;
        x2Scale_ = x2Scale;
        yScale_ = yScale;
    }

    void InitInputSeries2(
        const aclTensor* x1Offset, const aclTensor* x2Offset, const aclTensor* yOffset, const aclTensor* bias)
    {
        x1Offset_ = x1Offset;
        x2Offset_ = x2Offset;
        yOffset_ = yOffset;
        bias_ = bias;
    }

    void InitInputSeries3(bool transposeX1, bool transposeX2, int64_t groupSize, const aclIntArray* dims, bool keepDims)
    {
        transposeX1_ = transposeX1;
        transposeX2_ = transposeX2;
        groupSize_ = groupSize;
        dims_ = dims;
        keepDims_ = keepDims;
    }

    void InitOutputAndExcutor(aclTensor* out, aclOpExecutor* executor)
    {
        out_ = out;
        executor_ = executor;
    }

    aclnnStatus CheckInputs() const
    {
        auto ret = CheckInputExistence();
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        ret = CheckInputAttr();
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        ret = CheckDtype();
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        ret = CheckFormat();
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        ret = CheckInputShape();
        if (ret != ACLNN_SUCCESS) {
            return ret;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus PreProcess() const
    {
        // original shape must be set before contiguous
        x1_->SetOriginalShape(x1_->GetViewShape());
        x2_->SetOriginalShape(x2_->GetViewShape());
        x1Scale_->SetOriginalShape(x1Scale_->GetViewShape());
        x2Scale_->SetOriginalShape(x2Scale_->GetViewShape());
        return ACLNN_SUCCESS;
    }

    aclnnStatus ProcessL0() const
    {
        auto x1 = l0op::Contiguous(x1_, executor_);
        auto x1Scale = l0op::Contiguous(x1Scale_, executor_);
        auto x2Scale = l0op::Contiguous(x2Scale_, executor_);

        // 调用l0算子进行计算
        int64_t dtype = static_cast<int64_t>(out_->GetDataType());
        auto matmulRet = l0op::QuantMatmulReduceSum(
            x1, x2_, dims_, bias_, x1Scale, x2Scale, yScale_, x1Offset_, x2Offset_, yOffset_, nullptr, dtype,
            transposeX1_, transposeX2_, groupSize_, keepDims_, executor_);
        if (matmulRet == nullptr) {
            return ACLNN_ERR_INNER_NULLPTR;
        }

        // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
        auto viewCopyResult = l0op::ViewCopy(matmulRet, out_, executor_);
        if (viewCopyResult == nullptr) {
            return ACLNN_ERR_INNER_NULLPTR;
        }

        return ACLNN_SUCCESS;
    }

protected:
    const aclTensor* x1_ = nullptr;
    const aclTensor* x2_ = nullptr;
    const aclTensor* x1Scale_ = nullptr;
    const aclTensor* x2Scale_ = nullptr;
    const aclTensor* yScale_ = nullptr;
    const aclTensor* x1Offset_ = nullptr;
    const aclTensor* x2Offset_ = nullptr;
    const aclTensor* yOffset_ = nullptr;
    const aclTensor* bias_ = nullptr;
    bool transposeX1_ = false;
    bool transposeX2_ = false;
    int64_t groupSize_ = -1;
    const aclIntArray* dims_ = nullptr;
    bool keepDims_ = false;
    aclTensor* out_ = nullptr;
    aclOpExecutor* executor_ = nullptr;

private:
    aclnnStatus CheckInputExistence() const
    {
        // 必需的参数
        OP_CHECK_NULL(x1_, return ACLNN_ERR_PARAM_NULLPTR);
        OP_CHECK_NULL(x2_, return ACLNN_ERR_PARAM_NULLPTR);
        OP_CHECK_NULL(x1Scale_, return ACLNN_ERR_PARAM_NULLPTR);
        OP_CHECK_NULL(x2Scale_, return ACLNN_ERR_PARAM_NULLPTR);
        OP_CHECK_NULL(dims_, return ACLNN_ERR_PARAM_NULLPTR);
        OP_CHECK_NULL(out_, return ACLNN_ERR_PARAM_NULLPTR);
        if (executor_ == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_CREATE_EXECUTOR, "executor is nullptr.");
            return ACLNN_ERR_INNER_CREATE_EXECUTOR;
        }

        // 不支持参数
        if (yScale_ != nullptr && !yScale_->IsEmpty()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Do not support yScale now, yScale should be empty tensor or nullptr.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (x1Offset_ != nullptr && !x1Offset_->IsEmpty()) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Do not support x1Offset now, x1Offset should be empty tensor or nullptr.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (x2Offset_ != nullptr && !x2Offset_->IsEmpty()) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Do not support x2Offset now, x2Offset should be empty tensor or nullptr.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (yOffset_ != nullptr && !yOffset_->IsEmpty()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Do not support yOffset now, yOffset should be empty tensor or nullptr.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (bias_ != nullptr && !bias_->IsEmpty()) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Do not support bias now, bias should be empty tensor or nullptr.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckInputAttr() const
    {
        // Atrr参数仅支持默认值
        if (transposeX1_) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Do not support transposeX1 = True now.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (transposeX2_) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Do not support transposeX2 = True now.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (groupSize_ > 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Do not support groupSize > 0 now.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (keepDims_) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Do not support keepDims = True now.");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckInputShape() const
    {
        const auto &x1Shape = x1_->GetViewShape();
        const auto &x2Shape = x2_->GetViewShape();
        const auto &x1ScaleShape = x1Scale_->GetViewShape();
        const auto &x2ScaleShape = x2Scale_->GetViewShape();

        if (x1Shape.GetDimNum() != X_DIM_NUM || x2Shape.GetDimNum() != X_DIM_NUM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x1 dim number (%lu) and x2 dim number (%lu) should be 3.",
                    x1Shape.GetDimNum(), x2Shape.GetDimNum());
            return ACLNN_ERR_PARAM_INVALID;
        }

        auto bSizeX = x1Shape.GetDim(0);
        auto mSize = x1Shape.GetDim(1);
        auto kSizeX = x1Shape.GetDim(2);
        auto bSizeW = x2Shape.GetDim(0);
        auto kSizeW = x2Shape.GetDim(1);
        auto nSize = x2Shape.GetDim(2);

        if (bSizeX != bSizeW) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "batch size of x2 (%ld) should be same as x1 (%ld).", bSizeW, bSizeX);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (kSizeX != kSizeW) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x2 kSize(%ld) should be same as x1 kSize(%ld).", kSizeW, kSizeX);
            return ACLNN_ERR_PARAM_INVALID;
        }

        if (x2ScaleShape.GetDimNum() != X2_SCALE_DIM_NUM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x2Scale dim number should be %lu but got %lu.",
                    X2_SCALE_DIM_NUM, x2ScaleShape.GetDimNum());
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (x2ScaleShape.GetDim(0) != nSize) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x2Scale shape should be (%ld,) but got (%ld,).",
                    nSize, x2ScaleShape.GetDim(0));
            return ACLNN_ERR_PARAM_INVALID;
        }

        if (x1ScaleShape.GetDimNum() != X1_SCALE_DIM_NUM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x1Scale dim number should be %lu but got %lu.",
                    X1_SCALE_DIM_NUM, x1ScaleShape.GetDimNum());
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (x1ScaleShape.GetDim(0) != bSizeX || x1ScaleShape.GetDim(1) != mSize) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x1Scale shape should be (%ld, %ld) but got (%ld, %ld).",
                    bSizeX, mSize, x1ScaleShape.GetDim(0), x1ScaleShape.GetDim(1));
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckDtype() const
    {
        if (x1_->GetDataType() != op::DataType::DT_INT8) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "x1 data type only support DT_INT8(%d) but got %d.", op::DataType::DT_INT8,
                x1_->GetDataType());
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (x2_->GetDataType() != op::DataType::DT_INT8) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "x2 data type only support DT_INT8(%d) but got %d.", op::DataType::DT_INT8,
                x2_->GetDataType());
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (x1Scale_->GetDataType() != op::DataType::DT_FLOAT) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "x1Scale data type only support DT_FLOAT(%d) but got %d.",
                op::DataType::DT_FLOAT, x1Scale_->GetDataType());
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (x2Scale_->GetDataType() != op::DataType::DT_BF16) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "x2Scale data type only support DT_BF16(%d) but got %d.",
                op::DataType::DT_BF16, x2Scale_->GetDataType());
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (out_->GetDataType() != op::DataType::DT_BF16) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "out data type only support DT_BF16(%d) but got %d.", op::DataType::DT_BF16,
                out_->GetDataType());
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckFormat() const
    {
        const auto &x1Format = x1_->GetStorageFormat();
        const auto &x2Format = x2_->GetStorageFormat();
        const auto &x1ScaleFormat = x1Scale_->GetStorageFormat();
        const auto &x2ScaleFormat = x2Scale_->GetStorageFormat();
        const auto &outFormat = out_->GetStorageFormat();
        if (x1Format == Format::FORMAT_FRACTAL_NZ) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x1 format(%d) not support NZ(%d).", x1Format, Format::FORMAT_FRACTAL_NZ);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (x2Format != Format::FORMAT_FRACTAL_NZ) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x2 format(%d) only support NZ(%d).", x2Format, Format::FORMAT_FRACTAL_NZ);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (x1ScaleFormat == Format::FORMAT_FRACTAL_NZ) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "x1Scale format(%d) not support NZ(%d).", x1ScaleFormat,
                Format::FORMAT_FRACTAL_NZ);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (x2ScaleFormat == Format::FORMAT_FRACTAL_NZ) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "x2Scale format(%d) not support NZ(%d).", x2ScaleFormat,
                Format::FORMAT_FRACTAL_NZ);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (outFormat == Format::FORMAT_FRACTAL_NZ) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "out format(%d) not support NZ(%d).", outFormat, Format::FORMAT_FRACTAL_NZ);
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }
};

} // namespace

#ifdef __cplusplus
extern "C" {
#endif
aclnnStatus aclnnQuantMatmulReduceSumWeightNzGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale, const aclTensor* x2Scale,
    const aclTensor* yScale, const aclTensor* x1Offset, const aclTensor* x2Offset, const aclTensor* yOffset,
    const aclTensor* bias, bool transposeX1, bool transposeX2, int64_t groupSize, const aclIntArray* dims,
    bool keepDims, aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnQuantMatmulReduceSumWeightNz, DFX_IN(x1, x2, x1Scale, x2Scale), DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 参数校验和执行l0接口
    AclnnQuantMatmulReduceSum aclnnOp;
    aclnnOp.InitInputSeries1(x1, x2, x1Scale, x2Scale, yScale);
    aclnnOp.InitInputSeries2(x1Offset, x2Offset, yOffset, bias);
    aclnnOp.InitInputSeries3(transposeX1, transposeX2, groupSize, dims, keepDims);
    aclnnOp.InitOutputAndExcutor(out, uniqueExecutor.get());
    auto ret = aclnnOp.CheckInputs();
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    ret = aclnnOp.PreProcess();
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }
    ret = aclnnOp.ProcessL0();
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantMatmulReduceSumWeightNz(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantMatmulReduceSumWeightNz);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
