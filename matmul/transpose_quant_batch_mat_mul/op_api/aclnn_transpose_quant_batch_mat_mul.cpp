/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_transpose_quant_batch_mat_mul.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "level0/dot.h"
#include "level0/fill.h"
#include "matmul/common/op_host/op_api/matmul.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transdata.h"
#include "level0/unsqueeze.h"
#include "transpose_quant_batch_mat_mul.h"

#include "matmul/common/op_host/op_api/cube_util.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "op_api/op_api_def.h"

using namespace std;
using namespace op;
using namespace Ops::NN;

static const std::initializer_list<op::DataType> x1_SUPPORT_LIST = {
    DataType::DT_FLOAT8_E5M2, DataType::DT_FLOAT8_E4M3FN};
static const std::initializer_list<op::DataType> x2_SUPPORT_LIST = {
    DataType::DT_FLOAT8_E5M2, DataType::DT_FLOAT8_E4M3FN};
static const std::initializer_list<op::DataType> x1_SCALE_SUPPORT_LIST = {DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> x2_SCALE_SUPPORT_LIST = {DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16, DataType::DT_BF16};
static constexpr size_t EXPECTED_DIM = 3;
static constexpr size_t EXPECTED_SCALE_DIM = 1;
static constexpr int TQBMM_VALID_K = 512;
static constexpr int TQBMM_VALID_N = 128;

inline static bool CheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* out)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

inline static bool CheckDtypeValid(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale, const aclTensor* x2Scale, const aclTensor* out,
    const int32_t dtype)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, x1_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, x2_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x1Scale, x1_SCALE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2Scale, x2_SCALE_SUPPORT_LIST, return false);
    // Only support FP16 and BF16
    OP_CHECK_DTYPE_NOT_SUPPORT(out, OUT_DTYPE_SUPPORT_LIST, return false);
    // Dtype shoulde be same with out tensor data type
    if (static_cast<int32_t>(out->GetDataType()) != dtype) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dtype should be same with out dtype");
        return false;
    }
    return true;
}

inline static bool CheckScaleValid(const aclTensor* x1Scale, const aclTensor* x2Scale, int64_t M, int64_t N)
{
    // x1Scale
    if (x1Scale != nullptr) {
        OP_LOGD("X1Scale %s", op::ToString(x1Scale->GetViewShape()).GetString());
        auto dimTensorScale = x1Scale->GetViewShape().GetDimNum();
        int64_t scaleDim0 = x1Scale->GetViewShape().GetDim(0);
        if (dimTensorScale != EXPECTED_SCALE_DIM || scaleDim0 != M) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim of x1Scale != 1 or x1Scale dim 0 != M");
            return false;
        }
    }
    // x2Scale
    if (x2Scale != nullptr) {
        OP_LOGD("X2Scale %s", op::ToString(x2Scale->GetViewShape()).GetString());
        auto dimTensorScale = x2Scale->GetViewShape().GetDimNum();
        int64_t scaleDim0 = x2Scale->GetViewShape().GetDim(0);
        if (dimTensorScale != EXPECTED_SCALE_DIM || scaleDim0 != N) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim of x2Scale != 1 or x2Scale dim 0 != N");
            return false;
        }
    }
    return true;
}

static bool CheckPermValid(const aclIntArray* permX1, const aclIntArray* permX2, const aclIntArray* permY)
{
    if (permX1->Size() != EXPECTED_DIM || permX2->Size() != EXPECTED_DIM || permY->Size() != EXPECTED_DIM) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "The dims of the perm intArray should be 3, now permX1 dim: %ld , permX2 dim: %ld,  permY dim: %ld",
            permX1->Size(), permX2->Size(), permY->Size());
        return false;
    }

    // 当前支持转置场景
    auto x1Transpose = ((*permX1)[0] == 1 && (*permX1)[1] == 0 && (*permX1)[2] == 2); // 1 0 2
    auto x2Transpose = ((*permX2)[0] == 0 && (*permX2)[1] == 1 && (*permX2)[2] == 2); // 0 1 2
    auto yTranspose = ((*permY)[0] == 1 && (*permY)[1] == 0 && (*permY)[2] == 2);     // 1 0 2

    if (!x1Transpose) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the perm of x1 for DAV_3510 should be [1, 0, 2].");
        return false;
    }
    if (!x2Transpose) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the perm of x2 for DAV_3510 should be [0, 1, 2].");
        return false;
    }
    if (!yTranspose) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the perm of y for DAV_3510 should be [1, 0, 2].");
        return false;
    }
    return true;
}

static bool CheckShapeValid(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale, const aclTensor* x2Scale,
    const aclIntArray* permX1, const aclIntArray* permX2)
{
    op::Shape x1Shape = x1->GetViewShape();
    op::Shape x2Shape = x2->GetViewShape();
    int64_t x1KDim = x1->GetViewShape().GetDim((*permX1)[2]);
    int64_t x2KDim = x2->GetViewShape().GetDim((*permX2)[1]);
    int64_t M = x1->GetViewShape().GetDim((*permX1)[1]);
    int64_t K = x2->GetViewShape().GetDim((*permX2)[1]);
    int64_t N = x2->GetViewShape().GetDim((*permX2)[2]);

    if ((x1Shape.GetDimNum() != EXPECTED_DIM) || (x2Shape.GetDimNum() != EXPECTED_DIM)) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "The dims of the two inputs should be 3, now they are %s and %s",
            op::ToString(x1Shape).GetString(), op::ToString(x2Shape).GetString());
        return false;
    }
    if (x1KDim != x2KDim) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "The k-axis of the two inputs are different %s, %s",
            op::ToString(x1Shape).GetString(), op::ToString(x2Shape).GetString());
        return false;
    }
    // Check shape k n
    if (K != TQBMM_VALID_K || N != TQBMM_VALID_N) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "The shape of the x2 is not supported, now K are %ld, and N are %ld", K, N);
        return false;
    }

    return CheckScaleValid(x1Scale, x2Scale, M, N);
}

inline static aclnnStatus CheckParams(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale, const aclTensor* x2Scale, aclTensor* out,
    int32_t dtype, const aclIntArray* permX1, const aclIntArray* permX2, const aclIntArray* permY,
    int32_t batch_split_factor)
{
    // Only support DAV_3510
    if (GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_3510) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Only support DAV_3510");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // Check null
    CHECK_RET(CheckNotNull(x1, x2, out), ACLNN_ERR_PARAM_NULLPTR);

    // Check permX1, permX2, permY
    OP_CHECK(
        permX1 != nullptr && permX2 != nullptr && permY != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "PermX1 and permX2 and permY should not be nullptr."),
        return ACLNN_ERR_PARAM_INVALID);

    // Only support Scale not null
    OP_CHECK(
        x1Scale != nullptr && x2Scale != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "X1Scale and x2Scale cannot not be nullptr currently."),
        return ACLNN_ERR_PARAM_INVALID);

    if (batch_split_factor != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Batch_split_factor[%d] should be 1 currently.", batch_split_factor);
        return ACLNN_ERR_PARAM_INVALID;
    }

    CHECK_RET(CheckDtypeValid(x1, x2, x1Scale, x2Scale, out, dtype), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckPermValid(permX1, permX2, permY), ACLNN_ERR_PARAM_INVALID);

    // check shape
    CHECK_RET(CheckShapeValid(x1, x2, x1Scale, x2Scale, permX1, permX2), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static const aclTensor* BuildTransposeQuantBatchMatMulGraph(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale, const aclTensor* x2Scale, int32_t dtype,
    int32_t groupSize, const aclIntArray* permX1, const aclIntArray* permX2, const aclIntArray* permY,
    int32_t batchSplitFactor, aclOpExecutor* executor)
{
    // 连续性转换
    auto contiguousX1 = l0op::Contiguous(x1, executor);
    OP_CHECK(
        contiguousX1 != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input x1 perprocess failed, contiguous return nullptr."),
        return nullptr);
    auto reformX1 = l0op::ReFormat(contiguousX1, op::Format::FORMAT_ND);
    OP_CHECK(
        reformX1 != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input x1 perprocess failed, reformat return nullptr."), return nullptr);

    auto contiguousX2 = l0op::Contiguous(x2, executor);
    OP_CHECK(
        contiguousX2 != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input x2 perprocess failed, contiguous return nullptr."),
        return nullptr);
    auto reformX2 = l0op::ReFormat(contiguousX2, op::Format::FORMAT_ND);
    OP_CHECK(
        reformX2 != nullptr,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input x2 perprocess failed, reformat return nullptr."), return nullptr);

    auto contiguousX1Scale = x1Scale;
    if (contiguousX1Scale != nullptr) {
        contiguousX1Scale = l0op::Contiguous(x1Scale, executor);
        OP_CHECK(
            contiguousX1Scale != nullptr,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input x1Scale perprocess failed, contiguous return nullptr."),
            return nullptr);
    }
    auto contiguousX2Scale = x2Scale;
    if (contiguousX2Scale != nullptr) {
        contiguousX2Scale = l0op::Contiguous(x2Scale, executor);
        OP_CHECK(
            contiguousX2Scale != nullptr,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "The input x2Scale perprocess failed, contiguous return nullptr."),
            return nullptr);
    }

    // Invoke tqbmmm l0 api
    return l0op::TransposeQuantBatchMatMul(
        reformX1, reformX2, nullptr, contiguousX1Scale, contiguousX2Scale, dtype, groupSize, permX1, permX2, permY,
        batchSplitFactor, executor);
}

aclnnStatus aclnnTransposeQuantBatchMatMulGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* bias, const aclTensor* x1Scale, const aclTensor* x2Scale,
    const int32_t dtype, const int32_t groupSize, const aclIntArray* permX1, const aclIntArray* permX2,
    const aclIntArray* permY, const int32_t batchSplitFactor, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnTransposeQuantBatchMatMul,
        DFX_IN(x1, x2, bias, x1Scale, x2Scale, dtype, groupSize, permX1, permX2, permY, batchSplitFactor),
        DFX_OUT(out));

    // 固定写法, 创建OpExecutor
    auto unique_executor = CREATE_EXECUTOR();
    CHECK_RET(unique_executor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 入参检查
    auto ret = CheckParams(x1, x2, x1Scale, x2Scale, out, dtype, permX1, permX2, permY, batchSplitFactor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // 空tensor 处理
    if (x1->IsEmpty() || x2->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Not support empty tensor!");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 当前暂不支持bias
    if (bias != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Not support bias!");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 构建matmul计算图
    const aclTensor* tbmmOut = nullptr;
    tbmmOut = BuildTransposeQuantBatchMatMulGraph(
        x1, x2, x1Scale, x2Scale, dtype, groupSize, permX1, permX2, permY, batchSplitFactor, unique_executor.get());
    CHECK_RET(tbmmOut != nullptr, ACLNN_ERR_PARAM_INVALID);

    tbmmOut = l0op::Cast(tbmmOut, out->GetDataType(), unique_executor.get());
    CHECK_RET(tbmmOut != nullptr, ACLNN_ERR_PARAM_INVALID);
    auto viewCopyResult = l0op::ViewCopy(tbmmOut, out, unique_executor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_PARAM_INVALID);

    *workspaceSize = unique_executor->GetWorkspaceSize();
    unique_executor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnTransposeQuantBatchMatMul(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnTransposeQuantBatchMatMul);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
