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
#include "matmul/common/op_host/math_util.h"

using namespace std;
using namespace op;
using namespace Ops::NN;

static const std::initializer_list<op::DataType> x1_SUPPORT_LIST_FP8 = {DataType::DT_FLOAT8_E5M2,
                                                                        DataType::DT_FLOAT8_E4M3FN};
static const std::initializer_list<op::DataType> x2_SUPPORT_LIST_FP8 = {DataType::DT_FLOAT8_E5M2,
                                                                        DataType::DT_FLOAT8_E4M3FN};
static const std::initializer_list<op::DataType> x1_SUPPORT_LIST_MXFP8 = {DataType::DT_FLOAT8_E4M3FN};
static const std::initializer_list<op::DataType> x2_SUPPORT_LIST_MXFP8 = {DataType::DT_FLOAT8_E4M3FN};
static const std::initializer_list<op::DataType> x1_SCALE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT8_E8M0};
static const std::initializer_list<op::DataType> x2_SCALE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT8_E8M0};
static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16, DataType::DT_BF16};
static constexpr size_t EXPECTED_DIM = 3;
static constexpr size_t EXPECTED_SCALE_DIM = 1;
static constexpr size_t EXPECTED_MX_SCALE_DIM = 4;
static constexpr int TQBMM_VALID_K = 512;
static constexpr int TQBMM_VALID_N = 128;
static const uint64_t GROUP_M_OFFSET = 32;
static const uint64_t GROUP_N_OFFSET = 16;
static const uint64_t GROUP_MNK_BIT_SIZE = 0xFFFF;
static constexpr uint64_t K_ALIGNMENT64 = 64UL;
static const int64_t SUPPORTED_GROUP_SIZE = 32;
static const int64_t NUM_TWO = 2;
static const int64_t NUM_THREE = 3;

inline static bool CheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* out)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool IsMicroScaling(const aclTensor *x1Scale, const aclTensor *x2Scale) {
    if (x1Scale == nullptr || x2Scale == nullptr) {
        return false;
    }
    return x1Scale->GetDataType() == op::DataType::DT_FLOAT8_E8M0 &&
           x2Scale->GetDataType() == op::DataType::DT_FLOAT8_E8M0;
}

inline static bool CheckDtypeValid(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale, const aclTensor* x2Scale, const aclTensor* out,
    int32_t dtype)
{
    if (!IsMicroScaling(x1Scale, x2Scale)) {
        OP_CHECK_DTYPE_NOT_SUPPORT(x1, x1_SUPPORT_LIST_FP8, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(x2, x2_SUPPORT_LIST_FP8, return false);
    } else {
        OP_CHECK_DTYPE_NOT_SUPPORT(x1, x1_SUPPORT_LIST_MXFP8, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(x2, x2_SUPPORT_LIST_MXFP8, return false);
    }
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


inline static bool CheckScalex1Valid(const aclTensor* x1Scale, int64_t batch, int64_t m,
                                    int64_t numGroup, bool isMxFp)
{
    OP_LOGD("X1Scale %s", op::ToString(x1Scale->GetViewShape()).GetString());
    auto dimTensorScale = x1Scale->GetViewShape().GetDimNum();
    if (isMxFp) {
        if (dimTensorScale != EXPECTED_MX_SCALE_DIM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "MXFp8 Dim of x1Scale must be 4, which is %d", dimTensorScale);
            return false;
        }
        if (x1Scale->GetViewShape().GetDim(0) != m || x1Scale->GetViewShape().GetDim(1) != batch ||
            x1Scale->GetViewShape().GetDim(NUM_TWO) != numGroup ||
            x1Scale->GetViewShape().GetDim(NUM_THREE) != NUM_TWO) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The x1scale shape must be [%ld, %d, %ld, 2], which are [%ld, %ld, %d ,%d]", m, batch, numGroup,
                    x1Scale->GetViewShape().GetDim(0), x1Scale->GetViewShape().GetDim(1),
                    x1Scale->GetViewShape().GetDim(NUM_TWO), x1Scale->GetViewShape().GetDim(NUM_THREE));
            return false;
        }
    } else {
        if (dimTensorScale != EXPECTED_SCALE_DIM || x1Scale->GetViewShape().GetDim(0) != m) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim of x1Scale != 1 or x1Scale dim 0 != M");
            return false;
        }
    }
    return true;
}

inline static bool CheckScalex2Valid(const aclTensor* x2Scale, int64_t batch, int64_t n, int64_t numGroup,
                                     const aclIntArray* permX2, bool isMxFp)
{
    OP_LOGD("X2Scale %s", op::ToString(x2Scale->GetViewShape()).GetString());
    auto dimTensorScale = x2Scale->GetViewShape().GetDimNum();
    if (isMxFp) {
        int64_t scaleN = x2Scale->GetViewShape().GetDim((*permX2)[NUM_TWO]);
        int64_t scaleGroupNum = x2Scale->GetViewShape().GetDim((*permX2)[1]);
        if (dimTensorScale != EXPECTED_MX_SCALE_DIM) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "MXFp8 Dim of x2Scale must be 4, which is %d", dimTensorScale);
            return false;
        }
        std::vector<int64_t> dims = {batch, numGroup, n, NUM_TWO};
        if (x2Scale->GetViewShape().GetDim(0) != batch || scaleN != n || scaleGroupNum != numGroup ||
            x2Scale->GetViewShape().GetDim(NUM_THREE) != NUM_TWO) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                    "The x2scale shape must be [%ld, %d, %ld, 2], which are [%ld, %ld, %d ,%d]", batch,
                    dims[(*permX2)[1]], dims[(*permX2)[NUM_TWO]], x2Scale->GetViewShape().GetDim(0),
                    x2Scale->GetViewShape().GetDim(1), x2Scale->GetViewShape().GetDim(NUM_TWO),
                    x2Scale->GetViewShape().GetDim(NUM_THREE));
            return false;
        }
    } else {
        int64_t scaleDim0 = x2Scale->GetViewShape().GetDim(0);
        if (dimTensorScale != EXPECTED_SCALE_DIM || scaleDim0 != n) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim of x2Scale != 1 or x2Scale dim 0 != N");
            return false;
        }
    }
    return true;
}

inline static bool CheckScaleValid(const aclTensor* x1Scale, const aclTensor* x2Scale, int64_t batch, int64_t m,
                                   int64_t n, int64_t k, const aclIntArray* permX2, bool isMxFp)
{
    int64_t numGroup = MathUtil::CeilDivision(MathUtil::CeilDivision(k, SUPPORTED_GROUP_SIZE), NUM_TWO);
    // 对x1Scale的维度和shape信息进行校验
    if (x1Scale != nullptr && !CheckScalex1Valid(x1Scale, batch, m, numGroup, isMxFp)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x1Scale is invalid");
        return false;
    }
    // 对x2Scale的维度和shape信息进行校验
    if (x2Scale != nullptr && !CheckScalex2Valid(x2Scale, batch, n, numGroup, permX2, isMxFp)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x2Scale is invalid");
        return false;
    }
    return true;
}

static bool CheckPermValid(const aclIntArray* permX1, const aclIntArray* permX2, const aclIntArray* permY, bool isMxFp)
{
    if (permX1->Size() != EXPECTED_DIM || permX2->Size() != EXPECTED_DIM || permY->Size() != EXPECTED_DIM) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "The dims of the perm intArray should be 3, now permX1 dim: %ld , permX2 dim: %ld,  permY dim: %ld",
            permX1->Size(), permX2->Size(), permY->Size());
        return false;
    }

    // 当前支持转置场景
    auto allowedPermX1 = ((*permX1)[0] == 1 && (*permX1)[1] == 0 && (*permX1)[2] == 2); // 1 0 2
    auto allowedPermX2 = ((*permX2)[0] == 0 && (*permX2)[1] == 1 && (*permX2)[2] == 2); // 0 1 2
    auto allowedPermY = ((*permY)[0] == 1 && (*permY)[1] == 0 && (*permY)[2] == 2);     // 1 0 2
    std::string permX2ErrorInfo ="[0, 1, 2].";
    if (isMxFp) {
        allowedPermX2 = allowedPermX2 || ((*permX2)[0] == 0 && (*permX2)[1] == 2 && (*permX2)[2] == 1);
        permX2ErrorInfo = "[0, 1, 2] or [0, 2, 1].";
    } 

    if (!allowedPermX1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the perm of x1 should be [1, 0, 2].");
        return false;
    }
    if (!allowedPermX2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the perm of x2 should be %s", permX2ErrorInfo.c_str());
        return false;
    }
    if (!allowedPermY) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the perm of y should be [1, 0, 2].");
        return false;
    }
    return true;
}

static bool CheckShapeValid(const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale,
                            const aclTensor* x2Scale, const aclIntArray* permX1, const aclIntArray* permX2, bool isMxFp)
{
    op::Shape x1Shape = x1->GetViewShape();
    op::Shape x2Shape = x2->GetViewShape();
    int64_t x1KDim = x1->GetViewShape().GetDim((*permX1)[2]);
    int64_t x2KDim = x2->GetViewShape().GetDim((*permX2)[1]);
    int64_t batch = x1->GetViewShape().GetDim((*permX1)[0]);
    int64_t m = x1->GetViewShape().GetDim((*permX1)[1]);
    int64_t n = x2->GetViewShape().GetDim((*permX2)[2]);

    if (x1KDim != x2KDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The k-axis of the two inputs are different, now x1KDim are %ld, and x2KDim are %ld", x1KDim, x2KDim);
        return false;
    }
    if (!isMxFp) {
        // Check shape k n
        if (x1KDim != TQBMM_VALID_K || n != TQBMM_VALID_N) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The shape of the x2 is not supported, now K are %ld, and N are %ld",
                    x1KDim, n);
            return false;
        }
    } else {
        if (x1KDim % K_ALIGNMENT64 != 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "K must be a multiple of 64, now K are %ld", x1KDim);
            return false;
        }
    }

    return CheckScaleValid(x1Scale, x2Scale, batch, m, n, x1KDim, permX2, isMxFp);
}

static inline bool validGroupSize(uint64_t groupSizeM, uint64_t groupSizeN, uint64_t groupSizeK)
{
    return (groupSizeM == 0 || groupSizeM == 1) && (groupSizeN == 0 || groupSizeN == 1) &&
           groupSizeK == SUPPORTED_GROUP_SIZE;
}

static inline bool InferGroupSize(int64_t& groupSize, bool isMxFp)
{
    uint64_t groupSizeK = static_cast<uint64_t>(groupSize) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeN = (static_cast<uint64_t>(groupSize) >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeM = (static_cast<uint64_t>(groupSize) >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;
    if (isMxFp && !validGroupSize(groupSizeM, groupSizeN, groupSizeK)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The valid groupSizeM and groupSizeN must be 0 or 1,the valid groupSizeK must be 32.", groupSizeM,
                groupSizeN);
        return false;
    }
    OP_LOGD("Infered groupSize: groupSizeM: %lu, groupSizeN: %lu, groupSizeK: %lu.", groupSizeM, groupSizeN,
            groupSizeK);
    groupSize = groupSizeK;
    return true;
}

inline static aclnnStatus CheckParams(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale, const aclTensor* x2Scale, aclTensor* out,
    int32_t dtype, const aclIntArray* permX1, const aclIntArray* permX2, const aclIntArray* permY,
    int32_t batch_split_factor, int64_t groupSize)
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

    op::Shape x1Shape = x1->GetViewShape();
    op::Shape x2Shape = x2->GetViewShape();
    if ((x1Shape.GetDimNum() != EXPECTED_DIM) || (x2Shape.GetDimNum() != EXPECTED_DIM)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dims of the two inputs should be 3, now they are %s and %s",
                op::ToString(x1Shape).GetString(), op::ToString(x2Shape).GetString());
        return false;
    }
    if (batch_split_factor != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Batch_split_factor should be 1 currently.", batch_split_factor);
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(CheckDtypeValid(x1, x2, x1Scale, x2Scale, out, dtype), ACLNN_ERR_PARAM_INVALID);

    bool isMxFp = IsMicroScaling(x1Scale, x2Scale);
    CHECK_RET(CheckPermValid(permX1, permX2, permY, isMxFp), ACLNN_ERR_PARAM_INVALID);

    // check shape
    CHECK_RET(CheckShapeValid(x1, x2, x1Scale, x2Scale, permX1, permX2, isMxFp), ACLNN_ERR_PARAM_INVALID);

    // MxFp8场景groupSize判断
    CHECK_RET(InferGroupSize(groupSize, isMxFp), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static const aclTensor* BuildTransposeQuantBatchMatMulGraph(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale, const aclTensor* x2Scale, int32_t dtype,
    int64_t groupSize, const aclIntArray* permX1, const aclIntArray* permX2, const aclIntArray* permY,
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
    int32_t dtype, int64_t groupSize, const aclIntArray* permX1, const aclIntArray* permX2,
    const aclIntArray* permY, int32_t batchSplitFactor, aclTensor* out, uint64_t* workspaceSize,
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
    auto ret = CheckParams(x1, x2, x1Scale, x2Scale, out, dtype, permX1, permX2, permY, batchSplitFactor, groupSize);
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
