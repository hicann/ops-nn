/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_transpose_batch_mat_mul.h"

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
#include "matmul/mat_mul_v3/op_host/op_api/matmul.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/transdata.h"
#include "level0/unsqueeze.h"
#include "transpose_batch_mat_mul.h"

#include "matmul/common/op_host/op_api/cube_util.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "op_api/op_api_def.h"

using namespace std;
using namespace op;
using namespace Ops::NN;
#ifdef __cplusplus
extern "C" {
#endif
static const std::initializer_list<op::DataType> x1_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};
static const std::initializer_list<op::DataType> x2_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16};
static const std::initializer_list<op::DataType> x1_SCALE_SUPPORT_LIST = {DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> x2_SCALE_SUPPORT_LIST = {DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> SCALE_DTYPE_SUPPORT_LIST = {DataType::DT_INT64, DataType::DT_UINT64};
static const std::initializer_list<op::DataType> OUT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT, DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_INT8};
static constexpr size_t EXPECTED_DIM = 3;
static constexpr int BLOCK_SIZE = 128;
static constexpr int SUPPORTED_INNER_AXIS = 65536;

inline static bool CheckNotNull(const aclTensor* x1, const aclTensor* x2, const aclTensor* out)
{
    OP_CHECK_NULL(x1, return false);
    OP_CHECK_NULL(x2, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

inline static bool CheckDtypeValid(const aclTensor* x1, const aclTensor* x2,
                                   const aclTensor* scale, const aclTensor* out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x1, x1_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(x2, x2_SUPPORT_LIST, return false);
    if (scale != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(scale, SCALE_DTYPE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(x1, x1_SCALE_SUPPORT_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(x2, x2_SCALE_SUPPORT_LIST, return false);
    }
    OP_CHECK_DTYPE_NOT_SUPPORT(out, OUT_DTYPE_SUPPORT_LIST, return false);
    return true;
}

inline static bool CheckScaleValid(const aclTensor* scale, int64_t batchN)
{
    if (scale != nullptr) {
        OP_LOGD("scale %s", op::ToString(scale->GetViewShape()).GetString());
        auto dimTensorScale = scale->GetViewShape().GetDimNum();
        int64_t scaleDim = scale->GetViewShape().GetDim(0);
        if ((dimTensorScale != 1) || (scaleDim != batchN)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dimTensorScale[%zu] != 1 or Scale dim != batch mul N", dimTensorScale);
            return false;
        }
        if (scaleDim >= SUPPORTED_INNER_AXIS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "batch mul N should be less than 65536.");
            return false;
        }
    }
    return true;
}

namespace {
static bool CheckDavidLimit(const aclTensor* scale, const aclIntArray* perm_x1, const aclIntArray* perm_x2)
{
    auto x1_need_transpose = ((*perm_x1)[0] == 1 && (*perm_x1)[1] == 0 && (*perm_x1)[2] == 2) ||
                             ((*perm_x1)[0] == 0 && (*perm_x1)[1] == 1 && (*perm_x1)[2] == 2);
    auto x2_need_transpose = ((*perm_x2)[0] == 0 && (*perm_x2)[1] == 1 && (*perm_x2)[2] == 2) ||
                             ((*perm_x2)[0] == 0 && (*perm_x2)[1] == 2 && (*perm_x2)[2] == 1);
    if (!x1_need_transpose) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the perm of x1 for ASCEND910_95 should be [0,1,2] or [1,0,2].");
        return false;
    }
    if (!x2_need_transpose) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "the perm of x2 for ASCEND910_95 should be [0,1,2] or [0,2,1].");
        return false;
    }
    if (scale != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ASCEND910_95 not support scale.");
        return false;
    }
    return true;
}
}

static bool CheckShapeValid(const aclTensor* x1, const aclTensor* x2, const aclTensor* scale,
                            const aclIntArray* perm_x1, const aclIntArray* perm_x2)
{
    op::Shape x1Shape = x1->GetViewShape();
    op::Shape x2Shape = x2->GetViewShape();
    auto dimTensor1 = x1Shape.GetDimNum();
    auto dimTensor2 = x2Shape.GetDimNum();
    int64_t x1KDim = x1->GetViewShape().GetDim((*perm_x1)[2]);
    int64_t x2KDim = x2->GetViewShape().GetDim((*perm_x2)[1]);
    int64_t batchNum = x2->GetViewShape().GetDim((*perm_x2)[0]);
    int64_t K = x2->GetViewShape().GetDim((*perm_x2)[1]);
    int64_t N = x2->GetViewShape().GetDim((*perm_x2)[2]);

    if ((dimTensor1 != EXPECTED_DIM) || (dimTensor2 != EXPECTED_DIM)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dims of the two inputs should be 3, now they are %s and %s",
                op::ToString(x1Shape).GetString(), op::ToString(x2Shape).GetString());
        return false;
    }

    if (x1KDim != x2KDim) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The k-axis of the two inputs are different %s, %s",
                op::ToString(x1Shape).GetString(), op::ToString(x2Shape).GetString());
        return false;
    }
    
    auto x1_need_transpose = ((*perm_x1)[0] == 1 && (*perm_x1)[1] == 0 && (*perm_x1)[2] == 2);
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95) {
        if (!CheckDavidLimit(scale, perm_x1, perm_x2)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ASCEND910_95 Limit.");
            return false;
        }
    } else {
        if (x1_need_transpose && x1->GetViewShape().GetDim(1) * x1KDim >= SUPPORTED_INNER_AXIS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "batch mul k should be less than 65536.");
            return false;
        }
        if (!x1_need_transpose && x1KDim >= SUPPORTED_INNER_AXIS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "K should be less than 65536.");
            return false;
        }
        if (N >= SUPPORTED_INNER_AXIS || batchNum >= SUPPORTED_INNER_AXIS) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "batch and n should be less than 65536.");
            return false;
        }
        if ((K % BLOCK_SIZE != 0) || (N % BLOCK_SIZE != 0)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                  "The shape of the x2 is not supported, now they are %ld, %ld and %ld", batchNum, K, N);
            return false;
        }
    }

    return CheckScaleValid(scale, batchNum * N);
}

static inline bool CheckMathType(const aclTensor* x1, const aclTensor* x2, int8_t cubeMathType)
{
    bool x1Float = x1->GetDataType() == DataType::DT_FLOAT;
    bool x2Float = x2->GetDataType() == DataType::DT_FLOAT;
    auto promoteType = x1Float || x2Float ? DataType::DT_FLOAT : x1->GetDataType();
    return CheckCubeMathTypeForMm(promoteType, cubeMathType);
}

inline static aclnnStatus CheckParams(const aclTensor* x1, const aclTensor* x2,
                                      const aclTensor* scale, aclTensor* out, const aclIntArray* perm_x1,
                                      const aclIntArray* perm_x2, int8_t cubeMathType,
                                      int32_t batch_split_factor)
{
    CHECK_RET(CheckNotNull(x1, x2, out), ACLNN_ERR_PARAM_NULLPTR);
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_95 && cubeMathType == -1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cubeMathType[%d] can not be -1 for ASCEND910_95",
            cubeMathType);
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(CheckMathType(x1, x2, cubeMathType), ACLNN_ERR_PARAM_INVALID);
    if (scale != nullptr && batch_split_factor != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "batchSplitFactor should be 1 when the scale is not null.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (batch_split_factor <= 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "batch_split_factor[%d] should be greater than 0.",
            batch_split_factor);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (batch_split_factor > 1 && x1->GetViewShape().GetDim((*perm_x1)[0]) % batch_split_factor != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "batch_split_factor[%d] should be factor of batch[%ld]",
            batch_split_factor, x1->GetViewShape().GetDim((*perm_x1)[0]));
        return ACLNN_ERR_PARAM_INVALID;
    }

    CHECK_RET(CheckDtypeValid(x1, x2, scale, out), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShapeValid(x1, x2, scale, perm_x1, perm_x2), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static const aclTensor* BuildTransposeBatchMatMulGraph(const aclTensor* x1, const aclTensor* x2,
                                                       const aclTensor* scale, const aclIntArray* perm_x1,
                                                       const aclIntArray* perm_x2, const aclIntArray* perm_y,
                                                       int8_t cubeMathType, int32_t batch_split_factor,
                                                       aclOpExecutor *executor)
{
    // 连续性转换
    auto contiguousX1 = l0op::Contiguous(x1, executor);
    OP_CHECK(contiguousX1 != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
             "The input x1 perprocess failed, contiguouse return nullptr."),
             return nullptr);
    auto reformX1 = l0op::ReFormat(contiguousX1, op::Format::FORMAT_ND);
    OP_CHECK(reformX1 != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
             "The input x1 perprocess failed, reformat return nullptr."),
             return nullptr);

    auto contiguousX2 = l0op::Contiguous(x2, executor);
    OP_CHECK(contiguousX2 != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
             "The input x2 perprocess failed, contiguouse return nullptr."),
             return nullptr);
    auto reformX2 = l0op::ReFormat(contiguousX2, op::Format::FORMAT_ND);
    OP_CHECK(reformX2 != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
             "The input x2 perprocess failed, reformat return nullptr."),
             return nullptr);

    // scale非连续转连续以及转换dtype
    auto contiguousScale = scale;
    if (contiguousScale != nullptr) {
        contiguousScale = l0op::Contiguous(scale, executor);
        OP_CHECK(contiguousScale != nullptr, OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                 "THe input scale perprocess failed, contiguouse return nullptr."),
                 return nullptr);
    }

    // 构建matmul计算图
    return l0op::TransposeBatchMatMul(reformX1, reformX2, nullptr, contiguousScale, perm_x1, perm_x2,
                                      perm_y, cubeMathType == USE_HF32, batch_split_factor, executor);
}

aclnnStatus aclnnTransposeBatchMatMulGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* bias,
                                                      const aclTensor* scale, const aclIntArray* permX1,
                                                      const aclIntArray* permX2, const aclIntArray* permY,
                                                      int8_t cubeMathType, const int32_t batchSplitFactor,
                                                      aclTensor* out, uint64_t* workspaceSize,
                                                      aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnTransposeBatchMatMul,
        DFX_IN(x1, x2, bias, scale, permX1, permX2, permY, cubeMathType, batchSplitFactor), DFX_OUT(out));
    
    // 固定写法, 创建OpExecutor
    auto unique_executor = CREATE_EXECUTOR();
    CHECK_RET(unique_executor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 入参检查
    auto ret = CheckParams(x1, x2, scale, out, permX1, permX2, cubeMathType, batchSplitFactor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    
    // 空tensor 处理
    if (x1->IsEmpty() || x2->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "aclnnTransposeBatchMatMul do not support empty tensor!");
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (bias != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The bias is not support in TBMM.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 构建matmul计算图
    const aclTensor* tbmmOut = nullptr;
    tbmmOut = BuildTransposeBatchMatMulGraph(x1, x2, scale, permX1, permX2, permY,
                                             cubeMathType, batchSplitFactor, unique_executor.get());
    CHECK_RET(tbmmOut != nullptr, ACLNN_ERR_PARAM_INVALID);

    if (tbmmOut->IsEmpty()) {
        *workspaceSize = 0;
        unique_executor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    tbmmOut = l0op::Cast(tbmmOut, out->GetDataType(), unique_executor.get());
    CHECK_RET(tbmmOut != nullptr, ACLNN_ERR_PARAM_INVALID);
    auto viewCopyResult = l0op::ViewCopy(tbmmOut, out, unique_executor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_PARAM_INVALID);

    *workspaceSize = unique_executor->GetWorkspaceSize();
    unique_executor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnTransposeBatchMatMul(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                      const aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnTransposeBatchMatMul);

    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
