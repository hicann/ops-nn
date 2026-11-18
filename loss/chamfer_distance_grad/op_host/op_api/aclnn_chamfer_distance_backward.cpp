/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_chamfer_distance_backward.h"
#include "chamfer_distance_backward.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn/aclnn_base.h"
#include "level0/fill.h"
#include "level0/squeeze.h"
#include "level0/unsqueeze.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static bool CheckNotNull(const aclTensor* xyz1, const aclTensor* xyz2, const aclTensor* idx1, const aclTensor* idx2,
                         const aclTensor* gradDist1, const aclTensor* gradDist2, aclTensor* gradXyz1,
                         aclTensor* gradXyz2)
{
    OP_CHECK_NULL(xyz1, return false);
    OP_CHECK_NULL(xyz2, return false);
    OP_CHECK_NULL(idx1, return false);
    OP_CHECK_NULL(idx2, return false);
    OP_CHECK_NULL(gradDist1, return false);
    OP_CHECK_NULL(gradDist2, return false);
    OP_CHECK_NULL(gradXyz1, return false);
    OP_CHECK_NULL(gradXyz2, return false);
    return true;
}

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT,
                                                                       op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> IDX_DTYPE_SUPPORT_LIST = {op::DataType::DT_INT32};

static bool CheckDtypeValid(const aclTensor* xyz1, const aclTensor* xyz2, const aclTensor* idx1, const aclTensor* idx2,
                            const aclTensor* gradDist1, const aclTensor* gradDist2, const aclTensor* gradXyz1,
                            const aclTensor* gradXyz2)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(xyz1, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(xyz2, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(idx1, IDX_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(idx2, IDX_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gradDist1, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gradDist2, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gradXyz1, DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(gradXyz2, DTYPE_SUPPORT_LIST, return false);

    // 计算在FP32下进行、输出统一Cast到xyz1的dtype，参与计算的tensor必须为相同数据类型
    OP_CHECK_DTYPE_NOT_SAME(xyz1, xyz2, return false);
    OP_CHECK_DTYPE_NOT_SAME(xyz1, gradDist1, return false);
    OP_CHECK_DTYPE_NOT_SAME(xyz1, gradDist2, return false);
    OP_CHECK_DTYPE_NOT_SAME(xyz1, gradXyz1, return false);
    OP_CHECK_DTYPE_NOT_SAME(xyz1, gradXyz2, return false);

    return true;
}

constexpr int64_t XYZ_RANK = 3;
constexpr int64_t POINTWISE_RANK = 2;
constexpr int64_t BATCH_AXIS = 0;
constexpr int64_t POINT_AXIS = 1;
constexpr int64_t COORD_AXIS = 2;
constexpr int64_t COORD_DIM = 2;

// idx/gradDist的shape必须与对应xyz的前两维一致
static bool IsPointwiseShapeMatch(const aclTensor* pointwise, const aclTensor* xyz)
{
    return pointwise->GetViewShape().GetDim(BATCH_AXIS) == xyz->GetViewShape().GetDim(BATCH_AXIS) &&
           pointwise->GetViewShape().GetDim(POINT_AXIS) == xyz->GetViewShape().GetDim(POINT_AXIS);
}

// 与图模式InferShape(chamfer_distance_grad_infershape.cpp)等价的shape校验
static bool CheckShape(const aclTensor* xyz1, const aclTensor* xyz2, const aclTensor* idx1, const aclTensor* idx2,
                       const aclTensor* gradDist1, const aclTensor* gradDist2, const aclTensor* gradXyz1,
                       const aclTensor* gradXyz2)
{
    // 空输入由主流程提前返回成功，无需shape校验
    if (xyz1->IsEmpty() || xyz2->IsEmpty() || idx1->IsEmpty() || idx2->IsEmpty() || gradDist1->IsEmpty() ||
        gradDist2->IsEmpty()) {
        return true;
    }
    OP_CHECK_WRONG_DIMENSION(xyz1, XYZ_RANK, return false);
    OP_CHECK_WRONG_DIMENSION(xyz2, XYZ_RANK, return false);
    OP_CHECK_WRONG_DIMENSION(idx1, POINTWISE_RANK, return false);
    OP_CHECK_WRONG_DIMENSION(idx2, POINTWISE_RANK, return false);
    OP_CHECK_WRONG_DIMENSION(gradDist1, POINTWISE_RANK, return false);
    OP_CHECK_WRONG_DIMENSION(gradDist2, POINTWISE_RANK, return false);

    // xyz1/xyz2的最后一维固定为2
    if (xyz1->GetViewShape().GetDim(COORD_AXIS) != COORD_DIM || xyz2->GetViewShape().GetDim(COORD_AXIS) != COORD_DIM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The last dimensions of xyz1 and xyz2 must both be 2, but got %ld and %ld.",
                xyz1->GetViewShape().GetDim(COORD_AXIS), xyz2->GetViewShape().GetDim(COORD_AXIS));
        return false;
    }
    // kernel仅有一份batch_size/num，xyz1与xyz2的batch维、点数维必须一致
    if (xyz1->GetViewShape().GetDim(BATCH_AXIS) != xyz2->GetViewShape().GetDim(BATCH_AXIS)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "xyz1 and xyz2 must have the same batch dimension, but got %ld and %ld.",
                xyz1->GetViewShape().GetDim(BATCH_AXIS), xyz2->GetViewShape().GetDim(BATCH_AXIS));
        return false;
    }
    if (xyz1->GetViewShape().GetDim(POINT_AXIS) != xyz2->GetViewShape().GetDim(POINT_AXIS)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "xyz1 and xyz2 must have the same point dimension, but got %ld and %ld.",
                xyz1->GetViewShape().GetDim(POINT_AXIS), xyz2->GetViewShape().GetDim(POINT_AXIS));
        return false;
    }
    if (!IsPointwiseShapeMatch(idx1, xyz1)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "idx1 shape must match the first two dimensions of xyz1.");
        return false;
    }
    if (!IsPointwiseShapeMatch(idx2, xyz2)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "idx2 shape must match the first two dimensions of xyz2.");
        return false;
    }
    if (!IsPointwiseShapeMatch(gradDist1, xyz1)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradDist1 shape must match the first two dimensions of xyz1.");
        return false;
    }
    if (!IsPointwiseShapeMatch(gradDist2, xyz2)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gradDist2 shape must match the first two dimensions of xyz2.");
        return false;
    }
    // 输出shape必须与对应输入一致
    OP_CHECK_SHAPE_NOT_EQUAL(gradXyz1, xyz1, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(gradXyz2, xyz2, return false);
    return true;
}

static aclnnStatus CheckParams(const aclTensor* xyz1, const aclTensor* xyz2, const aclTensor* idx1,
                               const aclTensor* idx2, const aclTensor* gradDist1, const aclTensor* gradDist2,
                               aclTensor* gradXyz1, aclTensor* gradXyz2)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(xyz1, xyz2, idx1, idx2, gradDist1, gradDist2, gradXyz1, gradXyz2), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查数据类型是否正确
    CHECK_RET(CheckDtypeValid(xyz1, xyz2, idx1, idx2, gradDist1, gradDist2, gradXyz1, gradXyz2),
              ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入输出的shape是否满足要求
    CHECK_RET(CheckShape(xyz1, xyz2, idx1, idx2, gradDist1, gradDist2, gradXyz1, gradXyz2), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

// 空输入早退时非空输出需写入全0：空idx/gradDist表示无配对点、梯度贡献为0（issue#8186）
static aclnnStatus FillOutWithZeros(const aclTensor* out, aclOpExecutor* executor)
{
    auto outShape = out->GetViewShape();
    op::ShapeVector fillDims = op::ToShapeVector(outShape);
    auto fillDimsArray = executor->AllocIntArray(fillDims.data(), outShape.GetDimNum());
    CHECK_RET(fillDimsArray != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* fillDimTensor = executor->ConvertToTensor(fillDimsArray, op::DataType::DT_INT64);
    CHECK_RET(fillDimTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclScalar* zeroScalar = executor->AllocScalar(0);
    CHECK_RET(zeroScalar != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* zeroTensor = executor->ConvertToTensor(zeroScalar, out->GetDataType());
    CHECK_RET(zeroTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* zeroFillTensor = l0op::Fill(fillDimTensor, zeroTensor, fillDimsArray, executor);
    CHECK_RET(zeroFillTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(zeroFillTensor, out, executor) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChamferDistanceBackwardGetWorkspaceSize(const aclTensor* xyz1, const aclTensor* xyz2,
                                                         const aclTensor* idx1, const aclTensor* idx2,
                                                         const aclTensor* gradDist1, const aclTensor* gradDist2,
                                                         aclTensor* gradXyz1, aclTensor* gradXyz2,
                                                         uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);

    L2_DFX_PHASE_1(aclnnChamferDistanceBackward, DFX_IN(xyz1, xyz2, idx1, idx2, gradDist1, gradDist2),
                   DFX_OUT(gradXyz1, gradXyz2));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(xyz1, xyz2, idx1, idx2, gradDist1, gradDist2, gradXyz1, gradXyz2);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (xyz1->IsEmpty() || xyz2->IsEmpty() || idx1->IsEmpty() || idx2->IsEmpty() || gradDist1->IsEmpty() ||
        gradDist2->IsEmpty()) {
        // 空输入早退时非空输出需补写全0，避免调用方拿到SUCCESS后读到未初始化显存（issue#8186）
        if (!gradXyz1->IsEmpty()) {
            auto fillRet = FillOutWithZeros(gradXyz1, uniqueExecutor.get());
            CHECK_RET(fillRet == ACLNN_SUCCESS, fillRet);
        }
        if (!gradXyz2->IsEmpty()) {
            auto fillRet = FillOutWithZeros(gradXyz2, uniqueExecutor.get());
            CHECK_RET(fillRet == ACLNN_SUCCESS, fillRet);
        }
        *workspaceSize = uniqueExecutor->GetWorkspaceSize();
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto xyz1Contiguous = l0op::Contiguous(xyz1, uniqueExecutor.get());
    CHECK_RET(xyz1Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto xyz2Contiguous = l0op::Contiguous(xyz2, uniqueExecutor.get());
    CHECK_RET(xyz2Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto idx1Contiguous = l0op::Contiguous(idx1, uniqueExecutor.get());
    CHECK_RET(idx1Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto idx2Contiguous = l0op::Contiguous(idx2, uniqueExecutor.get());
    CHECK_RET(idx2Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradDist1Contiguous = l0op::Contiguous(gradDist1, uniqueExecutor.get());
    CHECK_RET(gradDist1Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto gradDist2Contiguous = l0op::Contiguous(gradDist2, uniqueExecutor.get());
    CHECK_RET(gradDist2Contiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto curType = xyz1Contiguous->GetDataType();
    auto xyz1Cast = l0op::Cast(xyz1Contiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
    CHECK_RET(xyz1Cast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto xyz2Cast = l0op::Cast(xyz2Contiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
    CHECK_RET(xyz2Cast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradDist1Cast = l0op::Cast(gradDist1Contiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
    CHECK_RET(gradDist1Cast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradDist2Cast = l0op::Cast(gradDist2Contiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
    CHECK_RET(gradDist2Cast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto out = l0op::ChamferDistanceGrad(xyz1Cast, xyz2Cast, idx1Contiguous, idx2Contiguous, gradDist1Cast,
                                         gradDist2Cast, uniqueExecutor.get());
    auto gradXyz1Contiguous = std::get<0>(out);
    auto gradXyz2Contiguous = std::get<1>(out);
    auto gradXyz1Cast = l0op::Cast(gradXyz1Contiguous, curType, uniqueExecutor.get());
    CHECK_RET(gradXyz1Cast != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto gradXyz2Cast = l0op::Cast(gradXyz2Contiguous, curType, uniqueExecutor.get());
    CHECK_RET(gradXyz2Cast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(gradXyz1Cast, gradXyz1, uniqueExecutor.get());
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyResult1 = l0op::ViewCopy(gradXyz2Cast, gradXyz2, uniqueExecutor.get());
    CHECK_RET(viewCopyResult1 != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChamferDistanceBackward(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                         aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChamferDistanceBackward);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
