/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_quant_batch_matmul_inplace_add.h"
#include "quant_batch_matmul_inplace_add_util.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include <dlfcn.h>
#include "securec.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "log/log.h"
#include "matmul/common/op_host/log_format_util.h"
#include "quant_batch_matmul_inplace_add.h"
#include "../../quant_batch_matmul_v4/op_host/op_api/quant_matmul_common_check.h"
#include "util/math_util.h"

using namespace op;
using namespace QBMMInplaceAdd;
using Ops::NN::IsTransposeLastTwoDims;
using Ops::NN::BoolToString;
using Ops::NN::FormatString;
using Ops::NN::StripEnclosingSquareBrackets;
using Ops::NN::SwapLastTwoDimValue;

namespace {
static aclnnStatus CheckNotNull(const QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params)
{
    OP_CHECK_NULL(params.x1, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(params.x2, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(params.x2Scale, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(params.yRef, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(params.x1ScaleOptional, return ACLNN_ERR_PARAM_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(const QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params)
{
    if (params.x1->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x1", op::ToString(params.x1->GetStorageFormat()).GetString(),
            "the format of x1 must be ND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x2->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x2", op::ToString(params.x2->GetStorageFormat()).GetString(),
            "the format of x2 must be ND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x2Scale->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x2Scale", op::ToString(params.x2Scale->GetStorageFormat()).GetString(),
            "the format of x2Scale must be ND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.yRef->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "yRef", op::ToString(params.yRef->GetStorageFormat()).GetString(),
            "the format of yRef must be ND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x1ScaleOptional->GetStorageFormat() != Format::FORMAT_ND) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x1Scale", op::ToString(params.x1ScaleOptional->GetStorageFormat()).GetString(),
            "the format of x1Scale must be ND");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus IsMxQuantDim(const QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params)
{
    auto x1ScaleDimNum = params.x1ScaleOptional->GetViewShape().GetDimNum();
    auto x2ScaleDimNum = params.x2Scale->GetViewShape().GetDimNum();
    if (x2ScaleDimNum != MX_X2_SCALE_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x2Scale",
            FormatString("%zuD", x2ScaleDimNum).c_str(),
            FormatString("when the quantization mode is mx, the shape dim of x2Scale must be %zu", MX_X2_SCALE_DIM).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x1ScaleDimNum != MX_X1_SCALE_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x1Scale",
            FormatString("%zuD", x1ScaleDimNum).c_str(),
            FormatString("when the quantization mode is mx, the shape dim of x1Scale must be %zu", MX_X1_SCALE_DIM).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    
    return ACLNN_SUCCESS;
}

static aclnnStatus IsPerTensorDim(const QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params)
{
    auto x1ScaleDimNum = params.x1ScaleOptional->GetViewShape().GetDimNum();
    auto x2ScaleDimNum = params.x2Scale->GetViewShape().GetDimNum();
    if (x1ScaleDimNum != PERTENSOR_SCALE_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x1Scale",
            FormatString("%zuD", x1ScaleDimNum).c_str(),
            FormatString("when the quantization mode is HiFloat8 per-tensor, the shape dim of x1Scale must be %zu",
                PERTENSOR_SCALE_DIM).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x2ScaleDimNum != PERTENSOR_SCALE_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x2Scale",
            FormatString("%zuD", x2ScaleDimNum).c_str(),
            FormatString("when the quantization mode is HiFloat8 per-tensor, the shape dim of x2Scale must be %zu",
                PERTENSOR_SCALE_DIM).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x1ScaleOptional->GetViewShape().GetDim(0) != 1) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x1Scale",
            StripEnclosingSquareBrackets(op::ToString(params.x1ScaleOptional->GetViewShape()).GetString()).c_str(),
            "when the quantization mode is HiFloat8 per-tensor, the shape of x1Scale must be [1]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x2Scale->GetViewShape().GetDim(0) != 1) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x2Scale",
            StripEnclosingSquareBrackets(op::ToString(params.x2Scale->GetViewShape()).GetString()).c_str(),
            "when the quantization mode is HiFloat8 per-tensor, the shape of x2Scale must be [1]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckInputOutDims(const QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params)
{
    auto x1DimNum = params.x1->GetViewShape().GetDimNum();
    auto x2DimNum = params.x2->GetViewShape().GetDimNum();
    auto yInputDimNum = params.yRef->GetViewShape().GetDimNum();
    if (x1DimNum != MX_X1_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x1", FormatString("%zuD", x1DimNum).c_str(),
            FormatString("the shape dim of x1 must be %zu", MX_X1_DIM).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (x2DimNum != MX_X2_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x2", FormatString("%zuD", x2DimNum).c_str(),
            FormatString("the shape dim of x2 must be %zu", MX_X2_DIM).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (yInputDimNum != Y_INPUT_DIM) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "yRef",
            FormatString("%zuD", yInputDimNum).c_str(),
            FormatString("the shape dim of yRef must be %zu", Y_INPUT_DIM).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static inline bool IsMicroScaling(const aclTensor* x1Scale, const aclTensor* x2Scale)
{
    if (x1Scale == nullptr || x2Scale == nullptr) {
        return false;
    }
    return x1Scale->GetDataType() == op::DataType::DT_FLOAT8_E8M0 &&
           x2Scale->GetDataType() == op::DataType::DT_FLOAT8_E8M0;
}

static inline bool IsHiFloat8Input(const aclTensor* x1, const aclTensor* x2)
{
    return x1->GetDataType() == op::DataType::DT_HIFLOAT8 && x2->GetDataType() == op::DataType::DT_HIFLOAT8;
}

static bool CheckMKN(int64_t m, int64_t k, int64_t n)
{
    if (m <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x1 M", std::to_string(m).c_str(),
            "the M dimension of x1 must be positive");
        return false;
    }
    if (k <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "K", std::to_string(k).c_str(),
            "the K dimension of x1 and x2 must be positive");
        return false;
    }
    if (n <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x2 N", std::to_string(n).c_str(),
            "the N dimension of x2 must be positive");
        return false;
    }
    return true;
}

static bool CheckGroupSize(const QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params)
{
    auto groupSize = params.groupSize;
    uint64_t groupSizeM = (static_cast<uint64_t>(groupSize) >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeN = (static_cast<uint64_t>(groupSize) >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeK = static_cast<uint64_t>(groupSize) & GROUP_MNK_BIT_SIZE;
    if (IsMicroScaling(params.x1ScaleOptional, params.x2Scale)) {
        if (groupSizeK != static_cast<uint64_t>(PERGROUP_GROUP_SIZE) || groupSizeM != 1UL || groupSizeN != 1UL) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "groupSize, groupSizeM, groupSizeN, groupSizeK",
                FormatString("%ld, %lu, %lu, %lu", groupSize, groupSizeM, groupSizeN, groupSizeK).c_str(),
                "when the quantization mode is mx, groupSize must be 4295032864 and Torch API group_sizes must be [1, 1, 32]");
            return false;
        }
    } else if (groupSize != 0L) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "groupSize, groupSizeM, groupSizeN, groupSizeK",
            FormatString("%ld, %lu, %lu, %lu", groupSize, groupSizeM, groupSizeN, groupSizeK).c_str(),
            "when the quantization mode is not mx, groupSize must be 0 and Torch API group_sizes must be [0, 0, 0] or None");
        return false;
    }
    OP_LOGD("QuantBatchMatmulInplaceAdd check group_size success.");
    return true;
}

struct MatmulShapeInfo {
    int64_t mDim;
    int64_t kDim;
    int64_t nDim;
    int64_t yRefMDim;
    int64_t yRefNDim;
};

static MatmulShapeInfo GetMatmulShapeInfo(const QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params)
{
    int64_t x1DimNum = params.x1->GetViewShape().GetDimNum();
    int64_t x2DimNum = params.x2->GetViewShape().GetDimNum();
    return {
        params.transposeX1 ? params.x1->GetViewShape().GetDim(x1DimNum - 1) :
                             params.x1->GetViewShape().GetDim(x1DimNum - PENULTIMATE_DIM),
        params.transposeX1 ? params.x1->GetViewShape().GetDim(x1DimNum - PENULTIMATE_DIM) :
                             params.x1->GetViewShape().GetDim(x1DimNum - 1),
        params.transposeX2 ? params.x2->GetViewShape().GetDim(x2DimNum - PENULTIMATE_DIM) :
                             params.x2->GetViewShape().GetDim(x2DimNum - 1),
        params.yRef->GetViewShape().GetDim(0),
        params.yRef->GetViewShape().GetDim(1)};
}

static aclnnStatus CheckShapeInfoMatch(
    const QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params, const MatmulShapeInfo& shapeInfo)
{
    int64_t x2DimNum = params.x2->GetViewShape().GetDimNum();
    int64_t x2KDim = params.transposeX2 ? params.x2->GetViewShape().GetDim(x2DimNum - 1) :
                                          params.x2->GetViewShape().GetDim(x2DimNum - PENULTIMATE_DIM);
    if (shapeInfo.kDim != x2KDim) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x1 K, x2 K",
            FormatString("%ld, %ld", shapeInfo.kDim, x2KDim).c_str(),
            "the K dimension of x1 and x2 must be equal");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (shapeInfo.mDim != shapeInfo.yRefMDim) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x1 M, yRef M",
            FormatString("%ld, %ld", shapeInfo.mDim, shapeInfo.yRefMDim).c_str(),
            "the M dimension of x1 and yRef must be equal");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (shapeInfo.nDim != shapeInfo.yRefNDim) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x2 N, yRef N",
            FormatString("%ld, %ld", shapeInfo.nDim, shapeInfo.yRefNDim).c_str(),
            "the N dimension of x2 and yRef must be equal");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckMxScaleLastDim(const QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params)
{
    if (!IsMicroScaling(params.x1ScaleOptional, params.x2Scale)) {
        return ACLNN_SUCCESS;
    }

    auto scale1LastDimValue = params.x1ScaleOptional->GetViewShape().GetDim(MX_X1_SCALE_DIM - 1);
    auto scale2LastDimValue = params.x2Scale->GetViewShape().GetDim(MX_X2_SCALE_DIM - 1);
    if (scale1LastDimValue != MXFP_MULTI_BASE_SIZE) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x1Scale",
            StripEnclosingSquareBrackets(op::ToString(params.x1ScaleOptional->GetViewShape()).GetString()).c_str(),
            FormatString("when the quantization mode is mx, the last dimension of x1Scale must be %d",
                MXFP_MULTI_BASE_SIZE).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (scale2LastDimValue != MXFP_MULTI_BASE_SIZE) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x2Scale",
            StripEnclosingSquareBrackets(op::ToString(params.x2Scale->GetViewShape()).GetString()).c_str(),
            FormatString("when the quantization mode is mx, the last dimension of x2Scale must be %d",
                MXFP_MULTI_BASE_SIZE).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static void GetExpectedScaleShape(
    const QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params, const MatmulShapeInfo& shapeInfo,
    op::Shape& x1ScaleExpectShape, op::Shape& x2ScaleExpectShape)
{
    if (!IsMicroScaling(params.x1ScaleOptional, params.x2Scale)) {
        x1ScaleExpectShape = {1};
        x2ScaleExpectShape = {1};
        return;
    }

    x1ScaleExpectShape = params.transposeX1 ?
        op::Shape{Ops::Base::CeilDiv(shapeInfo.kDim, SPLIT_SIZE), shapeInfo.mDim, MXFP_MULTI_BASE_SIZE} :
        op::Shape{shapeInfo.mDim, Ops::Base::CeilDiv(shapeInfo.kDim, SPLIT_SIZE), MXFP_MULTI_BASE_SIZE};
    x2ScaleExpectShape = params.transposeX2 ?
        op::Shape{shapeInfo.nDim, Ops::Base::CeilDiv(shapeInfo.kDim, SPLIT_SIZE), MXFP_MULTI_BASE_SIZE} :
        op::Shape{Ops::Base::CeilDiv(shapeInfo.kDim, SPLIT_SIZE), shapeInfo.nDim, MXFP_MULTI_BASE_SIZE};
}

static aclnnStatus CheckExpectedShapes(
    const QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params, const MatmulShapeInfo& shapeInfo)
{
    op::Shape x1ExpectShape = params.transposeX1 ? op::Shape{shapeInfo.kDim, shapeInfo.mDim} :
                                                  op::Shape{shapeInfo.mDim, shapeInfo.kDim};
    op::Shape x2ExpectShape = params.transposeX2 ? op::Shape{shapeInfo.nDim, shapeInfo.kDim} :
                                                  op::Shape{shapeInfo.kDim, shapeInfo.nDim};
    op::Shape x1ScaleExpectShape;
    op::Shape x2ScaleExpectShape;
    GetExpectedScaleShape(params, shapeInfo, x1ScaleExpectShape, x2ScaleExpectShape);
    op::Shape yExpectShape = {shapeInfo.mDim, shapeInfo.nDim};

    if (params.x1->GetViewShape() != x1ExpectShape) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x1",
            StripEnclosingSquareBrackets(op::ToString(params.x1->GetViewShape()).GetString()).c_str(),
            FormatString("the shape of x1 must be %s", op::ToString(x1ExpectShape).GetString()).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x1ScaleOptional->GetViewShape() != x1ScaleExpectShape) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x1Scale",
            StripEnclosingSquareBrackets(op::ToString(params.x1ScaleOptional->GetViewShape()).GetString()).c_str(),
            FormatString("the shape of x1Scale must be %s", op::ToString(x1ScaleExpectShape).GetString()).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x2->GetViewShape() != x2ExpectShape) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x2",
            StripEnclosingSquareBrackets(op::ToString(params.x2->GetViewShape()).GetString()).c_str(),
            FormatString("the shape of x2 must be %s", op::ToString(x2ExpectShape).GetString()).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x2Scale->GetViewShape() != x2ScaleExpectShape) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x2Scale",
            StripEnclosingSquareBrackets(op::ToString(params.x2Scale->GetViewShape()).GetString()).c_str(),
            FormatString("the shape of x2Scale must be %s", op::ToString(x2ScaleExpectShape).GetString()).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.yRef->GetViewShape() != yExpectShape) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "yRef",
            StripEnclosingSquareBrackets(op::ToString(params.yRef->GetViewShape()).GetString()).c_str(),
            FormatString("the shape of yRef must be %s", op::ToString(yExpectShape).GetString()).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(const QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params)
{
    CHECK_COND(CheckInputOutDims(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Check CheckInputOutDims failed.");
    if (params.transposeX1 != true || params.transposeX2 != false) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "transposeX1, transposeX2",
            FormatString("%s, %s", BoolToString(params.transposeX1), BoolToString(params.transposeX2)).c_str(),
            "transposeX1 must be true and transposeX2 must be false");
        return ACLNN_ERR_PARAM_INVALID;
    }
    MatmulShapeInfo shapeInfo = GetMatmulShapeInfo(params);
    CHECK_COND(CheckShapeInfoMatch(params, shapeInfo) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
        "CheckShapeInfoMatch failed.");

    if (!CheckMKN(shapeInfo.mDim, shapeInfo.kDim, shapeInfo.nDim)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "CheckMKN failed.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_COND(CheckMxScaleLastDim(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "CheckMxScaleLastDim failed.");
    return CheckExpectedShapes(params, shapeInfo);
}

static inline bool MxScaleContiguousProcess(const aclTensor*& mxScaleTensor, aclOpExecutor* executor)
{
    if (mxScaleTensor == nullptr || mxScaleTensor->GetViewShape().GetDimNum() < MX_SCALE_MAX_DIM) {
        OP_LOGD("MX scale no need to do contiguous process.");
        return true;
    }
    auto transposeFlag = false;
    int64_t dimNum = mxScaleTensor->GetViewShape().GetDimNum();
    int64_t lastDim = mxScaleTensor->GetViewShape().GetDim(dimNum - 1);
    int64_t lastSecondDim = mxScaleTensor->GetViewShape().GetDim(dimNum - PENULTIMATE_DIM);
    int64_t lastThirdDim = mxScaleTensor->GetViewShape().GetDim(dimNum - 3);    // 3: 倒数第3维
    if (mxScaleTensor->GetViewStrides()[dimNum - 3] == lastDim &&   // 3: 倒数第3维
        mxScaleTensor->GetViewStrides()[dimNum - PENULTIMATE_DIM] == lastDim * lastThirdDim) {
        int64_t tmpNxD = lastDim * lastSecondDim * lastThirdDim;
        transposeFlag = true;
        // 4: batch维度从倒数第4维起
        for (int64_t batchDim = dimNum - 4; batchDim >= 0; batchDim--) {
            if (mxScaleTensor->GetViewStrides()[batchDim] != tmpNxD) {
                transposeFlag = false;
                break;
            }
            tmpNxD *= mxScaleTensor->GetViewShape().GetDim(batchDim);
        }
        if (lastSecondDim == 1 && lastThirdDim == 1) {
            transposeFlag = false;
        }
    }
    if (transposeFlag) {
        op::Shape swapedShape = mxScaleTensor->GetViewShape();
        swapedShape.SetDim(dimNum - PENULTIMATE_DIM, lastThirdDim);
        swapedShape.SetDim(dimNum - 3, lastSecondDim);  // 3: 倒数第3维
        mxScaleTensor = executor->CreateView(mxScaleTensor, swapedShape, mxScaleTensor->GetViewOffset());
    } else {
        mxScaleTensor = l0op::Contiguous(mxScaleTensor, executor);
    }
    CHECK_RET(mxScaleTensor != nullptr, false);
    return true;
}

static aclnnStatus CheckInputDtypeValid(const QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params)
{
    if (!CheckType(params.x1->GetDataType(), X1_DTYPE_SUPPORT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x1", op::ToString(params.x1->GetDataType()).GetString(),
            FormatString("the dtype of x1 must be in dtype support list %s",
                op::ToString(X1_DTYPE_SUPPORT_LIST).GetString()).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckType(params.x2->GetDataType(), X2_DTYPE_SUPPORT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x2", op::ToString(params.x2->GetDataType()).GetString(),
            FormatString("the dtype of x2 must be in dtype support list %s",
                op::ToString(X2_DTYPE_SUPPORT_LIST).GetString()).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckMxfp8DtypeValid(const QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params)
{
    if (CheckInputDtypeValid(params) != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x1ScaleOptional->GetDataType() != op::DataType::DT_FLOAT8_E8M0) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x1Scale",
            op::ToString(params.x1ScaleOptional->GetDataType()).GetString(),
            "when the quantization mode is mx, the dtype of x1Scale must be FLOAT8_E8M0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x2Scale->GetDataType() != op::DataType::DT_FLOAT8_E8M0) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x2Scale",
            op::ToString(params.x2Scale->GetDataType()).GetString(),
            "when the quantization mode is mx, the dtype of x2Scale must be FLOAT8_E8M0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckType(params.yRef->GetDataType(), YREF_DTYPE_SUPPORT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "yRef", op::ToString(params.yRef->GetDataType()).GetString(),
            FormatString("the dtype of yRef must be in dtype support list %s",
                op::ToString(YREF_DTYPE_SUPPORT_LIST).GetString()).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    OP_LOGD("QuantBatchMatmulInplaceAdd CheckMxfp8DtypeValid success.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckHif8DtypeValid(const QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params)
{
    if (CheckInputDtypeValid(params) != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x1ScaleOptional->GetDataType() != op::DataType::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x1Scale",
            op::ToString(params.x1ScaleOptional->GetDataType()).GetString(),
            "when the quantization mode is HiFloat8 per-tensor, the dtype of x1Scale must be FLOAT32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (params.x2Scale->GetDataType() != op::DataType::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x2Scale",
            op::ToString(params.x2Scale->GetDataType()).GetString(),
            "when the quantization mode is HiFloat8 per-tensor, the dtype of x2Scale must be FLOAT32");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!CheckType(params.yRef->GetDataType(), YREF_DTYPE_SUPPORT_LIST)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "yRef", op::ToString(params.yRef->GetDataType()).GetString(),
            FormatString("the dtype of yRef must be in dtype support list %s",
                op::ToString(YREF_DTYPE_SUPPORT_LIST).GetString()).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (IsPerTensorDim(params) != ACLNN_SUCCESS) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x1Scale, x2Scale",
            FormatString("%s, %s", op::ToString(params.x1ScaleOptional->GetViewShape()).GetString(),
                op::ToString(params.x2Scale->GetViewShape()).GetString()).c_str(),
            "when the quantization mode is HiFloat8 per-tensor, the shapes of x1Scale and x2Scale must be [1]");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(const QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params)
{
    auto x1Dtype = params.x1->GetDataType();
    auto x2Dtype = params.x2->GetDataType();
    auto x1ScaleDtype = params.x1ScaleOptional->GetDataType();
    auto x2ScaleDtype = params.x2Scale->GetDataType();
    if ((x1Dtype == DataType::DT_FLOAT8_E4M3FN || x1Dtype == DataType::DT_FLOAT8_E5M2) &&
        (x2Dtype == DataType::DT_FLOAT8_E4M3FN || x2Dtype == DataType::DT_FLOAT8_E5M2)) {
        CHECK_COND(IsMxQuantDim(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Check IsMxQuantDim failed.");
        return CheckMxfp8DtypeValid(params);
    } else if (IsHiFloat8Input(params.x1, params.x2)) {
        return CheckHif8DtypeValid(params);
    } else {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize", "x1, x2, x1Scale, x2Scale",
            FormatString("%s, %s, %s, %s", op::ToString(x1Dtype).GetString(), op::ToString(x2Dtype).GetString(),
                op::ToString(x1ScaleDtype).GetString(), op::ToString(x2ScaleDtype).GetString()).c_str(),
            FormatString("when the dtypes of x1 and x2 are %s and %s, and the dtypes of x1Scale and x2Scale are %s "
                         "and %s, this dtype combination can not be supported",
                op::ToString(x1Dtype).GetString(), op::ToString(x2Dtype).GetString(),
                op::ToString(x1ScaleDtype).GetString(), op::ToString(x2ScaleDtype).GetString()).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
}

static aclnnStatus CheckParams(const QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params)
{
    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    OP_LOGD("QuantBatchMatmulInplaceAdd check params success.");

    return ACLNN_SUCCESS;
}

bool ReCalcGroupSize(uint64_t inputSize, uint64_t scaleSize, uint64_t& groupSize, const char* dimensionName)
{
    if (scaleSize == 0UL) {
        std::string scaleName = strcmp(dimensionName, "n") == 0 ? "x2Scale(scale)" : "x1Scale(pertokenScale)";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize",
            FormatString("%s dimension of %s", dimensionName, scaleName.c_str()).c_str(),
            std::to_string(scaleSize).c_str(),
            FormatString("the %s dimension of %s must be positive", dimensionName, scaleName.c_str()).c_str());
        return false;
    }
    if (groupSize == 0L) {
        if (inputSize % scaleSize != 0UL) {
            std::string scaleName = "x1Scale(pertokenScale)";
            std::string inputName = "x1";
            if (strcmp(dimensionName, "n") == 0) {
                scaleName = "x2Scale(scale)";
                inputName = "x2";
            }
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                "aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize",
                FormatString("%s dimension of %s, %s dimension of %s",
                    dimensionName, inputName.c_str(), dimensionName, scaleName.c_str()).c_str(),
                FormatString("%lu, %lu", inputSize, scaleSize).c_str(),
                FormatString(
                    "when groupSize in %s dimension is 0, the %s dimension of %s must be divisible by the %s dimension of %s to infer the real groupSize",
                    dimensionName, dimensionName, inputName.c_str(), dimensionName, scaleName.c_str()).c_str());
            return false;
        }
        groupSize = inputSize / scaleSize;
    }
    return true;
}

static inline bool InferGroupSize(QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params)
{
    auto x1 = params.x1;
    auto x2 = params.x2;
    auto x1Scale = params.x1ScaleOptional;
    auto x2Scale = params.x2Scale;
    if (x1Scale == nullptr || x1Scale->GetViewShape().GetDimNum() < 2 || x2Scale->GetViewShape().GetDimNum() < 2) {
        return true;
    }
    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x2DimNum = x2->GetViewShape().GetDimNum();
    auto x1ScaleDimNum = x1Scale->GetViewShape().GetDimNum();
    auto x2ScaleDimNum = x2Scale->GetViewShape().GetDimNum();
    auto transX1 = params.transposeX1;
    auto transX2 = params.transposeX2;
    uint64_t groupSizeK = static_cast<uint64_t>(params.groupSize) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeN = (static_cast<uint64_t>(params.groupSize) >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeM = (static_cast<uint64_t>(params.groupSize) >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;
    auto inputSizeM =
        transX1 ? x1->GetViewShape().GetDim(x1DimNum - 1) : x1->GetViewShape().GetDim(x1DimNum - PENULTIMATE_DIM);
    int64_t scaleSizeM = 0;
    if (IsMicroScaling(x1Scale, x2Scale)) {
        scaleSizeM = x1Scale->GetViewShape().GetDim(transX1 ? 1 : 0);
    } else {
        scaleSizeM = transX1 ? x1Scale->GetViewShape().GetDim(x1ScaleDimNum - 1) :
                               x1Scale->GetViewShape().GetDim(x1ScaleDimNum - PENULTIMATE_DIM);
    }
    CHECK_RET(ReCalcGroupSize(inputSizeM, scaleSizeM, groupSizeM, "m"), false);
    auto inputSizeK =
        transX1 ? x1->GetViewShape().GetDim(x1DimNum - PENULTIMATE_DIM) : x1->GetViewShape().GetDim(x1DimNum - 1);
    int64_t scaleSizeK = 0;
    if (IsMicroScaling(x1Scale, x2Scale)) {
        scaleSizeK = x1Scale->GetViewShape().GetDim(transX1 ? 0 : 1) * 2; //when scale type is e8m0, scalex1 shape is [m, k/2, 2] or [k/2, m, 2]
    } else {
        scaleSizeK = transX1 ? x1Scale->GetViewShape().GetDim(x1ScaleDimNum - PENULTIMATE_DIM) :
                               x1Scale->GetViewShape().GetDim(x1ScaleDimNum - 1);
    }
    CHECK_RET(ReCalcGroupSize(inputSizeK, scaleSizeK, groupSizeK, "k"), false);
    auto inputSizeN =
        transX2 ? x2->GetViewShape().GetDim(x2DimNum - PENULTIMATE_DIM) : x2->GetViewShape().GetDim(x2DimNum - 1);
    int64_t scaleSizeN = 0;
    if (IsMicroScaling(x1Scale, x2Scale)) {
        scaleSizeN = x2Scale->GetViewShape().GetDim(transX2 ? 0 : 1);
    } else {
        scaleSizeN = transX2 ? x2Scale->GetViewShape().GetDim(x2ScaleDimNum - PENULTIMATE_DIM) :
                               x2Scale->GetViewShape().GetDim(x2ScaleDimNum - 1);
    }
    CHECK_RET(ReCalcGroupSize(inputSizeN, scaleSizeN, groupSizeN, "n"), false);
    OP_LOGD(
        "Infered groupSize: groupSizeM: %lu, groupSizeN: %lu, groupSizeK: %lu.", groupSizeM, groupSizeN, groupSizeK);
    params.groupSize =
        static_cast<int64_t>((groupSizeM << GROUP_M_OFFSET) | (groupSizeN << GROUP_N_OFFSET) | groupSizeK);
    return true;
}

static bool QBMMIAGetTransposeAttrValue(const aclTensor* tensor, bool transpose, bool isSpecialCase = true)
{
    int64_t dim1 = tensor->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = tensor->GetViewShape().GetDimNum() - PENULTIMATE_DIM;
    // 对于torch的场景，NZ情况下二维某一维度为1的场景无法正确判断是否转置，资料呈现不支持非连续，代码默认连续
    if (static_cast<ge::Format>(ge::GetPrimaryFormat(tensor->GetStorageFormat())) == op::Format::FORMAT_FRACTAL_NZ &&
        (tensor->GetViewShape().GetDim(dim2) == 1 || tensor->GetViewShape().GetDim(dim1) == 1)) {
        return transpose;
    }
    // check if tensor is contiguous layout
    if (tensor->GetViewStrides()[dim2] == 1 &&
        (tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2))) {
        OP_LOGD("QuantBatchMatmulInplaceAdd GetTransposeAttrValue, find tensor is not contiguous.");
        const_cast<aclTensor*>(tensor)->SetViewShape(SwapLastTwoDimValue(tensor->GetViewShape()));
        // 如果不需要校验特殊case，则直接返回
        if (!isSpecialCase) {
            return !transpose;
        }
        if (!CheckSpecialCase(tensor, dim1, dim2)) {
            return !transpose;
        }
    }
    return transpose;
}

static aclnnStatus aclnnQuantBatchMatmulInplaceAddGetWorkspaceSizeCommon(
    QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams& params, aclOpExecutor* executor)
{
    // torch非连续转连续处理
    params.x2Scale = SetTensorToNDFormat(params.x2Scale);
    params.x1ScaleOptional = SetTensorToNDFormat(params.x1ScaleOptional);
    TensorContiguousProcess(params.x1, params.transposeX1, executor);
    TensorContiguousProcess(params.x2, params.transposeX2, executor);
    MxScaleContiguousProcess(params.x1ScaleOptional, executor);
    MxScaleContiguousProcess(params.x2Scale, executor);
    CHECK_RET(InferGroupSize(params), ACLNN_ERR_PARAM_INVALID);
    OP_LOGD("Infer groupSize success. groupSize: %ld.", params.groupSize);
    CHECK_COND(CheckGroupSize(params), ACLNN_ERR_PARAM_INVALID, "CheckGroupSize failed.");
    // 固定写法，参数检查
    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    bool transposeX1 = QBMMIAGetTransposeAttrValue(params.x1, params.transposeX1, true);
    bool transposeX2 = QBMMIAGetTransposeAttrValue(params.x2, params.transposeX2, true);
    // 空tensor校验
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        auto x1DimNum = params.x1->GetViewShape().GetDimNum();
        auto inputSizeM = transposeX1 ? params.x1->GetViewShape().GetDim(x1DimNum - 1) :
                                        params.x1->GetViewShape().GetDim(x1DimNum - PENULTIMATE_DIM);
        auto x2DimNum = params.x2->GetViewShape().GetDimNum();
        auto inputSizeN = transposeX2 ? params.x2->GetViewShape().GetDim(x2DimNum - PENULTIMATE_DIM) :
                                        params.x2->GetViewShape().GetDim(x2DimNum - 1);
        if (static_cast<ge::Format>(ge::GetPrimaryFormat(params.x2->GetStorageFormat())) == Format::FORMAT_FRACTAL_NZ) {
            if (inputSizeM == 0) {
                OP_LOGD("aclnnQuantBatchMatmulInplaceAdd nz m=0");
                return ACLNN_SUCCESS;
            }
        } else {
            if (inputSizeM == 0 || inputSizeN == 0) {
                OP_LOGD("aclnnQuantBatchMatmulInplaceAdd nd m/n=0");
                return ACLNN_SUCCESS;
            }
        }
    }
    // Invoke l0 operator QuantBatchMatmulInplaceAdd for calculation.
    auto result = l0op::QuantBatchMatmulInplaceAdd(
        params.x1, params.x2, params.x2Scale, params.yRef, params.x1ScaleOptional, transposeX1, transposeX2, params.groupSize,
        executor);
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyResult = l0op::ViewCopy(result, params.yRef, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

} // namespace

#ifdef __cplusplus
extern "C" {
#endif
aclnnStatus aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize(
    const aclTensor* x1, const aclTensor* x2, const aclTensor* x1ScaleOptional, const aclTensor* x2Scale, aclTensor* yRef,
    bool transposeX1, bool transposeX2, int64_t groupSize, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    QBMMInplaceAdd::QuantBatchMatmulInplaceAddParams params{x1,   x2,          x1ScaleOptional, x2Scale,
                                                            yRef, transposeX1, transposeX2,     groupSize};

    L2_DFX_PHASE_1(
        aclnnQuantBatchMatmulInplaceAdd, DFX_IN(x1, x2, x1ScaleOptional, x2Scale, yRef, transposeX1, transposeX2, groupSize),
        DFX_OUT(yRef));

    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    auto ret = aclnnQuantBatchMatmulInplaceAddGetWorkspaceSizeCommon(params, executorPtr);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    // Standard syntax, get the size of workspace needed during computation.
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantBatchMatmulInplaceAdd(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantBatchMatmulInplaceAdd);
    CHECK_COND(
        CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
        "This is an error in QuantBatchMatmulInplaceAdd launch aicore.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
