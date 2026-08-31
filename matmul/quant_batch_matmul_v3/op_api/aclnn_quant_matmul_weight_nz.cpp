/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_quant_matmul_v4.h"
#include <dlfcn.h>
#include "aclnn_quant_matmul_v3.h"
#include "aclnn_quant_matmul_weight_nz.h"
#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "log/log.h"
#include "matmul/common/op_host/log_format_util.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "quant_matmul_v3.h"
#include "matmul/common/op_host/op_api/quant_matmul_v4.h"
#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/transpose.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "util/math_util.h"
#include "quant_matmul_checker.h"
#include "quant_matmul_v4_common.h"

using namespace op;
using namespace quant_matmul_v4;
using Ops::Base::CeilDiv;
using Ops::NN::BoolToString;
using Ops::NN::FormatString;
using Ops::NN::IsTransposeLastTwoDims;
using Ops::NN::SwapLastTwoDimValue;

namespace {
static op::Shape GetWeightNzShape(const aclTensor* input, bool transpose, bool isA8W4Float)
{
    size_t viewDimNum = input->GetViewShape().GetDimNum();
    int64_t k = transpose ? input->GetViewShape().GetDim(viewDimNum - 1) :
                            input->GetViewShape().GetDim(viewDimNum - LAST_SECOND_DIM_INDEX);
    int64_t n = transpose ? input->GetViewShape().GetDim(viewDimNum - LAST_SECOND_DIM_INDEX) :
                            input->GetViewShape().GetDim(viewDimNum - 1);

    int64_t nz_k0_value_trans = SelectNzK0Value(input->GetDataType(), isA8W4Float);
    int64_t k1 = transpose ? CeilDiv(k, nz_k0_value_trans) : CeilDiv(k, NZ_K0_VALUE_BMM_BLOCK_NUM);
    int64_t n1 = transpose ? CeilDiv(n, NZ_K0_VALUE_BMM_BLOCK_NUM) : CeilDiv(n, nz_k0_value_trans);

    op::Shape weightNzShape;
    for (size_t i = 0; i < viewDimNum - LAST_SECOND_DIM_INDEX; i++) {
        weightNzShape.AppendDim(input->GetViewShape().GetDim(i));
    }
    if (transpose) {
        weightNzShape.AppendDim(k1);
        weightNzShape.AppendDim(n1);
    } else {
        weightNzShape.AppendDim(n1);
        weightNzShape.AppendDim(k1);
    }
    weightNzShape.AppendDim(NZ_STORAGE_PENULTIMATE_DIM);
    weightNzShape.AppendDim(nz_k0_value_trans);
    return weightNzShape;
}

static bool CheckWeightNzStorageShape(const op::Shape& nzShape, const op::Shape& storageShape)
{
    uint64_t nzDimMultiply = 1;
    uint64_t nzDimNum = nzShape.GetDimNum();
    for (uint64_t i = 0; i < nzDimNum; i++) {
        nzDimMultiply *= nzShape[i];
    }

    uint64_t storageDimMultiply = 1;
    uint64_t storageDimNum = storageShape.GetDimNum();
    for (uint64_t i = 0; i < storageDimNum; i++) {
        storageDimMultiply *= storageShape[i];
    }

    return nzDimMultiply == storageDimMultiply;
}

static const aclTensor* SetTensorToNZFormat(const aclTensor* input, op::Shape& shape, aclOpExecutor* executor)
{
    auto formatTensor = executor->CreateView(input, shape, input->GetViewOffset());
    CHECK_RET(formatTensor != nullptr, nullptr);
    formatTensor->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
    formatTensor->SetOriginalFormat(op::Format::FORMAT_ND);
    formatTensor->SetViewShape(input->GetViewShape());
    return formatTensor;
}

bool checkNotSupportParam(TupleTensor mandatoryTensors, const aclTensor* pertokenScale, const aclTensor* yScale,
                          const aclTensor* x1Offset, const aclTensor* yOffset, int64_t groupSize)
{
    auto& x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);

    if (x1Offset != nullptr && x1Offset->GetViewShape().GetShapeSize() != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Current version do not support x1Offset.");
        return false;
    }

    if (yOffset != nullptr && yOffset->GetViewShape().GetShapeSize() != 0 && !isA8W4Msd(x1, x2, scale, pertokenScale)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Current version do not support yOffset.");
        return false;
    }

    auto isFloatScale2D = [](const aclTensor* s) {
        return s != nullptr && s->GetDataType() == op::DataType::DT_FLOAT && s->GetViewShape().GetDimNum() > 1;
    };
    auto isPerblockQuantInput = [](const aclTensor* a, const aclTensor* b) {
        auto isQuant = [](const aclTensor* t) {
            if (t == nullptr) {
                return false;
            }
            auto dt = t->GetDataType();
            return dt == op::DataType::DT_INT8 || dt == op::DataType::DT_FLOAT8_E4M3FN ||
                   dt == op::DataType::DT_FLOAT8_E5M2 || dt == op::DataType::DT_HIFLOAT8;
        };
        return isQuant(a) && isQuant(b);
    };
    const bool isPerblockGroup = isPerblockQuantInput(x1, x2) && isFloatScale2D(pertokenScale) && isFloatScale2D(scale);

    if (!(isA8W4Float(x1, x2) || isMx(scale))) {
        if (yScale != nullptr && yScale->GetViewShape().GetShapeSize() != 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Current version do not support yScale.");
            return false;
        }

        if (groupSize != 0 && !isPerblockGroup) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Current version do not support groupSize.");
            return false;
        }
    }

    return true;
}

static void SetStorageShapeForNZ(aclTensor* tensor)
{
    // storageShape的倒数第一维要放大8倍， 比如(n/32,k/16,16,4) -> (n/32,k/16,16,32)
    auto storageShape = tensor->GetStorageShape();
    auto storageShapeDim = storageShape.GetDimNum();
    storageShape[storageShapeDim - 1] *= B4_PER_B32;
    tensor->SetStorageShape(storageShape);
}

static void UnpackB32ToB4(const aclTensor* tensorB32)
{
    DataType b32Dtype = tensorB32->GetDataType();
    DataType b4Dtype = DataType::DT_INT4;
    if (b32Dtype == DataType::DT_FLOAT) {
        b4Dtype = DataType::DT_FLOAT4_E2M1;
    }

    OP_LOGD("Unpack from %s to %s start.", op::ToString(b32Dtype).GetString(), op::ToString(b4Dtype).GetString());
    auto tensorB4 = const_cast<aclTensor*>(tensorB32);
    op::Shape tensorShape = tensorB4->GetViewShape();
    op::Strides newStride = tensorB4->GetViewStrides();
    auto viewShapeDim = tensorShape.GetDimNum();
    bool transposeTensor = false;
    auto changeDimIdx = viewShapeDim - 1;
    // 轴大于等于2才判断是否转置
    if (viewShapeDim >= 2 && IsTransposeLastTwoDims(tensorB4)) {
        transposeTensor = true;
        // 转置场景扩大倒数第2维
        changeDimIdx = viewShapeDim - 2;
    }
    tensorShape[changeDimIdx] = tensorShape[changeDimIdx] * B4_PER_B32;
    tensorB4->SetViewShape(tensorShape);
    tensorB4->SetDataType(b4Dtype);
    if (IsFormatNZ(tensorB4)) {
        SetStorageShapeForNZ(tensorB4);
    }

    if (transposeTensor) {
        auto strideSize = newStride.size();
        // 转置场景，B32承载B4时strides缩小了8倍，需要放大， 即（k*n/8, 1，k/8）->(k*n, 1, k)
        newStride[strideSize - 1] *= B4_PER_B32;
        tensorB4->SetViewStrides(newStride);
    }
    OP_LOGD("Current tensor transpose status: %d.", transposeTensor);
    OP_LOGD("Unpack from %s to %s finished.", op::ToString(b32Dtype).GetString(), op::ToString(b4Dtype).GetString());
}
} // namespace

static aclnnStatus modifyScaleStorageShape(const aclTensor* scale)
{
    auto scaleShape = scale->GetViewShape();
    auto scaleStorageShape = scale->GetStorageShape();
    int64_t dimNum = scaleStorageShape.GetDimNum();
    // 1维的storage shape需要修正为2维的viewShape，
    if (dimNum == 1) {
        uint64_t viewShapeMultiply = 1;
        uint64_t viewShapeDimNum = scaleShape.GetDimNum();
        for (uint64_t i = 0; i < viewShapeDimNum; i++) {
            viewShapeMultiply *= scaleShape[i];
        }

        uint64_t storageShapeMultiply = 1;
        storageShapeMultiply *= scaleStorageShape[0];
        if (viewShapeMultiply != storageShapeMultiply) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "The product of view shape dimensions %ld does not equal the product of storage shape dimensions %ld .",
                viewShapeMultiply, storageShapeMultiply);
            return ACLNN_ERR_PARAM_INVALID;
        }

        scale->SetStorageShape(scaleShape);
        OP_LOGD("modify storage shape to view shape finish.");
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus preProcessTensor(const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale,
                                    const aclTensor* x2Scale)
{
    if (x1->GetDataType() == op::DataType::DT_FLOAT8_E4M3FN && x2->GetDataType() == op::DataType::DT_FLOAT) {
        UnpackB32ToB4(x2);
        // mx场景下修正x1_scale和x2_scale的1维的storage shape
        if (x1Scale != nullptr) {
            auto ret = modifyScaleStorageShape(x1Scale);
            CHECK_RET(ret == ACLNN_SUCCESS, ret);
        }

        if (x2Scale != nullptr) {
            auto ret = modifyScaleStorageShape(x2Scale);
            CHECK_RET(ret == ACLNN_SUCCESS, ret);
        }
    }

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantMatmulWeightNzGetWorkspaceSize(const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale,
                                                     const aclTensor* x2Scale, const aclTensor* yScale,
                                                     const aclTensor* x1Offset, const aclTensor* x2Offset,
                                                     const aclTensor* yOffset, const aclTensor* bias, bool transposeX1,
                                                     bool transposeX2, int64_t groupSize, aclTensor* out,
                                                     uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnQuantMatmulWeightNz,
                   DFX_IN(x1, x2, x1Scale, x2Scale, yScale, x1Offset, x2Offset, yOffset, bias, transposeX1, transposeX2,
                          groupSize),
                   DFX_OUT(out));

    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    OP_CHECK_NULL(x1, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(x2, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(x2Scale, return ACLNN_ERR_PARAM_NULLPTR);
    OP_CHECK_NULL(out, return ACLNN_ERR_PARAM_NULLPTR);
    if (!checkNotSupportParam(std::tie(x1, x2, x2Scale), x1Scale, yScale, x1Offset, yOffset, groupSize)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto ret = quant_matmul_v4::internal::CheckWeightNzParamsDAV3510(x1, x2, x1Scale, x2Scale, out);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (x2 == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "QuantMatmul WeightNz do not support x2 is nullptr.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    int64_t viewDimNum = x2->GetViewShape().GetDimNum();
    if (viewDimNum < MIN_DIM_NUM_ND) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "x2's view dimNum should greater than 1, but is %ld.", viewDimNum);
        return ACLNN_ERR_PARAM_INVALID;
    }

    // 修改传入tensor的format和shape
    ret = preProcessTensor(x1, x2, x1Scale, x2Scale);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (!isA8W4Int(x1, x2)) {
        transposeX2 = quant_matmul_v4::internal::GetTransposeAttrValue(x2, transposeX2, false);
    }

    op::Shape weightNzShape = GetWeightNzShape(x2, transposeX2, isA8W4Float(x1, x2));
    if (!CheckWeightNzStorageShape(weightNzShape, x2->GetStorageShape())) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "x2'format only support NZ, but now x2's format is not NZ(Ascend affinity format). \
aclnnCalculateMatmulWeightSizeV2 and aclnnTransMatmulWeight can be used to convert the input format from ND to Ascend \
affinity format.");
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    x2 = SetTensorToNZFormat(x2, weightNzShape, uniqueExecutor.get());
    CHECK_RET(x2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    ret = quant_matmul_v4::internal::aclnnQuantMatmulGetWorkspaceSizeCommonProcess(
        std::tie(x1, x2, x2Scale), std::tie(x2Offset, x1Scale, bias, yScale, yOffset, groupSize),
        std::tie(transposeX1, transposeX2), out, uniqueExecutor.get(), "aclnnQuantMatmulWeightNzGetWorkspaceSize");
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
