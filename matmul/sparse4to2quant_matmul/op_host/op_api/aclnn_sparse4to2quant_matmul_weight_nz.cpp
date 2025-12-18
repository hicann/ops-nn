/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn_sparse4to2quant_matmul_weight_nz.h"
#include "matmul/common/op_host/op_api/matmul_util.h"
#include "sparse4to2quant_matmul.h"
#include "util/math_util.h"

using namespace op;
using Ops::Base::CeilDiv;
using Ops::Base::CeilAlign;
using Ops::NN::IsTransposeLastTwoDims;
using Ops::NN::SwapLastTwoDimValue;

namespace {
static const int MIN_DIM_NUM_ND = 2;
static const int MIN_DIM_NUM_NZ = 4;
static const size_t PENULTIMATE_DIM = 2;
static const int NZ_K1_INDEX = 3;
static const int NZ_K1_INDEX_TRANS = 4;
static const int NZ_STORAGE_PENULTIMATE_DIM = 16;
static const int NZ_STORAGE_LAST_DIM = 32;
static const int64_t NZ_K0_VALUE_INT8 = 16L;
static const int64_t NZ_K0_VALUE_INT8_TRANS = 32L;
static const int64_t LAST_AXIS_LIMIT = 65535L;
static const size_t SPARSE_ATOMIC_SIZE = 8;

struct Sparse4to2QuantMatmulParams {
    const aclTensor* x;
    const aclTensor* sparseWeight;
    const aclTensor* index;
    const aclTensor* xScale;
    const aclTensor* sparseWeightScale;
    const aclTensor* bias;
    aclTensor* out;
};

static inline bool CheckNotNull(const Sparse4to2QuantMatmulParams& params)
{
    OP_CHECK_NULL(params.x, return false);
    OP_CHECK_NULL(params.sparseWeight, return false);
    OP_CHECK_NULL(params.index, return false);
    OP_CHECK_NULL(params.out, return false);
    OP_CHECK_NULL(params.sparseWeightScale, return false);
    OP_CHECK_NULL(params.xScale, return false);
    return true;
}

static inline bool CheckDtypeValid(const Sparse4to2QuantMatmulParams& params)
{
    auto x = params.x;
    auto sparseWeight = params.sparseWeight;
    auto index = params.index;
    auto sparseWeightScale = params.sparseWeightScale;
    auto bias = params.bias;
    auto xScale = params.xScale;
    auto out = params.out;

    if (x->GetDataType() != op::DataType::DT_INT8 || sparseWeight->GetDataType() != op::DataType::DT_INT8) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Input x, sparseWeight dtype should be INT8, actual dtype are %s and %s.",
            op::ToString(x->GetDataType()).GetString(), op::ToString(sparseWeight->GetDataType()).GetString());
        return false;
    }

    if (index->GetDataType() != op::DataType::DT_UINT8) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Input index dtype should be UINT8, actual dtype is %s.",
            op::ToString(index->GetDataType()).GetString());
        return false;
    }

    if (bias != nullptr && bias->GetDataType() != op::DataType::DT_BF16) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Input bias dtype should be BF16, actual dtype is %s.",
            op::ToString(bias->GetDataType()).GetString());
        return false;
    }

    if (xScale != nullptr && xScale->GetDataType() != op::DataType::DT_FLOAT) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Input xScale dtype should be FLOAT32, actual dtype is %s.",
            op::ToString(xScale->GetDataType()).GetString());
        return false;
    }

    if (sparseWeightScale != nullptr && sparseWeightScale->GetDataType() != op::DataType::DT_FLOAT) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Input sparseWeightScale dtype should be FLOAT32, actual dtype is %s.",
            op::ToString(sparseWeightScale->GetDataType()).GetString());
        return false;
    }

    if (out->GetDataType() != op::DataType::DT_BF16) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Output dtype should be BF16, actual dtype is %s.",
            op::ToString(out->GetDataType()).GetString());
        return false;
    }

    return true;
}

static inline bool CheckFormatValid(const Sparse4to2QuantMatmulParams& params)
{
    OP_CHECK(
        params.x->GetStorageFormat() == op::Format::FORMAT_ND,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of x must be ND, actual is %s.",
            op::ToString(params.x->GetStorageFormat()).GetString()),
        return false);

    OP_CHECK(
        static_cast<ge::Format>(ge::GetPrimaryFormat(params.sparseWeight->GetStorageFormat())) ==
            Format::FORMAT_FRACTAL_NZ,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of sparseWeight must be FRACTAL_NZ, actual is %s.",
            op::ToString(params.sparseWeight->GetStorageFormat()).GetString()),
        return false);

    OP_CHECK(params.index->GetStorageFormat() == op::Format::FORMAT_ND,
             OP_LOGE(
                 ACLNN_ERR_PARAM_INVALID, "Format of index must be ND, actual is %s.",
                 op::ToString(params.index->GetStorageFormat()).GetString());
             , return false);

    if (params.xScale != nullptr) {
        OP_CHECK(params.xScale->GetStorageFormat() == op::Format::FORMAT_ND,
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Format of xScale must be ND, actual is %s.",
                op::ToString(params.xScale->GetStorageFormat()).GetString()),
            return false);
    }

    if (params.sparseWeightScale != nullptr) {
        OP_CHECK(params.sparseWeightScale->GetStorageFormat() == op::Format::FORMAT_ND,
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Format of sparseWeightScale must be ND, actual is %s.",
                op::ToString(params.sparseWeightScale->GetStorageFormat()).GetString()),
            return false);
    }

    if (params.bias != nullptr) {
        OP_CHECK(
            params.bias->GetStorageFormat() == op::Format::FORMAT_ND,
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "Format of bias must be ND, actual is %s.",
                op::ToString(params.bias->GetStorageFormat()).GetString()),
            return false);
    }

    OP_CHECK(params.out->GetStorageFormat() == op::Format::FORMAT_ND,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID, "Format of out must be ND, actual is %s.",
            op::ToString(params.out->GetStorageFormat()).GetString()),
        return false);
    return true;
}

static inline bool CheckDimRange(const Sparse4to2QuantMatmulParams& params)
{
    OP_CHECK_WRONG_DIMENSION(params.x, MIN_DIM_NUM_ND, return false);
    OP_CHECK_WRONG_DIMENSION(params.sparseWeight, MIN_DIM_NUM_ND, return false);
    OP_CHECK_WRONG_DIMENSION(params.index, MIN_DIM_NUM_ND, return false);
    OP_CHECK_WRONG_DIMENSION(params.out, MIN_DIM_NUM_ND, return false);
    if (params.bias != nullptr) {
        OP_CHECK_WRONG_DIMENSION(params.bias, 1, return false);
    }

    if (params.xScale != nullptr) {
        OP_CHECK_WRONG_DIMENSION(params.xScale, 1, return false);
    }

    if (params.sparseWeightScale != nullptr) {
        OP_CHECK_WRONG_DIMENSION(params.sparseWeightScale, 1, return false);
    }

    OP_LOGD("check dim-num range success");
    return true;
}

static inline bool CheckOutShape(const aclTensor* out, int64_t xMDim, int64_t weightNDim)
{
    auto outDimNum = out->GetViewShape().GetDimNum();
    OP_CHECK(
        outDimNum == MIN_DIM_NUM_ND,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Out shape must be 2 dimension actual is %ld.", outDimNum), return false);
    int64_t outMDim = out->GetViewShape().GetDim(outDimNum - PENULTIMATE_DIM);
    int64_t outNDim = out->GetViewShape().GetDim(outDimNum - 1);
    OP_CHECK(
        outMDim == xMDim,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "First dimension of out should be equal to m dimension of x, but fir dimension of out is %ld, m dimension "
            "of x is %ld.",
            outMDim, xMDim),
        return false);
    OP_CHECK(
        outNDim == weightNDim,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Second dim of out should be equal to n dim of sparseWeight , but out 2nd dim is %ld, n dim of "
            "sparseWeight is %ld.",
            outNDim, weightNDim),
        return false);
    return true;
}

static inline bool CheckDimValue(
    const aclTensor* bias, const aclTensor* xScale, const aclTensor* sparseWeightScale, int64_t mDim, int64_t nDim)
{
    if (bias != nullptr) {
        OP_CHECK_WRONG_DIMENSION(bias, 1, return false);
        if (bias->GetViewShape().GetDim(0) != nDim) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID, "The 1st dim bias should be equal to n dim of x2: %ld, but actually is %ld.",
                nDim, bias->GetViewShape().GetDim(0));
            return false;
        }
    }

    if (sparseWeightScale != nullptr) {
        OP_CHECK_WRONG_DIMENSION(sparseWeightScale, 1, return false);
        if (sparseWeightScale->GetViewShape().GetDim(0) != nDim) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "The last dim of sparseWeightScale should equal to n dim of sparseWeight: %ld, but actual is %ld.",
                nDim, sparseWeightScale->GetViewShape().GetDim(0));
            return false;
        }
    }

    if (xScale != nullptr) {
        OP_CHECK_WRONG_DIMENSION(xScale, 1, return false);
        if (xScale->GetViewShape().GetDim(0) != mDim) {
            OP_LOGE(
                ACLNN_ERR_PARAM_INVALID,
                "The 1st dim of xScale should be equal to m dim of x: %ld, but actually is %ld.", mDim,
                xScale->GetViewShape().GetDim(0));
            return false;
        }
    }
    return true;
}

static inline bool MaxDimCheck(
    int64_t xDimNum, int64_t weightKDim, const op::Shape xShape, const op::Shape& weightShape)
{
    OP_CHECK(
        xShape[xDimNum - 1] <= LAST_AXIS_LIMIT && weightShape[weightKDim - 1] <= LAST_AXIS_LIMIT,
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "The last dim of x or sparseWeight is larger than 65535, x is %ld, sparseWeight is %ld.",
            xShape[xDimNum - 1], weightShape[weightKDim - 1]),
        return false);
    return true;
}

static inline bool CheckShape(const Sparse4to2QuantMatmulParams& params)
{
    auto& xShape = params.x->GetViewShape();
    auto& weightShape = params.sparseWeight->GetViewShape();
    auto xDimNum = params.x->GetViewShape().GetDimNum();
    auto weightDimNum = params.sparseWeight->GetViewShape().GetDimNum();
    int64_t xKDim = xShape[xDimNum - 1];
    int64_t xMDim = xShape[xDimNum - PENULTIMATE_DIM];
    int64_t weightKDim = weightShape[weightDimNum - 1];
    int64_t weightNDim = weightShape[weightDimNum - PENULTIMATE_DIM];
    uint64_t xKDimAlign = CeilAlign(static_cast<uint64_t>(xKDim), SPARSE_ATOMIC_SIZE);

    OP_CHECK(
        xKDimAlign == static_cast<uint64_t>(weightKDim + weightKDim),
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "The k dim of x (8-aligned) should be twice that k of sparseWeight, but k dim of x is %lu, "
            "sparseWeight is %ld.", xKDimAlign, weightKDim),
        return false);

    CHECK_RET(MaxDimCheck(xDimNum, weightDimNum, xShape, weightShape), false);

    CHECK_RET(CheckDimValue(params.bias, params.xScale, params.sparseWeightScale, xMDim, weightNDim), false);

    CHECK_RET(CheckOutShape(params.out, xMDim, weightNDim), false);
    return true;
}

static inline bool CheckEmptyTensor(const Sparse4to2QuantMatmulParams& params)
{
    // scale, out和可选参数已在CheckShape函数校验，无需再次校验空tensor场景。
    if (params.x->IsEmpty() || params.sparseWeight->IsEmpty()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Sparse4to2QuantMatmul not support to process empty tensor for now.");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const Sparse4to2QuantMatmulParams& params)
{
    // 1. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(params), ACLNN_ERR_PARAM_INVALID);

    // 2. 检查format是否符合要求
    CHECK_RET(CheckFormatValid(params), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否符合要求
    CHECK_RET(CheckShape(params), ACLNN_ERR_PARAM_INVALID);

    // 4. 空Tensor处理逻辑
    CHECK_RET(CheckEmptyTensor(params), ACLNN_ERR_PARAM_INVALID);

    OP_LOGD("Sparse4to2QuantMatmul check params success.");
    return ACLNN_SUCCESS;
}

static inline bool TensorContiguousProcess(const aclTensor*& contiguousTensor, bool& transpose, aclOpExecutor* executor)
{
    if (contiguousTensor == nullptr) {
        OP_LOGD("Sparse4to2QuantMatmul no need to do contiguous process.");
        return true;
    }
    bool isNZTensor = static_cast<ge::Format>(ge::GetPrimaryFormat(contiguousTensor->GetStorageFormat())) ==
                      op::Format::FORMAT_FRACTAL_NZ;
    auto& stroageShape = contiguousTensor->GetStorageShape();
    auto transposeFlag = IsTransposeLastTwoDims(contiguousTensor);
    // swap tensor if its viewshape not satisfy request shape without adding a transpose node
    if (transposeFlag) {
        contiguousTensor = executor->CreateView(
            contiguousTensor, SwapLastTwoDimValue(contiguousTensor->GetViewShape()), contiguousTensor->GetViewOffset());
        transpose = !transpose;
    } else {
        contiguousTensor = l0op::Contiguous(contiguousTensor, executor);
    }
    if (isNZTensor) {
        contiguousTensor->SetStorageShape(stroageShape); // 对NZ的场景需要用原NZshape刷新
    }
    CHECK_RET(contiguousTensor != nullptr, false);
    return true;
}

static aclnnStatus SpecialOutputProcess(
    const aclTensor* x, const aclTensor* sparseWeight, const aclTensor* out, const aclTensor*& matmulRet,
    aclOpExecutor* executor)
{
    OP_LOGD("Sparse4to2QuantMatmul enter SpecialOutputProcess func.");
    auto xDimNum = x->GetViewShape().GetDimNum();
    auto weightDimNum = sparseWeight->GetViewShape().GetDimNum();
    auto& outShape = out->GetViewShape();
    auto outDimNum = outShape.GetDimNum();
    int64_t outMDim = outShape.GetDim(outDimNum - PENULTIMATE_DIM);
    // x and sparseWeight are 2 dim, output is 3 dim, have to reshape matmul result, otherwise viewcopy will fail.
    if (xDimNum == 2 && outDimNum == 3 && outMDim == 1 && weightDimNum == 2) {
        matmulRet = l0op::Reshape(matmulRet, outShape, executor);
    }
    CHECK_RET(matmulRet != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus PostMatmulCalcProcess(
    const aclTensor* matmulRet, const Sparse4to2QuantMatmulParams& params, aclOpExecutor* executor)
{
    CHECK_RET(matmulRet != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(
        SpecialOutputProcess(params.x, params.sparseWeight, params.out, matmulRet, executor) == ACLNN_SUCCESS,
        ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(matmulRet, params.out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus PreMatmulCalcProcess(Sparse4to2QuantMatmulParams& params, aclOpExecutor* executor)
{
    bool transposeX = false;
    CHECK_RET(executor != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(TensorContiguousProcess(params.x, transposeX, executor), ACLNN_ERR_INNER_NULLPTR);

    OP_CHECK(
        !transposeX, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Input x not support transpose for now"),
        return ACLNN_ERR_PARAM_INVALID);

    params.sparseWeight->SetOriginalShape(params.sparseWeight->GetViewShape());
    CHECK_RET(CheckDimRange(params), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static op::Shape GetWeightNzShape(const aclTensor* input)
{
    size_t viewDimNum = input->GetViewShape().GetDimNum();
    int64_t k = input->GetViewShape().GetDim(viewDimNum - 1);
    int64_t n = input->GetViewShape().GetDim(viewDimNum - PENULTIMATE_DIM);

    int64_t k1 = CeilDiv(k, NZ_K0_VALUE_INT8_TRANS);
    int64_t n1 = CeilDiv(n, NZ_K0_VALUE_INT8);

    op::Shape weightNzShape;
    for (size_t i = 0; i < viewDimNum - PENULTIMATE_DIM; i++) {
        weightNzShape.AppendDim(input->GetViewShape().GetDim(i));
    }

    weightNzShape.AppendDim(k1);
    weightNzShape.AppendDim(n1);
    weightNzShape.AppendDim(NZ_STORAGE_PENULTIMATE_DIM);
    weightNzShape.AppendDim(NZ_STORAGE_LAST_DIM);
    return weightNzShape;
}

static const aclTensor* SetTensorToNZFormat(const aclTensor* input, op::Shape& shape, aclOpExecutor* executor)
{
    auto formatTensor = executor->CreateView(input, shape, input->GetViewOffset());
    OP_CHECK_NULL(formatTensor, return nullptr);
    formatTensor->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
    formatTensor->SetOriginalFormat(op::Format::FORMAT_ND);
    formatTensor->SetViewShape(input->GetViewShape());
    return formatTensor;
}

} // namespace

aclnnStatus aclnnSparse4to2QuantMatmulWeightNzGetWorkspaceSize(
    const aclTensor* x, const aclTensor* sparseWeight, const aclTensor* index, const aclTensor* xScale,
    const aclTensor* sparseWeightScale, const aclTensor* biasOptional, aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(
        aclnnSparse4to2QuantMatmulWeightNz, DFX_IN(x, sparseWeight, index, xScale, sparseWeightScale, biasOptional),
        DFX_OUT(out));

    OP_CHECK(
        GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93 ||
            GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B,
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Sparse4to2QuantMatmul is not supported in current platform"),
        return ACLNN_ERR_PARAM_INVALID);

    OP_CHECK_NULL(sparseWeight, return ACLNN_ERR_INNER_NULLPTR);

    OP_CHECK_WRONG_DIMENSION(sparseWeight, MIN_DIM_NUM_ND, return ACLNN_ERR_PARAM_INVALID);

    auto weightNzShape = GetWeightNzShape(sparseWeight);
    if (weightNzShape != sparseWeight->GetStorageShape()) {
        OP_LOGE(
            ACLNN_ERR_PARAM_INVALID,
            "Input sparseWeight'format only support NZ with transpose, but now is not NZ(Ascend affinity format), view "
            "shape is %s, expected NZ storage shape is %s, actual NZ storage shape is %s",
            op::ToString(sparseWeight->GetViewShape()).GetString(), op::ToString(weightNzShape).GetString(),
            op::ToString(sparseWeight->GetStorageShape()).GetString());
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    uniqueExecutor->AbandonCache();

    sparseWeight = SetTensorToNZFormat(sparseWeight, weightNzShape, uniqueExecutor.get());
    OP_CHECK(
        sparseWeight != nullptr, OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Failed to reset format of sparseWeight."),
        return ACLNN_ERR_PARAM_INVALID);
    Sparse4to2QuantMatmulParams params({x, sparseWeight, index, xScale, sparseWeightScale, biasOptional, out});

    auto ret = PreMatmulCalcProcess(params, uniqueExecutor.get());
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    int64_t dtype = static_cast<int64_t>(out->GetDataType());

    // 调用l0算子进行计算
    auto matmulRet = l0op::Sparse4to2QuantMatmul(
        x, sparseWeight, index, xScale, sparseWeightScale, biasOptional, dtype, uniqueExecutor.get());
    CHECK_RET(PostMatmulCalcProcess(matmulRet, params, uniqueExecutor.get()) == ACLNN_SUCCESS, ret);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSparse4to2QuantMatmulWeightNz(
    void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSparse4to2QuantMatmulWeightNz);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}