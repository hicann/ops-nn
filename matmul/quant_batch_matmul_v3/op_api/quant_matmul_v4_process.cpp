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

namespace quant_matmul_v4 {
namespace internal {
static bool CheckSpecialCase(const aclTensor* tensor, int64_t firstLastDim, int64_t secondLastDim)
{
    if ((tensor->GetViewShape().GetDim(firstLastDim) == tensor->GetViewShape().GetDim(secondLastDim)) &&
        (tensor->GetViewShape().GetDim(secondLastDim) == 1)) {
        OP_LOGD("QuantMatmul special case, no need to set transpose attr value.");
        return true;
    }
    return false;
}

bool GetTransposeAttrValue(const aclTensor* tensor, bool transpose, bool checkSpecialCase)
{
    int64_t dim1 = tensor->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = tensor->GetViewShape().GetDimNum() - PENULTIMATE_DIM;
    // check if tensor is contiguous layout
    if (tensor->GetViewStrides()[dim2] == 1 &&
        (tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2))) {
        OP_LOGD("QuantMatmul GetTransposeAttrValue, find tensor is not contiguous.");
        const_cast<aclTensor*>(tensor)->SetViewShape(SwapLastTwoDimValue(tensor->GetViewShape()));
        // 如果不需要校验特殊case，则直接返回
        if (!checkSpecialCase) {
            return !transpose;
        }
        if (!CheckSpecialCase(tensor, dim1, dim2)) {
            return !transpose;
        }
    }
    return transpose;
}

static const aclTensor* SetTensorToNDFormat(const aclTensor* input)
{
    OP_LOGD("QuantMatmul set tensor to ND format.");
    auto formatTensor = const_cast<aclTensor*>(input);
    if (input->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ) {
        formatTensor->SetViewFormat(op::Format::FORMAT_ND);
        formatTensor->SetOriginalFormat(op::Format::FORMAT_ND);
        formatTensor->SetStorageFormat(op::Format::FORMAT_ND);
    }
    return formatTensor;
}

static aclIntArray* GetPerm(int64_t dim, aclOpExecutor* executor)
{
    CHECK_RET(dim >= MIN_DIM_NUM_ND, nullptr);
    std::vector<int64_t> valuePerm(dim);
    for (int64_t i = 0; i < dim; i++) {
        valuePerm[i] = i;
    }
    std::swap(valuePerm[dim - 1], valuePerm[dim - PENULTIMATE_DIM]);
    return executor->AllocIntArray(valuePerm.data(), dim);
}

static aclnnStatus TransposeAndTransDataForInputs(const aclTensor*& x1, const aclTensor*& x2, bool& transposeX1,
                                                  bool& transposeX2, aclOpExecutor* executor)
{
    if (transposeX1) {
        auto perm = GetPerm(x1->GetViewShape().GetDimNum(), executor);
        CHECK_RET(perm != nullptr, ACLNN_ERR_INNER_NULLPTR);
        x1 = l0op::Transpose(x1, perm, executor);
        CHECK_RET(x1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
        transposeX1 = !transposeX1;
    }
    if (static_cast<ge::Format>(ge::GetPrimaryFormat(x2->GetStorageFormat())) == Format::FORMAT_FRACTAL_NZ) {
        return ACLNN_SUCCESS;
    }
    x2 = SetTensorToNDFormat(x2);
    if (!transposeX2) {
        auto perm = GetPerm(x2->GetViewShape().GetDimNum(), executor);
        CHECK_RET(perm != nullptr, ACLNN_ERR_INNER_NULLPTR);
        x2 = l0op::Transpose(x2, perm, executor);
        CHECK_RET(x2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
        transposeX2 = !transposeX2;
    }
    x2 = l0op::TransData(x2, Format::FORMAT_FRACTAL_NZ, 0, executor);
    CHECK_RET(x2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus TransdataForX1(const aclTensor*& inputTensor, aclOpExecutor* executor)
{
    OP_LOGD("QuantMatmul enter TransdataForX1 func.");
    inputTensor = l0op::Contiguous(inputTensor, executor);
    OP_CHECK(inputTensor != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "The function Contiguous() return nullptr, which causes function TransdataForX1() to fail."),
             return ACLNN_ERR_INNER_NULLPTR);
    inputTensor = l0op::TransData(inputTensor, Format::FORMAT_FRACTAL_NZ, 0, executor);
    OP_CHECK(inputTensor != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                     "The function TransData() return nullptr, which causes function TransdataForX1() to fail."),
             return ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static inline bool TensorContiguousProcess(const aclTensor*& contiguousTensor, bool& transpose, aclOpExecutor* executor)
{
    if (contiguousTensor == nullptr) {
        OP_LOGD("QuantMatmul no need to do contiguous process.");
        return true;
    }
    bool isNZTensor = static_cast<ge::Format>(ge::GetPrimaryFormat(contiguousTensor->GetStorageFormat())) ==
                      op::Format::FORMAT_FRACTAL_NZ;
    auto storageShape = contiguousTensor->GetStorageShape();
    auto transposeFlag = IsTransposeLastTwoDims(contiguousTensor);
    // swap tensor if its viewshape not satisfy request shape without adding a transpose node
    if (transposeFlag) {
        contiguousTensor = executor->CreateView(contiguousTensor, SwapLastTwoDimValue(contiguousTensor->GetViewShape()),
                                                contiguousTensor->GetViewOffset());
        transpose = !transpose;
    } else {
        contiguousTensor = l0op::Contiguous(contiguousTensor, executor);
    }
    if (isNZTensor) {
        contiguousTensor->SetStorageShape(storageShape); // 对NZ的场景需要用原NZshape刷新
    }
    CHECK_RET(contiguousTensor != nullptr, false);
    return true;
}

static aclnnStatus SpecialOutputProcess(const aclTensor* x1, const aclTensor* x2, const aclTensor* out,
                                        const aclTensor*& matmulRet, aclOpExecutor* executor)
{
    // we have to reshape for case which x1 and x2 are 2 dims and out is 3 dims, otherwise, viewcopy will fail
    OP_LOGD("QuantMatmul enter SpecialOutputProcess func.");
    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x2DimNum = x2->GetViewShape().GetDimNum();
    auto outShape = out->GetViewShape();
    auto outDimNum = outShape.GetDimNum();
    int64_t outMDim = outShape.GetDim(outDimNum - 2);
    // speical case : x1 and x2 are 2 dim, output is 3 dim, have to reshape matmul result, otherwise viewcopy will fail.
    if (x1DimNum == 2 && outDimNum == 3 && outMDim == 1 && x2DimNum == 2) {
        matmulRet = l0op::Reshape(matmulRet, outShape, executor);
    }
    CHECK_RET(matmulRet != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckSupportSocVersion(bool isA4W4)
{
    SocVersion socVersion = GetCurrentPlatformInfo().GetSocVersion();
    NpuArch npuArch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    if (isA4W4) {
        // a4w4 support 910B 910_93 950，其余暂不支持
        switch (npuArch) {
            case NpuArch::DAV_2201:
            case NpuArch::DAV_3510:
                break;
            default: {
                OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "QuantBatchMatmul support for %s is not implemented in a4w4 scenario.",
                        op::ToString(socVersion).GetString());
                return ACLNN_ERR_RUNTIME_ERROR;
            }
        }
    } else {
        switch (npuArch) {
            case NpuArch::DAV_2201:
            case NpuArch::DAV_3510:
            case NpuArch::DAV_2002:
                break;
            default: {
                OP_LOGE(ACLNN_ERR_RUNTIME_ERROR, "QuantBatchMatmul support for %s is not implemented in a8w8 scenario.",
                        op::ToString(socVersion).GetString());
                return ACLNN_ERR_RUNTIME_ERROR;
            }
        }
    }
    return ACLNN_SUCCESS;
}

static const aclTensor* GetNDFormat(const aclTensor* input)
{
    const aclTensor* reformatedInput = input;
    if (input != nullptr) {
        reformatedInput = SetTensorToNDFormat(input);
    }
    return reformatedInput;
}

static aclTensor* ConvertTensorToInt4(const aclTensor* input, aclOpExecutor* executor)
{
    // 将int32的输入dtype修改为int4, 同时ViewShape和ViewStrides也从int32修改为int4所对应的。
    auto viewShape = input->GetViewShape();
    auto storageShape = input->GetStorageShape();
    auto viewShapeDim = viewShape.GetDimNum();
    viewShape[viewShapeDim - 1] = viewShape[viewShapeDim - 1] * INT4_NUMS_IN_INT32;
    auto inputTemp = executor->CreateView(input, viewShape, input->GetViewOffset());
    CHECK_RET(inputTemp != nullptr, nullptr);
    inputTemp->SetDataType(DataType::DT_INT4);
    if (input->GetStorageFormat() == op::Format::FORMAT_FRACTAL_NZ) {
        storageShape[storageShape.GetDimNum() - 1] = NZ_K0_VALUE_INT4_TRANS;
        storageShape[storageShape.GetDimNum() - MIN_DIM_NUM_NZ] = (viewShape[viewShapeDim - 1] +
                                                                   NZ_K0_VALUE_INT4_TRANS - 1) /
                                                                  NZ_K0_VALUE_INT4_TRANS;
        inputTemp->SetStorageShape(storageShape);
    }
    OP_LOGD("The conversion from int32 to int4 is completed.");
    return inputTemp;
}

static aclnnStatus InputPreProcessA4W4(const aclTensor*& x1, const aclTensor*& x2, bool& isA4W4,
                                       aclOpExecutor* executor)
{
    if (x1->GetDataType() == DataType::DT_INT32) {
        isA4W4 = true;
        x1 = ConvertTensorToInt4(x1, executor);
        CHECK_RET(x1 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (x2->GetDataType() == DataType::DT_INT32) {
        isA4W4 = true;
        x2 = ConvertTensorToInt4(x2, executor);
        CHECK_RET(x2 != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    isA4W4 = isA4W4 || (x1->GetDataType() == DataType::DT_INT4 || x2->GetDataType() == DataType::DT_INT4);
    return ACLNN_SUCCESS;
}

static aclnnStatus WeightNZCaseProcess(const aclTensor*& x2, bool& transposeX2, aclOpExecutor* executor)
{
    auto viewShape = x2->GetViewShape();
    auto viewShapeDim = viewShape.GetDimNum();
    bool isNotOneDim = viewShapeDim >= PENULTIMATE_DIM && viewShape[viewShapeDim - 1] != 1 &&
                       viewShape[viewShapeDim - PENULTIMATE_DIM] != 1;
    auto formatX2 = static_cast<ge::Format>(ge::GetPrimaryFormat(x2->GetStorageFormat()));
    // if plateform is not DAV3510 and weight is already in nz format, no need to set contiguous
    if (formatX2 != op::Format::FORMAT_FRACTAL_NZ ||
        (isNotOneDim && op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510)) {
        CHECK_RET(TensorContiguousProcess(x2, transposeX2, executor), ACLNN_ERR_INNER_NULLPTR);
    }
    if (static_cast<ge::Format>(ge::GetPrimaryFormat(x2->GetStorageFormat())) == op::Format::FORMAT_FRACTAL_NZ) {
        x2->SetOriginalShape(x2->GetViewShape());
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus A4W4CaseProcess(const aclTensor*& x1, const aclTensor*& x2, bool& isA4W4, aclOpExecutor* executor)
{
    CHECK_RET(InputPreProcessA4W4(x1, x2, isA4W4, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckSupportSocVersion(isA4W4) != ACLNN_ERR_RUNTIME_ERROR, ACLNN_ERR_RUNTIME_ERROR);
    return ACLNN_SUCCESS;
}

static aclnnStatus PostMatmulCalcProcess(const aclTensor* matmulRet, TupleTensor mandatoryTensors,
                                         aclOpExecutor* executor)
{
    CHECK_RET(matmulRet != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto out = std::get<INDEX_OUT_IN_TUPLE>(mandatoryTensors);
    CHECK_RET(SpecialOutputProcess(x1, x2, out, matmulRet, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(matmulRet, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
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
    int64_t lastThirdDim = mxScaleTensor->GetViewShape().GetDim(dimNum - 3); // 3: 倒数第3维
    if (mxScaleTensor->GetViewStrides()[dimNum - 3] == lastDim &&            // 3： 倒数第3维
        mxScaleTensor->GetViewStrides()[dimNum - PENULTIMATE_DIM] == lastDim * lastThirdDim) {
        int64_t tmpNxD = lastDim * lastSecondDim * lastThirdDim;
        transposeFlag = true;
        // 4：batch维度从倒数第4维起
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
        swapedShape.SetDim(dimNum - 3, lastSecondDim); // 3： 倒数第3维
        mxScaleTensor = executor->CreateView(mxScaleTensor, swapedShape, mxScaleTensor->GetViewOffset());
    } else {
        mxScaleTensor = l0op::Contiguous(mxScaleTensor, executor);
    }
    CHECK_RET(mxScaleTensor != nullptr, false);
    return true;
}

static aclnnStatus PreMatmulCalcProcess(TupleTensor& mandatoryTensors, TupleOptional& optionalTensors,
                                        TupleAttr& boolsTrans, bool& isA4W4, const aclTensor* out,
                                        aclOpExecutor* executor)
{
    auto& x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& perTokenScale = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    bool& transposeX1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(boolsTrans);
    bool& transposeX2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(boolsTrans);
    CHECK_RET(executor != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    CHECK_RET(CheckNotNull(std::tie(x1, x2, scale), out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(TensorContiguousProcess(x1, transposeX1, executor), ACLNN_ERR_INNER_NULLPTR);
    if (perTokenScale != nullptr) {
        if (IsMicroScaling(perTokenScale, scale)) {
            CHECK_RET(MxScaleContiguousProcess(scale, executor), ACLNN_ERR_INNER_NULLPTR);
            CHECK_RET(MxScaleContiguousProcess(perTokenScale, executor), ACLNN_ERR_INNER_NULLPTR);
        }
    }
    auto ret = WeightNZCaseProcess(x2, transposeX2, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (isA8W4Float(x1, x2)) {
        bool tempTransposeValue = false;
        CHECK_RET(CheckA8W4FloatQuantType(x1, x2, perTokenScale, scale), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(TensorContiguousProcess(perTokenScale, tempTransposeValue, executor), ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(TensorContiguousProcess(scale, tempTransposeValue, executor), ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(CheckInputAttrExistence(boolsTrans, mandatoryTensors, optionalTensors), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckDimRangeA8W4(mandatoryTensors, optionalTensors, out), ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckScaleDimRangeA8W4(mandatoryTensors, optionalTensors), ACLNN_ERR_PARAM_INVALID);
    } else {
        CHECK_RET(CheckDimRange(x1, x2, scale, out), ACLNN_ERR_PARAM_INVALID);
    }
    ret = A4W4CaseProcess(x1, x2, isA4W4, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    return ACLNN_SUCCESS;
}

static void GetDtypeAndTranspose(TupleTensor mandatoryTensors, int64_t& dtype, bool& transposeX1, bool& transposeX2)
{
    auto x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto out = std::get<INDEX_OUT_IN_TUPLE>(mandatoryTensors);
    dtype = static_cast<int64_t>(out->GetDataType());
    transposeX1 = GetTransposeAttrValue(x1, transposeX1, true);
    transposeX2 = GetTransposeAttrValue(x2, transposeX2, true);
    OP_LOGD("QuantMatmul attr transposeX1 is %d, transposeX2 is %d.", transposeX1, transposeX2);
}

static aclTensor* ProcessScaleTensor(const aclTensor* scale)
{
    auto castedScale = const_cast<aclTensor*>(scale);
    if (castedScale->GetDataType() == op::DataType::DT_INT64) {
        castedScale->SetDataType(op::DataType::DT_UINT64);
    }
    return castedScale;
}

static bool IsX1Transdata(const aclTensor* x1, const aclTensor* x2, int64_t dtype, bool transposeX1, bool transposeX2)
{
    if (transposeX1 == true || transposeX2 == true) {
        return false;
    }
    if (x1->GetStorageFormat() != op::Format::FORMAT_ND || x2->GetStorageFormat() != op::Format::FORMAT_FRACTAL_NZ) {
        return false;
    }
    if (x1->GetDataType() != op::DataType::DT_INT8) {
        return false;
    }
    if (dtype != static_cast<int>(op::DataType::DT_FLOAT16) && dtype != static_cast<int>(op::DataType::DT_BF16) &&
        dtype != static_cast<int>(op::DataType::DT_INT32)) {
        return false;
    }
    // innersize待校验
    Shape x1Shape = x1->GetViewShape();
    int64_t x1DimNum = x1Shape.GetDimNum();
    Shape x2Shape = x2->GetOriginalShape();
    int64_t x2DimNum = x2Shape.GetDimNum();
    if (x1DimNum != MIN_DIM_NUM_ND || x2DimNum != MIN_DIM_NUM_ND) {
        return false;
    }
    int64_t m = transposeX1 ? x1Shape.GetDim(x1DimNum - 1) : x1Shape.GetDim(x1DimNum - 2);
    int64_t k = transposeX1 ? x1Shape.GetDim(x1DimNum - 2) : x1Shape.GetDim(x1DimNum - 1);
    int64_t n = transposeX2 ? x2Shape.GetDim(x2DimNum - 2) : x2Shape.GetDim(x2DimNum - 1);
    int64_t innerSize = x1Shape.GetDim(x1DimNum - 1);
    // m校验
    bool isSupportedM = false;
    if ((m > M_RANGE1_LEFT && m <= M_RANGE1_RIGHT)) {
        isSupportedM = true;
    }
    if (innerSize % INNER_SIZE_MULTIPLE == 0 || !isSupportedM || k != K_VALUE || n != N_VALUE) {
        return false;
    }
    return true;
}

static void A8W4ProcessYScaleTensor(const aclTensor* x1Scale, const aclTensor* yScale)
{
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        // A8W4场景输入的INT64转为UINT64
        if (x1Scale == nullptr && yScale != nullptr && yScale->GetDataType() == op::DataType::DT_INT64) {
            auto castYScale = const_cast<aclTensor*>(yScale);
            castYScale->SetDataType(op::DataType::DT_UINT64);
            yScale = castYScale;
            OP_LOGD("The conversion from INT64 to UINT64 has been completed.");
        }
    }
}

static inline bool A8W4ValidGroupSize(uint64_t groupSizeM, uint64_t groupSizeN)
{
    return (groupSizeM == 0 && groupSizeN == 0) || (groupSizeM == 1 && groupSizeN == 1);
}

static inline bool A8W4InferGroupSize(int64_t& groupSize)
{
    uint64_t groupSizeK = static_cast<uint64_t>(groupSize) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeN = (static_cast<uint64_t>(groupSize) >> GROUP_N_OFFSET) & GROUP_MNK_BIT_SIZE;
    uint64_t groupSizeM = (static_cast<uint64_t>(groupSize) >> GROUP_M_OFFSET) & GROUP_MNK_BIT_SIZE;
    if (!A8W4ValidGroupSize(groupSizeM, groupSizeN)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("aclnnQuantMatmulWeightNzGetWorkspaceSize", "groupSizeM, groupSizeN",
                                               FormatString("%lu, %lu", groupSizeM, groupSizeN).c_str(),
                                               "groupSizeM and groupSizeN must both be 0 or 1");
        return false;
    }

    OP_LOGD("A8W4 after Inferred groupSize: groupSizeM: %lu, groupSizeN: %lu, groupSizeK: %lu.", groupSizeM, groupSizeN,
            groupSizeK);
    groupSize = groupSizeK;
    return true;
}

static aclnnStatus TensorPreProcess(const aclTensor* x1, const aclTensor* x2, const aclTensor* x1Scale,
                                    const aclTensor* yScale, int64_t& groupSize)
{
    bool isA8W4F = isA8W4Float(x1, x2);
    if (isA8W4F) {
        A8W4ProcessYScaleTensor(x1Scale, yScale);
        CHECK_RET(A8W4InferGroupSize(groupSize), ACLNN_ERR_PARAM_INVALID);
        OP_LOGD("Infer groupSize success. groupSize: %ld.", groupSize);
    }

    return ACLNN_SUCCESS;
}

static aclnnStatus SetReformtedX2(const aclTensor*& reformatedX1, const aclTensor*& reformatedX2, bool& transposeX1,
                                  bool& transposeX2, aclOpExecutor* executor)
{
    if (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P) {
        auto ret = TransposeAndTransDataForInputs(reformatedX1, reformatedX2, transposeX1, transposeX2, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    } else {
        reformatedX2 = SetTensorToNDFormat(reformatedX2);
    }
    return ACLNN_SUCCESS;
}

static inline aclnnStatus TransdataX1Process(bool isX1TransdataFlag, const aclTensor*& reformatedX1,
                                             aclOpExecutor* executor, bool isPpMatmul)
{
    auto socLongVersion = GetCurrentPlatformInfo().GetSocLongVersion();
    bool checkSocLongVersion = (socLongVersion == "Ascend910B3" || socLongVersion == "Ascend910B4" ||
                                socLongVersion == "Ascend910B4-1");
    auto coreNum = static_cast<int32_t>(GetCurrentPlatformInfo().GetCubeCoreNum());
    if ((isX1TransdataFlag && checkSocLongVersion && coreNum == CORE_NUM_20) || isPpMatmul) {
        auto ret = TransdataForX1(reformatedX1, executor);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
    }

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnQuantMatmulGetWorkspaceSizeCommonProcess(TupleTensor mandatoryTensors, TupleOptional optionalTensors,
                                                          TupleAttr boolsTrans, const aclTensor* out,
                                                          aclOpExecutor* executor, const char* apiName)
{
    auto& x1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& x2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& scale = std::get<INDEX_SCALE_IN_MANDTORY_TUPLE>(mandatoryTensors);
    auto& offset = std::get<INDEX_OFFSET_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto& pertokenScaleOptional = std::get<INDEX_PERTOKEN_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto& bias = std::get<INDEX_BIAS_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto& yScale = std::get<INDEX_Y_SCALE_IN_OPTIONAL_TUPLE>(optionalTensors);
    auto& yOffset = std::get<INDEX_Y_OFFSET_IN_OPTIONAL_TUPLE>(optionalTensors);
    int64_t groupSize = std::get<INDEX_GROUP_SIZE_IN_OPTIONAL_TUPLE>(optionalTensors);
    bool& transposeX1 = std::get<INDEX_X1_IN_MANDTORY_TUPLE>(boolsTrans);
    bool& transposeX2 = std::get<INDEX_X2_IN_MANDTORY_TUPLE>(boolsTrans);
    CHECK_RET(CheckNotNull(mandatoryTensors, out), ACLNN_ERR_PARAM_NULLPTR);
    bool isA8W4F = isA8W4Float(x1, x2);
    bool isA8W4I = isA8W4Int(x1, x2);
    bool isPseudoQuant = isA8W4F || isA8W4I;
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510 && !isPseudoQuant) {
        auto x1DimNum = x1->GetViewShape().GetDimNum();
        auto x2DimNum = x2->GetViewShape().GetDimNum();
        if (x1DimNum >= PENULTIMATE_DIM && x2DimNum >= PENULTIMATE_DIM) {
            auto inputSizeM = transposeX1 ? x1->GetViewShape().GetDim(x1DimNum - 1) :
                                            x1->GetViewShape().GetDim(x1DimNum - PENULTIMATE_DIM);
            auto inputSizeN = transposeX2 ? x2->GetViewShape().GetDim(x2DimNum - PENULTIMATE_DIM) :
                                            x2->GetViewShape().GetDim(x2DimNum - 1);
            if (static_cast<ge::Format>(ge::GetPrimaryFormat(x2->GetStorageFormat())) == Format::FORMAT_FRACTAL_NZ) {
                if (inputSizeM == 0) {
                    OP_LOGD("aclnnV4 nz m=0");
                    return ACLNN_SUCCESS;
                }
            } else {
                if (inputSizeM == 0 || inputSizeN == 0) {
                    OP_LOGD("aclnnV4 nd m/n=0");
                    return ACLNN_SUCCESS;
                }
            }
        }
    }
    auto ret = TensorPreProcess(x1, x2, pertokenScaleOptional, yScale, groupSize);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    bool isA4W4 = false;
    ret = PreMatmulCalcProcess(mandatoryTensors, optionalTensors, boolsTrans, isA4W4, out, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    bool biasTransposeValue = false;
    CHECK_RET(TensorContiguousProcess(bias, biasTransposeValue, executor), ACLNN_ERR_INNER_NULLPTR);
    bool scaleTransposeValue = false;
    CHECK_RET(TensorContiguousProcess(scale, scaleTransposeValue, executor), ACLNN_ERR_INNER_NULLPTR);
    bool offsetTransposeValue = false;
    CHECK_RET(TensorContiguousProcess(offset, offsetTransposeValue, executor), ACLNN_ERR_INNER_NULLPTR);
    bool perTokenScaleTransposeValue = false;
    CHECK_RET(TensorContiguousProcess(pertokenScaleOptional, perTokenScaleTransposeValue, executor),
              ACLNN_ERR_INNER_NULLPTR);
    auto reformatedX1 = SetTensorToNDFormat(x1);
    const aclTensor* reformatedX2 = x2;
    ret = SetReformtedX2(reformatedX1, reformatedX2, transposeX1, transposeX2, executor);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const aclTensor* reformatedScale = GetNDFormat(scale);
    const aclTensor* reformatedpertokenScaleOptional = GetNDFormat(pertokenScaleOptional);
    const aclTensor* reformatedBias = GetNDFormat(bias);
    const aclTensor* reformatedYScale = GetNDFormat(yScale);

    ret = CheckParams(
        std::tie(reformatedX1, reformatedX2, reformatedScale),
        std::tie(offset, reformatedpertokenScaleOptional, reformatedBias, reformatedYScale, yOffset, groupSize),
        std::tie(transposeX1, transposeX2), isA4W4, out, apiName);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    auto castedScale = ProcessScaleTensor(reformatedScale);
    int64_t dtype = 0;
    GetDtypeAndTranspose(std::tie(reformatedX1, reformatedX2, out), dtype, transposeX1, transposeX2);
    bool isX1TransdataFlag = IsX1Transdata(reformatedX1, reformatedX2, dtype, transposeX1, transposeX2);
    auto inputAShape = reformatedX1->GetViewShape();
    uint32_t M = inputAShape.GetDimNum() == NO_BATCH_DIM_SUM ? inputAShape[0] : inputAShape[1];
    auto socLongVersion = GetCurrentPlatformInfo().GetSocLongVersion();
    bool isPpMatmul = (GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P &&
                       ((M >= PPMATMUL_PRIORITY_M && bias != nullptr && !transposeX1 && transposeX2 &&
                         dtype != DataType::DT_BF16) ||
                        (pertokenScaleOptional != nullptr && !pertokenScaleOptional->IsEmpty())));
    ret = TransdataX1Process(isX1TransdataFlag, reformatedX1, executor, isPpMatmul);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const aclTensor* matmulRet = nullptr;
    if (isA8W4F || isA8W4I) {
        // 调用l0算子QuantBatchMatmulV4进行计算
        matmulRet = l0op::QuantBatchMatmulV4(
            reformatedX1, reformatedX2, reformatedBias, reformatedpertokenScaleOptional, castedScale, reformatedYScale,
            nullptr, nullptr, yOffset, nullptr, dtype, -1, transposeX1, transposeX2, groupSize, executor);
    } else {
        // 调用l0算子QuantBatchMatmulV3进行计算
        matmulRet = l0op::QuantBatchMatmulV3(reformatedX1, reformatedX2, castedScale, offset, reformatedBias,
                                             reformatedpertokenScaleOptional, dtype, transposeX1, transposeX2,
                                             groupSize, executor);
    }

    if (isPpMatmul) {
        CHECK_RET(matmulRet != nullptr, ACLNN_ERR_INNER_NULLPTR);
        const aclTensor* matmulNdRet = nullptr;
        matmulNdRet = l0op::TransData(matmulRet, Format::FORMAT_ND, 0, executor);

        CHECK_RET(PostMatmulCalcProcess(matmulNdRet, std::tie(x1, x2, out), executor) == ACLNN_SUCCESS, ret);
    } else {
        CHECK_RET(PostMatmulCalcProcess(matmulRet, std::tie(x1, x2, out), executor) == ACLNN_SUCCESS, ret);
    }
    return ACLNN_SUCCESS;
}
} // namespace internal
} // namespace quant_matmul_v4
