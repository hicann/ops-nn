/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_matmul_activation_quant_checker.cpp
 * \brief Input layout and contiguous-tensor handling for the ACLNN interfaces.
 */
#include "quant_matmul_activation_quant_checker.h"
#include "aclnn_kernels/transdata.h"

namespace QuantMatmulActivationQuantAclnnCheck {

using namespace op;
using namespace ge;
using Ops::Base::CeilDiv;
using Ops::NN::IsTransposeLastTwoDims;
using Ops::NN::SwapLastTwoDimValue;

bool CheckSpecialCase(const aclTensor* tensor, int64_t firstLastDim, int64_t secondLastDim)
{
    if (tensor->GetViewShape().GetDim(firstLastDim) == tensor->GetViewShape().GetDim(secondLastDim)) {
        OP_LOGD("Special case: transpose attribute does not need to be set.");
        return true;
    }
    return false;
}

bool GetTransposeAttrValue(const aclTensor* tensor, bool transpose)
{
    int64_t dim1 = tensor->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = tensor->GetViewShape().GetDimNum() - PENULTIMATE_DIM;
    // check if tensor is contiguous layout
    if (tensor->GetViewStrides()[dim2] == 1 && tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2)) {
        OP_LOGD("Detected a transposed/non-contiguous tensor layout; swapping the last two dimensions.");
        const_cast<aclTensor*>(tensor)->SetViewShape(SwapLastTwoDimValue(tensor->GetViewShape()));
        if (!CheckSpecialCase(tensor, dim1, dim2)) {
            return !transpose;
        }
    }
    return transpose;
}

op::Shape GetWeightNzShape(const aclTensor* input, bool transpose)
{
    size_t viewDimNum = input->GetViewShape().GetDimNum();
    int64_t k = transpose ? input->GetViewShape().GetDim(viewDimNum - LAST_FIRST_DIM_INDEX) :
                            input->GetViewShape().GetDim(viewDimNum - LAST_SECOND_DIM_INDEX);
    int64_t n = transpose ? input->GetViewShape().GetDim(viewDimNum - LAST_SECOND_DIM_INDEX) :
                            input->GetViewShape().GetDim(viewDimNum - LAST_FIRST_DIM_INDEX);

    bool isMXFP4 = input->GetDataType() == DataType::DT_FLOAT4_E2M1;
    int64_t nz_k0_value_trans = isMXFP4 ? NZ_K0_VALUE_INT4_TRANS : NZ_K0_VALUE_INT8_TRANS;
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

bool CheckWeightNzStorageShape(const op::Shape& nzShape, const op::Shape& storageShape)
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

const aclTensor* SetTensorToNZFormat(const aclTensor* input, op::Shape& shape, aclOpExecutor* executor)
{
    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "SetTensorToNZFormat failed: executor is null.");
        return nullptr;
    }
    auto formatTensor = executor->CreateView(input, shape, input->GetViewOffset());
    if (formatTensor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "SetTensorToNZFormat failed: unable to create the formatted tensor.");
        return nullptr;
    }
    formatTensor->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
    formatTensor->SetOriginalFormat(op::Format::FORMAT_ND);
    formatTensor->SetViewShape(input->GetViewShape());
    return formatTensor;
}

bool TensorContiguousProcess(const aclTensor*& contiguousTensor, bool& transpose, aclOpExecutor* executor)
{
    if (contiguousTensor == nullptr) {
        OP_LOGD("Input tensor is null; skipping contiguous conversion.");
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
        CHECK_RET(contiguousTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        transpose = !transpose;
    } else {
        contiguousTensor = l0op::Contiguous(contiguousTensor, executor);
    }
    CHECK_RET(contiguousTensor != nullptr, false);
    if (isNZTensor) {
        contiguousTensor->SetStorageShape(storageShape); // 对NZ的场景需要用原NZshape刷新
        contiguousTensor->SetOriginalShape(storageShape);
    }
    return true;
}

aclnnStatus WeightNZCaseProcess(const aclTensor*& x2, bool& transposeX2, aclOpExecutor* executor)
{
    // if weight is already in nz format, no need to set contiguous
    if (ge::GetPrimaryFormat(x2->GetStorageFormat()) == op::Format::FORMAT_FRACTAL_NZ ||
        ge::GetPrimaryFormat(x2->GetStorageFormat()) == op::Format::FORMAT_FRACTAL_NZ_C0_32) {
        x2->SetOriginalShape(x2->GetViewShape());
        if (ge::GetPrimaryFormat(x2->GetStorageFormat()) == op::Format::FORMAT_FRACTAL_NZ_C0_32) {
            CHECK_RET(SetSpecilNZTensorToNormalNZFormat(x2, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
        }
    } else {
        CHECK_RET(TensorContiguousProcess(x2, transposeX2, executor), ACLNN_ERR_INNER_NULLPTR);
    }
    return ACLNN_SUCCESS;
}

const aclTensor* SetTensorToNDFormat(const aclTensor* input)
{
    const aclTensor* output = nullptr;
    if (input == nullptr) {
        return output;
    }
    if (input->GetStorageFormat() != Format::FORMAT_FRACTAL_NZ) {
        OP_LOGD("Converting input tensor to ND format.");
        output = l0op::ReFormat(input, op::Format::FORMAT_ND);
    } else {
        OP_LOGD("Input tensor already has the required storage format; no conversion needed.");
        output = input;
    }
    return output;
}

aclnnStatus SetSpecilNZTensorToNormalNZFormat(const aclTensor*& input, aclOpExecutor* executor)
{
    OP_LOGD("Converting special NZ format to standard NZ format.");
    auto nzTensorTmp = executor->CreateView(input, input->GetViewShape(), input->GetViewOffset());
    CHECK_RET(nzTensorTmp != nullptr, ACLNN_ERR_INNER_NULLPTR);
    nzTensorTmp->SetViewFormat(op::Format::FORMAT_ND);
    nzTensorTmp->SetOriginalFormat(op::Format::FORMAT_ND);
    nzTensorTmp->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
    nzTensorTmp->SetStorageShape(input->GetStorageShape());
    nzTensorTmp->SetOriginalShape(input->GetOriginalShape());
    input = nzTensorTmp;
    return ACLNN_SUCCESS;
}

} // namespace QuantMatmulActivationQuantAclnnCheck
