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
#include "opdev/tensor_view_utils.h"

namespace QuantMatmulActivationQuantAclnnCheck {

using namespace op;
using Ops::Base::CeilDiv;
using Ops::NN::IsTransposeLastTwoDims;
using Ops::NN::SwapLastTwoDimValue;

op::Shape GetWeightNzShape(const aclTensor* input, bool transpose)
{
    const size_t viewRank = input->GetViewShape().GetDimNum();
    const int64_t k = transpose ? input->GetViewShape().GetDim(viewRank - LAST_FIRST_DIM_INDEX) :
                                  input->GetViewShape().GetDim(viewRank - LAST_SECOND_DIM_INDEX);
    const int64_t n = transpose ? input->GetViewShape().GetDim(viewRank - LAST_SECOND_DIM_INDEX) :
                                  input->GetViewShape().GetDim(viewRank - LAST_FIRST_DIM_INDEX);

    const bool isMxFp4 = input->GetDataType() == DataType::DT_FLOAT4_E2M1;
    const int64_t innerBlockSize = isMxFp4 ? NZ_K0_VALUE_INT4_TRANS : NZ_K0_VALUE_INT8_TRANS;
    const int64_t kBlockCount = transpose ? CeilDiv(k, innerBlockSize) : CeilDiv(k, NZ_K0_VALUE_BMM_BLOCK_NUM);
    const int64_t nBlockCount = transpose ? CeilDiv(n, NZ_K0_VALUE_BMM_BLOCK_NUM) : CeilDiv(n, innerBlockSize);

    op::Shape weightNzShape;
    for (size_t i = 0; i < viewRank - LAST_SECOND_DIM_INDEX; i++) {
        weightNzShape.AppendDim(input->GetViewShape().GetDim(i));
    }
    if (transpose) {
        weightNzShape.AppendDim(kBlockCount);
        weightNzShape.AppendDim(nBlockCount);
    } else {
        weightNzShape.AppendDim(nBlockCount);
        weightNzShape.AppendDim(kBlockCount);
    }
    weightNzShape.AppendDim(NZ_STORAGE_PENULTIMATE_DIM);
    weightNzShape.AppendDim(innerBlockSize);
    return weightNzShape;
}

bool CheckWeightNzStorageShape(const op::Shape& nzShape, const op::Shape& storageShape)
{
    if (nzShape.GetDimNum() != storageShape.GetDimNum()) {
        return false;
    }
    for (size_t i = 0; i < nzShape.GetDimNum(); ++i) {
        if (nzShape.GetDim(i) != storageShape.GetDim(i)) {
            return false;
        }
    }
    return true;
}

bool NormalizeWeightNzView(const aclTensor*& input, bool& transpose, aclOpExecutor* executor)
{
    CHECK_RET(input != nullptr && executor != nullptr, false);
    // NZ storage must describe the entire padded matrix starting at its base address.
    CHECK_RET(input->GetViewOffset() == 0, false);
    auto shape = input->GetViewShape();
    auto strides = input->GetViewStrides();
    CHECK_RET(shape.GetDimNum() >= LAST_SECOND_DIM_INDEX && strides.size() == shape.GetDimNum(), false);
    const auto last = shape.GetDimNum() - LAST_FIRST_DIM_INDEX;
    const auto penultimate = shape.GetDimNum() - LAST_SECOND_DIM_INDEX;
    // NZ batch strides may include padding; only the matrix view strides determine transpose.
    const bool transposeView = strides[penultimate] == 1 && strides[last] == shape.GetDim(penultimate) &&
                               !(shape.GetDim(last) == 1 && shape.GetDim(penultimate) == 1);
    if (transposeView) {
        shape = SwapLastTwoDimValue(shape);
        std::swap(strides[last], strides[penultimate]);
    }
    auto formatTensor = executor->CreateView(input, input->GetStorageShape(), 0);
    CHECK_RET(formatTensor != nullptr, false);
    formatTensor->SetViewShape(shape);
    formatTensor->SetViewStrides(strides);
    formatTensor->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
    formatTensor->SetViewFormat(op::Format::FORMAT_ND);
    formatTensor->SetOriginalFormat(op::Format::FORMAT_ND);
    formatTensor->SetOriginalShape(shape);
    input = formatTensor;
    transpose = transpose != transposeView;
    return true;
}

bool TensorContiguousProcess(const aclTensor*& contiguousTensor, bool& transpose, aclOpExecutor* executor)
{
    if (contiguousTensor == nullptr) {
        OP_LOGD("Input tensor is null; skipping contiguous conversion.");
        return true;
    }
    CHECK_RET(executor != nullptr, false);
    const bool isNzTensor = static_cast<ge::Format>(ge::GetPrimaryFormat(contiguousTensor->GetStorageFormat())) ==
                            op::Format::FORMAT_FRACTAL_NZ;
    const auto storageShape = contiguousTensor->GetStorageShape();
    // A singleton-axis ND view may be both contiguous and transposed. Keep an untransposed
    // contiguous view, while still allowing an existing transpose attribute to be eliminated.
    const bool transposeFlag = IsTransposeLastTwoDims(contiguousTensor) &&
                               (isNzTensor || transpose || !op::IsContiguous(contiguousTensor));
    // swap tensor if its viewshape not satisfy request shape without adding a transpose node
    if (transposeFlag) {
        contiguousTensor = executor->CreateView(contiguousTensor, SwapLastTwoDimValue(contiguousTensor->GetViewShape()),
                                                contiguousTensor->GetViewOffset());
        CHECK_RET(contiguousTensor != nullptr, false);
        transpose = !transpose;
    } else {
        contiguousTensor = l0op::Contiguous(contiguousTensor, executor);
    }
    CHECK_RET(contiguousTensor != nullptr, false);
    if (isNzTensor) {
        contiguousTensor->SetStorageShape(storageShape); // 对NZ的场景需要用原NZshape刷新
        contiguousTensor->SetOriginalShape(storageShape);
    }
    return true;
}

const aclTensor* SetTensorToNDFormat(const aclTensor* input)
{
    if (input == nullptr) {
        return nullptr;
    }
    if (input->GetStorageFormat() != Format::FORMAT_FRACTAL_NZ) {
        OP_LOGD("Converting input tensor to ND format.");
        return l0op::ReFormat(input, op::Format::FORMAT_ND);
    }
    OP_LOGD("Preserving FRACTAL_NZ storage; skipping ND conversion.");
    return input;
}

} // namespace QuantMatmulActivationQuantAclnnCheck
