/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file quant_matmul_common_check.cpp
 * \brief
 */
#include "quant_matmul_common_check.h"

using namespace op;
using namespace ge;
using Ops::NN::SwapLastTwoDimValue;
using Ops::NN::IsTransposeLastTwoDims;
using Ops::Base::CeilDiv;

bool CheckSpecialCase(const aclTensor *tensor, int64_t firstLastDim, int64_t secondLastDim) {
    if (tensor->GetViewShape().GetDim(firstLastDim) == tensor->GetViewShape().GetDim(secondLastDim)) {
        OP_LOGD("QuantMatmul special case, no need to set transpose attr value.");
        return true;
    }
    return false;
}

bool GetTransposeAttrValue(const aclTensor *tensor, bool transpose) {
    int64_t dim1 = tensor->GetViewShape().GetDimNum() - 1;
    int64_t dim2 = tensor->GetViewShape().GetDimNum() - PENULTIMATE_DIM;
    // check if tensor is contiguous layout
    if (tensor->GetViewStrides()[dim2] == 1 && tensor->GetViewStrides()[dim1] == tensor->GetViewShape().GetDim(dim2)) {
        OP_LOGD("QuantMatmul GetTransposeAttrValue, find tensor is not contiguous.");
        const_cast<aclTensor *>(tensor)->SetViewShape(SwapLastTwoDimValue(tensor->GetViewShape()));
        if (!CheckSpecialCase(tensor, dim1, dim2)) {
            return !transpose;
        }
    }
    return transpose;
}

op::Shape GetWeightNzShape(const aclTensor *input, bool transpose)
{
    size_t viewDimNum = input->GetViewShape().GetDimNum();
    int64_t k = transpose ? input->GetViewShape().GetDim(viewDimNum - 1)
                           : input->GetViewShape().GetDim(viewDimNum - LAST_SECOND_DIM_INDEX);
    int64_t n = transpose ? input->GetViewShape().GetDim(viewDimNum - LAST_SECOND_DIM_INDEX)
                           : input->GetViewShape().GetDim(viewDimNum - 1);

    int64_t nz_k0_value_trans = (input->GetDataType() == op::DataType::DT_INT32 || input->GetDataType() == op::DataType::DT_INT4) ? 
        NZ_K0_VALUE_INT4_TRANS : NZ_K0_VALUE_INT8_TRANS;
    int64_t k1 = transpose ? CeilDiv(k, nz_k0_value_trans) : CeilDiv(k, NZ_K0_VALUE_INT8_INT4);
    int64_t n1 = transpose ? CeilDiv(n, NZ_K0_VALUE_INT8_INT4) : CeilDiv(n, nz_k0_value_trans);

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

bool CheckWeightNzStorageShape(const op::Shape &nzShape, const op::Shape &storageShape)
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

const aclTensor *SetTensorToNZFormat(const aclTensor *input, op::Shape &shape, aclOpExecutor *executor)
{
    auto formatTensor = executor->CreateView(input, shape, input->GetViewOffset());
    formatTensor->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
    formatTensor->SetOriginalFormat(op::Format::FORMAT_ND);
    formatTensor->SetViewShape(input->GetViewShape());
    return formatTensor;
}

// 二维及以上的tensor都需要调
inline bool TensorContiguousProcess(const aclTensor *&contiguousTensor, bool &transpose,
                                           aclOpExecutor *executor) {
    if (contiguousTensor == nullptr || contiguousTensor->GetViewShape().GetDimNum() == 1) {
        OP_LOGD("QuantMatmul no need to do contiguous process.");
        return true;
    }
    auto transposeFlag = IsTransposeLastTwoDims(contiguousTensor);
    // swap tensor if its viewshape not satisfy request shape without adding a transpose node
    if (transposeFlag) {
        contiguousTensor = executor->CreateView(contiguousTensor, SwapLastTwoDimValue(contiguousTensor->GetViewShape()),
                                                contiguousTensor->GetViewOffset());
        transpose = !transpose;
    } else {
        contiguousTensor = l0op::Contiguous(contiguousTensor, executor);
    }
    CHECK_RET(contiguousTensor != nullptr, false);
    return true;
}

aclnnStatus WeightNZCaseProcess(const aclTensor *&x2, bool &transposeX2, aclOpExecutor *executor) {
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

aclTensor* ConvertTensorToInt4(const aclTensor* input, aclOpExecutor* executor)
{
    // 将int32的输入dtype修改为int4, 同时ViewShape和ViewStrides也从int32修改为int4所对应的。
    auto viewShape = input->GetViewShape();
    viewShape[viewShape.GetDimNum() - 1] = viewShape[viewShape.GetDimNum() - 1] * INT4_NUMS_IN_INT32;
    auto inputTemp = executor->CreateView(input, viewShape, input->GetViewOffset());
    inputTemp->SetDataType(DataType::DT_INT4);
    OP_LOGD("The conversion from int32 to int4 is completed.");
    return inputTemp;
}

void InputPreProcessA4W4(const aclTensor *&x1, const aclTensor *&x2, aclOpExecutor *executor)
{
    if (x2->GetDataType() == DataType::DT_INT32) {
        x2 = ConvertTensorToInt4(x2, executor);
    }
    if (x1->GetDataType() == DataType::DT_INT32) {
        x1 = ConvertTensorToInt4(x1, executor);
    }
}

aclnnStatus A4W4CaseProcess(const aclTensor *&x1, const aclTensor *&x2, aclOpExecutor *executor) {
    InputPreProcessA4W4(x1, x2, executor);
    return ACLNN_SUCCESS;
}

const aclTensor* SetTensorToNDFormat(const aclTensor *input) {
    OP_LOGD("QuantMatmul set tensor to ND format.");
    const aclTensor* output = nullptr;
    if (input == nullptr) {
        return output;
    }
    if (input->GetStorageFormat() != Format::FORMAT_FRACTAL_NZ) {
        output = l0op::ReFormat(input, op::Format::FORMAT_ND);
    } else {
        output = input;
    }
    return output;
}

const aclTensor* GetNDFormat(const aclTensor *input) {
    const aclTensor* reformatedInput = input;
    if (input != nullptr) {
        reformatedInput = SetTensorToNDFormat(input);
    }
    return reformatedInput;
}

void GetDtypeAndTranspose(TupleTensor mandatoryTensors, int64_t &dtype, bool &transposeX1,
                                 bool &transposeX2) {
    auto x1 = std::get<0>(mandatoryTensors);
    auto x2 = std::get<1>(mandatoryTensors);
    auto out = std::get<INDEX_OUT_IN_TUPLE>(mandatoryTensors);
    dtype = static_cast<int64_t> (out->GetDataType());
    transposeX1 = GetTransposeAttrValue(x1, transposeX1);
    transposeX2 = GetTransposeAttrValue(x2, transposeX2);
    OP_LOGD("QuantMatmul attr transposeX1 is %d, transposeX2 is %d.", transposeX1, transposeX2);
}

aclnnStatus SetSpecilNZTensorToNormalNZFormat(const aclTensor *&input, aclOpExecutor *executor) {
    OP_LOGD("QuantMatmulV4 set special NZ format to normal NZ format.");
    auto nzTensorTmp =  executor->CreateView(input, input->GetViewShape(), input->GetViewOffset());
    CHECK_RET(nzTensorTmp != nullptr, ACLNN_ERR_INNER_NULLPTR);
    nzTensorTmp->SetViewFormat(op::Format::FORMAT_ND);
    nzTensorTmp->SetOriginalFormat(op::Format::FORMAT_ND);
    nzTensorTmp->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
    nzTensorTmp->SetStorageShape(input->GetStorageShape());
    nzTensorTmp->SetOriginalShape(input->GetOriginalShape());
    input = nzTensorTmp;
    return ACLNN_SUCCESS;
}

aclnnStatus SpecialOutputProcess(const aclTensor *x1, const aclTensor *x2, const aclTensor *out,
                                        const aclTensor *&matmulRet, aclOpExecutor* executor) {
    // we have to reshape for case which x1 and x2 are 2 dims and out is 3 dims, otherwise, viewcopy will fail
    OP_LOGD("QuantMatmul enter SpecialOutputProcess func.");
    auto outShape = out->GetViewShape();
    auto outDimNum = outShape.GetDimNum();
    int64_t outMDim = outShape.GetDim(outDimNum - 2);
    auto x1DimNum = x1->GetViewShape().GetDimNum();
    auto x2DimNum = x2->GetViewShape().GetDimNum();
    // speical case : x1 and x2 are 2 dim, output is 3 dim, have to reshape matmul result, otherwise viewcopy will fail.
    if (x1DimNum == 2 && x2DimNum == 2 && outDimNum == 3 && outMDim == 1) {
        matmulRet = l0op::Reshape(matmulRet, outShape, executor);
    }
    CHECK_RET(matmulRet != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus PostMatmulCalcProcess(const aclTensor *matmulRet, const aclTensor *x1, const aclTensor *x2,
                                         const aclTensor *out, aclOpExecutor *executor) {
    CHECK_RET(matmulRet != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(SpecialOutputProcess(x1, x2, out, matmulRet, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    // 如果出参out是非连续Tensor，需要把计算完的连续Tensor转非连续
    auto viewCopyResult = l0op::ViewCopy(matmulRet, out, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}


