/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <string>
#include "aclnn_common.h"
#include "op_common/op_host/util/math_util.h"

namespace {
constexpr int64_t ALIGN_NUM = 2;
constexpr int64_t BLOCK_SIZE = 32;
constexpr int64_t DEFAULT_SCALE_ALG = 0;
static const size_t OFFSET_32_BITS = 32;
static const size_t OFFSET_16_BITS = 16;
static const uint64_t GROUP_MAX = 65535UL;
static const size_t GROUP_DIM = 3;

int64_t CheckAndGetGroupSize(at::IntArrayRef groupSizes)
{
    int64_t groups = 0;
    if (groupSizes.empty()) {
        return groups;
    }
    size_t groupDim = groupSizes.size();
    TORCH_CHECK(groupDim == GROUP_DIM, "group_sizes only support input with three elements, but got ", groupDim);
    int64_t groupM = static_cast<int64_t>(groupSizes[0]);
    int64_t groupN = static_cast<int64_t>(groupSizes[1]);
    int64_t groupK = static_cast<int64_t>(groupSizes[2]);
    bool isValidGroupParam = ((groupM <= GROUP_MAX && groupM >= 0) && (groupN <= GROUP_MAX && groupN >= 0) &&
                              (groupK <= GROUP_MAX && groupK >= 0));
    TORCH_CHECK(isValidGroupParam, "group param value must conform to range [0, 65535]");
    groups = static_cast<int64_t>((static_cast<uint64_t>(groupM) << OFFSET_32_BITS) +
                                  (static_cast<uint64_t>(groupN) << OFFSET_16_BITS) + static_cast<uint64_t>(groupK));
    return groups;
}

static bool IsTransposeLastTwoDims(const at::Tensor& tensor)
{
    if (tensor.dim() < 2 || tensor.dim() > 6) {
        return false;
    }
    int64_t dim1 = tensor.dim() - 1;
    int64_t dim2 = tensor.dim() - 2;
    if (tensor.stride(dim2) == 1 && tensor.stride(dim1) == tensor.size(dim2)) {
        int64_t tmpNxD = tensor.size(dim1) * tensor.size(dim2);
        for (int64_t batchDim = tensor.dim() - 3; batchDim >= 0; batchDim--) {
            if (tensor.stride(batchDim) != tmpNxD) {
                return false;
            }
            tmpNxD *= tensor.size(batchDim);
        }
        if (tensor.size(dim1) == 1 && tensor.size(dim2) == 1) {
            return false;
        }
        return true;
    }
    return false;
}

} // namespace

namespace cann_ops_nn {
namespace matmul {

std::tuple<at::Tensor, at::Tensor> QuantMatmulActivationQuant(
    const at::Tensor& x1, const at::Tensor& x2, const at::Tensor& x2Scale, const c10::optional<at::Tensor>& x1Scale,
    const c10::optional<at::Tensor>& bias, c10::optional<int64_t> outputDtype, c10::optional<int64_t> x1Dtype,
    c10::optional<int64_t> x2Dtype, c10::optional<int64_t> x1ScaleDtype, c10::optional<int64_t> x2ScaleDtype,
    c10::optional<std::vector<int64_t>>& groupSizes, c10::string_view activationType, c10::string_view quantMode,
    c10::string_view roundMode, c10::optional<int64_t> scaleAlg, double dstTypeMax, const c10::optional<at::Tensor>& x3)
{
    auto x1DimNum = x1.dim();
    auto x2DimNum = x2.dim();
    // Keep only the preconditions needed to inspect shapes and construct
    // output tensors here; ACLNN/tiling owns the operator contract checks.
    TORCH_CHECK(x1DimNum >= 2 && x2DimNum >= 2, "x1 and x2 must have at least two dimensions");
    TORCH_CHECK(x1.device().type() == at::kPrivateUse1 && x2.device() == x1.device() && x2Scale.device() == x1.device(),
                "x1, x2 and x2_scale must be on the same NPU device");
    TORCH_CHECK(!x1Scale.has_value() || !x1Scale->defined() || x1Scale->device() == x1.device(),
                "x1_scale must be on the same NPU device as x1");
    TORCH_CHECK(!bias.has_value() || !bias->defined() || bias->device() == x1.device(),
                "bias must be on the same NPU device as x1");

    TORCH_CHECK(!x3.has_value() || !x3->defined() || x3->device() == x1.device(),
                "x3 must be on the same NPU device as x1");
    TORCH_CHECK(!x3.has_value() || !x3->defined() || (x1DimNum == 2 && x2DimNum == 2 && x3->dim() == 2),
                "x1, x2 and x3 must have 2D logical shapes when x3 is provided; batch is not supported");
    int64_t x1Rows = x1.size(x1DimNum - 2);
    int64_t x2Cols = x2.size(x2DimNum - 1);

    int64_t m = 0;
    int64_t n = 0;
    // The Torch API keeps canonical (M,K)/(K,N) view shapes.  Leave the
    // explicit ACLNN flags at false; its checker derives transpose views from
    // the last-two strides (and normalizes the view before tiling).
    bool transposeX1 = false;
    bool transposeX2 = false;

    constexpr int64_t dtypeFloat4E2M1 = static_cast<int64_t>(DType::FLOAT4_E2M1);

    int64_t x1DtypeValue = x1Dtype.value_or(0);
    int64_t x2DtypeValue = x2Dtype.value_or(0);
    bool isMxFp4 = x1DtypeValue == dtypeFloat4E2M1 && x2DtypeValue == dtypeFloat4E2M1;

    if (isMxFp4) {
        bool transX1 = IsTransposeLastTwoDims(x1);
        bool transX2 = IsTransposeLastTwoDims(x2);
        m = !transX1 ? x1Rows : x1Rows * FP4_IN_INT8;
        n = transX2 ? x2Cols : x2Cols * FP4_IN_INT8;
    } else {
        m = x1Rows;
        n = x2Cols;
    }

    if (activationType == "swiglu") {
        // The ACLNN checker/tiling layer validates the SwiGLU N alignment and
        // transpose restrictions.  The bridge only needs the post-epilogue
        // width to allocate y and y_scale before dispatch.
        n /= 2;
    }

    c10::SmallVector<int64_t, op_infer::SIZE> outputSize;
    auto x1BatchNum = x1DimNum - 2;
    auto x2BatchNum = x2DimNum - 2;
    auto maxBatchNum = std::max(x1BatchNum, x2BatchNum);
    for (int64_t i = 0; i < maxBatchNum; i++) {
        int64_t x1Idx = x1BatchNum - (maxBatchNum - i);
        int64_t x2Idx = x2BatchNum - (maxBatchNum - i);
        int64_t x1Dim = (x1Idx >= 0) ? x1.size(x1Idx) : 1;
        int64_t x2Dim = (x2Idx >= 0) ? x2.size(x2Idx) : 1;
        TORCH_CHECK(x1Dim == x2Dim || x1Dim == 1 || x2Dim == 1,
                    "batch dims are not broadcastable: x1 batch dim = ", x1Dim, ", x2 batch dim = ", x2Dim);
        outputSize.push_back(std::max(x1Dim, x2Dim));
    }
    outputSize.push_back(m);
    outputSize.push_back(n);

    c10::SmallVector<int64_t, op_infer::SIZE> scaleSize = op_infer::array_to_small_vector(outputSize);
    int64_t scaleAxis = outputSize.size() - 1;
    int64_t scaleDim = Ops::Base::CeilDiv(scaleSize[scaleAxis], BLOCK_SIZE);
    scaleDim = Ops::Base::CeilDiv(scaleDim, ALIGN_NUM);
    scaleSize[scaleAxis] = scaleDim;
    scaleSize.emplace_back(ALIGN_NUM);

    aclDataType outputAclType;
    int64_t outputDtypeValue = outputDtype.value_or(x1Dtype.value_or(0));
    const bool isFp4Output = outputDtypeValue == dtypeFloat4E2M1;

    at::Tensor output;
    if (isFp4Output) {
        int64_t lastDim = outputSize[outputSize.size() - 1];
        TORCH_CHECK(lastDim % 2 == 0, "The last dim output shape must be divisible by 2 if "
                                      "output dtype is FLOAT4_E2M1");
        outputSize[outputSize.size() - 1] = lastDim / 2;
        output = at::empty(outputSize, at::TensorOptions().dtype(c10::ScalarType::Byte).device(x1.device()));
        outputAclType = GetAclDataType(outputDtypeValue);
    } else if (outputDtype.has_value() || x1Dtype.has_value()) {
        at::ScalarType scalarDtype = at::ScalarType::Float8_e5m2;
        outputAclType = GetAclDataType(outputDtypeValue);
        if (outputAclType == ACL_FLOAT8_E5M2) {
            scalarDtype = at::ScalarType::Float8_e5m2;
        } else if (outputAclType == ACL_FLOAT8_E4M3FN) {
            scalarDtype = at::ScalarType::Float8_e4m3fn;
        } else {
            TORCH_CHECK(false,
                        "unsupport output_dtype, only support output_dtype FLOAT8_E5M2, FLOAT8_E4M3FN or FLOAT4_E2M1");
        }
        output = at::empty(outputSize, at::TensorOptions().dtype(scalarDtype).device(x1.device()));
    } else {
        outputAclType = ConvertToAclDataType(x1.scalar_type());
        output = at::empty(outputSize, at::TensorOptions().dtype(x1.scalar_type()).device(x1.device()));
    }

    at::Tensor scale = at::empty(scaleSize,
                                 at::TensorOptions().dtype(c10::ScalarType::Float8_e8m0fnu).device(x1.device()));

    TensorWrapper outputWrapper = {output, outputAclType};
    TensorWrapper scaleWrapper = {scale, aclDataType::ACL_FLOAT8_E8M0};
    TensorWrapper x1Wrapper = {
        x1, x1Dtype.has_value() ? GetAclDataType(x1Dtype.value()) : ConvertToAclDataType(x1.scalar_type())};
    TensorWrapper x2Wrapper = {
        x2, x2Dtype.has_value() ? GetAclDataType(x2Dtype.value()) : ConvertToAclDataType(x2.scalar_type())};

    at::Tensor x1ScaleReal = x1Scale.has_value() ? x1Scale.value() : at::Tensor();

    aclDataType x1ScaleAclType = ACL_DT_UNDEFINED;
    if (x1ScaleDtype.has_value()) {
        x1ScaleAclType = GetAclDataType(x1ScaleDtype.value());
    } else if (x1ScaleReal.defined()) {
        x1ScaleAclType = ConvertToAclDataType(x1ScaleReal.scalar_type());
    }
    TensorWrapper x1ScaleWrapper = {x1ScaleReal, x1ScaleAclType};
    TensorWrapper x2ScaleWrapper = {x2Scale, x2ScaleDtype.has_value() ? GetAclDataType(x2ScaleDtype.value()) :
                                                                        ConvertToAclDataType(x2Scale.scalar_type())};

    int64_t scaleAlgValue = scaleAlg.has_value() ? scaleAlg.value() : DEFAULT_SCALE_ALG;

    at::IntArrayRef groupSizeRef = groupSizes.has_value() ? at::IntArrayRef(groupSizes.value()) : at::IntArrayRef{};
    int64_t groupSize = CheckAndGetGroupSize(groupSizeRef);

    std::string activationTypeString(activationType.data(), activationType.size());
    std::string quantModeString(quantMode.data(), quantMode.size());
    std::string roundModeString(roundMode.data(), roundMode.size());
    char* activationTypePtr = activationTypeString.data();
    char* quantModePtr = quantModeString.data();
    char* roundModePtr = roundModeString.data();

    bool isX2Nz = at_npu::native::get_npu_format(x2) == static_cast<int64_t>(ACL_FORMAT_FRACTAL_NZ);
    if (isX2Nz) {
        ACLNN_CMD(aclnnQuantMatmulActivationQuantWeightNzV2, x1Wrapper, x2Wrapper, x1ScaleWrapper, x2ScaleWrapper, bias,
                  x3, transposeX1, transposeX2, groupSize, activationTypePtr, quantModePtr, roundModePtr, scaleAlgValue,
                  dstTypeMax, outputWrapper, scaleWrapper);
    } else {
        ACLNN_CMD(aclnnQuantMatmulActivationQuantV2, x1Wrapper, x2Wrapper, x1ScaleWrapper, x2ScaleWrapper, bias, x3,
                  transposeX1, transposeX2, groupSize, activationTypePtr, quantModePtr, roundModePtr, scaleAlgValue,
                  dstTypeMax, outputWrapper, scaleWrapper);
    }

    return std::make_tuple(output, scale);
}
} // namespace matmul
} // namespace cann_ops_nn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quant_matmul_activation_quant", &cann_ops_nn::matmul::QuantMatmulActivationQuant,
          "QuantMatmulActivationQuant on NPU");
}
