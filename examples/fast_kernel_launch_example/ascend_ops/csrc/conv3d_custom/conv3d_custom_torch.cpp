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
 * \file conv3d_custom_torch.cpp
 * \brief
 */

#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include "acl/acl.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/DeviceUtils.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "tiling/platform/platform_ascendc.h"
#include "kernel_operator.h"

#include "conv/conv3d_v2/op_host/op_tiling/conv3d_tiling_engine.h"
#include "conv/conv3d_v2/op_kernel/conv3d_v2_kernel_dispatch.h"

namespace ascend_ops {

namespace Conv3dCustom {
using namespace std;
using namespace Ops::NN::Conv3dV2;
using namespace Ops::NN::Conv3dTilingEngineApi;

constexpr static uint8_t NCDHW_DIM = 5;
constexpr static uint8_t NDC1HWC0_DIM = 6;
constexpr static uint8_t FRACTAL_Z_3D_DIM = 4;
constexpr static uint8_t CONV_ATTRS_DIM = 3;
constexpr static uint8_t N_DIM_IDX_NCDHW = 0;
constexpr static uint8_t C_DIM_IDX_NCDHW = 1;
constexpr static uint8_t D_DIM_IDX_NCDHW = 2;
constexpr static uint8_t H_DIM_IDX_NCDHW = 3;
constexpr static uint8_t W_DIM_IDX_NCDHW = 4;
constexpr static uint8_t N_DIM_IDX_NDC1HWC0 = 0;
constexpr static uint8_t D_DIM_IDX_NDC1HWC0 = 1;
constexpr static uint8_t H_DIM_IDX_NDC1HWC0 = 3;
constexpr static uint8_t W_DIM_IDX_NDC1HWC0 = 4;
constexpr static uint8_t ATTRS_D_DIM_IDX_NCDHW = 0;
constexpr static uint8_t ATTRS_H_DIM_IDX_NCDHW = 1;
constexpr static uint8_t ATTRS_W_DIM_IDX_NCDHW = 2;

constexpr static uint8_t PBUFFERFLAG_L0A_DB_MASK = 1;
constexpr static uint8_t PBUFFERFLAG_L0B_DB_MASK = 2;

static std::map<at::ScalarType, uint8_t> K0_BY_DTYPE_MAP = {{at::kHalf, 16}, {at::kBFloat16, 16}, {at::kFloat, 8}};

template <typename T>
struct ToAscendDtype {
    using Type = float;
};

template <>
struct ToAscendDtype<c10::BFloat16> {
    using Type = bfloat16_t;
};

template <>
struct ToAscendDtype<c10::Half> {
    using Type = half;
};

template <typename T>
struct ToAscendBiasDtype {
    using Type = float;
};

template <>
struct ToAscendBiasDtype<c10::Half> {
    using Type = half;
};

template <typename InputType, typename WeightType, typename OutputType, typename BiasType>
__aicore__ void L0PingPongAllClose(
    __gm__ uint8_t* input, __gm__ uint8_t* weight, __gm__ uint8_t* bias, __gm__ uint8_t* output,
    const Conv3DV2TilingData& tilingData)
{
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_OFF) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::M_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::ALL_CLOSE, ConvBL1ByPass::BYPASS_OFF,
            GroupConvType::NoGroup_Conv, OutputOrder::M_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_OFF) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::HW_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::ALL_CLOSE, ConvBL1ByPass::BYPASS_OFF,
            GroupConvType::NoGroup_Conv, OutputOrder::HW_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_ON) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::M_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::ALL_CLOSE, ConvBL1ByPass::BYPASS_ON,
            GroupConvType::NoGroup_Conv, OutputOrder::M_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_ON) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::HW_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::ALL_CLOSE, ConvBL1ByPass::BYPASS_ON,
            GroupConvType::NoGroup_Conv, OutputOrder::HW_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
}

template <typename InputType, typename WeightType, typename OutputType, typename BiasType>
__aicore__ void L0BPingPongOpen(
    __gm__ uint8_t* input, __gm__ uint8_t* weight, __gm__ uint8_t* bias, __gm__ uint8_t* output,
    const Conv3DV2TilingData& tilingData)
{
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_OFF) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::M_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::L0B_OPEN, ConvBL1ByPass::BYPASS_OFF,
            GroupConvType::NoGroup_Conv, OutputOrder::M_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_OFF) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::HW_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::L0B_OPEN, ConvBL1ByPass::BYPASS_OFF,
            GroupConvType::NoGroup_Conv, OutputOrder::HW_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_ON) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::M_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::L0B_OPEN, ConvBL1ByPass::BYPASS_ON,
            GroupConvType::NoGroup_Conv, OutputOrder::M_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_ON) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::HW_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::L0B_OPEN, ConvBL1ByPass::BYPASS_ON,
            GroupConvType::NoGroup_Conv, OutputOrder::HW_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
}

template <typename InputType, typename WeightType, typename OutputType, typename BiasType>
__aicore__ void L0APingPongOpen(
    __gm__ uint8_t* input, __gm__ uint8_t* weight, __gm__ uint8_t* bias, __gm__ uint8_t* output,
    const Conv3DV2TilingData& tilingData)
{
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_OFF) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::M_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::L0A_OPEN, ConvBL1ByPass::BYPASS_OFF,
            GroupConvType::NoGroup_Conv, OutputOrder::M_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_OFF) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::HW_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::L0A_OPEN, ConvBL1ByPass::BYPASS_OFF,
            GroupConvType::NoGroup_Conv, OutputOrder::HW_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_ON) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::M_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::L0A_OPEN, ConvBL1ByPass::BYPASS_ON,
            GroupConvType::NoGroup_Conv, OutputOrder::M_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_ON) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::HW_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::L0A_OPEN, ConvBL1ByPass::BYPASS_ON,
            GroupConvType::NoGroup_Conv, OutputOrder::HW_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
}

template <typename InputType, typename WeightType, typename OutputType, typename BiasType>
__aicore__ void L0PingPongAllOpen(
    __gm__ uint8_t* input, __gm__ uint8_t* weight, __gm__ uint8_t* bias, __gm__ uint8_t* output,
    const Conv3DV2TilingData& tilingData)
{
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_OFF) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::M_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::ALL_OPEN, ConvBL1ByPass::BYPASS_OFF,
            GroupConvType::NoGroup_Conv, OutputOrder::M_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_OFF) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::HW_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::ALL_OPEN, ConvBL1ByPass::BYPASS_OFF,
            GroupConvType::NoGroup_Conv, OutputOrder::HW_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_ON) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::M_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::ALL_OPEN, ConvBL1ByPass::BYPASS_ON,
            GroupConvType::NoGroup_Conv, OutputOrder::M_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_ON) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::HW_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::ALL_OPEN, ConvBL1ByPass::BYPASS_ON,
            GroupConvType::NoGroup_Conv, OutputOrder::HW_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
}

template <typename T>
__global__ __aicore__ void Conv3dCustomKernel(
    __gm__ uint8_t* input, __gm__ uint8_t* weight, __gm__ uint8_t* bias, __gm__ uint8_t* output,
    const Conv3DV2TilingData tilingData)
{
    if constexpr (std::is_same_v<T, c10::Half> || std::is_same_v<T, c10::BFloat16> || std::is_same_v<T, float>) {
        using InputType = ConvType<TPosition::GM, ConvFormat::NDC1HWC0, typename ToAscendDtype<T>::Type>;
        using WeightType = ConvType<TPosition::GM, ConvFormat::FRACTAL_Z_3D, typename ToAscendDtype<T>::Type>;
        using OutputType = ConvType<TPosition::GM, ConvFormat::NDC1HWC0, typename ToAscendDtype<T>::Type>;
        using BiasType = ConvType<TPosition::GM, ConvFormat::ND, typename ToAscendBiasDtype<T>::Type>;

        bool l0aDbFlag = tilingData.conv3dApiTiling.pBufferFlag & PBUFFERFLAG_L0A_DB_MASK;
        bool l0bDbFlag = tilingData.conv3dApiTiling.pBufferFlag & PBUFFERFLAG_L0B_DB_MASK;
        if (!l0aDbFlag && !l0bDbFlag) {
            L0PingPongAllClose<InputType, WeightType, OutputType, BiasType>(input, weight, bias, output, tilingData);
            return;
        }
        if (!l0aDbFlag && l0bDbFlag) {
            L0BPingPongOpen<InputType, WeightType, OutputType, BiasType>(input, weight, bias, output, tilingData);
            return;
        }
        if (l0aDbFlag && !l0bDbFlag) {
            L0APingPongOpen<InputType, WeightType, OutputType, BiasType>(input, weight, bias, output, tilingData);
            return;
        }
        if (l0aDbFlag && l0bDbFlag) {
            L0PingPongAllOpen<InputType, WeightType, OutputType, BiasType>(input, weight, bias, output, tilingData);
            return;
        }
    }
}

template <typename T>
void Conv3dCustomApi(
    aclrtStream stream, const at::Tensor& input, const at::Tensor& weight, const torch::Tensor& output,
    const vector<int64_t> strideList, const vector<int64_t> paddingList, const vector<int64_t> dilationList,
    const vector<int64_t> oriInputShapeList, const vector<int64_t> oriWeightShapeList,
    const c10::optional<torch::Tensor>& bias, bool enableHF32)
{
    vector<int64_t> oriOutputShapeList = {
        output.size(N_DIM_IDX_NDC1HWC0), oriWeightShapeList[N_DIM_IDX_NCDHW], output.size(D_DIM_IDX_NDC1HWC0),
        output.size(H_DIM_IDX_NDC1HWC0), output.size(W_DIM_IDX_NDC1HWC0)};

    __gm__ uint8_t* input_ptr = (__gm__ uint8_t*)input.data_ptr<T>();
    __gm__ uint8_t* weight_ptr = (__gm__ uint8_t*)weight.data_ptr<T>();
    __gm__ uint8_t* bias_ptr = nullptr;
    __gm__ uint8_t* output_ptr = (__gm__ uint8_t*)output.data_ptr<T>();
    Conv3dTilingEngine conv3dTilingEngine;
    conv3dTilingEngine.SetOrgWeightShape(oriWeightShapeList);
    conv3dTilingEngine.SetOrgFmapShape(oriInputShapeList);
    conv3dTilingEngine.SetOrgOutputShape(oriOutputShapeList);
    conv3dTilingEngine.SetPadding(paddingList);
    conv3dTilingEngine.SetStride(strideList);
    conv3dTilingEngine.SetDilation(dilationList);
    conv3dTilingEngine.SetHF32(enableHF32);
    if constexpr (std::is_same_v<T, c10::Half>) {
        conv3dTilingEngine.SetDataType(
            Conv3dApiTiling::ConvDtype::FLOAT16, Conv3dApiTiling::ConvDtype::FLOAT16,
            Conv3dApiTiling::ConvDtype::FLOAT16);
    } else if constexpr (std::is_same_v<T, c10::BFloat16>) {
        conv3dTilingEngine.SetDataType(
            Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16, Conv3dApiTiling::ConvDtype::BF16);
    } else {
        conv3dTilingEngine.SetDataType(
            Conv3dApiTiling::ConvDtype::FLOAT32, Conv3dApiTiling::ConvDtype::FLOAT32,
            Conv3dApiTiling::ConvDtype::FLOAT32);
    }
    if (bias.has_value()) {
        if constexpr (std::is_same_v<T, c10::BFloat16> || std::is_same_v<T, float>) {
            bias_ptr = (__gm__ uint8_t*)bias.value().data_ptr<float>();
            conv3dTilingEngine.SetBias(true, Conv3dApiTiling::ConvDtype::FLOAT32);
        } else {
            bias_ptr = (__gm__ uint8_t*)bias.value().data_ptr<T>();
            conv3dTilingEngine.SetBias(true, Conv3dApiTiling::ConvDtype::FLOAT16);
        }
    }
    Conv3DV2TilingData tilingData;
    if (!conv3dTilingEngine.GetConv3DV2TilingData(tilingData)) {
        throw std::runtime_error("Failed to GetConv3DV2TilingData!");
    }
    uint32_t blockDim = tilingData.conv3dRunInfo.batchDim * tilingData.conv3dRunInfo.doDim *
                        tilingData.conv3dRunInfo.mDim * tilingData.conv3dRunInfo.nDim;
    Conv3dCustomKernel<T><<<blockDim, nullptr, stream>>>(input_ptr, weight_ptr, bias_ptr, output_ptr, tilingData);
}

template <>
void Conv3dCustomApi<double>(
    aclrtStream stream, const at::Tensor& input, const at::Tensor& weight, const torch::Tensor& output,
    const vector<int64_t> strideList, const vector<int64_t> paddingList, const vector<int64_t> dilationList,
    const vector<int64_t> oriInputShapeList, const vector<int64_t> oriWeightShapeList,
    const c10::optional<torch::Tensor>& bias, bool enableHF32)
{
    throw std::runtime_error("Double is not supported on aicore!");
}

torch::Tensor CreateOutputTensor(
    const torch::Tensor& input, const c10::IntArrayRef& oriInputShape, const c10::IntArrayRef& oriWeightShape,
    const c10::IntArrayRef& strides, const c10::IntArrayRef& pads, const c10::IntArrayRef& dilations)
{
    int64_t N = oriInputShape[N_DIM_IDX_NCDHW];
    int64_t D = oriInputShape[D_DIM_IDX_NCDHW];
    int64_t H = oriInputShape[H_DIM_IDX_NCDHW];
    int64_t W = oriInputShape[W_DIM_IDX_NCDHW];
    int64_t Co = oriWeightShape[N_DIM_IDX_NCDHW];
    vector<int64_t> kernelSize = {
        oriWeightShape[D_DIM_IDX_NCDHW], oriWeightShape[H_DIM_IDX_NCDHW], oriWeightShape[W_DIM_IDX_NCDHW]};

    int64_t Do = (D + 2 * pads[ATTRS_D_DIM_IDX_NCDHW] -
                  dilations[ATTRS_D_DIM_IDX_NCDHW] * (kernelSize[ATTRS_D_DIM_IDX_NCDHW] - 1) - 1) /
                     strides[ATTRS_D_DIM_IDX_NCDHW] +
                 1;
    int64_t Ho = (H + 2 * pads[ATTRS_H_DIM_IDX_NCDHW] -
                  dilations[ATTRS_H_DIM_IDX_NCDHW] * (kernelSize[ATTRS_H_DIM_IDX_NCDHW] - 1) - 1) /
                     strides[ATTRS_H_DIM_IDX_NCDHW] +
                 1;
    int64_t Wo = (W + 2 * pads[ATTRS_W_DIM_IDX_NCDHW] -
                  dilations[ATTRS_W_DIM_IDX_NCDHW] * (kernelSize[ATTRS_W_DIM_IDX_NCDHW] - 1) - 1) /
                     strides[ATTRS_W_DIM_IDX_NCDHW] +
                 1;

    TORCH_CHECK(N > 0, "Output of N has to be positive, but got ", N);
    TORCH_CHECK(Co > 0, "Output of C has to be positive, but got ", Co);
    TORCH_CHECK(Do > 0, "Output of D has to be positive, but got ", Do);
    TORCH_CHECK(Ho > 0, "Output of H has to be positive, but got ", Ho);
    TORCH_CHECK(Wo > 0, "Output of W has to be positive, but got ", Wo);

    uint8_t k0 = K0_BY_DTYPE_MAP[input.scalar_type()];

    return torch::zeros({N, Do, static_cast<int64_t>(Conv3dApiTiling::CeilDiv(Co, k0)), Ho, Wo, k0}, input.options());
}

torch::Tensor Conv3dCustomNpu(
    const torch::Tensor& input, const torch::Tensor& weight, const c10::IntArrayRef& strides,
    const c10::IntArrayRef& pads, const c10::IntArrayRef& dilations, const c10::IntArrayRef& oriInputShape,
    const c10::IntArrayRef& oriWeightShape, const c10::optional<torch::Tensor>& bias, bool enableHF32)
{
    TORCH_CHECK(torch_npu::utils::is_npu(input), "Input tensor must be on NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(weight), "Weight tensor must be on NPU device");
    if (bias.has_value()) {
        TORCH_CHECK(torch_npu::utils::is_npu(bias.value()), "Bias tensor must be on NPU device");
        if (input.scalar_type() == at::kBFloat16) {
            TORCH_CHECK(
                bias.value().scalar_type() == at::kFloat, "Bias type must be float when input dtype is bfloat16");
        }
    }
    TORCH_CHECK(input.dim() == NDC1HWC0_DIM, "Input must be 5D (N, D, C1, H, W, C0)");
    TORCH_CHECK(weight.dim() == FRACTAL_Z_3D_DIM, "Weight must be 4D (Kd*C1*Kh*Kw, N1, N0, C0)");
    TORCH_CHECK(strides.size() == CONV_ATTRS_DIM, "Stride must have 3 elements");
    TORCH_CHECK(pads.size() == CONV_ATTRS_DIM, "Padding must have 3 elements");
    TORCH_CHECK(dilations.size() == CONV_ATTRS_DIM, "Dilation must have 3 elements");
    TORCH_CHECK(oriInputShape.size() == NCDHW_DIM, "Origin Input must have 5 elements (N, C, D, H, W)");
    TORCH_CHECK(oriWeightShape.size() == NCDHW_DIM, "Origin Weight must have 5 elements (N, C, D, H, W)");

    auto output = CreateOutputTensor(input, oriInputShape, oriWeightShape, strides, pads, dilations);
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    vector<int64_t> oriInputShapeList(oriInputShape.begin(), oriInputShape.end());
    vector<int64_t> oriWeightShapeList(oriWeightShape.begin(), oriWeightShape.end());
    vector<int64_t> strideList(strides.begin(), strides.end());
    vector<int64_t> paddingList = {pads[ATTRS_D_DIM_IDX_NCDHW], pads[ATTRS_D_DIM_IDX_NCDHW],
                                   pads[ATTRS_H_DIM_IDX_NCDHW], pads[ATTRS_H_DIM_IDX_NCDHW],
                                   pads[ATTRS_W_DIM_IDX_NCDHW], pads[ATTRS_W_DIM_IDX_NCDHW]};
    vector<int64_t> dilationList(dilations.begin(), dilations.end());

    auto acl_call = [=]() -> int {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "Conv3dCustomNpu", [&] {
            Conv3dCustomApi<scalar_t>(
                stream, input, weight, output, strideList, paddingList, dilationList, oriInputShapeList,
                oriWeightShapeList, bias, enableHF32);
        });
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("Conv3dv2", acl_call);
    return output;
}

// Register Ascend implementations for conv3dv2
TORCH_LIBRARY_IMPL(ascend_ops, PrivateUse1, m)
{
    m.impl("conv3d_custom", Conv3dCustomNpu);
}

} // namespace Conv3dCustom
} // namespace ascend_ops
