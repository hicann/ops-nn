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
 * \brief 基于Ascend 910B处理器的3D卷积操作实现
 * 
 * 本文件实现了一个自定义的PyTorch扩展，用于在Ascend NPU上执行3D卷积操作。
 * 该实现使用了AscendC编程模型，通过tiling策略、L0 ping-pong技术和多缓冲优化数据传输和计算性能。
 * 
 * 主要功能：
 * 1. 支持多种数据类型：FP16、BF16、FP32
 * 2. 支持多种数据格式：NCDHW、NDC1HWC0、FRACTAL_Z_3D
 * 3. 使用tiling策略进行数据分块优化
 * 4. 支持L0 ping-pong技术（L0A、L0B、ALL_CLOSE、ALL_OPEN等模式）
 * 5. 支持bias偏置项
 * 6. 支持HF32混合精度模式
 * 7. 支持BL1 bypass优化
 * 8. 支持多种输出顺序模式（M_Mode、HW_Mode）
 * 
 * 数据格式说明：
 * - NCDHW: 标准格式 [N, C, D, H, W]
 * - NDC1HWC0: Ascend优化格式 [N, D, C1, H, W, C0]，其中C0=16
 * - FRACTAL_Z_3D: 权重格式 [Kd*C1*Kh*Kw, N1, N0, C0]
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

// 张量维度常量
constexpr static uint8_t NCDHW_DIM = 5;        // NCDHW格式的维度数
constexpr static uint8_t NDC1HWC0_DIM = 6;     // NDC1HWC0格式的维度数
constexpr static uint8_t FRACTAL_Z_3D_DIM = 4; // FRACTAL_Z_3D格式的维度数
constexpr static uint8_t CONV_ATTRS_DIM = 3;    // 卷积属性（stride、padding、dilation）的维度数

// NCDHW格式中各维度的索引
constexpr static uint8_t N_DIM_IDX_NCDHW = 0;  // N（批次）维度索引
constexpr static uint8_t C_DIM_IDX_NCDHW = 1;  // C（通道）维度索引
constexpr static uint8_t D_DIM_IDX_NCDHW = 2;  // D（深度）维度索引
constexpr static uint8_t H_DIM_IDX_NCDHW = 3;  // H（高度）维度索引
constexpr static uint8_t W_DIM_IDX_NCDHW = 4;  // W（宽度）维度索引

// NDC1HWC0格式中各维度的索引
constexpr static uint8_t N_DIM_IDX_NDC1HWC0 = 0;  // N（批次）维度索引
constexpr static uint8_t D_DIM_IDX_NDC1HWC0 = 1;  // D（深度）维度索引
constexpr static uint8_t H_DIM_IDX_NDC1HWC0 = 3;  // H（高度）维度索引
constexpr static uint8_t W_DIM_IDX_NDC1HWC0 = 4;  // W（宽度）维度索引

// 卷积属性（stride、padding、dilation）在3D空间中的索引
constexpr static uint8_t ATTRSkt_D_DIM_IDX_NCDHW = 0;  // D方向索引
constexpr static uint8_t ATTRS_H_DIM_IDX_NCDHW = 1;  // H方向索引
constexpr static uint8_t ATTRS_W_DIM_IDX_NCDHW = 2;  // W方向索引

// Ping-pong缓冲区标志掩码
constexpr static uint8_t PBUFFERFLAG_L0A_DB_MASK = 1;  // L0A双缓冲标志掩码
constexpr static uint8_t PBUFFERFLAG_L0B_DB_MASK = 2;  // L0B双缓冲标志掩码

// 数据类型到C0值的映射（C0是Ascend优化格式中通道块的大小）
// FP16/BF16: C0=16，FP32: C0=8
static std::map<at::ScalarType, uint8_t> K0_BY_DTYPE_MAP = {{at::kHalf, 16}, {at::kBFloat16, 16}, {at::kFloat, 8}};

/*!
 * \brief 类型特征：将PyTorch数据类型转换为AscendC数据类型
 * 
 * 该模板结构体用于类型转换，将PyTorch的标量类型转换为AscendC对应的数据类型。
 * 默认类型为float，特化版本处理BFloat16和Half类型。
 * 
 * @tparam T PyTorch数据类型
 */
template <typename T>
struct ToAscendDtype {
    using Type = float;  // 默认类型：float
};

// BFloat16特化
template <>
struct ToAscendDtype<c10::BFloat16> {
    using Type = bfloat16_t;
};

// Half（FP16）特化
template <>
struct ToAscendDtype<c10::Half> {
    using Type = half;
};

/*!
 * \brief 类型特征：将PyTorch数据类型转换为AscendC bias数据类型
 * 
 * 该模板结构体用于类型转换，将PyTorch的标量类型转换为AscendC bias对应的数据类型。
 * BFloat16和Float类型的bias使用float类型（提高精度），Half类型使用half类型。
 * 
 * @tparam T PyTorch数据类型
 */
template <typename T>
struct ToAscendBiasDtype {
    using Type = float;  // 默认类型：float
};

// Half（FP16）特化：bias使用half类型
template <>
struct ToAscendBiasDtype<c10::Half> {
    using Type = half;
};

/*!
 * \brief L0 ping-pong全部关闭模式：调用3D卷积kernel
 * 
 * 该函数在L0 ping-pong全部关闭的情况下，根据BL1 bypass标志和输出顺序模式，
 * 选择对应的3D卷积kernel实现。
 * 
 * @tparam InputType 输入数据类型
 * @tparam WeightType 权重数据类型
 * @tparam OutputType 输出数据类型
 * @tparam BiasType 偏置数据类型
 * @param input 输入张量指针
 * @param weight 权重张量指针
 * @param bias 偏置张量指针
 * @param output 输出张量指针
 * @param tilingData Tiling数据，包含所有kernel运行时需要的参数
 */
template <typename InputType, typename WeightType, typename OutputType, typename BiasType>
__aicore__ void L0PingPongAllClose(
    __gm__ uint8_t* input, __gm__ uint8_t* weight, __gm__ uint8_t* bias, __gm__ uint8_t* output,
    const Conv3DV2TilingData& tilingData)
{
    // BL1 bypass关闭 + M模式输出
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_OFF) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::M_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::ALL_CLOSE, ConvBL1ByPass::BYPASS_OFF,
            GroupConvType::NoGroup_Conv, OutputOrder::M_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
    // BL1 bypass关闭 + HW模式输出
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_OFF) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::HW_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::ALL_CLOSE, ConvBL1ByPass::BYPASS_OFF,
            GroupConvType::NoGroup_Conv, OutputOrder::HW_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
    // BL1 bypass开启 + M模式输出
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_ON) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::M_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::ALL_CLOSE, ConvBL1ByPass::BYPASS_ON,
            GroupConvType::NoGroup_Conv, OutputOrder::M_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
    // BL1 bypass开启 + HW模式输出
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_ON) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::HW_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::ALL_CLOSE, ConvBL1ByPass::BYPASS_ON,
            GroupConvType::NoGroup_Conv, OutputOrder::HW_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
}

/*!
 * \brief L0B ping-pong开启模式：调用3D卷积kernel
 * 
 * 该函数在L0B ping-pong开启的情况下，根据BL1 bypass标志和输出顺序模式，
 * 选择对应的3D卷积kernel实现。
 * 
 * @tparam InputType 输入数据类型
 * @tparam WeightType 权重数据类型
 * @tparam OutputType 输出数据类型
 * @tparam BiasType 偏置数据类型
 * @param input 输入张量指针
 * @param weight 权重张量指针
 * @param bias 偏置张量指针
 * @param output 输出张量指针
 * @param tilingData Tiling数据
 */
template <typename InputType, typename WeightType, typename OutputType, typename BiasType>
__aicore__ void L0BPingPongOpen(
    __gm__ uint8_t* input, __gm__ uint8_t* weight, __gm__ uint8_t* bias, __gm__ uint8_t* output,
    const Conv3DV2TilingData& tilingData)
{
    // BL1 bypass关闭 + M模式输出
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_OFF) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::M_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::L0B_OPEN, ConvBL1ByPass::BYPASS_OFF,
            GroupConvType::NoGroup_Conv, OutputOrder::M_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
    // BL1 bypass关闭 + HW模式输出
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_OFF) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::HW_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::L0B_OPEN, ConvBL1ByPass::BYPASS_OFF,
            GroupConvType::NoGroup_Conv, OutputOrder::HW_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
    // BL1 bypass开启 + M模式输出
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_ON) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::M_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::L0B_OPEN, ConvBL1ByPass::BYPASS_ON,
            GroupConvType::NoGroup_Conv, OutputOrder::M_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
    // BL1 bypass开启 + HW模式输出
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_ON) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::HW_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::L0B_OPEN, ConvBL1ByPass::BYPASS_ON,
            GroupConvType::NoGroup_Conv, OutputOrder::HW_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
}

/*!
 * \brief L0A ping-pong开启模式：调用3D卷积kernel
 * 
 * 该函数在L0A ping-pong开启的情况下，根据BL1 bypass标志和输出顺序模式，
 * 选择对应的3D卷积kernel实现。
 * 
 * @tparam InputType 输入数据类型
 * @tparam WeightType 权重数据类型
 * @tparam OutputType 输出数据类型
 * @tparam BiasType 偏置数据类型
 * @param input 输入张量指针
 * @param weight 权重张量指针
 * @param bias 偏置张量指针
 * @param output 输出张量指针
 * @param tilingData Tiling数据
 */
template <typename InputType, typename WeightType, typename OutputType, typename BiasType>
__aicore__ void L0APingPongOpen(
    __gm__ uint8_t* input, __gm__ uint8_t* weight, __gm__ uint8_t* bias, __gm__ uint8_t* output,
    const Conv3DV2TilingData& tilingData)
{
    // BL1 bypass关闭 + M模式输出
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_OFF) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::M_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::L0A_OPEN, ConvBL1ByPass::BYPASS_OFF,
            GroupConvType::NoGroup_Conv, OutputOrder::M_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
    // BL1 bypass关闭 + HW模式输出
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_OFF) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::HW_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::L0A_OPEN, ConvBL1ByPass::BYPASS_OFF,
            GroupConvType::NoGroup_Conv, OutputOrder::HW_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
    // BL1 bypass开启 + M模式输出
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_ON) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::M_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::L0A_OPEN, ConvBL1ByPass::BYPASS_ON,
            GroupConvType::NoGroup_Conv, OutputOrder::M_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
    // BL1 bypass开启 + HW模式输出
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_ON) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::HW_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::L0A_OPEN, ConvBL1ByPass::BYPASS_ON,
            GroupConvType::NoGroup_Conv, OutputOrder::HW_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
}

/*!
 * \brief L0 ping-pong全部开启模式：调用3D卷积kernel
 * 
 * 该函数在L0 ping-pong全部开启的情况下，根据BL1 bypass标志和输出顺序模式，
 * 选择对应的3D卷积kernel实现。
 * 
 * @tparam InputType 输入数据类型
 * @tparam WeightType 权重数据类型
 * @tparam OutputType 输出数据类型
 * @tparam BiasType 偏置数据类型
 * @param input 输入张量指针
 * @param weight 权重张量指针
 * @param bias 偏置张量指针
 * @param output 输出张量指针
 * @param tilingData Tiling数据
 */
template <typename InputType, typename WeightType, typename OutputType, typename BiasType>
__aicore__ void L0PingPongAllOpen(
    __gm__ uint8_t* input, __gm__ uint8_t* weight, __gm__ uint8_t* bias, __gm__ uint8_t* output,
    const Conv3DV2TilingData& tilingData)
{
    // BL1 bypass关闭 + M模式输出
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_OFF) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::M_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::ALL_OPEN, ConvBL1ByPass::BYPASS_OFF,
            GroupConvType::NoGroup_Conv, OutputOrder::M_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
    // BL1 bypass关闭 + HW模式输出
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_OFF) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::HW_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::ALL_OPEN, ConvBL1ByPass::BYPASS_OFF,
            GroupConvType::NoGroup_Conv, OutputOrder::HW_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
    // BL1 bypass开启 + M模式输出
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_ON) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::M_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::ALL_OPEN, ConvBL1ByPass::BYPASS_ON,
            GroupConvType::NoGroup_Conv, OutputOrder::M_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
    // BL1 bypass开启 + HW模式输出
    if (tilingData.conv3dApiTiling.bl1BypassFlag == static_cast<uint8_t>(ConvBL1ByPass::BYPASS_ON) &&
        tilingData.conv3dApiTiling.outputOrder == static_cast<int8_t>(OutputOrder::HW_Mode)) {
        DispatchConv3DV2Kernel<
            InputType, WeightType, OutputType, BiasType, ConvL0PingPong::ALL_OPEN, ConvBL1ByPass::BYPASS_ON,
            GroupConvType::NoGroup_Conv, OutputOrder::HW_Mode, QuantType::NO_QUANT>(
            input, weight, bias, nullptr, nullptr, output, nullptr, &tilingData);
        return;
    }
}

/*!
 * \brief 3D卷积自定义kernel函数
 * 
 * 该函数是3D卷积操作在AI Core上的入口点，负责：
 * 1. 根据数据类型定义输入、权重、输出、偏置的类型
 * 2. 根据ping-pong缓冲区标志选择对应的kernel实现
 * 3. 调用对应的kernel函数执行实际的3D卷积计算
 * 
 * Ping-pong策略：
 * - ALL_CLOSE: L0A和L0B都关闭ping-pong
 * - L0B_OPEN: 仅L0B开启ping-pong
 * - L0A_OPEN: 仅L0A开启ping-pong
 * - ALL_OPEN: L0A和L0B都开启ping-pong
 * 
 * @tparam T 数据类型（FP16、BF16、FP32）
 * @param input 输入张量指针（NDC1HWC0格式）
 * @param weight 权重张量指针（FRACTAL_Z_3D格式）
 * @param bias 偏置张量指针（ND格式）
 * @param output 输出张量指针（NDC1HWC0格式）
 * @param tilingData Tiling数据，包含所有kernel运行时需要的参数
 */
template <typename T>
__global__ __aicore__ void Conv3dCustomKernel(
    __gm__ uint8_t* input, __gm__ uint8_t* weight, __gm__ uint8_t* bias, __gm__ uint8_t* output,
    const Conv3DV2TilingData tilingData)
{
    // 仅支持FP16、BF16、FP32数据类型
    if constexpr (std::is_same_v<T, c10::Half> || std::is_same_v<T, c10::BFloat16> || std::is_same_v<T, float>) {
        // 定义输入、权重、输出、偏置的AscendC类型
        // 输入/输出使用NDC1HWC0格式，权重使用FRACTAL_Z_3D格式，偏置使用ND格式
        using InputType = ConvType<TPosition::GM, ConvFormat::NDC1HWC0, typename ToAscendDtype<T>::Type>;
        using WeightType = ConvType<TPosition::GM, ConvFormat::FRACTAL_Z_3D, typename ToAscendDtype<T>::Type>;
        using OutputType = ConvType<TPosition::GM, ConvFormat::NDC1HWC0, typename ToAscendDtype<T>::Type>;
        using BiasType = ConvType<TPosition::GM, ConvFormat::ND, typename ToAscendBiasDtype<T>::Type>;

        // 检查L0A和L0B的ping-pong缓冲区标志
        bool l0aDbFlag = tilingData.conv3dApiTiling.pBufferFlag & PBUFFERFLAG_L0A_DB_MASK;
        bool l0bDbFlag = tilingData.conv3dApiTiling.pBufferFlag & PBUFFERFLAG_L0B_DB_MASK;
        
        // 根据ping-pong标志选择对应的kernel实现
        if (!l0aDbFlag && !l0bDbFlag) {
            // L0A和L0B都关闭ping-pong
            L0PingPongAllClose<InputType, WeightType, OutputType, BiasType>(input, weight, bias, output, tilingData);
            return;
        }
        if (!l0aDbFlag && l0bDbFlag) {
            // 仅L0B开启ping-pong
            L0BPingPongOpen<InputType, WeightType, OutputType, BiasType>(input, weight, bias, output, tilingData);
            return;
        }
        if (l0aDbFlag && !l0bDbFlag) {
            // 仅L0A开启ping-pong
            L0APingPongOpen<InputType, WeightType, OutputType, BiasType>(input, weight, bias, output, tilingData);
            return;
        }
        if (l0aDbFlag && l0bDbFlag) {
            // L0A和L0B都开启ping-pong
            L0PingPongAllOpen<InputType, WeightType, OutputType, BiasType>(input, weight, bias, output, tilingData);
            return;
        }
    }
}

/*!
 * \brief 3D卷积自定义API函数模板
 * 
 * 该函数是3D卷积操作的核心实现，负责：
 * 1. 初始化tiling引擎并设置各种参数
 * 2. 根据数据类型设置tiling引擎的数据类型
 * 3. 处理bias偏置项
 * 4. 计算tiling数据
 * 5. 计算需要的AI Core数量
 * 6. 调用Conv3dCustomKernel执行实际的卷积计算
 * 
 * @tparam T 数据类型（FP16、BF16、FP32）
 * @param stream ACL流，用于异步执行
 * @param input 输入张量（NDC1HWC0格式）
 * @param weight 权重张量（FRACTAL_Z_3D格式）
 * @param output 输出张量（NDC1HWC0格式）
 * @param strideList 步长列表 [strideD, strideH, strideW]
 * @param paddingList padding列表 [padHead, padTail, padTop, padBottom, padLeft, padRight]
 * @param dilationList 膨胀列表 [dilationD, dilationH, dilationW]
 * @param oriInputShapeList 输入张量的原始形状（NCDHW格式）
 * @param oriWeightShapeList 权重张量的原始形状（NCDHW格式）
 * @param bias 偏置张量（可选）
 * @param enableHF32 是否启用HF32混合精度模式
 */
template <typename T>
void Conv3dCustomApi(
    aclrtStream stream, const at::Tensor& input, const at::Tensor& weight, const torch::Tensor& output,
    const vector<int64_t> strideList, const vector<int64_t> paddingList, const vector<int64_t> dilationList,
    const vector<int64_t> oriInputShapeList, const vector<int64_t> oriWeightShapeList,
    const c10::optional<torch::Tensor>& bias, bool enableHF32)
{
    // 获取输出张量的原始形状（NCDHW格式）
    vector<int64_t> oriOutputShapeList = {
        output.size(N_DIM_IDX_NDC1HWC0), oriWeightShapeList[N_DIM_IDX_NCDHW], output.size(D_DIM_IDX_NDC1HWC0),
        output.size(H_DIM_IDX_NDC1HWC0), output.size(W_DIM_IDX_NDC1HWC0)};

    // 获取输入、权重、输出张量的设备地址指针
    __gm__ uint8_t* input_ptr = (__gm__ uint8_t*)input.data_ptr<T>();
    __gm__ uint8_t* weight_ptr = (__gm__ uint8_t*)weight.data_ptr<T>();
    __gm__ uint8_t* bias_ptr = nullptr;
    __gm__ uint8_t* output_ptr = (__gm__ uint8_t*)output.data_ptr<T>();
    
    // 初始化tiling引擎
    Conv3dTilingEngine conv3dTilingEngine;
    if (!conv3dTilingEngine.Init()) {
        throw std::runtime_error("Failed to initialize Conv3dTilingEngine!");
    }
    
    // 设置tiling引擎的各种参数
    conv3dTilingEngine.SetOrgWeightShape(oriWeightShapeList);  // 设置权重形状
    conv3dTilingEngine.SetOrgFmapShape(oriInputShapeList);      // 设置特征图形状
    conv3dTilingEngine.SetOrgOutputShape(oriOutputShapeList);   // 设置输出形状
    conv3dTilingEngine.SetPadding(paddingList);                  // 设置padding
    conv3dTilingEngine.SetStride(strideList);                  // 设置步长
    conv3dTilingEngine.SetDilation(dilationList);              // 设置膨胀
    conv3dTilingEngine.SetHF32(enableHF32);                    // 设置HF32模式
    
    // 根据数据类型设置tiling引擎的数据类型
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
    
    // 处理bias偏置项
    if (bias.has_value()) {
        auto biasTensor = bias.value();
        TORCH_CHECK(biasTensor.dim() == 1, "Bias must be 1D (Cout), but got ", biasTensor.dim(), "D");
        TORCH_CHECK(
            biasTensor.numel() == oriWeightShapeList[N_DIM_IDX_NCDHW],
            "Bias numel must equal Cout (", oriWeightShapeList[N_DIM_IDX_NCDHW], "), but got ", biasTensor.numel());
        conv3dTilingEngine.SetBiasShape({static_cast<int64_t>(biasTensor.numel())});
        // BFloat16和Float类型的bias使用float类型，Half类型使用half类型
        if constexpr (std::is_same_v<T, c10::BFloat16> || std::is_same_v<T, float>) {
            bias_ptr = (__gm__ uint8_t*)bias.value().data_ptr<float>();
            conv3dTilingEngine.SetBias(true, Conv3dApiTiling::ConvDtype::FLOAT32);
        } else {
            bias_ptr = (__gm__ uint8_t*)bias.value().data_ptr<T>();
            conv3dTilingEngine.SetBias(true, Conv3dApiTiling::ConvDtype::FLOAT16);
        }
    } else {
        conv3dTilingEngine.SetBias(false, Conv3dApiTiling::ConvDtype::FLOAT32);
        conv3dTilingEngine.SetBiasShape({});
    }
    
    // 获取tiling数据
    Conv3DV2TilingData tilingData;
    if (!conv3dTilingEngine.GetConv3DV2TilingData(tilingData)) {
        throw std::runtime_error("Failed to GetConv3DV2TilingData!");
    }
    
    // 计算需要的AI Core数量
    uint32_t numBlocks = tilingData.conv3dRunInfo.batchDim * tilingData.conv3dRunInfo.doDim *
                        tilingData.conv3dRunInfo.mDim * tilingData.conv3dRunInfo.nDim;
    
    // 调用Conv3dCustomKernel执行实际的3D卷积计算
    Conv3dCustomKernel<T><<<numBlocks, nullptr, stream>>>(input_ptr, weight_ptr, bias_ptr, output_ptr, tilingData);
}

/*!
 * \brief Conv3dCustomApi的double类型特化版本
 * 
 * 该函数是Conv3dCustomApi的double类型特化版本，因为AI Core不支持double类型，
 * 所以直接抛出异常。
 * 
 * @param stream ACL流
 * @param input 输入张量
 * @param weight 权重张量
 * @param output 输出张量
 * @param strideList 步长列表
 * @param paddingList padding列表
 * @param dilationList 膨胀列表
 * @param oriInputShapeList 输入张量的原始形状
 * @param oriWeightShapeList 权重张量的原始形状
 * @param bias 偏置张量
 * @param enableHF32 是否启用HF32混合精度模式
 * @throw std::runtime_error Double类型不支持
 */
template <>
void Conv3dCustomApi<double>(
    aclrtStream stream, const at::Tensor& input, const at::Tensor& weight, const torch::Tensor& output,
    const vector<int64_t> strideList, const vector<int64_t> paddingList, const vector<int64_t> dilationList,
    const vector<int64_t> oriInputShapeList, const vector<int64_t> oriWeightShapeList,
    const c10::optional<torch::Tensor>& bias, bool enableHF32)
{
    throw std::runtime_error("Double is not supported on aicore!");
}

/*!
 * \brief 根据输入参数计算并创建输出张量
 * 
 * 该函数根据输入张量形状、权重形状、步长、padding和膨胀参数，
 * 计算输出张量的形状并创建一个全零的输出张量。
 * 
 * 输出形状计算公式（以高度H为例）：
 * Ho = (H + 2 * padH - dilationH * (kernelH - 1) - 1) / strideH + 1
 * 
 * 输出张量格式为NDC1HWC0（Ascend优化格式），其中C0的值由数据类型决定：
 * - FP16/BF16: C0=16
 * - FP32: C0=8
 * 
 * @param input 输入张量
 * @param oriInputShape 输入张量的原始形状 [N, C, D, H, W]
 * @param oriWeightShape 权重张量的原始形状 [Co, Ci, Kd, Kh, Kw]
 * @param strides 步长 [strideD, strideH, strideW]
 * @param pads padding [padD, padH, padW] (每个方向2面，共6面)
 * @param dilations 膨胀 [dilationD, dilationH, dilationW]
 * @return 输出张量（NDC1HWC0格式）
 */
torch::Tensor CreateOutputTensor(
    const torch::Tensor& input, const c10::IntArrayRef& oriInputShape, const c10::IntArrayRef& oriWeightShape,
    const c10::IntArrayRef& strides, const c10::IntArrayRef& pads, const c10::IntArrayRef& dilations)
{
    // 提取输入张量的各个维度
    int64_t N = oriInputShape[N_DIM_IDX_NCDHW];  // 批次大小
    int64_t D = oriInputShape[D_DIM_IDX_NCDHW];  // 输入深度
    int64_t H = oriInputShape[H_DIM_IDX_NCDHW];  // 输入高度
    int64_t W = oriInputShape[W_DIM_IDX_NCDHW];  // 输入宽度
    int64_t Co = oriWeightShape[N_DIM_IDX_NCDHW]; // 输出通道数
    vector<int64_t> kernelSize = {
        oriWeightShape[D_DIM_IDX_NCDHW], oriWeightShape[H_DIM_IDX_NCDHW], oriWeightShape[W_DIM_IDX_NCDHW]};

    // 计算输出张量的各个维度
    // 公式：(in_size + 2*pad - dilation*(kernel_size-1) - 1) / stride + 1
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

    // 验证输出形状的合法性
    TORCH_CHECK(N > 0, "Output of N has to be positive, but got ", N);
    TORCH_CHECK(Co > 0, "Output of C has to be positive, but got ", Co);
    TORCH_CHECK(Do > 0, "Output of D has to be positive, but got ", Do);
    TORCH_CHECK(Ho > 0, "Output of H has to be positive, but got ", Ho);
    TORCH_CHECK(Wo > 0, "Output of W has to be positive, but got ", Wo);

    // 获取C0的值（由数据类型决定）
    uint8_t k0 = K0_BY_DTYPE_MAP[input.scalar_type()];

    // 创建并返回全零输出张量（NDC1HWC0格式）
    // C1 = ceil(Co / C0)，其中C0=k0
    return torch::zeros({N, Do, static_cast<int64_t>(Conv3dApiTiling::CeilDiv(Co, k0)), Ho, Wo, k0}, input.options());
}

/*!
 * \brief 3D卷积NPU实现函数
 * 
 * 该函数是3D卷积操作在Ascend NPU上的主入口点，负责：
 * 1. 参数验证：检查输入、权重、bias是否在NPU设备上
 * 2. 形状验证：检查输入输出的维度是否正确
 * 3. 创建输出张量：根据输入参数计算输出形状（NDC1HWC0格式）
 * 4. 转换参数：将PyTorch参数转换为内部格式
 * 5. 数据类型分发：根据输入数据类型选择对应的模板实现
 * 6. 执行kernel：通过OpCommand调用Conv3dCustomApi
 * 
 * 数据格式说明：
 * - 输入：NDC1HWC0格式 [N, D, C1, H, W, C0]
 * - 权重：FRACTAL_Z_3D格式 [Kd*C1*Kh*Kw, N1, N0, C0]
 * - 输出：NDC1HWC0格式 [N, Do, C1, Ho, Wo, C0]
 * 
 * @param input 输入张量（NDC1HWC0格式）
 * @param weight 权重张量（FRACTAL_Z_3D格式）
 * @param strides 步长 [strideD, strideH, strideW]
 * @param pads padding [padD, padH, padW]
 * @param dilations 膨胀 [dilationD, dilationH, dilationW]
 * @param oriInputShape 输入张量的原始形状（NCDHW格式）
 * @param oriWeightShape 权重张量的原始形状（NCDHW格式）
 * @param bias 偏置张量（可选）
 * @param enableHF32 是否启用HF32混合精度模式
 * @return 输出张量（NDC1HWC0格式）
 */
torch::Tensor Conv3dCustomNpu(
    const torch::Tensor& input, const torch::Tensor& weight, const c10::IntArrayRef& strides,
    const c10::IntArrayRef& pads, const c10::IntArrayRef& dilations, const c10::IntArrayRef& oriInputShape,
    const c10::IntArrayRef& oriWeightShape, const c10::optional<torch::Tensor>& bias, bool enableHF32)
{
    // OptionalDeviceGuard确保后续操作在正确的设备上下文执行
    // 它会记录当前设备状态，执行完作用域代码后自动恢复
    const c10::OptionalDeviceGuard guard(input.device());
    
    // 验证输入张量在NPU设备上
    TORCH_CHECK(torch_npu::utils::is_npu(input), "Input tensor must be on NPU device");
    TORCH_CHECK(torch_npu::utils::is_npu(weight), "Weight tensor must be on NPU device");
    if (bias.has_value()) {
        TORCH_CHECK(torch_npu::utils::is_npu(bias.value()), "Bias tensor must be on NPU device");
        // BFloat16类型的输入，bias必须是float类型
        if (input.scalar_type() == at::kBFloat16) {
            TORCH_CHECK(
                bias.value().scalar_type() == at::kFloat, "Bias type must be float when input dtype is bfloat16");
        }
    }
    
    // 验证张量维度和参数大小
    TORCH_CHECK(input.dim() == NDC1HWC0_DIM, "Input must be 5D (N, D, C1, H, W, C0)");
    TORCH_CHECK(weight.dim() == FRACTAL_Z_3D_DIM, "Weight must be 4D (Kd*C1*Kh*Kw, N1, N0, C0)");
    TORCH_CHECK(strides.size() == CONV_ATTRS_DIM, "Stride must have 3 elements");
    TORCH_CHECK(pads.size() == CONV_ATTRS_DIM, "Padding must have 3 elements");
    TORCH_CHECK(dilations.size() == CONV_ATTRS_DIM, "Dilation must have 3 elements");
    TORCH_CHECK(oriInputShape.size() == NCDHW_DIM, "Origin Input must have 5 elements (N, C, D, H, W)");
    TORCH_CHECK(oriWeightShape.size() == NCDHW_DIM, "Origin Weight must have 5 elements (N, C, D, H, W)");

    // 创建输出张量（NDC1HWC0格式）
    auto output = CreateOutputTensor(input, oriInputShape, oriWeightShape, strides, pads, dilations);
    
    // 获取当前NPU流
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    
    // 转换参数为内部格式
    vector<int64_t> oriInputShapeList(oriInputShape.begin(), oriInputShape.end());
    vector<int64_t> oriWeightShapeList(oriWeightShape.begin(), oriWeightShape.end());
    vector<int64_t> strideList(strides.begin(), strides.end());
    vector<int64_t> paddingList = {pads[ATTRS_D_DIM_IDX_NCDHW], pads[ATTRS_D_DIM_IDX_NCDHW],
                                   pads[ATTRS_H_DIM_IDX_NCDHW], pads[ATTRS_H_DIM_IDX_NCDHW],
                                   pads[ATTRS_W_DIM_IDX_NCDHW], pads[ATTRS_W_DIM_IDX_NCDHW]};
    vector<int64_t> dilationList(dilations.begin(), dilations.end());

    // 定义ACL调用的lambda函数，在NPU上执行
    // 使用AT_DISPATCH_FLOATING_TYPES_AND2宏根据数据类型分发到对应的模板实现
    auto acl_call = [=]() -> int {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, input.scalar_type(), "Conv3dCustomNpu", [&] {
            Conv3dCustomApi<scalar_t>(
                stream, input, weight, output, strideList, paddingList, dilationList, oriInputShapeList,
                oriWeightShapeList, bias, enableHF32);
        });
        return 0;
    };
    
    // 通过OpCommand执行API调用
    at_npu::native::OpCommand::RunOpApiV2("Conv3dv2", acl_call);
    
    return output;
}

// 注册算子接口定义，声明函数签名
TORCH_LIBRARY_FRAGMENT(EXTENSION_MODULE_NAME, m)
{
    m.def(
        "conv3d_custom(Tensor input, Tensor weight, int[3] stride, int[3] padding, int[3] dilation, int[5] "
        "oriInputShape, int[5] oriWeightShape, Tensor? bias, bool enable_hf32 = False) -> Tensor");
}

// 注册NPU实现，将conv3d_custom算子映射到Conv3dCustomNpu函数
TORCH_LIBRARY_IMPL(EXTENSION_MODULE_NAME, PrivateUse1, m)
{
    m.impl("conv3d_custom", Conv3dCustomNpu);
}

} // namespace Conv3dCustom
} // namespace ascend_ops
