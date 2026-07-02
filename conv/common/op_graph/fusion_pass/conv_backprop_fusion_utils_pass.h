/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV_BACKPROP_FUSION_UTILS_PASS_H
#define CONV_BACKPROP_FUSION_UTILS_PASS_H

#include <algorithm>
#include <map>
#include <set>
#include <vector>
#include <string>

#include "ge/es_graph_builder.h"
#include "ge/es_tensor_holder.h"
#include "ge/ge_utils.h"
#include "platform/platform_info.h"
#include "platform/soc_spec.h"

namespace ops {
namespace ConvBackpropFusionUtils {

// Conv Backprop 公共常量定义
constexpr int32_t CONV_DIM_LENGTH = 5;
constexpr int32_t OUTPUT_INDEX = 0;
constexpr int32_t OUT_BACKPROP_INDEX = 2;
constexpr int64_t HF32_PRECISION_MODE_INT = 0x40;

// Transpose 算子的输入输出索引常量
constexpr int32_t TRANSPOSE_INPUT_X_INDEX = 0;      // Transpose 第一个输入（x）的索引
constexpr int32_t TRANSPOSE_INPUT_PERM_INDEX = 1;   // Transpose 第二个输入（perm）的索引
constexpr int32_t TRANSPOSE_OUTPUT_Y_INDEX = 0;     // Transpose 第一个输出（y）的索引

// Tensor 的默认输出索引常量
constexpr int32_t TENSOR_DEFAULT_OUTPUT_INDEX = 0;  // Tensor 的默认输出索引（大多数算子只有一个输出）

// ConvBackpropV2 算子的输入输出索引常量
constexpr int32_t CONV_BP_V2_INPUT_INDEX = 0;  // input_size 输入索引
constexpr int32_t CONV_BP_V2_FILTER_INDEX = 1;      // filter 输入索引
constexpr int32_t CONV_BP_V2_OUT_BACKPROP_INDEX = 2; // out_backprop 输入索引
constexpr int32_t CONV_BP_V2_OUTPUT_INDEX = 0;      // output 输出索引

// NDHWC格式的维度索引
constexpr int64_t N_DIM_NDHWC_INDEX = 0;
constexpr int64_t D_DIM_NDHWC_INDEX = 1;
constexpr int64_t H_DIM_NDHWC_INDEX = 2;
constexpr int64_t W_DIM_NDHWC_INDEX = 3;
constexpr int64_t C_DIM_NDHWC_INDEX = 4;

// NCDHW格式的维度索引
constexpr int64_t N_DIM_NCDHW_INDEX = 0;
constexpr int64_t C_DIM_NCDHW_INDEX = 1;
constexpr int64_t D_DIM_NCDHW_INDEX = 2;
constexpr int64_t H_DIM_NCDHW_INDEX = 3;
constexpr int64_t W_DIM_NCDHW_INDEX = 4;

// DHWCN格式的维度索引
constexpr int64_t D_DIM_DHWCN_INDEX = 0;
constexpr int64_t H_DIM_DHWCN_INDEX = 1;
constexpr int64_t W_DIM_DHWCN_INDEX = 2;
constexpr int64_t C_DIM_DHWCN_INDEX = 3;
constexpr int64_t N_DIM_DHWCN_INDEX = 4;

// 支持的SOC列表
const std::map<std::string, NpuArch> SUPPORT_SOC_LIST = {
    {"Ascend950", NpuArch::DAV_3510}
};

// Transpose排列常量
const std::vector<int32_t> FILTER_TRANSPOSE_PERM = {4, 3, 0, 1, 2};  // DHWCN -> NCDHW
const std::vector<int32_t> DEDY_TRANSPOSE_PERM = {0, 4, 1, 2, 3};    // NDHWC -> NCDHW
const std::vector<int32_t> OUTPUT_TRANSPOSE_PERM = {0, 2, 3, 4, 1};  // NCDHW -> NDHWC


// Transpose 节点配置结构体
struct TransposeNodeConfig {
    ge::es::EsTensorHolder input;  // 输入张量
    std::vector<int32_t> perm;     // 转置排列
    std::string name;              // 节点名称
    ge::Format format;             // 输出格式
    
    static TransposeNodeConfig Create(
        const ge::es::EsTensorHolder& inputTensor,
        const std::vector<int32_t>& permutation,
        const std::string& nodeName,
        ge::Format outputFormat) {
        TransposeNodeConfig config;
        config.input = inputTensor;
        config.perm = permutation;
        config.name = nodeName;
        config.format = outputFormat;
        return config;
    }
};

class ConvBackpropFusionUtilsPass {
public:

    static std::vector<int64_t> CalcTransposeShape(
        const std::vector<int64_t>& inputShape,
        const std::vector<int32_t>& perm);

    static void SetPlaceholderDesc(
        ge::es::EsTensorHolder& tensorHolder,
        int64_t idx,
        const ge::TensorDesc& desc);

    static bool InWhitelist(
        const std::vector<int64_t>& shape,
        const std::vector<std::vector<int64_t>>& whitelist);

    static bool CheckSocAndIntrinsic(
        const std::map<std::string, NpuArch>& supportSocList,
        NpuArch& npuArch);

    static int64_t GetAiCoreCount();

    static bool IsSupportedDtype(
        ge::DataType dtype,
        const std::set<ge::DataType>& supportedDtypes);

    static bool CreateTransposeNode(
        ge::es::EsGraphBuilder& builder,
        const TransposeNodeConfig& config,
        ge::es::EsTensorHolder& output,
        ge::TensorDesc& outDesc,
        const ge::AscendString& opType);
};

} // namespace ConvBackpropFusionUtils
} // namespace ops

#endif // CONV_BACKPROP_FUSION_UTILS_PASS_H
