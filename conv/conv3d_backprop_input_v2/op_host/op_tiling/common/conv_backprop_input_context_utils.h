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
 * \file conv_backprop_input_context_utils.h
 * \brief
 */
#ifndef CONV_BACKPROP_INPUT_CONTEXT_UTILS_H
#define CONV_BACKPROP_INPUT_CONTEXT_UTILS_H

#include "exe_graph/runtime/tiling_context.h"
#include "tbe_tiling_api.h"

namespace Ops {
namespace NN {
namespace Conv {
using namespace optiling;

constexpr uint64_t BT_BUFFER_SIZE = static_cast<uint64_t>(4 * 1024);    // Bias Buffer size
constexpr uint64_t FB_BUFFER_SIZE = static_cast<uint64_t>(6 * 1024);    // Scale Buffer size

struct Conv3dBpInputV2RunInfo {
    // Batch and group related
    int32_t batch_n;           // Batch size
    int32_t groups;
    int32_t real_g;            // Number of groups
    int32_t enlarge = 1; // group放大倍数

    // Input dimensions (dedx)
    int32_t dedx_d;            // Input depth
    int32_t dedx_cin;         // Input channels
    int32_t dedx_cin_g;         // Input channels per group
    int32_t dedx_cin1;         // Input channels per group
    int32_t dedx_cin1_g;       // Input channels per group (grouped)
    int32_t dedx_h;            // Input height
    int32_t dedx_w;            // Input width

    // Output dimensions (dedy)
    int32_t dedy_d;            // Output depth
    int32_t dedy_cout;        // Output channels
    int32_t dedy_cout_g;        // Output channels per group
    int32_t dedy_cout1;        // Output channels per group
    int32_t dedy_cout1_g;      // Output channels per group (grouped)
    int32_t dedy_h;            // Output height
    int32_t dedy_w;            // Output width

    // Kernel dimensions
    int32_t kernel_d;          // Kernel depth
    int32_t kernel_h;          // Kernel height
    int32_t kernel_w;          // Kernel width

    // Strides
    int32_t stride_d;          // Stride depth
    int32_t stride_h;          // Stride height
    int32_t stride_w;          // Stride width

    // Padding
    int32_t pad_h;             // Padding height
    int32_t pad_t;             // Padding top
    int32_t pad_u;             // Padding up
    int32_t pad_d;             // Padding down
    int32_t pad_l;             // Padding left
    int32_t pad_r;             // Padding right

    // Dilation
    int32_t dilation_d;        // Dilation depth
    int32_t dilation_h;        // Dilation height
    int32_t dilation_w;        // Dilation width

    // Backprop padding
    int32_t backprop_pad_h;    // Backprop padding height
    int32_t backprop_pad_t;    // Backprop padding top
    int32_t backprop_pad_u;    // Backprop padding up
    int32_t backprop_pad_d;    // Backprop padding down
    int32_t backprop_pad_l;    // Backprop padding left
    int32_t backprop_pad_r;    // Backprop padding right

    // Other flags
    int32_t hf32_flag;         // Flag for FP32 handling
    int32_t a_dtype_bytes = 2;
    int32_t b_dtype_bytes = 2;
    int32_t c_dtype_bytes = 2;
    int32_t initOutputFlag = 0;
    uint8_t enRelu = 0;

    ge::Format outBackpropFormat = ge::FORMAT_NCDHW;
    ge::Format filterFormat = ge::FORMAT_NCDHW;
    ge::Format yFormat = ge::FORMAT_NCDHW;
};

bool SetRunInfoToV2(gert::TilingContext* context, Conv3dBpInputV2RunInfo& runInfoV2, optiling::OpTypeV2 opType);
bool IsSocVersionFuse(const gert::TilingContext *context);
} // namespace Conv
} // namespace NN
} // namespace Ops

#endif // CONV_BACKPROP_INPUT_CONTEXT_UTILS_H