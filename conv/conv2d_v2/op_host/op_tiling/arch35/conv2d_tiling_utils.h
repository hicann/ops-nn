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
 * \file conv2d_tiling_utils.h
 * \brief
 */

#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_CONV2D_TILING_UTILS_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_CONV2D_TILING_UTILS_H

#include "common/op_host/op_tiling/arch35/conv_base.h"

namespace optiling {
namespace conv_ops_tiling {
using conv_tiling::DTYPE_TO_STR;

#define OPS_CHECK_NOT_NULL_WITH_CONTEXT(context, ptr)                                         \
  if ((ptr) != nullptr) {                                                                        \
    const char* name = ((context)->GetNodeName() == nullptr) ? "nil" : (context)->GetNodeName(); \
    OP_LOGE_WITHOUT_REPORT(name, "%s not support now, must be nullptr!", #ptr);                  \
    return ge::GRAPH_FAILED;                                                                     \
  }

constexpr uint32_t FORMAT_NCHW_N_INDEX = 0;
constexpr uint32_t FORMAT_NCHW_C_INDEX = 1;
constexpr uint32_t FORMAT_NCHW_H_INDEX = 2;
constexpr uint32_t FORMAT_NCHW_W_INDEX = 3;

constexpr uint32_t FORMAT_NCHW_DIM = 4;
constexpr uint32_t FORMAT_ND_DIM = 1;
constexpr uint32_t CONV2D_DIM_SIZE_LIMIT = 4;

constexpr uint32_t ATTR_STRIDE_INDEX = 0;
constexpr uint32_t ATTR_PAD_INDEX = 1;
constexpr uint32_t ATTR_DILATION_INDEX = 2;
constexpr uint32_t ATTR_GROUP_INDEX = 3;
constexpr uint32_t ATTR_DATAFORMAT_INDEX = 4;
constexpr uint32_t ATTR_PAD_MODE_INDEX = 6;
constexpr uint32_t ATTR_ENABLE_HF32_INDEX = 7;

constexpr uint32_t ATTR_QUANT_DTYPE_INDEX = 0;
constexpr uint32_t ATTR_QUANT_STRIDE_INDEX = 1;
constexpr uint32_t ATTR_QUANT_PAD_INDEX = 2;
constexpr uint32_t ATTR_QUANT_DILATION_INDEX = 3;
constexpr uint32_t ATTR_QUANT_GROUP_INDEX = 4;
constexpr uint32_t ATTR_QUANT_DATAFORMAT_INDEX = 5;
constexpr uint32_t ATTR_QUANT_OFFSETX_INDEX = 6;
constexpr uint32_t ATTR_QUANT_ROUNDMODE_INDEX = 7;

constexpr uint32_t EXTENDCONV_ATTR_STRIDES_INDEX = 0;
constexpr uint32_t EXTENDCONV_ATTR_PADS_INDEX = 1;
constexpr uint32_t EXTENDCONV_ATTR_DILATIONS_INDEX = 2;
constexpr uint32_t EXTENDCONV_ATTR_GROUPS_INDEX = 3;
constexpr uint32_t EXTENDCONV_ATTR_DATA_FORMAT_INDEX = 4;
constexpr uint32_t EXTENDCONV_ATTR_OFFSET_X_INDEX = 5;
constexpr uint32_t EXTENDCONV_ATTR_ROUND_MODE_INDEX = 6;
constexpr uint32_t EXTENDCONV_ATTR_PAD_MODE_INDEX = 7;
constexpr uint32_t EXTENDCONV_ATTR_ENABLE_HF32_INDEX = 8;
constexpr uint32_t EXTENDCONV_ATTR_ENABLE_RELU_0_INDEX = 9;
constexpr uint32_t EXTENDCONV_ATTR_ENABLE_RELU_1_INDEX = 10;
constexpr uint32_t EXTENDCONV_ATTR_DUAL_OUTPUT_INDEX = 11;
constexpr uint32_t EXTENDCONV_ATTR_DTYPE_0_INDEX = 12;
constexpr uint32_t EXTENDCONV_ATTR_DTYPE_1_INDEX = 13;

constexpr uint32_t EXTENDCONV_INPUT_SCALE_0_INDEX = 4;
constexpr uint32_t EXTENDCONV_INPUT_RELU_WIGHT_0_INDEX = 5;
constexpr uint32_t EXTENDCONV_INPUT_CLIP_VALUE_0_INDEX = 6;
constexpr uint32_t EXTENDCONV_INPUT_SCALE_1_INDEX = 7;
constexpr uint32_t EXTENDCONV_INPUT_RELU_WIGHT_1_INDEX = 8;
constexpr uint32_t EXTENDCONV_INPUT_CLIP_VALUE_1_INDEX = 9;
constexpr uint32_t EXTENDCONV_OUTPUT_1_INDEX = 1;

constexpr uint32_t PAD_TOP_INDEX = 0;
constexpr uint32_t PAD_BOTTOM_INDEX = 1;
constexpr uint32_t PAD_LEFT_INDEX = 2;
constexpr uint32_t PAD_RIGHT_INDEX = 3;

constexpr uint32_t LOAD3D_MAX_STRIDE_H_W = 63;
constexpr uint32_t LOAD3D_MAX_DILATION_H_W = 255;
constexpr uint32_t LOAD3D_MAX_PAD = 255;
constexpr uint64_t LOAD3D_MAX_FILTER_H_W = 511;

constexpr uint32_t BASICBLOCK_BOUNDARY_VALUE_64 = 64;
constexpr uint32_t BASICBLOCK_BOUNDARY_VALUE_128 = 128;
constexpr uint32_t BASICBLOCK_INIT_VALUE_64 = 64;
constexpr uint32_t BASICBLOCK_INIT_VALUE_128 = 128;
constexpr uint32_t BASICBLOCK_INIT_VALUE_256 = 256;
constexpr uint32_t BASICBLOCK_INIT_VALUE_512 = 512;
constexpr uint32_t BASICBLOCK_INIT_VALUE_1024 = 1024;

constexpr uint32_t COUT_LIMIT_128 = 128;
constexpr uint32_t ENABLE_MMODE_CONV1D_WO_LIMIT_128 = 128;

struct Conv2dOriginFormatAixsPosInfo {
    uint32_t nIndex = 0;
    uint32_t cIndex = 0;
    uint32_t hIndex = 0;
    uint32_t wIndex = 0;
};

// the function used by new MDC chip supporting fix-point operation
inline bool IsMdcSoc(const platform_ascendc::SocVersion shortSoc)
{
    return shortSoc == platform_ascendc::SocVersion::MC62CM12A;
}

// the function used by judgeing operation type
inline bool isQuantConv2D(const string& nodeType)
{
    return nodeType == "QuantConv2D";
}

inline bool isExtendConv2D(const string& nodeType)
{
    return nodeType == "ExtendConv2D";
}

const vector<string> CONV2D_SUPPORTED_NODETYPE = {
    "Conv2DV2",
    "QuantConv2D",
    "ExtendConv2D"
};
const vector<int64_t> EXTENDCONV2D_SUPPORTED_ATTR_DTYPE = {
    -1,
    0,
    1,
    2,
    27,
    34,
    36
};
}
}
#endif