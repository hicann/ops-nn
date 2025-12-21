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
 * \file conv3d_v2_tiling_utils.h
 * \brief
 */

#ifndef OPS_CONV_OP_TILING_CONV3D_V2_TILING_UTILS_H
#define OPS_CONV_OP_TILING_CONV3D_V2_TILING_UTILS_H

#include <vector>
#include "conv/common/op_host/op_tiling/arch35/conv_base.h"

namespace optiling {
namespace conv_ops_tiling {
using conv_tiling::DTYPE_TO_STR;
constexpr uint32_t CONV3D_DIM_SIZE_LIMIT = 5;

constexpr uint32_t FORMAT_NDC1HWC0_N_INDEX = 0;
constexpr uint32_t FORMAT_NDC1HWC0_D_INDEX = 1;
constexpr uint32_t FORMAT_NDC1HWC0_C1_INDEX = 2;
constexpr uint32_t FORMAT_NDC1HWC0_H_INDEX = 3;
constexpr uint32_t FORMAT_NDC1HWC0_W_INDEX = 4;
constexpr uint32_t FORMAT_NDC1HWC0_C0_INDEX = 5;
constexpr uint32_t FORMAT_NDC1HWC0_DIM = 6;

constexpr uint32_t FORMAT_ND_DIM = 1;

constexpr uint32_t FORMAT_FRACTAL_3D_DKCIN1KHKW_INDEX = 0;
constexpr uint32_t FORMAT_FRACTAL_3D_N1_INDEX = 1;
constexpr uint32_t FORMAT_FRACTAL_3D_N0_INDEX = 2;
constexpr uint32_t FORMAT_FRACTAL_3D_C0_INDEX = 3;
constexpr uint32_t FORMAT_FRACTAL_3D_DIM = 4;

constexpr uint32_t ATTR_STRIDE_INDEX = 0;
constexpr uint32_t ATTR_PAD_INDEX = 1;
constexpr uint32_t ATTR_DILATION_INDEX = 2;
constexpr uint32_t ATTR_GROUP_INDEX = 3;
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
constexpr uint32_t ATTR_QUANT_PADMODE_INDEX = 8;
 
constexpr uint32_t PAD_HEAD_INDEX = 0;
constexpr uint32_t PAD_TAIL_INDEX = 1;
constexpr uint32_t PAD_TOP_INDEX = 2;
constexpr uint32_t PAD_BOTTOM_INDEX = 3;
constexpr uint32_t PAD_LEFT_INDEX = 4;
constexpr uint32_t PAD_RIGHT_INDEX = 5;
constexpr uint32_t FORMAT_PAD_DIM = 6;

constexpr int64_t MAX_N_BF16_SHAPE = 1000000;
constexpr int64_t MAX_FM_D_BF16_SHAPE = 1000000;
constexpr int64_t MAX_FM_H_BF16_SHAPE = 1000000;
constexpr int64_t MAX_FM_W_BF16_SHAPE = 1000000;
constexpr int64_t MAX_KD_BF16_SHAPE = 1000000;
constexpr int64_t MAX_KH_BF16_SHAPE = 1000000;
constexpr int64_t MAX_KW_BF16_SHAPE = 1000000;
constexpr int64_t MAX_CIN_BF16_SHAPE = 1000000;
constexpr int64_t MAX_COUT_BF16_SHAPE = 1000000;
constexpr int64_t MAX_DOUT_SHAPE = 1000000;
constexpr int64_t MAX_HOUT_SHAPE = 1000000;
constexpr int64_t MAX_WOUT_SHAPE = 1000000;
constexpr int64_t MAX_STRIDE_D_SHAPE = 1000000;
constexpr int64_t MAX_STRIDE_H_SHAPE = 63;
constexpr int64_t MAX_STRIDE_W_SHAPE = 63;
constexpr int64_t MAX_DILATION_D_SHAPE = 1000000;
constexpr int64_t MAX_DILATION_H_SHAPE = 255;
constexpr int64_t MAX_DILATION_W_SHAPE = 255;
constexpr int64_t MAX_PAD_D_SHAPE = 1000000;
constexpr int64_t MAX_PAD_H_SHAPE = 255;
constexpr int64_t MAX_PAD_W_SHAPE = 255;

constexpr uint64_t LOAD3D_MAX_STRIDE_H_W = 63;
constexpr uint64_t LOAD3D_MAX_DILATION_H_W = 255;
constexpr uint64_t LOAD3D_MAX_PAD = 255;
constexpr uint64_t LOAD3D_MAX_FILTER_H_W_910_95 = 511;

constexpr uint32_t BLOCKDIM_DEC_NUM = 5;
constexpr uint32_t BLOCKDIM_BATCH_IDX = 0;
constexpr uint32_t BLOCKDIM_M_IDX = 1;
constexpr uint32_t BLOCKDIM_N_IDX = 2;
constexpr uint32_t BLOCKDIM_DO_IDX = 3;
constexpr uint32_t BLOCKDIM_GROUP_IDX = 4;

constexpr uint32_t PB_AL0_IDX = 0x01;
constexpr uint32_t PB_BL0_IDX = 0x02;
constexpr uint32_t PB_CL0_IDX = 0x04;
constexpr uint32_t PB_BL1_IDX = 0x10;
constexpr uint32_t PB_BL0_OFFSET = 1;
constexpr uint32_t PB_CL0_OFFSET = 2;
constexpr uint32_t PB_BL1_OFFSET = 4;

constexpr int COUNT_PARAMS_WITH_BIAS = 4; // [fmap, weight, bias, output]
constexpr int COUNT_PARAMS_WITHOUT_BIAS = 3; // [fmap, weight, output]

struct Conv3dOriginFormatAixsPosInfo {
    uint32_t nIndex = 0;
    uint32_t cIndex = 0;
    uint32_t dIndex = 0;
    uint32_t hIndex = 0;
    uint32_t wIndex = 0;
};

// the function used by judgeing operation type
inline bool isQuantConv3D(const string nodeType)
{
    return nodeType == "QuantConv3D";
}
}
}
#endif