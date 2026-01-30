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
 * \file conv3dv2_kernel.h
 * \brief
 */
 
#ifndef CONV3D_V2_KERNEL_H
#define CONV3D_V2_KERNEL_H
 
#include <stdint.h>
#include <string>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "conv3d_v2_tiling_data.h"

#define CONV_FMAP_TILING_FULLLOAD_AL1 0
#define CONV_FMAP_TILING_ONLY_M_FULLLOAD_AL1_AL0 1
#define CONV_FMAP_TILING_OTHER 2

#define CONV_WEIGHT_TILING_FULLLOAD_BL1 0
#define CONV_WEIGHT_TILING_ONLY_N_FULLLOAD_BL1_BL0 1
#define CONV_WEIGHT_TILING_OTHER 2

#define CONV_L1_PINGPONG_ALL_CLOSE 0
#define CONV_L1_PINGPONG_AL1_OPEN 1
#define CONV_L1_PINGPONG_BL1_OPEN 2
#define CONV_L1_PINGPONG_ALL_OPEN 3

#define CONV_L0_PINGPONG_ALL_CLOSE 0
#define CONV_L0_PINGPONG_AL0_OPEN 1
#define CONV_L0_PINGPONG_BL0_OPEN 2
#define CONV_L0_PINGPONG_ALL_OPEN 3

#define CONV_OUTPUT_ORDER_HW_MODE 0
#define CONV_OUTPUT_ORDER_M_MODE 1

#define CONV_ITER_ORDER_MITER_FIRST 0
#define CONV_ITER_ORDER_NITER_FIRST 1

#define CONV_GROUP_TYPE_NORMAL_CONV 0
#define CONV_GROUP_TYPE_ORI_GROUP_CONV 1
#define CONV_GROUP_TYPE_OPT_GROUP_CONV 2

#define CONV_ENABLE_SMALL_CHANNEL_CLOSE 0
#define CONV_ENABLE_SMALL_CHANNEL_OPEN 1

#define CONV_WEIGHT_UB_TRANS_CLOSE 0
#define CONV_WEIGHT_UB_TRANS_OPEN 1

#define CONV_FMAP_LOAD3D_MODE 0
#define CONV_FMAP_DMA_MODE 1

#define CONV_INNER_BATCH_SINGLE 0
#define CONV_INNER_BATCH_KERNEL_1X1_MULTI 1
#define CONV_INNER_BATCH_MULTI 2

void Conv3dv2Template(
    GM_ADDR x, GM_ADDR filter, GM_ADDR bias, GM_ADDR scale, GM_ADDR offset,
    GM_ADDR offset_w, GM_ADDR y, GM_ADDR workspace, Ops::NN::Conv3dV2::Conv3DV2TilingData& tiling,
    int8_t FmapTiling, int8_t WeightTiling, int8_t L1PingPong, int8_t L0PingPong,
    int8_t OutputOrder, int8_t IterOrder,
    const std::string& dtype,
    int32_t blockDim, aclrtStream stream);
 
#endif // CONV3D_V2_KERNEL_H
