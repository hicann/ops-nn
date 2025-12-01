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
 * \file max_pool3d_with_argmax_v2_base.h
 * \brief
 */

#ifndef OPP_MAX_POOL3D_WITH_ARGMAX_V2_BASE_H
#define OPP_MAX_POOL3D_WITH_ARGMAX_V2_BASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BLOCK_LEN_FP32 = 8;
constexpr uint32_t BLOCK_LEN_UINT8 = 32;
constexpr uint32_t MIN_TRANSPOSE_ROWS = 16;
constexpr uint32_t MAX_VEC_ELEMS_PER_REP_FP32 = 64;
constexpr uint32_t BLOCKS_IN_REP = 8;

const uint32_t MAX_UINT16 = 65536;
const uint32_t MAX_DIV = 2;
const uint32_t BITS_UINT8 = 8;
const uint32_t INT8_INT16 = 2;
const uint32_t D_DIM = 0;
const uint32_t H_DIM = 1;
const uint32_t W_DIM = 2;
const uint32_t DIMS_COUNT = 3;
const uint32_t TRANSDATA_MUTLIPLER = 2;
const uint32_t MASK_COUNT = 2; // use 2 masks: mask for max values and for nan values
const float EPS_COEFF = 0.5f;

#endif