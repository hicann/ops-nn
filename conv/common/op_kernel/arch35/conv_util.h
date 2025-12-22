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
 * \file conv_util.h
 * \brief
 */

#ifndef CONV_UTIL_H
#define CONV_UTIL_H

#include "kernel_utils.h"

namespace conv {
using namespace AscendC;

const static uint64_t L0A_SIZE = 65536;
const static uint64_t L0B_SIZE = 65536;
const static uint64_t L0C_SIZE = 262144;
const static uint64_t L0A_HALF_SIZE = 32768;
const static uint64_t L0B_HALF_SIZE = 32768;
const static uint64_t L0C_HALF_SIZE = 131072;
const static uint64_t C0_SIZE = 32;
const static uint64_t BT_SIZE = 64;
const static uint64_t PAD_IDX_T = 2;
const static uint64_t PAD_IDX_B = 3;
const static uint64_t PAD_IDX_L = 0;
const static uint64_t PAD_IDX_R = 1;
const static uint64_t MAX_PAD_R = 255;
const static uint64_t MIN_HI_WI = 1;
const static uint64_t W_TAIL_NUM = 2;
const static uint64_t AL1_DB_IDX = 0x08;
const static uint64_t AL1_DB_OFFSET = 3;
const static uint64_t PADDING_ALIGN_SIZE = 32;
const static uint32_t BLOCK_L0_N = 16;
const static uint32_t BLOCK_L0_M = 16;
const static uint32_t BT_BLOCK_SIZE = 32;
const static uint32_t L0_SYBC_DB_CLOSE = 0x0;
const static uint32_t L0_SYNC_DB_OPEN = 0x1;
const static uint8_t C04_CIN_SIZE  = 4;
const static uint8_t ROUND_MODE_RINT = 1;
const static uint8_t ROUND_MODE_ROUND = 2;
const static uint8_t ROUND_MODE_HYBRID = 3;
const static uint8_t DOUBLE_BUF = 2;
const static uint8_t BIT_OFFSET_8 = 8;
// Opt Group
const static uint16_t REG_SIZE = 256;
const static uint16_t CO0_LOOP_TIMES = 2;
const static uint16_t B8_CO0_LOOP_TIMES = 4;
const static uint8_t VEC_NUM = 2;
const static uint8_t VEC_ID_MAX = 16;
const static uint8_t CV_SYNC_ID_MTE3_MTE1 = 0;
const static uint8_t CV_SYNC_ID_MTE1_MTE3 = 1;
const static uint8_t CV_SYUC_ID_V_ROTATE = 2;
const static uint8_t NDDMA_DIMS_BASE = 1;
const static uint8_t NDDMA_DIMS_NO_TRANS = 2;
const static uint8_t NDDMA_DIMS = 3;
const static uint8_t NDDMA_DIMS_LOAD_FMAP = 3;
const static uint8_t NDDMA_HWC_DIMS = 4;
const static uint8_t NDDMA_LOOP0_INDEX = 0;
const static uint8_t NDDMA_LOOP1_INDEX = 1;
const static uint8_t NDDMA_LOOP2_INDEX = 2;
const static uint8_t NDDMA_LOOP3_INDEX = 3;
const static uint8_t CV_ENHANCE_MODE = 4;

// Load3d parameters offset
const static uint8_t MSTEP_OFFSET = 16;
const static uint8_t POSM_OFFSET = 48;
const static uint8_t POSK_OFFSET = 32;
const static uint8_t STRIDEH_OFFSET = 6;
const static uint8_t KERNELW_OFFSET = 12;
const static uint8_t KERNELH_OFFSET = 20;
const static uint8_t KERNELW_HIGHEST_BIT_OFFSET = 36;
const static uint8_t KERNELH_HIGHEST_BIT_OFFSET = 37;
const static uint8_t DILATIONW_OFFSET = 28;
const static uint8_t DILATIONH_OFFSET = 36;
const static uint8_t CIN_OFFSET = 48;

// Load2dv2 parameters offset
const static uint8_t DST_STRIDE_OFFSET = 16;
const static uint8_t K_START_OFFSET = 16;
const static uint8_t M_STEP_OFFSET = 32;
const static uint8_t K_STEP_OFFSET = 40;
 
// parameters mask value
const static uint64_t MASK_16 = 0xffff;
const static uint64_t MASK_8 = 0xff;
const static uint64_t MASK_6 = 0x3f;
const static uint64_t NINTH_BIT_MASK = 0x100;

// Dn2NzParam's max nValue
const static uint16_t N_VALUE_MAX = 65535;

const static uint64_t DEQ_SCALAR_ONE = 1065353216;
static constexpr IsResetLoad3dConfig CONV_LOAD3DV2_DEFAULT_CONFIG = {false, false};

constexpr FixpipeConfig CFG_COLUMN_MAJOR_FIXED_POINT = {
    CO2Layout::COLUMN_MAJOR,
    false,
};
constexpr FixpipeConfig CFG_ROW_MAJOR_FIXED_POINT = {
    CO2Layout::ROW_MAJOR,
    false,
};

#define ASCEND_IS_AIC_CONV ASCEND_IS_AIC
#define ASCEND_IS_AIV_CONV ASCEND_IS_AIV

// unit flag mode
constexpr uint8_t UNIT_FLAG_DISABLE = 0;
constexpr uint8_t UNIT_FLAG_RESERVED = 1;
// Enable MAD to be writable or fixpipe readable,
// and write/read can still be done after completion.
constexpr uint8_t UNIT_FLAG_ENABLE_ONLY = 2;
// Enable MAD to be writable or fixpipe readable,
// and flip the read-write status after writing/reading ends.
constexpr uint8_t UNIT_FLAG_ENABLE_WITH_FLIP = 3;

static __aicore__ inline uint64_t AlignB(uint64_t a, uint64_t b)
{
    return ((a + b - 1) / b) * b;
}

static __aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    return (a + b - 1) / b;
}

enum class QuantModeType : std::uint8_t {
    NO_QUANT = 0,
    SCALAR_QUANT,
    VECTOR_QUANT
};

}

#endif  // CONV_UTIL_H