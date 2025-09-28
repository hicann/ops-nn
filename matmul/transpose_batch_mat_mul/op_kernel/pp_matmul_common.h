/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pp_matmul_common.h
 * \brief
 */

#ifndef __PP_MATMUL_COMMON_H__
#define __PP_MATMUL_COMMON_H__

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace PpMatMulNS {

constexpr uint32_t L0_PINGPONG_BUFFER_SIZE = 16384;
constexpr uint32_t L1_PINGPONG_BUFFER_SIZE = 131072;
constexpr uint32_t CONST_8 = 8;
constexpr uint32_t CONST_16 = 16;
constexpr uint32_t CONST_128 = 128;
constexpr uint32_t CONST_256 = 256;
constexpr uint32_t CONST_512 = 512;
constexpr uint64_t ND2NZ_STRIDE_LIMIT = 65536;
constexpr uint32_t NEXT_TWO_EVENT = 2;
constexpr uint32_t NEXT_TWO_IDX = 2;
constexpr uint64_t BLOCK_SIZE_16 = 16;
constexpr uint64_t CUBE_MATRIX_SIZE_256 = 256;
constexpr uint64_t L1_BIAS_OFFSET = 510 * 1024;
constexpr uint64_t MAX_BT_SIZE = 1024;
constexpr uint64_t L0_PINGPONG_BUFFER_LEN = 32768;
constexpr uint64_t L1_PINGPONG_BUFFER_LEN = 262144;

struct MatCoord {
    uint64_t m{0};
    uint64_t k{0};
    uint64_t n{0};
};

__aicore__ FORCE_INLINE uint64_t RoundUp16(const uint64_t val)
{
    return (val + CONST_16 - 1) / CONST_16 * CONST_16;
}

__aicore__ FORCE_INLINE uint64_t RoundUp256(const uint64_t val)
{
    return (val + CONST_256 - 1) / CONST_256 * CONST_256;
}

__aicore__ FORCE_INLINE uint64_t RoundDown16(const uint64_t val) { return val / CONST_16 * CONST_16; }

__aicore__ FORCE_INLINE uint64_t CeilDiv256(const uint64_t dividend)
{
    return (dividend + CONST_256 - 1) / CONST_256;
}

__aicore__ FORCE_INLINE uint64_t Max(const uint64_t a, const uint64_t b) { return a > b ? a : b; }

}
#endif // __PP_MATMUL_COMMON_H__