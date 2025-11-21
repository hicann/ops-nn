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
 * \file conv3d_backprop_input_v2_common.h
 * \brief
 */
#ifndef CONV3D_BACKPROP_INPUT_V2_COMMON_H
#define CONV3D_BACKPROP_INPUT_V2_COMMON_H

namespace Ops {
namespace NN {
namespace Conv {

constexpr uint32_t HIF8_DATA_SIZE = 1;
constexpr uint32_t C04_COUT_SIZE = 4;
constexpr int32_t VECTOR_REG_WIDTH = 256;
constexpr int32_t ONE_BLOCK_SIZE = 32;
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint64_t TWO = 2;
constexpr uint32_t TWO_U32 = 2;
constexpr uint32_t FOUR_U32 = 4;
constexpr uint64_t ONE_U64 = 1;
constexpr uint32_t ONE_U32 = 1;
constexpr int32_t ONE_S32 = 1;
constexpr uint64_t BYTE_64 = 64;
constexpr uint32_t MAX_BASE_MN = 1024;
constexpr uint32_t BASIC_BLOCK_SIZE_512 = 512;
constexpr uint32_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint32_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint32_t BASIC_BLOCK_SIZE_64 = 64;
constexpr uint32_t BASIC_BLOCK_SIZE_32 = 32;

constexpr uint32_t B2_TRANSPOSE_AND_REVERSE = 1;
constexpr uint32_t B2_TRANSPOSE_ONLY = 3;
constexpr uint32_t B2_REVERSE_ONLY = 2;
constexpr uint32_t B2_NO_TRANSPOSE_NO_REVERSE = 0;

struct L0TilingParams {
    uint8_t al0Pbuffer = 1;
    uint8_t bl0Pbuffer = 1;
    uint8_t cl0Pbuffer = 1;
    uint32_t baseM = 16;
    uint32_t baseK = 16;
    uint32_t baseN = 16;
};

struct L1TilingParams {
    uint8_t al1Pbuffer = 1;
    uint8_t bl1Pbuffer = 1;
    uint32_t stepM = 1;
    uint32_t stepN = 1;
    uint32_t stepKa = 1;
    uint32_t stepKb = 1;
    uint8_t iterateOrder = 1;
};

struct CoreTilingParams {
    uint32_t singleCoreDin = 1;
    uint64_t singleCoreM = 1;
    uint32_t singleCoreCout = 16;
    uint64_t singleCoreCin = 16;
};

struct TilingRunInfo {
    uint64_t mValue = 0;
    uint64_t kValue = 0;
    uint64_t nValue = 0;
    uint64_t lenHkWkC0 = 0;
    bool enableC04Flag = false;
    uint32_t m0 = 16;
    uint32_t n0 = 16;
    uint32_t k0 = 16;
    bool enableVecTransFlag = false;
};

} // namespace Conv
} // namespace NN
} // namespace Ops

#endif // CONV3D_BACKPROP_INPUT_V2_COMMON_H