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
 * \file conv3d_backprop_filter_v2_common.h
 * \brief
 */
#ifndef CONV3D_DW_V2_COMMON_H
#define CONV3D_DW_V2_COMMON_H

namespace Ops {
namespace NN {
namespace Conv {
constexpr int32_t BLOCK_CUBE = static_cast<int32_t>(AscendC::BLOCK_CUBE);
constexpr uint32_t FP32_DATA_SIZE = 4;
constexpr uint32_t HIF8_DATA_SIZE = 1;
constexpr uint32_t C04_COUT_SIZE = 4;

constexpr uint32_t DB_ON = 2;
constexpr uint32_t DB_OFF = 1;
constexpr int32_t VECTOR_REG_WIDTH = 256;
constexpr int32_t ONE_BLOCK_SIZE = 32;
constexpr uint32_t BASIC_BLOCK_SIZE_512 = 512;
constexpr uint32_t BASIC_BLOCK_SIZE_256 = 256;
constexpr uint32_t BASIC_BLOCK_SIZE_128 = 128;
constexpr uint32_t BASIC_BLOCK_SIZE_64 = 64;
constexpr uint32_t BASIC_BLOCK_SIZE_32 = 32;

constexpr uint64_t KERNEL_HW_1 = 1;
constexpr uint64_t KERNEL_HW_4 = 4;
constexpr uint64_t KERNEL_HW_9 = 9;
constexpr uint64_t KERNEL_HW_16 = 16;
constexpr uint32_t NUM_HALF = 2;
constexpr uint32_t L1_DEPTH_16 = 16;
constexpr uint32_t L1_DEPTH_2 = 2;
constexpr uint32_t STEP_2 = 2;
constexpr uint32_t OUT_ALIGN_BYTE = 8;

constexpr int32_t SPLIT_WO_THRESHOLD = 512;
constexpr int32_t SPLIT_WO_SIZE = 128;

struct PlatformCompileInfo {
    platform_ascendc::SocVersion socVersion;
    uint32_t core_num = 1;
    uint64_t l0a_size = 0;
    uint64_t l0b_size = 0;
    uint64_t l0c_size = 0;
    uint64_t l1_size  = 0;
    uint64_t ub_size  = 0;
};
}
}
}
#endif  // CONV3D_DW_V2_COMMON_H