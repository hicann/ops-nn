/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gemm_v3_tiling_data.h
 * \brief
 */

#ifndef GEMM_V3_TILING_DATA_H
#define GEMM_V3_TILING_DATA_H

#include <cstdint>

#pragma pack(push, 8)
// 8 means 8 bytes aligned
struct alignas(8) GemmV3TilingData {
    uint32_t numBatchA{0};
    uint32_t numBatchB{0};
    uint32_t m{0};
    uint32_t k{0};
    uint32_t n{0};
    uint32_t transA{0};
    uint32_t transB{0};
    uint32_t m0{0};
    uint32_t k0{0};
    uint32_t n0{0};
    uint32_t mLoop{0};
    uint32_t kLoop{0};
    uint32_t nLoop{0};
    uint32_t coreLoop{0};
    uint32_t swizzleCount{0};
    uint32_t tilingKey{0};
    uint32_t blockDim{0};
    uint32_t swizzleDirect{0};
    uint32_t splitk{0};
    uint32_t enShuffleK{0};
    float alpha{0.0f};
    float beta{0.0f};
};
#pragma pack(pop)
#endif // GEMM_V3_TILING_DATA_H