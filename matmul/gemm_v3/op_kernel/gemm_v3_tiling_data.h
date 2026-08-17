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

enum BiasBroadcastType : uint32_t {
    BIAS_BCAST_NONE = 0,
    BIAS_BCAST_N = 1,      // [N]/[B,1,N], broadcast along M
    BIAS_BCAST_M = 2,      // [M,1]/[B,M,1], broadcast along N
    BIAS_BCAST_SCALAR = 3, // [1]/[B,1,1], scalar broadcast
};

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
    uint32_t biasBroadcastType{BIAS_BCAST_NONE};
    uint32_t reservedBiasBroadcast{0}; // Reserved for future bias-broadcast extensions
    uint64_t cBatchStride{0};          // C stride along Batch, in elements; 0 for broadcast.
    uint64_t cMStride{0};              // C stride along M, in elements; 0 for broadcast.
    uint64_t cNStride{0};              // C stride along N, in elements; 0 for broadcast.
};
#pragma pack(pop)
static_assert(sizeof(GemmV3TilingData) % sizeof(uint64_t) == 0, "GemmV3TilingData must be 8-byte aligned");
#endif // GEMM_V3_TILING_DATA_H
