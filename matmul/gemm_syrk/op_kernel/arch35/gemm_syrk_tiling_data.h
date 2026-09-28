/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gemm_syrk_tiling_data.h
 * \brief GemmSyrk flat tiling data (standalone tiling, no batch-matmul struct).
 *
 * The syrk contract forces one symmetric square block: baseM == baseN ==
 * mL1 == nL1 == baseBlock. The kernel wrapper derives the Blaze scheduler /
 * block params from this single size, so the tiling data carries the problem
 * shape, the symmetric block, the K tiling and the two scalars only.
 */

#pragma once

#ifndef __CCE_AICORE__
#include <cstdint>
#endif

#pragma pack(push, 8)
struct alignas(8) GemmSyrkTilingData {
    uint32_t m{0};
    uint32_t n{0}; // == m (syrk: N == M)
    uint32_t k{0};
    uint32_t batch{0};
    uint32_t baseBlock{0}; // baseM == baseN == mL1 == nL1, 16-aligned
    uint32_t baseK{0};     // L0 K step, 16-aligned
    uint32_t kL1{0};       // L1 K chunk (>= baseK), 16-aligned
    uint32_t usedCoreNum{0};
    float alpha{1.0F};
    float beta{1.0F};
};
#pragma pack(pop)
