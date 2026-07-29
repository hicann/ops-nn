/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file apply_gradient_descent_tiling_data.h
 * \brief apply_gradient_descent classic (ascend910b) tiling data struct.
 *        This plain struct is shared by the host tiling, the device kernel and the
 *        kernel UT. It is intentionally independent from the top-level ascend950
 *        (arch35) ApplyGradientDescent tiling data (distinct OpType, never co-compiled).
 */

#ifndef APPLY_GRADIENT_DESCENT_TILING_DATA_H
#define APPLY_GRADIENT_DESCENT_TILING_DATA_H

#include <cstdint>

struct ApplyGradientDescentTilingData {
    uint64_t totalDataCount; // total element count of var/delta/out
    uint64_t tileDataCount;  // max elements processed in one UB tile (block aligned)
    uint64_t blocksPerCore;  // base number of aligned blocks handled by each used core
    uint32_t needCoreNum;    // number of cores that participate in the computation
    uint32_t blockElems;     // number of elements in one aligned distribution block
    uint32_t remCoreNum;     // first remCoreNum cores handle one extra block
    uint32_t reserved;       // padding, keep struct size a multiple of 8 bytes
};

#endif // APPLY_GRADIENT_DESCENT_TILING_DATA_H
