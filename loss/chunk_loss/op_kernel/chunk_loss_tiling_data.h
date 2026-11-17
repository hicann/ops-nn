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
 * \file chunk_loss_tiling_data.h
 * \brief Kernel-side tiling mirror of op_host/chunk_loss_tiling.h (arch22 / arch35 shared).
 */
#ifndef CHUNK_LOSS_TILING_DATA_H_
#define CHUNK_LOSS_TILING_DATA_H_

#include "kernel_tiling/kernel_tiling.h"

#ifndef __CCE_KT_KERNEL__
#include <cstdint>
#endif

struct ChunkLossTilingData {
    uint32_t m;
    uint32_t h;
    uint32_t v;
    uint32_t aicNum;
    uint32_t nSplit;
    uint32_t deterministic;
    uint32_t tileM;
    float alpha;
    int64_t ignoreIndex;
    uint64_t syncBaseAddr;
};

#endif
