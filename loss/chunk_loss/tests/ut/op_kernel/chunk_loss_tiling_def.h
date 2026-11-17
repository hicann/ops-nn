/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHUNK_LOSS_TILING_DEF_H_
#define CHUNK_LOSS_TILING_DEF_H_

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#define __CCE_UT_TEST__

#ifndef DTYPE_HIDDEN
#define DTYPE_HIDDEN half
#endif

// Mirror op_kernel/chunk_loss_tiling_data.h so force-include wins and later include is skipped.
#ifndef CHUNK_LOSS_TILING_DATA_H_
#define CHUNK_LOSS_TILING_DATA_H_
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

template <typename T>
inline void InitChunkLossTilingData(uint8_t* tiling, T* const_data)
{
    memcpy(const_data, tiling, sizeof(T));
}

#ifndef GET_TILING_DATA_WITH_STRUCT
#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitChunkLossTilingData<tiling_struct>(tiling_arg, &tiling_data)
#endif

#ifndef REGISTER_TILING_DEFAULT
#define REGISTER_TILING_DEFAULT(tilingStruct)
#endif

#endif
