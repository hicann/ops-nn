/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __CONV3D_BACKPROP_FILTER_V2_TILING_H__
#define __CONV3D_BACKPROP_FILTER_V2_TILING_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#ifdef __CCE_KT_TEST__
#include "kernel_log.h"
#else
#define __aicore__ [aicore]
#endif
#pragma pack(1)
struct TConv3DDwTiling {
    uint32_t batch = 0;
    uint32_t cin = 0;
    uint32_t cout = 0;
    uint32_t cin1G = 0;
    uint32_t cout1G = 0;
    uint32_t dout = 0;
    uint32_t ho = 0;
    uint32_t wo = 0;
    uint32_t di = 0;
    uint32_t hi = 0;
    uint32_t wi = 0;
    uint32_t dk = 0;
    uint32_t hk = 0;
    uint32_t wk = 0;
    uint32_t group = 0;
    uint32_t strideD = 0;
    uint32_t strideH = 0;
    uint32_t strideW = 0;
    uint32_t padFront = 0;
    uint32_t padBack = 0;
    uint32_t padUp = 0;
    uint32_t padDown = 0;
    uint32_t padLeft = 0;
    uint32_t padRight = 0;
    uint32_t dilationD = 0;
    uint32_t dilationH = 0;
    uint32_t dilationW = 0;
    uint32_t channelSize = 0;
    uint32_t al0Pbuffer = 0;
    uint32_t bl0Pbuffer = 0;
    uint32_t cl0Pbuffer = 0;
    uint32_t al1Pbuffer = 0;
    uint32_t bl1Pbuffer = 0;
    uint32_t baseM = 0;
    uint32_t baseK = 0;
    uint32_t baseN = 0;
    uint32_t m0 = 0;
    uint32_t k0 = 0;
    uint32_t n0 = 0;
    uint32_t stepM = 0;
    uint32_t stepN = 0;
    uint32_t stepKa = 0;
    uint32_t stepKb = 0;
    uint32_t iterateOrder = 0;
    uint32_t bl1Bound = 0;
    uint32_t hf32Flag = 0;
    uint32_t singleCoreDk = 0;
    uint32_t singleCoreGroup = 0;
    uint32_t singleCoreCout = 0;
    uint32_t singleCoreHo = 0;
    uint64_t singleCoreBatch = 0;
    uint64_t singleCoreCin = 0;
    uint32_t singleCoreM = 0;
    uint32_t singleCoreN = 0;
    uint32_t singleCoreK = 0;
    uint32_t usedCoreNum = 0;
}__attribute__((__may_alias__));
#pragma pack()

#pragma pack(1)
struct Conv3DBackpropFilterV2Params {
    uint64_t batchDim = 0;
    uint32_t groupDim = 0;
    uint32_t mDim = 0;
    uint32_t kDim = 0;
    uint32_t nDim = 0;
    uint32_t dkDim = 0;
    uint32_t totalL1Size = 0;
}__attribute__((__may_alias__));
#pragma pack()

#pragma pack(1)
struct TConv3DDwBasicBlockTiling {
    uint32_t coreBindOrder = 0;
    uint32_t usedCoreNum = 0;
    uint32_t singleCoreM = 0;
    uint32_t singleCoreN = 0;
    uint32_t singleCoreK = 0;
}__attribute__((__may_alias__));
#pragma pack()

#pragma pack(1)
struct Conv3DBackpropFilterV2TilingData {
    Conv3DBackpropFilterV2Params params;
    TConv3DDwTiling dwTiling;
    TConv3DDwBasicBlockTiling basicBlockTiling;
}__attribute__((__may_alias__));
#pragma pack()

#if defined(__CCE_KT_TEST__)
template <class T>
inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, T *tilingdata)
#else
template <class T>
__inline__ __attribute__((always_inline)) __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, T *tilingdata)
#endif
{
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_CUBE__)
    constexpr uint32_t all_bytes = sizeof(T);
    constexpr uint32_t judge_bytes = all_bytes > 15 ? all_bytes - 15 : 0;
    uint32_t i = 0;
    for (; i < judge_bytes; i += 16) {
        (*(uint64_t*)((uint8_t*)tilingdata + i)) = (*(const __gm__ uint64_t*)((const __gm__ uint8_t *) p_tilingdata + i));
        (*(uint64_t*)((uint8_t*)tilingdata + i + 8)) = (*(const __gm__ uint64_t*)((const __gm__ uint8_t *) p_tilingdata + i + 8));
    }
    if (all_bytes & 0x00000008) {
        (*(uint64_t*)((uint8_t*)tilingdata + i)) = (*(const __gm__ uint64_t *)((const __gm__ uint8_t *) p_tilingdata + i));
        i += 8;
    }
    if (all_bytes & 0x00000004) {
        (*(uint32_t*)((uint8_t*)tilingdata + i)) = (*(const __gm__ uint32_t *)((const __gm__ uint8_t *) p_tilingdata + i));
        i += 4;
    }
    if (all_bytes & 0x00000002) {
        (*(uint16_t*)((uint8_t*)tilingdata + i)) = (*(const __gm__ uint16_t *)((const __gm__ uint8_t *) p_tilingdata + i));
        i += 2;
    }
    if (all_bytes & 0x00000001) {
        (*(uint8_t*)((uint8_t*)tilingdata + i)) = (*(const __gm__ uint8_t *)((const __gm__ uint8_t *) p_tilingdata + i));
    }
#endif
}
#define GET_TILING_DATA(tiling_data, tiling_arg) \
    Conv3DBackpropFilterV2TilingData tiling_data; \
    InitTilingData<Conv3DBackpropFilterV2TilingData>(tiling_arg, &tiling_data);


#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data; \
    InitTilingData<tiling_struct>(tiling_arg, &tiling_data);


#define GET_TILING_DATA_MEMBER(tiling_type, member, var, tiling) \
    auto typeVar##var = ((tiling_type *)0)->member;                   \
    decltype(typeVar##var) var;                                       \
    size_t offset##var = (size_t)(&((tiling_type*)0)->member);        \
    InitTilingData<decltype(typeVar##var)>(tiling + offset##var, &var);


#endif