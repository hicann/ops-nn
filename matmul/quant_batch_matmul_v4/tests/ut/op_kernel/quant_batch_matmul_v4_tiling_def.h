/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __QUANT_BATCH_MATMUL_V4_TILING_DEF_H__
#define __QUANT_BATCH_MATMUL_V4_TILING_DEF_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"
#include "quant_batch_matmul_v4_tiling_data.h"

#ifdef __CCE_KT_TEST__
#include "kernel_log.h"
#else
#define __aicore__ [aicore]
#endif


#if defined(__CCE_KT_TEST__)
template <class T>
inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, T *tilingdata)
#else
template <class T>
__inline__ __attribute__((always_inline)) __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, T *tilingdata)
#endif
{
    constexpr uint64_t all_bytes = sizeof(T);
#if defined(__CCE_KT_TEST__) || defined(__DAV_C220_CUBE__) || defined(__DAV_C310_CUBE__) || defined(__GET_CODE_CHANNEL__)
#if defined(__DAV_C100__) || defined(__CCE_KT_TEST__)
    constexpr uint32_t judge_bytes = all_bytes > 15 ? all_bytes - 15 : 0;
    uint32_t i = 0;
    if (judge_bytes > 0) {
        for (; i < judge_bytes; i += 16) {
            (*(uint64_t*)((uint8_t*)tilingdata + i)) = (*(const __gm__ uint64_t*)((const __gm__ uint8_t *)p_tilingdata + i));
            (*(uint64_t*)((uint8_t*)tilingdata + i + 8)) = (*(const __gm__ uint64_t*)((const __gm__ uint8_t *)p_tilingdata + i + 8));
        }
    }
    if (all_bytes & 0x00000008) {
        (*(uint64_t*)((uint8_t*)tilingdata + i)) = (*(const __gm__ uint64_t *)((const __gm__ uint8_t *)p_tilingdata + i));
        i += 8;
    }
    if (all_bytes & 0x00000004) {
        (*(uint32_t*)((uint8_t*)tilingdata + i)) = (*(const __gm__ uint32_t *)((const __gm__ uint8_t *)p_tilingdata + i));
        i += 4;
    }
    if (all_bytes & 0x00000002) {
        (*(uint16_t*)((uint8_t*)tilingdata + i)) = (*(const __gm__ uint16_t *)((const __gm__ uint8_t *)p_tilingdata + i));
        i += 2;
    }
    if (all_bytes & 0x00000001) {
        (*(uint8_t*)((uint8_t*)tilingdata + i)) = (*(const __gm__ uint8_t *)((const __gm__ uint8_t *)p_tilingdata + i));
    }
#else
    copy_data_align64((uint8_t*)tilingdata, (__gm__ uint8_t *)p_tilingdata, all_bytes);
#endif
#else
    LocalTensor<uint8_t> tilingDataInUb;
    GlobalTensor<uint8_t> tilingDataInGm;
    tilingDataInGm.SetGlobalBuffer((__gm__ uint8_t *)p_tilingdata);
    tilingDataInUb.InitBuffer(0, 192 * 1024); // 192 * 1024是UB大小
    tilingDataInUb.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
    __ubuf__ uint8_t *tilingdata_in_ub = (__ubuf__ uint8_t *)tilingDataInUb.GetPhyAddr();
    constexpr uint32_t len_burst = (all_bytes + 31) / 32;
#if defined(__DAV_C310__)
    DataCopyPad(tilingDataInUb, tilingDataInGm, {1, len_burst * 32, 0, 0}, {false, 0, 0, 0});
#else
    DataCopy(tilingDataInUb, tilingDataInGm, {1, len_burst, 0, 0});
#endif
    SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
#if defined(__DAV_C100__)
    constexpr uint32_t judge_bytes = all_bytes > 15 ? all_bytes - 15 : 0;
    uint32_t i = 0;
    if (judge_bytes > 0) {
        for (; i < judge_bytes; i += 16) {
            (*(uint64_t*)((uint8_t*)tilingdata + i)) = (*(__ubuf__ uint64_t*)((__ubuf__ uint8_t *)tilingdata_in_ub + i));
            (*(uint64_t*)((uint8_t*)tilingdata + i + 8)) = (*(__ubuf__ uint64_t*)((__ubuf__ uint8_t *)tilingdata_in_ub + i + 8));
        }
    }
    if (all_bytes & 0x00000008) {
        (*(uint64_t*)((uint8_t*)tilingdata + i)) = (*(__ubuf__ uint64_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + i));
        i += 8;
    }
    if (all_bytes & 0x00000004) {
        (*(uint32_t*)((uint8_t*)tilingdata + i)) = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + i));
        i += 4;
    }
    if (all_bytes & 0x00000002) {
        (*(uint16_t*)((uint8_t*)tilingdata + i)) = (*(__ubuf__ uint16_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + i));
        i += 2;
    }
    if (all_bytes & 0x00000001) {
        (*(uint8_t*)((uint8_t*)tilingdata + i)) = (*(__ubuf__ uint8_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + i));
    }
#else
    copy_data_align64((uint8_t*)tilingdata, (__ubuf__ uint8_t *)tilingdata_in_ub, all_bytes);
#endif
#endif
#ifndef __CCE_KT_TEST__
    PipeBarrier<PIPE_ALL>();
#endif
}
#define GET_TILING_DATA(tiling_data, tiling_arg) \
    QuantBatchMatmulV4TilingData tiling_data; \
    InitTilingData<QuantBatchMatmulV4TilingData>(tiling_arg, &tiling_data);


#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data; \
    InitTilingData<tiling_struct>(tiling_arg, &tiling_data);


#define GET_TILING_DATA_MEMBER(tiling_type, member, var, tiling) \
    auto typeVar##var = ((tiling_type *)0)->member;                   \
    decltype(typeVar##var) var;                                       \
    size_t offset##var = (size_t)(&((tiling_type*)0)->member);        \
    InitTilingData<decltype(typeVar##var)>(tiling + offset##var, &var);


#endif