/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __RELU_V3_HEADER__
#define __RELU_V3_HEADER__

#include "kernel_tiling/kernel_tiling.h"
#ifdef ASCENDC_CPU_DEBUG
#include "kernel_log.h"
#else
#ifndef __aicore__
#define __aicore__ [aicore]
#endif
#endif
#define REGISTER_TILINGDATA_SIZE(tiling_struct, counter)
#if defined(ASCENDC_CPU_DEBUG)
template <class T>
inline __aicore__ void InitTilingData(const __gm__ uint8_t* p_tilingdata, T* tilingdata)
#else
template <class T>
__inline__ __attribute__((always_inline)) __aicore__ void InitTilingData(const __gm__ uint8_t* p_tilingdata,
                                                                         T* tilingdata)
#endif
{
    constexpr uint64_t all_bytes = sizeof(T);
#if defined(ASCENDC_CPU_DEBUG) || (defined(__DAV_CUBE__) && __NPU_ARCH__ == 2201) || \
    (defined(__DAV_CUBE__) && __NPU_ARCH__ == 3101) || defined(__GET_CODE_CHANNEL__)
#if defined(__DAV_C100__) || defined(ASCENDC_CPU_DEBUG)
    constexpr uint32_t judge_bytes = all_bytes > 15 ? all_bytes - 15 : 0;
    uint32_t i = 0;
    if (judge_bytes > 0) {
        for (; i < judge_bytes; i += 16) {
            (*(uint64_t*)((uint8_t*)tilingdata + i)) = (*(const __gm__ uint64_t*)((const __gm__ uint8_t*)p_tilingdata +
                                                                                  i));
            (*(uint64_t*)((uint8_t*)tilingdata + i +
                          8)) = (*(const __gm__ uint64_t*)((const __gm__ uint8_t*)p_tilingdata + i + 8));
        }
    }
    if (all_bytes & 0x00000008) {
        (*(uint64_t*)((uint8_t*)tilingdata + i)) = (*(const __gm__ uint64_t*)((const __gm__ uint8_t*)p_tilingdata + i));
        i += 8;
    }
    if (all_bytes & 0x00000004) {
        (*(uint32_t*)((uint8_t*)tilingdata + i)) = (*(const __gm__ uint32_t*)((const __gm__ uint8_t*)p_tilingdata + i));
        i += 4;
    }
    if (all_bytes & 0x00000002) {
        (*(uint16_t*)((uint8_t*)tilingdata + i)) = (*(const __gm__ uint16_t*)((const __gm__ uint8_t*)p_tilingdata + i));
        i += 2;
    }
    if (all_bytes & 0x00000001) {
        (*(uint8_t*)((uint8_t*)tilingdata + i)) = (*(const __gm__ uint8_t*)((const __gm__ uint8_t*)p_tilingdata + i));
    }
#else
    copy_data_align64((uint8_t*)tilingdata, (__gm__ uint8_t*)p_tilingdata, all_bytes);
#endif
#else
#if __NPU_ARCH__ == 3101 && defined(__ASCENDC_ENABLE_VEC_TAIL_TILING_COPY__)
    constexpr uint32_t DC_PRLOAD_LOOP = (all_bytes) / 512;
    for (uint64_t loop_dc = 0; loop_dc < DC_PRLOAD_LOOP; loop_dc++) {
        uint64_t offset = loop_dc * 512;
        dc_preload((uint64_t*)tilingdata, offset);
    }
    uint64_t tiling_offset = ((uint64_t)(&tilingdata)) % 64;
    uint32_t DC_PRLOAD_LOOP1 = (all_bytes + 63 + tiling_offset) / 64;
    for (uint64_t loop_dc = 0; loop_dc < DC_PRLOAD_LOOP1; loop_dc++) {
        uint64_t offset = loop_dc * 64;
        dc_preload((uint64_t*)tilingdata, offset);
    }

    __ubuf__ uint8_t* tilingdata_in_ub = (__ubuf__ uint8_t*)get_imm(248 * 1024);
    constexpr uint32_t copy_tile_size = 8192;
    for (uint32_t stack_offset = 0; stack_offset < all_bytes; stack_offset += copy_tile_size) {
        uint32_t copy_size = stack_offset + copy_tile_size < all_bytes ? copy_tile_size : all_bytes - stack_offset;
        copy_gm_to_ubuf_align_v2(((__ubuf__ uint8_t*)tilingdata_in_ub), ((__gm__ uint8_t*)p_tilingdata) + stack_offset,
                                 0, 1, copy_size, 0, 0, false, 0, 0, 0);
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        copy_data_align64(((uint8_t*)tilingdata) + stack_offset, ((__ubuf__ uint8_t*)tilingdata_in_ub), copy_size);
        set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    }
#else
    __ubuf__ uint8_t* tilingdata_in_ub = (__ubuf__ uint8_t*)get_imm(0);
    constexpr uint32_t len_burst = (all_bytes + 31) / 32;
#if __NPU_ARCH__ == 3101 || __NPU_ARCH__ == 5102
    copy_gm_to_ubuf_align_v2((__ubuf__ uint8_t*)tilingdata_in_ub, (__gm__ uint8_t*)p_tilingdata, 0, 1, len_burst * 32,
                             0, 0, false, 0, 0, 0);
    constexpr uint32_t DC_PRLOAD_LOOP = (all_bytes) / 512;
    for (uint64_t loop_dc = 0; loop_dc < DC_PRLOAD_LOOP; loop_dc++) {
        uint64_t offset = loop_dc * 512;
        dc_preload((uint64_t*)tilingdata, offset);
    }
    uint64_t tiling_offset = ((uint64_t)(&tilingdata)) % 64;
    uint32_t DC_PRLOAD_LOOP1 = (all_bytes + 63 + tiling_offset) / 64;
    for (uint64_t loop_dc = 0; loop_dc < DC_PRLOAD_LOOP1; loop_dc++) {
        uint64_t offset = loop_dc * 64;
        dc_preload((uint64_t*)tilingdata, offset);
    }

#elif __NPU_ARCH__ == 3103 || __NPU_ARCH__ == 3003
    copy_gm_to_ubuf(((__ubuf__ void*)tilingdata_in_ub), (__gm__ void*)p_tilingdata, 0, 1, len_burst, 0, 0);
#elif __NPU_ARCH__ == 3113
    copy_gm_to_ubuf_align_v2((__ubuf__ uint8_t*)tilingdata_in_ub, (__gm__ uint8_t*)p_tilingdata, 0, 1, len_burst * 32,
                             0, 0, false, 0, 0);
#elif __NPU_ARCH__ != 3102
    copy_gm_to_ubuf(((__ubuf__ uint8_t*)tilingdata_in_ub), p_tilingdata, 0, 1, len_burst, 0, 0);
#else
    copy_gm_to_ubuf_align(((__ubuf__ uint8_t*)tilingdata_in_ub), (__gm__ uint8_t*)p_tilingdata, 0, 1, all_bytes, 0, 0,
                          0, 0);
#endif
    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
#if defined __DAV_C100__
    constexpr uint32_t judge_bytes = all_bytes > 15 ? all_bytes - 15 : 0;
    uint32_t i = 0;
    if (judge_bytes > 0) {
        for (; i < judge_bytes; i += 16) {
            (*(uint64_t*)((uint8_t*)tilingdata + i)) = (*(__ubuf__ uint64_t*)((__ubuf__ uint8_t*)tilingdata_in_ub + i));
            (*(uint64_t*)((uint8_t*)tilingdata + i + 8)) = (*(__ubuf__ uint64_t*)((__ubuf__ uint8_t*)tilingdata_in_ub +
                                                                                  i + 8));
        }
    }
    if (all_bytes & 0x00000008) {
        (*(uint64_t*)((uint8_t*)tilingdata + i)) = (*(__ubuf__ uint64_t*)((__ubuf__ uint8_t*)tilingdata_in_ub + i));
        i += 8;
    }
    if (all_bytes & 0x00000004) {
        (*(uint32_t*)((uint8_t*)tilingdata + i)) = (*(__ubuf__ uint32_t*)((__ubuf__ uint8_t*)tilingdata_in_ub + i));
        i += 4;
    }
    if (all_bytes & 0x00000002) {
        (*(uint16_t*)((uint8_t*)tilingdata + i)) = (*(__ubuf__ uint16_t*)((__ubuf__ uint8_t*)tilingdata_in_ub + i));
        i += 2;
    }
    if (all_bytes & 0x00000001) {
        (*(uint8_t*)((uint8_t*)tilingdata + i)) = (*(__ubuf__ uint8_t*)((__ubuf__ uint8_t*)tilingdata_in_ub + i));
    }
#else
    copy_data_align64((uint8_t*)tilingdata, (__ubuf__ uint8_t*)tilingdata_in_ub, all_bytes);
#endif
#endif // __ASCENDC_ENABLE_VEC_TAIL_TILING_COPY__
#endif
#ifndef ASCENDC_CPU_DEBUG
    pipe_barrier(PIPE_ALL);
#endif
}

#ifdef ASCENDC_TIME_STAMP_ON
#define GET_TILING_DATA(tiling_data, tiling_arg)                         \
    TestUtDefaultTilingStruct tiling_data;                               \
    InitTilingData<TestUtDefaultTilingStruct>(tiling_arg, &tiling_data); \
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_TILING_DATA));

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    REGISTER_TILINGDATA_SIZE(tiling_struct, __COUNTER__);                   \
    tiling_struct tiling_data;                                              \
    InitTilingData<tiling_struct>(tiling_arg, &tiling_data);                \
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_TILING_DATA_STRUCT));

#define GET_TILING_DATA_MEMBER(tiling_type, member, var, tiling)                     \
    REGISTER_TILINGDATA_SIZE(tiling_type, __COUNTER__);                              \
    decltype(((tiling_type*)0)->member) var;                                         \
    size_t offset##var = (size_t)(&((tiling_type*)0)->member);                       \
    InitTilingData<decltype(((tiling_type*)0)->member)>(tiling + offset##var, &var); \
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_TILING_DATA_MEMBER));

#else
#define GET_TILING_DATA(tiling_data, tiling_arg) \
    TestUtDefaultTilingStruct tiling_data;       \
    InitTilingData<TestUtDefaultTilingStruct>(tiling_arg, &tiling_data);
#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    REGISTER_TILINGDATA_SIZE(tiling_struct, __COUNTER__);                   \
    tiling_struct tiling_data;                                              \
    InitTilingData<tiling_struct>(tiling_arg, &tiling_data);
#define GET_TILING_DATA_MEMBER(tiling_type, member, var, tiling) \
    REGISTER_TILINGDATA_SIZE(tiling_type, __COUNTER__);          \
    decltype(((tiling_type*)0)->member) var;                     \
    size_t offset##var = (size_t)(&((tiling_type*)0)->member);   \
    InitTilingData<decltype(((tiling_type*)0)->member)>(tiling + offset##var, &var);
#endif
// micro attribute for pointer assess data

#define __tiling_data_ptr__ __gm__ const

#define GET_TILING_DATA_PTR_WITH_STRUCT(tiling_struct, dst_ptr, tiling_ptr) \
    REGISTER_TILINGDATA_SIZE(tiling_struct, __COUNTER__);                   \
    __tiling_data_ptr__ tiling_struct* dst_ptr = (__tiling_data_ptr__ tiling_struct*)tiling_ptr;

#define COPY_TILING_WITH_STRUCT(tiling_struct, src_ptr, dst_ptr)                                             \
    tiling_struct __ascendc_var##dst_ptr;                                                                    \
    copy_data_align64((uint8_t*)(&__ascendc_var##dst_ptr), (__gm__ uint8_t*)src_ptr, sizeof(tiling_struct)); \
    const tiling_struct* dst_ptr = &__ascendc_var##dst_ptr;

#define COPY_TILING_WITH_ARRAY(arr_type, arr_count, src_ptr, dst_ptr)                                               \
    arr_type __ascendc_var##dst_ptr[arr_count];                                                                     \
    copy_data_align64((uint8_t*)(&__ascendc_var##dst_ptr), (__gm__ uint8_t*)src_ptr, sizeof(arr_type) * arr_count); \
    const arr_type(*dst_ptr)[arr_count] = (const arr_type(*)[arr_count]) & __ascendc_var##dst_ptr;

#define TestUtDefaultTilingStruct ReluV3TilingData

constexpr int min(int in1, int in2) { return std::min(in1, in2); }

#endif // __RELU_V3_HEADER__
