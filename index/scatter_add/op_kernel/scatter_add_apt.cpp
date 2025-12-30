/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_add.cpp
 * \brief
 */

 #include "kernel_operator.h"
 #include "arch35/scatter_add_simt.h"
 #include "arch35/scatter_add_simd.h"
 #include "arch35/scatter_add_simd_support_atomicadd.h"
 #include "arch35/scatter_add_deterministic.h"
 #include "arch35/scatter_add_simt_sort.h"
 #include "arch35/scatter_add_simd_sort_support_atomicadd.h"
 
 using namespace AscendC;

 #define TILING_KEY_0                                0
 #define TILING_KEY_UNSORT_SIMT_ADDR32_SCALAR        10000000003333301000UL
 #define TILING_KEY_UNSORT_SIMT_ADDR32_TENSOR        10000000003333300000UL
 #define TILING_KEY_UNSORT_SIMT_ADDR64_SCALAR        10000000003333301100UL
 #define TILING_KEY_UNSORT_SIMT_ADDR64_TENSOR        10000000003333300100UL
 #define TILING_KEY_SORT_NOCAST_SIMT_ADDR32_SCALAR   10000000003333301001UL
 #define TILING_KEY_SORT_NOCAST_SIMT_ADDR32_TENSOR   10000000003333300001UL
 #define TILING_KEY_SORT_NOCAST_SIMT_ADDR64_SCALAR   10000000003333301101UL
 #define TILING_KEY_SORT_NOCAST_SIMT_ADDR64_TENSOR   10000000003333300101UL
 #define TILING_KEY_SORT_CAST1_SIMT_ADDR32_SCALAR    10000000003333311001UL
 #define TILING_KEY_SORT_CAST1_SIMT_ADDR32_TENSOR    10000000003333310001UL
 #define TILING_KEY_SORT_CAST1_SIMT_ADDR64_SCALAR    10000000003333311101UL
 #define TILING_KEY_SORT_CAST1_SIMT_ADDR64_TENSOR    10000000003333310101UL
 #define TILING_KEY_SORT_CAST2_SIMT_ADDR32_SCALAR    10000000003333321001UL
 #define TILING_KEY_SORT_CAST2_SIMT_ADDR32_TENSOR    10000000003333320001UL
 #define TILING_KEY_SORT_CAST2_SIMT_ADDR64_SCALAR    10000000003333321101UL
 #define TILING_KEY_SORT_CAST2_SIMT_ADDR64_TENSOR    10000000003333320101UL
 #define TILING_KEY_SORT_CAST3_SIMT_ADDR32_SCALAR    10000000003333331001UL
 #define TILING_KEY_SORT_CAST3_SIMT_ADDR32_TENSOR    10000000003333330001UL
 #define TILING_KEY_SORT_CAST3_SIMT_ADDR64_SCALAR    10000000003333331101UL
 #define TILING_KEY_SORT_CAST3_SIMT_ADDR64_TENSOR    10000000003333330101UL
 #define TILING_KEY_UNSORT_SIMD_SCALAR               10000000003333301010UL
 #define TILING_KEY_UNSORT_SIMD_TENSOR               10000000003333300010UL
 #define TILING_KEY_SORT_NOCAST_SIMD_SCALAR          10000000003333301011UL
 #define TILING_KEY_SORT_NOCAST_SIMD_TENSOR          10000000003333300011UL
 #define TILING_KEY_SORT_CAST1_SIMD_SCALAR           10000000003333311011UL
 #define TILING_KEY_SORT_CAST1_SIMD_TENSOR           10000000003333310011UL
 #define TILING_KEY_SORT_CAST2_SIMD_SCALAR           10000000003333321011UL
 #define TILING_KEY_SORT_CAST2_SIMD_TENSOR           10000000003333320011UL
 #define TILING_KEY_SORT_CAST3_SIMD_SCALAR           10000000003333331011UL
 #define TILING_KEY_SORT_CAST3_SIMD_TENSOR           10000000003333330011UL

 using namespace ScatterAdd;
 extern "C" __global__ __aicore__ void scatter_add(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y,
                                                   GM_ADDR workspace, GM_ADDR tiling)
 {
     if (workspace == nullptr) {
         return;
     }
     SetSysWorkspace(workspace);
     GM_ADDR userWs = GetUserWorkspace(workspace);
     if (userWs == nullptr) {
         return;
     }
     GET_TILING_DATA(tilingData, tiling);
     TPipe pipe;
     KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
 
     if (TILING_KEY_IS(TILING_KEY_UNSORT_SIMT_ADDR32_SCALAR)) {
         if constexpr (is_same<uint8_t, DTYPE_VAR>::value) {
             ScatterAddSimt<DTYPE_INDICES, DTYPE_VAR, uint32_t, uint32_t, true> op(tilingData, pipe);
             op.Init(var, indices, updates, userWs);
             op.Process();
         } else if constexpr(is_same<int8_t, DTYPE_VAR>::value) {
             ScatterAddSimt<DTYPE_INDICES, DTYPE_VAR, int32_t, uint32_t, true> op(tilingData, pipe);
             op.Init(var, indices, updates, userWs);
             op.Process();
         } else {
             ScatterAddSimt<DTYPE_INDICES, DTYPE_VAR, DTYPE_VAR, uint32_t, true> op(tilingData, pipe);
             op.Init(var, indices, updates, userWs);
             op.Process();
         }
     } else if (TILING_KEY_IS(TILING_KEY_UNSORT_SIMT_ADDR32_TENSOR)) {
         if (tilingData.isDeterminTemplate) {
             if constexpr (is_same<float, DTYPE_VAR>::value || is_same<half, DTYPE_VAR>::value || is_same<bfloat16_t, DTYPE_VAR>::value) {
                 ScatterAddDeterministicImpl<DTYPE_VAR, DTYPE_INDICES> op(tilingData, pipe);
                 op.Init(var, indices, updates, y, userWs);
                 op.Process();
             }
         } else {
             if constexpr (is_same<uint8_t, DTYPE_VAR>::value) {
                 ScatterAddSimt<DTYPE_INDICES, DTYPE_VAR, uint32_t, uint32_t, false> op(tilingData, pipe);
                 op.Init(var, indices, updates, userWs);
                 op.Process();
             } else if constexpr(is_same<int8_t, DTYPE_VAR>::value) {
                 ScatterAddSimt<DTYPE_INDICES, DTYPE_VAR, int32_t, uint32_t, false> op(tilingData, pipe);
                 op.Init(var, indices, updates, userWs);
                 op.Process();
             } else {
                 ScatterAddSimt<DTYPE_INDICES, DTYPE_VAR, DTYPE_VAR, uint32_t, false> op(tilingData, pipe);
                 op.Init(var, indices, updates, userWs);
                 op.Process();
             }
         }
     } else if (TILING_KEY_IS(TILING_KEY_UNSORT_SIMT_ADDR64_SCALAR)) {
         if constexpr (is_same<uint8_t, DTYPE_VAR>::value) {
             ScatterAddSimt<DTYPE_INDICES, DTYPE_VAR, uint32_t, uint64_t, true> op(tilingData, pipe);
             op.Init(var, indices, updates, userWs);
             op.Process();
         } else if constexpr(is_same<int8_t, DTYPE_VAR>::value) {
             ScatterAddSimt<DTYPE_INDICES, DTYPE_VAR, int32_t, uint64_t, true> op(tilingData, pipe);
             op.Init(var, indices, updates, userWs);
             op.Process();
         } else {
             ScatterAddSimt<DTYPE_INDICES, DTYPE_VAR, DTYPE_VAR, uint64_t, true> op(tilingData, pipe);
             op.Init(var, indices, updates, userWs);
             op.Process();
         }
     } else if (TILING_KEY_IS(TILING_KEY_UNSORT_SIMT_ADDR64_TENSOR)) {
         if (tilingData.isDeterminTemplate) {
             if constexpr (is_same<float, DTYPE_VAR>::value || is_same<half, DTYPE_VAR>::value || is_same<bfloat16_t, DTYPE_VAR>::value) {
                 ScatterAddDeterministicImpl<DTYPE_VAR, DTYPE_INDICES> op(tilingData, pipe);
                 op.Init(var, indices, updates, y, userWs);
                 op.Process();
             }
         } else {
             if constexpr (is_same<uint8_t, DTYPE_VAR>::value) {
                 ScatterAddSimt<DTYPE_INDICES, DTYPE_VAR, uint32_t, uint64_t, false> op(tilingData, pipe);
                 op.Init(var, indices, updates, userWs);
                 op.Process();
             } else if constexpr(is_same<int8_t, DTYPE_VAR>::value) {
                 ScatterAddSimt<DTYPE_INDICES, DTYPE_VAR, int32_t, uint64_t, false> op(tilingData, pipe);
                 op.Init(var, indices, updates, userWs);
                 op.Process();
             } else {
                 ScatterAddSimt<DTYPE_INDICES, DTYPE_VAR, DTYPE_VAR, uint64_t, false> op(tilingData, pipe);
                 op.Init(var, indices, updates, userWs);
                 op.Process();
             }
         }
     } else if (TILING_KEY_IS(TILING_KEY_SORT_NOCAST_SIMT_ADDR32_SCALAR)) {
         if constexpr (is_same<int8_t, DTYPE_VAR>::value || is_same<uint8_t, DTYPE_VAR>::value) {
             return;
         } else {
             ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, DTYPE_INDICES, uint32_t, true, CAST_0> op(tilingData, pipe);
             op.Init(var, indices, updates, userWs);
             op.Process();
         }
     } else if (TILING_KEY_IS(TILING_KEY_SORT_NOCAST_SIMT_ADDR32_TENSOR)) {
         if constexpr (is_same<int8_t, DTYPE_VAR>::value || is_same<uint8_t, DTYPE_VAR>::value) {
             return;
         } else {
             ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, DTYPE_INDICES, uint32_t, false, CAST_0> op(tilingData, pipe);
             op.Init(var, indices, updates, userWs);
             op.Process();
         }
     } else if (TILING_KEY_IS(TILING_KEY_SORT_NOCAST_SIMT_ADDR64_SCALAR)) {
         if constexpr (is_same<int8_t, DTYPE_VAR>::value || is_same<uint8_t, DTYPE_VAR>::value) {
             return;
         } else {
             ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, DTYPE_INDICES, uint64_t, true, CAST_0> op(tilingData, pipe);
             op.Init(var, indices, updates, userWs);
             op.Process();
         }
     } else if (TILING_KEY_IS(TILING_KEY_SORT_NOCAST_SIMT_ADDR64_TENSOR)) {
         if constexpr (is_same<int8_t, DTYPE_VAR>::value || is_same<uint8_t, DTYPE_VAR>::value) {
             return;
         } else {
             ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, DTYPE_INDICES, uint64_t, false, CAST_0> op(tilingData, pipe);
             op.Init(var, indices, updates, userWs);
             op.Process();
         }
     } else if (TILING_KEY_IS(TILING_KEY_SORT_CAST1_SIMT_ADDR32_SCALAR)) {
         if constexpr (is_same<int8_t, DTYPE_VAR>::value || is_same<uint8_t, DTYPE_VAR>::value) {
             return;
         } else {
             ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, int16_t, uint32_t, true, CAST_1> op(tilingData, pipe);
             op.Init(var, indices, updates, userWs);
             op.Process();
         }
     } else if (TILING_KEY_IS(TILING_KEY_SORT_CAST1_SIMT_ADDR32_TENSOR)) {
         if constexpr (is_same<int8_t, DTYPE_VAR>::value || is_same<uint8_t, DTYPE_VAR>::value) {
             return;
         } else {
             ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, int16_t, uint32_t, false, CAST_1> op(tilingData, pipe);
             op.Init(var, indices, updates, userWs);
             op.Process();
         }
     } else if (TILING_KEY_IS(TILING_KEY_SORT_CAST1_SIMT_ADDR64_SCALAR)) {
         if constexpr (is_same<int8_t, DTYPE_VAR>::value || is_same<uint8_t, DTYPE_VAR>::value) {
             return;
         } else {
             ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, int16_t, uint64_t, true, CAST_1> op(tilingData, pipe);
             op.Init(var, indices, updates, userWs);
             op.Process();
         }
     } else if (TILING_KEY_IS(TILING_KEY_SORT_CAST1_SIMT_ADDR64_TENSOR)) {
         if constexpr (is_same<int8_t, DTYPE_VAR>::value || is_same<uint8_t, DTYPE_VAR>::value) {
             return;
         } else {
             ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, int16_t, uint64_t, false, CAST_1> op(tilingData, pipe);
             op.Init(var, indices, updates, userWs);
             op.Process();
         }
     } else if (TILING_KEY_IS(TILING_KEY_SORT_CAST2_SIMT_ADDR32_SCALAR)) {
         if constexpr (is_same<int8_t, DTYPE_VAR>::value || is_same<uint8_t, DTYPE_VAR>::value) {
             return;
         } else {
             ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, int32_t, uint32_t, true, CAST_2> op(tilingData, pipe);
             op.Init(var, indices, updates, userWs);
             op.Process();
         }
     } else if (TILING_KEY_IS(TILING_KEY_SORT_CAST2_SIMT_ADDR32_TENSOR)) {
         if constexpr (is_same<int8_t, DTYPE_VAR>::value || is_same<uint8_t, DTYPE_VAR>::value) {
             return;
         } else {
             ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, int32_t, uint32_t, false, CAST_2> op(tilingData, pipe);
             op.Init(var, indices, updates, userWs);
             op.Process();
         }
     } else if (TILING_KEY_IS(TILING_KEY_SORT_CAST2_SIMT_ADDR64_SCALAR)) {
         if constexpr (is_same<int8_t, DTYPE_VAR>::value || is_same<uint8_t, DTYPE_VAR>::value) {
             return;
         } else {
             ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, int32_t, uint64_t, true, CAST_2> op(tilingData, pipe);
             op.Init(var, indices, updates, userWs);
             op.Process();
         }
     } else if (TILING_KEY_IS(TILING_KEY_SORT_CAST2_SIMT_ADDR64_TENSOR)) {
         if constexpr (is_same<int8_t, DTYPE_VAR>::value || is_same<uint8_t, DTYPE_VAR>::value) {
             return;
         } else {
             ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, int32_t, uint64_t, false, CAST_2> op(tilingData, pipe);
             op.Init(var, indices, updates, userWs);
             op.Process();
         }
     } else if (TILING_KEY_IS(TILING_KEY_SORT_CAST3_SIMT_ADDR32_SCALAR)) {
         if constexpr (is_same<int8_t, DTYPE_VAR>::value || is_same<uint8_t, DTYPE_VAR>::value) {
             return;
         } else {
             ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, int16_t, uint32_t, true, CAST_3> op(tilingData, pipe);
             op.Init(var, indices, updates, userWs);
             op.Process();
         }
     } else if (TILING_KEY_IS(TILING_KEY_SORT_CAST3_SIMT_ADDR32_TENSOR)) {
         if constexpr (is_same<int8_t, DTYPE_VAR>::value || is_same<uint8_t, DTYPE_VAR>::value) {
             return;
         } else {
             ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, int16_t, uint32_t, false, CAST_3> op(tilingData, pipe);
             op.Init(var, indices, updates, userWs);
             op.Process();
         }
     } else if (TILING_KEY_IS(TILING_KEY_SORT_CAST3_SIMT_ADDR64_SCALAR)) {
         if constexpr (is_same<int8_t, DTYPE_VAR>::value || is_same<uint8_t, DTYPE_VAR>::value) {
             return;
         } else {
             ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, int16_t, uint64_t, true, CAST_3> op(tilingData, pipe);
             op.Init(var, indices, updates, userWs);
             op.Process();
         }
     } else if (TILING_KEY_IS(TILING_KEY_SORT_CAST3_SIMT_ADDR64_TENSOR)) {
         if constexpr (is_same<int8_t, DTYPE_VAR>::value || is_same<uint8_t, DTYPE_VAR>::value) {
             return;
         } else {
             ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, int16_t, uint64_t, false, CAST_3> op(tilingData, pipe);
             op.Init(var, indices, updates, userWs);
             op.Process();
         }
     } else if (TILING_KEY_IS(TILING_KEY_UNSORT_SIMD_SCALAR)) {
         if (tilingData.isDeterminTemplate) {
             if constexpr (is_same<float, DTYPE_VAR>::value || is_same<half, DTYPE_VAR>::value || is_same<bfloat16_t, DTYPE_VAR>::value) {
                 ScatterAddDeterministicImpl<DTYPE_VAR, DTYPE_INDICES> op(tilingData, pipe);
                 op.Init(var, indices, updates, y, userWs);
                 op.Process();
             }
         } else {
             ScatterAddSIMDImpl<DTYPE_VAR, DTYPE_INDICES, true> op(tilingData, pipe);
             op.Init(var, indices, updates, y, userWs);
             op.Process();
         }
     } else if (TILING_KEY_IS(TILING_KEY_UNSORT_SIMD_TENSOR)) {
         if (tilingData.isDeterminTemplate) {
             if constexpr (is_same<float, DTYPE_VAR>::value || is_same<half, DTYPE_VAR>::value || is_same<bfloat16_t, DTYPE_VAR>::value) {
                 ScatterAddDeterministicImpl<DTYPE_VAR, DTYPE_INDICES> op(tilingData, pipe);
                 op.Init(var, indices, updates, y, userWs);
                 op.Process();
             }
         } else {
             if constexpr (platform::IsSupportAtomicAddTypeSIMD<DTYPE_VAR>()) {
                 ScatterAddSIMDSupportAtomicAdd<DTYPE_VAR, DTYPE_INDICES, false> op(tilingData, pipe);
                 op.Init(var, indices, updates, y, userWs);
                 op.Process();
             } else {
                 ScatterAddSIMDImpl<DTYPE_VAR, DTYPE_INDICES, false> op(tilingData, pipe);
                 op.Init(var, indices, updates, y, userWs);
                 op.Process();
             }
         }
     } else if (TILING_KEY_IS(TILING_KEY_SORT_NOCAST_SIMD_SCALAR)) {
         if constexpr (platform::IsSupportAtomicAddTypeSIMD<DTYPE_VAR>()) {
             ScatterAddSIMDSortSupportAtomicAdd<DTYPE_VAR, DTYPE_INDICES, DTYPE_INDICES, true, CAST_1> op(tilingData, pipe);
             op.Init(var, indices, updates, y, userWs);
             op.Process();
         } else {
             return;
         }
     } else if (TILING_KEY_IS(TILING_KEY_SORT_NOCAST_SIMD_TENSOR)) {
         if constexpr (platform::IsSupportAtomicAddTypeSIMD<DTYPE_VAR>()) {
             ScatterAddSIMDSortSupportAtomicAdd<DTYPE_VAR, DTYPE_INDICES, DTYPE_INDICES, false, CAST_1> op(tilingData, pipe);
             op.Init(var, indices, updates, y, userWs);
             op.Process();
         } else {
             return;
         }
     } else if (TILING_KEY_IS(TILING_KEY_SORT_CAST1_SIMD_SCALAR)) {
         if constexpr (platform::IsSupportAtomicAddTypeSIMD<DTYPE_VAR>()) {
             ScatterAddSIMDSortSupportAtomicAdd<DTYPE_VAR, DTYPE_INDICES, int16_t, true, CAST_1> op(tilingData, pipe);
             op.Init(var, indices, updates, y, userWs);
             op.Process();
         } else {
             return;
         }
     } else if (TILING_KEY_IS(TILING_KEY_SORT_CAST1_SIMD_TENSOR)) {
         if constexpr (platform::IsSupportAtomicAddTypeSIMD<DTYPE_VAR>()) {
             ScatterAddSIMDSortSupportAtomicAdd<DTYPE_VAR, DTYPE_INDICES, int16_t, false, CAST_1> op(tilingData, pipe);
             op.Init(var, indices, updates, y, userWs);
             op.Process();
         } else {
             return;
         }
     } else if (TILING_KEY_IS(TILING_KEY_SORT_CAST2_SIMD_SCALAR)) {
         if constexpr (platform::IsSupportAtomicAddTypeSIMD<DTYPE_VAR>()) {
             ScatterAddSIMDSortSupportAtomicAdd<DTYPE_VAR, DTYPE_INDICES, int32_t, true, CAST_2> op(tilingData, pipe);
             op.Init(var, indices, updates, y, userWs);
             op.Process();
         } else {
             return;
         }
     } else if (TILING_KEY_IS(TILING_KEY_SORT_CAST2_SIMD_TENSOR)) {
         if constexpr (platform::IsSupportAtomicAddTypeSIMD<DTYPE_VAR>()) {
             ScatterAddSIMDSortSupportAtomicAdd<DTYPE_VAR, DTYPE_INDICES, int32_t, false, CAST_2> op(tilingData, pipe);
             op.Init(var, indices, updates, y, userWs);
             op.Process();
         } else {
             return;
         }
     } else if (TILING_KEY_IS(TILING_KEY_SORT_CAST3_SIMD_SCALAR)) {
         if constexpr (platform::IsSupportAtomicAddTypeSIMD<DTYPE_VAR>()) {
             ScatterAddSIMDSortSupportAtomicAdd<DTYPE_VAR, DTYPE_INDICES, int16_t, true, CAST_3> op(tilingData, pipe);
             op.Init(var, indices, updates, y, userWs);
             op.Process();
         } else {
             return;
         }
     } else if (TILING_KEY_IS(TILING_KEY_SORT_CAST3_SIMD_TENSOR)) {
         if constexpr (platform::IsSupportAtomicAddTypeSIMD<DTYPE_VAR>()) {
             ScatterAddSIMDSortSupportAtomicAdd<DTYPE_VAR, DTYPE_INDICES, int16_t, false, CAST_3> op(tilingData, pipe);
             op.Init(var, indices, updates, y, userWs);
             op.Process();
         } else {
             return;
         }
     } else if (TILING_KEY_IS(TILING_KEY_0)) {
         return;
     }
 }