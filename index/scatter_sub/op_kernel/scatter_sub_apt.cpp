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
 * \file scatter_sub.cpp
 * \brief scatter_sub
 */

 #include "kernel_operator.h"
 #include "../scatter_add/arch35/scatter_add_simt.h"
 #include "../scatter_add/arch35/scatter_add_simd.h"
 #include "../scatter_add/arch35/scatter_add_simd_support_atomicadd.h"
 #include "../scatter_add/arch35/scatter_add_deterministic.h"
 #include "../scatter_add/arch35/scatter_add_simt_sort.h"
 #include "../scatter_add/arch35/scatter_add_simd_sort_support_atomicadd.h"
 
 using namespace AscendC;

 #define TILING_KEY_0                                0
 #define TILING_KEY_UNSORT_SIMT_ADDR32_SCALAR        10000000000333331000UL
 #define TILING_KEY_UNSORT_SIMT_ADDR32_TENSOR        10000000000333330000UL
 #define TILING_KEY_UNSORT_SIMT_ADDR64_SCALAR        10000000000333331100UL
 #define TILING_KEY_UNSORT_SIMT_ADDR64_TENSOR        10000000000333330100UL
 #define TILING_KEY_SORT_SIMT_ADDR32_SCALAR          10000000000333331001UL
 #define TILING_KEY_SORT_SIMT_ADDR32_TENSOR          10000000000333330001UL
 #define TILING_KEY_SORT_SIMT_ADDR64_SCALAR          10000000000333331101UL
 #define TILING_KEY_SORT_SIMT_ADDR64_TENSOR          10000000000333330101UL
 #define TILING_KEY_UNSORT_SIMD_SCALAR               10000000000333331010UL
 #define TILING_KEY_UNSORT_SIMD_TENSOR               10000000000333330010UL
 #define TILING_KEY_SORT_SIMD_SCALAR                 10000000000333331011UL
 #define TILING_KEY_SORT_SIMD_TENSOR                 10000000000333330011UL

 using namespace ScatterAdd;

 __aicore__ inline void ScatterAddUnsortSimtSubr32Scalar(
    GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR userWs, GM_ADDR tiling, TPipe &pipe)
{
    GET_TILING_DATA(tilingData, tiling);
    if constexpr (is_same<uint8_t, DTYPE_VAR>::value) {
        ScatterAddSimt<DTYPE_INDICES, DTYPE_VAR, uint32_t, uint32_t, true, SUB> op(tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if constexpr(is_same<int8_t, DTYPE_VAR>::value) {
        ScatterAddSimt<DTYPE_INDICES, DTYPE_VAR, int32_t, uint32_t, true, SUB> op(tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else {
        ScatterAddSimt<DTYPE_INDICES, DTYPE_VAR, DTYPE_VAR, uint32_t, true, SUB> op(tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    }
}

 __aicore__ inline void ScatterAddUnsortSimtSubr32Tensor(
    GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR userWs, GM_ADDR tiling, TPipe &pipe)
{
    GET_TILING_DATA(tilingData, tiling);
    if (tilingData.isDeterminTemplate) {
        if constexpr (is_same<half, DTYPE_VAR>::value || is_same<float, DTYPE_VAR>::value || is_same<bfloat16_t, DTYPE_VAR>::value) {
            ScatterAddDeterministicImpl<DTYPE_VAR, DTYPE_INDICES, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, y, userWs);
            op.Process();
        }
    } else {
        if constexpr (is_same<uint8_t, DTYPE_VAR>::value) {
            ScatterAddSimt<DTYPE_INDICES, DTYPE_VAR, uint32_t, uint32_t, false, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        } else if constexpr(is_same<int8_t, DTYPE_VAR>::value) {
            ScatterAddSimt<DTYPE_INDICES, DTYPE_VAR, int32_t, uint32_t, false, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        } else {
            ScatterAddSimt<DTYPE_INDICES, DTYPE_VAR, DTYPE_VAR, uint32_t, false, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        }
    }
}

 __aicore__ inline void ScatterAddUnsortSimtSubr64Scalar(
    GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR userWs, GM_ADDR tiling, TPipe &pipe)
{
    GET_TILING_DATA(tilingData, tiling);
    if constexpr (is_same<uint8_t, DTYPE_VAR>::value) {
        ScatterAddSimt<DTYPE_INDICES, DTYPE_VAR, uint32_t, uint64_t, true, SUB> op(tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if constexpr(is_same<int8_t, DTYPE_VAR>::value) {
        ScatterAddSimt<DTYPE_INDICES, DTYPE_VAR, int32_t, uint64_t, true, SUB> op(tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else {
        ScatterAddSimt<DTYPE_INDICES, DTYPE_VAR, DTYPE_VAR, uint64_t, true, SUB> op(tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    }
}

 __aicore__ inline void ScatterAddUnsortSimtSubr64Tensor(
    GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR userWs, GM_ADDR tiling, TPipe &pipe)
{
    GET_TILING_DATA(tilingData, tiling);
    if (tilingData.isDeterminTemplate) {
        if constexpr (is_same<float, DTYPE_VAR>::value || is_same<half, DTYPE_VAR>::value || is_same<bfloat16_t, DTYPE_VAR>::value) {
            ScatterAddDeterministicImpl<DTYPE_VAR, DTYPE_INDICES, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, y, userWs);
            op.Process();
        }
    } else {
        if constexpr (is_same<uint8_t, DTYPE_VAR>::value) {
            ScatterAddSimt<DTYPE_INDICES, DTYPE_VAR, uint32_t, uint64_t, false, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        } else if constexpr(is_same<int8_t, DTYPE_VAR>::value) {
            ScatterAddSimt<DTYPE_INDICES, DTYPE_VAR, int32_t, uint64_t, false, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        } else {
            ScatterAddSimt<DTYPE_INDICES, DTYPE_VAR, DTYPE_VAR, uint64_t, false, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        }
    }
}

 __aicore__ inline void ScatterAddSortSimtSubr32Scalar(
    GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR userWs, GM_ADDR tiling, TPipe &pipe)
{
    GET_TILING_DATA(tilingData, tiling);
    if constexpr (is_same<int8_t, DTYPE_VAR>::value || is_same<uint8_t, DTYPE_VAR>::value) {
        return;
    } else {
        if (tilingData.indicesCastMode == CAST_0) {
            ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, DTYPE_INDICES, uint32_t, true, CAST_0, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_1) {
            ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, int16_t, uint32_t, true, CAST_1, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_2) {
            ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, int32_t, uint32_t, true, CAST_2, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_3) {
            ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, int16_t, uint32_t, true, CAST_3, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_4) {
            ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, uint8_t, uint32_t, true, CAST_4, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_5) {
            ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, uint8_t, uint32_t, true, CAST_5, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        }
    }
}

 __aicore__ inline void ScatterAddSortSimtSubr32Tensor(
    GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR userWs, GM_ADDR tiling, TPipe &pipe)
{
    GET_TILING_DATA(tilingData, tiling);
    if constexpr (is_same<int8_t, DTYPE_VAR>::value || is_same<uint8_t, DTYPE_VAR>::value) {
        return;
    } else {
        if (tilingData.indicesCastMode == CAST_0) {
            ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, DTYPE_INDICES, uint32_t, false, CAST_0, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_1) {
            ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, int16_t, uint32_t, false, CAST_1, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_2) {
            ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, int32_t, uint32_t, false, CAST_2, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_3) {
            ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, int16_t, uint32_t, false, CAST_3, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_4) {
            ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, uint8_t, uint32_t, false, CAST_4, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_5) {
            ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, uint8_t, uint32_t, false, CAST_5, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        }
    }
}

 __aicore__ inline void ScatterAddSortSimtSubr64Scalar(
    GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR userWs, GM_ADDR tiling, TPipe &pipe)
{
    GET_TILING_DATA(tilingData, tiling);
    if constexpr (is_same<int8_t, DTYPE_VAR>::value || is_same<uint8_t, DTYPE_VAR>::value) {
        return;
    } else {
        if (tilingData.indicesCastMode == CAST_0) {
            ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, DTYPE_INDICES, uint64_t, true, CAST_0, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_1) {
            ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, int16_t, uint64_t, true, CAST_1, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_2) {
            ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, int32_t, uint64_t, true, CAST_2, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_3) {
            ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, int16_t, uint64_t, true, CAST_3, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_4) {
            ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, uint8_t, uint64_t, true, CAST_4, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_5) {
            ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, uint8_t, uint64_t, true, CAST_5, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        }
    }
}

 __aicore__ inline void ScatterAddSortSimtSubr64Tensor(
    GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR userWs, GM_ADDR tiling, TPipe &pipe)
{
    GET_TILING_DATA(tilingData, tiling);
    if constexpr (is_same<int8_t, DTYPE_VAR>::value || is_same<uint8_t, DTYPE_VAR>::value) {
        return;
    } else {
        if (tilingData.indicesCastMode == CAST_0) {
            ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, DTYPE_INDICES, uint64_t, false, CAST_0, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_1) {
            ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, int16_t, uint64_t, false, CAST_1, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_2) {
            ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, int32_t, uint64_t, false, CAST_2, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_3) {
            ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, int16_t, uint64_t, false, CAST_3, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_4) {
            ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, uint8_t, uint64_t, false, CAST_4, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_5) {
            ScatterAddSimtSort<DTYPE_INDICES, DTYPE_VAR, uint8_t, uint64_t, false, CAST_5, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, userWs);
            op.Process();
        }
    }
}

 __aicore__ inline void ScatterAddUnsortSimdScalar(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y,
                                                                 GM_ADDR userWs, GM_ADDR tiling, TPipe &pipe)
{
    GET_TILING_DATA(tilingData, tiling);
    if (tilingData.isDeterminTemplate) {
        if constexpr (is_same<bfloat16_t, DTYPE_VAR>::value || is_same<half, DTYPE_VAR>::value || is_same<float, DTYPE_VAR>::value) {
            ScatterAddDeterministicImpl<DTYPE_VAR, DTYPE_INDICES, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, y, userWs);
            op.Process();
        }
    } else {
        if constexpr (platform::IsSupportAtomicAddTypeSIMD<DTYPE_VAR>()) {
            ScatterAddSIMDSupportAtomicAdd<DTYPE_VAR, DTYPE_INDICES, true, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, y, userWs);
            op.Process();
        } else {
            ScatterAddSIMDImpl<DTYPE_VAR, DTYPE_INDICES, true, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, y, userWs);
            op.Process();
        }
    }
}

 __aicore__ inline void ScatterAddUnsortSimdTensor(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y,
                                                                 GM_ADDR userWs, GM_ADDR tiling, TPipe &pipe)
{
    GET_TILING_DATA(tilingData, tiling);
    if (tilingData.isDeterminTemplate) {
        if constexpr (is_same<float, DTYPE_VAR>::value || is_same<half, DTYPE_VAR>::value || is_same<bfloat16_t, DTYPE_VAR>::value) {
            ScatterAddDeterministicImpl<DTYPE_VAR, DTYPE_INDICES, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, y, userWs);
            op.Process();
        }
    } else {
        if constexpr (platform::IsSupportAtomicAddTypeSIMD<DTYPE_VAR>()) {
            ScatterAddSIMDSupportAtomicAdd<DTYPE_VAR, DTYPE_INDICES, false, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, y, userWs);
            op.Process();
        } else {
            ScatterAddSIMDImpl<DTYPE_VAR, DTYPE_INDICES, false, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, y, userWs);
            op.Process();
        }
    }
}

 __aicore__ inline void ScatterAddSortSimdScalar(
    GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR userWs, GM_ADDR tiling, TPipe &pipe)
{
    GET_TILING_DATA(tilingData, tiling);
    if constexpr (platform::IsSupportAtomicAddTypeSIMD<DTYPE_VAR>()) {
        if (tilingData.indicesCastMode == CAST_0) {
            ScatterAddSIMDSortSupportAtomicAdd<DTYPE_VAR, DTYPE_INDICES, DTYPE_INDICES, true, CAST_0, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, y, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_1) {
            ScatterAddSIMDSortSupportAtomicAdd<DTYPE_VAR, DTYPE_INDICES, int16_t, true, CAST_1, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, y, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_2) {
            ScatterAddSIMDSortSupportAtomicAdd<DTYPE_VAR, DTYPE_INDICES, int32_t, true, CAST_2, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, y, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_3) {
            ScatterAddSIMDSortSupportAtomicAdd<DTYPE_VAR, DTYPE_INDICES, int16_t, true, CAST_3, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, y, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_4) {
            ScatterAddSIMDSortSupportAtomicAdd<DTYPE_VAR, DTYPE_INDICES, uint8_t, true, CAST_4, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, y, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_5) {
            ScatterAddSIMDSortSupportAtomicAdd<DTYPE_VAR, DTYPE_INDICES, uint8_t, true, CAST_5, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, y, userWs);
            op.Process();
        }
    } else {
        return;
    }
}

 __aicore__ inline void ScatterAddSortSimdTensor(
    GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR userWs, GM_ADDR tiling, TPipe &pipe)
{
    GET_TILING_DATA(tilingData, tiling);
    if constexpr (platform::IsSupportAtomicAddTypeSIMD<DTYPE_VAR>()) {
        if (tilingData.indicesCastMode == CAST_0) {
            ScatterAddSIMDSortSupportAtomicAdd<DTYPE_VAR, DTYPE_INDICES, DTYPE_INDICES, false, CAST_0, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, y, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_1) {
            ScatterAddSIMDSortSupportAtomicAdd<DTYPE_VAR, DTYPE_INDICES, int16_t, false, CAST_1, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, y, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_2) {
            ScatterAddSIMDSortSupportAtomicAdd<DTYPE_VAR, DTYPE_INDICES, int32_t, false, CAST_2, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, y, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_3) {
            ScatterAddSIMDSortSupportAtomicAdd<DTYPE_VAR, DTYPE_INDICES, int16_t, false, CAST_3, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, y, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_4) {
            ScatterAddSIMDSortSupportAtomicAdd<DTYPE_VAR, DTYPE_INDICES, uint8_t, false, CAST_4, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, y, userWs);
            op.Process();
        } else if (tilingData.indicesCastMode == CAST_5) {
            ScatterAddSIMDSortSupportAtomicAdd<DTYPE_VAR, DTYPE_INDICES, uint8_t, false, CAST_5, SUB> op(tilingData, pipe);
            op.Init(var, indices, updates, y, userWs);
            op.Process();
        }
    } else {
        return;
    }
}

extern "C" __global__ __aicore__ void scatter_sub(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y,
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
        ScatterAddUnsortSimtSubr32Scalar(var, indices, updates, y, userWs, tiling, pipe);
    } else if (TILING_KEY_IS(TILING_KEY_UNSORT_SIMT_ADDR32_TENSOR)) {
        ScatterAddUnsortSimtSubr32Tensor(var, indices, updates, y, userWs, tiling, pipe);
    } else if (TILING_KEY_IS(TILING_KEY_UNSORT_SIMT_ADDR64_SCALAR)) {
        ScatterAddUnsortSimtSubr64Scalar(var, indices, updates, y, userWs, tiling, pipe);
    } else if (TILING_KEY_IS(TILING_KEY_UNSORT_SIMT_ADDR64_TENSOR)) {
        ScatterAddUnsortSimtSubr64Tensor(var, indices, updates, y, userWs, tiling, pipe);
    } else if (TILING_KEY_IS(TILING_KEY_SORT_SIMT_ADDR32_SCALAR)) {
        ScatterAddSortSimtSubr32Scalar(var, indices, updates, y, userWs, tiling, pipe);
    } else if (TILING_KEY_IS(TILING_KEY_SORT_SIMT_ADDR32_TENSOR)) {
        ScatterAddSortSimtSubr32Tensor(var, indices, updates, y, userWs, tiling, pipe);
    } else if (TILING_KEY_IS(TILING_KEY_SORT_SIMT_ADDR64_SCALAR)) {
        ScatterAddSortSimtSubr64Scalar(var, indices, updates, y, userWs, tiling, pipe);
    } else if (TILING_KEY_IS(TILING_KEY_SORT_SIMT_ADDR64_TENSOR)) {
        ScatterAddSortSimtSubr64Tensor(var, indices, updates, y, userWs, tiling, pipe);
    } else if (TILING_KEY_IS(TILING_KEY_UNSORT_SIMD_SCALAR)) {
        ScatterAddUnsortSimdScalar(var, indices, updates, y, userWs, tiling, pipe);
    } else if (TILING_KEY_IS(TILING_KEY_UNSORT_SIMD_TENSOR)) {
        ScatterAddUnsortSimdTensor(var, indices, updates, y, userWs, tiling, pipe);
    } else if (TILING_KEY_IS(TILING_KEY_SORT_SIMD_SCALAR)) {
        ScatterAddSortSimdScalar(var, indices, updates, y, userWs, tiling, pipe);
    } else if (TILING_KEY_IS(TILING_KEY_SORT_SIMD_TENSOR)) {
        ScatterAddSortSimdTensor(var, indices, updates, y, userWs, tiling, pipe);
    } else if (TILING_KEY_IS(TILING_KEY_0)) {
        return;
    }
}