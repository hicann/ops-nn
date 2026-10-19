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
 * \file scatter_update_apt.cpp
 * \brief A6(ascend960) kernel entry of scatter_update
 */

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../arch35/scatter_update_simt.h"
#include "../arch35/scatter_update_simt_sort.h"
#include "../arch35/scatter_update_simd_sort.h"
#include "../arch35/scatter_update_simd.h"
#include "scatter_update_simd_scatter_dma.h"
#include "../arch35/scatter_update_mask_simd.h"
#include "../arch35/scatter_update_deterministic_simd.h"
#include "../arch35/scatter_update_deterministic_simt.h"

#define TILING_KEY_0 0
#define TILING_KEY_SIMT_ADDR32_SCALAR 10000000000333331000UL
#define TILING_KEY_SIMT_ADDR32_TENSOR_DTYPE1 10000000100333330000UL
#define TILING_KEY_SIMT_ADDR32_TENSOR_DTYPE2 10000000200333330000UL
#define TILING_KEY_SIMT_ADDR32_TENSOR_DTYPE3 10000000300333330000UL
#define TILING_KEY_SIMT_ADDR32_TENSOR_DTYPE4 10000000400333330000UL
#define TILING_KEY_SIMT_ADDR32_TENSOR_DTYPE5 10000000500333330000UL
#define TILING_KEY_SIMT_ADDR64_SCALAR 10000000000333331100UL
#define TILING_KEY_SIMT_ADDR64_TENSOR_DTYPE1 10000000100333330100UL
#define TILING_KEY_SIMT_ADDR64_TENSOR_DTYPE2 10000000200333330100UL
#define TILING_KEY_SIMT_ADDR64_TENSOR_DTYPE3 10000000300333330100UL
#define TILING_KEY_SIMT_ADDR64_TENSOR_DTYPE4 10000000400333330100UL
#define TILING_KEY_SIMT_ADDR64_TENSOR_DTYPE5 10000000500333330100UL
#define TILING_KEY_SIMT_ADDR32_SCALAR_SORT 10000000000333331001UL
#define TILING_KEY_SIMT_ADDR32_TENSOR_SORT_DTYPE1 10000000100333330001UL
#define TILING_KEY_SIMT_ADDR32_TENSOR_SORT_DTYPE2 10000000200333330001UL
#define TILING_KEY_SIMT_ADDR32_TENSOR_SORT_DTYPE3 10000000300333330001UL
#define TILING_KEY_SIMT_ADDR32_TENSOR_SORT_DTYPE4 10000000400333330001UL
#define TILING_KEY_SIMT_ADDR32_TENSOR_SORT_DTYPE5 10000000500333330001UL
#define TILING_KEY_SIMT_ADDR64_SCALAR_SORT 10000000000333331101UL
#define TILING_KEY_SIMT_ADDR64_TENSOR_SORT_DTYPE1 10000000100333330101UL
#define TILING_KEY_SIMT_ADDR64_TENSOR_SORT_DTYPE2 10000000200333330101UL
#define TILING_KEY_SIMT_ADDR64_TENSOR_SORT_DTYPE3 10000000300333330101UL
#define TILING_KEY_SIMT_ADDR64_TENSOR_SORT_DTYPE4 10000000400333330101UL
#define TILING_KEY_SIMT_ADDR64_TENSOR_SORT_DTYPE5 10000000500333330101UL
#define TILING_KEY_SIMD_ADDR32_SCALAR 10000000000333331010UL
#define TILING_KEY_SIMD_ADDR32_TENSOR 10000000000333330010UL
#define TILING_KEY_SIMD_ADDR32_SCALAR_SORT 10000000000333331011UL
#define TILING_KEY_SIMD_ADDR32_TENSOR_SORT 10000000000333330011UL
#define TILING_KEY_SIMD_ADDR32_TENSOR_MASK 10000000001333330010UL
#define TILING_KEY_DETERMINISTIC_SIMD_SPLITCOL 10000000010333330010UL
#define TILING_KEY_DETERMINISTIC_SIMD_SPLITROW 10000000020333330010UL
#define TILING_KEY_DETERMINISTIC_SIMT_DTYPE1 10000000120333330000UL
#define TILING_KEY_DETERMINISTIC_SIMT_DTYPE2 10000000220333330000UL
#define TILING_KEY_DETERMINISTIC_SIMT_DTYPE3 10000000320333330000UL
#define TILING_KEY_DETERMINISTIC_SIMT_DTYPE4 10000000420333330000UL
#define TILING_KEY_DETERMINISTIC_SIMT_DTYPE5 10000000520333330000UL
#define TILING_KEY_SIMD_SCATTER_DMA_SCALAR 10000001000333331010UL
#define TILING_KEY_SIMD_SCATTER_DMA_TENSOR 10000001000333330010UL
#define TILING_KEY_SIMD_SCATTER_DMA_CAST_SCALAR 10000002000333331010UL
#define TILING_KEY_SIMD_SCATTER_DMA_CAST_TENSOR 10000002000333330010UL

using namespace ScatterUpdate;

template <typename VAR_T>
__aicore__ inline void SortSimtAddr32Scalar(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR userWs,
                                            const ScatterUpdateTilingData& tilingData)
{
    if (tilingData.indicesCastMode == CAST_NOT_CAST) {
        ScatterUpdateSimtSort<DTYPE_INDICES, VAR_T, DTYPE_INDICES, uint32_t, true, CAST_NOT_CAST> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT32_TO_INT16) {
        ScatterUpdateSimtSort<DTYPE_INDICES, VAR_T, int16_t, uint32_t, true, CAST_INT32_TO_INT16> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT64_TO_INT32) {
        ScatterUpdateSimtSort<DTYPE_INDICES, VAR_T, int32_t, uint32_t, true, CAST_INT64_TO_INT32> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT64_TO_INT16) {
        ScatterUpdateSimtSort<DTYPE_INDICES, VAR_T, int16_t, uint32_t, true, CAST_INT64_TO_INT16> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT32_TO_UINT8) {
        ScatterUpdateSimtSort<DTYPE_INDICES, VAR_T, uint8_t, uint32_t, true, CAST_INT32_TO_UINT8> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT64_TO_UINT8) {
        ScatterUpdateSimtSort<DTYPE_INDICES, VAR_T, uint8_t, uint32_t, true, CAST_INT64_TO_UINT8> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    }
}

template <typename VAR_T>
__aicore__ inline void SortSimtAddr32Tensor(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR userWs,
                                            const ScatterUpdateTilingData& tilingData)
{
    if (tilingData.indicesCastMode == CAST_NOT_CAST) {
        ScatterUpdateSimtSort<DTYPE_INDICES, VAR_T, DTYPE_INDICES, uint32_t, false, CAST_NOT_CAST> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT32_TO_INT16) {
        ScatterUpdateSimtSort<DTYPE_INDICES, VAR_T, int16_t, uint32_t, false, CAST_INT32_TO_INT16> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT64_TO_INT32) {
        ScatterUpdateSimtSort<DTYPE_INDICES, VAR_T, int32_t, uint32_t, false, CAST_INT64_TO_INT32> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT64_TO_INT16) {
        ScatterUpdateSimtSort<DTYPE_INDICES, VAR_T, int16_t, uint32_t, false, CAST_INT64_TO_INT16> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT32_TO_UINT8) {
        ScatterUpdateSimtSort<DTYPE_INDICES, VAR_T, uint8_t, uint32_t, false, CAST_INT32_TO_UINT8> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT64_TO_UINT8) {
        ScatterUpdateSimtSort<DTYPE_INDICES, VAR_T, uint8_t, uint32_t, false, CAST_INT64_TO_UINT8> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    }
}

template <typename VAR_T>
__aicore__ inline void SortSimtAddr64Scalar(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR userWs,
                                            const ScatterUpdateTilingData& tilingData)
{
    if (tilingData.indicesCastMode == CAST_NOT_CAST) {
        ScatterUpdateSimtSort<DTYPE_INDICES, VAR_T, DTYPE_INDICES, uint64_t, true, CAST_NOT_CAST> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT32_TO_INT16) {
        ScatterUpdateSimtSort<DTYPE_INDICES, VAR_T, int16_t, uint64_t, true, CAST_INT32_TO_INT16> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT64_TO_INT32) {
        ScatterUpdateSimtSort<DTYPE_INDICES, VAR_T, int32_t, uint64_t, true, CAST_INT64_TO_INT32> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT64_TO_INT16) {
        ScatterUpdateSimtSort<DTYPE_INDICES, VAR_T, int16_t, uint64_t, true, CAST_INT64_TO_INT16> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT32_TO_UINT8) {
        ScatterUpdateSimtSort<DTYPE_INDICES, VAR_T, uint8_t, uint64_t, true, CAST_INT32_TO_UINT8> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT64_TO_UINT8) {
        ScatterUpdateSimtSort<DTYPE_INDICES, VAR_T, uint8_t, uint64_t, true, CAST_INT64_TO_UINT8> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    }
}

template <typename VAR_T>
__aicore__ inline void SortSimtAddr64Tensor(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR userWs,
                                            const ScatterUpdateTilingData& tilingData)
{
    if (tilingData.indicesCastMode == CAST_NOT_CAST) {
        ScatterUpdateSimtSort<DTYPE_INDICES, VAR_T, DTYPE_INDICES, uint64_t, false, CAST_NOT_CAST> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT32_TO_INT16) {
        ScatterUpdateSimtSort<DTYPE_INDICES, VAR_T, int16_t, uint64_t, false, CAST_INT32_TO_INT16> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT64_TO_INT32) {
        ScatterUpdateSimtSort<DTYPE_INDICES, VAR_T, int32_t, uint64_t, false, CAST_INT64_TO_INT32> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT64_TO_INT16) {
        ScatterUpdateSimtSort<DTYPE_INDICES, VAR_T, int16_t, uint64_t, false, CAST_INT64_TO_INT16> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT32_TO_UINT8) {
        ScatterUpdateSimtSort<DTYPE_INDICES, VAR_T, uint8_t, uint64_t, false, CAST_INT32_TO_UINT8> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT64_TO_UINT8) {
        ScatterUpdateSimtSort<DTYPE_INDICES, VAR_T, uint8_t, uint64_t, false, CAST_INT64_TO_UINT8> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    }
}

template <typename VAR_T>
__aicore__ inline void SortSimdScalar(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR userWs,
                                      const ScatterUpdateTilingData& tilingData, TPipe& pipe)
{
    if (tilingData.indicesCastMode == CAST_NOT_CAST) {
        ScatterUpdateSimdSort<VAR_T, DTYPE_INDICES, DTYPE_INDICES, true, CAST_NOT_CAST> op(tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT32_TO_INT16) {
        ScatterUpdateSimdSort<VAR_T, DTYPE_INDICES, int16_t, true, CAST_INT32_TO_INT16> op(tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT64_TO_INT32) {
        ScatterUpdateSimdSort<VAR_T, DTYPE_INDICES, int32_t, true, CAST_INT64_TO_INT32> op(tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT64_TO_INT16) {
        ScatterUpdateSimdSort<VAR_T, DTYPE_INDICES, int16_t, true, CAST_INT64_TO_INT16> op(tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT32_TO_UINT8) {
        ScatterUpdateSimdSort<VAR_T, DTYPE_INDICES, uint8_t, true, CAST_INT32_TO_UINT8> op(tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT64_TO_UINT8) {
        ScatterUpdateSimdSort<VAR_T, DTYPE_INDICES, uint8_t, true, CAST_INT64_TO_UINT8> op(tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    }
}

template <typename VAR_T>
__aicore__ inline void SortSimdTensor(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR userWs,
                                      const ScatterUpdateTilingData& tilingData, TPipe& pipe)
{
    if (tilingData.indicesCastMode == CAST_NOT_CAST) {
        ScatterUpdateSimdSort<VAR_T, DTYPE_INDICES, DTYPE_INDICES, false, CAST_NOT_CAST> op(tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT32_TO_INT16) {
        ScatterUpdateSimdSort<VAR_T, DTYPE_INDICES, int16_t, false, CAST_INT32_TO_INT16> op(tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT64_TO_INT32) {
        ScatterUpdateSimdSort<VAR_T, DTYPE_INDICES, int32_t, false, CAST_INT64_TO_INT32> op(tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT64_TO_INT16) {
        ScatterUpdateSimdSort<VAR_T, DTYPE_INDICES, int16_t, false, CAST_INT64_TO_INT16> op(tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT32_TO_UINT8) {
        ScatterUpdateSimdSort<VAR_T, DTYPE_INDICES, uint8_t, false, CAST_INT32_TO_UINT8> op(tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT64_TO_UINT8) {
        ScatterUpdateSimdSort<VAR_T, DTYPE_INDICES, uint8_t, false, CAST_INT64_TO_UINT8> op(tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    }
}

template <typename IDX_SIZE_T, typename VAR_T>
__aicore__ inline void DeterministicSimdSort(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR userWs,
                                             const ScatterUpdateTilingData& tilingData, TPipe& pipe)
{
    if (tilingData.indicesCastMode == CAST_NOT_CAST) {
        ScatterUpdateDeterministicSimd<VAR_T, DTYPE_INDICES, IDX_SIZE_T, false, DTYPE_INDICES, CAST_NOT_CAST> op(
            tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT32_TO_INT16) {
        ScatterUpdateDeterministicSimd<VAR_T, DTYPE_INDICES, IDX_SIZE_T, false, int16_t, CAST_INT32_TO_INT16> op(
            tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT64_TO_INT32) {
        ScatterUpdateDeterministicSimd<VAR_T, DTYPE_INDICES, IDX_SIZE_T, false, int32_t, CAST_INT64_TO_INT32> op(
            tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT64_TO_INT16) {
        ScatterUpdateDeterministicSimd<VAR_T, DTYPE_INDICES, IDX_SIZE_T, false, int16_t, CAST_INT64_TO_INT16> op(
            tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT32_TO_UINT8) {
        ScatterUpdateDeterministicSimd<VAR_T, DTYPE_INDICES, IDX_SIZE_T, false, uint8_t, CAST_INT32_TO_UINT8> op(
            tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT64_TO_UINT8) {
        ScatterUpdateDeterministicSimd<VAR_T, DTYPE_INDICES, IDX_SIZE_T, false, uint8_t, CAST_INT64_TO_UINT8> op(
            tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    }
}

template <typename IDX_SIZE_T, typename VAR_T>
__aicore__ inline void DeterministicSimtSort(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR userWs,
                                             const ScatterUpdateTilingData& tilingData, TPipe& pipe)
{
    if (tilingData.indicesCastMode == CAST_NOT_CAST) {
        ScatterUpdateDeterministicSimt<VAR_T, DTYPE_INDICES, IDX_SIZE_T, false, DTYPE_INDICES, CAST_NOT_CAST> op(
            tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT32_TO_INT16) {
        ScatterUpdateDeterministicSimt<VAR_T, DTYPE_INDICES, IDX_SIZE_T, false, int16_t, CAST_INT32_TO_INT16> op(
            tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT64_TO_INT32) {
        ScatterUpdateDeterministicSimt<VAR_T, DTYPE_INDICES, IDX_SIZE_T, false, int32_t, CAST_INT64_TO_INT32> op(
            tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT64_TO_INT16) {
        ScatterUpdateDeterministicSimt<VAR_T, DTYPE_INDICES, IDX_SIZE_T, false, int16_t, CAST_INT64_TO_INT16> op(
            tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT32_TO_UINT8) {
        ScatterUpdateDeterministicSimt<VAR_T, DTYPE_INDICES, IDX_SIZE_T, false, uint8_t, CAST_INT32_TO_UINT8> op(
            tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (tilingData.indicesCastMode == CAST_INT64_TO_UINT8) {
        ScatterUpdateDeterministicSimt<VAR_T, DTYPE_INDICES, IDX_SIZE_T, false, uint8_t, CAST_INT64_TO_UINT8> op(
            tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    }
}

template <typename T>
struct GetComputeType {
    using type = typename std::conditional<
        std::is_same<T, hifloat4x2_t>::value, int8_t,                  // 优先处理特殊类型
        typename std::conditional<sizeof(T) == sizeof(int8_t), int8_t, // 处理1字节类型
                                  T                                    // 其他类型保持不变
                                  >::type>::type;
};

extern "C" __global__ __aicore__ void scatter_update(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR varRef,
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
    REGISTER_TILING_DEFAULT(ScatterUpdateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    using VAR_T = typename GetComputeType<DTYPE_VAR>::type;

    ///////////////////// SIMT //////////////////////////
    if (TILING_KEY_IS(TILING_KEY_SIMT_ADDR32_SCALAR)) {
        ScatterUpdateSimt<DTYPE_INDICES, VAR_T, uint32_t, true> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_ADDR32_TENSOR_DTYPE1)) {
        ScatterUpdateSimt<DTYPE_INDICES, int8_t, uint32_t, false> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_ADDR32_TENSOR_DTYPE2)) {
        ScatterUpdateSimt<DTYPE_INDICES, half, uint32_t, false> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_ADDR32_TENSOR_DTYPE3)) {
        ScatterUpdateSimt<DTYPE_INDICES, int32_t, uint32_t, false> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_ADDR32_TENSOR_DTYPE4)) {
        ScatterUpdateSimt<DTYPE_INDICES, int64_t, uint32_t, false> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_ADDR32_TENSOR_DTYPE5)) {
        ScatterUpdateSimt<DTYPE_INDICES, int4, uint32_t, false> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_ADDR64_SCALAR)) {
        ScatterUpdateSimt<DTYPE_INDICES, VAR_T, uint64_t, true> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_ADDR64_TENSOR_DTYPE1)) {
        ScatterUpdateSimt<DTYPE_INDICES, int8_t, uint64_t, false> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_ADDR64_TENSOR_DTYPE2)) {
        ScatterUpdateSimt<DTYPE_INDICES, half, uint64_t, false> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_ADDR64_TENSOR_DTYPE3)) {
        ScatterUpdateSimt<DTYPE_INDICES, int32_t, uint64_t, false> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_ADDR64_TENSOR_DTYPE4)) {
        ScatterUpdateSimt<DTYPE_INDICES, int64_t, uint64_t, false> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_ADDR64_TENSOR_DTYPE5)) {
        ScatterUpdateSimt<DTYPE_INDICES, int4, uint64_t, false> op(tilingData);
        op.Init(var, indices, updates, userWs);
        op.Process();

        ///////////////////// SIMT排序 //////////////////////////
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_ADDR32_SCALAR_SORT)) {
        SortSimtAddr32Scalar<VAR_T>(var, indices, updates, userWs, tilingData);
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_ADDR32_TENSOR_SORT_DTYPE1)) {
        SortSimtAddr32Tensor<int8_t>(var, indices, updates, userWs, tilingData);
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_ADDR32_TENSOR_SORT_DTYPE2)) {
        SortSimtAddr32Tensor<half>(var, indices, updates, userWs, tilingData);
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_ADDR32_TENSOR_SORT_DTYPE3)) {
        SortSimtAddr32Tensor<int32_t>(var, indices, updates, userWs, tilingData);
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_ADDR32_TENSOR_SORT_DTYPE4)) {
        SortSimtAddr32Tensor<int64_t>(var, indices, updates, userWs, tilingData);
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_ADDR32_TENSOR_SORT_DTYPE5)) {
        SortSimtAddr32Tensor<int4>(var, indices, updates, userWs, tilingData);
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_ADDR64_SCALAR_SORT)) {
        SortSimtAddr64Scalar<VAR_T>(var, indices, updates, userWs, tilingData);
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_ADDR64_TENSOR_SORT_DTYPE1)) {
        SortSimtAddr64Tensor<int8_t>(var, indices, updates, userWs, tilingData);
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_ADDR64_TENSOR_SORT_DTYPE2)) {
        SortSimtAddr64Tensor<half>(var, indices, updates, userWs, tilingData);
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_ADDR64_TENSOR_SORT_DTYPE3)) {
        SortSimtAddr64Tensor<int32_t>(var, indices, updates, userWs, tilingData);
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_ADDR64_TENSOR_SORT_DTYPE4)) {
        SortSimtAddr64Tensor<int64_t>(var, indices, updates, userWs, tilingData);
    } else if (TILING_KEY_IS(TILING_KEY_SIMT_ADDR64_TENSOR_SORT_DTYPE5)) {
        SortSimtAddr64Tensor<int4>(var, indices, updates, userWs, tilingData);

        ///////////////////// SIMD //////////////////////////
    } else if (TILING_KEY_IS(TILING_KEY_SIMD_ADDR32_SCALAR)) {
        ScatterUpdateSIMDImpl<VAR_T, DTYPE_INDICES, true> op(tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SIMD_ADDR32_TENSOR)) {
        ScatterUpdateSIMDImpl<VAR_T, DTYPE_INDICES, false> op(tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SIMD_SCATTER_DMA_SCALAR)) {
        ScatterUpdateScatterDMA<VAR_T, DTYPE_INDICES, true, false> op(tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SIMD_SCATTER_DMA_TENSOR)) {
        ScatterUpdateScatterDMA<VAR_T, DTYPE_INDICES, false, false> op(tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SIMD_SCATTER_DMA_CAST_SCALAR)) {
        ScatterUpdateScatterDMA<VAR_T, DTYPE_INDICES, true, true> op(tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SIMD_SCATTER_DMA_CAST_TENSOR)) {
        ScatterUpdateScatterDMA<VAR_T, DTYPE_INDICES, false, true> op(tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_SIMD_ADDR32_SCALAR_SORT)) {
        SortSimdScalar<VAR_T>(var, indices, updates, userWs, tilingData, pipe);
    } else if (TILING_KEY_IS(TILING_KEY_SIMD_ADDR32_TENSOR_SORT)) {
        SortSimdTensor<VAR_T>(var, indices, updates, userWs, tilingData, pipe);

        ///////////////////// maskSIMD //////////////////////////
    } else if (TILING_KEY_IS(TILING_KEY_SIMD_ADDR32_TENSOR_MASK)) {
        ScatterUpdateMaskSIMDImpl<VAR_T, DTYPE_INDICES> op(tilingData, pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();

        ///////////////////// Deterministic //////////////////////////
    } else if (TILING_KEY_IS(TILING_KEY_DETERMINISTIC_SIMD_SPLITCOL)) {
        ScatterUpdateDeterministicSimd<VAR_T, DTYPE_INDICES, int64_t, true, DTYPE_INDICES, CAST_NOT_CAST> op(tilingData,
                                                                                                             pipe);
        op.Init(var, indices, updates, userWs);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_DETERMINISTIC_SIMD_SPLITROW)) {
        if (tilingData.isIndicesSizeInt64) {
            DeterministicSimdSort<int64_t, VAR_T>(var, indices, updates, userWs, tilingData, pipe);
        } else {
            DeterministicSimdSort<int32_t, VAR_T>(var, indices, updates, userWs, tilingData, pipe);
        }
    } else if (TILING_KEY_IS(TILING_KEY_DETERMINISTIC_SIMT_DTYPE1)) {
        if (tilingData.isIndicesSizeInt64) {
            DeterministicSimtSort<int64_t, int8_t>(var, indices, updates, userWs, tilingData, pipe);
        } else {
            DeterministicSimtSort<int32_t, int8_t>(var, indices, updates, userWs, tilingData, pipe);
        }
    } else if (TILING_KEY_IS(TILING_KEY_DETERMINISTIC_SIMT_DTYPE2)) {
        if (tilingData.isIndicesSizeInt64) {
            DeterministicSimtSort<int64_t, half>(var, indices, updates, userWs, tilingData, pipe);
        } else {
            DeterministicSimtSort<int32_t, half>(var, indices, updates, userWs, tilingData, pipe);
        }
    } else if (TILING_KEY_IS(TILING_KEY_DETERMINISTIC_SIMT_DTYPE3)) {
        if (tilingData.isIndicesSizeInt64) {
            DeterministicSimtSort<int64_t, int32_t>(var, indices, updates, userWs, tilingData, pipe);
        } else {
            DeterministicSimtSort<int32_t, int32_t>(var, indices, updates, userWs, tilingData, pipe);
        }
    } else if (TILING_KEY_IS(TILING_KEY_DETERMINISTIC_SIMT_DTYPE4)) {
        if (tilingData.isIndicesSizeInt64) {
            DeterministicSimtSort<int64_t, int64_t>(var, indices, updates, userWs, tilingData, pipe);
        } else {
            DeterministicSimtSort<int32_t, int64_t>(var, indices, updates, userWs, tilingData, pipe);
        }
    } else if (TILING_KEY_IS(TILING_KEY_DETERMINISTIC_SIMT_DTYPE5)) {
        if (tilingData.isIndicesSizeInt64) {
            DeterministicSimtSort<int64_t, int4>(var, indices, updates, userWs, tilingData, pipe);
        } else {
            DeterministicSimtSort<int32_t, int4>(var, indices, updates, userWs, tilingData, pipe);
        }
    } else if (TILING_KEY_IS(TILING_KEY_0)) {
        return;
    }
}
