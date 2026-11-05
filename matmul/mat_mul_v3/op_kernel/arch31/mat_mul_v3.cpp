/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "../mat_mul_v3_tiling_data.h"
#include "mat_mul_al1_strip_kernel.h"
#include "mat_mul_base_kernel_arch31.h"
#include "mat_mul_splitk_al1_kernel.h"
#include "mat_mul_v3_l1_policy.h"

using ATypeNN = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
using BTypeNN = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, false>;
using BTypeNT = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half, true>;
using CType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;
using BiasType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;

__aicore__ inline int32_t GetMatMulV3RouteMarker(const MatmulTilingData& tilingData)
{
    constexpr uint32_t kRouteMarkerWord = 49U;
    return reinterpret_cast<const int32_t*>(&tilingData.matmulTiling)[kRouteMarkerWord];
}

// On dav-l311 Nd2NzParams::srcDValue is uint16_t.  With a non-transposed
// ND B matrix the source row stride is the full N, even when singleCoreN
// is tiled.  Enable the Matmul API's loop-based GM-to-L1 copy at that limit.
constexpr MatmulConfig MM_CFG_MDL_LARGE_N_STRIDE = GetMDLConfig(true);

template <class B_TYPE, const MatmulConfig& MM_CFG = MM_CFG_MDL>
__aicore__ inline void RunMatMulV3(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR cGM,
                                   GM_ADDR workspaceGM, const MatmulTilingData& tilingData)
{
    AscendC::TPipe pipe;
    MatmulBaseKernelArch31<ATypeNN, B_TYPE, CType, BiasType, MatmulBaseBlock, MM_CFG> op;
    op.Init(aGM, bGM, cGM, biasGM, offsetWGM, workspaceGM, &tilingData, &pipe);
    op.Process();
    op.End();
}

__aicore__ inline bool UseAl1Strip(const MatmulTilingData& tilingData)
{
    const uint64_t stripBytes = static_cast<uint64_t>(tilingData.matmulTiling.baseM) *
                                static_cast<uint64_t>(tilingData.matmulTiling.Ka) * sizeof(half);
    return GetMatMulV3RouteMarker(tilingData) == matmul_v3_arch31::kAl1StripMarker &&
           tilingData.matmulTiling.isBias == 0 && tilingData.matmulRunInfo.transA == 0U &&
           tilingData.matmulRunInfo.transB != 0U && tilingData.matmulTiling.baseM > 0 &&
           tilingData.matmulTiling.M <= tilingData.matmulTiling.baseM &&
           stripBytes <= static_cast<uint64_t>(matmul_v3_arch31::kAl1StripBudgetBytes);
}

__aicore__ inline bool UseSplitKAl1(const MatmulTilingData& tilingData)
{
    const uint64_t m = tilingData.matmulTiling.M;
    const uint64_t n = tilingData.matmulTiling.N;
    const uint64_t k = tilingData.matmulTiling.Ka;
    const uint64_t baseK = tilingData.matmulTiling.baseK;
    const uint64_t fullABytes = m * k * sizeof(half);
    const uint64_t stripK = baseK == 0U ? 0U :
                                          static_cast<uint64_t>(matmul_v3_arch31::kAl1StripBudgetBytes) /
                                              (m * sizeof(half)) / baseK * baseK;
    return GetMatMulV3RouteMarker(tilingData) == matmul_v3_arch31::kSplitKAl1Marker &&
           tilingData.matmulTiling.isBias == 0 && tilingData.matmulRunInfo.transA == 0U &&
           tilingData.matmulRunInfo.transB != 0U && m == static_cast<uint64_t>(tilingData.matmulTiling.baseM) &&
           m % matmul_v3_arch31::kCubeAlign == 0U && n % matmul_v3_arch31::kCubeAlign == 0U && baseK > 0U &&
           k % baseK == 0U && stripK >= baseK && stripK < k &&
           (n + matmul_v3_arch31::kAl1BaseN - 1) / matmul_v3_arch31::kAl1BaseN >= matmul_v3_arch31::kMinReuseNTiles &&
           fullABytes > static_cast<uint64_t>(matmul_v3_arch31::kAl1StripBudgetBytes);
}

__aicore__ inline void RunMatMulV3NT(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR cGM,
                                     GM_ADDR workspaceGM, const MatmulTilingData& tilingData)
{
    if (UseSplitKAl1(tilingData)) {
        AscendC::TPipe pipe;
        MatmulSplitKAl1KernelArch31<BTypeNT, CType, BiasType, MM_CFG_MDL> op;
        op.Init(aGM, bGM, cGM, workspaceGM, &tilingData, &pipe);
        op.Process();
        return;
    }
    if (UseAl1Strip(tilingData)) {
        AscendC::TPipe pipe;
        MatmulAl1StripKernelArch31<BTypeNT, CType, BiasType, MM_CFG_MDL> op;
        op.Init(aGM, bGM, cGM, &tilingData, &pipe);
        op.Process();
        return;
    }
    RunMatMulV3<BTypeNT>(aGM, bGM, biasGM, offsetWGM, cGM, workspaceGM, tilingData);
}

extern "C" __global__ __aicore__ void mat_mul_v3(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR offsetWGM,
                                                 GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MatmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MatmulTilingData, tilingData, tilingGM);
    GM_ADDR userWorkspace = GetUserWorkspace(workspaceGM);
    if (tilingData.matmulRunInfo.transA != 0U) {
        return;
    }
    if (tilingData.matmulRunInfo.transB != 0U) {
        RunMatMulV3NT(aGM, bGM, biasGM, offsetWGM, cGM, userWorkspace, tilingData);
    } else {
        if (tilingData.matmulTiling.N >= 65535) {
            RunMatMulV3<BTypeNN, MM_CFG_MDL_LARGE_N_STRIDE>(aGM, bGM, biasGM, offsetWGM, cGM, userWorkspace,
                                                            tilingData);
        } else {
            RunMatMulV3<BTypeNN>(aGM, bGM, biasGM, offsetWGM, cGM, userWorkspace, tilingData);
        }
    }
}
