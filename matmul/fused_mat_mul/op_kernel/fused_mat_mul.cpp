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
 * \file fused_mat_mul.cpp
 * \brief A2/A3 FusedMatMul relu. Wraps MatMulV3 / BatchMatMulV3 and enables FixPipe ReLU.
 */

#include "kernel_operator.h"
#include "../mat_mul_v3/mat_mul_v3_common.h"
#include "../mat_mul_v3/mat_mul_base_kernel.h"
#include "../mat_mul_v3/mat_mul_unaligned_base_kernel.h"
#include "../mat_mul_v3/mat_mul_l1_full_load.h"
#include "../batch_mat_mul_v3/batch_mat_mul_v3.h"
#include "../batch_mat_mul_v3/batch_mat_mul_v3_tiling_data.h"
#include "fused_matmul_tiling_key.h"

using namespace AscendC;
using namespace matmul;
using namespace MatmulV3;

#ifndef DTYPE_BIAS
#define DTYPE_BIAS DTYPE_X1
#endif

#ifndef FORMAT_FRACTAL_NZ
#define FORMAT_FRACTAL_NZ
#endif

#if defined(FORMAT_X1) && FORMAT_X1 == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_x1 = CubeFormat::NZ;
#else
constexpr CubeFormat format_x1 = CubeFormat::ND;
#endif

#if defined(FORMAT_X2) && FORMAT_X2 == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_x2 = CubeFormat::NZ;
#else
constexpr CubeFormat format_x2 = CubeFormat::ND;
#endif

#if defined(FORMAT_Y) && FORMAT_Y == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_y = CubeFormat::NZ;
#else
constexpr CubeFormat format_y = CubeFormat::ND;
#endif

__aicore__ inline constexpr MatmulConfig WithFixpipeRelu(MatmulConfig cfg)
{
    cfg.enableRelu = true;
    return cfg;
}

constexpr MatmulConfig MM_CFG_NO_PRELOAD_RELU = WithFixpipeRelu(MM_CFG_NO_PRELOAD);
constexpr MatmulConfig MM_CFG_MDL_RELU = WithFixpipeRelu(MM_CFG_MDL);
constexpr MatmulConfig MM_CFG_ORDER_M_RELU = WithFixpipeRelu(MM_CFG_ORDER_M);
constexpr MatmulConfig MM_CFG_MULTI_BATCH_OUT_RELU = WithFixpipeRelu(MM_CFG_MULTI_BATCH_OUT);

#define FMM_MM_IMPL(templateClass, ...)                                                            \
    do {                                                                                           \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y>;                       \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;           \
        TPipe pipe;                                                                                \
        __gm__ uint8_t* user = GetUserWorkspace(workspaceGM);                                      \
        if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 0) {        \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false>;          \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;          \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                          \
            op.Init(x1GM, x2GM, yGM, biasGM, x3GM, user, &tilingData, &pipe);                      \
            op.Process();                                                                          \
        } else if (tilingData.matmulRunInfo.transA == 1 && tilingData.matmulRunInfo.transB == 0) { \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true>;           \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;          \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                          \
            op.Init(x1GM, x2GM, yGM, biasGM, x3GM, user, &tilingData, &pipe);                      \
            op.Process();                                                                          \
        } else if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 1) { \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false>;          \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;           \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                          \
            op.Init(x1GM, x2GM, yGM, biasGM, x3GM, user, &tilingData, &pipe);                      \
            op.Process();                                                                          \
        } else {                                                                                   \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true>;           \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;           \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                          \
            op.Init(x1GM, x2GM, yGM, biasGM, x3GM, user, &tilingData, &pipe);                      \
            op.Process();                                                                          \
        }                                                                                          \
    } while (0)

#define FMM_BMM_IMPL_COMMON(templateClass, ...)                                                                       \
    do {                                                                                                              \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y>;                                          \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                              \
        TPipe pipe;                                                                                                   \
        __gm__ uint8_t* user = GetUserWorkspace(workspaceGM);                                                         \
        if (tilingData.matmulTiling.matmulRunInfo.transA == 0 && tilingData.matmulTiling.matmulRunInfo.transB == 0) { \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(x1GM, x2GM, yGM, biasGM, x3GM, user, &tilingData, &pipe);                                         \
            op.Process();                                                                                             \
        } else if (tilingData.matmulTiling.matmulRunInfo.transA == 1 &&                                               \
                   tilingData.matmulTiling.matmulRunInfo.transB == 0) {                                               \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true>;                              \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(x1GM, x2GM, yGM, biasGM, x3GM, user, &tilingData, &pipe);                                         \
            op.Process();                                                                                             \
        } else if (tilingData.matmulTiling.matmulRunInfo.transA == 0 &&                                               \
                   tilingData.matmulTiling.matmulRunInfo.transB == 1) {                                               \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                              \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(x1GM, x2GM, yGM, biasGM, x3GM, user, &tilingData, &pipe);                                         \
            op.Process();                                                                                             \
        } else {                                                                                                      \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true>;                              \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                              \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(x1GM, x2GM, yGM, biasGM, x3GM, user, &tilingData, &pipe);                                         \
            op.Process();                                                                                             \
        }                                                                                                             \
    } while (0)

#define FMM_BMM_IMPL_NORMAL(templateClass, ...)                                                                       \
    do {                                                                                                              \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y, false, LayoutMode::NORMAL>;               \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS, false, LayoutMode::NORMAL>;   \
        TPipe pipe;                                                                                                   \
        __gm__ uint8_t* user = GetUserWorkspace(workspaceGM);                                                         \
        if (tilingData.matmulTiling.matmulRunInfo.transA == 0 && tilingData.matmulTiling.matmulRunInfo.transB == 0) { \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false, LayoutMode::NORMAL>;         \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false, LayoutMode::NORMAL>;         \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(x1GM, x2GM, yGM, biasGM, x3GM, user, &tilingData, &pipe);                                         \
            op.Process();                                                                                             \
        } else if (tilingData.matmulTiling.matmulRunInfo.transA == 1 &&                                               \
                   tilingData.matmulTiling.matmulRunInfo.transB == 0) {                                               \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true, LayoutMode::NORMAL>;          \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false, LayoutMode::NORMAL>;         \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(x1GM, x2GM, yGM, biasGM, x3GM, user, &tilingData, &pipe);                                         \
            op.Process();                                                                                             \
        } else if (tilingData.matmulTiling.matmulRunInfo.transA == 0 &&                                               \
                   tilingData.matmulTiling.matmulRunInfo.transB == 1) {                                               \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false, LayoutMode::NORMAL>;         \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true, LayoutMode::NORMAL>;          \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(x1GM, x2GM, yGM, biasGM, x3GM, user, &tilingData, &pipe);                                         \
            op.Process();                                                                                             \
        } else {                                                                                                      \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true, LayoutMode::NORMAL>;          \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true, LayoutMode::NORMAL>;          \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(x1GM, x2GM, yGM, biasGM, x3GM, user, &tilingData, &pipe);                                         \
            op.Process();                                                                                             \
        }                                                                                                             \
    } while (0)

template <int LOADMODE, int SPLITCOREMODE, int FIXOPTI, int MIXND2NZ, int SPECIALOPT, int FP32ADDMM, int IS_BATCH,
          int BMM_L1_FULL, int BMM_MULTI, int BMM_LOAD, int BMM_OUT, int BMM_MIX>
__global__ __aicore__ void fused_mat_mul(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR biasGM, GM_ADDR x3GM, GM_ADDR yGM,
                                         GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MatmulTilingData);
    REGISTER_TILING_FOR_TILINGKEY("IS_BATCH == 1", BatchMatmulTilingData);
    if constexpr (IS_BATCH == FMM_A2_IS_MM && SPLITCOREMODE == FMM_MM_BASE_SPLIT_K && FIXOPTI == FMM_MM_BASE_FIXOPTI &&
                  MIXND2NZ == FMM_MM_MIXND2NZ_FALSE && SPECIALOPT == FMM_MM_K_NOT_SHIFT &&
                  FP32ADDMM == FMM_FP32_ADDMM_DISABLE) {
        GET_TILING_DATA_WITH_STRUCT(MatmulTilingData, tilingData, tilingGM);
        if constexpr (LOADMODE == FMM_MM_BASE_FULLLOAD) {
            FMM_MM_IMPL(MatmulBaseKernel, MatmulBaseBlock, MM_CFG_NO_PRELOAD_RELU);
        } else if constexpr (LOADMODE == FMM_MM_AL1_FULLLOAD) {
            FMM_MM_IMPL(MatmulBaseKernelAL1FullLoad, MatmulBaseBlock, MM_CFG_MDL_RELU);
        } else if constexpr (LOADMODE == FMM_MM_BL1_FULLLOAD) {
            FMM_MM_IMPL(MatmulBaseKernelBL1FullLoad, MatmulBaseBlock, MM_CFG_NO_PRELOAD_RELU);
        }
    } else if constexpr (IS_BATCH == FMM_A2_IS_MM && SPLITCOREMODE == FMM_MM_BASE_SPLIT_K &&
                         FIXOPTI == FMM_MM_BASE_FIXOPTI && MIXND2NZ == FMM_MM_MIXND2NZ_TRUE &&
                         SPECIALOPT == FMM_MM_K_NOT_SHIFT && FP32ADDMM == FMM_FP32_ADDMM_DISABLE &&
                         LOADMODE == FMM_MM_BASE_FULLLOAD) {
        GET_TILING_DATA_WITH_STRUCT(MatmulTilingData, tilingData, tilingGM);
        FMM_MM_IMPL(MatmulBaseUnAlignedKernel, MatmulBaseBlock, MM_CFG_NO_PRELOAD_RELU);
    } else if constexpr (IS_BATCH == FMM_A2_IS_BATCH && BMM_L1_FULL == FMM_BMM_L1_FULL_FALSE) {
        GET_TILING_DATA_WITH_STRUCT(BatchMatmulTilingData, tilingData, tilingGM);
        if constexpr (BMM_MULTI == FMM_BMM_MULTI_FALSE && BMM_OUT == FMM_BMM_OUT_FALSE &&
                      BMM_MIX == FMM_BMM_MIX_FALSE && BMM_LOAD == FMM_BMM_LOAD_BASE) {
            FMM_BMM_IMPL_COMMON(BatchMatMulCommonKernel, BatchMatMulCommonBaseBlock, MM_CFG_NO_PRELOAD_RELU);
        } else if constexpr (BMM_MULTI == FMM_BMM_MULTI_FALSE && BMM_OUT == FMM_BMM_OUT_FALSE &&
                             BMM_MIX == FMM_BMM_MIX_TRUE && BMM_LOAD == FMM_BMM_LOAD_BASE) {
            FMM_BMM_IMPL_COMMON(BatchMatMulUnalignedKernel, MatmulBaseBlock, MM_CFG_NO_PRELOAD_RELU);
        } else if constexpr (BMM_MULTI == FMM_BMM_MULTI_FALSE && BMM_OUT == FMM_BMM_OUT_FALSE &&
                             BMM_MIX == FMM_BMM_MIX_FALSE && BMM_LOAD == FMM_BMM_LOAD_AL1) {
            FMM_BMM_IMPL_COMMON(BatchMatMulCommonKernel, BatchMatMulCommonBaseBlock, MM_CFG_NO_PRELOAD_RELU,
                                MatmulCallBackFunc<nullptr, BmmCopyAL1, nullptr>);
        } else if constexpr (BMM_MULTI == FMM_BMM_MULTI_FALSE && BMM_OUT == FMM_BMM_OUT_FALSE &&
                             BMM_MIX == FMM_BMM_MIX_FALSE && BMM_LOAD == FMM_BMM_LOAD_BL1) {
            FMM_BMM_IMPL_COMMON(BatchMatMulCommonKernel, BatchMatMulCommonBaseBlock, MM_CFG_NO_PRELOAD_RELU,
                                MatmulCallBackFunc<nullptr, nullptr, BmmCopyBL1>);
        } else if constexpr (BMM_MULTI == FMM_BMM_MULTI_TRUE && BMM_LOAD == FMM_BMM_LOAD_BASE &&
                             BMM_MIX == FMM_BMM_MIX_FALSE && BMM_OUT == FMM_BMM_OUT_FALSE) {
            FMM_BMM_IMPL_NORMAL(BatchMatMulMultiBatchKernel, BatchMatMulMultiBatchBaseBlock, MM_CFG_ORDER_M_RELU);
        } else if constexpr (BMM_MULTI == FMM_BMM_MULTI_TRUE && BMM_LOAD == FMM_BMM_LOAD_BASE &&
                             BMM_MIX == FMM_BMM_MIX_FALSE && BMM_OUT == FMM_BMM_OUT_TRUE) {
            FMM_BMM_IMPL_NORMAL(BatchMatMulMultiBatchKernel, BatchMatMulMultiBatchBaseBlock,
                                MM_CFG_MULTI_BATCH_OUT_RELU);
        } else if constexpr (BMM_MULTI == FMM_BMM_MULTI_TRUE && BMM_LOAD == FMM_BMM_LOAD_BASE &&
                             BMM_MIX == FMM_BMM_MIX_TRUE && BMM_OUT == FMM_BMM_OUT_FALSE) {
            FMM_BMM_IMPL_NORMAL(BatchMatMulUnalignedMultiBatchKernel, BatchMatMulUnalignedMultiBatchBaseBlock,
                                MM_CFG_ORDER_M_RELU);
        } else if constexpr (BMM_MULTI == FMM_BMM_MULTI_TRUE && BMM_LOAD == FMM_BMM_LOAD_BASE &&
                             BMM_MIX == FMM_BMM_MIX_TRUE && BMM_OUT == FMM_BMM_OUT_TRUE) {
            FMM_BMM_IMPL_NORMAL(BatchMatMulUnalignedMultiBatchKernel, BatchMatMulUnalignedMultiBatchBaseBlock,
                                MM_CFG_MULTI_BATCH_OUT_RELU);
        }
    }
}
