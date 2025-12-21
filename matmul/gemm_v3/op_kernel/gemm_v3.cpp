/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gemm_v3.cpp
 * \brief
 */

// mat_mul_sc_splitk_kernel_gm_to_l1.h 中使用了编译器宏 DTYPE_X1，gemm_v3 需路由为 DTYPE_A
#ifndef DTYPE_X1
#define DTYPE_X1 DTYPE_A
#endif

#include "../mat_mul_v3/mat_mul_v3_common.h"
#include "../mat_mul_v3/mat_mul_base_kernel.h"
#include "../mat_mul_v3/mat_mul_deterministic_splitk_kernel.h"
#include "../mat_mul_v3/mat_mul_sc_splitk_kernel.h"
#include "../mat_mul_v3/mat_mul_sc_splitk_kernel_gm_to_l1.h"
#include "../mat_mul_v3/mat_mul_unaligned_base_kernel.h"
#include "../mat_mul_v3/mat_mul_cvp_base_kernel.h"
#include "../mat_mul_v3/mat_mul_unaligned_deterministic_splitk_kernel.h"
#include "../mat_mul_v3/mat_mul_unaligned_sc_splitk_kernel.h"
#include "../mat_mul_v3/mat_mul_unaligned_sc_splitk_kernel_gm_to_l1.h"
#include "../mat_mul_v3/mat_mul_optimized_fixpipe_algorithm.h"
#include "../mat_mul_v3/mat_mul_l1_full_load.h"
#include "../mat_mul_v3/mat_mul_v3_tiling_key.h"

#undef DTYPE_X1

#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
#include "../mat_mul_v3/mat_mul_multi_core_splitk_kernel.h"
#endif

using namespace AscendC;
using namespace matmul;
#ifndef DTYPE_BIAS
#define DTYPE_BIAS half
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

#define MMV3_IMPL_ACCU(templateFunc, cFormat, ...)                                                                   \
    do {                                                                                                             \
        using cType = MatmulType<AscendC::TPosition::GM, cFormat, DTYPE_Y>;                                          \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                             \
        if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 0) {                          \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_A, false>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_B, false>;                             \
            templateFunc<aType, bType, cType, biasType, __VA_ARGS__>(aGM, bGM, cGM, biasGM, tilingData, user, 1);    \
        } else if (tilingData.matmulRunInfo.transA == 1 && tilingData.matmulRunInfo.transB == 0) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_A, true>;                              \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_B, false>;                             \
            templateFunc<aType, bType, cType, biasType, __VA_ARGS__>(aGM, bGM, cGM, biasGM, tilingData, user, 1);    \
        } else if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 1) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_A, false>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_B, true>;                              \
            templateFunc<aType, bType, cType, biasType, __VA_ARGS__>(aGM, bGM, cGM, biasGM, tilingData, user, 1);    \
        } else {                                                                                                     \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_A, true>;                              \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_B, true>;                              \
            templateFunc<aType, bType, cType, biasType, __VA_ARGS__>(aGM, bGM, cGM, biasGM, tilingData, user, 1);    \
        }                                                                                                            \
    } while(0)

#define MMV3_IMPL_CLASS_0_1_ACCU(templateClass, aFormat, ...)                                                        \
    do {                                                                                                             \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y>;                                         \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                             \
        TPipe pipe;                                                                                                  \
        if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 0) {                          \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_A, false>;                               \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_B, false>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process(0, 1);                                                                                        \
        } else if (tilingData.matmulRunInfo.transA == 1 && tilingData.matmulRunInfo.transB == 0) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_A, true>;                                \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_B, false>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process(0, 1);                                                                                        \
        } else if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 1) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_A, false>;                               \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_B, true>;                              \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process(0, 1);                                                                                        \
        } else {                                                                                                     \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_A, true>;                                \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_B, true>;                              \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process(0, 1);                                                                                        \
        }                                                                                                            \
    } while (0)

#define MMV3_IMPL_CLASS_1_ACCU(templateClass, aFormat, ...)                                                          \
    do {                                                                                                             \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y>;                                         \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                             \
        TPipe pipe;                                                                                                  \
        if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 0) {                          \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_A, false>;                               \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_B, false>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process(1);                                                                                           \
        } else if (tilingData.matmulRunInfo.transA == 1 && tilingData.matmulRunInfo.transB == 0) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_A, true>;                                \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_B, false>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process(1);                                                                                           \
        } else if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 1) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_A, false>;                               \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_B, true>;                              \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process(1);                                                                                           \
        } else {                                                                                                     \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_A, true>;                                \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_B, true>;                              \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process(1);                                                                                           \
        }                                                                                                            \
    } while (0)


// cFormat is for tempCGlobal Nz out, not cTensor out
#define MMV3_IMPL_C_CLASS_ACCU(templateClass, aFormat, cFormat, ...)                                                 \
    do {                                                                                                             \
        using cType = MatmulType<AscendC::TPosition::GM, cFormat, DTYPE_Y>;                                          \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                             \
        TPipe pipe;                                                                                                  \
        if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 0) {                          \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_A, false>;                               \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_B, false>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process(0, 1);                                                                                        \
        } else if (tilingData.matmulRunInfo.transA == 1 && tilingData.matmulRunInfo.transB == 0) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_A, true>;                                \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_B, false>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process(0, 1);                                                                                        \
        } else if (tilingData.matmulRunInfo.transA == 0 && tilingData.matmulRunInfo.transB == 1) {                   \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_A, false>;                               \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_B, true>;                              \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process(0, 1);                                                                                        \
        } else {                                                                                                     \
            using aType = MatmulType<AscendC::TPosition::GM, aFormat, DTYPE_A, true>;                                \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_B, true>;                              \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                            \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                     \
            op.Process(0, 1);                                                                                        \
        }                                                                                                            \
    } while (0)


template <int LOADMODE, int SPLITCOREMODE, int FIXOPTI, int MIXND2NZ, int SPECIALOPT>
__global__ __aicore__ void gemm_v3(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR yGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MatmulTilingData);
    __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);
    GM_ADDR biasGM = nullptr;
    GM_ADDR offsetWGM = nullptr;
    GET_TILING_DATA(tilingData, tilingGM);
    if constexpr (
        LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_BASE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE &&
        SPECIALOPT == MAT_MUL_V3_K_SHIFT) {
        MMV3_IMPL_CLASS_0_1_ACCU(
            MatmulBaseKernel, format_x1, MatmulBaseBlock, MM_CFG_K_SHIFT
        );
    } else if constexpr (LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_BASE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_TRUE) {
        MMV3_IMPL_CLASS_0_1_ACCU(
            MatmulBaseUnAlignedKernel, format_x1, MatmulBaseBlock, MM_CFG_NO_PRELOAD
        );
    } else if constexpr (
        LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_SINGLE_CORE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE) {
        MMV3_IMPL_CLASS_1_ACCU(
            MatMulSingleCoreSplitKKernel, format_x1, MatmulSingleCoreSplitKBaseBlock, MM_CFG_PRELOAD_MK
        );
    } else if constexpr (
        LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_SINGLE_CORE_NKM_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE) {
        MMV3_IMPL_CLASS_1_ACCU(
            MatMulSingleCoreSplitKKernel, format_x1, MatmulSingleCoreSplitKBaseBlock, MM_CFG_PRELOAD_NK, true
        );
    } else if constexpr (
        LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_SINGLE_CORE_SPLIT_K_GM_TO_L1 &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE) {
        MMV3_IMPL_CLASS_1_ACCU(
            MatMulSingleCoreSplitKKernelGmToL1, format_x1, MatmulSingleCoreSplitKBaseBlock, MM_CFG_PRELOAD_MK
        );
    } else if constexpr (
        LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_SINGLE_CORE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_TRUE) {
        MMV3_IMPL_CLASS_1_ACCU(
            MatMulUnAlignedSingleCoreSplitKKernel, format_x1, MatmulSingleCoreSplitKBaseBlock, MM_CFG_PRELOAD_MK
        );
    } else if constexpr (
        LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_SINGLE_CORE_SPLIT_K_GM_TO_L1 &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_TRUE) {
        MMV3_IMPL_CLASS_1_ACCU(
            MatMulUnAlignedSingleCoreSplitKKernelGmToL1, format_x1, MatmulSingleCoreSplitKBaseBlock, MM_CFG_PRELOAD_MK
        );
    } else if constexpr (
        LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_DETERMINISTIC_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE) {
        MMV3_IMPL_ACCU(
            MatMulKernelDeterministicSplitK, format_x1, FIXPIPE_OPT_SELECT::BASE
        );
    } else if constexpr (
        LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_DETERMINISTIC_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_TRUE) {
        MMV3_IMPL_ACCU(
            MatMulUnAlignedKernelDeterministicSplitK, format_x1, FIXPIPE_OPT_SELECT::BASE
        );
    } else if constexpr (
        LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_MULTI_CORE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE) {
        MMV3_IMPL_ACCU(
            MatMulMultiCoreSplitK, format_x1, FIXPIPE_OPT_SELECT::BASE
        );
    } else if constexpr (
        LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_BASE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE) {
        MMV3_IMPL_CLASS_0_1_ACCU(
            MatmulBaseKernel, format_x1, MatmulBaseBlock, MM_CFG_NO_PRELOAD
        );
    } else if constexpr (
        LOADMODE == MAT_MUL_V3_AL1_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_BASE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE) {
        MMV3_IMPL_CLASS_0_1_ACCU(
            MatmulBaseKernelAL1FullLoad, format_x1, MatmulBaseBlock, MM_CFG_MDL
        );
    } else if constexpr (
        LOADMODE == MAT_MUL_V3_BL1_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_BASE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE) {
        MMV3_IMPL_CLASS_0_1_ACCU(
            MatmulBaseKernelBL1FullLoad, format_x1, MatmulBaseBlock, MM_CFG_NO_PRELOAD
        );
    } else if constexpr (
        LOADMODE == MAT_MUL_V3_BL1_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_BASE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_ENABLE_ALIGNOUT && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE) {
        MMV3_IMPL_CLASS_0_1_ACCU(
            MatmulBaseUnalignedNKernel, format_x1, MatmulBaseBlock, MM_CFG_NO_PRELOAD,
            MatmulCallBackFunc<nullptr, nullptr, CopyBL1>
        );
    } else if constexpr (
        LOADMODE == MAT_MUL_V3_BL1_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_BASE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_TRUE) {
        MMV3_IMPL_CLASS_0_1_ACCU(
            MatmulBaseUnAlignedKernelBL1FullLoad, format_x1, MatmulBaseBlock, MM_CFG_NO_PRELOAD,
            MatmulCallBackFunc<nullptr, nullptr, CopyBL1>
        );
    } else if constexpr (
        LOADMODE == MAT_MUL_V3_BL1_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_BASE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_FIXOPTI && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_TRUE_PARALLEL) {
        MMV3_IMPL_CLASS_0_1_ACCU(
            MatmulCvpBaseKernel, format_x1, MatmulBaseBlock, MM_CFG_NO_PRELOAD
        );
    } else if constexpr (
        LOADMODE == MAT_MUL_V3_BL1_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_BASE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_BASE_ENABLE_ALIGNOUT && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_TRUE) {
        MMV3_IMPL_CLASS_0_1_ACCU(
            MatmulBaseAToNZWithBL1FixpipeKernel, CubeFormat::NZ, MatmulBaseBlock, MM_CFG_NO_PRELOAD,
            MatmulCallBackFunc<nullptr, nullptr, CopyBL1>
        );
    } else if constexpr (
        LOADMODE == MAT_MUL_V3_BL1_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_BASE_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_VEC_NZ2ND_UNALIGNOUT && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE) {
        MMV3_IMPL_C_CLASS_ACCU(
            MatmulBaseUnalignedNKernel, format_x1, CubeFormat::NZ, MatmulBaseBlock, MM_CFG_NO_PRELOAD,
            MatmulCallBackFunc<nullptr, nullptr, CopyBL1>, FIXPIPE_OPT_SELECT::VEC_NZ2ND_UNALIGNOUT
        );
    } else if constexpr (
        LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_DETERMINISTIC_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_VEC_NZ2ND_UNALIGNOUT && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_FALSE) {
        MMV3_IMPL_ACCU(
            MatMulKernelDeterministicSplitK, CubeFormat::NZ, FIXPIPE_OPT_SELECT::VEC_NZ2ND_UNALIGNOUT
        );
    } else if constexpr (
        LOADMODE == MAT_MUL_V3_BASE_FULLLOAD && SPLITCOREMODE == MAT_MUL_V3_DETERMINISTIC_SPLIT_K &&
        FIXOPTI == MAT_MUL_V3_VEC_NZ2ND_UNALIGNOUT && MIXND2NZ == MAT_MUL_V3_MIXND2NZ_TRUE) {
        MMV3_IMPL_ACCU(
            MatMulUnAlignedKernelDeterministicSplitK, CubeFormat::NZ, FIXPIPE_OPT_SELECT::VEC_NZ2ND_UNALIGNOUT
        );
    }
}
