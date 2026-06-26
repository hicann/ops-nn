/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file batch_mat_mul_v3.cpp
 * \brief
 */

#include "../../inc/macro.h"

#if ASC_DEVKIT_MAJOR < 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR <= 0)
#define IS_BLAZE false
#else
#define IS_BLAZE true
#endif

#include "../../mat_mul_v3/arch35/mat_mul_tiling_data.h"
#include "batch_mat_mul_v3_tiling_key.h"
#include "batch_mat_mul_v3_asw_kernel_advanced.h"

#if !__FIXED_POINT_ONLY_CUBE_TO_L0C__
#include "batch_mat_mul_v3_asw_al1_full_load_kernel_advanced.h"
#include "batch_mat_mul_v3_asw_bl1_full_load_kernel_advanced.h"
#include "batch_mat_mul_v3_mergebatch_basicapi_cmct.h"
#include "batch_mat_mul_v3_matmul2mul_cmct.h"
#include "../../mat_mul_v3/arch35/mat_mul_input_k_eq_zero_clear_output.h"
#include "batch_mat_mul_v3_iterbatch_kernel_advanced.h"
#if IS_BLAZE
#include "batch_mat_mul_v3_asw_broadcast.h"
#include "../../mat_mul_v3/arch35/mat_mul_pingpong_basic.h"
#endif
#endif

#include "batch_mat_mul_v3_iterbatch_basicapi_cmct.h"
#include "../../mat_mul_v3/arch35/mat_mul_pingpong_basic_cmct.h"
#include "../../mat_mul_v3/arch35/mat_mul_streamk_basic_cmct.h"

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

#define BMMV3_IMPL_CLASS(templateClass, ...)                                                                          \
    do {                                                                                                              \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y, false, LayoutMode::NORMAL>;               \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS, false, LayoutMode::NORMAL>;   \
        TPipe pipe;                                                                                                   \
        if (tilingData.matmulTiling.matmulRunInfo.transA == 0 && tilingData.matmulTiling.matmulRunInfo.transB == 0) { \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false, LayoutMode::NORMAL>;         \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false, LayoutMode::NORMAL>;         \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process();                                                                                             \
        } else if (                                                                                                   \
            tilingData.matmulTiling.matmulRunInfo.transA == 1 && tilingData.matmulTiling.matmulRunInfo.transB == 0) { \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true, LayoutMode::NORMAL>;          \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false, LayoutMode::NORMAL>;         \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process();                                                                                             \
        } else if (                                                                                                   \
            tilingData.matmulTiling.matmulRunInfo.transA == 0 && tilingData.matmulTiling.matmulRunInfo.transB == 1) { \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false, LayoutMode::NORMAL>;         \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true, LayoutMode::NORMAL>;          \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process();                                                                                             \
        } else {                                                                                                      \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true, LayoutMode::NORMAL>;          \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true, LayoutMode::NORMAL>;          \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process();                                                                                             \
        }                                                                                                             \
    } while (0)

#define BMMV3_IMPL_CLASS_TRANS(transA, transB, templateClass, ...)                                                  \
    do {                                                                                                            \
        GET_TILING_DATA(tilingData, tilingGM);                                                                      \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y, false, LayoutMode::NORMAL>;             \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS, false, LayoutMode::NORMAL>; \
        TPipe pipe;                                                                                                 \
        using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, transA, LayoutMode::NORMAL>;          \
        using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, transB, LayoutMode::NORMAL>;          \
        templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                               \
        op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                        \
        op.Process();                                                                                               \
    } while (0)

#define BMMV3_IMPL_CLASS_COMMON(templateClass, ...)                                                                   \
    do {                                                                                                              \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y>;                                          \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                              \
        TPipe pipe;                                                                                                   \
        if (tilingData.matmulTiling.matmulRunInfo.transA == 0 && tilingData.matmulTiling.matmulRunInfo.transB == 0) { \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process();                                                                                             \
        } else if (                                                                                                   \
            tilingData.matmulTiling.matmulRunInfo.transA == 1 && tilingData.matmulTiling.matmulRunInfo.transB == 0) { \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true>;                              \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                             \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process();                                                                                             \
        } else if (                                                                                                   \
            tilingData.matmulTiling.matmulRunInfo.transA == 0 && tilingData.matmulTiling.matmulRunInfo.transB == 1) { \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false>;                             \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                              \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process();                                                                                             \
        } else {                                                                                                      \
            using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, true>;                              \
            using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, true>;                              \
            templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                             \
            op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);                                      \
            op.Process();                                                                                             \
        }                                                                                                             \
    } while (0)

#define BMMV3_IMPL_CLASS_COMMON_TRNAS(transA, transB, templateClass, ...)                \
    do {                                                                                 \
        GET_TILING_DATA(tilingData, tilingGM);                                           \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y>;             \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>; \
        TPipe pipe;                                                                      \
        using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, transA>;   \
        using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, transB>;   \
        templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                    \
        op.Init(aGM, bGM, cGM, biasGM, offsetWGM, user, &tilingData, &pipe);             \
        op.Process();                                                                    \
    } while (0)
template <
    int8_t BATCH_API_LEVEL, int8_t BATCH_A_TRANS, int8_t BATCH_B_TRANS, int8_t BATCH_ITER_MODEL, int8_t BMODEL,
    int8_t BATCH_FULL_LOAD, int8_t BATCH_L0C2OUT_MODEL>
__global__ __aicore__ void batch_mat_mul_v3(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    AscendC::InitSocState();
    __gm__ uint8_t* user = GetUserWorkspace(workspaceGM);

    constexpr bool aTran = (BATCH_A_TRANS == 1);
    constexpr bool bTran = (BATCH_B_TRANS == 1);

    using aLayout = std::conditional_t<aTran, layout::ColumnMajor, layout::RowMajor>;
    using bLayout = std::conditional_t<
        (format_x2 == CubeFormat::NZ), std::conditional_t<bTran, layout::Zn, layout::Nz>,
        std::conditional_t<bTran, layout::ColumnMajor, layout::RowMajor>>;
#if !__FIXED_POINT_ONLY_CUBE_TO_L0C__ && IS_BLAZE
    using layoutA = AscendC::Std::conditional_t<aTran, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using layoutB = AscendC::Std::conditional_t<
        bTran,
        AscendC::Std::conditional_t<
            (format_x2 == CubeFormat::NZ), AscendC::Te::ZNLayoutPtn, AscendC::Te::DNExtLayoutPtn>,
        AscendC::Std::conditional_t<
            (format_x2 == CubeFormat::NZ), AscendC::Te::NZLayoutPtn, AscendC::Te::NDExtLayoutPtn>>;
    // C GM layout
    using layoutC = AscendC::Te::NDExtLayoutPtn;
#endif

    REGISTER_TILING_DEFAULT(BatchMatMulV3TilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    if constexpr (
        BATCH_API_LEVEL == MAT_MUL_HIGH_LEVEL && BMODEL == MAT_MUL_BASIC && BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD &&
        BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY && BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
        BMMV3_IMPL_CLASS_COMMON_TRNAS(
            aTran, bTran, BatchMatMulV3Advanced::BatchMatMulAswKernel, BatchMatMulV3Advanced::BatchMatMulAswBlock,
            MM_CFG_NO_PRELOAD);
    } else if constexpr (
        BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC && BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD &&
        BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY && BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
        GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, tilingGM);
        MatmulV3Advanced::MatMulActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData.matMulTilingData, tilingData.batchDimAll);
    } else if constexpr (
        BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC && BATCH_FULL_LOAD == MAT_MUL_A_FULL_LOAD &&
        BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY && BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
        GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, tilingGM);
        MatmulV3Advanced::MatMulActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor, A_FULL_LOAD_MODE>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData.matMulTilingData, tilingData.batchDimAll);
    } else if constexpr (
        BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC && BATCH_FULL_LOAD == MAT_MUL_B_FULL_LOAD &&
        BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY && BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
        GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, tilingGM);
        MatmulV3Advanced::MatMulActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor, B_FULL_LOAD_MODE>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData.matMulTilingData, tilingData.batchDimAll);
    } else if constexpr (
        BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC && BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD &&
        BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY && BATCH_ITER_MODEL == MAT_MUL_ITER_BATCH_SINGLE_BIAS) {
        GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3IterBatchBasicTilingData, tilingData, tilingGM);
        BatchMatMulActIterBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if constexpr (
        BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_STREAM_K &&
        BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
        BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
        GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, tilingGM);
        MatMulStreamKActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                               MatMulL0C2Out::ON_THE_FLY>(aGM, bGM, biasGM, cGM, workspaceGM,
                                                          tilingData.matMulTilingData, tilingData.batchDimAll);
    } else if constexpr (
        BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_STREAM_K &&
        BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE &&
        BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
        GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, tilingGM);
        MatMulStreamKActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                               MatMulL0C2Out::ND_FIXPIPE_1_2>(aGM, bGM, biasGM, cGM, workspaceGM,
                                                              tilingData.matMulTilingData, tilingData.batchDimAll);
#if !__FIXED_POINT_ONLY_CUBE_TO_L0C__
    } else if constexpr (
        BATCH_API_LEVEL == MAT_MUL_TENSOR_LEVEL && BMODEL == MAT_MUL_BASIC && BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD &&
        BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY && BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
        GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3TilingData, tilingData, tilingGM);
#if IS_BLAZE
        BatchMatMulV3Advanced::BatchMatMulBroadcastKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layoutA, layoutB, layoutC>(
            aGM, bGM, biasGM, cGM, nullptr, tilingData);
#else
        BMMV3_IMPL_CLASS_COMMON_TRNAS(
            aTran, bTran, BatchMatMulV3Advanced::BatchMatMulAswKernel, BatchMatMulV3Advanced::BatchMatMulAswBlock,
            MM_CFG_NO_PRELOAD);
#endif
    } else if constexpr (
        BATCH_API_LEVEL == MAT_MUL_TENSOR_LEVEL_BASIC && BMODEL == MAT_MUL_BASIC &&
        BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD && BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY &&
        BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
        GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, tilingGM);
#if IS_BLAZE
        MatmulV3Advanced::MatMulBasicKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layoutA, layoutB, layoutC>(
            aGM, bGM, biasGM, cGM, nullptr, tilingData.matMulTilingData, tilingData.batchDimAll);
#else
        MatmulV3Advanced::MatMulActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData.matMulTilingData, tilingData.batchDimAll);
#endif
    } else if constexpr (
        BATCH_API_LEVEL == MAT_MUL_HIGH_LEVEL && BMODEL == MAT_MUL_BASIC && BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD &&
        BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY && BATCH_ITER_MODEL == MAT_MUL_ITER_BATCH_SINGLE_BIAS) {
        BMMV3_IMPL_CLASS_TRANS(
            aTran, bTran, BatchMatMulV3Advanced::BatchMatMulMultiBatchKernel,
            BatchMatMulV3Advanced::BatchMatMulMultiBatchBaseBlock, MM_CFG_MULTI_BATCH_OUT_SING_BIAS);
    } else if constexpr (
        BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC && BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD &&
        BATCH_L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE && BATCH_ITER_MODEL == MAT_MUL_ITER_BATCH_SINGLE_BIAS) {
        GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3IterBatchBasicTilingData, tilingData, tilingGM);
        BatchMatMulActIterBatchKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor, MatMulL0C2Out::ND_FIXPIPE_1_2>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if constexpr (
        BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC && BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD &&
        BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY && BATCH_ITER_MODEL == MAT_MUL_MERGE_BATCH) {
        GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3MergeBatchBasicTilingData, tilingData, tilingGM);
        BatchMatMulActMergeBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if constexpr (
        BATCH_API_LEVEL == MAT_MUL_BASIC_LEVEL && BMODEL == MAT_MUL_BASIC && BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD &&
        BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY && BATCH_ITER_MODEL == MAT_MUL_BATCH_MATMUL_TO_MUL &&
        IsSameType<DTYPE_X1, DTYPE_Y>::value) {
        GET_TILING_DATA_WITH_STRUCT(BatchMatMulToMulBasicTilingData, tilingData, tilingGM);
        BatchMatMulToMulActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if constexpr (
        BATCH_API_LEVEL == MAT_MUL_HIGH_LEVEL && BMODEL == MAT_MUL_K_EQUAL_ZERO && BATCH_FULL_LOAD == MAT_MUL_NO_FULL_LOAD &&
        BATCH_L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY && BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
        TPipe pipe;
        GET_TILING_DATA_WITH_STRUCT(MatMulV3KEqZeroBasicTilingData, tilingData, tilingGM);
        MatmulV3Advanced::MatMulInputKEqZeroClearOutput(biasGM, cGM, tilingData);
#endif
    }
}