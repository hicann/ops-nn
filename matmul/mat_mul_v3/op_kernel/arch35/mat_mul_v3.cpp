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
 * \file mat_mul_v3.cpp
 * \brief
 */

#if ASC_DEVKIT_MAJOR < 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR <= 0)
#define IS_BLAZE false
#else
#define IS_BLAZE true
#endif

#include "mat_mul_tiling_key.h"
#include "../mat_mul_v3_common.h"
#include "mat_mul_asw_kernel.h"
#include "mat_mul_tiling_data.h"
#include "mat_mul_full_load.h"

#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102))
#include "mat_mul_stream_k_kernel.h"
#include "mat_mul_fixpipe_opti.h"
#include "mat_mul_input_k_eq_zero_clear_output.h"
#include "mat_mul_streamk_basic_cmct.h"
#include "mat_mul_fixpipe_opti_basic_cmct.h"
#if IS_BLAZE
#include "mat_mul_streamk.h"
#endif
#endif

#include "mat_mul_pingpong_basic_cmct.h"
#include "mat_mul_to_mul_cmct.h"

using namespace Cmct;
using namespace Cmct::Gemm;
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

#define MMV3_IMPL_CLASS_TRANS(tilingData, tilingGM, transA, transB, workspace, templateClass, ...)     \
    do {                                                                                               \
        uint64_t tilingdata_offset =                                                                   \
            (sizeof(MatMulV3TilingData) > TILINGDATA_OFFSET) ?                                         \
                0 :                                                                                    \
                MMV3DivFloor(GetCurrentBlockIdx(), MMV3DivCeil(GetBlockNum(), TILINGDATA_SPLIT_NUM)) * \
                    TILINGDATA_OFFSET;                                                                 \
        GET_TILING_DATA_WITH_STRUCT(MatMulV3TilingData, tilingData, tilingGM + tilingdata_offset);     \
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_Y>;                     \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;               \
        TPipe pipe;                                                                                    \
        using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, transA>;                 \
        using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, transB>;                 \
        templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                                  \
        op.Init(aGM, bGM, cGM, biasGM, offsetWGM, workspace, &tilingData, &pipe);                      \
        op.Process();                                                                                  \
    } while (0)
template <
    int8_t API_LEVEL, int8_t A_TRANS, int8_t B_TRANS, int8_t BATCH_MODEL, int8_t MODEL, int8_t FULL_LOAD,
    int8_t L0C2OUT_MODEL>
__global__ __aicore__ void mat_mul_v3(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR offsetWGM, GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    constexpr bool aTran = (A_TRANS == 1);
    constexpr bool bTran = (B_TRANS == 1);

    using aLayout = std::conditional_t<aTran, layout::ColumnMajor, layout::RowMajor>;
    using bLayout = std::conditional_t<
        (format_x2 == CubeFormat::NZ), std::conditional_t<bTran, layout::Zn, layout::Nz>,
        std::conditional_t<bTran, layout::ColumnMajor, layout::RowMajor>>;
#if IS_BLAZE
    using layoutA = AscendC::Std::conditional_t<aTran, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using layoutB = AscendC::Std::conditional_t<
        bTran,
        AscendC::Std::conditional_t<
            (format_x2 == CubeFormat::NZ), AscendC::Te::ZNLayoutPtn, AscendC::Te::DNExtLayoutPtn>,
        AscendC::Std::conditional_t<
            (format_x2 == CubeFormat::NZ), AscendC::Te::NZLayoutPtn, AscendC::Te::NDExtLayoutPtn>>;
#endif

    REGISTER_TILING_DEFAULT(MatMulV3TilingDataCopy);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);

    if constexpr (
        API_LEVEL == MAT_MUL_BASIC_LEVEL && FULL_LOAD == MAT_MUL_NO_FULL_LOAD && MODEL == MAT_MUL_BASIC &&
        L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY) {
        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
        MatmulV3Advanced::MatMulActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor>(
            aGM, bGM, biasGM, cGM, nullptr, tilingData);
    } else if constexpr (
        API_LEVEL == MAT_MUL_BASIC_LEVEL && FULL_LOAD == MAT_MUL_B_FULL_LOAD && MODEL == MAT_MUL_BASIC &&
        L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY) {
        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
        MatmulV3Advanced::MatMulActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor, B_FULL_LOAD_MODE>(
            aGM, bGM, biasGM, cGM, nullptr, tilingData);
    } else if constexpr (
        API_LEVEL == MAT_MUL_BASIC_LEVEL && FULL_LOAD == MAT_MUL_A_FULL_LOAD && MODEL == MAT_MUL_BASIC &&
        L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY) { // A全载模板切换基础API kernel实现
        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
        MatmulV3Advanced::MatMulActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor, A_FULL_LOAD_MODE>(
            aGM, bGM, biasGM, cGM, nullptr, tilingData);
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102))
    } else if constexpr (
        API_LEVEL == MAT_MUL_TENSOR_LEVEL && FULL_LOAD == MAT_MUL_NO_FULL_LOAD && MODEL == MAT_MUL_STREAM_K &&
        L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY) {
        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
#if IS_BLAZE
        MatmulV3Advanced::MatMulStreamKKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layoutA, layoutB, Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
#else
        MatMulStreamKActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor, MatMulL0C2Out::ON_THE_FLY>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
#endif
    } else if constexpr (
        API_LEVEL == MAT_MUL_TENSOR_LEVEL && FULL_LOAD == MAT_MUL_NO_FULL_LOAD && MODEL == MAT_MUL_STREAM_K &&
        L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE) {
        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
#if IS_BLAZE
        MatmulV3Advanced::MatMulStreamKKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layoutA, layoutB, Blaze::Gemm::MatMulL0C2Out::ND_FIXPIPE_1_2>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
#else
        MatMulStreamKActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor, MatMulL0C2Out::ND_FIXPIPE_1_2>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
#endif
    } else if constexpr (
        API_LEVEL == MAT_MUL_BASIC_LEVEL && FULL_LOAD == MAT_MUL_NO_FULL_LOAD && MODEL == MAT_MUL_STREAM_K &&
        L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY) {
        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
        MatMulStreamKActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor, MatMulL0C2Out::ON_THE_FLY>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if constexpr (
        API_LEVEL == MAT_MUL_BASIC_LEVEL && FULL_LOAD == MAT_MUL_NO_FULL_LOAD && MODEL == MAT_MUL_STREAM_K &&
        L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE) {
        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
        MatMulStreamKActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor, MatMulL0C2Out::ND_FIXPIPE_1_2>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if constexpr (
        API_LEVEL == MAT_MUL_HIGH_LEVEL && FULL_LOAD == MAT_MUL_NO_FULL_LOAD && MODEL == MAT_MUL_K_EQUAL_ZERO &&
        L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY) {
        TPipe pipe;
        GET_TILING_DATA_WITH_STRUCT(MatMulV3KEqZeroBasicTilingData, tilingData, tilingGM);
        MatmulV3Advanced::MatMulInputKEqZeroClearOutput(biasGM, cGM, tilingData);
    } else if constexpr (
        API_LEVEL == MAT_MUL_BASIC_LEVEL && FULL_LOAD == MAT_MUL_B_FULL_LOAD && MODEL == MAT_MUL_BASIC &&
        L0C2OUT_MODEL == MAT_MUL_1V1_ND_ALIG_FIXPIPE) { // Fixpipe B全载fp16场景act kernel
        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
        MatmulV3Advanced::MatMulFixpipeOptiActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor, B_FULL_LOAD_MODE>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if constexpr (
        API_LEVEL == MAT_MUL_BASIC_LEVEL && FULL_LOAD == MAT_MUL_NO_FULL_LOAD && MODEL == MAT_MUL_BASIC &&
        L0C2OUT_MODEL == MAT_MUL_1V1_ND_ALIG_FIXPIPE) {
        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
        MatmulV3Advanced::MatMulFixpipeOptiActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if constexpr (
        API_LEVEL == MAT_MUL_BASIC_LEVEL && FULL_LOAD == MAT_MUL_NO_FULL_LOAD && MODEL == MAT_MUL_BASIC &&
        L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE) {
        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
        MatmulV3Advanced::MatMulFixpipeOptiActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if constexpr (
        API_LEVEL == MAT_MUL_BASIC_LEVEL && FULL_LOAD == MAT_MUL_B_FULL_LOAD && MODEL == MAT_MUL_BASIC &&
        L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE) { // Fixpipe B全载fp32场景切换act kernel
        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
        MatmulV3Advanced::MatMulFixpipeOptiActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor, B_FULL_LOAD_MODE>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
    } else if constexpr (
        API_LEVEL == MAT_MUL_BASIC_LEVEL && FULL_LOAD == MAT_MUL_NO_FULL_LOAD && MODEL == MAT_MUL_TO_MUL &&
        L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY) {
        GET_TILING_DATA_WITH_STRUCT(MatMulToMulBasicTilingData, tilingData, tilingGM);
        MatmulV3Advanced::MatMulToMulActKernel<
            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor>(
            aGM, bGM, biasGM, cGM, workspaceGM, tilingData);
#endif
    }
}
