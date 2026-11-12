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
 * \file fused_mat_mul.cpp
 * \brief
 */

#include "../inc/macro.h"

#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)) || \
    __FIXED_POINT_ONLY_CUBE_TO_L0C__
#include "../mat_mul_v3/mat_mul_v3_common.h"          // 共有,BLOCK_SIZE等定义
#include "./arch35/fused_mat_mul_tilingkey.h"         // 共有,tilingKey模板参数
#include "../mat_mul_v3/arch35/mat_mul_tiling_data.h" // 共有,tilingData定义

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
#include "./arch35/fused_mat_mul_tiling_data.h"              // 3510独有,FusedMatMulTilingData的定义
#include "cmct/kernel/kernel_matmul_mix.h"                   // 3510独有,KernelMatmulMix类
#include "cmct/kernel/kernel_matmul.h"                       // 3510独有,KernelMatmul类
#include "cmct/kernel/kernel_matmul_iterbatch.h"             // 没有用到,确认后可删除
#include "cmct/kernel/kernel_matmul_mix_without_que.h"       // 没有用到,确认后可删除
#include "../mat_mul_v3/arch35/mat_mul_streamk_basic_cmct.h" // 3510独有,低阶API实现,MM,streamK模板
#include "../mat_mul_v3/arch35/mat_mul_mix_basic_cmct.h"     // 3510独有,低阶API实现,MM,MixWithoutQue模板
#include "./arch35/fused_mat_mul_gelu_basic_cmct.h"          // 3510独有,低阶API实现,MM+GELU,MixWithoutQue模板
#include "./arch35/fused_mat_mul_sigmoid_asw_basic.h"        // Blaze MM+Sigmoid Fixpipe template
#include "../mat_mul_v3/arch35/mat_mul_fixpipe_opti_basic_cmct.h"      // 3510独有,低阶API实现,FixpipeOpti模板
#include "../mat_mul_v3/arch35/mat_mul_input_k_eq_zero_clear_output.h" // 3510独有,K=0清零输出
#include "./arch35/fused_mat_mul_input_k_eq_zero_copy_x3.h"            // 3510 K=0 add returns x3
#include "./arch35/fused_mat_mul_with_scale_add_asw_basic.h"           // BMM scale/add epilogue
#include "../mat_mul_v3/arch35/mat_mul_pingpong_basic.h"
#include "../mat_mul_v3/arch35/mat_mul_fixpipe.h"
#include "../mat_mul_v3/arch35/mat_mul_streamk.h"
#include "../mat_mul_v3/arch35/mat_mul_al1_full_load.h"
#include "../mat_mul_v3/arch35/mat_mul_bl1_full_load.h"
#include "arch35/fused_mat_mul_elemwise.h" // MM add/mul组装(独立文件, 对齐mat_mul_pingpong_basic.h模式)
#endif

#if __FIXED_POINT_ONLY_CUBE_TO_L0C__
#include "../batch_mat_mul_v3/arch35/batch_mat_mul_v3_asw_kernel_advanced.h" // FIXED_POINT_ONLY_CUBE_TO_L0C 独有,高阶API实现,BMM,aswt模板
#endif

#include "../batch_mat_mul_v3/arch35/batch_mat_mul_v3_iterbatch_basicapi_cmct.h" // 共有,低阶API实现,BMM,iterbatch模板
#include "../mat_mul_v3/arch35/mat_mul_pingpong_basic_cmct.h" // 共有,低阶API实现,MM/BMM,aswt模板
#include "../batch_mat_mul_v3/arch35/batch_mat_mul_v3_mergebatch_basicapi_cmct.h"

using namespace Cmct;
using namespace Gemm;
using namespace AscendC;

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

#if defined(FORMAT_X3) && FORMAT_X3 == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_x3 = CubeFormat::NZ;
#else
constexpr CubeFormat format_x3 = CubeFormat::ND;
#endif

#if __FIXED_POINT_ONLY_CUBE_TO_L0C__
#define BMMV3_IMPL_CLASS_COMMON_TRANS(transA, transB, templateClass, ...)                \
    do {                                                                                 \
        GET_TILING_DATA(tilingData, tilingGM);                                           \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y>;             \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>; \
        TPipe pipe;                                                                      \
        using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, transA>;   \
        using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, transB>;   \
        templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                    \
        op.Init(x1GM, x2GM, yGM, biasGM, nullptr, userWorkspaceGM, &tilingData, &pipe);  \
        if constexpr (OPTYPE == F_OPTYPE_QUANT || OPTYPE == F_OPTYPE_RELU_QUANT) {       \
            op.CacheQuantScalar(LoadQuantScalarFromGm(x3GM));                            \
        }                                                                                \
        op.Process();                                                                    \
    } while (0)

__aicore__ inline constexpr MatmulConfig GetCfgRelu()
{
    auto cfg = MM_CFG_NO_PRELOAD;
    cfg.enableRelu = true;
    return cfg;
}

constexpr MatmulConfig MM_CFG_NO_PRELOAD_RELU = GetCfgRelu();
#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
struct FusedMatMulKernelArgs {
    GM_ADDR x1GM;
    GM_ADDR x2GM;
    GM_ADDR biasGM;
    GM_ADDR x3GM;
    GM_ADDR yGM;
    GM_ADDR workspaceGM;
    GM_ADDR tilingGM;
    GM_ADDR userWorkspaceGM;
};

template <int8_t OPTYPE>
__aicore__ inline constexpr uint64_t GetFusedMatMulOpType()
{
    static_assert(OPTYPE == F_OPTYPE_NONE || OPTYPE == F_OPTYPE_16CAST32 || OPTYPE == F_OPTYPE_RELU ||
                      OPTYPE == F_OPTYPE_ADD || OPTYPE == F_OPTYPE_MUL || OPTYPE == F_OPTYPE_GELU_ERF ||
                      OPTYPE == F_OPTYPE_GELU_TANH,
                  "unsupported fused op type");
    if constexpr (OPTYPE == F_OPTYPE_NONE || OPTYPE == F_OPTYPE_16CAST32) {
        return OP_TYPE_EMPTY;
    } else if constexpr (OPTYPE == F_OPTYPE_RELU) {
        return OP_TYPE_RELU;
    } else if constexpr (OPTYPE == F_OPTYPE_ADD) {
        return OP_TYPE_ADD;
    } else if constexpr (OPTYPE == F_OPTYPE_MUL) {
        return OP_TYPE_MUL;
    } else if constexpr (OPTYPE == F_OPTYPE_GELU_ERF) {
        return OP_TYPE_GELU_ERF;
    } else {
        return OP_TYPE_GELU_TANH;
    }
}

template <int8_t FULL_LOAD, int8_t L0C2OUT_MODEL, int8_t OPTYPE, int8_t INNER_PRECISE, typename LayoutX1,
          typename LayoutX2>
__aicore__ inline void FusedMatMulBasicMm(const FusedMatMulKernelArgs& args)
{
    constexpr uint64_t fusedOpType = GetFusedMatMulOpType<OPTYPE>();
    if constexpr (OPTYPE == F_OPTYPE_GELU_ERF || OPTYPE == F_OPTYPE_GELU_TANH) {
        static_assert(FULL_LOAD == MAT_MUL_NO_FULL_LOAD && L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY,
                      "GELU MM only supports non-full-load on-the-fly output");
        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, args.tilingGM);
        MatmulV3Advanced::MatMulGeluMixWithoutQueActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, LayoutX1, LayoutX2,
                                                           layout::RowMajorAlign, fusedOpType>(
            args.x1GM, args.x2GM, args.biasGM, args.yGM, args.userWorkspaceGM, tilingData);
    } else {
        static_assert(
            OPTYPE != F_OPTYPE_NONE || (FULL_LOAD == MAT_MUL_NO_FULL_LOAD && L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY),
            "empty op type only supports basic non-full-load on-the-fly output");
        constexpr bool isAddMul = fusedOpType == OP_TYPE_ADD || fusedOpType == OP_TYPE_MUL;
        constexpr uint64_t fullLoadMode = FULL_LOAD == MAT_MUL_A_FULL_LOAD ? A_FULL_LOAD_MODE :
                                          FULL_LOAD == MAT_MUL_B_FULL_LOAD ? B_FULL_LOAD_MODE :
                                                                             MAT_MUL_NO_FULL_LOAD;
        static_assert(
            FULL_LOAD == MAT_MUL_NO_FULL_LOAD || FULL_LOAD == MAT_MUL_A_FULL_LOAD || FULL_LOAD == MAT_MUL_B_FULL_LOAD,
            "unsupported basic MM full-load mode");
        static_assert(
            L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY ||
                ((FULL_LOAD == MAT_MUL_NO_FULL_LOAD || FULL_LOAD == MAT_MUL_B_FULL_LOAD) &&
                 L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE) ||
                (isAddMul && FULL_LOAD == MAT_MUL_NO_FULL_LOAD && L0C2OUT_MODEL == MAT_MUL_1V1_ND_ALIG_FIXPIPE),
            "unsupported basic MM output mode");

        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, args.tilingGM);
        if constexpr (L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE) {
            if constexpr (isAddMul) {
                MatmulV3Advanced::MatMulFixpipeOptiActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, LayoutX1,
                                                             LayoutX2, layout::RowMajor, fullLoadMode, fusedOpType>(
                    args.x1GM, args.x2GM, args.biasGM, args.yGM, args.workspaceGM, tilingData, 0, args.x3GM);
            } else {
                MatmulV3Advanced::MatMulFixpipeOptiActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, LayoutX1,
                                                             LayoutX2, layout::RowMajor, fullLoadMode, fusedOpType>(
                    args.x1GM, args.x2GM, args.biasGM, args.yGM, args.workspaceGM, tilingData);
            }
        } else if constexpr (isAddMul) {
            MatmulV3Advanced::MatMulMixWithoutQueActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, LayoutX1, LayoutX2,
                                                           layout::RowMajor, fullLoadMode, fusedOpType, INNER_PRECISE>(
                args.x1GM, args.x2GM, args.biasGM, args.yGM, args.x3GM, tilingData);
        } else {
            MatmulV3Advanced::MatMulActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, LayoutX1, LayoutX2,
                                              layout::RowMajor, fullLoadMode, fusedOpType>(
                args.x1GM, args.x2GM, args.biasGM, args.yGM, nullptr, tilingData);
        }
    }
}

template <int8_t FULL_LOAD, int8_t L0C2OUT_MODEL, int8_t OPTYPE, int8_t INNER_PRECISE, typename LayoutX1,
          typename LayoutX2>
__aicore__ inline void FusedMatMulBasicBmm(const FusedMatMulKernelArgs& args)
{
    static_assert(
        OPTYPE == F_OPTYPE_NONE || OPTYPE == F_OPTYPE_RELU || OPTYPE == F_OPTYPE_ADD || OPTYPE == F_OPTYPE_MUL,
        "unsupported basic BMM op type");
    static_assert(OPTYPE != F_OPTYPE_NONE || FULL_LOAD == MAT_MUL_NO_FULL_LOAD,
                  "empty op type only supports non-full-load BMM");
    constexpr uint64_t fusedOpType = GetFusedMatMulOpType<OPTYPE>();
    constexpr bool isAddMul = fusedOpType == OP_TYPE_ADD || fusedOpType == OP_TYPE_MUL;
    constexpr uint64_t fullLoadMode = FULL_LOAD == MAT_MUL_A_FULL_LOAD ? A_FULL_LOAD_MODE :
                                      FULL_LOAD == MAT_MUL_B_FULL_LOAD ? B_FULL_LOAD_MODE :
                                                                         MAT_MUL_NO_FULL_LOAD;
    static_assert(
        FULL_LOAD == MAT_MUL_NO_FULL_LOAD || FULL_LOAD == MAT_MUL_A_FULL_LOAD || FULL_LOAD == MAT_MUL_B_FULL_LOAD,
        "unsupported basic BMM full-load mode");
    static_assert(L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY, "basic BMM only supports on-the-fly output");

    // Non-full-load BMM and full-load BMM use different tiling structures.
    if constexpr (FULL_LOAD == MAT_MUL_NO_FULL_LOAD) {
        GET_TILING_DATA_WITH_STRUCT(FusedMatMulTilingData, tilingData, args.tilingGM);
        const auto& batchTilingData = tilingData.matMulTilingData;
        if constexpr (isAddMul) {
            MatmulV3Advanced::MatMulMixWithoutQueActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, LayoutX1, LayoutX2,
                                                           layout::RowMajor, FULL_LOAD, fusedOpType, INNER_PRECISE>(
                args.x1GM, args.x2GM, args.biasGM, args.yGM, args.x3GM, batchTilingData.matMulTilingData,
                batchTilingData.batchDimAll, batchTilingData.batchX3);
        } else {
            MatmulV3Advanced::MatMulActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, LayoutX1, LayoutX2,
                                              layout::RowMajor, fullLoadMode, fusedOpType>(
                args.x1GM, args.x2GM, args.biasGM, args.yGM, args.workspaceGM, batchTilingData.matMulTilingData,
                batchTilingData.batchDimAll);
        }
    } else {
        GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, args.tilingGM);
        if constexpr (isAddMul) {
            MatmulV3Advanced::MatMulMixWithoutQueActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, LayoutX1, LayoutX2,
                                                           layout::RowMajor, FULL_LOAD, fusedOpType, INNER_PRECISE>(
                args.x1GM, args.x2GM, args.biasGM, args.yGM, args.x3GM, tilingData.matMulTilingData,
                tilingData.batchDimAll, tilingData.batchX3);
        } else {
            MatmulV3Advanced::MatMulActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, LayoutX1, LayoutX2,
                                              layout::RowMajor, fullLoadMode, fusedOpType>(
                args.x1GM, args.x2GM, args.biasGM, args.yGM, args.workspaceGM, tilingData.matMulTilingData,
                tilingData.batchDimAll);
        }
    }
}

template <int8_t BATCH_ITER_MODEL, int8_t L0C2OUT_MODEL, int8_t OPTYPE, int8_t INNER_PRECISE, typename LayoutX1,
          typename LayoutX2>
__aicore__ inline void FusedMatMulStreamK(const FusedMatMulKernelArgs& args)
{
    static_assert(OPTYPE == F_OPTYPE_16CAST32 || OPTYPE == F_OPTYPE_RELU || OPTYPE == F_OPTYPE_ADD ||
                      OPTYPE == F_OPTYPE_MUL || OPTYPE == F_OPTYPE_GELU_ERF || OPTYPE == F_OPTYPE_GELU_TANH,
                  "unsupported Stream-K op type");
    static_assert(BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH ||
                      (BATCH_ITER_MODEL == MAT_MUL_FOR_FUSED_BATCH &&
                       (OPTYPE == F_OPTYPE_RELU || OPTYPE == F_OPTYPE_ADD || OPTYPE == F_OPTYPE_MUL)),
                  "unsupported Stream-K batch model");
    constexpr uint64_t fusedOpType = GetFusedMatMulOpType<OPTYPE>();
    constexpr bool isAddMul = fusedOpType == OP_TYPE_ADD || fusedOpType == OP_TYPE_MUL;
    constexpr MatMulL0C2Out outputMode = L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY ? MatMulL0C2Out::ON_THE_FLY :
                                                                               MatMulL0C2Out::ND_FIXPIPE_1_2;
    static_assert(L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY || L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE,
                  "unsupported Stream-K output mode");

    if constexpr (BATCH_ITER_MODEL == MAT_MUL_FOR_FUSED_BATCH) {
        GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, args.tilingGM);
        if constexpr (isAddMul) {
            MatMulStreamKActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, LayoutX1, LayoutX2, layout::RowMajor,
                                   outputMode, fusedOpType, INNER_PRECISE>(
                args.x1GM, args.x2GM, args.biasGM, args.yGM, args.workspaceGM, tilingData.matMulTilingData,
                tilingData.batchDimAll, args.x3GM, tilingData.batchX3);
        } else {
            MatMulStreamKActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, LayoutX1, LayoutX2, layout::RowMajor,
                                   outputMode, fusedOpType>(args.x1GM, args.x2GM, args.biasGM, args.yGM,
                                                            args.workspaceGM, tilingData.matMulTilingData,
                                                            tilingData.batchDimAll);
        }
    } else {
        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, args.tilingGM);
        if constexpr (isAddMul) {
            MatMulStreamKActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, LayoutX1, LayoutX2, layout::RowMajor,
                                   outputMode, fusedOpType, INNER_PRECISE>(args.x1GM, args.x2GM, args.biasGM, args.yGM,
                                                                           args.workspaceGM, tilingData, 1, args.x3GM);
        } else {
            MatMulStreamKActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, LayoutX1, LayoutX2, layout::RowMajor,
                                   outputMode, fusedOpType>(args.x1GM, args.x2GM, args.biasGM, args.yGM,
                                                            args.workspaceGM, tilingData);
        }
    }
}

template <int8_t BATCH_ITER_MODEL, int8_t FULL_LOAD, int8_t L0C2OUT_MODEL, int8_t OPTYPE>
__aicore__ inline void FusedMatMulKEqZero(const FusedMatMulKernelArgs& args)
{
    static_assert(FULL_LOAD == MAT_MUL_NO_FULL_LOAD && L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY,
                  "unsupported K=0 output mode");
    static_assert(BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH || (BATCH_ITER_MODEL == MAT_MUL_FOR_FUSED_BATCH &&
                                                            (OPTYPE == F_OPTYPE_ADD || OPTYPE == F_OPTYPE_MUL)),
                  "unsupported K=0 batch model");
    static_assert(OPTYPE == F_OPTYPE_NONE || OPTYPE == F_OPTYPE_RELU || OPTYPE == F_OPTYPE_GELU_ERF ||
                      OPTYPE == F_OPTYPE_GELU_TANH || OPTYPE == F_OPTYPE_ADD || OPTYPE == F_OPTYPE_MUL,
                  "unsupported K=0 op type");

    if constexpr (BATCH_ITER_MODEL == MAT_MUL_FOR_FUSED_BATCH) {
        GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, args.tilingGM);
        if constexpr (OPTYPE == F_OPTYPE_ADD) {
            // Keep the batch tiling data so the copy kernel can handle x3 broadcasting.
            MatmulV3Advanced::MatMulInputKEqZeroCopyX3ToOutput<DTYPE_Y>(args.x3GM, args.yGM, tilingData);
        } else {
            TPipe pipe;
            MatMulV3KEqZeroBasicTilingData baseTilingData{};
            baseTilingData.totalDataAmount = static_cast<uint64_t>(tilingData.matMulTilingData.m) *
                                             tilingData.matMulTilingData.n * tilingData.batchDimAll;
            baseTilingData.aivNum = tilingData.matMulTilingData.usedCoreNum;
            MatmulV3Advanced::MatMulInputKEqZeroClearOutput(args.biasGM, args.yGM, baseTilingData);
        }
    } else {
        GET_TILING_DATA_WITH_STRUCT(MatMulV3KEqZeroBasicTilingData, tilingData, args.tilingGM);
        if constexpr (OPTYPE == F_OPTYPE_ADD) {
            MatmulV3Advanced::MatMulInputKEqZeroCopyX3ToOutput<DTYPE_Y>(args.x3GM, args.yGM, tilingData);
        } else {
            TPipe pipe;
            MatmulV3Advanced::MatMulInputKEqZeroClearOutput(args.biasGM, args.yGM, tilingData);
        }
    }
}

template <int8_t FULL_LOAD, int8_t L0C2OUT_MODEL, int8_t OPTYPE, int8_t INNER_PRECISE, typename LayoutX1,
          typename LayoutX2>
__aicore__ inline void FusedMatMulIterBatch(const FusedMatMulKernelArgs& args)
{
    static_assert(FULL_LOAD == MAT_MUL_NO_FULL_LOAD, "IterBatch only supports non-full-load output");
    static_assert(OPTYPE == F_OPTYPE_RELU || OPTYPE == F_OPTYPE_ADD || OPTYPE == F_OPTYPE_MUL,
                  "unsupported IterBatch op type");
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3IterBatchBasicTilingData, tilingData, args.tilingGM);
    if constexpr (OPTYPE == F_OPTYPE_RELU) {
        constexpr MatMulL0C2Out outputMode = L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY ? MatMulL0C2Out::ON_THE_FLY :
                                                                                   MatMulL0C2Out::ND_FIXPIPE_1_2;
        static_assert(L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY || L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE,
                      "unsupported relu IterBatch output mode");
        BatchMatMulActIterBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, LayoutX1, LayoutX2, layout::RowMajor,
                                      outputMode, OPTYPE>(args.x1GM, args.x2GM, args.biasGM, args.yGM, args.workspaceGM,
                                                          tilingData);
    } else {
        static_assert(L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE, "unsupported add/mul IterBatch output mode");
        BatchMatMulActIterBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, LayoutX1, LayoutX2, layout::RowMajor,
                                      MatMulL0C2Out::ND_FIXPIPE_1_2, OPTYPE, INNER_PRECISE>(
            args.x1GM, args.x2GM, args.biasGM, args.yGM, args.x3GM, tilingData);
    }
}

template <int8_t FULL_LOAD, int8_t L0C2OUT_MODEL, int8_t OPTYPE, int8_t INNER_PRECISE, typename LayoutX1,
          typename LayoutX2>
__aicore__ inline void FusedMatMulMergeBatch(const FusedMatMulKernelArgs& args)
{
    static_assert(FULL_LOAD == MAT_MUL_NO_FULL_LOAD, "MergeBatch only supports non-full-load output");
    static_assert(OPTYPE == F_OPTYPE_RELU || OPTYPE == F_OPTYPE_ADD || OPTYPE == F_OPTYPE_MUL,
                  "unsupported MergeBatch op type");
    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3MergeBatchBasicTilingData, tilingData, args.tilingGM);
    if constexpr (OPTYPE == F_OPTYPE_RELU) {
        static_assert(L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY, "unsupported relu MergeBatch output mode");
        BatchMatMulActMergeBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, LayoutX1, LayoutX2, layout::RowMajor,
                                       OPTYPE>(args.x1GM, args.x2GM, args.biasGM, args.yGM, args.workspaceGM,
                                               tilingData);
    } else {
        static_assert(L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE, "unsupported add/mul MergeBatch output mode");
        BatchMatMulActMergeBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, LayoutX1, LayoutX2, layout::RowMajor,
                                       OPTYPE, MERGE_BATCH_FIXPIPE_1V2, INNER_PRECISE>(
            args.x1GM, args.x2GM, args.biasGM, args.yGM, args.workspaceGM, tilingData, args.x3GM);
    }
}

template <int8_t BATCH_ITER_MODEL, int8_t MODEL, int8_t FULL_LOAD, int8_t L0C2OUT_MODEL, int8_t OPTYPE,
          int8_t INNER_PRECISE, typename LayoutX1, typename LayoutX2>
__aicore__ inline void FusedMatMulBasicDispatch(const FusedMatMulKernelArgs& args)
{
    if constexpr (MODEL == MAT_MUL_STREAM_K) {
        static_assert(FULL_LOAD == MAT_MUL_NO_FULL_LOAD, "Stream-K only supports non-full-load output");
        FusedMatMulStreamK<BATCH_ITER_MODEL, L0C2OUT_MODEL, OPTYPE, INNER_PRECISE, LayoutX1, LayoutX2>(args);
    } else if constexpr (MODEL == MAT_MUL_BASIC) {
        if constexpr (BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
            FusedMatMulBasicMm<FULL_LOAD, L0C2OUT_MODEL, OPTYPE, INNER_PRECISE, LayoutX1, LayoutX2>(args);
        } else if constexpr (BATCH_ITER_MODEL == MAT_MUL_FOR_FUSED_BATCH) {
            FusedMatMulBasicBmm<FULL_LOAD, L0C2OUT_MODEL, OPTYPE, INNER_PRECISE, LayoutX1, LayoutX2>(args);
        } else if constexpr (BATCH_ITER_MODEL == MAT_MUL_ITER_BATCH_SINGLE_BIAS) {
            FusedMatMulIterBatch<FULL_LOAD, L0C2OUT_MODEL, OPTYPE, INNER_PRECISE, LayoutX1, LayoutX2>(args);
        } else if constexpr (BATCH_ITER_MODEL == MAT_MUL_MERGE_BATCH) {
            FusedMatMulMergeBatch<FULL_LOAD, L0C2OUT_MODEL, OPTYPE, INNER_PRECISE, LayoutX1, LayoutX2>(args);
        } else {
            static_assert(AscendC::Std::always_false_v<decltype(BATCH_ITER_MODEL)>, "unsupported basic batch model");
        }
    } else {
        static_assert(AscendC::Std::always_false_v<decltype(MODEL)>, "unsupported fused matmul model");
    }
}

template <int8_t FULL_LOAD, int8_t L0C2OUT_MODEL, int8_t OPTYPE, int8_t INNER_PRECISE, bool aTran, bool bTran,
          typename layoutA, typename layoutB, typename layoutC>
__aicore__ inline void FusedMatMulTensorMm(const FusedMatMulKernelArgs& args)
{
    static_assert(OPTYPE != F_OPTYPE_NONE || (FULL_LOAD == MAT_MUL_NO_FULL_LOAD && L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY),
                  "empty op type only supports the basic non-full-load on-the-fly template");
    GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, args.tilingGM);
    if constexpr (OPTYPE == F_OPTYPE_SIGMOID) {
        static_assert(FULL_LOAD == MAT_MUL_NO_FULL_LOAD && L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE,
                      "Sigmoid only supports the Blaze Tensor Fixpipe template");
        FusedMatMulAdvanced::FusedMatMulSigmoidAswBasicKernel<DTYPE_X1, DTYPE_BIAS, aTran, bTran>(
            args.x1GM, args.x2GM, args.biasGM, args.yGM, args.workspaceGM, tilingData);
    } else if constexpr (OPTYPE == F_OPTYPE_ADD || OPTYPE == F_OPTYPE_MUL) {
        FusedMatMulAdvanced::FusedMatMulAddMulKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aTran, bTran, OPTYPE,
                                                     INNER_PRECISE, L0C2OUT_MODEL, FULL_LOAD>(
            args.x1GM, args.x2GM, args.biasGM, args.x3GM, args.yGM, args.workspaceGM, tilingData);
    } else if constexpr (OPTYPE == F_OPTYPE_RELU || OPTYPE == F_OPTYPE_NONE) {
        static_assert(
            FULL_LOAD == MAT_MUL_NO_FULL_LOAD || FULL_LOAD == MAT_MUL_A_FULL_LOAD || FULL_LOAD == MAT_MUL_B_FULL_LOAD,
            "unsupported Tensor MM full-load mode");
        static_assert(L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY ||
                          (L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE && FULL_LOAD != MAT_MUL_A_FULL_LOAD),
                      "unsupported Tensor MM output mode");
        if constexpr (FULL_LOAD == MAT_MUL_NO_FULL_LOAD && L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY) {
            MatmulV3Advanced::MatMulBasicKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layoutA, layoutB, layoutC,
                                                MAT_MUL_NO_FULL_LOAD, OPTYPE>(args.x1GM, args.x2GM, args.biasGM,
                                                                              args.yGM, nullptr, tilingData);
        } else if constexpr (FULL_LOAD == MAT_MUL_NO_FULL_LOAD) {
            MatmulV3Advanced::MatMulFixpipeOptiTensorKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layoutA, layoutB,
                                                            layoutC, MAT_MUL_NO_FULL_LOAD, L0C2OUT_MODEL, OPTYPE>(
                args.x1GM, args.x2GM, args.biasGM, args.yGM, args.workspaceGM, tilingData);
        } else if constexpr (FULL_LOAD == MAT_MUL_A_FULL_LOAD) {
            MatmulV3Advanced::MatMulAL1FullLoadKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layoutA, layoutB,
                                                      layoutC, A_FULL_LOAD_MODE, OPTYPE>(
                args.x1GM, args.x2GM, args.biasGM, args.yGM, nullptr, tilingData);
        } else if constexpr (L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY) {
            MatmulV3Advanced::MatMulBL1FullLoadKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layoutA, layoutB,
                                                      layoutC, B_FULL_LOAD_MODE, MAT_MUL_ON_THE_FLY, OPTYPE>(
                args.x1GM, args.x2GM, args.biasGM, args.yGM, nullptr, tilingData);
        } else {
            MatmulV3Advanced::MatMulBL1FullLoadKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layoutA, layoutB,
                                                      layoutC, B_FULL_LOAD_MODE, MAT_MUL_1V2_ND_ALIG_FIXPIPE, OPTYPE>(
                args.x1GM, args.x2GM, args.biasGM, args.yGM, args.workspaceGM, tilingData);
        }
    } else {
        static_assert(AscendC::Std::always_false_v<decltype(OPTYPE)>, "unsupported Tensor MM op type");
    }
}

template <int8_t FULL_LOAD, int8_t L0C2OUT_MODEL, int8_t OPTYPE, typename layoutA, typename layoutB, typename layoutC>
__aicore__ inline void FusedMatMulTensorBmm(const FusedMatMulKernelArgs& args)
{
    static_assert(OPTYPE != F_OPTYPE_NONE || (FULL_LOAD == MAT_MUL_NO_FULL_LOAD && L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY),
                  "empty op type only supports the basic non-full-load on-the-fly template");
    if constexpr (OPTYPE == F_OPTYPE_SCALE_ADD) {
        static_assert(FULL_LOAD == MAT_MUL_NO_FULL_LOAD && L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE,
                      "Tensor scale_add requires non-full-load Fixpipe 1V2 output");
        GET_TILING_DATA_WITH_STRUCT(FusedMatMulTilingData, tilingData, args.tilingGM);
        FusedMatMulAdvanced::FusedMatMulWithScaleAddAswBasicKernel<DTYPE_X1>(args.x1GM, args.x2GM, args.x3GM, args.yGM,
                                                                             args.workspaceGM, tilingData);
    } else if constexpr (OPTYPE == F_OPTYPE_RELU || OPTYPE == F_OPTYPE_NONE) {
        static_assert(
            FULL_LOAD == MAT_MUL_NO_FULL_LOAD || FULL_LOAD == MAT_MUL_A_FULL_LOAD || FULL_LOAD == MAT_MUL_B_FULL_LOAD,
            "unsupported Tensor BMM full-load mode");
        static_assert(L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY, "Tensor BMM only supports on-the-fly output");
        if constexpr (FULL_LOAD == MAT_MUL_NO_FULL_LOAD) {
            GET_TILING_DATA_WITH_STRUCT(FusedMatMulTilingData, tilingData, args.tilingGM);
            const auto& batchTilingData = tilingData.matMulTilingData;
            MatmulV3Advanced::MatMulBasicKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layoutA, layoutB, layoutC,
                                                MAT_MUL_NO_FULL_LOAD, OPTYPE>(
                args.x1GM, args.x2GM, args.biasGM, args.yGM, nullptr, batchTilingData.matMulTilingData,
                batchTilingData.batchDimAll);
        } else {
            GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, args.tilingGM);
            if constexpr (FULL_LOAD == MAT_MUL_A_FULL_LOAD) {
                MatmulV3Advanced::MatMulAL1FullLoadKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layoutA, layoutB,
                                                          layoutC, A_FULL_LOAD_MODE, OPTYPE>(
                    args.x1GM, args.x2GM, args.biasGM, args.yGM, nullptr, tilingData.matMulTilingData,
                    tilingData.batchDimAll);
            } else {
                MatmulV3Advanced::MatMulBL1FullLoadKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layoutA, layoutB,
                                                          layoutC, B_FULL_LOAD_MODE, MAT_MUL_ON_THE_FLY, OPTYPE>(
                    args.x1GM, args.x2GM, args.biasGM, args.yGM, nullptr, tilingData.matMulTilingData,
                    tilingData.batchDimAll);
            }
        }
    } else {
        static_assert(AscendC::Std::always_false_v<decltype(OPTYPE)>, "unsupported Tensor BMM op type");
    }
}

template <int8_t BATCH_ITER_MODEL, int8_t L0C2OUT_MODEL, int8_t OPTYPE, int8_t INNER_PRECISE, bool aTran, bool bTran,
          typename layoutA, typename layoutB>
__aicore__ inline void FusedMatMulTensorStreamK(const FusedMatMulKernelArgs& args)
{
    if constexpr (OPTYPE == F_OPTYPE_ADD || OPTYPE == F_OPTYPE_MUL) {
        static_assert(BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH, "Tensor add/mul Stream-K only supports MM");
        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, args.tilingGM);
        FusedMatMulAdvanced::FusedMatMulAddMulStreamKKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aTran, bTran,
                                                            OPTYPE, INNER_PRECISE, L0C2OUT_MODEL>(
            args.x1GM, args.x2GM, args.biasGM, args.x3GM, args.yGM, args.workspaceGM, tilingData);
    } else if constexpr (OPTYPE == F_OPTYPE_RELU) {
        static_assert(L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY || L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE,
                      "unsupported Tensor Stream-K output mode");
        constexpr auto outputMode = L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY ? Blaze::Gemm::MatMulL0C2Out::ON_THE_FLY :
                                                                          Blaze::Gemm::MatMulL0C2Out::ND_FIXPIPE_1_2;
        if constexpr (BATCH_ITER_MODEL == MAT_MUL_FOR_FUSED_BATCH) {
            GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, args.tilingGM);
            MatmulV3Advanced::MatMulStreamKKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layoutA, layoutB, outputMode,
                                                  OPTYPE>(args.x1GM, args.x2GM, args.biasGM, args.yGM, args.workspaceGM,
                                                          tilingData.matMulTilingData, tilingData.batchDimAll);
        } else if constexpr (BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
            GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, args.tilingGM);
            MatmulV3Advanced::MatMulStreamKKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, layoutA, layoutB, outputMode,
                                                  OPTYPE>(args.x1GM, args.x2GM, args.biasGM, args.yGM, args.workspaceGM,
                                                          tilingData);
        } else {
            static_assert(AscendC::Std::always_false_v<decltype(BATCH_ITER_MODEL)>,
                          "unsupported Tensor Stream-K batch model");
        }
    } else {
        static_assert(AscendC::Std::always_false_v<decltype(OPTYPE)>, "unsupported Tensor Stream-K op type");
    }
}

template <int8_t BATCH_ITER_MODEL, int8_t MODEL, int8_t FULL_LOAD, int8_t L0C2OUT_MODEL, int8_t OPTYPE,
          int8_t INNER_PRECISE, bool aTran, bool bTran, typename layoutA, typename layoutB, typename layoutC>
__aicore__ inline void FusedMatMulTensorDispatch(const FusedMatMulKernelArgs& args)
{
    if constexpr (MODEL == MAT_MUL_K_EQUAL_ZERO) {
        FusedMatMulKEqZero<BATCH_ITER_MODEL, FULL_LOAD, L0C2OUT_MODEL, OPTYPE>(args);
    } else if constexpr (MODEL == MAT_MUL_STREAM_K) {
        FusedMatMulTensorStreamK<BATCH_ITER_MODEL, L0C2OUT_MODEL, OPTYPE, INNER_PRECISE, aTran, bTran, layoutA,
                                 layoutB>(args);
    } else if constexpr (MODEL == MAT_MUL_BASIC) {
        if constexpr (BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
            FusedMatMulTensorMm<FULL_LOAD, L0C2OUT_MODEL, OPTYPE, INNER_PRECISE, aTran, bTran, layoutA, layoutB,
                                layoutC>(args);
        } else if constexpr (BATCH_ITER_MODEL == MAT_MUL_FOR_FUSED_BATCH) {
            FusedMatMulTensorBmm<FULL_LOAD, L0C2OUT_MODEL, OPTYPE, layoutA, layoutB, layoutC>(args);
        } else {
            static_assert(AscendC::Std::always_false_v<decltype(BATCH_ITER_MODEL)>, "unsupported Tensor batch model");
        }
    } else {
        static_assert(AscendC::Std::always_false_v<decltype(MODEL)>, "unsupported Tensor model");
    }
}

#endif
template <int8_t API_LEVEL, int8_t TRANS_MODEL, int8_t BATCH_ITER_MODEL, int8_t MODEL, int8_t FULL_LOAD,
          int8_t L0C2OUT_MODEL, int8_t OPTYPE, int8_t INNER_PRECISE>
__global__ __aicore__ void fused_mat_mul(GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR biasGM, GM_ADDR x3GM, GM_ADDR yGM,
                                         GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    AscendC::InitSocState();
    __gm__ uint8_t* userWorkspaceGM = AscendC::GetUserWorkspace(workspaceGM);

    constexpr bool aTran = (TRANS_MODEL == 1 || TRANS_MODEL == 3);
    constexpr bool bTran = (TRANS_MODEL == 2 || TRANS_MODEL == 3);
    using aLayout = std::conditional_t<aTran, layout::ColumnMajor, layout::RowMajor>;
    using bLayout = std::conditional_t<bTran, layout::ColumnMajor, layout::RowMajor>;

#if !__FIXED_POINT_ONLY_CUBE_TO_L0C__
    using layoutA = AscendC::Std::conditional_t<aTran, AscendC::Te::DNExtLayoutPtn, AscendC::Te::NDExtLayoutPtn>;
    using layoutB = AscendC::Std::conditional_t<
        bTran,
        AscendC::Std::conditional_t<(format_x2 == CubeFormat::NZ), AscendC::Te::ZNLayoutPtn,
                                    AscendC::Te::DNExtLayoutPtn>,
        AscendC::Std::conditional_t<(format_x2 == CubeFormat::NZ), AscendC::Te::NZLayoutPtn,
                                    AscendC::Te::NDExtLayoutPtn>>;
    // C GM layout
    using layoutC = AscendC::Te::NDExtLayoutPtn;
#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
    REGISTER_TILING_DEFAULT(FusedMatMulTilingData);
    const FusedMatMulKernelArgs args{x1GM, x2GM, biasGM, x3GM, yGM, workspaceGM, tilingGM, userWorkspaceGM};
    if constexpr (API_LEVEL == MAT_MUL_BASIC_LEVEL) {
        FusedMatMulBasicDispatch<BATCH_ITER_MODEL, MODEL, FULL_LOAD, L0C2OUT_MODEL, OPTYPE, INNER_PRECISE, aLayout,
                                 bLayout>(args);
    } else if constexpr (API_LEVEL == MAT_MUL_TENSOR_LEVEL) {
        FusedMatMulTensorDispatch<BATCH_ITER_MODEL, MODEL, FULL_LOAD, L0C2OUT_MODEL, OPTYPE, INNER_PRECISE, aTran,
                                  bTran, layoutA, layoutB, layoutC>(args);
    }
#endif

#if __FIXED_POINT_ONLY_CUBE_TO_L0C__
    REGISTER_TILING_DEFAULT(BatchMatMulV3TilingData);
    if constexpr (L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY && MODEL == MAT_MUL_BASIC) {
        if constexpr ( // high, aswt, from bmmv3
            API_LEVEL == MAT_MUL_HIGH_LEVEL && FULL_LOAD == MAT_MUL_NO_FULL_LOAD &&
            BATCH_ITER_MODEL == MAT_MUL_FOR_FUSED_BATCH) {
            if constexpr (OPTYPE == F_OPTYPE_QUANT) {
                BMMV3_IMPL_CLASS_COMMON_TRANS(aTran, bTran, BatchMatMulV3Advanced::BatchMatMulAswKernel,
                                              BatchMatMulV3Advanced::BatchMatMulAswBlock, MM_CFG_NO_PRELOAD, true);
            } else if constexpr (OPTYPE == F_OPTYPE_RELU_QUANT) {
                BMMV3_IMPL_CLASS_COMMON_TRANS(aTran, bTran, BatchMatMulV3Advanced::BatchMatMulAswKernel,
                                              BatchMatMulV3Advanced::BatchMatMulAswBlock, MM_CFG_NO_PRELOAD_RELU, true);
            } else {
                BMMV3_IMPL_CLASS_COMMON_TRANS(aTran, bTran, BatchMatMulV3Advanced::BatchMatMulAswKernel,
                                              BatchMatMulV3Advanced::BatchMatMulAswBlock, MM_CFG_NO_PRELOAD_RELU);
            }
        } else if constexpr ( // basic, aswt, from bmmv3
            API_LEVEL == MAT_MUL_BASIC_LEVEL && FULL_LOAD == MAT_MUL_NO_FULL_LOAD &&
            BATCH_ITER_MODEL == MAT_MUL_FOR_FUSED_BATCH) {
            GET_TILING_DATA_WITH_STRUCT(FusedMatMulTilingData, tilingData, tilingGM);
            const auto& batchTilingData = tilingData.matMulTilingData;
            MatmulV3Advanced::MatMulActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout,
                                              layout::RowMajor, MAT_MUL_NO_FULL_LOAD, OPTYPE>(
                x1GM, x2GM, biasGM, yGM, workspaceGM, batchTilingData.matMulTilingData, batchTilingData.batchDimAll,
                x3GM);
        } else if constexpr ( // basic, aswt al1 fullload, from bmmv3
            API_LEVEL == MAT_MUL_BASIC_LEVEL && FULL_LOAD == MAT_MUL_A_FULL_LOAD &&
            BATCH_ITER_MODEL == MAT_MUL_FOR_FUSED_BATCH) {
            GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, tilingGM);
            MatmulV3Advanced::MatMulActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout,
                                              layout::RowMajor, A_FULL_LOAD_MODE, OPTYPE>(
                x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData.matMulTilingData, tilingData.batchDimAll, x3GM);
        } else if constexpr ( // basic, aswt bl1 fullload, from bmmv3
            API_LEVEL == MAT_MUL_BASIC_LEVEL && FULL_LOAD == MAT_MUL_B_FULL_LOAD &&
            BATCH_ITER_MODEL == MAT_MUL_FOR_FUSED_BATCH) {
            GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, tilingGM);
            MatmulV3Advanced::MatMulActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout,
                                              layout::RowMajor, B_FULL_LOAD_MODE, OPTYPE>(
                x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData.matMulTilingData, tilingData.batchDimAll, x3GM);
        } else if constexpr ( // basic, iterbatch, from bmmv3
            API_LEVEL == MAT_MUL_BASIC_LEVEL && FULL_LOAD == MAT_MUL_NO_FULL_LOAD &&
            BATCH_ITER_MODEL == MAT_MUL_ITER_BATCH_SINGLE_BIAS) {
            GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3IterBatchBasicTilingData, tilingData, tilingGM);
            BatchMatMulActIterBatchKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                                          MatMulL0C2Out::ON_THE_FLY, OPTYPE>(x1GM, x2GM, biasGM, yGM, workspaceGM,
                                                                             tilingData, x3GM);
        } else if constexpr ( // basic, aswt, from mmv3
            API_LEVEL == MAT_MUL_BASIC_LEVEL && FULL_LOAD == MAT_MUL_NO_FULL_LOAD &&
            BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
            GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
            MatmulV3Advanced::MatMulActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout,
                                              layout::RowMajor, MAT_MUL_NO_FULL_LOAD, OPTYPE>(
                x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData, 0, x3GM);
        } else if constexpr ( // basic, aswt al1 fullload, from mmv3
            API_LEVEL == MAT_MUL_BASIC_LEVEL && FULL_LOAD == MAT_MUL_A_FULL_LOAD &&
            BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
            GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
            MatmulV3Advanced::MatMulActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout,
                                              layout::RowMajor, A_FULL_LOAD_MODE, OPTYPE>(
                x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData, 0, x3GM);
        } else if constexpr ( // basic, aswt bl1 fullload, from mmv3
            API_LEVEL == MAT_MUL_BASIC_LEVEL && FULL_LOAD == MAT_MUL_B_FULL_LOAD &&
            BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
            GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
            MatmulV3Advanced::MatMulActKernel<DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout,
                                              layout::RowMajor, B_FULL_LOAD_MODE, OPTYPE>(
                x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData, 0, x3GM);
        }
    }
#endif
}

#endif
