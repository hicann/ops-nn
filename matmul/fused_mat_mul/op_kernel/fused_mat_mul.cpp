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

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510 || __NPU_ARCH__ == 5102)
#include "../mat_mul_v3/mat_mul_v3_common.h"                                     // 共有,BLOCK_SIZE等定义
#include "./arch35/fused_mat_mul_tilingkey.h"                                    // 共有,tilingKey模板参数
#include "../mat_mul_v3/arch35/mat_mul_tiling_data.h"                            // 共有,tilingData定义

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
#include "./arch35/fused_mat_mul_tiling_data.h"                   // 3510独有,FusedMatMulTilingData的定义
#include "cmct/kernel/kernel_matmul_mix.h"                        // 3510独有,KernelMatmulMix类
#include "cmct/kernel/kernel_matmul.h"                            // 3510独有,KernelMatmul类
#include "cmct/kernel/kernel_matmul_iterbatch.h"                  // 没有用到,确认后可删除
#include "cmct/kernel/kernel_matmul_mix_without_que.h"            // 没有用到,确认后可删除
#include "../mat_mul_v3/arch35/mat_mul_streamk_basic_cmct.h"      // 3510独有,低阶API实现,MM,streamK模板
#include "../mat_mul_v3/arch35/mat_mul_mix_basic_cmct.h"          // 3510独有,低阶API实现,MM,MixWithoutQue模板
#include "../mat_mul_v3/arch35/mat_mul_fixpipe_opti_basic_cmct.h" // 3510独有,低阶API实现,FixpipeOpti模板
#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
#include "../batch_mat_mul_v3/arch35/batch_mat_mul_v3_asw_kernel_advanced.h" // 5102独有,高阶API实现,BMM,aswt模板
#endif

#include "../batch_mat_mul_v3/arch35/batch_mat_mul_v3_iterbatch_basicapi_cmct.h" // 共有,低阶API实现,BMM,iterbatch模板
#include "../mat_mul_v3/arch35/mat_mul_pingpong_basic_cmct.h"                    // 共有,低阶API实现,MM/BMM,aswt模板

using namespace Cmct;
using namespace Gemm;
using namespace AscendC;

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

#if defined(FORMAT_X3) && FORMAT_X3 == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_x3 = CubeFormat::NZ;
#else
constexpr CubeFormat format_x3 = CubeFormat::ND;
#endif

using L1TileShape = AscendC::Shape<_256, _256, _256>;
using L0TileShape = AscendC::Shape<_256, _256, _64>;

enum class FusionOpType : uint8_t
{
    ADD,
    MUL,
    GELU,
    GELU_ERF
};

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
#define BMMV3_IMPL_CLASS_COMMON_TRNAS(transA, transB, templateClass, ...)                \
    do {                                                                                 \
        GET_TILING_DATA(tilingData, tilingGM);                                           \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y>;             \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>; \
        TPipe pipe;                                                                      \
        using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, transA>;   \
        using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, transB>;   \
        templateClass<aType, bType, cType, biasType, __VA_ARGS__> op;                    \
        op.Init(x1GM, x2GM, yGM, biasGM, nullptr, user, &tilingData, &pipe);             \
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

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
template <typename MatmulKernel, typename BlockMmadArgs, typename BlockEpilogueArgs>
__aicore__ void inline KernelFunc(
    const FusedMatMulTilingData& tilingData, const BlockMmadArgs& mmadArgs, const BlockEpilogueArgs& epilogueArgs,
    GM_ADDR workspaceGM)
{
    using Arguments = typename MatmulKernel::Arguments;
    using Params = typename MatmulKernel::Params;
    MatmulShape shape{tilingData.M, tilingData.N, tilingData.K, 1};
    Arguments args = {
        shape,       // problem shape
        mmadArgs,    // mmad args
        epilogueArgs // epilogue args
    };
    Params params = MatmulKernel::InitParams(args, workspaceGM);
    AscendC::TPipe tPipe;
    MatmulKernel op;
    op(params);
}

template <typename LayoutX1, typename LayoutX2>
__aicore__ void inline FusedEmptyMatMul(
    GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR biasGM, GM_ADDR x3GM, GM_ADDR yGM, GM_ADDR workspaceGM,
    const FusedMatMulTilingData& tilingData)
{
    using BlockMmad = Block::BlockMmadBuilder<
        DTYPE_X1, LayoutX1, DTYPE_X2, LayoutX2, DTYPE_Y, layout::RowMajor, DTYPE_BIAS, layout::RowMajor, L1TileShape,
        L0TileShape, IterateKScheduler, MatmulMultiBlockWithLayout<>>;
    using MmadArgs = typename BlockMmad::Arguments;
    MmadArgs mmadArgs{x1GM, x2GM, yGM, biasGM};

    using Epilogue = Block::BlockEpilogueEmpty;
    using EpilogueArgs = typename Epilogue::Arguments;
    EpilogueArgs epilogueArgs{};
    using MatmulKernel = Kernel::KernelMatmul<MatmulShape, BlockMmad, Epilogue, IterateKScheduler>;
    KernelFunc<MatmulKernel, MmadArgs, EpilogueArgs>(tilingData, mmadArgs, epilogueArgs, workspaceGM);
}

template <typename LayoutX1, typename LayoutX2, FusionOpType optype>
__aicore__ void inline FusedOneEleMatMul(
    GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR biasGM, GM_ADDR x3GM, GM_ADDR yGM, GM_ADDR workspaceGM,
    const FusedMatMulTilingData& tilingData)
{
    using BlockMmad = Block::BlockMmadBuilder<
        DTYPE_X1, LayoutX1, DTYPE_X2, LayoutX2, DTYPE_Y, layout::RowMajorAlign, DTYPE_BIAS, layout::RowMajor,
        L1TileShape, L0TileShape, IterateKScheduler, MatmulMultiBlockWithLayout<>>;
    using MmadArgs = typename BlockMmad::Arguments;
    MmadArgs mmadArgs{x1GM, x2GM, yGM, biasGM};
    if constexpr (optype == FusionOpType::ADD) {
        using Epilogue = Block::BlockEpilogue<L0TileShape, DTYPE_Y, DTYPE_Y, Block::FusionAdd<DTYPE_Y, DTYPE_Y>>;
        using EpilogueArgs = typename Epilogue::Arguments;
        EpilogueArgs epilogueArgs{yGM, {x3GM}};
        using MatmulKernel = Kernel::KernelMatmulMix<MatmulShape, BlockMmad, Epilogue, IterateKScheduler>;
        KernelFunc<MatmulKernel, MmadArgs, EpilogueArgs>(tilingData, mmadArgs, epilogueArgs, workspaceGM);
    } else if constexpr (optype == FusionOpType::MUL) {
        using Epilogue = Block::BlockEpilogue<L0TileShape, DTYPE_Y, DTYPE_Y, Block::FusionMul<DTYPE_Y, DTYPE_Y>>;
        using EpilogueArgs = typename Epilogue::Arguments;
        EpilogueArgs epilogueArgs{yGM, {x3GM}};
        using MatmulKernel = Kernel::KernelMatmulMix<MatmulShape, BlockMmad, Epilogue, IterateKScheduler>;
        KernelFunc<MatmulKernel, MmadArgs, EpilogueArgs>(tilingData, mmadArgs, epilogueArgs, workspaceGM);
    }
}

template <typename LayoutX1, typename LayoutX2, Block::GeluApproxiMate mate>
__aicore__ void inline FusedGeluMatMul(
    GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR biasGM, GM_ADDR x3GM, GM_ADDR yGM, GM_ADDR workspaceGM,
    const FusedMatMulTilingData& tilingData)
{
    using BlockMmad = Block::BlockMmadBuilder<
        DTYPE_X1, LayoutX1, DTYPE_X2, LayoutX2, float, layout::RowMajorAlign, DTYPE_BIAS, layout::RowMajor, L1TileShape,
        L0TileShape, IterateKScheduler, MatmulMultiBlockWithLayout<>,
        Tile::TileCopy<Arch::DAV3510, Tile::CopyOutSplitMWithParams>>;
    using MmadArgs = typename BlockMmad::Arguments;
    MmadArgs mmadArgs{x1GM, x2GM, yGM, biasGM};
    using Epilogue = Block::BlockEpilogue<L0TileShape, DTYPE_Y, float, Block::FusionGelu<float, float, mate>>;
    using EpilogueArgs = typename Epilogue::Arguments;
    EpilogueArgs epilogueArgs{yGM, {nullptr}};
    using MatmulKernel = Kernel::KernelMatmulMix<MatmulShape, BlockMmad, Epilogue, IterateKScheduler>;
    KernelFunc<MatmulKernel, MmadArgs, EpilogueArgs>(tilingData, mmadArgs, epilogueArgs, workspaceGM);
}
#endif
template <
    int8_t API_LEVEL, int8_t TRANS_MODEL, int8_t BATCH_ITER_MODEL, int8_t MODEL, int8_t FULL_LOAD, int8_t L0C2OUT_MODEL,
    int8_t OPTYPE>
__global__ __aicore__ void fused_mat_mul(
    GM_ADDR x1GM, GM_ADDR x2GM, GM_ADDR biasGM, GM_ADDR x3GM, GM_ADDR yGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    __gm__ uint8_t* user = AscendC::GetUserWorkspace(workspaceGM);

    constexpr bool aTran = (TRANS_MODEL == 1 || TRANS_MODEL == 3);
    constexpr bool bTran = (TRANS_MODEL == 2 || TRANS_MODEL == 3);
    using aLayout = std::conditional_t<aTran, layout::ColumnMajor, layout::RowMajor>;
    using bLayout = std::conditional_t<bTran, layout::ColumnMajor, layout::RowMajor>;
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    REGISTER_TILING_DEFAULT(FusedMatMulTilingData);
    if constexpr (API_LEVEL == MAT_MUL_BASIC_LEVEL) { // 基础API
        if constexpr (OPTYPE == F_OPTYPE_NONE) { // opType=empty
            if constexpr (BATCH_ITER_MODEL == MAT_MUL_ITER_BATCH_SINGLE_BIAS) {
                if constexpr (L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY) {
                    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3IterBatchBasicTilingData, tilingData, tilingGM);
                    BatchMatMulActIterBatchKernel<
                        DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor>(
                        x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData);
                } else if constexpr (L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE) {
                    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3IterBatchBasicTilingData, tilingData, tilingGM);
                    BatchMatMulActIterBatchKernel<
                        DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                        MatMulL0C2Out::ND_FIXPIPE_1_2>(x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData);
                } else {
                    static_assert(AscendC::Std::always_false_v<decltype(L0C2OUT_MODEL)>, "not support yet");
                }
            } else if constexpr (MODEL == MAT_MUL_BASIC) {
                if constexpr (FULL_LOAD == MAT_MUL_NO_FULL_LOAD) {
                    if constexpr (L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY) {
                        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
                        MatmulV3Advanced::MatMulActKernel<
                            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                            MAT_MUL_NO_FULL_LOAD, OP_TYPE_EMPTY>(x1GM, x2GM, biasGM, yGM, nullptr, tilingData);
                    } else if constexpr (L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE) {
                        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
                        MatmulV3Advanced::MatMulFixpipeOptiActKernel<
                            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                            MAT_MUL_NO_FULL_LOAD, OP_TYPE_EMPTY>(x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData);
                    } else {
                        static_assert(AscendC::Std::always_false_v<decltype(L0C2OUT_MODEL)>, "not support yet");
                    }
                } else if constexpr (FULL_LOAD == MAT_MUL_A_FULL_LOAD) {
                    if constexpr (L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY) {
                        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
                        MatmulV3Advanced::MatMulActKernel<
                            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                            A_FULL_LOAD_MODE, OP_TYPE_EMPTY>(x1GM, x2GM, biasGM, yGM, nullptr, tilingData);
                    } else {
                        static_assert(AscendC::Std::always_false_v<decltype(L0C2OUT_MODEL)>, "not support yet");
                    }
                } else if constexpr (FULL_LOAD == MAT_MUL_B_FULL_LOAD) {
                    if constexpr (L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY) {
                        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
                        MatmulV3Advanced::MatMulActKernel<
                            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                            B_FULL_LOAD_MODE, OP_TYPE_EMPTY>(x1GM, x2GM, biasGM, yGM, nullptr, tilingData);
                    } else if constexpr (L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE) {
                        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
                        MatmulV3Advanced::MatMulFixpipeOptiActKernel<
                            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                            B_FULL_LOAD_MODE, OP_TYPE_EMPTY>(x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData);
                    } else {
                        static_assert(AscendC::Std::always_false_v<decltype(L0C2OUT_MODEL)>, "not support yet");
                    }
                } else {
                    static_assert(AscendC::Std::always_false_v<decltype(FULL_LOAD)>, "not support yet");
                }
            } else if constexpr (MODEL == MAT_MUL_STREAM_K) {
                if constexpr (L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY) {
                    GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
                    MatMulStreamKActKernel<
                        DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                        MatMulL0C2Out::ON_THE_FLY, OP_TYPE_EMPTY>(x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData);
                } else if constexpr (L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE) {
                    GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
                    MatMulStreamKActKernel<
                        DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                        MatMulL0C2Out::ND_FIXPIPE_1_2, OP_TYPE_EMPTY>(x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData);
                } else {
                    static_assert(AscendC::Std::always_false_v<decltype(L0C2OUT_MODEL)>, "not support yet");
                }
            } else {
                static_assert(AscendC::Std::always_false_v<decltype(MODEL)>, "not support yet");
            }
        } else if constexpr (OPTYPE == F_OPTYPE_16CAST32) { // 新增opType=16cast32场景 kernel同opType=empty
            if constexpr (MODEL == MAT_MUL_BASIC) {
                if constexpr (FULL_LOAD == MAT_MUL_NO_FULL_LOAD) {
                    if constexpr (L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY) {
                        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
                        MatmulV3Advanced::MatMulActKernel<
                            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                            MAT_MUL_NO_FULL_LOAD, OP_TYPE_EMPTY>(x1GM, x2GM, biasGM, yGM, nullptr, tilingData);
                    } else if constexpr (L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE) {
                        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
                        MatmulV3Advanced::MatMulFixpipeOptiActKernel<
                            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                            MAT_MUL_NO_FULL_LOAD, OP_TYPE_EMPTY>(x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData);
                    } else {
                        static_assert(AscendC::Std::always_false_v<decltype(L0C2OUT_MODEL)>, "not support yet");
                    }
                } else if constexpr (FULL_LOAD == MAT_MUL_A_FULL_LOAD) {
                    // 仅支持 on the fly
                    if constexpr (L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY) {
                        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
                        MatmulV3Advanced::MatMulActKernel<
                            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                            A_FULL_LOAD_MODE, OP_TYPE_EMPTY>(x1GM, x2GM, biasGM, yGM, nullptr, tilingData);
                    } else {
                        static_assert(AscendC::Std::always_false_v<decltype(L0C2OUT_MODEL)>, "not support yet");
                    }
                } else if constexpr (FULL_LOAD == MAT_MUL_B_FULL_LOAD) {
                    if constexpr (L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY) {
                        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
                        MatmulV3Advanced::MatMulActKernel<
                            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                            B_FULL_LOAD_MODE, OP_TYPE_EMPTY>(x1GM, x2GM, biasGM, yGM, nullptr, tilingData);
                    } else if constexpr (L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE) {
                        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
                        MatmulV3Advanced::MatMulFixpipeOptiActKernel<
                            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                            B_FULL_LOAD_MODE, OP_TYPE_EMPTY>(x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData);
                    } else {
                        static_assert(AscendC::Std::always_false_v<decltype(L0C2OUT_MODEL)>, "not support yet");
                    }
                } else {
                    static_assert(AscendC::Std::always_false_v<decltype(FULL_LOAD)>, "not support yet");
                }
            } else if constexpr (MODEL == MAT_MUL_STREAM_K) {
                if constexpr (L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY) {
                    GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
                    MatMulStreamKActKernel<
                        DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                        MatMulL0C2Out::ON_THE_FLY, OP_TYPE_EMPTY>(x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData);
                } else if constexpr (L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE) {
                    GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
                    MatMulStreamKActKernel<
                        DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                        MatMulL0C2Out::ND_FIXPIPE_1_2, OP_TYPE_EMPTY>(x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData);
                } else {
                    static_assert(AscendC::Std::always_false_v<decltype(L0C2OUT_MODEL)>, "not support yet");
                }
            } else {
                static_assert(AscendC::Std::always_false_v<decltype(MODEL)>, "not support yet");
            }
        } else if constexpr (OPTYPE == F_OPTYPE_RELU) { // Relu
            if constexpr (MODEL == MAT_MUL_BASIC) {
                if constexpr (FULL_LOAD == MAT_MUL_NO_FULL_LOAD) {
                    if constexpr (L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY) {
                        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
                        MatmulV3Advanced::MatMulActKernel<
                            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                            MAT_MUL_NO_FULL_LOAD, OP_TYPE_RELU>(x1GM, x2GM, biasGM, yGM, nullptr, tilingData);
                    } else if constexpr (L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE) {
                        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
                        MatmulV3Advanced::MatMulFixpipeOptiActKernel<
                            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                            MAT_MUL_NO_FULL_LOAD, OP_TYPE_RELU>(x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData);
                    } else {
                        static_assert(AscendC::Std::always_false_v<decltype(L0C2OUT_MODEL)>, "not support yet");
                    }
                } else if constexpr (FULL_LOAD == MAT_MUL_A_FULL_LOAD) {
                    // 仅支持 on the fly
                    if constexpr (L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY) {
                        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
                        MatmulV3Advanced::MatMulActKernel<
                            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                            A_FULL_LOAD_MODE, OP_TYPE_RELU>(x1GM, x2GM, biasGM, yGM, nullptr, tilingData);
                    } else {
                        static_assert(AscendC::Std::always_false_v<decltype(L0C2OUT_MODEL)>, "not support yet");
                    }
                } else if constexpr (FULL_LOAD == MAT_MUL_B_FULL_LOAD) {
                    if constexpr (L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY) {
                        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
                        MatmulV3Advanced::MatMulActKernel<
                            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                            B_FULL_LOAD_MODE, OP_TYPE_RELU>(x1GM, x2GM, biasGM, yGM, nullptr, tilingData);
                    } else if constexpr (L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE) {
                        GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
                        MatmulV3Advanced::MatMulFixpipeOptiActKernel<
                            DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                            B_FULL_LOAD_MODE, OP_TYPE_RELU>(x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData);
                    } else {
                        static_assert(AscendC::Std::always_false_v<decltype(L0C2OUT_MODEL)>, "not support yet");
                    }
                } else {
                    static_assert(AscendC::Std::always_false_v<decltype(FULL_LOAD)>, "not support yet");
                }
            } else if constexpr (MODEL == MAT_MUL_STREAM_K) {
                if constexpr (L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY) {
                    GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
                    MatMulStreamKActKernel<
                        DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                        MatMulL0C2Out::ON_THE_FLY, OP_TYPE_RELU>(x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData);
                } else if constexpr (L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE) {
                    GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
                    MatMulStreamKActKernel<
                        DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                        MatMulL0C2Out::ND_FIXPIPE_1_2, OP_TYPE_RELU>(x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData);
                } else {
                    static_assert(AscendC::Std::always_false_v<decltype(L0C2OUT_MODEL)>, "not support yet");
                }
            } else {
                static_assert(AscendC::Std::always_false_v<decltype(MODEL)>, "not support yet");
            }
        } else if constexpr (OPTYPE == F_OPTYPE_ADD) { // Add
            if constexpr (BATCH_ITER_MODEL == MAT_MUL_ITER_BATCH_SINGLE_BIAS) {
                if constexpr (L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE) {
                    GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3IterBatchBasicTilingData, tilingData, tilingGM);
                    BatchMatMulActIterBatchKernel<
                        DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor,
                        MatMulL0C2Out::ND_FIXPIPE_1_2, F_OPTYPE_ADD>(x1GM, x2GM, biasGM, yGM, x3GM, tilingData);
                } else {
                    static_assert(AscendC::Std::always_false_v<decltype(L0C2OUT_MODEL)>, "not support yet");
                }
            } else if constexpr (MODEL == MAT_MUL_BASIC) {
                if constexpr (
                    L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY || L0C2OUT_MODEL == MAT_MUL_1V1_ND_ALIG_FIXPIPE ||
                    L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE) { // tilingKey保持loc2Out切换基础API后不变
                    GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
                    // Add当前仅支持ASWT模板
                    MatmulV3Advanced::MatMulMixWithoutQueActKernel<
                        DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor, FULL_LOAD,
                        OP_TYPE_ADD>(x1GM, x2GM, biasGM, yGM, x3GM, tilingData);
                } else {
                    static_assert(AscendC::Std::always_false_v<decltype(L0C2OUT_MODEL)>, "not support yet");
                }
            } else { // 不支持非Basic
                static_assert(AscendC::Std::always_false_v<decltype(MODEL)>, "not support yet");
            }
        } else if constexpr (OPTYPE == F_OPTYPE_MUL) { // Mul
            if constexpr (MODEL == MAT_MUL_BASIC) {
                if constexpr (
                    L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY || L0C2OUT_MODEL == MAT_MUL_1V1_ND_ALIG_FIXPIPE ||
                    L0C2OUT_MODEL == MAT_MUL_1V2_ND_ALIG_FIXPIPE) { // tilingKey保持loc2Out切换基础API后不变
                    GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
                    // Add当前仅支持ASWT模板
                    MatmulV3Advanced::MatMulMixWithoutQueActKernel<
                        DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor, FULL_LOAD,
                        OP_TYPE_MUL>(x1GM, x2GM, biasGM, yGM, x3GM, tilingData);
                } else { 
                    static_assert(AscendC::Std::always_false_v<decltype(L0C2OUT_MODEL)>, "not support yet");
                }
            } else { // 不支持非Basic
                static_assert(AscendC::Std::always_false_v<decltype(MODEL)>, "not support yet");
            }
        }
    } else { // 高阶API
        GET_TILING_DATA_WITH_STRUCT(FusedMatMulTilingData, tilingData, tilingGM);
        if constexpr (OPTYPE == F_OPTYPE_GELU_ERF) {
            FusedGeluMatMul<aLayout, bLayout, Block::GeluApproxiMate::ERF>(
                x1GM, x2GM, biasGM, x3GM, yGM, workspaceGM, tilingData);
        } else if constexpr (OPTYPE == F_OPTYPE_GELU_TANH) {
            FusedGeluMatMul<aLayout, bLayout, Block::GeluApproxiMate::TANH>(
                x1GM, x2GM, biasGM, x3GM, yGM, workspaceGM, tilingData);
        }
    }
#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 5102)
    REGISTER_TILING_DEFAULT(BatchMatMulV3TilingData);
    if constexpr (OPTYPE == F_OPTYPE_RELU && L0C2OUT_MODEL == MAT_MUL_ON_THE_FLY && MODEL == MAT_MUL_BASIC) {
        if constexpr ( // high, aswt, from bmmv3
            API_LEVEL == MAT_MUL_HIGH_LEVEL && FULL_LOAD == MAT_MUL_NO_FULL_LOAD &&
            BATCH_ITER_MODEL == MAT_MUL_FOR_FUSED_BATCH) {
            BMMV3_IMPL_CLASS_COMMON_TRNAS(
                aTran, bTran, BatchMatMulV3Advanced::BatchMatMulAswKernel, BatchMatMulV3Advanced::BatchMatMulAswBlock,
                MM_CFG_NO_PRELOAD_RELU);
        } else if constexpr ( // basic, aswt, from bmmv3
            API_LEVEL == MAT_MUL_BASIC_LEVEL && FULL_LOAD == MAT_MUL_NO_FULL_LOAD &&
            BATCH_ITER_MODEL == MAT_MUL_FOR_FUSED_BATCH) {
            GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, tilingGM);
            MatmulV3Advanced::MatMulActKernel<
                DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor, MAT_MUL_NO_FULL_LOAD,
                OP_TYPE_RELU>(
                x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData.matMulTilingData, tilingData.batchDimAll);
        } else if constexpr ( // basic, aswt al1 fullload, from bmmv3
            API_LEVEL == MAT_MUL_BASIC_LEVEL && FULL_LOAD == MAT_MUL_A_FULL_LOAD &&
            BATCH_ITER_MODEL == MAT_MUL_FOR_FUSED_BATCH) {
            GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, tilingGM);
            MatmulV3Advanced::MatMulActKernel<
                DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor, A_FULL_LOAD_MODE,
                OP_TYPE_RELU>(
                x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData.matMulTilingData, tilingData.batchDimAll);
        } else if constexpr ( // basic, aswt bl1 fullload, from bmmv3
            API_LEVEL == MAT_MUL_BASIC_LEVEL && FULL_LOAD == MAT_MUL_B_FULL_LOAD &&
            BATCH_ITER_MODEL == MAT_MUL_FOR_FUSED_BATCH) {
            GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3BasicTilingData, tilingData, tilingGM);
            MatmulV3Advanced::MatMulActKernel<
                DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor, B_FULL_LOAD_MODE,
                OP_TYPE_RELU>(
                x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData.matMulTilingData, tilingData.batchDimAll);
        } else if constexpr ( // basic, iterbatch, from bmmv3
            API_LEVEL == MAT_MUL_BASIC_LEVEL && FULL_LOAD == MAT_MUL_NO_FULL_LOAD &&
            BATCH_ITER_MODEL == MAT_MUL_ITER_BATCH_SINGLE_BIAS) {
            GET_TILING_DATA_WITH_STRUCT(BatchMatMulV3IterBatchBasicTilingData, tilingData, tilingGM);
            BatchMatMulActIterBatchKernel<
                DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor, MatMulL0C2Out::ON_THE_FLY,
                F_OPTYPE_RELU>(x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData);
        } else if constexpr ( // basic, aswt, from mmv3
            API_LEVEL == MAT_MUL_BASIC_LEVEL && FULL_LOAD == MAT_MUL_NO_FULL_LOAD &&
            BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
            GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
            MatmulV3Advanced::MatMulActKernel<
                DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor, MAT_MUL_NO_FULL_LOAD,
                OP_TYPE_RELU>(x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData);
        } else if constexpr ( // basic, aswt al1 fullload, from mmv3
            API_LEVEL == MAT_MUL_BASIC_LEVEL && FULL_LOAD == MAT_MUL_A_FULL_LOAD &&
            BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
            GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
            MatmulV3Advanced::MatMulActKernel<
                DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor, A_FULL_LOAD_MODE,
                OP_TYPE_RELU>(x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData);
        } else if constexpr ( // basic, aswt bl1 fullload, from mmv3
            API_LEVEL == MAT_MUL_BASIC_LEVEL && FULL_LOAD == MAT_MUL_B_FULL_LOAD &&
            BATCH_ITER_MODEL == MAT_MUL_FOR_BATCH) {
            GET_TILING_DATA_WITH_STRUCT(MatMulV3BasicTilingData, tilingData, tilingGM);
            MatmulV3Advanced::MatMulActKernel<
                DTYPE_X1, DTYPE_X2, DTYPE_Y, DTYPE_BIAS, aLayout, bLayout, layout::RowMajor, B_FULL_LOAD_MODE,
                OP_TYPE_RELU>(x1GM, x2GM, biasGM, yGM, workspaceGM, tilingData);
        }
    }
#endif
}

#endif