/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file transpose_batch_mat_mul.cpp
 * \brief
 */
#include "transpose_batch_mat_mul.h"
#include "pp_matmul_ein_sum_kernel.h"
using namespace TransposeBatchMatmulSpace;

using namespace AscendC;
using namespace matmul;
#ifndef DTYPE_BIAS
#define DTYPE_BIAS half
#endif

#ifndef FORMAT_FRACTAL_NZ
#define FORMAT_FRACTAL_NZ
#endif

constexpr CubeFormat format_x1 = CubeFormat::ND;
constexpr CubeFormat format_x2 = CubeFormat::ND;
constexpr CubeFormat format_y = CubeFormat::ND;


#define BMMV3_IMPL_CLASS_COMMON(templateClass, Mode, ...)                                                             \
    do {                                                                                                              \
        using cType = MatmulType<AscendC::TPosition::GM, format_y, DTYPE_Y>;                                          \
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_BIAS>;                              \
        TPipe pipe;                                                                                                   \
                                                                                                                      \
        using aType = MatmulType<AscendC::TPosition::GM, format_x1, DTYPE_X1, false>;                                 \
        using bType = MatmulType<AscendC::TPosition::GM, format_x2, DTYPE_X2, false>;                                 \
        templateClass<aType, bType, cType, biasType, Mode, __VA_ARGS__> op;                                           \
        op.Init(aGM, bGM, cGM, biasGM, scalesGM, user, &tilingData, &pipe);                                           \
        op.Process();                                                                                                 \
                                                                                                                      \
    } while (0)

#define PPMATMUL_EINSUM_CLASS(templateClass, transA, transB)                                                            \
    do {                                                                                                                \
        TPipe pipe;                                                                                                     \
        PpMatMulNS::SetPadding<uint64_t>((uint64_t)0);                                                                  \
        PpMatMulNS::SetNdpara(1, 0, 0);                                                                                 \
        PpMatMulNS::SetAtomicnone();                                                                                    \
        if (tilingData.swizzlDirect == 0) {                                                                             \
            templateClass<0, transA, transB, DTYPE_X1, DTYPE_Y, PpMatMulNS::DataFormat::ND> op;                         \
            op.Init(aGM, bGM, cGM, &tilingData, &pipe);                                                                 \
            op.Process();                                                                                               \
        } else if (tilingData.swizzlDirect == 1) {                                                                      \
            templateClass<1, transA, transB, DTYPE_X1, DTYPE_Y, PpMatMulNS::DataFormat::ND> op;                         \
            op.Init(aGM, bGM, cGM, &tilingData, &pipe);                                                                 \
            op.Process();                                                                                               \
        }                                                                                                               \
        pipe.Destroy();                                                                                                 \
    } while (0)

extern "C" __global__ __aicore__ void transpose_batch_mat_mul(
    GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM, GM_ADDR scalesGM, GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (TILING_KEY_IS(10000000000000000001UL)) {
        GET_TILING_DATA(tilingData, tilingGM);
        BMMV3_IMPL_CLASS_COMMON(TransposeBatchMatMulKernel, static_cast<int>(TBMM_MODE::TRANS_BMM_TRANS), TransposeBatchMatMulBlock,
                                MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000000000000000000UL)) {
        GET_TILING_DATA(tilingData, tilingGM);
        BMMV3_IMPL_CLASS_COMMON(TransposeBatchMatMulKernel, static_cast<int>(TBMM_MODE::BMM_TRANS), TransposeBatchMatMulBlock,
                                MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000000000000000011UL)) {
        GET_TILING_DATA(tilingData, tilingGM);
        BMMV3_IMPL_CLASS_COMMON(TransposeBatchMatMulKernel, static_cast<int>(TBMM_MODE::TRANS_BMM_TRANS_TRANS), TransposeBatchMatMulBlock,
                                MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(10000000000000000010UL)) {
        GET_TILING_DATA(tilingData, tilingGM);
        BMMV3_IMPL_CLASS_COMMON(TransposeBatchMatMulKernel, static_cast<int>(TBMM_MODE::BMM_TRANS_TRANS), TransposeBatchMatMulBlock,
                                MM_CFG_NO_PRELOAD);
    } else if (TILING_KEY_IS(100)) { // [soc_version, transA, transB] -> [910b/c, false, false]
        GET_TILING_DATA_WITH_STRUCT(PpMatmulTilingData, tilingData, tilingGM);
        PPMATMUL_EINSUM_CLASS(PpMatMulNS::PpMatmulEinSum, false, false);
    } else if (TILING_KEY_IS(101)) { // [soc_version, transA, transB] -> [910b/c, false, true]
        GET_TILING_DATA_WITH_STRUCT(PpMatmulTilingData, tilingData, tilingGM);
        PPMATMUL_EINSUM_CLASS(PpMatMulNS::PpMatmulEinSum, false, true);
    }
}